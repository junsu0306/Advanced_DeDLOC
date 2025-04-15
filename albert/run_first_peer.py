#!/usr/bin/env python
"""
run_first_peer.py

이 스크립트는 분산 협업 학습 환경에서 코디네이터(첫 번째 피어)를 실행합니다.
기존의 run_first_peer.py 구성 요소(피어 state 동기화, 체크포인트 저장, DHT 기반 메트릭 수집 등)를 모두 유지하면서,
Adaptive Averaging과 Partial Staleness 기능을 적용한 AdaptivePartialStaleCollaborativeOptimizer를 사용합니다.

Arguments:
 - CoordinatorArguments, CollaborativeOptimizerArguments, AveragerArguments: 
      arguments.py에 정의된 인자들을 사용합니다.
 - DHT는 hivemind.dht를 통해 초기화되며, 각 피어는 self.prefix + "_metrics" 키로 메트릭을 기록합니다.
 - WandB 로깅 및 체크포인트 저장/업로드 기능도 포함합니다.
"""

from dataclasses import dataclass, field, asdict
import subprocess
import time
from typing import Optional

import torch
from torch_optimizer import Lamb
from transformers import BertForMaskedLM, BertConfig, HfArgumentParser, set_seed
import wandb
from whatsmyip.providers import GoogleDnsProvider
from whatsmyip.ip import get_ip

from arguments import BaseTrainingArguments, CollaborativeOptimizerArguments, AveragerArguments
import hivemind
from hivemind.utils.logging import get_logger
import metrics_utils

# 여기서 AdaptivePartialStaleCollaborativeOptimizer를 partial_stale_optimizer.py에서 가져옵니다.
from partial_stale_optimizer import AdaptivePartialStaleCollaborativeOptimizer

logger = get_logger(__name__)


@dataclass
class CoordinatorArguments(BaseTrainingArguments):
    """
    코디네이터(첫 번째 피어) 실행 시 필요한 인자들입니다.
    여러 초기 피어(initial_peers)를 지정하여, 한 피어가 죽더라도 다른 피어를 통해 협업을 유지합니다.
    """
    address: Optional[str] = field(
        default=None,
        metadata={
            "help": "이 머신의 네트워크 주소. 글로벌 실험 시 public IP, private run 시 내부 IP 사용"
        },
    )
    refresh_period: float = field(
        default=30, metadata={"help": "Coordinator가 DHT에서 키를 가져오는 주기(초)"}
    )
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "WandB에 기록할 프로젝트 이름"}
    )
    save_checkpoint_step_interval: int = field(
        default=5, metadata={"help": "피어 state를 저장하는 스텝 간격"}
    )
    # ALBERT → BERT-tiny 변경 (BertConfig 사용)
    model_config_path: str = field(
        default="https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/main/config.json",
        metadata={"help": "모델 config 파일 경로"}
    )
    repo_path: Optional[str] = field(
        default=None,
        metadata={"help": "모델 및 optimizer state를 업로드할 repository 경로"}
    )
    upload_interval: Optional[float] = field(
        default=300, metadata={"help": "모델 업로드 주기(초)"}
    )


class CheckpointHandler:
    def __init__(
        self,
        coordinator_args: CoordinatorArguments,
        collab_optimizer_args: CollaborativeOptimizerArguments,
        averager_args: AveragerArguments,
        dht: hivemind.DHT,
    ):
        self.save_checkpoint_step_interval = coordinator_args.save_checkpoint_step_interval
        self.repo_path = coordinator_args.repo_path
        self.upload_interval = coordinator_args.upload_interval
        self.previous_step = -1

        # 모델 생성: BertForMaskedLM 사용 (BERT-tiny로 조정)
        config = BertConfig.from_pretrained(coordinator_args.model_config_path)
        self.model = BertForMaskedLM(config)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        opt = Lamb(
            optimizer_grouped_parameters,
            lr=0.0005,  # BERT-tiny 학습률 조정
            weight_decay=0.01,
            clamp_value=10000.0,
            debias=True,
        )

        adjusted_target_batch_size = collab_optimizer_args.target_batch_size - collab_optimizer_args.batch_size_lead

        # 기존의 CollaborativeOptimizer 대신에 AdaptivePartialStaleCollaborativeOptimizer 사용
        # partial_stale와 use_adaptive_averaging 플래그는 여기서 True로 설정 (추후 arguments에서 조정 가능)
        self.collaborative_optimizer = AdaptivePartialStaleCollaborativeOptimizer(
            opt=opt,
            dht=dht,
            prefix=coordinator_args.experiment_prefix,
            compression_type=hivemind.utils.CompressionType.Value(collab_optimizer_args.compression),
            throughput=collab_optimizer_args.bandwidth,
            target_batch_size=adjusted_target_batch_size,
            client_mode=collab_optimizer_args.client_mode,
            verbose=True,
            start=True,
            # 아래 파라미터는 adaptive averaging 및 partial stale 기능 활성화를 위한 추가 인자.
            partial_stale=True,
            use_adaptive_averaging=True,
            lp_solver_timeout=averager_args.lp_solver_timeout,
            max_lp_iterations=averager_args.max_lp_iterations,
            **asdict(averager_args)
        )
        self.previous_timestamp = time.time()

    def is_time_to_save_state(self, cur_step):
        return self.save_checkpoint_step_interval is not None and (cur_step - self.previous_step >= self.save_checkpoint_step_interval)

    def save_state(self, cur_step):
        self.collaborative_optimizer.load_state_from_peers()
        self.previous_step = cur_step

    def is_time_to_upload(self):
        return self.repo_path is not None and (time.time() - self.previous_timestamp >= self.upload_interval)

    def upload_checkpoint(self, current_loss):
        self.model.save_pretrained(self.repo_path)
        torch.save(self.collaborative_optimizer.opt.state_dict(), f"{self.repo_path}/optimizer_state.pt")
        self.previous_timestamp = time.time()
        try:
            subprocess.run("git add --all", shell=True, check=True, cwd=self.repo_path)
            current_step = self.collaborative_optimizer.collaboration_state.optimizer_step
            subprocess.run(f"git commit -m 'Step {current_step}, loss {current_loss:.3f}'", shell=True, check=True, cwd=self.repo_path)
            subprocess.run("git push", shell=True, check=True, cwd=self.repo_path)
        except subprocess.CalledProcessError as e:
            logger.warning("체크포인트 업로드 중 오류: %s", e.output)


def main():
    # Argument parsing: CoordinatorArguments, CollaborativeOptimizerArguments, AveragerArguments
    parser = HfArgumentParser((CoordinatorArguments, CollaborativeOptimizerArguments, AveragerArguments))
    coordinator_args, collab_optimizer_args, averager_args = parser.parse_args_into_dataclasses()

    if coordinator_args.address is None:
        logger.warning("주소가 지정되지 않았습니다. DNS를 통해 주소 추론 중...")
        coordinator_args.address = get_ip(GoogleDnsProvider)

    experiment_prefix = coordinator_args.experiment_prefix
    validators, local_public_key = metrics_utils.make_validators(experiment_prefix)
    dht = hivemind.DHT(
        start=True,
        listen_on=coordinator_args.dht_listen_on,
        endpoint=f"{coordinator_args.address}:*",
        initial_peers=coordinator_args.initial_peers,
        record_validators=validators,
    )
    logger.info(f"DHT 루트 실행: {coordinator_args.address}:{dht.port}")

    if coordinator_args.wandb_project is not None:
        wandb.init(project=coordinator_args.wandb_project)

    current_step = 0
    checkpoint_handler = CheckpointHandler(coordinator_args, collab_optimizer_args, averager_args, dht)

    while True:
        # DHT에서 experiment_prefix + "_metrics" 키 아래의 메트릭 레코드를 가져옵니다.
        metrics_dict = dht.get(experiment_prefix + "_metrics", latest=True)
        if metrics_dict is not None:
            metrics_dict = metrics_dict.value
            metrics = [metrics_utils.LocalMetrics.parse_obj(metrics_dict[peer].value) for peer in metrics_dict]
            latest_step = max(item.step for item in metrics)
            if latest_step != current_step:
                logger.info("메트릭 수신: %d peer에서 메트릭 수집", len(metrics))
                for i, m in enumerate(metrics):
                    logger.info("피어 %d: %s", i, m)
                current_step = latest_step
                alive_peers = sum(1 for _ in metrics)
                num_batches = 0
                sum_loss = 0
                num_samples = 0
                sum_perf = 0
                sum_mini_steps = 0
                for item in metrics:
                    sum_loss += item.loss
                    sum_perf += item.samples_per_second
                    num_samples += item.samples_accumulated
                    sum_mini_steps += item.mini_steps
                current_loss = sum_loss / sum_mini_steps if sum_mini_steps > 0 else 0.0

                if coordinator_args.wandb_project is not None:
                    wandb.log({
                        "loss": current_loss,
                        "alive_peers": alive_peers,
                        "samples": num_samples,
                        "performance": sum_perf,
                        "step": latest_step,
                    })
                if checkpoint_handler.is_time_to_save_state(current_step):
                    checkpoint_handler.save_state(current_step)
                    if checkpoint_handler.is_time_to_upload():
                        checkpoint_handler.upload_checkpoint(current_loss)
                logger.info("Step #%d\tloss = %.5f", current_step, current_loss)
        else:
            logger.debug("DHT에서 메트릭 레코드를 찾지 못했습니다.")
        logger.debug("피어가 아직 살아있습니다...")
        time.sleep(coordinator_args.refresh_period)
