import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any
import time
import wandb
import torch
import transformers
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    BertTokenizerFast,  #  ALBERT → BERT-tiny 변경
    BertConfig,
    BertForPreTraining,
)
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import is_main_process
from torch_optimizer import Lamb

import hivemind
from arguments import CollaborationArguments, DatasetArguments, BertTrainingArguments
import metrics_utils

logger = logging.getLogger(__name__)
LRSchedulerBase = getattr(torch.optim.lr_scheduler, "_LRScheduler", None)


def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)


def get_model(training_args, config, tokenizer):
    # Find latest checkpoint in output_dir
    output_dir = Path(training_args.output_dir)
    logger.info(f'Checkpoint dir {output_dir}, contents {list(output_dir.glob("checkpoint*"))}')
    latest_checkpoint_dir = max(output_dir.glob("checkpoint*"), default=None, key=os.path.getctime)

    if latest_checkpoint_dir is not None:
        logger.info(f"Loading model from {latest_checkpoint_dir}")
        model = BertForPreTraining.from_pretrained(latest_checkpoint_dir)
    else:
        logger.info(f"Training from scratch")
        model = BertForPreTraining(config)
        model.resize_token_embeddings(len(tokenizer))

    return model


def get_optimizer_and_scheduler(training_args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    opt = Lamb(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        clamp_value=training_args.clamp_value,
        debias=True,
    )

    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.max_steps
    )

    return opt, scheduler


class CollaborativeCallback(transformers.TrainerCallback):
    def __init__(
        self,
        dht: hivemind.DHT,
        optimizer: hivemind.CollaborativeOptimizer,
        model: torch.nn.Module,
        local_public_key: bytes,
        statistics_expiration: float,
    ):
        super().__init__()
        self.model = model
        self.dht, self.collaborative_optimizer = dht, optimizer
        self.local_public_key = local_public_key
        self.statistics_expiration = statistics_expiration
        self.last_reported_collaboration_step = -1
        self.previous_state = self.get_current_state()
        self.samples = 0
        self.steps = 0
        self.loss = 0
        self.total_samples_processed = 0
        self.avg_latencies = []

    def on_train_begin(
        self, args: TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs
    ):
        logger.warning("Loading state from peers")
        self.collaborative_optimizer.load_state_from_peers()

    def on_step_end(
        self, args: TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs
    ):
        # Averaging latency 측정
        avg_start_time = time.time()
        control.should_log = True
        if not self.params_are_finite():
            self.load_from_state(self.previous_state)
            return control
        self.previous_state = self.get_current_state()

        # Averaging latency 기록 및 로깅
        avg_latency = time.time() - avg_start_time
        self.avg_latencies.append(avg_latency)
        wandb.log({
            "averaging_latency": avg_latency,
            "step": self.collaborative_optimizer.local_step
        })
        if len(self.avg_latencies) > 100:
            self.avg_latencies = self.avg_latencies[-100:]
        wandb.log({
            "averaging_latency_ema": sum(self.avg_latencies) / len(self.avg_latencies),
            "step": self.collaborative_optimizer.local_step
        })

        if state.log_history:
            self.loss += state.log_history[-1]["loss"]
            self.steps += 1
            if self.collaborative_optimizer.local_step != self.last_reported_collaboration_step:
                self.last_reported_collaboration_step = self.collaborative_optimizer.local_step
                self.total_samples_processed += self.samples
                samples_per_second = self.collaborative_optimizer.performance_ema.samples_per_second
                statistics = metrics_utils.LocalMetrics(
                    step=self.collaborative_optimizer.local_step,
                    samples_per_second=samples_per_second,
                    samples_accumulated=self.samples,
                    loss=self.loss,
                    mini_steps=self.steps,
                )
                logger.info(f"Step {self.collaborative_optimizer.local_step}")
                logger.info(f"Your current contribution: {self.total_samples_processed} samples")
                if self.steps:
                    logger.info(f"Loss of your model: {self.loss / self.steps}")

                self.loss = 0
                self.steps = 0
                self.dht.store(
                    key=self.collaborative_optimizer.prefix + "_metrics",
                    subkey=self.local_public_key,
                    value=statistics.dict(),
                    expiration_time=hivemind.get_dht_time() + self.statistics_expiration,
                    return_future=True,
                )

        self.samples = self.collaborative_optimizer.local_samples_accumulated

        return control

    @torch.no_grad()
    def get_current_state(self) -> Dict[str, Any]:
        return {"model": self.model.state_dict(), "opt": self.collaborative_optimizer.opt.state_dict()}

    @torch.no_grad()
    def load_from_state(self, state):
        self.model.load_state_dict(state["model"])
        self.collaborative_optimizer.opt.load_state_dict(state["opt"])

    @torch.no_grad()
    def params_are_finite(self):
        for param in self.model.parameters():
            if not torch.all(torch.isfinite(param)):
                return False
        return True


class NoOpScheduler(LRSchedulerBase):
    """Dummy scheduler for transformers.Trainer. The real scheduler is defined in CollaborativeOptimizer.scheduler"""

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def print_lr(self, *args, **kwargs):
        if self.optimizer.scheduler:
            return self.optimizer.scheduler.print_lr(*args, **kwargs)

    def step(self):
        logger.debug("Called NoOpScheduler.step")
        self._last_lr = self.get_lr()

    def state_dict(self):
        return {}

    def load_state_dict(self, *args, **kwargs):
        logger.debug("Called NoOpScheduler.load_state_dict")


def main():
    parser = HfArgumentParser((BertTrainingArguments, DatasetArguments, CollaborationArguments))
    training_args, dataset_args, collaboration_args = parser.parse_args_into_dataclasses()
    wandb.init()

    # collaboration_args_dict 생성 및 use_pairwise 추출
    collaboration_args_dict = asdict(collaboration_args)
    collaboration_args_dict.pop("wandb_project", None)
    use_pairwise = collaboration_args_dict.pop("use_pairwise", True)  # 기본값을 True로 설정

    logger.info(f"Found {len(collaboration_args.initial_peers)} initial peers: {collaboration_args.initial_peers}")
    if not collaboration_args.initial_peers:
        raise ValueError("Please specify at least one network endpoint in initial peers.")

    setup_logging(training_args)
    set_seed(training_args.seed)

    # 모델·토크나이저 준비
    config = BertConfig.from_pretrained(dataset_args.config_path, cache_dir=dataset_args.cache_dir)
    tokenizer = BertTokenizerFast.from_pretrained(dataset_args.tokenizer_path, cache_dir=dataset_args.cache_dir)
    model = get_model(training_args, config, tokenizer).to(training_args.device)

    # 옵티마이저·스케줄러
    opt, scheduler = get_optimizer_and_scheduler(training_args, model)

    # DHT 초기화
    validators, local_public_key = metrics_utils.make_validators(collaboration_args_dict["experiment_prefix"])
    dht = hivemind.DHT(
        start=True,
        initial_peers=collaboration_args_dict.pop("initial_peers"),
        listen=not collaboration_args_dict["client_mode"],
        listen_on=collaboration_args_dict.pop("dht_listen_on"),
        endpoint=collaboration_args_dict.pop("endpoint"),
        record_validators=validators,
    )

    # 파라미터 계산
    statistics_expiration = collaboration_args_dict.pop("statistics_expiration")
    total_batch_per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    adjusted_target = collaboration_args_dict.pop("target_batch_size") - collaboration_args_dict.pop("batch_size_lead")

    # ─────────── CollaborativeOptimizer 인스턴스화 ───────────
    collaborative_optimizer = hivemind.CollaborativeOptimizer(
        opt=opt,
        dht=dht,
        scheduler=scheduler,
        prefix=collaboration_args_dict.pop("experiment_prefix"),
        compression_type=hivemind.utils.CompressionType.Value(collaboration_args_dict.pop("compression")),
        batch_size_per_step=total_batch_per_step,
        throughput=collaboration_args_dict.pop("bandwidth"),
        target_batch_size=adjusted_target,
        client_mode=collaboration_args_dict.pop("client_mode"),
        verbose=True,
        start=True,
        auxiliary=True,
        allow_state_sharing=False,
        use_pairwise=use_pairwise,  # 명시적으로 전달
        **collaboration_args_dict,
    )

    while True:
        time.sleep(0.5)
        collaborative_optimizer.step_aux()


if __name__ == "__main__":
    main()
