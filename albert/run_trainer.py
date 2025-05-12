#!/usr/bin/env python

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

import torch
import transformers
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForLanguageModeling,

)
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import is_main_process
from transformers.trainer import Trainer
from torch_optimizer import Lamb

from transformers import BertForMaskedLM

import random
import wandb
import hivemind
from arguments import CollaborationArguments, DatasetArguments, BertTrainingArguments
import metrics_utils
from transformers import BertConfig, BertTokenizerFast, BertForPreTraining

from partial_stale_optimzer import PartialStaleCollaborativeOptimizer

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
    output_dir = Path(training_args.output_dir)
    logger.info(f'Checkpoint dir {output_dir}, contents {list(output_dir.glob("checkpoint*"))}')
    latest_checkpoint_dir = max(output_dir.glob("checkpoint*"), default=None, key=os.path.getctime)

    if latest_checkpoint_dir is not None:
        logger.info(f"Loading model from {latest_checkpoint_dir}")
        model = BertForMaskedLM.from_pretrained(latest_checkpoint_dir)
    else:
        logger.info(f"Training from scratch")
        model = BertForMaskedLM(config)
        model.resize_token_embeddings(len(tokenizer))

    return model




def get_optimizer_and_scheduler(training_args, model):
    """
    기존 Lamb 옵티마이저를 설정하고 스케줄러를 반환
    """
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

    # 1. 옵티마이저 설정(Lamb)
    opt = Lamb(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        clamp_value=10000.0,
        debias=True,
    )

    # 2. 스케줄러 설정 (예: get_linear_schedule_with_warmup)
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps
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
        trainer=None,
        enable_eval=True,
        force_report_interval: int = 500,
    ):
        super().__init__()
        self.model = model
        self.dht = dht
        self.collaborative_optimizer = optimizer
        self.local_public_key = local_public_key
        self.statistics_expiration = statistics_expiration
        self.last_reported_collaboration_step = -1
        self.previous_state = self.get_current_state()
        self.samples = 0
        self.steps = 0
        self.loss = 0.0
        self.total_samples_processed = 0
        self.trainer = trainer
        self.enable_eval = enable_eval
        self.force_report_interval = force_report_interval

    def on_train_begin(self, args, state, control, **kwargs):
        logger.warning("⏳ Loading initial state from peers...")
        self.collaborative_optimizer.load_state_from_peers()

    def on_step_end(self, args, state, control, **kwargs):
        control.should_log = True

        if not self.params_are_finite():
            self.load_from_state(self.previous_state)
            return control
        self.previous_state = self.get_current_state()

        if state.log_history:
            last_log = state.log_history[-1]
            if "loss" in last_log:
                self.loss += last_log["loss"]
                self.steps += 1

        should_report = (
            self.collaborative_optimizer.local_step != self.last_reported_collaboration_step
            or self.collaborative_optimizer.local_step % self.force_report_interval == 0
        )

        if should_report:
            self.last_reported_collaboration_step = self.collaborative_optimizer.local_step
            self.total_samples_processed += self.samples

            samples_per_second = self.collaborative_optimizer.performance_ema.samples_per_second
            statistics = metrics_utils.LocalMetrics(
                step=self.collaborative_optimizer.local_step,
                samples_per_second=samples_per_second,
                samples_accumulated=self.samples,
                loss=float(self.loss or 0.0),
                mini_steps=self.steps,
            )

            logger.info(f"[DHT] Step {statistics.step} | loss = {statistics.loss:.4f} | sps = {statistics.samples_per_second:.1f}")

            try:
                self.dht.store(
                    key=self.collaborative_optimizer.prefix + "_metrics",
                    subkey=self.local_public_key,
                    value=statistics.dict(),
                    expiration_time=hivemind.get_dht_time() + self.statistics_expiration,
                    return_future=False,
                )
            except Exception as e:
                logger.warning(f"[DHT Store] Failed to report metrics: {e}")

            self.loss = 0.0
            self.steps = 0

            if self.enable_eval:
                if self.trainer is not None and self.collaborative_optimizer.local_step % 500 == 0:
                    idx = random.randint(0, 19)
                    eval_dataset = load_from_disk(f"./eval_subsets/val_split_{idx}")
                    eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)
                    logger.info(f"📊 Eval result (subset {idx}): {eval_result}")

                    wandb.log({
                        "eval_loss": eval_result.get("eval_loss"),
                        "eval_accuracy": eval_result.get("eval_accuracy"),
                        "eval_subset": idx,
                        "step": self.collaborative_optimizer.local_step,
                    })

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

    training_args.fp16 = True
    training_args.fp16_full_eval = True  # ✅ 여기 추가
    
    # ✅ collaboration_args_dict 생성
    collaboration_args_dict = asdict(collaboration_args)

    # ✅ wandb_project는 CollaborativeOptimizer와 무관하므로 제거
    collaboration_args_dict.pop("wandb_project", None)

    # ✅ local_public_key 생성
    validators, local_public_key = metrics_utils.make_validators(collaboration_args_dict["experiment_prefix"])
# ✅ 그 다음 wandb 초기화
    project_name = collaboration_args.wandb_project or "default-peer-project"
    run_name = f"peer-{local_public_key[:6].hex()}"
    wandb.init(
        project=project_name,
        name=run_name,
        group="bert-exp-001",
        reinit=True,
    )


     # ⬇️ 여기에 추가
    # evaluation 설정
    training_args.evaluation_strategy = "steps" if training_args.enable_eval else "no"
    training_args.eval_steps = 500
    training_args.do_eval = training_args.enable_eval  # 이게 핵심!

    training_args.report_to = ["wandb"]

    logger.info(f"Found {len(collaboration_args.initial_peers)} initial peers: {collaboration_args.initial_peers}")
    if len(collaboration_args.initial_peers) == 0:
        raise ValueError("Please specify at least one network endpoint in initial peers.")

    collaboration_args_dict = asdict(collaboration_args)

    collaboration_args_dict.pop("wandb_project", None)  # ✅ 여기도 다시 필요함!

    setup_logging(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = BertConfig.from_pretrained(dataset_args.config_path, cache_dir=dataset_args.cache_dir)
    tokenizer = BertTokenizerFast.from_pretrained(dataset_args.tokenizer_path, cache_dir=dataset_args.cache_dir)
    model = get_model(training_args, config, tokenizer)
    model.to(training_args.device)

    tokenized_datasets = load_from_disk(dataset_args.dataset_path)
    # This data collator will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    opt, scheduler = get_optimizer_and_scheduler(training_args, model)

    validators, local_public_key = metrics_utils.make_validators(collaboration_args_dict["experiment_prefix"])
    dht = hivemind.DHT(
        start=True,
        initial_peers=collaboration_args_dict.pop("initial_peers"),
        listen=not collaboration_args_dict["client_mode"],
        listen_on=collaboration_args_dict.pop("dht_listen_on"),
        endpoint=collaboration_args_dict.pop("endpoint"),
        record_validators=validators,
    )

    total_batch_size_per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    statistics_expiration = collaboration_args_dict.pop("statistics_expiration")
    adjusted_target_batch_size = collaboration_args_dict.pop("target_batch_size") - collaboration_args_dict.pop(
        "batch_size_lead"
    )
    
  # ============ Partial Stale 분기 =============
    from hivemind import CollaborativeOptimizer  # 원본

    if training_args.partial_stale:
        logger.info("Using PartialStaleCollaborativeOptimizer (1-step delay).")
        collaborative_optimizer = PartialStaleCollaborativeOptimizer(
            partial_stale=True,
            opt=opt,
            dht=dht,
            scheduler=scheduler,
            prefix=collaboration_args_dict.pop("experiment_prefix"),
            compression_type=hivemind.utils.CompressionType.Value(collaboration_args_dict.pop("compression")),
            batch_size_per_step=total_batch_size_per_step,
            throughput=collaboration_args_dict.pop("bandwidth"),  # ✅ 명시적으로 전달
            target_batch_size=adjusted_target_batch_size,
            client_mode=collaboration_args_dict.pop("client_mode"),
            verbose=True,
            start=True,
            **collaboration_args_dict,  # 나머지 averaging 관련 인자들
        )
    else:
        logger.info("Using normal hivemind.CollaborativeOptimizer.")
        collaborative_optimizer = hivemind.CollaborativeOptimizer(
            opt=opt,
            dht=dht,
            scheduler=scheduler,
            prefix=collaboration_args_dict.pop("experiment_prefix"),
            compression_type=hivemind.utils.CompressionType.Value(collaboration_args_dict.pop("compression")),
            batch_size_per_step=total_batch_size_per_step,
            throughput=collaboration_args_dict.pop("bandwidth"),
            target_batch_size=adjusted_target_batch_size,
            client_mode=collaboration_args_dict.pop("client_mode"),
            verbose=True,
            start=True,
            **collaboration_args_dict,
        )
    # =============================================

    def compute_metrics_mlm(eval_pred):
        import numpy as np
        logits, labels = eval_pred

    # ✅ 명확히 GPU에서 CPU로 변환
        if hasattr(logits, "cpu"):
            logits = logits.cpu().numpy()
        if hasattr(labels, "cpu"):
            labels = labels.cpu().numpy()

        predictions = np.argmax(logits, axis=-1)
        mask = labels != -100
        correct = (predictions[mask] == labels[mask]).sum()
        total = mask.sum()
        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": float(accuracy)}


    class TrainerWithIndependentShuffling(Trainer):
        def get_train_dataloader(self) -> DataLoader:
            torch.manual_seed(hash(local_public_key))
            return super().get_train_dataloader()
    
    # 먼저 callback 인스턴스를 만든다
    callback = CollaborativeCallback(
    dht, collaborative_optimizer, model, local_public_key, statistics_expiration,
    trainer=None,
    enable_eval=training_args.enable_eval  # ✅ 여기에 플래그 전달
    )

    
# Trainer 인스턴스 생성
    trainer = TrainerWithIndependentShuffling(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        optimizers=(collaborative_optimizer, NoOpScheduler(collaborative_optimizer)),
        callbacks=[callback],  # ✅ 바로 위에서 만든 callback 인스턴스 사용
        compute_metrics=compute_metrics_mlm,
)

    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)

# ✅ 이제 trainer 객체를 callback에 넣어줌
    callback.trainer = trainer

    # Training
    if training_args.do_train:
        latest_checkpoint_dir = max(Path(training_args.output_dir).glob("checkpoint*"), default=None, key=os.path.getctime)
        trainer.train(model_path=latest_checkpoint_dir)
        # ✅ 수동으로 evaluate() 호출 (정상 동작 여부 확인)
    #*print("🔍 Running manual evaluation...")
    #result = trainer.evaluate()
    #print("✅ Eval result:", result)
    #print("eval_dataset size:", len(trainer.eval_dataset))


    


if __name__ == "__main__":
    main()