#!/usr/bin/env python

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict
import random

import torch
import transformers
import wandb
import hivemind
from hivemind import CollaborativeOptimizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BertConfig,
    BertTokenizerFast,
    BertForMaskedLM,
)
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import is_main_process
from transformers.trainer import Trainer
from torch_optimizer import Lamb

from arguments import CollaborationArguments, DatasetArguments, BertTrainingArguments
import metrics_utils

logger = logging.getLogger(__name__)
LRSchedulerBase = getattr(torch.optim.lr_scheduler, "_LRScheduler", None)

# ─── Dummy LR scheduler to satisfy transformers.Trainer API ─────────────────
class NoOpScheduler(LRSchedulerBase):
    def __init__(self, optimizer):
        super().__init__(optimizer)
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    def get_lr(self):
        return self._last_lr

    def step(self, *args, **kwargs):
        return

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class CollaborativeCallback(transformers.TrainerCallback):
    """
    Integrates hivemind CollaborativeOptimizer into the training loop,
    reporting metrics to DHT and wandb, and handling resynchronization.
    """
    def __init__(
        self,
        dht: hivemind.DHT,
        optimizer: hivemind.CollaborativeOptimizer,
        model: torch.nn.Module,
        local_public_key: bytes,
        statistics_expiration: float,
        trainer=None,
        enable_eval=True,
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
        self.eval_every = 500
        self.enable_eval = enable_eval

    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        logger.warning("Loading state from peers")
        self.collaborative_optimizer.load_state_from_peers()

    def on_step_end(self, args: TrainingArguments, state, control, **kwargs):
        control.should_log = True
        try:
            self.collaborative_optimizer.step()
        except Exception as e:
            logger.error(f"CollaborativeOptimizer.step() failed: {e}")

        if not self.params_are_finite():
            self.load_from_state(self.previous_state)
            return control
        self.previous_state = self.get_current_state()

        if state.log_history:
            last_log = state.log_history[-1]
            if "loss" in last_log:
                self.loss += last_log["loss"]
                self.steps += 1

        local_step = self.collaborative_optimizer.local_step
        if local_step != self.last_reported_collaboration_step:
            self.last_reported_collaboration_step = local_step
            self.total_samples_processed += self.samples
            samples_per_second = self.collaborative_optimizer.performance_ema.samples_per_second
            statistics = metrics_utils.LocalMetrics(
                step=local_step,
                samples_per_second=samples_per_second,
                samples_accumulated=self.samples,
                loss=float(self.loss) if self.steps > 0 else 0.0,
                mini_steps=self.steps,
            )
            logger.info(f"Step {local_step}")
            logger.info(f"Your current contribution: {self.total_samples_processed} samples")
            if self.steps:
                logger.info(f"Loss of your model: {self.loss/self.steps}")
            self.loss = 0.0
            self.steps = 0
            self.dht.store(
                key=self.collaborative_optimizer.prefix + "_metrics",
                subkey=self.local_public_key,
                value=statistics.dict(),
                expiration_time=hivemind.get_dht_time() + self.statistics_expiration,
                return_future=True,
            )
            if not self.enable_eval:
                logger.debug("🔒 Evaluation disabled (enable_eval=False)")
            elif self.trainer is not None and local_step % self.eval_every == 0:
                idx = random.randint(0, 19)
                eval_dataset = load_from_disk(f"./eval_subsets/val_split_{idx}")
                eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)
                logger.info(f"📊 Eval result (subset {idx}): {eval_result}")
                wandb.log({
                    "eval_loss": eval_result.get("eval_loss"),
                    "eval_accuracy": eval_result.get("eval_accuracy"),
                    "eval_subset": idx,
                    "step": local_step,
                })
        self.samples = self.collaborative_optimizer.local_samples_accumulated
        return control

    @torch.no_grad()
    def get_current_state(self) -> Dict[str, Any]:
        return {"model": self.model.state_dict(), "opt": self.collaborative_optimizer.opt.state_dict()}

    @torch.no_grad()
    def load_from_state(self, state: Dict[str, Any]):
        self.model.load_state_dict(state["model"])
        self.collaborative_optimizer.opt.load_state_dict(state["opt"])

    @torch.no_grad()
    def params_are_finite(self) -> bool:
        return all(torch.all(torch.isfinite(p)) for p in self.model.parameters())


def setup_logging(training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: {training_args.local_rank != -1}, "
        f"16-bits training: {training_args.fp16}"
    )
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)


def get_model(training_args: TrainingArguments, config: BertConfig, tokenizer: BertTokenizerFast) -> BertForMaskedLM:
    output_dir = Path(training_args.output_dir)
    logger.info(f'Checkpoint dir {output_dir}, contents {list(output_dir.glob("checkpoint*"))}')
    latest_checkpoint = max(output_dir.glob("checkpoint*"), default=None, key=os.path.getctime)
    if latest_checkpoint:
        logger.info(f"Loading model from {latest_checkpoint}")
        model = BertForMaskedLM.from_pretrained(latest_checkpoint)
    else:
        logger.info("Training from scratch")
        model = BertForMaskedLM(config)
        model.resize_token_embeddings(len(tokenizer))
    return model


def get_optimizer_and_scheduler(training_args: TrainingArguments, model: torch.nn.Module):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": training_args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = Lamb(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        clamp_value=10000.0,
        debias=True,
    )
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps,
    )
    return opt, scheduler


def main():
    parser = HfArgumentParser((BertTrainingArguments, DatasetArguments, CollaborationArguments))
    training_args, dataset_args, collaboration_args = parser.parse_args_into_dataclasses()

    # fp16 settings
    training_args.fp16 = True
    training_args.fp16_full_eval = True

    # prepare collaboration_args_dict and extract client_mode
    collaboration_args_dict = asdict(collaboration_args)
    # remove unwanted keys including client_mode to avoid duplicate keyword errors
    for key in ("wandb_project", "use_pairwise", "client_mode"):
        collaboration_args_dict.pop(key, None)
    # preserve client_mode flag separately
    local_client_mode = collaboration_args.client_mode

    # initialize validators and wandb
    validators, local_public_key = metrics_utils.make_validators(collaboration_args_dict["experiment_prefix"])
    project = collaboration_args.wandb_project or "default-peer-project"
    run_name = f"peer-{local_public_key[:6].hex()}"
    wandb.init(project=project, name=run_name, group="bert-exp-001", reinit=True)

    # set evaluation strategy
    training_args.evaluation_strategy = "steps" if training_args.enable_eval else "no"
    training_args.eval_steps = 500
    training_args.do_eval = training_args.enable_eval
    training_args.report_to = ["wandb"]

    if not collaboration_args.initial_peers:
        raise ValueError("Specify at least one initial peer endpoint.")

    setup_logging(training_args)
    set_seed(training_args.seed)

    # model & tokenizer
    config = BertConfig.from_pretrained(dataset_args.config_path, cache_dir=dataset_args.cache_dir)
    tokenizer = BertTokenizerFast.from_pretrained(dataset_args.tokenizer_path, cache_dir=dataset_args.cache_dir)
    model = get_model(training_args, config, tokenizer).to(training_args.device)

    # datasets & data collator
    datasets = load_from_disk(dataset_args.dataset_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    # optimizer & scheduler
    opt, scheduler = get_optimizer_and_scheduler(training_args, model)

    # DHT initialization
    dht = hivemind.DHT(
        start=True,
        initial_peers=collaboration_args.initial_peers,
        client_mode=local_client_mode,
        record_validators=validators,
    )

    total_batch_per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    statistics_expiration = collaboration_args_dict.pop("statistics_expiration")
    target_batch_size = collaboration_args_dict.pop("target_batch_size")

    # ============ Partial Stale 분기 =============
    if training_args.partial_stale:
        logger.info("Using PartialStaleCollaborativeOptimizer (1-step delay).")
        collaborative_optimizer = PartialStaleCollaborativeOptimizer(
            opt,
            dht=dht,
            prefix=collaboration_args.experiment_prefix,
            batch_size_per_step=total_batch_per_step,
            throughput=collaboration_args_dict.pop("bandwidth"),
            target_batch_size=target_batch_size,
            client_mode=local_client_mode,
            verbose=True,
            start=True,
            **collaboration_args_dict,
        )
    else:
        logger.info("Using hivemind.CollaborativeOptimizer.")
        collaborative_optimizer = CollaborativeOptimizer(
            opt,
            dht=dht,
            prefix=collaboration_args.experiment_prefix,
            batch_size_per_step=total_batch_per_step,
            throughput=collaboration_args_dict.pop("bandwidth"),
            target_batch_size=target_batch_size,
            client_mode=local_client_mode,
            verbose=True,
            start=True,
            **collaboration_args_dict,
        )

    def compute_metrics_mlm(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        if hasattr(logits, "cpu"): logits = logits.cpu().numpy()
        if hasattr(labels, "cpu"): labels = labels.cpu().numpy()
        preds = np.argmax(logits, axis=-1)
        mask = labels != -100
        correct = (preds[mask] == labels[mask]).sum()
        total = mask.sum()
        return {"accuracy": float(correct / total) if total > 0 else 0.0}

    class TrainerWithIndependentShuffling(Trainer):
        def get_train_dataloader(self) -> DataLoader:
            torch.manual_seed(hash(local_public_key))
            return super().get_train_dataloader()

    callback = CollaborativeCallback(
        dht=dht,
        optimizer=collaborative_optimizer,
        model=model,
        local_public_key=local_public_key,
        statistics_expiration=statistics_expiration,
        trainer=None,
        enable_eval=training_args.enable_eval,
    )

    trainer = TrainerWithIndependentShuffling(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=datasets["train"] if training_args.do_train else None,
        eval_dataset=datasets.get("validation") if training_args.do_eval else None,
        optimizers=(collaborative_optimizer, NoOpScheduler(collaborative_optimizer)),
        callbacks=[callback],
        compute_metrics=compute_metrics_mlm,
    )

    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)
    callback.trainer = trainer

    if training_args.do_train:
        latest_ckpt = max(Path(training_args.output_dir).glob("checkpoint*"), default=None, key=os.path.getctime)
        trainer.train(model_path=latest_ckpt)


if __name__ == "__main__":
    main()
