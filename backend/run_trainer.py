#!/usr/bin/env python

# === run_trainer.py ===

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
from transformers import BertConfig, BertTokenizerFast


from partial_stale_optimzer import PartialStaleCollaborativeOptimizer

logger = logging.getLogger(__name__)
LRSchedulerBase = getattr(torch.optim.lr_scheduler, "_LRScheduler", None)


def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
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
        clamp_value=10000.0,
        debias=True,
    )

    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps
    )
    return opt, scheduler


class CollaborativeCallback(transformers.TrainerCallback):
    def __init__(self, dht, optimizer, model, local_public_key, statistics_expiration, trainer=None, enable_eval=True):
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
        self.trainer = trainer
        self.eval_every = 500
        self.enable_eval = enable_eval

    def on_train_begin(self, args, state, control, **kwargs):
        logger.warning("Loading state from peers")
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

            if self.collaborative_optimizer.local_step != self.last_reported_collaboration_step:
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
                self.loss = 0
                self.steps = 0
                self.dht.store(
                    key=self.collaborative_optimizer.prefix + "_metrics",
                    subkey=self.local_public_key,
                    value=statistics.dict(),
                    expiration_time=hivemind.get_dht_time() + self.statistics_expiration,
                    return_future=True,
                )
                if self.enable_eval and self.trainer and self.collaborative_optimizer.local_step % self.eval_every == 0:
                    idx = random.randint(0, 19)
                    eval_dataset = load_from_disk(f"./eval_subsets/val_split_{idx}")
                    eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)
                    wandb.log({
                        "eval_loss": eval_result.get("eval_loss"),
                        "eval_accuracy": eval_result.get("eval_accuracy"),
                        "eval_subset": idx,
                        "step": self.collaborative_optimizer.local_step,
                    })

        self.samples = self.collaborative_optimizer.local_samples_accumulated
        return control

    @torch.no_grad()
    def get_current_state(self):
        return {"model": self.model.state_dict(), "opt": self.collaborative_optimizer.opt.state_dict()}

    @torch.no_grad()
    def load_from_state(self, state):
        self.model.load_state_dict(state["model"])
        self.collaborative_optimizer.opt.load_state_dict(state["opt"])

    @torch.no_grad()
    def params_are_finite(self):
        return all(torch.all(torch.isfinite(p)) for p in self.model.parameters())


class NoOpScheduler(LRSchedulerBase):
    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]
    def step(self):
        self._last_lr = self.get_lr()
    def state_dict(self):
        return {}
    def load_state_dict(self, *args, **kwargs):
        pass


def main():
    parser = HfArgumentParser((BertTrainingArguments, DatasetArguments, CollaborationArguments))
    training_args, dataset_args, collaboration_args = parser.parse_args_into_dataclasses()

    training_args.fp16 = True
    training_args.fp16_full_eval = True
    training_args.evaluation_strategy = "steps" if training_args.enable_eval else "no"
    training_args.eval_steps = 500
    training_args.do_eval = training_args.enable_eval
    training_args.report_to = ["wandb"]

    setup_logging(training_args)
    set_seed(training_args.seed)

    config = BertConfig.from_pretrained(dataset_args.config_path, cache_dir=dataset_args.cache_dir)
    tokenizer = BertTokenizerFast.from_pretrained(dataset_args.tokenizer_path, cache_dir=dataset_args.cache_dir)
    model = get_model(training_args, config, tokenizer).to(training_args.device)
    tokenized_datasets = load_from_disk(dataset_args.dataset_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    opt, scheduler = get_optimizer_and_scheduler(training_args, model)

    validators, local_public_key = metrics_utils.make_validators(collaboration_args.experiment_prefix)
    dht = hivemind.DHT(
        start=True,
        initial_peers=collaboration_args.initial_peers,
        listen=not collaboration_args.client_mode,
        listen_on=collaboration_args.dht_listen_on,
        endpoint=collaboration_args.endpoint,
        record_validators=validators,
    )

    adjusted_target_batch_size = collaboration_args.target_batch_size - collaboration_args.batch_size_lead
    batch_size_per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps

    if training_args.partial_stale:
        optimizer = PartialStaleCollaborativeOptimizer(
            partial_stale=True,
            opt=opt,
            dht=dht,
            scheduler=scheduler,
            prefix=collaboration_args.experiment_prefix,
            compression_type=hivemind.utils.CompressionType.Value(collaboration_args.compression),
            batch_size_per_step=batch_size_per_step,
            throughput=collaboration_args.bandwidth,
            target_batch_size=adjusted_target_batch_size,
            client_mode=collaboration_args.client_mode,
            verbose=True,
            start=True,
            **asdict(collaboration_args)
        )
    else:
        optimizer = hivemind.CollaborativeOptimizer(
            opt=opt,
            dht=dht,
            scheduler=scheduler,
            prefix=collaboration_args.experiment_prefix,
            compression_type=hivemind.utils.CompressionType.Value(collaboration_args.compression),
            batch_size_per_step=batch_size_per_step,
            throughput=collaboration_args.bandwidth,
            target_batch_size=adjusted_target_batch_size,
            client_mode=collaboration_args.client_mode,
            verbose=True,
            start=True,
            **asdict(collaboration_args)
        )

    def compute_metrics_mlm(eval_pred):
        import numpy as np
        logits, labels = eval_pred
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

    callback = CollaborativeCallback(dht, optimizer, model, local_public_key, collaboration_args.statistics_expiration, enable_eval=training_args.enable_eval)
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        optimizers=(optimizer, NoOpScheduler(optimizer)),
        callbacks=[callback],
        compute_metrics=compute_metrics_mlm,
    )
    callback.trainer = trainer

    if training_args.do_train:
        latest_checkpoint_dir = max(Path(training_args.output_dir).glob("checkpoint*"), default=None, key=os.path.getctime)
        trainer.train(model_path=latest_checkpoint_dir)


if __name__ == "__main__":
    main()

