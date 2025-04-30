#!/usr/bin/env python

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

import random
import torch
import transformers
import wandb
import hivemind
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

from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast
from arguments import CollaborationArguments, DatasetArguments, BertTrainingArguments
import metrics_utils
from partial_stale_optimzer import PartialStaleCollaborativeOptimizer

logger = logging.getLogger(__name__)
LRSchedulerBase = getattr(torch.optim.lr_scheduler, "_LRScheduler", None)


def setup_logging(training_args):
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


def get_model(training_args, config, tokenizer):
    output_dir = Path(training_args.output_dir)
    logger.info(f"Checkpoint dir {output_dir}, contents {list(output_dir.glob('checkpoint*'))}")
    latest = max(output_dir.glob("checkpoint*"), default=None, key=os.path.getctime)
    if latest:
        logger.info(f"Loading model from {latest}")
        model = BertForMaskedLM.from_pretrained(latest)
    else:
        logger.info("Training from scratch")
        model = BertForMaskedLM(config)
        model.resize_token_embeddings(len(tokenizer))
    return model


def get_optimizer_and_scheduler(training_args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": training_args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = Lamb(
        grouped,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        clamp_value=10000.0,
        debias=True,
    )
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.max_steps
    )
    return opt, scheduler



def main():
    parser = HfArgumentParser((BertTrainingArguments, DatasetArguments, CollaborationArguments))
    training_args, dataset_args, collaboration_args = parser.parse_args_into_dataclasses()

    # fp16 설정
    training_args.fp16 = True
    training_args.fp16_full_eval = True

    # 1 collaboration_args_dict 생성 및 불필요한 키 제거 (한 번만!)
    collaboration_args_dict = asdict(collaboration_args)
    for key in ("wandb_project", "use_pairwise"):
        collaboration_args_dict.pop(key, None)

    # 2 local_public_key 생성 및 wandb 초기화
    validators, local_public_key = metrics_utils.make_validators(collaboration_args_dict["experiment_prefix"])
    project = collaboration_args.wandb_project or "default-peer-project"
    run_name = f"peer-{local_public_key[:6].hex()}"
    wandb.init(project=project, name=run_name, group="bert-exp-001", reinit=True)

    # 3 evaluation 설정
    training_args.evaluation_strategy = "steps" if training_args.enable_eval else "no"
    training_args.eval_steps = 500
    training_args.do_eval = training_args.enable_eval
    training_args.report_to = ["wandb"]

    logger.info(f"Found {len(collaboration_args.initial_peers)} initial peers: {collaboration_args.initial_peers}")
    if not collaboration_args.initial_peers:
        raise ValueError("Specify at least one initial peer endpoint.")

    setup_logging(training_args)
    set_seed(training_args.seed)

    # 4 모델·토크나이저 불러오기
    config = BertConfig.from_pretrained(dataset_args.config_path, cache_dir=dataset_args.cache_dir)
    tokenizer = BertTokenizerFast.from_pretrained(dataset_args.tokenizer_path, cache_dir=dataset_args.cache_dir)
    model = get_model(training_args, config, tokenizer).to(training_args.device)

    # 5 데이터셋·콜레이터 준비
    datasets = load_from_disk(dataset_args.dataset_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    # 6 옵티마이저·스케줄러
    opt, scheduler = get_optimizer_and_scheduler(training_args, model)

    # 7 DHT 초기화
    dht = hivemind.DHT(
        start=True,
        initial_peers=collaboration_args_dict.pop("initial_peers"),
        listen=not collaboration_args_dict["client_mode"],
        listen_on=collaboration_args_dict.pop("dht_listen_on"),
        endpoint=collaboration_args_dict.pop("endpoint"),
        record_validators=validators,
    )

    # 8 배치 사이즈 계산
    total_batch_per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    statistics_expiration = collaboration_args_dict.pop("statistics_expiration")
    adjusted_target = collaboration_args_dict.pop("target_batch_size") - collaboration_args_dict.pop("batch_size_lead")

    # 9 CollaborativeOptimizer / PartialStale 분기
    if training_args.partial_stale:
        logger.info("Using PartialStaleCollaborativeOptimizer (1-step delay).")
        collaborative_optimizer = PartialStaleCollaborativeOptimizer(
            partial_stale=True,
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
            **collaboration_args_dict,
        )
    else:
        logger.info("Using normal hivemind.CollaborativeOptimizer.")
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
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
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
