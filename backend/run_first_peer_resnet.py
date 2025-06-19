#!/usr/bin/env python3

import argparse
import time
import ast
import subprocess
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/src")))

import torch
import wandb
import hivemind
from torchvision import models
from hivemind.utils.logging import get_logger

from optim.sgd_optimizer import SGDOptimizer
import metrics_utils

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_prefix", type=str, default="resnet_dht")
    parser.add_argument("--address", type=str, required=True, help="Your IP address (e.g., 127.0.0.1)")
    parser.add_argument("--dht_listen_on", type=str, default="0.0.0.0:31337")
    parser.add_argument("--initial_peers", type=str, default="[]")
    parser.add_argument("--wandb_project", type=str, default="resnet-dht-stellatrain")
    parser.add_argument("--upload_interval", type=float, default=60.0)
    parser.add_argument("--save_checkpoint_step_interval", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    # 안전한 파싱
    try:
        initial_peers = ast.literal_eval(args.initial_peers)
    except Exception as e:
        logger.error(f"Failed to parse initial_peers: {e}")
        initial_peers = []

    validators, local_public_key = metrics_utils.make_validators(args.experiment_prefix)

    dht = hivemind.DHT(
        start=True,
        listen_on=args.dht_listen_on,
        endpoint=f"{args.address}:*",
        initial_peers=initial_peers,
        record_validators=validators,
    )

    logger.info(f"Running DHT root at {args.address}:{dht.port}")

    # ⛔️ 실제 학습은 하지 않음 → opt/dummy model만 생성
    model = models.resnet18(num_classes=100)
    optimizer = SGDOptimizer(lr=0.01, momentum=0.9, weight_decay=0.0)

    collab_optimizer = hivemind.CollaborativeOptimizer(
        opt=optimizer,
        dht=dht,
        prefix=args.experiment_prefix,
        compression_type=hivemind.utils.CompressionType.NONE,
        target_batch_size=512,
        batch_size_per_step=0,  # ⛔️ 모니터링 노드는 학습 안 함
        client_mode=True,
        verbose=True,
        start=True,
        throughput=100_000_000,
        target_group_size=2,
    )

    if args.wandb_project:
        wandb.init(project=args.wandb_project, name="resnet18-monitor")

    current_step = 0
    last_upload_time = time.time()

    try:
        while True:
            metrics_entry = dht.get(args.experiment_prefix + "_metrics", latest=True)
            if metrics_entry is not None and isinstance(metrics_entry.value, dict):
                metrics_dict = metrics_entry.value
                metrics = [
                    metrics_utils.LocalMetrics.parse_obj(m["value"])
                    for m in metrics_dict.values()
                    if m and isinstance(m, dict) and "value" in m
                ]

                if metrics:
                    latest_step = max(item.step for item in metrics)
                    if latest_step != current_step:
                        total_loss = sum(m.loss for m in metrics)
                        total_mini_steps = sum(m.mini_steps for m in metrics)
                        current_loss = total_loss / total_mini_steps if total_mini_steps > 0 else 0.0

                        total_perf = sum(m.samples_per_second for m in metrics)
                        total_samples = sum(m.samples_accumulated for m in metrics)

                        logger.info(f"[Step {latest_step}] Metrics from {len(metrics)} peers")
                        logger.info(f"Loss: {current_loss:.4f}, Samples: {total_samples}, Perf: {total_perf:.2f}")
                        current_step = latest_step

                        if args.wandb_project:
                            wandb.log({
                                "step": latest_step,
                                "loss": current_loss,
                                "samples": total_samples,
                                "performance": total_perf,
                                "alive peers": len(metrics),
                            })

                        if time.time() - last_upload_time >= args.upload_interval:
                            last_upload_time = time.time()

            time.sleep(5)

    except KeyboardInterrupt:
        logger.info("[Shutdown] Gracefully shutting down...")
        if hasattr(collab_optimizer, "shutdown"):
            collab_optimizer.shutdown()
        if hasattr(dht, "shutdown"):
            dht.shutdown()
        if args.wandb_project:
            wandb.finish()


if __name__ == "__main__":
    main()
