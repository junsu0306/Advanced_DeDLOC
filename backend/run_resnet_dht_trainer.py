#!/usr/bin/env python3
from metrics_utils import make_validators
#2025_06_09_ver
import argparse
import ast
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import wandb
import hivemind

from compress.compressor_wrapper import CompressorWrapper
from optim.sgd_optimizer import SGDOptimizer
from metrics_utils import LocalMetrics
from hivemind import get_dht_time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compression", type=str, default="topk", choices=["none", "topk"])
    parser.add_argument("--k", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--project", type=str, default="resnet-dht-stellatrain")
    parser.add_argument("--name", type=str, default="resnet18-topk-sgd-peer")
    parser.add_argument("--experiment_prefix", type=str, default="resnet_dht")
    parser.add_argument("--initial_peers", type=str, default="[]")
    parser.add_argument("--dht_listen_on", type=str, default="127.0.0.1:*")
    parser.add_argument("--endpoint", type=str, default="127.0.0.1:*")
    parser.add_argument("--client_mode", action="store_true")
    parser.add_argument("--bandwidth", type=int, default=100_000_000)
    parser.add_argument("--target_batch_size", type=int, default=512)
    parser.add_argument("--batch_size_lead", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    wandb.init(project=args.project, name=args.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = models.resnet18(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGDOptimizer(lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    compressor = CompressorWrapper(method=args.compression, ratio=args.k) if args.compression != "none" else None
    initial_peers = ast.literal_eval(args.initial_peers)
    validators, local_public_key = make_validators(args.experiment_prefix)

    dht = hivemind.DHT(
        start=True,
        initial_peers=initial_peers,
        listen_on=args.dht_listen_on,
        endpoint=args.endpoint,
        listen=not args.client_mode,
        record_validators=validators,
    )

    collab_optimizer = hivemind.CollaborativeOptimizer(
        opt=optimizer,
        dht=dht,
        prefix=args.experiment_prefix,
        compression_type=hivemind.utils.CompressionType.NONE,
        target_batch_size=args.target_batch_size - args.batch_size_lead,
        batch_size_per_step=args.batch_size,
        client_mode=args.client_mode,
        verbose=True,
        start=True,
        target_group_size=2,
        throughput=args.bandwidth,
        averaging_timeout=60.0,
        averaging_expiration=90.0,
        min_refresh_period=5.0,
        max_refresh_period=15.0,
    )

    collab_optimizer.load_state_from_peers()
    time.sleep(5)

    print("[DEBUG] Checking initial collaboration state...")
    initial_state = collab_optimizer.fetch_collaboration_state()
    if initial_state is None or initial_state.samples_accumulated == 0:
        print("[WARNING] No collaboration samples accumulated yet. Check initial_peers connectivity.")

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad(model)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.data
                    try:
                        idx, val = compressor.compress(name, grad) if compressor else (
                            torch.arange(grad.numel(), device=grad.device), grad.view(-1)
                        )
                    except Exception as e:
                        print(f"[Compression Error] {name}: {e}")
                        idx = torch.arange(grad.numel(), device=grad.device)
                        val = grad.view(-1)
                    optimizer.optimize(param, name, val, idx)

            if batch_idx >= 5:
                # skip if group size or samples not met
                if collab_optimizer.collaboration_state.samples_accumulated <= 0:
                    print("[SKIP] Not enough samples yet, skipping step() to avoid division by zero.")
                    continue

                metrics = LocalMetrics(
                    step=collab_optimizer.local_step,
                    samples_per_second=collab_optimizer.performance_ema.samples_per_second,
                    samples_accumulated=collab_optimizer.local_samples_accumulated,
                    loss=loss.item(),
                    mini_steps=1,
                )
                dht.store(
                    key=args.experiment_prefix + "_metrics",
                    subkey=local_public_key,
                    value=metrics.dict(),
                    expiration_time=get_dht_time() + 60
                )

                collab_optimizer.step()
            else:
                print(f"[WARMUP] Skipping step at batch {batch_idx}")

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)

        acc = correct / total * 100
        wandb.log({"epoch": epoch + 1, "loss": total_loss, "accuracy": acc})
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.3f}, Acc: {acc:.2f}%")

    wandb.finish()
    if hasattr(collab_optimizer, "dht") and collab_optimizer.dht.is_alive():
        print("[Cleanup] Shutting down DHT node cleanly...")
        collab_optimizer.dht.shutdown()


if __name__ == "__main__":
    main()
