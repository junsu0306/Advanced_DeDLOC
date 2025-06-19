import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import wandb

from compress.compressor_wrapper import CompressorWrapper
from optim.sgd_optimizer import SGDOptimizer

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compression", type=str, default="topk", choices=["none", "topk"])
    parser.add_argument("--k", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--project", type=str, default="resnet-cifar100-stellatrain")
    parser.add_argument("--name", type=str, default="resnet18-topk-sgd")
    return parser.parse_args()

def train():
    args = parse_args()
    wandb.init(project=args.project, name=args.name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    model = models.resnet18(num_classes=100).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGDOptimizer(lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    compressor = None
    if args.compression == "topk":
        compressor = CompressorWrapper(k=args.k)

    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad(model)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.data
                    if compressor:
                        idx, val = compressor.compress(grad)
                    else:
                        idx = torch.arange(grad.numel(), device=grad.device)
                        val = grad.view(-1)
                    optimizer.optimize(param, name, val, idx)

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)

        acc = correct / total * 100
        wandb.log({"epoch": epoch + 1, "loss": total_loss, "accuracy": acc})
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.3f}, Acc: {acc:.2f}%")

    wandb.finish()

if __name__ == "__main__":
    train()

