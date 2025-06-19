import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from compress.compressor_wrapper import CompressorWrapper
from optim.sgd_optimizer import SGDOptimizer

print("[Check] Script started")

wandb.init(project="resnet-cifar100-stellatrain", name="resnet18-thresholdv-sgd-gpu")
print("[Check] wandb.init 완료")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Check] Using device: {device}")

model = models.resnet18(weights=None, num_classes=100)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = SGDOptimizer(lr=0.1, momentum=0.9, weight_decay=5e-4)
compressor = CompressorWrapper(method="thresholdv", ratio=0.1)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
print("[Check] DataLoader 생성 완료")

def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def train():
    print("[Check] train() 시작")
    start_time = time.time()
    for epoch in range(1, 101):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            torch.cuda.synchronize() if device.type == "cuda" else None

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad(model)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                grad_tensor = param.grad.detach().view(-1)

                idx, val = compressor.compress(name, grad_tensor)
                optimizer.optimize(param, name, val, idx)

            epoch_loss += loss.item()

        latency = time.time() - epoch_start
        accuracy = evaluate(model)
        avg_loss = epoch_loss / len(train_loader)

        print(f"[Result] Epoch {epoch} 완료 | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Latency: {latency:.2f} sec")

        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": accuracy,
            "epoch_latency_sec": latency
        })

        if accuracy >= 0.65:
            print(f"[Early Stop] Target accuracy {accuracy:.4f} reached.")
            break

    total_time = time.time() - start_time
    print(f"[Result] Total training time: {total_time:.2f} sec")
    wandb.log({"total_training_time": total_time})

print("[Check] train() 호출 직전")
train()

