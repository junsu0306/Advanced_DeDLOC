# ✅ test_resnet_cifar100_stellatrain.py
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from compress.topk_compressor import TopkCompressor
from optim.sgd_optimizer import SGDOptimizer

# W&B 초기화
wandb.init(project="resnet-cifar100-stellatrain", name="resnet18-topk-sgd")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 설정
model = models.resnet18(num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = SGDOptimizer(lr=0.01)
compressor = TopkCompressor(k_ratio=0.1)

# 데이터셋 로딩
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR100(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR100(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

# 정확도 측정 함수
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

# 학습 함수
def train():
    start_time = time.time()
    for epoch in range(1, 201):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # 파라미터별 gradient 압축 및 최적화
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                grad_tensor = param.grad.detach().view(-1)
                param_tensor = param.view(-1)
                idx, val = compressor.compress(grad_tensor)
                optimizer.optimize(param_tensor, name, val, idx)

            epoch_loss += loss.item()

        latency = time.time() - epoch_start
        accuracy = evaluate(model)
        avg_loss = epoch_loss / len(train_loader)

        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"[Latency] Epoch {epoch}: {latency:.2f} sec")
        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": accuracy,
            "epoch_latency_sec": latency
        })

        if accuracy >= 0.70:
            print(f"Target accuracy {accuracy:.4f} reached. Stopping early.")
            break

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} sec")
    wandb.log({"total_training_time": total_time})

if __name__ == "__main__":
    train()

