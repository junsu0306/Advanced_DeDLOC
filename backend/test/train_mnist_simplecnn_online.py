import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from compress.topk_compressor import TopkCompressor
from optim.sgd_optimizer import SGDOptimizer

print("[Check] Script started")

# ✅ W&B 온라인 모드 설정
wandb.init(project="mnist-topk-sgd", name="simplecnn-topk-sgd-gpu")
print("[Check] wandb.init 완료")

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Check] Using device: {device}")

# ✅ Simple CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = SGDOptimizer(lr=0.01, momentum=0.9, weight_decay=0.0)
compressor = TopkCompressor(k=0.1, num_threads=1)

# ✅ 데이터셋: MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
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
    for epoch in range(1, 21):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        for i, (images, labels) in enumerate(train_loader):
            print(f"[Check] Epoch {epoch} - Batch {i}")

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

                idx, val = compressor.compress(grad_tensor)
                optimizer.optimize(param, name, val, idx)  # ✅ param 직접 넘김

            epoch_loss += loss.item()

        latency = time.time() - epoch_start
        accuracy = evaluate(model)
        avg_loss = epoch_loss / len(train_loader)

        print(f"Epoch {epoch} 완료 - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"[Latency] Epoch {epoch}: {latency:.2f} sec")

        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": accuracy,
            "epoch_latency_sec": latency
        })

        if accuracy >= 0.98:
            print(f"Target accuracy {accuracy:.4f} reached. Stopping early.")
            break

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} sec")
    wandb.log({"total_training_time": total_time})

print("[Check] train() 호출 직전")
train()

