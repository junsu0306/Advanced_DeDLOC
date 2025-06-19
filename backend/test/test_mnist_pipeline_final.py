import time
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# MNIST 모델 정의
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


# 정확도 계산 함수
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


# 메인 파이프라인
def train_mnist_pipeline():
    wandb.init(project="mnist-pipeline-latency", name="mnist-latency-final")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

    target_accuracy = 0.90
    max_epochs = 20
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0
        batch_count = 0
        epoch_start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        epoch_end = time.time()
        latency = epoch_end - epoch_start

        accuracy = evaluate(model, test_loader, device)
        avg_loss = epoch_loss / batch_count

        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"[Latency] Epoch {epoch}: {latency:.2f} sec")

        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": accuracy,
            "epoch_latency_sec": latency
        })

        if accuracy >= target_accuracy:
            print(f"Target accuracy {accuracy:.4f} reached. Stopping early.")
            break

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} sec")


if __name__ == "__main__":
    train_mnist_pipeline()

