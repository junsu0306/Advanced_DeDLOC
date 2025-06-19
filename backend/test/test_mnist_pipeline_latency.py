import time
import wandb
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


# ✅ SimpleNet 구조 (우선순위 기반 테스트와 동일)
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 28*28)))
        return self.fc2(x)


# ✅ 정확도 평가
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train_mnist_pipeline():
    wandb.init(project="mnist-pipeline-latency", name="mnist-priority-matched")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # ✅ 우선순위 기반 테스트와 동일한 배치 사이즈
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    target_accuracy = 0.90
    max_epochs = 20
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0
        batch_count = 0
        epoch_start = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

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
    wandb.log({"total_training_time": total_time})


if __name__ == "__main__":
    train_mnist_pipeline()

