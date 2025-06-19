import time
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_resnet18_cifar100():
    wandb.init(project="resnet18-cifar100-latency", name="resnet18-cifar100")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=100).to(device)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    train_dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    target_accuracy = 0.70
    max_epochs = 1000
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0
        batch_count = 0
        epoch_start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        latency = time.time() - epoch_start
        acc = evaluate(model, test_loader, device)
        avg_loss = epoch_loss / batch_count

        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        print(f"[Latency] Epoch {epoch}: {latency:.2f} sec")

        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": acc,
            "epoch_latency_sec": latency
        })

        if acc >= target_accuracy:
            print(f"Target accuracy {acc:.4f} reached. Stopping early.")
            break

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} sec")
    wandb.log({"total_training_time": total_time})


if __name__ == "__main__":
    train_resnet18_cifar100()

