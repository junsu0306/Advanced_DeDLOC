import time
import wandb
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from engine.fasterdp_engine_full_v6 import FasterDpEngine
from engine.modules.module_compress import ModuleCompress
from engine.modules.module_cpu_optimize import ModuleCpuOptimize
from engine.modules.module_h2d_copy import ModuleH2DCopyPre
from engine.modules.module_d2h_copy import ModuleD2HCopy, ModuleD2HCopyPost
from engine.modules.module_cpu_gather import ModuleCpuGather

wandb.init(project="mnist-pipeline-latency")

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 28*28)))
        return self.fc2(x)

class DummyEngine(FasterDpEngine):
    def __init__(self):
        super().__init__()
        self._local_rank = 0
        self._world_size_internal = 2
        self.load_modules()

    def world_size(self): return self._world_size_internal
    def local_rank(self): return self._local_rank

    def record_stat_start(self, task, name): task.__dict__[f"__time_start_{name}"] = time.time()
    def record_stat_end(self, task, name):
        end = time.time()
        start = task.__dict__.get(f"__time_start_{name}", None)
        if start:
            elapsed = end - start
            wandb.log({f"latency/{name}": elapsed})
            print(f"[latency] {name}: {elapsed:.6f} sec")
        else:
            print(f"[stat_end] {name} (no start time)")

def train(model, loader, engine, optimizer_name="sgd"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    engine.module_compress = ModuleCompress(method="topk")
    engine.module_cpu_optimize = ModuleCpuOptimize(optimizer=optimizer_name)
    engine.module_h2d_copy_pre = ModuleH2DCopyPre()
    engine.module_d2h_copy = ModuleD2HCopy(node_rank=0)
    engine.module_d2h_copy_post = ModuleD2HCopyPost()
    engine.module_cpu_gather = ModuleCpuGather()

    start_time = time.time()

    for epoch in range(1, 100):
        correct, total, epoch_loss = 0, 0, 0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            epoch_loss += loss.item()

            # dummy priority trace log
            priority = epoch * 1000 + i
            print(f"[PRIORITY] iteration={epoch}, layer={i}, priority={priority}")

        acc = correct / total
        print(f"Epoch {epoch} - Loss: {epoch_loss/len(loader):.4f}, Accuracy: {acc:.4f}")
        wandb.log({"epoch": epoch, "loss": epoch_loss/len(loader), "accuracy": acc})

        if acc >= 0.90:
            print(f"Target accuracy {acc:.4f} reached. Stopping early.")
            break

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} sec")
    wandb.log({"total_training_time": total_time})

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                              batch_size=256, shuffle=True)
    model = SimpleNet()
    engine = DummyEngine()
    train(model, train_loader, engine)

