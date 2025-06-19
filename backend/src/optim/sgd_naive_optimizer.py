import torch
import threading

class SGDNaive(torch.nn.Module):
    def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0, dampening=0.0,
                 nesterov=False, maximize=False, device="cuda"):
        super(SGDNaive, self).__init__()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self.maximize = maximize
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.momentum_buffer = {}
        self.last_update = {}
        self.lock = threading.Lock()

    @torch.jit.ignore
    def acquire_lock(self):
        self.lock.acquire()

    @torch.jit.ignore
    def release_lock(self):
        self.lock.release()

    def optimize(self, param: torch.Tensor, name: str, grad_values: torch.Tensor, grad_indices: torch.Tensor):
        param = param.to(self.device)
        grad_values = grad_values.to(self.device)
        grad_indices = grad_indices.to(self.device).long()

        dense_grad = torch.zeros_like(param)
        dense_grad[grad_indices] = grad_values

        first = False
        tensor_len = param.numel()

        self.acquire_lock()
        if self.momentum != 0 and name not in self.momentum_buffer:
            self.momentum_buffer[name] = torch.zeros(tensor_len, device=self.device)
            self.last_update[name] = torch.zeros(tensor_len, dtype=torch.int32, device=self.device)
            first = True
        self.release_lock()

        lr = -self.lr if self.maximize else self.lr

        if self.weight_decay != 0:
            dense_grad += self.weight_decay * param

        if self.momentum:
            if not first:
                optim_b = self.momentum_buffer[name] * self.momentum + (1 - self.dampening) * dense_grad
            else:
                optim_b = dense_grad
        else:
            optim_b = dense_grad

        if self.nesterov:
            dense_grad += self.momentum * optim_b
        else:
            dense_grad = optim_b

        self.momentum_buffer[name] = optim_b
        param -= lr * dense_grad

    def configure(self, option_name, option_value):
        if option_name == "lr":
            self.lr = option_value
        elif option_name == "momentum":
            self.momentum = option_value
        elif option_name == "weight_decay":
            self.weight_decay = option_value
        elif option_name == "dampening":
            self.dampening = option_value
        elif option_name == "nesterov":
            self.nesterov = option_value
        elif option_name == "maximize":
            self.maximize = option_value
        else:
            raise ValueError(f"Unknown option: {option_name}")

SGDNaive = torch.compile(SGDNaive)

