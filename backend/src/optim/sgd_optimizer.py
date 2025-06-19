import torch
import math

class SGDOptimizer:
    def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0,
                 dampening=0.0, nesterov=False, maximize=False, smart_momentum=False):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self.maximize = maximize
        self.smart_momentum = smart_momentum
        self.state = {}  # {name: {'momentum_buffer': ..., 'last_iter': ...}}
        self.iter = 0

    def zero_grad(self, model):
        for param in model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def optimize(self, param, name, grad_values, grad_indices):
        device = param.device
        param_tensor = param.view(-1)

        grad = torch.zeros_like(param_tensor, device=device)
        grad_flat = grad.view(-1)
        grad_flat[grad_indices.to(device)] = grad_values.to(device)

        if not isinstance(self.weight_decay, torch.Tensor) and self.weight_decay != 0.0:
            grad += self.weight_decay * param_tensor

        if name not in self.state:
            self.state[name] = {
                'momentum_buffer': torch.zeros_like(param_tensor, device=device),
                'last_iter': torch.full_like(param_tensor, -1, dtype=torch.int32)
            }

        buf = self.state[name]['momentum_buffer']
        last_iter = self.state[name]['last_iter']
        current_iter = self.iter

        if self.smart_momentum and self.momentum != 0:
            delta_iter = (current_iter - last_iter).float()
            mask = (last_iter >= 0).float()
            momentum_coeff = mask * self.momentum ** delta_iter

            buf = buf * momentum_coeff + grad * (1.0 - self.dampening)
            self.state[name]['momentum_buffer'] = buf

            grad = grad + self.momentum * buf if self.nesterov else buf

            update_mask = torch.zeros_like(last_iter)
            update_mask[grad_indices.to(device)] = 1
            last_iter += update_mask.int()
            self.state[name]['last_iter'] = last_iter

        elif self.momentum != 0:
            buf = buf.to(device)  # ✅ 디바이스 일치 보장
            buf.mul_(self.momentum).add_(grad, alpha=(1.0 - self.dampening))
            grad = grad + self.momentum * buf if self.nesterov else buf
            self.state[name]['momentum_buffer'] = buf

        step = -self.lr if not self.maximize else self.lr
        param.data = (param_tensor + step * grad).view_as(param)

        self.iter += 1

    def step(self):
        pass

    @property
    def param_groups(self):
        return []

    # Hivemind 호환을 위한 state 저장
    def state_dict(self):
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "dampening": self.dampening,
            "nesterov": self.nesterov,
            "maximize": self.maximize,
            "smart_momentum": self.smart_momentum,
            "state": self.state,
            "iter": self.iter
        }

    def load_state_dict(self, state_dict):
        self.lr = state_dict["lr"]
        self.momentum = state_dict["momentum"]
        self.weight_decay = state_dict["weight_decay"]
        self.dampening = state_dict["dampening"]
        self.nesterov = state_dict["nesterov"]
        self.maximize = state_dict["maximize"]
        self.smart_momentum = state_dict["smart_momentum"]
        self.state = state_dict["state"]
        self.iter = state_dict["iter"]
