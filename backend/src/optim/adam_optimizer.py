import torch
import threading

class AdamOptimizer(torch.nn.Module):
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, weight_decay=0, eps=1e-8, 
                 amsgrad=False, maximize=False, device="cuda"):
        super(AdamOptimizer, self).__init__()
        self.lr = lr
        self.b1 = b1  # First moment coefficient
        self.b2 = b2  # Second moment coefficient
        self.weight_decay = weight_decay
        self.eps = eps
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.iteration = 1  # Start from 1 (tick in C++)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.lock = threading.Lock()  # Mutex replacement
        
        # State storage for first and second moments
        self.momentum_m = {}
        self.momentum_v = {}
        self.v_max = {} if amsgrad else None

    def optimize(self, param: torch.Tensor, name: str, grad_values: torch.Tensor, grad_indices: torch.Tensor):
        """ Adam Optimizer implementation using sparse gradients. """
        param = param.to(self.device)
        grad_values = grad_values.to(self.device)
        grad_indices = grad_indices.to(self.device).long()
        
        self.lock.acquire()
        
        if name not in self.momentum_m:
            self.momentum_m[name] = torch.zeros_like(param, device=self.device)
            self.momentum_v[name] = torch.zeros_like(param, device=self.device)
            if self.amsgrad:
                self.v_max[name] = torch.zeros_like(param, device=self.device)
        
        m = self.momentum_m[name]
        v = self.momentum_v[name]
        vmax = self.v_max[name] if self.amsgrad else None

        lr_t = self.lr * (torch.sqrt(torch.tensor(1 - self.b2 ** self.iteration, dtype=torch.float32)) / (1 - self.b1 ** self.iteration))

        with torch.no_grad():
            if self.weight_decay != 0:
                grad_values += self.weight_decay * param.index_select(0, grad_indices)
            
            # Update first and second moment estimates
            m.index_add_(0, grad_indices, (1 - self.b1) * (grad_values - m.index_select(0, grad_indices)))
            v.index_add_(0, grad_indices, (1 - self.b2) * (grad_values**2 - v.index_select(0, grad_indices)))
            
            # Bias correction
            mt_hat = m.index_select(0, grad_indices) / (1 - self.b1 ** self.iteration)
            vt_hat = v.index_select(0, grad_indices) / (1 - self.b2 ** self.iteration)
            
            # AMSGrad correction if enabled
            if self.amsgrad:
                vmax.index_put_((grad_indices,), torch.max(vmax.index_select(0, grad_indices), vt_hat))
                vt_hat = vmax.index_select(0, grad_indices)
            
            # Parameter update
            param.scatter_(0, grad_indices, param.index_select(0, grad_indices) - lr_t * mt_hat / (torch.sqrt(vt_hat) + self.eps))
        
        self.iteration += 1
        self.lock.release()
    
    def configure(self, option_name, option_value):
        """ Configure optimizer hyperparameters dynamically. """
        if option_name == "lr":
            self.lr = option_value
        elif option_name == "b1":
            self.b1 = option_value
        elif option_name == "b2":
            self.b2 = option_value
        elif option_name == "weight_decay":
            self.weight_decay = option_value
        elif option_name == "eps":
            self.eps = option_value
        elif option_name == "amsgrad":
            self.amsgrad = option_value
            if self.amsgrad and self.v_max is None:
                self.v_max = {}
        elif option_name == "maximize":
            self.maximize = option_value
        else:
            raise ValueError(f"Unknown option: {option_name}")

# Compile for optimization
AdamOptimizer = torch.compile(AdamOptimizer)
