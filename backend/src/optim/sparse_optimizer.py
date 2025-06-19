import torch
import numpy as np

class SparseOptimizer:
    def __init__(self, lr=1e-3):
        """ 초기화 """
        self.lr = lr

    def get_lr(self):
        """ 학습률 반환 """
        return self.lr

    def set_lr(self, lr):
        """ 학습률 설정 """
        self.lr = lr

    def check_validity(self, param, grad_values, grad_indices):
        """ 입력 데이터 유효성 검사 """
        assert param.is_contiguous(), "Parameter tensor must be contiguous"
        assert grad_values.is_contiguous(), "Gradient values tensor must be contiguous"
        assert grad_indices.is_contiguous(), "Gradient indices tensor must be contiguous"

        assert param.dtype == torch.float32, "Parameter tensor must be float32"
        assert grad_values.dtype == torch.float32, "Gradient values tensor must be float32"
        assert grad_indices.dtype == torch.int32, "Gradient indices tensor must be int32"

        assert grad_indices.numel() == grad_values.numel(), "Gradient indices and values must have the same number of elements"

    def optimize(self, param, name, grad_values, grad_indices):
        """ PyTorch 기반 최적화 실행 (Sparse Gradient 적용) """
        self.check_validity(param, grad_values, grad_indices)

        # Sparse gradient update
        param.data[grad_indices] -= self.lr * grad_values

    def optimize_raw(self, param, grad_values, grad_indices):
        """ NumPy 기반 최적화 실행 (메모리 직접 접근) """
        assert isinstance(param, np.ndarray), "param must be a NumPy array"
        assert isinstance(grad_values, np.ndarray), "grad_values must be a NumPy array"
        assert isinstance(grad_indices, np.ndarray), "grad_indices must be a NumPy array"

        # Sparse gradient update
        param[grad_indices] -= self.lr * grad_values

    def configure(self, option_name, option_value):
        """ 설정 변경 """
        if option_name == "learning_rate":
            assert isinstance(option_value, (float, int)), "learning_rate must be a float or int"
            self.lr = float(option_value)
        else:
            raise ValueError(f"Unknown option: {option_name}")

    def name(self):
        """ Optimizer 이름 반환 (추상 메서드) """
        raise NotImplementedError("Subclasses must implement 'name' method")
