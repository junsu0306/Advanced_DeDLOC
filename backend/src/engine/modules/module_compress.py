from typing import List
import torch
from compress.compressor_wrapper import CompressorWrapper  # 압축기 래퍼

TASK_PENDING_COMM = 5
TASK_PENDING_MODEL_COMPLETE = 6

class ModuleCompress:
    def __init__(self, method="topk", ratio=0.1):  # method/ratio 선택 가능
        self.name = "Compress"
        self.compressor = CompressorWrapper(method=method, ratio=ratio)

    def run(self, engine, task):
        if not task.valid():
            return

        engine.record_stat_start(task, "compress")

        grad_tensor = task.gpu_grad_tensor()
        if grad_tensor is None:
            raise ValueError("Gradient tensor is None during compression")

        grad_tensor = grad_tensor.view(-1).clone().detach()
        grad_tensor.requires_grad_(False)
        grad_tensor_numel = grad_tensor.numel()
        task.tensor_numel_ = grad_tensor_numel

        # residual compensation
        residual = engine.get_grad_residual(task)
        residual_tensor = torch.tensor(residual, dtype=grad_tensor.dtype, device=grad_tensor.device)
        grad_tensor = grad_tensor + residual_tensor  # ✅ out-of-place 연산으로 수정 완료

        compress_ratio = getattr(engine, "compress_ratio", 0.1)
        idx, val = self.compressor.compress(task.persistent_key(), grad_tensor, compress_ratio)

        if idx is None or val is None:
            raise RuntimeError("Compression failed: returned None")

        task.compressed_grad_idx_ptr_ = idx
        task.compressed_grad_val_ptr_ = val
        task.tensor_compressed_numel_ = len(idx)

        if task.tensor_compressed_numel_ == 0:
            raise RuntimeError("Compressed tensor is empty")

        # update residual
        residual_tensor.zero_()
        residual_tensor.index_add_(0, idx, grad_tensor[idx] - val)
        engine.map_compensate_grad[task.persistent_key()] = residual_tensor.tolist()

        engine.record_stat_end(task, "compress")

        if getattr(engine, "is_node_master", lambda: False)():
            engine.schedule_after_use(engine.module_grad_exchange, task)
        else:
            engine.schedule_after_use(engine.module_model_complete, task)

