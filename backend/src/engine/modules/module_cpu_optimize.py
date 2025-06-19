import torch
import numpy as np

class ModuleCpuOptimize:
    def __init__(self, optimizer="sgd"):
        self.name = "CpuOptimize"
        self.optimizer_name = optimizer

    def unique1d(self, input_tensor):
        input_np = input_tensor.cpu().numpy().astype(np.int32)
        unique = np.unique(input_np)
        return torch.tensor(unique, dtype=torch.int32)

    def run(self, engine, task):
        assert engine.is_node_master()
        engine.record_stat_start(task, "CpuOptimize")

        param_tensor = engine.tensor_from_mock(task.shared_cpu_tensor_entry_.param_)
        param_len = task.tensor_numel_
        grad_val = task.compressed_grad_val_ptr_
        grad_idx = task.compressed_grad_idx_ptr_
        grad_len = task.tensor_compressed_numel_

        assert grad_val is not None and grad_idx is not None

        world_size = engine.world_size()
        step_size = grad_len // world_size

        print(f"[DEBUG] grad_len = {grad_len}, world_size = {world_size}, step_size = {step_size}")

        merged_grad = torch.zeros_like(param_tensor)
        idx_list = []

        for i in range(world_size):
            offset = i * step_size
            tmp_idx = grad_idx[offset:offset + step_size].to(dtype=torch.int64)
            tmp_val = grad_val[offset:offset + step_size]
            if tmp_idx.numel() == 0:
                continue
            tmp_grad = torch.zeros_like(param_tensor)
            tmp_grad.index_put_((tmp_idx,), tmp_val)
            merged_grad += tmp_grad
            idx_list.append(tmp_idx)

        if len(idx_list) == 0:
            raise RuntimeError("No gradients were collected in CPU optimization.")

        merged_grad /= world_size
        concatenated_idx = torch.cat(idx_list)
        unique_idx = self.unique1d(concatenated_idx).to(dtype=torch.int64)
        compressed_grad = merged_grad.index_select(0, unique_idx)

        task.compressed_grad_val_ptr_ = compressed_grad
        task.compressed_grad_idx_ptr_ = unique_idx
        task.tensor_compressed_numel_ = unique_idx.numel()

        engine.record_stat_start(task, "CRIT_PATH_optimize_raw")

        if engine.sparse_optimizer is None:
            engine.get_sparse_optimizer(self.optimizer_name)

        # optimizer에 따라 호출 방식 분기
        if hasattr(engine.sparse_optimizer, "optimize_raw"):
            engine.sparse_optimizer.optimize_raw(
                param_tensor.data_ptr(),
                param_len,
                task.persistent_key(),
                compressed_grad.data_ptr(),
                unique_idx.data_ptr(),
                task.tensor_compressed_numel_
            )
        else:
            engine.sparse_optimizer.optimize(
                param_tensor,
                task.persistent_key(),
                compressed_grad,
                unique_idx
            )

        engine.record_stat_end(task, "CRIT_PATH_optimize_raw")
        engine.record_stat_end(task, "CpuOptimize")
        engine.record_stat_start(task, "CpuOptimizeBarrier")
        engine.schedule_after_barrier(engine.module_h2d_copy_pre, task, self.name)

        return "RESULT_SUCCESS"

