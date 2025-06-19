import torch

class ModuleCpuGather:
    def __init__(self):
        self.name = "CPU_Gather"

    def run(self, engine, task):
        engine.record_stat_end(task, "D2HCopyBarrier")
        engine.record_stat_start(task, "CpuGather")

        local_rank = engine.local_rank
        num_gpus = engine.node_world_size()

        dst_cpu_grad_tensor = engine.tensor_from_mock(task.shared_cpu_tensor_entry_.grad_[0])
        tensor_len = dst_cpu_grad_tensor.numel()
        start_idx = (tensor_len * local_rank) // num_gpus
        end_idx = (tensor_len * (local_rank + 1)) // num_gpus
        target_len = end_idx - start_idx

        for i in range(num_gpus):
            if i == 0:
                src = torch.tensor(engine.get_grad_residual(task), dtype=torch.float32)[start_idx:end_idx]
            else:
                src_tensor = engine.tensor_from_mock(task.shared_cpu_tensor_entry_.grad_[i])
                src = src_tensor[start_idx:end_idx]

            dst = dst_cpu_grad_tensor[start_idx:end_idx]
            engine.record_stat_start(task, f"CRIT_PATH_gather_{i}")
            dst += src
            engine.record_stat_end(task, f"CRIT_PATH_gather_{i}")

        task.shared_props_.shared_cpu_data_ready_[local_rank] = True

        engine.record_stat_end(task, "CpuGather")
        engine.record_stat_start(task, "CpuGatherBarrier")
        engine.schedule_after_barrier(engine.module_compress, task, self.name)

        return "RESULT_SUCCESS"

