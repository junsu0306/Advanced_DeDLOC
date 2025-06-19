import torch

class ModuleD2HCopy:
    def __init__(self, node_rank, num_workers=2):
        self.name = "D2H_Copy"
        self.use_cuda = torch.cuda.is_available()
        self.streams = (
            [torch.cuda.Stream(device=node_rank) for _ in range(num_workers)]
            if self.use_cuda else [None]
        )

    def run(self, engine, task):
        engine.record_stat_start(task, "Total")
        engine.record_stat_start(task, "D2HCopy")

        gpu_grad_tensor = task.gpu_grad_tensor()
        cpu_grad_tensor = engine.tensor_from_mock(
            task.shared_cpu_tensor_entry_.grad_[engine.local_rank]
        )

        if self.use_cuda and gpu_grad_tensor.is_cuda:
            stream = self.streams[engine.local_rank]
            with torch.no_grad():
                with torch.cuda.stream(stream):
                    cpu_grad_tensor.copy_(gpu_grad_tensor, non_blocking=True)
                    gpu_grad_tensor.record_stream(stream)
                    stream.synchronize()
        else:
            with torch.no_grad():
                cpu_grad_tensor.copy_(gpu_grad_tensor)

        engine.schedule_after_cuda(engine.module_d2h_copy_post, task)
        return "RESULT_SUCCESS"


class ModuleD2HCopyPost:
    def __init__(self):
        self.name = "D2H_Copy_Post"

    def run(self, engine, task):
        engine.record_stat_end(task, "D2HCopy")
        engine.record_stat_start(task, "D2HCopyBarrier")

        gpu_grad_tensor = task.gpu_grad_tensor()
        with torch.no_grad():
            gpu_grad_tensor.zero_()
            task.free_gpu_grad_tensor()

        engine.schedule_after_barrier(engine.module_cpu_gather, task, self.name)
        return "RESULT_SUCCESS"

