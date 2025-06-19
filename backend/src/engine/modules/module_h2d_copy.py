import torch

class ModuleH2DCopyPre:
    def __init__(self):
        self.name = "H2D_Copy_Pre"

    def run(self, engine, task):
        engine.record_stat_end(task, "CpuOptimizeBarrier")
        engine.record_stat_start(task, "H2DCopyPre")

        if getattr(engine, "SKIP_SOME_CRITICAL_PATHS", False):
            engine.schedule_after_model_complete(engine.module_h2d_copy_post, task)
        else:
            engine.schedule_after_model_complete(engine.module_h2d_copy, task)

        return "RESULT_SUCCESS"


class ModuleH2DCopy:
    def __init__(self, node_rank, num_workers=2):  # ✅ 핵심
        self.name = "H2D_Copy"
        self.use_cuda = torch.cuda.is_available()
        self.streams = (
            [torch.cuda.Stream(device=node_rank) for _ in range(num_workers)]
            if self.use_cuda else [None]
        )

    def run(self, engine, task):
        engine.record_stat_end(task, "H2DCopyPre")
        engine.record_stat_start(task, "H2DCopy")

        cpu_param_tensor = engine.tensor_from_mock(task.shared_cpu_tensor_entry_.param_)
        gpu_param_tensor = task.gpu_param_tensor()

        if self.use_cuda:
            stream = self.streams[0]
            with torch.cuda.stream(stream):
                gpu_param_tensor.copy_(cpu_param_tensor, non_blocking=True)
                stream.synchronize()
        else:
            gpu_param_tensor.copy_(cpu_param_tensor)

        engine.schedule_after_cuda(engine.module_h2d_copy_post, task)
        return "RESULT_SUCCESS"


class ModuleH2DCopyPost:
    def __init__(self):
        self.name = "H2D_Copy_Post"

    def run(self, engine, task):
        engine.record_stat_end(task, "H2DCopy")
        task.free_gpu_param_tensor()
        engine.update_model_version(task)
        engine.record_stat_end(task, "Total")
        engine.return_cpu_shmem_after_use(task)
        engine.schedule_terminate(task)

        return "RESULT_SUCCESS"

