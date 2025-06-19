
import torch
from types import SimpleNamespace

from engine.fasterdp_engine_full_v6 import FasterDpEngine
from engine.modules.module_compress import ModuleCompress
from engine.modules.module_h2d_copy import ModuleH2DCopyPre
from engine.modules.module_cpu_optimize import ModuleCpuOptimize
from engine.modules.module_d2h_copy import ModuleD2HCopy
from engine.modules.module_d2h_copy import ModuleD2HCopyPost
from engine.modules.module_cpu_gather import ModuleCpuGather
from optim.sgd_naive_optimizer import SGDNaive


class DummyComm:
    def queueTx(self, task, val, idx):
        print(f"[comm] queueTx called — simulated send with {len(idx)} elements.")

class DummyTask:
    def __init__(self):
        self._grad = torch.randn(100)
        self._param = torch.randn(100)
        self._shared = SimpleNamespace(shared_test_field_=[0, 0], shared_cpu_data_ready_=[False])
        self.shared_cpu_tensor_entry_ = SimpleNamespace(
            grad_=[self._grad.clone() for _ in range(2)],
            param_=self._param.clone(),
        )
        self.shared_props_ = self._shared
        self.test_field_ = 0
        self.tensor_numel_ = 100
        self.tensor_compressed_numel_ = 0
        self.compressed_grad_idx_ptr_ = None
        self.compressed_grad_val_ptr_ = None

    def valid(self): return True
    def persistent_key(self): return "layer_0"
    def gpu_grad_tensor(self): return self._grad.clone().detach().requires_grad_(True)
    def gpu_param_tensor(self): return self._param.clone().detach()
    def free_gpu_grad_tensor(self): pass
    def free_gpu_param_tensor(self): pass

class DummyEngine(FasterDpEngine):
    def __init__(self, local_rank):
        super().__init__()
        self._local_rank = local_rank
        self._world_size_internal = 2
        self.comm = DummyComm()
        self.map_compensate_grad = {}
        self.sparse_optimizer = SGDNaive()
        self.load_modules()

    def world_size(self): return self._world_size_internal
    def node_world_size(self): return self._world_size_internal
    def local_rank(self): return self._local_rank
    def tensor_from_mock(self, tensor): return tensor.clone().detach()

    def record_stat_start(self, task, name):
        print(f"[stat_start] {name}")
        setattr(task, f"__time_start_{name}", torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None)
        setattr(task, f"__time_end_{name}", torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None)
        if getattr(task, f"__time_start_{name}") is not None:
            getattr(task, f"__time_start_{name}").record()

    def record_stat_end(self, task, name):
        start = getattr(task, f"__time_start_{name}", None)
        end = getattr(task, f"__time_end_{name}", None)
        if end is not None:
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000.0
            print(f"[latency] {name}: {elapsed:.6f} sec")
        else:
            print(f"[stat_end] {name}")

    def schedule_after_use(self, module, task): print(f"→ scheduling {module.name}"); module.run(self, task)
    def schedule_after_barrier(self, module, task, tag): print(f"→ barrier to {module.name}"); module.run(self, task)
    def schedule_after_cuda(self, module, task): print(f"→ cuda to {module.name}"); module.run(self, task)
    def schedule_after_comm(self, task, next_state): print(f"→ dummy comm schedule, next_state={next_state}")
    def schedule_after_model_complete(self, module, task): print(f"→ model complete to {module.name}"); module.run(self, task)
    def update_model_version(self, task): print("[model_version] updated")
    def return_cpu_shmem_after_use(self, task): print("[shmem] returned")
    def schedule_terminate(self, task): print("[task] terminated")
    def get_grad_residual(self, task): return torch.zeros(task.tensor_numel_).tolist()
    def is_node_master(self): return True
    def compressor(self): return self.compressor

if __name__ == "__main__":
    engine = DummyEngine(local_rank=0)
    engine.module_cpu_optimize = ModuleCpuOptimize(optimizer="adam")
    engine.module_compress = ModuleCompress()
    engine.module_h2d_copy_pre = ModuleH2DCopyPre()
    engine.module_d2h_copy = ModuleD2HCopy(0)
    engine.module_d2h_copy_post = ModuleD2HCopyPost()
    engine.module_cpu_gather = ModuleCpuGather()

    task = DummyTask()

    print("\n[TEST] Begin running compress module → exchange")
    engine.module_compress.run(engine, task)

    print("\n[TEST] Begin running CPU optimize → H2D copy")
    engine.module_cpu_optimize.run(engine, task)
    engine.module_h2d_copy_pre.run(engine, task)

    print("\n[TEST] Begin running D2H copy")
    engine.module_d2h_copy.run(engine, task)

