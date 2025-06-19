import threading
from typing import Callable, Dict, Optional, List
from queue import Queue
from collections import defaultdict
from concurrent.futures import Future

class FasterDpEngine:
    def __init__(self):
        self.lst_futures: List[Future] = []
        self.lst_futures_write_mutex = threading.Lock()
        self.thread_pool = None  # Set externally

        self.cpu_shmem_use_map = {}
        self.cpu_shmem_use_map_mutex = threading.Lock()
        self.cpu_shmem_use_callback_map = {}
        self.cpu_shmem_use_map_cond = threading.Condition(self.cpu_shmem_use_map_mutex)

        self.layer_model_completed_version_map = defaultdict(int)
        self.layer_model_completed_callback_map = {}
        self.layer_model_completed_version_map_mutex = threading.Lock()
        self.layer_model_completed_version_map_cond = threading.Condition(self.layer_model_completed_version_map_mutex)

        self.cuda_wait_callback_map = {}
        self.cuda_wait_callback_map_mutex = threading.Lock()

        self.layer_model_version_map = defaultdict(int)
        self.layer_model_map_mutex = threading.Lock()
        self.train_task_map = {}
        self.train_task_map_mutex = threading.Lock()
        self.map_compensate_grad = {}
        self.map_compensate_grad_mutex = threading.Lock()

    def schedule(self, mod, task):
        task.state_update(TASK_PENDING)
        with self.lst_futures_write_mutex:
            self.lst_futures.append(
                self.thread_pool.enqueue_priority(task.priority(), lambda: self._run_module(mod, task, "schedule"))
            )

    def schedule_terminate(self, task):
        task.state_update(TASK_FINISHED)
        with self.lst_futures_write_mutex:
            self.lst_futures.append(
                self.thread_pool.enqueue_priority(task.priority(), lambda: self._free_task(task))
            )

    def _run_module(self, mod, task, debug_str):
        task.state_update(TASK_RUNNING)
        task.set_debug_message(debug_str)
        mod.run(self, task)
        task.state_update(TASK_IDLE)

    def _free_task(self, task):
        # Implement this method in context
        pass

    def return_cpu_shmem_after_use(self, task):
        spkey = (task.persistent_key(), task.iter())
        with self.cpu_shmem_use_map_mutex:
            assert spkey in self.cpu_shmem_use_map
            assert self.cpu_shmem_use_map[spkey] == task
            del self.cpu_shmem_use_map[spkey]
            self.cpu_shmem_use_map_cond.notify_all()
            task.shared_cpu_tensor_entry_ = None

    def update_model_version(self, task):
        skey = task.persistent_key()
        with self.layer_model_map_mutex:
            self.layer_model_version_map[skey] += 1
        # Notify threads waiting on model update

    def get_model_version(self, task):
        skey = task.persistent_key()
        with self.layer_model_map_mutex:
            return self.layer_model_version_map[skey]

    def get_completed_model_version(self, task):
        skey = task.persistent_key()
        with self.layer_model_completed_version_map_mutex:
            return self.layer_model_completed_version_map[skey]

    def find_task_by_key(self, key):
        with self.train_task_map_mutex:
            assert key in self.train_task_map
            return self.train_task_map[key]

    def task_state_update_by_key(self, key, desired_state):
        with self.train_task_map_mutex:
            task = self.train_task_map[key]
            assert task
            task.state_update(desired_state, False)

    def record_stat_start(self, task, event):
        pass  # Stats collection optional

    def record_stat_end(self, task, event):
        pass

    def get_grad_residual(self, task):
        skey = task.persistent_key()
        with self.map_compensate_grad_mutex:
            if skey not in self.map_compensate_grad:
                self.map_compensate_grad[skey] = [0.0] * task.tensor_numel_
            return self.map_compensate_grad[skey]

    def tensor_from_mock(self, mock):
        # This requires shm_manager module implementation
        pass

    def load_modules(self):
        self.module_d2h_copy = ModuleD2HCopy(self.local_rank)
        self.module_d2h_copy_post = ModuleD2HCopyPost()
        self.module_cpu_gather = ModuleCpuGather()
        self.module_compress = ModuleCompress()
        self.module_grad_exchange = ModuleGradExchange()
        self.module_cpu_optimize = ModuleCpuOptimize()
        self.module_h2d_copy_pre = ModuleH2DCopyPre()
        self.module_h2d_copy = ModuleH2DCopy(self.local_rank)
        self.module_h2d_copy_post = ModuleH2DCopyPost()
        self.module_barrier_checker = ModuleBarrierChecker()  # optional

    def check_model_ready(self, key):
        """ DHT를 통해 모델 동기화 여부를 확인하는 함수 """
        if hasattr(self, 'dht') and self.dht is not None:
            result = self.dht.get(f"{key}_ready")
            return result.value if result else False
        return False

