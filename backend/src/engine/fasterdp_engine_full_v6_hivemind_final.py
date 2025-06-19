# Python translation of FasterDpEngine (StellaTrain's core.cpp/core.h)

import threading
import torch
import torch.multiprocessing as mp
import numpy as np
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional

# === 압축기 및 옵티마이저 구현 import ===
from compress.thresholdv_compressor import ThresholdvCompressor
from compress.thresholdv16_compressor import ThresholdvCompressor16
from compress.topk_compressor import TopkCompressor

from optim.sgd_optimizer import SGDOptimizer as SGD
from optim.sgd_naive_optimizer import SGDNaive
from optim.adam_optimizer import AdamOptimizer as Adam


class StatEntry:
    def __init__(self):

    # === Hivemind 통신 연동 추가 ===
    self.dht = None
    self.averager = None

def set_dht(self, dht):
    self.dht = dht

def set_averager(self, averager):
    self.averager = averager

def dht_get(self, key):
    if self.dht is not None:
        return self.dht.get(key)
    return None

def dht_store(self, key, value, expiration=60):
    if self.dht is not None:
        import time
        expiration_time = time.time() + expiration
        self.dht.store(key, value, expiration_time=expiration_time)


        self.begin = None
        self.end = None


class StatItem:
    def __init__(self):
        self.entries: Dict[str, StatEntry] = {}
        self.lock = threading.Lock()


class TrainTaskSharedProps:
    def __init__(self, key):
        self.key = key
        self.valid = True
        self.finish_initiated = False
        self.is_finished = {}
        self.lock = threading.Lock()


class FasterDpEngine:
    def load_modules(self):
        from engine.modules.module_compress import ModuleCompress
        from engine.modules.module_grad_exchange import ModuleGradExchange
        from engine.modules.module_cpu_optimize import ModuleCpuOptimize
        from engine.modules.module_cpu_gather import ModuleCpuGather
        from engine.modules.module_barrier_checker import ModuleBarrierChecker
        from engine.modules.module_h2d_copy import ModuleH2DCopyPre, ModuleH2DCopy, ModuleH2DCopyPost
        from engine.modules.module_d2h_copy import ModuleD2HCopy, ModuleD2HCopyPost
        from engine.modules.module_null import ModuleNull

        self.module_compress = ModuleCompress()
        self.module_grad_exchange = ModuleGradExchange()
        self.module_cpu_optimize = ModuleCpuOptimize()
        self.module_cpu_gather = ModuleCpuGather()
        self.module_barrier_checker = ModuleBarrierChecker()
        self.module_h2d_copy_pre = ModuleH2DCopyPre()
        self.module_h2d_copy = ModuleH2DCopy(self.local_rank, num_workers=2)
        self.module_h2d_copy_post = ModuleH2DCopyPost()
        self.module_d2h_copy = ModuleD2HCopy(self.local_rank)
        self.module_d2h_copy_post = ModuleD2HCopyPost()
        self.module_null = ModuleNull()

    _instance = None

    def __init__(self):
        self.ready = False
        self.finished = False
        self.master = False
        self.node_master = False

        self.local_session_id = -1
        self.rank = -1
        self.local_rank = -1
        self._world_size_internal = -1
        self._node_world_size_internal = -1
        self.gradient_accumulation = 1

        self.model_staleness = 1
        self.first_backward = True

        self.compression_ratio = 0.99

        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.shm_manager = None
        self.comm_manager = None
        self.compressor = None
        self.sparse_optimizer = None

        self.barrier_cv = threading.Condition()
        self.finished_cv = threading.Condition()

        self.map_cpu_param_tensor: Dict[str, torch.Tensor] = {}
        self.map_gpu_param_tensor: Dict[str, torch.Tensor] = {}
        self.map_gpu_grad_tensor: Dict[str, torch.Tensor] = {}

        self.train_task_map: Dict[str, object] = {}
        self.train_task_set: set = set()

        self.shared_task_props: Dict[str, TrainTaskSharedProps] = {}
        self.shared_task_props_lock = threading.Lock()

        self.layer_model_version_map: Dict[str, int] = {}
        self.layer_iteration_cnt_map: Dict[str, int] = {}
        self.layer_model_completed_version_map: Dict[str, int] = {}
        self.layer_model_completed_version_map_cond = threading.Condition()
        self.layer_alloc_cnt_map: Dict[str, int] = {}
        self.layer_alloc_cnt_map_lock = threading.Lock()

        self.backward_delegate_list: List[Future] = []
        self.backward_delegate_lock = threading.Lock()
        self.backward_delegate_cond = threading.Condition(self.backward_delegate_lock)

        self.task_stat_map: Dict[str, StatItem] = {}
        self.task_stat_last_event = ""
        self.task_stat_map_lock = threading.Lock()

        self.set_initialized_params = set()
        self.set_initialized_params_lock = threading.Lock()

        self.init_model_recv_map: Dict[str, torch.Tensor] = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def configure(self, master_addr: str, master_port: int, world_size: int, rank: int,
                  local_session_id: int, local_world_size: int = 0, local_rank: int = 0,
                  method: str = "thresholdv16", gradient_accumulation: int = 1):
        self.master_addr = master_addr
        self.master_port = master_port
        self.rank = rank
        self.local_session_id = local_session_id
        self.local_rank = local_rank
        self._world_size_internal = world_size
        self._node_world_size_internal = local_world_size
        self.gradient_accumulation = gradient_accumulation

        self.node_master = (local_rank == 0)
        self.master = (rank == 0 and self.node_master)

        if method == "thresholdv":
            self.compressor = ThresholdvCompressor()
        elif method == "thresholdv16":
            self.compressor = ThresholdvCompressor16()
        elif method == "topk":
            self.compressor = TopkCompressor()
        else:
            raise ValueError(f"Unknown compression method: {method}")

        self.ready = True
        print("FasterDpEngine configured.")

    def configure_compressor(self, method="topk"):
        from compress.compressor_wrapper import CompressorWrapper
        self.compressor = CompressorWrapper(method)
        if method == "topk":
            from compress.topk_compressor import TopkCompressor as Compressor
        elif method == "thresholdv":
            from compress.thresholdv_compressor import ThresholdvCompressor as Compressor
        elif method == "thresholdv16":
            from compress.thresholdv16_compressor import ThresholdvCompressor16 as Compressor
        else:
            raise ValueError(f"Unknown compression method: {method}")
        self.compressor = Compressor()

    def configure_compression_ratio(self, ratio: float):
        assert 0 < ratio <= 1
        self.compression_ratio = ratio

    def get_sparse_optimizer(self, optimizer: str):
        if optimizer == "sgd":
            self.sparse_optimizer = SGD()
        elif optimizer == "sgd_naive":
            self.sparse_optimizer = SGDNaive()
        elif optimizer == "adam":
            self.sparse_optimizer = Adam()
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def compress(self, name: str, tensor: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if ratio < 0 or ratio > 1:
            raise ValueError("Ratio must be in [0, 1]")

        numel_to_select = int((1. - ratio) * tensor.numel())
        if numel_to_select == 0:
            return torch.empty(0, dtype=torch.int32), torch.empty(0)

        dst_idx = torch.empty(numel_to_select, dtype=torch.int32)
        dst_val = torch.empty(numel_to_select)

        total_len = self.compressor.compress(
            name,
            tensor.view(-1),
            numel_to_select,
            dst_idx,
            dst_val
        )

        return dst_idx[:total_len], dst_val[:total_len]


    def pre_train_init(self, layer_idx: int, name: str, gpu_param: torch.Tensor):
        skey = f"{layer_idx}@{name}"

        with self.set_initialized_params_lock:
            if skey in self.set_initialized_params:
                return
            self.set_initialized_params.add(skey)

        def hook_fn(grad):
            with threading.Lock():
                if skey not in self.map_gpu_grad_tensor:
                    self.map_gpu_grad_tensor[skey] = grad.clone()
                else:
                    self.map_gpu_grad_tensor[skey] += grad
            self.post_backward_process(layer_idx, name, self.map_gpu_grad_tensor[skey], gpu_param)

        gpu_param.register_hook(hook_fn)
        self.map_gpu_param_tensor[skey] = gpu_param
def post_backward_process(self, layer_idx: int, name: str, gpu_grad_tensor: torch.Tensor, gpu_param_tensor: torch.Tensor):
        skey = f"{layer_idx}@{name}"

        if self.first_backward:
            self.first_backward = False
            with self.barrier_cv:
                self.barrier_cv.notify_all()

        def task_fn():
            print(f"[DEBUG] Post-backward for {skey}")

            iter_idx = self.update_layer_model_version(skey)
            task = TrainTaskV2(iter_cnt=iter_idx, layer=layer_idx, key_str=name)
            task.assign_gpu_tensor(gpu_param_tensor, gpu_grad_tensor)

            self.train_task_map[task.key()] = task
            self.schedule(self.module_compress, task)

        self.schedule_after_use(task=None, module_fn=task_fn)


    def synchronize_backend(self):
        print("Synchronizing backend... waiting for all post-backward tasks to complete.")

        while self.backward_delegate_list:
            with self.backward_delegate_lock:
                remaining = []
                for i, fut in enumerate(self.backward_delegate_list):
                    if fut.done():
                        try:
                            fut.result()
                        except Exception as e:
                            print(f"Exception in task {i}: {e}")
                    else:
                        remaining.append(fut)
                self.backward_delegate_list = remaining
            time.sleep(0.1)

        print("All post-backward tasks completed.")

    def force_model_sync(self, skey: str, dry_run: bool = False):
        if not self.node_master:
            return
        if self._world_size_internal == 1:
            return

        if self.master:
            cpu_tensor = self.map_cpu_param_tensor.get(skey)
            if cpu_tensor is None:
                print(f"[WARN] No CPU tensor for {skey} to broadcast.")
                return
            for peer in range(1, self.world_size):
                self.init_model_recv_map[skey] = cpu_tensor.clone()  # simulate sending
        else:
            received = self.init_model_recv_map.get(skey)
            if received is None:
                print(f"[WARN] No received model for {skey}")
                return
            current = self.map_cpu_param_tensor.get(skey)
            if current is None:
                self.map_cpu_param_tensor[skey] = received.clone()
            elif not dry_run:
                self.map_cpu_param_tensor[skey].copy_(received)

    def retrieve_train_task_shared_props(self, key: str) -> TrainTaskSharedProps:
        with self.shared_task_props_lock:
            if key in self.shared_task_props:
                return self.shared_task_props[key]
            prop = TrainTaskSharedProps(key)
            prop.is_finished = {i: False for i in range(self.node_world_size)}
            self.shared_task_props[key] = prop
            return prop

    def update_layer_model_version(self, skey: str) -> int:
        with self.layer_alloc_cnt_map_lock:
            if skey not in self.layer_alloc_cnt_map:
                self.layer_alloc_cnt_map[skey] = 0
            iter_count = self.layer_alloc_cnt_map[skey]
            self.layer_alloc_cnt_map[skey] += 1
            return iter_count

    def notify_layer_usage_finished(self, skey: str, iter_idx: int):
        with self.layer_model_completed_version_map_cond:
            prev = self.layer_model_completed_version_map.get(skey, -1)
            self.layer_model_completed_version_map[skey] = max(prev, iter_idx - self.model_staleness)
            self.layer_model_completed_version_map_cond.notify_all()

    def notify_all_layer_usage_finished(self):
        with self.layer_model_completed_version_map_cond:
            for skey, iter_idx in self.layer_alloc_cnt_map.items():
                prev = self.layer_model_completed_version_map.get(skey, -1)
                self.layer_model_completed_version_map[skey] = max(prev, iter_idx - self.model_staleness)
            self.layer_model_completed_version_map_cond.notify_all()


    def pre_forward_process(self, layer_idx: int, name: str):
        skey = f"{layer_idx}@{name}"
        with self.layer_model_completed_version_map_cond:
            while self.layer_model_version_map.get(skey, 0) < self.model_staleness:
                self.layer_model_completed_version_map_cond.wait(timeout=0.1)

    def update_model_version(self, skey: str):
        with self.layer_model_completed_version_map_cond:
            self.layer_model_version_map[skey] = self.layer_model_version_map.get(skey, 0) + 1
            self.layer_model_completed_version_map_cond.notify_all()

    def get_model_version(self, skey: str) -> int:
        return self.layer_model_version_map.get(skey, 0)

    def get_completed_model_version(self, skey: str) -> int:
        return self.layer_model_version_map.get(skey, 0) - self.model_staleness

    def get_cpu_param_tensor(self, skey: str) -> torch.Tensor:
        tensor = self.map_cpu_param_tensor.get(skey)
        if tensor is None:
            raise KeyError(f"CPU parameter tensor for {skey} not found.")
        return tensor

    def print_current_stat(self):
        print("=== Task Status ===")
        for key, task in self.train_task_map.items():
            print(f"Task Key: {key} | Valid: {getattr(task, 'valid', True)}")
        print("===================")



    def record_stat_start(self, task_key: str, event: str):
        with self.task_stat_map_lock:
            if task_key not in self.task_stat_map:
                self.task_stat_map[task_key] = StatItem()
            stat_item = self.task_stat_map[task_key]
            with stat_item.lock:
                entry = stat_item.entries.get(event)
                if entry is None:
                    entry = StatEntry()
                    stat_item.entries[event] = entry
                entry.begin = time.time()
                self.task_stat_last_event = event

    def record_stat_end(self, task_key: str, event: str):
        with self.task_stat_map_lock:
            stat_item = self.task_stat_map.get(task_key)
            if not stat_item:
                return
            with stat_item.lock:
                entry = stat_item.entries.get(event)
                if entry:
                    entry.end = time.time()

    def get_grad_residual(self, skey: str, compressed_idx: torch.Tensor, compressed_val: torch.Tensor, total_shape: int) -> torch.Tensor:
        # reconstruct sparse tensor and subtract from original for residual
        reconstructed = torch.zeros(total_shape, dtype=compressed_val.dtype)
        reconstructed[compressed_idx] = compressed_val
        original = self.map_gpu_grad_tensor[skey]
        residual = original - reconstructed
        return residual

    def free_task(self, task):
        assert task
        shprops = task.shared_props
        with shprops.lock:
            shprops.is_finished[self.local_rank] = True
            if not shprops.finish_initiated:
                if all(shprops.is_finished.values()):
                    shprops.finish_initiated = True
                    task.valid = False
                    if task.key in self.train_task_map:
                        del self.train_task_map[task.key]
        print(f"Task {task.key} is freed.")

    def schedule_after_use(self, task, module_fn=None):
        if module_fn is None:
            module_fn = lambda: print(f"Running module for task {task.key}")
        future = self.thread_pool.submit(module_fn)
        with self.backward_delegate_lock:
            self.backward_delegate_list.append(future)
            self.backward_delegate_cond.notify()


    def find_task_by_key(self, key: str):
        return self.train_task_map.get(key, None)

    def task_state_update_by_key(self, key: str, new_state: str):
        task = self.find_task_by_key(key)
        if task is None:
            print(f"[WARN] Task with key {key} not found")
            return
        task.state = new_state
        print(f"[INFO] Task {key} updated to state {new_state}")

    def update_micro_iteration(self, skey: str):
        if skey not in self.layer_iteration_cnt_map:
            self.layer_iteration_cnt_map[skey] = 0
        self.layer_iteration_cnt_map[skey] += 1
        return self.layer_iteration_cnt_map[skey]

    def schedule_after_cuda(self, task, callback=None):
        def default_fn():
            print(f"[SCHEDULE] CUDA callback executed for task {task.key}")
        fn = callback if callback else default_fn
        self.schedule_after_use(task, fn)

    def schedule_after_barrier(self, task, callback=None):
        def default_fn():
            print(f"[SCHEDULE] Barrier passed for task {task.key}")
        fn = callback if callback else default_fn
        self.schedule_after_use(task, fn)

    def schedule_terminate(self, task, callback=None):
        def default_fn():
            print(f"[SCHEDULE] Terminate task {task.key}")
        fn = callback if callback else default_fn
        self.schedule_after_use(task, fn)

    def report_cuda_finished_impl(self, task_key: str):
        print(f"[CUDA] Task {task_key} CUDA work reported done")

# Placeholder implementations (to be replaced with actual ones from StellaTrain modules)


    def world_size(self):
        return self._world_size_internal

    def node_world_size(self):
        return self._node_world_size_internal

