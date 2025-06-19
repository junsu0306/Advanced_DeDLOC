import torch
import threading
import time
from typing import List, Tuple, Optional

MAX_TASK_NAME_LEN = 128
MAX_NUM_GPU_PER_NODE = 8
MAX_NUM_BARRIER = 4
NUM_MAX_PARAM_DIM = 8

TASK_INITIALIZED = 0
TASK_PENDING = 1
TASK_RUNNING = 2
TASK_PENDING_CUDA = 3
TASK_PENDING_BARRIER = 4
TASK_PENDING_COMM = 5
TASK_PENDING_MODEL_COMPLETE = 6
TASK_PENDING_USE = 7
TASK_FINISHED = 8
TASK_IDLE = 9

UNINITIALIZED = 0
INITIALIZED = 1
FINISHED = 2


class SharedCpuTensorMock:
    def __init__(self):
        self.valid = False
        self.offset = 0
        self.len = [0] * NUM_MAX_PARAM_DIM
        self.len_len = 0


class SharedCpuTensorEntry:
    def __init__(self):
        self.valid_ = False
        self.entry_dup_idx_ = 0
        self.persistent_key_ = ""
        self.param_version_ = -1
        self.grad_version_ = [-1] * MAX_NUM_GPU_PER_NODE
        self.param_ = SharedCpuTensorMock()
        self.grad_ = [SharedCpuTensorMock() for _ in range(MAX_NUM_GPU_PER_NODE)]

    def initialize(self, skey: str, dup_iter_idx: int):
        assert not self.valid_
        self.valid_ = True
        self.persistent_key_ = skey[:MAX_TASK_NAME_LEN]
        self.param_ = SharedCpuTensorMock()
        self.grad_ = [SharedCpuTensorMock() for _ in range(MAX_NUM_GPU_PER_NODE)]
        self.entry_dup_idx_ = dup_iter_idx
        self.param_version_ = -1
        self.grad_version_ = [-1] * MAX_NUM_GPU_PER_NODE


class TrainBarrier:
    def __init__(self):
        self.name_ = ""
        self.state_ = UNINITIALIZED
        self.count_ = 0
        self.mutex_ = threading.Lock()


class TrainTaskSharedProps:
    def __init__(self):
        self.valid_ = False
        self.key_ = ""
        self.mutex_ = threading.Lock()
        self.shared_cpu_data_ready_ = [False] * MAX_NUM_GPU_PER_NODE
        self.finish_initiated_ = False
        self.is_finished_ = [False] * MAX_NUM_GPU_PER_NODE
        self.shared_test_field_ = [0] * MAX_NUM_GPU_PER_NODE
        self.train_barrier_ = [TrainBarrier() for _ in range(MAX_NUM_BARRIER)]

    def initialize(self, key: str):
        assert not self.valid_
        self.valid_ = True
        self.key_ = key[:MAX_TASK_NAME_LEN]
        for i in range(MAX_NUM_GPU_PER_NODE):
            self.is_finished_[i] = False
            self.shared_cpu_data_ready_[i] = False
            self.shared_test_field_[i] = 0
        for barrier in self.train_barrier_:
            barrier.count_ = 0
            barrier.state_ = UNINITIALIZED

    def lock(self):
        self.mutex_.acquire()

    def unlock(self):
        self.mutex_.release()


class TrainTaskV2:
    engine_ = None  # To be set with FasterDpEngine

    @staticmethod
    def set_engine(engine):
        TrainTaskV2.engine_ = engine

    @staticmethod
    def to_key(iter_cnt: int, layer: int, key_str: str) -> str:
        return f"{iter_cnt}@{layer}@{key_str}"[:MAX_TASK_NAME_LEN]

    @staticmethod
    def to_persistent_key(layer: int, key_str: str) -> str:
        return f"{layer}@{key_str}"[:MAX_TASK_NAME_LEN]

    def __init__(self, iter_cnt: int, layer: int, key_str: str):

        self.source_peer_id_ = None  #  통신 기반 task의 peer ID
        self.version_ = 0  #  해당 task가 참조하는 peer version


        self.iter_ = iter_cnt
        self.valid_ = True
        self.state_ = TASK_INITIALIZED
        self.barrier_id_ = 0
        self.barrier_id_future_ = 0
        self.tensor_numel_ = 0
        self.tensor_compressed_numel_ = 0
        self.grad_sync_iter_ = 0
        self.compressed_grad_val_ptr_ = None
        self.compressed_grad_idx_ptr_ = None
        self.key_ = self.to_key(iter_cnt, layer, key_str)
        self.persistent_key_ = self.to_persistent_key(layer, key_str)
        self.shared_props_ = None
        self.shared_cpu_tensor_entry_ = None
        self.priority_ = iter_cnt * 1000 + layer
        self.state_history_ = []
        self.debug_log_ = []
        self.debug_msg_ = ""
        self.gpu_param_tensor_ = None
        self.gpu_grad_tensor_ = None
        self.grad_sync_start_ = time.time()
        self.test_field_ = 0
        self.mutex_ = threading.Lock()

    def assign_gpu_tensor(self, param_tensor, grad_tensor):
        self.gpu_param_tensor_ = param_tensor
        self.gpu_grad_tensor_ = grad_tensor

    def free_gpu_grad_tensor(self):
        self.gpu_grad_tensor_ = None

    def free_gpu_param_tensor(self):
        self.gpu_param_tensor_ = None

    def prepare_delete(self):
        assert not self.valid_
        self.gpu_param_tensor_ = None
        self.gpu_grad_tensor_ = None
        self.state_history_.clear()
        self.debug_log_.clear()
        self.debug_msg_ = ""
        self.shared_props_ = None
        self.shared_cpu_tensor_entry_ = None

    def state_update(self, desired_state, apply_lock=True):
        if apply_lock:
            with TrainTaskV2.engine_.train_task_map_mutex_:
                self._state_update_impl(desired_state)
        else:
            self._state_update_impl(desired_state)

    def _state_update_impl(self, desired_state):
        if TrainTaskV2.engine_ and self not in TrainTaskV2.engine_.train_task_set_:
            assert desired_state == TASK_IDLE
            return

        if desired_state == TASK_IDLE and not self.valid():
            return

        with self.mutex_:
            current_state = self.state_
            if desired_state == TASK_IDLE and current_state == TASK_RUNNING:
                self.state_ = TASK_IDLE
            elif desired_state == TASK_PENDING:
                assert current_state != TASK_PENDING
                self.state_ = TASK_PENDING
            elif desired_state == TASK_RUNNING:
                assert current_state == TASK_PENDING
                self.state_ = TASK_RUNNING
            elif desired_state in {
                TASK_PENDING_CUDA, TASK_PENDING_COMM, TASK_PENDING_MODEL_COMPLETE,
                TASK_PENDING_BARRIER, TASK_PENDING_USE
            }:
                assert current_state in {TASK_RUNNING, TASK_IDLE, TASK_INITIALIZED}
                self.state_ = desired_state
            elif desired_state == TASK_FINISHED:
                assert current_state in {TASK_RUNNING, TASK_IDLE}
                self.state_ = TASK_FINISHED
            else:
                raise AssertionError("Invalid state transition")

            self.state_history_.append((desired_state, self.state_))

    def set_debug_message(self, msg):
        self.debug_msg_ = msg

    def get_debug_message(self):
        return self.debug_msg_

    def valid(self):
        return self.valid_

    def invalidate(self):
        self.valid_ = False

    def iter(self):
        return self.iter_

    def key(self):
        return self.key_

    def persistent_key(self):
        return self.persistent_key_

    def state(self):
        return self.state_

    def gpu_param_tensor(self):
        return self.gpu_param_tensor_

    def gpu_grad_tensor(self):
        return self.gpu_grad_tensor_

    def grad_sync_iter(self):
        return self.grad_sync_iter_

    def priority(self):
        return self.priority_

    def state_history(self):
        assert self.valid_
        return self.state_history_


    def set_peer_id(self, peer_id: str):
        self.source_peer_id_ = peer_id

    def get_peer_id(self) -> Optional[str]:
        return self.source_peer_id_

    def set_version(self, version: int):
        self.version_ = version

    def get_version(self) -> int:
        return self.version_
