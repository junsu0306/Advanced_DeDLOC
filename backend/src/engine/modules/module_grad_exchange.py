import torch

TASK_PENDING_MODEL_COMPLETE = 6

class ModuleGradExchange:
    def __init__(self):
        self.name = "GradExchange"  # .name 속성 추가

    def run(self, engine, task):
        if not task.valid():
            return

        engine.record_stat_start(task, "grad_exchange")

        val = task.compressed_grad_val_ptr_
        idx = task.compressed_grad_idx_ptr_

        if val is None or idx is None:
            raise RuntimeError("Gradient exchange failed: Compressed values or indices are None")

        if not hasattr(engine, "comm"):
            raise AttributeError("Engine missing 'comm' for communication")

        # Placeholder 통신 로직 (향후 DeDLOC 연동 시 대체 예정)
        engine.comm.queueTx(task, val, idx)

        engine.record_stat_end(task, "grad_exchange")

        engine.schedule_after_comm(task, TASK_PENDING_MODEL_COMPLETE)

