import torch

TASK_PENDING_MODEL_COMPLETE = 6

class ModuleGradExchange:
    def __init__(self):
        self.name = "GradExchange"

    def run(self, engine, task):
        if not task.valid():
            return

        engine.record_stat_start(task, "grad_exchange")

        val = task.compressed_grad_val_ptr_
        idx = task.compressed_grad_idx_ptr_

        if val is None or idx is None:
            raise RuntimeError("Gradient exchange failed: Compressed values or indices are None")

        #  Hivemind Averager 연동
        if not hasattr(engine, "averager") or engine.averager is None:
            raise AttributeError("Engine missing 'averager' for Hivemind gradient exchange")

        # 압축 해제 (TopK 기준: idx, val → sparse tensor → dense로 복원)
        full_grad = torch.zeros_like(task.gpu_grad_tensor_)
        full_grad[idx] = val

        # 평균화 수행
        averaged_tensor = engine.averager.step(full_grad)

        # 평균된 gradient를 task.gpu_grad_tensor_에 직접 저장
        task.gpu_grad_tensor_.copy_(averaged_tensor)

        engine.record_stat_end(task, "grad_exchange")

        # 이후 단계로 전달
        engine.schedule_after_comm(task, TASK_PENDING_MODEL_COMPLETE)
