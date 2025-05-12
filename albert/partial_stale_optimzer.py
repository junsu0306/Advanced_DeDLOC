import logging
import torch
from hivemind.optim.collaborative import CollaborativeOptimizer as BaseCollaborativeOptimizer

logger = logging.getLogger(__name__)

class PartialStaleCollaborativeOptimizer(BaseCollaborativeOptimizer):
    """
    Partial Staleness (1-step delayed gradient application) + gradient 감쇠 및 timeout 적용.
    """

    def __init__(self, partial_stale=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partial_stale = partial_stale
        self.stale_grad_buffer = None
        self.last_applied_step = -1  # 마지막으로 stale grad를 적용한 스텝

    def step(self, batch_size: int = None, **kwargs):
        if not self.partial_stale:
            return super().step(batch_size=batch_size, **kwargs)

        # monkey-patch: grad를 적용하지 않고 buffer에 저장만 함
        orig_apply_accum = self.apply_accumulated_grads_
        local_grads = [None]

        def store_in_buffer(scale_by=None):
            param_list = [p for group in self.opt.param_groups for p in group["params"]]
            grads = []

            if self.reuse_grad_buffers:
                for p in param_list:
                    grads.append(None if p.grad is None else p.grad.clone())
            else:
                if self._grads is None:
                    self._grads = [torch.zeros_like(p) for p in param_list]
                if scale_by is not None:
                    for g in self._grads:
                        g.mul_(scale_by)
                for g in self._grads:
                    grads.append(g.clone())

            local_grads[0] = grads
            return  # opt.step()은 호출하지 않음

        self.apply_accumulated_grads_ = store_in_buffer
        super().step(batch_size=batch_size, **kwargs)
        self.apply_accumulated_grads_ = orig_apply_accum

        current_step = self.local_step

        # ✅ 이전 stale gradient 적용
        if self.stale_grad_buffer is not None:
            delay = current_step - self.last_applied_step
            if delay > 2:
                logger.warning(f"⚠️ Skipping stale gradient (delay={delay} steps)")
            else:
                self._apply_stale_grad(self.stale_grad_buffer, delay_steps=delay)

        # ✅ 이번 step의 gradient를 buffer에 저장
        if local_grads[0] is not None:
            self.stale_grad_buffer = local_grads[0]
            self.last_applied_step = current_step
        else:
            logger.debug("No grad from the super step. Possibly no peers or local step was skipped?")

        return

    def _apply_stale_grad(self, grad_list, delay_steps=1, gamma=0.95):
        """
        감쇠(decay)를 적용한 stale gradient 업데이트
        """
        decay = gamma ** delay_steps
        param_list = [p for group in self.opt.param_groups for p in group["params"]]

        if len(param_list) != len(grad_list):
            logger.warning("Mismatch: param_list len != grad_list len. Possibly a shape mismatch.")

        for p, g in zip(param_list, grad_list):
            if g is None:
                continue
            if p.grad is None:
                p.grad = decay * g.clone()
            else:
                p.grad.copy_(decay * g)

        self.opt.step()

        for p in param_list:
            if p.grad is not None:
                p.grad = None
