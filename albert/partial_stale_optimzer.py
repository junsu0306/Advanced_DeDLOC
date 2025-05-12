import logging
import torch
from hivemind.optim.collaborative import CollaborativeOptimizer as BaseCollaborativeOptimizer
import hivemind

logger = logging.getLogger(__name__)

class PartialStaleCollaborativeOptimizer(BaseCollaborativeOptimizer):
    """
    Partial Staleness Optimizer with:
    - Gradient delay tolerance
    - Gradient clipping
    - DHT heartbeat for stability
    - Fallback for missing gradients
    - Loss explosion guard
    """

    def __init__(
        self,
        partial_stale: bool = False,
        heartbeat_interval: int = 100,
        clip_grad_norm: float = 1.0,
        max_loss_threshold: float = 50.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.partial_stale = partial_stale
        self.stale_grad_buffer = None
        self.last_applied_step = -1
        self.heartbeat_interval = heartbeat_interval
        self.clip_grad_norm = clip_grad_norm
        self.max_loss_threshold = max_loss_threshold

    def step(self, batch_size: int = None, **kwargs):
        if not self.partial_stale:
            return super().step(batch_size=batch_size, **kwargs)

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
            return

        self.apply_accumulated_grads_ = store_in_buffer
        super().step(batch_size=batch_size, **kwargs)
        self.apply_accumulated_grads_ = orig_apply_accum

        current_step = self.local_step

        # Heartbeat to DHT to maintain active status
        if current_step % self.heartbeat_interval == 0:
            try:
                self.dht.store(
                    key=f"{self.prefix}_heartbeat",
                    subkey=self.peer_id,
                    value={"step": current_step},
                    expiration_time=hivemind.get_dht_time() + 60,
                    return_future=False,
                )
                logger.debug(f"[DHT] Heartbeat sent at step {current_step}")
            except Exception as e:
                logger.warning(f"[DHT] Heartbeat failed: {e}")

        # Apply stale grad with delay handling
        if self.stale_grad_buffer is not None:
            delay = current_step - self.last_applied_step
            if delay > 10:
                logger.warning(f"[PartialStale] ❌ Dropping stale_grad_buffer (delay={delay})")
                self.stale_grad_buffer = None
            else:
                self._apply_stale_grad(self.stale_grad_buffer, delay_steps=delay)
        else:
            logger.warning(f"[PartialStale] 🚫 No stale gradient buffer to apply at step {current_step}")

        # Buffer new grad for next step
        if local_grads[0] is not None:
            self.stale_grad_buffer = local_grads[0]
            self.last_applied_step = current_step
        else:
            logger.warning(f"[PartialStale] ⚠️ No gradients stored at step {current_step}.")

        return

    def _apply_stale_grad(self, grad_list, delay_steps=1, gamma=0.95):
        decay = gamma ** delay_steps
        param_list = [p for group in self.opt.param_groups for p in group["params"]]

        if len(param_list) != len(grad_list):
            logger.warning("[PartialStale] Mismatch: param_list len != grad_list len.")

        total_norm = 0.0
        for p, g in zip(param_list, grad_list):
            if g is None:
                continue
            scaled_grad = decay * g
            total_norm += scaled_grad.norm().item() ** 2
            if p.grad is None:
                p.grad = scaled_grad.clone()
            else:
                p.grad.copy_(scaled_grad)

        total_norm = total_norm ** 0.5

        if total_norm > self.max_loss_threshold:
            logger.warning(f"[PartialStale] 🚨 Skipping gradient step due to large norm: {total_norm:.2f} > {self.max_loss_threshold}")
            for p in param_list:
                if p.grad is not None:
                    p.grad = None
            return

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(param_list, max_norm=self.clip_grad_norm)
        self.opt.step()

        for p in param_list:
            if p.grad is not None:
                p.grad = None
