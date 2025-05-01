# file: partial_stale_collaborative.py

import logging
import torch
from hivemind.optim.collaborative import CollaborativeOptimizer as BaseCollaborativeOptimizer

logger = logging.getLogger(__name__)

class PartialStaleCollaborativeOptimizer(BaseCollaborativeOptimizer):
    """
    1-step delayed update (Partial Staleness) + optional pairwise fallback을 구현한 Optimizer.
    - iteration N에서 계산된 gradient는 apply 안 하고,
      iteration N+1에서 apply하도록 지연시킵니다.
    - use_pairwise=True일 때는 hivemind의 pairwise All-Reduce 경로를 탑니다.
    """
    def __init__(self, partial_stale: bool = False, *args, **kwargs):
        # ======= MODIFIED START: extract and remove use_pairwise flag =======
        # Extract and remove pairwise fallback option from kwargs
        use_pairwise = kwargs.pop("use_pairwise", False)
        # ======= MODIFIED END: use_pairwise extraction =======

        # ======= MODIFIED START: extract required BaseCollaborativeOptimizer parameters =======
        try:
            # Extract the optimizer instance
            opt = kwargs.pop("opt")
        except KeyError:
            raise TypeError("PartialStaleCollaborativeOptimizer requires 'opt' argument")
        try:
            # Extract the DHT instance
            dht = kwargs.pop("dht")
        except KeyError:
            raise TypeError("PartialStaleCollaborativeOptimizer requires 'dht' argument")
        # ======= MODIFIED END: opt and dht extraction =======

        # Initialize base optimizer with correct positional and keyword args
        super().__init__(opt, dht=dht, use_pairwise=use_pairwise, **kwargs)

    def step(self, batch_size: int = None, **kwargs):
        """
        partial_stale=True이면, 모든 연산은 정상 수행하되
        gradient apply(opt.step())만 한 스텝 늦춥니다.
        """
        if not self.partial_stale:
            return super().step(batch_size=batch_size, **kwargs)

        # 1) apply_accumulated_grads_만 가로채서 buffer에 저장하도록 임시 교체
        orig_apply = self.apply_accumulated_grads_
        local_grads = [None]

        def store_in_buffer(scale_by=None):
            # grad accumulators를 모아 local_grads[0]에 보관
            params = [p for g in self.opt.param_groups for p in g["params"]]
            grads = []
            if self.reuse_grad_buffers:
                for p in params:
                    grads.append(None if p.grad is None else p.grad.clone())
            else:
                if self._grads is None:
                    self._grads = [torch.zeros_like(p) for p in params]
                if scale_by is not None:
                    for g in self._grads:
                        g.mul_(scale_by)
                grads = [g.clone() for g in self._grads]
            local_grads[0] = grads
            # opt.step() 호출은 생략!

        self.apply_accumulated_grads_ = store_in_buffer

        # super.step() 호출 → averaging + local progress update는 수행하지만,
        # apply_accumulated_grads_가 store_in_buffer로 대체돼 있어 opt.step() 안 됨
        super().step(batch_size=batch_size, **kwargs)

        # 원래 함수로 복원
        self.apply_accumulated_grads_ = orig_apply

        # 2) 이전 iteration buffer가 있으면 그걸 apply
        if self.stale_grad_buffer is not None:
            self._apply_stale_grad(self.stale_grad_buffer)

        # 3) 이번 iteration의 averaged gradient를 buffer에 저장
        if local_grads[0] is not None:
            self.stale_grad_buffer = local_grads[0]
        else:
            logger.debug("No grads produced this iteration (maybe no peers or skip).")

        # 실제 반환값은 필요 없으므로 None
        return

    def _apply_stale_grad(self, grad_list):
        """
        buffer에 담긴 gradient를 실제로 p.grad에 덮어쓰고 opt.step() 호출
        """
        params = [p for g in self.opt.param_groups for p in g["params"]]
        if len(params) != len(grad_list):
            logger.warning("Mismatch between params and buffered grads lengths.")

        # p.grad에 버퍼 복사
        for p, g in zip(params, grad_list):
            if g is None:
                continue
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)

        # delayed apply
        self.opt.step()

        # grad를 None으로 초기화
        for p in params:
            p.grad = None
