from itertools import chain
from threading import Lock
from typing import Sequence, Dict, Iterator, Optional, Any, Tuple
from contextlib import nullcontext

import torch

from hivemind.client.averaging import DecentralizedAverager
from hivemind.client.averaging.group_info import GroupInfo
from hivemind.utils import nested_flatten, nested_pack, get_logger, run_in_background

logger = get_logger(__name__)


class TrainingAverager(DecentralizedAverager):
    """
    A high-level interface to DecentralizedAverager that averages trainable params or gradients for an optimizer.

    This averager implements a number of typical use cases that arise in collaborative optimization
    - averaging parameters or gradients or both
    - this peer's weight (e.g. based on its batch size) can be specified via averager.step(weight=...)
    - when out of sync, the averager will load the entire optimizer state from an up-to-date peer
    """

    def __init__(
        self,
        opt: torch.optim.Optimizer,
        *,
        average_parameters: bool,
        average_gradients: bool,
        average_opt_statistics: Sequence[str] = (),
        extra_tensors: Sequence[torch.Tensor] = (),
        initialize_optimizer: bool = True,
        use_pairwise: bool = False,
        **kwargs: Any,
    ):
        self.opt = opt
        self.extra_tensors = tuple(extra_tensors)
        self.local_step = 0
        self.opt_statistics = tuple(average_opt_statistics)
        self.average_parameters = average_parameters
        self.average_gradients = average_gradients
        self.lock_averager_step = Lock()

        # Initialize optimizer state if needed
        if initialize_optimizer:
            self._initialize_optimizer_state(opt)

        # Prepare tensors to average
        with torch.no_grad():
            averaged_tensors = [t.detach().cpu().float().clone() for t in self.local_tensors()]

        # Pass use_pairwise into DecentralizedAverager
        super().__init__(
            averaged_tensors=averaged_tensors,
            use_pairwise=use_pairwise,
            **kwargs,
        )

    @torch.no_grad()
    def step(
        self,
        data_lock: Optional[Lock] = None,
        wait: bool = True,
        **kwargs: Any,
    ) -> Optional[Sequence[torch.Tensor]]:
        """ Average optimizer weights and gradients with peers. """
        if not wait:
            return run_in_background(self.step, data_lock, wait=True, **kwargs)

        use_old_local = data_lock is not None
        if data_lock is None:
            data_lock = nullcontext()

        local_tensors = list(self.local_tensors())
        with self.lock_averager_step:
            # Copy local state into averager buffers
            with data_lock, self.get_tensors() as averaged_tensors:
                if use_old_local:
                    old_local = tuple(x.cpu().float().clone() for x in local_tensors)
                assert len(local_tensors) == len(averaged_tensors)
                for av, lt in zip(averaged_tensors, local_tensors):
                    av[...] = lt.cpu().float()

            # Run decentralized averaging
            gathered = super().step(**kwargs)

            # Merge back averaged results into local model/optimizer
            if gathered is not None:
                with data_lock, self.get_tensors() as averaged_tensors:
                    if use_old_local:
                        for av, lt, old in zip(averaged_tensors, local_tensors, old_local):
                            lt[...] += av.to(dtype=lt.dtype, device=lt.device) - old.to(dtype=lt.dtype, device=lt.device)
                    else:
                        for av, lt in zip(averaged_tensors, local_tensors):
                            lt[...] = av.to(dtype=lt.dtype, device=lt.device)

            self.local_step += 1
            return gathered

    def local_tensors(self, replace_none: bool = True) -> Iterator[torch.Tensor]:
        if self.average_parameters:
            for pg in self.opt.param_groups:
                yield from pg['params']
        if self.average_gradients:
            for pg in self.opt.param_groups:
                for p in pg['params']:
                    if p.grad is not None:
                        yield p.grad
                    elif replace_none:
                        yield torch.zeros_like(p)
        for stat in self.opt_statistics:
            for pg in self.opt.param_groups:
                for p in pg['params']:
                    yield self.opt.state[p][stat]
        yield from iter(self.extra_tensors)

    def get_current_state(self) -> Tuple[Dict, list]:
        with torch.no_grad():
            params = tuple(p.detach().cpu() for pg in self.opt.param_groups for p in pg['params'])
            extras = tuple(t.detach().cpu() for t in self.extra_tensors)
            meta, tensors = self._dump_optimizer_state(self.opt)
        metadata = dict(step=self.local_step, group_bits=self.get_group_bits(), optimizer_metadata=meta)
        return metadata, list(chain(params, extras, tensors))

    def load_state_from_peers(self, **kwargs: Any) -> None:
        parameters = [p for pg in self.opt.param_groups for p in pg['params']]
        parameters.extend(self.extra_tensors)
        num_params = len(parameters)

        loaded = super().load_state_from_peers(**kwargs)
        if loaded is None:
            return
        metadata, flat = loaded
        loaded_params = flat[:num_params]
        loaded_tensors = flat[num_params:]

        with torch.no_grad():
            for lp, p in zip(loaded_params, parameters):
                p[...] = lp
            self._load_optimizer_state(self.opt, metadata['optimizer_metadata'], loaded_tensors)

        self.local_step = max(self.local_step, metadata['step'])

    # ──────────────── 아래에 세 개의 static 메서드를 추가 ────────────────

    @staticmethod
    def _initialize_optimizer_state(opt: torch.optim.Optimizer) -> None:
        """Dummy backward + step to initialize optimizer internal buffers."""
        for pg in opt.param_groups:
            for p in pg['params']:
                if p.grad is None:
                    (p * 0).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    @staticmethod
    def _dump_optimizer_state(opt: torch.optim.Optimizer) -> Tuple[list, list]:
        """Flatten optimizer.state_dict() into (meta, tensors) for averaging."""
        flat_meta, flat_tensors = [], []
        with torch.no_grad():
            for item in nested_flatten(opt.state_dict()):
                if isinstance(item, torch.Tensor):
                    flat_meta.append({"type": "tensor", "index": len(flat_tensors)})
                    flat_tensors.append(item.cpu())
                else:
                    flat_meta.append({"type": "value", "value": item})
        return flat_meta, flat_tensors

    @staticmethod
    def _load_optimizer_state(
        optimizer: torch.optim.Optimizer,
        flat_meta: list,
        flat_tensors: Sequence[torch.Tensor]
    ) -> None:
        """Restore optimizer.state_dict() from flattened (meta, tensors)."""
        rebuilt = []
        for rec in flat_meta:
            if rec.get("type") == "tensor":
                rebuilt.append(flat_tensors[rec["index"]])
            else:
                rebuilt.append(rec["value"])
        with torch.no_grad():
            optimizer.load_state_dict(nested_pack(rebuilt, optimizer.state_dict()))
