""" An extension of averager that supports common optimization use cases. """
from itertools import chain
from threading import Lock
from typing import Sequence, Dict, Iterator, Optional, Any, Tuple
from contextlib import nullcontext

import torch

from hivemind.client.averaging import DecentralizedAverager
from hivemind.utils import nested_flatten, nested_pack, get_logger, run_in_background

logger = get_logger(__name__)


class TrainingAverager(DecentralizedAverager):
    """
    A high-level interface to DecentralizedAverager that averages trainable params or gradients for an optimizer.

    This averager implements a number of typical use cases that arise in collaborative optimization
    - averaging parameters or gradients or both (in future, this will support averaging learning rates as well)
    - this peer's weight (e.g. based on its batch size) can be specified via averager.step(weight=...)
    - when out of sync, the averager will load the entire optimizer state from an up-to-date peer

    :param opt: a pytorch optimizer to be averaged between peers (complete with model parameters)
    :param average_parameters: whether or not to average model parameters in self.step(...)
    :param average_gradients: whether or not to average model gradients in self.step(...)
    :param average_opt_statistics: if specified, average optimizer statistics with corresponding names in statedict
    :param initialize_optimizer: if True, this will run a speculative optimizer step with
      zero gradients to initialize all tensors. If False, please initialize the optimizer state manually.
    :param extra_tensors: if specified, these extra tensors will also be averaged and shared in load_state_from_peers.
    :note: you can use extra_tensors for averaging tensors that are updated outside of opt.step (e.g. batchnorm stats)
    :param kwargs: any additional parameters will be forwarded to DecentralizedAverager
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
            from hivemind.optim.base import initialize_optimizer_state
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

            # Run decentralized averaging (paired or full, depending on use_pairwise)
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
        """
        Iterate local trainer's tensors that should be averaged with peers

        :param replace_none: if True and average_gradients is True, None grads will be replaced with a zero tensors
          Otherwise, such gradients will be skipped. (this may cause inconsistencies with averaged_tensors)
        """
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
        """
        Get current model/optimizer state and when requested by a newbie peer. executed in the host process.
        :returns: a tuple of (serializable_small_metadata, sequence of torch tensors)
        """
        with torch.no_grad():
            params = tuple(p.detach().cpu() for pg in self.opt.param_groups for p in pg['params'])
            extras = tuple(t.detach().cpu() for t in self.extra_tensors)
            meta, tensors = self._dump_optimizer_state(self.opt)
        metadata = dict(step=self.local_step, group_bits=self.get_group_bits(), optimizer_metadata=meta)
        return metadata, list(chain(params, extras, tensors))
    
    def load_state_from_peers(self, **kwargs: Any) -> None:
        """
        Attempt to download the latest optimizer state from peers and update trainer parameters/statistics.
        :returns: whether or the averager succeeded in loading parameters
        """
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

@staticmethod
def initialize_optimizer_state(opt: torch.optim.Optimizer) -> None:
        # Run a dummy backward+step to initialize optimizer buffers
        for pg in opt.param_groups:
            for p in pg['params']:
                if p.grad is None:
                    (p * 0).sum().backward()
        opt.step()

@staticmethod
def dump_optimizer_state(opt: torch.optim.Optimizer) -> Tuple[list, list]:
    """ Convert optimizer state into a format of DecentralizedAverager's get_current_state/load_state_from_peers """
    with torch.no_grad():
        md, ts = [], []
        for elem in nested_flatten(opt.state_dict()):
            if isinstance(elem, torch.Tensor):
                md.append({'type': 'tensor', 'index': len(ts)})
                ts.append(elem.cpu())
            else:
                md.append({'type': 'value', 'value': elem})
        return md, ts

@staticmethod
def load_optimizer_state(
        optimizer: torch.optim.Optimizer,
        flat_metadata: Dict,
        flat_tensors: Sequence[torch.Tensor]
    ) -> Any:
        """
        Copy-pasted original implementation of load_optimizer_state for full fidelity:
        """
        flat_optimizer_state = []
        for elem in flat_metadata:
            if elem.get('type') == 'tensor' and isinstance(elem.get('index'), int):
                flat_optimizer_state.append(flat_tensors[elem['index']])
            elif elem.get('type') == 'value' and 'value' in elem:
                flat_optimizer_state.append(elem['value'])
        with torch.no_grad():
            try:
                return optimizer.load_state_dict(
                    nested_pack(flat_optimizer_state, structure=optimizer.state_dict())
                )
            except StopIteration:
                return optimizer
