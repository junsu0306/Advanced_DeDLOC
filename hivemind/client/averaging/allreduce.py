import asyncio
from typing import Sequence, Set, Dict, Tuple, AsyncIterator, Any, Optional
from enum import Enum

import grpc
import torch

from hivemind.utils import Endpoint, get_logger, ChannelCache, anext
from hivemind.utils import split_for_streaming, combine_from_streaming
from hivemind.utils.compression import serialize_torch_tensor, deserialize_torch_tensor
from hivemind.proto import averaging_pb2_grpc, runtime_pb2, averaging_pb2

logger = get_logger(__name__)
GroupID = bytes

class AveragingMode(Enum):
    NODE = 0
    CLIENT = 1
    AUX = 2

class AllReduceProtocol:
    """
    Internal butterfly AllReduce with optional pairwise fallback.
    """
    def __init__(
        self,
        *,
        group_id: GroupID,
        tensors: Sequence[torch.Tensor],
        endpoint: Endpoint,
        ordered_group_endpoints: Sequence[Endpoint],
        part_sizes: Tuple[int, ...],
        return_deltas: bool = False,
        modes: Optional[Sequence[AveragingMode]] = None,
        use_pairwise: bool = False,
        **kwargs,
    ):
        assert endpoint in ordered_group_endpoints, "endpoint not in group"
        self.group_id = group_id
        self.endpoint = endpoint
        self.ordered_group_endpoints = list(ordered_group_endpoints)
        self.part_sizes = part_sizes
        self.return_deltas = return_deltas
        self.use_pairwise = use_pairwise

        # modes: determine if each peer is NODE, CLIENT, or AUX
        if modes is None:
            modes = [AveragingMode.CLIENT if size == 0 else AveragingMode.NODE for size in part_sizes]
        assert any(m != AveragingMode.CLIENT for m in modes), "Must have at least one reducer"
        self.peer_modes: Dict[Endpoint, AveragingMode] = dict(zip(self.ordered_group_endpoints, modes))

        # split tensors into parts for each endpoint
        self.local_tensor_parts: Dict[Endpoint, torch.Tensor] = dict(zip(
            self.ordered_group_endpoints,
            split_into_parts(tensors, part_sizes)
        ))
        self.tensor_shapes = tuple(t.shape for t in tensors)

        # prepare accumulator for each round
        self.accumulator = torch.zeros_like(self.local_tensor_parts[self.endpoint])
        self.denominator = 0.0
        self.accumulated_from: Set[Endpoint] = set()
        self.averaged_part: asyncio.Future[torch.Tensor] = asyncio.Future()
        self.averaged_tensor_parts: Dict[Endpoint, torch.Tensor] = {}
        self.future: asyncio.Future[Sequence[torch.Tensor]] = asyncio.Future()

        # count senders (exclude CLIENT modes)
        self._reset_senders_count()
        if self.num_senders == 0:
            # immediate result if no senders
            self.future.set_result(tensors)

    def _reset_senders_count(self):
        self.num_senders = sum(1 for ep, m in self.peer_modes.items() if m != AveragingMode.CLIENT)

    def __await__(self):
        return self.future.__await__()

    def __contains__(self, endpoint: Endpoint) -> bool:
        return endpoint in self.local_tensor_parts

    @property
    def group_size(self) -> int:
        return len(self.ordered_group_endpoints)

    async def accumulate_part(self, source: Endpoint, remote_part: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
        assert not self.averaged_part.done(), "averaged_part done"
        assert not self.future.done(), "future done"
        assert source in self.local_tensor_parts, "unexpected source"
        assert source not in self.accumulated_from, "duplicate part"
        assert self.peer_modes[self.endpoint] != AveragingMode.CLIENT, "client cannot accumulate"

        # add and count weight
        self.accumulator.add_(remote_part, alpha=weight)
        self.denominator += weight
        self.accumulated_from.add(source)

        # once all senders have contributed, finalize this part
        if len(self.accumulated_from) == self.num_senders:
            avg = self.accumulator.div_(self.denominator)
            self.averaged_part.set_result(avg)
            if self.peer_modes[self.endpoint] != AveragingMode.AUX:
                self.register_averaged_part(self.endpoint, avg)
        return await self.averaged_part

    def register_averaged_part(self, source: Endpoint, averaged_part: torch.Tensor):
        assert not self.future.done(), "future done"
        assert source not in self.averaged_tensor_parts, "duplicate registration"
        self.averaged_tensor_parts[source] = averaged_part
        if len(self.averaged_tensor_parts) == len(self.local_tensor_parts):
            # reconstruct full tensors
            parts = [self.averaged_tensor_parts[ep] for ep in self.ordered_group_endpoints]
            outputs = restore_from_parts(parts, self.tensor_shapes)
            if self.return_deltas:
                originals = [t.clone() for t in outputs]
                outputs = [o - orig for o, orig in zip(outputs, originals)]
            self.future.set_result(outputs)

    def cancel(self) -> bool:
        if not self.future.done():
            self.future.cancel()
            if not self.averaged_part.done():
                self.averaged_part.cancel()
            return True
        return False

    def set_exception(self, exception: Exception) -> bool:
        if not self.future.done():
            self.future.set_exception(exception)
            if not self.averaged_part.done():
                self.averaged_part.cancel()
            return True
        return False

    async def run(self) -> Sequence[torch.Tensor]:
        """
        Perform binary-tree pairwise reduction until a single average remains.
        """
        # prepare working map of parts
        parts = {ep: self.local_tensor_parts[ep] for ep in self.ordered_group_endpoints}
        endpoints = list(self.ordered_group_endpoints)

        # repeat until one remains
        while len(endpoints) > 1:
            tasks = []
            next_round = []
            # pair off endpoints
            for i in range(0, len(endpoints), 2):
                if i + 1 < len(endpoints):
                    ep1, ep2 = endpoints[i], endpoints[i+1]
                    # only the participant in this process runs communication
                    if self.endpoint in (ep1, ep2):
                        tasks.append(self._pairwise_reduce(ep1, ep2, parts))
                    # choose ep1 as reducer for this pair
                    next_round.append(ep1)
                else:
                    # odd node passes through
                    next_round.append(endpoints[i])
            # run all pairwise exchanges
            results = await asyncio.gather(*tasks)
            for ep, avg in results:
                parts[ep] = avg
            endpoints = next_round

        # final registration
        if endpoints and endpoints[0] == self.endpoint:
            self.register_averaged_part(self.endpoint, parts[self.endpoint])
        return await self.future

    async def _pairwise_reduce(self, ep1: Endpoint, ep2: Endpoint, parts: Dict[Endpoint, torch.Tensor]) -> Tuple[Endpoint, torch.Tensor]:
        """
        Exchange parts between ep1 and ep2 and compute their average.
        Returns the averaged tensor assigned to ep1.
        """
        # temporarily restrict parts and reset accumulator
        orig_parts = self.local_tensor_parts
        self.local_tensor_parts = {ep1: parts[ep1], ep2: parts[ep2]}
        self.accumulator = torch.zeros_like(self.local_tensor_parts[self.endpoint])
        self.denominator = 0.0
        self.accumulated_from.clear()
        self.averaged_part = asyncio.Future()
        self._reset_senders_count()

        # perform communication from both peers
        await self._communicate_with_peer(ep1)
        await self._communicate_with_peer(ep2)
        avg_tensor = await self.averaged_part

        # restore original state
        self.local_tensor_parts = orig_parts
        return self.endpoint, avg_tensor

class AllReduceRunner(AllReduceProtocol, averaging_pb2_grpc.DecentralizedAveragingServicer):
    """
    gRPC servicer implementing butterfly AllReduce with pairwise fallback.
    """
    def __init__(
        self,
        *,
        group_id: GroupID,
        tensors: Sequence[torch.Tensor],
        endpoint: Endpoint,
        ordered_group_endpoints: Sequence[Endpoint],
        compression_type: runtime_pb2.CompressionType,
        chunk_size_bytes: int,
        part_sizes: Tuple[int, ...],
        weights: Tuple[float, ...],
        gathered: Dict[Endpoint, Any],
        return_deltas: bool = False,
        use_pairwise: bool = True,
        **kwargs,
    ):
        super().__init__(
            group_id=group_id,
            tensors=tensors,
            endpoint=endpoint,
            ordered_group_endpoints=ordered_group_endpoints,
            part_sizes=part_sizes,
            return_deltas=return_deltas,
            modes=None,
            use_pairwise=use_pairwise,
        )
        self.compression_type = compression_type
        self.chunk_size_bytes = chunk_size_bytes
        self.peer_weights: Dict[Endpoint, float] = dict(zip(ordered_group_endpoints, weights))

    def _get_peer_stub(self, peer: Endpoint):
        return ChannelCache.get_stub(peer, averaging_pb2_grpc.DecentralizedAveragingStub, aio=True)

    async def _send_error_to_peer(self, peer: Endpoint, code: averaging_pb2.MessageCode):
        stub = self._get_peer_stub(peer)
        stream = stub.rpc_aggregate_part()
        await stream.write(averaging_pb2.AveragingData(group_id=self.group_id, endpoint=self.endpoint, code=code))
        await stream.done_writing()

    async def _communicate_with_peer(self, peer: Endpoint):
        mode = self.peer_modes[peer]
        if mode == AveragingMode.CLIENT:
            return
        local_part = self.local_tensor_parts[peer]
        if peer == self.endpoint:
            await self.accumulate_part(peer, local_part, weight=self.peer_weights[peer])
            return
        stub = self._get_peer_stub(peer)
        stream = stub.rpc_aggregate_part()
        data = serialize_torch_tensor(local_part, self.compression_type, allow_inplace=False)
        chunks = split_for_streaming(data, self.chunk_size_bytes)
        await stream.write(averaging_pb2.AveragingData(
            code=averaging_pb2.PART_FOR_AVERAGING,
            group_id=self.group_id,
            endpoint=self.endpoint,
            tensor_part=next(chunks),
        ))
        for c in chunks:
            await stream.write(averaging_pb2.AveragingData(tensor_part=c))
        await stream.done_writing()
        msgs = [msg async for msg in stream]
        if not msgs or msgs[0].code != averaging_pb2.AVERAGED_PART:
            await self._send_error_to_peer(peer, averaging_pb2.INTERNAL_ERROR)
            raise AllreduceException(f"peer {peer} error: {msgs}")
        combined = combine_from_streaming([m.tensor_part for m in msgs])
        remote_part = deserialize_torch_tensor(combined)
        await self.accumulate_part(peer, remote_part, weight=self.peer_weights[peer])

    async def rpc_aggregate_part(self, stream: AsyncIterator[averaging_pb2.AveragingData],
                                 context: grpc.ServicerContext) -> AsyncIterator[averaging_pb2.AveragingData]:
        try:
            req = await anext(stream)
            if req.group_id != self.group_id:
                yield averaging_pb2.AveragingData(code=averaging_pb2.BAD_GROUP_ID)
                return
            if req.code != averaging_pb2.PART_FOR_AVERAGING:
                yield averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)
                return
            chunks = [req.tensor_part] + [msg.tensor_part async for msg in stream]
            data = combine_from_streaming(chunks)
            tensor = deserialize_torch_tensor(data)
            averaged = await self.accumulate_part(req.endpoint, tensor, weight=self.peer_weights[req.endpoint])
            payload = serialize_torch_tensor(
                (averaged - tensor) if self.return_deltas else averaged,
                self.compression_type,
                allow_inplace=False
            )
            out_chunks = split_for_streaming(payload, self.chunk_size_bytes)
            yield averaging_pb2.AveragingData(code=averaging_pb2.AVERAGED_PART, tensor_part=next(out_chunks))
            for c in out_chunks:
                yield averaging_pb2.AveragingData(tensor_part=c)
        except Exception as e:
            logger.debug(f"rpc_aggregate_part exception: {e}")
            yield averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)

class AllreduceException(Exception):
    pass

# utilities

def split_into_parts(tensors: Sequence[torch.Tensor], part_sizes: Tuple[int, ...]) -> Tuple[torch.Tensor, ...]:
    flat = torch.cat([t.flatten() for t in tensors])
    return torch.split_with_sizes(flat, part_sizes)


def restore_from_parts(chunks: Sequence[torch.Tensor], shapes: Sequence[torch.Size]) -> Tuple[torch.Tensor, ...]:
    flat = torch.cat(chunks)
    sizes = [s.numel() for s in shapes]
    parts = torch.split_with_sizes(flat, sizes)
    return tuple(p.reshape(shape) for p, shape in zip(parts, shapes))
