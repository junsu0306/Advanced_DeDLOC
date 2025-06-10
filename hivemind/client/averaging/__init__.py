from __future__ import annotations

import asyncio
import contextlib
import ctypes
import multiprocessing as mp
import os
import threading
import uuid
import weakref
import time
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import asdict
from typing import Sequence, Optional, Tuple, Any, Dict, AsyncIterator

import grpc
from grpc._cython.cygrpc import InternalError
import torch
import numpy as np

from hivemind.dht import DHT, DHTID, get_dht_time
from hivemind.client.averaging.allreduce import AllReduceRunner, AllreduceException, GroupID, AveragingMode
from hivemind.client.averaging.load_balancing import load_balance_peers
from hivemind.client.averaging.matchmaking import Matchmaking, MatchmakingException
from hivemind.client.averaging.group_info import GroupInfo, PeerInfo
from hivemind.proto import averaging_pb2, averaging_pb2_grpc, runtime_pb2
from hivemind.proto.averaging_pb2 import BandwidthProbeRequest, BandwidthProbeResponse
from hivemind.utils.grpc import ChannelCache, GRPC_KEEPALIVE_OPTIONS, split_for_streaming, combine_from_streaming
from hivemind.utils.compression import serialize_torch_tensor, deserialize_torch_tensor
from hivemind.utils.asyncio import anext, achain, aiter, switch_to_uvloop
from hivemind.utils.timed_storage import ValueWithExpiration, DHTExpiration
from hivemind.utils.serializer import MSGPackSerializer, SerializerBase
from hivemind.utils import Endpoint, Port, MPFuture, get_logger, TensorDescriptor

# flavour types
data_for_gather = Any
logger = get_logger(__name__)
DEFAULT_CHUNK_SIZE_BYTES = 2 ** 16


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)


class DecentralizedAverager(
        mp.Process,
        averaging_pb2_grpc.DecentralizedAveragingServicer,
        averaging_pb2_grpc.BandwidthProbeServicer):
    """
    Adaptive pair-wise allreduce with on-start bandwidth probing via rpc_ping.
    """
    serializer = MSGPackSerializer

    def __init__(
        self,
        averaged_tensors: Sequence[torch.Tensor],
        dht: DHT,
        *,
        start: bool,
        use_pairwise: bool = False,
        prefix: str,
        target_group_size: int,
        min_group_size: int = 2,
        initial_group_bits: Optional[str] = None,
        averaging_expiration: float = 15,
        request_timeout: float = 3,
        chunk_size_bytes: int = 2 ** 16,
        allreduce_timeout: Optional[float] = None,
        averaging_alpha: float = 1.0,
        compression_type: runtime_pb2.CompressionType = runtime_pb2.CompressionType.NONE,
        throughput: Optional[float] = None,
        min_vector_size: int = 0,
        auxiliary: bool = False,
        allow_state_sharing: Optional[bool] = None,
        listen: bool = True,
        listen_on: Endpoint = '0.0.0.0:*',
        daemon: bool = True,
        channel_options: Optional[Sequence[Tuple[str, Any]]] = None,
        probe_size: int = 512 * 1024,
        **kwargs,
    ):
        super().__init__()
        self.dht = dht
        self.listen, self.listen_on, self.kwargs = listen, listen_on, kwargs
        self.probe_size = probe_size
        self.prefix = prefix
        self._throughput = throughput or 0.0

        # Averaging parameters
        self.use_pairwise = use_pairwise
        self.matchmaking_kwargs = dict(
            prefix=prefix,
            initial_group_bits=initial_group_bits,
            target_group_size=target_group_size,
            min_group_size=min_group_size,
            averaging_expiration=averaging_expiration,
            request_timeout=request_timeout,
        )
        self.allreduce_kwargs = dict(
            compression_type=compression_type,
            chunk_size_bytes=chunk_size_bytes,
            min_vector_size=min_vector_size,
            use_pairwise=self.use_pairwise,
        )
        self._averaging_alpha, self._allreduce_timeout = averaging_alpha, allreduce_timeout

        # Process state
        self.daemon = daemon
        self._running_groups: Dict[GroupID, AllReduceRunner] = {}
        self._pipe, self.pipe = mp.Pipe(duplex=True)
        self._port = mp.Value(ctypes.c_uint32, 0)
        self._allow_state_sharing = mp.Value(ctypes.c_bool, 0)
        self.allow_state_sharing = (listen and not auxiliary) if allow_state_sharing is None else allow_state_sharing
        self._averager_endpoint: Optional[Endpoint] = None
        if not listen:
            self._averager_endpoint = f'client::{uuid.uuid4()}'

        # Shared tensors
        self._averaged_tensors = tuple(averaged_tensors)
        self.lock_averaged_tensors = mp.Lock()
        self.last_updated: DHTExpiration = -float('inf')
        for tensor in self._averaged_tensors:
            tensor.share_memory_()
        self.total_size = sum(map(torch.Tensor.numel, self._averaged_tensors))
        self.schema_hash = compute_schema_hash(self._averaged_tensors)

        # Start background fetcher and optionally run
        self.ready = mp.Event()
        background_fetcher = threading.Thread(
            daemon=True,
            target=_background_thread_fetch_current_state,
            args=[self.serializer, self.pipe, weakref.WeakMethod(self.get_current_state)],
        )
        background_fetcher.start()
        if start:
            self.run_in_background(await_ready=True)

    # ----------------
    # Bandwidth probe
    # ----------------
    def _perform_bandwidth_probe(self) -> float:
        payload = b'\0' * self.probe_size
        target = self.endpoint or self.listen_on
        channel = grpc.insecure_channel(target)
        stub = averaging_pb2_grpc.BandwidthProbeStub(channel)
        start = time.time()
        resp = stub.rpc_ping(BandwidthProbeRequest(payload=payload))
        rtt = time.time() - start
        mbits = (self.probe_size * 8) / 1e6
        mbps = (mbits * 2) / max(rtt, 1e-6)
        channel.close()
        return mbps

    async def rpc_ping(self, request: BandwidthProbeRequest, context: grpc.ServicerContext) -> BandwidthProbeResponse:
        return BandwidthProbeResponse(payload=request.payload)

    async def _store_bandwidth(self, mbps: float):
        key = f"{self.prefix}_bandwidth"
        try:
            await self.dht.store(key, value=mbps,
                                  expiration_time=get_dht_time()+600,
                                  return_future=True)
        except Exception:
            logger.warning(f"Failed to store bandwidth {mbps} Mbps to DHT key {key}")

    # ----------------
    # Server lifecycle
    # ----------------
    def run(self):
        thread = threading.Thread(target=self._run_internal, daemon=True)
        thread.start()
        thread.join()

    def _run_internal(self):
        loop = switch_to_uvloop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            async def _run():
                grpc.aio.init_grpc_aio()
                if self.listen:
                    server = grpc.aio.server(**self.kwargs, options=GRPC_KEEPALIVE_OPTIONS)
                    averaging_pb2_grpc.add_DecentralizedAveragingServicer_to_server(self, server)
                    averaging_pb2_grpc.add_BandwidthProbeServicer_to_server(self, server)
                    found = server.add_insecure_port(self.listen_on)
                    assert found != 0, f"Failed to bind {self.listen_on}"
                    self._port.value = found
                    await server.start()
                    mbps = await loop.run_in_executor(pool, self._perform_bandwidth_probe)
                    asyncio.create_task(self._store_bandwidth(mbps))
                    self._throughput = mbps
                else:
                    logger.debug("Averager running in client mode; skipping ping server.")

                self._matchmaking = Matchmaking(self.endpoint, self.schema_hash, self.dht,
                                                **self.matchmaking_kwargs,
                                                client_mode=not self.listen)
                if self.listen:
                    asyncio.create_task(self._declare_for_download_periodically())

                self._pending_group_assembled = asyncio.Event()
                self._pending_group_assembled.set()
                self.ready.set()

                while True:
                    method, args, kwargs = await loop.run_in_executor(pool, self._pipe.recv)
                    asyncio.create_task(getattr(self, method)(*args, **kwargs))

            loop.run_until_complete(_run())

    def run_in_background(self, await_ready=True, timeout=None):
        self.start()
        if await_ready and not self.ready.wait(timeout=timeout):
            raise TimeoutError(f"Server didn't notify .ready in {timeout} seconds")

    def shutdown(self) -> None:
        if self._parent_pid != os.getpid() or self.is_alive():
            self._pipe.send(('_SHUTDOWN', None))
            self.terminate()
        else:
            logger.warning("DHT shutdown has no effect: the process is not alive")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.endpoint})"

    # ----------------
    # Averaging step
    # ----------------
    async def _make_allreduce_runner(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> AllReduceRunner:
        try:
            # deserialize gathered blobs
            weights, throughputs, mode_ids, user_blobs = zip(
                *map(self.serializer.loads, group_info.gathered)
            )
            user_gathered = dict(
                zip(group_info.endpoints, map(self.serializer.loads, user_blobs))
            )

            # obtain PeerInfo list for LP
            peer_infos: Sequence[PeerInfo] = group_info.peer_infos()

            # solve LP for optimal part sizes
            part_sizes = await asyncio.get_event_loop().run_in_executor(
                None, load_balance_peers, self.total_size, peer_infos, min_vector_size
            )

            async with self.get_tensors_async() as averaged_tensors:
                return AllReduceRunner(
                    group_id=group_info.group_id,
                    tensors=averaged_tensors,
                    endpoint=self.endpoint,
                    ordered_group_endpoints=group_info.endpoints,
                    part_sizes=part_sizes,
                    weights=weights,
                    gathered=user_gathered,
                    return_deltas=True,
                    modes=tuple(map(AveragingMode, mode_ids)),
                    **kwargs
                )
        except Exception as e:
            raise MatchmakingException(f"Unable to create allreduce runner ({e})")

    @property
    def port(self) -> Optional[Port]:
        return self._port.value if self._port.value != 0 else None

    @property
    def allow_state_sharing(self) -> bool:
        return bool(self._allow_state_sharing.value)

    @allow_state_sharing.setter
    def allow_state_sharing(self, value: bool):
        if value is True and not self.listen:
            logger.warning("Cannot allow state sharing: averager in client mode cannot share state.")
        else:
            self._allow_state_sharing.value = value

    @property
    def endpoint(self) -> Optional[Endpoint]:
        if self.listen and self._averager_endpoint is None:
            assert self.port is not None, "Averager is not running yet"
            self._averager_endpoint = f"{self.dht.get_visible_address()}:{self.port}"
            logger.debug(f"Assuming averager endpoint to be {self._averager_endpoint}")
        return self._averager_endpoint

    def __repr__(self):
        return f"{self.__class__.__name__}({self.endpoint})"

    def run(self):
        """
        Run averager function in a background thread; this is needed to avoid a heisenbug with broken OMP on fork
        Turns out, using a non-main thread creates a separate OMP pool that works even if the original pool is corrupted
        Read more: https://github.com/pytorch/pytorch/issues/17199
        """
        thread = threading.Thread(target=self._run_internal, daemon=True)
        thread.start()
        thread.join()

    def _run_internal(self):
        """ Serve DecentralizedAverager forever. This function will not return until the averager is shut down """
        loop = switch_to_uvloop()
        # initialize asyncio synchronization primitives in this event loop
        with ThreadPoolExecutor(max_workers=1) as pipe_awaiter:
            async def _run():
                grpc.aio.init_grpc_aio()

                if self.listen:
                    server = grpc.aio.server(**self.kwargs, options=GRPC_KEEPALIVE_OPTIONS)
                    averaging_pb2_grpc.add_DecentralizedAveragingServicer_to_server(self, server)
                    found_port = server.add_insecure_port(self.listen_on)
                    assert found_port != 0, f"Failed to listen to {self.listen_on}"
                    self._port.value = found_port
                    await server.start()
                else:
                    logger.debug(f"The averager is running in client mode.")

                self._matchmaking = Matchmaking(self.endpoint, self.schema_hash, self.dht, **self.matchmaking_kwargs,
                                                client_mode=not self.listen)
                if self.listen:
                    asyncio.create_task(self._declare_for_download_periodically())

                self._pending_group_assembled = asyncio.Event()
                self._pending_group_assembled.set()
                self.ready.set()

                while True:
                    method, args, kwargs = await loop.run_in_executor(pipe_awaiter, self._pipe.recv)
                    asyncio.create_task(getattr(self, method)(*args, **kwargs))

            loop.run_until_complete(_run())

    def run_in_background(self, await_ready=True, timeout=None):
        """
        Starts averager in a background process. if await_ready, this method will wait until background dht
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready and not self.ready.wait(timeout=timeout):
            raise TimeoutError(f"Server didn't notify .ready in {timeout} seconds")

    def shutdown(self) -> None:
        """ Shut down the averager process """
        # TODO notify peers before terminating
        if self._parent_pid != os.getpid() or self.is_alive():
            self._pipe.send(('_SHUTDOWN', None))
            self.terminate()
        else:
            logger.warning("DHT shutdown has no effect: the process is not alive")

    def __del__(self):
        if self._parent_pid != os.getpid() or self.is_alive():
            self.shutdown()

    def step(self, gather: Optional[DataForGather] = None, weight: float = 1.0, timeout: Optional[float] = None,
             allow_retries: bool = True, wait: bool = True) -> Union[Optional[Dict[Endpoint, DataForGather]], MPFuture]:
        """
        Set up the averager to look for a group and run one round of averaging, return True on success, False on failure

        :param gather: optionally send this informaton to all peers in the next group and gather it from every groupmate
          (this operation is known as all-gather). The gathered data will be available as the output of this function.
        :param weight: averaging weight for this peer, int or float, must be strictly positive
        :param allow_retries: if averager fails to run one round of allreduce, this option will allow it to try again
          within the specified timeout
        :param timeout: if averager was unable to *find* a group in this many seconds, consider allreduce failedK
        :param wait: if True (default), return when finished. Otherwise return MPFuture and run in background.
        :returns: on success, update averaged_tensors and return group info; on failure, return None
        """
        if self.mode == AveragingMode.AUX and weight != 1:
            logger.warning("Averager is running in auxiliary mode, weight is unused.")
        else:
            assert isinstance(weight, (int, float)) and weight > 0, f"Expected a positive int/float, got {type(weight)}"

        future, _future = MPFuture.make_pair()
        gather_binary = self.serializer.dumps(gather)  # serialize here to avoid loading modules in the averager process
        self.pipe.send(('_step', [], dict(future=_future, gather_binary=gather_binary, weight=weight,
                                          allow_retries=allow_retries, timeout=timeout)))
        return future.result() if wait else future

    async def _step(self, *, future: MPFuture, gather_binary: bytes, weight: float,
                    allow_retries: bool, timeout: Optional[float]):
        loop = asyncio.get_event_loop()
        start_time = get_dht_time()
        group_id = None

        try:
            while not future.done():
                try:
                    self._pending_group_assembled.clear()
                    data_for_gather = self.serializer.dumps([weight, self._throughput, self.mode.value, gather_binary]) 
                    group_info = await self._matchmaking.look_for_group(timeout=timeout,
                                                                        data_for_gather=data_for_gather)
                    if group_info is None:
                        raise AllreduceException("Averaging step failed: could not find a group.")
                    group_id = group_info.group_id
                    allreduce_runner = await self._make_allreduce_runner(group_info, **self.allreduce_kwargs)
                    self._running_groups[group_id] = allreduce_runner
                    self._pending_group_assembled.set()
                    await asyncio.wait_for(allreduce_runner.run(), self._allreduce_timeout)
                    if self.mode != AveragingMode.AUX:
                        await loop.run_in_executor(None, self.update_tensors, allreduce_runner)

                    # averaging is finished, exit the loop
                    future.set_result(allreduce_runner.gathered)

                except (AllreduceException, MatchmakingException, AssertionError, StopAsyncIteration, InternalError,
                        asyncio.CancelledError, asyncio.InvalidStateError, grpc.RpcError, grpc.aio.AioRpcError) as e:
                    time_elapsed = get_dht_time() - start_time
                    if not allow_retries or (timeout is not None and timeout < time_elapsed):
                        logger.exception(f"Averager caught {repr(e)}")
                        future.set_exception(e)
                    else:
                        logger.warning(f"Averager caught {repr(e)}, retrying")

                finally:
                    _ = self._running_groups.pop(group_id, None)
                    self._pending_group_assembled.set()

        except BaseException as e:
            if not future.done():
                future.set_exception(e)
            raise
        finally:
            if not future.done():
                future.set_exception(RuntimeError("Internal sanity check failed: averager.step left future pending."
                                                  " Please report this to hivemind issues."))

    async def _make_allreduce_runner(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> AllReduceRunner:
        """ Use a group description found by Matchmaking to form AllreduceRunner """
        try:
            weights, throughputs, mode_ids, user_gathered = zip(*map(self.serializer.loads, group_info.gathered))
            user_gathered = dict(zip(group_info.endpoints, map(self.serializer.loads, user_gathered)))
            # compute optimal part sizes from peer throughputs
            modes = tuple(map(AveragingMode, mode_ids))
            incoming_throughputs = [thr if mode != AveragingMode.CLIENT else 0.0 for thr, mode in zip(throughputs, modes)]  # TODO: replace with proper load balancing
            part_sizes = await asyncio.get_event_loop().run_in_executor(
                None, load_balance_peers, self.total_size, incoming_throughputs, min_vector_size)
            async with self.get_tensors_async() as averaged_tensors:
                return AllReduceRunner(group_id=group_info.group_id, tensors=averaged_tensors, endpoint=self.endpoint,
                                       ordered_group_endpoints=group_info.endpoints, part_sizes=part_sizes,
                                       weights=weights, gathered=user_gathered, return_deltas=True, modes=modes, **kwargs)
        except Exception as e:
            raise MatchmakingException(f"Unable to create allreduce runner ({e}), group_info: {weights, throughputs, modes, user_gathered}")

    def update_tensors(self, allreduce_group: AllReduceRunner):
        """
        a private (extendable) method that applies changes from a finished allreduce to local tensors
        """
        assert allreduce_group.return_deltas and allreduce_group.future.done()
        averaging_deltas = allreduce_group.future.result()

        with torch.no_grad(), self.get_tensors() as local_tensors:
            assert len(local_tensors) == len(self._averaged_tensors)
            for tensor, update in zip(local_tensors, averaging_deltas):
                tensor.add_(update, alpha=self._averaging_alpha)
        self.last_updated = get_dht_time()

    @contextlib.contextmanager
    def get_tensors(self) -> Sequence[torch.Tensor]:
        """
        A contextmanager that gives user access to averaged tensors.
        It is guaranteed that the averager will not modify tensors while this context is active.
        Please do not modify the yielded tensors in-place after the context is released.
        """
        with self.lock_averaged_tensors:
            yield self._averaged_tensors
        self.last_updated = get_dht_time()

    @contextlib.asynccontextmanager
    async def get_tensors_async(self) -> Sequence[torch.Tensor]:
        """ Like get_tensors, but uses an asynchronous contextmanager """
        try:
            await asyncio.get_event_loop().run_in_executor(None, self.lock_averaged_tensors.acquire)
            yield self._averaged_tensors
        finally:
            self.lock_averaged_tensors.release()

    async def rpc_join_group(self, request: averaging_pb2.JoinRequest, context: grpc.ServicerContext
                             ) -> AsyncIterator[averaging_pb2.MessageFromLeader]:
        """ accept or reject a join request from another averager; if accepted, run him through allreduce steps """
        async for response in self._matchmaking.rpc_join_group(request, context):
            yield response

    async def rpc_aggregate_part(self, stream: AsyncIterator[averaging_pb2.AveragingData], context: grpc.ServicerContext
                                 ) -> AsyncIterator[averaging_pb2.AveragingData]:
        """ a groupmate sends us a part of his tensor; we should average it with other peers and return the result """
        request = await anext(stream)
        if request.group_id not in self._running_groups:
            # this handles a special case when leader accepted us to group AND began allreduce right away,
            # but his response with group_id was delayed and other peers got to us first
            await self._pending_group_assembled.wait()

        group = self._running_groups.get(request.group_id)
        if group is None:
            yield averaging_pb2.AveragingData(code=averaging_pb2.BAD_GROUP_ID)
            return

        async for message in group.rpc_aggregate_part(achain(aiter(request), stream), context):
            yield message

    async def _declare_for_download_periodically(self):
        download_key = f'{self._matchmaking.group_key_manager.prefix}.all_averagers'
        while True:
            if self.allow_state_sharing:
                asyncio.create_task(asyncio.wait_for(self.dht.store(
                    download_key, subkey=self.endpoint, value=self.last_updated,
                    expiration_time=get_dht_time() + self._matchmaking.averaging_expiration, return_future=True),
                    timeout=self._matchmaking.averaging_expiration))
            await asyncio.sleep(self._matchmaking.averaging_expiration)

    async def rpc_download_state(self, request: averaging_pb2.DownloadRequest, context: grpc.ServicerContext
                                 ) -> AsyncIterator[averaging_pb2.DownloadData]:
        """
        Get the up-to-date trainer state from a peer.
        The state consists of two parts: (serialized_metadata, tensors)

         - serialized_metadata is a small serialized bytestring meant to store scalars and hyperparameters
         - tensors is a sequence of pytorch tensors that represent model parameters or optimizer statistics
        """
        if not self.allow_state_sharing:
            return  # deny request and direct peer to the next prospective averager
        chunk_size_bytes = self.matchmaking_kwargs.get('chunk_size_bytes', DEFAULT_CHUNK_SIZE_BYTES)
        metadata, tensors = await self._get_current_state_from_host_process()

        for tensor in tensors:
            for part in split_for_streaming(serialize_torch_tensor(tensor), chunk_size_bytes):
                if metadata is not None:
                    yield averaging_pb2.DownloadData(tensor_part=part, metadata=metadata)
                    metadata = None
                else:
                    yield averaging_pb2.DownloadData(tensor_part=part)

    def get_current_state(self) -> Tuple[Any, Sequence[torch.Tensor]]:
        """
        Get current state and send it to a peer. executed in the host process. Meant to be overriden.
        :returns: a tuple of (small metadata, sequence of torch tensors)
        :note: metadata must be seriablizable with self.serializer (default = MSGPackSerializer)
        """
        with self.get_tensors() as tensors:
            return dict(group_key=self.get_group_bits()), tensors

    async def _get_current_state_from_host_process(self):
        """ Executed in the averager process inside rpc_download_state """
        future, _future = MPFuture.make_pair()
        self._pipe.send(('_TRIGGER_GET_CURRENT_STATE', _future))
        return await future

    def load_state_from_peers(self, wait=True) -> Optional[Tuple[Any, Sequence[torch.Tensor]]]:
        """
        Try to download the latest optimizer state one of the existing peer.
        :returns: on success, return a 2-tuple with (metadata, tensors), where

        - metadata is a small object containing metadata (e.g. hyperparameters, scalars, etc)
        - tensors is a sequence of pytorch tensors meant to contain peer's model weights and optimizer statistics

        The exact contents of both metadata and tensors are determined by get_current_state method
        """
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_load_state_from_peers', [], dict(future=_future)))
        return future.result() if wait else future

    async def _load_state_from_peers(self, future: MPFuture):
        try:
            key_manager = self._matchmaking.group_key_manager
            peer_priority, _ = self.dht.get(f"{key_manager.prefix}.all_averagers", latest=True) or ({}, None)
            peer_priority = {peer: float(info.value) for peer, info in peer_priority.items()
                             if isinstance(info, ValueWithExpiration) and isinstance(info.value, (float, int))}

            if not isinstance(peer_priority, dict) or len(peer_priority) == 0:
                logger.info(f"Averager could not load state from peers: peer dict empty or corrupted {peer_priority}.")
                future.set_result(None)
                return

            metadata = None
            for peer in sorted(peer_priority.keys(), key=peer_priority.get, reverse=True):
                if peer != self.endpoint:
                    logger.info(f"Downloading parameters from peer {peer}")
                    stream = None
                    try:
                        stub = ChannelCache.get_stub(peer, averaging_pb2_grpc.DecentralizedAveragingStub, aio=True)
                        stream = stub.rpc_download_state(averaging_pb2.DownloadRequest())
                        current_tensor_parts, tensors = [], []
                        async for message in stream:
                            if message.metadata:
                                metadata = self.serializer.loads(message.metadata)
                            if message.tensor_part.dtype and current_tensor_parts:
                                # tensor_part.dtype indicates the start of the new tensor, so we should wrap up this one
                                tensors.append(deserialize_torch_tensor(combine_from_streaming(current_tensor_parts)))
                                current_tensor_parts = []
                            current_tensor_parts.append(message.tensor_part)
                        if current_tensor_parts:
                            tensors.append(deserialize_torch_tensor(combine_from_streaming(current_tensor_parts)))

                        if not metadata:
                            logger.debug(f"Peer {peer} did not send its state.")
                            continue

                        logger.info(f"Finished downloading state from {peer}")
                        future.set_result((metadata, tensors))
                        self.last_updated = get_dht_time()
                        return
                    except BaseException as e:
                        logger.exception(f"Failed to download state from {peer} - {repr(e)}")
                    finally:
                        if stream is not None:
                            await stream.code()

        finally:
            if not future.done():
                logger.warning("Averager could not load state from peers: all requests have failed.")
                future.set_result(None)

    def get_group_bits(self, wait: bool = True):
        """
        :param wait: if True, return bits immediately. Otherwise return awaitable MPFuture
        :returns: averager's current group key bits (without prefix)
        """
        future, _future = MPFuture.make_pair()
        self.pipe.send(('_get_group_bits', [], dict(future=_future)))
        return future.result() if wait else future

    async def _get_group_bits(self, future: MPFuture):
        future.set_result(self._matchmaking.group_key_manager.group_bits)

    def set_group_bits(self, group_bits: str, wait: bool = True):
        """
        :param group_bits: group bits (string of '0' or '1') to be used in averager's group key
        :param wait: if True, wait until the update is confirmed by the averager. Otherwise return immediately
        """
        future, _future = MPFuture.make_pair()
        assert all(bit in '01' for bit in group_bits)
        self.pipe.send(('_set_group_bits', [], dict(group_bits=group_bits, future=_future)))
        return future.result() if wait else future

    async def _set_group_bits(self, group_bits: str, future: MPFuture):
        try:
            self._matchmaking.group_key_manager.group_bits = group_bits
            return future.set_result(None)
        except Exception as e:
            if not future.done():
                future.set_exception(e)


def is_power_of_two(n):
    """ Check whether n is a power of 2 """
    return (n != 0) and (n & (n - 1) == 0)


def _background_thread_fetch_current_state(serializer: SerializerBase, pipe: mp.connection.Connection,
                                           get_current_state_ref: weakref.WeakMethod):
    """
    Executed in the host process as a background thread. Fetches the averager state when asked by peers.
    :param serializer: a serializer with which to convert metadata into bytes
    :param pipe: DecentralizedAverager's control pipe (from host process side)
    :param get_current_state_ref: a WeakMethod wrapped around DecentralizedAverager.get_current_state (instance-bound)
    """
    while True:
        try:
            trigger, future = pipe.recv()
        except BaseException as e:
            logger.debug(f"Averager background thread finished: {repr(e)}")
            break
            
        if trigger == '_SHUTDOWN':
            break

        assert trigger == '_TRIGGER_GET_CURRENT_STATE'
        try:
            get_current_state = get_current_state_ref()
            if get_current_state is None:
                break
            state_metadata, state_tensors = get_current_state()
            del get_current_state

            state_metadata = serializer.dumps(state_metadata)
            state_tensors = tuple(tensor.cpu().detach().requires_grad_(tensor.requires_grad)
                                  for tensor in state_tensors)
            # note: we cast tensors to CPU on host side to avoid initializing cuda in the guest process
            future.set_result((state_metadata, state_tensors))
        except BaseException as e:
            future.set_exception(e)
            logger.warning(e)
            continue


def compute_schema_hash(tensors: Sequence[torch.Tensor]) -> bytes:
    schema_dicts = [{field: str(value) for field, value in asdict(TensorDescriptor.from_tensor(t)).items()} for t in tensors]
    return DHTID.generate(source=schema_dicts).to_bytes()

