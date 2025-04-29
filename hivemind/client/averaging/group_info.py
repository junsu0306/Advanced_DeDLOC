from dataclasses import dataclass
from typing import Tuple, List, Dict
import pickle
import socket
import time
from concurrent.futures import ThreadPoolExecutor

from hivemind.utils import Endpoint

@dataclass(frozen=True)
class PeerInfo:
    """
    Holds detailed metadata for a single peer.
    """
    endpoint: Endpoint
    compute_rate: float      # samples per second
    upload_bw: float         # upload bandwidth (Mbps)
    download_bw: float       # download bandwidth (Mbps)
    latency_map: Dict[Endpoint, float]  # measured latencies from local to each peer

@dataclass(frozen=True)
class GroupInfo:
    """ A group of peers assembled through decentralized matchmaking """
    group_id: bytes                            # unique identifier for this group
    endpoints: Tuple[Endpoint, ...]            # ordered sequence of peer endpoints
    gathered: Tuple[bytes, ...]                # binary metadata blobs from each peer

    @property
    def group_size(self) -> int:
        return len(self.endpoints)

    def __contains__(self, endpoint: Endpoint) -> bool:
        return endpoint in self.endpoints

    def peer_infos(self) -> List[PeerInfo]:
        """
        Deserialize each peer's metadata blob into a PeerInfo instance
        and measure TCP connect latency from this host to each peer concurrently.
        """
        infos: List[PeerInfo] = []
        for self_ep, blob in zip(self.endpoints, self.gathered):
            data = pickle.loads(blob)
            # unpack metrics
            if isinstance(data, dict) and 'peer_metrics' in data:
                info_dict = data['peer_metrics']
            else:
                info_dict = data
            # measure latencies in a thread pool
            def measure_latency(peer: Endpoint) -> (Endpoint, float):
                if peer == self_ep:
                    return peer, 0.0
                start = time.time()
                try:
                    host, port = peer
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.3)
                    sock.connect((host, port))
                    sock.close()
                    return peer, (time.time() - start) * 1000
                except Exception:
                    return peer, float('inf')

            peers_to_check = [ep for ep in self.endpoints if ep != self_ep]
            with ThreadPoolExecutor(max_workers=min(8, len(peers_to_check))) as executor:
                results = executor.map(measure_latency, peers_to_check)
            latency_map = {peer: latency for peer, latency in results}

            infos.append(PeerInfo(
                endpoint=self_ep,
                compute_rate=info_dict.get('compute_rate', 0.0),
                upload_bw=info_dict.get('upload_bw', 0.0),
                download_bw=info_dict.get('download_bw', 0.0),
                latency_map=latency_map
            ))
        return infos
