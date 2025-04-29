from typing import Sequence, Tuple
import numpy as np
import scipy.optimize

from hivemind.utils.logging import get_logger
from group_info import PeerInfo

logger = get_logger(__name__)

LOAD_BALANCING_LP_DECIMALS = 9

def load_balance_peers(vector_size: int, peer_infos: Sequence[PeerInfo], min_size: int = 0) -> Tuple[int, ...]:
    """
    Find an optimal partitioning of weights for butterfly all-reduce given peer information.
    :param vector_size: total size of the averaged vector (in elements, not bytes)
    :param peer_infos: sequence of PeerInfo instances containing upload_bw and download_bw in Mbps
    :param min_size: peers that can aggregate less than this many elements will be assigned nothing
    :returns: an integer tuple where i-th element is the number of weights assigned to i-th peer
    """
    # Compute effective throughput per peer using harmonic mean of upload and download bandwidths
    throughputs = []
    for pi in peer_infos:
        up = pi.upload_bw
        down = pi.download_bw
        if up > 0 and down > 0:
            eff = 2 * up * down / (up + down)
        else:
            eff = up or down or 0.0
        throughputs.append(eff)

    if any(t > 0 for t in throughputs):
        throughputs_array = np.asarray(throughputs, dtype=np.float64)
        scores = optimize_parts_lp(vector_size, throughputs_array, min_size)
    else:
        logger.error("All peers have zero bandwidth, assigning equal share.")
        scores = np.ones(len(peer_infos), dtype=np.float64)

    # Convert continuous scores to integer part sizes
    return tuple(hagenbach_bishoff(vector_size, scores))


def optimize_parts_lp(vector_size: int, throughputs: np.ndarray, min_size: int = 0) -> np.ndarray:
    """
    Solve LP to minimize total all-reduce time via minimax reduction
    (unchanged from original implementation).
    """
    assert np.all(throughputs >= 0) and np.any(throughputs > 0)
    throughputs = np.asarray(throughputs, dtype=np.float64)
    permutation = np.argsort(-throughputs)
    throughputs = throughputs[permutation]
    is_nonzero = throughputs != 0

    group_size = len(throughputs)
    num_variables = group_size + 1  # [w_1, ..., w_N, xi]
    c = np.zeros(num_variables, dtype=np.float64)
    c[-1] = 1.0  # optimize w.r.t. xi

    nonnegative_weights = -np.eye(group_size, num_variables, dtype=c.dtype), np.zeros(group_size, c.dtype)
    weights_sum_to_one = c[None, :] - 1.0, np.array([-1.0])
    coeff_per_variable = (group_size - 2.0) / np.maximum(throughputs, 10 ** -LOAD_BALANCING_LP_DECIMALS)
    coeff_matrix_minus_xi = np.hstack([np.diag(coeff_per_variable), -np.ones((group_size, 1), c.dtype)])
    xi_is_maximum = coeff_matrix_minus_xi[is_nonzero], -1.0 / throughputs[is_nonzero]
    force_max_weights = np.eye(group_size, M=num_variables, dtype=c.dtype), is_nonzero.astype(c.dtype)

    A, b = list(map(np.concatenate, zip(nonnegative_weights, weights_sum_to_one, xi_is_maximum, force_max_weights)))
    solution = scipy.optimize.linprog(c, A_ub=A, b_ub=b, method='interior-point')
    if solution.success:
        peer_scores = solution.x[:group_size]
        if np.max(peer_scores) >= min_size / float(vector_size):
            peer_scores[peer_scores < min_size / float(vector_size)] = 0.0
        peer_scores = np.round(peer_scores, LOAD_BALANCING_LP_DECIMALS)
    else:
        logger.error(f"Failed to solve load-balancing for bandwidths {throughputs}.")
        peer_scores = np.ones(group_size, c.dtype)

    return peer_scores[np.argsort(permutation)]


def hagenbach_bishoff(vector_size: int, scores: Sequence[float]) -> Sequence[int]:
    """
    Split a vector between participants based on continuous fractions.
    (unchanged from original implementation).
    """
    total_score = sum(scores)
    allocated = [int(vector_size * score_i / total_score) for score_i in scores]
    while sum(allocated) < vector_size:
        quotients = [score / (allocated[idx] + 1) for idx, score in enumerate(scores)]
        idx_max = quotients.index(max(quotients))
        allocated[idx_max] += 1
    return allocated
