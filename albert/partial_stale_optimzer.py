import logging
import time
import torch
import cvxpy as cp  # cvxpy 및 적절한 LP solver (예: GLPK_MI)가 설치되어 있어야 합니다.
from hivemind.optim.collaborative import CollaborativeOptimizer as BaseCollaborativeOptimizer
from hivemind.utils import get_dht_time

logger = logging.getLogger(__name__)

class AdaptivePartialStaleCollaborativeOptimizer(BaseCollaborativeOptimizer):
    """
    Adaptive Averaging과 Partial Staleness 기능을 결합한 Collaborative Optimizer.

    Parameters:
      partial_stale (bool): True이면 gradient 업데이트를 1 step 지연시키는 partial staleness 기능을 활성화합니다.
      use_adaptive_averaging (bool): True이면 각 step() 호출 전에 DHT에 저장된 피어 메트릭을 바탕으로 CVXPY를 사용하여
                                     LP 문제를 풀고 피어의 역할(gradient 계산 참여 여부)을 결정합니다.
      lp_solver_timeout (float): LP 솔버가 허용할 최대 계산 시간 (초 단위)
      max_lp_iterations (int): LP 솔버의 최대 반복 횟수
      기타 인자는 부모 클래스(BaseCollaborativeOptimizer)에 전달됩니다.
    """
    def __init__(self, partial_stale=False, use_adaptive_averaging=False,
                 lp_solver_timeout=0.05, max_lp_iterations=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partial_stale = partial_stale
        self.use_adaptive_averaging = use_adaptive_averaging
        self.lp_solver_timeout = lp_solver_timeout
        self.max_lp_iterations = max_lp_iterations
        self.stale_grad_buffer = None  # 이전 iteration에서 계산된 gradient를 저장
        self.last_optimization_time = None
        # 로컬 피어 ID: dht.local_public_key를 문자열로 변환하여 사용 (환경에 맞게 조정)
        self.local_peer_id = str(self.dht.local_public_key) if hasattr(self.dht, "local_public_key") else "unknown"
        self.last_peers_info = []  # 최근 수집한 피어 정보를 저장

    def step(self, batch_size: int = None, **kwargs):
        if self.use_adaptive_averaging:
            logger.info("Adaptive averaging 활성화: 최적 aggregation 전략 계산 시작...")
            # DHT에서 피어 메트릭 수집
            peers_info = self.collect_peers_info()
            self.last_peers_info = peers_info  # 이후 역할 판단에 사용
            # CVXPY를 이용해 LP 문제를 풀어 각 피어의 gradient 계산 참여 여부를 결정
            roles = self.solve_adaptive_averaging(peers_info)
            # LP 최적화 결과를 기반으로 로컬 피어의 역할(통신 전략) 업데이트
            self.update_communication_strategy(roles)
            self.last_optimization_time = time.time()
        if not self.partial_stale:
            return super().step(batch_size=batch_size, **kwargs)
        
        # Partial stale 기능: 1-step delay를 구현
        orig_apply_accum = self.apply_accumulated_grads_
        local_grads = [None]  # gradient를 저장할 mutable container
        
        def store_in_buffer(scale_by=None):
            """
            apply_accumulated_grads_의 대체 함수: 실제 optimizer.step()을 호출하지 않고
            계산된 gradient를 local_grads에 저장합니다.
            """
            param_list = []
            for group in self.opt.param_groups:
                param_list.extend(group["params"])
            grads = []
            if self.reuse_grad_buffers:
                for p in param_list:
                    if p.grad is None:
                        grads.append(None)
                    else:
                        grads.append(p.grad.clone())
            else:
                if self._grads is None:
                    self._grads = [torch.zeros_like(p) for p in param_list]
                if scale_by is not None:
                    for g in self._grads:
                        g.mul_(scale_by)
                for g in self._grads:
                    grads.append(g.clone())
            local_grads[0] = grads
            # optimizer.step() 호출하지 않음
            return
        
        # apply_accumulated_grads_를 일시적으로 대체 후 super().step() 호출
        self.apply_accumulated_grads_ = store_in_buffer
        super().step(batch_size=batch_size, **kwargs)
        self.apply_accumulated_grads_ = orig_apply_accum
        
        # 이전 iteration에서 저장된 gradient 버퍼가 있다면 현재 iteration에 적용
        if self.stale_grad_buffer is not None:
            self._apply_stale_grad(self.stale_grad_buffer)
        # 이번 iteration에서 계산된 gradient를 버퍼에 저장 (다음 iteration에 적용)
        if local_grads[0] is not None:
            self.stale_grad_buffer = local_grads[0]
        else:
            logger.debug("이번 step에서 gradient가 계산되지 않았습니다 (피어 없음 또는 step 스킵).")
        return

    def _apply_stale_grad(self, grad_list):
        """
        저장된 gradient 버퍼(grad_list)를 각 파라미터에 복사한 후, optimizer.step()을 호출해 파라미터를 업데이트합니다.
        """
        param_list = []
        for group in self.opt.param_groups:
            param_list.extend(group["params"])
        if len(param_list) != len(grad_list):
            logger.warning("파라미터 수와 gradient 버퍼 길이가 일치하지 않습니다.")
        for p, g in zip(param_list, grad_list):
            if g is None:
                continue
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)
        self.opt.step()
        for p in param_list:
            if p.grad is not None:
                p.grad = None

    def collect_peers_info(self):
        """
        Hivemind DHT에서 피어 메트릭을 수집합니다.
        
        가정: 각 피어의 메트릭(LocalMetrics)이 self.prefix + "_metrics" 키에 딕셔너리 형태로 저장되어 있습니다.
        'samples_per_second' 값을 compute_speed로 사용하며, 업/다운로드 및 latency 정보가 없으면 기본값을 사용합니다.
        
        반환:
          [
            {'peer_id': <peer id>, 'compute_speed': ..., 'upload': ..., 'download': ..., 'latency': ...},
            ...
          ]
        """
        peers_list = []
        key = self.prefix + "_metrics"
        try:
            record = self.dht.get(key, latest=True)
            if hasattr(record, "result"):
                record = record.result()
            if record is not None:
                metrics_dict = record.value  # 딕셔너리: {peer_id: metric_record, ...}
                for peer_key, record_obj in metrics_dict.items():
                    m = record_obj.value if hasattr(record_obj, "value") else record_obj
                    compute_speed = m.get("samples_per_second", 1.0)
                    upload = m.get("upload", 100.0)
                    download = m.get("download", 120.0)
                    latency = m.get("latency", 0.1)
                    peer_info = {
                        'peer_id': str(peer_key),
                        'compute_speed': compute_speed,
                        'upload': upload,
                        'download': download,
                        'latency': latency,
                    }
                    peers_list.append(peer_info)
            else:
                logger.warning("DHT에서 '%s' 메트릭 레코드를 가져오지 못했습니다.", key)
        except Exception as e:
            logger.warning("DHT 메트릭 수집 중 예외 발생: %s", e)
        if not peers_list:
            logger.warning("수집된 피어 메트릭이 없습니다. 기본값을 사용합니다.")
        return peers_list

    def solve_adaptive_averaging(self, peers_info):
        """
        수집된 피어 정보를 바탕으로 CVXPY를 이용해 LP 문제를 해결하여,
        각 피어가 gradient 계산에 참여할지 여부(c[i] ∈ {0,1})를 결정합니다.
        
        목적함수: 각 피어의 compute_speed를 가중치로 하여 총 compute 능력의 합을 최대화.
        제약 조건: 전체 피어 중 최소 50% 이상은 계산에 참여하도록 제한.
        
        반환: 각 피어에 대한 boolean 리스트 (True: compute mode, False: aggregator-only).
        """
        n = len(peers_info)
        if n == 0:
            return []
        c = cp.Variable(n, boolean=True)
        compute_speeds = [info['compute_speed'] for info in peers_info]
        objective = cp.Maximize(cp.sum(cp.multiply(compute_speeds, c)))
        constraints = [cp.sum(c) >= 0.5 * n]
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.GLPK_MI, max_iters=self.max_lp_iterations, timeout=self.lp_solver_timeout)
        except Exception as e:
            logger.warning("LP solver 예외 발생: %s", e)
        if c.value is None:
            logger.warning("LP solver 수렴 실패 – 기본값 사용 (모든 피어가 compute 참여)")
            roles = [True] * n
        else:
            roles = [bool(round(val)) for val in c.value]
        logger.info("LP 최적화 결과 (compute 참여 여부): %s", roles)
        return roles

    def update_communication_strategy(self, roles):
        """
        수집된 피어 메트릭(self.last_peers_info)과 LP 최적화 결과(roles)를 바탕으로,
        로컬 피어의 역할을 결정하고 이를 내부 통신 전략에 반영합니다.
        
        구현:
          - 로컬 피어의 ID(self.local_peer_id)를 사용하여, self.last_peers_info 내 인덱스를 찾습니다.
          - 해당 인덱스의 roles 값이 True이면 compute 모드, False이면 aggregator 모드로 설정합니다.
          - 실제로는 self.client_mode 플래그를 업데이트하여, 이후 통신 및 연산 로직에 영향을 줍니다.
        """
        peers_info = self.last_peers_info
        local_id = self.local_peer_id
        local_index = None
        for i, peer in enumerate(peers_info):
            if peer.get('peer_id') == local_id:
                local_index = i
                break
        if local_index is None:
            logger.warning("로컬 피어 ID '%s'가 peers_info에서 발견되지 않았습니다. 기본적으로 compute 모드로 설정합니다.", local_id)
            self.set_compute_mode()
            return
        local_role = roles[local_index]
        if local_role:
            self.set_compute_mode()
        else:
            self.set_aggregator_mode()

    def set_compute_mode(self):
        """
        로컬 피어의 역할을 compute 모드로 설정합니다.
        이 경우 해당 피어는 gradient 계산 및 전송을 수행합니다.
        """
        logger.info("로컬 피어를 COMPUTE 모드로 설정합니다.")
        self.client_mode = False
        # 추가로, 예를 들어 throughput, target_batch_size 등의 값을 조정할 수 있습니다.

    def set_aggregator_mode(self):
        """
        로컬 피어의 역할을 aggregator 모드로 설정합니다.
        이 경우 해당 피어는 주로 gradient aggregation 전담으로 처리하며,
        직접 gradient 계산에는 참여하지 않습니다.
        """
        logger.info("로컬 피어를 AGGREGATOR 모드로 설정합니다.")
        self.client_mode = True
        # aggregator 역할에 맞게 통신 대역폭 할당 등 내부 파라미터를 조정할 수 있습니다.
