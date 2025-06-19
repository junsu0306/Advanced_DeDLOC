import numpy as np
import torch
import threading
import logging
from compress.thresholdv_compressor import ThresholdvCompressor

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ThresholdvCompressor16:
    _global_threshold_map = {}  # 클래스 변수로 전역 캐시 유지
    _lock = threading.Lock()  # 전역 잠금

    def __init__(self, multicore=True):
        self.multicore = multicore
        logging.info("ThresholdvCompressor16 initialized with multicore=%s", multicore)

    def compress(self, name, src, k, dst_idx, dst_val, idx_offset=0):
        logging.info("Starting compression for %s with k=%d", name, k)
        return self.impl_simd_v2(name, src, k, dst_idx, dst_val, idx_offset)

    def impl_simd_v2(self, name, src, k, dst_idx, dst_val, idx_offset=0):
        """CPU 친화적인 Threshold 압축 (캐싱 포함)"""
        src_tensor = torch.tensor(src, dtype=torch.float32).contiguous()
        abs_src = torch.abs(src_tensor)

        # 캐싱된 임계값을 확인하고 없으면 새로 계산
        with ThresholdvCompressor16._lock:
            if name not in ThresholdvCompressor16._global_threshold_map:
                logging.info("Threshold not found in cache for '%s'. Computing new one.", name)
                threshold = torch.kthvalue(abs_src, max(1, len(abs_src) - k))[0].item()
                threshold_inc = threshold * 0.01
                ThresholdvCompressor16._global_threshold_map[name] = threshold
                ThresholdvCompressor16._global_threshold_map[f"{name}_inc"] = threshold_inc
                logging.info("Calculated new threshold for '%s': %.6f", name, threshold)
            else:
                threshold = ThresholdvCompressor16._global_threshold_map[name]
                threshold_inc = ThresholdvCompressor16._global_threshold_map[f"{name}_inc"]
                logging.info("Using cached threshold for '%s': %.6f", name, threshold)

        # 마스크 적용하여 인덱스 선택
        mask = abs_src >= threshold
        valid_indices = torch.where(mask)[0]
        valid_values = src_tensor[valid_indices]

        cnt_found = min(k, len(valid_indices))
        dst_idx[:cnt_found] = valid_indices[:cnt_found].cpu().numpy()
        dst_val[:cnt_found] = valid_values[:cnt_found].cpu().numpy()

        # AIMD 기반 임계값 조정 (Adaptive Increase/Multiplicative Decrease)
        if k > cnt_found:
            threshold *= 0.99  # 드롭을 줄임
        elif k < cnt_found:
            threshold += threshold_inc  # 더 많은 드롭 수행

        with ThresholdvCompressor16._lock:
            ThresholdvCompressor16._global_threshold_map[name] = threshold
            ThresholdvCompressor16._global_threshold_map[f"{name}_inc"] = threshold_inc
        logging.info("AIMD-adjusted threshold for '%s': %.6f", name, threshold)

        # 보조 버퍼 추가 (초과된 데이터를 저장하여 보완)
        remaining_k = k - cnt_found
        if remaining_k > 0:
            sorted_abs_src, sorted_idx = torch.sort(abs_src, descending=True)
            backup_indices = sorted_idx[:remaining_k].cpu().numpy()
            backup_values = src_tensor[backup_indices].cpu().numpy()
            
            dst_idx[cnt_found:cnt_found+remaining_k] = backup_indices
            dst_val[cnt_found:cnt_found+remaining_k] = backup_values
            cnt_found += remaining_k
            logging.info("Added backup elements to maintain k=%d", k)

        logging.info("Compression complete for '%s'. Found %d elements.", name, cnt_found)
        return cnt_found

    def decompress(self, compressed_idx, compressed_val, original_size):
        """CPU 친화적인 압축 해제"""
        decompressed = np.zeros(original_size, dtype=np.float32)
        decompressed[compressed_idx] = compressed_val
        return decompressed
