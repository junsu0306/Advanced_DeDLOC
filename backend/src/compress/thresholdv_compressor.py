import numpy as np
import torch
import threading
import logging

# 로깅 완전 비활성화
logger = logging.getLogger("ThresholdvCompressor")
logger.disabled = True

class ThresholdvCompressor:
    _global_threshold_map = {}
    _lock = threading.Lock()

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

    def compress(self, name, src, k, dst_idx, dst_val, idx_offset=0):
        return self.impl_simd_gpu(name, src, k, dst_idx, dst_val, idx_offset)

    def impl_simd_gpu(self, name, src, k, dst_idx, dst_val, idx_offset=0):
        device = "cuda" if self.use_gpu else "cpu"
        src_tensor = torch.tensor(src, dtype=torch.float32, device=device)
        abs_src = torch.abs(src_tensor)

        # threshold 캐싱 구조 (key = name)
        with ThresholdvCompressor._lock:
            if name not in ThresholdvCompressor._global_threshold_map:
                threshold = np.percentile(abs_src.cpu().numpy(), (1 - k / len(abs_src)) * 100)
                ThresholdvCompressor._global_threshold_map[name] = threshold
            else:
                threshold = ThresholdvCompressor._global_threshold_map[name]

        # 마스크 적용
        mask = abs_src >= threshold
        valid_indices = torch.nonzero(mask, as_tuple=True)[0]
        valid_values = src_tensor[valid_indices]

        cnt_found = valid_indices.numel()

        # 압축된 인덱스/값 복사
        num_to_copy = min(k, cnt_found)
        dst_idx[:num_to_copy].copy_(valid_indices[:num_to_copy].cpu().to(dtype=torch.int32))
        dst_val[:num_to_copy].copy_(valid_values[:num_to_copy].cpu())

        # AIMD 방식 threshold 조정 (C++ 방식과 동일)
        max_grad = torch.max(abs_src).item()
        traversal_ratio = 1.0
        find_ratio = cnt_found / k if k > 0 else 0.0

        if traversal_ratio > find_ratio:
            threshold *= 0.99
        elif traversal_ratio < find_ratio:
            threshold += 0.01 * max_grad

        ThresholdvCompressor._global_threshold_map[name] = threshold

        # 부족한 경우 백업 (top-k로 보완)
        remaining_k = k - num_to_copy
        if remaining_k > 0:
            sorted_abs_src, sorted_idx = torch.sort(abs_src, descending=True)
            backup_indices = sorted_idx[:remaining_k].cpu().numpy()
            backup_values = src_tensor[backup_indices].cpu().numpy()
            dst_idx[num_to_copy:num_to_copy+remaining_k].copy_(torch.from_numpy(backup_indices).to(dtype=torch.int32))
            dst_val[num_to_copy:num_to_copy+remaining_k].copy_(torch.from_numpy(backup_values))
            num_to_copy += remaining_k

        return num_to_copy

    def decompress(self, compressed_idx: np.ndarray, compressed_val: np.ndarray, original_size: int):
        device = "cuda" if self.use_gpu else "cpu"
        decompressed = torch.zeros(original_size, dtype=torch.float32, device=device)
        decompressed[compressed_idx] = torch.tensor(compressed_val, dtype=torch.float32, device=device)
        return decompressed
