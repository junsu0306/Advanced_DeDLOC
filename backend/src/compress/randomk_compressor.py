import numpy as np
import torch
import threading
import concurrent.futures
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Xorshift128Plus:
    def __init__(self, seed1=324, seed2=4444):
        """Xorshift128+ 난수 생성기 (Python 기본 int 타입 사용)"""
        self.state = [int(seed1), int(seed2)]  # ✅ numpy.uint64 대신 Python 기본 int 사용

    def next(self):
        """Xorshift128+ 난수 생성 (오버플로우 방지)"""
        s1, s0 = self.state
        s1 ^= (s1 << 23) & 0xFFFFFFFFFFFFFFFF  # ✅ 64비트 마스킹
        s1 ^= (s1 >> 17) & 0xFFFFFFFFFFFFFFFF
        s1 ^= s0
        s1 ^= (s0 >> 26) & 0xFFFFFFFFFFFFFFFF
        self.state[0] = s0
        self.state[1] = s1
        return (s0 + s1) & 0xFFFFFFFFFFFFFFFF  # ✅ 64비트 범위 유지

    def randint_unique(self, low, high, size):
        """C++ AVX2 방식처럼, 주어진 범위에서 유일한 난수 생성"""
        unique_indices = set()
        while len(unique_indices) < size:
            val = low + (self.next() % (high - low))  # ✅ 64비트 연산 오류 방지
            unique_indices.add(val)  # ✅ 중복 방지
        return np.array(list(unique_indices), dtype=np.int32)

# Random-K 압축기 (수정된 난수 생성기 적용)
class RandomkCompressor:
    def __init__(self, multicore=True, num_threads=4):
        self.multicore = multicore
        self.num_threads = num_threads
        self.rng = Xorshift128Plus()  # ✅ Xorshift128+ 사용
        logging.info(f"RandomkCompressor initialized with multicore={multicore}, num_threads={num_threads}")

    def compress(self, src, k, dst_idx, dst_val, idx_offset=0):
        """멀티스레딩을 지원하는 Random-K 압축 (CPU 최적화 난수 생성)"""
        logging.info(f"Starting compression: k={k}, data_size={len(src)}")
        assert idx_offset == 0, "C++ 코드와 동일하게 idx_offset은 0으로 고정됨."

        src = np.asarray(src, dtype=np.float32)
        dst_idx = np.asarray(dst_idx, dtype=np.int32)
        dst_val = np.asarray(dst_val, dtype=np.float32)

        if self.multicore:
            return self._compress_multicore(src, k, dst_idx, dst_val)
        else:
            return self.impl_simd(src, k)

    def _compress_multicore(self, src, k, dst_idx, dst_val):
        """멀티스레딩을 활용한 Random-K 압축 (ThreadPoolExecutor 사용)"""
        num_threads = min(self.num_threads, k)
        src_len = len(src)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            indices = self.rng.randint_unique(0, src_len, k)  # ✅ 유일한 난수 생성 방식 적용
            split_indices = np.array_split(indices, num_threads)  # 스레드별로 분할

            futures = []
            for thread_idx, chunk in enumerate(split_indices):
                logging.info(f"Thread {thread_idx}: Processing {len(chunk)} elements.")
                futures.append(
                    executor.submit(self._worker, src, chunk, dst_idx, dst_val, thread_idx)
                )

            concurrent.futures.wait(futures)

        logging.info("Compression complete.")
        return k

    def _worker(self, src, indices, dst_idx, dst_val, thread_idx):
        """멀티스레딩 작업을 수행하는 함수"""
        values = src[indices]
        start = thread_idx * len(indices)
        dst_idx[start:start + len(indices)] = indices
        dst_val[start:start + len(values)] = values
        logging.info(f"Thread {thread_idx}: Selected indices {indices}")

    def impl_simd(self, src, k):
        """SIMD 기반 Random-K 압축 (벡터 연산)"""
        logging.info(f"Executing SIMD compression: k={k}")
        src_tensor = torch.tensor(src, dtype=torch.float32)
        src_len = len(src_tensor)

        indices = self.rng.randint_unique(0, src_len, k)  # ✅ 유일한 난수 생성 방식 적용
        values = src_tensor[indices]

        logging.info(f"Selected indices: {indices}")
        return indices, values.numpy()
