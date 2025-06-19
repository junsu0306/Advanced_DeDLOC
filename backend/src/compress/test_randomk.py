import numpy as np
import torch
from randomk_compressor import RandomkCompressor
import time

def test_randomk_compression(k=100):
    print(f"\nTesting RandomkCompressor (k={k})...\n")

    # 테스트할 Gradient 데이터 생성 (랜덤한 텐서)
    torch.manual_seed(42)
    grad = np.random.randn(10000).astype(np.float32)  # 10,000개의 요소

    # RandomkCompressor 인스턴스 생성
    compressor = RandomkCompressor(multicore=True, num_threads=4)

    # Destination arrays
    dst_idx = np.zeros(k, dtype=np.int32)
    dst_val = np.zeros(k, dtype=np.float32)

    # Compression 실행
    start_time = time.time()
    cnt_found = compressor.compress(grad, k, dst_idx, dst_val, idx_offset=0)
    compression_time = time.time() - start_time

    # 결과 검증
    nonzero_compressed = np.count_nonzero(dst_val)

    # 테스트 결과 출력
    print(f"Compressed Nonzero Elements: {nonzero_compressed}")
    print(f"Compression Time: {compression_time:.6f} sec")

    # 검증 (압축된 요소 개수가 정확한지 확인)
    assert nonzero_compressed == k, "Compressed elements count should match k!"
    assert len(np.unique(dst_idx)) == k, "Selected indices should be unique!"
    print("RandomkCompressor Test Passed!\n")

# 실행 (k=100 설정)
if __name__ == "__main__":
    test_randomk_compression(k=100)
