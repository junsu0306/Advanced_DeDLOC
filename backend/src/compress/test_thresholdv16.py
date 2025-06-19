import numpy as np 
import torch
from thresholdv16_compressor import ThresholdvCompressor16
import time

def test_threshold_compression(k=100):
    print(f"\nTesting ThresholdvCompressor16 (k={k})...\n")

    # 테스트할 Gradient 데이터 생성 (랜덤한 텐서)
    torch.manual_seed(42)
    grad = np.random.randn(10000).astype(np.float32)  # 10,000개의 요소

    # Threshold Compressor 인스턴스 생성
    compressor = ThresholdvCompressor16(multicore=True)

    # Destination arrays
    dst_idx = np.zeros(k, dtype=np.int32)
    dst_val = np.zeros(k, dtype=np.float32)

    # Compression 실행
    start_time = time.time()
    cnt_found = compressor.compress("test_tensor", grad, k, dst_idx, dst_val, idx_offset=0)
    compression_time = time.time() - start_time

    # Decompression 실행
    decompressed_grad = compressor.decompress(dst_idx[:cnt_found], dst_val[:cnt_found], len(grad))

    # 결과 검증
    nonzero_compressed = np.count_nonzero(dst_val)
    nonzero_decompressed = np.count_nonzero(decompressed_grad)

    print(f"Compressed Nonzero Elements: {nonzero_compressed}")
    print(f"Decompressed Nonzero Elements: {nonzero_decompressed}")
    print(f"Compression Time: {compression_time:.6f} sec")

    assert nonzero_decompressed == nonzero_compressed, "Decompressed elements should match compressed elements!"
    print("ThresholdvCompressor16 Test Passed!\n")

if __name__ == "__main__":
    test_threshold_compression(k=100)
