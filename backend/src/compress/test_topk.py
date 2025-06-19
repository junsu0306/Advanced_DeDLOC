import torch
from topk_compressor import TopKCompressor
import time

def test_topk_compression(k=100, use_heap=False):
    print(f"\nTesting TopKCompressor (k={k}, use_heap={use_heap})...\n")

    #테스트할 Gradient 데이터 생성(랜덤한 tensor)
    torch.manual_seed(42)  #결과 재현성을 위해 시드 고정
    grad = torch.randn(10000)  #10,000개

    compressor = TopKCompressor(k, use_heap) #Top-K Compressor 인스턴스 생성

    #Compression 실행
    start_time = time.time()
    indices, values = compressor.compress(grad)
    compression_time = time.time() - start_time

    #Decompression 실행
    start_time = time.time()
    decompressed_grad = compressor.decompress(indices, values, grad.shape)
    decompression_time = time.time() - start_time

    #결과
    nonzero_original = (grad != 0).sum().item()
    nonzero_compressed = len(values)
    nonzero_decompressed = (decompressed_grad != 0).sum().item()

    #테스트 결과 출력
    print(f"Original Nonzero Elements: {nonzero_original}")
    print(f"Compressed Nonzero Elements: {nonzero_compressed}")
    print(f"Decompressed Nonzero Elements: {nonzero_decompressed}")
    print(f"Compression Time: {compression_time:.6f} sec")
    print(f"Decompression Time: {decompression_time:.6f} sec")

    #검증((압축 후 비율 = 예상값) 확인)
    assert nonzero_compressed <= nonzero_original, "Compression should reduce elements"
    assert nonzero_decompressed == nonzero_compressed, "Decompressed elements should match compressed elements"
    print(f"TopKCompressor Test Passed\n")

#실행(기본 방식: torch.topk())
test_topk_compression(k=100, use_heap=False)

#실행(힙정렬 방식)
test_topk_compression(k=100, use_heap=True)