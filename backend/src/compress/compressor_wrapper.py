import torch
from compress.topk_compressor import TopkCompressor
from compress.thresholdv_compressor import ThresholdvCompressor
from compress.thresholdv16_compressor import ThresholdvCompressor16

class CompressorWrapper:
    def __init__(self, method: str, ratio: float = 0.01):
        self.method = method.lower()
        self.ratio = ratio

        if self.method == "topk":
            self.compressor = TopkCompressor(k=self.ratio)
        elif self.method == "thresholdv":
            self.compressor = ThresholdvCompressor()
        elif self.method == "thresholdv16":
            self.compressor = ThresholdvCompressor16()
        else:
            raise ValueError(f"Unsupported compression method: {method}")

    def compress(self, name: str, tensor: torch.Tensor, ratio: float = None):
        ratio = ratio if ratio is not None else self.ratio

        # TopK 방식은 flatten 필수
        if self.method == "topk":
            tensor_flat = tensor.detach().view(-1)
            index, value = self.compressor.compress(tensor_flat)

            if index.numel() == 0 or value.numel() == 0:
                print(f"[Compression Error] {name}: compression returned empty result")

            return index, value

        else:
            numel_to_select = int(ratio * tensor.numel())
            if numel_to_select == 0:
                return torch.empty(0, dtype=torch.int32), torch.empty(0)

            dst_idx = torch.empty(numel_to_select, dtype=torch.int32)
            dst_val = torch.empty(numel_to_select, dtype=tensor.dtype)
            length = self.compressor.compress(name, tensor.view(-1), numel_to_select, dst_idx, dst_val)

            if length == 0:
                print(f"[Compression Error] {name}: threshold compression returned 0 length")

            return dst_idx[:length], dst_val[:length]
