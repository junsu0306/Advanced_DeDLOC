import torch
from compress.topk_compressor import TopkCompressor
from compress.thresholdv_compressor import ThresholdvCompressor
from compress.thresholdv16_compressor import ThresholdvCompressor16

class CompressorWrapper:
    def __init__(self, method: str):
        self.method = method.lower()
        if self.method == "topk":
            self.compressor = TopkCompressor()
        elif self.method == "thresholdv":
            self.compressor = ThresholdvCompressor()
        elif self.method == "thresholdv16":
            self.compressor = ThresholdvCompressor16()
        else:
            raise ValueError(f"Unsupported compression method: {method}")

    def compress(self, name: str, tensor: torch.Tensor, ratio: float):
        if self.method == "topk":
            return self.compressor.compress(tensor)
        else:
            numel_to_select = int(ratio * tensor.numel())
            if numel_to_select == 0:
                return torch.empty(0, dtype=torch.int32), torch.empty(0)
            dst_idx = torch.empty(numel_to_select, dtype=torch.int32)
            dst_val = torch.empty(numel_to_select, dtype=tensor.dtype)
            length = self.compressor.compress(name, tensor.view(-1), numel_to_select, dst_idx, dst_val)
            return dst_idx[:length], dst_val[:length]