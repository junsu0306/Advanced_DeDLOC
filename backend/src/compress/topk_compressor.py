import torch
import threading

class TopkCompressor:
    def __init__(self, k=0.1, num_threads=4, verbose=False):
        self.k = k
        self.num_threads = num_threads
        self.verbose = verbose

    def _local_topk(self, grad_chunk, k):
        total = grad_chunk.numel()
        k = int(k)

        if total == 0 or k <= 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0)

        grad_abs = grad_chunk.abs()

        if not torch.all(torch.isfinite(grad_abs)):
            if self.verbose:
                print(f"[TopK Warning] Non-finite values detected. Skipping.")
            return torch.empty(0, dtype=torch.long), torch.empty(0)

        if k >= grad_abs.numel():
            k = grad_abs.numel() - 1 if grad_abs.numel() > 1 else 1
        if k <= 0:
            k = 1

        try:
            topk = torch.topk(grad_abs, k, sorted=False)
            indices = topk.indices
            values = grad_chunk[indices]
            return indices, values
        except RuntimeError as e:
            if self.verbose:
                print(f"[TopK Error] torch.topk() failed: {e} → fallback to torch.sort()")
            try:
                _, sorted_indices = torch.sort(grad_abs, descending=True)
                sorted_indices = sorted_indices[:k]
                values = grad_chunk[sorted_indices]
                if sorted_indices.numel() == 0 and self.verbose:
                    print(f"[Sort Fallback Error] No indices selected after sort.")
                return sorted_indices, values
            except Exception as e2:
                if self.verbose:
                    print(f"[TopK Sort-Fallback Error] {e2}")
                return torch.empty(0, dtype=torch.long), torch.empty(0)

    def compress(self, grad: torch.Tensor):
        grad_cpu = grad.detach().cpu()
        total_numel = grad_cpu.numel()

        k_val = max(1, int(self.k * total_numel)) if self.k < 1 else int(self.k)
        k_val = min(k_val, total_numel)
        if self.verbose:
            print(f"[compress] total={total_numel}, k ratio={self.k}, k_val={k_val}")

        chunk_size = (total_numel + self.num_threads - 1) // self.num_threads
        local_results = [None for _ in range(self.num_threads)]
        threads = []

        def worker(thread_idx):
            try:
                start = thread_idx * chunk_size
                end = min(start + chunk_size, total_numel)
                if start >= end:
                    local_results[thread_idx] = (torch.empty(0, dtype=torch.long), torch.empty(0))
                    return
                grad_chunk = grad_cpu[start:end]
                local_k = min(k_val, grad_chunk.numel())
                if self.verbose:
                    print(f"[worker-{thread_idx}] chunk={grad_chunk.numel()}, local_k={local_k}")
                idx_chunk, val_chunk = self._local_topk(grad_chunk, local_k)
                idx_chunk += start
                local_results[thread_idx] = (idx_chunk, val_chunk)
            except Exception as e:
                if self.verbose:
                    print(f"[Thread-{thread_idx} Error] {e}")
                local_results[thread_idx] = (torch.empty(0, dtype=torch.long), torch.empty(0))

        for i in range(self.num_threads):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        valid_results = [r for r in local_results if r[0].numel() > 0]
        if not valid_results:
            if self.verbose:
                print("[TopkCompressor] Warning: No valid top-k results. Returning empty tensors.")
            return torch.empty(0, dtype=torch.long), torch.empty(0)

        all_indices = torch.cat([r[0] for r in valid_results], dim=0)
        all_values = torch.cat([r[1] for r in valid_results], dim=0)

        if all_values.numel() > k_val:
            try:
                final_topk = torch.topk(all_values.abs(), k_val, sorted=False)
                final_indices = all_indices[final_topk.indices]
                final_values = all_values[final_topk.indices]
            except Exception as e:
                if self.verbose:
                    print(f"[Final TopK Error] {e}")
                final_indices, final_values = all_indices, all_values
        else:
            final_indices, final_values = all_indices, all_values

        return final_indices, final_values
