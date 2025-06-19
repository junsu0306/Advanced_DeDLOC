import torch
from adam_optimizer import AdamOptimizer

def test_adam_optimizer():
    print("=== Optimized Adam Optimizer Test ===")

    # 초기 파라미터 설정
    param = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    optimizer = AdamOptimizer(lr=0.001, b1=0.9, b2=0.999, weight_decay=0.01, eps=1e-8, amsgrad=True)

    # Sparse Gradient (인덱스 기반 업데이트)
    grad_values = torch.tensor([0.1, 0.2], dtype=torch.float32, device=param.device)
    grad_indices = torch.tensor([0, 2], dtype=torch.int64, device=param.device)  # 첫 번째와 세 번째 요소만 업데이트

    # 옵티마이저 적용
    print("Before update:", param)
    optimizer.optimize(param, "layer1", grad_values, grad_indices)
    print("After update:", param)

    # AMSGrad 테스트
    optimizer.optimize(param, "layer1", grad_values, grad_indices)
    print("After AMSGrad update:", param)

    # Learning rate 변경 테스트
    optimizer.configure("lr", 0.005)
    optimizer.optimize(param, "layer1", grad_values, grad_indices)
    print("After LR=0.005 update:", param)

if __name__ == "__main__":
    test_adam_optimizer()
