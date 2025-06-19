import torch
from sgd_optimizer import SGDOptimizer

def test_sgd_optimizer():
    print("=== Optimized SGD Optimizer Test ===")

    # 초기 파라미터 설정
    param = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    optimizer = SGDOptimizer(lr=0.01, momentum=0.9, weight_decay=0.01, nesterov=True, smart_momentum=True)

    # Sparse Gradient (인덱스 기반 업데이트)
    grad_values = torch.tensor([0.1, 0.2], dtype=torch.float32, device=param.device)
    grad_indices = torch.tensor([0, 2], dtype=torch.int64, device=param.device)  # 첫 번째와 세 번째 요소만 업데이트

    # 옵티마이저 적용
    print("Before update:", param)
    optimizer.optimize(param, "layer1", grad_values, grad_indices)
    print("After update:", param)

    # Smart Momentum 테스트
    optimizer.optimize(param, "layer1", grad_values, grad_indices)
    print("After Smart Momentum update:", param)

    # Learning rate 변경 테스트
    optimizer.configure("lr", 0.05)
    optimizer.optimize(param, "layer1", grad_values, grad_indices)
    print("After LR=0.05 update:", param)

if __name__ == "__main__":
    test_sgd_optimizer()
