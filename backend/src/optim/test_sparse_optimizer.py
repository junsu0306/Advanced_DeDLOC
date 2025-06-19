import torch
import numpy as np
import unittest
from sparse_optimizer import SparseOptimizer  # 변환한 Python 파일 임포트

class TestSparseOptimizer(unittest.TestCase):

    def setUp(self):
        """ 테스트 전에 실행되는 초기화 코드 """
        self.optimizer = SparseOptimizer(lr=0.1)

    def test_get_set_lr(self):
        """ 학습률 (LR) 설정 및 가져오기 테스트 """
        self.optimizer.set_lr(0.05)
        self.assertEqual(self.optimizer.get_lr(), 0.05)

    def test_check_validity(self):
        """ 데이터 유효성 검사 테스트 """
        param = torch.randn(10, dtype=torch.float32, requires_grad=True)
        grad_values = torch.randn(5, dtype=torch.float32)
        grad_indices = torch.randint(0, 10, (5,), dtype=torch.int32)

        # 유효한 데이터 → 정상 실행
        self.optimizer.check_validity(param, grad_values, grad_indices)

        # 데이터 타입 오류 테스트
        with self.assertRaises(AssertionError):
            wrong_dtype = torch.randn(5, dtype=torch.float64)  # float64는 지원되지 않음
            self.optimizer.check_validity(param, wrong_dtype, grad_indices)

    def test_optimize(self):
        """ PyTorch 기반 optimize 테스트 """
        param = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32, requires_grad=True)
        grad_values = torch.tensor([0.1, 0.2], dtype=torch.float32)
        grad_indices = torch.tensor([1, 3], dtype=torch.int32)

        self.optimizer.optimize(param, "test", grad_values, grad_indices)
        
        # 수동으로 업데이트된 값 확인
        expected = torch.tensor([1.0, 2.0 - 0.1 * 0.1, 3.0, 4.0 - 0.1 * 0.2, 5.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(param, expected, atol=1e-6))

    def test_optimize_raw(self):
        """ NumPy 기반 optimize_raw 테스트 """
        param = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        grad_values = np.array([0.1, 0.2], dtype=np.float32)
        grad_indices = np.array([1, 3], dtype=np.int32)

        self.optimizer.optimize_raw(param, grad_values, grad_indices)
        
        # 수동으로 업데이트된 값 확인
        expected = np.array([1.0, 2.0 - 0.1 * 0.1, 3.0, 4.0 - 0.1 * 0.2, 5.0], dtype=np.float32)
        np.testing.assert_allclose(param, expected, atol=1e-6)

    def test_configure(self):
        """ configure 테스트 """
        self.optimizer.configure("learning_rate", 0.01)
        self.assertEqual(self.optimizer.get_lr(), 0.01)

        # 잘못된 설정값 입력 시 오류 발생 확인
        with self.assertRaises(ValueError):
            self.optimizer.configure("unknown_option", 1.0)

if __name__ == "__main__":
    unittest.main()
