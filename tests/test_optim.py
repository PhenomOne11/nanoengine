import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim
from src.optim.sgd import SGD
from src.optim.adamw import AdamW
from src.optim.lr_scheduler import CosineAnnealingLR

def test_sgd_momentum_parity():
    torch.manual_seed(42)
    p_custom = torch.randn(5, 5)
    p_torch = p_custom.clone().detach()

    p_custom.grad = torch.zeros_like(p_custom)
    p_torch.grad = torch.zeros_like(p_torch)

    my_opt = SGD([p_custom], lr=0.01, momentum=0.9, weight_decay=1e-4)
    torch_opt = optim.SGD([p_torch], lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    for _ in range(10):
        grad = torch.randn(5, 5)
        p_custom.grad.copy_(grad)
        p_torch.grad.copy_(grad)

        my_opt.step()
        torch_opt.step()

        assert torch.allclose(p_custom, p_torch, atol=1e-6), "SGD parameter divergence detected!"
        
def test_adamw_parity():
    torch.manual_seed(42)
    p_custom = torch.randn(5, 5)
    p_torch = p_custom.clone().detach()

    p_custom.grad = torch.zeros_like(p_custom)
    p_torch.grad = torch.zeros_like(p_torch)

    my_opt = AdamW([p_custom], lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2)
    torch_opt = optim.AdamW([p_torch], lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2)

    for _ in range(10):
        grad = torch.randn(5, 5)
        p_custom.grad.copy_(grad)
        p_torch.grad.copy_(grad)

        my_opt.step()
        torch_opt.step()

        assert torch.allclose(p_custom, p_torch, atol=1e-6), "AdamW parameter divergence detected!"
        
def test_cosine_scheduler():
    p = torch.randn(2, 2)
    opt = SGD([p], lr=0.1)
    sched = CosineAnnealingLR(opt, T_max=10, eta_min=0.0)

    for _ in range(10):
        sched.step()
    assert abs(opt.lr - 0.0) < 1e-6, "Cosine Annealing did not reach eta_min at T_max!"


if __name__ == "__main__":
    test_sgd_momentum_parity()
    test_adamw_parity()
    test_cosine_scheduler()
    print("Stage 4 optimization and scheduler parity tests passed successfully!")