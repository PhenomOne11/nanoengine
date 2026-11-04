import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from src.layers.norm import BatchNorm2d, LayerNorm, InstanceNorm2d, GroupNorm

def test_batchnorm2d_parity():
    torch.manual_seed(42)
    n, c, h, w = 4, 3, 6, 6

    x_custom = torch.randn(n, c, h, w)
    x_torch = x_custom.clone().detach().requires_grad_(True)

    my_bn = BatchNorm2d(c)
    torch_bn = nn.BatchNorm2d(c)
    
    my_bn.train()
    torch_bn.train()
    
    out_custom = my_bn(x_custom)
    out_torch = torch_bn(x_torch)
    assert torch.allclose(out_custom, out_torch, atol=1e-5), "BatchNorm2d train forward mismatch!"
    
    grad_out = torch.randn_like(out_custom)
    dx_custom = my_bn.backward(grad_out)
    out_torch.backward(grad_out)
    
    assert torch.allclose(my_bn.gamma.grad.squeeze(), torch_bn.weight.grad, atol=1e-5), "BatchNorm2d gamma grad mismatch!"
    assert torch.allclose(my_bn.beta.grad.squeeze(), torch_bn.bias.grad, atol=1e-5), "BatchNorm2d beta grad mismatch!"
    assert torch.allclose(dx_custom, x_torch.grad, atol=1e-5), "BatchNorm2d input grad dX mismatch!"
    assert torch.allclose(my_bn.running_mean, torch_bn.running_mean, atol=1e-5), "BatchNorm2d running_mean mismatch!"
    assert torch.allclose(my_bn.running_var, torch_bn.running_var, atol=1e-5), "BatchNorm2d running_var mismatch!"
    
    my_bn.eval()
    torch_bn.eval()

    eval_x = torch.randn(2, c, 6, 6)
    assert torch.allclose(my_bn(eval_x), torch_bn(eval_x), atol=1e-5), "BatchNorm2d eval forward mismatch!"
    
def test_layernorm_parity():
    torch.manual_seed(42)
    shape = (4, 16)
    x_custom = torch.randn(shape)
    x_torch = x_custom.clone().detach().requires_grad_(True)

    my_ln = LayerNorm(16)
    torch_ln = nn.LayerNorm(16)

    out_custom = my_ln(x_custom)
    out_torch = torch_ln(x_torch)
    assert torch.allclose(out_custom, out_torch, atol=1e-5), "LayerNorm forward mismatch!"

    grad_out = torch.randn_like(out_custom)
    dx_custom = my_ln.backward(grad_out)
    out_torch.backward(grad_out)

    assert torch.allclose(my_ln.gamma.grad, torch_ln.weight.grad, atol=1e-5), "LayerNorm gamma grad mismatch!"
    assert torch.allclose(my_ln.beta.grad, torch_ln.bias.grad, atol=1e-5), "LayerNorm beta grad mismatch!"
    assert torch.allclose(dx_custom, x_torch.grad, atol=1e-5), "LayerNorm input grad dX mismatch!"


def test_instancenorm2d_parity():
    torch.manual_seed(42)
    n, c, h, w = 2, 4, 8, 8
    x_custom = torch.randn(n, c, h, w)
    x_torch = x_custom.clone().detach().requires_grad_(True)

    my_in = InstanceNorm2d(c)
    torch_in = nn.InstanceNorm2d(c, affine=True)

    out_custom = my_in(x_custom)
    out_torch = torch_in(x_torch)
    assert torch.allclose(out_custom, out_torch, atol=1e-5), "InstanceNorm2d forward mismatch!"

    grad_out = torch.randn_like(out_custom)
    dx_custom = my_in.backward(grad_out)
    out_torch.backward(grad_out)

    assert torch.allclose(my_in.gamma.grad.squeeze(), torch_in.weight.grad, atol=1e-5), "InstanceNorm2d gamma grad mismatch!"
    assert torch.allclose(my_in.beta.grad.squeeze(), torch_in.bias.grad, atol=1e-5), "InstanceNorm2d beta grad mismatch!"
    assert torch.allclose(dx_custom, x_torch.grad, atol=1e-5), "InstanceNorm2d input grad dX mismatch!"


def test_groupnorm_parity():
    torch.manual_seed(42)
    n, c, h, w = 2, 6, 8, 8
    num_groups = 2
    x_custom = torch.randn(n, c, h, w)
    x_torch = x_custom.clone().detach().requires_grad_(True)

    my_gn = GroupNorm(num_groups=num_groups, num_channels=c)
    torch_gn = nn.GroupNorm(num_groups=num_groups, num_channels=c)

    out_custom = my_gn(x_custom)
    out_torch = torch_gn(x_torch)
    assert torch.allclose(out_custom, out_torch, atol=1e-5), "GroupNorm forward mismatch!"

    grad_out = torch.randn_like(out_custom)
    dx_custom = my_gn.backward(grad_out)
    out_torch.backward(grad_out)

    assert torch.allclose(my_gn.gamma.grad.squeeze(), torch_gn.weight.grad, atol=1e-5), "GroupNorm gamma grad mismatch!"
    assert torch.allclose(my_gn.beta.grad.squeeze(), torch_gn.bias.grad, atol=1e-5), "GroupNorm beta grad mismatch!"
    assert torch.allclose(dx_custom, x_torch.grad, atol=1e-5), "GroupNorm input grad dX mismatch!"


if __name__ == "__main__":
    test_batchnorm2d_parity()
    test_layernorm_parity()
    test_instancenorm2d_parity()
    test_groupnorm_parity()
    print("All normalization parity tests (BatchNorm2d, LayerNorm, InstanceNorm2d, GroupNorm) passed successfully!")