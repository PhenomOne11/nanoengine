import sys
from pathlib import Path

# Adds the project root (nanoengine/) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from src.layers.linear import Linear


def test_linear_forward_backward():
    torch.manual_seed(42)
    
    batch_size=8
    in_features=16
    out_features=4
    
    x_custom= torch.randn(batch_size, in_features)
    x_torch=x_custom.clone().detach().requires_grad_(True)
    
    my_linear=Linear(in_features,out_features)
    torch_linear=nn.Linear(in_features,out_features)
    
    with torch.no_grad():
        torch_linear.weight.copy_(my_linear.w.t())
        torch_linear.bias.copy_(my_linear.b)
        
    out_custom=my_linear(x_custom)
    out_torch=torch_linear(x_torch)
    
    assert torch.allclose(out_custom,out_torch, atol=1e-5), "Forward outputs do not match!"
    
    grad_out=torch.rand_like(out_custom)
    
    dx_custom=my_linear.backward(grad_out)
    out_torch.backward(grad_out)
    
    assert torch.allclose(my_linear.w.grad, torch_linear.weight.grad.t(), atol=1e-5), "Weight gradients do not match!"
    assert torch.allclose(my_linear.b.grad, torch_linear.bias.grad, atol=1e-5), "Bias gradients do not match!"
    assert torch.allclose(dx_custom, x_torch.grad, atol=1e-5), "Input gradients do not match!"
    
if __name__ == "__main__":
    test_linear_forward_backward()
    print("Linear forward & backward parity tests passed successfully!")