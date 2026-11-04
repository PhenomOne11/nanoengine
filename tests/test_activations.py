import sys
from pathlib import Path

# Add project root (nanoengine/) to sys.path so 'src' imports resolve
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from src.layers.activations import ReLU, LeakyReLU

def test_relu_parity():
    torch.manual_seed(42)
    
    x_custom=torch.randn(8,16)
    x_torch=x_custom.clone().detach().requires_grad_(True)
    
    my_relu=ReLU()
    torch_relu=nn.ReLU()
    
    out_custom=my_relu(x_custom)
    out_torch=torch_relu(x_torch)
    assert torch.allclose(out_custom, out_torch, atol=1e-5), "ReLU forward outputs do not match!"
    
    grad_out=torch.randn_like(out_custom)
    dx_custom=my_relu.backward(grad_out)
    out_torch.backward(grad_out)
    
    assert torch.allclose(dx_custom, x_torch.grad, atol=1e-5), "ReLU backward input gradients do not match!"
    
    
def test_leaky_relu_parity():
    torch.manual_seed(42)
    slope=0.02
    
    x_custom=torch.randn(8,16)
    x_torch=x_custom.clone().detach().requires_grad_(True)
    
    my_leaky=LeakyReLU(negative_slope=slope)
    torch_leaky=nn.LeakyReLU(negative_slope=slope)
    
    out_custom=my_leaky(x_custom)
    out_torch=torch_leaky(x_torch)
    assert torch.allclose(out_custom, out_torch, atol=1e-5), "LeakyReLU forward outputs do not match!"
    
    grad_out=torch.randn_like(out_custom)
    dx_custom=my_leaky.backward(grad_out)
    out_torch.backward(grad_out)
    
    assert torch.allclose(dx_custom, x_torch.grad, atol=1e-5), "LeakyReLU backward input gradients do not match!"
    
if __name__ == "__main__":
    test_relu_parity()
    test_leaky_relu_parity()
    print("ReLU & LeakyReLU forward and backward parity tests passed successfully!")
    