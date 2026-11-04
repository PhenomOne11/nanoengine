import sys
from pathlib import Path

# Add project root (nanoengine/) to sys.path so 'src' imports resolve
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from src.layers.conv import Conv2d

def test_conv2d_parity():
    torch.manual_seed(42)
    
    n,c_in,c_out=2,3,4
    h_in,w_in=8,8
    kernel_size=3
    stride=1
    padding=1
    
    x_custom=torch.randn(n, c_in,h_in,w_in)
    x_torch=x_custom.clone().detach().requires_grad_(True)
    
    my_conv=Conv2d(c_in,c_out,kernel_size,stride=stride,padding=padding)
    torch_conv=nn.Conv2d(c_in,c_out,kernel_size,stride=stride,padding=padding)
    
    with torch.no_grad():
        torch_conv.weight.copy_(my_conv.w)
        torch_conv.bias.copy_(my_conv.b)
        
    out_custom=my_conv(x_custom)
    out_torch=torch_conv(x_torch)
    assert torch.allclose(out_custom, out_torch, atol=1e-5), "Conv2d forward outputs do not match!"
    
    grad_out=torch.randn_like(out_custom)
    
    dx_custom=my_conv.backward(grad_out)
    out_torch.backward(grad_out)
    
    assert torch.allclose(my_conv.w.grad, torch_conv.weight.grad, atol=1e-5), "Conv2d weight gradients do not match!"
    assert torch.allclose(my_conv.b.grad, torch_conv.bias.grad, atol=1e-5), "Conv2d bias gradients do not match!"
    assert torch.allclose(dx_custom, x_torch.grad, atol=1e-5), "Conv2d input gradients (dX) do not match!"
    
if __name__ == "__main__":
    test_conv2d_parity()
    print("Conv2d forward & backward parity tests passed successfully!")