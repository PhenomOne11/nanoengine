from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from src.layers.activations import ReLU
from src.layers.containers import Sequential
from src.layers.conv import Conv2d


def test_sequential_network_parity():
  torch.manual_seed(42)

  x_custom = torch.randn(2, 1, 6, 6)
  x_torch = x_custom.clone().detach().requires_grad_(True)

  c1 = Conv2d(1, 4, kernel_size=3, padding=1)
  a1 = ReLU()
  c2 = Conv2d(4, 2, kernel_size=3, padding=0)

  my_model = Sequential(c1, a1, c2)

  torch_c1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
  torch_a1 = nn.ReLU()
  torch_c2 = nn.Conv2d(4, 2, kernel_size=3, padding=0)

  with torch.no_grad():
    torch_c1.weight.copy_(c1.w)
    torch_c1.bias.copy_(c1.b)
    torch_c2.weight.copy_(c2.w)
    torch_c2.bias.copy_(c2.b)

  torch_model = nn.Sequential(torch_c1, torch_a1, torch_c2)

  out_custom = my_model(x_custom)
  out_torch = torch_model(x_torch)
  assert torch.allclose(
      out_custom, out_torch, atol=1e-5
  ), "Sequential network forward output mismatch!"

  grad_out = torch.randn_like(out_custom)
  dx_custom = my_model.backward(grad_out)
  out_torch.backward(grad_out)

  assert torch.allclose(
      c1.w.grad, torch_c1.weight.grad, atol=1e-5
  ), "Conv1 weight grad mismatch!"
  assert torch.allclose(
      c1.b.grad, torch_c1.bias.grad, atol=1e-5
  ), "Conv1 bias grad mismatch!"
  assert torch.allclose(
      c2.w.grad, torch_c2.weight.grad, atol=1e-5
  ), "Conv2 weight grad mismatch!"
  assert torch.allclose(
      c2.b.grad, torch_c2.bias.grad, atol=1e-5
  ), "Conv2 bias grad mismatch!"
  assert torch.allclose(
      dx_custom, x_torch.grad, atol=1e-5
  ), "Input grad (dX) mismatch!"


if __name__ == "__main__":
  test_sequential_network_parity()
  print("Sequential multi-layer pipeline passed all parity tests successfully!")