import torch
import torch.nn as nn

class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def register_parameter(self, name_or_tensor, tensor=None):
        if tensor is None:
            tensor = name_or_tensor
            name = f"param_{len(list(self.parameters()))}"
        else:
            name = name_or_tensor

        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=torch.float32)

        param = nn.Parameter(tensor)
        super().register_parameter(name, param)
        return param
