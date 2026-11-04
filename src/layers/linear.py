import torch
import math
from src.layers.base import Module

class Linear(Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.w=self.register_parameter(torch.randn(in_features,out_features)* math.sqrt(1/in_features))
        self.b=self.register_parameter(torch.randn(out_features))
        
    def forward(self,inp):
        self.inp=inp
        out=inp@self.w+self.b
        return out
    
    def backward(self,grad_out):
        self.w.grad=self.inp.t() @ grad_out
        self.b.grad=grad_out.sum(0)
        return grad_out@self.w.t()