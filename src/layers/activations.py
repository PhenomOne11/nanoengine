import torch
from src.layers.base import Module

class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.inp=None
        
    def forward(self,inp):
        self.inp=inp
        return inp.clamp_min(0.0)
    
    def backward(self, grad_out):
        mask=(self.inp>0).float()
        return mask*grad_out
    

class LeakyReLU(Module):
    def __init__(self,negative_slope=0.01):
        super().__init__()
        self.negative_slope=float(negative_slope)
        self.inp=None
    
    def forward(self,inp):
        self.inp=inp
        return torch.where(inp>0,inp,inp*self.negative_slope)
    
    def backward(self,grad_out):
        mask=torch.where(self.inp>0,1.0,self.negative_slope)
        return mask*grad_out
    