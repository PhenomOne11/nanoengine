from src.layers.base import Module

class Sequential(Module):
    
    def __init__(self,*layers):
        super().__init__()
        self.layers=list(layers)
        
    def forward(self,x):
        for layers in self.layers:
            x=layers(x)
        return x
        
    def backward(self, grad_out):
        for layer in reversed(self.layers):
            grad_out=layer.backward(grad_out)
        return grad_out
        
    def parameters(self):
        params=[]
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
        
        