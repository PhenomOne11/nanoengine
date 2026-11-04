import torch

class SGD:
    def __init__(self,params, lr=1e-3, momentum=0.0, weight_decay=0.0, dampening=0.0):
        self.params=list(params)
        self.lr=lr
        self.momentum=momentum
        self.weight_decay=weight_decay
        self.dampening=dampening
        self.state={}
        
    def zero_grad(self):
        for p in self.params:
            if hasattr(p,"grad") and p.grad is not None:
                p.grad.zero_()
                
    def step(self):
        with torch.no_grad():
            for p in self.params:
                if not hasattr(p,"grad") or p.grad is None:
                    continue
                
                grad = p.grad() if callable(p.grad) else p.grad
                d_p=p.grad.clone().detach()
                
                if self.weight_decay!=0:
                    d_p=d_p.add(p, alpha=self.weight_decay)
                
                if self.momentum!=0:
                    if p not in self.state:
                        self.state[p]={"momentum_buffer": d_p.clone().detach()}
                    else:
                        buf=self.state[p]["momentum_buffer"]
                        buf.mul_(self.momentum).add_(d_p, alpha=1- self.dampening)
                    d_p=self.state[p]["momentum_buffer"]
                
                p.add_(d_p, alpha=-self.lr)