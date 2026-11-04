import math

class CosineAnnealingLR:
    def __init__(self,optimizer,T_max,eta_min=0.0):
        self.optimizer=optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.step_count = 0
        
    def step(self):
        self.step_count+=1
        cos_val = math.cos(math.pi * self.step_count / self.T_max)
        new_lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + cos_val)
        self.optimizer.lr = new_lr
        return new_lr
    