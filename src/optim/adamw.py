import torch
import math

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_num = 0
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.step_num += 1
        bias_correction1 = 1.0 - (self.beta1 ** self.step_num)
        bias_correction2 = 1.0 - (self.beta2 ** self.step_num)

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad.data
            if self.weight_decay != 0:
                p.data.mul_(1.0 - self.lr * self.weight_decay)
            self.m[i].mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            denom = (self.v[i].sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
            step_size = self.lr / bias_correction1
            p.data.addcdiv_(self.m[i], denom, value=-step_size)
