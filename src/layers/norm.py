import torch
from src.layers.base import Module

class BatchNorm2d(Module):
    def __init__(self,num_features, eps=1e-5,momentum=0.1):
        super().__init__()
        self.num_features=num_features
        self.eps=eps
        self.momentum=momentum
        
        self.gamma=self.register_parameter(torch.ones(1,num_features,1,1))
        self.beta=self.register_parameter(torch.zeros(1, num_features, 1,1))
        
        self.running_mean=torch.zeros(num_features)
        self.running_var=torch.ones(num_features)
        
        self.x_hat=None
        self.inv_std=None
        self.m=None
        
    def forward(self,x):
        n,c,h,w=x.shape
        self.m=n*h*w
        
        if self.training:
            mean=x.mean(dim=(0,2,3),keepdim=True)
            var=x.var(dim=(0,2,3),keepdim=True, unbiased=False)
            
            self.inv_std = 1.0 / torch.sqrt(var + self.eps)
            self.x_hat = (x - mean) * self.inv_std
            
            var_unbiased=x.var(dim=(0,2,3), unbiased=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_unbiased
        else:
            r_mean=self.running_mean.view(1,c,1,1)
            r_var=self.running_var.view(1,c,1,1)
            self.x_hat=(x - r_mean) / torch.sqrt(r_var + self.eps)
            
        return self.gamma * self.x_hat + self.beta
    
    def backward(self,grad_out):
        self.beta.grad=grad_out.sum(dim=(0,2,3), keepdim=True)
        self.gamma.grad=(grad_out*self.x_hat).sum(dim=(0,2,3), keepdim=True)
        
        if not self.training:
            r_var = self.running_var.view(1, self.num_features, 1, 1)
            return grad_out * self.gamma / torch.sqrt(r_var + self.eps)
        
        dxhat = grad_out * self.gamma
        sum_dxhat = dxhat.sum(dim=(0, 2, 3), keepdim=True)
        sum_dxhat_xhat = (dxhat * self.x_hat).sum(dim=(0, 2, 3), keepdim=True)

        dx = (self.inv_std / self.m) * (
            self.m * dxhat - sum_dxhat - self.x_hat * sum_dxhat_xhat
        )
        return dx
    
class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        
        self.gamma = self.register_parameter(torch.ones(self.normalized_shape))
        self.beta = self.register_parameter(torch.zeros(self.normalized_shape))
        
        self.x_hat = None
        self.inv_std = None
        self.dims = None
        self.m = None
        
    def forward(self,x):
        self.dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        self.m = 1
        for d in self.dims:
            self.m *= x.shape[d]
            
        mean = x.mean(dim=self.dims, keepdim=True)
        var = x.var(dim=self.dims, keepdim=True, unbiased=False)

        self.inv_std = 1.0 / torch.sqrt(var + self.eps)
        self.x_hat = (x - mean) * self.inv_std
        return self.gamma * self.x_hat + self.beta
    
    def backward(self, grad_out):
        param_dims = tuple(range(0, grad_out.ndim - len(self.normalized_shape)))

        if len(param_dims) > 0:
            self.beta.grad = grad_out.sum(dim=param_dims)
            self.gamma.grad = (grad_out * self.x_hat).sum(dim=param_dims)
        else:
            self.beta.grad = grad_out.clone()
            self.gamma.grad = (grad_out * self.x_hat).clone()
            
        dxhat = grad_out * self.gamma
        sum_dxhat = dxhat.sum(dim=self.dims, keepdim=True)
        sum_dxhat_xhat = (dxhat * self.x_hat).sum(dim=self.dims, keepdim=True)

        dx = (self.inv_std / self.m) * (
            self.m * dxhat - sum_dxhat - self.x_hat * sum_dxhat_xhat
        )
        return dx
    
class InstanceNorm2d(Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        self.gamma = self.register_parameter(torch.ones(1, num_features, 1, 1))
        self.beta = self.register_parameter(torch.zeros(1, num_features, 1, 1))

        self.x_hat = None
        self.inv_std = None
        self.m = None
        
    def forward(self,x):
        n, c, h, w = x.shape
        self.m = h * w

        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)

        self.inv_std = 1.0 / torch.sqrt(var + self.eps)
        self.x_hat = (x - mean) * self.inv_std
        return self.gamma * self.x_hat + self.beta
    
    def backward(self, grad_out):
        self.beta.grad = grad_out.sum(dim=(0, 2, 3), keepdim=True)
        self.gamma.grad = (grad_out * self.x_hat).sum(dim=(0, 2, 3), keepdim=True)

        dxhat = grad_out * self.gamma
        sum_dxhat = dxhat.sum(dim=(2, 3), keepdim=True)
        sum_dxhat_xhat = (dxhat * self.x_hat).sum(dim=(2, 3), keepdim=True)

        dx = (self.inv_std / self.m) * (
            self.m * dxhat - sum_dxhat - self.x_hat * sum_dxhat_xhat
        )
        return dx
    
class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        self.gamma = self.register_parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = self.register_parameter(torch.zeros(1, num_channels, 1, 1))

        self.x_hat = None
        self.inv_std = None
        self.m = None
        
    def forward(self, x):
        n, c, h, w = x.shape
        g = self.num_groups
        self.m = (c // g) * h * w
        
        x_grouped = x.view(n, g, c // g, h, w)
        mean = x_grouped.mean(dim=(2, 3, 4), keepdim=True)
        var = x_grouped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)

        inv_std = 1.0 / torch.sqrt(var + self.eps)
        x_hat_grouped = (x_grouped - mean) * inv_std

        self.x_hat = x_hat_grouped.view(n, c, h, w)
        self.inv_std = inv_std.repeat_interleave(c // g, dim=1).view(n, c, 1, 1)

        return self.gamma * self.x_hat + self.beta
    
    def backward(self, grad_out):
        n, c, h, w = grad_out.shape
        g = self.num_groups

        self.beta.grad = grad_out.sum(dim=(0, 2, 3), keepdim=True)
        self.gamma.grad = (grad_out * self.x_hat).sum(dim=(0, 2, 3), keepdim=True)

        dxhat = grad_out * self.gamma
        dxhat_grouped = dxhat.view(n, g, c // g, h, w)
        xhat_grouped = self.x_hat.view(n, g, c // g, h, w)

        sum_dxhat = dxhat_grouped.sum(dim=(2, 3, 4), keepdim=True)
        sum_dxhat_xhat = (dxhat_grouped * xhat_grouped).sum(dim=(2, 3, 4), keepdim=True)

        inv_std_grouped = self.inv_std.view(n, g, c // g, 1, 1).mean(dim=2, keepdim=True)
        dx_grouped = (inv_std_grouped / self.m) * (
            self.m * dxhat_grouped - sum_dxhat - xhat_grouped * sum_dxhat_xhat
        )
        return dx_grouped.view(n, c, h, w)
    
