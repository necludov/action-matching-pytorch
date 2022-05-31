import math
import numpy as np
import torch
import torch.nn.functional as F


class Normal:
    def __init__(self, mu, sigma):
        self.sigma = sigma
        self.mu = mu
        assert self.mu.dim() == 1
        self.dim = self.mu.shape[0]
        self.device = mu.device

    def log_prob(self, x):
        assert (x.shape[1] == self.dim)
        return -0.5 * torch.sum((x - self.mu.view([1,-1])) ** 2, dim=1) / self.sigma ** 2 

    def sample(self, n):
        eps = torch.empty([n, self.dim], device=self.device).normal_()
        return self.mu.view([1,-1]) + self.sigma * eps
    
    
class Laplace:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        assert self.mu.dim() == 1
        assert self.sigma.dim() == 0
        self.dim = self.mu.shape[0]
        self.device = mu.device
        self.mu = self.mu.view(1,self.dim)
        
        self.S_n = 2*math.pow(math.pi,0.5*self.dim)/math.gamma(0.5*self.dim)
        
        def F(t, n):
            if n == 1:
                return 1-torch.exp(-t)
            return -t**(n-1)*torch.exp(-t) + (n-1)*F(t,n-1)

        self.r = torch.linspace(0,15,2000).to(self.device).view(-1,1) # 15 set specifically for dim = 3
        self.r = torch.vstack([self.r, torch.tensor(30.).to(self.device)])
        self.CDF = (F(self.r,self.dim)/math.factorial(self.dim-1))
        
    def inv_CDF(self, u):
        assert u.dim() == 1
        u = u.view(-1,1)
        ids = (u > self.CDF.T).sum(1)
        return (u-self.CDF[ids-1])*(self.r[ids] - self.r[ids-1])/(self.CDF[ids] - self.CDF[ids-1]) + self.r[ids-1]
    
    def radial_prob(self, r):
        return r**(self.dim-1)*torch.exp(-r/self.sigma)/math.factorial(self.dim-1)/self.sigma**self.dim

    def log_prob(self, x):
        assert (x.shape[1] == self.dim)
        output = -(x - self.mu).norm(dim=1, keepdim=True)/self.sigma - self.dim*torch.log(self.sigma)
        output = output - math.log(self.S_n) - math.log(math.factorial(self.dim-1))
        return output

    def sample(self, n):
        angles = torch.empty([n, self.dim], device=self.device).normal_()
        angles = angles/angles.norm(dim=1, keepdim=True)
        r = self.inv_CDF(torch.empty(n, device=self.device).uniform_())
        return self.mu + angles*r*self.sigma
