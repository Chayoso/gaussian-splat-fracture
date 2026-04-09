import math
from typing import *
import torch
import torch.nn as nn
from torch import Tensor

from .abstract import Plasticity

class DruckerPragerPlasticity(Plasticity):
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([2.0e6]).log())
        self.register_buffer('nu', torch.Tensor([0.4]))
        self.register_buffer('friction_angle', torch.Tensor([25.0]))
        self.register_buffer('cohesion', torch.Tensor([0.0]))


    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:

        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()  # Fixed: use the passed log_E parameter
        if nu is None:
            nu = self.nu
        else:
            nu = nu  # Use the passed nu parameter
            
        friction_angle = self.friction_angle
        sin_phi = torch.sin(torch.deg2rad(friction_angle))
        alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
        cohesion = self.cohesion

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        if mu.dim() != 0:
            mu = mu.reshape(-1, 1)
            
        if la.dim() != 0:
            la = la.reshape(-1, 1)

        # warp svd
        U, sigma, Vh = self.svd(F)

        # prevent NaN
        threshold = 5e-2
        sigma = torch.clamp_min(sigma, threshold)

        epsilon = torch.log(sigma)
        trace = epsilon.sum(dim=1, keepdim=True)
        epsilon_hat = epsilon - trace / self.dim
        epsilon_hat_norm = torch.linalg.norm(epsilon_hat, dim=1, keepdim=True)
        epsilon_hat_norm = torch.clamp_min(epsilon_hat_norm, 1e-10) # avoid nan
        expand_epsilon = torch.ones_like(epsilon) * cohesion

        shifted_trace = trace - cohesion * self.dim
        cond_yield = (shifted_trace < 0).view(-1, 1)

        delta_gamma = epsilon_hat_norm + (self.dim * la + 2 * mu) / (2 * mu) * shifted_trace * alpha
        compress_epsilon = epsilon - (torch.clamp_min(delta_gamma, 0.0) / epsilon_hat_norm) * epsilon_hat

        # 🛡️ FIX: When cond_yield is false, don't change epsilon to cohesion constant, keep original epsilon
        epsilon = torch.where(cond_yield, compress_epsilon, epsilon)  # ✅ Keep original epsilon

        F = torch.matmul(torch.matmul(U, torch.diag_embed(epsilon.exp())), Vh)

        return F
    
    def name(self):
        return "DruckerPragerPlasticity"
    
class IdentityPlasticity(Plasticity):
    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        return F
    
    def name(self):
        return "IdentityPlasticity"
    
    
class SigmaPlasticity(Plasticity):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:
        J = torch.det(F)

        # unilateral incompressibility: https://github.com/penn-graphics-research/ziran2020/blob/master/Lib/Ziran/Physics/PlasticityApplier.cpp#L1084
        J = torch.clamp(J, min=0.05, max=1.2)

        # 🛡️ FIX: Don't change F to diag(J^1/3), only adjust volume
        # Instead of replacing F with diag(J^1/3), only adjust volume while preserving rotation/shear
        J_target = torch.pow(J, 1.0 / 3.0).view(-1, 1, 1)
        F_vol_adjusted = F * J_target / torch.clamp_min(J, 1e-8)
        
        return F_vol_adjusted
    
    def name(self):
        return "SigmaPlasticity"
    
    
class VonMisesPlasticity(Plasticity):
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer('log_E', torch.Tensor([2.0e6]).log())
        self.register_buffer('nu', torch.Tensor([0.4]))
        self.register_buffer('sigma_y', torch.Tensor([1.0e3]))


    def forward(self, F: Tensor, log_E: Optional[Tensor]=None, nu: Optional[Tensor]=None) -> Tensor:

        if log_E is None:
            E = self.log_E.exp()
        else:
            E = log_E.exp()
        if nu is None:
            nu = self.nu
            
        sigma_y = self.sigma_y

        mu = E / (2 * (1 + nu))
        if mu.dim() != 0:
            mu = mu.reshape(-1, 1)
        # warp svd
        U, sigma, Vh = self.svd(F)

        # prevent NaN
        threshold = 5e-2
        sigma = torch.clamp_min(sigma, threshold)

        epsilon = torch.log(sigma)
        trace = epsilon.sum(dim=1, keepdim=True)
        epsilon_hat = epsilon - trace / self.dim
        epsilon_hat_norm = torch.linalg.norm(epsilon_hat, dim=1, keepdim=True)
        epsilon_hat_norm = torch.clamp_min(epsilon_hat_norm, 1e-10) # avoid nan
        
        # 🛡️ FIX: Complete VonMisesPlasticity implementation
        # Von Mises yield criterion: sqrt(3/2 * ||dev(ε)||) <= σ_y / (2μ)
        delta_gamma = epsilon_hat_norm - sigma_y / (2 * mu)
        cond_yield = (delta_gamma > 0).view(-1, 1, 1)

        # Plastic correction: ε_plastic = ε - Δγ * dev(ε) / ||dev(ε)||
        yield_epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        yield_F = torch.matmul(torch.matmul(U, torch.diag_embed(yield_epsilon.exp())), Vh)

        # Apply plastic correction only where yielding occurs
        F = torch.where(cond_yield, yield_F, F)

        return F
    
    def name(self):
        return "VonMisesPlasticity"