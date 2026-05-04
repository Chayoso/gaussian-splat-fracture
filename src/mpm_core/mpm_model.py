from typing import *

import numpy as np
import torch
from torch import Tensor
from omegaconf import DictConfig

class MPMModel:
    def __init__(
        self, 
        sim_params: DictConfig,
        material_params: DictConfig, 
        init_pos: Tensor, 
        enable_train: bool=False,
        device: torch.device='cuda',
    ):
        # save simulation parameters
        self.num_grids: int = int(sim_params['num_grids'])
        self.dt: float = float(sim_params['dt'])
        self.gravity: Tensor = torch.tensor(sim_params['gravity'], device=device)
        self.boundary_condition: Optional[DictConfig] = sim_params.get('boundary_condition', None)
        
        self.dx: float = 1.0 / self.num_grids
        self.inv_dx: float = float(self.num_grids)
        
        self.clip_bound: float = float(sim_params.get('clip_bound', 0.5)) * self.dx
        self.damping = float(sim_params.get('damping', 1.0))
        assert self.clip_bound >= 0.0
        assert self.damping >= 0.0 and self.damping <= 1.0
        
        self.n_particles: int = init_pos.shape[0]
        self.init_pos: Tensor = init_pos.detach()
        
        self.center: np.ndarray = np.array(material_params['center'])
        self.size: np.ndarray = np.array(material_params['size'])
        self.vol: float = np.prod(self.size) / self.n_particles
        self.p_mass: float = material_params['rho'] * self.vol  # TODO: the mass can be non-constant.

        self.enable_train: bool = enable_train
        self.device: torch.device = device
        
        # init tensors
        num_grids = self.num_grids
        n_dim = 3 # 3D
        self.grid_mv = torch.empty((num_grids ** n_dim, n_dim), device=device)
        self.grid_m = torch.empty((num_grids ** n_dim,), device=device)
        grid_ranges = torch.arange(num_grids, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_ranges, grid_ranges, grid_ranges, indexing='ij')
        self.grid_x = torch.stack((grid_x, grid_y, grid_z), dim=-1).reshape(-1, 3).float() # (n_grid * n_grid * n_grid, 3)
        
        self.offset = torch.tensor([[i, j, k] for i in range(3) for j in range(3) for k in range(3)], device=device).float() # (27, 3)

        # bc
        self.pre_particle_process = []
        self.post_grid_process = []

        self.time = 0.0
        
        # Add particle_chunk attribute for laplacian computation
        self.particle_chunk = None        
        
    def reset(self) -> None:
        self.time = 0.0
    
    def __call__(self, x: Tensor, v: Tensor, C: Tensor, F: Tensor, stress: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.p2g2p(x, v, C, F, stress)
    
    def p2g2p(self, x: Tensor, v: Tensor, C: Tensor, F: Tensor, stress: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        
        # prepare constants
        dt = self.dt
        vol = self.vol
        p_mass = self.p_mass 
        dx = self.dx
        inv_dx = self.inv_dx 
        n_grids = self.num_grids
        n_particles = self.n_particles
        clip_bound = self.clip_bound
        
        # calculate temporary variables for both p2g and g2p (weight, dpos, index)
        px = x * inv_dx
        base = (px - 0.5).long() # (n_particles, 3)
        fx = px - base.float() # (n_particles, 3)
        
        w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1) ** 2,
                0.5 * (fx - 0.5) ** 2
        ]
        w = torch.stack(w, dim=-1) # (n_particles, 3, 3)
        w_e = torch.einsum('bi, bj, bk -> bijk', w[:, 0], w[:, 1], w[:, 2]) # (n_particles, 3, 3, 3)
        weight = w_e.reshape(-1, 27) # (n_particles, 27)
        
        dw = [
            fx - 1.5,
            -2.0 * (fx - 1.0),
            fx - 0.5
        ]
        dw = torch.stack(dw, dim=-1) # (n_particles, 3, 3)
        dweight = [
            torch.einsum('pi,pj,pk->pijk', dw[:, 0], w[:, 1], w[:, 2]),
            torch.einsum('pi,pj,pk->pijk', w[:, 0], dw[:, 1], w[:, 2]),
            torch.einsum('pi,pj,pk->pijk', w[:, 0], w[:, 1], dw[:, 2])
        ]
        dweight = inv_dx * torch.stack(dweight, dim=-1).reshape(-1, 27, 3) # (n_particles, 3, 3, 3, 3) -> (n_particles, 27, 3)
        
        dpos = (self.offset - fx.unsqueeze(1)) * dx # (n_particles, 27, 3)
        
        index = base.unsqueeze(1) + self.offset.unsqueeze(0).long() # (n_particles, 27, 3)
        index = (index[:, :, 0] * n_grids * n_grids + index[:, :, 1] * n_grids + index[:, :, 2]).reshape(-1) # (n_particles * 27)
        index = index.clamp(0, n_grids ** 3 - 1) # (n_particles * 27) TODO: simple clipping leads to some numerical problems, but it's acceptable for now.
        
        # zero grid
        self.grid_mv = torch.zeros_like(self.grid_mv)
        self.grid_m = torch.zeros_like(self.grid_m)
        
        # pre-particle operation
        for operation in self.pre_particle_process:
            operation(self, x, v)
        
        # p2g
        mv = -dt * vol * torch.einsum('bij, bkj -> bki', stress, dweight) +\
            p_mass * weight.unsqueeze(2) * (v.unsqueeze(1) + torch.einsum('bij, bkj -> bki', C, dpos)) # (n_particles, 3, 3), (n_particles, 27, 3) -> (n_particles, 27, 3)
        mv = mv.reshape(-1, 3) # (n_particles * 27, 3)
        
        m = weight * p_mass # (n_particles, 27)
        m = m.reshape(-1) # (n_particles * 27)
        
        self.grid_mv = self.grid_mv.index_add(dim=0, index=index, source=mv) # (n_grid * n_grid * n_grid, 3)
        self.grid_m = self.grid_m.index_add(dim=0, index=index, source=m) # (n_grid * n_grid * n_grid)        
        
        # grid update
        self.grid_update()
        
        # post-grid operation
        for operation in self.post_grid_process:
            operation(self)
        
        # g2p
        v = self.grid_mv.index_select(dim=0, index=index).reshape(-1, 27, 3) # (n_particles, 27, 3)
        C = torch.einsum('bij, bik -> bijk', v, dpos) # (n_particles, 27, 3), (n_particles, 27, 3) -> (n_particles, 27, 3, 3)
        new_F = torch.einsum('bij, bik -> bijk', v, dweight) # (n_particles, 27, 3), (n_particles, 27, 3) -> (n_particles, 27, 3, 3)
        
        v = (weight.unsqueeze(2) * v).sum(dim=1) # (n_particles, 3)
        C = (4.0 * inv_dx * inv_dx * weight.unsqueeze(2).unsqueeze(3) * C).sum(dim=1)# (n_particles, 3, 3)
        new_F = dt * new_F.sum(dim=1) # (n_particles, 3, 3)
        
        x = x + v * dt
        x = x.clamp(clip_bound, 1.0 - clip_bound)
        F = F + torch.bmm(new_F, F)
        F = F.clamp(-2.0, 2.0)
        self.time += dt

        return x, v, C, F

    @torch.no_grad()
    def p2g2p_subset(
        self,
        x: Tensor,
        v: Tensor,
        C: Tensor,
        F: Tensor,
        stress: Tensor,
        indices: Tensor,
    ) -> tuple:
        """Run p2g2p for a particle subset (one fragment). Does NOT advance self.time."""
        dt = self.dt
        vol = self.vol
        p_mass = self.p_mass
        dx = self.dx
        inv_dx = self.inv_dx
        n_grids = self.num_grids
        clip_bound = self.clip_bound

        x_s = x[indices]
        v_s = v[indices]
        C_s = C[indices]
        F_s = F[indices]
        stress_s = stress[indices]

        # Weight computation (identical to p2g2p lines 81-111)
        px = x_s * inv_dx
        base = (px - 0.5).long()
        fx = px - base.float()

        w = [0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1) ** 2,
             0.5 * (fx - 0.5) ** 2]
        w = torch.stack(w, dim=-1)
        w_e = torch.einsum('bi,bj,bk->bijk', w[:, 0], w[:, 1], w[:, 2])
        weight = w_e.reshape(-1, 27)

        dw = [fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5]
        dw = torch.stack(dw, dim=-1)
        dweight = [
            torch.einsum('pi,pj,pk->pijk', dw[:, 0], w[:, 1], w[:, 2]),
            torch.einsum('pi,pj,pk->pijk', w[:, 0], dw[:, 1], w[:, 2]),
            torch.einsum('pi,pj,pk->pijk', w[:, 0], w[:, 1], dw[:, 2])
        ]
        dweight = inv_dx * torch.stack(dweight, dim=-1).reshape(-1, 27, 3)

        dpos = (self.offset - fx.unsqueeze(1)) * dx

        index = base.unsqueeze(1) + self.offset.unsqueeze(0).long()
        index = (index[:, :, 0] * n_grids * n_grids
                 + index[:, :, 1] * n_grids
                 + index[:, :, 2]).reshape(-1)
        index = index.clamp(0, n_grids ** 3 - 1)

        # Zero grid
        self.grid_mv.zero_()
        self.grid_m.zero_()

        # p2g
        mv = (-dt * vol * torch.einsum('bij,bkj->bki', stress_s, dweight)
              + p_mass * weight.unsqueeze(2)
              * (v_s.unsqueeze(1) + torch.einsum('bij,bkj->bki', C_s, dpos)))
        mv = mv.reshape(-1, 3)
        m = (weight * p_mass).reshape(-1)

        self.grid_mv = self.grid_mv.index_add(0, index, mv)
        self.grid_m = self.grid_m.index_add(0, index, m)

        # Grid update + boundary conditions
        self.grid_update()
        for operation in self.post_grid_process:
            operation(self)

        # g2p
        v_g = self.grid_mv.index_select(0, index).reshape(-1, 27, 3)
        C_new = torch.einsum('bij,bik->bijk', v_g, dpos)
        new_F = torch.einsum('bij,bik->bijk', v_g, dweight)

        v_new = (weight.unsqueeze(2) * v_g).sum(dim=1)
        C_new = (4.0 * inv_dx * inv_dx
                 * weight.unsqueeze(2).unsqueeze(3) * C_new).sum(dim=1)
        new_F = dt * new_F.sum(dim=1)

        x_new = x_s + v_new * dt
        x_new = x_new.clamp(clip_bound, 1.0 - clip_bound)
        F_new = F_s + torch.bmm(new_F, F_s)
        F_new = F_new.clamp(-2.0, 2.0)

        # Write back to full arrays
        x[indices] = x_new
        v[indices] = v_new
        C[indices] = C_new
        F[indices] = F_new

        return x, v, C, F

    @torch.no_grad()
    def particle_gradient(self, x: Tensor, c: Tensor, bc: str = "neumann") -> Tensor:
        """
        Compute per-particle gradient of c via grid central difference:
          1) bin particle c to grid (weighted average)
          2) compute ∇c on grid with central differences
          3) gather back to particles by weights
        Returns:
          grad_p: (P, 3) per-particle gradient
        """
        # Force c to be 1D
        c = c.view(-1)

        n = self.num_grids
        dx = self.dx
        device = x.device
        P = x.shape[0]

        # 1) Bin to grid
        grid_c_sum = torch.zeros((n**3,), device=device, dtype=c.dtype)
        grid_w_sum = torch.zeros((n**3,), device=device, dtype=c.dtype)

        chunk = self.particle_chunk or P
        for i in range(0, P, chunk):
            j = min(i + chunk, P)
            weight, _, _, index = self._weight_and_index(x[i:j])
            w = weight  # (p,27)
            idx = index  # (p*27,)
            grid_c_sum.index_add_(0, idx, (w * c[i:j].unsqueeze(1)).reshape(-1))
            grid_w_sum.index_add_(0, idx, w.reshape(-1))

        grid_c = grid_c_sum / (grid_w_sum + 1e-12)  # (Nnodes,)

        # 2) Grid gradient (central difference)
        g = grid_c.view(n, n, n)  # (D,H,W)
        g5 = g.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

        if bc == "neumann":
            gp5 = torch.nn.functional.pad(g5, (1,1,1,1,1,1), mode="replicate")
        else:
            gp5 = torch.nn.functional.pad(g5, (1,1,1,1,1,1), mode="constant", value=0.0)

        gp = gp5[0, 0]  # (D+2, H+2, W+2)

        # Central difference: ∂c/∂x = (c[i+1] - c[i-1]) / (2*dx)
        grad_x = (gp[2:, 1:-1, 1:-1] - gp[:-2, 1:-1, 1:-1]) / (2.0 * dx)
        grad_y = (gp[1:-1, 2:, 1:-1] - gp[1:-1, :-2, 1:-1]) / (2.0 * dx)
        grad_z = (gp[1:-1, 1:-1, 2:] - gp[1:-1, 1:-1, :-2]) / (2.0 * dx)

        grad_grid = torch.stack([grad_x, grad_y, grad_z], dim=-1)  # (D,H,W,3)
        grad_flat = grad_grid.reshape(-1, 3)  # (n^3, 3)

        # 3) Gather back to particles
        grad_p = torch.empty((P, 3), device=device, dtype=c.dtype)
        for i in range(0, P, chunk):
            j = min(i + chunk, P)
            weight, _, _, index = self._weight_and_index(x[i:j])
            # Weighted average: grad_p[i] = Σ weight[i,k] * grad_flat[index[i,k]]
            grad_gather = grad_flat[index].view(-1, 27, 3)  # (p, 27, 3)
            grad_p[i:j] = (weight.unsqueeze(-1) * grad_gather).sum(dim=1)  # (p, 3)

        return grad_p

    @torch.no_grad()
    def particle_laplacian(self, x: Tensor, c: Tensor, bc: str = "neumann") -> Tensor:
        # 방어 코드: c를 무조건 1D로 강제
        c = c.view(-1)  # ✅ 항상 1D로 강제

        # 빠른 검증 체크 (한 번만 출력)
        if not hasattr(self, '_debug_printed'):
            print("c:", c.shape)   # -> torch.Size([P])
            self._debug_printed = True

        """
        Compute per-particle Laplacian of c via grid 6-neighbor stencil:
          1) bin particle c to grid (weighted average)
          2) compute ∆c on grid with replicate (Neumann) or zero (Dirichlet) padding
          3) gather back to particles by weights
        Returns:
          lap_p: (P,) per-particle Laplacian
        """
        n = self.num_grids
        dx = self.dx
        device = x.device
        P = x.shape[0]

        # 1) bin to grid
        grid_c_sum = torch.zeros((n**3,), device=device, dtype=c.dtype)
        grid_w_sum = torch.zeros((n**3,), device=device, dtype=c.dtype)

        chunk = self.particle_chunk or P
        for i in range(0, P, chunk):
            j = min(i + chunk, P)
            weight, _, _, index = self._weight_and_index(x[i:j])
            w = weight  # (p,27)
            idx = index  # (p*27,)
            grid_c_sum.index_add_(0, idx, (w * c[i:j].unsqueeze(1)).reshape(-1))
            grid_w_sum.index_add_(0, idx, w.reshape(-1))

        grid_c = grid_c_sum / (grid_w_sum + 1e-12)  # (Nnodes,)

        # 2) grid Laplacian (6-neighbor)
        g = grid_c.view(n, n, n)                # (D,H,W)
        g5 = g.unsqueeze(0).unsqueeze(0)        # (1,1,D,H,W) = (N,C,D,H,W)
        
        if bc == "neumann":
            # Neumann: 경계값 복제
            gp5 = torch.nn.functional.pad(g5, (1,1,1,1,1,1), mode="replicate")
        else:
            # Dirichlet(0): 상수 0 패딩
            gp5 = torch.nn.functional.pad(g5, (1,1,1,1,1,1), mode="constant", value=0.0)
        
        gp = gp5[0, 0]                          # (D+2, H+2, W+2)
        
        center = gp[1:-1,1:-1,1:-1]
        xp     = gp[2:  ,1:-1,1:-1]
        xm     = gp[ :-2,1:-1,1:-1]
        yp     = gp[1:-1,2:  ,1:-1]
        ym     = gp[1:-1, :-2,1:-1]
        zp     = gp[1:-1,1:-1,2:  ]
        zm     = gp[1:-1,1:-1, :-2]
        lap_grid = (xp + xm + yp + ym + zp + zm - 6.0*center) / (dx*dx)

        lap_flat = lap_grid.reshape(-1)

        # 3) gather back to particles
        lap_p = torch.empty((P,), device=device, dtype=c.dtype)
        for i in range(0, P, chunk):
            j = min(i + chunk, P)
            weight, _, _, index = self._weight_and_index(x[i:j])
            lap_p[i:j] = (weight * lap_flat[index].view(-1, 27)).sum(dim=1)
        return lap_p

    def _weight_and_index(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute weight functions and grid indices for particle-to-grid mapping.
        Returns:
            weight: (N, 27) weight functions
            dweight: (N, 27, 3) weight gradients
            dpos: (N, 27, 3) position differences
            index: (N*27,) flattened grid indices
        """
        px = x * self.inv_dx
        base = (px - 0.5).long()
        fx = px - base.float()
        
        w = [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1) ** 2,
            0.5 * (fx - 0.5) ** 2
        ]
        w = torch.stack(w, dim=-1)
        w_e = torch.einsum('bi, bj, bk -> bijk', w[:, 0], w[:, 1], w[:, 2])
        weight = w_e.reshape(-1, 27)
        
        dw = [
            fx - 1.5,
            -2.0 * (fx - 1.0),
            fx - 0.5
        ]
        dw = torch.stack(dw, dim=-1)
        dweight = [
            torch.einsum('pi,pj,pk->pijk', dw[:, 0], w[:, 1], w[:, 2]),
            torch.einsum('pi,pj,pk->pijk', w[:, 0], dw[:, 1], w[:, 2]),
            torch.einsum('pi,pj,pk->pijk', w[:, 0], w[:, 1], dw[:, 2])
        ]
        dweight = torch.stack(dweight, dim=-1)
        
        # ✅ dpos는 (offset - fx) * dx
        dx = self.dx
        dpos = (self.offset.unsqueeze(0) - fx.unsqueeze(1)) * dx   # (N, 27, 3)
        
        # ✅ 3D 인덱스 → 선형 인덱스 (반드시 LongTensor)
        #    base: (N,3), offset: (27,3)
        index3 = base.unsqueeze(1) + self.offset.long().unsqueeze(0)   # (N,27,3)
        # 경계 클램프(옵션)
        n = self.num_grids
        index3 = index3.clamp(0, n-1)
        
        # 선형화: i*n^2 + j*n + k
        index = (index3[..., 0]*n*n + index3[..., 1]*n + index3[..., 2]).reshape(-1)  # (N*27,)
        
        # 빠른 검증 체크 (한 번만 출력)
        if not hasattr(self, '_debug_printed_index'):
            print("index shape:", index.shape)  # -> torch.Size([p*27])
            self._debug_printed_index = True
        
        return weight, dweight, dpos, index

    def grid_update(self) -> None:
        selected_idx = self.grid_m > 1e-15
        self.grid_mv[selected_idx] = self.grid_mv[selected_idx] / (self.grid_m[selected_idx].unsqueeze(1))
        self.grid_mv = self.damping * (self.grid_mv + self.dt * self.gravity)

        # Absorbing boundary layer: exponential damping near grid edges
        # Prevents artificial stress wave reflection at boundaries
        n = self.num_grids
        absorb_width = max(3, int(n * 0.08))  # ~8% of grid on each side
        grid_coords = self.grid_x  # (n^3, 3), values in [0, n-1]
        # Distance to nearest boundary (in grid cells)
        dist_to_boundary = torch.min(grid_coords, (n - 1) - grid_coords).min(dim=1).values
        # Exponential damping: 1.0 in interior, ~0.0 at boundary
        absorb_mask = (dist_to_boundary < absorb_width).unsqueeze(1)
        if absorb_mask.any():
            alpha = (dist_to_boundary.unsqueeze(1) / absorb_width).clamp(0, 1)
            damping_factor = alpha ** 2  # quadratic falloff: 0 at edge, 1 at interior
            self.grid_mv = torch.where(absorb_mask, self.grid_mv * damping_factor, self.grid_mv)

        # Slip BC on the X/Y/Z domain walls: zero the wall-normal
        # velocity component for any grid cell within `slip_band` cells
        # of the boundary if its velocity points INTO the wall.  A
        # narrow 1-cell band leaves room for fast particles to jump
        # past the wall in a single substep (per-substep displacement
        # = v*dt can exceed dx for v_com_gain >= 35 m/s), so use a
        # 3-cell band by default to cover those cases.  Tangential
        # motion is preserved (slip, not stick).  Without this BC,
        # MPM's position-only clip_bound leaves the velocity intact
        # so particles get pinned to walls indefinitely.
        slip_band = max(1, int(getattr(self, "wall_slip_band_cells", 6)))
        # Slip BC on X/Y lateral walls (lo + hi) and Z roof (top).
        # The Z floor is handled separately just below using the
        # simulator-set `gravity_drop_ground_z` so the slip BC
        # position matches the actual game-floor position (otherwise
        # the viewer's auto-detected floor and the physics floor
        # disagree, and particles tunnel through to the domain bottom
        # before being stopped).
        for axis in (0, 1):
            v_axis = self.grid_mv[:, axis]
            wall_lo = grid_coords[:, axis] < float(slip_band)
            wall_hi = grid_coords[:, axis] > (n - 1) - float(slip_band)
            into_lo = wall_lo & (v_axis < 0.0)
            into_hi = wall_hi & (v_axis > 0.0)
            if bool(into_lo.any()):
                self.grid_mv[into_lo, axis] = 0.0
            if bool(into_hi.any()):
                self.grid_mv[into_hi, axis] = 0.0
        # Z roof
        v_z = self.grid_mv[:, 2]
        roof = grid_coords[:, 2] > (n - 1) - float(slip_band)
        into_roof = roof & (v_z > 0.0)
        if bool(into_roof.any()):
            self.grid_mv[into_roof, 2] = 0.0
        # Z floor: align with `gravity_drop_ground_z` (set by the
        # simulator at impact).  Slip BC fires ONLY for cells AT or
        # BELOW the ground level (idx <= ceil(ground_idx)).  A wider
        # band ABOVE ground would freeze particles in mid-air before
        # they reach the floor, since v_z=0 at any cell in the band
        # propagates to fragment particles via shape match.  A 2-cell
        # buffer is added so fast-falling particles that jump through
        # the ground cell in one substep are still caught.
        ground_z = float(getattr(self, "ground_z", 0.0))
        if ground_z > 0.0:
            ground_idx = ground_z * (n - 1)
            floor_band = ground_idx + 2.0  # tight buffer
            floor_mask = grid_coords[:, 2] < floor_band
            into_floor = floor_mask & (v_z < 0.0)
            if bool(into_floor.any()):
                self.grid_mv[into_floor, 2] = 0.0
        else:
            # Fall back to standard wall slip on Z lo (domain bottom)
            wall_lo_z = grid_coords[:, 2] < float(slip_band)
            into_lo_z = wall_lo_z & (v_z < 0.0)
            if bool(into_lo_z.any()):
                self.grid_mv[into_lo_z, 2] = 0.0
        
    def pre_p2g_operation(self) -> None:
        pass
    
    def post_grid_operation(self) -> None:
        pass