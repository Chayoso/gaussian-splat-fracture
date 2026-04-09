
from typing import *
import torch
from torch import Tensor

# Import SoftGate for adaptive activation gating
try:
    from .soft_gate import SoftGate, calibrate_threshold, hard_mask_by_threshold, hysteresis_update
except ImportError:
    # Fallback if soft_gate module is not available
    SoftGate = None
    calibrate_threshold = None
    hard_mask_by_threshold = None
    hysteresis_update = None

# Global SoftGate instance: maintains EMA state across frames
_gate = None

@torch.no_grad()
def update_phase_field(
    elasticity_module,
    F: Tensor,
    e_cat: Optional[Tensor],
    c: Tensor,
    num_grids: int,
    dt: float,
    lap: Optional[Tensor] = None,
    warmup_frames: int = 5,
    dC_max: float = 0.01,
    **kwargs,  # absorb legacy params gracefully
):
    """
    AT2 phase field fracture — staggered scheme (explicit).

    Governing equation:
        (1 + 2 l₀ H / Gc) c  −  l₀² ∇²c  =  2 l₀ H / Gc

    Explicit one-step update:
        c_eq = (H_ratio + l0² · lap_c) / (1 + H_ratio)
    where H_ratio = 2 l₀ H / Gc.

    Crack propagation mechanism:
        1) Damage degrades stiffness → stress concentrates at crack tip
        2) Stress concentration → H increases at crack tip
        3) l₀² ∇²c diffuses damage into neighboring undamaged material
        4) Irreversibility: c can only increase
    """
    device, dtype = F.device, F.dtype
    N = F.shape[0]

    # --- Material parameters from elasticity module ---
    Gc = getattr(elasticity_module, 'Gc', 50.0)
    l0 = getattr(elasticity_module, 'l0', 0.03)

    # --- 1) Compute tension energy ψ⁺ ---
    if hasattr(elasticity_module, 'tension_energy_density'):
        psi = elasticity_module.tension_energy_density(F).to(device=device, dtype=dtype)
    elif hasattr(elasticity_module, 'energy_density'):
        psi = elasticity_module.energy_density(F, e_cat).to(device=device, dtype=dtype)
    else:
        I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        C = F.transpose(1, 2) @ F
        Egl = 0.5 * (C - I)
        psi = (Egl * Egl).sum(dim=(1, 2))
    psi = torch.clamp(psi, min=0.0, max=1e8)  # Safety clamp only (not Gc-dependent)

    # --- 2) Warmup: skip H accumulation to avoid storing transient spikes ---
    if not hasattr(elasticity_module, "_frame_count"):
        elasticity_module._frame_count = 0
    elasticity_module._frame_count += 1

    if elasticity_module._frame_count <= warmup_frames:
        print(f"[PF warmup {elasticity_module._frame_count}/{warmup_frames}] "
              f"psi_max={psi.max().item():.2e}")
        return c

    # --- 3) History variable H (irreversible max of ψ⁺) — after warmup ---
    if not hasattr(elasticity_module, "_history_H"):
        elasticity_module._history_H = psi.clone()
    else:
        elasticity_module._history_H = torch.maximum(elasticity_module._history_H, psi)
    H = elasticity_module._history_H

    # --- 4) AT2 equilibrium solve (explicit) ---
    H_ratio = 2.0 * l0 * H / Gc  # (N,) dimensionless driving force

    if lap is None:
        lap = torch.zeros_like(c)

    # c_eq = (H_ratio + l0² · ∇²c) / (1 + H_ratio)
    # The l0² · ∇²c term propagates crack from damaged neighbors into undamaged regions
    c_eq = (H_ratio + l0 * l0 * lap) / (1.0 + H_ratio)
    c_eq = torch.clamp(c_eq, 0.0, 1.0)

    # --- 5) Rate limiter: prevent explosive growth ---
    dc = c_eq - c
    dc = torch.clamp(dc, 0.0, dC_max)  # only growth (irreversibility), capped per step
    c_new = c + dc

    # --- 6) Diagnostics (compact) ---
    n_growing = (dc > 1e-6).sum().item()
    lap_term = l0 * l0 * lap
    print(f"[AT2] psi_max={psi.max().item():.2e} H_max={H.max().item():.2e} "
          f"Gc={Gc:.1f} l0={l0:.4f} "
          f"c_max={c_new.max().item():.4f} c_mean={c_new.mean().item():.4f} "
          f"growing={n_growing}/{N} "
          f"lap_term=[{lap_term.min().item():.2f},{lap_term.max().item():.2f}]")

    return c_new


# ============================================================================
# Utility functions for SoftGate inference mode
# ============================================================================

def switch_to_inference_mode(elasticity_module, val_loader, model_psi, target=0.01, device="cuda"):
    """
    Switch SoftGate from training mode to inference mode
    
    This function calibrates a fixed threshold T* using validation data
    and switches the gate to use hard thresholding with hysteresis.
    
    Args:
        elasticity_module: The elasticity module containing SoftGate
        val_loader: Validation data loader for calibration
        model_psi: Function that computes psi from batch
        target: Target activation percentage
        device: Device to run on
        
    Returns:
        T_star: Calibrated threshold for inference
    """
    global _gate
    
    if _gate is None:
        print("Warning: SoftGate not initialized, cannot switch to inference mode")
        return None
    
    if calibrate_threshold is None:
        print("Warning: calibrate_threshold function not available")
        return None
    
    print("Calibrating SoftGate threshold for inference mode...")
    T_star = calibrate_threshold(model_psi, val_loader, target=target, device=device)
    
    # Store T* in the global gate instance for inference mode
    _gate._T_star = T_star
    _gate._inference_mode = True
    
    # Ensure T_star is a scalar for printing
    T_star_val = T_star
    if isinstance(T_star_val, torch.Tensor):
        T_star_val = T_star_val.item()
    print(f"Inference mode activated: T* = {T_star_val:.3e}")
    return T_star


def get_gate_statistics():
    """
    Get current SoftGate statistics for logging/monitoring
    
    Returns:
        dict: Gate statistics if available, None otherwise
    """
    global _gate
    
    if _gate is None:
        return None
    
    stats = {
        "target": _gate.target,
        "T_ema": _gate.T_ema.item() if hasattr(_gate, 'T_ema') and hasattr(_gate.T_ema, 'item') else None,
        "step": _gate.step.item() if hasattr(_gate, 'step') and hasattr(_gate.step, 'item') else None,
        "tau_current": _gate._anneal_tau() if hasattr(_gate, '_anneal_tau') else None,
        "inference_mode": getattr(_gate, '_inference_mode', False)
    }
    
    if stats["inference_mode"]:
        stats["T_star"] = getattr(_gate, '_T_star', None)
    
    return stats

