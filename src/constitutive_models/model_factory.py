"""
Constitutive Model Factory — create elasticity models from config.

Extracted from run.py to enable programmatic reuse in ForwardEngine.
"""

import torch
from omegaconf import OmegaConf

from src.constitutive_models.physical_constitutive_models import (
    PhaseFieldElasticity,
    CorotatedPhaseFieldElasticity,
)


def create_elasticity_model(config: OmegaConf, device: torch.device):
    """
    Create and configure an elasticity model from config.

    Args:
        config: Full simulation configuration (needs config.material.*)
        device: PyTorch device

    Returns:
        Elasticity module (PhaseFieldElasticity or CorotatedPhaseFieldElasticity)
    """
    model_type = config.material.get("constitutive_model", "phase_field")

    Gc = float(config.material.get("Gc", 100.0))
    l0 = float(config.material.get("l0", 0.03))

    if model_type == "corotated_phase_field":
        elasticity = CorotatedPhaseFieldElasticity(Gc=Gc, l0=l0).to(device)
    else:
        elasticity = PhaseFieldElasticity().to(device)
        elasticity.Gc = Gc
        elasticity.l0 = l0

    # Override buffer values with config parameters
    elasticity.log_E = torch.log(torch.tensor(
        [config.material.youngs_modulus], device=device))
    elasticity.nu = torch.tensor(
        [config.material.poissons_ratio], device=device)

    # Set damage degradation exponent
    elasticity.damage_exp = config.material.degradation_exponent
    elasticity.tangential_damage_scale = float(
        config.material.get("tangential_damage_scale", 0.35)
    )

    print(f"  - Model: {model_type}")
    print(f"  - Young's modulus: {config.material.youngs_modulus:.2e}")
    print(f"  - Poisson's ratio: {config.material.poissons_ratio}")
    print(f"  - Gc: {Gc:.1f}, l0: {l0:.4f}")
    print(f"  - Degradation exponent: {config.material.degradation_exponent}")
    print(f"  - Tangential damage scale: {elasticity.tangential_damage_scale:.2f}")

    return elasticity
