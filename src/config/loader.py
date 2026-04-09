"""
Configuration loading and override utilities.

Extracted from run.py to enable programmatic reuse in ForwardEngine.
"""

from pathlib import Path
from omegaconf import OmegaConf


# Maps simple parameter names to config paths for ForwardEngine.simulate()
PARAM_MAP = {
    "E": "material.youngs_modulus",
    "Gc": "material.Gc",
    "nu": "material.poissons_ratio",
    "density": "material.density",
    "l0": "material.l0",
    "num_frames": "rendering.total_frames",
    "gravity_z": "mpm.gravity.2",
}


def load_config(config_path: str) -> OmegaConf:
    """
    Load and validate YAML configuration.

    Args:
        config_path: Path to YAML config file

    Returns:
        OmegaConf configuration object
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)

    required_sections = [
        "simulation", "mesh", "particles", "mpm",
        "material", "phase_field", "gaussian_splatting", "rendering"
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    print(f"[Config] Loaded configuration from: {config_path}")
    print(f"  - Simulation: {config.simulation.name}")
    print(f"  - Mesh: {config.mesh.path}")
    print(f"  - Particles: {config.particles.target_count}")
    print(f"  - Frames: {config.rendering.total_frames}")

    return config


def apply_cli_overrides(config: OmegaConf, args) -> OmegaConf:
    """
    Apply command-line argument overrides to config.

    Args:
        config: Base configuration
        args: Parsed command-line arguments

    Returns:
        Modified configuration
    """
    if args.mesh is not None:
        config.mesh.path = args.mesh
        print(f"[Config] Override mesh: {args.mesh}")

    if args.output is not None:
        config.output.video_path = args.output
        print(f"[Config] Override output: {args.output}")

    if args.frames is not None:
        config.rendering.total_frames = args.frames
        print(f"[Config] Override frames: {args.frames}")

    if args.device is not None:
        config.device.type = args.device
        print(f"[Config] Override device: {args.device}")

    return config


def apply_overrides_dict(config: OmegaConf, overrides: dict) -> OmegaConf:
    """
    Apply parameter overrides from a dictionary.

    Used by ForwardEngine.simulate() to set material parameters programmatically.

    Args:
        config: Base configuration
        overrides: dict of {param_name: value}, e.g. {"E": 1e7, "Gc": 60000}
                   Keys can be PARAM_MAP short names or dotted config paths.

    Returns:
        Modified configuration
    """
    for key, value in overrides.items():
        if value is None:
            continue
        # Resolve short name to config path
        config_path = PARAM_MAP.get(key, key)
        OmegaConf.update(config, config_path, value)

    return config
