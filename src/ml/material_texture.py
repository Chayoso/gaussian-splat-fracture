"""
Procedural material textures for Gaussian Splats.

Generates per-Gaussian colors based on 3D position + material type.
Applied after Gaussian creation, before simulation starts.
"""

import torch
import numpy as np


def _perlin_noise_3d(positions: np.ndarray, scale: float = 10.0, seed: int = 0) -> np.ndarray:
    """Simple 3D value noise (not true Perlin, but fast approximation).

    Args:
        positions: (N, 3) positions
        scale: noise frequency
        seed: random seed

    Returns:
        (N,) noise values in [0, 1]
    """
    rng = np.random.RandomState(seed)
    # Hash-based noise: floor positions, hash to random values, interpolate
    p = positions * scale
    pi = np.floor(p).astype(int)
    pf = p - pi  # fractional part

    # Smooth interpolation weights
    u = pf * pf * (3 - 2 * pf)

    # Random values at 8 corners of each cell
    def hash_3d(x, y, z):
        h = (x * 73856093 ^ y * 19349663 ^ z * 83492791 + seed) % 1000000
        return (h / 1000000.0).astype(np.float32)

    c000 = hash_3d(pi[:, 0], pi[:, 1], pi[:, 2])
    c001 = hash_3d(pi[:, 0], pi[:, 1], pi[:, 2] + 1)
    c010 = hash_3d(pi[:, 0], pi[:, 1] + 1, pi[:, 2])
    c011 = hash_3d(pi[:, 0], pi[:, 1] + 1, pi[:, 2] + 1)
    c100 = hash_3d(pi[:, 0] + 1, pi[:, 1], pi[:, 2])
    c101 = hash_3d(pi[:, 0] + 1, pi[:, 1], pi[:, 2] + 1)
    c110 = hash_3d(pi[:, 0] + 1, pi[:, 1] + 1, pi[:, 2])
    c111 = hash_3d(pi[:, 0] + 1, pi[:, 1] + 1, pi[:, 2] + 1)

    # Trilinear interpolation
    c00 = c000 * (1 - u[:, 0]) + c100 * u[:, 0]
    c01 = c001 * (1 - u[:, 0]) + c101 * u[:, 0]
    c10 = c010 * (1 - u[:, 0]) + c110 * u[:, 0]
    c11 = c011 * (1 - u[:, 0]) + c111 * u[:, 0]
    c0 = c00 * (1 - u[:, 1]) + c10 * u[:, 1]
    c1 = c01 * (1 - u[:, 1]) + c11 * u[:, 1]
    return c0 * (1 - u[:, 2]) + c1 * u[:, 2]


def _fbm_noise(positions: np.ndarray, octaves: int = 4, scale: float = 8.0, seed: int = 0) -> np.ndarray:
    """Fractal Brownian Motion noise (multi-octave)."""
    result = np.zeros(len(positions))
    amplitude = 1.0
    freq = scale
    for i in range(octaves):
        result += amplitude * _perlin_noise_3d(positions, freq, seed + i * 17)
        amplitude *= 0.5
        freq *= 2.0
    return result / (2.0 - 2.0 ** (1 - octaves))  # normalize to [0, 1]


TEXTURE_FUNCTIONS = {}


def register_texture(name):
    def decorator(fn):
        TEXTURE_FUNCTIONS[name] = fn
        return fn
    return decorator


@register_texture("ice")
def texture_ice(positions: np.ndarray) -> np.ndarray:
    """Ice: translucent blue-white with internal cracks/veins.

    Returns: (N, 3) RGB colors
    """
    N = len(positions)

    # Base: light blue-white
    base_r = np.full(N, 0.75)
    base_g = np.full(N, 0.85)
    base_b = np.full(N, 0.95)

    # Subtle internal veins (high-frequency noise)
    vein = _fbm_noise(positions, octaves=5, scale=15.0, seed=42)
    vein_mask = (vein > 0.6).astype(np.float32) * 0.15

    # Depth-based translucency (center = slightly darker blue)
    center = positions.mean(axis=0)
    depth = np.linalg.norm(positions - center, axis=1)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    r = base_r - vein_mask * 0.1 - (1 - depth_norm) * 0.05
    g = base_g - vein_mask * 0.05
    b = base_b + vein_mask * 0.05

    # Subtle sparkle (random bright spots)
    sparkle = _perlin_noise_3d(positions, scale=50.0, seed=99)
    bright = (sparkle > 0.92).astype(np.float32) * 0.1
    r += bright
    g += bright
    b += bright

    return np.stack([np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)], axis=1)


@register_texture("ceramic")
def texture_ceramic(positions: np.ndarray) -> np.ndarray:
    """Ceramic: smooth cream/white with subtle glaze variation.

    Returns: (N, 3) RGB colors
    """
    N = len(positions)

    # Base: warm cream
    base_r = np.full(N, 0.92)
    base_g = np.full(N, 0.88)
    base_b = np.full(N, 0.82)

    # Subtle glaze variation (low-frequency)
    glaze = _fbm_noise(positions, octaves=3, scale=6.0, seed=7)

    r = base_r + (glaze - 0.5) * 0.08
    g = base_g + (glaze - 0.5) * 0.06
    b = base_b + (glaze - 0.5) * 0.04

    # Occasional tiny dark speckles (kiln spots)
    speckle = _perlin_noise_3d(positions, scale=40.0, seed=33)
    dark_spots = (speckle > 0.95).astype(np.float32) * 0.15
    r -= dark_spots
    g -= dark_spots
    b -= dark_spots

    return np.stack([np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)], axis=1)


@register_texture("concrete")
def texture_concrete(positions: np.ndarray) -> np.ndarray:
    """Concrete: rough gray with aggregate speckles.

    Returns: (N, 3) RGB colors
    """
    N = len(positions)

    # Base: medium gray
    base = np.full(N, 0.62)

    # Coarse aggregate (medium noise)
    aggregate = _fbm_noise(positions, octaves=3, scale=12.0, seed=11)

    # Fine sand texture (high-frequency)
    sand = _perlin_noise_3d(positions, scale=30.0, seed=55)

    # Dark aggregate spots
    dark_agg = (aggregate > 0.65).astype(np.float32) * 0.12
    # Light sand spots
    light_sand = (sand > 0.7).astype(np.float32) * 0.06

    gray = base + (aggregate - 0.5) * 0.1 - dark_agg + light_sand

    # Slight warm tint variation
    r = gray + 0.02
    g = gray
    b = gray - 0.02

    return np.stack([np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)], axis=1)


@register_texture("glass")
def texture_glass(positions: np.ndarray) -> np.ndarray:
    """Glass: very light blue-green, smooth, slight internal caustics."""
    N = len(positions)

    base_r = np.full(N, 0.82)
    base_g = np.full(N, 0.90)
    base_b = np.full(N, 0.88)

    # Internal caustic-like pattern
    caustic = _fbm_noise(positions, octaves=4, scale=10.0, seed=21)

    r = base_r + (caustic - 0.5) * 0.05
    g = base_g + (caustic - 0.5) * 0.08
    b = base_b + (caustic - 0.5) * 0.06

    return np.stack([np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)], axis=1)


@register_texture("wood")
def texture_wood(positions: np.ndarray) -> np.ndarray:
    """Wood: brown with grain rings."""
    N = len(positions)

    # Distance from center axis → ring pattern
    center = positions.mean(axis=0)
    dx = positions[:, 0] - center[0]
    dz = positions[:, 2] - center[2]
    ring_dist = np.sqrt(dx**2 + dz**2)
    rings = np.sin(ring_dist * 40.0) * 0.5 + 0.5

    grain = _fbm_noise(positions, octaves=3, scale=5.0, seed=77)

    base_r = 0.55 + rings * 0.15 + (grain - 0.5) * 0.1
    base_g = 0.38 + rings * 0.10 + (grain - 0.5) * 0.08
    base_b = 0.22 + rings * 0.05 + (grain - 0.5) * 0.05

    return np.stack([np.clip(base_r, 0, 1), np.clip(base_g, 0, 1), np.clip(base_b, 0, 1)], axis=1)


@register_texture("stone")
def texture_stone(positions: np.ndarray) -> np.ndarray:
    """Stone: gray-brown with veining."""
    N = len(positions)

    vein = _fbm_noise(positions, octaves=5, scale=8.0, seed=44)
    base = 0.55 + (vein - 0.5) * 0.2

    r = base + 0.03
    g = base
    b = base - 0.03

    return np.stack([np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)], axis=1)


# Per-material rendering properties
MATERIAL_PROPERTIES = {
    "ice":        {"opacity_scale": 0.08, "specular": 0.9, "shininess": 80},
    "glass":      {"opacity_scale": 0.6, "specular": 0.8, "shininess": 64},
    "ceramic":    {"opacity_scale": 1.0, "specular": 0.3, "shininess": 32},
    "concrete":   {"opacity_scale": 1.0, "specular": 0.05, "shininess": 8},
    "stone":      {"opacity_scale": 1.0, "specular": 0.1, "shininess": 16},
    "metal":      {"opacity_scale": 1.0, "specular": 0.7, "shininess": 64},
    "wood":       {"opacity_scale": 1.0, "specular": 0.1, "shininess": 8},
    "polymer":    {"opacity_scale": 0.9, "specular": 0.2, "shininess": 16},
    "biological": {"opacity_scale": 1.0, "specular": 0.15, "shininess": 16},
    "food":       {"opacity_scale": 1.0, "specular": 0.1, "shininess": 8},
    "composite":  {"opacity_scale": 1.0, "specular": 0.15, "shininess": 16},
    "other":      {"opacity_scale": 0.7, "specular": 0.5, "shininess": 32},
}


def get_material_properties(material_category: str) -> dict:
    """Get rendering properties for a material category."""
    return MATERIAL_PROPERTIES.get(material_category, {"opacity_scale": 1.0, "specular": 0.1, "shininess": 16})


def apply_material_texture(gaussians, positions: np.ndarray, material_category: str):
    """
    Apply procedural texture to Gaussians based on material type.

    Modifies gaussians._features_dc and _opacity in place.

    Args:
        gaussians: GaussianModel instance
        positions: (N, 3) Gaussian positions (numpy)
        material_category: material category string (ice, ceramic, concrete, etc.)
    """
    texture_fn = TEXTURE_FUNCTIONS.get(material_category)
    if texture_fn is None:
        print(f"[Texture] No texture for '{material_category}', using default gray")
        return

    colors = texture_fn(positions)  # (N, 3) RGB [0, 1]

    # Convert RGB to SH DC
    SH_C0 = 0.28209479177387814
    dc = (colors - 0.5) / SH_C0
    dc_tensor = torch.tensor(dc, dtype=torch.float32, device=gaussians._features_dc.device)
    gaussians._features_dc.data[:, 0, :] = dc_tensor

    # Apply opacity scaling (e.g. ice = semi-transparent)
    props = get_material_properties(material_category)
    opacity_scale = props["opacity_scale"]
    if opacity_scale < 1.0:
        cur_opacity = torch.sigmoid(gaussians._opacity.data)
        new_opacity = cur_opacity * opacity_scale
        new_opacity = new_opacity.clamp(1e-6, 1 - 1e-6)
        gaussians._opacity.data = torch.log(new_opacity / (1.0 - new_opacity))
        print(f"[Texture] Opacity scaled by {opacity_scale:.2f}")

    print(f"[Texture] Applied '{material_category}' texture to {len(positions)} Gaussians")
