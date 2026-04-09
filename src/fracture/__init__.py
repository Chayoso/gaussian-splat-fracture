"""
Gaussian-Manifold Fracture Field

Physics-informed fracture propagation on a render-native Gaussian manifold.
Replaces the volumetric phase-field → surface projection pipeline with
graph-based damage evolution directly on Gaussians.

Modules:
    graph_builder          - kNN graph construction on Gaussian positions
    physics_projector      - MPM tensile energy → Gaussian driving force
    gaussian_fracture_field - Graph-based damage evolution (Phase 1+2)
    gaussian_splitter      - Gaussian split/opening operations (Phase 3)
    graph_fragment_manager - Graph connectivity-based fragment detection (Phase 3)
"""

from .graph_builder import GaussianGraph
from .physics_projector import PhysicsProjector
from .tip_based_fracture_field import GaussianFractureField
from .gaussian_splitter import GaussianSplitter
from .graph_fragment_manager import GraphFragmentManager
from .crack_front import CrackFront
