"""
Gaussian-Manifold Fracture Field

Physics-informed fracture propagation on a render-native Gaussian manifold.
Fracture state lives directly on the Gaussian surface graph.
"""

from .graph_builder import GaussianGraph
from .physics_projector import PhysicsProjector
from .tip_based_fracture_field import GaussianFractureField
from .gaussian_splitter import GaussianSplitter
from .graph_fragment_manager import GraphFragmentManager
from .crack_front import CrackFront

__all__ = [
    "GaussianGraph",
    "PhysicsProjector",
    "GaussianFractureField",
    "GaussianSplitter",
    "GraphFragmentManager",
    "CrackFront",
]
