"""
Material Database — collection of (description, E, Gc, nu) pairs.

Used by MaterialPredictor for KNN retrieval in CLIP embedding space.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MaterialEntry:
    name: str
    description: str
    category: str
    E: float           # Young's modulus (Pa)
    Gc: float          # Fracture toughness (J/m^2)
    nu: float          # Poisson's ratio
    density: float     # kg/m^3


class MaterialDB:
    """Material property database for CLIP-based retrieval."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "data" / "material_db.json")
        self.entries = self._load(db_path)
        print(f"[MaterialDB] Loaded {len(self.entries)} materials from {db_path}")

    def _load(self, path: str) -> List[MaterialEntry]:
        with open(path, 'r') as f:
            data = json.load(f)
        return [MaterialEntry(**entry) for entry in data]

    def get_descriptions(self) -> List[str]:
        """Get all material descriptions for CLIP encoding."""
        return [f"{e.name}: {e.description}" for e in self.entries]

    def get_names(self) -> List[str]:
        return [e.name for e in self.entries]

    def get_params_array(self) -> np.ndarray:
        """(N, 3) array of [E, Gc, nu]."""
        return np.array([[e.E, e.Gc, e.nu] for e in self.entries])

    def get_full_params_array(self) -> np.ndarray:
        """(N, 4) array of [E, Gc, nu, density]."""
        return np.array([[e.E, e.Gc, e.nu, e.density] for e in self.entries])

    def search(self, query: str) -> List[MaterialEntry]:
        """Simple keyword search."""
        query_lower = query.lower()
        return [e for e in self.entries
                if query_lower in e.name.lower() or query_lower in e.description.lower()]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]
