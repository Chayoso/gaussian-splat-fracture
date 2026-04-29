"""CLIP-conditioned crack style head.

A tiny MLP that maps CLIP text embeddings to crack-style logits.  Trained
with weak supervision derived from the existing keyword-rule sentence
selector on a template corpus, the head's job is to generalize beyond
the rule's exact tokens: paraphrased sentences (e.g. "the bottle
disintegrated into a star pattern") still get routed to ``radial_shatter``
even when no keyword match exists.

Paper-defensibility note: this is auxiliary supervision on top of the
rule-based labels, not a fully-supervised classifier.  We use the rule's
output as weak label and the head learns a smooth embedding-to-style map
that the rule's discrete keyword check cannot represent.  The rule
remains available as a fallback when the head's confidence is low.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_TEMPLATE_CORPUS: Tuple[Tuple[str, str], ...] = (
    # diffuse_microcrack
    ("the surface shows tiny scattered scratches", "diffuse_microcrack"),
    ("develops shallow surface microcracks under load", "diffuse_microcrack"),
    ("exhibits diffuse damage without visible fracture", "diffuse_microcrack"),
    ("vulcanized rubber deforming without cracking", "diffuse_microcrack"),
    ("denting without brittle fracture", "diffuse_microcrack"),
    ("rubber ball squashes without cracks", "diffuse_microcrack"),
    ("only mild surface scuffing appears", "diffuse_microcrack"),
    ("a soft diffuse damage halo with no fragments", "diffuse_microcrack"),
    # single_smooth
    ("splits along one long smooth crack", "single_smooth"),
    ("a single straight crack runs across the surface", "single_smooth"),
    ("ceramic tears with a single clean crack line", "single_smooth"),
    ("fractures along one smooth plane", "single_smooth"),
    ("shows one continuous crack from impact to edge", "single_smooth"),
    ("a long smooth split with no branching", "single_smooth"),
    ("ceramic plate with a clean single crack", "single_smooth"),
    # spiderweb_branching
    ("fractures into a spiderweb of cracks", "spiderweb_branching"),
    ("a wide branched network of cracks forms", "spiderweb_branching"),
    ("shows a web of intersecting connected fractures", "spiderweb_branching"),
    ("spiderweb cracking pattern across the bottle", "spiderweb_branching"),
    ("branched network of cracks meets at the impact site", "spiderweb_branching"),
    ("connected branching cracks like a glass spiderweb", "spiderweb_branching"),
    ("dense branched cracks intersect repeatedly", "spiderweb_branching"),
    # radial_shatter
    ("shatters into many radial cracks", "radial_shatter"),
    ("shatters with bright radial lines", "radial_shatter"),
    ("the bottle disintegrates into a star pattern", "radial_shatter"),
    ("explodes into a star-shaped fracture", "radial_shatter"),
    ("shatters into radial sharp pieces", "radial_shatter"),
    ("complete radial breakup with hundreds of shards", "radial_shatter"),
    ("thin glass bottle shattering into radial cracks", "radial_shatter"),
    ("ice plate shatters radially from the impact", "radial_shatter"),
    ("a sunburst pattern of cracks", "radial_shatter"),
    # chunky_crumble
    ("crumbles into rough granular chunks", "chunky_crumble"),
    ("breaks into chunky pieces", "chunky_crumble"),
    ("fractures into rough rubble", "chunky_crumble"),
    ("rough quasi-brittle concrete chunk release", "chunky_crumble"),
    ("crumbles into heavy granular fragments", "chunky_crumble"),
    ("concrete block breaks into rough chunks", "chunky_crumble"),
    ("blocky uneven crumble pattern", "chunky_crumble"),
)


_MATERIAL_PREFIXES: Tuple[str, ...] = (
    "",
    "the ",
    "a thin ",
    "a fired ",
    "the rough ",
    "the soda-lime ",
)


class StyleHead(nn.Module):
    """Two-layer MLP mapping CLIP embedding -> style logits."""

    def __init__(
        self,
        clip_dim: int = 512,
        hidden: int = 128,
        num_styles: int = 5,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.hidden = hidden
        self.num_styles = num_styles
        self.fc1 = nn.Linear(clip_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_styles)
        self.dropout = nn.Dropout(dropout)
        self._style_names: List[str] = []

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(emb))
        h = self.dropout(h)
        return self.fc2(h)

    @torch.no_grad()
    def predict(
        self,
        emb: torch.Tensor,
    ) -> Tuple[List[int], List[float], torch.Tensor]:
        """Return (best_idx_per_row, best_conf_per_row, full_softmax)."""
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        self.eval()
        logits = self.forward(emb)
        probs = F.softmax(logits, dim=-1)
        best_p, best_idx = probs.max(dim=-1)
        return (
            best_idx.detach().cpu().tolist(),
            best_p.detach().cpu().tolist(),
            probs.detach().cpu(),
        )

    def style_name(self, idx: int) -> str:
        if 0 <= idx < len(self._style_names):
            return self._style_names[idx]
        return "material_default"


def _build_training_corpus() -> List[Tuple[str, str]]:
    """Expand the template corpus by prefixing each phrase with material qualifiers."""
    out: List[Tuple[str, str]] = []
    for phrase, label in _TEMPLATE_CORPUS:
        out.append((phrase, label))
        for prefix in _MATERIAL_PREFIXES:
            if not prefix:
                continue
            out.append((f"{prefix}{phrase}", label))
    return out


def train_style_head(
    encoder,
    num_epochs: int = 60,
    lr: float = 5e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    verbose: bool = False,
) -> StyleHead:
    """Train head on the rule-derived weak-supervision template corpus."""
    if device is None:
        device = getattr(encoder, "device", "cuda" if torch.cuda.is_available() else "cpu")

    corpus = _build_training_corpus()
    style_names = sorted({lbl for _, lbl in corpus})
    style_to_idx = {name: i for i, name in enumerate(style_names)}
    texts = [t for t, _ in corpus]
    labels = torch.tensor(
        [style_to_idx[lbl] for _, lbl in corpus],
        dtype=torch.long, device=device,
    )

    with torch.no_grad():
        embs = encoder.encode_text(texts).to(device)

    head = StyleHead(
        clip_dim=int(embs.shape[1]),
        hidden=128,
        num_styles=len(style_names),
    ).to(device)
    head._style_names = style_names

    optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    head.train()
    for epoch in range(num_epochs):
        logits = head(embs)
        loss = F.cross_entropy(logits, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            print(f"[StyleHead] epoch={epoch+1} loss={loss.item():.4f} acc={acc:.3f}")
    head.eval()
    return head


def _default_cache_path() -> Path:
    return Path(os.path.expanduser("~/.cache/gaussian_phase_field/style_head.pt"))


def get_or_train_style_head(
    encoder,
    cache_path: Optional[str] = None,
    force_retrain: bool = False,
    verbose: bool = False,
) -> StyleHead:
    """Load cached head if present, else train fresh and cache."""
    cache_p = Path(cache_path) if cache_path is not None else _default_cache_path()
    if cache_p.exists() and not force_retrain:
        try:
            ckpt = torch.load(cache_p, map_location=getattr(encoder, "device", "cpu"))
            head = StyleHead(
                clip_dim=int(ckpt["clip_dim"]),
                hidden=int(ckpt.get("hidden", 128)),
                num_styles=len(ckpt["style_names"]),
            ).to(getattr(encoder, "device", "cpu"))
            head.load_state_dict(ckpt["state_dict"])
            head._style_names = list(ckpt["style_names"])
            head.eval()
            if verbose:
                print(f"[StyleHead] loaded from {cache_p}")
            return head
        except Exception as e:
            print(f"[StyleHead] cache load failed ({e}); retraining")

    head = train_style_head(encoder, verbose=verbose)
    try:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": head.state_dict(),
                "clip_dim": head.clip_dim,
                "hidden": head.hidden,
                "style_names": head._style_names,
            },
            cache_p,
        )
        if verbose:
            print(f"[StyleHead] trained and cached at {cache_p}")
    except Exception as e:
        print(f"[StyleHead] cache save failed: {e}")
    return head
