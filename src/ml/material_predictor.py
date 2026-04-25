"""
Material Parameter Predictor — CLIP-based KNN + optional Transformer.

Given a text description (or image), predicts material parameters (E, Gc, nu)
by retrieving similar materials from the database and combining their properties.

Two modes:
  - "clip_knn": Weighted average of top-K materials in CLIP embedding space
  - "clip_transformer": KNN retrieval + cross-attention transformer refinement
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from src.ml.material_db import MaterialDB
from src.ml.clip_encoder import CLIPTextEncoder


class MaterialTransformerHead(nn.Module):
    """Small transformer that refines KNN-retrieved parameters.

    Input: query embedding (1, 512) + top-K embeddings (K, 512) + top-K params (K, 3)
    Output: refined parameters (3,) = [E, Gc, nu]
    """

    def __init__(self, embed_dim: int = 512, n_heads: int = 4,
                 n_layers: int = 2, k: int = 5):
        super().__init__()
        self.k = k

        # Project params into embedding space
        self.param_proj = nn.Linear(3, embed_dim)

        # Cross-attention: query attends to retrieved materials
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 2, dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head: predict log(E), log(Gc), nu
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, query_emb: torch.Tensor, knn_embs: torch.Tensor,
                knn_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_emb: (1, 512) query CLIP embedding
            knn_embs: (K, 512) top-K retrieved CLIP embeddings
            knn_params: (K, 3) top-K material params [log(E), log(Gc), nu]

        Returns:
            (3,) predicted [log(E), log(Gc), nu]
        """
        # Combine embedding + param info for retrieved materials
        param_features = self.param_proj(knn_params)  # (K, 512)
        knn_combined = knn_embs + param_features       # (K, 512)

        # Stack query + retrieved as sequence: (1, K+1, 512)
        seq = torch.cat([query_emb, knn_combined], dim=0).unsqueeze(0)  # (1, K+1, 512)

        # Transformer processes the sequence
        out = self.transformer(seq)  # (1, K+1, 512)

        # Take query position output → predict params
        query_out = out[0, 0, :]  # (512,)
        return self.output_head(query_out)  # (3,)


class MaterialPredictor:
    """
    Predict material parameters from text or image description.

    Usage:
        predictor = MaterialPredictor(mode="clip_knn")
        params = predictor.predict("ceramic mug")
        # → {"E": 7e10, "Gc": 50.0, "nu": 0.22, "density": 2400.0}
    """

    def __init__(self, mode: str = "clip_knn",
                 db_path: str = None,
                 clip_model: str = "ViT-B/32",
                 transformer_checkpoint: str = None,
                 k: int = 5,
                 device: str = None,
                 keyword_rerank: bool = True):
        """
        Args:
            mode: "clip_knn" or "clip_transformer"
            db_path: Path to material_db.json
            clip_model: CLIP variant
            transformer_checkpoint: Path to trained transformer weights
            k: Number of neighbors for KNN
            device: torch device
            keyword_rerank: Small lexical rerank on top of CLIP similarity.
                This keeps CLIP as the semantic retriever while preventing
                obvious material words such as "glass" from being displaced by
                visually similar but physically different materials.
        """
        self.mode = mode
        self.k = k
        self.keyword_rerank = bool(keyword_rerank)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load material database
        self.db = MaterialDB(db_path)

        # Load CLIP encoder
        self.encoder = CLIPTextEncoder(clip_model, device=device)

        # Pre-compute CLIP embeddings for all DB entries
        descriptions = self.db.get_descriptions()
        self.db_embeddings = self.encoder.encode_text(descriptions)  # (N, 512)
        print(f"[MaterialPredictor] Pre-encoded {len(descriptions)} material descriptions")

        # Optional transformer head
        self.transformer = None
        if mode == "clip_transformer":
            self.transformer = MaterialTransformerHead(k=k).to(device)
            if transformer_checkpoint:
                state = torch.load(transformer_checkpoint, map_location=device)
                self.transformer.load_state_dict(state)
                print(f"[MaterialPredictor] Loaded transformer from {transformer_checkpoint}")
            self.transformer.eval()

    @torch.no_grad()
    def predict(self, text: str, k: int = None) -> dict:
        """
        Predict material parameters from text description.

        Args:
            text: Material description (e.g. "ceramic mug", "glass bottle")
            k: Override number of neighbors

        Returns:
            dict with "E", "Gc", "nu", "density", "top_k_names", "top_k_scores"
        """
        if k is None:
            k = self.k
        k = min(k, len(self.db))

        # Encode query
        query_emb = self.encoder.encode_text(text)  # (1, 512)

        # Cosine similarity with all DB entries
        similarities = (query_emb @ self.db_embeddings.T).squeeze(0)  # (N,)
        rerank_bonus = self._keyword_rerank_bonus(text) if self.keyword_rerank else torch.zeros_like(similarities)
        retrieval_scores = similarities + rerank_bonus
        topk = retrieval_scores.topk(k)
        topk_idx = topk.indices
        topk_scores = topk.values
        topk_raw_scores = similarities[topk_idx]

        # Get parameter arrays
        params_all = self.db.get_full_params_array()  # (N, 4): E, Gc, nu, density
        topk_params = params_all[topk_idx.cpu().numpy()]  # (K, 4)
        topk_names = [self.db[i].name for i in topk_idx.cpu().numpy()]

        if self.mode == "clip_knn":
            result = self._predict_knn(topk_scores, topk_params)
        elif self.mode == "clip_transformer":
            result = self._predict_transformer(query_emb, topk_idx, topk_params)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        result["top_k_names"] = topk_names
        result["top_k_scores"] = topk_scores.cpu().numpy().tolist()
        result["top_k_raw_scores"] = topk_raw_scores.cpu().numpy().tolist()
        result["top_k_rerank_bonus"] = rerank_bonus[topk_idx].cpu().numpy().tolist()
        return result

    def _keyword_rerank_bonus(self, text: str) -> torch.Tensor:
        """Small query-aware prior for explicit material words.

        CLIP is good at broad visual semantics, but transparent or brittle
        objects often sit close together. A bounded bonus lets explicit words
        such as "glass", "rubber", or "sandstone" win retrieval without
        replacing CLIP with a brittle string matcher.
        """
        q = str(text).lower()
        bonuses = np.zeros(len(self.db), dtype=np.float32)

        def has_any(words):
            return any(word in q for word in words)

        for i, entry in enumerate(self.db.entries):
            name = entry.name.lower()
            category = entry.category.lower()
            bonus = 0.0

            if has_any(("glass", "bottle", "pane", "window", "shatter")):
                if category == "glass":
                    bonus += 0.085
                if "glass" in name:
                    bonus += 0.030
                if ("acrylic" in name or "pmma" in name) and not has_any(("acrylic", "pmma", "plexiglass")):
                    bonus -= 0.025

            if has_any(("porcelain", "ceramic", "mug", "china")):
                if category == "ceramic":
                    bonus += 0.070
                if any(token in name for token in ("porcelain", "stoneware", "china", "ceramic")):
                    bonus += 0.040

            if has_any(("concrete", "cement", "mortar")):
                if category == "concrete":
                    bonus += 0.085
                if any(token in name for token in ("concrete", "mortar", "cement")):
                    bonus += 0.035

            if has_any(("sandstone", "limestone", "marble", "stone", "rock")):
                if category == "stone":
                    bonus += 0.060
                if "sandstone" in q and "sandstone" in name:
                    bonus += 0.095

            if has_any(("rubber", "latex", "elastomer", "elastic")):
                if any(token in name for token in ("rubber", "latex", "neoprene", "silicone")):
                    bonus += 0.120
            elif has_any(("soft", "deforming", "deform")) and "rubber" in name:
                bonus += 0.055

            if has_any(("ice", "frozen")) and (category == "ice" or "ice" in name):
                bonus += 0.095

            if has_any(("wood", "timber", "bamboo")):
                if category == "wood":
                    bonus += 0.080
                if any(token in name for token in ("wood", "bamboo", "oak", "pine")):
                    bonus += 0.030

            if has_any(("metal", "steel", "aluminum", "iron")):
                if category == "metal":
                    bonus += 0.080
                if any(token in name for token in ("steel", "aluminum", "iron")):
                    bonus += 0.035

            bonuses[i] = np.float32(np.clip(bonus, -0.04, 0.16))

        return torch.as_tensor(bonuses, dtype=torch.float32, device=self.device)

    def _predict_knn(self, scores: torch.Tensor, topk_params: np.ndarray) -> dict:
        """Weighted average of top-K in log-space for E, Gc."""
        # Temperature-scaled softmax weights
        weights = torch.softmax(scores * 10.0, dim=0).cpu().numpy()  # (K,)

        # Log-space average for E and Gc (span orders of magnitude)
        log_E = np.sum(weights * np.log(topk_params[:, 0]))
        log_Gc = np.sum(weights * np.log(topk_params[:, 1]))
        nu = np.sum(weights * topk_params[:, 2])
        density = np.sum(weights * topk_params[:, 3])

        return {
            "E": float(np.exp(log_E)),
            "Gc": float(np.exp(log_Gc)),
            "nu": float(nu),
            "density": float(density),
        }

    def _predict_transformer(self, query_emb: torch.Tensor,
                              topk_idx: torch.Tensor,
                              topk_params: np.ndarray) -> dict:
        """Transformer-refined prediction from KNN retrieval."""
        if self.transformer is None:
            raise RuntimeError("Transformer not loaded. Use mode='clip_knn' or provide checkpoint.")

        knn_embs = self.db_embeddings[topk_idx]  # (K, 512)

        # Normalize params to log-space for transformer input
        log_params = np.zeros((len(topk_params), 3))
        log_params[:, 0] = np.log(topk_params[:, 0])  # log(E)
        log_params[:, 1] = np.log(topk_params[:, 1])  # log(Gc)
        log_params[:, 2] = topk_params[:, 2]           # nu (already small range)

        knn_params_t = torch.tensor(log_params, dtype=torch.float32, device=self.device)

        # Transformer prediction
        pred = self.transformer(query_emb.squeeze(0).unsqueeze(0),
                                knn_embs, knn_params_t)  # (3,)
        pred = pred.cpu().numpy()

        # Density from KNN (transformer doesn't predict it)
        weights = torch.softmax(
            (query_emb @ self.db_embeddings[topk_idx].T).squeeze(0) * 10.0,
            dim=0).cpu().numpy()
        density = np.sum(weights * topk_params[:, 3])

        return {
            "E": float(np.exp(pred[0])),
            "Gc": float(np.exp(pred[1])),
            "nu": float(np.clip(pred[2], 0.1, 0.45)),
            "density": float(density),
        }
