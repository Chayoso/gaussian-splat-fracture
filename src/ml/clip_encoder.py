"""
CLIP Text Encoder wrapper for material description embedding.

Uses OpenAI CLIP (ViT-B/32) to encode text descriptions into
512-dimensional vectors for KNN retrieval in material space.
"""

import torch
from typing import Union, List


class CLIPTextEncoder:
    """Encode text descriptions using pre-trained CLIP model."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-L/14, etc.)
            device: torch device (defaults to cuda if available)
        """
        try:
            import clip
        except ImportError:
            raise ImportError(
                "CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        self._clip = clip
        print(f"[CLIPEncoder] Loaded {model_name} on {device}")

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text(s) to normalized CLIP embedding vectors.

        Args:
            texts: Single string or list of strings

        Returns:
            (N, 512) normalized embedding tensor
        """
        if isinstance(texts, str):
            texts = [texts]
        tokens = self._clip.tokenize(texts, truncate=True).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float()

    @torch.no_grad()
    def encode_image(self, images) -> torch.Tensor:
        """
        Encode image(s) to normalized CLIP embedding vectors.

        Args:
            images: PIL Image or list of PIL Images

        Returns:
            (N, 512) normalized embedding tensor
        """
        from PIL import Image
        if isinstance(images, Image.Image):
            images = [images]
        image_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        features = self.model.encode_image(image_tensors)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float()
