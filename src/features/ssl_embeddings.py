"""SSL speech embedding extraction: HuBERT and WavLM.

Extracts mean-pooled last hidden state embeddings (768-dim) from
HuBERT-base and WavLM-base models.  All embeddings are cached to disk
after first extraction and never re-extracted if the cache exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..utils.logger import get_logger
from ..utils.seed import get_device

logger = get_logger(__name__)

MODEL_IDS = {
    "hubert": "facebook/hubert-base-ls960",
    "wavlm": "microsoft/wavlm-base",
}

_MODEL_CACHE: dict = {}


def _load_model(model_key: str) -> Tuple:
    """Load and cache a HuggingFace model and processor in memory.

    Parameters
    ----------
    model_key : str
        ``"hubert"`` or ``"wavlm"``.

    Returns
    -------
    model : transformers AutoModel
        The loaded model in eval mode on the detected device.
    feature_extractor : transformers AutoFeatureExtractor
        The corresponding feature extractor.
    device : torch.device
        The compute device used.
    """
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]

    from transformers import AutoFeatureExtractor, AutoModel
    import torch

    model_id = MODEL_IDS[model_key]
    logger.info("Loading SSL model: %s", model_id)

    device = get_device()
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model = model.to(device)
    model.eval()

    _MODEL_CACHE[model_key] = (model, feature_extractor, device)
    logger.info("SSL model loaded: %s — %.1f M params.", model_id, _count_params(model))
    return model, feature_extractor, device


def _count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def _embedding_cache_path(audio_path: Path, model_key: str, cache_dir: Path) -> Path:
    """Return path for the cached ``.npy`` embedding file."""
    return cache_dir / model_key / f"{audio_path.stem}.npy"


def extract_ssl_embedding(
    audio: np.ndarray,
    sr: int,
    model_key: str = "hubert",
    cache_dir: Optional[Path] = None,
    audio_path: Optional[Path] = None,
) -> np.ndarray:
    """Extract mean-pooled SSL embedding for one audio file.

    Parameters
    ----------
    audio : np.ndarray
        Mono float32 audio at 16 kHz (will be resampled if needed).
    sr : int
        Sample rate of *audio*.
    model_key : str
        ``"hubert"`` or ``"wavlm"``.
    cache_dir : Path, optional
        Root directory for embedding cache.  If provided and *audio_path*
        is given, the embedding is loaded from cache if available and saved
        after first extraction.
    audio_path : Path, optional
        Original file path used as the cache key.

    Returns
    -------
    np.ndarray, shape (768,)
        Mean-pooled embedding vector.

    Notes
    -----
    Audio is resampled to 16 kHz before passing to the model.
    HuBERT/WavLM process raw waveforms; no spectrogram is needed.

    Reference: Hsu et al. (2021) HuBERT; Chen et al. (2022) WavLM.
    """
    # Check cache first
    if cache_dir is not None and audio_path is not None:
        cached_path = _embedding_cache_path(audio_path, model_key, cache_dir)
        if cached_path.exists():
            logger.debug("SSL cache hit: %s", cached_path.name)
            return np.load(cached_path)

    import torch

    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    model, feature_extractor, device = _load_model(model_key)

    inputs = feature_extractor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # last_hidden_state: (batch, time, 768)
        hidden = outputs.last_hidden_state
        # Attention mask may not be returned by all models; fall back to mean
        if hasattr(inputs, "attention_mask") and "attention_mask" in inputs:
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            embedding = hidden.mean(dim=1)
        embedding = embedding.squeeze(0).cpu().numpy()

    # Cache
    if cache_dir is not None and audio_path is not None:
        cached_path = _embedding_cache_path(audio_path, model_key, cache_dir)
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cached_path, embedding)
        logger.debug("Cached SSL embedding: %s", cached_path.name)

    return embedding.astype(np.float32)
