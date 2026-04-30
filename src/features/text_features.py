"""BERT text feature extraction for multimodal conditions (C1-MM through C4-MM).

Extracts [CLS] token embeddings (768-dim) from BERT-base-uncased for each
fixed sentence.  Embeddings are cached keyed by (language, sentence_id) so
that each unique sentence is processed only once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..utils.logger import get_logger
from ..utils.seed import get_device

logger = get_logger(__name__)

_BERT_CACHE: dict = {}


def _load_bert(model_name: str = "bert-base-uncased") -> Tuple:  # noqa: F821
    """Load BERT model and tokenizer, cached in memory.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.

    Returns
    -------
    model : BertModel
    tokenizer : BertTokenizer
    device : torch.device
    """
    from typing import Tuple

    if model_name in _BERT_CACHE:
        return _BERT_CACHE[model_name]

    from transformers import AutoModel, AutoTokenizer

    logger.info("Loading BERT model: %s", model_name)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    _BERT_CACHE[model_name] = (model, tokenizer, device)
    return model, tokenizer, device


def _cache_key(language: str, sentence_id: str) -> str:
    return f"{language}_{sentence_id}"


def _text_cache_path(cache_dir: Path, language: str, sentence_id: str) -> Path:
    return cache_dir / "bert" / f"{language}_{sentence_id}.npy"


def extract_text_embedding(
    text: str,
    language: str,
    sentence_id: str,
    model_name: str = "bert-base-uncased",
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    """Extract BERT [CLS] token embedding for a sentence.

    Parameters
    ----------
    text : str
        The sentence text (looked up from ``sentences.json``).
    language : str
        ``"EN"`` or ``"YO"`` — used for cache keying.
    sentence_id : str
        Sentence identifier (e.g. ``"S1"``).
    model_name : str
        HuggingFace BERT model identifier.
    cache_dir : Path, optional
        Root directory for caching.  Embeddings are shared across all
        recordings with the same (language, sentence_id) pair.

    Returns
    -------
    np.ndarray, shape (768,)
        [CLS] token embedding from the last hidden layer.

    Notes
    -----
    Text is fixed per sentence, so this is computed once per unique
    (language, sentence_id) pair regardless of how many recordings
    use that sentence.
    """
    if cache_dir is not None:
        cached_path = _text_cache_path(cache_dir, language, sentence_id)
        if cached_path.exists():
            logger.debug("Text embedding cache hit: %s", cached_path.name)
            return np.load(cached_path)

    import torch

    model, tokenizer, device = _load_bert(model_name)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # [CLS] is the first token in last_hidden_state
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

    if cache_dir is not None:
        cached_path = _text_cache_path(cache_dir, language, sentence_id)
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cached_path, cls_embedding)
        logger.debug("Cached text embedding: %s", cached_path.name)

    return cls_embedding.astype(np.float32)


def extract_all_text_embeddings(
    sentences: Dict[str, Dict[str, str]],
    model_name: str = "bert-base-uncased",
    cache_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Extract and cache embeddings for all sentences in ``sentences.json``.

    Parameters
    ----------
    sentences : dict
        Loaded ``sentences.json`` (outer key = language, inner key = S1-S5).
    model_name : str
        BERT model identifier.
    cache_dir : Path, optional
        Cache directory.

    Returns
    -------
    dict
        Keys are ``"EN_S1"``, ``"YO_S3"``, etc.; values are (768,) arrays.
    """
    embeddings: Dict[str, np.ndarray] = {}
    for lang, sent_dict in sentences.items():
        for sid, text in sent_dict.items():
            if "[Yoruba translation" in text:
                logger.warning(
                    "Sentence %s/%s is a placeholder; skipping BERT embedding.", lang, sid
                )
                continue
            key = f"{lang}_{sid}"
            emb = extract_text_embedding(
                text, lang, sid, model_name=model_name, cache_dir=cache_dir
            )
            embeddings[key] = emb
    logger.info("Extracted text embeddings for %d sentences.", len(embeddings))
    return embeddings
