"""PyTorch MLP for C4 SSL embeddings and multimodal fusion conditions.

Architecture: Linear(input_dim → 256) → ReLU → Dropout(0.3) →
              Linear(256 → 64) → ReLU → Dropout(0.3) →
              Linear(64 → n_classes)

Supports full checkpoint/resume: training can be interrupted and resumed
from the last saved epoch without re-extracting features.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger
from ..utils.seed import get_device

logger = get_logger(__name__)


class MLPClassifier:
    """Wrapper around a PyTorch MLP with early stopping and checkpoint/resume.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input feature vectors.
    hidden_dims : list of int
        Sizes of hidden layers.  Defaults to ``[256, 64]``.
    dropout : float
        Dropout probability.  Defaults to 0.3.
    n_classes : int
        Number of output classes.
    learning_rate : float
        Adam learning rate.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    early_stopping_patience : int
        Stop if validation loss does not improve for this many epochs.
    random_seed : int
        Random seed.
    checkpoint_dir : Path, optional
        If provided, saves a checkpoint after each epoch and resumes
        automatically if a checkpoint exists.  Checkpoints are keyed by
        *checkpoint_name*.
    checkpoint_name : str
        Stem for checkpoint files (e.g. ``"C4_EN_mlp"``).

    Examples
    --------
    >>> mlp = MLPClassifier(input_dim=768, checkpoint_dir=Path(".cache/ckpt"),
    ...                     checkpoint_name="C4_EN_fold_P01")
    >>> mlp.fit(X_train, y_train, X_val, y_val)   # resumes if interrupted
    >>> y_pred, y_prob = mlp.predict(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        n_classes: int = 2,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 16,
        early_stopping_patience: int = 15,
        random_seed: int = 42,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_name: str = "mlp",
    ) -> None:
        if hidden_dims is None:
            hidden_dims = [256, 64]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.n_classes = n_classes
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = early_stopping_patience
        self.seed = random_seed
        self.device = get_device()
        self._model = None
        self._scaler = None

        self._ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._ckpt_name = checkpoint_name

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _ckpt_path(self) -> Optional[Path]:
        if self._ckpt_dir is None:
            return None
        return self._ckpt_dir / f"{self._ckpt_name}.pt"

    def _scaler_path(self) -> Optional[Path]:
        if self._ckpt_dir is None:
            return None
        return self._ckpt_dir / f"{self._ckpt_name}_scaler.pkl"

    def _save_checkpoint(
        self,
        epoch: int,
        optimizer,
        best_val_loss: float,
        patience_counter: int,
        best_state: Optional[Dict],
    ) -> None:
        """Persist training state to disk.

        Parameters
        ----------
        epoch : int
            Last completed epoch index (0-based).
        optimizer : torch.optim.Optimizer
            Optimizer whose state is saved.
        best_val_loss : float
            Best validation loss seen so far.
        patience_counter : int
            Current early stopping patience counter.
        best_state : dict or None
            State dict of the best model weights.
        """
        import torch

        if self._ckpt_dir is None:
            return
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": epoch,
            "model_state": self._model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "best_state": best_state,
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "n_classes": self.n_classes,
        }
        torch.save(payload, self._ckpt_path())

        # Save scaler separately (not a torch object)
        with open(self._scaler_path(), "wb") as f:
            pickle.dump(self._scaler, f)

        logger.debug("Checkpoint saved at epoch %d → %s", epoch + 1, self._ckpt_path())

    def _load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint from disk if it exists.

        Returns
        -------
        dict or None
            Checkpoint payload, or None if no checkpoint exists.
        """
        import torch

        path = self._ckpt_path()
        if path is None or not path.exists():
            return None

        payload = torch.load(path, map_location=self.device)
        logger.info(
            "Resuming MLP from checkpoint (epoch %d): %s",
            payload["epoch"] + 1,
            path,
        )

        scaler_path = self._scaler_path()
        if scaler_path and scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)

        return payload

    def clear_checkpoint(self) -> None:
        """Delete the checkpoint files for this run (call after successful completion)."""
        for p in [self._ckpt_path(), self._scaler_path()]:
            if p and p.exists():
                p.unlink()
                logger.debug("Deleted checkpoint: %s", p)

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------

    def _build_model(self):
        import torch.nn as nn

        layers = []
        in_dim = self.input_dim
        for h_dim in self.hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.n_classes))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "MLPClassifier":
        """Train the MLP, resuming from checkpoint if one exists.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_train, input_dim)
            Training features (unscaled).
        y_train : np.ndarray, shape (n_train,)
            Training labels.
        X_val : np.ndarray, optional
            Validation features for early stopping.
        y_val : np.ndarray, optional
            Validation labels.

        Returns
        -------
        self

        Notes
        -----
        If a checkpoint exists at ``checkpoint_dir / checkpoint_name.pt``,
        training resumes from the saved epoch automatically.  The scaler
        fitted on the original training data is also restored, so no data
        is required to be present in the same form as the original run.
        """
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)

        # ---- Attempt checkpoint restore ----
        ckpt = self._load_checkpoint()
        start_epoch = 0
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        if ckpt is not None:
            # Restore scaler (already loaded in _load_checkpoint)
            if self._scaler is None:
                logger.warning("Checkpoint found but scaler missing; refitting scaler.")
                self._scaler = StandardScaler()
                self._scaler.fit(X_train)
            # Build and restore model
            self._model = self._build_model().to(self.device)
            self._model.load_state_dict(ckpt["model_state"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            patience_counter = ckpt["patience_counter"]
            best_state = ckpt["best_state"]
        else:
            # Fresh start
            self._scaler = StandardScaler()
            self._scaler.fit(X_train)
            self._model = self._build_model().to(self.device)

        X_train_scaled = self._scaler.transform(X_train)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        if ckpt is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])

        criterion = nn.CrossEntropyLoss()

        X_t = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed))

        # Validation tensors
        if X_val is not None and y_val is not None:
            X_v = torch.tensor(
                self._scaler.transform(X_val), dtype=torch.float32
            ).to(self.device)
            y_v = torch.tensor(y_val, dtype=torch.long).to(self.device)
        else:
            X_v = y_v = None

        if start_epoch >= self.epochs:
            logger.info("Checkpoint at final epoch — skipping training.")
            if best_state is not None:
                self._model.load_state_dict(best_state)
            return self

        logger.info(
            "MLP training: epochs %d→%d, device=%s, checkpoint=%s",
            start_epoch, self.epochs, self.device,
            str(self._ckpt_path()) if self._ckpt_dir else "disabled",
        )

        epoch = start_epoch
        for epoch in range(start_epoch, self.epochs):
            self._model.train()
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self._model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            # Early stopping on validation
            if X_v is not None:
                self._model.eval()
                with torch.no_grad():
                    val_logits = self._model(X_v)
                    val_loss = criterion(val_logits, y_v).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self._model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(
                            "Early stopping at epoch %d (val_loss=%.4f).",
                            epoch + 1, val_loss,
                        )
                        break

            # Save checkpoint every epoch
            self._save_checkpoint(
                epoch, optimizer, best_val_loss, patience_counter, best_state
            )

        if best_state is not None:
            self._model.load_state_dict(best_state)

        logger.info("MLP training completed (%d epochs).", epoch + 1)
        self.clear_checkpoint()  # clean up on successful completion
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, input_dim)
            Feature matrix (unscaled).

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
        y_prob : np.ndarray, shape (n_samples, n_classes)
        """
        import torch
        import torch.nn.functional as F

        X_scaled = self._scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        return preds, probs

    # ------------------------------------------------------------------
    # Persistence (final model, not training checkpoint)
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the final trained model and scaler to disk.

        Parameters
        ----------
        path : Path
            Directory to save into.  Creates ``mlp_model.pt`` and
            ``mlp_scaler.pkl`` files.
        """
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self._model.state_dict(),
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "n_classes": self.n_classes,
            },
            path / "mlp_model.pt",
        )
        with open(path / "mlp_scaler.pkl", "wb") as f:
            pickle.dump(self._scaler, f)
        logger.info("MLP model saved to %s", path)

    @classmethod
    def load(cls, path: Path, device: Optional[str] = None) -> "MLPClassifier":
        """Load a previously saved MLP model.

        Parameters
        ----------
        path : Path
            Directory containing ``mlp_model.pt`` and ``mlp_scaler.pkl``.
        device : str, optional
            Override device (e.g. ``"cpu"``).

        Returns
        -------
        MLPClassifier
            Loaded and ready for inference.
        """
        import torch

        path = Path(path)
        payload = torch.load(path / "mlp_model.pt", map_location=device or "cpu")
        instance = cls(
            input_dim=payload["input_dim"],
            hidden_dims=payload["hidden_dims"],
            dropout=payload["dropout"],
            n_classes=payload["n_classes"],
        )
        instance._model = instance._build_model()
        instance._model.load_state_dict(payload["model_state"])
        instance._model.eval()

        with open(path / "mlp_scaler.pkl", "rb") as f:
            instance._scaler = pickle.load(f)

        logger.info("MLP model loaded from %s", path)
        return instance
