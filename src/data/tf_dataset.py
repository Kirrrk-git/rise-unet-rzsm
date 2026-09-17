"""
src/data/tf_dataset.py
----------------------
TensorFlow Data Pipeline and Ensemble Grouping Generator for RISE-UNet Model A0.

Enforces:
1. Ensemble Grouping Semantics: Batch size B must be an exact multiple of 11 (B in {11, 22, 33, ...}).
2. Case-Level Shuffling: Forecast cycles are shuffled at the case level to maintain internal
   11-member covariance (Member 0 is Control Forecast, Members 1-10 are Perturbed Forecasts).
3. Ground-Truth Target Alignment & Broadcasting: Case-level rolling verification targets Y_Wk
   (shape: (1, 32, 48, 1)) are broadcast to all 11 ensemble members ((11, 32, 48, 1)) at the loss interface.
4. Multi-Head Deep Supervision: Output target dictionary aligns with UNET_RZSM heads:
   ['RZSM_output_1', 'RZSM_output_2', 'RZSM_output_3'].
5. Pure NumPy & TensorFlow Compatibility: Implements both a standalone NumPy batch generator
   and a high-throughput tf.data.Dataset pipeline with prefetching.
6. Checkpoint Persistence: Provides validated save/restore mechanics with state inspection.
"""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Generator, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Parent EX29 constants
ENSEMBLE_MEMBERS: int = 11
GRID_HEIGHT: int = 32
GRID_WIDTH: int = 48
LEAD_CHANNELS: Dict[int, int] = {
    1: 11,  # 3 RZSM lags + 5 ERA5 atm + 3 S2S W1
    2: 12,  # 11 base + 1 recursive y_hat_W1
    3: 5,   # 3 RZSM lags + 2 recursive (y_hat_W1, y_hat_W2)
    4: 6,   # 3 RZSM lags + 3 recursive (y_hat_W1, y_hat_W2, y_hat_W3)
}
OUTPUT_HEADS: List[str] = ["RZSM_output_1", "RZSM_output_2", "RZSM_output_3"]
DEFAULT_FACTOR: float = 0.08


def validate_batch_size(batch_size: int) -> None:
    """
    Validates that the batch size adheres strictly to the 11-member ensemble grouping contract.
    """
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}")
    if batch_size % ENSEMBLE_MEMBERS != 0:
        raise ValueError(
            f"Invalid batch size {batch_size}: Batch size must be an exact multiple of "
            f"ensemble size ({ENSEMBLE_MEMBERS}) to preserve CRPS grouping semantics. "
            f"Valid sizes: {list(range(11, 111, 11))}"
        )


def load_case_npz(npz_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Loads serialized case NPZ file and returns array dictionary.
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Case artifact not found: {npz_path}")

    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def prepare_case_lead_tensors(
    case_data: Dict[str, np.ndarray],
    lead: int = 1,
    y_hat_prev: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts and prepares input tensor X and broadcast ground-truth target Y for a single forecast case.

    Parameters
    ----------
    case_data : Dict[str, np.ndarray]
        Loaded NPZ contents.
    lead : int
        Forecast lead week (1, 2, 3, 4).
    y_hat_prev : Optional[Dict[int, np.ndarray]]
        Dictionary of recursive predictions from prior leads (e.g. {1: y_hat_w1, 2: y_hat_w2, ...}),
        where each prediction tensor has shape (11, 32, 48, 1).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X: shape (11, 32, 48, C_lead), float32
        Y: shape (11, 32, 48, 1), float32 (broadcast across all 11 ensemble members)
    """
    if lead not in LEAD_CHANNELS:
        raise ValueError(f"Invalid lead {lead}. Supported leads: {list(LEAD_CHANNELS.keys())}")

    # 1. Input tensor preparation
    if lead == 1:
        x = case_data["x_w1"].astype(np.float32)
    elif lead == 2:
        x_base = case_data["x_w2_base"].astype(np.float32)
        if y_hat_prev is not None and 1 in y_hat_prev:
            y_rec = y_hat_prev[1].astype(np.float32)
        else:
            # Fallback zero recursive channel for unchained initialization
            y_rec = np.zeros((ENSEMBLE_MEMBERS, GRID_HEIGHT, GRID_WIDTH, 1), dtype=np.float32)
        x = np.concatenate([x_base, y_rec], axis=-1)
    elif lead == 3:
        x_base = case_data["x_w3_base"].astype(np.float32)
        rec_list = []
        for prev_l in (1, 2):
            if y_hat_prev is not None and prev_l in y_hat_prev:
                rec_list.append(y_hat_prev[prev_l].astype(np.float32))
            else:
                rec_list.append(np.zeros((ENSEMBLE_MEMBERS, GRID_HEIGHT, GRID_WIDTH, 1), dtype=np.float32))
        x = np.concatenate([x_base] + rec_list, axis=-1)
    elif lead == 4:
        x_base = case_data["x_w4_base"].astype(np.float32)
        rec_list = []
        for prev_l in (1, 2, 3):
            if y_hat_prev is not None and prev_l in y_hat_prev:
                rec_list.append(y_hat_prev[prev_l].astype(np.float32))
            else:
                rec_list.append(np.zeros((ENSEMBLE_MEMBERS, GRID_HEIGHT, GRID_WIDTH, 1), dtype=np.float32))
        x = np.concatenate([x_base] + rec_list, axis=-1)

    expected_channels = LEAD_CHANNELS[lead]
    if x.shape != (ENSEMBLE_MEMBERS, GRID_HEIGHT, GRID_WIDTH, expected_channels):
        raise ValueError(
            f"Unexpected shape for lead {lead} input: {x.shape}, expected "
            f"({ENSEMBLE_MEMBERS}, {GRID_HEIGHT}, {GRID_WIDTH}, {expected_channels})"
        )

    # 2. Ground-truth target extraction and broadcast across ensemble members
    target_key = f"y_w{lead}"
    y_single = case_data[target_key].astype(np.float32)  # shape: (1, 32, 48, 1)
    if y_single.shape != (1, GRID_HEIGHT, GRID_WIDTH, 1):
        raise ValueError(f"Unexpected target shape for {target_key}: {y_single.shape}")

    # Broadcast from (1, 32, 48, 1) to (11, 32, 48, 1)
    y_broadcast = np.repeat(y_single, repeats=ENSEMBLE_MEMBERS, axis=0)

    return x, y_broadcast


class A0CaseBatchGenerator:
    """
    Streaming batch generator that yields mini-batches of shape (B, 32, 48, C) and
    aligned multi-head targets of shape (B, 32, 48, 1).

    Enforces B = 11 * K, where K >= 1 cases are bundled into each batch.
    Shuffling is performed at the case (forecast cycle) level.
    """

    def __init__(
        self,
        case_paths: List[Union[str, Path]],
        lead: int = 1,
        batch_size: int = 11,
        shuffle: bool = True,
        seed: Optional[int] = 42,
        y_hat_prev: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    ):
        validate_batch_size(batch_size)
        self.case_paths = [Path(p) for p in case_paths]
        if not self.case_paths:
            raise ValueError("No case paths provided to A0CaseBatchGenerator.")

        self.lead = lead
        self.batch_size = batch_size
        self.cases_per_batch = batch_size // ENSEMBLE_MEMBERS
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.y_hat_prev = y_hat_prev or {}

    def __len__(self) -> int:
        """Returns the number of complete batches per epoch."""
        return len(self.case_paths) // self.cases_per_batch

    def __iter__(self) -> Generator[Tuple[np.ndarray, Dict[str, np.ndarray]], None, None]:
        indices = np.arange(len(self.case_paths))
        if self.shuffle:
            self.rng.shuffle(indices)

        num_batches = len(self)
        for b in range(num_batches):
            batch_case_indices = indices[b * self.cases_per_batch : (b + 1) * self.cases_per_batch]

            x_list = []
            y_list = []

            for idx in batch_case_indices:
                path = self.case_paths[idx]
                case_data = load_case_npz(path)
                case_id = str(case_data.get("case_id", path.stem))

                case_y_hat = self.y_hat_prev.get(case_id, None)
                x_case, y_case = prepare_case_lead_tensors(
                    case_data=case_data,
                    lead=self.lead,
                    y_hat_prev=case_y_hat,
                )
                x_list.append(x_case)
                y_list.append(y_case)

            x_batch = np.concatenate(x_list, axis=0)  # shape: (B, 32, 48, C_lead)
            y_batch = np.concatenate(y_list, axis=0)  # shape: (B, 32, 48, 1)

            # Deep supervision dictionary matching UNET_RZSM heads
            y_dict = {head: y_batch for head in OUTPUT_HEADS}

            yield x_batch, y_dict


def crps_exact_analytical(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eval_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Independent reference implementation of the standard continuous ranked probability score (CRPS)
    for an ensemble forecast:

        CRPS(F, y) = 1/M * sum_{m=1}^M |x_m - y| - 1/(2*M^2) * sum_{m=1}^M sum_{n=1}^M |x_m - x_n|

    Evaluates across groups of M=11 ensemble members and averages across spatial pixels and cases.
    If eval_mask is provided, evaluates strictly over active land cells.
    """
    y_t = np.squeeze(np.asarray(y_true, dtype=np.float32))
    y_p = np.squeeze(np.asarray(y_pred, dtype=np.float32))

    if y_p.ndim == 1:
        # 1D toy vector of ensemble members
        m_count = len(y_p)
        y_val = float(np.squeeze(y_t))
        mae = float(np.mean(np.abs(y_p - y_val)))
        diff_matrix = np.abs(y_p[:, None] - y_p[None, :])
        pairwise = float(np.sum(diff_matrix) / (2.0 * m_count * m_count))
        return mae - pairwise

    # Grid case: (B, H, W)
    batch_size = y_p.shape[0]
    if batch_size % ENSEMBLE_MEMBERS != 0:
        raise ValueError(f"Batch size ({batch_size}) must be divisible by {ENSEMBLE_MEMBERS}")

    num_cases = batch_size // ENSEMBLE_MEMBERS

    # Align y_t if single target per case
    if y_t.ndim == 2:
        y_t = y_t[np.newaxis, ...]
    if y_t.shape[0] == num_cases and batch_size == num_cases * ENSEMBLE_MEMBERS:
        y_t = np.repeat(y_t, ENSEMBLE_MEMBERS, axis=0)

    total_crps = 0.0
    for i in range(num_cases):
        start_idx = i * ENSEMBLE_MEMBERS
        end_idx = start_idx + ENSEMBLE_MEMBERS
        case_yt = y_t[start_idx:end_idx]  # (11, H, W)
        case_yp = y_p[start_idx:end_idx]  # (11, H, W)

        if eval_mask is not None:
            case_yt_eval = case_yt[:, eval_mask]  # (11, 126)
            case_yp_eval = case_yp[:, eval_mask]  # (11, 126)
            mae = float(np.mean(np.abs(case_yp_eval - case_yt_eval)))
            diff = np.abs(case_yp_eval[:, np.newaxis, :] - case_yp_eval[np.newaxis, :, :])
            pairwise = np.sum(diff, axis=(0, 1)) / (2.0 * ENSEMBLE_MEMBERS * ENSEMBLE_MEMBERS)
            mean_pairwise = float(np.nanmean(pairwise))
            total_crps += (mae - mean_pairwise)
        else:
            mae = np.mean(np.abs(case_yp - case_yt))
            diff = np.abs(case_yp[:, np.newaxis, :, :] - case_yp[np.newaxis, :, :, :])
            pairwise = np.sum(diff, axis=(0, 1)) / (2.0 * ENSEMBLE_MEMBERS * ENSEMBLE_MEMBERS)
            mean_pairwise = float(np.nanmean(pairwise))
            total_crps += (mae - mean_pairwise)

    return float(total_crps / num_cases)


def crps2d_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    factor: float = DEFAULT_FACTOR,
    eval_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Pure NumPy implementation of the author's spatial CRPS loss function (crps2d_tf).

    CRPS_exp = MAE - factor * std_ensemble
    Evaluates groups of 11 ensemble members independently and returns the mean score across cases.
    If eval_mask is provided, evaluates strictly over active land cells.

    Supports both:
    1. Single target per case: shape (1, H, W, 1) or (K, H, W, 1)
    2. Broadcast target: shape (11, H, W, 1) or (11*K, H, W, 1)
    Both representations evaluate to the mathematically identical CRPS result.
    """
    y_t = np.asarray(y_true, dtype=np.float32)
    y_p = np.asarray(y_pred, dtype=np.float32)

    # Squeeze trailing channel dimension if present: (B, H, W, 1) -> (B, H, W)
    if y_t.ndim == 4 and y_t.shape[-1] == 1:
        y_t = y_t[..., 0]
    if y_p.ndim == 4 and y_p.shape[-1] == 1:
        y_p = y_p[..., 0]

    # Handle 0D scalar target
    if y_t.ndim == 0:
        y_t = y_t.reshape(1)

    batch_size = y_p.shape[0]
    if batch_size % ENSEMBLE_MEMBERS != 0:
        raise ValueError(f"Batch size ({batch_size}) must be divisible by {ENSEMBLE_MEMBERS}")

    num_cases = batch_size // ENSEMBLE_MEMBERS

    # Target broadcasting invariance: if target has shape (K, H, W) or (H, W) or (K,) for single cases, broadcast to (11*K, ...)
    if y_t.ndim == 2 and y_p.ndim == 3:
        y_t = y_t[np.newaxis, ...]
    if y_t.shape[0] == num_cases and batch_size == num_cases * ENSEMBLE_MEMBERS:
        y_t = np.repeat(y_t, ENSEMBLE_MEMBERS, axis=0)
    elif y_t.shape[0] != batch_size:
        raise ValueError(f"Target batch size ({y_t.shape[0]}) does not match prediction batch size ({batch_size}) or num_cases ({num_cases})")

    total_crps = 0.0

    for i in range(num_cases):
        start_idx = i * ENSEMBLE_MEMBERS
        end_idx = start_idx + ENSEMBLE_MEMBERS

        case_yt = y_t[start_idx:end_idx]  # shape: (11, 32, 48)
        case_yp = y_p[start_idx:end_idx]  # shape: (11, 32, 48)

        if eval_mask is not None:
            case_yt_eval = case_yt[:, eval_mask]  # (11, 126)
            case_yp_eval = case_yp[:, eval_mask]  # (11, 126)
            mae = float(np.mean(np.abs(case_yp_eval - case_yt_eval)))
            member_std = np.nanstd(case_yp_eval, axis=0)
            dist = float(np.nanmean(member_std))
        else:
            mae = float(np.mean(np.abs(case_yp - case_yt)))
            member_std = np.nanstd(case_yp, axis=0)
            dist = float(np.nanmean(member_std))

        case_crps = mae - factor * dist
        total_crps += case_crps

    return float(total_crps / num_cases)



def create_a0_tf_dataset(
    case_paths: List[Union[str, Path]],
    lead: int = 1,
    batch_size: int = 11,
    shuffle: bool = True,
    seed: Optional[int] = 42,
    prefetch: bool = True,
) -> Any:
    """
    Constructs a high-throughput tf.data.Dataset from serialized A0 cases.

    Parameters
    ----------
    case_paths : List[Union[str, Path]]
        Paths to case NPZ files.
    lead : int
        Forecast lead week (1..4).
    batch_size : int
        Batch size, must be multiple of 11.
    shuffle : bool
        Whether to shuffle cases each epoch.
    seed : Optional[int]
        Random generator seed.
    prefetch : bool
        Whether to prefetch with AUTOTUNE.

    Returns
    -------
    tf.data.Dataset
        Streaming dataset yielding (X, {RZSM_output_1: Y, RZSM_output_2: Y, RZSM_output_3: Y}).
    """
    try:
        import tensorflow as tf
    except ImportError as err:
        raise ImportError(
            "TensorFlow is not installed in the active Python environment. "
            "Use A0CaseBatchGenerator directly for pure NumPy workflows, or execute inside Google Colab."
        ) from err

    validate_batch_size(batch_size)
    num_channels = LEAD_CHANNELS[lead]

    def generator_fn():
        gen = A0CaseBatchGenerator(
            case_paths=case_paths,
            lead=lead,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )
        for x_b, y_dict in gen:
            yield x_b, y_dict

    output_signature = (
        tf.TensorSpec(shape=(batch_size, GRID_HEIGHT, GRID_WIDTH, num_channels), dtype=tf.float32),
        {
            head: tf.TensorSpec(shape=(batch_size, GRID_HEIGHT, GRID_WIDTH, 1), dtype=tf.float32)
            for head in OUTPUT_HEADS
        },
    )

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_signature=output_signature,
    )

    if prefetch:
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def save_a0_checkpoint(
    model: Any,
    epoch: int,
    loss: float,
    checkpoint_dir: Union[str, Path],
    filename_prefix: str = "a0_checkpoint",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Saves model weights and execution metadata to checkpoint directory.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    weights_path = checkpoint_dir / f"{filename_prefix}_epoch{epoch:03d}.weights.h5"
    meta_path = checkpoint_dir / f"{filename_prefix}_epoch{epoch:03d}_meta.json"

    # Save model weights
    if hasattr(model, "save_weights"):
        model.save_weights(str(weights_path))
    else:
        # Fallback dictionary for testing mock models
        weights_dict = {f"layer_{i}": w for i, w in enumerate(model.get_weights())}
        np.savez_compressed(weights_path.with_suffix(".npz"), **weights_dict)

    # Save metadata
    meta = {
        "epoch": epoch,
        "loss": float(loss),
        "weights_file": str(weights_path.name),
        "extra": metadata or {},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Checkpoint successfully saved: {weights_path}")
    return weights_path


def restore_a0_checkpoint(
    model: Any,
    weights_path: Union[str, Path],
) -> None:
    """
    Restores model weights from saved checkpoint file.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        # Check npz alternative
        npz_alt = weights_path.with_suffix(".npz")
        if npz_alt.exists():
            weights_path = npz_alt
        else:
            raise FileNotFoundError(f"Checkpoint weights not found: {weights_path}")

    if str(weights_path).endswith(".npz"):
        with np.load(weights_path) as data:
            weights = [data[k] for k in sorted(data.files, key=lambda x: int(x.split("_")[-1]))]
        if hasattr(model, "set_weights") and len(weights) == len(model.get_weights()):
            model.set_weights(weights)
        elif hasattr(model, "trainable_variables") and len(weights) == len(model.trainable_variables):
            for w_target, w_arr in zip(model.trainable_variables, weights):
                w_target.assign(w_arr)
        else:
            # Fallback
            model.set_weights(weights)
    elif hasattr(model, "load_weights"):
        model.load_weights(str(weights_path))
    else:
        raise ValueError(f"Cannot restore weights into object: {type(model)}")

    logger.info(f"Checkpoint successfully restored from {weights_path}")


def save_a0_training_state(
    model: Any,
    optimizer: Any,
    epoch: int,
    step: int,
    learning_rate: float,
    loss: float,
    checkpoint_dir: Union[str, Path],
    filename_prefix: str = "a0_training_state",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Saves complete training state (model weights, optimizer variables, step/epoch counters,
    and metadata) to distinguish full training-state checkpointing from model-weight-only checkpointing.
    Preserves exact bit-for-bit numpy arrays of trainable and optimizer variables.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    weights_file = checkpoint_dir / f"{filename_prefix}_epoch{epoch:03d}_step{step:06d}_weights.npz"
    opt_file = checkpoint_dir / f"{filename_prefix}_epoch{epoch:03d}_step{step:06d}_opt.npz"
    meta_file = checkpoint_dir / f"{filename_prefix}_epoch{epoch:03d}_step{step:06d}_meta.json"

    # Save model trainable variables directly
    if hasattr(model, "trainable_variables"):
        w_dict = {f"var_{i}": w.numpy() for i, w in enumerate(model.trainable_variables)}
    elif hasattr(model, "get_weights"):
        w_dict = {f"var_{i}": w for i, w in enumerate(model.get_weights())}
    else:
        raise ValueError(f"Cannot extract weights from object: {type(model)}")
    np.savez_compressed(weights_file, **w_dict)

    # Save optimizer weights
    opt_weights = []
    if hasattr(optimizer, "variables") and optimizer.variables:
        opt_weights = [v.numpy() for v in optimizer.variables]
    elif hasattr(optimizer, "get_weights"):
        opt_weights = optimizer.get_weights()
    opt_dict = {f"opt_{i}": w for i, w in enumerate(opt_weights)}
    np.savez_compressed(opt_file, **opt_dict)

    meta = {
        "checkpoint_type": "FULL_TRAINING_STATE",
        "epoch": epoch,
        "step": step,
        "learning_rate": float(learning_rate),
        "loss": float(loss),
        "weights_file": str(weights_file.name),
        "optimizer_file": str(opt_file.name),
        "extra": metadata or {},
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Full training state successfully saved: {meta_file}")
    return meta_file


def restore_a0_training_state(
    model: Any,
    optimizer: Any,
    meta_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Restores full training state (model weights, optimizer variables, step, epoch).
    Returns metadata dict.
    """
    meta_path = Path(meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Training state metadata not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    checkpoint_dir = meta_path.parent
    weights_path = checkpoint_dir / meta["weights_file"]
    opt_path = checkpoint_dir / meta["optimizer_file"]

    # Restore model trainable variables directly
    if weights_path.suffix == ".npz" and hasattr(model, "trainable_variables"):
        with np.load(weights_path) as data:
            w_keys = sorted(data.files, key=lambda x: int(x.split("_")[-1]))
            if len(w_keys) == len(model.trainable_variables):
                for w_target, k in zip(model.trainable_variables, w_keys):
                    w_target.assign(data[k])
            else:
                restore_a0_checkpoint(model, weights_path)
    else:
        restore_a0_checkpoint(model, weights_path)

    # Build optimizer variables against model trainable variables
    if hasattr(optimizer, "build") and hasattr(model, "trainable_variables"):
        try:
            optimizer.build(model.trainable_variables)
        except Exception:
            pass

    # Restore optimizer variables directly
    if opt_path.exists():
        with np.load(opt_path, allow_pickle=True) as data:
            opt_keys = sorted(data.files, key=lambda x: int(x.split("_")[-1]))
            opt_weights = [data[k] for k in opt_keys]

        if hasattr(optimizer, "variables") and len(optimizer.variables) == len(opt_weights):
            for v_target, w in zip(optimizer.variables, opt_weights):
                v_target.assign(w)
        elif hasattr(optimizer, "set_weights"):
            try:
                optimizer.set_weights(opt_weights)
            except Exception as e:
                logger.warning(f"Could not fully set optimizer weights: {e}")

    logger.info(f"Full training state restored from {meta_path} (epoch {meta.get('epoch')}, step {meta.get('step')})")
    return meta


def normalize_assembled_case(
    case_data: Dict[str, np.ndarray],
    norm_cfg: Optional[Dict[str, Any]] = None,
    eval_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Normalizes real assembled case arrays into [0, 1] unit interval using
    frozen training normalization contract (contracts/A0/normalization_parameters.yaml)
    and enforces the ocean-buffer zero-filling invariant (~eval_mask -> 0.0).

    Returns dictionary with normalized arrays:
      - x_w1: (11, 32, 48, 11)
      - x_w2_base: (11, 32, 48, 11)
      - x_w3_base: (11, 32, 48, 3)
      - x_w4_base: (11, 32, 48, 3)
      - y_w1, y_w2, y_w3, y_w4: (11, 32, 48, 1) broadcast across 11 members
    """
    if norm_cfg is None:
        import yaml
        norm_file = Path(__file__).resolve().parent.parent.parent / "contracts" / "A0" / "normalization_parameters.yaml"
        with open(norm_file, "r", encoding="utf-8") as f:
            norm_cfg = yaml.safe_load(f)

    rzsm_p = norm_cfg["rzsm_parameters"]["seasonal_anomaly"]
    rzsm_min, rzsm_max = float(rzsm_p["min"]), float(rzsm_p["max"])

    atm_vars = ["pwat", "spfh", "tmax", "diff_temp", "hgt_pres"]
    s2s_vars = ["t2m", "d2m", "tcw"]

    def _scale(arr: np.ndarray, mi: float, ma: float) -> np.ndarray:
        denom = ma - mi
        if np.isclose(denom, 0.0):
            raise ValueError(f"Degenerate bounds: max ({ma}) == min ({mi})")
        return np.clip((arr - mi) / denom, 0.0, 1.0).astype(np.float32)

    # Lead 1: 3 RZSM lags + 5 ERA5 surface + 3 S2S W1 = 11 channels
    x_w1_norm = np.zeros_like(case_data["x_w1"], dtype=np.float32)
    for c in range(3):
        x_w1_norm[..., c] = _scale(case_data["x_w1"][..., c], rzsm_min, rzsm_max)
    for i, var in enumerate(atm_vars):
        p = norm_cfg["atmospheric_parameters"][var]
        x_w1_norm[..., 3 + i] = _scale(case_data["x_w1"][..., 3 + i], float(p["min"]), float(p["max"]))
    for i, var in enumerate(s2s_vars):
        p = norm_cfg["s2s_parameters"]["lead_1"][var]
        x_w1_norm[..., 8 + i] = _scale(case_data["x_w1"][..., 8 + i], float(p["min"]), float(p["max"]))

    # Lead 2 base: 3 RZSM lags + 5 ERA5 surface + 3 S2S W2 = 11 channels
    x_w2_base_norm = np.zeros_like(case_data["x_w2_base"], dtype=np.float32)
    for c in range(3):
        x_w2_base_norm[..., c] = _scale(case_data["x_w2_base"][..., c], rzsm_min, rzsm_max)
    for i, var in enumerate(atm_vars):
        p = norm_cfg["atmospheric_parameters"][var]
        x_w2_base_norm[..., 3 + i] = _scale(case_data["x_w2_base"][..., 3 + i], float(p["min"]), float(p["max"]))
    for i, var in enumerate(s2s_vars):
        p = norm_cfg["s2s_parameters"]["lead_2"][var]
        x_w2_base_norm[..., 8 + i] = _scale(case_data["x_w2_base"][..., 8 + i], float(p["min"]), float(p["max"]))

    # Lead 3 base: 3 RZSM lags
    x_w3_base_norm = np.zeros_like(case_data["x_w3_base"], dtype=np.float32)
    for c in range(3):
        x_w3_base_norm[..., c] = _scale(case_data["x_w3_base"][..., c], rzsm_min, rzsm_max)

    # Lead 4 base: 3 RZSM lags
    x_w4_base_norm = np.zeros_like(case_data["x_w4_base"], dtype=np.float32)
    for c in range(3):
        x_w4_base_norm[..., c] = _scale(case_data["x_w4_base"][..., c], rzsm_min, rzsm_max)

    # Targets Y_w1..Y_w4
    y_norm = {}
    for l in [1, 2, 3, 4]:
        raw_y = case_data[f"y_w{l}"].astype(np.float32)
        if raw_y.shape[0] == 1:
            raw_y = np.repeat(raw_y, 11, axis=0)
        y_norm[f"y_w{l}"] = _scale(raw_y, rzsm_min, rzsm_max)

    # Ocean buffer zero-filling invariant
    if eval_mask is not None:
        ocean = ~eval_mask
        x_w1_norm[:, ocean, :] = 0.0
        x_w2_base_norm[:, ocean, :] = 0.0
        x_w3_base_norm[:, ocean, :] = 0.0
        x_w4_base_norm[:, ocean, :] = 0.0
        for l in [1, 2, 3, 4]:
            y_norm[f"y_w{l}"][:, ocean, :] = 0.0

    return {
        "x_w1": x_w1_norm,
        "x_w2_base": x_w2_base_norm,
        "x_w3_base": x_w3_base_norm,
        "x_w4_base": x_w4_base_norm,
        **y_norm,
    }


def crps2d_tf(
    y_true: Any,
    y_pred: Any,
    factor: float = DEFAULT_FACTOR,
    eval_mask: Optional[Any] = None,
) -> Any:
    """
    TensorFlow graph-compatible implementation of the spatial CRPS loss function (parent EX29).

        CRPS = MAE_eval - factor * std_spatial_ensemble

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth target tensor of shape (B, 32, 48, 1), where B is a multiple of 11.
    y_pred : tf.Tensor
        Model output predictions of shape (B, 32, 48, 1).
    factor : float
        Spread reward factor (default: 0.08).
    eval_mask : Optional[tf.Tensor or np.ndarray]
        Binary evaluation mask of shape (32, 48) indicating 126 active land cells.

    Returns
    -------
    tf.Tensor
        Scalar spatial CRPS loss.
    """
    import tensorflow as tf

    y_t = tf.cast(y_true, tf.float32)
    y_p = tf.cast(y_pred, tf.float32)

    # Apply evaluation mask if provided
    if eval_mask is not None:
        mask = tf.cast(eval_mask, tf.float32)
        if len(mask.shape) == 2:
            mask = tf.expand_dims(tf.expand_dims(mask, 0), -1)  # (1, 32, 48, 1)
        diff = tf.abs(y_p - y_t) * mask
        num_eval_cells = tf.reduce_sum(mask)
        denom = num_eval_cells * tf.cast(tf.shape(y_t)[0], tf.float32)
        mae = tf.reduce_sum(diff) / tf.maximum(denom, 1.0)
    else:
        mae = tf.reduce_mean(tf.abs(y_p - y_t))

    # Ensemble spread calculation over chunks of 11 members
    batch_size = tf.shape(y_p)[0]
    num_cases = batch_size // ENSEMBLE_MEMBERS

    h = tf.shape(y_p)[1]
    w = tf.shape(y_p)[2]
    c = tf.shape(y_p)[3]

    y_p_cases = tf.reshape(y_p[: num_cases * ENSEMBLE_MEMBERS], (num_cases, ENSEMBLE_MEMBERS, h, w, c))
    # Numerically stable standard deviation: sqrt(var + eps) prevents NaN gradients at zero spread
    ens_var = tf.math.reduce_variance(y_p_cases, axis=1)  # (num_cases, H, W, C)
    ens_std = tf.sqrt(ens_var + 1e-7)

    if eval_mask is not None:
        mask_cases = tf.tile(mask, [num_cases, 1, 1, 1])
        ens_std = ens_std * mask_cases
        spread = tf.reduce_sum(ens_std) / tf.maximum(tf.reduce_sum(mask_cases), 1.0)
    else:
        spread = tf.reduce_mean(ens_std)

    return mae - factor * spread


