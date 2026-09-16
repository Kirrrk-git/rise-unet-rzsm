#!/usr/bin/env python3
"""
16_train_a0_production.py
-------------------------
Step 21K.3: Full Three-Seed Model A0 Production Training across all 4 Leads.

Governed by:
  - contracts/A0/mindanao_a0_production_contract.yaml
  - contracts/A0/normalization_parameters.yaml
  - contracts/A0/VERIFICATION_STATUS.yaml
  - mindanao_adaptation_master_plan.md (Step 21K.3)

Operational Contracts:
  1. Seeds: [42, 123, 456] (Predeclared, immutable).
  2. Sequential Recursive Training: Lead 1 -> y_hat_W1 -> Lead 2 -> y_hat_W2 -> Lead 3 -> y_hat_W3 -> Lead 4.
  3. Architecture: Genuine 1.63M-parameter UNET_RZSM with 3 deep-supervision heads [RZSM_output_1/2/3].
     - Lead 1: Cin=11, Params=1,627,139
     - Lead 2: Cin=12, Params=1,630,307
     - Lead 3: Cin=5,  Params=1,608,131
     - Lead 4: Cin=6,  Params=1,611,299
  4. Objective: Spatial CRPS loss (MAE_eval - 0.08 * std_spatial) with deep supervision weights [0.2, 0.3, 0.5].
  5. Optimization: Adam (lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7), ReduceLROnPlateau (factor=0.5, patience=3).
  6. Batch Size: Exact multiple of 11 (default: B=33, 3 cases x 11 ensemble members).
  7. Regularization & Early Stopping: Max epochs 40, patience 8 based on validation CRPS.
  8. Checkpoint Metric: Primary checkpoint per seed/lead is selected by minimum validation CRPS.
  9. Telemetry & Cloud Lake: Checkpoints, logs, and evaluation metrics synced to gs://rise-unet-rzsm/.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_a0_production")

# Default production specifications
DEFAULT_SEEDS: List[int] = [42, 123, 456]
DEFAULT_LEADS: List[int] = [1, 2, 3, 4]
DEFAULT_BATCH_SIZE: int = 33
DEFAULT_EPOCHS: int = 40
DEFAULT_PATIENCE: int = 8
DEFAULT_LEARNING_RATE: float = 0.0001
DEFAULT_DEEP_SUPERVISION_WEIGHTS: List[float] = [0.2, 0.3, 0.5]
ENSEMBLE_MEMBERS: int = 11
GRID_HEIGHT: int = 32
GRID_WIDTH: int = 48
ACTIVE_CELL_COUNT: int = 126

GCS_BUCKET: str = "gs://rise-unet-rzsm"


def load_eval_mask(repo_root: Path) -> np.ndarray:
    """Loads authoritative 126-cell binary evaluation mask."""
    mask_path = repo_root / "processed" / "grid" / "mindanao_eval_mask_025.nc"
    if not mask_path.exists():
        logger.info(f"Downloading evaluation mask from {GCS_BUCKET}/processed/grid/...")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["gcloud", "storage", "cp", f"{GCS_BUCKET}/processed/grid/mindanao_eval_mask_025.nc", str(mask_path)],
                check=True,
                capture_output=True,
            )
        except Exception:
            subprocess.run(
                ["gsutil", "cp", f"{GCS_BUCKET}/processed/grid/mindanao_eval_mask_025.nc", str(mask_path)],
                check=True,
            )
    ds = xr.open_dataset(mask_path)
    mask = ds["evaluation_mask"].values.astype(bool)
    assert np.sum(mask) == ACTIVE_CELL_COUNT, f"Evaluation mask must have exactly {ACTIVE_CELL_COUNT} active cells"
    return mask


def load_manifest_cases(
    manifest_path: Path,
    cases_dir: Path,
    limit: Optional[int] = None,
) -> List[Path]:
    """Loads case NPZ paths listed in a split manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    case_col = "case_id" if "case_id" in df.columns else df.columns[0]
    case_ids = df[case_col].tolist()

    if limit is not None and limit > 0:
        case_ids = case_ids[:limit]

    case_paths = []
    for cid in case_ids:
        # Check standard naming
        p = cases_dir / f"{cid}.npz"
        if not p.exists():
            # Check with CASE_ prefix
            alt = cases_dir / f"CASE_{cid}.npz"
            if alt.exists():
                p = alt
        case_paths.append(p)

    return case_paths


def evaluate_lead_metrics(
    model: Any,
    case_paths: List[Path],
    lead: int,
    y_hat_prev: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    eval_mask: Optional[np.ndarray] = None,
    batch_size: int = 11,
) -> Dict[str, float]:
    """
    Evaluates validation partition metrics for a given lead:
      - val_loss: Weighted multi-head spatial CRPS loss
      - val_crps: Final head (RZSM_output_3) spatial CRPS
      - val_mae: Active-cell Mean Absolute Error
      - val_rmse: Active-cell Root Mean Square Error
      - val_acc: Active-cell Anomaly Correlation Coefficient
    """
    from src.data.tf_dataset import (
        prepare_case_lead_tensors,
        crps2d_numpy,
        crps_exact_analytical,
        OUTPUT_HEADS,
    )

    y_preds_list = []
    y_trues_list = []

    for path in case_paths:
        if not path.exists():
            continue
        with np.load(path) as data:
            case_data = {k: data[k] for k in data.files}

        cid = path.stem.replace("CASE_", "")
        prev_preds = None
        if y_hat_prev is not None and cid in y_hat_prev:
            prev_preds = y_hat_prev[cid]

        x_case, y_case = prepare_case_lead_tensors(
            case_data=case_data,
            lead=lead,
            y_hat_prev=prev_preds,
        )

        preds = model(x_case, training=False)
        # Primary output is Head 3 (RZSM_output_3)
        p_head3 = preds[2].numpy() if hasattr(preds[2], "numpy") else preds[2]

        y_preds_list.append(p_head3)
        y_trues_list.append(y_case)

    if not y_preds_list:
        return {
            "val_crps": 999.0,
            "val_spatial_crps_proxy": 999.0,
            "val_exact_crps": 999.0,
            "val_mae": 999.0,
            "val_rmse": 999.0,
            "val_acc": 0.0,
        }

    y_pred_all = np.concatenate(y_preds_list, axis=0)  # (N*11, 32, 48, 1)
    y_true_all = np.concatenate(y_trues_list, axis=0)  # (N*11, 32, 48, 1)

    # Active land cell masking
    if eval_mask is not None:
        p_active = y_pred_all[:, eval_mask, 0]  # (N*11, 126)
        t_active = y_true_all[:, eval_mask, 0]  # (N*11, 126)
    else:
        p_active = y_pred_all[..., 0]
        t_active = y_true_all[..., 0]

    # Metrics
    mae = float(np.mean(np.abs(p_active - t_active)))
    rmse = float(np.sqrt(np.mean((p_active - t_active) ** 2)))

    # 1. Spatial CRPS Proxy (Parent EX29 author formulation: MAE - 0.08 * spatial_spread)
    crps_proxy = float(crps2d_numpy(y_true_all, y_pred_all, factor=0.08))

    # 2. Exact Ensemble CRPS (Standard Gneiting & Raftery 2007 / Hersbach 2000 formulation)
    crps_exact = float(crps_exact_analytical(y_true_all, y_pred_all))

    # Anomaly Correlation Coefficient (ACC)
    p_mean = np.mean(p_active)
    t_mean = np.mean(t_active)
    p_anom = p_active - p_mean
    t_anom = t_active - t_mean
    denom = np.sqrt(np.sum(p_anom ** 2) * np.sum(t_anom ** 2))
    acc = float(np.sum(p_anom * t_anom) / denom) if denom > 1e-12 else 0.0

    return {
        "val_crps": crps_proxy,                       # Checkpoint selection metric (Parent EX29 objective)
        "val_spatial_crps_proxy": crps_proxy,        # Spread-adjusted spatial proxy: MAE - 0.08 * std
        "val_exact_crps": crps_exact,                # Standard analytical ensemble CRPS (Hersbach 2000)
        "val_mae": mae,
        "val_rmse": rmse,
        "val_acc": acc,
    }


def generate_lead_predictions(
    model: Any,
    case_paths: List[Path],
    lead: int,
    y_hat_prev: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    eval_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Generates and returns frozen model predictions (M=11, 32, 48, 1) for a set of cases
    to serve as recursive input features for subsequent leads.
    """
    from src.data.tf_dataset import prepare_case_lead_tensors

    preds_dict = {}
    for path in case_paths:
        if not path.exists():
            continue
        with np.load(path) as data:
            case_data = {k: data[k] for k in data.files}

        cid = path.stem.replace("CASE_", "")
        prev_preds = None
        if y_hat_prev is not None and cid in y_hat_prev:
            prev_preds = y_hat_prev[cid]

        x_case, _ = prepare_case_lead_tensors(
            case_data=case_data,
            lead=lead,
            y_hat_prev=prev_preds,
        )

        preds = model(x_case, training=False)
        # Primary output is Head 3
        p3 = preds[2].numpy() if hasattr(preds[2], "numpy") else preds[2]

        # Enforce ocean buffer zero-filling on stored recursive predictions
        if eval_mask is not None:
            p3[:, ~eval_mask, :] = 0.0

        preds_dict[cid] = p3.astype(np.float32)

    return preds_dict


def train_single_lead(
    lead: int,
    seed: int,
    train_paths: List[Path],
    val_paths: List[Path],
    y_hat_train_prev: Optional[Dict[str, Dict[int, np.ndarray]]],
    y_hat_val_prev: Optional[Dict[str, Dict[int, np.ndarray]]],
    eval_mask: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Executes full production training for a single lead and random seed.
    """
    import tensorflow as tf
    from src.models.a0_unet import build_a0_unet, EXPECTED_A0_PARAMETER_COUNTS, LEAD_CHANNELS
    from src.data.tf_dataset import (
        crps2d_tf,
        save_a0_checkpoint,
        save_a0_training_state,
        restore_a0_checkpoint,
        prepare_case_lead_tensors,
        OUTPUT_HEADS,
    )

    # Set seed for determinism
    tf.random.set_seed(seed)
    np.random.seed(seed)

    logger.info("=" * 80)
    logger.info(f"--> TRAINING LEAD {lead} | SEED {seed} (Cin={LEAD_CHANNELS[lead]}, B={args.batch_size})")
    logger.info("=" * 80)

    # 1. Build genuine UNET_RZSM
    model = build_a0_unet(lead=lead)
    expected_params = EXPECTED_A0_PARAMETER_COUNTS[lead]
    actual_params = model.count_params()
    assert actual_params == expected_params, (
        f"Lead {lead} parameter count mismatch: expected {expected_params}, got {actual_params}"
    )
    logger.info(f"Model instantiated: {model.name} | Parameters: {actual_params:,}")

    # 2. Setup optimizer and learning rate schedule
    curr_lr = args.learning_rate
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=curr_lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
    )

    # Output directory
    seed_lead_dir = Path(args.output_dir) / f"seed_{seed}" / f"lead_{lead}"
    seed_lead_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load normalized training and validation tensors into memory for maximum GPU throughput
    logger.info(f"Pre-loading tensors for {len(train_paths)} train cases and {len(val_paths)} val cases...")
    train_x_list, train_y_list = [], []
    for path in train_paths:
        if not path.exists():
            continue
        with np.load(path) as d:
            case_data = {k: d[k] for k in d.files}
        cid = path.stem.replace("CASE_", "")
        prev_preds = y_hat_train_prev.get(cid) if y_hat_train_prev else None
        x_c, y_c = prepare_case_lead_tensors(case_data, lead=lead, y_hat_prev=prev_preds)
        train_x_list.append(x_c)
        train_y_list.append(y_c)

    num_train_cases = len(train_x_list)
    assert num_train_cases > 0, "No training cases available!"
    logger.info(f"Loaded {num_train_cases} training cases.")

    # Training state trackers
    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_crps": [],
        "val_spatial_crps_proxy": [],
        "val_exact_crps": [],
        "val_mae": [],
        "val_rmse": [],
        "val_acc": [],
        "learning_rate": [],
    }

    best_val_crps = float("inf")
    best_weights_path = seed_lead_dir / "best_model.weights.h5"
    patience_counter = 0
    lr_patience_counter = 0

    cases_per_batch = args.batch_size // ENSEMBLE_MEMBERS

    # 3. Epoch Training Loop
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Case-level shuffling
        case_indices = np.random.permutation(num_train_cases)
        train_losses = []

        # Step through batches
        for b_idx in range(0, num_train_cases, cases_per_batch):
            b_cases = case_indices[b_idx : b_idx + cases_per_batch]
            if len(b_cases) < cases_per_batch:
                continue  # Drop remainder to strictly enforce ensemble multiple B

            batch_x = np.concatenate([train_x_list[i] for i in b_cases], axis=0)  # (B, 32, 48, Cin)
            batch_y = np.concatenate([train_y_list[i] for i in b_cases], axis=0)  # (B, 32, 48, 1)

            with tf.GradientTape() as tape:
                preds = model(batch_x, training=True)
                loss1 = crps2d_tf(batch_y, preds[0], factor=0.08, eval_mask=eval_mask)
                loss2 = crps2d_tf(batch_y, preds[1], factor=0.08, eval_mask=eval_mask)
                loss3 = crps2d_tf(batch_y, preds[2], factor=0.08, eval_mask=eval_mask)

                # Deep supervision weights [0.2, 0.3, 0.5]
                w = DEFAULT_DEEP_SUPERVISION_WEIGHTS
                total_loss = w[0] * loss1 + w[1] * loss2 + w[2] * loss3

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_losses.append(float(total_loss.numpy()))

        mean_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # Evaluate on validation partition
        val_metrics = evaluate_lead_metrics(
            model=model,
            case_paths=val_paths,
            lead=lead,
            y_hat_prev=y_hat_val_prev,
            eval_mask=eval_mask,
            batch_size=args.batch_size,
        )

        val_crps = val_metrics["val_crps"]
        val_mae = val_metrics["val_mae"]
        val_rmse = val_metrics["val_rmse"]
        val_acc = val_metrics["val_acc"]
        epoch_dur = time.time() - epoch_start

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(val_crps)
        history["val_crps"].append(val_crps)
        history["val_spatial_crps_proxy"].append(val_metrics.get("val_spatial_crps_proxy", val_crps))
        history["val_exact_crps"].append(val_metrics.get("val_exact_crps", 0.0))
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(float(curr_lr))

        improved_flag = ""
        # Checkpoint selection based on minimum validation CRPS
        if val_crps < best_val_crps:
            best_val_crps = val_crps
            patience_counter = 0
            lr_patience_counter = 0
            improved_flag = " [BEST CHECKPOINT SAVED]"

            # Save primary best model weights
            save_a0_checkpoint(
                model=model,
                epoch=epoch,
                loss=val_crps,
                checkpoint_dir=seed_lead_dir,
                filename_prefix="best_model",
                metadata={"val_metrics": val_metrics, "lead": lead, "seed": seed},
            )

            # Save full training state
            save_a0_training_state(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=epoch * (num_train_cases // cases_per_batch),
                learning_rate=float(curr_lr),
                loss=val_crps,
                checkpoint_dir=seed_lead_dir,
                filename_prefix="best_state",
            )
        else:
            patience_counter += 1
            lr_patience_counter += 1

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} ({epoch_dur:.1f}s) | "
            f"Train Loss: {mean_train_loss:.4f} | "
            f"Val Proxy CRPS: {val_crps:.4f} | "
            f"Val Exact CRPS: {val_metrics.get('val_exact_crps', 0.0):.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val ACC: {val_acc:.4f} | "
            f"Patience: {patience_counter}/{args.patience}{improved_flag}"
        )

        # Learning Rate Reduction on Plateau
        if lr_patience_counter >= 3:
            curr_lr = max(curr_lr * 0.5, 1e-6)
            optimizer.learning_rate.assign(curr_lr)
            lr_patience_counter = 0
            logger.info(f"--> Learning rate reduced to {curr_lr:.2e}")

        # Early stopping check
        if patience_counter >= args.patience:
            logger.info(f"--> Early stopping triggered at epoch {epoch} (no CRPS improvement for {args.patience} epochs).")
            break

    total_training_time = time.time() - start_time
    logger.info(f"Lead {lead} (Seed {seed}) training complete in {total_training_time:.1f}s. Best Val CRPS: {best_val_crps:.4f}")

    # Restore best checkpoint weights before generating recursive predictions
    restore_a0_checkpoint(model, best_weights_path)

    # Re-evaluate best model
    final_metrics = evaluate_lead_metrics(
        model=model,
        case_paths=val_paths,
        lead=lead,
        y_hat_prev=y_hat_val_prev,
        eval_mask=eval_mask,
        batch_size=args.batch_size,
    )

    # Save training history JSON
    history_file = seed_lead_dir / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "lead": lead,
                "seed": seed,
                "best_val_metrics": final_metrics,
                "history": history,
                "training_time_seconds": total_training_time,
            },
            f,
            indent=2,
        )

    return model, history, final_metrics


def train_seed(
    seed: int,
    args: argparse.Namespace,
    eval_mask: np.ndarray,
    train_paths: List[Path],
    val_paths: List[Path],
) -> Dict[int, Dict[str, float]]:
    """
    Executes sequential 4-lead recursive training cascade for a given random seed:
      Lead 1 -> y_hat_W1 -> Lead 2 -> y_hat_W2 -> Lead 3 -> y_hat_W3 -> Lead 4
    """
    logger.info("*" * 85)
    logger.info(f">>> COMMENCING RECURSIVE PRODUCTION TRAINING FOR SEED {seed} <<<")
    logger.info("*" * 85)

    y_hat_train_cascade: Dict[str, Dict[int, np.ndarray]] = {}
    y_hat_val_cascade: Dict[str, Dict[int, np.ndarray]] = {}
    seed_results: Dict[int, Dict[str, float]] = {}

    for lead in args.leads:
        model, history, metrics = train_single_lead(
            lead=lead,
            seed=seed,
            train_paths=train_paths,
            val_paths=val_paths,
            y_hat_train_prev=y_hat_train_cascade,
            y_hat_val_prev=y_hat_val_cascade,
            eval_mask=eval_mask,
            args=args,
        )
        seed_results[lead] = metrics

        # Generate recursive predictions for downstream leads
        if lead < 4:
            logger.info(f"Generating recursive Lead {lead} predictions for downstream cascade...")
            train_preds = generate_lead_predictions(model, train_paths, lead, y_hat_train_cascade, eval_mask)
            val_preds = generate_lead_predictions(model, val_paths, lead, y_hat_val_cascade, eval_mask)

            # Accumulate into cascade dictionaries
            for cid, p in train_preds.items():
                if cid not in y_hat_train_cascade:
                    y_hat_train_cascade[cid] = {}
                y_hat_train_cascade[cid][lead] = p

            for cid, p in val_preds.items():
                if cid not in y_hat_val_cascade:
                    y_hat_val_cascade[cid] = {}
                y_hat_val_cascade[cid][lead] = p

    # GCS lake synchronization for seed artifacts
    if args.gcs_sync:
        seed_dir = Path(args.output_dir) / f"seed_{seed}"
        gcs_dest = f"{GCS_BUCKET}/checkpoints/A0/seed_{seed}/"
        logger.info(f"Syncing seed {seed} checkpoints to {gcs_dest}...")
        try:
            subprocess.run(["gcloud", "storage", "cp", "-r", str(seed_dir), gcs_dest], check=True)
            logger.info("[PASS] GCS sync completed.")
        except Exception as e:
            logger.warning(f"GCS sync warning: {e}")

    return seed_results


def main():
    parser = argparse.ArgumentParser(
        description="Step 21K.3: Model A0 Three-Seed Production Training across 4 Leads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds to train.")
    parser.add_argument("--leads", type=int, nargs="+", default=DEFAULT_LEADS, help="Forecast leads to train.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size (multiple of 11).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum epochs per lead.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Base learning rate.")
    parser.add_argument("--mode", type=str, choices=["full", "dry_run"], default="full", help="Execution mode.")
    parser.add_argument("--cases-dir", type=str, default="processed/cases/production", help="Directory containing case NPZs.")
    parser.add_argument("--output-dir", type=str, default="checkpoints/A0", help="Root checkpoint output directory.")
    parser.add_argument("--gcs-sync", action="store_true", help="Sync checkpoints and logs to GCS lake.")
    parser.add_argument("--limit-cases", type=int, default=0, help="Optional limit on cases (for rapid verification).")
    args = parser.parse_args()

    # Verify batch size
    if args.batch_size % ENSEMBLE_MEMBERS != 0:
        logger.error(f"Batch size {args.batch_size} is not a multiple of 11!")
        sys.exit(1)

    logger.info("=" * 85)
    logger.info(">>> STEP 21K.3: THREE-SEED MODEL A0 PRODUCTION TRAINING ENGINE <<<")
    logger.info(f"Seeds: {args.seeds} | Leads: {args.leads} | Mode: {args.mode} | Batch Size: {args.batch_size}")
    logger.info("=" * 85)

    # 1. Environment & GPU configuration
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"Detected {len(gpus)} GPU(s): {[g.name for g in gpus]}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    else:
        logger.warning("No GPU detected! Running on CPU. (Warning: Production training will be slow without GPU)")

    # 2. Evaluation mask
    eval_mask = load_eval_mask(REPO_ROOT)
    logger.info(f"Authoritative 126-cell evaluation mask loaded.")

    # 3. Determine case paths
    if args.mode == "dry_run":
        logger.info("Operating in DRY_RUN mode: using 8 pilot cases.")
        pilot_dir = REPO_ROOT / "processed" / "cases" / "pilot"
        train_paths = sorted(pilot_dir.glob("CASE_*.npz"))[:6]
        val_paths = sorted(pilot_dir.glob("CASE_*.npz"))[6:]
    else:
        train_manifest = REPO_ROOT / "manifests" / "splits" / "train_cases.csv"
        val_manifest = REPO_ROOT / "manifests" / "splits" / "val_cases.csv"
        cases_dir = REPO_ROOT / args.cases_dir

        limit = args.limit_cases if args.limit_cases > 0 else None
        all_train_paths = load_manifest_cases(train_manifest, cases_dir, limit=limit)
        all_val_paths = load_manifest_cases(val_manifest, cases_dir, limit=limit)

        existing_train = [p for p in all_train_paths if p.exists()]
        existing_val = [p for p in all_val_paths if p.exists()]

        # Load availability audit ledger for fail-closed cohort accounting
        audit_path = REPO_ROOT / "manifests" / "splits" / "cases_availability_audit.csv"
        df_audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame(columns=["split", "category"])
        train_audit = df_audit[df_audit["split"] == "TRAIN"]
        val_audit = df_audit[df_audit["split"] == "VAL"]

        expected_train_available = len(train_audit[train_audit["category"] == "USABLE_AVAILABLE"]) if len(train_audit) > 0 else 677
        expected_train_mars = len(train_audit[train_audit["category"] == "PROVIDER_UNAVAILABLE_MARS_NO_DATA"]) if len(train_audit) > 0 else 58
        expected_val_available = len(val_audit[val_audit["category"] == "USABLE_AVAILABLE"]) if len(val_audit) > 0 else 194
        expected_val_mars = len(val_audit[val_audit["category"] == "PROVIDER_UNAVAILABLE_MARS_NO_DATA"]) if len(val_audit) > 0 else 16

        if limit is None:
            logger.info("=" * 80)
            logger.info("VERIFYING FULL PRODUCTION COHORT ACCOUNTING (735 Train + 210 Val = 945 Cases)")
            logger.info("=" * 80)
            logger.info(f"Train Cohort: {len(existing_train)} on disk + {expected_train_mars} MARS provider exceptions = {len(existing_train) + expected_train_mars} / 735 scheduled")
            logger.info(f"Val Cohort:   {len(existing_val)} on disk + {expected_val_mars} MARS provider exceptions = {len(existing_val) + expected_val_mars} / 210 scheduled")

            assert len(existing_train) == expected_train_available, (
                f"CRITICAL COHORT BREACH: Expected {expected_train_available} valid training cases on disk, found {len(existing_train)}!"
            )
            assert len(existing_val) == expected_val_available, (
                f"CRITICAL COHORT BREACH: Expected {expected_val_available} valid validation cases on disk, found {len(existing_val)}!"
            )
            assert len(existing_train) + expected_train_mars == 735, (
                f"Train cohort conservation violated: {len(existing_train)} + {expected_train_mars} != 735"
            )
            assert len(existing_val) + expected_val_mars == 210, (
                f"Validation cohort conservation violated: {len(existing_val)} + {expected_val_mars} != 210"
            )
            assert len(existing_train) + expected_train_mars + len(existing_val) + expected_val_mars == 945, (
                f"Total cohort conservation violated: 735 train + 210 val != 945!"
            )
            logger.info("[PASS] Full production cohort verified: 871 active cases + 74 audited provider exceptions = 945 scheduled cases (0 unexplained gaps).")

            train_paths = existing_train
            val_paths = existing_val
        else:
            logger.info(f"Operating with limit-cases={limit}: {len(existing_train)} train and {len(existing_val)} val cases available on disk.")
            train_paths = existing_train
            val_paths = existing_val

    logger.info(f"Partition setup: Train = {len(train_paths)} cases | Validation = {len(val_paths)} cases.")

    # 4. Multi-Seed Training Loop
    all_seed_results: Dict[int, Dict[int, Dict[str, float]]] = {}
    total_start = time.time()

    for seed in args.seeds:
        seed_metrics = train_seed(
            seed=seed,
            args=args,
            eval_mask=eval_mask,
            train_paths=train_paths,
            val_paths=val_paths,
        )
        all_seed_results[seed] = seed_metrics

    # 5. Consolidated Multi-Seed Report
    total_elapsed = time.time() - total_start
    summary_report: Dict[str, Any] = {
        "step": "21K.3",
        "milestone": "Full Three-Seed Model A0 Production Training",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_elapsed_seconds": total_elapsed,
        "seeds": args.seeds,
        "leads": args.leads,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "per_seed_results": all_seed_results,
        "ensemble_synthesis": {},
    }

    # Compute mean and std across seeds per lead
    for lead in args.leads:
        metrics_keys = [
            "val_crps",
            "val_spatial_crps_proxy",
            "val_exact_crps",
            "val_mae",
            "val_rmse",
            "val_acc",
        ]
        lead_summary = {}
        for k in metrics_keys:
            vals = [all_seed_results[s][lead][k] for s in args.seeds if lead in all_seed_results[s] and k in all_seed_results[s][lead]]
            if vals:
                lead_summary[f"{k}_mean"] = float(np.mean(vals))
                lead_summary[f"{k}_std"] = float(np.std(vals))
        summary_report["ensemble_synthesis"][f"lead_{lead}"] = lead_summary

    # Save summary report
    summary_path = REPO_ROOT / "logs" / "a0_production_3seed_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    logger.info(f"Production 3-Seed summary exported to {summary_path}")

    # GCS sync of final summary
    if args.gcs_sync:
        try:
            subprocess.run(["gcloud", "storage", "cp", str(summary_path), f"{GCS_BUCKET}/logs/"], check=True)
            logger.info("[PASS] Summary synced to GCS lake.")
        except Exception as e:
            logger.warning(f"GCS sync warning: {e}")

    logger.info("=" * 85)
    logger.info(">>> MODEL A0 PRODUCTION TRAINING SUCCESSFULLY CONCLUDED <<<")
    logger.info("=" * 85)


if __name__ == "__main__":
    main()
