"""
src/data/case_builder.py
------------------------
Authoritative multi-lead input-target tensor hierarchy assembler for Mindanao RISE-UNet (Sub-Phase 21F).
Constructs the complete input-target hierarchy for a single forecast issue cycle adhering
strictly to Kyle Lesinger's author code (loadDataAllWeeks.py:L710-721) and the verified EX29 contract:

  - Lead 1 (W1): 11 channels -> 3 antecedent RZSM lags + 5 ERA5 surface obs + 3 ECMWF S2S W1 preds
  - Lead 2 (W2): 12 channels -> 3 antecedent RZSM lags + 5 ERA5 surface obs + 3 ECMWF S2S W2 preds + 1 recursive y_hat_w1
  - Lead 3 (W3): 5 channels  -> 3 antecedent RZSM lags + 2 recursive preds (y_hat_w1, y_hat_w2)
  - Lead 4 (W4): 6 channels  -> 3 antecedent RZSM lags + 3 recursive preds (y_hat_w1, y_hat_w2, y_hat_w3)

Ensemble Dimension:
  Input tensors maintain the M=11 ensemble member structure (Shape: [M=11, H=32, W=48, C_k]).
  Target tensors represent ground truth once per lead (Shape: [1, H=32, W=48, 1]),
  broadcast to [M=11, H=32, W=48, 1] strictly at the CRPS loss layer.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import xarray as xr
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_NORM_CONTRACT_PATH = REPO_ROOT / "contracts" / "A0" / "normalization_parameters.yaml"


def load_frozen_normalization_contract(
    contract_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Loads frozen machine-readable normalization parameters contract."""
    p = Path(contract_path) if contract_path is not None else DEFAULT_NORM_CONTRACT_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Normalization parameters contract not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def scale_channel(
    arr: np.ndarray,
    min_val: float,
    max_val: float,
    clip: bool = True
) -> np.ndarray:
    """Scales array to unit interval [0, 1] using training scalar bounds."""
    denom = max_val - min_val
    if np.isclose(denom, 0.0):
        raise ValueError(f"Degenerate bounds: max ({max_val}) equals min ({min_val})")
    scaled = (arr - min_val) / denom
    if clip:
        scaled = np.clip(scaled, 0.0, 1.0)
    return scaled.astype(np.float32)


def denormalize_predictions(
    y_norm: np.ndarray,
    norm_params: Optional[Dict[str, Any]] = None,
    target_key: str = "seasonal_anomaly"
) -> np.ndarray:
    """De-normalizes model predictions back to physical units using frozen contract."""
    if norm_params is None:
        norm_params = load_frozen_normalization_contract()
    p = norm_params["rzsm_parameters"][target_key]
    return (y_norm * (p["max"] - p["min"]) + p["min"]).astype(np.float32)


@dataclass(frozen=True)
class CaseTensorHierarchy:
    """Encapsulates the complete input-target tensor hierarchy for a single forecast case."""
    issue_date: str         # Primary forecast issuance date (hdate)
    hdate: str              # Historical hindcast issue date
    model_version_date: Optional[str]  # Operational model version date (ECMWF date)
    # Input tensors (M=11, 32, 48, C_k)
    x_w1: np.ndarray        # (11, 32, 48, 11)
    x_w2_base: np.ndarray   # (11, 32, 48, 11) - ready for y_hat_w1 concatenation to form 12 channels
    x_w3_base: np.ndarray   # (11, 32, 48, 3)  - ready for y_hat_w1, y_hat_w2 concatenation to form 5 channels
    x_w4_base: np.ndarray   # (11, 32, 48, 3)  - ready for y_hat_w1, y_hat_w2, y_hat_w3 concatenation to form 6 channels
    # Ground truth target tensors (1, 32, 48, 1)
    y_w1: np.ndarray
    y_w2: np.ndarray
    y_w3: np.ndarray
    y_w4: np.ndarray
    # Metadata
    channel_schedule: Dict[int, int]
    num_members: int = 11
    target_dates: Optional[Dict[int, str]] = None
    is_normalized: bool = False
    normalization_contract_used: Optional[str] = None


def assemble_single_a0_case(
    issue_date: Union[str, pd.Timestamp],
    rzsm_cube_ds: xr.Dataset,
    atmospheric_ds: xr.Dataset,
    s2s_ds: xr.Dataset,
    target_var_name: str = "rzsm_rolling_7d",
    eval_mask: Optional[np.ndarray] = None,
    normalize: bool = False,
    norm_params: Optional[Dict[str, Any]] = None,
) -> CaseTensorHierarchy:
    """
    Assembles real multi-lead input-target tensors for a single forecast issuance date.

    Parameters:
        issue_date: Forecast issuance date t_0 (e.g. '2015-01-15').
        rzsm_cube_ds: Production RZSM dataset with daily resolution covering [t_0 - 14d, t_0 + 28d].
        atmospheric_ds: ERA5 surface atmospheric dataset covering day t_0.
        s2s_ds: Harmonized ECMWF S2S reforecast dataset covering leads 1 and 2 across 11 members.
        target_var_name: Target RZSM variable (default: 'rzsm_rolling_7d' or 'rzsm_anom').
        eval_mask: Optional binary evaluation mask (32, 48) for zero-filling ocean cells.
        normalize: Whether to scale all channels into [0, 1] using frozen training contract.
        norm_params: Optional explicit normalization dictionary; loads from contracts/A0 if None.

    Returns:
        CaseTensorHierarchy with verified shapes and channel alignments.
    """
    t0 = pd.Timestamp(issue_date)
    num_members = len(s2s_ds["member"])
    h, w = len(rzsm_cube_ds["lat"]), len(rzsm_cube_ds["lon"])

    # Auto-detect target variable name variant
    if target_var_name not in rzsm_cube_ds:
        if "rzsm_0_100_rolling_7d" in rzsm_cube_ds:
            target_var_name = "rzsm_0_100_rolling_7d"
        elif "rzsm_rolling_7d" in rzsm_cube_ds:
            target_var_name = "rzsm_rolling_7d"
        elif "rzsm_anom" in rzsm_cube_ds:
            target_var_name = "rzsm_anom"
        elif "rzsm_0_100_anom" in rzsm_cube_ds:
            target_var_name = "rzsm_0_100_anom"

    # -------------------------------------------------------------------------
    # 1. Extract 3 Antecedent RZSM Lag Channels ([-1, -7, -14] days)
    # -------------------------------------------------------------------------
    lag_dates = [t0 - pd.Timedelta(days=d) for d in [1, 7, 14]]
    rzsm_lags = []
    for d in lag_dates:
        d_str = d.strftime("%Y-%m-%d")
        if d_str not in rzsm_cube_ds["time"].dt.strftime("%Y-%m-%d").values:
            raise KeyError(f"Antecedent lag date {d_str} not found in RZSM cube.")
        slice_2d = rzsm_cube_ds[target_var_name].sel(time=d_str).values
        if slice_2d.ndim == 3:
            slice_2d = slice_2d[0]
        rzsm_lags.append(slice_2d)

    # Shape: (3, 32, 48) -> (1, 32, 48, 3) -> broadcast to (M=11, 32, 48, 3)
    rzsm_lags_arr = np.stack(rzsm_lags, axis=-1)[np.newaxis, :, :, :]  # (1, 32, 48, 3)
    rzsm_lags_m11 = np.repeat(rzsm_lags_arr, num_members, axis=0).astype(np.float32)

    # -------------------------------------------------------------------------
    # 2. Extract 5 ERA5 Surface Atmospheric Channels at t_0
    #    (pwat, spfh, tmax, diff_temp, hgt_pres)
    # -------------------------------------------------------------------------
    t0_str = t0.strftime("%Y-%m-%d")
    atm_vars = ["pwat", "spfh", "tmax", "diff_temp", "hgt_pres"]
    atm_slices = []
    for var in atm_vars:
        if var not in atmospheric_ds:
            raise KeyError(f"Atmospheric variable {var} not found in atmospheric dataset.")
        # Select date
        var_slice = atmospheric_ds[var].sel(time=t0_str).values
        if var_slice.ndim == 3:
            var_slice = var_slice[0]
        atm_slices.append(var_slice)

    # Shape: (5, 32, 48) -> (1, 32, 48, 5) -> broadcast to (M=11, 32, 48, 5)
    atm_arr = np.stack(atm_slices, axis=-1)[np.newaxis, :, :, :]
    atm_m11 = np.repeat(atm_arr, num_members, axis=0).astype(np.float32)

    # -------------------------------------------------------------------------
    # 3. Extract 3 ECMWF S2S Dynamic Triplet Channels (t2m, d2m, tcw)
    # -------------------------------------------------------------------------
    s2s_vars = ["t2m", "d2m", "tcw"]
    # S2S Lead 1 (Week 1): shape (11, 32, 48, 3)
    s2s_w1_slices = [s2s_ds[v].sel(lead=1).values for v in s2s_vars]  # each is (11, 32, 48)
    s2s_w1_m11 = np.stack(s2s_w1_slices, axis=-1).astype(np.float32)

    # S2S Lead 2 (Week 2): shape (11, 32, 48, 3)
    s2s_w2_slices = [s2s_ds[v].sel(lead=2).values for v in s2s_vars]
    s2s_w2_m11 = np.stack(s2s_w2_slices, axis=-1).astype(np.float32)

    # -------------------------------------------------------------------------
    # 4. Extract Ground Truth Targets for Leads W1 to W4
    #    Verified parent EX29 target endpoint contract: L = (lead * 7) - 1
    #    Target W1: day t_0 + 6d  (captures trailing 7d mean over days t_0 .. t_0 + 6d)
    #    Target W2: day t_0 + 13d (captures trailing 7d mean over days t_0 + 7d .. t_0 + 13d)
    #    Target W3: day t_0 + 20d (captures trailing 7d mean over days t_0 + 14d .. t_0 + 20d)
    #    Target W4: day t_0 + 27d (captures trailing 7d mean over days t_0 + 21d .. t_0 + 27d)
    # -------------------------------------------------------------------------
    targets = []
    target_dates_dict = {}
    for lead_idx, lead_days in enumerate([6, 13, 20, 27], start=1):
        target_date = (t0 + pd.Timedelta(days=lead_days)).strftime("%Y-%m-%d")
        if target_date not in rzsm_cube_ds["time"].dt.strftime("%Y-%m-%d").values:
            raise KeyError(f"Target date {target_date} (+{lead_days}d) not found in RZSM cube.")
        tgt_2d = rzsm_cube_ds[target_var_name].sel(time=target_date).values
        if tgt_2d.ndim == 3:
            tgt_2d = tgt_2d[0]
        # Shape: (1, 32, 48, 1)
        targets.append(tgt_2d[np.newaxis, :, :, np.newaxis].astype(np.float32))
        target_dates_dict[lead_idx] = target_date

    y_w1, y_w2, y_w3, y_w4 = targets

    # -------------------------------------------------------------------------
    # 5. Normalization Scaling (Active Domain Min-Max into [0, 1])
    # -------------------------------------------------------------------------
    contract_source = None
    if normalize:
        if norm_params is None:
            norm_params = load_frozen_normalization_contract()
            contract_source = str(DEFAULT_NORM_CONTRACT_PATH)
        else:
            contract_source = "provided_parameters"

        # RZSM bounds selection
        if "anom" in target_var_name:
            rzsm_key = "seasonal_anomaly"
        elif "rolling_7d" in target_var_name:
            rzsm_key = "volumetric_rolling_7d"
        else:
            rzsm_key = "volumetric_raw" if "volumetric_raw" in norm_params.get("rzsm_parameters", {}) else "seasonal_anomaly"
        rzsm_p = norm_params["rzsm_parameters"][rzsm_key]
        rzsm_min, rzsm_max = float(rzsm_p["min"]), float(rzsm_p["max"])

        # Scale antecedent RZSM lags (all 3 lag channels)
        rzsm_lags_m11 = scale_channel(rzsm_lags_m11, rzsm_min, rzsm_max)

        # Scale atmospheric channels (5 channels: pwat, spfh, tmax, diff_temp, hgt_pres)
        for i, var in enumerate(atm_vars):
            p = norm_params["atmospheric_parameters"][var]
            atm_m11[..., i] = scale_channel(atm_m11[..., i], float(p["min"]), float(p["max"]))

        # Scale S2S Lead 1 channels (3 channels: t2m, d2m, tcw)
        for i, var in enumerate(s2s_vars):
            p = norm_params["s2s_parameters"]["lead_1"][var]
            s2s_w1_m11[..., i] = scale_channel(s2s_w1_m11[..., i], float(p["min"]), float(p["max"]))

        # Scale S2S Lead 2 channels (3 channels: t2m, d2m, tcw)
        for i, var in enumerate(s2s_vars):
            p = norm_params["s2s_parameters"]["lead_2"][var]
            s2s_w2_m11[..., i] = scale_channel(s2s_w2_m11[..., i], float(p["min"]), float(p["max"]))

        # Scale ground truth targets
        y_w1 = scale_channel(y_w1, rzsm_min, rzsm_max)
        y_w2 = scale_channel(y_w2, rzsm_min, rzsm_max)
        y_w3 = scale_channel(y_w3, rzsm_min, rzsm_max)
        y_w4 = scale_channel(y_w4, rzsm_min, rzsm_max)

    # -------------------------------------------------------------------------
    # 6. Construct Lead Input Tensors
    # -------------------------------------------------------------------------
    # Lead 1: 3 RZSM lags + 5 ERA5 obs + 3 S2S W1 preds = 11 channels
    x_w1 = np.concatenate([rzsm_lags_m11, atm_m11, s2s_w1_m11], axis=-1)  # (11, 32, 48, 11)

    # Lead 2 base: 3 RZSM lags + 5 ERA5 obs + 3 S2S W2 preds = 11 channels
    # (recursive y_hat_w1 is appended during cascade execution to form 12 channels)
    x_w2_base = np.concatenate([rzsm_lags_m11, atm_m11, s2s_w2_m11], axis=-1)  # (11, 32, 48, 11)

    # Lead 3 base: 3 RZSM lags (y_hat_w1, y_hat_w2 appended to form 5 channels)
    x_w3_base = rzsm_lags_m11.copy()  # (11, 32, 48, 3)

    # Lead 4 base: 3 RZSM lags (y_hat_w1, y_hat_w2, y_hat_w3 appended to form 6 channels)
    x_w4_base = rzsm_lags_m11.copy()  # (11, 32, 48, 3)

    # Apply evaluation mask if provided (zero-fill ocean/buffer)
    if eval_mask is not None:
        ocean = eval_mask == 0
        x_w1[:, ocean, :] = 0.0
        x_w2_base[:, ocean, :] = 0.0
        x_w3_base[:, ocean, :] = 0.0
        x_w4_base[:, ocean, :] = 0.0
        y_w1[:, ocean, :] = 0.0
        y_w2[:, ocean, :] = 0.0
        y_w3[:, ocean, :] = 0.0
        y_w4[:, ocean, :] = 0.0

    hdate = str(s2s_ds.attrs.get("hdate", t0_str))
    model_version_date = s2s_ds.attrs.get("model_version_date", None)
    if model_version_date == "unknown":
        model_version_date = None

    return CaseTensorHierarchy(
        issue_date=t0_str,
        hdate=hdate,
        model_version_date=model_version_date,
        x_w1=x_w1,
        x_w2_base=x_w2_base,
        x_w3_base=x_w3_base,
        x_w4_base=x_w4_base,
        y_w1=y_w1,
        y_w2=y_w2,
        y_w3=y_w3,
        y_w4=y_w4,
        channel_schedule={1: 11, 2: 12, 3: 5, 4: 6},
        num_members=num_members,
        target_dates=target_dates_dict,
        is_normalized=normalize,
        normalization_contract_used=contract_source,
    )


def simulate_recursive_cascade_step(
    x_base: np.ndarray,
    prior_predictions: list,
) -> np.ndarray:
    """
    Concatenates prior-lead recursive predictions into the downstream input tensor.

    Normalization Invariant:
    ------------------------
    Prior predictions (y_hat_w1, y_hat_w2, ...) generated by Model A0 are
    already normalized model outputs (in [0, 1] target space). They must
    NEVER be re-normalized using training predictor min/max transformations.
    This function strictly preserves this invariant by concatenating the model
    prediction tensors directly into the downstream feature channels without
    any secondary scaling or transformation.

    Parameters:
        x_base: Base feature tensor (M, H, W, C_base)
        prior_predictions: List of prior predictions [y_hat_w1, y_hat_w2, ...] each of shape (M, H, W, 1)

    Returns:
        Complete input tensor with recursive channels appended (M, H, W, C_base + len(prior_predictions)).
    """
    for idx, p in enumerate(prior_predictions, 1):
        if p.shape[:3] != x_base.shape[:3]:
            raise ValueError(
                f"Prior prediction {idx} spatial/ensemble shape {p.shape[:3]} does not match "
                f"base tensor {x_base.shape[:3]}."
            )
        if p.shape[-1] != 1:
            raise ValueError(f"Prior prediction {idx} must have 1 channel, got {p.shape[-1]}.")
        if not np.all(np.isfinite(p)):
            raise ValueError(f"Prior prediction {idx} contains NaN or Inf values.")

    return np.concatenate([x_base] + prior_predictions, axis=-1).astype(np.float32)


def verify_recursive_channel_semantics(
    x_full: np.ndarray,
    lead: int,
    prior_predictions: list,
) -> bool:
    """
    Verifies recursive channel semantics and exact ordering for Leads 2, 3, and 4.

    Contract:
    ---------
    - Lead 1: No recursive channels (Cin=11).
    - Lead 2: Cin=12. Channel index 11 must match y_hat_w1 exactly.
    - Lead 3: Cin=5. Channels 0..2 are base RZSM lags. Channel index 3 must match y_hat_w1;
      channel index 4 must match y_hat_w2.
    - Lead 4: Cin=6. Channels 0..2 are base RZSM lags. Channel index 3 must match y_hat_w1;
      channel index 4 must match y_hat_w2; channel index 5 must match y_hat_w3.

    Raises ValueError if channel counts, ordering, or values fail to match.
    Returns True upon strict verification.
    """
    if lead == 1:
        if len(prior_predictions) != 0:
            raise ValueError("Lead 1 expects 0 prior predictions.")
        return True

    expected_channels = {2: 12, 3: 5, 4: 6}
    if lead not in expected_channels:
        raise ValueError(f"Invalid lead {lead}. Must be 1, 2, 3, or 4.")

    if x_full.shape[-1] != expected_channels[lead]:
        raise ValueError(
            f"Lead {lead} expected {expected_channels[lead]} channels, got {x_full.shape[-1]}"
        )

    num_priors = len(prior_predictions)
    expected_priors = {2: 1, 3: 2, 4: 3}
    if num_priors != expected_priors[lead]:
        raise ValueError(
            f"Lead {lead} expects exactly {expected_priors[lead]} prior predictions, got {num_priors}."
        )

    start_idx = x_full.shape[-1] - num_priors
    for i, pred in enumerate(prior_predictions):
        ch_idx = start_idx + i
        extracted = x_full[..., ch_idx : ch_idx + 1]
        max_delta = float(np.max(np.abs(extracted - pred)))
        if max_delta > 1e-7:
            raise ValueError(
                f"Lead {lead} recursive channel at index {ch_idx} diverges from prior prediction {i+1}: "
                f"max delta = {max_delta}"
            )

    return True
