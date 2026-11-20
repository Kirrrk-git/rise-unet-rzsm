"""
src/models/scpw_mrgr.py
-----------------------
Thesis v3.0 architecture specifications and TensorFlow/NumPy implementations for:
  1. Authentic Baseline `modelRzsmReluExtraConv` ([32, 64, 128, 256], Conv2D)
  2. Stage 1: Season-Conditioned Predictor Weighting (`SCPW`, plus `[0,0]` control)
  3. Stage 2: Monotonic Reliability-Gated Recursive Refinement (`MRGR`, 6 directed
     handoffs with `RiskNet` + frozen reliability gate + zero-init `CorrectionNet`)
  4. Exact analytical parameter-count validators and MRGR mathematical helpers that
     run cleanly on CPU without requiring TensorFlow.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

GRID_HEIGHT: int = 32
GRID_WIDTH: int = 48

V3_LEAD_CHANNELS: Dict[int, int] = {1: 11, 2: 12, 3: 5, 4: 6}
V3_NONREC_CHANNELS: Dict[int, int] = {1: 11, 2: 11, 3: 3, 4: 3}

EXPECTED_EXTRACONV_BACKBONE_PARAMS: Dict[int, int] = {
    1: 10_240_723,
    2: 10_243_539,
    3: 10_223_827,
    4: 10_226_643,
}

EXPECTED_SCPW_PARAMS: Dict[int, int] = {
    1: 15_024,
    2: 15_024,
    3: 5_808,
    4: 5_808,
}

MRGR_DIRECTED_HANDOFFS: Tuple[Tuple[int, int], ...] = (
    (1, 2),
    (1, 3),
    (2, 3),
    (1, 4),
    (2, 4),
    (3, 4),
)

EXPECTED_MRGR_BRANCH_PARAMS: Dict[Tuple[int, int], int] = {
    (1, 2): 2_625,  # RiskNet=2,625; CorrectionNet=2,625 -> pair=5,250
    (1, 3): 2_049,
    (2, 3): 2_049,
    (1, 4): 2_049,
    (2, 4): 2_049,
    (3, 4): 2_049,
}


def compute_week_midpoint_date(issue_date: Union[str, pd.Timestamp], lead: int) -> pd.Timestamp:
    """Computes Thesis Eq. (midpoint): d_h(t) = t + 7*(h - 1) + 3 days."""
    if lead not in (1, 2, 3, 4):
        raise ValueError(f"lead must be in {{1, 2, 3, 4}}, got {lead}")
    return pd.Timestamp(issue_date) + pd.Timedelta(days=7 * (lead - 1) + 3)


def compute_scpw_calendar_descriptor(
    issue_dates: Union[str, pd.Timestamp, Iterable[Union[str, pd.Timestamp]]],
    lead: int,
    constant_control: bool = False,
) -> np.ndarray:
    """
    Computes SCPW periodic midpoint-date descriptor:
        s_h(t) = [sin(2*pi*(n - 1)/Y), cos(2*pi*(n - 1)/Y)]
    where n is 1-based day-of-year and Y in {365, 366} is leap-aware year length.
    Returns shape (N, 2) float32; returns zeros if `constant_control=True`.
    """
    if isinstance(issue_dates, (str, pd.Timestamp)):
        dates = [issue_dates]
    else:
        dates = list(issue_dates)

    if constant_control:
        return np.zeros((len(dates), 2), dtype=np.float32)

    out = np.empty((len(dates), 2), dtype=np.float32)
    for idx, dt in enumerate(dates):
        mid = compute_week_midpoint_date(dt, lead)
        n_doy = int(mid.dayofyear)
        year_len = 366 if mid.is_leap_year else 365
        angle = 2.0 * np.pi * (n_doy - 1) / float(year_len)
        out[idx, 0] = np.sin(angle)
        out[idx, 1] = np.cos(angle)
    return out


def compute_v3_parameter_budget() -> Dict[str, Any]:
    """Computes exact analytical parameter budgets for ExtraConv, SCPW, and MRGR."""
    scpw_calc: Dict[int, int] = {}
    for lead, c_nonrec in V3_NONREC_CHANNELS.items():
        conv3x3 = 3 * 3 * c_nonrec * 128 + 128
        mlp = (2 * 16 + 16) + (16 * 128 + 128)
        scpw_calc[lead] = conv3x3 + mlp

    mrgr_calc: Dict[Tuple[int, int], int] = {}
    for j, h in MRGR_DIRECTED_HANDOFFS:
        c_x = V3_NONREC_CHANNELS[h]
        # RiskNet / CorrectionNet branch:
        # 3x3_8(x_h) + 3x3_8(r_j) + 3x3_8(concat_24ch) + 1x1_1(8ch)
        branch = (3 * 3 * c_x * 8 + 8) + (3 * 3 * 1 * 8 + 8) + (3 * 3 * 24 * 8 + 8) + (1 * 1 * 8 * 1 + 1)
        mrgr_calc[(j, h)] = branch

    scpw_total = sum(scpw_calc.values())
    mrgr_total = sum(2 * v for v in mrgr_calc.values())
    backbone_total = sum(EXPECTED_EXTRACONV_BACKBONE_PARAMS.values())
    return {
        "backbone_extraconv_per_lead": dict(EXPECTED_EXTRACONV_BACKBONE_PARAMS),
        "backbone_total": backbone_total,
        "scpw_per_lead": scpw_calc,
        "scpw_total": scpw_total,
        "mrgr_per_branch": mrgr_calc,
        "mrgr_per_pair_total": {f"{j}_to_{h}": 2 * v for (j, h), v in mrgr_calc.items()},
        "mrgr_total": mrgr_total,
        "combined_refinement_overhead": scpw_total + mrgr_total,
    }


def compute_mrgr_scale_and_target(
    r_oof_phys: np.ndarray,
    y_true_phys: np.ndarray,
    eval_mask: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Computes Thesis Eqs. (2)-(3) on active cells over OOF training bank:
        c_{j->h} = max(P_95(|r_{j,OOF} - y_j|), 1e-6)
        q_{j->h} = |r_{j,OOF} - y_j| / (c_{j->h} + 1e-6)
    """
    m_bool = np.asarray(eval_mask) == 1
    abs_err = np.abs(r_oof_phys.astype(np.float64) - y_true_phys.astype(np.float64))
    active_abs = abs_err[..., m_bool]
    p95 = float(np.percentile(active_abs, 95.0))
    c_scale = max(p95, 1e-6)
    q_target = (abs_err / (c_scale + 1e-6)).astype(np.float32)
    return float(c_scale), q_target


def compute_huber_risk_loss(
    q_pred: np.ndarray,
    q_target: np.ndarray,
    eval_mask: np.ndarray,
    delta: float = 1.0,
) -> float:
    """Computes active-cell mean Huber loss (delta=1.0 in normalized error units)."""
    m_bool = np.asarray(eval_mask) == 1
    diff = (q_pred.astype(np.float64) - q_target.astype(np.float64))[..., m_bool]
    abs_d = np.abs(diff)
    quad = abs_d <= delta
    loss = np.where(quad, 0.5 * diff ** 2, delta * (abs_d - 0.5 * delta))
    return float(np.mean(loss))


def apply_mrgr_gate_and_refinement(
    r_j_phys: np.ndarray,
    d_theta: np.ndarray,
    c_scale: float,
    q_hat: Optional[np.ndarray] = None,
    ungated_ablation: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Applies Thesis Eqs. (5)-(7) WITHOUT clipping r^* to [0, 1]:
        G = exp(-q_hat) in (0, 1]
        delta = c * tanh(d_theta)
        r^* = r_j + (1 - G) * delta       [standard MRGR]
        r^* = r_j + delta                 [ungated_ablation=True]
    Returns (r_refined, diagnostic_counts) logging any out-of-[0, 1] excursions.
    """
    delta = float(c_scale) * np.tanh(d_theta.astype(np.float64))
    if ungated_ablation:
        gate = np.zeros_like(delta)
        weight = np.ones_like(delta)
    else:
        if q_hat is None:
            raise ValueError("q_hat is required unless ungated_ablation=True.")
        gate = np.exp(-np.maximum(q_hat.astype(np.float64), 0.0))
        weight = 1.0 - gate

    r_star = (r_j_phys.astype(np.float64) + weight * delta).astype(np.float32)
    excursion_diag = {
        "below_zero_count": int(np.sum(r_star < 0.0)),
        "above_one_count": int(np.sum(r_star > 1.0)),
        "min_val": float(np.min(r_star)),
        "max_val": float(np.max(r_star)),
        "gate_mean": float(np.mean(gate)),
    }
    return r_star, excursion_diag


def build_extraconv_unet_v3(
    lead: int = 1,
    height: int = GRID_HEIGHT,
    width: int = GRID_WIDTH,
    use_scpw: bool = False,
    using_deep_supervision: bool = True,
    output_channels: int = 1,
    name: Optional[str] = None,
) -> Any:
    """
    Constructs authentic parent `modelRzsmReluExtraConv` ([32, 64, 128, 256], Conv2D)
    with optional Stage 1 SCPW branch inserted strictly between initial 128-ch inception
    output u_h and following 64-ch conv_batchnorm_relu_block_SE_and_residual_connections.
    """
    if lead not in V3_LEAD_CHANNELS:
        raise ValueError(f"lead must be in {list(V3_LEAD_CHANNELS)}, got {lead}")
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(f"Grid dimensions ({height}, {width}) must be divisible by 16.")

    import tensorflow as tf
    from keras.layers import Conv2D, Dense, Input, MaxPool2D, Conv2DTranspose, Reshape, concatenate
    from keras.models import Model
    from parent_study_ex29.function import modelRzsmReluExtraConv as ExtraConv

    tf.keras.backend.set_image_data_format("channels_last")
    bn_axis = -1
    nb_filter = [32, 64, 128, 256]
    dr_init, dr_later = 0.1, 0.25
    kn = None
    in_ch = V3_LEAD_CHANNELS[lead]
    nonrec_ch = V3_NONREC_CHANNELS[lead]

    x_in = Input(shape=(height, width, in_ch), name=f"input_lead_{lead}")
    model_inputs: List[Any] = [x_in]

    u_h = ExtraConv.inception_block(
        x_in, nb_filter[0], nb_filter[0], dropout_rate=dr_init, depth_multiplier=True, kernel_norm=kn
    )

    if use_scpw:
        s_in = Input(shape=(2,), name=f"scpw_calendar_lead_{lead}")
        model_inputs.append(s_in)
        x_nonrec = x_in[..., :nonrec_ch]
        z_h = Conv2D(128, (3, 3), padding="same", activation="relu", name=f"scpw_conv_lead_{lead}")(x_nonrec)
        mlp_h = Dense(16, activation="relu", name=f"scpw_mlp_hidden_lead_{lead}")(s_in)
        mlp_out = Dense(
            128,
            activation="linear",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name=f"scpw_mlp_out_lead_{lead}",
        )(mlp_h)
        w_h = 1.0 + 0.25 * tf.tanh(mlp_out)
        w_h = Reshape((1, 1, 128), name=f"scpw_broadcast_lead_{lead}")(w_h)
        u_h = u_h + (w_h - 1.0) * z_h

    conv1_1 = ExtraConv.conv_batchnorm_relu_block_SE_and_residual_connections(
        u_h, nb_filter=nb_filter[1], dropout_rate=dr_later, kernel_norm=kn
    )
    pool1 = MaxPool2D((2, 2), strides=(2, 2))(conv1_1)

    conv2_1 = ExtraConv.inception_block(pool1, nb_filter[1], nb_filter[1], dropout_rate=dr_later, depth_multiplier=True, kernel_norm=kn)
    conv2_1 = ExtraConv.conv_batchnorm_relu_block_SE_and_residual_connections(conv2_1, nb_filter=nb_filter[2], dropout_rate=dr_later, kernel_norm=kn)
    pool2 = MaxPool2D((2, 2), strides=(2, 2))(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), padding="same", kernel_initializer=ExtraConv.kernel_initializer)(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], axis=bn_axis)
    conv1_2 = ExtraConv.inception_block(conv1_2, nb_filter[0], nb_filter[0], dropout_rate=dr_later, depth_multiplier=True, kernel_norm=kn)
    conv1_2 = ExtraConv.conv_batchnorm_relu_block_SE_and_residual_connections(conv1_2, nb_filter=nb_filter[1], dropout_rate=dr_later, kernel_norm=kn)

    conv3_1 = ExtraConv.conv_batchnorm_relu_block_SE_and_residual_connections(pool2, nb_filter=nb_filter[2], dropout_rate=dr_later, kernel_norm=kn)
    pool3 = MaxPool2D((2, 2), strides=(2, 2))(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), padding="same", kernel_initializer=ExtraConv.kernel_initializer)(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], axis=bn_axis)
    conv2_2 = ExtraConv.inception_block(conv2_2, nb_filter[1], nb_filter[1], dropout_rate=dr_later, depth_multiplier=True, kernel_norm=kn)
    conv2_2 = ExtraConv.conv_batchnorm_relu_block_SE_and_residual_connections(conv2_2, nb_filter=nb_filter[2], dropout_rate=dr_later, kernel_norm=kn)

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), padding="same", kernel_initializer=ExtraConv.kernel_initializer)(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], axis=bn_axis)
    conv1_3 = ExtraConv.inception_block(conv1_3, nb_filter[1], nb_filter[1], dropout_rate=dr_later, depth_multiplier=True, kernel_norm=kn)
    conv1_3 = ExtraConv.conv_batchnorm_relu_block_SE_and_residual_connections(conv1_3, nb_filter=nb_filter[2], dropout_rate=dr_later, kernel_norm=kn)

    conv4_1 = ExtraConv.conv_batchnorm_relu_block_SE_and_residual_connections(pool3, nb_filter=nb_filter[3], dropout_rate=dr_later, kernel_norm=kn)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), padding="same", kernel_initializer=ExtraConv.kernel_initializer)(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], axis=bn_axis)
    conv3_2 = ExtraConv.conv_batchnorm_relu_block_SE_and_residual_connections(conv3_2, nb_filter=nb_filter[2], dropout_rate=dr_later, kernel_norm=kn)

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), padding="same", kernel_initializer=ExtraConv.kernel_initializer)(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], axis=bn_axis)
    conv2_3 = ExtraConv.inception_block(conv2_3, nb_filter[1], nb_filter[1], dropout_rate=dr_later, depth_multiplier=False, kernel_norm=kn)
    conv2_3 = ExtraConv.conv_batchnorm_relu_block_SE_and_residual_connections(conv2_3, nb_filter=nb_filter[2], dropout_rate=dr_later, kernel_norm=kn)

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), padding="same", kernel_initializer=ExtraConv.kernel_initializer)(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], axis=bn_axis)
    conv1_4 = ExtraConv.inception_block(conv1_4, nb_filter[0], nb_filter[0], dropout_rate=dr_later, depth_multiplier=True, kernel_norm=kn)
    conv1_4 = ExtraConv.conv_batchnorm_relu_block_SE_and_residual_connections(conv1_4, nb_filter=nb_filter[1], dropout_rate=dr_later, kernel_norm=kn)

    out1 = Conv2D(output_channels, (1, 1), activation="relu", name="RZSM_output_1", padding="same", dtype=tf.float32)(conv1_2)
    out2 = Conv2D(output_channels, (1, 1), activation="relu", name="RZSM_output_2", padding="same", dtype=tf.float32)(conv1_3)
    out3 = Conv2D(output_channels, (1, 1), activation="relu", name="RZSM_output_3", padding="same", dtype=tf.float32)(conv1_4)

    outputs = [out1, out2, out3] if using_deep_supervision else out3
    m_name = name or f"UNET_RZSM_ExtraConv_W{lead}{'_SCPW' if use_scpw else ''}"
    return Model(inputs=model_inputs if len(model_inputs) > 1 else model_inputs[0], outputs=outputs, name=m_name)


def _build_mrgr_branch_tf(
    source_lead: int,
    consumer_lead: int,
    role: str,
    height: int = GRID_HEIGHT,
    width: int = GRID_WIDTH,
) -> Any:
    """Shared Keras builder for RiskNet (softplus) and CorrectionNet (zero-init linear)."""
    if (source_lead, consumer_lead) not in MRGR_DIRECTED_HANDOFFS:
        raise ValueError(f"Unsupported handoff ({source_lead} -> {consumer_lead}).")

    import tensorflow as tf
    from keras.layers import Conv2D, Input, concatenate
    from keras.models import Model

    c_x = V3_NONREC_CHANNELS[consumer_lead]
    x_in = Input(shape=(height, width, c_x), name=f"{role}_x_{source_lead}_to_{consumer_lead}")
    r_in = Input(shape=(height, width, 1), name=f"{role}_r_{source_lead}_to_{consumer_lead}")

    e_x = Conv2D(8, (3, 3), padding="same", activation="relu")(x_in)
    e_r = Conv2D(8, (3, 3), padding="same", activation="relu")(r_in)
    diff = tf.abs(e_x - e_r)
    fused = Conv2D(8, (3, 3), padding="same", activation="relu")(concatenate([e_x, e_r, diff], axis=-1))

    if role == "risknet":
        out = Conv2D(1, (1, 1), padding="same", activation="softplus", dtype=tf.float32)(fused)
    elif role == "correctionnet":
        out = Conv2D(
            1,
            (1, 1),
            padding="same",
            activation="linear",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            dtype=tf.float32,
        )(fused)
    else:
        raise ValueError(f"Unknown MRGR branch role: {role}")

    return Model(inputs=[x_in, r_in], outputs=out, name=f"MRGR_{role}_{source_lead}_to_{consumer_lead}")


def build_mrgr_risknet(source_lead: int, consumer_lead: int, height: int = GRID_HEIGHT, width: int = GRID_WIDTH) -> Any:
    """Constructs pair-specific MRGR RiskNet (no BN, no Dropout, Softplus head)."""
    return _build_mrgr_branch_tf(source_lead, consumer_lead, role="risknet", height=height, width=width)


def build_mrgr_correctionnet(source_lead: int, consumer_lead: int, height: int = GRID_HEIGHT, width: int = GRID_WIDTH) -> Any:
    """Constructs pair-specific MRGR CorrectionNet (no BN, no Dropout, zero-init head)."""
    return _build_mrgr_branch_tf(source_lead, consumer_lead, role="correctionnet", height=height, width=width)
