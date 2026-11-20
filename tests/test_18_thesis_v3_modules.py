"""
tests/test_18_thesis_v3_modules.py
---------------------------------
Unit test suite covering Thesis v3.0 foundations implemented offline:
  1. Kaggle Backup Store resolver (`open_kaggle_backup_object`, `restore_from_kaggle_backup`)
     with archive `.zip.bin` + loose mount resolution & MD5/CRC32C/size verification.
  2. Chronological purged 80/10/10 splits (`851 / 100 / 108`) and 9-year interval-purged
     nested OOF subset deduplication (`675 -> 183` fits).
  3. Re-mediated predictor pipeline (`ch10` `s2s_tcw` dynamic empirical scaling without [0.5, 15.0]
     clipping, `eps_floor=1e-6`, 7-day trailing lag `-1d` ERA5 anomalies, warmup checks).
  4. SCPW–MRGR parameter math (`41,664` + `25,740` = `67,404`), calendar descriptors,
     MRGR gating/Huber risk/unclipped `r*` invariants, plus guarded TF graph smoke test.
  5. Exact 110-draw empirical CRPS Eq. (15) equivalence, strict `< tau` drought Brier/reliability,
     28-calendar-day paired Moving-Block Bootstrap + Holm-Bonferroni, and RiskNet deciles.
"""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import google_crc32c
import numpy as np
import pandas as pd
import xarray as xr

from src.data.cloud_lake import (
    CloudLakeDataNotFoundError,
    _resolve_remote_artifact,
    find_consolidated_kaggle_dir,
    find_kaggle_index_dir,
    open_kaggle_backup_object,
    restore_from_consolidated_kaggle,
    restore_from_kaggle_backup,
)
from src.data.preprocessing_v3 import (
    ATM_VARS,
    S2S_VARS,
    V3PredictorNormalizationParams,
    assemble_v3_case_tensors,
    empirical_min_max_scale,
    fit_and_transform_era5_atmos,
    fit_s2s_training_anomalies,
)
from src.data.s2s import CANDIDATE_A_LATS, CANDIDATE_A_LONS
from src.data.splits_v3 import (
    build_chronological_purged_splits,
    filter_interval_purged_years,
    plan_nested_yearly_oof_folds,
)
from src.evaluation.v3_metrics_and_bootstrap import (
    calendar_block_paired_bootstrap,
    compute_dry_event_brier_and_reliability,
    compute_empirical_crps_draws,
    compute_perturbation_m_slope,
    compute_predictive_mean_deterministic_metrics,
    compute_risk_decile_validity,
    compute_spread_and_pi_coverage,
    evaluate_primary_w3_w4_family,
)
from src.models.scpw_mrgr import (
    EXPECTED_EXTRACONV_BACKBONE_PARAMS,
    EXPECTED_SCPW_PARAMS,
    apply_mrgr_gate_and_refinement,
    build_extraconv_unet_v3,
    build_mrgr_correctionnet,
    build_mrgr_risknet,
    compute_huber_risk_loss,
    compute_mrgr_scale_and_target,
    compute_scpw_calendar_descriptor,
    compute_v3_parameter_budget,
    compute_week_midpoint_date,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestKaggleCloudLakeBackupResolver(unittest.TestCase):
    """Tests Kaggle index & `.zip.bin`/loose restoration without contacting GCS."""

    def test_loose_and_archive_restore_with_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            idx_dir = tmp_path / "kaggle_input" / "mindanao-gcs-migration-index"
            loose_ds = tmp_path / "kaggle_input" / "mindanao-drought-data-gcs-migration" / "rise-unet-rzsm"
            arch_ds = tmp_path / "kaggle_input" / "rise-remaining-a-gcs-migration"
            idx_dir.mkdir(parents=True)
            loose_ds.mkdir(parents=True)
            arch_ds.mkdir(parents=True)

            loose_payload = b"mindanao-loose-netcdf-payload"
            (loose_ds / "raw_sample.nc").write_bytes(loose_payload)

            arch_payload = b"mindanao-archived-member-payload-with-bin-suffix"
            arch_zip = arch_ds / "rise-remaining-a-gcs-migration.zip.bin"
            with zipfile.ZipFile(arch_zip, "w") as zf:
                zf.writestr("packages/sample.zip.bin", arch_payload)

            def meta_for(name: str, data: bytes) -> dict:
                c = google_crc32c.Checksum()
                c.update(data)
                return {
                    "bucket": "rise-unet-rzsm",
                    "name": name,
                    "size": len(data),
                    "md5": base64.b64encode(hashlib.md5(data).digest()).decode("ascii"),
                    "crc32c": base64.b64encode(c.digest()).decode("ascii"),
                }

            rows = [
                {
                    "source": meta_for("raw/sample.nc", loose_payload),
                    "destination": {
                        "kind": "loose",
                        "dataset_ref": "aaronjalapon02/mindanao-drought-data-gcs-migration",
                        "dataset_version": 2,
                        "root_folder": "rise-unet-rzsm",
                        "relative_file": "raw_sample.nc",
                    },
                },
                {
                    "source": meta_for("packages/sample.zip", arch_payload),
                    "destination": {
                        "kind": "archive",
                        "dataset_ref": "aaronjalapon02/rise-remaining-a-gcs-migration",
                        "dataset_version": 1,
                        "archive_file": "rise-remaining-a-gcs-migration.zip.bin",
                        "archive_member": "packages/sample.zip.bin",
                    },
                },
            ]
            (idx_dir / "object-locations.json").write_text(json.dumps(rows), encoding="utf-8")

            self.assertEqual(find_kaggle_index_dir(idx_dir), idx_dir)

            with open_kaggle_backup_object(
                "packages/sample.zip",
                index_dir=idx_dir,
                kaggle_input_root=tmp_path / "kaggle_input",
            ) as (stream, meta):
                self.assertEqual(stream.read(), arch_payload)
                self.assertEqual(meta["size"], len(arch_payload))

            restored_loose = restore_from_kaggle_backup(
                "raw/sample.nc",
                tmp_path / "cache" / "sample.nc",
                index_dir=idx_dir,
                kaggle_input_root=tmp_path / "kaggle_input",
            )
            self.assertEqual(restored_loose.read_bytes(), loose_payload)

            # Corrupt loose file and ensure checksum error is raised cleanly
            (loose_ds / "raw_sample.nc").write_bytes(b"corrupted-payload-byte-count!")
            with self.assertRaises(IOError):
                restore_from_kaggle_backup(
                    "raw/sample.nc",
                    tmp_path / "cache" / "sample_corrupt.nc",
                    index_dir=idx_dir,
                    kaggle_input_root=tmp_path / "kaggle_input",
                )

            with self.assertRaises(CloudLakeDataNotFoundError):
                restore_from_kaggle_backup(
                    "nonexistent/blob.nc",
                    tmp_path / "cache" / "missing.nc",
                    index_dir=idx_dir,
                    kaggle_input_root=tmp_path / "kaggle_input",
                )

    def test_consolidated_kaggle_bundle_and_unpacked_fallback_resolution(self):
        import os
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            kinput = tmp_path / "kaggle_input"
            ds_mount = kinput / "rise-unet-rzsm"
            ds_mount.mkdir(parents=True)

            ckpt_bytes = b"keras-h5-weights-payload"
            era5_bytes = b"era5-land-netcdf-payload"
            with zipfile.ZipFile(ds_mount / "checkpoints_legacy_a0_best12.zip", "w") as zf:
                zf.writestr("checkpoints/A0/seed_42/lead_1/best_model.weights.h5", ckpt_bytes)
            with zipfile.ZipFile(ds_mount / "raw_era5_land.zip", "w") as zf:
                zf.writestr("raw/era5_land/2015_01.nc", era5_bytes)

            self.assertEqual(find_consolidated_kaggle_dir(kaggle_input_root=kinput), ds_mount)

            out_ckpt = restore_from_consolidated_kaggle(
                "checkpoints/A0/seed_42/lead_1/best_model.weights.h5",
                tmp_path / "out" / "best.weights.h5",
                kaggle_input_root=kinput,
            )
            self.assertEqual(out_ckpt.read_bytes(), ckpt_bytes)

            # Verify unpacked <bundle_stem>/<blob_name> mount compatibility
            kernel_mount = kinput / "rise-unet-rzsm-verify-and-consolidate"
            unpacked_target = kernel_mount / "processed_all" / "processed" / "grid" / "mask.nc"
            unpacked_target.parent.mkdir(parents=True)
            unpacked_target.write_bytes(b"unpacked-grid-mask")
            out_mask = restore_from_consolidated_kaggle(
                "processed/grid/mask.nc",
                tmp_path / "out" / "mask.nc",
                consolidated_dir=kernel_mount,
            )
            self.assertEqual(out_mask.read_bytes(), b"unpacked-grid-mask")

            with self.assertRaises(CloudLakeDataNotFoundError):
                restore_from_consolidated_kaggle(
                    "reference/parked_conus.zip",
                    tmp_path / "out" / "parked.zip",
                    consolidated_dir=ds_mount,
                )

            # Verify _resolve_remote_artifact prioritizes RISE_UNET_KAGGLE_CONSOLIDATED_DIR
            with mock.patch.dict(
                os.environ,
                {
                    "RISE_UNET_KAGGLE_CONSOLIDATED_DIR": str(ds_mount),
                    "RISE_UNET_ALLOW_GCS_FALLBACK": "0",
                },
                clear=False,
            ):
                resolved = _resolve_remote_artifact(
                    "raw/era5_land/2015_01.nc",
                    tmp_path / "resolved" / "2015_01.nc",
                )
                self.assertEqual(resolved.read_bytes(), era5_bytes)


class TestPurgedSplitsAndNestedOOF(unittest.TestCase):
    """Tests exact 851/100/108 splits and 9-year OOF fold isolation on the real audit CSV."""

    def test_chronological_purged_splits_and_oof_dedup(self):
        audit_csv = REPO_ROOT / "manifests" / "splits" / "cases_availability_audit.csv"
        train_df, val_df, test_df, meta = build_chronological_purged_splits(audit_csv)

        self.assertEqual(meta["usable_total"], 1073)
        self.assertEqual(meta["unpurged_counts"], {"train": 858, "val": 107, "test": 108})
        self.assertEqual(meta["purged_removed"], {"train": 7, "val": 7, "test": 0})
        self.assertEqual(len(train_df), 851)
        self.assertEqual(len(val_df), 100)
        self.assertEqual(len(test_df), 108)

        # Target-window boundary purging invariants
        self.assertLess(train_df["w4_end_date"].max(), val_df["issue_date"].min())
        self.assertLess(val_df["w4_end_date"].max(), test_df["issue_date"].min())
        self.assertEqual(train_df["w4_end_date"].max(), "2023-10-20")
        self.assertEqual(val_df["w4_end_date"].max(), "2024-11-24")
        self.assertEqual(test_df["w4_end_date"].max(), "2025-12-31")

        # 9-year nested OOF interval purging & subset deduplication
        oof = plan_nested_yearly_oof_folds(train_df["issue_date"].tolist())
        self.assertEqual(oof["naive_oof_stage_fits"], 675)
        self.assertEqual(oof["deduplicated_unique_stage_fits"], 183)

        keep, in_excl = filter_interval_purged_years(train_df["issue_date"].tolist(), [2018])
        kept_dates = pd.to_datetime(train_df.loc[keep, "issue_date"])
        kept_w4 = pd.to_datetime(train_df.loc[keep, "w4_end_date"])
        # Zero retained training cases have target windows touching 2018-01-01..2018-12-31
        touches_2018 = (kept_dates <= pd.Timestamp("2018-12-31")) & (kept_w4 >= pd.Timestamp("2018-01-01"))
        self.assertFalse(bool(np.any(touches_2018)))
        self.assertEqual(int(in_excl.sum()), 97)


class TestV3PreprocessingFixes(unittest.TestCase):
    """Verifies dynamic empirical scaling fixes ch10 tcw saturation & lag=-1d atmos timing."""

    def test_ch10_tcw_dynamic_scaling_and_atmos_lag_minus_1(self):
        h, w = len(CANDIDATE_A_LATS), len(CANDIDATE_A_LONS)
        eval_mask = np.zeros((h, w), dtype=bool)
        eval_mask[10:24, 15:24] = True
        self.assertEqual(int(eval_mask.sum()), 126)

        # Build daily atmos from 2014-12-25 to 2015-12-31 (includes warmup)
        times = pd.date_range("2014-12-25", "2015-12-31", freq="1D")
        rng = np.random.default_rng(7)
        atm_data = {
            var: (("time", "lat", "lon"), (10.0 + i + rng.normal(0, 1, size=(len(times), h, w))).astype(np.float32))
            for i, var in enumerate(ATM_VARS)
        }
        atm_ds = xr.Dataset(data_vars=atm_data, coords={"time": times, "lat": CANDIDATE_A_LATS, "lon": CANDIDATE_A_LONS})

        # Confirm missing warmup (starting 2015-01-01) raises ValueError when min_check_date='2015-01-01'
        with self.assertRaises(ValueError):
            fit_and_transform_era5_atmos(
                atm_ds.sel(time=slice("2015-01-01", None)),
                eval_mask=eval_mask,
                max_train_date="2015-12-31",
                min_check_date="2015-01-01",
                allow_incomplete_warmup=False,
            )

        anom_atm_ds, _, atm_bounds = fit_and_transform_era5_atmos(
            atm_ds,
            eval_mask=eval_mask,
            max_train_date="2015-12-31",
            min_check_date="2014-12-31",
        )
        # Verify eps_floor stability on constant field
        const_scaled, _ = empirical_min_max_scale(np.full((3, 3), 5.0, dtype=np.float32), 5.0, 5.0)
        self.assertTrue(np.allclose(const_scaled, 0.0))

        # Build realistic Mindanao S2S datasets where tcw spans [27.6, 53.7] (exceeding legacy [0.5, 15.0])
        issues = ["2015-01-15", "2015-04-15", "2015-07-15", "2015-10-15"]
        s2s_by_issue = {}
        for k_i, d in enumerate(issues):
            tcw_arr = np.linspace(27.66, 53.77, 2 * 11 * h * w, dtype=np.float32).reshape(2, 11, h, w)
            t2m_arr = (297.0 + 0.4 * k_i + rng.normal(0, 0.5, size=(2, 11, h, w))).astype(np.float32)
            d2m_arr = (293.0 + 0.3 * k_i + rng.normal(0, 0.5, size=(2, 11, h, w))).astype(np.float32)
            s2s_by_issue[d] = xr.Dataset(
                data_vars={
                    "t2m": (("lead", "member", "lat", "lon"), t2m_arr),
                    "d2m": (("lead", "member", "lat", "lon"), d2m_arr),
                    "tcw": (("lead", "member", "lat", "lon"), tcw_arr),
                },
                coords={"lead": [1, 2], "member": list(range(11)), "lat": CANDIDATE_A_LATS, "lon": CANDIDATE_A_LONS},
            )

        s2s_clim, s2s_bounds = fit_s2s_training_anomalies(s2s_by_issue, eval_mask, issues)
        rzsm_dates = pd.date_range("2015-01-01", "2015-03-01", freq="1D")
        rzsm_ds = xr.Dataset(
            data_vars={
                "rzsm_0_100_seasonal_anomaly": (
                    ("time", "lat", "lon"),
                    rng.uniform(-0.15, 0.15, size=(len(rzsm_dates), h, w)).astype(np.float32),
                )
            },
            coords={"time": rzsm_dates, "lat": CANDIDATE_A_LATS, "lon": CANDIDATE_A_LONS},
        )

        params = V3PredictorNormalizationParams(
            rzsm_min=-0.185,
            rzsm_max=0.157,
            atm_climatology={},
            atm_bounds=atm_bounds,
            s2s_climatology=s2s_clim,
            s2s_bounds=s2s_bounds,
        )
        case, _ = assemble_v3_case_tensors(
            issue_date="2015-01-15",
            rzsm_anom_ds=rzsm_ds,
            atmos_anom_ds=anom_atm_ds,
            s2s_ds=s2s_by_issue["2015-01-15"],
            eval_mask=eval_mask,
            norm_params=params,
        )
        ch10_active = case.x_w1[:, eval_mask, 10]
        # Legacy bug clipped ch10 identically to 1.0000 (std == 0); v3 must retain rich spread!
        self.assertGreater(float(np.std(ch10_active)), 0.05)
        self.assertLess(float(np.mean(ch10_active)), 0.99)
        self.assertEqual(float(np.max(np.abs(case.x_w1[:, ~eval_mask, :]))), 0.0)


class TestSCPWMRGRAndEvaluation(unittest.TestCase):
    """Tests SCPW-MRGR math, exact parameter counts, CRPS Eq. (15), Brier < tau, and 28d MBB."""

    def test_parameter_budgets_and_calendar_descriptor(self):
        budget = compute_v3_parameter_budget()
        self.assertEqual(budget["backbone_extraconv_per_lead"], EXPECTED_EXTRACONV_BACKBONE_PARAMS)
        self.assertEqual(budget["scpw_per_lead"], EXPECTED_SCPW_PARAMS)
        self.assertEqual(budget["scpw_total"], 41_664)
        self.assertEqual(budget["mrgr_total"], 25_740)
        self.assertEqual(budget["combined_refinement_overhead"], 67_404)

        self.assertEqual(compute_week_midpoint_date("2015-01-01", 1), pd.Timestamp("2015-01-04"))
        self.assertEqual(compute_week_midpoint_date("2015-01-01", 4), pd.Timestamp("2015-01-25"))

        s_nonleap = compute_scpw_calendar_descriptor(["2015-01-01"], lead=1)
        self.assertEqual(s_nonleap.shape, (1, 2))
        self.assertAlmostEqual(float(np.linalg.norm(s_nonleap[0])), 1.0, places=6)

        s_ctrl = compute_scpw_calendar_descriptor(["2015-01-01", "2020-02-26"], lead=2, constant_control=True)
        self.assertTrue(np.all(s_ctrl == 0.0))

    def test_mrgr_scale_huber_gate_and_unclipped_refinement(self):
        eval_mask = np.zeros((4, 4), dtype=bool)
        eval_mask[1:3, 1:3] = True
        r_oof = np.full((10, 4, 4), 0.30, dtype=np.float32)
        r_oof[0, 1, 1] = -0.05  # intentionally slightly negative prior
        y_true = np.full((10, 4, 4), 0.25, dtype=np.float32)

        c_scale, q_target = compute_mrgr_scale_and_target(r_oof, y_true, eval_mask)
        self.assertGreater(c_scale, 0.04)
        huber = compute_huber_risk_loss(q_target, q_target, eval_mask)
        self.assertAlmostEqual(huber, 0.0, places=10)

        r_star, diag = apply_mrgr_gate_and_refinement(
            r_oof[:1], np.full((1, 4, 4), -2.0), c_scale=c_scale, q_hat=np.full((1, 4, 4), 1.0)
        )
        # No silent clipping: negative excursions must remain negative and be counted
        self.assertLess(diag["min_val"], 0.0)
        self.assertGreater(diag["below_zero_count"], 0)

        r_ungated, _ = apply_mrgr_gate_and_refinement(r_oof[:1], np.full((1, 4, 4), 10.0), c_scale=0.1, ungated_ablation=True)
        self.assertAlmostEqual(float(r_ungated[0, 1, 2] - r_oof[0, 1, 2]), 0.1, delta=1e-4)

    def test_tensorflow_extraconv_scpw_mrgr_when_available(self):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            self.skipTest("TensorFlow not installed locally; graph verification runs on Colab/Kaggle GPU.")

        for lead, exp_base in EXPECTED_EXTRACONV_BACKBONE_PARAMS.items():
            m_base = build_extraconv_unet_v3(lead=lead, use_scpw=False)
            self.assertEqual(m_base.count_params(), exp_base)
            m_scpw = build_extraconv_unet_v3(lead=lead, use_scpw=True)
            self.assertEqual(m_scpw.count_params(), exp_base + EXPECTED_SCPW_PARAMS[lead])

        self.assertEqual(build_mrgr_risknet(1, 2).count_params(), 2_625)
        self.assertEqual(build_mrgr_correctionnet(3, 4).count_params(), 2_049)

    def test_crps_brier_strict_inequality_and_calendar_bootstrap(self):
        eval_mask = np.zeros((4, 4), dtype=bool)
        eval_mask[1:3, 1:3] = True
        rng = np.random.default_rng(12)
        draws = rng.normal(0.30, 0.04, size=(110, 4, 4))
        y_true = np.full((4, 4), 0.28)

        # Verify exact equality between O(M log M) Eq. 15 formula and O(M^2) brute force
        crps_fast = compute_empirical_crps_draws(draws, y_true, eval_mask)
        x_act = draws[:, eval_mask]
        y_act = y_true[eval_mask]
        brute_term1 = np.mean(np.abs(x_act - y_act[None, :]), axis=0)
        brute_term2 = np.mean(np.abs(x_act[:, None, :] - x_act[None, :, :]), axis=(0, 1)) / 2.0
        crps_brute = float(np.mean(brute_term1 - brute_term2))
        self.assertAlmostEqual(crps_fast, crps_brute, places=12)

        det = compute_predictive_mean_deterministic_metrics(draws, y_true, eval_mask)
        self.assertTrue(det["zero_variance_acc"])
        cov = compute_spread_and_pi_coverage(draws, y_true, eval_mask)
        self.assertGreater(cov["coverage_95"], 0.8)

        # Strict < tau check: exact boundary equality y == tau must NOT count as dry
        tau = np.full((4, 4), 0.28)
        exact_draws = np.full((110, 4, 4), 0.28)
        brier_res = compute_dry_event_brier_and_reliability(exact_draws, np.full((4, 4), 0.28), tau, eval_mask)
        self.assertEqual(brier_res["event_rate"], 0.0)
        self.assertEqual(brier_res["mean_forecast_prob"], 0.0)
        self.assertEqual(sum(brier_res["bin_counts"]), int(eval_mask.sum()))

        # 28-day paired calendar bootstrap & Holm-Bonferroni
        dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2023-10-21", "2024-10-28", freq="4D")]
        diff_w3 = np.full(len(dates), -0.008) + rng.normal(0.0, 0.002, size=len(dates))
        diff_w4 = np.full(len(dates), -0.012) + rng.normal(0.0, 0.002, size=len(dates))
        fam = evaluate_primary_w3_w4_family(dates, {3: diff_w3, 4: diff_w4}, block_days=28, n_boot=500, seed=42)
        self.assertLess(fam["W3"]["adj_p"], 0.05)
        self.assertLess(fam["W4"]["adj_p"], 0.05)

        dec = compute_risk_decile_validity(np.linspace(0, 1, 200), np.linspace(0, 0.2, 200), np.linspace(0, 0.15, 200))
        self.assertGreater(dec["spearman_rho"], 0.99)
        slope = compute_perturbation_m_slope({0.0: 0.04, 0.5: 0.045, 1.0: 0.05, 1.5: 0.06, 2.0: 0.07})
        self.assertAlmostEqual(slope, 0.02, places=6)


if __name__ == "__main__":
    unittest.main()
