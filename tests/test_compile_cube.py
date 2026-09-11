"""
tests/test_compile_cube.py
--------------------------
Automated unit test suite verifying the full production RZSM cube compilation pipeline,
multi-year temporal processing, locked Model A0 contracts, and census certification (Step 21D.4).
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from src.data.compile_cube import (
    ProductionCubeConfig,
    compile_production_rzsm_pipeline,
    verify_production_cube_census,
    FastLandAwareRemapper,
    process_era5_land_monthly_pair,
    process_era5_land_antecedent_file,
    compile_full_11yr_rzsm_cube,
)
from src.data.rzsm import remap_era5_land_to_candidate_a


class TestProductionCubePipeline(unittest.TestCase):
    """Verifies end-to-end multi-year RZSM compilation pipeline on synthetic grid data."""

    def setUp(self):
        # Create 7 years of daily data (2015-01-01 to 2021-12-31) on 32 x 48 grid
        dates = pd.date_range("2015-01-01", "2021-12-31", freq="D")
        lats = np.linspace(11.75, 4.0, 32)
        lons = np.linspace(116.0, 127.75, 48)

        # Realistic synthetic soil moisture: background + seasonal wave + spatial variation
        n_t = len(dates)
        t_arr = np.arange(n_t)
        seasonal_wave = 0.38 + 0.08 * np.sin(2 * np.pi * t_arr / 365.25)[:, None, None]
        spatial_grad = np.linspace(0.0, 0.05, 32)[:, None] + np.linspace(0.0, 0.05, 48)[None, :]
        data_arr = (seasonal_wave + spatial_grad[None, ...]).astype(np.float32)

        # 126 active evaluation cells in central Mindanao domain
        self.eval_mask = np.zeros((32, 48), dtype=np.int32)
        # Select 126 specific cells deterministically
        eval_indices = np.unravel_index(np.linspace(12*48+15, 26*48+35, 126, dtype=int), (32, 48))
        self.eval_mask[eval_indices] = 1
        self.assertEqual(int(self.eval_mask.sum()), 126)

        # Mask inactive cells with 0.0 in the raw array
        data_arr[:, self.eval_mask == 0] = 0.0

        self.daily_da = xr.DataArray(
            data_arr,
            coords={"time": dates, "lat": lats, "lon": lons},
            dims=["time", "lat", "lon"],
            name="rzsm_0_100",
        )

    def test_pipeline_execution_and_census_certification(self):
        """Verify full compilation execution, CF attributes, and census certification."""
        config = ProductionCubeConfig(
            train_start_year=2015,
            train_end_year=2021,
            rolling_window=7,
            min_periods=7,
            climatology_method="season",
            expected_eval_cells=126,
        )

        prod_ds = compile_production_rzsm_pipeline(
            self.daily_da,
            eval_mask=self.eval_mask,
            config=config,
        )

        # Verify dataset contents
        expected_vars = {
            "rzsm_0_100_raw",
            "rzsm_0_100_rolling_7d",
            "rzsm_0_100_seasonal_anomaly",
            "rzsm_0_100_normalized",
            "climatology_seasonal",
            "evaluation_mask",
        }
        self.assertTrue(expected_vars.issubset(set(prod_ds.data_vars)))

        # Verify dimensions and shapes
        self.assertEqual(prod_ds["rzsm_0_100_normalized"].shape, (len(self.daily_da.time), 32, 48))
        self.assertEqual(prod_ds["climatology_seasonal"].shape, (4, 32, 48))
        self.assertEqual(prod_ds.attrs["active_evaluation_cells"], 126)
        self.assertEqual(prod_ds.attrs["climatology_method"], "season")

        # Execute formal census audit
        census = verify_production_cube_census(prod_ds, eval_mask=self.eval_mask)

        # Assert mandatory audit pass criteria
        self.assertTrue(census["is_certified"])
        self.assertEqual(census["nans_active_cells"], 0)
        self.assertEqual(census["infs_active_cells"], 0)
        self.assertEqual(census["nonzero_ocean_cells"], 0)
        self.assertEqual(census["active_cells_count"], 126)
        self.assertEqual(census["ocean_cells_count"], 1410)
        self.assertGreaterEqual(census["active_min"], 0.0)
        self.assertLessEqual(census["active_max"], 1.0)

        expected_phrase = (
            "Zero NaNs/Infs across the 126 active evaluation cells; non-evaluation computational "
            "cells follow the frozen masking/zero-fill convention."
        )
        self.assertEqual(census["mandatory_reporting_standard"], expected_phrase)

    def test_out_of_sample_leakage_isolation(self):
        """Verify adding out-of-sample data (2022) does not alter training climatology or bounds."""
        config = ProductionCubeConfig(train_start_year=2015, train_end_year=2021)
        ds_baseline = compile_production_rzsm_pipeline(self.daily_da, eval_mask=self.eval_mask, config=config)

        # Append year 2022 with extreme perturbed data
        dates_2022 = pd.date_range("2022-01-01", "2022-12-31", freq="D")
        perturbed_vals = np.full((len(dates_2022), 32, 48), 0.99, dtype=np.float32)
        perturbed_vals[:, self.eval_mask == 0] = 0.0
        da_2022 = xr.DataArray(
            perturbed_vals,
            coords={"time": dates_2022, "lat": self.daily_da.lat, "lon": self.daily_da.lon},
            dims=["time", "lat", "lon"],
            name="rzsm_0_100",
        )
        da_extended = xr.concat([self.daily_da, da_2022], dim="time")

        ds_extended = compile_production_rzsm_pipeline(da_extended, eval_mask=self.eval_mask, config=config)

        # Climatology and bounds must be IDENTICAL to machine precision
        np.testing.assert_allclose(
            ds_baseline["climatology_seasonal"].values,
            ds_extended["climatology_seasonal"].values,
            rtol=1e-6,
            err_msg="Future data leakage! Year 2022 data contaminated training climatology.",
        )
        self.assertAlmostEqual(
            ds_baseline.attrs["train_min_scalar"],
            ds_extended.attrs["train_min_scalar"],
            places=6,
        )
        self.assertAlmostEqual(
            ds_baseline.attrs["train_max_scalar"],
            ds_extended.attrs["train_max_scalar"],
            places=6,
        )

    def test_fast_remapper_numerical_parity(self):
        """Assert FastLandAwareRemapper matches remap_era5_land_to_candidate_a bitwise."""
        ant_file = Path("pilot_raw/era5-land-2014-12-antecedent.nc")
        grid_nc = Path("processed/grid/mindanao_025deg.nc")
        mask_nc = Path("processed/grid/mindanao_eval_mask_025.nc")

        if not (ant_file.exists() and grid_nc.exists() and mask_nc.exists()):
            self.skipTest("Required foundation files missing for remapper test.")

        with xr.open_dataset(ant_file) as ds_ant, xr.open_dataset(grid_nc) as ds_grid, xr.open_dataset(mask_nc) as ds_mask:
            sample_2d = ds_ant["swvl1"].isel(valid_time=0)
            tgt_lats = ds_grid["lat"].values
            tgt_lons = ds_grid["lon"].values
            eval_mask = ds_mask["evaluation_mask"].values

            # Reference remapping
            ref_remapped = remap_era5_land_to_candidate_a(sample_2d, tgt_lats, tgt_lons, eval_mask)

            # Fast remapper
            remapper = FastLandAwareRemapper.from_source_and_target_grid(
                sample_source_da=sample_2d,
                target_grid_nc=grid_nc,
                eval_mask_nc=mask_nc,
            )
            fast_remapped = remapper.remap_source_da(sample_2d)

            # Assert bitwise numerical equivalence
            np.testing.assert_allclose(
                ref_remapped.values,
                fast_remapped.values,
                atol=1e-6,
                err_msg="FastLandAwareRemapper deviates from canonical remap_era5_land_to_candidate_a!",
            )

    def test_process_era5_land_antecedent_file(self):
        """Verify processing of the real 2014 antecedent support file."""
        ant_file = Path("pilot_raw/era5-land-2014-12-antecedent.nc")
        grid_nc = Path("processed/grid/mindanao_025deg.nc")
        mask_nc = Path("processed/grid/mindanao_eval_mask_025.nc")

        if not (ant_file.exists() and grid_nc.exists() and mask_nc.exists()):
            self.skipTest("Required foundation files missing for antecedent file test.")

        with xr.open_dataset(ant_file) as ds_ant:
            remapper = FastLandAwareRemapper.from_source_and_target_grid(
                sample_source_da=ds_ant["swvl1"],
                target_grid_nc=grid_nc,
                eval_mask_nc=mask_nc,
            )

        da_ant = process_era5_land_antecedent_file(ant_file, remapper=remapper)

        # Assert 20 days: 2014-12-12 to 2014-12-31
        self.assertEqual(len(da_ant.time), 20)
        self.assertEqual(str(pd.Timestamp(da_ant.time.values[0]).date()), "2014-12-12")
        self.assertEqual(str(pd.Timestamp(da_ant.time.values[-1]).date()), "2014-12-31")
        self.assertEqual(da_ant.shape, (20, 32, 48))

        # Check finite on active cells, 0.0 on buffer
        eval_mask = xr.open_dataset(mask_nc)["evaluation_mask"].values
        active_vals = da_ant.values[:, eval_mask == 1]
        buffer_vals = da_ant.values[:, eval_mask == 0]

        self.assertEqual(int(np.isnan(active_vals).sum()), 0)
        self.assertEqual(int(np.isinf(active_vals).sum()), 0)
        self.assertEqual(int(np.count_nonzero(buffer_vals)), 0)

    def test_compile_full_cube_partial_pipeline(self):
        """Verify end-to-end compilation with allow_partial=True on pilot_raw."""
        ant_file = Path("pilot_raw/era5-land-2014-12-antecedent.nc")
        if not ant_file.exists():
            self.skipTest("pilot_raw antecedent file missing.")

        config = ProductionCubeConfig(
            train_start_year=2014,
            train_end_year=2014,
            rolling_window=7,
            min_periods=7,
        )

        prod_ds, census = compile_full_11yr_rzsm_cube(
            archive_dir=Path("pilot_raw"),
            config=config,
            slice_nominal_period=False,
            allow_partial=True,
            verbose=False,
        )

        self.assertIn("rzsm_0_100_normalized", prod_ds.data_vars)
        self.assertTrue(census["is_certified"])
        self.assertEqual(census["nans_active_cells"], 0)
        self.assertEqual(census["nonzero_ocean_cells"], 0)


if __name__ == "__main__":
    unittest.main()

