#!/usr/bin/env python3
"""
Mindanao 0.25° Spatial Mask Generator
Project: Enhanced RISE-UNet for Subseasonal RZSM Drought Forecasting in Mindanao
Track: Track B (Mindanao Regional Adaptation)

Description:
  Computes the fractional boundary-coverage mask and derives the binary evaluation
  mask for the frozen 32 x 48 (0.25°) spatial domain over Mindanao using the
  authoritative PSA-NAMRIA regional administrative boundary.
  Area quadrature is computed strictly via WGS84 ellipsoidal geodesics (pyproj.Geod).

Outputs:
  - processed/grid/mindanao_fraction_025.nc (float32, values in [0.0, 1.0])
  - processed/grid/mindanao_eval_mask_025.nc (int8, values in {0, 1}, f >= 0.50)
"""

import os
import sys
import time
import hashlib
import sqlite3
import numpy as np
import shapely
import shapely.wkb
import shapely.geometry
from shapely import STRtree
from pyproj import Geod
import netCDF4 as nc

sys.stdout.reconfigure(line_buffering=True)

def compute_sha256(filepath: str) -> str:
    """Compute hex SHA-256 digest of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()

def main():
    print("=" * 80)
    print("MINDANAO 0.25° SPATIAL MASK GENERATION (WGS84 GEODESIC QUADRATURE)")
    print("=" * 80)
    
    # 1. Paths
    boundary_gpkg = os.path.join("processed", "boundary", "mindanao_analysis_boundary.gpkg")
    grid_nc = os.path.join("processed", "grid", "mindanao_025deg.nc")
    output_fraction_nc = os.path.join("processed", "grid", "mindanao_fraction_025.nc")
    output_eval_nc = os.path.join("processed", "grid", "mindanao_eval_mask_025.nc")
    
    assert os.path.exists(boundary_gpkg), f"Missing boundary: {boundary_gpkg}"
    assert os.path.exists(grid_nc), f"Missing coordinate grid: {grid_nc}"
    
    boundary_hash = compute_sha256(boundary_gpkg)
    print(f"Boundary File : {boundary_gpkg}")
    print(f"Boundary SHA256: {boundary_hash}")
    
    # 2. Load Boundary Geometry from GeoPackage
    t0 = time.time()
    conn = sqlite3.connect(boundary_gpkg)
    cur = conn.cursor()
    cur.execute("SELECT geom FROM mindanao_analysis_boundary")
    blob = cur.fetchone()[0]
    conn.close()
    
    flags = blob[3]
    envelope_type = (flags >> 1) & 0x07
    header_lens = {0: 8, 1: 40, 2: 56, 3: 56, 4: 72}
    h_len = header_lens.get(envelope_type, 8)
    boundary_geom = shapely.wkb.loads(blob[h_len:])
    
    poly_list = list(boundary_geom.geoms)
    tree = STRtree(poly_list)
    print(f"Loaded boundary ({len(poly_list)} sub-polygons) in {time.time() - t0:.2f}s")
    
    geod = Geod(ellps="WGS84")
    boundary_area_m2, _ = geod.geometry_area_perimeter(boundary_geom)
    boundary_area_km2 = abs(boundary_area_m2) / 1e6
    print(f"Authoritative boundary geodetic area: {boundary_area_km2:.2f} km^2")
    
    # 3. Load Coordinate Grid
    with nc.Dataset(grid_nc, "r") as ds:
        lats = ds.variables["lat"][:]
        lons = ds.variables["lon"][:]
    
    H, W = len(lats), len(lons)
    print(f"Frozen Grid Dimension: {H} latitudes x {W} longitudes ({H * W} cells)")
    
    # 4. Geodesic Area Quadrature
    t1 = time.time()
    fraction_mask = np.zeros((H, W), dtype=np.float32)
    cell_areas_km2 = np.zeros((H, W), dtype=np.float64)
    half_res = 0.125
    
    for i, lat in enumerate(lats):
        lat_min = lat - half_res
        lat_max = lat + half_res
        for j, lon in enumerate(lons):
            lon_min = lon - half_res
            lon_max = lon + half_res
            
            cell_box = shapely.geometry.box(lon_min, lat_min, lon_max, lat_max)
            cell_area_m2 = abs(geod.geometry_area_perimeter(cell_box)[0])
            cell_areas_km2[i, j] = cell_area_m2 / 1e6
            
            hits = tree.query(cell_box, predicate="intersects")
            if len(hits) == 0:
                fraction_mask[i, j] = 0.0
                continue
                
            total_inter_area = 0.0
            for idx in hits:
                inter = poly_list[idx].intersection(cell_box)
                if not inter.is_empty:
                    ia = abs(geod.geometry_area_perimeter(inter)[0])
                    total_inter_area += ia
                    
            f = total_inter_area / cell_area_m2
            fraction_mask[i, j] = float(np.clip(f, 0.0, 1.0))
            
    print(f"Calculated all {H * W} cells in {time.time() - t1:.2f}s")
    
    # 5. Derive Binary Evaluation Mask (f >= 0.50)
    eval_mask = (fraction_mask >= 0.50).astype(np.int8)
    
    # 6. Verification
    integrated_frac_area_km2 = np.sum(fraction_mask * cell_areas_km2)
    integrated_eval_area_km2 = np.sum(eval_mask * cell_areas_km2)
    diff_pct = abs(integrated_frac_area_km2 - boundary_area_km2) / boundary_area_km2 * 100
    
    print("-" * 80)
    print("INTEGRATION METRICS:")
    print(f"  Boundary Geodetic Area      : {boundary_area_km2:.2f} km^2")
    print(f"  Integrated Fractional Area  : {integrated_frac_area_km2:.2f} km^2 (Discrepancy: {diff_pct:.4f}%)")
    print(f"  Integrated Binary Mask Area : {integrated_eval_area_km2:.2f} km^2")
    print(f"  Active Boundary Cells (f>0) : {np.count_nonzero(fraction_mask > 0)} / {H * W} ({np.count_nonzero(fraction_mask > 0)/(H*W)*100:.2f}%)")
    print(f"  Binary Eval Cells (f>=0.50) : {np.count_nonzero(eval_mask == 1)} / {H * W} ({np.count_nonzero(eval_mask == 1)/(H*W)*100:.2f}%)")
    print("-" * 80)
    
    # 7. Write NetCDF Datasets
    # A. Fractional mask
    with nc.Dataset(output_fraction_nc, "w", format="NETCDF4") as ds:
        ds.title = "Mindanao 0.25° Fractional Regional Boundary Coverage Mask"
        ds.institution = "RISE-UNet Mindanao RZSM Drought Forecasting Project"
        ds.source = f"PSA-NAMRIA Regional Boundary (SHA-256: {boundary_hash})"
        ds.method = "WGS84 ellipsoidal geodesic quadrature via pyproj.Geod"
        ds.history = f"Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
        ds.Conventions = "CF-1.8"
        
        ds.createDimension("lat", H)
        ds.createDimension("lon", W)
        
        v_lat = ds.createVariable("lat", "f4", ("lat",))
        v_lat.standard_name = "latitude"
        v_lat.long_name = "latitude coordinate of grid cell center"
        v_lat.units = "degrees_north"
        v_lat.axis = "Y"
        v_lat[:] = lats
        
        v_lon = ds.createVariable("lon", "f4", ("lon",))
        v_lon.standard_name = "longitude"
        v_lon.long_name = "longitude coordinate of grid cell center"
        v_lon.units = "degrees_east"
        v_lon.axis = "X"
        v_lon[:] = lons
        
        v_frac = ds.createVariable("fractional_coverage", "f4", ("lat", "lon"), zlib=True)
        v_frac.long_name = "Fractional coverage of grid cell by Mindanao analysis boundary"
        v_frac.units = "1"
        v_frac.valid_range = np.array([0.0, 1.0], dtype=np.float32)
        v_frac[:] = fraction_mask

    # B. Binary evaluation mask
    with nc.Dataset(output_eval_nc, "w", format="NETCDF4") as ds:
        ds.title = "Mindanao 0.25° Binary Evaluation Mask (f >= 0.50)"
        ds.institution = "RISE-UNet Mindanao RZSM Drought Forecasting Project"
        ds.source = "Derived from mindanao_fraction_025.nc with inclusion threshold f >= 0.50"
        ds.history = f"Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
        ds.Conventions = "CF-1.8"
        
        ds.createDimension("lat", H)
        ds.createDimension("lon", W)
        
        v_lat = ds.createVariable("lat", "f4", ("lat",))
        v_lat.standard_name = "latitude"
        v_lat.long_name = "latitude coordinate of grid cell center"
        v_lat.units = "degrees_north"
        v_lat.axis = "Y"
        v_lat[:] = lats
        
        v_lon = ds.createVariable("lon", "f4", ("lon",))
        v_lon.standard_name = "longitude"
        v_lon.long_name = "longitude coordinate of grid cell center"
        v_lon.units = "degrees_east"
        v_lon.axis = "X"
        v_lon[:] = lons
        
        v_mask = ds.createVariable("evaluation_mask", "i1", ("lat", "lon"), zlib=True)
        v_mask.long_name = "Binary evaluation mask for Mindanao loss and verification"
        v_mask.units = "1"
        v_mask.valid_range = np.array([0, 1], dtype=np.int8)
        v_mask[:] = eval_mask

    print(f"Saved: {output_fraction_nc} (SHA-256: {compute_sha256(output_fraction_nc)})")
    print(f"Saved: {output_eval_nc} (SHA-256: {compute_sha256(output_eval_nc)})")
    print("✓ Mask generation complete.")

if __name__ == "__main__":
    main()
