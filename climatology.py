#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import dask
from loguru import logger
import dask.array as da
import xarray as xr

# Statistical constants
MIN_R2_THRESHOLD = 0.15  # Minimum variance explained for harmonic fit
MIN_DATA_POINTS = 4      # Minimum points needed for regression
N_STD_DEVIATIONS = 3     # Number of standard deviations for climatology bounds


def _auto_chunk_time(da_in):
    """
    Ensure time dimension is in a single chunk for resampling/groupby operations.
    Other dimensions are auto-chunked for memory efficiency.
    """
    chunk_dict = {}
    for dim, size in da_in.sizes.items():
        if dim == "time":
            chunk_dict[dim] = -1  # all time in one chunk for resample/groupby
        else:
            chunk_dict[dim] = "auto"
    return da_in.chunk(chunk_dict)


def _compute_monthly_means(da):
    """
    Compute monthly means grouped by calendar month (1..12).
    Return an xarray DataArray with a 'month' dimension (1..12).
    This is kept lazy (no .compute()) so caller can decide when to evaluate.
    """
    da = _auto_chunk_time(da)

    # Group by calendar month (1..12) — keeps 'month' as a dimension
    m = da.groupby("time.month").mean(dim="time")

    if "depth" in m.dims:
        m = m.mean(dim="depth")  # lazy depth averaging

    # Ensure months 1..12 exist and are in order
    all_months = xr.DataArray(np.arange(1, 13), dims="month", name="month")
    m = m.reindex(month=all_months)  # missing months will be NaN

    return m  # xarray DataArray with 'month' dim


def _compute_monthly_std(da):
    """
    Fully Dask-optimized monthly std.
    Returns an xarray DataArray indexed by 'month' (1..12), lazy.
    
    Handles cyclic boundary (December→January) with proper interpolation.
    
    Uses sequential indexing for interpolation to avoid non-monotonic error.
    """
    da = _auto_chunk_time(da)

    g = da.groupby("time.month").std(dim="time")

    if "depth" in g.dims:
        g = g.mean(dim="depth")  # lazy depth averaging

    # Ensure all months exist
    all_months = xr.DataArray(np.arange(1, 13), dims="month", name="month")
    g = g.reindex(month=all_months)

    # Rechunk month to a single chunk for interpolation (safe)
    g = g.chunk({"month": -1})

    # Cyclic boundary handling with sequential indexing
    # This avoids the "Index 'month' must be monotonically increasing" error
    if g.isnull().any():
        # Create padded array: [Nov, Dec, Jan...Dec, Jan, Feb]
        # Using month values: [11, 12, 1, 2, 3, ..., 12, 1, 2]
        pad_start = g.sel(month=[11, 12])
        pad_end = g.sel(month=[1, 2])
        padded = xr.concat([pad_start, g, pad_end], dim='month')
        
        # Create sequential index (0, 1, 2, ..., 15) to make it monotonically increasing
        padded_values = padded.values
        padded_sequential = xr.DataArray(
            padded_values,
            dims=['month'],
            coords={'month': np.arange(len(padded_values))}
        )
        
        # Now interpolation will work because index is sequential
        padded_sequential = padded_sequential.interpolate_na(dim="month", method="linear", max_gap=None)
        
        # Extract the middle 12 months (indices 2-13)
        g_interpolated = padded_sequential.isel(month=slice(2, 14))
        
        # Restore the original month coordinate (1..12)
        g = xr.DataArray(
            g_interpolated.values,
            dims=['month'],
            coords={'month': np.arange(1, 13)}
        )

    # Return xarray DataArray (still lazy)
    return g


def _harmonic_regression(ts):
    """
    Perform the 4-cycle harmonic regression used by OOI QARTOD.
    
    The regression model includes:
    - Constant term (mean)
    - Annual cycle (12-month period): sin(2πft), cos(2πft)
    - Semi-annual cycle (6-month): sin(4πft), cos(4πft)
    - Tertiary cycle (4-month): sin(6πft), cos(6πft)
    - Quarterly cycle (3-month): sin(8πft), cos(8πft)
    
    Args:
        ts: Array of monthly values (length N, typically 12)
    
    Returns:
        beta: Regression coefficients [9 values]
        r2: R-squared value (variance explained)
        N: Number of data points
    """
    f = 1 / 12  # Fundamental frequency (annual cycle)
    N = len(ts)
    t = np.arange(N)

    # Filter out NaN values
    mask = ~np.isnan(ts)
    ts_fit = ts[mask]
    t_fit = t[mask]
    n = len(ts_fit)

    if n < MIN_DATA_POINTS:
        logger.warning(f"Insufficient data for harmonic regression: {n} points (need >={MIN_DATA_POINTS})")
        return None, None, None

    # Design matrix: intercept + 4 harmonic pairs (8 coefficients)
    X = np.column_stack([
        np.ones(n),                        # beta[0]: intercept
        np.sin(2*np.pi*f*t_fit),          # beta[1]: annual sin
        np.cos(2*np.pi*f*t_fit),          # beta[2]: annual cos
        np.sin(4*np.pi*f*t_fit),          # beta[3]: semi-annual sin
        np.cos(4*np.pi*f*t_fit),          # beta[4]: semi-annual cos
        np.sin(6*np.pi*f*t_fit),          # beta[5]: tertiary sin
        np.cos(6*np.pi*f*t_fit),          # beta[6]: tertiary cos
        np.sin(8*np.pi*f*t_fit),          # beta[7]: quarterly sin
        np.cos(8*np.pi*f*t_fit),          # beta[8]: quarterly cos
    ])

    # Least squares regression
    beta, resid, rank, s = np.linalg.lstsq(X, ts_fit, rcond=None)

    # Calculate R-squared
    if ts_fit.size == 0:
        r2 = 0.0
    else:
        # resid is sum of squared residuals (scalar or 1-element array)
        rss = float(resid[0]) if (isinstance(resid, np.ndarray) and resid.size > 0) else float(resid)
        tss = np.sum((ts_fit - ts_fit.mean())**2)
        
        if tss == 0:
            # All values identical - model fits perfectly but explains no variance
            # Set r2 = 0 to indicate constant data
            r2 = 0.0
            logger.debug("Constant data detected (TSS=0), setting R²=0")
        else:
            r2 = 1.0 - (rss / tss)

    return beta, r2, N


def _compute_monthly_fit(monthly_means):
    """
    Compute the monthly fitted climatology curve from monthly means.
    
    Uses harmonic regression with 4 cycles. If regression fails (R² < 0.15)
    or cannot be computed, falls back to raw monthly means.
    
    Uses all computed harmonic coefficients (not just first 5).

    Args:
        monthly_means: pandas Series or xarray DataArray with 12 monthly values
    
    Returns:
        fitted_monthly: pandas Series with fitted values indexed 1..12
        r2: R-squared value (or None if fallback used)
    """
    # Accept either pandas Series or xarray DataArray
    if isinstance(monthly_means, xr.DataArray):
        # convert to pandas Series with month index 1..12
        monthly_means = pd.Series(monthly_means.values.flatten(), index=np.arange(1, 13))

    ts = monthly_means.values
    beta, r2, N = _harmonic_regression(ts)

    if beta is None or r2 is None or r2 < MIN_R2_THRESHOLD:
        # Fallback to raw monthly means (ensure index 1..12)
        logger.info(f"Using raw monthly means (R²={r2:.3f if r2 else 'N/A'} < {MIN_R2_THRESHOLD})")
        return monthly_means.groupby(monthly_means.index).mean().reindex(np.arange(1, 13)), r2

    # Use ALL harmonic coefficients for better fit
    # Construct fitted monthly climatology
    f = 1 / 12
    t = np.arange(N)

    fitted = (
        beta[0]                              # Intercept
        + beta[1]*np.sin(2*np.pi*f*t)       # Annual cycle
        + beta[2]*np.cos(2*np.pi*f*t)
        + beta[3]*np.sin(4*np.pi*f*t)       # Semi-annual cycle
        + beta[4]*np.cos(4*np.pi*f*t)
        + beta[5]*np.sin(6*np.pi*f*t)       # Tertiary cycle (4-month)
        + beta[6]*np.cos(6*np.pi*f*t)
        + beta[7]*np.sin(8*np.pi*f*t)       # Quarterly cycle (3-month)
        + beta[8]*np.cos(8*np.pi*f*t)
    )

    # Convert to monthly climatology by averaging fitted values by calendar month
    idx = monthly_means.index
    fitted_monthly = pd.Series(fitted[:len(idx)], index=idx).groupby(idx).mean()
    fitted_monthly = fitted_monthly.reindex(np.arange(1, 13))
    
    logger.debug(f"Harmonic regression successful (R²={r2:.3f})")
    
    return fitted_monthly, r2


def process_climatology(ds, param, sensor_range, **kwargs):
    """
    Process climatology test for a parameter.
    
    Computes monthly mean and standard deviation, applies harmonic regression
    to smooth the climatology, and calculates upper/lower bounds.
    
    Args:
        ds: xarray Dataset containing the data
        param: Parameter name to process
        sensor_range: [min, max] sensor limits
        **kwargs: Additional arguments (site, node, sensor, stream, was_decimated, original_points, final_points)
    
    Returns:
        Dictionary with 'lower', 'upper' (lists of 12 monthly values), and 'notes'
    """
    logger.info(f"[CLIM] Processing climatology for {param}")
    
    site   = kwargs.get('site')
    node   = kwargs.get('node')
    sensor = kwargs.get('sensor')
    stream = kwargs.get('stream')
    was_decimated = kwargs.get('was_decimated', False)
    original_points = kwargs.get('original_points', None)
    final_points = kwargs.get('final_points', None)

    results = {}
    da = ds[param]

    # Lazy masking: does not load data
    # Remove out-of-range and NaN values
    da = da.where(
        (da > sensor_range[0]) &
        (da < sensor_range[1]) &
        (~np.isnan(da))
    )
    
    # Check if any data remains after filtering
    if da.count().compute() == 0:
        logger.error(f"[CLIM] No valid data for {param} after filtering")
        # Return empty climatology with NaN
        return {
            "lower": [np.nan] * 12,
            "upper": [np.nan] * 12,
            "notes": "No valid data after filtering"
        }
    
    # Compute monthly statistics
    monthly_mean = _compute_monthly_means(da)
    monthly_std  = _compute_monthly_std(da)

    # Compute harmonic fit + R²
    fitted_monthly, r2 = _compute_monthly_fit(monthly_mean)

    # Construct note based on R²
    if r2 is None or r2 < MIN_R2_THRESHOLD:
        note = f"Using raw monthly means (low variance explained: R²={0.0 if r2 is None else r2:.3f})"
    else:
        note = f"Harmonic regression variance explained: R²={r2:.3f}"

    # Compute numeric monthly mean/std (small arrays)
    mm = monthly_mean.compute().values
    ms = monthly_std.compute().values
    
    # handling of incorrect array sizes
    if len(mm) != 12:
        logger.error(f"[CLIM] Monthly mean has {len(mm)} values instead of 12")
        # Pad with NaN instead of using np.resize (which cycles values)
        mm_padded = np.full(12, np.nan)
        mm_padded[:len(mm)] = mm
        mm = mm_padded

    if len(ms) != 12:
        logger.error(f"[CLIM] Monthly std has {len(ms)} values instead of 12")
        # Pad with NaN instead of using np.resize
        ms_padded = np.full(12, np.nan)
        ms_padded[:len(ms)] = ms
        ms = ms_padded
    
    # Check for any NaN or zero std (would cause invalid bounds)
    if np.any(np.isnan(ms)) or np.any(ms <= 0):
        logger.warning(f"[CLIM] Some months have NaN or zero standard deviation")
        # Replace NaN or zero with a small value based on overall std
        valid_std = ms[~np.isnan(ms) & (ms > 0)]
        if len(valid_std) > 0:
            replacement_std = np.median(valid_std)
        else:
            # Use 1% of sensor range as fallback
            replacement_std = (sensor_range[1] - sensor_range[0]) * 0.01
        ms = np.where((np.isnan(ms) | (ms <= 0)), replacement_std, ms)
        note += f" (Some std values replaced with {replacement_std:.2f})"

    # Calculate upper and lower bounds
    upper = mm + N_STD_DEVIATIONS * ms
    lower = mm - N_STD_DEVIATIONS * ms
    
    # Enforce sensor limits
    upper = np.minimum(upper, sensor_range[1])
    lower = np.maximum(lower, sensor_range[0])
    
    # Add decimation info to notes if data was decimated
    if was_decimated and original_points and final_points:
        note += f" Analysis performed on decimated dataset using LTTB (Largest Triangle Three Buckets) algorithm: {original_points:,} points reduced to {final_points:,} points."

    results = {
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "notes": note,
    }
    
    logger.info(f"[CLIM] Completed climatology for {param}")
    
    return results