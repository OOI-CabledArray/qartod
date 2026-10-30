#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from loguru import logger
import xarray as xr

# Statistical constants
MIN_R2_THRESHOLD = 0.15  # Minimum variance explained to use the harmonic fit
N_HARMONICS = 4          # Harmonic cycles for both the R² and the fitted curve (1=annual, 2=+semi-annual, ...)
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


def _cyclic_interp(g):
    """
    Fill NaN months (1..12) by linear interpolation that wraps December→January.
    """
    g = g.reindex(month=np.arange(1, 13)).compute()
    if not g.isnull().any():
        return g
    # Pad [Nov, Dec] + Jan..Dec + [Jan, Feb] on a sequential index, interpolate, keep the middle
    padded = xr.concat([g.isel(month=[10, 11]), g, g.isel(month=[0, 1])], dim="month")
    padded = padded.assign_coords(month=np.arange(16)).interpolate_na(dim="month", method="linear")
    return padded.isel(month=slice(2, 14)).assign_coords(month=np.arange(1, 13))


def _monthly_stat(da, stat):
    """
    Monthly statistic ('mean' or 'std') grouped by calendar month, all years pooled,
    with missing months filled cyclically. Returns a 12-value DataArray on 'month'.
    """
    g = getattr(_auto_chunk_time(da).groupby("time.month"), stat)(dim="time")
    if "depth" in g.dims:
        g = g.mean(dim="depth")
    return _cyclic_interp(g)


def _harmonic_design(t, n_harmonics=N_HARMONICS):
    """
    Design matrix: intercept + sin/cos pairs for periods of 12/k months, k = 1..n_harmonics.
    """
    w = 2 * np.pi * t / 12
    cols = [np.ones_like(w)]
    for k in range(1, n_harmonics + 1):
        cols += [np.sin(k * w), np.cos(k * w)]
    return np.column_stack(cols)


def _compute_monthly_fit(da):
    """
    Harmonic climatology fit, following OOI ooi-data-explorations (qartod/climatology.py),
    but using the same N_HARMONICS for both the R² and the fitted curve.

    1. Resample the full record to one mean per month (e.g. 120 values for 10 years).
    2. Least-squares fit of the N_HARMONICS-cycle model to that series.
    3. Average the fitted series by calendar month -> 12 values.

    Returns:
        fitted_monthly: 12-value DataArray on 'month', or None if the fit can't be made
        r2: variance explained (None if the fit can't be made)
    """
    ts = _auto_chunk_time(da).resample(time="1MS").mean()
    if "depth" in ts.dims:
        ts = ts.mean(dim="depth")
    ts = ts.compute()

    t = np.arange(ts.sizes["time"])
    valid = ts.notnull().values
    X = _harmonic_design(t[valid])
    y = ts.values[valid]

    if len(y) <= X.shape[1]:
        logger.warning(f"Insufficient monthly values for harmonic regression: {len(y)} (need >{X.shape[1]})")
        return None, None

    beta, resid, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    tss = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - resid[0] / tss) if (resid.size and tss > 0) else 0.0

    fitted = xr.DataArray(_harmonic_design(t) @ beta, dims="time", coords={"time": ts.time})
    fitted_monthly = fitted.groupby("time.month").mean().reindex(month=np.arange(1, 13))

    logger.debug(f"Harmonic regression ({N_HARMONICS} cycles): R²={r2:.3f}, {len(y)} monthly values")
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
        **kwargs: Additional arguments (site, node, sensor, stream, was_decimated, original_points, final_points, dec_method)
    
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
    dec_method = kwargs.get('dec_method', 'unknown method')

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
    
    # Center: harmonic fit if it explains enough variance, otherwise monthly means
    fitted_monthly, r2 = _compute_monthly_fit(da)
    if r2 is not None and r2 >= MIN_R2_THRESHOLD:
        center = _cyclic_interp(fitted_monthly)
        note = (f"Climatology based on a {N_HARMONICS}-cycle harmonic fit to the monthly means "
                f"± {N_STD_DEVIATIONS}x the monthly standard deviations. "
                f"Variance explained: R²={r2:.3f}.")
    else:
        center = _monthly_stat(da, "mean")
        note = (f"Climatology based on the monthly means ± {N_STD_DEVIATIONS}x the monthly standard deviations "
                f"(harmonic fit not used: R²={'N/A' if r2 is None else f'{r2:.3f}'} < {MIN_R2_THRESHOLD}).")

    mm = center.values
    ms = _monthly_stat(da, "std").values

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
        note += f" Analysis performed on decimated dataset using {dec_method}: {original_points:,} points reduced to {final_points:,} points."

    results = {
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "notes": note,
    }
    
    logger.info(f"[CLIM] Completed climatology for {param}")
    
    return results