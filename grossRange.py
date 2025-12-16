#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gross Range Test Implementation for QARTOD

This module calculates user-defined sensor ranges using statistical analysis.
For normally distributed data, uses mean ± 3σ. For non-normal data, uses
percentile-based ranges.
"""

import numpy as np
import pandas as pd
import dask.array as da_np
from dask.array import stats as da_stats
from dask.diagnostics import ProgressBar
from loguru import logger
from typing import Dict, List, Union

# Statistical constants
SKEWNESS_THRESHOLD = 1.0          # Absolute skewness threshold for normality
EXCESS_KURTOSIS_LOWER = -2.0      # Lower bound for excess kurtosis
EXCESS_KURTOSIS_UPPER = 2.0       # Upper bound for excess kurtosis
NORMAL_STD_MULTIPLIER = 3.0       # Number of std deviations for normal distribution
PERCENTILE_LOWER = 0.15           # Lower percentile for non-normal (99.7% coverage)
PERCENTILE_UPPER = 99.85          # Upper percentile for non-normal
DEFAULT_CHUNK_SIZE = 50000        # Default chunk size for Dask


def _ensure_chunked(ds, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Ensure dataset is chunked for Dask operations.
    
    Args:
        ds: xarray Dataset
        chunk_size: Chunk size for time dimension
    
    Returns:
        Chunked dataset
    """
    if not any(getattr(v.data, "chunks", None) for v in ds.data_vars.values()):
        if "time" in ds.dims:
            ds = ds.chunk({"time": chunk_size})
        else:
            ds = ds.chunk({dim: chunk_size for dim in ds.dims})
    return ds


def _remove_nans_from_dask_array(darr):
    """
    Remove NaN values from a Dask array.
    
    Args:
        darr: Dask array potentially containing NaNs
    
    Returns:
        Dask array with NaNs removed (1D, flattened)
    """
    # Flatten array and create mask for non-NaN values
    darr_flat = darr.flatten()
    mask = ~da_np.isnan(darr_flat)
    
    # Use compress to keep only non-NaN values
    # Note: This returns a 1D array
    darr_clean = darr_flat[mask]
    
    return darr_clean


def _check_normality(darr):
    """
    Test if data is approximately normally distributed.
    
    Uses skewness and excess kurtosis to determine normality:
    - Skewness should be close to 0 (symmetric)
    - Excess kurtosis should be close to 0 (normal tail behavior)
    
    Args:
        darr: Dask array of data (may contain NaNs)
    
    Returns:
        tuple: (is_normal, skew, excess_kurt)
    """
    logger.debug("Computing normality statistics...")
    
    # Remove NaNs before computing statistics
    darr_clean = _remove_nans_from_dask_array(darr)
    
    with ProgressBar():
        # Dask stats functions don't support nan_policy, so we work with clean data
        skew = da_stats.skew(darr_clean).compute()
        kurt = da_stats.kurtosis(darr_clean).compute()
    
    # Ensure we have scalar values, not NaN
    if np.isnan(skew) or np.isnan(kurt):
        logger.warning("NaN detected in skewness or kurtosis - assuming non-normal")
        return False, skew, np.nan
    
    # Convert to excess kurtosis (normal distribution has excess kurtosis = 0)
    excess_kurt = kurt - 3.0
    
    # Check both skewness and excess kurtosis
    is_normal = (
        abs(skew) < SKEWNESS_THRESHOLD and 
        EXCESS_KURTOSIS_LOWER < excess_kurt < EXCESS_KURTOSIS_UPPER
    )
    
    logger.debug(f"Skewness: {skew:.3f}, Excess Kurtosis: {excess_kurt:.3f}, Normal: {is_normal}")
    
    return is_normal, skew, excess_kurt


def _compute_percentile_range(darr):
    """
    Compute range using percentiles for non-normal data.
    
    Args:
        darr: Dask array of data (may contain NaNs)
    
    Returns:
        tuple: (lower, upper)
    """
    logger.info("Using percentile-based range (non-normal distribution)")
    
    # Remove NaNs before computing percentiles
    darr_clean = _remove_nans_from_dask_array(darr)
    
    with ProgressBar():
        # Use regular percentile on clean data
        lower = da_np.percentile(darr_clean, PERCENTILE_LOWER).compute()
        upper = da_np.percentile(darr_clean, PERCENTILE_UPPER).compute()
    
    # Ensure values are scalars and not NaN
    lower = float(lower)
    upper = float(upper)
    
    if np.isnan(lower) or np.isnan(upper):
        logger.error("NaN values in percentile calculation")
        raise ValueError("Percentile calculation resulted in NaN")
    
    return lower, upper


def _compute_normal_range(da):
    """
    Compute range using mean ± 3σ for normal data.
    
    Args:
        da: xarray DataArray (may contain NaNs)
    
    Returns:
        tuple: (lower, upper, mean, std)
    """
    logger.info("Using mean ± 3σ range (normal distribution)")
    
    with ProgressBar():
        # xarray's mean and std handle NaNs by default with skipna=True
        mu = da.mean(skipna=True).compute()
        sd = da.std(skipna=True).compute()
    
    # Ensure values are scalars
    mu = float(mu)
    sd = float(sd)
    
    # Check for NaN results
    if np.isnan(mu) or np.isnan(sd):
        logger.error("NaN values in mean/std calculation")
        raise ValueError("Mean/std calculation resulted in NaN")
    
    # Handle edge cases
    if sd == 0:
        logger.warning("Standard deviation is zero - data is constant")
        return mu, mu, mu, sd
    
    lower = mu - NORMAL_STD_MULTIPLIER * sd
    upper = mu + NORMAL_STD_MULTIPLIER * sd
    
    return lower, upper, mu, sd


def process_gross_range(
    ds, 
    param: str, 
    sensor_range: List[float], 
    **kwargs
) -> Dict[str, Union[float, str]]:
    """
    Memory-safe gross range calculation using Dask-backed xarray.
    
    Calculates user-defined quality control ranges based on statistical analysis:
    - For normally distributed data: mean ± 3 standard deviations
    - For non-normal data: 0.15th to 99.85th percentiles (99.7% coverage)
    
    The calculated ranges are constrained to stay within vendor sensor limits.
    
    Args:
        ds: xarray Dataset containing the parameter data
        param: Name of parameter to analyze
        sensor_range: [min, max] vendor sensor limits
        **kwargs: Additional context (site, node, sensor, stream, was_decimated, original_points, final_points)
    
    Returns:
        Dictionary with keys:
            - 'lower': Lower bound of user range
            - 'upper': Upper bound of user range
            - 'notes': Description of method used
    
    Raises:
        ValueError: If parameter not found or sensor_range invalid
    """
    site   = kwargs.get('site', 'unknown')
    node   = kwargs.get('node', 'unknown')
    sensor = kwargs.get('sensor', 'unknown')
    stream = kwargs.get('stream', 'unknown')
    was_decimated = kwargs.get('was_decimated', False)
    original_points = kwargs.get('original_points', None)
    final_points = kwargs.get('final_points', None)
    
    logger.info(f"[GROSS_RANGE] Processing {param} for {site}-{node}-{sensor}")
    
    # Validate inputs
    if param not in ds:
        raise ValueError(f"Parameter '{param}' not found in dataset")
    
    if len(sensor_range) != 2 or sensor_range[0] >= sensor_range[1]:
        raise ValueError(f"Invalid sensor_range: {sensor_range}. Must be [min, max] with min < max")
    
    results = {}
    
    # Ensure dataset is chunked
    ds = _ensure_chunked(ds)
    da = ds[param]
    
    # Create boolean mask for valid data
    valid_mask = (
        (da > sensor_range[0]) &
        (da < sensor_range[1]) &
        (~np.isnan(da))
    )
    
    # Check if any data remains after filtering
    data_count = valid_mask.sum().compute()
    if data_count == 0:
        logger.error(f"[GROSS_RANGE] No valid data for {param} after filtering")
        return {
            "lower": sensor_range[0],
            "upper": sensor_range[1],
            "notes": "No valid data - using sensor limits"
        }
    
    logger.info(f"[GROSS_RANGE] Processing {data_count} valid data points")
    
    # Apply mask - this creates an array with NaNs where mask is False
    da_filtered = da.where(valid_mask)
    
    # Get underlying dask array
    darr = da_filtered.data
    
    try:
        # Check normality (will handle NaNs internally)
        is_normal, skew, excess_kurt = _check_normality(darr)
        
        # Compute range based on distribution type
        if not is_normal:
            lower, upper = _compute_percentile_range(darr)
            notes = (
                f"Non-normal distribution (skew={skew:.3f}, excess_kurt={excess_kurt:.3f}): "
                f"using {PERCENTILE_LOWER}th and {PERCENTILE_UPPER}th percentiles "
                f"(99.7% coverage)."
            )
        else:
            lower, upper, mu, sd = _compute_normal_range(da_filtered)
            notes = (
                f"Normal distribution (skew={skew:.3f}, excess_kurt={excess_kurt:.3f}): "
                f"using mean ± {NORMAL_STD_MULTIPLIER} standard deviations "
                f"(μ={mu:.2f}, σ={sd:.2f})."
            )
        
    except ValueError as e:
        logger.error(f"[GROSS_RANGE] Error in statistical calculation: {e}")
        return {
            "lower": sensor_range[0],
            "upper": sensor_range[1],
            "notes": f"Statistical calculation failed: {str(e)} - using sensor limits"
        }
    
    # Enforce vendor sensor limits
    original_lower, original_upper = lower, upper
    lower = max(lower, sensor_range[0])
    upper = min(upper, sensor_range[1])
    
    if lower != original_lower or upper != original_upper:
        logger.info(f"[GROSS_RANGE] Range constrained by sensor limits: "
                   f"[{original_lower:.2f}, {original_upper:.2f}] → [{lower:.2f}, {upper:.2f}]")
        notes += f" Range constrained to sensor limits [{sensor_range[0]}, {sensor_range[1]}]."
    
    # Add decimation info to notes if data was decimated
    if was_decimated and original_points and final_points:
        notes += f" Analysis performed on decimated dataset using LTTB (Largest Triangle Three Buckets) algorithm: {original_points:,} points reduced to {final_points:,} points."
    
    results = {
        "lower": lower,
        "upper": upper,
        "notes": notes
    }
    
    logger.info(f"[GROSS_RANGE] Computed range: [{lower:.2f}, {upper:.2f}]")
    
    return results