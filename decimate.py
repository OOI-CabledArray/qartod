#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LTTB (Largest Triangle Three Buckets) Decimation

This module implements the LTTB-M (Modified) algorithm for time series downsampling.
The modification preserves regular time intervals while using the LTTB algorithm
for selecting representative Y-values.

Reference: Sveinn Steinarsson (2013), "Downsampling Time Series for Visual Representation"
"""

import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger
import xarray as xr
from functools import reduce


def _validate_threshold(data_length: int, threshold: int) -> None:
    """
    Validate decimation threshold.
    
    Args:
        data_length: Length of input data
        threshold: Desired output length
    
    Raises:
        ValueError: If threshold is invalid
    """
    if threshold < 2:
        raise ValueError(f"Threshold must be >= 2, got {threshold}")
    
    if threshold > data_length:
        raise ValueError(
            f"Threshold ({threshold}) cannot exceed data length ({data_length})"
        )
    
    if threshold == data_length:
        logger.warning("Threshold equals data length - no decimation will occur")


def _areas_of_triangles(a: np.ndarray, bs: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Calculate areas of triangles from vertex coordinates.

    Uses the standard triangle area formula with cross-product.
    Vectorized computation for all points in bin bs.

    Args:
        a: Previous point [x, y]
        bs: Current bin of points, shape (n, 2)
        c: Next bin centroid [x, y]
    
    Returns:
        Array of triangle areas, shape (n,)
    """
    # Standard triangle area: 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    bs_minus_a = bs - a
    a_minus_bs = a - bs
    
    areas = 0.5 * np.abs(
        (a[0] - c[0]) * (bs_minus_a[:, 1]) - (a_minus_bs[:, 0]) * (c[1] - a[1])
    )
    
    return areas


def _largest_triangle_three_buckets(data: np.ndarray, threshold: int) -> np.ndarray:
    """
    LTTB-M (Modified) downsampling algorithm.
    
    Divides data into bins and selects one representative point per bin.
    The selected point maximizes the triangle area formed with adjacent points.
    
    Modification: Uses middle point's X-coordinate to preserve time spacing,
    but max-area point's Y-coordinate for best representation.
    
    Args:
        data: Input data, shape (n, 2) with columns [time, value]
        threshold: Number of points in output (must be >= 2 and <= n)
    
    Returns:
        Decimated data, shape (threshold, 2)
    
    Raises:
        ValueError: If threshold is invalid
    """
    n = len(data)
    _validate_threshold(n, threshold)
    
    # Special case: no decimation needed
    if threshold == n:
        return data
    
    n_bins = threshold - 2  # Exclude first and last points
    
    # Prepare output array - first and last points are preserved
    out = np.zeros((threshold, 2))
    out[0] = data[0]
    out[-1] = data[-1]
    
    # Split middle data into bins
    data_bins = np.array_split(data[1:-1], n_bins)
    
    # Process each bin
    for i in range(len(data_bins)):
        this_bin = data_bins[i]
        
        # Determine next bin for centroid calculation
        if i < n_bins - 1:
            next_bin = data_bins[i + 1]
        else:
            # Last bin uses final point
            next_bin = data[-1:, :]
        
        # Get reference points
        a = out[i]                          # Previous selected point
        c = np.mean(next_bin, axis=0)      # Centroid of next bin
        
        # Calculate triangle areas for all points in current bin
        areas = _areas_of_triangles(a, this_bin, c)
        
        # LTTB-M modification: preserve time spacing
        middle_idx = len(this_bin) // 2
        middle = this_bin[middle_idx]
        
        # Use max-area point's Y value but middle point's X (time)
        max_area_idx = np.argmax(areas)
        point = this_bin[max_area_idx]
        
        out[i + 1] = np.array([middle[0], point[1]])
    
    return out


class LttbException(Exception):
    """Exception raised when LTTB decimation fails."""
    pass


def _perform_decimation(da: xr.DataArray, threshold: int) -> pd.DataFrame:
    """
    Perform LTTB decimation on a single DataArray.
    
    Args:
        da: xarray DataArray with time coordinate
        threshold: Target number of points
    
    Returns:
        DataFrame with decimated data
    
    Raises:
        LttbException: If decimation fails
    """
    logger.debug(f"Decimating {da.name}: {len(da)}  {threshold} points")
    
    # Convert time to numeric (preserve as datetime64 for now)
    time_numeric = da.time.values.astype('int64')  # nanoseconds since epoch
    values = da.values
    
    # Stack into [time, value] pairs
    data = np.column_stack([time_numeric, values])
    
    # Remove any rows with NaN values
    valid_mask = ~np.isnan(data).any(axis=1)
    data = data[valid_mask]
    
    if len(data) == 0:
        logger.warning(f"No valid data for {da.name} after removing NaN")
        return pd.DataFrame(columns=['time', da.name])
    
    # Adjust threshold if less data than expected
    actual_threshold = min(threshold, len(data))
    if actual_threshold != threshold:
        logger.info(f"Adjusted threshold for {da.name}: {threshold}  {actual_threshold}")
    
    try:
        decdata = _largest_triangle_three_buckets(data, actual_threshold)
    except Exception as e:
        logger.error(f"LTTB failed for {da.name}: {e}")
        raise LttbException(f"Decimation failed for {da.name}: {e}") from e
    
    # Convert back to DataFrame with proper datetime
    df = pd.DataFrame(decdata, columns=['time', da.name])
    df['time'] = pd.to_datetime(df['time'].astype('int64'), unit='ns')
    
    return df


def downsample(
    raw_ds: xr.Dataset,
    threshold: int,
) -> pd.DataFrame:
    """
    Perform LTTB downsampling on all variables in a dataset.
    
    Each variable is decimated independently to the specified threshold.
    Results are merged on the time coordinate.
    
    Args:
        raw_ds: Input xarray Dataset
        threshold: Target number of points for each variable
    
    Returns:
        DataFrame with decimated data for all variables
    
    Raises:
        ValueError: If threshold is invalid
        LttbException: If decimation fails for any variable
    
    Example:
        >>> ds = xr.open_zarr('data.zarr')
        >>> decimated_df = downsample(ds, threshold=5000)
        >>> print(len(decimated_df))  # ~5000 rows
    """
    if threshold < 2:
        raise ValueError(f"Threshold must be >= 2, got {threshold}")
    
    logger.info(f"Starting decimation with threshold={threshold}")
    
    # Get list of data variables (exclude coordinates)
    data_vars = [var for var in raw_ds.data_vars]
    
    if not data_vars:
        logger.warning("No data variables found in dataset")
        return pd.DataFrame()
    
    logger.info(f"Processing {len(data_vars)} variables: {data_vars}")
    
    # Decimate each variable
    df_list = []
    for var_name in data_vars:
        try:
            da = raw_ds[var_name]
            logger.info(f"Decimating {var_name}...")
            decdf = _perform_decimation(da, threshold)
            
            if len(decdf) > 0:
                df_list.append(decdf)
            else:
                logger.warning(f"Skipping {var_name} - no data after decimation")
        
        except LttbException as e:
            logger.error(f"Failed to decimate {var_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error decimating {var_name}: {e}")
            raise LttbException(f"Unexpected error decimating {var_name}") from e
    
    if not df_list:
        logger.error("No variables successfully decimated")
        return pd.DataFrame()
    
    logger.info("Merging decimated variables...")
    
    # Merge all DataFrames on time
    final_df = reduce(
        lambda left, right: pd.merge(left, right, on='time', how='outer'),
        df_list
    )
    
    # Sort by time
    final_df = final_df.sort_values('time').reset_index(drop=True)
    
    logger.info(f"Decimation complete: {len(final_df)} points")
    
    return final_df
