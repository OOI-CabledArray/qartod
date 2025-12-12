#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QARTOD Processing Utilities

Core utilities for loading, filtering, and processing oceanographic data
for QARTOD quality control analysis.

Dependencies:
    - AWS credentials (AWS_KEY, AWS_SECRET) must be set in environment
    - S3 bucket 'ooi-data' must be accessible
    - Zarr data format expected
"""

import argparse
import numpy as np
import pandas as pd
import sys
import decimate
from loguru import logger
from datetime import datetime
import dateutil.parser as parser
import json
import os
import pytz
import s3fs
import xarray as xr
from functools import lru_cache
from typing import Optional, Dict, List

# Constants
S3_BUCKET = 'ooi-data'
START_DATE = '2014-01-01T00:00:00'  # Historical data start
QARTOD_FAIL_FLAG = 4
QARTOD_SUSPECT_FLAG = 3

# Time conversion constant
NS_TO_HOURS = 1e9 * 60 * 60


def get_s3_kwargs() -> Dict[str, str]:
    """
    Get S3 credentials from environment variables.
    
    Returns:
        Dictionary with 'key' and 'secret' for S3 access
    
    Raises:
        ValueError: If credentials are not set
    """
    aws_key = os.environ.get("AWS_KEY")
    aws_secret = os.environ.get("AWS_SECRET")
    
    if not aws_key or not aws_secret:
        raise ValueError(
            "AWS credentials not found. Please set AWS_KEY and AWS_SECRET "
            "environment variables."
        )
    
    return {'key': aws_key, 'secret': aws_secret}


def inputs(argv=None):
    """
    Parse command line arguments.
    
    Args:
        argv: Command line arguments (defaults to sys.argv)
    
    Returns:
        Parsed arguments namespace
    """
    if argv is None:
        argv = sys.argv[1:]

    inputParser = argparse.ArgumentParser(
        description="Download and process instrument data to generate QARTOD lookup tables"
    )

    inputParser.add_argument(
        "-rd", "--refDes", 
        dest="refDes", 
        type=str, 
        required=True,
        help="Reference designator (e.g., CE01ISSM-MFD35-02-PRESFA000)"
    )
    inputParser.add_argument(
        "-co", "--cut_off", 
        dest="cut_off", 
        type=str, 
        required=False,
        help="Cut-off date for data (ISO format, e.g., 2023-12-31)"
    )
    inputParser.add_argument(
        "-d", "--decThreshold", 
        dest="decThreshold", 
        type=str, 
        required=True,
        help="Decimation threshold (target number of points, 0 for no decimation)"
    )
    inputParser.add_argument(
        "-v", "--userVars", 
        dest="userVars", 
        type=str, 
        required=True,
        help="Variables to process ('all' or specific variable name)"
    )

    args = inputParser.parse_args(argv)
    
    # Validate decimation threshold
    try:
        dec_threshold = int(args.decThreshold)
        if dec_threshold < 0:
            inputParser.error("decThreshold must be non-negative")
    except ValueError:
        inputParser.error("decThreshold must be an integer")
    
    return args


@lru_cache(maxsize=1)
def loadPID() -> Dict:
    """
    Load parameter ID dictionary from OOI preload database.
    
    Cached to avoid repeated network requests.
    
    Returns:
        Dictionary mapping PID to parameter info
    """
    logger.info("Loading parameter ID dictionary from GitHub")
    
    try:
        pid_url = (
            'https://raw.githubusercontent.com/oceanobservatories/'
            'preload-database/refs/heads/master/csv/ParameterDefs.csv'
        )
        pid_dict = pd.read_csv(
            pid_url, 
            usecols=['netcdf_name', 'id']
        ).set_index('id').T.to_dict()
        
        logger.info(f"Loaded {len(pid_dict)} parameter definitions")
        return pid_dict
    
    except Exception as e:
        logger.error(f"Failed to load PID dictionary: {e}")
        raise


def loadAnnotations(site: str) -> pd.DataFrame:
    """
    Load annotations from S3 for a given site.
    
    Args:
        site: Site designator (e.g., 'CE01ISSM')
    
    Returns:
        DataFrame of annotations (empty if not found)
    """
    logger.info(f"Loading annotations for {site}")
    
    try:
        fs = s3fs.S3FileSystem(**get_s3_kwargs())
        anno_file = f'{S3_BUCKET}/annotations/{site}.json'
        
        if not fs.exists(anno_file):
            logger.warning(f"No annotation file found for {site}")
            return pd.DataFrame()
        
        with fs.open(anno_file) as f:
            anno = json.load(f)
        
        anno_df = pd.DataFrame(anno)
        logger.info(f"Loaded {len(anno_df)} annotations for {site}")
        return anno_df
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in annotation file for {site}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load annotations for {site}: {e}")
        return pd.DataFrame()


def loadData(zarr_dir: str) -> xr.Dataset:
    """
    Load Zarr dataset from S3.
    
    Args:
        zarr_dir: Path to Zarr directory in S3 (relative to bucket)
    
    Returns:
        xarray Dataset
    
    Raises:
        Exception: If data cannot be loaded
    """
    logger.info(f"Loading data from {zarr_dir}")
    
    try:
        fs = s3fs.S3FileSystem(**get_s3_kwargs())
        zarr_store = fs.get_mapper(f'{S3_BUCKET}/{zarr_dir}')
        ds = xr.open_zarr(zarr_store, consolidated=True)
        
        logger.info(f"Loaded dataset with {len(ds.data_vars)} variables, "
                   f"{len(ds.time)} time points")
        return ds
    
    except Exception as e:
        logger.error(f"Failed to load data from {zarr_dir}: {e}")
        raise


def decimateData(xs: xr.Dataset, decimation_threshold: int) -> xr.Dataset:
    """
    Decimate dataset using LTTB algorithm.
    
    Args:
        xs: Input dataset
        decimation_threshold: Target number of points
    
    Returns:
        Decimated dataset
    """
    logger.info(f"Decimating data: {len(xs.time)}  {decimation_threshold} points")
    
    # Remove NaN time values
    xs = xs.where(~np.isnan(xs['time']), drop=True)
    
    # Perform decimation
    dec_data_df = decimate.downsample(xs, decimation_threshold)
    
    # Convert back to xarray Dataset
    dec_data = xr.Dataset.from_dataframe(dec_data_df)
    dec_data = dec_data.swap_dims({'index': 'time'})
    dec_data = dec_data.reset_coords()
    
    # Preserve original attributes
    dec_data.attrs = xs.attrs
    
    logger.info(f"Decimation complete: {len(dec_data.time)} points")
    
    return dec_data


def _convert_time(ms: Optional[float]) -> Optional[datetime]:
    """
    Convert milliseconds since epoch to datetime.
    
    Args:
        ms: Milliseconds since epoch (or None)
    
    Returns:
        datetime object (or None if input is None/NaN)
    """
    if ms is None or (isinstance(ms, float) and np.isnan(ms)):
        return None
    
    try:
        return datetime.utcfromtimestamp(float(ms) / 1000)
    except (ValueError, TypeError, OverflowError) as e:
        logger.warning(f"Invalid timestamp: {ms} ({e})")
        return None


def add_annotation_qc_flags(ds: xr.Dataset, annotations: pd.DataFrame, 
                           pid_dict: Dict) -> xr.Dataset:
    """
    Add annotation QC flags to dataset as data variables.
    
    Args:
        ds: xarray Dataset
        annotations: DataFrame of annotations
        pid_dict: Parameter ID dictionary
    
    Returns:
        Dataset with annotation QC flags added
    """
    # Convert to DataFrame if needed
    if isinstance(annotations, (list, dict)):
        annotations = pd.DataFrame(annotations)
    
    if annotations.empty:
        logger.info("No annotations to add")
        return ds
    
    # Convert flags to QARTOD codes
    codes = {
        None: 0,
        'pass': 1,
        'not_evaluated': 2,
        'suspect': 3,
        'fail': 4,
        'not_operational': 0,
        'not_available': 0,
        'pending_ingest': 0
    }
    annotations['qcFlag'] = annotations['qcFlag'].map(codes).fillna(0).astype('int32')
    
    # Filter for relevant stream
    stream = ds.attrs.get("stream")
    if stream:
        stream_mask = annotations["stream"].apply(
            lambda x: x == stream or x is None if pd.notna(x) else True
        )
        annotations = annotations[stream_mask]
    
    # Explode parameters (one row per parameter per annotation)
    annotations = annotations.explode(column="parameters")
    
    # Map PIDs to parameter names
    stream_annos = {}
    for pid in annotations["parameters"].unique():
        if pd.isna(pid):
            param_name = "rollup"
        else:
            pid_key = f'PD{int(pid)}'
            pid_info = pid_dict.get(pid_key)
            if pid_info and 'netcdf_name' in pid_info:
                param_name = pid_info['netcdf_name']
            else:
                logger.warning(f"Unknown PID: {pid_key}")
                param_name = f"unknown_param_{int(pid)}"
        stream_annos[param_name] = pid
    
    # Create flags for each parameter
    flags_dict = {}
    for param_name, pid in stream_annos.items():
        # Get annotations for this parameter
        if pd.isna(pid):
            param_annos = annotations[annotations["parameters"].isna()]
        else:
            param_annos = annotations[annotations["parameters"] == pid]
        
        param_annos = param_annos.sort_values(by="qcFlag")
        
        # Initialize flags array
        flags = pd.Series(
            np.zeros(len(ds.time), dtype='int32'), 
            index=ds.time.values
        )
        
        # Apply annotations
        for idx in param_annos.index:
            begin_dt = _convert_time(param_annos["beginDT"].loc[idx])
            end_dt = _convert_time(param_annos["endDT"].loc[idx])
            qc_flag = param_annos["qcFlag"].loc[idx]
            
            if begin_dt is None:
                continue
            
            if end_dt is None:
                end_dt = datetime.now()
            
            # Set flags for time range
            mask = (flags.index > begin_dt) & (flags.index < end_dt)
            flags[mask] = qc_flag
        
        flags_dict[param_name] = flags
    
    # Add flags to dataset
    for param_name, flags in flags_dict.items():
        var_name = f"{param_name.lower()}_annotations_qc_results"
        ds[var_name] = xr.DataArray(flags.values, dims="time")
        logger.debug(f"Added annotation flags: {var_name}")
    
    return ds


def filterData(data: xr.Dataset, node: str, site: str, sensor: str, 
               param: str, cut_off: Optional[str], annotations: pd.DataFrame, 
               pid_dict: Dict) -> xr.Dataset:
    """
    Filter dataset based on QC flags and annotations.
    
    WARNING: Modifies data in-place by setting failed values to NaN.
    
    Args:
        data: xarray Dataset
        node: Node designator
        site: Site designator
        sensor: Sensor designator
        param: Parameter name
        cut_off: Optional cut-off date (ISO format)
        annotations: DataFrame of annotations
        pid_dict: Parameter ID dictionary
    
    Returns:
        Filtered dataset
    """
    logger.info(f"Filtering data for {param}")
    
    # Validate parameter exists
    if param not in data.data_vars:
        logger.warning(f"Parameter {param} not found in dataset")
        return data
    
    # Prepare annotations
    if not annotations.empty and '@class' in annotations.columns:
        annotations = annotations.drop(columns=['@class'])
    
    if not annotations.empty:
        annotations['beginDate'] = pd.to_datetime(
            annotations.beginDT, unit='ms', errors='coerce'
        ).dt.strftime('%Y-%m-%dT%H:%M:%S')
        annotations['endDate'] = pd.to_datetime(
            annotations.endDT, unit='ms', errors='coerce'
        ).dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Add annotation flags
    data = add_annotation_qc_flags(data, annotations, pid_dict)
    
    # Apply QC summary flag
    qc_summary_var = f'{param}_qc_summary_flag'
    if qc_summary_var in data.variables:
        mask = data[qc_summary_var] == QARTOD_FAIL_FLAG
        data[param] = data[param].where(~mask)
        logger.debug(f"Applied QC summary filter: {mask.sum().values} points masked")
    
    # Apply QC results flag
    qc_results_var = f'{param}_qc_results'
    if qc_results_var in data.variables:
        mask = data[qc_results_var] == QARTOD_FAIL_FLAG
        data[param] = data[param].where(~mask)
        logger.debug(f"Applied QC results filter: {mask.sum().values} points masked")
    
    # Apply rollup annotations (affects all parameters)
    if 'rollup_annotations_qc_results' in data.variables:
        data = data.where(data.rollup_annotations_qc_results < QARTOD_FAIL_FLAG)
        logger.debug("Applied rollup annotation filter")
    
    # Apply parameter-specific annotations
    anno_var = f'{param}_annotations_qc_results'
    if anno_var in data.variables:
        data = data.where(data[anno_var] < QARTOD_SUSPECT_FLAG)
        logger.debug("Applied parameter annotation filter")
    
    # Apply time range filter
    if cut_off:
        cut = parser.parse(cut_off)
        cut = cut.astimezone(pytz.utc)
        end_date = cut.strftime('%Y-%m-%dT%H:%M:%S')
    else:
        cut = parser.parse(data.time_coverage_end)
        cut = cut.astimezone(pytz.utc)
        end_date = cut.strftime('%Y-%m-%dT%H:%M:%S')
    
    data = data.sel(time=slice(START_DATE, end_date))
    logger.info(f"Time range: {START_DATE} to {end_date}")
    
    return data


def processData(data: xr.Dataset, param: str) -> xr.Dataset:
    """
    Process data for a specific parameter.
    
    Currently a placeholder for future processing steps.
    
    Args:
        data: xarray Dataset
        param: Parameter name
    
    Returns:
        Processed dataset (currently unchanged)
    """
    logger.debug(f"Processing parameter: {param}")
    return data
