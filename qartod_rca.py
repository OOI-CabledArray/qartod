#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QARTOD Driver for Regional Cabled Array

Main orchestration script for running QARTOD tests on OOI data.
Handles both fixed platforms and profiling platforms with depth binning.
"""

from ast import literal_eval
import numpy as np
import pandas as pd
from loguru import logger
import traceback
from typing import Dict, List, Tuple, Optional, Any

import qartodProcessing as qp
import grossRange as gr
import climatology as ct
import export as ex

# Platform types
PLATFORM_FIXED = 'fixed'
PLATFORM_PROFILER = 'profiler'

# Test types
TEST_GROSS_RANGE = 'gross_range'
TEST_CLIMATOLOGY = 'climatology'

# Calculation types
CALC_INTEGRATED = 'int'
CALC_BINNED = 'binned'


def runQartod(test: str, data, param: str, limits: List[float], 
              **kwargs) -> Optional[Dict]:
    """
    Run specified QARTOD test on provided dataset.
    
    Args:
        test: Test name ('gross_range' or 'climatology')
        data: xarray Dataset
        param: Parameter name
        limits: Sensor limits [min, max]
        **kwargs: Additional context (site, node, sensor, stream, was_decimated, original_points, final_points)
    
    Returns:
        Dictionary of test results, or None if test not supported
    """
    site   = kwargs.get('site')
    node   = kwargs.get('node')
    sensor = kwargs.get('sensor')
    stream = kwargs.get('stream')
    was_decimated = kwargs.get('was_decimated', False)
    original_points = kwargs.get('original_points', None)
    final_points = kwargs.get('final_points', None)
    
    logger.debug(f"Running QARTOD test: {test} for {param}")
    
    qartod_results = None
    
    try:
        if TEST_GROSS_RANGE in test:
            qartod_results = gr.process_gross_range(
                data, param, limits, 
                site=site, node=node, sensor=sensor, stream=stream,
                was_decimated=was_decimated, 
                original_points=original_points,
                final_points=final_points
            )
        
        elif TEST_CLIMATOLOGY in test:
            qartod_results = ct.process_climatology(
                data, param, limits,
                site=site, node=node, sensor=sensor, stream=stream,
                was_decimated=was_decimated,
                original_points=original_points,
                final_points=final_points
            )
        
        else:
            logger.warning(f"Unsupported QARTOD test: {test}")
            return None
    
    except Exception as e:
        logger.error(f"Exception running test {test} for {param}: {e}")
        traceback.print_exc()
        return None
    
    return qartod_results


def run_binned_processing_for_param(
    data, 
    param: str, 
    press_param: str, 
    bins: List[Tuple[float, float]], 
    qartod_tests_dict: Dict, 
    qartod_tests: Dict,
    site: str, 
    node: str, 
    sensor: str, 
    stream: str, 
    cut_off: Optional[str], 
    annotations: pd.DataFrame, 
    pid_dict: Dict, 
    dec_threshold: int
) -> Dict[str, Dict]:
    """
    Process a parameter's profile tests for binned and integrated options.
    
    Args:
        data: xarray Dataset
        param: Parameter name
        press_param: Pressure parameter name
        bins: List of (lower, upper) depth bin tuples
        qartod_tests_dict: Test configuration per parameter
        qartod_tests: Test definitions
        site, node, sensor, stream: Platform identifiers
        cut_off: Optional date cutoff
        annotations: Annotations DataFrame
        pid_dict: Parameter ID dictionary
        dec_threshold: Decimation threshold
    
    Returns:
        Dictionary: {test_name: {'int': ..., 'binned': {bin: result, ...}}}
    """
    results_for_param = {}
    
    for test in qartod_tests_dict[param]['tests']:
        results_for_param[test] = {}
        
        # ========== INTEGRATED PROCESSING ==========
        if 'integrated' in qartod_tests[test]['profileCalc']:
            logger.info(f"Processing integrated test: {test}")
            
            # Work on a copy to avoid mutation
            data_param = data.copy(deep=False)
            
            # Track decimation info
            original_points = len(data_param['time'])
            was_decimated = False
            final_points = original_points
            
            # Decimate if necessary
            if (len(data_param['time']) > dec_threshold) and (dec_threshold > 0):
                data_param = qp.decimateData(data_param, dec_threshold)
                was_decimated = True
                final_points = len(data_param['time'])
            
            # Process and filter
            data_param = qp.processData(data_param, param)
            data_param = qp.filterData(
                data_param, node, site, sensor, param, 
                cut_off, annotations, pid_dict
            )
            
            try:
                results_for_param[test][CALC_INTEGRATED] = runQartod(
                    test, data_param, param, 
                    qartod_tests_dict[param]['limits'],
                    site=site, node=node, sensor=sensor, stream=stream,
                    was_decimated=was_decimated,
                    original_points=original_points,
                    final_points=final_points
                )
            except Exception as e:
                logger.error(f"Integrated test failed: {test}, {param}: {e}")
                traceback.print_exc()
                results_for_param[test][CALC_INTEGRATED] = None
        
        # ========== BINNED PROCESSING ==========
        if 'binned' in qartod_tests[test]['profileCalc']:
            logger.info(f"Processing binned test: {test}")
            results_for_param[test][CALC_BINNED] = {}
            
            for press_bin in bins:
                logger.info(f"Processing pressure bin: {press_bin}")
                
                # Create mask for this depth bin
                mask = (
                    (data[press_param] > press_bin[0]) & 
                    (data[press_param] < press_bin[1])
                )
                data_bin = data.where(mask, drop=False)
                data_bin = data_bin.dropna(dim="time", how="all")
                
                qartod_row = None
                
                try:
                    # Check if bin has data
                    has_data = False
                    try:
                        has_data = data_bin[press_param].notnull().any().compute()
                    except Exception:
                        has_data = bool(data_bin[press_param].notnull().any())
                    
                    if has_data:
                        # Track decimation info
                        original_points = len(data_bin['time'])
                        was_decimated = False
                        final_points = original_points
                        
                        # Decimate based on bin size
                        if (len(data_bin['time']) > dec_threshold) and (dec_threshold > 0):
                            data_bin = qp.decimateData(data_bin, dec_threshold)
                            was_decimated = True
                            final_points = len(data_bin['time'])
                        
                        # Process and filter
                        data_bin = qp.processData(data_bin, param)
                        data_bin = qp.filterData(
                            data_bin, node, site, sensor, param,
                            cut_off, annotations, pid_dict
                        )
                        
                        # Run QARTOD on bin
                        qartod_row = runQartod(
                            test, data_bin, param,
                            qartod_tests_dict[param]['limits'],
                            site=site, node=node, sensor=sensor, stream=stream,
                            was_decimated=was_decimated,
                            original_points=original_points,
                            final_points=final_points
                        )
                    else:
                        logger.info(f"No data for pressure bin: {press_bin}")
                
                except Exception as e:
                    logger.error(f"Failed to process bin {press_bin} for {test}: {e}")
                    traceback.print_exc()
                
                # Store result (may be None)
                results_for_param[test][CALC_BINNED][press_bin] = qartod_row
    
    return results_for_param


def _setup_profiler_bins(node: str) -> List[float]:
    """
    Create depth bins based on profiler node type.
    
    Args:
        node: Node designator (e.g., 'SF01A', 'DP01A')
    
    Returns:
        List of depth boundaries
    """
    if 'SF0' in node:
        # Shallow profiler: fine resolution near surface
        shallow_upper = np.arange(6, 105, 1)
        shallow_lower = np.arange(105, 200, 5)
        return np.concatenate((shallow_upper, shallow_lower), axis=0).tolist()
    
    elif 'DP0' in node:
        # Deep profiler: bins every 5m to max depth
        max_depth = {
            'DP01A': 2900, 
            'DP01B': 600, 
            'DP03A': 2600
        }
        max_d = max_depth.get(node, 2000)
        return np.arange(200, max_d, 5).tolist()
    
    else:
        # Default binning
        logger.warning(f"Unknown profiler node type: {node}, using default binning")
        return np.arange(0, 200, 10).tolist()


def runQartod_driver_main():
    """
    Main driver function for QARTOD processing.
    
    Orchestrates:
    1. Load configuration and data
    2. Process fixed or profiler platforms
    3. Export results to CSV tables
    """
    logger.info("=" * 60)
    logger.info("QARTOD Processing Started")
    logger.info("=" * 60)
    
    # Parse command line arguments
    args = qp.inputs()
    ref_des = args.refDes
    cut_off = args.cut_off
    dec_threshold = int(args.decThreshold)
    user_vars = args.userVars
    
    logger.info(f"Reference Designator: {ref_des}")
    logger.info(f"Decimation Threshold: {dec_threshold}")
    logger.info(f"User Variables: {user_vars}")
    if cut_off:
        logger.info(f"Cut-off Date: {cut_off}")
    
    # Load configuration files
    logger.info("Loading configuration files...")
    
    param_dict = (
        pd.read_csv(
            'parameterMap.csv', 
            converters={"variables": literal_eval, "limits": literal_eval}
        )
        .set_index('dataParameter')
        .T.to_dict()
    )
    
    sites_dict = (
        pd.read_csv(
            'siteParameters.csv', 
            converters={"variables": literal_eval}
        )
        .set_index('refDes')
        .T.to_dict('series')
    )
    
    qartod_tests = (
        pd.read_csv(
            'qartodTests.csv',
            converters={
                "output": literal_eval,
                "parameters": literal_eval,
                "profileCalc": literal_eval
            }
        )
        .set_index('qartodTest')
        .T.to_dict()
    )
    
    # Validate reference designator
    if ref_des not in sites_dict:
        logger.error(f"Reference designator not found: {ref_des}")
        raise ValueError(f"Unknown reference designator: {ref_des}")
    
    # Extract platform information
    platform = sites_dict[ref_des]['platformType']
    logger.info(f"Platform Type: {platform}")
    
    # Load parameter ID dictionary
    pid_dict = qp.loadPID()
    
    # Parse zarr file path
    zarr_file = sites_dict[ref_des]['zarrFile']
    site, node, port, instrument, method, stream = zarr_file.split('-')
    sensor = f'{port}-{instrument}'
    
    logger.info(f"Site: {site}, Node: {node}, Sensor: {sensor}, Stream: {stream}")
    
    # Load data and annotations
    logger.info("Loading data...")
    data = qp.loadData(zarr_file)
    
    logger.info("Loading annotations...")
    annotations = qp.loadAnnotations(ref_des)
    
    # Determine variables to process
    if 'all' in user_vars:
        data_vars = sites_dict[ref_des]['variables']
        logger.info(f"Processing all variables: {data_vars}")
    else:
        data_vars = [user_vars]
        logger.info(f"Processing single variable: {user_vars}")
    
    # Build QARTOD test configuration for each variable
    param_list = []
    qartod_tests_dict = {}
    
    for qc_var in data_vars:
        qartod_tests_dict[qc_var] = {}
        
        # Find matching parameter configuration
        qc_param_list = [
            i for i in param_dict 
            if qc_var in param_dict[i]['variables']
        ]
        
        if not qc_param_list:
            logger.warning(f"Variable not found in parameter map: {qc_var}")
            continue
        
        qc_param = qc_param_list[0]
        
        # Get tests applicable to this parameter
        qartod_tests_dict[qc_var]['tests'] = {
            t for t in qartod_tests 
            if qc_param in qartod_tests[t]['parameters']
        }
        
        qartod_tests_dict[qc_var]['limits'] = param_dict[qc_param]['limits']
        
        # Add all parameter variables to keep list
        for p in param_dict[qc_param]['variables']:
            param_list.append(p)
        
        logger.info(f"Variable {qc_var}: {len(qartod_tests_dict[qc_var]['tests'])} tests")
    
    # Add pressure parameter for profilers
    if PLATFORM_PROFILER in platform:
        if 'int_ctd_pressure' in data:
            param_list.append('int_ctd_pressure')
            logger.info("Added pressure parameter: int_ctd_pressure")
        elif 'sea_water_pressure' in data:
            param_list.append('sea_water_pressure')
            logger.info("Added pressure parameter: sea_water_pressure")
    
    # Drop unused variables to save memory
    all_vars = list(data.keys())
    drop_list = [item for item in all_vars if item not in param_list]
    if drop_list:
        data = data.drop_vars(drop_list)
        logger.info(f"Dropped {len(drop_list)} unused variables")
    
    # Initialize results dictionary
    qartod_dict = {}
    
    # ========== FIXED PLATFORM PROCESSING ==========
    if PLATFORM_FIXED in platform:
        logger.info("Processing FIXED platform")
        press_param = None
        
        # Track decimation info
        original_points = len(data['time'])
        was_decimated = False
        final_points = original_points
        
        # Decimate entire dataset once
        if ((len(data['time']) > dec_threshold) and (dec_threshold > 0)):
            data = qp.decimateData(data, dec_threshold)
            was_decimated = True
            final_points = len(data['time'])
        
        for param in data_vars:
            if param not in qartod_tests_dict:
                continue
            
            logger.info(f"Processing parameter: {param}")
            qartod_dict[param] = {}
            
            # Work on a copy for this parameter
            data_param = data.copy(deep=False)
            data_param = qp.processData(data_param, param)
            data_param = qp.filterData(
                data_param, node, site, sensor, param,
                cut_off, annotations, pid_dict
            )
            
            for test in qartod_tests_dict[param]['tests']:
                qartod_dict[param][test] = {}
                
                try:
                    qartod_dict[param][test][platform] = runQartod(
                        test, data_param, param,
                        qartod_tests_dict[param]['limits'],
                        site=site, node=node, sensor=sensor, stream=stream,
                        was_decimated=was_decimated,
                        original_points=original_points,
                        final_points=final_points
                    )
                except Exception as e:
                    logger.error(f"Test failed for {param}/{test}: {e}")
                    traceback.print_exc()
                    qartod_dict[param][test][platform] = None
    
    # ========== PROFILER PLATFORM PROCESSING ==========
    elif PLATFORM_PROFILER in platform:
        logger.info("Processing PROFILER platform")
        
        # Determine pressure parameter
        if 'sea_water_pressure' in data:
            press_param = 'sea_water_pressure'
        elif 'int_ctd_pressure' in data:
            press_param = 'int_ctd_pressure'
        else:
            logger.error(f"No pressure parameter found for profiler: {ref_des}")
            raise RuntimeError(
                'No pressure parameter found (sea_water_pressure or int_ctd_pressure). '
                'Unable to bin profiler data!'
            )
        
        logger.info(f"Using pressure parameter: {press_param}")
        
        # Create depth bins
        bin_list = _setup_profiler_bins(node)
        bins = [(bin_list[i], bin_list[i + 1]) for i in range(len(bin_list) - 1)]
        
        logger.info(f"Created {len(bins)} depth bins")
        
        # Process each parameter with binning
        for param in data_vars:
            if param not in qartod_tests_dict:
                continue
            
            logger.info(f"Processing parameter: {param}")
            qartod_dict[param] = {}
            
            results_for_param = run_binned_processing_for_param(
                data, param, press_param, bins, 
                qartod_tests_dict, qartod_tests,
                site, node, sensor, stream, 
                cut_off, annotations, pid_dict, dec_threshold
            )
            
            qartod_dict[param] = results_for_param
    
    else:
        logger.error(f"Unknown platform type: {platform}")
        raise ValueError(f"Unknown platform type: {platform}")
    
    # ========== EXPORT RESULTS ==========
    logger.info("Exporting results...")
    logger.debug("QARTOD Results Summary:")
    for param in qartod_dict:
        logger.debug(f"  {param}: {list(qartod_dict[param].keys())}")
    
    ex.exportTables(
        qartod_dict, qartod_tests_dict, 
        site, node, sensor, stream, 
        platform, press_param
    )
    
    logger.info("=" * 60)
    logger.info("QARTOD Processing Complete")
    logger.info("=" * 60)


if __name__ == '__main__':
    try:
        runQartod_driver_main()
    except Exception as e:
        logger.error(f"QARTOD processing failed: {e}")
        traceback.print_exc()
        sys.exit(1)
