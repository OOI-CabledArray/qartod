#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json
import pandas as pd
import os
from loguru import logger

# Constants for validation
VALID_TEST_KEYS = ['int', 'binned', 'fixed']

def build_climatology_object(clim, testKey):
    """
    Build climatology table with monthly ranges.
    
    Args:
        clim: Climatology data dictionary
        testKey: Test type ('int', 'binned', or 'fixed')
    
    Returns:
        DataFrame with climatology ranges, or None if data is invalid
    """
    # FIXED: Add defensive checks for None values
    if clim is None:
        logger.error(f"Climatology data is None for testKey={testKey}")
        return None
    
    if testKey not in VALID_TEST_KEYS:
        logger.error(f"Invalid testKey: {testKey}. Must be one of {VALID_TEST_KEYS}")
        return None
    
    # FIXED: Check if the testKey exists in the clim dictionary
    if testKey not in clim:
        logger.error(f"TestKey '{testKey}' not found in climatology data. Available keys: {list(clim.keys())}")
        return None
    
    if 'binned' in testKey:
        rows = []
        
        # FIXED: Check if binned data exists and is valid
        if not isinstance(clim['binned'], dict) or len(clim['binned']) == 0:
            logger.error("No valid binned climatology data found")
            return None
        
        for depth_bin, values in clim['binned'].items():
            # FIXED: Add None checks for values
            if values is None:
                logger.error(f"Climatology values are None for bin {depth_bin}")
                continue
            
            if not isinstance(values, dict):
                logger.error(f"Climatology values are not a dictionary for bin {depth_bin}: {type(values)}")
                continue
            
            if 'lower' not in values or 'upper' not in values:
                logger.error(f"Missing 'lower' or 'upper' keys for bin {depth_bin}")
                continue
            
            if values['lower'] is None or values['upper'] is None:
                logger.error(f"Climatology lower/upper bounds are None for bin {depth_bin}")
                continue
            
            # Validate that we have 12 months of data
            if len(values['lower']) != 12 or len(values['upper']) != 12:
                logger.error(f"Expected 12 monthly values for bin {depth_bin}, "
                           f"got lower={len(values['lower'])}, upper={len(values['upper'])}")
                continue
            
            monthRanges = [f"[{values['lower'][i]:.2f}, {values['upper'][i]:.2f}]" 
                          for i in range(12)]
            
            # Prepend the depth bin
            first = f"[{depth_bin[0]}, {depth_bin[1]}]"
            row = [first] + monthRanges
            rows.append(row)
        
        # FIXED: Check if any valid rows were created
        if len(rows) == 0:
            logger.error("No valid binned climatology data could be processed")
            return None
    else:
        # For integrated ('int') or fixed platform
        # FIXED: Add None and type checks
        if not isinstance(clim[testKey], dict):
            logger.error(f"Climatology data for {testKey} is not a dictionary: {type(clim[testKey])}")
            return None
        
        if 'lower' not in clim[testKey] or 'upper' not in clim[testKey]:
            logger.error(f"Missing 'lower' or 'upper' keys for {testKey}")
            return None
        
        if clim[testKey]['lower'] is None or clim[testKey]['upper'] is None:
            logger.error(f"Climatology lower/upper bounds are None for {testKey}")
            return None
        
        # Validate that we have 12 months of data
        if len(clim[testKey]['lower']) != 12 or len(clim[testKey]['upper']) != 12:
            logger.error(f"Expected 12 monthly values for {testKey}, "
                       f"got lower={len(clim[testKey]['lower'])}, upper={len(clim[testKey]['upper'])}")
            return None
        
        first = "[0, 0]"
        monthRanges = [f"[{clim[testKey]['lower'][i]:.2f}, {clim[testKey]['upper'][i]:.2f}]" 
                      for i in range(12)]
        rows = [[first] + monthRanges]
    
    df_clim = pd.DataFrame(rows)
    header = [""] + [f"[{i}, {i}]" for i in range(1, 13)]
    df_out = pd.concat([pd.DataFrame([header]), df_clim], ignore_index=True)
    
    return df_out


def build_gross_range_object(gr, testKey):
    """
    Build gross range table.
    
    Args:
        gr: Gross range data dictionary
        testKey: Test type ('int', 'binned', or 'fixed')
    
    Returns:
        DataFrame with gross range values, or None if data is invalid
    """
    # FIXED: Add defensive checks
    if gr is None:
        logger.error(f"Gross range data is None for testKey={testKey}")
        return None
    
    if testKey not in VALID_TEST_KEYS:
        logger.error(f"Invalid testKey: {testKey}. Must be one of {VALID_TEST_KEYS}")
        return None
    
    if testKey not in gr:
        logger.error(f"TestKey '{testKey}' not found in gross range data. Available keys: {list(gr.keys())}")
        return None
    
    if 'binned' in testKey:
        rows = []
        
        if not isinstance(gr['binned'], dict) or len(gr['binned']) == 0:
            logger.error("No valid binned gross range data found")
            return None
        
        for depth_bin, values in gr['binned'].items():
            # Validate data exists
            if values is None or not isinstance(values, dict):
                logger.error(f"Invalid values for bin {depth_bin}")
                continue
            
            if 'lower' not in values or 'upper' not in values:
                logger.error(f"Missing lower/upper values for bin {depth_bin}")
                continue
            
            lower_val = values['lower']
            upper_val = values['upper']
            
            # Handle both scalar and array cases
            if isinstance(lower_val, (list, tuple)):
                if len(lower_val) == 0:
                    logger.error(f"Empty lower value for bin {depth_bin}")
                    continue
                lower_val = lower_val[0]
            if isinstance(upper_val, (list, tuple)):
                if len(upper_val) == 0:
                    logger.error(f"Empty upper value for bin {depth_bin}")
                    continue
                upper_val = upper_val[0]
            
            gr_range = [f"[{float(lower_val):.2f}, {float(upper_val):.2f}]"]
            
            # Prepend the depth bin
            first = f"[{depth_bin[0]}, {depth_bin[1]}]"
            row = [first] + gr_range
            rows.append(row)
        
        if len(rows) == 0:
            logger.error("No valid binned gross range data could be processed")
            return None
    else:
        # For integrated ('int') or fixed platform
        if not isinstance(gr[testKey], dict):
            logger.error(f"Gross range data for {testKey} is not a dictionary: {type(gr[testKey])}")
            return None
        
        if 'lower' not in gr[testKey] or 'upper' not in gr[testKey]:
            logger.error(f"Missing lower/upper values for {testKey}")
            return None
        
        lower_val = gr[testKey]['lower']
        upper_val = gr[testKey]['upper']
        
        # Handle both scalar and array cases
        if isinstance(lower_val, (list, tuple)):
            if len(lower_val) == 0:
                logger.error(f"Empty lower value for {testKey}")
                return None
            lower_val = lower_val[0]
        if isinstance(upper_val, (list, tuple)):
            if len(upper_val) == 0:
                logger.error(f"Empty upper value for {testKey}")
                return None
            upper_val = upper_val[0]
        
        first = "[0, 0]"
        gr_range = [f"[{float(lower_val):.2f}, {float(upper_val):.2f}]"]
        rows = [[first] + gr_range]

    df_gr = pd.DataFrame(rows)
    df_out = pd.concat([pd.DataFrame([["", "gross_range"]]), df_gr], ignore_index=True)
    
    return df_out


def write_lookup_csv(outfile, headers, row):
    """
    Helper function to write CSV lookup files with minimal quoting.
    
    Uses QUOTE_MINIMAL to avoid double-quoting JSON strings.
    
    Args:
        outfile: Path to output file
        headers: List of column headers
        row: List of row values
    """
    try:
        with open(outfile, "w", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(headers)
            writer.writerow(row)
        logger.info(f"Exported: {outfile}")
    except IOError as e:
        logger.error(f"Failed to write {outfile}: {e}")
        raise


def exportTables(qartodDict, qartodTests_dict, site, node, sensor, stream, platform, pressParam):
    """
    Cleaned and corrected QARTOD export function.
    Produces:
      - climatology tables + lookup CSV
      - gross range tables + lookup CSV
    """

    folderPath = os.path.join(os.path.expanduser('~'), 'qartod_staging')
    os.makedirs(folderPath, exist_ok=True)
    logger.info(f"Exporting QARTOD tables to {folderPath}")

    for param, param_dict in qartodDict.items():
        limits = qartodTests_dict[param]['limits']

        # --------------------------------------------------
        #                   CLIMATOLOGY
        # --------------------------------------------------
        if 'climatology' in param_dict:

            for testKey in param_dict['climatology']:
                if testKey not in VALID_TEST_KEYS:
                    logger.error(f"Unexpected key in climatology dict: {testKey}")
                    continue

                try:
                    df_clim = build_climatology_object(param_dict['climatology'], testKey)
                    
                    # FIXED: Check if build failed and skip this testKey
                    if df_clim is None:
                        logger.warning(f"Skipping climatology export for {param}/{testKey} due to invalid data")
                        continue

                    outfile = os.path.join(
                        folderPath,
                        f"{site}-{node}-{sensor}-{param}.climatology_table.csv.{testKey}"
                    )
                    df_clim.to_csv(outfile, index=False, header=False)
                    logger.info(f"Exported climatology table: {outfile}")

                except Exception as e:
                    logger.error(f"Failed to export climatology table for {param}/{testKey}: {e}")
                    # FIXED: Continue instead of raising to allow other parameters to export
                    continue

            # ------------- Climatology lookup file -------------

            if 'profiler' in platform:
                zinp = pressParam
                notes = "Variance not reported for binned profiler climatology"
            else:
                zinp = None
                # take notes from any valid key
                val_keys = [k for k in param_dict['climatology'] if k in VALID_TEST_KEYS]
                
                # FIXED: Handle case where no valid keys have notes
                if val_keys:
                    clim_data = param_dict['climatology'][val_keys[0]]
                    if isinstance(clim_data, dict) and 'notes' in clim_data:
                        notes = clim_data['notes']
                    else:
                        notes = 'No notes available'
                else:
                    notes = 'No valid climatology data available'

            params_dict = {"inp": param, "tinp": "time", "zinp": zinp}

            climatologyTable = f"climatology_tables/{site}-{node}-{sensor}-{param}.climatology_table.csv"
            outfile = os.path.join(folderPath, f"{site}-{node}-{sensor}-{param}-climatology_test_values.csv")

            lookup_row = [
                site,
                node,
                sensor,
                stream,
                str(params_dict),       # Python dict format (single quotes)
                climatologyTable,
                "",
                notes
            ]

            try:
                with open(outfile, "w", newline="") as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(["subsite","node","sensor","stream",
                                     "parameters","climatologyTable","source","notes"])
                    writer.writerow(lookup_row)

                logger.info(f"Exported: {outfile}")

            except IOError as e:
                logger.error(f"Failed to write {outfile}: {e}")
                # FIXED: Continue instead of raising
                continue

        # --------------------------------------------------
        #                 GROSS RANGE
        # --------------------------------------------------
        if 'gross_range' in param_dict:

            for testKey in param_dict['gross_range']:
                if testKey not in VALID_TEST_KEYS:
                    logger.error(f"Unexpected key in gross_range dict: {testKey}")
                    continue

                try:
                    df_gr = build_gross_range_object(param_dict['gross_range'], testKey)
                    
                    # FIXED: Check if build failed and skip this testKey
                    if df_gr is None:
                        logger.warning(f"Skipping gross range export for {param}/{testKey} due to invalid data")
                        continue

                    outfile = os.path.join(
                        folderPath,
                        f"{site}-{node}-{sensor}-{param}.gross_range_table.csv.{testKey}"
                    )
                    df_gr.to_csv(outfile, index=False, header=False)
                    logger.info(f"Exported gross range table: {outfile}")

                except Exception as e:
                    logger.error(f"Failed gross range export {param}/{testKey}: {e}")
                    # FIXED: Continue instead of raising
                    continue

            # ---- Gross Range Lookup File ----
            val_keys = [k for k in param_dict['gross_range'] if k in VALID_TEST_KEYS]
            
            # FIXED: Handle case where no valid keys exist
            if not val_keys:
                logger.error(f"No valid gross range keys found for {param}")
                continue
            
            testKey = val_keys[0]

            lower_val = param_dict['gross_range'][testKey]['lower']
            upper_val = param_dict['gross_range'][testKey]['upper']

            # Normalize scalars
            if isinstance(lower_val, (list, tuple)):
                lower_val = float(lower_val[0]) if lower_val else 0.0
            if isinstance(upper_val, (list, tuple)):
                upper_val = float(upper_val[0]) if upper_val else 0.0

            qc_dict = {
                "subsite": site,
                "node": node,
                "sensor": sensor,
                "stream": stream,
                "parameters": {"inp": param},
                "gross_range_suspect": [float(f"{lower_val:.2f}"), float(f"{upper_val:.2f}")],
                "gross_range_fail": limits,
                "notes": param_dict['gross_range'][testKey].get("notes", "No notes available")
            }

            qcConfig = {
                "qartod": {
                    "gross_range_test": {
                        "suspect_span": qc_dict["gross_range_suspect"],
                        "fail_span": qc_dict["gross_range_fail"]
                    }
                }
            }

            lookup_row = [
                qc_dict['subsite'],
                qc_dict['node'],
                qc_dict['sensor'],
                qc_dict['stream'],
                str(qc_dict['parameters']),   # Python dict format
                str(qcConfig),
                "",
                qc_dict['notes']
            ]

            outfile = os.path.join(folderPath, f"{site}-{node}-{sensor}-{param}-gross_range_test_values.csv")

            try:
                with open(outfile, "w", newline="") as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(["subsite","node","sensor","stream","parameters",
                                     "qcConfig","source","notes"])
                    writer.writerow(lookup_row)

                logger.info(f"Exported: {outfile}")

            except IOError as e:
                logger.error(f"Failed to write {outfile}: {e}")
                # FIXED: Continue instead of raising
                continue

    logger.info("Export completed successfully")