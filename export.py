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
        DataFrame with climatology ranges
    """
    if testKey not in VALID_TEST_KEYS:
        raise ValueError(f"Invalid testKey: {testKey}. Must be one of {VALID_TEST_KEYS}")
    
    if 'binned' in testKey:
        rows = []
        for depth_bin, values in clim['binned'].items():
            # Validate that we have 12 months of data
            if len(values['lower']) != 12 or len(values['upper']) != 12:
                logger.error(f"Expected 12 monthly values for bin {depth_bin}, "
                           f"got lower={len(values['lower'])}, upper={len(values['upper'])}")
                raise ValueError(f"Invalid monthly data size for depth bin {depth_bin}")
            
            monthRanges = [f"[{values['lower'][i]:.2f}, {values['upper'][i]:.2f}]" 
                          for i in range(12)]
            
            # Prepend the depth bin
            first = f"[{depth_bin[0]}, {depth_bin[1]}]"
            row = [first] + monthRanges
            rows.append(row)
    else:
        # For integrated ('int') or fixed platform
        # Validate that we have 12 months of data
        if len(clim[testKey]['lower']) != 12 or len(clim[testKey]['upper']) != 12:
            logger.error(f"Expected 12 monthly values for {testKey}, "
                       f"got lower={len(clim[testKey]['lower'])}, upper={len(clim[testKey]['upper'])}")
            raise ValueError(f"Invalid monthly data size for {testKey}")
        
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
        DataFrame with gross range values
    """
    if testKey not in VALID_TEST_KEYS:
        raise ValueError(f"Invalid testKey: {testKey}. Must be one of {VALID_TEST_KEYS}")
    
    if 'binned' in testKey:
        rows = []
        for depth_bin, values in gr['binned'].items():
            # Validate data exists
            if 'lower' not in values or 'upper' not in values:
                logger.error(f"Missing lower/upper values for bin {depth_bin}")
                raise ValueError(f"Invalid gross range data for depth bin {depth_bin}")
            
            lower_val = values['lower']
            upper_val = values['upper']
            
            # Handle both scalar and array cases
            if isinstance(lower_val, (list, tuple)):
                if len(lower_val) == 0:
                    raise ValueError(f"Empty lower value for bin {depth_bin}")
                lower_val = lower_val[0]
            if isinstance(upper_val, (list, tuple)):
                if len(upper_val) == 0:
                    raise ValueError(f"Empty upper value for bin {depth_bin}")
                upper_val = upper_val[0]
            
            gr_range = [f"[{float(lower_val):.2f}, {float(upper_val):.2f}]"]
            
            # Prepend the depth bin
            first = f"[{depth_bin[0]}, {depth_bin[1]}]"
            row = [first] + gr_range
            rows.append(row)
    else:
        # For integrated ('int') or fixed platform
        if 'lower' not in gr[testKey] or 'upper' not in gr[testKey]:
            logger.error(f"Missing lower/upper values for {testKey}")
            raise ValueError(f"Invalid gross range data for {testKey}")
        
        lower_val = gr[testKey]['lower']
        upper_val = gr[testKey]['upper']
        
        # Handle both scalar and array cases
        if isinstance(lower_val, (list, tuple)):
            if len(lower_val) == 0:
                raise ValueError(f"Empty lower value for {testKey}")
            lower_val = lower_val[0]
        if isinstance(upper_val, (list, tuple)):
            if len(upper_val) == 0:
                raise ValueError(f"Empty upper value for {testKey}")
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
                    raise ValueError(f"Unexpected key in climatology dict: {testKey}")

                try:
                    df_clim = build_climatology_object(param_dict['climatology'], testKey)

                    outfile = os.path.join(
                        folderPath,
                        f"{site}-{node}-{sensor}-{param}.climatology_table.csv.{testKey}"
                    )
                    df_clim.to_csv(outfile, index=False, header=False)
                    logger.info(f"Exported climatology table: {outfile}")

                except Exception as e:
                    logger.error(f"Failed to export climatology table for {param}/{testKey}: {e}")
                    raise

            # ------------- Climatology lookup file -------------

            if 'profiler' in platform:
                zinp = pressParam
                notes = "Variance not reported for binned profiler climatology"
            else:
                zinp = None
                # take notes from any valid key
                val_keys = [k for k in param_dict['climatology'] if k in VALID_TEST_KEYS]
                notes = param_dict['climatology'][val_keys[0]].get('notes', 'No notes available')

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
                raise

        # --------------------------------------------------
        #                 GROSS RANGE
        # --------------------------------------------------
        if 'gross_range' in param_dict:

            for testKey in param_dict['gross_range']:
                if testKey not in VALID_TEST_KEYS:
                    raise ValueError(f"Unexpected key in gross_range dict: {testKey}")

                try:
                    df_gr = build_gross_range_object(param_dict['gross_range'], testKey)

                    outfile = os.path.join(
                        folderPath,
                        f"{site}-{node}-{sensor}-{param}.gross_range_table.csv.{testKey}"
                    )
                    df_gr.to_csv(outfile, index=False, header=False)
                    logger.info(f"Exported gross range table: {outfile}")

                except Exception as e:
                    logger.error(f"Failed gross range export {param}/{testKey}: {e}")
                    raise

            # ---- Gross Range Lookup File ----
            val_keys = [k for k in param_dict['gross_range'] if k in VALID_TEST_KEYS]
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
                raise

    logger.info("Export completed successfully")
