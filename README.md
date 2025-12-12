# QARTOD Processing for Regional Cabled Array

Automated Quality Assurance/Quality Control (QARTOD) processing system for Ocean Observatories Initiative (OOI) Regional Cabled Array data.

## Overview

This toolkit generates QARTOD lookup tables for oceanographic sensor data by computing statistical ranges and climatologies. It supports both fixed platforms (moorings) and profiling platforms (gliders, profilers) with depth binning.

### Features

- **Gross Range Test**: Calculates user-defined sensor ranges using statistical methods
- **Climatology Test**: Generates monthly climatology tables with seasonal patterns
- **Platform Support**: Handles fixed and profiling platforms
- **Memory Efficient**: Uses Dask for large dataset processing
- **AWS S3 Integration**: Direct data loading from OOI cloud storage

### QARTOD Standards

This implementation follows NOAA's QARTOD standards for real-time quality control:
- Flag values: 1 (pass), 2 (not evaluated), 3 (suspect), 4 (fail), 9 (missing)
- Statistical methods: mean ± 3σ for normal distributions, percentile-based for non-normal
- Seasonal climatology: Harmonic regression with 4-cycle decomposition

---

## Installation

### Requirements

- Python 3.8+
- AWS credentials with access to OOI S3 bucket

### Dependencies

```bash
pip install numpy pandas xarray dask loguru numba s3fs zarr
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/wruef/qartod.git
cd qartod
```

2. Set AWS credentials as environment variables:
```bash
export AWS_KEY="your_aws_access_key"
export AWS_SECRET="your_aws_secret_key"
```

3. Ensure configuration files are present:
- `parameterMap.csv` - Maps parameters to sensor ranges
- `siteParameters.csv` - Site/platform configurations
- `qartodTests.csv` - Test definitions and parameters

---

## Quick Start

### Basic Usage

Process all variables for a reference designator:

```bash
python qartod_rca.py \
    -rd CE01ISSM-MFD35-02-PRESFA000 \
    -d 10000 \
    -v all
```

### Command Line Arguments

| Argument | Short | Description | Required | Example |
|----------|-------|-------------|----------|---------|
| `--refDes` | `-rd` | Reference designator | Yes | `CE01ISSM-MFD35-02-PRESFA000` |
| `--decThreshold` | `-d` | Decimation threshold (0=none) | Yes | `10000` |
| `--userVars` | `-v` | Variables to process ('all' or name) | Yes | `all` or `sea_water_temperature` |
| `--cut_off` | `-co` | End date for data (ISO format) | No | `2023-12-31` |

---

## Examples

### Example 1: Fixed Platform (CTD Mooring)

Process temperature and salinity for a fixed mooring:

```bash
python qartod_rca.py \
    -rd CE01ISSM-MFD35-02-PRESFA000 \
    -d 5000 \
    -v all \
    -co 2023-12-31
```

**What this does:**
1. Loads data from S3 for the pressure sensor
2. Decimates to 5,000 points for faster processing
3. Processes all configured variables
4. Filters data through December 31, 2023
5. Generates gross range and climatology tables

**Output files** (in `~/qartod_staging/`):
```
CE01ISSM-MFD35-02-PRESFA000-sea_water_pressure.gross_range_table.csv.fixed
CE01ISSM-MFD35-02-PRESFA000-sea_water_pressure.climatology_table.csv.fixed
CE01ISSM-MFD35-02-PRESFA000-sea_water_pressure-gross_range_test_values.csv
CE01ISSM-MFD35-02-PRESFA000-sea_water_pressure-climatology_test_values.csv
```

### Example 2: Profiling Platform (CTD Profiler)

Process a profiler with depth binning:

```bash
python qartod_rca.py \
    -rd RS01SBPS-SF01A-2A-CTDPFA102 \
    -d 20000 \
    -v sea_water_temperature
```

**What this does:**
1. Loads CTD profiler data
2. Identifies pressure parameter (`int_ctd_pressure`)
3. Creates depth bins based on profiler type (SF = shallow, DP = deep)
4. Processes temperature for each depth bin
5. Generates both integrated and binned tables

**Output structure:**
```
# Integrated (full water column)
RS01SBPS-SF01A-2A-CTDPFA102-sea_water_temperature.gross_range_table.csv.int
RS01SBPS-SF01A-2A-CTDPFA102-sea_water_temperature.climatology_table.csv.int

# Binned (per depth range)
RS01SBPS-SF01A-2A-CTDPFA102-sea_water_temperature.gross_range_table.csv.binned
RS01SBPS-SF01A-2A-CTDPFA102-sea_water_temperature.climatology_table.csv.binned
```

### Example 3: Single Variable, No Decimation

Process only dissolved oxygen without decimation:

```bash
python qartod_rca.py \
    -rd CE02SHBP-LJ01D-06-CTDBPN106 \
    -d 0 \
    -v dissolved_oxygen
```

**Note:** Setting `-d 0` processes the full dataset (may be slow for large datasets).

### Example 4: Using Python API

You can also import and use the modules programmatically:

```python
import qartodProcessing as qp
import grossRange as gr
import climatology as ct

# Load data
data = qp.loadData('CE01ISSM-MFD35-02-PRESFA000-telemetered-presf_abc_dcl_tide_measurement')

# Run gross range test
sensor_limits = [0, 1000]  # PSI
results = gr.process_gross_range(
    data, 
    'sea_water_pressure', 
    sensor_limits,
    site='CE01ISSM',
    node='MFD35',
    sensor='02-PRESFA000',
    stream='presf_abc_dcl_tide_measurement'
)

print(f"User range: [{results['lower']:.2f}, {results['upper']:.2f}]")
print(f"Method: {results['notes']}")
```

---

## Understanding the Output

### Gross Range Tables

Gross range tables define user ranges (typically mean ± 3σ or percentile-based):

**Format:**
```csv
# Row 1: Header
"","gross_range"

# Row 2: Depth bin (for profilers) or [0,0] (for fixed), range
"[0, 0]","[9.50, 15.20]"
```

For profilers with binning:
```csv
"","gross_range"
"[6, 7]","[8.20, 14.50]"
"[7, 8]","[8.10, 14.30]"
"[10, 15]","[7.95, 14.10]"
```

### Climatology Tables

Climatology tables provide monthly [lower, upper] bounds:

**Format:**
```csv
# Row 1: Header with months
"","[1, 1]","[2, 2]","[3, 3]",...,"[12, 12]"

# Row 2+: Depth bin, monthly ranges
"[0, 0]","[8.20, 12.50]","[8.10, 12.30]","[8.50, 13.20]",...
```

**Example interpretation:**
- Depth bin [0, 0] (surface or integrated)
- January (month 1): Valid range is 8.20 to 12.50°C
- February (month 2): Valid range is 8.10 to 12.30°C
- etc.

### Lookup CSV Files

Lookup files map parameters to their table files:

**gross_range_test_values.csv:**
```csv
subsite,node,sensor,stream,parameters,qcConfig,source,notes
CE01ISSM,MFD35,02-PRESFA000,presf_abc_dcl_tide_measurement,"{""inp"": ""sea_water_pressure""}","{""qartod"": {""gross_range_test"": {""suspect_span"": [9.5, 15.2], ""fail_span"": [0, 1000]}}}","","Normal distribution (skew=0.123, excess_kurt=0.456): using mean ± 3 standard deviations."
```

**climatology_test_values.csv:**
```csv
subsite,node,sensor,stream,parameters,climatologyTable,source,notes
CE01ISSM,MFD35,02-PRESFA000,presf_abc_dcl_tide_measurement,"{""inp"": ""sea_water_temperature"", ""tinp"": ""time"", ""zinp"": null}","climatology_tables/CE01ISSM-MFD35-02-PRESFA000-sea_water_temperature.climatology_table.csv","","Harmonic regression variance explained: R²=0.856"
```

---

## Configuration Files

### parameterMap.csv

Maps OOI parameter names to sensor ranges:

```csv
dataParameter,variables,limits
sea_water_temperature,"['sea_water_temperature', 'temp']","[-5, 40]"
sea_water_pressure,"['sea_water_pressure', 'pressure']","[0, 1000]"
```

**Columns:**
- `dataParameter`: Standard parameter name
- `variables`: List of variable names in data (Python list as string)
- `limits`: Vendor sensor limits [min, max]

### siteParameters.csv

Defines platform configurations:

```csv
refDes,platformType,variables,zarrFile
CE01ISSM-MFD35-02-PRESFA000,fixed,"['sea_water_pressure']",CE01ISSM-MFD35-02-PRESFA000-telemetered-presf_abc_dcl_tide_measurement
RS01SBPS-SF01A-2A-CTDPFA102,profiler,"['sea_water_temperature', 'sea_water_practical_salinity']",RS01SBPS-SF01A-2A-CTDPFA102-streamed-ctdpf_sbe43_sample
```

**Columns:**
- `refDes`: Reference designator
- `platformType`: Either 'fixed' or 'profiler'
- `variables`: List of variables to process
- `zarrFile`: Path to Zarr dataset in S3 (relative to `ooi-data/` bucket)

### qartodTests.csv

Defines which tests apply to which parameters:

```csv
qartodTest,parameters,profileCalc,output
gross_range,"['sea_water_temperature', 'sea_water_pressure']","['integrated', 'binned']","['lower', 'upper']"
climatology,"['sea_water_temperature']","['integrated', 'binned']","['monthly_mean', 'monthly_std']"
```

**Columns:**
- `qartodTest`: Test name
- `parameters`: List of parameters this test applies to
- `profileCalc`: For profilers: 'integrated' (full column) and/or 'binned' (by depth)
- `output`: Expected output fields

---

## Advanced Usage

### Customizing Depth Bins

Edit `qartod_rca.py` function `_setup_profiler_bins()`:

```python
def _setup_profiler_bins(node: str) -> List[float]:
    if 'SF0' in node:
        # Custom shallow bins: every 1m from 0-20m
        return np.arange(0, 20, 1).tolist()
    # ... rest of function
```

### Processing Multiple Sites

Create a batch script:

```bash
#!/bin/bash
# process_all_sites.sh

SITES=(
    "CE01ISSM-MFD35-02-PRESFA000"
    "CE02SHBP-LJ01D-06-CTDBPN106"
    "RS01SBPS-SF01A-2A-CTDPFA102"
)

for site in "${SITES[@]}"; do
    echo "Processing $site..."
    python qartod_rca.py -rd "$site" -d 10000 -v all
    
    if [ $? -eq 0 ]; then
        echo "✓ $site completed"
    else
        echo "✗ $site failed"
    fi
done
```

Run with:
```bash
chmod +x process_all_sites.sh
./process_all_sites.sh
```

### Debugging

Enable debug logging:

```python
# Add to top of qartod_rca.py
from loguru import logger
logger.add("qartod_debug.log", level="DEBUG", rotation="10 MB")
```

Or use environment variable:
```bash
LOGURU_LEVEL=DEBUG python qartod_rca.py -rd CE01ISSM-MFD35-02-PRESFA000 -d 5000 -v all
```

---

## Troubleshooting

### Common Errors

**Error: `AWS credentials not found`**
```
Solution: Set environment variables:
export AWS_KEY="your_key"
export AWS_SECRET="your_secret"
```

**Error: `Reference designator not found: XXX`**
```
Solution: Add the reference designator to siteParameters.csv:
refDes,platformType,variables,zarrFile
XXX,fixed,"['parameter_name']",path-to-zarr-file
```

**Error: `No pressure parameter found`**
```
Solution: Profilers require either 'sea_water_pressure' or 'int_ctd_pressure'.
Check that the Zarr file includes pressure data.
```

**Error: `Parameter 'xxx' not found in dataset`**
```
Solution: Check the actual variable names in the Zarr file:
python -c "import qartodProcessing as qp; ds = qp.loadData('path'); print(list(ds.keys()))"
```

**Warning: `No valid data after filtering`**
```
Solution: Check sensor_range limits in parameterMap.csv. May be too restrictive.
Also check annotations - data may be flagged as fail.
```

### Performance Issues

**Slow processing:**
1. Increase decimation threshold: `-d 20000` → `-d 5000`
2. Process single variables: `-v sea_water_temperature` instead of `-v all`
3. Set a cut-off date: `-co 2022-12-31`

**Memory errors:**
1. Reduce decimation threshold
2. Process variables one at a time
3. Increase Dask chunk size in code

---

## Algorithm Details

### Gross Range Test

**For normally distributed data:**
```
User range = [μ - 3σ, μ + 3σ]
```
Where μ = mean, σ = standard deviation (99.73% coverage)

**For non-normal data:**
```
User range = [P₀.₁₅, P₉₉.₈₅]
```
Using 0.15th and 99.85th percentiles (99.7% coverage)

**Normality test:**
- Skewness: |skew| < 1.0
- Excess kurtosis: -2 < excess_kurt < 2

### Climatology Test

**Steps:**
1. Compute monthly mean and standard deviation
2. Apply 4-cycle harmonic regression:
   ```
   y(t) = β₀ + Σ[βᵢsin(2πfit) + βᵢ₊₁cos(2πfit)]
   ```
   Where i = 1,3,5,7 for annual, semi-annual, tertiary, and quarterly cycles

3. Calculate bounds:
   ```
   Lower = mean - 3 × std
   Upper = mean + 3 × std
   ```

4. If R² < 0.15, use raw monthly means instead

### Decimation (LTTB-M)

Largest Triangle Three Buckets - Modified algorithm:
1. Divide data into n bins
2. For each bin, select point that maximizes triangle area with adjacent points
3. Preserve time spacing by using middle point's X-coordinate
4. Use max-area point's Y-coordinate for best representation

---

## Output File Structure

```
~/qartod_staging/
├── CE01ISSM-MFD35-02-PRESFA000-sea_water_pressure.gross_range_table.csv.fixed
├── CE01ISSM-MFD35-02-PRESFA000-sea_water_pressure.climatology_table.csv.fixed
├── CE01ISSM-MFD35-02-PRESFA000-sea_water_pressure-gross_range_test_values.csv
├── CE01ISSM-MFD35-02-PRESFA000-sea_water_pressure-climatology_test_values.csv
└── ... (more files)
```

**File naming convention:**
```
{site}-{node}-{sensor}-{parameter}.{test}_table.csv.{calc_type}
{site}-{node}-{sensor}-{parameter}-{test}_test_values.csv
```

Where:
- `calc_type`: 'fixed', 'int' (integrated), or 'binned'
- `test`: 'gross_range' or 'climatology'

---

## API Reference

### qartodProcessing Module

#### `loadData(zarr_dir: str) -> xr.Dataset`
Load Zarr dataset from S3.

#### `loadAnnotations(site: str) -> pd.DataFrame`
Load quality control annotations for a site.

#### `filterData(data, node, site, sensor, param, cut_off, annotations, pid_dict) -> xr.Dataset`
Filter data based on QC flags and annotations.

#### `decimateData(xs: xr.Dataset, decimation_threshold: int) -> xr.Dataset`
Decimate dataset using LTTB algorithm.

### grossRange Module

#### `process_gross_range(ds, param, sensor_range, **kwargs) -> Dict`
Calculate gross range using statistical methods.

**Returns:**
```python
{
    'lower': float,      # Lower bound
    'upper': float,      # Upper bound
    'notes': str         # Method description
}
```

### climatology Module

#### `process_climatology(ds, param, sensor_range, **kwargs) -> Dict`
Generate monthly climatology with harmonic regression.

**Returns:**
```python
{
    'lower': [float] * 12,   # Monthly lower bounds
    'upper': [float] * 12,   # Monthly upper bounds
    'notes': str             # R² and method info
}
```

### export Module

#### `exportTables(qartodDict, qartodTests_dict, site, node, sensor, stream, platform, pressParam)`
Export all QARTOD tables to CSV files.

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all functions
- Use `logger` for all logging (no `print()`)

---

## Testing

Currently no automated tests exist. Recommended test framework:

```bash
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=.

# Generate coverage report
pytest --cov=. --cov-report=html
```

---

## License

[Specify your license here]

---

## Citation

If you use this code in research, please cite:

```
[Add citation information]
```

---

## References

- NOAA QARTOD Standards: https://ioos.noaa.gov/project/qartod/
- OOI Data Portal: https://ooinet.oceanobservatories.org/
- LTTB Algorithm: Steinarsson, S. (2013). "Downsampling Time Series for Visual Representation"

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/wruef/qartod/issues
- Email: [your-email@domain.com]

---

## Changelog

### Version 1.0.0 (Current)
- Initial production release
- Fixed critical array indexing bugs
- Added comprehensive error handling
- Improved documentation
- Added type hints throughout

---

## Acknowledgments

- Ocean Observatories Initiative (OOI) for data infrastructure
- NOAA IOOS for QARTOD standards
- Anthropic's Claude for code review assistance 🤖