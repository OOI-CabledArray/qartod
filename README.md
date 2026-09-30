# RCA QARTOD Lookup Tables

Generates QARTOD **gross range** and **climatology** lookup tables for Regional Cabled Array instruments. The data comes from the `ooi-data` S3 bucket (Zarr).

Each run handles one reference designator. It loads the Zarr stream and removes flagged and annotated data. It computes the test limits and writes CSVs to `~/qartod_staging/`.

---

## Quick start: gross range for one reference designator

### 1. Environment

Use Python 3. On some machines `python` still points to 2.7, so call `python3` explicitly.

```bash
pip install xarray dask zarr s3fs pandas numpy loguru python-dateutil pytz
export AWS_KEY=<your key>
export AWS_SECRET=<your secret>
```

### 2. Turn off climatology

No command-line flag chooses which test runs. The driver runs every test in `qartodTests.csv` whose `parameters` list includes your variable's parameter category. To run only gross range, delete (or cut out temporarily) the `climatology` row so the file looks like this:

```csv
qartodTest,output,parameters,profileCalc
gross_range,"['lookup']","['pressure','temperature', ... ]","['integrated']"
```

Put the row back afterwards if you want climatology on the next run.

### 3. Run

Run from the repo root, because the config CSVs are read with relative paths:

```bash
cd qartod
python3 qartod_rca.py \
    -rd RS01SLBS-LJ01A-12-CTDPFB101 \
    -v sea_water_temperature \
    -d 0
```

| Flag | Required | Meaning |
|---|---|---|
| `-rd`, `--refDes` | yes | Reference designator. It must be a `refDes` in `siteParameters.csv`. |
| `-v`, `--userVars` | yes | A single data variable name, such as `sea_water_temperature`, or `all` for every variable listed for that refDes in `siteParameters.csv`. |
| `-d`, `--decThreshold` | yes | Target number of points after decimation. `0` turns decimation off. |
| `-co`, `--cut_off` | no | End date (ISO format, e.g. `2024-12-31`). The default is the dataset's `time_coverage_end`. The start date is always `2014-01-01`. |

### 4. Output

The files are written to `~/qartod_staging/`:

| File | Contents |
|---|---|
| `<site>-<node>-<sensor>-<var>-gross_range_test_values.csv` | Lookup row with `qcConfig`: `suspect_span` (the computed range) and `fail_span` (sensor limits from `parameterMap.csv`), plus notes on the method |
| `<site>-<node>-<sensor>-<var>.gross_range_table.csv.fixed` | Range table (fixed platforms) |
| `<site>-<node>-<sensor>-<var>.gross_range_table.csv.int` | Range table (profilers; depth-integrated) |

For the example above, the lookup file is `~/qartod_staging/RS01SLBS-LJ01A-12-CTDPFB101-sea_water_temperature-gross_range_test_values.csv`.

---

## How gross range is calculated

The calculation is in `grossRange.py`, in `process_gross_range`:

1. Data outside the sensor limits (`limits` in `parameterMap.csv`) is dropped.
2. The code checks normality with skewness and excess kurtosis (`|skew| < 1` and `-2 < excess kurtosis < 2`).
3. **Normal:** suspect span = mean ± 5σ.
   **Non-normal:** suspect span = the 0.0000287th to 99.9999713th percentiles, which is about the same coverage as ±5σ.
4. The result is clipped to the sensor limits.

To change the width, edit the constants at the top of `grossRange.py` (`NORMAL_STD_MULTIPLIER`, `PERCENTILE_LOWER`, `PERCENTILE_UPPER`). Commented-out 3σ and 4σ presets are there too.

Before the calculation, `qartodProcessing.filterData` removes data that failed existing QC. It masks:
- values where `<var>_qc_summary_flag` or `<var>_qc_results` is 4 (fail)
- all variables wherever a rollup annotation (one with no parameters) is flagged fail
- values where a parameter-specific annotation is suspect (3) or worse

Annotations are read from `s3://ooi-data/annotations/<refDes>.json`.

---

## Configuration files

| File | Purpose |
|---|---|
| `siteParameters.csv` | One row per refDes: `platformType` (`fixed` / `profiler`), Zarr stream name, and the variables to process for `-v all` |
| `parameterMap.csv` | Maps a parameter category (e.g. `temperature`) to its data variable names and its sensor limits `[min, max]`. The limits are also the gross range **fail span**. |
| `qartodTests.csv` | Lists the tests to run, the parameter categories each test applies to, and the profiler calculation (`integrated` or `binned`) |
| `multiParameters.csv` | Instruments whose array variables are split into separate variables (e.g. SPKIR wavelengths become `spkir_downwelling_vector_412nm`, …) |

For a variable to get a gross range, it must be in the refDes row of `siteParameters.csv`, it must appear in some category's `variables` list in `parameterMap.csv`, and that category must be in the `gross_range` parameters list in `qartodTests.csv`.

Adding a new instrument:
1. Add a row to `siteParameters.csv`. `zarrFile` is `<refDes>-<method>-<stream>`.
2. Make sure each variable is mapped in `parameterMap.csv`.
3. Make sure the category is in the `parameters` list of the test you want in `qartodTests.csv`.

---

## Fixed vs. profiler platforms

- **Fixed:** the whole record goes into one gross range. When `-d > 0`, decimation uses LTTB (`decimate.py`).
- **Profiler:** gross range is `integrated`, meaning one range over all depths. Climatology is `binned` by depth (bins are set in `_setup_profiler_bins` in `qartod_rca.py`). When `-d > 0`, decimation uses `xarray.coarsen`. For PAR and SPKIR, the integrated range uses only data shallower than 10 m.

---

## Code layout

| File | Role |
|---|---|
| `qartod_rca.py` | CLI entry point and driver: loads config and data, routes to tests, calls export |
| `qartodProcessing.py` | Argument parsing, S3/Zarr loading, annotation loading, QC and annotation filtering |
| `grossRange.py` | Gross range calculation |
| `climatology.py` | Monthly climatology calculation |
| `decimate.py` | LTTB downsampling |
| `export.py` | Writes table and lookup CSVs to `~/qartod_staging/` |
