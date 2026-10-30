# RCA QARTOD Lookup Tables

Generates QARTOD **gross range** and **climatology** lookup tables for Regional Cabled Array instruments. The data comes from the `ooi-data` S3 bucket (Zarr).

Each run handles one reference designator. It loads the Zarr stream and removes flagged and annotated data. It computes the test limits and writes CSVs to `~/qartod_staging/`.

---

## Quick start: gross range for one reference designator

### 1. Environment

Use Python 3

```bash
pip install xarray dask zarr s3fs pandas numpy loguru python-dateutil pytz
export AWS_KEY=<your key>
export AWS_SECRET=<your secret>
```

### 2. Run

Run from the repo root, because the config CSVs are read with relative paths:

```bash
cd qartod
python3 qartod_rca.py \
    -rd RS01SLBS-LJ01A-12-CTDPFB101 \
    -v sea_water_temperature \
    -d 0 \
    -t gross_range
```

| Flag | Required | Meaning |
|---|---|---|
| `-rd`, `--refDes` | yes | Reference designator. It must be a `refDes` in `siteParameters.csv`. |
| `-v`, `--userVars` | yes | A single data variable name, such as `sea_water_temperature`, or `all` for every variable listed for that refDes in `siteParameters.csv`. |
| `-d`, `--decThreshold` | yes | Target number of points after decimation, by block-averaging in time (`xarray.coarsen`). `0` turns decimation off. |
| `-t`, `--tests` | no | Test(s) to run: `gross_range`, `climatology`, or both separated by a space. The default is every test in `qartodTests.csv`. A test still runs only on variables whose parameter category is in that test's `parameters` list. |
| `-co`, `--cut_off` | no | End date (ISO format, e.g. `2024-12-31`). The default is the dataset's `time_coverage_end`. The start date is always `2014-01-01`. |

### 3. Output

The files are written to `~/qartod_staging/`:

| File | Contents |
|---|---|
| `<site>-<node>-<sensor>-<var>-gross_range_test_values.csv` | Lookup row with `qcConfig` `fail_span` (sensor limits from `parameterMap.csv`). The computed statistical range is recorded in the notes, not applied. |
| `<site>-<node>-<sensor>-<var>.gross_range_table.csv.fixed` | Range table (fixed platforms) |
| `<site>-<node>-<sensor>-<var>.gross_range_table.csv.int` | Range table (profilers; depth-integrated) |

For the example above, the lookup file is `~/qartod_staging/RS01SLBS-LJ01A-12-CTDPFB101-sea_water_temperature-gross_range_test_values.csv`.

---

## How the tests are calculated

### Data preparation (both tests)

Before either test, `qartodProcessing.filterData` masks annotated data:

- all variables wherever a rollup annotation (one with no parameters) is flagged fail (4)
- values where a parameter-specific annotation is suspect (**3**) or worse

Annotations are read from `s3://ooi-data/annotations/<refDes>.json`. The `exclude` field is not used.

It then keeps only data from `2014-01-01` to the cut-off date (`-co`, or the dataset's `time_coverage_end`).

Inside each test, values outside the sensor limits in `parameterMap.csv` are dropped, along with NaNs. The comparison is strict: values exactly at a limit are dropped too.

### Gross range (`grossRange.py`)

The lookup row applies only the **fail span** (sensor limits), as defined in parameterMap.csv. A statistical range is still computed, recorded in the `notes` and the range table, but not exported as a suspect span: it flagged little that the climatology test did not also flag, and raised many false flags where data drifts.

The statistical range is one `[lower, upper]` pair per variable and is calculated as follows:

1. **Normality check.** Skewness and excess kurtosis are computed for all remaining values together. The data counts as **normal** when both conditions hold:
   - `|skewness| < 1.0`
   - `-1.0 < excess kurtosis < 2.0`

   Excess kurtosis comes from `dask.array.stats.kurtosis(..., fisher=True)`, so a normal distribution scores 0. The lower bound is −1 so that flat, uniform-like data (excess kurtosis −1.2, e.g. profiler pressure) is treated as non-normal; mean ± 5σ of such data reaches far outside the observed range.
2. **Statistical range:**

   | Distribution | Lower | Upper | Coverage |
   |---|---|---|---|
   | Normal | mean − **5σ** | mean + **5σ** | 99.99994% |
   | Non-normal | **0.0000287th** percentile | **99.9999713th** percentile | the same two-sided coverage as ±5σ |

   σ is the population standard deviation of all remaining values (xarray `std`, `ddof=0`). If σ = 0, the span collapses to the mean.
3. **Clipping.** The range is clipped to the sensor limits.
4. **Fail span.** The fail span is the sensor limits from `parameterMap.csv`, unchanged. This is the only span exported.

The lookup file stores `{"qartod": {"gross_range_test": {"fail_span": [min, max]}}}`. The `notes` column records the statistical range (rounded to 2 decimals), the method, skewness, excess kurtosis, μ and σ, whether clipping happened, and whether the data was decimated.

To change the width, edit the constants at the top of `grossRange.py`. Equivalent percentiles: 3σ is 0.135 / 99.865, 4σ is 0.0032 / 99.9968.

| Constant | Current value |
|---|---|
| `NORMAL_STD_MULTIPLIER` | `5.0` |
| `PERCENTILE_LOWER` / `PERCENTILE_UPPER` | `0.0000287` / `99.9999713` |
| `SKEWNESS_THRESHOLD` | `1.0` |
| `EXCESS_KURTOSIS_LOWER` / `EXCESS_KURTOSIS_UPPER` | `-1.0` / `2.0` |

If no valid data remains, or the statistics come out as NaN, the statistical range falls back to the sensor limits. The `notes` column says when that happens.

### Climatology (`climatology.py`)

The test produces 12 monthly `[lower, upper]` pairs per variable, or 12 per depth bin for profilers.

The method follows OOI's reference implementation ([ooi-data-explorations `qartod/climatology.py`](https://github.com/oceanobservatories/ooi-data-explorations/blob/master/python/ooi_data_explorations/qartod/climatology.py)), except that the same number of harmonic cycles is used for the R² and for the fitted curve.

1. **Monthly series.** The full record is resampled to one mean per calendar month (`resample(time="1MS")`). Ten years of data gives about 120 values.
2. **Harmonic fit.** A least-squares fit to that series:

   ```
   y(t) = β0 + Σ_{k=1..N} [ β_sk·sin(2πkt/12) + β_ck·cos(2πkt/12) ]      t in months
   ```

   `N_HARMONICS = 4` gives annual, semi-annual, 4-month and 3-month cycles (12, 6, 4 and 3 months). The same model is used for both the R² and the fitted curve. The fit needs more monthly values than model terms (currently more than 9). Fewer cycles give a smoother curve that misses sharp seasonal features: on shallow profiler chlorophyll and oxygen, 2 cycles flagged 2–4× as much training-period data as 4 cycles.
3. **Center curve.** The fitted series is averaged by calendar month to give 12 values.
   - **R² ≥ 0.15:** the fitted curve is the center. Because the curve is defined for every month in the record, months with no data still get a value.
   - **R² < 0.15, or too few monthly values to fit:** the center is the monthly means instead. These are computed with all years pooled (all Januaries together, and so on), and missing months are filled by linear interpolation that wraps December → January.
4. **Monthly σ.** The population standard deviation (`ddof=0`) of the original data for each calendar month, with all years pooled. Missing months are filled with the same wrapping interpolation. Any σ that is still NaN or ≤ 0 is replaced with the median of the valid monthly σ values. If there are none, it becomes 1% of the sensor range. The `notes` column records any replacement.
5. **Bounds:**

   ```
   lower[m] = center[m] − 3σ[m]
   upper[m] = center[m] + 3σ[m]
   ```

   The multiplier is **3 standard deviations** (`N_STD_DEVIATIONS = 3`). Both bounds are clipped to the sensor limits.

The `notes` column says which center was used, gives the number of cycles and the R², and records whether the data was decimated.

| Constant | Current value |
|---|---|
| `N_STD_DEVIATIONS` | `3` |
| `N_HARMONICS` | `4` |
| `MIN_R2_THRESHOLD` | `0.15` |

**Profilers.** Climatology is `binned` (set in `qartodTests.csv`), so steps 1–5 run separately for each depth bin. Bins come from `_setup_profiler_bins` in `qartod_rca.py`:

| Node | Bins |
|---|---|
| Shallow profiler (`SF0*`) | 1 m bins from 6–105 m, then 5 m bins from 105–195 m |
| Shallow profiler, pCO2 and pH | 10 m bins from 15–115 m, then 115–150 m and 150–195 m |
| Deep profiler (`DP0*`) | 5 m bins from 200 m to 2900 m (DP01A), 600 m (DP01B) or 2600 m (DP03A) |
| Other | 5 m bins from the minimum to the maximum pressure |

**Output.** For each variable, the test writes `<site>-<node>-<sensor>-<var>.climatology_table.csv.<fixed|int|binned>` and a lookup file, `<site>-<node>-<sensor>-<var>-climatology_test_values.csv`.
- **Table columns:** a header row of months, `[1, 1]` … `[12, 12]`.
- **Table rows:** one row per depth bin, `[zmin, zmax]`. Fixed platforms have a single `[0, 0]` row.
- **Cells:** `[lower, upper]`.
- **Lookup file:** its table path points to `climatology_tables/…climatology_table.csv`, and for profilers it sets `zinp` to the pressure variable.

### Known issues

- **Percentile path ≈ data extremes.** The 0.0000287th and 99.9999713th percentiles only differ from the data's minimum and maximum once there are about 3.5 million points. For non-normal data the statistical range is essentially the min and max of the annotation-filtered data.

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

- **Fixed:** the whole record goes into one gross range.
- **Profiler:** gross range is `integrated`, meaning one range over all depths. Climatology is `binned` by depth (bins are set in `_setup_profiler_bins` in `qartod_rca.py`). For PAR and SPKIR, the integrated range uses only data shallower than 10 m.

---

## Code layout

| File | Role |
|---|---|
| `qartod_rca.py` | CLI entry point and driver: loads config and data, routes to tests, calls export |
| `qartodProcessing.py` | Argument parsing, S3/Zarr loading, annotation loading, QC and annotation filtering |
| `grossRange.py` | Gross range calculation |
| `climatology.py` | Monthly climatology calculation |
| `export.py` | Writes table and lookup CSVs to `~/qartod_staging/` |
