# GIST: Ghost Infrastructure Spectral Tracker

**Ghost Infrastructure Spectral Tracker. Satellite-based detection of ghost infrastructure in Philippine public works.**



---

## Overview

GIST is an end-to-end pipeline that automates satellite-based verification of DPWH infrastructure projects using Google Earth Engine and Sentinel-2 imagery. It extends the Commission on Audit's existing but case-specific satellite fraud audit method to national scale, screening every reported project completion automatically without requiring a complaint or formal investigation to trigger the analysis.

The pipeline produces a ghost probability score (0–1) for each project, a spatial signal pattern, and before/after satellite thumbnails, all surfaced in a web dashboard backed by Supabase.

---

## Background

Between 2022 and 2025, the Department of Public Works and Highways received approximately ₱545 billion for flood control and related infrastructure. In October 2025, DPWH Secretary Vince Dizon confirmed that 421 of 8,000 inspected flood control sites were ghost projects: structures fully paid for that did not physically exist. Finance Secretary Ralph Recto told the Senate that ghost flood control projects produced economic losses of ₱42.3 billion to ₱118.5 billion over three years, equivalent to 95,000 to 266,000 foregone jobs.

The Commission on Audit, conducting expanded fraud audits from August 2025 onward, used physical inspections, drone surveillance, geotagged photography, and historical satellite imagery to establish non-existence. In multiple COA fraud audit reports filed with the Ombudsman, satellite imagery provided the chronological baseline. In one case, imagery taken 22 days after a contract's effectivity date already showed a pre-existing structure, indicating the project was paid for infrastructure that predated the contract. In another case, the structure DPWH representatives identified for auditors was 694 metres from the location specified in the approved engineering plans.

GIST scales this established but case-specific method across all 248,000 DPWH project records.

**Sources:** Philippine Star (Oct 10, 2025), GMA News (Feb 13, 2026), COA Fraud Audit Reports (Sept 2025–Feb 2026), Senate DBCC Briefing (Sept 2025).

---

## Project Structure

```
gist/
|
├── README.md
|
├── data/
│   ├── raw/                    # original .gpkg from DPWH / HuggingFace
│   ├── processed/              # all_projects.csv, flood_projects.csv
│   └── samples/                # sample_500.csv, coord_check.csv
|
├── scripts/
│   ├── 0_data_collection.py    # download dataset from HuggingFace, save as .gpkg
│   ├── 01_load_data.py         # validate coordinates, export pipeline-ready CSVs
│   ├── 02_pipeline.py          # GEE: build composites, compute spectral deltas
│   ├── 03_score.py             # classify signal patterns, ghost probability score
│   └── 04_upload.py            # push results and thumbnails to Supabase
|
├── dashboard/
│   └── index.html              # GIST dashboard (single-file, GitHub Pages ready)
|
├── docs/
│   ├── phantom_capital_paper_v2.docx
│   ├── phantom_capital_deck_v2.pptx
│   └── gist_oneshot.html
|
├── outputs/
│   ├── results.csv
│   ├── results_scored.csv
│   └── figures/
|
└── notebooks/
    └── exploration.ipynb
```

---

## Installation

```bash
pip install datasets geopandas pandas earthengine-api tqdm supabase shapely scikit-learn

# Authenticate with Google Earth Engine (one-time)
earthengine authenticate
```

---

## Data Sources

| Source | Description | License |
|---|---|---|
| `bettergovph/dpwh-transparency-data` | 248,000 DPWH project records with GPS coordinates, dates, budgets, contractors | CC0 1.0 |
| `COPERNICUS/S2_SR_HARMONIZED` | Sentinel-2 Surface Reflectance (Level 2A) via Google Earth Engine | Copernicus Open Access |
| COA Fraud Audit Reports 2025–2026 | Ground truth labels for confirmed ghost projects in Bulacan | Public record |

The DPWH dataset is downloaded from HuggingFace and saved as a GeoPackage using the data collection script:

```bash
python scripts/0_data_collection.py --output data/raw/dpwh_data.gpkg
```

This downloads all 248,000 records, attaches point geometry from the latitude and longitude columns, and writes a `.gpkg` file that the rest of the pipeline reads from.

---

## Usage

### Step 0: Download the dataset

```bash
python scripts/0_data_collection.py --output data/raw/dpwh_data.gpkg
```

This only needs to run once. It downloads the DPWH transparency dataset from HuggingFace and saves it locally as a GeoPackage.

### Step 1: Load and validate data

```bash
python scripts/01_load_data.py --input data/raw/dpwh_data.gpkg
```

Outputs:
- `all_projects.csv`: all records
- `flood_projects.csv`: flood control only, pipeline-ready
- `coord_check.csv`: projects with suspicious coordinates (outside PH bbox, null island, swapped lat/lon)

### Step 2: Run the GEE pipeline

```bash
# Test on stratified 500-row sample first (95% CI ±4.3%)
python scripts/02_pipeline.py \
  --input flood_projects.csv \
  --output results.csv \
  --project ee-fragotyron \
  --test \
  --spectral-workers 6 \
  --thumb-workers 5

# Full run
python scripts/02_pipeline.py \
  --input flood_projects.csv \
  --output results.csv \
  --project ee-fragotyron \
  --spectral-workers 6 \
  --thumb-workers 5
```

**`--test`** takes a proportional stratified sample by `completion_year × region` rather than `df.head(n)`, ensuring every year and region is represented.

**`--skip-thumbs`** skips thumbnail generation for faster spectral-only runs.

### Step 3: Score

```bash
python scripts/03_score.py --input outputs/results.csv --output outputs/results_scored.csv
```

### Step 4: Upload to Supabase

Run the SQL schema from the header of `04_upload.py` in the Supabase SQL editor first, then:

```bash
python scripts/04_upload.py \
  --input results_scored.csv \
  --url https://YOUR_PROJECT.supabase.co \
  --key YOUR_SERVICE_ROLE_KEY \
  --skip-thumbs
```

### Step 5: Deploy the dashboard

Edit `index.html` lines 7–8:

```js
const SUPABASE_URL      = 'https://YOUR_PROJECT.supabase.co'
const SUPABASE_ANON_KEY = 'YOUR_ANON_KEY'
```

Open locally by double-clicking `index.html`, or push to GitHub and enable Pages.

---

## Methodology

### Composite Construction

We use 180-day dry season windows (November through April) for both the before and after composites. Using the same seasonal window in both years prevents spectral drift from vegetation and moisture cycles, which matters for the Philippines, which receives 180–220 cloudy days annually in most provinces (Simonetti et al., 2021).

For projects where the dry season window yields fewer than 5 cloud-free scenes, the pipeline falls back to the full calendar year. A pixel-level validity check (`composite_has_data`) catches the edge case where scene count passes (e.g. 7 scenes) but every pixel at the specific location is cloud-masked, producing a black image.

### Spectral Indices

Nine indices are computed on each composite:

| Index | Direction | Weight | Notes |
|---|---|---|---|
| IBI | Positive = construction | 40% (primary) | Normalises against vegetation and water, robust to dry soil noise |
| TCB | Positive = construction | 30% (primary) | Sentinel-2 Tasseled Cap Brightness (Shi & Xu, 2019) |
| NDVI | Negative = construction | 10% (secondary) | Higher threshold (0.08); seasonal PH shifts reach 0.05-0.08 on unchanged land |
| MNDWI | Negative = construction | 10% (secondary) | Higher threshold for the same reason |
| BUI, TCG, TCW, NDBI, NDTI | Mixed | 10% combined | Supporting evidence |

NDBI uses B8A (narrow NIR) instead of B8 to avoid red-edge contamination from vegetation.

### Three Spatial Windows

COA fraud audits repeatedly documented DPWH representatives leading inspectors to project sites that did not match approved engineering plans. A single fixed buffer cannot handle this. Three windows are computed independently and never averaged. Their pattern of agreement is diagnostic:

| Window | Buffer | Reducer | Question |
|---|---|---|---|
| Tight | 30m | Mean | Did construction occur at the reported coordinate? |
| Broad | 100m | Mean | Did construction occur in the general vicinity? |
| Anywhere | 150m | Max | Did construction occur anywhere within 150m? |

**Key signal patterns:**

| Pattern | Interpretation | Action |
|---|---|---|
| T+B+A+ | Built at coordinate, construction confirmed | Low priority |
| T-B-A+ | COA-documented fraud type: coord displaced | Verify GPS first, then re-inspect |
| T-B-A- | No signal anywhere | Ghost candidate or displacement >150m; field check required |
| T+B-A- | Small point structure or noise | Manual review |

### Scoring

The heuristic scorer uses an IBI-primary approach:

1. **Pattern anchor** (60% weight): base probability from the spatial signal pattern
2. **Weighted index score** (40% weight): IBI and TCB lead; vegetation indices use higher thresholds to filter seasonal noise
3. **Displacement cap**: projects with T-B-A+ are capped at 0.60 (something was built nearby); projects with confirmed IBI signal are capped at 0.35

**Data quality gate:** projects where either composite has fewer than 3 cloud-free scenes, or where the pixel-level validity check fails, are capped at 0.50 (uncertain) and flagged as `data_quality: sparse`. These projects are excluded from ghost/suspect/legit counts in the dashboard header.

### Parallelism

The pipeline uses `ThreadPoolExecutor` (not `multiprocessing`) because GEE objects are not picklable. Threads release the GIL during HTTP I/O, allowing simultaneous GEE requests:

- `--spectral-workers` (default 4): parallel date groups within each batch, each firing one composite pair and three reduceRegions calls
- `--thumb-workers` (default 5): parallel thumbnail requests per date group
- Total concurrent GEE requests = spectral_workers x (3 + thumb_workers). Keep below 30 for non-commercial accounts.

---

## Limitations

**What the pipeline cannot detect:**

- **Excavation depth**: a structure built at half the specified depth produces the same surface spectral signature as a correctly-built structure after backfill. This is likely the most prevalent fraud type and is not caught here.
- **Material quality**: concrete mix ratio, rebar grade, and compaction density require physical material testing.
- **Coordinate displacement >150m**: T-B-A- is indistinguishable from a true ghost when the actual structure is more than 150 metres from the recorded coordinate. Both look identical from spectral analysis.
- **Pre-existing structures paid as new**: if fraud involves claiming an older structure as newly built, and that structure predates the Sentinel-2 era (pre-2017), the delta will correctly show zero change but the interpretation differs from a ghost project.

**Known data quality issues:**

- **2017 start year projects**: the before window falls in 2016, before reliable S2 SR Philippine coverage. The pipeline flags these and caps their scores.
- **Persistent cloud cover**: eastern Samar, Surigao, and other always-wet provinces may have too few cloud-free observations even in the dry season. The fallback expands to the full year but results remain less reliable.
- **DPWH coordinate accuracy**: COA fraud audits documented coordinates displaced by hundreds of metres. The three-window approach handles up to 150m; beyond that, field verification is required.

**Scorer limitations:**

The current scorer is a heuristic rule-based system, not a trained machine learning classifier. It is calibrated for audit triage, generating a ranked list for COA to prioritise inspections, not for legal determination. A trained classifier using COA's confirmed fraud audit coordinates as structured ground truth labels will replace this once sufficient labelled data is available.

---

## Dashboard

The `index.html` dashboard connects directly to Supabase and requires no build step. Features:

- Sortable, filterable project table (by status, year, name search)
- Projects with insufficient satellite data shown as "No Satellite Data" rather than "Likely Ghost", and excluded from the header counts
- Detail panel with ghost probability, expanded three-window signal display, before/after satellite thumbnails with scene counts, and spectral index matrix
- Data quality warning banner when composites were empty or sparse
- Google Maps link for every project coordinate

---

## References

- Xu, H. (2008). A new index for delineating built-up land features in satellite imagery. *International Journal of Remote Sensing*, 29(14), 4269–4276.
- Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. *International Journal of Remote Sensing*, 27(14), 3025–3033.
- Shi, T., & Xu, H. (2019). Derivation of Tasseled Cap Transformation Coefficients for Sentinel-2 MSI. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 12(10), 4038–4048.
- Simonetti, D., Pimple, U., Langner, A., & Marelli, A. (2021). Pan-Tropical Sentinel-2 Cloud-Free Annual Composite Datasets. *Data in Brief*, 39, 107488.
- Lacaux, J.P., et al. (2007). Classification of ponds from high-spatial resolution remote sensing. *Remote Sensing of Environment*, 106(1), 66–74.
- Commission on Audit. Fraud Audit Reports: DPWH Bulacan 1st DEO Flood Control Projects. Filed with the Office of the Ombudsman and ICI, September 2025 – February 2026.
- Development Budget Coordination Committee. Senate Committee on Finance Briefing, September 2025. Reported by GMA News Online.
- Dizon, V. Press briefing at the Independent Commission for Infrastructure. October 9, 2025. Reported by Philippine Star.
- Independent Commission for Infrastructure. Created under EO No. 94, September 15, 2025. Dissolved March 31, 2026. Reported by Filipino Times and Rappler.

---

## Acknowledgements

Data sourced from the DPWH Transparency Portal via the [bettergovph/dpwh-transparency-data](https://huggingface.co/datasets/bettergovph/dpwh-transparency-data) dataset (CC0 1.0). Special thanks to [@csiiiv](https://github.com/csiiiv) for the dpwh-transparency-data-api-scraper that enabled systematic data collection.

---

GIST: Ghost Infrastructure Spectral Tracker, 2026. All factual claims cited to named primary sources.
