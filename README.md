# GIST: Ghost Infrastructure Spectral Tracker

Tyron Rex Frago, April 2026

---

## Summary

This project uses Sentinel-2 satellite imagery and Google Earth Engine to verify whether Philippine DPWH flood control projects marked as completed were actually built. For each of 248,000 project records, GIST computes spectral change across three spatial sampling windows before and after the reported construction period. The most common signal pattern found was construction detectable only at 150 metres from the contracted coordinate, consistent with the site-mismatch fraud type the Commission on Audit documented in its 2025 Bulacan fraud audit reports.

## Live Demo

[Access the project here](https://tdfrago.github.io/yale-environmental-data-science-capstone/)

## Data Status

- Current dataset: **+24,000 records**
---

## Project Rationale

Between 2022 and 2025, the Philippine Department of Public Works and Highways received approximately 545 billion pesos for flood control and related infrastructure. In October 2025, DPWH Secretary Vince Dizon confirmed that 421 of 8,000 inspected flood control sites were ghost projects: structures fully paid for that did not physically exist. Finance Secretary Ralph Recto told the Senate that ghost flood control projects produced economic losses of 42.3 billion to 118.5 billion pesos over three years, equivalent to 95,000 to 266,000 foregone jobs (Senate DBCC Briefing, September 2025).

The Commission on Audit, conducting expanded fraud audits from August 2025, used physical inspections, drone surveillance, geotagged photography, and historical satellite imagery to establish non-existence in specific Bulacan cases. In one case, imagery taken 22 days after a contract's effectivity date already showed a pre-existing structure. In another, the structure DPWH representatives identified for auditors was 694 metres from the location specified in the approved engineering plans (COA Fraud Audit Reports, January 2026; GMA News, February 13, 2026).

COA's method works but is complaint-triggered and province-specific. GIST extends it to the full national project inventory, running automatically when a project is marked complete rather than waiting for a scandal to initiate a formal investigation.

---

## Project Questions

1. Which flood control projects show no spectral evidence of construction at or near their reported coordinates?
2. How prevalent is coordinate displacement relative to outright non-construction, and can the two be distinguished from spectral analysis alone?
3. What share of high-ghost-probability projects are affected by satellite data quality issues versus genuine absence of construction signal?

---

## Data Sources

**Dataset 1: DPWH Transparency Portal**

Source: bettergovph/dpwh-transparency-data, HuggingFace. License: CC0 1.0.

Coverage: 248,000 DPWH infrastructure project records with GPS coordinates, contract dates, budget amounts, contractor names, project categories, and completion status. Downloaded via the HuggingFace datasets API and converted to GeoPackage format.

Unit of Analysis: One row per project. The pipeline filters to flood control and drainage projects with status "Completed," coordinates within the Philippine bounding box, and start year 2017 or later.

**Dataset 2: Sentinel-2 SR Harmonized**

Source: COPERNICUS/S2_SR_HARMONIZED via Google Earth Engine. License: Copernicus Open Access.

Coverage: 10-metre resolution surface reflectance imagery from 2017 onward. Cross-sensor calibrated between Sentinel-2A and Sentinel-2B. Accessed through GEE's cloud computing infrastructure without local download.

**Dataset 3: COA Fraud Audit Reports (2025-2026)**

Source: Commission on Audit, filed with the Office of the Ombudsman and the Independent Commission for Infrastructure, September 2025 through February 2026.

Coverage: Confirmed ghost project coordinates in Bulacan province. Used as informal ground truth for evaluating the pipeline's signal patterns against known cases.

---

## Data Dictionary

All variables in the output scoring file (results_scored.csv) unless noted as derived.

| Variable | Type | Description |
|---|---|---|
| project_id | string | DPWH contract ID. Unique identifier. |
| project_name | string | Full project description from the transparency portal. |
| latitude | float | Project coordinate, decimal degrees WGS84. |
| longitude | float | Project coordinate, decimal degrees WGS84. |
| start_year | integer | Year extracted from startDate. Must be >= 2017 for pipeline inclusion. |
| completion_year | integer | Year extracted from completionDate. Defines the after composite window. |
| contract_amount | float | Allocated budget in Philippine pesos. |
| contractor | string | Name of the awarded contractor. |
| province | string | Extracted from the location column. |
| region | string | Administrative region. Used for stratified sampling. |
| tight_ibi | float | IBI delta (after - before) sampled at 30m mean. Primary construction signal. |
| broad_ibi | float | IBI delta sampled at 100m mean. |
| anywhere_ibi | float | IBI delta sampled at 150m max. Catches coordinate displacement up to 150m. |
| tight_ndvi | float | NDVI delta at 30m. Negative value = vegetation removed (construction signal). |
| [window]_[index] | float | Same structure for all 9 indices across all 3 windows. 27 delta columns total. |
| signal_pattern | string | Three-character code (e.g. T+B-A+) based on IBI threshold at each window. |
| signal_interpretation | string | Plain-language description of the signal pattern. |
| ghost_probability | float | Heuristic score 0.0 to 1.0. Higher = more likely ghost. |
| ghost_tier | string | "ghost" (>=0.70), "suspect" (0.40-0.69), or "built" (<0.40). |
| coord_displacement | string | "confirmed_built", "coord_displaced", or "ghost_or_lost_coord". |
| before_scene_count | integer | Number of cloud-free S2 scenes in the before composite window. |
| after_scene_count | integer | Number of cloud-free S2 scenes in the after composite window. |
| composite_valid | boolean | False if either composite failed the pixel-level validity check. |
| data_quality | string | "ok", "sparse_before", "sparse_after", "no_data", or "unknown". |
| before_thumb_url | string | Signed GEE URL for the before-period true colour thumbnail. |
| after_thumb_url | string | Signed GEE URL for the after-period true colour thumbnail. |

Note on scene counts: before_scene_count counts scenes passing the cloud cover filter before pixel-level masking. A composite with 7 scenes can still produce a black image if every pixel at that specific location is cloud-masked. The composite_valid column is the definitive check and samples the mean of B4 over the project area. Projects where composite_valid is False or either scene count is below 3 are capped at a ghost probability of 0.50 and excluded from the ghost and suspect counts in the dashboard.

---

## Analytical Goals

- Identify flood control projects with no spectral evidence of construction at or near their reported coordinates, producing a ranked audit priority list
- Distinguish between ghost projects (T-B-A-) and coordinate-displaced projects (T-B-A+) using the three-window spatial framework
- Flag projects with insufficient satellite data separately from projects with genuine no-signal findings, so data quality issues do not inflate ghost counts
- Surface results in a web dashboard showing before and after satellite thumbnails alongside the spectral signal matrix for each project

---

## Project Structure

```
gist/
|
├── README.md
|
├── data/
|   ├── raw/                    # original .gpkg from DPWH / HuggingFace
|   ├── processed/              # all_projects.csv, flood_projects.csv
|   └── samples/                # sample_500.csv, coord_check.csv
|
├── scripts/
|   ├── 0_data_collection.py    # download dataset from HuggingFace, save as .gpkg
|   ├── 01_load_data.py         # validate coordinates, export pipeline-ready CSVs
|   ├── 02_pipeline.py          # GEE: build composites, compute spectral deltas
|   ├── 03_score.py             # classify signal patterns, ghost probability score
|   └── 04_upload.py            # push results and thumbnails to Supabase
|
├── docs/
|   └── index.html              # GIST dashboard, single-file, GitHub Pages ready
|
├── outputs/
|   ├── results.csv
|   ├── results_scored.csv
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

## Usage

### Step 0: Download the dataset

```bash
python scripts/0_data_collection.py --output data/raw/dpwh_data.gpkg
```

Runs once. Downloads the DPWH transparency dataset from HuggingFace and saves it as a GeoPackage with point geometry attached.

### Step 1: Load and validate data

```bash
python scripts/01_load_data.py --input data/raw/dpwh_data.gpkg
```

Outputs: all_projects.csv, flood_projects.csv, and coord_check.csv (projects with coordinates outside the Philippine bounding box, at null island, or with swapped lat/lon).

### Step 2: Run the GEE pipeline

```bash
# Test on stratified 500-row sample first (95% CI +/-4.3%)
python scripts/02_pipeline.py \
  --input data/processed/flood_projects.csv \
  --output outputs/results.csv \
  --project YOUR_GEE_PROJECT_ID \
  --test \
  --spectral-workers 6 \
  --thumb-workers 5

# Full run
python scripts/02_pipeline.py \
  --input data/processed/flood_projects.csv \
  --output outputs/results.csv \
  --project YOUR_GEE_PROJECT_ID \
  --spectral-workers 6 \
  --thumb-workers 5
```

--test takes a proportional stratified sample by completion_year and region rather than the first N rows, ensuring every year and region is represented. --skip-thumbs skips thumbnail generation for faster spectral-only runs.

### Step 3: Score

```bash
python scripts/03_score.py \
  --input outputs/results.csv \
  --output outputs/results_scored.csv
```

### Step 4: Upload to Supabase

Run the SQL schema from the header of 04_upload.py in the Supabase SQL editor first, then:

```bash
python scripts/04_upload.py \
  --input outputs/results_scored.csv \
  --url https://YOUR_PROJECT.supabase.co \
  --key YOUR_SERVICE_ROLE_KEY \
  --skip-thumbs
```

### Step 5: Deploy the dashboard

Edit dashboard/index.html lines 7-8:

```js
const SUPABASE_URL      = 'https://YOUR_PROJECT.supabase.co'
const SUPABASE_ANON_KEY = 'YOUR_ANON_KEY'
```

Open locally by double-clicking the file, or push to GitHub and enable Pages from the dashboard/ folder.

---

## Methodology

### Composite Construction

GIST builds two cloud-free median composites for each project: one from the dry season before construction started and one from the dry season after the reported completion date. Both use the same November through April window in their respective years. Anchoring both composites to the same seasonal window is essential for the Philippines specifically, where vegetation and moisture indices shift 0.05 to 0.08 on unchanged land during dry season transitions (Simonetti et al., 2021).

The dry season window runs 180 days and typically yields 8 to 16 cloud-free observations. If fewer than 5 scenes pass the cloud cover filter, the pipeline falls back to the full calendar year. A pixel-level validity check then samples the mean of B4 over the project area. If the mean is zero, every pixel at that location was cloud-masked despite the scene count appearing sufficient, and the composite is flagged as invalid.

### Spectral Indices

Nine indices are computed on each composite. The delta (after minus before) is the signal.

| Index | Direction | Weight | Notes |
|---|---|---|---|
| IBI | Positive = construction | 40% primary | Normalises against vegetation and water, robust to dry soil noise |
| TCB | Positive = construction | 30% primary | Sentinel-2 Tasseled Cap Brightness (Shi and Xu, 2019) |
| NDVI | Negative = construction | 10% secondary | Threshold 0.08; seasonal PH shifts reach 0.05-0.08 on unchanged land |
| MNDWI | Negative = construction | 10% secondary | Same higher threshold |
| BUI, TCG, TCW, NDBI, NDTI | Mixed | 10% combined | Supporting evidence |

Vegetation indices are demoted to secondary signals with a higher threshold because seasonal land cover cycles in the Philippines produce IBI-range deltas on land that has not changed. NDBI uses B8A (narrow NIR) rather than B8 to avoid red-edge contamination from vegetation.

### Three Spatial Windows

COA fraud audit reports repeatedly documented DPWH representatives leading inspectors to project sites that did not match approved engineering plans. A single fixed buffer cannot handle this. GIST samples the spectral delta at three independent spatial windows whose results are never averaged. Their pattern of agreement is the diagnostic.

| Window | Buffer | Reducer | Question |
|---|---|---|---|
| Tight | 30m | Mean | Did construction occur at the reported coordinate? |
| Broad | 100m | Mean | Did construction occur in the general vicinity? |
| Anywhere | 150m | Max | Did construction occur anywhere within 150m? |

Key signal patterns:

| Pattern | Interpretation | Recommended Action |
|---|---|---|
| T+B+A+ | Built at coordinate, construction confirmed | Low priority |
| T-B-A+ | Coordinate displaced; something built within 150m | Verify GPS before re-inspection |
| T-B-A- | No signal anywhere | Ghost candidate or displacement >150m; field check required |
| T+B-A- | Small structure or noise at coordinate | Manual review |

### Scoring

The heuristic scorer combines a pattern-based anchor (60% weight) with a weighted index score (40% weight). IBI and TCB are the primary signals. Displacement caps prevent over-penalising projects where something was clearly built nearby. Projects with T-B-A+ are capped at 0.60; projects with confirmed IBI signal at the coordinate are capped at 0.35. Projects with invalid composites are capped at 0.50 and excluded from ghost and suspect counts in the dashboard.

---

## Limitations

**What the pipeline cannot detect**

Excavation depth is not observable from orbit. A structure built at half the specified depth produces the same surface spectral signature as a correctly-built structure after backfill. This is likely the most prevalent fraud type in Philippine flood control cases and is not caught here.

Material quality, concrete mix ratio, rebar grade, and compaction density all require physical testing and cannot be assessed from Sentinel-2 imagery.

Coordinate displacement greater than 150 metres produces a T-B-A- pattern indistinguishable from outright non-construction in spectral analysis. The three-window approach handles displacement up to 150 metres; beyond that, field verification is required.

Projects started before mid-2017 may fall outside the Sentinel-2 SR archive, producing empty before composites. The data quality gate flags these and caps their scores rather than marking them as ghost.

**Scorer limitations**

The current scorer is a heuristic rule-based system calibrated for audit triage, not a trained classifier. Ghost probability scores should be read as a ranked priority list for inspection, not as a legal determination of fraud. A trained classifier using COA's confirmed fraud audit coordinates as structured ground truth will replace the heuristic once sufficient labelled data is available.

**Sample limitations**

Results to date cover a stratified sample of approximately 500 of the full flood control project inventory. Ghost rate estimates from the sample are indicative, not definitive.

---

## Ethical Considerations

This pipeline is designed for audit triage, not for public naming of contractors or project officials as fraudulent. Ghost probability is a flag that a project warrants inspection, not a finding of wrongdoing. All output scores should be read alongside the signal pattern and the data quality flag before any conclusion is drawn.

The DPWH transparency dataset is public record published under CC0. No personally identifiable information is present. Contractor names that appear in the dataset are drawn from public procurement records.

The pipeline produces a ranked list intended for use by government audit institutions. It does not replace physical inspection or the formal fraud audit process. Any project flagged by GIST as a ghost candidate requires independent physical verification before any enforcement action.

---

## References

- Commission on Audit. Fraud Audit Reports: DPWH Bulacan 1st DEO Flood Control Projects. Filed with the Office of the Ombudsman and ICI, September 2025 through February 2026.
- Development Budget Coordination Committee. Senate Committee on Finance Briefing, September 2025. Reported by GMA News Online.
- Dizon, V. Press briefing at the Independent Commission for Infrastructure, October 9, 2025. Reported by Philippine Star.
- Independent Commission for Infrastructure. Created under EO No. 94, September 15, 2025. Dissolved March 31, 2026. Reported by Filipino Times and Rappler.
- Lacaux, J.P., Tourre, Y.M., Vignolles, C., Ndione, J.A., and Lafaye, M. (2007). Classification of ponds from high-spatial resolution remote sensing. Remote Sensing of Environment, 106(1), 66-74.
- Shi, T., and Xu, H. (2019). Derivation of Tasseled Cap Transformation Coefficients for Sentinel-2 MSI. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(10), 4038-4048.
- Simonetti, D., Pimple, U., Langner, A., and Marelli, A. (2021). Pan-Tropical Sentinel-2 Cloud-Free Annual Composite Datasets. Data in Brief, 39, 107488.
- Xu, H. (2006). Modification of normalised difference water index (NDWI) to enhance open water features in remotely sensed imagery. International Journal of Remote Sensing, 27(14), 3025-3033.
- Xu, H. (2008). A new index for delineating built-up land features in satellite imagery. International Journal of Remote Sensing, 29(14), 4269-4276.

---

## Acknowledgements

Dataset sourced from the DPWH Transparency Portal via [bettergovph/dpwh-transparency-data](https://huggingface.co/datasets/bettergovph/dpwh-transparency-data) (CC0 1.0). Thanks to [@csiiiv](https://github.com/csiiiv) for the dpwh-transparency-data-api-scraper that enabled systematic data collection from the DPWH Transparency API.
