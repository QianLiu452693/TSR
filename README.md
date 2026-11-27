# Crime Scene Temporal Knowledge Graph (Crime TKG) and Spatiotemporal Risk Reasoning

A crime-scene temporal knowledge graph (TKG) built from standardized crime records, with multi-scale spatiotemporal risk reasoning evaluated under temporal hold-out validation. The graph can be exported in a Neo4j-importable format and supports batch incremental updates.

- **Dataset**: Figshare, DOI [10.6084/m9.figshare.28106939](https://doi.org/10.6084/m9.figshare.28106939)
- **Final scale**: 302,710 entities and 1,916,810 quadruples
- **Reasoning protocol**: temporal hold-out validation, compared against a near-repeat baseline

---

## Table of Contents

- [Overview](#overview)
- [Dataset and Cleaning Funnel](#dataset-and-cleaning-funnel)
- [Spatiotemporal Alignment and Grid Coding](#spatiotemporal-alignment-and-grid-coding)
- [Graph Structure](#graph-structure)
  - [Entity Types](#entity-types)
  - [Relation Types](#relation-types)
- [Repository Structure](#repository-structure)
- [Requirements and Installation](#requirements-and-installation)
- [Quick Start](#quick-start)
- [Neo4j Import](#neo4j-import)
- [Incremental Update Mechanism](#incremental-update-mechanism)
- [Spatiotemporal Reasoning Model](#spatiotemporal-reasoning-model)
- [Key Results](#key-results)
- [Output Files](#output-files)
- [Limitations and Notes](#limitations-and-notes)
- [Citation](#citation)


---

## Overview

This project cleans and aligns raw crime records into standardized events, extracts crime-type, geographic, and temporal entities, constructs quadruples by rule-based extraction, and corrects edges with graph-based neighborhood aggregation. The resulting temporal knowledge graph is importable into a graph database and supports incremental updates. On top of the graph, a multi-scale spatiotemporal risk reasoning model is trained and evaluated under temporal hold-out to assess hotspot coverage.

> Causal-edge re-estimation is computed by NumPy neighborhood aggregation, **not** by a deep graph neural network. The reasoning model uses shared features with per-task logistic regression heads; see [Spatiotemporal Reasoning Model](#spatiotemporal-reasoning-model).

## Dataset and Cleaning Funnel

The pipeline starts from 1,030,854 raw records. After deduplication, outlier removal, and field completion, 724,399 records with complete fields remain.

| Stage | Rule | Count |
| --- | --- | ---: |
| Raw records | — | 1,030,854 |
| Case-number deduplication | Remove duplicates by case number | 65,999 |
| Temporal outliers | Unparseable, or year outside 2000–2019 | 51,309 |
| Irrelevant offense types | Non-street / non-community crime types | 99,938 |
| Isolated points | DBSCAN outliers | 16,940 |
| Administrative division | Point-in-polygon match against county-level boundaries | 703,955 |
| Administrative backfill | Backfilled by nearest neighbor and court name | 20,444 |
| Offense completion | Lexicon matching, supplemented by TF-IDF | 9,393 |
| Manual review list | Insufficient similarity, sent to manual review | 4,813 |
| **Cleaned** | **Records with complete fields** | **724,399** |

- The 4,813 offense records with insufficient similarity are kept in `manual_review_crime_type.csv`.

## Spatiotemporal Alignment and Grid Coding

- **Time standardization**: Unified to ISO 8601 format `YYYY-MM-DDThh:mm:ss+08:00`, with an accompanying Unix timestamp.
- **Coordinate projection**: After Gauss–Krüger projection, 500 m grids are defined within each prefecture-level city.
- **Grid code**: 12-digit code, structured as `6-digit prefecture code + 3-digit X + 3-digit Y`, yielding 209,315 grids.

## Graph Structure

Each quadruple contains: `entity1_id`, `relation_type`, `entity2_id`, `iso8601_timestamp`, `confidence`.

### Entity Types

Only entities actually appearing in the data are retained; **empty grids are not synthesized**.

| Category | Count | Composition |
| --- | ---: | --- |
| Crime types | 18 | Property crimes 7, crimes against personal rights 6, public-order offenses 5 |
| Geospatial | 212,362 | Province 27, prefecture 328, county 2,692, 500 m urban grid 209,315 |
| Time units | 90,330 | Hour 82,374, day 6,674, week 1,042, month 240 |

Entity attributes:

- Each offense type is annotated with its criminal-law article, constitutive elements, and secondary category. For example, theft (larceny) corresponds to Criminal Law Article 264 with 595,453 cases; robbery, snatch, and fraud are likewise written according to the specific provisions.
- Grids store the Gauss–Krüger coordinate bounds and per-square-kilometer case density (case count / 0.25). The source table has no population denominator, so **population density is not filled in**.
- Hour and day entities carry time-of-day, weekend, and holiday markers.

### Relation Types

| Relation group | Sub-relation | Count |
| --- | --- | ---: |
| Spatiotemporal | Occurs at time | 724,399 |
| Spatiotemporal | Co-located in same grid | 724,399 |
| Spatiotemporal | Spatial hierarchy membership | 212,335 |
| Spatiotemporal | Adjacent region association | 138,231 |
| Spatiotemporal | Temporal granularity membership | 95,722 |
| Spatiotemporal | Short-term succession within 24 h | 21,136 |
| Causal | Geographic crime generation | 441 |
| Type | Type exclusion | 122 |
| Type | Type transition | 14 |
| Type | Type similarity | 11 |

Confidence and re-estimation rules:

- Rule edges have confidence 1.0; short-term succession edges have confidence 0.98.
- Geographic crime-generation edges are first extracted by co-occurrence lift, then re-estimated by two-layer grid neighborhood mean aggregation; confidence is the average of the rule result and the neighborhood result. After neighborhood re-estimation, 430 rule edges are retained and 11 edges are supplemented by machine learning; 41 edges with confidence below 0.35 are removed.
- **Directed cross-type causal edges are not written.** Among 24-hour succession transitions, no type pair simultaneously satisfies: count ≥ 30, forward direction ≥ 1.2× the reverse, and lift ≥ 1.5. The closest pair, snatch and robbery, has only 31 and 33 occurrences on the two sides with near-symmetric directionality, so it is kept only in type relations.

## Repository Structure

```text
output/
├── cleaned_crime_records.csv        # Standardized event table
├── entities.csv                    # Entity dictionary
├── tkg_quadruples.csv              # Neo4j-importable quadruples
├── manual_review_crime_type.csv    # Manual review list for offenses
└── tkg/
    ├── crime_tkg.sqlite             # Graph store (SQLite)
    ├── tkg_report.json              # Graph scale statistics
    ├── neo4j/                       # Node/relation CSVs and import.cypher
    └── reason/
        ├── reason_report.json        # Full reasoning metrics
        ├── hotspot_24h.csv          # 24 h high-risk hotspot list
        ├── hotspot_7d.csv             # 7-day high-risk hotspot list
        ├── hotspot_30d.csv            # 30-day high-risk hotspot list
        ├── type_C0101_30d.csv        # Theft-specific 30-day list
        └── explanations.csv           # Historical evidence for each entry
```

## Requirements and Installation

- Python 3 (launch with `py -3` on Windows).
- Main dependencies: NumPy, scikit-learn (DBSCAN, TF-IDF, logistic regression, etc.); pinned versions are maintained in the repository `requirements.txt`.
- Neo4j is optional, required only for graph-database visualization and querying; Neo4j was not started on this machine.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Recompute the spatiotemporal reasoning results:

```bash
py -3 run_tkg_reason.py
```

Results are written to `output/tkg/reason/`; full metrics are in `reason_report.json`.

## Neo4j Import

Nodes, relations, and the import script are located in `output/tkg/neo4j/`:

1. Copy the CSVs in this directory to the Neo4j `import` directory;
2. Run the import script:

```bash
cypher-shell -f import.cypher
```

The script creates unique entity constraints, as well as indexes for hierarchy, offense codes, spatial hierarchy, temporal granularity, and relation time.

## Incremental Update Mechanism

Updates follow **rule matching → conflict rejection → layered writing**.

### Input Requirements

The new-case CSV must contain already standardized records with at least the following fields:

| Field | Description |
| --- | --- |
| `case_number` | Case identifier |
| `crime_type_id` | Offense entity identifier |
| `grid_id` | Grid identifier |
| `time_id` | Time-unit identifier |
| `iso8601` | ISO 8601 timestamp |
| `unix_timestamp` | Unix timestamp |
| `city_code` | Prefecture-level code |
| `grid_x` | Grid X coordinate code |
| `grid_y` | Grid Y coordinate code |

### Conflict and Writing Rules

- Unknown offense type or unknown grid: recorded in the conflict table and rejected.
- Existing case number: skipped.
- New time units, new edges, and `event_count` changes: written separately.
- Attribute changes are kept in `attribute_history`; batch records are kept in `update_log`.

### Commands

```bash
# Import a specified new-case file
py -3 run_tkg_update.py --input <new_cases.csv>

# Update on a scheduled cycle
py -3 run_tkg_update.py --schedule week

# Run scheduled update only when due
py -3 run_tkg_update.py --input <new_cases.csv> --if-due
```

`--schedule` accepts: `day`, `week`, `month`, `manual`.

> The update pipeline was validated with 5 duplicate cases and 3 cases shifted by 5 hours: the 5 duplicates were skipped, and the 3 inserted cases have been withdrawn from the store; the current database still holds 724,399 real cases.

## Spatiotemporal Reasoning Model

### Experimental Setup

- **Validation protocol**: temporal hold-out; training uses only cases before the cutoff point, compared against a near-repeat baseline (ranked by case count in the previous 30 days).
- **Evaluation cutoff**: 2019-08-13. The 30 days before this date still contain at least 1,000 cases; after that point document volume drops to dozens per month and cannot serve as a prediction window.
- **Test windows**: three 30-day windows starting 2019-05-15, with 2,740, 1,953, and 1,013 cases respectively.
- **High-risk tiers**: ranked by grid, top 1% for 24 h, top 2% for 7 days, and top 5% for 30 days.

### Features and Model

- Features: counts at grid itself, four-neighborhood, and county levels; night / weekend / holiday proportions; 14-day exponentially decayed near-repeat intensity.
- The three tasks share these features, each with its own logistic regression head; coefficients for recent case counts are constrained to be non-negative.
- In the 30-day model, the largest standardized weight is the share of theft in the last 90 days.
- The source data has no population density, traffic flow, or suspect identifiers; these are **not** included in the model.

## Key Results

### Risk Coverage Comparison

| Task | Model coverage | Previous-30-day case-count baseline |
| --- | ---: | ---: |
| 24 h | 22.7% | 16.1% |
| 7 days | 21.5% | 18.3% |
| 30 days | 36.9% | 20.5% |
| Theft · 7 days | 27.2% | 17.6% |
| Theft · 30 days | 37.6% | 20.1% |

- The theft-specific list is more concentrated than the all-offense list. Example: grid `GG330700197171` in Dongyang, Jinhua, had 24 thefts in the 90 days before decision-making, and 7 more within the held-out 30 days.
- Fraud and intentional injury have only dozens to tens of cases per month in the held-out period, with unstable coverage; **they cannot be treated as standalone early-warning results**.

### Other Measured Findings (deviating from planned targets; measured results prevail)

- Parameters reduced from 17 to 3, a 5.67× compression; after teacher-score distillation, 30-day coverage drops from 36.9% to 26.8%; a lightweight model retrained with only two features reaches 9.6% coverage.
- The weekly gated-recurrent model predicts theft with MAE of 79 cases; carrying over the previous week gives 31 cases, and the self-gated recurrent model does not beat the previous-week baseline.
- Theft weekly average peaks in October (1,833) and bottoms in February (1,416); fraud peaks in May with no November rise; intentional injury peaks in June.
- After stratifying by historical case volume, the night-time case share yields a 30-day theft risk ratio of 1.00; after propensity-score stratification it is 0.98. After removing the national weekly total, the weekly residual correlation between theft and intentional injury is -0.71.
- There are 3,651 cross-county adjacent grid pairs; 28 have at least 5 24-hour near-repeats in training. The strongest are Nanchang Xihu → Donghu (12) and Wuhan Jiang'an → Jianghan (11). These 20 strong edges did not reappear in the following 90 days. County offense structures are very close; cross-province cosine similarities are almost all above 0.99, with no distinguishable latent association.

## Output Files

| Path | Description |
| --- | --- |
| `output/cleaned_crime_records.csv` | Standardized event table |
| `output/entities.csv` | Entity dictionary |
| `output/tkg_quadruples.csv` | Neo4j-importable quadruples |
| `output/tkg/crime_tkg.sqlite` | Graph store (SQLite) |
| `output/tkg/tkg_report.json` | Graph scale statistics |
| `output/tkg/neo4j/` | Node/relation CSVs and `import.cypher` |
| `output/tkg/reason/reason_report.json` | Full reasoning metrics |
| `output/tkg/reason/hotspot_24h.csv` | 24 h high-risk hotspot list |
| `output/tkg/reason/hotspot_7d.csv` | 7-day high-risk hotspot list |
| `output/tkg/reason/hotspot_30d.csv` | 30-day high-risk hotspot list |
| `output/tkg/reason/type_C0101_30d.csv` | Theft-specific 30-day list |
| `output/tkg/reason/explanations.csv` | Historical evidence for each entry |

## Limitations and Notes

- Only objects actually appearing in the data are retained; empty grids are not synthesized.
- The source data lacks population, traffic, and suspect identifiers; these fields are not fabricated.
- Causal-edge re-estimation is NumPy neighborhood aggregation, not deep GNN training.
- Neo4j was not started locally; the import steps require a configured Neo4j environment.
- Types with insufficient samples, such as fraud and intentional injury, have unstable coverage and should not be used as standalone early-warning conclusions.

## Citation

Data source:

```text
Figshare Dataset. DOI: 10.6084/m9.figshare.28106939
https://doi.org/10.6084/m9.figshare.28106939
```



The license for this repository is governed by the `LICENSE` file in the root directory; if none is provided, add an appropriate open-source license before public distribution.
