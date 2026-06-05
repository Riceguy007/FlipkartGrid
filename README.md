# 🚦 Traffic Demand Prediction — Flipkart Grid Hackathon

**Score achieved: ~99.70 / 100** (R² ≈ 0.997)

## Problem Statement
Predict traffic demand at geographic locations (geohash) across different timestamps to help cities reduce congestion and improve urban mobility.

## Dataset
| File | Shape | Description |
|------|-------|-------------|
| `train.csv` | 77299 × 11 | Training data with demand labels |
| `test.csv` | 41778 × 10 | Test data (no demand column) |
| `submission.csv` | 41778 × 2 | Final predictions |

**Features:** `geohash`, `day`, `timestamp`, `RoadType`, `NumberofLanes`, `LargeVehicles`, `Landmarks`, `Temperature`, `Weather`

**Target:** `demand` (float, 0–1) — traffic demand at that location and time

**Metric:** `score = max(0, 100 * r2_score(actual, predicted))`

## Approach

### Key Insight
Traffic demand at a specific **location × time-of-day** is highly consistent.  
By encoding the historical mean demand for every `geohash × time_slot` combination, the model can almost perfectly predict demand.

### Feature Engineering
- **Timestamp parsing** → `hour`, `minute`, `time_slot` (0–95, 15-min intervals)
- **Cyclical encoding** → `hour_sin/cos`, `slot_sin/cos`
- **Peak hour flags** → `is_morning_peak`, `is_evening_peak`, `is_night`, `is_noon`
- **Geohash hierarchy** → `geo3`, `geo4`, `geo5`, `geo6` (prefixes for spatial grouping)
- **Temperature imputation** → filled by weather-group median

### Target Encodings (Most Powerful Features)
| Grouping | What it captures |
|----------|-----------------|
| `geohash` | Baseline demand per location |
| `geohash × time_slot` | ⭐ Location demand at each 15-min window |
| `geohash × hour` | Location demand by hour |
| `geohash × day` | Day-level location patterns |
| `geo4 × time_slot` | Area-level time patterns |
| `time_slot` | Global time-of-day pattern |
| `RoadType × time_slot` | Road-type time patterns |

### Models (Ensemble)
| Model | OOF R² | Score |
|-------|--------|-------|
| HistGradientBoosting (Config A) | 0.99612 | 99.61 |
| HistGradientBoosting (Config B) | 0.99615 | 99.62 |
| ExtraTreesRegressor | 0.99709 | 99.71 |
| **Weighted Blend** | **0.99700** | **99.70** |

Pure sklearn — no external dependencies (libomp-free).

## How to Run

```bash
# Install dependencies
pip install pandas scikit-learn numpy

# Run the python script
python solution.py

# Or run the Jupyter Notebook:
# Open solution.ipynb in Jupyter/VS Code and execute all cells.
```

## File Structure
```
FlipkartGrid/
├── solution.py                 # Full ML pipeline script
├── solution.ipynb              # Jupyter Notebook version (required for submission)
├── submission.csv              # Final predictions (41778 x 2, Index and demand)
├── FlipkartGrid_Submission.zip # Re-packaged ZIP for hackathon source upload
├── README.md                   # Project overview and run instructions
└── approach.txt                # ML approach details
```
