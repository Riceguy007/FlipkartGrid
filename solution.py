"""
Traffic Demand Prediction - Hackathon Solution
Pure sklearn: HistGradientBoostingRegressor ensemble (no libomp needed)
Evaluation: score = max(0, 100 * r2_score(actual, predicted))
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

# -------------------------------------------------------
# DATA_DIR: Path to the folder containing train.csv and test.csv.
# The script auto-detects the dataset location by checking
# three common places in order. You can also set DATA_DIR
# manually here if your files are somewhere else.
# DATA_DIR = r'C:\path\to\your\dataset'  # <-- uncomment and edit if needed
# -------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # folder of this .py file

CANDIDATE_DIRS = [
    os.path.join(SCRIPT_DIR, 'dataset'),           # dataset/ next to script
    os.path.join(SCRIPT_DIR, '..', 'dataset'),     # ../dataset/ (one level up)
    SCRIPT_DIR,                                     # same folder as script
    os.getcwd(),                                    # current working directory
]

DATA_DIR = None
for candidate in CANDIDATE_DIRS:
    if os.path.exists(os.path.join(candidate, 'train.csv')):
        DATA_DIR = candidate
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find train.csv. Please place train.csv and test.csv in the "
        "same folder as this script, or in a 'dataset/' subfolder next to it."
    )

print(f"Using dataset from: {os.path.abspath(DATA_DIR)}")
print("Loading data...")
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
print(f"Train: {train.shape}, Test: {test.shape}")

test_index = test['Index'].values

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()

    # Timestamp parsing
    df['hour']      = df['timestamp'].apply(lambda x: int(str(x).split(':')[0]))
    df['minute']    = df['timestamp'].apply(lambda x: int(str(x).split(':')[1]))
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15    # 0–95

    # Cyclical time
    df['hour_sin']  = np.sin(2 * np.pi * df['hour']      / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour']      / 24)
    df['slot_sin']  = np.sin(2 * np.pi * df['time_slot'] / 96)
    df['slot_cos']  = np.cos(2 * np.pi * df['time_slot'] / 96)

    # Day features
    df['day_mod7']         = df['day'] % 7
    df['is_weekend']       = (df['day_mod7'] >= 5).astype(int)
    df['is_morning_peak']  = ((df['hour'] >= 7)  & (df['hour'] <= 9)).astype(int)
    df['is_evening_peak']  = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_night']         = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_noon']          = ((df['hour'] >= 11) & (df['hour'] <= 13)).astype(int)
    df['is_early_morning'] = ((df['hour'] >= 5)  & (df['hour'] <= 7)).astype(int)

    # Geohash prefix hierarchy
    df['geo3'] = df['geohash'].str[:3]
    df['geo4'] = df['geohash'].str[:4]
    df['geo5'] = df['geohash'].str[:5]
    df['geo6'] = df['geohash']

    # Binary features
    df['LargeVehicles_enc'] = (df['LargeVehicles'] == 'Allowed').astype(float)
    df['Landmarks_enc']     = (df['Landmarks'] == 'Yes').astype(float)

    # Road type ordinal
    road_map = {'Residential': 1.0, 'Street': 2.0, 'Highway': 3.0}
    df['RoadType_enc'] = df['RoadType'].map(road_map)   # NaN stays NaN for HGBR

    # Weather ordinal
    weather_map = {'Sunny': 1.0, 'Rainy': 2.0, 'Foggy': 3.0, 'Snowy': 4.0}
    df['Weather_enc'] = df['Weather'].map(weather_map)  # NaN stays NaN for HGBR

    # Temperature: fill by weather group median
    df['Temperature'] = df.groupby('Weather')['Temperature'].transform(
        lambda x: x.fillna(x.median())
    )
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())

    return df

print("Engineering features...")
train = engineer_features(train)
test  = engineer_features(test)

# ─────────────────────────────────────────────
# 3. TARGET ENCODING (using full train)
# ─────────────────────────────────────────────
print("Computing target encodings...")

def add_target_enc(train, test, group_col, target='demand', prefix=None):
    if prefix is None:
        prefix = group_col if isinstance(group_col, str) else '_'.join(str(c) for c in group_col)
    stats = train.groupby(group_col)[target].agg(['mean', 'std', 'median', 'count']).reset_index()
    cols  = group_col if isinstance(group_col, list) else [group_col]
    stats.columns = cols + [f'{prefix}_mean', f'{prefix}_std', f'{prefix}_median', f'{prefix}_count']
    train_out = train.merge(stats, on=group_col, how='left')
    test_out  = test.merge(stats, on=group_col, how='left')
    return train_out, test_out

# Geohash-level (strongest signal — location = main demand driver)
train, test = add_target_enc(train, test, 'geo6',  prefix='geo6')
train, test = add_target_enc(train, test, 'geo5',  prefix='geo5')
train, test = add_target_enc(train, test, 'geo4',  prefix='geo4')
train, test = add_target_enc(train, test, 'geo3',  prefix='geo3')

# Spatiotemporal: geohash × time_slot (MOST POWERFUL)
train, test = add_target_enc(train, test, ['geo6', 'time_slot'],  prefix='geo6_slot')
train, test = add_target_enc(train, test, ['geo6', 'hour'],       prefix='geo6_hour')
train, test = add_target_enc(train, test, ['geo6', 'day'],        prefix='geo6_day')
train, test = add_target_enc(train, test, ['geo5', 'time_slot'],  prefix='geo5_slot')
train, test = add_target_enc(train, test, ['geo4', 'time_slot'],  prefix='geo4_slot')
train, test = add_target_enc(train, test, ['geo4', 'hour'],       prefix='geo4_hour')
train, test = add_target_enc(train, test, ['geo4', 'day'],        prefix='geo4_day')
train, test = add_target_enc(train, test, ['geo3', 'time_slot'],  prefix='geo3_slot')

# Time-only patterns
train, test = add_target_enc(train, test, 'time_slot', prefix='slot')
train, test = add_target_enc(train, test, 'hour',      prefix='hour')
train, test = add_target_enc(train, test, 'day',       prefix='day')

# Road/weather patterns
train, test = add_target_enc(train, test, ['RoadType_enc', 'time_slot'], prefix='road_slot')
train, test = add_target_enc(train, test, ['RoadType_enc', 'hour'],      prefix='road_hour')
train, test = add_target_enc(train, test, ['Weather_enc',  'time_slot'], prefix='wx_slot')
train, test = add_target_enc(train, test, ['NumberofLanes','time_slot'], prefix='lanes_slot')

# Geohash + road combination
train, test = add_target_enc(train, test, ['geo4', 'RoadType_enc'], prefix='geo4_road')
train, test = add_target_enc(train, test, ['geo4', 'NumberofLanes'], prefix='geo4_lanes')

print(f"  Train shape after encodings: {train.shape}")

# ─────────────────────────────────────────────
# 4. LABEL ENCODE STRING COLUMNS
# ─────────────────────────────────────────────
str_cols = ['geo3', 'geo4', 'geo5', 'geo6', 'timestamp', 'geohash']
all_df   = pd.concat([train, test], axis=0).reset_index(drop=True)

for col in str_cols:
    le = LabelEncoder()
    all_df[col] = le.fit_transform(all_df[col].astype(str))

train_len = len(train)
train = all_df.iloc[:train_len].copy()
test  = all_df.iloc[train_len:].copy()

# ─────────────────────────────────────────────
# 5. DEFINE FEATURE COLUMNS
# ─────────────────────────────────────────────
drop_cols = ['Index', 'demand', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
feature_cols = [c for c in train.columns if c not in drop_cols]
print(f"\nTotal features: {len(feature_cols)}")

X      = train[feature_cols].values.astype(np.float64)
y      = train['demand'].values.astype(np.float64)
X_test = test[feature_cols].values.astype(np.float64)

# ─────────────────────────────────────────────
# 6. MODEL A — HistGradientBoostingRegressor (fast, LightGBM-class)
# ─────────────────────────────────────────────
print("\n--- Training HistGradientBoosting - Config A ---")

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_a   = np.zeros(len(X))
pred_a  = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    m = HistGradientBoostingRegressor(
        loss='squared_error',
        learning_rate=0.03,
        max_iter=2000,
        max_leaf_nodes=255,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=42,
        verbose=0,
    )
    m.fit(X[tr_idx], y[tr_idx])
    oof_a[val_idx]  = m.predict(X[val_idx])
    pred_a         += m.predict(X_test) / N_FOLDS
    r2 = r2_score(y[val_idx], oof_a[val_idx])
    print(f"  Fold {fold+1}: R2={r2:.5f}  score={max(0,100*r2):.2f}")

r2_a = r2_score(y, oof_a)
print(f"HGB-A  OOF R2={r2_a:.5f}  ->  {max(0,100*r2_a):.2f}/100")

# ─────────────────────────────────────────────
# 7. MODEL B — HistGradientBoosting (deeper, more trees)
# ─────────────────────────────────────────────
print("\n--- Training HistGradientBoosting - Config B ---")

oof_b   = np.zeros(len(X))
pred_b  = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    m = HistGradientBoostingRegressor(
        loss='squared_error',
        learning_rate=0.01,
        max_iter=5000,
        max_leaf_nodes=127,
        min_samples_leaf=30,
        l2_regularization=0.05,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=75,
        random_state=123,
        verbose=0,
    )
    m.fit(X[tr_idx], y[tr_idx])
    oof_b[val_idx]  = m.predict(X[val_idx])
    pred_b         += m.predict(X_test) / N_FOLDS
    r2 = r2_score(y[val_idx], oof_b[val_idx])
    print(f"  Fold {fold+1}: R2={r2:.5f}  score={max(0,100*r2):.2f}")

r2_b = r2_score(y, oof_b)
print(f"HGB-B  OOF R2={r2_b:.5f}  ->  {max(0,100*r2_b):.2f}/100")

# ─────────────────────────────────────────────
# 8. MODEL C — ExtraTreesRegressor (different bias, good ensemble diversity)
# ─────────────────────────────────────────────
print("\n--- Training ExtraTrees ---")

oof_c   = np.zeros(len(X))
pred_c  = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    m = ExtraTreesRegressor(
        n_estimators=500,
        max_features=0.6,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    m.fit(X[tr_idx], y[tr_idx])
    oof_c[val_idx]  = m.predict(X[val_idx])
    pred_c         += m.predict(X_test) / N_FOLDS
    r2 = r2_score(y[val_idx], oof_c[val_idx])
    print(f"  Fold {fold+1}: R2={r2:.5f}  score={max(0,100*r2):.2f}")

r2_c = r2_score(y, oof_c)
print(f"ExtraT OOF R2={r2_c:.5f}  ->  {max(0,100*r2_c):.2f}/100")

# ─────────────────────────────────────────────
# 9. ENSEMBLE (weighted by OOF R²)
# ─────────────────────────────────────────────
# Only include models with positive R²
models = [(r2_a, pred_a, oof_a, 'HGB-A'),
          (r2_b, pred_b, oof_b, 'HGB-B'),
          (r2_c, pred_c, oof_c, 'ExtraT')]

total_r2 = sum(r for r, _, _, _ in models if r > 0)
w_list   = [(r / total_r2 if r > 0 else 0) for r, _, _, _ in models]

print(f"\nBlend weights - ", end='')
for (r2, _, _, name), w in zip(models, w_list):
    print(f"{name}:{w:.3f} ", end='')
print()

oof_blend  = sum(w * oof  for (_, _, oof, _), w in zip(models, w_list))
pred_blend = sum(w * pred for (_, pred, _, _), w in zip(models, w_list))

blend_r2 = r2_score(y, oof_blend)
print(f"Blended OOF R2={blend_r2:.5f}  ->  {max(0,100*blend_r2):.2f}/100")

# Clip to valid range
pred_blend = np.clip(pred_blend, 0.0, 1.0)

# ---------------------------------------------
# 10. SAVE SUBMISSION
# ---------------------------------------------
submission = pd.DataFrame({'Index': test_index, 'demand': pred_blend})
out_path   = 'submission.csv'
submission.to_csv(out_path, index=False)

print(f"\n[SUCCESS] Submission saved -> {out_path}")
print(submission.head(10).to_string())
print(f"\n[SCORE] Predicted hackathon score = {max(0, 100*blend_r2):.2f} / 100")
