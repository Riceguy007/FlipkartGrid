import sys
import os
import pandas as pd
import numpy as np

# Add FlipkartGrid folder to python path
sys.path.append('/Users/nishantakamalbaruah/Desktop/Flipkart gridlock/FlipkartGrid')

from solution import TrafficDemandPipeline

# 1. Create simulated train set with some missing values
train_data = {
    'Index': range(100),
    'geohash': ['qp02z1', 'qp02zt', 'qp08bj', 'qp08gt'] * 25,
    'day': [48, 49, 50, 51] * 25,
    'timestamp': ['0:0', '1:15', '2:30', '3:45'] * 25,
    'RoadType': ['Residential', 'Street', 'Highway', None] * 25,
    'NumberofLanes': [1, 2, 3, 4] * 25,
    'LargeVehicles': ['Allowed', 'Not Allowed', 'Allowed', None] * 25,
    'Landmarks': ['Yes', 'No', 'Yes', None] * 25,
    'Temperature': [30.0, 31.0, 25.0, np.nan] * 25,
    'Weather': ['Sunny', 'Rainy', 'Foggy', 'Snowy'] * 25,
    'demand': np.random.rand(100)
}
train_df = pd.DataFrame(train_data)

# 2. Create simulated test set with:
# - Unseen geohash ('qp9999')
# - Missing columns ('Landmarks')
# - Different timestamp formats ('2026-06-04 17:15:00' and '18:30')
# - Missing values (NaNs)
test_data = {
    'Index': range(10),
    'geohash': ['qp02z1', 'qp9999'] * 5,  # qp9999 is unseen!
    'day': [48, 52] * 5,
    'timestamp': ['2026-06-04 17:15:00', '18:30'] * 5,
    'RoadType': ['Residential', 'Highway'] * 5,
    'NumberofLanes': [2, np.nan] * 5,
    'LargeVehicles': ['Allowed', None] * 5,
    'Temperature': [np.nan, 20.0] * 5,
    'Weather': ['Sunny', 'Sunny'] * 5
}
test_df = pd.DataFrame(test_data)

print("Simulated DataFrames initialized successfully.")

# Run pipeline
pipeline = TrafficDemandPipeline(
    n_splits=3,
    smoothing=2,
    outlier_treatment='none',
    et_estimators=10,
    use_extra_trees=True
)

print("Fitting pipeline on simulated train data...")
pipeline.fit(train_df, target='demand')
print("Pipeline fitted successfully.")

print("Predicting on simulated test data...")
preds = pipeline.predict(test_df)
print("Predictions shape:", preds.shape)
print("Predictions:", preds)

assert len(preds) == 10, "Predictions size mismatch!"
assert not np.isnan(preds).any(), "NaNs detected in predictions!"
print("\n[SUCCESS] Test completed successfully! The pipeline handles unseen categories and missing columns/values perfectly.")
