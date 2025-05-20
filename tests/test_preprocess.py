import pytest
import pandas as pd
import numpy as np
from pathlib import Path  
from src.data_preprocess import partition_data, preprocess_data
import warnings

""" Run the following command in CLI to run all tests: 
        pytest tests/test_preprocess.py -v
    To clear the cache: 
        pytest --cache-clear
"""

@pytest.fixture
def raw_data_path():
    return "data/raw/creditcard.csv"

# ------- Test File Saves and  -------

def test_preprocess_data_saves_files(raw_data_path):

    # Run preprocessing
    _ = preprocess_data(data_path=raw_data_path)
    
    # Check files exist
    assert Path("data/preprocess/X_train.parquet").exists()
    assert Path("data/preprocess/y_test.parquet").exists()
    assert Path("data/preprocess/scaler_params.json").exists()

# ------- Test Data Transforms -------

def test_preprocess_transforms(raw_data_path):
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*_check_n_features.*")
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*_check_feature_names.*")

    X_train, _, y_train, _ = preprocess_data(data_path=raw_data_path)
  
    # 1. Class balance (after undersampling) -----------
    positive_ratio = y_train.mean()
    assert 0.4 < positive_ratio < 0.6, f"Expected balanced classes, got {positive_ratio:.3f}"

    # 2. Median should be centered near 0 after RobustScaler -----------
    time_median = X_train['Time'].median()
    amount_median = X_train['Amount'].median()
    assert np.isclose(time_median, 0, atol=0.1), f"Time median not centered: {time_median}"
    assert np.isclose(amount_median, 0, atol=0.1), f"Amount median not centered: {amount_median}"

    # 3. IQR after RobustScaler can deviate from1 -----------
    amount_iqr = X_train['Amount'].quantile(0.75) - X_train['Amount'].quantile(0.25)
    assert 0.8 < amount_iqr < 1.5, f"Amount IQR not in expected range: {amount_iqr}"

# ------- Test Data Partitioning for FL -------

def test_partition_data():

    X = pd.DataFrame({'feature': range(100)})
    y = pd.Series([0, 1] * 50)
    
    partitions = partition_data(X, y, num_clients=5)
    
    # Check number of partitions
    assert len(partitions) == 5
    
    # Check no data leakage
    all_indices = []
    for X_part, _ in partitions:
        all_indices.extend(X_part.index.tolist())
    assert len(set(all_indices)) == len(all_indices)  # No duplicates


def test_preprocess_reproducibility(raw_data_path):
    """Test that preprocessing is deterministic"""
    X1, _, _, _ = preprocess_data(data_path=raw_data_path, random_state=42)
    X2, _, _, _ = preprocess_data(data_path=raw_data_path, random_state=42)
    pd.testing.assert_frame_equal(X1, X2)