import pytest
import pandas as pd
import numpy as np
from src.data_preprocess import partition_data, preprocess_data
from pathlib import Path

@pytest.fixture
def raw_data_path():
    return "data/raw/creditcard.csv"

# ------- Test File Saves -------

def test_preprocess_data_saves_files(raw_data_path):

    # Run preprocessing
    _ = preprocess_data(input_path=raw_data_path)
    
    # Check files exist
    assert Path("data/preprocess/X_train.parquet").exists()
    assert Path("data/preprocess/y_test.parquet").exists()
    assert Path("data/preprocess/scaler_params.json").exists()
    assert Path("data/preprocess/sample_preprocessed.csv").exists()

# ------- Test Data Transforms -------

def test_preprocessing_transforms(raw_data_path):

    X_train, _, y_train, _ = preprocess_data(input_path=raw_data_path)
    
    # Check class balance
    assert 0.4 < y_train.mean() < 0.6  # Should be roughly balanced
    
    # Check scaling
    assert np.isclose(X_train['Time'].mean(), 0, atol=0.1)
    assert np.isclose(X_train['Amount'].std(), 1, atol=0.1)

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


def test_preprocessing_reproducibility(raw_data_path):
    """Test that preprocessing is deterministic"""
    X1, _, _, _ = preprocess_data(input_path=raw_data_path, random_state=42)
    X2, _, _, _ = preprocess_data(input_path=raw_data_path, random_state=42)
    pd.testing.assert_frame_equal(X1, X2)