
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split    
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path


def preprocess_data(
    data_path: str = "data/raw/creditcard.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.2,             # Proportion of data to be used for testing
    random_state: int = 42              # Random seed for reproducibility
    ) -> tuple:

    df = pd.read_csv(data_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---------- Preprocessing Steps ----------

    # 1. Scale Time and Amount
    time_scaler = RobustScaler()
    amount_scaler = RobustScaler()
    
    df['Time'] = time_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df['Amount'] = amount_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # 2. Handle class imbalance
    X = df.drop('Class', axis=1)
    y = df['Class']
    sampler = RandomUnderSampler(random_state=random_state)
    X_res, y_res = sampler.fit_resample(X, y)
    
    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_res
    )

    # ---------- Save Preprocessed Data ----------

    # Save scaler parameters
    scaler_params = {
        'time_mean': float(time_scaler.center_[0]),
        'time_scale': float(time_scaler.scale_[0]),
        'amount_mean': float(amount_scaler.center_[0]),
        'amount_scale': float(amount_scaler.scale_[0])
    }
    with open(f"{output_dir}/scaler_params.json", "w") as f:
        json.dump(scaler_params, f)
    
    # Save processed data
    X_train.to_parquet(f"{output_dir}/X_train.parquet")
    X_test.to_parquet(f"{output_dir}/X_test.parquet")
    y_train.to_frame().to_parquet(f"{output_dir}/y_train.parquet") 
    y_test.to_frame().to_parquet(f"{output_dir}/y_test.parquet")
    
    # Save sample of processed data for testing
    sample = X_train.head(100).copy()
    sample['Class'] = y_train.head(100)
    sample.to_csv(f"{output_dir}/sample_preprocessed.csv", index=False)
    
    return X_train, X_test, y_train, y_test


""" Partition data for federated learning

    Args:
        X: Features DataFrame
        y: Labels Series
        num_clients: Number of partitions to create
        output_dir: Directory to save partitions

    Returns:
        List of (X_partition, y_partition) tuples
"""

def partition_data(
    X: pd.DataFrame,
    y: pd.Series,
    num_clients: int = 5,
    output_dir: str = "data/partitions"
) -> list:
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create partitions
    partitions = []

    for i in range(num_clients):
        # Strided sampling for heterogeneity
        idx = np.arange(i, len(X), num_clients)
        X_part = X.iloc[idx]
        y_part = y.iloc[idx]
        
        # Save partition
        X_part.to_parquet(f"{output_dir}/client_{i}_X.parquet")
        y_part.to_frame().to_parquet(f"{output_dir}/client_{i}_y.parquet")
        partitions.append((X_part, y_part))
    
    return partitions