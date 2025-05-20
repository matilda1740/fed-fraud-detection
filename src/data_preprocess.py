
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler


def preprocess_data(
    data_path: str = "data/raw/creditcard.csv",
    output_dir: str = "data/preprocess",
    test_size: float = 0.2,             # Proportion of data to be used for testing
    random_state: int = 42              # Random seed for reproducibility
    ) -> tuple:

    df = pd.read_csv(data_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---------- Preprocessing Steps ----------

    # ----------- 1. Split BEFORE scaling -----------
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

     # ----------- 2. Fit scalers ONLY on training data -----------
    time_scaler = RobustScaler()
    amount_scaler = RobustScaler()

    X_train['Time'] = time_scaler.fit_transform(X_train[['Time']])
    X_train['Amount'] = amount_scaler.fit_transform(X_train[['Amount']])

    # Apply the same scaling to test set
    X_test['Time'] = time_scaler.transform(X_test[['Time']])
    X_test['Amount'] = amount_scaler.transform(X_test[['Amount']])

     # ----------- 3. Undersample ONLY the training set -----------
    sampler = RandomUnderSampler(random_state=random_state)
    X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)

    # ----------- 4. Save scaler parameters -----------
    scaler_params = {
        'time_center': float(time_scaler.center_[0]),
        'time_scale': float(time_scaler.scale_[0]),
        'amount_center': float(amount_scaler.center_[0]),
        'amount_scale': float(amount_scaler.scale_[0])
    }
    with open(f"{output_dir}/scaler_params.json", "w") as f:
        json.dump(scaler_params, f)
    # ----------- 5. Save preprocessed data -----------
    X_train_res.to_parquet(f"{output_dir}/X_train.parquet")
    X_test.to_parquet(f"{output_dir}/X_test.parquet")
    pd.DataFrame(y_train_res).to_parquet(f"{output_dir}/y_train.parquet")
    y_test.to_frame().to_parquet(f"{output_dir}/y_test.parquet")

    return X_train_res, X_test, y_train_res, y_test          


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