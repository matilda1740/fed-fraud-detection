import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

def get_dataloaders(batch_size=256, val_size=0.2):
    # Load preprocessed data
    X_train = pd.read_parquet("data/preprocess/X_train.parquet")
    y_train = pd.read_parquet("data/preprocess/y_train.parquet").squeeze()
    
    # Create Tensor datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train.values),
        torch.LongTensor(y_train.values)
    )
    
    # Split validation
    val_size = int(len(train_dataset) * val_size)
    train_size = len(train_dataset) - val_size
    train_data, val_data = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size*2, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader
