
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split

def get_datasetloaders(batch_size: int, seed: int):
    X = pd.read_parquet("data/preprocess/X_train.parquet").values
    y = pd.read_parquet("data/preprocess/y_train.parquet").squeeze().values
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
  
    indices = list(range(len(dataset)))

    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=seed
        )
    
     # create a fixed Generator for the train loader
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        Subset(dataset, train_idx), 
        batch_size=batch_size, 
        shuffle=True,
        generator=g,
        drop_last=False,
        )
    
    val_loader = DataLoader(
        Subset(dataset, val_idx), 
        batch_size=batch_size, 
        shuffle=False
        )
    
    return train_loader, val_loader