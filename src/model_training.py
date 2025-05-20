import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.model_selection import train_test_split

from src.model_base import FraudDetectionModel, FocalLoss


 # Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {
    "batch_size": 256,
    # "epochs": 50,
    "learning_rate": 0.0005,
    "weight_decay": 1e-5,
    "patience": 3  # For learning rate scheduling
}

def get_datasets():
    X = pd.read_parquet("data/preprocess/X_train.parquet").values
    y = pd.read_parquet("data/preprocess/y_train.parquet").squeeze().values
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return dataset, y

def get_datasetloaders():
    dataset, y = get_datasets()
  
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=CONFIG["batch_size"]*2, shuffle=False)
    
    return train_loader, val_loader


def model_evaluate(model, loader, device):
    model.eval()
    y_true, y_scores = [], []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            probabilities = F.softmax(outputs, dim=1)[:, 1]
            
            y_true.extend(y.cpu().numpy())
            y_scores.extend(probabilities.cpu().numpy())
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    f1 = f1_score(y_true, (np.array(y_scores) > 0.5).astype(int))
    return {
        'auc_prc': auc(recall, precision),
        'f1': f1,
        'y_true': y_true,
        'y_scores': y_scores    
    }

def model_training(model, loader, epochs=50, device=DEVICE):

    train_loader, val_loader = get_datasetloaders()

    criterion = FocalLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.0005, 
        weight_decay=1e-5
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=3
        )

    # Training loop
    best_score = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()

        # Validation
        val_metrics = model_evaluate(model, val_loader, device)
        scheduler.step(val_metrics['auc_prc'])
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"Val AUPRC: {val_metrics['auc_prc']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}\n")

        # Save best model
        if val_metrics['auc_prc'] > best_score:
            best_score = val_metrics['auc_prc']
            torch.save(model.state_dict(), "models/trained_model.pt")


