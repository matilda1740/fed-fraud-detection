import torch
import pytest
from src.model_base import FraudDetectionModel, FocalLoss
from src.model_dataloaders import get_dataloaders
@pytest.fixture
def sample_batch():
        return (
        torch.randn(32, 30),  # Features
        torch.randint(0, 2, (32,))  # Labels
    )  
# ------ Verify model processes inputs without errors 

def test_forward_pass(sample_batch):
    model = FraudDetectionModel()
    X, _ = sample_batch
    outputs = model(X)
    assert outputs.shape == (32, 2)        # Essential dimension check
    assert not torch.isnan(outputs).any()  # Critical numerical stability

# ----- Verify single training step completes successfully

def test_training_step(sample_batch):
    model = FraudDetectionModel()
    X, y = sample_batch
    loss = FocalLoss()(model(X), y)
    loss.backward()
    assert loss.item() > 0  

# ------ Verify data format and dimensions
def test_data_loading():
    train_loader, _ = get_dataloaders(batch_size=32)
    X, y = next(iter(train_loader))
    assert X.shape == (32, 30) and y.shape == (32,)  # Critical shape validation

# ----- Verify Custom focal loss handles class imbalance by computing non-zero positive values

def test_focal_loss_computation():
    criterion = FocalLoss()
    
    # Create proper integer labels
    logits = torch.randn(100, 2)
    targets = torch.cat([
        torch.zeros(90, dtype=torch.long),  # Use long type for class indices
        torch.ones(10, dtype=torch.long)
    ])
    
    loss = criterion(logits, targets)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    assert not torch.isnan(loss)

# ----- Verify model saves and loads correctly

