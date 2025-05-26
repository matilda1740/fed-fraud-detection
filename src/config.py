import torch

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {
    "batch_size": 256,
    "epochs": 50,
    "learning_rate": 0.0005,
    "weight_decay": 1e-5,
    "patience": 3,          # For learning rate scheduling - ReduceLROnPlateau
    "early_stop_k": 2       # extra epochs to wait beyond LR patience
}
