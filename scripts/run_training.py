
import os
import logging
import time
import numpy as np
from sympy import im
import wandb
import random
import torch    
from src.config import DEVICE, CONFIG
from src.data_getloaders import get_datasetloaders
from src.model_base import FraudDetectionModel, FocalLoss
from src.model_training import model_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()] #Terminal Output
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42

def set_global_seed(seed: int):
    """Set all relevant seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # for CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for deterministic convs (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    wandb.init(project="fed-fraud-detection", config=CONFIG, name=f"run_{int(time.time())}", save_code=True)
    set_global_seed(RANDOM_SEED)
    model = FraudDetectionModel().to(DEVICE)
    criterion = FocalLoss()
    train_loader, val_loader = get_datasetloaders(batch_size=CONFIG["batch_size"], seed=RANDOM_SEED)
    model_training(model, train_loader, val_loader, criterion, DEVICE, CONFIG, logger)
    wandb.finish()

if __name__ == "__main__":
    main()