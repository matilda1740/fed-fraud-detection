# Custom Flower client for Fraud Detection

import torch
import numpy as np

from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    confusion_matrix, precision_recall_curve, auc
)
import flwr as fl
from model_base import FraudDetectionModel, FocalLoss
from data_preprocess import create_synthetic_banks

# create_synthetic_banks()
bank_partitions, test_dataset = create_synthetic_banks(5)
print("✔️ Bank partitioning complete!")


class FlowerFraudClient(fl.client.NumPyClient):
    def __init__(self, bank_id, partition):
        self.bank_id = bank_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Convert the numpy arrays to PyTorch tensors
        X_train, y_train = partition['train']
        X_val, y_val = partition['val']

        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            ),
            batch_size=64,
            shuffle=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32)
            ),
            batch_size=64,
            shuffle=False
        )

        self.model = FraudDetectionModel().to(self.device)
        self.criterion = FocalLoss(alpha=0.8, gamma=2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)



    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {key: torch.tensor(val) for key, val in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):  

        print(f"\n🏦 Bank {self.bank_id} starting training...")
        
        self.set_parameters(parameters)

        self.model.train()
        for epoch in range(2):
            epoch_loss = 0.0
            for features, labels in self.train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            # print(f"Bank {self.bank_id} | Epoch {epoch+1} | Loss: {epoch_loss/len(self.train_loader):.4f}")
        
        return self.get_parameters({}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)

                # Calculate Focal Loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Store probabilities and labels
                all_probs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        y_true = np.array(all_labels)
        y_probs = np.array(all_probs) 
        y_pred = (y_probs > 0.5).astype(int) # Convert probabilities to binary predictions

        # Calculate metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    
        # Calculate AUC-PR 
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
        auc_pr = auc(recall_curve, precision_curve)
        avg_precision = average_precision_score(y_true, y_probs)
        
        # Calculate confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
        print(f"\nBank {self.bank_id} Fraud Detection Metrics:")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"AUC-PR: {auc_pr:.4f} | Avg Precision: {avg_precision:.4f}")
        print(f"Confusion Matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
        
        return total_loss/len(self.val_loader), len(self.val_loader.dataset),{
            'focal_loss': total_loss/len(self.val_loader), 
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc_pr': float(auc_pr),
            'avg_precision': float(avg_precision),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }



# FLOWER CLIENT FACTORY
from flwr.common import Context
from flwr.client import ClientApp

def client_fn(context: Context) -> FlowerFraudClient:

    try:
        bank_id = context.node_config['partition-id']
        partition = bank_partitions[bank_id]
        return FlowerFraudClient(bank_id, partition).to_client()
    except Exception as e:
        print(f"Client initialization failed: {e}")
        raise 

client_app = ClientApp(client_fn=client_fn)