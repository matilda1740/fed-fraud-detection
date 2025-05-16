# Federated Fraud Detection with Differential Privacy and Homomorphic Encryption

A secure framework for cross-bank fraud detection using Federated Learning (FL) with Differential Privacy (DP) and Homomorphic Encryption (HE).

## 📌 Overview

This repository contains two core modules:
- **Module 1**: Data Preprocessing & Partitioning  
- **Module 2**: Model Training & Evaluation

## ✨ Features

### Module 1: Data Preprocessing
- ✅ Robust feature scaling (`Time`/`Amount` columns)
- ✅ Class imbalance handling via undersampling
- ✅ Secure client data partitioning for FL
- ✅ Parquet format storage with scaler metadata
- ✅ Comprehensive unit tests

### Module 2: Model Training
- 🧠 Optimized neural architecture with dropout/BN
- ⚖️ Focal Loss for class imbalance mitigation
- 📉 Precision-Recall AUC & F1-score metrics
- 🚀 GPU acceleration support
- 🔄 Dynamic learning rate scheduling


## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/matilda1740/fed-fraud-detection.git
cd fed-fraud-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install project as editable package
pip install -e .

## 📂 Project Structure
.
├── data/
│   ├── raw/                   # Original datasets
│   ├── preprocess/            # PreProcessed features
│   └── partitions/            # Client-specific data splits
├── notebooks/
│   ├── 01_data_preprocess.ipynb
│   └── 02_model_train_eval.ipynb
├── src/
│   ├── __init__.py            # Empty File
│   ├── data_preprocess.py     # Module 1 core logic
│   ├── base_model.py          # Module 2 model architecture
│   └── training.py            # Training utilities
├── scripts/
│   ├── run_preprocess.py      # Module 1 entry point
│   └── model_training.py      # Module 2 entry point
├── tests/                     
│   ├── test_preprocess.py     # Module 1 tests
│   ├── test_model.py          # Module 2 tests
├── requirements.txt
└── setup.py


