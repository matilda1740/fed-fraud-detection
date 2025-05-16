# Federated Fraud Detection with Differential Privacy and Homomorphic Encryption

A secure framework for cross-bank fraud detection using Federated Learning (FL) with Differential Privacy (DP) and Homomorphic Encryption (HE).

## Module 1 - Data Preprocessing & Partitioning 

## 📌 Overview 

This branch contains the secure preprocessing pipeline for federated fraud detection with privacy-preserving data preparation.

## ✨ Features

- ✅ **Data Validation**: Sanity checks for missing values & data types 
- ✅ **Class Balancing**: `RandomUnderSampler` for fraud/legit ratio adjustment  
- ✅ **Robust Scaling**: `Time` and `Amount` feature normalization 
- ✅ **Client Partitioning**: Synthetic bank splits for federated learning  
- ✅ **Data Versioning**: Parquet + JSON metadata storage  
- ✅ **Data Testing**: Comprehensive unit tests

## 📂 Project Structure
<pre>
.
├── data/
│   ├── raw/                   # Original datasets
│   ├── preprocess/            # PreProcessed features
│   └── partitions/            # Client-specific data splits
├── notebooks/
│   ├── 01_exploratory.ipynb
├── src/
│   ├── __init__.py            # Empty File
│   ├── data_preprocess.py     # Module 1 core logic
├── scripts/
│   ├── run_preprocess.py      # Module 1 entry point
├── tests/                     
│   ├── test_preprocess.py     # Module 1 tests
├── requirements.txt
└── setup.py
    
</pre>
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
```

## 🚀 Usage

### Default Preprocessing

```bash
python scripts/run_preprocess.py
```

### Custom Parameters

```bash
python scripts/run_preprocess.py \
    --input data/raw/your_data.csv \
    --output data/custom_processed \
    --clients 8 \
    --test_size 0.15
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Module 1 specific tests
pytest tests/test_preprocess.py -v
```
