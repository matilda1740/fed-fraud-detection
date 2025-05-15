from src.data_preprocess import partition_data, preprocess_data
import argparse
# 
def main():
    parser = argparse.ArgumentParser(description='Preprocess transaction data for federated learning')
    parser.add_argument('--input', default='data/raw/creditcard.csv', help='Input CSV path')
    parser.add_argument('--output', default='data/preprocess', help='Output directory')
    parser.add_argument('--clients', type=int, default=5, help='Number of FL client partitions')
    args = parser.parse_args()

    print(f"Preprocessing data from {args.input}...\n")
    X_train, X_test, y_train, y_test = preprocess_data(
        data_path=args.input,
        output_dir=args.output
    )
    
    print("Creating FL partitions...")
    partition_data(X_train, y_train, num_clients=args.clients)
    
    print(f"✅ Preprocessing complete! Files saved to {args.output}")

if __name__ == "__main__":
    main()