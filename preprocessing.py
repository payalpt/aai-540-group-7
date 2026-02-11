
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    args, _ = parser.parse_known_args()
    
    # Load data
    input_path = "/opt/ml/processing/input/baseline_normalized.csv"
    df = pd.read_csv(input_path)
    
    print(f"Loaded dataset: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    # Combine for XGBoost format (target first)
    train_data = pd.concat([y_train.reset_index(drop=True), 
                            X_train.reset_index(drop=True)], axis=1)
    test_data = pd.concat([y_test.reset_index(drop=True), 
                           X_test.reset_index(drop=True)], axis=1)
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Save
    os.makedirs("/opt/ml/processing/train", exist_ok=True)
    os.makedirs("/opt/ml/processing/test", exist_ok=True)
    
    train_data.to_csv("/opt/ml/processing/train/train.csv", index=False, header=False)
    test_data.to_csv("/opt/ml/processing/test/test.csv", index=False, header=False)
    
    print("✅ Preprocessing complete")
