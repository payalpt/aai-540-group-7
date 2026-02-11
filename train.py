
import argparse
import os
import pandas as pd
import xgboost as xgb
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument("--objective", type=str, default="multi:softmax")
    parser.add_argument("--num_class", type=int, default=7)
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=4)
    parser.add_argument("--min_child_weight", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--verbosity", type=int, default=1)
    
    args, _ = parser.parse_known_args()
    
    # Load training data
    train_path = os.path.join("/opt/ml/input/data/train", "train.csv")
    train_data = pd.read_csv(train_path, header=None)
    
    print(f"Training data shape: {train_data.shape}")
    
    # Prepare data for XGBoost
    y_train = train_data.iloc[:, 0]
    X_train = train_data.iloc[:, 1:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Set parameters
    params = {
        'objective': args.objective,
        'num_class': args.num_class,
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'verbosity': args.verbosity
    }
    
    # Train model
    print(f"Training with params: {params}")
    model = xgb.train(params, dtrain, num_boost_round=args.num_round)
    
    # Save model
    model_dir = "/opt/ml/model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgboost-model")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved to {model_path}")
