
import json
import os
import pickle
import tarfile
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

if __name__ == "__main__":
    # Extract model
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="/opt/ml/processing/model")
    
    # Load model
    with open("/opt/ml/processing/model/xgboost-model", "rb") as f:
        model = pickle.load(f)
    
    # Load test data
    test_path = "/opt/ml/processing/test/test.csv"
    test_data = pd.read_csv(test_path, header=None)
    
    y_test = test_data.iloc[:, 0]
    X_test = test_data.iloc[:, 1:]
    
    print(f"Test data shape: {X_test.shape}")
    
    # Predict
    dtest = xgb.DMatrix(X_test)
    predictions = model.predict(dtest)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average='weighted', zero_division=0
    )
    
    print(f"\nModel Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Create evaluation report
    report = {
        "multiclass_classification_metrics": {
            "accuracy": {"value": accuracy},
            "precision": {"value": precision},
            "recall": {"value": recall},
            "f1": {"value": f1}
        }
    }
    
    # Save report
    output_dir = "/opt/ml/processing/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/evaluation.json", "w") as f:
        json.dump(report, f)
    
    print("✅ Evaluation complete")
