"""
Credit Risk Model Training & Evaluation Engine
Trains a production-grade Credit Risk Scoring model and generates credit scorecards.
"""
import csv
import json
import os
import math
import random
from math_core import (
    train_logistic_regression,
    sigmoid,
    dot_product,
    calculate_woe_iv,
    calculate_credit_score,
    calculate_auc_roc
)

def load_data(filepath):
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def calculate_mean_std(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 1.0
    return mean, std

def run_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "loan_data.csv")
    
    if not os.path.exists(data_path):
        print("Data not found, generating dataset first...")
        from generate_data import generate_banking_dataset
        generate_banking_dataset(5000, data_path)
        
    records = load_data(data_path)
    print(f" Loaded {len(records)} loan applications from database.")
    
    # 1. Feature Analysis: Weight of Evidence (WoE) & Information Value (IV)
    print("\n" + "=" * 60)
    print(" 1. BANKING FEATURE IMPORTANCE ANALYSIS (WOE & IV)")
    print("=" * 60)
    
    y_all = [int(r["loan_status"]) for r in records]
    
    features_to_check = [
        ("Loan % of Income", [float(r["loan_percent_income"]) for r in records]),
        ("Loan Interest Rate", [float(r["loan_int_rate"]) for r in records]),
        ("Annual Income", [float(r["person_income"]) for r in records]),
        ("Credit History Length", [float(r["cb_person_cred_hist_length"]) for r in records]),
        ("Applicant Age", [float(r["person_age"]) for r in records])
    ]
    
    for feat_name, vals in features_to_check:
        _, iv = calculate_woe_iv(vals, y_all)
        power = "STRONG" if iv >= 0.3 else "MEDIUM" if iv >= 0.1 else "WEAK"
        print(f"  • {feat_name:24}: IV = {iv:.4f}  [{power} PREDICTOR]")
        
    # 2. Preprocessing & Feature Engineering
    # Features:
    # 1. loan_percent_income (Standardized)
    # 2. loan_int_rate (Standardized)
    # 3. person_income (Log transformed + Standardized)
    # 4. cb_person_cred_hist_length (Standardized)
    # 5. person_age (Standardized)
    # 6. cb_person_default_on_file == 'Y' (1 or 0)
    # 7. home_RENT (1 or 0)
    # 8. home_OWN (1 or 0)
    
    feature_names = [
        "loan_percent_income",
        "loan_int_rate",
        "log_income",
        "cred_hist_length",
        "person_age",
        "past_default_Y",
        "home_RENT",
        "home_OWN"
    ]
    
    raw_loan_pct = [float(r["loan_percent_income"]) for r in records]
    raw_int_rate = [float(r["loan_int_rate"]) for r in records]
    raw_log_inc  = [math.log(float(r["person_income"])) for r in records]
    raw_cred_hist = [float(r["cb_person_cred_hist_length"]) for r in records]
    raw_age      = [float(r["person_age"]) for r in records]
    
    scalers = {
        "loan_percent_income": calculate_mean_std(raw_loan_pct),
        "loan_int_rate": calculate_mean_std(raw_int_rate),
        "log_income": calculate_mean_std(raw_log_inc),
        "cred_hist_length": calculate_mean_std(raw_cred_hist),
        "person_age": calculate_mean_std(raw_age)
    }
    
    X = []
    y = []
    
    for i, r in enumerate(records):
        row = [
            (raw_loan_pct[i] - scalers["loan_percent_income"][0]) / scalers["loan_percent_income"][1],
            (raw_int_rate[i] - scalers["loan_int_rate"][0]) / scalers["loan_int_rate"][1],
            (raw_log_inc[i] - scalers["log_income"][0]) / scalers["log_income"][1],
            (raw_cred_hist[i] - scalers["cred_hist_length"][0]) / scalers["cred_hist_length"][1],
            (raw_age[i] - scalers["person_age"][0]) / scalers["person_age"][1],
            1.0 if r["cb_person_default_on_file"] == "Y" else 0.0,
            1.0 if r["person_home_ownership"] == "RENT" else 0.0,
            1.0 if r["person_home_ownership"] == "OWN" else 0.0
        ]
        X.append(row)
        y.append(int(r["loan_status"]))
        
    # 3. Train/Test Split (80% Train, 20% Test)
    random.seed(101)
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    split_idx = int(0.80 * len(X))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    
    print(f"\n Dataset split: {len(X_train)} Train samples, {len(X_test)} Test samples.")
    
    # 4. Model Training via Gradient Descent
    print("\n" + "=" * 60)
    print(" 2. MODEL TRAINING (GRADIENT DESCENT OPTIMIZATION)")
    print("=" * 60)
    
    weights, bias = train_logistic_regression(X_train, y_train, learning_rate=0.08, epochs=600)
    
    print("\n Learned Model Coefficients (Feature Weights):")
    for fname, w in zip(feature_names, weights):
        direction = "Increases Risk (+)" if w > 0 else "Decreases Risk (-)"
        print(f"  • {fname:22}: beta = {w:+.4f}  ({direction})")
    print(f"  • Intercept (bias)      : beta_0 = {bias:+.4f}")
    
    # 5. Model Evaluation on Test Set
    print("\n" + "=" * 60)
    print(" 3. MODEL PERFORMANCE & BANKING METRICS")
    print("=" * 60)
    
    test_probs = [sigmoid(dot_product(row, weights) + bias) for row in X_test]
    threshold = 0.50
    test_preds = [1 if p >= threshold else 0 for p in test_probs]
    
    tp = sum(1 for yt, yp in zip(y_test, test_preds) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_test, test_preds) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_test, test_preds) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_test, test_preds) if yt == 1 and yp == 0)
    
    accuracy = (tp + tn) / len(y_test)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    auc, gini = calculate_auc_roc(y_test, test_probs)
    
    print(f"  • Accuracy        : {accuracy * 100:.2f}%")
    print(f"  • Precision       : {precision * 100:.2f}% (Quality of default alerts)")
    print(f"  • Recall (Sensitivity): {recall * 100:.2f}% (Defaults successfully caught)")
    print(f"  • F1-Score        : {f1:.4f}")
    print(f"  • ROC-AUC Score   : {auc:.4f}  [High Discriminatory Power]")
    print(f"  • Gini Coefficient: {gini:.4f}  [Banking Standard: Gini = 2*AUC - 1]")
    
    print(f"\n Confusion Matrix:")
    print(f"  ------------------------------------------")
    print(f"  | Actual Non-Default (0): TN={tn:4d} | FP={fp:4d} |")
    print(f"  | Actual Default (1)    : FN={fn:4d} | TP={tp:4d} |")
    print(f"  ------------------------------------------")
    
    # 6. Save Model Artifacts
    artifacts = {
        "feature_names": feature_names,
        "weights": weights,
        "bias": bias,
        "scalers": {k: {"mean": v[0], "std": v[1]} for k, v in scalers.items()},
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "roc_auc": auc,
            "gini": gini
        }
    }
    
    artifacts_path = os.path.join(current_dir, "model_artifacts.json")
    with open(artifacts_path, mode="w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=4)
        
    print(f"\n Model successfully saved to '{artifacts_path}'.")

if __name__ == "__main__":
    run_pipeline()
