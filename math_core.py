"""
Mathematical Foundations for Banking Credit Risk Scoring
Implements all core mathematical formulas from first principles:
- Sigmoid Function
- Binary Cross Entropy (Log Loss)
- Gradient Descent Optimization
- Weight of Evidence (WoE) & Information Value (IV)
- Credit Scorecard Scaling (PDO & FICO scaling)
- ROC-AUC & Gini Coefficient
"""
import math

def sigmoid(z):
    """
    Sigmoid activation function:
    maps any real-valued number into a probability between 0 and 1.
    sigma(z) = 1 / (1 + exp(-z))
    """
    # Prevent overflow
    if z < -700:
        return 0.0
    if z > 700:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))

def dot_product(v1, v2):
    """Computes the inner (scalar) product of two vectors."""
    return sum(x * y for x, y in zip(v1, v2))

def compute_log_loss(y_true, y_pred, eps=1e-15):
    """
    Binary Cross-Entropy Loss:
    L = - (1/N) * sum( y*ln(p) + (1-y)*ln(1-p) )
    """
    n = len(y_true)
    loss = 0.0
    for y, p in zip(y_true, y_pred):
        p_clipped = max(eps, min(1.0 - eps, p))
        loss += -(y * math.log(p_clipped) + (1.0 - y) * math.log(1.0 - p_clipped))
    return loss / n

def train_logistic_regression(X, y, learning_rate=0.05, epochs=600):
    """
    Batch Gradient Descent for Logistic Regression.
    Weights update rule:
    w := w - alpha * (1/m) * X^T * (p - y)
    """
    m = len(X)
    n_features = len(X[0])
    
    # Initialize weights to 0
    weights = [0.0] * n_features
    bias = 0.0
    
    for epoch in range(epochs):
        # Forward pass (Linear combination + Sigmoid)
        predictions = []
        for row in X:
            z = dot_product(row, weights) + bias
            p = sigmoid(z)
            predictions.append(p)
            
        # Compute Gradients
        grad_w = [0.0] * n_features
        grad_b = 0.0
        
        for i in range(m):
            error = predictions[i] - y[i]
            grad_b += error
            for j in range(n_features):
                grad_w[j] += error * X[i][j]
                
        grad_b /= m
        grad_w = [g / m for g in grad_w]
        
        # Gradient update
        bias -= learning_rate * grad_b
        weights = [w - learning_rate * gw for w, gw in zip(weights, grad_w)]
        
        if (epoch + 1) % 150 == 0:
            current_loss = compute_log_loss(y, predictions)
            print(f"  [Epoch {epoch+1}/{epochs}] Log-Loss: {current_loss:.4f}")
            
    return weights, bias

def calculate_woe_iv(feature_values, y, num_bins=5):
    """
    Calculates Weight of Evidence (WoE) and Information Value (IV) for a feature.
    WoE_i = ln( (% Goods_i) / (% Bads_i) )
    IV = sum( (% Goods_i - % Bads_i) * WoE_i )
    
    Standard Banking Interpretations of IV:
      < 0.02: Unpredictive
      0.02 to 0.10: Weak predictor
      0.10 to 0.30: Medium predictor
      0.30 to 0.50: Strong predictor
      > 0.50: Suspicious / Overfitting
    """
    total_goods = sum(1 for label in y if label == 0)
    total_bads = sum(1 for label in y if label == 1)
    
    if total_goods == 0 or total_bads == 0:
        return {}, 0.0
        
    # Binning continuous feature into quantiles
    sorted_pairs = sorted(zip(feature_values, y), key=lambda x: x[0])
    bin_size = len(sorted_pairs) // num_bins
    
    iv = 0.0
    woe_dict = {}
    
    for b in range(num_bins):
        start = b * bin_size
        end = (b + 1) * bin_size if b < num_bins - 1 else len(sorted_pairs)
        bin_data = sorted_pairs[start:end]
        
        bin_goods = sum(1 for _, label in bin_data if label == 0)
        bin_bads = sum(1 for _, label in bin_data if label == 1)
        
        # Smoothing to avoid log(0)
        dist_good = max(1e-5, bin_goods / total_goods)
        dist_bad = max(1e-5, bin_bads / total_bads)
        
        woe = math.log(dist_good / dist_bad)
        bin_iv = (dist_good - dist_bad) * woe
        iv += bin_iv
        
        bin_name = f"Bin_{b+1}[{bin_data[0][0]:.2f} - {bin_data[-1][0]:.2f}]"
        woe_dict[bin_name] = {
            "goods": bin_goods,
            "bads": bin_bads,
            "woe": round(woe, 4),
            "iv": round(bin_iv, 4)
        }
        
    return woe_dict, round(iv, 4)

def calculate_credit_score(prob_default, target_score=680, target_odds=20, pdo=40):
    """
    Standard Banking FICO Credit Scorecard Scaling Formula.
    Odds = (1 - p) / p  (Good / Bad ratio)
    Factor = PDO / ln(2)
    Offset = Target_Score - Factor * ln(Target_Odds)
    Score = Offset + Factor * ln(Odds)
    
    Calibrated so:
      ~1.0% default risk  -> ~770+ (Prime/AAA)
      ~5.0% default risk  -> ~680  (Near Prime/A)
      ~12.0% default risk -> ~625  (Borderline/BBB)
      >50.0% default risk -> <500  (Default/Reject/C)
    """
    factor = pdo / math.log(2)
    offset = target_score - (factor * math.log(target_odds))
    
    p = max(0.0001, min(0.9999, prob_default))
    good_to_bad_odds = (1.0 - p) / p
    
    score = offset + (factor * math.log(good_to_bad_odds))
    return int(round(max(300, min(850, score))))

def calculate_auc_roc(y_true, y_pred_probs):
    """
    Calculates Area Under the ROC Curve (ROC-AUC) and Gini Coefficient.
    Gini = 2 * AUC - 1
    In retail banking:
      AUC > 0.70 is acceptable
      AUC > 0.80 is very strong
      AUC > 0.85 is exceptional
    """
    # Sort by predicted probabilities descending
    sorted_pairs = sorted(zip(y_pred_probs, y_true), key=lambda x: x[0], reverse=True)
    
    total_pos = sum(y_true)
    total_neg = len(y_true) - total_pos
    
    if total_pos == 0 or total_neg == 0:
        return 0.5, 0.0
        
    tp = 0
    fp = 0
    auc = 0.0
    prev_fp = 0
    prev_tp = 0
    
    for prob, label in sorted_pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            # Trapezoidal rule for area under curve
            auc += (tp + prev_tp) / 2.0 * (fp - prev_fp)
            prev_fp = fp
            prev_tp = tp
            
    auc /= (total_pos * total_neg)
    gini = 2 * auc - 1
    return round(auc, 4), round(gini, 4)
