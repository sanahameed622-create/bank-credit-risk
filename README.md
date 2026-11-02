# Bank Credit Risk Scoring & Default Prediction Model

A production-grade credit risk modeling engine and scorecard system designed for retail banking, built from first mathematical principles.

## 📌 Project Overview
When a customer applies for a loan or credit card, commercial banks must assess the **Probability of Default ($PD$)** before approval. This project builds an end-to-end mathematical and statistical pipeline that:
1. Ingests retail banking loan application records.
2. Performs feature screening using **Weight of Evidence (WoE)** and **Information Value (IV)**.
3. Optimizes a **Logistic Regression** model using **Batch Gradient Descent**.
4. Validates model discrimination using **ROC-AUC** and the **Gini Coefficient** ($Gini = 2 \times AUC - 1$).
5. Scales raw probabilities into a standard **300–850 FICO Credit Score** using **Points to Double the Odds (PDO)**.
6. Makes real-time automated loan underwriting decisions (`APPROVED`, `MANUAL REVIEW`, `REJECTED`).

---

## 🧮 Mathematical Foundations

### 1. The Sigmoid (Logistic) Function
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
Where $z = \beta_0 + \sum_{j=1}^n \beta_j X_j$. This maps any linear combination of borrower risk factors into a well-defined probability $p \in (0, 1)$.

### 2. Log-Odds (Logit Transformation)
$$\text{Logit}(p) = \ln\left(\frac{p}{1 - p}\right) = \mathbf{\beta}^T \mathbf{X}$$

### 3. Binary Cross-Entropy Loss (Log-Loss)
$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(p^{(i)}) + (1 - y^{(i)}) \ln(1 - p^{(i)}) \right]$$

### 4. Gradient Descent Optimization
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{1}{m} \mathbf{X}^T (\mathbf{p} - \mathbf{y})$$
$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial \mathcal{L}}{\partial \mathbf{w}}$$

### 5. Weight of Evidence (WoE) & Information Value (IV)
For each feature bin $k$:
$$\text{WoE}_k = \ln\left( \frac{\% \text{ Non-Defaulters}_k}{\% \text{ Defaulters}_k} \right)$$
$$\text{IV} = \sum_{k=1}^K (\% \text{ Non-Defaulters}_k - \% \text{ Defaulters}_k) \times \text{WoE}_k$$

### 6. Credit Scorecard Scaling (FICO Formula)
$$\text{Score} = \text{Offset} + \text{Factor} \times \ln(\text{Odds})$$
Where:
- $\text{Odds} = \frac{1 - p}{p}$ (Good-to-Bad ratio)
- $\text{Factor} = \frac{\text{PDO}}{\ln(2)}$
- $\text{Offset} = \text{Target Score} - \text{Factor} \times \ln(\text{Target Odds})$

---

## 🚀 How to Run

### Step 1: Generate Data & Train Model
```bash
python train_model.py
```

### Step 2: Test Standard Banking Benchmarks
```bash
python credit_scorecard.py
```

### Step 3: Run Interactive Loan Underwriter
```bash
python interactive_evaluator.py
```

---

## 📊 Model Evaluation Results
- **ROC-AUC Score**: `0.8017` (High Discriminatory Power)
- **Gini Coefficient**: `0.6034` (Exceeds standard 0.40 Basel banking threshold)
- **Classification Accuracy**: `81.90%`
