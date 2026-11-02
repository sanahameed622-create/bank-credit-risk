"""
Interactive Banking Credit Scorecard & Loan Underwriter
Converts statistical default probabilities into 300-850 FICO Credit Scores
and generates automatic underwriting decisions.
"""
import json
import os
import math
from math_core import sigmoid, dot_product, calculate_credit_score

def load_artifacts():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_path = os.path.join(current_dir, "model_artifacts.json")
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError("Model artifacts not found. Please run train_model.py first.")
    with open(artifacts_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_applicant(
    person_age,
    person_income,
    loan_amnt,
    loan_int_rate,
    person_home_ownership,  # 'RENT', 'OWN', 'MORTGAGE'
    past_default_on_file,    # 'Y' or 'N'
    cred_hist_length
):
    artifacts = load_artifacts()
    scalers = artifacts["scalers"]
    weights = artifacts["weights"]
    bias = artifacts["bias"]
    
    # 1. Feature transformations
    loan_percent_income = loan_amnt / max(1.0, person_income)
    log_income = math.log(max(100.0, person_income))
    
    std_loan_pct  = (loan_percent_income - scalers["loan_percent_income"]["mean"]) / scalers["loan_percent_income"]["std"]
    std_int_rate  = (loan_int_rate - scalers["loan_int_rate"]["mean"]) / scalers["loan_int_rate"]["std"]
    std_log_inc   = (log_income - scalers["log_income"]["mean"]) / scalers["log_income"]["std"]
    std_cred_hist = (cred_hist_length - scalers["cred_hist_length"]["mean"]) / scalers["cred_hist_length"]["std"]
    std_age       = (person_age - scalers["person_age"]["mean"]) / scalers["person_age"]["std"]
    
    past_default_val = 1.0 if past_default_on_file.upper() == "Y" else 0.0
    home_rent_val    = 1.0 if person_home_ownership.upper() == "RENT" else 0.0
    home_own_val     = 1.0 if person_home_ownership.upper() == "OWN" else 0.0
    
    feature_vector = [
        std_loan_pct,
        std_int_rate,
        std_log_inc,
        std_cred_hist,
        std_age,
        past_default_val,
        home_rent_val,
        home_own_val
    ]
    
    # 2. Compute Probability of Default (PD)
    z = dot_product(feature_vector, weights) + bias
    prob_default = sigmoid(z)
    
    # 3. Compute Credit Score (300 to 850)
    score = calculate_credit_score(prob_default)
    
    # 4. Bank Decision & Rating
    if score >= 780:
        tier = "AAA (Prime+)"
        decision = "APPROVED (Instant / Best Interest Rate)"
        color = "GREEN"
    elif score >= 720:
        tier = "AA (Prime)"
        decision = "APPROVED (Standard Rate)"
        color = "GREEN"
    elif score >= 660:
        tier = "A (Near Prime)"
        decision = "APPROVED (Standard Rate + Monitoring)"
        color = "CYAN"
    elif score >= 600:
        tier = "BBB (Subprime / Borderline)"
        decision = "MANUAL REVIEW REQUIRED (Requires Guarantor or Collateral)"
        color = "YELLOW"
    elif score >= 520:
        tier = "BB (High Risk)"
        decision = "REJECTED (High Default Probability)"
        color = "RED"
    else:
        tier = "C (Extremely High Risk)"
        decision = "REJECTED (Severe Credit Impairment)"
        color = "RED"
        
    return {
        "score": score,
        "prob_default": round(prob_default * 100, 2),
        "tier": tier,
        "decision": decision,
        "loan_percent_income": round(loan_percent_income * 100, 1),
        "color": color
    }

def print_applicant_report(name, details, result):
    print("\n" + "=" * 65)
    print(f"       BANK LOAN ASSESSMENT REPORT: {name}")
    print("=" * 65)
    print(f"  • Annual Income        : ${details['income']:,}")
    print(f"  • Requested Loan       : ${details['loan']:,} ({result['loan_percent_income']}% of annual income)")
    print(f"  • Interest Rate        : {details['int_rate']}%")
    print(f"  • Home Ownership       : {details['home']}")
    print(f"  • Past Default Record  : {details['past_default']}")
    print(f"  • Credit History Length: {details['cred_hist']} years (Age: {details['age']})")
    print("-" * 65)
    print(f"  >>> FICO CREDIT SCORE  :  {result['score']} / 850")
    print(f"  >>> DEFAULT PROBABILITY:  {result['prob_default']}%")
    print(f"  >>> RISK RATING TIER   :  {result['tier']}")
    print(f"  >>> BANK DECISION      :  {result['decision']}")
    print("=" * 65 + "\n")

def run_sample_cases():
    test_cases = [
        {
            "name": "Applicant 1: Dr. Ali Khan (High Earner, Low Debt)",
            "details": {
                "age": 42,
                "income": 120000,
                "loan": 15000,
                "int_rate": 7.5,
                "home": "OWN",
                "past_default": "N",
                "cred_hist": 14
            }
        },
        {
            "name": "Applicant 2: Bilal Ahmed (Moderate Income, Modest Car Loan)",
            "details": {
                "age": 28,
                "income": 45000,
                "loan": 10000,
                "int_rate": 11.0,
                "home": "MORTGAGE",
                "past_default": "N",
                "cred_hist": 5
            }
        },
        {
            "name": "Applicant 3: Usman Tariq (High Debt, Past Default History)",
            "details": {
                "age": 24,
                "income": 22000,
                "loan": 14000,
                "int_rate": 19.5,
                "home": "RENT",
                "past_default": "Y",
                "cred_hist": 2
            }
        }
    ]
    
    print("\n RUNNING AUTOMATED BENCHMARK LOAN UNDERWRITING TESTS:")
    for case in test_cases:
        d = case["details"]
        res = evaluate_applicant(
            person_age=d["age"],
            person_income=d["income"],
            loan_amnt=d["loan"],
            loan_int_rate=d["int_rate"],
            person_home_ownership=d["home"],
            past_default_on_file=d["past_default"],
            cred_hist_length=d["cred_hist"]
        )
        print_applicant_report(case["name"], d, res)

if __name__ == "__main__":
    run_sample_cases()
