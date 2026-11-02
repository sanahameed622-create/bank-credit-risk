"""
Realistic Banking Loan Applicant Dataset Generator
Author: Mathematics & Credit Risk Project
"""
import csv
import random
import os

def generate_banking_dataset(num_records=5000, output_path="loan_data.csv"):
    random.seed(42)
    
    headers = [
        "applicant_id",
        "person_age",
        "person_income",
        "person_home_ownership",
        "person_emp_length",
        "loan_intent",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_default_on_file",
        "cb_person_cred_hist_length",
        "loan_status"  # 1 = Default, 0 = Non-Default
    ]
    
    home_ownerships = ["RENT", "OWN", "MORTGAGE"]
    intents = ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
    
    rows = []
    
    for i in range(1, num_records + 1):
        app_id = f"AP-{10000 + i}"
        age = int(random.triangular(20, 65, 29))
        income = int(random.expovariate(1 / 55000) + 18000)
        home = random.choices(home_ownerships, weights=[0.50, 0.15, 0.35])[0]
        emp_length = min(max(0, int(random.gauss(5, 3))), age - 18)
        intent = random.choice(intents)
        
        # Loan amount typically proportional to income
        max_loan_ratio = 0.55 if home == "OWN" else 0.40
        loan_amnt = int(random.uniform(2000, min(50000, income * max_loan_ratio + 5000)))
        loan_percent_income = round(loan_amnt / income, 2)
        
        # Interest rate correlates with risk
        base_rate = random.uniform(6.5, 18.5)
        if loan_percent_income > 0.35:
            base_rate += random.uniform(2.0, 4.5)
        loan_int_rate = round(min(24.0, max(5.0, base_rate)), 2)
        
        # Credit history length
        cred_hist_length = max(1, min(age - 18, int(random.gauss(6, 3))))
        
        # Past default on file
        past_default_prob = 0.12 if income > 60000 else 0.28
        past_default = "Y" if random.random() < past_default_prob else "N"
        
        # --- Mathematical Risk Modeling (True Ground Truth Probability of Default) ---
        # Logistic formula for realistic default risk:
        # z = beta0 + beta1*loan_percent_income + beta2*int_rate + beta3*past_default - beta4*income - beta5*cred_hist
        z = -2.6 \
            + (4.2 * loan_percent_income) \
            + (0.12 * (loan_int_rate - 10.0)) \
            + (1.5 if past_default == "Y" else 0.0) \
            + (0.6 if home == "RENT" else -0.4 if home == "OWN" else 0.0) \
            - (0.000015 * income) \
            - (0.06 * cred_hist_length)
        
        # Sigmoid function: p = 1 / (1 + e^-z)
        import math
        prob_default = 1.0 / (1.0 + math.exp(-z))
        
        # Actual outcome (Bernoulli trial)
        loan_status = 1 if random.random() < prob_default else 0
        
        rows.append([
            app_id,
            age,
            income,
            home,
            emp_length,
            intent,
            loan_amnt,
            loan_int_rate,
            loan_percent_income,
            past_default,
            cred_hist_length,
            loan_status
        ])
    
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"Dataset successfully created at '{output_path}' with {num_records} applicants.")
    defaults = sum(1 for r in rows if r[-1] == 1)
    print(f"Summary: Non-Defaults = {num_records - defaults} ({((num_records - defaults) / num_records)*100:.1f}%), Defaults = {defaults} ({(defaults / num_records)*100:.1f}%)")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "loan_data.csv")
    generate_banking_dataset(5000, output_file)
