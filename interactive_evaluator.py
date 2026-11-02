"""
Interactive Loan Application Evaluator CLI
Allows entering any applicant details to compute live Credit Score & Approval decision.
"""
import sys
from credit_scorecard import evaluate_applicant, print_applicant_report

def prompt_number(prompt_text, default_val, is_float=False):
    try:
        raw = input(f"{prompt_text} [Default: {default_val}]: ").strip()
    except EOFError:
        return default_val
    if not raw:
        return default_val
    try:
        return float(raw) if is_float else int(raw)
    except ValueError:
        return default_val

def prompt_choice(prompt_text, choices, default_val):
    try:
        raw = input(f"{prompt_text} ({'/'.join(choices)}) [Default: {default_val}]: ").strip().upper()
    except EOFError:
        return default_val
    return raw if raw in choices else default_val

def main():
    print("=" * 65)
    print("      BANK CREDIT RISK & LOAN APPROVAL CALCULATOR")
    print("=" * 65)
    
    age = prompt_number("Enter Applicant Age", 30)
    income = prompt_number("Enter Annual Income ($)", 60000)
    loan = prompt_number("Enter Loan Amount ($)", 12000)
    int_rate = prompt_number("Enter Interest Rate (%)", 9.5, is_float=True)
    home = prompt_choice("Home Ownership", ["RENT", "OWN", "MORTGAGE"], "RENT")
    past_default = prompt_choice("Past Default on Credit File", ["Y", "N"], "N")
    cred_hist = prompt_number("Credit History Length (years)", 6)
    
    details = {
        "age": age,
        "income": income,
        "loan": loan,
        "int_rate": int_rate,
        "home": home,
        "past_default": past_default,
        "cred_hist": cred_hist
    }
    
    result = evaluate_applicant(
        person_age=age,
        person_income=income,
        loan_amnt=loan,
        loan_int_rate=int_rate,
        person_home_ownership=home,
        past_default_on_file=past_default,
        cred_hist_length=cred_hist
    )
    
    print_applicant_report("Live Applicant Evaluation", details, result)

if __name__ == "__main__":
    main()
