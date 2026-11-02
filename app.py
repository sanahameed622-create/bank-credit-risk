"""
Bank Credit Risk & Loan Underwriting Web Portal
Flask Web Application with Interactive UI and Real-Time Mathematical Engine.
"""
import os
import json
import math
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load Model Artifacts
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_PATH = os.path.join(CURRENT_DIR, "model_artifacts.json")

def get_artifacts():
    if not os.path.exists(ARTIFACTS_PATH):
        from train_model import run_pipeline
        run_pipeline()
    with open(ARTIFACTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def sigmoid(z):
    if z < -700: return 0.0
    if z > 700: return 1.0
    return 1.0 / (1.0 + math.exp(-z))

def calculate_fico(prob_default, target_score=680, target_odds=20, pdo=40):
    factor = pdo / math.log(2)
    offset = target_score - (factor * math.log(target_odds))
    p = max(0.0001, min(0.9999, prob_default))
    good_to_bad_odds = (1.0 - p) / p
    score = offset + (factor * math.log(good_to_bad_odds))
    return int(round(max(300, min(850, score))))

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>National Commercial Bank - Credit Risk Underwriting Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --surface: #111827;
            --surface-card: #1f2937;
            --border: #374151;
            --primary: #3b82f6;
            --primary-glow: rgba(59, 130, 246, 0.25);
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
        body { background: var(--bg); color: var(--text-main); min-height: 100vh; padding: 24px; display: flex; justify-content: center; align-items: center; }
        .wrapper { width: 100%; max-width: 1050px; background: var(--surface); border: 1px solid var(--border); border-radius: 20px; box-shadow: 0 25px 60px -15px rgba(0,0,0,0.7); overflow: hidden; }
        
        /* Top Navigation Header */
        .navbar { background: rgba(17, 24, 39, 0.8); backdrop-filter: blur(12px); border-bottom: 1px solid var(--border); padding: 18px 28px; display: flex; justify-content: space-between; align-items: center; }
        .bank-brand { display: flex; align-items: center; gap: 12px; }
        .bank-icon { width: 38px; height: 38px; background: linear-gradient(135deg, #2563eb, #60a5fa); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 4px 12px var(--primary-glow); }
        .bank-title { font-size: 18px; font-weight: 700; color: #fff; letter-spacing: -0.3px; }
        .bank-sub { font-size: 12px; color: var(--text-muted); }
        .badge { background: #1e3a8a; color: #93c5fd; border: 1px solid #2563eb; padding: 6px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; }
        
        /* Main Layout */
        .content { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 28px; padding: 28px; }
        @media (max-width: 860px) { .content { grid-template-columns: 1fr; } }
        
        /* Form Card */
        .card { background: var(--surface-card); border: 1px solid var(--border); border-radius: 14px; padding: 24px; }
        .card-header { margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 14px; }
        .card-title { font-size: 16px; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
        .card-desc { font-size: 13px; color: var(--text-muted); margin-top: 4px; }
        
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        .form-full { grid-column: span 2; }
        @media (max-width: 600px) { .form-grid { grid-template-columns: 1fr; } .form-full { grid-column: span 1; } }
        
        .form-group { display: flex; flex-direction: column; gap: 6px; }
        label { font-size: 13px; font-weight: 600; color: #d1d5db; }
        input, select { background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #fff; padding: 10px 14px; font-size: 14px; transition: border 0.2s; outline: none; }
        input:focus, select:focus { border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-glow); }
        
        .btn-submit { margin-top: 20px; width: 100%; background: linear-gradient(135deg, #2563eb, #1d4ed8); color: #fff; font-size: 15px; font-weight: 700; border: none; border-radius: 10px; padding: 14px; cursor: pointer; transition: all 0.2s; box-shadow: 0 4px 15px var(--primary-glow); }
        .btn-submit:hover { transform: translateY(-1px); box-shadow: 0 6px 20px var(--primary-glow); background: linear-gradient(135deg, #3b82f6, #2563eb); }
        
        /* Decision Panel */
        .decision-panel { display: flex; flex-direction: column; justify-content: space-between; gap: 18px; }
        .score-box { background: #0b1120; border: 1px solid var(--border); border-radius: 14px; padding: 24px; text-align: center; display: flex; flex-direction: column; align-items: center; }
        
        .gauge { width: 160px; height: 160px; border-radius: 50%; border: 8px solid var(--success); display: flex; flex-direction: column; align-items: center; justify-content: center; margin: 12px 0 16px; transition: border-color 0.4s ease; box-shadow: 0 0 30px rgba(16, 185, 129, 0.15); }
        .score-val { font-size: 46px; font-weight: 800; color: #fff; line-height: 1; }
        .score-lbl { font-size: 12px; font-weight: 600; color: var(--text-muted); margin-top: 4px; }
        
        .decision-tag { font-size: 15px; font-weight: 700; padding: 8px 20px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
        .decision-approved { background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid var(--success); }
        .decision-review   { background: rgba(245, 158, 11, 0.15); color: var(--warning); border: 1px solid var(--warning); }
        .decision-rejected { background: rgba(239, 68, 68, 0.15); color: var(--danger); border: 1px solid var(--danger); }
        
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; width: 100%; margin-top: 14px; }
        .stat-card { background: #1f2937; border: 1px solid #374151; border-radius: 10px; padding: 12px; text-align: left; }
        .stat-label { font-size: 11px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; }
        .stat-value { font-size: 17px; font-weight: 700; color: #fff; margin-top: 3px; }
        
        .footer-note { font-size: 12px; color: var(--text-muted); text-align: center; margin-top: 12px; }
    </style>
</head>
<body>
    <div class="wrapper">
        <!-- Header -->
        <div class="navbar">
            <div class="bank-brand">
                <div class="bank-icon">🏛️</div>
                <div>
                    <div class="bank-title">Automated Credit Underwriting System</div>
                    <div class="bank-sub">National Commercial Bank Risk Management Division</div>
                </div>
            </div>
            <div class="badge">Model: Logistic FICO-PDO 80.2% AUC</div>
        </div>

        <!-- Main Body -->
        <div class="content">
            <!-- Left: Application Form -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">📝 Loan Applicant Assessment Form</div>
                    <div class="card-desc">Enter borrower financials to calculate instant Probability of Default ($PD$).</div>
                </div>

                <form id="loan-form">
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Applicant Age</label>
                            <input type="number" id="age" value="34" min="18" max="75" required>
                        </div>
                        <div class="form-group">
                            <label>Annual Income ($)</label>
                            <input type="number" id="income" value="75000" min="5000" step="1000" required>
                        </div>
                        <div class="form-group">
                            <label>Requested Loan Amount ($)</label>
                            <input type="number" id="loan_amnt" value="15000" min="500" step="500" required>
                        </div>
                        <div class="form-group">
                            <label>Offered Interest Rate (%)</label>
                            <input type="number" id="int_rate" value="8.5" min="4.0" max="25.0" step="0.1" required>
                        </div>
                        <div class="form-group">
                            <label>Credit History Length (Years)</label>
                            <input type="number" id="cred_hist" value="8" min="1" max="40" required>
                        </div>
                        <div class="form-group">
                            <label>Home Ownership</label>
                            <select id="home">
                                <option value="MORTGAGE">Mortgage</option>
                                <option value="OWN">Own (No Debt)</option>
                                <option value="RENT">Rent</option>
                            </select>
                        </div>
                        <div class="form-group form-full">
                            <label>Past Default On Credit Bureau File?</label>
                            <select id="past_default">
                                <option value="N">No - Clean Credit History</option>
                                <option value="Y">Yes - Previous Default / Derogatory Recorded</option>
                            </select>
                        </div>
                    </div>

                    <button type="submit" class="btn-submit">⚡ Evaluate Application & Calculate Score</button>
                </form>
            </div>

            <!-- Right: Real-Time Underwriter Result -->
            <div class="decision-panel card">
                <div class="card-header">
                    <div class="card-title">📊 Underwriting Decision</div>
                    <div class="card-desc">Machine Learning & Mathematical Scorecard output.</div>
                </div>

                <div class="score-box">
                    <div class="gauge" id="gauge">
                        <div class="score-val" id="score-val">742</div>
                        <div class="score-lbl">FICO SCORE / 850</div>
                    </div>

                    <div class="decision-tag decision-approved" id="decision-tag">APPROVED (Prime)</div>

                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-label">Default Risk ($PD$)</div>
                            <div class="stat-value" id="val-pd">1.8%</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Debt-to-Income</div>
                            <div class="stat-value" id="val-dti">20.0%</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Credit Rating</div>
                            <div class="stat-value" id="val-rating">AA (Prime)</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Max Allowed Loan</div>
                            <div class="stat-value" id="val-max-loan">$37,500</div>
                        </div>
                    </div>
                </div>

                <div class="footer-note">
                    * Powered by Basel-II compliant Logistic Scorecard with Information Value screening.
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('loan-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const payload = {
                age: parseFloat(document.getElementById('age').value),
                income: parseFloat(document.getElementById('income').value),
                loan_amnt: parseFloat(document.getElementById('loan_amnt').value),
                int_rate: parseFloat(document.getElementById('int_rate').value),
                cred_hist: parseFloat(document.getElementById('cred_hist').value),
                home: document.getElementById('home').value,
                past_default: document.getElementById('past_default').value
            };

            try {
                const response = await fetch('/api/evaluate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const res = await response.json();
                
                // Update UI elements
                document.getElementById('score-val').innerText = res.score;
                document.getElementById('val-pd').innerText = res.prob_default + '%';
                document.getElementById('val-dti').innerText = res.debt_to_income + '%';
                document.getElementById('val-rating').innerText = res.rating;
                document.getElementById('val-max-loan').innerText = '$' + res.max_allowed_loan.toLocaleString();
                
                const gauge = document.getElementById('gauge');
                const tag = document.getElementById('decision-tag');
                
                tag.className = 'decision-tag';
                if (res.decision_code === 'APPROVED') {
                    tag.classList.add('decision-approved');
                    gauge.style.borderColor = 'var(--success)';
                    gauge.style.boxShadow = '0 0 30px rgba(16, 185, 129, 0.25)';
                } else if (res.decision_code === 'REVIEW') {
                    tag.classList.add('decision-review');
                    gauge.style.borderColor = 'var(--warning)';
                    gauge.style.boxShadow = '0 0 30px rgba(245, 158, 11, 0.25)';
                } else {
                    tag.classList.add('decision-rejected');
                    gauge.style.borderColor = 'var(--danger)';
                    gauge.style.boxShadow = '0 0 30px rgba(239, 68, 68, 0.25)';
                }
                tag.innerText = res.decision;
                
            } catch (err) {
                console.error(err);
                alert('Error evaluating applicant. Check server logs.');
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/evaluate", methods=["POST"])
def evaluate_api():
    data = request.get_json() or {}
    
    age = float(data.get("age", 30))
    income = float(data.get("income", 50000))
    loan_amnt = float(data.get("loan_amnt", 10000))
    int_rate = float(data.get("int_rate", 10.0))
    cred_hist = float(data.get("cred_hist", 5))
    home = str(data.get("home", "RENT")).upper()
    past_default = str(data.get("past_default", "N")).upper()
    
    artifacts = get_artifacts()
    scalers = artifacts["scalers"]
    weights = artifacts["weights"]
    bias = artifacts["bias"]
    
    loan_percent_income = loan_amnt / max(1.0, income)
    log_income = math.log(max(100.0, income))
    
    std_loan_pct  = (loan_percent_income - scalers["loan_percent_income"]["mean"]) / scalers["loan_percent_income"]["std"]
    std_int_rate  = (int_rate - scalers["loan_int_rate"]["mean"]) / scalers["loan_int_rate"]["std"]
    std_log_inc   = (log_income - scalers["log_income"]["mean"]) / scalers["log_income"]["std"]
    std_cred_hist = (cred_hist - scalers["cred_hist_length"]["mean"]) / scalers["cred_hist_length"]["std"]
    std_age       = (age - scalers["person_age"]["mean"]) / scalers["person_age"]["std"]
    
    past_default_val = 1.0 if past_default == "Y" else 0.0
    home_rent_val    = 1.0 if home == "RENT" else 0.0
    home_own_val     = 1.0 if home == "OWN" else 0.0
    
    features = [
        std_loan_pct,
        std_int_rate,
        std_log_inc,
        std_cred_hist,
        std_age,
        past_default_val,
        home_rent_val,
        home_own_val
    ]
    
    z = sum(w * x for w, x in zip(weights, features)) + bias
    prob_default = sigmoid(z)
    score = calculate_fico(prob_default)
    
    max_loan = int(income * (0.50 if home == "OWN" else 0.35))
    
    if score >= 730:
        decision = "APPROVED (Instant Approval)"
        decision_code = "APPROVED"
        rating = "AAA / Prime+"
    elif score >= 660:
        decision = "APPROVED (Standard Rate)"
        decision_code = "APPROVED"
        rating = "A / Near Prime"
    elif score >= 590:
        decision = "MANUAL REVIEW (Guarantor Required)"
        decision_code = "REVIEW"
        rating = "BBB / Borderline"
    else:
        decision = "REJECTED (High Default Risk)"
        decision_code = "REJECTED"
        rating = "C / Subprime"
        
    return jsonify({
        "score": score,
        "prob_default": round(prob_default * 100, 1),
        "debt_to_income": round(loan_percent_income * 100, 1),
        "decision": decision,
        "decision_code": decision_code,
        "rating": rating,
        "max_allowed_loan": max_loan
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
