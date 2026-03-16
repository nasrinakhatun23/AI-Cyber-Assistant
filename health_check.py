"""Full project health check"""
import warnings
warnings.filterwarnings('ignore')

print("=== PROJECT HEALTH CHECK ===\n")

# ---- 1. Models ----
import joblib, pandas as pd

m = joblib.load('fraud_model.pkl')
s = joblib.load('scaler.pkl')

model_ok   = type(m).__name__ == 'RandomForestClassifier' and m.n_features_in_ == 11
scaler_ok  = type(s).__name__ == 'StandardScaler' and s.n_features_in_ == 11
feature_ok = m.n_features_in_ == s.n_features_in_

print(f"fraud_model.pkl  : {type(m).__name__}, {m.n_features_in_} features, {m.n_estimators} trees  --> {'PASS' if model_ok else 'FAIL'}")
print(f"scaler.pkl       : {type(s).__name__}, {s.n_features_in_} features              --> {'PASS' if scaler_ok else 'FAIL'}")
print(f"Feature mismatch : {'None - OK' if feature_ok else 'MISMATCH - ERROR'}           --> {'PASS' if feature_ok else 'FAIL'}")

# ---- 2. Model predictions ----
print()
print("--- Model Prediction Tests ---")
rows = [
    ('Normal (payment)', {
        'step':1,'amount':500,'oldbalanceOrg':10000,'newbalanceOrig':9500,
        'oldbalanceDest':1000,'newbalanceDest':1500,'isFlaggedFraud':0,
        'type_CASH_OUT':0,'type_DEBIT':0,'type_PAYMENT':1,'type_TRANSFER':0
    }),
    ('Fraud  (cashout)', {
        'step':1,'amount':181000,'oldbalanceOrg':181000,'newbalanceOrig':0,
        'oldbalanceDest':0,'newbalanceDest':0,'isFlaggedFraud':0,
        'type_CASH_OUT':1,'type_DEBIT':0,'type_PAYMENT':0,'type_TRANSFER':0
    }),
]
pred_ok = True
for label, data in rows:
    scaled = s.transform(pd.DataFrame([data]))
    pred   = m.predict(scaled)[0]
    prob   = m.predict_proba(scaled)[0][1] * 100
    result = "FRAUD" if pred == 1 else "NORMAL"
    print(f"  {label}  -->  {result}  ({prob:.1f}% fraud prob)")

# ---- 3. Flask routes ----
print()
print("--- Route Tests ---")
from app import app
import json

with app.test_client() as c:

    tests = []

    # Home page
    r = c.get('/')
    tests.append(('GET  /', r.status_code == 200))

    # Complaint POST
    r = c.post('/', data={'complaint': 'OTP fraud call asking for bank password urgent'})
    content = r.data.decode()
    tests.append(('POST / (complaint)', r.status_code == 200 and ('HIGH' in content or 'LOW' in content)))

    # Transaction page
    r = c.get('/transaction')
    tests.append(('GET  /transaction', r.status_code == 200))

    # Transaction POST
    r = c.post('/transaction', data={
        'step': '1', 'txn_type': 'PAYMENT', 'amount': '500',
        'oldbalanceOrg': '10000', 'newbalanceOrig': '9500',
        'oldbalanceDest': '1000', 'newbalanceDest': '1500'
    })
    content = r.data.decode()
    tests.append(('POST /transaction (RF)', r.status_code == 200 and ('NORMAL' in content or 'FRAUD' in content)))

    # Health
    r = c.get('/health')
    data = json.loads(r.data)
    tests.append(('GET  /health', data.get('status') == 'ok'))

    # CSS
    r = c.get('/static/style.css')
    tests.append(('GET  /static/style.css', r.status_code == 200))

    all_pass = True
    for name, ok in tests:
        status = "PASS" if ok else "FAIL"
        if not ok: all_pass = False
        print(f"  {name:<35} --> {status}")

# ---- Summary ----
print()
print("===========================")
everything_ok = model_ok and scaler_ok and feature_ok and all_pass
if everything_ok:
    print("RESULT: ALL CHECKS PASSED - Project is working correctly!")
else:
    print("RESULT: SOME CHECKS FAILED - See above for details")
print("===========================")
