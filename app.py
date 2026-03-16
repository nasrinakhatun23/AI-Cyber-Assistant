from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# -----------------------------
# Load Models
# -----------------------------
# Load Random Forest for risk assessment (optional)
try:
    rf_model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception:
    rf_model = None
    scaler = None

# Scam categories mapping with keywords
CATEGORY_KEYWORDS = {
    "Phishing/Identity Theft": [
        "otp", "password", "bank account", "verify", "account blocked",
        "suspended", "confirm", "credentials", "aadhar", "pan card",
        "kyc", "login", "click here", "link", "phishing", "identity"
    ],
    "Investment Fraud": [
        "investment", "crypto", "bitcoin", "profit", "double money",
        "trading", "stock", "scheme", "high return", "guarantee",
        "telegram", "whatsapp group", "forex", "nft", "wallet"
    ],
    "Online Shopping Scam": [
        "order", "delivery", "amazon", "flipkart", "refund", "product",
        "parcel", "courier", "shopping", "payment failed", "cod",
        "cashback", "offer", "discount", "fake website"
    ],
    "Romance Scam": [
        "love", "dating", "marriage", "friend", "instagram", "facebook",
        "relationship", "meet", "gift", "army", "foreign person",
        "lonely", "romance", "chat", "online friend"
    ],
    "Tech Support Scam": [
        "virus", "hacked", "microsoft", "windows", "computer", "laptop",
        "software", "technical", "support", "remote access", "pop up",
        "call center", "helpline", "repair", "update"
    ],
}

# -----------------------------
# Helper Functions
# -----------------------------
def predict_category(text):
    """Predict scam category using keyword matching"""
    text_lower = text.lower()
    scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[category] = score

    best = max(scores, key=scores.get)
    # If no keyword matched, return generic
    if scores[best] == 0:
        return "Phishing/Identity Theft"
    return best


def calculate_risk_score(complaint_text):
    """Calculate fraud risk score based on complaint features"""
    score = 0
    text_lower = complaint_text.lower()

    high_risk_keywords = [
        'urgent', 'lottery', 'winner', 'bank account', 'password',
        'otp', 'verify', 'suspend', 'foreign', 'inheritance',
        'prize', 'congratulations', 'click here', 'link', 'account blocked',
        'suspended', 'confirm', 'update', 'expire', 'act now'
    ]
    medium_risk_keywords = [
        'payment', 'transfer', 'investment', 'offer', 'deal',
        'money', 'cash', 'reward', 'refund'
    ]

    for keyword in high_risk_keywords:
        if keyword in text_lower:
            score += 10
    for keyword in medium_risk_keywords:
        if keyword in text_lower:
            score += 5

    return min(score, 100)


def predict_risk_level(fraud_score):
    return 1 if fraud_score > 40 else 0

# -----------------------------
# Home Route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    category = None
    complaint = None
    risk = None
    fraud_score = None

    if request.method == "POST":
        # Get form data
        complaint = request.form.get("complaint", "").strip()
        
        if complaint:
            # Predict scam category
            category = predict_category(complaint)
            
            # Calculate fraud score
            fraud_score = calculate_risk_score(complaint)
            
            # Predict risk level
            risk = predict_risk_level(fraud_score)

    return render_template("index.html", 
                         category=category,
                         complaint=complaint,
                         risk=risk,
                         fraud_score=fraud_score)


@app.route("/transaction", methods=["GET", "POST"])
def transaction():
    """Transaction Fraud Detection using trained Random Forest model"""
    result = None
    fraud_prob = None
    error = None

    if request.method == "POST":
        if rf_model is None or scaler is None:
            error = "Random Forest model not loaded."
        else:
            try:
                data = {
                    "step":           float(request.form.get("step", 1)),
                    "amount":         float(request.form.get("amount", 0)),
                    "oldbalanceOrg":  float(request.form.get("oldbalanceOrg", 0)),
                    "newbalanceOrig": float(request.form.get("newbalanceOrig", 0)),
                    "oldbalanceDest": float(request.form.get("oldbalanceDest", 0)),
                    "newbalanceDest": float(request.form.get("newbalanceDest", 0)),
                    "isFlaggedFraud": 0,
                    "type_CASH_OUT":  1 if request.form.get("txn_type") == "CASH_OUT"  else 0,
                    "type_DEBIT":     1 if request.form.get("txn_type") == "DEBIT"     else 0,
                    "type_PAYMENT":   1 if request.form.get("txn_type") == "PAYMENT"   else 0,
                    "type_TRANSFER":  1 if request.form.get("txn_type") == "TRANSFER"  else 0,
                }
                input_df = pd.DataFrame([data])
                scaled   = scaler.transform(input_df)
                pred     = rf_model.predict(scaled)[0]
                fraud_prob = round(rf_model.predict_proba(scaled)[0][1] * 100, 1)
                result   = "FRAUD" if pred == 1 else "NORMAL"
            except Exception as e:
                error = str(e)

    return render_template("transaction.html",
                           result=result,
                           fraud_prob=fraud_prob,
                           error=error)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)







    



# from flask import Flask, render_template, request, url_for, session, redirect
# from transformers import BertTokenizer, BertForSequenceClassification
# from authlib.integrations.flask_client import OAuth
# import torch
# import joblib
# import pandas as pd
# import numpy as np
# import os

# app = Flask(__name__)
# app.secret_key = "any_random_secret_string" # Kuch bhi secret likh dein

# # -----------------------------
# # Google OAuth Configuration
# # -----------------------------
# oauth = OAuth(app)
# google = oauth.register(
#     name='google',
#     client_id='YOUR_GOOGLE_CLIENT_ID', # Apni ID dalein
#     client_secret='YOUR_GOOGLE_CLIENT_SECRET', # Apna Secret dalein
#     access_token_url='https://accounts.google.com/o/oauth2/token',
#     authorize_url='https://accounts.google.com/o/oauth2/auth',
#     api_base_url='https://www.googleapis.com/oauth2/v1/',
#     client_kwargs={'scope': 'openid email profile'},
#     server_metadata_url='https://accounts.google.com/.well-known/openid-configuration'
# )

# # -----------------------------
# # Load Models (Wahi purana logic)
# # -----------------------------
# model_name = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# bert_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)
# rf_model = joblib.load("fraud_model.pkl")
# scaler = joblib.load("scaler.pkl")

# CATEGORIES = ["Phishing/Identity Theft", "Investment Fraud", "Online Shopping Scam", "Romance Scam", "Tech Support Scam"]

# # -----------------------------
# # Auth Routes
# # -----------------------------
# @app.route('/login')
# def login():
#     redirect_uri = url_for('authorize', _external=True)
#     return google.authorize_redirect(redirect_uri)

# @app.route('/callback')
# def authorize():
#     token = google.authorize_access_token()
#     resp = google.get('userinfo')
#     user_info = resp.json()
#     session['user'] = user_info
#     return redirect('/')

# @app.route('/logout')
# def logout():
#     session.pop('user', None)
#     return redirect('/')

# # -----------------------------
# # Home Route (Logic Updated)
# # -----------------------------
# @app.route("/", methods=["GET", "POST"])
# def home():
#     # Agar user logged in nahi hai, to login button dikhayein
#     if 'user' not in session:
#         return render_template("login_page.html") # Naya login page

#     category = None
#     complaint = None
#     risk = None
#     fraud_score = None
#     user = session.get('user')

#     if request.method == "POST":
#         complaint = request.form.get("complaint", "").strip()
#         if complaint:
#             category = predict_category(complaint)
#             fraud_score = calculate_risk_score(complaint)
#             risk = predict_risk_level(fraud_score)

#     return render_template("index.html", 
#                            category=category,
#                            complaint=complaint,
#                            risk=risk,
#                            fraud_score=fraud_score,
#                            user=user)

# # [Purane Helper Functions - predict_category, calculate_risk_score wahi rahenge]

# if __name__ == "__main__":
#     app.run(debug=True)