from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# -----------------------------
# Load Models
# -----------------------------
# Load BERT for complaint classification
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Load Random Forest for risk assessment
rf_model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Scam categories mapping
CATEGORIES = [
    "Phishing/Identity Theft",
    "Investment Fraud",
    "Online Shopping Scam",
    "Romance Scam",
    "Tech Support Scam"
]

# -----------------------------
# Helper Functions
# -----------------------------
def predict_category(text):
    """Predict scam category using BERT"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return CATEGORIES[predicted_class]

def calculate_risk_score(complaint_text):
    """Calculate fraud risk score based on complaint features"""
    # Simple rule-based scoring
    score = 0
    
    # Keyword-based scoring
    high_risk_keywords = ['urgent', 'lottery', 'winner', 'bank account', 'password', 
                          'otp', 'verify', 'suspend', 'foreign', 'inheritance',
                          'prize', 'congratulations', 'click here', 'link', 'account blocked',
                          'suspended', 'confirm', 'update', 'expire', 'act now']
    
    medium_risk_keywords = ['payment', 'transfer', 'investment', 'offer', 'deal',
                           'money', 'cash', 'reward', 'refund']
    
    text_lower = complaint_text.lower()
    
    # High risk keywords
    for keyword in high_risk_keywords:
        if keyword in text_lower:
            score += 10
    
    # Medium risk keywords
    for keyword in medium_risk_keywords:
        if keyword in text_lower:
            score += 5
    
    # Cap at 100
    score = min(score, 100)
    
    return score

def predict_risk_level(fraud_score):
    """Predict if transaction is high risk (1) or low risk (0)"""
    # Rule-based risk classification
    if fraud_score > 40:
        return 1
    else:
        return 0

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


if __name__ == "__main__":
    app.run(debug=True)







    



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