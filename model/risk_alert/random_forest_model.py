from sklearn.ensemble import RandomForestClassifier

def train_risk_model(X, y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

def predict_risk(model, features):
    prediction = model.predict([features])
    return prediction[0]