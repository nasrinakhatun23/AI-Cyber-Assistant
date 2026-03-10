import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("../../data/raw/Synthetic_Financial_datasets_log.csv")

# OPTIONAL: Sample 100k for faster training (remove if full training needed)
df = df.sample(100000, random_state=42)

# -----------------------------
# Encode Categorical Column
# -----------------------------
df = pd.get_dummies(df, columns=['type'], drop_first=True)

# -----------------------------
# Define Features & Target
# -----------------------------
X = df.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)
y = df['isFraud']

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Scale Data (Important)
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# -----------------------------
# Save Model & Scaler
# -----------------------------
joblib.dump(model, "../../fraud_model.pkl")
joblib.dump(scaler, "../../scaler.pkl")

print("✅ Model Saved Successfully!")