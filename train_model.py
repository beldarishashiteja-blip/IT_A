"""
NeuroFraud Shield - Model Training Script
Trains a RandomForest + Decision Tree ensemble on synthetic insurance claims data
and saves the model + scaler for serving via Flask API.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import json
import os

np.random.seed(42)
N = 5000

def generate_synthetic_data(n):
    fraud_ratio = 0.25
    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud

    def make_legit(count):
        return {
            "months_as_customer": np.random.randint(12, 480, count),
            "age": np.random.randint(25, 65, count),
            "policy_deductable": np.random.choice([500, 1000, 2000], count),
            "policy_annual_premium": np.random.uniform(500, 1800, count),
            "umbrella_limit": np.random.choice([0, 1000000, 2000000, 3000000], count),
            "capital_gains": np.random.uniform(0, 60000, count),
            "capital_loss": np.random.uniform(0, 30000, count),
            "incident_hour_of_the_day": np.random.randint(6, 22, count),
            "number_of_vehicles_involved": np.random.choice([1, 2], count, p=[0.7, 0.3]),
            "bodily_injuries": np.random.choice([0, 1], count, p=[0.8, 0.2]),
            "witnesses": np.random.randint(0, 4, count),
            "injury_claim": np.random.uniform(0, 10000, count),
            "property_claim": np.random.uniform(0, 8000, count),
            "vehicle_claim": np.random.uniform(0, 15000, count),
            "total_claim_amount": np.random.uniform(5000, 45000, count),
            "police_report_available": np.random.choice([0, 1], count, p=[0.2, 0.8]),
            "insured_sex": np.random.choice([0, 1], count),
            "insured_education_level": np.random.choice([0, 1, 2, 3, 4], count),
            "insured_occupation": np.random.randint(0, 10, count),
            "incident_type": np.random.choice([0, 1, 2, 3], count, p=[0.4, 0.3, 0.2, 0.1]),
            "collision_type": np.random.choice([0, 1, 2], count),
            "incident_severity": np.random.choice([0, 1, 2, 3], count, p=[0.1, 0.4, 0.35, 0.15]),
            "authorities_contacted": np.random.choice([0, 1, 2, 3], count, p=[0.1, 0.6, 0.2, 0.1]),
            "policy_state": np.random.choice([0, 1, 2], count),
            "insured_hobbies": np.random.randint(0, 20, count),
            "auto_make": np.random.randint(0, 15, count),
            "auto_year": np.random.randint(1995, 2020, count),
            "fraud_reported": np.zeros(count, dtype=int)
        }

    def make_fraud(count):
        d = make_legit(count)
        d["fraud_reported"] = np.ones(count, dtype=int)
        # Fraud patterns
        d["total_claim_amount"] = np.random.uniform(35000, 100000, count)
        d["incident_hour_of_the_day"] = np.random.choice(list(range(0,6)) + list(range(22,24)), count)
        d["number_of_vehicles_involved"] = np.random.choice([2, 3, 4], count)
        d["police_report_available"] = np.random.choice([0, 1], count, p=[0.6, 0.4])
        d["witnesses"] = np.random.choice([0, 1], count, p=[0.7, 0.3])
        d["months_as_customer"] = np.random.randint(1, 24, count)
        d["bodily_injuries"] = np.random.choice([1, 2], count, p=[0.5, 0.5])
        d["injury_claim"] = np.random.uniform(15000, 50000, count)
        d["incident_severity"] = np.random.choice([2, 3], count, p=[0.3, 0.7])
        return d

    legit = make_legit(n_legit)
    fraud = make_fraud(n_fraud)

    df_legit = pd.DataFrame(legit)
    df_fraud = pd.DataFrame(fraud)
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

print("Generating synthetic dataset...")
df = generate_synthetic_data(N)
print(f"Dataset shape: {df.shape}, Fraud rate: {df['fraud_reported'].mean():.1%}")

FEATURES = [
    "months_as_customer", "age", "policy_deductable", "policy_annual_premium",
    "umbrella_limit", "capital_gains", "capital_loss", "incident_hour_of_the_day",
    "number_of_vehicles_involved", "bodily_injuries", "witnesses",
    "injury_claim", "property_claim", "vehicle_claim", "total_claim_amount",
    "police_report_available", "insured_sex", "insured_education_level",
    "insured_occupation", "incident_type", "collision_type", "incident_severity",
    "authorities_contacted", "policy_state", "insured_hobbies", "auto_make", "auto_year"
]

X = df[FEATURES]
y = df["fraud_reported"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print("\nTraining models...")

rf = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=2,
                             class_weight="balanced", random_state=42)
rf.fit(X_train_sc, y_train)
rf_pred = rf.predict(X_test_sc)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1  = f1_score(y_test, rf_pred)

gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train_sc, y_train)
gb_pred = gb.predict(X_test_sc)
gb_acc = accuracy_score(y_test, gb_pred)
gb_f1  = f1_score(y_test, gb_pred)

print(f"RandomForest  → Accuracy: {rf_acc:.3f}  F1: {rf_f1:.3f}")
print(f"GradientBoost → Accuracy: {gb_acc:.3f}  F1: {gb_f1:.3f}")
print("\nClassification Report (RandomForest):")
print(classification_report(y_test, rf_pred, target_names=["Legit", "Fraud"]))

# Save best model
best_model = rf if rf_f1 >= gb_f1 else gb
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save feature names and importance
feat_importance = dict(zip(FEATURES, best_model.feature_importances_))
feat_sorted = sorted(feat_importance.items(), key=lambda x: -x[1])

with open("model_meta.json", "w") as f:
    json.dump({
        "features": FEATURES,
        "top_features": feat_sorted[:10],
        "accuracy": round(rf_acc, 4),
        "f1_score": round(rf_f1, 4),
        "model_type": type(best_model).__name__
    }, f, indent=2)

print(f"\n✅ Model saved: model.pkl | Scaler: scaler.pkl | Meta: model_meta.json")
print(f"Top 5 features: {[f for f,_ in feat_sorted[:5]]}")



