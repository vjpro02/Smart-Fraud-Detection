import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from utils import load_data, preprocess_data, split_data

def main():
    df = load_data("../data/creditcard.csv")
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/fraud_model.pkl")
    joblib.dump(scaler, "../models/scaler.pkl")

    print("Model trained and saved.")

if __name__ == "__main__":
    main()
