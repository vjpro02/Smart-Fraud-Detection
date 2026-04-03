import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    scaler = StandardScaler()
    if "Amount" in X.columns:
        X["Amount"] = scaler.fit_transform(X[["Amount"]])
    if "Time" in X.columns:
        X["Time"] = scaler.fit_transform(X[["Time"]])
    return X, y, scaler

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
