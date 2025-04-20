import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

MODEL_PATH = "iris_rf_model.joblib"

def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris.feature_names, iris.target_names

def train_model(df, features, model_path=MODEL_PATH):
    X = df[features]
    y = df['species']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, stratify=y_encoded, random_state=0
    )

    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    clf.fit(X_train, y_train)

    joblib.dump((clf, label_encoder, features), model_path)
    print(f"Model saved to {model_path}")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.2%}")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    print("\nConfusion Matrix:")
    print(cm_df)

    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

    print("\n🌿 Feature Importances:")
    fi = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    print(fi)
    fi.plot(kind='barh')
    plt.title("Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

def infer(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run with --mode train first.")

    clf, label_encoder, features = joblib.load(model_path)
    df, _, _ = load_data()
    X = df[features]
    y = label_encoder.transform(df['species'])

    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Inference Accuracy on full dataset: {acc:.2%}")

    # Show first 5 predictions
    print("\nFirst 5 predictions vs actual:")
    for i in range(5):
        pred_label = label_encoder.inverse_transform([y_pred[i]])[0]
        actual_label = label_encoder.inverse_transform([y[i]])[0]
        print(f"Predicted: {pred_label:<15} | Actual: {actual_label}")

def main():
    parser = argparse.ArgumentParser(description="Iris Random Forest Classifier")
    parser.add_argument("--mode", choices=["train", "infer"], required=True, help="Mode to run: train or infer")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to save/load the model")
    args = parser.parse_args()

    df, features, _ = load_data()

    if args.mode == "train":
        train_model(df, features, model_path=args.model_path)
    elif args.mode == "infer":
        infer(model_path=args.model_path)

if __name__ == "__main__":
    main()
