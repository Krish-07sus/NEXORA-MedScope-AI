# import pandas as pd
# import numpy as np
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression

# # ---------- LOAD DATA ----------
# df = pd.read_csv(
#     "/Users/krishsharma/Desktop/mimic-iv/labevents.csv",
#     usecols=["subject_id", "itemid", "valuenum"],
#     nrows=500000   # VERY IMPORTANT (avoid crash)
# )

# print("Loaded rows:", len(df))


# # ---------- ITEM ID MAP ----------
# ITEM_MAP = {
#     50861: "ALT",
#     50878: "AST",
#     50885: "Bilirubin",
#     50862: "Albumin",
#     51265: "Platelets"
# }

# # Filter required tests
# df = df[df["itemid"].isin(ITEM_MAP.keys())]

# # Map names
# df["test"] = df["itemid"].map(ITEM_MAP)

# # ---------- PIVOT ----------
# df = df.pivot_table(
#     index="subject_id",
#     columns="test",
#     values="valuenum",
#     aggfunc="mean"
# )

# df = df.dropna()

# print("After cleaning:", len(df))


# # ---------- TARGET ----------
# df["target"] = (
#     (df["ALT"] > 60) |
#     (df["AST"] > 60) |
#     (df["Bilirubin"] > 1.5) |
#     (df["Albumin"] < 3.5) |
#     (df["Platelets"] < 150)
# ).astype(int)


# # ---------- SPLIT ----------
# X = df.drop("target", axis=1)
# y = df["target"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# # ---------- SCALE ----------
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


# # ---------- MODELS ----------
# rf = RandomForestClassifier(n_estimators=100)
# gb = GradientBoostingClassifier()
# lr = LogisticRegression(max_iter=1000)

# rf.fit(X_train, y_train)
# gb.fit(X_train, y_train)
# lr.fit(X_train, y_train)


# # ---------- SAVE ----------
# joblib.dump(rf, "rf.pkl")
# joblib.dump(gb, "gb.pkl")
# joblib.dump(lr, "lr.pkl")
# joblib.dump(scaler, "scaler.pkl")

# print("✅ Model trained using MIMIC data successfully!")

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_MedScope-AI():

    print("📊 MedScope-AI Model Evaluation")

    # Load models
    rf = joblib.load("rf.pkl")
    gb = joblib.load("gb.pkl")
    lr = joblib.load("lr.pkl")
    scaler = joblib.load("scaler.pkl")

    # ⚠️ You must manually load your dataset again
    import pandas as pd
    df = pd.read_csv("/Users/krishsharma/Desktop/mimic-iv/labevents.csv",
                     usecols=["subject_id", "itemid", "valuenum"],
                     nrows=200000)

    ITEM_MAP = {
        50861: "ALT",
        50878: "AST",
        50885: "Bilirubin",
        50862: "Albumin",
        51265: "Platelets"
    }

    df = df[df["itemid"].isin(ITEM_MAP.keys())]
    df["test"] = df["itemid"].map(ITEM_MAP)

    df = df.pivot_table(index="subject_id",
                        columns="test",
                        values="valuenum",
                        aggfunc="mean").dropna()

    # SAME TARGET LOGIC (important)
    df["target"] = (
        (df["ALT"] > 60) |
        (df["AST"] > 60) |
        (df["Bilirubin"] > 1.5) |
        (df["Albumin"] < 3.5) |
        (df["Platelets"] < 150)
    ).astype(int)

    X = df.drop("target", axis=1)
    y = df["target"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_test = scaler.transform(X_test)

    # Ensemble prediction
    rf_pred = rf.predict(X_test)
    gb_pred = gb.predict(X_test)
    lr_pred = lr.predict(X_test)

    # Majority voting
    y_pred = (rf_pred + gb_pred + lr_pred) >= 2
    y_pred = y_pred.astype(int)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plot
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=["Low","High"],
                yticklabels=["Low","High"],
                cmap='Blues')

    plt.title("Confusion Matrix - MedScope-AI")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


if __name__ == "__main__":
    evaluate_MedScope-AI()