import joblib

files = [
    "heart_model.pkl",
    "kidney_model.pkl",
    "mimic_model.pkl"
]

for file in files:
    print("\n" + "="*60)
    print("FILE:", file)

    try:
        model = joblib.load(file)

        print("TYPE:", type(model))
        print("MODEL:", model)

        if hasattr(model, "n_features_in_"):
            print("INPUT FEATURES:", model.n_features_in_)

        if hasattr(model, "feature_names_in_"):
            print("FEATURE NAMES:", model.feature_names_in_)

        if hasattr(model, "classes_"):
            print("CLASSES:", model.classes_)

    except Exception as e:
        print("ERROR:", e)