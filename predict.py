import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


YES_NO_COLUMNS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
CATEGORICAL_COLUMNS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]


def _as_female(series: pd.Series) -> pd.Series:
    return series.map(lambda value: 1.0 if str(value).lower() == "female" else 0.0)


def _as_yes_no(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return series.map(lambda value: 1.0 if value == "Yes" else 0.0)


def _prepare_rows(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    df = df_raw.copy()
    if "TotalCharges" not in df.columns:
        raise ValueError("La colonne TotalCharges est absente.")
    total = df["TotalCharges"].replace(" ", np.nan)
    total = pd.to_numeric(total, errors="coerce")
    kept = total.notna()
    dropped = df.index[~kept]
    df = df.loc[kept].copy()
    df["TotalCharges"] = np.sqrt(total.loc[kept].astype(float))
    if "gender" in df.columns:
        df["gender"] = _as_female(df["gender"])
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = _as_yes_no(df["SeniorCitizen"])
    for column in YES_NO_COLUMNS:
        if column in df.columns:
            df[column] = _as_yes_no(df[column])
    return df, dropped


def _build_features(df: pd.DataFrame, scaler_names: list[str]) -> pd.DataFrame:
    features = pd.DataFrame(0.0, index=df.index, columns=scaler_names)
    for column in scaler_names:
        if column in df.columns:
            features[column] = df[column].astype(float)
    for column in CATEGORICAL_COLUMNS:
        if column not in df.columns:
            continue
        prefix = f"{column}_"
        levels = [name for name in scaler_names if name.startswith(prefix)]
        values = df[column].astype(str)
        for name in levels:
            features[name] = (values == name[len(prefix):]).astype(float)
    return features


def _bundle_names(bundle: dict, scaler, model) -> tuple[list[str], list[str]]:
    if "scaler_feature_names" in bundle:
        scaler_names = list(bundle["scaler_feature_names"])
    elif hasattr(scaler, "feature_names_in_"):
        scaler_names = list(scaler.feature_names_in_)
    else:
        scaler_names = list(bundle["feature_names"])
    if hasattr(model, "feature_names_in_"):
        model_names = list(model.feature_names_in_)
    else:
        model_names = list(bundle["feature_names"])
    return scaler_names, model_names


def predict_from_dataframe(df_raw: pd.DataFrame, pipeline: str | Path = "model_pipeline.joblib") -> pd.DataFrame:
    """Score raw customer rows with the saved churn pipeline.

    Rows whose TotalCharges cannot be read are omitted. The returned index
    is the index of the rows that were scored. If the bundle stores a
    "threshold", prediction is proba_churn >= threshold.
    """
    bundle = joblib.load(pipeline)
    scaler = bundle["scaler"]
    model = bundle["model"]
    scaler_names, model_names = _bundle_names(bundle, scaler, model)

    prepared, dropped = _prepare_rows(df_raw)
    if len(dropped):
        print(f"{len(dropped)} ligne(s) écartée(s) : TotalCharges manquant ou illisible.")
    features = _build_features(prepared, scaler_names)
    scaled = pd.DataFrame(
        scaler.transform(features[scaler_names]),
        columns=scaler_names,
        index=features.index,
    )
    model_input = scaled[model_names]
    if not hasattr(model, "feature_names_in_"):
        model_input = model_input.to_numpy()  # modèle entraîné sans noms de colonnes (étape d'un Pipeline)
    threshold = bundle.get("threshold")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(model_input)[:, 1]
        if threshold is None:
            scored = model.predict(model_input)
        else:
            scored = (proba >= threshold).astype(int)
        output = pd.DataFrame({"prediction": scored, "proba_churn": proba}, index=features.index)
    else:
        output = pd.DataFrame({"prediction": model.predict(model_input)}, index=features.index)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict churn using saved pipeline",
        epilog=(
            "Predict churn on a CSV using the saved pipeline (model_pipeline.joblib).\n\n"
            "Example:\n"
            "  python predict.py --input data.csv --output preds.csv --pipeline model_pipeline.joblib"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to input CSV with raw customer data")
    parser.add_argument("--output", required=True, help="Path to output CSV for predictions")
    parser.add_argument("--pipeline", default="model_pipeline.joblib", help="Path to saved pipeline bundle")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    pipe_path = Path(args.pipeline)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    if not pipe_path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {pipe_path}")

    df_raw = pd.read_csv(in_path)
    predictions = predict_from_dataframe(df_raw, pipe_path)
    output = predictions
    if "customerID" in df_raw.columns:
        output = pd.concat([df_raw.loc[predictions.index, ["customerID"]], predictions], axis=1)
    output.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
