import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


def _encode_binary_yes_no(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: 1 if x == 'Yes' else 0)


def _build_features(df_raw: pd.DataFrame, feature_names, scaler) -> pd.DataFrame:
    df = df_raw.copy()

    # Convert TotalCharges and drop rows with missing after conversion (as in notebook)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan).astype(float)
        df = df.dropna(subset=['TotalCharges'])

    # Encode gender
    if 'gender' in df.columns:
        df['gender'] = df['gender'].apply(lambda x: 1 if str(x).lower() == 'female' else 0)

    # Binary yes/no variables
    binary_vars = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for v in binary_vars:
        if v in df.columns and df[v].dtype == object:
            df[v] = _encode_binary_yes_no(df[v])

    # Base numeric vars
    numerical_vars = ['tenure', 'MonthlyCharges', 'TotalCharges']
    base_cols = [c for c in ['gender'] + binary_vars + numerical_vars if c in df.columns]

    # One-hot for remaining categoricals
    cat_mult = []
    for c in df.columns:
        if c not in base_cols and df[c].dtype == object:
            cat_mult.append(c)
    dummies = pd.get_dummies(data=df[cat_mult], columns=cat_mult, drop_first=True) if len(cat_mult) > 0 else pd.DataFrame(index=df.index)

    X = pd.concat([df[[c for c in base_cols if c in df.columns]], dummies], axis=1)

    # Transform skewed TotalCharges as in notebook
    if 'TotalCharges' in X.columns:
        X['TotalCharges'] = np.sqrt(X['TotalCharges'])

    # Align to training feature names
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    # Scale
    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=feature_names, index=X.index)


essage = """
Predict churn on a CSV using the saved pipeline (model_pipeline.joblib).

Example:
  python predict.py --input data.csv --output preds.csv --pipeline model_pipeline.joblib
"""


def main():
    parser = argparse.ArgumentParser(description='Predict churn using saved pipeline', epilog=essage,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, help='Path to input CSV with raw customer data')
    parser.add_argument('--output', required=True, help='Path to output CSV for predictions')
    parser.add_argument('--pipeline', default='model_pipeline.joblib', help='Path to saved pipeline bundle')
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    pipe_path = Path(args.pipeline)

    if not in_path.exists():
        raise FileNotFoundError(f'Input file not found: {in_path}')
    if not pipe_path.exists():
        raise FileNotFoundError(f'Pipeline file not found: {pipe_path}')

    # Load inputs
    df_raw = pd.read_csv(in_path)
    bundle = joblib.load(pipe_path)
    feature_names = bundle['feature_names']
    scaler = bundle['scaler']
    model = bundle['model']

    # Build features and predict
    Xp = _build_features(df_raw, feature_names, scaler)
    preds = model.predict(Xp)
    proba = model.predict_proba(Xp)[:, 1] if hasattr(model, 'predict_proba') else None

    # Save results
    out = pd.DataFrame({'prediction': preds}, index=Xp.index)
    if proba is not None:
        out['proba_churn'] = proba

    # If customerID exists, include it in output
    if 'customerID' in df_raw.columns:
        out = pd.concat([df_raw[['customerID']].reset_index(drop=True), out.reset_index(drop=True)], axis=1)

    out.to_csv(out_path, index=False)
    print(f'Saved predictions to {out_path}')


if __name__ == '__main__':
    main()
