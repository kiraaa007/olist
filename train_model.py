"""Train an Olist Random Forest model that predicts late delivery.

This follows the same workflow as sales-dataset-predict/train_model.py:
load and clean the data, create a binary target, select mixed features,
split the data, train/evaluate a pipeline, and save model.pkl.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_BASE_URL = (
    "https://raw.githubusercontent.com/kiraaa007/"
    "Olist-E-Commerce-Analytics/main/data"
)

DATA_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

MODEL_FEATURES = [
    "total_sales",
    "total_freight",
    "item_count",
    "estimated_delivery_days",
    "purchase_month",
    "product_category",
    "customer_state",
    "seller_state",
]

NUMERIC_FEATURES = MODEL_FEATURES[:5]
CATEGORICAL_FEATURES = MODEL_FEATURES[5:]
RANDOM_STATE = 42


def _mode_or_unknown(values: pd.Series) -> str:
    values = values.dropna().astype(str)
    if values.empty:
        return "Unknown"
    modes = values.mode()
    return str(sorted(modes.tolist())[0] if not modes.empty else values.iloc[0])


def _read_csv(filename: str, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Read locally when available; otherwise use the existing Olist data repo."""
    configured_dir = data_dir or os.getenv("OLIST_DATA_DIR")
    candidates = []
    if configured_dir:
        candidates.append(Path(configured_dir) / filename)
    candidates.extend(
        [
            Path(__file__).resolve().parent / "data" / filename,
            Path(__file__).resolve().parent.parent
            / "Olist-E-Commerce-Analytics"
            / "data"
            / filename,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    return pd.read_csv(f"{DATA_BASE_URL}/{filename}")


def prepare_order_data(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Build one leakage-safe analytical row per delivered order."""
    tables = {
        name: _read_csv(filename, data_dir)
        for name, filename in DATA_FILES.items()
    }

    orders = tables["orders"].drop_duplicates("order_id").copy()
    date_columns = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for column in date_columns:
        orders[column] = pd.to_datetime(orders[column], errors="coerce")

    orders = orders.loc[
        orders["order_status"].eq("delivered")
        & orders[date_columns].notna().all(axis=1)
    ].copy()

    customers = tables["customers"].drop_duplicates("customer_id")

    products = tables["products"].drop_duplicates("product_id").merge(
        tables["translation"].drop_duplicates("product_category_name"),
        on="product_category_name",
        how="left",
        validate="many_to_one",
    )
    products["product_category"] = products[
        "product_category_name_english"
    ].fillna(products["product_category_name"])

    sellers = tables["sellers"].drop_duplicates("seller_id")
    items = (
        tables["items"]
        .drop_duplicates()
        .merge(
            products[["product_id", "product_category"]],
            on="product_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            sellers[["seller_id", "seller_state"]],
            on="seller_id",
            how="left",
            validate="many_to_one",
        )
    )
    item_summary = items.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        total_sales=("price", "sum"),
        total_freight=("freight_value", "sum"),
        product_category=("product_category", _mode_or_unknown),
        seller_state=("seller_state", _mode_or_unknown),
    )

    payment_summary = (
        tables["payments"]
        .drop_duplicates()
        .groupby("order_id", as_index=False)
        .agg(total_payment_value=("payment_value", "sum"))
    )
    review_summary = (
        tables["reviews"]
        .drop_duplicates()
        .groupby("order_id", as_index=False)
        .agg(average_review_score=("review_score", "mean"))
    )

    data = (
        orders.merge(
            customers[["customer_id", "customer_unique_id", "customer_state"]],
            on="customer_id",
            how="left",
            validate="many_to_one",
        )
        .merge(item_summary, on="order_id", how="inner", validate="one_to_one")
        .merge(payment_summary, on="order_id", how="left", validate="one_to_one")
        .merge(review_summary, on="order_id", how="left", validate="one_to_one")
    )

    purchase = data["order_purchase_timestamp"]
    data["purchase_month"] = purchase.dt.month
    data["purchase_period"] = purchase.dt.to_period("M").astype(str)
    data["estimated_delivery_days"] = (
        data["order_estimated_delivery_date"] - purchase
    ).dt.total_seconds() / 86_400
    data["delivery_days"] = (
        data["order_delivered_customer_date"] - purchase
    ).dt.total_seconds() / 86_400

    # The actual delivery timestamp creates the historical target only. It is
    # not included in MODEL_FEATURES, preventing target leakage.
    data["is_late"] = (
        data["order_delivered_customer_date"]
        > data["order_estimated_delivery_date"]
    ).astype(int)
    data["delivery_status"] = data["is_late"].map(
        {0: "On Time", 1: "Late"}
    )

    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].fillna("Unknown").astype(str)

    data = data.drop_duplicates("order_id").reset_index(drop=True)
    if not data["order_id"].is_unique:
        raise AssertionError("Order aggregation produced duplicate order IDs.")
    if data["is_late"].nunique() != 2:
        raise ValueError("The late-delivery target must contain both classes.")
    return data


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )


def train_model(
    order_data: pd.DataFrame | None = None,
    data_dir: str | Path | None = None,
    model_file: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    data = order_data if order_data is not None else prepare_order_data(data_dir)
    X = data[MODEL_FEATURES]
    y = data["is_late"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "orders": int(len(data)),
        "late_orders": int(y.sum()),
        "late_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["On Time", "Late"],
            zero_division=0,
        ),
    }
    package = {
        "model": model,
        "features": MODEL_FEATURES,
        "target": "is_late",
        "metrics": metrics,
    }

    output = Path(
        model_file or os.getenv("OLIST_MODEL_FILE", "model.pkl")
    )
    # Save the fitted pipeline itself, matching sales-dataset-predict. The
    # Streamlit app only loads this artifact and never trains at launch.
    dump(model, output)

    if verbose:
        print("Dataset Loaded and Cleaned Successfully!")
        print(f"Orders       : {metrics['orders']:,}")
        print(f"Late Orders  : {metrics['late_orders']:,}")
        print(f"Late Rate    : {metrics['late_rate']:.2%}")
        print("\n========== MODEL EVALUATION ==========")
        for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            print(f"{name.replace('_', ' ').title():<10}: {metrics[name]:.4f}")
        print("\nConfusion Matrix")
        print(metrics["confusion_matrix"])
        print("\nClassification Report")
        print(metrics["classification_report"])
        print(f"Model saved as: {output}")
    return package


if __name__ == "__main__":
    train_model()
