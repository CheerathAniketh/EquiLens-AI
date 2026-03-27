import shap
import pandas as pd
import numpy as np

# Known proxy features that correlate with protected attributes
PROXY_KEYWORDS = [
    "gap", "zip", "cost", "insurance", "prestige",
    "address", "neighborhood", "redline", "parental"
]


def get_shap_values(model, X_train, X_test):
    explainer = shap.TreeExplainer(model)

    X_sample = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    importance = (
        pd.DataFrame({
            "feature": X_sample.columns,
            "importance": np.abs(shap_values).mean(axis=0)
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    top = importance.head(5).to_dict("records")

    # detect proxy variables from ACTUAL features in this dataset
    proxies = [
        f["feature"] for f in top
        if any(kw in f["feature"].lower() for kw in PROXY_KEYWORDS)
    ]

    return {
        "top_features": top,
        "proxy_features": proxies  # real proxies from actual CSV columns
    }