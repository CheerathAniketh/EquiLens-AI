import shap
import pandas as pd
import numpy as np


def get_shap_values(model, X_train, X_test):
    background = shap.sample(X_train, min(100, len(X_train)))
    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer.shap_values(X_test)

    # LinearExplainer on a binary classifier returns a single 2D array.
    # Guard anyway: if it comes back as a list (multiclass), take class-1 slice.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    importance = (
        pd.DataFrame({
            "feature": X_test.columns,
            "importance": np.abs(shap_values).mean(axis=0)
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "top_features": importance.head(5).to_dict("records")
        # shap_values array dropped — not used by frontend, saves bandwidth
    }