import pandas as pd

# Known label mappings for common encoded columns
LABEL_MAPS = {
    "sex":    {0: "Female", 1: "Male", "0": "Female", "1": "Male"},
    "gender": {0: "Female", 1: "Male", "0": "Female", "1": "Male"},
    "race":   {0: "Other", 1: "White", "0": "Other", "1": "White"},
    "income": {0: "<=50K", 1: ">50K", "0": "<=50K", "1": ">50K"},
}

# Values treated as "positive" outcome when target column is a string
POSITIVE_VALUES = {'yes', 'true', '1', 'hired', 'approved', 'positive', '>50k', 'accept', 'accepted', 'pass'}


def decode_group_label(sensitive_col, value):
    col = sensitive_col.lower()
    mapping = LABEL_MAPS.get(col)
    if mapping and value in mapping:
        return mapping[value]
    return str(value)

def get_group_stats(df, target_col, sensitive_col):
    stats = {}

    # handle string target columns
    target = df[target_col]
    if target.dtype == object or str(target.dtype) == 'string':
        target = target.str.strip().str.lower().apply(
            lambda x: 1 if x in POSITIVE_VALUES else 0
        )

    for group in df[sensitive_col].unique():
        mask = df[sensitive_col] == group
        label = decode_group_label(sensitive_col, group)
        stats[label] = {
            "count": int(mask.sum()),
            "positive_rate": round(float(target[mask].mean()), 4)
        }
    return stats


def calculate_spd(group_stats):
    rates = [v["positive_rate"] for v in group_stats.values()]
    return round(max(rates) - min(rates), 4)


def calculate_di(group_stats):
    rates = [v["positive_rate"] for v in group_stats.values()]
    return round(min(rates) / max(rates), 4) if max(rates) > 0 else 1.0


def get_severity(spd, di):
    if di < 0.6 or spd > 0.3:
        return "high"
    elif di < 0.8 or spd > 0.1:
        return "medium"
    return "low"


def analyze_bias(df, target_col, sensitive_col):
    group_stats = get_group_stats(df, target_col, sensitive_col)
    spd = calculate_spd(group_stats)
    di = calculate_di(group_stats)

    clean_stats = {
        str(group): {
            "count": int(vals["count"]),
            "positive_rate": float(vals["positive_rate"])
        }
        for group, vals in group_stats.items()
    }

    return {
        "group_stats": clean_stats,
        "spd": float(spd),
        "di": float(di),
        "bias_detected": bool(spd > 0.1 or di < 0.8),
        "severity": get_severity(spd, di)
    }