from collections import defaultdict


def build_prefix_groups(feature_cols):
    groups = defaultdict(list)

    allowed_prefixes = {
        "pslist", "dlllist", "handles", "ldrmodules", "malfind",
        "psxview", "modules", "svcscan", "callbacks"
    }

    for feat in feature_cols:
        if "." in feat:
            prefix = feat.split(".", 1)[0].strip().lower()
            if prefix in allowed_prefixes:
                groups[prefix].append(feat)
            else:
                groups["other"].append(feat)
        else:
            groups["other"].append(feat)

    return {k: v for k, v in groups.items() if len(v) > 0}