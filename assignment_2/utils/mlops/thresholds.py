# performance guardrails
PERF_THRESHOLDS = {  
    "auc_min": 0.62,
    "gini_min": 0.24,
    "accuracy_min": 0.70,
    "logloss_max": 0.60
}
# duplicate of PERF_THRESHOLDS , but remove text after "_" in key
PERF_MERTRICS = {k.split("_")[0]: v for k, v in PERF_THRESHOLDS.items()}

STABILITY_THRESHOLDS = {
    "psi_warn": 0.10,
    "psi_alert": 0.25,   # population score drift (PSI) alert threshold
    "csi_warn": 0.10,
    "csi_alert": 0.25   # feature drift (CSI) alert threshold (max across features)
}