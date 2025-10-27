# config.py
BASE = "/opt/airflow"
SCRIPTS = f"{BASE}/scripts"
DATA = f"{BASE}/data"
DATAMART = f"{BASE}/datamart"
FEATURE_STORE = f"{DATAMART}/gold/feature_store/"
LABEL_STORE = f"{DATAMART}/gold/label_store/"
MODEL_BANK = f"{DATAMART}/model_bank/"
PRED_STORE = f"{DATAMART}/gold/predictions/"
MONITOR_STORE = f"{DATAMART}/gold/monitoring/"

import os
raw_data_file = {
    "attributes": os.path.join(DATA, 'features_attributes.csv'),
    "financials": os.path.join(DATA, 'features_financials.csv'),
    "clickstream": os.path.join(DATA, 'feature_clickstream.csv'),
    "lms": os.path.join(DATA, 'lms_loan_daily.csv'),
}

raw_config = [
    {'src':'fe_attributes', 'filename':'features_attributes.csv', 'dir': f"{DATA}/"},    
    {'src':'fe_financials', 'filename':'features_financials.csv', 'dir': f"{DATA}/"},    
    {'src':'fe_clickstream', 'filename':'feature_clickstream.csv', 'dir': f"{DATA}/"},    
    {'src':'lms', 'filename':'lms_loan_daily.csv', 'dir': f"{DATA}/"},    
]
