from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "uci-secom.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "secom_processed.csv"

DB_URL = "mysql+pymysql://student:Pa55W0rd123%23@localhost:3307/pipeline_db"
TABLE_NAME = "secom_data"

RAW_TARGET_COLUMN = "Pass/Fail"
TARGET_COLUMN = "target"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.25

MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

METRICS_PATH = OUTPUT_DIR / "secom_model_performance_metrics.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "secom_feature_importance.csv"
MODEL_PATH = MODELS_DIR / "secom_yield_model.joblib"

MLFLOW_EXPERIMENT_NAME = "secom_yield_prediction"

MONITORING_REPORT_PATH = OUTPUT_DIR / "secom_monitoring_report.csv"
PREDICTION_LOG_PATH = OUTPUT_DIR / "secom_prediction_log.csv"