from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "semiconductor.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "semiconductor_processed.csv"

DB_URL = "mysql+pymysql://student:Pa55W0rd123%23@localhost:3307/pipeline_db"
TABLE_NAME = "semiconductor_data"

TARGET_COLUMN = "viable"
DROP_COLUMNS = ["sample_id", "data_source"]

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.25

MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = BASE_DIR / "data" / "outputs" / "model_performance_metrics.csv"
MODEL_PATH = MODELS_DIR / "semiconductor_viability_model.joblib"

MLFLOW_EXPERIMENT_NAME = "semiconductor_viability_prediction"