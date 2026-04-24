from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "semiconductor.csv"

DB_URL = "mysql+pymysql://student:Pa55W0rd123#@localhost:3306/pipeline_db"
TABLE_NAME = "semiconductor_data"

TARGET_COLUMN = "viable"
DROP_COLUMNS = ["sample_id", "data_source"]

# Model settings (used later)
RANDOM_STATE = 42
TEST_SIZE = 0.25

PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "semiconductor_processed.csv"