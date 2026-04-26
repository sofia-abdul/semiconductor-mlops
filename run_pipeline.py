from pipeline.validation import validate_data
from pipeline.ingestion import ingest_data
from pipeline.preprocessing import preprocess_data
from pipeline.training import train_pipeline
from pipeline.monitoring import monitor_pipeline


def run_pipeline():
    print("=== Starting SECOM MLOps Pipeline ===\n")

    print("Running validation...")
    validate_data()

    print("\nRunning ingestion...")
    ingest_data()

    print("\nRunning preprocessing...")
    preprocess_data()

    print("\nRunning training...")
    train_pipeline()

    print("\nRunning monitoring...")
    monitor_pipeline()

    print("\n=== Pipeline execution complete ===")


if __name__ == "__main__":
    run_pipeline()