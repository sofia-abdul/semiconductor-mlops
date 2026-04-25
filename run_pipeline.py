from pipeline.validation import validate_data
from pipeline.ingestion import ingest_data


if __name__ == "__main__":
    print("Running validation...")
    validate_data()

    print("\nRunning ingestion...")
    ingest_data()

    print("\nRunning preprocessing...")
    preprocess_data()

    print("\nRunning training...")
    train_pipeline()