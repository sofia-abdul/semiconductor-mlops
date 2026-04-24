if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/raw/semiconductor.csv")

    print("Dataset loaded successfully")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isnull().sum())