import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from pipeline.config import (
    METRICS_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)

warnings.filterwarnings("ignore")


def load_processed_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print("Loaded processed dataset")
    print("Shape:", df.shape)
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nData split complete")
    print("Training set:", X_train.shape)
    print("Testing set:", X_test.shape)

    print("\nTraining target distribution:")
    print((y_train.value_counts(normalize=True) * 100).round(2))

    print("\nTesting target distribution:")
    print((y_test.value_counts(normalize=True) * 100).round(2))

    return X_train, X_test, y_train, y_test


def get_model_grid() -> dict:
    model_grid = {
        "Logistic Regression": {
            "model": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "C": [0.1, 1.0, 10.0],
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
        },
    }

    if XGBOOST_AVAILABLE:
        model_grid["XGBoost"] = {
            "model": XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
                "subsample": [0.8, 1.0],
            },
        }

    return model_grid


def evaluate_model(model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(
            y_test,
            predictions,
            zero_division=0,
        ),
    }


def train_models(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    model_grid = get_model_grid()
    results = []

    selected_model = None
    selected_model_name = None
    selected_f1 = -1

    for model_name, config in model_grid.items():
        print(f"\nTraining model: {model_name}")

        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring="f1",
            cv=5,
            n_jobs=-1,
        )

        with mlflow.start_run(run_name=model_name):
            grid_search.fit(X_train, y_train)

            tuned_model = grid_search.best_estimator_

            cv_scores = cross_val_score(
                tuned_model,
                X_train,
                y_train,
                cv=5,
                scoring="f1",
                n_jobs=-1,
            )

            metrics = evaluate_model(tuned_model, X_test, y_test)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_params(grid_search.best_params_)

            mlflow.log_metric("cv_f1_mean", cv_scores.mean())
            mlflow.log_metric("cv_f1_std", cv_scores.std())
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])
            mlflow.log_metric("f1_score", metrics["f1_score"])

            mlflow.sklearn.log_model(tuned_model, artifact_path="model")

            print("Best parameters:", grid_search.best_params_)
            print("Cross-validation F1 mean:", round(cv_scores.mean(), 4))
            print("Cross-validation F1 std:", round(cv_scores.std(), 4))
            print("Test accuracy:", round(metrics["accuracy"], 4))
            print("Test precision:", round(metrics["precision"], 4))
            print("Test recall:", round(metrics["recall"], 4))
            print("Test F1-score:", round(metrics["f1_score"], 4))

            print("\nConfusion Matrix:")
            print(metrics["confusion_matrix"])

            print("\nClassification Report:")
            print(metrics["classification_report"])

            results.append(
                {
                    "model": model_name,
                    "best_params": grid_search.best_params_,
                    "cv_f1_mean": cv_scores.mean(),
                    "cv_f1_std": cv_scores.std(),
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                }
            )

            if metrics["f1_score"] > selected_f1:
                selected_f1 = metrics["f1_score"]
                selected_model = tuned_model
                selected_model_name = model_name

    results_df = pd.DataFrame(results).sort_values(
        by="f1_score",
        ascending=False,
    )

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(selected_model, MODEL_PATH)
    results_df.to_csv(METRICS_PATH, index=False)

    print("\nModel comparison:")
    print(results_df[["model", "cv_f1_mean", "accuracy", "precision", "recall", "f1_score"]])

    print(f"\nSelected model: {selected_model_name}")
    print(f"Selected F1-score: {selected_f1:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")

    return results_df


def train_pipeline() -> pd.DataFrame:
    df = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(df)
    results = train_models(X_train, X_test, y_train, y_test)
    return results


if __name__ == "__main__":
    train_pipeline()