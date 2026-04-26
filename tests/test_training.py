import pandas as pd

from pipeline.config import TARGET_COLUMN
from pipeline.training import evaluate_model, get_model_grid, split_data

from sklearn.linear_model import LogisticRegression


def test_split_data_separates_features_and_target():
    df = pd.DataFrame(
        {
            "feature_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "feature_2": [10, 20, 30, 40, 50, 60, 70, 80],
            TARGET_COLUMN: [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    X_train, X_test, y_train, y_test = split_data(df)

    assert TARGET_COLUMN not in X_train.columns
    assert TARGET_COLUMN not in X_test.columns
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})


def test_model_grid_contains_expected_models_and_parameters():
    model_grid = get_model_grid()

    expected_models = {
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
    }

    assert expected_models.issubset(set(model_grid.keys()))

    for model_config in model_grid.values():
        assert "model" in model_config
        assert "params" in model_config
        assert isinstance(model_config["params"], dict)


def test_evaluate_model_returns_expected_metrics():
    X_train = pd.DataFrame(
        {
            "feature_1": [0.1, 0.2, 0.8, 0.9],
            "feature_2": [1, 2, 8, 9],
        }
    )
    y_train = pd.Series([0, 0, 1, 1])

    X_test = pd.DataFrame(
        {
            "feature_1": [0.15, 0.85],
            "feature_2": [1.5, 8.5],
        }
    )
    y_test = pd.Series([0, 1])

    model = LogisticRegression().fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "confusion_matrix",
        "classification_report",
    }

    assert expected_metrics.issubset(metrics.keys())
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
    assert metrics["confusion_matrix"].shape == (2, 2)