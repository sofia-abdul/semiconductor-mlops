
## **Semiconductor Yield Prediction MLOps Pipeline**

### **Overview**

This project implements an end-to-end MLOps pipeline for predicting semiconductor manufacturing yield using the UCI SECOM dataset.

The pipeline takes raw sensor data and produces a deployed machine learning model capable of identifying defective wafers, with a focus on handling class imbalance and ensuring reliable performance in a production setting.

---

### **Pipeline Architecture**

The system follows a structured pipeline:

1. **Data Ingestion**
   Loads raw SECOM data into a MariaDB database using SQLAlchemy.

2. **Data Preprocessing**
   Handles missing values, removes low-variance features, and scales inputs for model training.

3. **Model Training**
   Trains and evaluates multiple models (Random Forest, XGBoost, Gradient Boosting, Logistic Regression) using cross-validation and selects the best-performing model.

4. **Model Deployment**
   Serves the trained model using a FastAPI application with `/health` and `/predict` endpoints.

5. **Monitoring**
   Tracks data quality, model performance, and prediction behaviour using a monitoring script and logs.

---

### **Tech Stack**

* Python
* pandas, scikit-learn
* Apache Airflow
* FastAPI + Uvicorn
* MariaDB + SQLAlchemy
* MLflow (experiment tracking)
* Docker (for containerised deployment)

---

### **Model Performance**

The final model selected was a **Random Forest classifier**, achieving:

* F1-score: 0.3846
* Recall: 0.5769
* Precision: 0.2885
* ROC-AUC: 0.8037

The classification threshold was adjusted to improve recall, reflecting the higher cost of missing defective wafers.

---

### **Evaluation Figures**

Evaluation outputs are available in the `figures/` directory, including:

* Class distribution
* Confusion matrix
* ROC curve
* Precision–Recall curve
* Feature importance

---

### **How to Run**

#### 1. Train the model

```bash
python -m pipeline.training
```

#### 2. Run the API

```bash
uvicorn pipeline.api:app --reload
```

Open in browser:

```
http://127.0.0.1:8000/docs
```

#### 3. Run monitoring

```bash
python -m pipeline.monitoring
```

---

### **Project Structure**

```
semiconductor-mlops/
│
├── pipeline/        # Core pipeline scripts
├── dags/            # Airflow DAG definitions
├── models/          # Saved trained models
├── data/outputs/    # Metrics and monitoring reports
├── figures/         # Evaluation visualisations
├── tests/           # Unit tests
├── requirements.txt
└── README.md
```

---

### **Notes**

* The pipeline is designed to be reproducible
* Monitoring enables detection of data drift and performance degradation
* The system can be extended with automated retraining and full containerised deployment

