import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import os

# MLflow 서버 시작 방법 (터미널에서 실행)
# mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# MLflow 트래킹 서버 URI 설정 (로컬 서버를 사용하는 경우)
mlflow.set_tracking_uri("http://localhost:5000")

# 실험 이름 설정
experiment_name = "diabetes-prediction"
mlflow.set_experiment(experiment_name)

# 데이터 로드
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 학습 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습 파라미터
n_estimators = 100
max_depth = 5

# MLflow 실험 시작
with mlflow.start_run() as run:
    # 실행 ID 출력
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")
    
    # 모델 학습
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # 예측 및 평가
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # 1. 메트릭 로깅
    mlflow.log_metric("rmse", rmse)
    
    # 2. 태그 로깅 (두 개)
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("data_source", "sklearn_diabetes")
    
    # 3. 아티팩트 저장 (예: 특성 중요도 그래프)
    # 특성 중요도 시각화
    feature_importance = model.feature_importances_
    feature_names = diabetes.feature_names
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)), feature_importance, align='center')
    plt.yticks(range(len(feature_importance)), feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    
    # 임시 파일로 저장 후 MLflow에 아티팩트로 로깅
    feature_importance_path = "feature_importance.png"
    plt.savefig(feature_importance_path)
    plt.close()
    
    mlflow.log_artifact(feature_importance_path)
    
    # 파라미터 로깅
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # 4. 모델 저장
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print(f"Model training completed. RMSE: {rmse:.4f}")
    print(f"Model and artifacts saved. Check the MLflow UI at http://localhost:5000")

# 저장된 임시 파일 삭제
if os.path.exists(feature_importance_path):
    os.remove(feature_importance_path)