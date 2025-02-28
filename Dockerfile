# Base image (Ubuntu + Miniconda)
FROM continuumio/miniconda3

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Conda 환경 생성 및 활성화
RUN conda create -n mlflow_env python=3.8 && conda init bash
SHELL ["/bin/bash", "-c"]

# Conda 환경 활성화 후 필요한 패키지 설치
RUN conda activate mlflow_env && \
    pip install --no-cache-dir mlflow gunicorn numpy pandas scikit-learn

# MLflow 모델 복사 (실제 모델 경로에 맞게 수정)
COPY ./model /app/model

# Gunicorn을 사용하여 MLflow 모델을 API 서버로 실행
CMD ["bash", "-c", "source activate mlflow_env && gunicorn -w 4 -b 0.0.0.0:8080 --timeout 60 'mlflow.models.serve:app'"]
