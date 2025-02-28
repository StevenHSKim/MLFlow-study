# MLFlow-study
이 프로젝트는 [YOLOv11-nano](https://github.com/ultralytics/ultralytics) 모델의 객체 탐지를 위한 하이퍼파라미터 최적화를 수행합니다.
Optuna를 활용하여 최적의 하이퍼파라미터를 탐색하고, MLflow를 이용해 실험 결과를 추적합니다.


## 폴더 구조
```bash
.
├── trainer.py          # HPO 실험 학습 파일
├── build_docker.sh     # 간단한 도커 빌드 예시 파일
├── Dockerfile          # 배포 커스터마이즈된 도커 예시 파일
├── script.sh           # HPO 실험 자동 실행 파일
├── environment.yaml    # 실험 환경 설정 파일
└── mlflow_tutorial.py  # 간단한 random forest 모델 mlflow 서빙 예시 파일
```


## 모델 및 데이터셋 다운로드
[Ultralytics](https://github.com/ultralytics/ultralytics)에 내장되어 있는 YOLOv11-nano 모델과, Pascal VOC 데이터셋을 사용합니다.

```bash
pip install ultralytics
```
