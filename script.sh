#!/bin/bash

# 세 가지 탐색 방법으로 YOLOv11 최적화 자동 실험 실행 스크립트

echo "Starting YOLOv11 optimization with TPE method..."
python '/userHome/userhome1/kimhaesung/MLFlow/yolov11_trainer_v2.py' --optimize --search_method tpe

echo "Starting YOLOv11 optimization with Grid method..."
python '/userHome/userhome1/kimhaesung/MLFlow/yolov11_trainer_v2.py' --optimize --search_method grid

echo "Starting YOLOv11 optimization with Random method..."
python '/userHome/userhome1/kimhaesung/MLFlow/yolov11_trainer_v2.py' --optimize --search_method random

echo "All experiments completed."