#!/bin/bash
RUN_ID="1249f7163fcd5495aae938d3333578878"

# Docker 이미지 빌드
mlflow models build-docker \
    --model-uri "s3://YOLOv11-PascalVOC/tpe_trial_4_20250228_045912/$RUN_ID/artifacts/model"

