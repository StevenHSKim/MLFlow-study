import os
import argparse
import yaml
import json
from pathlib import Path
import time
from datetime import datetime

import torch

import optuna
from optuna.samplers import TPESampler, GridSampler, RandomSampler
import mlflow

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER, colorstr

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv11 nano 모델 전이학습 및 평가')
    parser.add_argument('--data', type=str, default='VOC.yaml', help='데이터셋 설정 파일 경로')
    parser.add_argument('--img_size', type=int, default=640, help='이미지 크기')
    
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=5, help='학습 에폭 수')
    
    parser.add_argument('--workers', type=int, default=8, help='데이터 로딩 워커 수')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    parser.add_argument('--project', default='results', help='결과 저장 디렉토리')
    parser.add_argument('--name', default='YOLOv11-PascalVOC-v2', help='실험 이름')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='가장 최근 학습을 이어서 진행')
    
    parser.add_argument('--lr', type=float, default=0.01, help='학습률')
    
    parser.add_argument('--n_trials', type=int, default=5, help='Optuna 실험 횟수')
    parser.add_argument('--optimize', action='store_true', help='Optuna를 사용한 하이퍼파라미터 최적화 진행')
    
    parser.add_argument('--search_method', type=str, default='tpe', choices=['tpe', 'grid', 'random'], help='Optuna 하이퍼파라미터 탐색 방법 (tpe, grid, random)')
    
    return parser.parse_args()

def disable_yolo_mlflow():
    """
    YOLO의 MLflow 기능 비활성화 시도 (설정 파일 직접 수정)
    """
    try:
        config_file = os.path.expanduser("~/.config/Ultralytics/settings.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                settings = json.load(f)
            
            # MLflow 설정 비활성화
            if 'mlflow' in settings:
                settings['mlflow'] = False
            
            # 수정된 설정 저장
            with open(config_file, 'w') as f:
                json.dump(settings, f, indent=2)
            
            print(f"Ultralytics MLflow를 비활성화했습니다.")
        else:
            print(f"Ultralytics 설정 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"YOLO 설정 변경 중 오류 발생: {e}")

def objective(trial, args, best_model_info=None):
    """
    Optuna 최적화를 위한 목적 함수
    """
    
    # 실험 시작 시간
    start_time = time.time()
    
    # MLflow 실험 시작 - 독립적인 디렉토리 사용
    run_name = f"{args.search_method}_trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_tracking_uri(f"file://{os.path.abspath('optuna_mlruns')}")
    mlflow.set_experiment(f"{args.name}")
    
    # 선택한 탐색 방법에 따라 하이퍼파라미터 샘플링
    # TPE와 Random은 범위 내에서 샘플링하지만, Grid는 사전 정의된 값을 사용
    if args.search_method == 'grid':
        # Grid Search에서는 사전에 정의된 값만 사용 가능
        # suggest_float 대신에 suggest_categorical 사용
        lr_options = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # 그리드 탐색용 학습률 값
        lr = trial.suggest_categorical('lr', lr_options)
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
        weight_decay_options = [1e-5, 1e-4, 1e-3, 1e-2]  # 그리드 탐색용 가중치 감쇠 값
        weight_decay = trial.suggest_categorical('weight_decay', weight_decay_options)
        momentum_options = [0.8, 0.85, 0.9, 0.95, 0.99]  # 그리드 탐색용 모멘텀 값
        momentum = trial.suggest_categorical('momentum', momentum_options)
        warmup_epochs = trial.suggest_categorical('warmup_epochs', [1, 2, 3, 4, 5])
    else:  # tpe 또는 random
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
        warmup_epochs = trial.suggest_int('warmup_epochs', 1, 5)
    
    # 최적화에서 고정된 파라미터
    epochs = min(args.epochs, 30)  # 최적화에서는 에폭 수를 줄여 시간 절약
    
    # 결과 저장 경로 생성
    save_dir = Path(args.project) / f"{args.name}_trial_{trial.number}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # YOLOv11 nano 모델 로드
    model = YOLO('yolo11n.pt')
    
    # 학습 설정 - 유효한 인자만 사용
    train_args = {
        'data': args.data,
        'imgsz': args.img_size,
        'epochs': epochs,
        'batch': batch_size,
        'workers': args.workers,
        'device': args.device,
        'project': args.project,
        'name': f"{args.name}_trial_{trial.number}",
        'resume': args.resume,
        'lr0': lr,
        'lrf': 0.01,  # 최종 학습률 비율
        'momentum': momentum,
        'weight_decay': weight_decay,
        'warmup_epochs': warmup_epochs,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'optimizer': 'AdamW',
        'patience': 20,  # 최적화에서는 EarlyStopping 빨리 적용
        'save': True,
        'save_period': -1,
        'plots': True,
        'rect': False,
        'cos_lr': True,
    }
    
    map_metric = 0.0
    
    # with 문으로 MLflow 실험 세션 관리
    with mlflow.start_run(run_name=run_name) as run:
        try:
            # MLflow에 태그 설정 - 검색 방법론 추적을 위해
            mlflow.set_tag("search_method", args.search_method)
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("optimization_run", True)
            
            # MLflow에 파라미터 기록
            for key, value in train_args.items():
                mlflow.log_param(key, value)
            
            # 학습 실행
            print(colorstr('bold', 'green', f'Trial {trial.number} 학습 시작... (검색 방법: {args.search_method})'))
            results = model.train(**train_args)
            
            # 학습 후 결과를 수집하기 위해 모델 상태 확인 (필요한 경우)
            if hasattr(results, 'results'):
                for epoch, result in enumerate(results.results):
                    if isinstance(result, dict):
                        for k, v in result.items():
                            if isinstance(v, (int, float)):
                                mlflow.log_metric(f"epoch_{k}", v, step=epoch)
            
            # 검증 실행
            print(colorstr('bold', 'green', f'Trial {trial.number} 검증 시작...'))
            val_results = model.val(data=args.data, batch=batch_size, imgsz=args.img_size)
            
            # 주요 메트릭 추출
            metrics = {
                'mAP50': float(val_results.box.map50),
                'mAP50-95': float(val_results.box.map),
                'precision': float(val_results.box.mp),
                'recall': float(val_results.box.mr),
                'fitness': float(val_results.fitness)
            }
            
            # MLflow에 메트릭 기록
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # 훈련 시간 기록
            training_time = time.time() - start_time
            mlflow.log_metric('training_time', training_time)
            
            # 모델 가중치 저장
            best_model_path = save_dir / 'weights' / 'best.pt'
            if best_model_path.exists():
                mlflow.log_artifact(str(best_model_path))
                
                # 현재 모델이 지금까지 발견된 최고의 모델인지 확인
                current_map = metrics['mAP50-95']
                
                # 이 모델의 메타데이터 생성
                model_info = {
                    'model_type': 'YOLOv11-nano',
                    'dataset': args.data,
                    'img_size': args.img_size,
                    'search_method': args.search_method,
                    'trial_number': trial.number,
                    'mAP50': metrics['mAP50'],
                    'mAP50-95': current_map,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'fitness': metrics['fitness'],
                    'training_time': training_time,
                    'lr': lr,
                    'batch_size': batch_size,
                    'momentum': momentum,
                    'weight_decay': weight_decay,
                    'warmup_epochs': warmup_epochs,
                    'run_id': run.info.run_id,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # 현재 최고 성능 모델보다 좋은 성능을 보일 경우 정보 업데이트
                if best_model_info is None or current_map > best_model_info['map_metric']:
                    # MLflow에 모델 저장
                    # PyTorch 모델을 MLflow에 저장 (모델 등록에 사용됨)
                    mlflow.pytorch.log_model(
                        pytorch_model=model.model,  # YOLOv11 내부 PyTorch 모델
                        artifact_path="yolov11_model",
                        registered_model_name=f"YOLOv11_{args.name}_{args.search_method}_best",
                        metadata=model_info
                    )
                    
                    # YOLO 모델 형식으로 별도 저장
                    from urllib.parse import urlparse

                    # MLflow 아티팩트 URI에서 파일 경로 추출
                    artifact_uri = mlflow.get_artifact_uri()
                    if artifact_uri.startswith('file:'):
                        artifact_path = urlparse(artifact_uri).path  # URI에서 파일 경로 부분만 추출
                    else:
                        artifact_path = artifact_uri  # 다른 형태의 URI인 경우 처리 필요

                    # YOLO 모델 저장 경로 설정
                    yolo_model_dir = os.path.join(artifact_path, "yolo_model")
                    os.makedirs(yolo_model_dir, exist_ok=True)
                    model_save_path = os.path.join(yolo_model_dir, "best.pt")

                    # 모델 저장
                    model.save(model_save_path)
                    mlflow.log_artifact(model_save_path, "yolo_model")
                    

                    
                    # 모델 내보내기
                    print(colorstr('bold', 'green', '모델 내보내기...'))
                    export_formats = ['onnx', 'torchscript']
                    for export_format in export_formats:
                        try:
                            export_path = model.export(format=export_format, imgsz=args.img_size)
                            mlflow.log_artifact(str(export_path))
                        except Exception as e:
                            print(f"모델 내보내기 실패 ({export_format}): {e}")
                    
                    # 최고 성능 모델 정보 업데이트
                    best_model_info = {
                        'trial_number': trial.number,
                        'params': {
                            'lr': lr,
                            'batch_size': batch_size,
                            'momentum': momentum,
                            'weight_decay': weight_decay,
                            'warmup_epochs': warmup_epochs
                        },
                        'metrics': metrics,
                        'map_metric': current_map,
                        'model_path': str(best_model_path),
                        'run_id': run.info.run_id,
                        'training_time': training_time
                    }
                    
                    # 최고 성능 모델 정보 파일 저장
                    best_model_info_path = Path(args.project) / f"{args.name}_{args.search_method}_best_model_info.json"
                    with open(best_model_info_path, 'w') as f:
                        json.dump(best_model_info, f, indent=2)
                    
                    print(f"\n{colorstr('bold', 'blue', '새로운 최고 성능 모델 발견!')}")
                    print(f"Trial: {trial.number}, mAP50-95: {current_map:.4f}")
                    print(f"최고 성능 모델 정보가 저장되었습니다: {best_model_info_path}\n")
            
            # 최적화 목표 반환 (mAP50-95 사용)
            map_metric = metrics['mAP50-95']
            print(f"Trial {trial.number} 완료: mAP50-95 = {map_metric:.4f}")
            
        except Exception as e:
            print(f"Trial {trial.number} 중 오류 발생: {e}")
            # 실패한 경우 낮은 점수 반환
    
    return map_metric, best_model_info

def run_optimization(args):
    """
    Optuna를 사용한 하이퍼파라미터 최적화 실행
    """
    # MLflow 설정 - 독립적인 디렉토리 사용
    mlflow.set_tracking_uri(f"file://{os.path.abspath('optuna_mlruns')}")
    mlflow.set_experiment(f"{args.name}")
    
    # 디렉토리 생성
    os.makedirs("optuna_mlruns", exist_ok=True)
    
    # 데이터셋 설정
    data_dict = check_det_dataset(args.data)
    print(f"데이터셋 정보: {data_dict}")
    
    # 탐색 방법론 설정
    sampler = None
    if args.search_method == 'tpe':
        print(colorstr('bold', 'blue', f'TPE(Tree-structured Parzen Estimator) 방법으로 하이퍼파라미터 탐색 진행'))
        sampler = TPESampler(seed=42)  # 재현성을 위해 시드 설정
    elif args.search_method == 'random':
        print(colorstr('bold', 'blue', f'Random Search 방법으로 하이퍼파라미터 탐색 진행'))
        sampler = RandomSampler(seed=42)  # 재현성을 위해 시드 설정
    elif args.search_method == 'grid':
        print(colorstr('bold', 'blue', f'Grid Search 방법으로 하이퍼파라미터 탐색 진행'))
        # Grid Search는 사전에 정의된 탐색 공간이 필요함
        search_space = {
            'lr': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'batch_size': [4, 8, 16, 32],
            'weight_decay': [1e-5, 1e-4, 1e-3, 1e-2],
            'momentum': [0.8, 0.85, 0.9, 0.95, 0.99],
            'warmup_epochs': [1, 2, 3, 4, 5]
        }
        grid_points = {}
        for key, values in search_space.items():
            grid_points[key] = values
        # 참고: Grid Search의 경우 주어진 탐색 공간의 모든 조합을 시도하므로 
        # n_trials가 너무 작으면 일부 조합만 탐색할 수 있음
        sampler = GridSampler(grid_points)
        # Grid Search에서는 전체 탐색 공간의 크기를 계산
        total_grid_points = 1
        for values in search_space.values():
            total_grid_points *= len(values)
        print(f"Grid Search 전체 탐색 공간 크기: {total_grid_points}개 조합")
        if args.n_trials < total_grid_points:
            print(f"경고: 지정된 실험 횟수({args.n_trials})가 전체 Grid 탐색 공간({total_grid_points})보다 작습니다.")
            print(f"일부 조합만 탐색될 수 있습니다. 필요시 --n_trials 값을 늘려주세요.")
    
    # 최고 성능 모델 정보를 저장할 변수
    best_model_info = None
    
    # trial마다 최고의 모델을 추적하는 콜백 함수 정의
    def objective_wrapper(trial):
        nonlocal best_model_info
        map_metric, updated_best_model_info = objective(trial, args, best_model_info)
        best_model_info = updated_best_model_info  # 최고 성능 모델 정보 업데이트
        return map_metric
    
    # Optuna 최적화 시작
    print(colorstr('bold', 'green', f'Optuna 최적화 시작 - {args.n_trials} 실험 예정...'))
    study = optuna.create_study(
        direction="maximize",
        study_name=f"YOLOv11-Optimization-{args.name}-{args.search_method}",
        sampler=sampler
    )
    
    try:
        study.optimize(
            objective_wrapper,
            n_trials=args.n_trials,
            catch=(Exception,)
        )
        
        # 최적화 결과 출력
        print("\n" + "="*80)
        print(colorstr('bold', 'green', '최적화 완료!'))
        print(f"최적 파라미터: {study.best_params}")
        print(f"최적 mAP50-95: {study.best_value:.4f}")
        print(f"탐색 방법: {args.search_method}")
        print("="*80 + "\n")
        
        # 최고 성능 모델 정보 파일 경로
        best_model_info_path = Path(args.project) / f"{args.name}_{args.search_method}_best_model_info.json"
        
        if best_model_info and os.path.exists(best_model_info_path):
            print(colorstr('bold', 'green', f'최고 성능 모델 정보:'))
            print(f"Trial 번호: {best_model_info['trial_number']}")
            print(f"mAP50-95: {best_model_info['map_metric']:.4f}")
            print(f"하이퍼파라미터: {best_model_info['params']}")
            print(f"최고 성능 모델 경로: {best_model_info['model_path']}")
            print(f"MLflow Run ID: {best_model_info['run_id']}")
            print(f"전체 정보: {best_model_info_path}")
        
        # 최적화 결과 저장
        result_file = Path(args.project) / f"{args.name}_{args.search_method}_optuna_results.json"
        with open(result_file, 'w') as f:
            json.dump({
                'best_params': study.best_params,
                'best_value': study.best_value,
                'study_name': study.study_name,
                'search_method': args.search_method,
                'direction': 'maximize',
                'n_trials': args.n_trials,
                'best_trial_number': best_model_info['trial_number'] if best_model_info else None,
                'best_model_run_id': best_model_info['run_id'] if best_model_info else None,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        print(f"최적화 결과가 저장되었습니다: {result_file}")
        
    except Exception as e:
        print(f"최적화 중 오류 발생: {e}")

def main():
    # 인자 파싱
    args = parse_args()
    
    # CUDA 장치 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")
    
    # YOLO MLflow 비활성화 시도
    disable_yolo_mlflow()
    
    # MLflow 디렉토리 생성
    os.makedirs("optuna_mlruns", exist_ok=True)
    
    # 내장 VOC 데이터셋을 사용한다고 안내
    print(f"Ultralytics 내장 Pascal VOC 데이터셋을 사용합니다: '{args.data}'")
    print(f"하이퍼파라미터 탐색 방법: {args.search_method}")
    
    if args.optimize:
        # Optuna를 사용한 하이퍼파라미터 최적화 실행
        run_optimization(args)
    else:
        # 최적화 없이 기본 파라미터로 훈련 실행
        # 기존 run_final_training 함수는 제거하고 단일 trial로 optimization 실행
        print(colorstr('bold', 'green', '최적화 비활성화: 기본 파라미터로 단일 훈련 실행'))
        data_dict = check_det_dataset(args.data)
        
        # 기본 파라미터로 단일 trial 실행
        class SingleTrial:
            def __init__(self):
                self.number = 0
                
            def suggest_categorical(self, name, choices):
                if name == 'batch_size':
                    return args.batch_size
                return choices[0]  # 기본값으로 첫 번째 선택지 사용
                
            def suggest_float(self, name, low, high, **kwargs):
                if name == 'lr':
                    return args.lr
                elif name == 'momentum':
                    return 0.937
                elif name == 'weight_decay':
                    return 0.0005
                return low
                
            def suggest_int(self, name, low, high):
                if name == 'warmup_epochs':
                    return 3
                return low
        
        single_trial = SingleTrial()
        map_metric, best_model_info = objective(single_trial, args, data_dict)
        
        print("\n" + "="*80)
        print(colorstr('bold', 'green', '단일 훈련 완료!'))
        print(f"mAP50-95: {map_metric:.4f}")
        print("="*80 + "\n")

if __name__ == '__main__':
    main()