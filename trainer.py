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
    parser.add_argument('--name', default='YOLOv11-PascalVOC', help='실험 이름')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='가장 최근 학습을 이어서 진행')
    
    parser.add_argument('--lr', type=float, default=0.01, help='학습률')
    
    parser.add_argument('--n_trials', type=int, default=5, help='Optuna 실험 횟수')
    parser.add_argument('--optimize', action='store_true', help='Optuna를 사용한 하이퍼파라미터 최적화 진행')
    
    parser.add_argument('--search_method', type=str, default='tpe', choices=['tpe', 'grid', 'random'], help='Optuna 하이퍼파라미터 탐색 방법 (tpe, grid, random)')
    
    return parser.parse_args()

def disable_yolo_mlflow():
    """YOLO의 MLflow 기능 비활성화 시도 (설정 파일 직접 수정)"""
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

def objective(trial, args, data_dict):
    """Optuna 최적화를 위한 목적 함수"""
    
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
    model = YOLO('yolo11n.pt')  # 상대 경로 사용
    
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
            
            # 최적화 목표 반환 (mAP50-95 사용)
            map_metric = metrics['mAP50-95']
            print(f"Trial {trial.number} 완료: mAP50-95 = {map_metric:.4f}")
            
        except Exception as e:
            print(f"Trial {trial.number} 중 오류 발생: {e}")
            # 실패한 경우 낮은 점수 반환
    
    return map_metric

def run_optimization(args):
    """Optuna를 사용한 하이퍼파라미터 최적화 실행"""
    # MLflow 설정 - 독립적인 디렉토리 사용
    mlflow.set_tracking_uri(f"file://{os.path.abspath('optuna_mlruns')}")
    mlflow.set_experiment(f"{args.name}")
    
    # 디렉토리 생성
    os.makedirs("optuna_mlruns", exist_ok=True)
    
    # 내장 VOC 데이터셋을 사용하므로 YAML 파일 생성 부분은 제외
    
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
    
    # Optuna 최적화 시작
    print(colorstr('bold', 'green', f'Optuna 최적화 시작 - {args.n_trials} 실험 예정...'))
    study = optuna.create_study(
        direction="maximize",
        study_name=f"YOLOv11-Optimization-{args.name}-{args.search_method}",
        sampler=sampler
    )
    
    # 저장된 최적 결과를 담을 변수
    best_params = None
    best_value = 0.0
    
    try:
        study.optimize(
            lambda trial: objective(trial, args, data_dict),
            n_trials=args.n_trials,
            catch=(Exception,)
        )
        
        # 최적의 하이퍼파라미터와 결과 출력
        print("\n" + "="*80)
        print(colorstr('bold', 'green', '최적화 완료!'))
        print(f"최적 파라미터: {study.best_params}")
        print(f"최적 mAP50-95: {study.best_value:.4f}")
        print(f"탐색 방법: {args.search_method}")
        print("="*80 + "\n")
        
        best_params = study.best_params
        best_value = study.best_value
        
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
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        print(f"최적화 결과가 저장되었습니다: {result_file}")
        
    except Exception as e:
        print(f"최적화 중 오류 발생: {e}")
    
    # 최적의 모델로 최종 학습 및 평가 (최적값이 있는 경우)
    if best_params:
        run_final_training(args, best_params)
    else:
        print("최적화에 실패했습니다. 기본 파라미터로 학습을 진행합니다.")
        run_final_training(args)

def run_final_training(args, best_params=None):
    """최적 파라미터 또는 기본 파라미터로 최종 학습 및 평가 실행"""
    # MLflow 설정 - 독립적인 디렉토리 사용
    mlflow.set_tracking_uri(f"file://{os.path.abspath('final_mlruns')}")
    mlflow.set_experiment(f"YOLOv11-Final-{args.name}")
    
    # 디렉토리 생성
    os.makedirs("final_mlruns", exist_ok=True)
    
    # MLflow 실험 시작
    run_name = f"final_{args.search_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 내장 VOC 데이터셋을 사용하므로 YAML 파일 생성 부분은 제외
    
    # 결과 저장 경로 생성
    save_dir = Path(args.project) / f"{args.name}_{args.search_method}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # YOLOv11 nano 모델 로드
    model = YOLO('yolo11n.pt')  # 상대 경로 사용
    
    # 데이터셋 설정
    data_dict = check_det_dataset(args.data)
    print(f"데이터셋 정보: {data_dict}")
    
    # 학습 설정 - 최적 파라미터 적용
    if best_params:
        print(colorstr('bold', 'green', f'최적 하이퍼파라미터로 최종 학습 시작... (탐색 방법: {args.search_method})'))
        train_args = {
            'data': args.data,
            'imgsz': args.img_size,
            'epochs': args.epochs,
            'batch': best_params.get('batch_size', args.batch_size),
            'workers': args.workers,
            'device': args.device,
            'project': args.project,
            'name': f"{args.name}_{args.search_method}",
            'resume': args.resume,
            'lr0': best_params.get('lr', args.lr),
            'lrf': 0.01,
            'momentum': best_params.get('momentum', 0.937),
            'weight_decay': best_params.get('weight_decay', 0.0005),
            'warmup_epochs': best_params.get('warmup_epochs', 3.0),
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'optimizer': 'AdamW',
            'patience': 50,
            'save': True,
            'save_period': -1,
            'plots': True,
            'rect': False,
            'cos_lr': True,
        }
    else:
        print(colorstr('bold', 'green', f'기본 파라미터로 학습 시작... (탐색 방법: {args.search_method})'))
        train_args = {
            'data': args.data,
            'imgsz': args.img_size,
            'epochs': args.epochs,
            'batch': args.batch_size,
            'workers': args.workers,
            'device': args.device,
            'project': args.project,
            'name': f"{args.name}_{args.search_method}",
            'resume': args.resume,
            'lr0': args.lr,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'optimizer': 'AdamW',
            'patience': 50,
            'save': True,
            'save_period': -1,
            'plots': True,
            'rect': False,
            'cos_lr': True,
        }
    
    # 학습 시작 시간
    start_time = time.time()
    
    # with 문으로 MLflow 실험 세션 관리
    with mlflow.start_run(run_name=run_name) as run:
        try:
            # MLflow에 태그 설정 - 검색 방법론 및 실험 유형 추적을 위해
            mlflow.set_tag("search_method", args.search_method)
            mlflow.set_tag("run_type", "final_training")
            mlflow.set_tag("optimized", "True" if best_params else "False")
            
            # MLflow에 파라미터 기록
            for key, value in train_args.items():
                mlflow.log_param(key, value)
                
            # 학습 실행
            results = model.train(**train_args)
            
            # 학습 후 결과를 수집하기 위해 모델 상태 확인 (필요한 경우)
            if hasattr(results, 'results'):
                for epoch, result in enumerate(results.results):
                    if isinstance(result, dict):
                        for k, v in result.items():
                            if isinstance(v, (int, float)):
                                mlflow.log_metric(f"epoch_{k}", v, step=epoch)
                                
            # 검증 실행
            print(colorstr('bold', 'green', '검증 시작...'))
            val_results = model.val(data=args.data, batch=train_args['batch'], imgsz=args.img_size)
            print(f"검증 결과: {val_results}")
            
            # 주요 메트릭 추출 및 MLflow에 기록
            metrics = {
                'mAP50': float(val_results.box.map50),
                'mAP50-95': float(val_results.box.map),
                'precision': float(val_results.box.mp),
                'recall': float(val_results.box.mr),
                'fitness': float(val_results.fitness),
                'training_time': time.time() - start_time
            }
            
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # 테스트 데이터로 추론 수행
            print(colorstr('bold', 'green', '테스트 시작...'))
            test_results = model.predict(
                source=data_dict['test'],
                save=True,
                save_txt=True,
                save_conf=True,
                project=args.project,
                name=f"{args.name}_{args.search_method}_test"
            )
            print(f"테스트 완료: 결과가 {os.path.join(args.project, f'{args.name}_{args.search_method}_test')}에 저장되었습니다.")
            
            # 모델 내보내기
            print(colorstr('bold', 'green', '모델 내보내기...'))
            export_formats = ['onnx', 'torchscript']
            for export_format in export_formats:
                try:
                    export_path = model.export(format=export_format, imgsz=args.img_size)
                    mlflow.log_artifact(str(export_path))
                except Exception as e:
                    print(f"모델 내보내기 실패 ({export_format}): {e}")
            
            # 최종 모델 경로
            best_model_path = save_dir / 'weights' / 'best.pt'
            if best_model_path.exists():
                # 기존 log_artifact 방식
                mlflow.log_artifact(str(best_model_path))
                
                # 새로운 코드: MLflow에 모델 등록 및 저장
                print(colorstr('bold', 'green', 'MLflow에 최종 모델 저장 중...'))
                
                # 모델 메타데이터 설정
                model_info = {
                    'model_type': 'YOLOv11-nano',
                    'dataset': args.data,
                    'img_size': args.img_size,
                    'search_method': args.search_method,
                    'best_mAP50': metrics['mAP50'],
                    'best_mAP50-95': metrics['mAP50-95'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'training_time': metrics['training_time'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # 하이퍼파라미터 정보 추가
                if best_params:
                    model_info.update({
                        'optimized': True,
                        'best_lr': best_params.get('lr'),
                        'best_batch_size': best_params.get('batch_size'),
                        'best_momentum': best_params.get('momentum'),
                        'best_weight_decay': best_params.get('weight_decay'),
                        'best_warmup_epochs': best_params.get('warmup_epochs')
                    })
                    
                # MLflow에 피클 모델 저장 (파이토치 모델 형식으로)
                mlflow.pytorch.log_model(
                    pytorch_model=model.model,  # YOLOv11 내부 PyTorch 모델
                    artifact_path="yolov11_model",
                    registered_model_name=f"YOLOv11_{args.name}_{args.search_method}",
                    metadata=model_info
                )
                
                # YOLO 모델 형식 그대로 저장
                yolo_model_path = os.path.join(mlflow.get_artifact_uri(), "yolo_model")
                os.makedirs(os.path.dirname(yolo_model_path), exist_ok=True)
                model_save_path = os.path.join(os.path.dirname(yolo_model_path), "best.pt")
                model.save(model_save_path)
                mlflow.log_artifact(model_save_path, "yolo_model")
                
                print(f"최종 모델이 MLflow 모델 레지스트리에 등록되었습니다: YOLOv11_{args.name}_{args.search_method}")
        
        except Exception as e:
            print(f"학습 중 오류 발생: {e}")
    
    print(colorstr('bold', 'green', '모든 과정이 완료되었습니다!'))

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
    os.makedirs("final_mlruns", exist_ok=True)
    
    # 내장 VOC 데이터셋을 사용한다고 안내
    print(f"Ultralytics 내장 Pascal VOC 데이터셋을 사용합니다: '{args.data}'")
    print(f"하이퍼파라미터 탐색 방법: {args.search_method}")
    
    if args.optimize:
        # Optuna를 사용한 하이퍼파라미터 최적화 실행
        run_optimization(args)
    else:
        # 최적화 없이 기본 파라미터로 훈련 실행
        run_final_training(args)

if __name__ == '__main__':
    main()