# SemanticSegmantation (code-only)

    ## 프로젝트 개요
    - DDRNet 백본을 활용해 19개 클래스를 분류하는 도로 장면 시맨틱 세그멘테이션
        파이프라인.
    - 대용량 데이터셋과 모델 가중치는 `.gitignore`로 제외하고, 학습/추론/평가에
        필요한 코드만 버전 관리.
    - `image/<split>/cam*/`와 `labelmap/<split>/cam*/` 구조를 따르는 커스텀 데이
        터셋을 사용하며, `train`, `val`, `test` 등 분할 이름은 인자로 변경 가능.

    ## 주요 스크립트와 역할
    - `train.py`: 데이터셋 로더, 클래스 비중 고려 크롭·포토메트릭 증강, AMP/
        Channels-last 최적화, 코사인 러닝레이트 스케줄을 포함한 학습 루프. 에폭
        별 로그와 체크포인트를 저장.
    - `DDRNet.py`: DDRNet-23s 기반 모델 정의. 하이·로우 해상도 브랜치를 왕복시키
        는 Bi-directional aggregation과 DAPPM(모듈) 구성 포함.
    - `prediction.py`: 학습된 가중치로 `image/<split>` 전체를 추론하고, 클래스
        ID 마스크와 컬러맵 이미지를 각각 저장.
    - `evaluation.py`: 예측 마스크(`*_CategoryId.png`)와 GT 라벨을 매칭해 클래스
        별 및 mIoU를 계산.
    - `functions.py`: 공통 유틸(체크포인트 로드, 클래스 가중치 계산, 보조 손실
        지원 CE/OHEM/Focal Loss, Cosine/Poly 러닝레이트 스케줄러, 학습/검증 증강
        등) 모음.
    - `train_sq.sh`: `torchrun`으로 `train.py`를 실행하는 쉘 스크립트. 환경변수
        로 데이터 경로·배치·에폭 등을 손쉽게 조정 가능.
    - `audit_images.py`: 데이터 품질 감사용 스크립트. 밝기/채도/블러/헤이즈 지표
        를 CSV로 산출.
    - `computation_time.py`: THOP을 이용해 FLOPs, 파라미터 수, GPU 메모리 사용
        량, 반복 추론 속도를 측정.

    ## 데이터 및 출력 규칙
    - 학습 데이터
      - 이미지: `dataset_dir/image/<split>/cam*/xxx.png`
      - 라벨: `dataset_dir/labelmap/<split>/cam*/xxx_CategoryId.png`
      - 라벨명이 `_leftImg8bit` 또는 `_gtFine` 접미사를 가지는 경우도 자동 대응.
    - 예측 결과(predict)
      - 마스크: `result_dir/label/<split>/cam*/stem_CategoryId.png`
      - 컬러맵: `result_dir/colormap/<split>/cam*/stem.jpg`
    - 증강 시각화: `--dump_aug_dir` 지정 시 학습 도중 보강된 이미지/마스크 일부
        저장.

    ## 학습 워크플로 요약
    1. `train.py` 실행 시 `SegDataset`이 호출되어 랜덤 스케일, rare-class aware
        크롭, 좌우 반전, 선택적 포토메트릭 증강을 적용.
    2. 희귀 클래스가 포함된 샘플을 추가로 학습하고 싶다면 `--weighted_sampling
        --strong_list_txt path.txt --strong_weight 3.0` 옵션 사용.
    3. AMP(`--amp`)와 Channels-last(`--channels_last`)로 학습 속도 및 VRAM 효율
        을 개선.
    4. 체크포인트는 `result_dir/epoch_###.pth`, 최적 모델은 `model_best.pth`로
        저장되며, `log.txt`에 손실과 러닝레이트 기록.

    ## 추론 및 평가
    - 추론: `prediction.py --dataset_dir ... --split val --weight_path
        preds_val_best/label/val`
    - 스크립트는 입력/출력 해상도를 자동 정렬하고, 라벨 누락이나 해상도 불일치가
        감지되면 로그로 경고.

    ## 추가 유틸 사용 예시
    - 데이터 점검: `python audit_images.py` → `runs/image_audit.csv`
    - 모델 리소스 산출: `python computation_time.py` → FLOPs/latency 출력
        과 `FLOPs.txt`
    - 클래스 가중치 산출: `functions.compute_class_weights("학습데이터/labelmap/
        train")`를 별도 노트북 등에서 활용 가능.

    필요 시 README 내용을 그대로 복사해 개인 GitHub에 게시하면, 프로젝트 구조와
        코드 용도를 한눈에 설명할 수 있습니다.
