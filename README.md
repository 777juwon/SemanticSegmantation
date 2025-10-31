# Semantic Segmentation with DDRNet-23s

도로 장면 시맨틱 세그멘테이션 파이프라인. 대용량 데이터셋과 모델 가중치는 `.gitignore`로 제외하고, 학습·추론·평가에 필요한 **코드만** 버전 관리합니다.

## 프로젝트 개요
- DDRNet 백본으로 **19개 클래스** 분류.
- `image/<split>/cam*/`, `labelmap/<split>/cam*/` 구조의 커스텀 데이터셋 사용.
- 분할 이름(`train`, `val`, `test`)은 설정/인자로 변경 가능.
- 결과물, 데이터, 가중치 등 대용량 파일은 저장소에서 제외.

## 주요 스크립트와 역할
- **`DDRNet.py`**: DDRNet-23s 모델 정의. High/Low 해상도 브랜치의 양방향 결합과 DAPPM 모듈 구성.
- **`train.py`**: 데이터 로더, rare-class aware 크롭 및 포토메트릭 증강, AMP/Channels-last 최적화, 코사인 스케줄 기반 학습 루프, 에폭별 로그/체크포인트 저장.
- **`prediction.py`**: 학습된 가중치로 데이터셋 분할 전체 추론. 클래스 ID 마스크와 컬러맵 이미지 출력.
- **`evaluation.py`**: 예측 마스크와 GT 라벨 매칭 후 클래스별 IoU 및 mIoU 계산.
- **`functions.py`**: 공통 유틸(체크포인트 로드, 클래스 가중치 계산, CE/OHEM/Focal Loss, Cosine/Poly 스케줄러, 학습/검증 증강 등).
- **`train_sq.sh`**: 분산 실행/환경변수 기반 학습 설정 스크립트.
- **`audit_images.py`**: 데이터 품질 감사(밝기/채도/블러/헤이즈 지표) CSV 산출.
- **`computation_time.py`**: THOP 기반 FLOPs/파라미터 수/GPU 메모리/반복 추론 속도 산출.

## 데이터 및 출력 규칙
- **학습 데이터**
  - 이미지: `dataset_dir/image/<split>/cam*/xxx.png`
  - 라벨: `dataset_dir/labelmap/<split>/cam*/xxx_CategoryId.png`
  - `_leftImg8bit`, `_gtFine` 접미사도 자동 대응
- **예측 결과**
  - 마스크: `result_dir/label/<split>/cam*/stem_CategoryId.png`
  - 컬러맵: `result_dir/colormap/<split>/cam*/stem.jpg`
- **증강 시각화**
  - `--dump_aug_dir` 지정 시 학습 중 일부 보강 샘플 저장

## 학습 워크플로(요약)
1. `SegDataset`: 랜덤 스케일, rare-class aware 크롭, 좌우 반전, 선택적 포토메트릭 증강.
2. 희귀 클래스 샘플 가중 옵션(`weighted_sampling`, `strong_list_txt`, `strong_weight`) 지원.
3. AMP/Channels-last로 학습 속도 및 VRAM 효율 개선.
4. 체크포인트(`epoch_###.pth`), 베스트 모델(`model_best.pth`), 로그(`log.txt`) 관리.

## 주의 사항
- `.gitignore`로 이미지/데이터/가중치(예: `*.png`, `*.jpg`, `data/`, `dataset/`, `weights/`, `*.pt`, `*.pth`)를 제외합니다.
- 분할명과 카메라 폴더(`cam*`)는 환경에 맞게 조정 가능합니다.

## 라이선스 / 출처
- 코드 라이선스: (예: MIT/Apache-2.0)
- 외부 코드/모델 사용 시 출처 및 라이선스 표기
