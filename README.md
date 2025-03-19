# 전기차 번호판 판별 시스템

이 프로젝트는 차량 번호판 이미지를 입력받아 전기차 여부를 판별하는 시스템입니다.

## 주요 기능

- HSV 히스토그램 기반 특징 추출
- XGBoost와 LightGBM 앙상블 모델을 사용한 예측
- 실시간 이미지 처리 및 결과 저장

## 시스템 요구사항

- Python 3.8 이상
- OpenCV
- XGBoost
- LightGBM
- NumPy
- Pandas
- scikit-learn

## 설치 방법

1. 저장소 클론
```bash
git clone [repository-url]
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

3. 모델 파일 다운로드
- `xgb_model.pkl`과 `lgbm_model.pkl` 파일을 프로젝트 루트 디렉토리에 위치시킵니다.

## 사용 방법

1. 이미지와 JSON 파일 준비
   - 이미지 파일: 차량 번호판이 포함된 이미지
   - JSON 파일: 번호판 위치 정보가 포함된 JSON

2. 예측 실행
```bash
python test_ev_2.py
```

## 프로젝트 구조

- `ev_classifier_2.py`: 이미지 전처리 및 특징 추출
- `ev_predict_2.py`: 예측 모델 클래스
- `test_ev_2.py`: 메인 실행 스크립트

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 