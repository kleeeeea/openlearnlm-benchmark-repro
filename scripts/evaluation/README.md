# Evaluation Scripts

벤치마크 모델 평가 스크립트

## 빠른 시작

```bash
# 환경 설정
source .venv/bin/activate

# .env 파일에 API 키 설정
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...

# 평가 실행
python run_evaluation.py --category 01_기능_skills

# 통합 리포트 생성
python generate_integrated_report.py
```

## CLI 옵션

| 옵션 | 설명 |
|------|------|
| `--category` | 평가 카테고리 (01, 02, 03, 04) |
| `--pilot` | 파일럿 테스트 모드 |
| `--limit N` | 문항 수 제한 |
| `--resume` | 이전 진행 복구 (기본값: True) |
| `--report-only` | 리포트만 생성 |
| `--models` | 특정 모델만 평가 |

## 파일 구조

```
evaluation/
├── run_evaluation.py           # 메인 실행
├── generate_integrated_report.py
├── config.py                   # 설정
├── api/                        # API 클라이언트
├── engine/                     # 평가 엔진
└── results/                    # 결과 처리
```

## 상세 가이드

자세한 내용은 [평가 실행 가이드](../../../docs/EVALUATION_GUIDE.md) 참조
