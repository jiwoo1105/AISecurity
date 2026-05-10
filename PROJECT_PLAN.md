# Diffusion 기반 의료 AI 방어 기법 - 프로젝트 수행 계획서

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **주제** | Diffusion 기반 의료영상 적대적 공격 분석 및 의료 특화 방어 기법 연구 |
| **핵심 가설** | DiffAttack은 기존 PGD보다 자연스럽고 강력하며, 의료 특화 DiffPure가 이를 효과적으로 방어할 수 있다 |
| **데이터셋** | CheXpert (메인), NIH ChestX-ray14 (외부 검증) |
| **기간** | 총 6주 |

---

## 2. 시스템 아키텍처

```
입력 X-ray
    │
    ▼
┌─────────────────────────────┐
│  Module 1: FFT 주파수 분석   │
│  - 저/중/고주파 분리          │
│  - z-score 기반 이상 탐지     │
│  - Attack Map 생성           │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Module 2: 적응적 노이즈 정화 │
│  - Attack Map 기반 차등 노이즈 │
│  - RoentGen 기반 denoise     │
│  - 공격 영역 강화 정화        │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Module 3: 해부학적 일관성 검증│
│  - 정화 전/후 폐 분할 마스크  │
│  - Dice 계수 비교            │
│  - 임계값 미달 시 재정화      │
└──────────┬──────────────────┘
           │
           ▼
    정화된 X-ray → 분류기 입력
```

---

## 3. 주차별 상세 수행 계획

### Week 1: 환경 구축 + 베이스라인 분류기

#### 목표
- 개발 환경 세팅 및 CheXpert 데이터 전처리
- 베이스라인 분류기 학습 (AUC >= 0.85)

#### 세부 작업

| 일차 | 작업 내용 | 산출물 |
|------|----------|--------|
| Day 1-2 | GPU 환경 세팅 (CUDA, PyTorch), 프로젝트 구조 생성 | `requirements.txt`, 프로젝트 디렉토리 |
| Day 2-3 | CheXpert 다운로드, 전처리 파이프라인 구현 (224x224 resize, normalization) | `data/preprocessing.py` |
| Day 3-4 | ResNet-50 분류기 학습 (폐렴 binary classification) | `models/resnet50_baseline.pth` |
| Day 4-5 | DenseNet-121 분류기 학습 | `models/densenet121_baseline.pth` |
| Day 5 | 베이스라인 성능 평가 (AUC, Clean Accuracy) | `results/baseline_eval.csv` |

#### 프로젝트 디렉토리 구조
```
AISecurity/
├── data/
│   ├── preprocessing.py       # 데이터 전처리
│   ├── dataset.py             # PyTorch Dataset 클래스
│   └── chexpert/              # CheXpert 데이터 (gitignore)
├── models/
│   ├── classifier.py          # ResNet-50, DenseNet-121
│   └── checkpoints/           # 학습된 가중치
├── attacks/
│   ├── fgsm.py                # FGSM 공격
│   ├── pgd.py                 # PGD 공격
│   └── diff_attack.py         # DiffAttack 공격
├── defense/
│   ├── fft_analyzer.py        # FFT 주파수 분석 모듈
│   ├── vanilla_diffpure.py    # 기본 DiffPure
│   ├── medical_diffpure.py    # 의료 특화 DiffPure (핵심)
│   ├── adv_training.py        # Adversarial Training
│   ├── randomized_smoothing.py# Randomized Smoothing
│   └── anatomical_check.py    # 해부학적 일관성 검증
├── evaluation/
│   ├── metrics.py             # 평가 지표 (ASR, LPIPS, AUC, ECE 등)
│   ├── frequency_analysis.py  # FFT 시각화
│   └── eval_matrix.py         # 12개 시나리오 매트릭스 평가
├── configs/
│   └── config.yaml            # 하이퍼파라미터 설정
├── notebooks/
│   └── visualization.ipynb    # 결과 시각화
├── results/                   # 실험 결과
├── scripts/
│   ├── train_classifier.py    # 분류기 학습 스크립트
│   ├── run_attack.py          # 공격 실행 스크립트
│   └── run_defense.py         # 방어 실행 스크립트
├── PROJECT_PLAN.md
├── requirements.txt
└── README.md
```

#### 핵심 코드: 분류기 학습

```python
# models/classifier.py 핵심 구조
import torchvision.models as models

def get_classifier(name="resnet50", num_classes=2, pretrained=True):
    if name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
```

---

### Week 2: 전통 공격 구현 + FFT 분석 도구

#### 목표
- FGSM, PGD 공격 구현 및 분류기 공격 성공률 측정
- FFT 주파수 분석 도구 구축

#### 세부 작업

| 일차 | 작업 내용 | 산출물 |
|------|----------|--------|
| Day 1-2 | FGSM 공격 구현 (ε = 0.01, 0.02, 0.04, 0.08) | `attacks/fgsm.py` |
| Day 2-3 | PGD 공격 구현 (ε = 0.01~0.08, step=10, 20, 40) | `attacks/pgd.py` |
| Day 3-4 | FFT 주파수 분석 모듈 구현 | `defense/fft_analyzer.py` |
| Day 4-5 | 공격 평가 (ASR, LPIPS) + FFT 주파수 스펙트럼 시각화 | 공격 결과 리포트 |
| Day 5 | 정상 X-ray FFT 통계 (평균, 표준편차) 사전 계산 | `data/fft_statistics.pkl` |

#### 핵심 코드: FFT 주파수 분석

```python
# defense/fft_analyzer.py 핵심 로직
import torch
import torch.fft as fft

class FFTAnalyzer:
    def __init__(self, normal_stats_path):
        # 정상 X-ray의 주파수 통계 로드
        self.normal_mean, self.normal_std = load_stats(normal_stats_path)

    def compute_frequency_bands(self, image):
        """이미지를 저/중/고주파 대역으로 분리"""
        f = fft.fft2(image)
        f_shift = fft.fftshift(f)
        magnitude = torch.abs(f_shift)

        h, w = image.shape[-2:]
        center = (h // 2, w // 2)

        # 주파수 대역 마스크 생성
        low = create_band_mask(center, 0, h // 6)        # 저주파
        mid = create_band_mask(center, h // 6, h // 3)    # 중주파
        high = create_band_mask(center, h // 3, h // 2)   # 고주파

        return {
            'low': magnitude * low,
            'mid': magnitude * mid,
            'high': magnitude * high
        }

    def detect_attack_type(self, image):
        """z-score 기반 공격 유형 탐지"""
        bands = self.compute_frequency_bands(image)
        z_scores = {}

        for band_name, band_energy in bands.items():
            z = (band_energy.mean() - self.normal_mean[band_name]) / self.normal_std[band_name]
            z_scores[band_name] = z

        if z_scores['high'] > threshold_high:
            return 'pixel_attack'     # FGSM/PGD
        elif z_scores['mid'] > threshold_mid:
            return 'diffusion_attack'  # DiffAttack
        else:
            return 'clean'

    def generate_attack_map(self, image):
        """공격 영역 마스크 생성 (차등 노이즈 적용용)"""
        # 주파수 대역별 z-score → 공간 도메인 Attack Map 생성
        ...
        return attack_map  # [0, 1] 범위, 공격 의심 영역일수록 1에 가까움
```

#### FFT 통계 사전 계산 (정상 X-ray 기준선)

```python
# scripts/compute_fft_stats.py
# CheXpert 정상 라벨 subset에서 주파수 통계 추출
def compute_normal_fft_statistics(normal_dataset):
    all_bands = {'low': [], 'mid': [], 'high': []}
    analyzer = FFTAnalyzer(normal_stats_path=None)

    for image in normal_dataset:
        bands = analyzer.compute_frequency_bands(image)
        for k in all_bands:
            all_bands[k].append(bands[k].mean().item())

    stats = {}
    for k in all_bands:
        stats[k] = {
            'mean': np.mean(all_bands[k]),
            'std': np.std(all_bands[k])
        }

    torch.save(stats, 'data/fft_statistics.pkl')
```

---

### Week 3: DiffAttack + Vanilla DiffPure

#### 목표
- DiffAttack 의료영상 적용 및 "중주파 공격이 더 위협적" 가설 검증
- Vanilla DiffPure 구현 (비교 베이스라인)

#### 세부 작업

| 일차 | 작업 내용 | 산출물 |
|------|----------|--------|
| Day 1-2 | DiffAttack 구현 (Stable Diffusion latent space 기반) | `attacks/diff_attack.py` |
| Day 3 | DiffAttack vs PGD 정량 비교 (ASR, LPIPS) + FFT 시각화 | 비교 분석 리포트 |
| Day 3-4 | Vanilla DiffPure 구현 (일반 Diffusion 기반) | `defense/vanilla_diffpure.py` |
| Day 4-5 | Adversarial Training, Randomized Smoothing 베이스라인 구현 | `defense/adv_training.py`, `defense/randomized_smoothing.py` |
| Day 5 | 전통 공격 대상 방어 성능 평가 | 방어 결과 리포트 |

#### 핵심: DiffAttack vs PGD 비교 실험 설계

```
공격 비교 (Week 3):
┌────────────┬───────────┬───────────┬────────────────┐
│            │ ASR ↑     │ LPIPS ↓   │ FFT 시각화     │
├────────────┼───────────┼───────────┼────────────────┤
│ FGSM       │ ?         │ ?         │ 고주파 에너지 ↑ │
│ PGD        │ ?         │ ?         │ 고주파 에너지 ↑ │
│ DiffAttack │ ?         │ ?         │ 중주파 변형 ★   │
└────────────┴───────────┴───────────┴────────────────┘
가설: DiffAttack이 ASR은 비슷하거나 높으면서 LPIPS는 낮음 (더 자연스러움)
```

---

### Week 4: 의료 특화 DiffPure 구현 (핵심 기여)

#### 목표
- 3개 모듈 통합: FFT 주파수 분석 + RoentGen 적응적 정화 + 해부학적 일관성 검증

#### 세부 작업

| 일차 | 작업 내용 | 산출물 |
|------|----------|--------|
| Day 1 | RoentGen 모델 로드 및 의료영상 정화 파이프라인 구축 | RoentGen 연동 코드 |
| Day 2 | Attack Map 기반 차등 노이즈 주입 모듈 구현 | 적응적 노이즈 모듈 |
| Day 3 | 해부학적 일관성 검증 모듈 (폐 분할 + Dice 계수) | `defense/anatomical_check.py` |
| Day 4 | 3개 모듈 통합 → 의료 특화 DiffPure 완성 | `defense/medical_diffpure.py` |
| Day 5 | DiffAttack 대상 의료 특화 DiffPure 초기 평가 | 초기 결과 |

#### 핵심 코드: 의료 특화 DiffPure

```python
# defense/medical_diffpure.py
class MedicalDiffPure:
    def __init__(self, roentgen_model, fft_analyzer, lung_segmenter):
        self.roentgen = roentgen_model        # RoentGen (의료 특화 Diffusion)
        self.fft = fft_analyzer               # FFT 주파수 분석
        self.segmenter = lung_segmenter       # 폐 분할 모델
        self.dice_threshold = 0.95            # 해부학적 일관성 임계값

    def purify(self, x_attacked, max_retry=3):
        # Step 1: FFT 기반 공격 탐지 및 Attack Map 생성
        attack_type = self.fft.detect_attack_type(x_attacked)
        attack_map = self.fft.generate_attack_map(x_attacked)

        # Step 2: Attack Map 기반 차등 노이즈 추가
        # 공격 영역 → 강한 노이즈 / 정상 영역 → 약한 노이즈
        noise_level = self._adaptive_noise(attack_map, attack_type)
        x_noised = x_attacked + noise_level * torch.randn_like(x_attacked)

        # Step 3: RoentGen으로 denoise (의료영상 자연 분포 기반 복원)
        x_purified = self.roentgen.denoise(x_noised, noise_level)

        # Step 4: 해부학적 일관성 검증
        for attempt in range(max_retry):
            dice = self._check_anatomical_consistency(x_attacked, x_purified)
            if dice >= self.dice_threshold:
                return x_purified
            # Dice 미달 → 노이즈 레벨 낮춰서 재정화
            noise_level *= 0.8
            x_noised = x_attacked + noise_level * torch.randn_like(x_attacked)
            x_purified = self.roentgen.denoise(x_noised, noise_level)

        return x_purified

    def _adaptive_noise(self, attack_map, attack_type):
        """공격 유형과 Attack Map에 따른 차등 노이즈 레벨"""
        base = 0.1 if attack_type == 'pixel_attack' else 0.2
        return base * (0.3 + 0.7 * attack_map)  # 정상: 0.3x, 공격: 1.0x

    def _check_anatomical_consistency(self, x_original, x_purified):
        """폐 분할 마스크 비교 → Dice 계수"""
        mask_orig = self.segmenter(x_original)
        mask_puri = self.segmenter(x_purified)
        intersection = (mask_orig * mask_puri).sum()
        dice = 2 * intersection / (mask_orig.sum() + mask_puri.sum())
        return dice.item()
```

---

### Week 5: 종합 평가

#### 목표
- 12개 시나리오 매트릭스 완성
- 모든 평가 지표 측정 및 비교 분석

#### 12개 시나리오 매트릭스

```
              │ FGSM    │ PGD     │ DiffAttack │
──────────────┼─────────┼─────────┼────────────┤
Adv Training  │ S1      │ S2      │ S3         │
Rand Smoothing│ S4      │ S5      │ S6         │
Vanilla DiffP │ S7      │ S8      │ S9         │
Medical DiffP │ S10     │ S11     │ S12 ★      │
```

#### 평가 지표 (핵심 7개 + FFT 분석)

**공격 강도 평가 (2개):**
| 지표 | 의미 | 기대 결과 |
|------|------|----------|
| ASR | 공격 성공률 | DiffAttack >= PGD |
| LPIPS | 지각적 유사도 (낮을수록 자연스러움) | DiffAttack < PGD |

> FFT 주파수 스펙트럼 분석은 별도 시각화 도구로 활용 (PGD=고주파, DiffAttack=중주파 패턴 확인)

**방어 성능 평가 (3개):**
| 지표 | 의미 | 목표 |
|------|------|------|
| Clean Accuracy | 정상 이미지 정확도 (방어가 성능을 깎지 않는지) | >= 85% |
| Robust Accuracy | 공격 후 방어 정확도 (방어 효과의 핵심 지표) | Medical DiffPure가 최고 |
| AUC | 의료영상 표준 종합 분류 성능 | >= 0.80 |

**임상 신뢰도 평가 (2개):**
| 지표 | 의미 | 목표 |
|------|------|------|
| ECE | 예측 확신도와 실제 정답률의 괴리 (자신 있게 틀리는 위험) | < 0.05 |
| Anatomical SSIM | 폐 영역 해부학적 구조 보존도 (본 연구 차별점) | >= 0.90 |

> **제외 사유:** Recovery Rate(Robust Acc에서 역산 가능), SSIM(LPIPS와 역할 중복), Sensitivity/Specificity(AUC에 포함)

#### 세부 작업

| 일차 | 작업 내용 | 산출물 |
|------|----------|--------|
| Day 1-2 | 12개 시나리오 자동 실행 스크립트 작성 및 실행 | `scripts/run_all_scenarios.py` |
| Day 3 | NIH ChestX-ray14 외부 검증 | 일반화 성능 리포트 |
| Day 4 | 결과 시각화 (Robustness Curve, FFT 비교, 히트맵) | 시각화 결과 |
| Day 5 | Ablation Study (각 모듈 제거 시 성능 변화) | Ablation 결과 |

#### Ablation Study 설계

```
Full Model (FFT + RoentGen + Anatomical)  → 기준
- FFT 제거 (균일 노이즈)                   → FFT의 기여 측정
- RoentGen → 일반 Diffusion 교체           → 의료 특화의 기여 측정
- Anatomical Check 제거                    → 해부학 검증의 기여 측정
```

---

### Week 6: 버퍼 + 마무리

| 일차 | 작업 내용 |
|------|----------|
| Day 1-2 | 부족한 실험 보완, 추가 ε 값 실험 |
| Day 3 | 코드 정리, 주석, README 작성 |
| Day 4-5 | 최종 보고서 작성, 발표 자료 준비 |

---

## 4. AI 도구 활용 계획

### 4.1 사용할 AI 도구 및 역할 분담

| AI 도구 | 역할 | 활용 영역 |
|---------|------|----------|
| **Claude Code (CLI)** | 메인 코딩 파트너 | 코드 구현, 디버깅, 코드 리뷰, Git 관리 |
| **Claude (Web/Desktop)** | 연구 어드바이저 | 논문 분석, 수식 검증, 실험 설계 상담 |
| **GitHub Copilot** | 코드 자동완성 | 반복적 코드 작성 보조 |

### 4.2 Claude Code 활용 - 주요 프롬프팅 사례

#### (1) 코드 구현 요청

```
프롬프트 예시:
"CheXpert 데이터셋을 위한 PyTorch Dataset 클래스를 만들어줘.
- CSV 라벨 파일 파싱
- 폐렴 binary classification (Pneumonia vs Normal)
- 224x224 resize + ImageNet normalization
- train/val/test split 지원"
```

**활용 포인트:** 구체적인 요구사항을 명시하여 한 번에 원하는 코드를 얻음

#### (2) 디버깅

```
프롬프트 예시:
"PGD 공격을 돌렸는데 ASR이 0%가 나와.
현재 코드에서 gradient 계산 부분을 확인하고,
model.eval() 상태에서 gradient가 제대로 흐르는지 봐줘.
에러 로그: [로그 붙여넣기]"
```

**활용 포인트:** 에러 로그와 현재 상태를 함께 제공하여 정확한 원인 파악

#### (3) 성능 최적화

```
프롬프트 예시:
"DiffPure 정화가 이미지 한 장당 30초 걸려서 너무 느려.
배치 처리로 바꾸고, Diffusion step 수를 줄이면서도
정화 품질을 유지하는 방법을 제안해줘."
```

#### (4) 논문 기반 구현

```
프롬프트 예시:
"이 DiffAttack 논문의 Algorithm 1을 PyTorch로 구현해줘.
원래는 ImageNet용인데, 의료영상(흉부 X-ray, 224x224, grayscale)에
맞게 수정해야 해. 핵심 차이점은..."
```

**활용 포인트:** 논문의 알고리즘을 프로젝트 조건에 맞게 변환

#### (5) 평가 코드 작성

```
프롬프트 예시:
"12개 시나리오(3공격 x 4방어) 전체를 자동으로 돌리고,
결과를 CSV + 히트맵으로 저장하는 평가 스크립트를 만들어줘.
측정 지표: ASR, LPIPS, Clean Acc, Robust Acc, AUC, ECE, Anatomical SSIM"
```

#### (6) Git 커밋 관리

```
프롬프트 예시:
"지금까지 작업한 FFT 분석 모듈 변경사항을 커밋하고 push 해줘"
→ Claude Code가 자동으로 diff 확인 → 커밋 메시지 작성 → push
→ PR 생성 시 템플릿 자동 적용
```

### 4.3 AI 활용 워크플로우 (일일 작업 사이클)

```
1. 계획 수립
   └─ Claude에게 오늘의 작업 범위와 목표 설명
   └─ 구현 방향에 대해 상담 후 확정

2. 코드 구현
   └─ Claude Code로 핵심 로직 구현 요청
   └─ 생성된 코드 검토 → 필요 시 수정 요청
   └─ 단위 테스트 작성 요청

3. 디버깅
   └─ 에러 발생 시 에러 로그 + 코드 컨텍스트 제공
   └─ Claude가 원인 분석 → 수정 제안 → 적용

4. 실험 및 분석
   └─ 실험 결과를 Claude에게 공유
   └─ 결과 해석 및 다음 실험 설계 상담

5. 버전 관리
   └─ Claude Code로 커밋 + PR 생성
   └─ PR 템플릿에 맞춰 작업 내역 자동 정리
```

### 4.4 AI 활용 시 주의사항

| 원칙 | 설명 |
|------|------|
| **코드 이해 필수** | AI가 생성한 코드는 반드시 직접 이해한 후 사용. 블랙박스로 쓰지 않음 |
| **프롬프트 기록** | 주요 프롬프트와 결과를 Git 커밋 메시지나 PR에 기록하여 추적 가능하게 |
| **점진적 구현** | 한번에 전체 시스템을 요청하지 않고, 모듈 단위로 나눠서 구현 → 검증 반복 |
| **논문 대조** | AI가 구현한 알고리즘이 원 논문과 일치하는지 반드시 교차 검증 |

---

## 5. 커밋 & PR 전략 (버전 관리 증빙)

### 브랜치 전략

```
main ─────────────────────────────────────────
  │
  ├── feature/week1-baseline-classifier
  ├── feature/week2-fgsm-pgd-attack
  ├── feature/week2-fft-analyzer
  ├── feature/week3-diffusion-attack
  ├── feature/week3-vanilla-diffpure
  ├── feature/week4-medical-diffpure
  ├── feature/week5-evaluation-matrix
  └── feature/week6-final-report
```

### 커밋 컨벤션

```
feat: 새 기능 추가        (예: feat: PGD 공격 모듈 구현)
fix: 버그 수정            (예: fix: gradient detach 누락 수정)
exp: 실험 결과 추가        (예: exp: DiffAttack vs PGD ASR 비교)
docs: 문서 수정           (예: docs: README에 실행 방법 추가)
refactor: 코드 리팩토링    (예: refactor: FFT 분석 모듈 최적화)
chore: 기타               (예: chore: requirements.txt 업데이트)
```

### PR 작성 예시

```markdown
## 작업 내용
PGD 공격 모듈 구현 및 베이스라인 분류기 공격 성능 평가

## 변경 사항
- attacks/pgd.py: PGD 공격 클래스 구현 (L-inf norm, ε=0.01~0.08)
- evaluation/metrics.py: ASR, LPIPS 측정 함수 추가
- results/pgd_attack_results.csv: 공격 결과 기록

## 작업 유형
- [x] 새로운 기능 추가

## 테스트
- [x] ε=0.04에서 ASR 87% 확인
- [x] LPIPS 시각적 검증 완료

## 참고 사항
- AI 활용: Claude Code로 PGD 반복 로직 구현, gradient clipping 디버깅
```

---

## 6. 기대 결과 요약

| 항목 | 기대 결과 |
|------|----------|
| **공격 분석** | DiffAttack이 PGD 대비 ASR 유사/이상, LPIPS 30%+ 낮음 (더 자연스러움) |
| **방어 성능** | Medical DiffPure의 Robust Accuracy가 Vanilla DiffPure 대비 15%+ 향상 |
| **핵심 기여** | DiffAttack 대상 방어에서 Medical DiffPure만 유의미한 Robust Accuracy 달성 |
| **임상 신뢰** | ECE < 0.05, Anatomical SSIM >= 0.90 유지 |
