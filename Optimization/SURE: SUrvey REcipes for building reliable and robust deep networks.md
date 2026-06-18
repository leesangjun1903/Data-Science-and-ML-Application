# SURE: SUrvey REcipes for Building Reliable and Robust Deep Networks 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

SURE의 핵심 주장은 다음과 같습니다:

> **단일 기법만으로는 다양한 실세계 문제에서 균일하게 우수한 성능을 내기 어려우며, 모델 정규화(Regularization), 분류기(Classifier), 최적화(Optimization) 영역에 걸친 다양한 기법들의 시너지적 통합이 불확실성 추정의 신뢰성과 견고성을 실질적으로 향상시킨다.**

### 주요 기여

| 기여 | 내용 |
|---|---|
| **문제 분석** | 기존 방법들이 단일 태스크(failure prediction, OOD detection)에만 집중하며 복잡한 실세계 환경(데이터 손상, 노이즈 레이블, 클래스 불균형)에서 균일하게 우수하지 않음을 실험적으로 규명 |
| **SURE 프레임워크 제안** | 정규화(RegMixup, CRL) + 분류기(CSC) + 최적화(SAM, SWA)를 통합한 새로운 학습 레시피 제안 |
| **실세계 적용성 입증** | 태스크별 전문 설정 없이 노이즈 레이블(Animal-10N, Food-101N), 데이터 손상(CIFAR10-C), 클래스 불균형(CIFAR-LT) 등에서 SOTA급 성능 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

딥 신경망(DNN)은 **과신(Overconfidence)** 문제를 가지고 있어, 예측이 틀려도 높은 신뢰도 점수를 출력하는 경향이 있습니다. 이는 의료 진단, 자율주행, 로보틱스 등 안전 임계 분야에서 심각한 문제를 초래합니다.

기존 연구의 한계:
- **협소한 평가 범위**: failure prediction 또는 OOD detection 등 단일 태스크에만 집중
- **실세계 미검증**: 데이터 손상, 노이즈 레이블, 장기 꼬리 분포 등 복잡한 실세계 시나리오에서의 효과 미검증
- **상호 보완성 미활용**: 개별 기법들의 시너지 효과 미탐구

### 2.2 제안하는 방법 및 수식

SURE는 두 가지 핵심 전략으로 구성됩니다:

---

#### 전략 1: 어려운 샘플에 대한 엔트로피 증가 (Increasing Entropy for Hard Samples)

**총 손실 함수 (Total Loss):**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ce}} + \lambda_{\text{mix}}\mathcal{L}_{\text{mix}} + \lambda_{\text{crl}}\mathcal{L}_{\text{crl}} \tag{1}$$

여기서 $\lambda_{\text{mix}}$와 $\lambda_{\text{crl}}$은 각 손실 항의 기여도를 조절하는 하이퍼파라미터입니다.

---

**① RegMixup 정규화 ($\mathcal{L}_{\text{mix}}$)**

두 입력-타겟 쌍 $(\mathbf{x}_i, \mathbf{y}_i)$와 $(\mathbf{x}_j, \mathbf{y}_j)$에 대해 보간된 샘플 $(\tilde{\mathbf{x}}_i, \tilde{\mathbf{y}}_i)$를 생성합니다:

$$\tilde{\mathbf{x}}_i = m\mathbf{x}_i + (1-m)\mathbf{x}_j, \quad \tilde{\mathbf{y}}_i = m\mathbf{y}_i + (1-m)\mathbf{y}_j \tag{2}$$

혼합 계수 $m$은 Beta 분포를 따릅니다:

$$m \sim \text{Beta}(\beta, \beta), \quad \beta \in (0, \infty) \tag{3}$$

RegMixup 손실은 보간된 샘플에 대한 교차 엔트로피로 정의됩니다:

$$\mathcal{L}_{\text{mix}}(\tilde{\mathbf{x}}_i, \tilde{\mathbf{y}}_i) = \mathcal{L}_{\text{ce}}(\tilde{\mathbf{x}}_i, \tilde{\mathbf{y}}_i) \tag{4}$$

$\beta = 10$으로 설정하면 두 샘플이 강하게 혼합되어, 모델이 이러한 어려운 샘플에 대해 높은 엔트로피를 나타내도록 유도합니다.

---

**② 정확도 순위 손실 (Correctness Ranking Loss, $\mathcal{L}_{\text{crl}}$)**

두 입력 이미지 $\mathbf{x}_i$와 $\mathbf{x}_j$에 대해 모델의 신뢰도를 역사적 정확도 순위와 정렬시킵니다:

$$\mathcal{L}_{\text{crl}}(\mathbf{x}_i, \mathbf{x}_j) = \max(0, |c_i - c_j| - \text{sign}(c_i - c_j)(s_i - s_j)) \tag{5}$$

- $c_i, c_j$: 훈련 중 $\mathbf{x}_i, \mathbf{x}_j$의 역사적 정확 예측 비율
- $s_i, s_j$: softmax 신뢰도 점수
- 어려운 샘플(낮은 $c_i$)은 낮은 신뢰도(높은 엔트로피)를 갖도록 유도

---

**③ 코사인 유사도 분류기 (Cosine Similarity Classifier, CSC)**

마지막 선형 레이어를 코사인 분류기로 대체합니다:

$$s_i^k = \tau \cdot \cos(f_\theta(\mathbf{x}_i), w^k) = \tau \cdot \frac{f_\theta(\mathbf{x}_i)}{\|f_\theta(\mathbf{x}_i)\|_2} \cdot \frac{w^k}{\|w^k\|_2} \tag{6}$$

- $\tau$: 온도 하이퍼파라미터
- $f_\theta$: 특징 추출기 DNN
- $w^k$: $k$번째 클래스 프로토타입

CSC는 어려운 샘플을 여러 클래스 프로토타입과 등각 거리로 배치함으로써, 전통적 선형 분류기보다 높은 엔트로피를 유도합니다.

---

#### 전략 2: 평탄한 최솟값 강제 (Enforcing Flat Minima)

**① Sharpness-Aware Minimization (SAM)**

$$\min_\theta \max_{\|\epsilon\|_2 \leq \rho} \mathcal{L}_{\text{total}}(\theta + \epsilon) \tag{7}$$

- $\epsilon$: 섭동 벡터
- $\rho$: 탐색 반경 (neighborhood size)
- 손실 경관이 평탄한 영역의 파라미터를 찾아 일반화 성능을 향상

**② Stochastic Weight Averaging (SWA)**

$$\theta_{\text{SWA}} = \frac{1}{T}\sum_{t=1}^{T} \theta_t \tag{8}$$

- $\theta_t$: epoch $t$에서의 모델 가중치
- $T$: SWA가 적용되는 총 epoch 수
- 훈련 과정의 여러 가중치를 평균화하여 더 넓고 평탄한 최솟값에 수렴

### 2.3 모델 구조

SURE는 특정 아키텍처에 종속되지 않는 **학습 레시피(Recipe)**입니다.

```
입력 이미지
    ↓
[RegMixup 데이터 증강] ──────────────────┐
    ↓                                    │
Feature Extractor (fθ)                   │ RegMixup Loss
    ↓                                    │
Cosine Similarity Classifier (CSC)       │
    ↓                                    ↓
[분류 로짓]                    ┌── 총 손실 = Lce + λmix·Lmix + λcrl·Lcrl ──┐
    ↓                          │                                              │
Cross-Entropy Loss (Lce)      ← CRL (정확도 순위 기반 정규화)                │
                                └──────────────────────────────────────────┘
                                              ↓
                              [SAM + SWA 최적화 → 평탄한 최솟값]
```

**지원 백본 아키텍처:**
- ResNet-18/32, VGG16-BN, DenseNetBC, WideResNet-28, DeiT-Base

### 2.4 성능 향상

#### Failure Prediction (CIFAR-100, ResNet-18 기준)

| 방법 | Acc. ↑ | AURC ↓ | AUROC ↑ | FPR95 ↓ |
|---|---|---|---|---|
| MSP (Baseline) | 75.87 | 69.44 | 87.00 | 60.73 |
| RegMixup | 77.90 | 59.23 | 87.61 | 58.65 |
| CRL | 76.42 | 62.78 | 88.07 | 59.02 |
| FMFP | 77.82 | 55.03 | 88.59 | 59.79 |
| **SURE** | **80.49** | **45.81** | **88.73** | **58.91** |

#### 실세계 태스크 성능

| 태스크 | 데이터셋 | SURE | 이전 SOTA |
|---|---|---|---|
| 노이즈 레이블 학습 | Food-101N | **88.0%** | 86.7% (Jigsaw-ViT) |
| 노이즈 레이블 학습 | Animal-10N | **89.0%** | 88.5% (SSR+) |
| 데이터 손상 강건성 | CIFAR10-C | **89.6% AUROC** | 88.0% (FMFP) |
| 장기 꼬리 분류 | CIFAR10-LT (IF=10) | **94.96%** | 94.04% (GLMC+MaxNorm) |

### 2.5 한계점

1. **계산 비용 증가**: SAM은 각 업데이트 스텝에서 두 번의 순전파/역전파가 필요하여 훈련 시간이 약 2배 증가합니다.
2. **하이퍼파라미터 민감성**: $\lambda_{\text{mix}}$, $\lambda_{\text{crl}}$, $\tau$, $\rho$ 등 다수의 하이퍼파라미터를 검증 세트로 조율해야 합니다.
3. **대규모 데이터셋 확장성 미검증**: ImageNet 풀 스케일에서의 검증이 부족합니다.
4. **RegMixup 스케일링 실패**: 노이즈 레이블 태스크에서 RegMixup 단독 적용 시 수렴하지 않는 경우가 있음을 논문이 직접 인정합니다.
5. **DeiT-Base의 비일관성**: DeiT-Base에서 FPR95가 오히려 악화되는 경우가 관찰됩니다 (Table 1).

---

## 3. 모델의 일반화 성능 향상 가능성

SURE의 일반화 성능 향상은 여러 메커니즘을 통해 체계적으로 달성됩니다.

### 3.1 평탄한 손실 경관을 통한 일반화

SAM과 SWA의 결합이 일반화의 핵심입니다.

**SAM의 역할**: 식 (7)에서 SAM은 파라미터 공간에서 반경 $\rho$ 이내의 최악의 섭동에 대해서도 손실이 낮은 영역을 탐색합니다:

$$\min_\theta \max_{\|\epsilon\|_2 \leq \rho} \mathcal{L}_{\text{total}}(\theta + \epsilon)$$

이는 손실 경관의 **곡률(Sharpness)**을 줄여 훈련 분포 외 데이터에 대한 과적합을 방지합니다. 특히 Chen et al. (NeurIPS 2023) [8]의 이론적 분석에 따르면, SAM이 SGD보다 일반화하는 근본 원인은 암묵적 편향(implicit bias) 차이에 있습니다.

**SWA의 역할**: 식 (8)에서 SWA는 여러 체크포인트의 가중치를 평균화합니다:

$$\theta_{\text{SWA}} = \frac{1}{T}\sum_{t=1}^{T} \theta_t$$

이는 손실 경관에서 더 넓은 최솟값(wider minima)에 수렴하게 하며, Izmailov et al. (2018) [35]에 따르면 이것이 better generalization으로 이어집니다.

두 기법의 조합(FMFP 전략)은 개별 적용 대비 일관되게 높은 성능을 보입니다.

### 3.2 RegMixup을 통한 분포 외 일반화

$\beta = 10$의 높은 Beta 분포 파라미터는 두 샘플을 강하게 혼합한 가상의 보간 샘플을 생성합니다. 이 보간 샘플들은 훈련 분포의 **결정 경계 근방**에 위치하는 어려운 샘플로 간주될 수 있으며, 모델이 이러한 샘플에 높은 엔트로피(낮은 신뢰도)를 할당하도록 학습됩니다.

Pinto et al. (NeurIPS 2022) [59]의 분석에 따르면, Mixup을 정규화기로 사용하면 분류 정확도와 OOD 강건성이 동시에 향상됩니다. SURE는 이를 CRL, CSC, SAM, SWA와 통합하여 시너지를 극대화합니다.

### 3.3 코사인 유사도 분류기의 기하학적 일반화

CSC는 특징 벡터와 클래스 프로토타입 간의 **방향적 정렬(directional alignment)**에 집중합니다:

$$s_i^k = \tau \cdot \frac{f_\theta(\mathbf{x}_i)}{\|f_\theta(\mathbf{x}_i)\|_2} \cdot \frac{w^k}{\|w^k\|_2}$$

전통적 선형 분류기가 벡터 크기(magnitude)에 영향을 받는 것과 달리, CSC는 크기를 정규화함으로써:
- 어려운 샘플을 여러 클래스 프로토타입과 등각 거리에 배치 → 높은 엔트로피 유도
- Few-shot 학습 [23, 33]에서 검증된 강건한 특징 공간 구성

### 3.4 노이즈 레이블에서의 일반화

CRL (식 5)은 훈련 중 역사적 정확도 $c_i$를 신뢰도 $s_i$와 정렬시킵니다. 노이즈 레이블 환경에서:
- 실제로 어려운/노이즈가 있는 샘플은 역사적으로 낮은 $c_i$를 가짐
- 이들에게 낮은 신뢰도를 할당 → 손실 함수에서 자동적으로 낮은 가중치 부여 효과
- 결과적으로 **깨끗한 샘플 기반 학습에 집중**하여 테스트 정확도 향상

### 3.5 불확실성 기반 재가중치(Re-weighting)를 통한 일반화

장기 꼬리 분포에서 SURE는 1단계 학습 후 얻은 최대 소프트맥스 점수 $s_i$를 이용한 지수적 재가중치를 적용합니다:

$$w_i = e^{-s_i}$$

- 높은 신뢰도($s_i$ 높음) → 낮은 가중치: 이미 잘 학습된 다수 클래스 샘플
- 낮은 신뢰도($s_i$ 낮음) → 높은 가중치: 소수 클래스 또는 어려운 샘플

이 전략은 클래스 불균형 데이터에서 tail 클래스의 일반화 성능을 효과적으로 향상시킵니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### (1) "레시피" 패러다임의 확산
SURE는 단일 혁신적 기법보다 **기존 기법들의 체계적 통합**이 더 효과적일 수 있음을 보였습니다. 이는 향후 연구가 "어떤 기법을 어떻게 조합할 것인가"에 주목하게 하는 방향성을 제시합니다. NLP 분야의 "Training Tips" 연구들(예: RoBERTa)과 유사한 흐름이 CV 분야에서도 강화될 것입니다.

#### (2) 불확실성 추정의 실용화 기반 마련
SURE는 불확실성 추정이 단순한 이론적 연구를 넘어 노이즈 레이블, 장기 꼬리 분포 등 **실제 산업 문제에 직접 적용 가능**함을 보였습니다. 의료 AI, 자율주행 등 안전 임계 분야에서의 신뢰성 있는 AI 시스템 개발에 기여할 것입니다.

#### (3) 평탄한 최솟값과 불확실성 추정의 연결
SAM+SWA가 불확실성 추정에도 유효함을 보인 것은 **최적화 이론과 불확실성 이론의 교차점**에 대한 연구를 자극할 것입니다. 특히 "왜 평탄한 최솟값이 더 나은 불확실성 추정을 제공하는가"에 대한 이론적 분석이 필요합니다.

#### (4) 표준 벤치마크로의 활용 가능성
SURE가 다양한 아키텍처와 데이터셋에서 일관된 성능을 보임으로써, 향후 새로운 불확실성 추정 기법들의 **강력한 베이스라인**으로 활용될 가능성이 높습니다.

### 4.2 앞으로 연구 시 고려할 점

#### (1) 계산 효율성 개선
SAM은 그래디언트 계산을 두 번 수행해야 하므로 훈련 시간이 크게 증가합니다. 향후 연구에서는:
- Efficient SAM (ESAM), GSAM 등 SAM의 경량화 변형 적용
- 선택적 SAM 적용 (특정 레이어 또는 epoch에만 적용)

#### (2) 대규모 데이터셋 및 대형 모델로의 확장
현재 SURE는 주로 CIFAR 규모의 데이터셋에서 검증되었습니다. 향후에는:
- ImageNet-21K, LAION 등 대규모 데이터셋에서의 검증 필요
- GPT/ViT 등 대형 언어/비전 모델에서의 적용 가능성 탐구
- 파인튜닝 시나리오에서의 효율적 적용 방안 (현재 DeiT 파인튜닝 결과는 일부 지표에서 불일치)

#### (3) 하이퍼파라미터 자동 탐색
$\lambda_{\text{mix}}$, $\lambda_{\text{crl}}$, $\tau$, $\rho$ 등 다수의 하이퍼파라미터를 검증 세트로 조율해야 합니다. AutoML 또는 메타학습 기반의 자동 탐색 연구가 필요합니다.

#### (4) 이론적 기반 강화
SURE의 효과가 실험적으로 검증되었으나, 이론적 설명이 부족합니다:
- 각 기법들의 시너지 효과에 대한 수학적 분석
- 불확실성 추정 품질의 이론적 보장

#### (5) LLM/Foundation Model 시대에서의 재해석
최근 Foundation Model(CLIP, DINO, SAM 등)의 등장으로, 이러한 대형 모델을 백본으로 사용할 때의 불확실성 추정 연구가 필요합니다. SURE의 레시피가 파인튜닝 기반 시나리오에서도 유효한지 검증이 필요합니다.

#### (6) 다양한 도메인으로의 확장
현재 이미지 분류 중심의 평가를 넘어:
- 객체 탐지, 시맨틱 분할에서의 불확실성 추정
- 자연어 처리, 멀티모달 태스크로의 확장
- 시계열 및 그래프 데이터에서의 적용

---

## 5. 2020년 이후 최신 연구 비교 분석

| 논문 | 발표년도/학회 | 핵심 방법 | SURE 대비 차이점 |
|---|---|---|---|
| **RegMixup** [Pinto et al.] | NeurIPS 2022 | Mixup을 정규화기로 사용, $\beta=10$ | SURE의 구성 요소 중 하나. 단독으로는 노이즈 레이블에 수렴 실패 |
| **FMFP** [Zhu et al.] | ECCV 2022 | SAM+SWA 조합으로 평탄한 최솟값 탐색 | SURE의 최적화 전략 기반. SURE는 여기에 RegMixup+CRL+CSC 추가 |
| **OpenMix** [Zhu et al.] | CVPR 2023 | OOD 데이터를 추가로 활용한 Mixup | 추가 외부 데이터 필요. SURE는 추가 데이터 없이도 경쟁력 있는 성능 |
| **DDU** [Mukhoti et al.] | CVPR 2023 | Spectral Normalization으로 bi-Lipschitz 특성 부여 | 사전 정의된 입력 크기 필요, 확장성 부족. SURE는 이런 제약 없음 |
| **GLMC** [Du et al.] | CVPR 2023 | 전역-로컬 혼합 일관성 손실로 장기 꼬리 분류 | 태스크 특화 방법. SURE+재가중치가 경쟁력 있는 성능 |
| **BCL** [Zhu et al.] | CVPR 2022 | 균형 대조 손실로 장기 꼬리 분류 | 태스크 특화 방법. SURE가 일부 설정에서 능가 |
| **CRL** [Moon et al.] | ICML 2020 | 역사적 정확도와 신뢰도 정렬 | SURE의 구성 요소 중 하나로 통합 |
| **SAM** [Foret et al.] | ICLR 2021 | 샤프니스 인식 최소화 | SURE의 최적화 구성 요소. SWA와 결합 시 시너지 |
| **Jigsaw-ViT** [Chen et al.] | Pattern Recognition Letters 2023 | DeiT+추가 자기지도 손실로 노이즈 레이블 학습 | 추가 사전학습 데이터 필요. SURE가 Food-101N에서 88.0% vs 86.7% 능가 |

### 종합 비교

```
접근 방식 관점에서:
- 기존 연구: 단일 혁신 기법 제안 → 특정 태스크 최적화
- SURE: 기존 기법들의 시너지적 통합 → 다양한 태스크 범용 적용

일반화 관점에서:
- 태스크 특화 SOTA (GLMC, Jigsaw-ViT): 특정 태스크 최고 성능
- SURE: 태스크 특화 없이 comparable 또는 superior 성능

추가 데이터 필요성:
- OpenMix: OOD 외부 데이터 필요
- Jigsaw-ViT: 추가 사전학습 데이터 필요
- SURE: 추가 데이터 불필요 ✓
```

---

## 참고 자료

**주요 논문 (본문에서 직접 인용)**

1. **SURE 원문**: Li, Y., Chen, Y., Yu, X., Chen, D., & Shen, X. (2024). SURE: SUrvey REcipes for building reliable and robust deep networks. *CVPR 2024*. [Open Access: https://yutingli0606.github.io/SURE/]

2. **RegMixup**: Pinto, F., Yang, H., Lim, S.N., Torr, P., & Dokania, P. (2022). Using mixup as a regularizer can surprisingly improve accuracy & out-of-distribution robustness. *NeurIPS 2022*.

3. **FMFP**: Zhu, F., Cheng, Z., Zhang, X.-Y., & Liu, C.-L. (2022). Rethinking confidence calibration for failure prediction. *ECCV 2022*.

4. **SAM**: Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). Sharpness-aware minimization for efficiently improving generalization. *ICLR 2021*.

5. **SWA**: Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A.G. (2018). Averaging weights leads to wider optima and better generalization. *arXiv 2018*.

6. **CRL**: Moon, J., Kim, J., Shin, Y., & Hwang, S. (2020). Confidence-aware learning for deep neural networks. *ICML 2020*.

7. **MSP (Baseline)**: Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. *ICLR 2017*.

8. **OpenMix**: Zhu, F., Cheng, Z., Zhang, X.-Y., & Liu, C.-L. (2023). Openmix: Exploring outlier samples for misclassification detection. *CVPR 2023*.

9. **DDU**: Mukhoti, J., Kirsch, A., van Amersfoort, J., Torr, P.H.S., & Gal, Y. (2023). Deep deterministic uncertainty: A new simple baseline. *CVPR 2023*.

10. **Jigsaw-ViT**: Chen, Y., Shen, X., Liu, Y., Tao, Q., & Suykens, J.A.K. (2023). Jigsaw-ViT: Learning jigsaw puzzles in vision transformer. *Pattern Recognition Letters*.

11. **GLMC**: Du, F., Yang, P., Jia, Q., Nan, F., Chen, X., & Yang, Y. (2023). Global and local mixture consistency cumulative learning for long-tailed visual recognitions. *CVPR 2023*.

12. **SAM 이론 분석**: Chen, Z., Zhang, J., Kou, Y., Chen, X., Hsieh, C.-J., & Gu, Q. (2023). Why does sharpness-aware minimization generalize better than SGD? *NeurIPS 2023*.
