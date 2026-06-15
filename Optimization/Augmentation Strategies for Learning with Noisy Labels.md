# Augmentation Strategies for Learning with Noisy Labels

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Nishi et al., 2021)의 핵심 주장은 다음과 같습니다:

> **손실 분석(loss modeling)에 사용하는 증강(augmentation)과 역전파(backpropagation)에 사용하는 증강을 분리함으로써, 노이즈 레이블 학습(Learning with Noisy Labels, LNL) 성능을 크게 향상시킬 수 있다.**

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **AUGDESC 전략 제안** | 약한 증강(weak)으로 손실 분석, 강한 증강(strong)으로 역전파 |
| **워밍업 기간 분석** | 고노이즈 환경에서 강한 증강의 워밍업 적용이 역효과임을 실증 |
| **일반화 실험** | 하이퍼파라미터 튜닝 없이 기존 기법들에 적용하여 최대 5% 향상 |
| **SOTA 달성** | CIFAR-10 90% 대칭 노이즈에서 15% 이상 절대 정확도 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

실세계 데이터셋에서는 레이블 오염(noisy labels)이 빈번하게 발생합니다. 기존 LNL 알고리즘들은 주로:

1. **워밍업 기반 샘플 필터링**: 손실값 기준으로 깨끗한 샘플 선별
2. **의사 레이블(pseudo-label)**: 네트워크 출력을 다음 손실 계산에 활용

이 두 가지 기법을 활용해왔습니다. 그러나 **강한 증강 정책(AutoAugment, RandAugment 등)을 LNL에 어떻게 전략적으로 적용할 것인가**는 미탐구 영역이었습니다.

**핵심 딜레마**: 강한 증강은 일반화에 유리하지만, 무분별하게 적용하면 **네트워크 기억화 효과(memorization effect)**를 방해하여 노이즈/클린 샘플 구분이 어려워집니다.

---

### 2.2 제안하는 방법 및 수식

#### 기반 이론: 네트워크 기억화 효과

훈련 데이터 $D = (x_i, y_i)_{i=1}^{N}$에 대한 기본 크로스 엔트로피 손실:

$$l(\theta) = -\sum_{x,y \in D} y^T \log(h_\theta(x))$$

여기서 $h_\theta$는 신경망이 근사하는 함수입니다.

Arpit et al. (2017)의 발견에 따르면, **올바르게 레이블된 데이터가 잘못 레이블된 데이터보다 먼저 수렴**합니다.

#### Co-teaching 스타일 손실 (샘플 선별 방식)

깨끗한 집합 $C$와 노이즈 집합 $I$를 다음과 같이 정의:

$$C = \arg\min_{D: |D| \geq R(T)|D|} l(f, D)$$

$$I = D \setminus C$$

이를 이용한 손실:

$$l(\theta) = -\sum_{x,y \in C} y^T \log(h_\theta(x)) - 0 \cdot \sum_{x,y \in I} y^T \log(h_\theta(x))$$

여기서 노이즈 샘플에는 가중치 $0$을 부여하여 학습에서 제외합니다.

#### Arazo et al. 스타일 손실 (소프트 레이블 방식)

베타 혼합 모델(BMM)로 학습한 가중치 $W$를 이용:

$$l(\theta) = -\sum_{x,y \in D, w \in W} (1-w) y^T \log(h_\theta(x)) - \sum_{x \in D, w \in W} w z^T \log(h_\theta(x))$$

여기서 $z$는 입력 $x$에 대한 모델의 예측값, $w$는 샘플이 노이즈일 확률입니다.

#### DivideMix의 레이블 정제 과정

$$\bar{y}_b = w_b y_b + (1 - w_b) p_b$$

$$\hat{y}_b = \text{Sharpen}(\bar{y}_b, T)$$

$$\bar{q}_b = \frac{1}{2M} \sum_m \left( p_{\text{model}}(\hat{u}_{b,m}; \theta^{(1)}) + p_{\text{model}}(\hat{u}_{b,m}; \theta^{(2)}) \right)$$

$$\hat{q}_b = \text{Sharpen}(\bar{q}_b, T)$$

전체 손실:

$$\mathcal{L} = \mathcal{L}_X + \lambda_u \mathcal{L}_U + \lambda_r \mathcal{L}_{\text{reg}}$$

---

### 2.3 AUGDESC 전략

핵심 아이디어는 **두 종류의 증강을 역할에 따라 분리**하는 것입니다:

```
입력 이미지 x
    ├── Augment₁ (약한 증강) → 손실 분석 / 의사 레이블 생성
    └── Augment₂ (강한 증강) → 역전파 / 가중치 업데이트
```

#### 세 가지 AUGDESC 변형

| 변형 | 손실 분석 증강 | 역전파 증강 |
|------|--------------|------------|
| AUGDESC-WW | 약한(Weak) | 약한(Weak) |
| AUGDESC-SS | 강한(Strong) | 강한(Strong) |
| **AUGDESC-WS** | **약한(Weak)** | **강한(Strong)** |

실험 결과, **AUGDESC-WS**가 가장 우수한 성능을 보였습니다.

#### 강한 증강 파이프라인

$$x_{\text{strong}} = \text{Normalize}(\text{AutoAugment/RandAugment}(\text{RandomCrop}(\text{RandomFlip}(x))))$$

---

### 2.4 모델 구조

| 구성 요소 | 세부 사항 |
|-----------|----------|
| **백본 네트워크** | 18-layer PreAct ResNet |
| **옵티마이저** | SGD (momentum=0.9, weight decay=0.0005) |
| **배치 크기** | 128 |
| **학습률** | 초기 0.02, 약 150 에폭 후 1/10 감소 |
| **워밍업 기간** | CIFAR-10: 10 에폭, CIFAR-100: 30 에폭 |
| **증강 수 M** | 2 (고정) |
| **기반 알고리즘** | DivideMix (주 실험), Co-Teaching+, M-DYR-H |
| **강한 증강** | AutoAugment (주 실험), RandAugment (검증) |
| **Clothing1M** | ResNet-50 + ImageNet 사전학습 가중치 |

---

### 2.5 성능 향상

#### CIFAR-10 결과 (Table 3 기반)

| 방법 | 20% 노이즈 | 50% 노이즈 | 80% 노이즈 | 90% 노이즈 |
|------|-----------|-----------|-----------|-----------|
| DivideMix (baseline) | 96.1 | 94.6 | 92.9 | 76.0 |
| **DM-AugDesc-WS-WAW** | **96.3** | **95.4** | **93.8** | **91.9** |

- CIFAR-10 90% 대칭 노이즈: **76.0 → 91.9** (+15.9%p)
- 오류율 기준 약 **65% 감소**

#### Clothing1M 결과 (Table 4 기반)

| 방법 | 정확도 |
|------|--------|
| DivideMix | 74.76 |
| ELR+ | 74.81 |
| **DM-AugDesc-WS-SAW (ours)** | **75.11** |

#### 기존 기법 일반화 결과 (Table 6, 20% 노이즈)

| 방법 | CIFAR-10 Base | CIFAR-10 Aug | CIFAR-100 Base | CIFAR-100 Aug |
|------|--------------|-------------|---------------|--------------|
| Cross Entropy | 86.8 | **89.9** | 60.2 | **61.2** |
| M-DYR-H | 94.0 | **93.9** | 68.2 | **73.0** |
| DivideMix | 96.1 | **96.3** | 77.3 | **79.5** |

---

### 2.6 한계

1. **계산 비용 증가**: AUGDESC는 각 입력에 대해 두 번의 순전파가 필요하여 학습 시간이 증가합니다.
2. **노이즈 수준 의존성**: 워밍업 전략(강한 vs. 약한)이 노이즈 수준에 민감하게 반응하며, 사전에 노이즈 수준을 알아야 최적 전략을 선택할 수 있습니다.
3. **증강 정책 선택**: AutoAugment가 주로 사용되었으나, RandAugment의 하이퍼파라미터($N$, $M$) 튜닝이 필요하며 데이터셋/네트워크에 의존적입니다.
4. **CIFAR-100 고노이즈 한계**: 90% 노이즈 CIFAR-100에서는 여전히 41.2%로 낮은 성능을 보여, 복잡한 클래스 구조에서는 어려움이 존재합니다.
5. **비대칭 노이즈(asymmetric noise) 검증 부족**: 비대칭 40% 노이즈에 대한 실험은 제한적입니다.
6. **도메인 특화 증강 미탐구**: Clothing1M에 특화된 증강 정책 탐색이 이루어지지 않았습니다.

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 일반화 향상의 메커니즘

AUGDESC-WS의 일반화 향상은 다음 메커니즘을 통해 이루어집니다:

**① 표현 학습 강화 (Representation Learning Enhancement)**

강한 증강(AutoAugment/RandAugment)이 역전파에 사용되므로, 네트워크는 다양한 변형(rotate, invert, shear 등)에 불변한(invariant) 특징 표현을 학습합니다:

$$\theta^* = \arg\min_\theta \mathbb{E}_{x \sim D}\left[ l\left(\theta, \text{StrongAug}(x)\right) \right]$$

**② 손실 분포 보호 (Loss Distribution Preservation)**

약한 증강을 손실 분석에 사용함으로써, 클린/노이즈 샘플의 손실 분포 분리가 유지됩니다. 강한 증강을 손실 분석에 사용할 경우:

$$P(\text{loss}_{\text{clean}} > \text{loss}_{\text{noisy}}) \uparrow \quad \text{(바람직하지 않음)}$$

Figure 2가 이를 시각적으로 보여줍니다: 강한 증강 비율이 높아질수록 클린 샘플의 손실이 증가하고 노이즈 샘플의 손실이 감소하여, 양자 간 구분이 어려워집니다.

**③ 일관성 정규화 효과 (Consistency Regularization Effect)**

FixMatch(Sohn et al., 2020)의 아이디어와 유사하게, 같은 이미지에 대한 두 가지 증강 버전 간의 예측 일관성을 간접적으로 유도합니다:

$$\mathcal{L}_{\text{consistency}} \propto \left\| p_\theta(\text{WeakAug}(x)) - p_\theta(\text{StrongAug}(x)) \right\|_2^2$$

**④ 워밍업 전략의 적응적 일반화**

- **저노이즈 환경**: 강한 증강 워밍업(SAW) → 더 풍부한 특징 학습 가능
- **고노이즈 환경**: 약한 증강 워밍업(WAW) → 기억화 효과 보존, 노이즈 구분 정확도 유지

이는 다음과 같이 수식화할 수 있습니다:

$$\text{WarmUp Strategy} = \begin{cases} \text{SAW} & \text{if } \eta_{\text{noise}} < \eta_{\text{threshold}} \\ \text{WAW} & \text{if } \eta_{\text{noise}} \geq \eta_{\text{threshold}} \end{cases}$$

### 3.2 하이퍼파라미터 무관 일반화

논문에서 기존 기법들(Co-Teaching+, M-DYR-H, DivideMix)에 **하이퍼파라미터 변경 없이** AUGDESC를 적용하여 성능 향상을 달성했다는 점은 중요합니다. 이는 해당 전략이 알고리즘 구조에 독립적인 일반적 원리임을 시사합니다.

### 3.3 증강 정책 비교

RandAugment가 AutoAugment와 유사한 성능을 보임으로써, 증강 전략의 핵심이 **정책 자체가 아닌 약강 분리 원칙**에 있음을 확인:

| 정책 | CIFAR-10 90% (Best) | CIFAR-100 90% (Best) |
|------|--------------------|--------------------|
| AutoAugment | 91.9 | 41.2 |
| RandAugment | 89.6 | 36.8 |
| Baseline | 76.0 | 31.5 |

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

**① LNL 알고리즘 설계 패러다임 전환**

본 논문은 LNL 연구에서 증강 전략을 단순한 데이터 확장 도구가 아닌, **알고리즘적 구성 요소**로 바라보는 시각을 제시했습니다. 이후 연구들이 증강의 역할을 명시적으로 설계하는 방향으로 발전하는 데 기여했습니다.

**② 반지도 학습(SSL)과 LNL의 가교**

FixMatch의 아이디어를 LNL에 도입함으로써, SSL 기법들이 LNL에 광범위하게 적용될 수 있는 가능성을 열었습니다.

**③ 노이즈 진단 도구**

강한 워밍업이 유리한지 약한 워밍업이 유리한지를 통해 **노이즈 수준을 간접적으로 추정**할 수 있는 가능성을 제시했습니다 (Clothing1M의 경우 ~61.54% 노이즈를 이렇게 추정).

**④ 플러그인 가능한 모듈로서의 증강**

하이퍼파라미터 변경 없이 기존 기법에 적용 가능하다는 점에서, AUGDESC는 **플러그인 모듈**로서 미래 LNL 알고리즘 개발에 기본 구성 요소가 될 수 있습니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 연구들에 대한 비교는 논문에서 직접 인용된 것과 제 학습 데이터 기준(2021년 초)의 정보를 기반으로 합니다. 2021년 이후 발표된 논문들의 정확한 수치는 확인이 불가능하므로, 논문 내 직접 인용된 연구만 정확하게 기술하겠습니다.

#### 논문 내 비교된 2020년 이후 연구

**DivideMix (Li et al., 2020)**
- arXiv:2002.07394
- GMM 기반 샘플 분리 + MixMatch 반지도 학습 결합
- CIFAR-10 90% 노이즈: 76.0% → 본 논문으로 91.9%로 향상

**ELR+ (Liu et al., 2020)**
- "Early-learning regularization prevents memorization of noisy labels" (arXiv:2007.00151)
- Clothing1M: 74.81% → 본 논문(DM-AugDesc-WS-SAW): 75.11%

**ReMixMatch (Berthelot et al., 2020)**
- 강한/약한 증강 분리 개념을 SSL에서 사용
- 본 논문은 이를 LNL 도메인으로 확장

**FixMatch (Sohn et al., 2020)**
- arXiv:2001.07685
- SSL에서 약한/강한 증강 + 의사 레이블 결합
- 본 논문의 AUGDESC와 방향은 유사하나, LNL의 손실 모델링 보호라는 차별점 존재

#### 연구 방향 비교 요약

| 연구 | 핵심 아이디어 | 본 논문과의 관계 |
|------|-------------|----------------|
| DivideMix (2020) | GMM + MixMatch | 본 논문의 베이스라인 |
| ELR+ (2020) | 조기 학습 정규화 | 비교 대상 |
| FixMatch (2020) | SSL의 강/약 증강 분리 | 영감의 출처, LNL로 확장 |
| ReMixMatch (2020) | 분포 매칭 + 증강 앵커링 | 관련 SSL 방법론 |

---

### 4.3 향후 연구 시 고려할 점

#### ① 노이즈 수준 적응적 증강 스케줄링

현재 논문은 워밍업 전략을 노이즈 수준에 따라 수동으로 선택합니다. 향후에는:

$$\text{Aug Strength}(t) = f(\hat{\eta}_t, t)$$

노이즈 수준 $\hat{\eta}_t$를 온라인으로 추정하여 증강 강도를 자동으로 조절하는 **적응적 증강 스케줄러** 개발이 필요합니다.

#### ② 다양한 도메인 확장

- 텍스트 분류(NLP)에서의 노이즈 레이블 문제에 대한 적용 가능성 탐구
- 의료 이미지(Medical Imaging) 등 도메인 특화 증강 정책과의 결합

#### ③ 증강 정책 자동 최적화

논문에서 AutoAugment와 RandAugment를 사용했으나, LNL 특화 증강 정책 탐색:

$$\pi^* = \arg\max_\pi \text{Accuracy}(\text{LNL-Model}(\pi))$$

메타러닝 또는 강화학습 기반의 LNL 특화 증강 정책 학습이 기대됩니다.

#### ④ 계산 효율성 개선

두 번의 순전파로 인한 계산 비용 증가를 해결하기 위한:
- 지식 증류(knowledge distillation) 기반 경량화
- 증강 샘플 재사용 전략

#### ⑤ 비대칭 노이즈 및 인스턴스 의존적 노이즈

본 논문은 주로 대칭/비대칭 합성 노이즈를 다루었으나, 실세계의 **인스턴스 의존적 노이즈(instance-dependent noise)**에 대한 AUGDESC 효과 분석이 필요합니다.

#### ⑥ 손실 분포 이론적 분석

Figure 2로 경험적으로 제시된 손실 분포 변화에 대한 수학적 이론 체계화:

$$\mathbb{E}[l(\theta, \text{StrongAug}(x_{\text{clean}}))] \text{ vs } \mathbb{E}[l(\theta, \text{StrongAug}(x_{\text{noisy}}))]$$

두 기대값의 관계에 대한 이론적 바운드 도출이 향후 연구의 방향이 될 수 있습니다.

#### ⑦ 자기지도학습(Self-Supervised Learning)과의 결합

SimCLR, MoCo 등 대조 학습 기반 사전학습과 AUGDESC를 결합하면, 노이즈 환경에서의 표현 학습 품질을 더욱 향상시킬 수 있을 것으로 기대됩니다.

---

## 참고 자료

- **주 논문**: Nishi, K., Ding, Y., Rich, A., & Höllerer, T. (2021). *Augmentation Strategies for Learning with Noisy Labels*. arXiv:2103.02130v3 [cs.CV].
- Li, J., Socher, R., & Hoi, S. C. H. (2020). *DivideMix: Learning with Noisy Labels as Semi-Supervised Learning*. arXiv:2002.07394.
- Sohn, K., et al. (2020). *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence*. arXiv:2001.07685.
- Liu, S., et al. (2020). *Early-Learning Regularization Prevents Memorization of Noisy Labels*. arXiv:2007.00151.
- Cubuk, E. D., et al. (2019). *AutoAugment: Learning Augmentation Strategies from Data*. CVPR 2019.
- Cubuk, E. D., et al. (2020). *RandAugment: Practical Automated Data Augmentation with a Reduced Search Space*. CVPR Workshops 2020.
- Arpit, D., et al. (2017). *A Closer Look at Memorization in Deep Networks*. ICML 2017.
- Berthelot, D., et al. (2020). *ReMixMatch: Semi-Supervised Learning with Distribution Matching and Augmentation Anchoring*. ICLR 2020.
- Arazo, E., et al. (2019). *Unsupervised Label Noise Modeling and Loss Correction*. arXiv:1904.11238.
- Han, B., et al. (2018). *Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels*. NeurIPS 2018.
- **소스 코드**: https://github.com/KentoNishi/Augmentation-for-LNL
