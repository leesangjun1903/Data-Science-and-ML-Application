# SSR: An Efficient and Robust Framework for Learning with Unknown Label Noise 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 **LULN(Learning with Unknown Label Noise)** 이라는 새로운 문제 설정을 제안합니다. 이는 노이즈의 **유형(type)과 정도(degree) 모두 미지(unknown)** 인 상황에서 학습하는 문제입니다. 기존 방법들이 노이즈 유형에 대한 강한 가정과 복잡한 모듈 조합(semi-supervised learning, model co-training, self-supervised pre-training 등)에 의존하는 것과 달리, SSR은 **단순하고 효율적이며 강건한 프레임워크**를 제안합니다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **LULN 문제 정의** | 노이즈 유형·정도 모두 미지인 현실적 문제 설정 |
| **NPK 기반 샘플 선택** | 비모수적 KNN 분류기를 활용한 안정적인 클린 샘플 선택 |
| **PMC 기반 재레이블링** | 모수적 분류기의 고신뢰 예측을 활용한 점진적 재레이블링 |
| **Feature Consistency Loss** | 오픈셋 노이즈에 대한 일반화 향상을 위한 선택적 손실 함수 |
| **최소 하이퍼파라미터** | $\theta_s, \theta_r, K, \lambda$ 4개로 SOTA 달성 |
| **복잡한 메커니즘 불필요** | co-training, pre-training, SSL 없이 SOTA 초과 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**기존 방법들의 한계:**

1. **Probability-consistent methods**: 노이즈 전이 행렬(noise transition matrix) 기반 방법들은 오픈셋 노이즈를 모델링하기 어렵고, 중증 노이즈에서 성능이 저하됩니다.

2. **Probability-approximate methods (DivideMix 등)**: 특정 노이즈 패턴(예: 비대칭 노이즈에 대한 confidence penalty)에 대한 강한 가정을 포함하여 조건이 맞지 않을 때 성능이 저하됩니다.

3. **복잡한 하이퍼파라미터 튜닝 필요**: 노이즈 유형·비율에 따라 별도 설정이 필요합니다.

**본 논문이 가정하는 두 가지 필수 조건:**
- 이웃과 레이블이 일관성이 높은 샘플은 클린할 가능성이 높다
- 모델의 고신뢰 예측은 신뢰할 수 있다

---

### 2.2 제안 방법 및 수식

#### 문제 형식화

학습 데이터셋 $\mathcal{X} = \{\mathbf{x}\_i\}\_{i=1}^{N}$, $\mathbf{x}\_i \in \mathbb{R}^d$와 대응하는 원핫 레이블 $\mathcal{Y} = \{\mathbf{y}\_i\}_{i=1}^{N}$, $\mathbf{y}_i \in \{0,1\}^M$이 주어집니다.

- 인코더 $f$: 특징 추출기
- **PMC(Parametric Model Classifier)** $g_p$: 모수적 분류기, $\mathbf{p}_i \triangleq g_p(\mathbf{f}_i)$
- **NPK(Non-Parametric KNN classifier)** $g_q$: 비모수적 KNN 분류기, $\mathbf{q}_i \triangleq g_q(\mathbf{f}_i)$
- 특징 표현: $\mathbf{f}_i \triangleq f(\mathbf{x}_i)$

---

#### Step 1: 샘플 재레이블링 (PMC 기반)

PMC $g_p$의 예측 신뢰도가 임계값 $\theta_r$을 초과하면 재레이블링합니다:

$$l_i^r = \begin{cases} \arg\max_l \mathbf{p}_i(l), & \max_l \mathbf{p}_i(l) > \theta_r \\ l_i, & \max_l \mathbf{p}_i(l) \leq \theta_r \end{cases} $$

- 고신뢰 예측에만 재레이블링 적용 → 오픈셋 노이즈는 낮은 신뢰도로 인해 자동으로 배제
- PMC는 항상 상대적으로 클린한 서브셋으로 훈련되어 안정성 확보

---

#### Step 2: 클린 샘플 선택 (NPK 기반, 균형 이웃 투표)

**코사인 유사도 기반 이웃 탐색:**

$$s_{ij} \triangleq \frac{\mathbf{f}_i^T \mathbf{f}_j}{\|\mathbf{f}_i\|_2 \|\mathbf{f}_j\|_2}$$

샘플 $\mathbf{x}_i$의 $K$-최근접 이웃 인덱스 집합 $\mathcal{N}_i$에서 정규화 레이블 분포를 계산합니다:

$$\mathbf{q}_i' = \frac{1}{K} \sum_{n \in \mathcal{N}_i} \mathbf{y}_n^r$$

클래스 불균형을 보정한 균형 버전:

$$\mathbf{q}_i = \boldsymbol{\pi}^{-1} \mathbf{q}_i' $$

여기서 $\boldsymbol{\pi} = \sum_{i=1}^{N} \mathbf{y}_i^r$ (데이터셋 레이블 분포), $\boldsymbol{\pi}^{-1}$은 역수 원소를 가진 벡터입니다.

**일관성 측도(Consistency Measure):**

$$c_i = \frac{\mathbf{q}_i(l_i^r)}{\max_j \mathbf{q}_i(j)} $$

- $c_i = 1.0$: 이웃 투표가 현재 레이블과 완전히 일치 → 클린 샘플로 선택
- 임계값 $\theta_s = 1$ (기본값): 이웃 투표가 완전 일치하는 경우만 선택

---

#### Step 3: 모델 훈련

**지도 학습 손실 (클린 서브셋 대상):**

$$\mathcal{L}_{ce} = -{\mathbf{y}^r}^T \log g_p(f(\mathbf{x}))$$

**특징 일관성 손실 (전체 데이터 대상, 선택적):**

두 가지 다른 증강뷰 $\mathbf{x}_1, \mathbf{x}_2$에 대해:

$$\mathcal{L}_{fc} = -\frac{\mathbf{h}_1^\top \mathbf{h}_2}{\|\mathbf{h}_1\|_2 \|\mathbf{h}_2\|_2} $$

여기서 $\mathbf{h}\_1 \triangleq h_{pred}(h_{proj}(f(\mathbf{x}\_1)))$, $\mathbf{h}\_2 \triangleq h_{proj}(f(\mathbf{x}_2))$.

**전체 훈련 목적 함수:**

$$\mathcal{L} = \mathcal{L}_{ce} + \lambda \mathcal{L}_{fc} $$

- $\lambda = 0$: **SSR** (기본형)
- $\lambda \neq 0$ ( $\lambda = 1$ ): **SSR+** (특징 일관성 포함)

---

### 2.3 모델 구조

```
전체 구조 (SSR 프레임워크)
├── 인코더 f (ResNet 계열)
│   └── 특징 추출: f_i = f(x_i)
├── NPK g_q (비모수적 KNN 분류기)
│   ├── 코사인 유사도 계산
│   ├── K-최근접 이웃 탐색
│   ├── 균형 이웃 투표 q_i
│   └── 일관성 측도 c_i → 샘플 선택
├── PMC g_p (모수적 분류기, Linear Layer)
│   └── 고신뢰 예측 p_i → 재레이블링
└── (선택적) SSR+
    ├── Projector h_proj
    └── Predictor h_pred → 특징 일관성 손실 L_fc
```

**반복 훈련 과정 (Algorithm 1):**

```
for epoch in range(T):
    1. 재레이블링: (X, Y^r) ← PMC 기반 (Eq.3)
    2. 샘플 선택: (X_c, Y_c^r) ← NPK 기반 (Eq.1, Eq.2)
    3. 모델 훈련: L = L_ce + λL_fc (Eq.5)
```

---

### 2.4 성능 향상

#### CIFAR-10/100 합성 노이즈 (클로즈드셋)

| 방법 | CIFAR-10 90%sym | CIFAR-100 90%sym | 비고 |
|------|----------------|-----------------|------|
| DivideMix* | 76.0 | 31.5 | SSL+co-training |
| ELR+* | 78.7 | 33.4 | SSL 사용 |
| AugDesc* | 91.9 | 41.2 | SSL 사용 |
| C2D* | 93.6 | 58.7 | 사전학습 필요 |
| **SSR (ours)** | **94.6** | **61.8** | 없음 |
| **SSR+ (ours)** | **95.2** | **66.6** | 없음 |

*: semi-supervised learning, co-training, pre-training 사용

#### 실세계 노이즈 데이터셋

| 데이터셋 | 최고 기존 방법 | SSR+ | 개선폭 |
|---------|--------------|------|--------|
| WebVision Top1 | 79.16 (NGC) | **80.92** | +1.76% |
| Clothing1M | 75.11 (AugDesc*) | 74.83 | 경쟁적 |
| ANIMAL-10N | 84.1 (NCT) | **88.5** | +4.4% |

#### 복합 노이즈 (오픈셋 + 클로즈드셋)

| 방법 | 노이즈 0.3, 오픈비 0.5 | 노이즈 0.6, 오픈비 0.5 |
|------|----------------------|----------------------|
| EDM | 94.0 | 92.8 |
| **SSR+** | **96.2** | **95.2** |

---

### 2.5 한계점

1. **하이퍼파라미터 $\theta_r$의 노이즈 비율 의존성**: 높은 노이즈 비율($\geq 50\%$)에서는 $\theta_r = 0.8$, 낮은 노이즈에서는 $\theta_r = 0.9$를 사용하여 완전히 자동화되지 않았습니다.

2. **KNN의 계산 복잡도**: 대규모 데이터셋($N$이 매우 클 때)에서 KNN 탐색의 $O(N \cdot d)$ 복잡도가 병목이 될 수 있습니다. (논문에서는 CIFAR 50K 기준 9초로 무시 가능한 수준이나 수백만 규모에서는 재검토 필요)

3. **초기 특징 공간의 품질 의존성**: 랜덤 초기화에서도 작동하나, 초기 이웃 관계의 품질이 초기 샘플 선택 성능에 영향을 미칩니다.

4. **매우 높은 노이즈 비율(90%)에서의 초기 수렴 속도**: 재레이블링이 초기에 보수적으로 동작하여 초기 클린 서브셋이 매우 작을 수 있습니다.

5. **Clothing1M에서 경쟁적이지만 최고 아님**: AugDesc(75.11%) 대비 SSR+(74.83%)로 소폭 뒤집니다. 이는 Clothing1M의 매우 불균형한 노이즈 분포 특성 때문으로 분석됩니다.

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 NPK 기반 샘플 선택의 일반화 기여

NPK는 특징 공간의 **매니폴드 구조(manifold structure)**를 활용합니다. 딥 뉴럴 네트워크의 특징 공간은 랜덤 초기화에서도 의미 있는 이웃 관계를 형성하는 경향이 있어(입력 공간의 국소적 구조가 특징 공간에 부분적으로 보존됨), PMC보다 초기 단계에서 더 안정적인 선택이 가능합니다.

**일반화 관점에서의 핵심 메커니즘:**

$$\mathbf{q}_i = \boldsymbol{\pi}^{-1} \cdot \frac{1}{K}\sum_{n \in \mathcal{N}_i} \mathbf{y}_n^r$$

이 수식에서 $\boldsymbol{\pi}^{-1}$ 보정항은 클래스 불균형으로 인한 **선택 편향(selection bias)**을 완화하여, 소수 클래스에 대한 과소선택 문제를 방지합니다. 이는 실제 불균형 데이터셋(Clothing1M 등)에서의 일반화에 직접적으로 기여합니다.

### 3.2 특징 일관성 손실($\mathcal{L}_{fc}$)의 일반화 효과

$$\mathcal{L}_{fc} = -\frac{\mathbf{h}_1^\top \mathbf{h}_2}{\|\mathbf{h}_1\|_2 \|\mathbf{h}_2\|_2}$$

이는 SimSiam(Chen & He, 2021)에서 영감을 받은 **자기지도적 정규화** 항으로, 다음과 같은 일반화 이점을 제공합니다:

1. **전체 데이터 활용**: 노이즈 샘플(오픈셋 포함)도 특징 일관성 학습에 참여하여 표현 학습 품질을 높입니다.
2. **오픈셋 노이즈 완화**: 오픈셋 샘플은 재레이블링되지 않으나, $\mathcal{L}_{fc}$ 통해 인코더 $f$의 표현 능력 향상에 기여합니다.
3. **과적합 방지**: 클린 서브셋만으로 훈련할 때 발생할 수 있는 과적합을 전체 데이터의 증강 일관성으로 완화합니다.

**효과 확인:**

| 설정 | CIFAR-10 90%sym | CIFAR-100 90%sym |
|------|----------------|-----------------|
| SSR ($\lambda=0$) | 94.6 | 61.8 |
| SSR+ ($\lambda=1$) | **95.2** | **66.6** |

$\mathcal{L}_{fc}$ 추가로 90% 극단적 노이즈에서도 일관된 성능 향상이 확인됩니다.

### 3.3 점진적 재레이블링의 커리큘럼 학습 효과

재레이블링은 고신뢰 샘플부터 시작하여 점진적으로 훈련 풀을 확장합니다. 이는 **커리큘럼 학습(Curriculum Learning)**의 원리와 일치하며:

- 초기: 소수의 고품질 클린 샘플로 안정적 표현 학습
- 후기: 재레이블된 샘플 포함으로 훈련 데이터 다양성 증가

이 과정이 일반화에 미치는 수학적 근거: 높은 $\theta_r$ 임계값으로 재레이블된 샘플은 낮은 경사(gradient) 기여를 가지며, 잘못 재레이블된 경우에도 모델에 미치는 부정적 영향이 최소화됩니다.

### 3.4 균형 샘플링 전략

훈련 단계에서 소수 클래스 오버샘플링과 Eq.(1)의 $\boldsymbol{\pi}^{-1}$ 보정을 통해 클래스 불균형을 이중으로 처리합니다. 이는 특히 실세계 노이즈 데이터셋에서 소수 클래스에 대한 일반화를 향상시킵니다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

**① LULN 패러다임의 확산**

SSR이 제안한 LULN 설정은 이후 연구들이 더 현실적인 노이즈 조건(미지의 노이즈 유형과 비율)을 표준 벤치마크로 채택하도록 촉진할 것입니다. 이는 실용적 AI 시스템 개발에 더욱 부합합니다.

**② 비모수적 접근의 재평가**

NPK가 PMC보다 샘플 선택에서 안정적임을 실증적으로 보임으로써, 복잡한 확률 모델링 없이도 특징 공간의 구조적 정보만으로 강건한 레이블 노이즈 처리가 가능함을 입증했습니다. 이는 대규모 언어 모델(LLM)의 데이터 정제 파이프라인에도 적용 가능성이 있습니다.

**③ 단순성-성능 트레이드오프 재정의**

복잡한 semi-supervised learning, co-training 없이 SOTA를 달성함으로써, 노이즈 레이블 학습 분야에서의 "복잡성 = 성능"이라는 통념에 도전합니다.

**④ 특징 일관성 정규화의 범용성**

$\mathcal{L}_{fc}$는 노이즈 레이블 학습을 넘어 다양한 도메인(의료 영상, 자율주행, NLP)에서 레이블 품질이 불완전한 경우 일반화 향상을 위한 보조 손실로 활용될 수 있습니다.

---

### 4.2 향후 연구 시 고려사항

**① 대규모 데이터셋에서의 KNN 효율성**

현재 논문은 CIFAR(50K), WebVision(65K), Clothing1M(32K 배치)에서 검증하였습니다. 수백만 규모의 데이터셋에서 정확한 KNN 탐색은 현실적이지 않을 수 있습니다. **FAISS** 또는 **HNSW** 등의 근사 최근접 이웃(Approximate Nearest Neighbor) 알고리즘과의 통합이 필요합니다.

**② $\theta_r$의 적응적 설정**

현재 $\theta_r = 0.8$ (높은 노이즈) / $0.9$ (낮은 노이즈)를 수동으로 설정합니다. **자동 임계값 추정** (예: 손실 분포 기반 GMM, 엔트로피 기반 적응적 설정)을 통해 완전한 노이즈 불가지론(agnostic) 방법으로 발전시킬 필요가 있습니다.

**③ 텍스트/멀티모달 도메인 확장**

현재 논문은 이미지 분류에 집중합니다. NLP 또는 멀티모달(이미지-텍스트 쌍) 데이터에서의 노이즈 레이블 문제 (예: CLIP 기반 웹크롤링 데이터)에 SSR을 적용할 때 특징 공간의 성질이 다를 수 있으므로, **도메인별 유사도 측도** 재정의가 필요합니다.

**④ 장기꼬리 분포(Long-tail Distribution)와의 결합**

실세계 데이터셋은 노이즈 레이블과 클래스 불균형이 동시에 존재하는 경우가 많습니다. SSR의 $\boldsymbol{\pi}^{-1}$ 보정이 장기꼬리 분포에서도 충분한지, 또는 LogitAdjustment 등 전문 기법과의 결합이 필요한지 체계적 연구가 필요합니다.

**⑤ 이론적 보장 부재**

현재 SSR은 경험적 성능에 의존합니다. NPK 기반 선택의 **일관성(consistency)** 및 **수렴 보장(convergence guarantee)**에 대한 이론적 분석이 추후 연구에서 이루어져야 합니다.

**⑥ Foundation Model과의 시너지**

DINO, CLIP 등 대규모 사전 학습된 모델의 특징 공간은 이미 강한 의미적 이웃 구조를 가집니다. SSR의 NPK 선택 메커니즘을 이러한 사전 학습 특징과 결합하면 **웜업 단계 없이도** 더 정확한 초기 샘플 선택이 가능할 것입니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 핵심 방법 | 가정 | SSR 대비 |
|------|----------|------|---------|
| **DivideMix** (Li et al., 2020, arXiv:2002.07394) | GMM 손실 모델링 + MixMatch | 비대칭 노이즈 가정, co-training 필요 | SSR이 90%sym에서 +19.2% 우세 |
| **ELR+** (Liu et al., 2020, arXiv:2007.00151) | Early-learning regularization | co-training 필요 | SSR이 90%sym에서 +16.5% 우세 |
| **NGC** (Wu et al., 2021, arXiv:2108.11035) | KNN 그래프 + 위상적 필터링 | 오픈셋 노이즈 가정 | SSR이 더 단순하고 WebVision에서 +1.76% 우세 |
| **AugDesc** (Nishi et al., CVPR 2021) | 증강 전략 + co-training | co-training 필요 | SSR이 90%sym에서 +3.3% 우세 |
| **C2D** (Zheltonozhskii et al., 2021, arXiv:2103.13646) | 자기지도 사전학습 + 대조학습 | 사전학습 단계 필요 | SSR이 90%sym에서 +6.5% 우세 (사전학습 없이) |
| **EDM** (Sachdeva et al., WACV 2021) | DivideMix 확장 + 오픈셋 처리 | 오픈셋 비율 정보 필요 | 복합 노이즈에서 SSR+ +2.2% 우세 |
| **LongReMix** (Cordeiro et al., 2021, arXiv:2103.04173) | 고신뢰 샘플 재혼합 | 클로즈드셋 가정 | WebVision에서 SSR+ 우세 |
| **SSR** (Feng et al., **본 논문**, 2022) | NPK 선택 + PMC 재레이블링 | **최소 가정** | 기준점 |

### 비교 분석 요약

**SSR의 차별점:**
- C2D 대비: C2D는 자기지도 사전학습(MoCo 등)이 필요하지만 SSR은 **처음부터(from scratch)** 훈련하면서도 더 높은 성능 달성
- DivideMix 대비: semi-supervised learning(MixMatch) 없이 더 넓은 노이즈 조건에서 강건
- NGC 대비: KNN 그래프의 복잡한 위상 분석 없이 단순 KNN 분류로도 충분함을 입증

**SSR의 한계 (최신 연구 관점):**
- 2023년 이후 등장한 **대규모 언어 모델 기반 레이블 정제** (예: GPT-4를 활용한 레이블 검증) 방법들과의 비교는 이루어지지 않음
- **연속 레이블(soft label)** 환경이나 **회귀 작업**에서의 적용성은 검증되지 않음

---

## 참고 자료

1. **Chen Feng, Georgios Tzimiropoulos, Ioannis Patras.** "SSR: An Efficient and Robust Framework for Learning with Unknown Label Noise." *BMVC 2022*. arXiv:2111.11288v2.

2. **Junnan Li, Richard Socher, Steven CH Hoi.** "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." arXiv:2002.07394, 2020.

3. **Sheng Liu et al.** "Early-Learning Regularization Prevents Memorization of Noisy Labels." arXiv:2007.00151, 2020.

4. **Zhi-Fan Wu et al.** "NGC: A Unified Framework for Learning with Open-World Noisy Data." arXiv:2108.11035, 2021.

5. **Kento Nishi et al.** "Augmentation Strategies for Learning with Noisy Labels." *CVPR 2021*.

6. **Evgenii Zheltonozhskii et al.** "Contrast to Divide: Self-supervised Pre-training for Learning with Noisy Labels." arXiv:2103.13646, 2021.

7. **Ragav Sachdeva et al.** "EvidentialMix: Learning with Combined Open-set and Closed-set Noisy Labels." *WACV 2021*.

8. **Xinlei Chen, Kaiming He.** "Exploring Simple Siamese Representation Learning." *CVPR 2021*.

9. **Bo Han et al.** "A Survey of Label-noise Representation Learning: Past, Present and Future." arXiv:2011.04406, 2020.

10. **Hwanjun Song et al.** "Learning from Noisy Labels with Deep Neural Networks: A Survey." *IEEE TNNLS*, 2022.

11. **GitHub 코드**: https://github.com/MrChenFeng/SSR_BMVC2022
