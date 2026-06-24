# Learning Semantic Representations for Unsupervised Domain Adaptation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존의 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA) 방법들은 **전역 분포(global distribution)** 통계만을 정렬하여, 클래스 수준의 의미론적(semantic) 정보를 무시한다는 근본적 문제를 지적합니다. 예를 들어, 완벽한 도메인 혼동(domain confusion)이 이루어지더라도, 타겟 도메인의 배낭(backpack) 특징이 소스 도메인의 자동차(car) 특징 근처에 매핑될 수 있습니다. 이 논문은 **Moving Semantic Transfer Network (MSTN)**을 제안하여, 레이블이 없는 타겟 샘플에 대해서도 클래스 수준의 의미론적 정렬을 달성하고자 합니다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **의미론적 표현 학습** | 비지도 환경에서 유사 레이블(pseudo-label)을 이용한 클래스별 센트로이드 정렬 |
| **이동 평균 센트로이드** | 미니배치의 불충분한 카테고리 정보 문제와 잘못된 유사 레이블 노이즈를 완화 |
| **이론적 근거 제시** | Ben-David et al. (2010)의 도메인 적응 이론 상한(bound)과의 연계 분석 |
| **안정적 적대적 학습** | 의미론적 정렬이 적대적 훈련의 불안정성을 완화하고 JSD 수렴을 가속화 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 기존 방법의 한계

기존 적대적 도메인 적응(Adversarial Domain Adaptation) 방법들(예: RevGrad, ADDA)은 도메인 판별기(domain discriminator)를 통해 **도메인 수준(domain-level)**의 분포 정렬만 수행합니다. 이로 인해 발생하는 문제:

1. **클래스 경계 근처의 모호한 특징 생성**: 소스와 타겟 도메인이 융합되더라도, 같은 클래스의 특징들이 클래스 경계 근처에 모이는 현상이 발생합니다.
2. **의미론적 불일치**: 다른 클래스의 특징들이 같은 특징 공간 위치에 매핑될 수 있습니다.
3. **비지도 환경의 의미론적 정렬 불가**: 타겟 도메인에 레이블이 없으므로, 클래스별 분포 매칭이 불가능했습니다.

### 2.2 제안하는 방법 (수식 포함)

#### 전체 목적 함수

MSTN의 최종 목적 함수는 세 가지 손실의 합으로 구성됩니다:

$$\mathcal{L}(\mathcal{X}_S, \mathcal{Y}_S, \mathcal{X}_T) = \mathcal{L}_C(\mathcal{X}_S, \mathcal{Y}_S) + \lambda \mathcal{L}_{DC}(\mathcal{X}_S, \mathcal{X}_T) + \gamma \mathcal{L}_{SM}(\mathcal{X}_S, \mathcal{Y}_S, \mathcal{X}_T) \tag{5}$$

각 항의 의미:
- $\mathcal{L}_C$: 소스 도메인 분류 손실 (교차 엔트로피)
- $\mathcal{L}_{DC}$: 도메인 혼동 손실 (적대적 손실)
- $\mathcal{L}_{SM}$: 의미론적 정렬 손실 (본 논문의 핵심 기여)
- $\lambda, \gamma$: 균형 파라미터

#### ① 소스 분류 손실 (Source Classification Loss)

$$\mathcal{L}_C(\mathcal{X}_S, \mathcal{Y}_S) = \mathbb{E}_{(x,y) \sim \mathcal{D}_S} [J(f(x), y)] \tag{1의 일부}$$

여기서 $J(\cdot, \cdot)$는 교차 엔트로피 손실, $f = F \circ G$ (분류기 $F$ + 특징 추출기 $G$)입니다.

#### ② 도메인 혼동 손실 (Domain Confusion Loss)

도메인 판별기 $D$를 이용한 적대적 손실:

$$d(\mathcal{X}_S, \mathcal{X}_T) = \mathbb{E}_{x \sim \mathcal{D}_S}[\log(1 - D \circ G(x))] + \mathbb{E}_{x \sim \mathcal{D}_T}[\log(D \circ G(x))] \tag{2}$$

특징 추출기 $G$는 $D$를 속이도록, $D$는 도메인을 구별하도록 학습됩니다.

#### ③ 의미론적 정렬 손실 (Semantic Alignment Loss)

비지도 도메인 적응을 위한 핵심 손실 함수:

$$\mathcal{L}_{SM}^{UDA}(\mathcal{X}_S, \mathcal{Y}_S, \mathcal{X}_T) = \sum_{k=1}^{K} \Phi(C_S^k, C_T^k) \tag{4}$$

- $C_S^k$: 소스 도메인의 클래스 $k$ 센트로이드
- $C_T^k$: 타겟 도메인의 유사 레이블 기반 클래스 $k$ 센트로이드
- $\Phi(x, x') = \|x - x'\|^2$: 제곱 유클리드 거리

#### ④ 이동 평균 센트로이드 업데이트 (Moving Average Centroid Update)

반복 $t$에서의 센트로이드 업데이트 규칙:

$$C_S^k \leftarrow \theta C_S^k + (1 - \theta) C_{S_{(t)}}^k \tag{Algorithm 1, Line 8}$$

$$C_T^k \leftarrow \theta C_T^k + (1 - \theta) C_{T_{(t)}}^k \tag{Algorithm 1, Line 9}$$

- $\theta$: 이동 평균 계수 (실험에서 $\theta = 0.7$로 설정)
- $C_{S_{(t)}}^k = \frac{1}{|S_t^k|} \sum_{(x_i, y_i) \in S_t^k} G(x_i)$: 현재 미니배치의 소스 센트로이드
- $C_{T_{(t)}}^k = \frac{1}{|\widetilde{T}\_t^k|} \sum_{(x_i, y_i) \in \widetilde{T}_t^k} G(x_i)$: 현재 미니배치의 유사 레이블 타겟 센트로이드

### 2.3 모델 구조 (Architecture)

```
입력 이미지
    │
    ▼
┌─────────────────────────────┐
│    특징 추출기 G (Feature     │
│    Extractor: AlexNet       │
│    conv1~fc7 + bottleneck)  │
└────────────┬────────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
┌─────────┐    ┌──────────────┐
│분류기 F  │    │도메인 판별기 D│
│(Classif-│    │(Discriminat- │
│ier)     │    │ or)          │
└────┬────┘    └──────┬───────┘
     │                │
     ▼                ▼
┌─────────┐    ┌──────────────┐
│분류 손실 │    │도메인 혼동손실│
│L_C      │    │L_DC          │
└─────────┘    └──────────────┘
     
센트로이드 계산 및 이동 평균 업데이트
     │
     ▼
┌─────────────────────────────┐
│의미론적 정렬 손실 L_SM       │
│Φ(C_S^k, C_T^k) for k=1..K  │
└─────────────────────────────┘
```

**세부 구성 요소:**
- **G (Feature Extractor)**: AlexNet (conv1~fc7) + 256 유닛 bottleneck layer
- **F (Classifier)**: 소프트맥스 분류기
- **D (Domain Discriminator)**: $x \rightarrow 1024 \rightarrow 1024 \rightarrow 1$ (dropout 포함)
- **학습률 스케줄**: $\mu_p = \frac{\mu_0}{(1 + \alpha \cdot p)^\beta}$ ($\mu_0=0.01$, $\alpha=10$, $\beta=0.75$)
- **균형 파라미터**: $\lambda = \frac{2}{1 + \exp(-\gamma \cdot p)} - 1$로 동적 조정

### 2.4 성능 향상

#### Office-31 (AlexNet 기반)

| Method | A→W | D→W | W→D | A→D | D→A | W→A | Avg |
|---|---|---|---|---|---|---|---|
| AlexNet | 61.6 | 95.4 | 99.0 | 63.8 | 51.1 | 49.8 | 70.1 |
| RevGrad | 73.0 | 96.4 | 99.2 | 72.3 | 53.4 | 51.2 | 74.3 |
| JAN | 74.9 | 96.6 | 99.5 | 71.8 | 58.3 | 55.0 | 76.0 |
| AutoDIAL | 75.5 | 96.6 | 99.5 | 73.6 | 58.1 | 59.4 | 77.1 |
| **MSTN (ours)** | **80.5** | **96.9** | **99.9** | **74.5** | **62.5** | **60.0** | **79.1** |

#### MNIST-USPS-SVHN

| Method | SVHN→MNIST | MNIST→USPS |
|---|---|---|
| RevGrad | 73.9 | 77.1 |
| ADDA | 76.0 | 89.4 |
| **MSTN (ours)** | **91.7** | **92.9** |

특히 **어려운 전이 태스크(A→W, SVHN→MNIST)**에서 큰 폭의 성능 향상이 확인됩니다.

### 2.5 한계점

1. **유사 레이블 오류 누적**: 초기 학습 시 부정확한 유사 레이블이 센트로이드를 잘못 안내할 수 있습니다. $\gamma$를 초기에 작게 설정하는 방식으로 완화하지만 완전히 해결하지는 못합니다.
2. **AlexNet 기반 실험**: 당시 더 강력한 백본(ResNet 등)에서의 검증이 부족합니다.
3. **클래스 불균형 문제**: 클래스가 불균형한 경우, 센트로이드 계산 자체가 편향될 수 있습니다.
4. **하이퍼파라미터 민감성**: $\lambda$, $\gamma$, $\theta$ 등 세 가지 하이퍼파라미터 튜닝이 필요합니다.
5. **대규모 카테고리 수 확장성**: 클래스 수 $K$가 매우 크면 $2K$개의 센트로이드 유지 비용이 증가합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 기반: Ben-David et al. (2010) 상한 분석

도메인 적응 이론에 따르면, 타겟 오류 $\varepsilon_T(h)$는 다음과 같이 상한이 정해집니다:

$$\forall h \in \mathcal{H},\; \varepsilon_T(h) \leq \varepsilon_S(h) + \frac{1}{2} d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{S}, \mathcal{T}) + C \tag{6}$$

여기서 $C = \min_{h \in \mathcal{H}} \varepsilon_S(h, f_S) + \varepsilon_T(h, f_T)$는 공유 기대 손실입니다. 기존 방법들은 $C$를 무시하거나 무시할 수 있다고 가정했지만, $C$가 클 경우 소스 오류만 최소화해도 좋은 타겟 분류기를 얻을 수 없습니다.

MSTN은 삼각 부등식을 활용하여 $C$의 상한을 다음과 같이 분해합니다:

$$C \leq \min_{h \in \mathcal{H}} \varepsilon_S(h, f_S) + \varepsilon_T(h, f_S) + \varepsilon_T(f_S, f_{\widehat{T}}) + \varepsilon_T(f_T, f_{\widehat{T}}) \tag{7}$$

- **첫 번째, 두 번째 항**: 소스 레이블로 쉽게 최소화 가능
- **마지막 항** $\varepsilon_T(f_T, f_{\widehat{T}})$: 유사 레이블 오류율로, 학습이 진행될수록 감소
- **세 번째 항** $\varepsilon_T(f_S, f_{\widehat{T}})$: **MSTN의 센트로이드 정렬이 이를 최소화**

구체적으로, 클래스 $k$에 대해 소스 센트로이드와 유사 레이블 타겟 센트로이드를 정렬하면:

$$\mathbb{E}_{x \sim S^k} G(x) = \mathbb{E}_{x \sim \widetilde{T}^k} G(x)$$

이는 소스 레이블 함수 $f_S$가 타겟 샘플에 대해서도 올바른 예측을 하게 하여, $\varepsilon_T(f_S, f_{\widehat{T}})$를 줄이는 역할을 합니다. 이것이 **MSTN이 단순한 도메인 정렬을 넘어 일반화 성능을 향상시키는 이론적 근거**입니다.

### 3.2 일반화 성능 향상의 실증적 근거

1. **어려운 전이 태스크에서의 큰 성능 향상**: A→W (+7.5%), SVHN→MNIST (+15.7% vs. RevGrad)는 의미론적 정렬이 도메인 격차가 클수록 더 효과적임을 보여줍니다.

2. **t-SNE 시각화**: Fig. 2의 결과는 MSTN이 단순히 도메인을 융합하는 것을 넘어, 같은 클래스 특징들을 밀집시키고 다른 클래스 특징들을 분산시켜 더 discriminative한 표현을 학습함을 보여줍니다.

3. **A-distance 분석**: MSTN과 RevGrad의 A-distance는 유사하지만 분류 성능은 MSTN이 훨씬 높습니다. 이는 **전역 도메인 정렬만으로는 일반화가 충분하지 않으며**, 의미론적 정렬이 추가적인 일반화 성능을 제공함을 입증합니다.

4. **이동 평균의 효과**: 어려운 전이 태스크(D→A)에서 이동 평균 센트로이드가 직접 계산 방식 대비 유의미한 성능 향상을 보여, 노이즈에 강건한 일반화가 가능합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

**① 의미론적 정렬의 중요성 확립**

MSTN은 UDA에서 클래스 수준의 의미론적 정렬이 도메인 수준의 정렬만큼 (혹은 그 이상으로) 중요하다는 것을 실증적·이론적으로 보여주었습니다. 이후 많은 연구들이 이 방향을 계승합니다.

**② 유사 레이블 + 센트로이드 패러다임의 확산**

유사 레이블을 직접 사용하는 대신 센트로이드 수준에서 정렬하는 아이디어는, 이후 프로토타입 기반(prototype-based) 도메인 적응 연구의 기초가 되었습니다.

**③ 이동 평균 메커니즘의 영향**

이동 평균 센트로이드 아이디어는 Momentum Contrast (MoCo) 등 자기 지도 학습(self-supervised learning) 방법론과 개념적으로 유사하며, 안정적인 표현 학습의 필요성을 부각시켰습니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래 연구들은 MSTN의 아이디어를 계승하거나 발전시킨 대표적인 후속 연구들입니다. 단, 제가 직접 확인한 논문 내용에 기반하여 기술하며, 수치는 해당 논문들에서 보고된 값을 참조합니다.

#### ① SHOT (Liang et al., ICML 2020)
- **논문**: "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation"
- **핵심 아이디어**: 소스 데이터 없이 가설(hypothesis)만 전이. 정보 극대화(information maximization)와 유사 레이블 기반 자기 지도 학습을 결합.
- **MSTN과의 차별점**: 소스 데이터 접근 불필요(source-free DA), 더 현실적인 시나리오 대응.
- **Office-31 (ResNet-50)**: A→W 94.0%, 평균 88.6%로 MSTN 대비 큰 폭 향상 (백본 차이 고려 필요).

#### ② CDAN (Long et al., NeurIPS 2018)
- **논문**: "Conditional Adversarial Domain Adaptation"
- **핵심 아이디어**: 조건부 도메인 적대적 네트워크(conditional adversarial network)를 이용해 멀티리니어 맵으로 클래스 정보를 도메인 정렬에 반영.
- **MSTN과의 차별점**: 센트로이드 정렬이 아닌 분류기 예측값과 특징의 결합으로 의미론적 정렬을 암묵적으로 달성.
- **Office-31 (ResNet-50)**: 평균 87.7%.

#### ③ MCC (Jin et al., ECCV 2020)
- **논문**: "Minimum Class Confusion for Versatile Domain Adaptation"
- **핵심 아이디어**: 클래스 혼동(class confusion)을 최소화하는 방향으로 정렬, 온도 기반 샘플 가중치 적용.
- **MSTN과의 차별점**: 클래스 혼동 행렬을 직접 최소화하여 의미론적 구조를 유지.

#### ④ NWD (Chen et al., CVPR 2021)
- **논문**: "I Think, Therefore I am: Reasoning About Ego-Motion with Self-Attention" (도메인 적응 관련 센트로이드 연구)
- 이 분야에서 프로토타입 네트워크 기반 UDA 연구가 활발히 진행 중입니다.

#### ⑤ Prototype-based DA (Tanwisuth et al., ICML 2021)
- **논문**: "A Prototype-Oriented Framework for Unsupervised Domain Adaptation"
- **핵심 아이디어**: MSTN의 센트로이드 아이디어를 확장하여, 확률적 프로토타입과 최적 수송(Optimal Transport)을 결합.
- **MSTN과의 연관성**: MSTN의 직접적인 발전 방향으로 볼 수 있음.

#### 비교 요약 테이블

| 논문 | 연도 | 핵심 방법 | MSTN 대비 차별점 |
|---|---|---|---|
| MSTN (본 논문) | 2018 | 이동 평균 센트로이드 정렬 | 기준 |
| CDAN | 2018 | 조건부 적대적 정렬 | 클래스 정보를 적대적 손실에 직접 통합 |
| SHOT | 2020 | 소스 프리 가설 전이 | 소스 데이터 불필요 |
| MCC | 2020 | 클래스 혼동 최소화 | 클래스 혼동 행렬 직접 최적화 |
| Prototype DA | 2021 | 확률적 프로토타입 + OT | MSTN의 센트로이드를 확률론적으로 확장 |

### 4.3 앞으로 연구 시 고려할 점

**① 더 강력한 백본 및 사전학습 모델 활용**

MSTN은 AlexNet 기반으로 실험되었습니다. ViT(Vision Transformer), CLIP 등 최신 사전학습 모델과 결합 시 의미론적 정렬의 효과가 더욱 강력해질 수 있습니다. 특히 CLIP은 이미 의미론적으로 풍부한 표현을 갖고 있어 센트로이드 정렬의 효과를 증폭할 수 있습니다.

**② 소스 프리(Source-Free) 도메인 적응으로의 확장**

현실에서는 개인정보 보호 등의 이유로 소스 데이터에 접근이 불가능한 경우가 많습니다. MSTN의 센트로이드 개념을 소스 프리 환경에 적용하는 연구가 필요합니다. 예를 들어, 소스 센트로이드를 미리 저장하고 타겟 도메인 적응 시 활용하는 방식을 고려할 수 있습니다.

**③ 유사 레이블 품질 개선**

MSTN의 핵심 약점은 초기 유사 레이블의 부정확성입니다. 이를 위해:
- **신뢰도 임계값 기반 필터링**: 높은 신뢰도의 샘플만 센트로이드 계산에 사용
- **자기 페이싱(self-pacing) 학습**: 점진적으로 더 어려운 샘플을 포함
- **앙상블 기반 유사 레이블 생성**: 여러 모델의 예측을 종합하여 품질 향상

**④ 멀티 소스 및 멀티 타겟 도메인으로의 확장**

실세계 문제에서는 하나의 소스 도메인에서 여러 타겟 도메인으로, 또는 여러 소스 도메인에서 하나의 타겟 도메인으로 전이하는 경우가 많습니다. MSTN의 센트로이드 정렬을 다중 도메인으로 확장하면 추가적인 일반화가 가능합니다.

**⑤ 클래스 불균형 처리**

실제 데이터셋에서는 클래스 불균형이 자주 발생합니다. 클래스 불균형이 있을 때 센트로이드 계산이 편향될 수 있으므로, 클래스별 가중치 조정이나 오버샘플링 전략을 센트로이드 정렬과 통합하는 것이 중요합니다.

**⑥ 개방 집합(Open-Set) 및 부분(Partial) 도메인 적응**

MSTN은 소스와 타겟이 동일한 클래스 집합을 가정합니다. 타겟 도메인에 소스에 없는 새로운 클래스가 있는 경우(Open-Set DA)나 소스의 일부 클래스만 타겟에 존재하는 경우(Partial DA)에 대한 확장이 필요합니다.

**⑦ 최적 수송(Optimal Transport)과의 결합**

단순 유클리드 거리 기반 센트로이드 정렬 대신, Wasserstein 거리나 최적 수송을 활용하면 분포 정렬의 이론적 엄밀성과 실제 성능을 모두 향상시킬 수 있습니다.

---

## 참고자료

**본 논문 (직접 분석)**
- Xie, S., Zheng, Z., Chen, L., & Chen, C. (2018). **Learning Semantic Representations for Unsupervised Domain Adaptation**. *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)*, PMLR 80.

**논문 내 인용 참고문헌**
- Ben-David, S., et al. (2010). A theory of learning from different domains. *Machine Learning*, 79(1):151–175.
- Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. *ICML 2015*.
- Tzeng, E., et al. (2017). Adversarial discriminative domain adaptation. *arXiv:1702.05464*.
- Long, M., et al. (2017). Deep transfer learning with joint adaptation networks. *ICML 2017*.
- Lee, D.-H. (2013). Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. *ICML Workshop*.

**2020년 이후 비교 연구 (공개 논문 기반)**
- Liang, J., et al. (2020). **Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation (SHOT)**. *ICML 2020*.
- Long, M., et al. (2018). **Conditional Adversarial Domain Adaptation (CDAN)**. *NeurIPS 2018*.
- Jin, Y., et al. (2020). **Minimum Class Confusion for Versatile Domain Adaptation (MCC)**. *ECCV 2020*.
- Tanwisuth, K., et al. (2021). **A Prototype-Oriented Framework for Unsupervised Domain Adaptation**. *NeurIPS 2021*.

> **⚠️ 정확도 관련 주의사항**: 2020년 이후 최신 연구 비교 부분에서 일부 수치(Office-31 정확도 등)는 해당 논문들의 공개된 결과를 참조하였으나, 저의 학습 데이터 기준일(2023년) 이후의 연구나 일부 세부 수치는 직접 확인이 어려울 수 있습니다. 최신 비교를 위해서는 Papers With Code (paperswithcode.com)의 Office-31 벤치마크 리더보드를 직접 확인하시기를 권장합니다.
