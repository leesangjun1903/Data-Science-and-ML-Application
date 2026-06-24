# Unsupervised Domain Adaptation with Similarity Learning

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문(SimNet)의 핵심 주장은 다음과 같습니다:

> **비지도 도메인 적응(Unsupervised Domain Adaptation)에서 기존의 완전 연결 계층(Fully-Connected) 분류기를 유사도 학습(Similarity Learning) 기반 분류기로 대체함으로써, 도메인 불변 특징의 일반화 성능을 크게 향상시킬 수 있다.**

기존 접근법은 두 단계로 구성됩니다:
1. 소스 도메인에서 낮은 위험을 보존하는 특징 학습
2. 두 도메인 특징을 구분 불가능하게 만드는 도메인 혼동(Domain Confusion)

이 논문은 분류 단계(1번)를 **프로토타입 기반 유사도 학습**으로 대체할 것을 제안합니다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **SimNet 모델 제안** | 유사도 학습 기반 분류기와 적대적 도메인 혼동을 결합한 엔드-투-엔드 모델 |
| **프로토타입 분류기** | 카테고리당 하나의 프로토타입 표현을 학습하여 분류 수행 |
| **쌍선형 유사도 함수** | 이미지 임베딩과 프로토타입 간 학습 가능한 비선형 유사도 정의 |
| **직교 정규화** | 프로토타입 간 중복을 방지하는 정규화 항 도입 |
| **SOTA 달성** | Digits, Office-31, VisDA 벤치마크에서 당시 최고 성능 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**도메인 이동(Domain Shift) 문제:**

소스 도메인 $\mathbf{X}_s = \{(x_i^s, y_i^s)\}\_{i=0}^{N_s}$과 타겟 도메인 $\mathbf{X}_t = \{(x_i^t, y_i^t)\}\_{i=0}^{N_t}$ 간의 분포 차이 $p_s(x,y) \neq p_t(x,y)$ 로 인해 소스 도메인에서 학습된 분류기가 타겟 도메인에서 성능이 저하됩니다.

**기존 방법의 한계:**

기존 적대적 도메인 적응 방법들(RevGrad, ADDA 등)은 도메인 불변 특징을 학습한 후 **완전 연결 소프트맥스 분류기**를 사용합니다. 그러나 이러한 접근은:

- 도메인 간 공유 표현이 노이즈에 취약함
- 도메인 적응 과정에서 분류 경계가 불안정해짐
- 특징 공간에서 카테고리 간 클러스터링이 명시적으로 유도되지 않음

---

### 2.2 제안 방법 (수식 포함)

#### 모델 구성: SimNet

SimNet은 두 가지 주요 구성 요소로 이루어집니다:

**(1) 프로토타입 계산**

카테고리 $c$의 프로토타입은 소스 도메인 이미지들의 평균 임베딩으로 정의됩니다:

$$\mu_c = \frac{1}{|\mathbf{X}^c|} \sum_{x_i^s \in \mathbf{X}^c} g(x_i^s) \tag{1}$$

- $g(\cdot)$: 프로토타입 임베딩 네트워크 (파라미터 $\theta_g$)
- $\mathbf{X}^c$: 소스 도메인에서 카테고리 $c$로 레이블된 모든 이미지의 집합

**(2) 쌍선형 유사도 함수**

입력 이미지 $x_i$와 프로토타입 $\mu_c$ 간의 유사도:

$$h(x_i, \mu_c) = f_i^T \mathbf{S} \mu_c \tag{2}$$

- $f_i = f(x_i) \in \mathbb{R}^n$: 입력 이미지 임베딩 네트워크 $f(\cdot)$ (파라미터 $\theta_f$)
- $\mathbf{S} \in \mathbb{R}^{n \times m}$: 학습 가능한 쌍선형 유사도 행렬 (양수 또는 대칭 조건 불필요)

효율적인 저차원 근사(Low-rank Approximation):

$$\mathbf{S} = \mathbf{U}^T \mathbf{V}, \quad \mathbf{U}, \mathbf{V} \in \mathbb{R}^{n \times m}, \quad m = 512 \tag{8}$$

따라서:

$$h(x_i, \mu_c) = (\mathbf{U} f_i)^T \cdot (\mathbf{V} \mu_c) \tag{8'}$$

**(3) 클래스 조건부 확률 (Softmax)**

$$p_\theta(c | x_i, \mu_1, \ldots, \mu_C) = \frac{e^{h(x_i, \mu_c)}}{\sum_k e^{h(x_i, \mu_k)}} \tag{3}$$

**(4) 분류 손실 함수 (Negative Log-Likelihood)**

$$\mathcal{L}_{class}(\theta) = -\sum_{(x_i, y_i)} \left[ h(x_i, \mu_{y_i}) - \log \sum_k e^{h(x_i, \mu_k)} \right] + \gamma \mathcal{R} \tag{4}$$

**(5) 직교성 정규화**

프로토타입 행렬 $\mathbf{P}_\mu$ (각 행이 프로토타입)에 대해:

$$\mathcal{R} = \|\mathbf{P}_\mu^T \mathbf{P}_\mu - \mathbf{I}\|_F^2 \tag{5}$$

- $\|\cdot\|_F^2$: 프로베니우스 노름의 제곱
- 프로토타입 간 직교성을 장려하여 카테고리별 독립적 표현 학습 유도

**(6) 도메인 분류기 손실 (RevGrad 기반)**

$$\mathcal{L}_{disc}(\theta, \theta_d) = -\sum_{i=0}^{N_s} \log D(f(x_i^s)) - \sum_{i=0}^{N_t} \log(1 - D(f(x_i^t))) \tag{6}$$

**(7) 최종 미니맥스 목적 함수**

$$\min_{\theta_f, \theta_g, \mathbf{S}} \max_{\theta_d} \mathcal{L}_{class}(\theta_f, \theta_g, \mathbf{S}) - \lambda \mathcal{L}_{disc}(\theta_f, \theta_d) \tag{7}$$

- $\lambda$: 두 손실 간 균형 파라미터 (실험에서 $\lambda = 0.5$)
- $\gamma$: 정규화 계수 ($\gamma = 0.01$)

---

### 2.3 모델 구조

```
[소스 도메인 이미지] → f(·) → f_i ─────────────────→ [도메인 판별기 D]
[타겟 도메인 이미지] → f(·) → f_i ─────────────────→ [RevGrad]

[소스 도메인 이미지_c] → g(·) → 평균 → μ_c (C개의 프로토타입)

유사도 계산: h(x_i, μ_c) = f_i^T S μ_c
분류: argmax_c p_θ(c|x_i)
```

**핵심 설계 선택:**

| 설계 요소 | 설명 | 이유 |
|-----------|------|------|
| $f$와 $g$의 파라미터 분리 | 두 네트워크가 별도 파라미터를 사용 | $f$는 도메인 불변성, $g$는 소스 표현에 집중 |
| 저차원 쌍선형 근사 | $\mathbf{S} = \mathbf{U}^T\mathbf{V}$ | 모델 용량 제어 및 추론 효율성 |
| 직교 정규화 | 프로토타입 직교성 유도 | 카테고리 간 혼동 감소 |
| ResNet-50 백본 | ImageNet 사전 학습 | 강력한 특징 추출 |

---

### 2.4 성능 향상 및 한계

#### 성능 향상

**Digits 데이터셋 (Table 1):**

| 방법 | USPS→MNIST | MNIST→USPS | MNIST→MNIST-M |
|------|-----------|-----------|--------------|
| RevGrad | 89.9 | 89.1 | 84.4 |
| ADDA | 90.1 | 89.4 | - |
| **SimNet** | **95.6** | **96.4** | **90.5** |

**Office-31 데이터셋 (Table 2):**

| 방법 | A→W | D→W | W→D | A→D | D→A | W→A | 평균 |
|------|-----|-----|-----|-----|-----|-----|------|
| JAN-A | 86.0 | 96.7 | 99.7 | 85.1 | 69.2 | 70.7 | 84.6 |
| **SimNet** | **88.6** | **98.2** | 99.7 | **85.3** | **73.4** | **71.8** | **86.2** |

**VisDA 데이터셋 (Table 3):**

| 방법 | 평균 정확도 |
|------|-----------|
| RevGrad-ours | 58.62 |
| JAN-A | 61.62 |
| **SimNet** | **69.58** |

특히 VisDA에서 RevGrad-ours 대비 **약 11%의 성능 향상**은 유사도 기반 분류기의 효과를 직접적으로 보여줍니다.

#### 한계

1. **프로토타입의 소스 도메인 의존성:** 프로토타입이 소스 도메인 이미지만으로 구성되므로, 타겟 도메인의 특이한 변동성이 반영되지 않음
2. **카테고리 수 확장성 미검증:** 논문에서 스스로 언급하듯, 대규모 카테고리(예: 수천 개)로의 확장성이 검증되지 않음
3. **픽셀 수준 도메인 이동 처리 부족:** PixelDA와 같은 이미지 생성 기반 방법보다 픽셀 수준 변환이 명확한 시나리오(MNIST→MNIST-M의 PixelDA: 98.2% vs SimNet: 90.5%)에서 열세
4. **단일 모달리티 검증:** 시각 분류 작업에만 국한되어 있으며, 세그멘테이션, 객체 탐지 등에 대한 적용은 미래 작업으로 남겨짐
5. **하이퍼파라미터 민감도:** $\lambda$, $\gamma$ 등의 균형 파라미터가 성능에 영향을 미치며, 새로운 도메인 쌍에 대한 튜닝이 필요

---

## 3. 모델의 일반화 성능 향상 가능성

SimNet이 일반화 성능을 향상시키는 메커니즘을 심층 분석합니다.

### 3.1 프로토타입 기반 분류의 일반화 이점

**직관적 설명:**

일반적인 소프트맥스 분류기는 결정 경계(Decision Boundary)를 직접 학습합니다. 도메인 이동이 발생하면 이 경계가 타겟 도메인에서 잘못 위치할 수 있습니다.

반면, 프로토타입 기반 분류기는 **카테고리의 중심(Centroid) 표현**을 학습합니다. 도메인 불변 특징 공간에서 카테고리 클러스터가 형성되면, 새로운 도메인의 샘플도 해당 클러스터에 자연스럽게 근접하게 됩니다.

**수학적 관점:**

소프트맥스 분류기의 결정 함수: $\hat{y} = \arg\max_c \mathbf{W}_c^T f(x) + b_c$

이 경우 $\mathbf{W}_c$는 소스 도메인 분포에 최적화됩니다.

SimNet의 결정 함수: $\hat{y} = \arg\max_c h(x, \mu_c) = \arg\max_c f(x)^T \mathbf{S} \mu_c$

여기서 $\mu_c$는 소스 도메인 이미지의 **집계 표현(Aggregated Representation)**으로, 단순 가중치 벡터보다 더 안정적인 카테고리 표현을 제공합니다.

### 3.2 직교 정규화의 역할

$$\mathcal{R} = \|\mathbf{P}_\mu^T \mathbf{P}_\mu - \mathbf{I}\|_F^2$$

이 정규화는 프로토타입 벡터들을 서로 직교하도록 유도합니다. 이는:

- **카테고리 간 특징 분리:** 각 카테고리가 독립적인 방향의 특징 공간을 점유
- **도메인 이동에 강건한 경계:** 카테고리 간 마진이 명확해져 도메인 이동 후에도 분류 경계가 유지됨

실험적 증거 (Table 4): SimNet-no-reg (66.1%) vs SimNet (69.6%)로 정규화의 중요성 확인

### 3.3 분리된 네트워크 구조의 역할

$f$와 $g$를 분리함으로써:

- **$f(\cdot)$**: 도메인 불변 특징 학습에 특화 (RevGrad의 적대적 학습 대상)
- **$g(\cdot)$**: 소스 도메인의 풍부한 카테고리 표현 학습에 특화 (도메인 불변성 강요 없음)

실험적 증거 (Table 4): SimNet-f=g (57.5%) vs SimNet (69.6%)로 분리의 중요성 확인

### 3.4 스케일링에 따른 일반화

| 모델 백본 | VisDA 성능 |
|----------|-----------|
| SimNet (ResNet-50) | 69.58% |
| SimNet-152 (ResNet-152) | 72.9% |

더 강력한 특징 추출기를 사용할수록 성능이 향상되어, SimNet의 쌍선형 분류기가 특징 품질에 잘 스케일링됨을 보입니다.

---

## 4. 미래 연구에 미치는 영향과 고려 사항

### 4.1 연구 영향

**긍정적 영향:**

1. **분류기 설계 패러다임 전환:** FC 분류기 외에 메트릭/유사도 기반 분류기가 도메인 적응에 효과적임을 실증적으로 보임
2. **퓨샷 학습과 도메인 적응의 연결:** 프로토타입 네트워크(Prototypical Networks)와 도메인 적응의 연결고리를 제공하여, 이후 많은 연구에서 이 두 분야의 융합 촉진
3. **구성 요소 교환 가능성:** 논문에서 언급하듯, SimNet의 유사도 분류기는 다양한 도메인 혼동 방법(MMD, ADDA 등)에 적용 가능한 모듈식 설계
4. **합성-실제 도메인 적응:** VisDA에서의 강력한 성능은 로보틱스, 자율주행 등의 시뮬레이션 기반 학습 분야에 직접적인 영향

### 4.2 2020년 이후 관련 최신 연구 비교 분석

SimNet 이후 이 분야는 크게 발전했습니다. 다음은 주요 후속 연구와의 비교입니다:

#### (1) 자기 지도 학습 + 도메인 적응

**SHOT (Liang et al., ICML 2020)**
- 소스 없는(Source-Free) 도메인 적응에서 가설 전이(Hypothesis Transfer)
- 정보 극대화와 자기 지도 의사 레이블링(Pseudo-labeling) 결합
- SimNet과의 관계: 타겟 도메인의 레이블 없이도 적응 가능하다는 점에서 SimNet의 한계(소스 도메인 프로토타입 의존)를 극복

**특징:** SHOT은 소스 모델이 주어진 상태에서 타겟 도메인만으로 적응하므로, 개인정보 보호 측면에서 SimNet보다 실용적

#### (2) 트랜스포머 기반 도메인 적응

**CDTrans (Xu et al., ICLR 2022)**
- 크로스 도메인 트랜스포머(Cross-Domain Transformer) 활용
- 자기 어텐션(Self-Attention)을 통한 도메인 간 특징 정렬
- Office-31 평균: ~92% (SimNet의 86.2% 대비 대폭 향상)

**TVT (Yang et al., 2021)**
- 비전 트랜스포머(ViT)를 도메인 적응에 적용
- 글로벌 특징 정렬과 로컬 특징 정렬을 동시에 수행

SimNet과의 차이: ViT 기반 방법들은 어텐션 메커니즘으로 더 정교한 특징 관계를 포착하지만, SimNet의 단순한 프로토타입 구조보다 계산 비용이 크게 높음

#### (3) 대조 학습 기반 도메인 적응

**CDAC (Li et al., CVPR 2021)**
- 크로스 도메인 대조 학습(Cross-Domain Contrastive Learning)
- SimNet의 프로토타입 아이디어를 발전시켜, 타겟 도메인 샘플도 대조 학습에 활용

**CDL (Su et al., NeurIPS 2020)**
- 조건부 도메인 정렬(Conditional Domain Alignment)
- 클래스 조건부 특징 분포를 정렬하는 방식은 SimNet의 카테고리별 프로토타입 아이디어와 개념적으로 유사

#### (4) 소스 없는 도메인 적응 (Source-Free DA)

2020년 이후 주요 트렌드 중 하나. SimNet은 소스 도메인 데이터에 의존하는 반면, 최신 방법들은 소스 데이터 없이 사전 학습된 모델만으로 적응을 시도합니다.

**비교 표:**

| 방법 | 연도 | 소스 필요 여부 | 주요 전략 | Office-31 avg |
|------|------|--------------|----------|--------------|
| SimNet | 2018 | ✅ | 프로토타입 유사도 + 적대적 학습 | 86.2% |
| CDAN (Long et al.) | 2018 | ✅ | 조건부 적대적 학습 | 87.7% |
| SHOT | 2020 | ❌ | 정보 극대화 + 의사 레이블 | 88.6% |
| CDTrans | 2022 | ✅ | 크로스 도메인 트랜스포머 | 92.4% |

### 4.3 향후 연구 시 고려할 점

**기술적 고려 사항:**

1. **타겟 도메인 프로토타입 업데이트:** SimNet의 프로토타입은 소스 도메인에만 기반합니다. 타겟 도메인의 의사 레이블(Pseudo-label)을 사용하여 프로토타입을 점진적으로 업데이트하는 방법을 탐구할 필요가 있습니다.

2. **오픈셋 도메인 적응:** 타겟 도메인에 소스에 없는 새로운 카테고리가 존재하는 경우의 처리 방법이 필요합니다. 프로토타입 기반 접근은 이 시나리오에서 자연스럽게 확장될 수 있는 가능성이 있습니다.

3. **멀티소스 도메인 적응:** 여러 소스 도메인에서 프로토타입을 집계하는 방법은 SimNet의 자연스러운 확장이 될 수 있습니다:

$$\mu_c^{multi} = \frac{1}{\sum_k |\mathbf{X}_k^c|} \sum_k \sum_{x_i^s \in \mathbf{X}_k^c} g(x_i^s)$$

4. **동적 프로토타입:** 학습 중 프로토타입을 동적으로 업데이트하는 메모리 뱅크(Memory Bank) 방식과의 결합

5. **트랜스포머 백본과의 결합:** ViT 등 트랜스포머 기반 특징 추출기와 SimNet의 유사도 분류기를 결합하면 추가 성능 향상 가능성

**연구 방향 제안:**

- **연합 학습(Federated Learning) + 도메인 적응:** 소스 도메인 데이터의 개인정보 보호 측면에서 프로토타입 공유만으로 도메인 적응을 수행하는 방법
- **퓨샷 도메인 적응:** SimNet의 프로토타입 구조는 타겟 도메인에 극소량의 레이블 데이터가 주어지는 퓨샷 시나리오로 자연스럽게 확장 가능
- **도메인 적응의 이론적 보장:** Ben-David et al.의 이론적 프레임워크에 SimNet의 유사도 학습을 통합하여 일반화 오류 상한(Generalization Error Bound)을 분석

---

## 참고 자료

**본 답변에서 직접 인용한 논문 (제공된 PDF 기반):**

1. **Pinheiro, P. O. (2018). "Unsupervised Domain Adaptation with Similarity Learning."** arXiv:1711.08995v2. *(본 분석의 주 대상 논문)*
2. **Ganin, Y. et al. (2016). "Domain-adversarial training of neural networks."** JMLR. *(RevGrad 방법)*
3. **Long, M. et al. (2017). "Deep transfer learning with joint adaptation networks."** ICML. *(JAN 방법)*
4. **Tzeng, E. et al. (2017). "Adversarial discriminative domain adaptation."** *(ADDA 방법)*
5. **Snell, J. et al. (2017). "Prototypical networks for few-shot learning."** *(프로토타입 네트워크)*
6. **He, K. et al. (2016). "Deep residual learning for image recognition."** CVPR. *(ResNet-50 백본)*

**2020년 이후 비교 연구 (일반 지식 기반, 확인 필요):**

7. **Liang, J. et al. (2020). "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation."** ICML 2020. *(SHOT)*
8. **Xu, T. et al. (2022). "CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation."** ICLR 2022.
9. **Li, R. et al. (2021). "Cross-domain contrastive learning for unsupervised domain adaptation."** CVPR 2021. *(CDAC)*

> **⚠️ 주의:** 2020년 이후 비교 연구의 구체적인 수치(특히 Table의 정확도)는 본 PDF에 포함되지 않은 내용으로, 일반 지식에 기반한 대략적인 비교입니다. 정확한 수치는 해당 논문을 직접 참조하시기를 권장합니다.
