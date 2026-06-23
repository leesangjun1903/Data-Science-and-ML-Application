# Curriculum based Dropout Discriminator for Domain Adaptation (CD³A)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 적대적 도메인 적응(Adversarial Domain Adaptation)에서 사용되는 **단일 판별자(point estimate discriminator)** 는 과신(overconfident)된 추론을 생성하여 도메인 불변 특징 학습을 방해할 수 있다. 이를 해결하기 위해 **Monte Carlo(MC) Dropout 기반 앙상블 판별자**를 활용하여 **분포 기반(distribution-based) 판별**을 수행하는 방법을 제안한다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **MC Dropout 판별자** | 파라미터 수 증가 없이 앙상블 판별 효과 획득 |
| **CD³A** | 커리큘럼 방식으로 MC 샘플 수를 점진적으로 증가 |
| **D³A** | 고정된 MC 샘플 수를 사용하는 변형 모델 |
| **확장성(Scalability)** | 클래스 수에 무관하게 파라미터 수 일정 유지 |
| **실험적 분석** | 통계적 유의성, Proxy-A 거리, t-SNE 시각화 포함 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)** 문제:

- 소스 도메인 $\mathcal{D}_s = \{(x_i^s, y_i^s)\}\_{i=1}^{N_s} \in \mathcal{S}$ : 레이블 있음
- 타깃 도메인 $\mathcal{D}_t = \{(x_i^t)\}\_{i=1}^{N_t} \in \mathcal{T}$ : 레이블 없음

**기존 방법의 한계:**

1. **GRL (DANN)**: 단일 이진 판별자 → 단일 점 추정(point estimate) → 과신된 그래디언트
2. **MADA**: 클래스별 판별자 → 클래스 수 $K$에 비례하여 파라미터 증가 ( $\sim$ 98M for Office-31, 클래스당 $\sim$ 1.3M 추가)
3. 두 방법 모두 다중 모달 데이터 분포를 충분히 반영하지 못함

---

### 2.2 제안 방법 및 수식

#### 전체 손실 함수

$$\mathcal{L}(\theta_f, \theta_c, \theta_d) = \frac{1}{N_s}\sum_{x_i \in \mathcal{D}_s} \mathcal{L}_c(C(f(x_i)), y_i) - \frac{\lambda}{N}\sum_{j=0}^{K}\sum_{x_i \in \mathcal{D}_s \cup \mathcal{D}_t} \mathcal{L}_d(D_j(f(x_i)), d_i) \tag{1}$$

$$(\hat{\theta}_f, \hat{\theta}_c) = \arg\min_{\theta_f, \theta_c} \mathcal{L}(\theta_f, \theta_c, \hat{\theta}_d), \quad \hat{\theta}_d = \arg\min_{\theta_d} \mathcal{L}(\hat{\theta}_f, \hat{\theta}_c, \theta_d) \tag{2}$$

**변수 설명:**

- $\theta_f$: 특징 추출기(Feature Extractor) 파라미터
- $\theta_c$: 분류기(Classifier) 파라미터
- $\theta_d$: MC Dropout 판별자(Discriminator) 파라미터
- $\mathcal{L}_c$: 분류 손실 (Cross-Entropy)
- $\mathcal{L}_d$: 도메인 분류 손실 (Binary Cross-Entropy)
- $D_j$: $j$번째 MC 샘플링된 Dropout 판별자
- $K$: MC 샘플 수 (CD³A에서 훈련 진행에 따라 증가)
- $d_i = 0$ if $x_i \in \mathcal{D}_s$, $d_i = 1$ if $x_i \in \mathcal{D}_t$
- $\lambda$: 두 목적함수 간의 균형 파라미터

#### Proxy-A Distance (도메인 차이 측정)

$$d_{\mathcal{A}} = 2(1 - 2\varepsilon) \tag{3}$$

여기서 $\varepsilon$은 소스/타깃을 이진 분류하는 커널 SVM의 일반화 오류.  
$d_{\mathcal{A}}$가 작을수록 도메인 간 간격이 작음을 의미함.

#### 커리큘럼 기반 판별자 집합

$$\mathcal{D} = \{\{D_1\}, \{D_1, D_2\}, \ldots, \{D_1, D_2, \ldots, D_K\}\} \tag{4}$$

여기서 $\{D_1\} \subseteq \{D_1, D_2\}$ 관계로 판별자 집합의 역량(capacity)이 단조증가.

---

### 2.3 모델 구조

```
소스 도메인 Ds ──→ [공유 특징 추출기 f] ──→ [분류기 C] ──→ 분류 손실 Lc
                           │
                     (Reverse Gradient)
                           │
타깃 도메인 Dt ──→ [공유 특징 추출기 f] ──→ [MC Dropout 판별자]
                                              ├── D₁(f(x)) ──→ p₁(d|X)
                                              ├── D₂(f(x)) ──→ p₂(d|X)
                                              └── Dₖ(f(x)) ──→ pₖ(d|X)
                                              └── 도메인 손실 Ld (합산)
```

**구성 요소:**

| 구성요소 | 역할 | 업데이트 신호 |
|----------|------|--------------|
| 특징 추출기 $f$ | 공유 특징 생성 | 분류기 그래디언트 + **역방향 판별자 그래디언트 분포** |
| 분류기 $C$ | 소스 레이블 예측 | 소스 분류 손실 |
| MC Dropout 판별자 $D_j$ | 도메인 식별 | 도메인 분류 손실 |

**Bernoulli Dropout** 확률 $d$로 판별자 뉴런을 무작위 제거 → 각 포워드 패스마다 다른 서브네트워크(discriminator) 생성 → 앙상블 효과

---

### 2.4 성능 향상

#### Office-31 (AlexNet 기반)

| Method | A→W | D→W | W→D | A→D | D→A | W→A | Avg |
|--------|-----|-----|-----|-----|-----|-----|-----|
| GRL (DANN) | 73.0 | 96.4 | 99.2 | 72.3 | 52.4 | 50.4 | 74.1 |
| MADA | 78.5 | **99.8** | **100.0** | 74.1 | 56.0 | 54.5 | 77.1 |
| D³A(31) | 79.0 | 97.7 | **100.0** | 79.4 | 58.2 | 55.3 | 78.3 |
| **CD³A** | **82.3** | **99.8** | **100.0** | 81.1 | **58.2** | **55.6** | **79.5** |

- MADA 대비 A→W에서 **+3.8%**, A→D에서 **+7.0%** 향상
- GRL 대비 평균 **+5.4%** 향상

#### ImageCLEF (ResNet-50 기반)

| Method | I→P | P→I | I→C | C→I | C→P | P→C | Avg |
|--------|-----|-----|-----|-----|-----|-----|-----|
| CDAN | 77.2 | 88.3 | 98.3 | 90.7 | 76.7 | 94.0 | 87.5 |
| **CD³A** | **77.5** | **88.7** | 96.8 | **93.2** | **78.3** | **94.7** | **88.2** |

#### 파라미터 효율성

$$\text{MADA (Office-31)}: \sim 98\text{M 파라미터}$$
$$\text{CD}^3\text{A}: \sim 59\text{M 파라미터 (클래스 수와 무관하게 일정)}$$

---

### 2.5 한계

1. **하이퍼파라미터 민감성**: MC 샘플 수 $K$의 증가 스케줄 및 Dropout 확률 $d$를 수동으로 조정해야 함
2. **AlexNet 기반 평가 중심**: 논문 주요 실험이 AlexNet 기반이며, 최신 Transformer 기반 백본에 대한 검증 부재
3. **이론적 수렴 보장 부재**: 커리큘럼 방식의 MC 샘플 증가 전략에 대한 이론적 수렴 분석이 없음
4. **소스 레이블 의존성**: 완전한 비지도 설정에서 소스 레이블에 여전히 의존
5. **부분 도메인 적응(Partial DA)**: 타깃에 소스보다 적은 클래스가 존재하는 시나리오에 대한 검증 미비

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상 메커니즘

#### (1) 분포 기반 그래디언트

단일 판별자는 $\nabla_{\theta_f} \mathcal{L}_d$를 하나의 점 추정값으로 제공하지만, CD³A는 다음과 같이 그래디언트의 분포를 제공:

$$\mathbb{E}_{j \sim \text{Dropout}}\left[\nabla_{\theta_f} \mathcal{L}_d(D_j(f(x)), d)\right]$$

이 분포 기반 그래디언트는 특징 추출기가 **특정 판별자 구조에 과적합되는 것을 방지**하고, 진정한 도메인 불변 표현을 학습하도록 강제한다.

#### (2) Dropout의 정규화 효과

Gal & Ghahramani(2016)의 이론적 근거: Dropout은 **베이지안 근사(Bayesian Approximation)**로 해석될 수 있으며, 이는 예측 불확실성을 모델링하는 효과를 가짐.

$$p(y | x, \mathcal{D}) \approx \frac{1}{K}\sum_{j=1}^{K} p(y | x, \hat{\theta}_d^{(j)})$$

여기서 $\hat{\theta}_d^{(j)}$는 $j$번째 Dropout 마스크를 적용한 판별자 파라미터.

#### (3) 커리큘럼 학습의 일반화 기여

$$K(t) = \lceil K_{\max} \cdot \frac{t}{T} \rceil \quad (\text{훈련 단계 } t, \text{ 최대 단계 } T)$$

초기에는 단순한 판별자(작은 $K$)로 기본적인 도메인 불변 특징을 학습하고, 후반부에 복잡한 앙상블로 다중 모달 구조까지 학습함으로써 **학습 안정성과 최종 성능을 동시에 향상**.

#### (4) Proxy-A Distance 감소

실험 결과 CD³A는 GRL, MADA 대비 A→D, A→W 태스크에서 $d_\mathcal{A}$가 현저히 낮음 → **도메인 간 격차를 더 효과적으로 줄임** → 타깃 도메인에서의 일반화 성능 직접 향상.

#### (5) t-SNE 시각화

A→W 태스크에서 CD³A로 적응된 특징이 GRL 대비 소스/타깃 도메인이 더 혼합됨 → 학습된 표현의 도메인 불변성이 시각적으로도 확인됨.

#### (6) 데이터 효율성

보충 자료에서 소스 데이터의 절반만 사용해도 좋은 성능 유지 → 과적합 저감으로 **데이터 효율적 일반화** 달성.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

#### (1) 불확실성 인식 도메인 적응의 촉진
CD³A는 판별자에 **베이지안 불확실성 추정**을 도입한 선구적 시도로, 이후 연구들이 적응 과정의 불확실성을 명시적으로 다루는 방향을 열었음.

#### (2) 파라미터 효율적 앙상블의 표준화
클래스 수에 무관한 일정 파라미터 수라는 설계 원칙은 이후 대규모 범주 도메인 적응 연구에 중요한 기준점 제공.

#### (3) 커리큘럼 + 적대적 학습의 결합 가능성
도메인 적응에서 커리큘럼 방식을 판별자 역량 증가에 적용한 아이디어는 이후 **자기 훈련(Self-Training)**, **의사 레이블링(Pseudo-Labeling)** 과 결합된 커리큘럼 방법론 연구에 영향.

#### (4) Vision Transformer 시대의 적용 가능성
ViT 기반 특징 추출기와의 결합 시, MC Dropout의 적용 방식(어텐션 레이어 vs. MLP 헤드)에 대한 새로운 연구 방향 제시.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 핵심 방법 | CD³A와의 관계 | 주요 차이점 |
|------|------|----------|--------------|------------|
| **SHOT** (Liang et al., ICML 2020) | 2020 | 소스 없는 DA; 정보 극대화 + 의사 레이블 | 판별자 대신 정보이론적 손실 사용 | 소스 데이터 불필요; 확장성 더 높음 |
| **CDTrans** (Xu et al., ECCV 2022) | 2022 | Cross-attention Transformer for DA | ViT 백본 도입 | 어텐션 기반 도메인 정렬 |
| **TVT** (Yang et al., TMLR 2023) | 2023 | Transferable Vision Transformer | Transformer의 전이 가능성 활용 | 사전학습 모델의 fine-tuning 활용 |
| **DeiT-based DA** (Sun et al., 2022) | 2022 | ViT + 도메인 적응 | Transformer 특징 추출기 | CD³A의 CNN 백본과 대조적 |
| **ToAlign** (Wei et al., NeurIPS 2021) | 2021 | 태스크 지향 특징 정렬 | 분류 손실 기반 정렬 강화 | 도메인 판별자 없이 정렬 |

#### 세부 비교

**SHOT (Liang et al., ICML 2020)**:
- CD³A가 판별자의 강건성에 집중한 반면, SHOT은 **소스 데이터 없이** 정보 극대화와 의사 레이블로 적응
- 실용성(소스 데이터 프라이버시 보호) 측면에서 더 앞선 방향
- 그러나 CD³A의 불확실성 기반 앙상블 아이디어는 SHOT의 신뢰도 기반 의사 레이블과 개념적으로 연결됨

**Transformer 기반 방법들 (2022~2023)**:
- CD³A는 AlexNet/ResNet 기반으로, ViT 시대에는 백본 업데이트가 필요
- CD³A의 MC Dropout 판별자 아이디어는 Transformer의 어텐션 드롭아웃과 결합 가능성 존재

---

### 4.3 앞으로 연구 시 고려할 점

#### (1) 최신 백본과의 통합
```
고려사항: ViT, Swin Transformer 등에서 Dropout의 위치
→ Patch Embedding Layer vs. MLP Head vs. Attention Layer
→ 각 위치에서의 앙상블 효과 차이 분석 필요
```

#### (2) 소스-프리(Source-Free) 설정으로의 확장

소스 데이터 없이 MC Dropout 판별자를 어떻게 활용할 것인가:

$$\min_{\theta_f} \mathcal{H}(C(f(x_t))) - \mathbb{E}_j[\mathcal{H}(D_j(f(x_t)))]$$

불확실성 최소화와 도메인 불변성을 동시에 달성하는 새로운 손실 함수 설계 연구 필요.

#### (3) 커리큘럼 스케줄의 자동화

현재 MC 샘플 증가 스케줄은 수동 설계. 강화학습 또는 메타러닝 기반 **자동 커리큘럼 스케줄링** 연구 필요:

$$K^*(t) = \arg\max_{K} \mathbb{E}[\text{Target Accuracy}(t) | K]$$

#### (4) 오픈셋 도메인 적응으로의 확장

타깃 도메인에 소스에 없는 클래스가 존재하는 **오픈셋(Open-Set) DA** 시나리오에서 MC Dropout 판별자의 불확실성을 미지 클래스 탐지에 활용 가능.

#### (5) 멀티소스 도메인 적응

여러 소스 도메인에서 각각의 MC Dropout 판별자 분포를 학습하고 이를 혼합하는 방향:

$$\mathcal{L} = \sum_{s \in \mathcal{S}} w_s \cdot \frac{1}{K}\sum_{j=1}^{K} \mathcal{L}_d^{(s)}(D_j(f(x)), d)$$

#### (6) 이론적 보장 강화

벤-데이비드(Ben-David) 이론 프레임워크와 연계하여, MC Dropout 앙상블이 타깃 위험 상한(target risk bound)을 어떻게 줄이는지 이론적 증명 필요:

$$\epsilon_T(h) \leq \epsilon_S(h) + d_\mathcal{A}(\mathcal{D}_S, \mathcal{D}_T) + \lambda^*$$

CD³A가 $d_\mathcal{A}$를 줄인다는 실험적 증거는 있으나, MC 샘플 수 $K$와 $d_\mathcal{A}$ 감소량의 이론적 관계 규명 필요.

---

## 참고 자료

**주요 논문:**
- **Kurmi et al. (2019)**: "Curriculum based Dropout Discriminator for Domain Adaptation," arXiv:1907.10628v2
- **Ganin & Lempitsky (2015)**: "Unsupervised domain adaptation by backpropagation," ICML
- **Pei et al. (2018)**: "Multi-adversarial domain adaptation," AAAI (MADA)
- **Gal & Ghahramani (2016)**: "Dropout as a Bayesian approximation," ICML
- **Long et al. (2018)**: "Conditional adversarial domain adaptation," NeurIPS (CDAN)
- **Liang et al. (2020)**: "Do we really need to access the source data? Source hypothesis transfer for unsupervised domain adaptation," ICML (SHOT)
- **Ben-David et al. (2010)**: "A theory of learning from different domains," Machine Learning
- **Bengio et al. (2009)**: "Curriculum learning," ICML
- **Hara et al. (2016)**: "Analysis of dropout learning regarded as ensemble learning," ICANN

**arXiv 링크:** https://arxiv.org/abs/1907.10628

> **정확도 주의사항:** 2020년 이후 최신 연구 비교 분석 부분(SHOT, CDTrans, TVT 등)은 논문 원문에 포함되지 않은 내용으로, 해당 논문들의 일반적으로 알려진 내용을 기반으로 작성되었습니다. 구체적인 수치 비교는 각 논문의 원문을 직접 확인하시기 바랍니다.
