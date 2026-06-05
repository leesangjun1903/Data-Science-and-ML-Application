# Curriculum Loss: Robust Learning and Generalization against Label Corruption 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문(Lyu & Tsang, ICLR 2020)의 핵심 주장은 다음과 같습니다:

> **딥 신경망(DNN)이 잘못된 레이블(noisy label)까지 암기(memorize)하는 문제를 해결하기 위해, 0-1 손실(0-1 loss)의 강건한 성질을 보존하면서도 효율적으로 최적화 가능한 새로운 손실 함수 "Curriculum Loss(CL)"를 제안한다.**

CL은 기존의 합산(summation) 기반 서로게이트 손실보다 0-1 손실에 더 **타이트한 상한(tighter upper bound)**을 제공하며, 이를 통해 학습 과정에서 **노이즈 샘플을 자동으로 선별(curriculum learning 패러다임)**한다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **이론적 기여 1** | CL이 기존 서로게이트 손실보다 0-1 손실의 더 타이트한 상한임을 수학적으로 증명 |
| **이론적 기여 2** | $\mathcal{O}(n \log n)$ 시간 복잡도의 효율적 부분 최적화 알고리즘 제시 |
| **이론적 기여 3** | 커리큘럼 학습과 강건 학습(robust learning)의 이론적 연결고리 확립 |
| **실용적 기여 1** | 미니배치 기반 업데이트 지원으로 기존 딥러닝 프레임워크에 플러그인 형태로 적용 가능 |
| **실용적 기여 2** | 대규모 노이즈 비율을 처리하는 확장판 **Noise Pruned Curriculum Loss(NPCL)** 제안 |
| **실험적 기여** | MNIST, CIFAR-10, CIFAR-100, Tiny-ImageNet에서 GCE, Co-teaching 등 대비 우수한 성능 검증 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**핵심 문제: 레이블 오염(Label Corruption) 하에서의 DNN 학습**

Zhang et al. (2017)이 보였듯이, DNN은 완전히 잘못된 레이블까지 암기할 수 있는 표현력을 보유합니다. 이는 다음과 같은 두 가지 부문제를 야기합니다:

1. **소규모 노이즈 비율**: 0-1 손실이 이론적으로 강건하지만, 비미분성(non-differentiability)과 거의 모든 곳에서 0인 기울기 문제로 최적화 불가
2. **대규모 노이즈 비율**: 체계적 오류(systematic error)가 커져 일반화 성능이 급격히 저하

기존 방법들의 한계:
- **전이행렬 기반(Transition matrix)**: 노이즈 구조에 대한 강한 가정 필요
- **정규화 기반**: 충분하지 않은 강건성
- **샘플 선택 기반(Co-teaching 등)**: 휴리스틱에 의존, 두 네트워크 필요로 계산 비용 2배

### 2-2. 제안하는 방법 (수식 포함)

#### Step 1: 0-1 손실의 강건성 이론적 기반

논문은 Hu et al. (2018)의 **단조 관계 정리(Monotonic Relationship Theorem)**를 이론적 근거로 활용합니다.

$$\mathcal{R}(\theta) = \mathbb{E}_{p(x,y)}[l(g_\theta(x), y)] $$

$$\widehat{\mathcal{R}}(\theta) = \frac{1}{n}\sum_{i=1}^{n} l(g_\theta(x_i), y_i) $$

$$\mathcal{R}_{adv}(\theta) = \sup_{r \in \mathcal{U}_f} \mathbb{E}_{p(x,y)}[r(x,y)l(g_\theta(x), y)] $$

$$\widehat{\mathcal{R}}_{adv}(\theta) = \sup_{r \in \widehat{\mathcal{U}}_f} \frac{1}{n}\sum_{i=1}^{n} r_i l(g_\theta(x_i), y_i) $$

여기서:
- $p(x,y)$: 오염된 훈련 분포
- $q(x,y)$: 정제된 테스트 분포  
- $r(x,y) = q(x,y)/p(x,y)$: 밀도 비율
- $\mathcal{U}_f$: $f$-발산이 $\delta$로 제한된 분포 집합

**핵심 정리**: 0-1 손실 사용 시,

$$\text{If } \mathcal{R}_{adv}(\theta_1) < 1, \text{ then } \mathcal{R}(\theta_1) < \mathcal{R}(\theta_2) \iff \mathcal{R}_{adv}(\theta_1) < \mathcal{R}_{adv}(\theta_2) $$

즉, **0-1 손실로 경험적 위험을 최소화하는 것 = 최악의 경우(adversarial) 위험을 최소화하는 것**

#### Step 2: 0-1 손실의 수식 정의

이진 분류에서 분류 마진(classification margin)을 $u_i = \hat{y}_i y_i$ ($\hat{y}_i$: 예측, $y_i \in \{+1,-1\}$: 실제 레이블)로 정의할 때:

$$J(\mathbf{u}) = \sum_{i=1}^{n} \mathbf{1}(u_i < 0) $$

기존 서로게이트 손실 (합산 기반):

$$\widehat{J}(\mathbf{u}) = \sum_{i=1}^{n} l(u_i), \quad \text{where } l(u) \geq \mathbf{1}(u < 0) $$

#### Step 3: Curriculum Loss (CL) 정의

**[Theorem 2: Tighter Bound]**

$$Q(\mathbf{u}) = \min_{\mathbf{v} \in \{0,1\}^n} \max\left(\sum_{i=1}^{n} v_i l(u_i),\ n - \sum_{i=1}^{n} v_i + \sum_{i=1}^{n} \mathbf{1}(u_i < 0)\right) $$

이때 다음이 성립합니다:

$$J(\mathbf{u}) \leq Q(\mathbf{u}) \leq \widehat{J}(\mathbf{u})$$

여기서 $\mathbf{v} = [v_1, \ldots, v_n] \in \{0,1\}^n$은 **샘플 선택 인디케이터 벡터**입니다. $v_i = 1$이면 $i$번째 샘플이 학습에 선택됩니다.

#### Step 4: 미니배치 버전 (Corollary 1)

$$\widehat{Q}(\mathbf{u}) = \sum_{j=1}^{b} \min_{\mathbf{v} \in \{0,1\}^m} \max\left(\sum_{i=1}^{m} v_{ij} l(u_{ij}),\ m - \sum_{i=1}^{m} v_{ij} + \sum_{i=1}^{m} \mathbf{1}(u_{ij} < 0)\right) $$

$$J(\mathbf{u}) \leq Q(\mathbf{u}) \leq \widehat{Q}(\mathbf{u}) \leq \widehat{J}(\mathbf{u})$$

#### Step 5: 스케일드 상한 버전 (Theorem 3)

$$E(\mathbf{u}) = \min_{\mathbf{v} \in \{0,1\}^n} \max\left(\sum_{i=1}^{n} v_i l(u_i),\ n - \sum_{i=1}^{n} v_i\right) $$

$$J(\mathbf{u}) \leq 2E(\mathbf{u}) \leq 2\widehat{J}(\mathbf{u})$$

$E(\mathbf{u}) \leq Q(\mathbf{u})$이므로 이상값(outlier)에 더 둔감하지만, $Q(\mathbf{u})$가 학습 단계별 적응적 커리큘럼 구성에 더 유리합니다.

#### Step 6: 부분 최적화 알고리즘 (Algorithm 1)

다음의 부분 최적화 문제를 $\mathcal{O}(n \log n)$에 해결합니다:

$$\min_{\mathbf{v} \in \{0,1\}^n} \max\left(\sum_{i=1}^{n} v_i l(u_i),\ C - \sum_{i=1}^{n} v_i\right) $$

**알고리즘**: 손실 $l(u_i)$를 오름차순 정렬 후 누적합 $L_i$를 계산하여, $L_i \leq (C+1-i)$인 경우 $v_i = 1$ (선택), 아니면 $v_i = 0$ (제외).

최적해의 특성 (Proposition 1):

```math
L_{T^*} \leq C + 1 - T^*
```

```math
L_{T^*+1} > C - T^*
```

```math
\min_{\mathbf{v}} \max\left(\sum_{i=1}^n v_i l(u_i), C - \sum_{i=1}^n v_i\right) = \max(L_{T^*}, C - T^*)
```

여기서 $T^\* = \sum_{i=1}^n v\_i^\*$ (선택된 샘플 수), $L_{T^\*} = \sum_{i=1}^{T^*} l(u_i)$ (선택 샘플의 손실 합).

#### Step 7: Noise Pruned Curriculum Loss (NPCL)

대규모 노이즈 비율 $\epsilon \in [0,1]$을 처리하기 위한 확장:

$$\mathcal{L}(\mathbf{u}) = \min_{\mathbf{v} \in \{0,1\}^n} \max\left(\sum_{i=1}^{n} v_i l(u_i),\ C - \sum_{i=1}^{n} v_i\right) $$

임계값 $C$를 다음 두 가지로 설정:

$$C = (1-\epsilon)n \quad \text{또는} \quad C = (1-\epsilon)^2 n + (1-\epsilon)\sum_{i=1}^{n}\mathbf{1}(u_i < 0) $$

$C = (1-\epsilon)n$ 설정 시, Algorithm 1은 자동으로 가장 큰 손실을 가진 $\epsilon n$개 샘플을 제거합니다. 이후 나머지 $(1-\epsilon)n$ 샘플에 대해 기본 CL을 적용합니다:

$$\widetilde{\mathcal{L}}(\mathbf{u}) = \min_{\mathbf{v} \in \{0,1\}^{(1-\epsilon)n}} \max\left(\sum_{i=1}^{(1-\epsilon)n} v_i l(u_i),\ (1-\epsilon)n - \sum_{i=1}^{(1-\epsilon)n} v_i\right) $$

배치 기반 NPCL:

$$\widehat{\mathcal{L}}(\mathbf{u}) = \sum_{j=1}^{b} \min_{\mathbf{v} \in \{0,1\}^m} \max\left(\sum_{i=1}^{m} v_{ij} l(u_{ij}),\ \hat{C}_j - \sum_{i=1}^{m} v_{ij}\right) $$

$$\hat{C}_j = (1-\epsilon)^2 m + (1-\epsilon)\sum_{i=1}^{m} \mathbf{1}(u_{ij} < 0) $$

#### 멀티클래스 확장

멀티클래스의 경우 분류 마진은:

$$u = t_y - \max_{i \neq y} t_i $$

소프트 멀티클래스 힌지 손실:

$$S(\mathbf{t}, y) = \begin{cases} \max(1 - t_y + \max_{i \neq y} t_i,\ 0), & t_y - \max_{i \neq y} t_i \geq 0 \\ \max(1 - t_y + \text{LogSumExp}(\mathbf{t}),\ 0), & t_y - \max_{i \neq y} t_i < 0 \end{cases} $$

### 2-3. 모델 구조

논문은 새로운 네트워크 아키텍처를 제안하지 않고, **손실 함수 자체를 기여**로 합니다. 실험에서 사용된 네트워크:

| 데이터셋 | 아키텍처 |
|----------|----------|
| MNIST | CNN (단순 구조) |
| CIFAR-10/100 | 9층 CNN (LReLU, Max-pooling, Dropout) |
| CIFAR-10/100 (추가 실험) | DenseNet |
| Tiny-ImageNet | ResNet-18 |

**학습 설정**:
- Batch size: $m = 128$
- Epochs: $N = 200$
- Optimizer: Adam (Co-teaching과 동일한 하이퍼파라미터)
- Base loss: Hinge loss
- Burn-in period: MNIST 5 epoch, CIFAR 10 epoch (전체 배치로 소프트 힌지 손실 사용)

### 2-4. 성능 향상

#### MNIST 결과 (Table 5)

| 방법 | Sym-20% | Sym-50% | Pair-35% |
|------|---------|---------|---------|
| Standard | 93.78% | 65.81% | 70.50% |
| MentorNet | 96.68% | 90.53% | 89.62% |
| Co-teaching | 97.14% | 91.35% | 90.96% |
| Co-teaching+ | **99.41%** | 97.79% | 93.81% |
| GCE | 99.40% | 92.48% | 72.26% |
| **NPCL** | **99.41%** | **98.53%** | **97.90%** |

#### CIFAR-10 결과 (Table 6)

| 방법 | Sym-20% | Sym-50% | Pair-35% |
|------|---------|---------|---------|
| Standard | 76.62% | 49.92% | 62.26% |
| GCE | **84.68%** | 61.80% | 60.86% |
| Co-teaching | 82.13% | 74.28% | **77.77%** |
| **NPCL** | 84.30% | **77.66%** | 76.52% |

#### 주요 관찰:
- **노이즈 비율이 클수록(50%) NPCL의 우위가 더 뚜렷함**
- Co-teaching 대비 단일 네트워크만 사용하여 **공간/시간 복잡도 절반**
- GCE보다 대규모 노이즈에서 현저히 우수

### 2-5. 한계점

1. **노이즈 비율 $\epsilon$의 사전 지식 필요**: NPCL은 노이즈 비율을 알고 있어야 하며, 잘못 추정 시 성능이 저하됨 (특히 대규모 노이즈 Sym-50%에서 민감)
2. **불균형 분포 미처리**: 클래스 불균형 데이터에 대한 처리 불충분 (저자들도 향후 과제로 언급)
3. **비대칭 노이즈(Asymmetric noise)**: Pair-35%에서 Co-teaching 대비 소폭 낮은 성능
4. **노이즈 유형 가정**: 주로 무작위 레이블 오염을 가정하며, 구조적(structured) 노이즈에 대한 분석 부족
5. **이론적 수렴 보장 부재**: 딥러닝 환경에서의 이론적 수렴 보장을 제공하지 않음
6. **Burn-in period 설정**: 초기 에폭의 burn-in 기간 설정이 휴리스틱에 의존

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 0-1 손실과 Adversarial Risk의 단조 관계를 통한 일반화

일반화 성능 향상의 이론적 근거는 Hu et al. (2018)의 단조 관계에서 출발합니다. 오염된 훈련 분포 $p(x,y)$와 정제된 테스트 분포 $q(x,y)$ 사이의 $f$-발산이 $\delta$로 제한될 때:

$$G(\theta) := \sup_{q \in \widetilde{\mathcal{U}}_f} \mathbb{E}_{q(x,y)}[l(g_\theta(x), y)] $$

0-1 손실 사용 시, 이 worst-case risk를 최소화하는 것이 오염 분포 하에서의 경험적 위험 최소화와 **동치**입니다:

$$\widetilde{G}(\theta) := \mathbb{E}_{p(x,y)}[l(g_\theta(x), y)] $$

이는 **$\delta$를 사전에 알지 못해도** $\widetilde{G}(\theta)$를 최소화함으로써 자동으로 worst-case risk의 상한을 최소화함을 의미합니다.

### 3-2. Tighter Upper Bound가 일반화에 미치는 영향

$$J(\mathbf{u}) \leq Q(\mathbf{u}) \leq \widehat{J}(\mathbf{u})$$

CL이 기존 서로게이트 손실보다 0-1 손실에 더 가까운 상한을 제공한다는 것은:

- **노이즈 샘플(outlier)에 대한 과도한 가중치 부여 방지**: 기존 unbounded convex loss는 큰 손실값을 가진 노이즈 샘플에 큰 가중치를 부여하여 모델 파라미터를 왜곡시킴
- **최소화하는 대상이 실제 분류 오류율에 더 가까움**: 이는 최적화 목표와 실제 평가 지표(테스트 정확도)의 간극(surrogate gap)을 줄임

### 3-3. 커리큘럼 학습을 통한 일반화

Arpit et al. (2017)의 **memorization effect**에 따르면, DNN은 초기에 쉽고 정제된 패턴을 먼저 학습합니다. NPCL은 이를 활용하여:

1. **초기 학습 단계**: 큰 손실을 가진 노이즈 샘플을 제거 → 정제된 샘플로만 학습
2. **후기 학습 단계**: 모델이 성숙해짐에 따라 더 어려운 샘플도 선택적으로 학습

이는 점진적 커리큘럼을 통해 모델이 노이즈에 과적합되는 것을 방지하고, **더 나은 일반화 경계(generalization bound)**를 달성하게 합니다.

### 3-4. 레이블 오염과 일반 오염의 관계

레이블 오염은 일반 오염의 특수 케이스입니다. $q(x) = p(x)$ 조건 하에서:

$$G_y(\theta) := \sup_{q \in \widetilde{\mathcal{U}}_f \cap H} \mathbb{E}_{q(x,y)}[l(g_\theta(x), y)] \leq G(\theta) $$

즉, CL은 레이블 오염뿐만 아니라 **특징 오염(feature corruption)**에도 이론적으로 강건성을 보장합니다.

### 3-5. 실험적 일반화 성능 증거

- **노이즈 비율이 높을수록 일반화 격차가 커짐**: Symmetry-50%에서 NPCL이 GCE 대비 CIFAR-10에서 약 16%p 향상
- **Label Precision 지표**: NPCL은 선택된 미니배치 내 정제 샘플 비율을 높게 유지하여, 각 gradient update가 정제 레이블에 기반함
- **다양한 네트워크 아키텍처에서 일관된 개선**: DenseNet, ResNet-18 등에서도 동일한 경향

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4-1. 연구에 미치는 영향

#### 이론적 영향

1. **손실 함수 설계 패러다임 전환**: 개별 샘플 수준의 손실이 아닌, **샘플 집합 수준(set-level)의 손실** 설계로의 방향 제시
2. **커리큘럼 학습과 강건 학습의 통합 프레임워크**: 두 분야를 하나의 최적화 목표로 통합하는 가능성 제시
3. **Distributionally Robust Optimization(DRO)와의 연결**: 0-1 손실의 adversarial risk 단조 관계는 DRO 이론과의 접점을 형성

#### 실용적 영향

1. **단일 네트워크 플러그인**: Co-teaching 계열의 이중 네트워크 방식 대비 계산 효율성 측면에서 실용적 대안 제공
2. **레이블 노이즈 처리의 표준 도구로서의 가능성**: 간단한 구현으로 다양한 태스크에 적용 가능
3. **자동 샘플 선택**: 별도의 네트워크나 복잡한 메타러닝 없이 학습 중 자동으로 노이즈 샘플 식별

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의**: 아래 연구들은 본 논문 이후 발표된 관련 연구들이나, 제가 직접 접근한 문서가 아닌 제 학습 데이터 기반 정보입니다. 논문 제목과 핵심 내용의 정확성에 대해 주의를 요합니다. 가능하면 원문을 직접 확인하시기 바랍니다.

| 연구 | 주요 방법 | CL과의 비교 |
|------|----------|------------|
| **DivideMix** (Li et al., ICLR 2020) | GMM으로 정제/노이즈 샘플 분리 후 반지도학습 | 더 복잡한 파이프라인, 더 높은 성능 |
| **ELR (Early-Learning Regularization)** (Liu et al., NeurIPS 2020) | 초기 학습 예측을 정규화 항으로 활용 | 단일 네트워크, 암묵적 샘플 선택 |
| **CORES²** (Cheng et al., ICML 2021) | 신뢰도 점수 기반 샘플 선택 | 이론적 보장 강화 |
| **Sel-CL** (Li et al., CVPR 2022) | 대조 학습(contrastive learning) + 샘플 선택 | 자기지도 학습과의 결합 |
| **SOP** (Liu et al., ICML 2022) | Stochastic re-labeling + over-parameterization | 레이블 복원까지 수행 |

**주요 트렌드 비교**:

```
CL/NPCL (2020):
  - 단일 네트워크
  - 손실 함수 수준의 샘플 선택
  - 0-1 손실의 이론적 근거
  - 노이즈 비율 사전 지식 필요

후속 연구 트렌드:
  - 반지도/자기지도 학습과의 결합
  - 레이블 복원(label correction) 병행
  - 더 복잡한 노이즈 모델 처리
  - 노이즈 비율 자동 추정
```

### 4-3. 향후 연구 시 고려할 점

#### 방법론적 고려사항

1. **노이즈 비율 자동 추정 통합**
   - 현재 NPCL은 $\epsilon$을 사전에 알아야 하므로, GMM이나 베이지안 방법으로 $\epsilon$을 동적으로 추정하는 메커니즘 통합 필요
   
2. **클래스 불균형 처리**
   - 저자들이 직접 언급한 향후 과제로, 각 클래스별 다양성(diversity)을 고려한 CL 확장 필요
   - 클래스별 임계값 $C_k$를 다르게 설정하는 방향 고려

3. **비대칭 노이즈(Instance-dependent noise) 처리**
   - 현재는 주로 무작위 노이즈를 가정하나, 실제 환경에서는 특정 샘플의 특징에 따라 노이즈가 발생하는 인스턴스 의존적 노이즈가 더 일반적

4. **대조 학습(Contrastive Learning)과의 결합**
   - 자기지도 학습 기반의 표현 학습과 CL을 결합하면 샘플 선택의 품질을 높일 수 있음
   - 정제 샘플과 노이즈 샘플의 표현 공간상 분리를 활용

5. **레이블 복원(Label Correction) 기능 추가**
   - CL은 노이즈 샘플을 제거하지만, 일부 노이즈 샘플은 올바른 레이블로 복원하여 활용하면 데이터 효율성 향상 가능

6. **이론적 수렴 분석**
   - 딥러닝 환경에서의 CL 수렴 속도와 일반화 오류 경계(generalization error bound)에 대한 이론적 분석 부재

7. **Foundation Model 시대의 적용**
   - 대규모 사전학습 모델(LLM, CLIP 등)의 파인튜닝 시 레이블 노이즈 처리에 CL 적용 가능성 탐색
   - 프롬프트 학습(prompt learning) 환경에서의 적용

8. **연합 학습(Federated Learning)과의 결합**
   - 분산 환경에서 각 클라이언트의 레이블 노이즈를 처리하는 데 CL 원리 적용 가능

#### 실험적 고려사항

1. **실제 세계 노이즈 데이터셋 평가**: Clothing1M, WebVision 등 실제 웹 크롤링 데이터에서의 검증 필요
2. **다양한 베이스 손실 함수 실험**: 저자들이 언급한 것처럼 Hinge loss 외 다른 베이스 손실 함수 영향 분석
3. **대규모 모델에서의 확장성**: 현재 실험은 상대적으로 작은 CNN에 국한되어 있어, 대규모 Transformer 모델에서의 검증 필요

---

## 참고 자료

**주요 논문 (첨부 문서 기반)**:
- **Lyu, Y. & Tsang, I. W. (2020).** "Curriculum Loss: Robust Learning and Generalization against Label Corruption." *ICLR 2020.* arXiv:1905.10045v3

**논문 내 인용 문헌**:
- Hu, W., Niu, G., Sato, I., & Sugiyama, M. (2018). "Does distributionally robust supervised learning give robust classifiers?"
- Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). "Understanding deep learning requires rethinking generalization." *ICLR 2017.*
- Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., Tsang, I., & Sugiyama, M. (2018). "Co-teaching: Robust training of deep neural networks with extremely noisy labels." *NeurIPS 2018.*
- Yu, X., Han, B., Yao, J., Niu, G., Tsang, I., & Sugiyama, M. (2019). "How does disagreement help generalization against label corruption?" *ICML 2019.*
- Zhang, Z. & Sabuncu, M. (2018). "Generalized cross entropy loss for training deep neural networks with noisy labels." *NeurIPS 2018.*
- Arpit, D., et al. (2017). "A closer look at memorization in deep networks." *ICML 2017.*
- Jiang, L., Zhou, Z., Leung, T., Li, L.-J., & Fei-Fei, L. (2018). "MentorNet: Learning data-driven curriculum for very deep neural networks on corrupted labels."
- Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). "Curriculum learning." *ICML 2009.*
- Masnadi-Shirazi, H. & Vasconcelos, N. (2009). "On the design of loss functions for classification." *NeurIPS 2009.*
- Lee, K., Yun, S., Lee, K., Lee, H., Li, B., & Shin, J. (2019). "Robust inference via generative classifiers for handling noisy labels." *ICML 2019.*
- Ma, X., et al. (2018). "Dimensionality-driven learning with noisy labels." *ICML 2018.*
