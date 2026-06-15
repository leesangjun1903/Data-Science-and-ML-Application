# Multi-Objective Interpolation Training for Robustness to Label Noise

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

딥러닝 모델은 표준 크로스 엔트로피 손실 함수로 학습 시 **노이즈 레이블을 암기(memorization)**하여 성능이 저하된다. 기존 연구 대부분이 새로운 강건한 분류 손실 함수 설계에 집중한 반면, 본 논문은 **대조 학습(Contrastive Learning)**과 **준지도 학습(Semi-Supervised Learning)**을 상호 보완적으로 결합하는 **MOIT(Multi-Objective Interpolation Training)** 프레임워크를 제안한다.

### 주요 기여 (4가지)

| 기여 | 내용 |
|------|------|
| **① MOIT 프레임워크** | 감독 대조 학습 + 준지도 분류를 단일 하이퍼파라미터로 공동 학습 |
| **② ICL 손실** | Mixup을 대조 학습에 적용하여 노이즈 레이블로 인한 표현 열화 방지 |
| **③ 노이즈 탐지 전략** | k-NN 기반 소프트 레이블로 노이즈 샘플을 정확히 식별 |
| **④ MOIT+ 정제** | 탐지된 정제 데이터로 파인튜닝하여 추가 성능 향상 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

딥 신경망은 표준 학습 시 노이즈 레이블을 암기하는 경향이 있다. 기존 연구들은 주로:
- 새로운 강건한 손실 함수 설계 (분류 관점만 고려)
- 노이즈 탐지 후 레이블 보정

에 집중하였으며, **유사도 학습 프레임워크(대조 학습)를 노이즈 레이블 환경에 적용한 연구는 거의 없었다.** 특히 감독 대조 학습(Supervised Contrastive Learning, SCL)이 노이즈 레이블 존재 시 성능이 열화된다는 문제를 처음으로 분석하고 해결책을 제시한다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 기본 감독 대조 학습 손실

인코더 $f_\theta$와 프로젝션 헤드 $g_\phi$를 통해 L2 정규화된 표현 $z_i = w_i / \|w_i\|_2$를 학습한다.

**샘플별 손실:**

$$\mathcal{L}_i(z_i, y_i) = \frac{1}{2N_{y_i} - 1} \sum_{j=1}^{2N} \mathbb{1}_{i \neq j} \mathbb{1}_{y_i = y_j} P_{i,j} \tag{1}$$

$$P_{i,j} = -\log \frac{\exp(z_i \cdot z_j / \tau)}{\sum_{r=1}^{2N} \mathbb{1}_{r \neq i} \exp(z_i \cdot z_r / \tau)} \tag{2}$$

> **문제점:** 노이즈 레이블 환경에서 Eq.(1)은 잘못된 양성/음성 쌍을 선택하여 표현 품질이 저하된다. (Table 2에서 SCL의 40% 대칭 노이즈 시 72.66 → 58.32로 하락 확인)

---

#### 2.2.2 Interpolated Contrastive Learning (ICL)

Mixup을 대조 학습에 적용:

**입력 보간:**

$$x_i = \lambda x_a + (1 - \lambda) x_b, \quad \lambda \in [0,1] \sim \text{Beta}(\alpha, \alpha) \tag{3}$$

**보간된 대조 손실:**

$$\mathcal{L}_i^{MIX} = \lambda \mathcal{L}_i(z_i, y_a) + (1-\lambda)\mathcal{L}_i(z_i, y_b) \tag{4}$$

**메모리 뱅크 통합 최종 ICL 손실:**

$$\mathcal{L}^{ICL} = \mathcal{L}^{MIX} + \mathcal{L}^{MEM} \tag{5}$$

> ICL은 노이즈 샘플이 깨끗한 샘플과 같은 레이블을 가져도, 보간 연산으로 인해 노이즈 패턴의 암기를 어렵게 만든다.

---

#### 2.2.3 노이즈 탐지 전략 (k-NN 기반)

**기본 k-NN 소프트 레이블:**

$$p(c \mid x_i) = \frac{1}{K} \sum_{\substack{k=1 \\ x_k \in \mathcal{N}_i}}^{K} \mathbb{1}_{y_k \neq c} \tag{6}$$

**보정된 소프트 레이블 (핵심 개선):**

$$\hat{p}(c \mid x_i) = \frac{1}{K} \sum_{\substack{k=1 \\ x_k \in \mathcal{N}_i}}^{K} \mathbb{1}_{\hat{y}_k \neq c}, \quad \hat{y} = \arg\max_c p(c \mid x) \tag{7}$$

**불일치 점수 (노이즈 여부 판단):**

$$d_i = -y_i^T \log(\hat{p}) \tag{8}$$

**클래스별 정제 데이터셋 선택:**

$$\mathcal{D}_c = \{(x_i, y_i) : d_i \leq \gamma_c\} \tag{9}$$

> $\gamma_c$는 클래스 불균형 방지를 위해 동적으로 결정되는 임계값 (각 클래스별 중앙값 기준)

---

#### 2.2.4 준지도 분류 손실

**보간 기반 준지도 손실:**

$$\mathcal{L}_i^{SSL} = -\lambda \tilde{y}_a^T \log(h_i) - (1-\lambda)\tilde{y}_b^T \log(h_i) \tag{10}$$

**의사 레이블 결정:**

$$\tilde{y}_a = \begin{cases} y_a, & x_a \in \mathcal{D}_c \\ \bar{h}_a, & x_a \notin \mathcal{D}_c \end{cases} \tag{11}$$

**최종 MOIT 손실:**

$$\mathcal{L}^{MOIT} = \mathcal{L}^{ICL} + \mathcal{L}^{SSL} \tag{12}$$

---

#### 2.2.5 MOIT+ 정제 손실

부트스트래핑 보정을 포함한 파인튜닝:

$$\mathcal{L}_i^{MOIT+} = -\lambda \left[(\delta y_a + (1-\delta)\tilde{y}_a)^T \log(h_i)\right] - (1-\lambda)\left[(\delta y_b + (1-\delta)\tilde{y}_b)^T \log(h_i)\right] \tag{13}$$

> $\delta = 0.8$로 설정하여 원본 레이블에 더 높은 가중치 부여

---

### 2.3 모델 구조

```
[미니배치]
    ↓ (1st/2nd 뷰 생성, Mixup 적용)
[인코더 f_θ] → v_i (중간 표현)
    ├── [프로젝션 헤드 g_φ] → z_i → L^ICL 손실
    └── [분류기 h = g_ϕ]   → ŷ  → L^SSL 손실
         ↑
    [노이즈 탐지 (k-NN)] ← z_i (매 에포크마다)
         ↓
    [MOIT+ 파인튜닝] (탐지된 정제 데이터 D로)
```

- **인코더:** CIFAR → PreAct ResNet-18 (PRN-18), mini-ImageNet/WebVision → ResNet-18
- **프로젝션 헤드 + 분류기:** 128차원 출력을 가진 선형 레이어
- **메모리 뱅크:** CIFAR 20K, mini-ImageNet 100K, mini-WebVision 50K 샘플

---

### 2.4 성능 향상

#### CIFAR-10 (평균 정확도):

| 방법 | 평균 Avg |
|------|----------|
| CE (기준) | 72.50 |
| DivideMix | 85.02 |
| ELR | 86.22 |
| **MOIT** | **89.73** |
| **MOIT+** | **91.33** |

#### CIFAR-100 (평균 정확도):

| 방법 | 평균 Avg |
|------|----------|
| CE (기준) | 50.02 |
| DivideMix | 63.99 |
| ELR | 69.51 |
| **MOIT** | **68.85** |
| **MOIT+** | **71.69** |

#### mini-WebVision (Best):

| 방법 | 정확도 |
|------|--------|
| DivideMix | 76.08 |
| ELR | 73.00 |
| **MOIT** | **78.36** |
| **MOIT+** | **78.76** |

---

### 2.5 한계점

1. **고비율 대칭 노이즈 취약성:** 80% 노이즈 시 DivideMix, ELR 대비 일부 설정에서 열세 (ELR: 36.83 vs MOIT: 45.63 in CIFAR-100 S-80%)
2. **비대칭 노이즈의 ICL 효과 감소:** 의미론적으로 유사한 클래스 간 레이블 플립은 ICL이 더 쉽게 학습하는 정보를 제공하여 개선 폭이 작음
3. **웹 노이즈에서 MOIT+ 개선 폭 감소:** 분포 외(out-of-distribution) 샘플이 지배적인 웹 노이즈에서는 준지도 학습의 레이블 보정 이득이 감소
4. **인스턴스 의존 노이즈 미고려:** 논문 자체에서 향후 과제로 제시
5. **컴퓨팅 자원 의존성:** 메모리 뱅크 및 강한 데이터 증강 필요

---

## 3. 모델의 일반화 성능 향상 가능성

MOIT가 일반화 성능을 향상시키는 핵심 메커니즘은 다음 세 가지 시너지 효과에 있다.

### 3.1 ICL의 정규화 효과

**Mixup 보간 전략**은 다음 수식으로 표현되는 선형 관계를 부과함으로써:

$$x_i = \lambda x_a + (1-\lambda)x_b$$

$$\mathcal{L}_i^{MIX} = \lambda \mathcal{L}_i(z_i, y_a) + (1-\lambda)\mathcal{L}_i(z_i, y_b)$$

- 특징 공간에서 **볼록 결합(convex combinations)**을 강제하여 결정 경계를 매끄럽게 만든다.
- 노이즈 레이블 패턴의 암기를 방해하는 암묵적 정규화 역할을 한다.
- 이는 Mixup 원 논문(Zhang et al., 2018)의 **경험적 위험 최소화를 넘는 일반화** 원리와 일치한다.

### 3.2 대조 학습과 분류의 상호 강화

- **대조 학습 → 분류:** 노이즈 강건 표현 $z$가 정확한 k-NN 노이즈 탐지를 가능하게 하여, 더 정제된 레이블로 분류기 학습
- **분류 → 대조 학습:** Table 2에서 확인된 바와 같이, 분류 목표를 추가하면 SCL 및 ICL 단독 학습보다 더 좋은 표현 획득:

$$\text{MOIT: 67.42 (A-40\%)} > \text{ICL: 72.04} \approx \text{SCL: 68.00}$$

(단, 무노이즈 기준으로 MOIT > ICL > SCL 순으로 개선)

### 3.3 준지도 학습을 통한 정보 활용 극대화

$$\tilde{y}_a = \begin{cases} y_a & \text{(검증된 정제 샘플)} \\ \bar{h}_a & \text{(노이즈 샘플 → 의사 레이블)} \end{cases}$$

노이즈로 탐지된 샘플을 버리지 않고 비레이블 데이터로 활용하여:
- 데이터 효율성 증대
- 과적합 방지 (특히 고노이즈 비율에서)

### 3.4 클래스 균형 정제 데이터셋의 중요성

Table 4에서 확인:
- **Median 기준 균형화:** A-40% 71.42, S-40% 66.58
- **Unbalanced:** A-40% 69.58 (열세)

이는 클래스 불균형한 정제 데이터가 분류기 편향을 유발하는 문제를 방지하여 일반화 성능을 보호한다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### (a) 대조 학습 + 노이즈 레이블 연구의 새 방향 제시
MOIT는 **대조 학습 자체가 노이즈에 취약하다는 사실**을 최초로 체계적으로 분석하고 해결책을 제시했다. 이는 후속 연구들이 단순히 분류 손실 함수를 개선하는 것을 넘어, 표현 학습 자체의 강건성을 탐구하도록 유도한다.

#### (b) 다중 목표 학습 프레임워크의 확장 가능성
ICL $+$ SSL의 결합 방식은 다른 자기 지도(self-supervised) 방법들(예: SimCLR, MoCo, BYOL)과의 결합으로 확장 가능하며, 이는 **사전 훈련(pre-training) 단계에서의 노이즈 강건성** 연구로 이어질 수 있다.

#### (c) k-NN 기반 노이즈 탐지의 발전
$\hat{p}$를 이용한 보정 소프트 레이블 방식은:
- 신뢰 집합(trusted clean set) 없이 작동
- 가우시안 혼합 모델(GMM) 기반 탐지(DivideMix)의 대안 제시
- 클러스터링 기반 노이즈 탐지 연구에 영감 제공

#### (d) 단일 하이퍼파라미터 구성의 실용성
데이터셋과 노이즈 유형에 상관없이 단일 설정으로 경쟁력 있는 성능 달성 → **실제 산업 응용에서의 배포 용이성** 연구에 영향

---

### 4.2 앞으로 연구 시 고려할 점

#### (a) 인스턴스 의존 노이즈 (Instance-Dependent Noise)
논문에서 직접 언급한 한계로, 노이즈가 입력 데이터의 특성에 따라 다르게 발생하는 현실적 시나리오에 대한 확장이 필요하다. 최근 연구들(예: Instance-Dependent Label Noise Learning, PDN 등)과의 결합이 유망하다.

#### (b) 클래스 프로토타입 기반 단순화
논문은 향후 과제로 **클래스 프로토타입을 이용한 대조 학습 단순화**를 제안한다. 이는 메모리 뱅크 없이도 효율적인 양성/음성 샘플 선택이 가능하게 할 수 있다.

#### (c) 대규모 데이터셋과 비전-언어 모델로의 확장
CLIP 등 비전-언어 사전훈련 모델에서의 노이즈 강건성은 아직 미탐구 영역으로, MOIT의 원리를 적용한 연구가 필요하다.

#### (d) 노이즈 탐지 정밀도/재현율 트레이드오프
$\hat{p}$ vs. $p$ 비교에서 확인된 바와 같이 (정밀도 90.83/재현율 87.84 vs. 80.20/84.43), 탐지 임계값 $\gamma_c$의 동적 조정 전략이 중요하며, 이를 학습 가능한 파라미터로 만드는 연구가 필요하다.

#### (e) 계산 비용 최적화
메모리 뱅크 + 강한 데이터 증강 + 두 단계 훈련(MOIT → MOIT+)은 상당한 계산 비용을 요구한다. 지식 증류(Knowledge Distillation)나 효율적인 대조 학습 방법을 통한 경량화가 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 비교 방법들

| 방법 | 핵심 아이디어 | MOIT 대비 차이점 |
|------|--------------|----------------|
| **DivideMix** (Li et al., ICLR 2020) | GMM으로 노이즈 탐지 + MixMatch 준지도 학습 | 두 네트워크 앙상블 필요, 데이터셋별 하이퍼파라미터 재조정 필요 |
| **ELR** (Liu et al., NeurIPS 2020) | 조기 학습 정규화(Early-Learning Regularization)로 암기 방지 | 고노이즈 비율에서 성능 급락 (CIFAR-10 S-80%: 38.23 vs MOIT: 70.53) |
| **Mixup** (Zhang et al., ICLR 2018) | 입력/레이블 보간으로 암기 방지 | 대조 학습 없음, 노이즈 탐지 없음 |
| **DRPL** (Ortego et al., ICPR 2020) | 다양한 노이즈 분포에 강건한 학습 | 대조 학습 미활용 |

### 5.2 2020년 이후 관련 연구 동향

논문이 직접 참조하거나 논문의 방향성과 연관된 2020년 이후 연구들:

**대조 학습 계열:**
- **Supervised Contrastive Learning** (Khosla et al., arXiv 2020): MOIT의 기반이 된 SCL 방법으로, MOIT는 이를 노이즈 환경에서 사용 가능하도록 확장
- **MoCo v2** (Chen et al., arXiv 2020): 메모리 뱅크 기반 모멘텀 대조 학습

**노이즈 레이블 계열:**
- **ELR** (Liu et al., NeurIPS 2020): 조기 학습 시점의 예측을 정규화에 활용 → MOIT와 달리 명시적 노이즈 탐지 없음
- **Normalized Loss Functions** (Ma et al., ICML 2020): 대칭 손실 함수로 노이즈 강건성 확보 → 손실 함수 측면만 고려

### 5.3 MOIT의 차별성 요약

$$\underbrace{\mathcal{L}^{MOIT} = \mathcal{L}^{ICL} + \mathcal{L}^{SSL}}_{\text{대조 학습 + 준지도 분류의 상호 강화}}$$

| 특성 | DivideMix | ELR | **MOIT** |
|------|-----------|-----|---------|
| 대조 학습 활용 | ✗ | ✗ | ✓ |
| 준지도 학습 | ✓ | ✗ | ✓ |
| 신뢰 집합 불필요 | ✓ | ✓ | ✓ |
| 단일 네트워크 | ✗ | ✓ | ✓ |
| 고노이즈 강건성 | △ | △ | ✓ |
| 단일 하이퍼파라미터 | ✗ | △ | ✓ |

---

## 참고 자료

**논문 원문:**
- Ortego, D., Arazo, E., Albert, P., O'Connor, N.E., & McGuinness, K. (2021). *Multi-Objective Interpolation Training for Robustness to Label Noise*. arXiv:2012.04462v2.

**논문 내 핵심 참조 문헌:**
- Khosla, P. et al. (2020). *Supervised Contrastive Learning*. arXiv:2004.11362.
- Li, J., Socher, R., & Hoi, S.C.H. (2020). *DivideMix: Learning with Noisy Labels as Semi-supervised Learning*. ICLR 2020.
- Liu, S. et al. (2020). *Early-Learning Regularization Prevents Memorization of Noisy Labels*. NeurIPS 2020.
- Zhang, H. et al. (2018). *mixup: Beyond Empirical Risk Minimization*. ICLR 2018.
- He, K. et al. (2020). *Momentum Contrast for Unsupervised Visual Representation Learning*. CVPR 2020.
- Chen, T. et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. ICML 2020.
- Arazo, E. et al. (2019). *Unsupervised Label Noise Modeling and Loss Correction*. ICML 2019.
- Ma, X. et al. (2020). *Normalized Loss Functions for Deep Learning with Noisy Labels*. ICML 2020.
