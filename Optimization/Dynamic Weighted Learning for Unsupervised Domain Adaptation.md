# Dynamic Weighted Learning for Unsupervised Domain Adaptation

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문(Xiao & Zhang, 2021)의 핵심 주장은 다음과 같습니다:

> **기존 UDA 방법들은 도메인 정렬(Domain Alignment)과 클래스 판별성(Class Discriminability)을 독립적으로 학습하기 때문에, 어느 한 쪽의 과도한 학습이 다른 쪽을 저하시키는 "부정적 전이(Negative Transfer)" 문제를 야기한다.**

이를 해결하기 위해 두 학습 목표를 **동적으로 균형 조절**하는 Dynamic Weighted Learning(DWL)을 제안합니다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **① 동적 가중 학습 (DWL)** | 정렬 정도와 판별성 정도를 실시간 모니터링하여 두 손실 함수의 가중치를 동적으로 조절 |
| **② 샘플 가중치 기법** | 도메인 간 샘플 수 불균형으로 인한 모델 편향을 방지하는 샘플 재가중화 |
| **③ 범용성** | 동적 가중 메커니즘을 통해 다양한 도메인 이동 시나리오에 적용 가능한 플러그인 방식 제공 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 1: 정렬-판별성 불균형 (Alignment-Discriminability Imbalance)**

논문은 Figure 1을 통해 DANN과 MCD에서 이 문제를 실증적으로 보입니다:
- DANN: 도메인 정렬이 향상될수록 클래스 판별성(max $J(\mathbf{W})$ )이 감소
- MCD: Epoch 25 이후 판별성이 하락 추세를 보임

이론적 근거는 Ben-David et al.의 상한 부등식에 기반합니다:

$$\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2} d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_s, \mathcal{D}_t) + \lambda \tag{1}$$

여기서:
- $\epsilon_S(h)$: 소스 도메인에서의 기대 오류
- $d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_s, \mathcal{D}_t)$: 두 도메인 간의 $\mathcal{H}\Delta\mathcal{H}$-divergence
- $\lambda = \epsilon_S(h^\*) + \epsilon_T(h^*)$: 이상적 결합 가설의 결합 오류

$$d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_s, \mathcal{D}_t) = 2 \sup_{h, h' \in \mathcal{H}} \left| \Pr_{x \sim \mathcal{D}_s}(h(x) \neq h'(x)) - \Pr_{x \sim \mathcal{D}_t}(h(x) \neq h'(x)) \right| \tag{2}$$

**핵심 문제**: 과도한 정렬 학습은 $d_{\mathcal{H}\Delta\mathcal{H}}$를 줄이지만 $\lambda$를 증가시키고, 과도한 판별성 학습은 반대 효과를 낳습니다. DWL은 두 항을 **동시에** 감소시키는 것을 목표로 합니다.

**문제 2: 도메인 간 샘플 수 불균형**

기존 방법들은 두 도메인의 샘플 수 차이를 고려하지 않아, 샘플이 많은 도메인 쪽으로 모델이 편향됩니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 샘플 가중치 부여 (Sample Weighting)

각 도메인의 샘플 비율에 반비례하여 가중치를 부여합니다:

$$\hat{x}^s_i = a \left(1 + \frac{n_t}{n_s}\right) x^s_i, \quad i = 1, 2, \ldots, n_s \tag{4}$$

$$\hat{x}^t_j = a \left(1 + \frac{n_s}{n_t}\right) x^t_j, \quad j = 1, 2, \ldots, n_t \tag{5}$$

여기서 $a \in (0, 1]$는 가중치 정도를 제어하는 하이퍼파라미터입니다.

#### Step 2: 도메인 정렬 손실 (Domain Alignment Loss)

적대적 학습 기반의 표준 미니맥스 손실:

$$\min_{\theta_g} \max_{\theta_d} \mathcal{L}_{da}(\theta_g, \theta_d) = \mathbb{E}_{x^s_i \sim \mathcal{D}_s} \log[D(G(\hat{x}^s_i))] + \mathbb{E}_{x^t_j \sim \mathcal{D}_t} \log[1 - D(G(\hat{x}^t_j))] \tag{6}$$

#### Step 3: 클래스 판별성 손실 (Class Discrimination Loss)

MCD에서 영감을 받아 세 개의 분류기(C, C1, C2)를 활용:

$$\min_{\theta_g, \theta_c} \max_{\theta_{c_1}, \theta_{c_2}} \mathcal{L}_{cd} = \mathbb{E}_{x^t_j \sim \mathcal{D}_t} \left[ \|C_1(G(\hat{x}^t_j)) - C_2(G(\hat{x}^t_j))\|_1 \right.$$
$$\left. + \|C(G(\hat{x}^t_j)) - C_1(G(\hat{x}^t_j))\|_1 + \|C(G(\hat{x}^t_j)) - C_2(G(\hat{x}^t_j))\|_1 \right] \tag{7}$$

#### Step 4: 정렬/판별성 측정 지표

**정렬 척도 (MMD)**:

$$\text{MMD}(\mathcal{D}_s, \mathcal{D}_t) = \left\| \mathbb{E}_{x^s_i \sim \mathcal{D}_s} G(\hat{x}^s_i) - \mathbb{E}_{x^t_j \sim \mathcal{D}_t} G(\hat{x}^t_j) \right\|^2 \tag{8}$$

**판별성 척도 (LDA 기반)**:

$$\max_{\mathbf{W}} J(\mathbf{W}) = \frac{\text{tr}(\mathbf{W}^\top \mathbf{S}_b \mathbf{W})}{\text{tr}(\mathbf{W}^\top \mathbf{S}_w \mathbf{W})} \tag{9}$$

여기서 $\mathbf{S}_b$는 클래스 간 산포 행렬(between-class scatter matrix), $\mathbf{S}_w$는 클래스 내 산포 행렬(within-class scatter matrix)입니다.

#### Step 5: Min-Max 정규화

$$\widetilde{\text{MMD}}(\mathcal{D}_s, \mathcal{D}_t) = \frac{\text{MMD}(\mathcal{D}_s, \mathcal{D}_t) - \text{MMD}(\mathcal{D}_s, \mathcal{D}_t)_{\min}}{\text{MMD}(\mathcal{D}_s, \mathcal{D}_t)_{\max} - \text{MMD}(\mathcal{D}_s, \mathcal{D}_t)_{\min}} \tag{10}$$

$$\tilde{J}(\mathbf{W}) = \frac{J(\mathbf{W}) - J(\mathbf{W})_{\min}}{J(\mathbf{W})_{\max} - J(\mathbf{W})_{\min}} \tag{11}$$

#### Step 6: 동적 균형 인수 (Dynamic Balance Factor)

$$\tau = \frac{\widetilde{\text{MMD}}(\mathcal{D}_s, \mathcal{D}_t)}{\widetilde{\text{MMD}}(\mathcal{D}_s, \mathcal{D}_t) + (1 - \tilde{J}(\mathbf{W}))} \tag{12}$$

**$\tau$의 직관적 해석**:

| 상황 | $\widetilde{\text{MMD}}$ | $1 - \tilde{J}(\mathbf{W})$ | $\tau$ 값 | 효과 |
|------|----------|------------|---------|------|
| 정렬 훨씬 좋음 | $\approx 0$ | $\approx 1$ | $\approx 0$ | 판별성 학습 강조 |
| 판별성 훨씬 좋음 | $\approx 1$ | $\approx 0$ | $\approx 1$ | 정렬 학습 강조 |
| 균형 상태 | 동일 | 동일 | $\approx 0.5$ | 균등 학습 |

#### Step 7: 최종 학습 목적 함수

$$\min_{\theta_g, \theta_c} \max_{\theta_d, \theta_{c_1}, \theta_{c_2}} \sum_{i=1}^{n_s} \mathcal{L}_{ce}(C(G(x^s_i; \theta_g); \theta_c), y^s_i) + \tau \cdot \mathcal{L}_{da}(\theta_g, \theta_d) + (1-\tau) \cdot \mathcal{L}_{cd}(\theta_g, \theta_c, \theta_{c_1}, \theta_{c_2}) \tag{14}$$

---

### 2.3 모델 구조

```
입력: 소스 도메인(레이블 있음) + 타겟 도메인(레이블 없음)
         ↓
    [Sample Weighting]  ← 샘플 불균형 보정
         ↓
    [Feature Generator G]  ← ResNet-50/101 백본
    ↙          ↓          ↘
[Domain         [Main           [Auxiliary
Discriminator D] Classifier C]  Classifiers C1, C2]
    ↓                ↓              ↓
[L_da]           [L_ce]         [L_cd]
    ↘               ↓            ↙
         [τ 동적 가중치 조절]
              ↑
    [MMD 측정] + [LDA 판별성 측정]
```

**핵심 구성 요소**:
- **G (Feature Generator)**: ResNet-50/101 (ImageNet 사전 학습)
- **D (Domain Discriminator)**: 2-layer network + ReLU + Dropout(0.5)
- **C (Main Classifier)**: 2-layer network (2048×1024×#classes)
- **C1, C2 (Auxiliary Classifiers)**: 소스 도메인에서 지도 학습으로 사전 학습
- **τ**: 매 이터레이션마다 실시간 계산되는 동적 균형 인수

---

### 2.4 성능 향상

#### VisDA-2017 (ResNet-101)

| Method | Mean Accuracy (%) |
|--------|------------------|
| ResNet101 | 52.4 |
| DANN | 57.4 |
| MCD | 71.9 |
| BSP | 75.9 |
| **DWL (Ours)** | **77.1** |

#### Digits Dataset

| Method | M→U | U→M | S→M | Average |
|--------|-----|-----|-----|---------|
| MCD | 94.2 | 94.1 | 96.2 | 94.8 |
| ETD | 96.4 | 96.3 | 97.9 | 96.9 |
| **DWL (Ours)** | **97.3** | **97.4** | **98.1** | **97.6** |

#### Office-31 (ResNet-50)

| Method | Average |
|--------|---------|
| ETD | 86.2 |
| **DWL (Ours)** | **87.1** |

#### ImageCLEF-DA (ResNet-50)

| Method | Average |
|--------|---------|
| ETD | 89.7 |
| **DWL (Ours)** | **90.5** |

---

### 2.5 한계점

논문에서 명시적으로 언급된 한계와 분석을 통한 한계를 구분하면:

**① Min-Max 정규화의 한계**: 식 (10), (11)의 정규화는 전체 학습 과정에서의 최솟값/최댓값에 의존하므로, 학습 초기에는 안정적인 추정이 어렵습니다.

**② MMD 추정의 편향**: 미니배치 기반 MMD 추정은 배치 크기가 작을 때 분산이 커질 수 있습니다.

**③ 하이퍼파라미터 $a$의 민감도**: 샘플 가중치 하이퍼파라미터 $a$에 대한 체계적인 민감도 분석이 부족합니다.

**④ 계산 오버헤드**: 매 이터레이션마다 MMD와 LDA를 계산하므로, 대규모 데이터셋에서 계산 비용이 증가합니다.

**⑤ 실험 범위**: Office-Home, DomainNet 등 더 대규모 벤치마크에 대한 실험이 없습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화에 기여하는 요소

**① 이론적 상한 최소화**

DWL은 Ben-David et al.의 상한식에서 $d_{\mathcal{H}\Delta\mathcal{H}}$와 $\lambda$를 동시에 감소시키는 유일한 접근법입니다:

$$\epsilon_T(h) \leq \underbrace{\epsilon_S(h)}_{\text{소스 오류}} + \underbrace{\frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_s, \mathcal{D}_t)}_{\text{도메인 격차}} + \underbrace{\lambda}_{\text{결합 오류}}$$

기존 방법들이 두 번째 항만 줄이는 반면, DWL은 세 번째 항($\lambda$, 판별성과 관련)도 동시에 관리합니다.

**② 다양한 도메인 이동 시나리오에 대한 적응성**

$\tau$가 데이터의 현재 상태에 따라 자동으로 조절되므로:
- **도메인 격차가 작은 경우** (예: Digits): $\tau \to 0$으로 판별성 학습 강조
- **도메인 격차가 큰 경우** (예: VisDA-2017): $\tau \to 1$로 정렬 학습 강조

이 메커니즘은 논문의 수렴 분석(Figure 4)에서 실증적으로 확인됩니다. Digits 데이터셋에서 $\tau$가 초기에 0.5 이하로 빠르게 떨어지는 것은, 모델이 정렬보다 판별성이 더 필요함을 자동 감지한 결과입니다.

**③ 샘플 불균형 처리를 통한 일반화**

Office-31의 W→A (795 vs 2817 샘플) 과제에서 샘플 가중치와 $\tau$를 함께 사용했을 때 ablation study 결과:

| Sample Weighting | Balancing Factor $\tau$ | Accuracy (%) |
|:---:|:---:|:---:|
| ✗ | ✗ | 67.9 |
| ✓ | ✗ | 68.3 |
| ✗ | ✓ | 69.1 |
| ✓ | ✓ | **69.8** |

두 메커니즘의 시너지 효과가 일반화에 핵심적 역할을 합니다.

**④ 플러그인 확장성**

DWL의 동적 가중치 메커니즘은 BSP, ETD 등 기존 UDA 방법에 통합 가능하며, 이는 다양한 도메인 및 태스크에 대한 범용 일반화를 지원합니다.

### 3.2 일반화의 잠재적 한계

- LDA 기반 판별성 추정( $\max J(\mathbf{W})$ )은 **가우시안 분포 가정**에 민감하므로, 비선형적 분포를 가진 도메인에서 일반화 성능이 저하될 수 있습니다.
- 현재 실험은 주로 **이미지 분류** 태스크에 한정되어 있어, 객체 탐지, 시맨틱 세그멘테이션 등으로의 일반화 검증이 필요합니다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 향후 연구에 미치는 영향

**① 동적 학습 패러다임의 확산**

DWL이 제안한 "실시간 모니터링 → 동적 균형 조절" 패러다임은 UDA를 넘어 다양한 멀티태스크 학습 문제에 영향을 미칠 수 있습니다. 특히 정렬과 판별성의 상호작용을 명시적으로 모델링한 점은 이후 연구의 중요한 출발점이 됩니다.

**② 이론과 실험의 연결**

Ben-David의 상한식을 실험 설계의 직접적 동기로 활용한 방식은, 향후 UDA 연구가 이론적 보장을 더 엄밀하게 고려하도록 촉진합니다.

**③ 샘플 불균형 문제의 인식 제고**

이 논문이 UDA에서 도메인 간 샘플 불균형을 처음으로 명시적으로 다룬 연구 중 하나로, 이후 연구들이 데이터 불균형을 더 체계적으로 고려하는 계기가 됩니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래 비교는 논문에서 인용된 연구 및 DWL과 유사한 방향의 연구들을 기반으로 합니다:

| 연구 | 핵심 아이디어 | DWL과의 비교 |
|------|-------------|-------------|
| **BSP** (Chen et al., ICML 2019) | 배치 스펙트럼 패널티로 판별성 보장 | 정렬/판별성 가중치가 고정 (1:1), DWL은 동적 조절 |
| **ETD** (Li et al., CVPR 2020) | 주의 기반 전송 거리 + 엔트로피 정규화 | 초기 분포 상태에 의존적, DWL은 샘플 가중치로 보완 |
| **LWC** (Ye et al., CVPR 2020) | 경량 캘리브레이터 | 분리 가능한 컴포넌트 방식, DWL의 플러그인 아이디어와 유사 |
| **CAT** (Deng et al., ICCV 2019) | 클러스터-조건부 구조 탐색 | 클래스 조건부 구조에 집중, DWL은 전역적 균형 |

> **⚠️ 주의**: 2021년 이후 최신 연구(예: CDTrans, TVT, SPA 등)와의 직접 비교는 본 논문(arXiv 2021.03)이 해당 논문들을 참조하지 않았으므로, 정확한 수치 비교를 제공하기 어렵습니다. 과도한 추정을 피하기 위해 논문 내 비교 결과만을 기반으로 기술합니다.

### 4.3 향후 연구 시 고려할 점

**① 더 강건한 판별성 측정 방법 탐구**

LDA 기반 $\max J(\mathbf{W})$를 넘어서, 비선형 커널 기반 판별성 척도나 정보 이론적 측도(예: 상호 정보, 엔트로피)를 활용하면 일반화 성능이 향상될 수 있습니다.

**② 계산 효율성 개선**

매 이터레이션마다 MMD와 LDA를 계산하는 비용을 줄이기 위해 근사 추정 방법(예: 랜덤 푸리에 특징 기반 MMD)을 적용할 수 있습니다.

**③ 다중 소스 도메인 적응으로의 확장**

현재 DWL은 단일 소스-단일 타겟 시나리오만 다루고 있습니다. 다중 소스 도메인에서의 동적 균형 조절 메커니즘 설계가 중요한 향후 과제입니다.

**④ 비전 태스크 외 영역으로의 적용**

자연어 처리(NLP)나 시계열 데이터에서의 도메인 적응에 DWL의 동적 가중치 아이디어를 적용하는 연구가 필요합니다.

**⑤ 더 강력한 벤치마크 검증**

DomainNet (345 클래스, 6개 도메인), Office-Home 등 더 어렵고 큰 규모의 벤치마크에서의 검증이 필요합니다.

**⑥ 자기 지도 학습과의 결합**

최근 대두되는 자기 지도 학습(Self-supervised Learning) 프레임워크와 DWL의 동적 균형 아이디어를 결합하면, 타겟 도메인의 특징 표현 품질을 추가로 향상시킬 수 있습니다.

---

## 참고자료

- **주 논문**: Ni Xiao, Lei Zhang. "Dynamic Weighted Learning for Unsupervised Domain Adaptation." arXiv:2103.13814v1 [cs.LG], 22 Mar 2021.
- Ben-David, S., et al. "A theory of learning from different domains." *Machine Learning*, 79, 151–175, 2010.
- Saito, K., et al. "Maximum Classifier Discrepancy for Unsupervised Domain Adaptation." *CVPR*, 2018.
- Chen, X., et al. "Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation." *ICML*, 2019.
- Li, M., et al. "Enhanced Transport Distance for Unsupervised Domain Adaptation." *CVPR*, 2020.
- Ganin, Y., et al. "Domain-adversarial training of neural networks." *JMLR*, 17(1):2096–2030, 2017.
- Deng, Z., et al. "Cluster Alignment with a Teacher for Unsupervised Domain Adaptation." *ICCV*, 2019.
- Ye, S., et al. "Light-weight Calibrator: a Separable Component for Unsupervised Domain Adaptation." *CVPR*, 2020.
