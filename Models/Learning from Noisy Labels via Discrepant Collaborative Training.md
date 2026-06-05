# Learning from Noisy Labels via Discrepant Collaborative Training (DCT)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Han et al., WACV 2020)의 핵심 주장은 다음과 같습니다:

> **Co-Training 프레임워크에서 두 네트워크 간의 명시적 불일치(Discrepancy)를 Maximum Mean Discrepancy(MMD)를 통해 강제함으로써, 노이즈 레이블 환경에서 더욱 강인하고 판별력 있는 특징을 학습할 수 있다.**

기존 Co-Teaching 방식은 두 네트워크가 동일한 구조로 시작하기 때문에, 시간이 지날수록 두 네트워크의 표현이 수렴하여 서로 보완적인 역할을 잃게 됩니다. DCT는 이 문제를 해결하기 위해 **Diversity(다양성) 손실**과 **Consistency(일관성) 손실**을 명시적으로 설계합니다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **새로운 다양성 도입 방법** | MMD를 활용한 Co-Training 내 네트워크 간 통계적 불일치 극대화 |
| **클린 샘플 선별 개선** | 불일치 강제를 통해 노이즈 레이블 샘플 식별 능력 향상 |
| **노이즈 데이터 활용** | 노이즈 샘플도 다양성 학습에 활용 (DCT vs DCT-clean 비교) |
| **광범위한 실험** | MNIST, CIFAR10/100, CUB200-2011, CARS196 등 5개 데이터셋 검증 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

딥러닝 모델은 노이즈 레이블이 포함된 데이터셋에서 훈련 시 **노이즈를 쉽게 기억(memorize)** 하는 문제가 있습니다. 기존 해법의 한계는 다음과 같습니다:

- **노이즈 전이 행렬 추정 방식** (예: Goldberger et al.): 클래스 수 증가 시 추정이 매우 어려워짐
- **Co-Teaching** (Han et al., 2018): 동일 구조의 두 네트워크가 점차 수렴하여 보완성 약화
- **MentorNet**: 사전 훈련된 교사 네트워크 필요, 별도의 클린 검증 세트 요구

**핵심 문제**: 동일 구조, 동일 초기화 방식의 두 네트워크는 학습이 진행될수록 유사한 표현을 학습하게 되어 Co-Training의 보완성 전제가 무너짐.

---

### 2.2 제안하는 방법 및 수식

#### (1) Maximum Mean Discrepancy (MMD)

두 분포 $\mathcal{S}$와 $\mathcal{T}$ 사이의 MMD는 다음과 같이 정의됩니다:

$$\text{MMD}(\mathcal{S}, \mathcal{T}) = \left\| \frac{1}{n}\sum_{i=1}^{n}\Phi(\boldsymbol{X}_i^{\mathcal{S}}) - \frac{1}{m}\sum_{j=1}^{m}\Phi(\boldsymbol{X}_j^{\mathcal{T}}) \right\|_{\mathcal{H}}^{2} \tag{1}$$

커널 트릭(kernel trick)을 적용하면:

$$\text{MMD}(\mathcal{S}, \mathcal{T}) = \frac{1}{n^2}\sum_{i}\sum_{i'}k(\boldsymbol{X}_i^{\mathcal{S}}, \boldsymbol{X}_{i'}^{\mathcal{S}}) - \frac{1}{nm}\sum_{i}\sum_{j}k(\boldsymbol{X}_i^{\mathcal{S}}, \boldsymbol{X}_j^{\mathcal{T}}) + \frac{1}{m^2}\sum_{j}\sum_{j'}k(\boldsymbol{X}_j^{\mathcal{T}}, \boldsymbol{X}_{j'}^{\mathcal{T}}) \tag{2}$$

Gaussian 커널은 다음과 같이 사용됩니다:

$$k(\boldsymbol{u}, \boldsymbol{v}) = \exp\left(-\frac{\|\boldsymbol{u} - \boldsymbol{v}\|^2}{\sigma}\right) \tag{3}$$

---

#### (2) 클린 샘플 선별 전략 (Selection Strategy)

두 네트워크 $f$와 $g$에 대한 분류 손실:

$$\mathrm{L}_f(\boldsymbol{X}_i) = -\log\left(\frac{\exp(\boldsymbol{z}_i^f)}{\sum_{1}^{m}\exp(\boldsymbol{z}_j^f)}\right), \quad \mathrm{L}_g(\boldsymbol{X}_i) = -\log\left(\frac{\exp(\boldsymbol{z}_i^g)}{\sum_{1}^{m}\exp(\boldsymbol{z}_j^g)}\right) \tag{4}$$

여기서 $\boldsymbol{z}\_i^f = f_\theta(\boldsymbol{X}\_i)$, $\boldsymbol{z}\_i^g = g_{\hat{\theta}}(\boldsymbol{X}_i)$

미니배치 $N$에서 가장 낮은 손실을 보이는 $R$개의 샘플을 선택하여 교차 학습:

$$\mathrm{L}_1^f = \sum_{i=1}^{R} \mathrm{L}_f(\boldsymbol{X}_i) \quad \forall \boldsymbol{X}_i \in \mathcal{D}_g \tag{5}$$

$$\mathrm{L}_1^g = \sum_{i=1}^{R} \mathrm{L}_g(\boldsymbol{X}_i) \quad \forall \boldsymbol{X}_i \in \mathcal{D}_f \tag{6}$$

---

#### (3) Diversity Loss (다양성 손실) — $\mathrm{L}_2$

중간 레이어 $l$에서 두 네트워크의 특징 표현 분포 간 MMD를 **최대화**:

$$\mathrm{L}_2 = \text{MMD}(\boldsymbol{A}_i, \boldsymbol{B}_i) \tag{7}$$

$$\boldsymbol{A}_i = f_{\theta(1:l)}(\boldsymbol{X}_i), \quad \boldsymbol{B}_i = g_{\hat{\theta}(1:l)}(\boldsymbol{X}_i)$$

→ $\mathrm{L}_2$를 최대화하여 두 네트워크가 서로 다른 특징을 학습하도록 강제합니다.

---

#### (4) Consistency Loss (일관성 손실) — $\mathrm{L}_3$

Softmax 출력(클래스 확률 분포) 사이의 MMD를 **최소화**:

$$\mathrm{L}_3 = \text{MMD}(\boldsymbol{z}_i^f, \boldsymbol{z}_i^g) \tag{8}$$

→ 특징은 다르게 학습하되, 최종 분류 결과는 일치하도록 강제합니다.

---

#### (5) 최종 손실 함수

$$\mathrm{Loss}_f = \mathrm{L}_1^f + \lambda_3 \mathrm{L}_3 - \lambda_2 \mathrm{L}_2 \tag{9}$$

$$\mathrm{Loss}_g = \mathrm{L}_1^g + \lambda_3 \mathrm{L}_3 - \lambda_2 \mathrm{L}_2 \tag{10}$$

- $\lambda_2$: Diversity loss 가중치 (불일치 최대화)
- $\lambda_3$: Consistency loss 가중치 (일관성 최소화)
- $-\lambda_2 \mathrm{L}_2$: 음의 부호로 MMD를 **최대화** (gradient ascent 효과)

---

### 2.3 모델 구조

#### 소규모 데이터셋용 CNN (MNIST, CIFAR10/100)

| 레이어 | 구성 |
|---|---|
| Layer 1–3 | $3\times3$ conv, 128 LReLU (slope=0.01) |
| — | $2\times2$ max-pool (stride 2), Dropout ($p=0.25$) |
| Layer 4–6 | $3\times3$ conv, 256 LReLU |
| — | $2\times2$ max-pool (stride 2), Dropout ($p=0.25$) |
| Layer 7 | $3\times3$ conv, 512 LReLU |
| Layer 8 | $3\times3$ conv, 256 LReLU |
| Layer 9 | $3\times3$ conv, 128 LReLU |
| — | avg-pool |
| Layer 10 | FC: $128 \to K$, Softmax |

- **대규모 세밀 인식 데이터셋 (CUB200-2011, CARS196)**: ImageNet 사전훈련 **Inception-V1** 사용

#### DCT 구조 핵심 요소

```
입력 이미지
    ↓ (두 서브네트워크 f, g 병렬 처리)
[Layer 1 ~ Layer n]
    ↓
[D₁: Diversity Module (MMD 최대화)] ← 중간 레이어 (5번째 레이어)
    ↓
[Softmax 출력]
    ↓
[D₂: Consistency Module (MMD 최소화)] ← Softmax 이후
    ↓
손실 기반 클린 샘플 선별 → 교차 업데이트
```

---

### 2.4 성능 향상

#### MNIST, CIFAR10, CIFAR100 결과 (Table 3)

| 노이즈 유형 | 데이터셋 | F-correction | Decoupling | MentorNet | Co-Teaching | **DCT** |
|---|---|---|---|---|---|---|
| pairflip-45% | MNIST | 0.24 | 58.03 | 80.88 | 87.63 | **88.54** |
| symmetric-50% | MNIST | 79.61 | 81.15 | 90.05 | 91.32 | **94.21** |
| symmetric-20% | MNIST | **98.82** | 95.70 | 96.70 | 97.25 | 98.54 |
| pairflip-45% | CIFAR10 | 6.61 | 48.80 | 58.14 | 72.62 | **72.91** |
| symmetric-50% | CIFAR10 | 59.83 | 51.49 | 71.10 | 74.02 | **78.50** |
| symmetric-20% | CIFAR10 | 84.55 | 80.44 | 80.76 | 82.32 | **85.41** |
| pairflip-45% | CIFAR100 | 1.60 | 26.05 | 31.60 | 34.81 | **35.33** |
| symmetric-50% | CIFAR100 | 41.04 | 25.80 | 39.00 | 41.37 | **42.11** |
| symmetric-20% | CIFAR100 | **61.87** | 44.52 | 52.13 | 54.23 | 56.11 |

#### 세밀 인식 데이터셋 결과 (Table 4)

| 노이즈 유형 | 데이터셋 | Cross Entropy | Co-Teaching | **DCT** |
|---|---|---|---|---|
| symmetric-50% | CUB200-2011 | 40.80 | 54.64 | **57.24** |
| symmetric-20% | CUB200-2011 | 63.78 | 72.34 | **74.57** |
| symmetric-50% | CARS196 | 38.86 | 66.75 | **67.80** |
| symmetric-20% | CARS196 | 71.76 | 86.00 | **86.62** |

---

### 2.5 한계점

1. **하이퍼파라미터 민감성**: $\lambda_2$, $\lambda_3$, $\sigma$ 등 여러 하이퍼파라미터의 최적값이 데이터셋마다 다르며, 튜닝 비용이 높습니다.
2. **계산 복잡도**: MMD 계산은 $O(n^2)$ 복잡도를 가지므로, 대용량 미니배치에서 연산 부담이 증가합니다.
3. **노이즈 유형 제한**: 대칭(symmetric) 및 쌍(pair) 플리핑 노이즈만 실험하였으며, **인스턴스 의존적(instance-dependent) 노이즈**에 대한 검증이 없습니다.
4. **두 네트워크로 제한**: 이론적으로 3개 이상의 네트워크로 확장 가능하나, 실험은 2개로 한정됩니다.
5. **F-correction 대비 열세 (저노이즈)**: CIFAR100 symmetric-20%에서 F-correction(61.87%)보다 낮은 56.11%를 기록합니다.
6. **실제(real-world) 노이즈 데이터 검증 부재**: 인공적으로 생성한 노이즈만 사용하며, Web에서 수집한 실제 노이즈 데이터 실험이 없습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상의 핵심 메커니즘

DCT가 일반화 성능을 향상시키는 핵심 원리는 **앙상블 다양성(Ensemble Diversity)**과 **특징 공간 정규화(Feature Space Regularization)**입니다.

#### (a) Diversity Loss의 정규화 효과

$$-\lambda_2 \cdot \text{MMD}(\boldsymbol{A}_i, \boldsymbol{B}_i)$$

이 항은 두 네트워크의 중간 레이어 표현이 **RKHS 상에서 최대한 멀어지도록** 강제합니다. 이는 각 네트워크가 서로 다른 특징 서브스페이스를 탐색하게 하여, 노이즈에 의한 결정 경계 편향을 서로 상쇄하는 효과를 가집니다.

수식적으로, 노이즈 레이블 $\tilde{y}$가 포함된 경우:

$$\mathbb{E}[\mathrm{Loss}_f] = \mathbb{E}[\mathrm{L}_1^f] + \lambda_3 \mathbb{E}[\mathrm{L}_3] - \lambda_2 \mathbb{E}[\mathrm{L}_2]$$

$-\lambda_2 \mathbb{E}[\mathrm{L}_2]$ 항이 크면 두 네트워크는 서로 다른 로컬 미니마로 유도되어 앙상블 관점에서의 분산(variance)이 감소합니다.

#### (b) Consistency Loss의 수렴 보장

$$+\lambda_3 \cdot \text{MMD}(\boldsymbol{z}_i^f, \boldsymbol{z}_i^g)$$

특징 공간에서의 다양성에도 불구하고, 최종 클래스 분포가 일치하도록 강제함으로써 **두 네트워크가 동일한 의사결정으로 수렴**하도록 유도합니다. 이는 다양한 관점에서의 합의를 의미하며 일반화 성능에 기여합니다.

#### (c) 노이즈 샘플의 역설적 기여

Ablation Study(Table 5)에서 흥미로운 결과가 나타났습니다:

$$\text{DCT} > \text{DCT-clean} > \text{Co-Teaching}$$

DCT-clean은 Diversity Loss 계산 시 선별된 클린 샘플만 사용한 변형입니다:

$$\mathrm{L}_2 = \text{MMD}(\hat{\boldsymbol{A}}_i, \hat{\boldsymbol{B}}_i), \quad \hat{\boldsymbol{A}}_i = f_{\theta(1:l)}(\boldsymbol{X}_i) \; \forall \boldsymbol{X}_i \in \mathcal{D}_f, \quad \hat{\boldsymbol{B}}_i = g_{\hat{\theta}(1:l)}(\boldsymbol{X}_i) \; \forall \boldsymbol{X}_i \in \mathcal{D}_g \tag{11}$$

**노이즈 샘플 전체를 사용한 DCT가 더 우수**한 이유: 노이즈 레이블이 두 네트워크의 결정 경계를 다방향으로 교란하여 더 넓은 특징 공간을 탐색하게 하고, 결과적으로 더 강건한 결정 경계를 학습합니다.

#### (d) 세밀 인식(Fine-Grained) 데이터에서의 일반화

CUB200-2011과 CARS196처럼 **클래스 간 분산이 낮고, 클래스 내 분산이 높은** 세밀 인식 데이터에서 노이즈의 영향이 특히 큽니다. DCT의 Diversity Module은 두 네트워크가 서로 다른 세밀한 특징(예: 새의 부리 vs 날개 패턴)을 학습하도록 유도하여, 노이즈 환경에서도 보다 풍부한 표현을 학습합니다.

### 3.2 일반화 향상 가능성의 구체적 시나리오

| 시나리오 | DCT의 잠재적 기여 |
|---|---|
| **도메인 전이(Domain Transfer)** | 다양한 특징 학습으로 새로운 도메인에서의 적응력 향상 |
| **클래스 불균형** | 다양한 관점의 특징이 소수 클래스 표현 개선 |
| **반지도 학습** | 레이블 없는 데이터의 다양한 특징 활용 가능성 |
| **Few-Shot Learning** | 다양한 특징 서브스페이스 탐색으로 소량 데이터 일반화 가능성 |

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

#### (a) 분산 기반 다양성 도입의 일반화

DCT는 MMD를 통한 **통계적 다양성 강제**라는 새로운 패러다임을 제시합니다. 이는 단순히 노이즈 레이블 문제를 넘어 다음 분야에 영향을 미칩니다:

- **앙상블 학습**: 네트워크 다양성을 명시적으로 제어하는 새로운 앙상블 학습 방향
- **도메인 적응**: MMD 기반의 특징 정렬/분리를 결합한 적응 기법
- **자기지도 학습(Self-supervised Learning)**: 다양한 뷰 간의 분포 불일치를 활용한 표현 학습

#### (b) 노이즈 레이블 연구의 방향성 전환

DCT는 **노이즈 데이터를 단순히 제거할 대상이 아니라 활용 가능한 정보원**으로 바라보는 관점을 제시합니다. 이후 연구들(예: DivideMix, CORES²)도 노이즈 샘플의 준지도 학습적 활용을 탐색합니다.

#### (c) MMD 이외의 분산 측도 탐색 동기

논문 결론부에서 저자들이 직접 언급한 바와 같이, Sinkhorn Divergence, Optimal Transport, Wasserstein Distance 등 대안적 측도의 노이즈 학습 적용 연구를 촉진합니다.

---

### 4.2 향후 연구 시 고려할 점

#### (a) 인스턴스 의존적 노이즈(Instance-Dependent Noise) 대응

실제 데이터에서의 노이즈는 클래스가 아닌 **개별 샘플의 특성에 의존**합니다. 예를 들어, 이미지 품질이 낮거나 모호한 샘플일수록 레이블이 틀릴 가능성이 높습니다.

$$Q_{ij}^{\text{instance}} = P(\tilde{y}=j \mid y=i, \boldsymbol{x})$$

DCT의 전이 행렬 가정은 이를 처리하지 못하므로, 인스턴스별 노이즈 모델링과의 결합이 필요합니다.

#### (b) 확장 가능한 MMD 계산

MMD의 $O(n^2)$ 복잡도는 대규모 미니배치에서 병목이 됩니다. **Random Fourier Features** 기반의 근사 MMD나 **Mini-batch MMD**를 활용하면 확장성 문제를 완화할 수 있습니다:

$$\widehat{\text{MMD}}^2(\mathcal{S}, \mathcal{T}) \approx \frac{1}{D}\sum_{d=1}^{D}\left(\hat{\mu}_{\mathcal{S}}(\omega_d) - \hat{\mu}_{\mathcal{T}}(\omega_d)\right)^2$$

#### (c) 동적 하이퍼파라미터 스케줄링

$\lambda_2$와 $\lambda_3$를 고정된 값으로 설정하는 현재 방식 대신, 훈련 에폭에 따라 동적으로 조정하는 스케줄링 전략이 필요합니다. 초기에는 다양성을 강조하고, 후반에는 일관성을 강조하는 커리큘럼 방식을 고려할 수 있습니다.

#### (d) 실제 노이즈 데이터셋 검증

WebVision, Clothing1M과 같은 **실제 노이즈 레이블 데이터셋**에서의 검증이 필요합니다. 인공 노이즈 실험과 실제 노이즈 환경의 차이가 크기 때문에 실용성 평가에 필수적입니다.

#### (e) 세밀 인식의 약지도(Weakly-supervised) 학습과의 결합

CUB200-2011, CARS196 등에서의 결과가 유망하므로, **약지도 세밀 인식**과 DCT 프레임워크를 결합하는 연구가 가치 있을 것입니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래는 DCT와 관련된 2020년 이후 주요 노이즈 레이블 연구들입니다. **단, 아래 내용은 논문 원문에 없는 외부 연구이므로, 제가 알고 있는 범위 내에서 기술하며 부정확할 수 있음을 명시합니다.**

### 5.1 주요 후속 연구 비교

| 연구 | 핵심 아이디어 | DCT와의 차이점 |
|---|---|---|
| **DivideMix** (Li et al., ICLR 2020) | GMM으로 클린/노이즈 분리 + MixMatch 준지도 학습 | 노이즈 샘플을 레이블 없는 데이터로 준지도 학습에 활용; DCT보다 강력한 노이즈 처리 |
| **CORES²** (Cheng et al., NeurIPS 2021) | 샘플별 신뢰도 점수(credibility score) 학습 | 인스턴스 의존적 노이즈 처리; 보다 정교한 샘플 선별 |
| **ELR** (Liu et al., NeurIPS 2020) | Early Learning Regularization; 과거 예측 활용 | 네트워크 단일 사용; 기억 효과(memorization)를 명시적으로 억제 |
| **Co-learning** (Tan et al., 2021) | 메타 가중치와 Co-Training 결합 | DCT의 Co-Training을 메타 학습으로 확장 |
| **UNICON** (Karim et al., CVPR 2022) | 균일 선택과 GMM 기반 혼합 | 클래스 불균형 노이즈에 강건 |

### 5.2 DCT의 관점에서 본 최신 연구 트렌드

```
DCT (WACV 2020)
    ↓ 영향
[분포 기반 샘플 분리]         [준지도 학습 활용]         [인스턴스 의존 노이즈]
  DivideMix (2020)          MixUp 기반 방법들           CORES² (2021)
  ELR (2020)                FixMatch + 노이즈           PDN (2021)
```

**DCT의 차별점**: DCT는 **네트워크 간 분포 불일치를 명시적으로 제어**한다는 점에서 독창적입니다. 반면 DivideMix 등 이후 연구들은 노이즈 샘플을 완전히 버리지 않고 준지도 학습에 통합하는 방향으로 발전하였습니다. DCT의 "노이즈 샘플도 다양성 학습에 기여한다"는 통찰은 이 트렌드와 맥락을 같이합니다.

---

## 참고 자료 (출처)

1. **Han, Y., Roy, S.K., Petersson, L., & Harandi, M. (2020)**. *Learning from Noisy Labels via Discrepant Collaborative Training*. WACV 2020. (본 논문 원문 PDF)

2. **Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., Tsang, I., & Sugiyama, M. (2018)**. *Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels*. NeurIPS 2018. (본 논문 Reference [16])

3. **Gretton, A., Borgwardt, K.M., Rasch, M.J., Schölkopf, B., & Smola, A. (2012)**. *A Kernel Two-sample Test*. Journal of Machine Learning Research. (본 논문 Reference [15])

4. **Jiang, L., Zhou, Z., Leung, T., Li, L.J., & Fei-Fei, L. (2017)**. *MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks on Corrupted Labels*. (본 논문 Reference [19])

5. **Li, J., Socher, R., & Hoi, S.C. (2020)**. *DivideMix: Learning with Noisy Labels as Semi-supervised Learning*. ICLR 2020.

6. **Liu, S., Niles-Weed, J., Razavian, N., & Fernandez-Granda, C. (2020)**. *Early-Learning Regularization Prevents Memorization of Noisy Labels*. NeurIPS 2020.

7. **Patrini, G., Rozza, A., Menon, A.K., Nock, R., & Qu, L. (2017)**. *Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach*. CVPR 2017. (본 논문 Reference [33])

> **주의**: 섹션 5(최신 연구 비교)의 DivideMix, CORES², ELR, UNICON 관련 내용은 본 논문 원문에 포함되지 않은 외부 지식으로, 세부 수치나 정확한 비교는 해당 원논문을 직접 확인하시기 바랍니다.
