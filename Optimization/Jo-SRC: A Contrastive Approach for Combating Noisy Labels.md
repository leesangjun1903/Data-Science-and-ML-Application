# Jo-SRC: A Contrastive Approach for Combating Noisy Labels 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Jo-SRC는 **대조 학습(Contrastive Learning) 기반의 노이즈 레이블 대응 프레임워크**로, 기존 소손실(small-loss) 기반 샘플 선택 방법의 두 가지 핵심 한계를 해결합니다:

1. **미니배치 내 편향 선택 문제**: 기존 방법은 미니배치마다 고정 비율 $r$로 샘플을 선택하므로, 미니배치 간 노이즈 비율 불균형을 무시합니다.
2. **고손실(high-loss) 샘플 정보 낭비**: ID(In-Distribution) 노이즈 샘플과 OOD(Out-of-Distribution) 노이즈 샘플을 구분하지 않고 버립니다.

### 주요 기여

| 기여 | 설명 |
|---|---|
| **전역 클린 샘플 선택** | Jensen-Shannon Divergence로 미니배치 단위가 아닌 전역 선택 |
| **ID/OOD 구분** | 두 뷰(view)의 예측 일관성으로 ID와 OOD 노이즈 샘플 분리 |
| **결합 손실 함수** | 분류 손실 + 일관성 정규화 손실로 일반화 성능 향상 |
| **Mean-Teacher 재레이블링** | 노이즈 샘플에 대해 신뢰할 수 있는 의사 레이블 부여 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

DNN의 **메모리화 효과(memorization effect)**로 인해 노이즈 레이블로 학습 시 성능 저하가 발생합니다. 기존 방법들의 문제점:

- **가정의 비현실성**: 모든 미니배치의 노이즈 비율이 동일하다고 가정
- **폐쇄형 시나리오 한정**: 실제 환경에는 ID 노이즈와 OOD 노이즈가 혼재
- **정보 낭비**: 유용할 수 있는 고손실 샘플을 무조건 제거

---

### 2.2 제안 방법 및 수식

#### **배경: 표준 크로스 엔트로피 손실**

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_i^c \log(p_i^c) \tag{1}$$

여기서 $p_i^c$는 샘플 $x_i$의 클래스 $c$에 대한 예측 확률이며, $y_i^c$는 해당 원-핫 레이블입니다.

---

#### **Step 1: 전역 클린 샘플 선택 (Global Clean Sample Selection)**

예측 분포 $\mathbf{p}_i = [p_i^1, p_i^2, ..., p_i^C]$와 레이블 분포 $\mathbf{y}_i = [y_i^1, y_i^2, ..., y_i^C]$ 간의 **Jensen-Shannon Divergence(JSD)**를 이용:

$$d_i = D_{JS}(\mathbf{p}_i \| \mathbf{y}_i) = \frac{1}{2}D_{KL}\!\left(\mathbf{p}_i \,\Big\|\, \frac{\mathbf{p}_i + \mathbf{y}_i}{2}\right) + \frac{1}{2}D_{KL}\!\left(\mathbf{y}_i \,\Big\|\, \frac{\mathbf{p}_i + \mathbf{y}_i}{2}\right) \tag{2}$$

클린 샘플 가능성(likelihood):

$$\mathcal{P}_{clean}(x_i) = 1 - d_i \in [0, 1] \tag{3}$$

> **Criterion 3.1**: $\mathcal{P}\_{clean}(x) > \tau_{clean}$이면 클린 샘플로 선택

JSD는 $[0, 1]$로 유계(bounded)되므로, 미니배치 독립적인 **전역 선택 기준**으로 사용 가능합니다. 이는 크로스 엔트로피가 unbounded인 것과 대비됩니다.

**동적 임계값 조정:**

$$\tau_{clean} = \begin{cases} \dfrac{t}{t_w}\tau_c, & 1 \leq t \leq t_w \\[6pt] \dfrac{(t - t_w)\Delta\tau}{t_{max} - t_w} + \tau_c, & t_w < t \leq t_{max} \end{cases} \tag{12}$$

초기에는 임계값을 낮게 설정하여 많은 샘플로 학습하고, 이후 점차 높여 고품질 데이터만 사용합니다.

---

#### **Step 2: OOD/ID 샘플 구분 (Out-of-Distribution Detection)**

각 샘플 $x_i$에 대해 두 가지 서로 다른 증강(augmentation)을 적용하여 뷰 $v_i = T(x_i)$와 $v'_i = T'(x_i)$를 생성하고, 각각의 예측 $\mathbf{p}_i$와 $\mathbf{p}'_i$를 계산합니다.

OOD 가능성:

```math
\mathcal{P}_{ood}(x_i) = \min\!\left(1,\, \left|\text*{argmax}_c \mathbf{p}_i - \text*{argmax}_c \mathbf{p}'_i\right|\right)
```

> **Criterion 3.2**: Criterion 3.1에서 "unclean"으로 분류된 샘플 중, $\mathcal{P}\_{ood}(x_i) > \tau_{ood}$이면 OOD, 그렇지 않으면 ID 샘플

- **OOD 샘플**: 두 뷰의 예측이 불일치 → 모델이 혼란스러워함
- **ID 노이즈 샘플**: 레이블은 틀렸지만 예측은 일관성 있음

---

#### **Step 3: 레이블 재할당 (Label Re-assignment)**

세 그룹으로 분류된 샘플에 각각 다른 레이블 처리를 적용합니다.

**① 클린 샘플 ($\mathbb{S}_{clean}$)**: Label Smoothing Regularization(LSR) 적용

$$\tilde{y}_i^c = \begin{cases} 1 - \epsilon, & c = l_i \\ \dfrac{\epsilon}{C-1}, & c \neq l_i \end{cases} \tag{5}$$

**② ID 노이즈 샘플 ($\mathbb{S}_{id}$)**: Mean-teacher 모델로 의사 레이블 생성

$$\tilde{y}_i^c = p^c(x_i, \theta_{mt}) \tag{6}$$

**③ OOD 노이즈 샘플 ($\mathbb{S}_{ood}$)**: 균일 분포에 가까운 의사 레이블 (스케일 상수 $s=10$ 사용)

$$\tilde{y}_i^c = \frac{e^{p^c(x_i, \theta_{mt})/s}}{\sum_{j=1}^{C} e^{p^j(x_i, \theta_{mt})/s}} \tag{7}$$

**Mean-Teacher 파라미터 업데이트** (EMA):

$$\theta_{mt} \leftarrow \omega\theta_{mt} + (1 - \omega)\theta \tag{8}$$

---

#### **Step 4: 일관성 정규화 손실 (Consistency Regularization)**

$$\mathcal{L}_o = \frac{1}{N}\sum_{i=1}^{N} \rho_i \left(D_{KL}(\mathbf{p}_i \| \mathbf{p}'_i) + D_{KL}(\mathbf{p}'_i \| \mathbf{p}_i)\right) \tag{9}$$

$$\rho_i = \begin{cases} +1, & x_i \in \mathbb{S}_{clean} \cup \mathbb{S}_{id} \\ -1, & x_i \in \mathbb{S}_{ood} \end{cases}$$

- 클린 및 ID 샘플: 두 뷰의 예측 일관성 **강화**
- OOD 샘플: 두 뷰의 예측 불일치 **증가** → ID/OOD 분리 용이

---

#### **Step 5: 최종 결합 손실 함수**

$$\mathcal{L} = (1 - \alpha)\mathcal{L}_c + \alpha\mathcal{L}_o \tag{10}$$

$$\mathcal{L}_c = \frac{1}{N}\sum_{i=1}^{N}\left(-\sum_{c=1}^{C}\tilde{y}_i^c \log(p_i^c) - \sum_{c=1}^{C}\tilde{y}_i^c \log(p_i^{\prime c})\right) \tag{11}$$

---

### 2.3 모델 구조

```
입력 이미지 x_i
    ↓ (두 가지 증강)
  v_i (T(x_i))        v'_i (T'(x_i))
    ↓                      ↓
  공유 백본 네트워크 (shared weights)
    ↓                      ↓
  p_i (softmax)       p'_i (softmax)
    ↓
  ┌─────────────────────────┐
  │     Sample Selection     │
  │  JSD → P_clean          │
  │  Prediction agreement    │
  │  → P_ood                │
  └─────────────────────────┘
    ↓
  S_clean / S_id / S_ood 분류
    ↓
  레이블 재할당 (Eq.5/6/7)
    ↓
  Mean-Teacher (EMA 업데이트)
    ↓
  Joint Loss L = (1-α)L_c + αL_o
```

---

### 2.4 성능 향상

#### 합성 데이터셋 (CIFAR100N-C, CIFAR80N-O)

**CIFAR100N-C 결과 (정확도 %)**:

| 노이즈 유형-비율 | Standard | Co-teaching | JoCoR | **Jo-SRC** |
|---|---|---|---|---|
| Symmetry-20% | 35.14 | 43.73 | 53.01 | **58.15** |
| Symmetry-50% | 16.97 | 34.96 | 43.49 | **51.26** |
| Symmetry-80% | 4.41 | 15.15 | 15.49 | **23.80** |
| Asymmetry-40% | 27.29 | 28.35 | 32.70 | **38.52** |

**CIFAR80N-O 결과 (정확도 %)**:

| 노이즈 유형-비율 | Standard | Co-teaching | JoCoR | **Jo-SRC** |
|---|---|---|---|---|
| Symmetry-20% | 29.37 | 60.38 | 59.99 | **65.83** |
| Symmetry-50% | 13.87 | 52.42 | 50.61 | **58.51** |
| Symmetry-80% | 4.20 | 16.59 | 12.85 | **29.76** |
| Asymmetry-40% | 22.25 | 42.42 | 39.37 | **53.03** |

#### 실세계 데이터셋

| 데이터셋 | 최고 기존 방법 | **Jo-SRC** | 향상 |
|---|---|---|---|
| Clothing1M (ResNet-50) | DivideMix: 74.76% | **75.93%** | +1.17% |
| Food101N (ResNet-50) | DeepSelf: 85.11% | **86.66%** | +1.55% |

---

### 2.5 한계

1. **이진 분류 기반 OOD 탐지**: $\mathcal{P}_{ood}$가 argmax의 일치/불일치만 확인하는 단순한 이진 판단으로, 예측 확률의 분포 정보를 충분히 활용하지 못합니다.
2. **하이퍼파라미터 민감성**: $\tau_{clean}$, $\tau_{ood}$, $\alpha$, $\epsilon$, $\omega$ 등 다수의 하이퍼파라미터가 존재하며, 이를 최적 설정하는 데 어려움이 있습니다.
3. **높은 노이즈 비율에서의 성능 저하**: Symmetry-80%에서 Jo-SRC도 여전히 낮은 정확도(CIFAR100N-C: 23.80%)를 보입니다.
4. **단일 네트워크 의존**: 기존의 Co-teaching처럼 두 네트워크 간 상호 교정이 아닌 단일 네트워크와 EMA 기반 mean-teacher를 사용하여, 네트워크 편향(confirmation bias) 문제가 잠재합니다.
5. **계산 비용**: 각 샘플에 두 번의 증강 및 순전파(forward pass)가 필요하므로, 계산 비용이 증가합니다.

---

## 3. 모델 일반화 성능 향상 가능성 (중점 분석)

Jo-SRC의 일반화 성능 향상은 **세 가지 메커니즘**의 상호작용에서 비롯됩니다.

### 3.1 Label Smoothing Regularization (LSR)

클린 샘플에 Eq.(5)의 LSR을 적용하면:

$$\tilde{y}_i^c = \begin{cases} 1 - \epsilon, & c = l_i \\ \dfrac{\epsilon}{C-1}, & c \neq l_i \end{cases}$$

모델이 특정 클래스에 과도하게 확신하는 것을 방지하여, 과적합을 줄이고 **캘리브레이션(calibration)** 성능을 향상시킵니다. 실험에서 $\epsilon = 0.6$으로 상당히 큰 값을 사용하는 것이 특징입니다.

### 3.2 일관성 정규화 손실의 자기지도학습 효과

$\mathcal{L}_o$ 식(9)의 일관성 손실은:

- 동일 샘플의 두 증강 뷰가 유사한 표현을 가지도록 유도 → **자기지도학습(self-supervised learning)** 효과
- 이는 SimCLR, BYOL 등 대조학습 방법론과 유사한 원리로, **도메인 불변 표현(domain-invariant representation)** 학습을 촉진

논문의 Table 6 (ablation study)에서 이 효과를 확인할 수 있습니다:

| 모델 | 정확도 (%) |
|---|---|
| Jo-SRC-CIO (일관성 손실 없음) | 63.10 |
| **Jo-SRC (일관성 손실 포함)** | **65.83** |

→ 일관성 정규화만으로 **+2.73%p** 향상

### 3.3 ID 노이즈 샘플 활용

Ablation study Table 6:

| 모델 | 정확도 (%) |
|---|---|
| Jo-SRC-C (클린 샘플만) | 57.12 |
| Jo-SRC-CI (클린 + ID) | 61.32 |
| Jo-SRC-CIO (클린 + ID + OOD) | 63.10 |
| Jo-SRC (전체) | 65.83 |

ID 노이즈 샘플을 mean-teacher로 재레이블링하여 활용하면 **+4.20%p** 향상되며, OOD 샘플 활용 시 추가 **+1.78%p** 향상됩니다.

### 3.4 OOD 샘플을 균일 분포로 학습

OOD 샘플에 Eq.(7)의 근사 균일 분포 레이블을 부여하는 것은:
- 모델이 OOD 샘플에 대해 낮은 확신도(low confidence)를 갖도록 유도
- 이는 **정규화 효과**로 작용하여 일반화 성능 향상에 기여
- Open-set recognition 관점에서도 의미 있는 접근

### 3.5 동적 임계값 $\tau_{clean}$ 전략

학습 초기: 낮은 $\tau_{clean}$ → 많은 샘플 사용 → 다양한 패턴 학습  
학습 후기: 높은 $\tau_{clean}$ → 고품질 샘플만 사용 → 과적합 방지

이는 커리큘럼 학습(Curriculum Learning)의 원리와 유사하며, **학습 단계별 적응적 정규화**를 구현합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

#### (1) 전역 샘플 선택 패러다임의 확립
Jo-SRC는 미니배치 단위 선택에서 **전역(global) 선택**으로의 패러다임 전환을 제안합니다. 이후 연구들이 더 정교한 전역 선택 기준을 탐구하는 계기가 됩니다.

#### (2) 대조학습과 노이즈 레이블 대응의 융합
자기지도 대조학습(SimCLR, BYOL 등)의 원리를 노이즈 레이블 학습에 접목한 선구적 연구로, 이후 **대조학습 기반 노이즈 견고성 연구** 흐름을 형성합니다.

#### (3) 오픈셋 노이즈 시나리오 정의
ID/OOD 노이즈를 명시적으로 구분하고 각각에 맞는 처리를 제안함으로써, **오픈셋 노이즈 학습(open-set noisy label learning)** 연구 분야의 정립에 기여합니다.

#### (4) Mean-Teacher의 노이즈 학습 적용
Semi-supervised learning의 mean-teacher 모델을 노이즈 레이블 문제에 효과적으로 적용한 사례로, 이후 연구들의 벤치마크가 됩니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

#### DivideMix (Li et al., ICLR 2020)

| 특성 | DivideMix | Jo-SRC |
|---|---|---|
| 클린 샘플 선택 | GMM 기반 손실 분포 모델링 | JSD 기반 전역 선택 |
| 노이즈 처리 | Semi-supervised MixMatch | 재레이블링 + 결합 손실 |
| OOD 처리 | 별도 처리 없음 | ID/OOD 명시적 구분 |
| 네트워크 수 | 2개 (co-training) | 1개 + mean-teacher |
| Clothing1M 정확도 | 74.76% | 75.93% |

DivideMix는 GMM으로 손실 분포를 모델링하는 반면, Jo-SRC는 JSD를 통해 더 직관적인 전역 기준을 제공합니다.

#### CORES² (Cheng et al., NeurIPS 2021)

Jo-SRC 이후에 등장한 연구로, 샘플 선택의 신뢰성을 높이기 위해 **확인 편향(confirmation bias)** 문제를 집중적으로 다루었습니다. Jo-SRC의 단일 네트워크 의존성 문제를 부분적으로 해결하는 방향으로 발전하였습니다.

#### Sel-CL (Li et al., CVPR 2022)

대조학습을 노이즈 레이블 학습에 더 심층적으로 통합한 연구로, Jo-SRC의 대조학습 아이디어를 발전시킨 사례입니다. 인스턴스 레벨 대조학습을 통해 더 풍부한 표현을 학습합니다.

> **주의**: CORES²와 Sel-CL에 대한 정확한 수치 비교는 논문 원문을 직접 확인하시기 바랍니다. Jo-SRC 논문(2021년 3월)은 이들 연구보다 먼저 출판되었습니다.

---

### 4.3 향후 연구 시 고려할 점

#### (1) OOD 탐지의 고도화
현재의 argmax 일치 여부 기반 $\mathcal{P}_{ood}$는 매우 단순합니다. 향후 연구에서는:
- **에너지 기반(energy-based)** OOD 탐지
- **Mahalanobis distance** 활용
- **Bayesian uncertainty** 추정
등을 통해 더 정교한 OOD 구분이 가능합니다.

#### (2) 하이퍼파라미터 자동 조정
$\tau_{clean}$, $\alpha$, $\epsilon$ 등의 하이퍼파라미터는 데이터셋과 노이즈 유형에 민감합니다. **AutoML** 또는 **메타학습(meta-learning)** 기반의 자동 조정 메커니즘 연구가 필요합니다.

#### (3) 대규모 데이터셋에서의 확장성
현재 Clothing1M(~1M 이미지)까지는 검증되었으나, 더 대규모 데이터셋(수억 장)에서의 전역 선택 전략의 확장성을 검토해야 합니다.

#### (4) 다중 모달 및 다양한 도메인 적용
Jo-SRC는 주로 이미지 분류에 초점을 맞추고 있습니다. **텍스트, 오디오, 의료 영상** 등 다양한 도메인에서의 적용 가능성을 탐구할 필요가 있습니다.

#### (5) 확인 편향(Confirmation Bias) 문제
단일 네트워크와 mean-teacher를 사용할 경우, 초기 잘못된 예측이 의사 레이블을 통해 강화될 위험이 있습니다. 이를 해소하기 위한 **다중 뷰 앙상블** 또는 **불확실성 추정 기반 접근**이 필요합니다.

#### (6) 이론적 보장 부재
JSD 기반 전역 선택과 OOD 탐지에 대한 **이론적 수렴 보장** 및 **일반화 경계(generalization bound)** 분석이 부족합니다. 이에 대한 이론적 기반 연구가 요구됩니다.

#### (7) 증강 전략의 민감성
두 뷰 생성에 사용되는 증강 $T(\cdot)$와 $T'(\cdot)$의 선택이 OOD 탐지 성능에 큰 영향을 미칩니다. **증강 불변성(augmentation invariance)**과 OOD 탐지 간의 트레이드오프를 체계적으로 분석할 필요가 있습니다.

---

## 참고 자료

### 주요 참고 논문 (Jo-SRC 논문 내 인용)

1. **Jo-SRC 원본 논문**: Yazhou Yao et al., "Jo-SRC: A Contrastive Approach for Combating Noisy Labels," arXiv:2103.13029v1, 2021.
2. **DivideMix**: Junnan Li, Richard Socher, Steven CH Hoi, "DivideMix: Learning with Noisy Labels as Semi-Supervised Learning," ICLR, 2020.
3. **JoCoR**: Hongxin Wei, Lei Feng, Xiangyu Chen, Bo An, "Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization," CVPR, 2020.
4. **Co-teaching**: Bo Han et al., "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels," NeurIPS, 2018.
5. **SimCLR**: Ting Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations," arXiv:2002.05709, 2020.
6. **Mean Teacher**: Antti Tarvainen, Harri Valpola, "Mean Teachers are Better Role Models," NeurIPS, 2017.
7. **Memorization Effect**: Chiyuan Zhang et al., "Understanding Deep Learning Requires Rethinking Generalization," ICLR, 2017.
8. **LSR**: Christian Szegedy et al., "Rethinking the Inception Architecture for Computer Vision," CVPR, 2016.
9. **CRSSC**: Zeren Sun et al., "CRSSC: Salvage Reusable Samples from Noisy Data for Robust Learning," ACM MM, 2020.
10. **P-correction**: Kun Yi, Jianxin Wu, "Probabilistic End-to-end Noise Correction for Learning with Noisy Labels," CVPR, 2019.

> **면책 고지**: 2020년 이후 최신 연구(CORES², Sel-CL 등)의 세부 수치 비교는 Jo-SRC 논문 원문에 포함되지 않은 내용이므로, 개략적 특성 비교만 제공하였습니다. 정확한 수치는 해당 논문 원문을 직접 확인하시기 바랍니다.
