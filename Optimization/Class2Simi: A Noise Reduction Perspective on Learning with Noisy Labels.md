# Class2Simi: A Noise Reduction Perspective on Learning with Noisy Labels 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Class2Simi는 **노이즈 레이블 학습 문제를 "노이즈 감소(Noise Reduction)"의 관점**에서 접근합니다. 기존의 포인트와이즈(pointwise) 방식 대신, **데이터 포인트의 클래스 레이블을 데이터 쌍(pair)의 유사도 레이블(similarity label)로 변환**함으로써 노이즈 비율을 이론적으로 감소시킬 수 있음을 증명합니다.

> "Does learning in a pairwise manner mitigate label noise?" → **Yes**

### 주요 기여 (4가지)

| 기여 | 내용 |
|------|------|
| ① 노이즈 감소 관점 제시 | 클래스 레이블 → 유사도 레이블 변환으로 노이즈 비율 감소 |
| ② 유사도 전이 행렬 추정 | 클래스 전이 행렬 $T_c$로부터 $T_s$ 이론적 도출 |
| ③ 일반화 오차 분석 | 생성된 분류기의 일반화 오차 상한 이론 증명 |
| ④ 실험적 검증 | 합성 노이즈 및 실제 데이터셋에서 우수한 성능 입증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

딥러닝 모델은 노이즈 레이블이 있는 대규모 데이터셋에서 **레이블 노이즈를 그대로 암기(memorization)** 하는 경향이 있어 성능이 저하됩니다 (Zhang et al., 2017). 기존 방법들은 모두 **포인트와이즈 방식** (샘플 선택, 손실 보정, 레이블 보정 등)에 의존합니다.

**문제 형식화:**

- 인스턴스-클래스 쌍: $(X, Y) \in \mathcal{X} \times \{1, \ldots, c\}$
- 실제 관측 가능한 데이터: $\{(x_1, \bar{y}_1), \ldots, (x_n, \bar{y}_n)\}$ — 노이즈 레이블 포함
- 목표: 노이즈 분포 $\mathcal{D}_\rho$에서 학습하여 clean label에 대해 일반화되는 분류기 학습

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 클래스 → 유사도 레이블 변환

두 인스턴스 $(x_i, x_j)$에 대해 유사도 레이블을 다음과 같이 정의합니다:

$$H_{ij} = \begin{cases} 1 & \text{if } y_i = y_j \quad \text{(similar)} \\ 0 & \text{if } y_i \neq y_j \quad \text{(dissimilar)} \end{cases}$$

노이즈 클래스 레이블로 생성된 유사도 레이블은 $\bar{H}_{ij}$로 표기되며, 이 역시 노이즈를 포함합니다.

---

#### Step 2: 유사도 전이 행렬 도출 (Theorem 1)

**[Theorem 1]** 균형 잡힌 데이터셋( $c$개 클래스, 각 클래스 동일 수)과 클래스 의존 노이즈를 가정할 때, 클래스 전이 행렬 $T_c$ (여기서 $T_{c,ij} = P(\bar{Y}=j|Y=i)$ )로부터 유사도 전이 행렬 $T_s$의 원소는 다음과 같이 계산됩니다:

$$T_{s,00} = \frac{c^2 - c - \left(\sum_j \left(\sum_i T_{c,ij}\right)^2 - \|T_c\|^2_{\text{Fro}}\right)}{c^2 - c}$$

$$T_{s,01} = \frac{\sum_j \left(\sum_i T_{c,ij}\right)^2 - \|T_c\|^2_{\text{Fro}}}{c^2 - c}$$

$$T_{s,10} = \frac{c - \|T_c\|^2_{\text{Fro}}}{c}, \quad T_{s,11} = \frac{\|T_c\|^2_{\text{Fro}}}{c}$$

여기서 $\|T_c\|^2_{\text{Fro}}$는 클래스 전이 행렬의 Frobenius norm의 제곱입니다.

---

#### Step 3: 노이즈 유사도 레이블로부터 학습

클래스 사후 확률과 유사도 사후 확률의 관계를 다음과 같이 정의합니다:

$$\hat{S}_{ij} = f(X_i)^\top f(X_j)$$

노이즈 유사도 사후 확률과 클린 유사도 사후 확률의 관계:

$$P(\bar{H}_{ij} | X_i, X_j) = T_s^\top P(H_{ij} | X_i, X_j) \tag{1}$$

이를 통해 예측된 노이즈 유사도 사후 확률 $\hat{\bar{S}}\_{ij}$를 클린 유사도 사후 확률 $\hat{S}_{ij}$로부터 추정하고, 최적화 함수는 다음과 같이 정의됩니다:

$$\mathcal{L}_{c2s}(\bar{H}_{ij}, \hat{\bar{S}}_{ij}) = -\sum_{i,j} \left[ \bar{H}_{ij} \log \hat{\bar{S}}_{ij} + (1 - \bar{H}_{ij}) \log(1 - \hat{\bar{S}}_{ij}) \right]$$

---

#### Step 4: 노이즈율 감소 보장 (Theorem 2)

**[Theorem 2]** 균형 데이터셋과 클래스 의존 노이즈를 가정할 때, **클래스 수 $c \geq 8$이면** 노이즈 유사도 레이블의 노이즈 비율은 노이즈 클래스 레이블의 노이즈 비율보다 낮습니다.

예시: 클래스 노이즈율 0.5 → 유사도 노이즈율 0.25 (Figure 1 기준)

---

### 2.3 모델 구조

```
입력 (노이즈 클래스 레이블 포함 미니배치)
        ↓
[Stage 1] Neural Network (e.g., ResNet) + Softmax
  → g(X) = P̂(Ȳ|X) [노이즈 클래스 사후 확률]
  → T̂_c 추정 → T̂_s 계산
        ↓
[Stage 2] Neural Network + Softmax
  → f(X) = P̂(Y|X) [클린 클래스 사후 확률]
        ↓
[Pairwise Enumeration Layer]
  → Ŝ_ij = f(X_i)ᵀ f(X_j)
        ↓
[Similarity Transition Matrix Layer]
  → T_s^⊤ [Ŝ_ij, 1-Ŝ_ij]^⊤ = [S̄̂_ij, 1-S̄̂_ij]^⊤
        ↓
[Binary Cross-Entropy Loss: L_c2s]
```

**두 단계 알고리즘 (Algorithm 1):**

- **Stage 1:** 노이즈 클래스 레이블로 $g(X) = \hat{P}(\bar{Y}|X)$ 학습 → $\hat{T}_c$ 추정 → $\hat{T}_s$ 변환
- **Stage 2:** Stage 1의 모델 가중치를 로드하여 $\mathcal{L}_{c2s}$로 파인튜닝

**사용 백본:**
- MNIST: LeNet
- CIFAR-10: ResNet-26 (shake-shake regularization)
- CIFAR-100: ResNet-56 (pre-activation)
- News20: 3-layer MLP + GloVe
- Clothing1M*: Pre-trained ResNet-50

---

### 2.4 성능 향상

#### 합성 노이즈 (Symmetric/Asymmetric)

**CIFAR-10 (Sym-0.6 기준):**

| 방법 | 정확도 |
|------|--------|
| Co-teaching | 75.97% |
| Forward | 73.24% |
| Revision | 73.92% |
| **F-Class2Simi** | **79.45%** |

**CIFAR-100 (Sym-0.6 기준):**

| 방법 | 정확도 |
|------|--------|
| Co-teaching | 37.32% |
| Forward | 27.01% |
| Revision | 35.82% |
| **F-Class2Simi** | **40.38%** |

- CIFAR-10에서 약 **+5%p**, CIFAR-100에서 약 **+10%p** 향상 (노이즈율 0.6 기준)

#### 실제 데이터셋 (Clothing1M*)

| 방법 | 정확도 |
|------|--------|
| Co-teaching | 74.70% |
| Forward | 73.88% |
| **R-Class2Simi** | **75.76%** |

---

### 2.5 한계점

1. **정보 손실:** 유사도 레이블만으로는 클래스의 의미적 정보(semantic class identity)를 복원할 수 없음 → Stage 1 사전학습으로 보완하나 완전하지 않음

2. **클래스 수 제한:** Theorem 2는 $c \geq 8$일 때만 이론적으로 노이즈율 감소를 보장 (적은 클래스 수에서는 보장 없음)

3. **전이 행렬 추정 의존성:** $T_c$ 추정의 정확도에 성능이 영향받음. 단, Figure 4에서 $T_s$가 추정 오차에 상대적으로 강건함을 보임

4. **클러스터 혼동 문제:** Clothing1M에서 클래스 5와 클래스 3처럼 다수의 인스턴스가 다른 클래스에 속하는 경우 클러스터링이 실패할 수 있음 → 클래스 병합(Clothing1M*)으로 임시 해결

5. **계산 비용:** 미니배치 내 페어와이즈 연산으로 인해 배치 크기에 이차적(quadratic) 복잡도 발생 (단, 저자는 무시할 수 있는 오버헤드라 주장)

6. **인스턴스 의존 노이즈(Instance-Dependent Noise):** 클래스 의존 노이즈 가정에 기반하므로 보다 복잡한 노이즈 패턴에는 제한적

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 일반화 오차 분석 (Theorem 3)

**[Theorem 3]** $d$층 신경망의 파라미터 행렬 $W_1, \ldots, W_d$의 Frobenius norm이 각각 $M_1, \ldots, M_d$ 이하이고, 활성화 함수가 1-Lipschitz, positive-homogeneous, element-wise(ReLU 등)이며, 인스턴스 $X$가 $\|X\| \leq B$를 만족하고 손실함수가 $M$으로 상한될 때, 임의의 $\delta > 0$에 대해 확률 $1-\delta$ 이상으로:

$$R(\hat{f}) - R_n(\hat{f}) \leq M\sqrt{\frac{\log 1/\delta}{2n}} + \frac{(T_{s,11} - T_{s,01}) \cdot 2Bc(\sqrt{2d\log 2} + 1) \prod_{i=1}^d M_i}{T_{s,11}\sqrt{n}} \tag{2}$$

**핵심 해석:**

- **첫 번째 항** $M\sqrt{\frac{\log 1/\delta}{2n}}$: 표준 통계적 일반화 항 (샘플 수 $n$이 증가하면 감소)
- **두 번째 항**: 유사도 전이 행렬 $T_s$와 네트워크 복잡도에 의존하는 항

**일반화 향상 메커니즘:**

$$\frac{T_{s,11} - T_{s,01}}{T_{s,11}}$$

이 비율이 클수록 (즉, $T_{s,11} \gg T_{s,01}$) 일반화 오차 상한이 작아집니다. 이는 유사도 레이블의 정확도가 높을수록 일반화가 향상됨을 의미합니다.

### 3.2 일반화 향상의 실질적 원인

1. **노이즈율 감소 효과:** Theorem 2에 의해 유사도 레이블의 노이즈율이 클래스 레이블보다 낮으며, 낮은 노이즈율은 알고리즘의 일관성(consistency)과 성능을 향상시킴 (Patrini et al., 2017)

2. **이진 분류로의 축소:** 다중 클래스 문제를 이진 분류 문제로 변환함으로써 학습 난이도 감소

3. **인스턴스 의존 노이즈의 클래스 의존 노이즈 근사:** 노이즈율이 낮을수록 복잡한 인스턴스 의존 노이즈가 다루기 쉬운 클래스 의존 노이즈로 잘 근사됨 (Cheng et al., 2020)

4. **전이 행렬 강건성:** Figure 4에서 Forward 방법은 $T_c$ 추정 오차에 따라 정확도가 급격히 하락하는 반면, F-Class2Simi는 미미한 변동만 보임 → 실제 환경에서 더 높은 일반화 기대 가능

5. **메타 방법으로서의 유연성:** Class2Simi는 기존의 Sample Selection, Loss Correction, Label Correction 방법들과 결합 가능 → 다양한 시나리오에서 일반화 성능 향상 가능

### 3.3 Ablation Study에서의 인사이트

Table 4 (클린 데이터셋 실험)에서 유사도 손실함수 단독으로는 클린 데이터의 성능을 크게 향상시키지 않습니다. 이는 **일반화 향상이 유사도 손실 자체보다는 낮아진 노이즈율과 이진 분류 패러다임 전환에서 비롯됨**을 명확히 보여줍니다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

**① 페어와이즈 학습 패러다임의 확장**

Class2Simi는 노이즈 레이블 학습에서 페어와이즈 접근법의 가능성을 열었습니다. 이는 자기지도학습(Self-Supervised Learning), 대조학습(Contrastive Learning)과의 융합 연구를 자극할 수 있습니다. 실제로 SimCLR (Chen et al., 2020), MoCo (He et al., 2020) 등의 대조학습 프레임워크와의 시너지 가능성이 높습니다.

**② 노이즈 레이블 학습의 이론적 토대 강화**

Theorem 1~3을 통한 엄밀한 이론적 분석은 향후 노이즈 감소 알고리즘 설계에 있어 수학적 근거를 제공합니다. 특히 유사도 전이 행렬과 클래스 전이 행렬 간의 관계는 다양한 확장 연구에서 활용될 수 있습니다.

**③ 전이 행렬 추정 연구 촉진**

$T_c \to T_s$ 변환 과정에서 전이 행렬 추정의 중요성이 부각되어, 보다 정확하고 강건한 전이 행렬 추정 방법 연구를 촉진할 것입니다 (예: Dual-T (Yao et al., 2020), Zhang et al., 2021).

**④ 메타 프레임워크로서의 활용**

Class2Simi가 기존 방법들(Forward, Reweight 등) 위에 적용 가능한 메타 방법임을 보인 것은, 모듈러 방식의 노이즈 레이블 학습 프레임워크 설계에 영감을 줍니다.

---

### 4.2 향후 연구 시 고려할 점

**① 인스턴스 의존 노이즈(Instance-Dependent Noise) 확장**

현재 Class2Simi는 클래스 의존 노이즈를 주로 다룹니다. 실제 데이터에서 빈번한 인스턴스 의존 노이즈(Xia et al., 2020d; Zhu et al., 2021)로의 확장이 필요합니다. 이를 위해 인스턴스별 전이 행렬 추정과 페어와이즈 변환을 결합하는 연구가 필요합니다.

**② 대조학습과의 결합**

SimCLR, MoCo 등의 대조학습 프레임워크와 Class2Simi를 결합하여 데이터 증강 기반의 페어 생성과 노이즈 감소를 동시에 달성하는 방향이 유망합니다.

**③ 클래스 불균형 문제**

Theorem 1에서 균형 데이터셋을 가정하나 실제 데이터는 불균형합니다. Remark 1에서 불균형 확장을 언급하지만 실험적 검증이 부족하므로, 클래스 불균형과 노이즈 레이블이 동시에 존재하는 환경에서의 연구가 필요합니다.

**④ 더 적은 클래스 수에서의 적용**

Theorem 2의 $c \geq 8$ 제약을 완화하거나, 소수 클래스(예: 이진 분류)에서도 노이즈율 감소를 보장하는 이론적 확장이 필요합니다.

**⑤ 전이 행렬 없이 작동하는 확장**

$T_c$ 추정이 불가능하거나 부정확한 경우를 위해, Li et al. (2021)의 앵커 포인트 없는 방법론과 같이 전이 행렬 없이 동작하는 Class2Simi 변형 연구가 필요합니다.

**⑥ 개방형 세계(Open-Set) 노이즈 적용**

Clothing1M 사례처럼 실제 데이터에서는 Out-of-Distribution 클래스 레이블이 존재할 수 있습니다. 개방형 세계 노이즈(Xia et al., 2020a)와 Class2Simi의 결합이 중요한 연구 방향입니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 방법론 | Class2Simi와의 관계 | 주요 차이점 |
|------|--------|-------------------|------------|
| **Dual-T** (Yao et al., NeurIPS 2020) | 전이 행렬 이중 추정 | 전이 행렬 추정 개선 | $T_c$ 추정 오차 감소에 집중; 페어와이즈 변환 없음 |
| **DivideMix** (Li et al., ICLR 2020) | GMM 기반 샘플 분리 + MixUp | 샘플 선택 방식 | 반지도학습 결합; Class2Simi와 결합 가능성 높음 |
| **ELR** (Liu & Guo, NeurIPS 2020) | 조기 학습 정규화 | 정규화 관점 | 명시적 전이 행렬 불필요; 클래스 의존 노이즈 처리 방식 상이 |
| **CORES²** (Cheng et al., ICML 2021) | 신뢰 점수 기반 샘플 선택 | 인스턴스 의존 노이즈 처리 | 더 복잡한 노이즈 모델 처리 가능 |
| **Provably end-to-end** (Li et al., ICML 2021) | 앵커 포인트 없는 전이 행렬 학습 | Class2Simi의 Stage 1 개선 가능성 | $T_c$ 추정을 앵커 포인트 없이 수행 |
| **UNICON** (Karim et al., CVPR 2022) | 대조학습 + 균일 샘플 선택 | 대조학습 결합 관점 | 자기지도학습 활용; 유사도 관계 암묵적 사용 |
| **SOP** (Liu et al., ICML 2022) | 과최적화 방지 정규화 | 손실 함수 관점 | 전이 행렬 불필요; 더 일반적 적용 가능 |

**핵심 비교 인사이트:**

Class2Simi는 **이론적 노이즈 감소 보장**이라는 독특한 관점을 제공하는 반면, 최근 연구들은 대조학습, 반지도학습, 그래프 기반 방법 등 더 다양한 방향으로 발전하고 있습니다. Class2Simi의 메타 방법적 특성을 활용하여 DivideMix나 대조학습 기반 방법들과 결합하는 것이 특히 유망한 방향입니다.

---

## 참고 자료

- **주 논문:** Wu, S., Xia, X., Liu, T., Han, B., Gong, M., Wang, N., Liu, H., & Niu, G. (2021). *Class2Simi: A Noise Reduction Perspective on Learning with Noisy Labels*. ICML 2021. arXiv:2006.07831v2
- Patrini, G., et al. (2017). *Making deep neural networks robust to label noise: A loss correction approach*. CVPR 2017.
- Han, B., et al. (2018). *Co-teaching: Robust training of deep neural networks with extremely noisy labels*. NeurIPS 2018.
- Hsu, Y.-C., et al. (2019). *Multi-class classification without multi-class labels*. ICLR 2019.
- Yao, Y., et al. (2020). *Dual T: Reducing estimation error for transition matrix in label-noise learning*. NeurIPS 2020.
- Li, X., et al. (2021). *Provably end-to-end label-noise learning without anchor points*. ICML 2021.
- Cheng, J., et al. (2020). *Learning with bounded instance- and label-dependent label noise*. ICML 2020.
- Zhang, Y., Niu, G., & Sugiyama, M. (2021). *Learning noise transition matrix from only noisy labels via total variation regularization*. ICML 2021.
- Zhu, Z., Liu, T., & Liu, Y. (2021). *A second-order approach to learning with instance-dependent label noise*. CVPR 2021.
- Mohri, M., Rostamizadeh, A., & Talwalkar, A. (2018). *Foundations of Machine Learning*. MIT Press.
