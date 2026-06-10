# Training Binary Neural Networks through Learning with Noisy Supervision

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 Binary Neural Network(BNN)의 이진화 과정을 **학습 가능한 관점(learning perspective)**에서 재정의합니다. 기존의 수작업 규칙(hard thresholding, sign 함수)으로 가중치를 이진화하는 방식 대신, **전체 필터의 가중치를 하나의 입력으로 취급하는 매핑 함수(mapping function)**를 신경망으로 학습합니다. 이 과정에서 sign 함수로 얻은 이진 가중치를 **노이즈가 포함된 레이블(noisy supervision)**로 활용하며, 이 노이즈의 영향을 완화하기 위한 **비편향 추정기(unbiased estimator)**를 도입합니다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **새로운 BNN 학습 패러다임** | sign 함수 대신 학습 가능한 신경망 매핑으로 이진화 수행 |
| **노이즈 레이블 학습 접목** | BNN 학습을 noisy label learning 문제로 형식화 |
| **비편향 추정기 도입** | 이론적으로 수렴 보장이 있는 loss correction 방법 제안 |
| **일관된 성능 향상** | CIFAR-10, ImageNet에서 SOTA 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 BNN의 두 가지 핵심 문제점:

**문제 1: 독립적 이진화의 한계**
- 기존 방법은 각 가중치 원소를 독립적으로 이진화 → 원소 간 관계(inter-neuron relationship) 무시
- 수식으로 표현하면 기존 이진화:

$$\tilde{Q} = \text{sign}(W) \tag{1}$$

여기서 $W \in \mathbb{R}^{c \times k \times k}$이며, 각 원소가 개별적으로 처리됩니다.

**문제 2: STE(Straight Through Estimator)의 부정확한 기울기**
- 역전파 시 STE를 사용:

$$\frac{\partial \ell_{cls}}{\partial W} \approx \text{clip}\left(\frac{\partial \ell_{cls}}{\partial \tilde{Q}}, -1, 1\right) \tag{3}$$

- 이 추정된 기울기로 인해 일부 이진 가중치가 잘못된 값(+1 ↔ -1)으로 뒤집히는 노이즈 발생

---

### 2.2 제안 방법 (수식 포함)

#### Step 1: Binary Weight Mapping (이진 가중치 매핑)

필터 전체를 입력으로 받는 학습 가능한 매핑 함수 도입:

$$\hat{Q} = f_\theta(W) \tag{4}$$

여기서 $f_\theta$는 파라미터 $\theta$를 가진 신경망(3층 CNN)입니다. sign 함수와 달리, 필터 내 모든 원소의 관계를 종합하여 이진화합니다.

#### Step 2: 이상적 손실 함수 (Clean Label 기준)

Ground-truth 이진 가중치 $Q$가 있다면:

$$\ell(\hat{Q}, Q) = \|\hat{Q} - Q\|_F^2 = \sum(\hat{q} - q)^2 \tag{5, 6}$$

#### Step 3: Noisy Supervision (노이즈 레이블 활용)

실제로 $Q$는 알 수 없으므로, 사전 학습된 이진 모델의 가중치 $\tilde{Q}$(noisy label)를 사용:

$$\ell(\hat{Q}, \tilde{Q}) = \|\hat{Q} - \tilde{Q}\|_F^2 = \sum(\hat{q} - \tilde{q})^2 \tag{7}$$

그러나 이 손실은 노이즈에 의해 편향되므로 보정이 필요합니다.

#### Step 4: Class-Conditional Noise Model (노이즈 전이 모델)

노이즈 레이블이 다음 확률로 뒤집힌다고 가정:

$$P(\tilde{q} = -1 \mid q = +1) = \rho_{+1} \tag{8}$$
$$P(\tilde{q} = +1 \mid q = -1) = \rho_{-1} \tag{9}$$

BNN에서 양수/음수 가중치의 수가 유사하므로, $\rho = \rho_{+1} = \rho_{-1}$로 단순화합니다.

#### Step 5: Unbiased Loss Correction (비편향 손실 보정)

다음 조건을 만족하는 보정된 손실 $\tilde{\ell}$을 구합니다:

$$\mathbb{E}\left[\tilde{\ell}(\hat{Q}, \tilde{Q})\right] = \ell(\hat{Q}, Q) \tag{10}$$

연립방정식을 풀면:

$$\tilde{\ell}(\hat{q}, +1) = \frac{(1-\rho_{-1})\ell(\hat{q},+1) - \rho_{+1}\ell(\hat{q},-1)}{1-\rho_{+1}-\rho_{-1}} \tag{13}$$

$$\tilde{\ell}(\hat{q}, -1) = \frac{(1-\rho_{+1})\ell(\hat{q},-1) - \rho_{-1}\ell(\hat{q},+1)}{1-\rho_{+1}-\rho_{-1}} \tag{14}$$

통합하면:

$$\boxed{\tilde{\ell}(\hat{q}, \tilde{q}) = \frac{(1-\rho_{-\tilde{q}})\ell(\hat{q},\tilde{q}) - \rho_{\tilde{q}}\ell(\hat{q},-\tilde{q})}{1-\rho_{+1}-\rho_{-1}}} \tag{15}$$

#### Step 6: $\hat{q}$에 대한 기울기

$$\frac{\partial \tilde{\ell}_i}{\partial \hat{q}} = 2(\hat{q} - \tilde{q}) - \frac{4\rho_{\tilde{q}}\tilde{q}}{1-\rho_{+1}-\rho_{-1}} \tag{18}$$

#### Step 7: 전체 목적 함수

$$\mathcal{L} = \ell_{cls} + \alpha \sum_i \tilde{\ell}_i \tag{21}$$

여기서 $\alpha$는 분류 손실과 보정 손실 간의 균형 하이퍼파라미터입니다.

---

### 2.3 모델 구조

```
입력 이미지
    │
    ▼
[잠재 가중치 W (full-precision)]
    │         │
    │         ▼
    │    [매핑 모델 f_θ: 3층 CNN]
    │     (2c×c×3×3 → 2c×2c×3×3 → c×2c×3×3)
    │         │ BN + ReLU
    │         ▼
    │    [예측 이진 가중치 Q_hat]
    │         │
    │    sign(W)→[노이즈 레이블 Q_tilde]
    │         │         │
    │         └─[보정 손실 ℓ_tilde]─┐
    │                               │
    ▼                               ▼
[이진 합성곱 Y = B ⊛ Q_hat]   [분류 손실 ℓ_cls]
    │                               │
    └───────────[전체 손실 L]───────┘
```

**매핑 모델 상세:**
- 3층 CNN으로 구성
- 가중치 크기: $2c \times c \times 3 \times 3$, $2c \times 2c \times 3 \times 3$, $c \times 2c \times 3 \times 3$
- Padding=1, Stride=1 (출력 크기 유지)
- 중간층: BN + ReLU

---

### 2.4 성능 향상 및 한계

#### 성능 향상

**CIFAR-10 (ResNet-20 기반):**

| 방법 | 정확도 |
|---|---|
| Dorefa-Net (Baseline) | 85.06% |
| Fine-tuning | 85.32% |
| LNS (매핑만, 노이즈 없음) | 85.43% |
| **LNS (Ours)** | **85.78% (85.56±0.11%)** |

**ImageNet (ResNet-18):**

| 방법 | Top-1 | Top-5 |
|---|---|---|
| Bireal-Net+PReLU (Baseline) | 59.0% | 81.3% |
| IR-Net | 58.1% | 80.0% |
| **LNS (Ours)** | **59.4%** | **81.7%** |

**ImageNet (AlexNet):**

| 방법 | Top-1 |
|---|---|
| XNOR-Net | 44.2% |
| **LNS (Ours)** | **44.4%** |

#### 한계점

1. **추가 파라미터**: 각 레이어에 매핑 모델($f_\theta$)이 추가되어 학습 파라미터 증가
2. **추론 시 비용**: 학습 후 이진 가중치만 사용하지만, 학습 단계에서의 비용이 큼
3. **하이퍼파라미터 민감성**: $\alpha$, $\rho$ 값에 따라 성능 변동 존재 (특히 $\rho$가 너무 크면 성능 저하)
4. **사전 학습 의존**: 반드시 사전 학습된 BNN 모델이 필요 (완전한 end-to-end 처음부터 학습 불가)
5. **노이즈율 추정의 어려움**: $\rho$를 데이터에서 자동으로 추정하는 메커니즘 없음

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문은 일반화 성능 향상에 여러 메커니즘을 통해 기여합니다.

### 3.1 이론적 수렴 보장 (Theorem 1)

논문은 Natarajan et al. (2013)의 정리를 활용하여 다음을 보장합니다:

**Theorem 1:** 확률 $1-\delta$ 이상으로:

$$R_{\ell,D}(\hat{f}) \leq \min_{f \in \mathcal{F}} R_{\ell,D}(f) + 4L_\rho \mathfrak{R}(\mathcal{F}) + 2\sqrt{\frac{\log(1/\delta)}{2n}}$$

여기서:
- $\mathfrak{R}(\mathcal{F})$: 함수 클래스 $\mathcal{F}$의 **Rademacher 복잡도**
- $L_\rho \leq \frac{2L}{1-\rho_{+1}-\rho_{-1}}$: 보정 손실의 Lipschitz 상수
- $n$: 샘플 수

이 정리는 **노이즈 레이블로 학습해도 깨끗한 분포 $D$ 하에서의 $\ell$-리스크에 수렴**함을 보장합니다.

### 3.2 과적합 방지 메커니즘

논문에서 실험적으로 관찰된 중요한 사실:

> *"our method can alleviate over-fitting by imposing a noisy supervision on each layer as the noisy weights are often those that over-fit the training data"*

즉, 노이즈 레이블로 학습하는 것이 **암묵적 정규화(implicit regularization)** 효과를 가집니다:

- **훈련 손실**: LNS가 단순 fine-tuning보다 약간 높음
- **테스트 정확도**: LNS가 단순 fine-tuning보다 높음
- → 훈련/테스트 갭이 더 작아지는 **일반화 향상 효과**

### 3.3 가중치 뒤집기 비율(Flip Rate) 분석

실험에서 LNS의 flip rate가 단순 fine-tuning보다 일관되게 낮음:
- LNS는 **소수의 핵심 가중치만 수정** → 과적합된 가중치만 선별적으로 교정
- 이는 더 안정적인 학습 과정을 만들며 일반화에 기여

### 3.4 뉴런 간 관계 활용

매핑 모델이 필터 내 **뉴런 간 상관관계를 학습**함으로써:
- 단순 sign 함수보다 더 **표현력 있는 이진화** 가능
- 이진화 오류가 줄어들어 전반적인 특징 표현 품질 향상

### 3.5 일반화 성능의 한계

- Rademacher 복잡도 항 $4L_\rho \mathfrak{R}(\mathcal{F})$에서 $L_\rho$는 $\rho_{+1}+\rho_{-1}$이 커질수록 급격히 증가 → $\rho$ 값이 너무 크면 일반화 bound가 오히려 느슨해짐
- 실험에서도 $\rho = 0.2$ 이상이면 성능 저하 확인

---

## 4. 미래 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

**① BNN 학습 패러다임 전환**
- "이진화 = 단순 규칙"이 아닌 "이진화 = 학습 문제"라는 관점 제시
- 이후 연구들이 이진화 자체를 학습 목표로 삼는 방향성을 탐색하는 데 영향

**② 크로스 도메인 방법론 융합**
- Noisy label learning의 방법론을 BNN 학습에 성공적으로 접목
- 향후 knowledge distillation, self-supervised learning 등 다른 분야와의 융합 연구에도 모티베이션 제공

**③ 이론적 기반 강화**
- BNN 분야에서 수렴 보장이 있는 이론적 분석을 제시한 드문 사례
- 향후 BNN의 최적화 이론 연구에 기초 제공

### 4.2 앞으로 연구 시 고려할 점

**① 하이퍼파라미터 자동 최적화**
- 현재 $\rho$와 $\alpha$를 수동으로 설정해야 하므로, 이를 데이터로부터 자동 추정하는 메타학습 접근법 필요
- 예: Bayesian optimization이나 noise rate estimation 기법과의 결합

**② 활성화 이진화로의 확장**
- 본 논문은 주로 가중치 이진화에 집중
- 활성화 이진화에도 동일한 noisy supervision 접근법 적용 가능 여부 탐색 필요

**③ 매핑 모델의 효율성 개선**
- 각 레이어에 3층 CNN 매핑 모델 추가 → 학습 비용 증가
- 경량화 매핑 모델 설계 또는 레이어 간 매핑 모델 공유 전략 연구 필요

**④ 더 정교한 노이즈 모델**
- 현재는 class-conditional uniform noise를 가정
- 레이어별, 채널별 비균일 노이즈 모델링으로 확장 가능

**⑤ 트랜스포머 아키텍처로의 확장**
- 본 논문은 CNN에 한정되어 있어, Vision Transformer(ViT) 등의 BNN화에 LNS 방법 적용 연구 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래 최신 논문들에 대한 정보는 제 학습 데이터 기준(2024년 초까지)의 지식에 근거하며, 일부 세부 수치는 확인이 필요할 수 있습니다. 확실한 정보만 기재합니다.

### 5.1 관련 최신 연구 흐름

| 연구 방향 | 대표 논문 | LNS와의 관계 |
|---|---|---|
| **정보 보존 기반 BNN** | IR-Net (Qin et al., CVPR 2020) | LNS가 IR-Net 대비 ImageNet Top-1 1.3% 향상 |
| **실수 도메인 최적화** | ReActNet (Liu et al., ECCV 2020) | Distribution reshaping으로 LNS보다 높은 성능 보고 |
| **지식 증류 활용 BNN** | RBNN (Lin et al., NeurIPS 2020) | 전체 정밀도 모델의 지식을 BNN으로 전달 |
| **Binary ViT** | BiViT, BinaryViT 계열 | LNS의 접근법이 ViT 이진화에도 영감 제공 가능 |

### 5.2 ReActNet과의 비교 (Liu et al., ECCV 2020)

ReActNet은 활성화 분포를 재형성(reshape)하는 방식으로 ResNet-18 기반 BNN에서 약 **65.9% Top-1** (ImageNet)을 달성했다고 알려져 있습니다. 이는 LNS의 59.4%를 크게 상회하지만, 더 큰 채널 수와 추가 구조적 변경을 사용한 결과입니다.

**차이점 비교:**

| 항목 | LNS (Han et al., 2020) | ReActNet (Liu et al., 2020) |
|---|---|---|
| **핵심 아이디어** | Noisy label learning | 활성화 분포 재형성 |
| **이진화 대상** | 가중치 중심 | 가중치 + 활성화 |
| **이론적 보장** | O (Rademacher 복잡도) | 제한적 |
| **아키텍처 변경** | 최소 (매핑 모델만 추가) | 채널 확장 등 구조 변경 |

### 5.3 RBNN (Lin et al., NeurIPS 2020)과의 비교

RBNN은 회전 기반(rotation-based) 이진화와 지식 증류를 결합하여 BNN 성능을 높입니다. LNS의 noisy supervision 개념을 일종의 소프트 타겟으로 보면, RBNN의 증류 기반 접근법과 개념적으로 연결되는 부분이 있습니다.

### 5.4 종합 비교

```
BNN 성능 개선 접근법 분류 (2020 이후):

1. 손실 함수 설계 계열 (LNS 포함)
   → noisy label, loss correction, unbiased estimation

2. 아키텍처 설계 계열 (ReActNet, MeliusNet 등)
   → 채널 확장, 잔차 연결 강화

3. 지식 증류 계열 (RBNN, FDA-BNN 등)
   → 전체 정밀도 모델 → BNN 지식 전달

4. NAS(신경망 구조 탐색) 계열 (BNAS 등)
   → BNN에 최적화된 구조 자동 탐색
```

LNS는 **손실 함수 설계 계열의 선구적 연구**로, 이론적 엄밀성과 범용성(다양한 BNN 백본에 적용 가능)에서 차별성을 가집니다.

---

## 참고 자료

1. **Han, K., Wang, Y., Xu, Y., Xu, C., Wu, E., & Xu, C. (2020).** "Training Binary Neural Networks through Learning with Noisy Supervision." *Proceedings of the 37th International Conference on Machine Learning (ICML 2020)*, PMLR 119. arXiv:2010.04871v1

2. **Natarajan, N., Dhillon, I. S., Ravikumar, P. K., & Tewari, A. (2013).** "Learning with noisy labels." *NeurIPS 2013* — (Theorem 1의 이론적 기반)

3. **Rastegari, M., Ordonez, V., Redmon, J., & Farhadi, A. (2016).** "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks." *ECCV 2016*

4. **Liu, Z., Wu, B., Luo, W., Yang, X., Liu, W., & Cheng, K.-T. (2018).** "Bi-Real Net: Enhancing the Performance of 1-bit CNNs with Improved Representational Capability and Advanced Training Algorithm." *ECCV 2018*

5. **Zhou, S., Wu, Y., Ni, Z., Zhou, X., Wen, H., & Zou, Y. (2016).** "DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients." arXiv:1606.06160

6. **Qin, H., Gong, R., Liu, X., Shen, M., Wei, Z., Yu, F., & Song, J. (2020).** "Forward and Backward Information Retention for Accurate Binary Neural Networks." *CVPR 2020*

7. **Liu, Z., et al. (2020).** "ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions." *ECCV 2020* — (비교 분석 참고)

8. **Bengio, Y., Leonard, N., & Courville, A. C. (2013).** "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation." arXiv:1308.3432
