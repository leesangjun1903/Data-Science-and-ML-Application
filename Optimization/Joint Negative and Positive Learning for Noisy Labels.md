# Joint Negative and Positive Learning for Noisy Labels (JNPL)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 **노이즈 레이블(Noisy Labels)** 환경에서 CNN을 학습할 때, 기존 NLNL(Negative Learning for Noisy Labels)의 **3단계 파이프라인(NL → SelNL → SelPL)** 이 갖는 비효율성과 NL 손실 함수의 근본적인 **언더피팅(Underfitting) 문제**를 해결하고자 한다. 이를 위해 **단일 단계(Single-Stage)** 파이프라인인 **JNPL(Joint Negative and Positive Learning)** 을 제안한다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| 단일 단계 파이프라인 | NLNL의 3단계 → JNPL의 1단계로 단순화 |
| NL+ 손실 함수 | NL의 언더피팅 문제를 그래디언트 재설계로 해결 |
| PL+ 손실 함수 | 고신뢰도(clean) 데이터에 더 강한 그래디언트 제공 |
| 사전 지식 불필요 | 노이즈 비율/유형에 대한 사전 지식 없이 적용 가능 |
| SOTA 달성 | CIFAR10, CIFAR100, Clothing1M에서 최고 성능 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### (1) Positive Learning(PL)의 한계

기존 표준 크로스 엔트로피(CE) 기반의 Positive Learning은 "입력 이미지가 이 레이블에 속한다"는 직접적인 학습 방식이다:

$$\mathcal{L}_{PL}(f, y) = -\sum_{k=1}^{c} y_k \log p_k \tag{1}$$

노이즈 레이블이 존재할 경우, CNN이 오염된 레이블을 **암기(memorize)** 하는 과적합(overfitting) 위험이 높다.

#### (2) Negative Learning(NL)의 언더피팅 문제

NL은 보완 레이블(complementary label) $\bar{y}$를 이용해 "입력 이미지가 이 레이블에 속하지 않는다"고 학습한다:

$$\mathcal{L}_{NL}(f, \bar{y}) = -\sum_{k=1}^{c} \bar{y}_k \log(1 - p_k) \tag{2}$$

이 손실 함수의 그래디언트를 분석하면:

$$\nabla \mathcal{L}_{NL} = \frac{\partial \mathcal{L}_{NL}(f, \bar{y})}{\partial f_i} = \begin{cases} p_i & \text{if } i = \bar{y} \\ -\frac{p_{\bar{y}}}{1 - p_{\bar{y}}} p_i & \text{if } i \neq \bar{y} \end{cases} \tag{3}$$

**문제의 핵심:** 노이즈 데이터에서 실제 정답 클래스가 $\bar{y}$로 선택될 경우, $p_{\bar{y}}$가 높아지면 $-\frac{p_{\bar{y}}}{1-p_{\bar{y}}}p_i$ 값이 매우 커져, 정답 클래스 이외의 모든 클래스에 과도한 그래디언트가 전달되어 **언더피팅** 이 발생한다.

#### (3) NLNL의 비효율적 3단계 파이프라인

$$\text{NL} \rightarrow \text{SelNL} \rightarrow \text{SelPL}$$

각 단계를 순차적으로 수행해야 하므로 학습 비용이 크고, 특히 비대칭(asymm) 노이즈 유형에서 SelNL이 효과적이지 않은 문제가 존재한다.

---

### 2.2 제안 방법 (수식 포함)

#### JNPL의 통합 손실 함수

$$\mathcal{L}_{JNPL} = \mathcal{L}_{NL+} + \lambda \mathcal{L}_{PL+} \tag{4}$$

단, $\lambda = 0.01$로 고정하여 PL+가 NL+를 압도하지 않도록 스케일을 조정한다.

---

#### (A) NL+ 손실 함수

NL의 언더피팅 문제를 해결하기 위해, $p_{\bar{y}}$를 가중 인자로 도입:

$$\mathcal{L}_{NL+}(f, \bar{y}) = -(1 - p_{\bar{y}}) \sum_{k=1}^{c} \bar{y}_k \log(1 - p_k) \tag{5}$$

이에 대한 그래디언트:

$$\nabla \mathcal{L}_{NL+(i \neq \bar{y})} = (1 - p_{\bar{y}}) \nabla \mathcal{L}_{NL(i \neq \bar{y})} = -p_{\bar{y}} p_i \tag{6}$$

**핵심 원리:** $(1 - p_{\bar{y}})$가 가중 인자로 작용하여, $p_{\bar{y}}$가 높을 때(즉, $\bar{y}$가 실제 정답일 가능성이 높을 때) 그래디언트 크기를 자동으로 감소시킨다. 이를 통해 노이즈 데이터가 정답 클래스의 확률을 유지할 수 있게 된다.

---

#### (B) PL+ 손실 함수

고신뢰도 데이터에 더 강한 그래디언트를 제공하여 빠른 수렴을 달성:

$$\mathcal{L}_{PL+}(f, \hat{y}) = -\prod_{n=0}^{N}(1 + p_{\hat{y}}^{2^n}) \sum_{k=1}^{c} y_k \log p_k \tag{7}$$

이에 대한 그래디언트:

$$\nabla \mathcal{L}_{PL+} = \prod_{n=0}^{N}(1 + p_{\hat{y}}^{2^n}) \nabla \mathcal{L}_{PL} = -\prod_{n=0}^{N}(1 + p_{\hat{y}}^{2^n})(1 - p_{\hat{y}}) = -(1 - p_{\hat{y}}^{2^{N+1}}) \tag{8}$$

**핵심 원리:** 표준 CE 손실(PL)은 신뢰도가 낮은 데이터에 더 큰 그래디언트를 제공하는 반면, PL+는 $N=3$ 설정 시 고신뢰도 데이터에 더 강한 그래디언트를 제공하여 빠른 수렴을 가능케 한다.

**PL+ 데이터 선택 기준 (Algorithm 1):**

- $\hat{y} = \arg\max_i p_i$ (최대 확률 클래스)
- 모든 다른 클래스의 확률이 균등 분포 $\frac{1}{c}$ 미만일 때만 후보로 선정
- $p_{\hat{y}}$에 비례하는 베르누이 샘플링으로 최종 선택

---

### 2.3 모델 구조

```
[입력 데이터 (노이즈 포함)]
          ↓
    [CNN (ResNet34/50)]
          ↓
    [Softmax → p ∈ Δ^(c-1)]
          ↓
  ┌───────────────────────┐
  │  NL+ 손실 계산        │  ← 전체 배치에 적용
  │  (노이즈 필터링 역할) │
  └───────────────────────┘
          +
  ┌───────────────────────┐
  │  PL+ 손실 계산        │  ← Algorithm 1로 선택된 데이터에만 적용
  │  (빠른 수렴 역할)     │
  └───────────────────────┘
          ↓
   [JNPL 단일 단계 완료]
          ↓
  [필터링된 clean/noisy 분리]
          ↓
  [Pseudo-Labeling (반지도 학습)]
          ↓
  [최종 분류 성능 평가]
```

- **백본 모델:** CIFAR10/100 → ResNet34, Clothing1M → ResNet50 (ImageNet pretrained)
- **최적화:** SGD (momentum=0.9, weight decay= $10^{-4}$ )
- **학습률:** CIFAR: $10^{-2}$ (800 에폭에서 $\times 0.1$), Clothing1M: $10^{-3}$ (30 에폭에서 $\times 0.1$)

---

### 2.4 성능 향상

#### CIFAR10 결과 (ResNet34)

| 방법 | Symm 20 | Symm 40 | Symm 60 | Symm 80 | Asymm 10 | Asymm 40 |
|---|---|---|---|---|---|---|
| CE | 83.95 | 67.58 | 43.55 | 17.32 | 91.39 | 76.37 |
| Co-teaching | 91.08 | 88.08 | 80.96 | 21.13 | 94.20 | 70.20 |
| NLNL | **94.23** | **92.43** | 88.32 | - | **94.57** | 89.86 |
| **JNPL (Ours)** | 93.53 | 91.89 | **88.45** | **35.65** | 94.22 | **90.72** |

#### CIFAR100 결과 (ResNet34)

| 방법 | Symm 40 | Symm 60 | Asymm 30 | Asymm 40 |
|---|---|---|---|---|
| NLNL | 66.39 | 56.51 | 54.87 | 45.70 |
| **JNPL (Ours)** | **68.11** | **61.26** | **68.12** | **59.51** |

> CIFAR100의 Asymm 40% 노이즈에서 NLNL 대비 **약 14% 향상** — 가장 어려운 설정에서 압도적 우위.

#### Clothing1M 결과

| 방법 | Test Accuracy |
|---|---|
| PENCIL | 73.49 |
| **JNPL (Ours)** | **74.15** |

---

### 2.5 한계점

1. **하이퍼파라미터 민감성:** $\lambda = 0.01$, $N = 3$이 전체 실험에 고정되어 있으나, 도메인에 따른 최적값 탐색 필요성이 존재한다.
2. **극단적 노이즈(80%):** Symm 80%에서 35.65%로 타 방법(NLNL은 결과 미보고) 대비 여전히 낮은 성능을 보인다.
3. **계산 비용:** 1000 에폭 학습 + 추가 Pseudo-Labeling 단계로 전체 학습 시간이 상당히 길다.
4. **실험 범위:** Vision 태스크(이미지 분류) 위주로 자연어 처리, 음성 인식 등 타 도메인 검증이 부재하다.
5. **PL+ 선택 기준의 경직성:** 균등 분포 $\frac{1}{c}$ 임계값이 고정되어 있어, 클래스 불균형 데이터에 취약할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 위한 핵심 메커니즘

#### (A) 노이즈 유형에 대한 일반화

기존 NL의 SelNL은 비대칭(asymm) 노이즈에서 거의 효과가 없었다. 예를 들어 CAT ↔ DOG 양방향 노이즈에서 데이터가 균등 분포 이하로 내려가지 않아 SelNL이 동작하지 않는 문제가 있었다. 반면 NL+는 그래디언트를 부드럽게 감소시켜, $p_y < 0.5$ & $p_{\bar{y}} > 0.5$ 영역에서도 두 클래스를 점진적으로 분리 가능하다. Figure 5에서 JNPL은 대칭/비대칭 노이즈 모두에서 NLNL보다 높은 Average Precision(AP)을 달성한다.

#### (B) 노이즈 비율에 대한 일반화

$$\text{AP Gap(JNPL - NLNL)} \propto \text{noise rate}$$

노이즈 비율이 증가할수록 JNPL과 NLNL의 AP 격차가 벌어지는 현상이 관찰되었다. 이는 NL+의 그래디언트 설계($-p_{\bar{y}} p_i$)가 고노이즈 환경에서도 안정적으로 clean/noisy 분리를 유지함을 의미한다.

#### (C) 클래스 수에 대한 일반화

CIFAR100(100클래스)에서 NLNL은 노이즈 증가에 따라 AP가 급격히 저하되지만, JNPL은 CIFAR10과 유사한 수준의 강건성을 유지한다. 이는 JNPL이 **클래스 수가 증가해도 일반화 성능을 유지**할 수 있음을 시사한다.

#### (D) 실세계 데이터(Clothing1M)로의 일반화

실세계 노이즈(61.54% 레이블 정확도)를 갖는 Clothing1M에서 74.15%를 달성하여, 합성 노이즈 환경에서 학습된 방법론이 실세계로 이전 가능함을 입증한다.

### 3.2 일반화 향상을 위한 추가 가능성

- **Pseudo-Labeling과의 시너지:** JNPL으로 정밀하게 필터링된 데이터를 pseudo-label로 활용하면, 반지도 학습 단계에서 더 높은 품질의 레이블이 제공되어 최종 일반화 성능이 향상된다.
- **사전 지식 불필요:** 노이즈 비율/유형을 몰라도 적용 가능하므로, 실제 배포 환경에서의 일반화가 용이하다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려점

### 4.1 연구에 미치는 영향

#### (A) 손실 함수 설계 패러다임의 전환

JNPL은 **그래디언트 분석을 통한 손실 함수 재설계** 의 유효성을 명확히 입증했다. 단순한 경험적 설계가 아닌, 노이즈 데이터에서의 그래디언트 거동을 수학적으로 분석하고 이를 기반으로 손실 함수를 설계하는 방법론적 접근법이 향후 연구의 표준이 될 가능성이 있다.

#### (B) 단일 단계 학습의 실용성 강조

복잡한 다단계 파이프라인을 단순화하여 실용성을 높이는 방향성은, 특히 산업 현장에서 학습 파이프라인 설계 시 중요한 참고점이 된다.

#### (C) 보완 레이블 학습의 가능성 확장

NL+는 보완 레이블(complementary label)을 활용한 간접 학습이 다양한 노이즈 환경에서 효과적임을 보여줌으로써, **부분 레이블 학습(Partial Label Learning)**, **약한 지도 학습(Weakly Supervised Learning)** 분야로의 확장 연구를 촉진할 수 있다.

### 4.2 앞으로 연구 시 고려할 점

1. **Long-tail 분포와의 결합:** 클래스 불균형(Class Imbalance)이 심한 데이터에서 PL+의 균등 분포 기반 선택 기준이 적절한지 재검토 필요.

2. **Transformer 기반 모델 적용:** 본 논문은 CNN(ResNet) 위주로 검증되었다. Vision Transformer(ViT), BERT 등에서의 NL+/PL+ 적용 효과 분석이 필요하다.

3. **동적 $\lambda$ 스케줄링:** 학습 초기에는 NL+의 비중을 높이고, 수렴 후에는 PL+의 비중을 높이는 동적 조정 전략 탐구.

4. **대규모 실세계 데이터셋 검증:** Clothing1M 외 WebVision, Food-101N 등 다양한 실세계 노이즈 데이터셋에서의 검증 필요.

5. **노이즈 전이 행렬(Transition Matrix) 추정과의 결합:** JNPL의 필터링 결과를 노이즈 전이 행렬 추정에 활용하는 하이브리드 방식 탐구.

6. **자기지도 학습(Self-Supervised Learning)과의 결합:** SimCLR, MoCo 등 자기지도 사전 학습을 통해 얻은 강건한 표현과 JNPL을 결합하면 추가 성능 향상 기대.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 비교 대상 연구

아래 비교는 논문 내 인용 문헌 및 JNPL과 동일한 문제를 다루는 주요 연구들을 기반으로 한다.

| 방법 | 발표 연도 | 핵심 아이디어 | JNPL 대비 차이점 |
|---|---|---|---|
| **Co-teaching** (Han et al.) | NeurIPS 2018 | 두 네트워크가 서로의 clean 샘플을 가르침 | 노이즈 비율 사전 지식 필요, 고노이즈에서 성능 저하 |
| **JoCoR** (Wei et al.) | CVPR 2020 | 동의(Agreement)를 활용한 공동 학습 + 공동 정규화 | 노이즈 비율 사전 지식 필요 |
| **APL/NCE+RCE** (Ma et al.) | ICML 2020 | 정규화된 손실 함수(Active Passive Loss) | 고노이즈에서 성능 급감, 사전 지식 불필요 |
| **DivideMix** (Li et al.) | ICLR 2020 | 가우시안 혼합 모델(GMM)로 clean/noisy 분리 후 MixMatch 적용 | 매우 복잡한 파이프라인, CIFAR에서 높은 성능 |
| **JNPL (본 논문)** | arXiv 2021 | NL+/PL+ 단일 단계 파이프라인 | 사전 지식 불필요, 단순한 구조, 고노이즈 강건성 |

### 5.2 DivideMix와의 비교 (중요 연구)

**DivideMix** (Li et al., ICLR 2020)는 JNPL과 유사한 시기에 높은 주목을 받은 방법으로:
- GMM으로 손실 분포를 clean/noisy로 분리
- MixMatch 기반 반지도 학습 결합
- CIFAR10 Symm 90%에서 약 93%의 높은 정확도 달성

반면 JNPL은 구조가 단순하고 사전 지식이 불필요하지만, DivideMix 수준의 극단적 노이즈 처리 능력에서는 차이가 있을 수 있다. (본 논문은 DivideMix와의 직접 비교는 포함하지 않음.)

> **⚠️ 주의:** DivideMix (ICLR 2020), 이후의 SOP (NeurIPS 2022), Sel-CL (CVPR 2022) 등과의 정량적 직접 비교는 본 논문(arXiv 2104.06574v1)에 포함되어 있지 않으므로, 해당 수치 비교는 본 답변에서 제시하지 않습니다.

---

## 참고 자료

- **주 논문:** Youngdong Kim, Juseung Yun, Hyounguk Shon, Junmo Kim. "Joint Negative and Positive Learning for Noisy Labels." arXiv:2104.06574v1, 2021.
- **NLNL (선행 연구):** Youngdong Kim et al. "NLNL: Negative Learning for Noisy Labels." ICCV 2019, pp. 101–110.
- **Co-teaching:** Bo Han et al. "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels." NeurIPS 2018.
- **JoCoR:** Hongxin Wei et al. "Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization." CVPR 2020.
- **APL:** Xingjun Ma et al. "Normalized Loss Functions for Deep Learning with Noisy Labels." ICML 2020.
- **Symmetric CE:** Yisen Wang et al. "Symmetric Cross Entropy for Robust Learning with Noisy Labels." ICCV 2019.
- **Pseudo-Label:** Dong-Hyun Lee. "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks." ICML Workshop 2013.
- **Clothing1M Dataset:** Tong Xiao et al. "Learning from Massive Noisy Labeled Data for Image Classification." CVPR 2015.
- **ResNet:** Kaiming He et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
