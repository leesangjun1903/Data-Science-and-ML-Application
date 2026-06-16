# Robust Early-Learning: Hindering the Memorization of Noisy Labels 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음과 같습니다:

> **딥 네트워크의 파라미터는 모두 동등하지 않다.** 일부 파라미터(**Critical Parameters**)는 클린 레이블을 잘 피팅하고 일반화에 기여하는 반면, 나머지(**Non-Critical Parameters**)는 노이즈 레이블을 피팅하는 경향이 있어 일반화를 저해한다.

이 관찰을 기반으로, **Early Stopping 이전 단계에서** 노이즈 레이블의 부작용을 줄이는 **Robust Early-Learning** 방법론(CDR: Combating noisy labels with Different update Rules)을 제안합니다.

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| ① 파라미터 분류 방법 | 클린 레이블 피팅에 중요한지 여부에 따라 파라미터를 Critical/Non-Critical로 분류하는 새로운 기준 제시 |
| ② 차별적 업데이트 규칙 | 두 유형의 파라미터에 서로 다른 업데이트 규칙 설계 (Robust Positive Update / Negative Update) |
| ③ 실험적 검증 | 합성 노이즈 및 실세계 노이즈 데이터셋 모두에서 SOTA 대비 우수한 성능 입증 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

딥 네트워크는 **메모리제이션 효과(Memorization Effect)**로 인해 훈련 초기에는 클린 레이블을 먼저 학습하고, 이후 노이즈 레이블을 학습합니다. 이 특성을 활용한 **Early Stopping**이 효과적이나, 다음 한계가 존재합니다:

- Early Stopping 이전에도 노이즈 레이블이 **클린 레이블 메모리제이션을 방해**함
- 과도한 파라미터화(Over-parameterization)로 인해 노이즈 레이블에 결국 오버피팅됨
- Dropout, Weight Decay 등 일반적 정규화는 이 문제를 충분히 해결하지 못함

**핵심 연구 질문:** Early Stopping **이전** 단계에서 노이즈 레이블의 부작용을 어떻게 줄일 수 있는가?

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: Optimality Criterion 재해석

목적함수 $L(\mathcal{W}; S)$의 최적화에서 $G(t) = L(t\mathcal{W}; S)$로 정의하면:

$$G'(t) = \nabla L(t\mathcal{W}; S)^\top \mathcal{W}$$

$t=1$로 설정하면:

$$G'(1) = \langle \nabla L(\mathcal{W}; S), \mathcal{W} \rangle$$

최적점에서는 $\nabla L(\mathcal{W}; S) = \mathbf{0}$이므로 $G'(1) = 0$. 따라서 **스칼라 $G'(1)$으로 최적성을 판단**할 수 있습니다.

#### Step 2: 파라미터 중요도 판단 기준

각 파라미터 $w_i \in \mathcal{W}$에 대해 중요도 판단 기준 $g_i$를 다음과 같이 정의:

$$\boxed{g_i = |\nabla L(w_i; S) \times w_i|, \quad i \in [m]}$$

- $g_i$가 **크면** → $w_i$는 **Critical Parameter** (클린 레이블 피팅에 중요)
- $g_i$가 **작거나 0에 가까우면** → $w_i$는 **Non-Critical Parameter** (노이즈 레이블에 취약)

> **직관적 이유:** 그래디언트만 보면 비활성화된 파라미터($w_i \approx 0$)의 중요도를 무시하게 됨. 파라미터 값과 그래디언트의 곱을 사용함으로써 이를 보완.

#### Step 3: Critical 파라미터 수 결정

노이즈율 $\tau$를 이용하여 Critical 파라미터의 수를 결정:

$$\boxed{m_c = (1 - \tau) \cdot m}$$

**직관:** 노이즈율이 높을수록 클린 레이블이 적으므로, 필요한 Critical 파라미터 수도 적어짐.

#### Step 4-a: Critical 파라미터 — Robust Positive Update

$$\boxed{\mathcal{W}_c(k+1) \leftarrow \mathcal{W}_c(k) - \eta\left((1-\tau)\frac{\partial L(\mathcal{W}_c(k); \widetilde{S}^*)}{\partial \mathcal{W}_c(k)} + \lambda \text{sgn}(\mathcal{W}_c(k))\right)}$$

- **목적함수 그래디언트 + Weight Decay**를 모두 사용
- 그래디언트 계수 $(1-\tau)$: **과신뢰(over-confident) 업데이트 방지** (Gradient Decay 역할)

#### Step 4-b: Non-Critical 파라미터 — Negative Update

$$\boxed{\mathcal{W}_n(k+1) \leftarrow \mathcal{W}_n(k) - \eta\lambda\,\text{sgn}(\mathcal{W}_n(k))}$$

- **Weight Decay만 적용**, 목적함수 그래디언트는 사용하지 않음
- Non-Critical 파라미터의 값을 0으로 수렴시켜 **비활성화**함 → 노이즈 레이블에 오버피팅되지 않도록 방지

#### Step 5: 기본 최적화 목적함수

전체 목적함수는 $\ell_1$ 정규화를 포함:

$$\min L(\mathcal{W}; S) = \min \frac{1}{n}\sum_{i=1}^{n} L(\mathcal{W}; (\mathbf{x}_i, y_i)) + \lambda\|\mathcal{W}\|_1$$

---

### 2.3 모델 구조 및 알고리즘

**CDR (Combating noisy labels with Different update Rules) 알고리즘:**

```
Input: 초기 파라미터 W, 노이즈 훈련셋 Dt, 노이즈 검증셋 Dv, 
       학습률 η, weight decay λ, 노이즈율 τ

for T = 1, 2, ..., Tmax do:
    훈련셋 Dt 셔플
    for N = 1, ..., Nmax do:
        미니배치 D̄t 샘플링
        Eq.(3), Eq.(4)로 W를 Wc, Wn으로 분류
        Eq.(5)로 Wc 업데이트 (Robust Positive Update)
        Eq.(6)로 Wn 업데이트 (Negative Update)

Early Stopping: 검증셋 Dv에서 최소 분류 오류 달성 시 종료
Output: 업데이트된 파라미터 W
```

**사용된 네트워크 아키텍처:**

| 데이터셋 | 네트워크 | 배치 크기 |
|----------|----------|-----------|
| MNIST | LeNet | 32 |
| F-MNIST | ResNet-50 | 32 |
| CIFAR-10/100 | ResNet-50 | 64 |
| Food-101 | ResNet-50 (ImageNet pre-trained) | 32 |
| WebVision | Inception-ResNet v2 | 128 |

---

### 2.4 성능 향상

#### 합성 노이즈 데이터셋 (CIFAR-100, 대표적 결과)

| 방법 | Symmetric-40% | Asymmetric-40% | Pairflip-40% | Instance-40% |
|------|--------------|----------------|-------------|-------------|
| CE | 56.82 | 52.86 | 52.77 | 50.84 |
| GCE | 57.97 | 54.35 | 55.03 | 55.14 |
| Joint | 59.45 | 55.53 | 52.22 | 55.09 |
| **CDR** | **62.72** | **55.58** | **56.94** | **61.03** |

- Instance-40%에서 2위 대비 **약 6% 향상**

#### 실세계 노이즈 데이터셋

| 데이터셋 | CDR | 2위 방법 |
|----------|-----|----------|
| Food-101 | **86.36%** | T-Revision: 85.97% |
| WebVision (Top-1) | **61.85%** | APL: 61.27% |

---

### 2.5 한계

1. **노이즈율 $\tau$ 의존성:** Critical 파라미터 수 결정과 그래디언트 감쇠 계수 모두 $\tau$에 의존. 단, 논문은 추정치에 강건함을 실험으로 확인.
2. **이진 분류(Critical/Non-Critical)의 단순성:** 실제로는 파라미터 중요도가 연속적인 스펙트럼을 가질 수 있음.
3. **DivideMix, SELF 등 복합 기법과의 직접 비교 부재:** 논문 자체는 단일 기법임을 강조하지만, 현실에서는 복합 기법 대비 성능 비교가 필요함.
4. **이론적 수렴 보장 부재:** CDR의 수렴성에 대한 이론적 분석이 충분하지 않음.
5. **Large-scale 설정 한계:** 매 이터레이션마다 전체 파라미터를 $g_i$ 기준으로 정렬해야 하므로, 파라미터가 매우 많은 대형 모델에서 계산 비용 증가 우려.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 핵심 메커니즘: 왜 일반화가 향상되는가?

#### (a) 노이즈 레이블의 부작용 사전 차단

기존 Early Stopping은 노이즈 레이블에 의한 오버피팅이 시작된 **이후** 훈련을 멈추는 수동적 방어입니다. CDR은 **Early Stopping 이전 단계에서** 노이즈 레이블이 파라미터에 미치는 영향을 **능동적으로 차단**합니다.

$$\text{기존: } \underbrace{\text{클린 학습}}_{\text{초기}} \rightarrow \underbrace{\text{노이즈 영향 증가}}_{\text{중기}} \rightarrow \underbrace{\text{Early Stop}}_{}$$

$$\text{CDR: } \underbrace{\text{클린 학습 강화}}_{\mathcal{W}_c \text{ 업데이트}} \rightarrow \underbrace{\text{노이즈 영향 최소화}}_{\mathcal{W}_n \text{ 비활성화}} \rightarrow \underbrace{\text{Early Stop}}_{}$$

#### (b) Lottery Ticket Hypothesis 관점

논문은 Frankle & Carbin (2018)의 Lottery Ticket Hypothesis에서 영감을 받아, 일반화에 중요한 "winning ticket" 파라미터(= Critical Parameters)만을 적극적으로 훈련하고 나머지는 억제합니다. 이는 효과적인 **암묵적 모델 압축**으로 작용하여 일반화를 향상시킵니다.

#### (c) Gradient Decay $(1-\tau)$의 일반화 효과

Robust Positive Update의 그래디언트 계수 $(1-\tau)$는:
- 노이즈율이 높을수록 업데이트 보폭을 줄여 **과신뢰 업데이트 방지**
- 클린 샘플 비율에 비례한 적응적 학습 속도 조절

$$\mathcal{W}_c(k+1) \leftarrow \mathcal{W}_c(k) - \eta\left(\underbrace{(1-\tau)}_{\text{adaptive coefficient}}\frac{\partial L}{\partial \mathcal{W}_c(k)} + \lambda \text{sgn}(\mathcal{W}_c(k))\right)$$

#### (d) Weight Decay의 이중 역할

Non-Critical 파라미터에 적용된 Weight Decay는:
- 해당 파라미터를 0으로 수렴시켜 **비활성화**
- Arora et al. (2018)의 연구에 따르면, 이러한 압축이 오히려 **더 강한 일반화 보장**을 제공할 수 있음

#### (e) 실험적 증거 (Figure 2)

논문의 Figure 2 (CIFAR-100, 40% 노이즈)에서 확인:
- CDR은 훈련 초기 단계부터 CE보다 **높은 테스트 정확도** 유지
- CE는 노이즈 레이블에 의해 조기에 성능이 저하되는 반면, CDR은 **안정적인 상승 곡선** 유지

### 3.2 일반화 향상의 추가 가능성

1. **Semi-supervised Learning과의 결합:** Non-Critical 파라미터를 완전히 비활성화하는 대신, 가상 레이블(pseudo-label)로 활용하면 추가 개선 가능
2. **Contrastive Learning과의 결합:** Critical 파라미터가 표현 학습에 집중하도록 설계하면 특징 추출 능력 강화
3. **Curriculum Learning과의 결합:** 훈련 초기에는 $\tau$를 높게 설정하여 더 많은 파라미터를 Non-Critical로 처리하고, 점진적으로 줄이는 동적 스케줄링

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 관련 최신 연구들

| 논문 | 발표 | 핵심 방법 | CDR 대비 특징 |
|------|------|-----------|--------------|
| **DivideMix** (Li et al., 2020) | ICLR 2020 | 가우시안 혼합 모델로 클린/노이즈 샘플 분리 후 Semi-supervised learning | 복합 기법 (CDR은 단일 기법으로 이와 상보적) |
| **ELR (Early-Learning Regularization)** (Liu et al., 2020) | NeurIPS 2020 | 초기 예측을 타겟으로 정규화하여 메모리제이션 방지 | CDR과 동일 문제 의식, 접근법 차이 (샘플 vs 파라미터 레벨) |
| **CORES** (Cheng et al., 2021) | ICLR 2021 | Sample sieve를 통한 인스턴스 의존 노이즈 처리 | CDR은 파라미터 레벨, CORES는 샘플 레벨 |
| **ProMix** (2022~) | - | Prototype 기반 클린 샘플 분리 | CDR의 파라미터 분류 아이디어와 결합 가능성 |
| **SOP (Sample-Level Orthogonality)** (2022) | - | 직교 정규화로 노이즈 레이블 저항성 향상 | - |

### 4.2 CDR의 차별성

```
파라미터 레벨 접근          vs.        샘플 레벨 접근
(CDR, Lottery Ticket 기반)          (DivideMix, MentorNet 등)
        ↓                                    ↓
  "어떤 파라미터가                    "어떤 샘플이
   중요한가?"                          신뢰할 수 있는가?"
```

- 대부분의 최신 방법이 **샘플 선택(sample selection)** 관점인 반면, CDR은 **파라미터 관점**이라는 근본적 차이
- CDR은 "orthogonal to other methods"를 주장하여 **다른 방법과 결합 가능**

### 4.3 한계 극복을 위한 최신 연구 방향

| CDR 한계 | 후속 연구 방향 |
|----------|--------------|
| 노이즈율 추정 의존성 | Adaptive noise rate estimation (예: GMM 기반 자동 추정) |
| 이진 분류의 단순성 | Soft parameter importance scoring (연속값 중요도) |
| 대형 모델 계산 비용 | Layer-wise 또는 block-wise 중요도 계산으로 효율화 |

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

#### (a) 파라다임 전환: 파라미터 레벨 노이즈 학습

CDR은 기존의 **데이터 레벨(샘플 선택, 손실 함수 수정)**에서 벗어나 **파라미터 레벨**에서 노이즈를 처리하는 새로운 패러다임을 제시합니다. 이는 다음 연구 방향을 촉진합니다:

- 파라미터 중요도 기반 **적응형 정규화** 연구
- 노이즈 레이블 학습과 **네트워크 프루닝/압축**의 융합 연구

#### (b) Lottery Ticket Hypothesis의 응용 확장

Lottery Ticket Hypothesis를 **학습 중(dynamic)** 노이즈 레이블 환경에서 활용하는 선례를 제시. 향후 **동적 sparse training**과 노이즈 학습의 결합 연구가 기대됩니다.

#### (c) 상보적 프레임워크로서의 활용

논문 자체가 "simple and orthogonal to other methods"임을 강조하므로, DivideMix, CORES 등의 SOTA 방법과 CDR을 **결합하는 연구**가 활발해질 것으로 예상됩니다.

### 5.2 향후 연구 시 고려할 점

#### ① 동적 임계값 설정

현재 CDR은 $\tau$를 고정값으로 사용하지만, 실제로는 **훈련이 진행될수록 메모리제이션 패턴이 변화**합니다. 따라서:

$$m_c(t) = f(\tau, t) \cdot m \quad \text{(epoch에 따라 동적 조정)}$$

와 같은 **시간-적응형 파라미터 분류** 연구가 필요합니다.

#### ② 파라미터 중요도의 더 정교한 측정

현재 기준 $g_i = |\nabla L(w_i; S) \times w_i|$는 Hessian 정보를 무시합니다. 2차 정보를 활용하면:

$$g_i^{\text{2nd}} = |w_i^\top H_{ii} w_i| \quad \text{(Hessian 대각 성분 활용)}$$

더 정확한 중요도 추정이 가능하나 계산 비용 증가를 고려해야 합니다.

#### ③ Foundation Model / LLM 환경으로의 확장

사전학습된 대형 언어모델(LLM) 파인튜닝 시 노이즈 레이블 문제는 매우 중요합니다. CDR의 파라미터 분류 아이디어를 **LoRA**, **Adapter** 등의 Parameter-Efficient Fine-Tuning(PEFT)에 적용하는 연구가 유망합니다.

#### ④ 클린 검증셋 없는 환경(Fully Unsupervised Noise Detection)

CDR은 Early Stopping을 위해 **노이즈 검증셋**을 사용합니다. 완전한 클린 데이터 없는 환경에서도 작동하도록 **자기지도학습(self-supervised)** 기반의 Early Stopping 기준 개발이 필요합니다.

#### ⑤ 이론적 수렴 보장

현재 CDR에는 수렴 이론이 없습니다. Non-Critical 파라미터를 비활성화하는 이 업데이트 규칙의 **수렴 조건 및 속도에 대한 이론적 분석**이 후속 연구에서 필요합니다.

#### ⑥ 다른 노이즈 유형으로의 확장

- **Open-set noise** (훈련 시 관련 없는 클래스 포함)
- **Feature-dependent noise** (입력 특징에 의존하는 복잡한 노이즈)
- **Long-tail 분포 + 노이즈**의 복합 시나리오

---

## 참고 자료 출처

1. **주 논문:** Xia, X., Liu, T., Han, B., Gong, C., Wang, N., Ge, Z., & Chang, Y. (2021). *Robust early-learning: Hindering the memorization of noisy labels.* ICLR 2021. (첨부된 PDF)

2. **Lottery Ticket Hypothesis:** Frankle, J., & Carbin, M. (2018). *The lottery ticket hypothesis: Finding sparse, trainable neural networks.* arXiv:1803.03635.

3. **DivideMix:** Li, J., Socher, R., & Hoi, S. C. H. (2020). *DivideMix: Learning with noisy labels as semi-supervised learning.* ICLR 2020.

4. **Early-Learning Regularization (ELR):** Liu, S., Niles-Weed, J., Razavian, N., & Fernandez-Granda, C. (2020). *Early-learning regularization prevents memorization of noisy labels.* arXiv:2007.00151.

5. **Co-teaching:** Han, B., Yao, Q., Yu, X., et al. (2018). *Co-teaching: Robust training of deep neural networks with extremely noisy labels.* NeurIPS 2018.

6. **Memorization Effects:** Arpit, D., et al. (2017). *A closer look at memorization in deep networks.* arXiv:1706.05394.

7. **CORES:** Cheng, H., Zhu, Z., Li, X., et al. (2021). *Learning with instance-dependent label noise: A sample sieve approach.* ICLR 2021.

8. **GCE:** Zhang, Z., & Sabuncu, M. (2018). *Generalized cross entropy loss for training deep neural networks with noisy labels.* NeurIPS 2018.

9. **APL:** Ma, X., Huang, H., Wang, Y., et al. (2020). *Normalized loss functions for deep learning with noisy labels.* ICML 2020.

10. **Generalization via compression:** Arora, S., Ge, R., Neyshabur, B., & Zhang, Y. (2018). *Stronger generalization bounds for deep nets via a compression approach.* ICML 2018.
