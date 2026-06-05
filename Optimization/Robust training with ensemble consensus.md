# Robust Training with Ensemble Consensus

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음 두 가지 관찰에 기반합니다:

1. **Noisy 예제는 암기(memorization)를 통해 학습됨**: Noisy 예제는 다른 훈련 예제들과의 상관관계가 낮아 패턴 학습이 아닌 암기를 통해 학습됩니다.

2. **암기된 특징은 perturbation에 취약함**: DNN은 암기된 특징의 근방(neighborhood)으로 일반화하지 못하기 때문에, 특정 perturbation 하에서 noisy 예제의 손실(loss)은 쉽게 증가하지만, clean 예제의 손실은 안정적으로 작게 유지됩니다.

이를 수식으로 표현하면:

- **Noisy 예제에 대해**: $(x, y) \in \mathcal{D}_{noisy}$인 경우,

$$
(x, y) \in \mathcal{S}_{\epsilon, \mathcal{D}, \theta} \Rightarrow (x, y) \notin \mathcal{S}_{\epsilon, \mathcal{D}, \theta+\delta}
$$

- **Clean 예제에 대해**: $(x, y) \in \mathcal{D}_{clean}$인 경우,

$$
(x, y) \in \mathcal{S}_{\epsilon, \mathcal{D}, \theta} \Rightarrow (x, y) \in \mathcal{S}_{\epsilon, \mathcal{D}, \theta+\delta}
$$

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| LEC 프레임워크 제안 | Ensemble Consensus Filtering을 활용한 강건한 훈련 방법론 |
| 3가지 perturbation 변형 제안 | LNEC, LSEC, LTEC |
| Small-loss 기준의 한계 극복 | 기존 small-loss 기준만으로는 noisy 예제를 완전히 걸러낼 수 없음을 실험적으로 증명 |
| 계산 효율성 | LTEC는 단일 네트워크로 구현 가능 |
| 아키텍처 독립성 | SGD로 최적화되는 모든 아키텍처에 적용 가능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**핵심 문제**: 레이블 노이즈(Label Noise) 상황에서 DNN이 noisy 예제를 암기하여 과적합(overfitting)되는 문제입니다.

기존의 small-loss 기준 방법론의 문제점을 명확히 지적합니다. 노이즈 비율 $\epsilon$%가 주어졌을 때, 네트워크 $f_\theta$에서 $(100-\epsilon)$% small-loss 예제 집합 $\mathcal{S}_{\epsilon, \mathcal{D}, \theta}$에도 여전히 noisy 예제가 포함될 수 있습니다. 특히 **높은 노이즈 수준(sym-60%)에서 이 문제가 심각**해집니다.

### 2.2 제안하는 방법 (LEC: Learning with Ensemble Consensus)

#### 핵심 아이디어: Ensemble Consensus Filtering

$M$개의 perturbation $\delta_1, \delta_2, \ldots, \delta_M$을 $\theta$에 추가하여 생성된 앙상블 네트워크들의 small-loss 예제의 교집합을 clean 예제로 간주합니다:

$$
\bigcap_{m=1}^{M} \mathcal{S}_{\epsilon, \mathcal{D}, \theta+\delta_m}
$$

#### 알고리즘 구조 (LEC, Algorithm 1 기반)

**Warming-up 단계** ($t = 1$ to $T_w$):

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \frac{1}{|\mathcal{B}_b|} \sum_{(x,y) \in \mathcal{B}_b} CE(f_\theta(x), y)
$$

**Filtering 단계** ($t = T_w + 1$ to $T_{end}$):

$$
\theta_m = \theta + \delta_{m,b,t} \quad \text{(perturbation 추가)}
$$

$$
\mathcal{B}'_b = \bigcap_{m=1}^{M} \mathcal{S}_{\epsilon, \mathcal{B}_b, \theta_m} \quad \text{(Ensemble Consensus Filtering)}
$$

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \frac{1}{|\mathcal{B}'_b|} \sum_{(x,y) \in \mathcal{B}'_b} CE(f_\theta(x), y)
$$

### 2.3 세 가지 LEC 변형

#### (1) LNEC (Network-Ensemble Consensus)

**원리**: 동일 아키텍처의 여러 네트워크는 일반화 시 상관관계가 높고, 암기 시 상관관계가 낮다는 Morcos et al. (2018)의 관찰에 기반합니다.

- $M$개의 네트워크를 독립적으로 초기화 및 훈련
- Perturbation $\delta$는 $M$개 네트워크 간의 차이에서 발생
- **공간 복잡도**: $Mm$ (파라미터 수), **계산 복잡도**: $Mn$ (forward pass)

#### (2) LSEC (Self-Ensemble Consensus)

**원리**: 암기된 특징에 대한 예측은 불확실하고, 일반화된 특징에 대한 예측은 확실하다는 점을 활용합니다 (Lakshminarayanan et al., 2017).

- 단일 네트워크에서 **Dropout 등 확률적 연산**으로 $M$번의 stochastic forward pass 수행:

$$
\theta_m = \theta + \delta_m \quad \text{where } \delta_m \text{ comes from stochasticity (e.g., dropout)}
$$

$$
\mathcal{B}'_b = \bigcap_{m=1}^{M} \mathcal{S}_{\epsilon, \mathcal{B}_b, \theta_m}
$$

#### (3) LTEC (Temporal-Ensemble Consensus) - 최고 성능

**원리**: 훈련 중 비전형적(atypical) 특징은 전형적(typical) 특징보다 더 쉽게 망각된다는 Toneva et al. (2018)의 관찰에 기반합니다.

- Perturbation $\delta$는 **현재와 이전 epoch의 네트워크 차이**에서 발생
- 현재 epoch $t$와 이전 $\min(M-1, t-1)$ epoch의 small-loss 예제 교집합으로 훈련:

$$
\mathcal{B}'_b = \mathcal{P}_{t-(M-1)} \cap \mathcal{P}_{t-(M-2)} \cap \cdots \cap \mathcal{P}_{t-1} \cap \mathcal{S}_{\epsilon, \mathcal{B}_b, \theta}
$$

여기서 $\mathcal{P}_t$는 epoch $t$에서의 $(100-\epsilon)$% small-loss 예제 집합입니다.

- **메모리 효율적**: 네트워크 파라미터 대신 small-loss 예제 집합을 저장
- **공간 복잡도**: $m$ (단일 네트워크), **계산 복잡도**: forward pass $n$, backward pass $\leq n$

### 2.4 모델 구조

| 구성 요소 | 세부 내용 |
|---|---|
| 기본 아키텍처 | 9-conv layer (Laine & Aila, 2016 기반) |
| 추가 레이어 | Batch Norm, LReLU($\alpha=0.01$), MaxPooling, Dropout(0.25), Global Average Pooling |
| 최적화 | Adam, LR=0.1, Batch size=128, 200 epochs |
| LR 스케줄 | 마지막 120 epochs (MNIST/CIFAR-10), 100 epochs (CIFAR-100) 동안 선형 감소 |
| 검증용 아키텍처 | ResNet-20 (He et al., 2016) |

### 2.5 성능 향상

#### 랜덤 레이블 노이즈 결과 (최종 테스트 정확도, %)

| Dataset | Noise | Standard | Co-teaching | LTEC | LTEC-full |
|---|---|---|---|---|---|
| CIFAR-10 | sym-20% | 79.50 | 85.46 | **88.18** | 88.16 |
| CIFAR-10 | sym-60% | 41.91 | 75.01 | **80.38** | 79.13 |
| CIFAR-10 | asym-40% | 57.50 | 79.53 | **86.36** | 84.56 |
| CIFAR-100 | sym-60% | 20.79 | 43.36 | **46.24** | 45.77 |
| CIFAR-100 | asym-40% | 37.64 | 40.99 | **47.70** | 45.49 |

> CIFAR-100 asym-40%에서 Co-teaching 대비 약 **6%p** 성능 향상

#### ResNet-20 적용 결과 (LTEC)

| Noise | Standard | LTEC |
|---|---|---|
| sym-20% | 81.31 | **89.01** |
| sym-60% | 61.94 | **81.46** |
| asym-40% | 62.76 | **86.62** |

### 2.6 한계점

1. **노이즈 비율 $\epsilon$ 사전 필요**: 실제 환경에서는 정확한 노이즈 비율을 알기 어렵습니다. (단, 과대 추정 시 오히려 성능 향상 경향 있음)

2. **의미론적(Semantic) 노이즈에 취약**: 의미론적으로 생성된 noisy 예제는 다른 훈련 예제와 상관관계가 높아 ensemble consensus filtering으로 걸러내기 어렵습니다.

3. **Warming-up 과정의 한계**: Warming-up 단계에서 모든 clean 예제를 충분히 학습하기 어려워, filtering 초기에 clean 예제가 제거될 수 있습니다. (단, Recall이 학습 초반 50 epoch 이후 빠르게 회복됩니다)

4. **$M$ 값 선택의 민감성**: $M$이 너무 크면 오히려 너무 많은 예제가 제거되어 성능이 떨어집니다 ($M = \infty$일 때 급격한 성능 저하).

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상 메커니즘

LEC의 일반화 성능 향상은 근본적으로 **"암기 vs. 일반화"의 차이**를 exploit하는 데서 나옵니다.

**핵심 메커니즘 도식:**

```
[전체 훈련 데이터]
        ↓ Warming-up (SGD 초기 일반화 단계 활용)
[Small-loss 예제 집합 S_ε,D,θ]
        ↓ Ensemble Consensus Filtering
        ↓ ∩ᴹₘ₌₁ S_ε,D,θ+δₘ
[High-precision Clean 예제 집합] → 이 데이터로만 학습
        ↓
[향상된 일반화 성능]
```

### 3.2 Recall 분석을 통한 일반화 확인

논문에서 **Recall** = (훈련에 사용된 clean 예제 수) / (전체 clean 예제 수) × 100으로 정의하며, LTEC 실행 시 처음 50 epoch 내에 recall이 빠르게 증가함을 실험적으로 확인합니다.

이는 LEC가 처음에는 일부 clean 예제를 놓치더라도, 학습이 진행되면서 **점진적으로 더 많은 clean 예제를 회복**하여 일반화 성능을 높임을 의미합니다.

### 3.3 일반화 성능 향상에 기여하는 요소들

| 요소 | 역할 |
|---|---|
| **SGD Bias 활용** | SGD 최적화의 초기 단계에서 clean 패턴을 먼저 학습하는 특성을 warming-up에 활용 |
| **High Label Precision** | Ensemble consensus filtering으로 훈련 데이터의 label precision을 높여 더 순수한 데이터로 학습 |
| **과적합 방지** | Noisy 예제 제거로 DNN의 memorization 억제 |
| **아키텍처 독립성** | ResNet 포함 다양한 아키텍처에서도 일관적인 성능 향상 |

### 3.4 노이즈 추정 오류에 대한 강건성

과대 추정된 $\epsilon$ 값 사용 시 ($1.1\epsilon$), LTEC는 오히려 성능이 향상되는 경향이 있습니다. 이는 Clean 예제를 모두 학습하기 어렵기 때문에 더 넓은 범위에서 small-loss 예제를 선택하는 것이 실제로 도움이 됨을 시사합니다.

$$
\text{Label Precision} \nearrow \Rightarrow \text{Generalization} \nearrow
$$

---

## 4. 앞으로의 연구에 미치는 영향과 고려사항

### 4.1 앞으로의 연구에 미치는 영향

#### (1) Small-loss 패러다임의 발전

LEC는 기존 small-loss 기준의 한계를 명확히 하고, **더 정교한 샘플 선택 기준**의 필요성을 제시합니다. 이는 이후 DivideMix (Li et al., 2020), CORES² (Cheng et al., 2021) 등의 연구로 이어집니다.

#### (2) 앙상블 기반 불확실성 추정

LSEC의 아이디어는 **Bayesian Deep Learning**과 **Test-Time Augmentation**의 아이디어와 결합되어, 불확실성을 활용한 더 정교한 노이즈 레이블 탐지 연구의 기초가 됩니다.

#### (3) Semi-supervised Learning과의 연계

Noisy 예제를 완전히 버리는 대신, **unlabeled 데이터로 활용**하는 연구 방향을 암시합니다. (DivideMix 등이 이를 실현)

#### (4) 망각 현상(Forgetting) 연구

LTEC의 temporal ensemble 아이디어는 Toneva et al. (2018)의 **Example Forgetting** 연구와 연결되며, 학습 역학(training dynamics)을 활용한 데이터 정제 연구를 촉진합니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

#### 주요 후속 연구들

**① DivideMix (Li et al., NeurIPS 2020)**

LEC와 마찬가지로 small-loss 기준을 활용하지만, **GMM(Gaussian Mixture Model)을 사용하여 clean/noisy 분포를 분리**합니다. 또한 noisy 예제를 버리지 않고 unlabeled 데이터로 활용하는 semi-supervised 방식을 채택합니다.

$$
p(l | \ell_i) = \frac{w_{clean} \cdot \mathcal{N}(\ell_i; \mu_{clean}, \sigma^2_{clean})}{w_{clean} \cdot \mathcal{N}(\ell_i; \mu_{clean}, \sigma^2_{clean}) + w_{noisy} \cdot \mathcal{N}(\ell_i; \mu_{noisy}, \sigma^2_{noisy})}
$$

| 비교 항목 | LEC (LTEC) | DivideMix |
|---|---|---|
| 분류 방법 | Ensemble Consensus | GMM |
| Noisy 예제 처리 | 제거 | Unlabeled로 활용 |
| Semi-supervised | 미적용 | 적용 (MixMatch) |
| 추가 레이블 정보 | 불필요 | 불필요 |

**② CORES² (Cheng et al., ICLR 2021)**

**Confidence regularized self-training**으로 pseudo-label의 신뢰도를 활용하여 noisy 예제를 탐지합니다.

**③ ELR+ (Early-Learning Regularization, Liu et al., NeurIPS 2020)**

DNN의 초기 학습 단계(early learning)를 정규화하여 memorization을 방지합니다. LEC의 warming-up 개념과 유사한 동기를 공유합니다.

$$
\mathcal{L}_{ELR} = \frac{1}{n} \sum_{i=1}^{n} \left[ \mathcal{L}_{CE}(y_i, f(x_i; \theta)) - \lambda \log\left(1 - p_i \cdot f(x_i; \theta)\right) \right]
$$

**④ UNICON (Karim et al., CVPR 2022)**

**Contrastive Learning**을 noisy label 학습에 결합합니다. LEC의 앙상블 아이디어와 달리, representation learning 수준에서 clean/noisy를 분리합니다.

**⑤ SOP (Self-Supervised Noisy Label Learning, Liu et al., NeurIPS 2022)**

자기지도학습(self-supervised learning)을 전처리로 활용하여 LEC에서 지적한 warming-up의 한계를 극복하는 방향을 제시합니다.

#### 종합 비교 테이블 (CIFAR-10 sym-80% 기준, 참고용)

| 방법 | 연도 | CIFAR-10 (sym-80%) | 특징 |
|---|---|---|---|
| Co-teaching | 2018 | ~54% | Peer network |
| LTEC | 2020 | ~74% | Temporal ensemble |
| DivideMix | 2020 | ~93% | GMM + SSL |
| ELR+ | 2020 | ~93% | Early-learning reg. |
| UNICON | 2022 | ~95% | Contrastive learning |

> ⚠️ 주의: 위 수치는 각 논문의 실험 설정이 다를 수 있으므로 직접 비교 시 주의가 필요합니다. LTEC는 sym-60%까지 실험되었으며, sym-80% 수치는 추정치가 아닌 논문에 없습니다.

### 4.3 앞으로 연구 시 고려할 점

#### (1) Noisy 예제의 완전한 폐기 vs. 재활용

LEC는 noisy 예제를 훈련에서 제거하는 방식입니다. 그러나 **DivideMix처럼 noisy 예제를 unlabeled 데이터로 활용**하면 데이터 손실 없이 더 높은 성능을 달성할 수 있습니다. 향후 연구에서는 LEC의 필터링과 semi-supervised learning을 결합하는 것이 유망합니다.

#### (2) 노이즈 비율 추정의 자동화

LEC는 노이즈 비율 $\epsilon$을 사전에 알아야 합니다. **자동 노이즈 비율 추정** (예: GMM을 통한 동적 추정)과 결합하면 실용성이 크게 향상됩니다.

#### (3) 대규모 실제 데이터셋 검증

LEC는 MNIST, CIFAR-10/100에서 주로 검증되었습니다. **WebVision, Clothing1M, ANIMAL-10N 등 실제 noisy 데이터셋**에서의 검증이 필요합니다.

#### (4) Self-supervised Pretraining과의 결합

논문에서도 언급하듯, Hendrycks et al. (2019)의 pretraining이 filtering 과정에서 clean 예제 제거를 방지할 수 있습니다. **CLIP, DINO 등 강력한 self-supervised pretrained model**을 warming-up 대신 활용하면 성능이 크게 향상될 것으로 기대됩니다.

#### (5) 긴 꼬리 분포(Long-tail Distribution)와의 결합

실제 데이터는 클래스 불균형을 가지는 경우가 많습니다. **Long-tail 학습과 noisy label 학습을 동시에 고려**하는 연구가 필요합니다.

#### (6) $M$ 값의 적응적 선택

고정된 $M$ 값은 최적이 아닐 수 있습니다. **학습 진행 상황에 따라 $M$을 동적으로 조절**하는 적응적 앙상블 크기 선택 연구가 필요합니다.

#### (7) Contrastive Learning과의 결합

LEC의 ensemble consensus는 **표현 수준(representation level)에서의 일관성 측정**과 결합될 수 있습니다. 예를 들어, SimCLR이나 MoCo의 projection space에서 perturbation에 대한 표현의 안정성을 측정하면 더 정교한 noisy 탐지가 가능할 것입니다.

---

## 참고자료

**주요 참고 논문:**

1. **Lee, J., & Chung, S.-Y. (2020).** "Robust Training with Ensemble Consensus." *ICLR 2020.* (본 논문)

2. **Han, B., et al. (2018).** "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels." *NeurIPS 2018.*

3. **Li, J., et al. (2020).** "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." *ICLR 2020.*

4. **Liu, S., et al. (2020).** "Early-Learning Regularization Prevents Memorization of Noisy Labels." *NeurIPS 2020.*

5. **Arpit, D., et al. (2017).** "A Closer Look at Memorization in Deep Networks." *ICML 2017.*

6. **Toneva, M., et al. (2018).** "An Empirical Study of Example Forgetting during Deep Neural Network Learning." *arXiv:1812.05159.*

7. **Morcos, A., et al. (2018).** "Insights on Representational Similarity in Neural Networks with Canonical Correlation." *NeurIPS 2018.*

8. **Lakshminarayanan, B., et al. (2017).** "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS 2017.*

9. **Patrini, G., et al. (2017).** "Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach." *CVPR 2017.*

10. **Karim, M., et al. (2022).** "UniCon: Combating Label Noise Through Uniform Selection and Contrastive Learning." *CVPR 2022.*

11. **Cheng, H., et al. (2021).** "Learning with Instance-Dependent Label Noise: A Sample Sieve Approach." *ICLR 2021.*

> ⚠️ **정확도 주의사항**: 2020년 이후 최신 연구들의 수치 비교는 실험 설정이 다를 수 있으므로, 각 논문의 원본을 직접 확인하시기를 권장합니다. DivideMix, ELR+, UNICON의 수치는 각 논문에서 보고된 결과를 기반으로 하였으며, 실험 환경 차이로 인해 직접 비교에 한계가 있습니다.
