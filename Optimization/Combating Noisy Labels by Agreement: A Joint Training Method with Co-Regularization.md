# Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization (JoCoR)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 SOTA 방법인 **Decoupling**과 **Co-teaching+**는 두 네트워크 간의 **"불일치(Disagreement)"** 전략이 노이즈 레이블 학습의 핵심이라고 주장했습니다. 그러나 JoCoR은 이와 반대 관점에서 출발합니다:

> **"두 네트워크 간의 다양성을 줄이는 방향(Agreement Maximization)이 오히려 노이즈 레이블 문제를 더 효과적으로 해결한다."**

### 주요 기여

| 기여 | 설명 |
|------|------|
| **새로운 패러다임 제시** | Disagreement → Agreement로 관점 전환 |
| **Joint Training** | 두 네트워크를 하나의 손실 함수로 동시에 학습 |
| **Co-Regularization** | JS Divergence 기반 대조 손실로 두 네트워크의 예측 일치 유도 |
| **오류 흐름 분산** | Joint Loss 기반 small-loss 선택으로 편향된 샘플 선택의 오류가 한 네트워크에 누적되지 않도록 함 |
| **실험적 검증** | MNIST, CIFAR-10, CIFAR-100, Clothing1M에서 SOTA 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**심층 신경망(DNN)의 노이즈 레이블 기억(Memorization) 문제**

- DNN은 초기에 깨끗한 패턴을 먼저 학습하고, 이후 노이즈 레이블까지 기억(memorize)하는 경향이 있음 (Arpit et al., 2017)
- 기존 Disagreement 방법들의 한계:
  - **Decoupling**: 불일치 구간에 노이즈 레이블이 많아도 처리 불가
  - **Co-teaching+**: 극단적 노이즈(80%) 상황에서 mini-batch당 사용 가능한 샘플이 매우 적어짐
  - 두 방법 모두 선택된 샘플이 실제로 깨끗한 레이블임을 보장하지 못함

### 2.2 제안하는 방법 (수식 포함)

#### 전체 손실 함수 (Joint Loss)

$$\ell(\boldsymbol{x}_i) = (1 - \lambda) \cdot \ell_{\text{sup}}(\boldsymbol{x}_i, y_i) + \lambda \cdot \ell_{\text{con}}(\boldsymbol{x}_i) \tag{1}$$

여기서 $\lambda \in [0, 1]$는 두 손실의 균형을 조절하는 하이퍼파라미터.

#### 분류 손실 (Supervised Loss)

두 네트워크 $f(\boldsymbol{x}, \Theta_1)$, $f(\boldsymbol{x}, \Theta_2)$의 Cross-Entropy 합:

$$\ell_{\text{sup}}(\boldsymbol{x}_i, y_i) = \ell_{C1}(\boldsymbol{x}_i, y_i) + \ell_{C2}(\boldsymbol{x}_i, y_i)$$

$$= -\sum_{i=1}^{N}\sum_{m=1}^{M} y_i \log(p_1^m(\boldsymbol{x}_i)) - \sum_{i=1}^{N}\sum_{m=1}^{M} y_i \log(p_2^m(\boldsymbol{x}_i)) \tag{2}$$

#### 대조 손실 (Contrastive Loss / Co-Regularization)

Jensen-Shannon(JS) Divergence를 대칭 KL Divergence로 근사:

$$\ell_{\text{con}} = D_{\text{KL}}(\boldsymbol{p}_1 \| \boldsymbol{p}_2) + D_{\text{KL}}(\boldsymbol{p}_2 \| \boldsymbol{p}_1) \tag{3}$$

$$D_{\text{KL}}(\boldsymbol{p}_1 \| \boldsymbol{p}_2) = \sum_{i=1}^{N}\sum_{m=1}^{M} p_1^m(\boldsymbol{x}_i) \log \frac{p_1^m(\boldsymbol{x}_i)}{p_2^m(\boldsymbol{x}_i)}$$

$$D_{\text{KL}}(\boldsymbol{p}_2 \| \boldsymbol{p}_1) = \sum_{i=1}^{N}\sum_{m=1}^{M} p_2^m(\boldsymbol{x}_i) \log \frac{p_2^m(\boldsymbol{x}_i)}{p_1^m(\boldsymbol{x}_i)}$$

#### Small-loss 샘플 선택

각 mini-batch $D_n$에서 joint loss 기준으로 작은 손실의 샘플 집합 선택:

$$\tilde{D}_n = \arg\min_{D'_n : |D'_n| \geq R(t)|D_n|} \ell(D'_n) \tag{4}$$

선택된 샘플에 대한 평균 손실로 역전파:

$$L = \frac{1}{|\tilde{D}|} \sum_{\boldsymbol{x} \in \tilde{D}} \ell(\boldsymbol{x}) \tag{5}$$

#### R(t) 업데이트 (노이즈율 기반 동적 샘플 비율)

```math
R(t) = 1 - \min\left\{\frac{t}{T_k}\tau,\ \tau\right\}
```

초기에는 많은 샘플(큰 $R(t)$ )을 선택하고, 훈련이 진행될수록 점점 줄여 최종적으로 $1-\tau$ 비율 유지.

### 2.3 모델 구조 (Algorithm 1: JoCoR)

```
입력: 네트워크 f (파라미터 Θ₁, Θ₂), 학습률 η, 노이즈율 τ, 에폭 Tmax
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
for t = 1, 2, ..., Tmax:
  데이터셋 D를 셔플
  for n = 1, ..., Imax:
    mini-batch Dₙ 추출
    p₁ = f(x, Θ₁),  p₂ = f(x, Θ₂)    ← 두 네트워크 동시 예측
    joint loss ℓ 계산  (수식 1)
    small-loss 집합 D̃ₙ 선택  (수식 4)
    평균 손실 L 계산  (수식 5)
    Θ ← Θ - η∇L                        ← 두 네트워크 동시 업데이트
  R(t) 업데이트
출력: Θ₁, Θ₂
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**구조적 특징 (Pseudo-Siamese Network)**:
- 두 네트워크는 동일한 아키텍처, 다른 초기화 → 다양한 학습 능력 보유
- 파라미터는 별도이지만 **하나의 Joint Loss로 동시에 업데이트**
- 추론 시에는 각 네트워크가 독립적으로 예측 가능

**타 방법과의 비교**:

| 특징 | Decoupling | Co-teaching | Co-teaching+ | JoCoR |
|------|:---:|:---:|:---:|:---:|
| Small-loss 선택 | ✗ | ✓ | ✓ | ✓ |
| Cross Update | ✗ | ✓ | ✓ | ✗ |
| Joint Training | ✗ | ✗ | ✗ | ✓ |
| Disagreement | ✓ | ✗ | ✓ | ✗ |
| Agreement (Co-Reg) | ✗ | ✗ | ✗ | ✓ |

### 2.4 성능 향상

#### MNIST (Table 2)

| 노이즈 설정 | Standard | Co-teaching | Co-teaching+ | **JoCoR** |
|---|---|---|---|---|
| Symmetry-20% | 79.56 | 95.10 | 97.81 | **98.06** |
| Symmetry-50% | 52.66 | 89.82 | 95.80 | **96.64** |
| Symmetry-80% | 23.43 | 79.73 | 58.92 | **84.89** |
| Asymmetry-40% | 79.00 | 90.28 | 93.28 | **95.24** |

#### CIFAR-10 (Table 3)

| 노이즈 설정 | Standard | Co-teaching | Co-teaching+ | **JoCoR** |
|---|---|---|---|---|
| Symmetry-20% | 69.18 | 78.23 | 78.71 | **85.73** |
| Symmetry-50% | 42.71 | 71.30 | 57.05 | **79.41** |
| Symmetry-80% | 16.24 | 26.58 | 24.19 | **27.78** |
| Asymmetry-40% | 69.43 | 73.78 | 68.84 | **76.36** |

#### Clothing1M (Table 5, 실제 노이즈)

| Methods | Best | Last |
|---|---|---|
| Standard | 67.22 | 64.68 |
| Co-teaching | 69.21 | 68.51 |
| Co-teaching+ | 59.32 | 58.79 |
| **JoCoR** | **70.30** | **69.79** |

### 2.5 한계

1. **노이즈율($\tau$) 사전 지식 필요**: R(t) 계산에 $\tau$를 알아야 함 (실제 환경에서는 알기 어려움)
2. **하이퍼파라미터 $\lambda$ 민감성**: 클린 검증셋이 필요하거나, 노이즈 검증셋에서 small-loss 선택으로 대체해야 함
3. **계산 비용**: 두 네트워크를 동시에 학습하므로 단일 네트워크 대비 약 2배 메모리/연산 필요
4. **극단적 노이즈(Symmetry-80%)에서 불안정**: 표준편차가 크게 나타남 ($84.89 \pm 4.55$)
5. **CIFAR-100 Asymmetry-40%에서 Co-teaching+에 미세하게 뒤짐** ($32.70$ vs $33.62$)
6. **이론적 기반 미비**: 논문 자체에서 향후 과제로 명시

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Co-Regularization이 일반화에 기여하는 메커니즘

논문은 **Co-Regularization이 더 넓은(wider) 손실 최솟값(loss minimum)을 찾도록 유도**하여 일반화 성능을 향상시킨다고 주장합니다 (Zhang et al., 2018의 Deep Mutual Learning 인용).

**이론적 직관**: 두 네트워크가 서로의 예측에 동의하도록 강제될 때:
- 각 네트워크는 단독으로 찾을 수 없는 **편평한(flat) 손실 경관**으로 수렴
- Flat minimum은 일반적으로 더 높은 일반화 성능과 연관됨 (Sharp vs. Flat minimum 연구)

$$\text{일반화 효과: } \ell_{\text{con}} \downarrow \Rightarrow \text{두 네트워크 예측 일치} \Rightarrow \text{Flat Minimum 탐색} \Rightarrow \text{일반화} \uparrow$$

### 3.2 Ablation Study를 통한 검증

CIFAR-10 Symmetry-50% 실험:

| 방법 | 역할 | 관찰 |
|---|---|---|
| Standard+ | Small-loss만 적용 | 최고점 이후 하락 |
| Co-teaching | Cross-update | 최고점 이후 하락 |
| **Joint-only** ($\lambda=0$) | Joint Training만 | Co-teaching과 유사, 레이블 정밀도 더 높음 |
| **JoCoR** | Joint + Co-Reg | 지속적으로 높은 정확도 유지 |

→ **Co-Regularization이 없으면 일반화 성능이 점차 하락**, Co-Regularization이 핵심임을 확인

### 3.3 레이블 정밀도(Label Precision)와 일반화의 연관성

JoCoR는 훈련이 진행될수록 레이블 정밀도가 **지속적으로 증가**하는 반면, Co-teaching은 정점 이후 하락:

```math
\text{Label Precision} = \frac{\text{\# of clean labels in } \tilde{D}}{\text{\# of all selected labels in } \tilde{D}}
```

이는 Co-Regularization이 깨끗한 샘플 선택 능력을 시간이 지남에 따라 강화함을 의미하며, 이것이 일반화 성능 향상의 핵심 메커니즘.

### 3.4 CIFAR-10 Asymmetry-40%에서 관찰된 일반화 우위

- 처음 100 epoch: Co-teaching > JoCoR
- 100 epoch 이후: **JoCoR > Co-teaching** (지속적)

이는 JoCoR이 장기 훈련에서 더 강한 일반화 성능을 보임을 시사.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

#### (1) 패러다임 전환 촉매
- Disagreement → **Agreement** 기반 학습으로의 관점 전환을 제시
- 이후 연구들이 단순 샘플 선택을 넘어 **정규화 기반 접근법**으로 확장하는 데 영향

#### (2) 반지도학습과의 융합 가능성 제시
- Co-Regularization을 노이즈 레이블에 적용한 선구적 시도
- 이후 연구들이 선택된 클린 샘플 → **지도학습**, 나머지 → **비지도/반지도학습**으로 처리하는 하이브리드 방법 개발로 발전

#### (3) 손실 함수 설계에 대한 새로운 시각
- Joint Loss 개념이 이후 다중 네트워크 협력 학습 연구에 영향

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 연구들은 제가 훈련 데이터 기반으로 아는 내용이며, 논문 원문을 직접 확인하지 못한 부분이 있습니다. 정확한 수치와 내용은 원 논문을 참조하시기 바랍니다.

#### DivideMix (Li et al., ICLR 2020)
- **핵심 아이디어**: GMM(Gaussian Mixture Model)으로 클린/노이즈 샘플 분리 후, MixMatch 반지도학습 적용
- **JoCoR 대비 발전**: JoCoR이 단순 small-loss 선택에 의존하는 반면, DivideMix는 확률적 모델링으로 클린 샘플 분리 + 반지도학습으로 노이즈 샘플까지 활용
- **성능**: CIFAR-10 90% 노이즈에서 JoCoR 대비 크게 향상
- **한계**: 하이퍼파라미터 민감도 높음

#### ELR (Early Learning Regularization, Liu et al., NeurIPS 2020)
- **핵심 아이디어**: 초기 학습 단계의 예측을 temporal ensemble로 저장하고, 현재 예측과의 일치를 정규화 항으로 추가
- **JoCoR과의 유사점**: 정규화 기반 접근, memorization 억제
- **차이점**: 단일 네트워크, 시간적 일관성 활용

#### SELF (Nguyen et al., 2020) 및 CORES (Cheng et al., 2021)
- 신뢰도 기반 샘플 선택과 반지도학습을 결합

#### SOP (Shi et al., NeurIPS 2021)
- **핵심 아이디어**: 각 샘플마다 과최적화(over-parametrize)된 레이블 변수를 도입하여 노이즈를 흡수
- JoCoR보다 이론적으로 엄밀한 수렴 보장

#### 비교 표 (CIFAR-10 기준, 참고용)

| 방법 | 연도 | 접근법 | Sym-80% (참고치) |
|---|---|---|---|
| JoCoR | 2020 | Joint Loss + Co-Reg | ~27.78 |
| DivideMix | 2020 | GMM + MixMatch | ~93+ |
| ELR | 2020 | Temporal Regularization | ~85+ |
| SOP | 2021 | Over-parametrization | ~89+ |

> ⚠️ 위 수치는 서로 다른 네트워크 아키텍처와 설정에서 측정된 것으로, 직접 비교 시 주의 필요

**발전 방향 분석**: JoCoR 이후의 연구들은 공통적으로:
1. **확률적 노이즈 모델링** (단순 threshold → GMM 등)
2. **반지도학습과의 융합** (노이즈 샘플 버리지 않고 활용)
3. **이론적 보장 강화**

방향으로 발전했음을 알 수 있습니다.

### 4.3 향후 연구 시 고려할 점

#### (1) 노이즈율 추정의 자동화
현재 JoCoR은 $\tau$를 사전에 알아야 합니다. 향후 연구에서는:

$$\hat{\tau} = \text{argmin}_\tau \mathcal{V}(\tau; \mathcal{D}_{\text{val}})$$

검증셋 없이 자동으로 노이즈율을 추정하는 방법 통합이 필요합니다.

#### (2) 단일 네트워크로의 지식 증류
훈련 후 두 네트워크의 지식을 단일 네트워크로 증류하면 추론 비용을 절감할 수 있습니다:

$$\mathcal{L}_{\text{distill}} = D_{\text{KL}}(p_{\text{teacher}} \| p_{\text{student}})$$

#### (3) 이론적 기반 강화
- Rademacher Complexity 분석으로 Co-Regularization의 일반화 bound 도출
- PAC 학습 프레임워크에서의 수렴 보장

#### (4) 다양한 노이즈 유형 대응
- **Instance-dependent noise** (샘플별 다른 노이즈율): 현재 JoCoR은 주로 class-conditional noise 가정
- **Open-set noise** (훈련 레이블에 없는 클래스의 샘플)

#### (5) 대규모 데이터셋 및 대형 모델 적용
- 두 네트워크 동시 학습의 메모리 효율화
- Vision Transformer(ViT) 등 대형 모델에서의 적용성 검토

#### (6) 연속/온라인 학습 환경에서의 적용
- 데이터 스트림 환경에서 노이즈 레이블 실시간 처리

#### (7) 반지도학습과의 더 깊은 통합
선택된 clean 샘플과 나머지 샘플을 모두 활용:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{JoCoR}}(\tilde{D}) + \alpha \cdot \mathcal{L}_{\text{semi}}(D \setminus \tilde{D})$$

---

## 참고 자료

**주요 참고 문헌 (논문 원문 기재 기준)**

1. **본 논문**: Wei, H., Feng, L., Chen, X., & An, B. (2020). *Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization*. CVPR 2020. (제공된 PDF 원문)

2. Han, B., et al. (2018). *Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels*. NeurIPS 2018.

3. Yu, X., et al. (2019). *How does disagreement benefit co-teaching?* (Co-teaching+). arXiv:1901.04215.

4. Malach, E., & Shalev-Shwartz, S. (2017). *Decoupling "When to Update" from "How to Update"*. NeurIPS 2017.

5. Arpit, D., et al. (2017). *A Closer Look at Memorization in Deep Networks*. ICML 2017.

6. Zhang, Y., et al. (2018). *Deep Mutual Learning*. CVPR 2018.

7. Patrini, G., et al. (2017). *Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach*. CVPR 2017.

8. Sindhwani, V., Niyogi, P., & Belkin, M. (2005). *A Co-regularization Approach to Semi-supervised Learning with Multiple Views*. ICML Workshop.

9. Li, J., et al. (2020). *DivideMix: Learning with Noisy Labels as Semi-supervised Learning*. ICLR 2020.

10. Liu, S., et al. (2020). *Early-Learning Regularization Prevents Memorization of Noisy Labels*. NeurIPS 2020.
