# Self-Adaptive Training: beyond Empirical Risk Minimization 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **Empirical Risk Minimization(ERM)이 노이즈가 포함된 훈련 데이터에서 과적합(overfitting)을 유발하여 일반화 성능을 저하시킨다**는 문제를 지적하고, 이를 해결하기 위한 **Self-Adaptive Training(SAT)** 알고리즘을 제안합니다.

핵심 통찰은 다음과 같습니다:
- 딥러닝 모델이 학습 초기에 이미 유용한 정보를 예측값에 담고 있으며, 이 예측값을 훈련 과정에 피드백으로 활용하면 노이즈에 대한 견고성과 일반화 성능을 동시에 향상시킬 수 있다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| ERM 실패 분석 | 4가지 노이즈 유형에 걸쳐 ERM의 과적합 패턴을 체계적으로 분석 |
| SAT 알고리즘 제안 | 추가 계산 비용 없이 모델 예측을 훈련 목표에 동적으로 반영 |
| Double-descent 현상 해소 | ERM의 double-descent 오류-용량 곡선 vs. SAT의 단조 감소 곡선 |
| 라벨 노이즈 분류 | CIFAR 데이터셋에서 최대 9.3% 절대적 정확도 향상 |
| Selective Classification | 두 데이터셋에서 최대 50% 상대적 성능 향상 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**표준 ERM**은 다음 손실을 최소화합니다:

$$\mathcal{L}_{\text{ERM}}(f) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{j} y_{i,j} \log p_{i,j}$$

여기서 $p_i = \text{softmax}(f(x_i))$이고, $y_i$는 (노이즈가 포함될 수 있는) 레이블입니다.

**문제점:**
- ERM은 훈련 데이터의 40%가 오염되어도 훈련 정확도가 거의 100%에 수렴하는 **완전한 과적합** 발생
- 노이즈 종류에 따라 일반화 동작이 크게 달라지지만, 노이즈 훈련 세트의 정확도 곡선만으로는 이를 식별 불가
- Early stopping은 일부 노이즈 유형에서만 효과적이고 다른 유형에서는 오히려 성능 저하

### 2.2 제안하는 방법 및 수식

#### (1) 지수 이동 평균(Exponential Moving Average, EMA) 기반 목표 업데이트

$c$-class 분류 문제에서 $i$번째 샘플의 훈련 목표 $\boldsymbol{t}_i$를 다음과 같이 초기화하고 업데이트합니다:

$$\boldsymbol{t}_i \leftarrow \boldsymbol{y}_i \quad \text{(초기화, 첫 } E_s \text{ 에폭)}$$

$$\boldsymbol{t}_i \leftarrow \alpha \times \boldsymbol{t}_i + (1 - \alpha) \times \boldsymbol{p}_i \quad \text{(} e > E_s \text{ 이후 매 에폭)}$$

여기서:
- $\boldsymbol{p}_i = \text{softmax}(f(\boldsymbol{x}_i))$: 모델의 현재 예측값
- $\alpha \in [0,1)$: 모멘텀 항 (기본값: $\alpha = 0.9$)
- $E_s$: 초기 고정 에폭 수 (기본값: $E_s = 60$)

#### (2) 샘플 재가중화(Sample Re-weighting)

훈련 목표 $\boldsymbol{t}_i$로부터 각 샘플의 가중치를 다음과 같이 설정합니다:

$$w_i = \max_j \, t_{i,j}$$

이 가중치는 $w_i \in \left[\frac{1}{c}, 1\right]$ 범위를 가지며, 레이블 신뢰도를 반영합니다.

#### (3) 최종 SAT 손실 함수

위 두 요소를 결합하여 다음 목적함수를 SGD로 최소화합니다:

$$\mathcal{L}(f) = -\frac{1}{\sum_i w_i} \sum_i w_i \sum_j t_{i,j} \log p_{i,j} $$

분모의 $\sum_i w_i$는 샘플 가중치를 정규화하여 손실 스케일을 안정화합니다.

#### (4) 적대적 훈련 적용 (TRADES와 결합)

TRADES의 손실 함수:

```math
\mathbb{E}_{\boldsymbol{x},\boldsymbol{y}}\left\{ \text{CE}(\boldsymbol{p}(\boldsymbol{x}), \boldsymbol{y}) + \max_{\|\tilde{\boldsymbol{x}}-\boldsymbol{x}\|_\infty \leq \epsilon} \text{KL}(\boldsymbol{p}(\boldsymbol{x}), \boldsymbol{p}(\tilde{\boldsymbol{x}}))/\lambda \right\}
```

SAT는 위 수식의 CE 항을 식 (1)로 대체합니다.

#### (5) Selective Classification 적용

$(c+1)$번째 추상 클래스를 도입하고, 다음 손실을 최소화합니다:

$$\mathcal{L}(f) = -\frac{1}{m} \sum_i \left[ t_{i,y_i} \log p_{i,y_i} + (1 - t_{i,y_i}) \log p_{i,c} \right] $$

여기서:
- $t_{i,y_i}$가 작으면 불확실한 샘플로 간주하여 기권(abstain)을 학습
- $t_{i,y_i} \approx 1$이면 표준 cross-entropy로 수렴

#### 알고리즘 의사코드 요약

```
Algorithm 1: Self-Adaptive Training
입력: 데이터 {(x_i, y_i)}, 초기 목표 {t_i} = {y_i}, E_s = 60, α = 0.9
반복:
  미니배치 {(x_i, t_i)} 추출 (현재 에폭 e)
  p_i = softmax(f(x_i))
  if e > E_s: t_i = α × t_i + (1-α) × p_i
  w_i = max_j t_{i,j}
  L(f) = -(1/Σw_i) Σ_i w_i Σ_j t_{i,j} log p_{i,j}로 SGD 업데이트
종료 조건까지 반복
```

### 2.3 모델 구조

SAT는 **기존 네트워크 아키텍처를 전혀 변경하지 않습니다.** 실험에 사용된 백본 모델들:

| 실험 | 백본 모델 |
|---|---|
| CIFAR10/100 라벨 노이즈 | ResNet-34, Wide ResNet 28-10 |
| Double-descent 분석 | ResNet-18 (너비 변화) |
| 적대적 훈련 | WRN-34-10 |
| ImageNet | ResNet-50 |
| Selective Classification | VGG-16 (BN + Dropout) |

추가로 유지해야 하는 것은 **각 샘플의 누적 예측 벡터** $\boldsymbol{t}_i \in \mathbb{R}^c$ 뿐입니다. ImageNet 기준으로 약 $1.2 \times 10^6 \times 1000 \times 32\text{bit} \approx 4.47\text{GB}$의 메모리가 필요합니다.

### 2.4 성능 향상

#### 라벨 노이즈 분류 (CIFAR, Table 1)

| 백본 | 노이즈율 | ERM | DAC | Ours |
|---|---|---|---|---|
| ResNet-34 | 20% | 85.57 | 92.91 | **94.14** |
| ResNet-34 | 80% | 60.99 | 74.84 | **78.58** |
| WRN28-10 | 40% | 83.40 | 90.93 | **93.23** |
| WRN28-10 | 80% | 63.54 | 70.80 | **80.13** |

#### ImageNet (Table 3)

| 노이즈율 | ERM | Ours |
|---|---|---|
| 0% | 76.8 | **77.2** |
| 40% | 69.5 | **71.5** |

#### 적대적 훈련 (CIFAR10)

TRADES 대비 강인 정확도(Robust Accuracy) **1~3% 향상**

#### OOD 일반화 (Table 7, CIFAR10-C)

| 방법 | CIFAR10 | Level 3 | Level 5 |
|---|---|---|---|
| ERM | 95.32 | 77.26 | 58.91 |
| Ours | **95.80** | **78.83** | **60.77** |

### 2.5 한계점

1. **소용량 모델에서의 실패**: 표준 ResNet-18보다 10배 이상 작은 모델에서는 ERM보다 성능이 낮을 수 있음. 모델이 충분한 정보를 포착하지 못할 때 모호한 예측값을 피드백하면 오히려 학습을 방해함.

2. **인공적 노이즈 중심 평가**: 대부분의 실험이 인위적으로 주입된 노이즈를 대상으로 하며, 실제 환경의 자연 노이즈를 완전히 대표하지 못할 수 있음 (논문 자체에서도 인정).

3. **하이퍼파라미터 의존성**: $E_s$와 $\alpha$ 설정이 데이터셋 및 노이즈 수준에 따라 조정이 필요함 (다만 논문에서는 민감도가 낮다고 주장).

4. **메모리 비용**: 각 샘플의 소프트 레이블 벡터 유지 필요 (ImageNet 기준 약 4.47GB).

5. **이론적 보장 부족**: EMA 및 재가중화의 수렴 이론, 일반화 오차 한계에 대한 엄밀한 이론적 분석이 제시되지 않음.

---

## 3. 모델의 일반화 성능 향상 가능성 (심층 분석)

### 3.1 핵심 메커니즘: 왜 일반화가 향상되는가?

#### (a) 레이블 복원(Label Recovery) 효과

모델이 초기 에폭에서 정상 데이터를 먼저 학습하는 경향(clean data first fitting)을 활용합니다. EMA 목표 $\boldsymbol{t}_i$가 반복 업데이트되면서:

$$\boldsymbol{t}_i^{(e)} = \alpha^{e-E_s} \boldsymbol{y}_i + (1-\alpha)\sum_{k=E_s}^{e-1} \alpha^{e-1-k} \boldsymbol{p}_i^{(k)}$$

이 점진적 보정을 통해 노이즈 레이블을 실제 레이블로 수정합니다. 실험에서 CIFAR10 40% 노이즈 환경 기준 **레이블 복원 정확도 94.6%**, ImageNet 기준 **81.1%**를 달성합니다.

#### (b) 암묵적 정규화 효과

재가중화 $w_i = \max_j t_{i,j}$는 불확실한 샘플에 낮은 가중치를 부여하여:
- 노이즈 샘플의 기여도를 자동으로 줄임
- 높은 신뢰도 샘플에 더 집중하여 결정 경계를 더 명확하게 학습

#### (c) Double-Descent 현상 억제

ERM은 모델 용량(width parameter)에 따라 test error가 감소→증가→감소하는 double-descent를 보이지만, SAT는 단조 감소(single-descent)를 보입니다. 이는 SAT가 노이즈 과적합을 방지하여 편향-분산 트레이드오프를 개선한다는 것을 시사합니다.

$$\text{ERM: } \underbrace{\searrow \nearrow \searrow}_{\text{double-descent}} \quad \text{SAT: } \underbrace{\searrow}_{\text{single-descent}}$$

#### (d) OOD 일반화

CIFAR10-C 벤치마크에서 corruption이 심할수록(Level 5) SAT의 이점이 커집니다:
- Level 1: ERM 88.44% → SAT **89.41%** (+0.97%)
- Level 5: ERM 58.91% → SAT **60.77%** (+1.86%)

이는 SAT가 분포 이동(distribution shift)에 대한 암묵적 정규화를 제공함을 시사합니다.

### 3.2 일반화 향상의 이론적 배경

논문은 [Li et al., 2019]의 이론적 결과를 인용합니다: 과모수화된 신경망에서 gradient descent는 초기에 정상 레이블을 맞추고, 후기에 노이즈 레이블을 과적합합니다. SAT는 이 메커니즘을 활용하여:

1. $E_s$ 에폭 동안 모델이 유의미한 패턴을 학습하도록 ERM 방식으로 진행
2. $E_s$ 이후부터 모델 예측을 피드백하여 노이즈 과적합을 차단

---

## 4. 미래 연구에 미치는 영향 및 고려사항

### 4.1 미래 연구에 미치는 영향

#### (A) 소프트 레이블(Soft Label) 패러다임의 확장

SAT는 기존 label smoothing, mixup 등의 소프트 레이블 방법론과 달리 **동적으로 업데이트되는 데이터 적응형 소프트 레이블**을 사용합니다. 이는 정적 하이퍼파라미터 기반 소프트 레이블의 한계를 극복하며, 향후 연구에서 온라인 적응형 레이블 정제 방향으로 발전 가능합니다.

#### (B) 적대적 강건성(Adversarial Robustness) 연구

TRADES와의 결합이 성능 향상을 보이며, 이는 SAT가 다양한 적대적 훈련 알고리즘(PGD-AT, MART, AWP 등)의 플러그인 모듈로 활용될 수 있음을 시사합니다.

#### (C) Double-Descent 현상 이해

SAT가 double-descent를 억제한다는 실험 결과는 이 현상이 본질적으로 노이즈 과적합에 기인한다는 가설을 지지하며, 일반화 이론 연구에 새로운 방향을 제시합니다.

#### (D) 자기 훈련(Self-Training) 및 준지도 학습

SAT의 EMA 기반 예측 피드백 메커니즘은 자기 훈련, 지식 증류, 준지도 학습 등과 자연스럽게 연결됩니다.

### 4.2 앞으로 연구 시 고려할 점

#### (1) 이론적 분석 강화
- EMA 스킴의 수렴 보장 및 수렴 속도 분석
- 일반화 오차의 상한(generalization bound) 유도
- 노이즈율과 $\alpha$, $E_s$ 사이의 이론적 관계 규명

#### (2) 동적 하이퍼파라미터 조정
- 현재 $\alpha$, $E_s$는 수동 설정이 필요하므로, 노이즈율을 자동 추정하여 하이퍼파라미터를 적응적으로 조정하는 메커니즘 연구 필요

#### (3) 대규모 언어 모델(LLM)에의 적용
- RLHF 등에서 발생하는 인간 피드백 노이즈 처리에 SAT 원리를 적용할 수 있는지 검토
- 토큰 레벨의 EMA 목표 설정 가능성 탐색

#### (4) 클래스 불균형 및 롱테일 분포
- 현재 균일 노이즈 가정 → 실제 데이터의 클래스별 불균형 노이즈 처리 연구 필요

#### (5) 연합 학습(Federated Learning)과의 결합
- 클라이언트별 노이즈 이질성(heterogeneous noise) 환경에서의 SAT 적용 가능성

#### (6) 메모리 효율화
- 대규모 데이터셋에서 소프트 레이블 벡터 저장 비용 최적화 (양자화, 희소 표현 등)

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 논문들은 SAT의 영향을 받거나 유사한 방향의 후속 연구입니다. **단, 일부 연구는 제가 직접 해당 논문 전문을 확인하지 못했으므로, 알려진 내용을 바탕으로 기술합니다.**

### 5.1 노이즈 레이블 학습 분야

| 연구 | 핵심 방법 | SAT와의 관계 |
|---|---|---|
| **DivideMix** (Li et al., ICLR 2020) | GMM으로 노이즈/정상 샘플 분리 후 MixMatch 적용 | SAT보다 계산 비용 높지만 극단적 노이즈에서 경쟁적 성능 |
| **Noisy Student** (Xie et al., CVPR 2020) | 교사-학생 반복 자기 훈련 | 다중 훈련 반복 필요 (SAT는 단일 훈련 패스) |
| **ELR** (Liu et al., NeurIPS 2020) | Early-Learning Regularization, EMA 예측 활용 | SAT의 EMA와 유사하나 명시적 정규화 항 추가 |
| **C2D** (Zheltonozhskii et al., CVPR 2022) | 대조 학습 기반 노이즈 감지 | SAT와 직교적 방향, 결합 가능성 존재 |
| **SOP** (Liu et al., ICML 2022) | 과모수화된 보조 변수로 노이즈 모델링 | 이론적으로 SAT보다 엄밀한 수렴 보장 제공 |

### 5.2 적대적 강건성 분야

| 연구 | 핵심 방법 | SAT와의 관계 |
|---|---|---|
| **AWP** (Wu et al., NeurIPS 2020) | 가중치 perturbation 기반 적대적 훈련 | SAT + AWP 조합으로 추가 성능 향상 가능 |
| **HAT** (Rade & Moosavi-Dezfooli, ICLR 2022) | Helper-based adversarial training | 소프트 레이블 활용 측면에서 SAT와 연관 |

### 5.3 비교 분석 요약

```
                    계산 비용   노이즈 유형   이론적 보장   일반화 성능
SAT (2020)           낮음       다양         약함          강함
DivideMix (2020)     높음       레이블 위주   약함          매우 강함(극단적 노이즈)
ELR (2020)           낮음       레이블 위주   중간          강함
SOP (2022)           중간       레이블 위주   강함          강함
```

**SAT의 차별점:**
- **인스턴스 및 레이블 노이즈 모두 처리** 가능
- **추가 훈련 패스 없음** (단일 훈련 패스)
- **다양한 훈련 패러다임(자연/적대적 훈련)에 플러그인** 형태로 적용 가능

---

## 참고 자료

**주요 참고문헌 (논문 내 인용 기준):**

1. **Huang, L., Zhang, C., & Zhang, H. (2020).** Self-Adaptive Training: beyond Empirical Risk Minimization. *NeurIPS 2020*. arXiv:2002.10319

2. **Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2016).** Understanding deep learning requires rethinking generalization. arXiv:1611.03530

3. **Nakkiran, P., et al. (2019).** Deep double descent: Where bigger models and more data hurt. arXiv:1912.02292

4. **Li, M., Soltanolkotabi, M., & Oymak, S. (2019).** Gradient descent with early stopping is provably robust to label noise for overparameterized neural networks. arXiv:1903.11680

5. **Zhang, H., et al. (2019).** Theoretically principled trade-off between robustness and accuracy (TRADES). *ICML 2019*

6. **Xie, Q., et al. (2020).** Self-training with noisy student improves imagenet classification. *CVPR 2020*

7. **Belkin, M., et al. (2018).** Reconciling modern machine learning and the bias-variance trade-off. arXiv:1812.11118

8. **He, K., et al. (2016).** Deep residual learning for image recognition. *CVPR 2016*

9. **Hendrycks, D., & Dietterich, T. (2018).** Benchmarking neural network robustness to common corruptions and perturbations. *ICLR 2018* (CIFAR10-C 벤치마크)

10. **Wang, Y., et al. (2019).** Symmetric cross entropy for robust learning with noisy labels. *ICCV 2019*

**GitHub 코드 저장소:**
- https://github.com/LayneH/self-adaptive-training

> ⚠️ **정확도 관련 고지:** 2020년 이후 후속 연구들(DivideMix, ELR, SOP 등)의 세부 수치 비교는 해당 논문 원문을 직접 확인하지 못한 부분이 있어, 알려진 방향성 위주로 서술하였습니다. 정확한 수치 비교는 각 논문 원문을 참조하시기 바랍니다.
