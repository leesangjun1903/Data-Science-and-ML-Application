# A Second-Order Approach to Learning with Instance-Dependent Label Noise

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 연구들이 **일차 통계량(first-order statistics)** (모델 예측의 기댓값)에만 의존하여 노이즈를 처리했다면, 이 논문은 **인스턴스 의존적 레이블 노이즈(Instance-Dependent Label Noise, IDN)** 를 다루기 위해 **이차 통계량(second-order statistics)**, 즉 공분산(covariance) 항을 활용하는 새로운 접근법이 필요함을 주장한다.

### 주요 기여

1. **이차 통계량의 중요성 증명**: IDN 환경에서 노이즈 전이 행렬 $T(X)$와 Bayes 최적 레이블 간의 공분산 항이 학습에 미치는 영향을 이론적으로 분석
2. **CAL(Covariance-Assisted Learning) 손실 함수 제안**: IDN의 기대 위험을 클래스 의존적 노이즈(class-dependent noise) 문제로 변환하는 새로운 손실 함수 도출
3. **효율적인 이차 통계량 추정 절차 제공**: 실제 클린 레이블이나 노이즈 비율에 대한 사전 지식 없이 공분산 항을 추정
4. **이론적 성능 보장 및 실험 검증**: CIFAR10, CIFAR100, Clothing1M에서의 실험으로 방법 검증

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**IDN(Instance-Dependent Label Noise)** 환경에서의 학습 문제:

- 레이블 노이즈가 클래스뿐만 아니라 **개별 인스턴스의 특성(난이도)**에 따라 달라짐
- 인스턴스별로 다른 노이즈 비율이 존재 → **불균형(imbalance)** 유발
- 기존의 클래스 의존적 방법(class-dependent methods)을 직접 적용할 경우, 노이즈가 높은 예시들이 **비균일하게 다운웨이팅(down-weighting)** 되어 성능 저하 발생

노이즈 전이 행렬은 $T(X)$가 인스턴스마다 다르며 추정해야 할 파라미터 수가 $O(NK^2)$에 달해 직접 추정이 불가능함.

### 2.2 제안 방법 및 수식

#### 기본 설정

- 노이즈 데이터셋: $\tilde{D} := \{(x_n, \tilde{y}\_n)\}_{n \in [N]}$
- 인스턴스 의존 노이즈 전이 행렬:

$$T_{i,j}(X) = \mathbb{P}(\tilde{Y} = j \mid Y^* = i, X)$$

- 이진 분류에서의 오류율:

$$e_+(X) := \mathbb{P}(\tilde{Y} = -1 \mid Y^* = +1, X), \quad e_-(X) := \mathbb{P}(\tilde{Y} = +1 \mid Y^* = -1, X)$$

#### Peer Loss (기존 방법)

$$\ell_{\text{PL}}(f(x_n), \tilde{y}_n) := \ell(f(x_n), \tilde{y}_n) - \ell(f(x_{n_1}), \tilde{y}_{n_2})$$

#### Theorem 1: Peer Loss의 IDN 환경 성능 한계

```math
\mathbb{E}[\mathbb{1}(\tilde{f}^*_{\text{peer}}(X), Y^*)] \leq \frac{2(\epsilon_+ + \epsilon_-)}{1 - e_+ - e_-} + 2|p^* - 0.5|
```

여기서 $\epsilon_+, \epsilon_-$는 노이즈 비율 분산의 상한값. IDN이 존재할 경우($\epsilon_+ + \epsilon_- > 0$) 오차가 증가함을 보임.

#### Theorem 2: IDN 디커플링 (핵심 이론)

이진 분류에서 IDN을 가진 Peer Loss의 기대값:

$$\mathbb{E}_{\tilde{D}}[\mathbb{1}_{\text{PL}}(f(X), \tilde{Y})] = (1 - e_+ - e_-)\mathbb{E}_{D^*}[\mathbb{1}_{\text{PL}}(f(X), Y^*)]$$

```math
+ \text{Cov}_{D^*}(Z_1(X), \mathbb{1}(f(X), Y^*))
```

$$+ \text{Cov}_{D^*}(Z_2(X), \mathbb{1}(f(X), -1))$$

여기서:

$$Z_1(X) := 1 - e_+(X) - e_-(X), \quad Z_2(X) := e_+(X) - e_-(X)$$

- **첫 번째 항**: 클래스 의존적 노이즈와 동일한 평균 효과
- **두 번째, 세 번째 항**: IDN의 이질성으로 인한 추가적인 공분산 효과

#### Corollary 1: 다중 클래스 확장

전이 행렬이 $e_j = T_{i,j} = T_{k,j}, \forall i \neq j \neq k$를 만족할 때:

$$\mathbb{E}_{\tilde{D}}[\ell_{\text{PL}}(f(X), \tilde{Y})] = \left(1 - \sum_{i \in [K]} e_i\right)\mathbb{E}_{D^*}[\ell_{\text{PL}}(f(X), Y^*)]$$

```math
+ \sum_{j \in [K]} \mathbb{E}_{D_{Y^*}}\left[\text{Cov}_{D^*|Y^*}(T_{Y^*,j}(X), \ell(f(X), j))\right]
```

#### CAL 손실 함수 (제안 방법)

$$\ell_{\text{CAL}}(f(x_n), \tilde{y}_n) = \ell_{\text{PL}}(f(x_n), \tilde{y}_n)$$

```math
- \sum_{j \in [K]} \mathbb{E}_{D_{Y^*}}\left[\text{Cov}_{D^*|Y^*}(T_{Y^*,j}(X), \ell(f(X), j))\right]
```

#### Theorem 3: CAL의 최적성 보장

```math
\tilde{f}^*_{\text{CAL}} \in \arg\min_f \mathbb{E}_{D^*}[\mathbb{1}(f(X), Y^*)]
```

CAL 손실을 최소화하는 분류기가 Bayes 최적 분류기와 동일한 최소값을 가짐.

#### 공분산 추정 (SGD에서의 구현)

$$\hat{T}_{i,j}(x_n) = \mathbb{1}\{\hat{y}_n = i, \tilde{y}_n = j\} $$

샘플 단위 CAL 손실:

$$\ell_{\text{CAL}}(f(x_n), \tilde{y}_n) = \ell_{\text{PL}}(f(x_n), \tilde{y}_n) - \sum_{i,j \in [K]} \mathbb{1}\{y^*_n = i\}\left[(\hat{T}_{i,j}(x_n) - \hat{T}_{i,j}) \cdot \ell(f(x_n), j)\right]$$

#### Theorem 4: 불완전한 공분산 추정에서의 성능 보장

$\hat{D}^\tau$에서 올바른 샘플 비율 $\tau \in [0,1]$이고 $p^* = 0.5$일 때:

```math
\mathbb{E}[\mathbb{1}(\tilde{f}^*_{\text{CAL}-\tau}(X), Y^*)] \leq \frac{4(1-\tau)(\epsilon_+ + \epsilon_-)}{1 - e_+ - e_-}
```

$\tau \geq 0.5$이면 공분산 항 사용이 항상 도움이 됨을 보증.

### 2.3 모델 구조

**2단계 파이프라인**:

1. **$\hat{D}$ 구성 단계** (Algorithm 1):
   - CORES² [Cheng et al., 2021]의 샘플 시브(sample sieve) 방법으로 65 에폭 학습
   - 조정된 손실 $\ell_{\text{CORES}^2}(f(x_n), \tilde{y}\_n) - \alpha_{n,T}$와 임계값 $L_{\min}, L_{\max}$ 비교
   - 작으면 $\hat{y}\_n = \tilde{y}\_n$ (clean), 크면 $\hat{y}\_n = \arg\max_{y} f_{x_n}[y]$ (예측값), 중간이면 제거

2. **CAL 학습 단계**:
   - $\hat{D}$를 이용해 공분산 항 추정
   - CAL 손실로 100 에폭 재학습
   - 백본: ResNet34(CIFAR), ResNet50(Clothing1M)

### 2.4 성능 향상

**CIFAR10 테스트 정확도 (%):**

| 방법 | $\eta=0.2$ | $\eta=0.4$ | $\eta=0.6$ |
|------|-----------|-----------|-----------|
| CE (기준) | 85.45 | 76.23 | 59.75 |
| Peer Loss | 89.12 | 83.26 | 74.53 |
| CORES² | 91.14 | 83.67 | 77.68 |
| **CAL** | **92.01** | **84.96** | **79.82** |

**CIFAR100 테스트 정확도 (%):**

| 방법 | $\eta=0.2$ | $\eta=0.4$ | $\eta=0.6$ |
|------|-----------|-----------|-----------|
| CE (기준) | 57.79 | 41.15 | 25.68 |
| CORES² | 66.47 | 58.99 | 38.55 |
| **CAL** | **69.11** | **63.17** | **43.58** |

**Clothing1M (실제 인간 노이즈):** CAL **74.17%** vs CORES² 73.24%

### 2.5 한계

1. **$\hat{D}$ 구성 품질 의존성**: 공분산 추정 품질이 $\hat{D}$의 정확도($\tau$)에 크게 의존
2. **다중 클래스 전이 행렬 가정**: Corollary 1은 $T_{i,j} = T_{k,j}, \forall i \neq j \neq k$ 조건 필요
3. **2단계 학습의 비효율성**: $\hat{D}$ 구성과 CAL 학습이 분리되어 총 학습 비용 증가
4. **하이퍼파라미터 민감성**: $L_{\min}, L_{\max}, \beta$ 조정이 필요하며 클린 검증 셋이 있을 때 최적 성능
5. ** $p^* = 0.5$ 가정**: Theorem 4의 보장이 균형 잡힌 Bayes 최적 분포를 가정

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 공분산 항의 "소프트 보정" 효과

기존의 하드 레이블 보정(hard label correction)과 달리, CAL은 공분산 항을 통한 **소프트 보정(soft correction)** 을 수행한다. 이는:

```math
\ell_{\text{CAL}} = \ell_{\text{PL}} - \underbrace{\sum_{j \in [K]} \mathbb{E}_{D_{Y^*}}\left[\text{Cov}_{D^*|Y^*}(T_{Y^*,j}(X), \ell(f(X), j))\right]}_{\text{소프트 공분산 보정 항}}
```

원본 노이즈 레이블과 추정된 Bayes 최적 레이블의 정보를 **동시에 보존**하여 일반화 능력을 향상시킨다. Han et al.(2019)의 연구[13]에서도 두 정보를 모두 유지하는 것이 유익함을 확인.

### 3.2 다운웨이팅 효과의 보상

IDN은 자동으로 노이즈가 높은 예시를 다운웨이팅하는데, 이로 인해 **학습 분포의 편향(bias)** 이 발생한다. CAL은 공분산 항으로 이 불균형을 보상하여:

$$\mathbb{E}_{\tilde{D}}[\mathbb{1}_{\text{PL}}(f(X), \tilde{Y})] - \text{Cov}(\cdot) - \text{Cov}(\cdot) = (1-e_+-e_-)\mathbb{E}_{D^*}[\mathbb{1}_{\text{PL}}(f(X), Y^*)]$$

를 달성, **클래스 의존적 노이즈 문제로 변환**하여 기존 방법들을 적용 가능하게 함.

### 3.3 최악의 경우 보장(Worst-case Guarantee)

Theorem 4에 의해 $\tau \geq 0.5$인 경우 항상 Peer Loss보다 나은 성능이 보장된다. 이는 불완전한 $\hat{D}$에서도 **안정적인 일반화**를 제공함을 의미.

### 3.4 실험적 증거

Ablation study(Table 3)에서:
- 공분산 항만 사용: $\eta=0.6$에서 73.55% (CE 64.65% 대비 +8.9%)
- Peer 항만 사용: 78.74%
- **공분산 + Peer 결합: 81.54%** (상호 보완적 효과)

이는 이차 통계량이 일차 통계량과 독립적으로 일반화에 기여함을 보여준다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

1. **이차 통계량 패러다임의 확장**: 레이블 노이즈 연구에서 기댓값(1차)을 넘어 공분산(2차), 나아가 고차 통계량(higher-order statistics) 활용 방향을 제시

2. **IDN → CDN 변환 프레임워크**: 복잡한 IDN 문제를 더 잘 연구된 CDN(class-dependent noise) 문제로 변환하는 **일반적 프레임워크**로서 다양한 손실 함수에 적용 가능

3. **샘플 선택과 손실 함수의 결합**: $\hat{D}$ 구성을 위한 샘플 선택과 보정된 손실 함수를 결합하는 2단계 접근법이 새로운 연구 패턴으로 확립

4. **Bayes 최적 레이블의 역할**: 클린 레이블 대신 Bayes 최적 레이블 $Y^*$를 목표로 설정하는 관점이 향후 연구에 영향

### 4.2 앞으로 연구 시 고려할 점

1. **$\hat{D}$ 구성의 개선**: 현재 CORES²에 의존하는 $\hat{D}$ 구성을 대형 언어모델이나 자기지도학습(self-supervised learning)으로 대체하여 정확도 향상

2. **종단간(end-to-end) 학습**: 현재의 2단계 학습을 단일 통합 프레임워크로 발전시켜 오류 누적 및 계산 비용 감소

3. **고차 통계량 활용**: 공분산(2차)을 넘어 3차, 4차 통계량의 활용 가능성 탐색

4. **전이 행렬 가정 완화**: Corollary 1의 $T_{i,j} = T_{k,j}$ 가정을 완화하여 더 일반적인 IDN 설정으로 확장

5. **반지도 학습과의 결합**: DivideMix [Li et al., 2020]와 같은 반지도 학습 프레임워크에 CAL 통합

6. **Large Language Model 시대의 적용**: GPT, BERT 등 대규모 모델의 파인튜닝 과정에서 발생하는 노이지 레이블 문제에 CAL 적용 가능성

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 비교는 본 논문에서 직접 인용된 논문들 및 본 논문(2020년 12월 제출)과 연관된 연구들을 중심으로 정리합니다. **본 논문 출판 이후의 후속 연구에 대한 정보는 학습 데이터 기준으로 일부 불확실할 수 있으므로, 논문에서 직접 확인된 내용을 우선적으로 기술합니다.**

### 5.1 논문에서 직접 비교된 2020년 이후 관련 연구

| 연구 | 핵심 방법 | IDN 처리 | 특이사항 |
|------|---------|---------|---------|
| **CORES²** [Cheng et al., ICLR 2021] | 신뢰도 정규화기 + 샘플 시브 | 이론적 보장, 1차 통계량 | CAL의 기반이 되는 연구 |
| **PTD-R-V** [Xia et al., NeurIPS 2020] | 파트 의존적 노이즈 모델링 | 추가 가정 필요 | 데이터 증강 없이 비교 시 CAL이 우수 |
| **Peer Loss** [Liu & Guo, ICML 2020] | 노이즈 비율 불필요 | 클래스 의존 노이즈에 강건 | IDN에서는 성능 저하 |
| **Dual T** [Yao et al., NeurIPS 2020] | 이중 전이 행렬 | 추정 오류 감소 | 직접 비교 없음 |
| **DivideMix** [Li et al., ICLR 2020] | 반지도 학습 | 추가 기법 사용 | 비교 제외(추가 증강 사용) |

### 5.2 방법론적 비교

```
CAL vs 기존 방법:

1차 통계량 기반:    E[f(X)]              → Peer Loss, CORES²
이차 통계량 기반:   Cov(T(X), f(X))      → CAL (본 논문)
전이 행렬 추정:     T(X) 직접 추정       → Forward T, PTD
샘플 선택:          클린 샘플 선별        → Co-teaching, DivideMix
```

### 5.3 본 논문의 차별성

본 논문은 기존 연구들이 **1차 통계량**(기댓값)에 집중한 것과 달리, IDN 환경에서 노이즈 비율의 이질성을 포착하는 **공분산(2차 통계량)** 을 명시적으로 모델에 통합한 최초의 체계적 연구임.

---

## 참고 자료

- **본 논문**: Zhu, Z., Liu, T., & Liu, Y. (2021). "A Second-Order Approach to Learning with Instance-Dependent Label Noise." *arXiv:2012.11854v2*. Available at: https://arxiv.org/abs/2012.11854

- **Peer Loss** [논문 내 참조 24]: Liu, Y., & Guo, H. (2020). "Peer Loss Functions: Learning from Noisy Labels without Knowing Noise Rates." *ICML 2020*.

- **CORES²** [논문 내 참조 5]: Cheng, H., Zhu, Z., Li, X., Gong, Y., Sun, X., & Liu, Y. (2021). "Learning with Instance-Dependent Label Noise: A Sample Sieve Approach." *ICLR 2021*.

- **PTD** [논문 내 참조 40]: Xia, X., Liu, T., Han, B., et al. (2020). "Part-Dependent Label Noise: Towards Instance-Dependent Label Noise." *NeurIPS 2020*.

- **DivideMix** [논문 내 참조 19]: Li, J., Socher, R., & Hoi, S. C. H. (2020). "DivideMix: Learning with Noisy Labels as Semi-Supervised Learning." *ICLR 2020*.

- **Forward T** [논문 내 참조 29]: Patrini, G., et al. (2017). "Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach." *CVPR 2017*.

- **Co-teaching** [논문 내 참조 12]: Han, B., et al. (2018). "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels." *NeurIPS 2018*.

- **GitHub 구현**: https://github.com/UCSC-REAL/CAL
