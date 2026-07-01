# Central Moment Discrepancy (CMD) for Domain-Invariant Representation Learning

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **비지도 도메인 적응(Unsupervised Domain Adaptation)** 맥락에서, 소스 도메인과 타깃 도메인의 히든 레이어 활성화 분포 간의 불일치를 **고차 중심 모멘트(Central Moments)의 차이를 차수별(order-wise)로 명시적으로 최소화**함으로써 줄이는 새로운 정규화 방법인 **Central Moment Discrepancy (CMD)** 를 제안합니다.

기존의 MMD(Maximum Mean Discrepancy)가 커널 행렬 연산을 통해 가중 합산된 모멘트를 암묵적으로 매칭하거나, KL-divergence 기반 방법이 1차 모멘트(평균)만을 매칭하는 것과 달리, CMD는 **각 차수의 모멘트를 명시적으로 개별 매칭**합니다.

### 주요 기여 요약

| 기여 항목 | 내용 |
|---|---|
| 새로운 거리 함수 제안 | 확률분포 간 거리로서 CMD 정의 |
| 이론적 보증 | CMD가 콤팩트 구간 위의 확률분포 집합에서 **메트릭(metric)**임을 증명 |
| 수렴성 보증 | CMD 수렴 → 분포 수렴(convergence in distribution) 증명 |
| 계산 효율성 | 커널 행렬 불필요, 계산 복잡도 $\mathcal{O}(N(n+m))$ |
| 실험적 우수성 | Office 및 Amazon reviews 벤치마크에서 SOTA 달성 |
| 하이퍼파라미터 안정성 | $K \geq 3$ 범위에서 파라미터 변화에 둔감 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**비지도 도메인 적응** 문제를 다룹니다. 구체적으로:

- **소스 도메인** $D_S$: 레이블이 있는 데이터
- **타깃 도메인** $D_T$: 레이블이 없는 데이터
- **목표**: 타깃 리스크 $R_T(f) = \Pr_{(x,y) \sim D_T}(f(x) \neq y)$를 최소화하는 분류기 $f: \mathcal{X} \rightarrow \mathcal{Y}$ 학습

기존 접근법들의 한계:
- **KL-divergence(MKL)**: 1차 모멘트(평균)만 매칭
- **MMD**: 모멘트를 암묵적으로 가중 합산 매칭하며 커널 행렬 연산 필요 ( $\mathcal{O}(n^2+nm+m^2)$ ), 커널 파라미터 $\beta$ 튜닝 필요
- **DANN**: 새로운 분류기 훈련 필요, 추가 파라미터 및 연산 비용

### 2.2 제안 방법 및 수식

#### 기본 목적 함수

일반적인 도메인 적응 목적 함수:

$$\min_{\theta \in \Theta} \mathbf{E}(l(\theta, X_S, Y_S)) + \lambda \cdot d(A_H(\theta, X_S), A_H(\theta, X_T)) \tag{2}$$

여기서:
- $l(\theta, x, y) = -\sum_{i \in C} y_i \log(f_\theta(x)_i)$: 크로스 엔트로피 손실
- $d(\cdot, \cdot)$: 도메인 불일치 정규화 항
- $\lambda$: 정규화 가중치
- $A_H(\theta, X_S), A_H(\theta, X_T)$: 소스/타깃 히든 활성화

#### CMD 메트릭 정의 (Definition 1)

$X = (X_1, \ldots, X_n)$과 $Y = (Y_1, \ldots, Y_n)$이 콤팩트 구간 $[a,b]^N$ 위의 확률분포 $p$, $q$에서 독립동일분포로 추출된 경우:

$$\text{CMD}(p, q) = \frac{1}{|b-a|} \|\mathbb{E}(X) - \mathbb{E}(Y)\|_2 + \sum_{k=2}^{\infty} \frac{1}{|b-a|^k} \|c_k(X) - c_k(Y)\|_2 \tag{5}$$

여기서 $k$차 중심 모멘트 벡터:

$$c_k(X) = \left(\mathbb{E}\!\left(\prod_{i=1}^{N}(X_i - \mathbb{E}(X_i))^{r_i}\right)\right)_{\substack{r_1+\cdots+r_N=k \\ r_1,\ldots,r_n \geq 0}}$$

#### 경험적 CMD 정규화 항 (Definition 2)

실제 적용을 위한 경험적 추정치:

$$\text{CMD}_K(X, Y) = \frac{1}{|b-a|} \|\mathbf{E}(X) - \mathbf{E}(Y)\|_2 + \sum_{k=2}^{K} \frac{1}{|b-a|^k} \|C_k(X) - C_k(Y)\|_2 \tag{6}$$

여기서:
- $\mathbf{E}(X) = \frac{1}{|X|}\sum_{x \in X} x$: 경험적 기댓값 벡터
- $C_k(X) = \mathbf{E}((x - \mathbf{E}(X))^k)$: $k$차 표본 중심 모멘트 벡터
- $K$: 고려할 최대 모멘트 차수 (논문에서는 $K=5$ 권장)

#### 이론적 보증

**정리 1 (메트릭 성질)**:

$$\text{CMD}(p, q) = 0 \Rightarrow p = q$$

**정리 2 (분포 수렴)**:

$$\text{CMD}(p_n, p) \to 0 \Rightarrow p_n \xrightarrow{d} p$$

**명제 1 (상한 수렴)**:

$$\frac{1}{|b-a|^k}\|c_k(X) - c_k(Y)\|_2 \leq 2\sqrt{N}\left(\frac{1}{k+1}\left(\frac{k}{k+1}\right)^k + \frac{1}{2^{1+k}}\right) \tag{7}$$

이 상한은 $k \to \infty$에 따라 단조 감소하며 0으로 수렴하므로, 고차 모멘트 항은 전체 거리에 기여하는 정도가 줄어들어 **$K$의 선택이 크게 중요하지 않음**을 이론적으로 뒷받침합니다.

#### 비교 정리: MMD

$$\text{MMD}(X, Y)^2 = \mathbf{E}(K(X,X)) - 2\mathbf{E}(K(X,Y)) + \mathbf{E}(K(Y,Y)) \tag{3}$$

Gaussian 커널 $e^{-\beta\|x-y\|^2}$을 사용할 때 계산 복잡도: $\mathcal{O}(N(n^2+nm+m^2))$

CMD의 계산 복잡도: $\mathcal{O}(N(n+m))$ → **선형 복잡도**

### 2.3 모델 구조

#### Amazon Reviews 실험

```
입력 (5000차원 Bag-of-Words)
    ↓
Dense Layer (50 hidden nodes, sigmoid activation)
    ↓
Softmax Output
```

- 목적 함수: $\text{CMD}_K$ 정규화 포함한 식 (2)
- 최적화: Adagrad
- $\lambda = 1$, $K = 5$ (고정, 탐색 불필요)

#### Office 실험

```
VGG16 (사전학습 CNN)
    ↓
첫 번째 Dense Layer 출력 (특징 추출)
    ↓
Dense Layer (256 hidden nodes, sigmoid activation)
    ↓
Softmax Output (31 클래스)
```

- 최적화: Adadelta
- $\lambda = 1$, $K = 5$ (고정)

### 2.4 성능 향상

**Amazon Reviews 결과 (평균 정확도)**:

| 방법 | 평균 정확도 |
|------|-----------|
| Source Only | $.752 \pm .009$ |
| MMD | $.781 \pm .015$ |
| VFAE | $.784$ |
| DANN | $.763$ |
| **CMD** | **$.798 \pm .007$** |

→ 12개 태스크 중 9개에서 최고 성능, 나머지 3개에서 2위

**Office 결과 (평균 정확도)**:

| 방법 | 평균 정확도 |
|------|-----------|
| AdaBN (SOTA 이전) | $.767$ |
| VGG16 baseline | $.755$ |
| **CMD** | **$.799$** |

→ 이전 SOTA 대비 평균 **3.2% 이상** 향상

### 2.5 한계점

1. **콤팩트 구간 가정**: CMD의 이론적 보증은 $[a,b]^N$과 같은 콤팩트 구간을 전제로 함. ReLU 등 무한 범위의 활성화 함수 사용 시 클리핑/정규화가 필요하며 이론적 엄밀성이 약화됨.

2. **주변 분포(marginal distribution) 매칭**: 계산 효율성을 위해 결합 분포가 아닌 주변 분포의 중심 모멘트만 매칭. 변수 간 상관관계(의존성)가 있는 경우 결합 분포의 완전한 매칭 보증 불가.

3. **비교 대상의 제한**: 논문 발표 시점(2017)을 기준으로 Gaussian 커널 기반 이차 시간(quadratic-time) MMD와만 비교. 선형 시간 MMD, 다른 커널 등과의 비교가 부족.

4. **생성 모델 미적용**: 판별 모델에만 적용됨. 생성 모델(GAN 등)로의 확장 가능성을 제안하나 실증하지 않음.

5. **벤치마크 제한**: Office, Amazon reviews 두 벤치마크에만 국한된 검증.

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 이론적 근거

**정리 2**는 CMD 최소화가 단순히 특정 통계량을 맞추는 것이 아니라 **분포 수렴(convergence in distribution)** 을 보장함을 증명합니다:

$$\text{CMD}(p_n, p) \to 0 \Rightarrow p_n \xrightarrow{d} p$$

이는 누적분포함수(CDF)의 수렴을 의미하며, 소스-타깃 도메인의 잠재 표현 분포가 실질적으로 동일하게 수렴함을 보장합니다. 이는 Ben-David et al. (2010)의 이론적 프레임워크와 연결됩니다:

$$R_T(f) \leq R_S(f) + \text{domain discrepancy} + \text{complexity term}$$

도메인 불일치를 줄임으로써 타깃 리스크의 상한을 낮출 수 있으며, 이것이 곧 **일반화 성능 향상**의 근거입니다.

### 3.2 고차 모멘트 매칭의 중요성

| 모멘트 차수 | 포착하는 분포 특성 |
|---|---|
| 1차 (평균) | 위치(location) |
| 2차 (분산) | 퍼짐 정도(spread) |
| 3차 (왜도, skewness) | 분포의 비대칭성 |
| 4차 (첨도, kurtosis) | 꼬리의 두께 |
| $k \geq 5$차 | 더 세밀한 분포 형태 |

MKL이 1차만, MMD가 가중합으로 암묵적으로 처리하는 것과 달리, CMD는 **각 차수별 명시적 매칭**을 통해 분포 형태의 더 세밀한 정렬을 달성합니다.

### 3.3 파라미터 안정성과 일반화

파라미터 민감도 분석에서 $K \in \{3, 4, 5, 6, 7\}$ 범위에서 정확도 비율 차이가 **0.5% 미만**임이 확인되었습니다. 또한 히든 노드 수가 256에서 1536으로 증가해도 CMD의 성능 향상 비율(4~6%)이 유지되어, **스케일에 robust한 일반화 능력**을 보여줍니다.

이에 반해 MMD는 히든 노드 수 증가에 따라 성능 향상 비율이 감소하는 경향이 관찰되었습니다.

### 3.4 계산 효율성과 실용적 일반화

계산 복잡도가 $\mathcal{O}(N(n+m))$으로 **데이터 수에 대해 선형**이므로, 대규모 데이터셋에서도 적용 가능합니다. 이는 다양한 규모의 실제 응용에서의 일반화 가능성을 높입니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 연구 영향

#### (1) 도메인 적응 연구의 방향 전환
CMD는 단순히 새로운 거리 함수를 제안하는 것을 넘어, **모멘트의 차수별 명시적 매칭**이라는 새로운 패러다임을 제시했습니다. 이후 연구들은 단순 평균이나 분산만이 아닌 고차 통계량을 활용하는 방향으로 발전했습니다.

#### (2) 이론-실험 통합 연구의 선례
CMD는 실험적 성능 향상뿐만 아니라 **메트릭 성질 증명 및 수렴성 증명**을 함께 제시하였습니다. 이는 단순 휴리스틱이 아닌 이론적 기반을 갖춘 정규화 방법의 중요성을 부각시켰습니다.

#### (3) 계산 효율적 분포 매칭 연구 촉진
커널 행렬 없이 선형 시간 복잡도로 분포 매칭을 달성한 것은, 이후 **대규모 도메인 적응** 및 **연합 학습(Federated Learning)** 등 계산 자원이 제한된 환경에서의 연구에 영감을 주었습니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래 연구들은 CMD의 아이디어를 발전시키거나 관련 문제를 다루는 대표적 연구들입니다. 단, 이 분석은 제가 학습한 지식 범위 내의 내용이며 직접 논문을 검색·열람한 결과가 아님을 명시합니다.

#### (A) Optimal Transport 기반 도메인 적응

**Damodaran et al. (2018), "DeepJDOT"** 및 이후 관련 연구들은 Optimal Transport(OT) 이론, 특히 Wasserstein 거리를 활용합니다.

Wasserstein-1 거리:
$$W_1(p, q) = \inf_{\gamma \in \Pi(p,q)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]$$

**CMD 대비 비교**:
- OT는 분포 간의 기하학적 구조를 더 풍부하게 포착
- CMD는 계산이 훨씬 단순하고 효율적
- OT의 계산 복잡도는 일반적으로 더 높으며 근사 알고리즘 필요

#### (B) 적대적 학습 기반 발전: DANN → 이후 연구

Ganin et al. (2016)의 DANN에 비해 CMD가 Amazon reviews에서 성능이 우수했지만, 이후 **Conditional DANN (CDANN)**, **MDD (Margin Disparity Discrepancy)** 등이 제안되었습니다.

**Zhang et al. (2019), "Bridging Theory and Algorithm for Domain Adaptation"** 에서 제안된 MDD:

$$\text{MDD}(p, q) = \sup_{h, h' \in \mathcal{H}} \left(\mathbb{E}_p[\sigma(h'(x) \neq h(x))] - \mathbb{E}_q[\sigma(h'(x) \neq h(x))]\right)$$

이는 이론적으로 더 강한 보장을 제공하나, CMD보다 복잡한 훈련 과정이 필요합니다.

#### (C) 정보 이론 기반: MIM (Mutual Information Maximization)

**도메인 불변 표현 학습**에서 상호 정보량(Mutual Information)을 활용하는 접근:

$$\max_\theta I(Z; Y) - \beta \cdot I(Z; D)$$

여기서 $Z$는 잠재 표현, $Y$는 레이블, $D$는 도메인 인덱스입니다.

이는 CMD의 순수 분포 매칭과 달리 **레이블 정보를 명시적으로 고려**한다는 차이점이 있습니다.

#### (D) Transformer 기반 도메인 적응

2020년 이후 Vision Transformer(ViT) 및 BERT 계열 모델의 등장으로, 사전학습된 대형 모델에 CMD를 파인튜닝 방식으로 적용하는 연구가 가능해졌습니다. 그러나 Transformer의 attention mechanism이 자체적으로 도메인 불변 특징을 일부 학습할 수 있다는 점에서, CMD와의 시너지 및 상호 작용에 대한 연구가 필요합니다.

#### (E) 계산 효율성 측면의 비교

| 방법 | 복잡도 | 파라미터 튜닝 | 이론 보증 |
|------|--------|-------------|---------|
| MMD (Gaussian) | $\mathcal{O}(N(n^2+nm+m^2))$ | $\beta$ 튜닝 필요 | 있음 |
| DANN | $\mathcal{O}(Nn)$ | 추가 분류기 필요 | 있음 |
| **CMD** | $\mathcal{O}(N(n+m))$ | 불필요 ($K=5$ 고정) | 있음 |
| OT (Sinkhorn) | $\mathcal{O}(n^2/\epsilon)$ | $\epsilon$ 튜닝 필요 | 있음 |
| MDD | $\mathcal{O}(Nn)$ | 추가 훈련 필요 | 강함 |

### 4.3 향후 연구 시 고려할 점

#### (1) 비콤팩트 분포로의 이론 확장

CMD의 이론적 보증은 콤팩트 구간 $[a,b]^N$에 한정됩니다. ReLU와 같이 무한 범위를 갖는 활성화 함수를 사용할 때 이론적 엄밀성이 약해집니다. 향후 연구에서는:

- **가우시안 측도(Gaussian measure)** 위에서의 CMD 확장
- **무한 차원** 히든 공간에서의 이론 개발
- Wasserstein 거리와의 이론적 관계 규명

가 중요한 방향입니다.

#### (2) 결합 분포 매칭

현재 CMD 정규화 항(Definition 2)은 주변 분포(marginal distributions)의 중심 모멘트만 계산합니다. 변수들 간 의존성이 높은 경우 결합 분포 매칭이 필요하며, 이를 위한:

$$\text{CMD}_{\text{joint}}(p, q) = \sum_{k=1}^{K} \frac{1}{|b-a|^k} \|c_k^{\text{joint}}(X) - c_k^{\text{joint}}(Y)\|_2$$

형태의 결합 중심 모멘트 활용 방법 연구가 필요합니다. 단, 이는 차원의 저주(curse of dimensionality) 문제를 야기합니다.

#### (3) 생성 모델과의 결합

논문의 결론에서도 언급되었듯이, CMD를 **생성적 적대 신경망(GAN)** 혹은 **변분 오토인코더(VAE)** 에 적용하는 것은 자연스러운 확장 방향입니다:

$$\mathcal{L} = \mathcal{L}_{\text{reconstruction}} + \lambda_1 \cdot \mathcal{L}_{\text{adversarial}} + \lambda_2 \cdot \text{CMD}_K(p_\phi(z|x_S), p_\phi(z|x_T))$$

이는 표현 학습 과정에서 도메인 불변성을 강화하는 데 활용될 수 있습니다.

#### (4) 다중 소스 도메인 적응

현재 CMD는 단일 소스 → 단일 타깃의 이진 설정을 가정합니다. 여러 소스 도메인이 있는 경우:

$$d_{\text{multi}} = \sum_{i < j} \text{CMD}_K(p_i, p_j) + \sum_i \text{CMD}_K(p_i, q)$$

형태의 확장이 가능하나, 소스 도메인 간 균형 문제가 새로운 연구 과제입니다.

#### (5) Few-shot 및 Zero-shot 설정으로의 확장

현대 딥러닝에서는 매우 적은 타깃 도메인 샘플(few-shot) 혹은 전혀 없는(zero-shot) 상황에서의 적응이 중요합니다. 이 경우 경험적 CMD 추정의 분산이 커지므로:

- **정규화된 추정량(regularized estimator)** 개발
- **부트스트랩(bootstrap) 기반** 신뢰 구간 도입
- **메타 러닝(meta-learning)** 프레임워크와의 통합

이 필요합니다.

#### (6) 대규모 모델에서의 적용 가능성

GPT, BERT, ViT 등 대형 사전학습 모델에 CMD를 적용할 때:
- 어느 레이어의 활성화에 CMD를 적용할 것인가 (레이어 선택 문제)
- 배치 크기와 $K$의 관계 최적화
- 분산 학습 환경에서의 효율적 CMD 계산

에 대한 연구가 필요합니다.

---

## 참고 자료

1. **Zellinger, W., Grubinger, T., Lughofer, E., Natschläger, T., & Saminger-Platz, S. (2017).** "Central Moment Discrepancy (CMD) for Domain-Invariant Representation Learning." *ICLR 2017*. arXiv:1702.08811v3

2. **Ganin, Y., et al. (2016).** "Domain-adversarial training of neural networks." *Journal of Machine Learning Research*, 17(1):1–35.

3. **Gretton, A., et al. (2012).** "A kernel two-sample test." *Journal of Machine Learning Research*, 13:723–773.

4. **Louizos, C., et al. (2016).** "The variational fair auto encoder." *ICLR 2016*.

5. **Ben-David, S., et al. (2010).** "A theory of learning from different domains." *Machine learning*, 79(1-2):151–175.

6. **Long, M., et al. (2015).** "Learning transferable features with deep adaptation networks." *ICML 2015*.

7. **Sun, B., & Saenko, K. (2016).** "Deep CORAL: Correlation alignment for deep domain adaptation." *arXiv:1607.01719*.

8. **Li, Y., et al. (2016).** "Revisiting batch normalization for practical domain adaptation." *arXiv:1603.04779*.

9. **Egozcue, M., et al. (2012).** "The smallest upper bound for the pth absolute central moment of a class of random variables." *The Mathematical Scientist*.

10. **Billingsley, P. (2008).** *Probability and Measure*. John Wiley & Sons.

11. **Billingsley, P. (2013).** *Convergence of Probability Measures*. John Wiley & Sons.

> **⚠️ 주의사항**: 2020년 이후 최신 연구 비교 분석 부분(DeepJDOT, MDD, Conditional DANN 등)은 본 논문 원문에 포함된 내용이 아니며, 제 학습 데이터에 기반한 일반적 지식입니다. 해당 내용의 세부 수치나 결과는 원논문을 직접 확인하시기를 권장합니다.
