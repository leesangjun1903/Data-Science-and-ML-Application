# Peer Loss Functions: Learning from Noisy Labels without Knowing Noise Rates 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 **노이즈 비율(noise rates)을 사전에 알거나 추정하지 않고도, 노이즈가 있는 레이블로부터 효과적으로 학습할 수 있는 새로운 손실 함수 계열(peer loss functions)을 제안한다**는 것입니다.

기존의 대부분의 방법론은 노이즈 비율 $e_{+1} = P(\tilde{Y}=-1|Y=+1)$, $e_{-1} = P(\tilde{Y}=+1|Y=-1)$을 알고 있거나 별도로 추정해야 했습니다. 이 논문은 이 제약을 근본적으로 제거합니다.

### 주요 기여

1. **새로운 손실 함수 계열 제안**: 비대칭(asymmetric) 레이블 노이즈에 대해 형식적인 이론적 보장을 제공하며, 노이즈 비율의 사전 지식이나 추정이 불필요한 Peer Loss Functions를 제안
2. **이론적 최적성 보장**: ERM with peer loss가 깨끗한 데이터로 학습한 것과 동일하거나 근사-최적의 분류기를 복원함을 증명 (Theorem 2, 3, 4)
3. **리스크 보장 및 일반화 한계**: Theorem 5, 7을 통해 리스크 보장과 일반화 한계를 제공
4. **실험적 검증**: 10개의 UCI 벤치마크 및 CIFAR-10에서 경쟁력 있는 성능 입증
5. **오픈소스 구현**: https://github.com/gohsyi/PeerLoss

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 설정**: 이진 분류 문제에서 $(X, Y) \sim \mathcal{D}$이지만 학습자는 깨끗한 레이블 $Y$ 대신 노이즈 레이블 $\tilde{Y}$만 관찰합니다.

노이즈 모델(class-conditional noise, CCN):

$$e_{+1} := P(\tilde{Y}=-1|Y=+1), \quad e_{-1} := P(\tilde{Y}=+1|Y=-1)$$

조건: $0 \le e_{+1} + e_{-1} < 1$ (노이즈 레이블이 진짜 레이블과 양의 상관관계)

**기존 방법의 한계**: Natarajan et al. (2013)의 비편향 surrogate loss:

$$\tilde{\ell}(t, y) := \frac{(1-e_{-y}) \cdot \ell(t,y) - e_y \cdot \ell(t,-y)}{1 - e_{-1} - e_{+1}}$$

이 방법은 $e_{+1}, e_{-1}$을 반드시 알아야 합니다. 이것이 실무에서 병목이 됩니다.

---

### 2.2 제안하는 방법: Peer Loss

**핵심 아이디어**: Peer Prediction 문헌의 Correlated Agreement(CA) 메커니즘에서 영감을 받아, 각 샘플의 손실을 평가할 때 독립적으로 샘플링된 "peer 샘플"의 손실을 빼는 방식으로 노이즈 효과를 상쇄합니다.

#### Peer Loss의 정의 (Equation 5)

각 샘플 $(x_n, \tilde{y}\_n)$에 대해 독립적으로 샘플링된 peer 샘플 $(x_{n_1}, \tilde{y}_{n_2})$ (여기서 $n_1 \neq n_2$)를 이용하여:

$$\ell_{\text{peer}}(f(x_n), \tilde{y}_n) = \ell(f(x_n), \tilde{y}_n) - \ell(f(x_{n_1}), \tilde{y}_{n_2})$$

**왜 이것이 작동하는가**: 첫 번째 항은 현재 샘플에 대한 손실이고, 두 번째 "peer 항"은 독립적인 두 샘플로부터 만들어집니다. 이 peer 항이 노이즈 정보를 암묵적으로 인코딩하여, 노이즈 레이블에 과도하게 동의하는 것을 방지합니다.

#### ERM with Peer Loss

$$\hat{f}^*_{\ell_{\text{peer}}} = \arg\min_{f \in \mathcal{F}} \frac{1}{N} \sum_{n=1}^{N} \ell_{\text{peer}}(f(x_n), \tilde{y}_n)$$

#### 핵심 성질: 노이즈 불변성 (Lemma 2)

$$\mathbb{E}_{\tilde{\mathcal{D}}}[\ell_{\text{peer}}(f(X), \tilde{Y})] = (1 - e_{-1} - e_{+1}) \cdot \mathbb{E}_{\mathcal{D}}[\ell_{\text{peer}}(f(X), Y)]$$

이는 **노이즈 분포 위에서의 peer loss 기댓값이 깨끗한 분포 위에서의 peer loss 기댓값의 양의 스칼라 배수**임을 의미합니다. 따라서 노이즈 데이터에서 peer loss를 최소화하는 것이 깨끗한 데이터에서 최소화하는 것과 동치입니다.

**증명 스케치**: 

$$\mathbb{E}[\ell_{\text{peer}}(f(X), \tilde{Y})] = \mathbb{E}[\ell(f(X), \tilde{Y})] - \mathbb{E}[\ell(f(X_{n_1}), \tilde{Y}_{n_2})]$$

두 항을 각각 전개하면, $n_1$과 $n_2$의 독립성으로 인해 노이즈 관련 항이 상쇄되어:

$$= (1 - e_{-1} - e_{+1}) \cdot \mathbb{E}[\ell_{\text{peer}}(f(X), Y)]$$

---

### 2.3 이론적 보장

#### Theorem 2: 동일 클래스 분포 시 최적성

$p = P(Y=+1) = 0.5$이면:

$$\tilde{f}^*_{\mathbb{1}_{\text{peer}}} \in \arg\min_{f \in \mathcal{F}} R_{\mathcal{D}}(f)$$

즉, peer loss의 최소화자가 0-1 손실의 최소화자와 일치합니다.

#### Theorem 3: 불균형 분포 시 근사 최적성

$p \neq 0.5$이고 $\delta_p = P(Y=+1) - P(Y=-1)$이면:

$$|R_{\mathcal{D}}(\tilde{f}^*_{\mathbb{1}_{\text{peer}}}) - \min_{f \in \mathcal{F}} R_{\mathcal{D}}(f)| \leq |\delta_p|$$

#### $\alpha$-weighted Peer Loss

클래스 불균형 문제를 해결하기 위해 peer 항에 가중치 $\alpha$를 부여:

$$\ell_{\alpha\text{-peer}}(f(x_n), \tilde{y}_n) = \ell(f(x_n), \tilde{y}_n) - \alpha \cdot \ell(f(x_{n_1}), \tilde{y}_{n_2})$$

#### Theorem 4: $\alpha$-weighted의 최적 가중치

```math
\alpha^* = 1 - (1 - e_{-1} - e_{+1}) \cdot \frac{\delta_p}{\delta_{\tilde{p}}}
```

여기서 $\delta_{\tilde{p}} = P(\tilde{Y}=+1) - P(\tilde{Y}=-1)$. 이 $\alpha^*$에서:

```math
\tilde{f}^*_{\mathbb{1}_{\alpha^*\text{-peer}}} \in \arg\min_{f \in \mathcal{F}} R_{\mathcal{D}}(f)
```

특수 케이스:
- $p = 0.5 \Rightarrow \alpha^* = 1$ (표준 peer loss)
- $e_{-1} = e_{+1} \Rightarrow \alpha^* = 0$ (깨끗한 학습 설정으로 환원)

#### Theorem 5: 샘플 복잡도 보장

확률 $1-\delta$ 이상으로:

```math
R_{\mathcal{D}}(\hat{f}^*_{\mathbb{1}_{\alpha^*\text{-peer}}}) - R^* \leq \frac{1+\alpha^*}{1-e_{-1}-e_{+1}} \sqrt{\frac{2\log 2/\delta}{N}}
```

---

### 2.4 모델 구조

- **기반 모델**: 2층 ReLU Multi-Layer Perceptron (MLP)
- **손실 함수**: Peer loss (기존 cross-entropy를 peer loss로 대체)
- **학습 구조**: 표준 ERM 프레임워크 내에서 작동
- **Peer 샘플링**: 각 미니배치에서 랜덤하게 두 개의 추가 샘플을 추출하여 peer 항 계산
- **CIFAR-10**: ResNet 기반 (He et al., 2016)

---

### 2.5 성능 향상 및 한계

#### 성능 향상

**10개 UCI 벤치마크** 결과 (Table 1 요약):
- Twonorm ($e_{-1}=0.2, e_{+1}=0.4$): Peer **0.976**, Surrogate 0.919, NN 0.911
- Splice ($e_{-1}=0.2, e_{+1}=0.4$): Peer **0.901**, Surrogate 0.832, NN 0.714
- German ($e_{-1}=0.1, e_{+1}=0.3$): Peer **0.727** (Without prior equalization)

**CIFAR-10** (Table 2):

| 모델 | $\epsilon=0.2$ | $\epsilon=0.4$ |
|------|---------|---------|
| Cross Entropy | 86.67 | 82.09 |
| DMI | 85.11 | 81.67 |
| **Peer Loss** | **87.72** | **83.81** |

특히, **실제 노이즈 비율을 알고 있는 Surrogate Loss보다 더 좋거나 동등한 성능**을 여러 데이터셋에서 달성했습니다.

#### 한계

1. **균일 노이즈 가정**: 레이블 노이즈가 모든 훈련 샘플에 걸쳐 균일하다고 가정합니다(instance-independent noise)
2. **클래스 불균형 민감성**: $p \neq 0.5$일 때 $\alpha^*$ 계산에 일부 노이즈 정보($e_{+1}, e_{-1}$)가 간접적으로 필요합니다 (실제로는 validation 데이터로 튜닝 가능)
3. **비볼록성**: $\ell_{\alpha\text{-peer}}$는 일반적으로 볼록하지 않습니다 (단, 특정 조건에서 볼록성 보장)
4. **분산 증가**: peer 항의 랜덤 샘플링으로 인해 분산이 증가할 수 있습니다

---

## 3. 일반화 성능 향상 가능성

### 3.1 과적합 방지 메커니즘

Peer loss의 두 번째 항은 모델이 노이즈 레이블에 과도하게 맞추는 것을 명시적으로 방지합니다. 논문의 Figure 2와 Figure A2에서 확인할 수 있듯이, **peer loss는 cross-entropy loss와 달리 훈련 과정에서 과적합 없이 안정적인 수렴**을 보입니다.

수식적으로 이를 이해하면: peer 항 $\ell(f(x_{n_1}), \tilde{y}_{n_2})$의 기댓값은

$$\mathbb{E}[\ell(f(X_{n_1}), \tilde{Y}_{n_2})] = \mathbb{E}_X[\ell(f(X), -1)] \cdot P(\tilde{Y}=-1) + \mathbb{E}_X[\ell(f(X), +1)] \cdot P(\tilde{Y}=+1)$$

이 항은 모델이 노이즈 레이블 분포에 과도하게 집중하는 것을 **정규화 효과**로 억제합니다.

### 3.2 공식적인 일반화 한계 (Theorem 7)

확률 $1-\delta$ 이상으로:

```math
R_{\mathcal{D}}(\hat{f}^*_{\ell_{\alpha^*\text{-peer}}}) - R^* \leq \frac{1}{1-e_{-1}-e_{+1}} \cdot \Psi^{-1}_{\ell_{\alpha^*\text{-peer}}}\!\!\left(\min_{f \in \mathcal{F}} R_{\ell_{\alpha^*\text{-peer}},\tilde{\mathcal{D}}}(f) - \min_f R_{\ell_{\alpha^*\text{-peer}},\tilde{\mathcal{D}}}(f)\right) + 4(1+\alpha^*)L \cdot \Re(\mathcal{F}) + 2\sqrt{\frac{\log 4/\delta}{2N}}\left(1+(1+\alpha^*)(\bar{\ell}-\underline{\ell})\right)
```

여기서 $\Re(\mathcal{F})$는 Rademacher 복잡도.

이 한계는 세 가지 요소로 구성됩니다:
1. **추정 오류**: 가설 공간 $\mathcal{F}$ 내 근사 오류
2. **복잡도 항**: Rademacher 복잡도로 표현된 모델 복잡도
3. **통계적 항**: 샘플 수 $N$에 따라 $O(1/\sqrt{N})$으로 감소

### 3.3 분류 교정성 (Calibration)

Theorem 6에서 $\ell_{\alpha\text{-peer}}$의 교정성(calibration)을 증명합니다. 교정성은 대리 손실의 과잉 리스크가 0-1 과잉 리스크의 상한을 보장하는 성질입니다:

$$\Psi_\ell(R_{\mathcal{D}}(\tilde{f}) - R^*) \leq R_{\ell,\mathcal{D}}(\tilde{f}) - \min_f R_{\ell,\mathcal{D}}(f)$$

이 성질로 인해 peer loss를 최소화하면 진짜 0-1 리스크도 감소함이 보장됩니다.

### 3.4 날카로운 결정 경계

Figure 3, 4에서 시각적으로 확인할 수 있듯이, peer loss는 노이즈 하에서도 깨끗한 데이터로 학습한 것과 유사한 날카로운 결정 경계를 유지합니다. 이는 실제 일반화 성능 향상으로 직결됩니다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

**1. 노이즈 추정 없는 학습의 패러다임 전환**

이 논문은 "노이즈 비율을 알아야 한다"는 기존의 암묵적 전제를 깨고, 노이즈 정보를 손실 함수 구조에 암묵적으로 인코딩하는 새로운 방향을 제시했습니다. 이는 크라우드소싱, 의료 데이터, 웹 스크래핑 등 실제 환경에서 노이즈 비율 추정이 어려운 많은 응용 분야에 직접적 영향을 미칩니다.

**2. Peer Prediction과 Machine Learning의 융합**

경제학/게임이론 문헌(peer prediction)을 머신러닝에 접목한 선구적 연구로서, 두 분야의 융합 연구를 촉진할 수 있습니다.

**3. 정규화 관점에서의 해석**

Peer 항을 일종의 데이터 의존적 정규화기로 해석할 수 있어, 다양한 정규화 기법과의 결합 연구로 이어질 수 있습니다.

**4. 레이블 노이즈 이론 발전**

형식적인 이론적 보장(calibration, generalization bound, convexity condition)을 갖추었기 때문에, 후속 이론 연구의 기반이 됩니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> **주의**: 아래 논문들은 제가 학습 데이터 기반으로 알고 있는 내용이며, 2021년 이후 연구는 제 지식 한계로 인해 일부 세부 정보에서 부정확할 수 있습니다. 확실하지 않은 세부 수치는 표기하지 않겠습니다.

#### (1) Instance-Dependent Noise 연구

논문에서 스스로 한계로 지적한 "균일 노이즈 가정"을 극복하는 연구들이 등장했습니다:

- **Xia et al. (2020)**, "Parts-dependent Label Noise: Towards Instance-dependent Label Noise" — 인스턴스별로 노이즈 비율이 다른 설정으로 확장
- **Cheng et al. (2020)**, "Learning with Bounded Instance- and Label-dependent Label Noise" (ICML 2020) — 논문 자체에서 후속 연구로 언급

Peer loss는 이 설정에서 이론적 보장이 약화될 수 있어, 인스턴스별 노이즈로의 확장이 중요한 연구 방향입니다.

#### (2) Sample Selection 및 Meta-Learning 기반 접근

- **Han et al. (2018)**, "Co-teaching" 계열의 후속 연구들은 두 모델이 서로의 깨끗한 샘플을 선택하는 방식으로, peer loss의 "peer" 개념과 개념적으로 연결됩니다.

#### (3) Contrastive Learning과의 결합 가능성

2020년 이후 Self-Supervised Learning/Contrastive Learning의 부상으로, peer loss의 "비교 평가" 메커니즘이 대조 학습과 결합되는 연구 방향이 있습니다.

#### 비교 요약표

| 방법 | 노이즈 비율 필요 | 비대칭 노이즈 보장 | 이론적 보장 | 계산 복잡도 |
|------|:---------:|:---------:|:---------:|:---------:|
| Natarajan et al. (2013) | ✅ 필요 | ✅ | ✅ | 낮음 |
| Symmetric Loss (Ghosh et al., 2015) | ❌ 불필요 | ⚠️ 제한적 | ✅ | 낮음 |
| DMI (Xu et al., 2019) | ❌ 불필요 | ✅ | ⚠️ 제한적 | 높음 |
| **Peer Loss (Liu & Guo, 2020)** | **❌ 불필요** | **✅** | **✅** | **낮음** |

---

### 4.3 앞으로 연구 시 고려할 점

**1. 인스턴스 의존적 노이즈 설정으로의 확장**

현재 peer loss는 $e_{+1}, e_{-1}$이 모든 샘플에 걸쳐 일정하다고 가정합니다. 실제 데이터에서는 어려운 샘플일수록 노이즈가 더 많이 발생합니다. 이를 위해 특징 공간에서 노이즈 비율을 모델링하는 방향이 필요합니다.

**2. $\alpha$ 튜닝의 자동화**

$p \neq 0.5$인 경우 최적 $\alpha^*$를 계산하려면 일부 노이즈 정보가 필요합니다. 검증 데이터 없이 $\alpha$를 자동으로 추정하는 방법론이 필요합니다.

**3. 대규모 데이터셋 및 복잡한 모델로의 확장**

CIFAR-10에서 예비 결과만 제공되었으므로, ImageNet 수준의 대규모 데이터셋과 최신 Transformer 기반 모델(ViT 등)에서의 성능 검증이 필요합니다.

**4. Peer 샘플링 전략의 개선**

현재는 균일 랜덤 샘플링을 사용하지만, 분산 감소를 위한 더 정교한 샘플링 전략(예: 같은 클래스 내 샘플링, 중요도 가중 샘플링)이 성능을 향상시킬 수 있습니다.

**5. Semi-supervised 및 Few-shot Learning과의 결합**

논문에서도 언급했듯이, peer loss를 반지도 학습이나 소량 레이블 환경에 적용하는 연구가 유망합니다.

**6. 다중 레이블 및 구조적 예측으로의 확장**

현재는 이진 분류(및 일부 다중 클래스)만 다루었지만, 다중 레이블 분류, 시퀀스 레이블링 등으로의 확장이 필요합니다.

**7. 개인정보 보호(Differential Privacy)와의 결합**

논문 자체에서 differentially private ERM과의 결합을 향후 연구로 언급했습니다. 노이즈가 있는 레이블과 프라이버시 보호를 동시에 달성하는 것은 중요한 실용적 문제입니다.

---

## 참고자료

- **주 논문**: Yang Liu and Hongyi Guo, "Peer Loss Functions: Learning from Noisy Labels without Knowing Noise Rates," *Proceedings of the 37th International Conference on Machine Learning (ICML 2020)*, PMLR 119, 2020. (arXiv:1910.03231v7)
- Natarajan, N., Dhillon, I. S., Ravikumar, P. K., and Tewari, A., "Learning with noisy labels," *Advances in Neural Information Processing Systems*, 2013.
- Ghosh, A., Manwani, N., and Sastry, P., "Making risk minimization tolerant to label noise," *Neurocomputing*, 2015.
- Xu, Y., Cao, P., Kong, Y., and Wang, Y., "L_DMI: An Information-theoretic Noise-robust Loss Function," *NeurIPS*, 2019.
- Shnayder, V., Agarwal, A., Frongillo, R., and Parkes, D. C., "Informed truthfulness in multi-task peer prediction," *Proceedings of the 2016 ACM Conference on Economics and Computation*, 2016.
- Bartlett, P. L., Jordan, M. I., and McAuliffe, J. D., "Convexity, classification, and risk bounds," *Journal of the American Statistical Association*, 2006.
- Xia, X., Liu, T., Han, B., et al., "Parts-dependent label noise: Towards instance-dependent label noise," 2020.
- Cheng, J., Liu, T., Ramamohanarao, K., and Tao, D., "Learning with bounded instance- and label-dependent label noise," *ICML*, 2020.
