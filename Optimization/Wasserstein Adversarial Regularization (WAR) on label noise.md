# Wasserstein Adversarial Regularization (WAR) on label noise

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

WAR의 핵심 주장은 **레이블 노이즈 문제를 클래스 간 기하학적 유사성을 고려한 선택적(selective) 정규화로 해결**할 수 있다는 것입니다.

기존 Adversarial Regularization(AR)은 모든 클래스 쌍에 동일한 강도의 정규화를 적용하여, 유사한 클래스(예: wolfdog ↔ husky) 사이의 복잡한 결정 경계까지 과도하게 매끄럽게(over-smooth) 만들어버리는 문제가 있었습니다. WAR는 **Wasserstein 거리와 ground cost 행렬 $C$를 이용해 클래스 쌍마다 다른 정규화 강도를 부여**함으로써 이 문제를 해결합니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **선택적 정규화 프레임워크** | Wasserstein 거리 기반의 anisotropic(비등방성) 정규화 설계 |
| **Ground cost 설계 방법론** | word2vec 임베딩( $\text{WAR}\_{\text{w2v}}$), CNN 임베딩($\text{WAR}_{\text{embed}}$) 두 가지 제안 |
| **이론적 분석** | WAR가 가중 Total Variation의 상한을 최소화함을 증명 (Proposition 2) |
| **폭넓은 실험 검증** | 벤치마크(Fashion-MNIST, CIFAR-10/100), 실세계(Clothing1M, Vaihingen 위성영상) |
| **오픈셋 노이즈 대응** | 레이블 거부(rejection) 없이도 오픈셋 노이즈에 경쟁력 있는 성능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

딥러닝 모델은 **암기 능력(memorization)**이 매우 강해 노이즈 레이블에 과적합(overfitting)되기 쉽습니다 [Zhang et al., ICLR 2017]. 특히:

- **대칭 노이즈(symmetric noise)**: 레이블이 모든 클래스에 균등하게 뒤집힘 → 비현실적
- **비대칭 노이즈(asymmetric noise)**: 유사한 클래스 간에 레이블이 뒤집힘 → 실세계에서 더 흔함

기존 방법들의 한계:
- **데이터 정제 방법**: 클린 검증 데이터가 필요하거나 추가 레이블링 비용 발생
- **전이 확률 행렬 기반**: 노이즈 전이 행렬 추정 자체가 어려움
- **기존 AR/VAT**: 클래스 유사성을 무시한 등방성(isotropic) 정규화

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 기본 경험적 위험 최소화

주어진 레이블 데이터 $\{x_i, y_i\}\_{i=1,\ldots,N}$에 대해 분류기 $p_\theta$를 학습:

$$\arg\min_\theta \sum_{i=1}^{N} L(x_i, y_i, p_\theta)$$

크로스엔트로피 손실:

$$L_{\text{CE}}(x_i, y_i, p_\theta) = -\sum_{c=1}^{C} y_i^{(c)} \log p_\theta(x_i)^{(c)}$$

#### Step 2: Adversarial Regularization (AR)

노이즈 레이블 과적합을 막기 위한 전체 손실:

$$L_{\text{tot}}(x_i, y_i, p_\theta) = L_{\text{CE}}(x_i, y_i, p_\theta) + \beta R_{\text{AR}}(x_i, p_\theta)$$

여기서 $R_{\text{AR}}$은:

$$R_{\text{AR}}(x_i, p_\theta) = D(p_\theta(x_i + r_i^a), p_\theta(x_i))$$

$$\text{with} \quad r_i^a = \arg\max_{r_i, \|r_i\| \leq \varepsilon} D(p_\theta(x_i + r_i), p_\theta(x_i))$$

$D$로 KL 발산을 사용 시, 이는 **레이블 스무딩과 동치(Proposition 1)**:

$$L_{\text{tot}}(x_i, y_i, p_\theta) \equiv L_{\text{CE}}(x_i, (1-\gamma)y_i + \gamma y_i^a, p_\theta) - \gamma H(y_i^a)$$

여기서 $\gamma = \frac{\beta}{\beta+1} \in [0,1[$, $y_i^a = p_\theta(x_i + r_i^a)$.

**AR의 한계**: $D$가 등방성(isotropic)이므로 모든 클래스 쌍을 동일하게 처리 → 유사 클래스 간 복잡한 경계도 과도하게 평탄화.

#### Step 3: WAR — Wasserstein 기반 정규화

$D$를 **Wasserstein 거리(Optimal Transport 거리)**로 교체:

$$R_{\text{WAR}}(x_i) = OT_C^\lambda(p_\theta(x_i + r_i^a), p_\theta(x_i))$$

$$\text{with} \quad r_i^a = \arg\max_{r_i, \|r_i\| \leq \varepsilon} OT_C^\lambda(p_\theta(x_i + r_i), p_\theta(x_i))$$

**Entropic Regularized OT** (Sinkhorn 알고리즘):

$$OT_C^\lambda(\alpha, \beta) = \langle T_\lambda^*, C \rangle$$

$$\text{with} \quad T_\lambda^* = \arg\min_{T \in U(\alpha,\beta)} \langle T, C \rangle - \lambda H(T)$$

여기서:
- $U(\alpha, \beta) = \{T \mid T \geq 0, T\mathbf{1} = \alpha, T^\top \mathbf{1} = \beta\}$: 결합 확률분포 공간
- $C \in \mathbb{R}^{C \times C}$: **ground cost 행렬** (클래스 간 유사도 인코딩)
- $\lambda \geq 0$: 엔트로픽 정규화 강도 (논문에서 $\lambda = 0.05$ 사용)

#### Step 4: Ground Cost 행렬 $C$ 설계

**옵션 1 — $\text{WAR}_{\text{w2v}}$**: word2vec 임베딩 기반

$$C_{ij} = e^{-\|v_i - v_j\|_2}, \quad C_{ii} = 0$$

여기서 $v_i$는 클래스 $i$의 word2vec 임베딩 벡터.

**옵션 2 — $\text{WAR}_{\text{embed}}$**: 사전학습 CNN(ResNet-18)으로 임베딩된 클래스 중심(centroid) 간 거리 사용.

**특수 케이스 — $\text{WAR}_{0-1}$**: 0-1 cost matrix → Total Variation 손실과 동치, AR과 유사.

#### Step 5: 이론적 보장 (Proposition 2)

대칭 cost $C$ ( $C_{ii}=0$, $\forall i$ ) 하에서 $R_{\text{WAR}}$ 최소화는 가중 Total Variation의 상한을 최소화하는 것과 동치:

$$\underline{c} \cdot TV(p_\theta(x), p_\theta(x+r)) \leq \sum_k \underline{c}_k |p_\theta(x)_k - p_\theta(x+r)_k| \leq OT_C^\lambda(p_\theta(x), p_\theta(x+r))$$

여기서 $\underline{c}\_k = \min_{i, i\neq k} c_{k,i}$ (행 $k$의 최소 비대각 cost), $\underline{c} = \min_k \underline{c}_k$.

#### Step 6: Adversarial 방향 계산

$r=0$에서 OT의 2차 Taylor 전개:

$$OT_C^\lambda(p_\theta(x), p_\theta(x+r)) \underset{r=0}{\sim} \frac{1}{2} r^\top H_r r$$

Power iteration으로 헤시안 $H_r$의 주 고유벡터 $d$를 추정 후:

$$r = \varepsilon \frac{d}{\|d\|_2}$$

### 2.3 모델 구조

| 실험 | 모델 아키텍처 |
|------|--------------|
| Fashion-MNIST, CIFAR-10/100 | 9-layer CNN (BatchNorm + Dropout + LeakyReLU) |
| Clothing1M | ResNet-50 (ImageNet 사전학습) |
| Vaihingen 위성영상 | U-Net (5채널 입력) |
| Open-set noise | 6-layer CNN + FC |

**주요 하이퍼파라미터**:
- $\beta = 10$ (정규화 가중치)
- $\lambda = 0.05$ (Sinkhorn 엔트로픽 파라미터, 20 iterations)
- $\varepsilon = 0.005$ (perturbation 반경, Clothing1M은 0.5)

**계산 복잡도**: OT 복잡도는 클래스 수 $C$에 대해 $\mathcal{O}(b \times n_s \times C^2)$ (샘플 수에 대해 선형). AR 대비 약 20% 추가 비용.

### 2.4 성능 향상

**벤치마크 분류 (비대칭 40% 노이즈)**:

| 데이터셋 | CCE | Co-Teaching | Pencil | JoCoR | $\text{WAR}_{\text{w2v}}$ |
|----------|-----|-------------|--------|-------|--------------------------|
| Fashion-MNIST | 78.85% | 86.83% | 90.17% | 84.86% | **90.41%** |
| CIFAR-10 | 76.23% | 80.87% | 84.48% | 82.07% | **84.76%** |
| CIFAR-100 | 42.45% | 42.73% | 45.70% | 42.26% | **58.86%** |

특히 **CIFAR-100 40% 노이즈에서 약 15%p 향상**은 클래스 수가 많을수록 WAR의 ground cost 활용이 효과적임을 보여줌.

**실세계 노이즈 (Clothing1M)**:

| 방법 | 정확도 |
|------|--------|
| CCE | 68.80% |
| SL | 71.02% |
| $\text{WAR}_{\text{w2v}}$ (unsupervised) | **71.61%** |
| $\text{WAR}_{\text{w2v}}$ (val 기준) | **72.20%** |

### 2.5 한계

1. **Ground cost 설계 의존성**: word2vec이나 사전학습 임베딩에 의존하며, 클래스 이름의 의미적 관계가 명확하지 않은 도메인(의료 영상 등)에서 설계가 어려움.
2. **정적(static) Ground cost**: 훈련 중 $C$ 행렬을 고정 사용. 데이터 분포 변화에 적응하지 못함.
3. **하이퍼파라미터 민감도**: $\beta$ 값에 따라 성능 변동 존재 (Table 3 참조).
4. **클린 검증 데이터 미사용**: 현실적인 가정이지만, 클린 검증 데이터 활용 시 추가 성능 향상 여지.
5. **대규모 클래스 확장성**: OT 복잡도가 $C^2$에 비례하므로 클래스가 수천 개인 경우 추가 최적화 필요.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 WAR가 일반화를 개선하는 메커니즘

#### (1) Lipschitz 상수의 국소적 최소화

Proposition 2에 의해, $R_{\text{WAR}}$를 최소화하는 것은 $p_\theta$의 국소 Lipschitz 상수를 최소화하는 것과 유사합니다:

$$\min_\theta \mathbb{E}_x[R_{\text{WAR}}(x)] \approx \min_\theta \mathbb{E}_x\left[\|\nabla_x p_\theta(x)\|_{\text{weighted}}\right]$$

이는 **Sobolev 정규화**와 연결되어, 모델의 입력에 대한 기울기 크기를 제한함으로써 일반화를 향상시킵니다.

#### (2) 선택적 경계 복잡도 제어

$$\text{유사 클래스 쌍 (고비용)} \Rightarrow \text{복잡한 결정 경계 허용}$$

$$\text{비유사 클래스 쌍 (저비용)} \Rightarrow \text{단순한 결정 경계 강제}$$

이 **anisotropic 정규화**는 실제 클래스 분포의 구조를 반영하여, 노이즈로 인한 가짜 복잡성(spurious complexity)만 제거하고 진짜 클래스 간 구별력은 보존합니다.

#### (3) 적응적 레이블 스무딩

Proposition 1의 확장으로, WAR는 단순히 레이블을 균등하게 스무딩하는 것이 아니라 **클래스 유사도에 따라 차등적으로 스무딩**합니다:

$$\tilde{y}_i = (1-\gamma)y_i + \gamma y_i^a$$

여기서 $y_i^a$의 분포는 ground cost $C$에 의해 유도되어, 유사 클래스 방향으로 더 많이 스무딩됩니다.

#### (4) Open-set 노이즈에서의 일반화

WAR는 레이블 거부(rejection) 없이도 open-set 노이즈(SVHN: 82.03%, ImageNet32: 80.61%)에서 경쟁력 있는 성능을 보여, 일반적인 분포 외(out-of-distribution) 샘플에 대한 견고성도 향상됨을 시사합니다.

#### (5) $\beta$ 증가에 따른 비단조적(non-monotonic) 거동

Table 3에서 $\text{WAR}_{\text{w2v}}$는 $\beta$가 증가할수록 지속적으로 성능이 향상(최대 86.73% at $\beta=20$)하는 반면, AR은 $\beta=20$에서 57.36%로 급감합니다. 이는 WAR의 선택적 정규화가 **과정규화(over-regularization) 문제에 근본적으로 더 강인함**을 보여줍니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

#### (1) OT와 Robust Learning의 교차점 개척
WAR는 최적수송 이론을 레이블 노이즈 정규화에 성공적으로 적용한 선구적 연구로, **기하학적 구조를 활용한 노이즈 강건 학습** 방향을 제시합니다.

#### (2) Ground cost 설계의 중요성 부각
클래스 관계 사전 지식의 효과적인 인코딩이 노이즈 강건성과 직결됨을 실험적으로 증명하여, **메트릭 학습(metric learning)과 노이즈 강건 학습의 통합**을 촉진합니다.

#### (3) 범용적 정규화 프레임워크
WAR는 반지도 학습, 적대적 강건성, 도메인 적응 등 다양한 문제에 확장 가능한 **플러그인형 정규화 모듈**로 활용될 수 있습니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> **주의**: 아래는 논문에서 언급된 방법들과 WAR 이후 발전 동향을 바탕으로 작성하였으며, 2020년 이후 각 논문의 구체적 수치는 해당 논문 원문을 직접 확인하시기 바랍니다.

| 연구 방향 | 대표 접근법 | WAR와의 관계 |
|-----------|------------|--------------|
| **Semi-supervised 기반** | DivideMix (Li et al., ICLR 2020) | 깨끗한 샘플 선별 후 MixUp → WAR와 상호보완적 |
| **샘플 선택 + 정규화** | ELR (Liu et al., NeurIPS 2020) | 이른 학습(early learning) 정규화 → WAR의 레이블 스무딩 관점과 유사 |
| **그래프 기반** | PTD-R-V (Xia et al., NeurIPS 2020) | 노이즈 전이 행렬의 부분 레이블 의존성 모델링 |
| **대조 학습** | NGC (Wu et al., ICCV 2021) | 대조 학습으로 표현 학습 + WAR의 ground cost와 결합 가능성 |
| **동적 레이블 보정** | ProMix (Cordeiro et al.) | 프로토타입 기반 동적 레이블 수정 → WAR의 정적 ground cost 한계 보완 |

논문에서 직접 언급된 관련 연구:
- **JoCoR** [Wei et al., CVPR 2020]: Co-regularization 기반 공동 훈련 → WAR와 결합 가능
- **Entropic OT loss** [Damodaran et al., CVIU 2020]: WAR의 원저자들이 발표한 원격탐사 특화 버전

### 4.3 앞으로 연구 시 고려할 점

#### 기술적 개선 방향

**1. 동적 Ground Cost 학습**

현재 WAR는 훈련 전 고정된 $C$를 사용합니다. 아래와 같은 동적 업데이트가 효과적일 수 있습니다:

$$C^{(t+1)} = f(C^{(t)}, \text{Confusion}(p_\theta^{(t)}), \text{EmbeddingSpace}^{(t)})$$

혼동 행렬(confusion matrix)이나 현재 모델의 임베딩 공간을 반영한 적응적 cost 행렬 업데이트가 연구 과제입니다.

**2. 메트릭 학습과의 통합**

Ground cost $C$를 직접 학습 가능한 파라미터로 만들되, 노이즈 레이블로 인한 과적합 방지 메커니즘을 설계해야 합니다 [Cuturi & Avis, JMLR 2014].

**3. 대용량 클래스 확장**

OT 복잡도 $\mathcal{O}(C^2)$는 수천 개 클래스(예: ImageNet 1K)에서 병목이 됩니다. Sliced Wasserstein 거리나 스케일러블 OT 알고리즘의 적용을 고려해야 합니다.

**4. 대조 학습과의 결합**

레이블 정보 없이 학습된 표현(representation)으로부터 ground cost를 계산하면, 레이블 노이즈에 오염되지 않은 더 신뢰할 수 있는 $C$를 구성할 수 있습니다.

**5. 장기 꼬리 분포(Long-tail) 문제와의 통합**

클래스 불균형 + 레이블 노이즈가 동시에 존재하는 실세계 시나리오에서 WAR의 ground cost를 클래스 빈도에 따라 조정하는 연구가 필요합니다.

#### 이론적 연구 방향

**6. 일반화 오차 경계의 이론적 분석**

$$\mathcal{E}(p_\theta) \leq \hat{\mathcal{E}}_{\text{noisy}}(p_\theta) + \underbrace{\text{WAR 정규화 항}}_{\text{복잡도 제어}} + \underbrace{\mathcal{O}\left(\sqrt{\frac{\log(1/\delta)}{n}}\right)}_{\text{통계적 항}}$$

형태의 PAC 학습 이론적 보장을 도출하면 이론적 기여가 커집니다.

**7. 인스턴스 의존적(instance-dependent) 노이즈 모델링**

현재 WAR는 클래스 수준의 cost를 사용하지만, 개별 샘플마다 다른 노이즈 확률을 갖는 인스턴스 의존 노이즈에 대한 확장이 필요합니다.

#### 응용 연구 방향

**8. 의료 영상, NLP 등 도메인 특화 적용**

클래스 유사도가 전문가 지식으로 잘 정의되는 의료 영상 분류(양성 vs. 악성 등)에서 WAR의 ground cost를 임상 지식으로 설계하는 연구가 유망합니다.

**9. 연합 학습(Federated Learning)에서의 분산 노이즈**

여러 클라이언트에서 발생하는 이질적 레이블 노이즈에 WAR를 적용할 때, 각 클라이언트의 ground cost를 어떻게 집계할지가 연구 과제입니다.

---

## 참고 자료

**주 논문**:
- Fatras, K., Damodaran, B.B., Lobry, S., Flamary, R., Tuia, D., & Courty, N. (2021). **"Wasserstein Adversarial Regularization for learning with label noise."** arXiv:1904.03936v3 [cs.LG].

**논문 내 핵심 참고문헌**:
- Miyato, T., et al. (2018). "Virtual Adversarial Training." *IEEE TPAMI*. [VAT, 기반 방법]
- Cuturi, M. (2013). "Sinkhorn distances." *NeurIPS*. [Sinkhorn 알고리즘]
- Peyré, G. & Cuturi, M. (2019). *Computational Optimal Transport.* Now Publishers. [OT 이론]
- Patrini, G., et al. (2017). "Making DNNs robust to label noise." *CVPR*. [Forward/Backward correction]
- Han, B., et al. (2018). "Co-teaching." *NeurIPS*. [Co-Teaching]
- Wang, Y., et al. (2019). "Symmetric Cross Entropy." *ICCV*. [SL]
- Frogner, C., et al. (2015). "Learning with a Wasserstein loss." *NeurIPS*. [Wasserstein loss 학습]
- Damodaran, B.B., et al. (2020). "An entropic optimal transport loss for learning DNNs under label noise in remote sensing." *Computer Vision and Image Understanding*, 191, 102863.
- Mikolov, T., et al. (2013). "Word2vec." *NeurIPS*. [Ground cost 설계]
- He, K., et al. (2016). "Deep residual learning." *CVPR*. [ResNet-18/50]
