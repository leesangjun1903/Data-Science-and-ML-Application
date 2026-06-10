# Deep k-NN for Noisy Labels

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

**"Deep k-NN for Noisy Labels"** (Bahri, Jiang & Gupta, 2020, arXiv:2004.12289)의 핵심 주장은 다음과 같습니다:

> 사전 학습된 딥 모델의 **logit 레이어(중간 표현층)**에서 k-최근접 이웃(k-NN) 기반 필터링을 수행하면, 복잡한 최신 기법들과 동등하거나 더 나은 성능으로 노이즈 레이블을 제거할 수 있다.

### 주요 기여 (3가지)

| 기여 영역 | 내용 |
|-----------|------|
| **실험적 기여** | 딥 모델의 중간 레이어에 k-NN 필터링을 적용하면 최신 방법들과 경쟁하거나 능가함을 입증 |
| **이론적 기여** | 점근적으로 k-NN이 Bayes-optimal 레이블이 아닌 경우에만 오염된 샘플로 식별함을 증명 (Theorem 1, 2, 3) |
| **실용적 기여** | 하이퍼파라미터 $k$에 대한 강건성 및 클린 데이터셋 없이도 작동 가능함을 실증 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

현대 머신러닝에서 대규모 데이터셋의 레이블은 자동화된 방식(클릭, 크라우드소싱 등)으로 수집되어 필연적으로 **노이즈 레이블(noisy labels)** 이 포함됩니다. 이는 모델 성능 저하의 주요 원인이며, 기존 방법들은 다음의 문제점을 가집니다:

- **혼동 행렬 기반 방법**: 오염율(corruption rate) 추정이 어려움
- **보조 모델 기반 방법**: 신뢰할 수 있는 클린 데이터셋 필수
- **손실 함수 수정 방법**: 수렴 속도 저하 또는 정확도 손실

본 논문은 이러한 문제를 **단순하면서도 이론적으로 보장된** 방법으로 해결하고자 합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 알고리즘 구성

**Algorithm 1: Deep k-NN Filtering**

```
입력: D_noisy, D_clean (비어있어도 가능), k, 모델 아키텍처 A

1. Filtering Model Train Set Selection Procedure 실행:
   - D_clean을 70/30으로 분할: D_cleanTrain, D_cleanVal
   - 모델 M1: D_cleanTrain으로 학습
   - 모델 M2: D_cleanTrain ∪ D_noisy로 학습
   - D_cleanVal에서 더 좋은 모델의 학습 데이터를 선택

2. 선택된 데이터로 예비 모델 M 학습

3. M의 logit 레이어에서 k-NN 수행:
   - 각 샘플의 k개 이웃 레이블 다수결과 자신의 레이블 비교
   - 불일치 시 해당 샘플 제거 → D_filtered 생성

4. 최종 모델을 A로 D_filtered ∪ D_clean에서 재학습
```

#### k-NN 판별 함수

$$\eta_k(y; x) := \frac{1}{|N_k(x)|} \sum_{i=1}^{n} \mathbf{1}[y_i = y,\; x_i \in N_k(x)]$$

여기서 $N_k(x)$는 $x$의 $k$-최근접 이웃 집합이며, 예측은:

$$\hat{\eta}_k(x) := \arg\max_y \; \eta_k(y; x)$$

#### 오염 집합의 최소 쌍별 거리 (핵심 개념)

$$S_2(C) := \min_{x, x' \in C,\; x \neq x'} |x - x'|$$

이 값이 클수록 오염된 샘플들이 더 넓게 분산되어 있어, 더 적은 클린 샘플로도 노이즈를 효과적으로 제거할 수 있습니다.

#### $\Delta$ -interior 영역 정의

```math
\mathcal{X}^{\Delta} := \left\{ x \in \mathcal{X} : \left|\frac{1}{2} - \eta(x)\right| \geq \Delta \right\}
```

여기서 $\Delta$는 Bayes 최적 레이블에 대한 확률적 마진을 나타냅니다.

---

### 2.3 이론적 분석 (주요 정리들)

#### Theorem 1 (Fixed $\Delta$)

가정 1, 2, 3이 성립할 때, 다음의 $k$ 범위에서:

$$k \geq K_l \cdot \frac{1}{\Delta^2} \cdot \log^2(1/\delta) \cdot \log n$$

$$k \leq K_u \cdot \min\{S_2(C)^D,\; \Delta^{D/\alpha}\} \cdot n$$

$\mathcal{X}^{\Delta}$ 위의 모든 $x$에 대해 **최소 $1-\delta$의 확률**로, k-NN 예측이 레이블과 일치하는 경우는 오직 그 레이블이 Bayes-optimal인 경우뿐입니다.

#### Theorem 2 (수렴율, $\Delta \to 0$)

$$K_l \cdot \log^2(1/\delta) \cdot n^{\frac{\alpha}{\alpha+D}} \leq k \leq K_u \cdot S_2(C)^D \cdot n$$

일 때, 다음을 만족하는 $\Delta$에 대해 결과가 성립:

$$\Delta = K \cdot \left(\sqrt{\frac{\log n + \log(1/\delta)}{k}} + \left(\frac{k}{n}\right)^{\alpha/D}\right)$$

> **Remark 1**: $k = O(n^{2\alpha/(2\alpha+D)})$ 선택 시 $\Delta = \tilde{O}(n^{-\alpha/(2\alpha+D)})$로, 이는 비오염 설정에서의 **minimax-optimal rate**와 일치합니다.

#### Theorem 3 (Tsybakov Noise Condition 하에서의 수렴율)

추가 가정:

$$\mathbb{P}_{\mathcal{X}}(x \notin \mathcal{X}^{\Delta}) \leq C_{\beta} \cdot \Delta^{\beta}$$

이 성립할 때:

$$\mathbb{P}(\eta_k(x) \neq \eta^*(x)) \leq K \cdot \lambda^{\beta}$$

$$R_X - R^* \leq K' \cdot \lambda^{\beta+1}$$

$$\lambda = \left(\sqrt{\frac{\log n + \log(1/\delta)}{k}} + \left(\frac{k}{n}\right)^{\alpha/D}\right)$$

> **Remark 2**: $k = O(n^{2\alpha/(2\alpha+D)})$ 선택 시 excess risk의 수렴율은 $\tilde{O}(n^{-\alpha(\beta+1)/(2\alpha+D)})$로, Audibert et al. (2007)의 하한과 로그 인수 범위 내에서 일치합니다.

---

### 2.4 모델 구조

```
[입력: 노이즈 있는 훈련 데이터]
         ↓
[예비 딥 모델 학습 (ResNet-20, FC-DNN 등)]
         ↓
[Logit Layer (중간 표현 추출)]
         ↓
[k-NN 필터링: 이웃 레이블과 불일치 샘플 제거]
         ↓
[D_filtered 구성]
         ↓
[최종 딥 모델 재학습]
         ↓
[정제된 고성능 모델]
```

실험에 사용된 백본 모델:
- UCI 데이터셋: FC-DNN (hidden dim 100, ReLU)
- MNIST/Fashion-MNIST: 2층 FC-DNN (256 units, ReLU)
- CIFAR10/100, SVHN: **ResNet-20**

---

### 2.5 성능 향상

#### 노이즈 유형별 성능 (Uniform Noise 기준, Table 1)

| 데이터셋 | % Clean | Forward | GLC | Distill | **k-NN** |
|---------|---------|---------|-----|---------|----------|
| Letters | 5% | 4.55 | 2.33 | 2.48 | **2.05** |
| Phonemes | 5% | 7.89 | 2.12 | 1.79 | **1.26** |
| MNIST | 5% | 2.88 | 0.50 | 1.03 | **0.40** |
| CIFAR10 | 5% | 6.74 | 5.43 | 6.86 | **5.03** |
| SVHN | 5% | 5.04 | 1.99 | 3.56 | **1.62** |

(낮을수록 좋음: test error vs noise rate 곡선 아래 면적)

---

### 2.6 한계점

| 한계 | 설명 |
|------|------|
| **Hard Flip 노이즈 취약** | 의미적으로 유사한 클래스 간 구조적 노이즈에서 성능 저하 |
| **고차원 logit 공간** | CIFAR-100처럼 100차원 logit 공간에서 k-NN의 차원의 저주 발생 |
| **소규모 데이터셋에서 $k$ 민감성** | 데이터가 적을 경우 $k$ 선택이 중요해짐 |
| **계산 비용** | k-NN Classify는 런타임이 느림 (실용적 한계) |
| **이론적 갭** | 상한과 하한을 동시에 제공하는 완전한 이론적 틀 미완성 |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상의 핵심 메커니즘

#### (1) 노이즈 제거를 통한 훈련 데이터 품질 향상

노이즈 레이블이 포함된 데이터로 학습 시, 모델은 다음의 경험적 위험(empirical risk)을 최소화합니다:

$$\hat{R}_{\text{noisy}}(f) = \frac{1}{n} \sum_{i=1}^{n} \ell(f(x_i), \tilde{y}_i)$$

여기서 $\tilde{y}_i$는 오염된 레이블입니다. 반면, k-NN 필터링 후:

$$\hat{R}_{\text{filtered}}(f) = \frac{1}{|D_{\text{filtered}}|} \sum_{(x_i, y_i) \in D_{\text{filtered}}} \ell(f(x_i), y_i)$$

오염된 샘플이 제거됨으로써 $\hat{R}\_{\text{filtered}}$가 진정한 위험 $R(f) = \mathbb{E}_{(X,Y) \sim F}[\ell(f(X), Y)]$에 더 근접하게 됩니다.

#### (2) Bayes-optimal 수렴 보장

Theorem 1에 의해, k-NN 필터링은 점근적으로 Bayes-optimal 레이블을 가진 샘플만 통과시킵니다. 즉, 오염된 레이블이 모델의 의사결정 경계 학습에 미치는 악영향이 제거됩니다.

$$\eta^*(x) := \mathbf{1}\left[\eta(x) \geq \frac{1}{2}\right]$$

이를 통해 재학습된 최종 모델의 일반화 오차:

$$R_X - R^* \leq K' \cdot \lambda^{\beta+1}$$

이 minimax-optimal rate로 수렴하여 **진정한 Bayes 위험에 가까운 일반화 성능**을 달성합니다.

#### (3) 중간 표현의 견고성

노이즈 데이터로 학습된 예비 모델이더라도, **중간 레이어(logit space)의 표현은 여전히 클래스 구조를 포착**합니다. 이는 딥러닝의 표현 학습 능력이 어느 정도의 레이블 노이즈에 강건하다는 Rolnick et al. (2017)의 관찰과 일치합니다.

#### (4) $S_2(C)$와 일반화 성능의 관계

$$S_2(C) \uparrow \implies \text{더 적은 클린 샘플로도 높은 일반화 성능 달성}$$

Figure 1에서 확인된 바와 같이, 오염 샘플들이 더 넓게 분산될수록($S_2(C)$ 증가) 테스트 정확도가 향상됩니다.

#### (5) $k$의 안정성과 실용적 일반화

넓은 범위의 $k$ 값에서 일관된 성능을 보이므로, 검증 셋이 없는 실제 환경에서도 안정적인 일반화 성능을 기대할 수 있습니다.

---

## 4. 연구에 미치는 영향 및 앞으로 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### (A) 방법론적 영향

1. **단순성의 재발견**: 정교한 복잡한 방법보다 잘 설계된 단순 방법의 효과를 입증하여, 이후 연구에서 베이스라인 복잡도를 재고하게 만드는 계기 제공

2. **표현 공간의 중요성 부각**: Logit space를 메트릭 공간으로 활용하는 아이디어는 이후 contrastive learning, self-supervised learning과의 결합 연구로 이어질 수 있음

3. **데이터 중심 AI(Data-Centric AI)**: 모델 구조 개선보다 데이터 품질 향상이 성능에 더 큰 영향을 미칠 수 있다는 패러다임 강화

#### (B) 이론적 영향

- Finite-sample 수렴 보장을 노이즈 환경으로 확장한 최초 결과로서, 이후 노이즈 레이블 이론 연구의 기준점 제공
- $S_2(C)$ 개념은 오염 배열의 구조적 특성을 정량화하는 새로운 프레임워크로 발전 가능

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 2020년 이후 연구들에 대한 내용은 제가 학습한 데이터를 기반으로 한 것으로, 해당 논문들의 정확한 수치나 세부 내용은 원본 논문을 직접 확인하시기 바랍니다.

#### 주요 후속 연구들과의 비교

| 논문 | 연도 | 핵심 방법 | Deep k-NN과의 차이점 |
|------|------|----------|---------------------|
| **DivideMix** (Li et al., 2020) | 2020 | GMM으로 클린/노이지 샘플 분리 후 MixMatch | 확률론적 분리 vs. 결정론적 k-NN 필터링 |
| **SELFIE** (Song et al., 2019 → 확장) | 2020+ | 자기 교정(self-correcting) 메커니즘 | 레이블 수정 vs. 레이블 제거 |
| **ELR** (Liu et al., 2020) | 2020 | Early-learning regularization | 정규화 기반 vs. 데이터 필터링 기반 |
| **Noisy Student** (Xie et al., 2020) | 2020 | 의사 레이블 + 노이즈 주입 반복학습 | 자기학습 기반 vs. k-NN 기반 필터링 |
| **C2D** (Zheltonozhskii et al., 2022) | 2022 | Contrastive learning으로 노이즈 감지 | 자기지도 표현 vs. 지도학습 logit 표현 |

#### DivideMix와의 심층 비교

**DivideMix** (Li et al., Learning with Noisy Labels by Dividemix, ICLR 2020):
- GMM을 사용하여 각 샘플을 클린/노이즈로 확률적으로 분류
- MixMatch 반지도학습으로 두 세트를 통합 활용
- CIFAR-10에서 매우 높은 성능 보고

vs. **Deep k-NN**:
- 결정론적 하드 필터링 방식
- 이론적 보장이 명확함
- 클린 데이터셋 없이도 작동 가능
- 구현 복잡도가 상대적으로 낮음

#### Contrastive Learning 기반 접근법의 등장

2020년 이후 **contrastive learning** (SimCLR, MoCo 등)의 발전으로, 자기지도 학습으로 더 강건한 표현을 얻고 이를 노이즈 감지에 활용하는 방법들이 등장했습니다. 이는 Deep k-NN의 "표현 공간에서의 k-NN 활용" 아이디어를 발전시킨 것으로 볼 수 있습니다.

---

### 4.3 앞으로 연구 시 고려할 점

#### (1) 방법론적 고려사항

- **표현 품질 개선**: Contrastive learning이나 self-supervised pre-training으로 얻은 표현에 k-NN 필터링을 적용하면 성능이 향상될 수 있음
- **소프트 필터링**: 하드 바이너리 결정 대신, k-NN 불일치 정도를 가중치로 사용하는 소프트 필터링 탐구
- **적응적 $k$ 선택**: 데이터 밀도에 따라 로컬하게 $k$를 적응적으로 결정하는 방법

$$k(x) = f(\hat{p}_X(x), D, \text{noise rate estimate})$$

#### (2) Hard Flip 노이즈 대응

구조적 노이즈(semantically similar class confusion)에 강건한 방법 개발이 필요합니다:
- 클래스 간 의미적 거리를 반영한 가중 k-NN
- 계층적 레이블 구조 활용

#### (3) 고차원 문제 해결 (차원의 저주)

CIFAR-100과 같은 고차원 logit space에서의 성능 저하 해결을 위해:
- 차원 축소 기법 (PCA, UMAP 등) 결합
- 클래스 수에 따른 적응적 표현 공간 설계

#### (4) 이론적 연구 방향

- 노이즈 환경에서의 상한(upper bound)과 하한(lower bound)의 동시 도출
- $S_2(C)$ 외의 오염 구조 복잡도 측도 개발
- 비이진 분류 문제에서의 완전한 이론적 확장

#### (5) 실용적 고려사항

- **온라인 학습 환경**: 스트리밍 데이터에서 k-NN 필터링 실시간 적용
- **레이블 수정(correction) vs. 제거(removal)**: 필터링 외에 자동 레이블 교정으로 확장
- **클린 데이터 없는 환경 강화**: $D_{\text{clean}} = \emptyset$ 시나리오에서의 성능 향상

---

## 참고 자료

### 주요 원본 논문
- **Bahri, D., Jiang, H., & Gupta, M.** (2020). *Deep k-NN for Noisy Labels*. arXiv:2004.12289v1. [논문 원문, 제공된 PDF]

### 논문 내 인용 문헌 (주요)
- Chaudhuri, K. & Dasgupta, S. (2014). *Rates of convergence for nearest neighbor classification*. NeurIPS.
- Hendrycks, D. et al. (2018). *Using trusted data to train deep networks on labels corrupted by severe noise*. NeurIPS.
- Patrini, G. et al. (2017). *Making deep neural networks robust to label noise: A loss correction approach*. CVPR.
- Li, Y. et al. (2017). *Learning from noisy labels with distillation*. ICCV.
- Rolnick, D. et al. (2017). *Deep learning is robust to massive label noise*. arXiv:1705.10694.
- Amid, E. et al. (2019). *Robust bi-tempered logistic loss based on Bregman divergences*. NeurIPS.
- Lee, K. et al. (2019). *Robust inference via generative classifiers for handling noisy labels*. arXiv:1901.11300.
- Jiang, H. (2019). *Non-asymptotic uniform rates of consistency for k-nn regression*. AAAI.
- Audibert, J.-Y. & Tsybakov, A.B. (2007). *Fast learning rates for plug-in classifiers*. The Annals of Statistics.
- Reeve, H.W. & Kaban, A. (2019). *Fast rates for a kNN classifier robust to unknown asymmetric label noise*. arXiv:1906.04542.
- Wilson, D. (1972). *Asymptotic properties of nearest neighbor rules using edited data*. IEEE Trans. on Systems, Man and Cybernetics.

### 2020년 이후 비교 참고 연구 (일반 지식 기반)
- Li, J. et al. (2020). *DivideMix: Learning with Noisy Labels as Semi-supervised Learning*. ICLR 2020.
- Liu, S. et al. (2020). *Early-Learning Regularization Prevents Memorization of Noisy Labels*. NeurIPS 2020.
- Xie, Q. et al. (2020). *Self-training with Noisy Student improves ImageNet classification*. CVPR 2020.

> ⚠️ **정확도 고지**: 2020년 이후 최신 연구 비교 분석 부분은 제 학습 데이터 기반으로 작성된 것으로, 정확한 수치나 실험 결과는 해당 원본 논문을 직접 확인하시기 바랍니다. 제공된 PDF의 내용에 대해서는 원문 그대로 인용하여 높은 정확도를 유지하였습니다.
