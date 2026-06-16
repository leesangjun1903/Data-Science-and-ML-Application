# FINE Samples for Learning with Noisy Labels

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

FINE(Filtering Noisy Instances via their Eigenvectors)은 기존 노이즈 레이블 탐지 방법들이 손실값(loss value)이나 그래디언트에 의존하여 **오염된 선형 분류기(corrupted linear classifier)의 편향**을 받는 문제를 해결하고자 합니다. 대신, **잠재 표현(latent representation)의 주성분 분석(eigen decomposition)**을 활용하여 클린 샘플과 노이즈 샘플을 분리합니다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| 이론적 기반의 노이즈 탐지기 | Gram matrix의 고유벡터를 활용한 파생불필요(derivative-free) 탐지기 제안 |
| 이론적 보장 | 클린 데이터의 고유벡터에 대한 섭동(perturbation)의 상한(upper bound) 증명 |
| 세 가지 적용 방식 | 샘플 선택, SSL, 노이즈 강인 손실함수와의 협업 |
| 일반화 성능 향상 | 다양한 벤치마크에서 기존 방법 대비 일관된 성능 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

현대의 심층 신경망(DNN)은 노이즈 레이블을 쉽게 암기(memorize)하는 경향이 있어 일반화 성능이 저하됩니다. 기존 noise-cleansing 방법들의 주요 한계는 다음과 같습니다:

- **손실값 기반 탐지기의 편향**: 노이즈 레이블에 오염된 분류기의 posterior 정보를 사용 → 탐지 품질 저하
- **노이즈율 추정 필요**: Co-teaching 계열은 사전에 노이즈율을 알아야 함
- **하이퍼파라미터 민감성**: TopoFilter의 k-NN 기반 방법은 심한 노이즈 환경에서 하이퍼파라미터에 민감

**Maennel et al.(2020)**의 분석에 따르면, 신경망 가중치의 주성분이 무작위 레이블 데이터와 정렬되는 현상이 있어 분류기 자체가 노이즈에 오염될 수 있습니다.

---

### 2.2 제안하는 방법 (FINE)

#### 2.2.1 핵심 아이디어

FINE은 각 클래스별 데이터의 **Gram matrix**를 구성하고, 이를 고유분해(eigen decomposition)하여 얻은 **첫 번째 고유벡터(principal eigenvector)**에 대한 각 표현벡터의 정렬(alignment) 정도로 클린/노이즈 샘플을 분리합니다.

#### 2.2.2 수식 설명

**Step 1: Gram Matrix 구성**

클래스 $k$에 속하는 샘플들의 표현벡터 $\mathbf{z}_i = g(\mathbf{x}_i)$ (penultimate layer output)로 Gram matrix를 구성합니다:

$$\Sigma_k = \sum_{(x_i, y_i) \in \mathcal{D}, y_i=k} \mathbf{z}_i \mathbf{z}_i^\top$$

**Step 2: 고유분해**

$$\mathbf{U}_k, \Lambda_k \leftarrow \text{EigenDecomposition}(\Sigma_k)$$

첫 번째 고유벡터: $\mathbf{u}_k \leftarrow \text{first column of } \mathbf{U}_k$

**Step 3: FINE Score 계산 (Alignment Score)**

각 샘플 $i$의 FINE score는 표현벡터와 클래스 대표 고유벡터 간의 내적의 제곱입니다:

$$f_i = \langle \mathbf{u}_{y_i}, \mathbf{z}_i \rangle^2$$

**Step 4: GMM 기반 분리**

각 클래스 $k$의 FINE score 집합 $\mathcal{F}_k$에 **Gaussian Mixture Model(GMM)**을 피팅하여, 클린 확률이 임계값 $\zeta$보다 큰 샘플을 클린 집합 $\mathcal{C}$로 선택합니다:

$$\mathcal{C} \leftarrow \mathcal{C} \cup \text{GMM}(\mathcal{F}_k, \zeta), \quad \forall k = 1, \ldots, K$$

**Eq. (2): 고유벡터 검증 수식**

클린 데이터의 FINE score를 극대화하고, 노이즈 데이터의 FINE score를 극소화하는 단위벡터 $\mathbf{a}$를 찾는 목적함수:

$$\frac{1}{|\mathbf{X}|} \sum_{x_i \in \mathbf{X}} \langle \mathbf{a}, x_i \rangle^2 - \frac{1}{|\tilde{\mathbf{X}}|} \sum_{x_j \in \tilde{\mathbf{X}}} \langle \mathbf{a}, x_j \rangle^2$$

실험 결과, FINE의 고유벡터 $\mathbf{u}$가 이 식을 거의 최대화함을 확인하였습니다.

---

### 2.3 이론적 보장 (Theorem 1)

**가정:**

- **Assumption 1**: 특징 분포는 클린 클러스터와 노이즈 클러스터 두 개의 Gaussian으로 구성됨
- **Assumption 2**: $y=+1$인 클린 인스턴스의 특징은 단위벡터 $\mathbf{v}$에 정렬됨 ($\mathbb{E}_{\mathbf{x} \in \mathbf{X}}[\mathbf{x}] = \mathbf{v}$), $y=-1$인 노이즈 인스턴스는 $\mathbf{w}$에 정렬됨

**Theorem 1 (클린 데이터 고유벡터에 대한 섭동 상한):**

$$\left\| \mathbf{u}\mathbf{u}^\top - \mathbf{v}\mathbf{v}^\top \right\|_2 \leq \frac{3\tau \cos\theta + \mathcal{O}\left(\sigma^2\sqrt{\frac{d+\log(4/\delta)}{N_+}}\right)}{1 - \tau(\sin\theta + 3\cos\theta) - \mathcal{O}\left(\sigma^2\sqrt{\frac{d+\log(4/\delta)}{N_+}}\right)}$$

여기서:
- $\tau = \frac{N_-}{N_+}$: 노이즈/클린 인스턴스 비율
- $\theta = \angle(\mathbf{w}, \mathbf{v})$: 클린과 노이즈 대표 벡터 간의 각도
- $\sigma^2$: 백색 잡음의 분산
- $N_+$: 클린 인스턴스 수

**해석:** $\tau$가 작을수록(클린 비율↑), $\theta$가 $\frac{\pi}{2}$에 가까울수록(클린-노이즈 표현이 직교할수록) 섭동이 작아져 FINE의 탐지 품질이 향상됩니다.

---

### 2.4 모델 구조

```
입력 데이터
    ↓
Feature Extractor g(·) [예: ResNet34]
    ↓
Penultimate Layer 표현벡터 z_i
    ↓
클래스별 Gram Matrix Σ_k 구성
    ↓
Eigen Decomposition → 첫 번째 고유벡터 u_k
    ↓
FINE Score: f_i = ⟨u_{y_i}, z_i⟩²
    ↓
GMM 피팅 → 클린/노이즈 분리
    ↓
┌─────────────────┬──────────────────┬─────────────────────┐
│ Sample Selection│   SSL 접근법      │ 노이즈 강인 손실함수  │
│ (FINE, F-Co-    │ (F-DivideMix)    │ (GCE, SCE, ELR)    │
│  teaching)      │                  │ 와의 협업            │
└─────────────────┴──────────────────┴─────────────────────┘
```

---

### 2.5 성능 향상

#### CIFAR-10/CIFAR-100 벤치마크 (샘플 선택 방법)

| 방법 | CIFAR-10 Sym 80% | CIFAR-10 Asym 40% | CIFAR-100 Sym 80% | CIFAR-100 Asym 40% |
|------|:---:|:---:|:---:|:---:|
| Co-teaching | 66.3 ± 1.5 | 88.4 ± 2.8 | 20.5 ± 1.3 | 47.7 ± 1.2 |
| TopoFilter | 46.8 ± 1.0 | 87.5 ± 0.4 | 18.3 ± 1.7 | 56.6 ± 0.5 |
| CRUST | 64.8 ± 1.5 | 82.4 ± 0.0 | 21.7 ± 0.7 | 56.1 ± 0.5 |
| **FINE** | **69.4 ± 1.1** | **89.5 ± 0.1** | **25.6 ± 1.2** | **61.7 ± 1.0** |
| **F-Coteaching** | **74.2 ± 0.8** | **90.5 ± 0.2** | **31.6 ± 1.0** | **64.8 ± 0.7** |

#### Clothing1M (실제 데이터셋)

| 방법 | 정확도 |
|------|:---:|
| DivideMix | 74.30 |
| **F-DivideMix** | **74.37** |
| FINE | 72.91 |

---

### 2.6 한계점

1. **이진 분류 가정**: Theorem 1은 이진 분류 태스크를 기반으로 하며, 다중 클래스 설정으로의 이론적 확장이 완전하지 않음
2. **Gaussian 분포 가정**: 클린/노이즈 분포가 Gaussian이라는 가정이 실제 복잡한 데이터에서 항상 성립하지 않을 수 있음
3. **Instance-dependent noise**: 인스턴스별 조건부 노이즈(IDN)에 대한 명시적 처리가 부족
4. **Warm-up 의존성**: 초기 특징 추출기 학습(warmup) 품질에 성능이 의존적
5. **계산 비용**: 대규모 데이터셋에서의 Gram matrix 계산 비용 (단, 1% 데이터로도 코사인 유사도 0.99 달성 가능함을 실험으로 확인)
6. **Asymmetric noise 취약성**: 일부 설정에서 비대칭 노이즈에 대한 강건성이 제한적

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상의 핵심 메커니즘

FINE이 모델의 일반화 성능을 향상시키는 핵심 이유는 **분류기 편향 없이 표현 공간의 기하학적 정보를 활용**하기 때문입니다.

**기존 방법의 문제점:**

Maennel et al.(2020)에 따르면, 노이즈 레이블로 학습된 신경망의 가중치 주성분이 노이즈 데이터와 정렬되어 분류기 자체가 오염됩니다. 이 오염된 분류기의 손실값을 탐지 기준으로 사용하면 탐지 편향이 발생합니다.

**FINE의 해결책:**

$$f_i = \langle \mathbf{u}_{y_i}, \mathbf{z}_i \rangle^2$$

FINE score는 **선형 분류기의 출력(logit)을 사용하지 않고** 잠재 표현의 주성분 방향을 기준으로 정렬 정도를 측정하므로, 분류기 오염의 영향을 받지 않습니다.

### 3.2 세 가지 적용에서의 일반화 향상

#### (1) 샘플 선택 접근법에서의 일반화

FINE은 각 에폭마다 클린 데이터를 재선별하여 학습에 사용합니다. 이 과정에서:

- 노이즈 샘플의 암기(memorization) 방지
- 클린 샘플만으로 학습하여 결정 경계(decision boundary)의 품질 향상
- F-score 동역학에서 학습이 진행될수록 탐지 품질이 향상됨을 확인

#### (2) SSL 접근법 (F-DivideMix)에서의 일반화

DivideMix의 손실값 기반 필터링을 FINE으로 대체했을 때:

- 극단적 노이즈(90% symmetric)에서 DivideMix(75.4%) → F-DivideMix(**89.6%**)로 대폭 향상
- 노이즈 샘플을 레이블 없는 데이터로 활용하여 SSL 기법(MixMatch 등)으로 학습 → 정보 손실 최소화

이는 FINE이 극단적 노이즈 환경에서 더 정확하게 클린/노이즈를 분리하여 **레이블 없는 데이터의 활용 품질**을 향상시키기 때문입니다.

#### (3) 노이즈 강인 손실함수와의 협업

GCE, SCE, ELR 등 노이즈 강인 손실함수에 FINE을 결합하면:

- CIFAR-10 Sym 80%: CE+FINE → 기존 대비 일관적 향상
- CIFAR-100 Asym 40%: 모든 손실함수 조합에서 향상

**FINE이 일반화에 기여하는 이유:** 노이즈 강인 손실함수는 노이즈의 영향을 줄이지만, 여전히 노이즈 샘플을 학습에 포함합니다. FINE이 클린 샘플을 먼저 걸러내면 손실함수가 더 신뢰할 수 있는 데이터로 학습되어 시너지 효과가 발생합니다.

### 3.3 확장성과 일반화

- **1% 데이터로도 코사인 유사도 0.99**: 적은 데이터로도 고유벡터를 정확히 추정 가능 → 대규모 데이터셋에서의 실용성
- **노이즈율 불필요**: 노이즈율을 사전에 알 필요 없어 실제 환경에서의 적용성 향상
- **실제 데이터셋(Clothing1M) 검증**: 38.5% 추정 노이즈율의 실제 데이터에서도 효과 확인

---

## 4. 최신 연구 비교 분석 (2020년 이후)

아래 표는 논문에서 직접 비교하거나 관련성이 높은 2020년 이후 연구들을 정리한 것입니다. **단, 논문 외부의 최신 연구(2022년 이후)에 대한 비교 수치는 제 학습 데이터의 한계로 인해 100% 정확성을 보장하기 어려우므로, 논문 내에서 직접 비교된 연구를 중심으로 서술합니다.**

### 4.1 논문 내 직접 비교된 2020년 이후 연구

| 연구 | 핵심 방법 | FINE 대비 비교 |
|------|-----------|----------------|
| **TopoFilter** (Wu et al., 2020) | k-NN 기반 위상학적 필터링, 유클리디안 거리 사용 | FINE이 F-score와 정확도에서 일관되게 우월; 특히 심한 노이즈에서 TopoFilter 성능 급락(CIFAR-10 Sym 80%: 46.8% vs FINE 69.4%) |
| **CRUST** (Mirzasoleiman et al., 2020) | 저랭크 Jacobian 기반 클린 서브셋 선택 | FINE이 대부분 설정에서 우월 |
| **ELR** (Liu et al., 2020) | 조기 학습 정규화로 노이즈 암기 방지 | FINE+ELR 결합 시 추가 향상 |
| **DivideMix** (Li et al., 2020) | 손실 분포 GMM 피팅 + MixMatch SSL | F-DivideMix가 극단 노이즈에서 대폭 향상(90% sym: 75.4% → 89.6%) |
| **LongReMix** (Cordeiro et al., 2021) | 고신뢰 샘플 활용 강인 학습 | F-DivideMix와 유사하거나 일부 설정에서 경쟁적 |
| **DST** (Wei et al., 2021) | 데이터 선택 및 공동 학습 | F-DivideMix와 경쟁적 |
| **CORES²** (Cheng et al., 2020) | 신뢰도 정규화 기반 점진적 필터링 | Clothing1M에서 F-DivideMix(74.37%) > CORES²(73.24%) |

### 4.2 방법론적 차별성

```
손실값 기반 방법         기하학적 방법          FINE
(DivideMix, ELR 등)    (TopoFilter 등)
        ↓                    ↓              ↓
분류기 오염 취약         하이퍼파라미터 민감    고유벡터 기반
노이즈율 필요            유클리디안 거리       이론적 보장
                                           노이즈율 불필요
```

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5.1 향후 연구에 미치는 영향

#### (1) 표현 공간 기하학의 재조명
FINE은 손실값 대신 **잠재 표현의 기하학적 구조**를 활용하여 노이즈를 탐지한다는 패러다임을 제시합니다. 이는 다음 연구 방향을 자극합니다:
- 더 정교한 표현 공간 분석 기법 (예: Riemannian geometry, manifold learning)
- 자기지도학습(Self-supervised learning) 표현과의 결합

#### (2) 이론적 기반 탐지기의 발전 촉진
Theorem 1의 섭동 상한 증명은 노이즈 탐지기의 **이론적 분석 프레임워크**를 제시합니다. 향후:
- 다중 클래스, instance-dependent noise에 대한 이론 확장 연구
- 더 타이트한 상한/하한 증명 연구

#### (3) 플러그인 모듈로서의 활용
FINE은 기존 LNL 방법에 샘플 선택 모듈을 대체하는 **플러그인 형태**로 사용 가능합니다. 이는 모듈화된 LNL 연구 방향을 촉진합니다.

#### (4) 이상 탐지 및 OOD 탐지로의 확장
논문 자체에서 언급하듯이, FINE의 원리는 anomaly detection, novelty detection, out-of-distribution detection에도 적용 가능한 범용적 프레임워크입니다.

#### (5) 실용적 응용 확대
웹 크롤링 데이터, 의료 데이터, 자율주행 데이터 등 **레이블 품질 검증이 어려운 실제 환경**에서의 적용 가능성이 높습니다.

---

### 5.2 향후 연구 시 고려 사항

#### (A) 이론적 한계 극복

**Instance-Dependent Noise(IDN) 대응:**
현재 FINE의 이론은 symmetric/asymmetric 노이즈를 주로 가정합니다. 실제 환경에서 흔한 인스턴스 의존적 노이즈에 대한 이론 확장이 필요합니다:

$$P(\tilde{y} \neq y | x, y) \neq P(\tilde{y} \neq y | y)$$

위와 같은 인스턴스 의존 조건 하에서의 FINE의 성능 분석이 필요합니다.

**다중 클래스 이론 확장:**
Theorem 1은 이진 분류 가정 하에 증명되었으므로, $K$-클래스 설정으로의 엄밀한 이론 확장이 필요합니다.

#### (B) 표현 학습 품질과의 상호작용

FINE의 성능은 특징 추출기 $g(\cdot)$의 품질에 크게 의존합니다. 다음을 고려해야 합니다:

- **초기 warmup 전략**: 노이즈 환경에서 초기 특징 추출기를 어떻게 학습할 것인가
- **자기지도학습 표현 활용**: SimCLR, BYOL 등 레이블 없이 학습된 표현과의 결합
- **표현 공간의 차원**: 고차원 표현에서의 고유벡터 추정 안정성

#### (C) 계산 효율성 최적화

대규모 데이터셋에서:

$$\Sigma_k = \sum_{i: y_i=k} \mathbf{z}_i \mathbf{z}_i^\top \in \mathbb{R}^{d \times d}$$

$d$가 클 경우 메모리/계산 비용이 증가합니다. 논문에서 1% 샘플로도 0.99 코사인 유사도를 달성함을 보였지만, **온라인 고유분해(online eigen decomposition)** 알고리즘이나 **랜덤 프로젝션** 기법과의 결합을 고려할 필요가 있습니다.

#### (D) 동적 노이즈 환경

실제 환경에서는 노이즈 패턴이 시간에 따라 변할 수 있습니다(예: 웹 크롤링 데이터). FINE을 **동적/온라인 학습 환경**에 적용하는 연구가 필요합니다.

#### (E) 클래스 불균형과의 상호작용

노이즈 레이블 문제는 클래스 불균형과 동시에 발생하는 경우가 많습니다. FINE이 클래스 불균형 상황에서 어떻게 동작하는지 분석이 필요합니다:

$$N_k^{\text{clean}} \ll N_{k'}^{\text{clean}}, \quad k \neq k'$$

#### (F) Foundation Model 및 대형 언어 모델과의 결합

최근의 대형 사전학습 모델(CLIP, GPT 등)의 표현은 더욱 의미론적으로 풍부합니다. 이러한 표현에서 FINE의 고유벡터가 어떤 특성을 갖는지 분석하고, **프롬프트 학습 환경에서의 노이즈 레이블 문제**에 FINE을 적용하는 연구가 유망합니다.

#### (G) 개인정보 보호 및 윤리적 고려

논문 자체에서 언급하듯이, 강인한 모델 학습 기술이 **불법 데이터 수집(다크웹 크롤링 등)**에 악용될 수 있습니다. 향후 연구에서는:
- 데이터 출처 검증 메커니즘과의 결합
- 연방 학습(Federated Learning) 환경에서의 적용 (프라이버시 보호)

---

## 참고 자료

### 주요 참고 논문 (본 논문에서 인용된 문헌)

1. **Kim, T., Ko, J., Cho, S., Choi, J., & Yun, S.-Y. (2021).** "FINE Samples for Learning with Noisy Labels." *Advances in Neural Information Processing Systems (NeurIPS 2021).*
   - GitHub: https://github.com/Kthyeon/FINE_official

2. **Han, B., et al. (2018).** "Co-teaching: Robust training of deep neural networks with extremely noisy labels." *NeurIPS.*

3. **Yu, X., et al. (2019).** "How does disagreement help generalization against label corruption?" *arXiv:1901.04215.*

4. **Li, J., Socher, R., & Hoi, S. C. H. (2020).** "DivideMix: Learning with noisy labels as semi-supervised learning." *ICLR.*

5. **Wu, P., et al. (2020).** "A topological filter for learning with label noise." *arXiv:2012.04835.*

6. **Mirzasoleiman, B., Cao, K., & Leskovec, J. (2020).** "Coresets for robust training of neural networks against noisy labels." *arXiv:2011.07451.*

7. **Liu, S., et al. (2020).** "Early-learning regularization prevents memorization of noisy labels." *arXiv:2007.00151.*

8. **Maennel, H., et al. (2020).** "What do neural networks learn when trained with random labels?" *arXiv:2006.10455.*

9. **Lee, K., et al. (2019).** "Robust inference via generative classifiers for handling noisy labels." *ICML.*

10. **Zhang, Z. & Sabuncu, M. R. (2018).** "Generalized cross entropy loss for training deep neural networks with noisy labels." *arXiv:1805.07836.*

11. **Wang, Y., et al. (2019).** "Symmetric cross entropy for robust learning with noisy labels." *ICCV.*

12. **Cordeiro, F. R., et al. (2021).** "LongReMix: Robust learning with high confidence samples in a noisy label environment." *arXiv:2103.04173.*

13. **Wei, Y., et al. (2021).** "DST: Data selection and joint training for learning with noisy labels." *arXiv:2103.00813.*

14. **Cheng, H., et al. (2020).** "Learning with instance-dependent label noise: A sample sieve approach." *arXiv:2010.02347.*

15. **Zhang, C., et al. (2016).** "Understanding deep learning requires rethinking generalization." *arXiv:1611.03530.*

> **주의사항**: 2022년 이후의 관련 최신 연구(예: SOP, NGC, ProMix 등)에 대한 정량적 비교는 본 논문(NeurIPS 2021)의 범위를 벗어나며, 해당 연구들과의 직접 비교 수치를 확인할 수 없어 서술하지 않았습니다. 이 부분은 추가 문헌 검색을 권장합니다.
