# Coresets for Robust Training of Deep Neural Networks against Noisy Labels

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 **CRUST (Coresets for Robust Training)** 라는 방법론을 제안하며, 노이즈 레이블(noisy labels)이 포함된 데이터로 딥 신경망을 훈련할 때 **이론적 보장(theoretical guarantees)** 을 제공하는 최초의 원칙적 접근법 중 하나입니다.

핵심 아이디어는 다음과 같습니다:
> **야코비안 행렬(Jacobian Matrix)이 근사적으로 저랭크(approximately low-rank) 구조를 갖는 클린 데이터 포인트의 가중 부분집합(coresets)을 선택함으로써, 그래디언트 하강법이 노이즈 레이블을 과적합하지 않도록 한다.**

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **이론적 보장** | 노이즈 레이블에 대한 강건한 훈련의 수학적 보장 제공 (Theorem 4.1) |
| **효율적 Coreset 선택** | Submodular 최적화를 통한 $O(nk)$ 복잡도의 효율적 알고리즘 |
| **Mixup 통합** | Coreset 오차 감소를 위한 Mixup 기법 결합 |
| **보조 모델 불필요** | 추가적인 클린 데이터셋이나 보조 모델 없이 작동 |
| **성능 향상** | CIFAR-10 (80% 노이즈)에서 6%, mini WebVision에서 7% 정확도 향상 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

딥 신경망은 **임의의 레이블(random labeling)도 완전히 암기(memorize)** 할 수 있는 용량을 가집니다. 따라서 실제 데이터셋에서 흔히 발생하는 노이즈 레이블은 일반화 성능에 심각한 영향을 미칩니다.

기존 방법들의 한계:
- **노이즈 전이 행렬 추정**: 추정 자체가 어렵고 부정확
- **강건한 손실 함수**: 최첨단 성능 달성 불가
- **레이블 수정**: 과적합에 취약
- **정규화/조기 종료**: 낮은 노이즈 수준(~20%)에서만 효과적
- **보조 모델 기반 방법**: 추가 클린 데이터셋 필요, 동일 분포 가정 필요

**핵심 문제**: 기존 기법들은 노이즈 레이블 환경에서의 성능에 대한 **이론적 보장을 제공하지 못함**.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 문제 설정

데이터셋 $\mathcal{D} = \{(\boldsymbol{x}\_i, y_i)\}_{i=1}^n \subset \mathbb{R}^d \times \mathbb{R}$ 에서 $L$ -레이어 완전연결 신경망 $f(\boldsymbol{W}, \boldsymbol{x})$를 훈련합니다.

손실 함수 (제곱 손실):

$$\mathcal{L}(\boldsymbol{W}) = \frac{1}{2} \sum_{i \in V} (y_i - f(\boldsymbol{W}, \boldsymbol{x}_i))^2 $$

그래디언트 하강 업데이트:

$$\boldsymbol{W}^{\tau+1} = \boldsymbol{W}^{\tau} - \eta \nabla \mathcal{L}(\boldsymbol{W}^{\tau}, \boldsymbol{X}), \quad \nabla \mathcal{L}(\boldsymbol{W}, \boldsymbol{X}) = \mathcal{J}^T(\boldsymbol{W}, \boldsymbol{X})(f(\boldsymbol{W}, \boldsymbol{X}) - \boldsymbol{y}) $$

야코비안 행렬 $\mathcal{J}(\boldsymbol{W}, \boldsymbol{X}) \in \mathbb{R}^{n \times m}$:

$$\mathcal{J}(\boldsymbol{W}, \boldsymbol{X}) = \left[\frac{\partial f(\boldsymbol{W}, \boldsymbol{x}_1)}{\partial \boldsymbol{W}} \cdots \frac{\partial f(\boldsymbol{W}, \boldsymbol{x}_n)}{\partial \boldsymbol{W}}\right]^T $$

#### 2.2.2 야코비안의 정보 공간과 노이즈 공간 분리

야코비안의 특이값(singular values) 분해를 통해 두 공간으로 분리합니다:
- **정보 공간 (Information Space)** $\mathcal{I}$: 큰 특이값과 연관된 부분 공간
- **노이즈 공간 (Nuisance Space)** $\mathcal{N}$: 작은 특이값과 연관된 부분 공간

보완 부분공간 $\mathcal{S}\_-, \mathcal{S}\_+ \subset \mathbb{R}^n$에 대해 모든 단위 벡터 $\boldsymbol{v} \in \mathcal{S}\_+$, $\boldsymbol{w} \in \mathcal{S}_-$:

$$\alpha \leq \|\mathcal{J}^T(\boldsymbol{W}, \boldsymbol{X})\boldsymbol{v}\|_2 \leq \beta, \quad \|\mathcal{J}^T(\boldsymbol{W}, \boldsymbol{X})\boldsymbol{w}\|_2 \leq \mu $$

여기서 $0 \leq \mu \ll \alpha \leq \beta$입니다.

**핵심 관찰**:
1. 정보 공간에서의 학습은 빠르고 일반화가 잘 됨
2. 노이즈 레이블의 잔차(residual)는 노이즈 공간에 정렬되어 일반화를 방해함

#### 2.2.3 최적 Coreset 선택 (이론적 목표)

저랭크 야코비안 근사를 위한 최적 부분집합:

$$S^*(\boldsymbol{W}) = \arg\min_{S \subseteq V} \|\mathcal{J}^T(\boldsymbol{W}, \boldsymbol{X}) - P_S \mathcal{J}^T(\boldsymbol{W}, \boldsymbol{X})\|_2 \quad \text{s.t.} \quad |S| \leq k $$

단, 계산 복잡도 $O(\text{poly}(n, m, k))$로 인해 직접 풀기 어려움.

#### 2.2.4 실용적 해법: k-Medoids 기반 접근

그래디언트 공간에서 대표적인 클린 데이터 포인트를 선택하는 $k$-medoids 문제:

$$S^*(\boldsymbol{W}) \in \arg\min_{S \subseteq V} \sum_{i \in V} \min_{j \in S} d_{ij}(\boldsymbol{W}) \quad \text{s.t.} \quad |S| \leq k $$

여기서 $d_{ij}(\boldsymbol{W}) = \|\nabla \mathcal{L}(\boldsymbol{W}, \boldsymbol{x}_i) - \nabla \mathcal{L}(\boldsymbol{W}, \boldsymbol{x}_j)\|_2$ 는 데이터 포인트 $i$와 $j$의 그래디언트 간 비유사도(pairwise gradient dissimilarity)입니다.

#### 2.2.5 효율적인 그래디언트 비유사도 상한 계산

전체 그래디언트 계산 비용을 줄이기 위해 마지막 레이어 그래디언트를 활용한 상한:

$$d_{ij}(\boldsymbol{W}) = \|\nabla\mathcal{L}(\boldsymbol{W}, \boldsymbol{x}_i) - \nabla\mathcal{L}(\boldsymbol{W}, \boldsymbol{x}_j)\|_2 \leq c_1 \|\Sigma'_L(\boldsymbol{z}_i^L)\nabla_i^L\mathcal{L} - \Sigma'_L(\boldsymbol{z}_j^L)\nabla_j^L\mathcal{L}\|_2 + c_2 $$

여기서 $\Sigma'_L(\boldsymbol{z}_i^L)\nabla_i^L\mathcal{L}$는 데이터 포인트 $i$에 대한 마지막 레이어 입력에 대한 손실의 그래디언트이고, $c_1, c_2$는 상수입니다.

이를 통해 $d_{ij}^u = \|\Sigma'_L(\boldsymbol{z}_i^L)\nabla_i^L\mathcal{L} - \Sigma'_L(\boldsymbol{z}_j^L)\nabla_j^L\mathcal{L}\|_2$를 효율적으로 계산합니다.

#### 2.2.6 Submodular 최대화로의 변환

문제 (6)은 다음 단조(monotone) submodular facility location 함수 최대화로 변환됩니다:

$$S^*(\boldsymbol{W}) \in \arg\max_{S \subseteq V, |S| \leq k} F(S, \boldsymbol{W}), \quad F(S, \boldsymbol{W}) = \sum_{i \in V} \max_{j \in S} d_0 - d'_{ij}(\boldsymbol{W}) $$

- $d_0$는 $d_0 \geq d_{ij}^u(\boldsymbol{W})$를 만족하는 상수
- 고전적 그리디 알고리즘으로 $(1-1/e)$-근사 보장
- 계산 복잡도: $O(nk)$, 역전파 불필요

#### 2.2.7 가중 야코비안 구성

각 메도이드 $j \in S^\*$에 클러스터 크기 $r_j = \sum_{i \in V} \mathbb{1}[j = \arg\min_{s \in S^*} d_{is}]$를 가중치로 부여:

```math
\mathcal{J}_r(\boldsymbol{W}, \boldsymbol{X}_{S^*}) = \text{diag}([r_1, \cdots, r_k])\mathcal{J}(\boldsymbol{W}, \boldsymbol{X}_{S^*}) \in \mathbb{R}^{k \times m}
```

가중 야코비안의 특이값 경계:

```math
\sqrt{r_{\min}}\sigma_{\min}(\mathcal{J}(\boldsymbol{W}, \boldsymbol{X}_{S^*}), \mathcal{S}_+) \leq \sigma_{i \in [k]}(\mathcal{J}_r(\boldsymbol{W}, \boldsymbol{X}_{S^*}), \mathcal{S}_+) \leq \sqrt{r_{\max}}\|\mathcal{J}(\boldsymbol{W}, \boldsymbol{X}_{S^*})\|
```


업데이트 규칙:

```math
\boldsymbol{W}^{\tau+1} = \boldsymbol{W}^{\tau} - \eta \mathcal{J}_r^T(\boldsymbol{W}, \boldsymbol{X}_{S^*})(f(\boldsymbol{W}, \boldsymbol{X}_{S^*}) - \boldsymbol{y}_{S^*})
```

#### 2.2.8 Mixup을 통한 추가 오차 감소

클러스터 $V_j$에서 소수 샘플 $R_j \subset V_j \setminus \{j\}$를 선택하여 메도이드 $(x_j, y_j)$와 혼합:

$$\hat{\mathcal{D}}_j = \{(\hat{x}, \hat{y}) \mid \hat{x} = \lambda x_i + (1-\lambda)x_j, \; \hat{y} = \lambda y_i + (1-\lambda)y_j \; \forall i \in R_j\} $$

여기서 $\lambda \sim \text{Beta}(\alpha, \alpha) \in [0, 1]$이고 $\alpha \in \mathbb{R}^+$입니다.

---

### 2.3 모델 구조 (CRUST 알고리즘)

```
Algorithm: CRUST
Input: 노이즈 데이터셋 D, 반복 횟수 T
Output: 모델 파라미터 W^T

for τ = 1, ..., T:
  1. 현재 모델 예측으로 데이터 포인트 분류 (클래스별 분리)
  2. 각 클래스 c에서:
     a. 상한 그래디언트 비유사도 d^u_ij 계산 (마지막 레이어 활용)
     b. Submodular 그리디로 k·n_c 개 메도이드 선택
     c. 각 메도이드에 클러스터 내 소수 샘플과 Mixup 적용
     d. 클러스터 크기로 가중치 설정
  3. 가중 그래디언트 하강으로 파라미터 업데이트
```

**주요 설계 결정사항:**
- **클래스별 분리**: 노이즈 데이터 포인트가 서로 클러스터링되는 것을 방지
- **예측 기반 분류**: 관측된 노이즈 레이블 대신 모델 예측 사용 (성능 향상)
- **반복적 Coreset 갱신**: 매 에포크마다 업데이트
- **Coreset 크기**: 데이터셋의 50% (노이즈 비율보다 작게 설정)

---

### 2.4 이론적 보장 (Theorem 4.1)

**Theorem 4.1**: 제곱 손실에 그래디언트 하강을 적용하고, 야코비안 매핑이 $L$-smooth이며, 데이터셋의 레이블 마진이 $\delta$이고, CRUST가 찾은 코어셋에 $\rho < \delta/8$ 비율의 노이즈 레이블이 포함되어 있다고 가정하자.

코어셋이 야코비안 행렬을 최대 $\epsilon$ 오차로 근사한다면:

$$\epsilon \leq \mathcal{O}\left(\frac{\delta\alpha^2}{k\beta\log(\sqrt{k}/\rho)}\right)$$

여기서 $\alpha = \sqrt{r_{\min}}\sigma_{\min}(\mathcal{J}(\boldsymbol{W}, \boldsymbol{X}_S))$, $\beta = \|\mathcal{J}(\boldsymbol{W}, \boldsymbol{X})\| + \epsilon$

그리고 $L \leq \frac{\alpha\beta}{L\sqrt{2k}}$, 스텝 크기 $\eta = \frac{1}{2\beta^2}$일 때,

$$\tau \geq \mathcal{O}\left(\frac{1}{\eta\alpha^2}\log\left(\frac{\sqrt{n}}{\rho}\right)\right)$$

번의 반복 후 선택된 모든 원소를 올바르게 분류함.

---

### 2.5 성능 향상

**CIFAR-10 결과 (ResNet-32):**

| 방법 | Sym 20% | Sym 50% | Sym 80% | Asym 40% |
|------|---------|---------|---------|---------|
| Co-teaching | 89.1 | 82.1 | 16.2 | 84.6 |
| INCV (최강 기준선) | 89.7 | 84.8 | 52.3 | 86.0 |
| **CRUST** | **91.1** | **86.3** | **58.3** | **88.8** |

**mini WebVision 결과 (InceptionResNet-v2):**

| 방법 | WebVision Top-1 | ImageNet Top-1 |
|------|-----------------|-----------------|
| INCV | 65.24 | 61.60 |
| **CRUST** | **72.40** | **67.36** |

**Ablation Study (CIFAR-10, Sym Noise):**

| Coreset (label) | Coreset (pred) | w/o mixup | w/ mixup | Noise 20% | Noise 50% |
|:-:|:-:|:-:|:-:|:-:|:-:|
| ✓ | | ✓ | | 90.21 | 84.92 |
| ✓ | | | ✓ | 90.48 | 85.23 |
| | ✓ | ✓ | | 90.71 | 85.57 |
| | ✓ | | ✓ | **91.12** | **86.27** |

---

### 2.6 한계점

1. **Coreset 크기 민감성**: 너무 작으면(30%) 정보 공간을 충분히 커버하지 못하고, 너무 크면(70%) 노이즈를 포함하여 과적합 발생
2. **이론과 실제의 간극**: 일반화 성능 향상에 대한 공식적 증명은 미래 작업으로 남겨둠 (논문에서 명시적으로 언급)
3. **전체 데이터 역전파 필요**: 그래디언트 비유사도 상한 계산을 위해 여전히 전체 데이터셋에 대한 순전파(forward pass)가 필요
4. **노이즈 비율 사전 지식**: 최적 coreset 크기 결정을 위해 노이즈 비율에 대한 어느 정도의 사전 지식이 유용함
5. **이론적 가정의 현실성**: 충분히 넓은 네트워크, Jacobian L-smooth 가정 등은 실제 환경과 다소 괴리가 있을 수 있음
6. **비대칭 노이즈(Asymmetric Noise) 취약성**: 대칭 노이즈(Symmetric Noise) 대비 비대칭 노이즈에서 성능 향상이 상대적으로 제한적

---

## 3. 일반화 성능 향상 가능성 (중점 분석)

### 3.1 야코비안 분석을 통한 일반화 메커니즘

일반화 성능과 야코비안 특이값 구조의 관계는 다음과 같이 이해할 수 있습니다:

분류 오차는 레이블 벡터 중 노이즈 공간에 정렬된 부분에 의해 제어됩니다:

$$\text{Classification Error} \propto \frac{\|\Pi_{\mathcal{N}}(\boldsymbol{y})\|}{\sqrt{n}}$$

여기서 $\Pi_{\mathcal{N}}(\boldsymbol{y})$는 $\boldsymbol{y}$의 노이즈 공간 $\mathcal{N}$으로의 투영입니다.

**CRUST의 일반화 향상 원리:**

CRUST가 선택한 크기 $k$의 코어셋 $S$에 대해:

$$\frac{\|\Pi_{\mathcal{N}}(\boldsymbol{y}_S)\|}{\sqrt{k}} \leq \frac{\|\Pi_{\mathcal{N}}(\boldsymbol{y})\|}{\sqrt{n}}$$

즉, 선택된 클린 코어셋의 레이블 벡터는 정보 공간 $\mathcal{I}$에 더 잘 정렬되어 있어 일반화 성능이 향상됩니다.

### 3.2 일반화 향상의 세 가지 메커니즘

#### 메커니즘 1: 노이즈 비율 감소
전체 데이터셋의 노이즈 비율 $\rho_{total}$에서 코어셋의 노이즈 비율 $\rho_{coreset} \ll \rho_{total}$로 감소:

- **실험 결과**: 80% 대칭 노이즈 환경에서도 코어셋은 훨씬 낮은 노이즈 비율을 유지 (Figure 1(a))
- 노이즈 레이블을 가진 데이터는 그래디언트 공간에서 퍼져(spread out) 메도이드로 선택되기 어려움

#### 메커니즘 2: 저랭크 야코비안 강제
정보 공간에 집중된 훈련으로 일반화 가능한 패턴만 학습:

```math
\sigma_{\min}(\mathcal{J}_r(\boldsymbol{W}, \boldsymbol{X}_{S^*}), \mathcal{S}_+) \geq \sqrt{r_{\min}} \cdot \sigma_{\min}(\mathcal{J}(\boldsymbol{W}, \boldsymbol{X}_{S^*}), \mathcal{S}_+) > 0
```

선택된 부분집합의 야코비안이 정보 공간에서 큰 최소 특이값을 가져 학습이 빠르고 일반화가 잘 됨.

#### 메커니즘 3: Mixup을 통한 분포 확장
Mixup은 두 가지 방식으로 일반화를 향상:

1. **노이즈 레이블 효과 완화**: 메도이드가 노이즈 레이블을 포함할 경우, 클러스터 내 다른 샘플과의 보간으로 노이즈 효과 희석
2. **결정 경계 평활화**: 볼록 결합(convex combination)으로 훈련되어 더 부드러운 결정 경계 형성

### 3.3 훈련 동역학 분석

코어셋 업데이트를 통한 반복적 노이즈 감소:

- **초기 단계**: 그래디언트가 더 균일하게 분포되어 일부 노이즈 포함 가능
- **중간 단계**: 모델이 클린 패턴 학습 → 예측 기반 분류가 정확해짐 → 코어셋 품질 향상
- **후기 단계**: 클린 데이터에 강하게 집중, 노이즈 과적합 방지

Figure 1(a)에서 확인: 레이블 정확도(label accuracy)가 훈련이 진행될수록 지속적으로 향상.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

#### 4.1.1 이론적 영향
1. **데이터 중심 AI(Data-Centric AI)의 이론적 기반 제공**: 모델 아키텍처가 아닌 데이터 선택을 통한 성능 향상의 이론적 정당성 확립
2. **야코비안 분석의 실용적 활용**: 훈련 동역학을 야코비안 특성으로 이해하는 새로운 패러다임 제시
3. **Coreset 이론의 딥러닝 확장**: 전통적인 코어셋 이론을 비선형 딥러닝에 적용하는 방법론 제시

#### 4.1.2 실용적 영향
1. **의료 AI, 자율주행 등 안전 중요 시스템**: 이론적 보장이 필요한 영역에서의 노이즈 레이블 문제 해결 방향 제시
2. **웹 크롤링 데이터 학습**: 대규모 비정형 데이터 학습에서의 노이즈 처리 방법론
3. **데이터 효율적 학습**: Coreset이 전체 데이터셋보다 작으므로 계산 효율성도 향상

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의**: 아래 최신 연구들에 대한 정보는 제가 사전 학습된 지식에 기반하며, 각 논문의 세부 수치나 결과에 대해 100% 확신하기 어려운 부분이 있습니다. 핵심 방향성과 비교 분석 위주로 서술합니다.

#### 4.2.1 관련 연구 흐름

**① Semi-supervised Learning 기반 접근 (DivideMix, 2020)**
- Li et al., "DivideMix: Learning with Noisy Labels as Semi-supervised Learning," ICLR 2020
- GMM(Gaussian Mixture Model)을 사용하여 클린/노이즈 데이터 분리
- Semi-supervised 학습 기법(MixMatch) 적용
- **CRUST와의 차이**: DivideMix는 레이블 확률 모델링 기반, CRUST는 그래디언트 공간 분석 기반
- **CRUST의 우위**: 이론적 보장 제공, 보조 클린 데이터 불필요

**② 대조 학습 기반 접근 (Sel-CL, 2022)**
- Li et al., "Selective-Supervised Contrastive Learning with Noisy Labels," CVPR 2022
- 대조 학습(contrastive learning)을 활용하여 노이즈에 강건한 표현 학습
- **CRUST와의 관계**: CRUST의 그래디언트 공간 클러스터링 아이디어를 표현 공간으로 확장한 방향으로 볼 수 있음

**③ LLM/Foundation Model 활용 (2022-2024)**
- 사전 훈련된 대형 언어 모델이나 비전 모델의 표현을 활용한 노이즈 레이블 탐지
- **CRUST와의 관계**: CRUST의 coreset 선택 프레임워크를 강력한 특징 공간에서 적용하는 방향으로 확장 가능

#### 4.2.2 비교 분석 표

| 방법론 | 이론적 보장 | 보조 데이터 필요 | 높은 노이즈 강건성 | 계산 효율 |
|--------|:-----------:|:----------------:|:-----------------:|:---------:|
| **CRUST** (2020) | ✓ (강함) | ✗ | ✓ (80%까지) | ✓ |
| DivideMix (2020) | ✗ | ✗ | ✓ | △ |
| MentorNet (2018) | ✗ | ✓ | △ | △ |
| Co-teaching (2018) | ✗ | ✗ | △ | △ |
| INCV (2019) | ✗ | ✗ | △ | ✗ |

### 4.3 앞으로 연구 시 고려할 점

#### 4.3.1 이론적 확장 방향

1. **공식적 일반화 오차 경계 증명**: 논문 자체에서 "formal generalization proof를 미래 작업으로 남긴다"고 명시. 이 갭을 메우는 연구 필요

   목표: 다음 형태의 일반화 경계 증명
   $$\mathbb{E}[\text{Test Error}] \leq f\left(\frac{\|\Pi_{\mathcal{N}}(\boldsymbol{y}_S)\|}{\sqrt{k}}, k, n, \delta\right)$$

2. **비선형 Jacobian 구조 분석**: 현재 이론은 훈련 중 Jacobian 변화를 충분히 다루지 못함. 동적 Jacobian 구조에 대한 이론 개발 필요

3. **노이즈 타입별 이론적 분석**: 현재는 주로 대칭적 노이즈(symmetric noise)에 대한 분석. 다양한 노이즈 모델(instance-dependent noise 등)에 대한 이론 확장 필요

#### 4.3.2 방법론적 개선 방향

1. **적응적 Coreset 크기 결정**: 현재는 노이즈 비율에 대한 사전 지식이 필요. 자동으로 최적 크기를 결정하는 방법 개발

2. **Transformer/ViT와의 결합**: CRUST는 주로 ResNet 기반으로 실험. 어텐션 메커니즘을 가진 모델에서의 야코비안 구조 분석 및 coreset 선택 방법 연구

3. **Foundation Model과의 통합**: 사전 훈련된 대형 모델의 표현 공간에서 더 효과적인 coreset 선택 가능성

4. **온라인/스트리밍 설정**: 논문에서 언급했지만 실험하지 않은 스트리밍 submodular 최대화를 실제로 구현하여 대규모 데이터셋 적용

5. **다른 도메인 확장**:
   - NLP 도메인의 노이즈 레이블 문제에 CRUST 적용
   - 멀티모달 데이터에서의 노이즈 처리

#### 4.3.3 실용적 고려사항

1. **노이즈 비율 추정**: 실제 환경에서는 노이즈 비율을 모르는 경우가 많음. Coreset 크기를 자동 조정하는 방법 필요

2. **클래스 불균형과의 상호작용**: 클래스별로 coreset을 선택하므로, 클래스 불균형이 있을 때의 동작 분석 필요

3. **분산 학습(Distributed Training) 호환성**: 대규모 분산 환경에서의 submodular 최적화 효율화

4. **Instance-dependent Noise**: 특정 샘플에 종속적인 노이즈 패턴에 대한 CRUST의 효과 연구

---

## 참고 자료

**주요 참고 논문 (논문 내 인용)**:
1. **본 논문**: Mirzasoleiman, B., Cao, K., & Leskovec, J. (2020). "Coresets for Robust Training of Neural Networks against Noisy Labels." *NeurIPS 2020*.
2. Oymak, S., & Soltanolkotabi, M. (2020). "Towards moderate overparameterization: global convergence guarantees for training shallow neural networks." *IEEE Journal on Selected Areas in Information Theory*.
3. Han, B., et al. (2018). "Co-teaching: Robust training of deep neural networks with extremely noisy labels." *NeurIPS 2018*.
4. Mirzasoleiman, B., Bilmes, J., & Leskovec, J. (2019). "Coresets for data-efficient training of machine learning models." *arXiv:1906.01827*.
5. Zhang, H., et al. (2017). "mixup: Beyond empirical risk minimization." *arXiv:1710.09412*.
6. Chen, P., et al. (2019). "Understanding and utilizing deep neural networks trained with noisy labels." *ICML 2019*. (INCV)
7. Li, M., Soltanolkotabi, M., & Oymak, S. (2019). "Gradient descent with early stopping is provably robust to label noise for overparameterized neural networks." *arXiv:1903.11680*.
8. Hu, W., Li, Z., & Yu, D. (2019). "Understanding generalization of deep neural networks trained with noisy labels." *arXiv:1905.11368*.

**GitHub 코드**: https://github.com/snap-stanford/crust

---

> **⚠️ 최종 고지**: 2020년 이후 최신 연구 비교 부분(DivideMix 이후 연구들)은 제 사전 학습 지식에 기반한 것으로, 구체적인 수치나 세부 결과의 정확성을 100% 보장하기 어렵습니다. 반드시 해당 논문 원문을 직접 확인하시기 바랍니다. CRUST 논문 자체의 내용은 제공된 PDF를 직접 기반으로 작성하였습니다.
