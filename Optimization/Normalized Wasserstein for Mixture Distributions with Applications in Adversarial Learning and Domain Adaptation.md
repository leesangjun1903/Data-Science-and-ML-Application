# Normalized Wasserstein for Mixture Distributions with Applications in Adversarial Learning and Domain Adaptation

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

기존 Wasserstein 거리는 혼합 분포(mixture distribution)에서 **혼합 비율(mixture proportions)의 불균형 문제**를 처리하지 못한다. 두 혼합 분포가 동일한 혼합 성분(components)을 가지더라도 혼합 비율이 다르면 Wasserstein 거리가 크게 측정되어 잘못된 결합(coupling)이 발생하며, 이는 도메인 적응, 생성 모델 등에서 심각한 성능 저하를 야기한다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **NW 측도 제안** | 혼합 비율을 최적화 변수로 도입한 새로운 거리 측도 |
| **이론적 분석** | 적절한 모드 수 $k^*$ 추정을 위한 이론적 조건 제시 (Theorem 1) |
| **다양한 응용** | GAN, 도메인 적응, 적대적 클러스터링에 적용 |
| **실험적 검증** | MNIST, MNIST-M, VISDA, CIFAR-10, CelebA 등 다양한 벤치마크에서 성능 향상 확인 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 기존 Wasserstein 거리의 한계

표준 Wasserstein 거리는 다음과 같이 정의된다:

$$W(\mathbb{P}_X, \mathbb{P}_Y) := \min_{\mathbb{P}_{X,Y}} \mathbb{E}[\|X - Y\|]$$

$$\text{subject to: } \text{marginal}_X(\mathbb{P}_{X,Y}) = \mathbb{P}_X, \quad \text{marginal}_Y(\mathbb{P}_{X,Y}) = \mathbb{P}_Y$$

**핵심 문제**: 주변 분포(marginal distribution) 제약 조건이 혼합 비율이 서로 다른 두 혼합 분포 사이에서 **잘못된 모드 간 결합(cross-mode coupling)**을 강제한다.

예시:
- 소스 도메인: 클래스 1이 80%, 클래스 2가 20%
- 타겟 도메인: 클래스 1이 20%, 클래스 2가 80%
- → 기존 Wasserstein은 전혀 다른 클래스의 샘플들을 연결시켜 **부정적 전이(negative transfer)** 발생

---

### 2.2 제안하는 방법 (수식 포함)

#### Normalized Wasserstein (NW) 측도 정의

$\mathbf{G} := [\mathbf{G}\_1, \ldots, \mathbf{G}\_k]$를 $k$개의 생성기(generator) 함수 배열, $\mathbb{P}_{\mathbf{G},\pi}$를 혼합 분포라 할 때:

$$\mathcal{P}_{\mathbf{G},k} := \{\mathbb{P}_{\mathbf{G},\pi} : \mathbf{G} \in \mathcal{G}, \pi \in \Pi\} \tag{2}$$

**NW 측도** $W_N(\mathbb{P}_X, \mathbb{P}_Y)$는 다음과 같이 정의된다:

$$\boxed{W_N(\mathbb{P}_X, \mathbb{P}_Y) := \min_{\mathbf{G}, \pi^{(1)}, \pi^{(2)}} W(\mathbb{P}_X, \mathbb{P}_{\mathbf{G},\pi^{(1)}}) + W(\mathbb{P}_Y, \mathbb{P}_{\mathbf{G},\pi^{(2)}})} \tag{3}$$

$$\text{subject to: } \sum_{j=1}^{k} \pi_j^{(i)} = 1 \quad i = 1, 2, \qquad \pi_j^{(i)} \geq 0 \quad 1 \leq j \leq k, \quad i = 1, 2$$

**핵심 아이디어 두 가지**:
1. 두 분포 $\mathbb{P}\_X$, $\mathbb{P}\_Y$ 사이에 **동일한 혼합 성분 $\mathbf{G}$를 가지는 중간 분포** $\mathbb{P}\_{\mathbf{G},\pi^{(1)}}$, $\mathbb{P}_{\mathbf{G},\pi^{(2)}}$를 생성
2. 혼합 비율 $\pi^{(1)}, \pi^{(2)}$를 **최적화 변수**로 취급하여 비율 불균형을 정규화

**중요한 성질**: $\mathbb{P}\_X = \mathbb{P}\_{\mathbf{G},\pi^{(1)}}$이고 $\mathbb{P}\_Y = \mathbb{P}_{\mathbf{G},\pi^{(2)}}$이면 (즉, 동일한 성분에 다른 비율), $W_N = 0$이 된다. 반면 기존 Wasserstein 거리는 여전히 크다.

> ⚠️ NW는 엄밀한 의미의 **거리(distance)가 아닌 반거리(semi-distance)**임 (삼각 부등식 등 일부 조건 미충족)

---

#### 도메인 적응에서의 수식 (지도 학습)

레이블 정보 $Y_s$를 활용하여 소스 혼합 성분을 다음과 같이 정의:

$$\mathbf{G}_i(Z) \overset{\text{dist}}{=} f(X_s^{(i)}), \quad X_s^{(i)} = \{X_s | Y_s = i\}, \quad \forall 1 \leq i \leq k \tag{5}$$

도메인 적응 최적화 문제:

$$\min_{f \in \mathcal{F}} \min_{\pi} \mathcal{L}_{cl}(X_s, Y_s) + \lambda W\left(\sum_i \pi^{(i)} f(X_s^{(i)}), f(X_t)\right) \tag{6}$$

비지도 학습 설정:

$$\min_{f \in \mathcal{F}} \mathcal{L}_{unsup}(X_s) + \lambda W_N(f(X_s), f(X_t)) \tag{7}$$

---

#### Normalized Wasserstein GAN (NWGAN)

$$\min_{\mathbf{G}, \pi} W_N(\mathbb{P}_X, \mathbb{P}_{\mathbf{G},\pi}) \tag{8}$$

NW 거리의 단순화:

$$\min_{\mathbf{G},\pi} W_N(\mathbb{P}_X, \mathbb{P}_{\mathbf{G},\pi}) = \min_{\mathbf{G},\pi} W(\mathbb{P}_X, \mathbb{P}_{\mathbf{G},\pi}) \tag{9}$$

---

#### 적대적 클러스터링

클러스터 할당:

$$C(\mathbf{x}_i) = \arg\min_{1 \leq j \leq k} \min_{Z} \left[\|\mathbf{x}_i - \mathbf{G}_j(Z)\|^2\right] \tag{10}$$

모드 다양성을 위한 정규화 항:

$$R = \sum_{(i,j)|i>j} \pi_i \pi_j W(\mathbf{G}_i(Z), \mathbf{G}_j(Z)) \tag{11}$$

정규화된 NWGAN 최적화:

$$\min_{\mathbf{G},\pi} W(\mathbb{P}_X, \mathbb{P}_{\mathbf{G},\pi}) - \lambda_{reg} R$$

---

### 2.3 모델 구조

```
[NW 측도 기반 모델 구조]

입력 분포 PX ─────────────────────────────┐
                                          ▼
                         중간 분포 P_{G, π^(1)} 구성
                         ┌────────────────────────────┐
                         │  G = [G_1, ..., G_k]       │
                         │  π^(1) = [π_1,...,π_k]     │  ← 최적화 변수
                         └────────────────────────────┘
                                    │
                           W(PX, P_{G,π^(1)}) 계산
                                    │
입력 분포 PY ─────────────────────────────┐
                                          ▼
                         중간 분포 P_{G, π^(2)} 구성
                         ┌────────────────────────────┐
                         │  동일한 G 공유              │
                         │  π^(2) = [π_1,...,π_k]     │  ← 독립적 최적화
                         └────────────────────────────┘
                                    │
                           W(PY, P_{G,π^(2)}) 계산
                                    │
                    NW = min[W(PX, P_{G,π^(1)}) + W(PY, P_{G,π^(2)})]
```

**학습 방법**: Wasserstein 거리의 쌍대(dual) 계산과 유사한 **교대 경사 하강법(alternating gradient descent)**, $\pi$ 제약 조건은 **소프트맥스(softmax)** 함수로 처리

---

### 2.4 이론적 분석 (Theorem 1)

가정 조건:
- **(A1)** 동일 성분의 두 모드 간 Wasserstein 거리 $\leq \epsilon$
- **(A2)** 서로 다른 모드 간 최소 Wasserstein 거리 $> \delta$ (모드 분리성)
- **(A3)** 각 모드의 밀도 $\geq \eta$ (최소 모드 비율 보장)
- **(A4)** 각 생성기 $\mathbf{G}_i$가 정확히 하나의 모드 포착 가능

**Theorem 1**: $\mathbb{P}_X$와 $\mathbb{P}_Y$가 각각 $n_1$, $n_2$개의 혼합 성분을 가지고 $r$개가 겹칠 때, 최적 모드 수:

$$k^* = n_1 + n_2 - r$$

$k^*$는 $NW(k)$가 $O(\epsilon)$으로 작으면서 $NW(k) - NW(k-1)$이 $O(\delta\eta)$로 상대적으로 큰 가장 작은 $k$이다.

---

### 2.5 성능 향상 결과

#### 도메인 적응 (MNIST → MNIST-M, 불균형 데이터)

| 방법 | 3 modes | 5 modes | 10 modes |
|------|---------|---------|----------|
| Source only | 66.63% | 67.44% | 63.17% |
| DANN | 62.34% | 57.56% | 59.31% |
| Wasserstein | 61.75% | 60.56% | 58.22% |
| **NW (제안)** | **75.06%** | **76.16%** | **68.57%** |

#### 도메인 적응 (VISDA, synthetic → real)

| 방법 | 정확도 |
|------|--------|
| Source only | 53.19% |
| DANN | 68.06% |
| Wasserstein | 64.84% |
| **Normalized Wasserstein** | **73.23%** |

#### 균형 데이터셋에서의 성능 (MNIST → MNIST-M 전체)

| 방법 | 정확도 |
|------|--------|
| Source only | 60.22% |
| DANN | 85.24% |
| Wasserstein | 83.47% |
| **Normalized Wasserstein** | 84.16% |

> ✅ **균형 데이터에서도 NW는 기존 방법과 비슷한 성능을 유지** (성능 저하 없음)

#### 클러스터링 (불균형 MNIST)

| 방법 | Cluster Purity | NMI | ARI |
|------|---------------|-----|-----|
| k-means | 0.82 | 0.49 | 0.43 |
| GMM | 0.75 | 0.28 | 0.33 |
| **NW** | **0.98** | **0.94** | **0.97** |

#### 이미지 노이즈 제거 ($err_{recons,tgt}$)

| 방법 | 재구성 오차 |
|------|------------|
| Source only | 0.31 |
| Wasserstein | 0.52 |
| **Normalized Wasserstein** | **0.18** |
| Training on target (Oracle) | 0.08 |

---

### 2.6 한계점

1. **Semi-distance**: NW 측도는 거리의 모든 공리(삼각 부등식 등)를 만족하지 않는 **반거리**
2. **모드 수 $k$ 선택 문제**: 실제 응용에서 $k$를 사전에 알아야 하거나 추정해야 함
3. **계산 복잡도**: $k$개의 생성기를 동시에 학습해야 하므로 단일 생성기 대비 계산 비용 증가
4. **대규모 데이터셋 한계**: 실험이 비교적 소규모 데이터셋(MNIST, VISDA 부분집합)에 한정
5. **레이블 의존성**: 지도 학습 설정에서는 소스 도메인 레이블 정보가 필요
6. **모드 붕괴 방지 정규화 필요**: 정규화 항 (Eq. 11) 없이는 한 생성기가 여러 모드를 포착할 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상의 핵심 메커니즘

NW 측도가 일반화 성능을 향상시키는 원리는 다음과 같다:

#### (1) 부정적 전이(Negative Transfer) 방지

기존 방법에서 불균형 혼합 비율은 다음과 같은 문제를 야기한다:

$$\text{기존: } \min_{f} \mathcal{L}_{cl}(X_s, Y_s) + \lambda W(\mathbb{P}_{f(X_s)}, \mathbb{P}_{f(X_t)})$$

이 경우, 도메인 간 클래스 비율 불균형이 있으면 **서로 다른 클래스의 특징이 가깝게 매핑**되어 분류기의 일반화 성능이 저하된다. NW는 이를 방지한다:

$$\min_{f, \pi} \mathcal{L}_{cl}(X_s, Y_s) + \lambda W\left(\sum_i \pi^{(i)} f(X_s^{(i)}), f(X_t)\right)$$

#### (2) 데이터 불균형에 강인한 표현 학습

$\pi$를 학습하는 과정은 사실상 **인스턴스 가중치(instance weighting)**의 역할을 수행한다:
- 소스 도메인에서 과대 표현된 클래스의 가중치를 자동으로 낮춤
- 타겟 도메인에서 희귀한 모드도 잘 포착

$$\pi_i \text{가 타겟의 실제 분포에 맞게 조정됨} \Rightarrow \text{더 균형 잡힌 특징 공간 학습}$$

#### (3) 희귀 모드(Rare Mode) 포착

NWGAN 실험에서 확인된 바와 같이, 혼합 비율 최적화는:

$$\pi_i = \frac{i}{45}, \quad 1 \leq i \leq 9 \quad \text{(매우 불균형)}$$

인 경우에도 MGAN($\pi$ 고정) 대비 $\pi$ 추정 오차를 $0.7157 \to 0.0001$로 극적으로 개선하여, 희귀 모드에 대한 일반화 능력을 향상시킨다.

#### (4) 이론적 일반화 보장

Theorem 1의 가정 하에서, NW 측도는:
- 모드 성분이 동일하면 0에 수렴 ( $O(\epsilon)$ )
- 모드 성분이 다르면 $O(\delta\eta)$만큼 차이 발생

이는 모드 성분 차이에 **민감**하면서 비율 차이에는 **불변(invariant)**한 특성을 이론적으로 보장하며, 이것이 일반화 성능 향상의 핵심이다.

#### (5) 균형/불균형 데이터 모두에서의 강인성

Table 3에서 보듯이, 균형 데이터에서도 NW (84.16%)는 Wasserstein (83.47%)와 비교해 성능 저하 없이 동등한 수준을 유지한다. 이는 NW가 균형 데이터에서 자동으로 균등 비율을 학습하기 때문이다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (1) 불균형 데이터 학습 패러다임 전환
- 기존의 오버샘플링(SMOTE 등)이나 손실 함수 가중치 방식 대신, **거리 측도 자체를 정규화**하는 새로운 접근법 제시
- 데이터 전처리 없이 알고리즘 수준에서 불균형 문제 해결 가능성 열어줌

#### (2) 최적 전송(Optimal Transport) 이론의 확장
- 기존 OT의 엄격한 주변 분포 제약을 완화하는 방향으로 OT 이론 확장에 기여
- Unbalanced OT, Partial OT 등과의 연결 가능성 제시

#### (3) 도메인 적응 연구에의 영향
- 실제 산업 데이터는 대부분 클래스 불균형이 존재하므로, NW 기반 도메인 적응이 실용적 가치 높음
- 2020년 이후 클래스-조건부(class-conditional) 도메인 적응 연구에 영향

#### (4) 공정성(Fairness) AI 연구
- 서브그룹 불균형 문제를 다루는 공정성 학습에 NW 측도 적용 가능성

### 4.2 향후 연구 시 고려할 점

#### 이론적 측면
- **NW의 완전한 metric 성질 확보**: 현재 semi-distance인 NW를 완전한 거리 측도로 확장하거나, semi-distance로서의 이론적 한계를 명확히 분석
- **수렴 보장**: NW 최적화의 수렴성 및 지역 최솟값 문제에 대한 이론적 분석 부재
- **샘플 복잡도**: 유한 샘플에서 NW 추정의 통계적 성질 분석 필요

#### 실용적 측면
- **대규모 적용 가능성**: ImageNet 수준의 대규모 데이터셋에서의 확장성 검증
- **모드 수 $k$ 자동 결정**: 실용적인 자동 $k$ 탐색 알고리즘 개발
- **계산 효율성**: $k$개 생성기의 동시 학습에 따른 계산 비용 절감 방안

#### 응용 측면
- **연속적 혼합 비율 변화**: 실제 환경에서 시간에 따라 비율이 변하는 비정상적(non-stationary) 분포 처리
- **다중 소스 도메인 적응**에의 확장
- **자기지도학습(Self-supervised learning)**과의 결합

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **중요 고지**: 아래의 2020년 이후 연구 비교는 제공된 PDF 문서에 포함되지 않은 내용이며, 본 논문(ICCV 2019)이 발표된 이후의 연구 동향에 대한 일반적 지식을 바탕으로 작성하였습니다. 개별 논문의 구체적 수치나 방법론 세부 사항은 원문 확인을 권장합니다.

### 5.1 관련 연구 흐름

| 연구 방향 | 대표 연구 | NW와의 관계 |
|-----------|-----------|------------|
| **Unbalanced OT** | Séjourné et al., "Unbalanced Optimal Transport: Dynamic and Kantorovich Formulations" (2019 이후 발전) | NW와 유사하게 주변 분포 제약 완화, KL 발산으로 소프트 제약 |
| **Class-conditional DA** | CGDM, ATDOC 등 (2020-2022) | NW의 클래스별 정렬 아이디어를 발전 |
| **Partial OT** | 부분적 최적 전송으로 모드 불균형 처리 | NW와 상보적 접근 |
| **Diffusion 기반 생성모델** | DDPM, Stable Diffusion (2020-2023) | 혼합 분포 생성에 다른 접근 방식 |
| **Imbalanced Domain Adaptation** | SDAT, SPA 등 | NW 아이디어를 다양한 방식으로 발전 |

### 5.2 NW 대비 후속 연구의 개선점 및 한계

**NW의 상대적 강점 유지 영역**:
- 명시적인 혼합 비율 최적화를 통한 해석 가능성(interpretability)
- 혼합 비율 추정 자체가 부산물로 제공됨

**후속 연구에서 개선된 부분**:
- Unbalanced OT는 더 유연한 소프트 제약으로 주변 분포 미스매치 처리
- 대규모 데이터셋에서의 확장성 개선 (Sliced Wasserstein 등 활용)
- 자기지도학습 표현과의 결합으로 더 강력한 도메인 불변 특징 학습

---

## 참고 문헌 및 출처

**본 답변의 주요 참고자료:**

1. **[주 논문]** Balaji, Y., Chellappa, R., & Feizi, S. (2019). "Normalized Wasserstein for Mixture Distributions with Applications in Adversarial Learning and Domain Adaptation." *ICCV 2019*, pp. 6500-6508.

2. **[논문 내 참조 문헌]**
   - Arjovsky, M., Chintala, S., & Bottou, L. (2017). "Wasserstein GAN." *arXiv:1701.07875*
   - Ganin, Y., & Lempitsky, V. (2015). "Unsupervised Domain Adaptation by Backpropagation." *ICML 2015*
   - Gulrajani, I. et al. (2017). "Improved Training of Wasserstein GANs." *arXiv:1704.00028*
   - Villani, C. (2008). *Optimal Transport: Old and New*. Springer.
   - Peng, X. et al. (2017). "VisDA: The Visual Domain Adaptation Challenge." *arXiv:1710.06924*
   - Goodfellow, I. et al. (2014). "Generative Adversarial Nets." *NeurIPS 2014*
   - Hoang, Q. et al. (2018). "MGAN: Training Generative Adversarial Nets with Multiple Generators."

3. **코드 저장소**: https://github.com/yogeshbalaji/Normalized-Wasserstein

> **정확도 고지**: 본 답변은 제공된 PDF 논문을 기반으로 작성되었으며, 논문 내용에 대한 분석은 100% 문서 기반입니다. 2020년 이후 비교 연구 부분은 일반적 연구 동향 지식을 바탕으로 하였으며, 구체적 후속 논문의 수치나 방법론은 원문 확인을 권장합니다.
