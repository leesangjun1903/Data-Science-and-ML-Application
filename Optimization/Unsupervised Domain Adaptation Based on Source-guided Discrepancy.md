# Unsupervised Domain Adaptation Based on Source-guided Discrepancy

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
이 논문은 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)에서 소스 도메인과 타겟 도메인 간의 차이를 측정하는 새로운 불일치 척도인 **Source-guided Discrepancy (S-disc)**를 제안합니다. 기존의 불일치 척도들이 높은 계산 비용을 요구하거나 이론적 보장이 없는 문제를 해결하기 위해, **소스 도메인의 레이블 정보를 활용**하여 효율적이고 이론적으로 더 타이트한 일반화 오차 경계를 제공하는 척도를 설계합니다.

### 주요 기여 (4가지)
| 기여 | 내용 |
|------|------|
| ① 새로운 척도 제안 | S-disc (Definition 2): 소스 레이블을 활용한 도메인 불일치 측정 |
| ② 효율적 추정 알고리즘 | 0-1 손실에 대한 비용 민감 분류로 환원 (Algorithm 1) |
| ③ 통계적 보장 | 일관성(Consistency) 및 수렴률 $\mathcal{O}_p(n_T^{-1/2} + n_S^{-1/2})$ 증명 (Theorem 4) |
| ④ 더 타이트한 일반화 경계 | 기존 $\mathcal{X}$-disc 기반 경계보다 항상 더 좁은 경계 (Theorem 7, 8) |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**비지도 도메인 적응(UDA)**의 문제 설정은 다음과 같습니다:
- **소스 도메인**: 레이블이 있는 데이터 $\mathcal{S} = \{(x_j^S, y_j^S)\}_{j=1}^{n_S}$
- **타겟 도메인**: 레이블이 없는 데이터 $\mathcal{T} = \{x_i^T\}_{i=1}^{n_T}$
- **목표**: 타겟 도메인에서 기대 손실 $R_T^\ell(h, f_T) = \mathbb{E}_{x \sim P_T}[\ell(h(x), f_T(x))]$을 최소화하는 가설 $h$ 탐색

**기존 방법들의 한계:**

**$\mathcal{X}$-disc** (Mansour et al., 2009a):

$$\text{disc}_\mathcal{H}^\ell(P_T, P_S) = \sup_{h, h' \in \mathcal{H}} \left| R_T^\ell(h, h') - R_S^\ell(h, h') \right| \tag{1}$$

- 최악의 가설 쌍(worst-case pair)을 고려 → **느슨한 경계**, **높은 계산 비용** $O((n_T + n_S + d)^8)$

**$d_\mathcal{H}$** (Ben-David et al., 2007):
$$d_\mathcal{H}(P_T, P_S) = \sup_{h \in \mathcal{H}} \left| R_T^{\ell_{01}}(h, 1) - R_S^{\ell_{01}}(h, 1) \right|$$
- 계산은 효율적이나 **이론적 일반화 보장 없음**

**$\mathcal{Y}$-disc** (Zhang et al., 2012; Mohri & Medina, 2012):

$$\mathcal{Y}\text{-disc}_\mathcal{H}^\ell(P_T, P_S) = \sup_{h \in \mathcal{H}} |R_T^\ell(h, f_T) - R_S^\ell(h, f_S)|$$

- 타겟 도메인의 레이블링 함수 $f_T$ 필요 → **비지도 적응에 사용 불가**

---

### 2.2 제안하는 방법 (S-disc) 및 수식

#### S-disc 정의

**Definition 1 (Source-guided Discrepancy):**

```math
\varsigma_\mathcal{H}^\ell(P_{D_1}, P_{D_2}) = \sup_{h \in \mathcal{H}} \left| R_{D_1}^\ell(h, h_S^*) - R_{D_2}^\ell(h, h_S^*) \right|
```

여기서 $h_S^* = \arg\min_{h \in \mathcal{H}} R_S^\ell(h, f_S)$는 소스 도메인에서의 진정한 위험 최소화자(true risk minimizer)입니다.

**핵심 아이디어**: $\mathcal{X}$-disc에서 최적화되던 $h'$를 소스 도메인의 최적 가설 $h_S^*$로 **고정**합니다. 이로 인해:
- 최적화 변수가 단일 $h$로 줄어들어 계산 효율성 향상
- 소스 레이블 정보를 간접적으로 활용하여 더 정확한 도메인 차이 측정

**S-disc와 기존 척도의 관계:**

```math
\left| R_T^\ell(h, h_S^*) - R_S^\ell(h, h_S^*) \right| \leq \varsigma_\mathcal{H}^\ell(P_T, P_S) \leq \text{disc}_\mathcal{H}^\ell(P_T, P_S)
```

S-disc는 $\mathcal{X}$ -disc보다 항상 작거나 같으므로 더 타이트한 경계를 제공합니다.

---

#### 0-1 손실에 대한 S-disc 추정 (Theorem 2)

대칭 가설 클래스 $\mathcal{H}$에 대해 다음 등식이 성립합니다:

$$\varsigma_\mathcal{H}^{\ell_{01}}(\hat{P}_T, \hat{P}_S) = 1 - \min_{h \in \mathcal{H}} J^{\ell_{01}}(h) \tag{Theorem 2}$$

여기서:

```math
J^\ell(h) = \frac{1}{n_S} \sum_{j=1}^{n_S} \ell(h(x_j^S), h_S^*(x_j^S)) + \frac{1}{n_T} \sum_{i=1}^{n_T} \ell(h(x_i^T), -h_S^*(x_i^T))
```

이 수식의 의미: S-disc 추정이 **비용 민감 분류(cost-sensitive classification)** 문제로 환원됩니다.

---

#### Algorithm 1: S-disc 추정 알고리즘

```
입력: 레이블된 소스 데이터 S, 레이블 없는 타겟 데이터 T, 대체 손실 ℓ_sur, 가설 클래스 H
출력: ς_H(P̂_T, P̂_S)

Step 1 [소스 학습]: S_X를 이용해 분류기 ĥ_S 학습
Step 2 [의사 레이블링]:
   - S̃ = {(x, sign∘ĥ_S(x)) | x ∈ S_X}
   - T̃ = {(x, -sign∘ĥ_S(x)) | x ∈ T}
Step 3 [비용 민감 학습]: S̃와 T̃를 이용해 J^ℓ_sur 최소화하는 h'' 학습
반환: ς_H(P̂_T, P̂_S) = 1 - J^ℓ_01(h'')
```

**계산 복잡도**: SMO 알고리즘을 사용하면 $O((n_T + n_S)^3)$ — $\mathcal{X}$-disc의 $O((n_T + n_S + d)^8)$ 대비 압도적으로 효율적

---

### 2.3 모델 구조

S-disc의 추정 구조는 3단계 파이프라인으로 구성됩니다:

```
[소스 데이터 (레이블 있음)]         [타겟 데이터 (레이블 없음)]
         ↓                                    ↓
   Step 1: SVM 학습                    Step 2: 의사 레이블 생성
   ĥ_S = argmin R̂_S^ℓ(h, f_S)        T̃ = {(x, -sign∘ĥ_S(x))}
         ↓                                    ↓
              [Step 3: 비용 민감 SVM]
              minimize J^ℓ_sur(h) on S̃ ∪ T̃
                        ↓
              S-disc = 1 - J^ℓ_01(h'')
```

**특징**: 두 번의 SVM 학습으로 완성되는 간단한 구조

---

### 2.4 이론적 분석

#### 일관성 및 수렴률 (Theorem 4)

손실 함수 $\ell$이 $M > 0$으로 상계될 때, 임의의 $\delta \in (0,1)$에 대해 최소 $1-\delta$ 확률로:

$$\left| \varsigma_\mathcal{H}^\ell(\hat{P}_T, \hat{P}_S) - \varsigma_\mathcal{H}^\ell(P_T, P_S) \right|$$
$$\leq 2\mathfrak{R}_{P_T, n_T}(\ell \circ (\mathcal{H} \otimes \mathcal{H})) + 2\mathfrak{R}_{P_S, n_S}(\ell \circ (\mathcal{H} \otimes \mathcal{H})) + M\sqrt{\frac{\log \frac{4}{\delta}}{2n_T}} + M\sqrt{\frac{\log \frac{4}{\delta}}{2n_S}}$$

**Corollary 6 (0-1 손실에 대한 수렴률):** 가정 (4) 하에서:

$$\left| \varsigma_\mathcal{H}^\ell(\hat{P}_T, \hat{P}_S) - \varsigma_\mathcal{H}^\ell(P_T, P_S) \right| \leq \frac{\mathfrak{C}_{\mathcal{H} \otimes \mathcal{H}}}{\sqrt{n_T}} + \frac{\mathfrak{C}_{\mathcal{H} \otimes \mathcal{H}}}{\sqrt{n_S}} + \sqrt{\frac{\log \frac{4}{\delta}}{2n_T}} + \sqrt{\frac{\log \frac{4}{\delta}}{2n_S}}$$

수렴률: $\mathcal{O}_p(n_T^{-1/2} + n_S^{-1/2})$

---

#### 일반화 오차 경계 (Theorem 7)

손실 $\ell$이 삼각 부등식을 만족할 때 (예: 0-1 손실), 임의의 $h \in \mathcal{H}$에 대해:

```math
R_T^\ell(h, f_T) - R_T^\ell(h_T^*, f_T) \leq \underbrace{R_S^\ell(h, h_S^*)}_{\text{(i) 소스 경험 손실}} + \underbrace{R_T^\ell(h_S^*, h_T^*)}_{\text{(ii) 도메인 간 최적 가설 차이}} + \underbrace{\varsigma_\mathcal{H}^\ell(P_T, P_S)}_{\text{(iii) S-disc}}
```

**비교**: $\mathcal{X}$ -disc 기반 경계 (Mansour et al., 2009a, Theorem 8):

```math
R_T^\ell(h, f_T) - R_T^\ell(h_T^*, f_T) \leq R_S^\ell(h, h_S^*) + R_T^\ell(h_T^*, h_S^*) + \text{disc}_\mathcal{H}^\ell(P_T, P_S)
```

부등식 (3)에 의해 $\varsigma_\mathcal{H}^\ell(P_T, P_S) \leq \text{disc}_\mathcal{H}^\ell(P_T, P_S)$이므로, **S-disc 기반 경계가 항상 더 타이트**합니다.

#### 유한 샘플 일반화 경계 (Theorem 8)

$\ell = \ell_{01}$일 때, 임의의 $h \in \mathcal{H}$, $\delta \in (0,1)$에 대해 최소 $1-\delta$ 확률로:

```math
R_T^\ell(h, f_T) - R_T^\ell(h_T^*, f_T) \leq \hat{R}_S^\ell(h, h_S^*) + R_T^\ell(h_S^*, h_T^*) + \varsigma_\mathcal{H}^\ell(\hat{P}_T, \hat{P}_S)
```

$$+ \frac{\mathfrak{C}_{\mathcal{H} \otimes \mathcal{H}}}{\sqrt{n_T}} + \frac{\mathfrak{C}_{\mathcal{H} \otimes \mathcal{H}}}{\sqrt{n_S}} + \sqrt{\frac{\log\frac{5}{\delta}}{2n_T}} + 2\sqrt{\frac{\log\frac{5}{\delta}}{2n_S}}$$

---

### 2.5 성능 향상 및 한계

#### 실험 결과

**실험 1 (장남감 예제 - 소스 선택 정확성):**

| 척도 | $S_1$ vs Target | $S_2$ vs Target | 선택한 소스 | 실제 최적 소스 |
|------|---------|---------|------------|--------------|
| S-disc | 0.27 | 0.49 | $S_1$ ✓ | $S_1$ |
| $d_\mathcal{H}$ | 0.69 | 0.49 | $S_2$ ✗ | $S_1$ |

$S_1$으로 훈련한 분류기의 타겟 손실 = 0.0, $S_2$로 훈련한 분류기의 타겟 손실 = 0.49

**실험 2 (계산 시간 비교):**
- S-disc: ~ $10^{-2}$초
- $d_\mathcal{H}$: ~ $10^{-1}$초  
- $\mathcal{X}$-disc: ~ $10^4$초 (사실상 사용 불가)

**실험 3 (소스 선택 - MNIST vs MNIST-M):** S-disc는 샘플 수가 증가할수록 노이즈 소스를 효과적으로 필터링하지만, $d_\mathcal{H}$는 항상 1을 반환하여 구별 불가

#### 한계점

1. **이진 분류에 집중**: 현재 0-1 손실에 대한 효율적 추정 알고리즘이 주로 이진 분류에 특화
2. **소스 최적 가설 추정 오차**: $h_S^*$를 $\hat{h}_S$로 대체하는 과정에서 추정 오차 발생 가능
3. **선형-파라미터 모델 가정**: Rademacher 복잡도 분석이 특정 가설 클래스 구조에 의존
4. **딥러닝 확장 미검증**: 신경망 기반 특징 추출과의 통합이 실험적으로 검증되지 않음
5. **멀티클래스 확장 미제공**: 다중 클래스 분류로의 일반화 이론적 분석 부재

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상의 핵심 메커니즘

S-disc 기반 일반화 경계 (Theorem 7)를 분석하면:

```math
R_T^\ell(h, f_T) - R_T^\ell(h_T^*, f_T) \leq \underbrace{R_S^\ell(h, h_S^*)}_{\text{(i)}} + \underbrace{R_T^\ell(h_S^*, h_T^*)}_{\text{(ii)}} + \underbrace{\varsigma_\mathcal{H}^\ell(P_T, P_S)}_{\text{(iii)}}
```

**각 항목의 의미와 최소화 전략:**

| 항 | 의미 | 최소화 방법 |
|----|------|------------|
| (i) $R_S^\ell(h, h_S^*)$ | 소스에서 최적 가설과의 경험 손실 | 소스 ERM 수행 |
| (ii) $R_T^\ell(h_S^\*, h_T^*)$ | 두 도메인 간 최적 가설의 불일치 | 좋은 소스 선택 |
| (iii) $\varsigma_\mathcal{H}^\ell(P_T, P_S)$ | S-disc (도메인 간 분포 차이) | S-disc 기반 소스 선택/가중치 |

**핵심 통찰**: 소스와 타겟 도메인이 충분히 가깝다면 (ii)와 (iii)이 작아지므로, **소스 도메인에서 경험 손실을 최소화하는 것이 타겟 일반화 향상으로 직결**됩니다.

### 3.2 소스 선택을 통한 일반화 향상

Theorem 8에 의하면 $n_S, n_T \to \infty$ 시 지배적 항:

```math
R_T^\ell(h, f_T) - R_T^\ell(h_T^*, f_T) \approx \hat{R}_S^\ell(h, h_S^*) + R_T^\ell(h_S^*, h_T^*) + \varsigma_\mathcal{H}^\ell(\hat{P}_T, \hat{P}_S)
```

**S-disc를 소스 선택 기준으로 사용**하면, $\varsigma_\mathcal{H}^\ell$이 작은 소스를 선택함으로써 타겟 일반화 성능이 향상됩니다. $h_T^\*$가 $h_S^*$에 충분히 가깝다면, S-disc 기준으로 좋은 소스를 선택하는 것이 좋은 타겟 일반화를 보장합니다.

### 3.3 Rademacher 복잡도를 통한 분포 의존적 분석

기존 VC-차원 기반 분석보다 Rademacher 복잡도 기반 분석이 더 타이트한 이유:

$$\mathfrak{R}_{\mu, m}(\mathcal{H}) = \mathbb{E}_{x_1, \ldots, x_m} \mathbb{E}_\sigma \left[ \sup_{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^m \sigma_i h(x_i) \right]$$

- **분포 의존적**: 실제 데이터 분포 $\mu$에 따라 복잡도가 결정됨
- **덜 비관적**: VC-차원은 최악의 경우를 고려하지만, Rademacher 복잡도는 실제 분포에서의 복잡도 반영

선형-파라미터 모델 $\mathcal{H} = \{x \mapsto w^\top \phi(x) \mid \|w\|\_2 \leq \Lambda\}$에서:

$$\mathfrak{R}_{\mu, m}(\mathcal{H} \otimes \mathcal{H}) \leq \frac{\Lambda^2 D_\phi^2}{\sqrt{m}}$$

이는 수렴률이 $\mathcal{O}(m^{-1/2})$임을 보장합니다 (Lemma 5).

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

#### ① 이론적 측면
- **더 타이트한 도메인 차이 측정 패러다임**: "레이블 정보를 불일치 척도에 통합"하는 아이디어는 후속 연구들에게 새로운 방향성을 제시합니다. 특히 일부 레이블이 있는 준지도 도메인 적응(Semi-supervised DA)으로 확장 가능성이 높습니다.
- **Rademacher 복잡도 기반 분석의 표준화**: 기존 VC-차원 기반 분석보다 더 세밀한 이론적 보장 제공 방식이 후속 연구의 분석 도구로 채택될 가능성

#### ② 응용 측면
- **소스 선택(Source Selection)**: 다중 소스 환경에서 S-disc를 이용한 최적 소스 선택 전략이 실용적으로 활용될 수 있음
- **커리큘럼 학습 연계**: S-disc가 작은 소스-타겟 쌍을 먼저 학습하는 커리큘럼 전략에 응용 가능

#### ③ 딥러닝 도메인 적응으로의 확장
- 딥러닝 기반 도메인 적응에서 적대적 훈련(Adversarial Training)의 이론적 근거로 S-disc가 활용될 수 있음
- Domain-Adversarial Neural Networks (DANN) 등의 방법에 S-disc를 통합한 훈련 목적 함수 설계 가능

### 4.2 향후 연구 시 고려할 점

#### ① 딥러닝 환경으로의 확장
현재 S-disc는 선형-파라미터 모델과 커널 SVM 기반으로 검증되었습니다. 딥러닝 환경에서:
- **표현 학습(Representation Learning)과의 결합**: 특징 추출 네트워크와 S-disc를 통합한 end-to-end 훈련 방식 개발 필요
- **미니배치 추정**: 대규모 데이터셋에서 S-disc의 미니배치 기반 효율적 추정 알고리즘 개발

#### ② 다중 클래스 및 회귀로의 확장
- 논문의 이진 분류 제한을 다중 클래스 설정으로 일반화할 때 $h_S^*$의 역할 재정의 필요
- 회귀 태스크에서 S-disc의 이론적 분석 확장

#### ③ $R_T^\ell(h_S^\*, h_T^*)$ 항의 처리
일반화 경계의 두 번째 항 $R_T^\ell(h_S^\*, h_T^*)$는 타겟 레이블 없이 직접 최소화하기 어렵습니다:
- 이 항을 제어하기 위한 보조 정보 활용 방법 연구 필요
- 자기 지도 학습(Self-supervised Learning)을 통해 타겟 도메인 구조를 활용하는 방향 고려

#### ④ 적대적 견고성(Adversarial Robustness)과의 연계
도메인 시프트와 적대적 공격 모두 분포 변화를 유발하므로, S-disc를 견고한 모델 훈련에 활용하는 연구 가능

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 후속 연구 동향

#### ① 이론적 발전: 정보 이론적 경계

**Zhao et al. (2019), "On Learning Invariant Representations for Domain Adaptation"** (NeurIPS 2019)은 도메인 불변 표현 학습의 이론적 한계를 지적했습니다. 그들은 레이블 이동(label shift)이 있을 때 도메인 불변 표현이 오히려 일반화를 해칠 수 있음을 보였습니다:

$$R_T(h) \leq R_S(h) + d_{\mathcal{H}\Delta\mathcal{H}}(P_S, P_T) + \lambda^*$$

여기서 $\lambda^* = \min_{h \in \mathcal{H}} [R_S(h) + R_T(h)]$가 핵심 하한입니다. 이는 S-disc가 소스 레이블을 활용함으로써 $\lambda^*$에 해당하는 항을 더 잘 포착할 수 있음을 시사합니다.

#### ② 딥러닝 기반 도메인 적응과 이론적 결합

**Saito et al. (2018), "Maximum Classifier Discrepancy for Unsupervised Domain Adaptation"** (CVPR 2018)은 두 분류기 사이의 불일치를 최대화/최소화하는 적대적 훈련을 제안했습니다. 이 방법은 S-disc의 소스 가이드 아이디어와 유사하게 **두 분류기의 예측 불일치**를 활용하지만, 이론적 보장이 부족합니다.

| 비교 항목 | S-disc (본 논문) | MCD (Saito et al., 2018) |
|----------|-----------------|--------------------------|
| 이론적 보장 | ✓ (일반화 경계) | △ (경험적 검증) |
| 딥러닝 통합 | △ (SVM 기반) | ✓ (신경망) |
| 소스 레이블 활용 | ✓ | ✓ |
| 계산 효율성 | $O((n_T+n_S)^3)$ | 미니배치 SGD |

#### ③ 분포 정렬과 최적 수송(Optimal Transport)

**Damodaran et al. (2018), "DeepJDOT: Deep Joint Distribution Optimal Transport"** 및 후속 연구들은 Wasserstein 거리 기반 도메인 정렬을 딥러닝과 결합했습니다. 이들은 S-disc와 달리:
- 소스와 타겟의 **결합 분포(joint distribution)** 정렬 추구
- 레이블 정보를 비용 함수에 통합

S-disc의 소스 가이드 아이디어는 이러한 결합 분포 정렬 방법의 이론적 근거로 확장될 수 있습니다.

#### ④ 준지도 도메인 적응(Semi-supervised DA)

**Li et al. (2021), "Semi-supervised Domain Adaptation"** 계열 연구들은 타겟 도메인의 소량 레이블을 활용하는 방향으로 발전했습니다. S-disc의 $h_S^*$ 대신 **소스+타겟 결합 최적 가설**을 사용하는 확장을 고려해볼 수 있습니다:

```math
\tilde{\varsigma}_\mathcal{H}^\ell(P_T, P_S) = \sup_{h \in \mathcal{H}} \left| R_T^\ell(h, \tilde{h}^*) - R_S^\ell(h, \tilde{h}^*) \right|
```

여기서 $\tilde{h}^*$는 소스와 타겟 레이블 데이터 모두에서 학습된 가설.

#### ⑤ 테스트 타임 적응(Test-Time Adaptation, TTA)

2020년 이후 급부상한 TTA 연구들 (예: **Wang et al. (2021), "Tent: Fully Test-Time Adaptation by Entropy Minimization"**)은 타겟 도메인의 테스트 시점에서 실시간으로 적응합니다. 이 설정에서 S-disc는:
- 모델 선택 기준으로 활용 가능
- 언제 적응이 필요한지 판단하는 "적응 트리거"로 사용 가능

### 5.2 비교 테이블

| 방법 | 이론적 보장 | 레이블 활용 | 딥러닝 통합 | 계산 효율성 | 소스 선택 |
|------|------------|------------|------------|------------|----------|
| S-disc (본 논문, 2018) | ✓ 타이트한 경계 | 소스만 | △ SVM | $O((n_T+n_S)^3)$ | ✓ |
| $\mathcal{X}$-disc (Mansour et al., 2009) | ✓ 넓은 경계 | 없음 | △ | $O((n_T+n_S+d)^8)$ | △ |
| MCD (Saito et al., 2018) | △ | 소스만 | ✓ | SGD | △ |
| DANN (Ganin et al., 2016) | △ | 소스만 | ✓ | SGD | ✗ |
| DeepJDOT (2018) | △ | 소스만 | ✓ | 중간 | △ |
| Tent (Wang et al., 2021) | ✗ | 없음 | ✓ | 높음 | ✗ |

---

## 참고 자료

**주요 논문 (PDF 제공):**
- Kuroki, S., Charoenphakdee, N., Bao, H., Honda, J., Sato, I., & Sugiyama, M. (2019). **"Unsupervised Domain Adaptation Based on Source-guided Discrepancy"**. AAAI 2019. arXiv:1809.03839v3

**논문 내 인용 참고문헌:**
- Ben-David, S., Blitzer, J., Crammer, K., & Pereira, F. (2007). Analysis of representations for domain adaptation. *NIPS*.
- Mansour, Y., Mohri, M., & Rostamizadeh, A. (2009a). Domain adaptation: Learning bounds and algorithms. *COLT*.
- Mohri, M., & Medina, A. M. (2012). New analysis and algorithm for learning with drifting distributions. *ALT*.
- Zhang, C., Zhang, L., & Ye, J. (2012). Generalization bounds for domain adaptation. *NIPS*.
- Saito, K., Watanabe, K., Ushiku, Y., & Harada, T. (2018). Maximum classifier discrepancy for unsupervised domain adaptation. *CVPR*.
- Ganin, Y., et al. (2016). Domain-adversarial training of neural networks. *JMLR*.
- Bartlett, P. L., & Mendelson, S. (2002). Rademacher and Gaussian complexities: Risk bounds and structural results. *JMLR*.
- Mohri, M., Rostamizadeh, A., & Talwalkar, A. (2012). *Foundations of Machine Learning*. MIT Press.

**비교 분석을 위한 2020년 이후 관련 연구 (논문 제목):**
- Zhao, H., et al. (2019). "On Learning Invariant Representations for Domain Adaptation". *NeurIPS 2019*.
- Wang, D., et al. (2021). "Tent: Fully Test-Time Adaptation by Entropy Minimization". *ICLR 2021*.

> **⚠️ 정확도 주의사항**: 비교 분석 섹션의 2020년 이후 논문들(특히 TTA, DeepJDOT 등)의 세부 수치와 일부 참고 내용은 해당 논문 원문 PDF가 제공되지 않아, 논문명과 일반적으로 알려진 내용을 기반으로 기술했습니다. 정확한 수치 비교를 위해서는 해당 원문 논문을 직접 참조하시기 바랍니다.
