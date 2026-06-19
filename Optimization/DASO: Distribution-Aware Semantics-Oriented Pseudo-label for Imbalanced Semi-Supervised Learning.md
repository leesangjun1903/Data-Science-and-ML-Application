# DASO: Distribution-Aware Semantics-Oriented Pseudo-label for Imbalanced Semi-Supervised Learning

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

DASO는 **클래스 불균형(class imbalance)**과 **레이블/비레이블 데이터 간 분포 불일치(distribution mismatch)**라는 두 가지 문제로 인해 발생하는 pseudo-label의 편향(bias)을 완화하기 위한 반지도 학습(SSL) 프레임워크입니다.

기존 SSL 방법들은 다음의 근본적 문제를 가집니다:
- **선형 pseudo-label (Linear PL)**: 다수 클래스(head)로 편향
- **의미론적 pseudo-label (Semantic PL)**: 소수 클래스(tail)로 편향

DASO는 이 두 pseudo-label이 **상보적(complementary) 특성**을 가진다는 관찰에서 출발하여, 이를 분포 인식적으로 혼합(blend)합니다.

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| ① DASO Pseudo-label 프레임워크 | 두 종류의 pseudo-label을 클래스별로 적응적으로 혼합 |
| ② Semantic Alignment Loss | 균형 잡힌 피처 표현 학습을 위한 손실 함수 도입 |
| ③ 범용 프레임워크 | FixMatch, MixMatch, ReMixMatch 등 다양한 SSL 학습기에 통합 가능 |

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**문제 설정**: $K$ -class 반지도 학습에서 레이블 데이터 $\mathcal{X} = \{(x_n, y_n)\}\_{n=1}^{N}$와 비레이블 데이터 $\mathcal{U} = \{u_m\}_{m=1}^{M}$이 주어집니다.

**불균형 비율 정의**:

$$\gamma_l = \frac{\max_k N_k}{\min_k N_k} \gg 1, \quad \gamma_u = \frac{\max_k M_k}{\min_k M_k}$$

여기서 $\gamma_l$은 레이블 데이터의, $\gamma_u$는 비레이블 데이터의 불균형 비율입니다. **핵심 문제는 $\gamma_u$를 학습 중에 알 수 없다는 점**입니다.

**두 가지 핵심 문제**:
1. **클래스 불균형**: 다수 클래스로 편향된 pseudo-label → confirmation bias 심화
2. **분포 불일치** ($\gamma_l \neq \gamma_u$): 비레이블 데이터의 분포가 레이블과 다를 경우, 기존 방법(DARP, CReST+)의 가정이 위반됨

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 선형 Pseudo-label 생성

레이블 데이터의 feature $z^{(w)} = f_\theta^{\text{enc}}(\mathcal{A}_w(u))$로부터:

$$\hat{p} = \sigma\left(f_\phi^{\text{cls}}(z^{(w)})\right)$$

#### Step 2: 의미론적 Pseudo-label 생성 (Eq. 2)

균형 잡힌 프로토타입 집합 $\mathbf{C} = \{c_k\}_{k=1}^{K}$와의 코사인 유사도를 통해:

$$q = \sigma\left(\text{sim}(z, \mathbf{C}) \, / \, T_{\text{proto}}\right)$$

- $\text{sim}(\cdot, \cdot)$: 코사인 유사도
- $T_{\text{proto}}$: 온도 하이퍼파라미터
- 프로토타입 $c_k$는 클래스별 고정 크기 메모리 큐 $Q_k$ (크기 $L$로 균일화)에서 EMA 인코더 $f_{\theta'}^{\text{enc}}$로 추출된 feature의 평균으로 계산

**균형 프로토타입 생성 전략**:
- 큐 크기를 클래스별로 동일하게 고정 ($|Q_k| = L$, $\forall k$) → 소수 클래스 보정
- EMA 인코더 사용: $\theta' \leftarrow \rho\theta' + (1-\rho)\theta$ → 프로토타입 안정화

#### Step 3: 분포 인식 혼합 (Distribution-Aware Blending, Eq. 3)

$$\hat{p}' = (1 - \upsilon_{k'}) \hat{p} + \upsilon_{k'} \hat{q}$$

여기서 $k' = \arg\max_k \hat{p}_k$이고, 혼합 가중치 $\upsilon_k$는:

$$\upsilon_k = \frac{\hat{m}_k^{1/T_{\text{dist}}}}{\max_k \hat{m}_k^{1/T_{\text{dist}}}}$$

- $\hat{m}$: 현재 pseudo-label의 경험적 클래스 분포 (이전 몇 번의 iteration에서 $\hat{p}'$를 누적)
- $T_{\text{dist}}$: 혼합 강도를 조절하는 온도 파라미터

**직관**: 선형 PL이 head 클래스를 예측할수록 ($\hat{m}_k$가 클수록), $\upsilon_k$가 커져서 semantic PL을 더 많이 혼합 → 편향 교정

#### Step 4: Semantic Alignment Loss (Eq. 4)

$$\mathcal{L}_{\text{align}} = \mathcal{H}\left(\hat{q}, \, q^{(s)}\right)$$

여기서 $q^{(s)}$는 강하게 증강된 뷰 $z^{(s)} = f_\theta^{\text{enc}}(\mathcal{A}_s(u))$로부터 Eq. 2를 통해 계산됩니다.

**역할**: 동일한 비레이블 샘플의 두 뷰( $\mathcal{A}_w(u)$, $\mathcal{A}_s(u)$ )가 feature 공간에서 동일한 프로토타입에 일관되게 할당되도록 → 균형 잡힌 feature representation 학습

#### 최종 목적 함수 (Eq. 5)

$$\mathcal{L}_{\text{DASO}} = \mathcal{L}_{\text{cls}} + \lambda_u \mathcal{L}_{u} + \lambda_{\text{align}} \mathcal{L}_{\text{align}}$$

- $\mathcal{L}_{\text{cls}}$: 레이블 데이터에 대한 지도 손실 (cross-entropy)
- $\mathcal{L}_u$: $\hat{p}'$을 타겟으로 사용하는 비지도 손실
- $\mathcal{L}\_{\text{align}}$: semantic alignment 손실 ( $\lambda_{\text{align}} = 1$ )

---

### 2.3 모델 구조

```
입력
 ├─ 레이블 데이터 x
 │    └─ f_θ^enc (약한 증강) → z → f_φ^cls → L_cls
 │
 └─ 비레이블 데이터 u
      ├─ A_w(u): 약한 증강
      │    ├─ f_θ^enc → z^(w)
      │    │    ├─ f_φ^cls → Linear PL (p̂)
      │    │    └─ Similarity Classifier (C) → Semantic PL (q̂)
      │    └─ DASO Blend(p̂, q̂, T_dist) → DASO PL (p̂')
      │
      └─ A_s(u): 강한 증강
           ├─ f_θ^enc → z^(s)
           ├─ f_φ^cls → p^(s) → L_u (타겟: p̂')
           └─ Similarity Classifier (C) → q^(s) → L_align (타겟: q̂)

프로토타입 C: EMA 인코더 f_θ'^enc + 균일 크기 메모리 큐 Q_k
```

---

### 2.4 성능 향상

| 설정 | 방법 | CIFAR10-LT ($\gamma=100$, $N_1=500$) |
|------|------|--------------------------------------|
| - | FixMatch | 67.8% |
| $\gamma_l = \gamma_u$ | DARP | 74.5% |
| $\gamma_l = \gamma_u$ | CReST+ | 76.3% |
| $\gamma_l = \gamma_u$ | **DASO** | **76.0%** |
| $\gamma_u = 1$ (uniform) | FixMatch | 73.0% |
| $\gamma_u = 1$ (uniform) | CReST+ | 82.2% |
| $\gamma_u = 1$ (uniform) | **DASO** | **86.6%** |

**STL10-LT** ($\gamma_l=20$): FixMatch 대비 **+18.1%** 절대 성능 향상

**Semi-Aves** (대규모 실제 벤치마크): open-set 포함 시에도 최고 성능 유지 (47.9% vs FixMatch 46.1%)

**다른 SSL 학습기와의 통합 효과** (Table 3):
- MixMatch + DASO: $\gamma_u=1$에서 35.7% → 73.4% (+2.05×)
- ReMixMatch + DASO: $\gamma_u=1$에서 60.4% → 90.5% (+29.1% 절대 향상)

---

### 2.5 한계

1. **하이퍼파라미터 $T_{\text{dist}}$ 민감성**: 데이터셋과 분포에 따라 최적값이 달라짐 (CIFAR-10: 1.5 or 0.3, STL-10: 0.3, Semi-Aves: 0.5). 레이블이 부족한 환경에서 튜닝이 어려움
2. **비레이블 데이터를 완전한 비레이블로만 처리**: 비레이블 데이터의 분포 정보를 전혀 사용하지 않아, 만약 분포 정보가 일부 알려진 경우 이를 활용하는 방법과 통합이 필요
3. **open-set 클래스에 대한 명시적 처리 부재**: $\mathcal{U}_{\text{out}}$을 자연스럽게 낮은 신뢰도 영역으로 밀어내는 암묵적 효과는 확인되지만, 명시적 open-set detection은 미지원

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 편향 제거를 통한 일반화 향상

DASO의 일반화 성능 향상은 크게 세 가지 메커니즘에서 비롯됩니다.

#### (1) 분포 불가지론적(Distribution-Agnostic) 설계

기존 방법들은 $\gamma_u = \gamma_l$을 가정하거나 (DARP, CReST+), 비레이블 분포를 별도로 추정합니다. DASO는 **현재 pseudo-label의 경험적 분포 $\hat{m}$만을 사용**하여 가중치를 동적으로 조절:

$$\upsilon_k = \frac{\hat{m}_k^{1/T_{\text{dist}}}}{\max_k \hat{m}_k^{1/T_{\text{dist}}}}$$

이 설계 덕분에 $\gamma_u$가 알려지지 않은 실제 환경에서도 강건하게 동작합니다. 역전된 분포($\gamma_u = 1/100$)에서도 CReST+(62.9%)보다 DASO(71.0%)가 높은 성능을 보입니다.

#### (2) 균형 잡힌 Feature Representation

$\mathcal{L}_{\text{align}}$은 비레이블 데이터의 feature가 **레이블 공간의 프로토타입에 균형 있게 정렬**되도록 강제합니다. t-SNE 시각화에서 확인되듯이:
- FixMatch: tail 클래스(C8, C9)가 head 클래스 영역으로 산재
- DASO: tail 클래스가 명확히 분리된 클러스터 형성

이렇게 향상된 feature representation은:
- 선형 분류기의 편향 예측 감소에 기여
- 유사도 기반 분류기의 semantic pseudo-label 품질 향상에 재활용

#### (3) Recall-Precision 균형

| 방법 | 평균 Recall | 평균 Precision | 평균 정확도 |
|------|-------------|----------------|------------|
| FixMatch | 0.68 | 0.84 | 68.6% |
| USADTM | 0.74 | 0.57 | 72.3% |
| **DASO** | **0.79** | **0.76** | **76.3%** |

DASO는 소수 클래스의 recall을 높이면서 precision 저하를 최소화합니다. 이는 단순히 다수/소수 클래스 간의 트레이드오프가 아닌, **전체 클래스에 걸친 균형 잡힌 학습**을 가능하게 합니다.

#### (4) Open-Set 환경에서의 암묵적 강건성

Semi-Aves 실험 (E.5 분석)에서, DASO를 적용하면 $\mathcal{U}_{\text{out}}$ (open-set) 샘플들이 더 높은 엔트로피(낮은 신뢰도) 영역으로 이동합니다:
- FixMatch: $\approx 8k$개 out-of-class 샘플이 고신뢰도 영역
- DASO: $\approx 4k$개로 감소

이는 DASO가 open-set SSL로의 확장 가능성을 내포함을 시사합니다.

#### (5) 다양한 베이스라인과의 조합 가능성

DASO는 플러그인(plug-in) 방식으로 동작하여 일반화 성능 향상을 다양한 방향으로 확장합니다:

$$\text{FixMatch + LA + DASO}: 82.5\% \quad (\text{FixMatch}: 77.5\%, \text{LA만}: 82.0\%)$$
$$\text{FixMatch + ABC + DASO}: 80.1\% \quad (\text{FixMatch}: 67.8\%, \text{ABC만}: 78.9\%)$$

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 앞으로의 연구에 미치는 영향

#### (1) 불균형 SSL 연구의 패러다임 전환

DASO는 비레이블 데이터의 분포를 **사전에 알 필요 없이** 동적으로 편향을 교정하는 접근법을 제시합니다. 이는 실제 응용에서 더 현실적인 가정 하에 동작하는 SSL 방법론 연구를 촉진합니다.

#### (2) Pseudo-label 품질 지표로서 Recall-Precision 분석 프레임워크

DASO가 제안하는 **pseudo-label의 클래스별 recall/precision 분석**은 이후 연구에서 pseudo-label 품질을 평가하는 표준 분석 도구로 활용될 수 있습니다.

#### (3) 다중 분류기 앙상블 pseudo-labeling 방향 제시

선형 분류기와 유사도 기반 분류기의 상보적 특성을 활용한 접근은, **다수의 이질적 분류기를 결합**하는 방향의 연구 (예: 대조 학습 기반 분류기, 프로토타입 네트워크)에 영감을 줍니다.

#### (4) Open-Set SSL로의 확장

DASO의 암묵적 open-set 억압 효과는, 향후 **open-set class를 명시적으로 처리하는 SSL 프레임워크** 연구의 기초가 될 수 있습니다.

#### (5) 사회적 공정성(Fairness) 연구

논문이 언급하듯, 인구통계학적 불균형(성별, 인종 등)이 존재하는 데이터에서 분류기의 공정성을 향상시키는 연구에 DASO의 철학이 적용될 수 있습니다.

---

### 4.2 앞으로 연구 시 고려할 점

#### (1) 하이퍼파라미터 자동 적응 방법 개발

$T_{\text{dist}}$의 데이터 의존성은 레이블 부족 환경에서 심각한 문제입니다. 향후 연구에서는:
- **메타러닝(meta-learning)** 기반의 자동 $T_{\text{dist}}$ 설정
- **validation-free** 하이퍼파라미터 선택 방법
- 비레이블 데이터 분포 변화에 따른 온라인 적응 메커니즘

이 필요합니다.

#### (2) 더 다양한 분포 불일치 시나리오 검토

DASO는 $\gamma_u = 1$ (uniform)과 $\gamma_u = 1/100$ (역전)만을 극단 케이스로 실험했습니다. 실제 환경에서는:
- **점진적 분포 이동(gradual distribution shift)**
- **시간적 분포 변화(temporal distribution shift)**
- **도메인 갭이 있는 비레이블 데이터**

에 대한 연구가 필요합니다.

#### (3) 대규모 모델 및 파운데이션 모델과의 통합

ViT, CLIP 등 대규모 사전 학습 모델을 backbone으로 활용할 때 DASO의 프로토타입 기반 접근이 어떻게 동작하는지 검토가 필요합니다. 특히 **language-vision 모델의 semantic space를 프로토타입 구성에 활용**하는 방향이 유망합니다.

#### (4) Open-Set 및 Novel Class Discovery와의 결합

DASO는 비레이블 데이터의 open-set 샘플을 암묵적으로 억압하지만, 이를 **새로운 클래스 발견(novel class discovery)**에 활용하는 연구로 확장할 수 있습니다.

#### (5) 공정성(Fairness) 이슈 주의

논문이 지적하듯, DASO는 소수 클래스에 대한 과잉 보정(over-balance) 가능성이 있습니다. 공정성 지표(demographic parity, equalized odds 등)를 함께 고려한 설계가 필요합니다.

---

## 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 핵심 접근 | 비레이블 분포 가정 | DASO 대비 특이점 |
|------|-----------|-------------------|-----------------|
| **DARP** (NeurIPS 2020) | pseudo-label 분포를 레이블 분포로 정렬 (convex opt.) | $\gamma_u = \gamma_l$ 가정 | 분포 추정 필요, mismatch에 취약 |
| **CReST+** (CVPR 2021) | Self-training + Progressive Distribution Alignment | $\gamma_u = \gamma_l$ 가정 | 분포 역전 시 성능 크게 하락 |
| **ABC** (NeurIPS 2021 workshop) | Auxiliary Balanced Classifier로 균형 잡힌 분류 | 별도 가정 없음 | DASO와 상보적, 결합 시 최고 성능 |
| **DASO** (CVPR 2022) | 선형/의미론적 PL의 분포 인식 혼합 | **불필요** | 분포 불일치에 강건 |
| **Debiased SSL** (arXiv 2022) | 자연 불균형 pseudo-label에서 debiasing | - | zero-shot 및 SSL 통합 접근 |
| **SimMatch** (CVPR 2022) | 의미론적 유사도와 인스턴스 유사도 동시 활용 | - | 대조 학습 기반 |
| **SoftMatch** (ICLR 2023) | 신뢰도 임계값의 연성화(softening) | - | 균형 잡힌 샘플 활용 |

**핵심 비교 인사이트**:
- DARP, CReST+는 $\gamma_u = \gamma_l$ 가정이 깨질 경우 성능 저하가 심각
- DASO는 이러한 가정 없이도 일관된 성능 향상을 보여 **범용성(generality)** 측면에서 우위
- ABC와의 결합에서 DASO+ABC가 단독 방법보다 우수 → **보완적 관계**

---

## 참고 자료

- **주 논문**: Youngtaek Oh, Dong-Jin Kim, In So Kweon. "DASO: Distribution-Aware Semantics-Oriented Pseudo-label for Imbalanced Semi-Supervised Learning." CVPR 2022. arXiv:2106.05682v2
- Kihyuk Sohn et al. "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence." NeurIPS 2020.
- Jaehyung Kim et al. "Distribution Aligning Refinery of Pseudo-label for Imbalanced Semi-Supervised Learning (DARP)." NeurIPS 2020.
- Chen Wei et al. "CReST: A Class-Rebalancing Self-Training Framework for Imbalanced Semi-Supervised Learning." CVPR 2021.
- Hyuck Lee et al. "ABC: Auxiliary Balanced Classifier for Class-Imbalanced Semi-Supervised Learning." arXiv:2110.10368, 2021.
- Tao Han et al. "Unsupervised Semantic Aggregation and Deformable Template Matching for Semi-supervised Learning (USADTM)." NeurIPS 2020.
- Aditya Krishna Menon et al. "Long-tail Learning via Logit Adjustment." ICLR 2021.
- Xudong Wang et al. "Debiased Learning from Naturally Imbalanced Pseudo-Labels for Zero-Shot and Semi-Supervised Learning." arXiv:2201.01490, 2022.
- GitHub 코드: https://github.com/ytaek-oh/daso
