# ABC: Auxiliary Balanced Classifier for Class-Imbalanced Semi-Supervised Learning

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 반지도학습(SSL) 알고리즘은 클래스 균형 데이터셋을 가정하지만, 현실의 많은 데이터셋은 클래스 불균형(Class Imbalance)을 가진다. 클래스 불균형 상황에서 SSL 알고리즘을 그대로 사용하면, 편향된 예측이 레이블 없는 데이터의 학습에 활용되어 편향이 더욱 심화된다. 이 논문은 **보조 균형 분류기(Auxiliary Balanced Classifier, ABC)**를 제안하여 이 문제를 해결한다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **ABC 설계** | 기존 SSL 백본의 표현층에 단일 레이어 보조 분류기를 부착 |
| **0/1 마스킹 기법** | Bernoulli 분포 기반의 확률적 마스킹으로 클래스 균형 손실 구성 |
| **수정된 일관성 정규화** | 언레이블 데이터에 대한 클래스 균형 일관성 정규화 도입 |
| **엔드-투-엔드 학습** | 표현 학습과 분류기 학습의 동시 최적화 |
| **확장성** | 대규모 데이터셋(LSUN)에도 적용 가능한 경량 설계 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**클래스 불균형 반지도학습(Class-Imbalanced SSL, CISSL)** 문제:

- 기존 SSL 알고리즘(FixMatch, ReMixMatch 등)은 클래스 균형을 가정
- 클래스 불균형 상황에서는 다수 클래스(majority class)로 편향된 예측이 발생
- 이 편향된 예측이 언레이블 데이터 학습에 다시 사용되어 **편향이 자기강화(self-reinforcing)**됨
- 기존의 클래스 불균형 학습(CIL) 기법들은 레이블 정보가 필요하여 SSL에 직접 적용 불가

**문제 설정:**

- 레이블 데이터: $\mathcal{X} = \{(x_n, y_n) : n \in (1, \ldots, N)\}$
- 언레이블 데이터: $\mathcal{U} = \{u_m : m \in (1, \ldots, M)\}$
- 레이블 데이터 비율: $\beta = \frac{N}{M+N}$, 일반적으로 $\beta < 0.5$
- 클래스 불균형 비율: $\gamma = \frac{N_1}{N_L}$ (가장 많은 클래스 수 / 가장 적은 클래스 수), $\gamma \gg 1$

---

### 2.2 제안하는 방법 및 수식

#### (A) 백본 SSL 알고리즘

FixMatch 또는 ReMixMatch를 백본으로 사용:

- **약한 증강(weak augmentation)**: $\alpha(x_b)$ — 이미지 플립, 크롭
- **강한 증강(strong augmentation)**: $\mathcal{A}(x_b)$ — Cutout, RandAugment

백본의 분류층 직전 **표현층(representation layer)**에 ABC를 부착.

> 핵심 근거: 분류 알고리즘은 분류기(classifier)가 편향되어 있더라도 **고품질 표현(representation)**은 학습 가능 [Kang et al., 2020]

---

#### (B) 분류 손실 (Classification Loss) — 수식 (1), (2)

레이블 데이터에 대해 Bernoulli 마스크를 활용한 클래스 균형 분류 손실:

$$L_{cls} = \frac{1}{B} \sum_{b=1}^{B} M(x_b) \mathbf{H}(p_s(y|\alpha(x_b)), p_b) \tag{1}$$

$$M(x_b) = \mathcal{B}\left(\frac{N_L}{N_{y_b}}\right) \tag{2}$$

여기서:
- $\mathbf{H}$: 표준 교차 엔트로피 손실(cross-entropy loss)
- $\alpha(x_b)$: 약하게 증강된 레이블 데이터
- $p_s(y|\alpha(x_b))$: ABC의 예측 클래스 분포
- $p_b$: $x_b$에 대한 원-핫(one-hot) 레이블
- $\mathcal{B}(\cdot)$: Bernoulli 분포
- $N_L$: 가장 적은 클래스의 샘플 수
- $N_{y_b}$: $x_b$가 속한 클래스의 샘플 수

**해석:** 소수 클래스 데이터는 높은 확률로 마스크 값 1을 받아 학습에 포함되고, 다수 클래스 데이터는 낮은 확률로 마스크 값 1을 받아 사실상 언더샘플링 효과를 낸다. 그러나 백본은 전체 미니배치로 학습하므로 **정보 손실 없이 오버피팅도 방지**.

---

#### (C) 일관성 정규화 손실 (Consistency Regularization Loss) — 수식 (3), (4)

언레이블 데이터에 대해 클래스 균형 일관성 정규화:

$$L_{con} = \frac{1}{B} \sum_{b=1}^{B} \sum_{k=1}^{2} M(u_b) \mathbf{I}(\max(q_b) \geq \tau) \mathbf{H}(p_s(y|\mathcal{A}_k(u_b)), q_b) \tag{3}$$

$$M(u_b) = \mathcal{B}\left(\frac{N_L}{N_{\hat{q}_b}}\right) \tag{4}$$

여기서:
- $q_b = p_s(y|\alpha(u_b))$: 약하게 증강된 언레이블 데이터에 대한 소프트 의사 레이블(soft pseudo-label)
- $\mathcal{A}_1(u_b), \mathcal{A}_2(u_b)$: 두 개의 강하게 증강된 버전
- $\mathbf{I}(\cdot)$: 지시 함수(indicator function)
- $\tau$: 신뢰도 임계값(confidence threshold, $\tau = 0.95$)
- $\hat{q}_b$: $q_b$에서 얻은 원-핫 의사 레이블

**특징:**
- FixMatch와 달리 **소프트 의사 레이블** 사용 (하드 레이블의 엔트로피 최소화는 편향 가속 위험)
- 소프트 의사 레이블 기반 마스크로 **언레이블 데이터도 리샘플링 가능**
- 학습 초기에는 Bernoulli 파라미터를 1에서 $N_L/N_{\hat{q}_b}$로 **점진적으로 감소** (초기 부정확한 의사 레이블 과도한 리샘플링 방지)

---

#### (D) 총 손실 함수 — 수식 (5)

$$L_{total} = L_{cls} + L_{con} + L_{back} \tag{5}$$

- $L_{back}$: 백본 SSL 알고리즘의 손실 (FixMatch 또는 ReMixMatch의 기존 손실)
- **학습 시**: $L_{total}$ 전체를 사용하여 백본과 ABC를 **엔드-투-엔드** 동시 학습
- **추론 시**: ABC만 사용하여 예측 수행

---

### 2.3 모델 구조

```
입력 이미지
    │
    ▼
[Deep CNN (Wide ResNet-28-2)]
    │
    ▼
[표현층 (Representation Layer)]
    │            │
    ▼            ▼
[백본 분류기]   [ABC (단일 선형층)]
(편향됨)        (클래스 균형)
    │            │
    ▼            ▼
L_back         L_cls + L_con
    │            │
    └────────────┘
           │
       L_total (학습)
           │
    추론 시: ABC만 사용
```

**핵심 설계 원칙:**
- ABC는 단 **1개의 선형 레이어**로 구성 (CIFAR-10: 백본 파라미터의 0.09%, CIFAR-100: 0.87% 추가)
- 백본의 표현층을 공유하므로 메모리 및 연산 비용 최소화
- 0/1 마스크를 사용하여 백본과 ABC가 동일한 미니배치에서 학습 가능 → 표현 재사용, 연산 효율화

---

### 2.4 성능 향상

**주요 실험 결과 (Table 1, 메인 설정):**

| 알고리즘 | CIFAR-10-LT (전체/소수) | SVHN-LT (전체/소수) | CIFAR-100-LT (전체/소수) |
|---|---|---|---|
| FixMatch | 72.3 / 53.8 | 88.0 / 79.4 | 51.0 / 32.8 |
| FixMatch+DARP+cRT | 78.1 / 66.6 | 89.9 / 83.5 | 54.7 / 41.2 |
| **FixMatch+ABC** | **81.1 / 72.0** | **92.0 / 87.9** | **56.3 / 43.4** |
| ReMixMatch | 73.7 / 55.9 | 89.8 / 82.8 | 54.0 / 37.1 |
| ReMixMatch+DARP+cRT | 78.5 / 66.4 | 92.1 / 87.6 | 55.1 / 43.6 |
| **ReMixMatch+ABC** | **82.4 / 75.7** | **93.9 / 92.5** | **57.6 / 46.7** |

**애블레이션 스터디 결과 (Table 5, ReMixMatch+ABC, CIFAR-10-LT):**

| 조건 | 전체 | 소수 |
|---|---|---|
| 전체 제안 알고리즘 | **82.4** | **75.7** |
| 일관성 정규화 없음 | 79.4 | 66.9 |
| $L_{con}$에 0/1 마스크 없음 | 79.0 | 69.2 |
| $L_{cls}$에 0/1 마스크 없음 | 74.4 | 57.8 |
| 하드 의사 레이블 사용 | 70.2 | 75.1 |
| SSL 백본 없음 | 68.7 | 56.2 |
| 분리 학습(decoupled) | 79.5 | 72.3 |

---

### 2.5 한계점

논문에서 명시한 한계:

1. **동일 클래스 불균형 분포 가정**: 레이블 데이터와 언레이블 데이터가 **동일한 클래스 불균형 비율**을 가진다고 가정. 실제 상황에서는 언레이블 데이터의 클래스 분포를 알 수 없거나 다를 수 있음.
   - 저자들은 향후 연구로 클래스 분포 추정 모듈 도입 계획 언급

2. **백본 선택 의존성**: ABC의 성능은 백본 SSL 알고리즘의 표현 품질에 크게 의존 (t-SNE 분석에서 ReMixMatch > FixMatch 확인)

3. **대규모 데이터에서의 DARP 불안정성**: DARP는 대규모 LSUN 데이터셋에서 성능 저하를 보인 반면, ABC는 안정적. 이는 ABC의 강점이지만 동시에 다른 방법론들의 한계도 보여줌.

4. **사회적 편향 위험**: 소수 집단 식별 및 차별적 활용 가능성에 대한 윤리적 고려 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 고품질 표현 활용을 통한 일반화

**핵심 메커니즘:**

ABC는 백본(FixMatch, ReMixMatch)이 **전체 미니배치**로 학습한 표현을 활용한다. Kang et al. (2020)의 연구에 따르면, 불균형 데이터에서도 분류기가 아닌 **표현 레이어는 고품질 표현**을 학습할 수 있다. ABC는 이 표현을 공유하면서 **균형 분류기 역할**만 담당하므로:

- 오버샘플링(소수 클래스 반복 학습)으로 인한 **표현 오버피팅 방지**
- 언더샘플링(다수 클래스 데이터 제거)으로 인한 **정보 손실 방지**

$$\text{일반화 이득} = \underbrace{\text{고품질 표현 (백본 전체 데이터 학습)}}_{\text{정보 손실 없음}} + \underbrace{\text{균형 분류기 (ABC 마스킹)}}_{\text{편향 완화}}$$

### 3.2 소프트 의사 레이블을 통한 일반화

하드 의사 레이블 대신 **소프트 의사 레이블** 사용:

- 하드 의사 레이블: 엔트로피 최소화 → 특정 클래스로 과도하게 편향
- 소프트 의사 레이블: **불확실성을 보존**하여 보다 안정적인 결정 경계 형성
- 애블레이션에서 소프트 → 하드 변경 시 전체 정확도 82.4 → 70.2로 급격히 저하

### 3.3 점진적 Bernoulli 파라미터 감소를 통한 일반화

초기 학습에서 의사 레이블이 부정확할 때 과도한 리샘플링을 방지:

$$\mathcal{B}(\cdot) \text{ 파라미터}: 1 \xrightarrow{\text{점진적 감소}} \frac{N_L}{N_{\hat{q}_b}}$$

이를 통해 학습 초기의 불안정성을 줄이고, 정확한 의사 레이블 형성 후 리샘플링 효과를 극대화함으로써 **학습 안정성과 일반화 성능을 동시에 향상**.

### 3.4 엔드-투-엔드 학습을 통한 일반화

분리 학습(decoupled learning, 표현 학습 후 분류기 파인튜닝) 대비 **엔드-투-엔드 학습**의 우위:

- 분리 학습에서는 분류기 파인튜닝 시 언레이블 데이터 미활용
- 엔드-투-엔드에서는 ABC 학습 과정에서 언레이블 데이터의 일관성 정규화 신호가 **표현 학습에도 피드백**
- 분리 학습 대비 전체 정확도 +2.9%, 소수 클래스 정확도 +3.4% (Table 5)

### 3.5 대규모 데이터셋에서의 일반화 (LSUN)

LSUN ($7.5M$ 데이터, $256 \times 256$ 이미지)에서도 미니배치 기반 학습으로 확장 가능:
- DARP는 전체 언레이블 데이터에 대한 볼록 최적화(convex optimization) 필요 → 대규모에서 성능 저하
- ABC는 미니배치 기반이므로 데이터 규모와 무관하게 **일관된 성능 유지**

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

#### (A) 모듈형 설계 패러다임의 확산

ABC는 기존 SSL 알고리즘을 블랙박스로 활용하면서 **최소한의 변경으로 클래스 불균형 문제를 해결**하는 모듈형 접근을 제시한다. 이는:

- 새로운 SSL 알고리즘이 등장할 때마다 ABC를 그대로 부착하여 CISSL로 확장 가능
- SimMatch, FlexMatch 등 최신 SSL 알고리즘에도 동일한 방식 적용 가능성 제시

#### (B) 언레이블 데이터 리샘플링 가능성 개척

기존에는 레이블이 없는 데이터에 대한 리샘플링이 불가능했으나, 소프트 의사 레이블 기반 마스킹 기법을 통해 **언레이블 데이터의 클래스 균형 활용**이 가능함을 보임. 이는 향후 더 정교한 언레이블 데이터 활용 전략 연구의 기반이 된다.

#### (C) 표현 학습과 분류기의 상호작용 연구

엔드-투-엔드 학습이 분리 학습보다 우수함을 실증적으로 보임으로써, **표현 학습과 분류기 학습의 상호 의존성**에 대한 후속 연구를 자극한다.

#### (D) 공정성(Fairness) AI 연구와의 연계

클래스 불균형 문제는 알고리즘적 공정성과 밀접하게 연관된다. ABC는 소수 집단에 대한 예측 정확도를 높임으로써 **공정한 AI 시스템 구축**에 기여할 수 있으며, 이 방향의 연구를 촉진할 것으로 기대된다.

---

### 4.2 향후 연구 시 고려할 사항

#### (A) 클래스 분포 불일치 문제 해결

논문의 핵심 한계인 **레이블/언레이블 데이터의 동일 클래스 분포 가정** 완화가 필요:

- 실제 환경에서는 언레이블 데이터의 클래스 분포가 다를 수 있음
- 클래스 분포 추정 모듈(예: EM 알고리즘 기반, 메타러닝 기반) 통합 연구 필요
- 분포 불일치에 강건한(robust) 의사 레이블 생성 기법 개발 필요

#### (B) 더 강력한 백본과의 결합

- Vision Transformer (ViT), Swin Transformer 등 최신 아키텍처와의 결합 시 성능 향상 가능성 탐구
- 자기지도학습(Self-supervised Learning, 예: SimCLR, MoCo) 기반 표현 학습과의 결합

#### (C) 동적 마스킹 전략 연구

현재 Bernoulli 파라미터는 $N_L/N_{y_b}$로 고정되어 있으나:

- 학습 진행에 따라 **동적으로 마스킹 파라미터를 조정**하는 적응형(adaptive) 마스킹 전략 연구
- 클래스별 현재 예측 정확도를 반영한 **피드백 기반 마스킹** 설계

#### (D) 의사 레이블 품질 향상

- 현재 신뢰도 임계값 $\tau = 0.95$는 고정된 하이퍼파라미터
- **클래스별 적응형 임계값**(Flex-type threshold, 예: FlexMatch [Zhang et al., 2021]) 도입 가능성 검토
- 의사 레이블의 클래스 불균형을 고려한 더 정교한 신뢰도 측정 방법 연구

#### (E) 실제 환경(Wild Distribution) 적용성 검토

- 의료 영상, 자율주행, 금융 사기 탐지 등 **극단적 클래스 불균형**($\gamma > 1000$) 환경에서의 성능 검증
- 동적으로 변하는 클래스 분포(concept drift)에 대한 온라인 학습 방식으로의 확장

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 비교는 논문 내 인용 및 공개된 관련 연구를 기반으로 하며, 논문 발표 이후(2021년 NeurIPS 이후) 연구에 대한 정보는 제가 학습한 데이터 기준으로 작성하였습니다. 이 부분은 확인 가능한 범위 내에서만 기술합니다.

### 논문 내 비교 대상 (2020~2021년 연구)

| 연구 | 방법론 | 특징 | ABC 대비 한계 |
|---|---|---|---|
| **DARP** (Kim et al., NeurIPS 2020) | 볼록 최적화 기반 의사 레이블 정제 | 편향된 의사 레이블 개선 | 대규모 데이터에서 비효율, 분류기 편향 근본 해결 미흡 |
| **CReST+PDA** (Wei et al., 2021) | 소수 클래스 분류 데이터 반복 자기학습 | 언레이블 데이터의 소수 클래스 우선 활용 | 반복 재학습 필요, 전체 언레이블 데이터 로딩 필요 |
| **BALMS** (Ren et al., NeurIPS 2020) | 균형 메타 소프트맥스 손실 | CIL 최신 기법 | 언레이블 데이터 미활용 |
| **Decoupling** (Kang et al., ICLR 2020) | 표현-분류기 분리 학습 | 편향된 분류기 교체 | 언레이블 데이터 분류기 학습 미활용, 엔드-투-엔드 미적용 |

### ABC 이후 등장한 관련 연구 방향 (주의: 아래는 일반적 연구 트렌드로, 개별 논문의 정확한 결과는 직접 확인 필요)

| 연구 방향 | 설명 |
|---|---|
| **FlexMatch** (Zhang et al., NeurIPS 2021) | 클래스별 적응형 신뢰도 임계값 도입 → ABC의 고정 임계값 한계 보완 가능성 |
| **Distribution-Aware SSL** | 언레이블 데이터의 클래스 분포 추정을 통한 동적 재균형화 |
| **ViT 기반 CISSL** | 트랜스포머 기반 표현 학습과 CISSL 결합 |

---

## 참고자료

### 주요 참고 논문 (논문 내 인용)

1. **Lee, H., Shin, S., Kim, H. (2021).** "ABC: Auxiliary Balanced Classifier for Class-Imbalanced Semi-Supervised Learning." *NeurIPS 2021.* (본 논문)

2. **Sohn, K. et al. (2020).** "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence." *NeurIPS 2020.*

3. **Berthelot, D. et al. (2019).** "ReMixMatch: Semi-Supervised Learning with Distribution Matching and Augmentation Anchoring." *ICLR 2020.*

4. **Kim, J. et al. (2020).** "Distribution Aligning Refinery of Pseudo-label for Imbalanced Semi-supervised Learning (DARP)." *NeurIPS 2020.*

5. **Wei, C. et al. (2021).** "CReST: A Class-Rebalancing Self-Training Framework for Imbalanced Semi-Supervised Learning." *arXiv:2102.09559.*

6. **Kang, B. et al. (2020).** "Decoupling Representation and Classifier for Long-Tailed Recognition." *ICLR 2020.*

7. **Ren, J. et al. (2020).** "Balanced Meta-Softmax for Long-Tailed Visual Recognition (BALMS)." *NeurIPS 2020.*

8. **Cao, K. et al. (2019).** "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss." *NeurIPS 2019.*

9. **Yang, Y. and Xu, Z. (2020).** "Rethinking the Value of Labels for Improving Class-Imbalanced Learning." *NeurIPS 2020.*

10. **Zagoruyko, S. and Komodakis, N. (2016).** "Wide Residual Networks." *arXiv:1605.07146.*

### GitHub 코드
- ABC 공식 코드: https://github.com/LeeHyuck/ABC
