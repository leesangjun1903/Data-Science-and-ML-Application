# Partially View-aligned Representation Learning with Noise-robust Contrastive Loss

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문(CVPR 2021)은 **부분적으로 정렬된 멀티뷰 데이터(Partially View-aligned Problem, PVP)** 환경에서, 레이블 없이 표현 학습과 데이터 정렬을 동시에 수행할 수 있는 방법론인 **MvCLN(Multi-view Contrastive Learning with Noise-robust loss)**을 제안합니다.

핵심 주장은 다음 세 가지로 요약됩니다:

1. **인스턴스 수준 정렬보다 카테고리 수준 정렬이 클러스터링/분류에 더 적합하다.**
2. **뷰 정렬 문제를 식별(identification) 태스크로 재정의하고, 대조 학습으로 해결할 수 있다.**
3. **랜덤 샘플링으로 생성된 거짓 음성 쌍(False Negative Pairs, FNP)을 처리하는 노이즈 강건 대조 손실을 제안한다.**

### 주요 기여

| 기여 | 내용 |
|------|------|
| 카테고리 수준 정렬 | 인스턴스 수준( $O(N^3)$ )보다 접근성·확장성이 높은 카테고리 수준 정렬 제안 |
| 대조 학습 기반 정렬 | 뷰 정렬 문제를 대조 학습 프레임워크로 최초 재정의 |
| 노이즈 강건 손실 | FNP 영향을 적응적으로 억제하는 새로운 손실 함수 설계 |
| 노이즈 레이블 패러다임 확장 | 지도학습의 노이즈 레이블 개념을 뷰 대응 관계로 확장 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

기존 멀티뷰 표현 학습(MvRL)은 다음 두 가지 가정에 의존합니다:
- **완전성 가정**: 모든 뷰에 인스턴스가 존재
- **일관성 가정**: 서로 다른 뷰의 데이터가 엄격히 정렬됨

현실에서는 공간적·시간적 비동기로 인해 일부 데이터만 정렬되는 **PVP(Partially View-aligned Problem)**가 발생합니다.

기존 해법(헝가리안 알고리즘, PVC)의 한계:
- 헝가리안 알고리즘: 이종(heterogeneous) 원시 공간에 직접 적용 불가, 계산 복잡도 $O(N^3)$
- PVC: 인스턴스 수준 정렬만 수행 → 클러스터링/분류에서 과도한 제약

### 2.2 제안 방법 및 수식

#### 문제 공식화

$v$개의 뷰를 가진 부분 정렬 데이터셋:

$$\{\mathbf{X}^i\}_{i=1}^{v} = \{\mathbf{A}^i, \mathbf{U}^i\}_{i=1}^{v}$$

- $\mathbf{A}^i = \{a^i_1, a^i_2, \ldots, a^i_{N_1}\}$: 정렬된 데이터
- $\mathbf{U}^i = \{u^i_1, u^i_2, \ldots, u^i_{N_2}\}$: 미정렬 데이터

카테고리 수준 정렬 목표:

$$C(\mathbf{x}^1_k) = C(\mathbf{x}^2_k), \quad \forall k \in [1, N] \tag{1}$$

#### 기본 손실 함수

전체 손실:

$$\mathcal{L} = \frac{1}{2N} \sum_{i=1}^{N} \left( P\mathcal{L}^{pos}_i + (1-P)\mathcal{L}^{neg}_i \right) \tag{2}$$

($P=1$: 양성 쌍, $P=0$: 음성 쌍)

**양성 쌍 손실** (거리 최소화):

$$\mathcal{L}^{pos}_i = d(a^1_i, a^2_i) \tag{3}$$

$$d(a^1_i, a^2_i) = \|f_1(a^1_i) - f_2(a^2_i)\|^2_2 \tag{4}$$

**기존 대조 손실 (SIAMESE)**:

$$\mathcal{L}^{ctr}_i = \max(m - d(a^1_i, a^2_j), 0)^2 \tag{5}$$

#### 핵심: 노이즈 강건 대조 손실

$$\mathcal{L}^{neg}_i = \frac{1}{m} \max\left(m d^{\frac{1}{2}}(a^1_i, a^2_j) - d^{\frac{3}{2}}(a^1_i, a^2_j), 0\right)^2 \tag{6}$$

마진 $m$은 초기 상태에서 한 번만 계산:

$$m = \frac{1}{N_p}\sum d(a^1_i, a^2_i) + \frac{1}{N_n}\sum d(a^1_i, a^2_j) \tag{7}$$

#### 손실 함수의 수학적 분석

$\mathcal{L}^{neg}$의 거리 $d$에 대한 기울기:

$$\frac{\partial \mathcal{L}^{neg}}{\partial d} = \frac{\partial \left(\frac{1}{m}d^3 - 2d^2 + md\right)}{\partial d} = \frac{3}{m}d^2 - 4d + m \tag{8}$$

기울기를 0으로 놓으면 $d = m/3$ 또는 $d = m$ → 두 영역으로 분리:

| 영역 | 동작 | 효과 |
|------|------|------|
| $0 < d < m/3$ | **역방향 최적화** (기울기 역전) | FNP를 양성 쌍처럼 처리 |
| $m/3 < d < m$ | **느린 최적화** (기울기 감소) | FNP 영향 완화 |
| $d > m$ | 거의 영향 없음 | TNP 보존 |

기존 손실과의 기울기 차이:

$$\Delta = \frac{\partial \mathcal{L}^{neg}}{\partial d} - \frac{\partial \mathcal{L}^{ctr}}{\partial d} = \frac{\partial (d-m)^2}{\partial d} \geq 0 \tag{9}$$

즉, $m/3 < d < m$ 구간에서 본 논문의 손실이 항상 더 작은 기울기를 가집니다.

### 2.3 모델 구조

```
입력 (View 1) → Network f₁ (D-1024-1024-1024-10) → 잠재 표현
입력 (View 2) → Network f₂ (D-1024-1024-1024-10) → 잠재 표현
                                ↓
                    [Pair Construction]
                    - Positive: 정렬된 데이터 A
                    - Negative: 랜덤 샘플링 (비율 M=30)
                                ↓
                [두 단계 최적화 전략]
           Stage 1: 기존 대조 손실 (Eq. 5) → NP 평균 거리 ≥ m
           Stage 2: 노이즈 강건 손실 (Eq. 6)
                                ↓
              [Category-level Alignment]
         거리 행렬 D ∈ R^{N×N} → 최소 거리 기반 대응 쌍 결정
```

모든 레이어: Dense → BatchNorm → ReLU → Dropout

### 2.4 성능 향상

**클러스터링 성능 비교 (부분 정렬 설정)**:

| 데이터셋 | 지표 | PVC (NeurIPS'20) | MvCLN (평균) | 향상율 |
|---------|------|-----------------|-------------|--------|
| Caltech-101 | ARI | 17.98 | 38.34 | **+113.2%** |
| Reuters | ARI | 16.95 | 24.90 | **+46.9%** |
| NoisyMNIST | ACC | 81.84 | 91.05 | **+11.2%** |
| Scene-15 | ACC | 37.88 | 38.53 | +1.7% |

**절제 연구 (NoisyMNIST)**:

| $\mathcal{L}_{neg}$ 사용 | ACC | NMI | ARI | CAR |
|--------------------------|-----|-----|-----|-----|
| ✗ (vanilla) | 88.2 | 75.73 | 75.89 | 84.48 |
| ✓ (noise-robust) | **92.17** | **85.57** | **84.51** | **88.24** |

카테고리 수준 정렬률(CAR):

$$\text{CAR} = \frac{\sum_{i=1}^{N} \delta(C(\hat{x}^1_i), C(\hat{x}^2_i))}{N} \tag{10}$$

### 2.5 한계점

1. **양성 쌍도 노이즈가 될 수 있는 상황 미처리**: 현재는 음성 쌍의 노이즈만 처리
2. **이진 뷰 중심**: 이론적으로 다중 뷰 확장 가능하나 실험은 2뷰 중심
3. **마진 $m$ 고정**: 초기에 한 번만 계산되어 학습 과정에서 적응되지 않음
4. **대규모 데이터셋 제한**: NoisyMNIST에서 30,000개만 사용 (기존 방법들의 메모리/시간 제한으로 인한 다운샘플링)
5. **두 단계 전환 타이밍 민감성**: 전환 너무 이르거나 늦으면 성능 저하

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 카테고리 수준 정렬의 일반화 우위

인스턴스 수준 정렬의 랜덤 정렬 확률: $\frac{1}{N}$

카테고리 수준 정렬의 랜덤 정렬 확률: $\frac{1}{K}$

여기서 $K \ll N$이므로, 카테고리 수준 정렬이 **접근성과 확장성 측면에서 본질적으로 더 유리**합니다. 이는 다양한 도메인의 다운스트림 태스크(클러스터링, 분류)에 대한 일반화를 강화합니다.

### 3.2 정렬 비율에 따른 일반화 강건성

실험에서 정렬 비율을 10%~100%로 변화시킨 결과:
- 정렬 데이터가 **70% 이상**이면 성능이 포화 → 적은 정렬 데이터로도 패턴 학습 가능
- 10%의 정렬 데이터만으로도 의미 있는 성능 달성 → **데이터 효율적 일반화**

### 3.3 파라미터 강건성

음성/양성 비율 $M$을 1~50으로 변화시켜도 $M \in [20, 40]$ 범위에서 안정적 성능 유지 → **하이퍼파라미터에 대한 일반화 강건성** 확인

### 3.4 두 단계 최적화의 일반화 기여

Bengio et al.(2017)의 "신경망은 단순 패턴 먼저 학습" 원리를 활용:
- Stage 1: TNP(단순 패턴)와 FNP(복잡 패턴) 사이에 거리 차이 형성
- Stage 2: 형성된 차이를 기반으로 FNP 영향 선택적 억제

이 메커니즘은 **다양한 노이즈 수준에서도 안정적인 일반화**를 가능하게 합니다.

### 3.5 전환 타이밍의 일반화

전환 타이밍이 $[0.2m, 1.0m]$ 범위에서 안정적 → 데이터 적응적(data-driven) 전환으로 다양한 데이터셋에 자동 적응

### 3.6 완전 정렬 데이터 대비 경쟁력

부분 정렬 데이터로 학습한 MvCLN이 **완전 정렬 데이터로 학습한 기존 방법들과 경쟁적인 성능** 달성 → 실제 환경에서의 일반화 가능성 입증

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

#### 노이즈 레이블 패러다임의 확장
기존 노이즈 레이블 연구는 지도학습의 잘못된 클래스 레이블에 집중했으나, 본 논문은 **"뷰 대응 관계 자체가 노이즈"**라는 새로운 정의를 제시합니다. 이는 자기지도 학습, 대조 학습 전반에 걸쳐 새로운 연구 방향을 제시합니다.

#### 대조 학습 + 노이즈 강건성의 통합
SimCLR, MoCo 등 기존 대조 학습 방법들은 FNP 문제를 명시적으로 처리하지 않았습니다. MvCLN의 접근법은 이후 연구들이 FNP를 체계적으로 다루는 기반을 마련했습니다.

#### 멀티모달 학습에의 응용
비전-언어, 오디오-비디오 등 멀티모달 환경에서 데이터가 완전히 정렬되지 않는 실제 시나리오에 직접 적용 가능한 프레임워크를 제시합니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 발표 | 주요 특징 | MvCLN과의 차이 |
|------|------|-----------|--------------|
| **PVC** (Huang et al., NeurIPS 2020) | NeurIPS 2020 | 미분 가능 헝가리안 모듈, 인스턴스 수준 정렬 | MvCLN은 카테고리 수준 정렬로 더 확장성 높음 |
| **COMPLETER** (Lin et al., CVPR 2021) | CVPR 2021 | 불완전 멀티뷰에서 대조 예측 기반 | PDP 해결에 집중, PVP는 다루지 않음 |
| **SimCLR** (Chen et al., ICML 2020) | ICML 2020 | 단일 뷰 데이터 증강 기반 대조 학습 | FNP 문제 비처리, 멀티뷰 PVP 미지원 |
| **MoCo v2** (He et al., 2020) | arXiv 2020 | 모멘텀 인코더 기반 대조 학습 | FNP 미처리, 단일 뷰 중심 |
| **Contrastive Clustering** (Li et al., AAAI 2021) | AAAI 2021 | 행/열 수준의 대조 클러스터링 | 완전 정렬 가정, FNP 미처리 |

**핵심 차별점**: MvCLN은 PVP 환경에서의 FNP를 명시적으로 처리하는 최초의 대조 학습 방법으로, 이후 연구들에게 "정렬되지 않은 멀티뷰 데이터에서의 대조 학습"이라는 연구 방향을 개척했습니다.

### 4.3 앞으로 연구 시 고려할 점

#### (1) 양성 쌍 노이즈 처리
현재 MvCLN은 음성 쌍의 노이즈만 처리합니다. 실제 환경에서는 **알려진 대응 쌍(양성 쌍)도 오류**를 포함할 수 있습니다. 논문 자체에서도 미래 연구 방향으로 명시한 문제입니다:

> *"In the future, we plan to explore a more general solution for the situation wherein both positive and negative pairs would be contaminated by noise."*

#### (2) 마진의 적응적 갱신
현재 마진 $m$은 초기 1회만 계산됩니다. 학습 과정에서 데이터 분포가 변함에 따라 **동적으로 마진을 업데이트**하는 메커니즘이 성능 향상에 기여할 수 있습니다.

#### (3) 3개 이상의 뷰 일반화
현재 이론 및 실험이 $v=2$ 중심입니다. $v > 2$인 경우 쌍 구성 방식과 손실 함수 설계가 더 복잡해지며, **다중 뷰 간 상호 정보 활용** 전략이 필요합니다.

#### (4) 대규모 데이터셋 확장성
NoisyMNIST에서 30,000개로 다운샘플링한 것은 기존 방법의 한계 때문입니다. **배치 단위 근사 알고리즘** 또는 **계층적 샘플링 전략**을 통한 확장성 개선이 필요합니다.

#### (5) 트랜스포머 기반 백본 통합
현재 MvCLN은 MLP 기반 인코더를 사용합니다. Vision Transformer(ViT), BERT 등 **사전 훈련된 대형 모델**을 인코더로 활용하면 표현 품질과 일반화 성능을 크게 향상시킬 수 있습니다.

#### (6) 정렬 비율의 자동 추정
현재는 정렬 비율을 사전에 알고 있다고 가정합니다. 실제 환경에서는 **정렬 비율 자체를 자동 추정**하는 메커니즘이 필요합니다.

#### (7) 이론적 수렴 보장
두 단계 최적화 전략의 수렴성과 최적성에 대한 **엄밀한 이론적 분석**이 부족합니다. PAC 학습 이론이나 정보 이론적 관점에서의 분석이 향후 연구에서 필요합니다.

---

## 참고 자료

**주요 참고 논문 (논문 내 인용 기반)**:

- **본 논문**: Yang, M., Li, Y., Huang, Z., Liu, Z., Hu, P., & Peng, X. (2021). *Partially View-aligned Representation Learning with Noise-robust Contrastive Loss*. CVPR 2021. pp. 1134–1143.

- Huang, Z., Hu, P., Zhou, J. T., Lv, J., & Peng, X. (2020). *Partially view-aligned clustering*. **NeurIPS 2020**.

- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A simple framework for contrastive learning of visual representations*. arXiv:2002.05709.

- Lin, Y., Gou, Y., Liu, Z., Li, B., Lv, J., & Peng, X. (2021). *COMPLETER: Incomplete multi-view clustering via contrastive prediction*. **CVPR 2021**.

- Li, Y., Hu, P., Liu, Z., Peng, D., Zhou, J. T., & Peng, X. (2021). *Contrastive clustering*. **AAAI 2021**.

- Arpit, D., et al. (2017). *A closer look at memorization in deep networks*. **ICML 2017**.

- Hadsell, R., Chopra, S., & LeCun, Y. (2006). *Dimensionality reduction by learning an invariant mapping*. **CVPR 2006** (SIAMESE network).

- Song, H., Kim, M., Park, D., & Lee, J. G. (2020). *Learning from noisy labels with deep neural networks: A survey*. arXiv:2007.08199.
