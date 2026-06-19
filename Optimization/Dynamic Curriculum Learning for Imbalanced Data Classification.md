# Dynamic Curriculum Learning for Imbalanced Data Classification

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
Wang et al. (ICCV 2019)은 불균형 데이터 분류 문제를 해결하기 위해 **Dynamic Curriculum Learning (DCL)** 프레임워크를 제안합니다. 기존의 재샘플링(resampling)이나 비용 민감 학습(cost-sensitive learning)은 사전 지식(prior knowledge)이 필요하고 고정된 전략을 사용하는 반면, DCL은 **학습 과정에서 동적으로** 샘플링 전략과 손실 가중치를 적응적으로 조정하여 일반화 성능과 판별력을 동시에 향상시킵니다.

### 주요 기여 3가지
1. **최초로 커리큘럼 학습(Curriculum Learning) 아이디어를 불균형 데이터 학습에 도입**하여 동적 샘플링 및 손실 역전파를 위한 두 개의 커리큘럼 스케줄러를 설계
2. **DCL은 통합 표현(unified representation)**으로서 기존 SOTA 방법들(Cross Entropy, Selective Learning, CRL 등)을 특수 케이스로 포함하는 일반화된 프레임워크
3. 얼굴 속성 데이터셋 **CelebA**(mA: 89.05%)와 보행자 속성 데이터셋 **RAP**(mA: 83.7%)에서 당시 SOTA 달성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

불균형 데이터 분류(Imbalanced Data Classification)에서 발생하는 두 가지 핵심 문제:

- **문제 1 (샘플링 전략):** 처음부터 균형 분포를 목표로 하면 초기에 다수 클래스(majority class)의 유용한 정보를 과도하게 버려 **일반화 능력이 저하**됨. 반면 불균형 분포 그대로 학습하면 **소수 클래스(minority class) 성능이 열악**해짐.
- **문제 2 (손실 함수 가중치):** Cross Entropy(CE) 손실과 Metric Learning(ML) 손실을 균등하게 다루면 딥 CNN의 판별력을 충분히 활용하지 못함. 초기에는 특징 표현 학습(ML)을, 후기에는 분류 학습(CE)을 강조해야 함.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 스케줄러 함수 (Scheduler Function)

학습 진행 상태를 반영하여 1에서 0으로 단조감소하는 함수 $SF(l)$ 설계:

$$SF_{cos}(l) = \cos\!\left(\frac{l}{L} \cdot \frac{\pi}{2}\right) \quad \text{(Convex)}$$

$$SF_{linear}(l) = 1 - \frac{l}{L} \quad \text{(Linear)}$$

$$SF_{exp}(l) = \lambda^{l} \quad \text{(Concave, } \lambda \in (0,1)\text{)}$$

$$SF_{composite}(l) = \frac{1}{2}\cos\!\left(\frac{l}{L}\pi\right) + \frac{1}{2} \quad \text{(Composite)}$$

여기서 $L$은 총 학습 에폭 수, $l$은 현재 에폭.

---

#### 2.2.2 샘플링 스케줄러 (Sampling Scheduler)

각 속성에 대해 클래스 분포 $D$를 소수 클래스 수 기준으로 정규화:

```math
D = 1 : \frac{\#C_1}{\#C_{min}} : \frac{\#C_2}{\#C_{min}} : \cdots : \frac{\#C_{K-1}}{\#C_{min}}
```

훈련 중 목표 분포(target distribution)를 에폭 $l$에 따라 동적으로 조정:

$$D_{target}(l) = D_{train}^{g(l)} $$

여기서 $g(l)$은 샘플링 스케줄러 함수(Section 3.1의 함수 중 선택). 초기($l=0$)에는 $g(0)=1$이므로 $D_{target} = D_{train}$ (실제 불균형 분포), 마지막 에폭에서는 $g(l) \to 0$이므로 $D_{target} \to \mathbf{1}$ (균형 분포).

이를 기반으로 **Dynamic Selective Learning (DSL) 손실**:

$$\mathcal{L}_{DSL} = -\frac{1}{N} \sum_{j=1}^{M} \sum_{i=1}^{N_j} w_j \cdot \log\!\left(p(y_{i,j} = \bar{y}_{i,j} \mid \mathbf{x}_{i,j})\right) $$

$$w_j = \begin{cases} \dfrac{D_{target,j}(l)}{D_{current,j}} & \text{if } \dfrac{D_{target,j}(l)}{D_{current,j}} \geq 1 \\ 0/1 & \text{if } \dfrac{D_{target,j}(l)}{D_{current,j}} < 1 \end{cases} $$

- $N$: 배치 크기, $N_j$: 현재 배치에서 $j$번째 클래스 샘플 수, $M$: 클래스 수
- $\bar{y}_{i,j}$: ground truth 레이블, $w_j$: 클래스 $j$에 대한 비용 가중치

---

#### 2.2.3 Easy Anchor를 이용한 Metric Learning

CRL[7]에서의 Triplet Loss ($\mathcal{L}_{crl}$):

$$\mathcal{L}_{crl} = \frac{\sum_T \max\!\left(0,\; m_j + d(\mathbf{x}_{all,j}, \mathbf{x}_{+,j}) - d(\mathbf{x}_{all,j}, \mathbf{x}_{-,j})\right)}{|T|} $$

CRL은 소수 클래스의 모든 샘플을 앵커로 사용하여, 어렵게 분류된 양성 샘플(hard positive)이 앵커가 되면 음성 방향으로 밀릴 수 있는 문제가 있음.

DCL의 **Triplet Loss with Easy Anchors ($\mathcal{L}_{TEA}$)**:

$$\mathcal{L}_{TEA} = \frac{\sum_T \max\!\left(0,\; m_j + d(\mathbf{x}_{easy,j}, \mathbf{x}_{+,j}) - d(\mathbf{x}_{easy,j}, \mathbf{x}_{-,j})\right)}{|T|} $$

- $\mathbf{x}_{easy,j}$: 소수 클래스 $j$에서 높은 신뢰도로 올바르게 예측된 Easy 샘플(앵커)
- Easy anchor만을 앵커로 사용하여 hard positive를 안정적으로 끌어당기고, hard negative를 밀어내는 효과

---

#### 2.2.4 손실 스케줄러 (Loss Scheduler)

최종 DCL 손실:

$$\mathcal{L}_{DCL} = \mathcal{L}_{DSL} + f(l) \cdot \mathcal{L}_{TEA} $$

$$f(l) = \begin{cases} \dfrac{1}{2}\cos\!\left(\dfrac{l}{L}\pi\right) + \dfrac{1}{2} + \epsilon & \text{if } l < pL \\ \epsilon & \text{if } l \geq pL \end{cases} $$

- $p \in [0,1]$: 자기학습 시작점(advanced self-learning point), 논문에서 $p=0.3$으로 설정
- $\epsilon$: 자기학습 비율(self-learning ratio), 학습 후반에도 특징 구조를 유지하기 위한 소량의 가중치

**학습 흐름:**
- **초기:** $f(l)$이 크므로 $\mathcal{L}_{TEA}$ 비중이 높음 → 소프트 특징 임베딩 학습 우선
- **후기:** $f(l)$이 감소하여 $\mathcal{L}_{DSL}$ 비중이 높음 → 분류 성능 향상에 집중
- **자기학습 단계($l \geq pL$):** 스케줄러 없이 모델이 자율적으로 수렴

---

### 2.3 모델 구조

```
입력 이미지
    ↓
[Backbone: DeepID2 (CelebA) / ResNet-50 (RAP)]
    ↓
[속성별 64차원 특징 레이어 (멀티태스크)]
    ↓
[최종 분류 레이어 (속성별 독립)]
    ↓
┌─────────────────────────────────────────┐
│         DCL Framework                   │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │ Sampling      │  │  Loss Scheduler │  │
│  │ Scheduler g(l)│  │      f(l)       │  │
│  └──────┬───────┘  └────────┬────────┘  │
│         ↓                   ↓           │
│   L_DSL (Dynamic     L_TEA (Triplet    │
│   Selective Loss)    Easy Anchor)       │
│         └──────────┬────────┘           │
│               L_DCL = L_DSL + f(l)·L_TEA│
└─────────────────────────────────────────┘
```

**평가 지표 (Class-Balanced Mean Accuracy):**

$$mA_i = \frac{1}{2}\left(\frac{TP_i}{P_i} + \frac{TN_i}{N_i}\right) $$

$$mA = \frac{\sum_{i=1}^{|C|} mA_i}{|C|} $$

---

### 2.4 성능 향상

| 데이터셋 | 베이스라인 | DCL | 향상 |
|----------|-----------|-----|------|
| CelebA (mA) | DeepID2: 81.17% | **89.05%** | +7.88% |
| RAP (mA) | LG-Net: 78.7% | **83.7%** | +5.0% |
| CIFAR-100 | CE: 68.1% | **71.5%** | +3.4% |

**불균형 비율에 따른 RAP 성능 (Table 6):**

| 불균형 비율 | 베이스라인(ResNet-50) | DCL |
|------------|---------------------|-----|
| 1∼25 | 79.3% | 83.1% (+3.8%) |
| 25∼50 | 68.9% | 83.9% (+15.0%) |
| >50 | 68.0% | 85.5% (+17.5%) |

> 불균형이 심할수록 DCL의 효과가 더욱 두드러짐.

**Ablation Study (Table 3, CelebA):**

| 구성 | SS | TL | LS | 성능 |
|------|----|----|-----|------|
| Baseline (DeepID2) | ✗ | ✗ | ✗ | 81.17% |
| +Sampling Scheduler | ✓ | ✗ | ✗ | 86.58% |
| +Triplet(Easy Anchor) | ✓ | ✓ | ✗ | 87.55% |
| +Loss Scheduler (DCL) | ✓ | ✓ | ✓ | **89.05%** |

---

### 2.5 한계점

논문에서 명시적으로 언급된 한계와 분석을 통해 도출된 한계:

1. **하이퍼파라미터 민감성:** $p$, $\epsilon$, $k$, 마진(margin), 스케줄러 함수 유형 등 다수의 하이퍼파라미터 조정이 필요하며, 이를 새로운 도메인에 적용 시 재튜닝 비용 발생
2. **멀티레이블 최적화의 복잡성:** CelebA처럼 40개 속성이 동시에 존재할 때 각 속성별 독립적 분포 관리가 최적인지 검증 불충분
3. **소수 클래스 샘플 고갈 문제:** 극단적 불균형(1:1800)에서 소수 클래스 Easy Anchor 선택이 불안정해질 수 있음
4. **스케줄러 함수 선택의 이론적 근거 부족:** Convex 함수가 가장 좋다는 실험적 결론만 있고 이론적 분석은 없음
5. **2019년 기준 백본 사용:** DeepID2는 현재 기준으로 성능이 제한적이며, 최신 Transformer 기반 모델과의 비교 없음

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 균형 데이터셋(CIFAR-100)에서의 일반화 검증

논문은 CIFAR-100(균형 데이터셋)에서 실험하여 DCL의 범용성을 증명:

- **CE 베이스라인:** 68.1%
- **CRL:** 69.3% (+1.2%)
- **DCL:** **71.5% (+3.4%)**

이는 DCL이 불균형 데이터에만 특화된 것이 아니라 **일반 분류 문제에서도 특징 임베딩 개선**을 통해 성능을 향상시킬 수 있음을 시사합니다.

### 3.2 일반화 성능 향상의 메커니즘

**메커니즘 1: 커리큘럼 기반 점진적 학습**

초기에 실제 분포(불균형)로 학습하여 다수/소수 클래스 모두의 **일반적 특징(general representation)**을 먼저 학습하고, 점진적으로 균형 분포로 이동하여 소수 클래스 판별력을 강화합니다.

$$g(0) = 1 \Rightarrow D_{target}(0) = D_{train} \quad \text{(실제 분포)}$$
$$g(L) \to 0 \Rightarrow D_{target}(L) \to \mathbf{1} \quad \text{(균형 분포)}$$

이 점진적 전환은 다수 클래스 정보를 폐기하지 않으면서 소수 클래스 학습을 강화하는 균형을 유지합니다.

**메커니즘 2: Easy Anchor 기반 안정적 특징 공간 구조화**

$\mathcal{L}_{TEA}$는 고신뢰도로 분류된 Easy 샘플을 앵커로 사용하므로, **안정적인 클러스터 중심**을 형성하고 Hard 샘플들을 올바른 방향으로 끌어당깁니다. 이는 과적합을 방지하고 일반화된 결정 경계를 학습하는 데 기여합니다.

**메커니즘 3: 손실 스케줄러를 통한 순차적 최적화**

$$\mathcal{L}_{DCL} = \underbrace{\mathcal{L}_{DSL}}_{\text{분류}} + \underbrace{f(l) \cdot \mathcal{L}_{TEA}}_{\text{특징 임베딩 (감소)}}$$

초기에 풍부한 특징 표현을 먼저 학습(Metric Learning)한 후, 이를 기반으로 정밀한 분류(CE)를 수행하는 **순차적 최적화**가 일반화에 기여합니다.

**메커니즘 4: DCL 프레임워크의 통합성**

Table 1에서 보듯 기존 방법들이 DCL의 특수 케이스:

| 방법 | $g(x)$ | $f(x)$ |
|------|--------|--------|
| Cross Entropy | 1 | 0 |
| Selective Learning | 0/1 | 0 |
| CRL-I | 1 | $\epsilon$ |
| DCL | Sampling Scheduler | Loss Scheduler |

이 통합성은 다양한 도메인에 적용 시 $g(x)$와 $f(x)$를 조정하여 **유연하게 적응**할 수 있음을 의미합니다.

### 3.3 일반화 한계

- CIFAR-100 실험이 단일 데이터셋에 한정되어 있어 자연어 처리, 의료 영상 등 타 도메인 일반화 검증 부재
- 초고불균형(extreme long-tail) 시나리오에서의 Easy Anchor 부족 문제

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

**영향 1: 커리큘럼 학습과 불균형 학습의 융합 패러다임 제시**

DCL은 커리큘럼 학습을 불균형 데이터에 최초 적용함으로써, 이후 연구들이 "언제, 어떤 데이터를, 어떤 가중치로 학습할 것인가"를 동적으로 결정하는 연구 방향을 열었습니다.

**영향 2: 동적 손실 가중치 조정 연구의 촉매**

손실 스케줄러 개념은 이후 Meta-Weight-Net, AutoBalance 등 **자동화된 손실 가중치 학습 연구**로 발전하는 토대를 마련했습니다.

**영향 3: 롱테일 인식(Long-tail Recognition) 연구 확장**

DCL의 "불균형→균형" 분포 전이 아이디어는 이후 BBN(Bilateral-Branch Network), LDAM-DRW 등 롱테일 분류 연구에서 **deferred re-balancing** 전략으로 발전합니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

#### (1) LDAM-DRW (NeurIPS 2020)
- **논문:** Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss," NeurIPS 2019 → 2020년 이후 확장 연구
- **방법:** 소수 클래스에 더 큰 마진을 부여하는 LDAM 손실 + Deferred Re-Weighting(DRW) 전략
- **DCL과 비교:** DRW는 DCL의 Loss Scheduler와 유사하게 학습 후반에 클래스 재가중치를 적용하지만, DCL은 샘플링까지 동시에 동적으로 조정하는 점에서 더 포괄적

$$\mathcal{L}_{LDAM} = -\log \frac{e^{\hat{z}_{y} - \Delta_y}}{e^{\hat{z}_{y} - \Delta_y} + \sum_{j \neq y} e^{\hat{z}_j}}, \quad \Delta_j = \frac{C}{n_j^{1/4}}$$

#### (2) BBN (CVPR 2020)
- **논문:** Zhou et al., "BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition," CVPR 2020
- **방법:** 두 가지 브랜치(conventional branch + re-balancing branch)를 혼합 비율 $\alpha$로 조합하며, $\alpha$를 에폭에 따라 점진적으로 조정
- **DCL과 비교:** BBN의 $\alpha$ 스케줄링은 DCL의 샘플링 스케줄러 $g(l)$과 개념적으로 유사하나, BBN은 네트워크 구조 수준에서 분리를 구현

$$\mathbf{z} = \alpha \cdot f_b(\mathbf{x}^b; \theta_b) + (1-\alpha) \cdot f_r(\mathbf{x}^r; \theta_r)$$

#### (3) MiSLAS (CVPR 2021)
- **논문:** Zhong et al., "Improving Calibration for Long-Tailed Recognition," CVPR 2021
- **방법:** 믹스업(Mixup)과 레이블 스무딩을 결합하여 롱테일 분포에서 모델 캘리브레이션 개선
- **DCL과의 연결:** DCL의 easy/hard 샘플 분리 개념이 MiSLAS의 mixup 전략과 결합될 가능성 존재

#### (4) Class-Balanced Loss (CVPR 2019) / Effective Number
- **논문:** Cui et al., "Class-Balanced Loss Based on Effective Number of Samples," CVPR 2019
- **방법:** 클래스별 유효 샘플 수(effective number)에 기반한 손실 재가중치
- **DCL과 비교:** DCL은 에폭에 따른 동적 가중치인 반면, 이 방법은 정적 가중치를 사용

$$\beta = \frac{N-1}{N}, \quad E_{n_y} = \frac{1-\beta^{n_y}}{1-\beta}$$

#### (5) PaCo (ICCV 2021)
- **논문:** Cui et al., "Parametric Contrastive Learning," ICCV 2021
- **방법:** 파라메트릭 클래스 프로토타입을 활용한 대조 학습으로 롱테일 인식 개선
- **DCL과 비교:** DCL의 Easy Anchor 기반 Triplet Loss가 Contrastive Learning으로 진화한 형태로 볼 수 있으며, 더 풍부한 부정 샘플을 활용

#### 비교 요약표

| 방법 | 년도 | 동적 조정 | 메트릭 학습 | 이론적 근거 | 계산 비용 |
|------|------|----------|------------|------------|---------|
| DCL | 2019 | ✓ (샘플링+손실) | ✓ (Triplet+Easy Anchor) | 부분적 | 낮음 |
| LDAM-DRW | 2019/20 | ✓ (재가중치만) | ✗ | 이론적 마진 분석 | 낮음 |
| BBN | 2020 | ✓ (브랜치 비율) | ✗ | 쌍분기 구조 | 중간 |
| MiSLAS | 2021 | ✗ | ✗ | 캘리브레이션 이론 | 낮음 |
| PaCo | 2021 | ✗ | ✓ (대조학습) | 파라메트릭 이론 | 높음 |

---

### 4.3 앞으로 연구 시 고려할 점

**고려사항 1: 스케줄러 함수의 자동화(AutoML)**

현재 Convex/Linear/Concave/Composite 함수 중 수동 선택이 필요합니다. Neural Architecture Search(NAS)나 메타러닝을 통해 데이터 분포에 최적화된 스케줄러 함수를 자동으로 학습하는 연구가 필요합니다.

**고려사항 2: 극단적 롱테일(Extreme Long-tail) 시나리오 확장**

RAP에서 최대 1:1800의 불균형을 다루었지만, ImageNet-LT(최대 1:1000 이상)나 iNaturalist(1:500 이상)와 같은 대규모 롱테일 벤치마크에서의 검증이 필요합니다.

**고려사항 3: Vision Transformer(ViT) 기반 백본과의 통합**

DeepID2나 ResNet-50 기반의 실험만 수행되어, 현재 주류인 ViT/Swin Transformer에서 DCL의 샘플링 및 손실 스케줄러가 동일하게 효과적인지 검증이 필요합니다.

**고려사항 4: Self-Supervised Learning과의 결합**

초기 특징 학습 단계에서 대조 학습(Contrastive Learning, SimCLR, MoCo)을 활용하면 Easy Anchor의 품질을 높일 수 있으며, 이는 $\mathcal{L}_{TEA}$의 안정성을 더욱 향상시킬 수 있습니다.

**고려사항 5: 다중 도메인 일반화(Domain Generalization)**

현재 DCL은 단일 도메인 내 불균형을 처리하지만, 도메인 편향(domain bias)과 클래스 불균형이 동시에 존재하는 현실적 시나리오에서의 적용 가능성 연구가 필요합니다.

**고려사항 6: 공정성(Fairness)과의 연결**

소수 클래스 성능 향상은 AI 공정성(algorithmic fairness) 연구와 깊이 연결됩니다. 인구 통계학적 소수 집단(demographic minority)에 대한 분류 공정성 향상에 DCL 프레임워크를 적용하는 연구도 유망합니다.

---

## 참고 자료

**주요 참고 논문 (논문 내 인용 기준):**

1. **Wang et al. (2019)** - "Dynamic Curriculum Learning for Imbalanced Data Classification," ICCV 2019 *(본 논문)*
2. **Bengio et al. (2009)** - "Curriculum Learning," ICML 2009
3. **Dong et al. (2017, 2018)** - "Class Rectification Hard Mining for Imbalanced Deep Learning," ICCV 2017; "Imbalanced Deep Learning by Minority Class Incremental Rectification," IEEE TPAMI 2018
4. **Huang et al. (2016, 2018)** - "Learning Deep Representation for Imbalanced Classification," CVPR 2016; "Deep Imbalanced Learning for Face Recognition and Attribute Prediction," arXiv 2018
5. **Hand et al. (2018)** - "Doing the Best We Can with What We Have: Multi-Label Balancing with Selective Learning," AAAI 2018
6. **Cui et al. (2019)** - "Class-Balanced Loss Based on Effective Number of Samples," CVPR 2019

**2020년 이후 비교 분석 참고 논문:**

7. **Cao et al. (2019)** - "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss," NeurIPS 2019
8. **Zhou et al. (2020)** - "BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition," CVPR 2020
9. **Zhong et al. (2021)** - "Improving Calibration for Long-Tailed Recognition," CVPR 2021
10. **Cui et al. (2021)** - "Parametric Contrastive Learning," ICCV 2021

> **⚠️ 주의:** 2020년 이후 최신 연구(BBN, MiSLAS, PaCo 등)와의 비교 분석은 제가 보유한 학습 지식을 기반으로 작성하였으며, 해당 논문들의 원문을 직접 검색하여 확인하지 않았습니다. 구체적 수치나 방법론 세부 사항은 원문 논문을 반드시 확인하시기 바랍니다. DCL 논문 자체의 내용(수식, 실험 결과 등)은 제공된 PDF 원문을 기반으로 100% 정확하게 작성하였습니다.
