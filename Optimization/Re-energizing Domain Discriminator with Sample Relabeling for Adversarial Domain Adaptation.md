# Re-energizing Domain Discriminator with Sample Relabeling for Adversarial Domain Adaptation

## 📌 참고 자료
- **주 논문**: Jin, X., Lan, C., Zeng, W., & Chen, Z. (2021). "Re-energizing Domain Discriminator with Sample Relabeling for Adversarial Domain Adaptation." *ICCV 2021*, pp. 9174–9183.
- Ben-David et al. (2007). "Analysis of representations for domain adaptation." NeurIPS.
- Ganin et al. (2016). "Domain-adversarial training of neural networks." JMLR. (DANN)
- Long et al. (2018). "Conditional adversarial domain adaptation." NeurIPS. (CDAN)
- Zhang et al. (2018). "mixup: Beyond empirical risk minimization." ICLR.

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

적대적 도메인 적응(Adversarial Domain Adaptation) 학습에서, 학습이 진행될수록 소스·타겟 도메인의 피처 분포가 점점 정렬(aligned)됨에 따라 **도메인 판별기(domain discriminator)의 판별 능력이 저하**되어, 피처 추출기(feature extractor)에 대한 정렬 구동력(driving power)이 감소하는 근본적 최적화 문제가 존재한다.

이를 해결하기 위해 **동적 도메인 레이블(dynamic domain labels)** 을 활용한 샘플 재레이블링(sample relabeling) 전략, 즉 **RADA(Re-enforceable Adversarial Domain Adaptation)** 를 제안한다.

### 🏆 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| 문제 규명 | 판별기의 판별 능력 저하로 인한 최적화 정체 현상을 엔트로피 및 MMD 측정으로 실증 |
| 방법 제안 | 잘 정렬된 타겟 샘플을 소스로 재레이블하여 판별기를 재활성화하는 RADA 전략 |
| 범용성 | 네트워크 구조 변경 없이 기존 UDA 프레임워크에 플러그앤플레이 방식으로 적용 가능 |
| 성능 | 단일·다중 소스 UDA 벤치마크에서 SOTA 달성 |

---

## 2. 문제, 방법, 모델 구조, 성능, 한계 상세 설명

### 2.1 해결하고자 하는 문제

기존 적대적 UDA 방법(DANN, CDAN 등)은 **정적(static) 도메인 레이블**을 사용하여 학습한다. 학습이 진행될수록:

1. 소스·타겟 피처 분포가 점점 정렬됨
2. 정렬된 샘플들은 도메인 판별기가 구분하기 어려워짐
3. 판별기의 도메인 분류 엔트로피가 상승 → **판별 능력 저하**
4. 판별 능력 저하 → 피처 추출기에 대한 정렬 구동력 감소 → **최적화 정체**

이를 Figure 2에서 CDAN 기준으로 엔트로피 변화 및 MMD(Maximum Mean Discrepancy)로 실증하였다.

### 2.2 제안하는 방법 (수식 포함)

#### (1) 기본 적대적 UDA 학습 목적함수

객체 분류 손실:

$$\mathcal{L}_{cls} = \frac{1}{N_s} \sum_{i=1}^{N_s} \mathcal{L}_{ce}(C(F(\mathbf{x}_i^s)), y_i^s)$$

도메인 적대적 손실:

$$\mathcal{L}_{adv} = -\frac{1}{N_s} \sum_{i=1}^{N_s} \log D(F(\mathbf{x}_i^s)) - \frac{1}{N_t} \sum_{j=1}^{N_t} \log(1 - D(F(\mathbf{x}_j^t)))$$

전체 최적화 목표:

$$\min_{D} \mathcal{L}_{adv}$$

$$\min_{F, C} \mathcal{L}_{cls} - \lambda \mathcal{L}_{adv}$$

여기서 $F$: 피처 추출기, $C$: 객체 분류기, $D$: 도메인 판별기, $\lambda$: 균형 하이퍼파라미터.

#### (2) 정렬 정도 측정 (엔트로피 기반)

타겟 샘플의 정렬 여부를 도메인 분류 엔트로피로 측정:

$$\mathcal{H}(\mathbf{p}) = -p_0 \log p_0 - (1 - p_0) \log(1 - p_0)$$

여기서 $\mathbf{p} = [p_0, 1-p_0]$이며, $p_0$는 해당 샘플을 타겟 도메인으로 예측할 확률, $1-p_0$는 소스 도메인으로 예측할 확률이다. **엔트로피가 클수록 판별기의 불확실성이 높고, 해당 샘플이 잘 정렬된 것으로 판단**한다.

#### (3) 재레이블 기준

엔트로피가 임계값 $\tau$보다 크면 해당 타겟 샘플을 소스 도메인으로 재레이블:

$$\text{if } \mathcal{H}(\mathbf{p}_j^t) > \tau \Rightarrow \text{relabel } \mathbf{x}_j^t \text{ as source domain}$$

기본값: $\tau = 0.35$

#### (4) 업데이트된 소스 집합 내 Mixup

재레이블된 타겟 샘플 $\mathbf{f}_j^t$와 원래 소스 샘플 $\mathbf{f}_i^s$를 혼합하여 피처 공간의 연속성 확보:

$$\widetilde{\mathbf{f}^s} = \mathcal{M}_\alpha(\mathbf{f}_i^s, \mathbf{f}_j^t) = \alpha \mathbf{f}_i^s + (1 - \alpha)\mathbf{f}_j^t$$

여기서 $\alpha \sim U(0, 1)$이며, 생성된 $\widetilde{\mathbf{f}^s}$는 소스 도메인 레이블로 학습에 사용된다.

#### (5) RADA 시작 조건

판별기의 판별 능력 개선이 $K$번의 에폭 동안 관찰되지 않을 때 RADA 활성화 (기본값: $K=5$).

### 2.3 모델 구조

```
입력 이미지
    │
    ▼
┌─────────────────────┐
│   Feature Extractor F│  (ResNet-50/101 backbone)
│   (GRL을 통해 역전파) │
└─────────────────────┘
    │
    ├──────────────────────┐
    ▼                      ▼
┌──────────┐        ┌─────────────────────┐
│ Object   │        │  Domain Discriminator│
│Classifier│        │        D            │
│    C     │        │  (재레이블 샘플 포함)│
└──────────┘        └─────────────────────┘
    │                      │
    ▼                      ▼
 L_cls                  L_adv
                           │
                     RADA 활성화 시
                    ┌──────────────┐
                    │ 타겟 샘플    │
                    │ 엔트로피 계산│
                    │ > τ → 재레이블│
                    │ Mixup 적용   │
                    └──────────────┘
```

**핵심 구조적 특징:**
- 네트워크 아키텍처 자체는 변경하지 않음
- GRL(Gradient Reversal Layer)을 통한 기존 적대적 학습 유지
- 미니배치 단위로 재레이블링 수행 (온라인 방식)

### 2.4 성능 향상

#### Office-31 (ResNet-50)

| Method | Avg. |
|--------|------|
| CDAN (Baseline) | 87.9% |
| SRDC (CVPR'20) | 90.8% |
| **CDAN+RADA** | **91.1%** |

#### Office-Home

| Method | Avg. |
|--------|------|
| CDAN (Baseline) | 68.1% |
| SRDC (CVPR'20) | 71.3% |
| **CDAN+RADA** | **71.4%** |

#### VisDA-2017 (합성→실제, 대규모 도메인 갭)

| Method | Avg. |
|--------|------|
| CDAN (Baseline) | 70.8% |
| GVB (CVPR'20) | 75.3% |
| **CDAN+RADA** | **76.3%** |

#### Digit-Five (다중 소스)

| Method | Avg. |
|--------|------|
| CDAN (Baseline) | 88.7% |
| CMSS (ECCV'20) | 90.8% |
| **CDAN+RADA** | **93.2%** |

#### DomainNet (다중 소스, 대규모)

| Method | Avg. |
|--------|------|
| CDAN (Baseline) | 45.2% |
| CMSS (ECCV'20) | 46.5% |
| **CDAN+RADA** | **47.5%** |

**에블레이션 결과:**

| Method | Office-31 | Office-Home | VisDA-2017 |
|--------|-----------|-------------|------------|
| DANN (Baseline) | 83.42% | 60.05% | 61.23% |
| DANN+RADA w/o MU | 85.24% | 63.14% | 65.91% |
| DANN+RADA | 86.79% | 64.81% | 67.29% |
| CDAN (Baseline) | 87.90% | 68.11% | 70.82% |
| CDAN+RADA w/o MU | 89.58% | 70.25% | 75.62% |
| **CDAN+RADA** | **91.08%** | **71.37%** | **76.28%** |

### 2.5 한계점

논문에서 명시적으로 언급된 한계 및 추론 가능한 한계:

1. **임계값 $\tau$의 민감성**: $\tau$ 설정에 따라 성능이 변화하며, 데이터셋에 따라 최적값이 다를 수 있음. 논문에서는 단순히 $\tau = 0.35$로 고정하여 사용
2. **단순한 정렬 측정 지표**: 엔트로피만을 정렬 측정 지표로 사용하며, 더 정확한 측정 지표 탐구는 향후 연구로 남김 (논문에서 직접 인정)
3. **클래스 수준 정렬 미고려**: 도메인 레벨 정렬에만 집중하며, 클래스별 정렬 상태는 고려하지 않음
4. **부정적 전이(Negative Transfer) 가능성**: 잘못 정렬된 샘플이 재레이블될 경우 오히려 학습을 방해할 수 있음
5. **하이퍼파라미터 의존**: $\tau$와 $K$ 두 하이퍼파라미터 튜닝 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 관련 이론적 근거

Ben-David et al. (2007)의 이론에 따르면, 타겟 도메인의 오류 상한은 다음과 같이 표현된다:

$$\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2} d_{\mathcal{H} \Delta \mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda^*$$

여기서 $d_{\mathcal{H} \Delta \mathcal{H}}$는 $\mathcal{H}$-divergence (도메인 간 피처 분포 차이), $\lambda^*$는 이상적인 결합 오류다. **RADA는 도메인 판별기를 지속적으로 재활성화하여 이 $d_{\mathcal{H} \Delta \mathcal{H}}$를 더욱 효과적으로 줄임**으로써 이론적으로 타겟 도메인 오류 상한을 낮춘다.

### 3.2 일반화 성능 향상의 구체적 메커니즘

**① 지속적인 적대적 최적화 유지**

- 기존 방법에서 도메인 갭이 줄어들면서 판별기가 약해지는 문제를 동적 재레이블로 해결
- 판별기가 지속적으로 강한 신호를 피처 추출기에 전달 → 더 완전한 도메인 불변(domain-invariant) 피처 학습

**② 대규모 도메인 갭에서의 효과성 강화**

VisDA-2017(합성→실제)에서 개선 폭이 가장 큼:
- DANN 기준: $61.23\% \rightarrow 67.29\%$ (+6.06%)
- CDAN 기준: $70.82\% \rightarrow 76.28\%$ (+5.46%)

이는 도메인 갭이 클수록 정렬되지 않은 샘플이 많이 남아있어 RADA의 재활성화 전략이 더 큰 효과를 발휘함을 의미한다.

**③ Mixup을 통한 피처 공간 연속성 확보**

재레이블된 타겟 샘플과 원래 소스 샘플 사이의 피처 공간 간극을 Mixup으로 채워 **더 매끄러운(smooth) 피처 공간** 형성 → 보지 못한 데이터에 대한 일반화 향상

$$\widetilde{\mathbf{f}^s} = \alpha \mathbf{f}_i^s + (1-\alpha)\mathbf{f}_j^t, \quad \alpha \sim U(0,1)$$

**④ 플러그앤플레이 범용성**

- DANN, CDAN 등 다양한 베이스라인에 적용 가능
- 단일/다중 소스 UDA 모두에서 일관된 성능 향상
- t-SNE 시각화(Figure 5)에서 도메인 간 정렬 및 클래스 간 분리가 더 명확함을 확인

**⑤ 다중 소스 도메인 적응에서의 일반화**

Digit-Five, DomainNet에서도 SOTA 달성 → 단순히 단일 쌍의 도메인에 과적합되지 않고 다양한 도메인 조합에서 일반화됨을 증명

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

**① 동적 레이블 패러다임의 확립**

RADA는 UDA에서 정적 레이블의 한계를 지적하고 **동적 도메인 레이블** 개념을 도입했다. 이는 이후 연구에서 소프트 레이블(soft label), 확률적 레이블 등으로 확장될 수 있는 새로운 패러다임을 제시한다.

**② 판별기 약화 문제 인식의 확산**

기존 연구들이 간과했던 "판별기 성능 저하 → 최적화 정체"의 악순환을 명확히 규명함으로써, 이후 연구들이 판별기 학습 안정성을 더 주목하게 되는 계기 제공

**③ 플러그앤플레이 최적화 전략의 방향 제시**

네트워크 구조 변경 없이 학습 전략만으로 성능을 높이는 접근법은, 구조 설계 연구와 독립적으로 적용 가능한 **최적화 수준의 개선 전략** 연구를 촉진

**④ 엔트로피 기반 샘플 선택의 재조명**

CDAN이 엔트로피를 샘플 재가중치에 활용한 것을 넘어, RADA는 엔트로피를 **레이블 변경 기준**으로 활용 → 엔트로피의 다양한 활용 가능성 제시

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 논문 내 언급된 2020년 연구들을 중심으로 분석하며, 2020년 이후 발표된 외부 논문들에 대해서는 제가 학습한 지식 기반으로 서술하되, 논문 PDF에 직접 인용되지 않은 내용은 명확히 구분합니다.

#### 논문 내 언급된 2020년 연구와의 비교

| 논문 | 핵심 전략 | RADA와의 관계 |
|------|-----------|---------------|
| **GVB** (Cui et al., CVPR'20) | 점진적으로 도메인 특성 감소 (Gradually Vanishing Bridge) | 도메인 갭 축소 방향은 유사하나 판별기 재활성화 미고려 |
| **CMSS** (Yang et al., ECCV'20) | 커리큘럼 기반 소스 샘플 선택 | 소스 샘플 선택 vs. 타겟 샘플 재레이블 - 방향이 반대 |
| **SRDC** (Tang et al., CVPR'20) | 클러스터링 기반 구조 정규화 | 클래스 수준 정렬 vs. 도메인 수준 재활성화 - 상호 보완적 |
| **BNM** (Cui et al., CVPR'20) | 배치 핵 놈 최대화로 판별성 유지 | 피처 다양성 유지 측면에서 유사한 문제의식 |

#### 2020년 이후 연구 동향과의 연관성 (일반 지식 기반)

아래는 제가 학습한 지식에 기반한 내용으로, **논문 PDF에 직접 인용된 내용이 아님**을 명시합니다:

- **CDTrans** (Xu et al., ICCV 2021): Transformer 기반 도메인 적응으로, RADA의 CNN 기반 접근을 Transformer로 확장하는 연구 방향과 연결
- **PMTrans** (Zhu et al., ECCV 2022): 패치 기반 혼합 Transformer UDA - RADA의 Mixup 전략과 Transformer의 결합 가능성
- **SPA** 및 **TVT** 등 Transformer 기반 UDA: RADA의 플러그앤플레이 특성상 백본을 Transformer로 교체해도 적용 가능할 것으로 예상

### 4.3 향후 연구 시 고려할 점

**① 더 정교한 정렬 측정 지표 개발**

현재 엔트로피만을 사용하는 단순한 측정 방식을 다음으로 발전:
- 클래스 조건부 정렬 측정 (class-conditional alignment)
- 프로토타입 기반 거리 측정
- 상호정보량(mutual information) 활용

**② 클래스 수준 정렬과의 통합**

RADA는 도메인 수준(binary) 정렬에 집중하므로, **클래스 구분을 고려한 재레이블링** 전략 연구 필요:

$$\mathcal{H}_{cls}(\mathbf{p}) = -\sum_{c=1}^{C} p_c \log p_c$$

클래스 예측 엔트로피와 도메인 판별 엔트로피를 결합한 기준 개발

**③ 부정적 전이 방지 메커니즘**

잘못된 재레이블링으로 인한 부정적 전이(negative transfer) 방지를 위해:
- 의사 레이블(pseudo-label) 신뢰도와 연계
- 재레이블 비율 적응적 조절 (adaptive $\tau$)

**④ Transformer 기반 UDA로의 확장**

Vision Transformer(ViT) 기반 UDA 프레임워크에 RADA 적용 시 self-attention 메커니즘과의 상호작용 분석 필요

**⑤ 소프트 재레이블링 탐구**

현재의 하드(hard) 재레이블링($0$ 또는 $1$) 대신, 도메인 분류 확률을 활용한 소프트 재레이블링:

$$\tilde{y}_j^{domain} = \beta \cdot 0_{\text{target}} + (1-\beta) \cdot 1_{\text{source}}, \quad \beta = 1 - \mathcal{H}(\mathbf{p}_j^t) / \log 2$$

**⑥ 오픈셋(Open-Set) 및 부분 도메인 적응으로의 확장**

소스·타겟 도메인의 클래스 집합이 다른 경우(partial DA, open-set DA)에서의 RADA 적용 가능성 및 재레이블 기준 재설계 필요

**⑦ 대규모 언어-비전 모델(CLIP 등)과의 결합**

최근 CLIP 기반 UDA 연구에서, RADA의 동적 레이블 전략과 텍스트-이미지 정렬을 결합한 더 풍부한 도메인 정렬 가능성 탐구

---

## 📊 종합 정리

```
RADA의 핵심 사이클:

[도메인 정렬 진행]
        ↓
[판별기 판별 능력 저하] ← 기존 방법의 문제
        ↓
[RADA: 엔트로피 기반 재레이블링]
  H(p) > τ → 타겟 샘플을 소스로 재레이블
        ↓
[분포 간 분리도 증가]
        ↓
[판별기 재활성화 (re-energizing)]
        ↓
[피처 추출기에 강한 정렬 신호 전달]
        ↓
[더 완전한 도메인 피처 정렬]
        ↓
[타겟 도메인 일반화 성능 향상]
```

RADA는 구현의 단순성, 범용성, 이론적 타당성을 모두 갖추며 적대적 UDA의 최적화 효율성을 근본적으로 개선한 중요한 연구로, 향후 동적 레이블 기반 도메인 적응 연구의 중요한 출발점이 될 것으로 평가된다.
