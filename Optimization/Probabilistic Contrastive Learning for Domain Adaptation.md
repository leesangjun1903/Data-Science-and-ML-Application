# Probabilistic Contrastive Learning for Domain Adaptation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문의 핵심 주장은 다음과 같습니다:

> **표준 Feature Contrastive Learning(FCL)이 도메인 적응(Domain Adaptation)에서 효과가 제한적인 근본 원인은, 대조 학습 최적화 과정에서 분류기의 클래스 가중치(class weights, 마지막 FC 레이어의 가중치)가 고려되지 않아, 타겟 도메인의 피처가 대응하는 클래스 가중치에 가까이 클러스터링되지 못하는 편차(deviation) 문제에 있다.**

이를 해결하기 위해 **Probabilistic Contrastive Learning (PCL)** 을 제안합니다. PCL은 피처 대신 **확률(probability)** 을 사용하고, $\ell_2$ 정규화를 제거하는 두 가지 단순하지만 강력한 변경을 핵심으로 합니다.

### 주요 기여

1. **문제 정의의 새로운 관점**: FCL의 도메인 적응 실패 원인을 "피처-클래스 가중치 편차(feature-class weight deviation)"로 최초 명시
2. **PCL 방법론 제안**: 확률 기반 대조 학습이라는 간결하고 강력한 self-supervised 패러다임 제시
3. **광범위한 일반화 검증**: UDA 분류, SSDA, SSL, UDA 객체 탐지, UDA 의미론적 분할의 **5개 태스크**에서 일관된 성능 향상 증명

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

#### 문제의 발단: 표준 FCL의 한계

표준 FCL의 InfoNCE 손실은 다음과 같이 정의됩니다:

$$\ell_{\mathbf{f}_i} = -\log \frac{\exp(sg(\mathbf{f}_i)^\top g(\tilde{\mathbf{f}}_i))}{\sum_{j \neq i} \exp(sg(\mathbf{f}_i)^\top g(\mathbf{f}_j)) + \sum_k \exp(sg(\mathbf{f}_i)^\top g(\tilde{\mathbf{f}}_k))}$$

여기서 $g(\mathbf{f}) = \frac{\mathbf{f}}{\|\mathbf{f}\|_2}$ 는 $\ell_2$ 정규화, $s$는 스케일링 인수입니다.

**핵심 문제점**: 이 손실 함수에는 **클래스 가중치(class weights) 정보가 전혀 포함되지 않습니다.** 따라서 FCL로 피처의 판별력(discriminability)은 높아지더라도, 타겟 도메인의 피처가 소스 도메인에서 학습된 클래스 가중치로부터 여전히 벗어날 수 있습니다.

#### 실험적 근거

논문의 Figure 2에서:
- **기준(Baseline, MME)**: 64.3%
- **+FCL**: 64.5% (단 0.2% 향상)
- **FCL 후 GT로 분류기만 재학습 시**: 80.7% (6.1% 향상 가능성 확인)

이 실험 결과는 FCL이 피처 자체의 판별력은 향상시키지만, **피처와 클래스 가중치 사이의 편차가 여전히 크게 남아있음**을 명확히 보여줍니다.

### 2-2. 제안 방법 및 수식

#### 핵심 관찰: 확률과 One-hot의 관계

클래스 가중치 집합을 $W = (\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_C)$, 피처 벡터를 $\mathbf{f}_i$, 분류 확률을 $\mathbf{p}_i$라 할 때, $c$번째 클래스에 대한 확률은:

$$p_{i,c} = \frac{\exp(\mathbf{w}_c^\top \mathbf{f}_i)}{\sum_{j \neq c} \exp(\mathbf{w}_j^\top \mathbf{f}_i) + \exp(\mathbf{w}_c^\top \mathbf{f}_i)}$$

**핵심 통찰**: $\mathbf{f}\_i$가 해당 클래스 가중치 $\mathbf{w}\_c$에 가까울수록, $p_{i,c} \rightarrow 1$이 되고 $\{p_{i,j}\}_{j \neq c} \rightarrow 0$이 됩니다. 즉, **피처가 클래스 가중치에 가까울수록 확률 벡터는 one-hot 형태에 가까워집니다.**

#### 확률 내적의 수학적 성질

확률 벡터 $\mathbf{p}\_i = (p_{i,1}, \ldots, p_{i,C})$와 $\tilde{\mathbf{p}}\_i = (\tilde{p}\_{i,1}, \ldots, \tilde{p}_{i,C})$에 대해:

$$0 \leq p_{i,c} \leq 1, \quad 0 \leq \tilde{p}_{i,c} \leq 1, \quad \forall c \in \{1, \ldots, C\}$$

$$\|\mathbf{p}_i\|_1 = \sum_c p_{i,c} = 1, \quad \|\tilde{\mathbf{p}}_i\|_1 = \sum_c \tilde{p}_{i,c} = 1$$

따라서:

$$\mathbf{p}_i^\top \tilde{\mathbf{p}}_i = \sum_c p_{i,c} \tilde{p}_{i,c} \leq 1$$

**등호 조건**: $\mathbf{p}_i = \tilde{\mathbf{p}}_i$이고 두 벡터 모두 one-hot 형태일 때만 성립합니다.

이 성질이 PCL의 핵심 수학적 근거입니다. **$\mathbf{p}_i^\top \tilde{\mathbf{p}}_i$를 최대화하는 것이 곧 확률 벡터를 one-hot에 가깝게 만드는 것이며, 이는 피처를 클래스 가중치에 가깝게 만드는 것과 동치입니다.**

> **중요**: 이 $\ell_1$-norm 성질이 유지되어야 하므로, FCL처럼 $\ell_2$ 정규화를 적용하면 이 성질이 깨집니다. 이것이 $\ell_2$ 정규화를 제거해야 하는 이유입니다.

#### PCL 손실 함수 (최종)

$$\ell_{\mathbf{p}_i} = -\log \frac{\exp(s \mathbf{p}_i^\top \tilde{\mathbf{p}}_i)}{\sum_{j \neq i} \exp(s \mathbf{p}_i^\top \mathbf{p}_j) + \sum_k \exp(s \mathbf{p}_i^\top \tilde{\mathbf{p}}_k)}$$

**FCL 대비 두 가지 핵심 차이점**:
1. 피처 $\mathbf{f}_i$ 대신 **확률 $\mathbf{p}_i$** 사용 (Softmax 출력)
2. **$\ell_2$ 정규화 $g(\cdot)$ 제거**

#### 보조 정규화 손실 (UDA/SSDA에서)

FixMatch 적용 시 잘못 예측된 고신뢰도 샘플 문제를 완화하기 위한 정규화 항:

$$\ell_{reg} = -\sum_{i=1}^{N} \sum_{j=1}^{C} \frac{1}{C} \log p_t^{(i,j)}$$

### 2-3. 모델 구조

```
[입력 이미지 x_i] ──→ [Encoder E] ──→ [Feature f_i]
                                            │
[입력 이미지 x̃_i] ──→ [Encoder E] ──→ [Feature f̃_i]
                                            │
                              [Classifier F (FC Layer, W)]
                                            │
                              [Softmax] ──→ [확률 p_i, p̃_i]
                                            │
                              [PCL Loss 계산: p_i⊤p̃_i 최대화]
```

**모델 구성요소**:
- **Encoder $E$**: ResNet-34/50/101 등 백본 네트워크
- **Classifier $F$**: 가중치 $W = (\mathbf{w}_1, \ldots, \mathbf{w}_C)$를 가진 FC 레이어
- **Softmax**: 확률 벡터 생성
- **PCL 손실**: 확률 벡터 간 내적 기반 InfoNCE

**FCL vs PCL 비교**:

| 구분 | FCL | PCL |
|------|-----|-----|
| 입력 | 피처 $\mathbf{f}_i$ | 확률 $\mathbf{p}_i$ |
| 정규화 | $\ell_2$ 정규화 | 없음 ($\ell_1$-norm 자연 보존) |
| 클래스 가중치 | 무관 | 내재적 포함 |
| 손실 형태 | InfoNCE | InfoNCE |

### 2-4. 성능 향상

#### UDA 의미론적 분할 (SYNTHIA → Cityscapes)

| 방법 | mIoU-13 | mIoU-16 | 특이사항 |
|------|---------|---------|---------|
| BAPA* (재구현) | 60.1% | 53.3% | 비-증류 |
| **BAPA* + PCL** | **68.2%** | **60.3%** | 비-증류 |
| CPSL-D (CVPR'22) | 65.3% | 57.9% | 4×V100, 11일 |
| ProDA-D (CVPR'21) | 62.0% | 55.5% | 증류 사용 |

→ PCL은 증류 기법 없이 CPSL-D를 **mIoU-13 기준 +2.9%** 초과 달성, 훈련 비용은 대폭 절감 (1×3090, 5일)

#### SSDA (DomainNet, 3-shot, ResNet34)

| 방법 | Mean Acc |
|------|---------|
| MME (기준) | 69.5% |
| +FCL | 71.3% (+1.8%) |
| +PCL (본 논문) | 76.9% (+7.4%) |
| CLDA (NeurIPS'21) | 75.3% |
| ECACL-P† (ICCV'21) | 76.4% |
| **MME† + PCL** | **78.2%** |

#### UDA 분류 (Office-Home)

| 방법 | Avg Acc |
|------|---------|
| GVB* (CVPR'20) | 70.3% |
| +MetaAlign (CVPR'21) | 71.3% |
| **+PCL** | **72.3%** |
| GVB† + PCL | **74.5%** |

#### UDA 객체 탐지 (SIM10k → Cityscapes)

| 방법 | AP |
|------|---|
| RPA (CVPR'21) | 45.3% |
| **+PCL** | **47.8%** |

#### SSL (CIFAR-100, 400 labels)

| 방법 | Acc |
|------|-----|
| FixMatch* | 53.58% |
| +PCL | 57.62% |
| FlexMatch* | 61.78% |
| **+PCL** | **64.15%** |

### 2-5. 한계

논문이 명시적으로 언급하거나 분석에서 파악 가능한 한계:

1. **일반 비지도 표현 학습(GCL)으로의 직접 확장 어려움**: 비지도 사전학습 단계에서는 분류기가 없으므로 PCL을 직접 적용 불가. 저자들도 클러스터링 알고리즘 활용이 필요하다고 언급
2. **하이퍼파라미터 $s$ 의존성**: 태스크마다 스케일링 인수 $s$를 조정해야 함 (분류: $s=7$, 분할·탐지: $s=20$)
3. **사전 훈련된 분류기 필요**: PCL은 소스 도메인에서 학습된 클래스 가중치를 필요로 하므로, 완전 비지도 시나리오에서는 초기 클래스 가중치 품질이 성능에 영향
4. **False Negative 문제는 여전히 존재**: SPCL(Supervised PCL)로 개선 가능하나 복잡도 증가로 채택 안 함 (76.9% → 77.2% 미미한 향상)
5. **멀티-소스 도메인 적응 미검증**: 단일 소스에서 단일 타겟으로의 전이에만 집중

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. PCL이 일반화 성능을 향상시키는 메커니즘

PCL의 일반화 성능 향상은 다음 세 가지 관점에서 설명됩니다:

**① 피처-클래스 가중치 정합(Feature-Weight Alignment)**

$$\text{PCL이 작아질수록} \Leftrightarrow \mathbf{p}_i^\top \tilde{\mathbf{p}}_i \text{가 최대} \Leftrightarrow \mathbf{p}_i \text{가 one-hot} \Leftrightarrow \mathbf{f}_i \approx \mathbf{w}_{y_i}$$

소스 도메인에서 학습된 클래스 가중치 $\mathbf{w}_c$에 타겟 도메인 피처 $\mathbf{f}_i$를 가까이 당김으로써, 소스에서 타겟으로의 지식 이전이 더 효과적으로 이루어집니다.

**② InfoNCE 형태 유지를 통한 컴팩트 피처 학습**

단순 엔트로피 최소화(BCE, MaxSquares)는 개별 샘플에 독립적으로 작용하지만, PCL의 InfoNCE 형태는 **다른 샘플 간의 관계를 동시에 고려**하여 피처 공간의 클러스터 구조를 강화합니다:

$$\mathcal{L}_{PCL} = \mathbb{E}\left[-\log \frac{\exp(s \mathbf{p}_i^\top \tilde{\mathbf{p}}_i)}{\sum_{j \neq i} \exp(s \mathbf{p}_i^\top \mathbf{p}_j) + \sum_k \exp(s \mathbf{p}_i^\top \tilde{\mathbf{p}}_k)}\right]$$

이 형태는 동일 클래스 샘플은 가깝게, 다른 클래스 샘플은 멀게 학습하는 **강력한 대조 신호**를 제공합니다.

**③ 도메인 불변 클래스 경계 학습**

t-SNE 시각화(Figure 4)에서 확인되듯이:
- **MME**: 피처 중심과 클래스 가중치 간 큰 편차
- **MME+FCL**: 피처 클러스터는 개선되나 여전히 클래스 가중치와 편차 존재
- **MME+PCL**: 클래스 가중치가 피처 중심에 정확히 위치

### 3-2. 태스크 독립적 일반화 능력

| 태스크 | 베이스라인 | +PCL | 향상 |
|--------|----------|------|------|
| UDA 분류 (VisDA) | 80.4% | 82.5% | +2.1% |
| UDA 분할 (SYNTHIA mIoU-13) | 60.1% | 68.2% | +8.1% |
| UDA 탐지 (AP) | 45.3% | 47.8% | +2.5% |
| SSDA (DomainNet 3-shot) | 69.5% | 76.9% | +7.4% |
| SSL (CIFAR-100 FlexMatch) | 61.78% | 64.15% | +2.37% |

SSL에서도 효과가 나타나는 것은 **PCL이 도메인 적응에 국한되지 않고, "레이블 부족 상황에서 피처와 클래스 가중치 편차가 발생하는 모든 시나리오"** 에 적용 가능함을 보여줍니다.

### 3-3. 기존 방법과의 결합 시 상승 효과

PCL은 독립 모듈로서 기존 방법에 추가(plug-in)하는 방식으로 동작합니다:

$$\mathcal{L}_{total} = \mathcal{L}_{baseline} + \lambda \cdot \mathcal{L}_{PCL}$$

FixMatch와 결합 시 (MME† + PCL) 상승 효과가 특히 두드러집니다. 이는 PCL의 피처-가중치 정합 능력이 의사 레이블(pseudo-label) 방법의 정확도를 높이는 선순환을 만들기 때문입니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4-1. 도메인 적응 분야 최신 연구 비교

| 논문 | 발표 | 핵심 방법 | PCL 대비 차이점 |
|------|-----|---------|--------------|
| **PCL (본 논문)** | arXiv 2021 (v6: 2024) | 확률 기반 대조 학습 | 기준 논문 |
| **MetaAlign** (Wei et al., CVPR'21) | CVPR 2021 | 메타 최적화로 도메인 정렬-분류 조율 | PCL보다 복잡, 성능 하위 |
| **CPSL-D** (Li et al., CVPR'22) | CVPR 2022 | 클래스 균형 픽셀 자기레이블링 + 증류 | 증류로 복잡, SYNTHIA에서 PCL이 +2% 우세 |
| **CaCo** (Huang et al., CVPR'22) | CVPR 2022 | 카테고리 대조 학습 (UDA) | ProDA-D와 결합, PCL과 방향 유사하나 복잡 |
| **CLDA** (Singh, NeurIPS'21) | NeurIPS 2021 | 분류기를 프로젝션 헤드로 사용한 대조 학습 | $\ell_2$ 정규화 유지, 원리적으로 PCL과 다름 |
| **SDAT** (Rangwani et al., ICML'22) | ICML 2022 | 도메인 적대 학습의 스무딩 개선 | PCL (72.3%) < SDAT (72.2%) 근사 수준 |
| **NWD** (Chen et al., CVPR'22) | CVPR 2022 | 판별기를 분류기로 재사용 | PCL과 유사한 동기, 다른 구현 |
| **SLA** (Yu & Lin, CVPR'23) | CVPR 2023 | 소스 레이블 적응 SSDA | DomainNet에서 MME†+PCL(78.2%)이 우세 |
| **ProMM** (Huang et al., IJCAI'23) | IJCAI 2023 | 프로토타입 기반 멀티레벨 학습 SSDA | 77.4% vs PCL 76.9% (FixMatch 없이) |
| **HMA** (Zhou et al., ICCV'23) | ICCV 2023 | 위상동형 정렬 UDA | Office-Home 73.2% vs PCL(GVB†) 74.5% |

### 4-2. 방법론적 패러다임 비교

```
도메인 정렬 계열 (MMD, GAN 기반)
    ├── 장점: 이론적 보장
    └── 단점: 학습 불안정, 클래스 구조 무시

자기지도 대조 학습 계열 (SimCLR, MoCo 기반)
    ├── FCL: 피처 판별성 ↑, 클래스 가중치 편차 미해결
    ├── CLDA: 분류기 투영 헤드 + ℓ₂ (PCL-ℓ₂ 형태)
    └── PCL (본 논문): 확률 기반 + ℓ₂ 제거 ← 핵심 기여

의사 레이블 계열 (Self-training)
    ├── ProDA, CPSL: 증류 기법 활용
    └── FixMatch 계열: 신뢰도 임계값 기반

혼합 접근 (PCL + FixMatch)
    └── 성능 최고, 두 방법의 상호 보완
```

### 4-3. 대조 학습 발전 흐름과 PCL의 위치

- **SimCLR** (Chen et al., ICML'20): 프로젝션 헤드 + $\ell_2$, 도메인 적응 미고려
- **SupCon** (Khosla et al., NeurIPS'20): 지도 대조 학습, False Negative 해결
- **CLDA** (Singh, NeurIPS'21): 도메인 적응 + 분류기 프로젝션 헤드, 여전히 $\ell_2$ 유지
- **PCL** (본 논문): 확률 사용 + $\ell_2$ 제거, 피처-가중치 편차 문제 명시적 해결
- **CaCo** (Huang et al., CVPR'22): 카테고리 수준 대조 학습, 프로토타입 활용

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 연구에 미치는 영향

**① 새로운 분석 프레임워크 제공**

"피처-클래스 가중치 편차"라는 관점은 향후 도메인 적응 연구에서 방법론을 평가하는 새로운 기준이 될 수 있습니다. 어떤 방법이든 이 편차를 얼마나 효과적으로 줄이는지 측정하는 것이 중요한 평가 지표가 될 수 있습니다.

**② 플러그인 모듈로서의 가치**

PCL은 기존 모든 방법에 추가 가능한 모듈로 설계되어, 향후 연구들이 PCL을 기본 구성요소로 채택하고 그 위에 새로운 기여를 쌓는 방식의 연구가 증가할 것으로 예상됩니다.

**③ 확률 공간 표현 학습의 재조명**

피처 공간이 아닌 **확률 공간에서의 대조 학습**이라는 아이디어는, 분류기가 있는 모든 지도/반지도 학습 시나리오에 적용 가능한 일반 원리를 제시합니다.

**④ 비지도 표현 학습과의 연결**

저자들이 제안한 클러스터링 알고리즘과의 결합을 통한 일반 비지도 학습으로의 확장은, 향후 **클러스터링 기반 자기지도 학습(Self-supervised Learning with Clustering)** 연구의 중요한 방향이 됩니다.

### 5-2. 향후 연구 시 고려할 점

**① 소프트 레이블 환경으로의 확장**

현재 PCL은 hard one-hot 수렴을 목표로 합니다. 레이블 노이즈나 클래스 간 의미적 유사성이 있는 경우, **소프트 타겟(soft target)** 을 활용한 PCL 변형을 고려해야 합니다:

$$\mathbf{p}_i^{target} = (1-\epsilon)\mathbf{e}_{y_i} + \frac{\epsilon}{C}\mathbf{1}$$

여기서 $\epsilon$은 레이블 스무딩 계수입니다.

**② 클래스 불균형 문제**

PCL이 확률을 직접 사용하므로, 클래스 불균형이 심한 경우 다수 클래스 확률이 지배적이 될 수 있습니다. 클래스 빈도를 고려한 가중 PCL 손실의 설계가 필요합니다.

**③ 비지도 사전학습과의 통합**

Vision Transformer(ViT) 기반의 **DINO, MAE** 등과 PCL의 통합 가능성을 탐색해야 합니다. 이 경우 분류기 없이 작동하는 초기 단계에서 PCL을 어떻게 적용할지가 핵심 과제입니다.

**④ 멀티모달 도메인 적응으로의 확장**

텍스트-이미지, 음성-텍스트 등 멀티모달 시나리오에서 **각 모달리티의 분류기 가중치를 공유하거나 정렬**하는 방식으로 PCL을 확장하면 CLIP 류 모델의 도메인 적응에 활용 가능합니다.

**⑤ 이론적 보장 강화**

현재 논문은 주로 실험적 검증에 의존합니다. PCL이 도메인 갭을 줄이는 이론적 상한/하한을 도메인 적응 이론(Ben-David 등의 $\mathcal{H}$-divergence 이론)과 연결하여 분석하는 연구가 필요합니다.

**⑥ 동적 스케일링 인수 $s$의 자동화**

현재 $s$는 수동으로 설정됩니다. 학습 과정에서 피처 품질에 따라 $s$를 적응적으로 조정하는 메커니즘이 성능을 더욱 향상시킬 수 있습니다.

**⑦ 대규모 클래스 수 시나리오**

PCL은 확률 벡터의 차원이 클래스 수 $C$에 비례합니다. $C$가 매우 큰 경우(예: 1000개 이상 클래스) 계산 효율성과 수치 안정성 문제를 해결해야 합니다.

---

## 참고 자료

**주요 참고 논문 (본 문서 작성에 직접 사용)**

1. **Li et al. (2024)** - "Probabilistic Contrastive Learning for Domain Adaptation" - arXiv:2111.06021v6, 8 Jun 2024 *(본 분석의 주 대상 논문)*
2. **Chen et al. (ICML 2020)** - "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)"
3. **Khosla et al. (NeurIPS 2020)** - "Supervised Contrastive Learning"
4. **Oord et al. (2018)** - "Representation Learning with Contrastive Predictive Coding" (InfoNCE 원논문)
5. **Saito et al. (ICCV 2019)** - "Semi-supervised Domain Adaptation via Minimax Entropy (MME)"
6. **Singh (NeurIPS 2021)** - "CLDA: Contrastive Learning for Semi-supervised Domain Adaptation"
7. **Li et al. (CVPR 2022)** - "Class-balanced Pixel-level Self-labeling for Domain Adaptive Semantic Segmentation (CPSL)"
8. **Zhang et al. (CVPR 2021)** - "Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation (ProDA)"
9. **Huang et al. (CVPR 2022)** - "Category Contrast for Unsupervised Domain Adaptation in Visual Tasks (CaCo)"
10. **Sohn et al. (NeurIPS 2020)** - "FixMatch: Simplifying Semi-supervised Learning with Consistency and Confidence"
11. **Rangwani et al. (ICML 2022)** - "A Closer Look at Smoothness in Domain Adversarial Training (SDAT)"
12. **Zhou et al. (ICCV 2023)** - "Homeomorphism Alignment for Unsupervised Domain Adaptation (HMA)"
13. **Yan et al. (IJCAI 2022)** - "Multi-level Consistency Learning for Semi-supervised Domain Adaptation (MCL)"
14. **Huang et al. (IJCAI 2023)** - "Semi-supervised Domain Adaptation via Prototype-based Multi-level Learning (ProMM)"
15. **Yu and Lin (CVPR 2023)** - "Semi-supervised Domain Adaptation with Source Label Adaptation (SLA)"
