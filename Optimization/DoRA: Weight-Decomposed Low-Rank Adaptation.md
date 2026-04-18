# DoRA: Weight-Decomposed Low-Rank Adaptation

---

## 📌 참고 자료 (출처)

> **주 논문:**
> - Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., & Chen, M.-H. (2024). **DoRA: Weight-Decomposed Low-Rank Adaptation**. *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, PMLR 235. arXiv:2402.09353v6.
>
> **관련 비교 논문 (논문 내 인용 기반):**
> - Hu et al. (2022). **LoRA: Low-Rank Adaptation of Large Language Models**. ICLR 2022.
> - Kopiczko et al. (2024). **VeRA: Vector-based Random Matrix Adaptation**. ICLR 2024.
> - Dettmers et al. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs**. NeurIPS 2023.
> - Houlsby et al. (2019). **Parameter-Efficient Transfer Learning for NLP**. ICML 2019.
> - Salimans & Kingma (2016). **Weight Normalization**. NeurIPS 2016.
> - Zhang et al. (2023). **AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning**. ICLR 2023.
> - He et al. (2021). **Towards a Unified View of Parameter-Efficient Transfer Learning**. ICLR 2021.
> - Sung et al. (2022). **VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks**. CVPR 2022.
> - Liu et al. (2023a). **Visual Instruction Tuning (LLaVA)**. NeurIPS 2023.
> - Touvron et al. (2023). **LLaMA: Open and Efficient Foundation Language Models**. arXiv:2302.13971.

---

## 1. 핵심 주장과 주요 기여 요약

### 1.1 핵심 주장

DoRA는 사전학습된 가중치(pre-trained weight)를 **크기(magnitude)** 와 **방향(direction)** 두 성분으로 분해하여 파인튜닝함으로써, LoRA와 Full Fine-Tuning(FT) 사이의 **학습 능력 격차(capacity gap)** 를 체계적으로 해소할 수 있다고 주장합니다.

핵심 통찰은 다음과 같습니다:

> *"LoRA는 magnitude 변화와 direction 변화 사이에 강한 양(+)의 상관관계(r=0.83)를 보이는 반면, FT는 음(-)의 상관관계(r=-0.62)를 보인다. DoRA는 이 패턴을 FT에 가깝게 재현(r=-0.31)함으로써 더 세밀한 학습이 가능하다."*

### 1.2 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **① 새로운 분석 도구** | Weight Decomposition Analysis: FT와 LoRA의 학습 패턴 차이를 magnitude/direction 관점에서 최초로 정량 분석 |
| **② DoRA 방법 제안** | 추가 추론 비용 없이 FT에 근접한 학습 능력을 달성하는 새로운 PEFT 방법 |
| **③ 광범위한 검증** | NLP(LLaMA 계열), Vision-Language(LLaVA, VL-BART), Text-to-Image(SDXL) 등 다양한 도메인에서 LoRA 대비 일관된 성능 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**배경:**
대규모 사전학습 모델(LLM, LVLM)을 특정 태스크에 적응시키기 위해 Full Fine-Tuning(FT)을 수행하면 막대한 컴퓨팅 비용이 발생합니다. LoRA는 이를 해결하지만, FT와의 **정확도 격차**가 여전히 존재하며, 기존 연구들은 이를 단순히 "훈련 가능 파라미터 수의 부족" 때문이라고만 설명해 왔습니다.

**DoRA가 새롭게 발견한 문제:**
LoRA의 학습 패턴 자체가 FT와 본질적으로 다르다는 것입니다:

- **LoRA**: $\Delta D$ (방향 변화)와 $\Delta M$ (크기 변화) 사이 **양의 상관관계** → magnitude와 direction을 동시에 학습해야 해서 최적화가 복잡
- **FT**: **음의 상관관계** → 큰 방향 변화 시 작은 크기 변화, 또는 그 반대가 가능 → 더 세밀하고 효율적인 적응

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: Weight Decomposition (가중치 분해)

임의의 가중치 행렬 $W \in \mathbb{R}^{d \times k}$를 다음과 같이 분해합니다:

$$W = m \frac{V}{\|V\|_c} = \|W\|_c \frac{W}{\|W\|_c} $$

- $m \in \mathbb{R}^{1 \times k}$: **magnitude vector** (각 열 벡터의 크기)
- $V \in \mathbb{R}^{d \times k}$: **directional matrix** (방향 행렬)
- $\|\cdot\|_c$: 행렬의 각 열(column) 방향 벡터 노름(vector-wise norm)
- $V/\|V\|_c$의 각 열은 단위 벡터(unit vector)가 됩니다.

#### Step 2: LoRA 수식 (비교 기준)

$$W' = W_0 + \Delta W = W_0 + \underline{B}\underline{A} $$

- $W_0 \in \mathbb{R}^{d \times k}$: 동결된 사전학습 가중치
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$: 훈련 가능한 저랭크 행렬 ( $r \ll \min(d,k)$ )
- 밑줄: 훈련 가능 파라미터

#### Step 3: DoRA 공식 (핵심)

$$\boxed{W' = \underline{m} \frac{V + \Delta V}{\|V + \Delta V\|_c} = \underline{m} \frac{W_0 + \underline{B}\underline{A}}{\|W_0 + \underline{B}\underline{A}\|_c}} $$

- $\underline{m}$: 훈련 가능한 magnitude vector (초기값: $\|W_0\|_c$)
- $V = W_0$: 동결된 방향 행렬 (초기화)
- $\Delta V = \underline{B}\underline{A}$: LoRA로 학습되는 방향성 업데이트
- $B$, $A$: LoRA와 동일한 초기화 방식 적용 → 초기에 $W' = W_0$ 보장

**훈련 가능 파라미터:** $m$ (크기, $1 \times k$) + $B$, $A$ (방향, LoRA)

#### Step 4: 분석을 위한 크기/방향 변화 측정 수식

FT 가중치의 magnitude 변화:

$$\Delta M^t_{\text{FT}} = \frac{\sum_{n=1}^{k} |m^{n,t}_{\text{FT}} - m^n_0|}{k} $$

FT 가중치의 direction 변화:

$$\Delta D^t_{\text{FT}} = \frac{\sum_{n=1}^{k}(1 - \cos(V^{n,t}_{\text{FT}}, W^n_0))}{k} $$

#### Step 5: DoRA의 그래디언트 분석

Loss $\mathcal{L}$에 대한 DoRA의 그래디언트 (Eq. 5로부터 유도):

$$\nabla_{V'}\mathcal{L} = \frac{m}{\|V'\|_c}\left(I - \frac{V'V'^T}{\|V'\|^2_c}\right)\nabla_{W'}\mathcal{L} $$

$$\nabla_m \mathcal{L} = \frac{\nabla_{W'}\mathcal{L} \cdot V'}{\|V'\|_c} $$

**해석:**
- Eq. (6): 그래디언트가 $m/\|V'\|_c$로 스케일링되고 현재 가중치 방향으로부터 투영(projection)됨 → 그래디언트 공분산 행렬이 단위행렬에 가까워져 최적화 유리
- $V' = V + \Delta V$이므로 $\nabla_{V'}\mathcal{L} = \nabla_{\Delta V}\mathcal{L}$ → LoRA 학습 안정성 향상

#### Step 6: 훈련 메모리 절감 (실용적 수정)

$\|V + \Delta V\|_c$를 그래디언트 그래프에서 분리(detach)하여 상수 $C$로 처리:

$$\nabla_{V'}\mathcal{L} = \frac{m}{C}\nabla_{W'}\mathcal{L} \quad \text{where } C = \|V'\|_c $$

→ LLaMA 파인튜닝 시 GPU 메모리 **24.4% 절감**, VL-BART 시 **12.4% 절감**, 정확도 손실은 무시할 수 있는 수준 (LLaMA: 0.2%, VL-BART: 0%)

---

### 2.3 모델 구조

DoRA의 구조적 특징을 정리하면 다음과 같습니다:

```
[사전학습 가중치 W₀]
        ↓ 분해 (Decompose)
┌─────────────────────────────────┐
│  magnitude m = ||W₀||_c  [훈련가능] │
│  direction V = W₀         [동결]   │
└─────────────────────────────────┘
        ↓ 방향 업데이트 (LoRA)
┌─────────────────────────────────┐
│  ΔV = BA  (B∈R^{d×r}, A∈R^{r×k}) [훈련가능] │
└─────────────────────────────────┘
        ↓ 병합 (Merge)
[추론 가중치] W' = m · (W₀ + BA) / ||W₀ + BA||_c
```

**핵심 특징:**
- **추론 시 오버헤드 없음**: 학습 후 $m$, $B$, $A$를 $W_0$에 병합 → 원본 모델과 동일한 구조
- **LoRA와 호환**: $\Delta V$ 부분을 VeRA 등 다른 LoRA 변형으로 대체 가능 (DVoRA)
- **QLoRA와 호환**: QDoRA로 확장 가능 (4-bit 양자화 기반)

---

### 2.4 성능 향상

#### 상식 추론 (Commonsense Reasoning) - LLaMA 계열

| 모델 | 방법 | 파라미터(%) | 평균 정확도 | LoRA 대비 향상 |
|------|------|------------|------------|---------------|
| LLaMA-7B | LoRA | 0.83 | 74.7 | - |
| LLaMA-7B | **DoRA** | 0.84 | **78.4** | **+3.7%** |
| LLaMA-7B | DoRA† (rank/2) | 0.43 | 77.5 | +2.8% |
| LLaMA-13B | LoRA | 0.67 | 80.5 | - |
| LLaMA-13B | **DoRA** | 0.68 | **81.5** | **+1.0%** |
| LLaMA2-7B | LoRA | 0.83 | 77.6 | - |
| LLaMA2-7B | **DoRA** | 0.84 | **79.7** (DoRA†: **80.5**) | **+2.9%** |
| LLaMA3-8B | LoRA | 0.70 | 80.8 | - |
| LLaMA3-8B | **DoRA** | 0.71 | **85.2** | **+4.4%** |

#### 이미지/비디오-텍스트 이해 (VL-BART)

| 태스크 | FT | LoRA | DoRA | DoRA vs LoRA |
|--------|-----|------|------|-------------|
| Image-Text Avg. | 77.3 | 76.5 | **77.4** | **+0.9%** |
| Video-Text Avg. | 87.5 | 83.5 | **85.4** | **+1.9%** |

#### 시각적 지시 튜닝 (LLaVA-1.5-7B)

| 방법 | 파라미터(%) | 평균 점수 |
|------|------------|----------|
| FT | 100 | 66.5 |
| LoRA | 4.61 | 66.9 |
| **DoRA** | 4.63 | **67.6** |

#### VeRA와의 호환성 (MT-Bench, LLaMA2-7B)

| 방법 | 파라미터(%) | MT-Bench 점수 |
|------|------------|--------------|
| VeRA | 0.02 | 5.5 |
| **DVoRA** | 0.04 | **6.0** |
| LoRA | 2.31 | 5.7 |
| **DoRA** | 2.33 | **6.0** |

---

### 2.5 한계

논문에서 명시적으로 또는 암묵적으로 확인되는 한계점들:

1. **분석 범위의 제한**: Weight Decomposition Analysis가 주로 self-attention의 query/value 행렬에 집중되어 있으며, MLP 레이어 등 다른 구성요소에 대한 분석이 부족

2. **음성(audio) 도메인 미검증**: 논문 결론부에서 직접 언급 — *"we wish to explore the generalizability of DoRA in domains beyond language and vision, particularly in the field of audio."*

3. **추가 하이퍼파라미터**: magnitude vector $m$을 위한 학습률 조정이 필요할 수 있으며, LoRA 대비 약간의 추가 튜닝이 필요

4. **메모리 오버헤드**: 수정 없이 사용 시 역전파 시 추가 메모리 필요 (수정 적용 시 해소되나 근사치 도입)

5. **FT가 이미 LoRA보다 열등한 경우**: LLaVA 실험에서 FT가 LoRA보다 낮은 점수를 보이는 상황에서 DoRA의 개선폭이 제한적 (과적합 억제 측면에서는 오히려 강점일 수 있으나 이론적 설명 보완 필요)

6. **높은 랭크에서의 성능 저하**: 표 15에서 DoRA(r=64)의 HellaSwag 정확도가 40.7%로 급락하는 현상이 관찰되어, 특정 조건에서의 안정성 문제 존재

---

## 3. 일반화 성능 향상 가능성 (중점 분석)

### 3.1 왜 DoRA가 더 나은 일반화를 달성하는가?

DoRA의 일반화 성능 향상은 다음 세 가지 메커니즘으로 설명됩니다:

#### (1) 사전학습 지식의 보존 (Pre-trained Knowledge Preservation)

DoRA로 파인튜닝된 가중치는 LoRA 대비 사전학습 가중치와의 편차가 **magnitude와 direction 모두에서** 훨씬 작습니다 (Figure 3, Figure 8 참조).

이는 다음 가설을 지지합니다:

> *"a robust foundation model does not require significant alterations for effective downstream adaptation"*

사전학습된 가중치가 이미 풍부한 일반 지식을 담고 있으므로, 소폭의 정밀한 조정만으로 충분하며, 이것이 오히려 더 나은 일반화로 이어집니다.

#### (2) 세밀한 크기/방향 독립 제어 (Decoupled Magnitude-Direction Control)

LoRA의 근본적 문제: magnitude와 direction 업데이트가 **결합(coupled)** 되어 있어 미세 조정이 어렵습니다.

DoRA의 핵심 이점: 두 성분을 **독립적으로** 최적화함으로써:

- **크게 방향만 바꾸고 크기는 유지** (예: 의미적 변화가 큰 태스크)
- **크기만 조정하고 방향은 유지** (예: 스케일 조정만 필요한 태스크)

이 유연성이 FT의 학습 패턴을 모방하며, 더 나은 태스크 적응과 일반화를 가능하게 합니다.

#### (3) 데이터 효율성 (Data Efficiency)

DoRA와 DVoRA는 **훈련 데이터 크기가 작을 때도** LoRA/VeRA보다 일관되게 우수한 성능을 보입니다 (Figure 4, 9):

| 훈련 샘플 수 | DoRA vs LoRA | DVoRA vs VeRA |
|------------|-------------|--------------|
| 1,000 | **+0.29** | **+0.22** |
| 4,000 | +0.27 | +0.28 |
| 7,000 | **+0.30** | **+0.33** |
| 10,000 | +0.30 | +0.50 |

이는 DoRA가 **제한된 데이터 환경에서도** 효과적으로 사전학습 지식을 활용한다는 것을 의미하며, 실제 산업 응용에서의 일반화 성능 우위를 뒷받침합니다.

#### (4) 랭크 견고성 (Rank Robustness)

낮은 랭크에서 DoRA의 일반화 성능 우위가 더욱 두드러집니다:

$$\text{DoRA}(r=8) = 77.9\% \quad \text{vs} \quad \text{LoRA}(r=8) = 40.7\%$$

$$\text{DoRA}(r=4) = 61.9\% \quad \text{vs} \quad \text{LoRA}(r=4) = 39.5\%$$

즉, DoRA는 **극히 적은 파라미터로도 의미 있는 일반화 성능**을 유지합니다. 이는 파라미터 효율성과 일반화 사이의 트레이드오프를 LoRA보다 훨씬 유리한 방향으로 이동시킵니다.

#### (5) 과적합 억제 (Overfitting Suppression)

LLaVA 실험에서 FT는 과적합으로 인해 LoRA보다 낮은 성능(66.5 vs 66.9)을 보이지만, DoRA는 67.6으로 두 방법 모두를 상회합니다. 이는 DoRA의 가중치 분해 구조가 불필요한 방향 변화를 억제하여 **정규화(regularization) 효과**를 갖는다는 것을 시사합니다.

#### (6) QDoRA: 메모리 제약 환경에서의 일반화

QDoRA(DoRA + QLoRA)는 LLaMA3-8B에서 Orca-Math 100k 샘플 파인튜닝 시:

$$\text{QDoRA} = 0.56 > \text{Full FT} = 0.51 > \text{QLoRA} = 0.32$$

4-bit 양자화 환경에서도 Full FT를 능가하는 일반화 성능을 보이며, 이는 DoRA의 가중치 분해 원리가 양자화된 모델에서도 일반화 성능 향상에 기여함을 의미합니다.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 PEFT 방법론 비교 표

| 방법 | 연도 | 핵심 아이디어 | 추론 오버헤드 | 파라미터 효율 | 일반화 성능 |
|------|------|-------------|-------------|-------------|------------|
| **Adapter** (Houlsby et al.) | 2019 | 레이어 간 추가 모듈 삽입 | **있음** | 중간 | 중간 |
| **Prefix-Tuning** (Li & Liang) | 2021 | 소프트 토큰 추가 | **있음** | 높음 | 초기화 민감 |
| **LoRA** (Hu et al.) | 2022 | 저랭크 행렬로 가중치 변화 근사 | **없음** | 높음 | 중간 |
| **AdaLoRA** (Zhang et al.) | 2023 | SVD 기반 동적 랭크 할당 | **없음** | 높음 | 중간-높음 |
| **VeRA** (Kopiczko et al.) | 2024 | 공유 랜덤 행렬 + 스케일링 벡터 | **없음** | **매우 높음** | 중간 |
| **DoRA** (Liu et al.) | 2024 | magnitude/direction 분해 + LoRA | **없음** | 높음 | **높음** |

### 4.2 DoRA vs LoRA: 이론적 차이

$$\text{LoRA: } W' = W_0 + BA$$

$$\text{DoRA: } W' = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}$$

LoRA는 가중치 변화를 **덧셈적(additive)** 으로 모델링하는 반면, DoRA는 **정규화된 방향 + 크기 스케일링**으로 모델링합니다. 이 구조적 차이가 Weight Normalization(Salimans & Kingma, 2016)의 최적화 이점을 파인튜닝에 이식합니다.

### 4.3 AdaLoRA와의 비교

AdaLoRA (Zhang et al., 2023)는 SVD를 통해 중요도에 따라 랭크를 동적으로 할당합니다:

$$W = P \Lambda Q^T$$

여기서 $\Lambda$의 작은 특이값은 제거하여 파라미터를 절약합니다. 반면 DoRA는 랭크를 고정하되 가중치 분해 방식을 바꿔 학습 패턴 자체를 개선하는 직교적 접근입니다. 두 방법은 상호 보완적으로 결합될 가능성이 있습니다.

### 4.4 VeRA와의 호환 (DVoRA)

$$\text{DVoRA: } W' = m \cdot \frac{W_0 + \Lambda_b B \Lambda_d A}{\|W_0 + \Lambda_b B \Lambda_d A\|_c}$$

VeRA의 공유 랜덤 행렬 $B$, $A$에 학습 가능한 스케일링 벡터 $\Lambda_b$, $\Lambda_d$를 적용한 것을 DoRA의 방향 업데이트로 사용합니다. 이로써 **0.04%의 파라미터만으로** LoRA(2.31%)와 동등한 성능 달성이 가능합니다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

#### (1) PEFT 패러다임의 재정립
DoRA는 "파라미터 수가 부족하기 때문에 LoRA가 FT보다 열등하다"는 기존 통념을 **학습 패턴의 질적 차이** 문제로 재프레임하였습니다. 이는 향후 PEFT 연구에서 단순히 훈련 가능 파라미터를 늘리는 방향이 아닌, **학습 역학(learning dynamics)** 자체를 개선하는 방향으로의 패러다임 전환을 촉진할 것입니다.

#### (2) 가중치 분해 분석 프레임워크의 확산
Weight Decomposition Analysis는 다른 PEFT 방법들(Prefix-Tuning, Adapter 등)의 학습 패턴을 분석하는 범용 도구로 활용될 수 있습니다. 이는 PEFT 방법의 이론적 이해를 심화하는 데 기여할 것입니다.

#### (3) QDoRA의 민주화 가능성
QDoRA는 소비자용 GPU에서 수십억 파라미터 모델을 파인튜닝할 수 있게 하며, 이는 오픈소스 커뮤니티와 학계의 연구 접근성을 크게 향상시킬 것입니다.

#### (4) 멀티모달 모델 파인튜닝의 표준화
LLaVA, VL-BART 등 다양한 멀티모달 모델에서의 성능 향상 입증은, DoRA가 멀티모달 파인튜닝의 **de facto 표준**으로 자리잡을 가능성을 시사합니다.

#### (5) 생성 모델로의 확장
SDXL DreamBooth 실험에서의 우수한 개인화 성능은, **텍스트-이미지 생성 모델** 파인튜닝 분야에서도 DoRA가 중요한 역할을 할 것임을 예고합니다.

---

### 5.2 앞으로 연구 시 고려할 점

#### (1) 음성(Audio) 도메인 확장 검증
논문 저자들이 명시한 미래 방향입니다. Whisper, EnCodec 등 음성 모델에서의 DoRA 적용 가능성과 magnitude/direction 분해 패턴이 텍스트/이미지와 유사한지 검증이 필요합니다.

#### (2) 높은 랭크에서의 불안정성 분석
$r=64$ 설정에서 DoRA의 HellaSwag 정확도가 40.7%로 급락하는 현상(표 15)은 해결되지 않은 문제입니다. 최적 랭크 선택 기준이나 적응적 랭크 조정(AdaLoRA와의 결합) 연구가 필요합니다.

$$\text{Consider: DoRA + AdaLoRA} \rightarrow \text{Adaptive rank DoRA}$$

#### (3) Magnitude 성분의 정규화 전략
현재 $m$은 단순히 훈련 가능한 벡터로 설정되어 있습니다. L1/L2 정규화나 스파스성(sparsity) 유도를 통해 더 효율적인 magnitude 학습이 가능한지 탐구할 수 있습니다.

#### (4) 레이어별 선택적 적용 (Tuning Granularity)
표 6에서 보듯, 모든 레이어에 동일하게 DoRA를 적용하는 것이 최적이 아닐 수 있습니다. 어떤 레이어에 magnitude만, 어떤 레이어에 direction도 업데이트할지를 자동으로 결정하는 **적응적 적용 방법** 개발이 중요합니다.

#### (5) 상관관계 이론의 정밀화
DoRA가 FT의 음의 $\Delta D$ - $\Delta M$ 상관관계를 완전히 재현하지는 못합니다 (-0.31 vs -0.62). 이 격차를 좁히기 위한 이론적 분석과 개선 방향 탐구가 필요합니다.

#### (6) 연속 학습(Continual Learning) 및 재난적 망각
DoRA의 가중치 분해 구조가 연속 학습 시나리오에서 재난적 망각(catastrophic forgetting)을 얼마나 억제하는지 체계적인 분석이 부족합니다. 이는 특히 다중 태스크 적응 연구에서 중요한 주제입니다.

#### (7) 이론적 수렴 보장 (Convergence Guarantee)
현재 DoRA의 이론적 분석은 주로 그래디언트 해석에 머물러 있습니다. 수렴 속도나 최적해 도달 가능성에 대한 엄밀한 이론적 보장을 제시하는 연구가 필요합니다.

#### (8) 다른 아키텍처로의 확장
현재 DoRA는 주로 Transformer 기반 모델에 적용됩니다. CNN, State Space Model (Mamba 등), GNN 등 다른 아키텍처에서의 적용 가능성과 효과 검증이 필요합니다.

---

## 요약 다이어그램

```
[FT 학습 패턴]          [LoRA 학습 패턴]       [DoRA 학습 패턴]
ΔM ↑ → ΔD ↓           ΔM ↑ → ΔD ↑           ΔM ↑ → ΔD ↓
(음의 상관, -0.62)      (양의 상관, +0.83)      (음의 상관, -0.31)
    ↓                      ↓                      ↓
세밀한 적응 가능         결합된 업데이트          FT에 근접한 패턴
우수한 일반화            제한적 학습능력          개선된 일반화
```

DoRA는 Weight Normalization의 최적화 이점을 파인튜닝에 이식함으로써, 추가 추론 비용 없이 LoRA의 학습 패턴을 FT에 근접하게 만드는 **원리적으로 타당하고 실용적으로 검증된** PEFT 방법입니다. 이 논문은 PEFT 분야에서 "왜 특정 방법이 더 잘 작동하는가"에 대한 깊은 통찰을 제공하며, 향후 PEFT 연구의 중요한 이론적·실용적 토대가 될 것입니다.
