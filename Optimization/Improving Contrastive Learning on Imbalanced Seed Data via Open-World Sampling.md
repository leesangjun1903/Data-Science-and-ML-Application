# Improving Contrastive Learning on Imbalanced Seed Data via Open-World Sampling

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음과 같습니다: **불균형한 시드(seed) 데이터로 대조 학습(Contrastive Learning)을 수행할 때, 외부 소스에서 데이터를 무작위로 추가하는 것은 전체 정확도는 향상시키지만 클래스 균형성(balancedness) 개선에는 제한적이며 불안정하다.** 따라서, 원칙에 기반한 전략적 데이터 샘플링 프레임워크인 **Model-Aware K-center (MAK)** 를 제안하여 세 가지 원칙(Tailness, Proximity, Diversity)을 동시에 만족하는 외부 데이터를 선별함으로써, 더 균형 잡히고 일반화된 표현(representation)을 학습할 수 있다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **새로운 문제 정의** | 레이블 없이 오픈 월드 외부 데이터를 전략적으로 샘플링하여 불균형 대조 학습 개선 |
| **ECLE 제안** | 랜덤 증강의 노이즈를 제거한 새로운 테일 클래스 탐지 프록시 |
| **MAK 프레임워크** | Tailness + Proximity + Diversity를 통합한 통일된 최적화 프레임워크 |
| **실증적 성과** | ImageNet-100-LT에서 균형성(Std) 및 전체 정확도 모두 일관된 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 설정:**
- 레이블이 없는 소규모 시드 데이터셋에서 출발하되, 해당 데이터의 클래스 분포는 **장기 꼬리 분포(long-tail distribution)** 를 따름
- 인터넷 등 외부 소스에서 추가 비레이블 데이터를 수집하여 대조 학습을 강화하고자 함
- **핵심 도전과제:**
  1. 레이블 부재로 클래스 불균형 정도를 알 수 없음 → 기존 재샘플링/손실 재가중치 방법 적용 불가
  2. 불균형 시드 데이터로 학습된 백본은 테일 클래스를 제대로 학습하지 못해 편향 심화
  3. 오픈 월드 데이터의 OoD(Out-of-Distribution) 샘플을 레이블 없이 탐지하기 어려움

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 기본 대조 학습 손실 (SimCLR)

SimCLR의 $i$번째 샘플에 대한 손실:

$$\mathcal{L}_{\text{CL},i} = -\log \frac{s^{\tau}(A(v_i, \theta_{i,1}), A(v_i, \theta_{i,2}))} {s^{\tau}(A(v_i, \theta_{i,1}), A(v_i, \theta_{i,2})) + \sum_{v_i^- \in V^-} s^{\tau}(A(v_i, \theta_{i,1}), v_i^-)} $$

여기서:
- $A(\cdot, \theta)$: 랜덤 데이터 증강 함수, $\theta \sim \Theta$
- $s^{\tau}(a,b) = \exp(a \cdot b / \tau)$: 온도 $\tau$를 갖는 유사도 함수
- $v_i^-$: 음성 샘플(negative sample)

#### Step 2: Tailness — 경험적 대조 손실 기댓값 (ECLE)

단일 증강으로 계산한 대조 손실은 랜덤 증강에 매우 민감하므로, 이를 평활화(smoothing)하기 위해 다음 ECLE를 정의:

$$\mathcal{L}^{\mathcal{E}}_{\text{CL},i} = \mathbb{E}_{\theta_{i,1},\theta_{i,2} \sim \Theta} \left[\mathcal{L}_{\text{CL},i}(\theta_{i,1}, \theta_{i,2}; \tau, v_i, V^-)\right] $$

실제로는 $M$회 샘플링의 표본 평균으로 근사:

$$\mathcal{L}^{\mathcal{E}}_{\text{CL},i} \approx \frac{1}{M} \sum_{m=1}^{M} \mathcal{L}_{\text{CL},i}(\theta_{i,1}^{(m)}, \theta_{i,2}^{(m)})$$

- ECLE 값이 클수록 해당 샘플은 테일 클래스일 가능성이 높음
- 실험적으로 $M=10$ (SimCLR 기준 5번의 순전파)이 충분함을 검증

#### Step 3: Proximity — OoD 거부

시드 셋 $s^0$과 새 샘플 집합 $s^1$ 간의 평균 피처 거리:

$$D(s^0, s^1) = \frac{1}{|s^1|} \sum_{j \in s^1} \min_{i \in s^0} \Delta(x_i, x_j) $$

- $\Delta(x_i, x_j)$: 정규화된 코사인 거리(cosine distance)
- 효율성을 위해 $s^0$에서 K-means 클러스터링으로 프로토타입 집합 $s^0_p$를 생성, $D(s^0_p, s^1)$ 계산

#### Step 4: Diversity — K-center 다양성 촉진

샘플 집합의 다양성을 보장하는 minimax 목적함수:

$$H(s^1 \cup s^0, S_{\text{all}}) = \max_{i \in S_{\text{all}}} \min_{j \in s^1 \cup s^0} \Delta(x_i, x_j) $$

- $S_{\text{all}}$: 시드 + 외부 데이터 전체
- 이를 최소화 = K-center 그리디 알고리즘으로 다양한 커버 포인트 선택

#### Step 5: MAK 통합 최적화

세 원칙을 통합한 제약 최적화:

```math
\max_{s^1 : |s^1| \leq K} \left\{ \sum_{i \in s^1} \mathcal{L}^{\mathcal{E}}_{\text{CL},i} - D(s^0, s^1) - H(s^1 \cup s^0, S_{\text{all}}) \right\}
```

**그리디 알고리즘에서 사용하는 결합 스코어:**

$$q = \alpha N(\mathcal{L}^{\mathcal{E}}_{\text{CL},i}) - (1-\alpha) N(D(s^0, s^1)) $$

여기서 $N(v) = \dfrac{v - \text{mean}(v)}{\text{std}(v)}$는 정규화 함수, $\alpha \in (0,1)$은 가중치 계수.

### 2.3 모델 구조

```
[외부 데이터 풀 (S_all)]
       ↓
[1단계] 시드 데이터(s⁰)로 SimCLR + ResNet-50 사전학습
       ↓
[2단계] ECLE 계산 (M=10회 증강 반복)
[3단계] 프로토타입 기반 Proximity 거리 계산
       ↓
[4단계] 결합 스코어 q로 후보 풀 S' (크기 C=1.5K) 구성
       ↓
[5단계] K-center 그리디로 K개의 다양한 샘플 s¹ 선택
       ↓
[6단계] s⁰ + s¹ 합쳐서 SimCLR 재학습
       ↓
[평가] Linear Separability / Few-shot 성능 측정
```

**주요 하이퍼파라미터:**

| 파라미터 | 값 |
|----------|-----|
| 증강 반복 수 $M$ | 10 |
| K-means 클러스터 수 | 10 |
| 가중치 $\alpha$ | 0.3 |
| 후보셋 크기 $C$ | $1.5 \times K$ |
| 백본 | ResNet-50 |
| 사전학습 에폭 | 1000 |

### 2.4 성능 향상

#### 메인 결과 (ImageNet-100-LT, 예산 10K)

| 방법 | 프로토콜 | Many↑ | Medium↑ | Few↑ | Std↓ | All↑ |
|------|----------|-------|---------|------|------|------|
| 없음 | Linear | 71.2 | 65.3 | 62.7 | 3.6 | 67.3 |
| Random (IN900) | Linear | 74.6 | 69.7 | 66.1 | 3.5 | 71.2 |
| K-center (IN900) | Linear | 73.6 | 68.6 | 64.5 | 3.8 | 70.0 |
| **MAK (IN900)** | **Linear** | **76.1** | **70.8** | **69.3** | **3.0** | **72.7** |
| Random (IN900) | Few-shot | 56.6 | 48.6 | 43.7 | 5.3 | 51.1 |
| **MAK (IN900)** | **Few-shot** | **57.4** | **48.9** | **46.3** | **4.8** | **51.9** |

**핵심 성과:**
- Few 그룹(테일 클래스) 정확도: 랜덤 대비 **+3.2%p** (66.1 → 69.3)
- 불균형 지표 Std: **−0.5%p** 감소 (균형성 향상)
- OoD가 많은 IPM 데이터셋에서도 일관된 개선

#### 에블레이션 결과

각 컴포넌트를 단독 사용 시 오히려 랜덤보다 성능 저하:
- Tailness만: OoD 아웃라이어 다수 포함
- Proximity만: 중복 샘플 과잉 (다양성 없음)
- Diversity만: 음성 샘플 약화 (K-means와 동일한 문제)
- **Tailness + Proximity 조합 시 첫 개선 발생, Diversity 추가로 정확도 추가 향상**

### 2.5 한계점

1. **학술 데이터셋에만 검증**: 자율주행, 의료 등 실제 응용에서의 공정성·프라이버시 문제 미검토
2. **계산 비용**: ECLE 계산을 포함한 전체 오버헤드가 약 700 에폭 추가 학습에 상당
3. **단일 백본 프레임워크**: SimCLR 기반으로만 검증, BYOL·MoCo 등 다른 프레임워크 실험은 미래 작업으로 남김
4. **반복적 샘플링 미지원**: MAK가 단일 라운드 샘플링만 수행하며, 반복적 능동 학습 루프는 다루지 않음
5. **외부 데이터 접근성 가정**: 인터넷 규모의 외부 데이터가 항상 접근 가능하다고 가정

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상 메커니즘

MAK가 일반화 성능을 향상시키는 원리는 다음 세 축으로 분석됩니다:

#### (a) 균형 잡힌 표현 공간 형성

대조 학습의 일반화 능력은 특징 공간(feature space)이 클래스 전반에 걸쳐 균등하게 분포할수록 향상됩니다. MAK는 테일 클래스에 편향된 샘플링을 통해:

$$\phi = \frac{\text{target group's percentage in samples with 10\% highest loss}}{\text{data percentage of the target group}}$$

위 지표 $\phi$를 분석한 결과, $M=10$일 때 소수 클래스의 $\phi$가 다수 클래스보다 **약 2배** 높게 나타남 → ECLE가 테일 클래스를 신뢰성 있게 탐지함을 실증

#### (b) Few-shot 성능 향상

Few-shot 설정(전체 데이터의 1% 레이블만 사용한 파인튜닝)은 실제 다운스트림 태스크의 일반화 능력을 반영합니다:

- MAK (IN900, 10K): Few 그룹 few-shot 정확도 **46.3%** (랜덤 43.7% 대비 +2.6%p)
- 이는 특히 레이블이 희소한 실제 환경에서 MAK의 일반화 이점이 더욱 두드러짐을 의미

#### (c) OoD 거부를 통한 분포 내 일반화 유지

Proximity 컴포넌트는 외부 데이터 중 시드 분포에서 벗어난 샘플을 제거하여, 모델이 **목표 분포(target distribution)** 에 적합한 표현을 학습하도록 유도합니다. 이는 도메인 외 데이터로 인한 부정적 전이(negative transfer)를 방지합니다.

#### (d) 다운스트림 불균형 태스크에서의 일반화

ImageNet-100-LT를 직접 불균형 다운스트림 태스크로 사용한 평가에서:
- MAK: [accuracy=52.9, std=24.8]
- Random: [accuracy=52.1, std=25.4]
- 정확도 +0.8, 불균형도 −0.6 개선 → **불균형한 실제 환경에서도 일반화 능력 향상**

### 3.2 일반화 한계와 잠재적 위험

- 시드 데이터의 특징 추출기가 편향되면, ECLE와 Proximity 계산 모두 편향될 수 있음 (편향의 자기강화 문제)
- 외부 데이터와 시드 데이터의 도메인 격차가 클 경우 Proximity 기준이 왜곡될 수 있음

---

## 4. 2020년 이후 최신 연구 비교 분석

### 4.1 관련 연구 계보

```
SimCLR (Chen et al., 2020) ─────────────────────────────────┐
MoCo v2 (Chen et al., 2020) ────────────────────────────────┤
BYOL (Grill et al., 2020) ──────────────────────────────────┤
                                                              ↓
Yang & Xu (2020) - 불균형 학습에서 레이블 가치 재고 ──────────┤
Kang et al. (2021) - 균형 피처 공간 탐색 ───────────────────┤
Jiang et al. (ICML 2021) - Self-Damaging CL ────────────────┤
                                                              ↓
                          [본 논문: MAK, NeurIPS 2021]
```

### 4.2 주요 관련 연구와 비교

| 논문 | 핵심 아이디어 | 레이블 필요 | 외부 데이터 활용 | 불균형 처리 | 본 논문과 차이 |
|------|-------------|-----------|----------------|-----------|--------------|
| **SimCLR** (Chen et al., ICML 2020) | 기본 대조 학습 | ✗ | ✗ | ✗ | MAK의 백본 프레임워크 |
| **MoCo v2** (Chen et al., 2020) | 모멘텀 대조 학습 | ✗ | ✗ | ✗ | MAK 적용 가능한 프레임워크 |
| **BYOL** (Grill et al., 2020) | 음성 샘플 없는 자기지도 학습 | ✗ | ✗ | ✗ | MAK 확장 가능 대상 |
| **Yang & Xu** (NeurIPS 2020) | 불균형 학습에서 레이블 효과 분석 | ✓ (일부) | ✗ | ✓ | 레이블 사용, 외부 데이터 없음 |
| **Kang et al.** (ICLR 2021) | 균형 피처 공간 탐색 | ✓ | ✗ | ✓ | 레이블 사용 |
| **Self-Damaging CL** (Jiang et al., ICML 2021) | 점진적 손상으로 테일 클래스 강조 | ✗ | ✗ | ✓ | 외부 데이터 샘플링 없음 |
| **SSCL (Goyal et al., 2021)** | 야생 데이터로 자기지도 사전학습 | ✗ | ✓ | ✗ | 불균형 문제 미해결 |
| **본 논문 (MAK)** | ECLE + OoD거부 + 다양성 통합 | ✗ | ✓ | ✓ | **레이블 없이 외부 데이터 전략 샘플링** |

### 4.3 방법론적 차별성

**MAK vs. 기존 능동학습 (Core-set, Sener & Savarese, 2018):**
- Core-set은 지도학습용으로, 모델 손실 기반 정보만 활용
- MAK는 비지도 환경에서 **ECLE(모델 인식 테일 탐지) + K-center(다양성)** 를 통합

$$\underbrace{\text{Core-set}}_{\text{다양성만}} \xrightarrow{\text{확장}} \underbrace{\text{MAK}}_{\text{다양성 + 테일성 + 근접성}}$$

---

## 5. 향후 연구에 미치는 영향 및 고려사항

### 5.1 향후 연구에 미치는 영향

#### (a) 오픈 월드 자기지도 학습 패러다임 정립

본 논문은 **"레이블 없이 오픈 월드 데이터를 전략적으로 활용"** 하는 새로운 연구 방향을 개척했습니다. 인터넷 규모의 데이터를 무분별하게 수집하는 대신, 원칙 기반 선별이 필요하다는 인식을 확산시킵니다.

#### (b) 능동 학습 + 자기지도 학습의 교차점 확장

Core-set 능동학습을 비지도 환경으로 확장한 MAK의 접근법은, 자기지도 학습에서의 **데이터 효율적 학습(data-efficient learning)** 연구를 자극할 것입니다.

#### (c) 공정성 AI 연구와의 연결

테일 클래스 강화를 통해 소수 클래스 성능을 개선하는 MAK의 접근법은, **알고리즘 공정성(algorithmic fairness)** 연구와 자연스럽게 연결됩니다.

#### (d) 데이터 선택 및 데이터 중심 AI (Data-centric AI)

**데이터 중심 AI** 트렌드(모델보다 데이터 품질에 집중)에서 MAK의 원칙 기반 샘플링 철학은 중요한 참조점이 됩니다.

### 5.2 향후 연구 시 고려해야 할 점

#### (a) 다양한 대조 학습 프레임워크 확장
- 본 논문은 SimCLR에만 검증되었으므로, **MoCo v2, BYOL, SwAV, DINO** 등에 MAK 적용 가능성 탐구 필요
- 특히 음성 샘플 없이 학습하는 BYOL/SimSiam에서 ECLE의 해석이 달라질 수 있음

#### (b) 반복적(iterative) 샘플링 루프 설계
현재 MAK는 단일 라운드 샘플링이므로:

$$\text{Seed} \xrightarrow{\text{1회}} \text{MAK} \xrightarrow{} \text{추가 학습}$$

다음과 같은 반복 구조로 개선 가능:

$$\text{Seed} \xrightarrow{\text{MAK Round 1}} \text{업데이트된 모델} \xrightarrow{\text{MAK Round 2}} \cdots$$

#### (c) 더 강력한 OoD 탐지 메커니즘
- Proximity 기반 OoD 거부는 단순 거리 기준이므로, **에너지 기반 OoD 탐지(Energy-based OOD)** 나 **Mahalanobis Distance** 등 고도화된 방법과의 결합 고려

#### (d) 계산 효율성 개선
- ECLE 계산이 $M$회 증강 반복을 요구하므로, **메타러닝** 이나 **그래디언트 기반 추정** 으로 ECLE를 근사하는 경량화 연구 필요

#### (e) 공정성·편향 분석
- 논문이 직접 언급했듯, 자율주행·의료 진단 등 고위험 영역 적용 시 알고리즘이 특정 속성(인종, 성별 등)에 대한 편향을 강화하지 않는지 철저한 공정성 감사 필요

#### (f) 이론적 보장 강화
- MAK의 최적화 문제(식 5)는 NP-hard이며 그리디 근사를 사용하므로, 근사 비율(approximation ratio)에 대한 이론적 보장 연구가 필요

#### (g) 대규모 언어모델(LLM) 시대에의 적용
- **CLIP, ALIGN** 등 비전-언어 모델의 사전학습 데이터 선별에 MAK 원칙 적용 가능성 탐구
- 텍스트-이미지 쌍 데이터의 불균형 문제에 ECLE 개념의 확장 가능성

---

## 참고자료 (출처)

본 답변은 제공된 논문 PDF를 기반으로 작성되었으며, 비교 분석에서 언급된 관련 연구는 해당 논문의 참고문헌 목록에서 인용하였습니다.

**주 논문:**
- Jiang, Z., Chen, T., Chen, T., & Wang, Z. (2021). **Improving Contrastive Learning on Imbalanced Seed Data via Open-World Sampling**. *NeurIPS 2021*. [코드: https://github.com/VITA-Group/MAK]

**논문 내 인용 주요 참고문헌:**
- [1] Chen et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML 2020*.
- [3] He et al. (2020). Momentum Contrast for Unsupervised Visual Representation Learning. *CVPR 2020*.
- [4] Grill et al. (2020). Bootstrap Your Own Latent. *arXiv:2006.07733*.
- [6] Goyal et al. (2021). Self-supervised Pretraining of Visual Features in the Wild. *arXiv:2103.01988*.
- [9] Yang & Xu (2020). Rethinking the Value of Labels for Improving Class-Imbalanced Learning. *NeurIPS 2020*.
- [10] Kang et al. (2021). Exploring Balanced Feature Spaces for Representation Learning. *ICLR 2021*.
- [11] Jiang et al. (2021). Self-Damaging Contrastive Learning. *ICML 2021*.
- [20] Sener & Savarese (2018). Active Learning for Convolutional Neural Networks: A Core-Set Approach. *ICLR 2018*.
- [27] Liu et al. (2019). Large-Scale Long-Tailed Recognition in an Open World. *CVPR 2019*.
