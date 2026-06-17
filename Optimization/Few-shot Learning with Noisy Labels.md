# Few-shot Learning with Noisy Labels

> **참고 논문:** Kevin J Liang, Samrudhdhi B. Rangrej, Vladan Petrovic, Tal Hassner. "Few-shot Learning with Noisy Labels." arXiv:2204.05494v2 [cs.CV], 31 Jul 2022. (Facebook AI Research & McGill University)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 FSL(Few-Shot Learning) 방법들은 **support set의 레이블이 완벽하게 정확하다는 비현실적인 가정**에 의존한다. 현실에서는 소수의 샘플임에도 불구하고 오레이블(mislabeled sample)이 존재할 수 있으며, 이는 FSL 성능에 치명적인 영향을 미친다. 본 논문은 이 문제를 체계적으로 다룬 최초의 연구 중 하나이다.

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| ① 강건한 Prototype Aggregation | ProtoNet의 평균(mean) 대신 **중앙값(median)** 및 **유사도 가중(similarity weighting)** 방법 제안 |
| ② TraNFS 모델 | Transformer 기반 Noisy FSL 모델 설계 (attention으로 오레이블 샘플 억제) |
| ③ 포괄적 벤치마크 | MiniImageNet, TieredImageNet에서 3가지 노이즈 유형(symmetric, paired, outlier)으로 광범위한 실험 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**K-shot N-way FSL**에서 support set $S = \{x_1^{(1)}, x_2^{(1)}, ..., x_{K-1}^{(N)}, x_K^{(N)}\}$ 내에 오레이블 샘플이 포함될 때, 기존 방법의 성능이 급격히 저하된다.

- 10-shot, 5-way MiniImageNet 실험에서 ProtoNet은 오레이블 샘플 수가 증가함에 따라 정확도가 ~70%에서 ~20%대까지 추락
- 적은 샘플 수 특성상 **단 1개의 오레이블 샘플도 결정 경계(decision boundary)를 크게 왜곡**시킬 수 있음

**세 가지 노이즈 유형:**
1. **Symmetric label swap**: N-1개의 다른 클래스에서 균일 무작위로 오레이블 추출
2. **Paired label swap**: 각 클래스에 고정된 혼동 클래스를 배정하여 교란 (더 어려움)
3. **Outlier noise**: N-way episode 외부 클래스에서 샘플 추출

---

### 2.2 제안하는 방법 (수식 포함)

#### A. ProtoNet 기본 구조 (비교 기준)

ProtoNet의 클래스 프로토타입:

$$p^{(c)} = \frac{1}{K} \sum_i \mathcal{F}(x_i^{(c)}) \tag{1}$$

쿼리 분류:

$$y = \underset{c}{\arg\min} \; d(\mathcal{F}(x_q), p^{(c)}) \tag{2}$$

**문제:** mean 연산자는 outlier(오레이블 샘플)에 민감하여 prototype이 진짜 클래스 분포에서 멀어짐.

---

#### B. 정적 대안 방법들 (Static Alternatives)

**① 공간 중앙값 프로토타입 (Spatial Median Prototype)**

pseudo-Huber 손실 최소화로 중앙값 정의:

$$\mathcal{L}(p) = \sum_{i=1}^{K} \left( \sqrt{||p - h_i||_2^2 + \epsilon^2} - \epsilon \right) \tag{3}$$

Newton's method를 통한 반복 업데이트:

$$p(t+1) = p(t) - \mathcal{H}^{-1}(p(t)) \cdot \nabla\mathcal{L}(p(t)) \tag{4}$$

그래디언트:

$$\nabla\mathcal{L}(p) = \sum_{i=1}^{K} \frac{p - h_i}{\sqrt{||p - h_i||_2^2 + \epsilon^2}} \tag{5}$$

Hessian (대각 근사 적용 시):

$$p(t+1) = p(t) - \frac{\sum_{i=1}^{K} \frac{p(t) - h_i}{\sqrt{||p(t) - h_i||_2^2 + \epsilon^2}}}{\sum_{i=1}^{K} \frac{1}{\sqrt{||p(t) - h_i||_2^2 + \epsilon^2}}} \tag{7}$$

**② 유사도 가중 프로토타입 (Similarity Weighted Prototype)**

Squared Euclidean 유사도 점수:

$$a_i^{(c)} = -\frac{1}{K-1} \sum_{i \neq j} || h_i^{(c)} - h_j^{(c)} ||_2^2 \tag{8}$$

Absolute (L1) 유사도:

$$a_i^{(c)} = -\frac{1}{K-1} \sum_{i \neq j} | h_i^{(c)} - h_j^{(c)} | \tag{9}$$

Cosine 유사도:

$$a_i^{(c)} = \frac{1}{K-1} \sum_{i \neq j} \frac{h_i^{(c)} \cdot h_j^{(c)}}{||h_i^{(c)}|| \; ||h_j^{(c)}||} \tag{10}$$

Softmax 가중치 및 프로토타입 생성 (온도 파라미터 $T$):

$$w_i^{(c)} = \frac{\exp(a_i^{(c)}/T)}{\sum_j \exp(a_j^{(c)}/T)} \tag{11}$$

$$p^{(c)} = \sum_i w_i^{(c)} \mathcal{F}(x_i^{(c)}) \tag{12}$$

> $T \to 0$이면 최근접 샘플만 사용, $T \to \infty$이면 ProtoNet(mean)으로 수렴

---

#### C. TraNFS (Transformer for Noisy Few-Shot Learning)

**학습 기반 동적 집계 모델.** 입력 시퀀스:

$$\mathbf{h} = [h_1^{(1)}, h_2^{(1)}, ..., h_{K-1}^{(N)}, h_K^{(N)}]$$

**핵심 아이디어:**
- `CLS(c)` 토큰: 각 클래스의 프로토타입 출력 위치 (BERT에서 영감)
- `POS(c)` 토큰: 클래스 정체성 인코딩 (순서 불변성 유지)
- Self-attention으로 오레이블 샘플에 낮은 가중치 자동 학습

**손실 함수 (3가지 결합):**

Cross-entropy 손실:

$$\mathcal{L}_{\text{xent}} = -\sum_{c=1}^{N} y_q \cdot \log \left( \frac{\exp\left(-d\left(p^{(c)}, \mathcal{F}(x_q)\right)\right)}{\sum_{c'} \exp\left(-d\left(p^{(c')}, \mathcal{F}(x_q)\right)\right)} \right) \tag{13}$$

클린 프로토타입 손실 (올바르게 레이블된 샘플의 mean):

$$\hat{p}^{(c)} = \frac{1}{K - \sum_i o_i^{(c)}} \sum_{i} \mathbf{1}[o_i^{(c)}=0] \mathcal{F}(x_i^{(c)}) \tag{14}$$

$$\mathcal{L}_{\text{clean}} = \frac{1}{N} \sum_c ||p^{(c)} - \hat{p}^{(c)}||_2^2 \tag{15}$$

바이너리 outlier 분류 손실 (노이즈 여부 명시적 학습):

$$\mathcal{L}_{\text{bin}} = -\frac{1}{KN} \sum_{i,c} \left[ o_i^{(c)} \log \sigma(\mathcal{B}({h'}_i^{(c)})) + (1 - o_i^{(c)}) \log \left(1 - \sigma(\mathcal{B}({h'}_i^{(c)}))\right) \right] \tag{16}$$

최종 목적 함수:

$$\mathcal{L} = \mathcal{L}_{\text{xent}} + \lambda_c \mathcal{L}_{\text{clean}} + \lambda_b \mathcal{L}_{\text{bin}} \tag{17}$$

($\lambda_b = 0.5$, $\lambda_c = 5$ 사용)

---

### 2.3 모델 구조

```
[입력: Support Set 특징 시퀀스]
        ↓
[CNN Backbone (4-layer Conv, 고정)]
        ↓
[POS(c) 토큰 추가 → 클래스 정체성 인코딩]
[CLS(c) 토큰 연결 → 프로토타입 출력 위치]
        ↓
[Down-projection (직교 초기화, dim→128)]
        ↓
[Transformer Encoder (2~3층, 8-head self-attention)]
  - Layer 1: POS 인코딩으로 per-class 초점
  - Layer 2~3: 노이즈 샘플 억제 (attention ↓)
        ↓
[Up-projection]
        ↓
[CLS 위치 출력 → 클래스 프로토타입 p(c)]
[지원 샘플 위치 출력 → Binary Classifier B]
        ↓
[추론: ProtoNet 방식 (Transformer 미사용)]
```

---

### 2.4 성능 향상 및 한계

#### 성능 향상 (5-way 5-shot MiniImageNet 기준)

| 노이즈 유형 | 노이즈 비율 | ProtoNet | TraNFS-3 | 개선폭 |
|---|---|---|---|---|
| Symmetric swap | 40% | 51.41% | **56.65%** | +5.24%p |
| Symmetric swap | 60% | 38.33% | **42.60%** | +4.27%p |
| Paired swap | 40% | 47.77% | **53.96%** | +6.19%p |
| Outlier | 40% | 57.07% | **59.03%** | +1.96%p |
| 클린 (0%) | - | 68.27% | **68.53%** | ≈ 동등 |

- **Paired label swap 40%**에서 ProtoNet 대비 6.19%p 절대 향상, Oracle 대비 오차 41.7% 상대 감소
- 클린 데이터에서도 기존 방법과 동등한 성능 유지

#### 메타훈련 노이즈 비율 실험 (Table 4)

| 훈련 노이즈 | 0% | 20% | 40% | 60% |
|---|---|---|---|---|
| {0,20,40}% | **68.53** | **65.08** | **56.65** | **42.60** |
| 단일 0% | 69.10 | 63.56 | 52.85 | 39.19 |
| 단일 60% | 50.40 | 48.26 | 43.11 | 35.44 |

→ **다양한 노이즈 비율 혼합 훈련**이 가장 효과적

#### 한계점

1. **클린 메타훈련 가정**: 메타훈련 데이터셋 자체가 noisy한 경우는 다루지 않음
2. **클린 쿼리셋 가정**: 쿼리셋에 오레이블이 있으면 잘못된 그래디언트 발생 가능
3. **단순 CNN 백본**: 최신 고성능 백본(ViT 등) 미사용으로 절대 성능 상한 제한
4. **paired noise에 취약성**: 모든 오레이블이 동일 클래스에서 오는 경우 어려움
5. **고정 메타훈련 노이즈 분포**: 실제 테스트 노이즈 분포가 크게 다를 경우 일반화 불확실

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화에 기여하는 핵심 메커니즘

**① 다양한 노이즈 비율 메타훈련 (핵심)**

논문의 Table 4가 보여주듯, `{0%, 20%, 40%}` 노이즈를 혼합하여 메타훈련하면 단일 비율로 훈련한 모델보다 **모든 노이즈 수준에서 고르게 뛰어난 성능**을 보인다. 이는 실제 환경에서 노이즈 수준이 예측 불가능한 상황에 잘 대응할 수 있는 일반화를 의미한다.

**② Transformer Self-Attention의 동적 적응**

Figure 4의 시각화에서 확인되듯:
- Layer 1: POS 인코딩 기반으로 per-class 초점 형성
- Layer 2-3: 오레이블 샘플에 점진적으로 낮은 attention weight 부여

이 **데이터 적응형 가중 메커니즘**은 노이즈 패턴이 다른 다양한 태스크에도 자연스럽게 일반화된다.

**③ 순열 불변성 (Permutation Invariance)**

Transformer 구조는 support set 샘플의 순서에 무관하게 동작하며, 임의의 K-shot, N-way 설정을 처리할 수 있다. 이는 FSL의 에피소드별 무작위 구성에 자연스럽게 일반화된다.

**④ 보조 손실의 정규화 효과**

$\mathcal{L}\_{\text{clean}}$과 $\mathcal{L}_{\text{bin}}$은 단순히 분류 성능 향상뿐만 아니라 **노이즈 인식 표현 학습**을 강제하여, 모델이 새로운 클래스/도메인에서도 노이즈를 식별하는 능력을 일반화하도록 유도한다.

### 3.2 일반화 한계 및 잠재적 개선 방향

| 현재 한계 | 일반화 개선 가능성 |
|---|---|
| 단순 4-layer CNN | 사전훈련된 ViT/CLIP 백본 활용 시 표현력 대폭 향상 |
| 고정 노이즈 유형 가정 | 메타훈련 시 노이즈 유형 무작위 혼합으로 도메인 외 일반화 향상 가능 |
| 클린 쿼리 가정 | 쿼리 노이즈 처리 메커니즘 추가 시 더 넓은 적용 범위 |
| 이미지 도메인 한정 | 텍스트, 의료 데이터 등 다른 모달리티로의 확장 가능성 |

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

**① FSL 평가 프로토콜의 재정립**

이 논문은 FSL 연구에서 **"클린 support set" 가정을 제거한 현실적 벤치마크**의 필요성을 제기했다. 향후 FSL 연구에서 노이즈 조건 하의 성능 보고가 표준화될 가능성이 있다.

**② Transformer 기반 FSL의 확산**

TraNFS가 보여준 Transformer의 유효성은 이후 연구들이 FSL에서 attention 메커니즘을 노이즈 제거에 활용하는 방향을 촉진한다.

**③ 메타훈련 노이즈 증강 전략 정립**

다양한 비율의 노이즈를 메타훈련에 혼합하는 전략이 효과적임을 실증적으로 보였으며, 이는 **데이터 증강 관점에서의 노이즈 활용**이라는 새로운 연구 방향을 열었다.

**④ 실용적 FSL 시스템 설계에 기여**

자동화된 레이블링(웹 크롤링, 약지도 학습 등)을 사용하는 실제 시스템에서 FSL 적용 시 노이즈 강건성이 필수 요건임을 명확히 제시했다.

### 4.2 앞으로 연구 시 고려할 점

**① 노이즈 유형의 현실적 다양성 확장**
- 현재 연구는 symmetric, paired, outlier 3가지만 다루지만, 실제로는 annotation 오류, 레이블 모호성, 분류 체계 불일치 등 더 복잡한 노이즈가 존재
- 실제 noisy 데이터셋 (예: WebVision, Clothing1M)에서의 Few-shot 실험 필요

**② 메타훈련 데이터셋 노이즈 처리**
- 현재 논문은 메타훈련은 클린하다고 가정하지만, 대규모 웹 데이터 기반 메타훈련 시 이 가정이 깨짐
- CoteachingFSL, MentorNet과 같은 curriculum learning 아이디어와의 결합 연구 필요

**③ 쿼리셋 노이즈 강건성**
- 논문이 직접 지적한 한계로, 쿼리셋에 오레이블이 있으면 $\mathcal{L}_{\text{xent}}$ 자체가 오염됨
- Semi-supervised 노이즈 레이블 학습 기법과의 통합 필요

**④ 더 강력한 백본과의 결합**
- 논문은 4-layer Conv를 의도적으로 사용했지만, CLIP, DINOv2 등의 대형 모델과 결합 시 성능 상한을 높이는 연구 필요

**⑤ 계산 효율성**
- TraNFS의 메타훈련은 추가 계산 비용을 요구하며, 실시간 시스템 적용을 위한 경량화 연구 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의:** 아래 비교는 본 논문(2022)과 관련된 연구 동향을 바탕으로 하되, 논문 본문 내 인용된 문헌 및 공개된 연구 흐름을 기반으로 작성했습니다. 2022년 이후 발표된 특정 논문의 정확한 수치는 직접 검색을 권장합니다.

### 5.1 관련 연구 계보

| 논문 | 연도 | 접근법 | 본 논문과의 관계 |
|------|------|--------|----------------|
| RNNP (Mazumder et al.) | 2021 | k-means 기반 프로토타입 정제 | 비교 베이스라인으로 사용; TraNFS에 열세 |
| RapNets (Lu et al.) | 2020 | BiLSTM 기반 attention | 비교 대상; 단, 논문 내 실험에 미포함 |
| RW-MAML (Killamsetty et al.) | 2020 | bi-level 최적화로 샘플 가중 | OOD 태스크 혼합 가정으로 비현실적 설정 |
| FEAT (Ye et al., CVPR 2020) | 2020 | Set-to-set 함수로 임베딩 적응 | 노이즈 미고려, 클린 설정에서만 비교 |
| CrossTransformers (Doersch et al., NeurIPS 2020) | 2020 | 공간 인식 attention | 지역 특징 유사도, 노이즈 강건성 미설계 |

### 5.2 연구 흐름 분석

```
[노이즈 레이블 학습 (많은 데이터)]        [Few-Shot Learning]
    Co-teaching (NeurIPS 2018)                ProtoNet (NeurIPS 2017)
    MentorNet (ICML 2018)           +         MAML (ICML 2017)
    DivideMix (ICLR 2020)                     FEAT (CVPR 2020)
              ↓                                      ↓
         교차 영역 융합 시도 (미개척)
              ↓
    [본 논문: Few-shot + Noisy Labels, 2022]
    - 문제 정의 및 벤치마크 확립
    - TraNFS 제안
              ↓
    [향후 연구 방향]
    - Diffusion/생성 모델 활용 노이즈 정제
    - LLM 기반 레이블 신뢰도 추정
    - Cross-modal FSL with noisy labels
```

### 5.3 방법론적 비교

| 특성 | 본 논문 (TraNFS) | 일반 FSL 연구 추세 | 노이즈 레이블 연구 추세 |
|------|-----------------|-------------------|----------------------|
| 데이터 규모 | Few-shot (K≤10) | Few-shot | Many-shot |
| 노이즈 가정 | Support set noisy | 노이즈 없음 가정 | 전체 훈련셋 noisy |
| 핵심 메커니즘 | Transformer attention | 다양한 metric learning | 노이즈 전이 행렬, curriculum |
| 일반화 전략 | 다양한 노이즈 비율 메타훈련 | 에피소드 기반 메타훈련 | 클린 샘플 선택 |
| 백본 | 4-layer Conv (고정) | ResNet12, ViT 등 | 다양한 대형 모델 |

---

## 참고 자료

1. **주 논문:** Liang, K.J., Rangrej, S.B., Petrovic, V., & Hassner, T. "Few-shot Learning with Noisy Labels." arXiv:2204.05494v2, 2022.
2. **코드 저장소:** https://github.com/facebookresearch/noisy_few_shot
3. **ProtoNet:** Snell, J., Swersky, K., & Zemel, R. "Prototypical Networks for Few-shot Learning." NeurIPS, 2017.
4. **MAML:** Finn, C., Abbeel, P., & Levine, S. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML, 2017.
5. **RNNP:** Mazumder, P., Singh, P., & Namboodiri, V.P. "RNNP: A Robust Few-Shot Learning Approach." WACV, 2021.
6. **Co-teaching:** Han, B., et al. "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels." NeurIPS, 2018.
7. **Attention is All You Need:** Vaswani, A., et al. NeurIPS, 2017.
8. **MiniImageNet / Matching Networks:** Vinyals, O., et al. NeurIPS, 2016.
9. **TieredImageNet:** Ren, M., et al. ICLR, 2018.
10. **FEAT:** Ye, H.J., et al. "Few-shot Learning via Embedding Adaptation with Set-to-Set Functions." CVPR, 2020.
