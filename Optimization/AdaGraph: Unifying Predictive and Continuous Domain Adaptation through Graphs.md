# AdaGraph: Unifying Predictive and Continuous Domain Adaptation through Graphs

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

AdaGraph는 **Predictive Domain Adaptation(PDA)** 문제를 최초로 딥러닝 아키텍처로 해결한 논문입니다. 핵심 주장은 다음과 같습니다:

> *"타겟 도메인의 데이터가 전혀 없는 상황에서도, 메타데이터(metadata)와 보조 도메인(auxiliary domains) 간의 그래프 관계를 활용하면 타겟 도메인에 특화된 모델 파라미터를 예측할 수 있으며, 테스트 시점에 점진적으로 들어오는 데이터로 이를 지속적으로 개선할 수 있다."*

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| ① 최초의 PDA 딥 아키텍처 | PDA 문제를 위한 첫 번째 딥러닝 기반 프레임워크 제안 |
| ② 그래프 기반 메타데이터 인코딩 | 도메인 간 관계를 그래프로 모델링하여 도메인 파라미터 예측 |
| ③ 연속 도메인 적응(Continuous DA) 전략 | 테스트 시점에 스트리밍 데이터를 활용한 실시간 모델 정제 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**도메인 시프트(Domain Shift)** 문제는 훈련 도메인과 테스트 도메인의 데이터 분포가 달라 성능이 저하되는 현상입니다. 기존 DA 방법들은 훈련 시 타겟 데이터를 필요로 하지만, 현실에서는 모든 가능한 타겟 도메인의 데이터를 사전 수집하는 것이 불가능합니다.

**PDA 시나리오의 정의:**
- 훈련 시: 레이블된 소스 도메인 $\mathcal{S}$ + 레이블 없는 보조 도메인 $\mathcal{A} = \{A_1, \cdots, A_N\}$ + 각 도메인의 메타데이터
- 테스트 시: 타겟 도메인 $\mathcal{T}$의 메타데이터 $m_\mathcal{T}$만 주어짐 (데이터 없음)
- 목표: $m_\mathcal{T}$를 이용해 타겟 도메인 분류 모델의 파라미터를 예측

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 그래프 구성 및 엣지 가중치 정의

도메인 집합 $\mathcal{K} = \{k_1, \cdots, k_n\}$을 그래프 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$로 표현합니다. 두 도메인 $(v_1, v_2)$ 사이의 엣지 가중치는 메타데이터 거리로 정의됩니다:

$$\omega(v_1, v_2) = e^{-d(\phi(v_1), \phi(v_2))} \tag{1}$$

여기서 $d : \mathcal{M}^2 \rightarrow \mathbb{R}$은 메타데이터 공간 $\mathcal{M}$ 위의 거리 함수이며, 실험에서는 $d(x,y) = \frac{1}{2\sigma}\|x - y\|_2^2$, $\sigma = 0.1$ 사용.

#### Step 2: 타겟 도메인 파라미터 예측 (Parameter Prediction)

타겟 도메인 $\mathcal{T}$에 대한 파라미터 $\hat{\theta}_\mathcal{T}$를 근방 노드들의 가중 평균으로 추정합니다:

$$\hat{\theta}_\mathcal{T} = \psi(\mathcal{T}) = \frac{\sum_{(\mathcal{T}, v) \in \mathcal{E}'} \omega(\mathcal{T}, v) \psi(v)}{\sum_{(\mathcal{T}, v) \in \mathcal{E}'} \omega(\mathcal{T}, v)} \tag{2}$$

메타데이터 없이 이미지 $x$만 있을 경우, 이미지로부터 도메인 확률 분포 $p(v|x)$를 추정하여:

$$\hat{\theta}_x = \sum_{v \in \mathcal{V}} p(v|x) \cdot \psi(v) \tag{3}$$

#### Step 3: GraphBN (GBN) 레이어 — 도메인 특화 배치 정규화

기존 Domain-Adaptive BN (DABN)을 참조하되, 도메인 $k$에 특화된 통계량을 적용합니다:

$$\text{DA}_\text{BN}(x, k) = \gamma \cdot \frac{x - \mu_k}{\sqrt{\sigma_k^2 + \epsilon}} + \beta \tag{4}$$

AdaGraph에서는 이를 GBN으로 확장하여 그래프 구조를 반영합니다:

$$\text{GBN}(x, v) = \gamma_v \cdot \frac{x - \mu_v}{\sqrt{\sigma_v^2 + \epsilon}} + \beta_v \tag{6}$$

훈련 중 도메인 $v$에 속하는 배치 $\mathcal{B}_v$로 통계량을 업데이트합니다:

$$\hat{\mu}_v = \frac{1}{|\mathcal{B}_v|} \sum_{x \in \mathcal{B}_v} x, \quad \hat{\sigma}_v^2 = \frac{1}{|\mathcal{B}_v|} \sum_{x \in \mathcal{B}_v} (x - \mu_v)^2 \tag{7}$$

그래프 관계를 반영한 GBN forward pass:

$$\text{GBN}(x, v, \mathcal{G}) = \gamma_v^\mathcal{G} \cdot \frac{x - \mu_v}{\sqrt{\sigma_v^2 + \epsilon}} + \beta_v^\mathcal{G} \tag{9}$$

$$\nu_v^\mathcal{G} = \frac{\sum_{k \in \mathcal{K}} \omega(v, k) \cdot \nu_k}{\sum_{k \in \mathcal{K}} \omega(v, k)}, \quad \nu \in \{\beta, \gamma\} \tag{10}$$

#### Step 4: 손실 함수 — 소스 + 보조 도메인 공동 학습

소스 도메인: cross-entropy loss, 보조 도메인(비레이블): entropy minimization loss:

$$\mathcal{L}(\Theta^s) = -\frac{1}{|\mathcal{S}|} \sum_{(x,y) \in \mathcal{S}} \log(f_{\theta_\mathcal{S}}(y; x)) - \lambda \cdot \sum_{A_i \in \mathcal{A}} \frac{1}{|A_i|} \sum_{x \in A_i} \sum_{y \in \mathcal{Y}} f_{\theta_{A_i}}(y;x) \log f_{\theta_{A_i}}(y;x) \tag{8}$$

#### Step 5: 연속 적응 (Continuous Adaptation) — 통계량 업데이트

버퍼 $M$에 타겟 이미지를 저장하고 지수 이동 평균으로 통계량 갱신:

$$\mu_\mathcal{T} \leftarrow (1 - \alpha) \cdot \mu_\mathcal{T} + \alpha \cdot \mu_M$$
$$\sigma_\mathcal{T}^2 \leftarrow (1 - \alpha) \cdot \sigma_\mathcal{T}^2 + \alpha \cdot \frac{|M|}{|M|-1} \cdot \sigma_M^2 \tag{11}$$

$\gamma_\mathcal{T}, \beta_\mathcal{T}$ 파라미터는 엔트로피 손실로 추가 정제:

$$\mathcal{L}(\theta_\mathcal{T}) = -\frac{1}{|M|} \sum_{x \in M} \sum_{y \in \mathcal{Y}} f_{\theta_\mathcal{T}}(y; x) \log f_{\theta_\mathcal{T}}(y; x) \tag{13}$$

---

### 2.3 모델 구조

```
입력 이미지 x
    │
    ▼
[Conv] → [GBN] → [Conv] → [GBN] → ... → [FC]
           │                │
           └────────────────┘
               도메인 그래프 G
          (노드: 도메인, 엣지: 메타데이터 유사도)
               │
    ┌──────────┴──────────┐
    │  훈련 시 (파란 경로) │  테스트 시 (빨간 경로)
    │  알려진 도메인 z의    │  타겟 메타데이터 m_T →
    │  GBN 파라미터 사용   │  가상 노드 T 추가 →
    │                     │  수식 (2)로 파라미터 예측
    └─────────────────────┘
```

**핵심 구성 요소:**
- **백본**: ResNet-18 (ImageNet 사전학습)
- **GBN 레이어**: 각 BN을 대체하며 도메인별 $\{\mu_v, \sigma_v^2, \gamma_v, \beta_v\}$ 유지
- **그래프 $\mathcal{G}$**: 노드 = 도메인, 엣지 = 메타데이터 기반 유사도
- **버퍼 $M$**: 테스트 시 실시간 정제를 위한 고정 크기 슬라이딩 윈도우

---

### 2.4 성능 향상 결과

#### CompCars 데이터셋 (DeCaf 특징)

| 방법 | 평균 정확도 |
|------|------------|
| Baseline | 54.0% |
| MRG-Direct (이전 SOTA) | 58.1% |
| MRG-Indirect (이전 SOTA) | 58.2% |
| **AdaGraph (metadata)** | **60.1%** |
| **AdaGraph (images)** | **60.8%** |
| AdaGraph + Refinement | **60.9%** |
| DA upper bound | 60.9% |

#### Portraits 데이터셋 (ResNet-18, Ablation)

| 방법 | 십년 단위 | 지역 단위 |
|------|----------|----------|
| Baseline | 82.3% | 89.2% |
| AdaGraph BN | 86.3% | **91.6%** |
| AdaGraph Full | 87.0% | 91.0% |
| **AdaGraph + Refinement** | **88.6%** | **91.9%** |
| DA upper bound | 89.1% | 92.1% |

#### CarEvolution (연속 DA)

| 방법 | 정확도 |
|------|--------|
| Baseline SVM | 39.7% |
| CMA+GFK (이전 SOTA) | 43.0% |
| LLRESVM+EDA (이전 SOTA) | 44.3% |
| **Baseline + Refinement Full** | **47.3%** |

---

### 2.5 한계점

1. **메타데이터 의존성**: 메타데이터의 품질과 표현 방식에 크게 의존. 지역 기반 인코딩이 문화·역사적 요인을 충분히 포착하지 못하는 경우 성능 저하 (논문 내 Across Regions 시나리오에서 관찰)
2. **그래프 정적 구조**: 훈련 후 그래프 구조가 고정되어 새로운 도메인이 추가될 때 점진적 업데이트가 어려움 (저자들 스스로 한계로 인정, Future Work로 언급)
3. **도메인 관계 수동 정의**: 메타데이터가 없는 상황에서 도메인 간 관계를 자동으로 추론하는 능력이 제한적
4. **벤치마크 한정성**: 자동차/인물 사진 등 제한된 시각적 도메인에서만 검증. 더 다양한 도메인 시프트(의료 영상, 위성 이미지 등)로의 일반화는 미검증
5. **버퍼 크기 민감성**: 연속 적응 시 버퍼 크기($|M| = 16$)와 $\alpha = 0.1$ 하이퍼파라미터가 성능에 영향을 미치나, 자동 조정 메커니즘 없음

---

## 3. 일반화 성능 향상 가능성

### 3.1 그래프 기반 파라미터 보간의 일반화 효과

그래프를 통한 파라미터 예측 (수식 2)은 본질적으로 **비모수적 보간(non-parametric interpolation)** 으로, 알려진 도메인의 파라미터를 가중 평균하여 미지 도메인을 추정합니다. 이는:

$$\hat{\theta}_\mathcal{T} = \frac{\sum_{v \in \mathcal{V}} e^{-d(m_\mathcal{T}, \phi(v))} \cdot \theta_v}{\sum_{v \in \mathcal{V}} e^{-d(m_\mathcal{T}, \phi(v))}}$$

- **도메인 다양성 활용**: 보조 도메인의 수가 증가할수록 파라미터 공간을 더 세밀하게 커버하여 임의의 타겟에 더 잘 적응 (실험에서 20개 이상 보조 도메인 시 DA 상한에 근접)
- **정규화 효과**: 수식 (10)에서 그래프 엣지 가중치 기반 scale/bias 파라미터 계산은 암묵적 정규화로 작용하여 데이터가 적은 도메인에서 과적합 방지

### 3.2 GBN의 도메인 불변 표현 학습

$$\text{GBN}(x, v, \mathcal{G}) = \gamma_v^\mathcal{G} \cdot \frac{x - \mu_v}{\sqrt{\sigma_v^2 + \epsilon}} + \beta_v^\mathcal{G}$$

- 도메인 공통 파라미터 $\theta^a$ (conv, fc layers)는 모든 도메인을 가로질러 공유되어 **도메인 불변 특징(domain-invariant features)** 을 학습
- 도메인 특화 파라미터 $\theta_k^s$는 해당 도메인의 분포를 포착
- 이 분리 학습 전략은 소스 도메인 지식을 타겟에 효과적으로 전이

### 3.3 연속 적응을 통한 점진적 일반화

수식 (11)의 지수 이동 평균 업데이트는 타겟 도메인 데이터가 순차적으로 도착하는 현실적 시나리오에서 모델을 점진적으로 정제합니다:

$$\mu_\mathcal{T} \leftarrow (1-\alpha)\mu_\mathcal{T} + \alpha\mu_M$$

이는 **그래프 기반 초기화 → 데이터 기반 정제**의 2단계 일반화 전략으로, 잘못된 초기 파라미터 예측에서도 회복 가능합니다.

### 3.4 보조 도메인 수와 일반화 성능의 관계

논문의 Figure 4 (부록 A.3)에 따르면:
- 보조 도메인 수 증가 → 일반화 성능 단조 증가
- 약 20개 이상의 보조 도메인에서 DA 상한(target data 사용)에 근접
- 이는 **그래프 커버리지(graph coverage)** 가 일반화의 핵심 요소임을 시사

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

**① PDA 연구의 새로운 기반 제공**

AdaGraph는 PDA 시나리오를 위한 최초의 딥러닝 프레임워크로, 이후 연구들이 참조할 수 있는 기준선(baseline)을 확립했습니다. 특히:
- GBN 레이어의 모듈화된 설계는 다른 백본(ResNet-50, ViT 등)에 쉽게 적용 가능
- 메타데이터 기반 파라미터 예측 패러다임은 이후 메타러닝(meta-learning) 기반 DA 연구에 영향

**② 그래프 신경망(GNN)과 도메인 적응의 결합**

AdaGraph는 GNN이 도메인 관계를 모델링하는 데 유용함을 보였습니다. 이후 연구에서 더 정교한 그래프 구조(예: GCN, GAT)를 DA에 적용하는 흐름을 촉진했습니다.

**③ 메타데이터 활용 DA의 가능성 제시**

카메라 포즈, 타임스탬프 등 부수적 정보(side information)를 DA에 활용하는 아이디어는 자율주행, 의료 영상 등 메타데이터가 풍부한 실제 응용에 중요한 시사점을 줍니다.

**④ 연속 학습(Continual Learning)과 DA의 교차점**

테스트 시 스트리밍 데이터 적응 전략은 **Test-Time Adaptation(TTA)** 연구와 직접 연결됩니다. AdaGraph의 버퍼 기반 통계량 업데이트는 이후 TTT(Test-Time Training), TENT 등 TTA 방법들의 선구자적 접근입니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래의 비교는 논문 원문에 근거하지 않으며, 2020년 이후 발표된 독립적인 연구들과의 관계를 AI 연구자 관점에서 분석한 것입니다.

### 5.1 Test-Time Adaptation (TTA) 방향

**TENT (Wang et al., ICLR 2021)**
- *"Tent: Fully Test-Time Adaptation by Entropy Minimization"* (Wang et al., ICLR 2021)
- AdaGraph의 연속 적응과 유사하게 테스트 시 엔트로피 최소화를 사용
- 차이점: 그래프 구조 없이 배치 정규화 파라미터만 업데이트, 단일 타겟 도메인 가정

$$\mathcal{L}_\text{TENT} = -\sum_y \hat{p}(y) \log \hat{p}(y)$$

AdaGraph 수식 (13)과 형태가 동일하나, AdaGraph는 그래프 기반 초기화를 선행하는 점에서 더 구조적

**T3A (Iwasawa & Matsuo, NeurIPS 2021)**
- *"Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization"*
- 프로토타입 기반 분류기 조정으로 타겟 도메인 적응
- AdaGraph의 이미지 기반 파라미터 추정(수식 3)과 개념적으로 유사

### 5.2 도메인 일반화 (Domain Generalization) 방향

**DomainBed (Gulrajani & Lopez-Paz, ICLR 2021)**
- *"In Search of Lost Domain Generalization"* (Gulrajani & Lopez-Paz, ICLR 2021)
- 다양한 DG 방법들의 표준 벤치마크 제공
- AdaGraph의 한계(제한된 벤치마크)를 극복하기 위한 체계적 평가 프레임워크

**SWAD (Cha et al., NeurIPS 2021)**
- *"SWAD: Domain Generalization by Seeking Flat Minima"*
- 도메인 일반화에서 플랫한 손실 경관 탐색으로 일반화 향상
- AdaGraph가 GBN 정규화로 암묵적으로 추구하는 일반화와 다른 최적화 관점

### 5.3 그래프 기반 도메인 적응 방향

**GCAN (Ma et al., CVPR 2019 이후 확장 연구들)**

AdaGraph 이후 GNN을 더 심층적으로 활용하는 연구들:
- 노드가 도메인이 아닌 샘플을 나타내는 미시적(micro) 그래프 방식
- 도메인 간 관계를 학습 가능한 파라미터로 끝-대-끝 학습

### 5.4 비교 요약표

| 특성 | AdaGraph (2019) | TENT (2021) | DomainBed (2021) | GNN-based DA |
|------|----------------|-------------|------------------|--------------|
| 타겟 데이터 필요 (훈련) | ✗ | ✗ | ✗ | △ |
| 메타데이터 활용 | ✓ | ✗ | ✗ | ✗ |
| 그래프 구조 | 도메인 수준 | ✗ | ✗ | 샘플 수준 |
| 연속 적응 | ✓ | ✓ | ✗ | ✗ |
| 파라미터 예측 (미지 도메인) | ✓ | ✗ | ✗ | ✗ |

---

## 6. 향후 연구 시 고려사항

### 6.1 그래프 구조의 동적 학습

현재 AdaGraph는 메타데이터 거리 함수 $d$가 사전에 고정됩니다. 향후에는:

$$\omega_\theta(v_1, v_2) = \text{MLP}_\theta(\phi(v_1), \phi(v_2))$$

와 같이 **학습 가능한 엣지 가중치**를 도입하여 데이터 기반 도메인 관계 학습이 가능합니다.

### 6.2 메타데이터 없는 시나리오 강화

수식 (3)의 이미지 기반 도메인 확률 추정을 강화하여, 메타데이터가 전혀 없어도 이미지 특징으로부터 도메인 임베딩을 자동 추론하는 방향이 중요합니다.

### 6.3 대규모 언어모델(LLM)/비전-언어 모델과의 결합

텍스트 메타데이터(예: "도시 환경의 낮 시간 자율주행")를 CLIP 등의 모델로 인코딩하여 메타데이터 표현을 더 풍부하게 만드는 방향이 유망합니다.

### 6.4 개인 정보 보호(Privacy-Preserving) 연속 적응

버퍼 $M$에 실제 타겟 이미지를 저장하는 방식은 개인 정보 문제를 야기할 수 있습니다. **Federated Learning** 패러다임과 결합하여 원시 데이터 없이 통계량만 공유하는 방향이 필요합니다.

### 6.5 세그멘테이션/검출 등 태스크 확장

현재 분류 태스크에 한정되어 있으나, 자율주행의 의미론적 분할(semantic segmentation)과 같은 더 복잡한 태스크로의 확장이 실용적 가치를 높일 것입니다.

---

## 참고 자료

**원본 논문:**
- Mancini, M., Rota Buló, S., Caputo, B., & Ricci, E. (2019). **AdaGraph: Unifying Predictive and Continuous Domain Adaptation through Graphs**. arXiv:1903.07062v3. [https://arxiv.org/abs/1903.07062](https://arxiv.org/abs/1903.07062)

**논문 내 주요 인용 참고문헌:**
- Yang & Hospedales (CVPR 2016). *Multivariate regression on the grassmannian for predicting novel domains.* [ref. 36]
- Carlucci et al. (ICCV 2017). *Autodial: Automatic domain alignment layers.* [ref. 3]
- Li et al. (Pattern Recognition 2018). *Adaptive batch normalization for practical domain adaptation.* [ref. 20]
- Hoffman et al. (CVPR 2014). *Continuous manifold based adaptation for evolving visual domains.* [ref. 14]
- Li et al. (IEEE T-PAMI 2018). *Domain generalization and adaptation using low rank exemplar SVMs.* [ref. 19]

**2020년 이후 비교 분석을 위해 참조한 연구 (원문 외 추가 참조):**
- Wang, D. et al. (ICLR 2021). *Tent: Fully Test-Time Adaptation by Entropy Minimization.*
- Gulrajani, I. & Lopez-Paz, D. (ICLR 2021). *In Search of Lost Domain Generalization.*
- Cha, J. et al. (NeurIPS 2021). *SWAD: Domain Generalization by Seeking Flat Minima.*
- Iwasawa, Y. & Matsuo, Y. (NeurIPS 2021). *Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization.*

> **⚠️ 정확도 관련 고지:** 2020년 이후 최신 연구 비교 분석 부분은 논문 원문에 포함된 내용이 아니며, 해당 연구들의 논문 제목 및 학회 정보는 제가 알고 있는 범위에서 제시한 것입니다. 구체적인 수치 비교나 직접 인용이 필요한 경우 각 논문 원문을 직접 확인하시기 바랍니다.
