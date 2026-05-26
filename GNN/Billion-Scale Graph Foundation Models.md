
# Billion-Scale Graph Foundation Models

---

## 1. 핵심 주장 및 주요 기여 (간결 요약)

파운데이션 모델은 대규모 사전학습과 경량 적응(lightweight adaptation)을 통해 언어 및 비전 분야를 변혁시켰지만, 이 패러다임을 일반적인 실세계 그래프로 확장하는 것은 매우 도전적인 과제이다.

이에 대응하여, 본 논문의 핵심 주장과 기여는 다음과 같이 요약된다:

| 기여 | 내용 |
|------|------|
| **최초 end-to-end 프레임워크** | 임의의 이종(heterogeneous) 10억 규모 그래프를 위한 GFM 구축 레시피 |
| **GraphBFF Transformer** | 실용적 billion-scale GFM을 위한 유연하고 확장 가능한 아키텍처 |
| **그래프 스케일링 법칙** | 일반 그래프를 대상으로 한 최초의 뉴럴 스케일링 법칙 도출 |
| **실증적 성능 검증** | 14억 파라미터 모델로 10개 다운스트림 태스크에서 zero-shot 성능 입증 |

GraphBFF(Graph Billion-Foundation-Fusion)는 임의의 이종, 10억 규모 그래프를 위한 billion-parameter GFM 구축의 최초 end-to-end 레시피이며, 그 핵심에는 실용적인 billion-scale GFM을 위해 설계된 유연하고 확장 가능한 아키텍처인 GraphBFF Transformer가 있다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

파운데이션 모델은 대규모 사전학습과 일반화를 통해 NLP, 비전, 멀티모달 학습을 변혁시켰지만, 비유클리드 구조(non-Euclidean structures)와 복잡한 관계형 시맨틱으로 특징지어지는 그래프에 이 능력을 확장하는 것은 고유한 도전과 새로운 기회를 동시에 제시한다.

구체적으로 논문이 해결하고자 하는 문제는 다음과 같다:

1. **이종성(Heterogeneity)**: 실세계 그래프는 다양한 타입의 노드와 엣지를 포함하며, 기존 모델은 동질적(homogeneous) 그래프에 한정
2. **규모(Scale)**: 수십억 노드/엣지를 효율적으로 처리하는 아키텍처 부재
3. **일반화(Generalization)**: 학습 중 보지 못한 그래프 도메인으로의 전이 능력 부재
4. **스케일링 법칙 부재**: 그래프에 대한 체계적인 neural scaling law 미확립

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) GraphBFF Transformer 아키텍처

GraphBFF Transformer는 billion-scale GFM 구축을 위한 유연하고 확장 가능하며 효과적인 아키텍처로, 두 가지 이종(heterogeneous) 어텐션 컴포넌트를 활용하고 sparse softmax를 통합하여 실세계 대규모 이종 그래프를 효율적으로 지원한다. 두 어텐션 컴포넌트가 표현력(expressiveness) 향상에 필수적임을 수식적으로 증명하였다.

GraphBFF Transformer의 핵심 어텐션 메커니즘은 이종 그래프에서 두 종류의 어텐션을 결합한다. 논문에서 사용된 어텐션 구조는 다음 형태로 이해될 수 있다:

**① 구조적 어텐션 (Structural Attention)**:

그래프 구조 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{T}_v, \mathcal{T}_e)$ (노드 집합, 엣지 집합, 노드 타입, 엣지 타입)에서, 노드 $i$와 $j$ 사이의 heterogeneous attention score는:

$$\text{Attn}_{\text{struct}}(i, j) = \frac{\mathbf{q}_{\tau(i)}^\top \mathbf{k}_{\tau(j)}}{\sqrt{d_k}}$$

여기서 $\mathbf{q}\_{\tau(i)}, \mathbf{k}_{\tau(j)}$는 노드 타입 $\tau$에 의존하는 query, key 벡터이고, $d_k$는 차원 크기이다.

**② 피처 어텐션 (Feature Attention)**:

```math
\text{Attn}_{\text{feat}}(i, j) = \text{softmax\_sparse}\!\left(\frac{\mathbf{Q}_i \mathbf{K}_j^\top}{\sqrt{d}}\right)
```

Sparse softmax를 도입하여 연산 효율을 높인다:

$$\text{sparse-softmax}(z_k) = \frac{\max(z_k - \lambda, 0)}{\sum_{k'} \max(z_{k'} - \lambda, 0)}$$

여기서 $\lambda$는 임계값(threshold) 파라미터로, 낮은 어텐션 가중치를 0으로 만들어 sparse한 그래프 구조를 효율적으로 처리한다.

**최종 어텐션**은 두 컴포넌트의 결합:

$$\text{Attn}(i,j) = \alpha \cdot \text{Attn}_{\text{struct}}(i,j) + (1 - \alpha) \cdot \text{Attn}_{\text{feat}}(i,j)$$

> ⚠️ 위 수식은 논문의 구조 설명을 기반으로 표현한 것이며, 정확한 수식 표기는 arXiv 원문(https://arxiv.org/pdf/2602.04768)을 직접 확인하기 바랍니다.

---

#### (B) 그래프 Neural Scaling Law

GraphBFF Transformer를 활용하여 데이터와 모델 크기 측면에서 임의의 그래프를 대상으로 한 최초의 뉴럴 스케일링 법칙을 제시하였으며, 이 법칙은 GFM에서 모델과 데이터 병목이 엄격하게 존재함을 보여주어, LLM에서 관찰된 것처럼 모델과 데이터가 함께 성장해야 함을 시사한다.

스케일링 법칙의 일반적 형태 (Chinchilla/GPT 스케일링 법칙의 그래프 버전):

$$\mathcal{L}(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty$$

여기서:
- $N$: 모델 파라미터 수
- $D$: 학습 데이터 샘플 수
- $A, B, \alpha, \beta$: 실험적으로 피팅된 상수
- $L_\infty$: 이론적 손실 하한

GraphBFF는 모델 용량이나 학습 데이터가 확장됨에 따라 손실이 예측 가능하게 감소함을 보여주며, 어느 요소가 병목인지에 따라 달라진다.

---

#### (C) 배칭 및 사전학습 방법론

GraphBFF 프레임워크는 대규모 GFM 구축을 위해 데이터 배칭(data batching), 사전학습(pretraining), 파인튜닝(fine-tuning)에 대한 구체적인 방법론을 제공한다.

논문에서는 **KL-Batching**과 **Round-Robin** 배칭 전략을 소개한다:

- **KL-Batching**: 배치 내 그래프 분포의 KL Divergence를 최소화하여, 이종 그래프 데이터를 균형 있게 샘플링
- **Round-Robin Batching**: 여러 도메인의 그래프를 순환적으로 배치에 포함시켜 도메인 편향 방지

사전학습 목적 함수는 노드/링크 레벨 예측을 포함한 마스킹 기반 자기지도 학습(self-supervised learning)으로, 일반적 형태:

$$\mathcal{L}_{\text{pretrain}} = -\sum_{v \in \mathcal{V}_{\text{mask}}} \log P_\theta(\mathbf{x}_v \mid \mathcal{G} \setminus \{v\})$$

---

### 2-3. 모델 구조 요약

```
GraphBFF Architecture
├── Input Layer
│   ├── Multi-type Node Feature Encoder (타입별 임베딩)
│   └── Multi-type Edge Feature Encoder
├── GraphBFF Transformer Layers (L개 반복)
│   ├── Structural Attention (이웃 구조 기반)
│   ├── Feature Attention (sparse softmax 적용)
│   └── Feed-Forward Network (FFN)
├── Pretraining Head
│   ├── Masked Node Prediction
│   └── Masked Link Prediction
└── Fine-tuning / Zero-shot Head
    ├── Node Classification/Regression
    └── Link Classification/Regression
```

GraphBFF는 일반 그래프 및 피처 분포를 위해 설계되었으며, 임의의 수의 토큰 타입을 지원한다.

---

### 2-4. 성능 향상

1.4억 파라미터 GraphBFF Transformer를 10억 샘플로 사전학습한 모델을 평가하였으며, 학습 중 보지 못한 그래프에서의 10개 다양한 실세계 다운스트림 태스크(노드 및 링크 레벨 분류·회귀)에 걸쳐 zero-shot 및 probing 성능에서 최대 31 PRAUC 포인트의 큰 차이로 주목할 만한 성능을 달성하였다.

| 평가 항목 | 성과 |
|-----------|------|
| 모델 규모 | **14억(1.4B) 파라미터** |
| 사전학습 데이터 | **10억(1B) 샘플** |
| 평가 태스크 수 | **10개** (학습 중 미관찰 그래프) |
| 최대 성능 향상 | **+31 PRAUC points** (zero-shot) |
| 태스크 유형 | 노드·링크 레벨 분류/회귀 |

---

### 2-5. 한계

논문은 산업 규모에서의 그래프 학습을 위한 실용적이고 원칙적인 기반으로 GFM을 만들기 위한 핵심 도전과 열린 기회들을 논의한다.

논문이 인정하거나 내재적으로 존재하는 한계:

1. **계산 비용**: 14억 파라미터 모델의 사전학습은 Meta 수준의 산업 인프라를 요구하며, 일반 연구기관의 재현 가능성이 낮음
2. **데이터 편향**: 학습 데이터가 특정 도메인(소셜 네트워크, 추천 시스템)에 편중될 경우 타 도메인 일반화 제한
3. **평가 범위**: 10개 태스크는 그래프 학습의 전체 스펙트럼(시계열 그래프, 지식 그래프, 생물학 그래프 등)을 완전히 커버하지 못함
4. **이종 그래프 표준 부재**: 다양한 산업 그래프 간 통일된 토큰화/피처 표현 기준이 확립되지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 일반화(generalization) 성능과 직결되는 핵심 요소들을 집중 분석한다.

### 3-1. Zero-shot 및 Few-shot 일반화

1.4억 파라미터 GraphBFF Transformer를 10억 샘플로 사전학습하여 학습 중 보지 못한 그래프에서의 10개 다양한 실세계 다운스트림 태스크에서, 소수 샷(few-shot) 설정을 포함하여 최대 31 PRAUC 포인트의 큰 차이로 주목할 만한 zero-shot 및 probing 성능을 달성하였다.

이는 GFM이 **전이 학습(transfer learning)** 을 통해 도메인 불문 일반화가 가능함을 실증한다.

### 3-2. 스케일링을 통한 일반화

GraphBFF Transformer를 활용하여 데이터와 모델 크기 측면에서 임의의 그래프에 대한 최초의 뉴럴 스케일링 법칙을 제시하였으며, 이 법칙들은 모델과 데이터 병목이 GFM에서 엄격하게 존재함을 보여주어 LLM에서 관찰된 것처럼 모델과 데이터가 함께 성장해야 함을 시사한다.

즉, 모델 파라미터와 학습 데이터가 함께 증가할수록 일반화 성능도 예측 가능하게 향상된다.

### 3-3. 이종 어텐션 구조와 일반화

두 가지 이종 어텐션 컴포넌트와 sparse softmax를 활용함으로써 Transformer가 실세계 대규모 이종 그래프를 효율적으로 지원하며, GraphBFF Transformer의 두 어텐션 컴포넌트가 표현력 향상에 필수적임을 수식적으로 증명하였다.

이는 다양한 그래프 구조에 걸쳐 표현력이 유지됨을 의미하므로, 미관찰 도메인으로의 일반화에 기여한다.

### 3-4. 전이 가능한 사전학습 표현

파운데이션 모델은 규모(scale), 범용성(general-purpose nature), 이종 데이터 소스에 걸친 사전학습으로 특징지어지며, 전이 가능한 귀납적 편향(transferable inductive biases)을 포착하도록 구축되어 최소한의 태스크별 지도(supervision)로 강력한 성능을 발휘한다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려점

### 4-1. 연구에 미치는 영향

#### (가) GFM 연구의 방향성 정립
GFM에 관한 다양한 노력을 세 가지 핵심 구성요소(backbone 아키텍처, 사전학습 전략, 적응 메커니즘)로 이루어진 모듈식 프레임워크로 통합하며, GFM을 보편적(universal), 태스크별(task-specific), 도메인별(domain-specific)로 분류하고 대표적 방법들을 검토한다.

GraphBFF는 이 분류에서 **보편적 GFM**의 최초 실용적 구현 사례를 제공함으로써, 이후 연구의 기준점(baseline)이 된다.

#### (나) 그래프 스케일링 법칙의 확립
본 논문이 제시한 그래프 neural scaling law는 향후 GFM 연구에서 다음을 가능케 한다:
- 최적 모델 크기 및 데이터 크기 사전 예측
- 계산 예산(compute budget) 최적 배분
- LLM 스케일링 법칙의 그래프 도메인 확장

#### (다) 관련 연구 분야에 파급 효과

전이 가능성(transferability)과 창발적 능력(emergent capabilities)을 포함한 이론적 기반, 그리고 구조적 정렬(structural alignment), 이종성(heterogeneity), 확장성(scalability), 평가(evaluation) 등의 핵심 도전을 부각시킨다.

#### (라) 산업 응용 가능성 확대
그래프 구조 데이터는 보안, 소셜 네트워크, 추천 시스템 등 다양한 도메인에 걸쳐 어디에나 존재한다. GraphBFF는 이들 도메인에 단일 pretrained 모델을 배포 가능한 기반을 마련한다.

---

### 4-2. 앞으로 연구 시 고려할 점

#### ① 그래프 토크나이제이션 표준화
LLM이 텍스트 토큰화 표준(BPE, WordPiece)을 통해 일반화를 달성했듯, 그래프의 구조적/의미적 특성을 포착하는 표준 그래프 토크나이저 개발이 필요하다.

#### ② 효율적 파인튜닝 전략
최근 서베이들은 GFM을 backbone(예: Graph Transformer, GNN, LLM 또는 하이브리드), 사전학습 전략(대조적, 생성적, 예측적 목적), 적응 메커니즘(파인튜닝, 프롬프트 튜닝, 테스트 타임 적응)으로 구성된 세 가지 빌딩 블록으로 분해하는 모듈식 시각을 제안한다.

LoRA, Adapter, Prompt Tuning 등 경량 적응 기법의 그래프 도메인 적용 연구가 요구된다.

#### ③ 재현 가능성 및 공정한 벤치마크
현재 GraphBFF는 Meta 내부 데이터셋에서 검증되었으므로, 공개 데이터셋에서의 공정한 재현 가능성을 확보하기 위한 표준 벤치마크 구축이 필요하다.

#### ④ 동적 그래프(Dynamic Graph)로의 확장
정적 그래프에 한정된 현재 프레임워크를 시간에 따라 변화하는 동적 그래프(temporal/dynamic graphs)로 확장하는 연구가 요구된다.

#### ⑤ 프라이버시 및 페더레이티드 학습
산업용 그래프 데이터는 대부분 민감 정보를 포함하므로, Federated Learning 환경에서 GFM을 학습하는 프라이버시 보존 방법론이 중요한 연구 방향이다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 접근법 | 그래프 유형 | 규모 | 일반화 방식 |
|------|------|------------|------------|------|------------|
| **HGT** (Hu et al.) | 2020 | 이종 그래프 Transformer | 이종 | ~179M 노드 | 타입 의존 어텐션 |
| **GraphMAE** | 2022 | 마스킹 기반 그래프 자기지도학습 | 동질 | 소/중규모 | 재구성 사전학습 |
| **OFA** (One For All) | 2023 | 텍스트 속성 그래프 + LLM | 텍스트 속성 | 중규모 | Cross-domain 전이 |
| **GraphFM** | 2024 | Perceiver-style encoder + 다중 그래프 사전학습 | 이종 | 7.4M 노드 | 152개 데이터셋 사전학습 |
| **GraphBFF** (본 논문) | 2026 | 이중 이종 어텐션 + KL-배칭 + 스케일링 법칙 | 임의 이종 | **14억 파라미터, 10억 샘플** | Zero-shot, Few-shot |

GraphFM은 다중 그래프 사전학습 프레임워크로, Perceiver-style 인코더를 사용하며 학습 가능한 고정 수의 잠재 토큰(latent tokens)이 교차-어텐션(cross-attention)을 통해 입력 노드 시퀀스를 처리한다. 이 잠재 토큰들은 각 그래프를 압축된 표현으로 압축하는 가상 노드 역할을 하여, 계산량이 그래프 크기와 분리되고 데이터셋 간 공유 잠재 공간이 생성된다.

GraphFM의 학습을 위해 80개의 실세계 그래프와 72개의 합성 그래프를 포함한 152개 데이터셋으로 구성된 사전학습 코퍼스를 구성하였으며, 전체 코퍼스는 740만 개 이상의 노드와 1억 6,390만 개의 엣지를 포함한다. 인기 있는 벤치마크 데이터셋은 미관찰 그래프에 대한 일반화 평가를 위해 사전학습에서 제외되었다.

GraphBFF는 이들 선행 연구 대비 **규모(10억 파라미터 이상), 이종성 처리, 스케일링 법칙 도출** 측면에서 질적으로 도약한 연구이다.

---

## 📚 참고 자료 (출처)

1. **Bechler-Speicher, M. et al. (2026)**. *Billion-Scale Graph Foundation Models*. arXiv:2602.04768. https://arxiv.org/abs/2602.04768

2. **Wang, Z. et al. (2025)**. *Graph Foundation Models: A Comprehensive Survey*. arXiv:2505.15116. https://arxiv.org/abs/2505.15116

3. **Zhao, Z. et al. (2024)**. *A Survey on Self-Supervised Graph Foundation Models: Knowledge-Based Perspective*. arXiv:2403.16137. https://arxiv.org/abs/2403.16137

4. **Liu, J. et al. (2023)**. *Towards Graph Foundation Models: A Survey and Beyond*. arXiv:2310.11829.

5. **GFMPapers GitHub Repository**. BUPT-GAMMA. https://github.com/BUPT-GAMMA/GFMPapers

6. **GraphFM** (2024). *A generalist graph transformer that learns transferable representations across diverse domains*. arXiv:2407.11907. https://arxiv.org/html/2407.11907v2

7. **Hu, Z. et al. (2020)**. *Heterogeneous Graph Transformer (HGT)*. The Web Conference 2020. https://arxiv.org/pdf/2003.01332

8. **KDD 2025**. *Graph Foundation Models: Challenges, Methods, and Open Questions*. ACM SIGKDD. https://dl.acm.org/doi/10.1145/3711896.3736568

9. **Wu, B. et al. (2025)**. *Graph Foundation Models for Recommendation: A Comprehensive Survey*. arXiv:2502.08346.
