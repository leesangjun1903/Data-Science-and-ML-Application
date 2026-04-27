# TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

TokenFormer는 기존 Transformer의 **선형 투영(linear projection)**을 **토큰-파라미터 어텐션(Pattention)** 레이어로 대체함으로써, 모델 파라미터를 토큰으로 취급하는 완전한 어텐션 기반 아키텍처를 제안합니다. 이를 통해 **재학습 없이 점진적으로 모델을 확장**할 수 있습니다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **새로운 아키텍처** | 모델 파라미터를 토큰으로 토큰화하여 어텐션 메커니즘으로 처리 |
| **Pattention 레이어** | 입력 토큰이 Query, 파라미터 토큰이 Key/Value 역할 |
| **점진적 모델 스케일링** | 124M → 1.4B 파라미터로 재학습 없이 확장 |
| **훈련 비용 절감** | 동일 성능 대비 훈련 비용 약 $\frac{1}{10}$ 수준 |
| **다중 도메인 검증** | 언어 모델링 및 시각 모델링 모두에서 경쟁력 있는 성능 확인 |

---

## 2. 해결하고자 하는 문제, 제안 방법(수식), 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

기존 Transformer의 핵심 문제는 **선형 투영의 고정성**입니다:

$$Q = X \cdot W^Q, \quad K = X \cdot W^K, \quad V = X \cdot W^V$$

여기서 $W^Q, W^K \in \mathbb{R}^{d \times d_k}$, $W^V \in \mathbb{R}^{d \times d_v}$는 고정된 가중치 행렬입니다.

모델 크기를 늘리려면 채널 차원($d$) 변경이 필요하고, 이는 **전체 재학습**을 요구합니다. 이 문제는 두 가지 측면에서 발생합니다:

1. **Token-Token Interaction**: 어텐션 메커니즘 → 유연함
2. **Token-Parameter Interaction**: 선형 투영 → **고정된 구조, 유연하지 않음**

### 2.2 제안하는 방법: Pattention 레이어

#### 핵심 수식

입력 토큰 $\mathcal{I} \in \mathbb{R}^{T \times d_1}$과 출력 토큰 $\mathcal{O} \in \mathbb{R}^{T \times d_2}$에 대해, $n$개의 학습 가능한 파라미터 토큰:
- $K_P \in \mathbb{R}^{n \times d_1}$ (Key 파라미터 토큰)
- $V_P \in \mathbb{R}^{n \times d_2}$ (Value 파라미터 토큰)

**Pattention 연산:**

$$\text{Pattention}(X, K_P, V_P) = \Theta\left(X \cdot K_P^\top\right) \cdot V_P \tag{4}$$

여기서 $\Theta$는 수정된 softmax로, 표준 softmax의 $\exp + L_1\text{norm}$ 대신 ** $L_2\text{norm} + \text{GeLU}$ **를 사용합니다:

$$\hat{S}_i = f\left(\frac{A_i \times \sqrt{n}}{\sqrt{\sum_{j=1}^n |A_j|^2}}\right), \quad \forall i \in 1 \ldots n \tag{16}$$

이 선택의 이유는 표준 softmax의 그래디언트 소실 문제 때문입니다. 표준 softmax 그래디언트:

$$\frac{\partial S_i}{\partial A_j} = \frac{1}{\sqrt{d}} S_i(\mathbb{1}_{i=j} - S_j) \tag{15}$$

제안 방법의 그래디언트:

$$\frac{\partial \hat{S}_i}{\partial a_j} = \begin{cases} f' \frac{1}{\sqrt{n}} \frac{1}{\|A\|_2}(n - Z_i Z_j) & i = j \\ -f' \frac{1}{\sqrt{n}} \frac{1}{\|A\|_2} Z_i Z_j & i \neq j \end{cases} \tag{25}$$

$S_i S_j$ 대신 $Z_i Z_j$에 의존함으로써 더 부드러운 분포를 형성하고 **그래디언트 소실을 완화**합니다.

### 2.3 전체 모델 구조

#### TokenFormer 레이어 구조

**Pre-norm Transformer와 동일한 구조를 유지하되, 모든 선형 투영을 Pattention으로 대체:**

$$X_{\text{inter}} = X_{\text{in}} + \text{MHA}(\text{LN}(X_{\text{in}})) \tag{5}$$

$$X_{\text{out}} = X_{\text{inter}} + \text{FFN}(\text{LN}(X_{\text{inter}})) \tag{6}$$

**Multi-Head Self-Attention (단일 헤드 기준):**

$$Q = \text{Pattention}(X, K_P^Q, V_P^Q), \quad K = \text{Pattention}(X, K_P^K, V_P^K), \quad V = \text{Pattention}(X, K_P^V, V_P^V) \tag{7}$$

$$X_{\text{att}} = \text{softmax}\left[\frac{Q \cdot K^\top}{\sqrt{d}}\right] \cdot V \tag{8}$$

$$O_{\text{att}} = \text{Pattention}(X_{\text{att}}, K_P^O, V_P^O) \tag{9}$$

**Feed-Forward Network:**

$$O_{\text{ffn}} = \text{Pattention}(X_{\text{ffn}}, K_P^{\text{ffn}}, V_P^{\text{ffn}}) \tag{10}$$

#### 아키텍처 구성 요약

```
TokenFormer Layer:
├── Token-Parameter Attention (Pattention) → Q 생성
├── Token-Parameter Attention (Pattention) → K 생성
├── Token-Parameter Attention (Pattention) → V 생성
├── Token-Token Attention (표준 softmax)
├── Token-Parameter Attention (Pattention) → Output Projection
└── Token-Parameter Attention (Pattention) → FFN
```

### 2.4 점진적 모델 스케일링

기존 pre-trained 파라미터 토큰 $K_P^{\text{old}}, V_P^{\text{old}} \in \mathbb{R}^{n \times d}$에 새로운 파라미터 추가:

$$K_P^{\text{scale}} = \left[K_P^{\text{old}}, K_P^{\text{new}}\right], \quad V_P^{\text{scale}} = \left[V_P^{\text{old}}, V_P^{\text{new}}\right] \tag{11}$$

$$O = \text{Pattention}(X, K_P^{\text{scale}}, V_P^{\text{scale}}) \tag{12}$$

**Zero 초기화의 수학적 보장:**

새로운 Key 파라미터를 0으로 초기화하면:

$$\hat{A} = \begin{bmatrix} K_P \\ 0, \ldots, 0 \\ \vdots \end{bmatrix} \cdot X = \begin{bmatrix} A \\ 0 \\ \vdots \end{bmatrix} \tag{29}$$

$$\hat{S} = f\left(\frac{\hat{A}}{\sqrt{\sum_j \hat{A}_j^2}}\right) = \begin{bmatrix} S \\ 0 \\ \vdots \end{bmatrix} \tag{30}$$

$$\hat{O} = \left[V_P^\top, V_P^{\text{new}\top}\right] \cdot \hat{S} = O \tag{31}$$

따라서 기존 출력 분포를 **수학적으로 보장하며 보존**합니다.

### 2.5 성능 향상

#### 언어 모델링 (Zero-shot 평가, Pile 300B tokens)

| 모델 | #Param | Pile PPL↓ | 평균 Zero-shot Acc↑ |
|------|--------|-----------|---------------------|
| Pythia-160M | 160M | 29.64 | 40.1 |
| **TokenFormer-150M** | **150M** | **10.45** | **44.7** |
| Pythia-410M | 410M | 9.95 | 48.2 |
| **TokenFormer-450M** | **450M** | **8.28** | **52.0** |
| Pythia-1.3B | 1.3B | 7.51 | 55.2 |
| **TokenFormer-1.5B** | **1.5B** | **6.91** | **59.3** |

#### 시각 모델링 (ImageNet-1K Top-1 Acc)

| 모델 | #Param | Top-1 Acc |
|------|--------|-----------|
| ViT-B/16 (MAE) | 86M | 82.3 |
| **Ours-B/16** | **109M** | **82.5** |
| ViT-L/16 (MAE) | 307M | 82.6 |
| **Ours-L/16** | **407M** | **83.1** |

#### 스케일링 효율성

| 전략 | 1.4B 도달 비용 | Perplexity |
|------|----------------|------------|
| Transformer (scratch, 300B) | ~12,000 TPU hours | 11.63 |
| **TokenFormer (점진적, 30B)** | **~4,000 TPU hours** | **11.77** |
| Transformer (scratch, 30B) | ~4,000 TPU hours | 13.34 |

#### 연산 복잡도 비교

| 연산 | Transformer | TokenFormer |
|------|-------------|-------------|
| Total Non-Embedding Params | $N = 12 n_{\text{layer}} d_{\text{model}}^2$ | $N = 2n_{\text{layer}}d_{\text{token}}(n_q + n_k + n_v + n_o + n_{\text{ff}})$ |
| Total Training FLOPs | $2NT + 4n_{\text{layer}}d_{\text{model}}T^2$ | $2NT + 4n_{\text{layer}}d_{\text{token}}T^2$ |

TokenFormer는 $d_{\text{token}}$을 고정 유지하므로 **긴 시퀀스에서 Token-Token 연산 비용이 상수**로 유지됩니다.

### 2.6 한계점

논문이 명시적으로 인정한 한계(Section G.3):

1. **추론 오버헤드**: Token-parameter 연산은 네트워크 크기에 선형으로 증가 → 파라미터 확장 시 추론 속도 저하
2. **MoE 통합 과제**: 각 key-value 파라미터 쌍이 독립적 전문가로 기능할 때 효율적 통신이 어려움
3. **제한된 도메인**: 언어 및 시각 모델링 위주로 검증, 음성·그래프 등 다른 도메인 검증 미흡
4. **학습 초기 비용**: 최초 124M 모델은 여전히 처음부터 학습 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 지식 이전을 통한 일반화

TokenFormer의 점진적 스케일링은 **소형 모델의 학습된 표현을 보존**하면서 대형 모델로 확장합니다. 이는 일반화 관점에서 중요한 의미를 가집니다:

$$\hat{O} = O \quad \text{(Zero 초기화 시 출력 분포 불변)}$$

이 특성 덕분에 소형 모델이 학습한 일반적 표현을 손상 없이 대형 모델로 전달할 수 있습니다.

### 3.2 동적 파라미터 어텐션의 일반화 효과

표준 Transformer의 선형 투영:
$$O = X \cdot W$$

이는 모든 입력에 대해 동일한 가중치를 사용하는 **정적 연산**입니다. 반면 TokenFormer의 Pattention:

$$\text{Pattention}(X, K_P, V_P) = \Theta(X \cdot K_P^\top) \cdot V_P$$

입력 $X$에 따라 **동적으로 다른 파라미터 조합을 선택**합니다. 이는 사실상 **입력 조건부 파라미터 선택(input-conditioned parameter selection)** 메커니즘으로, 다양한 입력 분포에 더 유연하게 대응할 수 있습니다.

### 3.3 파라미터 효율적 파인튜닝(PEFT)을 통한 일반화

새로운 태스크 적응 시, 기존 파라미터는 고정하고 새로운 key-value 쌍만 추가:

$$K_P^{\text{new task}} = \left[K_P^{\text{pretrained}}, K_P^{\text{task-specific}}\right]$$

이는 LoRA와 개념적으로 유사하지만, 아키텍처 수준에서 네이티브로 지원됩니다. 도메인별 지식과 일반 지식의 **명시적 분리**가 가능하여 도메인 간 일반화에 유리합니다.

### 3.4 장문 컨텍스트 일반화

기존 Transformer의 훈련 비용:
$$\text{FLOPs} \propto 2NT + 4n_{\text{layer}}d_{\text{model}}T^2$$

TokenFormer의 훈련 비용:
$$\text{FLOPs} \propto 2NT + 4n_{\text{layer}}d_{\text{token}}T^2$$

$d_{\text{token}}$이 $d_{\text{model}}$보다 작게 유지되므로 **긴 컨텍스트에서 2차 항이 더 작음** → 장문 텍스트에서 더 효율적으로 학습 가능 → 긴 컨텍스트 일반화 능력 향상 기대.

### 3.5 멀티모달 일반화 잠재력

논문이 Future Work에서 제시한 비전-언어 통합 방법:

$$K_P^{\text{VL}} = \left[K_P^{\text{vision}}, K_P^{\text{language}}, K_P^{\text{alignment}}\right]$$

서로 다른 모달리티의 파라미터 토큰을 단순 연결로 통합할 수 있어, **모달리티 간 일반화**가 용이합니다.

### 3.6 해석 가능성을 통한 일반화 분석 가능성

Geva et al. (2021)의 연구에 따르면 FFN은 key-value 메모리 구조로 해석됩니다. TokenFormer는 이 구조를 명시적으로 구현함으로써:
- 어느 파라미터 토큰이 어떤 입력 패턴을 처리하는지 시각화 가능
- 과도하게 활성화되거나 비활성화된 파라미터 토큰 식별 가능
- 이를 통해 **일반화 실패 원인 분석 및 개선** 가능

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 스케일링 및 모델 재사용 관련 연구

| 연구 | 방법 | TokenFormer와의 비교 |
|------|------|---------------------|
| **Net2Net** (Chen et al., 2015/ICLR) | 뉴런 복제로 네트워크 확장 | 기존 분포 교란 발생; TokenFormer는 zero-init으로 분포 보존 |
| **Scaling Laws** (Kaplan et al., 2020/arXiv) | 모델/데이터 크기와 성능 관계 수립 | TokenFormer는 이 법칙을 준수하면서 비용 절감 |
| **bert2bert** (Chen et al., 2021/ACL) | 소형 BERT → 대형 BERT 재활용 | 파라미터 복사로 초기화; TokenFormer는 파라미터 축을 따라 자연스러운 확장 |
| **Learning to Grow** (Wang et al., 2023/ICLR) | 사전학습 모델을 효율적으로 성장 | 구조 변경 필요; TokenFormer는 차원 변경 없음 |
| **HyperCloning** (Samragh et al., 2024/arXiv) | 소형 모델로 대형 모델 초기화 | Table 8에서 TokenFormer가 PPL과 FLOPs 모두 우세 |

**Table 8 상세 비교 (논문 내):**

| 방법 | 모델 | 124M PPL | 354M PPL | 354M FLOPs | 757M PPL | 757M FLOPs |
|------|------|----------|----------|------------|----------|------------|
| Net2Net | Transformer | 16.4 | 14.3 | 6.8 | 12.1 | 10.1 |
| HyperCloning | Transformer | 16.4 | 14.1 | 6.8 | 12.0 | 10.1 |
| **Ours** | **TokenFormer** | **16.1** | **13.7** | **3.6** | **11.5** | **3.6** |

### 4.2 효율적 Transformer 관련 연구

| 연구 | 방법 | TokenFormer와의 관련성 |
|------|------|----------------------|
| **Efficient Transformers Survey** (Tay et al., 2020/arXiv) | Transformer 효율화 방법 체계화 | TokenFormer는 파라미터 축 확장이라는 새로운 방향 제시 |
| **Switch Transformer/MoE** (Fedus et al., 2022/JMLR) | 희소 MoE로 파라미터 효율화 | TokenFormer는 각 KV 쌍을 전문가로 볼 수 있는 dense MoE 해석 가능 |
| **LoRA** (Hu et al., 2022/ICLR) | 저랭크 행렬로 파인튜닝 | TokenFormer의 Zero-init 확장은 LoRA와 메커니즘 유사; 네이티브 아키텍처 수준 지원 |
| **Augmenting Self-Attention with Persistent Memory** (Sukhbaatar et al., 2020/ICLR) | 영구 메모리 토큰 추가 | TokenFormer와 유사한 개념; TokenFormer는 이를 모든 선형 투영에 일반화 |

### 4.3 대안 아키텍처 관련 연구

| 연구 | 방법 | TokenFormer와의 비교 |
|------|------|---------------------|
| **Mamba** (Gu & Dao, 2023/arXiv) | 선형 시간 SSM | Table 10에서 TokenFormer-900M이 Mamba-790M과 유사 성능; Mamba는 선형 복잡도, TokenFormer는 유연한 스케일링 |
| **RWKV** (Peng et al., 2023/EMNLP) | RNN+Transformer 혼합 | Table 10에서 TokenFormer-1.5B가 RWKV-1.5B보다 우세 (59.3 vs 54.3) |
| **H3** (Fu et al., 2023/ICLR) | SSM 기반 언어 모델 | Table 10에서 TokenFormer가 전반적으로 우세 |

### 4.4 비전 Transformer 관련 연구

| 연구 | 방법 | TokenFormer와의 비교 |
|------|------|---------------------|
| **ViT** (Dosovitskiy et al., 2021/ICLR) | 이미지 패치 토큰화 | TokenFormer-B/16이 ViT-B/16(MAE) 대비 82.5 vs 82.3 |
| **DeiT** (Touvron et al., 2021/ICML) | 지식 증류 ViT | DeiT-B/16 81.8 대비 TokenFormer-B/16 82.5 |
| **MAE** (He et al., 2022/CVPR) | 마스크 오토인코더 | TokenFormer는 MAE와 동일 하이퍼파라미터로 비교하여 우세 |

---

## 5. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

#### (1) 아키텍처 패러다임의 전환
"모든 것을 토큰화하라(Tokenize Everything)"는 개념은 데이터뿐만 아니라 **파라미터, 메모리, 상태**까지 토큰으로 처리하는 통합 프레임워크를 제시합니다. 이는:
- 기존의 데이터-파라미터 이분법적 사고를 탈피
- 어텐션 메커니즘 하나로 모든 계산을 통일하는 방향성 제시

#### (2) 지속적 학습(Continual Learning) 연구
Zero-init으로 새로운 파라미터 추가 시 기존 지식이 보존되므로, **재앙적 망각(Catastrophic Forgetting)** 문제의 근본적 해결책으로 발전 가능합니다.

#### (3) Foundation Model 개발 패러다임
현재 대형 언어 모델 개발은 막대한 비용의 재학습을 필요로 합니다. TokenFormer는 **점진적 성장 패러다임**을 통해:
- 소형 모델 → 중형 → 대형 모델로의 자연스러운 진화
- 각 단계별 지식의 누적 전달

#### (4) 파라미터 효율적 파인튜닝(PEFT) 발전
LoRA, Adapter 등의 방법과 달리, TokenFormer는 **아키텍처 수준**에서 PEFT를 지원합니다. 향후 태스크별 파라미터 토큰 집합의 효율적 관리 및 공유 연구로 발전 기대.

#### (5) MoE 패러다임과의 통합
각 key-value 쌍을 독립적 전문가로 해석하면, TokenFormer는 **Dense-to-Sparse MoE** 전환을 위한 자연스러운 시작점이 됩니다.

### 5.2 앞으로 연구 시 고려할 점

#### (1) 추론 효율성 개선 ⚠️
현재 Token-parameter 연산은 파라미터 수에 **선형 증가**합니다:
$$\text{FLOPs}_{\text{param}} \propto 2NT$$
대형 모델에서 추론 시 $N$이 매우 커질 수 있으므로, **희소 어텐션(Sparse Attention)** 또는 **MoE 라우팅**과의 통합이 필수적입니다.

#### (2) 학습 안정성의 체계적 분석
수정된 softmax ($L_2\text{norm} + \text{GeLU}$)가 경험적으로 효과적임을 보였지만, 이론적 수렴 보장이 부재합니다. 다양한 스케일과 데이터셋에서의 안정성 분석이 필요합니다.

#### (3) 파라미터 토큰 수($n$)의 최적화
현재 $n$은 숨겨진 차원 $d$와 동일하게 설정되어 있으나, 이것이 최적인지에 대한 체계적 연구가 필요합니다. Scaling Law 관점에서:
$$\text{Optimal } n = f(d, T, \text{dataset size})$$

#### (4) 더 다양한 도메인 검증
현재 논문은 언어 모델링과 이미지 분류에 집중되어 있습니다. 다음 도메인에서의 검증이 필요합니다:
- 음성 인식/생성
- 그래프 학습
- 강화 학습
- 멀티모달 생성 (이미지, 비디오)

#### (5) 비전-언어 통합 검증
논문이 Future Work로 제안한 비전-언어 통합은 아직 실험적으로 검증되지 않았습니다:
$$K_P^{\text{VL}} = \left[K_P^{\text{vision}}, K_P^{\text{language}}, K_P^{\text{new}}\right]$$
이 통합의 효과성과 최적 병합 전략에 대한 연구가 필요합니다.

#### (6) 기존 생태계와의 호환성
현재 TokenFormer는 GPT-2 스타일의 아키텍처와 호환되도록 설계되었으나, LLaMA, Falcon 등 다양한 최신 아키텍처와의 통합 방법론 연구가 필요합니다.

#### (7) 분산 학습 효율성
파라미터 토큰이 많아질수록 분산 학습 시 **메모리 분산 전략**이 복잡해집니다. 파라미터 병렬화(Parameter Parallelism)와의 통합 방법 연구가 필요합니다.

---

## 참고 자료

### 주요 참고 논문 (본 논문 내 인용)

1. **Wang et al. (2024) - TokenFormer** (본 논문): "TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters," arXiv:2410.23168v2, 2025. [[GitHub](https://github.com/Haiyang-W/TokenFormer)]

2. **Vaswani et al. (2017)**: "Attention is All You Need," NeurIPS 2017.

3. **Biderman et al. (2023)**: "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling," ICML 2023.

4. **Hu et al. (2022)**: "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.

5. **Chen et al. (2015)**: "Net2Net: Accelerating Learning via Knowledge Transfer," ICLR 2015.

6. **Kaplan et al. (2020)**: "Scaling Laws for Neural Language Models," arXiv:2001.08361.

7. **Gu & Dao (2023)**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752.

8. **Fedus et al. (2022)**: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," JMLR 2022.

9. **Geva et al. (2021)**: "Transformer Feed-Forward Layers Are Key-Value Memories," EMNLP 2021.

10. **Dosovitskiy et al. (2021)**: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR 2021.

11. **He et al. (2022)**: "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022.

12. **Tay et al. (2020)**: "Efficient Transformers: A Survey," arXiv:2009.06732.

13. **Sukhbaatar et al. (2020)**: "Augmenting Self-Attention with Persistent Memory," ICLR 2020.

14. **Peng et al. (2023)**: "RWKV: Reinventing RNNs for the Transformer Era," EMNLP 2023.

15. **Samragh et al. (2024)**: "Scaling Smart: Accelerating Large Language Model Pre-Training with Small Model Initialization," arXiv:2409.12903.

16. **Hendrycks & Gimpel (2016)**: "Gaussian Error Linear Units (GELUs)," arXiv:1606.08415.

17. **Wang et al. (2023c)**: "Learning to Grow Pretrained Models for Efficient Transformer Training," ICLR 2023.

---

> **⚠️ 정확도 주의사항**: 본 답변은 제공된 논문 PDF(arXiv:2410.23168v2)를 직접 분석하여 작성되었습니다. 2020년 이후 관련 연구 비교에서 TokenFormer 논문이 직접 인용하지 않은 외부 연구들과의 비교는 논문 내 실험 결과 및 공개된 정보 범위 내에서 기술하였습니다.
