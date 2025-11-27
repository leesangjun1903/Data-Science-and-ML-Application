# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

### 1. 핵심 주장과 주요 기여

**"Transformers are SSMs"**라는 제목으로 대표되는 이 논문의 핵심 주장은 **상태 공간 모델(State Space Models, SSM)과 Transformer의 주의 메커니즘이 이론적으로 밀접하게 연결되어 있다**는 것입니다. 저자들은 **구조화된 상태 공간 이중성(Structured State Space Duality, SSD)** 프레임워크를 제시하여 두 아키텍처의 관계를 체계적으로 설명합니다.[1]

**주요 기여는 다음과 같습니다:**

1. **구조화된 행렬과의 동치성**: SSM이 반-분리 가능 행렬(semiseparable matrices)과 동등하다는 것을 증명하여, SSM의 계산을 이 행렬의 곱셈 알고리즘으로 해석할 수 있음을 보임[1]

2. **선형 주의(Linear Attention) 이론의 개선**: 텐서 축약(tensor contraction) 언어를 통해 선형 주의의 증명을 단순화하고, **구조화된 마스크 주의(Structured Masked Attention, SMA)**로 일반화[1]

3. **SSM과 SMA의 이중성**: SSM의 선형 형태와 주의의 이차 형태가 서로 쌍대(dual) 관계에 있음을 증명[1]

4. **Mamba-2 아키텍처**: 이러한 이론적 통찰을 바탕으로 **2-8배 더 빠른** 새로운 SSM 레이어를 설계[1]

***

### 2. 해결하고자 하는 문제 및 제안하는 방법

#### 2.1 문제 정의

논문이 해결하고자 하는 핵심 문제들은:

- **SSM과 Transformer의 단절**: SSM의 개발이 Transformer 최적화와 독립적으로 진행되어, SSM을 이해하고 개선하기 어려움
- **SSM의 훈련 효율성**: 선택적 SSM(Selective SSM)은 우수한 성능을 보이지만 행렬 곱셈 단위(matrix multiplication units)를 활용하지 못해 하드웨어 효율성이 떨어짐
- **아키텍처 설계의 제약**: Transformer 생태계와의 통합이 어려워 SSM의 대규모 학습이 비효율적

#### 2.2 제안하는 방법 및 주요 수식

**기본 SSM 정의**:

$$ h_t = A_t h_{t-1} + B_t x_t $$
$$ y_t = C_t^\top h_t $$

여기서 $h_t \in \mathbb{R}^N$은 숨겨진 상태, $x_t$는 입력, $y_t$는 출력입니다.[1]

**행렬 변환 형태**:

논문의 핵심 혁신은 SSM을 행렬 변환으로 표현하는 것입니다:

$$ y_t = \sum_{s=0}^{t} C_t^\top A_t \cdots A_{s+1} B_s x_s $$

이를 행렬 형태로 나타내면:

$$ y = M x, \quad M_{ji} = C_j^\top A_j \cdots A_{i+1} B_i $$

여기서 $M \in \mathbb{R}^{T \times T}$는 **반-분리 가능 행렬**입니다.[1]

**반-분리 가능 행렬(N-Semiseparable Matrix)**:

$$ M_{ji} = C_j^\top A_j \cdots A_{i+1} B_i \quad \text{(순차 반-분리 가능 표현, SSS)} $$

이 행렬의 모든 부분행렬(lower triangular portion)의 랭크는 $N$ 이하입니다.[1]

**구조화된 마스크 주의(SMA)**:

선형 주의의 일반화로, 임의의 구조화된 행렬 $L$을 사용:

$$ Z = \text{contract}(V, K) \quad \text{// } (S, P, N) $$
$$ H = \text{contract}(L, Z) \quad \text{// } (T, P, N) $$
$$ Y = \text{contract}(Q, H) \quad \text{// } (T, P) $$

또는 동등하게 이차 형태로:

$$ Y = (L \circ (Q K^\top)) \cdot V $$

특히 1-반-분리 가능 마스크의 경우:

$$ L_{ij} = \begin{cases} a_i \times \cdots \times a_{j+1} & i \geq j \\ 0 & i < j \end{cases} $$

여기서 $a_i \in $은 입력 의존적 스칼라입니다.[1]

**SSD(State Space Duality) 알고리즘**:

블록 분해를 통한 효율적 계산:

1. **대각 블록** (Diagonal Blocks): 이차 SMA 형태로 계산
2. **저랭크 블록** (Low-Rank Blocks): 세 부분으로 인수분해

$$ \text{Right Factor: } B \text{ 블록 인수} $$
$$ \text{Center Factor: } A^{\times} \text{ 블록 인수 (1-SS 행렬)} $$
$$ \text{Left Factor: } C \text{ 블록 인수} $$

***

### 3. 모델 구조

#### 3.1 Mamba-2 아키텍처

Mamba-2는 다음과 같은 구조적 개선을 포함합니다:

**SSD 레이어의 핵심 개선사항**:

- **더 강한 A 구조**: 대각 구조에서 **스칼라 × 항등 구조**로 제한

$$ A_t = a_t \cdot I, \quad a_t \in \mathbb{R}^{(1,1)} $$

이로써 $A$를 스칼라로만 표현 가능.[1]

- **더 큰 헤드 차원**: Mamba-1의 $P=1$에서 $P \in \{64, 128\}$로 증가

- **더 큰 상태 크기**: 동일한 계산 비용 하에서 상태 확장 계수 $N$을 8배까지 증가 가능[1]

**다중 헤드 구조**:

Transformer의 다중 헤드 주의(Multi-Head Attention)와의 유사성 도입:

$$ \text{grouped-value attention (GVA) 구조로 데이터 의존적 프로젝션을 병렬화} $$

**시스템 최적화**:

- **텐서 병렬화(Tensor Parallelism)**: 동기화 포인트를 절반으로 감소
- **시퀀스 병렬화**: 재귀적 상태를 디바이스 간에 전달
- **가변 길이 시퀀스**: 패딩 제거 가능

#### 3.2 계산 복잡도

**정리 6.1** (SSD의 계산 효율성):[1]

상태 확장 계수 $N$, 헤드 차원 $P = N$일 때:

$$\text{Training FLOPs: } O(TN^2)$$
$$\text{Inference FLOPs: } O(TN)$$
$$\text{Inference Memory: } O(N^2)$$

모든 경계값이 타이트(tight)하며 행렬 곱셈으로 대부분의 연산이 구성됩니다.[1]

***

### 4. 성능 향상

#### 4.1 훈련 속도 개선

- **Mamba 대비 2-8배 빠른 훈련**: 블록 분해 알고리즘이 행렬 곱셈 유닛을 활용하여 달성[1]
- **FlashAttention-2와의 경쟁성**: 시퀀스 길이 2K에서 교점, 16K에서 6배 빠름[1]

#### 4.2 언어 모델 성능

**Chinchilla 스케일링 법칙 검증**:[1]

- Mamba-2는 Mamba와 Transformer++ (개선된 Transformer)를 **Pareto 지배**
- 동일한 perplexity에서 더 빠른 훈련/추론
- 동일한 벽시간(wall-clock time)에서 더 낮은 perplexity

**다운스트림 평가**:

- Pile에서 300B 토큰으로 훈련한 Mamba-2 2.7B: 
  - Mamba-2.8B 능가
  - Pythia-2.8B 능가
  - Pythia-6.9B까지 능가[1]

#### 4.3 합성 작업 성능

**다중 쿼리 연관 회수(Multi-Query Associative Recall, MQAR)** 작업:[1]

- 상태 크기를 제어한 경우에도 Mamba-2가 Mamba-1을 크게 능가
- 이는 더 큰 상태 크기뿐 아니라 SSD 구조 자체의 개선을 시사

***

### 5. 일반화 성능 향상 가능성

#### 5.1 이론적 일반화 근거

**정리 5.2** (효율적 자동회귀 주의의 특성화):[1]

모든 자동회귀 구조화된 마스크 주의는 반-분리 가능 행렬의 형태를 가져야 한다는 결과는 SSD의 **최적성**을 시사합니다.

#### 5.2 상태 확장을 통한 일반화 개선

최신 연구에 따르면:[2]

- **상태 크기의 증가**: Mamba-2는 Mamba-1의 최대 N=16 대비 N=64-256으로 확장 가능
- **더 큰 상태 용량**: 정보 저장 능력이 증가하여 복잡한 의존성 학습 가능
- **선택적 메커니즘의 개선**: 입력 의존적 $a_t$ 스칼라로 더 나은 정보 선택 가능

#### 5.3 장문맥(Long-Context) 학습 우수성

최신 비교 연구에 따르면:[3]

- SSM (Mamba-2)은 Transformer 모델과 달리 **KV 캐시 없이 장문맥 처리**
- 24GB 메모리에서:
  - Mamba-2: ~220K 시퀀스 길이 처리 가능
  - Transformer 모델: 65K 이하로 제한
  - **약 4배의 장문맥 우위**[3]

- 이는 정보 밀도가 높은 작업에서 **더 나은 일반화**를 의미

#### 5.4 구조적 편향의 개선

**"뇌처럼" 동작하는 구조**:[4]

- Transformer (데이터베이스 유사): 모든 관찰을 중요한 항목으로 기록
- SSM/Mamba (뇌 유사): 제한된 상태로 입력 처리
  - 이러한 편향은 특정 작업(예: 시계열 예측, 음성)에서 더 나은 귀납적 편향 제공
  - 장기 의존성 학습에 더 효율적

#### 5.5 최근 개선 방향[5]

**다중 스케일 SSM (MS-SSM)**:

$$\text{여러 해상도에서 시퀀스 동역학 모델링}$$

- 미세한 고주파 패턴과 거친 전역 추세를 모두 캡처
- 계산 효율성 유지 하에 일반화 성능 향상

***

### 6. 한계 및 제약사항

#### 6.1 표현력의 제한

- **비선형 연산 부족**: Softmax 제거로 일부 비선형성 상실[1]
- **고정된 상태 크기**: Transformer의 KV 캐시처럼 시퀀스 길이에 따라 확장 불가능
  - 이는 장문맥에서는 장점이지만 단기 정보 정확도는 제약[1]

#### 6.2 특정 작업에서의 성능 차이

최근 연구에 따르면:[6]

- **텍스트 재순위화 작업**: Mamba-2가 Transformer보다 학습 및 추론이 **덜 효율적**
- **복사 작업/귀납 헤드**: Mamba-1은 이들 작업에서 어려움[1]
  - Mamba-2가 개선되었지만 여전히 제약

#### 6.3 토큰 감소의 어려움

최근 연구에 따르면:[7]

- 기존 토큰 감소 방법을 SSM에 직접 적용 시 **성능이 급격히 감소**
- 특화된 SSM용 감소 방법 필요
  - Mamba-2에 대해 평균 **5.7%-13.1% 정확도 개선** 필요

#### 6.4 아키텍처 선택의 트레이드오프

**추론 효율성**:[8]

- SSM의 추론 속도는 전적으로 상태 차원 $N$에 의존
- Mamba-1 (N=16)이 특정 추론 작업에서 Mamba-2 (N=64-256)보다 더 효율적일 수 있음
- 이론적/실증적 분석이 여전히 진행 중

***

### 7. 논문의 앞으로의 영향 및 연구 고려 사항

#### 7.1 학계에 미치는 영향

**이론적 기여**:[1]

1. **행렬 변환 관점**: SSM을 구조화된 행렬의 곱셈으로 해석하여 새로운 알고리즘 설계 가능
2. **이중성 프레임워크**: SSM과 주의의 깊은 연결 규명으로 두 패러다임의 상호 이해 촉진

**실무적 적용**:[1]

1. **Transformer 최적화 기법의 이전**: 텐서 병렬화, 시퀀스 병렬화 등 Transformer 기법을 SSM에 적용 가능
2. **새로운 모델 설계 방향**: 구조화된 마스크 주의의 다양한 변형 가능 (Toeplitz, Fourier 등)[1]

#### 7.2 최근 연구 동향[2024-2025]

**1. 다양한 도메인으로의 확장**:

- **Vision**: VSSD (비-인과적 SSM), 2DMamba (거대 의료 이미지)[9][10]
- **Graph**: Graph Mamba (그래프 구조 데이터)[11]
- **Time Series**: C-Mamba (다변량 시계열)[12]
- **의료 영상**: SSM 기반 모델의 의료 이미징 적용[13]

**2. 하이브리드 모델**:

- **Zamba2**: Mamba-2와 Transformer의 조합으로 SOTA 성능과 효율성 달성[14]
- **Hydra**: 준-분리 가능 행렬 기반 양방향 SSM으로 GLUE에서 BERT 능가[9]

**3. 이론적 분석의 심화**:

- **일반화 오류 분석**: 선택적 SSM의 이론적 성능 보장 증명[2]
- **다중 스케일 모델링**: 계산 효율성 유지 하에 계층적 표현 학습[5]

#### 7.3 향후 연구 시 고려할 점

**1. 이론-실무 간극 해소**:

- Mamba-1 vs Mamba-2의 추론 효율성 트레이드오프에 대한 명확한 분석 필요
- 상태 크기의 최적값 결정 원리 규명

**2. 표현력 강화**:

- 비선형 연산 추가로 특정 작업(복사, 귀납)의 성능 개선
- Softmax 없는 주의 메커니즘의 최적화

**3. 도메인 특화 설계**:

- 비-인과적 작업(분류, 회귀)을 위한 SSM 확장 (예: Hydra)
- 구조화된 데이터(그래프, 트리)에 적합한 스캔 순서 설계

**4. 효율성 경계 탐색**:

- 토큰 감소, 양자화 등 압축 기법의 SSM 맞춤 개발
- 하드웨어 인식 구현의 최적화 (비-GEMM 연산 감소)

**5. 확장성 검증**:

- 초대형 모델(100B+ 파라미터)에서의 Mamba-2 성능 검증
- 장문맥 작업(책, 코드)에서의 일반화 능력 평가

***

### 결론

"Transformers are SSMs"는 **두 주요 시퀀스 모델 패러다임의 이론적 통합**을 이루어낸 중요한 논문입니다. 구조화된 상태 공간 이중성 프레임워크는 단순한 이론적 연결을 넘어 **2-8배 훈련 속도 개선**, **더 큰 상태 크기로 인한 일반화 능력 향상**, **장문맥 처리의 4배 우위** 등 실질적인 성능 개선을 가져왔습니다. 

특히 **비-선형 활성화 제거로 인한 표현력 제약**과 **특정 작업에서의 성능 차이** 등 남은 과제들이 있지만, 최근 연구의 다양한 확장(하이브리드 모델, 다중 스케일 구조, 비-인과적 변형)은 이러한 한계를 극복하고 SSM을 Transformer의 실질적인 대안으로 자리 잡히게 하는 방향으로 진행되고 있습니다.

***

## 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c6e3b26-33ca-491c-8f92-030169758f34/2405.21060v1.pdf)
[2](http://arxiv.org/pdf/2502.01473.pdf)
[3](https://arxiv.org/html/2507.12442v2)
[4](https://goombalab.github.io/blog/2025/tradeoffs/)
[5](https://openreview.net/forum?id=cCYWeCzAv0)
[6](https://arxiv.org/abs/2412.14354)
[7](https://arxiv.org/abs/2410.14725)
[8](https://goombalab.github.io/blog/2024/mamba2-part1-model/)
[9](https://arxiv.org/abs/2407.09941)
[10](https://arxiv.org/html/2412.00678)
[11](https://dl.acm.org/doi/10.1145/3637528.3672044)
[12](https://arxiv.org/abs/2406.05316)
[13](https://arxiv.org/abs/2406.03430)
[14](http://arxiv.org/pdf/2411.15242.pdf)
[15](https://arxiv.org/abs/2404.16112)
[16](https://www.semanticscholar.org/paper/88d6f9e3b3a5a99a525e89d80c92939d5c6bb33e)
[17](https://arxiv.org/abs/2405.02670)
[18](https://arxiv.org/abs/2412.09875)
[19](http://arxiv.org/pdf/2405.21060v1.pdf)
[20](https://arxiv.org/html/2407.18559)
[21](http://arxiv.org/pdf/2412.14354.pdf)
[22](https://arxiv.org/pdf/2410.14725.pdf)
[23](https://arxiv.org/html/2502.12627v1)
[24](https://arxiv.org/pdf/2312.00752.pdf)
[25](https://tinkerd.net/blog/machine-learning/state-space-models/)
[26](https://blog.outta.ai/178)
[27](https://arxiv.org/abs/2405.21060)
[28](https://www.themoonlight.io/en/review/long-context-state-space-video-world-models)
[29](https://junhan.blog/posts/Mamba2)
