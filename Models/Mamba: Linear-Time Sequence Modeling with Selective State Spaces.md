# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

### 1. 핵심 주장과 주요 기여

**Mamba** 논문의 핵심 주장은 **선형-시간 복잡도를 유지하면서도 Transformer 수준의 성능을 달성할 수 있다**는 것입니다. 이전의 구조화된 상태공간 모델(SSM)들이 이산(discrete) 데이터, 특히 언어와 같은 정보-밀집형 데이터에서 부족했던 근본적인 이유를 식별합니다: **입력에 기반한 내용-인식적 선택(content-based selection)** 능력의 부재입니다.[1]

주요 기여는 다음과 같습니다:

- **선택적 메커니즘(Selection Mechanism)**: SSM 매개변수를 입력의 함수로 만들어 모델이 시퀀스 길이 차원에서 정보를 선택적으로 전파하거나 잊을 수 있도록 함[1]
- **하드웨어-인식 알고리즘**: GPU 메모리 계층 구조를 활용한 병렬 스캔 알고리즘으로 시간 복잡도 $$O(BLD N)$$을 효율적으로 계산[1]
- **간소화된 아키텍처**: 어텐션이나 MLP 블록 없이 상태공간 모델만으로 구성된 Mamba 아키텍처[1]

***

### 2. 해결하고자 하는 문제와 제안 방법

#### 2.1 핵심 문제

Transformer의 자기-어텐션 메커니즘은 강력하지만 **이차 시간 복잡도** $O(L^2)$를 가지므로 긴 시퀀스에서 비효율적입니다. 선형 어텐션, 게이트 CNN, RNN 등 서브-이차 시간 아키텍처들이 개발되었지만, 언어와 같은 중요한 모달리티에서는 Transformer만큼 성능이 좋지 않았습니다.[1]

논문은 이 약점이 **이산 및 정보-밀집형 데이터에 대한 내용-기반 추론 능력의 부재**에서 비롯된다고 주장합니다. LTI(Linear Time-Invariant) 모델은 시간 불변 dynamics를 가지므로, 입력에 따라 선택적으로 정보를 필터링할 수 없습니다.[1]

#### 2.2 제안된 방법: 선택적 상태공간 모델(Selective SSM)

**기본 상태공간 모델의 구조**

전통적인 구조화된 SSM (S4)은 다음과 같이 정의됩니다:[1]

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

이산화 후 재귀 형태로는:[1]

$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = Ch_t$$

여기서 $\bar{A} = \exp(\Delta A)$, $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$ (Zero-Order Hold 이산화)[1]

**선택적 SSM의 혁신**

Algorithm 2에서 보듯이, 핵심 개선은 매개변수 $B$, $C$, $\Delta$를 **입력의 함수**로 만드는 것입니다:[1]

| 매개변수 | S4 (LTI) | S6 (선택적) |
|---------|----------|-----------|
| $A$ | $(D, N)$ 고정 | $(D, N)$ 고정 |
| $B$ | $(D, N)$ 고정 | $(B, L, N)$ 입력 의존 |
| $C$ | $(D, N)$ 고정 | $(B, L, N)$ 입력 의존 |
| $\Delta$ | $(D)$ 고정 | $(B, L, D)$ 입력 의존 |

구체적으로:[1]

- $s_B(x) = \text{Linear}_N(x)$
- $s_C(x) = \text{Linear}_N(x)$
- $s_\Delta(x) = \text{Broadcast}_D(\text{Linear}_1(x))$
- $\tau_\Delta = \text{softplus}$

이는 다음의 **게이팅 메커니즘과의 연결**을 가능하게 합니다 (Theorem 1):[1]

$$N=1, A=-1, B=1, s_\Delta = \text{Linear}(x), \tau_\Delta = \text{softplus}$$일 때:

$$g_t = \sigma(\text{Linear}(x_t))$$
$$h_t = (1-g_t)h_{t-1} + g_t x_t$$

이는 표준 RNN의 게이트 메커니즘과 동일합니다.[1]

#### 2.3 하드웨어-인식 알고리즘

선택적 매개변수 때문에 효율적인 global convolution을 사용할 수 없으므로, 논문은 **병렬 스캔 알고리즘**을 제안합니다. 핵심 기법들:[1]

1. **커널 융합(Kernel Fusion)**: 이산화와 재귀를 SRAM에서 수행
2. **병렬 스캔**: Work-efficient 병렬 스캔 알고리즘으로 순차성 극복
3. **재계산(Recomputation)**: 역전파 시 중간 상태를 저장하지 않고 재계산

결과적으로 메모리 요구사항은 FlashAttention과 동일하지만, 시간 복잡도는 $O(BLD\log L)$ (convolution 기반)에서 $O(BLD)$ (recurrent)로 개선됩니다.[1]

#### 2.4 단순화된 아키텍처

Mamba는 H3 아키텍처와 Transformer의 MLP 블록을 통합합니다:[1]

$$\text{Mamba Block} = \text{Linear Projection} \times \text{Activation} \times \text{Selective SSM} \times \text{Gated Linear Projection}$$

이를 확장 인자 $E=2$로 반복하여 Transformer의 $12D^2$ 매개변수와 일치시킵니다.[1]

***

### 3. 모델 구조 및 성능

#### 3.1 모델 구조 개요

Mamba는 다음의 **동질적 블록 구조**를 가집니다:[1]

```
입력 x (B, L, D)
    ↓
선형 투영 → 2×ED² 매개변수
    ↓
선택적 SSM 계층 → 시간 변화 dynamics
    ↓
활성화 함수 (SiLU/Swish)
    ↓
선형 투영 (ED² 매개변수)
    ↓
출력 y (B, L, D)
```

각 블록은 LayerNorm과 잔여 연결로 감싼 이 구조를 반복합니다.[1]

#### 3.2 성능 향상

**언어 모델링 (Scaling Laws)**

Pile 데이터셋에서 125M-1.3B 매개변수 범위로 평가한 결과:[1]

- **Transformer++** (PaLM/LLaMA 레시피): 강력한 기준선
- **Mamba**: 모든 기준선을 초과 (특히 긴 시퀀스에서)
- Mamba-3B는 **Transformer-7B의 성능 수준** (비슷한 사전학습 복잡도)

**다운스트림 평가 (Zero-shot)**[1]

Mamba-2.8B는 Pythia-2.8B, RWKV-3B보다 우수:

| 작업 | Pythia-2.8B | RWKV-3B | Mamba-2.8B |
|------|------------|---------|----------|
| 평균 복잡도 | 6.73 | 7.00 | **6.22** |
| Arc-E | 74.0% | 73.7% | **75.2%** |
| Arc-C | 64.1% | 67.8% | **69.7%** |
| HellaSwag | 74.0% | 73.7% | **75.2%** |

**종합 점수**: Mamba (63.3%) > OPT-6.7B (62.9%) > Pythia-6.9B (61.7%)[1]

**DNA 시퀀싱 (HG38)**

- 모델 크기 스케일링: HyenaDNA/Transformer++와 비교해 **3-4배 더 적은 매개변수**로 동등 성능[1]
- 컨텍스트 길이 스케일링: HyenaDNA는 긴 컨텍스트에서 악화되지만 Mamba는 **지속적으로 개선** (최대 1M 길이)[1]

**오디오 모델링 (SC09 음성 생성)**

- Mamba (6.1M): FID **0.94** → Mamba (24.3M): FID **0.67** (SaShiMi 1.99)[1]
- 소형 Mamba는 훨씬 더 큰 GAN/Diffusion 모델보다 우수[1]

**합성 작업**

- **Selective Copying**: S6 (Mamba의 선택적 층) **99.8%** vs. S4 18.3%, Hyena 30.1%[1]
- **Induction Heads**: Mamba는 256 길이에서 훈련하고 **1M 길이로 완벽 외삽** (모든 기준선 >2×)[1]

#### 3.3 계산 효율성

**훈련 시간**: 병렬 스캔이 PyTorch 기본 구현보다 **40배 빠름**[1]

**추론 처리량** (A100 80GB, 프롬프트 길이 2048):
- Mamba-1.4B: **1688 tokens/s** vs. Transformer-1.3B: **443 tokens/s** → **3.8배**
- Mamba-6.9B: **1814 tokens/s** vs. Transformer-6.7B: **490 tokens/s** → **3.7배**

KV 캐시 없이 훨씬 큰 배치 크기를 사용할 수 있기 때문입니다.[1]

***

### 4. 모델의 일반화 성능 향상 (중점)

#### 4.1 내용 기반 추론 능력

**선택 메커니즘의 역할**

$\Delta$ 매개변수는 현재 입력 $x_t$에 집중할지, 과거 상태를 유지할지 결정합니다:[1]

- $\Delta \to \infty$: 상태 리셋, 현재 입력 선택
- $\Delta \to 0$: 상태 유지, 현재 입력 무시

이는 세 가지 메커니즘적 효과를 제공합니다:[1]

**변수 간격(Variable Spacing)**: "um"과 같은 필러 토큰을 필터링하고 관련 정보만 기억

**컨텍스트 필터링(Filtering Context)**: 긴 컨텍스트에서도 성능이 **단조롭게 증가** (LTI 모델은 감소)

**경계 재설정(Boundary Resetting)**: 여러 문서를 연결할 때 시퀀스 간 정보 누수 방지

#### 4.2 외삽 능력 (Extrapolation)

**Induction Heads 작업 결과**:[1]

- 훈련: 길이 256
- 테스트: 길이 64-1,048,576

| 방법 | 테스트 길이 2⁶ | 2¹⁰ | 2²⁰ |
|-----|-------------|-----|-----|
| MHA-xPos | 0.85 | 0.70 | 0.04 |
| Hyena | 0.50 | 0.12 | 0.00 |
| Mamba | **1.00** | **1.00** | **1.00** |

Mamba는 4000배 더 긴 시퀀스로 **완벽하게 외삽**합니다.[1]

#### 4.3 스케일링 법칙 준수

Chinchilla 스케일링 프로토콜을 따를 때:[1]

- 모델 크기와 함께 **예측 가능한 성능 개선**
- 다양한 모달리티(언어, DNA, 오디오)에서 일관된 스케일링 행동
- Transformer와 유사한 스케일 지수, 더 나은 절대 성능

#### 4.4 다중 모달리티 일반화

| 모달리티 | 성능 지표 | 결과 |
|---------|----------|------|
| **언어** | 사전학습 복잡도 | Transformer 동급 이상 |
| **DNA** | 종 분류 정확도 (1M 길이) | 기준선 > 65% vs. Random 50% |
| **오디오** | FID (SC09) | 0.67 (Mamba-24M) vs. 0.74 (SaShiMi-23M) |

***

### 5. 모델의 한계

#### 5.1 연속-이산 스펙트럼 트레이드오프

논문에서 명시적으로 확인된 한계:[1]

- **이산 데이터(텍스트, DNA)**: 선택 메커니즘이 도움 → 실수 기반 SSM 선호
- **연속 신호(오디오, 비디오)**: 선택 메커니즘이 성능 해침 → 복소수 기반 SSM 선호

오디오에서는 "복소수" 매개변수를 사용해야 했습니다.[1]

#### 5.2 스케일링 평가의 제한

- 평가는 **1.3B 매개변수까지 제한**
- Llama(7B+), RWKV, RetNet 같은 큰 모델들과의 비교 부족
- SSM 스케일링의 엔지니어링 도전과제 미정의[1]

#### 5.3 다운스트림 적응성 미평가

- **혼합 정밀도 미세조정(MPFT)**: 맞춤형 CUDA 커널로 인해 도전적
- **매개변수 효율 미세조정(PEFT)**: 상태 역학의 복잡성
- **문맥 내 학습(ICL)**: 초기 평가 부족

후속 연구에서 Mamba의 다운스트림 학습 능력이 Transformer의 **38% 수준**임이 드러났습니다.[2]

#### 5.4 길이 외삽의 한계

최근 연구(2025)에서 Mamba의 **길이 외삽 능력**에 대한 문제가 지적되었습니다:[3]

- Mamba 모델의 숨겨진 상태가 입력 길이에 따라 **수렴하는 경향**
- 스펙트럼 분석: 전이 행렬 A의 고유값(eigenvalue)이 길이 확장을 방해
- 제안된 해결책: Δ 대신 A를 스케일링하면 더 나은 길이 외삽 가능[3]

***

### 6. 앞으로의 연구 영향과 고려사항

#### 6.1 최신 연구 기반 영향 분석

**6.1.1 기초 모델 백본으로서의 영향 (2024-2025)**

최근 연구들은 Mamba를 다양한 도메인의 기초 모델로 채용하고 있습니다:[4][5][6][7]

- **시계열 예측**: ss-Mamba (의미론적 임베딩 + 스플라인 기반 인코딩) - **선형 복잡도 유지하며 우수 성능**[4]
- **의료 이미지 분할**: Vision Mamba - 동적 도메인 일반화로 **교차-모달리티 견고성** 개선[5]
- **스펙트럼 이미징**: CS-Mamba - 마스크 훈련으로 **실제 데이터 일반화 능력 향상**[6]
- **음성 감정 인식**: HuBERT-Mamba - LSTM보다 우수, **63.75% 정확도**[7]
- **ECG 진단**: AMTCN (Mamba 기반) - 적응형 Mixup으로 **소표본 환경에서 82.48% 정확도** (21.88% 개선)[8]

**6.1.2 크로스 도메인 적응 및 전이 학습**

Transfer-Mamba는 **적응형 지식 클러스터링**으로 소수 샷(few-shot) 트래픽 예측에서 여러 도시 간 지식 전이를 실현했습니다. 이는 **Mamba의 일반화 성능이 선택 메커니즘의 이점과 결합될 때 강력**함을 시사합니다.[9]

**6.1.3 비전 백본으로의 확장**

- **GlobalMamba**: 이산 코사인 변환(DCT)을 통한 전역 직렬화로 ImageNet-1K에서 우수 성능[10]
- **DAMamba**: 동적 적응형 스캔으로 **기존 Vision Mamba 초과** (분류, 감지, 분할)[11]
- **MVNet**: 3D-CNN + Transformer + Mamba 하이브리드로 초분광 이미지 분류에서 높은 정확도 및 효율성[12]
- **SparX**: 신경 생물학 영감의 희소 교차 계층 연결로 시각 Mamba 확장성 개선[13]

**6.1.4 사전학습 패러다임**

- **MambaMIM**: 상태공간 토큰 보간(token-interpolation)을 통한 생성형 사전학습으로 의료 이미지에서 **긴 범위 표현 능력 향상**[14]
- **TSMamba**: 시계열 기초 모델 - **양방향 인코더**로 시간적 의존성 캡처[15]

#### 6.2 앞으로의 연구 시 고려할 점

**6.2.1 길이 외삽 및 스펙트럼 안정성**

최근 "Mamba Modulation" 연구(2025)는 전이 행렬 A의 스펙트럼 스케일링이 **Δ 스케일링보다 더 효과적**임을 보였습니다. 향후 연구는:[3]

- A 행렬의 동적 스케일링 메커니즘 개발
- 스펙트럼 수렴 거동에 대한 이론적 분석
- 초장문(100K+ 토큰) 시퀀스에서의 테스트

**6.2.2 다운스트림 학습 능력 강화**

최근 연구에서 Mamba의 문맥 내 학습이 Transformer의 38% 수준으로 낮다는 점이 드러났습니다. 개선 전략:[2]

- 맞춤형 PEFT(Parameter-Efficient Fine-Tuning) 알고리즘 설계
- 동적 시스템 이론을 활용한 혼합 정밀도 미세조정 안정성 증명
- 상태 역학을 고려한 프롬프트 튜닝 방법 개발

**6.2.3 연속-이산 데이터 통합**

- **적응형 매개변수화**: 입력 유형에 따라 자동으로 실수/복소수 전환
- **혼합 SSM 블록**: 단일 층 내에서 다중 모달리티 처리
- **교차 모달리티 사전학습**: 언어-비전-오디오 통합 기초 모델

**6.2.4 확장성 및 엔지니어링 과제**

- **대규모 모델 훈련**: 7B+ 매개변수에서의 성능 검증 필요
- **효율성 최적화**: 현재는 A100에 최적화되어 있으며, **다양한 하드웨어(TPU, H100, 모바일) 지원 필요**
- **양자화 및 프루닝**: 배포 환경에서의 추론 가속

**6.2.5 이론적 기반 강화**

- **선택 메커니즘의 근본 원리**: RNN 게이팅과의 연결을 넘어, 정보 이론적 해석
- **길이 외삽 한계의 원인**: 상태 수렴 수학적 분석
- **일반화 오류 경계**: PAC 러닝이나 VC 차원 관점에서의 분석

**6.2.6 특화된 응용 개발**

최신 연구의 패턴에 따르면:-[9][4]

- **시계열 예측**: 의미론적 + 시간적 인코딩 결합
- **의료 영상**: 도메인 일반화 메커니즘 통합
- **강화학습**: 경계 재설정 기능 활용한 에피소드 학습
- **로보틱스**: 장시간 상태 추적과 적응형 행동

***

## 종합 결론

Mamba는 **선형-시간 복잡도를 유지하면서도 Transformer 수준의 성능을 달성하는 기초 모델**로서 중요한 기여를 합니다. 선택 메커니즘은 이산 데이터에서의 내용-인식적 추론을 가능하게 하며, 하드웨어-인식 알고리즘은 이를 실용적으로 구현합니다.[1]

그러나 **다운스트림 학습 능력 제한**(ICL 38% 수준), **길이 외삽의 스펙트럼 문제**, **연속-이산 트레이드오프** 등이 아직 해결되지 않았습니다. 최신 연구들은 이 한계를 극복하기 위해 적응형 스캔, 스펙트럼 스케일링, 도메인별 사전학습 패러다임을 제안하고 있습니다.[2][3][1]

향후 Mamba가 **진정한 범용 기초 모델 백본**이 되려면:

1. 길이 외삽 안정성 개선
2. 다운스트림 미세조정 능력 강화
3. 다중 모달리티 통합
4. 대규모 모델에서의 검증

이 모두 필수적이며, 2025년의 활발한 연구들이 이 방향으로 진행 중입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/59851b3f-3b93-45f0-931b-73e825dafd48/2312.00752v2.pdf)
[2](https://openreview.net/forum?id=C3t6GMPnC5&noteId=lx43EPpt88)
[3](https://arxiv.org/pdf/2509.19633.pdf)
[4](https://arxiv.org/abs/2506.14802)
[5](https://link.springer.com/10.1007/s10278-025-01687-0)
[6](https://www.semanticscholar.org/paper/b1b576af4bb25459d41142c140137ed66394a5ec)
[7](https://ieeexplore.ieee.org/document/11189348/)
[8](https://ieeexplore.ieee.org/document/11212729/)
[9](https://www.sciencedirect.com/science/article/abs/pii/S1569190X25000012)
[10](http://arxiv.org/pdf/2410.10316.pdf)
[11](https://arxiv.org/html/2502.12627v1)
[12](https://arxiv.org/abs/2507.04409)
[13](https://ojs.aaai.org/index.php/AAAI/article/view/34103)
[14](https://arxiv.org/html/2408.08070)
[15](http://arxiv.org/pdf/2411.02941.pdf)
[16](https://ieeexplore.ieee.org/document/11178804/)
[17](https://www.mdpi.com/2227-7390/13/21/3481)
[18](https://arxiv.org/abs/2505.21198)
[19](https://pmc.ncbi.nlm.nih.gov/articles/PMC11871232/)
[20](https://arxiv.org/html/2412.00776v2)
[21](http://arxiv.org/pdf/2503.10392.pdf)
[22](http://arxiv.org/pdf/2404.18861.pdf)
[23](https://arxiv.org/pdf/2312.00752.pdf)
[24](https://www.themoonlight.io/ko/review/state-free-inference-of-state-space-models-the-transfer-function-approach)
[25](https://www.bitswithbrains.com/news/mamba:-building-more-powerful-and-efficient-foundation-models-without-transformers)
[26](https://www.sciencedirect.com/science/article/pii/S2590123025036606)
[27](https://arxiv.org/html/2506.14802v1)
[28](https://zilliz.com/learn/mamba-architecture-potential-transformer-replacement)
[29](https://www.nature.com/articles/s41598-025-87574-8)
