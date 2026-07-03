# Finite Scalar Quantization: VQ-VAE Made Simple

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문은 VQ-VAE에서 사용되는 **Vector Quantization(VQ)** 을 **Finite Scalar Quantization(FSQ)** 으로 대체할 수 있음을 주장합니다. FSQ는 잠재 표현을 소수의 차원(일반적으로 $d < 10$)으로 투영하고, 각 차원을 고정된 유한 집합의 값으로 양자화하는 단순한 방식입니다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| **단순성** | 보조 손실(commitment loss, entropy loss 등) 없이 작동 |
| **Codebook 붕괴 방지** | 설계 자체로 높은 코드북 활용률(≈100%) 달성 |
| **Drop-in 대체** | MaskGIT, UViM 등 다양한 아키텍처에 VQ 대신 FSQ 적용 가능 |
| **확장성** | 대형 코드북에서 VQ보다 우수한 성능 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

VQ-VAE의 핵심 문제인 **코드북 붕괴(Codebook Collapse)** 입니다:

- 코드북 크기 $|\mathcal{C}|$가 증가할수록 많은 코드워드가 사용되지 않음
- 이를 해결하기 위해 복잡한 보조 장치 필요:
  - Commitment Loss
  - Codebook Reseeding
  - Code Splitting
  - Entropy Penalty
  - EMA(Exponential Moving Average) 업데이트
- 이러한 복잡성은 최적화를 어렵게 만들고 대규모 코드북 확장을 방해

### 2.2 VQ의 기존 수식

VQ-VAE의 양자화 과정:

$$\hat{z} = \arg\min_{c \in \mathcal{C}} \|z - c\|$$

VQ-VAE의 손실 함수:

$$\mathcal{L}_{VQ} = \mathcal{L}_{recon} + \|sg[z_e] - e\|_2^2 + \beta \|z_e - sg[e]\|_2^2$$

여기서:
- $z_e$: 인코더 출력
- $e$: 가장 가까운 코드북 벡터
- $sg[\cdot]$: stop-gradient 연산
- $\beta$: commitment cost 계수

### 2.3 FSQ의 제안 방법 (수식)

**핵심 아이디어**: 각 스칼라 채널을 독립적으로 유한한 값 집합으로 양자화

**바운딩 함수**:

$$f(z_i) = \lfloor L/2 \rfloor \cdot \tanh(z_i)$$

**양자화 과정**:

$$\hat{z} = \text{round}(f(z))$$

**암묵적 코드북 크기** ($d$개 채널, 각 채널이 $L_i$개 값을 가질 때):

$$|\mathcal{C}| = \prod_{i=1}^{d} L_i$$

**Straight-Through Estimator(STE)를 통한 역전파**:

$$\text{round ste}(x) = x + \text{sg}(\text{round}(x) - x)$$

즉, 순전파에서는 $\text{round}(x)$를 사용하고, 역전파에서는 그래디언트를 그대로 통과시킵니다.

**일반화된 바운딩 함수** (짝수 $L$을 지원하기 위한 비대칭 버전):

$$\text{half l} = (L-1)(1-\epsilon)/2, \quad \text{offset} = \begin{cases} 0 & L \text{가 홀수} \\ 0.5 & L \text{가 짝수} \end{cases}$$

$$f(z) = \tanh(z + \tan(\text{offset}/\text{half l})) \cdot \text{half l} - \text{offset}$$

### 2.4 VQ vs FSQ 비교

| 항목 | VQ | FSQ |
|------|-----|-----|
| 양자화 방식 | $\arg\min_{c \in \mathcal{C}} \|z - c\|$ | $\text{round}(f(z))$ |
| 그래디언트 | STE | STE |
| 보조 손실 | Commitment, Codebook, Entropy | **없음** |
| 특수 기법 | EMA, Code Splitting, 재초기화 | **없음** |
| 추가 파라미터 | 코드북 벡터 $\mid \mathcal{C} \mid \times d$ | **없음** |
| 잠재 차원 $d$ | 보통 $d \geq 512$ | 보통 $d < 10$ |

### 2.5 하이퍼파라미터

주요 하이퍼파라미터: 채널 수 $d$와 채널당 레벨 수 $\mathcal{L} = [L_1, \ldots, L_d]$

**권장 설정** (Table 1):

| 목표 $|\mathcal{C}|$ | $2^8$ | $2^{10}$ | $2^{12}$ | $2^{14}$ | $2^{16}$ |
|---|---|---|---|---|---|
| 권장 $\mathcal{L}$ | $[8,6,5]$ | $[8,5,5,5]$ | $[7,5,5,5,5]$ | $[8,8,8,6,5]$ | $[8,8,8,5,5,5]$ |

> **경험적 규칙**: $L_i \geq 5$ 를 모든 채널에 적용하는 것이 최적 성능

### 2.6 모델 구조

FSQ는 기존 VQ-VAE 구조에서 양자화 모듈만 교체하는 방식:

```
입력 이미지 x
    ↓
  인코더 E
    ↓ (d차원으로 투영, d << VQ의 차원)
  FSQ 모듈: z → f(z) → round(f(z)) = ẑ
    ↓ (암묵적 코드북 내 정수 인덱스로 변환)
  디코더 D
    ↓
  재구성 x̂
```

**Stage I**: GAN 손실로 오토인코더 훈련
**Stage II**: 양자화된 표현 $\hat{z}$ 위에 트랜스포머 모델 훈련

**적용된 모델**:
1. **MaskGIT** (Chang et al., 2022): 이미지 생성 (ImageNet 256×256)
   - Stage I: VQ-GAN 오토인코더 → FSQ-GAN으로 교체
   - Stage II: Masked Transformer
2. **UViM** (Kolesnikov et al., 2022): 깊이 추정, 색상화, Panoptic 분할
   - 트랜스포머 기반 VQ-VAE → FSQ-VAE로 교체

### 2.7 성능 결과

**MaskGIT (ImageNet 256×256)**:

| 모델 | CFG | Sampling FID↓ | Precision↑ | Recall↑ | 코드북 사용률 |
|------|-----|--------------|-----------|---------|------------|
| MaskGIT (VQ) | 0.1 | 4.509 | 0.860 | 0.465 | 81% |
| MaskGIT (FSQ) | 0.2 | **4.534** | **0.864** | 0.453 | **100%** |
| ADM (Diffusion) | 1.5 | 4.59 | 0.83 | 0.52 | - |

**UViM 태스크**:

| 태스크 | 지표 | VQ | FSQ |
|--------|------|-----|-----|
| NYU Depth v2 | RMSE↓ | $0.468 \pm 0.012$ | $0.473 \pm 0.012$ |
| COCO Panoptic | PQ↑ | $43.4$ | $43.2$ |
| ImageNet 색상화 | FID-5k↓ | $16.90$ | $17.55$ |

> FSQ는 모든 태스크에서 VQ 대비 **0.5~3% 수준의 소폭 성능 저하**만을 보임

**코드북 스케일링 비교**:

- FSQ: 코드북 크기 $2^{16}$에서도 $>2^{15}$개 코드워드 활용 (사용률 ≈100%)
- VQ: $2^{11}$ 초과 시 사용률 급감, $2^{10}$개 이상 활용 불가

### 2.8 한계점

1. **소규모 코드북에서의 열세**: 작은 $|\mathcal{C}|$에서는 VQ의 표현력이 더 높아 VQ가 소폭 우세
2. **고정 격자 구조**: FSQ의 코드북은 균일 격자(uniform grid)로 고정되어 데이터 분포에 적응하지 못함
3. **모델링 복잡도 증가**: FSQ 표현이 트랜스포머로 모델링하기 약간 더 어려움 (Compression Cost 지표 상 VQ보다 높음)
4. **의미론적 코드 학습 부재**: 개별 코드가 고정된 시각적 개념을 학습하지 않음 (VQ도 유사)
5. **코드북 크기 포화**: $|\mathcal{C}| \approx 2^{12}$ 이상에서 Sampling FID 개선이 포화됨

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 코드북 완전 활용을 통한 일반화

FSQ의 가장 중요한 일반화 관련 특성은 **코드북 사용률 ≈ 100%** 입니다.

수학적으로, 코드북 사용률이 높다는 것은 잠재 공간의 정보 용량이 최대로 활용됨을 의미합니다:

$$H(\hat{Z}) \approx \log_2 |\mathcal{C}| \quad \text{(최대 엔트로피 달성)}$$

반면 VQ에서 사용률이 $\rho$라면:

$$H(\hat{Z}_{VQ}) \approx \log_2(\rho \cdot |\mathcal{C}|) < \log_2 |\mathcal{C}|$$

즉, FSQ는 가용한 표현 용량을 최대한 활용하므로 **더 풍부하고 다양한 데이터 분포를 학습**할 수 있습니다.

### 3.2 Context(사이드 정보) 부재 시 강건성

논문의 UViM Panoptic 분할 실험에서 주목할 만한 결과:

| 모델 | PQ (컨텍스트 있음) | PQ (컨텍스트 없음) | 성능 저하 |
|------|-----------------|-----------------|---------|
| VQ | 43.4 | 39.0 | **-4.4** |
| FSQ | 43.2 | 40.2 | **-3.0** |

FSQ는 보조 정보(RGB 이미지) 없이도 VQ보다 **덜 열화**됩니다. 이는 FSQ가 외부 컨텍스트에 덜 의존적인 더 강건한 내재적 표현을 학습함을 시사합니다.

이 결과는 FSQ의 잠재 표현이 **더 자기완결적(self-contained)** 이며, 다양한 도메인 이동(domain shift)에 더 강할 수 있음을 암시합니다.

### 3.3 스케일링 법칙과 일반화

FSQ는 코드북 크기에 따라 **예측 가능하고 단조로운 성능 향상**을 보입니다:

$$\text{Reconstruction FID} \propto -\log_2 |\mathcal{C}| \quad \text{(FSQ)}$$

반면 VQ는 $|\mathcal{C}| > 2^{11}$ 이후 성능이 오히려 저하됩니다. 이러한 예측 가능한 스케일링 특성은 **새로운 태스크나 데이터셋에 FSQ를 적용할 때 하이퍼파라미터 선택이 용이**함을 의미합니다.

### 3.4 보조 손실 제거가 일반화에 미치는 영향

VQ의 보조 손실들은 특정 태스크나 데이터셋에 과적합(overfitting)될 위험이 있습니다:

- **Commitment loss**: $\beta$ 가중치 튜닝이 태스크에 민감
- **Entropy loss**: 특정 분포 가정 포함

FSQ는 이러한 손실 없이도 자연스럽게 균일한 코드 사용을 달성하므로, **태스크 간 전이 학습** 및 **제로샷 일반화**에 더 유리할 수 있습니다.

### 3.5 파라미터 효율성과 일반화

FSQ는 VQ보다 **적은 파라미터**를 사용합니다. 예를 들어 $|\mathcal{C}|=2^{12}$, $d_{VQ}=512$인 경우:

$$\text{VQ 코드북 파라미터} = |\mathcal{C}| \times d_{VQ} = 4096 \times 512 \approx 2\text{M}$$

$$\text{FSQ 코드북 파라미터} = 0$$

파라미터 수가 적으면서 성능을 유지한다는 것은 FSQ가 **더 효율적인 귀납적 편향(inductive bias)** 을 갖고 있음을 의미하며, 이는 일반적으로 일반화 성능과 양의 상관관계를 가집니다.

### 3.6 멀티태스크 및 멀티모달 일반화 가능성

논문이 매우 **이질적인 태스크들(이미지 생성, 깊이 추정, 색상화, 분할)** 에서 FSQ의 유효성을 입증했다는 점이 중요합니다. 단일 방법론이 이렇게 다양한 태스크에서 일관된 성능을 보인다는 것은 **강한 일반성**의 증거입니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### 4.1.1 이산 표현 학습 분야의 단순화

FSQ는 VQ-VAE 계열 연구의 **진입 장벽을 크게 낮춥니다**. 복잡한 VQ 안정화 기법 없이도 고품질 이산 표현을 학습할 수 있으므로, 더 많은 연구자들이 이산 표현 기반 모델을 실험할 수 있게 됩니다.

#### 4.1.2 대형 언어 모델과의 통합

논문이 제시한 바와 같이, FSQ는 **멀티모달 LLM**의 시각적 토큰화에 적합합니다. VQ의 불안정성 없이 안정적인 이산 시각 토큰을 제공할 수 있어, 언어 모델과의 결합이 더 용이합니다.

#### 4.1.3 신경 압축 분야와의 교차

FSQ는 신경 압축 분야에서 영감을 받았으며, 역으로 표현 학습 분야의 발전이 신경 압축에도 기여할 수 있는 선순환 구조를 만듭니다.

### 4.2 앞으로의 연구 시 고려할 점

#### 4.2.1 적응적 코드북 구조 탐색

FSQ의 균일 격자는 데이터 분포에 무관하게 고정됩니다. 향후 연구에서는:

$$\mathcal{C}_{\text{adaptive}} = \{f_\theta(e_i) : i = 1, \ldots, |\mathcal{C}|\}$$

와 같이 **학습 가능한 비균일 격자**와 FSQ의 안정성을 결합하는 방향을 고려할 수 있습니다.

#### 4.2.2 잔차 양자화(RVQ)와의 결합

Residual Quantization은 양자화 오류를 반복적으로 줄이는 방식입니다:

$$\hat{z}^{(k)} = \hat{z}^{(k-1)} + \text{FSQ}(z - \hat{z}^{(k-1)})$$

FSQ-RVQ 결합은 고품질 오디오/비디오 코덱(SoundStream, EnCodec 계열)에 유망한 방향입니다.

#### 4.2.3 비등방성(Anisotropic) FSQ

현재 FSQ는 모든 채널에 동일하거나 수동으로 설정된 레벨을 사용합니다. 향후에는:

$$L_i^* = \arg\min_{L_i} \mathcal{L}(\mathcal{L} = [L_1, \ldots, L_d])$$

를 통해 **채널별 최적 레벨을 자동으로 학습**하는 방법을 탐색해야 합니다.

#### 4.2.4 이론적 분석의 필요성

FSQ의 경험적 성공에도 불구하고, 다음 이론적 질문들이 미해결 상태입니다:

- 왜 균일 격자가 VQ와 동등한 표현력을 제공하는가?
- FSQ의 근사 오류 상한은 얼마인가?

$$\mathbb{E}[\|z - \hat{z}\|^2] \leq ?$$

- STE를 통한 그래디언트 근사의 편향(bias)은 FSQ와 VQ에서 어떻게 다른가?

#### 4.2.5 FSQ와 확산 모델의 결합

최근 **Latent Diffusion Models (LDM)** 은 VQ-VAE 또는 오토인코더의 잠재 공간에서 확산 과정을 수행합니다. FSQ를 LDM의 토크나이저로 사용하는 연구가 필요합니다.

#### 4.2.6 하이퍼파라미터 민감도 및 자동화

논문은 $L_i \geq 5$ 라는 경험적 규칙을 제시했지만, 이를 넘어서는 **자동 하이퍼파라미터 최적화** 방법이 필요합니다:

```math
[d^*, L_1^*, \ldots, L_d^*] = \arg\min \mathcal{L}_{downstream} \quad \text{s.t.} \prod_i L_i \approx |\mathcal{C}|_{target}
```

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 VQ 개선 연구들과의 비교

| 연구 | 방법 | FSQ 대비 차이점 |
|------|------|--------------|
| **SQ-VAE** (Takida et al., 2022) | 확률적 양자화 + self-annealing | 여전히 보조 손실 필요, FSQ보다 복잡 |
| **Huh et al. (2023)** | 재파라미터화 + 교대 최적화 | STE 문제를 직접 해결 시도, 여전히 VQ 프레임워크 유지 |
| **Improved VQGAN** (Yu et al., 2021) | $l_2$-정규화, ViT 기반 | 코드북 붕괴 부분 완화, 여전히 learnable codebook |

### 5.2 VQ 대안 연구들과의 비교

| 연구 | 방법 | FSQ 대비 차이점 |
|------|------|--------------|
| **RVQ** (Lee et al., 2022; Zeghidour et al., 2021) | 잔차 양자화 계층 | FSQ와 직교적 접근, 결합 가능성 높음 |
| **Product Quantization** (El-Nouby et al., 2022) | 코드북을 소규모 코드북의 곱으로 분해 | FSQ와 개념적으로 유사하나, 여전히 learnable |

### 5.3 이산 표현 활용 생성 모델과의 비교

| 연구 | 특징 | FSQ 관련성 |
|------|------|----------|
| **DALL-E** (Ramesh et al., 2021) | VQ-VAE + Autoregressive Transformer | FSQ가 VQ를 대체 가능 |
| **MaskGIT** (Chang et al., 2022) | Masked Transformer + VQ-GAN | **본 논문에서 FSQ 적용 실증** |
| **MUSE** (Chang et al., 2023) | Text-to-image, Masked Transformer | FSQ 적용 가능성 있음 |

### 5.4 연속 잠재 공간 기반 모델과의 비교

| 연구 | 방법 | FSQ 대비 장단점 |
|------|------|--------------|
| **Stable Diffusion/LDM** (Rombach et al., 2022) | 연속 잠재 공간에서 확산 | 이산 토큰 불필요, 언어 모델과 결합 어려움 |
| **DiT** (Peebles & Xie, 2023) | Diffusion Transformer | 연속 공간, 이산 표현 불필요 |

> **주의**: 2022년 이후의 일부 연구(예: Stable Diffusion, DiT)에 대한 세부 비교는 본 논문(2023년 10월 제출) 범위를 일부 벗어나며, 해당 논문들의 직접적인 FSQ 비교 실험은 수행되지 않았습니다.

---

## 참고 자료

**주요 논문 (직접 인용된 문헌)**:

1. **Mentzer et al. (2023)** — "Finite Scalar Quantization: VQ-VAE Made Simple", arXiv:2309.15505v2 *(본 논문)*
2. **Van Den Oord et al. (2017)** — "Neural Discrete Representation Learning (VQ-VAE)", NeurIPS 2017
3. **Esser et al. (2020)** — "Taming Transformers for High-Resolution Image Synthesis (VQGAN)", CVPR 2021
4. **Chang et al. (2022)** — "MaskGIT: Masked Generative Image Transformer", CVPR 2022
5. **Kolesnikov et al. (2022)** — "UViM: A Unified Modeling Approach for Vision with Learned Guiding Codes", NeurIPS 2022
6. **Takida et al. (2022)** — "SQ-VAE: Variational Bayes on Discrete Representation with Self-Annealed Stochastic Quantization", arXiv:2205.07547
7. **Huh et al. (2023)** — "Straightening Out the Straight-Through Estimator", arXiv:2305.08842
8. **Dhariwal & Nichol (2021)** — "Diffusion Models Beat GANs on Image Synthesis", NeurIPS 2021
9. **Yu et al. (2021)** — "Vector-Quantized Image Modeling with Improved VQGAN", arXiv:2110.04627
10. **Lee et al. (2022)** — "Autoregressive Image Generation using Residual Quantization", CVPR 2022
11. **Zeghidour et al. (2021)** — "SoundStream: An End-to-End Neural Audio Codec", IEEE/ACM TASLP
12. **Łancucki et al. (2020)** — "Robust Training of Vector Quantized Bottleneck Models", IJCNN 2020
13. **Ballé et al. (2016)** — "End-to-End Optimized Image Compression", arXiv:1611.01704
14. **Ho & Salimans (2022)** — "Classifier-Free Diffusion Guidance", arXiv:2207.12598
15. **Chang et al. (2023)** — "Muse: Text-to-Image Generation via Masked Generative Transformers", arXiv:2301.00704
