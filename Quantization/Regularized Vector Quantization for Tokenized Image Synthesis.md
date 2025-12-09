# Regularized Vector Quantization for Tokenized Image Synthesis

### 1. 핵심 주장과 주요 기여

본 논문(Zhang et al., 2023)은 **정규화된 벡터 양자화(Regularized Vector Quantization, Reg-VQ)** 프레임워크를 제시하여 이미지를 이산 표현으로 양자화하는 과정에서 발생하는 근본적인 문제들을 해결합니다.[1]

논문의 핵심 주장은 기존 양자화 방법들이 결정론적(deterministic) 또는 확률론적(stochastic) 접근 방식의 상충되는 요구사항으로 인해 어려움을 겪고 있다는 점입니다. 정규화된 벡터 양자화는 두 가지 관점에서의 정규화를 통해 이러한 문제들을 효과적으로 완화합니다.[1]

**주요 기여는 다음 세 가지입니다:**

1. **사전 분포 정규화(Prior Distribution Regularization)**: 코드북 붕괴와 낮은 코드북 활용도를 방지하는 정규화 프레임워크
2. **확률론적 마스크 정규화(Stochastic Mask Regularization)**: 훈련과 추론 단계 간의 불일치를 완화하며 손상되지 않은 재구성 목표를 달성
3. **확률론적 대조 손실(Probabilistic Contrastive Loss, PCL)**: 확률론적 양자화 영역에서 손상된 재구성 목표를 적응적으로 완화하는 보정된 메트릭

***

### 2. 해결하고자 하는 문제

#### 2.1 결정론적 양자화의 문제점[1]

결정론적 방법(예: VQ-GAN)은 Argmax를 통해 가장 유사한 토큰을 선택합니다:

$$\text{quantized}_i = z_{\arg\max_k p_{i,k}}$$

이러한 접근 방식은 다음과 같은 심각한 문제를 야기합니다:

- **코드북 붕괴(Codebook Collapse)**: 코드북 임베딩의 대부분이 0에 가까운 무효 값으로 변하여 실제로 사용되지 않음
- **추론 단계 불일치**: 훈련 중에는 최고 확률의 토큰을 선택하지만, 생성 모델의 추론 단계에서는 분포에서 랜덤하게 샘플링하기 때문에 불일치 발생

#### 2.2 확률론적 양자화의 문제점[1]

확률론적 방법(예: Gumbel-VQ)은 Gumbel-Softmax를 사용하여 분포에서 토큰을 샘플링합니다:

$$\text{quantized}_i = \text{Gumbel-Softmax}(p_i, \tau)$$

이 방식의 한계는 다음과 같습니다:

- **낮은 코드북 활용도(Low Codebook Utilization)**: 모든 코드북 임베딩이 유효하지만, 실제로는 극히 일부만 사용됨
- **손상된 재구성 목표(Perturbed Reconstruction Objective)**: 랜덤하게 샘플된 토큰이 원본 이미지와 완벽하게 정렬되지 않아 이미지 재구성 품질 저하

#### 2.3 코드북 활용 비교

Figure 1에서 시각화된 바와 같이:[1]
- **VQ-GAN**: 대부분의 코드북 임베딩이 붕괴되어 무효
- **Gumbel-VQ**: 유효한 임베딩은 많지만 활용도가 매우 낮음
- **Reg-VQ**: 완전한 코드북 활용과 모든 임베딩의 유효성 달성

***

### 3. 제안하는 방법

#### 3.1 사전 분포 정규화[1]

균일 이산 분포를 사전 분포로 가정하여, 모든 코드북 임베딩이 균등하게 사용되도록 유도합니다:

$$P_{\text{prior}} = \left[\frac{1}{N}, \frac{1}{N}, \cdots, \frac{1}{N}\right]$$

여기서 $N$은 코드북 크기입니다.

후방 분포는 양자화된 원-핫 벡터 $p_i$의 평균으로 근사합니다:

$$P_{\text{post}} = \frac{\sum_{i=1}^{H \times W} p_i}{H \times W} = [p_1, p_2, \cdots, p_N]$$

KL 발산을 통해 두 분포 간의 불일치를 측정하고 최소화합니다:

$$\mathcal{L}_{kl} = \text{KL}(P_{\text{post}}, P_{\text{prior}}) = -\sum_{n}^{N} p_n \log \frac{1/N}{p_n} = \sum_{n}^{N} p_n \log(N \cdot p_n)$$

이 정규화는 최대 엔트로피 원칙에 따라 모든 코드북 임베딩의 정보 용량을 최대화합니다.

#### 3.2 확률론적 마스크 정규화[1]

결정론적 양자화와 확률론적 양자화 간의 최적 균형을 달성하기 위해, 공간적으로 마스크를 적용합니다.

마스크 $M \in \mathbb{R}^{H \times W}$를 무작위로 설정합니다:
- **'1'**: Gumbel 샘플링을 위한 영역
- **'0'**: Argmax를 위한 영역

재구성 손실은 다음과 같이 공식화됩니다:

$$\mathcal{L}_{\text{rec}} = \|X - G(X_{\text{argmax}} \odot (1 - M) + X_{\text{gumbel}} \odot M)\|_1$$

여기서 $\odot$는 요소별 곱셈(element-wise multiplication)입니다. Figure 3에서 보여지는 실험을 통해 **40% 마스킹 비율**이 최적의 재구성 및 생성 품질을 달성함을 확인했습니다.[1]

#### 3.3 확률론적 대조 손실(PCL)[1]

완벽한 이미지 재구성의 압박감을 완화하기 위해 탄력적 재구성(elastic reconstruction)을 추구합니다. 원본 이미지의 패치와 재구성된 이미지의 패치를 비교하여, 같은 공간 위치의 특징은 양수 쌍으로, 다른 위치의 특징은 음수 쌍으로 취급합니다.

기본 대조 손실은 다음과 같습니다:

$$\mathcal{L}_{\text{cl}} = -\frac{1}{L}\sum_{i=1}^{L} \log \frac{e^{y_i \cdot z_i / \tau}}{e^{y_i \cdot z_i / \tau} + \sum_{\substack{j=1 \\ j \neq i}}^{L} e^{y_i \cdot z_j / \tau}}$$

여기서 $Y = [y_1, y_2, \cdots, y_L]$과 $Z = [z_1, z_2, \cdots, z_L]$은 각각 원본 이미지와 재구성된 이미지에서 추출한 특징 패치이고, $\tau$는 온도 파라미터입니다.

**확률론적 대조 손실의 핵심 개선**: 샘플된 임베딩과 최고 일치하는 임베딩 간의 유클리드 거리를 기반으로 가중치를 도입하여, 더 큰 섭동을 일으키는 영역에 대해 더 강한 당기기 힘을 적용합니다:

$$w_i = \|z_s - z_q\|_2^2$$

정규화된 가중치 $\{w'\_i\}\_{i=1}^L$ 을 사용하여 ($\sum_{i=1}^N w'_i = 1$), 최종 PCL은:

$$\mathcal{L}_{\text{pcl}} = -\sum_{i=1}^L \log \frac{w'_i \cdot e^{y_i \cdot z_i / \tau}}{w'_i \cdot e^{y_i \cdot z_i / \tau} + \frac{1}{L} \sum_{\substack{j=1 \\ j \neq i}}^{L} e^{y_i \cdot z_j / \tau}}$$

음수 항을 $1/L$로 균형 맞춰, 기본 대조 손실 대비 음수 항이 과도하게 커지는 것을 방지합니다.[1]

#### 3.4 알고리즘 구현

Algorithm 1은 결정론적과 확률론적 양자화 영역에서의 정확한 순전파(forward propagation)와 역전파(backward propagation) 절차를 제시합니다:[1]

**결정론적 양자화 영역:**
- 순전파: Argmax를 사용하여 가장 유사한 토큰 선택
- 역전파: 미분 가능성을 위해 Softmax로 대체

**확률론적 양자화 영역:**
- 순전파: Gumbel 샘플을 더한 후 Argmax 적용
- 역전파: Gumbel-Softmax로 대체

***

### 4. 모델 구조

#### 4.1 전체 프레임워크

논문의 정규화된 양자화 프레임워크는 Figure 2에서 보여지는 바와 같이 다음의 핵심 컴포넌트로 구성됩니다:[1]

**인코더-코드북-디코더 구조:**
- **인코더(E)**: 입력 이미지 $X$로부터 공간적 토큰 분포 $\xi_i \in \mathbb{R}^N$ (여기서 $i \in [1, H \times W]$)를 생성
- **코드북(Z)**: 학습 가능한 임베딩 $Z = \{z_n\}_{n=1}^N \in \mathbb{R}^{N \times d}$ (N: 코드북 크기, d: 임베딩 차원)
- **양자화 과정**: 예측된 토큰 분포에 따라 확률론적 마스크를 적용하여 각 영역별로 다른 양자화 전략 선택
- **디코더(G)**: 양자화된 벡터를 받아 입력 이미지 재구성

#### 4.2 생성 모델과의 통합

훈련된 벡터 양자화 프레임워크는 생성 모델과 연계하여 다양한 이미지 합성 작업을 수행합니다:

**자동회귀 모델(Autoregressive Models):**
- Transformer 기반 아키텍처가 토큰 시퀀스 간의 의존성을 모델링
- 의미론적 이미지 합성: ADE20K, CelebA-HQ 데이터셋에서 평가

**확산 모델(Diffusion Models):**
- 이산 토큰의 확산 프로세스를 통해 순차적 생성
- 텍스트-이미지 합성: CUB-200, MS-COCO 데이터셋에서 평가

***

### 5. 성능 향상 및 일반화 성능

#### 5.1 정량적 평가[1]

Table 1에 제시된 종합 벤치마크 결과에서 Reg-VQ의 우수성이 명확히 드러납니다:

**의미론적 이미지 합성 (ADE20K):**
- FID[R] (재구성): VQ-GAN(28.17) → Reg-VQ(23.69) **약 16% 개선**
- FID[G] (생성): VQ-GAN(38.53) → Reg-VQ(34.47) **약 10.5% 개선**
- PSNR[R]: VQ-GAN(18.89) → Reg-VQ(18.44)

**의미론적 이미지 합성 (CelebA-HQ):**
- FID[R]: VQ-GAN(12.74) → Reg-VQ(10.09) **약 20.8% 개선**
- FID[G]: VQ-GAN(20.89) → Reg-VQ(16.97) **약 18.8% 개선**

**텍스트-이미지 합성 (CUB-200):**
- FID[R]: Gumbel-VQ(10.84) → Reg-VQ(10.84) **동등하지만 생성 품질에서 우월**
- FID[G]: Gumbel-VQ(20.39) → Reg-VQ(16.93) **약 17% 개선**

**텍스트-이미지 합성 (MS-COCO):**
- FID[R]: Gumbel-VQ(16.93) → Reg-VQ(14.14) **약 16.5% 개선**
- FID[G]: Gumbel-VQ(20.06) → Reg-VQ(19.91) **안정적 성능**

#### 5.2 코드북 활용 및 확장성[1]

Figure 8의 실험 결과는 다양한 코드북 크기에서 Reg-VQ의 우수한 확장성을 보여줍니다:

**재구성 성능 (ADE20K):**
- 코드북 크기 N=1024: VQ-GAN(~26), Reg-VQ(~18)
- 코드북 크기 N=8192: VQ-GAN(~25), Reg-VQ(~16)

**생성 성능 (ADE20K):**
- 코드북 크기 증가에 따른 Reg-VQ의 일관된 성능 향상
- VQ-GAN은 더 큰 코드북에서 코드북 붕괴로 인한 성능 포화

이는 Reg-VQ가 **코드북 크기의 증가에 따라 일관된 성능 개선을 제공**함을 시사합니다.

#### 5.3 절제 연구(Ablation Study)[1]

Table 2의 상세한 절제 연구는 각 컴포넌트의 기여도를 명확히 합니다:

| 모델 | FID[R] | PSNR[R] | FID[G] |
|------|--------|---------|--------|
| Baseline (VQ-GAN) | 28.17 | 18.89 | 38.53 |
| +PriorReg | 25.92 | 18.98 | 36.57 |
| +MaskReg | 25.11 | 18.56 | 35.03 |
| +CL | 24.21 | 18.49 | 34.91 |
| +PCL | **23.69** | **18.44** | **34.47** |

**주요 발견:**
- **사전 분포 정규화**: 재구성 품질에서 **2.25 FID 포인트** 개선, 생성 품질에서 **1.96 포인트** 개선
- **마스크 정규화**: 추가 **0.81 FID 포인트** 개선 (재구성), **1.54 포인트** 개선 (생성)
- **확률론적 대조 손실**: 기본 대조 손실 대비 **0.52 포인트** 추가 개선 (생성)

#### 5.4 마스킹 비율의 영향[1]

Figure 3의 분석에 따르면:
- **낮은 마스킹 비율** (0-20%): 높은 PSNR[R]이지만 생성 품질(FID[G])이 낮음
- **최적 마스킹 비율** (40%): 재구성과 생성 품질 간의 최적 균형
- **높은 마스킹 비율** (60-100%): 생성 품질이 악화되고 재구성 정확도 감소

이는 **확률론적 양자화와 결정론적 양자화의 적절한 비율이 모델 성능에 결정적**임을 시사합니다.

#### 5.5 모델의 일반화 성능 향상 가능성[1]

**Reg-VQ의 일반화 특성:**

1. **다양한 데이터셋 간의 일관된 성능**: ADE20K(의미론), CelebA-HQ(얼굴), CUB-200(새), MS-COCO(일반) 등 다양한 도메인에서 일관된 개선
   
2. **생성 모델 독립성**: 자동회귀 모델과 확산 모델 모두에서 우수한 성능 달성
   - 자동회귀 모델: FID[G] 최대 18.8% 개선
   - 확산 모델: FID[G] 평균 17% 개선

3. **코드북 크기 확장성**: Figure 8에서 코드북 크기 증가에 따라 VQ-GAN은 성능이 포화되지만, Reg-VQ는 계속 개선됨
   - N=1024에서 N=8192로 확대 시, VQ-GAN은 1 FID 포인트만 개선, Reg-VQ는 2 포인트 개선

4. **데이터 효율성**: 더 적은 수의 활용 가능한 토큰으로도 높은 품질의 이미지 표현 가능

5. **확률론적 정규화의 효과**: KL 발산을 통한 선택적 정규화가 **학습 안정성 증대** 및 **과적합 방지**에 기여

***

### 6. 한계와 제약 조건

#### 6.1 논문에서 명시된 한계[1]

**동일한 학습 목표의 한계:**
논문의 부록(Section C)에서 저자들이 인정하듯이, 현재 양자화 모델은 인코더와 디코더를 동일한 학습 목표로 훈련합니다:

> 현재 양자화 모델은 인코더와 디코더에서 동일한 학습 목표를 사용하여 훈련되고 있다. 하지만 토큰화된 이미지 합성의 경우, 인코더와 디코더는 실제로 다른 목표를 가진다: 인코더는 정확한 이산 표현 학습을 목표로 하고, 디코더는 현실적인 이미지 생성을 목표로 한다. 따라서 동일한 목표로 훈련하는 것은 차선책이며, 양자화와 생성 성능을 제약할 것이다.[1]

#### 6.2 기술적 제약

1. **사전 분포의 선택**: 현재 균일 분포 가정이 모든 데이터 분포에 최적인지 불명확
   
2. **마스킹 비율의 고정성**: 40%의 고정 마스킹 비율이 다양한 작업에 대해 최적인지 의문
   
3. **계산 복잡도**: 추가적인 KL 발산 계산과 PCL 계산으로 인한 훈련 시간 증가

4. **대규모 이미지 처리**: 논문의 실험은 256×256 해상도에 제한되어 있으며, 고해상도 이미지에서의 성능 미검증

***

### 7. 논문이 앞으로의 연구에 미치는 영향

#### 7.1 벡터 양자화 분야의 진전[2][3][4][5]

**Reg-VQ의 출현 이후 진행된 관련 연구들:**

1. **코드북 붕괴 문제의 근본적 분석**: 2024년 발표된 연구에서 벡터 양자화의 표현 붕괴를 체계적으로 분석하고, 분리된 코드북 최적화가 근본 원인임을 밝혔습니다.[5]

2. **SimVQ: 선형 변환을 통한 간단한 해결책**: 2024년 제시된 방법으로, 코드북을 학습 가능한 선형 변환층을 통해 재매개변수화하여 전체 선형 공간을 최적화함으로써 붕괴 문제를 해결합니다.[4]

3. **Soft Convex Quantization (SCQ)**: 2023년 제시된 방법으로, VQ를 미분 가능한 볼록 최적화 문제로 재공식화하여 한 차수 높은 코드북 활용 개선(예: LSUN에서 0.002 vs 2.8)을 달성했습니다.[3]

#### 7.2 다양한 양자화 기법의 발전[6][7][8][9][10]

**2023년 이후 제시된 대안 접근 방식:**

1. **XQ-GAN (2024)**: 잔차 양자화(Residual Quantization, RQ), 다중 스케일 잔차 양자화(MSVQ), 상품 양자화(PQ), 룩업-프리 양자화(LFQ), 바이너리 구형 양자화(BSQ) 등 다양한 최신 양자화 기법을 통합한 프레임워크 제시[6]

2. **Lookup-Free Quantization (LFQ)**: 코드북을 명시적으로 유지하지 않는 새로운 패러다임으로, 수치 안정성과 효율성을 향상시킵니다.[10]

3. **MGVQ (2024)**: 다중 서브-코드북(multi-group sub-codebook)을 사용하여 코드북의 표현 용량을 10억 배 이상 확대하며, ImageNet 256p에서 SDXL-VAE를 능가합니다.[8]

4. **TiTok (2024)**: 1차원 잠재 시퀀스로 이미지를 토큰화하는 Transformer 기반 토크나이저로, 256×256 이미지를 32개 토큰으로 압축합니다.[9]

#### 7.3 토큰화와 생성 모델의 분리[11][12]

**벡터 양자화 필요성에 대한 재고찰:**

1. **연속 공간에서의 자동회귀 생성**: 2024년 발표된 연구에서 이산 공간이 필수가 아니며, 확산 손실을 통해 연속 공간에서의 자동회귀 모델링이 가능함을 입증합니다.[11]

2. **비디오 생성에서의 VQ 제거**: NOVA 모델(2024)은 벡터 양자화 없이 비디오 자동회귀 모델링을 수행하여, 기존 방식 대비 낮은 훈련 비용으로 우수한 성능을 달성합니다.[12]

#### 7.4 의미론적 토큰화의 진화[13][14][15][10]

**Reg-VQ 이후의 지향점:**

1. **언어 가이드 토큰화(Language-Guided Tokenization)**: 2024년 CVPR에 수락된 연구로, 텍스트 캡션 정보를 활용하여 더욱 의미론적으로 풍부한 토큰화를 실현합니다.[15]

2. **이미지 이해를 통한 토큰화**: 2024년 발표된 연구에서 이미지 이해 모델의 인코더를 토크나이저로 변환하여, 의미론적 정보를 더욱 효과적으로 보존합니다.[13]

3. **토큰 구성(Token Composition)**: 2024년 CVPR 논문으로, 토큰 레벨의 감독을 통해 텍스트-이미지 확산 모델의 일관성을 향상시킵니다.[16]

***

### 8. 앞으로의 연구 시 고려할 점

#### 8.1 이론적 기초 강화

1. **일반화 성능에 대한 정보이론적 분석**: 최근 연구에서 이산 잠재 공간을 가진 VQ-VAE의 일반화 오차 한계를 도출하고 있습니다. Reg-VQ의 맥락에서:[17]
   - 사전 분포 정규화가 일반화 경계를 어떻게 개선하는지 이론적 분석
   - KL 발산 정규화 가중치의 최적값 도출

2. **마스킹 비율의 동적 조정**: 현재 고정된 40% 비율 대신:
   - 학습 단계에 따른 적응적 마스킹 비율 조정
   - 데이터셋 특성에 따른 최적 비율 자동 결정 메커니즘

#### 8.2 아키텍처 개선

1. **분리된 학습 목표의 도입**: 저자들이 제시한 미래 방향 중 가장 중요한 과제
   - 인코더: 판별력 있는 특징 학습
   - 디코더: 현실적 이미지 생성
   - 상호 정보를 최대화하면서도 목표 간 충돌 최소화하는 방법론 개발

2. **계층적 코드북 구조**: MGVQ의 다중 그룹 접근법과 결합
   - Reg-VQ의 정규화 메커니즘을 다층 코드북에 적용
   - 표현 용량과 효율성의 균형

3. **비대칭 인코더-디코더**: 최근 방향성 양자화 기법 활용
   - 정확한 이산 표현을 위한 정밀 인코더
   - 고충실도 생성을 위한 강력한 디코더

#### 8.3 데이터 및 확장성 개선

1. **고해상도 이미지 처리**: 현재 256×256 제한 극복
   - 계층적 토큰화와 결합한 고해상도 적응
   - 다중 스케일 코드북 활용

2. **도메인 간 일반화**: 더 다양한 데이터셋에서의 검증
   - 의료 영상, 위성 영상, 3D 데이터 등 특수 도메인
   - 도메인 이동 상황에서의 일반화 성능 분석

3. **데이터 효율성 향상**: 적은 양의 데이터로의 학습
   - 자기감독 학습과의 결합
   - 전이 학습 가능성 탐구

#### 8.4 다모달 확장

1. **멀티모달 토큰화**: 텍스트, 음성, 비디오 등과의 통합
   - 단일 코드북을 통한 멀티모달 표현
   - 크로스모달 일관성 보장

2. **조건부 생성의 정밀화**: 더욱 세밀한 제어 메커니즘
   - 토큰 레벨의 조건 부여
   - 공간-시간적 제약 조건 통합

#### 8.5 계산 효율성

1. **훈련 시간 단축**: 추가 정규화 항의 오버헤드 감소
   - PCL 계산의 경량화
   - 점진적 정규화 전략

2. **추론 속도 개선**: 실시간 적용을 위한 최적화
   - 양자화 과정의 단순화
   - 모바일/엣지 장비에서의 배포

#### 8.6 문제 해결 로드맵

**단기 (1-2년):**
- 분리된 인코더-디코더 학습 목표 개발
- 동적 마스킹 비율 조정 메커니즘
- 고해상도(512×512) 이미지 처리 실현

**중기 (2-3년):**
- 다양한 사전 분포 탐구 (균일 분포 이외)
- 멀티모달 토큰화 통합
- 도메인 특화 코드북 학습

**장기 (3-5년):**
- 토큰화와 생성 모델의 완전 통합
- 자기감독 학습과의 심층 결합
- 차세대 생성 모델 아키텍처의 기초 제공

***

### 9. 2020년 이후 관련 최신 연구 탐색

#### 9.1 벡터 양자화 기본 기술의 진화

**초기 기초 작업:**
- VQ-VAE (2017)과 VQ-GAN (2021)의 발전을 통해 이산 표현 학습의 기반 형성[18][16]

**2022년 주요 성과:**
- **VQ-Diffusion**: 확산 모델에서의 이산 토큰 처리를 가능하게 하여, 텍스트-이미지 생성에서 SOTA 달성[19]
- **Residual Quantization (RQ-VAE)**: 잔차 양자화를 통해 256×256 이미지를 8×8 해상도로 압축하는 효율성 달성[20]

**2023년 방향 변화:**
- **VQ 객체 재고찰**: 의미 압축과 세부 보존 사이의 상충 관계를 분석하고, Semantic-Quantized GAN 제시[21]
- **Efficient-VQGAN**: Vision Transformer 기반 로컬 어텐션으로 효율성과 재구성 품질 동시 개선[22]

#### 9.2 코드북 문제의 근본적 이해[23][24][3][4][5]

**2023-2024 연구 발견:**

| 연구 | 핵심 기여 | 성과 |
|------|----------|------|
| SCQ[3] | 미분 가능한 볼록 최적화로 VQ 공식화 | LSUN에서 한 차수 높은 코드북 활용 |
| EdVAE[23] | 증거론적 이산 VAE로 붕괴 완화 | 코드북 활용도 현저히 개선 |
| SimVQ[4] | 선형 변환층을 통한 코드북 재매개변수화 | 간단하면서도 효과적인 붕괴 방지 |
| 표현 붕괴 분석[5] | VQ의 붕괴 특성 체계적 연구 | 분리된 코드북 최적화가 근본 원인 |
| IBQ[24] | 인덱스 역전파 양자화 | 코드북 불안정성 해결 |

#### 9.3 대안적 접근법의 등장

**양자화 없는 방법:**
- **MAR (2024)**: 확산 손실을 통한 연속 공간 자동회귀 모델링이 가능함을 증명[11]
- **NOVA (2024)**: 비디오 생성에서 벡터 양자화를 완전히 제거하고도 우수한 성능 달성[12]

**새로운 양자화 패러다임:**
- **LFQ (2023)**: 코드북을 명시적으로 유지하지 않는 룩업-프리 방식[10]
- **MGVQ (2024)**: 다중 그룹 서브-코드북을 통한 표현 용량 10억 배 확대[8]

#### 9.4 고해상도 및 멀티모달 확장

**고해상도 토큰화:**
- **TiTok (2024)**: 1D 토큰화로 256×256을 32개 토큰으로 압축[9]
- **TexTok (2024)**: 텍스트 캡션 조건 토큰화로 의미론적 보존 강화[14]

**멀티모달 코드북:**
- **LG-VQ (2024)**: 언어 가이드 코드북 학습으로 멀티모달 표현 최적화[15]
- **VQ-KD (2024)**: 이미지 이해 모델에서 추출한 의미론적 특징 활용[13]

#### 9.5 대규모 모델에서의 응용

**확산 모델 최적화:**
- **벡터 양자화를 통한 SDXL 압축**: 2.6B 파라미터 모델을 3비트로 압축하면서 품질 유지[25]

**언어 모델 연계:**
- **Tokenization 품질이 생성 모델 성능에 미치는 영향 분석**
- 차세대 멀티모달 기초 모델의 핵심 컴포넌트로 확인

***

### 결론

"Regularized Vector Quantization for Tokenized Image Synthesis" 논문은 이미지 토큰화의 기본적인 문제들—코드북 붕괴, 낮은 코드북 활용도, 추론 단계 불일치, 손상된 재구성 목표—을 정교한 정규화 메커니즘을 통해 해결하는 중요한 기여를 제시합니다.

**이 연구의 의의:**

1. **이론적 기여**: 결정론적과 확률론적 양자화의 상충 관계를 명확히 하고, 체계적인 해결책 제시

2. **실용적 성능**: 다양한 데이터셋과 생성 모델에서 일관된 성능 향상 (평균 15-20% FID 개선)

3. **확장성**: 코드북 크기 증가에 따른 안정적 성능 향상으로 대규모 모델 적용 가능성 입증

4. **학문적 영향**: 이후 벡터 양자화 연구에서 코드북 문제의 근본적 이해와 다양한 해결책 개발의 촉발

향후 연구는 분리된 인코더-디코더 학습 목표, 적응적 정규화 메커니즘, 고해상도 이미지 처리, 그리고 멀티모달 토큰화 등을 통해 벡터 양자화 기술을 더욱 정밀화하고 효율화할 것으로 예상됩니다. 동시에 양자화 없는 방법들의 등장은 차세대 생성 모델 아키텍처에서 이산 표현의 필요성 자체를 재평가하게 하고 있습니다.[2][25][3][4][5][17][8][9][12][15][11][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/577fe494-7ea3-4546-912e-5e86535d871c/2303.06424v2.pdf)
[2](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[3](https://arxiv.org/abs/2310.03004)
[4](https://arxiv.org/abs/2411.02038)
[5](https://arxiv.org/pdf/2411.16550.pdf)
[6](https://arxiv.org/pdf/2412.01762.pdf)
[7](https://arxiv.org/pdf/2412.10208.pdf)
[8](https://arxiv.org/html/2507.07997v1)
[9](https://proceedings.neurips.cc/paper_files/paper/2024/file/e91bf7dfba0477554994c6d64833e9d8-Paper-Conference.pdf)
[10](https://openaccess.thecvf.com/content/CVPR2025/papers/Zha_Language-Guided_Image_Tokenization_for_Generation_CVPR_2025_paper.pdf)
[11](https://arxiv.org/abs/2406.11838)
[12](https://arxiv.org/abs/2412.14169)
[13](https://arxiv.org/pdf/2411.04406.pdf)
[14](https://arxiv.org/html/2412.05796v1)
[15](http://arxiv.org/pdf/2405.14206.pdf)
[16](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_TokenCompose_Text-to-Image_Diffusion_with_Token-level_Supervision_CVPR_2024_paper.pdf)
[17](https://arxiv.org/html/2505.19470v1)
[18](https://iopscience.iop.org/article/10.1149/MA2024-02422792mtgabs)
[19](https://arxiv.org/abs/2111.14822)
[20](https://ieeexplore.ieee.org/document/9879532/)
[21](https://arxiv.org/pdf/2212.03185.pdf)
[22](https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_Efficient-VQGAN_Towards_High-Resolution_Image_Generation_with_Efficient_Vision_Transformers_ICCV_2023_paper.pdf)
[23](https://arxiv.org/pdf/2310.05718.pdf)
[24](https://arxiv.org/html/2412.02692)
[25](https://openreview.net/pdf?id=s2tadViqwH)
[26](https://ejournal.polraf.ac.id/index.php/JIRA/article/view/663)
[27](https://academic.oup.com/jes/article/doi/10.1210/jendso/bvae163.290/7812919)
[28](https://iopscience.iop.org/article/10.1149/MA2024-02171692mtgabs)
[29](https://invergejournals.com/index.php/ijss/article/view/79)
[30](https://academic.oup.com/jes/article/doi/10.1210/jendso/bvae163.2236/7812033)
[31](https://iopscience.iop.org/article/10.1149/MA2024-023342mtgabs)
[32](https://academic.oup.com/jes/article/doi/10.1210/jendso/bvae163.2152/7812875)
[33](https://visniknew.donnuet.edu.ua/index.php/visnik/article/view/85)
[34](http://arxiv.org/pdf/2401.01272.pdf)
[35](https://arxiv.org/pdf/2310.05400.pdf)
[36](https://arxiv.org/pdf/2111.14822.pdf)
[37](https://arxiv.org/html/2310.03661v3)
[38](https://www.semanticscholar.org/paper/Regularized-Vector-Quantization-for-Tokenized-Image-Zhang-Zhan/bda43eaf9a239a5e03f928b1537309f2e7637fda)
[39](https://papers.nips.cc/paper_files/paper/2023/file/5e8023f07625374c6fdf3aa08bb38e0e-Paper-Conference.pdf)
[40](https://liner.com/ko/review/vectorquantized-image-modeling-with-improved-vqgan)
[41](https://proceedings.neurips.cc/paper_files/paper/2024/file/66e226469f20625aaebddbe47f0ca997-Paper-Conference.pdf)
[42](https://aaltodoc.aalto.fi/items/a2159572-c665-4f80-8df3-0a7336faa5cb)
[43](https://www.techscience.com/cmc/v83n2/60526)
[44](https://ieeexplore.ieee.org/document/10205414/)
[45](https://ieeexplore.ieee.org/document/10204804/)
[46](https://ieeexplore.ieee.org/document/10204062/)
[47](http://ieeexplore.ieee.org/document/7045730/)
[48](https://arxiv.org/pdf/2403.10071.pdf)
[49](https://arxiv.org/pdf/1906.06698.pdf)
[50](https://arxiv.org/html/2411.02038)
[51](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Regularized_Vector_Quantization_for_Tokenized_Image_Synthesis_CVPR_2023_paper.pdf)
[52](https://www.emergentmind.com/topics/vector-quantization-variational-autoencoder-vq-vae)
[53](https://openreview.net/pdf?id=rkE3y85ee)
[54](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Codebook_Transfer_with_Part-of-Speech_for_Vector-Quantized_Image_Modeling_CVPR_2024_paper.pdf)
[55](https://proceedings.neurips.cc/paper/2020/file/90c34175923a36ab7a5de4b981c1972f-Paper.pdf)
[56](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/reg-vq/)
[57](https://smcho1201.tistory.com/122)
[58](https://deepai.org/publication/invertible-gaussian-reparameterization-revisiting-the-gumbel-softmax)
[59](https://arxiv.org/html/2411.16550v1)
