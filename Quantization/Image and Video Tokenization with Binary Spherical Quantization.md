
# Image and Video Tokenization with Binary Spherical Quantization (BSQ-ViT)

> **논문 정보**
> - **제목:** Image and Video Tokenization with Binary Spherical Quantization
> - **저자:** Yue Zhao (UT Austin), Yuanjun Xiong (MThreads AI), Philipp Krähenbühl (UT Austin)
> - **arXiv:** [2406.07548](https://arxiv.org/abs/2406.07548) (2024년 6월 11일)
> - **학회:** ICLR 2025 (GitHub: zhaoyue-zephyrus/bsq-vit)

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 **Binary Spherical Quantization(BSQ)** 을 적용한 새로운 트랜스포머 기반 이미지·영상 토크나이저를 제안합니다. BSQ는 고차원 시각 임베딩을 저차원 초구면(hypersphere)으로 투영한 뒤 이진 양자화(binary quantization)를 적용하며, (1) 명시적 코드북 없이 파라미터 효율적이고, (2) 임의의 토큰 차원으로 확장 가능하며, (3) 최소한의 왜곡으로 시각 데이터를 최대 $100\times$ 압축할 수 있습니다.

### 주요 기여 요약

| 기여 항목 | 내용 |
|---|---|
| **BSQ 양자화** | 하이퍼스피어 투영 + 이진 양자화, 코드북 파라미터 불필요 |
| **통합 아키텍처** | 이미지·가변 길이 영상을 하나의 ViT 기반 인코더-디코더로 처리 |
| **효율적 엔트로피 정규화** | Bernoulli 분포 분해로 $O(2^L \times L) \to O(L)$ 복잡도 감소 |
| **SOTA 성능** | 이미지·영상 재구성 벤치마크에서 최고 성능, $2.4\times$ 처리량 향상 |
| **생성 모델 연동** | 마스크드 언어 모델(MLM)과 결합 시 GAN·Diffusion에 필적하는 이미지 합성 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

기존 VQ-VAE 방식에는 두 가지 주요 문제가 있습니다. 첫째, 대부분의 이미지 인코더가 CNN 기반으로 구성되어 있어 이미지용 공간 합성곱(spatial convolution)을 영상용 시공간 합성곱(spatial-temporal convolution)으로 바꾸려면 상당한 구조적 변경과 연산 비용 증가가 필요합니다. 영상을 이미지 시퀀스로 처리하면 최적화되지 않은 양자화 결과가 나타납니다.

둘째, 벡터 양자화(VQ)는 코드북 크기에 따라 확장성이 낮습니다. 런타임이 코드북 크기에 선형 비례하고, 작은 데이터셋에서 코드북이 쉽게 과적합되며, 이는 정적 시각 패턴과 동적 모션 패턴 모두를 표현해야 하는 영상 입력에서 특히 심각한 문제가 됩니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) Binary Spherical Quantization (BSQ)

BSQ-ViT의 영상 토크나이저는 인코더-디코더 구조의 이산화 병목(discretization bottleneck)을 따르며, 트랜스포머 기반 인코더, 트랜스포머 기반 디코더, 그리고 BSQ 레이어로 구성됩니다. BSQ는 잠재 코드를 저차원 구면 공간으로 투영하고, 이진 양자화를 적용한 뒤 디코더의 잠재 공간으로 다시 투영합니다. 이 저차원 구면 공간으로의 투영은 양자화 오차가 유계(bounded)되고 엔트로피 계산의 상당 부분이 개별 차원으로 인수분해(factorize)된다는 이론적 장점을 가집니다.

**BSQ의 핵심 과정:**

**① $\ell_2$ 정규화 (구면 투영)**

인코더 출력 $z \in \mathbb{R}^d$를 $L$차원 하이퍼스피어로 투영합니다:

$$\hat{z} = W z, \quad \hat{z} \in \mathbb{R}^L$$

$$\tilde{z} = \frac{\hat{z}}{\|\hat{z}\|_2} \in \mathcal{S}^{L-1}$$

**② 이진 양자화**

$$b_i = \text{sign}(\tilde{z}_i) \in \{-1, +1\}, \quad i = 1, \ldots, L$$

즉, 양자화된 코드 $\mathbf{b} = (b_1, b_2, \ldots, b_L) \in \{-1, +1\}^L$

**③ 암묵적 코드북 (Implicit Codebook)**

BSQ는 학습된 파라미터 없이 유효 어휘(vocabulary)가 구면 차원에 따라 지수적으로 증가하는 암묵적 코드북을 구성합니다. 코드북 크기가 커질수록 재구성 결과가 일관적으로 향상됩니다.

유효 코드북 크기: $|\mathcal{C}| = 2^L$

**④ Soft Quantization 및 엔트로피 정규화**

LFQ와 달리 BSQ는 유계된 양자화 오차를 가지고, soft 양자화 확률이 다수의 채널 독립적인 베르누이 분포의 단순 곱으로 환원되어 효율적인 엔트로피 정규화를 가능하게 합니다.

각 차원 $i$의 soft quantization 확률:

$$p_i = \sigma(\alpha \cdot \tilde{z}_i), \quad \alpha \text{는 온도 파라미터}$$

분해된 엔트로피 정규화:

$$\mathcal{H}(\mathbf{b}) \approx \sum_{i=1}^{L} \mathcal{H}(b_i) = -\sum_{i=1}^{L} \left[ p_i \log p_i + (1-p_i)\log(1-p_i) \right]$$

이 분해된 근사를 통해 $L$비트 soft 양자화의 엔트로피 계산 복잡도가 이론적으로 $O(2^L \times L)$에서 $O(L)$로 감소하며, 근사 오차는 최소하고 실제 성능 저하도 무시할 수 있는 수준입니다.

**⑤ 전체 학습 손실 함수**

BSQ-ViT의 전체 손실 함수는 commitment loss $\mathcal{L}\_\text{commit}$, 엔트로피 정규화 $\mathcal{L}\_\text{entropy}$, 지각적 손실 $\mathcal{L}_\text{LPIPS}$, 그리고 adversarial loss로 구성됩니다.

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{recon} + \lambda_\text{commit}\mathcal{L}_\text{commit} + \lambda_\text{entropy}\mathcal{L}_\text{entropy} + \lambda_\text{LPIPS}\mathcal{L}_\text{LPIPS} + \lambda_\text{adv}\mathcal{L}_\text{adv}$$

- $\mathcal{L}_\text{commit} = \|z - \text{sg}[z_q]\|_2^2$ (straight-through estimator 사용)
- $\mathcal{L}_\text{entropy}$: 코드 균등 사용 장려
- VQGAN을 따라 perceptual loss와 adversarial loss를 사용하며, 판별자로는 ViT-VQGAN에서 학습이 훨씬 안정적임을 보고한 StyleGAN을 사용합니다.

---

### 2-3. 모델 구조

이 논문은 Vision Transformer와 BSQ를 결합한 통합 시각 토크나이저를 제안합니다. 트랜스포머 기반 인코더-디코더는 블록 단위의 인과적 마스크(block-wise causal mask)를 활용하며, 재구성 시 현재 또는 이전 타임스탬프의 시각 토큰만을 사용합니다.

BSQ는 트랜스포머 인코더의 고차원 시각 임베딩을 저차원 하이퍼스피어로 먼저 투영한 뒤 이진 양자화를 수행합니다. 트랜스포머 인코더, 디코더, BSQ는 VQ-GAN 프레임워크에 원활하게 통합되어 종단간(end-to-end) 학습됩니다.

**구조 요약 다이어그램:**

```
입력 (이미지/가변 길이 영상)
        ↓
  ViT Encoder (Block-wise Causal Mask)
        ↓
  BSQ Layer:
    ① Linear Projection (d → L)
    ② ℓ₂ Normalization → 하이퍼스피어
    ③ Binary Quantization: {-1, +1}^L
    ④ Linear Projection back (L → d)
        ↓
  ViT Decoder (Block-wise Causal Mask)
        ↓
   재구성 출력
```

블록 단위 인과적 마스킹을 통해 동일한 아키텍처로 이미지와 가변 길이 영상 모두를 처리할 수 있는 통합 구조를 실현합니다.

---

### 2-4. 성능 향상

이미지 재구성에서 모델은 픽셀 수준 및 의미론적 지표 모두에서 최고 수준의 재구성 품질을 달성합니다. 특히, 최고 성능의 BSQ-ViT는 ImageNet-1k val에서 rFID **0.41**을 달성하여 2위(SDXL-VAE)보다 **43% 감소**하면서도 **2.4배 더 빠릅니다**.

영상 재구성에서는 UCF-101에서 FVD를 기존 대비 절반 이상 감소시킵니다(8.62 → 4.10).

또한 적응형 산술 코딩을 위한 자기회귀 사전(autoregressive prior)을 학습함으로써, JPEG2000/WebP(이미지)나 H.264/H.265(영상)와 같이 널리 사용되는 압축 표준에 필적하는 시각 압축 결과를 달성합니다.

이에 더해 BSQ-ViT는 마스크드 언어 모델(MLM)이 GAN 및 Diffusion 기반 방법에 필적하는 경쟁력 있는 이미지 합성 품질을 달성할 수 있도록 합니다.

**주요 성능 비교표 (논문 기준):**

| 모델 | rFID (ImageNet-1k) | FVD (UCF-101) | 처리량 |
|---|---|---|---|
| SDXL-VAE | ~0.72 | - | 1× |
| **BSQ-ViT (Ours)** | **0.41** | **4.10** | **2.4×** |
| LFQ (MAGVIT-v2) | 비교 대상 | 8.62 | - |

---

### 2-5. 한계점

논문 자체에서는 "이 역할에 대한 더 깊은 분석은 본 논문의 범위를 벗어난다"고 인정한 부분이 있습니다. 논문 내에서 확인 가능한 한계 및 향후 과제는 다음과 같습니다:

- **고해상도 영상의 확장성:** 현재 벤치마크는 주로 $128 \times 128$ 해상도 중심이며, 매우 고해상도 영상에서의 성능은 추가 검증 필요
- **블록 단위 인과 마스크의 트레이드오프:** 비인과적(non-causal) 변형이 모든 지표에서 약간 더 나은 성능을 보이는 것으로 나타나, 인과적 마스크 적용 시 일부 성능 손실이 존재합니다.
- **복잡한 모션 표현:** C-ViViT 등의 선행 연구에서 지적된 바와 같이, 인수분해된 인과적 ViT 방식은 효율성을 높이지만 시간에 걸친 복잡한 모션 모델링을 희생할 가능성이 있습니다.
- **이진 코드의 표현력 상한:** $L$비트 이진 코드는 최대 $2^L$개의 고유 토큰만 표현하므로, 극도로 다양한 시각 패턴에는 $L$ 값이 충분히 커야 함

---

## 3. 일반화 성능 향상 가능성

### 3-1. 통합 아키텍처를 통한 일반화

블록 단위 인과적 마스킹 덕분에 동일한 아키텍처로 이미지와 가변 길이 영상 모두를 처리할 수 있어, 별도의 아키텍처 수정 없이 다양한 입력 유형에 대해 일반화 능력을 제공합니다.

### 3-2. 어휘 크기 확장을 통한 일반화

이미지용으로 사전학습된 토크나이저를 영상에 그대로 쓸 수 있다는 주장도 있지만, 파인튜닝 후 영상 토크나이저가 훨씬 높은 재구성 품질을 보여주며, 이 차이는 유효 어휘 크기가 커질수록 더욱 두드러집니다. BSQ가 가능하게 한 증가된 어휘 크기가 영상 고유의 모션 및 블러 학습에 유리하다고 가설을 세웁니다.

### 3-3. 유계된 양자화 오차로 인한 일반화

BSQ의 양자화 오차가 유계(bounded)되어 있어 더 빠르고 안정적인 학습이 가능하며, 분해된 베르누이 분포를 이용한 엔트로피 정규화가 효율적으로 이루어집니다. 이는 본 적 없는 데이터 분포에서도 안정적인 코드 활용률을 유지할 수 있음을 의미합니다.

### 3-4. 파라미터 효율적 설계로 인한 과적합 방지

BSQ의 코드북은 암묵적이고 파라미터가 없으므로, 기존 VQ처럼 소규모 데이터셋에서 코드북이 과적합되는 문제를 원천적으로 방지합니다. 하이퍼스피어 매핑이 양자화 오차를 유계하여 일반화를 돕습니다.

### 3-5. 생성 및 압축 다운스트림 일반화

본 토크나이저는 경량화된 대안을 제시합니다: 토크나이제이션이 초기 지역적 손실 압축을 수행하고, 경량의 시퀀스 모델(~300M)이 전역적 영상 구조를 압축함으로써 다양한 다운스트림 작업에 적용 가능한 범용 표현을 학습합니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 관련 연구 계보

```
VQ-VAE (2017, Van Den Oord et al.)
    ↓
VQGAN (2021, Esser et al.) — GAN 정규화 도입
    ↓
ViT-VQGAN (2022, Yu et al.) — CNN → Transformer 전환
    ↓
MAGVIT (2023, Yu et al.) — 영상 특화 VQ
    ↓
MAGVIT-v2 / LFQ (2024, Yu et al.) — Lookup-Free Quantization
    ↓
BSQ-ViT (2024, Zhao et al.) — Binary Spherical Quantization ★
    ↓
VAEVQ, WeTok, MGVQ ... (2025)
```

ViT-VQGAN은 이미지 토크나이제이션을 위해 CNN 대신 트랜스포머 블록을 도입하였고, C-ViViT는 이 아이디어를 영상 토크나이제이션으로 확장하였습니다.

MAGVIT-v2는 영상(및 이미지)을 컴팩트한 이산 토큰으로 변환하는 영상 토크나이저로, 새로운 lookup-free 양자화 방법으로 대규모 어휘 학습이 가능하게 하고, 이미지와 영상 토크나이제이션에 공유 어휘를 사용할 수 있는 수정 방법을 제안하였습니다.

LFQ는 BSQ와 같은 이진화 기법을 사용하지만 출력을 정규화하지 않습니다. 이는 유계되지 않은 양자화 오차로 이어지며, 엔트로피 계산을 위한 soft 양자화를 단순하게 처리하기 어렵습니다.

**주요 방법 비교표:**

| 방법 | 연도 | 아키텍처 | 양자화 방식 | 코드북 | 이미지/영상 통합 | 코드북 파라미터 |
|---|---|---|---|---|---|---|
| VQ-VAE | 2017 | CNN | VQ (nearest neighbor) | 명시적 | ✗ | 有 |
| VQGAN | 2021 | CNN | VQ + GAN | 명시적 | ✗ | 有 |
| ViT-VQGAN | 2022 | ViT | VQ | 명시적 | ✗ | 有 |
| MAGVIT-v2/LFQ | 2024 | CNN+ViT | Lookup-Free (SQ) | 암묵적 | △ | 無 |
| **BSQ-ViT** | **2024** | **ViT** | **Binary Spherical** | **암묵적** | **✓** | **無** |

VQ-VAE와 VQGAN 및 그 변형들은 코드북을 통해 인덱싱 가능한 토큰을 생성하여 자기회귀 및 확산 모델을 가능하게 하지만, 코드북 붕괴(codebook collapse)와 의미적 손실 문제가 종종 발생합니다.

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

**① 이진 토큰 기반 멀티모달 LLM 통합 가능성**

BSQ가 생성하는 이진 코드 $\{-1, +1\}^L$은 자연어 토큰과 유사한 이산 심볼이므로, 텍스트-이미지-영상 통합 처리 언어 모델과의 결합 연구가 더욱 활발해질 것으로 예상됩니다.

**② 신경 압축(Neural Compression) 연구 촉진**

자기회귀 사전(autoregressive prior)을 학습하여 적응형 산술 코딩에 적용함으로써 최신 영상 압축 표준과 비교할 수 있는 결과를 달성하였는데, 이는 신경망 기반 영상 압축 코덱 개발에 직접적인 영향을 줄 것입니다.

**③ 암묵적 코드북 패러다임의 확산**

BSQ는 학습된 파라미터 없이도 코드북의 유효 어휘가 구면 차원에 따라 지수적으로 증가하는 암묵적 코드북을 구성하며, 코드북 크기 증가가 재구성 성능 향상으로 일관적으로 이어집니다. 이 패러다임은 다양한 모달리티(오디오, 3D 포인트 클라우드 등) 토크나이제이션 연구로 확장될 가능성이 큽니다.

**④ 생성 모델 평가 기준 재설정**

BSQ-ViT는 마스크드 언어 모델이 GAN·Diffusion 기반 방법과 경쟁할 수 있는 이미지 합성 품질을 가능하게 합니다. 이는 Diffusion 모델 중심의 생성 연구 패러다임에 재검토를 유발할 수 있습니다.

---

### 5-2. 향후 연구 시 고려할 점

**① 비트 수 $L$ 선택의 최적화**

코드북 크기가 $2^L$이므로, $L$이 너무 작으면 표현력이 부족하고, 너무 크면 학습 불안정 및 메모리 증가 문제가 발생할 수 있습니다. 태스크와 데이터 복잡도에 따른 $L$ 자동 선택 방법 연구가 필요합니다.

**② 초고해상도·초장 영상으로의 확장**

현재 실험은 제한된 해상도 및 길이의 영상 중심입니다. $4K$ 이상 초고해상도 또는 수십 분 이상 초장 영상에 대한 확장 연구가 필요합니다.

**③ 이진 코드의 의미론적 정렬 (Semantic Alignment)**

현재 BSQ는 재구성 품질 중심으로 설계되었으나, 생성된 이진 코드가 의미론적으로 일관된 클러스터를 형성하는지, 이를 다운스트림 이해 태스크(분류, 검색 등)에 활용할 수 있는지 추가 분석이 요구됩니다.

**④ 도메인 일반화 실험 부재**

현재 성능 검증은 ImageNet, UCF-101 등 표준 벤치마크에 집중되어 있으므로, 의료 영상·위성 영상·스포츠 중계 등 특수 도메인에서의 일반화 성능 검증 연구가 향후 필요합니다.

**⑤ 압축 효율과 생성 품질 간의 트레이드오프 심화 분석**

비인과적 모델이 모든 지표에서 약간 우수한 성능을 보인다는 점은, 가변 길이 영상 지원을 위한 인과성 제약이 성능에 일정 영향을 미침을 시사하므로, 이 트레이드오프를 최소화하는 설계 연구가 필요합니다.

**⑥ 멀티모달 확장**

BSQ는 차원 축소, 이진 양자화, 엔트로피 기반 최적화를 결합한 방법론으로서, 오디오·텍스트·3D 포인트 클라우드 등 다양한 모달리티에의 적용 가능성을 탐색하는 연구가 기대됩니다.

---

## 📚 참고 자료 및 출처

| # | 자료명 | 링크/출처 |
|---|---|---|
| 1 | **[주논문]** Zhao et al., "Image and Video Tokenization with Binary Spherical Quantization," ICLR 2025 | https://arxiv.org/abs/2406.07548 |
| 2 | **[공식 코드]** GitHub: zhaoyue-zephyrus/bsq-vit | https://github.com/zhaoyue-zephyrus/bsq-vit |
| 3 | **[OpenReview]** ICLR 2025 논문 리뷰 페이지 | https://openreview.net/forum?id=yGnsH3gQ6U |
| 4 | **[논문 PDF]** arXiv PDF | https://arxiv.org/pdf/2406.07548 |
| 5 | **[논문 HTML]** arXiv HTML 전문 | https://arxiv.org/html/2406.07548v1 |
| 6 | **[NSF 공식]** NSF PAR 제출본 | https://par.nsf.gov/servlets/purl/10631957 |
| 7 | **[리뷰]** The Moonlight Literature Review | https://www.themoonlight.io/en/review/image-and-video-tokenization-with-binary-spherical-quantization |
| 8 | **[설명 블로그]** BSQ Paper Explained (Gyanendra Das) | https://gyanendradas.substack.com/p/bsq-paper-explained |
| 9 | **[HuggingFace]** Paper Page | https://huggingface.co/papers/2406.07548 |
| 10 | **[비교 논문]** Yu et al., "Language Model Beats Diffusion — Tokenizer is Key to Visual Generation" (MAGVIT-v2/LFQ), ICLR 2024 | https://arxiv.org/html/2310.05737v2 |
| 11 | **[비교 논문]** ResearchGate PDF | https://www.researchgate.net/publication/381319136 |
| 12 | **[후속 연구]** VAEVQ: Enhancing Discrete Visual Tokenization | https://arxiv.org/html/2511.06863v1 |

> ⚠️ **정확도 관련 주의:** 본 답변의 수식 중 일부(특히 세부 파라미터 값)는 공개된 논문 원문 및 공식 코드를 최대한 참조하였으나, 논문 전체 접근이 제한적인 부분의 수식 세부 사항은 확인된 범위 내에서만 기술하였습니다. 최종 확인을 위해서는 반드시 원 논문(arXiv:2406.07548) 전문을 참조하시기 바랍니다.
