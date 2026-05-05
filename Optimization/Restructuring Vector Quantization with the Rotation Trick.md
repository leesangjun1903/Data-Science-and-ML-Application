# Restructuring Vector Quantization with the Rotation Trick

이 논문은 VQ-VAE의 비미분 가능 벡터 양자화 층을 ‘회전(rotation)+스케일링’ 선형 변환으로 재구성해, 양자화 연산의 기하학적 정보를 보존한 채로 encoder 쪽으로 그라디언트를 전달하는 새로운 추정자(“rotation trick”)를 제안하고, 이로써 코드북 사용률·양자화 오차·재구성 품질을 전반적으로 크게 개선했다고 주장합니다. 특히, 기존의 Straight-Through Estimator(STE)나 “정확한” 그라디언트(=AutoEncoder식 학습)보다 더 나은 일반화(재구성 FID·IS, 동영상 FVD 등)와 안정적인 학습(코드북 붕괴 방지)을 달성합니다.[^1][^2][^3][^4][^5]

***

## 1. 핵심 주장과 주요 기여

- **핵심 주장 1 – 새 그라디언트 추정자:**
VQ-VAE의 양자화 단계에서 encoder 출력 $e$를 최근접 코드북 벡터 $q$로 바로 치환하는 대신, $e$를 $q$ 방향으로 회전·스케일링한 $\tilde{q}$를 decoder 입력으로 사용하고, 역전파에서는 $\partial \tilde{q} / \partial e$를 회전·스케일링 행렬로 취급하여 그라디언트를 전달하는 “rotation trick”을 제안합니다.[^2][^3][^6][^1]
- **핵심 주장 2 – 각도(angle)를 보존하는 그라디언트 이동:**
STE는 $\nabla_q L$의 방향과 크기를 그대로 복사해 모든 $e$에 동일하게 적용하지만, rotation trick은 “ $q$와 $\nabla_q L$ 사이의 각도”가 $e$로 이동한 후에도 유지되도록 그라디언트를 회전시키므로, 같은 코드북 셀 안의 점들도 위치에 따라 서로 다른 업데이트를 받게 됩니다.[^5][^1]
- **핵심 주장 3 – 코드북 사용률 및 양자화 오차 개선:**
이 기하학적 특성 덕분에 셀 경계 근처의 점은 다른 코드북 region으로 밀려나(codebook utilization↑), 중심부의 점은 코드북 벡터 쪽으로 당겨져(quantization error↓), 손실 압축 측면에서 바람직한 “정보 용량↑, distortion↓”를 동시에 달성한다고 분석·실험으로 보입니다.[^7][^1][^2]
- **핵심 주장 4 – 광범위한 설정에서의 실증:**
ImageNet VQ-VAE, VQGAN(autoregressive·latent diffusion), ViT-VQGAN, TimeSformer 기반 video VQ-VAE 등 11개 설정에서 rotation trick을 적용하면, 동일한 아키텍처·하이퍼파라미터에서 재구성 FID/IS, 코드북 사용률, 양자화 오차가 일관되게 개선됩니다.[^6][^1][^2]
- **핵심 주장 5 – 이론적 해석(일반화 관점):**
저자들은 STE와 rotation trick을 각각 “서로 다른 좌표계에서의 parallel transport”로 해석하면서, rotation trick이 encoder를 AutoEncoder처럼 과적합시키지 않으면서도 양자화 구조를 충분히 반영하는 유리한 inductive bias를 제공한다고 주장합니다.[^1]

***

## 2. 이 논문이 해결하려는 문제

### 2.1 기본 VQ-VAE와 STE의 한계

VQ-VAE에서는 encoder 출력 $e$를 코드북 $\mathcal{C} = \{q_j\}_{j=1}^{|\mathcal{C}|}$의 최근접 벡터로 양자화합니다.[^1][^2]

$$
Q(q = i \mid e) =
\begin{cases}
1 & \text{if } i = \arg\min_{1 \le j \le |\mathcal{C}|} \| e - q_j \|_2^2 \\
0 & \text{otherwise}
\end{cases}
$$

decoder는 $q$를 입력으로 재구성 $\tilde{x}$를 만들고, VQ-VAE의 기본 손실은 (코드북·커밋먼트 항 포함시) 다음과 같습니다.[^1]

$$
L(\tilde{x}) = \|x - \tilde{x}\|_2^2 + \|\mathrm{sg}(e) - q\|_2^2 + \beta \| e - \mathrm{sg}(q)\|_2^2
$$

여기서 $\mathrm{sg}(\cdot)$는 stop-gradient, $\beta$는 보통 $[0.25, 2]$ 사이의 상수입니다.[^1]

문제는 양자화 $Q(\cdot)$가 비미분 가능이라 $\partial q/\partial e$를 정의할 수 없다는 점이며, 전통적으로 **Straight-Through Estimator(STE)** 가 이를 우회합니다.[^8][^1]

$$
\frac{\partial q}{\partial e} \approx I
$$

즉, 역전파에서 양자화를 무시하고, decoder 입력 $q$에서 나온 그라디언트를 encoder 출력 $e$에 그대로 복사합니다. 이때 같은 코드북 region(하나의 Voronoi 셀)에 있는 모든 $e$는 위치와 상관없이 동일한 그라디언트 업데이트를 받으므로,[^1]

- 코드북 region 내부의 구조(“ $e$가 $q$에 얼마나 가깝고 어느 방향에 있는지”)가 gradient에 전혀 반영되지 않고,[^1]
- 그 결과 codebook collapse, under-utilization, 높은 quantization error 등의 문제가 쉽게 발생합니다.[^8][^1]


### 2.2 “정확한” 그라디언트 대안의 문제

저자들은 “STE를 대체하기 위해 $e$에서의 정확한 그라디언트를 근사/계산해 보자”는 자연스러운 대안 두 가지를 분석합니다.[^1]

1. **Hessian-based 2차 근사:**

$$
L_e \approx L_q + (\nabla_q L)^{\top}(e-q) + \frac12 (e-q)^{\top} (\nabla_q^2 L) (e-q)
$$

로 전개해 $\partial L / \partial e$를 근사.[^1]
2. **Double forward pass + 작은 $\lambda$:**
양자화가 없는 경로($q = e$)로 또 한 번 forward하여 $L_e$를 계산하고,

$$
L = L_q + \lambda L_e
$$

를 사용하되 decoder 파라미터에는 $\lambda$를 곱해 영향 축소, encoder에는 $\lambda^{-1}$ 스케일을 곱해 사실상 “정확한 AutoEncoder gradient”를 encoder에 적용.[^1]

그러나 이들은 이 접근이 encoder를 **고전적 AutoEncoder처럼** 학습시키게 되어 과적합 및 일반화 성능 저하를 유발하며, 실제 실험에서도 FID/IS가 크게 나빠진다고 보고합니다. 이 점이 “정확한 그라디언트가 항상 좋은 것은 아니다”라는 중요한 교훈이자, 본 논문의 일반화 논의의 출발점입니다.[^1]

***

## 3. 제안 방법: Rotation Trick (수식 포함)

### 3.1 기본 아이디어와 수식

encoder 출력 $e$, 그에 대응하는 최근접 코드북 벡터 $q = Q(e)$가 주어졌다고 합시다. rotation trick의 forward는 다음과 같이 정의됩니다.[^1]

1. $e$를 $q$ 방향으로 **회전**시키는 정규 직교 행렬 $R$을 구성합니다. (예: Householder reflection 두 번으로 구현)[^1]
2. $\lambda = \|q\| / \|e\|$ 로 **스케일링** 계수를 정의합니다.[^1]
3. decoder 입력으로 사용할 벡터를

$$
\tilde{q} = \lambda R e
$$

로 둡니다.[^1]

중요한 점은, forward에서 $Q(e)$로 얻은 $q$와 $R, \lambda$ 모두 $e$의 함수이지만, **역전파에서는 이들을 상수로 취급(detach)** 한다는 것입니다. 따라서 Jacobian은[^1]

$$
\frac{\partial \tilde{q}}{\partial e} = \lambda R
$$

로 간단하며, 비미분 가능 양자화 $Q(\cdot)$를 통과할 필요 없이 encoder로 그라디언트를 보낼 수 있습니다.[^1]

### 3.2 각도 보존(Angle preservation) 특성

STE와의 핵심 차이는 “어떤 기하학적 양을 보존하는가” 입니다.[^1]

- **STE:**
$\nabla_q L$의 방향·크기 자체를 보존하면서 $q \to e$로 단순 평행이동합니다.

$$
\nabla_e L = \nabla_q L
$$

따라서 $e$가 $q$에서 얼마나 떨어져 있는지와 무관하게, 같은 region에 있는 모든 $e$가 동일한 업데이트를 받습니다.[^1]

- **Rotation Trick:**
$\nabla_q L$을 $R$에 의해 회전시켜 $e$로 옮기되,
“ $q$와 $\nabla_q L$ 사이의 각도”가
“ $e$와 $\nabla_e L$ 사이의 각도”로 그대로 유지되게 합니다.[^1]

즉,

$$
\angle(q, \nabla_q L) = \angle(e, \nabla_e L)
$$

이 성립하도록 $\nabla_e L = \lambda R \nabla_q L$이 정의됩니다.[^1]

이 특성 때문에, 같은 Voronoi cell 내의 점이라도 $e$가 $q$와 이루는 각도 $\theta$에 따라 이동 방향과 크기가 차별화됩니다.[^1]

### 3.3 Voronoi 셀에서의 “push–pull” 행동

저자들은 $e$ – $q$ 사이 각도 $\theta$, $q$ – $\nabla_q L$ 사이 각도 $\phi$를 정의하고, rotation trick이 각 region 내부 점들을 어떻게 움직이는지 분석합니다.[^1]

- **$-\pi/2 < \phi < \pi/2$ (gradient가 $q$와 같은 방향)**:
$\theta$가 큰(코드북 벡터와 멀리 떨어진) 점들은 STE보다 더 멀리 밖으로 밀려 나가며, 종종 인접 또는 이전에 거의 사용되지 않던 코드북 region으로 넘어갑니다.[^1]
→ 코드북 사용률 증가, 정보 용량 증가.[^1]
- **$\pi/2 < \phi < 3\pi/2$ (gradient가 $q$와 반대 방향)**:
같은 region의 점들 간 거리가 줄어들고, 특히 큰 $\theta$를 가진 점들이 $q$ 주변으로 당겨져 cluster를 형성합니다.[^1]
→ 양자화 오차 감소, encoder가 특정 코드북 벡터에 “lock-on”하기 쉬워짐.[^1]

이 “경계 점은 밀어내고, 중심부는 끌어당기는(push–pull)” 효과가, **코드북 사용률 증가 + 양자화 오차 감소**라는 VQ의 두 가지 상충 목표를 동시에 만족시킬 수 있는 이유라고 저자들은 설명합니다.[^1]

***

## 4. 모델 구조와 학습 설정

### 4.1 구조: 어디에 무엇을 바꾸는가

구조적 변화는 매우 제한적입니다.[^1]

- encoder/decoder 아키텍처는 기존 VQ-VAE/VQGAN/ViT-VQGAN/TimeSformer-VQGAN과 동일하게 유지합니다.[^1]
- 코드북 lookup(유클리드 거리 또는 cosine similarity), EMA 기반 코드북 업데이트 등도 기존 구현을 그대로 사용합니다.[^1]
- **유일한 차이점**은 “역전파 시 $\partial q / \partial e$를 무엇으로 두느냐”인데,
    - 기존: $\partial q / \partial e = I$ (STE)
    - 제안: $\partial \tilde{q} / \partial e = \lambda R$ (rotation trick) 입니다.[^6][^1]

즉, **forward 출력은 동일**(여전히 $q$를 기반으로 재구성)이고, **backward만 바꾸는 방식**이라 drop-in replacement에 가깝습니다.[^6][^1]

### 4.2 평가 설정 개요

논문은 총 11개 설정에서 rotation trick vs STE(및 기타 baselines)를 비교합니다.[^1]

- **이미지 VQ-VAE** (ImageNet, 256×256)
    - latent 32×32×32, codebook 1024
    - latent 64×64×3, codebook 8192
    - 거리: 유클리드/코사인 둘 다 실험.[^1]
- **VQGAN (Esser et al. 2021)** – autoregressive-friendly 설정 및 latent diffusion-friendly 설정 모두.[^6][^1]
- **ViT-VQGAN (Yu et al. 2021)** – ViT encoder/decoder, factorized codes, L2 normalization.[^1]
- **TimeSformer-VQGAN** – BAIR robot pushing, UCF101 video reconstruction.[^1]

모든 경우, 아키텍처·하이퍼파라미터는 공개 레포/기존 논문을 그대로 따르고, 단지 gradient estimator만 바꾸었다고 명시되어 있습니다.[^6][^1]

***

## 5. 성능 향상 및 한계

### 5.1 정량적 성능 개선

대표적인 결과만 요약하면 다음과 같습니다.

- **기본 VQ-VAE (ImageNet, 64×64×3, codebook 8192):**
    - rotation trick 적용 전: 코드북 사용률 100%이나 양자화 오차가 상대적으로 크고, reconstruction FID/IS가 열등.[^1]
    - rotation trick 적용 후: 양자화 오차가 10배 수준으로 줄어들고, reconstruction FID 감소, IS 증가 등 재구성 품질이 눈에 띄게 향상됩니다.[^1]
- **latent diffusion용 VQGAN (Rombach et al. 2022 구현, ImageNet):**
    - 한 설정에서는 reconstruction FID가 5.0 → 1.1, reconstruction IS가 141.5 → 200.2로 크게 개선되고, 코드북 사용률은 2% → 27%로 13.5배 증가, 양자화 오차는 두 자릿수 크기 감소(“두 order of magnitude”)를 보입니다.[^6][^1]
    - 다른 설정(64×64×3, codebook 8192)에서는 validation FID ~0.53 → ~0.27, reconstruction IS ~20.6 → ~28.0, codebook usage ~15% → ~86%로 개선됩니다.[^1]
- **ViT-VQGAN (ImageNet, 8×8×32, codebook 8192):**
    - rotation trick으로 codebook usage 0.3% → 2.2%, validation reconstruction FID 29.2 → 11.2, reconstruction IS도 증가.[^1]
- **TimeSformer-VQGAN (Video, BAIR/UCF101):**
    - STE 기반 모델은 극심한 codebook collapse(사용률 <1% 수준)로 학습이 붕괴하는 반면, rotation trick 적용 시 codebook usage가 30–40% 수준으로 유지되고, FVD가 크게 개선됩니다.[^1]

또한 저자들이 비교한 두 대안 – Hessian 근사와 double forward exact gradient – 는 대부분 설정에서 FID·IS가 크게 악화되어 rotation trick의 우수성이 강조됩니다.[^1]

### 5.2 한계와 실패 케이스

저자들도 rotation trick의 한계를 명시합니다.[^1]

- **$e$ 또는 $q$의 노름이 0에 가까운 경우:**
$\|e\| \approx 0$ 또는 $\|q\| \approx 0$이고, 특히 둘 사이 각도가 둔각(> $\pi/2$ )이 되면 gradient를 “과도하게 회전(over-rotate)”시키게 됩니다.[^1]
    - 이때 $\nabla_e L$과 $\nabla_q L$의 방향이 반대로 뒤집혀, “ $e \approx q$일 때 $\nabla_e L \approx \nabla_q L$”이어야 한다는 VQ의 자연스러운 inductive bias를 깨뜨립니다.[^1]
- **노름이 매우 커서 원점에서 멀리 떨어진 경우:**
모든 벡터가 큰 상수 벡터 $d$만큼 이동하면 $\angle(e, q)$가 작아지고 $\lambda R$가 점차 항등 변환에 가까워져, rotation trick이 사실상 STE로 수렴합니다.[^1]
→ 이 경우 rotation의 효과가 줄어드는 것이 이론적으로 분석되어 있습니다.[^1]
- **reflection 기반 변형의 실패:**
한 번의 Householder reflection으로 $e$를 $q$에 맞추는 “reflection trick” 도 실험했으나, gradient의 orthogonal 성분을 반대로 뒤집어버려 VQ-VAE/VQGAN 모두 수렴이 악화되고 FID/IS가 크게 나빠졌다고 보고합니다.[^1]
- **특정 코드북 제약과의 상호작용:**
코드북 벡터의 노름을 강하게 0 근처로 묶는 제약(예: 강한 정규화)이 있을 경우, 위의 obtuse-angle 문제 때문에 STE보다 나쁜 성능을 낼 수 있다고 한정합니다.[^1]

***

## 6. 일반화 성능과 Rotation Trick

### 6.1 AutoEncoder-style 정확 그라디언트 vs Rotation Trick

논문은 “정확한 그라디언트가 일반화에 반드시 도움이 되지 않는다”는 것을 강조합니다.[^1]

- **Hessian / exact gradient 접근:**
encoder는 사실상 “양자화를 무시한 AutoEncoder”의 그라디언트로 학습되고, decoder는 VQ-VAE loss로 학습되는 미스매치 구조가 되면서, 훈련 데이터 재구성은 좋아지더라도 test distribution에서의 VQ 구조 활용이 비효율적이거나 과적합 경향을 보입니다.[^1]
실제로 이 방법들은 대부분 설정에서 rotation trick보다 큰 reconstruction loss, 높은 FID, 낮은 IS를 보입니다.[^1]
- **Rotation Trick:**
encoder가 받는 gradient는 여전히 “양자화된 구조”를 강하게 반영합니다.
    - Voronoi 셀 경계의 점은 다른 코드로 “푸시”되어 codebook 사용률과 latent 다양성을 늘리고,[^1]
    - 셀 중심부의 점은 코드북 벡터 주위로 “풀”되어 quantization error를 줄이며,[^1]
    - 양자화 구조를 존중하는 inductive bias를 유지하면서도, AutoEncoder식 과적합을 피할 수 있습니다.[^1]

따라서 rotation trick은 encoder에 **“적당히 부정확하지만 구조를 보존하는”** 그라디언트를 제공함으로써, 오히려 일반화에 유리한 손실 landscape를 만들고 있다고 해석할 수 있습니다.[^1]

### 6.2 기하학적 해석: 평행 이동(Parallel Transport) 관점

부록에서는 STE와 rotation trick을 Riemannian geometry 관점에서 재해석합니다.[^1]

- **STE:**
유클리드 metric을 쓰는 Cartesian 좌표계에서, $\nabla_q L$을 어떤 곡선 $\gamma(t) : q \to e$를 따라 “그대로 유지”하는 parallel transport에 해당합니다.[^1]
→ gradient가 경로와 무관하게 일정하므로, Voronoi 셀 내부 전체가 “checkerboard” 같은 piecewise-constant gradient field가 됩니다.[^1]
- **Rotation Trick:**
반대로, 정규화된 hyperspherical 좌표계(반지름·각도를 기준으로 한 좌표)에 맞춰 basis를 회전한 뒤, 그 좌표계에서 gradient를 parallel transport하고 다시 Cartesian으로 되돌린 것과 동치라고 증명합니다.[^1]
→ 이때는 각도 정보(angular structure)가 자연스럽게 보존되며, 셀 내부 gradient가 위치에 따라 매끄럽게 달라집니다.[^1]

이 관점에서 보면 rotation trick은 “양자화된 latent 공간이 사실상 고차원 구면 위에서 작동한다”는 가정을 내재한 inductive bias를 encoder에 부여하는 것으로, 이는 codebook의 각도 구조를 보존하면서도 안정적인 일반화를 돕는 기하학적 장치로 볼 수 있습니다.[^1]

***

## 7. 앞으로의 연구에 미치는 영향과 고려할 점

### 7.1 향후 연구에 대한 영향

1. **VQ-기반 생성 모델의 기본 구성 요소로 채택 가능성**
이미 공식 코드와 lucidrains의 vector-quantize 라이브러리에 rotation trick이 기본 옵션으로 통합되고 있으며, ImageNet·FFHQ/CelebA-HQ·BAIR/UCF101 등 다양한 데이터셋에서의 효과가 확인된 만큼, 앞으로 VQGAN, latent diffusion, VideoGPT류 모델의 “표준 VQ 계층”으로 채택될 가능성이 큽니다.[^9][^10][^6][^1]
2. **코드북 설계·정규화 연구와의 결합**
entropy/KL 기반 codebook regularization, code splitting/resurrection, hyperbolic metric, multi-codebook 등 기존 코드북 개선 기법들과 rotation trick을 병행하면, gradient estimator·codebook 설계 양 측면에서 상보적인 개선을 얻을 수 있습니다.[^11][^12][^1]
3. **adaptive scaling $\gamma(e)$의 일반화**
부록에서는 $\tilde{q} = \gamma(e) R e + (q - \gamma(e) R e)$ 꼴의 더 일반적인 family를 제시하며, $\gamma(e) = \|q\|/\|e\|$와 $\gamma(e)=1$을 비교한 결과를 보고합니다.[^1] 향후에는 multi-task learning에서의 loss weight adaptation처럼, 데이터·훈련 단계에 따라 $\gamma(e)$를 동적으로 조절하는 방법이 일반화 성능과 stability를 더 높일 여지가 있습니다.[^1][^13][^14]
4. **비-이미지 도메인 및 구조적 데이터로의 확장**
protein 구조 VQ, graph representation VQ, robot skill abstraction 등 다양한 도메인에서 rotation 기반 gradient 메커니즘이 이미 응용되거나 인용되고 있습니다. 이는 “양자화된 토큰”이 쓰이는 거의 모든 분야에서 rotation trick류 기법이 일반화 향상을 가져올 잠재력이 있음을 시사합니다.[^15][^12][^9]

### 7.2 향후 연구 시 기술적으로 고려할 점

- **노름 제약과 각도 분포:**
encoder 출력과 코드북 벡터의 노름 분포가 rotation trick의 동작에 직접적인 영향을 주므로, L2 normalisation, weight decay, spectral norm 등의 정규화가 어떤 각도 분포를 만드는지 면밀히 관찰할 필요가 있습니다.[^1]
- **metric 선택(Euclidean vs cosine vs hyperbolic)과의 상호작용:**
논문은 유클리드/코사인 lookup 모두에서 rotation trick을 실험했으나, hyperbolic VQ나 Gaussian VAE 기반 대안(PCA-VAE 등)에서는 다른 기하학이 적용되므로, rotation-like gradient가 그 기하학과 어떻게 상호 작용하는지를 별도 분석해야 합니다.[^13][^11][^1]
- **downstream 및 task-level generalization 평가:**
현재 평가는 주로 재구성 품질(FID/IS/FVD) 중심입니다.[^1]
representation learning(예: classification, retrieval, RL planning) 관점에서 rotation trick이 어떤 일반화·전달 학습 성능을 보이는지, 특히 VQ 코드가 토큰화된 입력으로 쓰이는 LLM+vision 모델에서의 효과를 체계적으로 평가하는 것이 후속 연구의 중요한 방향입니다.[^16][^10]
- **학습 안정성·스케일링 법칙 분석:**
매우 큰 codebook(수십만 코드)이나 깊은 hierarchical VQ 구조에서 rotation trick이 어떤 scaling behavior를 보이는지, OptVQ, VQBridge와 같은 안정화 기법과의 조합에서 학습 안정성이 어떻게 변하는지에 대한 이론/실증 연구가 필요합니다.[^17][^18][^11]

***

## 8. 2020년 이후 관련 최신 연구 비교 (요약)

아래는 open-access 논문 중, VQ와 STE/gradient·codebook 개선을 다루는 최근 연구 일부를 rotation trick과 비교한 것입니다.


| 연도 | 논문 (제목/링크) | 핵심 아이디어 | Rotation Trick과의 차이 |
| :-- | :-- | :-- | :-- |
| 2024 | **“Restructuring Vector Quantization with the Rotation Trick” – C. Fifty et al., arXiv:2410.06424**[^2][^3][^4] | VQ-VAE의 양자화 층을 회전+스케일 선형 변환으로 재구성하고, 각도 보존 gradient($\partial \tilde{q}/\partial e = \lambda R$)를 통해 codebook utilization·quantization error·FID/IS를 광범위한 설정에서 개선.[^1][^5] | gradient estimator 자체를 교체하는 기하학적 접근; 아키텍처·손실 구조는 거의 건드리지 않음. |
| 2021 | **“Vector-Quantized Image Modeling with Improved VQGAN” – J. Yu et al., arXiv:2110.04627**[^1] | cosine similarity lookup, factorized code, ViT encoder/decoder, perceptual/adversarial loss 설계로 VQGAN 성능과 안정성을 개선.[^1] | 여전히 STE 사용; codebook 설계·아키텍처 측면의 개선으로, rotation trick은 이 구조 위에 gradient estimator로 추가될 수 있음(실험에서도 결합 시 성능 향상 관찰).[^1] |
| 2023 | **“Straightening Out the Straight-Through Estimator: Overcoming Optimization Challenges in Vector Quantized Networks” – M. Huh et al., ICML 2023**[^8][^19][^20] | codebook gradient sparsity와 commitment loss 비대칭을 분석하고, 코드벡터의 affine reparameterization, alternating optimization, 개선된 commitment loss로 STE 기반 VQ의 안정성을 향상.[^20] | 여전히 “복사형” STE 구조를 유지하면서 gradient approximation을 더 잘 맞추는 방향; rotation trick은 “복사”를 포기하고 각도 정보를 보존하는 다른 종류의 estimator. 두 접근은 상보적으로 결합 가능. |
| 2023 | **“Regularized Vector Quantization for Tokenized Image Synthesis” – J. Zhang et al., CVPR 2023**[^1] | codebook 사용률을 uniform에 가깝게 유지하기 위해 KL divergence penalty, entropy regularization 등을 도입해 token diversity와 generative 성능 개선.[^1] | codebook 사용률을 손실 항으로 직접 제어하는 접근; rotation trick은 gradient field를 바꿔 자연스럽게 사용률을 올리는 접근. 둘을 결합하면 손실·gradient 양 측면에서 시너지 가능. |
| 2024 | **“Preventing Local Pitfalls in Vector Quantization via Optimal Transport (OptVQ)” – Y. Zhang et al., arXiv:2412.15195**[^11] | 최근접 이웃 대신 Sinkhorn 기반 optimal transport로 assignment를 수행하여 local minima 및 코드북 붕괴를 방지.[^11] | 양자화 assignment(앞단)를 바꾸는 방법; rotation trick은 assignment는 그대로 두고 backprop의 gradient 이동(뒷단)을 바꾸므로, 개념적으로 직교적인 개선. |
| 2026 | **“PCA-VAE: Differentiable Subspace Quantization without Codebook Collapse” – Y. Guo et al., arXiv:2602.18904**[^13] | codebook을 아예 없애고, Oja’s rule 기반 online PCA bottleneck을 사용해 완전 미분 가능한 subspace quantization을 제안, CelebA-HQ 등에서 VQ-GAN보다 더 적은 비트로 더 나은 재구성 달성.[^13] | “VQ 자체를 대체”하는 접근으로, rotation trick은 기존 VQ-VAE 패러다임 내부에서 gradient propagation만 바꾸는 대안. 설계 선택(토큰화 vs 연속 subspace)에 따라 상호 배타적일 수 있음. |
| 2025 | **“VAEVQ: Enhancing Discrete Visual Tokenization through Variational Latent Quantization” – (VLQ/RCS/DCR framework)**[^16] | VAE latent 위에서 VQ를 수행하고, pre/post-quantization 정렬(RCS)과 코드 분포 정규화(DCR)를 통해 codeword activation과 token 품질 증진.[^16] | VAE prior와 코드북 분포 정합에 초점을 둔 probabilistic 접근. rotation trick은 deterministic VQ에서 gradient 흐름의 기하학을 조정한다는 점에서 서로 보완 가능. |

이처럼 2020년 이후 연구들은

- (1) **gradient estimator 개선(STE 분석·rotation trick)**,[^8][^1]
- (2) **codebook 설계·정규화 개선(regularized VQ, OptVQ, VAEVQ)**,[^12][^16][^11][^1]
- (3) **VQ 자체 대체(PCA-VAE 등)**[^13]
라는 세 방향으로 진화하고 있으며, 본 논문의 rotation trick은 (1)에 속하면서도 (2)와 자연스럽게 결합 가능한 구조라는 점에서 앞으로의 토큰화·압축·생성 모델 연구에 중요한 레고 블록 역할을 할 것으로 보입니다.[^10][^2][^6][^1]
<span style="display:none">[^21][^22][^23][^24][^25][^26][^27][^28]</span>

<div align="center">⁂</div>

[^1]: 2410.06424v2.pdf

[^2]: https://arxiv.org/abs/2410.06424

[^3]: https://arxiv.org/html/2410.06424v1

[^4]: https://openreview.net/forum?id=GMwRl2e9Y1

[^5]: https://linnk.ai/insight/neural-networks/improving-vq-vae-performance-by-propagating-gradients-through-vector-quantization-with-the-rotation-trick-U9kk5RFA/

[^6]: https://github.com/cfifty/rotation_trick/blob/main/README.md

[^7]: https://iclr.cc/virtual/2025/oral/31881

[^8]: https://arxiv.org/abs/2305.08842

[^9]: https://arxiv.org/html/2506.03863v3

[^10]: https://www.themoonlight.io/en/review/restructuring-vector-quantization-with-the-rotation-trick

[^11]: https://arxiv.org/html/2412.15195v1

[^12]: https://arxiv.org/html/2508.06588v2

[^13]: https://arxiv.org/html/2602.18904v1

[^14]: https://arxiv.org/html/2602.17133v1

[^15]: https://www.biorxiv.org/content/10.1101/2025.10.01.679833v2.full-text

[^16]: https://arxiv.org/html/2511.06863v1

[^17]: https://arxiv.org/html/2509.10140v1

[^18]: https://arxiv.org/html/2601.22244v2

[^19]: http://arxiv.org/abs/2305.08842v1

[^20]: https://proceedings.mlr.press/v202/huh23a.html

[^21]: https://www.semanticscholar.org/paper/Restructuring-Vector-Quantization-with-the-Rotation-Fifty-Junkins/ca2804b4a91f363c23dfded07c49d6d40ead95e9

[^22]: https://arxiv.org/html/2512.06609v1

[^23]: https://arxiv.org/html/2411.01801v1

[^24]: https://www.youtube.com/watch?v=JhQGnlcTC5Q

[^25]: https://huggingface.co/papers/2305.08842

[^26]: https://x.com/s_scardapane/status/1882390452625957109

[^27]: https://blog.csdn.net/RachelRicher/article/details/138794460

[^28]: https://liner.com/ko/review/understanding-straightthrough-estimator-in-training-activation-quantized-neural-nets

