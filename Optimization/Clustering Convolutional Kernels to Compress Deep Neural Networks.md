# Clustering Convolutional Kernels to Compress Deep Neural Networks

### 1. 핵심 주장 및 주요 기여

**논문의 중심 주장**은 사전학습된 CNN에서 K-평균 클러스터링을 통해 2D 컨볼루션 커널을 클러스터링하여 신경망을 압축할 수 있다는 것입니다. 이는 단순히 매개변수를 줄이는 것이 아니라, 공간적 패턴의 중복성을 체계적으로 활용합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)

**주요 기여점**은 다음과 같습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)

1. **구조화된 커널 클러스터링**: 벡터 양자화와 달리 기하학적 의미를 가진 2D 커널을 클러스터링하는 첫 번째 방법으로, 신경망의 고유한 공간 구조를 고려합니다.

2. **변환 불변 클러스터링(TIC)**: 수평 뒤å집기, 수직 뒤집기, 90도 회전을 통해 하나의 중심점이 최대 8개의 서로 다른 커널을 대표하도록 함으로써, 정규화 효과를 제공합니다.

3. **효율적 가속화**: 중복된 계산을 제거하는 "Add-then-conv"와 "Conv-then-add" 알고리즘을 제시합니다.

4. **일반화 성능 향상**: ResNet-18이 ImageNet에서 10배 이상의 압축률에서 원본 모델보다 우수한 정확도를 달성했습니다.

***

### 2. 해결하고자 하는 문제

**근본적 문제**는 CNN의 과도한 모수로 인한 메모리 사용량과 계산 비용입니다. 특히 모바일이나 임베디드 환경에서는 높은 성능의 GPU 없이 네트워크를 배포하기 어렵습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)

**기존 방법의 한계**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)
- **가중치 제거 기반 방법**: 스칼라 양자화만 사용하여 구조적 중복성을 간과
- **저위수 분해**: 계산 구조의 변화로 인한 하드웨어 비호환성
- **이진/삼진 네트워크**: 극단적 양자화로 인한 성능 저하

**논문의 통찰**은 "대규모 신경망에 백만 개 이상의 3×3 커널이 있으며, 많은 커널이 서로 비슷한 공간적 패턴을 가진다"는 관찰에서 출발합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)

***

### 3. 제안 방법 및 수식

#### 3.1 기본 클러스터링 공식

**정규화된 커널에 대한 K-평균 목적 함수**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)

$$\arg\min_M \sum_{n=1}^{k} \sum_{\hat{w}_{ij}^m \in W_n} \left\| \hat{w}_{ij}^m - \mu_n \right\|_2^2$$

여기서:
- $\hat{w}\_{ij}^m = w_{ij}^m / s_{ij}^m$ (정규화 커널)
- $s_{ij}^m = \text{sign}(w_{ij}^{m*}) \|w_{ij}^m\|_2$ (스케일 매개변수)
- $w_{ij}^{m*}$ (커널의 중심 픽셀)
- $\mu_n$ (클러스터 $W_n$의 중심점)

**핵심 아이디어**: 분자(norm)의 크기는 다르지만 비슷한 모양의 커널들을 같은 클러스터에 할당하고, 부호(sign)까지 고려하여 대칭성도 활용합니다.

#### 3.2 훈련 시 컨볼루션 수식

**클러스터링 후 압축된 컨볼루션**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)

$$y_j = \sum_{i=1}^{C_{in}} s_{ij} \mu_{l_{ij}} * x_i$$

여기서:
- $\mu_{l_{ij}}$ (커널 $w_{ij}$에 할당된 중심점)
- $s_{ij}$ (학습 가능한 스케일 매개변수)
- $l_{ij}$ (클러스터 인덱스)

**장점**: 전체 커널 대신 중심점과 스케일만 저장하면 되므로, 로그₂k + b_s 비트만 필요합니다.

#### 3.3 압축률 계산

**압축률 공식**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)

$$r_{comp} = \frac{N \cdot b_w \cdot h \cdot w}{N(\log_2 k + b_s) + k \cdot b_w \cdot h \cdot w} \approx \frac{b_w \cdot h \cdot w}{\log_2 k + b_s}$$

여기서:
- $N$ (총 커널 수)
- $k$ (클러스터 수)
- $b_w$ (원본 가중치 비트, 보통 32)
- $h, w$ (커널 크기)
- $b_s$ (스케일 매개변수 비트, 논문에서는 16)

이 공식은 크기 대비 압축이 커널 수 대비 클러스터 수의 로그비에 거의 무관함을 보여줍니다.

#### 3.4 변환 불변 클러스터링(TIC)

**TIC 목적 함수**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)

$$\arg\min_{M,T} \sum_{n=1}^{k} \sum_{\hat{w}_{ij}^m \in W_n} \left\| \hat{w}_{ij}^m - \Phi_{t_{ij}^m}(\mu_n) \right\|_2^2$$

**압축된 컨볼루션(TIC 적용)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)

$$y_j = \sum_{i=1}^{C_{in}} \Phi_{t_{ij}}\left( s_{ij} \mu_{l_{ij}} \right) * x_i$$

여기서 $\Phi_t$ (수평 뒤집기, 수직 뒤집기, 회전 등의 기하학적 변환)

**정규화 효과**: 8개의 변환을 허용하면, k개의 중심점이 최대 8k개의 커널을 대표할 수 있습니다. 이는 추가적인 정규화 제약으로 작용합니다.

#### 3.5 가속화 알고리즘

**Add-then-Conv 방식**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)

$$y_j = \sum_{n=1}^{k} \mu_n * \left( \sum_{i=1}^{C_{in}} \delta[n = l_{ij}] s_{ij} x_i \right)$$

**Conv-then-Add 방식**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)

$$y_j = \sum_{i=1}^{C_{in}} s_{ij} z_{i, l_{ij}}$$

여기서 $z_{i,k} = \mu_k * x_i$ (사전 계산된 중간 특성)

***

### 4. 모델 구조 및 훈련 방식

**아키텍처 접근법**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)

1. **사전학습 모델 사용**: 기존 학습된 VGG-16, ResNet, DenseNet 등의 모델에서 출발
2. **커널 정규화 및 클러스터링**: 모든 커널을 K-평균으로 클러스터링
3. **세밀한 조정(Fine-tuning)**: 클러스터 할당을 고정하고 중심점과 스케일만 재학습

**훈련 설정**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)
- 초기 학습률: 5×10⁻³ (기본 학습률 0.1과 다름)
- 에포크: 300 에포크
- 최적화: SGD with momentum 0.9
- 정규화: L2 정규화 (VGG: 5×10⁻⁴, 기타: 1×10⁻⁴)

**다중 클러스터링 전략**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)
- M2: 처음 10개 레이어와 마지막 3개 레이어를 분리하여 클러스터링
- M13: 각 레이어별 개별 클러스터링
- 첫 번째 컨볼루션은 작아서 min(k, 192) 클러스터 사용

***

### 5. 성능 향상 결과

#### 5.1 CIFAR-10 실험

| 모델 | 설정 | 오류율(%) | 크기 배수 | FLOPs 배수 |
|------|------|---------|---------|----------|
| VGG-16 Baseline | - | 5.98 | 58.8MB | 313M |
| VGG-16-C128-TIC2 | 128개 + TIC | 5.92 | 4.91MB (12.0x) | 145M (2.16x) |
| VGG-16-C64-TIC4 | 64개 + TIC | 6.25 | 4.91MB (12.0x) | 145M (2.16x) |
| DenseNet-BC-C128N | 128개, 스케일 제거 | 4.44 | 27KB (37.5x) | 57M (1.96x) |

#### 5.2 ImageNet ResNet-18 결과

| 모델 | Top-1 오류(%) | Top-5 오류(%) | 압축률 | 가속률 |
|------|-------------|------------|------|------|
| ResNet-18 Baseline | 30.2 | 10.9 | - | - |
| C256 | 30.5 | 11.0 | 11.1x | 1.68x |
| **C1024** | **30.1** | **10.7** | **10.3x** | **1.27x** |
| C1024N (스케일 제거) | 32.3 | 12.2 | 23.6x | 1.35x |

**핵심 발견**: ResNet-18의 C1024 설정은 원본 모델(30.2%)보다 Top-1 오류가 낮습니다(30.1%). 이는 클러스터링의 정규화 효과를 시사합니다.

#### 5.3 프루닝 결합 효과

**VGG-16 + 프루닝**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)

| 설정 | 오류율(%) | 크기 | 압축률 |
|------|----------|------|------|
| VGG-16 기준선 | 5.98 | 58.8MB | - |
| VGG-16-필터 프루닝 | 5.91 | 21.0MB | 2.80x |
| VGG-16-C512 | 6.16 | 5.11MB | 11.5x |
| **VGG-16-필터 프루닝-C512** | **5.99** | **2.35MB** | **31.8x** |

***

### 6. 모델의 일반화 성능 향상 가능성

#### 6.1 변환 불변 클러스터링의 정규화 효과

**TIC의 작동 원리**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)
- 수평/수직 뒤집기, 회전 변환을 통해 8개의 서로 다른 커널을 하나의 중심점으로 표현
- 이는 강력한 구조적 제약으로 작용하여 과적합을 방지

**실험 증거**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)
- VGG-16-C128-TIC2 (오류: 5.92%)가 VGG-16-C128 (오류: 6.24%)보다 우수
- VGG-16-C256 (오류: 6.16%)과 비교하여 거의 같은 압축률에서 더 낮은 오류

#### 6.2 무게 공유의 일반화 효과

**이론적 배경**:
- 무게 공유는 모수 공간의 크기를 감소시켜 **구조적 정규화** 역할
- 여러 커널이 같은 중심점을 공유하면, 각 중심점에 대한 그래디언트가 여러 계층에서 누적
- 이는 **더 견고한 최적화 경로** 형성

**실증적 증거**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)
- 원본 모델 대비 높은 압축률에서도 정확도 유지 또는 향상
- 특히 **DenseNet-BC-C64N**에서 37.5배 압축에도 불구하고 4.60% 오류율 달성

#### 6.3 "Occam's Hill" 현상과의 관계

최근 연구에 따르면, 가벼운 모델 압축은 정규화 효과를 통해 **일반화 성능을 향상**시킬 수 있습니다: [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11965593/)

- **압축 초기 단계**: 모델이 과적합되지 않으므로 정확도 향상
- **과도한 압축**: 표현력 부족으로 정확도 저하

본 논문의 결과는 이 현상을 확인하는 초기 증거입니다.

#### 6.4 다층 클러스터링의 용량 증가

**다층 클러스터링의 이점**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)
- 단일 클러스터링: k개 중심점 → $\log_2 k$ 비트
- 다층 클러스터링: nk개 효과적 중심점 → 여전히 $\log_2 k$ 비트

**결과**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)
- VGG-16-C64-M13: 오류율 6.21% (단일 C64: 6.44%)
- 일반화를 위해 필요한 중심점 수를 동적으로 증가시킴

***

### 7. 한계(Limitations)

#### 7.1 기술적 한계

**1. 병목 구조와의 부적응성**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)
- DenseNet-BC와 ResNet-Bottleneck의 1×1 컨볼루션에서 효율이 떨어짐
- 1×1 커널은 공간 구조가 없어 클러스터링 이점 제한

**2. 하드웨어 호환성 문제**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)
- cuDNN과의 호환성 부족 (이론적 FLOPs만 보고)
- 실제 가속은 그룹 컨볼루션 구현 필요

**3. 스케일 매개변수의 추가 비용**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)
- 각 커널마다 스케일을 저장해야 함
- 압축률 감소: 3×3 커널의 경우 36바이트 → 지수 + 스케일 = log₂k + 16비트
- 최대 12배 정도 스케일의 오버헤드

#### 7.2 방법론적 한계

**1. 정규화 기법의 임의성**:
- 중심 픽셀 기반 정규화가 최적인지 불명확
- 다른 정규화 방식의 비교 부족

**2. 클러스터 수 선택의 어려움**:
- k값에 따른 성능-압축 트레이드오프가 문제마다 다름
- 자동 선택 기법 미제시

**3. 세밀한 조정 효율성**:
- 기본 학습률과 다른 5×10⁻³ 학습률 사용 필요
- 각 모델마다 하이퍼파라미터 튜닝 필요

***

### 8. 2020년 이후 관련 최신 연구 비교 분석

#### 8.1 로또 티켓 가설(Lottery Ticket Hypothesis, LTH)

**핵심 개념**: [ojs.aaai](https://ojs.aaai.org/index.php/AAAI/article/view/16980)
무작위로 초기화된 신경망 내에는 독립적으로 훈련할 때 원본 네트워크의 성능에 필적하는 부분 네트워크("당첨 티켓")가 존재합니다.

**LTH와의 비교**:

| 측면 | 클러스터링 커널 | 로또 티켓 가설 |
|-----|-------------|-----------|
| **주요 목표** | 공간적 커널 중복성 제거 | 가중치 중요도 기반 프루닝 |
| **구조성** | 구조화(2D 커널 단위) | 비구조화(가중치 수준) |
| **정규화** | 명시적(무게 공유) | 암시적(초기화 선택) |
| **하드웨어 효율** | 제한적 | 더 나음 |
| **이론적 기반** | 기하학적 변환 불변성 | PAC-베이즈 이론 [arxiv](https://arxiv.org/abs/2205.07320) |
| **일반화 분석** | 변환 불변성의 정규화 | 손실 지형의 평탄함 [arxiv](https://arxiv.org/abs/2205.07320) |

**최신 진전**: [arxiv](https://arxiv.org/abs/2403.15022)
- Few-shot 이미지 분류에서 LTH 적용: 조기 프루닝으로 과적합 완화
- PAC-베이즈 분석: 플랫한 최솟값과 좋은 일반화의 상관관계 규명

#### 8.2 미분 가능한 K-평균(DKM, 2022)

**혁신점**: [arxiv](https://arxiv.org/pdf/2108.12659.pdf)
- K-평균 클러스터링을 어텐션 문제로 재구성
- DNN 매개변수와 클러스터링 중심점을 **동시 최적화**

**기존 방법과의 차이**:

| 측면 | Son et al.(2018) | DKM(2022) |
|-----|-----------------|----------|
| **학습 방식** | 이미 클러스터링된 고정 할당 | 동적 클러스터링 |
| **목적 함수** | 단계 1: 클러스터링, 단계 2: 세밀 조정 | 단일 목적 함수에서 공동 최적화 |
| **수렴성** | 보장 안 됨 | 수렴성 분석 가능 |
| **계산 비용** | O(Nk) K-평균 → O(N) 백프로파게이션 | 더 복잡한 어텐션 메커니즘 |

#### 8.3 지식 증류의 진화(Knowledge Distillation)

**2018년 논문의 관점**: 가중치 공유를 통한 암시적 지식 전달

**2020-2025년 주요 발전**: [aclanthology](https://aclanthology.org/2021.acl-long.162/)

1. **가중치 증류(Weight Distillation, 2021)**: [aclanthology](https://aclanthology.org/2021.acl-long.162/)
   - 교사 네트워크의 모든 매개변수를 학생에게 전달
   - 성과: 기계 번역에서 1.88-2.94배 가속, 경쟁력 있는 BLEU 점수

2. **연합 학습과 KD(Federated Learning with KD, 2022)**: [nature](https://www.nature.com/articles/s41467-022-29763-x)
   - 상호 지식 증류로 멘토-멘티 모델 모두 개선
   - 동적 정밀도 근사로 통신 효율성 98% 향상

3. **설명 가능한 KD(Explainability-based KD, 2025)**: [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S003132032400846X)
   - 클래스 활성화 맵(CAM) 기반 지식 전달
   - 레이블 관련 정보와 구조 관련 정보 모두 포함

**일반화 관점**: 최신 KD 방법들도 정규화 효과를 인정하며, 이는 본 논문의 관찰을 확장합니다.

#### 8.4 최신 프루닝-양자화 하이브리드(2022-2025)

**Global Composite Compression(2023)**: [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10173752/)
$$\text{Compression Ratio} = \frac{\text{Original Size}}{\text{Pruned Size} + \text{Quantized Weights}}$$
- K-평균 + Wasserstein 거리로 20배 압축 달성
- 원본: 정확도 99%, 새 방법: 정확도 99% (동일)

**압축 순서 문제(Order of Compression, 2024)**: [arxiv](https://arxiv.org/pdf/2403.17447.pdf)
- 프루닝 → 양자화의 순서가 효율에 큰 영향
- 최적 순서를 찾는 체계적 접근법 제시

#### 8.5 트랜스포머 압축(2024-2025)

**CompressTracker(2025)**: [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hong_General_Compression_Framework_for_Efficient_Transformer_Object_Tracking_ICCV_2025_paper.pdf)
- 단계 분할 전략(stage division)으로 구조 제약 제거
- 성과: SUTrack 2.42배 가속, 99% 성능 유지

**달성 메커니즘**:

$$\text{FLOPs}\_{compressed} = \sum_{s=1}^{S} \text{FLOPs}\_s^{student} \ll \sum_{s=1}^{S} \text{FLOPs}_s^{teacher}$$

***

### 9. 논문의 앞으로의 연구 영향 및 고려사항

#### 9.1 패러다임 전환

**본 논문이 제시한 관점**:
1. 단순 "가중치 제거"에서 "구조적 중복성 발견"으로 관점 전환
2. 스칼라 양자화에서 **벡터 양자화(특히 2D 커널)**로 한계 설정
3. 프루닝과 다른 **명시적 정규화 메커니즘** 제시

**영향**:
- 이후 클러스터링 기반 압축 방법들의 프레임워크 제공
- DKM(2022)의 동적 클러스터링도 본 방법을 기반으로 발전

#### 9.2 미해결 문제와 향후 연구 방향

**1. 정규화 메커니즘의 이론화**
- **현재**: TIC가 정규화 효과를 낸다는 경험적 증거만 있음
- **필요**: 무게 공유와 변환 불변성이 손실 지형(loss landscape)에 미치는 영향을 수학적으로 분석

**2. 자동 클러스터 수 선택**
- **문제**: 각 레이어마다 최적 k값이 다름
- **솔루션**: 엔트로피 기반, 기울기 기반 자동 선택 알고리즘 필요

**3. 이질적 양자화**
- **현재**: 모든 커널이 같은 클러스터 수 사용
- **개선**: 중요도가 높은 레이어는 더 많은 중심점, 낮은 레이어는 적은 중심점 할당

**4. 대규모 언어 모델(LLM)로 확장**
- **현재(2025)**: LLM 압축은 주로 양자화, 프루닝, KD에 초점 [arxiv](https://arxiv.org/pdf/2504.14772.pdf)
- **기회**: 2D 커널 개념을 1D 선형 레이어로 확장하는 연구

**5. 하드웨어-소프트웨어 공최적화**
- **현재**: 이론적 FLOPs만 보고, cuDNN 미지원
- **필요**: GPU, TPU, 모바일 칩 등에 최적화된 구현

#### 9.3 최신 동향과의 통합 방향

**하이브리드 접근(Hybrid Approach)**:
$$L_{total} = L_{KD} + \lambda_1 \cdot L_{pruning} + \lambda_2 \cdot L_{clustering}$$

최신 연구들은 단일 기법이 아닌 **다중 기법의 조합**을 강조합니다: [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10173752/)
1. 클러스터링으로 구조적 중복성 제거
2. 프루닝으로 채널 또는 필터 제거
3. 지식 증류로 정규화 강화
4. 양자화로 추가 메모리 절감

**구현 순서의 중요성**: [arxiv](https://arxiv.org/pdf/2403.17447.pdf)
최적의 압축 순서를 따르면 추가 성능 개선:

$$\text{Accuracy}\_{sequential} > \text{Accuracy}_{parallel}$$

#### 9.4 일반화 성능 향상의 근본 원리

**세 가지 메커니즘**:

1. **구조적 제약 정규화(Structural Constraint Regularization)**
   - 무게 공유 → 매개변수 공간의 차원 감소
   - 손실 함수의 평탄한 영역으로 수렴 유도

2. **기하학적 불변성 정규화(Geometric Invariance Regularization)**
   - 변환 불변 클러스터링 → 회전/반사에 강인한 표현 학습
   - 데이터 증강과 유사한 정규화 효과

3. **이중 기울기 효과(Dual Gradient Effect)**
   - 한 중심점에 여러 커널이 클러스터링 → 각 중심점이 여러 컨텍스트에서 최적화
   - 더 견고한 표현 학습

**실증 증거**: [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11965593/)
- 경량 압축(compression ratio < 5x): 정확도 향상
- 중간 압축(5x < ratio < 20x): 정확도 유지
- 극단적 압축(ratio > 50x): 정확도 저하

***

### 10. 최종 평가 및 권고사항

#### 10.1 학문적 기여도

이 논문은 **구조화된 모델 압축의 초기 마일스톤**이며:
1. 2D 커널의 기하학적 성질을 활용한 첫 번째 체계적 접근
2. 변환 불변성이 정규화로 작용할 수 있음을 보임
3. 10배 이상의 압축에서 정확도 개선 가능성 제시

#### 10.2 향후 연구 시 고려할 점

**단기(1-2년)**:
- DKM처럼 동적 클러스터링으로 고정 할당 문제 해결
- 각 레이어별 적응적 k값 선택 알고리즘
- 실제 하드웨어에서의 가속 구현 및 벤치마킹

**중기(3-5년)**:
- 트랜스포머 아키텍처에 적응
- 확률론적 확장(소프트 클러스터링)
- 증분 학습 시 클러스터링의 유연성 개선

**장기(5년 이상)**:
- LLM 시대의 새로운 압축 패러다임
- 유니버설 프리트레인 모델의 효율적 미세조정
- 프라이버시 보존 분산 클러스터링

#### 10.3 실무 적용 제안

**적합한 시나리오**:
- ✅ 에지 디바이스의 추론 최적화
- ✅ 메모리 제약이 심한 임베디드 시스템
- ✅ 모바일 애플리케이션(이미지 분류)
- ✅ 구조화된 프루닝과 조합 사용

**부적합한 시나리오**:
- ❌ 실시간 성능이 매우 중요한 경우(하드웨어 최적화 필요)
- ❌ 1×1 컨볼루션이 많은 병목 구조 네트워크
- ❌ 극도로 극단적인 압축(50배 이상)

#### 10.4 기대 효과

**압축 효율**:
- VGG-16 타입: 10-12배 (정확도 유지)
- ResNet-18 타입: 10-23배 (설정에 따라)
- DenseNet 타입: 37-61배 (미세 조정 필요)

**일반화 향상**:
- 경량 압축에서 0.1-0.5% 정확도 개선
- 변환 불변성을 활용한 데이터 증강 효과

***

### 참고 문헌

 Son, S., Nah, S., & Lee, K. M. (2018). Clustering Convolutional Kernels to Compress Deep Neural Networks. ECCV 2018. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8c5699dc-87b2-4e97-95bf-dafef4089633/Sanghyun_Son_Clustering_Kernels_for_ECCV_2018_paper.pdf)

 Global Composite Compression of Deep Neural Network in Wireless Sensor Networks, IEEE 2023. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10173752/)

 DKM: Differentiable K-Means Clustering Layer for Neural Network Compression, arxiv 2022. [arxiv](https://arxiv.org/pdf/2108.12659.pdf)

 Order of Compression: A Systematic and Optimal Sequence to Combinationally Compress CNN, arxiv 2024. [arxiv](https://arxiv.org/pdf/2403.17447.pdf)

 Weight Distillation: Transferring the Knowledge in Neural Network Parameters, ACL 2021. [aclanthology](https://aclanthology.org/2021.acl-long.162/)

 A survey of model compression techniques: past, present, and future, Nature Frontiers 2025. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11965593/)

 Communication-efficient federated learning via knowledge distillation, Nature Communications 2022. [nature](https://www.nature.com/articles/s41467-022-29763-x)

 Winning Lottery Tickets in Deep Generative Models, AAAI 2020. [ojs.aaai](https://ojs.aaai.org/index.php/AAAI/article/view/16980)

 Exploration and Optimization of Lottery Ticket Hypothesis for Few-shot Image Classification, IEEE 2024. [arxiv](https://arxiv.org/abs/2403.15022)

 Analyzing Lottery Ticket Hypothesis from PAC-Bayesian Theory Perspective, arxiv 2022. [arxiv](https://arxiv.org/abs/2205.07320)

 Proving the Lottery Ticket Hypothesis: Pruning is All You Need, ICML 2020. [proceedings.mlr](https://proceedings.mlr.press/v119/malach20a/malach20a.pdf)

 CompressTracker: General Compression Framework for Efficient Transformer Object Tracking, ICCV 2025. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hong_General_Compression_Framework_for_Efficient_Transformer_Object_Tracking_ICCV_2025_paper.pdf)

 The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks, ICLR 2019. [arxiv](https://arxiv.org/abs/1803.03635)

 Knowledge Distillation and Dataset Distillation of Large Language Models, arxiv 2025. [arxiv](https://arxiv.org/pdf/2504.14772.pdf)

 Explainability-based knowledge distillation, Neurocomputing 2025. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S003132032400846X)
