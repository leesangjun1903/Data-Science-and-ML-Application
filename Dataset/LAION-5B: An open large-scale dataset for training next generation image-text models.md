
# LAION-5B: An open large-scale dataset for training next generation image-text models

## 요약 (Executive Summary)

**LAION-5B**는 Common Crawl에서 필터링한 58억 개의 CLIP 필터된 이미지-텍스트 쌍으로 구성된 최대 규모의 공개 멀티모달 데이터셋입니다. 이 중 23.2억 개는 영어, 22.6억 개는 다국어, 12.7억 개는 언어 미분류 샘플로 이루어져 있습니다. 본 논문은 연구 커뮤니티에 공개적으로 접근 가능한 대규모 이미지-텍스트 데이터셋의 부족 문제를 해결합니다. 기존에는 OpenAI의 CLIP(4억 개), Google의 BASIC(66억 개) 등 대규모 데이터셋이 비공개였으므로, 소수의 대형 연구기관만이 멀티모달 모델 연구에 접근할 수 있었습니다. LAION-5B의 공개는 **민주화된 AI 연구**를 가능하게 하며, 독립적인 모델 개발과 검증을 촉진합니다.[1]

***

## 1. 핵심 주장과 기여 (Core Claims and Contributions)

### 1.1 주요 문제 인식

논문은 두 가지 핵심 문제를 제시합니다:[1]

1. **데이터셋 부재**: CLIP과 DALLE-E의 성공 이후 대규모 이미지-텍스트 데이터셋이 필수적이 되었으나, 공개적으로 이용 가능한 데이터셋이 없음
2. **연구 독점화**: 대규모 데이터셋 접근성 제약으로 인해 대형 기관의 연구에만 집중되어 투명성과 재현성 저하

### 1.2 주요 기여

| 기여 영역 | 내용 | 규모 |
|----------|------|------|
| **데이터셋** | 공개 이미지-텍스트 데이터셋 | 5.85억 개 쌍 |
| **성능 검증** | CLIP, GLIDE, Stable Diffusion 재구현 성공 | 경쟁력 있는 영점샷 정확도 달성[1] |
| **인프라** | 온라인 검색 도구, 안전성 필터 | KNN 인덱싱, NSFW/워터마크 탐지[1] |
| **다국어 지원** | 100개 이상 언어 포함 | 기존 대비 100배 증가 |

***

## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 문제의 구조

**핵심 문제**: 대규모 비정제(noisy) 이미지-텍스트 데이터로부터 고품질 학습 신호 추출

**제안된 해결책**: CLIP 기반 자동 필터링 파이프라인

### 2.2 데이터 수집 방법론 및 수식

#### 단계 1: 웹 페이지 필터링
Common Crawl의 HTML IMG 태그에서 alt-text를 추출하여 이미지-텍스트 쌍 생성

#### 단계 2: 유사성 기반 필터링 (핵심)

필터링은 이미지 임베딩 $\mathbf{i}$와 텍스트 임베딩 $\mathbf{t}$ 간의 코사인 유사도(cosine similarity)로 수행됩니다:[1]

$$\text{similarity}(\mathbf{i}, \mathbf{t}) = \frac{\mathbf{i} \cdot \mathbf{t}}{\|\mathbf{i}\| \|\mathbf{t}\|}$$

필터링 임계값:[1]
- **영어**: 0.28 이상
- **다국어**: 0.26 이상

이 필터링으로 원본 50억 개 이미지 중 약 90%를 제거하고, 5.8억 개의 고품질 쌍 유지[1]

### 2.3 CLIP 컨트래스티브 손실 함수

LAION-5B로 훈련하는 모델의 핵심 목적함수는 CLIP의 컨트래스티브 손실입니다. 배치 크기 $N$에 대해:

$$L = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)} + \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(T_i, I_j) / \tau)} \right]$$

여기서:
- $I_i, T_i$: $i$번째 이미지-텍스트 쌍
- $\tau$: 온도 매개변수 (일반적으로 0.07)
- $\text{sim}(\cdot, \cdot)$: 코사인 유사도

이 대칭적 손실은 이미지에서 텍스트로의 매칭과 텍스트에서 이미지로의 매칭을 동시에 학습합니다.[1]

### 2.4 모델 구조

#### ViT 기반 비전 인코더
논문에서 평가한 모델들:
- **ViT-B/32**: 기본 규모
- **ViT-B/16**: 중간 규모
- **ViT-L/14**: 대규모 (OpenAI CLIP 대응)

각 인코더는 이미지를 $d$ 차원 벡터로 변환 ($d$ = 일반적으로 512):

$$\mathbf{i} = \text{ViT}_{\theta}(\text{image})$$

#### 텍스트 인코더
Transformer 기반 텍스트 인코더는 텍스트를 동일한 $d$ 차원으로 변환:

$$\mathbf{t} = \text{TextEncoder}_{\phi}(\text{tokenize(caption)})$$

***

## 3. 성능 향상 및 일반화 능력

### 3.1 영점샷(Zero-shot) 분류 성능

LAION-400M 및 LAION-2B-en으로 훈련한 모델의 ImageNet-1k 성능:[1]

| 모델 | LAION-400M | LAION-2B-en | OpenAI CLIP | 개선도 |
|------|-----------|-----------|-----------|--------|
| **ViT-B/32** | 62.9% | 65.7% | 63.3% | +2.4%p |
| **ViT-B/16** | 67.0% | - | 68.3% | -1.3%p |
| **ViT-L/14** | 72.8% | 75.2% | 75.6% | -0.3%p |

### 3.2 분포 이동(Distribution Shift) 강건성

ImageNet 분포 이동 데이터셋 성능:[1]

| 데이터셋 | LAION-2B-en | OpenAI CLIP | 차이 |
|---------|-----------|-----------|------|
| ImageNet-R | 87.4% | 87.9% | -0.5%p |
| ImageNet Sketch | 63.3% | 59.6% | **+3.7%p** |
| ObjectNet | 65.5% | 69.0% | -3.5%p |

**결과 해석**: LAION 모델은 특정 분포 이동(예: 스케치)에서 우수하나 다른 이동에서는 약간 뒤떨어짐. 이는 LAION의 데이터 특성과 Common Crawl의 편향을 반영합니다.

### 3.3 계산 규모와 성능의 관계

스케일 법칙(Scaling Laws) 분석 (Figure 4 기반):

$$\text{Accuracy} \propto (\text{GMACS} \times \text{samples seen})^{\alpha}$$

여기서 $\alpha \approx 0.07-0.12$ (log-log 공간에서의 기울기)

- VTAB+: 정확도가 계산량의 로그에 비례하여 증가
- 400M에서 2B로 증가 시, 동일 계산량에서 **1-2%p 정확도 향상**[1]

이는 **데이터 규모가 모델 성능 향상의 핵심 요소**임을 입증합니다.

### 3.4 다운스트림 작업 성능

#### 이미지-텍스트 검색 (MSCOCO)
- **텍스트 검색 R@1**: 59.3% (CLIP WIT: 58.4%) **+0.9%p**[1]
- **이미지 검색 R@1**: 42% (CLIP WIT: 37.8%) **+4.2%p**[1]

#### 퓨샷 선형 프로브 (Few-shot Linear Probe)
16개 샘플 기준 ImageNet 정확도:
- ViT-B/32 LAION-400M: ~73%
- ViT-B/32 OpenAI CLIP: ~72%
- **거의 동등한 전이 학습 성능**[1]

***

## 4. 모델의 일반화 성능 향상 (특별 초점)

### 4.1 일반화의 세 가지 차원

#### 1) 작업 일반화 (Task Generalization)
- VTAB+ 벤치마크에서 35개 작업 평균 성능 측정
- ViT-L/14 LAION-2B-en: 54.6% (ViT-L/14 OpenAI CLIP: 55.7%)
- 차이는 미미하지만, **LAION은 공개 데이터로 동등 성능 달성**

#### 2) 분포 외 일반화 (OOD Generalization)
LAION의 분포 이동 강건성 분석:

$$\text{OOD Gap} = \text{Accuracy}(\text{In-Distribution}) - \text{Accuracy}(\text{OOD})$$

결과:
- **ImageNet-v2**: -2.0%p (약한 분포 이동)
- **ObjectNet**: -3.5%p (강한 분포 이동)

이는 OpenAI CLIP (-6.0%p)보다 우수하여, **LAION의 다양한 데이터 특성이 분포 이동 강건성 강화**함을 시사합니다.[1]

#### 3) 언어 일반화 (Linguistic Generalization)
다국어 2.26억 개 샘플:
- 상위 5개 언어: 러시아어(10.6%), 프랑스어(7.4%), 독일어(6.6%), 스페인어(6.6%), 중국어(6.3%)[1]
- **기존 100배 규모 다국어 데이터 확보**로 저자원 언어 연구 가능

### 4.2 일반화 성능 향상의 메커니즘

#### 데이터 다양성
- **Domain Coverage**: 제품, 미술, 과학, 뉴스 등 광범위한 웹 도메인
- **장황 분포(Long-tail Distribution)**: 네거티브한 측면도 포함되어 편향 감소

#### 필터링의 영향
**분석 발견**: 더 큰 CLIP 모델(ViT-L/14)로 필터링하면 데이터 품질 향상 가능[1]

$$P(\text{select} | \text{model size}) = f(\text{ViT-B/32}) < f(\text{ViT-L/14})$$

### 4.3 스케일과 일반화의 관계 (최신 분석)

**"No 'Zero-Shot' Without Exponential Data" (2024) 논문의 발견**:[2]

멀티모달 모델의 "영점샷" 일반화는 실제로는 지수적 데이터 스케일링을 요구합니다:

$$\text{Performance} \propto \exp(\lambda \cdot \log(\text{Data Size}))$$

- LAION 기반 모델도 예외가 아니며, 어떤 개념에 대해 선형 개선을 위해 지수적 더 많은 데이터 필요
- 이는 LAION-5B의 규모가 여전히 충분하지 않을 수 있음을 시사

***

## 5. 기술적 한계

### 5.1 데이터 중복

Common Crawl에 포함된 이미지 중복:
- **CLIP 임베딩 기반 제거 가능하나** 아직 미실행
- 다운스트림 벤치마크와의 잠재적 중복 문제

$$\text{Data Leakage Risk} = \text{Overlap}(\text{LAION}, \text{Benchmark Test Set})$$

OpenAI는 CLIP 평가에서 중복 영향이 제한적임을 보였으나, 체계적 측정 필요[1]

### 5.2 Alt-text 품질 문제

Alt-text의 특성:[1]
- SEO 스팸 (검색 엔진 최적화 키워드)
- 일관성 없는 키워드 나열
- 부정확한 이미지 설명

**한계**: 텍스트 품질이 모델 훈련에 직접 영향

### 5.3 CLIP 필터링의 편향 전이

필터링 과정의 순환 문제:
- ViT-B/32로 필터링 → CLIP의 편향 상속
- 예: 추상적 작업(물체 계산)에서 약한 성능 유지[1]

### 5.4 구조화된 작업 성능 부족

VTAB의 구조화된 작업들(CLEVR, DSPRITES, SmallNORB 등):
- CLEVR Counts: ~5-10% 정확도 (매우 낮음)
- 원인: 심도 예측, 위치/각도 예측 등 수치적 추론 부재

***

## 6. 안전성 및 윤리 고려사항

### 6.1 컨텐츠 필터링

제공된 탐지기:[1]

1. **NSFW 분류기** (MLP on CLIP embeddings)
   - 학습 데이터: 682K 이미지 (5개 범주)
   - 범주: neutral, drawing, porn, hentai, sexy
   - 3% 이미지가 NSFW 태그됨[1]

2. **워터마크 탐지**

3. **유해 콘텐츠 분류**

### 6.2 편향과 차별

**주요 우려사항**:[1]
- 인종, 성별, 종교 편향 전파
- 개인정보 노출 위험

**대응책**:
- 탐지기 제공 (완벽하지 않음)
- 사용자의 책임 있는 사용 촉구
- 학술 연구 용도 권장

***

## 7. 최신 연구에 미치는 영향 및 고려사항

### 7.1 LAION-5B가 가져온 변화

#### a) 이미지 생성 모델 민주화
**Stable Diffusion (2022)**:[3]
- LAION-2B-en, LAION-Aesthetic의 부분집합으로 훈련
- 공개적으로 이용 가능한 최초의 고품질 텍스트-이미지 모델
- **연구와 실무 활용의 판도 변화**

#### b) 다중모드 모델 연구 개방화

**주요 활용**:[1]
- BLIP (Vision-Language Pretraining)
- MAGMA (VQA 작업)
- VQ-Diffusion (확산 기반 생성)

### 7.2 최근 발견사항과 개선 방향 (2023-2025)

#### 1) 개념 분포의 장황 특성
**"The Neglected Tails in Vision-Language Models" (2024)**:[4]
- LAION은 장황 분포를 가진 개념 편향 존재
- 드문 개념에 대한 성능 저하
- **해결책**: 타겟 서브셋 생성 및 재가중화

#### 2) 공간 추론의 약점
**"What's Up with Vision-Language Models?" (2023)**:[5]
- 기본적인 공간 관계 추론 부족 (위/아래, 좌/우)
- LAION 기반 모델도 동일한 한계
- **원인**: 학습 데이터에서 공간 관계의 명시적 표현 부재

#### 3) 다중모달 콘텐츠 편향 분석
**"Into the LAIONs Den" (2023)**:[6]
- 증오 표현, 인종차별적 스테레오타입 존재
- 웹 크롤 기반 데이터의 내재적 한계
- **개선 필요**: 활발한 큐레이션과 커뮤니티 기여

### 7.3 향후 연구 시 고려사항

#### 1) 데이터 품질 개선
```
권장사항:
- ViT-L/14 이상 모델로 재필터링
- 다단계 필터링 (CLIP + 추가 분류기)
- 커뮤니티 주석(annotation) 수집
```

#### 2) 균형 있는 부분집합 생성
$$\text{Balanced Subset} = \text{Arg}\min_S \text{KL}(P_{\text{target}} || P_S)$$

예: LAION-Aesthetic (120M, 미학적 우수 이미지)[1]

#### 3) 지역 정렬(Fine-grained Alignment) 강화
- 토큰 수준의 이미지-텍스트 정렬
- 예: FILIP (Fine-grained Interactive Language-Image Pre-training)

#### 4) 다국어 모델의 공정성 평가
- 고자원 언어 대비 저자원 언어의 성능 격차 분석
- 다국어 편향 평가 벤치마크 개발

***

## 8. 결론 및 향후 과제

### 8.1 LAION-5B의 의의

| 차원 | 기여 | 파급력 |
|------|------|--------|
| **연구 민주화** | 공개 대규모 데이터셋 제공 | 독립 연구자 진입 가능 |
| **투명성** | 편향과 한계 명시 | 책임 있는 AI 개발 촉진 |
| **생태계** | 오픈소스 도구 제공 | Stable Diffusion 등 혁신 모델 가능 |
| **다국어** | 100배 다국어 데이터 | 저자원 언어 연구 기회 |

### 8.2 남은 과제

1. **데이터 품질**: 알트텍스트 의존성 극복, 더 정교한 필터링
2. **편향 완화**: 체계적인 편향 분석 및 제거 메커니즘
3. **특정 작업 성능**: 구조화된 작업(계산, 공간 추론) 개선
4. **확장성**: 최신 모델(예: ViT-L/14 기반 재필터링) 적용

### 8.3 최종 평가

LAION-5B는 **대규모 멀티모달 AI 연구의 민주화**를 이루었으나, 웹 크롤 데이터의 본질적 한계를 여전히 가집니다. 향후 연구는:

$$\text{Quality} = \text{Scale} \times \text{Curation} \times \text{Diversity}$$

의 균형을 맞춰야 하며, 커뮤니티 기반의 지속적 개선이 필수적입니다. 특히 편향 분석, 미세 정렬, 그리고 특정 응용 분야에 최적화된 부분집합 개발이 향후 주요 과제입니다.[6][4][2][1]

***

## 참고: 수식 요약

**1. 필터링 유사도**
$$\text{sim}(\mathbf{i}, \mathbf{t}) = \frac{\mathbf{i} \cdot \mathbf{t}}{\|\mathbf{i}\| \|\mathbf{t}\|}$$

**2. 대칭 CLIP 손실**
$$L = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)} + \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(T_i, I_j) / \tau)} \right]$$

**3. 스케일 법칙**
$$\text{Accuracy} \propto (\text{GMACS} \times \text{samples})^{\alpha}$$

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0fb9ac65-c0a1-4c35-8882-2b9e795c3c16/2210.08402v1.pdf)
[2](https://arxiv.org/abs/2404.04125)
[3](https://github.com/CompVis/stable-diffusion)
[4](https://arxiv.org/html/2401.12425v1)
[5](https://aclanthology.org/2023.emnlp-main.568.pdf)
[6](https://arxiv.org/pdf/2311.03449.pdf)
[7](https://arxiv.org/abs/2210.08402)
[8](https://arxiv.org/pdf/2204.09817.pdf)
[9](https://arxiv.org/pdf/2306.04387.pdf)
[10](http://arxiv.org/pdf/2108.10904v3.pdf)
[11](http://arxiv.org/pdf/2412.12940.pdf)
[12](https://www.eleuther.ai/papers-blog/laion-5b-an-open-large-scale-dataset-for-training-next-generation-image-text-models)
[13](https://www.academia.edu/99077201/LAION_5B_An_open_large_scale_dataset_for_training_next_generation_image_text_models)
[14](https://ar5iv.labs.arxiv.org/html/2210.08402)
[15](https://openai.com/index/clip/)
[16](https://openreview.net/pdf?id=M3Y74vmsMcY)
[17](https://www.semanticscholar.org/paper/LAION-5B:-An-open-large-scale-dataset-for-training-Schuhmann-Beaumont/e5c8960eb2ec034ffbd353ef39fd1cb541d3c7c9)
[18](https://www.sciencedirect.com/science/article/abs/pii/S0885230824001311)
[19](https://laion.ai/laion-5b-a-new-era-of-open-large-scale-multi-modal-datasets/)
[20](https://arxiv.org/abs/2509.25638)
[21](https://ieeexplore.ieee.org/document/9607477/)
[22](https://dl.acm.org/doi/10.1145/3653644.3665209)
[23](https://link.springer.com/10.1007/s11043-024-09680-w)
[24](https://ieeexplore.ieee.org/document/11033867/)
[25](https://academic.oup.com/mnras/article/520/1/24/6989854)
[26](https://ieeexplore.ieee.org/document/10377903/)
[27](https://arxiv.org/abs/2204.09943)
[28](https://arxiv.org/abs/2506.18773)
[29](https://ieeexplore.ieee.org/document/11124123/)
[30](https://arxiv.org/pdf/2304.08480.pdf)
[31](https://arxiv.org/pdf/2305.20088.pdf)
[32](https://arxiv.org/pdf/2302.11084v2.pdf)
[33](https://arxiv.org/html/2309.14580)
[34](https://arxiv.org/pdf/2302.06232.pdf)
[35](https://arxiv.org/pdf/2209.13430.pdf)
[36](https://arxiv.org/pdf/2405.18570.pdf)
[37](https://www.mdpi.com/1099-4300/24/9/1303/pdf?version=1663222147)
[38](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Understanding_the_Behaviour_of_Contrastive_Loss_CVPR_2021_paper.pdf)
[39](https://arxiv.org/abs/2408.06781)
[40](https://www.anyscale.com/blog/processing-2-billion-images-for-stable-diffusion-model-training-definitive-guides-with-ray-series)
[41](https://www.reddit.com/r/learnmachinelearning/comments/18ef8ot/i_dont_understand_clapclip_paper_contrastive_loss/)
[42](https://arxiv.org/abs/2311.02236)
[43](https://lilianweng.github.io/posts/2021-05-31-contrastive/)
[44](https://openaccess.thecvf.com/content/ICCV2021/papers/Hendrycks_The_Many_Faces_of_Robustness_A_Critical_Analysis_of_Out-of-Distribution_ICCV_2021_paper.pdf)
[45](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
[46](https://www.abhik.xyz/concepts/losses/contrastive-loss)
