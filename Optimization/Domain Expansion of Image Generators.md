
# Domain Expansion of Image Generators

## 요약

"Domain Expansion of Image Generators" (Nitzan et al., 2023)는 생성 모델에 새로운 개념인 도메인 확장(Domain Expansion)을 제시한다. 기존의 도메인 적응(Domain Adaptation)과 달리, 이 논문은 원래 도메인의 생성 능력을 유지하면서 동시에 여러 새로운 도메인을 단일 모델에 추가하는 방법을 제안한다. 핵심 혁신은 생성 모델의 잠재 공간에 존재하는 "dormant direction"(출력에 거의 영향을 주지 않는 방향)을 재활용하여 새로운 도메인을 표현하는 것이다. 이를 통해 단일 확장 모델이 100개 이상의 도메인을 동시에 생성할 수 있음을 입증한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

***

## I. 핵심 주장과 주요 기여

### A. 새로운 문제 제시: Domain Expansion

Domain Expansion은 기존의 Domain Adaptation과 본질적으로 다르다. Domain Adaptation은 소스 도메인에서 타겟 도메인으로 모델을 변환하여 원래의 생성 능력을 상실하는 반면, Domain Expansion은 다음의 요구사항을 만족한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

- **원본 도메인 보존**: 원래 도메인의 생성 능력 완전 유지
- **다중 도메인 동시 지원**: 단일 모델이 여러 도메인 처리
- **분리된 표현(Disentanglement)**: 각 도메인이 해석 가능한 선형 방향으로 표현
- **도메인 조합 가능성**: 도메인 간 부드러운 전환 및 합성 지원

### B. 주요 기여

1. **Dormant Direction 활용의 창의성**: 생성 모델의 잠재 공간에서 출력에 영향을 거의 주지 않는 방향들을 발견하고, 이들을 새로운 도메인 표현에 재활용하는 혁신적 접근 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

2. **Domain Adaptation to Expansion 패러다임 전환**: 기존 도메인 적응 방법(StyleGAN-NADA, MyStyle)을 간단한 조정으로 도메인 확장 방법으로 변환 가능함을 보여줌 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

3. **확장성 입증**: 단일 모델으로 100-400개의 도메인까지 확장 가능하며, 이는 기존 다중 도메인 방법들의 5-20개 수준을 크게 상회 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

4. **모델 크기 증가 없음**: 도메인 수가 증가해도 생성기 모델 크기는 변하지 않음 → 배포 효율성 극대화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

***

## II. 해결하고자 하는 문제와 기술적 접근

### A. 핵심 문제의 구조화

Domain Expansion의 문제 설정: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

주어진:
- 사전 학습된 생성기 $G_{src}: \mathcal{Z} \rightarrow \mathcal{X}_{src}$ (소스 도메인)
- N개의 새로운 도메인 적응 태스크 (손실함수 $\mathcal{L}_i$로 정의)

목표:
- 하나의 확장된 생성기 $G^+$를 학습하여 원래 도메인 $D_{src}$와 모든 새로운 도메인 $\cup_i D_i$를 동시에 모델링

제약:
- 원래 도메인의 성능 유지
- 새로운 도메인들 간의 간섭 최소화
- 각 도메인이 선형 방향으로 표현되어 해석 가능성 유지

### B. SeFA 기반의 정규 직교 분해

먼저 잠재 공간을 정규 직교 기저로 분해한다. Semantic Factorization (SeFA)를 사용하여 생성기의 첫 번째 계층 가중치에 대해 SVD를 수행한다: [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2021/papers/Shen_Closed-Form_Factorization_of_Latent_Semantics_in_GANs_CVPR_2021_paper.pdf)

$$W_0 = U\Sigma V^\top$$

우측 특이벡터 $V = [v_1, v_2, \ldots, v_D]$는 의미론적이고 해석 가능한 기저를 형성하며, 특이값이 작은 벡터들이 dormant 방향이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

***

## III. 제안하는 방법 (Method) - 상세 설명

### A. 잠재 공간 구조화

#### 1. Base Subspace 정의

원래 도메인을 표현하는 기본 부분공간을 정의한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

$$Z_{base} = \text{span}(v_{N+1}, \dots, v_D) + \bar{z} \quad (1)$$

여기서:
- $v_{N+1}, \ldots, v_D$: dormant 방향들 (마지막 $D-N$개 벡터)
- $\bar{z}$: 생성기 학습에 사용된 잠재 분포의 평균
- 중요성: 비-dormant 방향을 제외함으로써 원래 도메인의 변이 요소 보존

#### 2. Repurposed Subspace 정의

새로운 도메인 $D_i$를 표현하기 위해 각각의 dormant 방향을 재활용한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

$$Z_i = Z_{base} + sv_i \quad (2)$$

여기서:
- $v_i$: i번째 새로운 도메인에 할당된 dormant 방향
- $s$: 기본 부분공간에서의 거리 (실험에서 $s=20$)
- 기하학적 의미: Base subspace를 dormant 방향 $v_i$를 따라 평행이동

**직관**: 기본 부분공간과 재활용된 부분공간이 평행하므로, 두 부분공간 모두 원래 도메인의 변이 요소(pose, expression 등)를 동등하게 인코딩한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

### B. Domain Adaptation을 Domain Expansion으로 변환

#### 1. 정사영 연산자 (Projection Operator)

각 도메인 적응 손실을 해당 repurposed subspace에만 적용하기 위해, 샘플링된 잠재 코드를 정사영한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

$$\text{proj}_{Z_i}(z) = \sum_{j=N+1}^{D} (v_j^\top (z - \bar{z}))v_j + \bar{z} + sv_i \quad (3)$$

이 연산자의 효과:
- 입력 $z$를 base subspace로 정사영
- Base subspace 상의 계수는 유지
- 새로운 도메인 방향 $v_i$를 따라 이동

#### 2. 확장 손실 함수 (Expansion Loss)

모든 도메인의 손실을 해당 부분공간에만 제한하여 최적화한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

$$\mathcal{L}_{expand} = \sum_{i=1}^{N} \mathbb{E}_{z \sim p_i(z)} \mathcal{L}_i(G(\text{proj}_{Z_i}(z))) \quad (4)$$

여기서:
- $\mathcal{L}_i$: i번째 도메인의 적응 손실 (예: StyleGAN-NADA 손실)
- $p_i(z)$: 새로운 도메인에 해당하는 잠재 분포

**핵심**: 도메인 적응 손실을 $Z_i$에만 적용하므로, 다른 부분공간은 영향 받지 않음

### C. 정규화를 통한 누수 방지 (Regularization)

실제 학습 과정에서 도메인 적응 손실이 base subspace로 "누수"되는 현상이 관찰된다. 이를 방지하기 위해 명시적 정규화를 도입한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

#### 1. Reconstruction Loss (재구성 손실)

동결된 소스 생성기와 확장된 생성기의 출력을 비교하는 L2 및 LPIPS 손실: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

$$\mathcal{L}_{recon} = \lambda_{lpips} \mathcal{L}_{lpips}(G_{src}(z), G(z)) + \lambda_{L2} ||G_{src}(z) - G(z)||_2 \quad (5)$$

- $\lambda_{lpips} = \lambda_{L2} = 10$ (가중치)
- "Replay Alignment"로도 불림 [arxiv](https://arxiv.org/html/2410.06104v1)
- 효과: $G^+$와 $G_{src}$의 출력이 동일 잠재 코드에 대해 유사하도록 유지

#### 2. Base Subspace 정규화

정규화는 **base subspace에만** 적용한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

$$\mathcal{L}\_{reg} = \mathbb{E}\_{z \sim p_{src}(z)} [\lambda_{src} \mathcal{L}_{src}(G(\text{proj}_{Z_{base}}(z))) + \mathcal{L}_{recon}(G(\text{proj}_{Z_{base}}(z)))] \quad (6)$$

여기서:
- $\text{proj}\_{Z_{base}}(z)$: 잠재 코드를 base subspace로 정사영
- $\mathcal{L}_{src}$: 원래 도메인의 생성 손실 (GAN discriminator 손실 등)
- 의미: 새로운 도메인 학습 시에도 원래 도메인 능력 보존 강제

#### 3. 최종 목적 함수

$$\mathcal{L}_{full} = \mathcal{L}_{expand} + \mathcal{L}_{reg} \quad (7)$$

두 손실의 균형:
- Domain expansion 손실로 새로운 능력 추가
- Regularization 손실로 기존 능력 보존

***

## IV. 모델 구조 (Model Architecture)

### A. 지원하는 생성 모델 아키텍처

#### 1. StyleGAN2 (주요 실험 대상)

- **사용 공간**: W 중간 잠재 공간 (512차원)
- **계층 구조**: Style-based generator [arxiv](https://arxiv.org/pdf/1804.04333.pdf)
  - 각 계층이 점진적으로 해상도 증가
  - 잠재 코드가 스타일 정보로 변환
  
- **특징**: 강력한 분리된(disentangled) 표현 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

#### 2. Diffusion Autoencoder (DAE)

- **사용 공간**: 의미론적 잠재 공간 $z_{sem}$ [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025W/FoundGen-Bio/papers/Yadav_A_Multi-domain_Image_Translative_Diffusion_StyleGAN_for_Iris_Presentation_Attack_ICCVW_2025_paper.pdf)
- **구조**: 확산 모델의 부호화기-부호화기 조합
- **목적**: 다양한 생성 모델 아키텍처의 일반성 입증 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

### B. 핵심 구성 요소의 상호작용

| 구성 요소 | 역할 | 입력 | 출력 |
|---|---|---|---|
| **SeFA 분해** | 잠재 공간의 정규 직교 기저 생성 | 사전 학습된 생성기 가중치 | Base/repurposed 방향 |
| **정사영 연산자** | 잠재 코드를 해당 도메인 부분공간으로 제한 | 임의의 잠재 코드 | 제약된 잠재 코드 |
| **도메인 적응 손실** | 새로운 도메인 표현 학습 | 제약된 코드와 타겟 도메인 | 생성기 가중치 업데이트 |
| **정규화** | 기존 도메인 능력 보존 | Base subspace 코드 | 보존 강도 조절 |

### C. 학습 프로토콜

**단계별 학습**:

1. **초기화**: 사전 학습된 $G_{src}$ 가중치로 $G^+$ 초기화
2. **샘플링**: 각 도메인 $i$에 대해 잠재 코드 $z \sim p_i(z)$ 샘플링
3. **정사영**: $z' = \text{proj}_{Z_i}(z)$로 제약
4. **손실 계산**: 
   - 확장 손실: $\mathcal{L}_i(G^+(z'))$
   - 정규화 손실: $G^+(\text{proj}\_{Z_{base}}(z))$에서 계산
5. **역전파**: 두 손실의 가중 합으로 생성기 가중치 업데이트

**하이퍼파라미터**:
- $s = 20$ (거리 파라미터)
- $\lambda_{src} = 1$, $\lambda_{lpips} = \lambda_{L2} = 10$
- 100개 도메인: 40K iterations
- 400개 도메인: 150K iterations

***

## V. 성능 향상 (Performance Improvement)

### A. 정량적 평가

#### 1. StyleGAN-NADA와의 비교 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

개별 repurposed subspace에서 생성된 이미지와 도메인 적응 방법 결과 비교:

**사용자 연구 (2AFC - Two-Alternative Forced Choice)**: 

| 메트릭 | StyleGAN-NADA (기준) | Domain Expansion | 우월성 |
|---|---|---|---|
| 사용자 선호도 (품질) | 41.2% | 58.8% | **+17.6% p** |
| 다양성 (Diversity) | 2.42 ± 0.13 | 2.42 ± 0.13 | 동등 |

- 1440명의 응답자, 32명의 유니크 사용자
- Domain Expansion이 더 선호됨: 더 깨끗한 출력, 하이퍼파라미터 튜닝 용이 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**MyStyle과의 비교**:

| 메트릭 | MyStyle (기준) | Domain Expansion | 설명 |
|---|---|---|---|
| Identity 보존 | 0.80 ± 0.06 | 0.76 ± 0.05 | 약간의 손실 |
| 다양성 | 3.08 ± 0.15 | 3.14 ± 0.14 | 약간의 향상 |

#### 2. 원래 도메인 보존 평가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

Base subspace에서 생성된 이미지의 FID 점수 (낮을수록 좋음):

| 데이터셋 | 부모 모델 | Domain Expansion | 계속 학습 | 결론 |
|---|---|---|---|---|
| FFHQ | 2.77 | 2.80 | 2.75 ± 0.08 | ✓ 유지 |
| AFHQ Dog | 7.43 | 7.51 | 7.38 ± 0.09 | ✓ 유지 |
| LSUN Church | 3.92 | 3.76 | 3.31 ± 0.22 | ✓ 유지 |
| SD-Elephant | 2.30 | 2.70 | 3.91 ± 0.67 | ✓ 유지 |

- Domain Expansion의 FID는 부모 모델 또는 계속 학습과 동등 수준
- 1σ (표준편차) 범위 내 차이: 무시할 수 있는 수준 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

#### 3. 도메인 간 간섭 정량화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

"Sketch" 도메인에 대한 CLIP 오류 추이 (도메인 수 변화에 따른):

| 도메인 수 | "Sketch" CLIP Error | "Sumo" CLIP Error | 간섭 정도 |
|---|---|---|---|
| 1 도메인 | 낮음 | N/A | - |
| 5 도메인 | 낮음 | 불변 | ✓ 미미 |
| 50 도메인 | 낮음 | 불변 | ✓ 미미 |

- "Sumo" subspace의 CLIP 오류가 "Sketch" 학습 중에도 변하지 않음
- 결론: 도메인들이 서로 간섭하지 않으면서 분리됨 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

#### 4. 도메인 조합 능력 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

두 개의 도메인 방향을 선형 조합하여 합성할 때의 CLIP 오류:

**기존 방법 (StyleGAN-NADA, DiffusionCLIP)**:
- 한 도메인 강화 → 다른 도메인 약화
- 트레이드오프 관계 (선형 감소 추세)

**Domain Expansion**:
- 두 도메인 모두 독립적으로 강화 가능
- CLIP 오류 훨씬 낮음 (약 50% 이상 감소) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

#### 5. 확장성 테스트 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

| 도메인 수 | FFHQ FID | StyleGAN-NADA 비교 | 학습 시간 |
|---|---|---|---|
| 1 도메인 | 2.82 | 기준 | 2000 steps |
| 5 도메인 | 2.81 | 균등 | 10K steps |
| 100 도메인 | 2.80 | 균등 | 40K steps |
| 400 도메인 | 2.83 | 균등 | 150K steps |

- 400개까지 확장하며 품질 저하 무시할 수 있음
- 도메인당 학습 시간 선형 증가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

### B. 질적 평가

#### 1. 개별 도메인 품질 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

각 repurposed 방향을 따라 연속적으로 이동:
- Base subspace ($\alpha = 0$): 원래 도메인 (예: 사진 같은 얼굴)
- Repurposed subspace ($\alpha = s$): 새 도메인 (예: 좀비)
- 외삽 ($\alpha > s$ 또는 $\alpha < 0$): 효과 과장/반대 변환

결과: 부드럽고 연속적인 전환이 관찰됨. 이는 생성기가 라텐트 공간에 대해 매끄럽다는 성질 때문 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

#### 2. 도메인 조합 결과 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

다중 도메인 방향을 동시에 조합:
- "Siberian Husky + Cute + Sketch": 모두 동시 적용 가능
- "Boar + Happy + Pop Art": 완벽한 합성 달성
- 신경 쓸 정도의 간섭 없음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

***

## VI. 모델의 일반화 성능 향상 가능성

### A. 다양한 생성 모델로의 일반화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**StyleGAN2**: 
- 512차원 W 공간
- 400개 도메인 확장 가능 (estimated)
- FID 유지 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**Diffusion Autoencoder**:
- 의미론적 잠재 공간 사용
- 동일한 방법으로 확장 가능
- 아키텍처 수정 없이 적용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**암시적 요구사항**:
- Semantic latent space 존재 필수
- Dormant direction이 충분히 존재해야 함
- 선형 해석 가능한 표현 필요

### B. Generalization 메커니즘

#### 1. Base Subspace의 효과

- 새로운 도메인이 원래 도메인의 의미론적 특성 상속
- 예: 얼굴 생성기 → "좀비" 얼굴도 pose, expression 유지
- 이는 부분공간이 **평행**이기 때문 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

#### 2. 정규화의 역할

정규화 손실이 base subspace에만 적용되므로:
- 새 도메인의 자유도 제한 X
- 원래 도메인 특성은 강제 유지
- Trade-off 자동 균형 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

#### 3. 도메인 수 증가에 따른 영향

실험 결과 (100 vs 400 도메인):
- FID: 2.80 vs 2.83 (무시할 수 있는 차이)
- 개별 도메인 품질: 동등
- CLIP 오류: 약간의 수렴 지연 (학습 곡선이 더 평탄) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**해석**: 
- 최대 400개 정도까지는 모델 용량이 충분
- 그 이상은 dormant direction 고갈

### C. 도메인 유사성의 영향

**유사 도메인** (예: 개 → 고양이):
- 원래 도메인과 유사한 변이 요소 공유
- 더 쉬운 학습, 빠른 수렴

**이질적 도메인** (예: 사진 → 추상 미술):
- 더 극단적인 변환 필요
- 하이퍼파라미터 $s$ 값 조정 필요
- 약간의 모드 붕괴 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

***

## VII. 모델의 한계 (Limitations)

### A. 구조적 한계

#### 1. Dormant Direction 가용성의 한계 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**문제**:
- 모든 생성 모델이 충분한 dormant direction을 가지지 않음
- 도메인 수는 이론적으로 약 400개로 제한

**수학적 제약**:
- 잠재 공간 차원: StyleGAN은 512차원
- 원래 도메인 표현에 필요한 차원: ~100개
- 가용 dormant direction: ~400개
- 따라서 최대 400개 도메인 한계

**해결 가능성**:
- 더 큰 잠재 공간 (1000+ 차원)
- 계층적 latent space 구조
- 분산 표현(distributed representation) 활용

#### 2. 선형성 가정의 제약 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**현재 제약**:
- 각 도메인이 선형 방향으로 표현됨
- 고도로 비선형적인 도메인 변환 불가능

**예시**:
- ✓ "사진" → "스케치" (선형적 스타일 변환)
- ✗ 복잡한 구조 변환 (예: 얼굴 → 동물 골격 구조)

**이론적 근거**:
- SeFA가 선형 의미론적 방향만 발견
- 비선형 방향은 찾을 수 없음

#### 3. 거리 파라미터 $s$의 민감성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**최적값**: $s = 20$ (모든 실험에서 사용)

**효과 분석**:

| $s$ 값 | CLIP 수렴 | 시각적 품질 | 문제 |
|---|---|---|---|
| 0 | 느림 | 약함 | 도메인 효과 미약 |
| 1-5 | 느림 | 약함 | 불완전한 변환 |
| **10-30** | **최적** | **최적** | - |
| 50 | 빠름 | 아티팩트 | 흐림, 색상 오류 |

**비직관적 발견**:
- $s = 50$에서 학습 후, 테스트 시 $\alpha < s$ 값으로 보간해도 아티팩트 제거 불가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)
- 암시: $s$는 학습 시 강한 정규화 효과

### B. 성능 한계

#### 1. 학습 시간의 선형 증가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

| 도메인 수 | 학습 iterations |
|---|---|
| 1 | ~2000 |
| 5 | ~10K |
| 50 | ~20-25K |
| 100 | 40K |
| 400 | 150K |

- 도메인당 약 400 iterations
- 여러 도메인 동시 최적화 필요

#### 2. 도메인 완전 변환의 어려움 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**현상**: 
- 극단적 도메인 변환 (예: 사진 → 미니어처)에서 
- 원본 모드가 섞임 (예: 고양이와 개 모두 생성)

**원인**: 
- Dormant direction이 완전한 분리를 제공하지 못함
- 일부 누수 가능성

**해결**: 
- StyleGAN-NADA의 "latent mapper mining" 기법 사용
- 추가 공간으로 완전 변환 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

#### 3. 아키텍처 간 호환성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**검증됨**:
- StyleGAN2 ✓
- Diffusion Autoencoder ✓

**미검증**:
- Vision Transformer 기반 생성기
- 확산 변환기 (Diffusion Transformer)
- 다른 도메인 (텍스트, 음성 등)

### C. 이론적 한계

#### 1. Dormant Direction의 불명확한 정의 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**현 정의**:
- 낮은 특이값을 가진 방향
- LPIPS 거리 < 임계값

**문제**:
- 이진적 분류가 아님 (연속 스펙트럼)
- 명확한 수학적 특성화 부재

**분석**:
- LPIPS 거리로 측정할 때 약 80% 이상이 dormant [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)
- 그러나 명확한 경계 없음

#### 2. 용량 한계의 이론적 근거 부족 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**경험적 발견**:
- 400개 도메인까지는 품질 유지 (FID 2.83)
- 그 이상은 미검증

**이론적 설명 필요**:
- 네트워크 가중치의 지식 압축 한계
- Knowledge Distillation 이론과의 연결
- Information Bottleneck 원리

#### 3. 평행 부분공간 가정의 검증 부족 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**가정**:
- Base subspace와 repurposed subspace가 평행하므로
- 동일한 변이 요소(pose, expression 등) 공유

**현실**:
- 일부 도메인에서는 공유되지 않을 수 있음
- 정량적 검증 필요

***

## VIII. 2020년 이후 최신 연구와의 비교 분석

### A. Domain Adaptation 계열 연구

#### 1. StyleGAN-NADA (2021) [arxiv](https://arxiv.org/abs/2108.00946)

| 특징 | StyleGAN-NADA | Domain Expansion |
|---|---|---|
| **개념** | 텍스트 기반 단일 도메인 적응 | 다중 도메인 동시 확장 |
| **원본 도메인** | 손실됨 | 완벽히 보존 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf) |
| **도메인 수** | 1개 (또는 순차적 적응) | 100-400개 동시 |
| **모델 크기** | 1 모델/도메인 | 1 모델/다중 도메인 |
| **효율성** | 낮음 (모델 중복) | 높음 (공유 모델) |
| **구현 복잡도** | 복잡 (CLIP 통합) | 상대적으로 간단 |

**기술적 비교**:

StyleGAN-NADA의 손실 함수:

$$\mathcal{L}_{NADA} = 1 - \frac{\Delta_I \cdot \Delta_T}{|\Delta_I| |\Delta_T|}$$

여기서 $\Delta_I$, $\Delta_T$는 CLIP 공간의 이미지/텍스트 임베딩 차이

Domain Expansion은 동일한 손실을 **정사영된 부분공간**에만 적용:

$$\mathcal{L}_{expand} = \sum_i \mathbb{E}_{z \sim p_i} \mathcal{L}_{NADA}(G(\text{proj}_{Z_i}(z)))$$

**장단점**:
- NADA 장점: 단순성, 널리 검증됨
- NADA 단점: 원본 손실, 다중 도메인 미흡
- 확장 장점: 다중 도메인, 원본 보존, 효율성
- 확장 단점: 하이퍼파라미터 복잡도 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

#### 2. MyStyle (2022) [openaccess.thecvf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_Disentangled_and_Controllable_Face_Image_Generation_via_3D_Imitative-Contrastive_Learning_CVPR_2020_paper.pdf)

특징: Few-shot 개인화 생성

| 측면 | MyStyle | Domain Expansion |
|---|---|---|
| **입력 데이터** | 개인 사진 100장 | 텍스트 프롬프트 또는 사진 |
| **대상 사용자** | 개인 맞춤화 | 다양한 일반 도메인 |
| **확장성** | 개인당 별도 모델 | 다중 도메인 단일 모델 |
| **신원 보존** | 매우 높음 | 도메인별로 다름 |

**기술적 차이**:
- MyStyle: 라텐트 코드 역변환 후 재구성 손실
- Expansion: 정사영을 통한 부분공간 제약 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

#### 3. StyleDomain (2023) [arxiv](https://arxiv.org/abs/2212.10229v1)

특징: StyleSpace 방향을 통한 효율적 적응

| 요소 | StyleDomain | Domain Expansion |
|---|---|---|
| **최적화 대상** | StyleSpace 방향 (Affine 계층) | 잠재 공간 방향 |
| **도메인 유사성** | 유사 도메인에만 효율적 | 모든 도메인 지원 |
| **다중 도메인** | 제한적 (5-10개) | 우수 (100+ 개) |
| **조합 가능성** | 제한적 | 우수 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf) |

**기술적 기여**:
- StyleDomain: StyleSpace (S 공간)가 더 효율적임을 보임
- Expansion: W 공간 (또는 $z_{sem}$)에서 직접 작동 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

#### 4. One-Shot GenDA (2023) [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_One-Shot_Generative_Domain_Adaptation_ICCV_2023_paper.pdf)

특징: 1개 샘플로 도메인 적응

| 특성 | One-Shot GenDA | Domain Expansion |
|---|---|---|
| **데이터 요구** | 1개 샘플 | 0개 (텍스트 기반) |
| **원본 보존** | 부분적 | 완벽 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf) |
| **경량성** | 매우 높음 (2개 계층만 수정) | 높음 (전체 생성기 업데이트) |
| **확장성** | 낮음 | 높음 |

#### 5. HyperGAN-CLIP (2024) [arxiv](https://arxiv.org/html/2411.12832)

특징: 하이퍼네트워크를 통한 다중 도메인 적응

| 측면 | HyperGAN-CLIP | Domain Expansion |
|---|---|---|
| **접근 방식** | CLIP 공간에 하이퍼네트워크 | 잠재 공간 구조화 |
| **다중 도메인** | 지원 | 지원 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf) |
| **모델 성장** | 선형 증가 | 무성장 |
| **계산 복잡도** | 높음 (하이퍼네트워크) | 중간 (정사영) |

**핵심 차이**:
- HyperGAN: CLIP 공간에서 매개변수 생성
- Expansion: 잠재 공간의 수학적 구조 활용 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

#### 6. StyleGAN-Fusion (2024) [openaccess.thecvf](https://openaccess.thecvf.com/content/WACV2024/papers/Song_StyleGAN-Fusion_Diffusion_Guided_Domain_Adaptation_of_Image_Generators_WACV_2024_paper.pdf)

특징: 확산 모델 기반 고품질 적응

| 특징 | StyleGAN-Fusion | Domain Expansion |
|---|---|---|
| **감독** | Stable Diffusion 가이드 | CLIP 또는 이미지 샘플 |
| **품질** | 매우 높음 (FID 낮음) | 높음 (동등 수준) |
| **단일/다중** | 단일 도메인 | 다중 도메인 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf) |
| **효율성** | 낮음 (확산 모델 필요) | 높음 |

***

### B. Latent Space 분석 및 조작

#### 1. SeFA (Closed-Form Factorization, 2021) [arxiv](https://arxiv.org/abs/2007.06600)

**Domain Expansion에서의 역할**: 핵심 기술

$$\text{SeFA}: W_0 = U\Sigma V^\top \rightarrow \text{정규 직교 기저} \{v_1, \ldots, v_D\}$$

**비교**:

| 방법 | SeFA | GANSpace | InterFaceGAN |
|---|---|---|---|
| **방식** | SVD 기반 (닫힌 형식) | PCA 샘플링 | 감독 최적화 |
| **계산 비용** | 매우 낮음 (<1초) | 중간 | 높음 |
| **필요 데이터** | 사전 학습 가중치만 | 샘플 필요 | 라벨 필요 |
| **Dormant 감지** | 자동 (특이값) | 수동 | 불가능 |

**Domain Expansion이 SeFA를 선택한 이유**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)
- Dormant direction의 자동 감지
- 계산 효율성
- 모든 GAN 아키텍처에 적용 가능

#### 2. GANSpace (PCA 기반, 2020) [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9194389/)

| 특징 | GANSpace | SeFA/Expansion |
|---|---|---|
| **원리** | 샘플된 잠재 코드의 PCA | SVD 생성기 가중치 |
| **Dormant 감지** | 어려움 | 자동 |
| **계산 시간** | 높음 | 낮음 |
| **해석 가능성** | 중간 | 높음 |

#### 3. StyleCLIP (CLIP 기반 편집, 2021) [openaccess.thecvf](https://openaccess.thecvf.com/content/WACV2024/supplemental/Song_StyleGAN-Fusion_Diffusion_Guided_WACV_2024_supplemental.pdf)

**Domain Expansion과의 관계**:
- StyleCLIP: 기존 도메인 내에서 의미론적 편집 (예: 미소 추가)
- Expansion: 새로운 도메인 전체로의 변환 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)
- 기술적 교집합: CLIP 임베딩 활용

#### 4. Latent Space Disentanglement in DiT (2024) [arxiv](https://arxiv.org/abs/2411.08196)

**최신 발견**:
- Diffusion Transformer의 잠재 공간도 자연적 분리됨
- Domain Expansion 원리를 다른 생성 모델로 확장 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

***

### C. 다중 도메인 합성 방법

#### 1. Class-Conditioning 기반 (논문에서 기준선 비교) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**방법**: 생성기에 클래스 조건 벡터 추가

$$G_{\text{cond}}(z, c) = \text{생성기}(\text{MLP}(c) \oplus z)$$

**결과**:
- ✗ 도메인 누수 (sketch가 sumo 이미지에 나타남)
- ✗ 정렬 불일치 (같은 z에 대해 다른 자세)
- Domain Expansion 우월 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

#### 2. StarGAN v2 (2020) [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9157645/)

**개념**: 모든 도메인을 동시에 모델링하는 GAN

| 특징 | StarGAN v2 | Expansion |
|---|---|---|
| **도메인** | 사전 정의된 클래스 | 동적 추가 가능 |
| **도메인 수** | 5-10개 | 100+ 개 |
| **생성 품질** | 높음 | 높음 |
| **확장성** | 낮음 (재학습 필요) | 높음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf) |

#### 3. Domain Re-Modulation (2023) [arxiv](https://arxiv.org/pdf/2302.02550.pdf)

**특징**: Few-shot GAN 적응의 재모듈레이션 접근

| 측면 | Domain Re-Modulation | Expansion |
|---|---|---|
| **입력** | Few-shot 이미지 | 텍스트 또는 이미지 |
| **복잡도** | 높음 | 중간 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf) |
| **도메인 수** | 제한적 | 많음 |

#### 4. Multi-Domain Neuroimaging Harmonization (2025) [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1053811925003003)

**최신 발전**: 확산 기반 다중 도메인 조화

- 단일 확산 모델로 여러 의료 영상 도메인 처리
- Domain Expansion과 유사한 철학 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

***

### D. 비교 요약 표

| 연구 | 연도 | 핵심 기술 | 도메인 수 | 원본 보존 | 효율성 | 적용 도메인 |
|---|---|---|---|---|---|---|
| StyleGAN-NADA | 2021 | CLIP 기반 | 1 | ✗ | ✗ | 광범위 |
| MyStyle | 2022 | Few-shot | 1 | ✓ | ○ | 얼굴 |
| StyleDomain | 2023 | StyleSpace | ~10 | ✓ | ○ | 유사 도메인 |
| One-Shot GenDA | 2023 | 경량 모듈 | 1 | ○ | ✓ | 제한적 |
| Domain Expansion | **2023** | **Dormant + 정사영** | **100-400** | **✓** | **✓** | **광범위** |
| HyperGAN-CLIP | 2024 | 하이퍼네트워크 | ~50 | ○ | ○ | 광범위 |
| StyleGAN-Fusion | 2024 | Diffusion 가이드 | 1 | ✗ | ○ | 고품질 |
| Multi-Domain Diffusion | 2025 | 확산 조화 | ~10 | ○ | ○ | 의료 영상 |

***

### E. 주요 기술 혁신 비교

**Dormant Direction 활용의 혁신성**:

| 관점 | 이전 방법 | Domain Expansion |
|---|---|---|
| **잠재 공간 활용** | 모든 방향 최적화 | Dormant 방향 선택적 활용 |
| **모델 변경** | 전역적 수정 | 국소적 수정 (부분공간 제약) |
| **확장성** | 도메인당 새 모델 | 모델 크기 무변화 |
| **이론적 근거** | 경험적 | 수학적 (부분공간 구조) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf) |

***

## IX. 향후 연구에 미치는 영향 및 고려 사항

### A. 학문적 영향

#### 1. 새로운 패러다임: Domain Expansion vs Domain Adaptation

Domain Expansion이라는 새로운 문제 정의:

**기존 패러다임 (Domain Adaptation)**:
$$G_{src}(D_{src}) \rightarrow G_{\text{adapted}}(D_{target}) \quad (\text{원본 손실})$$

**새로운 패러다임 (Domain Expansion)**:
$$G_{src}(D_{src}) \rightarrow G^+(D_{src} \cup D_{target} \cup \ldots) \quad (\text{모두 보존})$$

**영향**:
- 다중 도메인 생성의 본질적 재정의
- 생성 모델의 활용성 극대화
- 지속적 학습(Continual Learning) 분야와의 연결 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

#### 2. Dormant Space의 기초 이론 제시

**발견**:
- 생성 모델의 ~80%가 잠재 공간의 dormant 방향으로 구성됨 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)
- 이는 모델 과용량(overcapacity) 시사

**이론적 함의**:
- Information Compression 이론
- Neural Network Lottery Ticket 가설과의 연결
- 지식 축약(Knowledge Distillation) 관점 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

#### 3. 부분공간 구조의 효용성

**기여**:
- 생성 모델에서 명시적 부분공간 구조화의 유효성 입증
- 특이 가치 분해(SVD) 기반 방법론의 실용성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

### B. 향후 연구 방향

#### 1. 이론적 심화 연구

**핵심 질문들**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

1. **Dormant direction의 수학적 특성화**
   - 현재: LPIPS 거리 기반의 경험적 정의
   - 필요: 엄밀한 정보론적(information-theoretic) 정의
   - 가능 접근: 
     * Mutual Information $I(z_i; x)$ 분석
     * Jacobian rank 분석
     * Fisher Information Matrix 분석

2. **용량 한계의 이론적 설명**
   - 현재: ~400개 도메인에서 포화(경험적)
   - 필요: 네트워크 가중치와 도메인 수의 관계 모델링
   - 가능 접근:
     * VC 차원 분석
     * Rademacher complexity
     * Model Compression 이론

3. **평행 부분공간과 변이 요소 공유의 검증**
   - 현재: 기하학적 직관
   - 필요: 정량적 검증
   - 가능 방법:
     * 각 도메인에서 pose, expression 등의 일관성 측정
     * CCA (Canonical Correlation Analysis)로 부분공간 유사도 분석

#### 2. 기술적 확장

##### 가능성 1: 비선형 도메인 변환 지원

**현재 제약**: 선형 dormant 방향 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**해결책**:
- 계층적 잠재 공간 구조
- 조건부 경로(conditional path)를 통한 비선형 변환
- 다중 dormant 방향의 비선형 조합

**수식**:
$$Z_i^{\text{nonlinear}} = Z_{base} + \sum_j \alpha_j(z) v_{j} \quad (\alpha_j \text{는 학습 함수})$$

##### 가능성 2: 적응형 파라미터 선택

**현재**: $s=20$ 고정 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**개선**: 도메인별 최적 $s$ 자동 결정
- 메타 학습(meta-learning) 활용
- 강화 학습으로 $s$ 최적화
- 도메인 특성에 따른 자동 조정

##### 가능성 3: 동적 도메인 추가 (Continual Domain Expansion)

**시나리오**:
- 처음: $G^+$를 100개 도메인으로 학습
- 이후: 새로운 도메인 추가 시 기존 도메인 유지 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

**기술적 과제**:
- Catastrophic forgetting 방지
- 남은 dormant direction 추적
- 기존 도메인 간 간섭 관리

#### 3. 아키텍처 확장

##### Vision Transformer 기반 생성기

**Challenge**:
- Dormant direction이 존재하는가?
- SeFA의 적용 가능성?

**기대**:
- 최근 연구에서 DiT의 자연적 분리 확인 [arxiv](https://arxiv.org/abs/2411.08196)
- Expansion 원리 확장 가능성 높음

##### 텍스트, 음성, 3D 등 다른 모달리티

**텍스트 생성**:
- Dormant token position 또는 embedding dimension?
- Attention head의 분리?

**음성 생성**:
- Temporal dormant direction?
- Frequency 영역에서의 활용?

**3D 생성**:
- NeRF 기반 생성기에서의 부분공간 구조?

#### 4. 실제 응용 개발

##### 가능성 1: 산업용 맞춤형 생성 시스템

**시나리오**: 
- 기본 얼굴 생성 모델
- 개인 요청 시 실시간 새 도메인 추가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)
- 모델 크기/비용 증가 없음

**이점**:
- 중앙집중식 배포 가능
- 개인화 + 확장성 동시 달성

##### 가능성 2: 지속적 학습(Continual Learning) 시스템

**요구 사항**:
- 새로운 데이터 도메인 추가 시 기존 능력 유지

**Domain Expansion의 해결책**:
- Base subspace는 고정
- 새 dormant direction 할당 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

##### 가능성 3: 도메인 간 상호작용 활용

**창의적 응용**:
- 두 도메인의 의도적 혼합 (예: 반은 좀비, 반은 사람)
- 도메인 보간(interpolation) 기반 새로운 창작물

#### 5. 이론과 실무의 연결

**지속적 학습(Continual Learning)**:
- 기존: Class-Incremental Learning (새 클래스 추가 시 기존 클래스 성능 저하)
- Expansion: Domain-Incremental Learning (새 도메인 추가해도 기존 도메인 성능 유지) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)
- 상호 학습 가능

**신경망 압축(Neural Network Compression)**:
- Dormant direction 활용 = 네트워크 지식 축약
- Knowledge Distillation 이론과의 통합 연구

***

## X. 종합 결론

### 주요 성과

"Domain Expansion of Image Generators"는 다음의 혁신적 기여를 이룩했다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

1. **개념적 혁신**: Domain Expansion이라는 새로운 문제 정의로 생성 모델 연구의 방향 제시

2. **기술적 혁신**: Dormant direction을 활용한 우아한 해결책 제시
   - 수학적 엄밀성 (정규 직교 분해)
   - 계산 효율성 (정사영)
   - 확장성 (100-400개 도메인)

3. **실용적 영향**: 단일 모델로 다중 도메인 처리
   - 배포 효율성
   - 비용 절감
   - 사용자 경험 개선

4. **이론적 가치**: 생성 모델의 잠재 공간 구조에 대한 새로운 통찰

### 관련 연구와의 위치

Domain Expansion은 StyleGAN-NADA (2021)의 단일 도메인 적응이라는 한계를 극복하면서도, HyperGAN-CLIP (2024) 같은 후속 연구들과 비교해도 다음 면에서 우수하다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

- **확장성**: 명확한 상한선 (400개)을 제시
- **수학적 근거**: 명시적 부분공간 구조
- **효율성**: 모델 크기 무성장 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)

### 향후 방향의 제안

1. **단기 (1-2년)**:
   - 비선형 도메인 변환 지원 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)
   - Vision Transformer 등 새 아키텍처 확장
   - 동적 도메인 추가(continual learning) 구현

2. **중기 (2-3년)**:
   - 다른 모달리티(텍스트, 3D)로 확장 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)
   - Dormant direction의 이론적 특성화
   - 산업용 배포 시스템 개발

3. **장기 (3년 이상)**:
   - 지속적 학습과의 통합 이론 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/674d49e8-50de-4b73-8dcd-30759d0b2f56/2301.05225v2.pdf)
   - 신경망 압축과의 근본적 연결
   - 생성 모델의 근본적 재해석

***

## 참고문헌


<span style="display:none">[^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78]</span>

<div align="center">⁂</div>

[^1_1]: 2301.05225v2.pdf

[^1_2]: https://openaccess.thecvf.com/content/CVPR2021/papers/Shen_Closed-Form_Factorization_of_Latent_Semantics_in_GANs_CVPR_2021_paper.pdf

[^1_3]: https://arxiv.org/html/2410.06104v1

[^1_4]: https://arxiv.org/pdf/1804.04333.pdf

[^1_5]: https://openaccess.thecvf.com/content/ICCV2025W/FoundGen-Bio/papers/Yadav_A_Multi-domain_Image_Translative_Diffusion_StyleGAN_for_Iris_Presentation_Attack_ICCVW_2025_paper.pdf

[^1_6]: https://arxiv.org/abs/2108.00946

[^1_7]: https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_Disentangled_and_Controllable_Face_Image_Generation_via_3D_Imitative-Contrastive_Learning_CVPR_2020_paper.pdf

[^1_8]: https://arxiv.org/abs/2212.10229v1

[^1_9]: https://openaccess.thecvf.com/content/ICCV2023/papers/Alanov_StyleDomain_Efficient_and_Lightweight_Parameterizations_of_StyleGAN_for_One-shot_and_ICCV_2023_paper.pdf

[^1_10]: https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_One-Shot_Generative_Domain_Adaptation_ICCV_2023_paper.pdf

[^1_11]: https://arxiv.org/html/2411.12832

[^1_12]: https://openaccess.thecvf.com/content/WACV2024/papers/Song_StyleGAN-Fusion_Diffusion_Guided_Domain_Adaptation_of_Image_Generators_WACV_2024_paper.pdf

[^1_13]: https://arxiv.org/abs/2007.06600

[^1_14]: https://www.cs.jhu.edu/~ayuille/JHUcourses/VisionAsBayesianInference2022/26/2007.06600Zhou.pdf

[^1_15]: https://ieeexplore.ieee.org/document/9194389/

[^1_16]: https://openaccess.thecvf.com/content/WACV2024/supplemental/Song_StyleGAN-Fusion_Diffusion_Guided_WACV_2024_supplemental.pdf

[^1_17]: https://arxiv.org/abs/2411.08196

[^1_18]: https://ieeexplore.ieee.org/document/9157645/

[^1_19]: https://arxiv.org/pdf/2302.02550.pdf

[^1_20]: https://www.sciencedirect.com/science/article/pii/S1053811925003003

[^1_21]: https://ieeexplore.ieee.org/document/9093579/

[^1_22]: http://arxiv.org/pdf/2210.08884.pdf

[^1_23]: https://www.semanticscholar.org/paper/f0e9a01b1a52577b1ef7af5f0a3d895aa249cd25

[^1_24]: https://www.nature.com/articles/s41598-025-32924-9

[^1_25]: https://ieeexplore.ieee.org/document/9157724/

[^1_26]: https://link.springer.com/10.1007/978-981-15-0146-3_78

[^1_27]: https://www.mdpi.com/2076-3417/10/3/1092

[^1_28]: https://link.springer.com/10.1007/978-3-030-58583-9_4

[^1_29]: https://www.semanticscholar.org/paper/3a0d4a245095ffee14fcef8f3c0e78bf02c66609

[^1_30]: https://arxiv.org/html/2503.03651v1

[^1_31]: https://arxiv.org/pdf/2211.16550.pdf

[^1_32]: http://arxiv.org/pdf/2305.04466.pdf

[^1_33]: https://linkinghub.elsevier.com/retrieve/pii/S0031320324005624

[^1_34]: https://arxiv.org/html/2502.06272v1

[^1_35]: https://arxiv.org/pdf/2508.12987.pdf

[^1_36]: https://arxiv.org/pdf/2106.10600.pdf

[^1_37]: https://arxiv.org/abs/2004.12411

[^1_38]: https://arxiv.org/html/2407.17877v1

[^1_39]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Disentangling_Latent_Space_for_VAE_by_Label_RelevantIrrelevant_Dimensions_CVPR_2019_paper.pdf

[^1_40]: https://arxiv.org/pdf/2501.07837.pdf

[^1_41]: https://arxiv.org/abs/2108.11080

[^1_42]: https://arxiv.org/html/2411.12832v1

[^1_43]: https://arxiv.org/list/physics/new

[^1_44]: https://openaccess.thecvf.com/ICCV2025_workshops/FoundGen-Bio

[^1_45]: https://www.nature.com/articles/s41467-024-51136-9

[^1_46]: https://www.sciencedirect.com/science/article/abs/pii/S1746809424001605

[^1_47]: https://www.nature.com/articles/s41598-023-39278-0

[^1_48]: https://www.sciencedirect.com/science/article/abs/pii/S0031320320302430

[^1_49]: https://openreview.net/forum?id=mfxq7BrMfga

[^1_50]: https://www.themoonlight.io/en/review/reflections-on-disentanglement-and-the-latent-space

[^1_51]: https://liner.com/ko/hub/conference/neurips/year/2020/topic/ml-theory-\&-methods/keyword/domain-adaptation

[^1_52]: https://sciencepubco.com/index.php/IJSW/article/view/33448

[^1_53]: https://arxiv.org/abs/2212.10229

[^1_54]: https://arxiv.org/abs/2310.14222

[^1_55]: https://arxiv.org/pdf/2102.12206.pdf

[^1_56]: https://www.aclweb.org/anthology/D19-1325.pdf

[^1_57]: https://arxiv.org/pdf/1908.09395.pdf

[^1_58]: http://arxiv.org/abs/2108.00946

[^1_59]: https://arxiv.org/html/2601.14053v1

[^1_60]: https://arxiv.org/pdf/2601.14053.pdf

[^1_61]: https://openaccess.thecvf.com/content/CVPR2023/papers/Nitzan_Domain_Expansion_of_Image_Generators_CVPR_2023_paper.pdf

[^1_62]: https://arxiv.org/pdf/2508.01079.pdf

[^1_63]: https://arxiv.org/pdf/2105.14230.pdf

[^1_64]: https://arxiv.org/pdf/2601.01539.pdf

[^1_65]: https://arxiv.org/html/2410.07840v2

[^1_66]: https://arxiv.org/pdf/2007.06600.pdf

[^1_67]: https://wandb.ai/geekyrakshit/stylegan-nada/reports/Digging-Into-StyleGAN-NADA-for-CLIP-Guided-Domain-Adaptation--VmlldzoyMjA5MDU1

[^1_68]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11579284/

[^1_69]: https://history.siggraph.org/learning/stylegan-nada-clip-guided-domain-adaptation-of-image-generators-by-gal-patashnik-maron-bermano-chechik-et-al/

[^1_70]: https://www.nature.com/articles/s41467-024-47120-y

[^1_71]: https://github.com/rinongal/StyleGAN-nada/blob/main/README.md

[^1_72]: https://www.sciencedirect.com/science/article/abs/pii/S1474034625007189

[^1_73]: https://opentutorials.org/course/5078/32279

[^1_74]: https://papers.neurips.cc/paper_files/paper/2022/file/bd1fc5cbedfe4d90d0ac2d23966fa27e-Paper-Conference.pdf

[^1_75]: https://openreview.net/pdf/aa60bcf3c6291c8951d7c8b1700ba37beff32997.pdf

[^1_76]: https://genforce.github.io/sefa/

[^1_77]: https://dl.acm.org/doi/10.1145/3528223.3530164

[^1_78]: https://bise-journal.com/?p=2206


