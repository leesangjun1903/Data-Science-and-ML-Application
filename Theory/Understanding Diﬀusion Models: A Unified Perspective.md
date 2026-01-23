# Understanding Diffusion Models: A Unified Perspective

### 1. 핵심 요약 및 주요 기여

Calvin Luo의 논문 "Understanding Diffusion Models: A Unified Perspective"(2022년 8월)는 확산 모델에 대한 수학적 기초를 통합적으로 설명하는 교육용 리뷰 논문입니다. 이 논문의 핵심 기여는 확산 모델을 변분 자동인코더(VAE)의 계층적 확장으로 보여주고, 세 가지 수학적으로 동등한 목표 함수를 도출했다는 점입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/297b0b6e-4d65-4881-88d0-4165b0fc2052/2208.11970v1.pdf)

**주요 기여:**

1. **통합 이론 틀**: 확산 모델을 마르코프 계층적 변분 자동인코더(MHVAE)의 특수한 경우로 표현
2. **삼중 등가성 증명**: 원본 이미지 예측, 노이즈 예측, 스코어 함수 학습의 수학적 동등성 확립
3. **스코어 기반 모델과의 연결**: 트위디 공식(Tweedie's Formula)을 통해 확산 모델과 스코어 기반 생성 모델의 명시적 연결
4. **유도 기법의 체계화**: 분류기 기반 및 분류기 비기반 가이던스의 수학적 유도
5. **명확한 수학적 표현**: 분산 예약(variance scheduling) 및 신호대잡음비(SNR) 매개변수화

***

### 2. 해결하고자 하는 문제 및 제안하는 방법

#### 문제 정의

확산 모델은 뛰어난 생성 성능을 보이지만, 그 수학적 기초가 산재되어 있었습니다. 다양한 공식화(DDPM, 스코어 매칭, SDE 기반)가 존재하면서 이들 간의 관계가 명확하지 않았습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/297b0b6e-4d65-4881-88d0-4165b0fc2052/2208.11970v1.pdf)

#### 제안 방법

논문은 증거 하한(ELBO, Evidence Lower Bound)에서 출발하여 점진적으로 확산 모델을 유도합니다:

**단계 1: ELBO 기초**

$$\log p(x) = \log \int p(x,z)dz \geq E_{q_\phi(z|x)}\left[\log \frac{p(x,z)}{q_\phi(z|x)}\right]$$

KL 발산 분해:

$$\log p(x) = E_{q_\phi(z|x)}\left[\log \frac{p(x,z)}{q_\phi(z|x)}\right] + D_{KL}(q_\phi(z|x) \| p(z|x))$$

**단계 2: 변분 확산 모델(VDM) 구조**

세 가지 핵심 제약:
- 잠재 차원 = 데이터 차원
- 인코더는 선형 가우시안: $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)I)$
- 최종 분포는 표준 가우시안: $p(x_T) = \mathcal{N}(0, I)$

**단계 3: ELBO의 두 가지 유도**

*첫 번째 유도 (일관성 항):*

$$\text{ELBO} = E_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)] - E_{q(x_{T-1}|x_0)}[D_{KL}(q(x_T|x_{T-1})\|p(x_T))] - \sum_{t=1}^{T-1} E_{q(x_{t-1},x_{t+1}|x_0)}[D_{KL}(q(x_t|x_{t-1})\|p_\theta(x_t|x_{t+1}))]$$

*두 번째 유도 (저분산 형태):*

베이즈 규칙을 이용하여 각 항이 최대 하나의 확률 변수에 대한 기댓값으로 표현:

$$\text{ELBO} = E_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)] - D_{KL}(q(x_T|x_0)\|p(x_T)) - \sum_{t=2}^{T} E_{q(x_t|x_0)}[D_{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))]$$

**단계 4: 지시 분포 도출**

가우시안 성질을 이용한 정확한 지시 분포:

$$q(x_{t-1}|x_t, x_0) \sim \mathcal{N}\left(\frac{\sqrt{\alpha_t(1-\bar{\alpha}_{t-1})}x_t + \sqrt{\bar{\alpha}_{t-1}(1-\alpha_t)}x_0}{1-\bar{\alpha}_t}, \frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}I\right)$$

여기서 $\bar{\alpha}\_t = \prod_{i=1}^t \alpha_i$

***

### 3. 모델 구조 및 수식

#### A. 변분 확산 모델의 핵심 수식

**전향 과정 (Noising Process):**

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon_0, \quad \epsilon_0 \sim \mathcal{N}(0, I)$$

**역향 과정 (Denoising Process):**

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}}\epsilon_0 + \sqrt{\sigma_q^2(t)}\epsilon$$

**신호대잡음비 매개변수화:**

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$$

최적화 목표는 다음과 같이 단순화:

$$\arg\min_\theta \frac{1}{2}(\text{SNR}(t-1) - \text{SNR}(t))\||\hat{x}_\theta(x_t,t) - x_0||_2^2$$

#### B. 세 가지 동등한 목표 함수

**1. 이미지 예측:**

$$\mu_\theta(x_t, t) = \frac{\sqrt{\alpha_t(1-\bar{\alpha}_{t-1})}x_t + \sqrt{\bar{\alpha}_{t-1}(1-\alpha_t)}\hat{x}_\theta(x_t,t)}{1-\bar{\alpha}_t}$$

최적화: $\arg\min_\theta \mathbb{E}[\|\hat{x}_\theta(x_t,t) - x_0\|_2^2]$

**2. 노이즈 예측:**

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}}\hat{\epsilon}_\theta(x_t,t)$$

최적화: $\arg\min_\theta \mathbb{E}\left[\left\|\hat{\epsilon}_\theta(x_t,t) - \epsilon_0\right\|_2^2\right]$

**3. 스코어 매칭:**

트위디 공식에 의해:

$$E[\mu_x|x] = x + \Sigma \nabla_x \log p(x)$$

따라서:

$$\sqrt{\bar{\alpha}_t}x_0 = x_t + (1-\bar{\alpha}_t)\nabla_{x_t}\log p(x_t)$$

$$\nabla_{x_t}\log p(x_t) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\epsilon_0$$

최적화:

$$\arg\min_\theta \mathbb{E}\left[\|s_\theta(x_t,t) - \nabla_{x_t}\log p(x_t)\|_2^2\right]$$

#### C. 조건부 생성을 위한 가이던스

**분류기 기반 가이던스:**

$$\nabla \log p(x_t|y) = \nabla \log p(x_t) + \gamma \nabla \log p(y|x_t)$$

**분류기 비기반 가이던스:**

$$\nabla \log p(x_t|y) = \gamma \nabla \log p(x_t|y) + (1-\gamma)\nabla \log p(x_t)$$

***

### 4. 성능 향상 및 일반화 능력

#### A. 논문의 암시적 성능 향상 요소

**1. 분산 예약의 설계:**

- $\bar{\alpha}_t$의 단조성이 안정적인 학습을 보장
- SNR의 지수 감소는 신호 손실을 방지

**2. 저분산 ELBO 유도:**

두 번째 유도가 첫 번째보다 작은 분산을 갖는 몬테 카를로 추정을 가능하게 함

**3. 스코어 함수와 노이즈의 연결:**

$$\nabla \log p(x_t) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\epsilon_0$$

이는 점진적 노이즈 제거를 스코어 기반 해석으로 통합

#### B. 2023-2025년 최신 연구의 일반화 성능 개선

** 일반화 이론 (NeurIPS 2023):** [arxiv](https://arxiv.org/html/2311.01797)

- 일반화 간격 상한: $O(n^{-2/5})$ (표본 크기에 대해)
- 모델 용량 스케일링: $O(m^{-4/5})$ (차원의 저주 회피)
- 초기 멈춤(Early-stopping)이 최적 일반화 시점 결정

| 지표 | 결과 |
|------|------|
| 표본 크기 의존성 | $n^{-2/5}$ 다항식 수렴 |
| 모델 용량 의존성 | $m^{-4/5}$ 스케일링 |
| 차원 저주 회피 | YES - 대수 복잡도 |
| 최적 멈춤 시점 | 조기(~800 에포크) |

** 확률 유동 거리(2024):** [arxiv](https://arxiv.org/html/2505.20123v1)

- 암기에서 일반화로의 전환: $N/\sqrt{|\theta|}$ 지배
- 이중 하강 현상(Double Descent) 관찰
- 제한된 데이터에서 초기 학습 후 성능 저하

** 데이터 제약 상황에서의 우수성 (2024):** [arxiv](https://arxiv.org/html/2507.15857v1)

- 확산 모델이 반복 데이터를 더 잘 활용
- 암묵적 데이터 증강: 다양한 토큰 순서 노출
- 자동회귀 모델 대비 우수한 다운스트림 성능

| 시나리오 | 확산 모델 | 자동회귀 |
|---------|---------|---------|
| 단일 에포크 | 더 낮은 검증 손실 | 더 높은 성능 |
| 반복 데이터 | **우수 (↑)** | 성능 정체 |
| 다운스트림 작업 | **우수 (↑)** | 표준 성능 |

#### C. 아키텍처 기반 개선사항

** 기술적 개선:** [milvus](https://milvus.io/ai-quick-reference/what-techniques-help-improve-the-generalization-of-diffusion-models)

1. **다양한 학습 데이터**: LAION-5B 같은 대규모 데이터셋 사용
2. **적응형 컴포넌트**: 
   - U-Net 아키텍처의 주의 메커니즘
   - 교차 주의 계층(cross-attention)
3. **정규화 전략**:
   - EMA (지수 이동 평균)를 통한 가중치 안정화
   - 분류기 비기반 가이던스
   - 동적 역값 처리(dynamic thresholding)

***

### 5. 한계 및 개선 방향

#### A. 논문에서 명시된 한계

1. **생물학적 타당성 부재**: 
   - 인간은 반복적인 노이즈 제거로 생성하지 않음
   - 직관적 설명이 어려움

2. **해석 불가능한 잠재 표현**:
   - 인코더가 선형 가우시안으로 고정됨
   - VAE처럼 의미론적 구조를 학습하지 않음

3. **잠재 차원 제약**:
   - 잠재 차원 = 입력 차원 (압축 불가)
   - 계산 효율성 저해

4. **비용 높은 샘플링**:
   - T개의 역향 단계 필요 (T는 매우 큼)
   - 50-1000 스텝 범위

#### B. 2023-2025년 연구의 해결 전략

** 효율적 확산 모델 (2025):** [arxiv](https://arxiv.org/html/2502.06805v1)

- **흐름 매칭(Flow Matching)**: 벡터 필드 직접 학습
- **ODE 기반 샘플링**: 확정적 경로로 속도 향상
- **예측기-보정기(Predictor-Corrector) 스킴**: 더 나은 정확성
- **적응형 솔버**: 연속 시간 수치 적분 최적화

** 모델 압축:** [arxiv](http://arxiv.org/pdf/2410.11795.pdf)

- **지식 증류**: 큰 모델을 작은 모델로 증류
- **양자화**: 정밀도 감소로 메모리 절약
- **프루닝**: 불필요한 매개변수 제거

** 일반화 조정 (2024):** [arxiv](https://arxiv.org/abs/2412.07229)

- **조정된 스코어 기반 생성 모델(MSGM)**: 원하지 않는 콘텐츠 생성 방지
- **스코어 함수 조정**: SDE 과정 중 점진적 리다이렉션

***

### 6. 논문의 연구 영향 및 향후 고려사항

#### A. 기여한 연구 영향

**이론적 기초 강화:**

1. **통합 수학 틀**의 확립으로 다양한 확산 모델 공식화가 동등함을 증명
2. **스코어 기반과 확산 모델의 연결** 명시로 두 패러다임 통합
3. **ELBO 분해**를 통한 명확한 최적화 목표 제시

**실무적 영향:**

1. **교육적 자료**: 명확한 수학적 표현으로 신진 연구자 진입 장벽 낮춤
2. **구현 가이드**: 세 가지 등가 공식이 다양한 구현 전략 제시
3. **가이던스 기법 표준화**: 분류기/비분류기 기반 방법의 이론적 근거

#### B. 향후 연구 시 고려할 점

**1. 연속 시간 확장:**

- SDE 기반 확산으로 무한 타임스텝 극한 고려
- 반사 경계 조건(reflecting boundary) 도입 가능성

**2. 다양한 데이터 구조:**

- 그래프, 점운 구름(point cloud), 함수형 데이터 적용
- 매니폴드 기반 확산 모델 개발

**3. 메모리화 방지:**

에서 지적: "SGM이 순진한 경험적 위험 최소화 시 메모리화 경향" [emergentmind](https://www.emergentmind.com/topics/score-based-generative-modeling)
- 정규화 기법 강화 필요
- 확률론적 생성성 보증 메커니즘

**4. 불확실성 정량화:**

- 스코어 함수 추정 오류의 영향 분석
- 바셀슈타인 불확실성 전파 연구

**5. 역문제 해결:**

- 인페인팅, 초해상도, 색상화 등에서의 조건부 생성
- 물리 기반 제약조건 통합

**6. 다중 모드 분포:**

에서 확인: "모드 간 거리가 클수록 일반화 성능 저하" [arxiv](https://arxiv.org/html/2311.01797)
- 모드 시프트(Mode Shift) 현상 극복 필요
- 계층적 가우시안 혼합 모델 연구

***

### 7. 2020년 이후 관련 최신 연구 비교 분석

| 논문/저작 | 연도 | 주요 기여 | 본 논문과의 관계 |
|----------|------|---------|-----------------|
| **Ho et al. (DDPM)** | 2020 | 실용적 확산 모델 구현 | 본 논문의 이론적 기초 제공 |
| **Song & Ermon (NCSN)** | 2019 | 스코어 매칭 기초 | 본 논문의 스코어 해석 근거 |
| **Song et al. (Score SDE)** | 2020 | 연속 시간 SDE 프레임워크 | 본 논문의 무한 타임스텝 확장 제시 |
| **Kingma et al. (VDM)** | 2021 | 학습 가능 분산 일정 | 본 논문의 SNR 매개변수화 확장 |
| **Saharia et al. (Imagen)** | 2022 | 텍스트-이미지 SOTA | 분류기 비기반 가이던스 응용 |
| **Rombach et al. (Stable Diffusion)** | 2022 | 잠재 공간 확산 | 효율성 개선 실현 |
| ** [arxiv](https://arxiv.org/html/2311.01797) (일반화 이론)** | 2023 | 엄격한 일반화 상한 | 본 논문의 수렴성 분석 보강 |
| ** [arxiv](https://arxiv.org/html/2505.20123v1) (PFD 메트릭)** | 2024 | 일반화 측정 메트릭 | 실무적 평가 방법론 제시 |
| ** [arxiv](https://arxiv.org/html/2507.15857v1) (데이터 제약)** | 2024 | 확산 모델의 데이터 효율성 | 암묵적 정규화 메커니즘 설명 |
| ** [arxiv](https://arxiv.org/html/2502.06805v1) (효율성 조사)** | 2025 | 알고리즘·시스템 최적화 | 실무 구현 가이드 제공 |

***

### 8. 결론

Calvin Luo의 "Understanding Diffusion Models: A Unified Perspective"는 **수학적 엄밀성과 명확한 설명을 통해 확산 모델의 기초를 통합했습니다**. 본 논문은 다양한 공식화(VAE, 스코어 매칭, SDE)의 동등성을 증명함으로써 후속 연구의 이론적 토대를 제공했습니다.

**핵심 혁신:**

1. **ELBO 분해**: 일관성 항과 저분산 형태의 이중 유도로 최적화 가능성 개선
2. **삼중 등가성**: $x_0$, $\epsilon_0$, $\nabla \log p(x_t)$ 예측의 수학적 동등성 증명
3. **스코어 함수 해석**: 트위디 공식을 통한 노이즈와 스코어의 연결 ($\nabla \log p = -\epsilon_0/\sqrt{1-\bar{\alpha}_t}$)

**일반화 성능 향상 경로:**

- **이론적**: 조기 멈춤으로 $O(n^{-2/5})$ 수렴 성능 달성
- **실무적**: EMA, 동적 역값 처리, 분류기 비기반 가이던스로 품질 향상
- **효율적**: 흐름 매칭, ODE 샘플링으로 계산 비용 획기적 감소

**2025년 관점의 의의:**

본 논문은 확산 모델이 **단순 생성 도구에서 통일된 이론적 틀로 발전**하는 과정의 핵심이었습니다. 후속 연구들(일반화 이론, 메모리화 방지, 다양한 데이터 구조 확장)은 모두 이 기초 위에서 성장했으며, 2025년 현재 확산 모델은 **가장 성숙한 생성 모델 패러다임**으로 자리잡았습니다.

***

**주요 참고 문헌**

<span style="display:none">[^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48]</span>

<div align="center">⁂</div>

[^1_1]: 2208.11970v1.pdf

[^1_2]: https://arxiv.org/html/2311.01797

[^1_3]: https://arxiv.org/html/2505.20123v1

[^1_4]: https://arxiv.org/html/2507.15857v1

[^1_5]: https://milvus.io/ai-quick-reference/what-techniques-help-improve-the-generalization-of-diffusion-models

[^1_6]: https://arxiv.org/html/2502.06805v1

[^1_7]: http://arxiv.org/pdf/2410.11795.pdf

[^1_8]: https://arxiv.org/abs/2412.07229

[^1_9]: https://www.emergentmind.com/topics/score-based-generative-modeling

[^1_10]: https://arxiv.org/abs/2504.16081

[^1_11]: https://academic.oup.com/eurpub/article/doi/10.1093/eurpub/ckaf161.1198/8302058

[^1_12]: https://academic.oup.com/eurpub/article/doi/10.1093/eurpub/ckaf161.1795/8303206

[^1_13]: https://link.springer.com/10.1007/s10461-025-04922-5

[^1_14]: https://www.ahajournals.org/doi/10.1161/cir.151.suppl_1.P3114

[^1_15]: https://www.ahajournals.org/doi/10.1161/cir.151.suppl_1.P3006

[^1_16]: https://www.ahajournals.org/doi/10.1161/cir.151.suppl_1.P2066

[^1_17]: https://www.ahajournals.org/doi/10.1161/cir.151.suppl_1.P2169

[^1_18]: http://medrxiv.org/lookup/doi/10.1101/2025.06.30.25330369

[^1_19]: https://comien.org/index.php/comien/article/view/42

[^1_20]: https://arxiv.org/pdf/2209.00796v8.pdf

[^1_21]: https://arxiv.org/pdf/2210.09292.pdf

[^1_22]: http://arxiv.org/pdf/2209.04747v2.pdf

[^1_23]: https://arxiv.org/pdf/2305.00624.pdf

[^1_24]: https://arxiv.org/pdf/2402.16369.pdf

[^1_25]: https://arxiv.org/pdf/2308.13142.pdf

[^1_26]: https://arxiv.org/abs/2209.00796

[^1_27]: https://yonsei.elsevierpure.com/en/publications/diffusion-models-a-comprehensive-survey-of-methods-and-applicatio

[^1_28]: https://pure.korea.ac.kr/en/publications/score-based-generative-modeling-through-stochastic-evolution-equa/

[^1_29]: https://s-space.snu.ac.kr/handle/10371/209605

[^1_30]: https://yang-song.net/blog/2021/score/

[^1_31]: https://proceedings.neurips.cc/paper_files/paper/2023/file/06abed94583030dd50abe6767bd643b1-Paper-Conference.pdf

[^1_32]: https://www.sciencedirect.com/science/article/abs/pii/S1568494625007811

[^1_33]: https://www.lgresearch.ai/blog/view?seq=396

[^1_34]: https://arxiv.org/abs/2407.00783

[^1_35]: https://arxiv.org/abs/2405.13726

[^1_36]: https://zilliz.com/ai-faq/what-techniques-help-improve-the-generalization-of-diffusion-models

[^1_37]: https://ar5iv.labs.arxiv.org/html/2110.00473

[^1_38]: https://arxiv.org/html/2502.06805v2

[^1_39]: https://arxiv.org/html/2209.00796v15

[^1_40]: https://arxiv.org/abs/2011.13456

[^1_41]: https://arxiv.org/html/2311.01797v4

[^1_42]: https://arxiv.org/html/2508.10875v2

[^1_43]: https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_Moderating_the_Generalization_of_Score-based_Generative_Model_ICCV_2025_paper.pdf

[^1_44]: https://arxiv.org/abs/2506.00849

[^1_45]: https://arxiv.org/html/2501.11430v1

[^1_46]: https://arxiv.org/html/2410.08549v1

[^1_47]: https://arxiv.org/html/2509.16499v2

[^1_48]: https://openaccess.thecvf.com/content/WACV2024/papers/Niemeijer_Generalization_by_Adaptation_Diffusion-Based_Domain_Extension_for_Domain-Generalized_Semantic_Segmentation_WACV_2024_paper.pdf
