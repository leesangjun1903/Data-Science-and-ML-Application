# Sharpness-Aware Minimization for Efficiently Improving Generalization

### 1. 핵심 주장 및 주요 기여
**"Sharpness-Aware Minimization for Efficiently Improving Generalization"** 논문(Foret et al., ICLR 2021)은 과매개변수화(overparameterized) 신경망의 일반화 성능 향상을 위한 혁신적인 최적화 기법을 제시합니다. 이 연구의 핵심 통찰은 단순히 훈련 손실(training loss)을 최소화하는 것으로는 좋은 일반화를 보장할 수 없다는 점입니다. 신경망의 손실 지형(loss landscape)은 다양한 지역 최솟값(local minima)을 가지며, 이들이 동일한 훈련 손실을 가질지라도 테스트 성능은 크게 다를 수 있습니다.[1]

논문의 주요 기여는 다음과 같습니다:
- **SAM 알고리즘**: 손실 값(loss value)과 손실의 예리함(loss sharpness)을 동시에 최소화하는 최적화 기법
- **PAC-Bayesian 일반화 경계**: 손실 지형의 국소적 특성과 일반화 능력의 관계를 수학적으로 증명
- **m-sharpness 개념**: 배치 크기에 따라 변하는 새로운 예리함 측정 지표
- **광범위한 실증 검증**: CIFAR-10/100, ImageNet, 전이 학습, 노이즈 레이블 학습 등 다양한 설정에서의 성능 향상 입증

***

### 2. 문제 정의 및 제안 방법
#### 2.1 핵심 문제

현대 딥러닝에서 가장 큰 도전은 과매개변수화 모델이 훈련 데이터에 과적합되기 쉽다는 점입니다. 특히, 훈련 손실 $L_S(w)$는 낮지만 테스트 손실 $L_D(w)$는 높은 경우가 빈번합니다. 이는 최적화 알고리즘이 수렴하는 최솟값의 특성이 중요함을 의미합니다.

#### 2.2 이론적 근거

논문은 다음과 같은 PAC-Bayesian 일반화 경계를 증명합니다:

$$L_D(w) \leq \max_{\|\epsilon\|_2 \leq \rho} L_S(w + \epsilon) + h\left(\frac{\|w\|_2^2}{\rho^2}\right)$$

이를 다시 쓰면:

$$L_D(w) \leq \underbrace{\left[\max_{\|\epsilon\|_2 \leq \rho} L_S(w + \epsilon) - L_S(w)\right]}_{\text{Sharpness}} + L_S(w) + \lambda\|w\|_2^2$$

여기서 제곱괄호 안의 항은 **손실 지형의 예리함(sharpness)**을 나타냅니다. 이 경계는 더 평탄한 최솟값을 찾을수록 더 나은 일반화를 기대할 수 있음을 시사합니다.[1]

#### 2.3 SAM의 최적화 목표

$$\min_w L_{S}^{\text{SAM}}(w) + \lambda\|w\|_2^2$$

여기서:

$$L_S^{\text{SAM}}(w) \triangleq \max_{\|\epsilon\|_p \leq \rho} L_S(w + \epsilon)$$

이는 **min-max 최적화 문제**로, 내부 최대화 문제를 해결한 후 외부 최소화를 수행합니다.[1]

#### 2.4 효율적인 알고리즘 구현

내부 최대화 문제를 1차 Taylor 전개로 근사하면:

$$\hat{\epsilon}(w) = \rho \frac{\text{sign}(\nabla_w L_S(w)) |\nabla_w L_S(w)|^{q-1}}{\|\nabla_w L_S(w)\|_q^{1/p}}$$

여기서 $1/p + 1/q = 1$입니다.[1]

최종적으로 다음의 그래디언트 근사를 얻습니다:

$$\nabla_w L_S^{\text{SAM}}(w) \approx \nabla_w L_S(w + \hat{\epsilon}(w))$$

**Algorithm 1: SAM 의사 코드**

```
입력: 훈련 집합 S, 손실 함수 l, 배치 크기 b, 
       학습률 η, 근처 크기 ρ
초기화: 가중치 w₀, t = 0
반복:
  배치 B 샘플링
  그래디언트 계산: ∇L_B(w)
  섭동 계산: ε̂(w) (식 2)
  SAM 목적 그래디언트: g = ∇L_B(w + ε̂(w))
  가중치 업데이트: w_{t+1} = w_t - ηg
  t = t + 1
반복 종료 시 w_t 반환
```

***

### 3. 성능 향상 및 광범위한 실증 평가
SAM은 다양한 벤치마크에서 일관된 성능 개선을 달성했습니다:

**CIFAR-10/100에서의 성능:**
- CIFAR-10: 2.2% → 1.6% (0.6%p 개선, WideResNet 기준)
- CIFAR-100: 10.6% → 10.3% (0.3%p 개선, PyramidNet+ShakeDrop 기준)[1]
- 이는 이미 정교한 정규화(Shake-Shake, ShakeDrop)가 적용된 모델에서도 추가 개선을 달성

**ImageNet 대규모 실험:**
- ResNet-152 (400 에포크): 20.9% → 18.4% (2.5%p 개선)
- 특히 SAM은 에포크 증가에도 과적합되지 않는 특성 보유[1]

**전이 학습(Finetuning) 설정:**
- EfficientNet-b7 평균: 7.68% → 7.44% 오류율
- EfficientNet-L2는 이전 SOTA를 상당히 초과[1]

| 데이터셋 | SAM | SGD | 개선율 |
|---------|-----|-----|--------|
| CIFAR-10 | 1.6% | 2.2% | 27.3% |
| CIFAR-100 | 10.3% | 10.6% | 2.8% |
| ImageNet | 18.4% | 20.3% | 9.4% |
| SVHN | 0.99% | 1.14% | 13.2% |
| Fashion-MNIST | 3.59% | 3.86% | 7.0% |

***

### 4. 일반화 성능 향상 메커니즘
#### 4.1 손실 지형의 기하학적 특성

논문은 Hessian 행렬의 스펙트럼을 통해 SAM의 효과를 분석했습니다. WideResNet-40-10을 CIFAR-10에서 300 스텝 학습한 결과:[1]

- **최대 고유값 (λ_max)**:
  - SGD: 약 24.2
  - SAM: 약 1.0
  
- **고유값 비율 (λ_max/λ_5)**:
  - SGD: 11.4 (예리한 최솟값)
  - SAM: 2.6 (평탄한 최솟값)

이는 SAM이 실제로 손실 곡면의 곡률을 감소시킨다는 것을 명확히 보여줍니다.

#### 4.2 m-Sharpness와 일반화의 관계

흥미로운 발견 중 하나는 **배치 크기 m에 따른 예리함 측정의 중요성**입니다. 병렬 학습에서 각 가속기가 크기 m인 데이터 부분집합에서 독립적으로 SAM 업데이트를 계산할 때, 이러한 구성적 m-sharpness는 전체 훈련 집합을 사용한 표준 sharpness보다 **일반화 격차와의 상관성이 더 높습니다**.[1]

이는 다음을 시사합니다:
- 더 작은 m 값이 더 나은 일반화 성능을 제공
- 이는 병렬화 필요성과 우연히 일치

#### 4.3 노이즈 레이블에 대한 견고성

SAM은 라벨 노이즈에 대한 자연스러운 견고성을 제공합니다:

| 노이즈율 | SAM | MentorMix | Bootstrap+SAM |
|---------|-----|----------|----------------|
| 20% | 95.1% | 95.6% | 95.4% |
| 40% | 93.4% | 94.2% | 94.2% |
| 60% | 90.5% | 91.3% | 91.8% |
| 80% | 77.9% | 81.0% | 79.9% |

SAM의 섭동 기반 접근이 노이즈 방해에 대한 견고성을 자연스럽게 제공하는 것으로 보입니다.[1]

***

### 5. 방법의 한계
#### 5.1 이론적 한계
1. **선형화 근사의 정당성**: 1차 Taylor 전개가 모든 상황에서 충분한 근사인지에 대한 의문 존재
2. **2차 항의 역할**: 실험적으로 2차 항 제외 시 성능이 더 좋음이 관찰되었으나, 그 이유가 명확하지 않음[1]
3. **Theorem 1의 느슨함**: PAC-Bayesian 경계는 실제 성능보다 상당히 느슨할 수 있음

#### 5.2 계산 효율성
1. **연산 비용**: 각 업데이트마다 두 번의 역전파가 필요하므로 표준 SGD 대비 약 2배의 계산 비용
2. **메모리 사용**: 섭동된 모델의 그래디언트 계산으로 인한 메모리 오버헤드
3. **하이퍼파라미터 튜닝**: ρ 값의 선택이 중요하며, 데이터셋마다 최적값이 상이

#### 5.3 실무적 한계
1. **노이즈가 많은 환경**: 배치 정규화 등 확률적 요소가 많은 경우 섭동 추정의 불안정성
2. **매우 큰 모델**: 메모리 제약으로 인한 적용 어려움
3. **온라인 학습**: 한 번의 샘플로 섭동을 계산하기 어려움

***

### 6. 2020년 이후 관련 최신 연구 비교 분석
#### 6.1 효율성 개선 연구

**ESAM (Efficient SAM, 2021)**[2]
- SAM의 계산 비용을 2배에서 40% 증가로 감소
- 확률적 가중치 섭동(Stochastic Weight Perturbation) 도입
- Sharpness-민감 데이터 선택 기법 제안
- CIFAR-100에서 SAM과 유사한 성능 달성

**K-SAM (2022)**[3]
- 상위 k개의 손실이 큰 샘플만 사용하여 효율성 향상
- SGD 수준의 계산 비용으로 일반화 개선

#### 6.2 이론적 진전

**Unified SAM (2025)**[4]
- SAM과 비정규화 버전(USAM)의 통합 분석
- Polyak-Łojasiewicz 조건 하에서 수렴 보장
- 임의의 샘플링 패러다임 지원

**Friendly-SAM (2024, CVPR)**[5]
- 배치 특정 확률적 그래디언트 노이즈의 중요성 규명
- 지수이동평균(EMA)으로 전체 그래디언트 추정 및 제거
- F-SAM이 SAM보다 더 우수한 견고성 제공
- 이론적 수렴 증명 포함

**Eigen-SAM (2025)**[6]
- Hessian 최대 고유값의 명시적 정규화
- 섭동 벡터와 상위 고유벡터의 정렬 강조
- 3차 확률미분방정식(SDE) 분석으로 동역학 규명

#### 6.3 기하학적 확장

**HyperbolicSAM (2025)**[7]
- Poincaré ball manifold에서 SAM 일반화
- 계층적 구조를 가진 데이터(지식 그래프, 분류 체계)에 최적화
- CIFAR-10에서 2.34% 오류율 달성 (Euclidean SAM의 2.86% 대비)

**Monge SAM (2025)**[8]
- 손실 곡면에서 유도된 리만 메트릭 사용
- 재매개변수화 불변성(reparameterization invariance) 달성
- 안장점에 덜 끌려가는 성질

#### 6.4 도메인 특화 적용

**DGSAM (2025) - Domain Generalization**[9]
- 영역 일반화에서 "가짜 평탄 최솟값" 문제 해결
- 개별 영역 내에서의 예리함 최소화 강조
- 전역 sharpness 보다 개별 영역별 sharpness가 일반화에 더 유용함을 증명

**Modality-Aware SAM (2025)**[10]
- 다중모달 학습(멀티미디어, 음성+비전)에 최적화
- Shapley 값으로 지배적 모달리티 식별
- 불균형 모달 기여 문제 해결

**Focal-SAM (2025) - Long-tail Classification**[11]
- 장꼬리 분포 분류에 클래스별 가중치 적용
- 헤드 클래스와 테일 클래스 간 손실 곡면의 기하학적 차이 반영
- ImbSAM, CC-SAM 대비 효율성과 성능 개선

**CA-SAM (2025) - Noisy Labels**[12]
- Clean-aware sharpness minimization
- 깨끗한 샘플과 노이즈 샘플의 섭동 방향 불일치 분석
- 과거 모델 예측으로 데이터 분할

#### 6.5 강건성 및 안정성 개선

**GCSAM (2025) - Gradient Centralized SAM**[13]
- 그래디언트 중심화 기법 도입
- 계산 효율성과 노이즈에 대한 견고성 개선
- 배치 정규화 변동에 덜 민감

**CR-SAM (2023) - Curvature Regularized**[14]
- 곡률 정규화 항 추가
- PAC-Bayes 경계를 통한 이론적 정당화

#### 6.6 응용 분야 확대

**의료 및 생명과학 응용:**
- Raman 분광기를 통한 세균 저항성 진단에서 2.7% 평균 정확도 향상[15]
- 생물의학 이미지 분석에서 일반화 성능 개선[16]

**컴퓨터 비전 확장:**
- Vision Transformer(ViT)를 포함한 최신 아키텍처 지원
- 반지도 학습, 자기지도 학습에 통합

***

### 7. 향후 연구의 방향성 및 고려사항
#### 7.1 이론적 미해결 문제

1. **m-Sharpness의 완전한 이해**
   - m-sharpness가 왜 전체 데이터셋 기반 sharpness보다 일반화를 더 잘 예측하는가?
   - 이것이 단순 경험적 현상인지, 근본적인 이론적 이유가 있는지 규명 필요

2. **2차 항의 역할**
   - SAM 유도 과정에서 2차 항을 제외할 때 오히려 성능이 향상되는 현상의 이해
   - 이는 현재의 이론적 프레임워크의 불완전성을 시사

3. **재매개변수화 불변성**
   - 손실 기하학이 모델 매개변수화에 어떻게 의존하는가?
   - Monge SAM의 리만 기하학적 접근의 일반화 가능성

#### 7.2 알고리즘적 개선 방향

1. **계산 효율성의 한계 돌파**
   - 현재 ESAM도 40% 추가 비용 필요
   - Hessian-무료(Hessian-free) 방식의 더 효율적인 구현
   - GPU/TPU 메모리 효율성 개선

2. **적응형 ρ 자동 선택**
   - 데이터셋별 ρ 최적값 자동 결정 메커니즘
   - 학습 과정 중 동적 ρ 조정

3. **하이브리드 방식**
   - SAM과 다른 정규화 기법(mixup, cutmix 등)의 상호작용 분석
   - MentorSAM 등 결합 방식의 체계적 개발

#### 7.3 실무적 적용 시 주의사항

1. **배치 크기의 영향**
   - m-sharpness 개념에 따르면, 작은 배치에서의 학습이 더 좋은 일반화 제공
   - 계산 자원과 일반화 성능의 트레이드오프 고려 필요

2. **다양한 도메인에서의 검증 필요**
   - 현재 대부분의 연구는 이미지 분류에 집중
   - NLP, 강화학습, 추천 시스템 등에서의 효과성 검증 필요

3. **프로덕션 환경에서의 고려사항**
   - 실시간 추론 성능: 학습 시간 증가가 배포 후 추론에는 영향 없음
   - 모델 저장소: SAM으로 학습한 모델은 일반적인 가중치 형태로 저장 가능
   - 전이 학습: SAM으로 학습한 사전학습 모델의 품질 우수성 확인

#### 7.4 학제 간 융합 방향

1. **통계학과의 연계**
   - PAC-Bayesian 틀과 빈도주의적 일반화 경계의 통합
   - 최적 수렴률(optimal convergence rate) 달성 가능성

2. **수학적 최적화 관점**
   - min-max 문제의 더 정교한 해석
   - 이중 경시 강화(bilevel optimization) 이론의 적용

3. **신경과학과의 연관성**
   - 뇌의 학습 과정이 손실 곡면의 평탄한 영역을 선호하는 메커니즘
   - 진화적 최적화의 관점

***

### 8. 결론 및 임팩트 평가
Sharpness-Aware Minimization은 손실 지형의 기하학과 일반화의 연결을 실용적으로 구현한 획기적 연구입니다. 이 논문 이후 수십 개의 확장 연구가 생성되었으며, 각각이 특정 도메인 또는 문제 설정에서 추가적인 개선을 달성했습니다.

**핵심 성과:**
- 광범위한 벤치마크에서 일관된 0.3~2.5%의 오류율 감소
- 이미 정규화가 적용된 모델에서도 추가 개선 달성
- 이론적 일반화 경계의 제공
- 단순하고 구현 용이한 알고리즘

**남아있는 도전:**
- 계산 효율성 (2배의 비용)
- 하이퍼파라미터 민감성 (ρ 선택)
- 이론과 실제 성능 간의 격차

**향후 전망:**
SAM의 성공은 손실 지형 관점의 타당성을 입증했으며, 향후 연구는 (1) 더 효율적인 구현, (2) 더 정교한 이론적 이해, (3) 더 넓은 적용 도메인 확대로 진행될 것으로 예상됩니다. 특히 대규모 언어 모델과 멀티모달 모델에서의 SAM 적용이 미개척 영역입니다.

***

### 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b408df71-d606-4808-82d4-685a89f32833/2010.01412v3.pdf)
[2](https://arxiv.org/abs/2110.03141)
[3](https://arxiv.org/pdf/2210.12864.pdf)
[4](https://arxiv.org/abs/2503.02225)
[5](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Friendly_Sharpness-Aware_Minimization_CVPR_2024_paper.pdf)
[6](https://arxiv.org/abs/2501.12666)
[7](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13993/3092692/Hyperbolic-SAM--sharpness-aware-minimization-in-hyperbolic-space-for/10.1117/12.3092692.full)
[8](https://arxiv.org/abs/2502.08448)
[9](https://arxiv.org/abs/2503.23430)
[10](https://arxiv.org/abs/2510.24919)
[11](https://arxiv.org/abs/2505.01660)
[12](https://www.nature.com/articles/s41598-025-85679-8)
[13](https://pmc.ncbi.nlm.nih.gov/articles/PMC12599882/)
[14](http://arxiv.org/pdf/2312.13555.pdf)
[15](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13307/3049180/Sharpness-aware-minimization-SAM-improves-generalization-performance-of-bacterial-Raman/10.1117/12.3049180.full)
[16](https://pmc.ncbi.nlm.nih.gov/articles/PMC12121992/)
[17](https://www.semanticscholar.org/paper/a2cd073b57be744533152202989228cb4122270a)
[18](https://arxiv.org/pdf/2212.04343.pdf)
[19](https://arxiv.org/pdf/2501.11584.pdf)
[20](http://arxiv.org/pdf/2110.03141.pdf)
[21](https://arxiv.org/pdf/2305.15817.pdf)
[22](http://arxiv.org/pdf/2403.12350.pdf)
[23](https://arxiv.org/pdf/2303.00565.pdf)
[24](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Flatness-Aware_Minimization_for_Domain_Generalization_ICCV_2023_paper.pdf)
[25](https://liner.com/review/improving-generalization-universal-adversarial-perturbation-via-dynamic-maximin-optimization)
[26](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Seeking_Consistent_Flat_Minima_for_Better_Domain_Generalization_via_Refining_CVPR_2025_paper.pdf)
[27](https://arxiv.org/abs/2503.12793)
[28](https://arxiv.org/abs/2403.12350)
[29](https://arxiv.org/html/2511.04808v1)
[30](https://arxiv.org/html/2503.12793v3)
[31](https://arxiv.org/abs/2010.01412)
[32](https://pdfs.semanticscholar.org/ee54/53052a78ca394d0cfd40fc9f0ab7ee0a9b4b.pdf)
[33](https://arxiv.org/pdf/2501.13864.pdf)
[34](https://pdfs.semanticscholar.org/42d6/45ed82a3fa6206d1ae119acd09f9ef031834.pdf)
[35](https://arxiv.org/html/2501.13864v1)
[36](https://arxiv.org/pdf/2302.05185.pdf)
[37](https://www.arxiv.org/pdf/2511.10714.pdf)
