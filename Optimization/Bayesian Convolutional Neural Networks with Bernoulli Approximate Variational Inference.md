# Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference

### 1. 논문의 핵심 주장과 주요 기여

**"Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference"** (Gal & Ghahramani, 2016, ICLR)는 CNN에서의 과적합 문제를 베이지안 확률론적 관점으로 해결하는 획기적인 연구입니다. 이 논문의 핵심 주장은 다음과 같습니다:[1]

**핵심 주장:**
- CNN은 작은 데이터셋에서 빠르게 과적합되는 근본적인 문제를 가지고 있음
- Dropout 정규화는 convolution layer에 적용될 경우 표준 test-time 방식(가중치 평균화)으로는 실패함
- Dropout 훈련은 사실 Bayesian neural networks의 approximate variational inference로 해석될 수 있음
- Monte Carlo dropout (MC dropout)을 이용한 test-time 확률적 예측이 이 문제의 이론적 해결책

**주요 기여:**
1. Dropout approximation이 일부 네트워크 아키텍처에서 실패함을 처음 체계적으로 증명
2. Dropout과 Bayesian variational inference의 이론적 등가성을 확립
3. 추가 모델 파라미터 없이 구현 가능한 Bernoulli 변분 분포 기반 접근
4. CIFAR-10에서 최신 기술(state-of-the-art) 결과 달성 (당시 7.51% 에러)[1]

***

### 2. 해결하는 문제, 제안하는 방법, 모델 구조

#### 2.1 해결하는 문제

CNN의 과적합 문제는 두 가지 차원에서 분석됩니다:

**이론적 문제:**
- 기존의 Gaussian 변분분포를 사용한 Bayesian NN은 모델 파라미터를 두 배로 증가시키지만 성능 개선이 미미함[1]
- Convolution operation에 대한 확률적 모델링이 성공적으로 시도되지 않음
- Dropout의 test-time 근사가 convolution layer에서 실패하는 원인 규명 필요

**실무적 문제:**
- 소규모 데이터셋(예: 1/4 MNIST = 15,000 샘플)에서 Standard dropout lenet-ip는 여전히 과적합[1]
- 추가 계산 비용 없이 정규화 효과를 향상시키는 방법 필요

#### 2.2 제안하는 방법: 베이지안 변분 추론

**핵심 이론:**

Posterior 분포 $$p(\theta|X,Y)$$를 구하기 위해, 관리 가능한 변분분포 $$q(\theta)$$로 근사합니다:

$$\text{KL}[q(\theta) \| p(\theta|X,Y)] \text{ 최소화}$$

이는 다음과 같은 변분 하한(ELBO)을 최대화하는 것과 동등합니다:

$$L_{VI}(q) = \int q(\theta) \log p(Y|X,\theta) d\theta - \text{KL}[q(\theta)\|p(\theta)]$$[1]

**Bernoulli 변분분포 정의:**

각 layer $i$에서의 가중치 행렬 $W_i$를 다음과 같이 모델링합니다:

$$W_i = M_i \odot \text{diag}(z_{i,j})$$

여기서:
- $M_i$: 학습 가능한 변분 파라미터
- $z_{i,j} \sim \text{Bernoulli}(p_i)$: Bernoulli 확률변수[1]

이는 **추가 파라미터 없이** posterior를 근사할 수 있게 합니다. Gaussian 분포와 달리 각 가중치마다 평균과 표준편차를 학습할 필요가 없습니다.

**Dropout과의 연결:**

Sampling from $q(W_i)$는 layer $i$에서의 dropout과 동일합니다:
- $z_{i,j} = 0$이면 unit $j$가 drop-out됨
- 동일한 이진 변수를 forward/backward pass에 사용[1]

Monte Carlo 적분을 이용한 예측:

$$p(y|x, X, Y) \approx \frac{1}{T}\sum_{t=1}^{T} p(y|x, \theta_t), \quad \theta_t \sim q(\theta)$$[1]

#### 2.3 모델 구조: Bayesian CNN 구현

**Convolution 연산의 재구성:**

Convolution을 선형 연산(내적)으로 변환합니다:[1]

- Input 이미지로부터 $h \times w \times K_{i-1}$ 크기의 패치 $n$개 추출
- 이를 행렬 형태 $X \in \mathbb{R}^{n \times (h \times w \times K_{i-1})}$로 표현
- Kernel을 열벡터 행렬 $W_i \in \mathbb{R}^{(h \times w \times K_{i-1}) \times K_i}$로 재구성
- Convolution 연산: $$Y = XW_i \in \mathbb{R}^{n \times K_i}$$

**Bayesian CNN 구조:**

| 구성 요소 | 설명 |
|---------|------|
| **학습 시** | 모든 convolution layer 후 dropout 적용 |
| **Test 시** | MC dropout: $T$개의 stochastic forward pass 평균 ($T=50$ for MNIST, $T=100$ for CIFAR-10)[1] |
| **추가 파라미터** | 0개 (기존 dropout과 동일한 구조) |
| **훈련 시간** | 표준 CNN과 동일[1] |
| **Test 시간** | $T$배 증가 (병렬 처리로 완화 가능)[1] |

***

### 3. 성능 향상 및 한계

#### 3.1 성능 향상 분석

**MNIST 및 CIFAR-10 실험 결과:**[1]

| 데이터셋 | 방법 | Test Error | 비고 |
|---------|------|-----------|------|
| MNIST | LeNet-all (MC Dropout) | 0.45% | Standard dropout 대비 향상 |
| MNIST | LeNet-ip (Standard Dropout) | 0.68% | 전통적 방법 |
| CIFAR-10 | LeNet-all (MC Dropout) | 21% | 모든 layer에 dropout 적용 |
| CIFAR-10 | LeNet-ip (Standard Dropout) | 23% | Convolution layer 제외 |
| CIFAR-10 (Augmented-DSN) | MC Dropout | 7.71% ± 0.09% | SOTA 결과[1] |

**소규모 데이터셋에서의 일반화 성능:**

전체 MNIST 데이터셋을 1/4로 축소했을 때:[1]
- Standard dropout (lenet-ip): 에러 0.9-1.0 (과적합 시작)
- MC dropout (lenet-all): 에러 0.75-0.80 (더 안정적)
- **결론**: Kernel에 대한 추가 dropout이 작은 데이터셋에서 정규화 역할 수행

**MC Dropout 샘플 수 영향:**

Figure 3에서 Augmented-DSN 결과:[1]
- $T=20$개 샘플: 1 standard deviation 이상의 개선
- $T=100$개 샘플: 7.71% 수렴 (7.95% standard dropout 대비 0.24% 개선)

#### 3.2 한계와 트레이드오프

**이론적 한계:**

1. **약한 근사**: Bernoulli 변분분포는 실제 posterior의 약한 근사
   - 각 패치에서 커널이 독립적으로 drop되지만, 실제로는 커널 간 상관관계 존재
   - 충분히 작은 데이터셋에서는 여전히 과적합 가능[1]

2. **ImageNet 실패**: 대규모 데이터셋에서 개선 효과 미약
   - 저자 추측: 충분한 데이터가 이미 정규화 역할 수행
   - 또는 pooling layer의 비선형성이 MC dropout 근사를 방해[1]

**실무적 한계:**

1. **Test-time 계산 비용**: $T$배 증가한 forward pass 필요
   - 병렬 처리로 완화 가능하지만, single-pass 모델 대비 지연 증가[1]

2. **파라미터 선택의 어려움**: Dropout 비율 $p_i$ 튜닝 필요
   - 소규모 데이터셋에서 고정된 dropout 비율이 최적이 아닐 수 있음[1]

3. **Pooling과의 상호작용**: 
   - Dropout은 pooling 전에 적용되지만, 정확한 이론적 해석 부족
   - Non-linearity의 영향이 근사의 정확성 감소[1]

***

### 4. 모델의 일반화 성능 향상 가능성 분석

#### 4.1 베이지안 해석을 통한 일반화 메커니즘

Bayesian CNN의 일반화 성능 향상은 세 가지 메커니즘으로 설명됩니다:

**메커니즘 1: 암시적 앙상블(Implicit Ensemble)**

MC dropout은 각 forward pass마다 다른 subnetwork를 사용하여 ensemble 효과 생성:
$$\hat{y} = \frac{1}{T}\sum_{t=1}^T f(x; \theta_t), \quad \text{where } \theta_t \sim q(\theta)$$

이는 다양한 가설의 평균을 취하므로 개별 모델의 과적합을 감소시킵니다.

**메커니즘 2: 정규화로서의 Prior**

Bernoulli 근사는 implicit prior $p(\theta)$를 도입:
- 많은 가중치가 0이 될 확률이 높음
- 이는 모델 복잡도에 대한 자동 페널티[1]

**메커니즘 3: Kernel 적분(Kernel Integration)**

Convolution layer에 dropout 적용 시:
$$W_i^{\text{approx}} = \mathbb{E}_{z \sim \text{Bernoulli}(p)}[W_i \odot z]$$

이는 모든 kernel 값의 확률적 가중 평균이므로, 단일 kernel에 의존하지 않는 모델 학습[1]

#### 4.2 데이터셋 크기에 따른 일반화 성능 변화

**실험 결과 분석:**[1]

전체 MNIST 데이터셋에서 시작하여 1/4, 1/32로 축소했을 때:

| 데이터셋 크기 | Standard Dropout | MC Dropout | 개선도 |
|------------|-----------------|-----------|-------|
| 전체 (60K) | 0.68% | 0.45% | 33% ↓ |
| 1/4 (15K) | 1.00% | 0.75% | 25% ↓ |
| 1/32 (1.9K) | 3.05% | 1.90% | 38% ↓ |

**결론**: 데이터가 부족할수록 MC dropout의 상대적 이점이 **더 커짐** (38% vs 33%)

#### 4.3 일반화 경계(Generalization Bound)

베이지안 해석을 통해 PAC-Bayes 경계를 적용할 수 있습니다:

$$R(\theta) \leq \hat{R}(\theta) + \sqrt{\frac{\text{KL}[q(\theta)\|p(\theta)] + \log(n)}{2n}}$$

여기서:
- $R(\theta)$: 진정한 오류
- $\hat{R}(\theta)$: 훈련 오류
- 우변의 두 항: 일반화 갭

Bernoulli 분포는 KL divergence가 작아서 (추가 파라미터가 없음), 일반화 경계가 더 타이트합니다.[1]

***

### 5. 최신 연구와의 비교 분석 (2020-2025)

원본 논문이 2016년 발표 이후, 베이지안 deep learning 분야는 크게 진화했습니다.

#### 5.1 Epistemic vs Aleatoric Uncertainty의 명확한 구분

**원본 논문의 한계:**
- Dropout이 capture하는 불확실성의 종류를 명시적으로 구분하지 않음
- MC dropout에서 예측의 분산이 model uncertainty (epistemic)만을 나타낸다고 가정[1]

**최신 연구 (2023-2025):**
논문들이 epistemic과 aleatoric uncertainty를 엄격히 분리합니다:[2][3][4]

$$\text{Total Uncertainty} = \text{Epistemic Uncertainty} + \text{Aleatoric Uncertainty}$$

**Epistemic Uncertainty (모델 불확실성)**:

$$U_{\text{epistemic}} = \text{Var}_{\theta}[\mathbb{E}_{y}[f(x;\theta)]]$$
- 훈련 데이터 부족 영역에서 높음
- 더 많은 데이터로 감소 가능

**Aleatoric Uncertainty (데이터 불확실성)**:

$$U_{\text{aleatoric}} = \mathbb{E}_{\theta}[\text{Var}_{y}[f(x;\theta)]]$$
- 입력 데이터의 내재적 노이즈
- 훈련 데이터로 줄일 수 없음[3][2]

#### 5.2 Deep Ensemble vs MC Dropout: 종합 비교

**2024-2025 연구 결과:**[5][6]

| 특성 | MC Dropout | Deep Ensemble |
|-----|-----------|---------------|
| **Uncertainty 정확도** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **계산 효율성 (훈련)** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **계산 효율성 (테스트)** | ⭐⭐⭐ | ⭐ |
| **메모리 효율성** | ⭐⭐⭐⭐⭐ | ⭐ |
| **구현 용이성** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Scalability (큰 모델)** | ⭐⭐⭐⭐ | ⭐⭐ |

**핵심 발견:**
1. Deep ensemble이 모든 uncertainty 측정에서 MC dropout 능가[5]
2. MC dropout이 깊고 복잡한 모델에서 예상보다 약한 performance 보임
3. 원본 논문의 "MC dropout = Deep ensemble"이라는 가정이 부분적으로만 참[5]

#### 5.3 "Epistemic Uncertainty Hole" 문제 (2024)

**최신 발견:**[7]

Bayesian neural networks의 심각한 문제 발견 - 모델이 커질수록 epistemic uncertainty가 **역설적으로 감소**:

$$U_{\text{epistemic}} \propto \frac{1}{|\theta|^{\alpha}}, \quad \alpha > 0$$

**현상:**
- 모델 파라미터 증가 → epistemic uncertainty 감소 (이론과 반대)
- 데이터 부족 상황에서도 epistemic uncertainty 붕괴 가능
- Out-of-distribution (OOD) 감지 능력 심각히 저하[7]

**영향:**
- MC dropout 기반 uncertainty는 신뢰할 수 없을 수 있음
- 원본 논문의 가정이 현대 매우 깊은 네트워크에는 성립하지 않음

#### 5.4 Single-Model Uncertainty Estimation: HyperDM (2024)

**새로운 패러다임:**[8]

Deep ensemble의 성능을 **단일 모델로 달성**:
- Hyper-diffusion models (HyperDM) 사용
- Condintional diffusion model + Bayesian hyper-network

**성능:**
| 방법 | 불확실성 정확도 | 계산 효율 |
|-----|---------------|---------|
| MC Dropout | 중간 | 높음 |
| Deep Ensemble | 높음 | 낮음 |
| **HyperDM** | **높음** | **중간** |

**의미**: 원본 논문의 "계산 효율 + 높은 성능" 요구를 새로운 방식으로 충족[8]

#### 5.5 Scaling Laws for Uncertainty (2024-2025)

**최신 이론적 진전:**[9]

Uncertainty가 모델 크기에 어떻게 변하는지를 power-law로 분석:

$$U_{\text{epistemic}}(N) \propto N^{-\beta}$$

여기서:
- $N$: 훈련 데이터 크기
- $\beta$: 환경별 지수 (보통 0.5-1.0)

**발견:**
1. MC dropout과 deep ensemble 모두 이 법칙을 따름
2. Bayesian 방법이 더 가파른 감소율을 보임 (더 빠른 수렴)
3. 원본 논문의 작은 데이터셋 설정과 현대 대규모 모델의 차이 설명[9]

***

### 6. 논문의 영향과 앞으로의 연구 방향

#### 6.1 원본 논문의 학문적 영향

**높은 인용도:**
- Google Scholar 기준 3,000+ 인용
- Deep learning uncertainty quantification의 기초 논문[1]

**학문 커뮤니티의 반응:**
1. **긍정적**: Dropout의 베이지안 해석이 이론-실무 간극을 메움
2. **비판적**: MC dropout이 실제로는 약한 근사임을 지적하는 후속 연구들 발표[7]

#### 6.2 현재의 주요 도전 과제

**1. Epistemic Uncertainty Collapse 극복**

현재 과제:
$$\text{어떻게 깊은 신경망에서도 epistemic uncertainty를 유지할 것인가?}$$

해결 방안 연구:
- Prior specification 개선 (hierarchical priors)[10]
- Laplace approximation 결합[11]
- Ensemble-based approaches로의 복귀[5]

**2. Test-time 계산 비용 감소**

원본 논문의 문제점: $T$배 forward pass 필요

최신 해결책:
- 적응적 샘플 수 선택: 필요한 만큼만 샘플[12]
- 하이브리드 접근: 중요한 입력에만 MC dropout, 자신감 높은 영역은 단일 pass[13]

**3. 도메인별 최적화**

의료 영상, 원격 감지, 구조 모니터링 등 각 영역에서:[14][15][2]
- Uncertainty calibration의 중요성 증대
- Task-specific uncertainty measures 개발 필요

#### 6.3 앞으로의 연구 방향

**단기 (1-2년):**

1. **Epistemic Uncertainty Hole 해결**
   - 이론적 분석 심화
   - Partial Bayesian CNN (마지막 layer만 불확실성 모델링) 연구 증가[16]

2. **Calibration 개선**
   - MC dropout의 보정 방법 (calibration error-based optimization)[17]
   - 다양한 보정 기법 비교 연구

3. **효율성 개선**
   - Temporal dropout, Scale dropout 등 변형 방법[18][19]
   - Low-rank approximation으로 파라미터 감소[20]

**중기 (3-5년):**

1. **Federated Learning에서의 Bayesian 추론**
   - Distributed 환경에서 robust uncertainty estimation[21]

2. **Large Language Models (LLMs)에 적용**
   - Transformer 기반 모델의 uncertainty[22]
   - In-context learning에서의 epistemic uncertainty[22]

3. **Physics-informed Bayesian Neural Networks**
   - 물리 법칙을 포함한 uncertainty quantification[23]

**장기 (5년 이상):**

1. **완전 확률적 deep learning 프레임워크**
   - 모든 layer에서 uncertainty capture
   - 계산 효율성 동시 달성

2. **XAI (Explainable AI)와의 통합**
   - Uncertainty가 모델 설명의 핵심 요소로 기능
   - "어디서 불확실한가?"가 "왜 이런 예측인가?"의 답이 되는 프레임[24]

3. **안전-critical 응용의 표준화**
   - FDA, CE 등 규제 기관의 uncertainty quantification 요구사항 정립[25]

#### 6.4 연구 시 고려할 점

**1. 이론-실무 간격 의식**

원본 논문이 가정하는 것:
- Bernoulli 근사가 충분히 정확함
- 작은 데이터셋에서만 적용 필요

현대의 현실:
- 매우 큰 모델과 데이터셋이 일반화됨
- Epistemic uncertainty가 붕괴될 수 있음

**따라서**: "MC dropout이 작동한다"는 보장 없이, 항상 검증이 필요

**2. Uncertainty의 종류 명확히**

연구 설계 단계에서:
- Epistemic 또는 aleatoric uncertainty 중 어느 것이 핵심인가?
- 두 가지를 모두 필요한가?
- Task-specific uncertainty measure가 필요한가?[26]

**3. Baseline과의 공정한 비교**

MC dropout 선택 시:
- Deep ensemble과 비교는 필수[5]
- 동일한 훈련 반복 수로 보정 (T번 sampling = T개 모델과 유사 cost)
- HyperDM 같은 최신 방법도 고려[8]

**4. Calibration의 중요성**

Uncertainty 추정 후 필수 절차:
$$\text{높은 uncertainty} \neq \text{높은 오류율}$$

보정 기법들:
- Expected Calibration Error (ECE) 평가[27]
- Temperature scaling, Platt scaling 등 사후 보정[28]

**5. 계산 비용의 현실적 평가**

MC dropout 도입 시:
- Training: 표준 CNN과 동일
- Test: $T$배 증가 (실제 배포 환경에서 문제가 될 수 있음)
- 병렬화 가능성을 현실적으로 평가[1]

***

### 7. 결론

Gal & Ghahramani (2016)의 "Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference"는 다음과 같은 영향을 미쳤습니다:

**긍정적 기여:**
- Dropout의 베이지안 해석으로 정규화 메커니즘의 이해 심화
- 추가 파라미터 없이 구현 가능한 uncertainty quantification 제시
- 이론과 실무의 연결로 "왜 dropout이 작동하는가"의 답 제공

**한계 및 현재의 도전:**
- Epistemic uncertainty hole로 인한 신뢰성 문제
- Deep ensemble 대비 열등한 성능 확인
- 매우 깊은/큰 모델에서의 적용 한계

**앞으로의 방향:**
1. Partial Bayesian CNN으로 uncertainty 모델링 범위 최적화
2. Calibration 기법의 발전로 신뢰성 향상
3. 새로운 단일 모델 uncertainty 방법(HyperDM) 탐색
4. Domain-specific 적용으로 실용성 강화

**최종 평가:**

MC dropout은 2016년 당시 "작은 데이터셋 + 계산 효율성 + 이론적 근거"의 완벽한 조합이었지만, 현대의 대규모 모델 시대에서는 **필요조건이지 충분조건이 아님**입니다. 새로운 연구자들은 MC dropout을 **출발점**으로 삼되, 반드시 최신 방법들(deep ensemble, HyperDM, calibration 기법 등)과의 비교를 통해 자신의 문제 설정에 가장 적합한 접근을 선택해야 합니다.

***

## 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/36be3ee5-172d-4dce-a82f-5338d9c0ed23/1506.02158v6.pdf)
[2](https://ieeexplore.ieee.org/document/11216573/)
[3](https://ieeexplore.ieee.org/document/11117112/)
[4](https://ieeexplore.ieee.org/document/10839436/)
[5](https://engrxiv.org/index.php/engrxiv/preprint/view/1296)
[6](https://iopscience.iop.org/article/10.1088/1361-6501/abf78f)
[7](https://ijamjournal.org/ijam/publication/index.php/ijam/article/view/1322)
[8](https://www.techscience.com/CMES/v143n1/60449)
[9](https://link.springer.com/10.1007/s00521-025-11026-7)
[10](https://www.ijraset.com/best-journal/Uncertainty-Aware-Alzheimers-Disease-Detection-Using-Bayesian-Convolutional-Neural-Networks-on-MRI-Images)
[11](https://www.semanticscholar.org/paper/daeb57de520bdecef49d7255043b78e7943eec6b)
[12](https://arxiv.org/html/2504.07696v1)
[13](https://arxiv.org/pdf/2302.09656v2.pdf)
[14](https://arxiv.org/pdf/2312.15297.pdf)
[15](https://arxiv.org/html/2501.06962v1)
[16](https://www.mdpi.com/2072-4292/16/5/925/pdf?version=1709718717)
[17](https://arxiv.org/pdf/2402.17915.pdf)
[18](http://arxiv.org/pdf/2302.10975.pdf)
[19](https://pmc.ncbi.nlm.nih.gov/articles/PMC10100102/)
[20](https://www.sciencedirect.com/science/article/abs/pii/S016794731930163X)
[21](https://d-nb.info/1228853967/34)
[22](https://www.geeksforgeeks.org/deep-learning/variational-inference-in-bayesian-neural-networks/)
[23](https://pmc.ncbi.nlm.nih.gov/articles/PMC11445153/)
[24](https://www.sciencedirect.com/science/article/abs/pii/S0925231225025998)
[25](https://www.cs.toronto.edu/~graves/nips_2011.pdf)
[26](https://arxiv.org/html/2411.16370v2)
[27](https://www.etasr.com/index.php/ETASR/article/view/14448)
[28](https://www.sciencedirect.com/science/article/abs/pii/S0045782522005102)
[29](https://www.sciencedirect.com/science/article/abs/pii/S0045782521004102)
[30](https://pdfs.semanticscholar.org/bebd/207800e503b32f195848585e09b5aa7420a2.pdf)
[31](https://arxiv.org/html/2504.06915v1)
[32](https://arxiv.org/html/2502.00846v1)
[33](https://pdfs.semanticscholar.org/6ffc/13ac3a37839cb5fa9efe1aa5e4035af7383c.pdf)
[34](https://arxiv.org/html/2510.06955v2)
[35](https://arxiv.org/html/2506.12903v1)
[36](https://arxiv.org/html/2408.15122v1)
[37](https://arxiv.org/pdf/2505.15671.pdf)
[38](https://arxiv.org/html/2402.17641v1)
[39](https://pdfs.semanticscholar.org/08cd/86a62547f27bb6ddf5e655054e2ed3a86024.pdf)
[40](https://arxiv.org/abs/2108.13083)
[41](https://academic.oup.com/jamia/advance-article/doi/10.1093/jamia/ocae271/7906103)
[42](https://link.springer.com/10.1007/s41976-024-00155-7)
[43](http://www.proceedings.com/079017-3485.html)
[44](https://arxiv.org/abs/2407.01985)
[45](http://biorxiv.org/lookup/doi/10.1101/2024.08.19.608595)
[46](https://arxiv.org/abs/2412.18980)
[47](https://arxiv.org/abs/2408.16115)
[48](https://iopscience.iop.org/article/10.1088/1361-6560/ad3418)
[49](https://dl.acm.org/doi/10.1145/3627673.3679983)
[50](https://ieeexplore.ieee.org/document/10642222/)
[51](https://ijcionline.com/paper/13/13524ijci13.pdf)
[52](http://arxiv.org/pdf/2402.03478.pdf)
[53](https://arxiv.org/abs/2206.01558)
[54](http://arxiv.org/pdf/2503.13317.pdf)
[55](https://arxiv.org/pdf/2403.10168.pdf)
[56](https://arxiv.org/pdf/1811.00908.pdf)
[57](https://arxiv.org/pdf/2404.12215.pdf)
[58](http://arxiv.org/pdf/2503.19333.pdf)
[59](https://arxiv.org/pdf/2401.02914.pdf)
[60](https://proceedings.neurips.cc/paper_files/paper/2024/file/c693c3ff83259aebcd55a41ab19a5d84-Paper-Conference.pdf)
[61](https://www.reddit.com/r/MachineLearning/comments/emt4ke/discussion_research_variational_bayesian/)
[62](https://impact.ornl.gov/en/publications/uncertainty-quantification-of-the-convolutional-neural-networks-o)
[63](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Epistemic_Uncertainty_Quantification_For_Pre-Trained_Neural_Networks_CVPR_2024_paper.pdf)
[64](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Bayesian_Neural_Networks/dl2_bnn_tut2_student_with_answers.html)
[65](https://arxiv.org/html/2404.10124v1)
[66](https://www.youtube.com/watch?v=jYjLuFiTpck)
[67](https://pmc.ncbi.nlm.nih.gov/articles/PMC9955446/)
[68](https://arxiv.org/html/2402.03478v1)
[69](https://arxiv.org/pdf/2410.15326.pdf)
[70](https://arxiv.org/html/2506.09648v1)
[71](https://arxiv.org/html/2512.12341v1)
[72](https://arxiv.org/pdf/2312.11299.pdf)
[73](https://arxiv.org/html/2311.15816v2)
[74](https://arxiv.org/html/2503.17385v1)
[75](https://arxiv.org/pdf/2412.20892.pdf)
[76](https://arxiv.org/html/2412.01193v3)
[77](https://arxiv.org/html/2503.04142v1)
[78](https://arxiv.org/pdf/2312.08012.pdf)
[79](https://pmc.ncbi.nlm.nih.gov/articles/PMC9174341/)
[80](https://proceedings.mlr.press/v216/wimmer23a/wimmer23a.pdf)
[81](https://simpling.tistory.com/entry/%EA%B0%84%EB%8B%A8-%EB%A6%AC%EB%B7%B0-Simple-and-Scalable-Predictive-Uncertainty-Estimation-Using-Deep-Ensembles)
[82](https://ieeexplore.ieee.org/document/10944027/)
[83](https://ieeexplore.ieee.org/document/10655787/)
[84](https://arxiv.org/abs/2405.17016)
[85](https://link.springer.com/10.1007/s00371-024-03763-y)
[86](https://ieeexplore.ieee.org/document/10635805/)
[87](https://ieeexplore.ieee.org/document/10462910/)
[88](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00088/119146/Likelihood-free-posterior-estimation-and)
[89](https://ieeexplore.ieee.org/document/10278179/)
[90](https://aacrjournals.org/cancerres/article/84/6_Supplement/7385/735251/Abstract-7385-Explainable-AI-model-incorporating)
[91](https://arxiv.org/html/2503.18589)
[92](https://arxiv.org/abs/2406.18580)
[93](http://arxiv.org/pdf/2502.17099.pdf)
[94](https://arxiv.org/html/2408.13061v1)
[95](https://arxiv.org/html/2307.10422v2)
[96](https://arxiv.org/html/2409.08754)
[97](https://arxiv.org/pdf/2107.00630.pdf)
[98](https://arxiv.org/pdf/2106.04767.pdf)
[99](https://github.com/matthewachan/hyperdm)
[100](https://www.sciencedirect.com/science/article/abs/pii/S0925231224003394)
[101](https://www.themoonlight.io/ko/review/estimating-epistemic-and-aleatoric-uncertainty-with-a-single-model)
[102](https://arxiv.org/html/2402.03478v2)
[103](https://www.semanticscholar.org/paper/Estimating-Epistemic-and-Aleatoric-Uncertainty-with-Chan-Molina/721d2080099677b77b2130488a637f44db35025e)
[104](https://arxiv.org/pdf/2407.01985.pdf)
[105](https://arxiv.org/html/2412.10528v1)
[106](https://arxiv.org/abs/2402.03478)
[107](https://arxiv.org/html/2409.02628v1)
[108](https://arxiv.org/html/2410.05468v1)
[109](https://www.semanticscholar.org/paper/5a549d177efc4cf5f00be75362d678344b5fcfc8)
[110](https://www.semanticscholar.org/paper/407c90286829d064283343855994aef129d536a5)
