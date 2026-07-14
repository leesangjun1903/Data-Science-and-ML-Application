# Stochastic Classifiers for Unsupervised Domain Adaptation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 Local Alignment(LA) 기반 UDA 방법들은 **2개의 분류기**만 사용하지만, 더 많은 분류기를 사용할수록 성능이 향상된다는 것을 발견하였다. 그러나 단순히 분류기 수를 늘리면 파라미터 수 증가 및 과적합 위험이 발생한다. 이를 해결하기 위해 **분류기 가중치를 확률 분포로 모델링**하는 STAR(STochastic clAssifieRs)를 제안한다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **(1)** 문제 발굴 | LA 기반 UDA에서 분류기 수의 중요성을 최초로 체계적으로 분석 |
| **(2)** 방법 제안 | 파라미터 수 증가 없이 무한에 가까운 분류기를 활용하는 STAR 프레임워크 제안 |
| **(3)** 성능 검증 | 이미지 분류 및 시맨틱 세그멘테이션에서 다양한 LA 방법에 STAR 적용 시 SOTA 달성 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**문제 1: 최적 분류기 수 불명확**

기존 LA 방법(MCD, CLAN 등)은 관행적으로 2개의 분류기를 사용하지만, 아래 실험 결과와 같이 최적 분류기 수는 태스크에 따라 다르며 일반적으로 2개보다 많을 때 성능이 높다.

**문제 2: 단순 분류기 추가의 한계**
- 분류기 수 $K$에 대해 쌍별 불일치 계산 복잡도: $O(K^2)$
- 파라미터 수 선형 증가 → 과적합 위험
- Table 1에서 확인: MCD의 SVHN→MNIST 태스크에서 분류기($C$) 파라미터(12.6M)가 특징 추출기($G$) 파라미터(25.5M)에 육박

---

### 2-2. 제안 방법 (수식 포함)

#### 핵심 아이디어: 분류기를 분포로 모델링

기존 결정론적 분류기:
$$\phi_1, \phi_2 \in \mathbb{R}^d \quad \text{(고정된 가중치 벡터)}$$

STAR의 확률적 분류기:
$$\phi \sim \mathcal{N}(\mu, \Sigma)$$

여기서:
- $\mu$: 분포의 평균 벡터 (최종 추론 시 사용되는 분류기 가중치)
- $\Sigma$: 대각 공분산 행렬 (분류기 간 불일치 정도를 표현)
- 학습 가능한 파라미터: $\mu$와 $\Sigma$의 대각 원소 $\sigma$

**파라미터 수 비교:**
- 기존 2-분류기: $2d$개 파라미터
- STAR: $\mu \in \mathbb{R}^d$ + $\sigma \in \mathbb{R}^d$ = $2d$개 파라미터 → **동일!**

#### Reparameterization Trick (역전파 가능화)

샘플링은 미분 불가능하므로 재매개변수화 기법 적용:

$$\tilde{\phi}_1 = \mu + \sigma \odot \epsilon_1, \quad \tilde{\phi}_2 = \mu + \sigma \odot \epsilon_2$$

여기서:
- $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, I)$: 표준 정규분포에서 독립 샘플
- $\odot$: 원소별 곱(element-wise product)
- $\sigma$: $\Sigma$의 대각 원소

이를 통해 $\mu$와 $\sigma$에 대한 기울기를 계산 가능하다.

---

### 2-3. 모델 구조

#### MCD + STAR (이미지 분류)

기존 MCD의 3단계 최적화:

**Step A** (특징 추출기 + 분류기 학습, 소스 도메인 분류):
$$\min_{\theta, \phi_1, \phi_2} \ell(f_{\phi_1}(g_\theta(x_S)), y_S) + \ell(f_{\phi_2}(g_\theta(x_S)), y_S)$$

**Step B** (분류기 불일치 최대화, 타겟 도메인):
$$\max_{\phi_1, \phi_2} \|f_{\phi_1}(g_\theta(x_T)) - f_{\phi_2}(g_\theta(x_T))\|_1$$

**Step C** (특징 추출기 최적화, 불일치 최소화):
$$\min_{\theta} \|f_{\phi_1}(g_\theta(x_T)) - f_{\phi_2}(g_\theta(x_T))\|_1$$

STAR 적용 시: $\{\phi_1, \phi_2\}$를 $\{\tilde{\phi}_1, \tilde{\phi}_2\}$로 교체 (샘플링된 가중치)

#### CLAN + STAR (시맨틱 세그멘테이션)

CLAN의 적대적 손실:

$$\ell^{(A)}_{\theta,\phi_1,\phi_2,\psi}(x_S, x_T) = -\log(h_\psi(f_{\phi_1}(g_\theta(x_S)))) - \log(h_\psi(f_{\phi_2}(g_\theta(x_S)))) -\rho\log(1 - h_\psi(f_{\phi_1}(g_\theta(x_T)))) -\rho\log(1 - h_\psi(f_{\phi_2}(g_\theta(x_T))))$$

$$\min_{\psi} \max_{\theta, \phi_1, \phi_2} \ell^{(A)}_{\theta,\phi_1,\phi_2,\psi}(x_S, x_T)$$

가중치 불일치 손실 (STAR 적용 시 $\Sigma$의 분산으로 대체):
$$\ell^{(W)}_{\phi_1, \phi_2} = \frac{\phi_1^T \phi_2}{\|\phi_1\|\|\phi_2\|}$$

여기서 가중치 인자 $\rho$는:
$$\rho = 1 - \frac{p_1^T p_2}{\|p_1\|\|p_2\|}, \quad p_i = f_{\phi_i}(g_\theta(x_T))$$

---

### 2-4. 성능 향상

| 태스크 | 기준 방법 | STAR | 개선 |
|--------|-----------|-------|------|
| SVHN→MNIST | MCD: 96.2% | **98.8%** | +2.6% |
| MNIST→USPS | MCD: 96.5% | **97.8%** | +1.3% |
| USPS→MNIST | MCD: 94.1% | **97.7%** | +3.6% |
| VisDA (객체 분류) | MCD: 71.9% | **82.7%** | **+10.8%** |
| GTA5→Cityscapes | CLAN†: 42.9 mIoU | **43.6 mIoU** | +0.7 |
| Synthia→Cityscapes | CLAN†: 46.0 mIoU | **48.1 mIoU** | +2.1 |

**주목할 점:**
- 복잡한 태스크(VisDA)일수록 STAR의 개선 효과가 더 크다
- 낮은 표준편차 → 랜덤 초기화에 덜 민감 (안정성 향상)

---

### 2-5. 한계

1. **Gaussian 분포 가정의 제한성**: 실제 분류기 분포가 비가우시안일 경우 모델 오명세(model misspecification) 발생 가능
2. **추론 시 단순화**: 테스트 시 샘플링 앙상블 대신 $\mu$만 사용 → 불확실성 정보 미활용
3. **LA 방법에만 적용 가능**: 분류기를 여러 개 사용하지 않는 GA 방법에는 직접 적용 불가
4. **세그멘테이션 개선 폭 소폭**: GTA5→Cityscapes에서 공식 보고 CLAN 대비 0.4% mIoU 향상에 그침 (재현된 코드 기준으로는 0.7% 향상)
5. **비선형 분류기 미지원**: 마지막 FC 레이어만 확률화하며, 더 깊은 분류기 구조에서의 효과는 미검증

---

## 3. 모델 일반화 성능 향상 가능성

### 3-1. 분산($\Sigma$)의 역할과 일반화

학습 수렴 후 $\Sigma$의 분포 변화를 분석하면:

- **초기**: $\sigma$ 값이 균일 분포 (Figure 4a)
- **수렴 후**: 일부 $\sigma$ 값이 크게 증가, 이분포 형태 (Figure 4b)

이 메커니즘의 일반화 향상 원리:

$$\text{큰 } \sigma_i \Rightarrow \text{샘플링된 분류기의 다양성} \uparrow \Rightarrow \text{미정렬 특징 탐지 능력} \uparrow$$
$$\text{작은 } \sigma_i \Rightarrow \text{소스 도메인 판별력 보존} \Rightarrow \text{분류 정확도 유지}$$

이 두 효과의 균형이 **타겟 도메인 일반화를 향상**시킨다.

### 3-2. 무한 분류기 앙상블 효과

매 학습 반복 시 독립적으로 샘플링되는 분류기:
$$\tilde{\phi}^{(t)} \sim \mathcal{N}(\mu^{(t)}, \Sigma^{(t)}), \quad t = 1, 2, \ldots, T$$

전체 학습 과정에서 사실상 $T$개의 서로 다른 분류기를 통해 학습되며, 이는 암묵적 앙상블 효과를 제공한다. Dempster-Shafer 증거 이론에 따르면 더 많은 다양한 분류기가 미정렬 영역을 더 포괄적으로 탐지한다.

### 3-3. Implicit Regularization 효과

STAR는 분류기 공간에서 분포를 학습함으로써:

$$\mathcal{L}_{STAR} = \mathbb{E}_{\tilde{\phi}_1, \tilde{\phi}_2 \sim \mathcal{N}(\mu, \Sigma)}[\mathcal{L}_{MCD}(\tilde{\phi}_1, \tilde{\phi}_2)]$$

이는 기댓값을 통한 손실 평활화(loss smoothing) 효과를 가지며, 분류기가 특정 데이터 포인트에 과적합되는 것을 억제한다.

### 3-4. 토이 실험에서의 일반화 확인

Two-moon 데이터셋 실험에서:
- **Source Only**: 타겟 도메인 경계 크게 오정렬
- **MCD**: 하부 문 우측 끝 일부 미정렬 잔존
- **STAR**: **모든** 타겟 샘플 정확 분류 → 강화된 결정 경계 일반화

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4-1. 연구에 미치는 영향

**① 확률적 UDA 프레임워크의 선구적 역할**

STAR는 UDA에 확률적 딥러닝(Stochastic Deep Learning)을 최초로 도입하였으며, 이후 불확실성 인식 도메인 적응 연구의 토대를 마련하였다.

**② 플러그인 모듈로서의 범용성**

기존 LA 방법(MCD, CLAN 등)에 최소한의 수정으로 적용 가능한 플러그인 형태는, 이후 연구에서 모듈형 UDA 개선 방법론 설계의 방향을 제시하였다.

**③ 분류기 다양성의 중요성 부각**

단순 2-분류기 패러다임에서 벗어나 분류기 다양성(classifier diversity)을 명시적으로 설계하는 연구 방향을 촉발하였다.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

아래 분석은 논문에서 언급된 연구 흐름과 일반적인 UDA 연구 동향을 바탕으로 하며, 개별 논문의 세부 수치는 해당 논문 원문 확인을 권장한다.

| 연구 방향 | 대표 접근법 | STAR와의 관계 |
|-----------|------------|--------------|
| **Transformer 기반 UDA** | CDTrans, TVT (2021~2022) | STAR의 분류기 다양성 개념을 Self-attention 메커니즘과 결합 가능 |
| **소스-프리 DA (Source-Free DA)** | SHOT, NRC (2020~2021) | 소스 데이터 없이 타겟만으로 적응; STAR의 불확실성 모델링이 의사 레이블 품질 향상에 기여 가능 |
| **테스트 타임 적응 (TTA)** | TTT, TENT (2020~2021) | 추론 시 $\mathcal{N}(\mu, \Sigma)$에서 앙상블 샘플링으로 확장 가능 |
| **프롬프트 기반 DA** | DAPrompt, PADCLIP (2022~2023) | CLIP 등 대형 모델의 분류기 헤드에 STAR 개념 적용 가능성 |

**STAR의 차별성 유지 요소:**
- 파라미터 추가 없이 무한 분류기 효과 달성
- Reparameterization trick으로 End-to-end 학습 가능
- 임의의 LA 기반 방법에 적용 가능한 범용성

---

### 4-3. 향후 연구 시 고려사항

#### (1) 분포 선택의 다양화
현재 Gaussian 분포 $\mathcal{N}(\mu, \Sigma)$로 고정되어 있으나:
$$p(\phi) = \text{Gaussian} \Rightarrow \text{Normalizing Flows, VAE, Diffusion 기반 분포로 확장}$$
복잡한 다봉형(multi-modal) 분류기 분포 모델링 가능성 탐색

#### (2) 소스-프리 환경으로의 확장
실제 배포 환경에서는 소스 데이터 접근이 불가한 경우가 많음:
- 저장된 $\mathcal{N}(\mu, \Sigma)$ 파라미터만으로 타겟 도메인 적응
- 의사 레이블(pseudo-label)의 불확실성을 $\Sigma$로 측정하여 노이즈 필터링

#### (3) 대형 언어/비전 모델과의 통합
CLIP, ViT 등 대형 모델의 분류 헤드에 STAR를 적용:
$$\phi_{CLIP} \sim \mathcal{N}(\mu_{pretrained}, \Sigma_{adapted})$$
프리트레인된 $\mu$를 초기화에 활용하고 $\Sigma$만 적응

#### (4) 이론적 보장 강화
현재 STAR의 성능 향상은 경험적 관찰에 기반하므로:
- PAC-Bayes 이론을 통한 일반화 오류 상한(generalization error bound) 도출
- 도메인 갭과 $\Sigma$의 관계에 대한 이론적 분석 필요

$$\mathcal{R}_{target} \leq \mathcal{R}_{source} + d_{\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda + f(\Sigma)$$

#### (5) 멀티 소스/멀티 타겟 도메인 확장
단일 소스→타겟 설정을 넘어:
- 여러 소스 도메인에 대해 각각의 $\mathcal{N}(\mu_k, \Sigma_k)$ 학습 후 혼합
- 타겟 도메인 분포 이동(distribution shift) 강도를 $\Sigma$로 정량화

#### (6) 추론 시 분산 활용
현재 추론 시 $\mu$만 사용하는 단순화에서 벗어나:

$$\hat{y} = \frac{1}{M}\sum_{m=1}^{M} f_{\tilde{\phi}_m}(g_\theta(x)), \quad \tilde{\phi}_m \sim \mathcal{N}(\mu, \Sigma)$$

Monte Carlo 추론을 통한 불확실성 정량화 및 신뢰도 보정(calibration) 연구

---

## 참고 자료

**주요 참고 문헌 (논문 내 인용 기준):**
1. **본 논문**: Lu, Z., Yang, Y., Zhu, X., Liu, C., Song, Y.-Z., & Xiang, T. (2020). "Stochastic Classifiers for Unsupervised Domain Adaptation." *CVPR 2020*, pp. 9111–9120.
2. Saito, K., et al. (2018). "Maximum Classifier Discrepancy for Unsupervised Domain Adaptation." *CVPR 2018*. [MCD]
3. Luo, Y., et al. (2019). "Taking a Closer Look at Domain Shift: Category-Level Adversaries for Semantics Consistent Domain Adaptation." *CVPR 2019*. [CLAN]
4. Kingma, D.P., & Welling, M. (2014). "Auto-Encoding Variational Bayes." *ICLR 2014*. [VAE / Reparameterization Trick]
5. Ganin, Y., & Lempitsky, V. (2015). "Unsupervised Domain Adaptation by Backpropagation." *ICML 2015*. [DANN]
6. Ganin, Y., et al. (2016). "Domain-Adversarial Training of Neural Networks." *JMLR 2016*.
7. Lee, C.-Y., et al. (2019). "Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation." *CVPR 2019*. [SWD]

> **⚠️ 주의사항**: 2020년 이후 최신 연구(CDTrans, SHOT, TENT 등)와의 비교 분석 부분은 일반적인 UDA 연구 동향에 기반한 분석이며, 각 논문의 구체적 수치나 STAR와의 직접 비교 실험은 본 논문에 포함되어 있지 않으므로 해당 원논문 참조를 권장합니다.
