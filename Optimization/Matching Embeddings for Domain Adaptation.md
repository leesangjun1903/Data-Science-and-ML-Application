# Matching Embeddings for Domain Adaptation (AVDA)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
본 논문은 **Adversarial Variational Domain Adaptation (AVDA)**를 제안합니다. 핵심 주장은 다음과 같습니다:

> "소스 도메인의 풍부한 레이블 데이터와 타겟 도메인의 소수 레이블 데이터를 활용하여, **가우시안 혼합 모델(GMM)** 기반의 잠재 공간에서 같은 클래스의 소스·타겟 샘플을 동일한 가우시안 컴포넌트에 매핑함으로써 도메인 적응을 효과적으로 수행할 수 있다."

### 주요 기여
| 기여 항목 | 설명 |
|-----------|------|
| **AVDA 프레임워크** | 변분 추론 + 적대적 학습의 통합 |
| **GMM 기반 임베딩** | 클래스별 가우시안 컴포넌트 정렬 |
| **Semi-supervised 활용** | 레이블/비레이블 타겟 샘플 동시 활용 |
| **Few-shot 적응** | 소수 레이블만으로 SOTA 달성 |
| **생성 모델 통합** | 분류를 돕는 생성적 프로세스 포함 |

---

## 2. 세부 분석

### 2.1 해결하고자 하는 문제

**도메인 시프트(Domain Shift)** 문제: 소스 도메인 $p^s(\mathbf{x}^s, y^s)$와 타겟 도메인 $p^t(\mathbf{x}^t, y^t)$의 분포가 달라 ($p^s \neq p^t$), 소스에서 학습한 모델이 타겟에서 성능이 저하되는 문제를 해결합니다.

**기존 방법의 한계:**
- UDA(Unsupervised DA) 방법: 도메인 시프트가 클 때 성능 저하
- Few-shot DA 방법: 비레이블 데이터 미활용, 과적합 문제
- 기존 Semi-supervised DA: 소수 레이블의 이점을 충분히 활용 못함

**문제 정의:**
- 소스 도메인: $\mathcal{D}^s = \{(\mathbf{x}_i^s, y_i^s)\}\_{i=1}^{n^s}$ (다수의 레이블 샘플)
- 타겟 레이블: $\mathcal{D}^t = \{(\mathbf{x}_i^t, y_i^t)\}\_{i=1}^{n^t}$ (소수의 레이블 샘플)
- 타겟 비레이블: $\mathcal{D}^u = \{(\mathbf{x}_i^u)\}\_{i=1}^{n^u}$ (다수의 비레이블 샘플)
- 목표: $n^t$가 매우 작은 few-shot 시나리오에서 공유 임베딩 공간 $\mathbf{z}$ 학습

---

### 2.2 제안 방법 및 수식

#### 생성 모델 정의

소스와 타겟의 결합 확률 분포:

$$p(\mathbf{x}^s, y^s, \mathbf{z}^s) = p(y^s)p(\mathbf{z}^s|y^s)p(\mathbf{x}^s|\mathbf{z}^s) \tag{1}$$

$$p(\mathbf{x}^t, y^t, \mathbf{z}^t) = p(y^t)p(\mathbf{z}^t|y^t)p(\mathbf{x}^t|\mathbf{z}^t) \tag{2}$$

각 확률 분포의 정의:

$$p(y^s) = \text{Cat}(y^s|\pi^s) \tag{3}$$

$$p(\mathbf{z}^s|y^s) = \mathcal{N}(\mathbf{z}^s|\mu(y^s), \sigma^2(y^s)\mathbf{I}) \tag{4}$$

$$p_\theta(\mathbf{x}^s|\mathbf{z}^s) = \text{Ber}(\mathbf{x}^s|\mu_x(\mathbf{z}^s, \theta)) \;\text{or}\; \mathcal{N}(\mathbf{x}^s|\mu_x(\mathbf{z}^s,\theta), \sigma^2_x(\mathbf{z}^s,\theta)\mathbf{I}) \tag{5}$$

타겟 도메인도 동일한 GMM 파라미터 $\mu(y^s), \sigma^2(y^s)$를 공유하여 **소스·타겟 정렬**:

$$p(\mathbf{z}^t|y^t) = \mathcal{N}(\mathbf{z}^t|\mu(y^s), \sigma^2(y^s)\mathbf{I}) \tag{7}$$

#### 사후 분포 근사 (Encoder)

$$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z}|\mu_\phi(\mathbf{x}), \sigma^2_\phi(\mathbf{x})\mathbf{I}) \tag{9}$$

$$q_\phi(y|\mathbf{x}) = \text{Cat}(y|\pi_\phi(\mathbf{x})) \tag{10}$$

#### 변분 목적 함수 (Variational Objective)

**[1] 소스 지도 목적함수 (Supervised Source Objective):**

$$\log p(\mathbf{x}, y) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x})\|p(\mathbf{z}|y)) + \log p(y) = \mathcal{L}^s_{ELBO} \tag{11}$$

$$\mathcal{L}^s_{sup} = -\mathcal{L}^s_{ELBO} + \alpha^s \mathbb{E}_{(\mathbf{x},y)\sim\mathcal{D}^s}[-\log q_\phi(y|\mathbf{x})] \tag{12}$$

**[2] 타겟 지도 목적함수 (Supervised Target Objective):**

$$\mathcal{L}^t_{sup} = -\mathcal{L}^t_{ELBO} + \alpha^t \mathbb{E}_{(\mathbf{x},y)\sim\mathcal{D}^t}[-\log q_\phi(y|\mathbf{x})] \tag{14}$$

**[3] 비지도 목적함수 (Unsupervised Objective):**

$$\log p(\mathbf{x}) \geq \mathcal{L}^u_{ELBO} = -\mathcal{L}_{unsup} \tag{15}$$

비레이블 샘플의 클래스 확률은 GMM으로부터 직접 계산:

$$q(y|\mathbf{x}) = p(y|\mathbf{z}) = \frac{p(\mathbf{z}|y)p(y)}{\sum_{k=1}^{K} p(\mathbf{z}|y=k)p(y=k)} \tag{17}$$

**[4] 전체 변분 목적함수:**

$$\min_{\phi,\theta,\rho} \mathcal{L}^v = \gamma \mathcal{L}_{sup} + (1-\gamma)\mathcal{L}_{unsup} \tag{18}$$

여기서 $\mathcal{L}\_{sup} = \mathcal{L}^s_{sup} + \mathcal{L}^t_{sup}$, $\gamma$는 레이블/비레이블 중요도 조절 하이퍼파라미터

**지도 목적함수 전개 (Gaussian 가정):**

$$\mathcal{L}_{sup} = \sum_{i=1}^{D}\left[\log\sigma_{x|i} + \frac{(x_i - \mu_{x|i})^2}{2\sigma^2_{|i}}\right] + \frac{1}{2}\sum_{j=1}^{J}\left[\log(\sigma^2(y)|_j) + \frac{\sigma^2_\phi(\mathbf{x})|_j}{\sigma^2(y)|_j} + \frac{(\mu_\phi(\mathbf{x})|_j - \mu(y)|_j)^2}{\sigma^2(y)|_j}\right] - \sum_{j=1}^{J}(1 + \log\sigma^2(\mathbf{x})|_j) + H(q_\phi(y|\mathbf{x}), y) \tag{19}$$

**비지도 목적함수 전개:**

$$\mathcal{L}_{unsup} = \sum_{i=1}^{D}\left[\log\sigma_{x|i} + \frac{(x_i - \mu_{x|i})^2}{2\sigma^2_{|i}}\right] + \frac{1}{2}\sum_{k=1}^{K}q_\phi(y_k|\mathbf{x})\sum_{j=1}^{J}\left[\log\sigma^2(y_k)|_j + \frac{\sigma^2_\phi(\mathbf{x})|_j}{\sigma^2(y_k)|_j} + \frac{(\mu_\phi(\mathbf{x})|_j - \mu(y_k)|_j)^2}{\sigma^2(y_k)|_j}\right] - \sum_{k=1}^{K}q_\phi(y_k|\mathbf{x})\log\frac{\pi_k}{q_\phi(y_k|\mathbf{x})} - \sum_{j=1}^{J}(1 + \log\sigma^2(\mathbf{x})|_j) \tag{20}$$

#### 적대적 목적함수 (Adversarial Objective)

**판별자 (Discriminator) 최적화:**

$$\min_{w} \mathcal{L}_D = -\mathbb{E}_{q_\phi(\mathbf{z}^s|\mathbf{x}^s)}[\log D_w(\mathbf{z}^s)] - \mathbb{E}_{q_\phi(\mathbf{z}^t|\mathbf{x}^t)}[\log(1-D_w(\mathbf{z}^t))] \tag{24}$$

**인코더 (Encoder) 적대 최적화:**

$$\min_{\phi} \mathcal{L}_A = \mathbb{E}_{q_\phi(\mathbf{z}^t|\mathbf{x}^t)}[\log(1 - D_w(\mathbf{z}^t))] \tag{25}$$

**재파라미터화 트릭 (Reparametrization Trick):**

$$\mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma^2_\phi(\mathbf{x}) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \tag{23}$$

---

### 2.3 모델 구조

```
[소스 입력 x^s] ──┐
                   ├──→ [공유 인코더 φ] ──→ [잠재 공간 z (GMM)]
[타겟 입력 x^t] ──┘          │                    │
                              │              [판별자 w]
                              │           (도메인 판별)
                    ┌─────────┴─────────┐
               [소스 디코더 θ]    [타겟 디코더 ρ]
               재구성 x^s         재구성 x^t
```

| 구성 요소 | 파라미터 | 역할 |
|-----------|----------|------|
| 공유 인코더 | $\phi$ | 소스·타겟 공통 잠재 표현 |
| 소스 디코더 | $\theta$ | 소스 데이터 재구성 |
| 타겟 디코더 | $\rho$ | 타겟 데이터 재구성 |
| 도메인 판별자 | $w$ | 소스/타겟 구분 (적대 학습) |
| GMM 파라미터 | $\mu(y), \sigma^2(y)$ | 클래스별 가우시안 컴포넌트 |

**구현 세부사항:**
- 임베딩 차원: 20차원
- 옵티마이저: Adam ( $\beta_1=0.9$, $\beta_2=0.999$, lr= $0.0001$ )
- 배치 크기: 128 (소스, 레이블 타겟, 비레이블 타겟 각각)
- 하이퍼파라미터: $\alpha^s = \alpha^t = 1$ , $\gamma = 0.9$
- 사전 학습(pretraining) 후 전체 학습 진행

---

### 2.4 성능 향상 결과

#### 실험 1: MNIST→USPS (2,000 소스 샘플)

| Method | 0-shot | 1-shot | 3-shot | 5-shot | 7-shot |
|--------|--------|--------|--------|--------|--------|
| CCSA | 65.40 | 85.00 | 90.10 | 92.40 | 92.90 |
| FADA | 65.40 | 89.10 | 91.90 | 93.40 | 94.40 |
| d-SNE | 73.01 | 92.90 | 93.55 | 95.13 | 96.13 |
| **AVDA (ours)** | **97.34** | **97.54** | **97.71** | **97.80** | **97.83** |

#### 실험 2: 3가지 적응 태스크 (1~10-shot)

| Method | M→U (1-shot) | U→M (1-shot) | S→M (1-shot) |
|--------|-------------|-------------|-------------|
| F-CADA | 97.20 | 97.50 | 94.80 |
| **AVDA (best)** | **98.23** | **98.38** | **96.60** |

→ 전반적으로 **0.08%~1.88% 정확도 향상**

#### Ablation Study (S→M, 5-shot)

| 모델 변형 | 정확도 |
|-----------|--------|
| AVDA (전체) | **97.56** |
| AVDA $_{WD}$ (판별자 제거) | 91.57 ± 1.02 |
| AVDA $_{WP}$ (사전학습 제거) | 65.23 ± 4.67 |

→ 도메인 판별자와 사전학습 모두 성능에 필수적

---

### 2.5 한계점

1. **실험 도메인 제한**: 디지털 숫자 데이터셋(MNIST, USPS, SVHN)에만 검증 → 자연 이미지, 텍스트, 의료 데이터 등에 대한 검증 부재
2. **하이퍼파라미터 민감도**: $\gamma$, $\alpha^s$, $\alpha^t$ 등의 수동 튜닝 필요, 도메인마다 최적값 상이 가능
3. **확률적 학습의 불안정성**: 결과가 랜덤 시드에 따라 변동 (best vs. random 결과 차이 존재)
4. **사전학습 의존성**: 사전학습 없이 훈련 시 성능이 65.23%로 급락 → 초기화에 민감
5. **GMM 컴포넌트 수 고정**: 클래스 수 $K$를 미리 알아야 함 → 오픈셋 시나리오 부적합
6. **단순 적대적 손실**: 기본 GAN 손실 사용 → 학습 불안정 가능성 (Wasserstein 거리 등 미사용)
7. **확장성**: 대규모 고해상도 데이터에 대한 계산 비용 검증 미흡

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 가능하게 하는 핵심 메커니즘

#### (1) GMM 기반 클래스 조건 잠재 공간
$$p(\mathbf{z}|y) = \mathcal{N}(\mathbf{z}|\mu(y), \sigma^2(y)\mathbf{I})$$

같은 클래스의 소스·타겟 샘플이 **동일한 가우시안 컴포넌트**에 매핑되도록 강제함으로써, 클래스 경계가 잠재 공간에서 명확히 분리됩니다. 이는 타겟 도메인의 비레이블 샘플에 대한 **구조적 일반화**를 가능하게 합니다.

#### (2) 생성 모델의 정규화 효과
변분 추론을 통한 ELBO 최대화는 다음 두 항목을 동시에 최적화합니다:
- **재구성 항**: $\mathbb{E}\_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]$ → 데이터의 본질적 구조 학습
- **KL 정규화 항**: $D_{KL}(q_\phi(\mathbf{z}|\mathbf{x})\|p(\mathbf{z}|y))$ → 과적합 방지

이 정규화는 단순 판별 모델보다 **더 강건한 표현**을 학습하게 합니다.

#### (3) 도메인 불변 특징 학습
적대적 학습을 통해 $q(\mathbf{z}^s) \approx q(\mathbf{z}^t)$가 되도록 강제함으로써, 인코더가 도메인에 무관한 **공유 특징**을 학습합니다. 이는 새로운 도메인에 대한 일반화 기반이 됩니다.

#### (4) Semi-supervised 학습의 일반화 기여
비레이블 타겟 샘플의 클래스 확률을 GMM으로부터 soft하게 추정:

$$q(y|\mathbf{x}) = p(y|\mathbf{z}) = \frac{p(\mathbf{z}|y)p(y)}{\sum_{k=1}^{K}p(\mathbf{z}|y=k)p(y=k)}$$

이를 통해 레이블이 없는 타겟 샘플도 클래스 경계 학습에 기여하여, **적은 레이블로도 높은 일반화 성능**이 가능합니다.

#### (5) "속도 적응(Speed of Adaptation)"
논문이 강조하는 핵심 특성으로, 1-shot 및 3-shot에서 이미 경쟁력 있는 성능을 달성합니다. 이는 GMM 구조가 사전 지식(클래스 구조)을 효과적으로 인코딩하기 때문입니다.

### 3.2 일반화 한계와 개선 방향

| 한계 | 잠재적 개선 |
|------|-------------|
| 숫자 데이터에만 검증 | 자연어, 의료 이미지 등 다양한 도메인 검증 필요 |
| GMM의 등방성 공분산 | 풀 공분산 행렬 도입으로 더 복잡한 분포 모델링 |
| 고정된 클래스 수 | Dirichlet Process 기반 비모수 GMM 도입 |

---

## 4. 앞으로의 연구에 미치는 영향 및 고려점

### 4.1 연구에 미치는 영향

#### (1) 생성 모델 + 도메인 적응의 통합 패러다임 제시
AVDA는 VAE 기반 생성 모델을 도메인 적응에 통합한 초기 연구 중 하나로, 이후 **생성 모델을 DA에 활용하는 연구의 방향성**을 제시합니다.

#### (2) 클래스 조건 정렬의 중요성 강조
단순히 전체 분포를 정렬하는 것이 아닌, **클래스별 정렬**의 중요성을 GMM을 통해 명확히 증명했습니다. 이는 이후 클래스 조건 DA 연구들의 이론적 기반이 됩니다.

#### (3) Semi-supervised DA 벤치마크 기여
Few-shot 시나리오에서의 평가 프로토콜과 결과는 이후 SSDA 연구의 **기준선(baseline)**으로 활용될 수 있습니다.

#### (4) 변분 추론의 DA 적용 가능성 확대
변분 목적함수의 구체적 유도 과정을 제시하여, **다른 도메인 및 모달리티**에 적용하는 연구의 기반을 제공합니다.

---

### 4.2 향후 연구 시 고려할 점

#### (1) 대규모 및 다양한 도메인으로의 확장
- **Office-31, DomainNet, VisDA** 등 표준 DA 벤치마크 적용 필요
- NLP, 의료 영상, 시계열 등 다양한 모달리티 검증 필요
- 현재 숫자 데이터셋만으로는 일반화 주장에 한계

#### (2) 더 강력한 정규화 기법 통합
- **Wasserstein 거리** 기반 적대 학습으로 학습 안정성 향상
- **Spectral Normalization** 등을 통한 판별자 정규화
- **Mixup** 또는 **CutMix** 기반 데이터 증강과의 결합

#### (3) 동적 GMM 구조 탐색
- 현재 K개 컴포넌트 고정 → **오픈셋(Open-set) DA** 시나리오 부적합
- Dirichlet Process Mixture Model 또는 **동적 클러스터링** 도입 고려

#### (4) 트랜스포머 기반 인코더 통합
- 2020년 이후 **Vision Transformer(ViT)** 등 대형 사전학습 모델과의 결합
- 사전학습 모델의 전이 학습 능력 + AVDA의 정렬 메커니즘 시너지

#### (5) 이론적 일반화 경계 분석
- 현재 경험적 결과만 제시 → **PAC-Bayes 이론** 또는 **도메인 적응 이론(Ben-David et al.)** 기반의 일반화 경계 도출

#### (6) 하이퍼파라미터 자동화
- $\gamma$, $\alpha^s$, $\alpha^t$ 등의 자동 탐색 (AutoML, NAS 기법 적용)
- 메타학습(Meta-learning) 기반 하이퍼파라미터 최적화

#### (7) 다중 소스 도메인 적응으로 확장
- 현재 단일 소스 → **Multi-source DA**로의 확장 (여러 GMM 혼합 구조)

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의**: 아래 비교는 제공된 논문(arXiv:1909.11651v4, 2021)의 내용과 제 학습 데이터에 기반합니다. 2020년 이후 각 논문의 정확한 수치는 원문을 반드시 확인하세요.

### 5.1 주요 후속 연구 동향

#### (1) Semi-supervised DA 분야

**MME (Minimax Entropy, Saito et al., ICCV 2019)**
- AVDA가 직접 비교한 방법 [38]
- 미니맥스 엔트로피를 통해 타겟 특징의 경계 조정
- AVDA가 few-shot에서 전반적으로 우수

**CDAC (Li et al., CVPR 2021)**
- Cross-Domain Adaptive Clustering
- 의사 레이블(pseudo-label)과 클러스터링 결합
- 더 복잡한 도메인 (Office-Home 등)에서 검증

| 비교 항목 | AVDA | CDAC |
|-----------|------|------|
| 잠재 공간 구조 | GMM (확률적) | 클러스터링 (결정론적) |
| 생성 모델 | ✅ VAE 기반 | ❌ |
| 비레이블 활용 | Soft GMM 할당 | 의사 레이블 |
| 실험 도메인 | 숫자 데이터 | Office-Home 등 |

#### (2) 트랜스포머 기반 DA

**TVT (Yang et al., 2022), CDTrans (Xu et al., ICLR 2022)**
- Vision Transformer를 DA에 적용
- 대규모 사전학습 모델의 강력한 전이 능력 활용
- AVDA 대비 훨씬 큰 모델 파라미터 (수억 개)

**AVDA와의 차이:**
- AVDA: 경량 구조 (LeNet 수준), 명시적 확률 모델
- TVT/CDTrans: 대형 Transformer, 암묵적 특징 정렬

#### (3) 클래스 조건 정렬 강화 연구

**CLDA (Singh, NeurIPS 2021)**
- Contrastive Learning for DA
- 같은 클래스 내 소스·타겟 유사도 극대화
- AVDA의 GMM 정렬과 유사한 동기, 다른 구현

**ProDA (Zhang et al., CVPR 2021)**
- Prototype Distribution Alignment
- 클래스 프로토타입 기반 분포 정렬
- GMM의 평균 $\mu(y)$와 개념적 유사성

#### (4) Source-Free DA (2020년 이후 새로운 패러다임)

**SHOT (Liang et al., ICML 2020), NRC (Yang et al., NeurIPS 2021)**
- **소스 데이터 없이** 타겟 도메인만으로 적응
- 프라이버시 보호, 실용성 측면에서 새로운 패러다임

**AVDA의 한계**: 소스 데이터를 훈련 시 필요로 함 → Source-Free 시나리오 미대응

### 5.2 종합 비교 테이블

| 방법 | 연도 | 레이블 활용 | 생성 모델 | 클래스 조건 | 확장 가능성 |
|------|------|-------------|-----------|-------------|-------------|
| AVDA (본 논문) | 2021 | Semi-sup | ✅ VAE+GMM | ✅ GMM 컴포넌트 | 제한적 |
| MME | 2019 | Semi-sup | ❌ | ❌ | 중간 |
| CDAC | 2021 | Semi-sup | ❌ | ✅ 클러스터링 | 높음 |
| TVT | 2022 | Semi-sup | ❌ | △ | 매우 높음 |
| SHOT | 2020 | Unsup | ❌ | △ | 높음 |
| ProDA | 2021 | Unsup | ❌ | ✅ 프로토타입 | 높음 |

### 5.3 AVDA의 차별성과 현재 위치

```
2019: AVDA 제안 (GMM + VAE + 적대적 학습)
        ↓ 기여: 클래스 조건 정렬, 생성 모델 통합
2020: Source-Free DA 등장 (SHOT)
        ↓ 새로운 패러다임으로 일부 한계 노출
2021: 대형 사전학습 모델 기반 DA (CDTrans 등)
        ↓ 성능 면에서 AVDA 능가 가능
2022~: 확산 모델(Diffusion), 대형 언어 모델 기반 DA 등장
```

**결론적으로**, AVDA는 GMM 기반 확률적 정렬이라는 독창적 아이디어를 제시했으나, 이후 **더 강력한 사전학습 모델과 다양한 실험 설정**이 표준이 되면서 숫자 데이터셋 한정의 검증이 주요 한계로 지적됩니다. 그러나 **생성 모델을 통한 명시적 확률 정렬**이라는 아이디어는 여전히 이론적으로 가치 있으며, 확산 모델 기반 DA 연구 등과 연계될 가능성이 있습니다.

---

## 참고 자료

**주 논문:**
- Pérez-Carrasco, M., Cabrera-Vives, G., Protopapas, P., Astorga, N., & Belhaj, M. (2021). *Matching Embeddings for Domain Adaptation*. arXiv:1909.11651v4

**논문 내 핵심 참고문헌 (직접 인용됨):**
- [6] Ganin, Y. et al. (2016). *Domain-adversarial training of neural networks*. JMLR 17(59)
- [17] Jiang, Z. et al. (2017). *Variational deep embedding: An unsupervised and generative approach to clustering*. IJCAI 2017
- [20] Kingma, D.P. & Welling, M. (2014). *Auto-encoding variational Bayes*. ICLR 2014
- [38] Saito, K. et al. (2019). *Semi-supervised domain adaptation via minimax entropy*. ICCV 2019
- [55] Zou, H. et al. (2019). *Consensus adversarial domain adaptation*. AAAI 2019
- [30] Motiian, S. et al. (2017). *Unified deep supervised domain adaptation and generalization*. ICCV 2017
- [29] Motiian, S. et al. (2017). *Few-shot adversarial domain adaptation*. NeurIPS 2017
- [48] Xu, X. et al. (2019). *d-SNE: Domain adaptation using stochastic neighborhood embedding*. CVPR 2019

**2020년 이후 비교 연구 (일반 학술 지식 기반, 원문 확인 권장):**
- Liang, J. et al. (2020). *Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation*. ICML 2020
- Li, J. et al. (2021). *Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation*. CVPR 2021
- Xu, T. et al. (2022). *CDTrans: Cross-Domain Transformer for Unsupervised Domain Adaptation*. ICLR 2022
