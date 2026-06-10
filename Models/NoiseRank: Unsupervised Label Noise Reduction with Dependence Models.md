# NoiseRank: Unsupervised Label Noise Reduction with Dependence Models 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

NoiseRank는 **감독(supervision) 없이** Markov Random Fields(MRF)를 활용하여 레이블 노이즈를 탐지하고 제거하는 비지도 학습 기반 프레임워크입니다. 기존 방법들이 정제된 레이블(clean labels)이나 노이즈 분포에 대한 사전 지식을 필요로 하는 것과 달리, NoiseRank는 순수하게 데이터 내부의 의존성 구조(dependence structure)만을 활용합니다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **완전 비지도 노이즈 탐지** | ground-truth 레이블, 노이즈 분포 사전 정보 불필요 |
| **해석 가능성 (Interpretability)** | 알고리즘 및 출력이 인간이 이해 가능한 구조 |
| **모달리티 독립성** | 이미지, 텍스트, 멀티모달 등 어떤 도메인에도 적용 가능 |
| **아키텍처 독립성** | 분류기 구조 및 최적화 방법에 무관 |
| **실험적 검증** | Food101-N (~20% 노이즈), Clothing-1M (~40% 노이즈)에서 SOTA 달성 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

현실에서 수집되는 데이터셋은 다음과 같은 이유로 **레이블 노이즈**를 포함합니다:

- 어노테이터의 편향, 부주의, 역량 부족
- 웹 스크래핑, 크라우드소싱, 기계 생성 레이블링 등 자동화 채널
- 높은 모호성을 가진 도메인(예: 혐오 발언 탐지, 의료 이미지)

기존 방법들의 문제점:
- **지도 학습 기반 방법**: 정제된 레이블 혹은 노이즈 전이 행렬(noise transition matrix) 필요
- **노이즈 강건 분류기**: 특정 아키텍처 및 손실 함수에 의존
- **비지도 방법(아웃라이어 제거 등)**: 노이즈 샘플이 아닌 하드 샘플까지 제거하는 문제 발생

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 벡터 표현 학습

노이즈가 있는 데이터셋을 다음과 같이 정의합니다:

$$\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$$

여기서 $x_i \in \mathbb{R}^m$은 벡터 표현, $y_i \in \{1, 2, \ldots, C\}$는 레이블(잠재적으로 오류 포함)입니다.

인스턴스 간 유사도는 유클리드 거리로 정의합니다:

$$d(x_i, x_j) = \|x_i - x_j\|_2$$

#### Step 2: 클래스 프로토타입 생성

각 클래스에 대해 K-means 클러스터링으로 프로토타입을 선택합니다. 클래스당 클러스터 수는 다음 규칙으로 결정합니다:

$$\text{프로토타입 수} = \left\lfloor \sqrt{\rho/2} \right\rfloor$$

여기서 $\rho$는 클래스당 평균 인스턴스 수입니다.

#### Step 3: 레이블 예측 생성

각 프로토타입 $i$에 대해 가중 k-최근접 이웃(k-NN) 분류기로 예측 레이블 $y'_i$를 생성합니다:

$$y'_i = \underset{v \in \{1,2\ldots C\}}{\arg\max} \sum_{x_j \in \mathcal{N}(x_i)} \kappa(x_i, x_j) \mathbb{1}\{y_j = v\} \tag{1}$$

거리 커널 함수:

$$\kappa(x_i, x_j) = \frac{1}{b + d(x_i, x_j)^e} \tag{2}$$

여기서 $b > 0$은 편향 파라미터, $e > 0$은 가중치 지수입니다.

#### Step 4: MRF 기반 의존성 모델 구성

MRF의 결합 확률은 다음과 같습니다:

$$P(x_i, y_i, \mathcal{D}, Y) = \frac{1}{Z} \prod_{c \in C(G)} \phi(c; \Lambda) \tag{3}$$

정규화 항 $Z$ 계산이 고비용이므로, 랭킹 목적으로 다음과 같이 단순화합니다:

$$P(x_i, y_i | \mathcal{D}, Y) \overset{rank}{=} \sum_{c \in C(G)} \lambda_c f(c) \tag{5}$$

#### Step 5: NoiseRank 점수 함수

4가지 클리크 유형을 정의하고 최종 랭킹 점수를 계산합니다:

$$P(x_i, y_i|\mathcal{D}, Y) \overset{rank}{=} \sum_{(x_i,x_j)\in c_{11}} -\kappa(x_i, x_j)$$
$$+ \sum_{(x_i,x_j)\in c_{10}} (1-\alpha)\kappa(x_i, x_j) + \sum_{(x_i,x_j)\in c_{01}} \alpha\kappa(x_i, x_j)$$
$$+ \sum_{(x_i,x_j)\in c_{00}} (\alpha \times b_f)\kappa(x_i, x_j) \tag{6}$$

**클리크 유형 설명:**

| 클리크 유형 | 조건 | 가중치 $\lambda_c$ | 의미 |
|-------------|------|--------------------|------|
| $c_{11}$ | $y_i = y_j$ | $-1$ | 같은 레이블 → 보상(낮은 노이즈 가능성) |
| $c_{10}$ | $y_i \neq y_j$, $y'_j = y_j$ | $1 - \alpha$ | 잘못된 투표를 했지만 예측에 영향 없음 |
| $c_{01}$ | $y_i \neq y_j$, $y'_j \neq y_j$, $y'_j \neq y_i$ | $\alpha$ | 오예측에 간접 기여 |
| $c_{00}$ | $y_i \neq y_j$, $y'_j = y_i$ | $\alpha \times b_f$ | 오예측에 직접 기여 → 강한 페널티 |

#### Step 6: 노이즈 제거 임계값 적용

$$w(x_i) = \begin{cases} 0, & \text{if } P(x_i, y_i|\mathcal{D}, Y) > \delta \\ 1, & \text{otherwise} \end{cases} \tag{7}$$

### 2.3 모델 구조

```
[노이즈 데이터셋 D]
      ↓
[ResNet-50 표현 학습 (256차원 병목층)]
      ↓
[K-means 클러스터링 → 클래스 프로토타입 P]
      ↓
[k-NN 탐색 → 레이블 예측 Y']
      ↓
[NoiseRank MRF 점수 계산 (Eq. 6)]
      ↓
[랭킹 기반 노이즈 제거 (임계값 δ)]
      ↓
[정제된 데이터셋으로 모델 파인튜닝]
      ↓ (반복)
[반복적 성능 향상]
```

### 2.4 성능 향상

**Food-101N (노이즈 약 20%):**

| 방법 | 유형 | Top-1 정확도 (%) |
|------|------|-----------------|
| None (noisy) | - | 81.44 |
| CleanNet | 지도 | 83.95 |
| DeepSelf | 비지도 | 85.11 |
| **NoiseRank (반복)** | **비지도** | **85.78** |

**Clothing-1M (노이즈 약 40%):**

| 방법 | 유형 | Top-1 정확도 (%) |
|------|------|-----------------|
| None (noisy) | - | 68.94 |
| PENCIL | 비지도 | 73.49 |
| DeepSelf | 비지도 | 74.45 |
| **NoiseRank** | **비지도** | **73.82** |

**노이즈 탐지 Recall:**
- Food-101N: **85.61%** (CleanNet 지도학습 71.06% 대비 우수)
- Clothing-1M: **74.18%** (CleanNet 지도학습 69.40% 대비 우수)

### 2.5 한계점

1. **표현 품질 의존성**: 초기 표현(embedding)의 품질이 노이즈 탐지 성능에 직접 영향을 미침
2. **하이퍼파라미터 민감성**: $\alpha$, $b_f$, $k$, $\delta$ 등 여러 하이퍼파라미터 조정 필요
3. **계산 비용**: 대규모 데이터셋에서 $\mathcal{O}(N|P|)$ 쌍에 대한 계산 필요 (FAISS로 완화)
4. **고노이즈 한계**: Clothing-1M (~40% 노이즈)에서 DeepSelf 대비 소폭 낮은 성능
5. **레이블 수정 미지원**: 노이즈 제거만 수행하고, 레이블을 올바르게 수정하는 기능 없음
6. **클래스 불균형 취약성**: 심각한 클래스 불균형 상황에서의 성능 검증 부족

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 반복적 학습 프레임워크를 통한 일반화

NoiseRank의 가장 중요한 일반화 메커니즘은 **반복적 정제(iterative refinement)** 입니다:

$$\text{표현 학습} \rightarrow \text{NoiseRank 노이즈 탐지} \rightarrow \text{데이터 정제} \rightarrow \text{파인튜닝} \rightarrow \text{반복}$$

이 과정에서:
- 파인튜닝된 모델이 더 나은 표현을 생성
- 더 나은 표현은 더 정확한 노이즈 탐지를 가능하게 함
- 더 정확한 노이즈 탐지는 더 깨끗한 훈련 데이터를 생성
- 결과적으로 모델의 일반화 성능이 향상됨

### 3.2 대규모 준지도 학습에서의 일반화

YFCC100m (9,920만 이미지)에 NoiseRank를 적용한 실험에서:
- 노이즈 제거 없음: ImageNet-1K 79.06%
- 상위 1.8% 노이즈 제거: **79.34%**
- 동일 비율 랜덤 제거: 78.96%

이는 NoiseRank가 **사전학습 데이터의 품질을 향상**시켜, 다운스트림 태스크의 일반화 성능을 개선함을 보여줍니다.

### 3.3 도메인 독립적 일반화

- **아키텍처 독립성**: 표준 CNN, Transformer 등 모든 분류 아키텍처에 적용 가능
- **모달리티 독립성**: 이미지, 텍스트, 멀티모달 데이터에 동일한 프레임워크 적용
- **손실 함수 독립성**: 표준 크로스엔트로피 손실만으로도 효과적

### 3.4 노이즈 강건성과 일반화의 관계

$$\text{일반화 오차} \leq \underbrace{\epsilon_{\text{noise}}}_{\text{노이즈로 인한 오차}} + \underbrace{\epsilon_{\text{model}}}_{\text{모델 복잡도 오차}}$$

NoiseRank는 $\epsilon_{\text{noise}}$를 줄임으로써 전체 일반화 오차를 감소시킵니다. 특히:
- Food-101N에서 오류율 약 **11% 감소** (CleanNet 대비)
- Clothing-1M에서 오류율 약 **16% 감소** (noisy train 대비)

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

**① 비지도 학습 패러다임의 확장**
- NoiseRank는 레이블 노이즈 처리에서 순수 비지도 접근이 가능함을 실증
- 향후 자기지도학습(self-supervised learning)과의 결합 연구를 촉진

**② LLM/Foundation Model 시대의 데이터 정제**
- 웹에서 자동 수집된 데이터(예: LAION-5B 등)의 노이즈 처리에 NoiseRank 철학이 적용 가능
- 멀티모달 Foundation Model의 사전학습 데이터 정제 파이프라인으로 확장 가능

**③ 해석 가능한 AI(XAI) 연구**
- 가장 가까운 이웃과 함께 노이즈 탐지 이유를 시각적으로 설명하는 접근법은 XAI 연구에 영감 제공

**④ 준지도/반지도 학습과의 융합**
- YFCC100m 실험에서 보인 것처럼, 기계 생성 레이블의 품질 개선을 통해 대규모 준지도 학습 성능 향상 가능

### 4.2 앞으로 연구 시 고려할 점

**① 표현 품질과 노이즈 탐지의 상호 의존성 해결**
- 초기 표현이 노이즈로 오염된 경우, 클러스터링 품질이 낮아져 프로토타입이 부정확해질 수 있음
- **제안**: 자기지도 사전학습(SimCLR, BYOL, DINO 등)을 통해 레이블 독립적인 초기 표현 학습

**② 레이블 수정(correction) 기능 추가**
- 현재 NoiseRank는 노이즈 샘플을 제거만 하며, 올바른 레이블로 수정하지 않음
- **제안**: MRF 점수와 함께 레이블 재할당(re-labeling) 메커니즘 결합

**③ 하이퍼파라미터 자동화**
- $\alpha$, $b_f$, $k$, $\delta$ 등의 하이퍼파라미터를 수동으로 탐색해야 함
- **제안**: 베이지안 최적화 또는 메타학습 기반 자동 하이퍼파라미터 선택

**④ 극단적 노이즈 비율(>50%)에서의 성능 검증**
- 현재 최대 ~40% 노이즈에서 검증됨
- **제안**: 합성 노이즈(symmetric/asymmetric)를 사용한 극단적 노이즈 환경 평가

**⑤ 멀티레이블 및 계층적 레이블 환경으로 확장**
- 현재 단일 레이블 분류에 한정
- **제안**: 다중 레이블, 계층적 레이블 구조에서의 MRF 클리크 재설계

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 관련 연구 비교

| 논문 | 연도 | 접근 방식 | 지도 필요 | 핵심 아이디어 | NoiseRank 대비 |
|------|------|-----------|-----------|---------------|----------------|
| **DivideMix** (Li et al.) | 2020 | 반지도 | 불필요 | GMM으로 클린/노이즈 분리, MixMatch 활용 | 더 높은 성능이나 특정 구조 의존 |
| **SELF** (Nguyen et al.) | 2020 | 비지도 | 불필요 | Self-ensemble 기반 레이블 정제 | 앙상블 비용 높음 |
| **ELR** (Liu et al.) | 2020 | 비지도 | 불필요 | Early Learning Regularization | 손실 함수 수정 필요 |
| **Confident Learning** (Northcutt et al.) | 2021 | 비지도 | 불필요 | 신뢰 구간 기반 노이즈 탐지 | 확률 추정에 의존, 해석성 유사 |
| **CORES** (Cheng et al.) | 2021 | 비지도 | 불필요 | 손실 값 기반 샘플 선택 | 손실 함수 수정 필요 |
| **SOP** (Liu et al.) | 2022 | 비지도 | 불필요 | 최적화 기반 노이즈 수정 | 더 높은 성능, 해석성 낮음 |
| **Cleanlab** (Northcutt et al.) | 2021 | 비지도 | 불필요 | Confident Learning의 소프트웨어 구현 | 실용적 도구로서 경쟁력 |

### 5.2 DivideMix와의 상세 비교

**DivideMix** (Li et al., ICLR 2020)는 NoiseRank와 같은 시기에 발표된 강력한 경쟁 방법입니다:

$$p(\text{clean} | \ell_i) = \frac{\mathcal{B}(\ell_i; \mu_{\text{clean}}, \sigma_{\text{clean}})}{\mathcal{B}(\ell_i; \mu_{\text{clean}}, \sigma_{\text{clean}}) + \mathcal{B}(\ell_i; \mu_{\text{noisy}}, \sigma_{\text{noisy}})}$$

| 비교 항목 | NoiseRank | DivideMix |
|-----------|-----------|-----------|
| 아키텍처 독립성 | ✅ 높음 | ❌ 낮음 (MixMatch 필요) |
| 해석 가능성 | ✅ 높음 | ❌ 낮음 |
| 성능 (CIFAR-10, 90% 노이즈) | 미검증 | **~93%** |
| 도메인 일반성 | ✅ 높음 | ❌ 제한적 |

### 5.3 Confident Learning과의 비교

**Confident Learning** (Northcutt et al., JAIR 2021)은 NoiseRank와 유사하게 비지도 방식으로 노이즈를 탐지합니다:

$$\hat{C}_{y \tilde{y}} = \sum_{i: \hat{y}_i = y} \mathbb{1}[\tilde{y}_i = \tilde{y}, p(\tilde{y}|x_i; \theta) \geq t_{\tilde{y}}]$$

- **공통점**: 비지도, 아키텍처 독립적
- **차이점**: Confident Learning은 소프트맥스 출력 기반, NoiseRank는 MRF 기반 의존성 모델 사용

### 5.4 Foundation Model 시대에서의 NoiseRank의 위치

2022년 이후 CLIP, ALIGN 등 대규모 멀티모달 모델이 등장하면서:
- 웹 크롤링 데이터의 노이즈 처리 중요성 더욱 증가
- NoiseRank의 **모달리티 독립성**이 멀티모달 데이터 정제에 직접 적용 가능
- LAION-5B 등 대규모 데이터셋에 대한 확장성 검증이 필요

---

## 참고 자료

**주 논문:**
- Sharma, K., Donmez, P., Luo, E., Liu, Y., & Yalniz, I. Z. (2020). *NoiseRank: Unsupervised Label Noise Reduction with Dependence Models*. arXiv:2003.06729v1

**비교 분석에 활용된 관련 논문:**
- Li, J., Socher, R., & Hoi, S. C. (2020). *DivideMix: Learning with Noisy Labels as Semi-supervised Learning*. ICLR 2020.
- Northcutt, C. G., Jiang, L., & Chuang, I. L. (2021). *Confident Learning: Estimating Uncertainty in Dataset Labels*. JAIR 70.
- Liu, S., Niles-Weed, J., Razavian, N., & Fernandez-Granda, C. (2020). *Early-Learning Regularization Prevents Memorization of Noisy Labels*. NeurIPS 2020.
- Liu, Y., Bai, Y., Gao, S., & Zhu, J. (2022). *Self-Supervised Label Correction for Training with Noisy Labels*. NeurIPS 2022.
- Lee, K. H., He, X., Zhang, L., & Yang, L. (2018). *CleanNet: Transfer Learning for Scalable Image Classifier Training with Label Noise*. CVPR 2018.
- Han, J., Luo, P., & Wang, X. (2019). *Deep Self-Learning from Noisy Labels*. ICCV 2019.
- Metzler, D., & Croft, W. B. (2005). *A Markov Random Field Model for Term Dependencies*. SIGIR 2005.
