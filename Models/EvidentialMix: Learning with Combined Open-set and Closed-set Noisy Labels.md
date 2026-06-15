# EvidentialMix: Learning with Combined Open-set and Closed-set Noisy Labels 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

EvidentialMix(EDM)는 실제 데이터셋에서 **Open-set 노이즈**(훈련 레이블 집합에 없는 클래스의 샘플)와 **Closed-set 노이즈**(훈련 레이블 집합 내의 잘못된 레이블)가 **동시에 공존**한다는 현실을 반영하여, 이 두 가지 노이즈를 **통합적으로 처리**하는 최초의 체계적 방법론을 제안합니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **새로운 문제 정의** | Open-set + Closed-set 결합 노이즈 문제를 최초로 체계적으로 정의 |
| **새로운 벤치마크** | $\rho$, $\omega$ 두 변수로 노이즈 비율을 제어하는 벤치마크 평가 체계 |
| **EvidentialMix 알고리즘** | Subjective Logic(SL) 손실함수를 활용한 3분류(clean/closed/open) 분리 메커니즘 |
| **우수한 특징 표현** | t-SNE 분석을 통해 Open-set 샘플을 독립 클러스터로 분리하는 능력 입증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 연구의 한계:

- **Closed-set 전용 방법** (DivideMix, SELF): 모든 샘플이 훈련 클래스 중 하나에 속한다고 가정 → Open-set 노이즈에 취약
- **Open-set 전용 방법** (ILON): 노이즈 샘플의 가중치만 감소 → Closed-set 노이즈의 유용한 정보 손실

**핵심 문제**: 두 종류의 노이즈가 공존할 때, 각 샘플의 노이즈 유형을 **정확히 식별**하고 **차별적으로 처리**해야 함

### 2.2 문제 정의 (수식)

훈련 세트 $\mathcal{D} = \{(\mathbf{x}\_i, \mathbf{y}\_i)\}_{i=1}^{|\mathcal{D}|}$ 에서:

**Closed-set 노이즈** (노이즈율 $\zeta \in [0,1]$):

$$(\mathbf{x}_i, \mathbf{y}_i) \in \mathcal{D},\quad \mathbf{y}_i = \mathbf{y}_i^* \text{ with prob. } 1-\zeta, \quad \mathbf{y}_i \sim r(\mathcal{Y}, \theta_r) \text{ with prob. } \zeta$$

**Open-set 노이즈** (노이즈율 $\eta \in [0,1]$):

$$(\mathbf{x}_i, \mathbf{y}_i) \in \mathcal{D}' \text{ where } \mathcal{Y}' \cap \mathcal{Y} = \emptyset, \quad \mathbf{y}_i \sim r(\mathcal{Y}, \theta_r)$$

**결합 노이즈** ($\rho, \omega \in [0,1]$):

$$\begin{cases} 1 - \rho & \text{: clean samples from } \mathcal{D} \text{ with } \mathbf{y}_i = \mathbf{y}_i^* \\ \omega \times \rho & \text{: closed-set noisy samples from } \mathcal{D} \text{ with } \mathbf{y}_i \sim r(\mathcal{Y}, \theta_r) \\ (1-\omega) \times \rho & \text{: open-set noisy samples from } \mathcal{D}' \text{ with } \mathbf{y}_i \sim r(\mathcal{Y}, \theta_r) \end{cases}$$

이 설정은 $\omega \in \{0, 1\}$ 일 때 각각 순수 Open-set 또는 순수 Closed-set 문제로 귀결되어 **일반화된 프레임워크**를 구성합니다.

### 2.3 제안 방법: EvidentialMix (EDM)

#### 2.3.1 모델 구조

EDM은 두 네트워크를 동시에 학습합니다:

```
[NetS] ─── SL Loss → GMM 분류 → clean/closed-set/open-set 분리
              ↑                              ↓
           재레이블링 ← [NetD] ← MixMatch (clean + closed-set만 사용)
```

- **NetS** ( $f_{\theta^{(S)}}(c|\mathbf{x})$ ): Subjective Logic 손실로 훈련 → 불확실성 추정
- **NetD** ( $f_{\theta^{(D)}}(c|\mathbf{x})$ ): DivideMix 손실 + MixMatch로 훈련 → 분류 수행

#### 2.3.2 Subjective Logic (SL) 손실함수

$$\mathcal{L}^{(S)} = \frac{1}{|\mathcal{D}|} \sum_{i=1}^{|\mathcal{D}|} \ell^{(S)}(\mathbf{x}_i, \mathbf{y}_i, \theta^{(S)})$$

$$\ell^{(S)}(\mathbf{x}_i, \mathbf{y}_i, \theta^{(S)}) = \sum_{c=1}^{|\mathcal{Y}|} \left( y_i(c) - \frac{\alpha_{ic}}{S_i} \right)^2 + \frac{\alpha_{ic}(S_i - \alpha_{ic})}{S_i^2(S_i + 1)}$$

여기서:
- $\alpha_{ic} = \varphi(f_{\theta^{(S)}}(c|\mathbf{x}_i)) + 1$ ($\varphi$: ReLU 활성화 함수)
- $S_i = \sum_{c=1}^{|\mathcal{Y}|} \alpha_{ic}$ (Dirichlet 분포의 농도 파라미터 합)

**SL 손실의 핵심 특성**:

| 샘플 유형 | 모델 출력 | 손실값 |
|-----------|-----------|--------|
| **Clean** | 자신있고 올바른 예측 | 낮음 (low) |
| **Closed-set 노이즈** | 자신있지만 틀린 예측 | 높음 (high) |
| **Open-set 노이즈** | 불확실한 예측 | 중간 (mid) |

#### 2.3.3 GMM을 이용한 3분류

$\psi$-컴포넌트 GMM을 손실값 분포 $\{\ell^{(S)}(\mathbf{x}\_i, \mathbf{y}_i, \theta^{(S)})\}\_{i=1}^{|\mathcal{D}|}$에 적합:

$$w_i = p(\mathcal{G} | \ell^{(S)}(\mathbf{x}_i, \mathbf{y}_i, \theta^{(S)})) \quad \text{(clean, 평균} \leq \mu_{\min}\text{)}$$

$$w_i^{cl} = p(\mathcal{G}^{cl} | \ell^{(S)}(\mathbf{x}_i, \mathbf{y}_i, \theta^{(S)})) \quad \text{(closed-set, 평균} \geq \mu_{\max}\text{)}$$

$$w_i^{op} = p(\mathcal{G}^{op} | \ell^{(S)}(\mathbf{x}_i, \mathbf{y}_i, \theta^{(S)})) \quad \text{(open-set, 평균} \in (\mu_{\min}, \mu_{\max})\text{)}$$

구현 파라미터: $\psi = 20$, $\mu_{\min} = 0.3$, $\mu_{\max} = 0.7$

#### 2.3.4 NetD 훈련 손실 (DivideMix Loss)

$$\mathcal{L}^{(D)} = \mathcal{L}^{(\mathcal{X})} + \lambda^{(U)}\mathcal{L}^{(\mathcal{U})} + \lambda^{(reg)}\mathcal{L}^{(reg)}$$

$$\mathcal{L}^{(\mathcal{X})} = -\frac{1}{|\mathcal{X}'|} \sum_{(\hat{\mathbf{x}}, \hat{\mathbf{y}}) \in \mathcal{X}'} \sum_{c=1}^{|\mathcal{Y}|} \hat{y}(c) \log(p_{\theta^{(D)}}(c|\hat{\mathbf{x}}))$$

$$\mathcal{L}^{(\mathcal{U})} = \frac{1}{|\mathcal{U}'|} \sum_{(\hat{\mathbf{u}}, \hat{\mathbf{q}}) \in \mathcal{U}'} \|\hat{\mathbf{q}} - p_{\theta^{(D)}}(:|\hat{\mathbf{u}})\|_2^2$$

$$\mathcal{L}^{(reg)} = \sum_{c=1}^{|\mathcal{Y}|} \frac{1}{|\mathcal{Y}|} \log \left( \frac{1}{|\mathcal{X}'| + |\mathcal{U}'|} \sum_{\mathbf{x} \in (\mathcal{X}' \cup \mathcal{U}')} p_{\theta^{(D)}}(c|\mathbf{x}) \right)$$

#### 2.3.5 NetS의 재레이블링

$$\hat{c}_i = \arg\max_{c \in \mathcal{Y}} \left[ (w_i^{cl}) p_{\theta^{(D)}}(c|\mathbf{x}_i) + (1 - w_i^{cl}) \mathbf{y}_i(c) \right]$$

$$\hat{\mathbf{y}}_i = \text{onehot}(\hat{c}_i)$$

**추론 시**: $c^* = \arg\max_{c \in \mathcal{Y}} p_{\theta^{(D)}}(c|\mathbf{x})$ (NetD만 사용)

### 2.4 성능 향상

**Table 1 요약** (CIFAR-10 기반, CIFAR-100/ImageNet32를 Open-set으로 사용):

| 방법 | $\rho=0.3$, ImageNet32 | $\rho=0.6$, ImageNet32 |
|------|----------------------|----------------------|
| RoG | 89.5~91.9% | 82.9~87.8% |
| ILON | 85.8~91.8% | 77.3~87.7% |
| DivideMix | 92.4~94.3% | 92.5~94.7% |
| **EDM (Ours)** | **93.2~95.2%** | **91.2~94.1%** |

- 20개 노이즈 설정 중 **17개에서 최고 성능** 달성
- 일부 설정에서 차선 대비 **3% 이상 향상**

### 2.5 한계

1. **$\rho = 0.6$, $\omega = 0$ (순수 Open-set, 고노이즈)** 에서 DivideMix 대비 성능 열세 (91.2% vs 92.5%)
2. **비대칭(asymmetric) 노이즈 미지원**: 비대칭 Closed-set 및 비대칭 Open-set 노이즈 처리 미검토
3. **의미론적(semantic) 노이즈 미고려**: 시각적으로 유사한 클래스 간 혼동 노이즈 처리 부재
4. **GMM 하이퍼파라미터 민감성**: $\psi=20$, $\mu_{\min}=0.3$, $\mu_{\max}=0.7$ 설정이 데이터셋에 따라 최적이 아닐 수 있음
5. **두 네트워크 동시 훈련**으로 인한 계산 비용 증가
6. **실제 대규모 데이터셋 검증 부족**: 소규모 합성 노이즈 실험에 한정

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 일반화를 높이는 핵심 메커니즘

#### (A) Open-set 샘플의 훈련 배제로 인한 과적합 방지

Open-set 샘플은 훈련 클래스에 속하지 않으므로, 이를 강제로 특정 클래스에 매핑하면 결정 경계(decision boundary)가 왜곡됩니다. EDM은 이를 **훈련에서 제외**하여 순수한 클래스 경계 학습을 보장합니다:

$$\mathcal{X} = \{(\mathbf{x}_i, \mathbf{y}_i, w_i) \mid w_i > \max(w_i^{op}, w_i^{cl})\}$$

→ Open-set 오염 없이 깨끗한 결정 경계 학습 가능

#### (B) MixMatch를 통한 데이터 효율성 향상

Closed-set 노이즈 샘플을 **레이블 없는(unlabeled) 데이터**처럼 활용하여 반지도학습:

$$\hat{\mathbf{y}}_b = \text{TempSharpen}_T(w_b \mathbf{y}_b + (1 - w_b)\mathbf{p}_b)$$

$w_b$를 가중치로 원본 레이블과 모델 예측을 보간 → 노이즈 레이블의 영향을 완화하면서도 유용한 특징 정보 활용

#### (C) 우수한 특징 공간 구조

t-SNE 분석 결과 EDM은:
- 각 알려진 클래스에 대한 **독립적이고 명확한 클러스터** 형성
- Open-set 샘플을 **별도 클러스터**로 분리
- DivideMix, ILON과 달리 Open-set 샘플의 클래스 영역 침투 방지

이는 **전이학습(transfer learning)** 시나리오에서도 더 풍부하고 순수한 특징 표현 제공 가능성을 의미합니다.

#### (D) 불확실성 정량화를 통한 신뢰성 있는 예측

Dirichlet 분포 기반 SL 손실은 단순 점 추정(point estimate)이 아닌 **예측 분포**를 학습:

$$\alpha_{ic} = \varphi(f_{\theta^{(S)}}(c|\mathbf{x}_i)) + 1, \quad S_i = \sum_{c=1}^{|\mathcal{Y}|} \alpha_{ic}$$

$S_i$가 클수록 높은 확신, 작을수록 높은 불확실성 → Out-of-distribution 감지에 활용 가능하여 배포 환경(deployment)에서의 일반화 신뢰도 향상

#### (E) 정규화 손실의 역할

$$\mathcal{L}^{(reg)} = \sum_{c=1}^{|\mathcal{Y}|} \frac{1}{|\mathcal{Y}|} \log \left( \frac{1}{|\mathcal{X}'| + |\mathcal{U}'|} \sum_{\mathbf{x}} p_{\theta^{(D)}}(c|\mathbf{x}) \right)$$

예측 확률의 균일 분포를 장려하여 **클래스 불균형 과적합 방지** 및 일반화 향상

### 3.2 일반화 향상의 실험적 증거

- **"Last" vs "Best" 정확도 차이 최소화**: EDM의 Last와 Best 정확도 차이가 RoG, ILON에 비해 훨씬 작음 → 훈련 후반부 안정성 = 일반화 능력
  - 예) CIFAR-100, $\rho=0.6$, $\omega=0.75$: EDM Best 93.7%, Last 93.4% (차이 0.3%) vs ILON Best 78.4%, Last 48.7% (차이 29.7%)

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (A) 새로운 문제 패러다임 정립
기존의 이분법적(Open-set vs. Closed-set) 노이즈 연구 패러다임을 넘어 **결합 노이즈(combined noise)**라는 새로운 연구 방향을 개척하였습니다. 향후 노이즈 레이블 연구에서 이 설정을 표준 평가 프로토콜로 채택할 가능성이 높습니다.

#### (B) 불확실성 정량화의 노이즈 처리 응용 확산
SL 손실을 통한 분류 불확실성 정량화가 노이즈 처리에 효과적임을 입증하여, **Bayesian Deep Learning**, **Evidential Neural Networks** 등 불확실성 기반 방법론의 노이즈 레이블 문제 응용을 촉진할 것으로 예상됩니다.

#### (C) 실용적 웹 크롤링 데이터 활용 연구 촉진
Google Images 등 상업적 검색엔진을 통한 데이터 수집 시 발생하는 실제 노이즈 환경과의 높은 유사성으로, **웹 스케일 약지도학습(weakly-supervised learning)** 연구에 직접적인 영향을 미칠 것입니다.

### 4.2 향후 연구 시 고려할 점

#### (A) 비대칭 노이즈로의 확장
논문 자체에서 미래 연구로 제안한 바와 같이, **비대칭(asymmetric) 노이즈** 및 **의미론적(semantic) 노이즈**를 결합 노이즈 프레임워크에 통합하는 연구가 필요합니다. 특히 CIFAR-10과 ImageNet 간의 노이즈 전이 행렬(noise transition matrix) 정의 방법론 개발이 과제입니다.

#### (B) GMM 대체 분리 방법 탐색
$\psi=20$의 GMM은 하이퍼파라미터에 민감하고 계산 비용이 있습니다. **Energy-based model**, **Normalizing Flow**, **KDE(Kernel Density Estimation)** 등 대안적 밀도 추정 방법과의 비교 연구가 필요합니다.

#### (C) Bayesian 불확실성 방법 통합
논문이 제안한 Bayesian 접근법 탐색과 관련하여, **MC-Dropout**, **Deep Ensemble**, **Stochastic Weight Averaging (SWA)** 등의 방법을 SL 손실 대신 활용하는 연구가 가치 있을 것입니다.

#### (D) 대규모 실제 데이터셋 검증
현재 실험이 CIFAR-10 기반 합성 노이즈에 한정되어 있어, **WebVision**, **Clothing-1M**, **ANIMAL-10N** 등 실제 노이즈 데이터셋에서의 검증이 필요합니다.

#### (E) 온라인/연속 학습 환경 적용
데이터가 스트림 형태로 유입되는 환경에서 Open-set/Closed-set 노이즈가 동적으로 변화하는 경우의 적응적 처리 방법 연구가 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래는 EvidentialMix의 논문 내 레퍼런스 및 연구 방향과 관련된 후속 연구들입니다. **단, 본 논문(arXiv:2011.05704v1)에서 직접 인용되지 않은 2020년 이후 논문들은 제가 직접 확인할 수 없으므로, 논문 내에서 확인된 정보 및 해당 분야의 주요 연구 흐름을 바탕으로 서술합니다.**

### 논문 내 확인된 관련 방법론과의 비교

| 방법 | 노이즈 유형 | 핵심 메커니즘 | EDM 대비 한계 |
|------|------------|-------------|-------------|
| **DivideMix** [Li et al., 2020] | Closed-set | 2-component GMM + MixMatch | Open-set 식별 불가 |
| **SELF** [Nguyen et al., 2019] | Closed-set | 앙상블 + 재레이블링 | Open-set 취약 |
| **ILON** [Wang et al., 2018] | Open-set | LOF 기반 가중치 조정 | Closed-set 정보 손실 |
| **RoG** [Lee et al., 2019] | Both (별도) | 생성 분류기 앙상블 | 결합 노이즈 미처리 |

### 연구 트렌드 분석 (논문 기반 추론)

1. **준지도학습(SSL) + 노이즈 처리 결합**: DivideMix → EDM 방향의 발전은 SSL 기반 접근법이 노이즈 레이블 문제의 주류 해결책임을 시사

2. **불확실성 정량화**: Evidential Deep Learning(Sensoy et al., 2018 [21])의 노이즈 처리 응용은 향후 **예측 신뢰도**를 노이즈 식별에 활용하는 연구의 선례

3. **다중 노이즈 유형 처리**: EDM이 제시한 결합 노이즈 프레임워크는 향후 더 다양한 노이즈 유형(instance-dependent noise, feature-dependent noise 등)을 통합하는 연구로 확장될 것으로 예상

---

## 참고 자료

본 답변은 아래 자료를 기반으로 작성되었습니다:

1. **Sachdeva, R., Cordeiro, F. R., Belagiannis, V., Reid, I., & Carneiro, G. (2020).** "EvidentialMix: Learning with Combined Open-set and Closed-set Noisy Labels." *arXiv:2011.05704v1*. (제공된 PDF 원문)

2. **Li, J., Socher, R., & Hoi, S. C. H. (2020).** "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." *arXiv:2002.07394* (논문 내 참조 [14])

3. **Wang, Y., Liu, W., Ma, X., Bailey, J., Zha, H., Song, L., & Xia, S. T. (2018).** "Iterative Learning with Open-set Noisy Labels." *arXiv:1804.00092* (논문 내 참조 [24])

4. **Sensoy, M., Kaplan, L., & Kandemir, M. (2018).** "Evidential Deep Learning to Quantify Classification Uncertainty." *NeurIPS 2018* (논문 내 참조 [21])

5. **Berthelot, D., Carlini, N., Goodfellow, I., Papernot, N., Oliver, A., & Raffel, C. (2019).** "MixMatch: A Holistic Approach to Semi-Supervised Learning." *arXiv:1905.02249* (논문 내 참조 [1])

6. **Lee, K., Yun, S., Lee, K., Lee, H., Li, B., & Shin, J. (2019).** "Robust Inference via Generative Classifiers for Handling Noisy Labels." *arXiv:1901.11300* (논문 내 참조 [13])

> ⚠️ **정확도 관련 안내**: 2020년 이후 EDM을 직접 비교한 후속 논문들(예: SOP, UNICON, ProMix 등)에 대한 상세 비교는 해당 논문들의 원문을 직접 확인할 수 없어 구체적 수치를 제시하지 않았습니다. 관련 최신 연구는 arXiv 및 NeurIPS/ICML/ICLR 2021-2024 proceedings에서 "open-set noisy labels", "combined noise" 키워드로 검색하시기를 권장합니다.
