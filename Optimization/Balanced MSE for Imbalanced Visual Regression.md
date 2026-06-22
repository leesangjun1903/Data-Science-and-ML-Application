# Balanced MSE for Imbalanced Visual Regression

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문은 회귀(Regression) 태스크에서 가장 널리 사용되는 손실 함수인 **Mean Square Error(MSE)가 불균형 레이블 분포(Imbalanced Label Distribution) 상황에서 구조적으로 비효율적**임을 통계적 관점에서 규명하고, 이를 해결하는 새로운 손실 함수 **Balanced MSE**를 제안합니다.

핵심 통찰은 다음과 같습니다:

> MSE를 최소화하는 것은 훈련 분포 $p_{\text{train}}(\boldsymbol{y}|\boldsymbol{x})$의 NLL(Negative Log-Likelihood)을 최소화하는 것과 동치이며, 이는 테스트 분포 $p_{\text{bal}}(\boldsymbol{y}|\boldsymbol{x})$와의 분포 불일치(Distribution Mismatch)를 야기한다.

### 주요 기여 (Three-fold Contributions)

| 기여 | 내용 |
|------|------|
| **①** | MSE의 불균형 회귀에서의 비효율성을 통계적으로 규명하고, Balanced MSE 제안 |
| **②** | 다양한 실세계 시나리오를 위한 구현 옵션(GAI, BMC, BNI) 설계 — 특히 BMC는 사전 분포 지식 불필요 |
| **③** | 최초의 고차원(Multi-dimensional) 불균형 회귀 벤치마크 **IHMR** 제안 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 문제 정의

실세계 시각적 회귀 태스크(나이 추정, 자세 추정, 깊이 추정 등)에서 훈련 데이터의 레이블 분포는 심각하게 편향(Skewed)되어 있습니다. 예를 들어, 나이 추정 데이터셋에서는 성인 데이터가 압도적으로 많고, 어린이나 노인 데이터는 희소합니다.

**수학적 문제 정의:**

- 훈련 분포: $p_{\text{train}}(\boldsymbol{x}, \boldsymbol{y})$ — 레이블 분포 $p_{\text{train}}(\boldsymbol{y})$가 편향(Skewed)
- 테스트 분포: $p_{\text{bal}}(\boldsymbol{x}, \boldsymbol{y})$ — 레이블 분포 $p_{\text{bal}}(\boldsymbol{y})$가 균일(Uniform)
- 가정: 레이블 조건부 분포 $p(\boldsymbol{x}|\boldsymbol{y})$는 훈련/테스트 동일

**목표:** $p_{\text{train}}(\boldsymbol{y}|\boldsymbol{x})$가 아닌 $p_{\text{bal}}(\boldsymbol{y}|\boldsymbol{x})$를 학습

#### MSE의 한계 (통계적 분석)

MSE는 예측 분포를 가우시안으로 모델링할 때의 NLL과 동치입니다:

$$\text{MSE}(\boldsymbol{y}, \boldsymbol{y}_{\text{pred}}) = \|\boldsymbol{y} - \boldsymbol{y}_{\text{pred}}\|_2^2 \quad (3.1)$$

$$p(\boldsymbol{y}|\boldsymbol{x};\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{y}; \boldsymbol{y}_{\text{pred}}, \sigma_{\text{noise}}^2 \mathbf{I}) \quad (3.2)$$

베이즈 정리(Bayes' Rule)에 의해:

$$\frac{p_{\text{train}}(\boldsymbol{y}|\boldsymbol{x})}{p_{\text{bal}}(\boldsymbol{y}|\boldsymbol{x})} \propto \frac{p_{\text{train}}(\boldsymbol{y})}{p_{\text{bal}}(\boldsymbol{y})} \quad (3.3)$$

이 식은 희소 레이블($p_{\text{train}}(\boldsymbol{y})$가 낮은 경우) 에서 **MSE로 훈련된 회귀기는 체계적으로 과소 추정(Underestimate)** 함을 의미합니다.

기존 해결책인 **재가중치 기법(Reweighting)**도 불충분합니다: 훈련 분포가 심하게 편향될수록 오차가 급격히 커지고, 노이즈에도 민감하게 반응합니다.

---

### 2.2 제안 방법 (수식 포함)

#### 핵심 이론: 통계적 변환 정리 (Theorem 1)

$$p_{\text{train}}(\boldsymbol{y}|\boldsymbol{x}) = \frac{p_{\text{bal}}(\boldsymbol{y}|\boldsymbol{x}) \cdot p_{\text{train}}(\boldsymbol{y})}{\int_Y p_{\text{bal}}(\boldsymbol{y}'|\boldsymbol{x}) \cdot p_{\text{train}}(\boldsymbol{y}') \, d\boldsymbol{y}'} \quad (3.4)$$

**증명 개요:** Bayes 정리를 적용하면:

$$p_{\text{train}}(\boldsymbol{y}|\boldsymbol{x}) = p_{\text{bal}}(\boldsymbol{y}|\boldsymbol{x}) \cdot \frac{p_{\text{train}}(\boldsymbol{y})}{p_{\text{bal}}(\boldsymbol{y})} \cdot \frac{p_{\text{bal}}(\boldsymbol{x})}{p_{\text{train}}(\boldsymbol{x})}$$

여기서 미지의 증거비율(Evidence Ratio) $\frac{p_{\text{bal}}(\boldsymbol{x})}{p_{\text{train}}(\boldsymbol{x})}$는 $\int_Y p_{\text{train}}(\boldsymbol{y}|\boldsymbol{x})\,d\boldsymbol{y} = 1$ 조건으로 소거됩니다.

#### Balanced MSE 정의 (Definition 3.1)

회귀기가 직접 $p_{\text{bal}}(\boldsymbol{y}|\boldsymbol{x};\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{y};\boldsymbol{y}\_{\text{pred}}, \sigma_{\text{noise}}^2\mathbf{I})$를 예측하도록 하고, Theorem 1을 통해 $p_{\text{train}}(\boldsymbol{y}|\boldsymbol{x};\boldsymbol{\theta})$로 변환한 후 NLL을 계산합니다:

```math
\begin{aligned}
L &= -\log p_{\text{train}}(\boldsymbol{y}|\boldsymbol{x};\boldsymbol{\theta}) \\
&= -\log \frac{p_{\text{bal}}(\boldsymbol{y}|\boldsymbol{x};\boldsymbol{\theta}) \cdot p_{\text{train}}(\boldsymbol{y})}{\int_Y p_{\text{bal}}(\boldsymbol{y}'|\boldsymbol{x};\boldsymbol{\theta}) \cdot p_{\text{train}}(\boldsymbol{y}') \, d\boldsymbol{y}'} \\
&\cong -\log \mathcal{N}(\boldsymbol{y};\boldsymbol{y}_{\text{pred}}, \sigma_{\text{noise}}^2\mathbf{I}) \\
&\quad + \log \int_Y \mathcal{N}(\boldsymbol{y}';\boldsymbol{y}_{\text{pred}}, \sigma_{\text{noise}}^2\mathbf{I}) \cdot p_{\text{train}}(\boldsymbol{y}') \, d\boldsymbol{y}'
\end{aligned} \quad (3.6)
```

> **해석:** 첫 번째 항은 표준 MSE와 동치이며, 두 번째 항이 **균형화 항(Balancing Term)**으로 불균형을 보정합니다. $p_{\text{train}}(\boldsymbol{y})$가 균일하면 두 번째 항이 상수가 되어 표준 MSE로 환원됩니다.

---

### 2.3 구현 옵션 (Implementation Options)

#### ① GMM 기반 해석적 적분 (GAI: GMM-based Analytical Integration)

$p_{\text{train}}(\boldsymbol{y})$를 가우시안 혼합 모델(GMM)로 모델링:

$$p_{\text{train}}(\boldsymbol{y}) = \sum_{i=1}^K \phi_i \mathcal{N}(\boldsymbol{y};\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) \quad (3.10)$$

두 가우시안의 곱이 비정규화 가우시안임을 이용하면 최종 손실:

$$L = -\log \mathcal{N}(\boldsymbol{y};\boldsymbol{y}_{\text{pred}}, \sigma_{\text{noise}}^2\mathbf{I}) + \log \sum_{i=1}^K \phi_i \cdot \mathcal{N}(\boldsymbol{y}_{\text{pred}};\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i + \sigma_{\text{noise}}^2\mathbf{I}) \quad (3.12)$$

**장점:** 닫힌 형태(Closed-form), 다차원 레이블 공간 지원

#### ② 배치 기반 몬테카를로 (BMC: Batch-based Monte-Carlo)

$p_{\text{train}}(\boldsymbol{y})$에 대한 사전 지식 **불필요** — 배치 내 레이블을 $p_{\text{train}}(\boldsymbol{y})$의 샘플로 간주:

$$L = -\log \mathcal{N}(\boldsymbol{y};\boldsymbol{y}_{\text{pred}},\sigma_{\text{noise}}^2\mathbf{I}) + \log \sum_{i=1}^N \mathcal{N}(\boldsymbol{y}_{(i)};\boldsymbol{y}_{\text{pred}}, \sigma_{\text{noise}}^2\mathbf{I}) \quad (3.14)$$

Softmax with temperature 형태로 재작성:

$$L = -\log \frac{\exp(-\|\boldsymbol{y}_{\text{pred}} - \boldsymbol{y}\|_2^2 / \tau)}{\sum_{\boldsymbol{y}' \in B_{\boldsymbol{y}}} \exp(-\|\boldsymbol{y}_{\text{pred}} - \boldsymbol{y}'\|_2^2 / \tau)} \quad (3.15)$$

여기서 $\tau = 2\sigma_{\text{noise}}^2$는 온도 계수(Temperature Coefficient)입니다.

> **흥미로운 점:** BMC 형태는 배치 내 분류(Batch Classification)와 동치이며, 자기지도 학습(Self-supervised Learning)의 대조 손실(Contrastive Loss)과 유사한 구조를 가집니다.

#### ③ 빈 기반 수치 적분 (BNI: Bin-based Numerical Integration)

KDE(Kernel Density Estimation)로 추정된 $p_{\text{train}}(\boldsymbol{y})$를 활용:

$$L = -\log \mathcal{N}(\boldsymbol{y};\boldsymbol{y}_{\text{pred}},\sigma_{\text{noise}}^2\mathbf{I}) + \log \sum_{i=1}^N p_{\text{train}}(\boldsymbol{y}_{(i)}) \cdot \mathcal{N}(\boldsymbol{y}_{(i)};\boldsymbol{y}_{\text{pred}}, \sigma_{\text{noise}}^2\mathbf{I}) \quad (3.16)$$

주로 1차원 레이블 공간에 적용, 기존 KDE 기반 방법과 호환.

#### ④ 최적 노이즈 스케일 학습

$\sigma_{\text{noise}}$를 학습 가능한 파라미터로 설정하여 $\boldsymbol{y}_{\text{pred}}$와 공동 최적화(Joint Optimization)합니다. 추가적인 하이퍼파라미터 탐색 없이 거의 최적 성능 달성이 가능합니다.

---

### 2.4 불균형 분류와의 연결

Theorem 1은 분류 문제에도 적용됩니다. 이산 레이블 공간 $Y$에서 Softmax를 사용하면:

$$p_{\text{train}}(y|\boldsymbol{x};\boldsymbol{\theta}) = \frac{\exp(\eta[y]) \cdot p_{\text{train}}(y)}{\sum_{y' \in Y} \exp(\eta[y']) \cdot p_{\text{train}}(y')} \quad (3.9)$$

이는 불균형 분류 문헌의 **Logit Adjustment** 기법(Menon et al., 2021)과 동일한 형태입니다. **Balanced MSE와 Logit Adjustment는 Theorem 1의 두 가지 다른 인스턴스화**임을 보임으로써, 불균형 분류와 회귀를 통합적 통계 프레임워크로 설명하는 최초의 작업입니다.

---

### 2.5 모델 구조

Balanced MSE는 **손실 함수 수준의 개입**이므로, 특정 모델 아키텍처를 요구하지 않습니다. 실험에서 사용된 백본은 다음과 같습니다:

| 태스크 | 백본 | 비고 |
|--------|------|------|
| 나이 추정 (IMDB-WIKI-DIR) | ResNet-50 | 마지막 선형 레이어 재훈련 (RRT 방식) |
| 깊이 추정 (NYUD2-DIR) | ResNet-50 기반 Encoder-Decoder | Hu et al., 2019 아키텍처 |
| 인체 메쉬 복원 (IHMR) | SPIN (사전학습) + 선형 회귀기 | SMPL 파라미터 ($\boldsymbol{\theta} \in \mathbb{R}^{24\times3}$, $\boldsymbol{\beta} \in \mathbb{R}^{10}$) |

---

### 2.6 성능 향상

#### 나이 추정 (IMDB-WIKI-DIR)

| 방법 | bMAE↓ (All) | bMAE↓ (Few) | MAE↓ (All) |
|------|------------|------------|-----------|
| Vanilla | 13.92 | 32.78 | 8.06 |
| RRT+LDS | 13.09 | 30.26 | **7.79** |
| **Ours (BMC)** | 12.69 | 28.28 | 8.08 |
| **Ours (GAI)** | **12.66** | **28.14** | 8.12 |

희소 레이블(Few) 그룹에서 **~2.1 bMAE 향상**, 나이 20세 미만 및 70세 이상에서 특히 두드러진 개선.

#### 깊이 추정 (NYUD2-DIR)

| 방법 | RMSE↓ (All) | RMSE↓ (Few) | $\delta_1$↑ (Few) |
|------|------------|------------|-----------------|
| Vanilla+LDS | 1.387 | 1.954 | 0.630 |
| **Ours (BNI)** | 1.283 | 1.736 | **0.723** |
| **Ours (GAI)** | **1.251** | **1.703** | 0.715 |

#### 불균형 인체 메쉬 복원 (IHMR)

| 방법 | bMPVPE↓ (All) | bMPJPE↓ (All) | bPA-MPJPE↓ (All) |
|------|--------------|--------------|----------------|
| SPIN-RT | 116.1 | 99.58 | 66.53 |
| **Ours (BMC)** | 113.9 | 97.87 | 65.90 |
| **Ours (GAI)** | **112.7** | **96.70** | **64.69** |

---

### 2.7 한계점

논문에서 직접 인정한 한계 및 추가 분석된 한계:

1. **레이블 불균형만 해결**: 데이터셋 내 다른 편향(예: 인구통계적 편향, 측정 편향)은 해결하지 못함
2. **등방성 가우시안 노이즈 가정**: 이방성(Anisotropic) 또는 입력 의존적(Input-dependent) 노이즈는 고려하지 않아, 깊이 추정처럼 픽셀 간 의존성이 있는 경우 BMC 정확도 저하
3. **적분 계산의 근사**: GMM 이외의 복잡한 분포에서 GAI 사용 시 분포 모델링의 표현력 제한
4. **BMC의 배치 크기 의존성**: 작은 배치에서 $p_{\text{train}}(\boldsymbol{y})$ 추정 분산 증가
5. **L1/Huber 손실 확장 미완**: Theorem 1을 라플라시안 등 다른 분포로 확장하는 것은 향후 과제로 남김

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상의 이론적 근거

Balanced MSE가 일반화 성능을 향상시키는 핵심 메커니즘은 **Bayes-Optimal 예측으로의 수렴**입니다:

$$\boldsymbol{y}_{\text{pred}} = \arg\max_{\boldsymbol{y}} \mathcal{N}(\boldsymbol{y};\boldsymbol{y}_{\text{pred}},\sigma_{\text{noise}}^2\mathbf{I}) = \arg\max_{\boldsymbol{y}} p_{\text{bal}}(\boldsymbol{y}|\boldsymbol{x};\boldsymbol{\theta})$$

이는 균형 테스트 세트에 대해 **Bayes-Optimal 예측임이 정의에 의해 보장**됩니다.

### 3.2 분포 편향 제거를 통한 일반화

$$\frac{p_{\text{train}}(\boldsymbol{y}|\boldsymbol{x})}{p_{\text{bal}}(\boldsymbol{y}|\boldsymbol{x})} \propto \frac{p_{\text{train}}(\boldsymbol{y})}{p_{\text{bal}}(\boldsymbol{y})}$$

이 비율이 레이블 $\boldsymbol{y}$에 따라 다르기 때문에, 표준 MSE로 훈련된 모델은 **빈번한 레이블에 편향된(Biased) 결정 경계**를 학습합니다. Balanced MSE는 이 편향을 통계적으로 보정하여, 훈련 데이터의 레이블 분포와 무관하게 $p_{\text{bal}}(\boldsymbol{y}|\boldsymbol{x})$를 추정합니다.

### 3.3 분포 이동(Distribution Shift) 강인성

실험 결과에서 확인된 Balanced MSE의 일반화 특성:

- **훈련 분포 왜도(Skewness) 불변성**: 정규 분포와 지수 분포 모두에서, 왜도가 증가해도 성능이 안정적으로 유지됨 (Fig. 2)
- **비선형 회귀에서의 일관된 효과**: $y=\tan(x)$, $y=x^2$, $y=e^x$, $y=\log(x)$ 등 다양한 비선형 함수에서 오라클에 가장 근접 (Fig. 4)
- **랜덤 시드 강인성**: 재가중치는 랜덤 시드에 따라 성능 변동이 크지만, Balanced MSE는 안정적 (Fig. 7)

### 3.4 고차원 일반화 (최초)

기존 불균형 회귀 연구($d=1$)와 달리, Balanced MSE는 $d>1$의 다차원 레이블 공간을 지원합니다:

- IHMR 벤치마크: $\boldsymbol{\theta} \in \mathbb{R}^{24\times3}$, $\boldsymbol{\beta} \in \mathbb{R}^{10}$의 SMPL 파라미터
- 2D 회귀 실험에서 균일 주변 분포(Marginal Distribution) 달성 확인

### 3.5 보완적 기법과의 시너지

- **FDS(Feature Distribution Smoothing)**와 직교적(Orthogonal)으로 결합 가능
- **PM-Net** (프로토타입 메모리 방식)과 상호보완적 관계: PM-Net이 회귀 초기화를 개선하면, Balanced MSE는 훈련 목표를 균형화

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

#### ① 통합 이론 프레임워크 제공

Theorem 1은 불균형 분류와 회귀를 **단일 통계 프레임워크**로 통합합니다. 이는 분류 문헌의 풍부한 기법(Margin-based Methods, Mixup, Decoupling 등)을 회귀 문제로 체계적으로 이전(Transfer)하는 이론적 기반을 제공합니다.

#### ② 손실 함수 재설계의 방향성 제시

- L1 손실 → 라플라시안 분포 가정 → Balanced L1
- Huber 손실 → 혼합 분포 가정 → Balanced Huber  
  (Meyer, 2021의 허버 손실 확률적 해석 활용 가능)

#### ③ 자기지도 학습과의 연결

BMC의 형태(Eq. 3.15)가 대조 학습(Contrastive Learning)의 InfoNCE 손실과 구조적으로 유사하여, **불균형 자기지도 학습** 연구로 확장 가능성 있음.

#### ④ 공정성(Fairness) 연구에 기여

연속형 레이블의 불균형 문제를 다루므로, 나이·키·체중 등 민감한 연속 속성에 관한 **알고리즘적 공정성** 연구에 직접 적용 가능.

#### ⑤ 다중 모달(Multi-modal) 회귀로의 확장

Theorem 1의 일반성 덕분에, 조건부 분포가 단봉(Unimodal) 가우시안이 아닌 **혼합/복잡 분포**로 모델링되는 경우(예: 생성 모델 기반 회귀)에도 확장 연구가 가능합니다.

---

### 4.2 앞으로 연구 시 고려할 점

#### ① 노이즈 모델 정교화

현재는 등방성 가우시안 노이즈 $\epsilon \sim \mathcal{N}(0, \sigma_{\text{noise}}^2\mathbf{I})$를 가정하지만:

- **이방성 공분산**: $\epsilon \sim \mathcal{N}(0, \boldsymbol{\Sigma}_{\text{noise}})$ — 레이블 차원 간 상관관계 반영
- **입력 의존적 노이즈**: $\sigma_{\text{noise}} = f_\phi(\boldsymbol{x})$ — 이분산성(Heteroscedasticity) 처리
- **비가우시안 노이즈**: 라플라시안, 스튜던트-t 분포 등 이상치 강인성 확보

#### ② 레이블 분포 추정의 정확성

GAI 사용 시 GMM으로 $p_{\text{train}}(\boldsymbol{y})$를 표현하는 제약이 있습니다. 고차원·복잡 분포에서:

- **Normalizing Flows** 활용으로 더 유연한 분포 표현
- **VAE/GAN 기반 생성 모델**로 $p_{\text{train}}(\boldsymbol{y})$ 모델링 (논문에서도 미래 방향으로 제시)

#### ③ 테스트 시간 분포 이동 대응

현재 방법은 훈련 시 $p_{\text{train}}(\boldsymbol{y})$를 활용하지만, 배포 환경(Deployment)에서 레이블 분포가 다를 경우에 대한 대응 전략이 필요합니다.

#### ④ 배치 크기 및 계산 효율성

BMC는 배치 내 $N^2$ 쌍의 거리 계산이 필요하므로, 대용량 배치 또는 고차원 레이블에서 계산 비용이 증가할 수 있습니다. 효율적인 근사 방법 연구가 필요합니다.

#### ⑤ 레이블 불균형 외 편향 처리

논문도 인정하듯이, 실세계 데이터에는 레이블 불균형 이외의 편향(예: 데이터 수집 편향, 확인 편향)이 존재합니다. Balanced MSE와 **인과적 추론(Causal Inference)** 기법의 결합이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 불균형 분류 관련 선행/동시대 연구

| 논문 | 방법 | Balanced MSE와의 관계 |
|------|------|----------------------|
| **Menon et al. (ICLR 2021)** "Long-tail learning via logit adjustment" | 로짓에 $\log p_{\text{train}}(y)$ 오프셋 추가 | Theorem 1의 분류 특수 케이스와 동치 — 통합 이론으로 설명 가능 |
| **Hong et al. (CVPR 2021)** "Disentangling label distribution for long-tailed visual recognition" | 레이블 분포 분리 학습 | 분류 문제에 특화; Balanced MSE는 연속 레이블로 확장 |
| **Wang et al. (CVPR 2021)** "Seesaw Loss for long-tailed instance segmentation" | 정분류율 기반 동적 재가중치 | 이산 레이블 공간에만 적용; Balanced MSE는 연속 레이블 처리 |

### 5.2 불균형 회귀 관련 동시대 연구

| 논문 | 방법 | Balanced MSE와의 비교 |
|------|------|----------------------|
| **Yang et al. (ICML 2021)** "Delving into deep imbalanced regression (DIR)" | LDS(Label Distribution Smoothing), FDS(Feature Distribution Smoothing), KDE 재가중치 | Balanced MSE의 직접 비교 대상; Balanced MSE가 bMAE 기준 일관되게 우수 |
| **Steininger et al. (Machine Learning, 2021)** "Density-based weighting for imbalanced regression" | KDE 기반 손실 재가중치 | 재가중치의 한계를 Balanced MSE 논문이 합성 벤치마크에서 직접 반증 |
| **Rong et al. (arXiv 2020)** "Chasing the tail in monocular 3D human reconstruction with prototype memory (PM-Net)" | 프로토타입 메모리로 희소 자세 초기화 | IHMR 벤치마크에서 Balanced MSE의 GAI가 전반적 bMPVPE에서 우수; PM-Net은 극단 tail-5%에서 강점 → 상호보완적 |

### 5.3 Balanced MSE 이후 관련 연구 방향

논문 발표(2022년 CVPR) 이후 영향을 받은 연구 방향들:

- **불균형 회귀의 대조 학습 통합**: BMC의 대조 손실 유사성에 착안한 후속 연구 가능성
- **의료 영상 분야 적용**: 희소 병리 데이터의 연속적 중증도 추정에 활용 가능
- **자율주행**: 극단적 기상 조건(드문 레이블)의 깊이/거리 추정 개선

> **주의:** 2022년 이후 Balanced MSE를 직접 인용·확장한 특정 논문들의 세부 내용은 제공된 논문 원문의 범위를 벗어나므로, 구체적 인용 수치나 후속 논문 제목을 명시하기 어렵습니다. 위 내용은 논문 내 인용 분석과 연구 흐름에 기반합니다.

---

## 참고 자료

1. **Jiawei Ren, Mingyuan Zhang, Cunjun Yu, Ziwei Liu.** "Balanced MSE for Imbalanced Visual Regression." *CVPR 2022.* arXiv:2203.16427 ← **본 분석의 주요 출처**

2. **Yuzhe Yang, Kaiwen Zha, Ying-Cong Chen, Hao Wang, Dina Katabi.** "Delving into Deep Imbalanced Regression." *ICML 2021.*

3. **Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, Sanjiv Kumar.** "Long-tail learning via logit adjustment." *ICLR 2021.*

4. **Michael Steininger, Konstantin Kobs, Padraig Davidson, Anna Krause, Andreas Hotho.** "Density-based weighting for imbalanced regression." *Machine Learning, 2021.*

5. **Yu Rong, Ziwei Liu, Chen Change Loy.** "Chasing the tail in monocular 3D human reconstruction with prototype memory." *arXiv:2012.14739, 2020.*

6. **Jonathon Byrd, Zachary Lipton.** "What is the effect of importance weighting in deep learning?" *ICML 2019.*

7. **Youngkyu Hong, Seungju Han, et al.** "Disentangling label distribution for long-tailed visual recognition." *CVPR 2021.*

8. **Gregory P. Meyer.** "An alternative probabilistic interpretation of the Huber loss." *CVPR 2021.*
