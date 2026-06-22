# Delving into Deep Imbalanced Regression

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **Deep Imbalanced Regression (DIR)** 이라는 새로운 문제 설정을 공식적으로 정의하고, 연속형 타겟을 가진 불균형 데이터에서의 학습 문제가 기존 분류 기반 불균형 학습 방법으로는 근본적으로 해결될 수 없음을 실증적으로 보입니다.

핵심 주장은 다음과 같습니다:
- **연속형 레이블 공간에서는 경험적(empirical) 레이블 밀도 분포가 실제 불균형을 정확히 반영하지 못한다.**
- 인접한 타겟 값 사이의 유사성(continuity)을 명시적으로 활용하는 분포 평활화(distribution smoothing)가 연속 회귀 불균형 문제에 본질적으로 적합하다.

### 주요 기여 (4가지)

| 기여 | 내용 |
|------|------|
| 문제 정의 | DIR 태스크를 공식적으로 정의 |
| 방법론 제안 | LDS (Label Distribution Smoothing) + FDS (Feature Distribution Smoothing) |
| 벤치마크 구축 | 5개 대규모 DIR 데이터셋 공개 |
| 실험 검증 | 다양한 도메인(CV, NLP, 헬스케어)에서 우수한 성능 입증 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**DIR (Deep Imbalanced Regression)** 은 다음 세 가지 요소를 동시에 다룹니다:

1. **연속형 타겟의 불균형 데이터**에서 학습
2. **특정 타겟 값에 대한 결측 데이터(missing data)** 처리
3. **전체 타겟 범위**에 대한 일반화

기존 분류 기반 불균형 학습 방법의 한계:
- 클래스 간 hard boundary를 가정 → 연속 공간에 직접 적용 불가
- SMOTER, SMOGN 등은 고차원 데이터(이미지)에서 선형 보간이 의미 없는 샘플을 생성
- 연속 레이블의 유사성(인접 타겟 간 정보 공유)을 무시

**핵심 관찰:** CIFAR-100(분류)과 IMDB-WIKI(회귀)에 동일한 레이블 밀도 분포를 적용했을 때, 분류에서는 Pearson 상관계수 $-0.76$으로 오차가 밀도와 강하게 연동되지만, 회귀에서는 $-0.47$로 경험적 밀도가 실제 불균형을 반영하지 못함을 실증합니다.

---

### 2.2 제안하는 방법

#### 문제 세팅

훈련 데이터 $\{(\mathbf{x}\_i, y_i)\}_{i=1}^{N}$에서 $\mathbf{x}_i \in \mathbb{R}^d$, $y_i \in \mathbb{R}$ (연속형).  
레이블 공간 $\mathcal{Y}$를 $B$개의 등간격 빈(bin)으로 분할:

$$\mathcal{Y} = [y_0, y_1) \cup [y_1, y_2) \cup \cdots \cup [y_{B-1}, y_B)$$

딥 뉴럴 네트워크 $f(\mathbf{x};\theta)$를 통해 특징 $\mathbf{z} = f(\mathbf{x};\theta)$를 추출하고, 회귀 함수 $g(\cdot)$으로 최종 예측 $\hat{y} = g(\mathbf{z})$.

---

#### ✅ 방법 1: Label Distribution Smoothing (LDS)

**동기:** 연속 레이블 공간에서 인접한 레이블의 데이터는 정보를 공유하므로, 경험적 밀도 분포 $p(y)$는 실제 불균형을 과대/과소 표현한다.

**수식:** 대칭 커널(symmetric kernel) $k(y, y')$를 이용하여 효과적 레이블 밀도 분포를 추정:

$$\tilde{p}(y') \triangleq \int_{\mathcal{Y}} k(y, y') p(y) \, dy \tag{1}$$

- $p(y)$: 훈련 데이터의 레이블 $y$의 등장 횟수
- $\tilde{p}(y')$: 레이블 $y'$의 효과적 밀도 추정치

대칭 커널 조건:
$$k(y, y') = k(y', y), \quad \nabla_y k(y, y') + \nabla_{y'} k(y', y) = 0, \quad \forall y, y' \in \mathcal{Y}$$

가우시안 커널 및 라플라시안 커널이 이 조건을 만족 (실험에서 가우시안이 최고 성능).

**활용 예시 (비용 민감 재가중치):**
$$w_i = \frac{c}{\tilde{p}(y_i)} \propto \frac{1}{\tilde{p}(y_i)}$$

LDS로 추정된 밀도의 역수로 손실 함수를 재가중하여 불균형을 보정.

LDS 적용 시 손실:

$$\mathcal{L}_{\text{LDS}} = \frac{1}{m} \sum_{i=1}^{m} w_i \mathcal{L}(\hat{y}_i, y_i)$$

LDS 후 상관계수가 $-0.47 \to -0.83$으로 향상되어, 실제 불균형을 훨씬 잘 반영함을 보입니다.

---

#### ✅ 방법 2: Feature Distribution Smoothing (FDS)

**동기:** 데이터가 충분하다면 연속 타겟 공간의 연속성은 특징 공간에도 반영되어야 하지만, 불균형 데이터에서는 샘플이 부족한 빈의 특징 통계가 편향된다.

**빈별 특징 통계 추정:**

$$\boldsymbol{\mu}_b = \frac{1}{N_b} \sum_{i=1}^{N_b} \mathbf{z}_i \tag{2}$$

$$\boldsymbol{\Sigma}_b = \frac{1}{N_b - 1} \sum_{i=1}^{N_b} (\mathbf{z}_i - \boldsymbol{\mu}_b)(\mathbf{z}_i - \boldsymbol{\mu}_b)^\top \tag{3}$$

**커널 평활화를 통한 통계 보정:**

$$\tilde{\boldsymbol{\mu}}_b = \sum_{b' \in \mathcal{B}} k(y_b, y_{b'}) \boldsymbol{\mu}_{b'} \tag{4}$$

$$\tilde{\boldsymbol{\Sigma}}_b = \sum_{b' \in \mathcal{B}} k(y_b, y_{b'}) \boldsymbol{\Sigma}_{b'} \tag{5}$$

**화이트닝 및 재채색(whitening & re-coloring)을 통한 특징 보정:**

$$\tilde{\mathbf{z}} = \tilde{\boldsymbol{\Sigma}}_b^{\frac{1}{2}} \boldsymbol{\Sigma}_b^{-\frac{1}{2}} (\mathbf{z} - \boldsymbol{\mu}_b) + \tilde{\boldsymbol{\mu}}_b \tag{6}$$

- 먼저 원래 분포를 화이트닝(표준화)한 후, 평활화된 통계로 재채색
- 보정된 특징 $\tilde{\mathbf{z}}$가 최종 회귀 함수로 전달됨

**모멘텀 업데이트 (Exponential Moving Average):**

$$\boldsymbol{\mu}_b^{(e+1)} \leftarrow \alpha \cdot \boldsymbol{\mu}_b^{(e)} + (1 - \alpha) \cdot \boldsymbol{\mu}_b$$

$$\boldsymbol{\Sigma}_b^{(e+1)} \leftarrow \alpha \cdot \boldsymbol{\Sigma}_b^{(e)} + (1 - \alpha) \cdot \boldsymbol{\Sigma}_b$$

- 각 에폭마다 누적 통계를 업데이트하여 안정적 추정치 유지
- 모멘텀 $\alpha = 0.9$로 설정

---

### 2.3 모델 구조

| 데이터셋 | 사용 아키텍처 |
|----------|--------------|
| IMDB-WIKI-DIR, AgeDB-DIR | ResNet-50 (backbone) + FDS 보정 레이어 |
| STS-B-DIR | BiLSTM (2-layer, 1500D) + GloVe (300D) |
| NYUD2-DIR | ResNet-50 기반 Encoder-Decoder |
| SHHS-DIR | CNN-RNN (ResNet 블록 + SRU) |

**FDS 통합 방식:**
- 최종 특징 맵(feature map) 이후에 **특징 보정 레이어(feature calibration layer)** 삽입
- 에폭 단위로 누적 통계 업데이트, 에폭 내에서는 평활화 통계 고정
- 추론 시 평활화 모듈 제거 가능 (학습이 진행될수록 $L_1$ 거리가 감소)

---

### 2.4 성능 향상

#### IMDB-WIKI-DIR (나이 추정)

| 방법 | MAE All | MAE Med. | MAE Few |
|------|---------|----------|---------|
| VANILLA | 8.06 | 15.12 | 26.33 |
| SQINV + LDS + FDS (Best) | **7.78** | **12.61** | **22.19** |
| 향상 폭 | +0.41 | +2.71 | +4.14 |

#### SHHS-DIR (건강 점수 예측)

| 방법 | MAE All | MAE Many | MAE Med. | MAE Few |
|------|---------|----------|----------|---------|
| VANILLA | 15.36 | 12.47 | 13.98 | 16.94 |
| INV + LDS + FDS | **13.76** | **11.12** | **12.18** | **15.07** |
| 향상 폭 | +1.60 | +1.41 | +1.80 | +1.87 |

**주요 관찰:**
- SMOTER, SMOGN은 고차원 이미지 데이터에서 오히려 성능 저하
- LDS + FDS 결합이 대부분의 경우 최고 성능
- Many-shot 영역은 소폭 유지/향상, Medium/Few-shot 영역에서 큰 폭 향상

---

### 2.5 한계

1. **빈(bin) 크기 설정의 임의성:** 최소 해상도 $\delta y = y_{b+1} - y_b$를 수동으로 설정해야 하며, 이는 태스크에 따라 상이함
2. **FDS의 계산 비용:** 에폭마다 모든 빈의 공분산 행렬을 추정해야 하므로, 특징 차원이 매우 클 경우 분산(variance) 근사를 사용해야 함
3. **커널 하이퍼파라미터 선택:** 커널 크기 $l$, 표준편차 $\sigma$ 등의 선택이 필요 (실험적으로는 로버스트하지만)
4. **극단적 불균형 시 Many-shot 영역 소폭 저하:** 재가중치 적용 시 많은 샘플이 있는 영역의 성능이 미세하게 감소할 수 있음
5. **연속성 가정:** 인접 타겟 간 정보가 실제로 유사하다는 가정이 성립하지 않는 경우(예: 불연속적 현상) 적용이 제한적

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 일반화 성능 향상과 직접적으로 관련된 내용을 세 가지 측면에서 분석합니다.

### 3.1 Zero-shot 영역으로의 일반화 (Extrapolation & Interpolation)

논문은 훈련 데이터가 전혀 없는 타겟 값 영역(zero-shot)에 대한 일반화를 명시적으로 평가합니다.

**실험 결과 (IMDB-WIKI-DIR 서브셋, Table 6):**

| 방법 | MAE All | MAE w/ data | MAE Interp. | MAE Extrap. |
|------|---------|-------------|-------------|-------------|
| VANILLA | 11.72 | 9.32 | 16.13 | 18.19 |
| VANILLA + LDS + FDS | **10.27** | **8.11** | **13.71** | **17.02** |
| 향상 | +1.45 | +1.21 | **+2.42** | +1.17 |

- **보간(interpolation) 영역에서 더 큰 향상** 관찰
- 이는 LDS가 인접 레이블의 정보를 활용하여 누락된 영역을 간접적으로 보완하기 때문
- FDS는 특징 통계를 평활화하여 희소 영역의 표현력을 보강

### 3.2 다양한 불균형 분포에서의 로버스트 일반화

다양한 형태의 불균형 분포(1~4개의 왜곡 가우시안 피크)에서 일관된 성능 향상을 보임 (Table 18):

| 분포 형태 | VANILLA MAE | LDS+FDS MAE | 향상 |
|-----------|-------------|-------------|------|
| 1 peak (zero-shot 포함) | 22.67 | 19.21 | +3.46 |
| 2 peaks (interp/extrap) | 11.72 | 10.27 | +1.45 |
| 3 peaks (zero-shot 포함) | 20.11 | 17.76 | +2.35 |
| 4 peaks (interp/extrap) | 12.16 | 11.13 | +1.03 |

상대적 MAE 향상이 **8.8%~12.4%** 범위에서 일관되게 관찰되어, 분포 변화에 대한 로버스트한 일반화를 입증합니다.

### 3.3 FDS의 특징 공간 보정을 통한 일반화

FDS의 핵심 메커니즘은 **편향된 특징 표현을 교정**하여 일반화를 향상시킵니다:

**FDS 효과 분석 (Figure 8(c)):**
- 학습이 진행될수록 실제 통계 $\{\boldsymbol{\mu}_b, \boldsymbol{\Sigma}_b\}$와 평활화 통계 $\{\tilde{\boldsymbol{\mu}}_b, \tilde{\boldsymbol{\Sigma}}_b\}$ 간의 $L_1$ 거리가 점점 감소
- 이는 **모델이 평활화 없이도 자연스럽게 연속적 특징을 학습하게 됨**을 의미
- 최종적으로 추론 시 평활화 모듈을 제거해도 성능이 유지됨

이는 FDS가 단순한 후처리가 아니라 **학습 과정 자체를 정규화(regularization)하는 역할**을 함을 시사합니다.

### 3.4 다양한 손실 함수와의 호환성

$L_1$ 손실, MSE 손실, Huber 손실 모두에서 유사한 성능 향상 관찰 (Table 15):
- STS-B-DIR에서 MSE 향상 범위: LDS $3.3\%\sim6.2\%$, FDS $3.3\%\sim6.2\%$ (손실 함수 무관)
- 이는 **LDS/FDS가 특정 손실 함수에 종속되지 않고 범용적으로 작동함**을 의미

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

#### (1) 새로운 문제 패러다임 수립
DIR을 공식적으로 정의함으로써, 기존에 분류 문제로 처리되거나 무시되던 연속 타겟 불균형 문제를 독립적 연구 분야로 격상시켰습니다. 헬스케어, 기상 예측, 금융 위험 평가 등 실세계 응용에서 DIR 설정이 광범위하게 적용될 수 있습니다.

#### (2) 평가 지표 혁신
논문이 제안한 **Error Geometric Mean (GM)**:

$$\text{GM} = \left(\prod_{i=1}^{N} e_i\right)^{\frac{1}{N}}, \quad e_i = |y_i - \hat{y}_i|$$

이 지표는 예측 공정성(fairness)을 강조하여 불균형 회귀에서 편향된 평가를 방지합니다.

#### (3) 특징 공간 분포 보정의 새로운 시각
FDS는 **특징 공간의 인접성(feature-space continuity)**을 명시적으로 활용하는 새로운 접근법을 제시합니다. 이는 도메인 적응(domain adaptation), 연속 학습(continual learning), 메타러닝(meta-learning) 등에 응용 가능한 아이디어를 제공합니다.

#### (4) 벤치마크의 기여
5개의 대규모 DIR 벤치마크(IMDB-WIKI-DIR, AgeDB-DIR, STS-B-DIR, NYUD2-DIR, SHHS-DIR)는 후속 연구의 표준 평가 환경을 제공합니다.

---

### 4.2 앞으로 연구 시 고려할 점

#### (1) 커널 선택 이론화
현재 커널(가우시안/라플라시안/삼각형)의 선택은 경험적입니다. 데이터의 특성에 따라 최적 커널을 자동으로 학습하거나 선택하는 **adaptive kernel selection** 방법론이 필요합니다.

#### (2) 빈 크기(bin size) 자동 결정
현재 빈 크기 $\delta y$는 도메인 지식에 의존합니다. 데이터로부터 최적 해상도를 자동으로 학습하는 방법(예: 계층적 빈 분할, 베이지안 최적화)이 필요합니다.

#### (3) 대규모 특징 차원에서의 확장성
FDS의 공분산 추정은 특징 차원이 클 경우 계산 비용이 급증합니다. 현재 논문에서도 분산(variance)으로 근사하는 방법을 사용하지만, 더 효율적인 저차원 근사(예: 저랭크 근사, 확률적 공분산 추정)가 필요합니다.

#### (4) 온라인/스트리밍 환경에서의 DIR
실시간 데이터 스트림에서 레이블 분포가 시간에 따라 변하는 경우, 정적인 LDS/FDS로는 대응이 어렵습니다. 동적 분포 추적 및 적응적 평활화 메커니즘 연구가 필요합니다.

#### (5) 다차원 연속 타겟으로의 확장
현재 DIR은 단일 연속 타겟 $y \in \mathbb{R}$를 대상으로 하지만, 다차원 연속 타겟 $\mathbf{y} \in \mathbb{R}^K$로의 확장(예: 3D 자세 추정, 다변량 건강 지표)이 중요한 연구 방향입니다.

#### (6) 인과적 접근과의 결합
데이터 불균형의 근본적 원인을 인과 그래프로 모델링하고, 반사실적(counterfactual) 데이터 증강을 통해 DIR을 해결하는 접근이 유망합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래는 본 논문(2021, ICML)과 관련된 2020년 이후 주요 연구들입니다.

> **⚠️ 주의:** 아래 비교는 제공된 논문 PDF와 공개 학술 정보를 기반으로 하며, 일부 세부 수치는 해당 논문을 직접 확인하시기 바랍니다.

### 5.1 비교 표

| 논문 | 방법 핵심 | 차별점 vs DIR | 한계 |
|------|-----------|--------------|------|
| **DIR (Yang et al., 2021, ICML)** | LDS + FDS (커널 평활화) | DIR 최초 정의 및 벤치마크 | 빈 크기 수동 설정, 단일 타겟 |
| **SMOGN (Branco et al., 2017)** | 데이터 보간 기반 오버샘플링 | 고차원 데이터에서 성능 저하 | 비선형 공간에서 부적합 |
| **Balanced MSE (Ren et al., 2022, NeurIPS)** | 타겟 분포 기반 손실 재정의 | 베이지안 관점에서 불균형 회귀 이론화 | 분포 추정 필요 |
| **RankSim (Gong et al., 2022, ICML)** | 특징 공간의 순위 유사성 정규화 | 연속 레이블의 순서 정보 활용 | 페어링 샘플링 비용 |
| **ConR (Keramati et al., 2023)** | 대조 학습 기반 연속 회귀 정규화 | 자기지도(self-supervised) 방식 통합 | 대조 학습의 배치 크기 의존성 |

### 5.2 주요 발전 방향 분석

#### (1) Balanced MSE (Ren et al., 2022)

기존 MSE 손실이 암묵적으로 훈련 데이터의 타겟 분포를 사전(prior)으로 가정한다는 것을 이론적으로 분석하고, 이를 보정하는 **Balanced MSE** 손실을 제안:

$$\mathcal{L}_{\text{Balanced-MSE}} = \mathbb{E}_{p_{\text{train}}(y)} \left[ \frac{p_{\text{train}}(y)}{p_{\text{test}}(y)} \cdot (\hat{y} - y)^2 \right]$$

DIR의 LDS가 비모수적(non-parametric) 밀도 추정에 기반한 반면, Balanced MSE는 **분포적 관점에서 이론적 정당성**을 제공합니다.

#### (2) RankSim (Gong et al., 2022)

연속 레이블 공간에서 **랭킹 유사성(rank similarity)**을 정규화 항으로 추가:

$$\mathcal{L}_{\text{RankSim}} = \mathcal{L}_{\text{reg}} + \lambda \cdot \mathcal{L}_{\text{rank}}$$

여기서 $\mathcal{L}_{\text{rank}}$는 레이블 공간의 순서가 특징 공간에서도 보존되도록 강제합니다. DIR의 FDS가 **분포 통계를 전이**하는 방식과 달리, RankSim은 **상대적 순서 관계**를 보존합니다.

#### (3) 연구 트렌드 요약

```
2017~2020: 분류 기반 불균형 학습 연구 집중
    ↓
2021 (DIR): 연속 회귀 불균형 문제 공식 정의 + LDS/FDS
    ↓
2022~2023: 이론적 정당화(Balanced MSE), 순위 기반(RankSim),
           대조 학습 기반(ConR) 등으로 확장
    ↓
현재: 다차원 타겟, 온라인 학습, 인과적 접근 등 탐색 중
```

---

## 참고 자료

- **주요 논문:**
  - Yang, Y., Zha, K., Chen, Y., Wang, H., & Katabi, D. (2021). *Delving into Deep Imbalanced Regression*. ICML 2021. arXiv:2102.09554v2
  - GitHub: https://github.com/YyzHarry/imbalanced-regression

- **논문 내 인용 문헌:**
  - Branco, P., Torgo, L., & Ribeiro, R. P. (2017). SMOGN. PMLR.
  - Torgo, L., et al. (2013). SMOTE for Regression. ICAI.
  - Kang, B., et al. (2020). Decoupling Representation and Classifier for Long-Tailed Recognition. ICLR.
  - Cao, K., et al. (2019). Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss. NeurIPS.
  - Sun, B., Feng, J., & Saenko, K. (2016). Return of Frustratingly Easy Domain Adaptation. AAAI.
  - Rothe, R., Timofte, R., & Gool, L. V. (2018). Deep Expectation of Real and Apparent Age. IJCV.
  - He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
  - Parzen, E. (1962). On Estimation of a Probability Density Function and Mode. Annals of Mathematical Statistics.

- **관련 후속 연구 (제목 참고용):**
  - Ren, J., et al. (2022). *Balanced MSE for Imbalanced Visual Regression*. NeurIPS 2022.
  - Gong, T., et al. (2022). *RankSim: Ranking Similarity Regularization for Deep Imbalanced Regression*. ICML 2022.
