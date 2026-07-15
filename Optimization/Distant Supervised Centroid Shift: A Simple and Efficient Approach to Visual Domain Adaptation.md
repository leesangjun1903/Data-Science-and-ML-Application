# Distant Supervised Centroid Shift: A Simple and Efficient Approach to Visual Domain Adaptation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Liang et al., CVPR 2019)은 기존 도메인 적응(Domain Adaptation) 방법들의 **계산 복잡도 문제**와 **소스 도메인 데이터 의존성 문제**를 동시에 해결하는 간단하고 효율적인 방법을 제안합니다.

**핵심 아이디어**: 타겟 도메인의 클래스별 중심점(centroid)이 소스 도메인의 중심점으로부터 적절히 이동(shift)된 부분공간(subspace)을 탐색함으로써, 도메인 간 공변량 이동(covariate shift)을 해결합니다.

### 주요 기여 (5가지)

| 기여 | 설명 |
|------|------|
| **Privacy-preserving (원격 감독, Distant Supervision)** | 소스 도메인 원본 데이터 대신 클래스별 통계량($\hat{\mu}_r$, $\hat{\Sigma}_r$)만 사용 |
| **선형 시간 복잡도** | 기존 방법들의 $\mathcal{O}(n^2)$ 대비 $\mathcal{O}(T_i k d^2 + T_i T_o C n_t d)$ 달성 |
| **수렴 보장** | 교대 최소화(Alternating Minimization) 알고리즘의 이론적 수렴성 증명 |
| **범용성** | 단일/다중 소스 도메인 적응 및 도메인 일반화로 확장 가능 |
| **보완성** | DAN, RevGrad 등 기존 딥러닝 방법과 결합 시 성능 향상 |

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)**: 레이블이 있는 소스 도메인 $\mathcal{X}_s$와 레이블이 없는 타겟 도메인 $\mathcal{X}_t$ 간의 분포 불일치(covariate shift) 해결.

기존 방법들의 문제점:
- **딥러닝 기반**: 대규모 소스 도메인 의존, 높은 계산 비용, 배치 처리로 인한 전역 손실 최적화 어려움
- **부분공간 학습 기반**: MMD 행렬 계산으로 인한 $\mathcal{O}(n^2)$ 복잡도, 대규모 데이터셋 처리 불가
- **프라이버시 문제**: 소스 도메인 원본 데이터를 반드시 필요로 함

### 2.2 제안 방법 (수식 포함)

#### 문제 설정

소스 도메인: 레이블된 데이터 $X_s = \{(x_i^s, y_i^s)\}_{i=1}^{n_s}$

타겟 도메인: 레이블 없는 데이터 $X_t = \{x_i^t\}_{i=1}^{n_t}$

소스 도메인에서 필요한 정보 (원본 데이터 대신):

$$\hat{\mu}_r = \frac{1}{m_r} \sum_{y_i^s = r} x_i^s, \quad \hat{\Sigma}_r = \frac{1}{m_r} \sum_{y_i^s = r} (x_i^s - \hat{\mu}_r)(x_i^s - \hat{\mu}_r)^T$$

#### 핵심 목적 함수

**소스 도메인 분류 목적** (within-class scatter 최소화):

$$\min_{W} \frac{\sum_i d_W(x_i^s, \hat{\mu}_{y_i^s})}{\sum_i d_W(x_i^s, \mathbf{0})} = \frac{\text{trace}(W S_w^s W^T)}{\text{trace}(W S_t^s W^T)} $$

여기서 $d_W(x, x') = \|Wx - Wx'\|_2^2$

**통합 목적 함수 (Centroid Shift 포함)**:

$$\min_{W, \Delta_r, \hat{y}_i^t} \frac{\sum_r \sum_{y_i^s=r} d_W(x_i^s, \hat{\mu}_r)}{\sum_i d_W(x_i^s, \mathbf{0})} + \sum_r \beta_r \|W\Delta_r\|_2^2 + \frac{\sum_r \sum_{\hat{y}_i^t=r} d_W(x_i^t, \hat{\mu}_r + \Delta_r)}{\sum_i d_W(x_i^t, \mathbf{0})} $$

- $\Delta_r \in \mathbb{R}^{d \times 1}$: 클래스 $r$의 perturbation 변수 (centroid shift)
- $\beta_r$: 클래스 크기에 따른 균형 파라미터

**최종 Ratio-Trace 완화 목적 함수**:

```math
\min_{W, \Delta_r^t, \hat{y}_i^t} \text{trace}\left\{ \frac{W(S_w^s + S_w^t + \sum_r \beta_r \Delta_r \Delta_r^T + \lambda I)W^T}{W(S_t^s + S_t^t)W^T} \right\}
```

$$\text{s.t.} \quad W(S_t^s + S_t^t)W^T = I$$

### 2.3 최적화: 교대 최소화 (Alternating Minimization)

세 가지 서브문제 각각에 대한 **닫힌 형태(Closed-form) 해**를 도출합니다.

#### Step 1: W-step (부분공간 학습)

$\{\Delta_r\}$와 $\{\hat{y}_i^t\}$ 고정 시, 일반화 고유값 분해(GEVD)로 해결:

$$\min_{W \in \mathbb{R}^{k \times d}} \text{trace}\{(W S_t W^T)^{-1}(W(S_w + \lambda I)W^T)\} $$

$$S_t w_a = \gamma_a (S_w + \lambda I) w_a$$

- $S_t = S_t^s + S_t^t$ (전체 산포 행렬)
- $S_w = S_w^s + S_w^t + \sum_r \beta_r \Delta_r \Delta_r^T$ (within-class 산포 행렬)
- 가장 작은 $k$개의 일반화 고유값에 대응하는 고유벡터로 $W$ 구성

#### Step 2: $\hat{Y}^t$-step (유사 레이블 추정)

$W$와 $\{\Delta_r\}$ 고정 시, 각 타겟 샘플에 대해 독립적으로:

$$\hat{y}_i^t = \arg\max_{r \in [1,C]} h_r x_i^t - b_r $$

여기서:
- $S_p = W^T(WS_tW^T)^{-1}W$ (양정치 행렬)
- $h_r = 2(\hat{\mu}_r + \Delta_r)S_p \in \mathbb{R}^{1 \times d}$
- $b_r = -(\hat{\mu}_r + \Delta_r)^T S_p (\hat{\mu}_r + \Delta_r)$

#### Step 3: $\Delta$-step (Centroid Shift 갱신)

$W$와 $\{\hat{y}_i^t\}$ 고정 시, 각 클래스 $r$에 대해:

$$\Delta_r = \left(\sum_{\hat{y}_i^t = r} x_i^t - n_r \hat{\mu}_r\right) / (\beta_r + n_r) $$

$$\hat{\mu}_r + \Delta_r = \left(\beta_r \hat{\mu}_r + \sum_{\hat{y}_i^t = r} x_i^t\right) / (\beta_r + n_r) $$

**핵심 해석**: 갱신된 타겟 중심점은 소스 클래스 평균과 유사 타겟 클래스 평균의 **가중 보간(weighted interpolation)**입니다. $\beta_r/n_r$가 클수록 타겟 중심점이 소스 중심점에 가까워집니다.

#### 알고리즘 흐름

```
입력: 소스 통계량 {m_r, μ̂_r, Σ̂_r}, 타겟 데이터 {x_i^t}
1. 산포 행렬 S^s_w, S^s_t, S^t_t 계산
2. S^t_w = 0, Δ_r = 0으로 초기화 → W 계산 (Eq.7)
3. {ŷ^t_i} 추정 → S^t_w 갱신
4. β_r = α * l_r 계산
5. 외부 반복 (최대 T_o = 10회):
   6. 내부 반복 (최대 T_i = 5회):
      - {Δ_r} 갱신 (Eq.11)
      - {ŷ^t_i} 갱신 (Eq.9)
   7. S^t_w 갱신 및 W 재계산 (Eq.7)
출력: W, {Δ_r}, {ŷ^t_i}
```

### 2.4 모델 구조

```
소스 도메인 통계량            타겟 도메인 데이터
{m_r, μ̂_r, Σ̂_r}              {x_i^t}
        ↓                           ↓
   S^s_w, S^s_t 계산          S^t_t 계산
        ↓___________________________|
              ↓
    [부분공간 학습: W via GEVD]
              ↓
    [유사 레이블 추정: ŷ^t_i]
              ↓
    [Centroid Shift 갱신: Δ_r]
              ↓
    [수렴할 때까지 반복]
              ↓
    최종 분류: 최근접 중심점(Nearest Centroid)
```

**확장 모델**:
- **MCS $_A$** (Domain Generalization): 다중 소스에서 복수 중심점 사용
- **MCS $_B$** (Domain Generalization): 투영된 소스 인스턴스로 SVM 훈련
- **MCS $_C$** (Multi-source): 다중 소스를 단일 소스로 결합
- **MCS $_D$** (Multi-source): 소스별 산포 행렬 합산

### 2.5 성능 향상

#### Office31 데이터셋 (AlexNet-FC7 특징)

| 방법 | A→D | A→W | D→A | D→W | W→A | W→D | **평균** |
|------|-----|-----|-----|-----|-----|-----|----------|
| JDA | 66.5 | 68.8 | 56.3 | 97.7 | 53.5 | 99.6 | 73.7 |
| DICE | 66.7 | 71.4 | 56.5 | 96.9 | 58.6 | 99.8 | 75.0 |
| RevGrad | 72.3 | 73.0 | 53.4 | 96.4 | 51.2 | 99.2 | 74.3 |
| JAN-A | 72.8 | 75.2 | 57.5 | 96.6 | 56.3 | 99.6 | 76.3 |
| **MCS(ours)** | **71.9** | **75.1** | **58.8** | 96.7 | 57.2 | 99.4 | **76.5** |

#### 딥러닝 방법과 결합 시 성능 (Office31, ResNet-50)

| 방법 | 평균 정확도 (%) |
|------|----------------|
| RevGrad | 81.8 |
| **RevGrad + MCS** | **87.8** (+6.0) |
| DAN | 81.7 |
| **DAN + MCS** | **87.4** (+5.7) |
| CDAN+E | 87.7 |

### 2.6 한계점

1. **얕은 특징 변환(Shallow Feature Transformation) 의존**: 깊은 표현 학습의 비선형성을 완전히 활용하지 못함
2. **최근접 중심점 분류기의 한계**: Clipart처럼 클래스 내 다양성이 큰 경우 성능 저하
3. **가우시안 분포 가정**: 소스 도메인이 다변량 가우시안 분포를 따른다고 가정하여 비가우시안 분포에는 부적합
4. **국소 최솟값 수렴**: 교대 최소화 특성상 전역 최적해가 아닌 국소 최솟값에 수렴 가능
5. **유사 레이블의 노이즈**: 초기 유사 레이블 품질에 전체 성능이 민감하게 반응

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 도메인 일반화(Domain Generalization)로의 확장

타겟 도메인 정보가 전혀 없는 상황에서도 MCS를 적용 가능합니다.

**도메인 일반화 시 목적 함수 변형**:

$$S_w = \sum_{s \in \text{sources}} S_w^s, \quad S_t = \sum_{s \in \text{sources}} S_t^s$$

$$\Delta_r = \hat{\mu}_r^{s_1} - \hat{\mu}_r^{s_2} \quad \text{(소스 간 클래스 평균 차이)}$$

$\hat{Y}^t$와 $\Delta_r$ 학습 없이 **1회 통과(one-pass) 알고리즘**으로 처리됩니다.

**VLCS 데이터셋 결과**:

| 방법 | V | L | C | S | 평균 (10개 태스크 중 승리) |
|------|---|---|---|---|--------------------------|
| CIDG | - | - | - | - | 2/10 태스크 |
| **MCS $_A$ (ours)** | - | - | - | - | **4/10 태스크** |
| **MCS $_B$ (ours)** | - | - | - | - | **4/10 태스크** |

또한 MCS의 표준편차가 CIDG보다 현저히 작아 **안정성이 우수**합니다.

### 3.2 Centroid Shift의 일반화 메커니즘

식 (12)에서:

$$\hat{\mu}_r^{\text{target}} = \frac{\beta_r \hat{\mu}_r^{\text{source}} + \sum_{\hat{y}_i^t=r} x_i^t}{\beta_r + n_r}$$

이는 **Bayesian 관점의 사전 지식(소스 중심점) + 관측 데이터(타겟 인스턴스)의 결합**으로 해석 가능합니다:

- $\beta_r \to \infty$: 타겟 중심점 = 소스 중심점 (전이 없음)
- $\beta_r \to 0$: 타겟 중심점 = 타겟 클래스 평균 (완전한 자기지도)
- **최적 $\beta_r$**: 두 극단 사이의 균형점 → 도메인 간 지식 전이

### 3.3 일반화 향상을 위한 구체적 메커니즘

1. **프라이버시 보존(Distant Supervision)**: 소스 원본 데이터 대신 통계량 사용으로 다양한 실제 환경에서 적용 가능성 증대

2. **모듈성(Modularity)**: 딥러닝 특징 추출기와 결합 가능한 플러그인 형태
   - 딥러닝 모델의 중간 특징을 입력으로 사용
   - RevGrad+MCS: 81.8 → 87.8%, DAN+MCS: 81.7 → 87.4%

3. **다중 소스 적응**: 여러 소스 도메인의 통계량을 결합하여 더 강건한 표현 학습
   - MCS $_D$ : 소스별 산포 행렬 합산으로 각 소스 도메인의 고유 정보 보존

4. **선형 시간 복잡도**: 대규모 데이터셋(MNIST 60K, SVHN 73K 등)에도 적용 가능
   - 기존 방법들은 메모리 제약으로 적용 불가 → MCS는 가능

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (1) 효율적 도메인 적응의 기준선(Baseline) 확립
복잡한 GAN 기반 방법들과 비교할 수 있는 **간단하고 강력한 기준선** 제공. 이후 연구들이 단순히 복잡성만 늘리는 것이 아닌 실질적 성능 향상을 입증해야 하는 기준이 됨.

#### (2) Privacy-Preserving 도메인 적응 연구 촉진
소스 데이터에 접근하지 않는 도메인 적응(source-free domain adaptation) 연구의 선구적 역할. 이후 SHOT (ICML 2020), G-SFDA (ICCV 2021) 등의 연구로 이어짐.

#### (3) 중심점 기반 표현 학습의 재조명
Prototype/Centroid 기반 방법론의 유효성을 도메인 적응에서 입증. 이후 ECACL, PAC, FixBi 등의 프로토타입 기반 연구에 영향.

#### (4) 딥러닝과 얕은 방법의 결합 가능성 제시
딥러닝 특징 + 얕은 부분공간 방법의 시너지 효과 실증. 딥러닝 기반 도메인 적응의 사후 처리(post-processing) 모듈로서의 가능성.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

#### Source-Free Domain Adaptation (소스 프리 도메인 적응)

| 논문 | 방법 | MCS와의 관계 |
|------|------|--------------|
| **SHOT** (Liang et al., ICML 2020) | 소스 모델 고정, 타겟 특징 추출기 최적화 (정보 최대화 + 유사 레이블) | MCS의 원격 감독 개념 계승, 딥러닝으로 확장 |
| **G-SFDA** (Yang et al., ICCV 2021) | 그래프 구조 활용 소스프리 적응 | MCS의 중심점 아이디어 + 그래프 정규화 |
| **NRC** (Yang et al., NeurIPS 2021) | 이웃 관계 기반 클러스터링 | MCS의 유사 레이블 전략 확장 |

**MCS의 원격 감독 개념**은 이후 Source-Free DA의 핵심 아이디어로 발전:

$$\text{MCS: } \{\hat{\mu}_r, \hat{\Sigma}_r\} \xrightarrow{\text{확장}} \text{SHOT: 소스 모델 파라미터만 사용}$$

#### 프로토타입 기반 도메인 적응

| 논문 | 방법 | MCS와의 차이 |
|------|------|-------------|
| **ECACL** (Li et al., ICCV 2021) | 중심점 정렬 + 대조 학습 | MCS에 대조 학습(Contrastive Learning) 추가 |
| **PAC** (Tanwisuth et al., NeurIPS 2021) | 프로토타입 기반 정렬 + 최적 수송 | MCS의 중심점 + OT 결합 |
| **ATDOC** (Liu et al., CVPR 2021) | 이웃 클러스터링 기반 타겟 구조 활용 | MCS의 유사 레이블 전략 개선 |

#### 도메인 일반화 발전

| 논문 | 방법 | MCS 기여와의 연관성 |
|------|------|-------------------|
| **DomainBed** (Gulrajani & Lopez-Paz, ICLR 2021) | 표준 벤치마크 및 ERM 기준선 제시 | MCS의 강력한 기준선 필요성 공감 |
| **SWAD** (Cha et al., NeurIPS 2021) | 가중 평균 앙상블 | MCS의 단순함 + 강력한 성능 철학 공유 |
| **MIRO** (Cha et al., ECCV 2022) | 상호정보 정규화 | 클래스 표현의 안정성 추구 (MCS의 중심점 안정화와 유사) |

#### 요약 비교표

| 방법 | 소스 데이터 필요 | 계산 복잡도 | 딥러닝 통합 | 수렴 보장 |
|------|----------------|------------|------------|----------|
| **MCS (2019)** | 통계량만 | $\mathcal{O}(n)$ | 플러그인 | ✅ |
| SHOT (2020) | ❌ (모델만) | 높음 | End-to-end | ❌ |
| NRC (2021) | ❌ | 중간 | End-to-end | ❌ |
| ECACL (2021) | ✅ | 높음 | End-to-end | ❌ |

### 4.3 앞으로 연구 시 고려할 점

#### (1) 비가우시안 분포 처리
현재 소스 도메인의 가우시안 가정을 완화하기 위한 방법 연구 필요:
- **Gaussian Mixture Model** 또는 **Normalizing Flows** 기반 분포 추정
- 분포 가정에 강건한 중심점 표현 (e.g., median-of-means estimator)

#### (2) 유사 레이블 품질 개선
초기 유사 레이블의 노이즈가 전체 성능에 영향:
- 신뢰도 기반 가중 학습(Confidence-based Reweighting)과 결합
- 대조 학습(Contrastive Learning)으로 표현의 구별력 강화

#### (3) 대규모 클래스 수 처리
중심점 기반 방법은 클래스 수 $C$가 매우 클 때 계산량 증가:
- 계층적 중심점 구조 도입
- 근사 최근접 이웃(Approximate Nearest Neighbor) 알고리즘 활용

#### (4) 연속 도메인 적응(Continual/Online DA)으로 확장
MCS의 온라인 갱신 가능성:

$$\hat{\mu}_r^{(t+1)} \leftarrow \frac{\beta_r \hat{\mu}_r^{\text{source}} + n_r^{(t)} \hat{\mu}_r^{(t)} + x_{\text{new}}}{\beta_r + n_r^{(t)} + 1}$$

순차적 중심점 갱신을 통한 실시간 도메인 적응 구현 가능.

#### (5) 딥러닝과의 End-to-End 통합
현재 MCS는 딥러닝의 사후 처리(post-processing)로 사용:
- MCS 목적 함수를 역전파 가능한 손실로 변환
- 특징 추출기와 공동 훈련(Joint Training) 연구 필요

#### (6) 공정성(Fairness)과 프라이버시의 균형
원격 감독(Distant Supervision)의 프라이버시 보존 특성을 활용:
- 연합 학습(Federated Learning)과의 결합
- 차등 프라이버시(Differential Privacy) 보장 통계량 전송

---

## 참고 자료

**주요 논문**
- Liang, J., He, R., Sun, Z., & Tan, T. (2019). **Distant Supervised Centroid Shift: A Simple and Efficient Approach to Visual Domain Adaptation**. *CVPR 2019*, pp. 2975–2984. (첨부 PDF)

**논문 내 인용 참고문헌 (주요)**
- Long, M., et al. (2013). Transfer feature learning with joint distribution adaptation (JDA). *ICCV*.
- Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation (RevGrad). *ICML*.
- Long, M., et al. (2018). Conditional adversarial domain adaptation (CDAN). *NeurIPS*.
- Zhang, J., et al. (2017). Joint geometrical and statistical alignment for visual domain adaptation (JGSA). *CVPR*.

**2020년 이후 비교 연구 (논문 제목 기반, 직접 확인 필요)**
- Liang, J., et al. (2020). Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation (SHOT). *ICML 2020*.
- Yang, S., et al. (2021). Generalized Source-free Domain Adaptation (G-SFDA). *ICCV 2021*.
- Yang, S., et al. (2021). Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation (NRC). *NeurIPS 2021*.
- Gulrajani, I., & Lopez-Paz, D. (2021). In Search of Lost Domain Generalization (DomainBed). *ICLR 2021*.

> ⚠️ **정확도 주의**: 2020년 이후 비교 연구들의 구체적 수치와 세부 방법론은 해당 논문을 직접 확인하시기 바랍니다. 본 답변에서 MCS와의 비교 분석은 각 논문의 핵심 아이디어를 기반으로 한 개념적 비교이며, 실험 수치는 첨부된 MCS 논문 원문에서 직접 인용한 내용에 한해 정확성을 보장합니다.
