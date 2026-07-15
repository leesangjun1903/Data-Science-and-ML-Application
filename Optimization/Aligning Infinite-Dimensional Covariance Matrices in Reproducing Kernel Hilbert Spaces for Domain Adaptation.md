# Aligning Infinite-Dimensional Covariance Matrices in Reproducing Kernel Hilbert Spaces for Domain Adaptation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
기존 도메인 적응(Domain Adaptation) 방법들은 **입력 공간(Input Space)**에서 분포 불일치를 줄이는 방식에 집중했지만, 커널 기반 학습기(Kernel-based Learning Machine)의 성능은 **재생 커널 힐베르트 공간(Reproducing Kernel Hilbert Space, RKHS)**에서의 통계적 특성에 의존한다. 따라서 분포 정렬을 RKHS에서 수행하는 것이 더 효과적이다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **이론적 일반화** | 유클리드 공간의 공분산 정렬 문제를 무한 차원 RKHS로 확장 |
| **두 가지 정렬 방법** | 커널 화이트닝-컬러링 맵(KWC), 커널 최적 수송 맵(KOT) 제안 |
| **닫힌 형식 표현** | 커널 행렬(Kernel Matrix)을 통한 계산 가능한 닫힌 형식 도출 |
| **Out-of-Sample 처리** | 새로운 데이터에 대해 모델 재계산 없이 일반화 가능 |
| **실험 검증** | 248개 도메인 적응 태스크에서 최신 기법 대비 우수한 성능 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**도메인 시프트(Domain Shift)** 문제: 소스 도메인(학습 데이터)과 타겟 도메인(테스트 데이터)의 분포가 다를 때 모델 성능이 저하되는 현상.

- 소스 도메인: 레이블된 데이터 $\{X^s, \vec{l}\_X\} = \{(x_i, l_{x_i})\}_{i=1}^{N_s}$
- 타겟 도메인: 레이블 없는 데이터 $Y^t = \{y_j\}_{j=1}^{N_t}$

**기존 방법의 한계:**
- 선형 CORAL(Sun et al., 2016): 선형 상관관계만 고려, 비선형 구조 처리 불가
- 최적 수송(Courty et al., 2017): Transductive 설정에서만 적용 가능
- TCA, JDA: 입력 공간에서의 정렬만 수행

---

### 2.2 제안 방법 (수식 포함)

#### [Step 1] 유클리드 공간에서의 공분산 정렬 기초

소스/타겟 공분산 행렬 $\Sigma_s, \Sigma_t \in \mathbb{R}^{n \times n}$이 주어졌을 때, 선형 변환 $T: \mathbb{R}^n \rightarrow \mathbb{R}^n$을 찾아 다음을 만족시킨다:

$$T \Sigma_s T^T = \Sigma_t $$

**① 화이트닝-컬러링 맵 (Whitening-Coloring Map):**

$$T_{\text{WC}} = \Sigma_t^{\frac{1}{2}} \Sigma_s^{-\frac{1}{2}} $$

랭크 결핍(Rank-Deficient) 경우 Moore-Penrose 유사역행렬 적용:

$$\hat{T}_{\text{WC}} = \Sigma_t^{\frac{1}{2}} (\Sigma_s^{\frac{1}{2}})^{\dagger} $$

> **정리 1:** $\text{Im}(\Sigma_t) \subseteq \text{Im}(\Sigma_s)$이면 $\hat{T}\_{\text{WC}} \Sigma\_s \hat{T}\_{\text{WC}}^T = \Sigma\_t$

**② 최적 수송 맵 (Optimal Transport Map):**

Monge 최적 수송 문제:

```math
\min_{\mathcal{T}} \int_{\mathbb{R}^n} c(\vec{x}, \mathcal{T}(\vec{x})) d\mu_s \quad \text{s.t.} \quad \mathcal{T}_{\#}\mu_s = \mu_t
```

$\mu_s = \mathcal{N}(\vec{0}, \Sigma_s)$, $\mu_t = \mathcal{N}(\vec{0}, \Sigma_t)$일 때 최적해:

$$T_{\text{OT}} = \Sigma_t^{\frac{1}{2}} \left( \Sigma_t^{\frac{1}{2}} \Sigma_s \Sigma_t^{\frac{1}{2}} \right)^{-\frac{1}{2}} \Sigma_t^{\frac{1}{2}} $$

랭크 결핍 경우:

$$\hat{T}_{\text{OT}} = \Sigma_t^{\frac{1}{2}} \left( \Sigma_t^{\frac{1}{2}} \Sigma_s \Sigma_t^{\frac{1}{2}} \right)^{\dagger\frac{1}{2}} \Sigma_t^{\frac{1}{2}} $$

> **정리 2:** $\text{Ker}(\Sigma_s) \cap \text{Im}(\Sigma_t) = \{\vec{0}\}$이면 $\hat{T}\_{\text{OT}} \Sigma_s \hat{T}_{\text{OT}}^T = \Sigma_t$

---

#### [Step 2] RKHS에서의 공분산 기술자 추정

특징 맵 $\phi: \mathcal{X} \rightarrow \mathcal{H}_K$를 통해 데이터를 RKHS로 매핑. RKHS 데이터 행렬 $\Phi_X = [\phi(x_1), ..., \phi(x_N)]$.

**① 최대우도추정(MLE):**

$$\text{MC} = \Phi_X J_N J_N^T \Phi_X^T + \rho I_{\mathcal{H}_K} $$

여기서 $J_N = \frac{1}{\sqrt{N}}(I_N - \frac{1}{N}\vec{1}_N \vec{1}_N^T)$는 센터링 행렬.

**② 계산 효율적 추정(CEE):** 인수 분석 모델 $\phi(x) = \mu + \mathcal{A}\vec{z} + \epsilon$ 가정 하에:

$$W_X = J_N V_d (I_d - \rho \Lambda_d^{-1})^{\frac{1}{2}} $$

$$\text{EC} = \Phi_X W_X W_X^T \Phi_X^T + \rho I_{\mathcal{H}_K} $$

여기서 $V_d$, $\Lambda_d$는 $C_{XX} = J_N^T K_{XX} J_N$의 상위 $d$ 고유쌍.

---

#### [Step 3] RKHS에서의 정렬 (핵심 Proposition)

**Proposition 1 - 커널 화이트닝-컬러링 맵:**

$$k\hat{T}_{\text{WC}} = \Phi_Y J_{N_t} C_{YY}^{\dagger\frac{1}{2}} \left[ C_{YX} A J_{N_s} \Phi_X^T + \frac{1}{\sqrt{\rho}} J_{N_t} \Phi_Y^T \right] $$

여기서:
- $C_{YY} = J_{N_t}^T K_{YY} J_{N_t}$, $C_{YX} = J_{N_t}^T K_{YX} J_{N_s}$ (센터링된 커널 행렬)
- $A = \sum_{k=1}^{r} \frac{1}{\lambda_k}\left(\frac{1}{\sqrt{\lambda_k + \rho}} - \frac{1}{\sqrt{\rho}}\right) \vec{v}_k \vec{v}_k^T$

**Proposition 2 - 커널 최적 수송 맵:**

$$k\hat{T}_{\text{OT}} = \Phi_Y W_Y \left[ C_{YX}^w C_{XY}^w + \rho(\Lambda_Y - \rho I_d) \right]^{\dagger\frac{1}{2}} W_Y^T \Phi_Y^T $$

여기서 $C_{YX}^w = W_Y^T K_{YX} W_X$.

---

#### [Step 4] 도메인 불변 커널 행렬 구성

소스 데이터를 타겟 도메인으로 이동:

$$\Psi_s \rightarrow \Psi_{s \to t} = k\hat{T}_\triangle(\Psi_s) $$

도메인 불변 커널 행렬:

$$\tilde{K} = \begin{bmatrix} \triangle\tilde{K}_{ss} & \triangle\tilde{K}_{ts}^T \\ \triangle\tilde{K}_{ts} & \triangle\tilde{K}_{tt} \end{bmatrix} = \begin{bmatrix} \Psi_{s\to t}^T \Psi_{s\to t} & \Psi_{s\to t}^T \Psi_t \\ \Psi_t^T \Psi_{s\to t} & \Psi_t^T \Psi_t \end{bmatrix} $$

**KWC 적용 시:**

$$\text{WC}\tilde{K}_{ss} = N_s B^T B, \quad \text{WC}\tilde{K}_{ts} = \sqrt{N_s N_t} C_{YY}^{\frac{1}{2}} B = \sqrt{N_s N_t} U_Y \Lambda_Y^{\frac{1}{2}} U_Y^T B $$

**KOT 적용 시:**

$$\text{OT}\tilde{K}_{ss} = N_s D^T (\Lambda_Y - \rho I_d) D, \quad \text{OT}\tilde{K}_{ts} = \sqrt{N_s N_t} J_{N_t} K_{YY} W_Y D $$

---

#### [Step 5] 분류기 적용

**커널 릿지 회귀(KRR):**

$$\vec{l}_Y = (\triangle\tilde{K}_{ts})(\triangle\tilde{K}_{ss} + \gamma I_{N_s})^{-1} \vec{l}_X $$

**커널 SVM:**

$$\vec{l}_Y = (\triangle\tilde{K}_{ts})(\vec{\alpha} \odot \vec{l}_X) + \vec{b} $$

---

### 2.3 모델 구조

```
원본 데이터 (Source/Target)
        ↓  커널 함수 φ(·)
   RKHS 매핑
        ↓
공분산 기술자 추정 (MLE/CEE)
        ↓
RKHS 정렬 (KWC 또는 KOT)
        ↓
도메인 불변 커널 행렬 K̃ 구성
        ↓
커널 기반 분류기 (SVM/KRR) 학습 및 예측
```

---

### 2.4 성능 향상

**주요 실험 결과:**

| 데이터셋 | KWC (%) | KOT (%) | 최선 비교 기법 (%) |
|----------|---------|---------|-----------------|
| COIL20 (평균) | 88.68 | **90.77** | SKM: 88.75 |
| Office-Caltech SURF (평균) | **50.93** | 50.15 | CORAL: 48.8 |
| Office-Caltech DeCAF6 (평균) | **89.52** | 89.36 | JDOT: 89.04 |
| 20-Newsgroups RBF (평균) | **92.25** | 92.14 | TCA: 88.49 |
| Reuters-21578 RBF (평균) | **79.79** | 78.49 | TKL: 77.52 |

**계산 시간 비교 (Comp vs Rec, 선형 커널):**

| SVM | TCA | SKM | KWC | KOT |
|-----|-----|-----|-----|-----|
| 3.45s | 93.39s | 273.32s | 23.76s | **9.09s** |

### 2.5 한계점

1. **가우시안 분포 가정:** 최적 수송 맵 유도 시 가우시안 분포를 가정하므로, 분포가 크게 다를 경우 이론적 보장 약화
2. **기하학적 구조 미고려:** 논문 자체에서도 향후 과제로 언급 — 데이터의 기하학적 구조(Geometry)를 추가로 고려할 필요
3. **레이블 정보 미활용:** 비지도 방식이므로 소스 레이블의 클래스별 조건부 분포 정렬 미수행
4. **하이퍼파라미터 민감성:** $\rho$, $d$, $\gamma$ 등 여러 하이퍼파라미터 선택에 의존

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Out-of-Sample 일반화

본 논문의 핵심 강점 중 하나는 **귀납적(Inductive) 설정**에서의 일반화 능력이다.

기존 최적 수송 기반 방법(Courty et al., 2017)은 새 데이터가 들어올 때마다 최적화 문제를 재계산해야 하는 transductive 방식이었다. 반면, 본 논문의 방법은 이미 학습된 $\Psi_{s \to t}$에 새로운 타겟 샘플 $\bar{Y}^t$의 내적을 직접 계산:

$$k\tilde{T}_\triangle(\Psi_{\bar{Y}}) \text{와 } \Psi_{s \to t} \text{의 내적을 재계산 없이 직접 획득}$$

**Out-of-Sample 실험 결과 (DeCAF6 평균 정확도):**

| 방법 | SVM | TCA | GFK | KWC | KOT |
|------|-----|-----|-----|-----|-----|
| 평균 | 78.0 | 82.5 | 83.7 | **87.8** | 86.6 |

### 3.2 일반화 성능 향상의 이론적 근거

**분포 이동 최소화:**

$k\hat{T}_\triangle$는 변환된 소스 공분산 기술자와 타겟 공분산 기술자를 정렬:

$$k\hat{T}_\triangle(\text{MC}_s) \approx \text{MC}_t$$

이는 두 도메인의 2차 통계량(Second-order Statistics)을 RKHS에서 일치시키므로, **커널 기반 분류기의 일반화 오차(Generalization Error) 상한을 낮추는 효과**가 있다. 도메인 적응 이론(Ben-David et al.)에 의하면, 도메인 간 분포 차이가 줄어들수록 타겟 도메인에서의 오차 상한도 감소한다.

**비선형 구조 포착:**

$$k(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}_K}$$

RBF 커널 등을 사용하면 무한 차원 특징 공간에서 비선형 고차 통계량까지 정렬 가능하므로, 선형 CORAL보다 복잡한 도메인 시프트에 대응 가능.

**저차원 부분공간 활용:**

CEE 추정기에서 $d \ll N$으로 설정하여 주요 고유구조만 사용하면, 노이즈 방향의 분포 차이에 과적합(Overfitting)되는 것을 방지하여 일반화 성능 향상에 기여.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

**① 커널 방법과 도메인 적응의 가교 역할**

이 논문은 RKHS 공분산 정렬이라는 이론적 틀을 제시하여, 이후 커널 기반 도메인 적응 연구의 기반이 되었다. 특히 비벡터(non-vector) 데이터(그래프, 문자열, 매니폴드 등)에 대한 도메인 적응 연구의 방향을 제시하였다.

**② 딥러닝 시대와의 접목 가능성**

DeCAF6 특징(심층 신경망 출력)을 입력으로 사용했을 때 높은 성능을 보였다는 것은, **딥러닝으로 추출된 특징에 커널 정렬을 후처리(Post-processing)로 적용하는 파이프라인**의 유효성을 시사한다.

**③ 최적 수송 이론의 심층 활용 촉진**

RKHS에서의 최적 수송 맵 유도는, 이후 **Sliced Wasserstein Distance**, **Sinkhorn Algorithm** 등을 활용한 더 정교한 분포 정렬 방법들의 이론적 배경이 되었다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> **주의:** 아래 연구들은 본 논문과 관련된 연구 흐름을 서술하나, 개별 논문의 구체적 수치나 세부 내용에 대해서는 해당 원문 확인을 권장한다.

**① Deep CORAL (Sun & Saenko, 2016 이후 확장)**

- 딥러닝 레이어 내에서 공분산 정렬을 End-to-End로 수행
- 본 논문의 CORAL 확장과 유사한 방향이나, 본 논문은 RKHS를 통해 비선형성을 더 엄밀하게 다룸

**② DANN / CDAN (Ganin et al., 2016; Long et al., 2018)**

- 도메인 판별자(Domain Discriminator)를 이용한 적대적 학습 방식
- 본 논문과 달리 레이블 조건부 분포까지 정렬하지만, 이론적 보장이 상대적으로 약함

**③ DeepJDOT (Damodaran et al., 2018)**

- JDOT를 딥러닝에 통합하여 특징 공간과 레이블 공간을 동시에 정렬
- 본 논문의 KOT와 이론적으로 연관되나, 딥 네트워크 내에서 동작

**④ Optimal Transport for Domain Adaptation (관련 후속 연구들)**

- Sinkhorn 정규화를 활용한 계산 효율화
- 본 논문의 KOT가 정확한 최적 수송을 추구한 것과 달리 근사적 접근

**비교 표:**

| 방법 | 비선형 정렬 | Out-of-Sample | 이론적 보장 | 계산 효율성 | 레이블 활용 |
|------|-----------|--------------|------------|------------|------------|
| **KWC/KOT (본 논문)** | ✅ | ✅ | ✅ (Theorem 1,2) | ✅ (KOT) | ❌ |
| CORAL | ❌ | ✅ | 제한적 | ✅ | ❌ |
| DANN | ✅ | ✅ | 제한적 | 보통 | ✅ |
| DeepJDOT | ✅ | ✅ | 제한적 | 낮음 | ✅ |

### 4.3 향후 연구 시 고려할 점

**① 기하학적 구조와의 결합**

저자들도 향후 과제로 언급했듯, **리만 기하학(Riemannian Geometry)**을 RKHS 공분산 정렬에 통합하면 데이터의 매니폴드 구조를 보존하면서 더 정확한 도메인 정렬이 가능하다.

$$\text{고려 가능한 목적함수: } \min_T \underbrace{d(\text{MC}_{s}^T, \text{MC}_t)}_{\text{공분산 정렬}} + \lambda \underbrace{\mathcal{R}(T)}_{\text{기하학적 정규화}}$$

**② 클래스 조건부 정렬**

현재 방법은 주변 분포(Marginal Distribution) 정렬에 집중하나, 클래스별 조건부 분포 $P(Y|X)$까지 정렬하면 성능 향상 기대 가능.

**③ 딥러닝과의 End-to-End 통합**

RKHS 정렬 손실을 딥러닝 학습 목적함수에 통합:

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \alpha \| k\hat{T}_\triangle(\text{MC}_s) - \text{MC}_t \|_F^2$$

**④ 다중 소스/타겟 도메인 확장**

복수의 소스 도메인이나 타겟 도메인이 있는 경우, RKHS 공분산 정렬을 확장한 다중 도메인 프레임워크 개발 필요.

**⑤ 커널 선택 자동화**

적합한 커널 함수 선택이 성능에 크게 영향을 미치므로, **메타 학습(Meta-learning)** 또는 **Neural Tangent Kernel** 등을 활용한 자동 커널 설계 연구가 필요하다.

---

## 참고 자료

1. **Zhang, Z., Wang, M., Huang, Y., & Nehorai, A. (2018).** "Aligning Infinite-Dimensional Covariance Matrices in Reproducing Kernel Hilbert Spaces for Domain Adaptation." *CVPR 2018*, pp. 3437–3445. *(본 논문 원문)*

2. **Sun, B., Feng, J., & Saenko, K. (2016).** "Return of Frustratingly Easy Domain Adaptation." *AAAI*, Vol. 6, p. 8.

3. **Courty, N., Flamary, R., Tuia, D., & Rakotomamonjy, A. (2017).** "Optimal Transport for Domain Adaptation." *IEEE Transactions on PAMI*, 39(9):1853–1865.

4. **Pan, S. J., Tsang, I. W., Kwok, J. T., & Yang, Q. (2011).** "Domain Adaptation via Transfer Component Analysis." *IEEE Transactions on Neural Networks*, 22(2):199–210.

5. **Dowson, D. & Landau, B. (1982).** "The Fréchet Distance between Multivariate Normal Distributions." *Journal of Multivariate Analysis*, 12(3):450–455.

6. **Harandi, M., Salzmann, M., & Porikli, F. (2014).** "Bregman Divergences for Infinite Dimensional Covariance Matrices." *CVPR 2014*, pp. 1003–1010.

7. **Zhou, S. K. & Chellappa, R. (2006).** "From Sample Similarity to Ensemble Similarity: Probabilistic Distance Measures in RKHS." *IEEE Transactions on PAMI*, 28(6):917–929.
