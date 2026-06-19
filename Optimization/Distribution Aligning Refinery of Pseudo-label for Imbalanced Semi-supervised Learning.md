# Distribution Aligning Refinery of Pseudo-label (DARP) for Imbalanced Semi-supervised Learning

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 반지도학습(SSL) 알고리즘들은 **클래스 분포가 균형적이라는 가정** 하에 설계되었으나, 실제 환경에서는 클래스 불균형이 흔하게 발생한다. 이 논문은 불균형 환경에서 SSL이 다수 클래스(majority class)에 편향된 의사 레이블(pseudo-label)을 생성하여 소수 클래스(minority class)의 성능을 오히려 **기준 모델(labeled only)보다 악화**시킬 수 있음을 최초로 체계적으로 규명하고, 이를 해결하는 DARP 알고리즘을 제안한다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **문제 규명** | 불균형 SSL에서 pseudo-label의 편향이 진짜 클래스 불균형비보다 훨씬 크다는 것을 실증 ($\gamma=150$ → pseudo-label $\gamma=1046$) |
| **알고리즘 제안** | DARP: pseudo-label을 실제 클래스 분포에 맞게 "부드럽게" 정제하는 볼록 최적화 기반 반복 알고리즘 |
| **이론적 보장** | Theorem 1을 통해 알고리즘의 수렴성 및 유일 최적해 수렴을 증명 |
| **범용성** | MixMatch, ReMixMatch, FixMatch 등 임의의 SSL 알고리즘에 플러그인 방식으로 적용 가능 |
| **성능** | 최대 77.2%(MixMatch), 31.4%(ReMixMatch), 53.1%(FixMatch)의 상대적 테스트 오류 감소 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**설정:** $K$개 클래스 분류 문제. 레이블 데이터셋 $\mathcal{D}^{\text{labeled}} = \{(x\_n^{\text{labeled}}, y_n^{\text{labeled}})\}\_{n=1}^{N}$, 비레이블 데이터셋 $\mathcal{D}^{\text{unlabeled}} = \{x_m^{\text{unlabeled}}\}_{m=1}^{M}$.

클래스 $k$의 레이블/비레이블 데이터 수를 각각 $N_k$, $M_k$라 할 때, **클래스 불균형 비율**:

$$\gamma_l = \frac{\max_k N_k}{\min_k N_k}, \quad \gamma_u = \frac{\max_k M_k}{\min_k M_k} \gg 1$$

**핵심 문제:** 편향된 모델이 생성한 pseudo-label $\{\hat{y}_m^{\text{unlabeled}}\}$의 분포가 진짜 레이블 분포보다 훨씬 더 불균형하여, SSL이 소수 클래스에 오히려 해롭게 작용:

$$\hat{y}_m^{\text{unlabeled}} \in [0,1]^K, \quad \sum_{k=1}^{K} \hat{y}_m^{\text{unlabeled}}(k) = 1$$

---

### 2.2 제안 방법 (수식 포함)

#### (1) 핵심 최적화 문제

주어진 원본 pseudo-label $\{\hat{y}\_m^{\text{unlabeled}}\}$을 정제하여 실제 클래스 분포 $\{M_k\}_{k=1}^K$에 맞추는 볼록 최적화:

$$\underset{\{\hat{y}_m\}_{m=1}^M}{\text{minimize}} \quad \sum_{m=1}^{M} w_m D_{KL}(\hat{y}_m \| \hat{y}_m^{\text{unlabeled}}) \tag{1}$$

$$\text{subject to} \quad \sum_{m=1}^{M} \hat{y}_m(k) = M_k, \; \forall k, \quad \sum_{k=1}^{K} \hat{y}_m(k) = 1, \; \forall m, \quad \hat{y}_m(k) \in [0,1], \; \forall m,k$$

- **목적함수:** KL-divergence를 최소화하여 원본 pseudo-label 정보 보존
- **제약 조건 1:** $\sum_{m=1}^M \hat{y}_m(k) = M_k$ → 실제 클래스 분포 매칭
- **가중치:** 더 확신이 높은(엔트로피가 낮은) pseudo-label에 더 큰 가중치 부여

$$w_m := \left(H(\hat{y}_m^{\text{unlabeled}})\right)^{-1}$$

여기서 $H(\cdot)$는 Shannon 엔트로피.

#### (2) 소음 항목 제거 (노이즈 필터링)

클래스 $k$에 대해 상위 $\delta \cdot M_k$개의 비레이블 데이터 집합 $\mathcal{U}_k$를 정의하고:

$$\hat{y}_m^0(k) \leftarrow \begin{cases} \hat{y}_m^{\text{unlabeled}}(k) & \text{if } x_m^{\text{unlabeled}} \in \mathcal{U}_k \\ 0 & \text{otherwise} \end{cases} \tag{2}$$

이는 엔트로피 최소화(entropy minimization)와 유사한 효과를 가진다.

#### (3) 이중 좌표 상승법 (Dual Coordinate Ascent)

최적화 (1)의 라그랑지안 쌍대(Lagrangian dual)를 유도하면:

$$\underset{\lambda \in \mathbb{R}^n, \nu \in \mathbb{R}^m}{\text{maximize}} \; g(\lambda, \nu) = -\sum_{i=1}^n \sum_{j=1}^m w_i A_{ij} e^{-\frac{w_i + \lambda_i + \nu_j}{w_i}} - \sum_{i=1}^n \lambda_i r_i - \sum_{j=1}^m \nu_j c_j \tag{4}$$

변수 치환 $\alpha_m = e^{-\frac{\lambda_m + w_m}{w_m}}$, $\beta_k = e^{-\nu_k}$을 통해 Algorithm 1이 이 쌍대 문제의 좌표 상승법임을 보인다:

**홀수 반복:** 

$$\alpha_m^t \leftarrow \left(\sum_{k=1}^K \hat{y}_m^0(k)(\beta_k^{t-1})^{\frac{1}{w_m}}\right)^{-1}$$

**짝수 반복:** 

$$\beta_k^t \leftarrow \text{SolveZ}_{\geq 0}\left(\sum_{m=1}^M \hat{y}_m^0(k)\alpha_m^{t-1} Z^{\frac{1}{w_m}} - M_k\right)$$

**최종 출력:**

$$\hat{y}_m^{\text{out}}(k) \leftarrow \hat{y}_m^0(k) \alpha_m^T (\beta_k^T)^{\frac{1}{w_m}}$$

**Theorem 1.** \*Algorithm 1의 출력은 $T \rightarrow \infty$에 따라 (1)의 유일 최적해로 수렴한다. (단, 모든 실행 가능한 $\{\hat{y}\_m\}$에 대해 $\sum_{m=1}^M w_m D_{KL}(\hat{y}_m \| \hat{y}_m^{\text{unlabeled}}) = \infty$인 경우 제외)*

#### (4) 비레이블 데이터의 클래스 분포 추정

$\{M_k\}$를 알 수 없을 때, 혼동 행렬(confusion matrix) $C^{\text{unlabeled}} \in \mathbb{R}^{K \times K}$을 이용:

$$\begin{bmatrix} M_1 \\ \vdots \\ M_K \end{bmatrix} = \left(C^{\text{unlabeled}}\right)^{-1} \times \begin{bmatrix} \sum_{m=1}^M f_1(x_m^{\text{unlabeled}}) \\ \vdots \\ \sum_{m=1}^M f_K(x_m^{\text{unlabeled}}) \end{bmatrix}$$

$$C_{ij}^{\text{unlabeled}} := \frac{\sum_{m: y_m^{\text{unlabeled}}(j)=1} f_i(x_m^{\text{unlabeled}})}{|\{m \mid y_m^{\text{unlabeled}}(j) = 1\}|}$$

실제로는 레이블 데이터셋으로 $C^{\text{unlabeled}}$을 근사.

---

### 2.3 모델 구조

DARP는 **독립적인 플러그인 모듈**로, 특정 백본 아키텍처를 변경하지 않는다:

```
[임의의 SSL 알고리즘] → 원본 pseudo-label 생성
        ↓
[DARP 모듈]
  Step 1: 노이즈 항목 제거 (δ 하이퍼파라미터)
  Step 2: DualCoordinateAscent 실행 (T=10 반복)
  Step 3: 정제된 pseudo-label 출력
        ↓
[모델 학습에 정제된 pseudo-label 사용]
```

- **백본:** Wide ResNet-28-2 (실험 기준)
- **적용 주기:** 매 10 iteration마다 DARP 실행
- **워밍업:** 학습 초반 40% 구간에서는 DARP 비적용 (pseudo-label이 아직 불안정하므로)
- **추가 연산 비용:** 기존 SSL 알고리즘 대비 최대 20% 추가 시간

---

### 2.4 성능 향상

#### CIFAR-10 ($\gamma_l = \gamma_u$)

| 알고리즘 | $\gamma=50$ (bACC/GM) | $\gamma=100$ (bACC/GM) | $\gamma=150$ (bACC/GM) |
|----------|----------------------|----------------------|----------------------|
| MixMatch | 73.2/68.9 | 64.8/49.0 | 62.5/42.5 |
| **MixMatch+DARP** | **75.2/72.8** | **67.9/61.2** | **65.8/56.5** |
| FixMatch | 79.2/77.8 | 71.5/66.8 | 68.4/59.9 |
| **FixMatch+DARP** | **81.8/80.9** | **75.5/73.0** | **70.4/64.9** |

#### CIFAR-10 ($\gamma_l=100$, 다양한 $\gamma_u$, 상대적 오류 감소율)

| 알고리즘 | $\gamma_u=1$ | $\gamma_u=50$ | $\gamma_u=150$ | $\gamma_u=100$ (reversed) |
|----------|-------------|--------------|---------------|--------------------------|
| MixMatch+DARP | **-77.2%/-84.4%** | -11.8%/-27.0% | -3.62%/-15.7% | -48.0%/-63.6% |
| ReMixMatch*+DARP | -31.4%/-32.5% | -1.72%/-1.49% | -1.53%/-2.64% | -19.5%/-22.5% |
| FixMatch+DARP | -53.1%/-73.8% | -13.3%/-17.0% | -10.9%/-18.4% | -31.3%/-60.3% |

---

### 2.5 한계

1. **분포 추정의 한계:** 레이블/비레이블 데이터가 동일 입력 분포에서 나온다는 가정에 의존. 분포 이동(distribution shift)이 클 경우 혼동 행렬 추정이 부정확할 수 있음.
2. **하이퍼파라미터 민감성:** $\delta$ (노이즈 제거 비율)에 따라 성능이 달라짐. 작은 $\delta$는 편향 감소, 큰 $\delta$는 원본 정보 보존.
3. **계산 가정:** 비레이블 데이터의 진짜 클래스 분포($\{M_k\}$)를 알거나 추정해야 함.
4. **초기 단계 제외:** 학습 초반 40%에서 DARP 미적용 → 초기 학습의 효율성 제한.
5. **벤치마크 한계:** 주로 CIFAR-10/100, STL-10 등 이미지 분류에 집중. NLP, 의료 등 타 도메인 검증 부족.
6. **동적 분포 변화:** 학습 중 분포가 동적으로 변하는 경우에 대한 적응 메커니즘 부재.

---

## 3. 모델 일반화 성능 향상 관련 분석

### 3.1 일반화 향상 메커니즘

DARP는 다음 세 가지 측면에서 일반화 성능을 향상시킨다:

#### (a) 소수 클래스에 대한 편향 보정

균형 테스트(balanced test criterion) 기준 정확도(bACC)를 최적화하기 위해, pseudo-label의 분포를 실제 분포에 강제로 맞춤. 이는 소수 클래스에 대한 표현 학습을 강화:

$$\text{bACC} = \frac{1}{K}\sum_{k=1}^K \text{TPR}_k$$

DARP 적용 후 소수 클래스의 혼동 행렬이 대각선에 집중되는 현상을 Figure 2에서 확인 가능.

#### (b) 정보 보존과 분포 정렬의 균형

KL-divergence를 목적함수로 사용함으로써 원본 pseudo-label의 개별 품질을 최대한 보존하면서 전체 분포를 조정. 이는 단순히 클래스 레이블을 재배정하는 방법보다 정보 손실이 적다.

#### (c) 엔트로피 정규화 효과

노이즈 항목 제거($\delta$ 파라미터)는 엔트로피 최소화(entropy minimization, Grandvalet & Bengio, 2005)와 유사한 효과를 통해 결정 경계를 날카롭게 만든다. 더 자신 있는 예측에 더 큰 가중치($w_m$)를 부여함으로써 신뢰할 수 있는 pseudo-label의 영향력을 증가시킨다.

#### (d) 분포 불일치($\gamma_l \neq \gamma_u$) 상황에서의 일반화

레이블-비레이블 데이터의 분포가 다를수록 DARP의 효과가 더 크게 나타남. STL-10에서 ReMixMatch+DARP가 최대 36.0%/44.0% 오류 감소를 달성한 것은, 비레이블 데이터의 분포를 올바르게 활용할 때 일반화 성능이 크게 향상됨을 보여준다.

### 3.2 일반화와 편향-분산 트레이드오프

$$\text{(일반화 오류)} \approx \text{bias}^2 + \text{variance} + \text{noise}$$

DARP는 주로 **bias 감소**에 기여:
- 다수 클래스 편향 → **분포 정렬을 통한 편향 감소**
- 소수 클래스 과소 표현 → **$M_k$ 제약을 통한 균형 있는 학습**

단, 소수 클래스에 억지로 pseudo-label을 할당하는 과정에서 **variance 증가** 가능성 존재 (이는 $\delta$ 파라미터로 제어).

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (a) 문제 인식의 전환
이 논문은 SSL 연구에서 **"클래스 불균형 + 반지도학습"의 교차점**을 정식 연구 문제로 확립했다. 이전까지 두 분야는 독립적으로 연구되었으나, DARP는 두 문제를 통합적으로 다루는 첫 번째 체계적 접근법이다.

#### (b) 플러그인 패러다임의 확산
"임의의 SSL 알고리즘에 적용 가능한 후처리 모듈" 패러다임은 이후 연구들에 영향을 미쳐, 다음과 같은 후속 연구들이 등장:

- **ABC (Appendix-Balanced Classifier, Lee et al., 2021):** 불균형 SSL에서 보조 균형 분류기 활용
- **CoSSL (Cui et al., 2022):** 클래스 불균형 SSL을 위한 공동 학습 프레임워크
- **DASO (Oh et al., 2022):** 분포 인식 의미론적 과적합 방지

#### (c) 평가 지표의 표준화
balanced accuracy(bACC)와 geometric mean(GM)을 불균형 SSL의 표준 평가 지표로 제시함으로써, 이후 연구들의 공정한 비교를 가능하게 했다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래 내용은 제공된 논문 원문 외에 제가 학습한 지식 기반의 분석입니다. 일부 세부 수치나 내용은 부정확할 수 있으므로, 반드시 원본 논문을 직접 확인하시기 바랍니다.

| 논문 | 핵심 아이디어 | DARP와의 차이 |
|------|--------------|--------------|
| **CReST (Wei et al., CVPR 2021)** | 자기 훈련 기반 점진적 클래스 균형화 | 분포 추정을 반복적 자기훈련으로 해결; DARP는 최적화 기반 |
| **DASO (Oh et al., CVPR 2022)** | 의미론적 분포 인식 pseudo-label | 특징 공간의 의미 구조 활용; DARP는 분포 통계만 사용 |
| **CoSSL (Cui et al., ECCV 2022)** | 불균형 SSL을 위한 공동 학습 | 표현 학습과 분류기 학습을 동시에 균형화 |
| **ACR (Wei et al., NeurIPS 2023)** | 적응적 신뢰도 기반 pseudo-label | 동적으로 confidence threshold 조정 |
| **SimMatch (Zheng et al., CVPR 2022)** | 인스턴스-클래스 수준 일관성 | 인스턴스 유사성 기반 soft pseudo-label |

**핵심 차이점 비교:**

```
DARP: 분포 수준 정렬 (distribution-level alignment)
       → 볼록 최적화, 이론적 보장, 플러그인 방식

후속 연구들: 특징 공간 + 분포 + 동적 임계값을 통합
            → 더 복잡한 학습 파이프라인, end-to-end 학습
```

---

### 4.3 향후 연구 시 고려할 점

#### (a) 분포 추정 개선
현재 DARP는 혼동 행렬 기반 분포 추정에 의존하나, 이는 레이블/비레이블 데이터가 동일 입력 분포에서 나온다는 강한 가정을 요구한다. 다음을 고려할 수 있다:
- **Optimal Transport 기반 분포 추정**으로 더 강건한 추정 가능
- **베이지안 추정**을 통한 불확실성 정량화

#### (b) 동적 분포 적응
학습 과정에서 pseudo-label의 품질이 향상됨에 따라 목표 분포 자체가 변화할 수 있다. **온라인 최적화** 프레임워크로의 확장이 필요하다.

#### (c) 오픈셋 불균형 SSL
실제 환경에서는 비레이블 데이터에 **학습 클래스에 없는 샘플(out-of-distribution)** 이 포함될 수 있다. 이를 동시에 처리하는 통합 프레임워크가 필요하다.

#### (d) 장기 꼬리 분포의 다양한 패턴
논문은 지수적 감소(exponential decay) 패턴만 실험. **Pareto 분포, 스텝 불균형** 등 다양한 패턴에서의 검증이 필요하다.

$$N_k = N_1 \cdot \gamma_l^{-\frac{k-1}{K-1}} \quad \text{(현재 논문의 설정)}$$

다양한 $N_k$ 패턴에 대한 일반화 필요.

#### (e) 대형 언어 모델(LLM) 시대의 적용
LLM을 활용한 반지도학습(예: 프롬프트 기반 pseudo-label 생성)에서도 클래스 불균형 문제가 발생할 수 있다. DARP의 최적화 프레임워크를 LLM 기반 자기 훈련에 적용하는 방향이 흥미로운 연구 방향이다.

#### (f) 다중 레이블 및 계층적 분류로의 확장
현재 DARP는 단일 레이블, 단일 수준 분류 문제에 한정. **다중 레이블 불균형 SSL** 및 **계층적 분류**로의 확장이 실용적 가치를 높일 것이다.

#### (g) 공정성(Fairness)과의 연계
소수 클래스 = 취약 집단(underrepresented group)으로 볼 때, DARP의 편향 제거 메커니즘은 **알고리즘적 공정성** 연구와 연결될 수 있다. 단순 정확도 외에 공정성 지표(demographic parity, equalized odds 등)를 목적함수에 통합하는 방향을 고려해야 한다.

---

## 참고 자료

**주요 참고 논문 (논문 원문에서 인용된 것들):**

1. **Kim, J. et al. (2020).** "Distribution Aligning Refinery of Pseudo-label for Imbalanced Semi-supervised Learning." *NeurIPS 2020.* arXiv:2007.08844v2

2. **Berthelot, D. et al. (2019).** "MixMatch: A Holistic Approach to Semi-Supervised Learning." *NeurIPS 2019.*

3. **Berthelot, D. et al. (2020).** "ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring." *ICLR 2020.*

4. **Sohn, K. et al. (2020).** "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence." arXiv:2001.07685.

5. **Cao, K. et al. (2019).** "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss." *NeurIPS 2019.*

6. **Kang, B. et al. (2020).** "Decoupling Representation and Classifier for Long-Tailed Recognition." *ICLR 2020.*

7. **Tarvainen, A. & Valpola, H. (2017).** "Mean teachers are better role models." *NeurIPS 2017.*

8. **Miyato, T. et al. (2018).** "Virtual Adversarial Training." *IEEE TPAMI 2018.*

9. **Grandvalet, Y. & Bengio, Y. (2005).** "Semi-supervised Learning by Entropy Minimization." *NeurIPS 2005.*

10. **Boyd, S. & Vandenberghe, L. (2004).** *Convex Optimization.* Cambridge University Press.
