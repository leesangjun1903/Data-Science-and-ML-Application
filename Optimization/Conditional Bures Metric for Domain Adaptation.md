# Conditional Bures Metric for Domain Adaptation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(You-Wei Luo & Chuan-Xian Ren, 2021)의 핵심 주장은 다음과 같습니다:

> **기존 UDA 방법들은 주변 분포(marginal distribution) $P^s_X \neq P^t_X$의 불일치만 고려하고, 레이블 분포에 포함된 판별 정보(discriminant information)를 무시한다. 이를 해결하기 위해 조건부 분포(conditional distribution) $P^s_{X|Y} \neq P^t_{X|Y}$의 불일치를 측정할 수 있는 새로운 메트릭인 Conditional Kernel Bures (CKB) 메트릭을 제안한다.**

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **이론적 기여** | RKHS에서의 조건부 분포 불일치를 측정하는 CKB 메트릭 정의 및 kernel embedding 성질 증명 |
| **추정 기여** | 암묵적 feature map 없이 계산 가능한 CKB 메트릭의 경험적 추정식 및 일관성(consistency) 이론 제시 |
| **실용적 기여** | CKB 메트릭 기반 조건부 분포 정합 네트워크(CKB Network) 구축 및 SOTA 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**조건부 분포 이동(Conditional Distribution Shift) 문제**

기존 UDA 방법의 문제점은 다음과 같습니다:

$$P^s_X \neq P^t_X \quad \text{(주변 분포 이동만 고려)}$$

그러나 실제로는 조건부 분포도 이동합니다:

$$P^s_{X|Y} \neq P^t_{X|Y}$$

이를 무시하면 Figure 1에서 보듯이, 클래스 경계가 잘못 정렬(misaligned conditional distribution)되어 분류 성능이 저하됩니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 조건부 공분산 연산자 (Conditional Covariance Operator)

RKHS $\mathcal{H}\_X$에서의 교차 공분산 연산자(cross-covariance operator) $\mathbf{R}_{XY}: \mathcal{H}_Y \rightarrow \mathcal{H}_X$는 다음과 같이 정의됩니다:

$$\mathbf{R}_{XY} = \mathbb{E}_{XY}\left[(\phi(X) - \mu_X) \otimes (\psi(Y) - \mu_Y)\right]$$

조건부 공분산 연산자는:

$$\mathbf{R}_{XX|Y} = \mathbf{R}_{XX} - \mathbf{R}_{XY}\mathbf{R}^{-1}_{YY}\mathbf{R}_{YX}$$

이 연산자는 다음을 만족합니다:

$$\langle f, \mathbf{R}_{XX|Y} f \rangle_{\mathcal{H}_Y} = \mathbb{E}_Y\left[\mathrm{Var}_{X|Y}[f(X)|Y]\right], \quad \forall f \in \mathcal{H}_X$$

---

#### Step 2: Bures 메트릭에서 CKB 메트릭으로의 확장

**기존 Bures 메트릭** (PSD 행렬 공간 $\mathbb{S}^+(d)$에서):

$$d^2_B(\Sigma^s_{XX}, \Sigma^t_{XX}) = \mathrm{tr}\left(\Sigma^s_{XX} + \Sigma^t_{XX} - 2\Sigma^{st}_{XX}\right)$$

여기서 $\Sigma^{st}\_{XX} = \sqrt{\sqrt{\Sigma^s_{XX}}\Sigma^t_{XX}\sqrt{\Sigma^s_{XX}}}$

**Kernel Bures 메트릭** (무한차원 RKHS로 확장):

$$d^2_{KB}(\mathbf{R}^s_{XX}, \mathbf{R}^t_{XX}) = \mathrm{tr}\left(\mathbf{R}^s_{XX} + \mathbf{R}^t_{XX} - 2\mathbf{R}^{st}_{XX}\right)$$

**[Definition 1] Conditional Kernel Bures (CKB) 메트릭** (핵심 기여):

$$\boxed{d^2_{CKB}(\mathbf{R}^s_{XX|Y}, \mathbf{R}^t_{XX|Y}) = \mathrm{tr}\left(\mathbf{R}^s_{XX|Y} + \mathbf{R}^t_{XX|Y} - 2\mathbf{R}^{st}_{XX|Y}\right)}$$

여기서:

$$\mathbf{R}^{st}_{XX|Y} = \sqrt{\sqrt{\mathbf{R}^s_{XX|Y}}\mathbf{R}^t_{XX|Y}\sqrt{\mathbf{R}^s_{XX|Y}}}$$

**[Theorem 1] CKB 메트릭의 조건부 분포 메트릭 성질:**

$(X, \mathcal{B}_X)$가 locally compact Hausdorff 공간이고, $k$가 $c_0$-universal 커널이며, $(\phi(X), \psi(Y))$가 $\mathcal{H}_X \oplus \mathcal{H}_Y$에서 Gaussian 랜덤 변수라고 가정하면:

$$d_{CKB}(\mathbf{R}^s_{XX|Y}, \mathbf{R}^t_{XX|Y}) = 0 \implies P^s_{X|Y} = P^t_{X|Y}$$

---

#### Step 3: CKB 메트릭의 경험적 추정 (Empirical Estimation)

정규화된 조건부 공분산 추정:

$$\hat{\mathbf{R}}_{XX|Y} = \hat{\mathbf{R}}_{XX} - \hat{\mathbf{R}}_{XY}\left(\hat{\mathbf{R}}_{YY} + \varepsilon I\right)^{-1}\hat{\mathbf{R}}_{YX}$$

보조 행렬 정의:

$$\mathbf{B}_s \triangleq I_n - \frac{1}{n\varepsilon}\left[\mathbf{G}^s_Y - \mathbf{G}^s_Y\left(\mathbf{G}^s_Y + \varepsilon n I_n\right)^{-1}\mathbf{G}^s_Y\right] = \varepsilon n \left(\mathbf{G}^s_Y + \varepsilon n I_n\right)^{-1}$$

$$\mathbf{B}_t \triangleq \varepsilon m \left(\mathbf{G}^t_Y + \varepsilon m I_m\right)^{-1}$$

여기서 $\mathbf{G}^s_{X/Y} = H_n K^s_{XX/YY} H_n$, $\mathbf{G}^t_{X/Y} = H_m K^t_{XX/YY} H_m$는 중앙화 커널 행렬입니다.

고유값 분해(EVD)로 $\mathbf{B}_s = \mathbf{U}_s \mathbf{D}_s \mathbf{U}^T_s = \mathbf{C}_s \mathbf{C}^T_s$를 얻고, 조건부 공분산 연산자를 재공식화하면:

$$\hat{\mathbf{R}}^s_{XX|Y} = \frac{1}{n}\Phi_s H_n \mathbf{C}_s (\Phi_s H_n \mathbf{C}_s)^T$$

**[Theorem 2] CKB 메트릭의 명시적 경험적 추정:**

$$\hat{d}^2_{CKB}(\hat{\mathbf{R}}^s_{XX|Y}, \hat{\mathbf{R}}^t_{XX|Y}) = \varepsilon \cdot \mathrm{tr}\left[\mathbf{G}^s_X(\varepsilon n I_n + \mathbf{G}^s_Y)^{-1}\right] + \varepsilon \cdot \mathrm{tr}\left[\mathbf{G}^t_X(\varepsilon m I_m + \mathbf{G}^t_Y)^{-1}\right] - \frac{2}{\sqrt{nm}}\left\|(H_m \mathbf{C}_t)^T K^{ts}_{XX}(H_n \mathbf{C}_s)\right\|_*$$

여기서 $\|\cdot\|_*$는 핵 노름(nuclear norm)입니다.

---

#### Step 4: 수렴 보장 (Convergence Guarantee)

**[Theorem 3] CKB 메트릭 추정의 일관성:**

정규화 파라미터 $\varepsilon_{n'}$이 $\varepsilon_{n'} \to 0$, $\varepsilon_{n'}\sqrt{n'} \to \infty$ ($n' = \min\{n,m\} \to \infty$)를 만족하면:

$$\left|\hat{D}^{(n')}_{CKB} - D_{CKB}\right| \to 0 \quad (n' \to \infty)$$

수렴 속도는 $\left(\frac{1}{\varepsilon'_n \sqrt{n'}}\right)^{1/2}$입니다.

---

### 2.3 모델 구조 (Conditional Distribution Matching Network)

모델은 3가지 손실 함수의 합을 최소화합니다:

**1) 소스 도메인 분류 손실 (Cross-Entropy):**

$$\mathcal{L}_{CE} = \sum_{i=1}^{K}\sum_{j=1}^{n} -y^s_{ij}\log \hat{y}^s_{ij}$$

**2) 타겟 도메인 엔트로피 최소화 손실:**

$$\mathcal{L}_{Ent} = \sum_{i=1}^{K}\sum_{j=1}^{m} -\hat{y}^t_{ij}\log \hat{y}^t_{ij}$$

**3) CKB 정합 손실:**

$$\mathcal{L}_{CKB} = \hat{d}^2_{CKB}(\hat{\mathbf{R}}^s_{XX|Y}, \hat{\mathbf{R}}^t_{XX|Y})$$

**[모델 CKB] 조건부 정합 목적 함수:**

$$\min_{F, C} \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{Ent} + \lambda_2 \mathcal{L}_{CKB}$$

**[모델 CKB+MMD] 결합 분포 정합까지 고려한 목적 함수:**

$$\min_{F, C} \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{Ent} + \lambda_2(\mathcal{L}_{CKB} + \mathcal{L}_{MMD})$$

여기서 $\mathcal{L}\_{MMD} = \|\Psi_s \mathbf{1}\_n/n - \tilde{\Psi}\_t \mathbf{1}\_m/m\|^2_{\mathcal{H}_Y}$는 레이블 주변 분포 정합 항입니다.

---

### 2.4 성능 향상

| 데이터셋 | 백본 | 기존 SOTA | CKB | CKB+MMD |
|---------|------|-----------|-----|---------|
| Office-Home | ResNet-50 | DMP: 68.1% | 68.5% | **68.7%** |
| ImageCLEF-DA | ResNet-50 | ETD: 89.7% | **90.2%** | 89.7% |
| Office10 | AlexNet | DMP: 91.2% | **91.8%** | 91.9% |
| Digits (M→U) | LeNet | ETD: 96.4% | 96.3% | **96.6%** |
| Digits (U→M) | LeNet | CyCADA: 96.5% | **96.6%** | 96.3% |

**시간 복잡도:** CKB 메트릭의 배치 단위 계산 복잡도는 $\mathcal{O}(db^2_s)$로 DNN에 비해 추가 부담이 적습니다.

---

### 2.5 한계점

논문에서 명시적으로 언급된 한계 및 추론 가능한 한계:

1. **가우시안 가정(Gaussian assumption):** Theorem 1의 증명이 $(\phi(X), \psi(Y))$가 RKHS에서 Gaussian 랜덤 변수라는 가정에 의존하며, 이는 실제 데이터에서 항상 성립하지 않을 수 있습니다.
2. **수렴 속도:** 조건부 공분산 연산자의 수렴 속도 $\frac{1}{\varepsilon_n\sqrt{n}}$보다 느린 $\left(\frac{1}{\varepsilon'_n\sqrt{n'}}\right)^{1/2}$의 수렴 속도를 가집니다.
3. **의사 레이블(Pseudo Label) 의존성:** 타겟 도메인의 레이블이 없어 pseudo label을 사용하므로, 초기 예측 정확도가 낮을 경우 CKB 추정 품질이 저하될 수 있습니다.
4. **계산 비용:** 전체 CKB 메트릭의 계산 복잡도가 $\mathcal{O}(\max(c,d,m,n)(n^2+m^2+mn))$으로 대규모 데이터셋에서 확장성 문제가 있을 수 있습니다.
5. **멀티소스 도메인 적응** 및 **오픈셋(open-set)** 시나리오로의 확장이 아직 미탐구 상태입니다.

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 이론적 일반화 보장

Zhao et al.(2019)의 일반화 경계에 기반하여, 조건부 분포 정합이 타겟 도메인의 위험(risk)을 줄임을 이론적으로 지지합니다:

$$\epsilon_t(h) \leq \epsilon_s(h) + d_{CKB}(P^s_{X|Y}, P^t_{X|Y}) + \lambda^*$$

CKB 메트릭을 최소화함으로써 $d_{CKB}$ 항이 감소하고, 이는 직접적으로 타겟 도메인의 일반화 오류를 줄입니다.

### 3.2 판별력(Discriminability) 향상

기존 주변 분포 정합은 아래와 같이 서로 다른 클래스가 같은 위치로 정렬될 수 있습니다:

$$P^s_{X|Y=k} \neq P^t_{X|Y=k} \quad \text{(클래스별 불일치 발생 가능)}$$

CKB는 클래스 조건부 분포를 직접 정합함으로써:

- **클래스 간 분리성(inter-class separability)** 향상
- **클래스 내 밀집성(intra-class compactness)** 향상

t-SNE 시각화(Figure 4)에서 이를 확인할 수 있습니다.

### 3.3 엔트로피 최소화와의 상호 강화

$$\mathcal{L}_{Ent} \downarrow \Rightarrow \text{더 정확한 pseudo label} \Rightarrow \text{더 정확한 } \mathcal{L}_{CKB} \text{ 추정} \Rightarrow \text{더 나은 도메인 정합}$$

이 순환적 강화 메커니즘이 일반화 성능을 지속적으로 향상시킵니다.

### 3.4 OT의 해석 가능성

CKB 메트릭이 곧 조건부 분포 간 Optimal Transport 비용임을 증명하여, 지식 전이(knowledge transfer)의 메커니즘을 직관적으로 해석할 수 있습니다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

**이론적 측면:**
- 조건부 분포의 RKHS 임베딩 이론을 OT 프레임워크로 연결한 최초 시도로서, 이 방향의 후속 이론 연구를 촉진합니다.
- 조건부 공분산 연산자를 이용한 분포 메트릭 설계 패러다임을 제시합니다.

**방법론적 측면:**
- 주변 분포 정합에서 조건부/결합 분포 정합으로의 패러다임 전환을 명확히 합니다.
- 명시적이고 계산 가능한 조건부 분포 불일치 측도 제공으로 다양한 UDA 아키텍처에 플러그인(plug-in)으로 활용 가능합니다.

**응용적 측면:**
- 의료 영상, 자율주행 등 도메인 이동이 심한 실제 응용에서의 활용 가능성 확대

### 4.2 향후 연구 시 고려할 점

| 연구 방향 | 고려 사항 |
|----------|-----------|
| **Gaussian 가정 완화** | 비가우시안 분포에서도 성립하는 이론적 틀 확장 필요 |
| **수렴 속도 개선** | 더 빠른 수렴 속도를 가지는 추정량 설계 |
| **확장성(Scalability)** | 대규모 데이터셋에서의 근사 알고리즘 개발 (예: Nyström 근사, random features) |
| **멀티소스 확장** | 다중 소스 도메인에서의 CKB 정합 이론 확장 |
| **Source-free DA** | 소스 데이터 없이 타겟 도메인에서만 적응하는 시나리오 |
| **의사 레이블 품질** | 초기 pseudo label의 불확실성을 고려한 강건한 CKB 추정 |
| **Vision-Language 모델** | CLIP 등 대형 모델과의 결합에서 조건부 분포 정합 역할 탐구 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 비교는 제공된 논문 내용과 제가 학습한 지식에 기반합니다. **2021년 8월 이후 최신 논문의 구체적 수치는 확인에 한계가 있음을 밝힙니다.**

| 연구 | 방법 | 핵심 아이디어 | CKB와의 차이 |
|------|------|--------------|-------------|
| **CDAN** (Long et al., NeurIPS 2018) | 조건부 적대적 학습 | 다선형 맵(multilinear map)으로 조건부 변수 인코딩 | 적대적 학습 기반, 이론적 보장 약함 |
| **ETD** (Li et al., CVPR 2020) | 강화 전이 거리 | 분류기 예측을 피드백으로 OT 비용 재가중 | 주변 분포 OT, 조건부 분포 비명시적 |
| **DMP** (Luo et al., IEEE TPAMI 2020) | 판별적 다양체 전파 | 구조 학습 기반 도메인 정합 | CKB보다 계산 비용 큼 |
| **SHOT** (Liang et al., ICML 2020) | Source-free DA | 정보 최대화 + 의사 레이블 | 소스 데이터 불필요하나 조건부 분포 명시적 정합 없음 |
| **NWD** (Chen et al., 2022 계열) | 정규화 Wasserstein | 클래스별 분포 매칭 | CKB의 RKHS 기반 이론보다 단순 |
| **CDTrans** (Xu et al., ICLR 2022) | Transformer 기반 DA | Cross-domain attention으로 조건부 정보 캡처 | 명시적 분포 메트릭 없음 |
| **PMTrans** (Zhu et al., ECCV 2022) | Patch Mix Transformer | ViT 기반 도메인 정합 | 대형 모델 활용, 이론적 보장 미흡 |

**CKB의 차별점:**
- 수학적으로 엄밀한 조건부 분포 메트릭 정의 및 일관성 이론 제공
- 암묵적 feature map 없이 명시적으로 계산 가능
- OT 이론과 RKHS 이론의 통합으로 해석 가능성 제공

**CKB의 한계 (최신 연구 대비):**
- Vision Foundation Model(ViT, CLIP 등) 기반 방법들과의 성능 격차 존재 가능
- Source-free, Test-time adaptation 등 새로운 설정으로 확장 필요

---

## 참고자료

1. **논문 원문:** You-Wei Luo, Chuan-Xian Ren. "Conditional Bures Metric for Domain Adaptation." arXiv:2108.00302v1, 2021.
2. Fukumizu, K., Bach, F. R., Jordan, M. I. "Kernel dimension reduction in regression." *The Annals of Statistics*, 37(4):1871–1905, 2009.
3. Zhang, Z., Wang, M., Nehorai, A. "Optimal transport in reproducing kernel hilbert spaces: Theory and applications." *IEEE TPAMI*, 2019.
4. Long, M., Cao, Z., Wang, J., Jordan, M. I. "Conditional adversarial domain adaptation." *NeurIPS*, 2018.
5. Li, M., Zhai, Y.-M., Luo, Y.-W., Ge, P., Ren, C.-X. "Enhanced transport distance for unsupervised domain adaptation." *CVPR*, 2020.
6. Luo, Y.-W., Ren, C.-X., Dao-Qing, D., Yan, H. "Unsupervised domain adaptation via discriminative manifold propagation." *IEEE TPAMI*, 2020.
7. Zhao, H., Tachet Des Combes, R., Zhang, K., Gordon, G. "On learning invariant representations for domain adaptation." *ICML*, 2019.
8. Ben-David, S., Blitzer, J., Crammer, K., Pereira, F. "Analysis of representations for domain adaptation." *NeurIPS*, 2007.
9. Klebanov, I., Schuster, I., Sullivan, T.J. "A rigorous theory of conditional mean embeddings." *SIAM Journal on Mathematics of Data Science*, 2020.
10. Gretton, A., et al. "A kernel two-sample test." *JMLR*, 13(3):723–773, 2012.
