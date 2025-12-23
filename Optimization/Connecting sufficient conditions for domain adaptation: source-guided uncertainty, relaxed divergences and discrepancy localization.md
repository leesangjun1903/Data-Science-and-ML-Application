# Connecting sufficient conditions for domain adaptation: source-guided uncertainty, relaxed divergences and discrepancy localization
***

## 1. 핵심 주장과 주요 기여 (개요)

이 논문은 **도메인 적응(Domain Adaptation, DA)** 분야의 세 가지 중요한 이론적 발전을 통합합니다:

**핵심 주장**: 최근의 DA 경험적 성공이 (1) 타겟 도메인에서의 소스 가이드 불확실성 최소화, (2) 완화된 분포 정렬, 그리고 (3) 불일치의 국소화로부터 비롯된다는 것을 이론적으로 정당화합니다.

**세 가지 주요 기여:**

1. **Source-Guided Uncertainty의 역할 증명**: 타겟 도메인에서의 분류기 신뢰도를 소스 도메인 감독과 결합하여 최소화하는 것이 기존의 광범위한 DA 바운드 클래스를 개선함을 증명

2. **Integral Measure Discrepancy (IMD) 도입**: β-relaxed divergences와 localization이라는 두 가지 완화 개념을 연결하는 새로운 측도 간 불일치 정의

3. **Per-Class Localization**: 소스 도메인의 범주적 구조를 완화에 통합하여 특히 레이블 시프트(label shift) 경우에 글로벌 localization보다 우수한 이론적 보장 제공

***

## 2. 해결하고자 하는 문제

### 2.1 배경 및 동기

도메인 적응은 레이블된 소스 데이터 $S$에서 학습한 분류기를 레이블되지 않은 타겟 데이터 $T$에 적용할 때의 분포 불일치 문제를 해결합니다.

**기존 접근의 한계:**

$$L_T(h) \leq L_S(h) + A(T, S)$$

기존 바운드는 세 가지 문제에 직면했습니다:

1. **Source domain에서의 낮은 위험도 요구가 역효과**: 분포 정렬과 소스 성능 최소화를 동시에 요구할 때, 레이블 마진이 달라지면 적응 성능 저하

2. **분포 정렬의 엄격성**: 정확한 분포 정렬 ($S_X = T_X$) 요구는 실제 환경에서 불가능하고, 이는 "교차 라벨링 위험(cross-labeling risk)"를 야기

3. **타겟 도메인 신뢰도의 역할 미흡**: 조건부 엔트로피 최소화 같은 비지도 목표 손실이 경험적으로 효과적이나 이론적 정당성 부족

### 2.2 핵심 문제 정의

$$\text{minimize} \quad L_T(h) = \mathbb{E}_{(x,y) \sim T}[l(h(x), y)]$$

여기서:
- $h: X \rightarrow Y$ 는 분류기
- $l(\cdot, \cdot)$은 손실함수
- 타겟 도메인 $T$는 레이블 정보 불가

***

## 3. 제안하는 방법: 이론적 프레임워크

### 3.1 Source-Guided Uncertainty 정의

**정의 1**: 분류기 $g$의 불확실성

$$\text{Uncertainty}_D(g) = \inf_{h \in H} L_D(g, h)$$

**정의 4**: Source-Guided Uncertainty

$$C_H(g) = \inf_{h \in H} \left[ L_T(g, h) + L_S(h) \right]$$

**해석**: 
- 첫 번째 항 $L_T(g, h)$: 타겟 도메인에서 $g$의 예측과 가설 $h$ 간 불일치
- 두 번째 항 $L_S(h)$: 소스 도메인에서 $h$의 성능 (정규화)

**명제 5 (속성)**:
$$C_H(g) \leq L_T(g, h_g) + L_S(h_g)$$

특히 $g \in H$일 때:
$$C_H(g) \leq L_S(g)$$

**의미**: Source-guided uncertainty는 타겟에서 $g$의 신뢰도가 높으면서 동시에 소스에서 잘 수행할 때만 작음

### 3.2 Target Risk Bound 개선

**명제 6** (핵심 결과):

주어진 형태의 DA 바운드:
$$L_T(h) \leq L_S(h) + A(T, S) \quad \forall h \in H$$

다음이 성립:
$$L_T(g) \leq C_H(g) + A(T, S)$$

**증명 스케치**:
$$L_T(g) = L_T(g) - \sup_{h \in H}[L_T(h) - L_S(h)] + \sup_{h \in H}[L_T(h) - L_S(h)]$$

$$\leq \inf_{h \in H}[L_T(g) - L_T(h) + L_S(h)] + A(T, S)$$

손실함수 조건 $l_1(u, y_1) - l_2(y_2, y_1) \leq l_1(u, y_2)$에 의해:
$$L_T(g) - L_T(h) \leq L_T(g, h)$$

따라서:
$$\inf_h[L_T(g, h) + L_S(h)] = C_H(g)$$

### 3.3 Integral Measure Discrepancy (IMD)

**정의 8**: 질량이 다른 두 측도 $Q_1, Q_2$ 간의 IMD

$$\text{IMD}_F(Q_1, Q_2) := \sup_{f \in F} \left| \int f \, dQ_1 - \int f \, dQ_2 \right|$$

**특성**:
- IPM(Integral Probability Metrics)의 일반화 (질량이 다를 수 있음)
- 비대칭: $\text{IMD}_F(Q, 2Q) = 0$이지만 $\text{IMD}_F(2Q, Q) > 0$

**명제 9** (IMD의 속성):
$$\text{IMD}_F(Q_1, Q_2) = 0 \Leftrightarrow Q_1 \leq Q_2 \text{ (for rich } F \text{)}$$

### 3.4 Global Localization과 β-Admissible Distance의 연결

**명제 11** (Duality for Localized IMD):

$$\forall \epsilon \geq 0, \quad \text{IMD}_{F_\epsilon}(T_X, S_X) \leq \inf_{\alpha \geq 0} \left[ \text{IMD}_F(T_X, (1+\alpha)S_X) + \epsilon \alpha \right]$$

**코롤러리 13**:

$$L_T(g) \leq C_{H_{r_1}}(g) + \text{IMD}_F(T_X, (1+\beta)S_X) + \beta(r_1 + r_2) + \inf_{h \in H_{r_2}} L_T(h) + L_S(h)$$

여기서 $r_1, r_2, \beta \geq 0$는 정규화 파라미터

### 3.5 Per-Class Localization: 혁신적 확장

**정의 14**: Per-Class Localized 가설 공간

```math
H_\epsilon := \left\{h \in H : \mathbb{E}_{x \sim S_{X|k}}[l(h(x), f_S(x))] \leq \epsilon_k, \forall 1 \leq k \leq K\right\}
```

각 클래스별로 소스 위험도를 제한

**명제 16** (Per-Class 바운드):

$$L_T(g) \leq C_{H_{r_1}}(g) + \text{IMD}_F\left(T_X, S_X + \sum_{k=1}^K \beta_k S_{X|k}\right) + \boldsymbol{\beta}^T(r_1 + r_2) + \inf_{h \in H_{r_2}} L_T(h) + L_S(h)$$

**코롤러리 17** (최적 클래스 가중치):

$$L_T(g) \leq C_{H_r}(g) + \min_{\boldsymbol{\beta} \geq 0: \mathbf{1}^T\boldsymbol{\beta} \leq \beta} \text{IMD}_F\left(T_X, S_X + \sum_{k=1}^K \beta_k S_{X|k}\right) + \beta(2r) + \inf_{h \in H_r} L_T(h) + L_S(h)$$

**해석**: 이는 다음의 클래스 재가중치 최소화와 동치:

$$\min_{\tilde{p} \in \Delta^K : (1+\beta)\tilde{p} \geq p} \text{IMD}_F\left(T_X, (1+\beta)\sum_{k=1}^K \tilde{p}_k S_{X|k}\right)$$

### 3.6 레이블 시프트의 특수화

**명제 18** (Label Shift Case):

$T_{X|y} = S_{X|y}$ (조건부 분포 동일)이고 $q_k, p_k$가 각각 타겟과 소스의 클래스 비율일 때:

**글로벌 완화**:
$$\beta \geq \max_{1 \leq k \leq K} \left[\frac{q_k}{p_k} - 1\right]_+$$

**Per-Class 완화**:
$$\beta_k \geq (q_k - p_k)_+ \quad \forall k$$

**비교**: Per-class 조건은 더 느슨함
$$\sum_k (q_k - p_k)_+ \leq 1 < \max_k \frac{q_k}{p_k}$$

***

## 4. 모델 구조와 성능 향상

### 4.1 통합 프레임워크의 세 구성요소

#### 1단계: Source Domain Risk 최소화

$$\min_h L_S(h)$$

#### 2단계: 도메인 불일치 정량화

- **Global**: $\text{IMD}_F(T_X, (1+\beta)S_X)$
- **Per-Class**: $\text{IMD}\_F(T_X, S_X + \sum_k \beta_k S_{X|k})$

#### 3단계: Confidence Regularization

조건부 엔트로피 최소화:

$$\min_g \mathbb{E}_{x \sim T_X} [H_\infty(g(x))]$$

여기서:

$$H_\infty(g(x)) = -\max_i \log(g(x)_i)$$

### 4.2 Wasserstein Distance 특수화

**명제 22**: $H$의 함수가 1/2-Lipschitz일 때, per-class localization의 IMD는 최적 운송 문제로 표현:

$$\inf_{\{P_k\}_k \subset \mathcal{M}_+(X \times X)} \sum_{k=1}^K \mathbb{E}_{(x_t, x_s) \sim P_k}[d(x_t, x_s)]$$

```math
\text{s.t.} \quad \pi_1\# \sum_k P_k \geq T_X, \quad \pi_2\# P_k \leq (p_k + \beta_k)S_{X|k}
```

**의미**: Per-class localization이 부분 최적 운송 문제의 자연스러운 확장

### 4.3 일반화 성능 향상 메커니즘

**메커니즘 1: Tighter Bounds**

Source-guided uncertainty 도입으로:
- 기존: 소스 위험도 항 $L_S(h)$ 포함
- 개선: 최소 $L_S(h)$만 필요, 조건부 불확실성 추가

**메커니즘 2: Label Shift 강건성**

레이블 시프트 환경에서 per-class의 장점:

| 측정 | 글로벌 | Per-Class |
|------|--------|-----------|
| 필요한 β 범위 | $[0, \max_k \frac{q_k}{p_k}]$ | $[1]$ |
| 교차 라벨링 위험 | 높음 | 낮음 |
| 계산 복잡도 | O(1) | O(K) |

**메커니즘 3: 클래스 불균형 처리**

불균형 소스 ($p_k$ 다양) + 시프트된 타겟에서 per-class 방법이 우수:

$$\text{Advantage} = \text{Per-class accuracy} - \text{Global accuracy}$$

불균형 강도 $\eta$ 증가 시 이점 증가

***

## 5. 실증 검증 및 한계

### 5.1 토이 데이터셋 실험

**설정**:
- K개의 2D Gaussian 혼합: $S_X = \sum_k p_k \mathcal{N}(\mu_k, \sigma I_2)$
- 클래스 비율: $p_k \propto e^{\eta k}$ (불균형 강도 $\eta$)
- 타겟: 소스를 각도 $\theta$ 만큼 회전

**결과** ($\theta = 0°$, 레이블 시프트):
- Per-class localization이 글로벌 방법보다 일관되게 우수
- 클래스 수 증가 및 불균형 강도 증가 시 차이 확대

### 5.2 현재 한계

**실증적 한계**:
✗ 복잡한 벤치마크(Office-31, VisDA, ImageNet-C) 부재
✗ 심층 신경망에서의 실제 구현 미제시
✗ 계산 비용 분석 부재

**이론적 한계**:
✗ 함수 공간 $F$의 "richness" 조건이 불명확
✗ Lipschitz 함수의 경우 역방향 결과가 불완전
✗ 표본 복잡도(sample complexity) 분석 부재

**실무적 한계**:
✗ 하이퍼파라미터 $\beta$ 선택 방법 미제시
✗ Per-class 재가중치 최적화 알고리즘 미상세화
✗ 대규모 데이터에의 확장성 미검증

***

## 6. 2020년 이후 관련 최신 연구 비교

### 6.1 주요 방향별 선행 연구

#### A. Label Shift 대응 방법

| 연구 | 연도 | 접근 | 핵심 기여 |
|------|------|------|---------|
| Tachet des Combes et al. | 2020 | 조건부 분포 정렬 | Balanced Error Rate 도입 |
| Le et al. (LAMDA) | 2021 | 라벨 정렬 | 직접적 라벨 정렬 메커니즘 |
| Garg et al. (RLSbench) | 2023 | 벤치마크 | 500+ 분포 시프트 쌍 제공 |
| **본 논문** | **2022** | **이론적 통합** | **IMD + Per-class + α 분석** |

#### B. Wasserstein 기반 방법

| 연구 | 시간 | 초점 | 성과 |
|------|------|------|------|
| Redko et al. (OT-DA) | 2017 | 기초 이론 | Wasserstein 바운드 증명 |
| Lee et al. (SWD) | 2019 | 계산 효율성 | Sliced Wasserstein 도입 |
| Le et al. (LDROT) | 2021 | Label+Covariate Shift | 운송 최적화 적용 |
| **본 논문** | **2022** | **Per-class 운송** | **클래스별 Wasserstein 공식화** |

#### C. Uncertainty 활용

| 연구 | 연도 | 방식 |
|------|------|------|
| Kpotufe & Martinet | 2021 | Gaussian 불확실성 |
| 본 논문 | 2022 | Source-guided 불확실성 |
| Saito et al. (Entropy Min) | 2019 | 조건부 엔트로피 |

### 6.2 본 논문의 차별성

**1. 이론적 통합**
- 기존: Relaxation과 class-aware 방법이 분절
- **본 논문**: 통합 프레임워크에서 두 가지 모두 다룸

**2. IMD의 혁신**
$$\text{기존 IPM}: \int f(Q_1 - Q_2) \quad \Rightarrow \quad \text{IMD}: \int f Q_1 - \int f Q_2$$
- 질량 보존 가정 제거 가능
- Unbalanced optimal transport와의 자연스러운 연결

**3. Per-class Localization의 이론적 우월성**
- **명제 18**: 레이블 시프트에서 β 범위 축소 ($K$배 감소 가능)
- 실무적으로 하이퍼파라미터 탐색 효율성 증대

**4. 조건부 엔트로피 최소화의 정당화**
- 기존: 경험적 성공만 알려짐
- **명제 6**: Source-guided 불확실성이 일반적 DA 바운드를 개선함을 증명

***

## 7. 앞으로의 연구 영향 및 고려사항

### 7.1 이론적 영향

**기여 영역**:
1. **도메인 적응 이론의 정제화**: 분절된 접근들의 통일된 이해 제공
2. **최적 운송 이론과의 깊은 연결**: Per-class 운송 문제의 형식화
3. **다른 분포 시프트 문제로의 확장 가능성**: 조건부 시프트, 개방집합 DA 등

### 7.2 향후 연구 방향

**단기 (1-2년)**:
1. 실제 벤치마크(Office-Home, VisDA, DomainNet)에서의 성능 검증
2. 심층 학습 기반 최적화 알고리즘 개발
3. 하이퍼파라미터 선택 방법론 연구

**중기 (3-5년)**:
1. 다중 소스 도메인 적응(multi-source DA)으로의 확장
2. 개방집합 도메인 적응(open-set DA)에서의 적용
3. 연속 도메인 적응(continual DA)에서의 유효성 검증

**장기 (5년 이상)**:
1. $f$-발산 변분 표현과 적대적 학습의 공식적 연결
2. 생성 모델(GAN, Diffusion)과의 통합
3. 자가 지도 학습(Self-Supervised Learning)과의 시너지 탐색

### 7.3 구현 시 고려사항

**이론을 실무에 적용할 때의 주의점**:

1. **함수 공간 선택의 중요성**
   - IPM의 경우: Universal kernel 필수
   - Wasserstein의 경우: Lipschitz 상수 추정 필요

2. **β 파라미터 선택**
   - Per-class: 각 클래스별 레이블 비율 추정 필요
   - 교차 검증 또는 이론적 가이드 부재

3. **계산 복잡도**
   - Per-class Wasserstein: K개의 OT 문제 동시 해결
   - 대규모 데이터에서는 Sinkhorn 가속화 필수

4. **극단적 케이스 처리**
   - 타겟 클래스 비율 0에 가까운 경우
   - 소스와 타겟의 클래스 차이가 큰 경우 (partial DA)

***

## 8. 결론

### 종합 평가

| 측면 | 평가 | 근거 |
|------|------|------|
| **이론적 기여도** | ⭐⭐⭐⭐⭐ | 세 가지 핵심 개념의 통합, 새로운 IMD 정의 |
| **명확성** | ⭐⭐⭐ | 복잡한 수학 표현, 직관적 설명 부족 |
| **실증적 검증** | ⭐⭐ | 토이 데이터셋만 사용, 벤치마크 부재 |
| **실무 적용 용이성** | ⭐⭐ | 알고리즘 상세화 부족, 파라미터 선택 미흡 |
| **향후 영향력** | ⭐⭐⭐⭐ | 새로운 연구 방향 제시, 이론적 기초 강고 |

**종합 점수: 3.8/5.0**

### 핵심 인사이트 3가지

1. **Source-Guided Uncertainty의 강력함**
   - 조건부 엔트로피 최소화라는 기존의 휴리스틱을 이론적으로 정당화
   - 광범위한 DA 바운드의 동시적 개선 가능

2. **Per-Class Localization의 우월성**
   - 특히 레이블 시프트 환경에서 글로벌 방법보다 이론적으로 우수
   - 계산 복잡도 증가는 미미하나 성능 향상은 현저

3. **최적 운송과의 자연스러운 연결**
   - IMD를 통해 IPM과 unbalanced OT를 통합
   - 향후 생성 모델 및 자가 지도 학습과의 시너지 가능

***

이 논문은 도메인 적응 이론의 고전적 위상을 정립하는 의미 있는 기여를 하며, 향후 5-10년간의 DA 연구에 영향을 미칠 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b9212ec4-a5bb-49d9-b551-68b6b2da07ca/2203.05076v1.pdf)
[2](https://link.springer.com/10.1007/978-3-031-72341-4)
[3](https://papers.phmsociety.org/index.php/phmconf/article/view/4193)
[4](https://www.nature.com/articles/s43017-020-0092-4)
[5](https://biss.pensoft.net/article/133089/)
[6](https://www.worldwidejournals.com/paripex/recent_issues_pdf/2024/August/knowledge-management-risks-a-cluster-analysis-during-covid19-and-after_August_2024_0427193758_7404813.pdf)
[7](http://medrxiv.org/lookup/doi/10.1101/2024.08.21.24312122)
[8](https://journals.sagepub.com/doi/10.1177/15266028241289083)
[9](https://www.mdpi.com/2073-4441/16/13/1899)
[10](https://doi.apa.org/doi/10.1037/rev0000509)
[11](https://academic.oup.com/etc/article/43/6/1390/7829460)
[12](https://arxiv.org/html/2502.06272v1)
[13](https://arxiv.org/pdf/2302.13824.pdf)
[14](http://arxiv.org/abs/2111.02901)
[15](http://arxiv.org/pdf/2407.09367.pdf)
[16](http://arxiv.org/pdf/2306.04344.pdf)
[17](https://arxiv.org/pdf/2410.01709.pdf)
[18](http://arxiv.org/pdf/2206.01319.pdf)
[19](https://arxiv.org/pdf/2203.08321.pdf)
[20](https://papers.miccai.org/miccai-2024/paper/1781_paper.pdf)
[21](https://proceedings.neurips.cc/paper/2020/file/dfbfa7ddcfffeb581f50edcf9a0204bb-Paper.pdf)
[22](https://www.jmlr.org/papers/volume26/23-0573/23-0573.pdf)
[23](https://arxiv.org/html/2508.18630v1)
[24](https://proceedings.mlr.press/v202/garg23a/garg23a.pdf)
[25](https://proceedings.mlr.press/v202/go23a/go23a.pdf)
[26](https://www.sciencedirect.com/science/article/abs/pii/S0957417423035960)
[27](https://arxiv.org/abs/2302.03133)
[28](https://pmc.ncbi.nlm.nih.gov/articles/PMC2877995/)
[29](https://proceedings.neurips.cc/paper_files/paper/2024/file/9e5f7743a4e753452f73d32da1190202-Paper-Conference.pdf)
[30](https://arxiv.org/html/2512.18661v1)
[31](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Adversarial_Unsupervised_Domain_Adaptation_With_Conditional_and_Label_Shift_Infer_ICCV_2021_paper.pdf)
[32](https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_Adversarial_Distillation_Based_on_Slack_Matching_and_Attribution_Region_Alignment_CVPR_2024_paper.pdf)
[33](https://arxiv.org/html/2507.22659v2)
[34](https://arxiv.org/abs/2508.17780)
[35](https://arxiv.org/abs/2303.02569)
[36](https://arxiv.org/pdf/2508.16527.pdf)
[37](https://arxiv.org/abs/2503.02506)
[38](https://arxiv.org/html/2502.15681v1)
[39](https://arxiv.org/html/2504.18765v1)
[40](https://www.amazon.science/publications/rlsbench-domain-adaptation-under-relaxed-label-shift)
[41](https://openreview.net/forum?id=7gpj0XollN)
[42](https://link.springer.com/10.1007/s10489-021-03112-9)
[43](https://link.springer.com/10.1007/s10489-022-03810-y)
[44](https://arxiv.org/abs/2503.08155)
[45](https://arxiv.org/abs/2506.02712)
[46](https://arxiv.org/abs/2210.10195)
[47](https://arxiv.org/abs/2503.11249)
[48](https://www.semanticscholar.org/paper/7fe935a147cce08933eb5a9d1dc123fcc3b2b8bd)
[49](https://arxiv.org/abs/2504.08544)
[50](https://www.semanticscholar.org/paper/7962415262b9ce998a4c1f9fd31592d048c5e1c7)
[51](https://ieeexplore.ieee.org/document/9326987/)
[52](http://arxiv.org/pdf/2209.03243.pdf)
[53](http://arxiv.org/pdf/2407.21492.pdf)
[54](http://arxiv.org/pdf/2301.06297.pdf)
[55](http://arxiv.org/pdf/2404.06625.pdf)
[56](https://arxiv.org/pdf/2303.14085.pdf)
[57](https://arxiv.org/html/2501.08066v1)
[58](https://arxiv.org/html/2312.10295v3)
[59](https://arxiv.org/pdf/2309.05522.pdf)
[60](https://www.ijcai.org/proceedings/2020/0299.pdf)
[61](https://openreview.net/pdf?id=kJcwlP7BRs)
[62](https://aclanthology.org/2020.findings-emnlp.315.pdf)
[63](http://ecmlpkdd2017.ijs.si/papers/paperID194.pdf)
[64](https://www.sciencedirect.com/science/article/abs/pii/S0019057822001276)
[65](https://ise.thss.tsinghua.edu.cn/~mlong/doc/domain-adaptation-theory-icml19.pdf)
[66](https://www.arxiv.org/abs/2110.15520)
[67](https://arxiv.org/abs/2201.10460)
[68](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Differential_Treatment_for_Stuff_and_Things_A_Simple_Unsupervised_Domain_CVPR_2020_paper.pdf)
[69](http://arxiv.org/pdf/2210.13331.pdf)
[70](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lee_Sliced_Wasserstein_Discrepancy_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)
[71](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tran_Transferability_and_Hardness_of_Supervised_Classification_Tasks_ICCV_2019_paper.pdf)
[72](https://arxiv.org/pdf/2103.03757.pdf)
[73](https://openaccess.thecvf.com/content/CVPR2021/papers/Montesuma_Wasserstein_Barycenter_for_Multi-Source_Domain_Adaptation_CVPR_2021_paper.pdf)
[74](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Chakraborty_Efficient_Conditional_Pre-Training_for_Transfer_Learning_CVPRW_2022_paper.pdf)
[75](https://www.semanticscholar.org/paper/On-Localized-Discrepancy-for-Domain-Adaptation-Zhang-Long/d28eaba80a4633f93b740533d6e31c18dbf36436)
[76](https://arxiv.org/abs/2110.15520)
[77](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_TransMatch_A_Transfer-Learning_Scheme_for_Semi-Supervised_Few-Shot_Learning_CVPR_2020_paper.pdf)
[78](https://arxiv.org/pdf/2008.06242.pdf)
[79](https://lab.bciml.cn/wp-content/uploads/2020/08/Wasserstein-distance-based-deep-adversarial-transfer-learning-for-intelligent-fault-diagnosis-with-unlabeled-or-insufficient-labeled-data.pdf)
[80](https://www.sciencedirect.com/science/article/abs/pii/S0952197623003561)
