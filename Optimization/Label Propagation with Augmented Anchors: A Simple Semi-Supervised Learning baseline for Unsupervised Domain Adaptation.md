# Label Propagation with Augmented Anchors: A Simple Semi-Supervised Learning baseline for Unsupervised Domain Adaptation (A²LP)

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음과 같습니다:

> **UDA(Unsupervised Domain Adaptation)와 SSL(Semi-Supervised Learning)은 문제 구조상 밀접하게 연관되어 있으나, 기존 UDA 방법들은 SSL 기법을 직접 적용할 때 발생하는 도메인 시프트(domain shift) 문제를 간과하고 있다.**

특히 그래프 기반 SSL 방법인 Label Propagation(LP)을 UDA에 직접 적용하면, 소스-타겟 도메인 간 분포 차이로 인해 같은 클래스 인스턴스 간의 친화도(affinity)가 낮아져 성능이 저하된다는 문제를 이론적으로 분석하고, 이를 해결하는 **A²LP(Label Propagation with Augmented Anchors)** 알고리즘을 제안합니다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| **이론적 분석** | UDA에서 LP 적용 시 친화도 행렬 구성 조건을 Proposition 1로 형식화 |
| **알고리즘 제안** | 가상 인스턴스(augmented anchors)를 생성하여 LP를 개선하는 A²LP |
| **실증적 검증** | Office-31, ImageCLEF-DA, VisDA-2017 벤치마크에서 SOTA 달성 |
| **모듈성** | MSTN, CAN 등 기존 SOTA 방법에 플러그인 방식으로 적용 가능 |
| **코드 공개** | https://github.com/YBZh/Label-Propagation-with-Augmented-Anchors |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

#### 문제 배경

UDA에서 LP를 직접 사용할 경우 발생하는 근본적인 문제는 **도메인 시프트로 인한 친화도 행렬의 품질 저하**입니다.

LP의 정규화 항은 다음과 같이 분해됩니다:

$$\mathcal{R}(X;\mathbf{F}) = \sum_{y_i=y_j} a_{ij}\left\|\frac{\mathbf{f}_i}{\sqrt{d_{ii}}} - \frac{\mathbf{f}_j}{\sqrt{d_{jj}}}\right\|^2 + \sum_{y_i \neq y_j} a_{ij}\left\|\frac{\mathbf{f}_i}{\sqrt{d_{ii}}} - \frac{\mathbf{f}_j}{\sqrt{d_{jj}}}\right\|^2 \tag{4}$$

이상적인 친화도 행렬은:
- 같은 클래스 쌍: $a_{ij}$를 **최대한 크게**
- 다른 클래스 쌍: $a_{ij}$를 **최대한 작게**

그러나 **UDA 환경에서는** 소스(labeled)와 타겟(unlabeled) 간 분포 차이로 인해, 같은 클래스임에도 불구하고 $a_{ij}$ 값이 현저히 감소합니다.

> 논문 Fig. 1에서 시각적으로 확인: SSL(93.5%) vs UDA(64.8%) vs UDA with Anchors(79.5%)

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: SSL 메타 목적함수

SSL의 일반적인 목적함수:

$$\mathcal{Q}(\mathbf{P}) = \mathcal{L}(X_L, Y_L; \mathbf{P}) + \lambda \mathcal{R}(X; \mathbf{P}) \tag{1}$$

엔트로피 최소화(EM)를 UDA에 적용한 형태:

$$\mathcal{Q}(\mathbf{P}) = \underbrace{\sum_{i=1}^{l} \ell(\mathbf{p}_i, y_i)}_{\mathcal{L}(X_L,Y_L;\mathbf{P})} + \lambda \underbrace{\sum_{i=l+1}^{n} \sum_{j=1}^{K} -p_{ij}\log p_{ij}}_{\mathcal{R}(X;\mathbf{P})} \tag{2}$$

#### Step 2: LP의 최적화 목적함수

$$\mathcal{Q}(\mathbf{F}) = \underbrace{\sum_{i=1}^{n} \|\mathbf{f}_i - \mathbf{y}_i\|^2}_{\mathcal{L}(X_L,Y_L;\mathbf{F})} + \lambda \underbrace{\sum_{i,j} a_{ij}\left\|\frac{\mathbf{f}_i}{\sqrt{d_{ii}}} - \frac{\mathbf{f}_j}{\sqrt{d_{jj}}}\right\|^2}_{\mathcal{R}(X;\mathbf{F})} \tag{3}$$

#### Step 3: 친화도 행렬 구성

$k$-최근접 이웃 그래프 기반 친화도 행렬:

$$a_{ij} := \begin{cases} \varepsilon(\mathbf{v}_i, \mathbf{v}_j), & \text{if } i \neq j \wedge \mathbf{v}_i \in \text{NN}_k(\mathbf{v}_j) \\ 0, & \text{otherwise} \end{cases} \tag{6}$$

코사인 유사도를 사용: $\varepsilon(\mathbf{v}_i, \mathbf{v}_j) = \frac{\langle \mathbf{v}_i, \mathbf{v}_j \rangle}{\|\mathbf{v}_i\|\|\mathbf{v}_j\|}$

#### Step 4: LP의 닫힌 형태 해

$$\mathbf{F}^* = (\mathbf{I} - \alpha \mathbf{S})^{-1} \mathbf{Y} \tag{7}$$

여기서 $\alpha = \frac{2\lambda}{2\lambda+1}$, $\mathbf{S} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$

#### Step 5: 엔트로피 기반 인스턴스 가중치

각 비레이블 인스턴스 $x_i$에 대한 가중치:

$$w_i := 1 - \frac{H(\mathbf{p}_i^*)}{\log(K)} \tag{8}$$

여기서 $H(\cdot)$는 엔트로피 함수이며, $p_{ij}^* = f_{ij}^* / \sum_j f_{ij}^*$. $w_i \in [0,1]$임이 보장됩니다.

- **낮은 엔트로피** → 높은 신뢰도 → $w_i$가 1에 가까움
- **높은 엔트로피** → 낮은 신뢰도 → $w_i$가 0에 가까움

#### Step 6: 가상 인스턴스(Augmented Anchors) 생성

클래스 $k$에 대한 가상 인스턴스:

$$(\hat{\mathbf{v}}_{n+k}, k) = \left(\sum_{\mathbf{x}_i \in X_U} \frac{\mathbf{1}(k=\hat{y}_i) w_i \phi_{\theta_e}(\mathbf{x}_i)}{\sum_{\mathbf{x}_j \in X_U} \mathbf{1}(k=\hat{y}_j) w_j},\ k\right) \tag{9}$$

즉, **같은 클래스로 예측된 타겟 인스턴스들의 신뢰도 가중 평균**을 가상 앵커로 사용합니다.

#### Step 7: 특징 집합 및 레이블 행렬 업데이트

$$V = V \cup \{\hat{\mathbf{v}}_{n+1}, \cdots, \hat{\mathbf{v}}_{n+K}\}, \quad \mathbf{Y} = \begin{bmatrix} \mathbf{Y} \\ \mathbf{I} \end{bmatrix},\quad n = n + K \tag{10}$$

#### Step 8: 계산 가속화 (선형 시스템 풀기)

CG(Conjugate Gradient) 방법 적용:

$$(\mathbf{I} - \alpha \mathbf{S})\mathbf{F}^* = \mathbf{Y} \tag{11}$$

$k$-NN 그래프 구성: $O(n^2) \rightarrow O(n^{1.1})$ (NN-Descent 사용)

---

### 2.3 이론적 토대: Proposition 1

> **Proposition 1.** 데이터가 이상적인 클러스터 가정($a_{ij} = 0$, $\forall y_i \neq y_j$)을 만족한다고 가정할 때, 어떤 데이터 인스턴스 $x_m$과 레이블된 인스턴스 $x_n \in X_L$ 사이의 0값 원소 $a_{mn}$을 양수로 증가시키면($y_m = y_n$인 경우), 분류 정확도 $Acc$ (식 5)는 감소하지 않으며, 원래 $\hat{y}_m \neq y_m$이었을 때 증가한다.

$$Acc := \frac{|\{\mathbf{x}_i \in X_U : \hat{y}_i = y_i\}|}{|X_U|} \tag{5}$$

**Remark 1:** 가상 인스턴스(augmented anchor)가 동일 클래스 인스턴스들을 이웃으로 가지면, LP의 $Acc$는 단조 증가(non-decreasing)합니다.

---

### 2.4 모델 구조

전체 프레임워크는 **두 단계의 교대 학습(Alternating Learning)**으로 구성됩니다:

```
┌─────────────────────────────────────────────────────┐
│              전체 교대 학습 루프                        │
│                                                     │
│  ┌─────────────────────────┐                        │
│  │    A²LP (슈도 레이블링)   │                        │
│  │  1. 친화도 행렬 A 구성    │ ──→ 고품질 슈도 레이블 제공 │
│  │  2. F* = (I-αS)⁻¹Y 풀기 │                        │
│  │  3. 엔트로피 기반 가중치   │                        │
│  │  4. 가상 인스턴스 생성    │                        │
│  │  5. V, Y 업데이트 (N회)  │                        │
│  └─────────────────────────┘                        │
│              ↕                                      │
│  ┌─────────────────────────┐                        │
│  │  도메인 불변 특징 학습    │                        │
│  │  (e.g., MSTN, CAN)      │ ──→ 더 나은 특징 표현 제공 │
│  └─────────────────────────┘                        │
└─────────────────────────────────────────────────────┘
```

**백본**: ImageNet 사전학습 ResNet (마지막 FC 레이어 제외)
- 특징 추출기 $\phi: \mathcal{X} \rightarrow \mathbb{R}^d$ (파라미터: $\theta_e$)
- 분류기 $f: \mathbb{R}^d \rightarrow \mathbb{R}^K$ (파라미터: $\theta_c$, FC 레이어 1개)

**학습률 어닐링**:

$$\eta_p = \frac{\eta_0}{(1+\mu p)^\beta}$$

여기서 $\eta_0=0.01$, $\mu=10$, $p$: 학습 진행도 (0→1)

---

### 2.5 성능 향상

#### Office-31 (ResNet-50)

| 방법 | A→W | D→W | W→D | A→D | D→A | W→A | **Avg** |
|------|-----|-----|-----|-----|-----|-----|---------|
| LP | 81.1 | 96.8 | 99.0 | 82.3 | 71.6 | 73.1 | 84.0 |
| **A²LP (ours)** | **87.7** | **98.1** | **99.0** | **87.8** | **75.8** | **75.9** | **87.4** |
| CAN (재현) | 94.0 | 98.5 | 99.7 | 94.8 | 78.1 | 76.7 | 90.3 |
| **CAN + A²LP** | **93.4** | **98.8** | **100.0** | **96.1** | **78.1** | **77.6** | **90.7** |

#### VisDA-2017

| 방법 | ResNet-50 | ResNet-101 |
|------|-----------|------------|
| LP | 69.8 | 73.9 |
| **A²LP** | **78.7** | **82.7** |
| MSTN + A²LP | 81.5 | 83.7 |
| CAN + A²LP | **86.5** | **87.6** |

**핵심 개선 포인트:**
- LP 대비 A²LP: 평균 **+3.4%p** 향상 (Office-31 기준)
- VisDA-2017에서 LP 대비 약 **+9%p** 향상

---

### 2.6 한계

1. **이상적 클러스터 가정 의존성**: Proposition 1은 $a_{ij}=0$ ($y_i \neq y_j$)인 이상적 조건을 전제하므로, 실제 데이터에서는 완전한 성능 보장이 어렵습니다.

2. **노이즈 민감성**: 초기 슈도 레이블의 노이즈가 60% 이상이면 vanilla LP보다 성능이 저하됩니다 (Table 1 참조).

3. **계산 비용**: 전체 친화도 행렬 구성은 $O(n^2)$이며, 대규모 데이터셋(VisDA-2017)에서는 A²LP 변형(소스 데이터를 클래스 센터로 대체)을 사용해야 합니다.

4. **도메인 시프트 해결의 부분성**: 가상 인스턴스만으로는 대규모 도메인 시프트를 완전히 극복하기 어려우며, 도메인 불변 특징 학습과의 결합이 필수적입니다.

5. **전이 가능성의 제약**: 클래스 분포가 극도로 불균형하거나, 클러스터 가정이 크게 위반되는 경우 성능 저하가 예상됩니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상의 핵심 메커니즘

#### (A) 엔트로피 기반 신뢰도 가중치의 역할

$$w_i = 1 - \frac{H(\mathbf{p}_i^*)}{\log(K)}$$

- 불확실한 예측(높은 엔트로피)을 가진 인스턴스에 낮은 가중치를 부여하여 노이즈 전파를 억제합니다.
- 이는 모델이 **확신도 높은 예측에 기반해 학습**하도록 유도하여 일반화를 돕습니다.

#### (B) 클러스터 중심 기반 가상 인스턴스의 일반화 효과

가상 앵커는 같은 클래스의 타겟 인스턴스들의 **가중 평균**으로 생성됩니다:

$$\hat{\mathbf{v}}_{n+k} = \frac{\sum_{\mathbf{x}_i \in X_U} \mathbf{1}(k=\hat{y}_i) w_i \phi_{\theta_e}(\mathbf{x}_i)}{\sum_{\mathbf{x}_j \in X_U} \mathbf{1}(k=\hat{y}_j) w_j}$$

이 가상 앵커는 **클래스 프로토타입(class prototype)** 역할을 하며:
- 개별 인스턴스보다 노이즈에 강건합니다.
- 타겟 도메인의 클래스별 분포를 대표합니다.
- LP 그래프에서 소스-타겟 연결 가중치(PoW)를 증가시켜 레이블 전파의 품질을 향상합니다.

#### (C) 교대 학습의 상호 강화 효과 (Mutual Reinforcement)

```
A²LP → 고품질 슈도 레이블
         ↓
도메인 불변 특징 학습 → 개선된 특징 공간
         ↓
더 좋은 친화도 행렬 → A²LP 성능 향상
         ↓
         (...반복)
```

이 순환 구조는 **자기 강화(self-reinforcing)** 메커니즘으로, 각 반복마다 일반화 성능이 점진적으로 향상됩니다.

#### (D) PoW(Percent of Connection Weight) 지표로 본 일반화

$$PoW = \frac{W_{lu}}{W_{all}}$$

- $W_{lu}$: 레이블-비레이블 데이터 간 같은 클래스 쌍의 유사도 합
- $W_{all} = \sum_{i,j} a_{ij}$

A²LP 반복이 진행될수록 PoW가 증가하며 (Fig. 4), 이는 타겟 도메인에서의 레이블 전파 품질이 향상되고 있음을 의미합니다.

#### (E) 다양한 UDA 방법으로의 일반화 가능성

A²LP는 **플러그인 모듈**로서 MSTN, CAN 등 다양한 도메인 불변 특징 학습 방법과 결합 가능하며, 이는 방법의 일반화 가능성을 실증적으로 보여줍니다.

### 3.2 일반화 성능의 한계 (이론적 관점)

Ben-David et al. (2010)의 UDA 이론적 경계에 따르면:

$$\epsilon_T(h) \leq \epsilon_S(h) + d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda^*$$

A²LP는 슈도 레이블을 통한 타겟 데이터 활용으로 $\epsilon_S(h)$ 항을 간접적으로 줄이지만, 도메인 발산 $d_{\mathcal{H}\Delta\mathcal{H}}$를 직접 최소화하지는 않으며, 이를 위해 도메인 불변 특징 학습이 병행되어야 합니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### (A) SSL과 UDA의 방법론적 통합 촉진

이 논문은 SSL 기법을 UDA에 단순 적용하는 것이 아닌, **UDA의 도메인 시프트 특성을 반영한 SSL 기법의 설계**가 필요함을 명시적으로 보여줬습니다. 이는 향후 연구에서 SSL-UDA 공동 설계(co-design)의 필요성을 강조합니다.

#### (B) 그래프 기반 방법의 UDA 재조명

그래프 구조와 친화도 행렬의 품질이 UDA 성능에 결정적임을 Proposition 1로 이론화함으로써, **그래프 품질 향상 연구**의 방향을 제시했습니다.

#### (C) 교대 학습 패러다임의 보편화

A²LP의 교대 학습 구조는 이후 다수의 UDA 연구에서 유사한 방식으로 채택되었습니다 (슈도 레이블링 + 도메인 정렬의 반복 학습).

#### (D) 플러그인 모듈 연구의 방향 제시

기존 SOTA 방법에 최소한의 변경으로 적용 가능한 플러그인 모듈 설계는 이후 **경량화·모듈화 연구**의 중요한 방향을 제시했습니다.

---

### 4.2 향후 연구 시 고려할 점

#### (A) 이론적 보강
- Proposition 1의 이상적 클러스터 가정을 완화하는 이론적 분석
- 실제 노이즈가 있는 환경에서의 수렴 보장 연구

#### (B) 가상 인스턴스 생성 방법 개선
- 단순 가중 평균 대신 생성 모델(GAN, VAE, Diffusion Model)을 활용한 더 현실적인 가상 인스턴스 생성
- 클래스 내 다양성(intra-class diversity)을 고려한 앵커 생성

#### (C) 동적 그래프 구조
- 고정된 $k$-NN 그래프 대신 학습 중 동적으로 갱신되는 그래프 구조 연구
- Graph Neural Network(GNN)와의 결합

#### (D) 오픈셋·파셜 UDA 적용
- 소스와 타겟의 클래스 공간이 불일치하는 **Open-Set UDA**나 **Partial DA** 환경에서의 적용성 검토

#### (E) 대규모 데이터셋 및 비전-언어 모델과의 통합
- Foundation Model(CLIP, DINO 등)의 특징 표현을 A²LP의 친화도 행렬 구성에 활용
- 대규모 데이터셋에서의 계산 효율성 개선

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 내용은 논문에서 직접 인용된 내용이 아니며, 제가 학습한 지식 기반의 일반적인 분석입니다. 개별 논문의 정확한 수치 등은 원본을 반드시 확인하시기 바랍니다.

### 5.1 주요 관련 연구 흐름

| 연구 방향 | 대표 방법 | A²LP와의 관계 |
|----------|----------|-------------|
| **자기 학습 기반** | NRC (NIPS 2021), SHOT (ICML 2020) | A²LP와 유사한 슈도 레이블링, 그러나 그래프 구조 미사용 |
| **프로토타입 기반** | ProDA (CVPR 2021), ATDOC (CVPR 2021) | A²LP의 클래스 센터 개념과 유사, 더 정교한 프로토타입 사용 |
| **대조 학습 기반** | CDTrans (ICLR 2022), SPA (2022) | 대조 학습으로 도메인 불변 특징 학습, LP 개념 미포함 |
| **Foundation Model 활용** | CLIP 기반 DA 방법 (2022-) | 대규모 사전학습으로 도메인 시프트 자체를 줄이는 방향 |

### 5.2 A²LP 대비 주요 차별점

**SHOT (Liang et al., ICML 2020)**
- 소스 모델을 동결하고 타겟 특징만 학습
- 정보 최대화 + 슈도 레이블 기반
- A²LP와 달리 그래프 구조 명시적 미사용

**ProDA (Zhang et al., CVPR 2021)**
- 프로토타입 기반 도메인 정렬
- A²LP의 가중 평균 앵커 개념과 개념적으로 유사하나, 최적 수송(Optimal Transport)을 활용
- 더 정교한 프로토타입 정렬 메커니즘

**NRC (Yang et al., NeurIPS 2021)**
- 이웃 관계 기반 클러스터링
- 상호 이웃(mutual nearest neighbors)을 활용한 그래프 구성
- A²LP보다 더 정교한 그래프 기반 방법

### 5.3 종합 비교

A²LP의 가장 큰 장점은 **단순성과 모듈성**입니다. 복잡한 새로운 아키텍처 없이 기존 LP를 확장하여 UDA 성능을 크게 향상시켰다는 점에서, 이후 연구들이 더 복잡한 방법을 설계할 때 **기준선(baseline)**으로서의 역할을 충실히 수행합니다.

---

## 참고 자료

1. **Zhang, Y., Deng, B., Jia, K., & Zhang, L. (2020).** *Label Propagation with Augmented Anchors: A Simple Semi-Supervised Learning baseline for Unsupervised Domain Adaptation.* ECCV 2020. (본 논문)

2. **Zhou, D., Bousquet, O., Lal, T.N., Weston, J., & Schölkopf, B. (2004).** *Learning with local and global consistency.* Advances in Neural Information Processing Systems 16.

3. **Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J.W. (2010).** *A theory of learning from different domains.* Machine Learning, 79(1-2), 151–175.

4. **Ganin, Y., et al. (2016).** *Domain-adversarial training of neural networks.* JMLR 17(1), 2096–2030.

5. **Long, M., Cao, Y., Wang, J., & Jordan, M.I. (2015).** *Learning transferable features with deep adaptation networks.* ICML 2015.

6. **Kang, G., Jiang, L., Yang, Y., & Hauptmann, A.G. (2019).** *Contrastive adaptation network for unsupervised domain adaptation.* CVPR 2019.

7. **Grandvalet, Y., & Bengio, Y. (2005).** *Semi-supervised learning by entropy minimization.* NeurIPS 17.

8. **Iscen, A., Tolias, G., Avrithis, Y., & Chum, O. (2019).** *Label propagation for deep semi-supervised learning.* CVPR 2019.

9. **Chapelle, O., Schölkopf, B., & Zien, A. (Eds.). (2006).** *Semi-Supervised Learning.* MIT Press.

10. **GitHub Repository**: https://github.com/YBZh/Label-Propagation-with-Augmented-Anchors
