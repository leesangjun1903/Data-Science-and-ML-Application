# Adversarial Domain Adaptation Being Aware of Class Relationships

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존의 적대적 도메인 적응(Adversarial Domain Adaptation, ADA) 방법들은 레이블 공간(label space)에서의 **클래스 간 의미적 관계(inter-class semantic relationships)**를 무시한 채 단순히 소스-타겟 도메인 간 분포를 정렬해왔다. 본 논문은 **클래스 간 관계(class relationships)가 도메인 간에 일관되게 유지된다**는 직관적 가정에 기반하여, 이 구조 정보를 ADA 훈련 과정에 명시적으로 주입함으로써 전이 학습 성능을 향상시킬 수 있다고 주장한다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **단일 다중 클래스 도메인 판별자 설계** | 클래스별 별도 판별자 대신 하나의 공유 판별자에 클래스별 브랜치를 설계 |
| **클래스 관계 정렬 정규화 항 도입** | 레이블 예측기와 도메인 판별자의 클래스 간 의존 구조 불일치를 최소화 |
| **파라미터 효율성 개선** | MADA 대비 파라미터 수를 약 $10^7 \rightarrow 10^6$ 수준으로 감소 |
| **부분 도메인 적응(PDA) 강건성** | 소스 도메인에 잉여 클래스가 존재하는 어려운 설정에서도 우수한 성능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**핵심 문제:** 기존 ADA 방법들은 다중 클래스 분류 환경에서 **다봉(multimodal) 데이터 분포**를 제대로 포착하지 못한다.

구체적 문제점은 두 가지다:

1. **전체 분포 정렬의 한계**: DANN처럼 도메인 전체 분포만 맞추면 클래스별 조건부 분포의 불일치가 남아 잘못된 정렬(false alignment)이 발생한다.
2. **클래스별 독립 판별자의 직교성 가정**: MADA처럼 각 클래스에 별도 판별자를 할당하면 클래스 간 의존 구조를 묵시적으로 무시(직교성 가정)하며, 파라미터 수도 클래스 수에 비례하여 증가한다.

---

### 2.2 제안 방법 (수식 포함)

#### Step 1: 클래스 간 의존 구조 모델링

$K$개 클래스 분류에서 출력층 가중치 행렬 $\mathbf{W}^{[L]} \in \mathbb{R}^{(N_{L-1}+1) \times N_L}$의 행 벡터가 다변량 가우시안 분포를 따른다고 가정한다:

$$\mathbf{W}^{[L]}_i \sim \mathcal{N}(0, \mathbf{\Omega}^{-1}), \quad i = 1, \cdots, N_{L-1}+1$$

여기서 $\mathbf{\Omega} \in \mathbb{R}^{K \times K}$는 정밀도 행렬(precision matrix)로, 각 off-diagonal 원소가 클래스 간 편상관(partial correlation)을 포착한다. 기본 분류 목적함수:

$$\min \sum_{m} L_y(f(\mathbf{x}_m), \mathbf{y}_m) $$

정밀도 행렬 학습을 위한 최적화:

$$\min_{\mathbf{\Omega}} -d \log \det(\mathbf{\Omega}) + \text{Tr}(\mathbf{W}^{[L]} \mathbf{\Omega} \mathbf{W}^{[L]\top}), \quad s.t. \quad \mathbf{\Omega} \succeq 0 $$

스펙트럼 정리(spectral theorem)에 의해 닫힌 형태(closed-form) 해:

$$\mathbf{\Omega} = d\left(\mathbf{W}^{[L]\top} \mathbf{W}^{[L]}\right)^{-1}$$

---

#### Step 2: 단일 다중 클래스 도메인 판별자 $G_d^*$

기존 DANN의 이진 도메인 판별자를 **단일 공유 레이어 + 클래스별 브랜치** 구조로 대체한다.

DANN의 기본 목적함수:

$$\min \frac{1}{M_s} \sum_{\mathbf{x}_m \in \mathcal{D}_s} L_y(G_y(G_f(\mathbf{x}_m)), \mathbf{y}_m) + \frac{\lambda_{adv}}{M_s + M_t} \sum_{\mathbf{x}_m \in \mathcal{D}_s \cup \mathcal{D}_t} L_d(G_d(\mathcal{R}(G_f(\mathbf{x}_m))), \mathbf{d}_m) $$

다중 클래스 ADA 목적함수로 확장:

$$\min \frac{1}{M_s} \sum_{\mathbf{x}_m \in \mathcal{D}_s} L_y(G_y(G_f(\mathbf{x}_m)), \mathbf{y}_m) + \frac{\lambda_{adv}}{M_s + M_t} \sum_{k=1}^{K} \sum_{\mathbf{x}_m \in \mathcal{D}_s \cup \mathcal{D}_t} \tilde{y}_m^k L_d^k(G_d^*(\mathcal{R}(G_f(\mathbf{x}_m))), \mathbf{d}_m) $$

여기서:
- $\tilde{\mathbf{y}}\_m = \{\tilde{y}_m^k\}\_{k=1}^K$: 소스 도메인에서는 $\mathbf{y}_m$의 원-핫 인코딩, 타겟 도메인에서는 예측 확률 벡터 $\hat{\mathbf{y}}_m$
- $\mathcal{R}(\cdot)$: Gradient Reversal Layer (GRL)의 의사함수

---

#### Step 3: RADA — 클래스 관계 인식 정규화

레이블 예측기 $G_y$와 도메인 판별자 $G_d^*$로부터 각각 정밀도 행렬 추정:

$$\min_{\mathbf{\Omega}_y} -d_y \log \det(\mathbf{\Omega}_y) + \text{Tr}(\mathbf{W}_y^{[L]} \mathbf{\Omega} \mathbf{W}_y^{[L]\top}); \quad s.t. \quad \mathbf{\Omega}_y \succeq 0 $$

$$\Rightarrow \mathbf{\Omega}_y = d_y \left(\mathbf{W}_y^{[L]\top} \mathbf{W}_y^{[L]}\right)^{-1}$$

$$\min_{\mathbf{\Omega}_d} -d_d \log \det(\mathbf{\Omega}_d) + \text{Tr}(\mathbf{W}_d^{[L]} \mathbf{\Omega} \mathbf{W}_d^{[L]\top}); \quad s.t. \quad \mathbf{\Omega}_d \succeq 0 $$

$$\Rightarrow \mathbf{\Omega}_d = d_d \left(\mathbf{W}_d^{[L]\top} \mathbf{W}_d^{[L]}\right)^{-1}$$

두 정밀도 행렬 간 KL 발산으로 구조 불일치 측정:

$$D_{KL}(\mathbf{\Omega}_y \| \mathbf{\Omega}_d) = \text{Tr}(\mathbf{\Omega}_y^{-1} \mathbf{\Omega}_d) - \log \det(\mathbf{\Omega}_y^{-1} \mathbf{\Omega}_d) - K $$

$$D_{KL}(\mathbf{\Omega}_d \| \mathbf{\Omega}_y) = \text{Tr}(\mathbf{\Omega}_d^{-1} \mathbf{\Omega}_y) - \log \det(\mathbf{\Omega}_d^{-1} \mathbf{\Omega}_y) - K $$

$\mathbf{\Omega}_y$, $\mathbf{\Omega}_d$의 닫힌 형태 해를 대입하면 최종 정규화 항:

**$d \rightarrow y$ 방향:**

```math
L_R = \text{Tr}\!\left(\mathbf{W}_y^{[L]} \left(\mathbf{W}_d^{[L]\top} \mathbf{W}_d^{[L]}\right)^{-1} \mathbf{W}_y^{[L]\top}\right) - \frac{d_y}{d_d}\left\{\log \det\!\left(\mathbf{W}_y^{[L]\top} \mathbf{W}_y^{[L]}\right) - \log \det\!\left(\mathbf{W}_d^{[L]\top} \mathbf{W}_d^{[L]}\right)\right\}
```

**$y \rightarrow d$ 방향:**

```math
L_R = \text{Tr}\!\left(\mathbf{W}_d^{[L]} \left(\mathbf{W}_y^{[L]\top} \mathbf{W}_y^{[L]}\right)^{-1} \mathbf{W}_d^{[L]\top}\right) - \frac{d_d}{d_y}\left\{\log \det\!\left(\mathbf{W}_d^{[L]\top} \mathbf{W}_d^{[L]}\right) - \log \det\!\left(\mathbf{W}_y^{[L]\top} \mathbf{W}_y^{[L]}\right)\right\}
```

**최종 RADA 목적함수:**

$$\min \frac{1}{M_s} \sum_{\mathbf{x}_m \in \mathcal{D}_s} L_y(G_y(G_f(\mathbf{x}_m)), \mathbf{y}_m) + \lambda_R L_R + \frac{\lambda_{adv}}{M_s + M_t} \sum_{k=1}^{K} \sum_{\mathbf{x}_m \in \mathcal{D}_s \cup \mathcal{D}_t} \tilde{y}_m^k L_D^k(G_d^*(\mathcal{R}(G_f(\mathbf{x}_m))), \mathbf{d}_m) $$

---

### 2.3 모델 구조

```
Input x_m
    ↓
Feature Extractor G_f (ResNet-50 backbone)
    ↓              ↓ (GRL 적용)
Label Predictor G_y    Multi-class Domain Discriminator G_d*
[출력층: W_y^[L]]       [공유 레이어 → 클래스별 노드: W_d^[L]]
    ↓                      ↓
Loss L_y              Loss L_d = Σ_k ỹ_m^k L_d^k
    ↓                      ↓
Ω_y 추정           ←→  Ω_d 추정
              ↓
    구조 불일치 정규화 L_R (KL Divergence)
```

- **$G_f$**: ResNet-50 (ImageNet 사전학습) 백본
- **$G_y$**: 소스 도메인 레이블 예측용 FC 레이어
- **$G_d^*$**: $x \rightarrow 512 \rightarrow K$ 구조의 단일 판별자 (클래스별 브랜치 포함)

---

### 2.4 성능 향상

#### ImageCLEF-DA (ResNet-50)

| 방법 | I→P | P→I | I→C | C→I | C→P | P→C | **평균** |
|---|---|---|---|---|---|---|---|
| DANN | 75.0 | 86.0 | 96.2 | 87.0 | 74.3 | 91.5 | 85.0 |
| MADA | 75.0 | 87.9 | 96.0 | 88.8 | 75.2 | 92.2 | 85.8 |
| **RADA $_{d \to y}$ ** | **79.2** | **92.4** | **97.5** | **91.1** | **76.6** | **95.3** | **88.7** |

#### Office-31 (ResNet-50)

| 방법 | A→W | D→W | W→D | A→D | D→A | W→A | **평균** |
|---|---|---|---|---|---|---|---|
| DANN | 82.0 | 96.9 | 99.1 | 79.7 | 68.2 | 67.4 | 82.2 |
| MADA | 90.0 | 97.4 | 99.6 | 87.8 | 70.3 | 66.4 | 85.2 |
| **RADA $_{d \to y}$ ** | **91.5** | **98.9** | **100.0** | **90.7** | **71.5** | **71.3** | **87.3** |

#### 부분 도메인 적응 (PDA, Office-31: 31→10 클래스)

| 방법 | 평균 |
|---|---|
| MADA | 73.1 |
| **RADA $_{d \to y}$ ** | **89.6** |

---

### 2.5 한계

1. **닫힌 형태 정밀도 행렬 역행렬 계산**: $\left(\mathbf{W}^{[L]\top}\mathbf{W}^{[L]}\right)^{-1}$ 계산 시 $K$가 매우 클 경우 계산 비용이 증가하며, 행렬이 특이(singular)에 가까울 수 있다.
2. **가우시안 구조 가정**: 클래스 간 의존성을 가우시안 그래피컬 모델로만 표현하는데, 비선형적/비가우시안 관계는 포착하기 어렵다.
3. **소스 도메인 레이블 의존**: UDA 설정이나 PDA에서 타겟 도메인의 의사 레이블(pseudo-label) 품질에 성능이 민감할 수 있다.
4. **제한된 벤치마크**: ImageCLEF-DA, Office-31만 평가하여 대규모 데이터셋(Office-Home, VisDA 등)에서의 검증이 부재하다.
5. **하이퍼파라미터 민감성**: $\lambda_R = 0.01$로 고정하는데, 도메인/태스크별 최적값이 다를 수 있다.

---

## 3. 모델 일반화 성능 향상 관련 분석

### 3.1 일반화 성능 향상의 이론적 근거

RADA의 일반화 성능 향상은 **Proxy A-Distance (PAD)**로 검증된다:

$$d_\mathcal{A} = 2(1 - 2\epsilon)$$

여기서 $\epsilon$은 도메인 분류기(예: SVM)의 분류 오류율이다. 낮은 PAD는 소스-타겟 도메인 간 피처 분포 불일치가 작음을 의미하며, 더 나은 일반화 능력을 시사한다. 논문의 Figure 5에서 RADA는 $C \rightarrow P$, $A \rightarrow W$ 두 태스크 모두에서 DANN, MADA보다 낮은 PAD를 달성한다.

### 3.2 일반화에 기여하는 메커니즘

**① 클래스 구조 보존을 통한 부정적 전이 방지**

클래스 간 의미적 관계(예: *ring binder*와 *paper notebook*의 양의 편상관)를 도메인 정렬 과정에서 보존함으로써, 유사 클래스 간 오분류가 감소한다. Figure 4에서 MADA가 *ring binder*를 *paper notebook*으로 오분류하는 경우를 RADA가 회피하는 것을 확인할 수 있다.

**② PDA에서의 강건성**

부분 도메인 적응(소스에 잉여 클래스 존재) 환경에서 클래스 관계 구조가 잉여 클래스의 부정적 전이를 억제하는 암묵적 필터 역할을 한다. Table 3에서 MADA 대비 평균 +16.5%의 성능 향상이 이를 뒷받침한다.

**③ 도메인 불변 피처의 질적 향상**

t-SNE 시각화(Figure 3)에서 RADA의 임베딩이 DANN, MADA보다 클래스별로 더 명확하게 분리됨을 확인할 수 있다. 이는 클래스 관계 구조 정보가 **모드(mode) 포착 능력**을 향상시켜 도메인 불변이면서도 판별력 있는 피처를 학습하게 한다는 것을 시사한다.

**④ 파라미터 효율성과 과적합 억제**

MADA($>10^7$ 파라미터) 대비 RADA($\sim 10^6$ 파라미터)의 단순화된 구조는 과적합 위험을 낮추고 소규모 도메인 데이터셋에서 더 나은 일반화를 가능하게 한다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

**① 레이블 공간 구조 활용의 패러다임 전환**

RADA는 ADA에서 피처 공간 정렬에만 집중했던 기존 패러다임에서 벗어나, **레이블 공간의 구조적 정보를 능동적으로 활용**하는 새로운 방향을 제시한다. 이는 이후 조건부 정렬(conditional alignment) 연구의 중요한 선행 연구로 자리매김한다.

**② 플러그인 모듈로서의 확장성**

저자들이 명시적으로 "add-on"임을 언급하듯, RADA의 정규화 항 $L_R$은 DANN뿐만 아니라 CDAN, SDAT 등 다양한 ADA 프레임워크에 결합 가능하다는 연구 방향을 열어준다.

**③ 구조 학습과 전이 학습의 교차점**

Gaussian Graphical Model을 전이 학습에 적용한 이 연구는 **그래프 기반 도메인 적응** 연구의 선구적 사례로, 이후 지식 그래프(KG)나 GNN을 활용한 도메인 적응 연구에 영향을 미친다.

---

### 4.2 향후 연구 시 고려할 점

**① 비선형 클래스 관계 모델링**

현재의 가우시안 정밀도 행렬은 선형 편상관만 포착한다. **그래프 신경망(GNN)** 또는 **어텐션 메커니즘** 기반의 비선형 클래스 관계 모델링이 더 복잡한 의미 구조를 포착할 수 있다.

**② 동적 클래스 관계 업데이트**

훈련 초기에는 타겟 도메인 의사 레이블의 품질이 낮으므로, 학습 진행에 따라 클래스 관계를 동적으로 업데이트하는 **커리큘럼 학습(curriculum learning)** 전략이 필요하다.

**③ 대규모 클래스 환경 ($K \gg 100$)**

$K$가 커질수록 $K \times K$ 정밀도 행렬의 역행렬 계산 비용이 $O(K^3)$으로 증가한다. **희소 정밀도 행렬 근사** 또는 **저랭크(low-rank) 분해** 방법이 필요하다.

**④ 열린 집합 도메인 적응(Open-Set DA)으로의 확장**

PDA에서의 강건성이 확인된 만큼, 타겟 도메인에 소스에 없는 미지 클래스가 존재하는 **열린 집합 설정**에서 클래스 관계 구조를 이용한 미지 클래스 탐지 연구로 확장 가능하다.

**⑤ 자기 지도 학습(Self-Supervised Learning)과의 결합**

최근 DINO, MoCo 등 자기 지도 학습으로 사전학습된 백본에서 클래스 간 의미 구조가 더 풍부하게 인코딩된다. 이러한 백본과 RADA의 구조 정렬 정규화를 결합하면 더욱 강력한 일반화가 기대된다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 비교는 논문에서 직접 인용된 연구가 아닌, RADA와 관련된 후속 연구들에 대한 분석입니다. 해당 연구들의 구체적 수치는 논문 원문을 직접 확인하시기 바랍니다.

| 연구 | 핵심 아이디어 | RADA와의 관계 |
|---|---|---|
| **CDAN** (Long et al., NeurIPS 2018) | 조건부 피처 정렬 (판별자 입력에 클래스 예측 결합) | RADA의 직접적 선행/경쟁 연구; 클래스 정보 활용 방식의 차이 |
| **MDD** (Zhang et al., ICML 2019) | 마진 분산 불일치(Margin Disparity Discrepancy) 최소화 | 이론적 기반 강화, 구조 학습 미포함 |
| **SDAT** (Rangwani et al., ICML 2022) | 날카로운 손실 지형(sharp loss landscape) 회피를 통한 일반화 | SAM 최적화 기반; RADA의 구조 정규화와 상보적 결합 가능 |
| **CDTrans** (Xu et al., ICLR 2022) | Transformer 기반 크로스-도메인 어텐션 | 비선형 클래스 관계를 어텐션으로 암묵적 포착 |
| **PMTrans** (Zhu et al., ECCV 2022) | 패치 혼합(patch-mix) 전략 + Transformer | 피처 수준 구조 활용; 레이블 공간 구조는 미포함 |

**⚠️ 주의:** 위 2020년 이후 연구들의 세부 수치 비교는 해당 원논문을 직접 확인하시기 바랍니다. 본 분석에서는 논문 PDF에 포함된 내용과 연구 방향의 일반적 맥락을 기준으로 기술하였습니다.

---

## 참고 자료

**본 논문 (직접 참조):**
- Wang, Z., Jing, B., Ni, Y., Dong, N., Xie, P., & Xing, E. (2019). *Adversarial Domain Adaptation Being Aware of Class Relationships*. arXiv:1905.11931v1.

**논문 내 인용 문헌 (직접 참조):**
- Ganin, Y., et al. (2016). *Domain-adversarial training of neural networks*. JMLR, 17(1):2096–2030.
- Pei, Z., et al. (2018). *Multi-adversarial domain adaptation*. AAAI.
- Long, M., et al. (2018). *Conditional adversarial domain adaptation*. NeurIPS.
- Jiang, Y.-G., et al. (2018). *Exploiting feature and class relationships in video categorization with regularized deep neural networks*. IEEE TPAMI, 40(2):352–364.
- Zhang, Y. & Yeung, D.-Y. (2010). *A convex formulation for learning task relationships in multi-task learning*. UAI.
- Cao, Z., et al. (2018). *Partial adversarial domain adaptation*. ECCV.
- He, K., et al. (2016). *Deep residual learning for image recognition*. CVPR.
