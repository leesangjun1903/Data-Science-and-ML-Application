# Transferrable Prototypical Networks for Unsupervised Domain Adaptation (TPN) 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

TPN은 **Prototypical Networks**를 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)에 재설계하여, 소스와 타겟 도메인 간의 **다중 입도(multi-granular) 도메인 불일치**를 클래스 수준과 샘플 수준에서 동시에 줄이는 것이 가능하다고 주장합니다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **General-purpose Adaptation** | 각 클래스의 프로토타입을 서로 다른 도메인에서 임베딩 공간 내에서 가깝게 유지 (클래스 수준) |
| **Task-specific Adaptation** | 각 샘플에 대한 서로 다른 프로토타입 분류기의 점수 분포를 KL-divergence로 정렬 (샘플 수준) |
| **Pseudo Label 전략** | 소스 프로토타입 기반 분류기로 타겟 샘플에 의사 레이블(pseudo label)을 부여 |
| **이론적 분석** | Ben-David et al.(2010)의 이론을 확장하여 TPN의 오차 상한(error bound) 제시 |
| **성능** | VisDA 2017에서 단일 모델 기준 80.4% 달성 (당시 SOTA) |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

기존 UDA 방법(MMD 기반, Domain Discriminator 기반)은 소스·타겟 도메인 전체의 **홀리스틱(holistic) 분포 차이**만 줄이려 하였으며, 아래 두 가지 측면이 미탐색 상태였습니다:

1. **클래스 수준의 도메인 불일치**: 타겟 샘플에 레이블이 없어 클래스별 도메인 차이를 직접 측정하기 어려움
2. **샘플-분류기 관계**: 각 샘플이 각 도메인의 분류기에서 동일한 결정을 내리도록 유도하는 메커니즘 부재

---

### 2-2. 제안하는 방법 및 수식

#### (1) Prototypical Networks 기초

클래스 $c$의 프로토타입:

$$\mu_c = \frac{1}{|S_c|} \sum_{x_i \in S_c} f(x_i; \theta) \tag{1}$$

샘플 $x_i$의 클래스 $c$에 대한 점수 분포 (softmax over distances):

$$\mathbf{P}_{ic} = p(y_i = c | x_i) = \frac{e^{-d(f(x_i;\theta), \mu_c)}}{\sum_{c'} e^{-d(f(x_i;\theta), \mu_{c'})}} \tag{2}$$

소스 데이터에 대한 분류 손실:

$$L_S(x_i) = -\log p(y_i = c | x_i) \tag{3}$$

---

#### (2) 세 종류의 프로토타입 계산

소스 전용, 타겟 전용(의사 레이블 사용), 소스+타겟 혼합 프로토타입:

$$\mu_c^s = \frac{1}{|S_c^s|} \sum_{x_i^s \in S_c^s} f(x_i^s; \theta)$$

$$\mu_c^t = \frac{1}{|\hat{S}_c^t|} \sum_{x_i^t \in \hat{S}_c^t} f(x_i^t; \theta)$$

$$\mu_c^{st} = \frac{1}{|S_c^s| + |\hat{S}_c^t|} \left( \sum_{x_i^s \in S_c^s} f(x_i^s; \theta) + \sum_{x_i^t \in \hat{S}_c^t} f(x_i^t; \theta) \right) \tag{4}$$

---

#### (3) General-purpose Adaptation: 클래스 수준 손실 ($L_G$)

RKHS(Reproducing Kernel Hilbert Space) 내에서 서로 다른 도메인의 클래스 프로토타입 간 거리를 최소화:

$$L_G\left(\{\mu_c^s\}, \{\mu_c^t\}, \{\mu_c^{st}\}\right) = \frac{1}{C}\sum_{c=1}^C \|\tilde{\mu}_c^s - \tilde{\mu}_c^t\|_{\mathcal{H}}^2 + \frac{1}{C}\sum_{c=1}^C \|\tilde{\mu}_c^s - \tilde{\mu}_c^{st}\|_{\mathcal{H}}^2 + \frac{1}{C}\sum_{c=1}^C \|\tilde{\mu}_c^t - \tilde{\mu}_c^{st}\|_{\mathcal{H}}^2 \tag{5}$$

> **MMD와의 관계**: MMD는 전체 도메인의 홀리스틱 프로토타입 간 거리를 측정하는 반면, $L_G$는 **클래스별** 프로토타입 간의 세밀한 정렬을 수행합니다.

$$\mu^s = \frac{1}{|S^s|}\sum_{x_i^s \in S^s} \phi(x_i^s), \quad \mu^t = \frac{1}{|S^t|}\sum_{x_i^t \in S^t} \phi(x_i^t)$$

$$L_{MMD} = \|\mu^s - \mu^t\|_{\mathcal{H}}^2 \tag{6}$$

---

#### (4) Task-specific Adaptation: 샘플 수준 손실 ($L_T$)

각 샘플에 대해 세 종류의 프로토타입 분류기가 만드는 점수 분포 간 KL-divergence를 최소화:

$$L_T\left(\{\mathbf{P}_i^s\}, \{\mathbf{P}_i^t\}, \{\mathbf{P}_i^{st}\}\right) = \frac{1}{|S^s|+|\hat{S}^t|}\sum_{x_i} D_{KL}(\mathbf{P}_i^s, \mathbf{P}_i^t) + \frac{1}{|S^s|+|\hat{S}^t|}\sum_{x_i} D_{KL}(\mathbf{P}_i^s, \mathbf{P}_i^{st}) + \frac{1}{|S^s|+|\hat{S}^t|}\sum_{x_i} D_{KL}(\mathbf{P}_i^t, \mathbf{P}_i^{st}) \tag{7}$$

여기서 대칭 KL-divergence:

$$D_{KL}(\mathbf{P}_i^s, \mathbf{P}_i^t) = \frac{1}{2}\left(d_{KL}(\mathbf{P}_i^s \| \mathbf{P}_i^t) + d_{KL}(\mathbf{P}_i^t \| \mathbf{P}_i^s)\right)$$

$$d_{KL}(\mathbf{P}_i^s \| \mathbf{P}_i^t) = \sum_{c=1}^C \mathbf{P}_{ic}^s \log\frac{\mathbf{P}_{ic}^s}{\mathbf{P}_{ic}^t}$$

---

#### (5) 최종 최적화 목적 함수

$$\min_\theta \frac{1}{|S^s|}\sum_{x_i^s \in S^s} L_S(x_i^s) + \alpha L_G\left(\{\mu_c^s\}, \{\mu_c^t\}, \{\mu_c^{st}\}\right) + \beta L_T\left(\{\mathbf{P}_i^s\}, \{\mathbf{P}_i^t\}, \{\mathbf{P}_i^{st}\}\right) \tag{8}$$

실험에서 $\alpha = \beta = 1$로 고정.

---

### 2-3. 이론적 분석 (오차 상한)

**Lemma 1**: 가설 $h \in \mathcal{H}$에 대해:

$$\left| \epsilon_\gamma(h) - \epsilon_t(h, y^t) \right| \leq (1-\gamma)\left(\frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}^s, \mathcal{D}^t) + \lambda\right) + \gamma\rho \tag{10}$$

- $d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}^s, \mathcal{D}^t)$: 가설 공간에서의 도메인 불일치
- $\lambda$: 결합 이상적 가설의 오차 (일반적으로 무시 가능)
- $\rho$: 의사 레이블의 노이즈 비율 → 반복 학습으로 지속 감소

혼합 학습 오차:

$$\epsilon_\gamma(h) = \gamma \epsilon_t(h, \hat{y}^t) + (1-\gamma)\epsilon_s(h, y^s) \tag{9}$$

$$h^* = \arg\min \epsilon_s(h, y^s) + \epsilon_t(h, y^t) \tag{11}$$

---

### 2-4. 모델 구조

```
입력 배치(소스 레이블 + 타겟 비레이블)
        ↓
임베딩 함수 f(x; θ) [CNN: LeNet or ResNet-50]
        ↓
[Step 1] 소스 프로토타입으로 타겟 의사 레이블 부여
        ↓
[Step 2] 세 종류 프로토타입 계산: μ^s_c, μ^t_c, μ^{st}_c
        ↓
  ┌─────────────────────┬─────────────────────┐
  │  General-purpose    │   Task-specific     │
  │  (클래스 수준, L_G) │  (샘플 수준, L_T)   │
  └─────────────────────┴─────────────────────┘
        ↓
전체 손실 최소화: L_S + α·L_G + β·L_T (End-to-end)
        ↓
추론: 세 프로토타입 집합 중 하나로 분류
```

---

### 2-5. 성능 결과

#### Digits Image Transfer (%)

| Method | M→U | U→M | S→M |
|---|---|---|---|
| Source-only | 75.2 | 57.1 | 60.1 |
| ADDA | 89.4 | 90.1 | 76.0 |
| MCD | 90.0 | 88.5 | 83.3 |
| **TPN (제안)** | **92.1** | **94.1** | **93.0** |
| Train-on-target | 92.3 | 96.8 | 96.8 |

#### VisDA 2017 Synthetic→Real (%)

| Method | Mean Acc. |
|---|---|
| JAN | 66.5 |
| MCD | 71.9 |
| S-En + Mini-aug | 74.2 |
| **TPN (단일 모델)** | **80.4** |
| S-En + Test-aug | 82.8 |

---

### 2-6. 한계점

1. **의사 레이블 노이즈**: 초기 타겟 레이블 정확도가 약 55.3%로 낮으며, 초반에 노이즈가 누적될 위험이 있음
2. **임계값 고정**: 의사 레이블 신뢰 임계값(0.6)이 고정되어 도메인별 최적값이 다를 수 있음
3. **하이퍼파라미터 민감성**: $\alpha, \beta$ 튜닝 없이 1로 고정하였으나, 도메인 조합에 따른 최적화가 제한적
4. **폐쇄 집합 가정**: 소스와 타겟의 클래스 집합이 동일하다고 가정 (open-set UDA 미지원)
5. **대규모 클래스 수 확장성**: 클래스 수가 많을 때 프로토타입 계산 비용이 증가

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 다중 입도 정렬의 효과

TPN은 기존 방법들이 간과한 **클래스 수준**과 **샘플 수준**의 도메인 불일치를 동시에 줄임으로써 더 세밀한 도메인 불변 표현을 학습합니다. t-SNE 시각화에서도 확인되듯, 학습이 진행될수록 소스와 타겟 분포가 클래스별로 잘 정렬됩니다.

### 3-2. 프로토타입 기반 분류기의 장점

- 프로토타입은 **클래스 전체 분포의 평균**을 나타내므로, 개별 샘플 노이즈에 덜 민감합니다.
- 세 종류의 프로토타입($\mu^s_c, \mu^t_c, \mu^{st}_c$) 중 어떤 것을 사용해도 정확도 변동이 **0.002 이하**로 안정적 → 표현의 도메인 불변성 입증

### 3-3. 오차 상한의 점진적 개선

Lemma 1에서 오차 상한의 세 항 중:
- $d_{\mathcal{H}\Delta\mathcal{H}}$는 $L_G, L_T$로 감소
- $\rho$(의사 레이블 노이즈)는 반복 학습으로 점진적 감소 (44.7% → ~19.6%)

이 선순환 구조가 일반화 성능 향상의 이론적 근거를 제공합니다.

### 3-4. 한계와 향후 방향

- 의사 레이블의 초기 품질이 낮은 경우(대규모 도메인 갭), $\rho$가 빠르게 감소하지 않을 수 있음
- **self-training과의 결합**, **data augmentation** 도입 시 일반화 성능 추가 향상 가능성 있음

---

## 4. 연구에 미치는 영향과 앞으로 고려할 점

### 4-1. 앞으로의 연구에 미치는 영향

| 영향 분야 | 설명 |
|---|---|
| **프로토타입 기반 DA** | Few-shot learning과 도메인 적응의 융합 방향 제시 |
| **클래스 조건부 정렬** | 클래스 레벨 도메인 정렬의 중요성을 실증적으로 확인 |
| **자기 레이블링 활용** | Pseudo label + 반복 정제 방식의 UDA 적용 가능성 확대 |
| **이론적 기반** | 멀티그레인 DA의 오차 상한 분석 틀 제공 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 연구들은 제가 학습한 지식을 기반으로 서술하며, 논문 원문을 직접 확인하지 못한 경우 부정확할 수 있으므로, 반드시 원문을 검토하시기 바랍니다.

### 5-1. TPN과 주요 후속 연구 비교

#### (1) **SHOT** (Liang et al., ICML 2020)
- **핵심 아이디어**: 소스 모델의 가중치를 고정하고, 타겟 도메인 데이터만으로 특징 추출기를 미세 조정 (소스 데이터 불필요)
- **TPN과의 차이**: TPN은 소스·타겟 데이터를 동시에 사용하며 프로토타입을 명시적으로 정렬하는 반면, SHOT은 정보 최대화(IM) + 의사 레이블로 소스 없이 적응
- **성능**: VisDA에서 약 87.6% (TPN 80.4% 대비 우세)

#### (2) **ATDOC** (Liu et al., CVPR 2021)
- **핵심 아이디어**: 이웃 기반 집계를 통한 타겟 도메인 분류기 최적화
- **프로토타입 연관성**: TPN의 프로토타입과 유사하게 클래스 중심(centroid)을 활용하지만, 근방 정보를 추가 활용
- **개선점**: 노이즈 의사 레이블 문제를 더 효과적으로 완화

#### (3) **NRC** (Yang et al., NeurIPS 2021)
- **핵심 아이디어**: 소스 데이터 없이 타겟 구조 내 상호 근방 관계(Neighborhood Reciprocity Clustering) 활용
- **TPN 대비**: 소스 데이터 의존성 제거라는 측면에서 더 실용적

#### (4) **TVT** (Yang et al., CVPR 2023)
- **핵심 아이디어**: Vision Transformer(ViT) 기반으로 transferability를 토큰 수준에서 측정
- **TPN과의 차이**: TPN이 CNN 기반 프로토타입 정렬인 반면, TVT는 Transformer의 어텐션 구조 내에서 전이 가능성 모델링

### 5-2. 종합 비교 표

| 연구 | 연도 | 클래스 수준 정렬 | 샘플 수준 정렬 | 소스 데이터 필요 | 백본 | VisDA 성능 |
|---|---|---|---|---|---|---|
| **TPN** | 2019 | ✅ (프로토타입 RKHS) | ✅ (KL-div) | ✅ | ResNet-50 | 80.4% |
| SHOT | 2020 | 부분적 (IM) | ✅ | ❌ | ResNet-101 | ~87.6% |
| ATDOC | 2021 | ✅ (centroid) | ✅ | ✅ | ResNet-101 | ~86.2% |
| NRC | 2021 | ✅ (cluster) | ✅ | ❌ | ResNet-101 | ~85.0% |
| TVT | 2023 | ✅ | ✅ | ✅ | ViT | ~92.4% |

> **주의**: 위 수치는 백본(ResNet-50 vs ResNet-101 vs ViT) 차이가 있어 직접 비교 시 주의가 필요합니다.

---

### 6. 앞으로 연구 시 고려할 점

1. **Source-free DA로의 확장**: 소스 데이터 프라이버시 문제가 부각되면서, 소스 없이 타겟 적응을 수행하는 방향 (SHOT 계열)이 실용적으로 중요해짐

2. **의사 레이블 품질 향상**: 단순 임계값(0.6) 방식보다 **불확실성 기반 샘플 선택** 또는 **혼합 학습(mixup)** 적용으로 노이즈 저감 가능

3. **Open-set/Partial DA**: TPN은 폐쇄 집합 가정을 전제하므로, 클래스 불일치 상황(open-set/partial DA)으로의 확장 필요

4. **Vision-Language 모델 활용**: CLIP 등 대규모 사전학습 모델을 백본으로 사용하면 프로토타입의 의미론적 표현력이 크게 향상될 수 있음

5. **다중 소스 도메인**: 단일 소스 → 단일 타겟 구조를 다중 소스로 확장 시 프로토타입 집계 전략 재설계 필요

6. **Transformer 기반 재설계**: ViT의 self-attention이 자연스럽게 프로토타입 유사 클러스터링을 수행하므로, TPN의 아이디어를 Transformer 구조에 통합하면 성능 향상 기대

---

## 참고 자료 (출처)

1. **논문 원문**: Yingwei Pan, Ting Yao, Yehao Li, Yu Wang, Chong-Wah Ngo, Tao Mei. *"Transferrable Prototypical Networks for Unsupervised Domain Adaptation"*. arXiv:1904.11227v1, 2019.

2. **Ben-David et al.** (2010). *"A theory of learning from different domains"*. Machine Learning.

3. **Snell et al.** (2017). *"Prototypical networks for few-shot learning"*. NeurIPS.

4. **Long et al.** (2015). *"Learning transferable features with deep adaptation networks (DAN)"*. ICML.

5. **Tzeng et al.** (2017). *"Adversarial discriminative domain adaptation (ADDA)"*. CVPR.

6. **Saito et al.** (2018). *"Maximum classifier discrepancy for unsupervised domain adaptation (MCD)"*. CVPR.

7. **Liang et al.** (2020). *"Do we really need to access the source data? Source hypothesis transfer for unsupervised domain adaptation (SHOT)"*. ICML.

8. **Gretton et al.** (2012). *"A kernel two-sample test (MMD)"*. JMLR.

> ⚠️ SHOT, ATDOC, NRC, TVT의 구체적 수치는 제 학습 데이터 기반 추정치이므로, 정확한 수치는 각 원문 논문을 직접 확인하시기 바랍니다.
