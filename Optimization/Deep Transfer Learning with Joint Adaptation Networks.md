# Deep Transfer Learning with Joint Adaptation Networks (JAN) 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존의 딥 전이 학습 방법들은 소스 도메인과 타겟 도메인 간의 **주변 분포(marginal distribution)** $P(\mathbf{X}^s)$와 $Q(\mathbf{X}^t)$만을 정렬하였으나, 실제 도메인 시프트는 **입력 특징과 출력 레이블의 결합 분포(joint distribution)** $P(\mathbf{X}, \mathbf{Y})$에서 발생한다. JAN은 이 결합 분포를 직접 정렬함으로써 보다 효과적인 도메인 적응이 가능하다고 주장한다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **JMMD 제안** | 결합 분포 간 불일치를 측정하는 Joint Maximum Mean Discrepancy 정의 |
| **JAN 아키텍처** | 다수 도메인 특화 레이어의 결합 분포를 동시에 정렬하는 전이 네트워크 |
| **JAN-A (Adversarial)** | JMMD를 극대화하는 적대적 학습 전략으로 분포 구별력 강화 |
| **선형 시간 추정** | JMMD의 선형 시간 불편 추정기 도출로 대규모 학습 가능 |
| **SOTA 달성** | Office-31, ImageCLEF-DA 벤치마크에서 당시 최고 성능 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 정의:**

소스 도메인 $\mathcal{D}\_s = \{(\mathbf{x}^s_i, \mathbf{y}^s_i)\}^{n_s}\_{i=1}$와 레이블이 없는 타겟 도메인 $\mathcal{D}\_t = \{\mathbf{x}^t_j\}^{n_t}_{j=1}$이 주어졌을 때, 결합 분포 $P(\mathbf{X}^s, \mathbf{Y}^s) \neq Q(\mathbf{X}^t, \mathbf{Y}^t)$인 상황에서 타겟 리스크를 최소화하는 것이다.

$$R_t(f) = \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim Q}[f(\mathbf{x}) \neq \mathbf{y}]$$

**기존 방법의 한계:**

- DDC, DAN, RevGrad 등은 각 레이어의 **주변 분포만** 독립적으로 정렬
- 조건부 분포 $P(\mathbf{Y}|\mathbf{X})$의 시프트(conditional shift)를 무시
- 레이어 간 상호작용(interaction)을 반영하지 못함

**AlexNet 기준 문제 레이어:**

깊은 신경망에서 특징은 일반(general) → 특화(specific) 방향으로 변화하며, $\mathcal{L} = \{fc6, fc7, fc8\}$ 레이어의 활성화에 도메인 시프트가 잔존한다.

---

### 2.2 제안 방법 및 수식

#### Step 1: 힐베르트 공간 임베딩 (Hilbert Space Embedding)

분포 $P$를 재생 핵 힐베르트 공간(RKHS) $\mathcal{H}$의 원소로 표현:

$$\mu_{\mathbf{X}}(P) \triangleq \mathbb{E}_{\mathbf{X}}[\phi(\mathbf{X})] = \int_{\Omega} \phi(\mathbf{x}) \, dP(\mathbf{x}) \tag{1}$$

유한 샘플 $\mathcal{D}_{\mathbf{X}} = \{x_1, \ldots, x_n\}$에 대한 경험적 추정:

$$\hat{\mu}_{\mathbf{X}} = \frac{1}{n} \sum_{i=1}^{n} \phi(x_i) \tag{2}$$

#### Step 2: 결합 분포의 커널 임베딩

$m$개 변수 $\mathbf{X}^1, \ldots, \mathbf{X}^m$의 결합 분포 임베딩 (텐서 곱 특징 공간):

$$\mathcal{C}_{\mathbf{X}^{1:m}}(P) \triangleq \mathbb{E}_{\mathbf{X}^{1:m}}\left[\otimes^m_{\ell=1} \phi^\ell(\mathbf{X}^\ell)\right] = \int_{\times^m_{\ell=1} \Omega^\ell} \left(\otimes^m_{\ell=1} \phi^\ell(\mathbf{x}^\ell)\right) dP(\mathbf{x}^1, \ldots, \mathbf{x}^m) \tag{3}$$

경험적 결합 임베딩:

$$\hat{\mathcal{C}}_{\mathbf{X}^{1:m}} = \frac{1}{n} \sum_{i=1}^{n} \otimes^m_{\ell=1} \phi^\ell(\mathbf{x}^\ell_i) \tag{4}$$

#### Step 3: MMD (Maximum Mean Discrepancy)

두 분포 $P, Q$의 차이를 측정하는 커널 이-표본 검정 통계량:

$$D_{\mathcal{H}}(P, Q) \triangleq \sup_{f \in \mathcal{H}} \left(\mathbb{E}_{\mathbf{X}^s}[f(\mathbf{X}^s)] - \mathbb{E}_{\mathbf{X}^t}[f(\mathbf{X}^t)]\right) \tag{5}$$

경험적 추정 (이차 복잡도):

$$\hat{D}_{\mathcal{H}}(P, Q) = \frac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} k(\mathbf{x}^s_i, \mathbf{x}^s_j) + \frac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} k(\mathbf{x}^t_i, \mathbf{x}^t_j) - \frac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} k(\mathbf{x}^s_i, \mathbf{x}^t_j) \tag{6}$$

#### Step 4: Joint MMD (JMMD) — 핵심 기여

다중 레이어 $\mathcal{L}$에서의 결합 분포 간 불일치 측정:

$$D_{\mathcal{L}}(P, Q) \triangleq \left\| \mathcal{C}_{\mathbf{Z}^{s,1:|\mathcal{L}|}}(P) - \mathcal{C}_{\mathbf{Z}^{t,1:|\mathcal{L}|}}(Q) \right\|^2_{\otimes^{|\mathcal{L}|}_{\ell=1} \mathcal{H}^\ell} \tag{8}$$

경험적 JMMD 추정 (이차):

$$\hat{D}_{\mathcal{L}}(P, Q) = \frac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} \prod_{\ell \in \mathcal{L}} k^\ell(\mathbf{z}^{s\ell}_i, \mathbf{z}^{s\ell}_j) + \frac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} \prod_{\ell \in \mathcal{L}} k^\ell(\mathbf{z}^{t\ell}_i, \mathbf{z}^{t\ell}_j) - \frac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} \prod_{\ell \in \mathcal{L}} k^\ell(\mathbf{z}^{s\ell}_i, \mathbf{z}^{t\ell}_j) \tag{9}$$

> **MMD와 JMMD의 핵심 차이:** JMMD에서는 각 레이어 $\ell$의 커널 함수 $k^\ell(\mathbf{z}^\ell_i, \mathbf{z}^\ell_j)$에 다른 레이어 $\mathcal{L} \setminus \ell$의 영향을 반영한 **비균일 가중치**가 적용된다. 이는 레이어 간 상호작용을 포착한다.

#### Step 5: JAN 학습 목표

$$\min_f \frac{1}{n_s} \sum_{i=1}^{n_s} J(f(\mathbf{x}^s_i), \mathbf{y}^s_i) + \lambda \hat{D}_{\mathcal{L}}(P, Q) \tag{10}$$

여기서 $J(\cdot, \cdot)$는 교차 엔트로피 손실이며, $\lambda > 0$는 JMMD 패널티의 가중치이다.

#### Step 6: 선형 시간 JMMD 추정

미니배치 SGD에 적합한 선형 시간 불편 추정기:

$$\hat{D}_{\mathcal{L}}(P, Q) = \frac{2}{n} \sum_{i=1}^{n/2} \left(\prod_{\ell \in \mathcal{L}} k^\ell(\mathbf{z}^{s\ell}_{2i-1}, \mathbf{z}^{s\ell}_{2i}) + \prod_{\ell \in \mathcal{L}} k^\ell(\mathbf{z}^{t\ell}_{2i-1}, \mathbf{z}^{t\ell}_{2i})\right)$$
$$- \frac{2}{n} \sum_{i=1}^{n/2} \left(\prod_{\ell \in \mathcal{L}} k^\ell(\mathbf{z}^{s\ell}_{2i-1}, \mathbf{z}^{t\ell}_{2i}) + \prod_{\ell \in \mathcal{L}} k^\ell(\mathbf{z}^{t\ell}_{2i-1}, \mathbf{z}^{s\ell}_{2i})\right) \tag{11}$$

#### Step 7: JAN-A (적대적 학습)

커널 기반 MMD의 기울기 소실 문제를 해결하기 위해 신경망 파라미터 $\theta$로 JMMD를 극대화:

$$\min_f \max_\theta \frac{1}{n_s} \sum_{i=1}^{n_s} J(f(\mathbf{x}^s_i), \mathbf{y}^s_i) + \lambda \hat{D}_{\mathcal{L}}(P, Q; \theta) \tag{12}$$

---

### 2.3 모델 구조

```
[소스 데이터 X^s] ──┐
                    ├── [CNN Backbone (AlexNet/ResNet)] ──→ [Z^{s1}, ..., Z^{s|L|}] ──→ Y^s
[타겟 데이터 X^t] ──┘                                      [Z^{t1}, ..., Z^{t|L|}] ──→ Y^t
                                                                     ↓
                                                              JMMD 최소화
```

| 구성 요소 | JAN | JAN-A |
|---|---|---|
| **백본** | AlexNet / ResNet-50 | 동일 |
| **적응 레이어 (AlexNet)** | $\mathcal{L} = \{fc6, fc7, fc8\}$ | 동일 |
| **적응 레이어 (ResNet)** | $\mathcal{L} = \{pool5, fc\}$ | 동일 |
| **분포 정렬** | JMMD 최소화 | JMMD 최소화 + $\theta$ 최대화 |
| **커널** | Gaussian (median 휴리스틱) | 추가 FC 레이어로 함수 클래스 확장 |
| **학습률 스케줄** | $\eta_p = \frac{\eta_0}{(1 + \alpha p)^\beta}$ | 동일 |
| **적응 인자 스케줄** | $\lambda_p = \frac{2}{1 + \exp(-\gamma p)} - 1$ | 동일 |

---

### 2.4 성능 향상

#### Office-31 (ResNet 기준)

| 방법 | A→W | D→W | W→D | A→D | D→A | W→A | **평균** |
|---|---|---|---|---|---|---|---|
| ResNet | 68.4 | 96.7 | 99.3 | 68.9 | 62.5 | 60.7 | 76.1 |
| DAN | 80.5 | 97.1 | 99.6 | 78.6 | 63.6 | 62.8 | 80.4 |
| RevGrad | 82.0 | 96.9 | 99.1 | 79.7 | 68.2 | 67.4 | 82.2 |
| **JAN** | **85.4** | **97.4** | **99.8** | **84.7** | **68.6** | **70.0** | **84.3** |
| **JAN-A** | **86.0** | 96.7 | 99.7 | **85.1** | **69.2** | **70.7** | **84.6** |

#### ImageCLEF-DA (ResNet 기준)

| 방법 | 평균 |
|---|---|
| RTN | 83.9 |
| **JAN** | **85.8** |

---

### 2.5 한계점

1. **커널 선택 민감성:** Gaussian 커널이 고차원 자연 이미지 공간에서 복잡한 거리를 충분히 포착하지 못할 수 있다 (Arjovsky et al., 2017).
2. **타겟 레이블 미활용:** 완전한 비지도 학습으로 조건부 분포 $P(\mathbf{Y}|\mathbf{X})$를 직접 정렬하지 못한다.
3. **레이어 수 증가에 따른 텐서 곱 확장:** $|\mathcal{L}|$이 커질수록 텐서 곱 공간의 차원이 폭발적으로 증가한다.
4. **하이퍼파라미터 $\lambda$ 민감성:** $\lambda$에 따라 성능이 종 모양 곡선을 그리므로 세심한 튜닝이 필요하다.
5. **벤치마크 편향:** Office-31과 ImageCLEF-DA는 비교적 소규모 데이터셋으로, 대규모 실세계 시나리오에서의 일반화 보장이 부족하다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 근거

Ben-David et al. (2010)의 도메인 적응 이론에 따르면 타겟 리스크는 다음과 같이 경계 지어진다:

$$R_t(f) \leq R_s(f) + d_{\mathcal{A}}(P, Q) + \lambda^* \tag{이론적 경계}$$

여기서 $d_{\mathcal{A}}(P, Q)$는 $\mathcal{A}$-거리, $\lambda^*$는 두 도메인에서 최적 가설의 공유 오류다. JAN은 결합 분포 정렬을 통해 $d_{\mathcal{A}}$를 감소시켜 이 경계를 직접 줄인다.

**실험적 검증:**

- 논문의 Figure 3(a): JAN의 $\mathcal{A}$-거리가 CNN, DAN 대비 현저히 낮음
- Figure 3(b): JAN의 JMMD 값이 CNN, DAN 대비 현저히 낮음
- Figure 2(c)(d): t-SNE 시각화에서 타겟 카테고리가 더 명확하게 구분됨

### 3.2 일반화 향상 메커니즘

```
결합 분포 정렬
    ↓
특징 레이어: P(X^s) ≈ Q(X^t) 달성
분류기 레이어: P(Y^s) ≈ Q(Y^t) 달성
    ↓
레이어 간 상호작용 반영 (텐서 곱 커널)
    ↓
더 강한 도메인 불변 표현 학습
    ↓
타겟 도메인 일반화 향상
```

### 3.3 일반화 성능에 영향을 미치는 요인

| 요인 | JAN에서의 처리 | 효과 |
|---|---|---|
| **주변 분포 시프트** | JMMD의 특징 레이어 항 | 직접 감소 |
| **레이블 분포 시프트** | JMMD의 분류기 레이어 항 | 직접 감소 |
| **레이어 간 상관관계** | 텐서 곱으로 포착 | 간접 보정 |
| **표현력 부족** | JAN-A의 적대적 학습 | 함수 클래스 확장 |

### 3.4 한계와 개선 여지

- **조건부 시프트 해결 미흡:** 타겟 레이블이 없으므로 $P(\mathbf{Y}|\mathbf{X})$를 직접 정렬할 수 없다. 의사 레이블(pseudo-label)이나 자기 지도 학습(self-supervised learning)을 결합하면 개선 가능하다.
- **도메인 외 일반화(Out-of-domain generalization):** 훈련 시 보지 못한 완전히 새로운 도메인에 대한 일반화 능력은 별도로 보장되지 않는다.

---

## 4. 연구에 미치는 영향 및 향후 연구 시 고려할 점

### 4.1 이 논문이 앞으로의 연구에 미치는 영향

**① 결합 분포 정렬의 중요성 확립**

JAN은 도메인 적응에서 주변 분포만이 아닌 결합 분포 정렬이 필수적임을 실증적으로 보여주었다. 이후 연구들이 보다 정교한 결합 분포 추정 및 정렬 방법을 탐구하는 데 직접적인 동기를 제공했다.

**② 커널-적대적 학습의 결합**

JAN-A는 커널 방법과 적대적 학습의 장점을 결합한 선구적 사례로, 이후 CDAN, MDD 등의 방법론적 발전에 영감을 주었다.

**③ 다층 적응의 표준화**

다수의 도메인 특화 레이어를 동시에 적응하는 접근법은 이후 도메인 적응 연구의 표준적 설계 원칙으로 자리잡았다.

**④ 분포 측도 설계에 대한 방향 제시**

단순 MMD에서 JMMD로의 확장은 이후 Wasserstein 거리, 최적 수송(Optimal Transport) 기반 방법론의 발전을 촉진했다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

#### 4.2.1 CDAN (Conditional Domain Adversarial Networks)

**논문:** Long et al., "Conditional Adversarial Domain Adaptation," NeurIPS 2018

JAN의 결합 분포 개념을 발전시켜, 특징 표현과 분류기 예측의 **다중선형 조건 결합**을 활용:

$$h(\mathbf{z}, \hat{\mathbf{y}}) = \mathbf{z} \otimes \hat{\mathbf{y}}$$

**JAN과의 비교:**

| 항목 | JAN | CDAN |
|---|---|---|
| 결합 방식 | 레이어 활성화의 텐서 곱 (커널) | 특징 × 예측 외적 |
| 레이블 정보 | 간접 (fc8 레이어) | 소프트맥스 예측 직접 활용 |
| 적대적 학습 | JAN-A에서 선택적 | 기본 구조에 내장 |

#### 4.2.2 MDD (Margin Disparity Discrepancy)

**논문:** Zhang et al., "Bridging Theory and Algorithm for Domain Adaptation," ICML 2019

새로운 이론적 경계를 기반으로 한 분포 불일치 측도 제안:

$$d_{f,f'}(\mathcal{D}_s, \mathcal{D}_t) = \mathbb{E}_{\mathcal{D}_s}[\text{margin}_f(\mathbf{x})] - \mathbb{E}_{\mathcal{D}_t}[\text{margin}_f(\mathbf{x})]$$

JAN 대비 더 강력한 이론적 보장을 제공한다.

#### 4.2.3 SHOT (Source Hypothesis Transfer)

**논문:** Liang et al., "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation," ICML 2020

소스 데이터 없이 소스 모델만으로 적응하는 **소스-무관(source-free)** 도메인 적응:

$$\min_\theta -\frac{1}{n_t}\sum_{i=1}^{n_t} \sum_{k=1}^{K} \hat{p}_{ik} \log \hat{p}_{ik} + \lambda \|h(\mathbf{x}^t; \theta) - \bar{h}\|^2$$

**JAN 대비 장점:** 프라이버시 보호 및 소스 데이터 미보유 시나리오 대응

#### 4.2.4 TVT (Transferable Vision Transformer)

**논문:** Yang et al., "TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation," WACV 2023

Vision Transformer(ViT) 기반 도메인 적응:
- 셀프 어텐션 메커니즘이 자연스럽게 전역 특징 정렬을 지원
- JAN의 CNN 기반 접근을 Transformer로 확장

#### 4.2.5 최적 수송(Optimal Transport) 기반 방법

**논문:** Damodaran et al., "DeepJDOT: Deep Joint Distribution Optimal Transport for Unsupervised Domain Adaptation," ECCV 2018

$$\mathcal{W}_c(P, Q) = \min_{\gamma \in \Pi(P,Q)} \int_{\mathcal{X} \times \mathcal{X}} c(\mathbf{x}, \mathbf{x}') \, d\gamma(\mathbf{x}, \mathbf{x}')$$

Wasserstein 거리를 활용하여 결합 분포를 더 직접적으로 정렬하며, JAN의 MMD 기반 정렬의 이론적 약점을 보완한다.

#### 4.2.6 비교 요약표

| 방법 | 연도 | 분포 정렬 방식 | 레이블 활용 | 이론적 보장 | 비고 |
|---|---|---|---|---|---|
| **JAN** | 2017 | JMMD (커널, 결합) | 간접 | 중간 | 결합 분포 선구 |
| CDAN | 2018 | 적대적 (결합) | 소프트맥스 예측 | 중간 | 클래스 조건부 |
| MDD | 2019 | 마진 불일치 | 예측 활용 | 강함 | 이론 강화 |
| DeepJDOT | 2018 | Wasserstein OT | 결합 레이블 | 강함 | 기하학적 |
| SHOT | 2020 | 엔트로피 + 다양성 | 의사레이블 | 중간 | 소스 불필요 |
| TVT | 2023 | 어텐션 기반 | 없음 | 약함 | ViT 기반 |

---

### 4.3 향후 연구 시 고려할 점

**① 더 강력한 분포 불일치 척도 선택**

JMMD의 커널 선택 문제를 극복하기 위해:
- **Wasserstein 거리** 기반 방법론 탐구
- **스펙트럼 정규화** 또는 **립시츠 제약** 도입
- **학습 가능한 커널** (deep kernel) 설계

**② 조건부 분포 정렬 강화**

타겟 레이블 부재 문제를 극복하기 위해:
- **의사 레이블(pseudo-labeling)** + 자기 훈련(self-training) 결합
- **프로토타입 기반** 클래스 중심 정렬
- **일관성 정규화(consistency regularization)** 적용

**③ 소스-무관 도메인 적응**

데이터 프라이버시 규제(GDPR 등) 대응을 위한 소스 데이터 없는 적응 연구가 중요하다.

**④ 대규모 모델과의 통합**

- **Vision Transformer, CLIP** 등 대규모 사전 학습 모델에서의 도메인 적응 전략 재설계
- JMMD를 어텐션 레이어에 적용하는 방법론 탐구

**⑤ 다중 소스 및 다중 타겟 도메인 확장**

현실에서는 단일 소스→단일 타겟이 아닌 **다대다 도메인 시나리오**가 일반적이다. JMMD의 텐서 곱 구조를 다중 도메인으로 확장하는 연구가 필요하다.

**⑥ 분포 시프트 유형 진단 체계**

Covariate shift, conditional shift, dataset shift 중 어떤 유형이 지배적인지 자동으로 진단하고 적응 전략을 선택하는 **메타 학습 기반** 접근법이 유망하다.

**⑦ 강건성 및 적대적 공격 내성**

도메인 적응 모델이 적대적 예제(adversarial examples)에 취약할 수 있으므로, 강건 도메인 적응(robust domain adaptation) 연구가 필요하다.

---

## 참고 자료

**주요 논문 (본문에서 직접 인용):**

1. **Long, M., Zhu, H., Wang, J., & Jordan, M. I. (2017).** "Deep Transfer Learning with Joint Adaptation Networks." *Proceedings of the 34th International Conference on Machine Learning (ICML 2017)*, PMLR 70. *(본 분석의 주 대상 논문)*

2. **Gretton, A., Borgwardt, K., Rasch, M., Schölkopf, B., & Smola, A. (2012).** "A kernel two-sample test." *Journal of Machine Learning Research (JMLR)*, 13:723–773.

3. **Ganin, Y. & Lempitsky, V. (2015).** "Unsupervised domain adaptation by backpropagation." *ICML 2015*.

4. **Long, M., Cao, Y., Wang, J., & Jordan, M. I. (2015).** "Learning transferable features with deep adaptation networks." *ICML 2015*.

5. **Ben-David, S. et al. (2010).** "A theory of learning from different domains." *Machine Learning*, 79(1-2):151–175.

6. **Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014).** "How transferable are features in deep neural networks?" *NeurIPS 2014*.

7. **Song, L., Huang, J., Smola, A., & Fukumizu, K. (2009).** "Hilbert space embeddings of conditional distributions." *ICML 2009*.

8. **Arjovsky, M., Chintala, S., & Bottou, L. (2017).** "Wasserstein GAN." *arXiv:1701.07875*.

**2020년 이후 비교 분석에 참고한 논문:**

9. **Long, M. et al. (2018).** "Conditional Adversarial Domain Adaptation." *NeurIPS 2018*.

10. **Zhang, Y. et al. (2019).** "Bridging Theory and Algorithm for Domain Adaptation." *ICML 2019*.

11. **Liang, J. et al. (2020).** "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation." *ICML 2020*.

12. **Damodaran, B. B. et al. (2018).** "DeepJDOT: Deep Joint Distribution Optimal Transport for Unsupervised Domain Adaptation." *ECCV 2018*.

13. **Yang, J. et al. (2023).** "TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation." *WACV 2023*.

> **⚠️ 정확도 주의:** 2020년 이후 비교 분석에 인용된 논문들(9–13번)의 세부 수치와 수식은 본 문서에 첨부된 JAN 원문 PDF 외의 외부 자료를 참조한 것으로, 일부 세부 내용은 해당 원문을 직접 확인하여 검증하실 것을 권장합니다. JAN 논문 자체의 내용(1–8번)은 첨부된 원문 PDF를 기반으로 작성되었습니다.
