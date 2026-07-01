# Unsupervised Domain Adaptation with Residual Transfer Networks (RTN) 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

RTN 논문의 가장 핵심적인 주장은 다음 두 가지입니다:

> **"소스 도메인과 타겟 도메인의 분류기(classifier)는 동일하지 않으며, 그 차이를 잔차 함수(residual function)로 모델링할 수 있다."**

기존 심층 도메인 적응 방법들(DAN, RevGrad 등)은 **공유 분류기 가정(shared-classifier assumption)**, 즉 특징 공간을 정렬하면 소스 분류기를 타겟에 그대로 사용할 수 있다는 가정에 의존했습니다. RTN은 이 가정이 현실적으로 성립하지 않을 수 있음을 지적하고, **분류기 적응(classifier adaptation)** 과 **특징 적응(feature adaptation)** 을 동시에 수행하는 통합 프레임워크를 제안합니다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **잔차 분류기 적응** | 소스-타겟 분류기 간 차이를 잔차 블록으로 명시적 모델링 |
| **텐서 MMD** | 다중 레이어 특징을 텐서 곱으로 융합한 후 단일 MMD 페널티로 정렬 |
| **엔트로피 최소화** | 타겟 도메인의 저밀도 경계 구조를 활용한 분류기 정제 |
| **단대단(end-to-end) 학습** | 위 세 가지를 하나의 손실 함수로 통합하여 역전파로 훈련 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)** 환경에서:

- 소스 도메인 $\mathcal{D}_s = \{(\mathbf{x}_i^s, y_i^s)\}\_{i=1}^{n_s}$: 레이블이 있는 데이터
- 타겟 도메인 $\mathcal{D}_t = \{\mathbf{x}_j^t\}\_{j=1}^{n_t}$: 레이블이 없는 데이터
- 두 도메인의 분포 $p \neq q$

목표는 타겟 리스크를 최소화하는 분류기를 학습하는 것입니다:

$$R_t(f_t) = \mathbb{E}_{(\mathbf{x},y)\sim q}[f_t(\mathbf{x}) \neq y]$$

**문제의 핵심 이중성:**

$$p(\mathbf{x}) \neq q(\mathbf{x}) \quad \text{(특징 불일치)} \quad \text{AND} \quad f_s(\mathbf{x}) \neq f_t(\mathbf{x}) \quad \text{(분류기 불일치)}$$

Ben-David et al. (2010) [22]의 이론에 따르면, 이상적 결합 가설의 결합 오차(combined error)가 크면 단일 분류기로 소스와 타겟 모두를 잘 처리할 수 없습니다. 기존 방법들은 특징 정렬만 수행하고 분류기 불일치를 무시했다는 것이 핵심 문제입니다.

---

### 2.2 제안 방법 (수식 포함)

#### (A) 소스 도메인 학습 손실

$$\min_{f_s} \frac{1}{n_s} \sum_{i=1}^{n_s} L(f_s(\mathbf{x}_i^s), y_i^s) \tag{1}$$

여기서 $L(\cdot, \cdot)$은 교차 엔트로피 손실입니다.

---

#### (B) 특징 적응: 텐서 MMD (Feature Adaptation)

다중 레이어 $\ell \in \mathcal{L} = \{fcb, fcc\}$의 특징을 텐서 곱으로 융합합니다:

$$\mathbf{z}_i^s \triangleq \bigotimes_{\ell \in \mathcal{L}} \mathbf{x}_i^{s\ell}, \quad \mathbf{z}_j^t \triangleq \bigotimes_{\ell \in \mathcal{L}} \mathbf{x}_j^{t\ell}$$

융합된 특징에 대해 MMD를 최소화합니다:

$$\min_{f_s, f_t} D_{\mathcal{L}}(\mathcal{D}_s, \mathcal{D}_t) = \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} \frac{k(\mathbf{z}_i^s, \mathbf{z}_j^s)}{n_s^2} + \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} \frac{k(\mathbf{z}_i^t, \mathbf{z}_j^t)}{n_t^2} - 2\sum_{i=1}^{n_s}\sum_{j=1}^{n_t} \frac{k(\mathbf{z}_i^s, \mathbf{z}_j^t)}{n_s n_t} \tag{2}$$

여기서 커널 함수는 가우시안 커널입니다:

$$k(\mathbf{z}, \mathbf{z}') = e^{-\|\text{vec}(\mathbf{z}) - \text{vec}(\mathbf{z}')\|^2 / b}$$

> **DAN과의 차이점:** DAN은 $|\mathcal{L}|$개의 독립적인 MMD 페널티를 사용하지만, RTN은 텐서 곱으로 특징을 융합한 뒤 **단 하나의 MMD 페널티**를 사용하여 레이어 간 상호작용을 포착하고 하이퍼파라미터 수를 줄입니다.

---

#### (C) 분류기 적응: 잔차 전이 블록 (Classifier Adaptation)

ResNet의 잔차 학습 아이디어를 분류기 적응에 적용합니다:

$$f_S(\mathbf{x}) = f_T(\mathbf{x}) + \Delta f(\mathbf{x}) \tag{3}$$

여기서:
- $f_T(\mathbf{x})$: 타겟 분류기 출력 (fc 레이어, softmax 이전)
- $\Delta f(\mathbf{x})$: 잔차 함수 (fc1–fc2 레이어로 학습)
- $f_S(\mathbf{x})$: 소스 분류기 출력 (element-wise 덧셈 후)
- 최종 확률: $f_s(\mathbf{x}) \triangleq \sigma(f_S(\mathbf{x}))$, $f_t(\mathbf{x}) \triangleq \sigma(f_T(\mathbf{x}))$

**설계 핵심:** $f_S$를 잔차 블록의 출력으로 설정하여 소스 레이블 데이터로 역전파가 가능하게 합니다. $f_T$를 출력으로 설정하면 타겟 레이블이 없어 학습이 불가능합니다.

심층 잔차 학습 조건: $|\Delta f(\mathbf{x})| \ll |f_T(\mathbf{x})| \approx |f_S(\mathbf{x})|$

---

#### (D) 엔트로피 최소화 (Entropy Minimization)

타겟 분류기가 타겟 데이터의 저밀도 영역을 통과하도록 엔트로피를 최소화합니다:

$$\min_{f_t} \frac{1}{n_t} \sum_{i=1}^{n_t} H\left(f_t\left(\mathbf{x}_i^t\right)\right) \tag{4}$$

$$H(f_t(\mathbf{x}_i^t)) = -\sum_{j=1}^{c} f_j^t(\mathbf{x}_i^t) \log f_j^t(\mathbf{x}_i^t)$$

여기서 $c$는 클래스 수, $f_j^t(\mathbf{x}_i^t)$는 $\mathbf{x}_i^t$가 클래스 $j$에 속할 확률입니다.

---

#### (E) 통합 목적 함수: RTN

$$\min_{f_S = f_T + \Delta f} \underbrace{\frac{1}{n_s} \sum_{i=1}^{n_s} L(f_s(\mathbf{x}_i^s), y_i^s)}_{\text{소스 분류 손실}} + \underbrace{\frac{\gamma}{n_t} \sum_{i=1}^{n_t} H\left(f_t\left(\mathbf{x}_i^t\right)\right)}_{\text{엔트로피 페널티}} + \underbrace{\lambda \, D_{\mathcal{L}}(\mathcal{D}_s, \mathcal{D}_t)}_{\text{텐서 MMD 페널티}} \tag{5}$$

여기서 $\lambda$와 $\gamma$는 각각 텐서 MMD와 엔트로피 페널티의 균형 파라미터입니다.

---

### 2.3 모델 구조

```
입력(Xs, Xt)
    ↓
[AlexNet/ResNet 백본] (conv layers: fine-tuning)
    ↓
[fcb: 병목 레이어] ──────────────────────────────┐
    ↓                                              │
[fcc: 분류기 레이어] → fT(x)                      │
    ↓              ↘                               │
[fc1 → fc2]         ────→ fS(x) = fT(x) + Δf(x)  │
  (잔차 블록)              ↓                       │
                      소스 손실 L(fS, ys)         │
                                                   ↓
                      ←── Tensor MMD(fcb⊗fcc) ────┘
                      엔트로피 최소화 H(ft(xt))
```

**주요 구성 요소:**

| 레이어 | 역할 |
|---|---|
| conv1~5 (AlexNet) | 범용 특징 추출 (fine-tuning) |
| fcb (병목 레이어) | 특징 차원 축소 + 적응 |
| fcc (분류기 레이어) | 타겟 분류기 $f_T(\mathbf{x})$ |
| fc1–fc2 (잔차 레이어) | 잔차 함수 $\Delta f(\mathbf{x})$ 학습 ($c \times c$ 유닛) |
| Tensor MMD 모듈 | fcb⊗fcc 특징 분포 정렬 |
| 엔트로피 모듈 | 타겟 분류기 저밀도 분리 강화 |

---

### 2.4 성능 향상 및 한계

#### 성능 결과

**Office-31 벤치마크 (평균 정확도):**

| 방법 | A→W | D→W | W→D | A→D | D→A | W→A | **Avg** |
|---|---|---|---|---|---|---|---|
| AlexNet | 60.6 | 95.4 | 99.0 | 64.2 | 45.5 | 48.3 | 68.8 |
| DAN | 68.5 | 96.0 | 99.0 | 66.8 | 50.0 | 49.8 | 71.7 |
| RevGrad | 73.0 | 96.4 | 99.2 | - | - | - | - |
| **RTN (full)** | **73.3** | **96.8** | **99.6** | **71.0** | **50.5** | **51.0** | **73.7** |

**Office-Caltech 벤치마크 (평균 정확도):**

| 방법 | **Avg** |
|---|---|
| DAN | 90.1 |
| **RTN (full)** | **93.4** |

**에이블레이션 연구 (Office-31 Avg):**

| 변형 | Avg |
|---|---|
| RTN (mmd) | 72.1 |
| RTN (mmd+ent) | 72.9 |
| RTN (mmd+ent+res) | **73.7** |

#### 한계점

1. **AlexNet 기반의 제한:** 논문의 실험이 주로 AlexNet을 백본으로 사용하여 더 강력한 백본(ResNet-50 등)에서의 성능이 직접적으로 검증되지 않음

2. **클래스 불균형 미처리:** 소스-타겟 간 클래스 분포 차이(label shift)를 명시적으로 다루지 않음

3. **소규모 벤치마크:** Office-31, Office-Caltech 등 상대적으로 소규모 데이터셋에서만 검증

4. **하이퍼파라미터 민감성:** $\lambda$, $\gamma$, 대역폭 $b$ 등 여러 하이퍼파라미터 조정이 필요하며, 이를 타겟 레이블 없이 선택하는 것이 어려움

5. **엔트로피-잔차 상호 의존성:** 실험 결과에서 엔트로피 최소화와 잔차 모듈을 단독으로 사용하면 잔차 함수가 영 매핑(zero mapping)을 학습하는 경향이 있어 반드시 함께 사용해야 함

6. **계산 복잡도:** 텐서 MMD는 $O(n^2)$ 복잡도를 가지며, 선형 시간 근사가 필요함

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상의 이론적 근거

Ben-David et al. (2010)의 이론에 따르면 타겟 도메인의 리스크 상한은:

$$R_t(f_t) \leq R_s(f_s) + d_{\mathcal{H}\Delta\mathcal{H}}(p, q) + \lambda^*$$

여기서 $\lambda^\* = R_s(h^\*) + R_t(h^*)$는 이상적 결합 가설의 결합 오차입니다.

RTN은 이 세 항을 동시에 줄이는 방향으로 설계되었습니다:

| 리스크 항 | RTN의 대응 메커니즘 |
|---|---|
| $R_s(f_s)$ | 소스 분류 손실 최소화 (수식 1) |
| $d_{\mathcal{H}\Delta\mathcal{H}}(p, q)$ | 텐서 MMD로 분포 정렬 (수식 2) |
| $\lambda^*$ | 잔차 블록으로 분류기 간 간극 축소 (수식 3) |

### 3.2 일반화를 높이는 핵심 설계 원리

**① 잔차 분류기의 정규화 효과**

잔차 학습의 조건 $|\Delta f(\mathbf{x})| \ll |f_T(\mathbf{x})|$은 타겟 분류기가 소스 분류기에서 크게 벗어나지 않도록 암묵적 정규화를 제공합니다. 이는 타겟 도메인에서의 과적합(overfitting)을 방지하고 일반화를 촉진합니다.

**② 텐서 MMD의 고차 통계량 포착**

단순 평균 임베딩 매칭(1차 통계량)을 넘어, 텐서 곱 $\mathbf{z} = \bigotimes_{\ell \in \mathcal{L}} \mathbf{x}^\ell$은 레이어 간 고차 상호작용을 포착합니다. RKHS(재생 커널 힐베르트 공간)에서의 매칭은 분포 간의 더 세밀한 차이를 줄여 도메인 불변 표현 학습을 강화합니다.

**③ 엔트로피 최소화의 반지도 학습 효과**

엔트로피 최소화는 타겟 데이터의 클러스터 구조를 활용하여 분류 경계가 저밀도 영역을 통과하도록 합니다. 이는 레이블 없는 타겟 데이터를 효과적으로 활용하는 반지도 학습(semi-supervised learning) 효과를 제공합니다.

**④ 다중 레이어 적응**

특징 적응을 $\mathcal{L} = \{fcb, fcc\}$ 두 레이어에서 동시에 수행함으로써, 단일 레이어 적응보다 더 강건한 도메인 불변 표현을 학습합니다.

### 3.3 일반화 성능의 실험적 근거

- **어려운 전이 태스크에서의 성능 향상이 두드러짐:** $A \rightarrow W$ (60.6% → 73.3%), $C \rightarrow W$ 등 소스-타겟 분포 차이가 큰 태스크에서 기존 방법 대비 큰 폭의 향상
- **쉬운 전이 태스크에서의 안정성:** $D \rightarrow W$, $W \rightarrow D$ 같이 이미 성능이 높은 태스크에서도 성능 저하 없음
- **t-SNE 시각화:** RTN의 타겟 도메인 예측이 DAN보다 클래스 간 거리가 더 크게 나타남

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 RTN이 이후 연구에 미친 영향

**① 분류기 적응의 중요성 인식 확산**

RTN은 특징 정렬만으로는 충분하지 않음을 실증적으로 보여줌으로써, 이후 연구들이 **분류기 수준의 적응**을 표준 구성 요소로 포함하는 방향으로 발전하는 데 기여했습니다.

**② 잔차 학습의 도메인 적응 적용 가능성 제시**

컴퓨터 비전의 ResNet 아이디어를 도메인 적응의 분류기 간 관계 모델링에 창의적으로 적용한 것은 이후 다양한 잔차 기반 전이학습 연구의 선례가 되었습니다.

**③ 엔트로피 최소화의 재조명**

Grandvalet & Bengio (2004)의 엔트로피 최소화 원리를 UDA에 통합하는 방식은 이후 셀프 트레이닝(self-training) 및 의사 레이블(pseudo-label) 기반 방법들의 발전에 영향을 주었습니다.

**④ 다중 손실 통합 프레임워크의 표준화**

분류 손실 + 분포 정렬 손실 + 엔트로피 손실을 하나의 목적 함수로 통합하는 방식은 이후 UDA 연구의 설계 패턴으로 자리잡았습니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 후속 연구

#### (1) CDAN (Conditional Domain Adversarial Networks, NeurIPS 2018)
Long et al.의 후속 연구로, 조건부(conditional) 분포 정렬을 통해 클래스 정보를 도메인 적응에 통합합니다. RTN의 특징-분류기 상호작용 아이디어를 발전시켜 다중선형 조건부(multilinear conditioning)를 사용합니다.

$$\min_G \max_D \mathbb{E}[\log D(\mathbf{h} \otimes \hat{\mathbf{y}})] + \mathbb{E}[\log(1 - D(\mathbf{h}' \otimes \hat{\mathbf{y}}'))]$$

#### (2) MDD (Margin Disparity Discrepancy, ICML 2019)
Zhang et al.이 제안한 이론적으로 강건한 도메인 적응 방법으로, Ben-David 이론의 $\mathcal{H}\Delta\mathcal{H}$ 분산을 정밀하게 최소화합니다.

#### (3) SHOT (Shot: Source Hypothesis Transfer, ICML 2020)
Liang et al.의 연구로, **소스 모델 없이** 타겟 도메인만으로 적응하는 소스 없는 도메인 적응(source-free DA)을 제안합니다. RTN의 엔트로피 최소화 아이디어를 발전시켜 정보 극대화(information maximization)를 사용합니다:

$$\min_{f_t} \frac{1}{n_t}\sum_{i=1}^{n_t} H(f_t(\mathbf{x}_i^t)) - H\left(\frac{1}{n_t}\sum_{i=1}^{n_t} f_t(\mathbf{x}_i^t)\right)$$

#### (4) NWD/ATDOC (Neighborhood Invariance, 2020–2021)
타겟 데이터의 이웃 구조를 활용하여 의사 레이블을 생성하고 분류기를 적응하는 방법들로, RTN의 분류기 적응 개념을 비지도 방식으로 확장합니다.

#### (5) CDTrans (Cross-Domain Transformer, ICLR 2022)
Xu et al.이 제안한 트랜스포머 기반 도메인 적응 방법으로, 소스-타겟 간 패치 수준의 대응 관계를 학습합니다.

#### (6) PMTrans (Patch Mix Transformer, ECCV 2022)
비전 트랜스포머(ViT)를 백본으로 사용하며, 패치 믹스업을 통해 중간 도메인을 생성하여 분포 간극을 메웁니다.

#### (7) SPA (Source-Free Domain Adaptation, 2022–2023)
소스 데이터 접근 없이 사전 훈련된 소스 모델만으로 타겟에 적응하는 프라이버시 보존 도메인 적응 연구 분야입니다.

### 5.2 RTN 대비 방법론적 비교표

| 측면 | RTN (2016) | CDAN (2018) | SHOT (2020) | CDTrans (2022) |
|---|---|---|---|---|
| **백본** | AlexNet | ResNet-50 | ResNet-50/101 | ViT |
| **분류기 적응** | ✅ 잔차 블록 | ❌ (암묵적) | ✅ 프로토타입 | ❌ |
| **특징 정렬** | ✅ 텐서 MMD | ✅ 조건부 적대적 | ❌ | ✅ 크로스 어텐션 |
| **엔트로피 활용** | ✅ | ❌ | ✅ (정보 극대화) | ❌ |
| **소스 필요** | ✅ | ✅ | ❌ | ✅ |
| **Office-31 Avg** | 73.7% | 82.7% | 88.6% | 90.7% |

> ⚠️ **주의:** CDTrans, SHOT 등의 Office-31 수치는 해당 논문에서 보고된 값이며, 백본 아키텍처(ViT vs AlexNet)가 다르므로 직접 비교 시 주의가 필요합니다.

### 5.3 RTN의 한계와 최신 연구의 극복 방향

| RTN의 한계 | 최신 연구의 해결 방향 |
|---|---|
| AlexNet 백본의 성능 제약 | ResNet, ViT 등 강력한 백본 도입 (CDAN, CDTrans) |
| 소스 데이터 필요 | 소스 없는 도메인 적응 (SHOT, 2020) |
| 클래스 불균형 미처리 | 클래스 조건부 정렬 (CDAN, MCC) |
| MMD의 커널 선택 문제 | 적대적 훈련으로 대체 (DANN, CDAN) |
| 소규모 벤치마크 | DomainNet (345 클래스, 6 도메인) 등 대규모 벤치마크 도입 |

---

## 6. 향후 연구 시 고려할 점

### 6.1 방법론적 고려사항

**① 더 강력한 분류기 적응 방법 탐색**

RTN의 잔차 분류기 적응은 선형적 연결에 기반합니다. 비선형 분류기 적응(예: 어텐션 기반의 적응적 분류기)이나, 메타 학습(meta-learning)을 활용한 빠른 분류기 적응이 유망한 연구 방향입니다.

**② 소스 없는 도메인 적응(Source-Free DA)으로의 확장**

실제 응용에서 소스 데이터의 프라이버시 문제나 저장 제약이 있을 수 있습니다. RTN의 엔트로피 최소화 구성 요소는 소스 없는 설정에서도 활용 가능하며, 이를 발전시키는 연구가 필요합니다.

**③ 클래스 수준(class-level) 정렬 강화**

RTN의 텐서 MMD는 **주변 분포(marginal distribution)** 정렬에 초점을 맞춥니다. 클래스 조건부 분포 $p(x|y)$까지 정렬하는 조건부 도메인 적응 연구(예: CDAN)가 더 정밀한 적응을 가능하게 합니다.

**④ 대규모 데이터셋 및 다중 소스/타겟 설정**

Office-31 수준을 넘어 DomainNet, VisDA-2017 같은 대규모 벤치마크, 그리고 다중 소스 도메인(multi-source DA) 또는 다중 타겟 도메인(multi-target DA) 설정에서의 확장이 필요합니다.

**⑤ 트랜스포머 아키텍처와의 통합**

ViT 기반 모델에서 RTN의 잔차 분류기 적응 개념을 재해석하거나, 크로스 어텐션 메커니즘을 활용하여 분류기 간 관계를 더 세밀하게 모델링하는 것이 가능합니다.

### 6.2 이론적 고려사항

**① 적응 복잡도와 표본 복잡도 분석**

잔차 분류기 블록의 표현 능력과 필요한 소스 데이터 양 사이의 관계에 대한 이론적 보장이 부족합니다. PAC-Bayes 이론이나 Rademacher 복잡도를 활용한 일반화 경계 분석이 필요합니다.

**② 개방형 집합(Open-Set) 및 부분(Partial) 도메인 적응**

RTN은 소스와 타겟의 클래스 공간이 동일하다고 가정합니다. 타겟 도메인에 새로운 클래스가 존재하거나(open-set DA) 클래스 공간이 일부만 겹치는(partial DA) 현실적 시나리오에 대한 확장이 필요합니다.

**③ 공정성(Fairness)과 분포 이동의 관계**

도메인 이동이 특정 인구통계 그룹에 불균등한 영향을 미칠 수 있습니다. 공정성 제약 조건 하에서의 도메인 적응 연구가 점점 중요해지고 있습니다.

---

## 참고 자료

**본 논문 (주 참고 자료):**
- Long, M., Zhu, H., Wang, J., & Jordan, M. I. (2016). **Unsupervised Domain Adaptation with Residual Transfer Networks**. *Advances in Neural Information Processing Systems (NIPS 2016)*. [제공된 PDF]

**논문 내 인용 참고 자료:**
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*. [논문 내 참고문헌 8]
- Long, M., Cao, Y., Wang, J., & Jordan, M. I. (2015). Learning transferable features with deep adaptation networks. *ICML 2015*. [논문 내 참고문헌 5]
- Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. *ICML 2015*. [논문 내 참고문헌 6]
- Ben-David, S., et al. (2010). A theory of learning from different domains. *Machine Learning Journal*. [논문 내 참고문헌 22]
- Grandvalet, Y., & Bengio, Y. (2004). Semi-supervised learning by entropy minimization. *NIPS 2004*. [논문 내 참고문헌 28]
- Gretton, A., et al. (2012). A kernel two-sample test. *JMLR*. [논문 내 참고문헌 27]

**2020년 이후 비교 연구 (일반적으로 알려진 연구들):**
- Liang, J., et al. (2020). Do we really need to access the source data? Source hypothesis transfer for unsupervised domain adaptation. *ICML 2020*. (SHOT)
- Long, M., et al. (2018). Conditional adversarial domain adaptation. *NeurIPS 2018*. (CDAN)
- Xu, T., et al. (2022). CDTrans: Cross-domain transformer for unsupervised domain adaptation. *ICLR 2022*.

> ⚠️ **정확도 관련 주의사항:** 2020년 이후 최신 연구들의 수치(Office-31 정확도 등)는 해당 논문에서 보고된 값을 참조했으나, 제가 직접 PDF를 검토한 자료는 제공된 RTN 논문에 한합니다. 최신 연구 비교 수치는 각 해당 논문을 직접 확인하시기 바랍니다.
