# Collaborative and Adversarial Network (CAN) for Unsupervised Domain Adaptation 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
기존 도메인 적응 방법(DANN 등)은 **오직 도메인 비정보적(domain-uninformative) 표현만을 학습**하여, 타겟 도메인의 특성 정보가 손실되는 문제가 있다. CAN은 **낮은 블록에서는 도메인 정보적(domain-informative) 표현을, 높은 블록에서는 도메인 비정보적 표현을 동시에 학습**함으로써, 도메인 불변성과 분류 판별력을 함께 확보한다.

### 주요 기여
| 기여 | 설명 |
|------|------|
| **CAN** | 다중 블록에 도메인 분류기를 부착하여 협력적(collaborative) + 적대적(adversarial) 학습을 통합 |
| **iCAN** | 이미지 분류기와 도메인 분류기를 함께 활용한 의사 레이블(pseudo-label) 반복 선택으로 학습 데이터 확장 |
| **적응적 임계값** | 소스 도메인 분류 정확도에 기반한 동적 임계값 설정 방식 제안 |
| **가중치 자동 최적화** | 각 블록의 손실 가중치 $\lambda_k$를 자동으로 학습하여 하이퍼파라미터 수 감소 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**도메인 편이(Domain Shift) 문제**: 소스 도메인과 타겟 도메인의 데이터 분포 불일치로 인해 소스에서 학습된 모델이 타겟에서 성능이 저하됨.

기존 DANN의 문제점:
- 마지막 블록 하나에만 도메인 분류기를 부착 → **낮은 층의 도메인 특이적 정보(모서리, 엣지 등) 손실**
- 도메인 비정보적 표현만 학습 → **타겟 도메인 고유 특성 정보 소실**
- 비지도 환경에서 타겟 레이블 미활용

### 2.2 제안하는 방법 (수식 포함)

#### (A) 도메인 정보적 특징 학습 (Domain Informative Feature Learning)

이미지 $\mathbf{x}$에 대해 블록 $F$의 출력 $\mathbf{f} = F(\mathbf{x}; \boldsymbol{\theta})$를 도메인 분류기 $D$가 구분하도록 학습:

$$\min_{\boldsymbol{\theta}, \mathbf{w}} \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_D\left(D(F(\mathbf{x}_i; \boldsymbol{\theta}); \mathbf{w}), d_i\right) \tag{1}$$

- $d_i \in \{0, 1\}$: 도메인 레이블 (0=소스, 1=타겟)
- $\mathcal{L}_D$: 교차 엔트로피 손실

#### (B) 도메인 비정보적 특징 학습 (Domain Uninformative Feature Learning, DANN 방식)

$$\max_{\boldsymbol{\theta}} \min_{\mathbf{w}} \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_D\left(D(F(\mathbf{x}_i; \boldsymbol{\theta}); \mathbf{w}), d_i\right) \tag{2}$$

Gradient Reversal Layer를 통해 역전파 시 기울기의 부호를 반전하여 구현.

#### (C) CAN의 협력-적대 통합 학습 (핵심 수식)

$m$개의 블록 각각에 도메인 분류기를 부착하고, 가중치 $\lambda_k$를 자동 학습:

$$\min_{\Theta_F, \boldsymbol{\lambda}} \mathcal{L}_{CAN} = \sum_{k=1}^{m-1} \lambda_k \min_{\mathbf{w}_k} \mathcal{L}_D(\boldsymbol{\theta}_k, \mathbf{w}_k) + \lambda_m \min_{\mathbf{w}_m} \mathcal{L}_D(\boldsymbol{\theta}_m, \mathbf{w}_m) \tag{3}$$

$$\text{s.t.} \quad \sum_{k=1}^{m-1} \lambda_k = \lambda_0, \quad |\lambda_k| \leq \lambda_0$$

- $\lambda_k \geq 0$: 해당 블록에서 **도메인 정보적** 표현 학습 (협력적 학습)
- $\lambda_k < 0$ (특히 마지막 블록 $\lambda_m$): **도메인 비정보적** 표현 학습 (적대적 학습)

#### (D) 소스 분류 손실

$$\mathcal{L}_{src} = \frac{1}{N_s} \sum_{i=1}^{N_s} \mathcal{L}_C\left(C(F(\mathbf{x}_i^s; \Theta_F); \mathbf{c}), y_i^s\right) \tag{4}$$

#### (E) CAN 최종 목적 함수

$$\min_{\Theta_F, \mathbf{c}, \lambda_k \in \Lambda} \mathcal{L} = \mathcal{L}_{src} + \mathcal{L}_{CAN} \tag{5}$$

---

### 2.3 iCAN: 의사 레이블 기반 점진적 확장

#### 적응적 임계값 (Adaptive Threshold)

소스 분류 정확도 $A$에 기반한 동적 임계값:

$$T_C = \frac{1}{1 + e^{-\rho \cdot A}} \tag{6}$$

$$A = \frac{1}{N_s} \sum_{i=1}^{N_s} I\left(y_i^s, \arg\max_c p_c(\mathbf{x}_i^s)\right), \quad I(a,b) = \begin{cases} 1, & \text{if } a = b \\ 0, & \text{otherwise} \end{cases} \tag{7}$$

#### 샘플 선택 함수

$$s(\mathbf{x}_i^t) = \sigma(p_{\tilde{y}_i^t}(\mathbf{x}_i^t),\, T_C), \quad \sigma(a,b) = \begin{cases} 1, & \text{if } a > b \\ 0, & \text{otherwise} \end{cases} \tag{8}$$

#### 도메인 분류기 기반 가중치 함수

도메인 분류기 출력 $d(\mathbf{x}_i^t)$가 0.5에 가까울수록 높은 가중치 부여:

$$h(\mathbf{x}_i^t; z) = -|z \cdot (d(\mathbf{x}_i^t) - 0.5)|^\alpha + 1 \tag{9}$$

$$w(\mathbf{x}_i^t; z) = \beta \cdot \sigma(h(\mathbf{x}_i^t; z), 0) + \max(h(\mathbf{x}_i^t; z), 0) \tag{10}$$

#### 의사 레이블 타겟 분류 손실

$$\mathcal{L}_{tar} = \frac{1}{N_t} \sum_{i=1}^{N_t} s(\mathbf{x}_i^t) \cdot w(\mathbf{x}_i^t; z) \cdot \mathcal{L}_C\left(C(F(\mathbf{x}_i^t; \Theta_F); \mathbf{c}), \tilde{y}_i^t\right) \tag{11}$$

#### iCAN 최종 목적 함수

$$\min_{\Theta_F, \mathbf{c}, z, \lambda_k \in \Lambda} \mathcal{L}_{total} = \mathcal{L}_{src} + \mathcal{L}_{tar} + \mathcal{L}_{CAN} \tag{12}$$

---

### 2.4 모델 구조

```
입력 이미지
    │
    ├── Feature Extraction Block 1 (F₁, θ₁)
    │       └── Domain Classifier D₁ [λ₁ > 0: 협력적 학습]
    │
    ├── Feature Extraction Block 2 (F₂, θ₂)
    │       └── Domain Classifier D₂ [λ₂ > 0: 협력적 학습]
    │
    ├── ...
    │
    └── Feature Extraction Block m (Fₘ, θₘ)
            ├── Domain Classifier Dₘ [λₘ < 0: 적대적 학습, Rev. Grad.]
            └── Image Classifier C → 분류 손실 ℒ_C
```

- **백본**: ResNet50 (ImageNet 사전학습)
- **블록 구성**: ResNet50을 4개 블록으로 분할 (10번째, 22번째, 40번째, 49번째 레이어 이후에 도메인 분류기 부착)
- **도메인 분류기**: FC 레이어로 구성
- **최적화**: SGD, 학습률 $\eta_0 = 0.0015$, INV 스케줄

---

### 2.5 성능 향상

**Office-31 데이터셋 결과**:

| 방법 | A→W | W→A | A→D | D→A | W→D | D→W | **평균** |
|------|-----|-----|-----|-----|-----|-----|---------|
| ResNet50 | 73.5 | 59.8 | 76.5 | 56.7 | 99.0 | 93.6 | 76.5 |
| DANN | 79.3 | 63.2 | 80.7 | 65.3 | 99.6 | 97.3 | 80.9 |
| JAN | 86.0 | 70.7 | 85.1 | 69.2 | 99.7 | 96.7 | 84.6 |
| **CAN** | 81.5 | 63.4 | 85.5 | 65.9 | 99.7 | 98.2 | **82.4** |
| **iCAN** | 92.5 | 69.9 | 90.1 | 72.1 | 100.0 | 98.8 | **87.2** |

**ImageCLEF-DA 데이터셋 결과**:

| 방법 | I→P | P→I | I→C | C→I | C→P | P→C | **평균** |
|------|-----|-----|-----|-----|-----|-----|---------|
| JAN | 76.8 | 88.0 | 94.7 | 89.7 | 74.2 | 91.7 | 85.8 |
| **CAN** | 78.2 | 87.5 | 94.2 | 89.5 | 75.8 | 89.2 | **85.7** |
| **iCAN** | 79.5 | 89.7 | 94.7 | 89.9 | 78.5 | 92.0 | **87.4** |

### 2.6 한계

1. **하이퍼파라미터 민감성**: $\lambda_0$, $\lambda_m$, $\alpha$, $\beta$ 등 다수의 하이퍼파라미터 존재 (Table 3, 5 참고)
2. **의사 레이블 노이즈**: 초기 단계에서 잘못된 의사 레이블이 누적될 위험성 (Confirmation Bias)
3. **계산 비용**: 다중 도메인 분류기 및 반복적 재학습으로 인한 연산량 증가
4. **평가 데이터셋 제한**: Office-31, ImageCLEF-DA 두 벤치마크에 국한
5. **블록 수 설계**: 최적 블록 분할 방식이 아키텍처마다 다를 수 있어 범용 적용 어려움
6. **이진 도메인 가정**: 소스-타겟 이진 분류 기반으로, 다중 소스 도메인 확장에 한계

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 다층 표현 학습을 통한 일반화

CAN의 핵심 철학은 **계층적 표현의 역할 분리**에 있다:

- **낮은 블록** (협력적 학습, $\lambda_k > 0$): 모서리, 텍스처 등 도메인 고유 특징 보존 → **클래스 판별력 유지**
- **높은 블록** (적대적 학습, $\lambda_m < 0$): 도메인 불변 표현 학습 → **도메인 간 전이 가능성 향상**

실험에서 $\lambda_k$가 낮은 블록에서는 양수, 높은 블록에서는 음수로 자동 수렴하는 것이 확인되어, **네트워크가 자연스럽게 계층적 역할을 학습**함을 보여준다.

### 3.2 iCAN의 점진적 분포 이동 보정

$$\mathcal{D}_{train}^{(t+1)} = \mathcal{D}_s \cup \mathcal{D}_{pseudo}^{(t)}$$

반복적 의사 레이블 선택을 통해 학습 데이터의 분포가 소스에서 타겟으로 점진적으로 이동:
- **도메인 분류기 점수 0.5 근방** 샘플 우선 선택 → 이미 도메인 불변적인 샘플부터 활용
- **적응적 임계값** $T_C$로 학습 초기의 노이즈 레이블 방지

### 3.3 일반화 향상의 이론적 근거

Ben-David et al.의 도메인 적응 이론에 따르면, 타겟 오류 상한은:

$$\epsilon_T(h) \leq \epsilon_S(h) + d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda$$

- **$d_{\mathcal{H}\Delta\mathcal{H}}$** (도메인 불일치): 적대적 학습으로 감소
- **$\epsilon_S(h)$** (소스 오류): 분류 손실 최소화로 감소
- **$\lambda$** (이상적 결합 오류): 협력적 학습으로 판별력 유지 → 감소 기여

CAN은 세 항 모두를 동시에 최적화하는 구조를 갖는다.

### 3.4 다양한 아키텍처 적용 가능성

수식 (3)의 $\mathcal{L}_{CAN}$은 AlexNet, VGG, ResNet, DenseNet 등 **임의의 CNN 아키텍처에 플러그인 방식으로 통합 가능**하여 범용 일반화 가능성을 지닌다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (1) 다층 도메인 정렬 패러다임의 확립
CAN은 단일 층 정렬(DANN)에서 **계층적 다층 정렬**로의 패러다임 전환을 촉진했다. 이후 연구들이 중간 레이어 표현을 적극 활용하는 방향으로 발전하는 데 기여했다.

#### (2) 협력-적대 학습의 균형 문제 제기
낮은 층의 정보 보존 vs. 높은 층의 도메인 불변성이라는 **트레이드오프를 명시적으로 정의**함으로써, 이후 연구에서 레이어별 적응 전략 설계의 중요성을 인식시켰다.

#### (3) 도메인 분류기를 이용한 의사 레이블 선택
단순 분류 신뢰도 기반 선택에서 **도메인 분류기를 보조 기준으로 사용**하는 아이디어는 이후 자기 학습(Self-training) 기반 도메인 적응 연구에 영향을 주었다.

### 4.2 향후 연구 시 고려할 점

#### (1) Confirmation Bias 완화
의사 레이블 선택 시 초기 오류가 누적되는 문제를 완화하기 위해:
- **교사-학생(Teacher-Student)** 프레임워크 도입
- **혼합 증강(MixUp)** 기반 정규화
- **불확실성 추정(Uncertainty Estimation)** 기반 필터링

#### (2) 클래스 조건부 정렬 (Class-Conditional Alignment)
CAN은 클래스 레이블을 고려하지 않은 **주변 분포(marginal distribution)** 정렬에 집중한다. 이후 연구에서는:
- 클래스별 도메인 정렬 (Conditional Domain Adversarial Network 등)
- 클래스 프로토타입 기반 정렬 방식이 더 효과적임이 밝혀짐

#### (3) 다중 소스 도메인 확장
이진 소스-타겟 구조를 넘어, 여러 소스 도메인에서 타겟으로의 적응 문제에 CAN 프레임워크를 어떻게 확장할지 연구 필요.

#### (4) Transformer 아키텍처와의 결합
Vision Transformer(ViT)는 CNN과 달리 명확한 "블록" 계층 구조를 가지므로, **각 Transformer 블록에 도메인 분류기를 부착**하는 CAN 방식의 적용 가능성 검토가 필요.

#### (5) 공정성 및 도메인 편향
특정 도메인에 치우친 의사 레이블 선택이 **특정 클래스에 대한 편향**을 야기할 수 있으므로, 클래스 균형을 고려한 샘플링 전략이 필요.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 핵심 방법 | CAN 대비 차이점 |
|------|------|-----------|----------------|
| **CDAN** (Long et al., NeurIPS 2018) | 2018 | 클래스 조건부 도메인 정렬 (Conditional Adversarial) | 클래스-도메인 상호작용 명시적 모델링, CAN은 클래스 무관 정렬 |
| **SHOT** (Liang et al., ICML 2020) | 2020 | 소스 없이 타겟에서만 적응 (Source-Free DA) | 소스 데이터 불필요, 정보 최대화 기반, CAN은 소스 접근 필요 |
| **MDD** (Zhang et al., ICML 2019) | 2019 | 마진 불일치 기반 이론적 적응 | 이론적 상한 최소화, 단일 도메인 분류기 사용 |
| **Self-Ensemble** (French et al., ICLR 2018) | 2018 | 평균 교사 모델 기반 자기 앙상블 | 의사 레이블 노이즈에 더 강건 |
| **TransDA / TVT** (Yang et al., 2021) | 2021 | Vision Transformer 기반 도메인 적응 | 자기 주의 메커니즘으로 글로벌 컨텍스트 활용, CAN은 CNN 전용 |
| **CDTrans** (Xu et al., ICLR 2022) | 2022 | Cross-Domain Transformer | 교차 도메인 주의 메커니즘, 패치 수준 정렬 |
| **PMTrans** (Zhu et al., ECCV 2022) | 2022 | Patch-Mix Transformer for DA | 패치 혼합으로 도메인 간 중간 분포 생성 |
| **SSRT** (Sun et al., CVPR 2022) | 2022 | Safe Self-Refinement for DA | 불확실성 인식 의사 레이블, CAN의 iCAN 아이디어를 발전 |

### 핵심 트렌드 분석

```
CAN (2018) → 다층 적대적 학습
    │
    ├── 클래스 조건부 정렬 (CDAN, 2018)
    ├── 소스 프리 DA (SHOT, 2020) ← 소스 데이터 접근 불필요
    ├── Transformer 기반 (TVT, CDTrans, 2021-2022)
    └── 강건한 의사 레이블링 (SSRT, 2022)
```

**CAN의 의사 레이블 아이디어(iCAN)**는 이후 SHOT, SSRT 등의 자기 학습 기반 DA 연구로 발전했으나, **Confirmation Bias 문제**를 더 정교하게 다루는 방향으로 진화했다.

**CAN의 다층 정렬 아이디어**는 Transformer 기반 연구에서 각 Transformer 블록의 표현을 계층적으로 정렬하는 방식으로 계승되었다.

---

## 참고 자료

**원 논문**
- Zhang, W., Ouyang, W., Li, W., & Xu, D. (2018). **Collaborative and Adversarial Network for Unsupervised Domain Adaptation**. *CVPR 2018*, pp. 3801–3809.

**논문 내 인용 참고문헌 (핵심)**
- Ganin, Y. et al. (2016). **Domain-Adversarial Training of Neural Networks**. *JMLR*, 17(59):1–35. [DANN]
- Long, M. et al. (2015). **Learning Transferable Features with Deep Adaptation Networks**. *ICML*. [DAN]
- Long, M. et al. (2017). **Deep Transfer Learning with Joint Adaptation Networks**. *ICML*. [JAN]
- He, K. et al. (2016). **Deep Residual Learning for Image Recognition**. *CVPR*.

**2020년 이후 비교 연구**
- Liang, J. et al. (2020). **Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation**. *ICML 2020*. [SHOT]
- Yang, J. et al. (2021). **TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation**. *arXiv:2108.05988*.
- Xu, T. et al. (2022). **CDTrans: Cross-Domain Transformer for Unsupervised Domain Adaptation**. *ICLR 2022*.
- Sun, T. et al. (2022). **Safe Self-Refinement for Transformer-based Domain Adaptation**. *CVPR 2022*. [SSRT]

> **주의**: 2020년 이후 비교 연구 부분의 정확도(특히 구체적 수치)는 논문 원문을 직접 확인하시기를 권장합니다. 위 비교 분석은 각 논문의 핵심 아이디어 수준에서 기술되었으며, 세부 구현 및 성능 수치는 원 논문을 참조하시기 바랍니다.
