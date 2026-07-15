# Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 **비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)** 에서 두 가지 핵심 개념을 결합합니다:

1. **Task-specific decision boundary**를 활용한 특징 분포 정렬 (MCD 프레임워크 [Saito et al., CVPR 2018])
2. **Wasserstein 거리**를 기반으로 한 기하학적으로 의미 있는 불일치 측정

이 두 개념을 통합하여 **Sliced Wasserstein Discrepancy (SWD)** 를 제안하며, 이를 통해 소스 도메인의 지지(support) 밖에 있는 타겟 샘플을 효과적으로 감지하고 정렬합니다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **원칙적 방법론** | 최적 운송 이론(Optimal Transport)과 task-specific decision boundary를 결합 |
| **효율적 end-to-end 학습** | Sliced Wasserstein Discrepancy의 변분적 공식화를 통한 효율적 학습 |
| **기하학적 의미** | 분포가 겹치지 않아도 의미 있는 거리 측정 가능 |
| **범용성** | 분류, 의미론적 분할, 객체 탐지 등 다양한 태스크에 적용 가능 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

딥러닝 모델은 **도메인 이동(domain shift)** 문제로 인해 학습 도메인과 다른 도메인에서 성능이 크게 저하됩니다. 특히 이 논문은 **공변량 이동(covariate shift)** 에 집중합니다:

- **소스 도메인**: 레이블이 있는 데이터 $\{X_s, Y_s\}$
- **타겟 도메인**: 레이블이 없는 데이터 $X_t$

기존 MCD 방법 [Saito et al., 2018]의 한계:
- 불일치 손실로 $L_1$ 거리를 사용 → 두 분류기의 출력 확률 분포가 **겹칠 때만** 효과적
- 분포가 겹치지 않는 경우 기하학적으로 의미 있는 그래디언트 제공 불가

### 2.2 제안하는 방법 (수식 포함)

#### 학습 프레임워크 (3단계)

**Step 1**: 소스 도메인에서 Generator $G$와 Classifier $C_1, C_2$ 학습

$$\min_{G, C_1, C_2} \mathcal{L}_s(X_s, Y_s) $$

**Step 2**: $G$를 고정하고, 타겟 도메인에서 불일치를 **최대화**하도록 분류기 학습

$$\min_{C_1, C_2} \mathcal{L}_s(X_s, Y_s) - \mathcal{L}_{DIS}(X_t) $$

**Step 3**: 분류기를 고정하고, 타겟 도메인에서 불일치를 **최소화**하도록 $G$ 학습

$$\min_{G} \mathcal{L}_{DIS}(X_t) $$

#### 최적 운송과 Wasserstein 거리

Monge 문제: 확률 공간 $\Omega$에서 두 확률 측도 $\mu, \nu \in \mathcal{P}(\Omega)$에 대해

```math
\inf_{\mathcal{T}_\# \mu = \nu} \int_\Omega c(\mathbf{z}, \mathcal{T}(\mathbf{z})) d\mu(\mathbf{z})
```

Kantorovich 완화 (결합 분포 $\gamma$를 탐색):

$$\inf_{\gamma \in \Pi(\mu, \nu)} \int_{\Omega \times \Omega} c(\mathbf{z}_1, \mathbf{z}_2) d\gamma(\mathbf{z}_1, \mathbf{z}_2) $$

$q$-Wasserstein 거리:

$$W_q(\mu, \nu) = \left( \inf_{\gamma \in \Pi(\mu, \nu)} \int_{\Omega \times \Omega} c(\mathbf{z}_1, \mathbf{z}_2)^q d\gamma(\mathbf{z}_1, \mathbf{z}_2) \right)^{1/q} $$

#### Sliced Wasserstein Discrepancy (SWD)

1-Wasserstein 거리를 직접 최적화하는 것은 선형 프로그래밍이 필요하여 계산 비용이 높습니다. 이를 해결하기 위해 **슬라이스 변분 공식화**를 도입합니다:

$$\text{SWD}(\mu, \nu) = \int_{S^{d-1}} W_1(\mathcal{R}_\theta \mu, \mathcal{R}_\theta \nu) \, d\theta $$

여기서:
- $\mathcal{R}_\theta$: 단위 구면 $S^{d-1}$ 위의 방향 $\theta$로의 **1차원 선형 투영** 연산
- $\theta$: $\mathbb{R}^d$에서 단위 구면 $S^{d-1}$ 위의 균등 측도

이산 확률 측도에 대한 SWD 계산:

$$\text{SWD}(\mu, \nu) = \sum_{m=1}^{M} \sum_{i=1}^{N} c\left(\mathcal{R}_{\theta_m} \mu_{\alpha(i)}, \mathcal{R}_{\theta_m} \nu_{\beta(i)}\right) $$

여기서:
- $M$: 무작위 투영 수 (논문에서 $M=128$이 적합함을 실험적으로 확인)
- $\alpha, \beta$: 1D 투영값을 정렬하는 순열
- $c$: 이차 비용 함수 (quadratic cost)

**핵심 아이디어**: 1차원 최적 운송 문제는 **정렬(sorting)** 만으로 폐쇄형 해(closed-form solution)를 가지므로, 선형 프로그래밍 없이 효율적으로 계산 가능합니다.

### 2.3 모델 구조

```
입력 X
   ↓
[Feature Generator G]  ← 백본 네트워크 (CNN/ResNet/VGG)
   ↓
[p₁] ← Classifier C₁ (task-specific)
[p₂] ← Classifier C₂ (task-specific)
   ↓
SWD 계산:
  1. θ₁,...,θₘ를 S^(d-1)에서 샘플링
  2. p₁, p₂를 각 θ 방향으로 선형 투영 (Rθp₁, Rθp₂)
  3. 각 투영된 값을 정렬
  4. 정렬된 값들 간의 비용 합산 → L_DIS
```

**태스크별 구현:**
- **분류**: ResNet-101 (ImageNet 사전학습) + 3층 FC 분류기
- **의미론적 분할**: VGG-16/ResNet-101 + PSPNet 디코더
- **객체 탐지**: SSD + Inception-V2 백본 (분류 및 바운딩 박스 회귀 출력 모두에 SWD 적용)

### 2.4 성능 향상

| 태스크 | 데이터셋 | MCD [58] | SWD (ours) | 향상 |
|--------|----------|-----------|------------|------|
| 숫자 인식 | SVHN→MNIST | 96.2% | **98.9%** | +2.7% |
| 교통 표지 | SYNSIG→GTSRB | 94.4% | **98.6%** | +4.2% |
| 숫자 인식 | MNIST→USPS | 96.5% | **98.1%** | +1.6% |
| 숫자 인식 | USPS→MNIST | 94.1% | **97.1%** | +3.0% |
| 이미지 분류 | VisDA 2017 | 71.9% | **76.4%** | +4.5% |
| 의미론적 분할 | GTA5→Cityscapes (ResNet) | - | **44.5% mIoU** | - |
| 의미론적 분할 | Synthia→Cityscapes (ResNet) | - | **48.1% mIoU** | - |
| 객체 탐지 | VisDA 2018 | 4.7 mAP | **5.9 mAP** | +25% 상대적 향상 |

### 2.5 한계

1. **계산 비용**: 투영 수 $M$에 비례하는 추가 계산 비용 (단, 선형 프로그래밍보다 훨씬 효율적)
2. **클래스 불균형 문제**: 타겟 샘플의 클래스 분포를 명시적으로 고려하지 않음
3. **적대적 학습의 불안정성**: 3단계 학습 과정에서 수렴 보장이 어려울 수 있음
4. **하이퍼파라미터 의존성**: 투영 수 $M$ 설정 필요 (실험적으로 $M=128$ 권장)
5. **레이블 공간 구조 미활용**: 클래스 간 의미론적 관계를 명시적으로 반영하지 않음
6. **심각한 도메인 이동**: 성능이 여전히 Oracle 대비 상당한 격차 존재

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 왜 SWD가 일반화에 유리한가

#### (1) 기하학적으로 의미 있는 그래디언트

기존 $L_1$ 불일치 손실은 두 분류기 출력이 겹칠 때만 유효한 그래디언트를 제공합니다. 반면 Wasserstein 거리는 **분포가 겹치지 않아도** 의미 있는 거리를 측정합니다:

$$W_1(p_1, p_2) \text{는 } p_1 \cap p_2 = \emptyset \text{인 경우에도 유한하고 미분 가능}$$

이는 타겟 도메인에서 소스의 지지(support) 밖에 있는 샘플들을 더 효과적으로 당겨올 수 있음을 의미하며, **결정 경계 근처의 타겟 샘플 처리**에 특히 유리합니다.

#### (2) 유도된 다양체(Manifold)의 활용

SWD를 적대적 방식으로 최적화함으로써, Generator $G$는 소스와 타겟의 특징 다양체를 자연스럽게 정렬하는 방향으로 학습됩니다. T-SNE 시각화 (논문 Figure 2(c)(d))에서 확인되듯, SWD 적응 후 소스와 타겟 특징이 훨씬 더 판별력 있게 분리됩니다.

#### (3) Task-specific 불일치의 중요성

단순한 특징 분포 정렬(feature-level alignment)과 달리, SWD는 **분류기의 출력 공간**에서 불일치를 측정합니다. 이는 단순히 피처를 도메인 불변하게 만드는 것이 아니라, **실제 분류 태스크의 결정 경계 관점**에서 정렬하므로 일반화에 더 직접적으로 기여합니다.

논문에서 MMD 및 DANN 등 순수 분포 매칭 방법이 일부 클래스에서 Source Only보다 **성능이 나빠지는** 경우 (VisDA 분류 결과)를 통해 이를 확인할 수 있습니다.

#### (4) 과제 독립적(Task-agnostic) 설계

SWD는 출력 공간의 구조에 대한 사전 가정이 없습니다:
- 분류: 소프트맥스 출력에 적용
- 의미론적 분할: 픽셀별 출력에 적용
- 객체 탐지: 분류 + 바운딩 박스 회귀 출력 모두에 적용

이러한 범용성은 다양한 실제 환경에서의 **일반화 가능성**을 직접적으로 보여줍니다.

#### (5) 결정 경계 분석 (Toy Experiment)

논문의 Supplementary Material (Figure 4)에서 인터트위닝 문(Intertwining Moons) 데이터셋을 사용한 실험:
- **Source Only**: 타겟 샘플의 Region 1, 2 모두 잘못 분류
- **MCD**: Region 1만 올바르게 적응, Region 2는 실패
- **SWD**: 모든 영역에서 올바른 결정 경계 학습

이는 SWD가 **더 복잡한 도메인 이동 패턴**에 대해서도 일반화 능력이 뛰어남을 시사합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### (1) 최적 운송 이론의 도메인 적응 통합

SWD는 최적 운송 이론을 실용적인 딥러닝 프레임워크에 통합하는 선구적 사례로서, 이후 연구에서 다양한 OT 기반 도메인 적응 방법의 발전을 촉진했습니다.

#### (2) 범용 불일치 측정의 표준화

Task-specific 불일치 측정이 단순 분포 매칭보다 우수하다는 것을 여러 태스크에서 입증함으로써, 이후 UDA 연구의 방향성을 **task-aware alignment** 쪽으로 전환하는 데 기여했습니다.

#### (3) 멀티태스크 도메인 적응 연구 촉진

분류, 분할, 탐지 등 다양한 태스크에 동일한 프레임워크를 적용할 수 있음을 보여줌으로써, **통합 도메인 적응 프레임워크** 연구의 기반을 마련했습니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 내용 중 일부는 논문 원문에 직접 언급되지 않은 사항으로, 제가 알고 있는 2020년 이후 주요 연구 흐름에 기반합니다. 특정 수치나 세부 내용의 정확성에 대해서는 원 논문을 직접 확인하시길 권장합니다.

#### 비교 표

| 방법 | 주요 아이디어 | SWD 대비 개선점 | 한계 |
|------|-------------|----------------|------|
| **SHOT** (ICML 2020) | 소스 없는 도메인 적응; 정보 최대화 + 의사 레이블 | 소스 데이터 불필요 | OT 기반 기하학적 정렬 부재 |
| **SDAT** (ICML 2022) | 더 날카로운 결정 경계 최적화를 위한 적대적 훈련 개선 | 수렴 안정성 향상 | 계산 복잡도 증가 |
| **OT-DA 계열** | Unbalanced OT, partial OT 등 확장 | 클래스 불균형 처리 | 계산 비용 높음 |
| **Source-free DA** | 소스 데이터 없이 적응 (개인정보 보호 관점) | 실용성 향상 | 성능 상한 존재 |

#### 주요 연구 동향

**1. 소스 없는 도메인 적응 (Source-free DA)**
- SWD는 소스 데이터에 의존하는 반면, 개인정보 보호 및 데이터 접근 제약으로 인해 소스 없는 설정이 중요해짐
- 관련 연구: Liang et al., "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation," ICML 2020

**2. 오픈셋 및 부분 도메인 적응**
- 소스와 타겟의 클래스 공간이 완전히 겹치지 않는 현실적 시나리오
- SWD의 논문도 이를 미래 연구 방향으로 언급 (open set adaptation [59], zero-shot domain adaptation [49])

**3. 자기지도 학습과의 결합**
- 대조 학습(contrastive learning) 기반의 도메인 불변 표현 학습
- SWD의 task-specific 접근법과 자기지도 방법의 결합 가능성

**4. Transformer 기반 도메인 적응**
- ViT(Vision Transformer) 백본을 활용한 도메인 적응
- SWD 프레임워크는 백본 독립적이므로 Transformer 기반 구조에도 적용 가능

### 4.3 앞으로 연구 시 고려할 점

#### (1) 클래스 균형 문제
타겟 도메인의 클래스 분포가 불균형할 경우, SWD가 특정 클래스에 편향될 수 있습니다. 클래스 조건부 SWD 또는 클래스 균형 가중치 도입을 고려해야 합니다.

$$\text{SWD}_{class}(\mu, \nu) = \sum_{k=1}^{K} w_k \cdot \text{SWD}(\mu_k, \nu_k)$$

#### (2) 계산 효율성 개선
- $M=128$ 투영은 실용적이지만, 고해상도 출력(의미론적 분할)에서 추가 비용 발생
- 적응적 투영 수 선택 또는 중요도 기반 투영 샘플링 연구 필요

#### (3) 이론적 보장 강화
- 현재 SWD 기반 적응의 **일반화 오차 경계(generalization bound)**에 대한 이론적 분석이 부족
- Ben-David et al.의 도메인 적응 이론과 SWD를 연결하는 연구 필요

$$\varepsilon_T(h) \leq \varepsilon_S(h) + d_{\mathcal{H}\Delta\mathcal{H}}(D_S, D_T) + \lambda$$

여기서 $d_{\mathcal{H}\Delta\mathcal{H}}$를 SWD로 대체하는 이론적 연결 고리 확립이 필요합니다.

#### (4) 멀티소스 및 멀티타겟 도메인 적응
- 단일 소스-타겟 쌍을 넘어 다수의 도메인 간 SWD 계산 확장
- Wasserstein Barycenter 개념의 활용 가능성

#### (5) 개인정보 및 연합학습(Federated Learning)과의 결합
- 소스 데이터에 직접 접근하지 않고 SWD 원칙을 적용하는 방법
- 모델 가중치만을 공유하는 설정에서의 도메인 적응

#### (6) 자기지도 학습 및 대조 학습과의 통합
```
SWD 손실 + 대조 학습 손실의 결합:
L_total = L_s + α·L_SWD + β·L_contrastive
```

#### (7) 대규모 언어 모델(LLM) 시대의 도메인 적응
- Foundation Model의 파인튜닝 시 SWD 원칙 적용
- 프롬프트 튜닝 기반 도메인 적응과의 결합

---

## 참고 자료

### 논문 원문
- **Lee, C.-Y., Batra, T., Baig, M. H., & Ulbricht, D.** (2019). "Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation." *arXiv:1903.04064v1* [cs.CV]

### 논문 내 주요 참고문헌
- Saito, K., et al. "Maximum Classifier Discrepancy for Unsupervised Domain Adaptation." *CVPR 2018.* [논문 내 참조 [58]]
- Arjovsky, M., et al. "Wasserstein Generative Adversarial Networks." *ICML 2017.* [논문 내 참조 [1]]
- Villani, C. "Optimal Transport, Old and New." *Springer-Verlag, 2009.* [논문 내 참조 [73]]
- Rabin, J., et al. "Wasserstein Barycenter and its Application to Texture Mixing." *SSVM 2011.* [논문 내 참조 [53]]
- Bonneel, N., et al. "Sliced and Radon Wasserstein Barycenters of Measures." *JMIV 2015.* [논문 내 참조 [4]]
- Damodaran, B. B., et al. "DeepJDOT: Deep Joint Distribution Optimal Transport for Unsupervised Domain Adaptation." *ECCV 2018.* [논문 내 참조 [12]]
- Courty, N., et al. "Optimal Transport for Domain Adaptation." *TPAMI 2016.* [논문 내 참조 [10]]
- Ganin, Y., & Lempitsky, V. "Unsupervised Domain Adaptation by Backpropagation." *ICML 2014.* [논문 내 참조 [17]]

### 2020년 이후 관련 연구 (참고용 - 독자적 확인 권장)
- Liang, J., et al. "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation." *ICML 2020.*
- Ben-David, S., et al. "A Theory of Learning from Different Domains." *Machine Learning 2010.*

> ⚠️ **면책 조항**: 2020년 이후 최신 연구 비교 분석 부분은 제가 학습한 지식에 기반하며, 일부 세부 사항은 실제 논문과 차이가 있을 수 있습니다. 정확한 수치 및 방법론은 해당 논문을 직접 확인하시기 바랍니다.
