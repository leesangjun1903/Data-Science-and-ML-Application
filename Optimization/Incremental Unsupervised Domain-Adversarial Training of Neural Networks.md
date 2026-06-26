# Incremental Unsupervised Domain-Adversarial Training of Neural Networks

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Gallego et al., 2020, arXiv:2001.04129)은 **비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)** 문제를 **점진적(incremental)** 관점에서 접근하는 새로운 전략인 **iDANN(incremental DANN)** 을 제안합니다.

기존의 도메인 적응(DA) 방법들은 소스 도메인과 타겟 도메인을 동시에 학습하여 도메인 불변 특징(domain-invariant features)을 한 번에 학습하는 방식이었습니다. 이 논문은 **타겟 도메인 샘플을 점진적으로 소스 도메인에 편입**시키면서 반복 학습하면, 기존 DANN보다 훨씬 높은 성능을 얻을 수 있다고 주장합니다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **점진적 DA 프레임워크** | DANN을 기반으로 반복적 레이블링 전략(iDANN) 제안 |
| **두 가지 샘플 선택 정책** | Confidence Policy 및 kNN Policy 제안 |
| **Label Smoothing 적용** | 노이즈 레이블 완화를 위한 soft target 훈련 |
| **범용적 적용 가능성** | 제안 방법이 기반 DA 모델에 독립적임을 입증 |
| **성능 향상 입증** | 7개 소스-타겟 쌍 중 5개에서 SOTA 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**도메인 불일치(Domain Shift)** 문제:
훈련 데이터(소스 도메인 $\mathcal{D}_S$)와 테스트 데이터(타겟 도메인 $\mathcal{D}_T$)의 분포가 다를 경우, 학습된 모델의 성능이 크게 저하됩니다. 특히 **비지도 DA** 설정에서는 타겟 도메인에 레이블이 전혀 없습니다.

**공식적 정의:**

- 레이블된 소스 집합: $S = \{(x_i, y_i)\}_{i=1}^{n} \sim (\mathcal{D}_S)^n$
- 레이블 없는 타겟 집합: $T = \{(x_i)\}_{i=1}^{n'} \sim (\mathcal{D}_T)^{n'}$
- 목표: $\mathcal{D}_T$에 대한 레이블 분류기 $h: X \rightarrow Y$ 학습

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 기반 모델: DANN (Domain-Adversarial Neural Network)

DANN은 세 개의 모듈로 구성됩니다:
- $G_f$: 특징 추출기(Feature Extractor)
- $G_y$: 레이블 분류기(Label Classifier)
- $G_d$: 도메인 분류기(Domain Classifier)

**Gradient Reversal Layer (GRL)** 를 통해 $G_f$는 도메인 불변 특징을 학습합니다.

DANN의 학습 파라미터 업데이트 수식:

$$\theta_f \leftarrow \theta_f - \mu \left( \frac{\partial \mathcal{L}_y}{\partial \theta_f} - \lambda \frac{\partial \mathcal{L}_d}{\partial \theta_f} \right) \tag{1}$$

여기서:
- $\theta_f$: $G_f$의 가중치
- $\mu$: 학습률(learning rate)
- $\mathcal{L}_y$: 레이블 분류기의 손실 함수
- $\mathcal{L}_d$: 도메인 분류기의 손실 함수
- $\lambda$: 도메인 적응 강도를 조절하는 하이퍼파라미터

GRL은 역전파 시 $-\lambda$를 기울기에 곱하여 $G_f$가 도메인을 구별하지 못하도록 강제합니다.

#### 2.2.2 iDANN: 점진적 DANN

**핵심 아이디어**: 훈련된 DANN 모델이 타겟 도메인 샘플에 대해 높은 확신을 가지는 샘플을 선택하여 소스 도메인에 추가하고, 이를 반복합니다.

**알고리즘 1 (iDANN)의 핵심 흐름:**

```
while T ≠ ∅ do
    G_f, G_y ← DANN 훈련({S, T, e, b, λ})
    B̂_r ← selection_policy(G_f, G_y, T, r)
    S ← S ∪ B̂_r
    T ← T \ B̂_r
    e ← e_inc
    r ← β·r
end while
T̂ ← {(x_i, y_i) | x_i ~ D_T, y_i = G_y(G_f(x_i))}
CNN 훈련(T̂, e, b)
```

주요 하이퍼파라미터:
- $e$: 초기 훈련 에포크 수 (300)
- $e_{inc}$: 점진적 단계의 에포크 수 (25)
- $r$: 매 반복에서 선택하는 샘플 비율 (초기 5%)
- $\beta$: 반복마다 $r$을 증가시키는 상수 (1.5)

#### 2.2.3 샘플 선택 정책

**① Confidence Policy (확신도 기반)**

Softmax 함수 출력:

$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{L} e^{z_j}}, \quad i = 1, \ldots, L, \quad \mathbf{z} = (z_1, \ldots, z_L) \in \mathbb{R}^L \tag{2}$$

각 샘플에 대한 최대 예측 확률값을 기준으로 내림차순 정렬 후 상위 $r$개 선택합니다.

**② kNN Policy (기하학적 특징 공간 기반)**

소스 도메인 특징 집합 $F_S = G_f(S)$를 구성한 후, 타겟 샘플의 $k$-최근접 이웃이 $G_y$가 예측한 레이블과 일치하는 경우에만 선택합니다:

$$\text{선택 조건: } y_i = l \text{ AND } m = k$$

여기서 $l$은 kNN 예측 레이블, $m$은 $k$개 이웃 중 같은 레이블의 수입니다.

#### 2.2.4 Label Smoothing

최종 CNN 훈련 시 레이블 노이즈 완화를 위해 label smoothing 적용:

$$y'_i = (1 - \epsilon) y_i + \frac{\epsilon}{L} \tag{3}$$

여기서 $\epsilon$은 스무딩 파라미터(소량의 상수), $L$은 총 클래스 수입니다.

---

### 2.3 모델 구조

총 3가지 CNN 아키텍처를 사용하였으며, 모두 DANN의 원 논문에서 사용된 것과 동일합니다:

| 모델 | 특징 추출기 | 레이블 분류기 | 도메인 분류기 | 사용 데이터 |
|---|---|---|---|---|
| Model 1 | Conv(32,5,5)+MaxPool, Conv(48,5,5)+MaxPool | FC(100), FC(100), FC(L) | FC(100), FC(1) | MNIST 계열 |
| Model 2 | Conv(64,5,5)+MaxPool, Conv(64,5,5)+MaxPool, Conv(128,5,5) | FC(3072), FC(2048), FC(L) | FC(1024), FC(1024), FC(1) | SVHN 포함 |
| Model 3 | Conv(96,5,5)+MaxPool, Conv(144,3,3)+MaxPool, Conv(256,5,5)+MaxPool | FC(512), FC(L) | FC(1024), FC(1024), FC(1) | 교통 표지판 |

모든 Conv 및 FC 레이어에 **ReLU** 활성화 함수 적용, 레이블 분류기 출력은 **Softmax**, 도메인 분류기 출력은 **Sigmoid** 사용.

---

### 2.4 성능 향상

**iDANN vs DANN 비교 (Table 5 기준):**

| Source → Target | CNN Src. | DANN | iDANN(1) | iDANN(2) | CNN Tgt. |
|---|---|---|---|---|---|
| MNIST → MNIST-M | 55.71% | 78.70% | 96.09% | **96.67%** | 97.34% |
| MNIST → Syn Numbers | 32.14% | 44.66% | 80.79% | **84.82%** | 99.34% |
| Syn Numbers → MNIST | 60.04% | 89.35% | 98.13% | **99.35%** | 98.94% |
| Syn Signs → GTSRB | 69.79% | 85.28% | 96.31% | **98.00%** | 97.89% |
| **Average** | 57.53% | 68.23% | 84.28% | **85.91%** | 96.95% |

- iDANN은 DANN 대비 평균 **약 16% 향상**
- 최고의 경우 MNIST → Syn Numbers에서 **약 36% 향상**
- kNN Policy는 Confidence Policy 대비 평균 **6.36% 향상**, 마지막 반복에서 최대 **24.85% 향상**

**SOTA 비교 (Table 6 기준):**

7개 소스-타겟 쌍 중 **5개에서 최고 성능** 달성. 특히 MNIST → Syn Numbers에서 기존 최고 성능 대비 **약 30% 향상**.

---

### 2.5 한계점

1. **계산 비용**: 반복적 DANN 재훈련으로 인해 계산 비용이 기본 DANN에 비해 상당히 증가
2. **중지 기준 미확립**: 타겟 샘플 예측이 신뢰할 수 없을 때를 감지하는 원칙적인 중지 기준이 없음
3. **SVHN → MNIST 성능 저하**: 일부 소스-타겟 쌍(SVHN → MNIST)에서 DTA 등 특화 방법에 비해 낮은 성능
4. **제한된 아키텍처**: 소스-타겟 쌍별 최적화된 아키텍처가 아닌 DANN 원 논문의 아키텍처 사용
5. **입력 유형 제한**: 이미지 분류에만 평가, 시퀀스 등 다른 입력 유형에 대한 검증 미흡
6. **하이퍼파라미터 민감성**: $\lambda \geq 10^{-1}$ 시 학습 불안정

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 일반화 성능과 관련된 핵심 메커니즘을 중점적으로 살펴보면 다음과 같습니다.

### 3.1 점진적 도메인 적응을 통한 일반화

iDANN의 핵심은 **쉬운 샘플에서 어려운 샘플로** 순차적으로 타겟 도메인 정보를 학습하는 것입니다. t-SNE 시각화(Fig. 6)에서 확인되듯이:

- **초기 반복**: 소스 도메인 클러스터에 이미 근접한 타겟 샘플만 선택 → 안전한 지식 이전
- **중간 반복**: 선택된 샘플이 도메인 불변 특징 학습을 강화 → 중심 클러스터(미분류 타겟 샘플) 축소
- **후기 반복**: 가장 복잡한 타겟 샘플까지 학습 → 미분류 클러스터 최소화

이 과정은 **커리큘럼 학습(Curriculum Learning)** 과 유사한 효과를 가지며, 점진적으로 더 어려운 경우를 학습함으로써 모델의 일반화 능력이 강화됩니다.

### 3.2 Label Smoothing의 정규화 효과

수식 (3)의 label smoothing:

$$y'_i = (1 - \epsilon) y_i + \frac{\epsilon}{L}$$

노이즈가 포함된 의사 레이블(pseudo-label)을 soft target으로 처리함으로써 **과적합을 방지**합니다. 실제로 논문에서는 일부 소스-타겟 쌍(MNIST-M → MNIST, Syn Numbers → SVHN)에서 iDANN이 **정답 타겟 레이블로 훈련한 CNN보다 높은 성능**을 보였는데, 이는 잘못 할당된 레이블이 **정규화 효과**를 발휘했기 때문으로 해석됩니다.

### 3.3 kNN Policy의 분포 일치 강화

kNN 정책은 단순히 네트워크 확신도만 보는 것이 아니라, **특징 공간의 기하학적 구조**를 활용합니다. 타겟 샘플의 특징이 소스 도메인의 동일 클래스 클러스터 내에 위치할 때만 선택하므로:

- **분포 정렬(Distribution Alignment)** 이 더 정확하게 이루어짐
- 경계 근처의 불확실한 샘플 배제로 **노이즈 레이블 최소화**
- 결과적으로 후기 반복에서 최대 24.85%의 추가 성능 향상

### 3.4 소스 도메인 정확도 유지

기존 DANN은 도메인 적응 과정에서 소스 도메인 성능이 저하될 수 있으나, iDANN은 **소스 도메인 정확도를 유지**하면서 타겟 도메인 성능을 향상시킵니다. 이는 모델이 두 도메인에 모두 일반화될 수 있음을 의미합니다.

### 3.5 범용성

저자들이 명시적으로 언급하듯, iDANN은 **기반 DA 모델에 독립적**입니다. 즉, 더 강력한 DA 알고리즘이 개발될수록 iDANN 프레임워크를 적용했을 때의 성능도 비례적으로 향상될 수 있어, **장기적 일반화 가능성**이 높습니다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

**① 점진적/커리큘럼 도메인 적응 연구 방향 제시**

iDANN은 도메인 적응을 일회성 최적화가 아닌 **반복적 정제(iterative refinement)** 과정으로 바라보는 새로운 패러다임을 제시합니다. 이는 이후 연구에서 커리큘럼 학습과 DA를 결합하는 시도로 이어질 수 있습니다.

**② 의사 레이블링(Pseudo-Labeling)과 DA의 융합**

iDANN은 실질적으로 **선택적 의사 레이블링**을 DA에 접목한 형태입니다. 이후 연구에서 더 정교한 불확실성 추정 방법(예: Bayesian 딥러닝, MC Dropout)과 결합될 수 있습니다.

**③ 자기지도학습(Self-Supervised Learning)과의 연계**

타겟 도메인을 점진적으로 레이블링하는 아이디어는 최근 활발한 **자기지도학습** 및 **반지도학습** 연구와 자연스럽게 연결됩니다.

**④ 범용 DA 프레임워크로서의 가능성**

기반 DA 알고리즘에 독립적이라는 특성 덕분에, DANN 이외에도 ADDA, VADA, Transformer 기반 DA 등 최신 모델에 iDANN 전략을 적용하는 연구로 확장될 수 있습니다.

### 4.2 앞으로 연구 시 고려할 점

**① 원칙적 중지 기준(Stop Criterion) 설계**

현재 iDANN은 타겟 샘플이 모두 소진될 때까지 반복합니다. 그러나 후기 반복에서 예측 신뢰도가 급격히 하락하는 경향이 있으므로, **불확실성 기반 중지 기준**이 필요합니다. 예를 들어:
- 엔트로피 기반 불확실성 임계값 설정
- 베이지안 딥러닝을 통한 예측 신뢰 구간 추정

**② 대규모 데이터셋 및 복잡한 도메인으로 확장**

본 논문은 주로 숫자 및 교통 표지판 데이터셋을 사용했습니다. ImageNet 수준의 대규모 데이터셋이나 의료 영상, NLP 등 다른 도메인에서의 검증이 필요합니다.

**③ 현대적 아키텍처와의 결합**

ResNet, Vision Transformer(ViT), CLIP 등 강력한 최신 아키텍처와 결합 시 성능이 크게 향상될 가능성이 있으며, 이에 대한 체계적 연구가 필요합니다.

**④ 노이즈 레이블의 정량적 영향 분석**

점진적 과정에서 누적되는 노이즈 레이블이 최종 성능에 미치는 영향을 더 정밀하게 분석하고, 이를 최소화하기 위한 **능동적 샘플 재검토(active sample review)** 메커니즘이 필요합니다.

**⑤ 계산 효율성 개선**

반복적 DANN 재훈련은 계산 비용이 높습니다. **지식 증류(Knowledge Distillation)** 또는 **메타러닝(Meta-Learning)** 을 활용하여 빠른 적응이 가능한 경량화 버전 개발이 필요합니다.

**⑥ 다중 소스/타겟 도메인으로의 확장**

현재는 단일 소스-단일 타겟 구조이지만, 실세계에서는 다수의 소스 도메인과 타겟 도메인이 존재합니다. **Multi-source DA** 또는 **Continual DA** 로의 확장이 중요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> **주의**: 아래의 최신 연구 비교는 제 훈련 데이터 기준(2024년 초까지)의 일반적 지식에 근거하며, 개별 논문의 세부 수치는 원문 확인을 권장합니다.

### 5.1 Self-Training 및 Pseudo-Label 기반 연구

iDANN과 유사한 의사 레이블링 접근법이 이후 연구에서 더욱 발전하였습니다:

| 연구 | 핵심 아이디어 | iDANN과의 차이 |
|---|---|---|
| **SHOT** (Liang et al., ICML 2020) | 소스 프리(source-free) DA: 소스 데이터 없이 타겟 도메인만으로 적응 | 소스 데이터 접근 불필요 |
| **NRC** (Yang et al., NeurIPS 2021) | Neighborhood Reciprocal Clustering을 통한 의사 레이블 정제 | 클러스터 구조 명시적 활용 |
| **ATDOC** (Liu et al., CVPR 2021) | Attention 기반 도메인 적응 + 의사 레이블 | Transformer 구조 활용 |

**iDANN과의 비교:**
- iDANN은 kNN 기반 기하학적 선택으로 신뢰성 높은 샘플만 선택하는 반면, 이후 연구들은 더 정교한 클러스터링이나 그래프 기반 방법을 사용
- SHOT 등은 소스 데이터 없이도 작동하는 **Source-Free DA** 로 실용성을 높임

### 5.2 Vision Transformer 기반 DA

**CDTrans** (Xu et al., ICLR 2022), **TVT** (Yang et al., 2021) 등은 Transformer의 self-attention 메커니즘을 DA에 적용:
- Transformer의 강력한 특징 추출 능력으로 도메인 불변 표현 학습
- iDANN의 점진적 접근법을 Transformer 기반 모델에 적용하면 추가 성능 향상 가능

### 5.3 Source-Free Domain Adaptation

이후 연구의 중요한 흐름 중 하나는 **소스 데이터 없는 DA**입니다. iDANN은 소스 데이터를 계속 활용하는데, 실제 산업 환경에서는 프라이버시 문제로 소스 데이터 접근이 불가할 수 있습니다. 이를 해결하는 연구 방향이 주목받고 있습니다.

### 5.4 Test-Time Adaptation (TTA)

**TTT** (Sun et al., 2020), **TENT** (Wang et al., ICLR 2021) 등은 테스트 시점에 실시간으로 모델을 적응시키는 방법을 제안:
- 배치 정규화 레이어의 통계량을 타겟 도메인에 맞게 업데이트
- iDANN보다 훨씬 빠른 적응 가능, 단 점진적 레이블링 없음

### 5.5 비교 요약

```
iDANN의 위치:
[낮은 적응 속도] ←→ [높은 성능]
  TTT/TENT     iDANN     Oracle(타겟 지도학습)
(빠르지만 단순)  (느리지만 정확한 점진적 적응)  (상한선)
```

iDANN은 **오프라인 배치 DA 시나리오**에서, 특히 도메인 간 격차가 크고 충분한 타겟 데이터가 있을 때 강점을 발휘합니다. 반면 실시간 스트리밍 데이터나 소스 데이터 접근이 불가한 환경에서는 최신 Source-Free DA나 TTA 방법이 더 적합합니다.

---

## 참고 자료

**주요 참고 논문 (논문 내 인용 기준):**

1. **Gallego, A.-J., Calvo-Zaragoza, J., & Fisher, R. B. (2020).** *Incremental Unsupervised Domain-Adversarial Training of Neural Networks.* arXiv:2001.04129v1
2. **Ganin, Y., et al. (2016).** *Domain-adversarial training of neural networks.* Journal of Machine Learning Research, 17, 1–35.
3. **Ben-David, S., et al. (2010).** *A theory of learning from different domains.* Machine Learning, 79, 151–175.
4. **Szegedy, C., et al. (2016).** *Rethinking the inception architecture for computer vision.* CVPR 2016. (Label Smoothing 출처)
5. **Shu, R., et al. (2018).** *A DIRT-T approach to unsupervised domain adaptation.* ICLR 2018. (VADA)
6. **Lee, S., et al. (2019).** *Drop to Adapt: Learning discriminative features for unsupervised domain adaptation.* ICCV 2019.
7. **Damodaran, B. B., et al. (2018).** *DeepJDOT.* ECCV 2018.
8. **Liang, J., et al. (2020).** *Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation.* ICML 2020. (SHOT - 비교 분석용)
9. **Wang, D., et al. (2021).** *Tent: Fully Test-Time Adaptation by Entropy Minimization.* ICLR 2021. (비교 분석용)
10. **Kouw, W. M., & Loog, M. (2019).** *A review of domain adaptation without target labels.* IEEE TPAMI.
