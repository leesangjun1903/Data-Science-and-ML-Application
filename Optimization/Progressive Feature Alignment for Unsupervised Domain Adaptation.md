# Progressive Feature Alignment for Unsupervised Domain Adaptation (PFAN)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

PFAN(Progressive Feature Alignment Network)은 비지도 도메인 적응(UDA)에서 기존 의사 레이블(pseudo-label) 기반 방법들의 **오류 누적(error accumulation)** 문제를 해결하기 위해, 타겟 도메인의 **클래스 내 분산(intra-class variation)**을 활용하여 도메인 간 판별적 특징을 **점진적(progressive)**으로 정렬한다.

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| **EHTS** (Easy-to-Hard Transfer Strategy) | 쉬운 샘플부터 어려운 샘플로 점진적으로 신뢰할 수 있는 의사 레이블 샘플 선택 |
| **APA** (Adaptive Prototype Alignment) | 클래스별 소스-타겟 프로토타입을 적응적으로 정렬하여 거짓 레이블 영향 억제 |
| **Temperature Softmax** | 소스 분류기의 수렴 속도를 늦춰 소스 과적합(overfitting) 방지 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 의사 레이블 기반 UDA 방법들은 두 가지 치명적인 한계를 가진다:

1. **강한 선제 가정(strong pre-assumption)**: 올바르게 레이블된 샘플이 잘못 레이블된 샘플의 편향을 감소시킨다고 가정하지만, 도메인 불일치가 클 경우 이 가정이 성립하지 않음
2. **오류 누적(error accumulation)**: 의사 레이블된 샘플 기반으로 범주 손실을 역전파하면 거짓 레이블이 연쇄적으로 악영향을 미침

또한 타겟 도메인 내에 세 가지 유형의 샘플이 존재한다:
- **Easy samples**: 소스 도메인과 충분히 가까워 즉시 올바른 의사 레이블 부여 가능
- **Hard samples**: 소스 도메인과 멀리 위치하여 분류 경계가 모호
- **False-easy samples**: 높은 확률로 잘못된 클래스에 배정되는 위험 샘플

---

### 2.2 제안 방법 (수식 포함)

#### (1) Easy-to-Hard Transfer Strategy (EHTS)

**소스 프로토타입 계산:**

$$c_k^S = \frac{1}{N_k^s} \sum_{(x_i^s, y_i^s) \in D_k^s} G(x_i^s) \tag{1}$$

여기서 $D_k^s$는 클래스 $k$에 속하는 소스 샘플 집합, $N_k^s$는 해당 샘플 수, $G$는 특징 추출기(feature extractor).

**코사인 유사도 기반 의사 레이블 부여:**

$$\psi(x_j^t) = CS(G(x_j^t), c_k^S), \quad k = \{1, 2, ..., C\} \tag{2}$$

타겟 샘플 $x_j^t$의 의사 레이블: $\hat{y}_j^t = k'$, where $k' = \arg\max_k \psi_k(x_j^t)$

**적응적 임계값 (sigmoid 기반 점진적 조절):**

$$\tau = \frac{1}{1 + e^{-\mu \cdot (m+1)}} - 0.01 \tag{3}$$

여기서 $\mu$는 상수(실험에서 $\mu = 0.8$), $m$은 학습 단계.

**샘플 선택 함수:**

$$\forall x_j \in D_t^k \big|_{k=1}^C, \quad w_j = \begin{cases} 1 & \text{if } \psi \geq \tau \\ 0 & \text{if } \psi < \tau \end{cases} \tag{4}$$

---

#### (2) Adaptive Prototype Alignment (APA)

**프로토타입 간 거리 측정:**

$$d(c_k^S, c_k^T) = \left\| c_k^S - c_k^T \right\|^2 \tag{5}$$

**초기 글로벌 타겟 프로토타입:**

$$c_{k(0)}^T = \frac{1}{\hat{D}_t^k} \sum_{(x_j^t, y_j^t) \in \hat{D}_t^k} G(x_j^t) \tag{6}$$

**누적 로컬 프로토타입 (mini-batch 기반):**

$$\bar{c}_{k(I)}^t = \frac{1}{I} \sum_{i=1}^{I} c_{k(i)}^t \tag{7}$$

**글로벌 프로토타입 적응적 업데이트:**

$$\rho_t = CS(\bar{c}_{k(I)}^t, c_{k(I-1)}^T)$$

$$c_{k(I)}^T \leftarrow \rho_t^2 \bar{c}_{k(I)}^t + (1 - \rho_t^2) c_{k(I-1)}^T \tag{8}$$

**APA 손실함수:**

$$\mathcal{L}_{apa}(\theta_g) = \sum_{k=1}^C d(c_{k(I)}^S, c_{k(I)}^T) \tag{9}$$

---

#### (3) Temperature Softmax (비포화 소스 분류기)

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)} \tag{10}$$

여기서 $T > 1$ (실험에서 $T = 1.8$)로 설정하여 소스 분류 손실의 수렴 속도를 늦추고, 소스 샘플에 대한 과적합을 방지.

---

#### (4) 전체 최적화 목적함수

**도메인 판별기 손실:**

$$\mathcal{L}_d(\theta_g, \theta_d) = \mathbb{E}_{x \sim D_s}[\log D(G(x))] + \mathbb{E}_{x \sim \hat{D}_t}[\log D(1 - G(x))] \tag{11}$$

**전체 미니맥스 목적함수:**

$$\min_{\theta_g, \theta_f} \max_{\theta_d} \sum_{i=1}^{n_s} \mathcal{L}_c(F(G(x_i^s; \theta_g); \theta_f), y_i^s) + \lambda \mathcal{L}_d(\theta_g, \theta_d) + \gamma \mathcal{L}_{apa}(\theta_g) \tag{12}$$

여기서 $\mathcal{L}_c$는 표준 크로스 엔트로피 손실, $\lambda$와 $\gamma$는 손실 균형 조절 가중치.

---

### 2.3 모델 구조

```
입력 (소스/타겟 도메인)
        ↓
  [G: Feature Extractor]  ← θ_g (공유 가중치)
        ↓              ↓
  [F: Label Predictor]  [D: Domain Discriminator]
  Softmax with T        Domain Confusion Loss
  Cross-Entropy Loss
        ↑
  [EHTS] → 신뢰 가능한 의사 레이블 샘플 선택
        ↓
  [APA]  → 클래스별 소스-타겟 프로토타입 정렬
```

- **두 단계 학습**: Stage-1(소스 초기화) → Stage-2(EHTS+APA 반복 학습)
- **백본**: Office-31, ImageCLEF-DA에는 AlexNet(ImageNet 사전학습), 디지털 데이터셋에는 CNN 사용

---

### 2.4 이론적 근거 (Domain Adaptation 이론)

Ben-David et al. (2010)의 도메인 적응 이론에 기반:

$$\forall h \in \mathcal{H}, \quad R_T(h) \leq R_S(h) + \frac{1}{2} d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{S}, \mathcal{T}) + C \tag{13}$$

세 항목의 역할:
- $R_S(h)$: 소스 도메인 오류 → **Temperature Softmax**로 소스 과적합 방지
- $d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{S}, \mathcal{T})$: 도메인 불일치 → **적대적 학습**으로 최소화
- $C$: 이상적 결합 가설의 공유 오류 → **EHTS + APA**로 범주 정렬을 통해 최소화

**Theorem 1** (공유 오류 상한):

$$C \leq \min_{h \in \mathcal{H}} R_S(h, f_S) + R_{T'}(h, f_{\hat{T}}) + 2R_{T'}(f_S, f_{\hat{T}}) + R_{T'}(f_T, f_{\hat{T}}) \tag{15}$$

- EHTS는 $R_{T'}(f_T, f_{\hat{T}})$ 최소화 → 거짓 레이블 비율 감소
- APA는 $R_{T'}(f_S, f_{\hat{T}})$ 최소화 → 범주 분포 정렬

---

### 2.5 성능 향상

#### Office-31 (AlexNet 기반)

| 방법 | A→W | D→W | W→D | A→D | D→A | W→A | **Avg** |
|------|-----|-----|-----|-----|-----|-----|---------|
| MSTN | 80.5 | 96.9 | 99.9 | 74.5 | 62.5 | 60.0 | 79.1 |
| **PFAN** | **83.0** | **99.0** | **99.9** | **76.3** | **63.3** | **60.8** | **80.4** |

#### 디지털 데이터셋

| 방법 | MNIST→SVHN | SVHN→MNIST | MNIST→USPS |
|------|-----------|-----------|-----------|
| ATT | 52.8 | 85.0 | - |
| MSTN | 수렴 불가 | 91.7 | 92.9 |
| **PFAN** | **57.6** | **93.9** | **95.0** |

특히 어려운 전이 태스크(MNIST→SVHN)에서 기존 최고 성능 대비 **+4.8%** 개선.

---

### 2.6 한계점

1. **백본 의존성**: AlexNet 기반으로 실험하여 ResNet 등 최신 백본과의 비교 부재
2. **하이퍼파라미터 민감성**: $T$, $\mu$, $\lambda$, $\gamma$ 등 다수의 하이퍼파라미터 수동 설정 필요
3. **Closed-set 가정**: 소스와 타겟 도메인이 동일한 클래스를 공유한다는 가정 → 실제 환경의 open-set/partial DA에는 직접 적용 어려움
4. **계산 비용**: 글로벌 프로토타입의 반복적 업데이트로 대규모 데이터셋에서 계산 비용 증가 가능
5. **False-easy sample 완전 제거의 한계**: EHTS가 일부 거짓 레이블 샘플을 포함할 수 있으며, APA만으로 완전한 억제를 보장하지는 못함

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상의 핵심 메커니즘

PFAN은 세 가지 관점에서 모델의 일반화 성능을 향상시킨다:

#### (a) 점진적 의사 레이블링을 통한 견고성 확보

식 (3)의 sigmoid 기반 임계값 $\tau$는 학습 초반에는 매우 신뢰도 높은 샘플만 선택하고, 학습이 진행될수록 점차 더 어려운 샘플도 수용하는 커리큘럼 학습(curriculum learning) 전략을 구현한다. 이는 **초기 학습의 견고성**을 확보하여 일반화 성능을 높인다.

#### (b) 글로벌 프로토타입 정렬의 통계적 안정성

단순 미니배치 기반 프로토타입 정렬 대신 **누적 글로벌 프로토타입**(식 7, 8)을 사용함으로써, 개별 배치의 노이즈와 거짓 레이블의 영향을 통계적으로 평균화한다. 이는 **클래스 조건부 분포의 안정적 정렬**을 보장하여 타겟 도메인에서의 일반화 성능을 향상시킨다.

#### (c) 비포화 소스 분류기의 역할

Temperature Softmax ($T > 1$)는 소스 분류 손실의 수렴을 늦춤으로써 모델이 소스 도메인에 과적합되는 것을 방지한다. 이는 이론적으로 식 (13)에서 $R_S(h)$의 조기 최소화가 $C$ 항목에 악영향을 주는 것을 방지하는 역할을 한다.

수식으로 표현하면, 비포화 분류기의 목표는:

$$\min_{h} \left[ R_S(h, f_S) + R_{T'}(h, f_{\hat{T}}) \right]$$

에서 $R_S$의 과도한 최소화를 억제하고 $R_{T'}$의 최소화에 더 많은 용량을 할당하는 것이다.

### 3.2 Ablation Study를 통한 일반화 기여도 검증

| 변형 모델 | A→W | SVHN→MNIST | 의미 |
|-----------|-----|-----------|------|
| PFAN (Random) | 77.0 | 87.2 | 점진적 선택의 중요성 |
| PFAN (Full) | 81.9 | 92.5 | 노이즈 제거의 중요성 |
| PFAN (woAPA) | 76.4 | 82.0 | 범주 정렬의 핵심 기여 |
| PFAN (woT) | 80.6 | 92.1 | 비포화 분류기의 기여 |
| **PFAN** | **83.0** | **93.9** | 모든 요소 결합 |

- **PFAN vs PFAN(Full)**: 전체 타겟 샘플을 사용하면 오히려 성능이 저하되는데, 이는 신뢰도 기반 선택의 일반화 기여를 명확히 보여줌
- **PFAN vs PFAN(woAPA)**: APA 제거 시 A→W에서 약 6.6%p 성능 저하로, 범주 정렬이 일반화의 핵심임을 입증

---

## 4. 향후 연구에 대한 영향과 고려사항

### 4.1 향후 연구에 미치는 영향

#### (a) 커리큘럼 학습의 UDA 적용 확산

EHTS의 Easy-to-Hard 전략은 이후 연구들에서 **자기 지시 학습(self-paced learning)**과 **커리큘럼 도메인 적응** 연구를 촉진하였다. 의사 레이블의 신뢰도를 점진적으로 조절하는 아이디어는 SHOT(2020), NRC(2021) 등에서 더욱 발전된다.

#### (b) 프로토타입 기반 정렬의 패러다임 제시

APA의 클래스별 프로토타입 정렬 개념은 이후 **프로토타입 네트워크 기반 도메인 적응** 연구의 기초가 되었다. 특히 ProDA(2021)와 같은 연구에서 더욱 정교한 프로토타입 기반 정렬이 제안된다.

#### (c) 이론적 프레임워크의 활용

Ben-David 이론을 바탕으로 $C$ 항목의 최소화 필요성을 명시적으로 지적한 것은, 이후 **클래스 수준 정렬의 이론적 정당화** 연구에 기여하였다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ 이하 최신 연구 비교는 PFAN 논문 자체에는 포함되지 않으며, 제 학습 지식(2023년까지)을 기반으로 기술합니다. 일부 세부 수치는 부정확할 수 있습니다.

#### (a) SHOT (ICML 2020) - "Do We Really Need to Access the Source Data?"

- **핵심 아이디어**: 소스 데이터 없이 소스 모델만으로 타겟 도메인에 적응(source-free DA)
- **PFAN과의 비교**:
  - PFAN은 소스 데이터에 직접 접근이 필요하지만, SHOT은 소스 모델(frozen classifier)만 사용
  - 두 방법 모두 의사 레이블과 정보 극대화를 활용하지만, SHOT은 소스 데이터 프라이버시 문제를 추가로 해결
- **Office-31 평균 정확도**: ResNet-50 기준 SHOT ~90.1% vs PFAN ~80.4% (AlexNet) → 백본 차이로 직접 비교는 어려움

#### (b) CDAN (NeurIPS 2018 → 이후 확장)

- **핵심 아이디어**: 멀티선형 조건부(multilinear conditioning)으로 판별적 도메인 판별기 설계
- **PFAN과의 비교**: PFAN이 프로토타입 정렬로 범주 정렬을 명시적으로 수행하는 반면, CDAN은 조건부 특징 공간에서 판별기를 학습

#### (c) TransDA / SSRT (CVPR 2021, 2022)

- **핵심 아이디어**: Vision Transformer를 UDA에 적용
- **PFAN과의 비교**: PFAN의 CNN 기반 접근과 달리 Self-Attention 메커니즘을 통해 글로벌 의존성을 포착하여 더 높은 성능 달성

#### (d) NRC (NeurIPS 2021) - "Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation"

- **핵심 아이디어**: 소스 데이터 없이 타겟 도메인의 이웃 구조를 활용한 자기 지도 학습
- **PFAN과의 비교**: PFAN의 프로토타입 정렬과 유사하게 클러스터 구조를 활용하지만, 소스 데이터 의존성을 완전히 제거

#### 비교 요약 테이블

| 방법 | 소스 데이터 필요 | 범주 정렬 | 점진적 선택 | 백본 | Office-31 Avg |
|------|:-:|:-:|:-:|------|:---:|
| **PFAN (2019)** | ✅ | ✅ (APA) | ✅ (EHTS) | AlexNet | 80.4 |
| CDAN (2018) | ✅ | 간접적 | ❌ | ResNet-50 | ~87.7 |
| SHOT (2020) | ❌ | ✅ | ✅ | ResNet-50 | ~90.1 |
| NRC (2021) | ❌ | ✅ | ❌ | ResNet-50 | ~91.0 |

---

### 4.3 향후 연구 시 고려해야 할 점

#### (a) 더 강력한 백본으로의 전환
PFAN은 AlexNet 기반으로 평가되었으나, 현재 연구 트렌드는 ResNet-50/101, ViT 등을 사용한다. **PFAN의 EHTS와 APA 개념을 Transformer 기반 백본에 통합**하는 연구가 유의미할 것이다.

#### (b) Source-Free 시나리오로의 확장
소스 데이터 프라이버시 문제가 현실적인 제약으로 부상하면서, PFAN의 프로토타입 기반 정렬을 소스 모델만으로 수행하는 방법론 연구가 필요하다. 구체적으로:

$$c_k^S \leftarrow \text{소스 모델에서 추출한 클래스 중심}$$

을 사전 저장(precomputed)하는 방식으로 확장 가능하다.

#### (c) Open-Set / Partial DA로의 일반화
PFAN은 소스와 타겟이 동일한 클래스 집합을 공유한다고 가정하지만, 실제 환경에서는 unknown class가 존재한다. EHTS의 임계값 $\tau$를 활용하여 unknown class를 거부(rejection)하는 메커니즘 도입이 필요하다.

#### (d) 온라인 적응 설정에서의 적용
현재 PFAN은 오프라인 배치 학습을 가정하지만, 스트리밍 데이터 환경에서의 점진적 도메인 적응(Continual/Online DA)에 EHTS 전략을 확장하는 연구 방향이 존재한다.

#### (e) 프로토타입 품질 향상
APA의 프로토타입 추정 품질이 성능에 직결되므로:
- 가우시안 혼합 모델(GMM)을 활용한 클래스 내 다중 프로토타입 학습
- 온라인 클러스터링과의 결합

등을 고려할 수 있다.

#### (f) 멀티소스 도메인 적응으로의 확장
단일 소스 도메인 가정을 완화하여, 복수의 소스 도메인으로부터 지식을 통합하는 멀티소스 DA에서 PFAN의 APA를 어떻게 설계할지가 중요한 연구 과제이다.

---

## 참고자료 출처

**주 논문:**
- Chen, C., Xie, W., Huang, W., Rong, Y., Ding, X., Huang, Y., Xu, T., & Huang, J. (2019). **Progressive Feature Alignment for Unsupervised Domain Adaptation**. *CVPR 2019*, pp. 627–636. (제공된 PDF)

**논문 내 인용 문헌 (주요):**
- Ben-David, S., et al. (2010). A theory of learning from different domains. *Machine Learning*, 79(1-2):151–175.
- Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. *ICML*.
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv:1503.02531*.
- Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. *NeurIPS*.
- Tzeng, E., et al. (2017). Adversarial discriminative domain adaptation. *CVPR*.
- Xie, S., et al. (2018). Learning semantic representations for unsupervised domain adaptation. *ICML*.

**2020년 이후 비교 연구 (지식 기반, 원문 직접 참조 없음):**
- Liang, J., et al. (2020). Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation. *ICML 2020*. (SHOT)
- Yang, S., et al. (2021). Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation. *NeurIPS 2021*. (NRC)
- Long, M., et al. (2018). Conditional Adversarial Domain Adaptation. *NeurIPS 2018*. (CDAN)
