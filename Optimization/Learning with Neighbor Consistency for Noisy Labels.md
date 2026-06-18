# Learning with Neighbor Consistency for Noisy Labels

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문은 **Neighbor Consistency Regularization (NCR)** 을 제안합니다. 핵심 아이디어는 다음과 같습니다:

> *"특징 공간(feature space)에서 유사한 훈련 예제들은 유사한 예측값을 가져야 한다."*

노이즈 레이블 학습에서 모델의 예측 자체를 pseudo-label로 사용하는 기존 방식 대신, **특징 표현의 이웃 구조(neighborhood structure)** 를 활용하여 잘못된 레이블의 영향을 완화합니다.

### 주요 기여

| 기여 항목 | 내용 |
|----------|------|
| **방법론적 기여** | NCR이라는 단순한 추가 정규화 손실 항 제안 |
| **이론적 기여** | 전통적인 transductive label propagation의 inductive 버전으로 해석 가능 |
| **실험적 기여** | 합성 노이즈(CIFAR-10/100) 및 실제 노이즈(mini-WebVision, Clothing1M 등) 모두에서 경쟁력 있는 성능 달성 |
| **호환성** | Mixup 등 기존 정규화 기법과 결합 가능 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

딥러닝 모델은 **임의의 레이블도 암기(memorize)** 할 수 있다는 것이 알려져 있습니다 [Zhang et al., 2016]. 대규모 데이터 수집 과정에서 발생하는 노이즈 레이블 문제를 기존 방법들은 주로 다음 방식으로 해결하려 했습니다:

- **모델 예측 기반 pseudo-label 생성** → 과적합 위험
- **복수 모델 유지** (Co-teaching 등) → 복잡한 학습 절차
- **다단계 학습** → 구현 복잡도 증가

NCR은 이런 복잡성 없이 **단일 손실 항 추가**만으로 노이즈에 강건한 학습을 구현하고자 합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 기본 분류 손실 (Supervised Loss)

$$L_S(X, Y; \theta, W) := \frac{1}{m} \sum_{i=1}^{m} \ell(\sigma(\mathbf{z}_i), y_i) \tag{1}$$

- $m$: 미니배치 크기
- $\sigma$: softmax 함수
- $\mathbf{z}_i = h_W(\mathbf{v}_i)$: 분류기 출력 logit
- $\mathbf{v}\_i = g_\theta(x_i)$: 특징 추출기 출력 ($d$차원 벡터)
- $\ell(\mathbf{q}, \mathbf{p})$: cross-entropy 손실

#### Step 2: 이웃 간 유사도 정의

두 예제 $x_i$, $x_j$ 사이의 유사도는 **코사인 유사도**로 정의됩니다:

$$s_{i,j} = \cos(\mathbf{v}_i, \mathbf{v}_j) = \frac{\mathbf{v}_i^T \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|} \tag{*}$$

ReLU 비선형성 이후 특징값은 비음수이므로 $s_{i,j} \in [0, 1]$. 또한 자기 유사도는 $s_{i,i} = 0$으로 설정하여 지배적 영향 방지.

#### Step 3: NCR 손실 함수

$$L_{\text{NCR}}(X, Y; \theta, W) := \frac{1}{m} \sum_{i=1}^{m} D_{\text{KL}} \left( \sigma(\mathbf{z}_i / T) \,\Bigg\|\, \sum_{j \in \text{NN}_k(\mathbf{v}_i)} \frac{s_{i,j}}{\sum_k s_{i,k}} \cdot \sigma(\mathbf{z}_j / T) \right) \tag{3}$$

- $D_{\text{KL}}$: KL-divergence
- $T$: temperature (실험에서 $T=2$ 고정)
- $\text{NN}_k(\mathbf{v}_i)$: $i$번째 예제의 특징 공간에서 $k$개 최근접 이웃 집합
- $\frac{s_{i,j}}{\sum_k s_{i,k}}$: 유사도 기반 가중치 정규화 (합이 1이 되도록)

이 손실은 예제 $x_i$의 예측이 이웃들의 **가중 평균 예측**과 유사해지도록 강제합니다.

#### Step 4: 최종 손실 함수

$$L(X, Y; \theta, W) := (1 - \alpha) \cdot L_S(X, Y; \theta, W) + \alpha \cdot L_{\text{NCR}}(X, Y; \theta, W) \tag{4}$$

- $\alpha \in [0, 1]$: NCR 손실의 영향도 제어 하이퍼파라미터
- 실험에서 노이즈가 있는 경우 $\alpha = 0.9$가 최적

#### 참고: Label Propagation과의 대응 관계

고전적 label propagation [Zhou et al., 2003]은 다음 목적함수를 최소화합니다:

$$\mathcal{Q}(P) = \frac{1}{2}\mu \sum_{i=1}^{n} \|P_i - Y_i\|^2 + \frac{1}{2} \sum_{i,j=1}^{n} W_{ij} \left\| \frac{1}{\sqrt{D_{ii}}} P_i - \frac{1}{\sqrt{D_{jj}}} P_j \right\|^2 \tag{2}$$

NCR의 $L_S$는 식 (2)의 **fitting constraint** 항에, $L_{\text{NCR}}$은 **smoothing constraint** 항에 각각 대응됩니다.

#### 관련 방법들과의 비교

**Bootstrapping Loss [Reed et al., 2015]:**

$$L_B(X, Y; \theta, W) := \frac{1}{m} \sum_{i=1}^{m} \left[ (1-\alpha) \cdot \ell(\sigma(\mathbf{z}_i), y_i) + \alpha \cdot \ell(\sigma(\mathbf{z}_i), \sigma_B(\mathbf{z}_i)) \right] \tag{5}$$

→ NCR은 모델 예측 대신 **이웃 표현**으로부터 bootstrapping

**Label Smoothing [Szegedy et al., 2016]:**

$$L_{\text{LS}}(X, Y; \theta, W) := \frac{1}{m} \sum_{i=1}^{m} \left[ (1-\alpha) \cdot \ell(\sigma(\mathbf{z}_i), y_i) + \alpha \cdot \ell\left(\sigma(\mathbf{z}_i), \frac{1}{C}\mathbf{1}\right) \right] \tag{6}$$

→ NCR은 균등 분포 대신 **이웃 기반 분포**로 대체한 변형으로 볼 수 있음

---

### 2.3 모델 구조

```
입력 이미지 x_i
      ↓
[특징 추출기 g_θ] → v_i (d차원 특징 벡터)
      ↓                    ↓
[분류기 h_W]          [코사인 유사도 계산]
      ↓                    ↓
    z_i              이웃 집합 NN_k(v_i)
      ↓                    ↓
  Cross-Entropy      NCR Loss (KL-div)
      ↓                    ↓
         최종 손실 L = (1-α)L_S + α·L_NCR
```

- **백본**: ResNet-18 (CIFAR, mini-ImageNet), ResNet-50 (WebVision, Clothing1M)
- **분류기**: 일반 dot-product linear classifier (대부분), cosine classifier (mini-WebVision, WebVision)
- **추가 계산 비용**: $O(m^2(d+c))$ — GPU의 행렬 곱 최적화 덕분에 실용적
- **배치 크기 요구사항**: 클래스 수 이상의 배치 크기 필요 (WebVision: 1024)

---

### 2.4 성능 향상

#### 합성 노이즈 (CIFAR-10, CIFAR-100)

| Method | CIFAR-10 20% | CIFAR-10 80% | CIFAR-100 20% | CIFAR-100 80% |
|--------|-------------|-------------|--------------|--------------|
| Standard | 83.9 | 25.9 | 61.5 | 10.4 |
| ELR+ | 94.9 | 90.9 | 76.3 | 57.2 |
| **Ours+ (NCR+ELR)** | **95.2** | **91.6** | **76.6** | **58.0** |

#### 실제 노이즈 (Realistic Noise)

| Method | mini-WebVision | Clothing1M |
|--------|---------------|-----------|
| Standard | 75.8 | 71.7 |
| NCR | 77.1 | 74.4 |
| NCR+Mixup+DA | **80.5** | 74.6 |
| CleanNet | — | **74.7** |

- mini-WebVision에서 이전 최고 대비 **+1.2%** 향상
- mini-ImageNet-Red에서 Standard 대비 최대 **+4.9%** 향상

#### 0% 노이즈에서의 성능

흥미롭게도 노이즈가 없는 경우에도 NCR이 성능을 향상시킵니다 (mini-ImageNet-Red 0%: 70.9% → 72.1%). 이는 **일반적인 정규화 효과**가 있음을 의미합니다.

---

### 2.5 한계점

논문에서 명시적으로 언급한 한계:

1. **초기 특징 표현 의존성**: NCR은 적절한 특징 표현이 이미 학습되어 있다고 가정합니다. 이를 극복하기 위해 $e$ epoch 동안 NCR 없이 사전 훈련하지만, 이 추가 하이퍼파라미터를 제거하는 것이 미래 연구 과제.

2. **극단적 노이즈 취약성**: mini-ImageNet-Blue 80% 노이즈에서 모델이 underfitting 발생 (clean/noisy 예제 모두 신뢰도 ≈ 0).

3. **배치 크기 민감성**: 클래스 수만큼의 배치 크기가 필요하여 대규모 클래스 분류에서 메모리 비용 증가.

4. **상관 노이즈 처리 한계**: mini-ImageNet-Red처럼 노이즈 클래스와 진짜 클래스가 시각적으로 유사한 경우 (상관된 노이즈) clean/noisy 분리가 어려움.

5. **노이즈 비율 사전 지식 불필요하지만 하이퍼파라미터 튜닝 필요**: 각 노이즈 비율별로 $\alpha$, $k$, $e$ 튜닝이 필요하며, 실제 환경에서는 노이즈 비율을 모를 수 있음.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 메커니즘 분석

NCR이 일반화 성능을 높이는 메커니즘은 다음 세 가지 관점에서 설명할 수 있습니다:

#### (A) 특징 공간의 클래스 분리도 향상

논문의 Figure 4는 NCR 학습 후 mini-ImageNet-Blue에서 **within-class 유사도 분포**와 **between-class 유사도 분포**가 훨씬 명확하게 분리됨을 보여줍니다. 이는:

$$\text{Class Separability} \propto \frac{\mu_{\text{within-class}} - \mu_{\text{between-class}}}{\sigma_{\text{pooled}}}$$

이 분리도가 높아질수록 새로운 unseen 데이터에 대한 일반화 성능이 향상됩니다.

#### (B) Noisy Label의 영향 감쇄

NCR 손실 $L_{\text{NCR}}$은 예제 $x_i$의 레이블 $y_i$가 noisy하더라도, 이웃 $x_j$들의 **집합적 예측**을 통해 올바른 방향으로 유도합니다:

$$\hat{y}_i^{\text{NCR}} = \sum_{j \in \text{NN}_k(\mathbf{v}_i)} \frac{s_{i,j}}{\sum_k s_{i,k}} \cdot \sigma(\mathbf{z}_j / T)$$

이 가중 평균은 이웃의 다수결 효과로 노이즈를 자연스럽게 완화합니다.

#### (C) Inductive vs Transductive 학습

기존 Label Propagation은 **transductive** (학습 시 본 데이터만 분류 가능)이지만, NCR은 **inductive** (새로운 데이터에도 적용 가능)입니다. 이는 다음을 의미합니다:

- 모델 $f_{\theta,W}$가 학습 후에도 임의의 새 입력 $x_{\text{test}}$에 직접 적용 가능
- 테스트마다 새 그래프 $W$를 재구성할 필요 없음

#### (D) 과적합 방지 효과

Figure 3에서 Standard 모델은 noisy 예제에 대해 $p \approx 1$ (완전 과적합)을 보이는 반면, NCR은 noisy 예제에 $p \approx 0$을 부여하여 **memorization을 효과적으로 억제**합니다.

### 3.2 0% 노이즈에서의 일반화 개선

논문은 노이즈가 전혀 없는 경우에도 NCR이 성능을 향상시킨다고 보고합니다:

| Dataset | Standard | NCR |
|---------|---------|-----|
| mini-ImageNet-Red 0% | 70.9% | **72.1%** |
| mini-ImageNet-Blue 0% | 72.7% | **73.4%** |

이는 NCR이 단순한 노이즈 처리 기법을 넘어 **일반적인 정규화기(general regularizer)** 로 기능함을 시사합니다. 인접한 예제들의 예측 일관성을 강제함으로써 결정 경계(decision boundary)가 더 매끄럽게 형성됩니다.

### 3.3 Feature Embedding 품질 향상

특징 공간의 품질 향상은 downstream task (전이 학습, few-shot learning 등)에서의 일반화 성능에도 긍정적 영향을 미칠 가능성이 높습니다. NCR은 backpropagation을 통해 특징 추출기 $g_\theta$까지 gradient를 전달하여 특징 공간 자체를 개선합니다.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 주요 관련 연구 비교

| 방법 | 핵심 아이디어 | NCR과의 차이점 | Clothing1M |
|------|------------|--------------|-----------|
| **DivideMix [Li et al., ICLR 2020]** | GMM으로 clean/noisy 분리 후 반지도학습 | 복수 모델, 2단계 학습 필요 | 74.8% |
| **ELR+ [Liu et al., NeurIPS 2020]** | Early Learning Regularization (ELR) | 모델 예측 기반 정규화 | 74.8% |
| **MOIT+ [Ortego et al., 2020]** | 이웃 예측 비교로 noisy 탐지 + SSL | 탐지와 재학습 분리 단계 | — |
| **GJS [Englesson & Azizpour, 2021]** | Generalized Jensen-Shannon divergence | 데이터 증강 일관성만 활용 | — |
| **NCR (본 논문, 2022)** | 이웃 특징 유사도 기반 일관성 정규화 | 단일 단계, 단순 손실 추가 | 74.6% |

### 4.2 방법론적 비교

```
복잡도 측면:
DivideMix >> ELR+ > MOIT+ > GJS ≈ NCR

성능 측면 (mini-WebVision):
NCR+Mixup+DA (80.5%) > GJS (79.3%) > ELR+ (77.8†%) > DivideMix (76.3%) > ELR (76.3†%)
```

### 4.3 NCR의 차별성

1. **단순성**: 기존 방법들과 달리 추가 모델, 데이터 분리, 별도 학습 단계 불필요
2. **Inductive 특성**: DivideMix, MOIT+ 등은 학습 시 전체 데이터에 의존하는 반면 NCR은 미니배치 내에서 온라인으로 동작
3. **보완성**: ELR과 결합(NCR + ELR = "Ours+")하면 CIFAR 벤치마크에서 SOTA 달성
4. **노이즈 비율 추정 불필요**: DivideMix 등이 노이즈 비율 추정을 필요로 하는 반면 NCR은 불필요

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

#### (A) 방법론적 영향

1. **레이블 정보 전파 패러다임 확장**: NCR은 label propagation을 온라인/inductive 방식으로 발전시켰으며, 이 아이디어는 semi-supervised learning, few-shot learning 등에도 적용 가능합니다.

2. **특징 공간 활용의 재발견**: 모델 출력 대신 중간 특징 표현을 활용하는 방식이 과적합 저항성이 높다는 것을 실증하였으며, 이는 self-supervised learning과의 연결 가능성을 시사합니다.

3. **단순성의 가치 입증**: 복잡한 다단계 학습 없이도 경쟁력 있는 성능을 달성할 수 있음을 보여줌으로써, 실제 산업 적용 가능성이 높은 방법론의 방향을 제시합니다.

#### (B) 응용 영역 확장 가능성

- **의료 영상 분류**: 전문가 레이블링 비용이 높아 노이즈가 필연적인 영역
- **자연어 처리**: 텍스트 분류에서의 레이블 노이즈 문제에도 NCR 원리 적용 가능
- **반지도 학습**: 레이블 없는 데이터에 대한 일관성 제약으로 확장 가능
- **연속 학습(Continual Learning)**: 새로운 태스크 학습 시 이전 지식 보존에 활용 가능

### 5.2 앞으로 연구 시 고려할 점

#### (A) 미해결 문제

1. **초기화 하이퍼파라미터 $e$ 제거**: 논문 자체에서 미래 연구 과제로 명시. 자동으로 NCR 활성화 시점을 결정하는 메커니즘 (예: 특징 공간 성숙도 측정) 연구 필요.

   $$e^* = \arg\min_{e} \mathbb{E}[\text{Validation Error}(e)]$$

2. **극단적 노이즈(80% 이상) 처리**: 현재 NCR은 고노이즈 환경에서 underfitting 발생. 이를 해결하기 위한 **적응적 $\alpha$ 스케줄링** 연구 필요:

   $$\alpha(t) = \alpha_{\max} \cdot \left(1 - e^{-\lambda t}\right)$$

3. **OOD(Out-of-Distribution) 예제 탐지와의 결합**: 논문에서도 제안하듯이, OOD 예제를 먼저 걸러내고 NCR을 적용하면 더 강건한 학습이 가능할 것입니다.

#### (B) 스케일 확장

1. **대규모 클래스 분류**: 배치 크기 $\geq$ 클래스 수라는 요구사항이 1000개 이상 클래스 시나리오에서 메모리 병목. **계층적 이웃 탐색** 또는 **근사 최근접 이웃(ANN)** 방법론 도입 필요:

   $$O(m^2(d+c)) \rightarrow O(m \cdot k \cdot \log m)$$

2. **자기지도/대조학습과의 통합**: SimCLR, MoCo 등의 특징 공간 품질이 NCR의 효과를 극대화할 수 있습니다. Self-supervised pretraining + NCR fine-tuning 파이프라인 연구.

#### (C) 이론적 연구 필요 사항

1. **수렴 분석**: NCR 추가 시 SGD의 수렴 조건과 수렴 속도에 대한 이론적 분석 부재. 특히 $\alpha$와 수렴 속도의 관계 규명 필요.

2. **노이즈 모델 일반화**: 현재 NCR은 **noisy label이 feature space에서 random하게 분포**한다는 암묵적 가정에 의존. 구조적(structured) 노이즈나 adversarial 노이즈에 대한 이론적 분석 필요.

3. **최적 온도 $T$ 이론**: $T=2$를 경험적으로 고정하고 있으나, 이론적으로 최적 온도를 유도하는 연구가 필요합니다.

#### (D) 공정성 및 사회적 고려사항

논문 자체에서도 언급하였듯이:
- **웹 스크래핑 데이터의 편향(Bias)**: NCR이 노이즈 데이터에서 더 잘 학습할수록 데이터의 편향이 증폭될 위험
- **프라이버시**: 자동 수집 데이터 사용 시 저작권 및 동의 문제
- **공정성 평가 지표 추가**: 단순 정확도 외에 다양한 인구통계 그룹에 대한 공정성 메트릭 도입 필요

---

## 참고 자료

1. **Iscen, A., Valmadre, J., Arnab, A., & Schmid, C. (2022).** "Learning with Neighbor Consistency for Noisy Labels." *arXiv:2202.02200v2 [cs.CV]* — **본 논문 (직접 분석)**

2. **Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2003).** "Learning with Local and Global Consistency." *NeurIPS 2003.* — Label Propagation 원본

3. **Li, J., Socher, R., & Hoi, S. C. H. (2020).** "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." *ICLR 2020.*

4. **Liu, S., Niles-Weed, J., Razavian, N., & Fernandez-Granda, C. (2020).** "Early-Learning Regularization Prevents Memorization of Noisy Labels." *NeurIPS 2020.*

5. **Ortego, D., Arazo, E., Albert, P., O'Connor, N. E., & McGuinness, K. (2020).** "Multi-Objective Interpolation Training for Robustness to Label Noise." *arXiv:2012.04462.*

6. **Englesson, E., & Azizpour, H. (2021).** "Generalized Jensen-Shannon Divergence Loss for Learning with Noisy Labels." *arXiv:2105.04522.*

7. **Iscen, A., Tolias, G., Avrithis, Y., & Chum, O. (2019).** "Label Propagation for Deep Semi-supervised Learning." *CVPR 2019.*

8. **Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2016).** "Understanding Deep Learning Requires Rethinking Generalization." *arXiv:1611.03530.*

9. **Reed, S., Lee, H., Anguelov, D., Szegedy, C., Erhan, D., & Rabinovich, A. (2015).** "Training Deep Neural Networks on Noisy Labels with Bootstrapping." *ICLR 2015.*

10. **Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018).** "mixup: Beyond Empirical Risk Minimization." *ICLR 2018.*

> **⚠️ 정확도 주의사항**: 본 답변은 제공된 논문 PDF (arXiv:2202.02200v2)를 직접 분석하여 작성되었습니다. 2020년 이후 최신 연구 비교 부분(섹션 4)에서 언급된 외부 논문들의 세부 수치는 본 논문의 Table에 인용된 값을 기반으로 하였습니다. NCR 이후 발표된 추가 후속 연구(2022년 이후)에 대해서는 제공된 자료의 범위를 벗어나므로 별도 검색 없이는 확인이 어려워 이 부분의 비교는 논문 내 실험 결과에 한정하였습니다.
