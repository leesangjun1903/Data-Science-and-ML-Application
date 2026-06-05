# Simple and Effective Regularization Methods for Training on Noisily Labeled Data with Generalization Guarantee

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

과파라미터화된 딥 뉴럴 네트워크는 어떤 레이블도 완벽히 fitting할 수 있어, **노이즈 레이블 존재 시 일반화 성능이 저하**된다. 이 논문은 두 가지 간단한 정규화 방법이 이론적·실험적으로 노이즈 레이블 환경에서도 깨끗한 데이터 분포에 대한 **일반화 보장(generalization guarantee)**을 제공함을 증명한다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| 방법론 제안 | RDI(초기화까지의 거리 정규화)와 AUX(보조 변수 추가) 두 가지 정규화 방법 제안 |
| 이론적 분석 | NTK(Neural Tangent Kernel) 이론을 통해 두 방법이 kernel ridge regression과 동치임을 증명 |
| 일반화 보장 | 노이즈 레이블로 학습해도 깨끗한 분포에서의 일반화 bound 도출 (네트워크 크기 무관) |
| 실험적 검증 | MNIST, CIFAR-10에서 early stopping과 동등 혹은 그 이상의 성능 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**핵심 문제:** 과파라미터화된 신경망은 SGD 등 1차 최적화로 임의 레이블까지 완벽히 fitting 가능하다(Zhang et al., 2017). 이로 인해 노이즈 레이블이 존재할 때 test 성능이 급격히 저하된다.

**기존 해결책의 한계:**
- **Early stopping**: 실험적으로 효과적이나 이론적 설명 부재, 검증 곡선 모니터링 필요
- **노이즈 분포 추정, surrogate loss, meta-learning 등**: 복잡하고 추가 가정 필요

---

### 2.2 제안하는 방법 (수식 포함)

#### Method 1: RDI (Regularization by Distance to Initialization)

초기 파라미터 $\boldsymbol{\theta}(0)$에서의 거리를 정규화 항으로 추가:

$$L_{\lambda}^{\text{RDI}}(\boldsymbol{\theta}) = \frac{1}{2} \sum_{i=1}^{n} \left(f(\boldsymbol{\theta}, \boldsymbol{x}_i) - \tilde{y}_i\right)^2 + \frac{\lambda^2}{2} \|\boldsymbol{\theta} - \boldsymbol{\theta}(0)\|^2 \tag{3}$$

**직관:** 파라미터가 초기화로부터 멀리 이동할수록 패널티가 부과되어, 노이즈 레이블에 과적합하는 것을 방지한다.

#### Method 2: AUX (Adding Auxiliary Variable)

각 학습 샘플 $i$에 대해 훈련 가능한 보조 변수 $b_i \in \mathbb{R}$을 추가:

$$L_{\lambda}^{\text{AUX}}(\boldsymbol{\theta}, \boldsymbol{b}) = \frac{1}{2} \sum_{i=1}^{n} \left(f(\boldsymbol{\theta}, \boldsymbol{x}_i) + \lambda b_i - \tilde{y}_i\right)^2 \tag{4}$$

여기서 $\boldsymbol{b} = (b_1, \ldots, b_n)^\top \in \mathbb{R}^n$은 $\boldsymbol{b}(0) = \boldsymbol{0}$으로 초기화된다. **테스트 시에는 보조 변수 $b_i$를 제거하고 $f(\boldsymbol{\theta}, \cdot)$만 사용한다.**

**직관:** 보조 변수가 레이블 노이즈를 "흡수"하여, 신경망 자체는 깨끗한 패턴만 학습하도록 유도한다.

---

### 2.3 모델 구조 및 이론적 연결

#### NTK 근사

충분히 넓은 신경망에서 파라미터가 초기화 근방에 머물기 때문에 다음 1차 근사가 성립한다:

$$f(\boldsymbol{\theta}, \boldsymbol{x}) \approx f(\boldsymbol{\theta}(0), \boldsymbol{x}) + \langle \nabla_{\boldsymbol{\theta}} f(\boldsymbol{\theta}(0), \boldsymbol{x}), \boldsymbol{\theta} - \boldsymbol{\theta}(0) \rangle \tag{1}$$

$\phi(\boldsymbol{x}) = \nabla_{\boldsymbol{\theta}} f(\boldsymbol{\theta}(0), \boldsymbol{x})$로 정의하면 NTK는:

$$k(\boldsymbol{x}, \boldsymbol{x}') = \langle \phi(\boldsymbol{x}), \phi(\boldsymbol{x}') \rangle$$

초기 출력이 0에 가깝다고 가정하면:

$$f(\boldsymbol{\theta}, \boldsymbol{x}) \approx \phi(\boldsymbol{x})^\top (\boldsymbol{\theta} - \boldsymbol{\theta}(0)) \tag{5}$$

#### Theorem 4.1: Kernel Ridge Regression과의 동치성

두 방법 모두 gradient descent로 수렴 시 **동일한** kernel ridge regression 해를 학습함:

$$f^*(\boldsymbol{x}) = k(\boldsymbol{x}, \boldsymbol{X})^\top \left(k(\boldsymbol{X}, \boldsymbol{X}) + \lambda^2 \boldsymbol{I}\right)^{-1} \tilde{\boldsymbol{y}} \tag{8}$$

학습률 조건: $\eta \leq \frac{1}{\|k(\boldsymbol{X}, \boldsymbol{X})\| + \lambda^2}$

**핵심 관찰:** 정규화 없이 학습하면 $k(\boldsymbol{x}, \boldsymbol{X})^\top (k(\boldsymbol{X}, \boldsymbol{X}))^{-1} \tilde{\boldsymbol{y}}$ (노이즈 레이블 완벽 fitting), 정규화를 추가하면 kernel matrix에 $\lambda^2 \boldsymbol{I}$가 더해져 **kernel ridge regression** 해가 된다.

다중 출력의 경우 ($K$-class):

$$f^{(h)}(\boldsymbol{x}) = k(\boldsymbol{x}, \boldsymbol{X})^\top \left(k(\boldsymbol{X}, \boldsymbol{X}) + \lambda^2 \boldsymbol{I}\right)^{-1} \tilde{\boldsymbol{y}}^{(h)}, \quad h \in [K] \tag{9}$$

---

### 2.4 실험 설정 및 성능

| 설정 | 데이터셋 | 모델 | 결과 |
|------|----------|------|------|
| Setting 1 | MNIST ("5" vs "8") | 2-layer FC (10,000 hidden neurons) | GD+AUX ≈ GD+RDI ≈ early stopping |
| Setting 2 | CIFAR ("airplanes" vs "automobiles") | 11-layer CNN (192 channels each) | SGD+AUX, SGD+RDI > vanilla SGD |
| Setting 3 | CIFAR-10 (10 classes) | ResNet-34 | AUX가 early stopping보다 우수 |

**CIFAR-10 결과 (Table 1):**

| 방법 | Noise 0 | Noise 0.2 | Noise 0.4 | Noise 0.6 |
|------|---------|-----------|-----------|-----------|
| Normal CCE (early stop) | 94.05 | 89.73 | 86.35 | 79.13 |
| MSE+AUX (best) | **94.32** | **92.40** | **88.95** | **83.95** |
| Zhang & Sabuncu (2018) | — | 89.83 | 87.62 | 82.70 |

**주요 장점:** AUX를 사용하면 validation accuracy가 훈련 내내 단조 증가(monotone increasing)하여, early stopping 없이도 마지막 epoch 결과를 사용할 수 있다.

---

### 2.5 한계

1. **NTK 가정의 제약:** 이론은 충분히 넓은 신경망(NTK regime)에서만 성립. ResNet-34 (Setting 3)는 NTK regime에서 동작하지 않으나 실험적으로는 효과적 → 이론-실험 간 간극 존재
2. **$\lambda$ 하이퍼파라미터 튜닝 필요:** 최적 $\lambda$ 선택이 성능에 중요한 영향
3. **AUX의 메모리 오버헤드:** 훈련 샘플마다 보조 변수 $b_i$ 추가 필요
4. **SGD 환경에서의 이론 부재:** 이론은 full-batch GD에서 증명됨

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화 보장 이론 (Theorem 5.1: Additive Label Noise)

노이즈 레이블 $\tilde{y}_i = y_i + \varepsilon_i$ (평균 0, subgaussian, 파라미터 $\sigma$)에서:

$$\mathbb{E}_{(\boldsymbol{x},y)\sim\mathcal{D}}\left[\ell(f^*(\boldsymbol{x}), y)\right] \leq \frac{\lambda + O(1)}{2} \sqrt{\frac{\boldsymbol{y}^\top (k(\boldsymbol{X},\boldsymbol{X}))^{-1} \boldsymbol{y}}{n}} + O\!\left(\frac{\sigma}{\lambda}\right) + \Delta \tag{10}$$

$$\Delta = O\!\left(\sigma\sqrt{\frac{\log(1/\delta)}{n}} + \frac{\sigma}{\lambda}\sqrt{\frac{\log(1/\delta)}{n}} + \sqrt{\frac{\log\frac{n}{\delta\lambda}}{n}}\right)$$

**핵심 해석:**
- **Bound이 노이즈 레이블 $\tilde{\boldsymbol{y}}$가 아닌 깨끗한 레이블 $\boldsymbol{y}$에만 의존한다.**
- 노이즈 없이 학습 시 bound: $O\!\left(\sqrt{\frac{\boldsymbol{y}^\top(k(\boldsymbol{X},\boldsymbol{X}))^{-1}\boldsymbol{y}}{n}}\right)$
- 노이즈 있을 때: $\frac{\lambda + O(1)}{2} \cdot O\!\left(\sqrt{\frac{\boldsymbol{y}^\top(k(\boldsymbol{X},\boldsymbol{X}))^{-1}\boldsymbol{y}}{n}}\right) + O\!\left(\frac{\sigma}{\lambda}\right)$
- $\lambda = n^c$ ($c > 0$ 작은 상수)로 선택하면 두 번째 항 $O(\sigma/\lambda) \to 0$, 첫 번째 항은 $O(n^c)$ factor만큼만 증가 → **노이즈 무관 학습과 거의 동등한 bound**

### 3.2 이진 분류 일반화 보장 (Theorem 5.2)

노이즈율 $p$ ($0 \leq p < \frac{1}{2}$)에서 이진 분류 오류:

$$\Pr_{(\boldsymbol{x},y)\sim\mathcal{D}}[\text{sgn}(f^*(\boldsymbol{x})) \neq y] \leq \frac{\lambda + O(1)}{2}\sqrt{\frac{\boldsymbol{y}^\top(k(\boldsymbol{X},\boldsymbol{X}))^{-1}\boldsymbol{y}}{n}} + \frac{1}{1-2p} O\!\left(\frac{\sqrt{p}}{\lambda} + \sqrt{\frac{p\log\frac{1}{\delta}}{n}} + \sqrt{\frac{\log\frac{n}{\delta\lambda}}{n}}\right)$$

### 3.3 다중 클래스 분류 일반화 보장 (Theorem 5.3)

$$\Pr_{(\boldsymbol{x},c)\sim\mathcal{D}}\!\left[c \notin \arg\max_{h\in[K]} f^{(h)}(\boldsymbol{x})\right] \leq \frac{1}{\text{gap}}\!\left(\frac{\lambda+O(1)}{2}\sum_{h=1}^K \sqrt{\frac{(\boldsymbol{q}^{(h)})^\top(k(\boldsymbol{X},\boldsymbol{X}))^{-1}\boldsymbol{q}^{(h)}}{n}} + K \cdot O\!\left(\frac{1}{\lambda} + \sqrt{\frac{\log\frac{1}{\delta'}}{n}} + \sqrt{\frac{\log\frac{n}{\delta'\lambda}}{n}}\right)\right)$$

여기서 $\text{gap} = \min_{c,c' \in [K], c\neq c'} (p_{c,c} - p_{c',c})$이며, **bound이 깨끗한 레이블 인코딩 $\boldsymbol{Q} = \boldsymbol{P} \cdot \boldsymbol{Y}$에만 의존**한다.

### 3.4 일반화 성능 향상의 핵심 메커니즘

```
노이즈 레이블 훈련
        ↓
RDI or AUX 정규화
        ↓
NTK regime에서 Kernel Ridge Regression 동치
        ↓
λ²I 정규화 효과: 노이즈 성분 억제
        ↓
깨끗한 데이터 분포에서 일반화 보장
```

**Rademacher Complexity 기반 분석:**
- RKHS norm bound: $\|f^*\|_{\mathcal{H}} \leq \sqrt{\boldsymbol{y}^\top(k(\boldsymbol{X},\boldsymbol{X})+\lambda^2\boldsymbol{I})^{-1}\boldsymbol{y}} + \frac{\sigma}{\lambda}(\sqrt{n} + \sqrt{2\log(1/\delta)})$
- 경험적 Rademacher complexity: $\hat{\mathcal{R}}_S(\mathcal{F}_B) \leq \frac{B\sqrt{\text{tr}[k(\boldsymbol{X},\boldsymbol{X})]}}{n}$

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (1) NTK 이론의 실용적 응용 확장
이 논문은 NTK 이론을 **실용적 노이즈 레이블 학습**에 직접 연결한 선구적 연구로, NTK 프레임워크의 응용 범위를 크게 확장했다.

#### (2) 이론-실험 간극 연구 촉진
ResNet-34가 NTK regime 밖에서도 효과적이라는 관찰은 **"왜 NTK 이론 밖에서도 작동하는가?"** 라는 새로운 연구 질문을 제기한다.

#### (3) 정규화 방법론의 재해석
early stopping, weight decay 등 기존 정규화 기법들을 **kernel ridge regression의 관점**에서 통합적으로 이해하는 틀을 제공한다.

#### (4) 노이즈 레이블 학습의 이론적 기준선 제공
두 방법이 공식적인 generalization bound를 갖추고 있어, 이후 연구들이 이를 **이론적 baseline**으로 활용할 수 있다.

---

### 4.2 앞으로 연구 시 고려할 점

#### (1) NTK regime 가정의 완화
현실적인 신경망(ResNet, Transformer 등)은 NTK regime에서 동작하지 않는 경우가 많다. **feature learning regime**에서의 이론 분석이 필요하다.

#### (2) 더 현실적인 노이즈 모델
논문은 symmetric noise, class-conditional noise를 다루지만, 실제 데이터에서는:
- **Instance-dependent noise** (입력에 따라 다른 노이즈 확률)
- **Structured/adversarial noise** 등 더 복잡한 모델 고려 필요

#### (3) $\lambda$ 자동 선택
최적 $\lambda$를 데이터에 맞게 자동으로 선택하는 방법(예: cross-validation, Bayesian 접근) 연구가 필요하다. $\lambda = n^c$ 형태의 이론적 제안은 실용성이 낮다.

#### (4) SGD 환경에서의 이론 확장
현재 이론은 full-batch GD에 국한됨. **mini-batch SGD**에서의 이론적 분석이 필요하다.

#### (5) AUX의 메모리 효율화
대규모 데이터셋에서 $n$개의 보조 변수를 유지하는 것은 비효율적일 수 있어, **클러스터 기반 공유** 등의 효율화 방안 연구가 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> **주의:** 아래는 논문 내용과 제가 학습한 데이터(2021년 초까지)를 기반으로 한 분석이며, 일부 최신 논문의 세부 수치는 확인이 불가능할 수 있습니다. 확인 가능한 범위에서만 기술합니다.

### 5.1 관련 연구 흐름

#### (1) Instance-Dependent Noise 방향
- **Cheng et al., "Learning with Instance-Dependent Label Noise: A Sample Sieve Approach" (ICLR 2021)**: 인스턴스별 노이즈를 다루는 샘플 선별 방법 제안. 본 논문의 class-conditional 가정을 완화하는 방향.

#### (2) Semi-supervised Learning 통합
- **DivideMix (Li et al., ICLR 2020)**: 노이즈 샘플을 unlabeled 데이터로 처리하는 semi-supervised 접근. 본 논문보다 복잡하지만 더 높은 실험 성능 달성.

#### (3) 이론적 발전
- **Berthon et al., "Confidence Scores Make Instance-dependent Label-noise Learning Possible" (ICML 2021)**: 인스턴스 의존 노이즈에서의 식별 가능성(identifiability) 연구.

### 5.2 비교 요약

| 측면 | 본 논문 (Hu et al., 2020) | 2020년 이후 연구 동향 |
|------|--------------------------|----------------------|
| 방법 복잡도 | 매우 단순 (minimal modification) | 상대적으로 복잡 (두 네트워크, GMM 모델링 등) |
| 이론적 보장 | 명시적 generalization bound | 대부분 이론 보장 부족 |
| 노이즈 모델 | Symmetric, class-conditional | Instance-dependent noise로 확장 |
| 실험 성능 | 경쟁적이나 SOTA 대비 일부 격차 | 더 높은 절대 성능 |
| 적용 용이성 | 매우 높음 | 구현 복잡도 높음 |

---

## 참고자료

**주 논문:**
- Wei Hu, Zhiyuan Li, Dingli Yu. "Simple and Effective Regularization Methods for Training on Noisily Labeled Data with Generalization Guarantee." *ICLR 2020*. arXiv:1905.11368v4.

**논문 내 인용 주요 참고문헌:**
- Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural tangent kernel: Convergence and generalization in neural networks. arXiv:1806.07572.
- Arora, S. et al. (2019a). On exact computation with an infinitely wide neural net. arXiv:1904.11955.
- Arora, S. et al. (2019b). Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. arXiv:1901.08584.
- Lee, J. et al. (2019). Wide neural networks of any depth evolve as linear models under gradient descent. arXiv:1902.06720.
- Zhang, C. et al. (2017). Understanding deep learning requires rethinking generalization. ICLR 2017.
- Zhang, Z. & Sabuncu, M. (2018). Generalized cross entropy loss for training deep neural networks with noisy labels. NeurIPS 2018.
- Li, M., Soltanolkotabi, M., & Oymak, S. (2019). Gradient descent with early stopping is provably robust to label noise. arXiv:1903.11680.
- Bartlett, P. L. & Mendelson, S. (2002). Rademacher and Gaussian complexities. JMLR.
- Han, B. et al. (2018). Co-teaching: Robust training of deep neural networks with extremely noisy labels. NeurIPS 2018.
- Neyshabur, B. et al. (2019). The role of over-parametrization in generalization of neural networks. ICLR 2019.
