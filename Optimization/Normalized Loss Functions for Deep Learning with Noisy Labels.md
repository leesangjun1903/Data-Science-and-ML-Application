# Normalized Loss Functions for Deep Learning with Noisy Labels

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Ma et al., ICML 2020)은 두 가지 핵심 주장을 제시합니다:

1. **이론적 주장**: 단순한 정규화(normalization)를 적용하면, **어떤 손실 함수든 노이즈 레이블에 강인(robust)하게 만들 수 있다.**
2. **실용적 주장**: 단순히 강인함(robustness)만으로는 DNN을 정확하게 학습시키기에 충분하지 않으며, 기존의 강인한 손실 함수들은 **과소적합(underfitting)** 문제를 겪는다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| 이론적 통찰 | 정규화로 임의 손실 함수의 노이즈 내성 보장 |
| 문제 발견 | 기존 강인 손실 함수의 과소적합 문제 규명 |
| 프레임워크 제안 | Active Passive Loss (APL) 프레임워크 |
| 실증적 우월성 | 벤치마크에서 SOTA 대비 큰 폭의 성능 향상 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**노이즈 레이블 환경에서의 DNN 학습**은 두 가지 상충하는 요구사항을 가집니다:

- **강인성(Robustness)**: 잘못된 레이블에 의한 과적합을 방지
- **충분한 학습(Sufficient Learning)**: 정확한 분류 성능을 달성

기존 방법들의 한계:
- **CE 손실**: 노이즈 레이블에 취약하여 과적합 발생
- **MAE, RCE**: 이론적으로 강인하지만 수렴 속도가 느리고 과소적합 발생
- **GCE, SCE**: 부분적으로만 강인함 (partially robust)

### 2-2. 제안하는 방법 및 수식

#### (A) 정규화 손실 함수 (Normalized Loss)

임의의 손실 함수 $\mathcal{L}$에 대해 다음과 같이 정규화합니다:

$$\mathcal{L}_{\text{norm}} = \frac{\mathcal{L}(f(\boldsymbol{x}), y)}{\sum_{j=1}^{K} \mathcal{L}(f(\boldsymbol{x}), j)} \tag{1}$$

정규화된 손실은 $\mathcal{L}_{\text{norm}} \in [0, 1]$을 만족합니다.

**Normalized Cross Entropy (NCE)**:

$$\text{NCE} = \frac{-\sum_{k=1}^{K} q(k|\boldsymbol{x})\log p(k|\boldsymbol{x})}{-\sum_{j=1}^{K}\sum_{k=1}^{K} q(y=j|\boldsymbol{x})\log p(k|\boldsymbol{x})} = \log_{\prod_k^K p(k|\boldsymbol{x})} p(y|\boldsymbol{x}) \tag{2}$$

**Normalized MAE (NMAE)**:

$$\text{NMAE} = \frac{\sum_{k=1}^{K}|p(k|\boldsymbol{x}) - q(k|\boldsymbol{x})|}{\sum_{j=1}^{K}\sum_{k=1}^{K}|p(k|\boldsymbol{x}) - q(y=j|\boldsymbol{x})|} = \frac{1}{K-1}(1-p(y|\boldsymbol{x})) = \frac{1}{2(K-1)} \cdot \text{MAE} \tag{3}$$

**Normalized RCE (NRCE)**:

$$\text{NRCE} = \frac{-\sum_{k=1}^{K} p(k|\boldsymbol{x})\log q(k|\boldsymbol{x})}{-\sum_{j=1}^{K}\sum_{k=1}^{K} p(k|\boldsymbol{x})\log q(y=j|\boldsymbol{x})} = \frac{1}{K-1}(1-p(y|\boldsymbol{x})) = \frac{1}{A(K-1)} \cdot \text{RCE} \tag{4}$$

**Normalized Focal Loss (NFL)**:

$$\text{NFL} = \frac{-\sum_{k=1}^{K} q(k|\boldsymbol{x})(1-p(k|\boldsymbol{x}))^\gamma \log p(k|\boldsymbol{x})}{-\sum_{j=1}^{K}\sum_{k=1}^{K} q(y=j|\boldsymbol{x})(1-p(k|\boldsymbol{x}))^\gamma \log p(k|\boldsymbol{x})} = \log_{\prod_k^K (1-p(k|\boldsymbol{x}))^\gamma p(k|\boldsymbol{x})} (1-p(y|\boldsymbol{x}))^\gamma p(y|\boldsymbol{x}) \tag{5}$$

#### (B) 이론적 보장 (Lemmas)

**Lemma 1** (대칭 노이즈): 노이즈율 $\eta < \frac{K-1}{K}$이면, 모든 정규화 손실 함수 $\mathcal{L}_{\text{norm}}$은 대칭(균일) 레이블 노이즈에 내성을 가집니다.

**증명 핵심**:

$$R^\eta(f) = R(f)\left(1 - \frac{\eta K}{K-1}\right) + \frac{\eta}{K-1}$$

```math
\Rightarrow R^\eta(f^*) - R^\eta(f) = \left(1 - \frac{\eta K}{K-1}\right)(R(f^*) - R(f)) \leq 0
```

**Lemma 2** (비대칭 노이즈): $R(f^\*) = 0$이고 $0 \leq \mathcal{L}\_{\text{norm}}(f^*(\boldsymbol{x}), k) \leq \frac{1}{K-1}$인 경우, 노이즈율 $\eta_{jk} < 1 - \eta_y$이면 비대칭 노이즈에도 내성을 가집니다.

#### (C) Active Passive Loss (APL) 프레임워크

논문은 손실 함수를 두 가지로 분류합니다:

- **Active Loss**: 레이블 $y$에 해당하는 클래스의 확률만 명시적으로 최대화
  - $\forall (x,y) \in \mathcal{D},\ \forall k \neq y:\ \ell(f(x), k) = 0$
  - 예: CE, NCE, FL, NFL
- **Passive Loss**: 레이블 $y$ 외 다른 클래스의 확률도 명시적으로 최소화
  - $\forall (x,y) \in \mathcal{D},\ \exists k \neq y:\ \ell(f(x), k) \neq 0$
  - 예: MAE, NMAE, RCE, NRCE

**APL 공식**:

$$\mathcal{L}_{\text{APL}} = \alpha \cdot \mathcal{L}_{\text{Active}} + \beta \cdot \mathcal{L}_{\text{Passive}}, \quad \alpha, \beta > 0 \tag{6}$$

**Lemma 3**: $\mathcal{L}\_{\text{Active}}$와 $\mathcal{L}\_{\text{Passive}}$ 모두 노이즈 내성을 가지면, $\mathcal{L}\_{\text{APL}} = \alpha \cdot \mathcal{L}\_{\text{Active}} + \beta \cdot \mathcal{L}_{\text{Passive}}$도 노이즈 내성을 가집니다.

구체적인 APL 조합:
1. $\alpha \cdot \text{NCE} + \beta \cdot \text{MAE}$
2. $\alpha \cdot \text{NCE} + \beta \cdot \text{RCE}$
3. $\alpha \cdot \text{NFL} + \beta \cdot \text{MAE}$
4. $\alpha \cdot \text{NFL} + \beta \cdot \text{RCE}$

### 2-3. 모델 구조

논문은 새로운 네트워크 아키텍처를 제안하지 않고, **표준 DNN 아키텍처에 손실 함수 수준의 개입**만 합니다:

| 데이터셋 | 네트워크 |
|---|---|
| MNIST | 4-layer CNN |
| CIFAR-10 | 8-layer CNN |
| CIFAR-100 | ResNet-34 |
| WebVision | ResNet-50 |

### 2-4. 성능 향상

**CIFAR-10, 대칭 노이즈 60% ($\eta=0.6$)**:

| 방법 | 정확도 (%) |
|---|---|
| CE | 40.90 |
| NLNL (이전 SOTA) | 72.85 |
| **NFL+RCE (APL)** | **79.78** |
| **NCE+RCE (APL)** | **79.78** |

**CIFAR-100, 대칭 노이즈 80% ($\eta=0.8$)**:

| 방법 | 정확도 (%) |
|---|---|
| CE | 7.58 |
| GCE | 16.18 |
| NLNL | 11.01 |
| **NCE+MAE (APL)** | **25.50** |
| **NCE+RCE (APL)** | **25.80** |

**WebVision (실세계 노이즈)**:

| 방법 | Top-1 정확도 (%) |
|---|---|
| CE | 58.88 |
| GCE | 53.68 |
| SCE | 61.76 |
| **NCE+MAE** | **62.36** |
| **NCE+RCE** | **62.64** |

### 2-5. 한계

1. **하이퍼파라미터 의존성**: $\alpha, \beta$ 튜닝이 필요하며, 데이터셋 복잡도에 따라 최적값이 크게 달라짐
2. **점근적 조건**: Lemma 2의 $R(f^*) = 0$ 가정은 실제로 항상 만족되지 않음
3. **노이즈 모델 가정**: 노이즈가 입력에 조건부 독립이라는 가정에 의존
4. **오픈셋 노이즈**: 완전히 새로운 클래스에서 발생하는 오픈셋 노이즈에 대한 분석 부재
5. **대규모 실험 부족**: WebVision 실험이 Mini 설정(50 클래스)에 한정됨

---

## 3. 일반화 성능 향상 가능성

### 3-1. APL이 일반화를 향상시키는 메커니즘

#### (A) 과적합-과소적합 동시 해결

- **Active Loss (NCE/NFL)**: 노이즈 레이블로 인한 **과적합 방지** (정규화 효과)
- **Passive Loss (MAE/RCE)**: 학습 신호를 보완하여 **과소적합 방지**

두 손실의 결합은 편향-분산 트레이드오프(bias-variance tradeoff)를 보다 균형 있게 조절합니다.

#### (B) 손실 지형(Loss Landscape) 관점

NCE 손실의 과소적합 원인을 분석하면:

$$\text{NCE} = \frac{P}{P+Q}, \quad P = -\log p_y, \quad Q = -\sum_{k\neq y}\log p_k$$

훈련 중 $P$가 고정되어도 $Q$가 증가할 수 있으며, 이는 $p_{k\neq y} = \frac{1-p_y}{K-1}$(최고 엔트로피)에서 최대가 됩니다. 이 경우 손실이 감소해도 모델이 실제로 학습하지 못합니다.

Passive Loss는 이 $Q$ 항을 명시적으로 최소화하여 **올바른 방향의 gradient를 제공**합니다.

#### (C) 노이즈 내성과 일반화의 관계

정규화 손실의 노이즈 내성 조건 하에서:

```math
R^\eta(f^*) - R^\eta(f) = \left(1 - \frac{\eta K}{K-1}\right)(R(f^*) - R(f)) \leq 0
```

이는 **노이즈 환경의 최적해가 클린 환경의 최적해와 일치**함을 보장합니다. 이는 단순히 훈련 정확도가 아닌 **실제 분포(clean distribution)에 대한 일반화 성능**을 보장합니다.

#### (D) 복잡한 데이터셋에서의 일반화

실험 결과, 데이터셋이 복잡할수록 (CIFAR-100 > CIFAR-10):
- Active loss의 비중 $\alpha$를 늘려야 함 (더 많은 판별 학습)
- Passive loss의 비중 $\beta$를 줄여야 함

이는 **데이터 복잡도에 따른 일반화 전략이 다르다**는 중요한 인사이트입니다.

#### (E) WebVision에서의 실세계 일반화

훈련은 노이즈 레이블로 하였으나, 평가는 **클린 ILSVRC12 검증셋**으로 수행하여, APL이 단순히 훈련 세트가 아닌 실제 분포에 대한 일반화 성능도 향상시킴을 실증적으로 검증합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려점

### 4-1. 앞으로의 연구에 미치는 영향

#### (A) 손실 함수 설계 패러다임의 전환

APL 프레임워크는 손실 함수 설계를 "**강인성 vs. 학습 효율**"의 이분법에서 "**두 성질의 상호 보완적 결합**"으로 전환시킵니다. 향후 연구에서는:
- 새로운 Active/Passive 손실 쌍의 발굴
- 동적으로 $\alpha, \beta$를 조정하는 적응형 APL 설계

#### (B) 이론적 토대 확장

정규화를 통한 임의 손실의 강인화 정리는:
- 새로운 손실 함수를 이론적 강인성 보장과 함께 설계하는 기반 원리로 활용 가능
- 노이즈 내성 분석을 CE 이외 손실로 일반화하는 방향

#### (C) 다른 분야로의 적용

- **자연어 처리(NLP)**: 웹 크롤링 데이터의 노이즈 레이블
- **의료 이미지**: 어노테이터 간 의견 불일치(annotator disagreement)
- **페더레이션 러닝**: 비중앙화 환경에서의 레이블 품질 불균일

### 4-2. 앞으로 연구 시 고려할 점

1. **인스턴스 의존 노이즈(Instance-dependent Noise)**: 본 논문의 이론은 노이즈가 입력에 독립적이라고 가정하나, 실제로는 입력 특성에 따라 노이즈가 발생할 수 있음
2. **하이퍼파라미터 자동화**: $\alpha, \beta$ 선택을 메타러닝이나 베이지안 최적화로 자동화
3. **오픈셋/롱테일 노이즈**: 훈련 중 보지 못한 클래스에서 오는 노이즈에 대한 대응
4. **Semi-supervised 학습과의 결합**: 일부 클린 레이블과 다수 노이즈 레이블이 혼재하는 환경
5. **대형 사전학습 모델(LLM/ViT)**: Transformer 기반 모델에서 APL의 효과 검증

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 이하의 최신 연구 내용은 제 훈련 데이터에 기반한 것이며, 제공된 PDF 외의 논문에 대해서는 세부 수치의 정확성을 100% 보장하기 어렵습니다. 잘 알려진 연구들의 주요 방향을 기술합니다.

### 5-1. 주요 후속 연구 방향

#### (A) 샘플 선택 기반 방법과의 결합

**DivideMix** (Li et al., NeurIPS 2020)는 Gaussian Mixture Model로 클린/노이즈 샘플을 분리하고, semi-supervised learning을 적용합니다. APL과 달리 복잡한 훈련 파이프라인이 필요하지만 높은 성능을 보입니다.

비교:
| 측면 | APL | DivideMix |
|---|---|---|
| 구현 복잡도 | 낮음 (손실 함수 교체만) | 높음 (GMM + 두 네트워크) |
| 노이즈 모델 추정 필요 | 불필요 | 불필요 |
| 이론적 보장 | 있음 | 경험적 |

#### (B) 사전학습 모델 기반 접근

**Noisy Student** (Xie et al., CVPR 2020), **EfficientNet + Noisy Labels** 등의 연구는 대규모 사전학습을 통해 노이즈 레이블 영향을 줄이는 방향을 탐구합니다. APL은 이러한 사전학습 파인튜닝 단계에서도 손실 함수로 직접 적용 가능합니다.

#### (C) 인스턴스 의존 노이즈 대응

**Instance-Dependent Noise** 연구들(예: Cheng et al., 2021)은 APL의 이론적 가정인 "노이즈가 입력에 독립"을 깨는 더 현실적인 시나리오를 다룹니다. APL은 이 경우 이론적 보장이 약해질 수 있습니다.

#### (D) Contrastive Learning과의 결합

**SupCon** (Khosla et al., NeurIPS 2020) 등의 대조 학습은 노이즈 레이블 환경에서도 유용한 표현을 학습할 수 있음을 보였습니다. APL의 Active/Passive 구분을 대조 학습의 positive/negative pair 개념과 연결짓는 연구 방향이 흥미롭습니다.

### 5-2. 종합 비교표

| 방법 | 강인성 보장 | 과소적합 해결 | 구현 복잡도 | 추가 클린 데이터 필요 |
|---|---|---|---|---|
| CE | ❌ | ✅ | 낮음 | ❌ |
| MAE/RCE | ✅ | ❌ | 낮음 | ❌ |
| GCE | 부분적 | ✅ | 낮음 | ❌ |
| SCE | 부분적 | ✅ | 낮음 | ❌ |
| **APL (이 논문)** | **✅** | **✅** | **낮음** | **❌** |
| DivideMix | 경험적 | ✅ | 높음 | ❌ |
| MentorNet | 경험적 | ✅ | 높음 | ✅ |

---

## 참고 자료

**주요 참고 논문 (PDF에서 인용된 것들)**:
- **본 논문**: Ma, X., Huang, H., Wang, Y., Romano, S., Erfani, S., & Bailey, J. (2020). *Normalized Loss Functions for Deep Learning with Noisy Labels*. ICML 2020. arXiv:2006.13554
- Ghosh, A., Kumar, H., & Sastry, P. (2017). *Robust loss functions under label noise for deep neural networks*. AAAI 2017.
- Zhang, Z., & Sabuncu, M. R. (2018). *Generalized cross entropy loss for training deep neural networks with noisy labels*. NeurIPS 2018.
- Wang, Y., Ma, X., Chen, Z., Luo, Y., Yi, J., & Bailey, J. (2019). *Symmetric cross entropy for robust learning with noisy labels*. ICCV 2019.
- Kim, Y., Yim, J., Yun, J., & Kim, J. (2019). *NLNL: Negative learning for noisy labels*. CVPR 2019.
- Charoenphakdee, N., Lee, J., & Sugiyama, M. (2019). *On symmetric losses for learning from corrupted labels*. ICML 2019.
- Han, B., Yao, Q., Yu, X., et al. (2018). *Co-teaching: Robust training of deep neural networks with extremely noisy labels*. NeurIPS 2018.
- Patrini, G., et al. (2017). *Making neural networks robust to label noise: A loss correction approach*. CVPR 2017.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. CVPR 2016.

**2020년 이후 관련 연구 (일반적으로 알려진 것들)**:
- Li, J., Socher, R., & Hoi, S. C. H. (2020). *DivideMix: Learning with Noisy Labels as Semi-supervised Learning*. ICLR 2020.
- Khosla, P., et al. (2020). *Supervised Contrastive Learning*. NeurIPS 2020.
