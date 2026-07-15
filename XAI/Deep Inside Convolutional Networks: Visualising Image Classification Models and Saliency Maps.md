# Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
Simonyan et al. (2014)은 학습된 딥 컨볼루션 네트워크(ConvNet)의 **내부 표현을 시각화**할 수 있으며, 이를 통해 모델이 각 클래스에 대해 무엇을 학습했는지 이해할 수 있다고 주장합니다. 핵심은 **입력 이미지에 대한 클래스 점수의 그래디언트(gradient)**를 계산함으로써 두 가지 시각화가 가능하다는 것입니다.

### 3가지 주요 기여

| 기여 | 설명 |
|------|------|
| **클래스 모델 시각화** | 클래스 점수를 최대화하는 이미지를 수치적으로 생성 → 모델이 학습한 클래스 개념 시각화 |
| **이미지별 살리언시 맵** | 단일 역전파로 특정 이미지에서 클래스 관련 픽셀 중요도 계산 → 약지도 객체 분할에 활용 |
| **DeconvNet과의 관계 규명** | 그래디언트 기반 시각화가 DeconvNet(Zeiler & Fergus, 2013)의 일반화임을 이론적으로 증명 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

딥러닝 모델, 특히 ConvNet은 높은 성능을 보이지만 **"블랙박스(black-box)"** 특성으로 인해 모델 내부에서 무슨 일이 일어나는지 이해하기 어렵습니다. 구체적으로:

- 모델이 특정 클래스에 대해 **어떤 시각적 패턴**을 학습했는가?
- 특정 입력 이미지에서 **어느 픽셀이 분류 결정에 중요**한가?
- 별도의 분할(segmentation) 레이블 없이 **약지도(weakly supervised) 방식**으로 객체 위치를 파악할 수 있는가?

---

### 2.2 제안하는 방법 (수식 포함)

#### 방법 1: 클래스 모델 시각화 (Class Model Visualisation)

클래스 $c$에 대한 점수 $S_c(I)$를 최대화하는 $L_2$-정규화된 이미지를 탐색:

$$\arg\max_{I} \; S_c(I) - \lambda \|I\|_2^2 \tag{1}$$

- $\lambda$: 정규화 파라미터
- 최적화는 역전파(back-propagation)로 수행 (가중치는 고정, 입력 이미지를 최적화)
- 초기값: 영(zero) 이미지에서 시작 후 훈련 평균 이미지 추가
- **Softmax 사후확률** $P_c$가 아닌 **비정규화 점수** $S_c$ 사용 → 다른 클래스 점수 억제 방지

$$P_c = \frac{\exp S_c}{\sum_{c'} \exp S_{c'}} \tag{softmax}$$

#### 방법 2: 이미지별 클래스 살리언시 맵 (Image-Specific Class Saliency)

선형 모델로의 1차 테일러 전개(first-order Taylor expansion):

$$S_c(I) \approx w^T I + b \tag{3}$$

여기서 $w$는 이미지 $I_0$ 근방에서의 편미분:

$$w = \frac{\partial S_c}{\partial I}\Bigg|_{I_0} \tag{4}$$

**살리언시 맵 $M \in \mathbb{R}^{m \times n}$ 계산:**

- **그레이스케일**: $M_{ij} = |w_{h(i,j)}|$
- **RGB(다채널)**: $M_{ij} = \max_c \; |w_{h(i,j,c)}|$

즉, 각 픽셀 $(i,j)$에서 모든 색상 채널에 걸친 그래디언트의 최대 절댓값을 취합니다.

#### 방법 3: 약지도 객체 분할 (Weakly Supervised Object Localisation)

1. 살리언시 맵에서 상위 **95% 분위수** 이상 픽셀 → 전경(foreground) GMM 추정
2. 하위 **30% 분위수** 이하 픽셀 → 배경(background) GMM 추정
3. **GraphCut** 색상 분할 적용 → 가장 큰 연결 전경 컴포넌트를 객체 마스크로 설정

---

### 2.3 모델 구조

논문에서 사용한 ConvNet은 AlexNet(Krizhevsky et al., 2012)과 유사하며, ILSVRC-2013 데이터셋(120만 이미지, 1000 클래스)으로 학습:

```
conv64 → conv256 → conv256 → conv256 → conv256
       → full4096 → full4096 → full1000
```

- **성능**: Top-1 / Top-5 오류율 = **39.7% / 17.7%** (단일 ConvNet 기준, AlexNet의 40.7%/18.2% 대비 소폭 향상)
- **추가 정규화**: 이미지 무작위 영역 제로화(zero-out) 기반 jittering 적용

#### DeconvNet과의 수학적 관계

| 레이어 | 역전파(본 논문) | DeconvNet(Zeiler) | 동등성 |
|--------|--------------|------------------|--------|
| Conv | $\frac{\partial f}{\partial X_n} = \frac{\partial f}{\partial X_{n+1}} \star \hat{K}_n$ | $R_n = R_{n+1} \star \hat{K}_n$ | **동일** |
| ReLU | $\frac{\partial f}{\partial X_n} = \frac{\partial f}{\partial X_{n+1}} \mathbf{1}(X_n > 0)$ | $R_n = R_{n+1} \mathbf{1}(R_{n+1} > 0)$ | **유사(차이 있음)** |
| Max-pooling | switch = $\arg\max_{q \in \Omega(p)} X_n(q)$ | switch 동일 사용 | **동일** |

ReLU에서만 차이 발생: 역전파는 **입력** $X_n$의 부호를 사용하고, DeconvNet은 **출력 재구성** $R_{n+1}$의 부호를 사용합니다.

---

### 2.4 성능 향상 및 한계

#### 성능 향상
- ILSVRC-2013 약지도 객체 위치 파악: Top-5 오류율 **46.4%** 달성
  - 완전지도 파트 기반 모델(Fisher vector) 대비 개선: **50.0% → 46.4%**
  - 단, 완전지도 우승자(29.9%)에는 미달
- 단일 역전파 패스만으로 살리언시 맵 계산 → **매우 빠른 계산**

#### 한계
1. **노이즈가 많은 시각화**: 클래스 모델 이미지가 인간에게 직관적으로 해석하기 어려움
2. **선형 근사의 한계**: 1차 테일러 전개는 국소적 근사로 전역적 해석에 부정확
3. **불완전한 객체 커버리지**: 살리언시 맵이 가장 판별적인 부분만 부각시켜 전체 객체를 놓칠 수 있음
4. **아키텍처 의존성**: 특정 ConvNet에 국한된 실험
5. **ReLU 처리의 불일치**: DeconvNet과의 완전한 등가 관계 미성립

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문이 모델의 일반화와 관련하여 직접·간접적으로 기여하는 측면들을 분석합니다.

### 3.1 정규화를 통한 일반화

클래스 모델 시각화 목적 함수(식 1)에서 $L_2$ 정규화 항 $-\lambda\|I\|_2^2$를 사용합니다. 이는 단순히 시각화 안정성을 위한 것이지만, **개념적으로 일반화를 위한 제약**을 가합니다:

$$\arg\max_{I} \; S_c(I) - \lambda \|I\|_2^2$$

- 과도하게 특이한(극단적인) 픽셀 패턴을 억제
- 더 일반적이고 대표적인 클래스 이미지를 생성

### 3.2 약지도 학습을 통한 일반화 가능성

살리언시 맵이 **별도의 분할 레이블 없이** 객체 위치를 파악한다는 점은 일반화 측면에서 중요합니다:

- 이미지 레이블만으로 학습된 모델이 픽셀 단위의 공간 정보를 암묵적으로 학습했음을 시사
- 이는 **데이터 효율적 학습(data-efficient learning)**의 가능성을 제시
- 추가 어노테이션 없이 새로운 태스크(분할, 탐지)로 지식 전이 가능

### 3.3 모델 디버깅을 통한 일반화 개선

살리언시 맵 시각화는 모델이 **잘못된 특징(spurious correlations)**에 의존하는지 검출하는 도구로 활용 가능:

- 예: 모델이 배경 컨텍스트에 과도하게 의존 → 도메인 변화에 취약
- 이를 감지하면 **데이터 증강**, **재학습**, **특징 선택 개선** 등으로 일반화 향상 도모
- 이는 이후 **XAI(Explainable AI) 기반 모델 개선** 연구의 토대가 됨

### 3.4 이미지 jittering의 일반화 기여

논문의 ConvNet 훈련에서 사용된 **무작위 영역 제로화(zero-out jittering)**는:

$$I_{\text{train}} = I \odot M_{\text{random}}, \quad M_{\text{random}} \in \{0,1\}^{m \times n}$$

- Dropout의 공간적 변형으로 볼 수 있음
- 모델이 특정 픽셀에 과의존하지 않도록 강제 → 일반화 성능 향상

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 후속 연구에 미친 영향

#### XAI(설명 가능한 AI) 분야의 토대 형성
이 논문은 현재 XAI의 주류를 이루는 여러 방법론들의 선구자입니다:

```
Simonyan et al. (2014) - Vanilla Gradient (살리언시 맵)
         ↓
Guided Backpropagation (Springenberg et al., 2015)
         ↓
Grad-CAM (Selvaraju et al., 2017)
         ↓
Integrated Gradients (Sundararajan et al., 2017)
         ↓
SHAP (Lundberg & Lee, 2017)
```

#### 약지도 학습(Weakly Supervised Learning) 발전 촉진
- CAM(Class Activation Mapping, Zhou et al., 2016)으로 직접 이어짐
- 분할/탐지 태스크에서 이미지 레이블만 사용하는 연구 흐름 형성

### 4.2 앞으로 연구 시 고려할 점

#### (1) 살리언시 맵의 신뢰성 문제
- **Adversarial 취약성**: Ghassemi et al. (2020) 등의 연구에서 살리언시 맵이 입력의 미세한 변화에도 크게 달라질 수 있음을 지적
- **고려할 점**: 살리언시 맵의 **안정성(stability)**과 **일관성(consistency)** 검증 메트릭 개발 필요

#### (2) 인과성 vs. 상관성 구분
- 살리언시 맵은 상관 관계(correlation)를 보여주지만 인과 관계(causality)는 보장하지 않음
- **고려할 점**: 반사실적(counterfactual) 설명 방법론과의 결합 연구

#### (3) 정량적 평가 기준 부재
- 시각화의 질을 평가하는 객관적 기준이 없음
- **고려할 점**: Pointing Game, Insertion/Deletion 메트릭 등 정량적 평가 표준화

#### (4) 트랜스포머 아키텍처로의 확장
- 본 논문은 CNN에 국한되나, 현재 주류는 Vision Transformer(ViT)
- **고려할 점**: Attention 기반 모델에서의 시각화 방법론 재정립

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 후속 연구 비교표

| 연구 | 방법 | 핵심 개선점 | 한계 극복 |
|------|------|------------|----------|
| **Simonyan et al. (2014)** | Vanilla Gradient | 그래디언트 기반 시각화 최초 체계화 | 노이즈 많음, 선형 근사 |
| **Selvaraju et al., Grad-CAM (2017)** | 마지막 Conv 레이어 그래디언트 GAP | 클래스 판별적 위치 강조, 고해상도 | 마지막 레이어만 |
| **Sundararajan et al., Integrated Gradients (2017)** | $\int_0^1 \frac{\partial F(x'+\alpha(x-x'))}{\partial x_i} d\alpha$ | 공리(axioms) 기반 이론적 보장 | 기준 이미지 선택 의존 |
| **Chefer et al., Transformer Explainability (2021)** | Attention + 그래디언트 결합 | ViT에 적용 가능 | 계산 복잡도 |
| **Ghassemi et al. (2020)** | 살리언시 신뢰성 분석 | 살리언시의 취약성 체계적 분석 | - |
| **Fel et al., CRAFT (2023)** | 개념 기반 설명 | 인간 해석 가능한 개념 단위 설명 | 개념 정의 주관성 |

### 5.2 Integrated Gradients와의 비교

Integrated Gradients(Sundararajan et al., 2017)는 Simonyan의 방법을 이론적으로 발전시킨 대표적 연구입니다:

$$\text{IntegratedGrad}_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha(x-x'))}{\partial x_i} d\alpha$$

- $x'$: 기준(baseline) 이미지 (일반적으로 영 이미지)
- **공리 만족**: 완전성(Completeness), 민감성(Sensitivity), 구현 불변성(Implementation Invariance)
- Simonyan의 방법은 $\alpha=1$에서의 단일 그래디언트 평가에 해당하는 특수 케이스

### 5.3 Grad-CAM과의 비교

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial S_c}{\partial A_{ij}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

- $A^k$: $k$번째 특징 맵
- Simonyan의 픽셀 수준 그래디언트 대신 **특징 맵 수준의 가중 평균** 사용
- 더 매끄럽고 클래스 판별적인 지도 생성

### 5.4 Vision Transformer 시대의 살리언시 (2021~)

Chefer et al. (2021) "Transformer Interpretability Beyond Attention Visualization"은 Attention 행렬과 그래디언트를 결합:

$$\tilde{A}^{(l)} = \mathbb{E}_h \left(\nabla A^{(l)}_h \odot A^{(l)}_h\right)^+$$

Simonyan의 그래디언트 개념이 Transformer 구조에도 지속적으로 영향을 미치고 있음을 보여줍니다.

---

## 결론 요약

Simonyan et al. (2014)는 딥러닝 해석 가능성 연구의 **초석**으로, 다음을 확립했습니다:
1. 역전파 그래디언트를 시각화 도구로 활용하는 패러다임
2. 약지도 객체 분할로의 응용 가능성
3. DeconvNet의 이론적 일반화

이 논문의 한계(노이즈, 선형 근사, 정량적 평가 부재)는 이후 10년간의 XAI 연구 아젠다를 형성했으며, 현재도 **신뢰할 수 있는 AI 설명 방법론** 개발이라는 핵심 과제로 이어지고 있습니다.

---

## 참고자료

1. **Simonyan, K., Vedaldi, A., & Zisserman, A. (2014).** Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps. *arXiv:1312.6034v2*
2. **Zeiler, M. D., & Fergus, R. (2014).** Visualizing and Understanding Convolutional Networks. *ECCV 2014* (arXiv:1311.2901)
3. **Selvaraju, R. R., et al. (2017).** Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*
4. **Sundararajan, M., Taly, A., & Yan, Q. (2017).** Axiomatic Attribution for Deep Networks. *ICML 2017*
5. **Zhou, B., et al. (2016).** Learning Deep Features for Discriminative Localization. *CVPR 2016*
6. **Springenberg, J. T., et al. (2015).** Striving for Simplicity: The All Convolutional Net. *ICLR Workshop 2015*
7. **Chefer, H., Gur, S., & Wolf, L. (2021).** Transformer Interpretability Beyond Attention Visualization. *CVPR 2021*
8. **Ghassemi, M., et al. (2020).** The false hope of current approaches to explainable artificial intelligence in health care. *The Lancet Digital Health*
9. **Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).** ImageNet Classification with Deep Convolutional Neural Networks. *NIPS 2012*
10. **Erhan, D., et al. (2009).** Visualizing Higher-Layer Features of a Deep Network. *University of Montreal Technical Report*
