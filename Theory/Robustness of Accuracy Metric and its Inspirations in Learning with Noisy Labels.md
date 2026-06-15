# Robustness of Accuracy Metric and its Inspirations in Learning with Noisy Labels

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문의 중심 주장은 **정확도(Accuracy) 지표 자체가 노이즈 레이블 학습에서 견고(robust)하다**는 것입니다. 구체적으로, 대각 우세(diagonally-dominant) 클래스 조건부 노이즈 하에서:

> *"노이즈 분포에서 정확도를 최대화하는 분류기는 클린 분포에서도 정확도를 최대화하도록 보장된다."*

### 주요 기여

| 기여 영역 | 내용 |
|-----------|------|
| **이론적 (훈련)** | Theorem 1 + Theorem 2를 통해, 충분히 많은 노이즈 샘플로 훈련 정확도를 최대화하면 근사적 최적 분류기를 얻을 수 있음을 증명 |
| **이론적 (검증)** | Theorem 3를 통해, 노이즈 검증 세트가 신뢰할 수 있는 모델 선택 도구임을 최초로 이론적으로 정당화 |
| **실험적** | NTS(Noisy best Teacher and Student) 프레임워크 제안 및 CIFAR-10/100, Clothing1M에서 성능 향상 입증 |
| **이론적 설명** | 데이터 증강이 노이즈 레이블 학습에서 일반화를 향상시키는 이유를 이론적으로 설명 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**기존 연구의 한계:**
- 기존 연구들은 robust loss function(GCE, DMI 등)을 제안하여 노이즈 레이블에 대응했지만, **정확도 지표 자체의 견고성**은 증명되지 않았음
- 노이즈 검증 세트를 사용하는 것이 실용적으로 활용되었지만(**Zhang and Sabuncu 2018; Nguyen et al. 2019; Xia et al. 2019, 2020**), **이론적 정당화가 부재**했음
- 하이퍼파라미터 튜닝과 early stopping을 위한 클린 검증 세트 가정은 현실적으로 불합리함

**해결 목표:**
1. 정확도 지표의 이론적 견고성 증명
2. 노이즈 샘플만으로 신뢰할 수 있는 모델 선택 방법 제시
3. 노이즈 레이블 학습의 본질적 메커니즘 설명

---

### 2.2 수학적 배경 및 주요 수식

#### 기본 설정

$K$-클래스 분류에서 노이즈 전이 행렬(Noise Transition Matrix) $T \in [0,1]^{K \times K}$:

$$\Pr[\tilde{Y} = j | Y = i] = T_{i,j} \tag{1}$$

노이즈율(noise rate):

$$\varepsilon = 1 - \sum_{i \in \mathcal{Y}} \Pr[Y=i] T_{i,i}$$

**클린 분포에서의 정확도:**

$$A_D(h) := \mathbb{E}_{(X,Y)\sim D}[\mathbf{1}(h(X) = Y)] = \Pr[h(X) = Y] \tag{2}$$

**훈련 정확도 (노이즈 샘플 $\tilde{S} = \{(x_i, \tilde{y}\_i)\}_{i=1}^m$):**

$$A_{\tilde{S}}(h) := \frac{1}{m}\sum_{i=1}^{m}\mathbf{1}(h(x_i) = \tilde{y}_i) \tag{3}$$

**노이즈 분포에서의 정확도:**

$$A_{\tilde{D}}(h) := \mathbb{E}_{(X,\tilde{Y})\sim\tilde{D}}[\mathbf{1}(h(X) = \tilde{Y})] = \Pr[h(X) = \tilde{Y}] \tag{4}$$

**혼동 행렬(Confusion Matrix):**

$$C_{i,j}(h) := \Pr[h(X) = j | Y = i] \tag{5}$$

---

#### Theorem 1: 정확도 지표의 견고성

**가정:**
- **Assumption 1**: $X$는 $Y$를 conditioning하면 $\tilde{Y}$와 독립 (클래스 조건부 노이즈)
- **Assumption 2**: $T$는 대각 우세 행렬, 즉 $\forall i, T_{i,i} > \max_{j \neq i} T_{i,j}$

**결론 (i):**

$$\max_{h \in \mathcal{H}} A_{\tilde{D}}(h) = A_{\tilde{D}}(h^*) = 1 - \varepsilon \tag{6}$$

즉, 노이즈 분포에서의 최대 정확도는 $1-\varepsilon$이며, 전역 최적 분류기 $h^*$만이 이를 달성함.

**결론 (ii):**

$$A_{\tilde{D}}(h) \to \max_{h \in \mathcal{H}} A_{\tilde{D}}(h) \Rightarrow A_D(h) \to 1 \tag{7}$$

**수렴 속도 (Appendix A):**

$$0 \leq 1 - A_D(h) \leq \frac{1}{\min_{i,j \in \mathcal{Y}, j\neq i}(T_{i,i} - T_{i,j})} \cdot \left(\max_{h \in \mathcal{H}} A_{\tilde{D}}(h) - A_{\tilde{D}}(h)\right) \tag{13}$$

**증명의 핵심 (Lemma 1 활용):**

$$A_{\tilde{D}}(h) = \sum_{i \in \mathcal{Y}}\left(\Pr[Y=i]\sum_{j \in \mathcal{Y}} T_{i,j} \cdot C_{i,j}(h)\right) \leq \sum_{i \in \mathcal{Y}} \Pr[Y=i] \cdot T_{i,i} = 1 - \varepsilon \tag{11}$$

등호는 $C(h) = I$ (단위행렬), 즉 $h$가 전역 최적 분류기일 때만 성립.

---

#### Theorem 2: 일반화 한계 (Generalization Bound)

VC 차원 $d_{VC}(\mathcal{H})$를 가지는 가설 공간에서, 확률 $1-\delta$ 이상으로:

$$A_{\tilde{D}}(h) - A_{\tilde{S}}(h) \geq -\sqrt{\frac{8\left(d_{VC} \cdot (\ln(2m/d_{VC}) + 1) + \ln(4/\delta)\right)}{m}} \tag{8}$$

**의미:** 훈련 샘플 수 $m$이 충분히 크면 bound가 0에 수렴 → 훈련 정확도를 최대화하면 노이즈 분포에서의 정확도도 근사적으로 최대화됨 → Theorem 1에 의해 전역 최적 분류기에 근접.

---

#### Theorem 3: 노이즈 검증 한계 (Validation Bound)

노이즈 검증 세트 $\tilde{V} = \{(x_i, \tilde{y}\_i)\}_{i=1}^n$에서의 검증 정확도:

$$A_{\tilde{V}}(h) := \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}(h(x_i) = \tilde{y}_i) \tag{9}$$

Hoeffding 부등식을 이용하여, 확률 $1-\delta$ 이상으로:

$$A_{\tilde{D}}(h) - A_{\tilde{V}}(h) \geq -\sqrt{\frac{\ln(1/\delta)}{2n}} \tag{10}$$

**수치 예시:** $n=1000$, $\delta=0.01$이면 gap $\leq 0.048$ (확률 0.99 이상).

---

### 2.3 모델 구조: NTS 프레임워크

```
[노이즈 훈련 데이터]
        ↓
   Teacher 훈련 (CE/GCE/Co-T/DMI 등)
        ↓
   노이즈 검증 세트로 최고 정확도 모델 선택
   → Noisy best Teacher (NT)
        ↓
   NT의 예측 레이블로 Student 훈련
        ↓
   노이즈 검증 세트로 최고 정확도 모델 선택
   → Noisy best Student (NS)
```

**핵심 특징:**
- 추가적인 클린 샘플 불필요
- 별도의 복잡한 훈련 전략 없이 cross-entropy loss로 구현 가능
- Theorem 1 & 3에 의해 노이즈 검증 세트의 신뢰성이 이론적으로 보장

**실험 환경:**
- CIFAR-10/100: Wide ResNet-28-10 (WRN-28-10)
- Clothing1M: ResNet-50 (ImageNet 사전학습)

---

### 2.4 성능 향상

#### CIFAR-10 (WRN-28-10, CE 기반)

| 설정 | Last | NT | NS |
|------|------|----|----|
| Uniform 0.4 | 74.18 | 90.09 | **93.76** |
| Uniform 0.6 | 59.15 | 82.48 | **87.88** |
| Asymmetric 0.4 | 79.47 | 89.07 | **91.49** |

#### Clothing1M (ResNet-50)

| 방법 | Last (Noisy Val) | NT (Noisy Val) | NS (Noisy Val) |
|------|-----------------|-----------------|-----------------|
| CE | 68.05 | 69.12 | **70.21** |
| DMI | 72.01 | 72.18 | **72.82** |
| DivideMix (ensemble) | 73.84 | 74.18 | **74.36** |

클린/노이즈 검증 결과가 유사 → 노이즈 검증 세트의 신뢰성 입증.

---

### 2.5 한계점

1. **클래스 조건부 노이즈 가정**: 인스턴스 의존적(instance-dependent) 노이즈에는 이론이 직접 적용되지 않음
2. **대각 우세 조건 필요**: 노이즈율이 50% 이상인 극단적 경우 Assumption 2가 위반될 수 있음
3. **VC 차원 기반 bound의 느슨함**: 실제 딥러닝 모델에 적용 시 tight하지 않을 수 있음
4. **샘플 수 의존성**: Theorem 2의 효과는 충분히 많은 샘플을 전제로 하며, 소규모 데이터셋에서는 효과가 제한적
5. **전이 행렬 $T$ 미지 가정**: 실제로 $T$를 모르는 상황에서 이론의 실용적 적용에 한계

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 세 가지 훈련 단계 특성화

논문은 훈련 샘플 수 $m$에 따른 모델 특성을 세 단계로 체계적으로 설명:

**Case 1) $m$이 매우 작을 때:**
$$A_{\tilde{S}}(h) = 1, \quad C(h) \approx \text{random}, \quad A_D(h) \approx \text{random}$$
→ 일반화 불가, 고분산 상태

**Case 2) $m$이 중간 수준일 때:**
$$A_{\tilde{S}}(h) \approx 1, \quad C(h) \approx T, \quad A_D(h) \approx 1-\varepsilon$$
→ 노이즈 전이 프로세스를 예측하는 단계

**Case 3) $m$이 충분히 클 때:**
$$A_{\tilde{S}}(h) \approx 1-\varepsilon, \quad C(h) \approx I, \quad A_D(h) \approx 1$$
→ 전역 최적 분류기 근접 달성

### 3.2 데이터 증강의 이론적 설명

기존에는 직관적으로만 알려진 "노이즈 샘플에 데이터 증강을 적용하면 일반화가 향상된다"는 현상을:

> *"증강된 샘플은 동일한 노이즈를 가지지만, 훈련 샘플과 노이즈 분포 사이의 gap을 좁혀준다"*

는 이론적 근거로 설명. Theorem 2의 bound에서 유효 샘플 수 $m$을 증가시키는 효과:

$$-\sqrt{\frac{8\left(d_{VC} \cdot (\ln(2m/d_{VC}) + 1) + \ln(4/\delta)\right)}{m}} \xrightarrow{m \to \infty} 0$$

### 3.3 Early Stopping의 이론적 정당화

노이즈 검증 세트를 통해 최적의 early-stopped 모델을 선택함으로써:
- DNNs의 memorization effect (단순하고 정확한 패턴을 먼저 학습 후 노이즈를 기억하는 현상)를 활용
- 클린 샘플 없이도 최적 checkpoint 선택 가능

### 3.4 Teacher-Student 구조를 통한 일반화

NTS에서 Student가 Teacher보다 더 나은 성능을 보이는 이유:
- NT(최적 Teacher)는 early stopping된 모델로 노이즈를 덜 기억한 상태
- NS는 NT의 soft prediction을 학습하여 knowledge distillation 효과
- Soft label이 일종의 regularization 역할 → 일반화 성능 향상

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

**이론적 영향:**
- 정확도 지표가 robust loss function과 동등한 견고성을 가질 수 있다는 새로운 관점 제시
- 노이즈 레이블 학습의 이론적 기반 강화: 훈련-검증-모델선택의 전체 파이프라인에 대한 이론 완성
- 노이즈 검증 세트의 신뢰성에 대한 최초의 이론적 정당화 → 향후 모든 노이즈 레이블 연구에서 표준 검증 방식으로 활용 가능

**실용적 영향:**
- 복잡한 robust loss function 없이도 단순 cross-entropy로 경쟁력 있는 성능 달성 가능
- 하이퍼파라미터 튜닝, early stopping의 표준 절차 확립
- 클린 검증 세트가 필요 없어 실제 noisy label 환경(crowdsourcing, web scraping)에 직접 적용 가능

### 4.2 2020년 이후 관련 최신 연구 비교 분석

본 논문의 내용을 바탕으로, 논문 내에서 직접 언급된 2020년 이후 연구 및 관련 연구 방향을 비교 분석합니다.

**⚠️ 주의:** 아래의 2021년 이후 논문들은 본 논문(arXiv 2020.12)에서 직접 인용되지 않은 것들이며, 제가 일반적 지식으로 언급하는 것으로 **100% 정확성을 보장하기 어려우므로** 논문 내에서 직접 언급된 연구들을 중심으로 비교합니다.

#### 논문 내 언급된 2020년 주요 연구와의 비교

| 연구 | 접근 방식 | 본 논문과의 관계 |
|------|-----------|-----------------|
| **DivideMix** (Li et al., ICLR 2020) | Semi-supervised learning 방식, 두 네트워크로 clean/noisy 분리 | NTS가 DivideMix를 teacher로 사용 시 추가 성능 향상 달성 |
| **DMI** (Xu et al., NeurIPS 2019) | 정보이론 기반 robust loss | NTS 적용 시 DMI의 불안정성(asymmetric noise에서 급격한 성능 저하) 완화 |
| **GCE** (Zhang & Sabuncu, NeurIPS 2018) | 일반화 cross-entropy, robust loss | 이미 robust하지만 NTS 적용 시 추가 향상 |
| **Co-teaching** (Han et al., NeurIPS 2018b) | 두 네트워크 상호 학습 | NTS 적용으로 일관된 성능 향상 |

#### 인스턴스 의존적 노이즈 연구 방향

논문에서 언급된 **Xia et al. (2020)** - "Part-dependent label noise: Towards instance-dependent label noise"는 본 논문의 한계인 클래스 조건부 노이즈 가정을 넘어서는 방향을 제시합니다. 향후 연구에서 인스턴스 의존적 노이즈에도 적용 가능한 이론적 확장이 필요합니다.

### 4.3 향후 연구 시 고려해야 할 점

**1. 이론적 확장 방향:**

$$\Pr[\tilde{Y} = j | X = x, Y = i] = T_{i,j}(x)$$

인스턴스 의존적 노이즈 전이 행렬 $T(x)$로의 확장 시, Assumption 2(대각 우세)가 pointwise하게 만족되어야 하는 조건 분석 필요.

**2. 대규모 딥러닝 환경 고려:**
- Theorem 2의 VC 차원 기반 bound는 딥러닝의 과도한 파라미터화(overparameterization)를 제대로 설명하지 못함
- PAC-Bayes bound나 uniform stability 기반의 더 tight한 bound 개발 필요

**3. 노이즈 전이 행렬 추정:**
- 본 논문은 $T$의 존재를 가정하지만 추정하지는 않음
- $T$를 모르는 상태에서 Theorem 1의 Assumption 2 위반 여부를 실용적으로 확인하는 방법 개발 필요
- **Dual T** (Yao et al., NeurIPS 2020)처럼 $T$ 추정 기법과의 결합 연구 고려

**4. 노이즈 검증 세트의 구성:**
- 노이즈 검증 세트를 어떻게 효과적으로 구성할지 (sampling strategy) 추가 연구 필요
- 클래스 불균형이 있을 때의 영향 분석

**5. Semi-supervised 및 Self-supervised 학습과의 결합:**
- DivideMix처럼 label이 없는 샘플을 semi-supervised로 활용하는 방식과 NTS를 결합하면 추가 성능 향상 가능
- Self-supervised pretraining을 통한 초기화가 Case 2→3 전환을 가속화할 수 있는지 탐구

**6. 다중 주석자(Multi-annotator) 환경:**
- crowdsourcing에서 각 주석자마다 다른 $T$를 가질 때 본 이론의 확장 필요
- 개인별 노이즈 모델링과 Theorem 1의 연계

**7. 연속/회귀 레이블 노이즈:**
- 현재 이론은 분류 문제에 한정되어 있으며, 회귀 문제에서의 analogous theorem 개발 필요

---

## 참고자료

**주요 참고 논문 (본 논문 직접 인용):**

1. **Chen, P., Ye, J., Chen, G., Zhao, J., & Heng, P.-A. (2020).** "Robustness of Accuracy Metric and its Inspirations in Learning with Noisy Labels." arXiv:2012.04193v1. *(본 분석의 주 논문)*

2. **Zhang, Z., & Sabuncu, M. (2018).** "Generalized cross entropy loss for training deep neural networks with noisy labels." *NeurIPS 2018.*

3. **Li, J., Socher, R., & Hoi, S. C. (2020).** "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." *ICLR 2020.*

4. **Han, B., et al. (2018b).** "Co-teaching: Robust training of deep neural networks with extremely noisy labels." *NeurIPS 2018.*

5. **Xu, Y., et al. (2019).** "L_DMI: A Novel Information-theoretic Loss Function for Training Deep Nets Robust to Label Noise." *NeurIPS 2019.*

6. **Natarajan, N., et al. (2013).** "Learning with noisy labels." *NeurIPS 2013.*

7. **Vapnik, V. N. (1999).** "An overview of statistical learning theory." *IEEE Transactions on neural networks.*

8. **Xia, X., et al. (2020).** "Part-dependent label noise: Towards instance-dependent label noise." *NeurIPS 2020.*

9. **Yao, Y., et al. (2020).** "Dual T: Reducing estimation error for transition matrix in label-noise learning." *NeurIPS 2020.*

10. **Ghosh, A., Kumar, H., & Sastry, P. (2017).** "Robust loss functions under label noise for deep neural networks." *AAAI 2017.*

11. **Ma, X., et al. (2020).** "Normalized Loss Functions for Deep Learning with Noisy Labels." *ICML 2020.*

12. **Bentkus, V., et al. (2004).** "On Hoeffding's inequalities." *The Annals of Probability.*

13. **Arpit, D., et al. (2017).** "A closer look at memorization in deep networks." *ICML 2017.*

14. **Chen, P., et al. (2019b).** "Understanding and Utilizing Deep Neural Networks Trained with Noisy Labels." *ICML 2019.*
