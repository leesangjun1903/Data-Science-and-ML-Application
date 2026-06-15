# Sharpness-Aware Minimization for Efficiently Improving Generalization

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

SAM(Sharpness-Aware Minimization)의 핵심 주장은 다음과 같습니다:

> **"훈련 손실값만 최소화하는 것은 일반화에 충분하지 않으며, 손실 경관(loss landscape)의 기하학적 구조—특히 평탄성(flatness)—을 동시에 최적화해야 한다."**

과도하게 파라미터화된 현대 신경망에서는 동일한 낮은 훈련 손실값을 가진 여러 극소점(minima)이 존재하지만, 이들은 일반화 성능에서 현저한 차이를 보입니다. SAM은 단순히 낮은 손실값을 갖는 파라미터가 아니라, **파라미터 주변 이웃 전체**에서 균일하게 낮은 손실값을 갖는 파라미터를 탐색합니다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **SAM 알고리즘 제안** | 손실값과 손실 곡률을 동시에 최소화하는 min-max 최적화 절차 |
| **광범위한 실증 검증** | CIFAR-{10,100}, ImageNet, 파인튜닝 태스크 등에서 SOTA 성능 달성 |
| **레이블 노이즈 강건성** | 노이즈 레이블 전용 기법들과 동등한 수준의 강건성 확인 |
| **m-sharpness 개념 제안** | 일반화 격차와의 상관관계가 높은 새로운 날카로움 측도 제시 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

현대의 과도하게 파라미터화된 딥러닝 모델에서 발생하는 다음 문제를 해결합니다:

- **훈련 손실과 일반화 격차의 불일치**: 유사한 훈련 손실값을 가진 서로 다른 극소점들이 매우 다른 일반화 성능을 보임
- **날카로운 극소점(Sharp Minima) 수렴 문제**: SGD 등의 표준 최적화기는 훈련 데이터에는 적합하지만 날카로운 극소점에 수렴하는 경향이 있음
- **손실 경관 기하학의 미활용**: 기존 연구들이 평탄한 극소점과 일반화 간의 연관성을 이론적으로 제시했음에도, 실용적이고 확장 가능한 알고리즘이 부재했음

### 2.2 제안하는 방법 (수식 포함)

#### 이론적 동기: 일반화 경계 정리

임의의 $\rho > 0$에 대해 분포 $\mathcal{D}$로부터 생성된 훈련 세트 $S$에 대해 높은 확률로 다음이 성립합니다:

$$L_{\mathcal{D}}(\boldsymbol{w}) \leq \max_{\|\boldsymbol{\epsilon}\|_2 \leq \rho} L_S(\boldsymbol{w} + \boldsymbol{\epsilon}) + h\left(\|\boldsymbol{w}\|_2^2 / \rho^2\right)$$

여기서 $h: \mathbb{R}\_+ \rightarrow \mathbb{R}_+$는 순증가 함수입니다. 이를 명시적으로 분해하면:

$$\underbrace{\left[\max_{\|\boldsymbol{\epsilon}\|_2 \leq \rho} L_S(\boldsymbol{w} + \boldsymbol{\epsilon}) - L_S(\boldsymbol{w})\right]}_{\text{날카로움 항(Sharpness Term)}} + L_S(\boldsymbol{w}) + h\left(\|\boldsymbol{w}\|_2^2 / \rho^2\right)$$

#### SAM 최적화 목표

$$\min_{\boldsymbol{w}} L_S^{SAM}(\boldsymbol{w}) + \lambda\|\boldsymbol{w}\|_2^2 \quad \text{where} \quad L_S^{SAM}(\boldsymbol{w}) \triangleq \max_{\|\boldsymbol{\epsilon}\|_p \leq \rho} L_S(\boldsymbol{w} + \boldsymbol{\epsilon}) $$

여기서 $\rho \geq 0$은 이웃 크기 하이퍼파라미터, $p \in [1, \infty]$입니다 (실증적으로 $p=2$가 최적).

#### 내부 최대화 문제의 효율적 근사

내부 최대화 문제를 $\boldsymbol{\epsilon}$ 주변에서의 1차 테일러 전개로 근사합니다:

$$\hat{\boldsymbol{\epsilon}}(\boldsymbol{w}) \triangleq \underset{\|\boldsymbol{\epsilon}\|_p \leq \rho}{\arg\max} \ L_S(\boldsymbol{w} + \boldsymbol{\epsilon}) \approx \underset{\|\boldsymbol{\epsilon}\|_p \leq \rho}{\arg\max} \ \boldsymbol{\epsilon}^T \nabla_{\boldsymbol{w}} L_S(\boldsymbol{w})$$

이 근사 문제의 해는 쌍대 노름(dual norm) 문제로부터 도출됩니다:

$$\hat{\boldsymbol{\epsilon}}(\boldsymbol{w}) = \rho \cdot \text{sign}\left(\nabla_{\boldsymbol{w}} L_S(\boldsymbol{w})\right) |\nabla_{\boldsymbol{w}} L_S(\boldsymbol{w})|^{q-1} \bigg/ \left(\|\nabla_{\boldsymbol{w}} L_S(\boldsymbol{w})\|_q^q\right)^{1/p} $$

여기서 $1/p + 1/q = 1$. $p=2$인 경우 이는 단순히 그래디언트를 $\rho$ 크기로 정규화하는 것과 동일합니다:

$$\hat{\boldsymbol{\epsilon}}(\boldsymbol{w}) = \rho \cdot \frac{\nabla_{\boldsymbol{w}} L_S(\boldsymbol{w})}{\|\nabla_{\boldsymbol{w}} L_S(\boldsymbol{w})\|_2}$$

#### 최종 그래디언트 근사

2차 항을 제거하여 계산 효율성을 높인 최종 그래디언트 근사:

$$\nabla_{\boldsymbol{w}} L_S^{SAM}(\boldsymbol{w}) \approx \nabla_{\boldsymbol{w}} L_S(\boldsymbol{w})\big|_{\boldsymbol{w} + \hat{\boldsymbol{\epsilon}}(\boldsymbol{w})} $$

#### SAM 알고리즘 (의사코드)

```
입력: 훈련 세트 S, 손실 함수 l, 배치 크기 b, 학습률 η, 이웃 크기 ρ
출력: SAM으로 훈련된 모델

가중치 w₀ 초기화, t = 0
while not converged do:
    1. 배치 B = {(x₁,y₁),...,(xb,yb)} 샘플링
    2. 배치 훈련 손실의 그래디언트 ∇w L_B(w) 계산
    3. 수식 (2)에 따라 ε̂(w) 계산  ← 첫 번째 역전파
    4. SAM 목적함수의 그래디언트 근사 계산:
       g = ∇w L_B(w)|_{w+ε̂(w)}  ← 두 번째 역전파
    5. 가중치 업데이트: w_{t+1} = wt - η·g
    t = t + 1
return w_t
```

**계산 비용**: SAM은 매 반복마다 **2번의 역전파**가 필요하므로, 표준 SGD 대비 약 2배의 계산 비용이 발생합니다.

### 2.3 모델 구조

SAM은 특정 모델 구조에 종속되지 않는 **옵티마이저 수준의 방법론**입니다. 논문에서 검증된 모델들은 다음과 같습니다:

- **이미지 분류 (From Scratch)**: WideResNet-28-10, Shake-Shake (26 2x96d), PyramidNet, PyramidNet+ShakeDrop
- **ImageNet 대규모 실험**: ResNet-50, ResNet-101, ResNet-152
- **파인튜닝**: EfficientNet-b7 (ImageNet 사전학습), EfficientNet-L2 (ImageNet+JFT 사전학습)
- **노이즈 레이블**: ResNet-32

### 2.4 성능 향상

#### CIFAR-{10, 100} 결과

| 모델 | 데이터증강 | SAM (CIFAR-10) | SGD (CIFAR-10) | SAM (CIFAR-100) | SGD (CIFAR-100) |
|------|----------|----------------|----------------|-----------------|-----------------|
| WRN-28-10 (1800 epoch) | AA | **1.6±0.1** | 2.2±<0.1 | **12.8±0.2** | 16.1±0.2 |
| PyramidNet+ShakeDrop | AA | **1.4±<0.1** | 1.6±<0.1 | **10.3±0.1** | 10.6±0.1 |

#### ImageNet 결과 (ResNet)

| 모델 | Epoch | SAM Top-1 | Standard Top-1 |
|------|-------|-----------|----------------|
| ResNet-50 | 400 | **20.9%** | 22.3% |
| ResNet-101 | 400 | **19.0%** | 22.3% |
| ResNet-152 | 400 | **18.4%** | 20.9% |

특히 주목할 점은 SAM은 에폭 수를 늘릴수록 **지속적으로 성능이 향상**되는 반면, 표준 훈련은 400 에폭에서 **심각한 과적합**이 발생한다는 것입니다.

#### 파인튜닝 결과

| 데이터셋 | EfficientNet-L2 + SAM | EfficientNet-L2 | 이전 SOTA |
|---------|----------------------|-----------------|----------|
| CIFAR-10 | **0.30%** | 0.34% | 0.63% (BiT-L) |
| CIFAR-100 | **3.92%** | 4.07% | 6.49% (BiT-L) |
| ImageNet | **11.39%** | 11.8% | 11.45% (ViT) |

#### Hessian 스펙트럼 분석

SAM으로 훈련된 모델은 수렴 시 Hessian의 최대 고유값 $\lambda_{max}$가 SAM 없이 훈련한 경우 대비 현저히 낮습니다:
- **SAM 미적용**: $\lambda_{max} \approx 24.2$, $\lambda_{max}/\lambda_5 \approx 11.4$
- **SAM 적용**: $\lambda_{max} \approx 1.0$, $\lambda_{max}/\lambda_5 \approx 2.6$

### 2.5 한계점

1. **계산 비용 증가**: 매 반복마다 2번의 역전파가 필요하여 훈련 시간이 약 2배 증가
2. **하이퍼파라미터 $\rho$ 튜닝 필요**: 데이터셋과 모델에 따라 최적 $\rho$ 값이 다름 (기본값 $\rho=0.05$는 상당히 범용적이나 최적은 아닐 수 있음)
3. **1차 근사의 한계**: 내부 최대화 문제의 1차 테일러 전개 근사는 수렴 후반부에서 정확도가 낮아짐
4. **2차 항 효과 미해명**: 2차 항 제거가 오히려 성능을 향상시키는 이유가 불명확함
5. **이론적 간극**: m-sharpness가 이론적 상한보다 실험적으로 더 나은 일반화 예측자임이 밝혀졌으나, 그 이론적 설명이 미비
6. **NLP/다른 도메인 검증 부족**: 주로 컴퓨터 비전 태스크에서 검증됨

---

## 3. 일반화 성능 향상 가능성 (중점 분석)

### 3.1 이론적 근거: PAC-Bayesian 일반화 경계

논문 부록에 제시된 엄밀한 PAC-Bayesian 경계는 다음과 같습니다:

$$L_{\mathcal{D}}(\boldsymbol{w}) \leq \max_{\|\boldsymbol{\epsilon}\|_2 \leq \rho} L_S(\boldsymbol{w} + \boldsymbol{\epsilon}) + \sqrt{\frac{k \log\left(1 + \frac{\|\boldsymbol{w}\|_2^2}{\rho^2}\left(1 + \sqrt{\frac{\log n}{k}}\right)^2\right) + 4\log\frac{n}{\delta} + \tilde{O}(1)}{n-1}} $$

여기서:
- $n = |S|$: 훈련 데이터 크기
- $k$: 파라미터 수
- $\rho$: 이웃 크기
- $\delta$: 실패 확률

이 경계는 **테스트 손실이 (i) 이웃 내 최대 훈련 손실과 (ii) 파라미터 노름에 의존하는 복잡도 항의 합으로 상한된다**는 것을 보여줍니다. SAM은 이 상한의 첫 번째 항인 $\max_{\|\boldsymbol{\epsilon}\|_2 \leq \rho} L_S(\boldsymbol{w} + \boldsymbol{\epsilon})$를 직접 최소화합니다.

### 3.2 m-sharpness: 새로운 일반화 측도

SAM의 실용적 구현에서는 전체 훈련 세트가 아닌 **미니배치 단위로 섭동 $\epsilon$의 최대화**를 수행합니다. 이를 m-sharpness라고 정의합니다:

$$\text{m-sharpness} = \max_{\|\boldsymbol{\epsilon}\|_p \leq \rho} \frac{1}{m} \sum_{i \in B_m} l(\boldsymbol{w} + \boldsymbol{\epsilon}, x_i, y_i) - L_S(\boldsymbol{w})$$

여기서 $B_m$은 크기 $m$의 미니배치. 실험 결과:
- **$m$이 작을수록** 더 나은 일반화 성능을 보임
- **$m$이 작을수록** 실제 일반화 격차와의 상관관계(상호정보량)가 높아짐
- 이는 데이터 병렬화(multiple accelerators)와의 시너지 효과를 자연스럽게 제공

이 발견은 기존 이론이 제시한 전체 훈련 세트 기반 날카로움보다, 데이터 포인트별 날카로움이 실제 일반화를 더 잘 예측한다는 새로운 통찰을 제공합니다.

### 3.3 레이블 노이즈 강건성과 일반화의 관계

SAM이 파라미터 섭동에 강건한 파라미터를 탐색하는 특성은, 훈련 레이블 노이즈로 인한 손실 경관의 섭동에도 강건성을 부여합니다:

$$\text{CIFAR-10 노이즈 레이블 실험 (노이즈율 40\%):}$$
- SAM: **93.4%** 정확도
- SGD: 68.8% 정확도
- MentorMix (노이즈 전용 방법): **94.2%** 정확도

이는 SAM의 일반화 향상 메커니즘이 단순한 정규화를 넘어, **훈련 분포의 노이즈에 대한 근본적 강건성**을 제공함을 시사합니다.

### 3.4 평탄한 극소점과 일반화의 관계

SAM으로 훈련된 모델은 더 평탄한 손실 경관에 수렴하며, 이는 다음을 의미합니다:

1. **더 나은 OOD(Out-of-Distribution) 일반화**: 입력 분포의 미세한 변화에 덜 민감
2. **과적합 저항**: 더 많은 에폭을 훈련해도 과적합이 덜 발생 (ResNet-152의 경우, 200→400 에폭에서 표준 훈련은 과적합, SAM은 성능 향상 지속)
3. **데이터 증강과의 상호보완**: AutoAugment, Cutout 등과 결합 시 추가적인 성능 향상

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (1) 최적화 이론 관점
SAM은 **목적함수 자체를 변경**하여 일반화를 추구하는 새로운 패러다임을 제시합니다. 기존 연구가 학습률, 배치 크기, 모멘텀 등의 최적화기 하이퍼파라미터 튜닝에 집중했던 것과 달리, SAM은 손실 경관의 기하학을 직접 조작합니다. 이는 최적화와 일반화 이론의 통합을 촉진시킵니다.

#### (2) 일반화 이론 관점
m-sharpness의 발견은 **데이터 포인트 수준의 날카로움**이 일반화의 더 나은 예측자임을 시사하며, 기존 PAC-Bayesian 이론의 확장 방향을 제시합니다. 특히:
- 배치 단위 날카로움과 일반화의 관계 연구
- 개별 데이터 포인트의 기여도(per-sample sharpness) 연구

#### (3) 실용적 영향
- **기존 파이프라인과의 호환성**: SAM은 기존 옵티마이저를 단순 교체하는 방식으로 적용 가능
- **다양한 도메인으로의 확장**: NLP, 강화학습, 생성 모델 등에서의 적용 가능성
- **모델 경량화와의 결합**: 지식 증류, 프루닝 등과 결합 시 일반화-효율성 트레이드오프 개선 가능성

### 4.2 2020년 이후 관련 최신 연구 비교 분석

SAM 이후 다양한 후속 연구들이 등장했습니다 (논문 내에 직접 언급되지 않은 연구들은 제가 확인할 수 있는 범위 내에서 기술하되, 불확실한 수치는 제시하지 않겠습니다):

#### ASAM (Adaptive SAM, 2021)
- **논문**: Kwon et al., "ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks" (ICML 2021)
- **핵심 아이디어**: SAM의 $\rho$-ball이 모든 파라미터에 동일하게 적용되는 문제를 해결하기 위해 **적응형(adaptive) 이웃 크기** 도입
- **개선점**: 파라미터별 스케일을 고려한 노름 정의:

$$\hat{\boldsymbol{\epsilon}} = \underset{\|\boldsymbol{T}_{\boldsymbol{w}}^{-1}\boldsymbol{\epsilon}\|_p \leq \rho}{\arg\max} L_S(\boldsymbol{w} + \boldsymbol{\epsilon})$$

여기서 $\boldsymbol{T}_{\boldsymbol{w}}$는 파라미터 스케일을 반영하는 행렬. 이는 배치 정규화 등으로 인한 스케일 불변성 문제를 해결합니다.

#### LookSAM (2022)
- **핵심 아이디어**: SAM의 2배 계산 비용을 줄이기 위해 그래디언트를 **주기적으로 재사용**하는 방법 제안
- **개선점**: 매 스텝마다 섭동을 계산하지 않고 k 스텝마다 한 번만 계산하여 계산 효율 향상

#### Fisher SAM / 다양한 효율적 SAM 변형
- 계산 비용 절감을 위한 다양한 근사 방법 연구들이 등장

#### SAM과 Transformer/LLM의 결합
- Vision Transformer(ViT) 및 대규모 언어 모델에서 SAM의 효과 연구
- 특히 파인튜닝 시 과적합 방지에 효과적임이 보고됨

#### 이론적 이해 심화
- SAM이 암묵적으로 Hessian의 trace를 최소화한다는 이론적 분석
- SAM과 암묵적 정규화(implicit regularization)의 관계 연구

### 4.3 앞으로 연구 시 고려할 점

#### 알고리즘 개선 측면
1. **계산 효율성**: 2배 계산 비용 문제는 실제 산업 적용에서 중요한 장벽. 1.x배 수준의 오버헤드로 유사한 효과를 내는 방법 연구 필요
2. **$\rho$ 적응적 조정**: 훈련 단계별, 레이어별로 최적 $\rho$가 다를 수 있으므로 적응형 $\rho$ 스케줄링 연구
3. **2차 항의 역할 규명**: 2차 항 제거가 왜 성능을 향상시키는지에 대한 이론적 설명 필요

#### 이론적 측면
1. **m-sharpness의 이론적 정당화**: 작은 $m$이 더 좋은 일반화 예측자인 이유에 대한 엄밀한 이론 필요
2. **비볼록 최적화 수렴 보장**: 비볼록 손실 경관에서 SAM의 수렴 성질에 대한 이론적 분석 부족
3. **날카로운 극소점도 일반화 가능**: Dinh et al. (2017)이 지적한 바와 같이, 리파라미터화를 통해 날카로운 극소점도 일반화될 수 있다는 반론에 대한 대응 이론 필요

#### 실용적 측면
1. **NLP/LLM 도메인 검증**: 본 논문은 주로 컴퓨터 비전에서 검증되었으므로, 자연어처리에서의 체계적 검증 필요
2. **연합 학습(Federated Learning)과의 결합**: 분산 환경에서 SAM의 m-sharpness가 특히 유리할 수 있음
3. **데이터 효율성**: 소수 샷(few-shot) 학습 등 데이터가 제한된 환경에서의 효과 연구
4. **다른 정규화 기법과의 이론적 연결**: Dropout, BatchNorm, Mixup과 SAM의 관계를 통합적으로 이해하는 프레임워크 필요
5. **강건성(Robustness) 연구**: 적대적 공격(adversarial attack)에 대한 SAM의 강건성과 일반화 강건성의 관계 규명

---

## 참고 자료

**주요 참고 논문 (제공된 PDF 원문)**:
- Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). **Sharpness-Aware Minimization for Efficiently Improving Generalization**. *ICLR 2021*. arXiv:2010.01412v3

**논문 내 인용된 핵심 관련 연구**:
- Shirish Keskar, N. et al. (2016). On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima. arXiv:1609.04836
- Dziugaite, G. K., & Roy, D. M. (2017). Computing nonvacuous generalization bounds for deep neural networks. arXiv:1703.11008
- Jiang, Y. et al. (2019). Fantastic Generalization Measures and Where to Find Them. arXiv:1912.02178
- Izmailov, P. et al. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. arXiv:1803.05407
- Chaudhari, P. et al. (2016). Entropy-SGD: Biasing Gradient Descent Into Wide Valleys. arXiv:1611.01838
- McAllester, D. A. (1999). PAC-Bayesian model averaging. *COLT 1999*
- Hochreiter, S., & Schmidhuber, J. (1997). Flat minima. *Neural Computation*, 9(1):1–42

**후속 연구 (SAM 이후)**:
- Kwon, J. et al. (2021). ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks. *ICML 2021*. arXiv:2102.11600

> **주의**: 본 답변의 SAM 원문 관련 내용은 제공된 PDF(arXiv:2010.01412v3)를 직접 참조하여 작성하였습니다. 후속 연구(ASAM, LookSAM 등)에 대한 세부 수치 및 성능 비교는 원문 논문을 직접 확인하시기를 권장합니다.

# Sharpness-Aware Minimization for Efficiently Improving Generalization

### 1. 핵심 주장 및 주요 기여
**"Sharpness-Aware Minimization for Efficiently Improving Generalization"** 논문(Foret et al., ICLR 2021)은 과매개변수화(overparameterized) 신경망의 일반화 성능 향상을 위한 혁신적인 최적화 기법을 제시합니다. 이 연구의 핵심 통찰은 단순히 훈련 손실(training loss)을 최소화하는 것으로는 좋은 일반화를 보장할 수 없다는 점입니다. 신경망의 손실 지형(loss landscape)은 다양한 지역 최솟값(local minima)을 가지며, 이들이 동일한 훈련 손실을 가질지라도 테스트 성능은 크게 다를 수 있습니다.[1]

논문의 주요 기여는 다음과 같습니다:
- **SAM 알고리즘**: 손실 값(loss value)과 손실의 예리함(loss sharpness)을 동시에 최소화하는 최적화 기법
- **PAC-Bayesian 일반화 경계**: 손실 지형의 국소적 특성과 일반화 능력의 관계를 수학적으로 증명
- **m-sharpness 개념**: 배치 크기에 따라 변하는 새로운 예리함 측정 지표
- **광범위한 실증 검증**: CIFAR-10/100, ImageNet, 전이 학습, 노이즈 레이블 학습 등 다양한 설정에서의 성능 향상 입증

***

### 2. 문제 정의 및 제안 방법
#### 2.1 핵심 문제

현대 딥러닝에서 가장 큰 도전은 과매개변수화 모델이 훈련 데이터에 과적합되기 쉽다는 점입니다. 특히, 훈련 손실 $L_S(w)$는 낮지만 테스트 손실 $L_D(w)$는 높은 경우가 빈번합니다. 이는 최적화 알고리즘이 수렴하는 최솟값의 특성이 중요함을 의미합니다.

#### 2.2 이론적 근거

논문은 다음과 같은 PAC-Bayesian 일반화 경계를 증명합니다:

$$L_D(w) \leq \max_{\|\epsilon\|_2 \leq \rho} L_S(w + \epsilon) + h\left(\frac{\|w\|_2^2}{\rho^2}\right)$$

이를 다시 쓰면:

$$L_D(w) \leq \underbrace{\left[\max_{\|\epsilon\|_2 \leq \rho} L_S(w + \epsilon) - L_S(w)\right]}_{\text{Sharpness}} + L_S(w) + \lambda\|w\|_2^2$$

여기서 제곱괄호 안의 항은 **손실 지형의 예리함(sharpness)**을 나타냅니다. 이 경계는 더 평탄한 최솟값을 찾을수록 더 나은 일반화를 기대할 수 있음을 시사합니다.[1]

#### 2.3 SAM의 최적화 목표

$$\min_w L_{S}^{\text{SAM}}(w) + \lambda\|w\|_2^2$$

여기서:

$$L_S^{\text{SAM}}(w) \triangleq \max_{\|\epsilon\|_p \leq \rho} L_S(w + \epsilon)$$

이는 **min-max 최적화 문제**로, 내부 최대화 문제를 해결한 후 외부 최소화를 수행합니다.[1]

#### 2.4 효율적인 알고리즘 구현

내부 최대화 문제를 1차 Taylor 전개로 근사하면:

$$\hat{\epsilon}(w) = \rho \frac{\text{sign}(\nabla_w L_S(w)) |\nabla_w L_S(w)|^{q-1}}{\|\nabla_w L_S(w)\|_q^{1/p}}$$

여기서 $1/p + 1/q = 1$입니다.[1]

최종적으로 다음의 그래디언트 근사를 얻습니다:

$$\nabla_w L_S^{\text{SAM}}(w) \approx \nabla_w L_S(w + \hat{\epsilon}(w))$$

**Algorithm 1: SAM 의사 코드**

```
입력: 훈련 집합 S, 손실 함수 l, 배치 크기 b, 
       학습률 η, 근처 크기 ρ
초기화: 가중치 w₀, t = 0
반복:
  배치 B 샘플링
  그래디언트 계산: ∇L_B(w)
  섭동 계산: ε̂(w) (식 2)
  SAM 목적 그래디언트: g = ∇L_B(w + ε̂(w))
  가중치 업데이트: w_{t+1} = w_t - ηg
  t = t + 1
반복 종료 시 w_t 반환
```

***

### 3. 성능 향상 및 광범위한 실증 평가
SAM은 다양한 벤치마크에서 일관된 성능 개선을 달성했습니다:

**CIFAR-10/100에서의 성능:**
- CIFAR-10: 2.2% → 1.6% (0.6%p 개선, WideResNet 기준)
- CIFAR-100: 10.6% → 10.3% (0.3%p 개선, PyramidNet+ShakeDrop 기준)[1]
- 이는 이미 정교한 정규화(Shake-Shake, ShakeDrop)가 적용된 모델에서도 추가 개선을 달성

**ImageNet 대규모 실험:**
- ResNet-152 (400 에포크): 20.9% → 18.4% (2.5%p 개선)
- 특히 SAM은 에포크 증가에도 과적합되지 않는 특성 보유[1]

**전이 학습(Finetuning) 설정:**
- EfficientNet-b7 평균: 7.68% → 7.44% 오류율
- EfficientNet-L2는 이전 SOTA를 상당히 초과[1]

| 데이터셋 | SAM | SGD | 개선율 |
|---------|-----|-----|--------|
| CIFAR-10 | 1.6% | 2.2% | 27.3% |
| CIFAR-100 | 10.3% | 10.6% | 2.8% |
| ImageNet | 18.4% | 20.3% | 9.4% |
| SVHN | 0.99% | 1.14% | 13.2% |
| Fashion-MNIST | 3.59% | 3.86% | 7.0% |

***

### 4. 일반화 성능 향상 메커니즘
#### 4.1 손실 지형의 기하학적 특성

논문은 Hessian 행렬의 스펙트럼을 통해 SAM의 효과를 분석했습니다. WideResNet-40-10을 CIFAR-10에서 300 스텝 학습한 결과:[1]

- **최대 고유값 (λ_max)**:
  - SGD: 약 24.2
  - SAM: 약 1.0
  
- **고유값 비율 (λ_max/λ_5)**:
  - SGD: 11.4 (예리한 최솟값)
  - SAM: 2.6 (평탄한 최솟값)

이는 SAM이 실제로 손실 곡면의 곡률을 감소시킨다는 것을 명확히 보여줍니다.

#### 4.2 m-Sharpness와 일반화의 관계

흥미로운 발견 중 하나는 **배치 크기 m에 따른 예리함 측정의 중요성**입니다. 병렬 학습에서 각 가속기가 크기 m인 데이터 부분집합에서 독립적으로 SAM 업데이트를 계산할 때, 이러한 구성적 m-sharpness는 전체 훈련 집합을 사용한 표준 sharpness보다 **일반화 격차와의 상관성이 더 높습니다**.[1]

이는 다음을 시사합니다:
- 더 작은 m 값이 더 나은 일반화 성능을 제공
- 이는 병렬화 필요성과 우연히 일치

#### 4.3 노이즈 레이블에 대한 견고성

SAM은 라벨 노이즈에 대한 자연스러운 견고성을 제공합니다:

| 노이즈율 | SAM | MentorMix | Bootstrap+SAM |
|---------|-----|----------|----------------|
| 20% | 95.1% | 95.6% | 95.4% |
| 40% | 93.4% | 94.2% | 94.2% |
| 60% | 90.5% | 91.3% | 91.8% |
| 80% | 77.9% | 81.0% | 79.9% |

SAM의 섭동 기반 접근이 노이즈 방해에 대한 견고성을 자연스럽게 제공하는 것으로 보입니다.[1]

***

### 5. 방법의 한계
#### 5.1 이론적 한계
1. **선형화 근사의 정당성**: 1차 Taylor 전개가 모든 상황에서 충분한 근사인지에 대한 의문 존재
2. **2차 항의 역할**: 실험적으로 2차 항 제외 시 성능이 더 좋음이 관찰되었으나, 그 이유가 명확하지 않음[1]
3. **Theorem 1의 느슨함**: PAC-Bayesian 경계는 실제 성능보다 상당히 느슨할 수 있음

#### 5.2 계산 효율성
1. **연산 비용**: 각 업데이트마다 두 번의 역전파가 필요하므로 표준 SGD 대비 약 2배의 계산 비용
2. **메모리 사용**: 섭동된 모델의 그래디언트 계산으로 인한 메모리 오버헤드
3. **하이퍼파라미터 튜닝**: ρ 값의 선택이 중요하며, 데이터셋마다 최적값이 상이

#### 5.3 실무적 한계
1. **노이즈가 많은 환경**: 배치 정규화 등 확률적 요소가 많은 경우 섭동 추정의 불안정성
2. **매우 큰 모델**: 메모리 제약으로 인한 적용 어려움
3. **온라인 학습**: 한 번의 샘플로 섭동을 계산하기 어려움

***

### 6. 2020년 이후 관련 최신 연구 비교 분석
#### 6.1 효율성 개선 연구

**ESAM (Efficient SAM, 2021)**[2]
- SAM의 계산 비용을 2배에서 40% 증가로 감소
- 확률적 가중치 섭동(Stochastic Weight Perturbation) 도입
- Sharpness-민감 데이터 선택 기법 제안
- CIFAR-100에서 SAM과 유사한 성능 달성

**K-SAM (2022)**[3]
- 상위 k개의 손실이 큰 샘플만 사용하여 효율성 향상
- SGD 수준의 계산 비용으로 일반화 개선

#### 6.2 이론적 진전

**Unified SAM (2025)**[4]
- SAM과 비정규화 버전(USAM)의 통합 분석
- Polyak-Łojasiewicz 조건 하에서 수렴 보장
- 임의의 샘플링 패러다임 지원

**Friendly-SAM (2024, CVPR)**[5]
- 배치 특정 확률적 그래디언트 노이즈의 중요성 규명
- 지수이동평균(EMA)으로 전체 그래디언트 추정 및 제거
- F-SAM이 SAM보다 더 우수한 견고성 제공
- 이론적 수렴 증명 포함

**Eigen-SAM (2025)**[6]
- Hessian 최대 고유값의 명시적 정규화
- 섭동 벡터와 상위 고유벡터의 정렬 강조
- 3차 확률미분방정식(SDE) 분석으로 동역학 규명

#### 6.3 기하학적 확장

**HyperbolicSAM (2025)**[7]
- Poincaré ball manifold에서 SAM 일반화
- 계층적 구조를 가진 데이터(지식 그래프, 분류 체계)에 최적화
- CIFAR-10에서 2.34% 오류율 달성 (Euclidean SAM의 2.86% 대비)

**Monge SAM (2025)**[8]
- 손실 곡면에서 유도된 리만 메트릭 사용
- 재매개변수화 불변성(reparameterization invariance) 달성
- 안장점에 덜 끌려가는 성질

#### 6.4 도메인 특화 적용

**DGSAM (2025) - Domain Generalization**[9]
- 영역 일반화에서 "가짜 평탄 최솟값" 문제 해결
- 개별 영역 내에서의 예리함 최소화 강조
- 전역 sharpness 보다 개별 영역별 sharpness가 일반화에 더 유용함을 증명

**Modality-Aware SAM (2025)**[10]
- 다중모달 학습(멀티미디어, 음성+비전)에 최적화
- Shapley 값으로 지배적 모달리티 식별
- 불균형 모달 기여 문제 해결

**Focal-SAM (2025) - Long-tail Classification**[11]
- 장꼬리 분포 분류에 클래스별 가중치 적용
- 헤드 클래스와 테일 클래스 간 손실 곡면의 기하학적 차이 반영
- ImbSAM, CC-SAM 대비 효율성과 성능 개선

**CA-SAM (2025) - Noisy Labels**[12]
- Clean-aware sharpness minimization
- 깨끗한 샘플과 노이즈 샘플의 섭동 방향 불일치 분석
- 과거 모델 예측으로 데이터 분할

#### 6.5 강건성 및 안정성 개선

**GCSAM (2025) - Gradient Centralized SAM**[13]
- 그래디언트 중심화 기법 도입
- 계산 효율성과 노이즈에 대한 견고성 개선
- 배치 정규화 변동에 덜 민감

**CR-SAM (2023) - Curvature Regularized**[14]
- 곡률 정규화 항 추가
- PAC-Bayes 경계를 통한 이론적 정당화

#### 6.6 응용 분야 확대

**의료 및 생명과학 응용:**
- Raman 분광기를 통한 세균 저항성 진단에서 2.7% 평균 정확도 향상[15]
- 생물의학 이미지 분석에서 일반화 성능 개선[16]

**컴퓨터 비전 확장:**
- Vision Transformer(ViT)를 포함한 최신 아키텍처 지원
- 반지도 학습, 자기지도 학습에 통합

***

### 7. 향후 연구의 방향성 및 고려사항
#### 7.1 이론적 미해결 문제

1. **m-Sharpness의 완전한 이해**
   - m-sharpness가 왜 전체 데이터셋 기반 sharpness보다 일반화를 더 잘 예측하는가?
   - 이것이 단순 경험적 현상인지, 근본적인 이론적 이유가 있는지 규명 필요

2. **2차 항의 역할**
   - SAM 유도 과정에서 2차 항을 제외할 때 오히려 성능이 향상되는 현상의 이해
   - 이는 현재의 이론적 프레임워크의 불완전성을 시사

3. **재매개변수화 불변성**
   - 손실 기하학이 모델 매개변수화에 어떻게 의존하는가?
   - Monge SAM의 리만 기하학적 접근의 일반화 가능성

#### 7.2 알고리즘적 개선 방향

1. **계산 효율성의 한계 돌파**
   - 현재 ESAM도 40% 추가 비용 필요
   - Hessian-무료(Hessian-free) 방식의 더 효율적인 구현
   - GPU/TPU 메모리 효율성 개선

2. **적응형 ρ 자동 선택**
   - 데이터셋별 ρ 최적값 자동 결정 메커니즘
   - 학습 과정 중 동적 ρ 조정

3. **하이브리드 방식**
   - SAM과 다른 정규화 기법(mixup, cutmix 등)의 상호작용 분석
   - MentorSAM 등 결합 방식의 체계적 개발

#### 7.3 실무적 적용 시 주의사항

1. **배치 크기의 영향**
   - m-sharpness 개념에 따르면, 작은 배치에서의 학습이 더 좋은 일반화 제공
   - 계산 자원과 일반화 성능의 트레이드오프 고려 필요

2. **다양한 도메인에서의 검증 필요**
   - 현재 대부분의 연구는 이미지 분류에 집중
   - NLP, 강화학습, 추천 시스템 등에서의 효과성 검증 필요

3. **프로덕션 환경에서의 고려사항**
   - 실시간 추론 성능: 학습 시간 증가가 배포 후 추론에는 영향 없음
   - 모델 저장소: SAM으로 학습한 모델은 일반적인 가중치 형태로 저장 가능
   - 전이 학습: SAM으로 학습한 사전학습 모델의 품질 우수성 확인

#### 7.4 학제 간 융합 방향

1. **통계학과의 연계**
   - PAC-Bayesian 틀과 빈도주의적 일반화 경계의 통합
   - 최적 수렴률(optimal convergence rate) 달성 가능성

2. **수학적 최적화 관점**
   - min-max 문제의 더 정교한 해석
   - 이중 경시 강화(bilevel optimization) 이론의 적용

3. **신경과학과의 연관성**
   - 뇌의 학습 과정이 손실 곡면의 평탄한 영역을 선호하는 메커니즘
   - 진화적 최적화의 관점

***

### 8. 결론 및 임팩트 평가
Sharpness-Aware Minimization은 손실 지형의 기하학과 일반화의 연결을 실용적으로 구현한 획기적 연구입니다. 이 논문 이후 수십 개의 확장 연구가 생성되었으며, 각각이 특정 도메인 또는 문제 설정에서 추가적인 개선을 달성했습니다.

**핵심 성과:**
- 광범위한 벤치마크에서 일관된 0.3~2.5%의 오류율 감소
- 이미 정규화가 적용된 모델에서도 추가 개선 달성
- 이론적 일반화 경계의 제공
- 단순하고 구현 용이한 알고리즘

**남아있는 도전:**
- 계산 효율성 (2배의 비용)
- 하이퍼파라미터 민감성 (ρ 선택)
- 이론과 실제 성능 간의 격차

**향후 전망:**
SAM의 성공은 손실 지형 관점의 타당성을 입증했으며, 향후 연구는 (1) 더 효율적인 구현, (2) 더 정교한 이론적 이해, (3) 더 넓은 적용 도메인 확대로 진행될 것으로 예상됩니다. 특히 대규모 언어 모델과 멀티모달 모델에서의 SAM 적용이 미개척 영역입니다.

***

### 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b408df71-d606-4808-82d4-685a89f32833/2010.01412v3.pdf)
[2](https://arxiv.org/abs/2110.03141)
[3](https://arxiv.org/pdf/2210.12864.pdf)
[4](https://arxiv.org/abs/2503.02225)
[5](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Friendly_Sharpness-Aware_Minimization_CVPR_2024_paper.pdf)
[6](https://arxiv.org/abs/2501.12666)
[7](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13993/3092692/Hyperbolic-SAM--sharpness-aware-minimization-in-hyperbolic-space-for/10.1117/12.3092692.full)
[8](https://arxiv.org/abs/2502.08448)
[9](https://arxiv.org/abs/2503.23430)
[10](https://arxiv.org/abs/2510.24919)
[11](https://arxiv.org/abs/2505.01660)
[12](https://www.nature.com/articles/s41598-025-85679-8)
[13](https://pmc.ncbi.nlm.nih.gov/articles/PMC12599882/)
[14](http://arxiv.org/pdf/2312.13555.pdf)
[15](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13307/3049180/Sharpness-aware-minimization-SAM-improves-generalization-performance-of-bacterial-Raman/10.1117/12.3049180.full)
[16](https://pmc.ncbi.nlm.nih.gov/articles/PMC12121992/)
[17](https://www.semanticscholar.org/paper/a2cd073b57be744533152202989228cb4122270a)
[18](https://arxiv.org/pdf/2212.04343.pdf)
[19](https://arxiv.org/pdf/2501.11584.pdf)
[20](http://arxiv.org/pdf/2110.03141.pdf)
[21](https://arxiv.org/pdf/2305.15817.pdf)
[22](http://arxiv.org/pdf/2403.12350.pdf)
[23](https://arxiv.org/pdf/2303.00565.pdf)
[24](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Flatness-Aware_Minimization_for_Domain_Generalization_ICCV_2023_paper.pdf)
[25](https://liner.com/review/improving-generalization-universal-adversarial-perturbation-via-dynamic-maximin-optimization)
[26](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Seeking_Consistent_Flat_Minima_for_Better_Domain_Generalization_via_Refining_CVPR_2025_paper.pdf)
[27](https://arxiv.org/abs/2503.12793)
[28](https://arxiv.org/abs/2403.12350)
[29](https://arxiv.org/html/2511.04808v1)
[30](https://arxiv.org/html/2503.12793v3)
[31](https://arxiv.org/abs/2010.01412)
[32](https://pdfs.semanticscholar.org/ee54/53052a78ca394d0cfd40fc9f0ab7ee0a9b4b.pdf)
[33](https://arxiv.org/pdf/2501.13864.pdf)
[34](https://pdfs.semanticscholar.org/42d6/45ed82a3fa6206d1ae119acd09f9ef031834.pdf)
[35](https://arxiv.org/html/2501.13864v1)
[36](https://arxiv.org/pdf/2302.05185.pdf)
[37](https://www.arxiv.org/pdf/2511.10714.pdf)
