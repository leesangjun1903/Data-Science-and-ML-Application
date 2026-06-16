# Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Northcutt et al., NeurIPS 2021 Datasets & Benchmarks Track)은 다음의 핵심 주장을 제시합니다:

> **머신러닝의 주요 벤치마크 테스트 셋에는 라벨 오류가 광범위하게 존재하며, 이 오류들은 모델 선택 결정을 불안정하게 만들고, 특히 고용량(high-capacity) 모델이 저용량(low-capacity) 모델보다 실제로 열등할 수 있음을 테스트 라벨 수정 전에는 알 수 없다.**

### 주요 기여 (4가지)

| 기여 | 내용 |
|------|------|
| **①** | 10개 주요 ML 벤치마크 테스트 셋에서 광범위한 라벨 오류 발견 |
| **②** | 각 테스트 셋을 정제·수정하는 오픈소스 리소스 제공 (`labelerrors.com`, `github.com/cleanlab/label-errors`) |
| **③** | 고용량 모델이 잘못된 라벨에 더 잘 맞추지만(원본 정확도↑), 교정된 라벨에서는 저용량 모델에 역전됨을 입증 |
| **④** | 테스트 라벨 오류 비율의 소폭 증가만으로도 모델 선택이 뒤바뀜을 수치로 제시 |

---

## 2. 해결하고자 하는 문제, 제안 방법(수식), 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

기존 연구들은 **훈련 셋의 라벨 노이즈**를 주로 다뤄왔으나, **테스트 셋의 라벨 오류**는 상대적으로 간과되었습니다. 테스트 셋은 암묵적으로 "정답"으로 간주되어 왔기 때문입니다. 그러나 테스트 셋의 라벨 오류는:

- 모델 순위(벤치마크)를 불안정하게 만들고
- 실무에서 잘못된 모델 배포 결정을 유도하며
- 고용량 모델의 "진짜 일반화 성능"을 과대평가하게 만듭니다

### 2.2 제안 방법: Confident Learning (CL) 기반 라벨 오류 탐지

#### Step 1. Confident Joint 추정

노이즈가 있는 관측 라벨 $\tilde{y}$와 미지의 실제 라벨 $y^*$ 사이의 결합 분포를 추정합니다.

**Class-conditional 노이즈 가정:**

```math
p(\tilde{y} \mid y^*, \boldsymbol{x}) = p(\tilde{y} \mid y^*)
```

즉, 노이즈는 데이터 $\boldsymbol{x}$가 아닌 실제 클래스 $y^*$에만 의존한다고 가정합니다.

**Confident Joint** $\boldsymbol{C}_{\tilde{y}, y^*}$는 다음과 같이 추정됩니다:

```math
\boldsymbol{C}_{\tilde{y}, y^*} = \left|\left\{\boldsymbol{x} \in X_{\tilde{y}=i} : \hat{p}(\tilde{y}=j; \boldsymbol{x}, \boldsymbol{\theta}) \geq t_j \right\}\right|
```

여기서 $t_j$는 클래스 $j$에 대한 **per-class 임계값**으로:

$$t_j = \frac{1}{|X_{\tilde{y}=j}|} \sum_{\boldsymbol{x} \in X_{\tilde{y}=j}} \hat{p}(\tilde{y}=j; \boldsymbol{x}, \boldsymbol{\theta}) $$

이 임계값은 클래스 불균형에 대한 강건성을 제공합니다.

#### Step 2. 결합 분포 $Q_{\tilde{y}, y^*}$ 정규화

```math
\hat{Q}_{\tilde{y}=i, y^*=j} = \frac{\dfrac{C_{\tilde{y}=i, y^*=j}}{\sum_{j \in [m]} C_{\tilde{y}=i, y^*=j}} \cdot |X_{\tilde{y}=i}|}{\displaystyle\sum_{i \in [m], j \in [m]} \left(\dfrac{C_{\tilde{y}=i, y^*=j}}{\sum_{j \in [m]} C_{\tilde{y}=i, y^*=j}} \cdot |X_{\tilde{y}=i}|\right)}
```

#### Step 3. 라벨 오류 비율 및 개수 추정

$$\rho = 1 - \sum_{i \in [m]} \hat{p}(\tilde{y}=i, y^*=i)$$

라벨 오류 개수: $\rho \cdot n$

#### Step 4. 라벨 오류 후보 선별

정규화된 마진(normalized margin)으로 상위 $\rho \cdot n$개 예시를 선택:

$$\text{score}(\boldsymbol{x}) = \hat{p}(\tilde{y}=i; \boldsymbol{x}) - \max_{j \neq i} \hat{p}(\tilde{y}=j; \boldsymbol{x})$$

마진 점수가 낮을수록(음수에 가까울수록) 라벨 오류 가능성이 높습니다.

### 2.3 주요 정의 (벤치마크 안정성 분석용 용어체계)

| 집합 | 정의 |
|------|------|
| $\mathcal{B}$ (benign set) | CL이 오류로 플래그하지 않았거나, 인간 검토자가 원본 라벨 유지에 동의한 예시 집합 |
| $\mathcal{U}$ (unknown-label set) | 인간 검토자가 단일 정답에 합의하지 못한 예시 집합 |
| $\mathcal{P}$ (pruned set) | $\mathcal{D} \setminus \mathcal{U}$ |
| $\mathcal{C}$ (correctable set) | 인간 검토에서 다른 라벨로 합의된 예시 집합, $\mathcal{C} = \mathcal{P} \setminus \mathcal{B}$ |
| $N$ (noise prevalence) | $N = \dfrac{ \mathcal{C} }{ \mathcal{P} }$ |

**원본 정확도** $\tilde{A}$: 기존 라벨 기준 정확도

**교정 정확도** $A^*$: 수정된 라벨 기준 정확도 (실무에서 실제로 중요한 지표)

**모델 $\mathcal{M}$의 노이즈 비율별 기대 정확도:**

$$A(x; \mathcal{M}) = \frac{A_\mathcal{C}(\mathcal{M}) \cdot |\mathcal{C}| + (1-x) \cdot A_\mathcal{B}(\mathcal{M}) \cdot |\mathcal{B}|}{|\mathcal{C}| + (1-x) \cdot |\mathcal{B}|}$$

여기서 $x$는 benign set $\mathcal{B}$에서 제거되는 비율이며, $x: 0 \to 1$로 변화시켜 다양한 노이즈 수준을 시뮬레이션합니다.

### 2.4 실험 설계: 모델 구조 및 데이터셋

**사용 데이터셋 (10개):**

| 유형 | 데이터셋 |
|------|----------|
| 이미지 | MNIST, CIFAR-10, CIFAR-100, Caltech-256, ImageNet, QuickDraw |
| 텍스트 | 20news, IMDB, Amazon Reviews |
| 오디오 | AudioSet |

**주요 실험 모델:**
- **ImageNet**: ResNet-18/50/101/152, NASNet-Large, DenseNet, VGG 시리즈, Inception, Xception (총 34개)
- **CIFAR-10**: VGG-11/13/16/19, ResNet-18/34/50 등 (총 13개)

**라벨 오류 검출 모델**: ResNet-50 (ImageNet), VGG (CIFAR 계열), Wide ResNet-50-2 (Caltech-256), FastText (텍스트), VGG (오디오)

### 2.5 주요 발견: 라벨 오류 규모

| 데이터셋 | 크기 | 추정 오류율 |
|----------|------|-------------|
| MNIST | 10,000 | 0.15% |
| CIFAR-10 | 10,000 | 0.54% |
| CIFAR-100 | 10,000 | 5.85% |
| ImageNet (val) | 50,000 | 5.83% (전문가 검토 시 ~20%) |
| QuickDraw | 50,426,266 | 10.12% |
| IMDB | 25,000 | 2.90% |
| Amazon Reviews | 9,996,437 | 3.90% |
| AudioSet | 20,371 | 1.35% |
| **평균** | - | **≥ 3.3%** |

### 2.6 성능 향상 분석: 벤치마크 순위 역전

**ImageNet 기준 (교정 라벨 적용 시 순위 변화):**

| 모델 | 원본 라벨 순위 | 교정 라벨 순위 |
|------|--------------|--------------|
| NASNet-Large | 1/34 | 29/34 |
| Xception | 2/34 | 24/34 |
| ResNet-18 | 34/34 | **1/34** |
| ResNet-50 | 20/24 | **2/24** |

**CIFAR-10 기준:**
- VGG-19: 원본 2위 → 교정 11위
- VGG-11: 원본 11위 → 교정 **1위**

**핵심 수치 (벤치마크 불안정성):**
- ImageNet: 노이즈 비율이 2.9% → **9%** 로 증가 시 ResNet-18이 ResNet-50 역전
- CIFAR-10: 노이즈 비율이 0.2% → **5~7%** 로 증가 시 VGG-11이 VGG-19 역전

### 2.7 한계점

1. **CL의 실패 모드**: Case 1 ($\hat{p}(\tilde{y}=j; \boldsymbol{x}, \boldsymbol{\theta}) < t_j$) 또는 Case 2 (다른 클래스 $k \neq j$에 잘못 배정)로 인해 일부 라벨 오류를 놓치거나 정상 라벨을 오류로 분류

2. **인간 검토자 한계**: MTurk 작업자의 검토 시간(평균 5초)이 전문가(67초)보다 짧아 복잡한 케이스에서 오판 발생

3. **노이즈 비율 과소추정**: CL이 플래그하지 않은 예시의 ~16%도 오류임이 전문가 검토로 확인 → ImageNet 실제 오류율은 6%가 아닌 **~20%**로 추정

4. **원인 미규명**: 고용량 모델이 교정 라벨에서 열등한 이유가 훈련 셋 노이즈 과적합인지, 하이퍼파라미터 튜닝 시 검증 셋 노이즈 과적합인지, 또는 라벨 분포 이동에 대한 민감성인지 명확히 규명하지 못함

5. **교정 라벨의 불완전성**: 5/5 동의 임계값에서도 교정 라벨이 항상 옳지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 핵심 발견: "저용량 모델의 역설적 일반화 우위"

본 논문의 가장 중요하고 놀라운 발견은 **노이즈가 있는 실제 환경에서 저용량 모델이 더 나은 일반화 성능을 보인다**는 것입니다.

```
원본(노이즈) 라벨 기준:    NASNet > ResNet-50 > ... > ResNet-18
교정(정확한) 라벨 기준:    ResNet-18 > ResNet-50 > ... > NASNet
```

### 3.2 메커니즘 분석

이 현상의 원인에 대해 논문은 두 가지 가설을 제시합니다:

**가설 1: 저용량 모델의 암묵적 정규화 효과**

고용량 모델(NASNet 등)은 모든 통계적 패턴을 더 정밀하게 학습합니다. 이는 **체계적 라벨 오류 패턴**까지도 학습한다는 것을 의미합니다. 반면 저용량 모델은 표현력의 한계로 인해 이러한 노이즈 패턴을 세밀하게 근사하지 못하여, 역설적으로 **더 진실에 가까운 특징**을 학습합니다.

수식으로 표현하면, 고용량 모델이 최소화하는 손실은:

$$\mathcal{L}_{\text{high-cap}} \approx \mathbb{E}_{(\boldsymbol{x}, \tilde{y})} [\ell(f(\boldsymbol{x}), \tilde{y})]$$

여기서 $\tilde{y}$에는 노이즈가 포함되어 있어, 결과적으로 $\tilde{y}$의 분포를 더 정확히 모델링합니다. 하지만 실제로 원하는 것은:

```math
\mathcal{L}_{\text{ideal}} = \mathbb{E}_{(\boldsymbol{x}, y^*)} [\ell(f(\boldsymbol{x}), y^*)]
```

이 두 손실의 괴리가 노이즈 비율 $N$에 비례하여 커집니다.

**가설 2: 벤치마크 과적합 (비전통적 과적합)**

대형 최신 모델들은 수년간 원본(노이즈 포함) 테스트 셋 기준으로 아키텍처/하이퍼파라미터가 선택되어 왔습니다. 이는 체계적인 라벨 오류 패턴에 점진적으로 과적합되는 현상을 초래합니다.

### 3.3 일반화 성능 향상을 위한 실용적 함의

1. **교정된 테스트 셋 사용**: $A^*$를 기준으로 모델 선택 시 실제 배포 성능을 더 정확히 예측

2. **노이즈가 많은 도메인에서의 모델 선택**: 노이즈 비율 $N$이 높은 실제 애플리케이션에서는 단순/소형 모델이 더 유리할 수 있음

3. **데이터 품질 우선 전략**: 훈련 예산이 제한될 경우, 훈련 라벨보다 **테스트 라벨 품질**을 우선시하는 것이 권장됨

4. **라벨 오류 탐지를 파이프라인에 통합**: CL 기반 필터링을 데이터 수집 후 표준 절차로 채택

### 3.4 기존 이론과의 연계

Arpit et al. (ICML 2017) "A Closer Look at Memorization in Deep Networks"와 Harutyunyan et al. (ICML 2020) "Improving Generalization by Controlling Label-Noise Information"의 이론적 분석을 실제 벤치마크 스케일에서 실증적으로 확인한 것이 본 논문의 기여입니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

**① 데이터 중심 AI(Data-Centric AI) 패러다임 강화**

이 논문은 "모델을 개선하는 것"에서 "데이터를 개선하는 것"으로 연구 패러다임을 전환하는 강력한 동기를 제공합니다. Andrew Ng이 주창한 Data-Centric AI 운동의 핵심 근거 중 하나로 기능하며, 이후 `cleanlab` 라이브러리의 확산과 함께 데이터 품질 도구 생태계를 활성화했습니다.

**② 벤치마크 설계 방법론의 재검토**

- 기존 벤치마크에 대한 신뢰를 재검토하게 만들었으며
- 새로운 벤치마크 구축 시 **라벨 품질 검증 절차**를 의무화하는 방향으로의 전환을 촉구
- "벤치마크 위생(benchmark hygiene)"이라는 개념이 주목받게 됨

**③ 노이즈 라벨 학습 연구의 방향 전환**

- 합성 노이즈가 아닌 **자연 발생 노이즈(naturally occurring noise)**를 다루는 연구의 필요성 강조
- 훈련 셋만이 아닌 **테스트 셋 노이즈의 영향**을 분석하는 새로운 연구 방향 개척

**④ 모델 복잡도와 강건성 간의 관계 재규명**

"더 큰 모델이 항상 더 좋다"는 통념에 반례를 제공하며, 노이즈 환경에서 모델 선택 전략의 재고를 촉구합니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

#### 4.2.1 데이터 품질 및 라벨 오류 탐지

| 연구 | 핵심 내용 | 본 논문과의 관계 |
|------|----------|-----------------|
| **Cleanlab (Northcutt et al., JAIR 2021)** "Confident Learning: Estimating Uncertainty in Dataset Labels" | CL 프레임워크의 공식 이론 정립 | 본 논문의 방법론적 기반 |
| **Beyond Noisy Labels (Jiang et al., ICML 2020)** | 합성 노이즈의 한계를 지적하고 실제 노이즈 환경 연구 필요성 제시 | 본 논문의 동기와 일치 |
| **CROWDLAB (Goh et al., 2022)** | 크라우드소싱 라벨을 위한 CL 확장 | 본 논문의 MTurk 검증 방법론을 발전 |

#### 4.2.2 벤치마크 신뢰성 관련 연구

| 연구 | 핵심 내용 |
|------|----------|
| **Recht et al. (ICML 2019)** "Do ImageNet Classifiers Generalize to ImageNet?" | 새로운 테스트 셋에서 정확도 하락 관찰 → 본 논문이 라벨 오류가 그 원인 중 하나임을 시사 |
| **Shankar et al. (ICML 2020)** "Evaluating Machine Accuracy on ImageNet" | ImageNet 정확도 평가의 문제점 분석 |
| **Tsipras et al. (ICML 2020)** "From ImageNet to Image Classification" | ImageNet 벤치마크의 맥락화 |

#### 4.2.3 LLM 시대의 라벨 오류 연구

LLM(Large Language Model)의 등장으로 라벨 오류 탐지에 새로운 가능성이 열렸습니다. LLM을 활용한 자동 라벨 검증 및 수정 연구들이 활발히 진행 중이며, 이는 본 논문이 제안한 "인간 검증" 단계를 부분적으로 자동화할 수 있는 방향으로 발전하고 있습니다. 단, 이 분야는 현재도 빠르게 발전 중이므로 구체적 논문명은 제가 확실히 확인할 수 없어 명시를 삼가겠습니다.

### 4.3 향후 연구 시 고려할 점

**① 테스트 셋 품질 감사(Test Set Auditing)의 표준화**

새로운 데이터셋을 공개할 때 CL 등 자동화 도구를 사용한 라벨 오류 사전 검사를 **표준 프로세스**로 채택해야 합니다.

```python
# 실용적 권장 사항 (Cleanlab 활용 예시 개념)
from cleanlab.filter import find_label_issues
label_issues = find_label_issues(labels, pred_probs)
# 인간 검증 대상을 90% 이상 줄일 수 있음
```

**② 노이즈 비율(N)을 고려한 모델 선택 전략**

$$\text{모델 선택 기준} = \begin{cases} A^* & \text{if 교정 테스트 셋 사용 가능} \\ \text{저용량 모델 우선} & \text{if } N \text{이 높은 환경} \end{cases}$$

**③ 라벨 오류의 원인별 분류 및 대응**

본 논문이 제시한 4가지 오류 유형(correctable, multi-label, neither, non-agreement)에 따른 차별화된 처리 전략 연구가 필요합니다.

**④ 고용량 모델 과적합 원인의 엄밀한 분석**

논문이 미해결로 남긴 핵심 질문:
- 훈련 셋 노이즈 과적합 vs. 검증 셋 노이즈 과적합(하이퍼파라미터 튜닝) vs. 라벨 분포 이동 민감성

이 세 원인의 기여도를 정량적으로 분리하는 연구가 필요합니다.

**⑤ 도메인 특화 벤치마크에서의 확장**

의료 영상, 법률 문서, 금융 데이터 등 **고위험(high-stakes) 도메인**에서의 라벨 오류는 벤치마크 문제를 넘어 실제 피해로 이어질 수 있으므로 특별한 주의가 필요합니다.

**⑥ 라벨 검증 예산의 최적 배분**

논문이 미해결로 남긴 또 다른 질문: 주어진 라벨 검증 예산을 훈련 셋과 테스트 셋 사이에 어떻게 최적으로 배분할 것인가에 대한 이론적·실증적 연구가 필요합니다.

---

## 참고자료

**주 논문:**
- Northcutt, C. G., Athalye, A., & Mueller, J. (2021). *Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks*. NeurIPS 2021 Datasets and Benchmarks Track. arXiv:2103.14749v4.

**논문 내 핵심 참고문헌:**
- Northcutt, C. G., Jiang, L., & Chuang, I. (2021). Confident learning: Estimating uncertainty in dataset labels. *Journal of Artificial Intelligence Research*, 70:1373–1411. [참고문헌 33]
- Arpit, D., et al. (2017). A closer look at memorization in deep networks. *ICML*. [참고문헌 2]
- Harutyunyan, H., et al. (2020). Improving generalization by controlling label-noise information in neural network weights. *ICML*. [참고문헌 13]
- Recht, B., et al. (2019). Do ImageNet classifiers generalize to ImageNet? *ICML*. [참고문헌 37]
- Shankar, V., et al. (2020). Evaluating machine accuracy on ImageNet. *ICML*. [참고문헌 40]
- Tsipras, D., et al. (2020). From ImageNet to image classification. *ICML*. [참고문헌 44]
- Jiang, L., et al. (2020). Beyond synthetic noise: Deep learning on controlled noisy labels. *ICML*. [참고문헌 18]

**오픈소스 리소스:**
- https://labelerrors.com
- https://github.com/cleanlab/label-errors
- https://github.com/cleanlab/cleanlab
