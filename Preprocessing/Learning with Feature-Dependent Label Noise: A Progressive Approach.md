# Learning with Feature-Dependent Label Noise: A Progressive Approach

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문은 실세계의 레이블 노이즈가 **특징(feature)에 의존적(feature-dependent)** 이며 이질적(heterogeneous)임을 강조하고, 기존의 i.i.d. 노이즈 가정이 현실을 제대로 반영하지 못한다는 문제를 지적합니다. 이를 해결하기 위해 **Polynomial Margin Diminishing (PMD) 노이즈** 패밀리를 새롭게 정의하고, 이에 기반한 **Progressive Label Correction (PLC)** 알고리즘을 제안합니다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| 새로운 노이즈 패밀리 정의 | PMD 노이즈: 결정 경계 근방 외 영역에서 노이즈 상한이 다항식으로 감소 |
| 이론적 수렴 보장 | PLC가 Bayes 최적 분류기에 수렴함을 증명한 **최초의 data-recalibrating 방법** |
| 실용적 성능 | CIFAR-10/100, Clothing1M, Food-101N, ANIMAL-10N에서 SOTA 달성 |
| 일반화된 노이즈 프레임워크 | i.i.d., BCN 노이즈를 모두 포괄하는 더 넓은 이론적 설정 제공 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 접근법의 한계:

- **i.i.d. 가정 기반 방법**: 이론적 보장은 있으나 현실에서 성능 저하
- **Data-recalibrating 방법**: 실용적 성능은 우수하나 이론적 근거 부재, 하이퍼파라미터 튜닝에 민감

본 논문은 이 두 가지 한계를 동시에 극복하고자 합니다.

---

### 2.2 PMD 노이즈 정의 (수식)

**Definition 1 (PMD Noise)**

노이즈 함수 $\tau_{0,1}(\boldsymbol{x})$와 $\tau_{1,0}(\boldsymbol{x})$가 **Polynomial Margin Diminishing (PMD)** 하다는 것은, 상수 $t_0 \in (0, \frac{1}{2})$와 $c_1, c_2 > 0$가 존재하여 다음을 만족하는 것입니다:

$$\tau_{1,0}(\boldsymbol{x}) \leq c_1[1 - \eta(\boldsymbol{x})]^{1+c_2}; \quad \forall \eta(\boldsymbol{x}) \geq \frac{1}{2} + t_0$$

$$\tau_{0,1}(\boldsymbol{x}) \leq c_1 \eta(\boldsymbol{x})^{1+c_2}; \quad \forall \eta(\boldsymbol{x}) \leq \frac{1}{2} - t_0$$

여기서:
- $\eta(\boldsymbol{x}) = P[y=1|\boldsymbol{x}]$: 사후 확률 (posterior probability)
- $t_0$: 결정 경계 근방의 "마진(margin)"
- $|\eta(\boldsymbol{x}) - \frac{1}{2}| < t_0$ 영역에서는 노이즈 레벨이 **임의적으로** 허용됨

기존 노이즈와의 비교:

| 노이즈 유형 | 특징 | PMD 포함 여부 |
|---|---|---|
| Uniform (i.i.d.) | $\tau$가 상수 | ✅ 포함 |
| BCN (Boundary Consistent) | $\tau$가 단조 감소 | ✅ 포함 |
| PMD | 마진 바깥에서만 다항식 상한 | ✅ 가장 일반적 |

---

### 2.3 제안 방법: PLC 알고리즘

#### 직관적 아이디어

1. 분류기가 **높은 신뢰도**를 보이는 데이터 포인트는 PMD 노이즈 가정 하에 Bayes 최적 분류기와 일치할 가능성이 높음
2. 이러한 "pure region"에서부터 시작해 레이블을 점진적으로 교정

#### 알고리즘 (Algorithm 1: Progressive Label Correction)

```
Input: 노이즈 데이터셋 S̃, 신경망 f(x), 스텝 크기 β, 초기/종료 임계값 (T₀, T_end), 워밍업 m, 총 라운드 N
```

**레이블 교정 기준**:

$$\tilde{y}_i^t \leftarrow \mathbb{I}_{\{f(\boldsymbol{x}_i) \geq \frac{1}{2}\}}$$

**조건**: $\left|f(\boldsymbol{x}_i) - \frac{1}{2}\right| \geq \theta$ 일 때 교정 수행, 여기서 $\theta = \frac{1}{2} - T$

**임계값 업데이트**:

$$T \leftarrow \min(T(1+\beta), T_{end})$$

**다중 클래스 확장**: 분류기 예측 $h_x = \arg\max_i f_i(\boldsymbol{x})$에 대해,

$$|f_{h_x}(\boldsymbol{x}) - f_{\tilde{y}}(\boldsymbol{x})| > \theta \Rightarrow \tilde{y} \leftarrow h_x$$

실용적으로는 로그 차이 $|\log f_{\tilde{y}}(\boldsymbol{x}) - \log f_{h_x}(\boldsymbol{x})|$를 사용합니다.

---

### 2.4 이론적 분석

#### 핵심 정의

**Definition 2 (Level set $(\alpha, \epsilon)$-consistency)**

$$|f(\boldsymbol{x}) - \eta(\boldsymbol{x})| \leq \alpha \mathbb{E}_{(\boldsymbol{z},\tilde{y}) \sim D(\boldsymbol{z}, \tilde{\eta}(\boldsymbol{z}))} \left[\mathbb{1}_{\{\tilde{y}_z \neq \eta^*(\boldsymbol{z})\}}(\boldsymbol{z}) \middle| \left|\eta(\boldsymbol{z}) - \frac{1}{2}\right| \geq \left|\eta(\boldsymbol{x}) - \frac{1}{2}\right|\right] + \epsilon$$

**Definition 3 (Level set bounded distribution)**

분포 $D$가 $(c_\*, c^\*)$ -bounded: $0 < c_* \leq g(t) \leq c^\*$, 밀도-불균형 비율 $\ell_D = c^\*/c_*$

#### 주요 정리 (Theorem 1)

**Assumption 1** 하에서, PMD 마진 $t_0$을 가진 임의의 노이즈 $\tau$에 대해, $e_0 = \max\!\left(t_0, \frac{\alpha + \epsilon}{1 + 2\alpha}\right)$로 정의하면:

초기화 조건:
1. $T_0 < \frac{1}{2} - e_0$
2. $m \geq \frac{\ell\alpha}{\epsilon} \log\!\left(\frac{2T_0}{1 - 2e_0}\right)$
3. $N \geq m + \frac{1}{\beta} \log\!\left(\frac{T_0}{3\epsilon}\right)$
4. $T_{end} \leq 3\epsilon$
5. $\frac{\epsilon}{\alpha\ell} \leq \beta \leq \frac{2\epsilon}{\alpha\ell}$

를 만족하면:

```math
\mathbb{P}_{\boldsymbol{x} \sim D}\left[y_{f_{final}}(\boldsymbol{x}) = \eta^*(\boldsymbol{x})\right] \geq 1 - 3c^*\epsilon
```

#### 보조 정리 구조

**Lemma 1 (One round purity improvement)**: Pure level set $L(e, \eta)$가 존재하면, 한 라운드 후 새 level set의 크기가 다음을 만족:

$$\frac{1}{2} - e_{new} \geq \left(1 + \frac{\epsilon}{\alpha\ell}\right)\left(\frac{1}{2} - e\right)$$

**Lemma 2 (Warm-up rounds)**: $m \geq \frac{\ell\alpha}{\epsilon}\log\!\left(\frac{2T_0}{1-2e_0}\right)$ 라운드 후, $L\!\left(\frac{1}{2} - T_0, \eta\right)$이 pure해짐

**Lemma 3**: Lemma 1 + Lemma 2를 결합하여 최종 수렴 보장

---

### 2.5 모델 구조

- **기본 백본**: ResNet-50 (ImageNet 사전학습), VGG-19 (ANIMAL-10N)
- **출력**: Softmax 출력 $f(\boldsymbol{x}) \in [0,1]$을 신뢰도로 활용
- **손실 함수**: 표준 Cross-Entropy (교정된 레이블에 적용)
- **최적화**: SGD, 배치 크기 128, 초기 학습률 0.01

---

### 2.6 성능 향상

#### CIFAR-10/100 (합성 특징 의존 노이즈)

| Dataset | Noise | Standard | LRT | **PLC (ours)** |
|---|---|---|---|---|
| CIFAR-10 | Type-I (35%) | 78.11 | 80.98 | **82.80** |
| CIFAR-10 | Type-II (70%) | 45.57 | 44.67 | **46.04** |
| CIFAR-100 | Type-II (35%) | 57.83 | 57.25 | **63.68** |
| CIFAR-100 | Type-I (70%) | 39.32 | 45.29 | **45.92** |

#### 실세계 데이터셋

| Dataset | Best Baseline | **PLC** |
|---|---|---|
| Clothing1M | 73.49 (PENCIL) | **74.02** |
| Food-101N | 83.95 (CleanNet) | **85.28** |
| ANIMAL-10N | 81.8 (SELFIE) | **83.4** |

---

### 2.7 한계점

1. **점근적 분석(asymptotic analysis)만 제공**: 샘플 복잡도(sample complexity)에 대한 유한 샘플 보장이 없음. 논문 자체에서 "leave the sample complexity for future work"라고 명시
2. **이진 분류 이론**: 이론적 분석은 이진 분류에 집중되며, 다중 클래스는 휴리스틱 확장
3. **워밍업 epoch 필요**: 초기 네트워크가 과적합되기 전에 정지해야 하며, 이는 추가 튜닝 필요
4. **PMD 가정의 검증 어려움**: 실세계에서 PMD 조건 만족 여부를 사전에 확인하기 어려움
5. **하이퍼파라미터 민감도**: $T_0$, $\beta$, $m$ 등의 파라미터 설정이 성능에 영향을 미침 (단, 실험에서 상당한 강인성 보임)

---

## 3. 일반화 성능 향상 가능성

### 3.1 이론적 일반화 보장

PLC의 핵심 이론적 강점은 **Bayes-consistency**입니다:

```math
\mathbb{P}_{\boldsymbol{x} \sim D}\left[y_{f_{final}}(\boldsymbol{x}) = \eta^*(\boldsymbol{x})\right] \geq 1 - 3c^*\epsilon
```

이는 충분한 데이터가 있을 때 최종 분류기가 Bayes 최적 분류기에 수렴함을 의미하며, 이것이 일반화 성능의 이론적 상한을 보장합니다.

### 3.2 일반화를 향상시키는 메커니즘

#### (a) Progressive Curriculum (점진적 교육과정)

신뢰도 임계값 $\theta$를 점진적으로 낮추는 전략은 일종의 **커리큘럼 학습**으로 작동합니다:
- 초기: 고신뢰도 (쉬운) 샘플만 교정 → 안정적인 초기 학습
- 후기: 저신뢰도 (어려운) 샘플까지 점진적 포함 → 결정 경계 근방의 어려운 샘플도 점진적 처리

$$\theta = \frac{1}{2} - T, \quad T \text{가 점진적으로 증가}$$

#### (b) Label Purity 단조 증가

Lemma 1에 의해 각 라운드마다 "pure" 영역이 기하급수적으로 확장:

$$\frac{1}{2} - e_{new} \geq \left(1 + \frac{\epsilon}{\alpha\ell}\right)^{t} \cdot \left(\frac{1}{2} - e_0\right)$$

레이블 순도가 증가함에 따라 모델이 더 깨끗한 신호로 학습되어 일반화 성능이 향상됩니다.

#### (c) Feature-Dependent Noise 처리의 중요성

i.i.d. 가정 기반 방법들은 결정 경계 근방의 어려운 샘플에서 과도하게 보수적이거나 잘못된 교정을 수행합니다. PLC는 PMD 가정 하에서 이러한 영역을 명시적으로 처리하여 **결정 경계 근방 샘플의 일반화 성능을 개선**합니다.

#### (d) 실험적 증거

- CIFAR-100에서 Type-II 35% 노이즈: Standard 57.83% → PLC **63.68%** (약 6%p 향상)
- 하이브리드 노이즈(Type-I + 60% Uniform)에서: Standard 35.97% → PLC **51.68%** (약 16%p 향상)
- 이는 단순한 노이즈 제거를 넘어 **분포 전반에 걸친 일반화**가 개선됨을 시사

### 3.3 일반화 한계

- 이론적 보장이 **점근적**이므로, 유한 샘플에서는 $3c^*\epsilon$ 오차 항이 결정 경계 근방 샘플의 일반화를 제한
- Out-of-distribution(OOD) 일반화에 대한 보장은 제공되지 않음

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (a) 이론적 프레임워크 확립
PMD 노이즈 패밀리는 feature-dependent 레이블 노이즈 연구의 **표준 이론적 설정**으로 자리잡을 가능성이 높습니다. 이는:
- 새로운 노이즈 패밀리의 비교 기준점 제공
- Bayes-consistency 증명을 위한 방법론적 템플릿 제시

#### (b) Data-recalibrating 방법의 이론화
기존에 경험적으로만 사용되던 data-recalibrating 방법(Tanaka et al., Yi & Wu 등)에 이론적 근거를 부여하는 연구 방향을 제시합니다.

#### (c) 실용적 레이블 노이즈 학습의 발전
- 실세계 크라우드소싱 데이터, 웹 스크래핑 데이터에서의 노이즈 처리 연구에 직접적 영향
- LLM 시대의 자동 레이블링(auto-labeling) 오류 처리에 응용 가능

### 4.2 향후 연구 시 고려할 점

#### (a) 샘플 복잡도 이론
논문이 명시적으로 남긴 미해결 문제입니다. 유한 샘플에서 얼마나 많은 데이터가 필요한지에 대한 이론적 분석이 필요합니다:
$$n = O\left(\frac{?}{\epsilon^2}\right) \text{ 샘플이 필요}$$

#### (b) 다중 클래스 이론 확장
현재 이진 분류에 집중된 이론을 다중 클래스로 엄밀하게 확장하는 연구가 필요합니다.

#### (c) PMD 노이즈 검증 방법
실세계에서 주어진 데이터셋이 PMD 조건을 만족하는지 **사전 검증**하거나, 더 일반적인 조건을 발견하는 연구가 필요합니다.

#### (d) 반지도 학습(Semi-supervised Learning)과의 결합
PLC의 레이블 교정 메커니즘을 반지도 학습 프레임워크와 결합하면 더 적은 클린 데이터로 높은 성능을 달성할 수 있을 것입니다.

#### (e) 대형 언어 모델(LLM) 시대의 적용
- GPT-4, Claude 등 LLM으로 자동 생성된 레이블의 노이즈는 feature-dependent일 가능성이 높음
- PMD 가정이 LLM 기반 레이블링 오류에 적용 가능한지 검토 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 비교는 논문의 reference와 제 사전 학습 지식에 기반하며, 일부 2021년 이후 연구는 사전 지식에 의존합니다. 정확도에 대한 주의가 필요합니다.

### 5.1 동시대/후속 연구 비교

| 연구 | 방법 | 노이즈 가정 | 이론적 보장 | 비고 |
|---|---|---|---|---|
| **PLC (본 논문, ICLR 2021)** | Progressive label correction | PMD (feature-dependent) | Bayes-consistency 수렴 보장 | 최초의 이론 보장 data-recalibrating |
| Chen et al. (AAAI 2021) - "Beyond Class-Conditional Assumption" | 네트워크 예측 평균화 | Instance-dependent | ❌ 이론 보장 없음 | 실용적이나 이론 부재 |
| Cheng et al. (ICML 2020) - "Learning with Bounded Instance- and Label-Dependent Label Noise" | Active learning | Instance-dependent (bounded) | ✅ 일부 보장 | Oracle 필요 (클린 레이블 쿼리) |
| Menon et al. (Machine Learning 2018) - "Learning from Binary Labels with Instance-Dependent Noise" | Loss function design | Instance-dependent | ✅ Bayes-consistency | 딥러닝 미적용 |

### 5.2 2021년 이후 관련 연구 방향

> **주의**: 아래 내용은 제 사전 학습 지식에 기반하며, 논문 PDF에 직접 인용된 내용이 아닙니다. 정확성을 위해 원문 확인을 권장합니다.

#### Instance-Dependent Noise 연구 흐름
- **Part-dependent label noise** 연구: 이미지의 특정 패치(part)에 따라 노이즈가 결정되는 더 구체적인 모델
- **Noisy Labels with Large Language Models**: ChatGPT 등 LLM을 활용한 레이블 노이즈 감지 및 교정

#### Semi-supervised + Noisy Labels 결합
- DivideMix (Li et al., ICLR 2020): GMM을 사용한 클린/노이즈 샘플 분리 + MixMatch
  - PLC와 비교: DivideMix는 반지도 학습을 활용하여 실용적 성능이 높지만, PMD와 같은 일반 노이즈 이론 없음

#### Foundation Model 기반 접근
- 사전학습된 대형 모델의 특징 공간을 활용한 노이즈 감지 연구가 활발히 진행 중
- PLC의 PMD 가정이 이 맥락에서 어떻게 적용될지는 미탐구 영역

### 5.3 PLC의 차별성 유지

2020년 이후에도 **이론적 수렴 보장을 갖춘 feature-dependent 노이즈 처리 방법**은 여전히 희소하며, PLC의 이론적 기여는 현재까지도 유의미합니다.

---

## 참고 자료

**주요 참고 논문 (논문 PDF 내 인용)**:
1. Zhang, Y., Zheng, S., Wu, P., Goswami, M., & Chen, C. (2021). *Learning with Feature-Dependent Label Noise: A Progressive Approach.* ICLR 2021. (본 논문)
2. Cheng, J., Liu, T., Ramamohanarao, K., & Tao, D. (2020). *Learning with Bounded Instance- and Label-Dependent Label Noise.* ICML 2020.
3. Menon, A. K., Van Rooyen, B., & Natarajan, N. (2018). *Learning from Binary Labels with Instance-Dependent Noise.* Machine Learning.
4. Chen, P., Ye, J., Chen, G., Zhao, J., & Heng, P.-A. (2021). *Beyond Class-Conditional Assumption: A Primary Attempt to Combat Instance-Dependent Label Noise.* AAAI 2021.
5. Tanaka, D., Ikami, D., Yamasaki, T., & Aizawa, K. (2018). *Joint Optimization Framework for Learning with Noisy Labels.* CVPR 2018.
6. Yi, K., & Wu, J. (2019). *Probabilistic End-to-end Noise Correction for Learning with Noisy Labels.* CVPR 2019.
7. Zheng, S., Wu, P., Goswami, A., Goswami, M., Metaxas, D., & Chen, C. (2020). *Error-Bounded Correction of Noisy Labels.* ICML 2020.
8. Du, J., & Cai, Z. (2015). *Modelling Class Noise with Symmetric and Asymmetric Distributions.* AAAI 2015.
9. Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). *Understanding Deep Learning Requires Rethinking Generalization.* ICLR 2017.
10. Song, H., Kim, M., & Lee, J.-G. (2019). *SELFIE: Refurbishing Unclean Samples for Robust Deep Learning.* ICML 2019.

**코드 저장소**: https://github.com/pxiangwu/PLC
