# Unsupervised Domain Adaptation by Backpropagation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Ganin & Lempitsky (2015)의 논문은 **소스 도메인의 레이블 데이터**와 **타겟 도메인의 비레이블 데이터**만을 사용하여, 두 도메인 모두에서 효과적으로 동작하는 **도메인 불변(domain-invariant)** 특징을 학습할 수 있다고 주장합니다.

핵심은 다음 두 조건을 동시에 만족하는 특징을 학습하는 것입니다:

1. **(i) 판별력(Discriminativeness)**: 소스 도메인에서 클래스 레이블을 잘 예측할 수 있어야 함
2. **(ii) 도메인 불변성(Domain-invariance)**: 소스와 타겟 도메인 간의 분포 차이에 민감하지 않아야 함

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **Gradient Reversal Layer (GRL)** | 역전파 시 그래디언트 부호를 반전시키는 단순하고 범용적인 레이어 제안 |
| **End-to-End 학습** | 특징 추출, 도메인 적응, 분류기 학습을 단일 역전파 알고리즘으로 통합 |
| **이론적 근거** | $\mathcal{H}\Delta\mathcal{H}$-distance와의 연결을 통한 이론적 정당화 |
| **범용성** | 거의 모든 피드포워드 모델에 적용 가능 |
| **State-of-the-art 성능** | Office 데이터셋 등 여러 벤치마크에서 당시 최고 성능 달성 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

딥러닝 모델은 대규모 레이블 데이터에 의존하지만, 현실에서는 다음과 같은 상황이 빈번합니다:

- **도메인 시프트(Domain Shift)**: 훈련(소스) 분포 $\mathcal{S}(x, y)$와 테스트(타겟) 분포 $\mathcal{T}(x, y)$가 다름
- **비레이블 타겟 도메인**: 타겟 도메인에 레이블이 전혀 없음 (비지도 도메인 적응)
- **기존 방법의 한계**: 고정된 특징 표현을 사용하거나, 특징 학습과 도메인 적응이 분리된 파이프라인

논문은 **특징 학습(representation learning)과 도메인 적응을 하나의 학습 과정으로 통합**하는 것을 목표로 합니다.

---

### 2.2 제안 방법 및 수식

#### 모델 분해

입력 $\mathbf{x}$에 대해 모델은 세 부분으로 분해됩니다:

$$\mathbf{f} = G_f(\mathbf{x}; \theta_f) \quad \text{(Feature Extractor)}$$

$$\hat{y} = G_y(\mathbf{f}; \theta_y) \quad \text{(Label Predictor)}$$

$$\hat{d} = G_d(\mathbf{f}; \theta_d) \quad \text{(Domain Classifier)}$$

#### 목적 함수 (Objective Functional)

$$E(\theta_f, \theta_y, \theta_d) = \sum_{\substack{i=1..N \\ d_i=0}} L_y^i(\theta_f, \theta_y) - \lambda \sum_{i=1..N} L_d^i(\theta_f, \theta_d) \tag{1}$$

- $L_y^i$: 레이블 예측 손실 (소스 도메인 샘플에만 적용, 다항 로지스틱 손실)
- $L_d^i$: 도메인 분류 손실 (모든 샘플에 적용, 이진 교차 엔트로피)
- $\lambda$: 두 목적 간의 균형을 조절하는 하이퍼파라미터

#### 안장점(Saddle Point) 조건

$$(\hat{\theta}_f, \hat{\theta}_y) = \underset{\theta_f, \theta_y}{\arg\min} \; E(\theta_f, \theta_y, \hat{\theta}_d) \tag{2}$$

$$\hat{\theta}_d = \underset{\theta_d}{\arg\max} \; E(\hat{\theta}_f, \hat{\theta}_y, \theta_d) \tag{3}$$

즉, $\theta_f$와 $\theta_y$는 $E$를 **최소화**, $\theta_d$는 $E$를 **최대화**하는 **미니맥스(minimax)** 최적화 문제입니다.

#### SGD 업데이트 규칙

$$\theta_f \leftarrow \theta_f - \mu \left( \frac{\partial L_y^i}{\partial \theta_f} - \lambda \frac{\partial L_d^i}{\partial \theta_f} \right) \tag{4}$$

$$\theta_y \leftarrow \theta_y - \mu \frac{\partial L_y^i}{\partial \theta_y} \tag{5}$$

$$\theta_d \leftarrow \theta_d - \mu \frac{\partial L_d^i}{\partial \theta_d} \tag{6}$$

식 (4)에서 $-\lambda \frac{\partial L_d^i}{\partial \theta_f}$ 항이 핵심: 도메인 분류 손실을 **최대화**하는 방향으로 특징 추출기를 업데이트합니다.

#### Gradient Reversal Layer (GRL)

GRL은 순전파와 역전파에서 서로 다른 동작을 하는 "의사 함수(pseudo-function)"로 정의됩니다:

$$R_\lambda(\mathbf{x}) = \mathbf{x} \quad \text{(순전파: 항등 변환)} \tag{7}$$

$$\frac{dR_\lambda}{d\mathbf{x}} = -\lambda \mathbf{I} \quad \text{(역전파: 그래디언트에} -\lambda \text{ 곱함)} \tag{8}$$

GRL을 도입하면 전체 목적 함수를 다음과 같이 표준 SGD로 최적화 가능한 형태로 변환할 수 있습니다:

$$\tilde{E}(\theta_f, \theta_y, \theta_d) = \sum_{\substack{i=1..N \\ d_i=0}} L_y\Big(G_y(G_f(\mathbf{x}_i;\theta_f);\theta_y), y_i\Big) + \sum_{i=1..N} L_d\Big(G_d(R_\lambda(G_f(\mathbf{x}_i;\theta_f));\theta_d), y_i\Big) \tag{9}$$

GRL 삽입 후 표준 역전파를 적용하면 식 (4)-(6)의 업데이트가 자동으로 구현됩니다.

#### 적응 인자 스케줄링

학습 초반 도메인 분류기의 노이즈 신호를 억제하기 위해 $\lambda$를 점진적으로 증가시킵니다:

$$\lambda_p = \frac{2}{1 + \exp(-\gamma \cdot p)} - 1 \tag{14}$$

여기서 $p$는 학습 진행률(0~1), $\gamma = 10$으로 설정합니다.

---

### 2.3 모델 구조

```
입력 x
    │
    ▼
┌─────────────────────┐
│  Feature Extractor  │  (CNN 2~3개 conv layer, θ_f)
│       G_f           │
└─────────┬───────────┘
          │ f (특징 벡터)
    ┌─────┴──────┐
    │            │
    ▼            ▼ [GRL: ×(-λ)]
┌───────┐   ┌──────────────┐
│Label  │   │   Domain     │
│Predictor  │   Classifier │
│G_y(θ_y)   │   G_d(θ_d)  │
└───┬───┘   └──────┬───────┘
    │              │
    ▼              ▼
클래스 레이블 y   도메인 레이블 d
   (Loss L_y)     (Loss L_d)
```

- **Feature Extractor**: 2~3개 합성곱 레이어 (AlexNet 기반 사전학습 가능)
- **Label Predictor**: 소스 도메인 레이블 예측 (훈련 및 테스트 시 사용)
- **Domain Classifier**: $x \to 1024 \to 1024 \to 2$ (MNIST: $x \to 100 \to 2$)

---

### 2.4 이론적 근거: $\mathcal{H}\Delta\mathcal{H}$-distance

Ben-David et al. (2010)의 이론에 의하면 타겟 도메인에서의 오류에 대한 상한이 존재합니다:

$$\varepsilon_{\mathcal{T}}(h) \leq \varepsilon_{\mathcal{S}}(h) + \frac{1}{2} d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{S}, \mathcal{T}) + C \tag{11}$$

여기서 $d_{\mathcal{H}\Delta\mathcal{H}}$는 두 도메인 분포 간의 불일치 거리입니다:

$$d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{S}, \mathcal{T}) = 2 \sup_{h_1, h_2 \in \mathcal{H}} |P_{\mathbf{f}\sim\mathcal{S}}[h_1(\mathbf{f}) \neq h_2(\mathbf{f})] - P_{\mathbf{f}\sim\mathcal{T}}[h_1(\mathbf{f}) \neq h_2(\mathbf{f})]| \tag{10}$$

GRL을 통한 역전파는 $d_{\mathcal{H}_p\Delta\mathcal{H}_p}(\mathcal{S}, \mathcal{T})$를 줄이는 방향으로 표현 공간을 변환하여, 이론적으로 타겟 도메인에서의 오류 상한을 낮춥니다.

---

### 2.5 성능 향상

**Table 1: 소규모 이미지 분류 실험 결과**

| Method | MNIST→MNIST-M | SYN→SVHN | SVHN→MNIST | SYN SIGNS→GTSRB |
|---|---|---|---|---|
| Source Only | .5225 | .8674 | .5490 | .7900 |
| SA (Fernando et al., 2013) | .5690 (+4.1%) | .8644 (-5.5%) | .5932 (+9.9%) | .8165 (+12.7%) |
| **Proposed (DANN)** | **.7666 (+52.9%)** | **.9109 (+79.7%)** | **.7385 (+42.6%)** | **.8865 (+46.4%)** |
| Train on Target (상한) | .9596 | .9220 | .9942 | .9980 |

**Table 2: Office 데이터셋 결과**

| Method | Amazon→Webcam | DSLR→Webcam | Webcam→DSLR |
|---|---|---|---|
| GFK | .214 | .691 | .650 |
| DDC | .605 | .948 | .985 |
| DAN | .645 | .952 | .986 |
| Source Only | .642 | .961 | .978 |
| **DANN (Proposed)** | **.730** | **.964** | **.992** |

---

### 2.6 한계

1. **MNIST→SVHN 실패**: 도메인 간 격차가 너무 클 경우 적응 실패 (논문에서 명시)
2. **하이퍼파라미터 $\lambda$ 민감성**: 적절한 $\lambda$ 설정이 성능에 중요하나 비지도 방식으로 최적화하기 어려움
3. **도메인 분류기 용량 제한**: 단순한 도메인 분류기로 복잡한 도메인 격차를 완전히 포착하지 못할 가능성
4. **클래스 조건부 정렬 부재**: 도메인 전반에 걸쳐 정렬하지만 클래스별 조건부 정렬은 수행하지 않음
5. **소규모 데이터셋 한계**: Office 데이터셋에서는 사전학습 모델(AlexNet) 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 도메인 불변 특징과 일반화 이론

논문은 $\mathcal{H}\Delta\mathcal{H}$-distance를 줄임으로써 타겟 도메인의 오류 상한을 낮출 수 있음을 보였습니다. 이는 **도메인 불변 특징이 일반화 성능 향상에 직결**됨을 의미합니다.

$$\varepsilon_{\mathcal{T}}(h) \leq \underbrace{\varepsilon_{\mathcal{S}}(h)}_{\text{소스 오류}} + \underbrace{\frac{1}{2}d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{S},\mathcal{T})}_{\text{DANN이 줄이는 항}} + C$$

GRL을 통해 $d_{\mathcal{H}\Delta\mathcal{H}}$를 최소화하면 타겟 도메인 오류가 소스 도메인 오류에 근접하게 됩니다.

### 3.2 정규화 효과

논문의 실험에서 **도메인 분류기를 제거하면 과적합이 더 심해진다**는 관찰이 있었습니다. 즉, GRL 기반 도메인 적응이 **암묵적 정규화(implicit regularization)** 역할을 합니다:

- 도메인 불변 특징 = 소스 도메인 특유의 노이즈 패턴에 과적합되지 않는 특징
- 이러한 특징은 **더 범용적(generalizable)** 표현을 의미

### 3.3 합성→실제 데이터 일반화

SYN NUMBERS→SVHN 실험에서 갭의 **79.7%**를 커버하고, SYN SIGNS→GTSRB에서 **46.4%**를 커버함으로써, 합성 데이터로 훈련된 모델이 실제 데이터로 일반화되는 실용적 가능성을 보였습니다.

### 3.4 반지도 학습으로의 확장 가능성

논문은 반지도 설정(430개의 타겟 레이블 공개)에서도 추가적인 성능 향상이 가능함을 보여, 레이블 수량에 따른 유연한 일반화가 가능함을 시사합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 후속 연구에 미치는 영향

본 논문은 **도메인 적응 분야의 패러다임**을 바꿨으며, 다음과 같은 광범위한 영향을 미쳤습니다:

1. **적대적 학습(Adversarial Training)의 도메인 적응 적용**: GAN의 판별자-생성자 구조와 유사한 아이디어가 도메인 적응에서 표준으로 자리잡음
2. **End-to-End 도메인 적응**: 분리된 파이프라인 대신 통합 학습의 가능성 제시
3. **다양한 태스크로의 확장**: NLP, 음성 인식, 의료 영상 등으로 DANN 원리가 확산

### 4.2 2020년 이후 관련 최신 연구 비교 분석

#### (1) CDAN (Conditional Domain Adversarial Network)

> Long et al., "Conditional Adversarial Domain Adaptation," NeurIPS 2018

DANN의 한계인 **클래스 조건부 정렬 부재**를 해결합니다. 도메인 분류기에 클래스 예측 정보를 조건으로 추가합니다:

$$G_d(\mathbf{f} \otimes \hat{y}; \theta_d)$$

여기서 $\otimes$는 외적(outer product)을 의미하여, 클래스별 도메인 정렬이 가능합니다.

#### (2) MDD (Margin Disparity Discrepancy)

> Zhang et al., "Bridging Theory and Algorithm for Domain Adaptation," ICML 2019

$\mathcal{H}\Delta\mathcal{H}$-distance의 한계를 보완하여 **Margin Disparity Discrepancy**라는 새로운 이론적 척도를 도입합니다:

$$\text{MDD}(\mathcal{S}, \mathcal{T}) = \sup_{h \in \mathcal{H}} \left(\mathbb{E}_{\mathcal{T}}[\phi(\sigma(h))] - \mathbb{E}_{\mathcal{S}}[\phi(\sigma(h))]\right)$$

이를 통해 DANN보다 이론적으로 더 엄밀한 일반화 경계를 제공합니다.

#### (3) SHOT (Source Hypothesis Transfer)

> Liang et al., "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation," ICML 2020

**소스 데이터 없이** 사전학습된 소스 모델만을 사용하여 타겟 도메인에 적응합니다. DANN이 소스 데이터를 학습 시 필요로 하는 반면, SHOT은 **소스-프리(source-free)** 도메인 적응을 실현합니다. 이는 데이터 프라이버시 문제를 해결하는 중요한 발전입니다.

$$\min_{\theta_f} \mathcal{H}(\hat{p}) - \mathbb{E}_{\hat{p}}[\log \hat{p}]$$

(정보 최대화 + 슈도 레이블 기반 최적화)

#### (4) TransDA / Transformer 기반 DA

> Xu et al., "CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation," ICLR 2022

Vision Transformer(ViT)를 백본으로 사용하여 **어텐션 메커니즘**으로 소스-타겟 간 크로스 도메인 정렬을 수행합니다. DANN의 CNN 백본 의존성을 극복하고 더 강력한 특징 추출을 가능하게 합니다.

#### (5) 비교 요약표

| 방법 | 핵심 아이디어 | DANN 대비 개선점 | 한계 |
|---|---|---|---|
| **DANN** (2015) | GRL + 도메인 분류기 | 기준점 | 클래스 무조건 정렬, 소스 필요 |
| **CDAN** (2018) | 클래스 조건부 적대 학습 | 클래스별 정렬 | 소스 필요 |
| **MDD** (2019) | 마진 불일치 최소화 | 이론적 엄밀성 | 복잡한 최적화 |
| **SHOT** (2020) | 소스-프리 적응 | 데이터 프라이버시 | 소스 모델 필요 |
| **CDTrans** (2022) | Transformer 기반 | 강력한 특징 추출 | 계산 비용 |

### 4.3 향후 연구 시 고려할 점

#### ① 이론적 측면
- $\mathcal{H}\Delta\mathcal{H}$-distance 기반 경계가 느슨할 수 있으므로, **더 타이트한 일반화 경계** 이론 개발 필요
- 도메인 적응 성공/실패 조건에 대한 **이론적 충분조건** 탐구

#### ② 클래스 조건부 정렬
- DANN은 클래스 정보를 무시하고 전역적 분포를 맞추므로 **부정적 전이(negative transfer)** 위험
- 클래스별, 인스턴스별 적응 메커니즘 강화 필요

#### ③ 소스 데이터 프라이버시
- 실제 응용에서 소스 데이터 접근이 불가한 경우가 많으므로 **소스-프리 도메인 적응** 연구 필요
- 연합학습(Federated Learning)과의 결합 탐구

#### ④ 멀티 소스 / 멀티 타겟 확장
- DANN은 단일 소스-타겟 쌍에 설계됨
- 여러 소스/타겟 도메인을 동시에 처리하는 **멀티 도메인 적응** 방향 탐구

#### ⑤ 대형 언어 모델(LLM) / Vision-Language 모델과의 결합
- 사전학습된 대형 모델(CLIP, GPT 등)과 도메인 적응 기법의 시너지 탐구
- Prompt tuning, LoRA 등 효율적 미세조정 방법과의 결합

#### ⑥ 하이퍼파라미터 $\lambda$ 자동화
- $\lambda$ 스케줄링이 성능에 민감하므로 **메타학습** 또는 **AutoML** 기반의 자동 조율 연구

---

## 참고 자료

1. **Ganin, Y., & Lempitsky, V. (2015).** "Unsupervised Domain Adaptation by Backpropagation." *Proceedings of the 32nd ICML*, JMLR: W&CP volume 37. *(본 논문 원문 PDF)*

2. **Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W. (2010).** "A theory of learning from different domains." *Machine Learning, 79(1-2)*, 151-175.

3. **Long, M., Cao, Z., Wang, J., & Jordan, M. I. (2018).** "Conditional Adversarial Domain Adaptation." *NeurIPS 2018.*

4. **Zhang, Y., Liu, T., Long, M., & Jordan, M. (2019).** "Bridging Theory and Algorithm for Domain Adaptation." *ICML 2019.*

5. **Liang, J., Hu, D., & Feng, J. (2020).** "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation." *ICML 2020.*

6. **Xu, T., Chen, W., Wang, P., Wang, F., Li, H., & Jin, R. (2022).** "CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation." *ICLR 2022.*

7. **Goodfellow, I., et al. (2014).** "Generative Adversarial Nets." *NeurIPS 2014.* *(GRL과 아이디어 연관)*

8. **Shimodaira, H. (2000).** "Improving predictive inference under covariate shift by weighting the log-likelihood function." *Journal of Statistical Planning and Inference, 90(2)*, 227-244.
