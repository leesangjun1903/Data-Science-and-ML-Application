# Residual Parameter Transfer for Deep Domain Adaptation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Rozantsev et al., 2017, arXiv:1711.07714)은 기존 도메인 적응(Domain Adaptation) 방법들이 소스-타겟 도메인 간 **파라미터를 공유하거나 제한적으로만 변환**하는 한계를 지적하며, **잔차 변환 네트워크(Residual Transformation Network)** 를 통해 소스 스트림의 파라미터로부터 타겟 스트림의 파라미터를 유연하게 예측하는 방식을 제안합니다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **잔차 파라미터 변환 구조** | 각 레이어별 auxiliary residual network로 소스→타겟 파라미터 변환 |
| **자동 복잡도 선택** | Group Lasso 기반 rank 자동 결정 메커니즘 |
| **파라미터 효율성** | 기존 방법 대비 파라미터 수 2.5배(vs [10]) 및 1.5배(vs [11]) 감소 |
| **범용 아키텍처 호환** | ResNet-50 등 매우 깊은 네트워크에도 적용 가능 |
| **지도/비지도 학습 모두 지원** | 타겟 도메인 레이블 유무와 무관하게 동작 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

기존 도메인 적응 접근법의 두 가지 주요 한계:

1. **불변 특징 학습 방식** (DANN[8], MMD 기반): 소스-타겟의 특징 분포를 통계적으로 일치시키는 과정에서 **유용한 도메인 특이적 정보가 손실**될 수 있음.

2. **파라미터 명시적 변환 방식**:
   - Domain Separation Networks[10]: 파라미터 수 **4배 증가**
   - Two-stream[11]: 파라미터 변환을 **선형 scale-shift로만 제한**, 레이어 공유 여부를 validation으로 결정해 deep architecture에 비실용적

이에 본 논문은 **유연하고 파라미터 효율적인 비선형 파라미터 변환** 메커니즘을 제안합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 벡터형 잔차 변환 (기본 형태)

소스 파라미터 $\theta_i^s$로부터 타겟 파라미터 $\theta_i^t$를 다음과 같이 변환합니다:

$$\theta_i^t = \mathbf{B}_i \sigma(\mathbf{A}_i^\top \theta_i^s + \mathbf{d}_i) + \theta_i^s, \quad \forall i \in \Omega \tag{1}$$

- $\sigma(\cdot) \in \{\tanh, \text{ReLU}\}$: 비선형 활성화 함수
- $\mathbf{A}_i, \mathbf{B}_i \in \mathbb{R}^{M_i \times k_i}$: 변환 행렬
- $k_i$: 변환의 rank (복잡도 제어), $k_i = 0$이면 파라미터 완전 공유
- $\mathbf{d}_i \in \mathbb{R}^{k_i}$: bias 항

그러나 Eq.(1)은 $k_i$ 증가 시 $(2M_i + 1)$개의 파라미터가 추가되어 메모리 비효율적입니다.

#### Step 2: 행렬형 잔차 변환 (최종 제안)

레이어 파라미터를 행렬 $\Theta_i^s, \Theta_i^t \in \mathbb{R}^{C_i \times N_i}$로 재표현하고:

$$\Theta_i^t = \mathbf{B}_i^1 \sigma\!\left(\left(\mathbf{A}_i^1\right)^\top \Theta_i^s \mathbf{A}_i^2 + \mathbf{D}_i\right)\left(\mathbf{B}_i^2\right)^\top + \Theta_i^s \tag{2}$$

| 기호 | 설명 |
|------|------|
| $\mathbf{A}_i^1, \mathbf{B}_i^1 \in \mathbb{R}^{C_i \times l_i}$ | 좌측 변환 행렬 |
| $\mathbf{A}_i^2, \mathbf{B}_i^2 \in \mathbb{R}^{N_i \times r_i}$ | 우측 변환 행렬 |
| $\mathbf{D}_i \in \mathbb{R}^{l_i \times r_i}$ | 내부 bias 행렬 |
| $l_i, r_i$ | 좌/우 변환 rank |

**파라미터 수 비교:**
- Eq.(1): $(2N_i C_i + 1)k_i$
- Eq.(2): $2(N_i r_i + C_i l_i) + r_i l_i$ → $\{l_i, r_i\} \ll \{N_i, C_i\}$이므로 훨씬 효율적

---

### 2.3 손실 함수 구성

#### 고정 복잡도 손실 (Fixed Transformation Complexity)

$$\mathcal{L}_{\text{fixed}} = \mathcal{L}_{\text{class}} + \mathcal{L}_{\text{disc}} + \mathcal{L}_{\text{stream}} \tag{4}$$

**① 분류 손실 ($\mathcal{L}_{\text{class}}$):**  
소스 및 (레이블이 있는 경우) 타겟 도메인의 표준 cross-entropy 분류 손실.

**② 도메인 불일치 손실 ($\mathcal{L}_{\text{disc}}$):**  
적대적 도메인 혼동 손실. 보조 도메인 분류기 $\phi$를 학습:

$$\mathcal{L}_{DC}(y_n) = -\frac{1}{N}\sum_{n=1}^{N}\left[y_n \log(\hat{y}_n) + (1-y_n)\log(1-\hat{y}_n)\right] \tag{5}$$

$$\mathcal{L}_{\text{disc}} = \mathcal{L}_{DC}(1 - y_n) \tag{6}$$

네트워크는 도메인 분류기를 속이는 방향으로 특징을 학습합니다 (적대적 학습).

**③ 스트림 손실 ($\mathcal{L}_{\text{stream}}$):**  
잔차 변환의 크기를 적절히 유지하는 정규화 항:

$$\mathcal{L}_{\text{stream}} = \lambda_s \left(\mathcal{L}_\omega - \mathcal{Z}(\mathcal{L}_\omega)\right) \tag{7}$$

$$\mathcal{L}_\omega = \sum_{i \in \Omega} \left\| \mathbf{B}_i^1 \sigma\!\left(\left(\mathbf{A}_i^1\right)^\top \Theta_i^s \mathbf{A}_i^2 + \mathbf{D}_i\right)\left(\mathbf{B}_i^2\right)^\top \right\|^2_{Fro} \tag{8}$$

$\mathcal{Z} = \log(\cdot)$는 barrier 함수로, $\mathcal{L}_\omega = 1$일 때 최소화되고 0 또는 $\infty$로 가면 발산하여 변환이 소멸하거나 과도하게 커지는 것을 방지합니다.

---

### 2.4 자동 복잡도 선택 (Automated Complexity Selection)

내부 변환 행렬:

$$\mathcal{T}_i = \left(\mathbf{A}_i^1\right)^\top \Theta_i^s \mathbf{A}_i^2 + \mathbf{D}_i \in \mathbb{R}^{l_i \times r_i} \tag{9}$$

Group Lasso 기반 열(column) 정규화:

$$R_c(\{\mathcal{T}_i\}) = \sum_{i \in \Omega} \left(\sqrt{N_i} \sum_c \|(\mathcal{T}_i)_{\bullet c}\|_2\right) \tag{10}$$

행(row)에 대한 정규화 $R_r(\{\mathcal{T}_i\})$도 유사하게 정의됩니다.

최종 완전 손실 함수:

$$\mathcal{L} = \mathcal{L}_{\text{fixed}} + \lambda_r (R_c + R_r) \tag{11}$$

#### Proximal Gradient Descent 최적화

$$\mathcal{T}_i^* = \underset{\mathcal{T}_i}{\arg\min} \frac{1}{2t}\left\|\mathcal{T}_i - \hat{\mathcal{T}}_i\right\|_2^2 + \lambda_r\left(R_c(\mathcal{T}_i) + R_r(\mathcal{T}_i)\right) \tag{12}$$

이를 두 단계로 분리 풀이:

$$\bar{\mathcal{T}}_i = \underset{\mathcal{T}_i}{\arg\min} \frac{1}{4t}\left\|\mathcal{T}_i - \hat{\mathcal{T}}_i\right\|_2^2 + \lambda_r R_c(\mathcal{T}_i)$$

$$\mathcal{T}_i^* = \underset{\mathcal{T}_i}{\arg\min} \frac{1}{4t}\left\|\mathcal{T}_i - \bar{\mathcal{T}}_i\right\|_2^2 + \lambda_r R_r(\mathcal{T}_i) \tag{13}$$

두 부분 문제 모두 closed-form 해를 가집니다 (Yuan & Lin, 2006[24]).

---

### 2.5 모델 구조

```
[Source Stream]     [Residual Transform]     [Target Stream]
Conv Layer Params ──→ A¹ᵢ, A²ᵢ, B¹ᵢ, B²ᵢ, Dᵢ ──→ Conv Layer Params
Conv Layer Params ──→ Residual Transform ──→ Conv Layer Params
FC Layer Params   ──→ Residual Transform ──→ FC Layer Params
       ↓                                            ↓
Source Features                             Target Features
       ↓                    ↓                       ↓
  Classification      Domain Classifier       Adversarial Loss
       Loss          (Discriminator)
```

- **학습 단계**: ① 소스 사전학습 → ② 잔차 변환 네트워크와 소스 스트림 공동 학습 (적대적 도메인 적응) → ③ 테스트 시 변환된 파라미터로 타겟 예측

---

### 2.6 성능 향상

#### SVHN → MNIST (비지도 적응)

| 모델 | 정확도 (LeNet) |
|------|--------------|
| 소스 학습만 | 60.1 [±1.10] |
| DANN [8] | 80.7 [±1.58] |
| ADDA [22] | 76.0 [±0.18] |
| Two-stream [11] | 82.8 [±0.20] |
| Domain Separation [10] | 82.78 |
| **Ours** | **84.7 [±0.17]** |

#### UAV 감지 (Synth → Real, 지도 적응)

| 모델 | AP |
|------|-----|
| 소스 학습만 | 0.377 |
| DANN [8] | 0.715 [±0.004] |
| ADDA [22] | 0.731 [±0.005] |
| Two-stream [11] | 0.732 [±0.003] |
| **Ours** | **0.743 [±0.006]** |

#### Office 데이터셋 (ResNet-50, 비지도)

| Domain pair | DANN [8] | Ours |
|-------------|----------|------|
| A→D | 79.1 | **82.7 [±0.3]** |
| A→W | 78.9 | **81.5 [±0.7]** |
| D→W | 97.5 | **98.0 [±0.1]** |

---

### 2.7 한계

1. **두 스트림의 계산 비용**: 학습 시 소스와 타겟 두 스트림 모두 forward/backward pass가 필요.
2. **네트워크 구조 변경 불가**: 레이어 수나 뉴런 수 자체의 적응은 미지원 (저자들도 향후 연구 과제로 언급).
3. **심각한 도메인 이동의 한계**: 변환이 소스 파라미터에 종속적이므로 도메인 차이가 매우 클 경우 제한적.
4. **Proximal Gradient의 수렴 복잡성**: Adam과 proximal operator의 교대 최적화 과정이 수렴 분석을 복잡하게 만듦.
5. **평가 데이터셋의 다양성 부족**: 주로 이미지 분류와 감지에 국한, NLP 등 다른 도메인 검증 없음.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 높이는 핵심 메커니즘

#### (a) 선택적 파라미터 적응
자동 rank 선택을 통해 도메인 차이가 적은 레이어는 파라미터를 거의 공유하고($r_i, l_i \approx 0$), 차이가 큰 레이어는 높은 rank로 강하게 변환합니다. Table 4의 LeNet 실험에서:

$$[\text{conv1: } [32,32] \to [31,31]] \quad [\text{conv2: } [32,32] \to [9,7]]$$

이는 저수준 특징(conv1)은 도메인 간 공통성이 높고, 고수준 특징(conv2, full3)은 도메인 특이적임을 자동으로 발견한 결과입니다.

#### (b) 과적합 방지를 위한 이중 정규화

**Stream Loss의 barrier 함수** ($\mathcal{L}_\omega = 1$ 근방 유지):
- $\mathcal{L}_\omega \to 0$: 변환 소멸 방지 (소스=타겟 강제)
- $\mathcal{L}_\omega \to \infty$: 과도한 변환 방지 (완전 분리)

**Group Lasso 정규화**:

$$R_c(\{\mathcal{T}_i\}) = \sum_{i \in \Omega}\left(\sqrt{N_i}\sum_c\|(\mathcal{T}_i)_{\bullet c}\|_2\right)$$

불필요한 변환 차원을 0으로 만들어 **유효 파라미터 수를 줄이고 과적합을 방지**합니다.

#### (c) 적대적 특징 정렬

$\mathcal{L}_{\text{disc}}$를 통해 소스-타겟 최종 특징 분포를 정렬함으로써, 하나의 분류기가 두 도메인 모두에서 잘 동작하도록 보장합니다.

### 3.2 일반화 성능의 잠재적 향상 방향

1. **다중 소스 도메인 확장**: 여러 소스 도메인으로부터 잔차 변환을 학습하면 더 강인한 타겟 적응 가능
2. **레이어 구조 적응**: 저자들이 미래 연구로 제시한 뉴런 수/레이어 수 자동 적응
3. **사전 학습 모델과의 결합**: Foundation Model의 파라미터에 잔차 변환 적용 시 적은 데이터로 강력한 적응 가능

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

#### (a) 파라미터 공간에서의 도메인 적응 패러다임 확립
기존의 특징 공간(feature space) 정렬 중심의 연구에서 **파라미터 공간(parameter space) 변환** 관점으로의 전환을 촉진했습니다. 이는 이후 다음 연구들에 직접적 영향을 미쳤습니다:

- **Meta-learning과의 결합**: MAML(Finn et al., 2017) 이후 파라미터 변환 개념이 few-shot 학습과 연결
- **Hypernetwork 연구**: 한 네트워크가 다른 네트워크의 가중치를 생성하는 구조 (Ha et al., 2017)의 도메인 적응 확장
- **Adapter 기반 전이학습**: LoRA(Hu et al., 2022)와 같이 저rank 잔차 행렬로 파라미터를 효율적으로 적응시키는 기법과 개념적 연결

#### (b) Group Lasso를 이용한 자동 구조 선택
네트워크 구조를 학습 중 자동으로 결정하는 방법론에 기여했으며, 이후 Neural Architecture Search(NAS)와 pruning 연구의 이론적 기반 중 하나로 활용됩니다.

#### (c) 잔차 연결의 도메인 적응 적용
ResNet의 잔차 연결 개념을 **파라미터 변환**에 적용한 참신한 발상으로, "도메인이 유사한 경우 잔차가 0에 가까워진다"는 우아한 귀납적 편향(inductive bias)을 제공합니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래 비교는 논문의 핵심 아이디어와 관련된 후속 연구 흐름을 중심으로 정리하였습니다. **단, 이후 연구들의 구체적 수치 비교는 본 논문 PDF에 포함되지 않으므로, 해당 부분은 공개된 논문 정보를 바탕으로 서술하며 불확실한 세부 수치는 포함하지 않겠습니다.**

| 연구 | 핵심 접근법 | 본 논문과의 관계 |
|------|------------|----------------|
| **LoRA** (Hu et al., 2022, ICLR) | 저rank 행렬로 LLM 파라미터 효율적 fine-tuning: $W = W_0 + \Delta W = W_0 + BA$ | 본 논문의 잔차 파라미터 변환과 직접적으로 유사한 구조. LoRA는 NLP 도메인으로 확장한 형태로 볼 수 있음 |
| **SpotTune** (Guo et al., 2019, CVPR) | 입력별로 fine-tuning vs. 고정 레이어를 동적 선택 | 본 논문의 자동 레이어별 복잡도 선택과 유사한 문제의식 |
| **TransMDA** 계열 연구 | Vision Transformer 기반 도메인 적응 | 본 논문의 파라미터 변환 개념을 attention 메커니즘에 적용 |
| **SHOT** (Liang et al., 2020, ICML) | 소스 없는(source-free) 도메인 적응, hypothesis transfer | 소스 스트림 없이 적응하는 방향으로 발전 |
| **DAN/CDAN** 계열 | 조건부 도메인 정렬 강화 | 본 논문의 적대적 손실 $\mathcal{L}_{\text{disc}}$ 개선 방향 |

#### LoRA와의 핵심 비교

$$\text{LoRA: } h = W_0 x + \Delta W x = W_0 x + B A x$$

$$\text{본 논문: } \Theta_i^t = \mathbf{B}_i^1 \sigma\!\left(\left(\mathbf{A}_i^1\right)^\top \Theta_i^s \mathbf{A}_i^2 + \mathbf{D}_i\right)\left(\mathbf{B}_i^2\right)^\top + \Theta_i^s$$

LoRA는 **선형 저rank 변환**인 반면, 본 논문은 **비선형 활성화 포함 + 자동 rank 선택**이라는 차별점이 있습니다. 단, LoRA는 파라미터 효율성과 확장성에서 더 단순하고 실용적인 방향으로 발전하였습니다.

---

### 4.3 향후 연구 시 고려할 점

#### (a) 기술적 고려사항

1. **Source-Free 도메인 적응으로의 확장**  
   최근 프라이버시 및 보안 문제로 소스 데이터에 대한 접근이 제한되는 실제 시나리오가 증가하고 있습니다. 본 논문의 방법론을 소스 데이터 없이 잔차 변환만으로 적응 가능하도록 확장하는 연구가 필요합니다.

2. **대규모 사전학습 모델(Foundation Models)과의 통합**  
   GPT, CLIP, ViT 등 대규모 사전학습 모델에서 잔차 파라미터 변환을 적용하면, 적은 파라미터로 강력한 도메인 적응이 가능할 것입니다. LoRA와 본 논문의 비선형 잔차 변환을 결합하는 연구가 유망합니다.

3. **다중 타겟 도메인 처리**  
   단일 소스 → 단일 타겟의 구도를 넘어, 단일 소스 → 복수 타겟으로의 동시 적응을 위한 확장이 필요합니다.

4. **동적 rank 조정**  
   현재는 학습 과정에서 rank가 감소하는 방향으로만 조정됩니다. 온라인 학습 환경에서 도메인 변화에 따라 rank를 동적으로 증가시킬 수 있는 메커니즘 연구가 필요합니다.

5. **NLP 및 멀티모달 도메인으로의 확장 검증**  
   현재 논문은 컴퓨터 비전 과제에만 검증되어 있어, 텍스트, 오디오, 멀티모달 데이터에서의 적용 가능성 연구가 요구됩니다.

#### (b) 이론적 고려사항

6. **수렴 보장**  
   Adam과 proximal gradient descent의 교대 최적화에 대한 이론적 수렴 분석이 부재합니다. 향후 연구에서 수렴 조건에 대한 이론적 보장이 제시되어야 합니다.

7. **도메인 이동 정도와 최적 rank의 관계**  
   $\|P_S - P_T\|$ (도메인 분포 차이)와 최적 변환 rank $\{l_i^\*, r_i^*\}$ 사이의 이론적 관계 규명이 필요합니다.

8. **일반화 경계(Generalization Bound) 도출**  
   Ben-David et al.의 도메인 적응 이론 프레임워크를 활용하여 잔차 파라미터 변환의 타겟 오류 경계를 이론적으로 분석할 필요가 있습니다.

---

## 참고자료

- **주 논문**: Rozantsev, A., Salzmann, M., & Fua, P. (2017). *Residual Parameter Transfer for Deep Domain Adaptation*. arXiv:1711.07714v1 [cs.CV]
- **[8]** Ganin, Y. et al. (2016). *Domain-Adversarial Training of Neural Networks*. JMLR, vol. 17.
- **[10]** Bousmalis, K. et al. (2016). *Domain Separation Networks*. NeurIPS.
- **[11]** Rozantsev, A. et al. (2017). *Beyond Sharing Weights for Deep Domain Adaptation*. arXiv Preprint.
- **[13]** He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- **[22]** Tzeng, E. et al. (2017). *Adversarial Discriminative Domain Adaptation*. CVPR.
- **[24]** Yuan, M. & Lin, Y. (2006). *Model selection and estimation in regression with grouped variables*. JRSS-B.
- **[25]** Alvarez, J. & Salzmann, M. (2016). *Learning the Number of Neurons in Deep Networks*. NeurIPS.
- **[26]** Kingma, D. & Ba, J. (2015). *Adam: A Method for Stochastic Optimisation*. ICLR.
- **후속 관련 연구**: Hu, E. J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. (arXiv:2106.09685)
- **후속 관련 연구**: Liang, J. et al. (2020). *Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation*. ICML 2020.
