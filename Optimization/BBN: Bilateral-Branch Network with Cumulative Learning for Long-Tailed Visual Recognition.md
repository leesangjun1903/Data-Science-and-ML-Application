# BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 발견 (Key Discovery)

BBN 논문의 가장 중요한 발견은 기존 클래스 재균형(class re-balancing) 전략의 **이중적 효과**를 실험적으로 규명한 것입니다:

> **"Re-balancing 방법은 분류기 학습(classifier learning)은 크게 향상시키지만, 동시에 표현 학습(representation learning)의 품질을 손상시킨다."**

이를 그림으로 표현하면:

```
기존 Re-balancing 접근법:
✅ Classifier Learning  →  향상 (tail 클래스 분류 결정 경계 개선)
❌ Representation Learning → 손상 (intra-class 분포의 응집력 저하)
```

### 주요 기여 (Main Contributions)

| 기여 항목 | 내용 |
|-----------|------|
| **메커니즘 규명** | Re-balancing 전략이 어떻게 작동하는지 2단계 실험으로 최초 분석 |
| **BBN 모델 제안** | 표현 학습과 분류기 학습을 동시에 처리하는 통합 양방향 브랜치 네트워크 |
| **누적 학습 전략** | α를 epoch에 따라 적응적으로 조정하는 Cumulative Learning Strategy |
| **대회 우승** | iNaturalist 2019 대규모 종 분류 대회 1위 달성 |
| **오픈소스 공개** | [https://github.com/Megvii-Nanjing/BBN](https://github.com/Megvii-Nanjing/BBN) |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**Long-Tailed Distribution Problem:**
- 실세계 데이터셋은 소수의 head 클래스가 대부분의 데이터를 차지하고, 대다수의 tail 클래스는 극소량의 샘플만 보유
- 기존 re-balancing 방법들은 분류 정확도를 높이나, **표현 학습의 품질을 희생**시키는 트레이드오프 문제 존재

**구체적 문제점:**
- **Re-sampling**: tail 데이터 과적합(over-sampling) 또는 전체 분포 under-fitting(under-sampling) 위험
- **Re-weighting**: 원래 데이터 분포를 왜곡시켜 보편적 특징 학습 방해

**2단계 검증 실험 결과 (CIFAR-100-IR50 기준):**

```
표현학습 방법 고정 시 (수직 방향 비교):
  RW/RS classifier → CE classifier보다 낮은 오류율 ✅ (분류기 학습 향상)

분류기 학습 방법 고정 시 (수평 방향 비교):
  CE representation → RW/RS representation보다 낮은 오류율 ✅ (표현 학습 우수)
```

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) 출력 Logit 계산

두 브랜치에서 나온 특징 벡터 $\mathbf{f}_c \in \mathbb{R}^D$ (Conventional Branch)와 $\mathbf{f}_r \in \mathbb{R}^D$ (Re-balancing Branch)를 가중 결합:

$$\mathbf{z} = \alpha \mathbf{W}_c^\top \mathbf{f}_c + (1 - \alpha) \mathbf{W}_r^\top \mathbf{f}_r $$

여기서 $\mathbf{z} \in \mathbb{R}^C$는 예측 logit 벡터, $\mathbf{W}_c, \mathbf{W}_r \in \mathbb{R}^{D \times C}$는 각 브랜치의 분류기 가중치.

#### (B) Softmax 확률

$$\hat{p}_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}} $$

#### (C) 가중 Cross-Entropy 손실 함수

$$\mathcal{L} = \alpha E(\hat{\mathbf{p}}, y_c) + (1 - \alpha) E(\hat{\mathbf{p}}, y_r) $$

여기서 $E(\cdot, \cdot)$은 cross-entropy 손실 함수, $y_c$와 $y_r$은 각 브랜치의 레이블.

#### (D) Reversed Sampler 확률 (클래스 불균형 역전)

클래스 $i$의 샘플 수를 $N_i$, 전체 최대 샘플 수를 $N_{max}$라 할 때:

$$P_i = \frac{w_i}{\sum_{j=1}^{C} w_j}, \quad \text{where} \quad w_i = \frac{N_{max}}{N_i} $$

→ 샘플 수가 적은(tail) 클래스일수록 더 높은 샘플링 확률 부여

#### (E) 누적 학습 α 스케줄러 (Parabolic Decay)

$$\alpha = 1 - \left(\frac{T}{T_{max}}\right)^2 $$

- $T$: 현재 epoch, $T_{max}$: 총 학습 epoch 수
- 학습 초기($T \to 0$): $\alpha \to 1$ → Conventional Branch 중심 (표현 학습 집중)
- 학습 후기($T \to T_{max}$): $\alpha \to 0$ → Re-balancing Branch 중심 (분류기 학습 집중)

**α 변화 직관:**

$$\alpha > 0.5 \Rightarrow \text{표현 학습 강조 단계}$$
$$\alpha \leq 0.5 \Rightarrow \text{분류기 학습 강조 단계}$$

---

### 2-3. 모델 구조 (Architecture)

```
┌─────────────────────────────────────────────────────────┐
│                    BBN Architecture                      │
│                                                         │
│  [Uniform Sampler]          [Reversed Sampler]          │
│       (x_c, y_c)                 (x_r, y_r)            │
│          ↓                            ↓                 │
│  ┌──────────────┐           ┌──────────────┐            │
│  │ Conv. Branch │ ←Share→  │ Re-bal Branch│            │
│  │ (Shared CNN  │ Weights  │ (Shared CNN  │            │
│  │  except last │           │  except last │            │
│  │  residual    │           │  residual    │            │
│  │  block)      │           │  block)      │            │
│  └──────┬───────┘           └──────┬───────┘            │
│         ↓ GAP                      ↓ GAP                │
│        f_c                         f_r                  │
│         ↓                           ↓                   │
│      α × W_c^T f_c  +  (1-α) × W_r^T f_r              │
│         └──────────────┬────────────┘                   │
│                     Softmax                             │
│                        ↓                               │
│              α·E(p̂, y_c) + (1-α)·E(p̂, y_r)           │
│                    [Loss]                               │
└─────────────────────────────────────────────────────────┘
```

**핵심 설계 요소:**

| 구성 요소 | 역할 |
|-----------|------|
| **Conventional Branch** | Uniform Sampler → 원래 분포 유지 → 보편적 표현 학습 |
| **Re-balancing Branch** | Reversed Sampler → tail 클래스 집중 → 분류기 재조정 |
| **Weight Sharing** | 마지막 residual block 제외 전체 공유 → 파라미터 효율성 |
| **Adaptor (α)** | Epoch 기반 자동 가중치 조정 → 점진적 학습 전환 |

**추론(Inference) 단계:**
- $\alpha = 0.5$ 로 고정 (두 브랜치 동등 가중치)
- 두 logit을 element-wise addition으로 최종 예측

---

### 2-4. 성능 향상

#### Long-tailed CIFAR (Top-1 Error Rate ↓, ResNet-32)

| Method | CIFAR-10 IR=100 | CIFAR-10 IR=50 | CIFAR-100 IR=100 | CIFAR-100 IR=50 |
|--------|:-:|:-:|:-:|:-:|
| CE | 29.64 | 25.19 | 61.68 | 56.15 |
| LDAM-DRW | 22.97 | 18.97 | 57.96 | 53.38 |
| **BBN (Ours)** | **20.18** | **17.82** | **57.44** | **52.98** |

- CIFAR-10 IR=100에서 LDAM-DRW 대비 **2.79%p 개선**

#### iNaturalist (Top-1 Error Rate ↓, ResNet-50)

| Method | iNat 2018 | iNat 2017 |
|--------|:-:|:-:|
| CE | 42.84 | 45.38 |
| LDAM-DRW (2×) | 33.88 | 38.19 |
| **BBN (2×)** | **30.38** | **34.25** |

- iNat 2018: **+3.50%p**, iNat 2017: **+3.94%p** 개선 (vs. LDAM-DRW 2×)

---

### 2-5. 한계점

1. **추론 시 계산 비용**: 두 브랜치를 모두 사용하므로 단일 모델 대비 연산량 증가 (Weight Sharing으로 완화)
2. **α 스케줄링 수동 설계**: 포물선 감쇠 함수가 최적이나, 다른 도메인에서의 일반성 검증 필요
3. **Object Detection 미적용**: 논문 스스로 "향후 long-tailed detection에 적용하겠다"고 언급
4. **Backbone 의존성**: ResNet 기반 실험만 수행; Transformer 계열 backbone과의 호환성 미검증
5. **극단적 불균형 시나리오**: iNaturalist 수준의 대규모 데이터셋에서는 더 긴 학습 스케줄(2×)이 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. BBN이 일반화를 높이는 핵심 메커니즘

#### (A) 표현 학습 품질 보존

논문의 Table 5에서 확인:

| 표현 학습 방식 | Error Rate |
|:-:|:-:|
| CE | 58.62 |
| RW | 63.17 |
| RS | 63.71 |
| **BBN-CB** | **58.89** |
| BBN-RB | 61.09 |

→ BBN의 Conventional Branch(BBN-CB)는 순수 CE와 거의 동일한 표현 품질 유지  
→ **원래 데이터 분포의 통계적 특성을 보존함으로써 보편적 특징(universal features) 학습**

#### (B) Cross-Dataset 일반화 실험

논문 Section 3에서 **중요한 일반화 실험** 수행:
- CIFAR-100-IR50에서 학습된 feature extractor를 **CIFAR-10-IR50**에 전이
- CE로 학습된 표현이 RW/RS보다 다른 도메인에서도 더 낮은 오류율 달성

$$\text{CIFAR-100으로 학습된 특징} \xrightarrow{\text{Transfer}} \text{CIFAR-10 분류}$$

→ BBN의 Conventional Branch는 CE와 유사한 cross-domain transferability를 가짐

#### (C) Intra-class 응집력 (Compactness) 향상

보충 자료 Figure 6에서:
- CE로 학습된 표현: 각 클래스 centroid까지의 평균 거리 **작음** (응집력 높음)
- RW/RS 표현: centroid까지의 거리 **큼** (특히 head 클래스에서 두드러짐)
- BBN: CE 수준의 응집력 유지 + tail 클래스 분류 성능 향상

$$\text{Compactness} = \frac{1}{|S_i|} \sum_{\mathbf{x} \in S_i} \|\mathbf{f}(\mathbf{x}) - \boldsymbol{\mu}_i\|_2$$

(여기서 $\boldsymbol{\mu}_i$는 클래스 $i$의 centroid, $S_i$는 클래스 $i$의 샘플 집합)

#### (D) 분류기 가중치 균형성

Figure 4 (논문)에서 $\ell_2$-norm 분석:
- CE: $\ell_2$-norm이 long-tailed 분포와 동일한 패턴 (head 클래스 편향)
- RW/RS: 비교적 평탄하나 분산(σ)이 큼
- **BBN-ALL**: σ = 0.148 (가장 작음) → **균형 잡힌 분류기 가중치**

$$\sigma_{\text{BBN-ALL}} = 0.148 < \sigma_{\text{CE}} = 0.422 < \sigma_{\text{RS}} = 0.329$$

### 3-2. 일반화 관점에서의 추가적 가능성

1. **Fine-grained recognition**: iNaturalist처럼 세밀한 분류 작업에서도 우수한 성능 입증
2. **전이 학습(Transfer Learning) 잠재력**: CE 수준의 표현 품질 → pre-trained model로서의 활용 가능성
3. **다양한 불균형 비율**: IR=10, 50, 100 모두에서 일관된 성능 → 불균형 정도에 무관한 일반화

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4-1. 연구에 미치는 영향

#### (A) 패러다임 전환: "단일 최적화" → "이중 학습 목표"

BBN은 기존의 단일 최적화 목표(분류 정확도)에서 **표현 학습과 분류기 학습을 명시적으로 분리**하는 패러다임을 제시했습니다. 이는 이후 연구들의 공통 기반이 되었습니다.

#### (B) 두 단계 학습의 이론적 정당화

BBN은 기존 2-stage fine-tuning을 **end-to-end 연속 학습**으로 통합함으로써, 이후 decoupled training 연구들에 이론적 토대를 제공했습니다.

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

#### 논문 1: **Decoupling Representation and Classifier for Long-Tailed Recognition** (Kang et al., ICLR 2020)

| 항목 | BBN | Decoupling (Kang et al.) |
|------|-----|--------------------------|
| **핵심 아이디어** | 두 브랜치 동시 훈련 | 명시적 2단계 분리 훈련 |
| **표현 학습** | Uniform Sampler (동시) | 1단계에서 CE로만 학습 |
| **분류기 조정** | Reversed Sampler (동시) | 2단계에서 별도 fine-tuning |
| **훈련 방식** | End-to-end | Sequential (2-stage) |
| **분류기 재조정** | Weight Sharing + α | τ-normalized classifier, cRT 등 |

- BBN은 두 학습 목표를 동시에 달성하는 **end-to-end** 방식의 우위를 가지나, Kang et al.은 더 다양한 분류기 재조정(cRT, LWS, τ-norm 등)을 체계적으로 분석

**참고문헌**: Kang, B., Xie, S., Rohrbach, M., et al. "Decoupling representation and classifier for long-tailed recognition." ICLR 2020.

---

#### 논문 2: **Long-Tail Learning via Logit Adjustment** (Menon et al., ICLR 2021)

| 항목 | BBN | Logit Adjustment |
|------|-----|------------------|
| **접근법** | 구조적 (두 브랜치) | 손실 함수 수정 |
| **핵심 아이디어** | 표현/분류기 학습 분리 | 클래스 빈도 기반 logit 보정 |
| **수식** | $\mathbf{z} = \alpha\mathbf{W}_c^\top\mathbf{f}_c + (1-\alpha)\mathbf{W}_r^\top\mathbf{f}_r$ | $\tilde{z}_i = z_i - \tau \log \pi_i$ |
| **계산 비용** | 2배 (두 브랜치) | 거의 없음 (logit 조정만) |

- Logit Adjustment는 이론적으로 Bayes-optimal 분류 경계에 근접하는 방법을 제시하며, BBN보다 계산 효율적이나 표현 학습 품질 보존 메커니즘은 부재

**참고문헌**: Menon, A. K., et al. "Long-tail learning via logit adjustment." ICLR 2021.

---

#### 논문 3: **Balanced Softmax Loss** (Ren et al., NeurIPS 2020)

$$\log \frac{e^{z_y + \log n_y}}{\sum_{j} e^{z_j + \log n_j}}$$

- 클래스 빈도를 logit에 직접 통합하여 불균형 문제 해결
- BBN의 역방향 샘플링과 달리 손실 함수 수준에서 해결 → 단순하지만 BBN의 구조적 이점(표현 보존) 부재

**참고문헌**: Ren, J., et al. "Balanced meta-softmax for long-tailed visual recognition." NeurIPS 2020.

---

#### 논문 4: **MiSLAS** (Zhong et al., CVPR 2021)

- BBN의 누적 학습 아이디어를 계승하여 **Mixup + 두 단계 학습 스케줄**을 결합
- 표현 학습 단계에서 Mixup을 적극 활용, 분류기 재조정 단계에서 label-smoothing 적용

**참고문헌**: Zhong, Z., et al. "Improving calibration for long-tailed recognition." CVPR 2021.

---

#### 논문 5: **RIDE** (Wang et al., ICLR 2021)

| 항목 | BBN | RIDE |
|------|-----|------|
| **브랜치 수** | 2 (고정) | 다중 (동적) |
| **다양성 확보** | Uniform/Reversed Sampler | Diversity Loss |
| **특징** | α로 학습 전환 | 지식 증류 기반 앙상블 |

- BBN의 이중 브랜치 아이디어를 확장하여 **다중 분류기 앙상블 + 다양성 손실**로 발전

**참고문헌**: Wang, X., et al. "Long-tailed recognition by routing diverse distribution-aware experts." ICLR 2021.

---

#### 논문 6: **PaCo** (Cui et al., ICCV 2021)

- **Supervised Contrastive Learning** + 파라미터화된 클래스 대표 벡터를 통해 tail 클래스의 표현 품질을 향상
- BBN이 구조적으로 해결한 표현-분류기 트레이드오프를 **대조 학습(contrastive learning)**으로 접근

**참고문헌**: Cui, J., et al. "Parametric contrastive learning." ICCV 2021.

---

### 4-3. 향후 연구 시 고려할 점

#### ① α 스케줄링의 자동화 및 적응형 설계

현재 BBN의 포물선 감쇠 함수는 수동으로 설계된 것입니다. 향후 연구에서는:

$$\alpha_T = f_\theta(T, \text{validation performance}, \text{class distribution})$$

처럼 **학습 가능한(learnable) α 스케줄러** 설계가 필요합니다.

#### ② Transformer 기반 Backbone과의 통합

BBN은 ResNet 기반으로 설계되었으나, Vision Transformer(ViT) 계열 모델에서의 적용 시:
- Self-attention 메커니즘이 이미 tail 클래스에 attention을 더 부여하는지 검증 필요
- Patch-level feature와 BBN 브랜치 구조의 호환성 검토

#### ③ Long-Tailed Object Detection/Segmentation 확장

논문 스스로 한계로 인정한 부분으로, **LVIS 데이터셋**을 활용한:
- Instance segmentation에서 BBN 구조의 적용
- Region Proposal Network(RPN)과 분류 헤드에 각각 다른 샘플링 전략 적용 가능성

#### ④ Few-Shot Learning과의 결합

BBN의 Reversed Sampler는 tail 클래스를 자연스럽게 few-shot 설정처럼 다룹니다:

$$\text{Long-tail learning} \xrightarrow{\text{BBN의 관점}} \text{Many-shot + Few-shot 동시 학습}$$

이를 메타학습(Meta-learning)과 결합하면 더 강력한 일반화가 가능합니다.

#### ⑤ 데이터 불균형 + 노이즈 레이블 동시 처리

실제 데이터셋은 불균형과 노이즈 레이블이 함께 존재합니다:
- Conventional Branch: 노이즈 레이블에 robust한 표현 학습
- Re-balancing Branch: 노이즈가 적은 tail 클래스 집중 학습
- 두 목표의 통합 최적화 방향 연구 필요

#### ⑥ 동적 불균형 (Dynamic Class Imbalance) 대응

스트리밍 데이터나 continual learning 환경에서 클래스 분포가 시간에 따라 변화하는 경우:

$$P(y=i|t) \neq P(y=i|t+\Delta t)$$

BBN의 Reversed Sampler를 동적으로 업데이트하는 온라인 학습 프레임워크 설계가 필요합니다.

---

## 참고 자료 및 출처

**1차 자료 (Primary Source):**
- **Zhou, B., Cui, Q., Wei, X.-S., & Chen, Z.-M. (2020). "BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition." CVPR 2020. arXiv:1912.02413v4**

**2차 자료 (Related Works cited in analysis):**
- Kang, B., Xie, S., Rohrbach, M., et al. "Decoupling Representation and Classifier for Long-Tailed Recognition." *ICLR 2020.* arXiv:1910.09217
- Menon, A. K., Jayasumana, S., Rawat, A. S., et al. "Long-tail learning via logit adjustment." *ICLR 2021.* arXiv:2007.07314
- Ren, J., Yu, C., Ma, X., et al. "Balanced Meta-Softmax for Long-Tailed Visual Recognition." *NeurIPS 2020.* arXiv:2007.10740
- Zhong, Z., Cui, J., Liu, S., & Jia, J. "Improving Calibration for Long-Tailed Recognition." *CVPR 2021.* arXiv:2104.00466
- Wang, X., Lian, L., Miao, Z., et al. "Long-Tailed Recognition by Routing Diverse Distribution-Aware Experts." *ICLR 2021.* arXiv:2010.01809
- Cui, J., Zhong, Z., Liu, S., et al. "Parametric Contrastive Learning." *ICCV 2021.* arXiv:2107.05694
- Cao, K., Wei, C., Gaidon, A., et al. "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss." *NeurIPS 2019.* (LDAM, [3] in original paper)
- Cui, Y., Jia, M., Lin, T.-Y., et al. "Class-Balanced Loss Based on Effective Number of Samples." *CVPR 2019.* (CB-Focal, [5] in original paper)
