# Understanding and Improving Early Stopping for Learning with Noisy Labels

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음과 같습니다:

> **DNN의 각 레이어는 노이즈 레이블에 대해 서로 다른 민감도를 가지며, 출력층에 가까운 후반부(latter) 레이어일수록 노이즈의 영향을 더 빠르고 심각하게 받는다.**

기존의 Early Stopping은 네트워크 전체를 하나의 단위로 보고 단일 중단점을 결정하지만, 이는 각 레이어의 노이즈 민감도 차이를 무시하여 레이어 간 **상충(antagonistic) 효과**를 유발한다는 것을 지적합니다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| **분석적 기여** | DNN 레이어별 노이즈 민감도 차이를 실험적으로 규명 |
| **방법론적 기여** | Progressive Early Stopping (PES) 제안 |
| **실용적 기여** | 기존 SOTA 방법(DivideMix, ELR+)과 결합하여 새로운 SOTA 달성 |
| **효율성** | 계산 비용이 기존 방법 대비 크게 증가하지 않음 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**배경: Memorization Effect**

딥러닝 모델은 훈련 초기에는 깨끗한(clean) 패턴을 먼저 학습하고, 이후 노이즈 레이블을 암기(memorize)하는 경향이 있습니다 (Arpit et al., 2017). 이를 **Memorization Effect**라 합니다.

기존 Early Stopping의 목적함수:

$$\min_{\Theta} \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(f(\mathbf{x}_i; \Theta), \tilde{y}_i) \quad \text{(Eq. 1)}$$

여기서:
- $f(\cdot; \Theta)$: 모델 파라미터 $\Theta$를 가진 딥 분류기
- $\mathcal{L}$: Cross-entropy 손실 함수
- $\tilde{y}_i$: 노이즈가 포함된 레이블

**문제점:**

기존 방법들은 단일 중단 에폭 $T$를 전체 네트워크에 적용하나, 아래 그림처럼 레이어마다 최적 중단점이 다릅니다:
- 9번째 레이어(전반부): 더 많은 에폭에서도 성능 유지
- 17번째 레이어(중반부): 중간 정도의 민감도
- 최종 레이어(후반부): 소수의 에폭에서도 급격한 성능 저하

이로 인해 단일 중단점 선택은 **레이어 간 상충 효과(antagonistic effect)**를 유발합니다.

---

### 2.2 제안 방법: Progressive Early Stopping (PES)

#### 네트워크 분할 정의

전체 네트워크 $f(\cdot; \Theta)$를 $L$개의 DNN 파트로 분리:

$$\mathbf{z}_1 = f_1(\mathbf{x}; \Theta_1)$$
$$\mathbf{z}_l = f_l(\mathbf{z}_{l-1}; \Theta_l), \quad l = 2, \ldots, L \quad \text{(Eq. 2)}$$

#### 학습 절차 (수식 포함)

**Step 1**: 전체 네트워크를 $T_1$ 에폭 동안 훈련하여 전반부 파라미터 $\Theta_1^*$ 획득:

$$\min_{\Theta_1 \ldots \Theta_L} \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(f(\mathbf{x}_i; \Theta_1, \ldots, \Theta_L), \tilde{y}_i) \quad \text{(Eq. 3)}$$

**Step 2**: 이전 파트의 파라미터를 고정하고, $l$번째 파트를 $T_l$ 에폭 동안 재초기화 후 훈련:

$$\min_{\Theta_l \ldots \Theta_L} \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(f(\mathbf{x}_i; \Theta_1^*, \ldots, \Theta_{l-1}^*, \Theta_l, \ldots, \Theta_L), \tilde{y}_i), \quad l = 2 \ldots L \quad \text{(Eq. 4)}$$

**핵심 제약 조건:**

$$T_1 \geq T_2 \geq \cdots \geq T_L$$

후반부 레이어일수록 더 적은 에폭을 사용하여 노이즈 암기를 방지합니다.

---

### 2.3 신뢰 예시(Confident Examples) 추출

PES로 훈련된 모델에서 신뢰할 수 있는 예시를 추출:

$$\mathcal{D}_l = \{(\mathbf{x}_i, \tilde{y}_i) \mid \tilde{y}_i = \hat{y}_i, \, i = 1, \ldots, n\}$$
$$\hat{y}_i = \underset{k \in \{1,\ldots,K\}}{\arg\max} \frac{1}{2}\left[f^k(\text{Augment}(\mathbf{x}_i); \Theta) + f^k(\text{Augment}(\mathbf{x}_i); \Theta)\right] \quad \text{(Eq. 5)}$$

클래스 불균형 문제를 완화하기 위한 가중치 손실함수:

$$\mathcal{L}_c = \sum_{i=1}^{N} w_{y_i} \mathcal{L}_p(\tilde{y}_i, f(\mathbf{x}_i; \Theta)) \quad \text{(Eq. 6)}$$

여기서 $w_i = \sigma_i / \left(\sum_{j=1}^{K} \sigma_j\right)$이며, $\sigma_k$는 $k$번째 클래스에 속하는 신뢰 예시의 수입니다.

---

### 2.4 반지도 학습과의 결합

신뢰 예시를 레이블 데이터로, 나머지를 언레이블 데이터로 분류:

$$\begin{cases} \mathcal{D}_l = \{(\mathbf{x}_i, \tilde{y}_i) \mid \tilde{y}_i = \hat{y}_i, \, i = 1, \ldots, n\} \\ \mathcal{D}_u = \{\mathbf{x}_i \mid \tilde{y}_i \neq \hat{y}_i, \, i = 1, \ldots, n\} \end{cases} \quad \text{(Eq. 7)}$$

MixMatch 프레임워크를 활용하여 최종 분류기를 훈련합니다.

---

### 2.5 모델 구조

#### 실험에 사용된 아키텍처

| 실험 설정 | 아키텍처 | 파트 분할 | $T_1$ | $T_2$ | $T_3$ |
|-----------|----------|-----------|--------|--------|--------|
| CIFAR-10 (CE) | ResNet-18 | 3파트 | 25 | 7 | 5 |
| CIFAR-100 (CE) | ResNet-34 | 3파트 | 30 | 7 | 5 |
| CIFAR-10 (Semi) | PreAct ResNet-18 | 2파트 | 20 | 5 | - |
| CIFAR-100 (Semi) | PreAct ResNet-18 | 2파트 | 35 | 5 | - |
| Clothing-1M | Pretrained ResNet-50 | 2파트 | 20 | 7 | - |

ResNet-18 기준 파트 분할:
- **Part 1**: Block 4 이전의 모든 레이어
- **Part 2**: Block 4
- **Part 3**: 최종 분류 레이어

---

### 2.6 성능 향상

#### CE Loss만 사용 (반지도 학습 없음) — CIFAR-10

| 방법 | Sym-20% | Sym-50% | Pair-45% | Inst-20% | Inst-40% |
|------|---------|---------|---------|---------|---------|
| CE | 84.00 | 75.51 | 63.34 | 85.10 | 77.00 |
| Co-teaching | 87.16 | 72.80 | 70.11 | 86.54 | 80.98 |
| CDR | 89.72 | 82.64 | 73.67 | 90.41 | 83.07 |
| **PES (Ours)** | **92.38** | **87.45** | **88.43** | **92.69** | **89.73** |

#### 반지도 학습 포함 — CIFAR-10

| 방법 | Sym-20% | Sym-50% | Sym-80% |
|------|---------|---------|---------|
| DivideMix | 95.6 | 94.6 | 92.9 |
| ELR+ | 94.9 | 93.6 | 90.4 |
| **PES+Semi** | **95.9** | **95.1** | **93.1** |

특히 **Pairflip 45% 노이즈**에서 기존 SOTA 대비 **8% 이상** 향상됩니다 (CIFAR-10, CIFAR-100 모두).

#### 훈련 시간 비교 (CIFAR-10, Sym-50%)

| 방법 | 훈련 시간 |
|------|---------|
| CE | 0.9h |
| **PES (Ours)** | **1.0h** |
| ELR+ | 2.2h |
| DivideMix | 5.5h |
| PES+Semi | 3.1h |

---

### 2.7 한계점

1. **추가 하이퍼파라미터**: DNN을 파트로 분할함으로써 $T_1, T_2, \ldots, T_L$ 등 추가 하이퍼파라미터 튜닝이 필요합니다.
2. **분할 기준 불명확**: 어떤 기준으로 레이어를 분할할지에 대한 이론적 근거가 부족합니다.
3. **Transformer 계열 미검증**: ResNet 계열에 집중되어 있어 Vision Transformer(ViT) 등에서의 효과는 검증되지 않았습니다.
4. **실세계 노이즈 한계**: Clothing-1M에서 단일 네트워크 기준 경쟁력 있는 성능을 보이지만, 앙상블 기반 방법과 비교 시 여전히 개선 여지가 있습니다.

---

## 3. 일반화 성능 향상 가능성

### 3.1 왜 PES가 일반화를 향상시키는가?

**Memorization Effect의 계층적 관점:**

기존 연구(Arpit et al., 2017)에서 DNN은 먼저 클린 패턴을 학습하고 나중에 노이즈를 암기한다고 알려져 있습니다. PES는 이를 **레이어 단위로 세분화**합니다:

$$\text{노이즈 영향도}: \Theta_1 < \Theta_2 < \cdots < \Theta_L$$

후반부 레이어는 손실 함수로부터 직접적인 그래디언트를 받기 때문에 노이즈에 더 민감합니다. PES는 이를 작은 $T_l$ 값으로 제어하여 **각 레이어가 적절한 시점에 중단**되도록 합니다.

### 3.2 신뢰 예시 품질 향상

Table 1 (CIFAR-10)에서:

| 지표 | Early Stopping | PES | 향상 |
|------|---------------|-----|------|
| Test Accuracy (Sym-50%) | 70.76 | 75.87 | **+5.11%** |
| Label Recall (Sym-50%) | 75.18 | 81.03 | **+5.85%** |
| Label Precision (Sym-50%) | 94.65 | 95.46 | **+0.81%** |

높은 **Label Recall**은 더 많은 클린 예시를 반지도 학습의 레이블 데이터로 활용할 수 있음을 의미하며, 이는 최종 모델의 일반화 성능을 직접적으로 향상시킵니다.

### 3.3 분산 감소(Variance Reduction)

Figure 2에서 관찰되듯이, PES는 기존 Early Stopping 대비 **훨씬 안정적인 학습 곡선**을 보입니다. 이는 다음을 의미합니다:

- 하이퍼파라미터 선택에 덜 민감
- 다양한 노이즈 유형(Symmetric, Pairflip, Instance)에서 일관된 성능
- 최종 모델의 신뢰성 향상

### 3.4 표현 학습(Representation Learning) 관점

전반부 레이어는 충분히 많은 에폭 동안 훈련되어 **좋은 특징 표현(feature representation)**을 학습하고, 후반부 레이어는 적은 에폭으로 노이즈 암기를 방지합니다. 이는 Li et al. (2020, arXiv:2012.12896)의 "Noisy labels can induce good representations" 연구와도 일치하며, 전반부 레이어의 표현이 노이즈에 상대적으로 강건함을 지지합니다.

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 4.1 향후 연구에 미치는 영향

#### 이론적 영향

1. **레이어별 노이즈 민감도 이론화**: PES는 레이어별 노이즈 민감도의 차이를 실험적으로 보였지만, 이를 수학적으로 정형화하는 연구가 필요합니다. 예를 들어, 각 레이어의 그래디언트 노름과 노이즈 민감도의 관계를 분석하는 연구로 이어질 수 있습니다.

2. **Memorization Effect의 세분화된 이해**: 기존의 전체 네트워크 관점에서 레이어 단위의 관점으로 Memorization Effect를 재해석하는 이론적 프레임워크 구축에 기여합니다.

#### 방법론적 영향

1. **동적 레이어 분할**: 고정된 분할 대신 훈련 중 동적으로 레이어 중단점을 결정하는 방법론 연구
2. **자동화된 하이퍼파라미터 탐색**: $T_1, T_2, \ldots, T_L$을 자동으로 결정하는 Neural Architecture Search(NAS) 또는 AutoML 접근법
3. **다른 아키텍처 적용**: Vision Transformer, Graph Neural Network 등에서 레이어별 노이즈 민감도 분석

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래 비교는 논문에서 직접 인용된 연구들과 일반적으로 알려진 연구 트렌드를 기반으로 서술하며, 논문 외부의 구체적 수치는 확인된 정보만 기술합니다.

#### 논문에서 직접 비교된 최신 방법들 (2020년 이후)

| 방법 | 연도 | 핵심 아이디어 | PES와의 비교 |
|------|------|-------------|-------------|
| **DivideMix** (Li et al., ICLR 2020) | 2020 | Beta Mixture Model로 클린/노이즈 분리 후 반지도 학습 | PES+Semi가 CIFAR-10/100 전반에서 우세 |
| **ELR+** (Liu et al., NeurIPS 2020) | 2020 | 초기 학습 정규화(ELR)로 노이즈 암기 방지 | PES+Semi가 대부분 설정에서 우세 |
| **CDR** (Xia et al., ICLR 2021) | 2021 | Robust early-learning으로 노이즈 암기 억제 | PES가 모든 설정에서 크게 우세 (CE 기준) |
| **Class2Simi** (Wu et al., ICML 2021) | 2021 | 노이즈 감소 관점에서 레이블 학습 | 직접 비교 없음 |

#### PES의 차별성

```
기존 방법: 전체 네트워크 → 단일 중단점 T
PES:       파트별 분리 → 복수 중단점 T₁ ≥ T₂ ≥ ... ≥ T_L
```

PES는 기존 방법들과 **직교적(orthogonal)**인 접근으로, 기존 방법의 기반 모델을 개선하는 **플러그인(plug-in)** 방식으로 활용 가능합니다.

### 4.3 앞으로 연구 시 고려할 점

#### 단기적 고려사항

1. **하이퍼파라미터 자동화**
   - $T_1, T_2, \ldots, T_L$ 값을 검증 세트 없이 자동으로 결정하는 방법 필요
   - 노이즈율 추정과 연계한 적응적(adaptive) 에폭 설정 연구

2. **분할 전략 최적화**
   - 단순히 블록 단위가 아닌, 그래디언트 민감도 분석 기반의 최적 분할점 탐색
   - Attention 기반 중요도 측정을 활용한 레이어 그룹화

3. **다양한 노이즈 유형 대응**
   - Open-set noise (학습 데이터에 없는 클래스의 노이즈) 처리
   - Feature-dependent noise에 대한 PES의 효과 분석

#### 장기적 고려사항

4. **대규모 모델 적용**
   - GPT, ViT 등 대형 모델에서의 레이어별 노이즈 민감도 분석
   - Fine-tuning 시나리오(전이 학습)에서의 PES 적용 가능성

5. **이론적 보장**
   - PES의 수렴 보장(convergence guarantee) 이론 정립
   - 노이즈율과 최적 $T_l$ 값 사이의 이론적 관계 규명

6. **다른 도메인 확장**
   - NLP(자연어 처리)에서의 노이즈 레이블 학습에 PES 적용
   - 의료 영상 등 고비용 레이블링 도메인에서의 활용

7. **Self-supervised / Contrastive Learning과의 결합**
   - 레이어별 대조 학습(contrastive learning)을 통한 표현 학습 강화
   - 사전 훈련된 모델의 각 레이어가 이미 좋은 표현을 가지고 있을 때 PES의 전략 재설계

---

## 참고 자료

**주요 논문 (PDF 원문)**
- Bai, Y., Yang, E., Han, B., Yang, Y., Li, J., Mao, Y., Niu, G., & Liu, T. (2021). **Understanding and Improving Early Stopping for Learning with Noisy Labels**. *NeurIPS 2021*. arXiv:2106.15853v2

**논문 내 인용 참고문헌**
- Arpit, D. et al. (2017). A closer look at memorization in deep networks. *ICML*. [노이즈 레이블 Memorization Effect의 근거]
- Li, J., Socher, R., & Hoi, S. C. H. (2020). DivideMix: Learning with noisy labels as semi-supervised learning. *ICLR 2020*.
- Liu, S. et al. (2020). Early-learning regularization prevents memorization of noisy labels. *NeurIPS 2020*.
- Xia, X. et al. (2021). Robust early-learning: Hindering the memorization of noisy labels. *ICLR 2021*.
- Han, B. et al. (2018). Co-teaching: Robust training of deep neural networks with extremely noisy labels. *NeurIPS 2018*.
- Berthelot, D. et al. (2019). MixMatch: A holistic approach to semi-supervised learning. *NeurIPS 2019*.
- He, K. et al. (2016). Deep residual learning for image recognition. *CVPR 2016*.
- Zhang, C. et al. (2017). Understanding deep learning requires rethinking generalization. *ICLR 2017*.
- Li, J. et al. (2020). Noisy labels can induce good representations. arXiv:2012.12896.

**코드 저장소**
- https://github.com/tmllab/PES
