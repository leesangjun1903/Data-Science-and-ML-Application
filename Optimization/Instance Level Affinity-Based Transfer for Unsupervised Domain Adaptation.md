# Instance Level Affinity-Based Transfer for Unsupervised Domain Adaptation

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

"Instance Level Affinity-Based Transfer for Unsupervised Domain Adaptation" (Sharma et al., 2021, arXiv:2104.01286)의 핵심 주장은 다음과 같습니다:

> 기존 비지도 도메인 적응(UDA) 방법들은 **전역적(global) 분포 정렬**에만 집중하여 클래스 경계 근처에서의 잘못된 정렬(misalignment)과 부정적 전이(negative transfer)를 유발한다. 이를 해결하기 위해 **인스턴스 수준의 유사도(instance-level affinity)**를 활용한 새로운 적응 프레임워크 ILA-DA를 제안한다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **ILA-DA 프레임워크** | 인스턴스 수준 유사도 기반 도메인 적응 프레임워크 제안 |
| **MSC Loss** | Multi-Sample Contrastive Loss: 다중 양성/음성 쌍을 동시에 처리 |
| **Affinity Matrix** | kNN 기반 유사도 행렬 구성 + 유사도 비율 테스트(Similarity Ratio Test)로 노이즈 필터링 |
| **범용성** | DANN, CDAN 등 기존 adversarial 방법에 plug-in 형태로 적용 가능 |
| **세밀한 분류(Fine-grained) 적응** | Birds-31 데이터셋에서 레이블 계층 없이 SOTA 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### (1) 전역 정렬의 한계

기존 적대적 학습 기반 방법들(DANN, ADDA 등)은 소스와 타겟 도메인의 **전역 분포 정렬**만 수행합니다. 이는:

- 클래스별 세부 구조를 무시
- 클래스 경계 근처에서 노이즈 발생
- 서로 다른 클래스 간 부정적 전이(negative transfer) 유발

#### (2) 유사도 레이블 노이즈 문제

타겟 도메인이 완전히 레이블이 없는 상황(unlabeled)에서:

- 소스 분류기를 그대로 사용한 pseudo-label은 학습 초기 단계에서 매우 노이즈가 큼
- 기존의 클래스 수준 정렬도 pseudo-label의 품질에 종속적

#### (3) 세밀한 분류 도메인에서의 어려움

같은 클래스 내 높은 분산(large intra-class variation)과 클래스 간 낮은 분산(small inter-class variation)을 동시에 처리해야 하는 fine-grained 설정에서의 도전

---

### 2.2 제안 방법 (수식 포함)

#### 전체 구조

$$f = \mathcal{G}(x)$$

공유 피처 추출기 $\mathcal{G}$를 통해 소스와 타겟의 피처를 추출합니다.

#### (1) Supervised Classification Loss

$$\mathcal{L}_{sup} = \mathbb{E}_{(x,y) \sim \mathcal{D}^s} [-\log[\mathcal{C}(\mathcal{G}(x))]_y] \tag{1}$$

소스 도메인의 레이블 데이터에 대한 Cross-Entropy Loss입니다.

#### (2) Adversarial Domain Alignment Loss

$$\mathcal{L}_{adv} = \mathbb{E}_{x \sim \mathcal{D}^t} [-\log \mathcal{D}(\mathcal{G}(x))] \tag{2}$$

$$\mathcal{L}_D = -\mathbb{E}_{x \sim \mathcal{D}^s} [\log \mathcal{D}(\mathcal{G}(x))] - \mathbb{E}_{x \sim \mathcal{D}^t} [\log(1 - \mathcal{D}(\mathcal{G}(x)))] \tag{3}$$

$\mathcal{L}\_D$와 $\mathcal{L}_{adv}$의 Min-Max 학습으로 도메인 불변 피처를 학습합니다.

#### (3) Affinity Matrix 구성

타겟 샘플 $x_j \in B_T$에 대해 kNN으로 pseudo-label $\hat{y}_j$를 할당한 뒤:

$$A_{ij} = \begin{cases} 1, & \text{if } y_i = \hat{y}_j \\ -1, & \text{if } y_i \neq \hat{y}_j \end{cases}$$

#### (4) 유사도 비율 테스트(Similarity Ratio Test)로 노이즈 필터링

타겟 샘플 $x_j$의 pseudo-label 신뢰도 점수:

$$\Gamma_j = \frac{\sum_{x_i \in N^l_j} \phi(f_j, f_i)}{\sum_{x_i \in N^u_j} \phi(f_j, f_i)} \tag{7}$$

- $N^l_j$: $x_j$의 이웃 중 같은 클래스(like) 소스 샘플 집합
- $N^u_j$: $x_j$의 이웃 중 다른 클래스(unlike) 소스 샘플 집합

상위 $\mu$ 비율의 샘플만 신뢰 가능한 pseudo-label로 사용하고, 나머지는 $A_{ij} = 0$으로 처리합니다.

#### (5) Multi-Sample Contrastive (MSC) Loss

소스 샘플 $x_i$에 대한 손실:

$$\mathcal{L}^i_{MSC} = -\log \frac{\sum_{j \in B^{i+}_T} e^{\phi(f_i, f_j)}}{\sum_{j \in B^{i+}_T} e^{\phi(f_i, f_j)} + \sum_{j \in B^{i-}_T} e^{\phi(f_i, f_j)}} \tag{4}$$

- $B^{i+}\_T = \{x_j \in B_T \mid A_{ij} = 1\}$: 양성(positive) 타겟 샘플 집합
- $B^{i-}\_T = \{x_j \in B_T \mid A_{ij} = -1\}$: 음성(negative) 타겟 샘플 집합

미니배치 전체에 대한 평균:

$$\mathcal{L}_{MSC} = \frac{1}{|B_S|} \sum_{i \in B_S} \mathcal{L}^i_{MSC} \tag{5}$$

#### (6) 유사도 메트릭 $\phi$

$$\phi(f_i, f_j) = \frac{1}{1 + \|f_i - f_j\|^2} \tag{6}$$

정규화된 역 유클리드 거리(Normalized Inverse Euclidean Distance)를 사용합니다.

---

### 2.3 모델 구조

```
[소스 입력] ─┐
             ├─► [공유 피처 추출기 G(.)] ─┬─► [분류기 C(.)] ──► L_sup
[타겟 입력] ─┘                           ├─► [판별기 D(.)] ──► L_adv / L_D
                                         └─► [Affinity Matrix A 구성]
                                                   │
                                                   ▼
                                         [MSC Loss 계산] ──► L_MSC
```

**전체 목적 함수:**

$$\mathcal{L}_{total} = \mathcal{L}_{sup} + \lambda_1 \mathcal{L}_{adv} + \lambda_2 \mathcal{L}_{MSC}$$

> **주의**: 논문에 전체 목적 함수의 통합 가중치($\lambda_1, \lambda_2$)가 명시적 수식으로 표현되지 않으며, DANN/CDAN의 기존 가중치 설정을 따릅니다.

**백본 아키텍처:**
- Digits: LeNet
- Office-31, Birds-31: ResNet-50 (ImageNet 사전학습)

---

### 2.4 성능 향상

#### Digits 데이터셋

| 방법 | M→U | U→M | S→M | 평균 |
|---|---|---|---|---|
| DANN | 90.8 | 93.95 | 83.11 | 89.29 |
| **ILA-DA (DANN)** | **92.43** | **97.32** | **91.84** | **93.83** |
| CDAN | 93.9 | 96.9 | 88.5 | 93.1 |
| **ILA-DA (CDAN)** | **94.87** | **97.47** | **92.30** | **94.88** |

#### Office-31 데이터셋

| 방법 | A→W | D→W | W→D | A→D | D→A | W→A | 평균 |
|---|---|---|---|---|---|---|---|
| CDAN+E | 94.1 | 98.6 | **100.0** | 92.9 | 71.0 | 69.3 | 87.7 |
| **ILA-DA (CDAN)** | **95.72** | **99.25** | **100.0** | **93.37** | **72.10** | **75.40** | **89.30** |

#### Birds-31 데이터셋 (Fine-grained)

| 방법 | C→I | I→C | I→N | N→I | C→N | N→C | 평균 |
|---|---|---|---|---|---|---|---|
| PAN (fine-grained 특화) | 69.79 | 90.46 | 88.10 | 75.03 | 84.19 | 92.51 | 83.34 |
| **ILA-DA (CDAN)** | **72.77** | **93.83** | **90.36** | **78.09** | **86.58** | **94.53** | **86.03** |

Birds-31에서 레이블 계층(label hierarchy) 없이 fine-grained 특화 방법 PAN을 약 **3% 초과 달성**합니다.

---

### 2.5 한계점

논문에서 명시적으로 언급한 한계:

1. **메모리 집약적(Memory-intensive)**: 매 이터레이션마다 Affinity Matrix를 $O(n^2)$ 복잡도로 구성해야 함
2. **클래스 수 제한**: 현재 메모리 한계로 처리 가능한 카테고리 수에 제한이 있음
3. **클래스 균형 샘플링 필요**: 소스 미니배치에서 클래스 균형 샘플링이 필수적이며, 클래스 수가 매우 많을 경우 비효율적
4. **하이퍼파라미터 민감도**: $k$ (kNN 이웃 수), $\mu$ (샘플링 비율) 등의 하이퍼파라미터 조정이 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능을 높이는 메커니즘

#### (A) 인스턴스 수준 관계 활용

기존 방법들이 전역 또는 클래스 수준 정렬에 그쳤다면, ILA-DA는 **샘플 쌍(pairwise)** 수준에서 유사도를 명시적으로 모델링합니다. 이는 다음을 가능하게 합니다:

$$\text{일반화 오차} \leq \mathcal{O}\left(\sqrt{\frac{d_\mathcal{H}(P_s, P_t)}{n}}\right)$$

Ben-David et al. (2010)의 이론에 따르면, 소스-타겟 간 분포 차이 $d_\mathcal{H}$를 줄일수록 타겟 도메인에서의 일반화 오차도 줄어들며, ILA-DA의 인스턴스 수준 정렬은 이 $d_\mathcal{H}$를 더 효과적으로 감소시킵니다.

#### (B) 클래스 내 집합(Intra-class Clustering) + 클래스 간 분리(Inter-class Separation) 동시 달성

MSC Loss의 분자는 양성 쌍(같은 클래스 소스-타겟)의 유사도를 최대화하고, 분모의 음성 쌍은 다른 클래스 간 거리를 최대화합니다:

$$\mathcal{L}^i_{MSC} = -\log \frac{\sum_{j \in B^{i+}_T} e^{\phi(f_i, f_j)}}{\sum_{j \in B^{i+}_T} e^{\phi(f_i, f_j)} + \sum_{j \in B^{i-}_T} e^{\phi(f_i, f_j)}}$$

이는 특히 **fine-grained 분류**에서 일반화 성능을 크게 향상시킵니다.

#### (C) 부정적 전이(Negative Transfer) 방지

유사도 비율 테스트($\Gamma_j$)를 통해 노이즈가 많은 pseudo-label을 걸러냄으로써, 잘못된 클래스 간 정렬로 인한 부정적 전이를 방지합니다. 이는 특히 **도메인 이동이 큰 경우**의 일반화 성능에 직접적으로 기여합니다.

#### (D) Pairwise Signal의 노이즈 강건성

논문은 Hsu & Kira (2015), Hsu et al. (2019)를 인용하여, **쌍별 유사도 신호(pairwise similarity signal)**가 카테고리 예측보다 레이블 오류에 더 강건하다고 주장합니다. 이는 타겟 도메인에서 레이블이 없더라도 신뢰할 수 있는 훈련 신호를 제공하여 일반화 성능을 높입니다.

#### (E) $O(n^2)$ pseudo-label의 풍부한 학습 신호

미니배치 크기 $n$에 대해 $O(n^2)$개의 pseudo-label을 생성하므로, 일부를 필터링한 후에도 충분한 양의 학습 신호가 남아있어 안정적인 학습이 가능합니다.

#### (F) Plug-in 범용성으로 인한 일반화

ILA-DA는 어떤 adversarial 방법에도 결합 가능한 구조로, 특정 도메인이나 데이터셋에 과적합되지 않는 범용적 설계를 가집니다.

### 3.2 일반화 성능 관련 실험적 증거

tSNE 시각화(Figure 4)에서:
- DANN은 소스-타겟 도메인 정렬에는 성공하지만, 타겟 도메인 내 클래스 분리가 불충분
- ILA-DA는 도메인 정렬과 동시에 **클래스 간 판별적 피처(discriminative features)** 형성

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (A) Contrastive Learning과 UDA의 융합 방향 제시

ILA-DA는 Self-Supervised Contrastive Learning(MoCo, SimCLR)의 아이디어를 도메인 적응에 적용하는 새로운 패러다임을 제시합니다. 이후 연구들이 contrastive learning을 UDA에 더 체계적으로 통합하는 방향에 영향을 미칩니다.

#### (B) 샘플 수준 전이의 중요성 확립

도메인-수준 또는 클래스-수준 정렬을 넘어, **인스턴스 수준(instance-level)** 관계가 도메인 적응에서 중요함을 실험적으로 입증하여, 이후 연구의 세밀한 분석을 촉진합니다.

#### (C) Fine-grained Domain Adaptation 연구 촉진

레이블 계층 없이 fine-grained 적응에서 SOTA를 달성함으로써, 전문 도메인(의료 영상, 위성 영상 등)의 세밀한 적응 연구에 기초를 제공합니다.

#### (D) Plug-in 모듈 설계 패러다임

기존 방법에 추가 손실 함수로 결합 가능한 설계는, 이후 UDA 연구에서 **모듈식(modular)** 설계를 장려합니다.

---

### 4.2 향후 연구 시 고려해야 할 점

#### (A) 메모리 효율적인 Affinity Matrix 구성

현재 $O(n^2)$ 복잡도의 affinity matrix 구성은 클래스 수가 많아질수록 병목이 됩니다. 향후 연구에서는:

- **Approximate Nearest Neighbor(ANN)** 방법(FAISS 등) 활용
- **Memory Bank** 기반 접근법(MoCo 스타일)을 통한 배치 크기 독립적 구성
- **Hierarchical Sampling** 전략으로 클래스 수 확장 가능성

#### (B) 동적 Affinity Matrix 업데이트

현재는 매 이터레이션마다 미니배치 내에서만 affinity를 구성합니다. **전체 데이터셋** 수준의 동적 affinity 업데이트를 통해 더 정확한 유사도 추정이 가능할 것입니다.

#### (C) 멀티-도메인(Multi-domain) 적응으로의 확장

ILA-DA는 소스→타겟의 단일 쌍을 가정합니다. 여러 소스 도메인 또는 여러 타겟 도메인을 동시에 처리하는 **다중 도메인 적응(Multi-source/Multi-target DA)** 설정으로의 확장 연구가 필요합니다.

#### (D) 오픈셋(Open-set) 및 파셜(Partial) 도메인 적응

타겟 도메인이 소스 도메인에 없는 클래스를 포함하는 **오픈셋 시나리오**에서 affinity matrix 기반 방법이 어떻게 작동하는지 연구가 필요합니다.

#### (E) Vision Transformer(ViT) 백본과의 결합

논문은 ResNet-50을 사용하지만, 최근 ViT 기반 피처 추출기와의 결합 시 self-attention이 제공하는 구조적 유사도 정보와 MSC Loss의 시너지 효과를 탐구할 가치가 있습니다.

#### (F) 이론적 분석 강화

MSC Loss가 도메인 격차(domain gap)를 줄이는 메커니즘에 대한 **이론적 보장**이 부족합니다. Ben-David et al.의 이론적 프레임워크를 MSC Loss에 적용한 수렴 분석이 필요합니다.

#### (G) 클래스 불균형 처리

소스 미니배치는 클래스 균형 샘플링을 사용하지만, **실제 데이터는 클래스 불균형**이 심합니다. 불균형 설정에서의 affinity matrix 구성 전략 연구가 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 제가 학습한 지식 범위 내에서 제공되며, 논문 본문에 명시된 내용 외의 최신 연구들은 제 학습 데이터(2021년 이전)를 기반으로 기술합니다. 2022년 이후 논문의 세부 수치는 제가 직접 확인할 수 없어 일부 내용은 생략합니다.

### 5.1 관련 최신 연구 비교표

| 논문 | 핵심 방법 | ILA-DA와의 차이점 |
|---|---|---|
| **SHOT** (Liang et al., ICML 2020) | 소스 없이 타겟만으로 정보 최대화 (Information Maximization) | 소스 데이터 없는 Source-free DA; 인스턴스 수준 아닌 클래스 수준 정보 활용 |
| **MCC** (Jin et al., ECCV 2020) | 최소 클래스 혼동 손실(Minimum Class Confusion) | 클래스 혼동 행렬 기반; 쌍별 인스턴스 관계보다 클래스 전체 통계 활용 |
| **CDTrans** (Xu et al., ICLR 2022) | Cross-Domain Transformer 기반 적응 | ViT 백본 활용; 어텐션 메커니즘으로 유사도 계산 |
| **PMTrans** (Zhu et al., ECCV 2022) | Patch Mix Transformer | 패치 수준 혼합(mixing)으로 도메인 간 중간 도메인 생성 |
| **SSRT** (Sun et al., CVPR 2022) | Safe Self-Refinement for Transformer-based UDA | Self-training과 Transformer 결합 |

### 5.2 ILA-DA와 최신 연구 흐름 비교

#### (1) Contrastive Learning 기반 UDA

ILA-DA는 contrastive learning을 UDA에 도입한 초기 연구 중 하나입니다. 이후 다음 연구들이 유사한 방향을 발전시켰습니다:

- **CDCL** (Wang et al., 2021): Cross-Domain Contrastive Learning for UDA
- **SWD** 기반 연구들: Sliced Wasserstein Distance와 contrastive learning 결합

ILA-DA의 차별점은 **kNN 기반 pseudo-label과 유사도 비율 테스트의 결합**으로, 후속 연구들보다 노이즈 필터링이 명시적입니다.

#### (2) Source-Free Domain Adaptation 트렌드

2021년 이후 소스 데이터 없이 적응하는 **Source-Free DA** 연구가 급증했습니다(SHOT, G-SFDA 등). ILA-DA는 소스 데이터를 필요로 하므로, 이 설정으로의 확장이 필요한 연구 방향입니다.

#### (3) Vision Transformer 기반 DA

CDTrans, PMTrans 등 ViT 기반 방법들은 self-attention이 자연스럽게 패치 간 유사도를 계산한다는 점에서 ILA-DA의 affinity matrix와 개념적으로 유사하나, 아키텍처 수준에서 통합되어 있습니다. ILA-DA는 백본에 독립적인 plug-in 방식이라는 장점이 있습니다.

#### (4) 한계 비교

| 비교 항목 | ILA-DA | 최신 연구 (ViT 기반) |
|---|---|---|
| 메모리 복잡도 | $O(n^2)$ affinity matrix | Self-attention: $O(n^2)$ (패치 수준) |
| 백본 유연성 | 높음 (plug-in) | 일부 백본 종속적 |
| 이론적 보장 | 제한적 | 일부 이론 분석 있음 |
| Source-Free 설정 | 불가 | 일부 가능 |

---

## 참고 자료

**논문 본문 (주요 참고)**
- Astuti Sharma, Tarun Kalluri, Manmohan Chandraker, "Instance Level Affinity-Based Transfer for Unsupervised Domain Adaptation," arXiv:2104.01286v1, 2021. (제공된 PDF)

**논문 내 인용 문헌 (본문 직접 참조)**
- [4] Ben-David et al., "A theory of learning from different domains," *Machine Learning*, 2010.
- [17] Ganin & Lempitsky, "Unsupervised domain adaptation by backpropagation," *ICML*, 2015.
- [36] Long et al., "Conditional adversarial domain adaptation," *NeurIPS*, 2018.
- [55] Saito et al., "Maximum classifier discrepancy for UDA," *CVPR*, 2018.
- [66] Tzeng et al., "Adversarial discriminative domain adaptation," *CVPR*, 2017.
- [71] Wang et al., "Progressive adversarial networks for fine-grained domain adaptation," *CVPR*, 2020.
- [47] van den Oord et al., "Representation learning with contrastive predictive coding," arXiv:1807.03748, 2018.
- [22] He et al., "Momentum contrast for unsupervised visual representation learning," *CVPR*, 2020.
- [25] Hsu & Kira, "Neural network-based clustering using pairwise constraints," arXiv:1511.06321, 2015.
- [1] Arora et al., "A theoretical analysis of contrastive unsupervised representation learning," arXiv:1902.09229, 2019.

**2020년 이후 관련 연구 (비교 분석에 활용, 저의 사전 학습 지식 기반)**
- Liang et al., "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation (SHOT)," *ICML*, 2020.
- Jin et al., "Minimum Class Confusion for Versatile Domain Adaptation (MCC)," *ECCV*, 2020.
- Xu et al., "CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation," *ICLR*, 2022. *(세부 수치 미확인)*

> ⚠️ **정확도 관련 고지**: 섹션 5의 2022년 이후 최신 연구 내용은 저의 학습 데이터 한계로 인해 일부 내용이 부정확할 수 있습니다. 최신 수치 비교는 해당 논문을 직접 확인하시기를 권장합니다.
