# TransAdapter: Vision Transformer for Feature-Centric Unsupervised Domain Adaptation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

TransAdapter는 **Swin Transformer를 기반**으로 비지도 도메인 적응(UDA)을 수행하는 새로운 프레임워크로, 기존 CNN 기반 방법의 한계(복잡한 도메인 관계 포착 불가, 장거리 의존성 모델링 부족)를 극복하고자 한다. 세 가지 핵심 모듈을 통해 **태스크 특화 정렬 모듈 없이도** 다양한 응용에 적응 가능한 범용 UDA 솔루션을 제시한다.

### 주요 기여 (4가지)

| 기여 | 설명 |
|------|------|
| **Graph Domain Discriminator (GDD)** | GCN 기반 비유클리드 관계 모델링, 코사인 유사도 기반 인접 행렬 |
| **Adaptive Double Attention (ADA)** | 윈도우 + 이동 윈도우 이중 어텐션 + 엔트로피 기반 재가중 |
| **Cross-Feature Transform (CFT)** | 양방향 교차 어텐션 + 게이팅 메커니즘으로 동적 특징 변환 |
| **픽셀 수준 특징 변환** | CutMix/MixUp + 고신뢰도 의사 레이블(pseudo-label) 활용 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**도메인 갭(Domain Gap)** 문제:
- 소스 도메인($D_s = \{(X_s, C_s)\}$, 레이블 있음)에서 타겟 도메인($D_t = \{X_t\}$, 레이블 없음)으로 지식 전이 시 발생하는 분포 불일치
- 기존 CNN의 한계: 지역적 공간 상관관계에만 집중, 장거리 의존성 및 복잡한 도메인 관계 포착 불가
- 기존 Swin Transformer의 한계: 지역화된 어텐션으로 인한 장거리 의존성 부족, 고정된 윈도우 분할로 인한 도메인 특화 뉘앙스 일반화 어려움

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) Adaptive Double Attention (ADA)

엔트로피 기반 재가중을 통해 이전 가능한 특징을 우선시:

$$H(F_{\text{graph}}) = -\sum_{i} F_{\text{graph}} \log(F_{\text{graph}}) \tag{1}$$

엔트로피 값으로 어텐션 스코어를 재가중:

$$A = \frac{QK^T}{\sqrt{d}} \odot H(F_{\text{graph}})$$

$$A_{\text{shift}} = \frac{QK_{\text{shift}}^T}{\sqrt{d}} \odot H(F_{\text{graph}}) \tag{2}$$

재가중된 스코어를 결합하여 최종 어텐션 출력:

$$MADA = \text{Softmax}(\text{Concat}(A, A_{\text{shift}})) \times [V; V_{\text{shift}}] \tag{3}$$

트랜스포머 블록 최종 출력:

$$Z_{\text{MADA}} = MADA(\text{LN}(Z_{l-1})) + Z_{l-1}$$

$$Z_l = \text{MLP}(\text{LN}(Z_{\text{MADA}})) + Z_{\text{MADA}} \tag{4}$$

---

#### (B) Graph Domain Discriminator (GDD)

코사인 유사도 기반 인접 행렬 구성:

$$\frac{P(\mathbf{x}_i) \cdot P(\mathbf{x}_j)}{|P(\mathbf{x}_i)||P(\mathbf{x}_j)|} \tag{5}$$

- 3개의 그래프 합성곱 레이어(GCL) + ReLU 활성화
- **Gradient Reversal Layer (GRL)**: min-max 최적화로 도메인 불변 특징 추출 촉진
- $(N-2)$번째 블록(로컬) + $N$번째 블록(글로벌) 특징을 동시에 활용하여 계층적 정렬

---

#### (C) Cross-Feature Transform (CFT)

양방향 교차 어텐션:

$$F_{s2t} = \text{Softmax}\left(f(X_s)^\top g(X_t)\right)$$

$$F_{t2s} = \text{Softmax}\left(g(X_t)^\top f(X_s)\right) \tag{6}$$

학습 가능한 파라미터 $\gamma$를 이용한 게이팅 메커니즘:

$$\text{Attn}_{gating} = (1 - \sigma(\gamma)) \cdot F_{s2t} + \sigma(\gamma) \cdot F_{t2s} \tag{7}$$

쌍별(pairwise) 거리와 게이팅 출력 결합:

$$F_{out} = \left(\text{Attn}_{gating} \times \|F_{s2t} - F_{t2s}\|_2^2\right) + X_t \tag{8}$$

---

#### (D) 목적 함수 (Objective Function)

소스 도메인 분류 손실:

$$\mathcal{L}_{cls} = \text{CE}(F_{cls}, y_s) \tag{9}$$

로컬 및 글로벌 적응 손실:

$$\mathcal{L}_{local} = \frac{1}{2}\left(\text{CE}(F^{src}_{ADV\_local}, \hat{y}^{src}) + \text{CE}(F^{tgt}_{ADV\_local}, \hat{y}^{tgt})\right)$$

$$\mathcal{L}_{global} = \frac{1}{2}\left(FL(F^{src}_{ADV\_global}, \hat{y}^{src}) + FL(F^{tgt}_{ADV\_global}, \hat{y}^{tgt})\right) \tag{10}$$

전체 손실 ($\hat{y}^{src} = 1$, $\hat{y}^{tgt} = 0$):

$$\mathcal{L}_{total} = \lambda_{local}\mathcal{L}_{local} + \lambda_{global}\mathcal{L}_{global} + \mathcal{L}_{cls} \tag{11}$$

$\lambda_{local} = 0.1$, $\lambda_{global} = 0.01$ (모든 DA 태스크에 동일 적용)

---

### 2.3 모델 구조

```
[소스 이미지 H×W×3]          [타겟 이미지 H×W×3]
        ↓ Pixelwise Feature Transform      ↓
  Patch Partition + Linear Embedding    Patch Partition + Linear Embedding
        ↓ Stage 1~4 (Swin Blocks)           ↓ Stage 1~4 (Swin Blocks)
  Cross Feature Transform (CFT) ←→  Cross Feature Transform (CFT)
  [랜덤 선택된 블록에서 적용]
        ↓                                   ↓
  Adaptive Double Attention (ADA) Module
  [Graph Domain Discriminator → 엔트로피 행렬 H 생성]
        ↓
  Fs_ADV_local / Fs_ADV_global      Ft_ADV_local / Ft_ADV_global
        ↓
  Classifier (Fcls)
```

---

### 2.4 성능 향상

#### 주요 벤치마크 결과

| 데이터셋 | Swin-B (기준선) | TransAdapter-B | 주요 경쟁자 |
|----------|----------------|----------------|-------------|
| **Office-31** | 89.8% | **95.5%** | PMTrans-B: 95.3% |
| **Office-Home** | 79.7% | **89.4%** | PMTrans-B: 89.0% |
| **VisDA-2017** | 73.9% | **91.2%** | BCAT-B: 89.2% |
| **DomainNet** | 41.2% | **53.7%** | SSRT: 45.2% |

#### Ablation Study (Office-31 기준, Swin-B 백본)

| 구성 | Office-31 | Office-Home | VisDA-2017 | DomainNet |
|------|-----------|-------------|------------|-----------|
| Swin-B (baseline) | 89.8% | 79.7% | 73.9% | 41.2% |
| +GDD | 91.7% | 81.6% | 78.8% | 47.2% |
| +Pixelwise Transform | 92.9% | 82.6% | 79.8% | 49.3% |
| +CFT | 93.5% | 84.1% | 84.6% | 51.7% |
| +ADA (TransAdapter) | **95.5%** | **87.5%** | **90.2%** | **53.7%** |

---

### 2.5 한계

논문에서 명시적으로 언급한 한계:

1. **계산 복잡도 증가**: 윈도우 어텐션 + 이동 윈도우 어텐션의 동시 처리로 인한 연산량 증가
2. **태스크 특화 모듈 부재**: 객체 탐지(detection) 및 세그멘테이션(segmentation)을 위한 태스크 특화 적응 메커니즘 미구현
3. **대규모 사전학습 의존성**: Swin Transformer의 ImageNet 사전학습에 여전히 의존

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 CFT의 동적 적용을 통한 과적합 방지

CFT 모듈은 **각 반복(iteration)마다 랜덤으로 선택된 트랜스포머 블록 이후에 적용**된다. 이 동적 적용 방식은:

- 모델이 특정 블록에 종속되지 않도록 하여 **과적합(overfitting) 위험 감소**
- 다양한 레벨의 특징에서 도메인 정렬이 이루어지도록 유도
- Sun et al. (2022)의 SSRT에서 영감을 받은 접근법으로, 트랜스포머 기반 자기 정제(self-refinement)의 개념을 확장

$$F_{out} = \left(\text{Attn}_{gating} \times \|F_{s2t} - F_{t2s}\|_2^2\right) + X_t$$

특히 쌍별 거리 항 $\|F_{s2t} - F_{t2s}\|_2^2$은 두 도메인 간 **특징 불일치를 명시적으로 측정**하여 게이팅 출력과 결합함으로써, 도메인 차이가 클수록 더 강한 적응 신호를 제공한다.

---

### 3.2 엔트로피 기반 어텐션의 일반화 기여

$$H(F_{\text{graph}}) = -\sum_{i} F_{\text{graph}} \log(F_{\text{graph}})$$

- **낮은 엔트로피** → 도메인 간 정렬이 잘 된 특징 → 높은 가중치 부여
- **높은 엔트로피** → 도메인 특화 노이즈 → 낮은 가중치 부여

이 메커니즘은 **이전 가능한 특징(transferable features)을 자동으로 선별**하여 새로운 도메인에서도 일관된 표현을 학습하게 한다. t-SNE 시각화(Figure 5)에서 TransAdapter-B가 가장 밀집되고 잘 정렬된 클러스터를 형성함을 확인할 수 있다.

---

### 3.3 GDD의 비유클리드 관계 모델링

기존 DANN과 달리 GDD는 **샘플 간 관계를 그래프 구조로 모델링**한다:

$$\text{Adjacency}_{ij} = \frac{P(\mathbf{x}_i) \cdot P(\mathbf{x}_j)}{|P(\mathbf{x}_i)||P(\mathbf{x}_j)|}$$

- **스케일 불변(scale-invariant)** 코사인 유사도 사용으로 다양한 크기/조명 조건에서 robust
- 얕은 특징 ($(N-2)$번째 블록)과 깊은 특징 ($N$번째 블록)을 동시에 활용하는 **계층적 정렬**
- DomainNet과 같은 **대규모 멀티 도메인 설정**(6개 도메인, 345 카테고리)에서 53.7% 달성, 경쟁 방법 대비 +8.5% 향상

---

### 3.4 픽셀 수준 특징 변환과 의사 레이블의 역할

CutMix와 MixUp을 **고신뢰도 의사 레이블로 가이드**하여 적용:

- 노이즈가 많은 의사 레이블 문제를 **신뢰도 임계값**으로 완화
- 소스 데이터에만 적용함으로써 타겟 도메인 순수성 유지
- CFT와 상호 보완적으로 작동: CFT가 타겟 특징 변환을 담당하므로 직접적인 타겟 변환 불필요

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 4.1 연구에 미치는 영향

#### (1) 트랜스포머 기반 UDA의 새로운 설계 원칙 제시

TransAdapter는 **도메인 정렬을 어텐션 메커니즘 내부에 통합**하는 새로운 패러다임을 보여준다. 기존의 분리된 도메인 적응 모듈(예: DANN의 독립적 판별기) 대신, 엔트로피 기반 재가중을 어텐션 계산 자체에 포함시키는 접근은 이후 연구의 중요한 참고점이 된다.

#### (2) 그래프 기반 도메인 정렬의 가능성 확대

GCN을 도메인 판별기로 활용하여 **비유클리드 공간에서의 도메인 관계 모델링** 가능성을 입증했다. 이는 향후 지식 그래프, 소셜 네트워크 데이터 등 비정형 데이터의 도메인 적응 연구로 확장될 수 있다.

#### (3) 멀티모달/대규모 언어-비전 모델과의 연계 가능성

Du et al. (2024)의 Domain-Agnostic Mutual Prompting처럼, TransAdapter의 특징 정렬 원칙을 **CLIP, ALIGN 등 대규모 비전-언어 모델**에 적용하는 연구가 촉진될 것으로 예상된다.

---

### 4.2 앞으로 연구 시 고려할 점

#### (1) 계산 효율성 문제
이중 어텐션(윈도우 + 이동 윈도우)과 GCN의 동시 적용은 **메모리 및 연산 비용을 상당히 증가**시킨다. 향후 연구에서는:
- **선형 어텐션(Linear Attention)** 또는 **희소 어텐션(Sparse Attention)** 기반의 경량화
- GCN을 **근사 그래프 알고리즘**으로 대체 (예: FastGCN, GraphSAGE)
- **지식 증류(Knowledge Distillation)**를 통한 소형 모델로의 이전

#### (2) 태스크 특화 확장
논문에서 직접 언급한 한계로, **객체 탐지 및 세그멘테이션**에 대한 확장이 필요하다:
- 밀집 예측(Dense Prediction) 태스크를 위한 특징 피라미드 네트워크(FPN)와의 통합
- 픽셀 수준 의사 레이블 생성 전략 개선

#### (3) 소스 프리(Source-Free) 및 멀티 소스 설정
현재 TransAdapter는 소스 데이터 접근을 가정하지만, **프라이버시 보호** 관점에서 소스 프리 도메인 적응(SFDA)으로의 확장이 중요해지고 있다. Sanyal et al. (2023)의 Domain-Specificity Inducing Transformers와 같은 접근법을 참고할 필요가 있다.

#### (4) 의사 레이블 품질 개선
신뢰도 임계값 기반 필터링은 단순하지만 **클래스 불균형 문제**를 야기할 수 있다. 향후:
- **교사-학생(Teacher-Student)** 프레임워크를 통한 점진적 의사 레이블 정제
- **분포 보정(Distribution Calibration)** 기법 적용

#### (5) 멀티 도메인 적응으로의 확장
DomainNet과 같은 6개 도메인 설정에서의 **멀티 소스 도메인 적응(MSDA)** 성능을 더욱 개선하기 위해, CFT의 양방향 메커니즘을 다수의 소스 도메인으로 확장하는 연구가 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 백본 | 핵심 아이디어 | Office-Home (Avg) | VisDA-2017 (Avg) |
|------|------|------|---------------|-------------------|------------------|
| **SHOT** (Liang et al.) | 2020 | ResNet | 소스 가설 전이, 정보 극대화 | 71.8% | 82.9% |
| **CDTrans** (Xu et al.) | 2021 | DeiT-B | 교차 도메인 트랜스포머, 3방향 어텐션 | 80.5% | 88.4% |
| **TVT** (Yang et al.) | 2023 | ViT-B | 이전 가능한 비전 트랜스포머, 토큰 가중 | 83.6% | 86.7% |
| **SSRT** (Sun et al.) | 2022 | ViT-B | 안전한 자기 정제, 동적 의사 레이블 | 85.4% | 88.7% |
| **BCAT** (Wang et al.) | 2022 | Swin-B | 양방향 교차 어텐션 트랜스포머 | 86.6% | 89.2% |
| **PMTrans** (Zhu et al.) | 2023 | Swin-B | 패치 믹스 트랜스포머, 게임 이론적 관점 | 89.0% | 88.0% |
| **PADCLIP** (Lai et al.) | 2023 | ViT-B | CLIP 기반 적응적 편향 제거 | 86.7% | 90.9% |
| **PDA** (Bai et al.) | 2024 | ViT-B | 프롬프트 기반 분포 정렬 | 85.7% | 89.7% |
| **TransAdapter** (Ours) | 2024 | Swin-B | GDD + ADA + CFT 통합 | **89.4%** | **91.2%** |

### 분석 요점

1. **트랜스포머 백본의 우세**: 2021년 이후 DeiT, ViT, Swin Transformer가 ResNet을 빠르게 대체하며 UDA 성능의 기준점을 높였다.

2. **어텐션 메커니즘의 진화**: CDTrans(3방향 어텐션) → BCAT(양방향 교차 어텐션) → TransAdapter(이중 어텐션 + 엔트로피 재가중)로 점진적으로 정교해지고 있다.

3. **그래프 구조 활용의 차별점**: 대부분의 경쟁 방법이 어텐션 기반 정렬에 집중하는 반면, TransAdapter는 GCN으로 **비유클리드 도메인 관계**를 명시적으로 모델링하는 차별화된 접근을 취한다.

4. **대규모 데이터셋(DomainNet)에서의 우위**: CDTrans(45.2%), SSRT(45.2%)와 비교하여 TransAdapter(53.7%)는 **약 8.5%p의 현저한 성능 격차**를 보이며, 복잡한 다중 도메인 시나리오에서의 강건성을 입증한다.

5. **PADCLIP과의 비교**: CLIP 기반의 대규모 비전-언어 사전학습을 활용한 PADCLIP이 VisDA-2017에서 90.9%로 TransAdapter(91.2%)에 근접하는 성능을 보이는 것은, 향후 TransAdapter에 대규모 사전학습 모델 활용을 통합하는 연구 방향의 가능성을 시사한다.

---

## 참고 자료

1. **주논문**: Doruk, A. E., Oztop, E., & Ates, H. F. (2024). *TransAdapter: Vision Transformer for Feature-Centric Unsupervised Domain Adaptation*. arXiv:2412.04073v1.

2. Liu, Z. et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*. ICCV 2021. arXiv:2103.14030.

3. Xu, T. et al. (2021). *CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation*. arXiv:2109.06165.

4. Sun, T. et al. (2022). *Safe Self-Refinement for Transformer-based Domain Adaptation*. CVPR 2022.

5. Yang, J. et al. (2023). *TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation*. WACV 2023.

6. Zhu, J., Bai, H., & Wang, L. (2023). *Patch-Mix Transformer for Unsupervised Domain Adaptation: A Game Perspective*. CVPR 2023.

7. Wang, X., Guo, P., & Zhang, Y. (2022). *Domain Adaptation via Bidirectional Cross-Attention Transformer*. arXiv:2201.05887.

8. Lai, Z. et al. (2023). *PADCLIP: Pseudo-labeling with Adaptive Debiasing in CLIP for Unsupervised Domain Adaptation*. ICCV 2023.

9. Bai, S. et al. (2024). *Prompt-based Distribution Alignment for Unsupervised Domain Adaptation*. AAAI 2024.

10. Du, X., Li, C., & Zhao, K. (2024). *Domain-Agnostic Mutual Prompting for Unsupervised Domain Adaptation*. CVPR 2024.

11. Ganin, Y. & Lempitsky, V. (2015). *Unsupervised Domain Adaptation by Backpropagation*. ICML 2015.

12. Sanyal, R., Gupta, A., & Roy, P. (2023). *Domain-Specificity Inducing Transformers for Source-Free Domain Adaptation*. arXiv:2308.14023.

13. Alijani, S., Zhou, Y., & Wang, M. (2024). *A Comprehensive Survey on Vision Transformers in Domain Adaptation and Domain Generalization*. arXiv:2404.04452.
