
# Deep Domain Adaptation by Geodesic Distance Minimization

## Executive Summary

본 논문은 비지도 시각 도메인 적응(Unsupervised Visual Domain Adaptation) 분야에서 Deep CORAL 방법을 개선한 **Deep LogCORAL**을 제안합니다. 핵심 기여는 공분산 행렬 간 거리를 측정할 때 기하학적으로 더 정확한 Log-Euclidean 거리(리만 다양체 위의 측지거리)를 사용하고, 1차 통계정보(평균)와 2차 통계정보(공분산)를 동시에 활용한다는 점입니다. Office 데이터셋 실험에서 기존 Deep CORAL 대비 평균 2.28% 성능향상을 달성했으며, 이는 도메인 적응 분야의 통계 기반 방법의 이론적 기초를 강화하는 의미 있는 발전입니다.

***

## 1. 해결하고자 하는 문제

### 1.1 도메인 적응의 근본적 문제

전통적 기계학습은 훈련 데이터와 테스트 데이터가 동일한 분포를 따른다는 가정을 기반으로 합니다. 그러나 실제 시각인식 응용에서는 이 가정이 성립하지 않습니다. 예를 들어, Amazon 온라인 상품 이미지로 학습한 물체 인식 모델을 DSLR 카메라나 웹캠 이미지에 적용하면 성능이 급격히 저하됩니다. 이를 **도메인 시프트(domain shift)** 또는 **분포 불일치(distribution mismatch)**라 합니다.

데이터 주석 비용이 높아 모든 도메인에 대해 충분한 레이블된 데이터를 획득하기 어렵다는 실무 제약이 존재합니다. 따라서 레이블된 소스 도메인에서 레이블 없는 타겟 도메인으로 지식을 전이하는 **비지도 도메인 적응**이 중요합니다.

### 1.2 Deep CORAL의 한계

Deep CORAL은 소스와 타겟 도메인의 공분산 행렬(covariance matrix) 간 유클리드 거리를 최소화합니다: [arxiv](https://arxiv.org/pdf/2508.12987.pdf)

$$L_{CORAL} = \frac{1}{4d^2} \|C_S - C_T\|_F^2$$

여기서 $C_S$, $C_T$는 각각 소스와 타겟 도메인의 공분산 행렬입니다.

**Deep CORAL의 문제점**:
1. **기하학적 부정확성**: 공분산 행렬은 양의 준정부호(positive semi-definite, PSD) 행렬로서 유클리드 공간이 아닌 리만 다양체 위에 존재합니다. 유클리드 거리는 이 기하학적 구조를 무시합니다.

2. **정보 손실**: 2차 통계정보(공분산)만 사용하고 1차 통계정보(평균)는 무시하여 불완전한 분포 정렬을 초래합니다.

3. **스케일링 부작용(Swelling Effect)**: 유클리드 거리에서 리만 다양체 위의 점들이 거리에 따라 부자연스럽게 팽창할 수 있습니다.

***

## 2. 제안하는 방법

### 2.1 Log-Euclidean 거리와 리만 다양체

**기본 개념**: PSD 행렬의 집합은 리만 다양체를 형성합니다. 두 PSD 행렬 $A$, $B$의 리만 거리는 측지거리(geodesic distance)로 정의되며, Log-Euclidean 메트릭을 사용하여 근사할 수 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/16c73371-1123-4a06-adec-c209bc8adf89/1707.09842v2.pdf)

Log-Euclidean 메트릭의 핵심 성질은 로그 함수가 리만 다양체를 평탄(flat)한 유클리드 공간으로 변환한다는 것입니다. 이를 통해 복잡한 리만 기하학 계산을 유클리드 공간에서의 간단한 계산으로 대체할 수 있습니다.

### 2.2 LogCORAL 손실 함수

공분산 행렬의 특이값 분해(Singular Value Decomposition, SVD):

$$C_S = U_S \Sigma_S U_S^T, \quad C_T = U_T \Sigma_T U_T^T$$

로그 연산자 정의:

$$\log(C_S) = U_S \log(\Sigma_S) U_S^T$$

여기서 $\log(\Sigma_S)$는 대각행렬의 대각 원소에 로그를 적용합니다.

**LogCORAL 손실**:

$$L_{LogCORAL} = \frac{1}{4d^2} \|\log(C_S) - \log(C_T)\|_F^2$$

이는 리만 다양체의 측지거리를 근사하며, 유클리드 거리보다 기하학적으로 더 적절합니다.

### 2.3 역전파 계산

LogCORAL 손실의 역전파는 행렬 미분 및 리만 기하학을 이용합니다. $C_S' = \log(C_S)$, $C_T' = \log(C_T)$로 표기하면:

$$\frac{\partial L_{LogCORAL}}{\partial C_S} = \frac{1}{2d^2}(C_S' - C_T') \frac{\partial C_S'}{\partial C_S}$$

여기서 편미분 항은:

$$\frac{\partial C_S'}{\partial C_S} = U_S(P^T \circ (U_S^T dU_S))_{sym} U_S^T + U_S(d\Sigma_S)_{diag} U_S^T$$

각 성분의 정의:

$$P(i,j) = \begin{cases} \frac{1}{\sigma_i - \sigma_j} & i \neq j \\ 0 & i = j \end{cases}$$

$$dU_S = 2\left(\frac{\partial L_{LogCORAL}}{\partial C_S'}\right)_{sym} U_S \log(\Sigma_S)$$

$$d\Sigma_S = \Sigma_S^{-1} U_S^T \left(\frac{\partial L_{LogCORAL}}{\partial C_S'}\right)_{sym} U_S$$

여기서 $\circ$는 Hadamard 곱(요소별 곱), $(\cdot)\_{sym}$ 은 대칭화 연산, $(\cdot)_{diag}$는 대각 유지 연산입니다.

### 2.4 평균 손실 도입

1차 통계정보를 활용하기 위해 도메인 간 평균 벡터 거리를 추가합니다:

$$L_{Mean} = \frac{1}{2d} \|{\mathbf{1}}^T D_S - {\mathbf{1}}^T D_T\|_2^2$$

여기서 $\mathbf{1}$은 모든 원소가 1인 벡터, $D_S$, $D_T$는 각각 소스와 타겟 도메인의 특징입니다. 이는 Maximum Mean Discrepancy(MMD) 이론과 밀접합니다.

### 2.5 모델 구조

엔드-투-엔드 학습 구조는 다음과 같습니다:

**총 손실함수**:

$$L_{Total} = L_{cls} + \lambda_1 L_{LogCORAL} + \lambda_2 L_{Mean}$$

- $L_{cls}$: 소스 도메인의 분류 손실 (교차 엔트로피)
- $\lambda_1$, $\lambda_2$: 하이퍼파라미터
- fc7 (또는 fc8) 레이어에서 LogCORAL과 Mean 손실을 계산

**학습 안정성**: 이동 평균(moving average)을 사용하여 각 배치의 공분산과 평균을 누적합니다:

$$C^{new} = 0.9 \cdot C^{old} + 0.1 \cdot C^{batch}$$

$$M^{new} = 0.9 \cdot M^{old} + 0.1 \cdot M^{batch}$$

***

## 3. 성능 향상 및 실험 결과

### 3.1 벤치마크 및 실험 설정

**데이터셋**: Office-31 (알라메딕)
- Amazon (A): 31개 카테고리, ~2,817개 이미지
- Webcam (W): 31개 카테고리, ~795개 이미지  
- DSLR (D): 31개 카테고리, ~498개 이미지

**모델**: AlexNet (ImageNet 사전학습)

**적응 시나리오**: 6가지 (A→W, D→W, A→D, W→D, W→A, D→A)

### 3.2 정량적 결과

| 방법 | A→W | D→W | A→D | W→D | W→A | D→A | 평균 |
|------|-----|-----|-----|-----|-----|-----|------|
| CNN (적응 없음) | 63.34% | 95.21% | 65.14% | 99.26% | 49.23% | 51.37% | 70.59% |
| Deep CORAL | 66.12% | 95.24% | 66.38% | 99.24% | 50.71% | 53.12% | 71.80% |
| **LogCORAL만 사용** | 68.83% | 95.23% | 68.64% | 99.52% | 50.94% | 51.73% | 72.48% |
| **Mean만 사용** | 66.29% | 95.56% | 68.67% | 99.51% | 49.83% | 50.74% | 71.77% |
| **LogCORAL + Mean (제안)** | **70.15%** | **95.45%** | **69.41%** | **99.46%** | **51.57%** | **51.15%** | **72.87%** |

**성능 개선**:
- 대비 CNN: **+2.28%** (평균 정확도)
- 대비 Deep CORAL: **+1.07%** (평균 정확도)
- 5/6 도메인 적응 시나리오에서 최고 성능 달성

### 3.3 절제 연구(Ablation Study)

표 2의 절제 연구는 각 손실 함수 성분의 기여도를 분석합니다:

- **LogCORAL 단독**: Deep CORAL 대비 +0.68% 개선 → Log-Euclidean 거리의 효과 입증
- **Mean 단독**: Deep CORAL 대비 -0.03% (약간의 성능 저하) → 1차 정보 단독으로는 부족
- **LogCORAL + Mean**: Deep CORAL 대비 +1.07% 개선 → 1차와 2차 정보의 상보성 입증

### 3.4 손실 함수 상관관계 분석

Figure 4 시각화 분석:

- **(a) CNN 기저선**: LogCORAL과 Mean 손실이 모두 증가 (미적응 상태)
- **(b) Mean 최적화**: Mean 손실은 급격히 감소하나, LogCORAL 손실은 안정적 → 두 손실 간 약한 상관관계
- **(c) LogCORAL 최적화**: LogCORAL 손실은 감소하나, Mean 손실은 증가 → 직교 특성 강화
- **(d) 결합 최적화**: 두 손실이 모두 감소하며 최고 성능 달성

**해석**: 1차와 2차 통계정보는 서로 약한 상관관계를 가지므로, 동시에 최소화하면 도메인 갭의 다양한 측면을 포괄적으로 해소할 수 있습니다.

***

## 4. 모델의 일반화 성능 향상 메커니즘

### 4.1 이론적 기반

**도메인 적응의 상한(Upper Bound)**[참고문헌 참조]:

$$\epsilon_T(\mathcal{h}) \leq \epsilon_S(\mathcal{h}) + d_{H\Delta H}(S, T) + \lambda$$

여기서:
- $\epsilon_T(\mathcal{h})$: 타겟 도메인 오류
- $\epsilon_S(\mathcal{h})$: 소스 도메인 오류
- $d_{H\Delta H}(S,T)$: H-발산(domain discrepancy)
- $\lambda$: 이상적 가설 오류

Deep LogCORAL의 개선 메커니즘:

1. **도메인 차이 감소**: Log-Euclidean 거리가 기하학적으로 정확하므로 더 효과적인 분포 정렬을 달성하여 $d_{H\Delta H}(S,T)$를 더 크게 감소
2. **특징 판별성 보존**: 1차 정보만으로는 부족하나, 2차 정보와 결합하면 클래스 간 판별성을 유지하면서 도메인 불변성 확보
3. **최적화 안정성**: 이동 평균과 역전파의 안정적 계산으로 수렴성 보장

### 4.2 모델 일반화의 조건

논문의 실험 결과는 다음을 시사합니다:

**조건 1 - 적절한 거리 메트릭의 중요성**
- Deep CORAL의 유클리드 거리는 선형 변환에 불변성이 없음
- Log-Euclidean 거리는 리만 불변성 성질로 기하학적으로 자연스러운 변환 포착

**조건 2 - 다중 통계 차수 활용의 필수성**
- 2차 정보(공분산)만으로는 부족: 특정 도메인 쌍(예: W→A, D→A)에서 성능 정체
- 1차 정보(평균)와의 결합으로 2-3%의 성능 향상 달성

**조건 3 - 배치 단위 누적의 필요성**
- 이동 평균을 통한 누적으로 학습 안정성 확보
- 배치 크기 변화에 대한 강건성 증대

### 4.3 일반화 성능 향상의 실제 사례

**A→W 적응 (온라인 쇼핑 이미지 → 웹캠 이미지)**:
- 배경, 조명, 각도 등 시각적 특성이 크게 다름
- Deep CORAL: 66.12% → LogCORAL+Mean: 70.15% (**+4.03%p**)
- 도메인 간 공분산 구조의 미묘한 차이를 Log-Euclidean 메트릭이 포착

**W→D 적응 (웹캠 → DSLR 카메라)**:
- DSLR의 높은 해상도가 특징 분포 변경 유발
- 평균 정렬만으로는 부족하나, 공분산 정렬 추가로 최고 성능 달성 (99.46%)

***

## 5. 방법의 한계

### 5.1 컴퓨팅 복잡도

SVD 계산이 필요하므로 특징 차원 $d$에 대해 $O(d^3)$ 복잡도. 고차원 특징에서:
- AlexNet fc8 (4096차원): 실용적이나
- 더 큰 네트워크에서는 계산 비용 증가

**해결책**: 특징 차원 축소 또는 근사 기법 필요

### 5.2 대규모 도메인 갭에서의 성능

Office-31 자체가 비교적 작은 도메인 갭을 가짐. VisDA 같은 더 큰 갭(합성 이미지 → 실제 이미지)에서 성능 미검증.

### 5.3 다중 도메인 시나리오 미지원

- 다중 소스 도메인 적응(Multi-source DA): 구조 확장 필요
- 부분 도메인 적응(Partial DA): 클래스 집합 불일치 처리 전략 부재

### 5.4 적응 메커니즘의 순수성

- 도메인 차이 손실과 분류 손실의 가중치 결정이 경험적
- 이론적 지침 없음

***

## 6. 2020년 이후 관련 최신 연구 비교 분석

### 6.1 자기지도학습(Self-Supervised Learning) 기반 접근

**주요 연구**:
- **DCCL (Domain Confused Contrastive Learning, 2022)**: 도메인 퍼즐을 통해 대비학습을 UDA에 적용 [arxiv](https://arxiv.org/abs/2207.04564)
- **Contrastive Domain Adaptation (2021)**: SimCLR을 도메인 적응에 확장, 거짓 음성(false negatives) 제거 [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2021W/WiCV/papers/Thota_Contrastive_Domain_Adaptation_CVPRW_2021_paper.pdf)

**Deep LogCORAL과의 비교**:

| 특징 | Deep LogCORAL | 자기지도 대비학습 |
|------|----------------|-----------------|
| 감독 정보 | 소스 레이블 필수 | 레이블 불필요 가능 |
| 손실함수 | 통계적 정렬 (거리 기반) | 대비적 정렬 (유사성 기반) |
| 계산 복잡도 | 중간 ( $O(d^3)$ ) | 높음 (배치 간 비교) |
| 성능 | Office: 72.87% | 최신: 75%+ 보고 |
| 우점 | 해석가능성, 이론적 기초 | 확장성, 대규모 데이터 |

### 6.2 비전 파운데이션 모델(VFM) 기반 적응

**주요 연구**:
- **CLIP 기반 적응 (2023-2025)**: 언어-이미지 정렬을 도메인 적응에 활용
- **DINOv2 정렬 기반 도메인 적응 (2025)**: 대규모 자기지도 모델의 특징과 소형 모델 정렬 [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2025/papers/Lavoie_Large_Self-Supervised_Models_Bridge_the_Gap_in_Domain_Adaptive_Object_CVPR_2025_paper.pdf)

**최신 방법의 특징**:
- 사전학습된 대규모 모델의 강력한 특징 활용
- 문맥 정보(언어, 의미)를 직접 포함
- Office-31: 85%+ 달성 가능

**Deep LogCORAL의 위치**:
- 기초 이론 제공: VFM 기반 방법들도 통계 정렬 개념 활용
- 소형 모델/제약 환경: 여전히 효율적
- 해석 가능성: 기하학적 기초가 명확

### 6.3 Source-Free Domain Adaptation (SFDA)

**문제 정의**: 학습 시 소스 데이터 접근 불가, 사전학습 모델만 이용

**주요 연구**:
- **SHOT (2020)**: 소스 모델 가중치 제약 + 타겟 클러스터링 [openaccess.thecvf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Model_Adaptation_Unsupervised_Domain_Adaptation_Without_Source_Data_CVPR_2020_paper.pdf)
- **3C-GAN (2020)**: 생성 적대 네트워크로 타겟 스타일 데이터 생성 [openaccess.thecvf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Model_Adaptation_Unsupervised_Domain_Adaptation_Without_Source_Data_CVPR_2020_paper.pdf)
- **Survey (2024)**: SFDA 시스템 리뷰 [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/38490115/)

**Deep LogCORAL의 제약**:
- 소스 도메인 공분산 계산 필요
- SFDA로 직접 확장 불가능

**향상 가능성**:
- 소스 공분산 사전계산 저장 후 활용 가능
- 메타-러닝 결합으로 확장 가능

### 6.4 도메인 일반화(Domain Generalization) 패러다임

**차이점**:
- **도메인 적응**: 타겟 도메인 데이터(비레이블) 활용 O
- **도메인 일반화**: 미시 타겟 도메인, 시스 레이블 데이터만으로 일반화

**최신 경향 (2024-2025)**:
- **Meta-Learning**: 다중 도메인에서 메타-학습으로 일반화
- **Discrete Domain Generalization (2025)**: 이산 특징 코드북으로 의미 수준 정렬 [arxiv](https://arxiv.org/pdf/2504.06572.pdf)
- **Continuous Domain Generalization (NeurIPS 2025)**: 연속 도메인 변화 모델링 [neurips](https://neurips.cc/virtual/2025/poster/118589)

**Deep LogCORAL의 통합 가능성**:
- Log-Euclidean 메트릭이 다중 도메인 공분산 정렬에 유용
- 메타-학습과 결합 시 더 강건한 일반화 가능

### 6.5 멀티모달 및 멀티소스 적응

**주요 연구**:
- **Multi-source Domain Adaptation Survey (2020)**: 여러 소스에서의 지식 결합 [semanticscholar](https://www.semanticscholar.org/paper/45b932394eb565c18c2d8043721e79b478ae38f1)
- **Link-based Contrastive Learning for One-Shot UDA (CVPR 2025)**: 극소수 소스 샘플로 적응 [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Link-based_Contrastive_Learning_for_One-Shot_Unsupervised_Domain_Adaptation_CVPR_2025_paper.pdf)

**Deep LogCORAL 확장**:

다중 소스 시나리오에서의 가중 공분산 정렬:

$$L_{MS-LogCORAL} = \sum_{k=1}^{K} w_k \|\log(C_{S_k}) - \log(C_T)\|_F^2$$

여기서 $w_k$는 소스 $k$의 신뢰도 가중치.

***

## 7. 앞으로의 연구 방향 및 고려사항

### 7.1 이론적 발전 방향

**1. 리만 거리의 다양한 선택**
- 현재: Log-Euclidean (선형 근사)
- 향상 가능: 정보 기하학(Information Geometry) 거리, Wasserstein 거리 등
- 각 거리가 도메인 적응에 미치는 이론적 영향 분석

**2. 고차 통계정보 활용**
- 현재: 1차(평균), 2차(공분산)
- 향상 가능: 3차(왜도), 4차(첨도) 정보 통합
- 필요성 검증 및 계산 효율성 개선

**3. 도메인 차이 상한의 개선**

현재 상한:
$$\epsilon_T \leq \epsilon_S + d_{H\Delta H} + \lambda$$

개선된 상한 제안:
$$\epsilon_T \leq \epsilon_S + \alpha \cdot d_{LogCORAL} + \beta \cdot d_{Mean} + \lambda$$

여기서 $\alpha$, $\beta$를 통계적으로 유도.

### 7.2 방법론적 개선

**1. 계산 효율화**
```
제안: 특징 차원 축소 후 LogCORAL 적용
- PCA 또는 주성분 분석으로 차원 축소
- 근사 SVD (Randomized SVD) 활용
- 시간복잡도: O(d^3) → O(d'^3) where d' << d
```

**2. 적응 가중치 자동 결정**

현재: 고정 하이퍼파라미터 $\lambda_1$, $\lambda_2$

향상:
$$L_{Total} = L_{cls} + f(\Delta_{domain}) \cdot L_{LogCORAL} + g(\Delta_{domain}) \cdot L_{Mean}$$

여기서 $f$, $g$는 도메인 차이 크기에 적응하는 함수.

**3. 배치 정규화와의 통합**

배치 정규화(Batch Normalization)는 미니배치 통계를 사용하므로 도메인 적응과 상호작용:
- 도메인별 배치 정규화 파라미터 관리
- Adaptive Instance Normalization 고려

### 7.3 응용 확장

**1. Source-Free 적응 확장**
```
방법: 소스 공분산 사전계산 저장
1. 학습 단계: $\bar{C}_S = E[C_S]$ 계산 및 저장
2. 적응 단계: 타겟 특징만으로
   $L_{SFDA} = \|\log(C_T) - \log(\bar{C}_S)\|_F^2$
   + 엔트로피 최소화 손실
```

**2. 부분 도메인 적응(Partial DA)**

소스 클래스 > 타겟 클래스 시나리오:
$$L_{Partial} = L_{LogCORAL} + L_{Mean} + \sum_{c=1}^{C_T} \max(0, m - \text{confidence}_c)$$

**3. 열린 집합 적응(Open-Set DA)**

미지의 타겟 클래스 처리:
- 클래스 외 탐지 손실 추가
- 특징 공간 이상치 탐지

### 7.4 최신 트렌드와의 결합

**1. 자기지도학습(SSL) 통합**

$$L_{Total} = L_{cls} + L_{LogCORAL} + L_{Mean} + L_{SSL}$$

여기서 $L_{SSL}$은 대비 손실이나 마스크 예측 손실.

**2. VFM 기반 특징 공간 적응**

$$L = L_{LogCORAL}(\text{feat}_{small}, \text{feat}_{VFM}) + L_{task}$$

사전학습 대규모 모델의 특징 공간에 소형 모델 정렬.

**3. 멀티모달 적응**

이미지와 텍스트 정보 결합:
$$L = L_{visual} + L_{text} + L_{cross-modal}$$

### 7.5 실제 배포 시 고려사항

**1. 프라이버시 보호**
- 소스 데이터 저장 최소화
- 통계(공분산, 평균)만 유지 가능

**2. 적응 비용 최소화**
- 온라인 학습: 스트리밍 데이터로 점진적 적응
- 연합 학습: 중앙 수집 없이 분산 적응

**3. 견고성 검증**
- 적대적 샘플에 대한 견고성
- 분포 외(Out-of-Distribution) 탐지 능력

***

## 8. 결론

**Deep LogCORAL**은 리만 기하학의 관점에서 도메인 적응 문제에 접근한 의미 있는 연구입니다. Log-Euclidean 거리를 통한 측지거리 도입과 1차·2차 통계정보의 통합은 이론적으로 견고하고 실험적으로 검증된 개선을 제공합니다.

### 핵심 성과
- 기하학적으로 정확한 거리 메트릭 제안
- Office-31에서 Deep CORAL 대비 1.07%, CNN 대비 2.28% 성능향상
- 통계 기반 도메인 적응의 이론적 기초 강화

### 향후 영향
- 2020년 이후 연구: 자기지도학습, VFM, 도메인 일반화 등이 주류
- Deep LogCORAL의 유산: 통계적 정렬의 중요성 재확인, 리만 기하학의 가치 입증
- 융합 가능성: 최신 기법(SSL, VFM, 메타-러닝)과 결합 시 성능 재상향 기대

### 실무 적용성
- 중소 규모 도메인 갭: 강력한 기초 모델
- 계산 제약 환경: 효율적 선택지
- 해석 가능성 필요: 기하학적 기초의 명확성

이 논문은 단순한 성능 개선을 넘어 도메인 적응의 기하학적 이해를 심화시켰으며, 후속 연구들이 더욱 정교한 방법론을 개발하는 토대를 제공했습니다.

***

## 참고 자료

<span style="display:none">[^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92]</span>

<div align="center">⁂</div>

[^1_1]: https://arxiv.org/pdf/2508.12987.pdf

[^1_2]: 1707.09842v2.pdf

[^1_3]: https://arxiv.org/abs/2207.04564

[^1_4]: https://openaccess.thecvf.com/content/CVPR2021W/WiCV/papers/Thota_Contrastive_Domain_Adaptation_CVPRW_2021_paper.pdf

[^1_5]: https://openaccess.thecvf.com/content/CVPR2025/papers/Lavoie_Large_Self-Supervised_Models_Bridge_the_Gap_in_Domain_Adaptive_Object_CVPR_2025_paper.pdf

[^1_6]: https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Model_Adaptation_Unsupervised_Domain_Adaptation_Without_Source_Data_CVPR_2020_paper.pdf

[^1_7]: https://pubmed.ncbi.nlm.nih.gov/38490115/

[^1_8]: https://arxiv.org/pdf/2504.06572.pdf

[^1_9]: https://neurips.cc/virtual/2025/poster/118589

[^1_10]: https://www.semanticscholar.org/paper/45b932394eb565c18c2d8043721e79b478ae38f1

[^1_11]: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Link-based_Contrastive_Learning_for_One-Shot_Unsupervised_Domain_Adaptation_CVPR_2025_paper.pdf

[^1_12]: https://ieeexplore.ieee.org/document/8809926/

[^1_13]: http://www.thieme-connect.de/DOI/DOI?10.1055/s-0040-1702009

[^1_14]: https://iopscience.iop.org/article/10.1088/1361-6501/ab64aa

[^1_15]: https://dl.acm.org/doi/10.1145/3320269.3384718

[^1_16]: https://ieeexplore.ieee.org/document/9096390/

[^1_17]: https://www.mdpi.com/2227-9032/8/4/437

[^1_18]: https://link.springer.com/10.1007/978-3-030-68107-4_20

[^1_19]: https://www.mdpi.com/2079-9292/9/12/2140

[^1_20]: https://link.springer.com/10.1007/s40435-020-00669-0

[^1_21]: https://arxiv.org/html/2502.06272v1

[^1_22]: http://arxiv.org/pdf/2403.10834.pdf

[^1_23]: https://arxiv.org/pdf/1607.01719.pdf

[^1_24]: https://arxiv.org/pdf/2410.16020v1.pdf

[^1_25]: https://www.aclweb.org/anthology/P18-1099.pdf

[^1_26]: https://arxiv.org/pdf/1811.05443.pdf

[^1_27]: https://arxiv.org/pdf/1502.02791.pdf

[^1_28]: https://www.aclweb.org/anthology/D19-6109.pdf

[^1_29]: https://arxiv.org/html/2507.09420v1

[^1_30]: https://arxiv.org/html/2407.17877v1

[^1_31]: https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1410.pdf

[^1_32]: https://arxiv.org/abs/2312.04066

[^1_33]: https://arxiv.org/pdf/2304.02104.pdf

[^1_34]: https://arxiv.org/pdf/2502.14214.pdf

[^1_35]: https://arxiv.org/html/2404.09685v1

[^1_36]: https://www.arxiv.org/pdf/2503.10685v2.pdf

[^1_37]: https://arxiv.org/html/2510.01660v1

[^1_38]: https://arxiv.org/pdf/2106.10600.pdf

[^1_39]: https://arxiv.org/abs/2502.10694

[^1_40]: https://arxiv.org/abs/2412.17325

[^1_41]: https://openreview.net/pdf/8fcab8944f41b0e7d3e2f35e07f915aed86e5a34.pdf

[^1_42]: https://www.sciencedirect.com/science/article/abs/pii/S0952197623003561

[^1_43]: https://dl.acm.org/doi/10.1145/3400066

[^1_44]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04650.pdf

[^1_45]: https://cvpr.thecvf.com/virtual/2024/poster/30672

[^1_46]: https://www.sciencedirect.com/science/article/pii/S2213846323000627

[^1_47]: https://www.sciencedirect.com/science/article/abs/pii/S0952197625030787

[^1_48]: https://icml.cc/virtual/2025/poster/44848

[^1_49]: https://openreview.net/forum?id=ZWsjNcUDIs

[^1_50]: https://dmqa.korea.ac.kr/activity/seminar/415

[^1_51]: https://ieeexplore.ieee.org/document/10502168/

[^1_52]: https://www.sciencedirect.com/science/article/abs/pii/S0020025521007751

[^1_53]: https://arxiv.org/abs/2508.07514

[^1_54]: https://arxiv.org/abs/2506.10097

[^1_55]: https://www.semanticscholar.org/paper/ba32e73da73d62e34f721d2592661241850821fb

[^1_56]: https://arxiv.org/abs/2507.07495

[^1_57]: https://journals.sagepub.com/doi/10.1177/00131644251344973

[^1_58]: https://www.frontiersin.org/articles/10.3389/frobt.2025.1604472/full

[^1_59]: https://arxiv.org/abs/2505.15422

[^1_60]: https://ieeexplore.ieee.org/document/11119540/

[^1_61]: https://ieeexplore.ieee.org/document/11012116/

[^1_62]: https://arxiv.org/abs/2507.14239

[^1_63]: http://arxiv.org/pdf/2208.00898.pdf

[^1_64]: http://arxiv.org/pdf/1710.03463.pdf

[^1_65]: https://arxiv.org/abs/2210.14507

[^1_66]: https://arxiv.org/pdf/2401.08464.pdf

[^1_67]: http://arxiv.org/pdf/2308.09931.pdf

[^1_68]: https://arxiv.org/pdf/2407.15085.pdf

[^1_69]: https://arxiv.org/pdf/2209.14926.pdf

[^1_70]: https://arxiv.org/html/2412.05551v1

[^1_71]: https://www.arxiv.org/pdf/2510.04441.pdf

[^1_72]: https://pubmed.ncbi.nlm.nih.gov/38373127/

[^1_73]: https://arxiv.org/pdf/2503.06288.pdf

[^1_74]: https://arxiv.org/abs/2509.09935

[^1_75]: https://arxiv.org/abs/2404.11269

[^1_76]: https://arxiv.org/abs/2412.02856

[^1_77]: https://arxiv.org/html/2509.09935v1

[^1_78]: https://arxiv.org/abs/2206.06243

[^1_79]: https://arxiv.org/html/2509.00351v1

[^1_80]: https://arxiv.org/html/2511.12410v1

[^1_81]: https://arxiv.org/abs/2103.15566

[^1_82]: https://www.sciencedirect.com/science/article/abs/pii/S0950705124013996

[^1_83]: https://www.sciencedirect.com/science/article/abs/pii/S0262885620302110

[^1_84]: https://www.ijcai.org/proceedings/2024/0111.pdf

[^1_85]: https://papers.neurips.cc/paper_files/paper/2020/file/bb7946e7d85c81a9e69fee1cea4a087c-Paper.pdf

[^1_86]: https://s-space.snu.ac.kr/handle/10371/175302

[^1_87]: https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2025.125120

[^1_88]: https://openaccess.thecvf.com/content/WACV2021/papers/Achituve_Self-Supervised_Learning_for_Domain_Adaptation_on_Point_Clouds_WACV_2021_paper.pdf

[^1_89]: https://cvpr.thecvf.com/virtual/2025/workshop/32364

[^1_90]: https://www.arxiv.org/abs/2509.09935

[^1_91]: https://www.sciencedirect.com/science/article/abs/pii/S095219762300578X

[^1_92]: https://arxiv.org/abs/1907.10915
