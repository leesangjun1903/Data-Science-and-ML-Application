# One-Step Generalization Ratio Guided Optimization for Domain Generalization

### 1. 핵심 주장 및 주요 기여 요약

본 논문은 **GENIE (Generalization-ENhancing Iterative Equalizer)**라는 새로운 도메인 일반화(Domain Generalization, DG) 최적화 알고리즘을 제안합니다. GENIE의 핵심 주장은 기존 최적화 방법들이 **매개변수 불균형(parameter imbalance)** 문제를 간과하고 있다는 것입니다. 구체적으로, 일부 매개변수가 최적화 과정을 지배하면서 도메인 특정적 특성(spurious correlations)에 과적합되는 현상을 해결합니다.[1]

**주요 기여는 다음 세 가지입니다:**

1. **OSGR 기반 최적화 원칙 도입**: 매개변수별 손실 감소 기여도와 그래디언트 정렬을 정량화하는 One-Step Generalization Ratio(OSGR)를 최적화기에 통합했습니다.[1]

2. **동적 사전조건화(Dynamic Preconditioning)**: OSGR을 균등하게 유지하는 사전조건화 인수(preconditioning factor)를 통해 모든 매개변수가 공평하게 기여하도록 보장합니다.[1]

3. **도메인-무관 최적화기**: 아키텍처 수정이나 데이터 조작 없이 기존 DG 알고리즘과 무결합적으로 통합 가능합니다.[1]

***

### 2. 해결하는 문제 및 제안 방법

#### 문제 정의

도메인 일반화의 핵심 도전은 다음과 같습니다:[1]

- **도메인 시프트 문제**: 훈련 도메인의 분포와 테스트 도메인의 분포가 상이할 때 성능 저하
- **허위 상관관계(Spurious Correlations)**: 모델이 도메인-특정적 특성에 과적합되어 일반화 불가능
- **기울기 정렬의 한계**: 기존 방법들(예: GRAD-MATCH)은 지배적 방향으로만 기울기를 정렬하여 여전히 허위 상관관계를 강화함

#### 제안 방법: 수식 포함

**2.1 One-Step Generalization Ratio (OSGR) 기본**

OSGR은 단일 기울기 업데이트 후 테스트 손실과 훈련 손실의 감소 비율을 측정합니다:[1]

$$R(Z, n) = \frac{\mathbb{E}_{D,D' \sim \mathcal{Z}^n}[\Delta L_{D'}]}{\mathbb{E}_{D \sim \mathcal{Z}^n}[\Delta L_D]}$$

여기서 $\Delta L_{D'}$와 $\Delta L_D$는 테스트와 훈련 데이터에서의 손실 변화를 나타냅니다. 더 높은 OSGR은 더 나은 일반화를 의미합니다.[1]

**Theorem 3.1**로부터 OSGR을 매개변수별 통계와 연결하면:[1]

$$R(Z, n) = 1 - \frac{1}{n}\sum_{j \in \mathcal{J}} \frac{\mathbb{E}_{D \sim \mathcal{Z}^n}[g_j^2]}{\sum_{j' \in \mathcal{J}} \mathbb{E}_{D \sim \mathcal{Z}^n}[g_{j'}^2]} \cdot \frac{1}{r_j + \frac{1}{n}}$$

여기서 $r_j = \frac{g_j^2}{\rho_j^2}$는 Gradient Signal-to-Noise Ratio (GSNR)입니다.[1]

**2.2 GENIE의 핵심: 사전조건화 인수**

GENIE는 매개변수 $j$에 대해 다음 사전조건화 인수를 제안합니다:[1]

$$p_j = \frac{1}{\mathbb{E}_{D \sim \mathcal{Z}^n}[\sqrt{g_j^2}] \cdot \left(r_j + \frac{1}{n}\right)}$$

이를 통해 얻어지는 OSGR은:[1]

$$R'(Z, n) = 1 - \frac{1}{n}\mathbb{E}_{j \in \mathcal{J}}\left(r_j + \frac{1}{n}\right)$$

**Key Insight**: 사전조건화 전, 큰 $g_j^2$이지만 낮은 GSNR을 가진 매개변수가 OSGR 식의 감산 항을 과도하게 팽창시킬 수 있습니다. 제안된 사전조건화는 이를 완화하여 OSGR을 개선합니다.[1]

**2.3 수정된 Adam 형태의 GENIE 알고리즘**

Algorithm 1: GENIE 업데이트 규칙[1]

$$m_t \leftarrow \beta m_{t-1} + (1-\beta)g_t$$

$$v_t \leftarrow \beta v_{t-1} + (1-\beta)g_t^2$$

$$\sigma_t^2 = v_t - m_t^2 \quad \text{(분산 추정)}$$

$$r_j = \tanh\left(\frac{1}{\sigma_t^2}\right) \cdot m_t^2$$

$$\hat{g}_t \leftarrow \frac{m_t}{1-\beta^t} \cdot \frac{1}{v_t} \cdot r_t \quad \text{(사전조건화된 기울기)}$$

**노이즈 주입**:
$$\text{Noise}_t \leftarrow \xi_t \left(1 - \tanh\left(\frac{1}{\sigma_t^2}\right)\right), \quad \xi_t \sim \mathcal{N}(0, \sigma^2)$$

**무작위 마스크**:
$$M_j \sim \text{Bernoulli}(p), \quad \hat{g}_t \leftarrow (\hat{g}_t + \text{Noise}_t) \odot M$$

**매개변수 업데이트**:
$$\theta_{t+1} \leftarrow \theta_t - \alpha \hat{g}_t$$

***

### 3. 모델 구조 및 이론적 기초

#### 3.1 모델 아키텍처

GENIE는 **최적화기 중심 설계**이므로 네트워크 아키텍처 수정이 불필요합니다. ResNet-50(ImageNet 사전학습)이 주된 백본으로 사용되었습니다.[1]

**3단계 구조**:
1. **사전조건화(Preconditioning)**: 매개변수별 GSNR을 기반으로 기울기 스케일링
2. **노이즈 주입(Noise Injection)**: 저분산 매개변수에 더 강한 노이즈 주입으로 탐색 강화
3. **무작위 마스크(Random Mask)**: 베르누이 분포 기반 드롭아웃으로 과적합 안정화[1]

#### 3.2 이론적 기초

**3.2.1 Convergence Analysis (Theorem 3.11)**

비볼록 설정에서 GENIE의 수렴 속도:[1]

$$\mathbb{E}[\|\nabla L(\theta)\|^2] \leq O\left(\frac{1}{P_l}\left(1 + \frac{G \cdot S_u^2}{2}\right) \cdot \frac{1}{\sqrt{\hat{T}}}\right)$$

여기서 $P_l$은 사전조건화 값의 하한, $G$는 기울기 l2 노름 상한, $S_u$는 분산의 하한입니다.[1]

**결론**: GENIE는 SGD의 수렴률 $O(T^{-1/2})$를 유지하면서 더 견고한 일반화를 달성합니다.[1]

**3.2.2 PAC-Bayes해석 (Theorem 3.6)**

GENIE의 사전조건화는 PAC-Bayes 일반화 경계의 KL 발산 항 최소화와 일치합니다:[1]

$$[\nabla_{\theta_t} \text{KL}(\tilde{p}\|\pi)]_j = \frac{1}{\mathbb{E}[g_j^2]} \cdot \frac{\mathbb{E}[g_j]^2}{\rho_j^2} \cdot g_{j,t}$$

이는 SAM(Sharpness-Aware Minimization)과 달리 **샤프니스와 일반화 경계를 동시에 최소화**함을 의미합니다.[1]

**3.2.3 OSGR 비교 (Corollary 3.3)**

$$R_{\text{GENIE}} \geq R_{\text{SGD}} \approx R_{\text{Adam}}$$

균일 가중 가정 하에서 Jensen 부등식을 적용하면 GENIE가 더 높은 OSGR을 달성함을 증명할 수 있습니다.[1]

***

### 4. 성능 향상 및 한계

#### 4.1 성능 향상

**4.1.1 최적화기 비교 (표 2)**[1]

| 데이터셋 | PACS | VLCS | OfficeHome | TerraIncognita | DomainNet | 평균 |
|---------|------|------|-----------|---------------|-----------|------|
| Adam | 84.2 | 77.3 | 67.6 | 44.4 | 43.0 | 63.3 |
| SAM | 85.3 | 78.2 | 68.0 | 45.7 | 43.4 | 64.1 |
| FAD | 88.2 | 78.9 | 69.2 | 45.7 | 44.4 | 65.3 |
| **GENIE** | **87.8** | **80.7** | **69.7** | **52.0** | **44.1** | **66.9** |

**핵심 개선사항**:[1]
- Adam 대비 5.69% 향상
- SGD 대비 6.36% 향상
- SAM 대비 4.37% 향상
- TerraIncognita에서 특히 강력 (52.0% vs FAD 45.7%)

**4.1.2 계산 효율성 (표 3)**[1]

GENIE는 SAM 대비 **1.3배 더 빠른 학습** (SAM은 단계별 이중 기울기 계산 필요)[1]

**4.1.3 기존 DG 알고리즘과의 통합 (표 5)**[1]

| 알고리즘 | Adam | GENIE |
|---------|------|-------|
| CORAL | 69.3 | 71.9 |
| RSC | 68.2 | 71.4 |
| ERM (기본) | 68.9 | 72.6 |

**무결합적 개선**: 아키텍처 수정 없이 평균 2-3% 성능 향상[1]

**4.1.4 단일 도메인 일반화 (SDG, 표 4)**[1]

| 방법 | PACS | VLCS | OfficeHome | TerraIncognita | 평균 |
|-----|------|------|-----------|---------------|------|
| Adam | 64.3 | 56.2 | 50.7 | 33.5 | 51.2 |
| GENIE | 69.5 | 69.9 | 58.6 | 36.0 | 58.5 |

향상도: **7.3 percentage points**[1]

#### 4.2 일반화 성능 향상 가능성 분석

**4.2.1 OSGR 추적 분석 (Figure 4)**[1]

논문의 실험에서 VLCS 데이터셋에 대해:
- **GENIE의 OSGR**: 1.0에 가까움 (가장 안정적)
- **SAM의 OSGR**: 더 낮은 값 (불안정)
- **해석**: 높은 OSGR은 테스트 손실과 훈련 손실의 감소가 균형잡혀 있음을 의미

**4.2.2 손실 경면 시각화 (Figure 5)**[1]

FashionMNIST의 시뮬레이션된 손실 경면에서:
- SGD/Adam: 가파른 방향으로 빠르게 수렴 (조기 과적합)
- **GENIE**: 더 평탄한 극소값으로 수렴 (더 일반화 가능한 솔루션)[1]

**4.2.3 특성 공간 시각화 (Figure 3)**[1]

PACS 데이터셋 (Sketch 테스트):
- GENIE는 모든 도메인에서 더 명확한 클래스 분리 달성
- 도메인-불변 특성 학습의 효과적성 입증

#### 4.3 한계 및 제약사항

**4.3.1 이론적 한계**

1. **n 항 무시**: 실제 구현에서 $1/n$ 항을 무시하는데, 소규모 배치에서는 부정확성 야기 가능[1]

2. **Tanh 함수 정규화**: 분산 폭발 방지를 위해 $\tanh(1/\sigma_t^2)$를 사용하지만, 이로 인한 이론-실제 간극 존재[1]

3. **균일 가중 가정**: Jensen 부등식 기반 OSGR 비교는 $W_j = 1/|J|$ 가정 하에서만 엄격함[1]

**4.3.2 경험적 한계**

1. **초매개변수 민감성**: 드롭아웃 확률 $p$와 계수 $\beta$에 민감할 수 있음 (Figure 2에서 성능 변동 관찰)[1]

2. **DomainNet 성능**: 44.1%로 다른 데이터셋 대비 상대적으로 낮은 성능 (6개 도메인의 복잡성)[1]

3. **다중 소스 의존성**: 5개 표준 DG 벤치마크에서만 평가 - 더 다양한 영역(medical imaging, 산업 응용)에서의 검증 부족[1]

4. **Ablation 결과의 혼합**: Table 6에서 노이즈 주입과 마스크가 항상 성능을 개선하지는 않음 (Cartoon, Photo 도메인에서 감소)

***

### 5. 모델의 일반화 성능 향상 가능성

#### 5.1 핵심 메커니즘

GENIE가 일반화를 개선하는 방식은 **매개변수별 기여도의 균등화**입니다:[1]

1. **높은 GSNR 매개변수 억제**: 과도하게 예측력 높은 매개변수의 영향력 감소
2. **낮은 GSNR 매개변수 부스팅**: 노이즈 주입으로 탐색 강화
3. **균형잡힌 학습**: 모든 매개변수가 공평하게 기여하도록 강제[1]

#### 5.2 도메인 불변 특성 학습

**UMAP 시각화 분석 (Figure 3)**:[1]

- **GENIE 학습 후**: 클래스별로 명확한 군집 형성, 도메인 경계 무시
- **기존 방법**: 도메인별 분리 경향, Sketch 도메인에서 특히 약함

**의미**: GENIE는 도메인-특정 스타일보다 **의미론적 구조**에 집중하도록 유도

#### 5.3 평탄한 극소값 도달

**이론적 근거 (Theorem 3.6, PAC-Bayes)**:[1]

$$\text{일반화 경계} = \underbrace{\mathbb{E}_{\theta \sim \tilde{p}}[L(\theta)]}_{\text{경험적 손실}} + \underbrace{\frac{1}{2}KL(\tilde{p}\|\pi)}_{\text{복잡도 항}}$$

GENIE는 경험적 손실 안정성 ($q_j = \mathbb{E}[g]^2/\mathbb{E}[g^2]$)과 KL 발산 최소화 ($1/\rho_j^2$ 항)를 동시에 달성[1]

#### 5.4 학습 동역학의 개선

**손실 경면 시뮬레이션**:
- 빠른 수렴이 반드시 좋은 것이 아님
- 일반화 가능한 특성은 훈련 후기에 학습됨
- **GENIE의 완만한 경로**: 평탄한 극소값 도달로 후기 일반화 특성 학습 촉진[1]

***

### 6. 2020년 이후 관련 최신 연구 비교 분석

#### 6.1 주요 관련 연구 분류

**6.1.1 그래디언트 기반 접근**

| 방법 | 연도 | 핵심 아이디어 | 한계 |
|-----|------|-----------|------|
| GRAD-MATCH | 2022 | 도메인 간 기울기 정렬 | 지배적 방향만 고려, 허위 상관관계 강화 가능 |
| GSNR Dropout | 2023 | 높은 GSNR 매개변수 드롭 | 기울기 정렬 미조정, 불균형한 업데이트 |
| **GENIE** | 2025 | OSGR 기반 동적 균등화 | 모든 매개변수 공평 기여 보장 |

**6.1.2 샤프니스-인식 최적화 (Sharpness-Aware Optimizers)**

| 방법 | 연도 | 아이디어 | 성능 (PACS) |
|-----|------|--------|----------|
| SAM | 2020 | 급격한 극소값 회피 | 85.3% |
| GAM | 2023 | 기울기 노름 정규화 | 86.1% |
| FAD | 2023 | 평탄성 최적화 | 88.2% |
| **GENIE** | 2025 | OSGR 균등화 + 평탄성 | **87.8%** |

**평가**: GENIE는 FAD와 경쟁적이지만, **일반성과 무결합성에서 우수** (architecture modification 불필요)[1]

**6.1.3 데이터 증강 기반 접근 (2021-2024)**

| 방법 | 원리 | 적용 범위 |
|-----|------|---------|
| MixStyle | 인스턴스 정규화 통계 혼합 | 이미지 분류만 |
| MixDomain | 도메인 간 샘플 보간 | 다중 소스 요구 |
| SGDG | 시맨틱 안내 확산 생성 | 계산 비용 높음 |
| **GENIE** | 최적화기 중심 (도메인-무관) | 모든 DG/SDG 방법 지원 |

**6.1.4 페더레이션 도메인 일반화 (2023-2024)**

최근 연구 방향: 프라이버시 보존 도메인 일반화[2-6]
- RFDG (2024): 강화학습 기반 샘플 가중치
- FedDG-GA (2023): 분산 집계 가중치 조정

**GENIE와의 관계**: 페더레이션 설정의 로컬 최적화기로 활용 가능성 있음

#### 6.2 이론적 진화

**2020-2024 이론적 발전**:

1. **GSNR의 일반화 관계** (Liu et al., 2020 → 현재)
   - 초기: GSNR과 일반화의 정성적 연관
   - 현재: OSGR을 통한 정량적 매개변수별 분석 (GENIE)[1]

2. **PAC-Bayes 적용 확대**
   - SAM (2020): 샤프니스 중심
   - GENIE (2025): 샤프니스 + KL 발산 동시 최소화[1]

3. **수렴 분석 정교화**
   - 이전: 기본 SGD 수렴률만 고려
   - GENIE: GSNR 하한을 포함한 세밀한 수렴률 분석[1]

#### 6.3 실험적 벤치마크 진화

**DomainBed 표준화 (Gulrajani & Lopez-Paz, 2021) 이후의 변화**:

| 시기 | 성과 | 도전 |
|-----|------|------|
| 2021 | ERM이 많은 DG 방법 능가 | 공정한 비교 어려움 |
| 2022-2023 | 최적화 중요성 인식 (SAM, FAD) | 하이퍼파라미터 민감성 |
| 2024-2025 | **매개변수 불균형** 인식 | OSGR 메트릭 도입 (GENIE) |

***

### 7. 앞으로의 연구 영향 및 고려사항

#### 7.1 학술적 영향

**7.1.1 이론적 기여**

1. **OSGR의 최적화기 설계 원리화**: OSGR을 단순 진단 지표에서 **최적화기 설계의 핵심 원리**로 전환
   - 영향: 향후 최적화기 설계의 이론적 기준 제시

2. **매개변수 불균형의 정식화**: DG 문제의 새로운 관점 제시
   - 기존 연구: 도메인 정렬, 기울기 정렬에 집중
   - **GENIE**: 매개변수별 기여도 균등화[1]

3. **통합적 일반화 분석**: 수렴성과 일반화를 동시에 분석하는 프레임워크 제시[1]

**7.1.2 방법론적 영향**

- **사전조건화의 새로운 활용**: 일반화 개선을 위한 사전조건화 설계 (기존: 수렴성만 고려)
- **PAC-Bayes와 최적화의 연결**: 보이지 않는 영역의 이론과 실제 최적화 알고리즘 연결[1]

#### 7.2 실무적 활용 가능성

**7.2.1 직접 적용 분야**

1. **컴퓨터 비전**
   - 의료 이미지 분석 (도메인: 스캐너, 프로토콜, 환자군)
   - 자율주행 (도메인: 날씨, 조명, 카메라)
   - 원격 감지 (도메인: 계절, 시간, 위치)

2. **자연어 처리**
   - 감정 분석 (도메인: 소셜미디어 플랫폼, 언어, 시간)
   - 개체명 인식 (도메인: 업종별 텍스트)

3. **산업 응용**
   - 기계 결함 진단 (도메인: 장비 종류, 노후도)
   - 품질 관리 (도메인: 생산 배치, 환경 조건)

**7.2.2 프레임워크로의 통합 경로**

- PyTorch Lightning, Hugging Face Transformers의 최적화기 추가 후보[1]
- DomainBed 벤치마크의 기본 최적화기로 채택 가능성[1]

#### 7.3 향후 연구 시 고려할 점

**7.3.1 이론적 확장 방향**

1. **강화된 수렴 분석**
   - 현재: 비볼록 설정의 기본 수렴률
   - **필요**: 특정 DG 손실 함수(예: 대조 학습)에 대한 세밀한 분석

2. **OSGR 한계 규명**
   - $1/n$ 항의 영향 정량화
   - 소규모 배치 시나리오에서의 이론적 보장

3. **다중 도메인 시나리오에서의 이론**
   - 현재: 일반적 설정
   - **필요**: 도메인 수에 따른 수렴 속도 의존성 분석

**7.3.2 경험적 개선 방향**

1. **초매개변수 적응화**
   - 현재: 고정된 $p$ (드롭아웃 확률), $\beta$ (지수 이동 평균)
   - **제안**: 데이터셋별 자동 선택 메커니즘

2. **다양한 아키텍처 평가**
   - 현재: ResNet-50만 주로 평가
   - **필요**: Vision Transformer, EfficientNet 등에서의 성능

3. **비전 도메인 외 적용**
   - 논문: 이미지 분류에 집중
   - **필요**: 시계열, 그래프, 텍스트 도메인에서의 평가

**7.3.3 안정성 및 견고성**

1. **노이즈 주입의 영향 분석**
   - Table 6 Ablation에서 일부 도메인에서 성능 저하 (Cartoon, Photo)
   - **개선**: 도메인 특성에 적응하는 노이즈 주입 전략

2. **매개변수 초기화의 민감성**
   - 현재: 표준 초기화만 다룸
   - **필요**: 극단적 초기화 시나리오에서의 안정성 분석

3. **클래스 불균형 설정**
   - 현재: 균형잡힌 데이터셋 가정
   - **필요**: 롱테일 분포에서의 성능 (SAMALTDG 참고)[2]

**7.3.4 상호작용 효과 연구**

1. **다른 DG 방법과의 상호작용**
   - Table 5에서 CORAL, RSC와 통합하지만, 더 깊은 분석 필요
   - **질문**: 어떤 DG 방법과 가장 잘 맞는가?

2. **정규화 기법과의 조합**
   - 배치 정규화, 계층 정규화와의 상호작용
   - 드롭아웃 (GENIE 내장)과의 중복성

#### 7.4 해결되지 않은 문제

**7.4.1 이론적 미해결 문제**

1. **OSGR과 일반화 간극**: 실제로 높은 OSGR이 모든 도메인에서 일반화를 보장하는가?
   - 반례: SAM이 높은 OSGR을 가질 수 없는 이유의 근본적 이해

2. **매개변수 불균형의 필요성**: 불균형이 정말 해롭기만 한가?
   - 일부 작업에서는 특정 매개변수의 "우선권"이 유익할 수 있음[1]

3. **도메인 시프트 유형에 따른 효과 차이**: 
   - Figure 3 결과는 스타일 시프트에 집중
   - 의미론적 시프트(semantic shift)에서의 성능은?

**7.4.2 실무적 미해결 문제**

1. **계산 오버헤드의 숨겨진 비용**: 
   - Table 3에서 시간만 비교하지만, 메모리 사용량은?
   - 대규모 모델(예: ViT-L)에서의 확장성?

2. **온라인 학습 설정에서의 적용 가능성**:
   - 현재: 배치 학습만 고려
   - 스트리밍 도메인 전환 시나리오는?

3. **설명 가능성(Interpretability)**:
   - GENIE가 어떤 특성을 학습하는지 가시화 방법 부재[1]

***

### 8. 결론 및 종합 평가

GENIE는 **매개변수 불균형 문제의 인식과 해결**이라는 도메인 일반화의 새로운 관점을 제시합니다. One-Step Generalization Ratio를 최적화 원리로 체계화함으로써 이론적 엄밀성과 경험적 우수성을 동시에 달성했습니다.[1]

**핵심 강점**:
- 무결합적 설계로 기존 DG 방법과 자유롭게 결합
- 강고한 이론적 기초 (수렴성, 일반화, PAC-Bayes)
- 5개 표준 벤치마크에서 일관된 성능 개선[1]

**개선의 여지**:
- 더 다양한 도메인(의료, 산업)에서의 검증 필요
- 비전 외 모달리티(텍스트, 시계열)에서의 적용성
- 하이퍼파라미터 자동 선택 메커니즘 개발

GENIE는 2025년 ICML에서 채택됨으로써 최적화 관점의 도메인 일반화 연구에 새로운 방향을 제시하는 의미 있는 기여입니다.[3][1]

***

### 참고 자료 색인

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fef28cd4-2600-4c02-b434-2cc2b05d79e5/10381_One_Step_Generalization.pdf)
[2](https://ojs.aaai.org/index.php/AAAI/article/view/29431)
[3](https://proceedings.mlr.press/v267/cho25c.html)
[4](https://ieeexplore.ieee.org/document/10678174/)
[5](https://ieeexplore.ieee.org/document/10692575/)
[6](https://ieeexplore.ieee.org/document/10646594/)
[7](https://link.springer.com/10.1007/s00521-024-10353-5)
[8](https://ieeexplore.ieee.org/document/10202578/)
[9](https://www.mdpi.com/1424-8220/23/14/6511)
[10](https://arxiv.org/abs/2405.01022)
[11](https://ieeexplore.ieee.org/document/10203192/)
[12](https://ieeexplore.ieee.org/document/10073578/)
[13](http://arxiv.org/pdf/2110.04545.pdf)
[14](https://arxiv.org/pdf/2102.08604.pdf)
[15](http://arxiv.org/pdf/2302.06874.pdf)
[16](https://arxiv.org/pdf/2302.02350.pdf)
[17](https://arxiv.org/pdf/2401.08464.pdf)
[18](https://arxiv.org/pdf/2211.04393.pdf)
[19](https://arxiv.org/pdf/2206.00047.pdf)
[20](http://arxiv.org/pdf/2307.13492.pdf)
[21](https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2025.125120)
[22](https://pmc.ncbi.nlm.nih.gov/articles/PMC7643753/)
[23](https://icml.cc/virtual/2025/poster/45152)
[24](https://www.sciencedirect.com/science/article/pii/S0031320323007264)
[25](https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Unleashing_the_Power_of_Gradient_Signal-to-Noise_Ratio_for_Zero-Shot_NAS_ICCV_2023_paper.pdf)
[26](https://neurips.cc/virtual/2022/65650)
[27](https://arxiv.org/pdf/2103.03097.pdf)
[28](http://proceedings.mlr.press/v139/rudner21a/rudner21a.pdf)
[29](https://hongyanz.github.io/publications/AAAI_Lost.pdf)
[30](https://paperswithcode.com/task/domain-generalization/codeless)
[31](https://arxiv.org/html/2511.16979v2)
[32](https://openaccess.thecvf.com/content/ICCV2023/papers/Michalkiewicz_Domain_Generalization_Guided_by_Gradient_Signal_to_Noise_Ratio_of_ICCV_2023_paper.pdf)
[33](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Flatness-Aware_Minimization_for_Domain_Generalization_ICCV_2023_paper.pdf)
[34](https://arxiv.org/html/2509.00351v1)
[35](https://arxiv.org/html/2310.07361)
[36](https://arxiv.org/abs/2503.23430)
[37](https://arxiv.org/html/2512.10818v1)
[38](https://arxiv.org/pdf/2001.07384.pdf)
[39](https://arxiv.org/html/2508.09418v1)
[40](https://arxiv.org/html/2508.21769v2)
[41](https://www.sciencedirect.com/science/article/abs/pii/S0031320323004351)
[42](https://pmc.ncbi.nlm.nih.gov/articles/PMC11230655/)
