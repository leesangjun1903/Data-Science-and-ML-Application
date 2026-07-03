# SENTRY: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

SENTRY는 **비지도 도메인 적응(UDA)** 에서 동시에 발생하는 **공변량 이동(covariate shift)** 과 **레이블 분포 이동(label distribution shift, LDS)** 문제를 해결하기 위해 제안된 알고리즘입니다. 기존의 무조건적 엔트로피 최소화(entropy minimization) 기반 self-training은 초기화가 불량한 경우 오류가 누적되는 문제(error accumulation)가 발생합니다. SENTRY는 **일관성 위원회(committee of random image transformations)** 를 통해 타깃 인스턴스의 신뢰성을 판단하고, 신뢰 가능한 샘플에는 엔트로피 최소화를, 신뢰할 수 없는 샘플에는 엔트로피 최대화를 적용하는 **선택적 엔트로피 최적화** 전략을 제시합니다.

### 주요 기여

| # | 기여 항목 | 설명 |
|---|-----------|------|
| 1 | **신규 선택 기준** | 랜덤 이미지 변환 위원회 하의 예측 일관성으로 신뢰 가능 샘플 식별 |
| 2 | **선택적 엔트로피 최적화 목적함수** | 일관된 샘플: 엔트로피 최소화, 비일관 샘플: 엔트로피 최대화 |
| 3 | **의사 클래스 균형 샘플링** | 소스: 실제 레이블, 타깃: 의사레이블(pseudolabel) 기반 균형 샘플링 |
| 4 | **SOTA 달성** | 31개 도메인 이동 중 27개에서 최고 성능 달성 (DomainNet, OfficeHome, VisDA) |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 UDA 방법들은 두 가지 핵심 가정 아래 동작합니다:

$$P_S(y) = P_T(y) \quad \text{(레이블 분포 동일 가정)}$$

그러나 현실에서는 소스 도메인이 균형 분포를 가져도 타깃 도메인은 멱함수(power-law) 분포를 따르는 경우가 많습니다 (예: DomainNet, LVIS, MSCOCO). 이때 단순 분포 매칭(distribution matching)은 실패합니다.

또한, 기존 조건부 엔트로피 최소화(CEM) 기반 self-training의 문제는:

- 도메인 이동 하에서 모델 신뢰도(confidence)가 잘못 보정(miscalibrated)될 수 있음
- 초기 오정렬(misaligned) 샘플에 대한 CEM 적용 시, 오류를 강화하는 **오류 누적(error accumulation)** 발생

$$\mathcal{L}_{ENT} = \mathbb{E}_{\mathbf{x}_T \sim \mathcal{P}_T}\left[\mathcal{H}_{\Theta}(y|\mathbf{x}_T)\right] = \mathbb{E}_{\mathbf{x}_T \sim \mathcal{P}_T}\left[\sum_{c=1}^{C} -p_\Theta(y=c|\mathbf{x}_T)\log p_\Theta(y=c|\mathbf{x}_T)\right] \tag{2}$$

---

### 2.2 제안 방법 (수식 포함)

#### Step 1. 소스 도메인 사전학습

$$\mathcal{L}_{CE} = \mathbb{E}_{(\mathbf{x}_S, y_S) \sim \mathcal{P}_S}\left[\mathcal{L}_{CE}(h(\mathbf{x}_S), y_S)\right] \tag{1}$$

#### Step 2. 예측 일관성 기반 선택

타깃 인스턴스 $\mathbf{x}_T \sim \mathcal{P}_T$에 대해 $k$개의 변환 버전을 생성합니다:

$$\{a_1(\mathbf{x}_T), a_2(\mathbf{x}_T), \ldots, a_k(\mathbf{x}_T)\}$$

의사 레이블 $\hat{y}\_T = \arg\max\, p_\Theta(y|\mathbf{x}_T)$를 기준으로:

- **일관(Consistent)**: $\hat{y}\_T = \arg\max\, p_\Theta(y|a_i(\mathbf{x}_T))$인 변환 버전이 다수( $> k/2$ )인 경우
- **비일관(Inconsistent)**: 그 반대의 경우

#### Step 3. 선택적 엔트로피 최적화 (SENTRY 핵심)

$$\mathcal{L}_{\text{SENTRY}}(\mathbf{x}_T) = \begin{cases} +\mathcal{H}_\Theta(y|a_i(\mathbf{x}_T)) & \text{if consistent} \\ -\mathcal{H}_\Theta(y|a_j(\mathbf{x}_T)) & \text{if inconsistent} \end{cases} \tag{4}$$

여기서 $i$는 마지막 일관 변환 버전, $j$는 마지막 비일관 변환 버전의 인덱스입니다.

#### Step 4. 정보 엔트로피 손실 (LDS 대응)

마지막 $Q$개 타깃 인스턴스의 예측 분포 $q(\hat{y})$에 대해:

$$\mathcal{L}_{IE} = \mathbb{E}_{\mathbf{x}_T \sim \mathcal{P}_T}\left[\sum_{c=1}^{C} p_\Theta(y=c|\mathbf{x}_T)\log q(\hat{y}=c)\right] \tag{3}$$

이 손실은 예측의 다양성을 장려하여 다수 클래스로의 편향을 방지합니다.

#### Step 5. 전체 최적화 목적함수

$$\arg\min_\Theta \quad \mathbb{E}_{(\mathbf{x}_S, y_S) \overset{\text{bal}}{\sim} \mathcal{P}_S} \mathcal{L}_{CE} + \mathbb{E}_{\mathbf{x}_T \overset{\text{pbal}}{\sim} \mathcal{P}_T} \left[\lambda_{IE}\mathcal{L}_{IE} + \lambda_{\text{SENTRY}}\mathcal{L}_{\text{SENTRY}}\right] \tag{5}$$

여기서 $\lambda_{IE} = 0.1$, $\lambda_{\text{SENTRY}} = 1.0$이며, $\overset{\text{bal}}{\sim}$은 클래스 균형 샘플링, $\overset{\text{pbal}}{\sim}$은 의사 클래스 균형 샘플링을 의미합니다.

---

### 2.3 모델 구조

```
[소스 데이터 (레이블 균형 샘플링)]
         ↓
    ResNet-50 기반 CNN (h: X → Y)
    - 마지막 선형 레이어: C-way FC (Xavier 초기화, bias 없음)
    - L2-정규화 활성화
    - Softmax (temperature T=0.05)
         ↓
[타깃 데이터 (의사레이블 균형 샘플링)]
    ↓ RandAugment (N=3 변환, k=3 위원회)
    ↓ Consistency Checker (다수결 투표)
    ↓
  ┌──────────┐         ┌──────────────┐
  │ 일관(C)  │→ 엔트로피 최소화   │
  │ 비일관(IC)│→ 엔트로피 최대화   │
  └──────────┘         └──────────────┘
         ↓
   L_SENTRY + L_IE + L_CE 통합 최적화
```

**구현 세부사항:**
- 백본: ResNet-50 (DomainNet, OfficeHome, VisDA) / LeNet (DIGITS)
- 증강: RandAugment (14종 변환 중 N=3개 순차 적용, severity M=2.0)
- 위원회 크기: $k=3$
- 의사레이블 큐 크기: $Q=256$
- 프레임워크: PyTorch

---

### 2.4 성능 향상

| 벤치마크 | 비교 방법 | SENTRY | 개선폭 |
|----------|-----------|--------|--------|
| DomainNet (12 shifts 평균) | InstaPBM [22] (77.84%) | **81.39%** | +3.55% |
| OfficeHome RS-UT (6 shifts 평균) | MDD+I.A [19] (61.67%) | **65.25%** | +3.58% |
| OfficeHome Standard (12 shifts 평균) | InstaPBM [22] (69.2%) | **72.2%** | +2.7% |
| VisDA2017 | InstaPBM [22] (76.3%) | **76.7%** | +0.4% |
| SVHN→MNIST-LT (IF=100) | InstaPBM [22] (65.9%) | **85.6%** | +19.7% |

전체 31개 도메인 이동 중 27개에서 SOTA 달성.

---

### 2.5 한계

1. **이진 분류적 선택 전략**: 일관/비일관의 이분법적 분류는 연속적인 신뢰도 스펙트럼을 반영하지 못함
2. **의사레이블 의존성**: 클래스 균형 샘플링이 의사레이블의 정확도에 의존하며, 초기 의사레이블이 부정확할 경우 효과 감소
3. **하이퍼파라미터 민감도**: $\lambda_{IE}$, $\lambda_{\text{SENTRY}}$, $k$, $N$, $Q$ 등 여러 하이퍼파라미터 존재 (단, $k$, $N$에 대한 민감도는 낮음을 실험적으로 확인)
4. **분류 태스크 한정**: 세그멘테이션, 객체 탐지 등 다른 비전 태스크로의 직접 확장은 검증되지 않음
5. **일관성 체커의 정밀도 한계**: 약 75-80% 정밀도로, 20-25%의 오분류 가능성 존재

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 적응적 선택 전략의 일반화 효과

SENTRY의 핵심 일반화 메커니즘은 **증강 불변성 정규화(augmentation invariance regularization)** 입니다. 일관 샘플에 대해 원본 이미지가 아닌 **증강된 버전**에 대해 엔트로피를 최소화함으로써:

$$\mathcal{L}_{\text{SENTRY}}^{\text{consistent}} = +\mathcal{H}_\Theta(y|a_i(\mathbf{x}_T)) \quad \text{(증강 버전에 대한 엔트로피 최소화)}$$

이 접근은 두 가지 일반화 효과를 동시에 달성합니다:
- **과적합 방지**: 데이터 증강이 내재된 정규화 역할 수행
- **적응적 커버리지 확대**: 학습이 진행될수록 선택되는 인스턴스 비율이 점진적으로 증가

실험 결과, 학습 에포크 증가에 따라 엔트로피 최소화 선택 비율이 지속적으로 증가하고 최대화 비율이 감소하는 **적응적 자기조절 특성**이 관찰되었습니다.

### 3.2 레이블 분포 이동에 대한 강건성

의사 클래스 균형 샘플링과 $\mathcal{L}_{IE}$의 결합은 LDS 하에서의 일반화를 강화합니다:

- SVHN→MNIST-LT에서 imbalance factor(IF)=100인 극단적 불균형 상황에서도 $85.6\%$ 달성 (2위 대비 +19.7%)
- 이는 단순 도메인 적응을 넘어 **클래스 불균형 분포에도 견고한 일반화 능력**을 보여줌

### 3.3 엔트로피 최대화의 이론적 근거

논문은 이진 분류 케이스에서 엔트로피 최대화의 효과를 이론적으로 분석합니다. 오분류 샘플($y=0$, $0.5 \leq p < 1$)에 대해:

$$\mathcal{L}_{EM} = p\log p + (1-p)\log(1-p)$$

$$\mathcal{L}_{BCE} = -[y\log(p) + (1-y)\log(1-p)]$$

$$\nabla_p \mathcal{L}_{EM} = \log\left(\frac{p}{1-p}\right), \quad \nabla_p \mathcal{L}_{BCE} = \frac{1-y}{1-p} \quad (y=0)$$

두 그래디언트의 방향이 강하게 상관되어, 잘못 정렬된 샘플에 대한 엔트로피 최대화가 **실제 레이블 기준의 지도 학습과 유사한 효과**를 낸다는 것을 보여줍니다. 이는 일반화 관점에서 비지도 신호만으로도 지도 신호에 근접한 학습이 가능함을 시사합니다.

### 3.4 일관성 체커의 정밀도와 일반화

실험 분석(Figure 4)에서 위원회 일관성 전략의 정밀도가 에포크 진행에 따라:
- 정확 샘플 식별 정밀도: **75~80% 수준에서 안정적 유지**
- 클래스별로도 편향 없이 모든 클래스의 선택 비율이 증가

이는 **클래스 특이적 편향 없이 균형 잡힌 일반화**가 이루어짐을 보여줍니다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

#### (1) 선택적 자기학습(Selective Self-Training) 패러다임 확립
SENTRY는 "모든 샘플을 동등하게 사용하는" 기존 패러다임에서 벗어나, **샘플 신뢰도 기반 선택적 최적화**라는 새로운 패러다임을 제시합니다. 이는 이후 다양한 UDA 연구에서 중요한 설계 원칙으로 채택될 것으로 예상됩니다.

#### (2) 일관성 기반 신뢰도 평가의 보편화
예측 일관성을 신뢰도 지표로 사용하는 접근은 캘리브레이션 문제를 우회하며, 반지도 학습(semi-supervised learning)이나 지속적 학습(continual learning)에도 적용 가능합니다.

#### (3) LDS + 공변량 이동 통합 벤치마크의 필요성 환기
논문은 LDS를 명시적으로 고려한 벤치마크의 필요성을 강조하며, 향후 UDA 연구에서 LDS 조건을 표준 평가 항목으로 포함하는 방향을 촉진합니다.

#### (4) 엔트로피 양방향 최적화 아이디어
단순 엔트로피 최소화를 넘어 **최소화 + 최대화의 비대칭 조합**은 이후 능동 학습(active learning), 이상 탐지(anomaly detection), 오픈셋 인식(open-set recognition) 연구에 영감을 줄 수 있습니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | 핵심 아이디어 | SENTRY와의 관계 |
|------|------|---------------|----------------|
| **NRC** (Yang et al., NIPS 2021) | 이웃 클러스터링 기반 UDA | 타깃 샘플 간 구조적 관계 활용 | SENTRY의 일관성 선택과 상보적 |
| **SHOT** (Liang et al., ICML 2020) | 소스 없는 DA, 정보 최대화 + 의사레이블 | 소스 데이터 없이 타깃만으로 적응 | SENTRY는 소스 데이터 필요, SHOT은 불필요 |
| **T3A** (Iwasawa & Matsuo, NIPS 2021) | 테스트 시간 분류기 조정 | 프로토타입 기반 테스트 타임 적응 | SENTRY는 학습 시 적응, T3A는 추론 시 적응 |
| **FixMatch** (Sohn et al., 2020) | 반지도 학습의 일관성 + 신뢰도 임계값 | 고신뢰 샘플만 self-training | SENTRY는 비일관 샘플도 활용(최대화)하는 점이 차별화 |
| **DAPL** (Ge et al., 2022) | CLIP 기반 프롬프트 학습으로 UDA | 대규모 사전학습 모델 활용 | SENTRY는 순수 CNN 기반, 비전-언어 모델 미활용 |

**핵심 비교 포인트:**
- SENTRY는 **비일관 샘플에 대한 엔트로피 최대화**라는 독특한 전략으로 FixMatch 등 단순 신뢰도 필터링 방법과 차별화됨
- 그러나 **ViT/CLIP 등 대형 사전학습 모델** 기반 최신 방법들과의 성능 비교는 논문에서 다루지 않음

---

### 4.3 앞으로 연구 시 고려할 점

#### (1) 대규모 사전학습 모델과의 통합
ResNet-50 기반에서 ViT, CLIP, DINOv2 등 대형 모델로의 확장 시, 일관성 위원회 전략이 여전히 유효한지 검증 필요. 특히 **대형 모델은 이미 강한 표현력**을 가지므로 일관성 기준 자체가 달라질 수 있습니다.

#### (2) 연속적 신뢰도 점수 도입
현재 이진 분류(일관/비일관) 대신 연속적 신뢰도 점수를 사용하면:

$$w(\mathbf{x}_T) = \frac{|\mathcal{C}|}{k} \in [0, 1] \quad \text{(일관 변환 비율)}$$

이를 가중치로 활용하는 **부드러운 선택적 엔트로피 최적화**가 더 세밀한 제어를 가능하게 할 수 있습니다.

#### (3) 세그멘테이션 / 탐지 태스크로의 확장
픽셀 수준 또는 객체 수준 일관성 체크로 확장 시, 계산 비용 증가 문제를 어떻게 처리할지 설계 필요.

#### (4) 소스 데이터 없는(source-free) UDA 환경
SHOT, AaD 등 소스 없는 DA 트렌드에 맞춰 소스 데이터 없이도 SENTRY 원칙을 적용할 수 있는 방법 모색이 필요합니다.

#### (5) 이론적 수렴 보장
선택적 엔트로피 최적화의 수렴 조건과 최적화 안정성에 대한 이론적 분석이 부재하므로, 향후 PAC-Bayes 또는 정보 이론적 관점에서의 분석이 요구됩니다.

#### (6) 동적 하이퍼파라미터 조정
$\lambda_{\text{SENTRY}}$, $\lambda_{IE}$를 학습 진행에 따라 동적으로 조정하는 커리큘럼 학습 전략이 성능 향상에 기여할 수 있습니다.

---

## 참고자료

- **논문 원문**: Viraj Prabhu, Shivam Khare, Deeksha Kartik, Judy Hoffman. "SENTRY: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation." *ICCV 2021*, pp. 8558–8567.
- **논문 PDF**: 제공된 첨부 파일 (ICCV 2021 Open Access)
- **공식 코드**: https://github.com/virajprabhu/SENTRY
- **참조 논문**:
  - Li et al. (2020). "Rethinking distributional matching based domain adaptation." arXiv:2006.13352 (InstaPBM)
  - Tan et al. (2020). "Class-imbalanced domain adaptation: An empirical odyssey." ECCV Workshops (COAL)
  - Sohn et al. (2020). "FixMatch: Simplifying semi-supervised learning with consistency and confidence." arXiv:2001.07685
  - Cubuk et al. (2020). "RandAugment: Practical automated data augmentation with a reduced search space." CVPR Workshops
  - Grandvalet & Bengio (2005). "Semi-supervised learning by entropy minimization." CAP
  - Bahat et al. (2019). "Natural and adversarial error detection using invariance to image transformations." arXiv:1902.00236
