# Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponent-Guided Optimization

### 1. 논문의 핵심 주장 및 기여도 요약
이 논문은 **단일 도메인 일반화(Single Domain Generalization, SDG)** 문제를 해결하기 위해 **동역학계 이론(Dynamical Systems Theory)**의 라이아푸노프 지수(Lyapunov Exponent, LE) 개념을 신경망 최적화에 처음으로 도입한 혁신적인 연구입니다.[1]

**핵심 주장의 논리:**
- 기존 적대적 데이터 증강 방법들이 국소적 섭동(localized perturbations)에만 집중하여 매개변수 공간의 전역 구조를 포착하지 못함
- 신경망 학습을 이산 시간 동역학계로 모델링하고, LE를 이용해 모델이 **카오스의 경계(edge of chaos)** 근처에서 학습되도록 유도하면 안정성과 적응성의 최적 균형을 달성 가능
- LE를 피드백 메커니즘으로 삼아 학습률을 동적으로 조정함으로써 더 광범위한 매개변수 공간 탐색과 도메인 일반화 성능 향상 달성

**주요 기여:**
- 신경망 훈련을 동역학계로 개념화하고 LE로 카오스 경계 근처 여부를 정량 측정하는 이론적 프레임워크 구축[1]
- LEAwareSGD라는 새로운 최적화기 제안으로, LE 기반 학습률 동적 조정이 일반화 성능 향상[1]
- PACS, OfficeHome, DomainNet 등 표준 벤치마크에서 기존 최고 성능 방법 대비 최대 9.47% 성능 향상 달성[1]

***

### 2. 해결하는 문제와 제안 방법의 수식 및 모델 구조
#### 2.1 문제 정의

**단일 도메인 일반화(SDG)의 도전:**
- 단 하나의 소스 도메인 데이터로만 학습해야 함
- 목표 도메인은 훈련 중에 알 수 없음 → 심각한 도메인 시프트 대응 필요
- 훈련 데이터 다양성 부족으로 인한 과적합 위험[1]

#### 2.2 라이아푸노프 지수 기반 모델 섭동 이론

**기본 구조:**
표준 경사 하강법 업데이트:
$$\theta_{t+1} = \theta_t - \eta_t \nabla L(\theta_t)$$

여기서 $\eta_t$는 학습률, $L(\theta_t)$는 손실함수입니다.[1]

**섭동 진화:**
초기 섭동 $\delta\theta_0$를 도입하고, 섭동된 매개변수 $\tilde{\theta}_t := \theta_t + \delta\theta_t$에 대해:

$$\tilde{\theta}_{t+1} = \tilde{\theta}_t - \eta_t \nabla L(\tilde{\theta}_t)$$

섭동의 업데이트:

$$\delta\theta_{t+1} = \delta\theta_t - \eta_t \left[\nabla L(\tilde{\theta}_t) - \nabla L(\theta_t)\right]$$[1]

**헤시안을 통한 근사:**
손실함수의 1차 테일러 전개를 이용:

$$\nabla L(\tilde{\theta}_t) = \nabla L(\theta_t) + H[L(\theta_t)]\delta\theta_t + o(\|\delta\theta_t\|^2)$$

여기서 $H[L(\theta_t)]$는 헤시안 행렬입니다. 고차 항을 무시하면:[1]
$$\delta\theta_{t+1} = (I - \eta_t H[L(\theta_t)]) \delta\theta_t$$

**반복적 섭동 전파:**
$$\delta\theta_t = (I - \eta_{t-1}H[L(\theta_{t-1})])(I - \eta_{t-2}H[L(\theta_{t-2})])  \cdots (I - \eta_0 H[L(\theta_0)]) \delta\theta_0$$[1]

**라이아푸노프 지수 정의:**
$$LE = \lim_{t \to \infty} \frac{1}{t} \ln \left( \frac{\|\delta\theta_t\|}{\|\delta\theta_0\|} \right)$$

학습률과 헤시안의 관계:
$$LE \geq \lim_{t \to \infty} \frac{1}{t} \sum_{i=0}^{t-1} \ln(1 - \eta_i \|H[L(\theta_i)]\|)$$
$$LE \leq \lim_{t \to \infty} \frac{1}{t} \sum_{i=0}^{t-1} \ln(\|I - \eta_i H[L(\theta_i)]\|)$$[1]

#### 2.3 LE 기반 학습률 동적 조정 메커니즘

**핵심 혁신:**
$$\eta_{t+1} = \eta_t \cdot \exp(-\beta \cdot \Delta LE_t) \quad \text{if} \quad \Delta LE_t > 0$$

여기서 $\Delta LE_t = LE_t - LE_{t-1}$은 연속 에포크 간 LE 변화량, $\beta$는 감도 제어 하이퍼파라미터입니다.[1]

**직관적 의미:**
- $\Delta LE_t > 0$: 모델이 카오스 경계로 접근 → 학습률 감소로 더 세밀한 탐색
- $\Delta LE_t \leq 0$: 안정 영역 → 학습률 유지로 탐색 속도 유지

#### 2.4 적대적 데이터 증강과의 통합

**결합 최적화 목표:**

$$\min_\theta \max_\omega \mathbb{E}_{(x,y) \sim D_S} \left[\ell(\theta; \tau(x;\omega), y) - \lambda d_\theta(\tau(x;\omega), x)\right] + \frac{\gamma}{2} \|\theta\|_2^2$$

[1]

여기서:
- $\tau(x;\omega)$: 의미론적 변환(semantic transformation)
- $\ell(\theta; \cdot)$: 예측 손실
- $\lambda$: 적대적 손실과 특성 일관성의 균형 파라미터
- $d_\theta(\cdot)$: 원본과 변환 샘플 간의 특성 거리
- $\gamma$: 가중치 감소 정규화(헤시안이 양의 정부호 유지 → LE 음수 유도)[1]

#### 2.5 훈련 알고리즘 구조

**Algorithm 1: LEAwareSGD의 절차**[1]
1. 초기 매개변수 $\theta_0$, 학습률 $\eta_0$, 가중치 파라미터 $\beta$ 입력
2. 각 반복 $t = 0, 1, ..., N-1$에 대해:
   - 그래디언트 계산: $\nabla_\theta L(\theta_t)$
   - 매개변수 업데이트: $\theta_{t+1} = \theta_t - \eta_t \nabla_\theta L(\theta_t)$
   - 방정식 6을 이용한 섭동 계산
   - 방정식 7로 LE 계산
   - $\Delta LE_t = LE_t - LE_{t-1}$ 계산
   - $\Delta LE_t > 0$이면 $\eta_{t+1} = \eta_t \cdot \exp(-\beta \cdot \Delta LE_t)$, 아니면 $\eta_{t+1} = \eta_t$

***

### 3. 성능 향상 및 한계
#### 3.1 성능 향상 결과

**벤치마크 성능 개선:**[1]

| 데이터셋 | 기존 최고(AdvST) | LEAwareSGD | 향상도 |
|---------|-----------------|-----------|--------|
| PACS | 67.06% | 69.46% | +2.40% |
| OfficeHome | 52.60% | 54.38% | +1.78% |
| DomainNet | 27.22% | 28.15% | +0.93% |

[1]

**저데이터 환경에서의 강력한 성능:**[1]

| 데이터 비율 | AdvST | LEAwareSGD | 향상도 |
|-----------|-------|-----------|--------|
| 10% | 49.26% | 58.73% | **+9.47%** |
| 20% | 53.55% | 61.78% | +8.23% |
| 50% | 60.18% | 66.44% | +6.26% |

[1]

**여러 백본 네트워크에서의 일관된 개선:**[1]
- ResNet-34: 70.42% → 73.68% (+3.26%)
- ResNet-50: 68.27% → 71.37% (+3.10%)
- ResNet-101: 70.32% → 74.32% (+4.00%)
- ResNet-152: 71.92% → 75.34% (+3.42%)

[1]

**다른 최적화기와의 비교:**[1]
- Adam: 66.43%
- AdamW: 66.83%
- RMSprop: 62.34%
- SGD: 67.06%
- **LEAwareSGD: 69.46%**

#### 3.2 한계 및 제약사항

**1. 계산 복잡도:**
- PACS에서 평균 훈련 시간: 1.99시간 (AdvST 1.90시간 대비 약 5% 증가)
- OfficeHome에서: 1.19시간 (AdvST 0.75시간보다 약 59% 증가)
- LE 계산 오버헤드로 인한 시간 비용 발생[1]

**2. 매개변수 민감도:**
- 하이퍼파라미터 $\beta$: PACS에서는 1e-3, OfficeHome에서는 1e-2가 최적 (데이터셋 간 편차 있음)
- 가중치 감소 $\gamma$: PACS에서 5e-4, OfficeHome에서 1e-5로 큰 편차 필요[1]

**3. 특정 도메인에서의 약화된 성능:**
- DomainNet의 Quickdraw(Q) 도메인에서 6.70% 달성 (SimDE 6.85%보다 0.15% 낮음)
- 추상화 수준이 높은 도메인에 대한 특화 증강 기법 부재[1]

**4. 이론적 한계:**
- LE 근사에서 고차 테일러 항 무시로 인한 근사 오차[1]
- 실제 헤시안 계산의 계산 비용이 높아 근사값 사용 필요
- 카오스 경계 위치의 정확한 정의 부재[1]

**5. 일반화 가능성의 불확실성:**
- 다중 도메인 일반화(Multi-Source DG) 설정에서는 비교 가능한 성능 수준 (84.52% vs. PSDG 84.34%)[1]
- 더 대규모 데이터셋과 복잡한 도메인 생성 작업에 대한 검증 부족[1]

[1]

***

### 4. 모델 일반화 성능 향상 가능성
#### 4.1 일반화 향상의 핵심 메커니즘

**카오스의 경계(Edge of Chaos)의 의미:**
논문의 핵심 가설은 신경망이 **카오스의 경계** 근처에서 학습할 때 최적의 일반화 성능을 달성한다는 것입니다.[1]

- **안정 영역($LE \ll 0$)**: 고정점에 수렴하여 새로운 패턴 학습 능력 제한 → 과적합
- **혼돈 영역($LE \gg 0$)**: 섭동에 극도로 민감하여 불안정한 학습
- **경계 영역($LE \approx 0$)**: 안정성과 적응성의 최적 균형 → 최대 일반화 능력[1]

**실험적 증거:**[1]
- t-SNE 시각화: LEAwareSGD가 기존 방법(ADA, ME-ADA, AdvST)보다 넓은 매개변수 공간 탐색
- LE 동역학 분석: 논문의 방법이 모든 도메인에서 LE 값을 영점 근처에 유지

#### 4.2 데이터 부족 환경에서의 강화된 일반화

**저데이터 레짐의 우월성:**
10% 데이터만으로 AdvST 대비 9.47% 향상은 다음을 시사합니다:[1]
1. 광범위한 매개변수 공간 탐색으로 더 강력한 특성(feature) 발견
2. 도메인 불변 특성의 학습에 더 유리한 최적화 궤적
3. 제한된 데이터에서의 과적합 억제 효과

#### 4.3 다양한 백본 구조에서의 일관된 개선

**아키텍처 불가지론적 특성:**
ResNet-18부터 ResNet-152까지 모든 깊이에서 일관된 3-4% 향상을 달성한 점은 방법의 범용성을 시사합니다.[1]

#### 4.4 다른 증강 방법과의 호환성

**통합 가능성:**
LE-aware 최적화를 ADA, ME-ADA, AdvST 등과 결합했을 때:
- ADA: +0.52% (단독 62.14%)
- ME-ADA: +2.30% (단독 62.52%)
- AdvST: +2.40% (단독 69.46%)

이는 LEAwareSGD가 **독립적인 최적화 개선 메커니즘**으로 다른 데이터 증강 기법과 상호보완적임을 의미합니다.[1]

#### 4.5 다중 도메인 일반화로의 확장 가능성

**Leave-One-Domain-Out 실험:**
- 3개 도메인으로 학습, 1개 도메인 테스트 설정에서 PACS 84.52% 달성
- PSDG 84.34%와 비교하여 경쟁력 있는 수준 유지[1]
- 단일 도메인 학습 문제에 최적화되었으나 다중 도메인으로도 적용 가능함을 증명

***

### 5. 관련 최신 연구 비교 분석 (2020년 이후)
#### 5.1 적대적 데이터 증강 기반 방법

**ADA (2018, 재확인됨 2020+)**[1]
- 방식: 기본적인 적대적 샘플 생성
- 성과: PACS 61.11%
- 한계: 국소적 섭동만 적용, 넓은 탐색 공간 활용 불가

**ME-ADA (2020)**[1]
- 방식: 최대 엔트로피 원칙으로 다양한 적대적 샘플 생성
- 성과: PACS 60.22%
- 한계: 여전히 매개변수 공간의 세부 탐색 제한

**AdvST (2024)**[1]
- 방식: 다양한 데이터 증강 기법을 학습 가능한 파라미터로 개선
- 성과: PACS 67.06%, OfficeHome 52.60%
- 한계: 최적화 과정에 동역학계 이론 미반영

#### 5.2 도메인 확장 기반 방법

**SimDE (2023)**[1]
- 방식: 이중 분류기 지도 하에 도메인 확장
- 성과: PACS에서 유사 수준, DomainNet 26.98%
- 특징: 생성 모델 사용하지 않으면서도 의사 도메인 생성

**PSDG (2024)**[1]
- 방식: 훈련 시간 및 테스트 시간 학습 모두 적용
- 성과: PACS 67.14%
- 한계: 논문에서는 공정한 비교를 위해 테스트 타임 학습 제외

**SimDE 등 생성 모델 기반 방법 (2023+)**[1]
- DRSF (2025): 잠재 확산 모델(LDM)을 이용한 의사 도메인 생성
- 한계: 계산 비용 높음, 합성 데이터 품질 의존

#### 5.3 최적화 관점의 개선

**SAM (Sharpness-Aware Minimization, 2020)**[1]
- 방식: 손실 경계의 예각함을 최소화
- 성과: 일반 도메인 일반화 및 SDG에서 유용
- 한계: 직접적인 동역학계 이론 활용 없음

**GSAM (2023)**[1]
- 방식: 서로게이트 갭 도입으로 SAM 개선
- 성과: 예각함과 손실 모두 낮은 영역 탐색
- 한계: LE 기반 동적 조정 메커니즘 없음

**엣지 오브 카오스 이론 (Zhang et al., 2021)**[1]
- 주요 기여: 신경망이 카오스 경계에서 최고의 일반화 성능 달성
- 제한: 이론적 분석만 제공, 실제 SDG 최적화에 직접 적용되지 않음
- **LEAwareSGD의 혁신**: 이 원리를 직접 SDG 최적화에 처음 적용

#### 5.4 텍스트 기반 및 목표 지향 방법

**TDG (Text-Guided Domain Generalization, 2023)**[1]
- 방식: 시각-언어 모델을 활용하여 도메인 일반화
- 한계: 텍스트 설명의 가용성 필요

**TO-SDG (Target-Oriented SDG, 2024)**[1]
- 방식: 목표 도메인의 텍스트 설명 활용
- 특징: SDG의 새로운 확장 문제
- 한계: 목표 도메인 정보의 사전 지식 필요

#### 5.5 비교 종합 표

| 방법 | 발표연도 | 핵심 아이디어 | PACS | OfficeHome | DomainNet | 주요 장점 | 주요 한계 |
|-----|--------|-----------|------|-----------|----------|---------|---------|
| ADA | 2018 | 적대적 데이터 증강 | 61.11% | 44.75% | 24.26% | 기초 방법 | 국소적 섭동만 사용 |
| ME-ADA | 2020 | 최대 엔트로피 적대적 증강 | 60.22% | 45.35% | 24.63% | 다양성 개선 | 제한된 탐색 |
| SAM | 2020 | 예각함 최소화 | - | - | - | 이론적 기초 | SDG 최적화 미흡 |
| SimDE | 2023 | 도메인 확장 | 65+ | - | 26.98% | 효율적 | 생성 모델 품질 의존 |
| AdvST | 2024 | 학습 가능한 증강 | 67.06% | 52.60% | 27.22% | 강력한 베이스라인 | 동역학계 이론 미반영 |
| **LEAwareSGD** | **2025** | **LE 기반 동적 학습률** | **69.46%** | **54.38%** | **28.15%** | **일관된 향상, 저데이터 우수** | **계산 비용, 하이퍼파라미터 민감도** |

[1]

***

### 6. 앞으로의 연구에 미치는 영향 및 고려사항
#### 6.1 이론적 발전 방향

**1. 동역학계 관점의 심화 연구:**
- LE와 일반화 간의 정량적 관계식 도출
- 신경망의 위상 공간에서 카오스 경계의 정확한 수학적 특성화[1]
- 다양한 아키텍처(Transformer, Vision Transformer 등)에서의 LE 행동 분석

**2. 헤시안 근사의 정확성 개선:**
- 현재 1차 테일러 전개의 고차 항 영향 분석[1]
- 효율적인 헤시안 추정 알고리즘 개발
- 확률적 LE 추정 방법 개발

#### 6.2 실용적 확장 방향

**1. 계산 효율성 개선:**
- LE 계산을 위한 근사 기법 (예: 대각 헤시안 근사)
- GPU 병렬화를 통한 계산 가속
- 샘플링 기반 LE 추정으로 오버헤드 감소[1]

**2. 하이퍼파라미터 자동 조정:**
- 데이터셋 특성에 기반한 $\beta$ 자동 선택 메커니즘
- $\gamma$ 값의 적응적 조정으로 일반화 가능성 향상
- 메타-러닝을 통한 파라미터 최적화[1]

**3. 다양한 도메인 생성 작업 확대:**
- 물체 탐지(Object Detection)에의 적용[1]
- 의료 영상 분석에서의 SDG 적용
- 자율주행 및 로봇 비전 태스크 확대

#### 6.3 관련 분야로의 파급 효과

**1. 전이 학습(Transfer Learning) 개선:**
- 사전 학습 모델의 새로운 도메인으로의 적응 최적화
- 풋샷 러닝(Few-Shot Learning)에서의 빠른 수렴

**2. 강건성(Robustness) 강화:**
- 적대적 공격(Adversarial Attack) 방어 개선
- 분포 외 일반화(Out-of-Distribution Generalization) 확대
- 자연적 분포 시프트에 대한 저항성 강화[1]

**3. 연속 학습(Continual Learning):**
- 새로운 도메인이 순차적으로 추가될 때의 적응형 학습
- 망각(Catastrophic Forgetting) 완화[1]

#### 6.4 다학제적 연구 기회

**1. 물리학 및 복잡계 이론과의 교점:**
- 신경망 학습의 상전이(Phase Transition) 분석
- 통계 역학(Statistical Mechanics) 관점의 신경망 이해

**2. 생물학적 영감 모델:**
- 생물학적 신경망의 카오스 동역학 관찰
- 뇌의 학습 메커니즘과의 유사성 탐구

#### 6.5 실제 배포 시 고려사항

**1. 계산 리소스 최적화:**
- 엣지 컴퓨팅 환경에서의 경량화 버전 개발
- 임베디드 시스템 적용 가능성 검토[1]

**2. 설명 가능성(Interpretability):**
- LE 값이 높은 이유의 직관적 해석 도구 개발
- 의사결정 과정의 투명성 확보

**3. 안정성 보증:**
- 다양한 하드웨어 및 데이터 환경에서의 검증 필요
- 실시간 애플리케이션에서의 성능 보증[1]

#### 6.6 미해결 연구 문제

**1. 카오스 경계의 모드 문제:**
- 최적 LE 값이 정확히 무엇인가? (항상 0에 매우 가까워야 하는가?)
- 도메인과 작업에 따라 최적값이 변하는가?[1]

**2. 확장성 한계:**
- 매우 큰 모델(수십억 파라미터)에서 방법의 실효성은?
- ImageNet 규모의 데이터셋에서의 성능은?[1]

**3. 이론-실제 간극:**
- 논문의 LE 기반 이론이 실제 일반화 성능 향상을 완전히 설명하는가?
- 다른 요인의 기여도는 얼마나 되는가?[1]

***

### 7. 결론
**LEAwareSGD는 다음의 관점에서 의의 있는 연구입니다:**

1. **이론적 혁신:** 동역학계 이론의 LE 개념을 SDG 최적화에 처음으로 성공적으로 도입하여, 신경망 훈련을 새로운 각도에서 이해하는 프레임워크 제시[1]

2. **실질적 성능 개선:** PACS에서 최대 9.47% (저데이터 환경), 평균 2-2.5%의 일관된 향상을 달성하여 현실적 가치 입증[1]

3. **범용성:** 다양한 백본, 데이터셋, 증강 기법과 호환되는 방법론으로 광범위한 적용 가능성 제시[1]

4. **미래 지향성:** 동역학계, 카오스 이론 등 물리학 기반 원리를 머신러닝에 도입하는 새로운 방향 제시[1]

**동시에 극복해야 할 과제:**
- 계산 효율성 개선으로 실제 배포 가능성 확대[1]
- 하이퍼파라미터 민감도 감소를 통한 사용성 개선
- 초대규모 모델과 데이터셋에서의 확장성 검증[1]

이 논문은 일반화 문제를 푸는 새로운 관점을 제시함으로써, 향후 10년간의 도메인 일반화 및 최적화 연구에 중요한 영향을 미칠 것으로 예상됩니다. 특히 **물리학과 머신러닝의 교점**에서 펼쳐질 흥미로운 연구의 출발점이 될 가능성이 높습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3752e603-4e6d-421d-8f1c-d570f5f54be0/2507.04302v1.pdf)
[2](https://www.semanticscholar.org/paper/96bf2787b22ed4116133f01b2c066cd67e7d6c0f)
[3](https://www.science.org/doi/10.1126/science.adp8778)
[4](https://link.springer.com/10.1007/s15010-024-02336-4)
[5](https://academic.oup.com/nar/article/53/D1/D356/7874847)
[6](https://arxiv.org/abs/2406.07250)
[7](https://diabetesjournals.org/diabetes/article/73/Supplement_1/1997-LB/155679/1997-LB-Curation-of-an-AI-Ready-Dataset-Using)
[8](https://doi.apa.org/doi/10.1037/spq0000637)
[9](https://aacrjournals.org/cancerres/article/84/17_Supplement/B074/747178/Abstract-B074-Clonal-decomposition-and-DNA)
[10](https://academic.oup.com/eurpub/article/doi/10.1093/eurpub/ckae144.145/7843583)
[11](https://ashpublications.org/blood/article/144/Supplement%201/4825/533294/Phase-1-Study-of-Anitocabtagene-Autoleucel-for-the)
[12](https://arxiv.org/html/2503.13617v1)
[13](https://arxiv.org/abs/2210.14507)
[14](http://arxiv.org/pdf/2402.18447.pdf)
[15](http://arxiv.org/pdf/2312.12720.pdf)
[16](http://arxiv.org/pdf/2304.07261.pdf)
[17](https://arxiv.org/abs/2108.11726)
[18](http://arxiv.org/pdf/2308.09931.pdf)
[19](http://arxiv.org/pdf/2411.02920.pdf)
[20](https://openaccess.thecvf.com/content/CVPR2023W/L3D-IVU/papers/Xu_SimDE_A_Simple_Domain_Expansion_Approach_for_Single-Source_Domain_Generalization_CVPRW_2023_paper.pdf)
[21](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2022.818799/full)
[22](https://scipost.org/submissions/scipost_202110_00024v2/)
[23](https://papers.miccai.org/miccai-2024/704-Paper3740.html)
[24](https://test-sprott.physics.wisc.edu/pubs/paper387.pdf)
[25](https://www.semanticscholar.org/paper/Edge-of-chaos-as-a-guiding-principle-for-modern-Zhang-Feng/1027e9693736ce90fd5c33419984621a8b0a6f60)
[26](https://paperswithcode.com/task/single-source-domain-generalization)
[27](https://openreview.net/pdf?id=37Fh1MiR5Ze)
[28](https://www.worldscientific.com/doi/10.1142/S2972335323500011)
[29](https://arxiv.org/pdf/2103.03097.pdf)
[30](https://arxiv.org/html/2509.00351v1)
[31](https://arxiv.org/html/2501.15928v1)
[32](https://www.arxiv.org/pdf/2508.17655.pdf)
[33](https://openaccess.thecvf.com/content/CVPR2024/papers/Peng_Single_Domain_Generalization_for_Crowd_Counting_CVPR_2024_paper.pdf)
[34](https://arxiv.org/html/2406.16161v1)
[35](https://arxiv.org/pdf/2505.20030.pdf)
[36](https://arxiv.org/abs/2312.12720)
[37](https://arxiv.org/abs/2410.05988)
[38](https://arxiv.org/abs/2107.09437)
[39](https://arxiv.org/abs/2308.00918)
[40](https://ui.adsabs.harvard.edu/abs/2025CNSNS.14008397Y/abstract)
[41](https://tajanthan.github.io/misc/docs/chaos.pdf)
