# Explainable AI for Trees: From Local Explanations to Global Understanding

### 1. 논문의 핵심 주장과 주요 기여

본 논문은 트리 기반 머신러닝 모델의 해석 가능성을 획기적으로 개선한 **TreeExplainer**를 제시합니다. 핵심 주장은 tree-based 모델(랜덤 포레스트, 그래디언트 부스팅, 의사결정나무)이 실무에서 가장 널리 사용되는 비선형 모델임에도 불구하고, 예측 설명에 대한 연구가 부족하다는 것입니다.

**세 가지 주요 기여:**

1. **다항식 시간 알고리즘**: 게임 이론 기반의 **Shapley 값**을 정확하게 계산하는 첫 번째 다항식 시간 알고리즘 개발
2. **상호작용 효과 측정**: 로컬 피처 간 상호작용 효과를 직접 측정하는 새로운 설명 유형 도입
3. **글로벌 모델 이해 도구**: 다수의 로컬 설명을 결합하여 모델의 전역 구조를 파악하는 종합 도구 제시

***

### 2. 문제 정의, 제안 방법, 모델 구조 및 성능

#### 2.1 해결하고자 하는 문제

**기존 방법의 한계:**

- **의사결정 경로 보고**: 많은 트리로 이루어진 앙상블 모델에서는 거의 도움이 되지 않음
- **Saabas 휴리스틱 방법**: 피처의 트리 깊이에 따라 할당 편향이 발생하여 **일관성(consistency)** 문제 야기
- **모델-무관(model-agnostic) 방법**: 계산 비효율과 높은 샘플링 변동성으로 인해 큰 데이터셋에는 실용적이지 못함

#### 2.2 제안하는 방법: SHAP 값과 TreeExplainer

**SHAP 값의 수학적 정의:**

각 피처 i의 로컬 기여도는 모든 가능한 피처 순서에 대해 조건부 기댓값의 변화를 평균화하여 계산됩니다:

$$\phi_i(f, x) = \sum_{R \in \mathcal{R}} \frac{1}{M!} \left[ f_x(P_i^R \cup \{i\}) - f_x(P_i^R) \right]$$

여기서:
- $\mathcal{R}$: 모든 피처 순서의 집합
- $P_i^R$: 순서 R에서 피처 i 이전의 모든 피처
- $f_x(S)$: 피처 집합 S가 주어졌을 때 모델 출력의 조건부 기댓값
- $M$: 총 피처 수

**SHAP 값의 세 가지 속성 보장:**

$$\text{Property 1 (로컬 정확성):} \quad f(x) = \phi_0(f) + \sum_{i=1}^{M} \phi_i(f, x)$$

$$\text{Property 2 (일관성):} \quad f'_x(S) - f'_x(S \backslash i) \geq f_x(S) - f_x(S \backslash i) \Rightarrow \phi_i(f', x) \geq \phi_i(f, x)$$

$$\text{Property 3 (누락):} \quad f_x(S \cup i) = f_x(S) \Rightarrow \phi_i(f, x) = 0$$

**Tree SHAP 알고리즘의 복잡도 개선:**

| 방법 | 복잡도 | 특징 |
|------|--------|------|
| 기존 Shapley 방법 | $O(TLM2^M)$ | 지수 함수적 |
| Tree SHAP | $O(TLD^2)$ | 다항식 시간 |
| 균형잡힌 트리 | $O(TL \log^2 L)$ | 실무적 효율성 |

여기서 T는 트리 수, L은 최대 리프 수, D는 최대 깊이입니다.

#### 2.3 모델 구조

**Tree SHAP 알고리즘의 핵심 구조:**

알고리즘 2는 트리를 재귀적으로 탐색하면서 각 리프로 흘러들어가는 모든 가능한 부분집합의 비율을 추적합니다:

```
Algorithm 2: TREESHAP 
입력: 인스턴스 x, 트리 정보
출력: 각 피처의 SHAP 값

1. 경로 추적(path tracking): 트리 깊이의 부분집합 크기 추적
2. 가중치 계산: Equation 5의 Shapley 가중치 적용
3. 특성 분해: 각 리프에서 경로의 각 피처에 대해 기여도 계산
```

**SHAP 상호작용 값:**

피처 쌍 간의 상호작용을 행렬 형태로 표현:

$$\phi_{i,j}(f, x) = \text{game theory의 Shapley interaction index 적용}$$

대각선 원소는 주효과(main effects), 비대각선은 상호작용 효과(interaction effects)입니다.

#### 2.4 성능 향상

**의료 데이터셋에서의 비교 성능:**

| 모델 | 사망률 (C-통계) | 만성신장질환 (PR 곡선 하 면적) | 수술 시간 (R²) |
|------|-----------------|--------------------------------|-----------------|
| 그래디언트 부스팅 | **0.821** | **0.890** | **0.674** |
| 선형 모델 | 0.813 | 0.871 | 0.595 |
| 신경망 | 0.816 | 0.872 | 0.629 |

**계산 효율성:**

- TreeExplainer: Saabas 및 모델-무관 방법 대비 **수천 배 빠름**
- Saabas의 chronic kidney disease 데이터셋: 상호작용 값 계산에 3 CPU년 소요
- TreeExplainer: 동일 작업 완료 가능 (정확한 값, 샘플링 변동성 없음)

**설명 방법 성능 벤치마크:**

21개의 평가 지표에서 TreeExplainer는 일관되게 다른 방법들을 능가했습니다. 특히:
- Saabas 방법의 일관성 문제 해결
- 사용자 직관과 100% 일치 (12개 시나리오)

#### 2.5 한계점

1. **조건부 기댓값 계산의 가정**: 피처 독립성 가정이 위배될 때 왜곡 가능성
2. **배경 샘플(reference set) 선택의 민감성**: 결과가 기댓값 추정 방식에 영향받음
3. **비선형 변환 처리**: 모델 손실함수 설명 시 근사를 사용하여 정확성 저하
4. **고차원 데이터의 확장성**: 피처 수가 매우 많을 때 계산 복잡도 증가

***

### 3. 일반화 성능 향상 가능성

#### 3.1 논문의 직접적 기여

**비선형 데이터에서의 해석 가능성 개선:**

논문은 중요한 역설을 발견했습니다: 단순한 선형 모델이 더 해석하기 쉽다고 여겨지지만, **비선형 데이터에서는 오히려 저편향(low-bias) tree 모델이 더 해석 가능**합니다.

시뮬레이션 결과(Figure 2D):
- 비선형성 증가에 따라 선형 모델은 **관련 없는 피처에 잘못된 가중치** 할당
- 그래디언트 부스팅은 **올바른 피처만 선택** 유지

이는 모델 설명의 충실성(faithfulness)이 정확도뿐만 아니라 **모델 편향과도 연관**됨을 시사합니다.

```math
\text{설명 오류} =  (f(\text{모델\ 정확도},\text{편향},\text{설명\ 방법\ 질}))
```

#### 3.2 로컬 설명에서 글로벌 패턴으로의 일반화

**5가지 글로벌 이해 도구:**

**1) 로컬 모델 요약 (SHAP 요약도):**
- 희귀하지만 고영향 효과 발견 가능
- 기존 feature importance는 놓치는 장기 꼬리(long tail) 패턴 포착

**2) 로컬 피처 의존성:**
- 개별 샘플의 비선형 관계 시각화
- 상호작용 효과로 인한 수직 분산(vertical dispersion) 정량화

**3) 로컬 상호작용 분해:**
대각 행렬 원소를 이용한 주효과와 상호작용 분리:

$$\phi_i(f, x) = \underbrace{\sum_j \phi_{i,j}(f, x) \quad (j=i)}_{\text{주효과}} + \underbrace{\sum_{j \neq i} \phi_{i,j}(f, x)}_{\text{상호작용}}$$

**4) 배포된 모델 모니터링:**
손실함수에 대한 설명을 통해 시간에 따른 피처 드리프트 감지 가능.

의료 데이터셋의 실증 예:
- 의도적 데이터 라벨링 오류 즉시 감지
- 미발견 EMR 설정 문제 파악
- 심방세동 제거술 처리 시간 변화 추적 (p = 5.4 × 10⁻¹⁹)

**5) 설명 임베딩 기반 클러스터링:**
- 비지도 학습의 거리 지표 문제 해결
- 지도 클러스터링: 결과 관련 피처에 가중치 자동 적용

#### 3.3 일반화 성능 향상 메커니즘

**모델 일반화와 설명 가능성의 연결:**

1. **특성 선택 강화**: SHAP 값 기반 특성 선택이 전통적 방법(Gain, 순열 테스트) 대비 우월한 특성 복구율 달성 (Supplementary Figure 7)
   - 평균 p-값 < 10⁻⁷로 통계적으로 유의미

2. **과적합(overfitting) 감소**: 
   - TreeExplainer로 로컬 예측 분석 → 과적합 피처 식별
   - 배포 모니터링 → 피처 드리프트 초기 감지 → 모델 재훈련 시점 결정

3. **상호작용 효과 고려**:
   - SHAP 상호작용 값이 고차 상호작용 효과 포착
   - 기존 특성 선택 방법(순열)보다 **고차 상호작용 효과를 반영**하여 더 정확한 특성 순위 제공

***

### 4. 논문의 임상적 및 실무적 영향

#### 4.1 의료 적용 사례

**사망률 위험 예측 (NHANES I):**
- 14,407 명, 79개 피처, 20년 추적
- 음성 부작용: "조기 사망하는 많은 방법이 있지만 조기 장수하는 방법은 거의 없음" → 효과 분포의 우측 스큐 설명

**만성신장질환 진행 예측 (CRIC):**
- 3,939 명의 CKD 환자, 333개 피처
- 발견: 혈소판 수와 혈중 요소질소의 상호작용이 신장 기능 저하 가속화
- 임상적 해석: 염증이 고요소혈증과 상호작용하여 신장 기능 악화

**병원 수술 시간 예측:**
- 147,000 절차, 2,185개 피처
- 배포 후 모니터링으로 데이터 파이프라인 오류와 피처 드리프트 검출

***

### 5. 논문이 앞으로의 연구에 미치는 영향

#### 5.1 설명 가능 AI 분야의 변화

**학술적 영향:**
1. **게임 이론과 ML의 연결**: Shapley 값의 "유일성 정리"가 설명 방법의 이론적 기초 제공
2. **모델-특정(model-specific) 효율성 방향**: 모델-무관 접근의 한계를 극복하는 새로운 패러다임 제시
3. **로컬에서 글로벌로의 확장**: 로컬 설명이 단순히 개별 예측 해석에 그치지 않고 글로벌 인사이트 생성

**실무 채택:**
- SHAP은 2019년 이후 가장 널리 사용되는 XAI 방법론이 됨
- XGBoost, LightGBM, CatBoost 등 주요 라이브러리에 통합
- 금융, 의료, 제조 등 다양한 도메인에서 채택

#### 5.2 2020년 이후 관련 최신 연구 동향

**A. 상호작용 효과 분석의 고도화**

1. **Beyond TreeSHAP (2024)**: 임의 차수의 Shapley 상호작용 계산[1]
   - 기존 SHAP 상호작용 값은 2차 상호작용만 지원
   - 고차 상호작용(3차 이상)을 효율적으로 계산 가능

2. **Succinct Interaction-Aware Explanations (2024)**: 지수적 크기의 상호작용 설명을 간결하게 표현[2]
   - NSHAP 프레임워크: 모든 특성 부분집합의 가산 중요성 보고
   - 해석 가능성과 완전성의 균형

3. **CLE-SH (2024)**: SHAP 값 해석의 통계적 유효성 검증[3]
   - 중요 피처 수 자동 결정
   - 각 피처의 패턴 및 상호작용을 통계적 유의성 기반으로 제시

**B. 계산 효율성 개선**

1. **Amortized SHAP via Sparse Fourier (2024)**: 반복 계산 최적화[4]
   - Tree와 black-box 모델에서 SHAP 값 계산 가속화
   - 실시간 애플리케이션 적용 가능성 증가

2. **SHEP (2024)**: 선형 복잡도의 근사 방법[5]
   - 지수 복잡도를 선형으로 감소
   - 실시간 고장진단 모니터링에 적용

**C. 일반화 성능과 연결**

1. **SHAP-Guided Regularization (2025)**: 정규화 기법 통합[6]
   - SHAP 값을 활용하여 허위 상관관계(spurious correlation) 감지
   - 과적합 감소로 일반화 성능 향상 입증

2. **Latent SHAP (2022)**: 인코딩된 특성 공간에서의 설명[7]
   - 깊은 모델의 저수준 특성에 대한 설명 개선
   - 사용자-해석 가능한 피처 공간으로 변환

**D. 모델 구조별 확장**

1. **G-DeepSHAP (2022)**: 복합 모델 체인 설명[8]
   - 선형, 깊은 신경망, 트리 모델의 연쇄 구조 해석
   - 배포된 독점 모델(proprietary models) 설명 가능

2. **Interpretable Additive Models with Shapley (2025)**: GAM과 SHAP 통합[9]
   - 선형성과 해석 가능성을 유지하면서 Shapley 값 정확성 보장
   - 단일 순방향 패스로 SHAP 값 계산

**E. 실무 응용 확대**

2020년 이후 의료, 금융, 제조 분야의 주요 사례:

| 분야 | 응용 | 연도 | 성과 |
|------|------|------|------|
| **의료** | 갑상선암 재발 예측 | 2025 | CatBoost + SHAP: 97% 정확도, 주요 바이오마커 식별 |
| **의료** | 당뇨병성 신부전 예측 | 2024 | XGBoost + SHAP: AUC 0.966, 대사체-당뇨병 상호작용 발견 |
| **의료** | 뇌졸중-폐렴 위험 | 2024 | SHAP로 고위험군 조기 식별, 조기 중재 시간 32% 단축 |
| **금융** | 신용카드 사기 탐지 | 2025 | SHAP + LIME: 95% 정확도, 거래 특성 자동 해석 |
| **금융** | 신용점수 평가 | 2025 | 규제 준수(설명 의무) 달성, 고객 신뢰 30% 증가 |
| **제조** | IoT 사이버보안 | 2025 | RF + SHAP: 99.99% 탐지율, 임계 네트워크 특성 식별 |
| **제조** | 헬리콥터 엔진 진단 | 2024 | AdaBoost + SHAP: 실시간 고장 분류, 예방 유지보수 |

***

### 6. 앞으로 연구 시 고려할 점

#### 6.1 이론적 고려사항

**1) 조건부 기댓값의 정의 문제:**

논문에서 제시한 $f_x(S) = E[f(x) | x_S]$는 암묵적으로 특정 배경 분포(background distribution)를 가정합니다. 그러나:

- **전주변 vs. 개입적(interventional) Shapley 값**: 두 가지 선택지의 해석상 차이
- **기댓값 추정 편향**: 배경 샘플 크기에 따른 SHAP 값 편향 가능성
- **특성 의존성 처리**: 완전히 독립적인 특성은 실제로 드물고, 조건부 기댓값 계산 시 왜곡 발생

**권장 해결책:**
- 여러 기댓값 정의를 비교 분석
- 신뢰도 구간(confidence interval) 제시
- 특성 상관관계 구조를 명시적으로 모델링

**2) Shapley 값의 고유성 정리의 한계:**

논문의 Theorem 1은 세 가지 속성(로컬 정확성, 일관성, 누락)을 만족하는 **유일한** 설명 방법이 Shapley 값임을 보여줍니다. 그러나:

- 이 세 속성이 **항상 바람직한가?** (예: 도메인 지식의 비대칭 가중치 필요한 경우)
- **집단 공정성(group fairness)** vs. **개인 공정성(individual fairness)** 트레이드오프
- 고차 상호작용을 무시하는 가산 모델의 한계

**권장 관점:**
- 속성 세트의 유연성 모색
- 다목적 설명 프레임워크(multi-objective explanations) 개발

#### 6.2 실무적 고려사항

**1) 배포 후 모니터링 체계화:**

논문은 병원 절차 시간 예측에서 세 가지 미발견 문제를 조기 탐지했습니다. 하지만:

- **정상 변동 vs. 이상 변동의 경계**: 통계적 임계값 설정의 어려움
- **자동 경고 시스템**: 임상가의 의사결정 오버로드 가능성
- **개인정보 보호**: 설명 값 저장 및 접근 통제

**권장 프레임워크:**
```
배포 모니터링 = 수집(SHAP 값) + 변화 감지 + 자동 재훈련 + 의사결정 지원
```

**2) 모델 업데이트와 설명 일관성:**

재훈련 후 동일 샘플의 SHAP 값 변화:
- 모델 변경이 의도된 개선인지 확인 필요
- 설명의 "drift" 모니터링

**3) 사용자 이해의 격차:**

SHAP 값은 이론적으로 최적이지만, 사용자(의사, 관리자) 이해도:
- **Force plot vs. Beeswarm plot vs. Dependence plot**: 상황별 최적 시각화 선택
- **상호작용 효과의 복잡성**: 2차 이상 상호작용 이해 어려움
- **맥락 정보 부재**: 순수 SHAP 값만으로는 임상적 의사결정 불충분

**권장 전략:**
- 도메인 전문가와의 협업 필수
- 대화형 설명 시스템(interactive explanations) 개발
- 신뢰도와 불확실성 정량화

#### 6.3 미충족 연구 필요 영역

**1) 고차원 데이터에서의 확장:**
- 피처 수 > 1,000인 경우의 실용적 계산 방법
- 특성 상관 구조를 활용한 압축 표현(compressed representations)

**2) 시계열 및 동적 데이터:**
- TreeExplainer는 정적 데이터 가정
- 시계열 모델에 대한 확장 필요

**3) 인과 추론과의 통합:**
- SHAP 값은 기술적(descriptive)이지 인과적(causal)이 아님
- Causal SHAP 개발 필요

**4) 공정성과의 교집합:**
- SHAP 값이 편향된 예측을 충분히 탐지하는가?
- 공정한 설명(fair explanations)의 정의 필요

***

### 결론

**"Explainable AI for Trees: From Local Explanations to Global Understanding"**은 설명 가능 AI의 이정표적 논문입니다. TreeExplainer의 다항식 시간 정확 계산은 이론과 실무의 격차를 좁혔고, 로컬 설명의 집계를 통한 글로벌 인사이트 도출은 새로운 분석 패러다임을 제시했습니다.

**특히 주목할 점:**

1. **일반화 성능과의 연결**: 모델의 낮은 편향성이 설명 충실성을 향상시킨다는 발견은 정확도와 해석 가능성이 선택 관계가 아님을 시사

2. **배포 모니터링의 중요성**: 모델 손실 함수 설명을 통한 시스템적 오류 탐지는 실무적 영향이 큼

3. **2020년 이후 확산**: SHAP은 표준 XAI 방법론이 되었으며, 상호작용 분석, 계산 효율화, 공정성 통합 등 다층적 확장 진행 중

**향후 연구 방향:**
- 고차 상호작용과 인과 효과의 통합
- 실시간 계산을 위한 근사 방법과 정확성의 균형
- 도메인-특정 설명 프레임워크의 개발

이 논문은 단순한 방법 제시를 넘어, 설명 가능성 자체를 재정의한 근본적 기여를 했다고 평가됩니다.[10][11][1][2][3][4][5][6][7][8][9]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/eddb7335-93c1-448a-ad7a-0259bd1e66ee/1905.04610v1.pdf)
[2](https://ieeexplore.ieee.org/document/11005238/)
[3](https://www.icck.org/article/abs/jsspa.2025.321501)
[4](https://ieeexplore.ieee.org/document/11135406/)
[5](https://asmedigitalcollection.asme.org/MSEC/proceedings/MSEC2025/89022/V002T17A001/1222974)
[6](https://www.ijraset.com/best-journal/explainable-ai-in-cancer-diagnosis-enhancing-interpretability-with-shap--on-benign-and-malignant-tumor-detection)
[7](https://journals.lww.com/10.1097/MD.0000000000042667)
[8](https://ieeexplore.ieee.org/document/10968935/)
[9](https://ieeexplore.ieee.org/document/11203339/)
[10](https://ieeexplore.ieee.org/document/11017105/)
[11](https://link.springer.com/10.1007/s00704-025-05741-3)
[12](http://arxiv.org/pdf/2410.06300.pdf)
[13](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202400304)
[14](https://arxiv.org/abs/1905.04610)
[15](http://arxiv.org/pdf/2407.00506.pdf)
[16](https://arxiv.org/ftp/arxiv/papers/2210/2210.04533.pdf)
[17](http://arxiv.org/pdf/2404.11208.pdf)
[18](https://arxiv.org/html/2410.04883v1)
[19](https://arxiv.org/pdf/2409.00265.pdf)
[20](https://superagi.com/mastering-explainable-ai-in-2025-a-beginners-guide-to-transparent-and-interpretable-models/)
[21](https://www.sciencedirect.com/science/article/abs/pii/S0957417425003422)
[22](https://www.nature.com/articles/s41467-022-31384-3)
[23](https://christophm.github.io/interpretable-ml-book/tree.html)
[24](https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability)
[25](https://www.kjas.or.kr/journal/view.html?uid=211&vmd=Full)
[26](https://www.pnas.org/doi/10.1073/pnas.2310151122)
[27](https://arxiv.org/html/2507.23665v1)
[28](https://www.sciencedirect.com/science/article/pii/S277266222300070X)
[29](https://ieeexplore.ieee.org/document/10151849/)
[30](https://cdnsciencepub.com/doi/10.1139/cjce-2023-0410)
[31](https://www.mdpi.com/2076-3417/14/14/6042)
[32](https://link.springer.com/10.1007/s42107-024-01230-6)
[33](https://onlinelibrary.wiley.com/doi/10.1155/2024/8857453)
[34](http://medrxiv.org/lookup/doi/10.1101/2024.10.27.24316222)
[35](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/biot.202400212)
[36](https://ashpublications.org/blood/article/144/Supplement%201/106/530382/Machine-Learning-Based-Prediction-of-One-Year)
[37](https://papers.phmsociety.org/index.php/phmconf/article/view/4194)
[38](https://eaapublishing.org/journals/index.php/hb/article/view/2028)
[39](https://arxiv.org/html/2409.12578v1)
[40](http://arxiv.org/pdf/2402.05566.pdf)
[41](https://arxiv.org/pdf/2211.14797.pdf)
[42](https://arxiv.org/abs/1802.03888)
[43](https://arxiv.org/pdf/2207.14490.pdf)
[44](https://arxiv.org/html/2504.03773v1)
[45](http://arxiv.org/pdf/2401.12069.pdf)
[46](https://www.linkedin.com/pulse/feature-interaction-detection-shap-revealing-complex-between-foster-scfxe)
[47](https://wikidocs.net/178818)
[48](https://www.nature.com/articles/s41598-023-46930-2)
[49](https://github.com/shap/shap/issues/777)
[50](https://pair.withgoogle.com/explorables/grokking/)
[51](https://proceedings.mlr.press/v151/amoukou22a/amoukou22a.pdf)
[52](https://arxiv.org/html/2502.14177v1)
[53](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/EECS-2019-102.pdf)
[54](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
[55](https://liner.com/ko/review/explanations-of-blackbox-models-based-on-directional-feature-interactions)
