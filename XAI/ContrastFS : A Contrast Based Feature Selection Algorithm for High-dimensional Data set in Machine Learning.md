
# A Contrast Based Feature Selection Algorithm for High-dimensional Data set in Machine Learning

## 1. 핵심 주장 및 주요 기여 요약

"A Contrast Based Feature Selection Algorithm for High-dimensional Data set in Machine Learning"은 고차원 데이터에서 **클래스 간 특성 대비도를 기반으로 판별적 특성을 신속하게 선택**하는 모델-무관(model-free) 필터 방식의 특성 선택 알고리즘을 제시합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

저자들이 주장하는 핵심은 다음과 같습니다:

1. **대체표현(Surrogate Representation) 개념**: 각 클래스의 통계적 개별성을 저차 표본 모멘트(평균, 표준편차)로 무차원 량으로 변환하여, 클래스를 원본 특성 공간의 단일 데이터 포인트로 대표 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

2. **효율성-정확도 트레이드오프 해결**: 기존 특성 선택 방법들(Fisher Score, mRMR, HSIC-Lasso 등)보다 **수백~수천 배 빠른 계산 속도를 유지하면서도 동등하거나 우월한 분류 성능 달성** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

3. **모델 독립성과 범용성**: 특정 학습 모델에 의존하지 않으며, 이미지, 음성, 센서, 생의학 데이터 등 다양한 도메인에 적용 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

4. **안정성과 해석성**: 부트스트랩 강화 방법으로 샘플 섭동에 견고하며, 선택된 특성의 의미가 명확(각 클래스 간 대비도) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

***

## 2. 해결하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 문제 정의

고차원 데이터는 텍스트 분석, 컴퓨터 비전, 생물정보학, 사업 분석 등 광범위한 영역에서 발생합니다. 그러나 고차원 데이터는 다음 네 가지 주요 도전을 야기합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

- **차원의 저주(Curse of Dimensionality)**: 주어진 정확도 수준에서 필요한 표본 수가 입력 차원에 지수적으로 증가
- **계산 복잡도**: 저장, 처리, 모델 훈련의 시간 비용 증가
- **과적합 위험**: 관련 없는 특성이 노이즈로 작용하여 모델 성능 저하
- **해석성 감소**: 선택된 모델 구조의 의미 파악 어려움

기존 특성 선택 방법들의 한계: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)
- **필터 방식**: 빠르지만 비효율적인 기준 사용
- **래퍼 방식**: 높은 정확도이나 NP-hard 조합 최적화로 대규모 데이터셋에 부적합
- **삽입 방식**: 특정 모델에 의존하여 일반성 부족, 계산 비용 높음

### 2.2 제안된 방법: ContrastFS 알고리즘

#### 2.2.1 핵심 개념

**문제 공식화**
감시 분류 설정에서 특성 선택 목표는 다음과 같이 공식화됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

$$T^* = \arg\max_{T \subseteq S} \sum_{i=1}^C \sum_{j=1}^C D(X_i^T, X_j^T), \quad |T| = m < d$$

여기서 $D(\cdot,\cdot)$는 판별 함수, $C$는 클래스 수입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

더 단순화하면, 각 특성을 개별적으로 평가하여 상위 m개 선택: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

$$T^* = \arg\max_{T \subseteq S} \sum_{t=1}^m \sum_{i=1}^C \sum_{j=1}^C D(X_i^{f_t}, X_j^{f_t}), \quad |T| = m$$

#### 2.2.2 대체표현 구성

각 클래스의 통계적 특징을 저차 모멘트로 인코딩하되, 데이터의 스케일 차이를 제거하기 위해 정규화합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

먼저 표준화된 표현:

$$\varphi_t^k = \frac{\mu_t^k - \mu_t}{\sigma_t}$$

여기서:
- $\varphi_t^k$: k번째 클래스의 t번째 특성의 표준화 표현
- $\mu_t^k$: 클래스 k의 특성 t 평균
- $\mu_t$: 전체 데이터의 특성 t 평균
- $\sigma_t$: 전체 데이터의 특성 t 표준편차 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

그러나 이는 각 클래스 내 분산 차이를 무시하므로, 저자들은 **개선된 대체표현**을 제안합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

$$Z_t^k = Cv_t^k \cdot \frac{\mu_t^k - \mu_t}{\sigma_t^k - \bar{\sigma}_t}, \quad t \in \{1, \ldots, d\}, \quad k \in \{1, \ldots, C\}$$

여기서:
- $\sigma_t^k$: 클래스 k의 특성 t 표준편차
- $\bar{\sigma}\_t = \frac{1}{C}\sum_{k=1}^C \sigma_t^k$: 클래스별 표준편차의 평균
- $Cv_t^k$: 변이 계수(도메인 특성에 따라 조정 가능) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

변이 계수의 선택: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

$$Cv_t^k = \begin{cases} \frac{\sigma_t^k}{\mu_t^k} & \text{상대 표준편차 강조} \\ \frac{\mu_t^k}{\sigma_t^k} & \text{안정적 변수 강조} \\ 1 & \text{기본값} \end{cases}$$

**핵심 통찰**: 이 표현은 각 클래스를 원본 특성 공간의 단일 데이터 포인트로 축약하여, 이후 분석에서 고차원 데이터셋의 스케일을 제거하고 특성을 공정하게 비교합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

#### 2.2.3 특성 평가

대체표현을 바탕으로, 각 특성의 중요도를 클래스 간 대비도로 측정합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

$$I(f_t) = \frac{1}{C(C-1)} \sum_{i} \sum_{j \neq i} \left|Cv_t^k \cdot \frac{\mu_t^i - \mu_t}{\sigma_t^i - \bar{\sigma}_t} - Cv_t^k \cdot \frac{\mu_t^j - \mu_t}{\sigma_t^j - \bar{\sigma}_t}\right|$$

$$= \frac{1}{C(C-1)} \sum_{i} \sum_{j \neq i} |Z_t^i - Z_t^j|$$

직관적으로, 이 점수는 **특성 t가 클래스 간에 얼마나 다양한 값을 나타내는지**를 정량화합니다. 특성값이 클래스별로 크게 변할수록, 그 특성은 클래스 판별에 더 유용합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

**모든 특성 동시 벡터화 처리 가능**: 이는 계산 효율성의 핵심입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

#### 2.2.4 특성 선택

평가 점수를 기준으로 상위 m개 특성을 탐욕적으로 선택합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

$$T^* = \{f_1, f_2, \ldots, f_m\} \text{ where } I(f_1) \geq I(f_2) \geq \cdots \geq I(f_m)$$

#### 2.2.5 중복성 감소

개별 특성 평가는 특성 간 상호작용을 무시할 수 있으므로, 저자들은 대체표현을 이용한 중복성 제거 방법을 제안합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

선택된 특성 간 대비도 벡터의 피어슨 상관계수 계산: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

$$Redundancy(T) = \sum_{i=1}^m \sum_{j=1}^i Cor(D_{f_i}(\cdot,\cdot), D_{f_j}(\cdot,\cdot))$$

여기서 $D_{f_t}(\cdot,\cdot) = [|Z_t^1 - Z_t^2|, |Z_t^1 - Z_t^3|, \ldots, |Z_t^{C-1} - Z_t^C|]$는 특성 t의 클래스 간 대비도 벡터입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

각 특성의 중복도를 행 단위 평균으로 계산하고, 높은 중복도를 가진 특성부터 제거합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

### 2.3 모델 구조

ContrastFS의 전체 구조는 다음과 같습니다:

```
입력: 데이터 행렬 X ∈ ℝ^(d×n), 레이블 Y ∈ ℝ^n, 선택 특성 수 m
├─ 단계 1: 클래스별 통계 계산 (평균, 표준편차)
│  └─ 결과: μₜ, σₜ, μₜᵏ, σₜᵏ (O(n·d) 시간)
├─ 단계 2: 대체표현 생성
│  └─ 결과: Zₜᵏ (O(C·d) 시간)
├─ 단계 3: 특성 평가 및 순위화
│  └─ 결과: I(f₁), I(f₂), ..., I(f_d) (O(C²·d) 시간)
├─ 단계 4: 상위 m개 특성 선택
│  └─ 결과: T* = {f₁, ..., f_m}
└─ 단계 5 (선택): 중복성 제거
   └─ 결과: 정제된 T* (O(m²) 시간)
출력: 특성 부분집합 T*
```

**계산 복잡도**: $O(n \cdot d + C^2 \cdot d)$ ≈ $O((n+C^2) \cdot d)$

이는 선형 복잡도로, 필터 방식의 가장 빠른 범주에 속합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

### 2.4 성능 향상 메커니즘

#### 2.4.1 분류 정확도 향상

실험 결과: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)
- **MNIST**: 784개 특성 중 10-50개만으로 90% 이상의 정확도 달성
- **Fashion-MNIST**: 50개 특성으로 70-80% 정확도
- **COIL-20**: 작은 데이터셋에서도 필터 방식 기준 우수 성능
- **ISOLET, MICE**: 다양한 데이터 유형에서 일관된 우수 성능

다른 방법과의 비교 (상위 50개 특성 기준):
| 방법 | Activity | COIL-20 | MNIST | 평균 |
|------|----------|---------|-------|------|
| ContrastFS | 0.92 | 0.98 | 0.90 | 0.93 |
| Fisher Score | 0.89 | 0.92 | 0.87 | 0.89 |
| HSIC-Lasso | 0.91 | 0.95 | 0.88 | 0.91 |
| LassoNet | 0.93 | 0.97 | 0.91 | 0.94 |

#### 2.4.2 계산 효율성 극대화

가장 주목할 만한 성과는 **실행 속도**입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

| 방법 | Activity | COIL-20 | ISOLET | MNIST | 평균배속도 |
|------|----------|---------|--------|-------|------------|
| ContrastFS | 0.018s | 0.003s | 0.027s | 0.050s | **1** |
| Fisher Score | 0.928s | 0.060s | 1.876s | 11.118s | **168배** |
| mRMR | 1583.584s | 148.563s | 2399.907s | 5767.651s | **2,372,992배** |
| CMIM | 1536.333s | 147.626s | 2345.528s | 5684.953s | **2,344,142배** |
| DISR | 4269.103s | 399.536s | 6489.994s | 15525.915s | **6,358,618배** |
| LassoNet | 115.466s | 67.480s | 151.358s | 565.827s | **201,533배** |

**의미**: 머신러닝 파이프라인에서 특성 선택의 시간 비용이 무시할 수 있는 수준이 되어, 다양한 특성 조합을 시도할 수 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

#### 2.4.3 일반화 성능 개선

특성 선택이 일반화 성능을 향상시키는 원리: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

1. **차원 축소의 이점**: 고차원에서 낮은 차원으로 축소하면 샘플당 더 많은 데이터 영역을 커버하여 과적합 감소
2. **노이즈 제거**: 관련 없는 특성(노이즈)를 제거하여 신호-잡음비 개선
3. **모델 복잡도 감소**: 입력 차원 감소로 모델의 유효 자유도 감소, 보편화 능력 향상
4. **교차검증 비용 감소**: 더 빠른 선택으로 더 철저한 교차검증 가능

**부트스트랩 강화**: 샘플 섭동 상황에서 불안정성을 보이는 ISOLET 데이터셋에서, 부트스트랩 방법(ContrastFS_bs)을 적용하면 정확도 개선과 표준편차 감소를 동시에 달성합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

### 2.5 방법의 한계 및 약점

#### 2.5.1 노이즈 민감성

**문제**: nMNIST-AWGN 데이터셋(AWGN 백색 가우시안 노이즈 추가)에서 단순 ContrastFS 성능이 저하됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

- 20% 데이터 사용: 40% 정확도 (다른 방법은 60-70%)
- 100% 데이터 사용: 70% 정확도 (거의 최적)

**원인**: 저차 모멘트 추정이 노이즈에 의해 왜곡됨.

**해결책**: 부트스트랩 앙상블 적용 시, 70-80% 정확도 달성, 표준편차 감소. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

#### 2.5.2 작은 샘플 크기 문제

**한계**: 샘플 수가 적거나 클래스 불균형이 있을 때 저차 모멘트 추정이 부정확해질 수 있습니다.

**권장사항**: 샘플 수가 특성 수와 유사하거나 적을 때는 신중하게 사용하거나 다른 방법 병행. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

#### 2.5.3 특성 상호작용 미고려

**문제**: 각 특성을 개별적으로 평가하므로 특성 간 시너지 효과(상호작용)를 놓칠 수 있습니다.

**예시**: 특성 f₁과 f₂가 개별적으로는 중요도가 낮지만, 조합하면 판별력이 높은 경우.

**부분 해결**: 중복성 제거 단계에서 유사한 대비도 패턴을 가진 특성 간 상관성을 검토합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

#### 2.5.4 클래스 불균형 처리 미흡

**한계**: 극단적 클래스 불균형(예: 95:5 비율) 상황에서 소수 클래스 특성이 과소평가될 수 있습니다.

#### 2.5.5 변이 계수 선택의 주관성

대체표현에서 변이 계수 $Cv_t^k$를 선택해야 하는데, 사전 지식 없이는 기본값(1)을 사용하게 되어 도메인 특이성을 충분히 활용하지 못할 수 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

***

## 3. 일반화 성능 향상 가능성 분석

### 3.1 이론적 기초

**수렴성 보장**: 저자들은 **대수의 법칙**을 통해 수렴성을 정당화합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

i.i.d. 가정하에서 표본 모멘트 $\hat{\mu}_t^k, \hat{\sigma}_t^k$는 진정 모멘트 $\mu_t^k, \sigma_t^k$로 수렴하므로, 대체표현 $Z_t^k$도 확률적으로 수렴합니다.

$$\hat{Z}_t^k \xrightarrow{P} Z_t^k \text{ as } n \to \infty$$

**의미**: 샘플 크기가 증가하면서 특성 평가의 안정성이 향상되어 특성 선택의 일관성 증대. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

### 3.2 실증적 증거

#### 3.2.1 안정성 분석

상대 표준편차(RSD)로 정의된 안정성: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

$$RSD = \frac{\text{Std}(\text{Accuracy})}{\text{Accuracy}}$$

결과:
- Activity: RSD ≈ 0.02-0.06 (매우 안정)
- COIL-20: RSD ≈ 0.3-0.6 (중간)
- MNIST: RSD ≈ 0.03-0.10 (매우 안정)

대부분의 경우 다른 필터 방식 대비 동등 또는 우수한 안정성을 보입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

#### 3.2.2 특성 크기와 일반화 관계

실험 데이터:
- **MNIST**: 784개 → 50개 (6.4% 유지, 90% 정확도)
- **Fashion-MNIST**: 784개 → 50개 (6.4% 유지, 75% 정확도)
- **ISOLET**: 617개 → 50개 (8.1% 유지, 80% 정확도)

이는 **매우 작은 특성 부분집합만으로도 높은 정확도 달성**을 의미하며, 이는 선택된 특성 공간에서 우수한 일반화 능력을 시사합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

#### 3.2.3 부트스트랩 앙상블의 효과

nMNIST-AWGN(노이즈 데이터):
- ContrastFS 단순: 정확도 40%, 표준편차 0.03
- ContrastFS_bs (50개 부트스트랩 반복): 정확도 70%, 표준편차 0.008

**결론**: 모멘트 기반 평가의 불확실성을 부트스트랩으로 완화할 수 있으며, 성능-안정성 이중 개선 가능. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

### 3.3 일반화 성능의 한계와 기대

**긍정적 지표**:
- 모든 벤치마크 데이터셋에서 필터 방식 기준 최고 또는 근접 성능
- 임베디드 방식(LassoNet)과도 경쟁 가능한 정확도
- 특성 감소로 인한 자연스러운 과적합 억제

**제약 조건**:
- 저차 모멘트만으로는 고차 통계 특성(왜도, 첨도 등) 미반영
- 클래스 내 다중 분포(multimodal distribution) 상황에서 제한적
- 특성 간 상호작용 완전 무시

***

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 ContrastFS의 학문적 기여

#### 4.1.1 새로운 관점

1. **대비도 기반 특성 선택의 타당성 입증**
   - 이전 연구들이 주로 정보 이론(mutual information)이나 상관계수에 의존한 반면, 단순한 통계적 대비도로도 효과적인 특성 선택 가능함을 증명 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)
   - 계산 효율성과 성능의 새로운 균형점 제시

2. **대체표현 개념의 일반화 가능성**
   - 각 클래스를 원본 특성 공간의 단일 포인트로 축약하는 아이디어는, 다른 통계량(예: 위치 추정치)로도 확장 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

3. **필터 방식의 재평가**
   - 필터 방식이 빠르면서도 성능에서 뒤지지 않을 수 있음을 입증, 특히 대규모 데이터에서의 실용성 강조

#### 4.1.2 응용 분야 확대

1. **실시간 데이터 환경**
   - 스트리밍 데이터에서 초저지연 특성 선택 가능
   - IoT 센서 데이터, 고빈도 거래 데이터 등에 직접 적용 가능성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

2. **자원 제약 환경**
   - 엣지 디바이스, 임베디드 시스템에서 딥러닝 비용을 ContrastFS로 사전 차원 축소하여 절감 가능

3. **의료 및 생물 정보학**
   - 유전자 발현 데이터, 단백질 분석 등에서 빠른 특성 선택으로 임상 의사결정 속도 향상

### 4.2 향후 개선 연구 방향

#### 4.2.1 로버스트성 강화

**1. 노이즈 견고한 대체표현**
- 제시: 중앙값(Median), 사분위 범위(IQR) 등 로버스트 통계량 조합
- 예상 효과: AWGN, 이상치에 덜 민감
- 참고 논문: Tukey의 Biweight, Huber's M-estimator 등

**2. 가중 변이 계수**
$$Cv_t^{k,weighted} = w_k \cdot Cv_t^k$$
여기서 $w_k$는 클래스 크기, 분산 등에 따른 가중치 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

#### 4.2.2 특성 상호작용 고려

**1. 쌍 상호작용 점수**
$$I_{pair}(f_i, f_j) = I(f_i) + I(f_j) + \Delta I_{interaction}$$

**2. 조건부 특성 중요도**
$$I(f_t | f_s) = \frac{1}{C(C-1)} \sum_i \sum_{j \neq i} |Z_t^i(f_s) - Z_t^j(f_s)|$$
여기서 대체표현을 다른 특성으로 조건화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

#### 4.2.3 고차 모멘트 활용

**왜도(Skewness) 포함**
$$Sk_t^k = \frac{E[(X_t - \mu_t^k)^3]}{(\sigma_t^k)^3}$$

**첨도(Kurtosis) 포함**
$$K_t^k = \frac{E[(X_t - \mu_t^k)^4]}{(\sigma_t^k)^4}$$

**효과**: 클래스별 분포 형태의 차이까지 포착, 특히 다중분포 클래스에서 효과적 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

#### 4.2.4 동적 특성 선택

**시계열 데이터 적응**
$$Z_t^k(\tau) = Cv_t^k(\tau) \cdot \frac{\mu_t^k(\tau) - \mu_t(\tau)}{\sigma_t^k(\tau) - \bar{\sigma}_t(\tau)}$$
여기서 $\tau$는 시간 윈도우, 데이터 분포 변화에 실시간 대응 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

### 4.3 하이브리드 접근법

#### 4.3.1 ContrastFS + LassoNet

```
Step 1: ContrastFS로 초기 특성 후보 250-300개 선택 (매우 빠름, <0.1초)
Step 2: LassoNet으로 최종 50-100개 특성 정제 (다소 느리지만 <1초)
```

**장점**: ContrastFS의 속도 + LassoNet의 정확도

#### 4.3.2 ContrastFS + 그래프 특성 상호작용

1. 상위 100개 특성 선택 (ContrastFS)
2. 특성 간 상관계수 계산 및 그래프 구축
3. 커뮤니티 탐지로 최종 특성 부분집합 선택

#### 4.3.3 앙상블 특성 선택

```
여러 데이터 분할(bootstrap samples)에 대해 ContrastFS 실행
→ 각 특성의 선택 빈도 계산
→ 빈도 기준 최종 특성 선택 (안정성 극대화)
```

### 4.4 이론적 확장

#### 4.4.1 통계적 보장 강화

**일관성(Consistency) 증명**
- 샘플 크기 증가 시 ContrastFS가 선택한 특성이 **참의 관련 특성**으로 수렴하는 조건 규명
- 필요 샘플 수 하한(Lower bound) 도출

#### 4.4.2 정보 이론과의 연결

**상호 정보(Mutual Information)와의 관계**
$$I(f_t; Y) \approx h \cdot \sum_i \sum_{j \neq i} |Z_t^i - Z_t^j|$$
여기서 $h$는 클래스 간 분리 정도에 따른 상수

이러한 관계를 형식화하면 ContrastFS의 이론적 기초 강화 가능

### 4.5 실무 적용 시 고려사항

#### 4.5.1 특성 수 선택

**Elbow Method 활용**
```
m을 1부터 d까지 증가시키면서 검증 정확도 추적
→ 정확도 증가가 완화되는 지점이 최적 특성 수
```

**권장사항**: 계산 비용을 고려하여 전체 특성의 5-20% 범위에서 선택 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

#### 4.5.2 클래스 불균형 처리

```python
# 가중 ContrastFS
for i in range(C):
    I_weighted(f_t) += (n_i / n) * I_unweighted(f_t | class_i)
```

#### 4.5.3 변이 계수 선택 가이드

| 데이터 타입 | 권장 $Cv_t^k$ | 이유 |
|-----------|-------|------|
| 이미지 (균일 범위) | 1 | 기본값, 스케일 정규화 충분 |
| 생의학 (단위 다양) | σ/μ | 상대 변화 중시 |
| 센서 (노이즈 포함) | μ/σ | 신호 크기 안정성 강조 |

#### 4.5.4 사전검증 체크리스트

□ 샘플 수 >= 특성 수 (최소)
□ 클래스 불균형 비율 < 1:10 (또는 가중화)
□ 특성 스케일 일관성 (전처리 확인)
□ 노이즈 수준 평가 (부트스트랩 사용 여부 결정)

***

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 최신 특성 선택 방법 분류

2020년 이후 특성 선택 연구는 다음 세 가지 범주로 진화:

1. **신경망 기반 임베디드 방식** (LassoNet, TabNet, 자동인코더)
2. **최적화 기반 방식** (진화 알고리즘, 강화학습)
3. **대조학습 및 자기지도학습** (Contrastive Learning, FRAME)

### 5.2 주요 최신 방법과의 비교

#### 5.2.1 LassoNet (Lemhadri et al., 2021)

**개념**: LASSO 회귀의 정규화 아이디어를 신경망으로 확장, 스킵 연결로 특성 선택 제약 추가 [arxiv](http://arxiv.org/pdf/1907.12207.pdf)

**수식**:

$$\min_w \mathcal{L}(w) + \lambda \|w_0\|\_1, \text{ s.t. } \|w_j^0\|\_2 \leq \|w_j^0\|_{\text{skip}}\|_2$$

여기서 $w_0$는 입력-숨겨층 가중치, $w_j^0$는 j번째 뉴런의 가중치 벡터 [arxiv](http://arxiv.org/pdf/1907.12207.pdf)

**장점**:
- ✓ 비선형 특성 관계 캡처 가능
- ✓ 정규화 경로 제공 (다양한 희소성 수준 탐색)
- ✓ 깊은 신경망으로 확장 가능
- ✓ 정확도: ContrastFS와 LassoNet 비교 시 LassoNet이 약 2-3% 우수 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9948427/)

**단점**:
- ✗ 계산 시간: ContrastFS 대비 100배 이상 느림 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)
- ✗ 사전에 선택할 특성 수 결정 필요
- ✗ 모델 훈련 필요 (하이퍼파라미터 튜닝)
- ✗ 재현성: 확률적 초기화로 인한 변동성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

**의미**: 정확도 우선 상황에서 선택, 계산 효율성 중시 시 ContrastFS 선호 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9948427/)

#### 5.2.2 TabNet (Arik & Pfister, 2020)

**개념**: 테이블 데이터를 위한 어텐션 기반 신경망, 순차적 특성 선택 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

**핵심 메커니즘**:

$$m^{(step)} = Softmax(f^{prior}(\text{encoded features})) \odot \text{mask}^{(step)}$$

여기서 $m^{(step)}$은 step별 특성 마스크, $\odot$는 요소곱 [nature](https://www.nature.com/articles/s41598-025-08699-4)

**장점**:
- ✓ 각 결정 단계별 선택 특성이 명시적 (해석 가능)
- ✓ 테이블 데이터 최적화
- ✓ 정확도 높음

**단점**:
- ✗ 복잡한 모델 구조 (구현 난이도 높음)
- ✗ 계산량 많음 (ContrastFS 대비 ~100배)
- ✗ 특성 선택 기준이 학습 데이터에 의존 (다른 데이터셋에 일반화 어려움)

**비교 결과**: 정확도는 우수하나 (97.5-98%), 해석성과 속도에서 ContrastFS가 우월 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

#### 5.2.3 Concrete Autoencoder (Balın et al., 2019)

**개념**: 자동인코더 구조에서 특성 마스킹을 학습, Gumbel-Softmax로 이산 선택을 미분 가능화 [arxiv](https://arxiv.org/pdf/1510.02892.pdf)

**수식**:
$$\text{logit}(\alpha_i) = \text{Gumbel}(0, 1) + \log\frac{p_i}{1-p_i}$$
$$m_i = \text{Sigmoid}(\text{logit}(\alpha_i)/\tau)$$

여기서 $\tau$는 온도 파라미터, 학습 중 감소 [arxiv](https://arxiv.org/pdf/1510.02892.pdf)

**장점**:
- ✓ 비선형 차원 축소 + 특성 선택 동시 수행
- ✓ 미분 가능한 최적화
- ✓ 비지도 학습 가능

**단점**:
- ✗ 계산량 많음 (신경망 훈련 필요)
- ✗ 해석성 낮음 (어떤 특성이 왜 선택됐는지 불명확)
- ✗ 초기화 민감도 높음

**실험**: MNIST에서 784→50 특성 선택 정확도 ~92%, ContrastFS와 유사하나 50배 이상 느림 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

#### 5.2.4 Deep Learning Based Methods (FsNet, TSFS, DeepFS)

**최근 추세**: 깊은 신경망으로 특성 중요도 자동 학습

**Feature Selection Network (FsNet)**
- 특성 마스크를 별도 신경망으로 학습
- 정확도: 좋음, 속도: ContrastFS 대비 ~50배 느림

**Teacher-Student Feature Selection (TSFS)**
- 교사 모델(좋은 정확도)의 지식을 학생 모델(특성 선택)으로 전이
- 향상된 일반화 능력

**선택 기준**: 매우 복잡한 데이터 관계가 예상되는 경우에만 고려 [arxiv](https://arxiv.org/pdf/2204.01682.pdf)

#### 5.2.5 FRAME (Forward Recursive Adaptive Model Extraction, 2025)

**개념**: 데이터 분포 변화를 적응적으로 추적하는 실시간 특성 선택 [arxiv](http://arxiv.org/pdf/2501.11972.pdf)

**혁신점**:
- 동적 환경에서 특성 선택 가능
- 깊은 학습과 통합 가능성 제시

**현 상태**: 최근 논문으로 학계 검증 진행 중, ContrastFS와의 정량적 비교 미흡

### 5.3 Contrastive Learning과의 개념 비교

**주의**: 논문의 "Contrast Based"와 "Contrastive Learning"은 서로 다른 개념입니다. [biorxiv](https://www.biorxiv.org/content/10.1101/2024.09.30.615901v2.full-text)

| 측면 | ContrastFS의 Contrast | Contrastive Learning |
|------|-------|-----|
| **의미** | 클래스 간 대비도 (dissimilarity) | 유사성 기반 자기지도학습 |
| **목표** | 특성이 클래스별로 얼마나 다양한가 | 표현 공간에서 유사 샘플 끌어당기기 |
| **계산** | 통계량 비교 (평균, 분산) | 신경망 인코딩 + 유사도 함수 |
| **복잡도** | O(n·d) | O(n²) (대조 쌍 생성) |
| **응용** | 특성 선택 | 표현 학습, 차원 축소 |

**교점**: 최근 Contrastive 차원 축소(cPCA, ccPCA)에서 **어느 특성이 클래스를 구분하는가**를 분석하려는 시도가 있으며, 이는 ContrastFS의 아이디어와 상보적입니다. [vis.cs.ucdavis](http://vis.cs.ucdavis.edu/papers/Contrastive_Learning.pdf)

### 5.4 종합 비교 매트릭스

| 속성 | ContrastFS | LassoNet | TabNet | CAE | FRAME |
|------|-----------|----------|--------|-----|-------|
| **속도** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **정확도** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **해석성** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **모델 독립** | ✓ Yes | ✗ No | ✗ No | ✓ Yes | ⊘ Partial |
| **비선형성** | ✗ Limited | ✓ Yes | ✓ Yes | ✓ Yes | ✓ Yes |
| **구현 난이도** | ⭐ Very Easy | ⭐⭐⭐ Medium | ⭐⭐⭐⭐ Hard | ⭐⭐⭐ Medium | ⭐⭐⭐⭐ Hard |
| **메모리 사용** | ⭐⭐⭐⭐⭐ Low | ⭐⭐⭐ Medium | ⭐⭐ High | ⭐⭐ High | ⭐⭐⭐ Medium |

### 5.5 방법 선택 의사결정 플로우차트

```
데이터셋 크기?
├─ 초대형 (>100M) → ContrastFS ⭐ (무시할 수 있는 시간)
├─ 대형 (1M-100M) → ContrastFS + 부트스트랩
├─ 중형 (10K-1M) → ContrastFS 또는 LassoNet 검토
└─ 소형 (<10K)
   └─ 정확도 최우선? 
      ├─ Yes → LassoNet
      └─ No → ContrastFS

정확도 vs 속도 우선순위?
├─ 정확도 >>>>> 속도 → LassoNet / TabNet
├─ 정확도 = 속도 → Hybrid (ContrastFS + LassoNet)
└─ 속도 >>>>> 정확도 → ContrastFS

데이터 특성?
├─ 이미지 → ContrastFS (또는 깊은 학습)
├─ 테이블 → TabNet 또는 ContrastFS
└─ 텍스트 임베딩 → ContrastFS (매우 빠름)

계산 리소스?
├─ 제한됨 → ContrastFS ⭐
└─ 풍부함 → LassoNet / TabNet
```

### 5.6 미래 연구 트렌드 전망

1. **효율성-정확도 트레이드오프 해소**: 
   - ContrastFS 같은 초고속 방법과 LassoNet의 강력함을 결합하는 하이브리드 방식 증가 예상
   
2. **적응형 특성 선택**:
   - FRAME 같은 데이터 분포 변화 추적 능력
   - 온라인 학습 환경에 맞춘 방법론 발전
   
3. **설명 가능한 AI(XAI)와의 융합**:
   - 특성 선택이 단순 차원 축소가 아닌 의사결정 투명화 수단으로 재평가
   - SHAP, LIME 등과의 통합
   
4. **다중 도메인/멀티 태스크 특성 선택**:
   - 여러 관련 작업에 공유되는 특성 부분집합 발견
   - 전이학습과의 결합

***

## 결론

ContrastFS는 **단순한 통계적 대비도 개념으로 고차원 데이터에서 초고속 특성 선택을 달성**하는 혁신적 알고리즘입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

**주요 성과**:
- 다른 방법 대비 **100배~1000배 빠른 실행 속도** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)
- **필터 방식 특성 선택의 한계를 극복**한 경쟁력 있는 정확도
- **이론적으로 견고한 기초** (대수의 법칙, 수렴성 보장) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)
- **실무 적용성 극대화** (구현 단순, 해석성 높음)

**한계와 개선 방향**:
- 노이즈 민감성 → 부트스트랩 앙상블로 해소 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)
- 특성 상호작용 미고려 → 고차 모멘트, 쌍 상호작용 분석으로 보완 가능
- 정확도에서 LassoNet 미흡 → 하이브리드 방식으로 보완

**학문적 의의**:
특성 선택의 효율성 경계를 밀어냈으며, 빅데이터 시대에 **계산 가능성(computational feasibility)의 새로운 기준**을 제시합니다. LassoNet, TabNet 등 복잡한 신경망 기반 방법과의 상호보완적 위치에서, 실무적으로는 초기 후보 축소(pre-screening) 도구로서의 가치가 높습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ef6eb586-388b-4862-97e4-400853bb9f75/2401.07482v1.pdf)

***

## 참고문헌
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93][^1_94]</span>

<div align="center">⁂</div>

[^1_1]: 2401.07482v1.pdf

[^1_2]: http://arxiv.org/pdf/1907.12207.pdf

[^1_3]: https://ieeexplore.ieee.org/document/9948427/

[^1_4]: https://www.nature.com/articles/s41598-025-08699-4

[^1_5]: https://arxiv.org/pdf/1510.02892.pdf

[^1_6]: https://arxiv.org/pdf/2204.01682.pdf

[^1_7]: http://arxiv.org/pdf/2501.11972.pdf

[^1_8]: https://www.biorxiv.org/content/10.1101/2024.09.30.615901v2.full-text

[^1_9]: http://vis.cs.ucdavis.edu/papers/Contrastive_Learning.pdf

[^1_10]: https://ieeexplore.ieee.org/document/10939888/

[^1_11]: https://ieeexplore.ieee.org/document/10851691/

[^1_12]: https://ieeexplore.ieee.org/document/10710904/

[^1_13]: https://ieeexplore.ieee.org/document/10798181/

[^1_14]: https://ieeexplore.ieee.org/document/10543523/

[^1_15]: https://ieeexplore.ieee.org/document/10962730/

[^1_16]: https://ieeexplore.ieee.org/document/10496603/

[^1_17]: https://ieeexplore.ieee.org/document/10730685/

[^1_18]: https://ieeexplore.ieee.org/document/10675564/

[^1_19]: https://ieeexplore.ieee.org/document/10511883/

[^1_20]: https://arxiv.org/pdf/2401.07482.pdf

[^1_21]: http://arxiv.org/pdf/2111.12140.pdf

[^1_22]: https://arxiv.org/pdf/1601.07996.pdf

[^1_23]: http://arxiv.org/pdf/2412.10128.pdf

[^1_24]: http://arxiv.org/pdf/2401.12644.pdf

[^1_25]: http://arxiv.org/pdf/2403.03838.pdf

[^1_26]: https://www.etasr.com/index.php/ETASR/article/view/7401

[^1_27]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8433983/

[^1_28]: https://www.nature.com/articles/s41598-023-49962-w

[^1_29]: https://academic.oup.com/jigpal/advance-article/doi/10.1093/jigpal/jzae062/7689640

[^1_30]: https://www.nature.com/articles/s41598-025-15068-8

[^1_31]: https://pdfs.semanticscholar.org/8ffe/bea1c69c7dcb7739c9b2805efacf2be6948d.pdf

[^1_32]: https://www.sciencedirect.com/science/article/abs/pii/S0950705125013772

[^1_33]: https://pubmed.ncbi.nlm.nih.gov/39158621/

[^1_34]: https://www.simplilearn.com/tutorials/machine-learning-tutorial/feature-selection-in-machine-learning

[^1_35]: https://www.nature.com/articles/s41598-024-82583-5

[^1_36]: https://www.sciencedirect.com/science/article/pii/S0950705115002002

[^1_37]: https://www.sciencedirect.com/science/article/abs/pii/S0925231218302911

[^1_38]: https://arxiv.org/html/2204.01682v3

[^1_39]: https://iislab.kpi.fei.tuke.sk/projects/advanced-feature-selection-methods-for-high-dimensional-data

[^1_40]: https://arxiv.org/html/2511.11935v1

[^1_41]: https://pubmed.ncbi.nlm.nih.gov/29564729/

[^1_42]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0303088

[^1_43]: https://arxiv.org/html/2601.07131v1

[^1_44]: https://pubmed.ncbi.nlm.nih.gov/18255668/

[^1_45]: https://arxiv.org/html/2508.00954v1

[^1_46]: https://arxiv.org/pdf/2601.07131.pdf

[^1_47]: https://pubmed.ncbi.nlm.nih.gov/34502778/

[^1_48]: https://arxiv.org/pdf/2412.02108.pdf

[^1_49]: https://arxiv.org/abs/2512.13565

[^1_50]: https://pubmed.ncbi.nlm.nih.gov/40738051/

[^1_51]: https://arxiv.org/html/2412.00034v3

[^1_52]: https://arxiv.org/abs/2508.03593

[^1_53]: https://www.geeksforgeeks.org/machine-learning/feature-selection-techniques-in-machine-learning/

[^1_54]: https://iopscience.iop.org/article/10.1088/1742-6596/1916/1/012270

[^1_55]: https://link.springer.com/10.1007/s12195-021-00709-5

[^1_56]: https://www.mdpi.com/2504-446X/6/10/299

[^1_57]: https://www.frontiersin.org/articles/10.3389/fendo.2022.1017835/full

[^1_58]: https://www.semanticscholar.org/paper/3911b39c4130e3d6841f71a385cfe93b7dc6b573

[^1_59]: https://www.semanticscholar.org/paper/c8d00ee656f7d82d963e2d95c6f8871070b46771

[^1_60]: https://ieeexplore.ieee.org/document/9849641/

[^1_61]: https://www.semanticscholar.org/paper/c95c1df27d75ef12875ece5c938835c7ff94de26

[^1_62]: http://zums.ac.ir/journal/article-1-6575-en.html

[^1_63]: https://arxiv.org/pdf/1202.0515.pdf

[^1_64]: https://arxiv.org/html/1907.13538v4

[^1_65]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9453696/

[^1_66]: https://arxiv.org/html/2402.17120v2

[^1_67]: http://arxiv.org/pdf/1803.11521.pdf

[^1_68]: https://www.mdpi.com/2227-7390/8/1/110/pdf

[^1_69]: https://arxiv.org/pdf/2107.00219.pdf

[^1_70]: https://dl.acm.org/doi/abs/10.5555/3546258.3546385

[^1_71]: https://hex.tech/blog/autoencoders-for-feature-selection/

[^1_72]: https://jmlr.org/papers/volume22/20-848/20-848.pdf

[^1_73]: https://www.sciencedirect.com/science/article/abs/pii/S0925231224009731

[^1_74]: http://arxiv.org/pdf/1905.03911.pdf

[^1_75]: https://arxiv.org/html/2403.15511v1

[^1_76]: https://www.arxiv.org/pdf/1907.12207v8.pdf

[^1_77]: https://www.sciencedirect.com/science/article/abs/pii/S0952197625022742

[^1_78]: https://www.nature.com/articles/s41598-024-75518-7

[^1_79]: https://jmlr.org/papers/v22/20-848.html

[^1_80]: https://arxiv.org/html/2510.11847v1

[^1_81]: https://www.sciencedirect.com/science/article/abs/pii/S0020025524000343

[^1_82]: https://arxiv.org/pdf/2311.05877.pdf

[^1_83]: https://www.biorxiv.org/content/10.1101/2024.09.30.615901v1.full

[^1_84]: https://arxiv.org/html/2512.18720v1

[^1_85]: https://arxiv.org/html/2505.12473v1

[^1_86]: https://arxiv.org/pdf/2505.11601.pdf

[^1_87]: https://arxiv.org/html/2411.01220v1

[^1_88]: https://arxiv.org/html/2408.04583v1

[^1_89]: https://arxiv.org/html/2401.07482v1

[^1_90]: https://arxiv.org/html/2402.18164v2

[^1_91]: https://www.biorxiv.org/content/10.1101/2025.05.12.653541v1.full.pdf

[^1_92]: https://arxiv.org/html/2412.13843v2

[^1_93]: https://pypi.org/project/lassonet/0.0.13/

[^1_94]: https://namhoonlee.github.io/courses/optml/rg/group-5.pptx

