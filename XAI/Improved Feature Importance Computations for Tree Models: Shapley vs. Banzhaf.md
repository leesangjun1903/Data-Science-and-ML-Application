# Improved Feature Importance Computations for Tree Models: Shapley vs. Banzhaf

### 1. 논문의 핵심 요약

본 논문(Karczmarz et al., 2021)은 트리 앙상블 모델의 특성 중요도(feature importance) 계산에 있어 Shapley values와 Banzhaf values를 비교 분석하는 연구이다. **핵심 주장은 Banzhaf values가 Shapley values보다 더 직관적이고 계산 효율적이며 수치적으로 안정적이면서도, 실무에서는 본질적으로 동일한 특성 순서를 제공한다는 것이다.** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6f61fbac-cff1-40dd-9d44-52825e1e7934/2108.04126v1.pdf)

**주요 기여는 3가지이다:**

1. **알고리즘 개선**: Shapley values 계산을 O(TLD²+n)에서 O(TLD+n)으로 개선(D배 가속화), Banzhaf values는 최적의 O(TL+n) 시간에 계산 가능하게 함

2. **이론적 증명**: 단조함수에 대해 Shapley와 Banzhaf 값이 동일한 특성 순서를 제공함을 수학적으로 증명

3. **실증적 검증**: 6개 데이터셋에서 Banzhaf의 우수성(계산 속도, 수치 안정성)을 광범위하게 입증

***

### 2. 해결하는 문제와 제안하는 방법

#### 2.1 배경 및 문제점

기계학습 모델의 해석성(explainability)은 의료 진단, 금융 승인 등 고위험 애플리케이션에서 필수적이다. Shapley values는 게임 이론의 개념을 기반으로 하여 합리적인 특성 기여도를 제공하지만, **2가지 핵심 문제가 있다:**

- **계산복잡도**: Naive 계산은 O(2^n)의 기하급수적 복잡도를 가지며, Lundberg et al.(2020)의 TreeSHAP PATH 알고리즘도 O(TLD²+n)으로 깊은 트리에서 병목

- **수치 안정성**: 깊이 ~50인 트리부터 부동소수점 오차가 feature ordering 자체를 변경할 정도로 심각함

#### 2.2 Shapley와 Banzhaf의 정의

**Shapley value** (특성 i에 대해):

$$\phi_i = \frac{1}{n} \sum_{S \subseteq U \setminus \{i\}} \binom{n-1}{|S|}^{-1} (g(S \cup \{i\}) - g(S))$$

여기서 g(S)는 특성 집합 S의 효용함수이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6f61fbac-cff1-40dd-9d44-52825e1e7934/2108.04126v1.pdf)

**Banzhaf value** (특성 i에 대해):

$$\beta_i = \frac{1}{2^{n-1}} \sum_{S \subseteq U \setminus \{i\}} (g(S \cup \{i\}) - g(S))$$

**핵심 차이**: Shapley는 각 coalition S에 대해 크기에 따른 가중치 $\binom{n-1}{|S|}^{-1}$를 적용하나, Banzhaf는 모든 coalition에 동일 가중치를 부여한다.

#### 2.3 트리 모델에서의 계산

트리 앙상블 모델 f(x)에 대해 특성 집합 S의 효용을 근사하는 함수:

$$g(S) = \sum_{l \in L(T)} P[l, S] \cdot f(l)$$

여기서 P[l, S]는 알고리즘 1(subtree coverage)로 계산되는 가중치로, $E[f(x_S, X_{U\S})]$ 를 근사한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6f61fbac-cff1-40dd-9d44-52825e1e7934/2108.04126v1.pdf)

#### 2.4 개선된 알고리즘의 핵심 아이디어

**동적 계획법 기반 상태 전이:**

상태 벡터를 정의하여 인접한 상태 간 O(|G|) 시간에 전이:

$$\Psi(v, G) = (\phi(v, G, k))_{k=0}^{|G|}$$

Shapley의 경우, 특성 y를 추가할 때 (Lemma 4):

$$\phi(v, G \cup \{y\}, k) = \frac{|G|+1-k}{|G|+2} \phi(v, G, k) + \frac{k}{|G|+2} \frac{[x_y \in I_{v,y}]}{c_v(y)} \phi(v, G, k-1)$$

Banzhaf의 경우는 더 단순 (Lemma 6):

$$\beta(v, G \cup \{y\}) = \frac{1}{2}\left(1 + \frac{[x_y \in I_{v,y}]}{c_v(y)}\right) \beta(v, G)$$

**Bottom-up 최적화:**

기본 알고리즘이 각 leaf l에서 모든 부분집합 F_l\{i}를 계산하여 O(LD²) 시간을 소비하는 문제를 해결하기 위해, 트리의 각 노드 v에서 같은 특성으로 분할하는 leaf 그룹 L_v를 정의하고 한 번에 처리:

$$\phi_i = \sum_{v \in T: d_v = i} \left(\frac{[x_i \in I_{v,i}]}{c_v(i)} - 1\right) \cdot \Phi^{-}(v)$$

여기서 $\Phi^{-}(v) = \sum_{l \in L_v} f(l) \cdot \phi(l, F_l \setminus \{d_v\})$를 보조 벡터로 유지하여 O(LD) 시간에 계산.

***

### 3. 모델 구조 및 성능 특성

#### 3.1 알고리즘 복잡도 비교

| 알고리즘 | 복잡도 | 특징 |
|---------|--------|------|
| Naive Shapley 계산 | O(2^n · TL) | 실제 불가능 |
| TreeSHAP PATH (기존) | O(TLD² + n) | Shapley values만 지원 |
| TreeSHAP Fast (제안) | O(TLD + n) | **D배 개선** |
| Banzhaf (제안) | **O(TL + n)** | **최적, 추가 D배 개선** |

#### 3.2 수치 안정성 분석

실제 깊이가 깊은 트리에서 발생하는 부동소수점 누적 오차:

**Synthetic Sparse 사례** (깊이 50의 unbalanced 트리):
- Shapley values의 오차: ~1.0 (예측 값 범위 내에서 완전히 쓸모 없는 수준)
- Banzhaf values의 오차: <0.01 (무시할 수 있는 수준)

**실제 데이터**:
- NHANES GBDT (깊이 4): Shapley 구현 간 top 10 특성 ordering 차이 발생
- Health Insurance DT (깊이 60): Shapley와 Banzhaf의 ordering 대폭 상이 (수치 오차가 주 원인)

#### 3.3 특성 순서의 일치성

| 데이터셋 | 상위 3개 | 상위 10개 | 상위 20개 |
|---------|---------|----------|----------|
| Boston GBDT | 0.02 | 1.05 | - |
| Health Insurance GBDT | 0.02 | 0.73 | - |
| Flights GBDT | 0.23 | 3.08 | 8.63 |

**Cayley distance 분석**: 상위 n개 특성을 Shapley에서 Banzhaf로 변환하기 위해 필요한 평균 교환 수. 대부분의 경우 수용 가능한 수준.

#### 3.4 실행 속도 개선

| 데이터셋 | Ban 시간 | TreeSHAP Fast | TreeSHAP 원본 | 개선율 |
|---------|---------|--|--|--|
| Flights GBDT | 13m 18s | 1h 47m | 1h 50m | **8배** |
| Flights DT | 14m 28s | 5h 23m | 5h 9m | **22배** |

***

### 4. 일반화 성능 향상과의 연결

#### 4.1 이론적 근거: 단조함수에서의 동치성

**정리 (4절)**: 균일 분포 하이퍼큐브 {0,1}^k에서 정의된 단조 함수 f에 대해, weight function w가 만족하는 모든 power index 중:

$$\Omega_i^w = \sum_{x \in D} |\omega_i^w(x)| = \frac{1}{2} \sum_{x \in D} |f(x) - f(x^{\neg i})|$$

는 w와 무관하게 동일하다. 따라서 **global importance에서 $\Phi_i = B_i$**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6f61fbac-cff1-40dd-9d44-52825e1e7934/2108.04126v1.pdf)

**실무적 의미**:
- 실제 데이터가 균일 분포에 가까울수록(많은 특성 조합 존재) Shapley와 Banzhaf의 일치도 상승
- 이는 대규모 테이블 데이터에서 두 방법이 근본적으로 동등함을 시사

#### 4.2 특성 선택(Feature Selection)으로의 확장

더 빠르고 안정적인 Banzhaf 계산은 다음을 가능하게 한다:

1. **반복적 특성 선택**: 각 반복에서 모든 특성의 중요도를 빠르게 재계산
2. **앙상블 안정성**: 부트스트랩 표본마다 특성 순서 비교로 안정성 평가
3. **신뢰 구간 계산**: 여러 표본의 중요도 분포에서 uncertainty 추정

#### 4.3 모델 일반화에 미치는 영향

특성 중요도의 **수치 안정성 개선**은:
- **Feature selection 신뢰도**: 중요도 ranking이 일관성 있게 유지되어 선택된 특성 부분집합의 안정성 향상
- **Transfer learning**: 원본 데이터에서 학습한 특성 순서가 새로운 데이터에도 일관성 있게 적용될 가능성 높음
- **하이퍼파라미터 최적화**: 특성 공학 단계에서 더 신뢰할 수 있는 피드백 제공

***

### 5. 한계 및 제약조건

#### 5.1 이론적 한계

1. **효율성 공리 위반**: Banzhaf는 efficiency axiom (기여도 합 = 전체 기여도 변화)을 만족하지 않음
   - 이는 비용 분배 문제에서는 문제가 될 수 있으나, 특성 중요도 설명에는 미미한 영향
   
2. **단조성 가정**: 4절의 동치 정리는 단조 함수에만 적용
   - 실제 트리는 단조가 아닐 수 있음 (예: 상호 작용 효과)
   
3. **독립성 가정**: g(S) 근사(Algorithm 1)는 특성 독립성을 가정
   - 특성 상관이 높은 경우 오차 증가 가능

#### 5.2 실험적 한계

1. **데이터셋 규모**: 4개 실제 + 2개 합성 데이터셋만 평가
2. **모델 제약**: 트리 기반 모델만 적용 (신경망, 선형 모델 제외)
3. **깊이 분석 부족**: Extreme depth (>100)는 합성 데이터에서만 테스트
4. **다중 비교 미적용**: 많은 dataset-algorithm 조합에서 유의성 검증 부재

#### 5.3 실무적 고려사항

- **설명의 해석성**: Banzhaf가 더 직관적이지만, 효율성 위반으로 설명 방식 조정 필요
- **기존 도구 호환성**: SHAP 라이브러리는 Shapley 기반으로 설계되어 Banzhaf 통합에 비용 필요
- **규제 환경**: 금융/의료에서 "공식적" 방법 사용 요구 가능

***

### 6. 2020년 이후 최신 관련 연구

#### 6.1 Shapley Values 개선 및 확장

**1) 정확한 Shapley 계산** (Amoukou et al., 2023) [proceedings.mlr](https://proceedings.mlr.press/v151/amoukou22a/amoukou22a.pdf)
- Categorical 변수의 encoding 불변성(invariance principle) 증명
- TreeSHAP가 종속 특성에서 편향될 수 있음 보여줌
- 새로운 추정자 제안 (Leaf estimator): exponential in depth instead of width

**2) 고차 상호작용** (Beyond TreeSHAP, 2024) [arxiv](http://arxiv.org/pdf/2401.12069.pdf)
- Shapley interactions: 특성 조합의 영향 정량화
- TreeSHAP PATH 기반 효율적 계산 알고리즘
- 개별 데이터 포인트에서 상호작용 시각화

**3) 이론적 보증** (Unified Framework, 2025) [arxiv](https://arxiv.org/abs/2506.05216)
- KernelSHAP의 첫 번째 이론적 수렴 보증 제시
- Model-agnostic 추정자들의 variance-bias tradeoff 분석
- MNIST, CIFAR10 고차원 데이터에서 검증

#### 6.2 Banzhaf Values의 활용 확대

**1) Data Banzhaf** (Wang et al., 2023) [proceedings.mlr](https://proceedings.mlr.press/v206/wang23e/wang23e.pdf)
- Banzhaf value를 데이터 가치 평가에 적용
- Stochastic 알고리즘에서 Shapley보다 우수한 견고성(robustness)
- Noisy label detection에서 superior performance
- Maximum Sample Reuse (MSR) 추정자: $O(nm)$ 시간, Shapley보다 $m^2$ 배 빠름

**2) 신경망 기반 추정** (InfluenceNet, 2024) [arxiv](https://arxiv.org/html/2503.08381v1)
- 신경망으로 Banzhaf/Shapley-Shubik power index 근사
- 대규모 coalition (n≥10)에서 기존 방법보다 빠름
- Multi-agent system 분석의 확장성 개선

**3) 계층적 투표 게임** (Hierarchical Voting, 2025) [arxiv](https://arxiv.org/pdf/2501.06871.pdf)
- Multiplicative BPI decomposition: $O(d^2K)$ (K = branching factor)
- 깊은 트리/계층 구조에서 Banzhaf 계산 가속화
- 어휘 선택(NLP) 응용 시연

#### 6.3 특성 중요도의 메타 연구

**1) 종합 리뷰** (Shapley Value in Data Science, 2025) [mdpi](https://www.mdpi.com/2227-7390/13/10/1581)
- Model-agnostic 근사: Random Order Value, LASSO-based 값, multilinear extension sampling
- 최신 확장: Distributional Shapley, Weighted Shapley, reinforcement learning 응용
- 현실 응용: 보건, 금융, 산업, 디지털 경제
- 향후 방향: 데이터 자산 가격 책정

**2) ShapleyVIC - 변동성 정량화** (Shapley Variable Importance Cloud, 2022-2024) [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2666389922000253)
- Nearly optimal models 간 특성 중요도 변동성 명시적 정량화
- SAGE (Shapley Additive Global Impact) 기반 global importance
- 부트스트랩으로 신뢰 구간 계산
- SHAP과 seamless integration으로 local + global + across-models 설명

**3) ShapG - 그래프 기반 특성 중요도** (2025) [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197625004099)
- 특성 간 correlation을 그래프로 표현
- Model-agnostic 전역 설명 방법
- 특성 의존성 구조 명시화

#### 6.4 실제 의료/과학 응용 (2024-2025)

**임상 응용**:
- XGBoost+SHAP으로 급성 신부전(AKI) 예측 및 해석 [jtd.amegroups](https://jtd.amegroups.com/article/view/99317/html)
- 신경망 seizure detection에서 SHAP로 뇌파 특성 가시화 [arxiv](https://arxiv.org/html/2601.05095v1)
- 췌장암 예측에서 explainability를 regulatory compliance의 핵심으로 인정 [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/41479795/)

**생물학 응용**:
- QTL mapping: SHAP-XGBoost로 epistasis 검출 [biorxiv](http://biorxiv.org/lookup/doi/10.1101/2024.01.15.575690)
- CRISPR 가이드 RNA 설계: SHAP로 각 염기 위치의 중요도 매핑 [arxiv](https://www.arxiv.org/pdf/2508.20130.pdf)

**환경/자원**:
- 지하수 잠재력 매핑: CatBoost+SHAP (AUC 0.8778) [mdpi](https://www.mdpi.com/2073-4441/17/10/1520)
- 실시간 스트림: KernelSHAP 개선으로 시계열 데이터 적용 [arxiv](https://arxiv.org/pdf/2210.02176.pdf)

***

### 7. 논문의 영향과 미래 연구 방향

#### 7.1 핵심 기여의 현장 영향

1. **계산 효율성**: D배 가속화는 대규모 데이터셋에서 실시간 설명 가능하게 함
   - 의료: 신환 입원 시 수초 내 위험 인수 설명
   - 금융: 대출 승인 거부 이유 즉시 설명

2. **수치 안정성**: Banzhaf의 우수성이 깊은 트리 모델(depth >20)의 신뢰성 확보
   - XGBoost는 기본 depth 6-10이지만, 복잡 데이터는 depth 20-50 사용
   - 이런 모델에서 Shapley는 위험, Banzhaf가 안전

3. **이론-실무 연결**: 단조함수 동치성 정리가 언제 두 방법이 동등한지 명확히 함

#### 7.2 앞으로의 연구 방향

**A. 이론적 확장**

1. **비단조 함수 분석**: 
   - 현실 트리(상호작용 효과)에서 Shapley-Banzhaf 차이 정량화
   - 차이가 언제 무시할 수 있는지 경계 설정

2. **특성 상관 처리**:
   - Conditional expectation 기반 g(S) 정의에서 Banzhaf 성능
   - Copula 기반 특성 생성에서의 안정성 분석

3. **모델 불확실성**:
   - 앙상블 모델(Random Forest, GBDT)에서 트리 간 특성 중요도 분산
   - Bayesian 프레임워크에서의 Banzhaf 해석

**B. 알고리즘 혁신**

1. **GPU/병렬 계산**:
   - O(TL+n) Banzhaf를 GPU의 massive parallelism과 결합
   - 수백만 행 데이터셋에서 밀리초 단위 계산

2. **Approximate Banzhaf**:
   - Variance reduction 기법으로 sampling 기반 추정자 개발
   - Approximation quality vs. computational cost tradeoff 분석

3. **온라인/스트리밍 업데이트**:
   - 새로운 트리 추가 시 특성 중요도 incrementally 업데이트
   - Concept drift 환경에서의 안정성

**C. 응용 확장**

1. **비트리 모델 지원**:
   - 신경망에 대한 Banzhaf 계산 (현재는 TreeSHAP 같은 최적화 없음)
   - 선형/일반화선형 모델에서의 효율적 계산

2. **인과추론 통합**:
   - Causal Shapley values (Heskes et al., 2020+)와 Banzhaf 비교
   - Counterfactual 설명과의 연결

3. **공정성 분석**:
   - 특성 중요도로 demographic parity 위반 감지
   - Bias mitigation 전략의 효과 평가에 Banzhaf 적용

**D. 안정성 및 신뢰성**

1. **오차 경계 이론**:
   - Banzhaf가 깊이 D인 트리에서 보장하는 오차 bound 도출
   - 수치 정밀도(float32 vs float64)에 따른 threshold

2. **부트스트랩 기반 신뢰 구간**:
   - 각 특성의 중요도에 대한 95% CI 계산
   - Hypothesis testing (특성 A > 특성 B?)

3. **설명의 견고성**:
   - Adversarial perturbation에 대한 특성 순서 안정성
   - Anchors의 규칙 기반 설명과 연계

***

### 8. 결론

본 논문은 **트리 앙상블 모델의 특성 중요도 계산에서 실무적으로 더 나은 대안을 제시한다.** Shapley values의 이론적 우월성(efficiency axiom)에도 불구하고, Banzhaf values는:

1. **계산 효율**: O(TL+n) 최적 복잡도로 10배 이상 빠름
2. **수치 안정성**: 깊은 트리에서 부동소수점 오차 거의 없음  
3. **실무 동등성**: 단조함수와 실제 데이터에서 동일한 특성 순서 제공
4. **직관성**: "특성이 추가될 때 예측 변화 확률"이라는 명확한 해석

**특히 의료, 금융 등 설명 가능성이 중요한 고위험 분야에서, Banzhaf는 속도와 안정성 측면에서 Shapley를 대체하거나 보완할 강력한 도구가 될 수 있다.**

향후 연구는 GPU 기반 병렬화, 신경망 확장, 인과추론 통합, 그리고 설명의 견고성 분석으로 진행될 것으로 예상된다.

***

### 참고문헌

 Karczmarz, A., Mukherjee, A., Sankowski, P., & Wygocki, P. (2021). "Improved Feature Importance Computations for Tree Models: Shapley vs. Banzhaf." arXiv:2108.04126v1. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6f61fbac-cff1-40dd-9d44-52825e1e7934/2108.04126v1.pdf)

 Zhang, C., et al. (2025). "Machine learning-based model for the prediction of acute kidney injury." Journal of Translational Data. [jtd.amegroups](https://jtd.amegroups.com/article/view/99317/html)

 φ-test: Global Feature Selection. (2025). Semanticscholar. [semanticscholar](https://www.semanticscholar.org/paper/cbd6c7e616ac02f493b9151169f55527ab221223)

 A Unified Framework for Provably Efficient Algorithms. (2025). arXiv:2506.05216. [arxiv](https://arxiv.org/abs/2506.05216)

 Applying gradient tree boosting to QTL mapping. (2025). bioRxiv. [biorxiv](http://biorxiv.org/lookup/doi/10.1101/2024.01.15.575690)

 Groundwater Potential Mapping. (2025). Water Journal. [mdpi](https://www.mdpi.com/2073-4441/17/10/1520)

 The Shapley Value in Data Science. (2025). MDPI Mathematics. [mdpi](https://www.mdpi.com/2227-7390/13/10/1581)

 Beyond TreeSHAP: Efficient Computation. (2024). arXiv:2401.12069. [arxiv](http://arxiv.org/pdf/2401.12069.pdf)

 Feature Importance for Time Series Data. (2022). arXiv:2210.02176. [arxiv](https://arxiv.org/pdf/2210.02176.pdf)

 Accurate Shapley Values for Tree Models. (2023). PMLR. [proceedings.mlr](https://proceedings.mlr.press/v151/amoukou22a/amoukou22a.pdf)

 InfluenceNet: AI Models for Banzhaf and Shapley. (2024). arXiv:2503.08381. [arxiv](https://arxiv.org/html/2503.08381v1)

 New feature importance method based on Shapley value. (2025). Engineering. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197625004099)

 Data Banzhaf Framework. (2023). PMLR. [proceedings.mlr](https://proceedings.mlr.press/v206/wang23e/wang23e.pdf)

 Advanced Multimodal Learning for Seizure Detection. (2024). arXiv:2601.05095. [arxiv](https://arxiv.org/html/2601.05095v1)

 Banzhaf Power in Hierarchical Voting Games. (2025). arXiv:2501.06871. [arxiv](https://arxiv.org/pdf/2501.06871.pdf)

 AI for CRISPR Guide RNA Design. (2025). arXiv:2508.20130. [arxiv](https://www.arxiv.org/pdf/2508.20130.pdf)

 Explainable AI in Pancreatic Cancer. (2025). PubMed. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/41479795/)

 Shapley Variable Importance Cloud. (2022-2024). ScienceDirect. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2666389922000253)
