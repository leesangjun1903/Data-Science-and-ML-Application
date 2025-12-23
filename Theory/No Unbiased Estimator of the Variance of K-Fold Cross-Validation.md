# No Unbiased Estimator of the Variance of K-Fold Cross-Validation

### 1. 핵심 요약: Bengio & Grandvalet (2004)

**"No Unbiased Estimator of the Variance of K-Fold Cross-Validation"**은 머신러닝에서 가장 보편적으로 사용되는 모델 평가 기법인 K-fold CV의 근본적 한계를 수학적으로 증명한 고전적 연구입니다. 이 논문의 가장 중요한 기여는 **Theorem 6**으로, 모든 데이터 분포에 대해 K-fold CV 분산의 보편적(universal) 불편 추정량이 존재하지 않음을 증명했습니다.[1]

#### 핵심 주장
논문은 다음을 보여줍니다:
- K-fold CV는 unbiased 추정량이지만, 그 분산 추정은 본질적으로 편향됨
- 이는 훈련 세트와 테스트 세트 간의 겹침으로 인한 오류의 상관구조 때문
- 순진한 추정자(naive estimator)는 이 상관성을 무시하여 실제 분산의 30-50%만 추정

### 2. 문제 진술과 제안 방법

#### 2.1 K-fold CV의 수학적 모델

표준 K-fold CV 추정량:[1]

$$\widehat{CV}(D) = \frac{1}{K}\sum_{k=1}^{K}\frac{1}{m}\sum_{z_i \in T_k} L(A(D_k), z_i)$$

여기서:
- $D = \{z_1, ..., z_n\}$: 원본 데이터 (크기 $n$)
- $T_k$: $k$번째 테스트 블록 (크기 $m = n/K$)
- $D_k$: $T_k$를 제외한 훈련 세트
- $L$: 손실 함수

#### 2.2 분산 구조의 핵심 발견

**Lemma 1-3**: 공분산 행렬 $\Sigma = \text{Cov}(e_i, e_j)$의 블록 구조는 3개의 파라미터만으로 완전히 결정됨:[1]

$$\Theta = \frac{1}{n^2}\sum_{i,j}\text{Cov}(e_i, e_j) = \frac{1}{n}\sigma^2 + \frac{m-1}{n}\omega + \frac{n-m}{n}\gamma$$

여기서:
- **$\sigma^2$**: 분산 성분 (같은 테스트 블록 내 독립)
- **$\omega$**: 같은 블록 내 공분산 (훈련 세트 겹침으로 인함)
- **$\gamma$**: 다른 블록 간 공분산 (테스트 블록이 모든 훈련 세트에 포함되므로)

#### 2.3 불편성 불가능성 증명

**Theorem 6**의 증명 아이디어:[1]

2차 형식의 모든 불편 추정량은 다음 형태:

$$\widehat{\theta} = a s_1 + b s_2 + c s_3$$

여기서 $s_1, s_2, s_3$은 불변 통계량:

$$s_1 = \frac{1}{n}\sum_{i=1}^{n}e_i^2$$

$$s_2 = \frac{1}{n(m-1)}\sum_{k=1}^{K}\sum_{\substack{i,j \in T_k \\ i \neq j}} e_i e_j$$

$$s_3 = \frac{1}{n(n-m)}\sum_{k,\ell \neq k}\sum_{\substack{i \in T_k \\ j \in T_\ell}} e_i e_j$$

불편성 조건:

$$a(σ^2 + μ^2) + b(ω + μ^2) + c(γ + μ^2) = \frac{1}{n}\sigma^2 + \frac{m-1}{n}\omega + \frac{n-m}{n}\gamma$$

모든 분포에서 성립하려면:[1]

$$a = \frac{1}{n}, \quad b = \frac{m-1}{n}, \quad c = \frac{n-m}{n}, \quad a+b+c = 0$$

**이 네 개의 미지수에 대해 4개의 선형 독립적 제약이 있으므로 유일한 해가 존재하지 않음** → 불편 추정량 불가능.

### 3. 고유값 분해와 일반화 성능의 의미

#### 3.1 공분산 행렬의 스펙트럼 구조

**Lemma 7**: 공분산 행렬 $\Sigma$의 고유값:[1]

$$\lambda_1 = \sigma^2 - \omega \quad \text{(다중도 } n-K\text{)}$$

$$\lambda_2 = \sigma^2 + (m-1)\omega - m\gamma \quad \text{(다중도 } K-1\text{)}$$

$$\lambda_3 = \sigma^2 + (m-1)\omega + (n-m)\gamma \quad \text{(다중도 1)}$$

#### 3.2 일반화 성능 향상의 한계

고유값 분석의 함의:[1]

- **$\lambda_1, \lambda_2$**: $n-K + K-1 = n-1$개의 자유도 → 이들은 독립적 샘플로 추정 가능
- **$\lambda_3$**: 단 1개의 자유도 → CV의 전체 분산은 $\lambda_3/n$에 정확히 비례
- **$\widehat{CV}$는 고유벡터 $\mathbf{1}$에 정사영** → $\lambda_3$에 해당하는 방향이 정확히 평균

결론: **한 번의 K-fold CV 실행만으로는 $\lambda_3$를 추정할 방법이 없음** → 일반화 오류 불확실성을 정확히 정량화 불가.

### 4. 모델 구조와 성능 분석

#### 4.1 실험 설계 (3가지 시나리오)

**Experiment 2**: 합성 데이터의 3가지 분산 성분 분해[1]

1. **이상치 없음 (standard setup)**
   - $y_i = \sqrt{3/d}\sum_{k=1}^{d}x_{ik} + \varepsilon_i$
   - σ²이 지배적 (작은 표본), γ가 중요 (큰 표본)

2. **이상치 포함 (robust case)**
   - $y_i \sim N(0, 100I)$ with probability 0.05
   - $\omega, \gamma$ 모두 σ²과 동등한 크기 → **순진한 추정자의 심각한 과소추정**

3. **실제 분류 문제 (Letter dataset)**
   - σ²이 전체 분산의 50-70%만 설명
   - 나머지 30-50%는 ω와 γ로 설명 → **공분산 항 무시 불가**

#### 4.2 K 선택이 분산 분해에 미치는 영향

**Figure 5**: K=2~100에 따른 σ², ω, γ 기여도 변화[1]

- K↑ (더 많은 작은 폴드):
  - σ² 증가 (각 훈련 세트 작아짐)
  - ω, γ의 상대적 역할 변화
  - 최적 K는 문제마다 다름 (보편적 규칙 없음)

### 5. 2020-2025 최신 연구와의 비교 분석

#### 5.1 Nested Cross-Validation (Bates et al., 2021)[2]

**핵심 발견**: K-fold CV의 정체성 재정의

$$\text{CV는 } Err_{XY} \text{를 추정하지 않고, } Err = E[Err_{XY}] \text{를 추정함}$$

**Theorem 1** (선형 모델 OLS에서):[2]

$$\widehat{Err}^{(CV)} \perp\!\!\perp Err_{XY} \mid X$$

즉, CV 추정량과 실제 모델의 오류는 조건부로 독립 → CV는 다른 훈련 세트에 대한 평균 오류를 추정하는 것.

**Nested CV 해결책**:[2]

```
외부 루프: K-fold split (I_k 고정)
내부 루프: (K-1)-fold CV on (D \ I_k)
```

**Theorem 3** (MSE의 불편 추정량):[2]

$$E\left[\widehat{MSE}_{K,n}\right] = MSE_{K-1, n'}$$

여기서 $n' = n(K-1)/K$.

**성능 개선**: 신뢰구간 커버리지 69% → 94% (로지스틱 회귀, n=90)[2]

#### 5.2 Irredundant K-fold (Aguilar-Ruiz, 2025)[3]

**핵심 혁신**: 훈련 중복 제거로 공분산 구조 변경

표준 kF vs IkF의 비교:[3]

| 특성 | 표준 kF | IkF |
|------|--------|-----|
| 훈련 집합 크기 | $(k-1)n/k$ | $n/k$ |
| 각 샘플 사용 (훈련) | $k-1$번 | 1번 |
| 훈련 세트 겹침 | $(k-2)/k$ (예: 80%, k=10) | 0% |
| 공분산 구조 | 양수 (평가 편향 저하) | 0 (정직한 분산) |

**분산 분해 (Eq. 10)**:[3]

$$Var(\widehat{\theta}) = \frac{1}{k^2}\left[\sum_{i=1}^{k}Var(\widehat{\theta}_i) + 2\sum_{i<j}Cov(\widehat{\theta}_i, \widehat{\theta}_j)\right]$$

- 표준 kF: $Cov(\widehat{\theta}_i, \widehat{\theta}_j) > 0$ → 분산 저평가
- IkF: $Cov(\widehat{\theta}_i, \widehat{\theta}_j) \approx 0$ → 참 분산 반영

**실험 결과 (10개 데이터셋)**:[3]

| 지표 | 표준 kF | IkF | 비율 |
|------|--------|-----|------|
| 평균 정확도 | 87.7% | 85.6% | 1.027 |
| F-score | 86.2% | 82.6% | 1.048 |
| 계산 시간 | 3.71초 | 0.97초 | 2.89배 |

**해석**: 훈련 데이터 감소로 약 2.7% 정확도 저하, 하지만 2.9배 속도 향상 및 **더 신뢰할 수 있는 분산 추정**

#### 5.3 Bootstrap 기반 접근 (Cai et al., 2023)[4]

**아이디어**: Bootstrap으로 CV 추정량의 표준오차 직접 추정[4]

$$SE(CV) = \sqrt{\frac{1}{R-1}\sum_{r=1}^{R}\left(\widehat{CV}_r - \overline{\widehat{CV}}\right)^2}$$

장점:
- 계산 비용이 nested CV보다 낮음
- 다양한 손실 함수에 적용 가능

한계:
- 여전히 편향 존재 (소표본)
- Bengio-Grandvalet의 근본적 불가능성을 극복하지 못함

#### 5.4 다른 접근: PAC-Bayesian 상한선 (Gorriz et al., 2024)[5]

**K-fold Cross Upper Bounding Validation (CUBV)**:[5]

$$R(f_N) \leq R_N(f_N) + \Delta(N,\mathcal{F})$$

여기서 $\Delta$는 집중 부등식 기반 상한선:

$$\Delta(N,\mathcal{F}) = \min_{1 \leq i \leq k}\frac{1}{2\lambda_i - 1}\left[\sqrt{R_N(\omega) + \frac{2\lambda_i^2}{N}D(Q, Q_u)} + \frac{\ln k}{\eta}\right]$$

**장점**: 최악의 경우 오류 경계로 신뢰구간 제공[5]

**한계**: 선형 분류기에만 적용, 매우 보수적 경계

### 6. 일반화 성능 향상과의 관계

#### 6.1 Bengio-Grandvalet의 한계에서 비롯된 문제

모델 선택 문제:

$$\hat{m} = \arg\max_m \widehat{CV}(m)$$

하지만 $\text{SE}(\widehat{CV}(m))$을 정확히 모르면:
- 유의미한 차이인지 확인 불가
- 신뢰 구간 구성 불가
- 하이퍼파라미터 튜닝의 불확실성 정량화 불가

**결과**: 모델 성능 향상의 신뢰도 저하

#### 6.2 Nested CV의 개선

**Bates et al.**의 접근:[2]

1. 외부 CV: 최종 일반화 오류 추정
2. 내부 CV: 하이퍼파라미터 튜닝
3. 중첩 구조: $E_{\text{outer}}[\widehat{CV}_{\text{inner}}]$ 추정 가능

결과:
- Confidence interval coverage: 69% → 94%
- 모델 비교의 신뢰도 향상

#### 6.3 IkF의 관점

**Aguilar-Ruiz**의 주장:[3]

공분산 제거로 분산 추정이 정직해짐:

$$\text{Bias-Variance Tradeoff 더 명확하게 해석 가능}$$

일반화 성능 평가:
- **낙관 편향 감소**: 과소평가 가능성 있음 (약 2-3%)
- **신뢰성 향상**: 안전-중심 애플리케이션에 적합 (의료, 금융)

### 7. 향후 연구 시 고려할 점

#### 7.1 이론적 확장

1. **비선형 모델**: 현재 이론은 선형 OLS에 중점
   - 신경망, SVM 등에 대한 공분산 구조 분석 필요
   - 알고리즘 안정성과의 관계 규명

2. **고차원 설정**: Bengio-Grandvalet은 $p$ 고정, $n \to \infty$만 고려
   - 비율 점근 ($n/p \to \lambda > 1$) 분석 필요
   - Random projection의 영향 분석

3. **적응적 공분산 추정**: 현재 모든 $\omega, \gamma$ 동등하게 취급
   - 블록별 이질성 모델링
   - 동적 K 선택 전략

#### 7.2 실무 적용

1. **하이퍼파라미터 선택**
   - K의 최적값: 현재 k=5 또는 10 관례 → 문제별 최적화 필요
   - IkF와 nested CV의 계산 비용 대비 성능

2. **신뢰도 정량화**
   - 모델 평가의 신뢰 구간 표준화
   - 다양한 데이터 특성에 따른 지침 수립

3. **응용 분야별 고려**
   - **의료/금융**: 보수적 평가 → IkF 권장
   - **하이퍼파라미터 튜닝**: 계산 효율성 → IkF
   - **모델 비교**: 정확한 불확실성 → Nested CV

#### 7.3 기술적 개선

1. **병렬화 전략**
   - Nested CV의 내부 루프 병렬 처리
   - 다중 GPU 환경에서의 최적화

2. **메모리 효율성**
   - IkF의 소규모 훈련 세트 활용
   - 스트리밍 데이터에 대한 확장

3. **통합 프레임워크**
   - CV 기반 모델 선택 + 신뢰도 평가 통합
   - AutoML에 불확실성 정량화 포함

### 결론

Bengio & Grandvalet (2004)의 근본적 발견—**K-fold CV 분산에 대한 보편적 불편 추정량이 존재할 수 없다**—는 20년 후에도 여전히 유효하며, 이를 극복하기 위한 다양한 접근이 제시되었습니다.

**Nested Cross-Validation** (Bates et al., 2021)은 중첩 구조로 다중 실현을 확보하여 MSE의 불편 추정량을 제공하지만, 계산 비용이 약 10배 증가합니다.

**Irredundant K-fold** (Aguilar-Ruiz, 2025)는 훈련 중복을 제거하여 공분산 구조 자체를 변경함으로써 분산 추정을 정직하게 하고, 2.9배의 계산 속도 향상을 달성합니다. 이는 안전-중심 애플리케이션에서 모델 성능의 신뢰성을 높입니다.

**일반화 성능 향상의 관점**에서:
- 표준 CV로는 신뢰 구간을 정확히 구성할 수 없으므로 모델 선택의 신뢰도가 낮음
- Nested CV나 IkF를 통해 불확실성을 명시적으로 정량화하면 신뢰할 수 있는 모델 개선이 가능
- 문제의 특성(표본 크기, 데이터 이질성, 안정성 요구)에 따라 방법 선택이 필수

향후 연구는 비선형 모델에 대한 이론적 확장, 실무 데이터에서의 공분산 구조 적응적 학습, 그리고 AutoML과의 통합을 통해 신뢰할 수 있는 기계학습 평가 체계를 완성해야 합니다.

***

### 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/14e61fe7-8dc8-4d9f-82af-0789284e1ed0/grandvalet04a.pdf)
[2](https://arxiv.org/pdf/2307.00260.pdf)
[3](https://arxiv.org/html/2507.20048v1)
[4](https://arxiv.org/pdf/2104.00673v2.pdf)
[5](https://arxiv.org/pdf/2401.16407.pdf)
[6](https://ojs.serambimekkah.ac.id/jnkti/article/view/9128)
[7](https://www.semanticscholar.org/paper/2f1d0e071300ec8092a3d488e37a3fb9508dbe11)
[8](https://arxiv.org/abs/2508.01325)
[9](https://www.mdpi.com/2075-4418/15/14/1794)
[10](https://onlinelibrary.wiley.com/doi/10.1111/sjos.12777)
[11](https://www.semanticscholar.org/paper/3999bb53985803194e8560d2a410336f84258127)
[12](http://nzjforestryscience.nz/index.php/nzjfs/article/view/49)
[13](https://essd.copernicus.org/articles/17/2873/2025/)
[14](http://biorxiv.org/lookup/doi/10.1101/2025.11.14.688354)
[15](https://link.springer.com/10.1007/s00477-025-03041-w)
[16](https://www.frontiersin.org/articles/10.3389/fpls.2021.734512/pdf)
[17](https://figshare.com/articles/journal_contribution/Cross-validation_what_does_it_estimate_and_how_well_does_it_do_it_/22512229/1/files/39974221.pdf)
[18](http://arxiv.org/pdf/1812.10820.pdf)
[19](https://arxiv.org/pdf/1904.02438.pdf)
[20](https://arxiv.org/pdf/1706.05801.pdf)
[21](https://arxiv.org/pdf/2511.12698.pdf)
[22](https://arxiv.org/pdf/2403.20313.pdf)
[23](https://arxiv.org/html/2407.02754v1)
[24](https://arxiv.org/pdf/2507.20048.pdf)
[25](https://pmc.ncbi.nlm.nih.gov/articles/PMC10612467/)
[26](https://pmc.ncbi.nlm.nih.gov/articles/PMC11412612/)
[27](https://www.sciencedirect.com/science/article/abs/pii/S030147972200442X)
[28](https://www.sciencedirect.com/topics/computer-science/unbiased-estimator)
[29](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
[30](http://ieeexplore.ieee.org/document/8698831/)
[31](https://arxiv.org/html/2510.08359)
[32](https://arxiv.org/pdf/2509.12406.pdf)
[33](https://arxiv.org/pdf/2507.06266.pdf)
[34](https://arxiv.org/html/2507.01970v1)
[35](https://arxiv.org/html/2508.11004v2)
[36](https://arxiv.org/html/2409.03697v1)
[37](https://arxiv.org/html/2509.24877v1)
[38](https://arxiv.org/pdf/2505.21444.pdf)
[39](https://www.biorxiv.org/content/10.1101/2025.05.16.654565v1.full-text)
[40](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0315955)
[41](http://cicip.sxu.edu.cn/docs/2019-04/3727948ecb644b0c8053fa03d15803d1.pdf)
[42](https://onlinelibrary.wiley.com/doi/10.1111/mice.12992)
[43](https://www.ijsat.org/papers/2025/1/1305.pdf)
[44](https://www.tandfonline.com/doi/abs/10.1080/02331888.2025.2581835)
