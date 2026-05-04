
# Minimum Description Length Revisited

> **논문 정보**
> - **저자**: Peter Grünwald, Teemu Roos
> - **저널**: *International Journal of Mathematics for Industry*, Vol. 11, No. 1 (2019/2020), Article 1930001
> - **arXiv**: [1908.08484](https://arxiv.org/abs/1908.08484)
> - **DOI**: [10.1142/S2661335219300018](https://doi.org/10.1142/S2661335219300018)

---

## 1. 핵심 주장 및 주요 기여 (Executive Summary)

이 논문은 통계학, 머신러닝, 패턴 인식의 일반적인 문제에 적용할 수 있는 귀납적 추론 이론인 **최소 기술 길이(MDL) 원리**에 대한 최신의 포괄적인 소개 및 개요를 제공한다.

MDL은 원래 데이터 압축 아이디어에 기반하고 있지만, 이 논문은 압축 이론에 대한 사전 지식 없이도 읽을 수 있도록 구성되어 있으며, 2007년 마지막 포괄적 개요가 작성된 이후의 모든 주요 발전을 반영하고 있다.

### 🔑 핵심 주장 요약

주요 기여로는 **모델 선택, 평균화, 가설 검증을 위한 새로운 방법들**과 **MDL 추정기의 최초 완전한 일반적 정의**가 포함된다. 이 발전들을 통합하여, MDL은 페널티 우도(penalized likelihood)와 베이지안 접근법 모두의 강력한 확장으로 볼 수 있으며, 여기서 페널티 함수와 사전 분포는 더 일반적인 **luckiness 함수**로 대체되고, 평균-케이스 방법론은 더 강건한 **최악-케이스 접근법**으로 대체되며, AIC vs BIC 및 교차검증 vs 베이즈와 같이 고전적으로 매우 다르게 보이던 방법들을 통합된 관점에서 바라볼 수 있게 된다.

---

## 2. 해결하고자 하는 문제

### 2-1. 문제 정의

MDL 아이디어는 이론적으로 매우 강력한 것으로 증명되었고 성공적인 실용적 구현 사례도 상당수 존재하지만, 대규모 배포는 두 가지 문제에 의해 방해를 받아왔다. 첫째, MDL을 적용하기 위해 통계학과 정보 이론 두 분야 모두의 기초 지식이 필요했다. 이를 해결하기 위해 본 논문은 처음으로 **정보 이론 없이** MDL 원리를 제시한다.

두 번째 문제는 많은 고전적 MDL 절차들이 계산 집약적이거나(예: MDL 변수 선택), 빅데이터 시대에 덜 적합하다는 것이다.

### 2-2. MDL 원리의 핵심 철학

MDL 원리는 통계학, 머신러닝, 패턴 인식의 일반적인 문제에 적용될 수 있는 귀납적 추론 이론이다. 광의적으로 말하면, **주어진 데이터에 대한 최선의 설명은 그 데이터의 가장 짧은 기술(description)에 의해 제공된다**는 것이다.

---

## 3. 제안하는 방법 및 수식

### 3-1. MDL의 기본 구조: 두 파트 코드 (Two-Part Code)

두 파트 코드는 파라메트릭 모델에 대해 도입 및 분석되었다. 이는 **보편 코드(universal code)**, 또는 동등하게 **보편 확률 분포**의 가장 단순한 사례로서, MDL 이론의 핵심 개념이다.

두 파트 코드에서, 데이터 $z^n$에 대한 총 기술 길이는 다음과 같이 정의된다:

$$L_{\text{two-part}}(z^n, M) = L(\hat{\theta}(z^n)) + L(z^n \mid \hat{\theta}(z^n))$$

여기서:
- $L(\hat{\theta}(z^n))$: 최대 우도 추정치 $\hat{\theta}$의 코드 길이 (모델의 복잡도)
- $L(z^n \mid \hat{\theta}(z^n))$: 추정된 파라미터 하에서 데이터의 코드 길이 (데이터 적합도)

두 파트 코드의 구현은 직관적이다: 후보 모델 집합에서 **모델 코딩 비용**(비트) + **모델이 주어졌을 때 데이터 코딩 비트 수**의 합을 최소화함으로써 모델을 선택한다.

### 3-2. 정규화 최대 우도 (NML: Normalized Maximum Likelihood)

실제로 가장 많이 채택되는 MDL 버전은 **정규화 최대 우도(NML)**로, 효과적인 해법과 우아한 형식론을 제공한다.

NML 분포는 다음 **최소최대 문제(minimax problem)**의 해로 정의된다:

$$p_{\text{NML}}(z^n) = \frac{p(z^n; \hat{\theta}(z^n))}{\int_{y^n} p(y^n; \hat{\theta}(y^n))\, dy^n}$$

여기서 분모는 **파라메트릭 복잡도(Parametric Complexity, COMP)**라고 불린다:

$$\text{COMP}(M) = \log \int_{y^n} p(y^n; \hat{\theta}(y^n))\, dy^n$$

따라서 NML 코드 길이는:

$$L_{\text{NML}}(z^n) = -\log p_{\text{NML}}(z^n) = -\log p(z^n; \hat{\theta}(z^n)) + \text{COMP}(M)$$

### 3-3. 점근적 전개 (Asymptotic Expansion): BIC와의 연결

충분히 정규적인 파라메트릭 모델에 대한 $\text{COMP}(M)$의 점근적 전개는 다음과 같다:

$$\text{COMP}(M) \approx \frac{k}{2} \log \frac{n}{2\pi} + \log \int_{\Theta} \sqrt{\det I(\theta)}\, d\theta$$

여기서 $k$는 파라미터 수, $n$은 데이터 수, $I(\theta)$는 **피셔 정보 행렬(Fisher Information Matrix)**이다.

이 전개의 첫 번째 항 $\frac{k}{2} \log n$이 바로 BIC의 페널티 항에 해당하며, MDL이 BIC를 특수 사례로 포함함을 보여준다.

### 3-4. 베이지안 보편 분포와의 통합

베이지안 보편 분포가 NML의 특수 사례로 볼 수 있다는 사실이 최근에야 완전히 명확해졌으며, 같은 논문에서 베이지안, 두 파트, NML 분포를 모두 특수 사례로 갖는 더 일반적인 공식화가 제시되었다.

### 3-5. 사전 순차적(Prequential) 플러그인 분포

사전 순차적 플러그인 분포는 모델 $M$에 대한 합리적인 추정기 $\hat{\theta}$를 먼저 취한 후, 다음과 같이 정의된다:

$$p_{\text{preq}}^{\hat{\theta}}(z^n) := \prod_{i=1}^{n} p_{\hat{\theta}(z^{i-1})}(z_i \mid z^{i-1})$$

### 3-6. MDL 모델 선택 공식

가산적(countable) 모델 집합 $\{M_\gamma : \gamma \in \Gamma\}$이 주어졌을 때, 각 개별 모델 $M_\gamma$를 단일 분포 $\bar{p}_\gamma$와 연관시키는 아이디어를 활용한다.

이는 MDL에서 모델 선택과 추정이 본질적으로 동일한 것임을 보여준다.

최적 모델은 다음을 통해 선택된다:

$$\hat{\gamma}_{\text{MDL}} = \arg\min_{\gamma \in \Gamma} \left[ -\log \bar{p}_\gamma(z^n) \right]$$

---

## 4. 모델 구조

본 논문이 다루는 MDL의 주요 구성 요소:

| 구성 요소 | 설명 |
|---|---|
| **Two-Part Code** | 파라미터 + 데이터의 직접적 코딩 |
| **NML (Normalized ML)** | 미니맥스 최적 보편 분포 |
| **Prequential Distribution** | 순차적 예측 기반 보편 분포 |
| **Bayesian Universal Distribution** | 사전 분포 기반 혼합 |
| **Luckiness NML** | 가중 NML의 일반화 형태 |

섹션 4에서는 그래프 모델(베이지안 네트워크 등)에 대한 모델 선택을 위한 **NML 유형 분포의 빠른 계산**에 관한 최신 발전을 검토하며, 이는 표준 베이지안 방법보다 실제로 더 강건한 방법들로 이어진다.

불규칙(비지수 족) 모델, 예를 들어 계층적 잠재 변수 모델의 경우, Watanabe가 제안한 **WAIC(Widely Applicable Information Criterion)** 및 **WBIC(Widely Applicable Bayesian Information Criterion)**과의 연결도 논의된다.

---

## 5. 모델의 일반화 성능 향상 가능성 🔍

이것이 본 논문의 가장 중요한 기여 중 하나이다.

### 5-1. PAC-MDL 경계와 딥러닝에의 적용

섹션 6.4 (PAC-MDL 경계와 딥러닝)는 MDL이 딥러닝(수백만 개의 파라미터를 가질 수 있는)에 대해 유용한 직관을 제공함을 보여준다. PAC-베이지안 경계(PAC-Bayesian Bounds)는 **분류기의 일반화 성능이 (a) 파라미터를 기술하는 데 필요한 비트 수가 적을수록, (b) 파라미터가 주어졌을 때 데이터를 기술하는 데 필요한 비트 수가 적을수록 직접적으로 개선**되는 양에 연결될 수 있음을 보여준다.

이를 수식으로 표현하면, PAC-MDL 경계의 일반적 형태는:

$$R(h) \leq \hat{R}(h) + \sqrt{\frac{L_{\text{MDL}}(h) + \ln(1/\delta)}{2n}}$$

여기서 $R(h)$는 진정한 위험(true risk), $\hat{R}(h)$는 경험적 위험, $L_{\text{MDL}}(h)$는 MDL 코드 길이이다.

### 5-2. 지도학습으로의 확장 — MDL에서 LASSO로

섹션 6.2는 지도학습을 완전히 처리할 수 있고 정규성 가정 없는 제곱 오차 및 0/1 손실을 포함한 다양한 손실 함수에 사용할 수 있는 MDL 접근법을 논의한다. 이는 예측기 $f$를 밀도 $p_f(x, y) \propto \exp(-\ell(f(x), y))$와 연관시켜, 이 밀도에 대한 로그 손실이 $(x, y)$에서 $f$의 손실과 선형 관계가 되도록 달성된다.

### 5-3. 모델 오지정(Misspecification) 하에서의 일반화

섹션 6.3은 데이터가 모든 고려된 모델이 틀리지만 일부는 좋은 예측으로 이어지는 분포에서 올 때 무슨 일이 발생하는지를 검토한다. 대부분의 MDL 접근법이 오지정 하에서 수렴을 보여줄 수 없는 이유가 **비초압축 특성(no-hypercompression property)**과 관련이 있음이 밝혀졌다.

### 5-4. 과파라미터화 모델에서의 일반화 (이론적 확장)

복잡도는 일반화 성능에 영향을 주는 통계적 학습 이론의 기본 개념이다. 파라미터 수는 저차원 설정에서는 성공적이지만, 훈련 샘플보다 파라미터가 많은 과파라미터화 설정에서는 잘 정당화되지 않는다. 이에 Rissanen의 MDL 원리에 기반한 복잡도 측도를 재검토하고, **과파라미터화 모델에도 유효한 새로운 MDL 기반 복잡도(MDL-COMP)**를 정의하였다.

이 MDL-COMP는 단순히 파라미터 수나 $d$와 $n$의 단순 함수가 아니라, **공분산 행렬 $X^\top X$의 고유값과 잡음 분산으로 스케일된 실제 파라미터 간의 상호작용**에 따라 달라진다.

---

## 6. 성능 향상 및 한계

### 6-1. 성능 향상

NML 코드 길이는 MDL 원리 하에서 객관적이고 파라미터 없는 모델 선택의 기반을 형성한다. 최대화된 우도(데이터 적합도)와 파라메트릭 복잡도 항을 함께 인코딩함으로써, NML 기준은 **유한 샘플에서 BIC/AIC보다 더 강하게 과유연한 모델을 페널티화하는** 엄격한 오컴의 면도날(Occam's razor)을 구현한다.

고차원 변수 선택에서 MDL 기반 방법은 일관성이 있으며, **강건 및 적응적 라쏘(lasso)를 능가**하고 극단적인 차원에서도 경쟁력을 유지한다 (Wei et al., 2022).

### 6-2. 한계

지난 10년 동안 이러한 문제들을 대부분 해결하는 흥미로운 발전이 있었다. MDL은 페널티 우도와 베이지안 접근법 모두의 강력한 확장으로 볼 수 있지만, 이를 구현하는 실용적 방법들의 통합적 이해가 여전히 진행 중이다.

- **계산 복잡도**: 많은 MDL 절차(특히 NML 계산)가 계산 집약적임
- **COMP의 유한성**: NML 적용을 위해 파라메트릭 복잡도 $\text{COMP}(M)$이 유한해야 하는 제약
- **연속 데이터 처리**: 연속 데이터에서의 NML 정규화는 기하학적 측도 이론을 필요로 함

---

## 7. 앞으로의 연구에 미치는 영향 및 고려할 점

### 7-1. 연구적 영향

**차분 기술 길이(DDL, Differential Description Length)**는 MDL을 정제하여 훈련 데이터의 파티션에서의 코드 길이 차이를 사용하여 일반화 오차를 직접 추정하며, 실험적으로 교차 검증 및 베이지안 증거보다 **우수한 하이퍼파라미터 선택**을 보여주었다.

MDL은 임의적인 정규화 가중치나 그리드 탐색 없이 자동으로 오컴의 면도날 페널티를 적용하여 **과적합을 방지**한다.

### 7-2. 연구 시 고려할 점 (Checklist)

1. **과파라미터화 체제 (Overparameterized Regime)**: 과파라미터화 모델에도 유효한 MDL-COMP는 Ridge 추정기 클래스에 의해 유도된 인코딩에 대한 최적성 기준을 통해 정의된다. 딥러닝 등 과파라미터화 모델에 MDL을 적용할 때는 고전적 NML이 아닌 이러한 확장된 복잡도 측도를 사용해야 한다.

2. **이중 하강 현상 (Double Descent)**: 최근 관찰된 과파라미터화 모델에서의 이중 하강 현상이 **비이상적 추정기 선택의 결과**일 수 있음을 MDL-COMP 분석이 시사한다.

3. **비유클리드 공간 확장**: Rm-NML(리만 다양체 NML)을 활용하면, MDL 원리를 리만 기하학적 데이터 공간에 **기하학적 일관성을 보존하면서** 적용할 수 있게 된다.

4. **계산 효율성**: MDL의 최적화는 일반적으로 이산 및 보편 코딩으로 인해 비미분가능하므로, 유전 알고리즘, 시뮬레이티드 어닐링, 조합론적 그리디 탐색이 모델 탐색에 자주 사용된다.

---

## 8. 2020년 이후 관련 최신 연구 비교 분석

**(1) MDL for High-Dimensional Variable Selection (Wei et al., 2022)**: 고차원 회귀 및 가산 모델에서 MDL 기반 변수 선택이 일관적이며, 강건 및 적응형 라쏘를 능가하고 극단적 차원에서도 경쟁력이 있다.

**(2) MDL-COMP for Overparameterized Models (Dwivedi et al., 2020/2023)**: MDL-COMP가 in-sample MSE의 상한임을 증명하였고, 데이터 기반 Prac-MDL-COMP가 Ridge 회귀에서의 테스트 MSE 최적화를 위한 하이퍼파라미터 튜닝을 안내하며, 제한된 데이터 설정에서 때로 교차검증을 능가하면서 항상 계산 비용을 절감함을 보였다.

**(3) Quotient-NML for Bayesian Networks (Silander et al., 2024)**: 베이지안 네트워크에서 qNML(Quotient-NML)은 로컬 1D-NML의 비율을 사용하여 분해 가능하고, 하이퍼파라미터가 없으며, 점수 동등한 기준을 구성한다.

**(4) α-NML (Bondaschi et al., 2022)**: NML을 일반화하여 레니 발산(Rényi-divergence) 기반 후회(regret)를 최소화하며, 혼합(베이지안) 예측기와 최악-케이스(NML) 예측기 사이를 보간하여 NML이 적용 불가한 경우에도 강건하다.

**(5) Singular MDL for Neural Networks (2025)**: MDL 원리를 신경망과 같은 특이 모델(singular models)로 일반화하였다. 이러한 모델이 손실 경관에서 중복성(redundancy)을 보이며, 이 **퇴화(degeneracy)**가 Hessian에 의한 곡률보다 모델 압축 가능성의 더 중요한 기여 요인임을 밝혔다.

**(6) Symbolic Regression via MDLformer (Yu et al., 2024)**: MDL 기반 MDLformer 탐색은 오차 최소화의 함정을 피하고 기존 접근법보다 훨씬 앞서 실제 수식 구조를 복원한다.

---

## 📊 방법 비교 요약 표

| 방법 | MDL 기반 여부 | 장점 | 단점 |
|---|---|---|---|
| **BIC** | 부분적 | 간단, 빠름 | 복잡도 측정 조잡 |
| **AIC** | X | 예측 성능 우수 | 일관성 부족 |
| **Cross-Validation** | X | 실용적 | 계산 비용 높음 |
| **NML (이 논문)** | ✅ | 미니맥스 최적, 파라미터 없음 | COMP 계산 어려움 |
| **MDL-COMP** | ✅ | 과파라미터화 지원 | 오라클 의존성 |
| **PAC-MDL** | ✅ | 딥러닝 적용 가능 | 경계가 느슨할 수 있음 |
| **α-NML** | ✅ | 더 강건, 범용 | 파라미터 튜닝 필요 |

---

## 📚 참고 자료 및 출처

1. **Grünwald, P. & Roos, T. (2019)**, *Minimum Description Length Revisited*, International Journal of Mathematics for Industry, Vol. 11, No. 1, Article 1930001. DOI: [10.1142/S2661335219300018](https://doi.org/10.1142/S2661335219300018)
2. **arXiv 원문**: [https://arxiv.org/abs/1908.08484](https://arxiv.org/abs/1908.08484)
3. **ResearchGate**: [https://www.researchgate.net/publication/337623996](https://www.researchgate.net/publication/337623996_Minimum_Description_Length_Revisited)
4. **University of Helsinki Research Portal**: [https://researchportal.helsinki.fi/en/publications/minimum-description-length-revisited](https://researchportal.helsinki.fi/en/publications/minimum-description-length-revisited)
5. **Geoinfotheory Summary (2023)**: [https://geoinfotheory.org/wp-content/uploads/2023/10/HVG-Summary-of-Grunwald-_-Roos-Minimum-Description-Length-Revisited-081023.pdf](https://geoinfotheory.org/wp-content/uploads/2023/10/HVG-Summary-of-Grunwald-_-Roos-Minimum-Description-Length-Revisited-081023.pdf)
6. **Dwivedi et al. (2020/2023)**, *Revisiting minimum description length complexity in overparameterized models*, arXiv:2006.10189. [https://arxiv.org/abs/2006.10189](https://arxiv.org/abs/2006.10189)
7. **Emergent Mind – MDL Objective**: [https://www.emergentmind.com/topics/minimum-description-length-mdl-objective](https://www.emergentmind.com/topics/minimum-description-length-mdl-objective)
8. **Emergent Mind – NML**: [https://www.emergentmind.com/topics/normalized-maximum-likelihood-nml](https://www.emergentmind.com/topics/normalized-maximum-likelihood-nml)
9. **Timaeus Research (2025)**, *Minimum Description Length Meets Singular Learning Theory*: [https://timaeus.co/research/2025-10-13-smdl](https://timaeus.co/research/2025-10-13-smdl)
10. **PMC Review (2022)**, *A Short Review on Minimum Description Length: An Application to Dimension Reduction in PCA*: [https://pmc.ncbi.nlm.nih.gov/articles/PMC8871178/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8871178/)
11. **Semantic Scholar**: [https://www.semanticscholar.org/paper/Minimum-Description-Length-Revisited-Grünwald-Roos/1bd5b3b7403938ed25b0b71504f3bdd95f8ba97c](https://www.semanticscholar.org/paper/Minimum-Description-Length-Revisited-Grünwald-Roos/1bd5b3b7403938ed25b0b71504f3bdd95f8ba97c)
12. **Grünwald, P. publications page**: [https://homepages.cwi.nl/~pdg/publicationpage.html](https://homepages.cwi.nl/~pdg/publicationpage.html)
