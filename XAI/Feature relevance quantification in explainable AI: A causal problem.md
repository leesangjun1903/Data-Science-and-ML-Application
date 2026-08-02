# Feature relevance quantification in explainable AI: A causal problem

---

## 1. Executive Summary (10문장 이내)

1. 본 논문은 Explainable AI(XAI)에서 Shapley 값을 이용한 피처 기여도 정량화 시 발생하는 개념적 혼란을 인과추론(causal inference) 관점에서 해소하고자 한다.
2. 핵심 논쟁은 "제거된 피처를 어떤 확률 분포로 대체할 것인가"이며, 조건부 기댓값(observational conditional expectation)과 주변 기댓값(marginal expectation) 중 어느 것이 올바른지에 관한 것이다.
3. 저자들은 Pearl(2000)의 do-연산자(do-operator) 개념을 빌려, 피처 제거는 *관측(observation)*이 아닌 *개입(intervention)*으로 해석해야 한다고 주장한다.
4. 개입적 관점에서 피처를 고정하면 나머지 피처들은 조건부가 아닌 자연적 주변 분포(marginal distribution)에서 샘플링되어야 한다.
5. 이 관점에서 SHAP(Lundberg & Lee, 2017)의 이론적 정당화는 오류이지만, 실제 구현에서 조건부 기댓값을 주변 기댓값으로 *근사*하므로 결과적으로 올바른 방향을 사용하고 있다.
6. 문제는 Aas et al.(2019) 등이 이 "근사"를 "개선"하려다 오히려 조건부 기댓값을 더 정확히 추정하는 방향으로 나아간 것이며, 저자들은 이를 개념적으로 결함이 있다고 비판한다.
7. Lemma 1을 통해 조건부 기댓값 기반 Shapley 값은 실제로 무관한 피처에 대해서도 비零 기여도를 할당하는 "민감성(Sensitivity) 공리 위반" 문제를 야기함을 증명한다.
8. 수치 실험(다변량 가우시안 및 실제 데이터셋)을 통해 주변 기댓값 기반 Shapley 값이 이론적 ground truth에 더 가까움을 보인다.
9. 저자들은 대칭성(symmetry) 공리 위반 주장(Sundararajan & Najmi, 2019)에도 반박하며, 평균으로부터의 편차가 큰 피처가 더 많은 기여도를 받는 것은 오히려 직관적으로 타당하다고 주장한다.
10. 결론적으로, 피처 기여도 정량화는 본질적으로 인과적 문제이며, 개입적 확률(interventional probability)에 기반한 주변 기댓값이 올바른 기준임을 주장한다.

---

### 1-1. 연구의 목적과 필요성

**목적:**
- XAI에서 Shapley 값 계산 시 사용되는 확률 분포 선택 문제를 인과추론 언어로 명확히 정리
- SHAP 패키지의 이론적 근거가 잘못되었음을 지적하고, "개선" 시도들의 개념적 오류 경고

**필요성:**
- 딥러닝 모델의 블랙박스 문제와 adversarial 취약성으로 인해 AI 해석가능성 수요 급증 (p.1)
- 공정한 의사결정 및 법적·윤리적 요구사항 충족을 위해 알고리즘 판단 근거 설명 필요 (p.1, Dwork et al., 2012; Kilbertus et al., 2017)
- 관측적(observational) vs 개입적(interventional) 조건부 분포의 혼동이 이론·실무에서 지속적으로 발생 (p.2)
- Aas et al.(2019), Lundberg et al.(2018) 등 후속 연구들이 잘못된 방향으로 SHAP을 "개선"하고 있음 (p.4)

---

## 2. 핵심 주장과 근거 표

| 번호 | 핵심 주장 | 근거 | 위치 |
|------|-----------|------|-------|
| ① | 피처 제거 시 조건부 기댓값이 아닌 **주변 기댓값**이 올바른 선택 | Pearl의 do-연산자: 개입(intervention)은 나머지 피처의 자연 분포를 유지 | p.4–5, Eq.(4),(14) |
| ② | SHAP의 이론적 정당화는 오류이나 실제 구현은 우연히 올바른 방향 | SHAP은 조건부 기댓값을 주변 기댓값으로 근사(feature independence 가정)하며 결과적으로 올바른 것을 사용 | p.4, Eq.(16) |
| ③ | 조건부 기댓값 기반 Shapley 값은 무관한 피처에도 비零 기여도 부여 | Lemma 1 및 Example 1: $f(x_1, x_2) = x_1$에서 $X_2$ 무관하나 $\phi_2 \neq 0$ | p.5, Lemma 1 |
| ④ | "개선된" SHAP(Aas et al., 2019)은 개념적으로 결함 | 조건부 기댓값을 더 정확히 추정하는 방향은 오히려 잘못된 것을 정확히 계산 | p.4, footnote 3 |
| ⑤ | 대칭성 공리 위반은 문제가 아님 | 평균에서 더 멀리 떨어진 피처 값이 더 큰 기여도를 갖는 것은 직관적으로 타당 | p.6 |
| ⑥ | 수치 실험에서 주변 기댓값 기반 Shapley 값이 ground truth에 더 근접 | 다변량 가우시안(n=3,10) 및 UCI HAR 데이터셋 실험 | p.7–9, Fig.4, Fig.5 |

---

### 2-1. 해결 문제 / 제안 방법 / 모델 구조 / 성능 / 한계

#### 📌 해결하고자 하는 문제

Shapley 값으로 피처 기여도를 계산할 때, **제거된(dropped) 피처를 어떤 분포로 대체**할 것인가의 문제. 구체적으로 아래 두 선택지 중 어느 것이 올바른가:

$$f_T(\mathbf{x}) := \mathbb{E}[f(\mathbf{x}_T, \mathbf{X}_{\bar{T}})|\mathbf{X}_T = \mathbf{x}_T] \quad \text{(조건부 기댓값, SHAP 이론)} $$

$$f_T(\mathbf{x}) := \mathbb{E}[f(\mathbf{x}_T, \mathbf{X}_{\bar{T}})] \quad \text{(주변 기댓값, 본 논문 주장)} $$

---

#### 📌 제안하는 방법 (수식 포함)

**[Pearl의 do-연산자를 이용한 개입적 기댓값]**

인과 구조에서 $X_T$를 $\mathbf{x}_T$로 고정하는 개입(intervention)의 효과:

$$\mathbb{E}[Y|do(X_T = \mathbf{x}_T)] = \mathbb{E}[f(\mathbf{x}_T, \mathbf{X}_{\bar{T}})] $$

이는 나머지 피처 $\mathbf{X}_{\bar{T}}$가 조건 없이 자연적 주변 분포에서 샘플링됨을 의미.

**[Shapley 값 정의]**

$$\phi_i := \sum_{T \subseteq U \setminus \{i\}} \frac{1}{n\binom{n-1}{|T|}} C(i|T) $$

여기서 $C(j|T) := g(T \cup \{j\}) - g(T)$이고, 세트 함수:

$$g(T) := f_T(\mathbf{x}) - f_\emptyset(\mathbf{x}) = \mathbb{E}[f(\mathbf{x}_T, \mathbf{X}_{\bar{T}})] - \mathbb{E}[f(\mathbf{X})]$$

**[KernelSHAP의 가중 최소제곱 표현]**

```math
\min_{\phi_1,\ldots,\phi_n} \left\{ \sum_{T \subseteq U} \left[g(T) - \sum_{j \in T} \phi_j\right]^2 k(U,T) \right\}
```

$$k(U,T) = \frac{(|U|-1)}{\binom{|U|}{|T|}|T|(|U|-|T|)}$$

**[KernelSHAP의 주변 기댓값 근사]**

$$f_{T,\text{KernelSHAP}}(\mathbf{x}) \approx \frac{1}{K}\sum_k f(\mathbf{x}_T, \mathbf{x}^k_{\bar{T}}) $$

여기서 $\mathbf{x}^k_{\bar{T}}$는 $\mathbf{X}_{\bar{T}}$에서 샘플링된 값.

**[다변량 가우시안 조건부 분포 - Aas et al. 방법 설명 목적]**

$$\mathbb{P}(\mathbf{X}_{\bar{T}}|\mathbf{X}_T = \mathbf{x}_T) = \mathcal{N}(\boldsymbol{\mu}_{\bar{T}|T}, \boldsymbol{\Sigma}_{\bar{T}|T})$$

$$\boldsymbol{\mu}_{\bar{T}|T} = \boldsymbol{\mu}_{\bar{T}} + \Sigma_{T\bar{T}}\Sigma_{TT}^{-1}(\mathbf{x}_T - \boldsymbol{\mu}_T)$$

$$\boldsymbol{\Sigma}_{\bar{T}|T} = \Sigma_{\bar{T}\bar{T}} - \Sigma_{\bar{T}T}\Sigma_{TT}^{-1}\Sigma_{T\bar{T}}$$

---

#### 📌 모델 구조

별도의 딥러닝 모델을 제안하는 논문이 아님. 프레임워크 구조는 다음과 같음:

```
입력 x = (x_1, ..., x_n)
        ↓
피처 서브셋 T 선택
        ↓
제거된 피처 X_{T̄} 샘플링 방식 결정
  ├── [잘못된 방법] 조건부 기댓값: E[f(x_T, X_{T̄}) | X_T = x_T]
  └── [올바른 방법] 주변 기댓값: E[f(x_T, X_{T̄})]
        ↓
Shapley 값 φ_i 계산 (가중 최소제곱 또는 직접 계산)
        ↓
피처 기여도 해석: f(x) - E[f(X)] = Σ φ_i
```

---

#### 📌 성능 향상

| 실험 설정 | 지표 | 주변 기댓값 (제안) | 조건부 기댓값 (기존) |
|-----------|------|-------------------|---------------------|
| 다변량 가우시안 n=3 | Shapley 오차 분포 | 0 근방에 강하게 집중 (파란색) | 넓게 분산 (빨간색) |
| 다변량 가우시안 n=10 | Shapley 오차 분포 | 0 근방에 강하게 집중 (파란색) | 넓게 분산 (빨간색) |
| UCI HAR 데이터셋 (실제 데이터) | Shapley 오차 분포 | 좁은 분포 (파란색) | 넓은 분포 (빨간색) |

*(Fig. 4, Fig. 5 기반)*

---

#### 📌 한계

| 한계 | 설명 |
|------|------|
| 비선형 함수의 ground truth 검증 어려움 | 실험은 선형 함수로만 수행(ground truth 계산 가능성 때문). 비선형 모델에서의 검증 부재 |
| 인과 구조 가정 | 저자들의 프레임워크는 "입력 → 출력"의 단순 인과 구조 가정. 실제 세계의 복잡한 인과 구조 미반영 |
| 주변 분포 추정 문제 | 주변 기댓값도 정확한 $P(\mathbf{X}_{\bar{T}})$ 추정이 필요하며 고차원에서 어려울 수 있음 |
| 커널 추정 실험의 차원 제한 | 조건부 기댓값의 커널 추정 실험은 저차원에서만 좋은 근사 제공 (p.8) |
| 대칭성 공리에 대한 반박의 주관성 | 대칭성 위반이 "문제가 아니다"는 주장은 직관적 논거에 기반하며 공리적으로 완전히 해소되지 않음 |

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| 두 가지 simplified function 정의 (조건부 vs 주변) | p.4, Eq.(3)(4) |
| Pearl의 do-연산자를 통한 개입적 기댓값 정당화 | p.4–5, Eq.(5)(14), Figure 1 |
| 인과 구조 도식: 관측 시나리오 vs 개입 시나리오 | p.6, Figure 2 |
| Lemma 1: 조건부 기댓값의 민감성 공리 위반 | p.5, Lemma 1, Eq.(6)–(13) |
| 대칭성 공리 반박 (Sundararajan & Najmi 재반박) | p.6, Figure 3 (= Table 3 from Sundararajan & Najmi, 2019) |
| Shapley 값의 가중 최소제곱 표현 | p.7, Eq.(15) |
| KernelSHAP 근사 방식 | p.7, Eq.(16) |
| 다변량 가우시안 실험 결과 | p.8, Figure 4 |
| UCI HAR 데이터셋 실험 결과 | p.9, Figure 5 |

---

## 4. 저자 보고 결과 vs 내 해석 분리

### 🔵 저자가 직접 보고한 내용

**연구 주제:**
> "We discuss which distribution is the right one for dropped features" (p.1, Abstract)

**방법 (수식):**
- 주변 기댓값이 개입적 조건부 분포와 일치함을 Pearl의 backdoor criterion으로 증명 (p.4-5, Eq.14)
- Shapley 값 계산의 가중 최소제곱 공식 제시 (p.7, Eq.15)

**결과:**
> "Figure 4 shows the errors $\phi_j - \text{contr}_j(\mathbf{x})$... The very precise results for the marginal expectation are mainly from feature 1 [whose coefficient is 0]." (p.8)
> 
> "approximating the marginal expectation is computationally inexpensive compared to the approximation of the conditional expectation with kernel estimation." (p.7)

---

### 🟠 내 해석

1. **저자의 핵심 공헌은 기술적 novelty보다 개념적 명료화(conceptual clarification)에 있다.** 새로운 알고리즘을 제안하기보다는, 기존 SHAP의 이론-구현 불일치를 인과 언어로 해소하는 것이 목적이다.

2. **"SHAP이 우연히 올바른 것을 하고 있다"는 주장은 아이러니하다.** 저자들은 SHAP의 이론적 근거는 틀렸지만 구현은 맞다고 주장하는데, 이는 SHAP 개발자(Lundberg & Lee)가 인과적 함의를 인식하지 못한 채 올바른 실무적 결정을 내렸다는 의미이다.

3. **실험의 ground truth 설정이 논문의 주장에 유리하게 구성되어 있다.** 선형 함수 $f(\mathbf{x}) = \alpha_0 + \sum_i \alpha_i x_i$의 ground truth는 $\alpha_j(x_j - \mathbb{E}[X_j])$로, 이는 주변 기댓값 기반 계산과 완벽히 일치하도록 설계된 실험이다 (⚠️ 통계적 취약점, 섹션 5 참조).

4. **Lemma 1의 "민감성 공리 위반"은 공리의 적용 범위 문제이기도 하다.** 저자들도 인정하듯, Shapley 값은 집합 함수 $\tilde{g}$에 대한 민감성을 만족하므로 엄밀한 의미의 공리 위반은 아니다. 실용적 의미의 민감성이 문제인 것이다.

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치 ⚠️

| 번호 | 취약점 | 설명 |
|------|--------|------|
| ⚠️1 | **선형 함수에 한정된 실험** | Ground truth $\alpha_j(x_j - \mathbb{E}[X_j])$는 오직 선형 함수에서만 도출 가능. 비선형 모델(딥러닝 등)에서의 검증 없음 (p.7) |
| ⚠️2 | **UCI HAR 데이터의 target 선택 임의성** | "4개 피처 무작위 선택, 그 중 1개를 target으로"라는 설정이 1000번 반복되지만, 피처 선택의 편향 가능성 미검토 (p.9) |
| ⚠️3 | **다변량 가우시안 실험에서 $\alpha_1=0$ 설정** | "매우 정밀한 결과는 주로 피처 1(계수=0인 피처)에서 나온다"고 저자가 직접 인정. 조건부 기댓값에 불리한 케이스를 강조하는 구조 (p.8) |
| ⚠️4 | **SHAPR의 하이퍼파라미터 $\sigma^2=0.1$ 고정** | 커널 추정의 bandwidth $\sigma^2=0.1$은 SHAPR의 default값으로, 최적 튜닝 없이 비교. 다른 $\sigma^2$ 값에서의 성능 미보고 (p.9) |
| ⚠️5 | **비교 불가능한 수치** | Fig.4의 오차 히스토그램은 수치 범위를 직접 보고하지 않고 시각적 분포만 제시. 정량적 MSE 등 통계량 없음 |
| ⚠️6 | **인과 구조 가정의 미검증** | Figure 1, 2의 인과 구조 가정(공통 원인 Z 존재 등)이 실험 데이터에서 성립하는지 검증되지 않음 |

---

## 6. 문서가 답하지 않는 질문

| 번호 | 미해답 질문 |
|------|------------|
| Q1 | 비선형(non-linear) 모델에서도 주변 기댓값 기반 Shapley 값이 조건부 기댓값보다 인과적으로 올바른가? ground truth를 어떻게 정의하고 검증할 것인가? |
| Q2 | 피처 간 의존성이 매우 강한 경우(ex. 다중공선성), 주변 기댓값 기반 Shapley 값은 off-manifold 샘플링 문제를 어떻게 처리하는가? |
| Q3 | 인과 구조가 완전히 알려지지 않은(unknown causal structure) 경우 이 프레임워크를 어떻게 적용할 수 있는가? |
| Q4 | TreeSHAP, DeepSHAP 등 특정 모델에 특화된 SHAP 변형들에도 동일한 비판이 적용되는가? |
| Q5 | 주변 기댓값 사용이 고차원(수백~수천 피처)에서 계산적으로 실현 가능한가? |
| Q6 | 조건부 기댓값의 "민감성 공리 위반"이 실제 의사결정에서 어느 정도의 오류를 야기하는가? (실제 피해 사례 없음) |
| Q7 | SHAP 이외의 다른 XAI 방법(LIME, LRP, DeepLIFT)들도 동일한 인과적 문제를 갖는가? |
| Q8 | 제안된 프레임워크를 이미지나 텍스트 등 비정형 데이터에 적용할 때 "피처의 주변 분포"를 어떻게 정의해야 하는가? |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.4) — 인과 구조와 관측적/개입적 조건부 분포의 차이

```
         Z (공통 원인, 잠재)
        /|\
       / | \
      X1 X2 X3
         |
         Y
```

**해석:** 공통 원인 $Z$가 $X_1, X_2, X_3$를 동시에 생성하는 구조에서, $X_1 = x_1$을 *관측*하면 $Z$를 통해 $X_2, X_3$의 분포도 바뀐다. 따라서 $\mathbb{E}[Y|X_1=x_1]$는 $X_1$의 순수 효과가 아닌 $X_2, X_3$의 효과를 혼동(confound)한다. 반면 $X_1$에 *개입*(do)하면 $X_2, X_3$는 자연 분포를 유지하므로:
$$\mathbb{E}[Y|do(X_1=x_1)] = \int \mathbb{E}[Y|x_1,x_2,x_3]p(x_2,x_3)dx_2dx_3$$
이 도식은 논문의 핵심 주장인 "피처 제거 = 개입"을 직관적으로 설명하는 가장 중요한 그림이다.

---

### Figure 2 (p.6) — 예측 시나리오의 인과 구조 (관측 vs 개입)

```
[Top - 관측 시나리오]
X̃₁ X̃₂ X̃₃ X̃₄ X̃₅  (실제 피처)
 ↓   ↓   ↓   ↓   ↓
X₁  X₂  X₃  X₄  X₅  (알고리즘 입력)
         ↓
         Y

[Bottom - 개입 시나리오]
X̃₁ X̃₂ X̃₃ X̃₄ X̃₅  (실제 피처)
 ↓   ↓   ↓   ↓   ↓
x₁  x₂  X₃  X₄  X₅  (X₁,X₂ 개입 고정)
         ↓
         Y
```

**해석:** 알고리즘의 입력-출력 인과 구조와 실제 세계의 피처 인과 구조를 분리함으로써, 저자들은 "현실 세계에서 피처들 간의 인과 관계를 알 필요 없이" 개입적 기댓값(= 주변 기댓값)을 사용할 수 있음을 보인다. $X_1, X_2$를 $x_1, x_2$로 고정할 때 $X_3, X_4, X_5$는 자연 주변 분포 $P_{X_3,...,X_5}$에서 샘플링된다.

---

### Figure 3 / Table (p.6) — 대칭성 공리 위반 반박 예시

| Probability | $X_1$ | $X_2$ | $f = X_1 + X_2$ |
|-------------|--------|--------|-----------------|
| $(1-p)(1-q)$ | 1 | 1 | 2 |
| $(1-p)q$ | 1 | 2 | 3 |
| $(1-q)p$ | 2 | 1 | 3 |
| $p \cdot q$ | 2 | 2 | 4 |

**해석:** Sundararajan & Najmi(2019)는 이 예시에서 $(x_1, x_2) = (2,2)$일 때 $x_1$이 기여도 $(1-p)$, $x_2$가 $(1-q)$를 받아 $p \neq q$이면 대칭성이 위반된다고 주장했다. 저자들은 이에 반박하여, $X_1$과 $X_2$가 서로 다른 주변 분포를 가질 때 자신의 평균에서 더 멀리 떨어진 값이 더 큰 기여도를 받는 것은 직관적으로 타당하다고 주장한다. 이진 함수 $\tilde{g}$는 실제로 비대칭이므로 Shapley 값의 대칭성 공리도 위반하지 않는다.

---

### Figure 4 (p.8) — 다변량 가우시안 실험 결과 히스토그램

**해석:** 3차원(좌) 및 10차원(우) 다변량 가우시안 분포에서 선형 함수에 대한 Shapley 오차 분포를 보여준다. 파란색(주변 기댓값)은 오차가 0 근방에 매우 강하게 집중되어 있고, 빨간색(조건부 기댓값)은 넓게 분산되어 있다. 저자들은 "매우 정밀한 결과는 계수가 0인 피처(무관한 피처)에서 나온다"고 밝혔다 — 이는 ⚠️ 조건부 기댓값이 무관한 피처에 비零 기여도를 할당하는 Lemma 1을 수치적으로 확인하는 결과로, 논문의 핵심 주장을 실험적으로 지지한다.

---

### Figure 5 (p.9) — UCI HAR 데이터셋 실험 결과 히스토그램

**해석:** 실제 Human Activity Recognition 데이터셋(561 피처, 10,299 샘플)에서 선형 모델에 대한 Shapley 오차를 보인다. 파란색(SHAP의 주변 기댓값 근사)은 0 근방에 날카로운 피크를 형성하고, 빨간색(SHAPR의 조건부 기댓값 추정)은 더 넓은 분포를 보인다. ⚠️ 단, SHAPR의 $\sigma^2=0.1$ 고정, 피처 선택의 무작위성 등 실험 설계상 한계가 있으며, 이 데이터셋에서 피처들이 공통 원인(label)을 공유한다는 점을 저자들도 명시하고 있어 조건부 분포가 "틀린" 선택임을 지지하는 구조다.

---

## 8. 결론: 시사점 및 후속 연구

### 8-0. 저자들이 제시한 시사점과 후속 연구 계획

**시사점 (p.9):**
1. XAI에서 피처 기여도 정량화는 본질적으로 **인과적 문제**이다.
2. 조건부 기댓값이 아닌 **개입적(= 주변) 기댓값**이 올바른 개념적 선택이다.
3. SHAP의 이론적 근거는 오류이나 구현은 올바른 방향을 사용한다.
4. 타 연구자들이 SHAP을 "개선"하려는 시도(Aas et al., 2019 등)는 개념적으로 결함이 있다.
5. 대칭성 공리 위반 비판(Sundararajan & Najmi, 2019)은 재해석할 여지가 있다.

**논문 내 명시적 후속 연구 계획:** 없음 (이 논문은 개념적 명료화 논문으로 구체적 future work 미제시).

---

### 8-1. 모델의 일반화 성능 향상 가능성

이 논문은 일반화 성능을 직접 다루지는 않지만, 주변 기댓값 기반 Shapley 값은 일반화와 다음과 같이 연결된다:

**① Off-manifold 문제와 일반화:**
조건부 기댓값 기반 방법은 피처 공간의 *실현 가능한(feasible) 영역* 내에서 샘플링하므로, 훈련 분포를 벗어나는 입력(out-of-distribution)을 사용하지 않는다는 장점이 주장된다. 반면 주변 기댓값은 독립 샘플링으로 인해 off-manifold 입력 $f(\mathbf{x}\_T, \mathbf{X}_{\bar{T}})$를 생성할 수 있다. 이는 모델이 훈련 분포 밖에서 어떻게 동작하는지에 대한 *일반화 신뢰도* 문제와 직결된다.

**② 특성 선택(Feature Selection)을 통한 간접 일반화:**
주변 기댓값이 무관한 피처에 정확히 $\phi = 0$을 부여하므로 (Lemma 1), 올바른 피처 기여도 평가는 관련 없는 피처를 더 정확히 식별하고 제거할 수 있게 한다. 이는 모델 압축(model compression) 및 정규화(regularization)를 통한 일반화 성능 향상에 기여할 수 있다.

**③ 분포 이동(Distribution Shift) 상황:**

$$\text{주변 기댓값 기반: } \phi_i = \sum_T \frac{|T|!(n-|T|-1)!}{n!}[\mathbb{E}[f(\mathbf{x}_T, \mathbf{X}_{\bar{T}})] - \mathbb{E}[f(\mathbf{x}_{T\setminus i}, \mathbf{X}_{\overline{T\setminus i}})]]$$

이 방식은 피처들이 독립적으로 변동할 수 있다는 가정하에 각 피처의 순수 기여를 측정하므로, 훈련-테스트 분포가 다른 상황(covariate shift)에서도 피처의 *인과적 기여도*가 안정적으로 유지될 수 있다.

**④ 한계:** 그러나 훈련 데이터의 밀도가 낮은 영역에서 주변 기댓값 계산에 사용되는 샘플이 모델의 신뢰할 수 없는 예측 구역에 해당할 수 있으며, 이때의 설명은 불안정할 수 있다. 이 문제는 아직 해소되지 않은 열린 문제다.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **중요 고지:** 이하 내용 중 2020년 이후 연구에 대한 설명은 본 문서(논문 PDF)에 포함되지 않은 외부 정보에 해당합니다. 저의 훈련 데이터(2023년 초까지)에 기반한 내용이며, 개별 논문의 세부 수치나 일부 주장은 실제 논문 확인이 필요합니다.

#### 주요 후속/관련 연구 비교

| 연구 | 핵심 입장 | 본 논문과의 관계 |
|------|-----------|----------------|
| **Chen et al. (2020)** "True to the Model or True to the Data?" *arXiv:2006.16234* | 조건부 기댓값(on-manifold)이 "데이터에 충실"하고, 주변 기댓값이 "모델에 충실"하다는 이분법 제시 | 본 논문의 이분법을 확장하되, 어느 쪽이 "옳다"는 단언을 피함 |
| **Slack et al. (2020)** "Fooling LIME and SHAP" *AAAI/AIES* | 주변 기댓값(off-manifold) 사용 시 adversarial 조작에 취약함을 실증 | 본 논문 주장에 대한 실용적 반론; off-manifold 문제 부각 |
| **Fryer et al. (2021)** "Shapley Values for Feature Selection" *arXiv* | Shapley 기반 피처 선택에서 조건부 vs 주변 기댓값의 결과 차이 분석 | 본 논문의 이론을 피처 선택 맥락에서 실증 |
| **Aas et al. (2021)** "Explaining individual predictions..." *Journal of Computational Statistics* | 조건부 기댓값을 더 정확히 추정하는 방법론 지속 개발 | 본 논문이 "개념적으로 결함"이라고 비판한 방향의 계속된 발전 |
| **Janzing et al. (2020)** "Feature relevance..." *AISTATS 2020* | 본 논문의 저널/학회 버전으로, 동일 저자들의 확장판 | 본 논문의 직접적 후속 |
| **Sundararajan & Najmi (2020)** "The many Shapley values for model explanation" *ICML* | "baseline Shapley", "interventional Shapley" 등 다양한 Shapley 값 분류 체계 제시 | 본 논문의 비판(대칭성 공리 위반)에 대한 간접적 응답 |
| **Covert et al. (2021)** "Explaining by Removing" *JMLR* | 피처 제거 기반 설명의 통합 프레임워크 제시, 조건부/주변 기댓값 모두 특수 케이스로 포함 | 본 논문의 이분법을 더 넓은 프레임으로 흡수 |

#### 본 논문이 앞으로의 연구에 미치는 영향

1. **인과적 XAI 분야 형성:** 이 논문은 XAI 문제를 인과추론 언어로 해석하는 흐름을 강화했으며, "Causal SHAP", "Asymmetric Shapley" 등 인과 기반 설명 방법론의 이론적 기초로 인용된다.

2. **조건부 vs 주변 기댓값 논쟁의 공식화:** 이전에는 암묵적이던 이 구분을 명시화함으로써, 이후 연구들이 자신의 방법론적 선택을 명시적으로 정당화하도록 유도했다.

3. **TreeSHAP의 변화:** 논문의 footnote 3에서도 언급되었듯, TreeExplainer는 이미 이에 맞게 수정되었다. 이는 논문의 실질적 영향력을 보여준다.

#### 앞으로 연구 시 고려할 점

| 고려사항 | 설명 |
|---------|------|
| **Off-manifold vs On-manifold 트레이드오프 명시** | 주변 기댓값은 인과적으로 올바르지만 off-manifold 샘플을 생성할 수 있음. 연구 목적(인과 추론 vs 모델 행동 설명)에 따라 명시적 선택 필요 |
| **비선형 모델에서의 이론적 확장** | 본 논문의 실험은 선형 함수에 한정. 딥러닝 등 비선형 모델에서 주변 기댓값이 여전히 "올바른" 선택인지 이론적 증명 필요 |
| **피처 의존성 정도에 따른 방법 선택** | 피처가 강하게 상관된 경우 주변 기댓값과 조건부 기댓값의 차이가 극대화됨. 실제 적용 시 데이터의 피처 의존성 구조 사전 분석 권장 |
| **인과 그래프 미지(unknown causal structure) 상황** | 실제 응용에서 인과 구조는 대부분 알려지지 않음. 부분적으로 알려진 인과 구조 하에서의 Shapley 값 계산 방법론 연구 필요 |
| **설명의 안정성(stability) 평가** | 두 방법 모두 샘플링 기반 근사를 사용하므로 설명의 분산(variance) 비교 연구 필요 |
| **고차원 데이터에서의 확장성** | 주변 기댓값 계산도 고차원에서 지수적 복잡도를 가짐. 효율적 근사 방법 연구 필요 |

---

## 참고 자료 (본 분석에서 직접 인용된 원문 참고문헌)

본 논문 내 참고문헌:
- **Janzing, D., Minorics, L., & Blöbaum, P. (2019).** "Feature relevance quantification in explainable AI: A causal problem." *arXiv:1910.13413v2*
- **Lundberg, S. & Lee, S. (2017).** "A unified approach to interpreting model predictions." *NeurIPS 30*, pp.4765–4774.
- **Pearl, J. (2000).** *Causality.* Cambridge University Press.
- **Aas, K., Jullum, M., & Løland, A. (2019).** "Explaining individual predictions when features are dependent: More accurate approximations to Shapley values." *arXiv:1903.10464*
- **Sundararajan, M. & Najmi, A. (2019).** "The many Shapley values for model explanation." *arXiv:1908.08474*
- **Datta, A., Sen, S., & Zick, Y. (2016).** "Algorithmic transparency via quantitative input influence." *IEEE S&P 2016*, pp.598–617.
- **Sundararajan, M., Taly, A., & Yan, Q. (2017).** "Axiomatic attribution for deep networks." *ICML 2017*, vol.70, pp.3319–3328.
- **Shapley, L. (1953).** "A value for n-person games." *Contributions to the Theory of Games (AM-28)*, vol.2.
- **Charnes, A. et al. (1988).** "Extremal principle solutions of games in characteristic function form." *Econometrics of Planning and Efficiency*, 11:123–133.
- **Anguita, D. et al. (2013).** "A public domain dataset for human activity recognition using smartphones." *ESANN 2013*.
- **Zhao, Q. & Hastie, T. (2019).** "Causal interpretations of black-box models." *Journal of Business & Economic Statistics*.
- **Friedman, J.H. (2001).** "Greedy function approximation: A gradient boosting machine." *Annals of Statistics*, pp.1189–1232.
- **Lundberg, S., Erion, G., & Lee, S. (2018).** "Consistent individualized feature attribution for tree ensembles." *arXiv:1802.03888*
