# Relative Importance and Value

---

## ⚠️ 사전 고지

본 논문(ssrn-2255827.pdf)은 **2005년 예비(preliminary) 버전**으로, Figure/Table이 본문에 존재하지 않습니다. 따라서 Figure/Table 번호 인용은 불가하며, 페이지 번호 기반으로 인용합니다. 확인되지 않는 내용은 명시적으로 표기합니다.

---

## 1. Executive Summary (10문장 이내)

Barry Feldman(2005)의 "Relative Importance and Value"는 통계·계량경제 모델에서 독립변수의 **상대적 중요도(Relative Importance)**를 공리적으로 정의하는 프레임워크를 제시한다. 기존에 광범위하게 사용되던 t-통계량, 공분산 분해(Covariance Decomposition), 평균법(Averaging Method)은 각각 심각한 허용성(admissibility) 결함을 가진다. 저자는 4가지 허용성 기준(비음성, 적절 배제, 적절 포함, 완전 기여)을 제안하고, 이를 모두 만족하는 유일한 방법이 **비례 한계 분해(Proportional Marginal Decomposition, PMD)**임을 증명한다. PMD는 3개의 공리(익명성, 한계적 적절 배제, 동등 비례 효과)로부터 도출되며, 변수 순서에 대한 확률 분포의 기댓값으로 정의된다. 핵심 결과는 PMD의 분해 성분이 협력 게임(cooperative game)의 **비례 가치(Proportional Value)**와 동일함을 보이는 정리(Theorem 3.1)이다. 평균법은 Shapley 값에 해당하며 적절 배제 기준을 위반하고, 공분산 분해는 비음성·적절 포함 기준을 동시에 위반한다. PMD는 일관 추정량(consistent estimator)임이 증명되며, 부트스트랩 신뢰구간 구성이 가능하다. 또한 일반화된 PMD 연속체($\alpha$-파라미터화)를 통해 Shapley 값을 극한점($\alpha \to 0$)으로 포함하는 더 넓은 허용 가능 측도 군이 존재함을 보인다. 저자는 상대적 중요도가 본질적으로 **비선형 측도**임을 주장하며, 협력 게임 이론과 통계적 변수 중요도 측정의 교차점에서 새로운 이론적 기반을 마련한다.

---

### 1-1. 연구의 목적과 필요성

**목적:**
통계 및 계량경제 모델에서 독립변수의 상대적 중요도를 공리적으로 엄밀하게 정의하고, 기존 측도의 결함을 진단하며, 허용 가능한 새로운 측도(PMD)를 제안하는 것.

**필요성** (pp. 1–2):
- Firth(1998)에 따르면 상대적 중요도 개념은 Hooker & Yule(1908)까지 거슬러 올라가지만, 보편적으로 수용된 일반적 측도는 부재
- 의학(Healy 1990; Schemper 1993), 보험(Frees 1998), 경영과학(Soofi et al. 2000), 사회과학(Kruskal & Majors 1989) 등 다양한 분야에서 더 나은 측도의 필요성이 제기됨
- 실무에서 t-통계량이 사실상(de facto) 중요도 측도로 사용되나, 이는 **완전 기여(full contribution)**를 측정하지 못함 (p. 2)
- 기존 제안 측도들은 비공리적·임시방편적(ad hoc)이라는 비판을 받아 왔음 (Heckman 1995; Goldberger & Manski 1995)

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거/방법 | 위치(페이지) | 결론 |
|---|-----------|-----------|-------------|------|
| 1 | 기존 t-통계량은 상대적 중요도의 신뢰할 수 있는 측도가 아님 | 두 변수의 상관이 높을수록 공동 한계 기여는 증가하나 개별 유의수준은 감소 | p. 2 | t-통계량 사용 부적절 |
| 2 | 평균법(Averaging/LMG)은 적절 배제 기준 위반 | Lemma 4.5: 진정한 $\beta^*_i = 0$인 변수도 양의 중요도를 가질 수 있음 | pp. 2–3, 13 | 평균법 부적절 |
| 3 | 공분산 분해($CVD$)는 비음성·적절 포함 기준 동시 위반 | Lemma 4.4, 4.7: 음수 값 가능, $\beta^* \neq 0$이어도 $CVD=0$ 가능 | pp. 12–13 | $CVD$ 부적절 |
| 4 | PMD는 4개 허용성 기준 모두 만족하는 유일한 알려진 측도 | Lemma 4.3, 4.6, 4.8; Theorem 3.1 | pp. 12–15, 18 | PMD 권장 |
| 5 | PMD 분해 성분 = 협력 게임의 비례 가치(Proportional Value) | Lemma 3.2, Theorem 3.1; 비례 잠재력(ratio potential) 동치 증명 | pp. 8–9 | 게임이론과 통계의 연결 |
| 6 | PMD는 일관 추정량(consistent estimator) | Lemma 3.3; Slutsky 정리 적용 | p. 10 | 표본 수렴 보장 |
| 7 | 일반화 PMD 연속체($\alpha > 0$) 전체가 허용 가능 | Lemma 5.4; $\alpha=0$ 극한이 Shapley 값 | pp. 15–16 | 허용 가능 측도군 존재 |
| 8 | 상대적 중요도는 본질적으로 비선형 측도 | 두 선형 추정기(평균법, CVD) 모두 허용 불가; 비례적 측도 연속체만 허용 가능 | p. 19 | 비선형성이 본질적 특성 |

---

### 2-1. 해결 문제 / 제안 방법 / 모델 구조 / 성능·한계 상세 설명

#### 🔴 해결하고자 하는 문제

통계 모델에서 독립변수들이 상호 상관되어 있을 때, 각 변수의 기여도를 어떻게 정당하게 분배할 것인가? 기존 측도들은:

- **t-통계량**: 완전 기여 불만족 (p. 2)
- **평균법(LMG/AMCV)**: 적절 배제 불만족 (p. 13, Lemma 4.5)
- **공분산 분해($CVD$)**: 비음성·적절 포함 불만족 (pp. 12–13, Lemma 4.4, 4.7)

#### 🟢 제안하는 방법: 비례 한계 분해 (PMD)

**분석 프레임워크** (p. 4):

모델 $\Theta$, 독립변수 집합 $N = \{1, 2, \ldots, n\}$, 성능 측도 $\mu$에 대해 협력 게임 $w$를 다음과 같이 정의:

$$w(S) = \mu_\Theta(N) - \mu_\Theta(N \setminus S) $$

변수 순서 $r \in \mathcal{R}(N)$에서 위치 $i$의 한계 기여:

$$M_i(r) = w(S^r_i) - w(S^r_{i-1}), \quad S^r_0 = w(\emptyset) = 0 $$

가능도 함수(likelihood function) $L(r)$ 기반 확률 분포:

```math
\mathbf{p}(r^*) = \frac{L(r^*)}{\sum_{r \in \mathcal{R}(N)} L(r)}
```

변수 $i$의 PMD 기댓값:

$$\phi_i(w) = E_{\mathbf{p}}[M_i(r)] = \sum_{r \in \mathcal{R}(N)} \mathbf{p}(r) M_{r(i)}(r) $$

**3개 공리로부터 가능도 함수 유도** (pp. 6–8):

- **공리 3.1 익명성**: $MC(r^\*) = MC(r) \Rightarrow L(r^*) = L(r)$
- **공리 3.2 한계적 적절 배제**: $\beta^*\_i \to 0 \Rightarrow \lim_{k \to \infty} \varphi_i(w_k) = 0$
- **공리 3.3 동등 비례 효과**: $\left|\frac{\partial \ln L(r)}{\partial \ln w(S)}\right| = 1$

세 공리를 결합하면 (p. 7, Eq. 5):

$$-\ln L(r) = c_r + \sum_{S \in r} \ln w(S)$$

따라서 (Lemma 3.1):

$$L(r) = \left(\prod_{S \in r} w(S)\right)^{-1}$$

정규화 인수(normalizing factor):

$$P(N) = \left(\sum_{r \in \mathcal{R}(N)} L(r)\right)^{-1} $$

$$\varphi_i(w) = P(N) \sum_{r \in \mathcal{R}(N)} L(r) M_{r(i)}(r) $$

**핵심 정리** (Theorem 3.1, p. 9):

$$\varphi_i(w) = \frac{P(N)}{P(N \setminus i)}$$

이는 협력 게임의 **비례 가치(Proportional Value)**의 정의와 동일.

**2변수 모델의 직관적 형태** (p. 10, Eq. 9):

$$\varphi_i(w) = \frac{w(\bar{i})}{w(\bar{i}) + w(\bar{j})} \cdot w(\overline{ij})$$

**비율 잠재력을 통한 계산** (Appendix A, p. 19, Eq. 19):

$$P(S) = w(S) \left(\sum_{i \in S} P(S \setminus i)^{-1}\right)^{-1}, \quad P(\emptyset) = c > 0$$

**일반화 PMD** (Section 5.1, p. 15):

$$L(r, \alpha) = \left(\prod_{S \in r} w(S)\right)^{-\alpha}, \quad \alpha > 0 $$

$$\varphi^\alpha_i(w) = P(N,\alpha) \sum_{r \in \mathcal{R}(N)} L(r, \alpha) M_{r(i)}(r)$$

- $\alpha \to 0$: Shapley 값(평균법)으로 수렴 (Lemma 5.3)
- $\alpha = 1$: 표준 PMD(비례 가치)

#### 🔵 모델 구조

```
통계 모델 Θ
    ↓
협력 게임 w(S) 구성 [Eq. 1]
    ↓
공리 3개 → 가능도 L(r) = (∏w(S))⁻¹ [Lemma 3.1]
    ↓
확률 분포 p(r) = P(N)·L(r) [Eq. 3, 6]
    ↓
기댓값 φᵢ(w) [Eq. 4, 7]
    ↓
비례 가치 φᵢ(w) = P(N)/P(N\i) [Theorem 3.1]
    ↓
비율 잠재력 재귀 공식으로 효율 계산 [Eq. 19]
```

#### 🟡 성능 향상 및 한계

**성능 (저자 주장):**
- 4개 허용성 기준 모두 만족하는 유일한 알려진 측도
- 일관 추정량 (Lemma 3.3)
- OLS, 최대가능도 등 다양한 모델에 적용 가능
- 부트스트랩 신뢰구간 구성 가능 (Section 3.6)

**한계:**
- 계산 복잡도: $n$개 변수 모델에서 $n!$개 순서 평가 필요 → 대규모 모델에 비실용적 (p. 18)
- PMD는 일반적으로 편향 추정량 (unbiased 아님, consistent만) (p. 9)
- $w(S) = 0$인 집합 존재 시 특별 처리 필요 (Section 3.4, Eq. 8)
- 완전 다중공선성(perfect multicollinearity) 환경에서 한계 존재 (p. 9)
- 표본 분포의 해석적 특성화 어려움 (p. 10)

---

## 3. 주장별 페이지 표시

| 주장 | 위치 |
|------|------|
| t-통계량의 부적절성 | p. 2 |
| 4개 허용성 기준 정의 | pp. 3–4 (Section 2) |
| 분석 프레임워크 및 $w(S)$ 정의 | p. 4, Eq. (1) |
| 3개 공리 (익명성, 적절 배제, 동등 비례 효과) | pp. 6–7 (Section 3.2) |
| 가능도 $L(r)$ 유도 | p. 7, Eq. (5); Lemma 3.1 |
| Theorem 3.1 (PMD = 비례 가치) | p. 9 |
| $w(S)=0$ 처리 | p. 9, Eq. (8) |
| 일관성 증명 | p. 10, Lemma 3.3 |
| 2변수 직관적 형태 | p. 10, Eq. (9) |
| 평균법 = Shapley 값 | p. 11, Lemma 4.1 |
| 공분산 분해 = Shapley 값 (다른 게임) | p. 12, Lemma 4.2 |
| 비음성 기준 충족/위반 | p. 12, Lemma 4.3–4.4 |
| 적절 배제 충족/위반 | p. 13, Lemma 4.5 |
| 적절 포함 충족/위반 | p. 13, Lemma 4.6–4.7 |
| 완전 기여 | pp. 13–14, Lemma 4.8–4.9 |
| 일반화 PMD ($\alpha$-연속체) | pp. 14–15, Section 5.1 |
| 보정(calibration) 논의 | p. 16, Section 5.2 |
| 엔트로피 논의 | pp. 16–17, Section 5.3 |
| 비율 잠재력 계산법 | p. 19, Appendix A, Eq. (19) |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 📌 저자가 직접 보고한 결과

| 항목 | 저자 보고 내용 | 위치 |
|------|--------------|------|
| 연구 주제 | 통계 모델에서 독립변수 상대적 중요도의 공리적 정의 | p. 1, Abstract |
| 방법 | PMD: 공리 3개로부터 $L(r) = (\prod_{S \in r} w(S))^{-1}$ 유도, 비례 가치와의 동치 증명 | Lemma 3.1, Theorem 3.1 |
| 평균법의 위반 | "A variable with a true beta of zero can have a positive relative importance" | p. 3 |
| CVD의 이중 위반 | 비음성·적절 포함 기준 모두 위반 | pp. 12–13 |
| PMD 유일성 | "PMD and the generalization...are the only relative importance measures known to be admissible" | p. 18 |
| 일관성 | Slutsky 정리 적용으로 $\text{plim } \varphi(w) = \varphi^*(w)$ | p. 10, Lemma 3.3 |
| 엔트로피 | Shapley 값의 엔트로피 = $\ln n!$, $\partial E/\partial\alpha \mid _{\alpha=1} < 0$ | pp. 16–17, Eq. (18) |

### 📌 리뷰어(분석자)의 해석

| 항목 | 해석 |
|------|------|
| 게임이론 연결의 의의 | 단순한 수학적 동치를 넘어, 변수 중요도 측정이 협력적 의사결정 문제와 구조적으로 동일함을 시사. 경제학적 "공정한 배분" 원리를 통계에 도입 |
| 비선형성 주장 | 저자는 이를 주장하지만, 이것이 PMD의 장점인지 아니면 해석의 복잡성을 높이는 단점인지는 응용 맥락에 따라 다를 수 있음 |
| Shapley 값(평균법)의 위상 | 저자는 평균법을 부적절로 분류하나, SHAP(SHapley Additive exPlanations) 등 현대 AI 해석 가능성 연구에서 Shapley 값이 광범위하게 사용되는 것은 맥락 의존적 적절성을 시사 |
| $\alpha$ 보정 거부 | 저자의 보정 거부 논리("상대적 중요도가 통계적 유의성의 함수가 되어선 안 됨")는 설득력 있으나, 이는 규범적 선택이지 수학적 필연성은 아님 |
| 계산 복잡도 | $n!$ 문제는 저자가 언급(p. 18)하나, 재귀 공식(Eq. 19)으로 해결 가능하다고 제시. 그러나 수치적 안정성 및 대규모 모델에서의 실용성은 충분히 논의되지 않음 |

---

## 5. 통계적 취약점 및 비교 불가능한 수치

| 항목 | 문제점 | 위치 |
|------|--------|------|
| ⚠️ 실증 데이터 없음 | 본 논문(v1.1)에는 수치 예시나 시뮬레이션 결과가 없음. 동반 논문 Feldman(2005) 참조를 유도하나 본 논문에서 검증 불가 | p. 3 ("companion paper") |
| ⚠️ 표본 분포 미특성화 | PMD의 비선형성으로 인해 표본 분포의 해석적 특성화가 어려움을 저자가 인정 | p. 10 |
| ⚠️ 일관성만 증명, 비편향성 없음 | $E[\varphi(w)] \neq \varphi^*(w)$ 일반적으로 성립. 편향의 크기는 알 수 없음 | p. 9 |
| ⚠️ 허용성 기준의 규범성 | 4개 기준 자체가 저자의 주관적 선택. 선형성을 배제한 근거가 "compelling statistical basis가 없다"는 소극적 주장 | p. 5 |
| ⚠️ $\alpha=1$ 선택의 자의성 | "there is no apparent rationale for any other choice"라는 진술은 최소한의 근거이며, 이 선택의 통계적 최적성은 증명되지 않음 | p. 7 |
| ⚠️ 엔트로피 결과의 의미 불명확 | $\partial E/\partial\alpha \mid _{\alpha=1} < 0$의 "complete significance is not clear at this point"를 저자 스스로 인정 | p. 17 |
| ⚠️ 비교 불가능한 수치 | 기존 측도들과의 성능 비교(예: MSE, 추정 편향 크기)가 없어 실질적 개선량을 수치로 확인 불가 | 전체 |

---

## 6. 논문이 답하지 않는 질문

1. **PMD의 유한 표본 편향(finite sample bias)은 얼마나 큰가?** 일관성만 증명되었고 편향의 크기나 수렴 속도는 불명확 (p. 9)

2. **변수 수가 많을 때(예: $n > 20$) 실용적 계산이 가능한가?** 재귀 공식(Eq. 19)이 제시되나 수치적 안정성, 계산 복잡도 상세 분석 부재 (p. 18)

3. **비선형 모델(신경망, 부스팅 등)에 PMD를 직접 적용할 수 있는가?** 논문은 OLS·MLE 맥락에서만 논의

4. **$\alpha \neq 1$인 일반화 PMD 중 특정 맥락에서 최적인 $\alpha$를 선택하는 원칙은 무엇인가?** 저자는 보정(calibration)을 거부하지만 대안을 제시하지 않음 (Section 5.2)

5. **표본 크기가 작을 때 PMD 추정의 신뢰성은?** 부트스트랩 제안이 있으나 구체적 시뮬레이션 없음 (Section 3.6)

6. **완전 다중공선성 환경에서 PMD의 정확한 거동은?** "virtually impossible in practice"라고 언급하나 이론적 보장 불완전 (p. 9)

7. **시계열 및 패널 데이터 모델에 PMD를 어떻게 확장하는가?** 언급 없음

8. **PMD가 인과 추론(causal inference) 맥락에서 변수 중요도를 올바르게 측정하는가?** 상관 기반 프레임워크이므로 인과성과의 관계 불명확

9. **다중 종속변수(multivariate) 모델로의 확장은 가능한가?** 논문은 단일 종속변수 모델로 제한

10. **PMD와 다른 허용 가능 측도 간의 경험적 차이는 실제로 얼마나 중요한가?** 수치 비교 부재

---

## 7. 가장 중요한 이론적 구조 5개 해석

> ⚠️ 본 논문에는 Figure/Table이 없으므로, 핵심 이론 구조(수식/개념 블록)를 "가장 중요한 5개 요소"로 대체하여 해석합니다.

### 🔑 핵심 구조 1: 협력 게임 $w(S)$ 정의 (p. 4, Eq. 1)

$$w(S) = \mu_\Theta(N) - \mu_\Theta(N \setminus S)$$

**해석:** 독립변수 집합 $S$의 "가치"를 전체 모델 성능에서 $S$를 제외했을 때의 성능을 뺀 값으로 정의. 이는 모든 부분 집합에 대해 정의되어 협력 게임의 구조를 형성. $w(S) \geq 0$이 단조 성능 측도에서 항상 보장되어 **비음성의 구조적 기반**을 제공. 핵심은 단순히 개별 변수의 한계 기여가 아니라, 모든 가능한 부분집합의 기여를 고려함으로써 변수 간 상호작용을 자연스럽게 포착한다는 점.

---

### 🔑 핵심 구조 2: 가능도 함수 $L(r)$ (Lemma 3.1, pp. 7–8)

$$L(r) = \left(\prod_{S \in r} w(S)\right)^{-1}$$

**해석:** 세 공리로부터 수학적으로 유일하게 결정되는 순서 $r$의 가능도. 직관적으로 한 순서 $r$에 포함된 집합들의 게임 가치가 클수록 그 순서의 가능도는 **낮아짐**. 이는 "이미 설명력이 높은 집합이 우선 진입하는 순서는 올바른 순서일 가능성이 낮다"는 역설적 논리를 반영. 이 가능도가 단순히 가정된 것이 아니라 공리적으로 **유일하게 도출**된다는 점이 이 논문의 가장 강력한 수학적 기여.

---

### 🔑 핵심 구조 3: 비례 가치와의 동치 (Theorem 3.1 + Lemma 3.2, pp. 8–9)

$$\varphi_i(w) = \frac{P(N)}{P(N \setminus i)}, \quad \text{where } P(N \setminus i) = \left(\sum_{r \in \mathcal{R}(N)} L(r) M_{r(i)}(r)\right)^{-1}$$

**해석:** 통계적 상대 중요도 측정 문제가 협력 게임 이론의 **비례 가치(Proportional Value)**와 수학적으로 동일함을 증명. 이는 단순한 수학적 일치가 아니라, 변수 중요도 배분이 협력적 이익 배분의 원리(보다 많이 기여하는 플레이어가 비례적으로 더 많이 받음)와 구조적으로 같다는 심층적 의미를 가짐. Shapley 값(평균법)이 "모든 플레이어 동등 협상력"을 의미하는 반면, 비례 가치는 "기여 비례 협상력"을 의미.

---

### 🔑 핵심 구조 4: 허용성 기준 비교표 (Section 4.4, pp. 12–14)

| 기준 | PMD | 평균법(AMCV) | 공분산분해(CVD) |
|------|-----|-------------|--------------|
| 비음성 | ✅ (Lemma 4.3) | ✅ (Lemma 4.3) | ❌ (Lemma 4.4) |
| 적절 배제 | ✅ (설계상) | ❌ (Lemma 4.5) | ✅ |
| 적절 포함 | ✅ (Lemma 4.6) | ✅ (Lemma 4.6) | ❌ (Lemma 4.7) |
| 완전 기여 | ✅ (Lemma 4.8) | ✅ (Lemma 4.8) | ✅ (Lemma 4.9) |

**해석:** 이 비교가 논문 실용적 기여의 핵심. 평균법은 Bayesian 균일 사전확률(모든 순서 동등)에 해당하여 "정보 없음" 가정을 내재. 반면 PMD는 순서의 확률을 그 순서에 포함된 집합들의 게임 가치로 조건부화하여 데이터 기반의 차별화된 확률을 부여. 두 선형 측도(평균법, CVD)가 모두 적어도 하나의 기준을 위반한다는 사실은 **상대적 중요도가 본질적으로 비선형 개념**임을 강하게 시사.

---

### 🔑 핵심 구조 5: 일반화 PMD 연속체 (Section 5.1, p. 15)

$$L(r, \alpha) = \left(\prod_{S \in r} w(S)\right)^{-\alpha}, \quad \alpha > 0$$

$$\varphi^\alpha_i(w) = P(N,\alpha) \sum_{r \in \mathcal{R}(N)} L(r,\alpha) M_{r(i)}(r)$$

$$\lim_{\alpha \to 0} \varphi^\alpha_i(w) = Sh_i(w) \quad \text{(Lemma 5.3)}$$

**해석:** 단일 PMD가 아니라 $\alpha$에 의해 파라미터화된 허용 가능 측도의 **연속 군(family)**이 존재함을 보임. $\alpha = 1$이 비례 가치(PMD), $\alpha \to 0$이 Shapley 값(평균법). 이는 상대적 중요도 측정이 스펙트럼을 형성하며, 맥락에 따라 다른 점이 더 적절할 수 있음을 의미. 다만 저자는 $\alpha$를 데이터로부터 추정(보정)하는 것을 거부하며, $\alpha = 1$이 자연스러운 기준점임을 주장. 이 연속체의 발견은 허용 불가 측도(평균법 자체는 부적절 배제 위반)와 허용 가능 측도($\alpha$-연속체 전체)를 명확히 구분하는 이론적 공헌.

---

## 8. 결론: 시사점, 후속 연구, 추가 방향

### 8-0. 저자가 제시한 시사점 및 후속 연구 계획 (p. 18)

**저자 시사점:**
1. PMD는 4개 허용성 기준을 모두 만족하는 유일한 알려진 측도
2. 상대적 중요도와 협력 게임 가치 함수 간의 심층적 연결 확인
3. 상대적 중요도는 본질적으로 비선형 측도
4. 비례 모드(proportional mode)가 통계적 상대 중요도의 더 나은 표현을 제공

**저자 제시 후속 연구:**
- Feldman(2005) 동반 논문에서 실증 응용(헤지펀드 분석 포함) 예정
- 상대적 중요도의 표본 분포 특성 추가 연구 필요 (p. 4)
- 보정($\alpha$ 추정)의 이론적 한계 추가 탐구 가능성 (Section 5.2)

---

### 8-1. 모델의 일반화 성능 향상 가능성

#### 현재 논문의 일반화 범위

저자는 PMD가 최소자승(OLS), 최대가능도(MLE) 등 다양한 모델과 성능 측도에 적용 가능하다고 주장 (p. 2). 단조 성능 측도 $\mu$이면 되므로 이론적 적용 범위는 넓음.

#### 일반화 향상을 위한 핵심 과제

**1. 고차원 모델로의 확장**

현재 $n!$ 계산 복잡도 문제가 있으나, 비율 잠재력 재귀식(Eq. 19)으로 $2^n$ 계산으로 감소 가능. 그러나 $n > 30$이면 여전히 비실용적. 향후 방향:

$$P(S) = w(S)\left(\sum_{i \in S} P(S \setminus i)^{-1}\right)^{-1}$$

이 재귀식의 **몬테카를로 근사(Monte Carlo approximation)** 또는 **희소 집합(sparse coalition) 근사**를 통해 대규모 모델 적용 가능성 탐색 필요.

**2. 비선형/비모수 모델로의 확장**

PMD는 성능 측도 $\mu$가 단조이면 적용 가능하므로, 이론적으로 다음에 적용 가능:
- 랜덤 포레스트의 OOB 오류 감소량
- 신경망의 손실 함수 감소량
- 가우시안 프로세스의 예측 분산 감소량

그러나 이 경우 $w(S)$ 추정 자체의 불확실성이 누적되어 **PMD 추정의 편향이 증폭**될 수 있음. 이에 대한 이론적 분석이 필요.

**3. 인과적 상대 중요도로의 확장**

현재 PMD는 순수 예측적(predictive) 프레임워크. 인과 그래프와 결합하여:

$$w_{\text{causal}}(S) = \mu_\Theta(N) - \mu_\Theta(N \setminus S)|_{\text{do}(X_S = \text{const})}$$

형태의 개입(interventional) 기반 게임을 구성하면 인과적 기여도 측정이 가능하나, 이는 별도의 공리 체계를 요구.

**4. 그룹 변수 중요도**

관련 변수들을 하나의 집합으로 취급하는 **집합적 PMD**:

$$\varphi_G(w) = \frac{P(N)}{P(N \setminus G)}$$

이는 이미 이론적으로 가능하나 (비율 잠재력의 집합 버전), 그룹 내 변수 배분 원칙이 추가로 필요.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ 이하는 2020년 이후 AI/통계 분야의 공개된 연구 동향에 기반한 분석입니다. 특정 논문과의 직접 대조는 해당 논문 접근 없이 이루어졌으므로, 구체적 수치 비교는 제시하지 않습니다.

#### 📊 Feldman(2005) PMD와 현대 연구의 비교

| 비교 항목 | Feldman (2005) PMD | SHAP (Lundberg & Lee, 2017) | Integrated Gradients (Sundararajan et al., 2017) | Kernel SHAP / TreeSHAP |
|-----------|-------------------|----------------------------|------------------------------------------------|----------------------|
| 이론적 기반 | 비례 가치 (Proportional Value) | Shapley 값 (공리 4개) | 공리적 귀속 방법 | Shapley 값 근사 |
| 허용성 기준 | 4개 모두 만족 | 비음성 ❌ (음수 가능), 적절 배제 ❌ | 완전 기여 충족, 다른 기준 부분 충족 | Shapley와 동일한 한계 |
| 계산 복잡도 | $O(2^n)$ (재귀) | $O(2^n)$ (정확), $O(n)$ (근사) | $O(n \cdot k)$ ($k$: 적분 단계) | $O(n \log n)$ (Tree) |
| 적용 범위 | 선형/MLE 모델 중심 | 임의 모델 | 미분 가능 모델 | 트리 기반 모델 |
| 실증 검증 | 없음 (v1.1) | 광범위한 실증 검증 | 이미지, NLP 등 | Kaggle 등 실무 적용 |

#### 🌐 현대 AI에서의 시사점

**1. SHAP의 광범위한 채택과 PMD의 재조명 필요성**

Lundberg & Lee(2017)의 SHAP은 Shapley 값 기반으로, Feldman이 지적한 **적절 배제 기준 위반**을 내재적으로 가짐. 그럼에도 불구하고 실무에서 광범위하게 채택된 이유는 계산 효율성(TreeSHAP)과 시각화 도구의 풍부함 때문. 이는 이론적 엄밀성과 실용성 간의 트레이드오프를 보여주며, PMD가 이론적으로 우월하더라도 도구 생태계 없이는 채택되기 어려움을 시사.

**2. LIME과의 비교**

Ribeiro et al.(2016)의 LIME은 국소 선형 근사 기반으로, Feldman의 "완전 기여" 기준을 전역적으로 만족하지 못함. PMD는 전역적 분해이므로 이론적으로 더 완전하지만, 개별 예측 설명에는 국소 방법이 더 직관적일 수 있음.

**3. 인과적 변수 중요도 연구**

2020년 이후 Pearl의 인과 프레임워크를 SHAP에 결합한 연구들(Causal SHAP, 2021 등)은 PMD의 예측적 프레임워크의 한계를 보완하는 방향. PMD를 개입적(interventional) 성능 측도와 결합하면 인과적 상대 중요도 측정이 가능하나, 이는 추가적 식별 가정이 필요.

**4. 대규모 언어 모델(LLM) 맥락**

LLM에서 특성 중요도 개념은 토큰/층 수준으로 확장. Attention 가중치가 중요도 대리 측도로 사용되지만 이는 Feldman의 어떤 허용성 기준도 만족한다고 보기 어려움. PMD의 프레임워크를 LLM 특성 중요도에 적용하려면 근본적인 확장이 필요.

#### 📌 앞으로의 연구에 미치는 영향 및 고려 사항

**Feldman(2005)이 미치는 영향:**

1. **공리적 프레임워크의 표준화**: 상대적 중요도 측도를 평가할 때 4개 허용성 기준(비음성, 적절 배제, 적절 포함, 완전 기여)을 체크리스트로 사용하는 방법론적 기여

2. **비례 가치의 통계적 재해석**: 게임이론의 비례 가치가 통계적 맥락에서 의미를 가짐을 보임으로써, 게임이론-통계학 간 학제적 연구의 기반 제공

3. **Shapley 기반 방법의 한계 명시**: SHAP 시대에 Feldman의 논문은 Shapley 값 기반 방법이 이론적으로 최선이 아님을 상기시키는 비판적 참조점

**향후 연구 시 고려할 점:**

| 고려 사항 | 구체적 내용 |
|-----------|-----------|
| **계산 효율성과 이론적 엄밀성의 균형** | PMD는 이론적으로 우월하나, 근사 알고리즘 개발 없이는 대규모 응용 불가 |
| **비선형 모델 확장** | $w(S)$ 추정의 불확실성이 PMD 추정에 어떻게 전파되는지 이론적 분석 필요 |
| **인과성 vs. 예측** | 중요도 측도의 목적(설명/인과/예측)을 명확히 하고 각 목적에 맞는 기준 적용 |
| **고차원 데이터** | $n > 50$ 이상에서의 PMD 근사 방법 및 오차 한계 개발 |
| **실증 검증** | 다양한 도메인(의료, 금융, NLP)에서 PMD vs. Shapley 기반 방법의 실제 차이 측정 |
| **불확실성 정량화** | 부트스트랩을 넘어선 PMD의 베이지안 신뢰구간 개발 |
| **그룹 중요도** | 상관된 변수 군(예: 원-핫 인코딩된 범주형 변수)에 대한 집합적 PMD 이론 정립 |

---

## 📚 참고 자료 (논문 내 인용 문헌)

본 분석에서 직접 참조한 문헌:

- **Feldman, Barry (2005)**. "Relative Importance and Value." SSRN Working Paper 2255827. Version 1.1, March 22, 2005.
- **Feldman, Barry (1999)**. *The Proportional Value of a Cooperative Game*. WoPEc.
- **Feldman, Barry (2002)**. *A Dual Theory of Cooperative Value*. SSRN abstract=317284.
- **Ortmann, K. M. (2000)**. "The proportional value for positive cooperative games." *Mathematical Methods of Operations Research*, 51:235–48.
- **Shapley, Lloyd S. (1953)**. "Additive and Non-Additive Set Functions." Ph.D. Thesis, Princeton University.
- **Lindeman, R. H., Merenda, P. F., and Gold, R. Z. (1980)**. *Introduction to Bivariate and Multivariate Analysis*. Scott Foresman.
- **Kruskal, William H. (1987)**. "Relative importance by averaging over orderings." *American Statistician*, 41:6–10.
- **Soofi, E. S., Retzer, J. J., and Yasai-Ardekani, M. (2000)**. "A framework for measuring importance of variables." *Decision Sciences*, 31:1–31.
- **Hart, S. and Mas-Colell, A. (1989)**. "Potential, value and consistency." *Econometrica*, 57:589–614.
- **Pratt, John W. (1987)**. "Dividing the indivisible." *Proceedings of the Second International Tampere Conference in Statistics*, 245–260.
- **Firth, David (1998)**. *Relative Importance of Explanatory Variables*. Nuffield College, Oxford.

> **현대 비교 연구 (접근 기반 분석, 직접 인용 아님):**
> - Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." *NeurIPS 2017*. (SHAP)
> - Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" *KDD 2016*. (LIME)
> - Sundararajan, M., Taly, A., & Yan, Q. (2017). "Axiomatic attribution for deep networks." *ICML 2017*. (Integrated Gradients)
