# Bayesian Model Averaging: A Systematic Review and Conceptual Classification

---

## 1. Executive Summary (10문장 이내)

Fragoso & Louzada Neto (2015)는 1996–2014년 사이 발표된 BMA(Bayesian Model Averaging) 관련 587편의 논문을 체계적으로 분석한 리뷰 논문이다.  
BMA는 모델 불확실성을 베이즈 추론으로 다루어 모델 선택, 결합 추정, 예측을 동시에 수행하는 방법론이다.  
저자들은 Hoeting et al. (1999), Wasserman (2000) 이후 15년간의 BMA 발전을 포괄하는 종합 검토가 없었음을 지적하며 연구의 필요성을 정당화한다.  
4개 데이터베이스(Scopus, Web of Science, ScienceDirect, MathSciNet)를 활용한 체계적 문헌 검색 후 8개 항목의 개념적 분류 체계(CCS)를 개발·적용하였다.  
분석 결과 BMA의 가장 일반적인 활용은 모델 선택(40%)이며, 예측(28%), 결합 추정(19%) 순이었다.  
사전 분포로는 균등(vague) 사전 분포가 50% 이상 논문에서 채택되었고, 증거 추정은 BIC 근사가 가장 많이 사용되었다.  
MCMC 방법은 예상보다 활용 비율이 낮았으며, 시뮬레이션 연구나 데이터 기반 검증이 부족한 논문이 다수였다.  
BMA는 생명과학(34%), 경제학(25%), 물리공학(25%), 통계·기계학습(16%) 순으로 폭넓게 적용되었다.  
저자들은 복잡한 모델로의 확장, 사전 분포 연구, 계산 비용 논의, 검증 방법 적용이 향후 연구의 핵심 과제임을 제안한다.

### 1-1. 연구의 목적과 필요성

**목적:**
- BMA 문헌에 대한 개념적 분류 체계(CCS) 제공 (p.4)
- 연구 동향 요약 및 미래 방향 식별 (p.4)
- BMA를 복잡한 모델에 적용하려는 연구자를 위한 실질적 가이드 제공 (p.5)

**필요성:**
- 단일 모델 선택은 **모델 불확실성(model uncertainty)을 무시**하여 과도한 확신과 위험한 의사결정을 초래 (p.2)
- Hoeting et al. (1999), Wasserman (2000) 이후 포괄적 리뷰 부재 — **MCMC 혁명 이후 15년간의 발전을 다루지 못함** (p.4)
- BMA 적용 방식의 다양성과 가정이 문헌에 산재하여 있으나, 이를 체계적으로 정리한 연구 없음 (p.1)

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 출처 위치 |
|---|---|---|
| BMA는 단일 모델 선택보다 예측 위험이 낮다 | 로그 손실 하에서 BMA 예측이 최소 위험 달성 (Madigan & Raftery, 1994) | p.4, Eq.(6) |
| 균등 사전 분포가 압도적으로 많이 사용됨 | 297편(>50%) 논문에서 $\pi(M_l)=1/K$ 채택 | p.24 |
| BIC 근사가 가장 널리 사용되는 증거 추정 방법 | 190편 논문에서 BIC 기반 근사 사용 | p.25 |
| MCMC 활용도가 예상보다 낮음 | 563편 중 261편만 MCMC 사용(약 46%) | p.26 |
| 시뮬레이션 및 검증 연구가 심각하게 부족 | 358편(61%)이 어떠한 검증도 수행하지 않음 | p.28 |
| BMA 문헌이 회귀 모델 변수 선택에 과도하게 집중 | 231편(~40%)이 모델 선택, 대부분 회귀 변수 선택 | p.19 |
| BMA 적용 분야가 생명과학·경제학에 집중 | 생명과학 201편(34.24%), 경제학 149편(25.38%) | p.21, p.23 |
| 사후 예측 검증(Posterior Predictive Check)이 거의 사용되지 않음 | 587편 중 단 1편만 적용 | p.28 |

---

## 2-1. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 해결하고자 하는 문제

**문제 1: 모델 불확실성(Model Uncertainty)**

단일 최적 모델 선택은 모델 불확실성을 무시한다. $K$개의 모델 $M_1, \ldots, M_K$가 있을 때, 하나를 선택하면 나머지 모델이 가진 정보를 폐기하는 문제가 발생한다.

**문제 2: 대규모 모델 공간(High-Dimensional Model Space)**

$p$개의 공변량이 있는 회귀 모델에서 가능한 모델 수는 $2^p$로 기하급수적으로 증가 (p.11, Eq.13):

$$Y = \beta \mathbf{X} + e$$

**문제 3: 주변 우도(Evidence) 계산의 난점**

$$\pi(\mathbf{Y}|M_l) = \int L(\mathbf{Y}|\theta_l, M_l)\pi(\theta_l|M_l)d\theta_l \quad \text{(Eq. 2)}$$

이 적분은 일반적으로 해석적 해가 없음.

---

### 제안하는 방법 (수식 포함)

**BMA의 핵심 수식 체계:**

**[1] 사후 파라미터 분포]** (p.2, Eq.1)

$$\pi(\theta_l|\mathbf{Y}, M_l) = \frac{L(\mathbf{Y}|\theta_l, M_l)\pi(\theta_l|M_l)}{\int L(\mathbf{Y}|\theta_l, M_l)\pi(\theta_l|M_l)d\theta_l}$$

**[2] 사후 모델 확률]** (p.3, Eq.3)

$$\pi(M_l|\mathbf{Y}) = \frac{\pi(\mathbf{Y}|M_l)\pi(M_l)}{\sum_{m=1}^{K} \pi(\mathbf{Y}|M_m)\pi(M_m)}$$

**[3] BMA 결합 예측/추정]** (p.3, Eq.6)

$$\pi(\Delta|\mathbf{Y}) = \sum_{l=1}^{K} \pi(\Delta|\mathbf{Y}, M_l)\pi(M_l|\mathbf{Y})$$

**[4] 베이즈 인수(Bayes Factor)]** (p.3, Eq.4)

$$BF_{lm} = \frac{\pi(M_l|\mathbf{Y})}{\pi(M_m|\mathbf{Y})}$$

---

**증거 추정 방법들:**

**[5] 중요도 샘플링(Monte Carlo)]** (p.9, Eq.7)

$$\widehat{\pi(\mathbf{Y}|M_l)} = \frac{1}{R}\sum_{r=1}^{R} \frac{L(\mathbf{Y}|\theta_r, M_l)\pi(\theta_r|M_l)}{w(\theta_r)}$$

**[6] MCMC 기반 추정]** (p.10, Eq.8)

```math
\widehat{\pi(\mathbf{Y}|M_l)} = \left\{\frac{1}{R}\sum_{r=1}^{R} \frac{w(\theta_l^{(r)})}{L(\mathbf{Y}|\theta_l^{(r)}, M_l)\pi(\theta_l^{(r)}|M_l)}\right\}^{-1}
```

**[7] 밀도 비율법(Chib, 1995)]** (p.10, Eq.10)

```math
\widehat{\pi(\mathbf{Y}|M_l)} = \frac{L(\mathbf{Y}|\theta_l^*, M_l)\pi(\theta_l^*|M_l)}{\pi(\theta_l^*|\mathbf{Y}, M_l)}
```

**[8] Laplace 근사]** (p.10, Eq.11)

$$\pi(\mathbf{Y}|M_l) \approx (2\pi)^{\frac{p_l}{2}}\sqrt{|\Psi_l|}L(\mathbf{Y}|\tilde{\theta}_l, M_l)\pi(\tilde{\theta}_l|M_l)$$

오차: $O(n^{-1})$

**[9] BIC 근사]** (p.10, Eq.12)

$$2\log B_{lm} \approx 2\left[\log\left(L(\mathbf{Y}|\hat{\theta}_l, M_l)\right) - \log\left(L(\mathbf{Y}|\hat{\theta}_m, M_m)\right)\right] - (p_l - p_m)\log N$$

오차: $O(1)$ — Laplace보다 정확도 낮음

---

**모델 공간 탐색 방법:**

**[10] Occam's Window]** (p.11, Eq.14)

```math
A = \left\{M_k : \frac{\max_l P(M_l|\mathbf{Y})}{P(M_k|\mathbf{Y})} \leq c\right\}
```

일반적으로 $c = 20$ 설정 (p-value 0.05 기준 모방)

**[11] SSVS 지시변수]** (p.12, Eq.15)

$$\gamma_l = \begin{cases} 1, & \text{변수 } l \text{이 모델에 포함} \\ 0, & \text{그 외} \end{cases}$$

**[12] MC³ 수용 확률]** (p.12, Eq.16)

```math
\alpha(m,l) = \min\left\{1, \frac{P(M_m|\mathbf{Y})}{P(M_l|\mathbf{Y})}\right\}
```

**[13] 앙상블 예측 (Raftery et al., 2005)]** (p.20, Eq.17)

$$g(y|f_1,\ldots,f_K) = \sum_{l=1}^{K} w_l g_l(y|f_l), \quad \sum_{l=1}^{K} w_l = 1$$

---

### 모델 구조

**CCS 8개 분류 항목 구조** (Table 1, p.34):

| 항목 | 분류 범주 |
|---|---|
| 1. BMA 활용 방식 | 모델 선택 / 결합 추정 / 결합 예측 / 개념 논의 / 리뷰 |
| 2. 적용 분야 | 통계·ML / 물리공학 / 생명과학 / 경제·인문 |
| 3. 모델 사전 분포 | 균등 / 문헌 / 전문가 도출 / NA |
| 4. 증거 추정 방식 | MC / 해석적 근사 / MCMC / 밀도 비율 |
| 5. 고차원 처리 | 차원 축소 / 확률적 탐색(MCMC) |
| 6. MCMC 사용 여부 | Yes / No |
| 7. 시뮬레이션 연구 | Yes / No |
| 8. 데이터 기반 검증 | 교차검증 / K-fold / 사후예측검증 / 없음 |

---

### 성능 향상 및 한계

| 구분 | 내용 |
|---|---|
| **성능 향상** | 로그 손실 하에서 BMA는 단일 모델보다 낮은 예측 위험 달성 (p.4) |
| **성능 향상** | Occam's Window + Leaps&Bounds 조합으로 대규모 모델 공간 효율 탐색 (p.11) |
| **한계 1** | 증거 계산이 일반적으로 해석 불가능 — BIC 근사가 과다 사용되며 복잡 모델에 부적절 (p.29) |
| **한계 2** | 균등 사전 분포 과의존 — 최적이 아닐 수 있음 (p.29) |
| **한계 3** | 회귀 변수 선택 외의 모델 선택 문제 적용 매우 제한적 (p.29) |
| **한계 4** | 검증 연구 부족 — 587편 중 358편이 검증 없음 (p.28) |
| **한계 5** | 계산 비용 논의 부재 — 대규모 데이터·모델 공간에 대한 처리 방안 미흡 (p.29) |

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 위치 |
|---|---|
| BMA의 핵심 수식 (사후 파라미터, 모델 확률) | p.2-3, Eq.(1)-(6) |
| 문헌 검색 절차 및 587편 선정 | p.6, Fig.1 |
| CCS 8개 항목 전체 목록 | p.34, Table 1 |
| 연도별 출판 수 증가 추세 | p.15, Fig.2 |
| 저널별 출판 분포 (상위 100) | p.16-18, Table 2 |
| 상위 10인 저자 목록 | p.35, Table 3 |
| 분야별 BMA 활용 방식 비교 | p.22, Fig.3 |
| MCMC 연도별 활용 비율 | p.27, Fig.4 |
| 시뮬레이션 연도별 비율 | p.28, Fig.5 |
| 모델 사전 분포 분석 (균등 297편) | p.24 |
| 증거 추정 방식 분석 (BIC 190편) | p.25 |
| 결론의 5가지 시사점 | p.29-30 |

---

## 4. 저자 직접 보고 vs. 검토자 해석 분리

### 저자가 직접 보고한 결과

- 587편 논문 분석 (1996–2014), 4개 데이터베이스 활용 (p.6)
- 모델 선택 231편(39.4%), 예측 161편(27.4%), 결합 추정 111편(18.9%) (p.19-20)
- 균등 사전 분포 297편(50.6%), 문헌 사전 98편(16.6%), 전문가 도출 63편 (p.24)
- BIC 근사 190편, MCMC 기반 113편, NA(해석적 해) 232편 (p.25)
- MCMC 사용 261편, 미사용 302편 (563편 기준) (p.26)
- 시뮬레이션 연구 없음 375편, 있음 193편 (568편 기준) (p.27)
- 데이터 검증 없음 358편, 단순 분할 179편, K-fold 17편, 사후예측검증 1편 (p.28)
- 생명과학 201편(34.24%), 경제학 149편(25.38%), 물리공학 146편(24.82%), 통계·ML 92편 (p.21-23)
- Adrian Raftery 34편으로 압도적 1위 (p.18, Table 3)

### 검토자 해석

- **MCMC 활용 정체**: 소프트웨어 보급(BUGS, JAGS)에도 불구하고 MCMC 비율이 ~50%에서 정체된 것은, BIC 기반 분석 도구의 편리성이 복잡한 MCMC 구현의 장벽을 뛰어넘음을 시사한다.
- **검증 부재의 심각성**: 60% 이상의 논문이 검증을 수행하지 않는 것은 BMA가 모델 불확실성 해소를 목적으로 하면서도 방법 자체의 불확실성을 검증하지 않는 방법론적 역설이다.
- **사전 분포 연구의 공백**: 균등 사전 분포의 편의적 채택은 사전 정보가 충분한 분야(예: 의학 임상시험)에서 실질적 손실을 유발할 수 있다.
- **BIC 남용 문제**: BIC는 $O(1)$ 오차를 가지며 정규성 등 규제 조건을 요구하므로, 복잡한 계층 모델이나 비선형 모델에 적용 시 추정 신뢰도 저하 가능성이 있다.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 항목 | 문제점 | ⚠️ 경고 수준 |
|---|---|---|
| **분야별 비율 비교** | 4개 분야 분류가 저자의 주관적 판단에 의존 (p.8) — 생명과학/통계 경계 불명확 | ⚠⚠ |
| **MCMC 비율(46%)** | 24편 논문 제외(설명 불충분) — 제외 기준의 주관성 (p.26) | ⚠ |
| **시뮬레이션 비율** | 19편 불명확으로 제외 — 약 3.2% 손실 (p.27) | ⚠ |
| **연도별 추세(Fig.2)** | 3년 이동평균 사용으로 단기 변동 평활화 — 실제 연도별 급증/급감 은폐 가능 (p.14) | ⚠ |
| **저널 상위 100 집계(Table 2)** | Weather Resources Research 18편(3.07%) — 이는 Raftery et al.(2005) 앙상블 방법의 영향을 반영하며, BMA 전반의 대표성이 아님 | ⚠⚠ |
| **Raftery 34편** | 자기 인용 가능성, 협업 연구 포함으로 단순 비교 불가 (Table 3) | ⚠ |
| **사후 예측 검증 1편** | 극단적으로 낮은 빈도 — 검색어 제한으로 누락 가능성 배제 불가 (p.28) | ⚠⚠ |
| **"BMA explicitly employing" 기준** | 2차 배제 기준의 주관적 적용 가능성 — 116편 제외 (p.5) | ⚠⚠ |

> **⚠ 중요 주의**: 이 논문은 **정량적 메타분석이 아닌 내용 분석(content analysis)** 기반이므로, 제시된 비율들은 통계적 유의성 검증 없이 기술적(descriptive) 수치로만 해석해야 한다.

---

## 6. 문서가 답하지 않는 질문

1. **BMA의 예측 성능 우월성의 실증적 크기**: BMA가 단일 모델 대비 얼마나 더 나은 예측 성능을 보이는가? 논문은 "위험이 낮다"고 주장하나 구체적인 성능 수치 없음 (p.4).

2. **사전 분포 선택이 결과에 미치는 영향의 정량화**: 균등 사전 vs. 전문가 도출 사전이 사후 모델 확률에 미치는 실질적 차이는? 논문은 분류만 할 뿐 비교 없음.

3. **BIC 남용의 실제 오류 규모**: BIC 근사를 부적절하게 사용한 경우(복잡 모델)의 추정 오류가 얼마나 큰가?

4. **계산 비용**: BMA 구현 시 실질적인 계산 시간/자원 소모에 대한 벤치마크 또는 비교 데이터 없음 (p.29에서 미흡이라고 지적만 함).

5. **서로 다른 증거 추정 방법 간의 체계적 비교**: 어느 상황에서 어떤 방법이 최선인가에 대한 실증 비교 없음.

6. **BMA의 모델 오지정(misspecification)에 대한 강건성**: 모든 후보 모델이 틀렸을 경우 BMA의 행동에 대한 논의 없음.

7. **비영어권·비저널 문헌의 BMA 동향**: 컨퍼런스 논문, 학위논문, 단행본 제외로 누락 가능성 (p.29).

8. **2015년 이후 최신 딥러닝 기반 방법과의 관계**: Bayesian Deep Learning, Dropout 기반 근사 BMA 등과의 비교 없음 (논문 출판 시점의 한계).

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.6) — 문헌 검색 절차 플로우차트

**내용**: 4개 데이터베이스 → 1,116개 초기 검색 → 중복 제거 및 1차 기준 적용 → 703편 → 2차 기준 적용(116편 제외) → **최종 587편**.

**해석**: 전체 검색 결과의 약 47.5%가 최종 선정되었다. 2차 배제(116편)는 BMA를 명시적으로 활용하지 않은 논문을 제거한 것으로, 이 기준의 주관성이 일정 편향을 유발할 수 있다. 플로우차트는 PRISMA 가이드라인을 따르며 재현 가능성을 보장한다.

---

### Figure 2 (p.15) — 연도별 출판 수 및 3년 이동평균

**내용**: 1996–2014년 BMA 논문 수를 연도별로 표시하며 3년 이동평균 곡선 중첩.

**해석**: **3단계 성장 패턴** 확인:
- **1996–2000**: 초기 단계 (연 5편 미만) — 방법론은 있으나 소프트웨어·인지도 부족
- **2000–2005**: 성장기 (연 15–25편) — Hoeting et al.(1999), BMA R 패키지 보급의 영향
- **2005–2014**: 가속 성장기 (연 25→75+편) — MCMC 소프트웨어 보편화, 컴퓨팅 파워 증가

검토자 해석: 성장세가 2014년에도 계속되므로 2015년 이후 더 가파른 증가가 예상되었으며, 실제로 현재까지 BMA 관련 연구는 급증하였다.

---

### Figure 3 (p.22) — 분야별 BMA 활용 방식 누적 막대그래프

**내용**: 4개 분야(Med-LS, Phys-Eng, Stat-ML, Hum-Eco) × 5가지 활용 방식(Model Choice, Estimation, Prediction, Discussion, Revision) 비율.

**해석**:
- **Phys-Eng(물리공학)**: 예측(Prediction)이 압도적 — Raftery et al.(2005) 앙상블 예보의 영향
- **Hum-Eco(경제·인문)**: 모델 선택(Model Choice)이 가장 높음 — 성장 결정요인 탐색이 주 목적
- **Stat-ML(통계·ML)**: 개념 논의(Discussion)와 리뷰(Revision) 비율이 상대적으로 높음 — 방법론 개발 중심
- **Med-LS(의료·생명)**: 모델 선택과 결합 추정이 함께 높음 — 진단 모델 및 유전체 연구 반영

검토자 해석: 분야별 활용 방식의 차이는 각 분야의 **주요 연구 질문 구조**를 반영하며, BMA의 범용성을 동시에 보여준다.

---

### Figure 4 (p.27) — 연도별 MCMC 사용 비율

**내용**: 1996–2014년 각 연도의 MCMC 사용(빨간)/미사용(청록) 비율 누적 막대그래프.

**해석**:
- MCMC 사용 비율은 **2000년대 초 일시적 급증 후 ~50%에서 안정화**
- 소프트웨어 보급(BUGS, JAGS)에도 불구하고 MCMC가 지배적이 되지 않음
- BIC 기반 방법의 **간편성**이 MCMC 채택의 장벽을 지속적으로 형성

검토자 해석: 이는 **실용성 vs. 이론적 엄밀성의 트레이드오프**를 보여주며, 사용자 친화적인 MCMC 도구 개발의 필요성을 시사한다.

---

### Figure 5 (p.28) — 연도별 시뮬레이션 연구 비율

**내용**: 1996–2014년 시뮬레이션 수행(빨간)/미수행(청록) 비율.

**해석**:
- 어떤 연도에도 시뮬레이션이 50% 이상이었던 해가 없음
- 전체적으로 시뮬레이션 미수행 논문이 우세 (전체 375/568 ≈ 66%)
- 컴퓨팅 파워 증가에도 불구하고 개선 없음

검토자 해석: BMA 논문이 방법 검증보다 **적용에 치중**하고 있음을 보여주며, 이는 검증 없는 적용이 BMA의 불확실성 처리 목적과 상충한다는 저자들의 비판(p.30)을 뒷받침한다.

---

## 8. 결론: 시사점, 후속 연구 계획, 추가 방향 제시

### 8-1. 저자가 제시한 시사점 (p.29-30)

저자들은 5가지 핵심 관찰을 제시한다:

1. **모델 선택의 편중**: BMA는 이론적으로 어떤 문제에도 적용 가능하나, 실제로는 회귀 변수 선택에 과도하게 집중 → 다양한 모델 구조로의 확장 필요
2. **방법론 발전의 정체**: 1990년대 후반 이후 통계 방법론 발전이 미미 → 복잡한 모델에 적용 가능한 새로운 증거 추정 방법 개발 필요
3. **계산 비용 논의 부재**: 대규모 데이터/모델 공간에서의 계산 효율성 연구 부족
4. **사전 분포 연구 미흡**: 균등 사전 분포 의존도 탈피를 위한 참조 사전(reference prior) 연구 필요
5. **검증 부재**: 시뮬레이션 및 데이터 기반 검증이 부족 → 과도한 신뢰의 위험

### 8-1. 모델 일반화 성능 향상 가능성 (중점)

BMA가 일반화 성능을 향상시키는 이론적 근거는 다음과 같다:

**이론적 근거:**

$$\pi(\Delta|\mathbf{Y}) = \sum_{l=1}^{K} \pi(\Delta|\mathbf{Y}, M_l)\pi(M_l|\mathbf{Y})$$

이 식은 각 모델의 예측 분포를 사후 확률로 가중 평균하여, **단일 모델의 과적합(overfitting) 위험을 분산**시킨다. Madigan & Raftery(1994)가 증명한 바와 같이, 로그 손실 하에서 BMA는 최소 예측 위험을 달성한다.

**일반화 성능 향상을 위한 구체적 방향:**

| 방향 | 설명 |
|---|---|
| **더 나은 검증 프로토콜** | K-fold 교차 검증 의무화 — 17편만 사용한 현 상황 개선 필요 |
| **복잡 모델로의 확장** | 신경망, 비모수 모델에 대한 BMA 적용 가능 증거 추정법 개발 |
| **사후 예측 검증 활성화** | 587편 중 1편만 사용한 Posterior Predictive Check를 표준화 |
| **앙상블 가중치 최적화** | EM 기반 Raftery et al.(2005) 방법의 이론적 최적성 강화 |
| **적응적 사전 분포** | 데이터 기반 사전 분포(empirical Bayes)와의 결합으로 과소/과대추정 방지 |

---

### 8-2. 2020년 이후 최신 연구 비교 분석

> **⚠ 주의**: 아래 내용은 2015년 논문 이후의 연구 동향에 대한 일반적 지식을 바탕으로 한 해석이며, 특정 최신 논문들에 대해 100% 정확성을 보장하기 어려운 부분이 있습니다. 인용 시 원본 논문 직접 확인을 권장합니다.

#### 2020년 이후 주요 연구 동향

| 연구 방향 | 내용 | Fragoso & Louzada와의 관계 |
|---|---|---|
| **Bayesian Deep Learning** | MC Dropout (Gal & Ghahramani, 2016 이후 확산)이 BMA의 근사로 해석됨 | 논문이 예견한 "복잡 모델로의 BMA 확장" 실현 |
| **Stacking/Super-learning** | Bayesian Stacking (Yao et al., 2018)이 로그 손실 최적화 측면에서 BMA와 비교 연구됨 | BMA의 대안으로 제시 — 논문이 지적한 BMA 적용 다양화의 필요성 반영 |
| **대규모 변수 선택** | LASSO-BMA 결합, spike-and-slab prior 발전 | 논문이 지적한 고차원 문제 해결 방향 |
| **자동화 BMA(AutoML)** | R의 `BMS`, `BAS` 패키지 발전, Python 생태계 확장 | 논문이 지적한 소프트웨어 의존성 심화 |
| **비모수적 BMA** | Gaussian Process와 BMA 결합 | 논문의 "(일반화)선형 모델 집중" 한계 극복 방향 |

#### 이 논문이 앞으로의 연구에 미치는 영향

1. **분류 프레임워크 제공**: CCS는 이후 BMA 적용 논문들이 방법론적 선택을 명시하는 표준 체크리스트로 활용 가능
2. **공백 식별**: 복잡 모델 증거 추정, 사전 분포 연구, 검증 방법의 공백을 명확히 제시하여 후속 연구의 로드맵 역할
3. **메타 분석의 기준점**: 2014년까지의 BMA 문헌 현황 스냅샷으로서, 이후 연구들과의 비교를 위한 기준점(baseline)

#### 향후 연구 시 고려할 점

1. **Bayesian Deep Learning과의 통합**: BMA를 신경망 앙상블 관점에서 재해석하고, dropout 기반 근사 BMA의 이론적 기반 강화

2. **확장 가능한 증거 추정**: 대규모 데이터에서 Variational Inference(VI) 기반 증거 추정 방법 개발 — 현재 BIC의 대안

$$\log \pi(\mathbf{Y}|M_l) \geq \mathcal{L}(q) = \mathbb{E}_q[\log L(\mathbf{Y}|\theta_l, M_l)] - KL(q(\theta_l)||\pi(\theta_l|M_l))$$

(변분 하한을 증거 추정에 활용)

3. **공정한 비교 실험 설계**: BMA vs. Stacking vs. Bagging의 체계적 비교를 위한 표준 벤치마크 구축

4. **사전 분포의 자동 선택**: 하이퍼파라미터 사전 분포에 대한 계층적 베이즈 접근 — 균등 사전 의존 탈피

5. **계산 효율화**: GPU 기반 MCMC, 병렬 RJMCMC 구현으로 BMA의 계산 장벽 해소

6. **검증 표준화**: 모든 BMA 적용 논문에서 최소한 K-fold 교차 검증 수행을 권장하는 가이드라인 수립

---

## 참고자료

**주요 인용 논문 (논문 내 참고문헌):**

- Fragoso, T.M. & Louzada Neto, F. (2015). *Bayesian model averaging: A systematic review and conceptual classification*. arXiv:1509.08864v1 [stat.ME]
- Hoeting, J.A., Madigan, D., Raftery, A.E. & Volinsky, C.T. (1999). Bayesian model averaging: a tutorial. *Statistical Science*, 382–401.
- Raftery, A.E. (1996). Approximate Bayes factors and accounting for model uncertainty in generalised linear models. *Biometrika*, 83(2), 251–266.
- Kass, R.E. & Raftery, A.E. (1995). Bayes factors. *Journal of the American Statistical Association*, 90(430), 773–795.
- Madigan, D. & Raftery, A.E. (1994). Model selection and accounting for model uncertainty in graphical models using Occam's window. *JASA*, 89(428), 1535–1546.
- Green, P.J. (1995). Reversible jump Markov chain Monte Carlo computation and Bayesian model determination. *Biometrika*, 82(4), 711–732.
- Raftery, A.E., Gneiting, T., Balabdaoui, F. & Polakowski, M. (2005). Using Bayesian model averaging to calibrate forecast ensembles. *Monthly Weather Review*, 133(5), 1155–1174.
- George, E.I. & McCulloch, R.E. (1993). Variable selection via Gibbs sampling. *JASA*, 88(423), 881–889.
- Wasserman, L. (2000). Bayesian model selection and model averaging. *Journal of Mathematical Psychology*, 44(1), 92–107.
- Chib, S. (1995). Marginal likelihood from the Gibbs output. *JASA*, 90(432), 1313–1321.
- Moher, D. et al. (2009). Preferred reporting items for systematic reviews and meta-analyses: the PRISMA statement. *Annals of Internal Medicine*, 151(4), 264–269.
- Hsieh, H.-F. & Shannon, S.E. (2005). Three approaches to qualitative content analysis. *Qualitative Health Research*, 15(9), 1277–1288.
- Fernandez, C., Ley, E. & Steel, M.F.J. (2001). Benchmark priors for Bayesian model averaging. *Journal of Econometrics*, 100(2), 381–427.
- Friel, N. & Wyse, J. (2012). Estimating the evidence — a review. *Statistica Neerlandica*, 66(3), 288–308.
- Gelfand, A.E. & Dey, D.K. (1994). Bayesian model choice: asymptotics and exact calculations. *JRSS-B*, 501–514.
