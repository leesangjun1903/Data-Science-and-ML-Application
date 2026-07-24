# Conformal Prediction Under Covariate Shift

(1)~(7)은 **제공된 논문 원문**(Tibshirani, Barber, Candès, Ramdas, *"Conformal Prediction Under Covariate Shift"*, arXiv:1904.06019v3, 2020; NeurIPS 2019)에 근거하며, (8-2)의 2020년 이후 최신 연구 비교는 제 학습된 지식(2026년 1월 기준)에 근거합니다. 후자는 라이브 출처로 재검증하지 못했으므로 확신도가 낮은 항목은 명시적으로 표시하겠습니다.

또한 정확성을 위해 미리 정정합니다: **이 논문에는 표(Table)가 없고, 그림(Figure)은 총 4개뿐입니다.** 따라서 요청 7의 "가장 중요한 그림 5개"는 실제로 존재하는 4개 전부를 해석하는 것으로 대신하겠습니다.

---

## 1. Executive Summary (10문장 이내)

1. 이 논문은 conformal prediction을 **교환가능성(exchangeability)이 깨진 데이터**, 특히 covariate shift 상황으로 확장한다.
2. Covariate shift란 학습·테스트의 공변량 분포는 다르지만( $P_X \neq \tilde{P}\_X$ ) 조건부 분포는 동일한( $P_{Y|X}$ 불변) 설정을 말한다 (식 (6)).
3. 핵심 아이디어는 각 학습점의 nonconformity score에 **가능도비(likelihood ratio) $w(x)=d\tilde{P}_X/dP_X$에 비례하는 가중치**를 부여해 가중 경험분포의 분위수를 쓰는 것이다 (Corollary 1, 식 (7)~(8)).
4. 이 가중 방식은 분포 무관(distribution-free)하게 유한표본에서 $1-\alpha$ 이상의 커버리지를 보장한다.
5. 저자들은 이를 **weighted exchangeability**라는 일반 개념으로 정식화하고, covariate shift를 그 특수 사례로 유도한다 (Definition 1, Lemma 2·3, Theorem 2).
6. $w$를 모를 때는 학습/테스트 공변량을 이진 분류 문제로 두고 로지스틱 회귀·랜덤포레스트로 추정한다 (식 (12)).
7. 실증(airfoil, $N=1503$, $d=5$)에서 보정 없는 conformal은 shift 시 커버리지가 90.2%→82.2%로 붕괴하지만, 가중 방식은 90.8%(oracle), 91.0%(추정 가중치)로 복원된다 (Figure 2).
8. 가중치로 인해 유효표본크기 $\hat{n}=\|w(X\_{1:n})\|\_1^2/\|w(X_{1:n})\|_2^2$ 가 줄어 커버리지·구간 길이의 분산이 커진다 (p.6, Figure 2·3).
9. 저자들은 그래프 모델 covariate shift, 결측 공변량, 국소 조건부 커버리지 근사 등 확장 가능성을 논의한다 (Section 4).
10. 결론적으로 covariate shift 하에서는 가중 conformal이 일반 conformal과 **동일한 계산량**으로 유효한 예측구간을 준다는 것이 기여다.

### 1-1. 연구의 목적과 필요성

목적은 **분포 무관·유한표본 예측구간을 covariate shift로 확장**하는 것입니다.  
기존 conformal prediction(Vovk et al., 2005)은 학습·테스트가 교환가능(적어도 동일 분포)이어야 식 (1)의 커버리지 $\mathbb{P}\{Y_{n+1}\in \hat{C}\_n(X_{n+1})\}\ge 1-\alpha$가 성립합니다.  
그러나 현실에서는 배포 환경(테스트)의 입력 분포가 학습과 다른 경우가 흔하고(예: 데이터 수집 편향, 도메인 변화), 이때 일반 conformal은 커버리지 보장을 잃습니다. 필요성은 실증으로 뒷받침되는데, 저자들의 airfoil 실험에서 shift 시 커버리지가 목표 90%에서 82.2%로 떨어집니다(Figure 2 상단, p.6).  
Covariate shift 문헌은 주로 추정량·모델선택 보정에 집중했고, **분포 무관 예측구간 보정은 저자들이 아는 한 새로운 기여**라고 명시합니다 (Remark 4, p.4).

---

## 2. 핵심 주장과 근거 (표)

| # | 핵심 주장 | 근거 | 위치 |
|---|-----------|------|------|
| C1 | 교환가능 데이터에서 conformal은 $1-\alpha \le$ 커버리지 $\le 1-\alpha+\frac{1}{n+1}$ | Lemma 1, Theorem 1 (증명 제시) | p.2 |
| C2 | Covariate shift에서 가중치 $p_i^w(x)$를 쓰면 커버리지 $\ge 1-\alpha$ 회복 | Corollary 1 (이론), 정규화 상수 무관(Remark 3) | p.4 |
| C3 | 위 결과는 **weighted exchangeability**의 특수 사례 | Definition 1, Lemma 2·3, Theorem 2 (증명) | p.11–13 |
| C4 | 보정 없는 conformal은 shift 시 심각한 undercoverage | 90.2% → 82.2% (5000회 평균) | Figure 2 상단, p.6 |
| C5 | Oracle 가중치는 커버리지 복원, 단 분산 증가 | 90.8%, 유효표본크기로 분산 설명됨 | Figure 2 중단, p.6–7 |
| C6 | 추정 가중치(로지스틱/RF)도 유효 커버리지 달성 | 둘 다 91.0% | Figure 2 하단, p.8 |
| C7 | 가중 구간은 동일 유효표본 대비 더 길다(과대 아님, $\mu_0$ 미보정 탓) | Figure 3 중단, p.8 | p.8–9 |
| C8 | Shift 없을 때 가중 적용해도 큰 손해 없음 | Figure 4 | p.10 |
| C9 | 계산: 일반적으로 조합적 난제이나 covariate shift에서는 일반 conformal과 동일 비용 | Section 4 | p.14 |
| C10 | 국소 조건부 커버리지 근사 가능(단, 점 $x_0$마다 재계산 필요) | 식 (18)–(21) | p.15 |

### 2-1. 문제 · 방법(수식) · 모델 구조 · 성능 · 한계

**해결하고자 하는 문제.** 학습 $(X_i,Y_i)\overset{iid}{\sim}P=P_X\times P_{Y|X}$, 테스트 $(X_{n+1},Y_{n+1})\sim \tilde{P}=\tilde{P}\_X\times P_{Y|X}$로 공변량 분포만 다를 때(식 (6), p.3), 분포 무관하게

$$\mathbb{P}\{Y_{n+1}\in \hat{C}_n(X_{n+1})\}\ge 1-\alpha$$

를 만족하는 예측구간을 만드는 것.

**제안 방법.** Nonconformity score (식 (4), 예: $S((x,y),Z)=|y-\hat\mu(x)|$)를 계산하고, 학습점 점질량에 가능도비 비례 가중치를 부여합니다 (식 (7), p.4):

$$p_i^w(x)=\frac{w(X_i)}{\sum_{j=1}^n w(X_j)+w(x)},\qquad p_{n+1}^w(x)=\frac{w(x)}{\sum_{j=1}^n w(X_j)+w(x)},\qquad w=\frac{d\tilde{P}_X}{dP_X}.$$

예측구간(Corollary 1, 식 (8)):

```math
\hat{C}_n(x)=\Big\{y\in\mathbb{R}: V_{n+1}^{(x,y)}\le \mathrm{Quantile}\Big(1-\alpha;\ \textstyle\sum_{i=1}^n p_i^w(x)\,\delta_{V_i^{(x,y)}}+p_{n+1}^w(x)\,\delta_\infty\Big)\Big\}.
```

Split 버전(식 (10), p.4)에서는 $\mu_0$를 사전 데이터로 고정해:

$$\hat{C}_n(x)=\mu_0(x)\pm \mathrm{Quantile}\Big(1-\alpha;\ \textstyle\sum_{i=1}^n p_i^w(x)\,\delta_{|Y_i-\mu_0(X_i)|}+p_{n+1}^w(x)\,\delta_\infty\Big).$$

**일반 구조(weighted exchangeability).** 결합밀도가 $f(v_1,\dots,v_n)=\prod_{i=1}^n w_i(v_i)\cdot g(v_1,\dots,v_n)$($g$는 순열 불변)로 인수분해되면 weighted exchangeable이라 정의합니다 (Definition 1, p.12). 독립·비동일분포 표본은 $w_i=dP_i/dP_1$로 이 조건을 만족하며(Lemma 2), covariate shift는 $w_i\equiv1\ (i\le n),\ w_{n+1}=w(x)$인 특수 사례입니다 (Section 3.5, p.14).

**가중치 추정(식 (12), p.8).** 학습($C=0$)·테스트($C=1$) 라벨로 분류기를 학습하면 $\frac{P(C=1|x)}{P(C=0|x)}=\frac{P(C=1)}{P(C=0)}\cdot\frac{d\tilde{P}_X}{dP_X}(x)$이므로, 정규화 상수를 무시하고

$$\hat{w}(x)=\frac{\hat p(x)}{1-\hat p(x)}.$$

**성능.** 이론적으로 유한표본 하한 보장(엄밀 증명). 실증(airfoil)에서 커버리지: 90.2%(shift 없음) / 82.2%(shift, 미보정) / 90.8%(oracle) / 91.0%(로지스틱, RF 각각) (Figure 2). "향상"이라기보다 **보장 복원**입니다.

**한계.** (a) 상한 커버리지는 가중치 상황에서 일반적으로 보장 불가(Remark 5, p.13, 최대 점프 $\max_i p_i^w$가 클 수 있음). (b) 유효표본 감소로 분산↑. (c) 조합적 난제(일반 가중 conformal). (d) $\mu_0$가 shift에 미보정이면 구간이 길어짐(p.8). (e) 국소 조건부 커버리지는 점마다 재계산 필요(p.15).

---

## 3. 페이지/그림 번호 표시

위 표(2절)와 2-1절에 위치를 병기했습니다. 핵심만 다시: 식 (6)–(8), Corollary 1 → p.3–4 / 유효표본크기 식 → p.6 / 식 (12) → p.8 / Definition 1, Lemma 2·3 → p.11–13 / Theorem 2 → p.13 / Section 3.5 증명 → p.14 / 국소 조건부(식 18–21) → p.15 / Figure 1 → p.6, Figure 2 → p.7, Figure 3 → p.9, Figure 4 → p.10.

---

## 4. 저자 보고 결과 vs. 제 해석 (분리)

**저자가 직접 보고한 것:**
- 이론: Corollary 1, Theorem 2로 유한표본 하한 커버리지 보장(증명 포함).
- 실증 수치(5000 trial 평균): 90.2 / 82.2 / 90.8 / 91.0(로지스틱)·91.0(RF)% (Figure 2).
- 유효표본크기가 감소해 분산이 커지며, 동일 유효표본의 미가중 conformal이 가중 conformal의 과대분산을 "완전히 설명"한다(p.6–7, purple vs orange).
- $\mu_0$를 shift에 맞춰 보정하지 않았기에 oracle 가중 구간이 더 길다(p.8).
- RF는 유연해서 가중치가 상수에서 더 벗어나 분산이 약간 크나, 심한 과적합은 아니다(p.10).

**제 해석(원문에 없는 판단):**
- 이 방법의 이득은 "예측 정확도 향상"이 아니라 **잘못된 신뢰(undercoverage)의 교정**입니다. 커버리지가 82%→91%로 "복원"된 것이지 90%를 넘어선 것은 목표치 근방일 뿐입니다.
- 로지스틱이 데이터생성식 (11) $w(x)=\exp(x^\top\beta)$와 정확히 정합(well-specified)이므로, 실증은 **추정 가중치에 유리하게 세팅된 최선의 시나리오**로 볼 수 있습니다. 잘못 지정된 분류기에서의 강건성은 이 실험만으로 알 수 없습니다.
- 가중치 추정 오류가 커버리지에 미치는 영향은 이론적으로 다루어지지 않았고, 실증도 우호적 조건이라 실무 일반화에는 주의가 필요합니다.

---

## 5. 통계적으로 취약한 부분 · 비교 불가능한 수치

**통계적 취약점:**
- **단일 데이터셋·저차원**: airfoil($N=1503$, $d=5$) 한 개. 고차원·다른 도메인 일반화 근거 없음.
- **인위적 shift**: 지수 틸팅 $w(x)=\exp(x^\top\beta),\ \beta=(-1,0,0,0,1)$(식 (11))로 shift를 인위 생성. 실제 자연 발생 shift가 아님.
- **불확실성 미보고**: 90.2%, 82.2% 등은 5000회 평균만 제시되고 **표준오차·신뢰구간이 없음**. 히스토그램 분산은 보이나 요약 통계의 정밀도는 수치로 없음.
- **추정 가중치 유효성 이론 부재**: 식 (12) 추정 시의 커버리지 보장은 이론이 아니라 실증으로만 제시.
- **상한 커버리지 미보장**: 가중 상황에서 상한이 없음(Remark 5). 즉 과대커버(구간 과대)를 통제하는 보장이 없음.

**비교 불가능/주의 수치:**
- **구간 길이 비교(Figure 3)는 동등 조건이 아님**: oracle 가중 vs 유효표본 축소 미가중 비교에서 $\mu_0$가 shift 미보정이므로 길이 차이를 방법 우열로 해석 불가(저자도 명시, p.8).
- **로지스틱 91.0% vs RF 91.0%**: 소수점 한 자리 동일값이나 분산·길이 특성이 달라(RF가 더 변동적, 일부 극단적 장구간) 평균 커버리지만으로 두 방법을 동급이라 결론 내릴 수 없음(p.8).
- RF는 확률을 [0.01, 0.99]로 **클리핑**해 무한 가중치를 방지했고(각주 3, p.8, 약 2% 케이스), 이 전처리로 인해 두 방법의 수치가 직접 비교 가능하지 않음.

---

## 6. 문서가 답하지 않는 질문

- 가중치 추정 오차 $\hat{w}-w$가 커버리지에 미치는 영향의 **정량적/이론적** 관계는?
- 분류기가 **잘못 지정**되었을 때 커버리지가 얼마나 붕괴하는가?
- **고차원** 공변량에서 밀도비 추정이 어려울 때의 성능은? (Discussion에서 문제 제기만 하고 해결은 미제시)
- $P_{Y|X}$가 변하는 **일반 분포 shift(label shift, concept drift)**에는? (이 논문은 covariate shift에 한정)
- **조건부 커버리지**를 단일 밴드로 모든 $x_0$에서 달성하는 방법은? (불가능성만 언급, 미해결로 남김, p.15)
- 유효표본크기 식이 "휴리스틱"으로 불리는데, 정확한 커버리지-분산 트레이드오프 이론은?
- 다양한 실제 데이터·다양한 shift 강도에 대한 광범위 벤치마크는? (단일 실험만)

---

## 7. 가장 중요한 그림 해석 (전체 4개)

**Figure 1 (p.6)** — Airfoil의 1·5번째 공변량에 대한 $y$ 산점도(상단)와 틸팅 전(검정)·후(파랑) 커널밀도(하단). 오직 1·5차원만 틸팅되며, 상단의 이분산성이 covariate shift가 커버리지에 영향을 줄 수 있음을 시사. *해석: 이 그림은 "문제 존재"의 시각적 동기 부여.*

**Figure 2 (p.7)** — 커버리지 히스토그램. 상단: 미보정 shift(파랑, 평균 82.2%)의 좌측 이동 → undercoverage. 중단: oracle 가중(주황 90.8%)이 목표 복원, 유효표본 축소 미가중(보라)과 겹침 → 분산 원인 설명. 하단: 추정 가중치(로지스틱 회색, RF 초록) 모두 91.0%. *해석: 논문의 핵심 실증 근거. C4·C5·C6 전부 여기서 나옴.*

**Figure 3 (p.9)** — 구간 **길이** 히스토그램. 상단: shift 유무와 무관하게 미가중 길이 동일(학습 절차·데이터 동일). 중단: oracle 가중이 동등 유효표본 미가중보다 김($\mu_0$ 미보정 탓). 하단: RF가 로지스틱보다 변동 크고 일부 매우 김. *해석: 커버리지 복원의 "비용"이 길이·분산 증가임을 보여줌.*

**Figure 4 (p.10)** — shift가 **없을 때** 가중 적용. 미가중(빨강)과 로지스틱(회색)은 거의 동일, RF(초록)만 약간 분산 큼. *해석: 가중 방식이 shift 없는 상황에서도 큰 손해가 없다는 안전성(robustness) 근거. 즉 "일단 가중을 켜도 안전"하다는 실무적 함의.*

(5번째 그림은 원문에 존재하지 않아 생성하지 않습니다.)

---

## 8. 결론: 시사점 · 후속 연구

**저자 제시 시사점(Section 4, p.14–15):** covariate shift에서 가중 conformal은 일반 conformal과 동일 계산량으로 유효 커버리지를 준다.  
$w$ 미지 시 비라벨 테스트 공변량으로 추정 가능. 저자가 제시한 후속 방향:  
(i) **그래프 모델 covariate shift** — $Z\to X\to Y$ 구조에서 저차원 $Z$의 가능도비만 추정( $\tilde{P}\_{Z,X}/P_{Z,X}=\tilde{P}_Z/P_Z$ ).  
(ii) **결측 공변량(요약통계만 공유)** — 병원 A/B 프라이버시 사례에서 $Z$ 가능도비만으로 가중.  
(iii) **국소 조건부 커버리지** — 커널 $K((x-x_0)/h)$를 가중치로 사용(식 18–21), 단 점마다 재계산 필요·단일 밴드 불가.

**제가 제안하는 추가 방향:** 가중치 추정 오차의 유한표본 커버리지 이론화; 잘못 지정된 밀도비 하에서의 강건성 보장; 고차원 밀도비 추정과의 결합(딥러닝 기반 판별기); $P_{Y|X}$ 변화까지 포괄하는 일반 분포 shift; 유효표본 감소를 완화하는 가중치 안정화/정규화.

### 8-1. 모델의 일반화 성능 향상 가능성

**중요한 구분**: 이 방법은 예측 모델 $\mu_0$ 자체의 일반화(정확도)를 높이지 않습니다. 오히려 **불확실성 정량화의 일반화(테스트 분포 하에서의 커버리지 타당성)**를 회복시킵니다. 일반화 관점의 함의는 다음과 같습니다.

- **분포 이동 하 신뢰성**: 학습 분포에 국한된 커버리지 보장을 테스트 분포로 "전이"시켜, 배포 환경에서 예측구간이 오도(misleading)되지 않게 함. 이는 안전한 배포의 전제 조건입니다.
- **점 예측 개선과의 결합 여지**: 저자도 언급하듯(p.8) $\mu_0$를 covariate shift에 맞게(예: importance-weighted 학습) 재적합하면 구간 길이를 줄일 수 있음 → 점 예측 일반화 개선과 커버리지 보정이 **직교적(orthogonal)으로 결합** 가능.
- **유효표본크기 한계**: $\hat{n}=\|w\|_1^2/\|w\|_2^2$가 작아지면 통계적 효율(길이·분산)이 나빠지므로, 가중치 안정화가 일반화 효율 향상의 핵심 지점.

### 8-2. 2020년 이후 최신 연구 비교 (확신도 표시)

아래는 제 학습 지식(2026-01 기준)에 근거하며, **라이브 출처로 재검증하지 못했으므로** 서지 정보는 반드시 직접 확인하시기 바랍니다.

- **확신도 높음 — Barber, Candès, Ramdas, Tibshirani (2023), "Conformal prediction beyond exchangeability," *Annals of Statistics*.** 본 논문 저자들의 직접 후속작으로, 알려진 가능도비를 요구하지 않고 **고정 가중치·비교환 데이터(시계열, 분포 드리프트)**까지 일반화. 본 논문의 weighted exchangeability를 넘어서는 확장.
- **확신도 높음 — Gibbs & Candès (2021), "Adaptive Conformal Inference Under Distribution Shift," NeurIPS.** 온라인으로 $\alpha$를 적응 갱신해 **미지·시변 분포 shift**에 대응. 본 논문이 가정한 "고정·기지 covariate shift"의 강한 제약을 완화.
- **확신도 높음 — Lei & Candès (2021), "Conformal inference of counterfactuals and individual treatment effects," *JRSS-B*.** 인과추론(잠재결과)의 가중 conformal 응용. 본 논문 프레임워크가 인과·처치효과로 확장된 대표 사례.
- **확신도 중간 — Candès, Lei, Ren (2023), "Conformalized survival analysis."** 검열(censoring) 가중을 weighted conformal로 처리.
- **확신도 중간 — Podkopaev & Ramdas (2021), label shift 하 분포 무관 불확실성 정량화.** covariate shift가 아닌 **label shift**로의 확장.
- **확신도 중간 — Prinster 등, "JAWS" (jackknife+ under covariate shift).** split이 아닌 jackknife+ 계열로 가중 아이디어 확장.
- **확신도 낮음 — Fannjiang 등 (2022), "feedback covariate shift" (생체분자 설계, PNAS 계열로 기억).** 테스트 분포가 모델에 의존해 변하는 피드백 상황. *서지 확인 권장.*

**앞으로의 연구에 미치는 영향:** 이 논문은 "가중 재조정으로 conformal을 비교환 상황에 이식한다"는 **템플릿**을 제시했고, 이후 인과추론·생존분석·시계열·label shift·온라인 적응으로 확산되는 출발점이 되었습니다. 핵심 유산은 weighted exchangeability라는 통합 언어입니다.

**향후 연구 시 고려사항:**
- 가능도비를 **정확히 안다**는 가정은 강함 → 추정 오차 전파의 이론화 필요.
- 유효표본 감소(길이 팽창) 완화 기법(가중치 절단·정규화, 자기정규화)의 커버리지-효율 트레이드오프.
- 고차원 밀도비 추정의 신뢰성이 전체 보장의 병목.
- covariate shift 외 일반 shift(concept drift)로 갈수록 정적 보장은 약해지므로 온라인·적응적 접근과의 결합.

---

## 참고자료 / 출처

**1차 출처(분석 대상, 원문):**
- R. J. Tibshirani, R. F. Barber, E. J. Candès, A. Ramdas, *"Conformal Prediction Under Covariate Shift,"* arXiv:1904.06019v3 [stat.ME], 6 Jul 2020 (NeurIPS 2019). — 본 답변 1~7절의 근거.

**원문이 인용한 주요 문헌(유효표본크기·기반 이론):**
- Gretton et al. (2009), "Covariate shift by kernel mean matching."
- Reddi et al. (2015), "Doubly robust covariate shift correction."
- Vovk et al. (2005), *Algorithmic Learning in a Random World.*
- Lei et al. (2018), "Distribution-free predictive inference for regression," *JASA.*
- Dua & Graff (2019), UCI ML Repository (airfoil).

**8-2절 최신 비교(제 학습 지식 기반, 라이브 미검증 — 직접 확인 권장):**
- Barber, Candès, Ramdas, Tibshirani (2023), *Ann. Statist.* — "beyond exchangeability."
- Gibbs & Candès (2021), NeurIPS — Adaptive Conformal Inference.
- Lei & Candès (2021), *JRSS-B* — counterfactuals/ITE.
- Candès, Lei, Ren (2023) — conformalized survival analysis.
- Podkopaev & Ramdas (2021) — label shift.
- Prinster et al. — JAWS.
- Fannjiang et al. (2022) — feedback covariate shift *(서지 확신도 낮음).*
