# Conformalized Quantile Regression

## 1. Executive Summary (10문장 이내)

1. 이 논문은 예측 구간(prediction interval)을 구성하는 **Conformalized Quantile Regression (CQR)** 기법을 제안한다 (초록, p.1).
2. Conformal prediction은 분포 가정 없이 유한 표본에서 유효한 커버리지를 보장하지만, 구간 길이가 입력 공간 전반에서 거의 일정해 불필요하게 보수적이다 (초록; §3, p.4).
3. Quantile regression은 이질분산(heteroscedasticity)에 적응적이지만, 유효 커버리지는 특정 모델·점근 조건에서만 보장된다 (p.1–2).
4. CQR은 두 접근을 결합해 **conformal의 유한 표본 보장**과 **quantile regression의 적응성**을 동시에 얻는다 (p.2).
5. 데이터를 proper training set과 calibration set으로 나눈 뒤, 훈련셋에서 하위·상위 조건부 분위수 $\hat q_{\alpha_{lo}}, \hat q_{\alpha_{hi}}$를 추정한다 (§4, p.4).
6. Calibration set에서 conformity score $E_i$(식 9)를 계산해 초기 구간을 보정(conformalize)한다 (식 10–11, p.5).
7. Theorem 1은 교환가능성(exchangeability)만 가정하면 커버리지 $\geq 1-\alpha$가 성립함을 증명한다 (p.5–6).
8. 임의의 quantile regression 알고리즘(random forest, neural net 등)을 감쌀 수 있어 방법론적으로 유연하다 (p.2, p.6).
9. 11개 벤치마크 · 20회 분할 · 총 2,200회 실험에서 CQR이 평균적으로 가장 짧은 구간을 생성했다 (Table 1, p.10; §6.3).
10. 즉 CQR은 "분포 무관 + 유한 표본 유효 + 이질분산 적응"이라는 세 조건을 동시에 만족하는 절차를 제시한다 (§7 결론, p.11).

### 1-1. 연구의 목적과 필요성
목적은 **① 강한 분포 가정(예: 정규성) 없이 유한 표본에서 유효한 커버리지를 보장하면서, ② 각 지점에서 가능한 한 짧은(정보량이 큰) 예측 구간**을 만드는 절차를 세우는 것이다 (§1, p.1). 필요성은 신약 효능·신용 부도 위험처럼 고위험 의사결정에서 "예측값"뿐 아니라 "예측의 불확실성"을 정량화해야 하기 때문이다 (§1 첫 문단). 기존 conformal은 길이가 고정적이라 이질분산 데이터에서 비효율적이고, 순수 quantile regression은 유효성 보장이 약하다는 공백을 메우려는 동기다 (§1).

---

## 2. 핵심 주장 · 근거 표

| 핵심 주장 | 근거 (저자 보고) | 위치 |
|---|---|---|
| 기존 split conformal 구간은 길이가 $2Q_{1-\alpha}(R,\mathcal{I}\_2)$로 고정되어 $X_{n+1}$에 무관 | 식 (8) 분석, "length is fixed" | §3, p.4 |
| 순수 quantile regression은 유한 표본 커버리지 보장이 없어 undercover 가능 | NN 구간이 "substantially undercover" | p.4 |
| CQR은 conformity score로 초기 구간을 보정해 유효성 확보 | 식 (9)(10)(11) | §4, p.5 |
| CQR은 임의 QR 알고리즘 래핑 가능(RF, NN, 딥러닝) | "wrap around any algorithm" | p.2, p.6 |
| 교환가능성만으로 커버리지 $\geq 1-\alpha$ 보장 | Theorem 1 증명 (Lemma 2 활용) | p.5–6 |
| 상한 커버리지 $\leq 1-\alpha + \frac{1}{ \mid \mathcal{I}_2 \mid +1}$ (score 서로 다를 때) | Theorem 1 후반부 | p.6 |
| 좌·우 꼬리 독립 제어(비대칭 보정) 가능 | Theorem 2 | p.7 |
| CQR이 평균 구간 길이에서 최고 성능 | Table 1: CQR NN 1.40, CQR RF 1.41 | p.10 |
| 11개 중 10개 데이터셋에서 두 경쟁 기법 모두 능가 | §6.3 | p.10–11 |

### 2-1. 상세 설명

**(A) 해결하고자 하는 문제.** 식 (1)의 marginal 커버리지 $P\{Y_{n+1}\in C(X_{n+1})\} \geq 1-\alpha$를 유한 표본·분포 무관으로 보장하면서, 이질분산에 맞춰 구간 길이를 국소적으로 조정하는 것 (§1).

**(B) 제안 방법 (수식).**
분위수 함수 정의:

$$q_\alpha(x) := \inf\{y\in\mathbb{R}: F(y\mid X=x)\geq \alpha\}$$

Pinball(check) loss로 분위수 추정 (식 5–6):

$$\hat q_\alpha(x)=f(x;\hat\theta),\quad \hat\theta=\arg\min_\theta \frac{1}{n}\sum_{i=1}^n \rho_\alpha(Y_i, f(X_i;\theta))+\mathcal{R}(\theta)$$

$$\rho_\alpha(y,\hat y)=\begin{cases}\alpha(y-\hat y) & y-\hat y>0\\ (1-\alpha)(\hat y - y) & \text{otherwise}\end{cases}$$

Conformity score (식 9):

$$E_i := \max\{\hat q_{\alpha_{lo}}(X_i)-Y_i,\ Y_i-\hat q_{\alpha_{hi}}(X_i)\}$$

보정된 구간 (식 10–11):

$$C(X_{n+1})=\Big[\hat q_{\alpha_{lo}}(X_{n+1})-Q_{1-\alpha}(E,\mathcal{I}_2),\ \hat q_{\alpha_{hi}}(X_{n+1})+Q_{1-\alpha}(E,\mathcal{I}_2)\Big]$$

$$Q_{1-\alpha}(E,\mathcal{I}_2):=(1-\alpha)\Big(1+\tfrac{1}{|\mathcal{I}_2|}\Big)\text{-th empirical quantile of } \{E_i\}$$

$E_i$는 언더커버리지(구간 밖)일 때 양수 오차, 오버커버리지(구간 안)일 때 음수가 되어 **양방향 오차를 모두 부호로 반영**한다 (p.5).

**(C) 모델 구조.** 

① 데이터를 $\mathcal{I}\_1$(proper training), $\mathcal{I}\_2$(calibration)로 분할 → ② $\mathcal{I}\_1$에서 $\hat q_{\alpha_{lo}}, \hat q_{\alpha_{hi}}$ 학습 → ③ $\mathcal{I}\_2$에서 $E_i$ 계산 → ④ $Q_{1-\alpha}(E,\mathcal{I}_2)$로 구간 확장/축소 (Algorithm 1, p.6). 신경망 구현 시 하위·상위 분위수를 2차원 출력으로 공유하여 계산 비용 절감 (p.7, 항목 2).

**(D) 성능 향상.** Table 1 (α=0.1, 11 데이터셋 평균):

| Method | Avg. Length | Avg. Coverage |
|---|---|---|
| Ridge | 3.06 | 90.03 |
| Ridge Local | 2.94 | 90.13 |
| Random Forests | 2.24 | 89.99 |
| Random Forests Local | 1.82 | 89.95 |
| Neural Net | 2.16 | 89.92 |
| Neural Net Local | 1.81 | 89.95 |
| **CQR Random Forests** | **1.41** | **90.33** |
| **CQR Neural Net** | **1.40** | **90.05** |
| *Quantile Random Forests | *2.23 | *92.62 |
| *Quantile Neural Net | *1.49 | *88.51 |

CQR이 커버리지 90%를 지키면서 최단 길이를 달성 (p.9–10). Quantile crossing 후처리 시 CQR NN 길이 1.40→1.35로 소폭 개선 (p.10).

**(E) 한계 (저자 보고).** ① Facebook 데이터셋에서 conformity score에 동점(ties)이 있어 Theorem 1의 상한이 적용되지 않고 과도하게 보수적 (§6.3, p.11). ② Theorem 2의 비대칭 보정은 커버리지 강화 대가로 구간이 길어짐: CQR NN 1.40→1.58, CQR RF 1.41→1.57 (p.10).

---

## 3. 페이지 / Figure / Table 번호 표기
위 1·2절 표와 아래 서술에 각 주장의 위치(§, p., Figure, Table, 식 번호)를 모두 병기했습니다.

---

## 4. 저자 보고 결과 vs 제 해석 (분리)

**연구 주제**
- 저자 보고: conformal prediction + quantile regression 결합으로 적응적이면서 유효한 구간 생성 (초록).
- 제 해석: 이 논문은 이후 "adaptive conformal" 계열의 사실상 표준 baseline이 되는 성격을 가진다(널리 인용됨). — *이는 제 판단이며 논문 본문 주장 아님.*

**방법 (수식)**
- 저자 보고: 식 (9)(10)(11)의 conformity score와 inflated quantile이 핵심이며, Theorem 1로 유효성 증명 (p.5–6).
- 제 해석: 식 (9)의 $\max$ 구조가 "부호 있는 잔차"를 만들어 오버커버리지를 음수로 흡수하는데, 이 점이 순수 QR 대비 CQR이 더 짧아진 주된 메커니즘으로 보인다. 저자도 "signed conformity scores"로 언급하나(p.10), 그 기여도의 정량 분해(ablation)는 제시하지 않았다.

**결과**
- 저자 보고: 11개 중 10개에서 CQR 우세, 평균 길이 최단 (Table 1, §6.3).
- 제 해석: CQR이 데이터를 덜 쓰는(calibration 분리) 조건에서도 전체 데이터를 쓴 순수 QR을 이긴 것은 인상적이나, 이는 순수 QR의 과대커버(RF 92.62%)에서 오는 "길이 절약"과 분위수 크로스밸리데이션 튜닝이 함께 작용한 결과이므로, "동일 조건 우위"로 일반화하기엔 근거가 제한적이다. — *제 해석.*

---

## 5. 통계적으로 취약한 부분 · 비교 불가능한 수치

- **커버리지가 다른 방법 간 길이 직접 비교의 한계:** Quantile Random Forests는 92.62%로 과대커버, Quantile Neural Net은 88.51%로 과소커버 (Table 1). 커버리지가 명목 90%에서 벗어난 방법의 길이는 **CQR의 90% 수준 길이와 직접 비교 불가**(짧아도 유효성 미달일 수 있음). 저자도 별표(*)로 "유한 표본 보장 없음"을 명시 (Table 1 각주).
- **동점(ties) 문제:** Facebook 데이터셋에서 score 동점으로 Theorem 1 상한 미적용 → 해당 CQR RF 결과는 과보수라 다른 데이터셋과 동일선상 비교가 어렵다 (§6.3).
- **분산/불확실성 미보고:** Table 1은 평균만 제시하고 표준오차·신뢰구간을 수치로 주지 않는다(Figure 3–6의 boxplot으로 산포를 시각화할 뿐). 방법 간 차이의 통계적 유의성 검정은 없다. — *제 지적.*
- **스케일 정규화로 인한 절대 길이의 의미 제한:** 응답변수를 평균 절댓값으로 나눠 재척도화(§6)했으므로 "길이" 수치는 데이터셋 간 절대 비교보다 상대 비교에 적합하다. — *제 해석.*
- **marginal vs conditional:** 보장은 marginal 커버리지(식 1)뿐이며, 조건부(각 $x$별) 커버리지는 보장되지 않는다.

---

## 6. 문서가 답하지 않는 질문

- proper training : calibration 최적 분할 비율은? (실험은 동일 크기만 사용, §6)
- 교환가능성이 깨지는 분포 변화(covariate/label shift) 상황의 성능은? (본문은 §7에서 후속 [50]만 언급)
- 조건부 커버리지(conditional coverage)를 얼마나 달성하는가?
- Full(비분할) conformal 변형의 실제 성능 수치는? (각주 2에서 존재만 언급)
- calibration set 크기가 구간 길이에 미치는 정량적 영향은?
- 계산 비용의 정량 비교(런타임)는?
- 고차원 $p$에서의 거동, 분위수 크로스밸리데이션 튜닝의 커버리지 안정성 정량 분석은?

---

## 7. 가장 중요한 그림 5개 해석

1. **Figure 1 (p.3) — Pinball loss.** $z=y-\hat y$에 대해 기울기가 $\alpha$(우), $1-\alpha$(좌)인 비대칭 손실. 분위수 추정이 왜 비대칭 페널티로 조건부 분위수를 잡는지를 시각화. CQR의 학습 목표(식 5–6)의 근간.
2. **Figure 2 (p.5) — 시뮬레이션 비교.** (a) split(고정 길이 2.91), (b) local(2.86, 부분 적응), (c) CQR(1.99, 강한 적응), (d) 길이의 $X$-의존성 곡선. CQR 구간이 $X$에 따라 넓어지고 좁아지며, 분위수 추정 경계가 보정 경계와 거의 일치함을 보여 **적응성의 핵심 증거**.
3. **Figure 3 (p.14, meps_19/20/21) — 데이터셋별 길이·커버리지 boxplot.** 세 방법 색상(빨강 split / 회색 local / 파랑 CQR)으로 CQR이 커버리지 90% 부근을 유지하며 최단 길이임을 반복 확인. Figure 4–6도 동일 형식(§6.3).
4. **Figure 7 (p.19) — 전체 범위 산점도.** 식 (18)의 마지막 항이 만든 소수의 큰 이상치(outlier)를 −60~40 범위로 보여줌. Figure 2가 왜 잘린 범위였는지, 그리고 CQR의 이상치 견고성 논의의 배경.
5. **Figure 8 (p.19) — 조건부 중앙값 기반 local 변형.** median 추정(QR forest)으로 local conformal을 구성해도 평균 길이 2.86으로 Figure 2b와 거의 동일 → "평균 대신 중앙값" 교체만으로는 적응성 개선이 미미함을 보이는 대조 실험.

(참고: 정량 핵심 결과는 그림이 아닌 **Table 1**에 있음.)

---

## 8. 결론 · 시사점 · 후속 연구

**저자 제시 시사점·후속 계획 (§7, p.11):**
- CQR은 교환가능성이라는 약한 가정만으로 유한 표본 miscoverage를 제어하면서 이질분산에 적응한다는 점을 결론으로 강조.
- 후속 방향으로 **conformal predictive distributions**([49])로의 확장(구간이 아닌 예측 확률분포 추정)을 제시하고, 동시기 독립 연구 [17]과의 연결을 언급.

**추가 후속 연구 방향 (제 제안):** 조건부 커버리지 강화, 분포 변화 하 견고성, calibration 크기·분할 비율 자동화, 동점 처리(무작위화 tie-breaking) 등.

### 8-1. 모델의 일반화 성능 향상 가능성 (중점)
- 저자 논거(p.8, "Limitations"): local conformal은 **훈련 잔차가 최적화로 편향**되어 테스트 오차를 과소추정 → 적응성 손실. 반면 CQR은 목표가 조건부 분위수 추정이라 충분한 데이터에서 두 분위수를 안정적으로 근사, **일반화(테스트 커버리지)와 짧은 길이를 함께** 얻는다는 논리.
- 일반화 관점 강점: (i) underlying 알고리즘 무관하게 커버리지 보장 → 모델 오지정(misspecification)에 견고 (Theorem 1); (ii) 분위수 수준을 CV로 튜닝해도 보장 유지(p.6–7) → 과적합 위험이 있는 딥러닝 QR에도 안전망 제공.
- 제 해석: 다만 보장은 marginal이므로, 새로운 분포(공변량 변화)에서의 "일반화"는 별도 기법(가중 conformal 등)이 필요하다.

### 8-2. 2020년 이후 최신 연구 비교 (불확실성 표시)

> **주의:** 현재 세션에서 웹 검색을 사용할 수 없어, 아래는 제 학습 지식(대략 2026년 1월까지)에 근거합니다. 연도·서지 세부는 부정확할 수 있으니 **원문 확인을 권장**합니다. 확신이 낮은 항목은 그렇게 표기했습니다.

- **분포 변화 대응:** *"Conformal Prediction Under Covariate Shift"* (Tibshirani, Foygel Barber, Candès, Ramdas) — 가중치로 교환가능성 완화. 본 논문 참고문헌 [50]이 이 계열의 초기 버전(2019)으로 보임(확신 보통).
- **시계열/온라인:** *"Adaptive Conformal Inference Under Distribution Shift"* (Gibbs & Candès, 2021년경) — 시간에 따라 $\alpha$를 온라인 갱신. CQR과 결합 가능. (연도 확신 보통)
- **조건부 커버리지 개선:** Feldman·Bates·Romano의 CQR 확장 계열(orthogonal/conditional quantile regression 방향)이 있다고 기억하나, **정확한 제목·연도는 확신하지 못함**(추정하지 않겠습니다).
- **개론·표준화:** Angelopoulos & Bates, *"A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"* (2021~2023) — CQR을 표준 예시로 소개(확신 높음, 정확한 판본/연도는 확인 필요).
- **분포적 예측:** Chernozhukov 등의 distributional/conformal 관련 연구가 §7에서 저자가 예고한 방향과 맞물림(세부 확신 보통).

**본 논문이 이후 연구에 미친 영향 (제 해석):** CQR은 "conformal + quantile regression"을 결합한 대표 절차로, 이후 회귀 불확실성 정량화 연구의 강력한 baseline이자 확장 출발점이 되었다고 평가됨(널리 인용). **향후 연구 시 고려점:** (1) 교환가능성 가정의 현실적 성립 여부, (2) marginal→conditional 커버리지 격차, (3) 동점·이산 반응 처리, (4) calibration 크기가 유효성·효율에 미치는 트레이드오프, (5) 딥러닝 QR의 분위수 튜닝 안정성.

---

## 참고 자료 (출처)

**1차 출처 (직접 근거):**
- Y. Romano, E. Patterson, E. J. Candès, *"Conformalized Quantile Regression,"* arXiv:1905.03222v1 [stat.ME], 8 May 2019. (본 분석의 모든 §·식·Figure·Table 인용)
- 논문 내 인용: [18] Koenker & Bassett, *Regression Quantiles* (1978); [22] Meinshausen, *Quantile Regression Forests* (2006); [15] Lei et al., *Distribution-Free Predictive Inference for Regression* (2018); [24] Steinwart & Christmann (2011); [50] Barber, Candès, Ramdas, Tibshirani, *Conformal Prediction Under Covariate Shift*, arXiv:1904.06019 (2019); [17] Vovk et al., *Conformal Calibrators*, arXiv:1902.06579 (2019); [49] Vovk et al., *Nonparametric Predictive Distributions Based on Conformal Prediction* (2017).

**8-2에서 언급한 외부 문헌 (제 학습 지식 기반, 서지 세부 미검증):**
- Gibbs & Candès, *Adaptive Conformal Inference Under Distribution Shift* (2021년경).
- Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*.
- (조건부 커버리지 개선 계열 논문은 제목·연도 확신 부족으로 구체 서지 미기재.)

정확도를 위해, 8-2의 외부 문헌은 검색 기능을 켜시면 서지 정보를 확인해 드릴 수 있습니다. 원하시면 이 분석을 Word/PDF 파일로도 정리해 드리겠습니다.
