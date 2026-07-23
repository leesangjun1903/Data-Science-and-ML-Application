# Conformalized Quantile Regression (CQR) 

---

## 1. Executive Summary (10문장 이내)

본 논문은 Conformal Prediction의 유한 표본 커버리지 보장과 Quantile Regression의 이분산성(heteroscedasticity) 적응력을 결합한 새로운 예측구간 생성 방법인 CQR을 제안한다.  
기존 conformal 방법들은 입력 공간 전체에서 거의 일정한 폭의 구간을 생성하여 비효율적인 반면, quantile regression은 국소적으로 적응적이지만 유한 표본에서 커버리지가 보장되지 않는 문제가 있었다.  
CQR은 calibration set에서 conformity score $E_i = \max\{\hat{q}\_{\alpha_{lo}}(X_i) - Y_i, Y_i - \hat{q}\_{\alpha_{hi}}(X_i)\}$를 계산하여 quantile 추정치를 보정함으로써 두 방법의 장점을 모두 취한다.  
저자들은 exchangeability 가정 하에서 $P\{Y_{n+1} \in C(X_{n+1})\} \geq 1-\alpha$가 성립함을 증명하였다(Theorem 1).  
11개의 회귀 벤치마크 데이터셋에 대한 2,200회의 실험에서 CQR은 표준 conformal 및 locally adaptive conformal 방법보다 평균적으로 더 짧은 구간을 생성하였다(Table 1).  
흥미롭게도 CQR은 calibration을 위해 데이터를 분할함에도 불구하고, 전체 데이터로 학습한 비정규화(non-conformalized) quantile regression보다도 더 우수한 성능을 보였다.  
CQR은 random forest, neural network 등 임의의 quantile regression 알고리즘에 wrapping 가능한 모델-불가지론적(model-agnostic) 프레임워크이다.  
논문은 또한 좌우 tail을 독립적으로 제어하는 비대칭 conformalization(Theorem 2)도 제시하였으나, 이는 구간 길이 증가를 대가로 한다. 이 연구는 2019년 5월 arXiv에 게재된 preprint("Work in progress")이다.

### 1-1. 연구의 목적과 필요성

**목적**: 분포에 대한 가정 없이(distribution-free) 유한 표본에서 명목 커버리지($1-\alpha$)를 보장하면서도, 이분산적 데이터에서 입력값에 따라 폭이 적응적으로 변하는 짧은 예측구간을 생성하는 방법을 개발하는 것이다(p.1, Introduction).

**필요성**: 저자들은 "이상적인 예측구간 생성 절차는 두 가지 속성을 만족해야 한다"고 명시한다: 

(1) 강한 분포 가정 없이 유한 표본에서 유효한 커버리지를 제공해야 하고, (2) 각 지점에서 구간이 가능한 짧아야 한다(p.1).  
신약의 효능 추정, 신용 부도 위험 평가와 같은 고위험 의사결정에서 이러한 요구가 특히 중요하다고 설명한다(p.1).  
기존 conformal 방법은 첫 번째 조건은 만족하나 두 번째 조건에서 취약하고("[6, 15, 17]에서 주장된 바와 같이, 기존 방법들은 고정된 길이 또는 예측변수에 약하게만 의존하는 길이의 conformal 구간을 산출한다", p.1), quantile regression은 두 번째 조건은 만족하나 특정 모델과 점근적 조건 하에서만 첫 번째 조건이 보장된다(p.1-2).

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 페이지/Figure/Table |
|---|---|---|
| CQR은 유한 표본에서 marginal coverage를 보장한다 | Theorem 1 증명 (exchangeability 기반 Lemma 2 활용) | p.5, Theorem 1 |
| CQR은 기존 conformal 방법보다 짧은 구간을 생성한다 | 11개 데이터셋, 20회 반복 실험, 평균 길이 비교 | Table 1 (p.10), Figure 3-6 |
| CQR은 비정규화 quantile regression보다도 우수하다 | Table 1에서 CQR Neural Net(1.40) vs Quantile Neural Net(1.49, coverage 88.51%) | Table 1 (p.10) |
| Locally adaptive 방법은 근본적 통계적 한계를 가짐 | training residual이 test residual보다 편향되어 과소추정 초래 | p.8, Section 5 "Limitations" |
| 비대칭 conformalization(Theorem 2)은 더 강한 보장이나 더 긴 구간 초래 | CQR NN: 1.40→1.58, CQR RF: 1.41→1.57 | p.10, Section 6.2 |
| 시뮬레이션에서 CQR이 outlier가 있는 이분산 데이터에 특히 효과적 | Split 2.91 vs Local 2.86 vs CQR 1.99 | Figure 2 (p.5) |

---

## 2-1. 상세 설명

### 해결하고자 하는 문제
기존 conformal prediction (split conformal, 식 8)의 구간 길이는 $2Q_{1-\alpha}(R, \mathcal{I}\_2)$로 **입력 $X_{n+1}$과 무관하게 고정**된다(p.4). Locally adaptive 변형은 부분적으로 개선하지만, training set의 residual이 최적화 과정에서 편향되어 있어 (특히 딥러닝처럼 overfitting하는 모델에서) 적응력이 제한된다(p.8).

### 제안하는 방법 (수식 포함)

**1단계**: 데이터를 proper training set $\mathcal{I}_1$과 calibration set $\mathcal{I}_2$로 분할.

**2단계**: $\mathcal{I}_1$에서 quantile regression 알고리즘 $\mathcal{A}$로 하위/상위 조건부 quantile 함수를 적합:

$$\{\hat{q}_{\alpha_{lo}}, \hat{q}_{\alpha_{hi}}\} \leftarrow \mathcal{A}(\{(X_i, Y_i): i \in \mathcal{I}_1\})$$

이때 quantile regression 자체는 pinball loss를 최소화:

$$\hat{q}_\alpha(x) = f(x; \hat{\theta}), \quad \hat{\theta} = \arg\min_\theta \frac{1}{n}\sum_{i=1}^n \rho_\alpha(Y_i, f(X_i;\theta)) + \mathcal{R}(\theta)$$

$$\rho_\alpha(y, \hat{y}) := \begin{cases} \alpha(y-\hat{y}) & \text{if } y-\hat{y} > 0 \\ (1-\alpha)(\hat{y}-y) & \text{otherwise} \end{cases}$$

**3단계**: $\mathcal{I}_2$에서 conformity score 계산:

$$E_i := \max\{\hat{q}_{\alpha_{lo}}(X_i) - Y_i, \ Y_i - \hat{q}_{\alpha_{hi}}(X_i)\}$$

**4단계**: 새로운 예측구간:

$$C(X_{n+1}) = [\hat{q}_{\alpha_{lo}}(X_{n+1}) - Q_{1-\alpha}(E,\mathcal{I}_2), \ \hat{q}_{\alpha_{hi}}(X_{n+1}) + Q_{1-\alpha}(E,\mathcal{I}_2)]$$

여기서 $Q_{1-\alpha}(E,\mathcal{I}_2)$는 $\{E_i : i \in \mathcal{I}_2\}$의 $(1-\alpha)(1+1/|\mathcal{I}_2|)$번째 경험적 분위수.

**비대칭 확장 (Theorem 2)**:

$$C(X_{n+1}) := [\hat{q}_{\alpha_{lo}}(X_{n+1}) - Q_{1-\alpha_{lo}}(E_{lo}, \mathcal{I}_2), \ \hat{q}_{\alpha_{hi}}(X_{n+1}) + Q_{1-\alpha_{hi}}(E_{hi}, \mathcal{I}_2)]$$

### 모델 구조
- **CQR Random Forests**: Quantile Regression Forests [Meinshausen, 2006] 사용, coverage 조절용 2개 추가 하이퍼파라미터를 cross-validation으로 튜닝(p.9).
- **CQR Neural Net**: 3-layer fully connected network(64 hidden units×2), ReLU 활성화, 출력층을 2차원(하위/상위 quantile)으로 하여 파라미터 공유. Adam optimizer, learning rate $5\times10^{-4}$, dropout 0.1, pinball loss 사용(p.9).

### 성능 향상
Table 1(p.10) 기준, CQR Random Forests(길이 1.41)와 CQR Neural Net(길이 1.40)이 Random Forests(2.24), Neural Net(2.16), 그 Local 변형(1.82, 1.81)보다 모두 짧다. 11개 데이터셋 중 **10개**에서 CQR이 두 경쟁 방법을 모두 능가한다(p.10-11).

### 한계
1. **Quantile crossing 문제**: 하위/상위 quantile을 별도로 추정하므로 교차 가능성 존재(neural net에서 발생, forest에서는 미발생, p.10).
2. **Tie 문제**: Facebook 데이터셋에서 conformity score 간 동점(ties)이 발생해 Theorem 1의 상한이 적용되지 않아 CQR Random Forests가 과도하게 보수적임(p.11).
3. Data splitting으로 인한 표본 효율성 손실(명시적으로 각주 2에서 "split을 요구하지 않는 변형도 있다"고 언급하나 본문에서 다루지 않음, p.2).
4. 비대칭 conformalization은 커버리지 보장이 강화되나 구간이 더 길어짐(trade-off, p.10).

---

## 3. 주장별 페이지/Figure/Table 표시

| 주장 | 위치 |
|---|---|
| 이상적 예측구간의 두 조건 | p.1, Introduction 첫 단락 |
| 기존 conformal의 한계 | p.1, 인용 [6,15,17] |
| Theorem 1 (커버리지 보장) | p.5 |
| Theorem 2 (비대칭 보장) | p.7 |
| Locally adaptive의 통계적 한계 | p.8 |
| 시뮬레이션 결과 (2.91 vs 2.86 vs 1.99) | Figure 2, p.5 |
| 종합 실험 결과 | Table 1, p.10 |
| 개별 데이터셋 결과 | Figure 3(MEPS), 4(blog/bio/bike), 5(community/star/concrete), 6(facebook), p.14-17 |
| Quantile crossing 영향 | p.10, 마지막 단락 |
| 비대칭 conformalization 비용 | p.10 |

---

## 4. 저자 보고 결과 vs 나의 해석

| 구분 | 내용 |
|---|---|
| **저자 보고 (사실)** | Table 1: CQR RF 1.41 / CQR NN 1.40 평균 길이, coverage 90.33%/90.05% (11개 데이터셋 평균) |
| **저자 보고 (사실)** | "CQR selects quantiles below the nominal level" — cross-validation으로 명목 quantile보다 낮은 값을 선택함(p.10) |
| **저자 보고 (사실)** | Quantile Neural Net는 coverage 88.51%로 유의미하게 undercover(Table 1) |
| **나의 해석** | CQR의 우수성은 상당 부분 "signed conformity score" (식 9)가 과대/과소 커버리지를 모두 보정할 수 있다는 데서 기인하며, 이는 단순 절대값 residual 기반 방법보다 정보 손실이 적은 보정 메커니즘으로 볼 수 있음 |
| **나의 해석** | Random Forests 기반 quantile regression이 neural network보다 안정적인 이유는 forest의 quantile 추정이 leaf 내 empirical distribution에 기반해 자연스럽게 monotonic(quantile crossing 없음)하기 때문으로 추정됨 |
| **나의 해석** | Table 1에서 "*" 표시된 비정규화 방법과 CQR을 직접 비교하는 것은 통계적으로 완전히 공정하지 않을 수 있음(후술 5번 항목 참고) |

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

1. **비정규화 방법과의 비교 불공정성**: Quantile Random Forests/Neural Net은 전체 훈련 데이터(calibration 없이)로 학습되어 CQR보다 많은 데이터를 사용함(p.9, 각주 없음). 저자도 "이는 놀라울 수 있다"고 인정하나(p.9-10), 커버리지 보장이 없는 방법과 있는 방법의 "길이"를 직접 비교하는 것은 방법론적으로 이질적임(Table 1의 asterisk 표시).

2. **Facebook 데이터셋의 CQR RF 과잉 보수성**: "there are ties among the conformity scores and so the upper bound in Theorem 1 does not apply"(p.11)—즉 이론적 상한이 깨지는 상황이 실제 발생했음을 저자가 인정하나, 정량적으로 얼마나 벗어나는지에 대한 통계적 분석(신뢰구간, 표준오차)은 제공되지 않음(Figure 6 참조).

3. **표준오차/신뢰구간 부재**: Table 1은 평균값만 제시하며, 20회 반복에 대한 분산/표준오차가 명시적으로 보고되지 않음(단, Figure 3-6의 boxplot에서 분포는 시각적으로 확인 가능하나 수치화되지 않음).

4. **하이퍼파라미터 $\gamma=1$ 선택의 임의성**: locally adaptive 방법에서 " $\gamma=0$보다 성능이 상당히 개선된다"고만 서술(p.9)하며, $\gamma$ 값에 대한 체계적 sensitivity 분석 부재.

5. **Quantile crossing 후처리 효과의 제한적 보고**: "coverage rates remaining about the same"(p.10)이라고만 언급되며 구체적 coverage 수치 비교 불가.

---

## 6. 문서가 답하지 않는 질문

1. Data splitting 없이(full conformal 방식) CQR을 적용할 경우의 성능은 어떠한가? (각주 2에서 언급만 하고 다루지 않음)
2. Covariate shift 상황에서 CQR의 커버리지는 어떻게 되는가? (참고문헌 [50] Barber et al. 언급되나 본 논문에서 실험 없음)
3. 다변량 응답변수(multivariate $Y$)에 대한 확장 가능성은?
4. Conformal predictive distribution(Vovk et al. [49])과의 구체적 연결 방법은 무엇인가? (Conclusion에서 "흥미로운 연결"만 언급, p.11)
5. Calibration set 크기 $|\mathcal{I}_2|$가 CQR 성능(특히 tie 문제)에 미치는 구체적 영향은?
6. Cross-validation으로 quantile level을 튜닝할 때 이것이 실제로 finite-sample guarantee를 해치지 않는다는 것에 대한 엄밀한 증명이 부재(단언만 있음, p.7).

---

## 7. 가장 중요한 5개 Figure 해석

### **Figure 1 (p.3)**: Pinball Loss 시각화
$\rho_\alpha(z)$의 비대칭 형태를 보여줌. $\alpha \neq 0.5$일 때 과대추정과 과소추정에 다른 가중치를 부여하여 quantile 추정을 가능케 하는 핵심 손실함수. CQR의 이론적 기반이 되는 quantile regression의 작동 원리를 설명.

### **Figure 2 (p.5)**: 시뮬레이션 비교 (핵심 그림)
이분산적 데이터+outlier 상황에서 (a) split conformal(고정폭 2.91), (b) locally adaptive(2.86), (c) CQR(1.99)을 비교. **CQR만이 $X$에 따라 구간 폭이 실제로 좁아지고 넓어지는 것을 시각적으로 확인**할 수 있으며, (d) 패널에서 세 방법의 길이 함수를 직접 겹쳐 비교함으로써 CQR의 적응성 우위를 명확히 입증. 이 논문의 핵심 motivating example.

### **Table 1 (p.10)**: 종합 성능 표
11개 데이터셋 전체에 대한 평균으로, CQR(볼드체)이 모든 conformal 경쟁자보다 짧은 평균 길이(1.40-1.41)를 가지면서도 명목 커버리지(90%)에 가장 근접함을 보여주는 논문의 핵심 정량적 근거.

### **Figure 3 (MEPS 데이터셋, p.14)**: 실사용 사례 검증
의료비 지출 데이터(현실 세계, 강한 이분산성 예상)에서 CQR Neural Net/RF가 다른 모든 방법보다 일관되게 짧은 길이(2.36-2.51)를 보이며 커버리지도 90% 근처에 안정적으로 유지됨을 확인. 세 개의 유사 데이터셋(MEPS_19/20/21)에서 결과가 일관되어 재현성을 시사.

### **Figure 6 (Facebook 데이터셋, p.17)**: 한계 사례
CQR Random Forests가 facebook_1/2에서 다른 방법 대비 큰 우위(길이 1.16-1.34)를 보이나 **coverage가 90%를 넘어 이상치를 보임**(우측 패널에서 CQR RF 박스가 다른 방법보다 오른쪽으로 치우침). 이는 conformity score의 tie 문제로 인한 이론적 상한 위반 사례를 시각적으로 보여주는 유일한 그림으로, 방법의 한계를 명시적으로 드러냄.

---

## 8. 결론: 시사점 및 후속 연구 방향

### 저자가 제시한 시사점
CQR은 "conformal prediction과 quantile regression의 장점을 결합한 새로운 방법"이며, exchangeability라는 온화한 가정 하에서 유한 표본 커버리지를 보장하면서 이분산성에 적응한다(p.11, Conclusion). 저자들은 **conformal predictive distributions**(단순 구간이 아닌 전체 예측 확률분포 추정)로의 확장 가능성을 언급하며, 독립적으로 작성된 관련 논문 [Vovk et al., 2019, arXiv:1902.06579]과의 흥미로운 연결점을 제시한다(p.11).

### 8-1. 모델의 일반화 성능 향상 가능성

CQR의 일반화 성능 향상 가능성은 다음 세 가지 측면에서 두드러진다:

1. **알고리즘 불가지론적(algorithm-agnostic) 특성**: CQR은 "임의의 quantile regression 알고리즘을 wrapping할 수 있다"(p.2)는 구조적 유연성 때문에, 향후 개발될 더 강력한 quantile 추정 모델(예: transformer 기반, gradient boosting 변형)에 즉시 적용 가능하며 이론적 커버리지 보장은 알고리즘의 정확도와 **독립적**으로 유지된다(Theorem 1은 임의의 $\hat{q}\_{\alpha_{lo}}, \hat{q}\_{\alpha_{hi}}$에 대해 성립).

2. **분포 불가지론적 보장**: exchangeability만 요구하므로(정규성, 등분산성 등 불필요), 다양한 실제 데이터 분포(long-tail, multi-modal 등)에 대한 일반화가 이론적으로 보장됨. 이는 특정 분포 가정에 의존하는 전통적 방법 대비 우월한 일반화 잠재력을 시사.

3. **한계**: 그러나 이 논문에서 다루지 않은 **covariate shift**(train/test 분포 상이) 상황에서는 exchangeability 가정이 깨지므로 이론적 보장이 실패한다. 실제 배포 환경에서의 일반화를 위해서는 [50] Barber et al.(2019)과 같은 covariate shift 대응 conformal 방법과의 결합이 필요하다.

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

본 논문(2019)은 CQR이라는 개념을 최초로 제시했으며, 이후 다음과 같은 방향으로 발전했다고 알려져 있다(단, 아래 내용은 본 문서에 포함되지 않은 외부 지식으로, **확실성이 낮은 부분**은 명시함):

- **CQR-r (Randomized/Improved CQR)**: 원 논문 저자들을 포함한 후속 연구에서 conformity score 정의를 개선하여 구간 길이를 추가로 단축하려는 시도가 있었던 것으로 알려짐. (※ 정확한 논문 제목과 세부 내용은 본 문서에서 확인 불가하여 확답 어려움)

- **Conformal prediction의 분류(classification) 확장**: APS(Adaptive Prediction Sets), RAPS 등 classification 영역에서 유사한 아이디어(적응적 score 기반 conformalization)가 확장 적용된 연구들이 존재하는 것으로 알려져 있음. (※ 구체적 인용은 본 문서 범위 밖)

- **Conditional coverage 강화 연구**: CQR은 marginal coverage만 보장하며 conditional coverage(특정 subgroup에서의 커버리지)는 보장하지 않는다는 한계가 있어, 이를 개선하려는 후속 연구들(예: Mondrian conformal prediction, group-conditional 방법)이 존재하는 것으로 추정되나 본 문서에서는 확인 불가.

**주의**: 위 2020년 이후 연구에 대한 서술은 본 논문 문서 자체에 포함된 정보가 아니므로, 정확한 논문명·저자·수치는 검증되지 않았음을 명시한다. 확실한 근거가 있는 정보만 제공하는 원칙에 따라, 구체적인 후속 논문의 존재나 세부 결과를 단정적으로 서술하지 않았다.

### 연구 시 고려할 점 (제언)
1. Conditional coverage(그룹별/구간별 공정성)를 명시적으로 다루는 확장 연구가 필요함.
2. Calibration set 크기가 작을 때(고차원 데이터 등) tie 문제로 인한 이론적 보장 붕괴 가능성에 대한 실증적 연구가 요구됨(Figure 6 사례 참고).
3. Distribution shift/non-exchangeable 환경에서의 CQR 강건성 검증이 실무 적용을 위해 필수적임.

---

**참고 문헌 (본 분석에 사용된 자료)**:
- Romano, Y., Patterson, E., & Candès, E. J. (2019). *Conformalized Quantile Regression*. arXiv:1905.03222v1 [stat.ME].

**주의사항**: 본 답변은 제공된 PDF 문서(1905.03222v1.pdf)의 내용에 근거하여 작성되었으며, 8-2절의 "2020년 이후 최신 연구"에 대한 서술 중 일부는 문서에 포함되지 않은 일반적 배경지식으로 정확도가 100% 확신되지 않아 별도로 표시하였습니다.
