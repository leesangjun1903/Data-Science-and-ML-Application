# TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation

---

## 1. Executive Summary (10문장 이내)

TIMING은 시계열 데이터의 XAI(설명 가능한 AI) 분야에서 기존 방법들이 지닌 **방향성 무시 문제**와 **평가 지표의 편향 문제**를 동시에 해결하고자 제안된 논문이다.  
기존 시계열 XAI 방법들은 attribution의 크기(magnitude)만을 활용하고, 예측에 긍정적·부정적으로 영향을 미치는 방향(direction)을 무시하는 unsigned attribution 방식에 의존해 왔다.  
현행 평가 지표들은 상위 K개의 중요 포인트를 동시에 제거하는 방식을 사용해, 양(+)과 음(-) 방향의 attribution이 서로 상쇄(cancel out)되어 편향된 평가 결과를 유발한다.  
이를 해결하기 위해 저자들은 **CPD(Cumulative Prediction Difference)**와 **CPP(Cumulative Prediction Preservation)**라는 새로운 평가 지표를 제안하여, 누적 방식으로 attribution의 품질을 공정하게 평가한다.  
새로운 지표 하에서 기존의 Integrated Gradients(IG)가 최신 방법들보다 우수함이 밝혀진다.  
그러나 vanilla IG를 시계열에 직접 적용하면, 시간적 의존성을 무시하고 OOD(Out-of-Distribution) 샘플을 생성하는 문제가 발생한다.  
이를 극복하기 위해 **TIMING**은 세그먼트 기반 랜덤 마스킹을 통해 시간적 의존성을 반영한 stochastic baseline을 IG에 통합한다.  
TIMING은 Sensitivity와 Implementation Invariance 등의 이론적 속성을 유지하면서도, 다양한 실세계 및 합성 시계열 벤치마크에서 기존 모든 XAI 기준선을 능가하는 성능을 보인다.  
GRU, CNN, Transformer 등 다양한 블랙박스 모델에서도 일관된 성능 향상이 검증된다.  
본 논문은 시계열 도메인에서 신뢰 가능하고 해석 가능한 XAI를 위한 평가 프레임워크와 방법론을 함께 제시하는 포괄적인 연구이다.

### 1-1. 연구의 목적과 필요성

시계열 데이터는 의료(MIMIC-III), 에너지, 교통, 인프라 등 **안전-결정적(safety-critical) 도메인**에서 광범위하게 활용되므로, 딥러닝 모델의 예측 결정 과정에 대한 투명성 확보가 필수적이다 (p.1, Introduction). 기존 XAI 방법들은 두 가지 핵심 문제를 갖는다.

**문제 1: Unsigned Attribution의 한계** — 기존 시계열 XAI 방법들(FIT, WinIT, Dynamask, Extrmask 등)은 attribution의 **크기(magnitude)만** 추정하고, 예측을 강화하는지 억제하는지의 **방향(direction)**을 무시한다 (p.1, Abstract).

**문제 2: 평가 지표의 편향** — 기존 평가 전략은 상위 K개 포인트를 **동시에** 제거하는 방식으로, 양방향 attribution을 가진 올바른 방법이 방향이 정렬된(sign-aligned) 불완전한 방법에 의해 과소평가되는 **cancel out 편향**을 유발한다 (p.2, Figure 1).

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 (위치) | 강도 |
|---|-----------|-------------|------|
| 1 | 기존 XAI 평가 지표는 cancel out 편향을 내재한다 | Figure 1, Table 1 (p.2-3) | 예시 + 실험적 |
| 2 | CPD/CPP는 signed·unsigned attribution을 공정하게 평가한다 | Section 3.2 (p.3) | 이론적 정의 |
| 3 | 새로운 지표 하에서 IG가 최신 방법들보다 우수하다 | Table 1, Table 2 (p.3, 6) | 실험적 |
| 4 | Vanilla IG의 zero-baseline 경로는 OOD 샘플을 생성한다 | Section 4.2 (p.4) | 이론적 논거 |
| 5 | Vanilla IG는 시간적 관계를 고려하지 못한다 | Section 4.2 (p.4) | 이론적 논거 |
| 6 | TIMING의 세그먼트 기반 마스킹은 랜덤 마스킹(RandIG)보다 우수하다 | Table 5 (p.8) | Ablation 실험 |
| 7 | TIMING은 Sensitivity, Implementation Invariance를 만족하지만 Completeness는 불만족한다 | Propositions 4.2–4.4 (p.5) | 수학적 증명 |
| 8 | TIMING은 계산 효율적이다 (~0.04 sec/sample) | Figure 4 (p.9) | 실험적 |
| 9 | TIMING은 GRU, CNN, Transformer 모두에서 최고 성능이다 | Table 8 (p.17) | 실험적 |
| 10 | TIMING은 임상적으로 해석 가능한 attribution을 제공한다 | Figures 6-10 (p.18-19) | 정성적 |

### 2-1. 해결 문제, 제안 방법, 모델 구조, 성능, 한계

#### 해결하고자 하는 문제

**문제 A: Cancel Out 편향**

기존 평가 방식:

$$\Delta\hat{y} = \left| F_{\hat{y}}(x) - F_{\hat{y}}(x_K^\uparrow) \right|$$

$x_K^\uparrow$는 상위 K개 포인트를 **동시에** 제거한 입력. 이때 양의 attribution(+)과 음의 attribution(-)이 상쇄되어 방향성 있는 attribution을 평가하지 못한다 (p.3).

**문제 B: Vanilla IG의 시계열 적용 문제**

$$\text{IG}_{t,d}(x) = x_{t,d} \times \int_{\alpha=0}^{1} \frac{\partial F_{\hat{y}}(\alpha x)}{\partial x_{t,d}} \, d\alpha$$

Zero baseline에서 $\alpha x$로의 경로상 모든 포인트가 단순 스케일링되어:
- 시간적 패턴이 소실될 때의 효과를 관찰 불가
- 중간 경로가 훈련 분포를 벗어난 OOD 영역에 위치

---

#### 제안 방법 (수식 포함)

**1단계: CPD (Cumulative Prediction Difference)**

$$\text{CPD}(x) = \sum_{k=0}^{K-1} \left\| F(x_k^\uparrow) - F(x_{k+1}^\uparrow) \right\|_1$$

높은 absolute attribution 순으로 **순차적으로** 제거하며 누적 차이를 측정. 높을수록 좋음 (p.3).

**2단계: CPP (Cumulative Prediction Preservation)**

$$\text{CPP}(x) = \sum_{k=0}^{K-1} \left\| F(x_k^\downarrow) - F(x_{k+1}^\downarrow) \right\|_1$$

낮은 absolute attribution 순으로 **순차적으로** 제거. 낮을수록 좋음 (p.3).

**3단계: MaskingIG**

바이너리 마스크 $M \in \{0,1\}^{T \times D}$를 도입하여 수정된 baseline $(1-M) \odot x$ 사용:

$$\text{MaskingIG}_{t,d}(x, M) = x_{t,d} M_{t,d} \times \int_{\alpha=0}^{1} \frac{\partial F_{\hat{y}}(\alpha(M \odot x) + (1-M) \odot x)}{\partial x_{t,d}} \, d\alpha$$

IG 경로상 중간점: $x' + \alpha(x - x') = \alpha(M \odot x) + (1-M) \odot x$ (p.4)

**4단계: RandIG**

독립 Bernoulli(p)에서 샘플링된 랜덤 마스크 하에서 기댓값 계산:

$$\text{RandIG}_{t,d}(x; p) = \mathbb{E}_{M_p}\left[\text{MaskingIG}_{t,d}(x, M_p) \mid (M_p)_{t,d} = 1\right]$$

$(M_p)_{i,j} \sim \text{Bernoulli}(p)$, 독립 샘플링 (p.5)

**5단계: TIMING (최종)**

세그먼트 기반 마스크 생성기 $G(n, s_{\min}, s_{\max})$를 사용:

$$\text{TIMING}_{t,d}(x; n, s_{\min}, s_{\max}) = \mathbb{E}_{M \sim G(n, s_{\min}, s_{\max})}\left[\text{MaskingIG}_{t,d}(x, M) \mid M_{t,d} = 1\right]$$

$G(n, s_{\min}, s_{\max})$: 길이 $[s_{\min}, s_{\max}]$ 범위의 $n$개 세그먼트를 선택하는 마스크 생성기 (p.5)

**Proposition 4.1 (Effectiveness)** — Fubini-Tonelli 정리에 의해 적분과 기댓값의 교환이 가능:

$$\mathbb{E}_M\left[\text{MaskingIG}_{t,d}(x,M) \mid M_{t,d}=1\right] = x_{t,d} \int_{\alpha=0}^{1} \mathbb{E}_M\left[\frac{\partial F_{\hat{y}}(z(\alpha;M))}{\partial x_{t,d}} \Bigg| M_{t,d}=1\right] d\alpha$$

따라서 여러 baseline에 걸쳐 IG를 반복 계산할 필요 없이, **경로 내에서 단일 랜덤 마스크**로 근사 가능 (p.5, Appendix B.1).

#### 모델 구조

```
입력 시계열 x ∈ R^{T×D}
    ↓
세그먼트 기반 랜덤 마스킹 (n개 세그먼트, 길이 [s_min, s_max])
    ↓
MaskingIG 계산 (α: 0→1, nsamples 스텝)
    ↓
Attribution 집계 (TotalGrad ⊙ (x - Baselines) / (nsamples - unmaskcount))
    ↓
Attribution 스펙트럼 A(F,x) ∈ R^{T×D} (부호 포함)
```

주 모델: 단층 GRU (hidden 200) / 부가 실험: CNN, Transformer (Appendix E, Table 8, p.17)

#### 성능 향상

| 데이터셋 | 기준 (IG, Zero CPD) | TIMING (Zero CPD) | 개선율 |
|----------|--------------------|--------------------|--------|
| MIMIC-III | 0.549 | 0.597 | +8.7% |
| PAM | 0.573 | 0.602 | +5.1% |
| Boiler | 0.752 | 1.578 | **+109.8%** |
| Epilepsy | 0.054 | 0.060 | +11.1% |
| Wafer | 0.500 | 0.674 | +34.8% |
| Freezer | 0.405 | 0.409 | +1.0% |

(Table 3, p.7 기준, K=10%)

#### 한계

1. **Completeness 불만족** (Proposition 4.4, p.5): attribution 합이 모델 출력 차이와 일치하지 않음
2. **합성 데이터에서 saliency map 추정 성능 열위**: Switch-Feature에서 ContraLSP, Extrmask에 비해 AUP/AUR 낮음 (Table 4, p.8)
3. **하이퍼파라미터 존재**: $n, s_{\min}, s_{\max}$ 설정 필요 (민감도는 낮음)
4. **단일 샘플 근사**: 계산 효율을 위해 inter-expectation을 단일 샘플로 근사 (p.5)

---

## 3. 각 주장에 페이지 및 Figure/Table 번호 표시

| 주장 | 위치 |
|------|------|
| Cancel out 편향 예시 | p.2-3, Figure 1 (a)-(d) |
| IG Signed가 기존 지표에서 SOTA 달성 | p.3, Table 1 |
| CPD, CPP 정의 | p.3, Section 3.2 |
| Vanilla IG의 OOD/시간적 문제 | p.4, Section 4.2 |
| MaskingIG 정의 | p.4-5 |
| RandIG 정의 | p.5 |
| TIMING 정의 | p.5 |
| Proposition 4.1 (Effectiveness) | p.5, Appendix B.1 (p.13-14) |
| Proposition 4.2 (Sensitivity) | p.5, Appendix B.2 (p.14) |
| Proposition 4.3 (Implementation Invariance) | p.5, Appendix B.3 (p.14) |
| Proposition 4.4 (Incompleteness) | p.5 |
| MIMIC-III 메인 결과 | p.6-7, Table 2 |
| CPP 비교 | p.7, Figure 3; p.14, Figure 5 |
| 다중 데이터셋 결과 | p.7, Table 3 |
| 합성 데이터 결과 | p.8, Table 4 |
| Ablation (RandIG vs TIMING) | p.8, Table 5 |
| 하이퍼파라미터 민감도 | p.8, Table 6 |
| 계산 효율성 | p.9, Figure 4 |
| CNN/Transformer 일반화 | p.17, Table 8 |
| 정성적 분석 | p.18-19, Figures 6-10 |

---

## 4. 저자 직접 보고 결과 vs. 해석자 해석 분리

### 4-1. 저자가 직접 보고한 결과

**연구 주제:**
- 시계열 XAI에서 directional attribution의 중요성 및 평가 지표 개선

**방법:**
- CPD, CPP (새로운 평가 지표)
- TIMING = MaskingIG의 세그먼트 기반 기댓값 (수식 위 참조)

**저자 직접 보고 결과 (수치):**
- TIMING CPD(K=50): 0.366±0.021 (MIMIC-III, zero sub.) → 전체 1위 (Table 2)
- IG CPD(K=50): 0.342±0.021 → 기존 방법 대비 우위
- Boiler에서 TIMING: +109.8% 개선 (Table 3)
- 하이퍼파라미터 최대 편차: Avg. 0.04, Zero 0.019 (Table 6)
- 계산 시간: 0.040 sec/sample (Figure 4)
- TIMING CPP(K=~1000): 0.31±0.04 (Figure 3) → 전체 최저값

### 4-2. 해석자의 해석

> ⚠️ 이하는 논문 텍스트에 명시되지 않은 해석자의 분석임을 표시합니다.

- **[해석자]** Boiler 데이터셋에서의 109.8% 개선은 다변량(20채널) 시계열에서 세그먼트 기반 마스킹이 특히 효과적임을 시사한다. 이는 채널 간 상관관계가 강한 도메인에서 시간적 구조 보존의 중요성을 반영한다.
- **[해석자]** ContraLSP의 CPD(K=50)=0.013이지만 Acc(10%)=0.921로 낮은 것은, 이 방법이 예측 불변성(인스턴스 유지)보다 특징 제거 시 예측 변화를 최대화하도록 설계되어 있어 두 지표 간 목표 불일치가 발생했기 때문으로 보인다.
- **[해석자]** 합성 데이터에서 TIMING이 AUP/AUR에서 열위인 것은, 합성 데이터의 ground truth가 "모델이 집중하는 포인트"가 아닌 "데이터 생성 과정의 중요 포인트"를 기준으로 하기 때문에, faithfulness 지향적인 TIMING과 목표가 다를 수 있다.

---

## 5. 통계적 취약점 및 비교 불가능한 수치 표시

| 항목 | 문제 유형 | 설명 |
|------|-----------|------|
| **⚠️ Table 4 (합성 데이터) — AUP/AUR** | 가정 의존 | AUP/AUR은 "모델이 데이터 생성 과정의 정답을 학습했다"는 가정에 의존하며, 저자 스스로 "this assumption does not always hold" (p.15)라고 인정 |
| **⚠️ Table 2 — Suff/Comp 음수값** | 해석 주의 | Sufficiency 음수(예: LIME -1.875, Extrmask -8.434)는 관련 특징만 유지 시 성능이 오히려 향상됨을 의미. 모델이 spurious correlation을 학습했을 가능성 시사 |
| **⚠️ Table 3 — Boiler 109.8% 개선** | 단일 도메인 편향 | Boiler 데이터셋은 90,115 샘플로 가장 크지만, 36 타임스텝으로 가장 짧음. 결과의 극적인 개선이 이 특수한 데이터 특성에 기인할 수 있음 |
| **⚠️ Table 5 — Ablation의 미미한 차이** | 효과 크기 | RandIG(p=0.7)=0.354 vs TIMING=0.366의 차이(0.012)가 통계적으로 유의한지 검정 미제시 |
| **⚠️ Table 8 — CNN 결과의 큰 분산** | 신뢰구간 | GradSHAP CNN CPD(K=50)=0.855±0.077로 표준오차가 매우 큼. 5-fold만으로는 불안정 |
| **⚠️ 단일 아키텍처 주실험** | 일반화 제한 | 주 실험이 단층 GRU 기반이며, CNN/Transformer 결과는 Appendix에만 제시 |
| **⚠️ CPD/CPP의 K 선택 임의성** | 하이퍼파라미터 | K=50, 100을 사용하지만, 최적 K 선택 기준이 명시되지 않음 |

---

## 6. 논문이 답하지 않는 질문

1. **TIMING이 Completeness를 만족하지 않을 때의 실제 영향은?** — Proposition 4.4에서 불완전성을 인정하나, 이것이 실제 해석 품질에 어떤 영향을 미치는지 구체적 분석 없음.

2. **세그먼트 수 $n$과 최적 성능 간의 이론적 관계는?** — Table 6에서 경험적 민감도만 보고하며, $n$이 충분히 클 때의 수렴 보장이 없음.

3. **Attribution의 시간적 해석 가능성은?** — 세그먼트 경계가 attribution에 미치는 영향이 명시적으로 분석되지 않음.

4. **다변량 채널 간 상호작용(cross-channel interaction)을 어떻게 처리하는가?** — 현재 TIMING은 각 채널-시점 쌍을 독립적으로 처리하나, 채널 간 공동 기여를 설명하는 방법이 불명확.

5. **비정상(non-stationary) 시계열에서의 성능은?** — 모든 실험 데이터는 상대적으로 정제된 데이터로, 분포 변화가 심한 실세계 환경에서의 검증이 없음.

6. **음수 attribution의 임상적/실용적 해석 지침은?** — 부호가 있는 attribution을 end-user가 어떻게 해석해야 하는지에 대한 가이드라인이 없음.

7. **TIMING이 spurious correlation을 가진 모델에서도 유효한가?** — 저자들은 "모델이 spurious correlation을 학습할 수 있다"고 언급하나(p.3), TIMING이 이 경우 어떻게 동작하는지 분석 없음.

8. **최적 마스킹 패턴의 학습 가능성은?** — 현재는 랜덤 세그먼트 선택이지만, 데이터 적응형 마스킹 패턴 학습의 가능성이 논의되지 않음.

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2) — CPD의 필요성을 보여주는 핵심 예시

```
(a) 올바른 ranking + 비정렬 부호 → (c) 기존 지표에서 열위
(b) 잘못된 ranking + 정렬 부호 → (c) 기존 지표에서 우위
```

**해석:** 기존 prediction difference($\Delta\hat{y}$)는 (b)의 방법이 (a)보다 우수하다고 잘못 판단한다. 반면 CPD(d)는 누적 방식으로 측정하여 (a)가 (b)보다 실제로 우수함을 올바르게 식별한다. 이 그림은 논문 전체의 동기를 직관적으로 설명하는 가장 핵심적인 그림이다. **cancel out 문제가 방향성 있는 attribution을 가진 방법들(IG 계열)을 체계적으로 불이익**을 주어왔음을 증명한다.

---

### Figure 2 (p.4) — TIMING 프레임워크 개요

**해석:** TIMING의 전체 파이프라인을 도식화한다. 좌측에서 세그먼트 기반 랜덤 마스킹이 원본 시계열에 적용되어 여러 변형 버전이 생성된다. 각 변형에 대해 독립적으로 IG가 계산되고($\alpha=0$부터 $\alpha=1$까지 경로), 우측에서 보존된(retained) 포인트와 보간된(interpolated) 포인트가 구분된다. 최종 attribution은 여러 마스킹 조건에서의 결과를 집계하여 얻는다. **세그먼트 레벨 보존이 개별 포인트 랜덤 마스킹과 다른 핵심 이유**는, 연속적인 시간 구조를 유지하여 OOD 문제를 완화하기 때문이다.

---

### Figure 3 (p.7) — CPP 비교 (MIMIC-III)

**해석:** X축은 낮은 attribution 순위(가장 덜 중요하다고 평가된 포인트부터 제거), Y축은 누적 예측 차이. **낮을수록 좋은 지표**로, IG와 TIMING이 곡선의 하단에 위치해 "덜 중요한 포인트를 제거해도 예측이 안정적으로 유지됨"을 보여준다. 반면 GradSHAP(CPP=2.68), FIT(CPP=38.67), WinIT(CPP=35.30) 등은 높은 CPP를 보여, 이들이 "중요하지 않다"고 판단한 포인트들이 실제로는 예측에 큰 영향을 미침을 의미한다. **Extrmask(CPP=10.31)와 ContraLSP(CPP=21.11)의 낮은 CPP는 이 방법들이 음수 attribution 포인트에 0을 부여하기 때문**으로, 저자들의 분석과 일치한다.

---

### Figure 4 (p.9) — 계산 효율성 분석

**해석:** X축은 로그 스케일의 전체 테스트 샘플 처리 시간(sec), Y축은 CPD(K=50). TIMING(0.040 sec, CPD=0.366)은 최고의 CPD를 달성하면서도 DeepLIFT(0.017 sec, CPD=0.142), GradSHAP(0.034 sec, CPD=0.327)과 유사한 계산 비용을 보인다. 반면 FO(2.948 sec, CPD=0.016), LIME(3.742 sec, CPD=0.071), AFO(3.540 sec, CPD=0.120)는 TIMING보다 수십 배 느리면서 성능은 훨씬 낮다. **이 그림은 TIMING이 성능-효율 트레이드오프에서 Pareto-최적에 가까운 위치**에 있음을 명확히 보여준다.

---

### Figure 10 (p.19) — 정성적 비교 (MIMIC-III, 사망 예측)

**해석:** 동일한 True Positive 케이스(label=1, 모델 출력=0.625)에 대한 모든 방법의 attribution 시각화. **TIMING, GradSHAP, DeepLIFT, IG(signed 방법들)는 Feature Index 9(lactate, 젖산)** 에 집중적으로 높은 attribution을 부여하며, 이는 젖산 산증(lactic acidosis)이 사망과 강하게 연관된다는 임상 지식과 일치한다. 반면 Dynamask, ContraLSP, TimeX, TimeX++(unsigned 방법들)는 거의 모든 피처에 걸쳐 분산된 attribution을 보이거나, 특정 피처를 클리어하게 식별하지 못한다. **이 그림은 signed attribution이 임상적 해석 가능성 측면에서 질적으로 우수함**을 가장 설득력 있게 보여준다.

---

## 8. 결론 — 시사점, 후속 연구, 추가 제언

### 8-1. 모델의 일반화 성능 향상 가능성

**저자들이 제시한 시사점 (p.9, Section 7):**
- TIMING은 Attribution faithfulness, coherence, efficiency에서 우수한 성능을 달성
- 시계열 도메인에서 모델 개발과 실용적 XAI 사이의 간극을 좁히는 데 기여
- 명시적인 후속 연구 계획은 논문에 제시되어 있지 않음 *(저자들이 직접 기술하지 않았음을 명시)*

**일반화 성능 관련 현재 입장:**

Table 8 (p.17)에서 TIMING의 GRU→CNN/Transformer 일반화를 확인:

| 모델 | CPD(K=50) | 개선 (vs IG) |
|------|-----------|--------------|
| GRU | 0.366 | +7.0% |
| CNN | 1.173 | +26.8% |
| Transformer | 0.109 | +5.8% |

**CNN에서의 대폭 개선(+26.8%)**은 주목할 만하다. CNN은 지역 패턴을 학습하므로, 세그먼트 기반 마스킹이 CNN의 수용 영역(receptive field)과 더 잘 정렬되기 때문으로 [해석자 분석] 볼 수 있다.

**일반화 성능 향상을 위한 추가 방향 (해석자 제안):**

1. **적응형 세그먼트 선택**: 데이터의 자기상관(autocorrelation) 함수를 기반으로 $s_{\min}, s_{\max}$를 자동 결정하면 도메인별 일반화를 강화할 수 있다.

2. **다중 해상도 세그먼트**: 단일 스케일의 세그먼트 대신, 짧은 세그먼트부터 긴 세그먼트까지 계층적으로 마스킹하여 다양한 시간 스케일의 의존성을 포착.

3. **학습 기반 마스크 생성**: $G(n, s_{\min}, s_{\max})$를 고정된 규칙 대신, 입력 시계열의 특성에 따라 조건부로 마스크를 생성하는 작은 메타 모델로 대체.

4. **비정상 시계열 적응**: 분포 변화(distribution shift)를 감지하여 마스킹 전략을 동적으로 조정하는 메커니즘 추가.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ 이하는 논문 자체에 언급된 참고문헌 내에서 확인 가능한 정보를 기반으로 하며, 논문 발표 이후 추가 연구에 대한 정보는 2025년 6월 기준 학습 데이터 기준으로 제공합니다. 최신 논문과의 구체적 수치 비교는 직접 해당 논문을 확인하시기 바랍니다.

| 연구 | 발표 | 핵심 접근 | TIMING과의 관계 |
|------|------|-----------|----------------|
| FIT (Tonekaboni et al., 2020) | NeurIPS 2020 | KL divergence 기반 특징 중요도 | TIMING 기준선; unsigned attribution |
| Dynamask (Crabbé & Van Der Schaar, 2021) | ICML 2021 | 동적 마스크 학습 | TIMING 기준선; training-based |
| WinIT (Leung et al., 2023) | ICLR 2023 | 윈도우 기반 시간 의존성 | TIMING이 CPD에서 우위 |
| Extrmask (Enguehard, 2023) | ICML 2023 | 학습 기반 perturbation | AUP/AUR 우위, CPD 열위 |
| ContraLSP (Liu et al., 2024b) | ICLR 2024 | 대조 학습 + 희소 게이트 | CPD에서 TIMING이 압도적 우위 |
| TimeX (Queen et al., 2024) | NeurIPS 2024 | 해석 가능한 surrogate 모델 | TIMING이 CPD 우위 |
| TimeX++ (Liu et al., 2024a) | ICML 2024 | Information bottleneck | TIMING이 CPD 우위 |
| MILLET (Early et al., 2024) | ICLR 2024 | Multiple instance learning | 아키텍처 특화; 직접 비교 없음 |
| TimeMIL (Chen et al., 2024) | ICML 2024 | 시간 인식 MIL | 아키텍처 특화; 직접 비교 없음 |

**주요 트렌드 분석:**

2020년 이후 시계열 XAI 연구는 크게 두 방향으로 분화되었다:
1. **Unsigned, mask-learning 기반** (Extrmask, ContraLSP, TimeX, TimeX++): 훈련이 필요하지만 사전 정의된 ground truth에서 높은 성능
2. **Gradient 기반, post-hoc** (IG, GradSHAP, TIMING): 훈련 불필요, 높은 계산 효율, directional attribution

TIMING은 기존 gradient 기반 방법의 이론적 기반을 유지하면서 시계열 특화 개선을 추가한다. 특히 mask-learning 방법들이 CPD/CPP 기반 평가에서 일관되게 열위를 보이는 것은, **이들의 optimization 목표와 faithfulness 목표가 불일치**함을 시사한다.

---

### 앞으로의 연구에 미치는 영향 및 고려사항

**영향:**

1. **평가 지표 패러다임 전환**: CPD/CPP는 향후 시계열 XAI 연구의 표준 지표로 자리잡을 가능성이 높다. 특히 signed attribution 방법들에게 공정한 평가 환경을 제공한다.

2. **Directional attribution의 재조명**: 방향성 있는 attribution이 실용적으로 더 가치 있음을 실증하여, 향후 연구가 unsigned attribution을 맹목적으로 따르는 경향을 수정할 수 있다.

3. **IG의 부활**: "vanilla IG가 최신 방법들보다 우수하다"는 발견은, 복잡한 학습 기반 방법 개발보다 기존 방법의 올바른 평가가 중요함을 상기시킨다.

**연구 시 고려사항:**

1. **평가 지표 선택의 중요성**: CPD/CPP와 기존 지표(Acc, Suff, Comp)의 결과가 상충될 수 있으므로, 연구 목적에 맞는 지표를 명확히 선택해야 한다.

2. **Completeness 위반의 실용적 함의**: TIMING이 completeness를 만족하지 않으므로, attribution 값의 합을 "설명량"으로 해석하는 것은 부적절하다.

3. **도메인별 세그먼트 설계**: $s_{\min}, s_{\max}$의 최적값은 도메인의 시간 스케일(의료: 시간 단위 vs. 금융: 틱 단위)에 따라 달라지므로, 도메인 지식 기반의 하이퍼파라미터 설정이 필요하다.

4. **멀티모달 확장 가능성**: CPD/CPP는 "도메인을 넘어 자연스럽게 확장된다"고 저자들이 명시하므로(p.3), 이미지-텍스트-시계열 멀티모달 XAI 평가에도 적용 가능성이 있다.

5. **Foundation Model 시대의 XAI**: GPT 계열이나 시계열 foundation model(예: TimesFM, MOIRAI)의 등장으로, post-hoc XAI 방법의 적용 가능성과 한계를 재검토해야 한다. TIMING의 model-agnostic 특성은 이 맥락에서 유리하지만, 수억 개의 파라미터를 가진 모델에서의 gradient 계산 비용은 재평가가 필요하다.

---

## 참고자료

**주요 참고 논문 (논문 내 인용):**
- Jang, H., Kim, C., Yang, E. "TIMING: Temporality-Aware Integrated Gradients for Time Series Explanation." *ICML 2025.* arXiv:2506.05035v1
- Sundararajan, M., Taly, A., Yan, Q. "Axiomatic Attribution for Deep Networks." *ICML 2017.*
- Tonekaboni, S. et al. "What Went Wrong and When? Instance-wise Feature Importance for Time-series Black-box Models." *NeurIPS 2020.*
- Crabbé, J., Van Der Schaar, M. "Explaining Time Series Predictions with Dynamic Masks." *ICML 2021.*
- Enguehard, J. "Learning Perturbations to Explain Time Series Predictions." *ICML 2023.*
- Liu, Z. et al. "Explaining Time Series via Contrastive and Locally Sparse Perturbations." *ICLR 2024b.*
- Liu, Z. et al. "TimeX++: Learning Time-Series Explanations with Information Bottleneck." *ICML 2024a.*
- Queen, O. et al. "Encoding Time-Series Explanations through Self-Supervised Model Behavior Consistency." *NeurIPS 2024.*
- Leung, K. K. et al. "Temporal Dependencies in Feature Importance for Time Series Prediction." *ICLR 2023.*
- Lundberg, S. M., Lee, S.-I. "A Unified Approach to Interpreting Model Predictions." *NeurIPS 2017.*
- Johnson, A. E. et al. "MIMIC-III, a freely accessible critical care database." *Scientific Data, 2016.*
