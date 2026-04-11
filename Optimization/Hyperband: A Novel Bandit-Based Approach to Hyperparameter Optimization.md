# Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Hyperband 논문(Li et al., 2018, JMLR)의 핵심 주장은 다음과 같습니다:

> **하이퍼파라미터 최적화 문제를 순수 탐색(pure-exploration) 비확률적(non-stochastic) 무한-암(infinite-armed) 밴딧 문제로 정식화하고, 적응적 자원 할당과 조기 중단(early-stopping)을 통해 랜덤 서치를 가속화하는 알고리즘 Hyperband를 제안한다.**

베이지안 최적화가 *설정 선택(configuration selection)*에 집중하는 것과 달리, Hyperband는 *설정 평가(configuration evaluation)*를 가속화하는 직교적 접근을 취합니다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **알고리즘** | SuccessiveHalving의 "n vs B/n" 문제를 해결하는 Hyperband 알고리즘 제안 |
| **이론적 기여** | 비확률적 무한-암 밴딧(NIAB) 문제 최초 정식화 및 이론 보장 |
| **실용적 성능** | Bayesian 최적화 대비 5×~30× 속도 향상 |
| **일반성** | 반복 횟수, 데이터 서브샘플링, 피처 서브샘플링 등 다양한 자원 유형 지원 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**SuccessiveHalving의 "n versus B/n" 문제:**

주어진 예산 $B$에서 설정 수 $n$을 증가시키면 설정당 평균 자원 $B/n$이 감소합니다. 이 트레이드오프에서 최적의 $n$을 사전에 알 수 없다는 것이 근본적인 문제입니다.

- **$n$이 클 때**: 많은 설정을 탐색하지만 각 설정에 적은 자원 → 품질이 빠르게 드러나는 경우 유리
- **$n$이 작을 때**: 적은 설정을 탐색하지만 각 설정에 충분한 자원 → 수렴이 느린 경우 유리

### 2.2 제안 방법 및 수식

#### SuccessiveHalving 알고리즘

SuccessiveHalving은 $n$개의 설정에 균등하게 자원을 배분하고, 성능이 낮은 절반을 반복적으로 제거합니다. 라운드 $i$에서 남은 설정 수 $n_i$와 자원 $r_i$는:

$$n_i = \lfloor n \eta^{-i} \rfloor, \quad r_i = r \eta^i$$

여기서 $\eta$는 제거 비율을 제어하는 파라미터입니다.

#### Hyperband 알고리즘 (Algorithm 1)

**입력:** $R$ (단일 설정에 할당 가능한 최대 자원), $\eta$ (기본값 $\eta = 3$)

**초기화:**
$$s_{\max} = \lfloor \log_\eta(R) \rfloor, \quad B = (s_{\max} + 1) \cdot R$$

**알고리즘:**
$$\text{for } s \in \{s_{\max}, s_{\max}-1, \ldots, 0\}:$$
$$n = \left\lceil \frac{B}{R} \cdot \frac{\eta^s}{s+1} \right\rceil, \quad r = R\eta^{-s}$$

각 브래킷 $s$ 내부의 SuccessiveHalving 라운드 $i$에서:

$$n_i = \lfloor n\eta^{-i} \rfloor, \quad r_i = r\eta^i$$

$$L = \{\text{run then return val loss}(t, r_i) : t \in T\}$$

$$T = \text{top k}(T, L, \lfloor n_i / \eta \rfloor)$$

**예시 (R=81, η=3):**

| 브래킷 $s$ | 초기 설정 수 $n_0$ | 최소 자원 $r_0$ | 최대 자원 |
|-----------|----------------|--------------|---------|
| $s=4$ | 81 | 1 | 81 |
| $s=3$ | 27 | 3 | 81 |
| $s=2$ | 9 | 9 | 81 |
| $s=1$ | 6 | 27 | 81 |
| $s=0$ | 5 | 81 | 81 |

#### 이론적 수식

**하이퍼파라미터 최적화 문제 정식화:**

$\mathcal{X}$를 유효한 하이퍼파라미터 설정의 공간, $\ell_k: \mathcal{X} \to [0,1]$을 자원 $k$ 할당 시의 손실 함수로 정의하면:

$$\ell_* = \lim_{k \to R} \ell_k, \quad \nu_* = \inf_{x \in \mathcal{X}} \ell_*(x)$$

**envelope 함수 $\gamma$:** 수렴 속도를 정의하는 단조 감소 함수

$$\sup_i |\ell_{i,j} - \ell_{i,*}| \leq \gamma(j), \quad \forall j \in \mathbb{N} $$

**CDF 가정:**

$$\mathbb{P}(\nu_i - \nu_* \leq \epsilon) = F(\nu_* + \epsilon) $$

**파라미터화 (해석 용이성을 위해):**

$$\gamma(j) \simeq \left(\frac{1}{j}\right)^{1/\alpha} $$

```math
F(x) \simeq \begin{cases} (x - \nu_*)^\beta & \text{if } x \geq \nu_* \\ 0 & \text{if } x < \nu_* \end{cases}
```

여기서 $\alpha$는 수렴 속도(클수록 느린 수렴), $\beta$는 좋은 설정의 희귀성을 나타냅니다.

**Theorem 5 (Hyperband 보장):** 위 파라미터화 하에서 총 예산 $T$ 이후 반환된 설정 $\hat{\imath}_T$에 대해:

$$\nu_{\hat{\imath}_T} - \nu_* \leq c \left( \frac{\overline{\log}(T)^3 \overline{\log}(\log(T)/\delta)}{T} \right)^{1/\max\{\alpha, \beta\}}$$

여기서 $c = \exp(O(\max\{\alpha, \beta\}))$이고 $\overline{\log}(x) = \log(x)\log\log(x)$입니다.

**비교:** 균등 할당(uniform allocation)의 경우:

$$\nu_{\hat{\imath}_T} - \nu_* \leq c \left( \frac{\log(T)\log(\log(T)/\delta)}{T} \right)^{1/(\alpha+\beta)}$$

SuccessiveHalving은 $\Delta^{-\max\{\alpha,\beta\}}$에 비례하는 예산이 필요한 반면, 균등 할당은 $\Delta^{-(\alpha+\beta)}$에 비례하는 예산이 필요하여 **SuccessiveHalving이 최대 $\Delta^{-\min\{\alpha,\beta\}}$배 효율적**입니다.

### 2.3 모델 구조

Hyperband 자체는 특정 ML 모델 구조가 아니라 **메타 알고리즘(meta-algorithm)**입니다.

```
Hyperband
├── 외부 루프: s = smax, ..., 0 (브래킷 순회)
│   └── 내부 루프: SuccessiveHalving 실행
│       ├── 라운드 i = 0, ..., s
│       │   ├── 각 설정에 r_i 자원 할당하여 학습
│       │   └── 상위 1/η 설정만 유지
│       └── 최종 1개 설정 출력
└── 전체 최솟값 반환
```

**필요한 세 가지 함수:**
1. `get_hyperparameter_configuration(n)`: $n$개의 설정 샘플링
2. `run_then_return_val_loss(t, r)`: 설정 $t$에 자원 $r$ 할당 후 검증 손실 반환
3. `top_k(configs, losses, k)`: 상위 $k$개 설정 선택

### 2.4 성능 향상

| 실험 | 비교 대상 | 속도 향상 |
|------|---------|---------|
| CIFAR-10 (딥러닝) | Bayesian 최적화 | >10× |
| MRBI (딥러닝) | 랜덤 서치 | >20× |
| 커널 분류 (CIFAR-10) | Bayesian 최적화 | >30× |
| 커널 분류 (CIFAR-10) | 랜덤 서치 | ~70× |
| 피처 서브샘플링 | Bayesian 최적화 | ~6× |

### 2.5 한계점

1. **"n vs B/n" 부분 해결**: 여전히 $(s_{\max}+1)$배의 추가 비용 발생
2. **조기 중단 효과 제한**: 설정 품질이 초기에 드러나지 않는 경우(느린 수렴) 효과 감소
3. **자원 의존적 하이퍼파라미터 문제**: 최대 트리 깊이처럼 자원에 따라 최적값이 변하는 하이퍼파라미터는 처리 어려움
4. **소규모 데이터셋**: 초기화 오버헤드가 학습 시간보다 큰 경우 효과 미미 (117 데이터셋 실험에서 확인)
5. **설정 선택의 무작위성**: 기본적으로 랜덤 샘플링에 의존하여 탐색 공간 활용 한계
6. **높은 차원 탐색 공간**: 매우 고차원에서는 충분한 설정 탐색을 위해 더 공격적인 브래킷 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화와 하이퍼파라미터의 관계

논문 도입부에서 명시적으로 강조합니다:

> *"Hyperparameters are inputs to a machine learning algorithm that govern how the algorithm's performance generalizes to new, unseen data."*

즉, Hyperband는 **일반화 성능을 직접 최적화하는 프레임워크**입니다. 구체적으로:

### 3.2 일반화 향상 메커니즘

#### (a) 더 많은 설정 탐색 → 더 좋은 일반화 설정 발견

최대 자원 $R$, 파라미터 $\eta$가 주어졌을 때, 가장 탐색적인 브래킷 $s = s_{\max}$에서 탐색하는 설정 수:

$$n_{\max} = \left\lceil \frac{B}{R} \cdot \frac{\eta^{s_{\max}}}{s_{\max}+1} \right\rceil \approx \frac{B}{R} \cdot \frac{R}{s_{\max}+1} \cdot \frac{s_{\max}+1}{1} = B$$

이는 랜덤 서치가 동일 예산으로 탐색하는 설정 수($B/R$)의 약 $R/(s_{\max}+1)$배입니다. 더 많은 설정을 탐색할수록:

$$\mathbb{P}\left(\min_{i=1,\ldots,n} \nu_i - \nu_* \geq \Delta\right) = (1 - F(\nu_* + \Delta))^n \approx e^{-nF(\nu_*+\Delta)}$$

$n$이 클수록 좋은 일반화 성능($\nu_i$가 작은)을 가진 설정을 발견할 확률이 지수적으로 증가합니다.

#### (b) 검증 손실 기반의 조기 중단

Hyperband는 **검증 손실(validation loss)**을 기준으로 설정을 평가합니다. 학습 손실이 아닌 검증 손실을 사용함으로써:

- 과적합(overfitting)된 설정이 조기에 제거됨
- 학습-검증 일반화 갭이 작은 설정이 생존 가능성 높음

#### (c) 정규화 하이퍼파라미터 최적화

논문의 실험에서 최적화한 하이퍼파라미터들(학습률, 가중치 감쇠 L2 패널티, 드롭아웃 등)은 모두 **일반화 성능에 직접적인 영향**을 미치는 파라미터들입니다. Hyperband가 이러한 파라미터들의 더 넓은 공간을 탐색할 수 있으므로 일반화 성능 향상에 기여합니다.

#### (d) 검증-테스트 갭 관찰

흥미롭게도 117 데이터셋 실험(Section 4.2.1)에서:

> *"Bayesian methods outperform Hyperband and random search in test error performance but also exhibit signs of overfitting to the validation set, as they outperform Hyperband by a larger margin on the validation error rank."*

즉, Bayesian 최적화는 검증 오류에 과적합되는 경향이 있는 반면, Hyperband는 **검증-테스트 일반화 갭이 더 작을 수 있음**을 시사합니다. 이는 Hyperband가 검증 데이터에 덜 편향된 최적화를 수행하기 때문입니다.

#### (e) CIFAR-10 최종 성능

논문에서 Hyperband가 찾은 최적 설정을 **학습+검증 전체 데이터로 재학습**했을 때:

$$\text{Test Error} = 17.0\%$$

이는 동일 아키텍처에서 인간 전문가(18%)보다 낮은 오류율로, **Hyperband가 실질적인 일반화 성능 향상을 이끌어냄**을 보여줍니다.

### 3.3 일반화 한계 요인

- **검증 셋 과적합**: 하이퍼파라미터 최적화 자체가 검증 셋에 편향될 수 있음
- **조기 중단의 오판**: 초기에 낮은 검증 손실을 보여도 최종적으로 과적합되는 설정이 생존할 수 있음
- **자원 의존성**: 데이터셋 크기에 따라 최적 정규화 강도가 다를 수 있으나, 서브샘플링 단계에서는 다른 크기로 평가됨

---

## 4. 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

#### (a) AutoML 분야에 대한 영향

Hyperband는 **효율적인 하이퍼파라미터 최적화의 표준 기준**이 되었습니다. 특히:

- Ray Tune, Optuna, HuggingFace Trainer 등 주요 ML 프레임워크에 통합
- NAS(Neural Architecture Search)에서 조기 중단 전략의 기초로 활용

#### (b) 이론적 기여의 파급효과

비확률적 무한-암 밴딧(NIAB) 문제의 최초 정식화는 이후 연구들이 다양한 변형을 분석하는 기반이 되었습니다.

#### (c) 하이브리드 방법론 촉진

논문 자체에서도 언급하듯, Klein et al. (2017b)은 Hyperband와 Bayesian 최적화를 결합하여 BOHB를 개발했습니다. 이처럼 Hyperband는 **설정 평가와 설정 선택의 결합 연구**를 촉진했습니다.

### 4.2 앞으로 연구 시 고려할 점

#### (a) 병렬화 고려

논문에서도 언급하듯 브래킷 간 독립성을 이용한 병렬화가 가능합니다. 그러나 **비동기 병렬 실행 시 자원 불균형** 문제를 주의해야 합니다.

#### (b) 지식 이전(Knowledge Transfer) 활용

동일 작업의 이전 실험 결과를 활용하여 랜덤 샘플링 대신 **메타 학습 기반의 지능적 초기화**를 Hyperband에 통합하는 연구가 필요합니다.

#### (c) 자원 의존적 하이퍼파라미터 처리

정규화 파라미터처럼 **자원 규모에 따라 최적값이 달라지는 하이퍼파라미터**는 Hyperband의 조기 중단 판단을 왜곡할 수 있습니다. 자원 크기에 따른 정규화나 정규화 파라미터 재설계를 고려해야 합니다.

#### (d) 검증 전략 설계

검증-테스트 과적합을 방지하기 위해 **Nested Cross-Validation** 또는 **Hold-out 검증 셋의 계층적 분리**가 필요합니다.

#### (e) 다중 충실도(Multi-Fidelity) 설계

단순한 자원량 외에도 **데이터 품질, 모델 복잡도, 학습 에포크 수** 등 다차원적 충실도 지표를 고려한 확장 연구가 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 BOHB (Falkner et al., 2018) — Hyperband의 직접 후계

**"BOHB: Robust and Efficient Hyperparameter Optimization at Scale"** (Falkner, Klein, Hutter, ICML 2018)

Hyperband의 랜덤 샘플링을 **TPE(Tree-structured Parzen Estimator) 기반의 베이지안 모델**로 대체합니다.

| 특성 | Hyperband | BOHB |
|------|-----------|------|
| 설정 선택 | 랜덤 샘플링 | 베이지안 모델 |
| 설정 평가 | SuccessiveHalving | SuccessiveHalving |
| 이론 보장 | O | 부분적 |
| 수렴 속도 | 빠른 초기 | 빠른 초기 + 나중에도 우수 |

### 5.2 ASHA (Li et al., 2020) — 비동기 Hyperband

**"A System for Massively Parallel Hyperparameter Tuning"** (Li et al., MLSys 2020)

**Asynchronous Successive Halving Algorithm (ASHA)**는 동기적 브래킷 구조를 **비동기 방식**으로 변환합니다.

핵심 아이디어: 느린 설정을 기다리지 않고, 완료된 설정 중 상위 $1/\eta$만 즉시 승격.

$$\text{비동기 환경에서 } s_{\max} \text{배의 추가 overhead 제거}$$

**장점:** 분산 컴퓨팅 환경에서 자원 낭비 대폭 감소, 실제 Wall-clock Time 기준 성능 우수

### 5.3 DEHB (Awad et al., 2021)

**"DEHB: Evolutionary Hyperband for Scalable, Robust and Efficient Hyperparameter Optimization"** (IJCAI 2021)

Hyperband의 브래킷 구조에 **차분 진화 알고리즘(Differential Evolution)**을 결합하여 설정 공간을 더 효율적으로 탐색합니다.

### 5.4 비교 분석 표

| 방법 | 설정 선택 | 설정 평가 | 병렬화 | 이론 보장 | 대규모 확장성 |
|------|---------|---------|-------|---------|------------|
| **Hyperband** | 랜덤 | SuccessiveHalving | 브래킷 단위 | O (NIAB) | 중간 |
| **BOHB** | 베이지안(TPE) | SuccessiveHalving | 브래킷 단위 | 부분 | 중간 |
| **ASHA** | 랜덤 | 비동기 SH | 설정 단위 | 부분 | 높음 |
| **DEHB** | 진화 알고리즘 | SH 기반 | 가능 | 제한적 | 높음 |

### 5.5 NAS와의 연계

2020년 이후 **Neural Architecture Search (NAS)** 분야에서 Hyperband의 조기 중단 원리가 광범위하게 활용됩니다. 예를 들어:

- **DARTS** 이후 one-shot NAS 방법들은 저자원 평가(low-fidelity evaluation)를 통해 아키텍처를 조기 제거하는 방식을 채택하며, 이는 Hyperband의 핵심 아이디어와 동일합니다.

---

## 참고 자료

**주요 논문 (본 분석의 주된 출처):**
- Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018). **Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization**. *Journal of Machine Learning Research*, 18(1), 1-52. (arXiv:1603.06560v4)

**2020년 이후 비교 논문:**
- Li, L., Jamieson, K., Rostamizadeh, A., Gonina, E., Ben-Tzur, J., Hardt, M., ... & Talwalkar, A. (2020). **A System for Massively Parallel Hyperparameter Tuning**. *MLSys 2020*.
- Falkner, S., Klein, A., & Hutter, F. (2018). **BOHB: Robust and Efficient Hyperparameter Optimization at Scale**. *ICML 2018*.
- Awad, N. H., Mallik, N., & Hutter, F. (2021). **DEHB: Evolutionary Hyperband for Scalable, Robust and Efficient Hyperparameter Optimization**. *IJCAI 2021*.

> **정확도 관련 주의사항:** ASHA, BOHB, DEHB 논문의 세부 수식 및 실험 수치는 제가 직접 해당 논문 원문을 확인하지 못한 내용이 포함되어 있습니다. 비교 분석 표의 내용은 Hyperband 원논문 내 관련 언급 및 일반적으로 알려진 사실에 기반하였으며, 정확한 수치 비교를 위해서는 해당 논문 원문 직접 확인을 권장합니다.
