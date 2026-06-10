# How does Early Stopping Help Generalization against Label Noise?

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

딥 뉴럴 네트워크(DNN)가 noisy label을 심각하게 암기(memorize)하기 **이전**에 학습을 조기 종료(early stopping)하고, 해당 시점부터 **Maximal Safe Set**을 활용하여 학습을 재개함으로써 **어떠한 유형의 label noise에도 noise-free 학습**이 가능하다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **Prestopping 제안** | Early stopping + Maximal Safe Set 학습의 2단계 학습 전략 |
| **Error-prone Period 개념 정의** | false-labeled 샘플의 급격한 memorization이 발생하는 구간 규명 |
| **두 가지 실용적 Heuristic 제시** | Validation Heuristic / Noise-Rate Heuristic |
| **Prestopping+ 개발** | SELFIE의 sample refurbishment와 결합한 확장 버전 |
| **광범위한 실험 검증** | CIFAR-10/100, ANIMAL-10N, FOOD-101N에서 SOTA 대비 0.4–8.2%p 개선 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 기존 방법의 한계

기존의 **loss-based sample selection** 방법들(Co-teaching, SELFIE 등)은 small-loss 샘플을 true-labeled로 간주한다. 그러나 이 방식은 **symmetric noise**에서는 잘 작동하지만, **pair noise** 및 **real-world noise**에서는 true-labeled와 false-labeled 샘플의 손실 분포가 겹쳐 분리가 어렵다.

$$\text{문제: pair/real-world noise} \Rightarrow \mathcal{L}_{\text{true}} \approx \mathcal{L}_{\text{false}} \Rightarrow \text{separation 실패}$$

#### 핵심 관찰

논문은 다음 두 가지 **memorization effect** 특성을 실험적으로 발견하였다:

1. **Noise type에 따라 false-labeled 샘플의 memorization 속도가 다르다**: pair noise에서 false-labeled 샘플의 memorization이 더 빠르게 시작된다.
2. **Error-prone Period의 존재**: noise type과 무관하게, 학습 후반부에 false-labeled 샘플의 memorization이 급격히 증가하는 구간이 존재하며, 이 구간의 학습은 일반화 성능에 전혀 도움이 되지 않는다.

---

### 2.2 제안 방법 (수식 포함)

#### 기본 표기법

- 학습 데이터: $\tilde{\mathcal{D}} = \{x_i, \tilde{y}\_i\}_{i=1}^{N}$ (noisy label 포함)
- 진짜 label: $y_i^* \in \{1, 2, \ldots, k\}$

#### [Phase I] 표준 학습 (Early Stopping 전)

$$\theta_{t+1} = \theta_t - \alpha \nabla \left( \frac{1}{|\mathcal{B}_t|} \sum_{x \in \mathcal{B}_t} \mathcal{L}(x, \tilde{y}; \theta_t) \right) $$

여기서 $\alpha$는 learning rate, $\mathcal{B}_t$는 mini-batch이다.

#### [Memorized Sample 정의]

샘플 $x$의 최근 $q$ epoch에 대한 예측 기록을 $H_x^t(q) = \{\hat{y}\_{t_1}, \hat{y}\_{t_2}, \ldots, \hat{y}_{t_q}\}$로 정의할 때:

$$P(y | x, t; q) = \frac{\sum_{\hat{y} \in H_x^t(q)} [\hat{y} = y]}{|H_x^t(q)|} $$

샘플 $x$가 noisy label $\tilde{y}$를 가질 때, $\arg\max_y P(y|x,t;q) = \tilde{y}$이면 **memorized sample**로 정의된다.

#### [Memorization Precision & Recall]

$\mathcal{M}_t$를 time $t$에서의 memorized sample 집합이라 할 때:

```math
MP = \frac{|\{(x, \tilde{y}) \in \mathcal{M}_t : \tilde{y} = y^*\}|}{|\mathcal{M}_t|}, \quad MR = \frac{|\{(x, \tilde{y}) \in \mathcal{M}_t : \tilde{y} = y^*\}|}{|\{(x, \tilde{y}) \in \tilde{\mathcal{D}} : \tilde{y} = y^*\}|}
```


- **최적 Early Stop Point**: $MP$와 $MR$이 교차하는 시점 → precision과 recall의 최적 trade-off

#### [Early Stop Point 결정: 두 가지 Heuristic]

| Heuristic | 조건 | 판단 기준 |
|-----------|------|-----------|
| **Validation Heuristic** | clean validation set 보유 | validation error가 최소인 시점 |
| **Noise-Rate Heuristic** | noise rate $\tau$ 알고 있을 때 | training error $\leq \tau \times 100\%$ 시점 |

#### [Phase II] Maximal Safe Set 기반 학습

Early stop point $t_{stop}$ 이후, memorized 샘플로 구성된 **Maximal Safe Set** $\mathcal{S}_t$를 활용:

$$\theta_{t+1} = \theta_t - \alpha \nabla \left( \frac{1}{|\mathcal{B}_t'|} \sum_{x \in \mathcal{B}_t'} \mathcal{L}(x, \tilde{y}; \theta_t) \right), \quad \mathcal{B}_t' = \{x \mid x \in \mathcal{S}_t \cap \mathcal{B}_t\} $$

$\mathcal{S}\_{t_{stop}} = \mathcal{M}\_{t_{stop}}$에서 시작하여 매 iteration마다 refinement된다.

#### [Prestopping+: Sample Refurbishment와 결합]

SELFIE의 refurbishment 개념을 결합하여:

$$\theta_{t+1} = \theta_t - \alpha \nabla \left( \frac{1}{|\{x \mid x \in \mathcal{R}_t \cap \mathcal{S}_{t_{end}}\}|} \left( \sum_{x \in \mathcal{R}_t \cap \mathcal{S}_{t_{end}}^c} \mathcal{L}(x, y^{refurb}; \theta_t) + \sum_{x \in \mathcal{S}_{t_{end}}} \mathcal{L}(x, \tilde{y}; \theta_t) \right) \right) $$

여기서 $\mathcal{R}_t$는 refurbished sample set, $y^{refurb}$는 수정된 label이다.

---

### 2.3 모델 구조

Prestopping은 특정 아키텍처에 종속되지 않는 **학습 전략**이다.

```
[Phase I: Noisy Training]
  - 표준 SGD로 전체 noisy 데이터 학습
  - Validation / Noise-Rate Heuristic으로 t_stop 결정
  - 최적 t_stop에서 θ 저장
         ↓
[Transition: Maximal Safe Set 초기화]
  - S_{t_stop} = M_{t_stop} (memorized 샘플 집합)
         ↓
[Phase II: Noise-free Training]
  - S_t만 사용하여 학습 재개
  - 매 iteration마다 S_t 갱신 (점진적 확장 및 정제)
```

**실험에 사용된 아키텍처:**
- DenseNet (L=40, k=12)
- VGG-19
- 벤치마크: CIFAR-10, CIFAR-100, ANIMAL-10N, FOOD-101N

**주요 하이퍼파라미터:**
- History length $q = 10$ (grid search 결과)
- Learning rate: 0.1 (50%, 75% epoch에서 1/5 감소)
- Momentum: 0.9, Batch size: 128

---

### 2.4 성능 향상 및 한계

#### 성능 향상

| 데이터셋 | Noise 유형 | 개선폭 (vs. SOTA) |
|----------|-----------|-------------------|
| CIFAR-10/100 | Pair Noise (40%) | 2.2pp–18.1pp |
| CIFAR-10/100 | Symmetric Noise (40%) | 0.3pp–11.7pp |
| ANIMAL-10N (τ≈8%) | Real-world | 0.4pp–4.6pp |
| FOOD-101N (τ≈18.4%) | Real-world | 0.5pp–8.2pp |

Maximal Safe Set의 품질:
- Symmetric noise: Label Precision > **0.99**, Label Recall > **0.81** (80 epoch 이후)
- Pair noise: Label Precision > **0.94**, Label Recall > **0.84** (80 epoch 이후)

#### 한계

1. **최소한의 추가 감독 필요**: clean validation set 또는 noise rate $\tau$ 중 하나가 반드시 필요하다.
2. **Noise-Rate Heuristic의 부정확성**: true-labeled 샘플이 false-labeled 샘플보다 항상 먼저 memorize된다는 가정이 완벽히 성립하지 않아, validation heuristic 대비 성능이 낮다.
3. **대규모 클래스에서 Prestopping+의 한계**: 클래스 수가 많을수록(FOOD-101N) label refurbishment의 정확도가 떨어져 Prestopping+가 오히려 Prestopping보다 성능이 낮을 수 있다.
4. **계산 비용**: 매 iteration마다 $P(y|x,t;q)$를 기반으로 $\mathcal{S}_t$를 갱신해야 하므로 추가 연산이 필요하다.
5. **이론적 보장 부재**: 어떠한 조건에서 early stop point가 최적인지에 대한 엄밀한 이론적 분석이 제시되지 않았다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 저하의 근본 원인 분석

논문은 noisy label 환경에서 DNN의 일반화 성능 저하가 **error-prone period** 동안의 과적합에서 비롯됨을 실험적으로 규명하였다. 이 기간 동안:

$$\text{Training Loss} \downarrow \text{ (false-labeled 샘플 memorization)} \Rightarrow \text{Test Error} \uparrow$$

### 3.2 Prestopping이 일반화를 향상시키는 메커니즘

**① Error-prone Period 제거:**

```
Default 학습:     [유익한 학습 구간] → [Error-prone Period] → 일반화 성능 저하
Prestopping 학습: [유익한 학습 구간] → [Early Stop] → [Maximal Safe Set 학습] → 일반화 성능 유지
```

**② Maximal Safe Set의 점진적 정제:**

- 초기 $\mathcal{S}\_{t_{stop}}$은 precision이 높은 clean 샘플로 구성
- 이후 Phase II에서 네트워크가 발전함에 따라 더 어려운(hard-yet-informative) clean 샘플이 $\mathcal{S}_t$에 추가
- Memorization Recall이 점차 증가 → 데이터 활용도 향상

**③ 노이즈 유형 불가지론적(agnostic) 일반화:**

기존 loss-based 분리가 실패하는 pair noise / real-world noise에서도:

$$\mathcal{S}_t \text{ (memorization 기반)} \approx \{(x, \tilde{y}) : \tilde{y} = y^*\} \text{ (높은 precision)}$$

이를 통해 **noise type에 무관하게** 일관된 일반화 성능 향상을 달성한다.

**④ Prestopping+에서의 추가 일반화:**

Prestopping+는 CIFAR-100의 기존 clean 데이터셋에도 잔존하는 label noise를 발견·수정함으로써 (e.g., "Boy" → "Baby", "Mouse" → "Hamster"), **벤치마크 데이터셋의 암묵적 노이즈**까지 처리하여 일반화를 추가 향상시킨다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### ① 2단계 학습 패러다임의 확산
Prestopping이 제시한 "early stop → noise-free 재학습"의 2단계 전략은 이후 연구들에서 **표준 설계 패턴**으로 자리 잡을 가능성이 높다. 이 방향성은 이후 DivideMix(Li et al., 2020), SOP(Liu et al., 2022) 등에서도 유사하게 채택되었다.

#### ② Memorization Effect의 정량적 분석 도구 제공
Memorization Precision/Recall의 정의 (수식 3)는 노이즈 학습 연구에서 **표준 분석 지표**로 활용될 수 있다.

#### ③ 실용적 Heuristic의 제안
Clean validation set 또는 noise rate 중 하나만으로도 동작하는 구조는 실제 현장 적용 가능성을 크게 높이며, **반지도 학습(semi-supervised learning)** 프레임워크와의 결합 연구를 자극한다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 논문 내 직접 언급된 내용과 해당 논문들의 공개된 내용을 기반으로 하며, 본 논문 PDF에 직접 포함된 내용이 아닌 부분은 명시합니다.

#### DivideMix (Li et al., NeurIPS 2020)
- **방법**: GMM으로 clean/noisy 샘플을 분리한 후, MixMatch 기반 반지도학습 적용
- **Prestopping과의 차이**: DivideMix는 loss 분포를 GMM으로 모델링하여 분리하는 반면, Prestopping은 memorization history 기반으로 분리
- **장점**: 반지도학습을 통해 noisy 샘플도 unlabeled data로 활용
- **한계**: pair noise에서 GMM 분리 성능 저하 가능성 (Prestopping이 지적한 문제와 동일)

#### SOP (Liu et al., ICML 2022)
- **방법**: noisy label을 latent variable로 모델링하여 공동 최적화
- **Prestopping과의 차이**: SOP는 label correction을 통한 접근, Prestopping은 sample selection 기반
- **이론적 강점**: SOP는 수렴에 대한 이론적 보장 제공

#### ELR (Early-Learning Regularization, Liu et al., NeurIPS 2020)
- **방법**: early learning 단계의 예측을 정규화 항으로 활용하여 noisy label memorization 억제
- **Prestopping과의 관련성**: early learning 단계의 유용성을 공유하나, 학습을 명시적으로 중단하지 않는다는 점에서 차별화
- **장점**: 단일 네트워크로 동작하여 계산 효율적

| 방법 | Noise 유형 범용성 | 계산 비용 | 추가 감독 필요 | 이론 보장 |
|------|-----------------|-----------|---------------|-----------|
| **Prestopping** | ✅ 높음 | 중간 | 낮음 (val set 또는 τ) | ❌ |
| **DivideMix** | 중간 | 높음 (2 networks) | 없음 | ❌ |
| **SOP** | ✅ 높음 | 중간 | 없음 | ✅ |
| **ELR** | 중간 | 낮음 | 없음 | 부분적 |

---

### 4.3 앞으로의 연구 시 고려할 점

#### ① Early Stop Point의 자동화·이론화
현재 validation heuristic과 noise-rate heuristic 모두 부정확한 근사치이다. 추후 연구에서는:
- **이론적으로 최적 stop point를 보장하는 기준** 개발 필요
- PAC-Bayes bound나 stability 분석을 활용한 이론적 근거 마련

#### ② Clean Validation Set 의존성 탈피
- $\tau$를 사전에 알지 못하거나 clean validation set이 없는 **완전 비감독 설정**에서의 동작 보장이 필요
- 노이즈 비율 추정(noise rate estimation)과 결합한 **자기지도 학습(self-supervised)** 방향 탐색

#### ③ 대규모 데이터·모델로의 확장
- **Foundation Model (LLM, ViT 등)** 환경에서 memorization dynamics가 다르게 나타날 수 있음
- Transformer 기반 모델에서 memorization history의 의미가 달라질 수 있어 재정의 필요

#### ④ Semi-supervised Learning과의 결합 심화
- Phase II에서 $\mathcal{S}_t$에 포함되지 않는 샘플을 **unlabeled data로 활용**하는 방향 (DivideMix의 접근과 융합)
- 이를 통해 데이터 활용 효율성 대폭 향상 가능

#### ⑤ 다양한 태스크로의 적용
- 현재는 이미지 분류에 집중되어 있으나, **자연어 처리, 음성 인식, 그래프 학습** 등 다양한 도메인에서의 memorization dynamics 분석 필요

#### ⑥ Instance-dependent Noise 대응
- 현재 논문은 class-level noise transition matrix를 가정하지만, 실제 환경에서는 **샘플별로 다른 noise pattern**이 존재 (instance-dependent noise)
- 이 경우 memorization-based separation의 유효성 재검증 필요

---

## 참고 자료

1. **본 논문 (주요 참고 문헌):**
   - Hwanjun Song, Minseok Kim, Dongmin Park, Jae-Gil Lee. "How does Early Stopping Help Generalization against Label Noise?" *ICML 2020 Workshop on Uncertainty and Robustness in Deep Learning.* arXiv:1911.08059v3, 2020.

2. **논문 내 직접 인용 문헌:**
   - Arpit, D. et al. "A closer look at memorization in deep networks." *ICML*, 2017.
   - Han, B. et al. "Co-teaching: Robust training of deep neural networks with extremely noisy labels." *NeurIPS*, 2018.
   - Song, H. et al. "SELFIE: Refurbishing unclean samples for robust deep learning." *ICML*, 2019.
   - Li, M., Soltanolkotabi, M., and Oymak, S. "Gradient descent with early stopping is provably robust to label noise for overparameterized neural networks." arXiv:1903.11680, 2019.
   - Zhang, C. et al. "Understanding deep learning requires rethinking generalization." *ICLR*, 2017.
   - Yu, X. et al. "How does disagreement help generalization against label corruption?" *ICML*, 2019.
   - Song, H. et al. "Learning from noisy labels with deep neural networks: A survey." arXiv:2007.08199, 2020.

3. **비교 분석 참고 문헌 (2020년 이후):**
   - Li, J. et al. "DivideMix: Learning with noisy labels as semi-supervised learning." *ICLR*, 2020.
   - Liu, S. et al. "Early-Learning Regularization Prevents Memorization of Noisy Labels." *NeurIPS*, 2020.
   - Liu, B. et al. "Self-Supervised Label Correction with Meta-Prototype for Noisy Labels." *ICML*, 2022. *(SOP 관련)*
