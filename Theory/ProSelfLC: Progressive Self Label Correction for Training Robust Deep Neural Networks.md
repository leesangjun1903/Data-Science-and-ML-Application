# ProSelfLC: Progressive Self Label Correction for Training Robust Deep Neural Networks

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

ProSelfLC(Progressive Self Label Correction)는 딥러닝 학습 중 **"학습자를 얼마나 신뢰할 것인가"** 라는 핵심 문제를 해결하기 위해 제안된 방법입니다. 두 가지 근본적 질문에 답합니다:

1. **Self Label Correction에서 학습자의 신뢰 정도를 어떻게 자동으로 결정할 것인가?**
2. **저엔트로피(high confidence) 상태를 패널티로 줄 것인가, 보상으로 장려할 것인가?**

### 주요 기여

| 기여 항목 | 내용 |
|----------|------|
| 이론적 통합 분석 | CCE, LS, CP, LC를 엔트로피 및 KL 발산 관점에서 통합 분석 |
| 최초의 적응적 Self LC | 학습 시간과 예측 엔트로피를 기반으로 신뢰도를 점진적으로 조정 |
| 엔트로피 최소화 옹호 | confidence penalty 관행에 대한 반론 및 이론적 방어 |
| 실험적 검증 | 클린/노이즈 환경 모두에서 우수성 입증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 기존 방법들의 한계

**출력 정규화(Output Regularisation, OR):**
- Label Smoothing(LS): 과신뢰 예측을 단순히 완화하지만, 학습자의 지식을 활용하지 못함
- Confidence Penalty(CP): 낮은 엔트로피를 패널티로 적용하여 의미 있는 학습 방향과 상충

**기존 Self LC 방법들의 고정된 $\epsilon$:**
- Bootstrapping: 학습 전 과정에서 $\epsilon$ 고정
- Joint Optimisation: 단계별 학습으로 인한 사람의 개입 필요 ($\epsilon = 1$ 고정)
- Tf-KD $_{self}$: 2단계 학습으로 시간 소모적

핵심 문제: **학습이 진행됨에 따라 모델의 지식이 성장하는데, 이를 반영한 동적 신뢰도 조정이 없었음**

---

### 2.2 수학적 기반 및 제안 방법

#### 기본 기호 정의

- 학습 데이터: $\mathbf{X} = \{(\mathbf{x}\_i, y_i)\}_{i=1}^{N}$
- Softmax 출력(예측 확률):

$$\mathbf{p}(j|\mathbf{x}) = \exp(z_j) \Big/ \sum_{m=1}^{C} \exp(z_m) \tag{1}$$

- 원-핫 레이블: $\mathbf{q}(j|\mathbf{x}) = 1$ if $j = y$, else $0$

#### 기존 방법들의 수식 정리

**Standard CCE:**

$$L_{\text{CCE}}(\mathbf{q}, \mathbf{p}) = H(\mathbf{q}, \mathbf{p}) = \mathbb{E}_{\mathbf{q}}(-\log \mathbf{p}) = \text{KL}(\mathbf{q} \| \mathbf{p}) \tag{2, 8}$$

**Label Smoothing (LS):**

$$\tilde{\mathbf{q}}_{\text{LS}} = (1-\epsilon)\mathbf{q} + \epsilon\mathbf{u}, \quad \mathbf{u}_j = \frac{1}{C}$$

$$L_{\text{CCE+LS}}(\mathbf{q}, \mathbf{p}; \epsilon) = (1-\epsilon)H(\mathbf{q}, \mathbf{p}) + \epsilon H(\mathbf{u}, \mathbf{p})$$

$$= (1-\epsilon)\text{KL}(\mathbf{q}\|\mathbf{p}) + \epsilon\text{KL}(\mathbf{u}\|\mathbf{p}) + \epsilon \cdot \text{const} \tag{3, 9}$$

**Confidence Penalty (CP):**

$$L_{\text{CCE+CP}}(\mathbf{q}, \mathbf{p}; \epsilon) = (1-\epsilon)H(\mathbf{q}, \mathbf{p}) - \epsilon H(\mathbf{p}, \mathbf{p})$$

$$= (1-\epsilon)\text{KL}(\mathbf{q}\|\mathbf{p}) + \epsilon\text{KL}(\mathbf{p}\|\mathbf{u}) - \epsilon \cdot \text{const} \tag{4, 10}$$

**Label Correction (LC):**

$$\tilde{\mathbf{q}}_{\text{LC}} = (1-\epsilon)\mathbf{q} + \epsilon\mathbf{p}$$

$$L_{\text{CCE+LC}}(\mathbf{q}, \mathbf{p}; \epsilon) = (1-\epsilon)H(\mathbf{q}, \mathbf{p}) + \epsilon H(\mathbf{p}, \mathbf{p})$$

$$= (1-\epsilon)\text{KL}(\mathbf{q}\|\mathbf{p}) - \epsilon\text{KL}(\mathbf{p}\|\mathbf{u}) + \epsilon \cdot \text{const} \tag{5, 11}$$

> **핵심 통찰:** LS와 CP는 $\mathbf{p}$를 균등분포 $\mathbf{u}$ 방향으로 당기는 항 $+\text{KL}(\cdot\|\mathbf{u})$를 포함하지만, LC는 $-\text{KL}(\mathbf{p}\|\mathbf{u})$ 항으로 $\mathbf{p}$를 $\mathbf{u}$에서 멀어지게 함 → **엔트로피 최소화 장려**

#### 방법 요약 비교표

| | CCE | LS | CP | LC |
|---|---|---|---|---|
| 학습 타겟 | $\mathbf{q}$ | $(1-\epsilon)\mathbf{q}+\epsilon\mathbf{u}$ | $(1-\epsilon)\mathbf{q}-\epsilon\mathbf{p}$ | $(1-\epsilon)\mathbf{q}+\epsilon\mathbf{p}$ |
| 엔트로피 최소화 | - | 패널티 | 패널티 | 보상 |
| 유사도 구조 | 없음 | 없음 | 없음 | **있음** |
| 의미 클래스 | 주석 기반 | 주석 기반 | 주석 기반 | **주석+학습** |

---

### 2.3 ProSelfLC 핵심 수식

$$\boxed{
\begin{cases}
\text{Loss:} & L(\tilde{\mathbf{q}}_{\text{ProSelfLC}}, \mathbf{p}; \epsilon_{\text{ProSelfLC}}) = H(\tilde{\mathbf{q}}_{\text{ProSelfLC}}, \mathbf{p}) \\
\text{Label:} & \tilde{\mathbf{q}}_{\text{ProSelfLC}} = (1 - \epsilon_{\text{ProSelfLC}})\mathbf{q} + \epsilon_{\text{ProSelfLC}}\mathbf{p} \\
\epsilon_{\text{ProSelfLC}} & = g(t) \times l(\mathbf{p})
\end{cases}
} \tag{7}$$

**전역 신뢰 점수 (Global Trust Score):**

$$g(t) = h(t/\Gamma - 0.5,\ B) = \frac{1}{1 + \exp(-(t/\Gamma - 0.5) \times B)} \in (0, 1)$$

- $t$: 현재 반복 횟수(iteration counter)
- $\Gamma$: 총 반복 횟수
- $B$: 성장 속도를 조절하는 task-dependent 파라미터 (validation set으로 탐색)

**지역 신뢰 점수 (Local Trust Score):**

$$l(\mathbf{p}) = 1 - H(\mathbf{p})/H(\mathbf{u}) \in (0, 1)$$

- $H(\mathbf{p})$: 예측 분포의 엔트로피
- $H(\mathbf{u})$: 균등 분포의 엔트로피 (= $\log C$, 정규화 상수)

#### 설계 논리

| 학습 단계 | $g(t)$ | $l(\mathbf{p})$ (non-confident) | $l(\mathbf{p})$ (confident) | $\epsilon_{\text{ProSelfLC}}$ 해석 |
|---------|--------|------|------|------|
| 초기 ($t < \Gamma/2$) | $< 0.5$ | 작음 (e.g., 0.1) | 작음 (e.g., 0.1) | 매우 작음 → 인간 주석 우선 |
| 후기 ($t > \Gamma/2$) | $> 0.5$ | 중간 (e.g., 0.09) | **큼 (e.g., 0.81)** | **자신의 예측 신뢰** |

---

### 2.4 모델 구조

```
입력 x
    ↓
임베딩 네트워크 f(·): ℝᴰ → ℝᴷ
    ↓
선형 분류기 g(·): ℝᴷ → ℝᶜ (FC layer)
    ↓
로짓 벡터 z ∈ ℝᶜ
    ↓
Softmax → 예측 분포 p
    ↓
[ProSelfLC 동적 레이블 계산]
    ε_ProSelfLC = g(t) × l(p)
    q̃ = (1-ε)q + εp
    ↓
손실 계산: H(q̃, p)
    ↓
역전파 (End-to-end)
```

- **추가 모델 불필요**: 자신의 예측 $\mathbf{p}$만 활용
- **End-to-end 학습**: 단계별 학습 없이 단일 학습 과정으로 완료
- **무시할 수 있는 추가 연산 비용**: $\epsilon_{\text{ProSelfLC}}$ 계산은 경량 연산

---

### 2.5 성능 향상

#### 표준 이미지 분류 (Clean Setting)

| Dataset | CCE | LS (best) | CP (best) | Boot-soft (best) | **ProSelfLC (best)** |
|---------|-----|-----------|-----------|-----------------|---------------------|
| CIFAR-100 | 69.0% | 69.9% | 69.5% | 69.1% | **70.3%** |
| ImageNet 2012 | 75.5% | 75.3% | 75.2% | 75.8% | **76.0%** |

> LS와 CP는 ImageNet에서 $\epsilon$ 증가 시 오히려 성능 저하 발생

#### 합성 레이블 노이즈 (CIFAR-100, ResNet-44)

| 방법 | 비대칭 20% | 비대칭 30% | 비대칭 40% | 대칭 20% | 대칭 40% | 대칭 60% |
|------|-----------|-----------|-----------|---------|---------|---------|
| CCE | 66.6 | 63.4 | 59.5 | 58.0 | 50.1 | 37.9 |
| LS | 67.9 | 66.4 | 65.0 | 63.8 | 57.2 | 46.5 |
| CP | 67.7 | 66.0 | 64.4 | 64.0 | 56.8 | 44.1 |
| Boot-soft | 66.9 | 65.3 | 61.0 | 63.2 | 59.0 | 44.8 |
| **ProSelfLC** | **68.7** | **68.5** | **67.9** | **64.8** | **59.3** | **47.7** |

#### 실세계 레이블 노이즈 (Clothing 1M, ~38.46% noise)

| CCE | LS | CP | Boot-soft | **ProSelfLC** | Joint-soft | MD-DYR-SH |
|-----|----|----|-----------|--------------|-----------|-----------|
| 71.8 | 72.6 | 72.4 | 72.3 | **73.4** | 72.2 | 71.0 |

---

### 2.6 한계점

논문에서 명시적으로 언급되거나 분석 가능한 한계:

1. **하이퍼파라미터 의존성**: $B$와 $\Gamma$는 task-dependent하며 validation set 탐색 필요. 노이즈 율이 높을수록 작은 $B$가 유리하여 노이즈 율을 사전에 알아야 최적 설정 가능
2. **비완벽한 노이즈 교정**: 일부 noisy label은 높은 신뢰도로 암기(memorization)될 수 있음 (Figure 3b에서 ProSelfLC의 wrong fitting이 ~12% 수준으로 비록 가장 낮지만 0이 아님)
3. **stage-wise KD 방법과의 결합 미탐색**: 저자 스스로 "중요한 미래 연구 과제"로 언급
4. **CV 외 분야 검증 부족**: NLP, 음성 등 다른 도메인에서의 유효성 미검증
5. **대규모 레이블 노이즈(>60%)에서의 안정성**: 논문 실험 범위는 최대 60% 대칭 노이즈까지

---

## 3. 일반화 성능 향상 가능성

### 3.1 일반화 메커니즘 분석

ProSelfLC의 일반화 향상은 다음 세 가지 메커니즘에서 비롯됩니다:

#### (1) 유사도 구조(Similarity Structure) 학습

$$\tilde{\mathbf{q}}_{\text{ProSelfLC}}(j|\mathbf{x}) = (1-\epsilon)\mathbf{q}(j|\mathbf{x}) + \epsilon\mathbf{p}(j|\mathbf{x})$$

원-핫 레이블 대신 모델의 예측 분포를 반영함으로써, 클래스 간 유사성 정보를 학습 타겟에 내재화합니다. 예를 들어, "고양이" 이미지에 대해 "호랑이"가 약간의 확률을 가지는 소프트 타겟이 학습되어, 클래스 경계에 대한 유연한 일반화가 가능합니다.

#### (2) 동적 신뢰도를 통한 과적합 방지

초기 학습 단계($t < \Gamma/2$)에서:
$$\epsilon_{\text{ProSelfLC}} < 0.5, \quad \forall \mathbf{p}$$

모델이 아직 신뢰할 수 없는 초기 단계에서 자신의 예측에 과도하게 의존하지 않아, **초기 과적합(early overfitting)을 방지**합니다.

후기 학습 단계($t > \Gamma/2$)에서 신뢰도 높은 예측에 대해서만 높은 $\epsilon$을 부여:
$$\epsilon_{\text{ProSelfLC}} = g(t) \times l(\mathbf{p}) \approx 0.81 \quad \text{(when } g(t)=0.9, l(\mathbf{p})=0.9\text{)}$$

#### (3) 의미적 클래스 교정(Semantic Class Correction)

후기 단계에서 $\epsilon_{\text{ProSelfLC}} > 0.5$이고 예측이 주석과 불일치할 때:

$$\arg\max_j \mathbf{p}(j|\mathbf{x}) \neq \arg\max_j \mathbf{q}(j|\mathbf{x}) \Rightarrow \tilde{\mathbf{q}}_{\text{ProSelfLC}} \text{의 semantic class가 } \mathbf{p} \text{로 교체}$$

예시: $\mathbf{p} = [0.95, 0.01, 0.04]$, $\mathbf{q} = [0, 0, 1]$, $\epsilon = 0.8$일 때:
$$\tilde{\mathbf{q}}_{\text{ProSelfLC}} = 0.2 \times [0,0,1] + 0.8 \times [0.95, 0.01, 0.04] = [0.76, 0.008, 0.232]$$

→ 실제 잘못된 주석을 자동으로 교정, **노이즈 레이블 환경에서 일반화 대폭 향상**

#### (4) 엔트로피 최소화의 재정의(Meaningful Low-entropy Status)

Figure 3(d)~(f)에서 확인된 핵심 발견:
- LS, CP: 높은 엔트로피 유지 → 일반화 약간 향상
- **ProSelfLC: 가장 낮은 엔트로피** → **가장 높은 일반화 성능**

이는 "confident = overfitting"이라는 기존 통념을 반박하며, **의미 있는 저엔트로피 상태로의 수렴이 일반화를 향상**시킴을 증명합니다.

#### (5) 모델 캘리브레이션 개선

Appendix B의 ECE(Expected Calibration Error) 결과:

| 온도 스케일링 $T$ | ProSelfLC ECE | CCE ECE |
|----------------|--------------|---------|
| $T=1$ | 15.71% | 40.98% |
| $T=1/4$ | 4.24% | 18.27% |
| $T=1/8$ | **2.39%** | 9.94% |

→ 더 잘 캘리브레이션된 모델은 실제 배포 환경에서 더 신뢰할 수 있는 예측을 제공

### 3.2 일반화와 관련한 이론적 근거

ProSelfLC의 일반화 향상은 두 가지 잘 확립된 명제에 기반합니다:

**명제 A (Arpit et al., ICML 2017):** 딥 뉴럴 네트워크는 노이즈를 피팅하기 전에 의미 있는 패턴을 먼저 학습한다.

$$\Rightarrow g(t) \text{가 초기에 작아서 인간 주석 우선} \Rightarrow \text{의미 있는 패턴 학습 보호}$$

**명제 B (Grandvalet & Bengio, 2005/2006):** 엔트로피 최소화 원칙은 준지도 학습에서 강력하다.

$$\Rightarrow l(\mathbf{p}) = 1 - H(\mathbf{p})/H(\mathbf{u}) \text{로 확신 있는 예측 장려} \Rightarrow \text{의미 있는 저엔트로피 상태 정의}$$

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 미래 연구에 미치는 영향

#### (A) 동적 레이블 수정 패러다임의 확장

ProSelfLC는 **고정된 하이퍼파라미터 → 동적/적응적 조정**으로의 패러다임 전환을 선도했습니다. 이후 연구들은 다음 방향으로 발전할 수 있습니다:

- 학습 시간과 예측 불확실성 외에 **추가 신호**(예: 손실 값, 그래디언트 크기, 인스턴스 난이도)를 신뢰도 측정에 통합
- **메타러닝** 기반으로 신뢰도 함수 $g(t)$와 $l(\mathbf{p})$ 자체를 학습하는 방향

#### (B) 지식 증류(KD)와의 통합

논문에서 KD가 LC의 특수 형태임을 증명(Proposition 2):

$$L_{\text{KD}}(\mathbf{q}, \mathbf{p}_t, \mathbf{p}) = (1-\epsilon)H(\mathbf{q}, \mathbf{p}) + \epsilon H(\mathbf{p}_t, \mathbf{p}) \Rightarrow \tilde{\mathbf{q}}_{\text{KD}} = (1-\epsilon)\mathbf{q} + \epsilon\mathbf{p}_t$$

이는 **ProSelfLC를 teacher 학습에 적용하여 더 강력한 teacher 생성**이 가능함을 시사합니다.

#### (C) 엔트로피 최소화 원칙의 재정립

"confidence penalty = good regularisation"이라는 기존 통념에 이론적 반론을 제시함으로써, 앞으로의 정규화 기법 설계 시 **맹목적 confidence penalty 회피 및 의미 있는 저엔트로피 상태 정의**가 중요한 기준이 될 것입니다.

#### (D) 준지도/자기지도 학습과의 연계

ProSelfLC의 설계 원칙(초기에는 외부 정보 신뢰, 후기에는 자체 지식 활용)은 준지도 학습의 pseudo-labeling 전략과 본질적으로 연결되어 있어, 미래 준지도/자기지도 학습 연구에 영감을 제공합니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래 연구들은 논문 내 인용 및 공개된 관련 문헌을 바탕으로 분석합니다.

#### (A) DivideMix (Li et al., ICLR 2020) [논문 ref 25]

| 비교 항목 | ProSelfLC | DivideMix |
|---------|-----------|-----------|
| 방법론 | Self LC (단일 모델) | 준지도 학습 + co-training |
| 노이즈 처리 | 동적 신뢰도로 레이블 교정 | GMM으로 클린/노이즈 분리 후 MixMatch |
| 추가 모델 | 불필요 | 2개 네트워크 필요 |
| 계산 비용 | 낮음 | 높음 |
| 강점 | 단순성, 이론적 근거 | 매우 높은 노이즈율에서 강건 |

DivideMix는 극단적 노이즈(>80%)에서 더 강하지만, ProSelfLC는 추가 모델 없이 준수한 성능을 달성합니다.

#### (B) Tf-KD $_{self}$ (Yuan et al., CVPR 2020) [논문 ref 55]

| 비교 항목 | ProSelfLC | Tf-KD $_{self}$ |
|---------|-----------|---------------|
| 학습 방식 | End-to-end | 2단계 학습 |
| $\epsilon$ 결정 | 자동 (학습 시간 + 엔트로피) | 수동 튜닝 |
| 인간 개입 | 최소 | 필요 |
| 이론적 관점 | LC = KD 증명 포함 | KD 관점 |

저자들은 Tf-KD $_{self}$의 첫 번째 단계를 ProSelfLC로 대체하면 더 강력한 teacher가 생성될 수 있다고 제안합니다.

#### (C) 이후 관련 연구 트렌드 (2020~2023, 논문 기반 추론)

> **주의**: 이하 내용은 논문에서 직접 인용된 범위를 넘어서며, 제 학습 데이터에 기반한 연구 트렌드 분석입니다. 개별 논문의 세부 수치는 검증이 필요합니다.

**① 노이즈 레이블 학습 트렌드:**
- **Noise-robust Loss Functions**: Generalized Cross Entropy(GCE, Zhang et al., NeurIPS 2018), Symmetric Cross Entropy(SL, Wang et al., ICCV 2019)와 비교 시 ProSelfLC는 레이블 교정 관점으로 상호 보완적
- **Sample Selection + LC 결합**: ELR(Early-Learning Regularization), CORES 등의 연구가 ProSelfLC의 단계적 신뢰 원칙을 발전시킨 것으로 볼 수 있음

**② 자기 지식 활용(Self-Knowledge):**
- **Self-Distillation 계열**: 모델 자신의 이전 상태를 teacher로 활용하는 연구들이 ProSelfLC와 개념적으로 연결됨
- Mean Teacher, Temporal Ensembling 등의 consistency regularization 방법들과의 통합 가능성

**③ ProSelfLC의 차별점 (정리):**

$$\text{ProSelfLC의 핵심 기여} = \underbrace{g(t)}_{\text{시간적 신뢰}} \times \underbrace{l(\mathbf{p})}_{\text{확신도 신뢰}} = \epsilon_{\text{ProSelfLC}}$$

이 두 신호의 곱으로 신뢰도를 결정하는 방식은 기존 연구에서 찾기 어려운 독창적 기여입니다.

---

### 4.3 향후 연구 시 고려할 점

#### (1) 신뢰도 함수의 일반화

현재 $g(t)$는 시그모이드 형태로 고정되어 있습니다:
$$g(t) = \frac{1}{1 + \exp(-(t/\Gamma - 0.5) \times B)}$$

**고려사항**: 데이터셋 특성(노이즈 유형, 클래스 수)에 따라 최적 형태가 다를 수 있으므로, **메타러닝으로 $g(t)$ 함수 자체를 학습**하는 방향 탐색이 필요합니다.

#### (2) 대규모 노이즈 환경에서의 안정성

논문 실험의 한계인 60% 이상의 극단적 노이즈 환경에서:
$$\epsilon_{\text{ProSelfLC}} \text{가 너무 커질 경우 오류 전파(error propagation) 위험}$$

DivideMix류의 **clean/noisy sample 분리 전략**과 ProSelfLC를 결합하는 연구가 유망합니다.

#### (3) 도메인 확장

- **NLP 분야**: 텍스트 분류, NER 등에서 노이즈 레이블 문제는 더 심각하며, ProSelfLC의 원칙 적용 가능성 검토 필요
- **의료 AI**: 전문가 주석의 노이즈가 높은 의료 이미지 분야에서 실용적 가치가 클 것

#### (4) 장기 학습 안정성

Figure 3에서 ProSelfLC가 장기 노출에도 강건함을 보이나, **매우 긴 학습(수백 에포크)**에서의 동작 분석이 필요합니다.

#### (5) 이론적 수렴 보장

현재 논문은 경험적 증거에 강하지만, ProSelfLC의 수렴성에 대한 **엄밀한 이론적 분석**이 부족합니다. 특히:
- $\epsilon_{\text{ProSelfLC}}$의 동적 변화가 학습의 수렴성에 미치는 영향
- 최적 $\epsilon^*$ 존재 및 ProSelfLC가 이에 수렴하는지 여부

---

## 참고 자료

**기본 논문:**
- Xinshao Wang, Yang Hua, Elyor Kodirov, David A. Clifton, Neil M. Robertson. "ProSelfLC: Progressive Self Label Correction for Training Robust Deep Neural Networks." arXiv:2005.03788v6 [cs.LG], CVPR 2021. https://arxiv.org/abs/2005.03788

**논문 내 주요 인용 문헌:**
- Arpit et al. "A closer look at memorization in deep networks." ICML 2017.
- Grandvalet & Bengio. "Semi-supervised learning by entropy minimization." NeurIPS 2005.
- Grandvalet & Bengio. "Entropy regularization." Semi-supervised learning, 2006.
- Reed et al. "Training deep neural networks on noisy labels with bootstrapping." ICLR Workshop 2015.
- Tanaka et al. "Joint optimization framework for learning with noisy labels." CVPR 2018.
- Yuan et al. "Revisiting knowledge distillation via label smoothing regularization." CVPR 2020.
- Li et al. "Dividemix: Learning with noisy labels as semi-supervised learning." ICLR 2020.
- Wang et al. "Symmetric cross entropy for robust learning with noisy labels." ICCV 2019.
- Zhang & Sabuncu. "Generalized cross entropy loss for training deep neural networks with noisy labels." NeurIPS 2018.
- Hinton et al. "Distilling the knowledge in a neural network." NeurIPS Workshop 2015.
- Szegedy et al. "Rethinking the inception architecture for computer vision." CVPR 2016. (Label Smoothing)
- Pereyra et al. "Regularizing neural networks by penalizing confident output distributions." ICLR Workshop 2017.
- He et al. "Deep residual learning for image recognition." CVPR 2016.
- Guo et al. "On calibration of modern neural networks." ICML 2017.

**GitHub 코드:**
- https://github.com/XinshaoAmosWang/ProSelfLC-CVPR2021
