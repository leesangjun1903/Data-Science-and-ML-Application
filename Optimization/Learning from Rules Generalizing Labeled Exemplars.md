# Learning from Rules Generalizing Labeled Exemplars

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **규칙(Rules)과 레이블된 예시(Labeled Exemplars)를 결합한 새로운 지도학습 패러다임**을 제안합니다. 핵심 아이디어는 인간이 데이터를 검토할 때 자연스럽게 수행하는 두 가지 행동—**개별 인스턴스 레이블링**과 **규칙으로의 일반화**—을 동시에 활용하는 것입니다.

> "규칙은 예시(exemplar)의 노이즈 있는 일반화(noisy generalization)로 취급될 수 있다."

### 주요 기여 (4가지)

| # | 기여 내용 |
|---|-----------|
| 1 | **Rule-Exemplar 지도 패러다임** 제안: 규칙과 예시를 쌍(pair)으로 수집하는 자연스러운 방법론 |
| 2 | **ImplyLoss(Implication Loss)** 설계: 잠재 커버리지 변수를 통한 규칙 노이즈 제거 및 소프트 함의 손실 함수 도입 |
| 3 | **5개 태스크** (질문 분류, 스팸 탐지, 시퀀스 레이블링, 레코드 분류)에서 기존 방법 대비 우수한 성능 입증 |
| 4 | 기존 노이즈 학습 프레임워크(Snorkel, Posterior Reg. 등) 대비 일관된 성능 우위 실증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**핵심 문제:** 레이블 데이터 부족 상황에서 규칙(Rules)만 사용하면 노이즈가 심하고, 레이블만 사용하면 커버리지가 제한됨.

```
기존 딜레마:
- 규칙(Rules): 효율적이나 노이즈 多 (인간의 과잉 일반화)
- 인스턴스 레이블: 정확하나 비용이 높고 커버리지 小
```

**구체적 도전:**
- 규칙이 적용되는 인스턴스 중 일부는 잘못 레이블됨 (over-generalization)
- 여러 규칙이 충돌할 수 있음 (conflicting rules)
- 기존 노이즈 내성 방법론들은 **규칙별(rule-specific) 노이즈 패턴**을 포착하지 못함

**형식적 설정:**

- $\mathcal{X}$: 인스턴스 공간, $\mathcal{Y} = \{1, \ldots, K\}$: 클래스 레이블 공간
- 레이블 집합: $L = \{(\mathbf{x}_1, \ell_1, e_1), \ldots, (\mathbf{x}_n, \ell_n, e_n)\}$
  - $e_i \in \{R_1, \ldots, R_m, \emptyset\}$: 인스턴스 $\mathbf{x}_i$의 예시 규칙
- 비레이블 집합: $U = \{\mathbf{x}_{n+1}, \ldots, \mathbf{x}_N\}$
- 각 규칙 $R_j$의 커버 집합: $H_j = \{x \in U \cup L : R_j(x) = \ell_j\}$

---

### 2.2 제안 방법 (수식 포함)

#### (A) 노이즈 모델: 잠재 커버리지 변수 (Latent Coverage Variables)

규칙 $R_j$가 인스턴스 $\mathbf{x}\_i$를 올바르게 커버하는지를 나타내는 **잠재 베르누이 변수** $r_{ji}$ 도입:

$$r_{ji} = \begin{cases} 1 & \text{if } R_j \text{가 } \mathbf{x}_i\text{에 대해 과잉 일반화되지 않음 (올바른 레이블)} \\ 0 & \text{if } R_j \text{가 } \mathbf{x}_i\text{에 대해 과잉 일반화됨 (노이즈 레이블)} \end{cases}$$

- 커버리지 분포: $P^j_\phi(r_j | \mathbf{x})$ — 파라미터 $\phi$를 가진 별도 네트워크로 모델링

#### (B) 세 가지 손실 함수

**① 분류기 로그우도 (Classifier Log-likelihood)**

$$\max_\theta \, LL(\theta) = \max_\theta \sum_{(\mathbf{x}_i, \ell_i) \in L} \log P_\theta(\ell_i | \mathbf{x}_i) \tag{1}$$

**② 커버리지 변수 로그우도 (Coverage Variable Log-likelihood)**

$$LL(\phi) = \sum_{(\mathbf{x}_i, \ell_i, e_i) \in L} \left( \log P^{e_i}_\phi(r^i_{e_i} = 1 | \mathbf{x}_i) + \sum_{j: \mathbf{x}_i \in H_j \wedge \ell_i \neq \ell_j} \log P^j_\phi(r_{ji} = 0 | \mathbf{x}_i) \right. \left. - \sum_{j: \mathbf{x}_i \in H_j \wedge \ell_i = \ell_j} \text{Generalized-XENT}(P^j_\phi(r_j | \mathbf{x}_i), r_{ji}=1) \right) \tag{2}$$

각 항의 의미:
- 첫 번째 항: 예시-규칙 쌍에서 $r_{ji}=1$ 강제
- 두 번째 항: 레이블 불일치 시 $r_{ji}=0$ 강제
- 세 번째 항: 레이블 일치 시 노이즈 내성 손실 적용

**③ 소프트 함의 손실 (Soft Implication Loss / Negative Implication Loss)**

규칙 $R_j$의 하드 제약:

$$r_{ji} = 1 \Rightarrow y_i = \ell_j \quad \forall \mathbf{x}_i \in H_j \tag{3}$$

이를 소프트 확률로 변환:

$$\log\left(1 - P^j_\phi(r_j = 1 | \mathbf{x})(1 - P_\theta(\ell_j | \mathbf{x}))\right) \tag{4}$$

이 손실의 특성:
- $P^j_\phi(r_j=1|\mathbf{x}) \approx 1$이지만 $P_\theta(\ell_j|\mathbf{x}) \approx 0$일 때 → 큰 패널티
- $P^j_\phi(r_j=1|\mathbf{x}) \approx 0$일 때 → 손실 표면이 평탄 (해당 인스턴스의 $y$ 지도학습 효과적으로 철회)

#### (C) 최종 학습 목적함수 (ImplyLoss)

$$\min_{\theta, \phi} \, -LL(\theta) - LL(\phi) - \gamma \sum_{j; \mathbf{x} \in H_j \cap U} \log\left(1 - P^j_\phi(r_j = 1 | \mathbf{x})(1 - P_\theta(\ell_j | \mathbf{x}))\right) \tag{5}$$

- $\gamma$: 규칙 기여도를 조절하는 하이퍼파라미터

#### (D) 추론 (Inference)

테스트 시 분류기와 커버리지 네트워크를 결합한 소프트 보팅:

$$s(y|\mathbf{x}) = P_\theta(y|\mathbf{x}) + \frac{\sum_{R_j \in G} \delta(\ell_j = y) P^j_\phi(1|\mathbf{x}) + \delta(\ell_j \neq y) P^j_\phi(0|\mathbf{x})}{|G|} \tag{6}$$

- $G$: $P^j_\phi(1|\mathbf{x}) > 0.5$인 규칙들의 집합

---

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────┐
│              공유 임베딩 레이어                   │
│    (ELMo for NLP / One-hot for Census 등)        │
└──────────────┬────────────────────┬──────────────┘
               │                    │
               ▼                    ▼
   ┌─────────────────┐    ┌──────────────────────┐
   │ 분류 네트워크    │    │    규칙 네트워크       │
   │ P_θ(y|x)        │    │  P^j_φ(r_j=1|x)      │
   │ Multi-layer ReLU│    │  Input: [embed; rule_id]│
   │ → Softmax       │    │  Multi-layer ReLU     │
   │ (K classes)     │    │  → Sigmoid            │
   └─────────────────┘    └──────────────────────┘
```

**세 모듈:**

| 모듈 | 역할 | 구조 |
|------|------|------|
| 임베딩 레이어 | 입력 특징 표현 | ELMo (NLP), BoW (YouTube), One-hot (Census) |
| 분류 네트워크 $P_\theta(y \mid \mathbf{x})$ | 레이블 예측 | 2×512 ReLU + Softmax |
| 규칙 네트워크 $P^j_\phi(r_j \mid \mathbf{x})$ | 커버리지 예측 | concat([embed; rule_id]) → 2×512 ReLU + Sigmoid |

---

### 2.4 성능 향상 결과

#### 주요 실험 결과 (Only-L 대비 gain)

| 방법 | Question (Acc) | MIT-R (F1) | YouTube (Acc) | SMS (F1) | Census (Acc) |
|------|:-:|:-:|:-:|:-:|:-:|
| L+Umaj | -1.4 | +0.0 | +0.8 | +3.5 | +0.9 |
| Noise-tolerant | -0.5 | +0.0 | +1.7 | +2.9 | +1.0 |
| L2R (Ren et al.) | +0.3 | -15.4 | +2.5 | +2.3 | +2.9 |
| L+Usnorkel | -0.7 | +0.0 | +2.7 | +3.5 | +1.0 |
| Posterior Reg. | -0.8 | -0.1 | -2.9 | +1.8 | -0.8 |
| **ImplyLoss (Ours)** | **+11.7** | **+0.8** | **+3.2** | **+4.2** | **+1.7** |

**주목할 점:**
- Question 데이터셋에서 기존 최고 +0.3 대비 **+11.7** 포인트 향상
- 5개 데이터셋 모두에서 **일관된 양의 이득** (다른 방법들은 일부 데이터셋에서 음수)
- 디노이징 후 규칙 precision이 모든 데이터셋에서 **91% 이상**으로 향상

#### 한계점

1. **규칙 커버리지 의존성**: SMS처럼 커버리지 40%에 불과할 경우, 규칙의 이점이 제한적
2. **규칙 설계 비용**: 여전히 인간의 규칙 작성 노력 필요 (Question: 90분, MIT-R: 45분)
3. **대규모 레이블 데이터로의 한계**: 레이블 증가 시 성능 격차 감소 (Figure 5)
4. **규칙 표현력 한계**: 규칙이 블랙박스 함수로 취급되어 규칙 간 구조적 관계 미활용
5. **하이퍼파라미터 민감성**: $\gamma$, $q$ 등 검증 데이터로 튜닝 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 가능하게 하는 핵심 메커니즘

#### (A) 규칙 디노이징을 통한 데이터 확장

규칙을 통해 **무레이블 데이터 $U$를 효과적으로 학습 데이터로 확장**합니다. 단순히 노이즈 레이블로 추가하는 것이 아니라, 커버리지 변수 $P^j_\phi$가 신뢰할 수 있는 인스턴스만 선별:

$$\text{실효 레이블 데이터} = L \cup \{\mathbf{x} \in U : P^j_\phi(r_j=1|\mathbf{x}) > \text{threshold}\}$$

Question 데이터셋 기준: 68개 레이블 → 규칙으로 4,637개 인스턴스 커버 (약 68배 확장)

#### (B) Soft Implication Loss의 일반화 기여

네거티브 함의 손실의 구조는 **과적합 방지와 일반화에 유리**합니다:

$$\log\left(1 - P^j_\phi(r_j=1|\mathbf{x})(1-P_\theta(\ell_j|\mathbf{x}))\right)$$

이 손실은:
- $P^j_\phi$가 낮은 값을 예측할 때 표면이 평탄 → 노이즈 있는 예제를 자동으로 무시
- 강한 하드 제약 대신 소프트 제약으로 과도한 편향 방지

#### (C) 사전학습 모델과의 시너지

소량의 레이블($|L|$이 매우 작을 때)에서도 ELMo와 같은 **사전학습 임베딩**을 고정 사용하여 특징 표현의 질을 유지. 이를 통해:

$$\text{Few-shot generalization} = \underbrace{\text{Pre-trained features}}_{\text{일반 지식}} + \underbrace{\text{ImplyLoss}}_{\text{태스크별 규칙 활용}}$$

#### (D) 규칙의 체계적 노이즈 포착

기존 방법들은 입력 독립적 혼동 행렬(confusion matrix)을 가정하지만, 이 논문은 **입력 의존적(instance-dependent) 노이즈 모델**을 사용:

$$P^j_\phi(r_j|\mathbf{x}) : \text{인스턴스별 규칙 적용 여부 예측}$$

이는 동일 규칙이라도 인스턴스에 따라 다른 노이즈 특성을 포착 → **도메인 내 분포 외(out-of-distribution) 예제에 더 강인**

#### (E) 실험적 증거: 일반화 성능 분석

**규칙 Precision 향상 후 일반화:**

| 데이터셋 | 원래 Rule Precision | 디노이징 후 Precision | 향상 |
|---------|:--:|:--:|:--:|
| Question | 63.8% | ~98% | +34.2%p |
| MIT-R | 80.7% | >91% | +10.3%p |
| SMS | 97.3% | >99% | +1.7%p |

**레이블 크기 증가 실험 (Question 데이터셋):**
- $|L|=68$: ImplyLoss가 Only-L 대비 +11.7 포인트 우위
- $|L|=800$: 격차가 좁혀지나 여전히 ImplyLoss가 우위
- → **레이블이 적을수록 일반화 이점이 극대화**됨

---

## 4. 미래 연구에 대한 영향 및 고려 사항

### 4.1 이 논문이 미치는 영향

#### (A) 약한 지도학습(Weak Supervision) 패러다임 혁신

기존 Snorkel(Ratner et al., 2016) 계열의 **규칙 가중치 기반 접근**에서 **규칙의 경계를 직접 학습하는 비선형 접근**으로 패러다임 전환을 이끌었습니다. 이후 연구들에서 다음과 같은 방향으로 발전이 이루어졌습니다:

```
ImplyLoss (ICLR 2020)
    │
    ├─→ WRENCH (Zhang et al., 2021): 약한 지도학습 벤치마크 정립
    ├─→ CAGE (Chatterjee et al., 2020): 그래픽 모델 기반 규칙 통합
    └─→ FlyingSquid (Fu et al., 2020): 더 효율적인 약한 지도학습
```

#### (B) 노이즈 레이블 학습과 규칙 기반 학습의 연결

**입력 의존적 노이즈 모델**을 소프트 함의 구조와 결합한 것은, 이후 다음 연구들에 영향:

- **SPEAR** (Abhijeet et al., 2021 - 같은 그룹): 프로그래밍 방식의 약한 지도학습 확장
- Pre-LLM 시대의 데이터 효율적 학습의 기초가 됨

#### (C) LLM 시대와의 연결

2020년 이후 GPT-3/4, ChatGPT와 같은 **대형 언어 모델(LLM)의 In-Context Learning**이 일종의 "규칙 없는 임플리시트 규칙 학습"으로 볼 수 있으며, ImplyLoss의 아이디어—**예시로부터 일반화된 패턴을 활용**—와 개념적으로 연결됩니다.

---

### 4.2 최신 연구 비교 분석 (2020년 이후)

#### (A) WRENCH 벤치마크 (Zhang et al., NeurIPS 2021)

- **논문**: "WRENCH: A Comprehensive Benchmark for Weak Supervision"
- **관련성**: ImplyLoss를 포함한 약한 지도학습 방법들을 22개 데이터셋에서 체계적으로 비교
- **시사점**: ImplyLoss는 특정 도메인에서 강점을 보이나, 모든 태스크에서 항상 최고는 아님
- **개선 방향**: 더 다양한 규칙 유형과 도메인에서의 평가 필요

#### (B) CAGE (Chatterjee et al., ACL 2020)

- **논문**: "Robust Data Programming with Precision-Guided Labeling Functions"
- **방법**: 레이블 함수의 정밀도를 그래픽 모델로 직접 추정
- **ImplyLoss 대비 차이**: CAGE는 규칙 정밀도의 사전 정보를 활용 가능, ImplyLoss는 이를 학습
- **한계**: 복잡한 규칙 의존성 처리가 어려움

#### (C) FlyingSquid (Fu et al., ICML 2020)

- **논문**: "Fast and Three-rious: Speeding Up Weak Supervision with Triplet Methods"
- **방법**: 트리플렛 방법을 이용한 빠른 생성 모델 학습
- **비교**: FlyingSquid는 속도 면에서 우위, ImplyLoss는 정확도 면에서 우위
- **핵심 차이**: FlyingSquid는 exemplar 정보를 활용하지 않음

#### (D) SPEAR (Abhijeet Awasthi et al., EMNLP 2021)

- **논문**: "SPEAR: Semi-supervised Data Programming in Python"
- **관련성**: ImplyLoss 저자들의 후속 연구로, 프레임워크를 확장
- **개선점**: 레이블 함수 작성의 프로그래밍 인터페이스 제공, 더 다양한 학습 알고리즘 지원

#### (E) Annotation-Efficient Learning (Zhou et al., 2022)

최근의 **프롬프트 기반 약한 지도학습** 연구들은 ImplyLoss의 아이디어를 LLM 환경으로 확장:

- GPT 계열 모델을 "규칙 생성기"로 사용
- 생성된 규칙의 노이즈를 ImplyLoss 유사 방식으로 처리

---

### 4.3 앞으로 연구 시 고려할 점

#### ① 더 강력한 임베딩과의 결합

```
현재: ELMo (1024-dim)
→ 권장: BERT/RoBERTa fine-tuning과 ImplyLoss 결합
   - 임베딩 파라미터도 함께 학습 (현재는 고정)
   - Low-rank adaptation (LoRA) 등과 결합 가능
```

#### ② LLM을 활용한 규칙 자동 생성

$$\text{LLM}(\mathbf{x}_i, \ell_i) \rightarrow R_j \quad \text{(자동 규칙 생성)}$$

- 인간의 규칙 작성 비용 제거
- LLM이 생성한 규칙의 노이즈를 ImplyLoss로 처리
- **연구 과제**: LLM 생성 규칙의 노이즈 특성과 인간 생성 규칙의 차이 분석

#### ③ 동적 규칙 커버리지 임계값

현재 추론 시 $P^j_\phi(1|\mathbf{x}) > 0.5$ 고정 임계값 사용:

$$s(y|\mathbf{x}) = P_\theta(y|\mathbf{x}) + \frac{\sum_{R_j \in G} \cdots}{|G|}$$

→ **개선**: 태스크별, 데이터셋별 적응적 임계값 학습

#### ④ 규칙 간 구조적 관계 모델링

현재 모델은 규칙 간 갈등을 암묵적으로만 처리:

$$r_{ji} = 1 \text{ AND } r_{ki} = 1 \text{ AND } \ell_j \neq \ell_k \Rightarrow \text{충돌}$$

→ **개선**: 그래픽 모델이나 Attention 메커니즘으로 규칙 간 의존성 명시적 모델링

#### ⑤ 도메인 외 일반화 (OOD Generalization)

- 현재 실험: 동일 분포 내 테스트만 수행
- **권장 실험**: 규칙이 특정 도메인에서 학습되고 다른 도메인에서 테스트되는 cross-domain 설정

#### ⑥ 규칙의 해석 가능성 연구

$P^j_\phi(r_j|\mathbf{x})$가 예측하는 커버리지 경계의 시각화 및 해석:

- 어떤 언어적/특징적 패턴에서 규칙이 무효화되는지 분석
- 이를 통해 더 정밀한 규칙 작성 가이드라인 제공

#### ⑦ 멀티모달 확장

현재 주로 텍스트 도메인에 집중:

$$\text{규칙}: \text{image region} \rightarrow \text{object class}$$

→ 비전-언어 멀티모달 약한 지도학습으로 확장 가능성

---

## 참고 자료

**주요 참고 논문 (이 논문에서 인용된 문헌):**

1. **Awasthi et al. (2020)** - "Learning from Rules Generalizing Labeled Exemplars" (ICLR 2020) — 본 논문
2. **Ratner et al. (2016)** - "Data Programming: Creating Large Training Sets, Quickly" (NeurIPS 2016)
3. **Zhang & Sabuncu (2018)** - "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels" (NeurIPS 2018)
4. **Ren et al. (2018b)** - "Learning to Reweight Examples for Robust Deep Learning"
5. **Hu et al. (2016)** - "Harnessing Deep Neural Networks with Logic Rules" (ACL 2016)
6. **Peters et al. (2018)** - "Deep Contextualized Word Representations" (ELMo)
7. **Ganchev et al. (2010)** - "Posterior Regularization for Structured Latent Variable Models" (JMLR)
8. **Veit et al. (2017)** - "Learning from Noisy Large-Scale Datasets with Minimal Supervision" (CVPR)

**2020년 이후 관련 최신 연구:**

9. **Zhang et al. (2021)** - "WRENCH: A Comprehensive Benchmark for Weak Supervision" (NeurIPS 2021 Datasets Track)
10. **Fu et al. (2020)** - "Fast and Three-rious: Speeding Up Weak Supervision with Triplet Methods" (ICML 2020)
11. **Bach et al. (2019)** - "Snorkel DryBell: A Case Study in Deploying Weak Supervision at Industrial Scale" (SIGMOD 2019)

**코드 및 데이터:**
- GitHub: https://github.com/awasthiabhijeet/Learning-From-Rules

---

> **⚠️ 정확도 관련 주의:** 2020년 이후 최신 연구 비교 분석 부분(WRENCH, CAGE, FlyingSquid와의 직접 비교)은 해당 논문들의 공개 내용에 기반하였으나, 논문 간 직접적 성능 수치 비교는 실험 설정 차이로 완전히 동등하지 않을 수 있습니다. SPEAR 논문의 경우 동일 저자 그룹의 후속 연구임을 확인하였으나, 세부 성능 수치는 원문 논문을 직접 확인하시길 권장합니다.
