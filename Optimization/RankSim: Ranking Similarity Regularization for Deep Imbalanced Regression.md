
# RankSim: Ranking Similarity Regularization for Deep Imbalanced Regression 

> **논문 정보**
> - **저자**: Yu Gong, Greg Mori, Fred Tung
> - **발표**: ICML 2022 (Spotlight)
> - **arXiv**: [2205.15236](https://arxiv.org/abs/2205.15236)
> - **코드**: https://github.com/BorealisAI/ranksim-imbalanced-regression

---

## 1. 핵심 주장 및 주요 기여 (간결 요약)

### 핵심 주장

데이터 불균형 문제, 즉 소수의 레이블에 대부분의 샘플이 집중되는 현상은 딥 뉴럴 네트워크 학습을 어렵게 만든다. 분류(classification)와 달리, 회귀(regression)에서의 레이블은 연속적이고 잠재적으로 무한하며 자연적인 순서(ordering)를 형성한다. 이러한 회귀의 고유한 특성은 레이블 공간의 관계 정보를 활용하는 새로운 기법을 필요로 한다.

이에 RankSim은 **"레이블 공간에서 가까운 샘플은 피처 공간에서도 가까워야 한다"** 는 귀납적 편향(inductive bias)을 핵심 주장으로 제시합니다.

### 주요 기여 요약

| 기여 항목 | 설명 |
|---|---|
| **새로운 정규화 기법** | RankSim 정규화 손실 함수 제안 |
| **전역적(Global) 관계 포착** | 근거리 + 원거리 레이블 관계를 모두 반영 |
| **기존 방법과의 상보성** | Re-weighting, Two-stage training, Distribution smoothing 등과 결합 가능 |
| **SOTA 달성** | 3개의 공개 불균형 회귀 벤치마크에서 최고 성능 달성 |

기존의 분포 스무딩(distribution smoothing) 방법들과 달리, RankSim은 근거리 및 원거리 관계를 모두 포착하며, 특정 샘플에 대해 레이블 공간에서의 이웃 정렬 리스트와 피처 공간에서의 이웃 정렬 리스트가 일치하도록 유도한다. RankSim은 re-weighting, 2단계 학습, 분포 스무딩 등 기존의 불균형 학습 기법들과 상보적(complementary)이며, IMDB-WIKI-DIR, AgeDB-DIR, STS-B-DIR 3개의 불균형 회귀 벤치마크에서 최고 성능을 달성한다.

---

## 2. 해결 문제 / 제안 방법 / 모델 구조 / 성능 및 한계

### 2-1. 해결하고자 하는 문제

분류에 비해 불균형 데이터로 딥 회귀 모델을 학습하는 방법은 아직 충분히 연구되지 않았다. 분류 네트워크가 이산적 레이블을 예측하는 것과 달리, 회귀 네트워크는 연속적인 값을 예측한다. 이러한 레이블 공간의 연속성은 깊은 불균형 회귀를 불균형 분류와 다르게 만든다. 한편으로, 목표값이 무한하고 경계가 없기 때문에 불균형 분류를 위한 많은 방법들이 적용 불가능하다.

또한, LDS(Label Distribution Smoothing)와 FDS(Feature Distribution Smoothing) 같은 기존 방법들은 가우시안 커널을 레이블 밀도 또는 피처 공간에 적용하여 근거리 레이블 관계만 포착한다. 그러나 이 방법들은 근방(nearby) 레이블 값만 고려하는 "국소적(local)" 귀납적 편향만을 인코딩한다는 한계가 있다.

### 2-2. 제안하는 방법 (수식 포함)

#### ① 핵심 아이디어

RankSim 정규화는 레이블 공간에서 가까운 항목이 피처 공간에서도 가까워야 한다는 귀납적 편향을 도입한다. RankSim은 LDS/FDS와 같은 스무딩 기법보다 더 "전역적(global)"인 레이블-피처 공간 관계에 대한 귀납적 편향을 인코딩하며, 근거리뿐만 아니라 원거리 관계까지 포착한다.

#### ② 유사도 행렬 구성

행렬 $S^y$와 $S^z$는 각각 레이블 공간과 피처 공간의 쌍별 유사도(pairwise similarity)를 인코딩한다. 특정 입력 샘플에 대해, RankSim은 정렬된 이웃 목록이 일치하도록 유도한다.

미니배치 $\mathcal{M} = \{(x_i, y_i)\}_{i=1}^{|\mathcal{M}|}$ 에 대해:

- **레이블 유사도 행렬** $S^y$:

$$S^y_{[i,j]} = \sigma^y(y_i, y_j)$$

- **피처 유사도 행렬** $S^z$:

$$S^z_{[i,j]} = \sigma^z(z_i, z_j)$$

여기서 $\sigma^z$는 코사인 유사도(cosine similarity)와 같이 벡터 위에서 정의된 유사도 함수이다.

#### ③ RankSim 손실 함수 (핵심 수식)

부분집합 $\mathcal{M}$에 대한 RankSim 정규화 손실은 다음과 같이 정의된다:

$$\mathcal{L}_{\text{RankSim}} = \sum_{i=1}^{|\mathcal{M}|} \ell\left(\mathbf{rk}(S^{y}_{[i,:]}), \, \mathbf{rk}(S^{z}_{[i,:]})\right) \tag{1}$$

여기서 $[i,:]$는 행렬의 $i$번째 행을 의미하며, 랭킹 유사도 함수 $\ell$은 두 입력 벡터 간의 차이를 페널티로 부과한다. 구체적으로 $\ell$로는 평균 제곱 오차(MSE)를 채택하며, 이는 레이블 공간과 피처 공간 랭킹 벡터의 **스피어만 상관계수(Spearman Correlation)**를 최대화하는 것과 동등하다.

#### ④ 미분 불가능 문제 해결 (Differentiable Ranking)

랭킹(ranking)은 기본적으로 계단 함수(step-wise function)를 형성하여 미분이 불가능하다. 이를 해결하기 위해, 미분 가능한(differentiable) 소프트 랭킹 연산자를 사용하여 그래디언트를 계산한다.

#### ⑤ 최종 훈련 목적 함수

전체 손실 함수는 기존 회귀 손실 $\mathcal{L}_{\text{reg}}$와 RankSim 정규화 항의 가중합으로 구성됩니다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{reg}} + \lambda \cdot \mathcal{L}_{\text{RankSim}} \tag{2}$$

여기서 $\lambda$는 RankSim 정규화의 강도를 조절하는 하이퍼파라미터입니다.

### 2-3. 모델 구조

유사도 행렬은 레이블에 대한 유사도 함수와 표현(representation)에 대한 유사도 함수를 이용해 각각 레이블 유사도 행렬과 표현 유사도 행렬로 구성된다. 손실은 두 유사도 행렬의 랭킹에 손실 함수를 적용한 결과들의 합이 된다.

RankSim 자체는 데이터셋의 불균형 문제를 직접 해결하지는 않는다. 따라서 이 방법은 Focal-R, RRT, SQINV 등 기존의 불균형 처리 방법에 추가적으로 적용되며, LDS와 FDS도 함께 사용된다.

**구조적 흐름도:**

```
입력 데이터 (x_i, y_i)
       ↓
  백본 네트워크 (Backbone)
       ↓
  특징 벡터 z_i (Feature Embedding)
       ↓
  ┌─────────────────────────────┐
  │  레이블 유사도 행렬 S^y      │
  │  피처 유사도 행렬 S^z        │
  │  → 각 행에 대해 rk() 적용   │
  │  → L_RankSim 계산           │
  └─────────────────────────────┘
       ↓
  L_total = L_reg + λ · L_RankSim
       ↓
  역전파(Backpropagation) 및 파라미터 업데이트
```

### 2-4. 성능 향상

실험에서는 Vanilla(기본 네트워크), Focal-R(회귀에 적용된 focal loss), RRT(회귀에 적용된 2단계 훈련), SQINV(제곱근 역빈도 재가중치)의 4가지 그룹으로 나누어 평가를 진행하였다.

RankSim은 MAE와 GM(기하평균) 지표 모두에서, 그리고 다양한 방법들에 걸쳐 성능 향상을 보여주었다. 애블레이션 연구에서는 MSE가 최적의 손실 함수이며, 코사인 유사도가 최적의 유사도 함수임을 확인하였다.

### 2-5. 한계점

논문 및 후속 연구에서 드러난 한계:

1. RankSim 자체는 데이터셋의 불균형 문제를 직접 해결하지 않으므로, 기존의 불균형 처리 방법과 반드시 결합해야 한다.

2. **배치 의존성**: 손실 함수가 미니배치 내 샘플 간의 관계를 기반으로 계산되기 때문에, 배치 크기와 샘플 구성에 민감할 수 있습니다.

3. **계산 복잡도**: 배치 내 모든 샘플 쌍에 대해 $O(|\mathcal{M}|^2)$ 유사도 행렬을 계산해야 하므로, 배치 크기가 클수록 계산 비용이 증가합니다.

4. 후속 연구인 BSAM은 불균형 회귀를 불균형 일반화 문제로 재정의하면서, RankSim과 같은 기존 방법이 소수 레이블 영역에 대한 균일한 일반화 능력을 보장하지 못한다는 점을 지적하였다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화 관점에서의 RankSim의 강점

RankSim 정규화는 최신 스무딩 방법들보다 레이블-피처 공간 관계에 대해 더 "전역적(global)"인 귀납적 편향을 인코딩하며, 근거리뿐만 아니라 원거리 관계까지 포착한다.

이는 다음과 같은 일반화 이점을 제공합니다:

- **소수 레이블(minority label) 영역에서의 표현 향상**: 소수 샘플은 레이블 공간에서 이웃하는 다수 샘플로부터 풍부한 피처 구조 신호를 간접적으로 전달받음
- **표현 공간의 순서 구조 보존**: $\mathcal{L}_{\text{RankSim}}$ 최소화는 스피어만 상관계수 최대화와 동치이므로, 연속적 레이블 순서 구조가 피처 공간에 반영됨
- **과적합 억제**: 정규화 항이 피처 공간의 지나친 붕괴(collapse) 방지

### 3-2. 소수 레이블 일반화의 이론적 근거

$$\text{Spearman's } \rho = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2-1)} \tag{3}$$

여기서 $d_i = \text{rk}(S^y_{[i,:]}) - \text{rk}(S^z_{[i,:]})$이며, $\rho \to 1$이 될수록 레이블-피처 공간의 순위 구조가 일치합니다. 이를 최대화하는 것은 소수 레이블 샘플이 레이블 공간에서의 관계를 피처 공간에서도 잘 유지하도록 강제하여 일반화를 향상시킵니다.

관련 후속 연구에서는 순서성(ordinality) 보존이 타깃 $Y$에 대한 표현 $Z$의 조건부 엔트로피 $H(Z|Y)$를 감소시킨다는 점을 밝혔으며, 피처 공간에서 타깃의 유사도 관계를 보존하기 위한 최적 수송(optimal transport) 기반 정규화를 도입하여 $H(Z|Y)$를 줄였다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 핵심 아이디어 | 한계 |
|---|---|---|---|
| **LDS/FDS** (Yang et al.) | ICML 2021 | 레이블/피처 공간에 가우시안 커널 스무딩 | 근거리(local) 관계만 포착 |
| **Balanced MSE** (Ren et al.) | NeurIPS 2022 | MSE에서 불균형이 예측으로 전파되지 않도록 복원 | 피처 공간 구조 미반영 |
| **RankSim** (Gong et al.) | ICML 2022 | 레이블-피처 공간 순위 유사도 정규화 | 불균형 자체 미해결, 배치 의존성 |
| **RnC** (Zha et al.) | NeurIPS 2023 | 대조적 방식으로 연속 랭킹 인식 임베딩 학습 | - |
| **ConR** (Keramati et al.) | ICLR 2024 | 피처 공간의 잘못된 근접성 페널티 부과 | - |
| **BSAM** (2025) | arxiv 2025 | SAM 기반 균형 일반화 보장 | - |

**세부 비교:**

RankSim은 레이블과 피처 간의 유사도 순서 대응(correspondence)을 장려함으로써 로컬 및 글로벌 의존성 모두를 활용한다. Balanced MSE는 MSE가 불균형을 예측 단계로 전달하지 않도록 균형 잡힌 예측 분포를 복원한다.

ConR은 레이블 공간과 피처 공간 간의 불일치를 식별하고 페널티를 부과하며, 레이블 유사도에 비례하여 잘못된 근접성에 페널티를 주는 대조적 방식을 사용한다. ConR은 깊은 불균형 회귀에서 피처 붕괴(feature collapse)를 해결하기 위한 대조 정규화기를 제안하였다.

RnC(Rank-n-Contrast)는 레이블 공간에서 상대적 순서를 기반으로 샘플을 대조함으로써 연속적이고 랭킹 인식 임베딩을 학습하며, Ordinal Entropy는 엔트로피 기반 정규화를 통해 로컬 순서 관계를 강화한다.

BSAM은 손실 지형(loss landscape)의 날카로움(sharpness)과 타겟 재가중치 메커니즘을 원칙적으로 통합하여 불균형 회귀 문제에 효과적으로 대응하는 방법으로, 소수 샘플 처리에서 기존 SAM의 한계를 분석하고 전체 데이터 분포에 걸쳐 모델 일반화를 균형 있게 향상시킨다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5-1. 앞으로의 연구에 미치는 영향

1. **"표현 공간 순서화"라는 새로운 패러다임 정립**
   표현 공간 교정(representation-space calibration)은 잠재 피처 공간에서 직접 구조적 규칙성을 강화함으로써 불균형 문제를 해결하며, RankSim은 레이블의 쌍별 랭킹 구조를 반영하도록 표현 공간을 명시적으로 교정한다.

2. **후속 연구(ConR, RnC, BSAM 등)의 직접적 촉발**
   RankSim이 레이블과 피처 간의 유사도 순서 대응을 장려하는 방식은 이후 대조 학습(contrastive learning) 기반의 ConR 등 다양한 후속 연구의 직접적인 동기가 되었다.

3. **연속 레이블 회귀와 표현 학습의 연결**
   회귀에서 예측 타깃은 자연적인 순서를 형성한다. 예를 들어, 나이 추정의 경우 가장 어린 사람부터 가장 나이 많은 사람까지 순서를 매길 수 있다. 이러한 레이블의 자연적 순서를 활용하여 신경망이 학습하는 표현을 정규화하는 방향으로 연구가 발전하고 있다.

### 5-2. 앞으로 연구 시 고려할 점

#### ① 배치 구성 전략 (Batch Sampling Strategy)
RankSim의 손실 계산은 미니배치 내 샘플 쌍에 의존하므로, 소수 레이블이 적절히 포함되도록 배치를 구성하는 **불균형 인식 배치 샘플링(imbalanced-aware batch sampling)** 전략 설계가 중요합니다.

$$\mathcal{M}_{\text{balanced}} = \arg\max_{\mathcal{M}} \sum_{i \neq j} \mathbb{1}[|y_i - y_j| > \delta] \cdot \mathbb{1}[(x_i, y_i) \in \text{minority}]$$

#### ② 소수 레이블 일반화 보장
BSAM은 불균형 회귀를 불균형 일반화 문제로 재정의하고, 전체 관측 공간에 걸쳐 회귀 모델의 균일한 일반화 능력을 강제하는 방법을 제안한다. 이 관점을 RankSim과 결합하여, 소수 레이블 영역에서의 일반화 갭(generalization gap)을 이론적으로 분석하는 연구가 필요합니다.

#### ③ 멀티모달/고차원 레이블 확장
현재 RankSim은 스칼라 레이블(scalar label)을 가정합니다. 다차원 연속 레이블이나 구조화된 출력(structured output) 공간에서의 랭킹 유사도 정의가 향후 과제입니다.

$$S^y_{[i,j]} = \exp\left(-\frac{\|y_i - y_j\|^2}{2\tau^2}\right) \quad \text{(다차원 레이블 확장 예시)}$$

#### ④ 이론적 수렴 보장
랭킹 기반 정규화가 소수 레이블 샘플의 표현 품질을 얼마나 향상시키는지에 대한 이론적 수렴 분석(convergence analysis)과 일반화 경계(generalization bound) 도출이 부재합니다.

#### ⑤ 태스크 확장 가능성
RnC의 사례처럼 일반적인 회귀 문제(general regression)에서도 연속적이고 랭킹 인식 임베딩을 대조적으로 학습하는 방향으로의 확장이 활발히 연구되고 있다. 의료 영상, 기상 예측, 금융 시계열 등 다양한 도메인으로의 적용 가능성을 검토할 필요가 있습니다.

---

## 📚 참고자료 목록

| # | 제목 | 출처 |
|---|---|---|
| 1 | **RankSim: Ranking Similarity Regularization for Deep Imbalanced Regression** | arXiv:2205.15236 / ICML 2022 |
| 2 | RankSim ICML 2022 Spotlight | https://icml.cc/virtual/2022/spotlight/16496 |
| 3 | RankSim (PMLR 공식 proceedings) | https://proceedings.mlr.press/v162/gong22a.html |
| 4 | RankSim HTML 상세 (ar5iv) | https://ar5iv.labs.arxiv.org/html/2205.15236 |
| 5 | RBC Borealis 공식 게시 | https://rbcborealis.com/publications/ranksim-ranking-similarity-regularization-for-deep-imbalanced-regression/ |
| 6 | Semantic Scholar | https://www.semanticscholar.org/paper/RankSim .../7f8883d82c4bf7111e91d8bab8b954dda9345e4a |
| 7 | **ConR: Contrastive Regularizer for Deep Imbalanced Regression** | arXiv:2309.06651 / ICLR 2024 |
| 8 | ConR RBC Borealis 블로그 | https://rbcborealis.com/research-blogs/conr-contrastive-regularizer-for-deep-imbalanced-regression/ |
| 9 | **Balanced Sharpness-Aware Minimization for Imbalanced Regression (BSAM)** | arXiv:2508.16973 |
| 10 | **Rank-n-Contrast (RnC)** | NeurIPS 2023, Zha et al. |
| 11 | **Delving into Deep Imbalanced Regression (LDS/FDS)** | ICML 2021, Yang et al. |
| 12 | Highly Imbalanced Regression with Tabular Data | arXiv:2509.16339 |
| 13 | RankSim 리뷰 블로그 | https://dgriessler.github.io/gong.html |

---

> ⚠️ **정확성 안내**: 본 답변은 공개된 논문 원문(arXiv, PMLR proceedings), 공식 RBC Borealis 게시물, 및 관련 후속 연구를 기반으로 작성되었습니다. 수식의 세부 표기는 논문 원문 PDF(proceedings.mlr.press/v162/gong22a/gong22a.pdf)를 직접 확인하시길 권장합니다.
