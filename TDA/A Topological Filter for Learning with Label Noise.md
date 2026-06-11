# A Topological Filter for Learning with Label Noise

---

## 1. 핵심 주장 및 주요 기여 (간결 요약)

### 핵심 주장

본 논문(Wu et al., NeurIPS 2020)은 **레이블 노이즈(label noise)** 문제를 해결하기 위해, 기존의 사후확률(posterior probability) 기반 접근법에서 벗어나 **잠재 표현 공간(latent representational space)의 위상학적(topological) 구조**를 활용하는 새로운 방법론 **TopoFilter**를 제안합니다.

핵심 관찰: *이상적인 특징 표현(ideal feature representation)이 주어졌을 때, 깨끗한 데이터는 클러스터를 형성하고, 노이즈 데이터는 고립(isolated)된다.*

### 주요 기여

1. **TopoFilter 알고리즘**: KNN 그래프 위에서 각 클래스의 최대 연결 성분(Largest Connected Component, LCC)을 찾고, $\zeta$-필터링으로 외곽 데이터를 제거하는 반복적 정제 방법
2. **이론적 보장**: 클린 데이터 수집의 **순수성(Purity)**과 **풍부성(Abundancy)**을 확률론적으로 증명한 최초의 위상학 기반 방법
3. **실증적 우수성**: CIFAR-10, CIFAR-100, Clothing1M에서 당시 최신 기법(SOTA) 대비 일관적 성능 향상
4. **범용성**: 손실 함수에 독립적이며 다양한 노이즈 유형/레벨에 강건

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

딥러닝 모델은 강한 기억화(memorization) 능력으로 인해 노이즈 레이블을 그대로 학습하여 일반화 성능이 크게 저하됩니다. 기존 방법들은 주로 노이즈 분류기의 예측 신뢰도(posterior probability)에 기반하여 클린 데이터를 선별하는데, 이는:

- 이론적 보장이 없음
- 차원 축소 과정에서 정보 손실 발생
- 특정 노이즈 패턴에 취약

본 논문은 **잠재 공간의 공간적 위상 구조**라는 더 풍부한 정보를 활용하여 이를 극복하고자 합니다.

### 2.2 제안 방법 및 수식

#### 기본 설정 및 표기

- 특징 공간 $\mathcal{X} \subset \mathbb{R}^d$, 레이블 공간 $\mathcal{Y} = \{0, 1\}$
- 진짜(clean) 레이블 $y$, 관측된(noisy) 레이블 $\tilde{y}$
- 노이즈 전이 확률: $y = i$가 $\tilde{y} = j$로 뒤집힐 확률 $\tau_{ij}$ (x에 독립)
- 클린/노이지 사후 확률: $\eta_i(\boldsymbol{x}) = P(y=i|\boldsymbol{x})$, $\tilde{\eta}_i(\boldsymbol{x}) = P(\tilde{y}=i|\boldsymbol{x})$
- 이 두 확률은 선형 관계를 만족:

$$\tilde{\eta}_i(\boldsymbol{x}) = (1 - \tau_{01} - \tau_{10})\eta_i(\boldsymbol{x}) + \tau_{1-i,i}, \quad \forall i \in \{0,1\}$$

- 초수준 집합(Superlevel set): $L(t) = \{\boldsymbol{x} \mid \max(\eta_1(\boldsymbol{x}), \eta_0(\boldsymbol{x})) \geq t\}$

#### 핵심 공간 분할

```math
A_i^+ = \left\{\boldsymbol{x} : \eta_i(\boldsymbol{x}) > \max\!\left(\tfrac{1}{2},\, \tfrac{1/2 - \max(\tau_{10},\tau_{01})}{2(1-\tau_{10}-\tau_{01})}\right)\right\}
```

```math
A_i^- = \left\{\boldsymbol{x} : \eta_i(\boldsymbol{x}) < \min\!\left(\tfrac{1}{2},\, \tfrac{1/2 - \max(\tau_{10},\tau_{01})}{2(1-\tau_{10}-\tau_{01})}\right)\right\}
```

$$A^b = \mathcal{X} \setminus (A_i^+ \cup A_i^-)$$

$A_i^+$: 클린/노이즈 베이즈 분류기가 동일하게 $i$로 예측하는 "좋은 영역" → TopoFilter가 수집 목표로 하는 영역  
$A^b$: 결정 경계 근처의 불확실 영역 → 제거 대상

#### 순수성(Purity) 정의

**최소 순수성:**
$$\ell_{S_n,\mathcal{A}} := \min_{i\in\{0,1\}} \min_{\boldsymbol{x}\in C_i} P(y=i \mid \tilde{y}=i, \boldsymbol{x}) = \min_{i\in\{0,1\}} \min_{\boldsymbol{x}\in C_i} \tau_{ii}\frac{\eta_i(\boldsymbol{x})}{\tilde{\eta}_i(\boldsymbol{x})}$$

**평균 순수성:**

$$\ell'_{S_n,\mathcal{A}} := \sum_{i\in\{0,1\}} \frac{1}{|C_i|}\sum_{\boldsymbol{x}\in C_i} \tau_{ii}\frac{\eta_i(\boldsymbol{x})}{\tilde{\eta}_i(\boldsymbol{x})}$$

#### 알고리즘 (TopoFilter)

**Algorithm 1** 의 핵심 단계:

1. **Early stopping**: 초기 에폭 $m$까지 노이즈 데이터 전체로 학습 → 초기 표현 획득
2. **KNN 그래프 구성**: 잠재 특징 $\boldsymbol{x}$ 위에 mutual $k$-NN 그래프 $G$ 구성
   - 엣지 집합: $E = \{(\boldsymbol{x}_1, \boldsymbol{x}_2) \mid \boldsymbol{x}_1 \in KNN(\boldsymbol{x}_2) \text{ or } \boldsymbol{x}_2 \in KNN(\boldsymbol{x}_1)\}$
3. **클래스별 서브그래프**: 각 클래스 $i$에 대해 해당 레이블 데이터만으로 $G_i$ 구성
4. **최대 연결 성분(LCC)**: 각 $G_i$에서 최대 연결 성분 $Q_i$ 추출
   - $C \leftarrow \bigcup_i Q_i$
5. **$\zeta$-필터링**: $C$ 내의 각 점 $\boldsymbol{x}$에 대해, $S$(각 클래스 LCC의 합집합)에서의 $k$-최근접 이웃 중 같은 레이블을 가진 비율이 $\zeta$ 이상인 경우만 클린으로 선별
   - $\boldsymbol{x}$가 레이블 $\tilde{y}$를 가질 때: $|KNN(\boldsymbol{x}) \cap \{\text{레이블}=\tilde{y}\}| / k \geq \zeta$
6. **재학습**: 선별된 클린 데이터 $\hat{S} \leftarrow C$로 네트워크 재학습 → 표현 개선 → 반복

#### 이론적 보장 정리

**가정:**
- **A1**: $f(\boldsymbol{x})$ (특징 공간 밀도)는 컴팩트 지지(compact support)
- **A2**: $\forall i\in\{0,1\}$, $\eta_i(\boldsymbol{x})$는 연속(continuous)
- **A3**: $\forall i\in\{0,1\}$, $A_i^+$는 연결 집합(connected set)
- **A4**: $\tau_{10}, \tau_{01} \in \left[0, \frac{1}{2}\right)$

**정리 1 (순수성 보장):** $\forall \delta>0$, $\forall \zeta > \frac{1+|\tau_{10}-\tau_{01}|}{2}$, 충분히 큰 $n$과 적절한 $k \in [c_1(\zeta)\log^q n,\, c_2 n]$에 대해:

$$P\!\left[\left(\ell_{S_n,\mathcal{A}_\zeta} - \ell_{S_n,\mathcal{A}_0}\right) > g_1(\zeta)\right] \geq 1 - \delta$$

여기서:

```math
g_1(\zeta) \in \left[\frac{[2\zeta+1+|\tau_{10}-\tau_{01}|) - 4\max(\tau_{10},\tau_{01})]\min(\tau_{11},\tau_{00})}{[2\zeta+1+|\tau_{10}-\tau_{01}|](1-\tau_{10}-\tau_{01})},\; 1\right]
```

**정리 2 (풍부성 보장):** 수집된 클린 데이터 수 

```math
n_c = \#\left\{\bigcup_i C^{(i)}(\zeta)\right\}
```

에 대해:

```math
P\!\left[\left|\frac{n_c}{n} - \mu(L(\zeta))\right| \leq \epsilon\right] \geq 1 - \delta
```

즉, 알고리즘이 수집하는 데이터 비율은 초수준 집합 $L(\zeta)$의 확률 측도 $\mu(L(\zeta))$에 수렴합니다.

**핵심 보조 정리:**

- **Lemma 1 (연결성)**: $k = \Omega(\log^q n)$이면 $L(t) \cap X_i$는 $G_i$에서 연결됨 (w.p. $\geq 1-\delta$)
- **Lemma 2 (고립성)**: $k$가 적절히 작으면 $X_i(\zeta)$와 $X_i^c(\zeta')$ 사이에 엣지 없음 (w.p. $\geq 1-\delta$)  
  (여기서 $\zeta' = \frac{1}{2}\!\left(\zeta + \frac{1+|\tau_{10}-\tau_{01}|}{2}\right)$ )
- **Lemma 3 ($\zeta$-필터링)**: 필터링 후 $L(\zeta')^c$에 속하는 점이 수집 집합에 없음 (w.p. $\geq 1-\delta$)

#### $\zeta$ 선택 전략

- 초기 에폭: $\zeta$를 높게 설정(예: $3/4$) → 높은 순수성, 낮은 풍부성
- 후기 에폭: $\zeta \to (1/2 + \epsilon)$으로 감소 → 풍부성 증가, 베이즈 결정 경계 근방 데이터 포함
- 실제로는 $\zeta = 0.5$ 고정으로도 강건한 성능

### 2.3 모델 구조

```
[전체 훈련 데이터 (noisy)]
        ↓ Early stopping (에폭 m까지)
[초기 노이즈 분류기]
        ↓ 잠재 특징 추출 (penultimate layer)
[KNN 그래프 G 구성]
        ↓ 클래스별 서브그래프 G_i
[최대 연결 성분 Q_i 추출] ← TopoCC 단계
        ↓ ζ-필터링 (이웃 레이블 비율 체크)
[정제된 클린 데이터 C] ← TopoFilter 단계
        ↓ C로만 재학습
[개선된 분류기] → 더 나은 특징 표현 → 반복
```

- **백본**: ResNet-18 (CIFAR-10/100), ResNet-50 (Clothing1M), PointNet (ModelNet40)
- **KNN 그래프**: CUDA 가속, C++로 LCC 계산 → 반복당 ~1초
- **선별 주기**: 5 에폭마다 데이터 선별 수행

### 2.4 성능 향상

**CIFAR-10 (균일 노이즈 60%):**

| 방법 | 정확도 |
|------|--------|
| Standard | 73.7% |
| Co-teaching | 79.0% |
| PENCIL | 74.3% |
| **TopoFilter** | **80.5%** |

**CIFAR-100 (쌍 뒤집기 노이즈 40%):**

| 방법 | 정확도 |
|------|--------|
| Standard | 42.3% |
| RoG | 58.8% |
| PENCIL | 61.9% |
| **TopoFilter** | **62.4%** |

**Clothing1M (실세계 노이즈):**

| 방법 | 정확도 |
|------|--------|
| Standard | 68.94% |
| PENCIL | 73.49% |
| **TopoFilter** | **74.10%** |

모든 노이즈 설정에서 통계적으로 유의미한 개선 (95% 신뢰 수준의 비쌍 t-검정 확인).

### 2.5 한계점

1. **이론적 한계**:
   - 이진 분류에 대한 이론 → 다중 클래스로의 직접 확장 필요
   - 수렴(convergence) 결과 미제공 (일회성(one-shot) 보장만)
   - 초기 표현의 질에 이론이 암묵적으로 의존

2. **실용적 한계**:
   - 극도로 높은 노이즈율(80%)에서 성능 급락
   - 클래스 간 의미적으로 유사한 경우 KNN 그래프의 신뢰도 저하
   - 대규모 데이터셋에서 KNN 계산 비용 (Clothing1M: ~25초/에폭)
   - 클린 검증 데이터에 여전히 의존 (모델 선택 용도)
   - 잠재 공간 차원 및 초기 표현 품질에 민감할 가능성 존재

3. **방법론적 한계**:
   - 연결 성분 외의 고차 위상 정보(Persistent Homology 등) 미활용
   - 동일 클래스 내 다중 서브클러스터가 존재할 경우 취약 가능성

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상의 핵심 메커니즘

TopoFilter가 일반화 성능을 향상시키는 근본 원인은 다음과 같습니다:

**① 깨끗한 훈련 데이터 확보 → 과적합 방지**

딥러닝의 기억화 현상(Zhang et al., ICLR 2017)에 따르면, 모델은 노이즈 레이블을 포함한 데이터에도 과적합합니다. TopoFilter는 정리 1에 의해 수집된 데이터의 순수성을 보장하므로:

$$P\!\left[\ell_{S_n,\mathcal{A}_\zeta} - \ell_{S_n,\mathcal{A}_0} > g_1(\zeta)\right] \geq 1 - \delta$$

이는 모델이 실제로 깨끗한 데이터로만 학습하게 되어 **훈련-테스트 분포 불일치를 최소화**합니다.

**② 표현 학습의 선순환**

```
깨끗한 데이터로 학습
    ↓
더 나은 잠재 표현
    ↓
클린/노이즈 데이터 더 잘 분리
    ↓
더 순수한 클린 데이터 수집
    ↓ (반복)
```

Figure 1(c)에서 에폭이 증가할수록 클린 데이터 클러스터가 더욱 명확히 형성됨을 t-SNE 시각화로 확인.

**③ 전역 위상 구조 활용 → 지역 아티팩트 내성**

기존 방법들이 개별 샘플의 손실/확률값에 의존하는 반면, TopoFilter는 **집단(group) 행동**을 봅니다. 소수의 노이즈 데이터가 작은 클러스터를 형성하더라도, 클래스 전체의 LCC에서 배제됩니다.

**④ 베이즈 최적 분류기와의 일관성 (Theorem 3)**

$$\forall \boldsymbol{x} \in \bigcup_i C^{(i)}(\zeta): \quad P[\tilde{y}(\boldsymbol{x}) = h^*(\boldsymbol{x})] \geq 1 - \delta$$

수집된 데이터의 레이블이 베이즈 최적 분류기의 예측과 일치하므로, 이 데이터로 학습한 모델은 이론적으로 베이즈 최적에 수렴할 수 있습니다.

**⑤ 하이퍼파라미터에 대한 강건성 → 다양한 환경에서의 일반화**

- 검증 집합 크기/품질에 강건: 노이즈 검증 집합에서도 유사한 성능
- $k_c, k_o, \zeta$ 변화에 성능 거의 불변
- 특징 차원에 무관

이는 모델이 특정 하이퍼파라미터 설정에 과도하게 의존하지 않음을 의미하며, **미지의 데이터셋에서도 안정적 성능**을 기대할 수 있습니다.

**⑥ 다양한 도메인으로의 이전 가능성**

이미지(CIFAR, Clothing1M)뿐만 아니라 3D 포인트 클라우드(ModelNet40)에서도 우수한 성능을 보여, 도메인 독립적 일반화 가능성을 시사합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

**① 위상수학적 방법론의 딥러닝 적용 촉진**

본 논문은 대수적 위상수학 개념(연결 성분, persistent homology)을 레이블 노이즈 학습에 성공적으로 적용한 선례를 확립했습니다. 이는 다음 연구 방향을 열었습니다:
- Persistent homology 기반 정규화 손실 (논문 결론에서 미래 작업으로 언급)
- 잠재 공간의 기하학/위상학적 분석을 통한 모델 신뢰도 측정

**② 잠재 공간의 공간적 구조 중요성 재인식**

사후 확률이라는 스칼라 값보다 잠재 공간의 풍부한 기하학적 정보가 더 유용할 수 있다는 점을 실증적·이론적으로 보였습니다. 이는 표현 학습(representation learning) 연구에도 시사점을 제공합니다.

**③ 이론적 보장의 새로운 기준 제시**

레이블 노이즈 분야에서 순수성 + 풍부성의 동시 보장이라는 이론적 프레임워크를 제시하여, 후속 연구들이 이를 벤치마크로 삼아 더 강한 이론적 보장을 추구하게 합니다.

**④ 데이터 선별 패러다임의 발전**

TopoFilter의 방법론은 단순한 손실 기반 선별을 넘어, 데이터의 집단적 구조를 활용하는 패러다임을 제안했습니다. 이는 준지도 학습(semi-supervised learning), 연속 학습(continual learning), 페더레이션 학습(federated learning)에서의 데이터 품질 관리에도 응용 가능합니다.

### 4.2 향후 연구 시 고려할 사항

**① 높은 노이즈율 환경에서의 한계 극복**

80% 균일 노이즈에서 성능이 급락하는 문제를 해결하기 위해:
- 초기 표현 개선을 위한 자기지도 학습(self-supervised learning) 사전 훈련
- 노이즈율 추정 기법과의 결합
- 의미적 유사 클래스 간 노이즈(semantic noise)에 특화된 위상 분석

**② 수렴 이론 수립**

논문 자체가 명시적으로 언급하듯, 반복 과정에서의 수렴 보장이 미해결 문제입니다. 이를 위해:
- 각 에폭에서 클린 데이터 집합의 단조 증가/순수성 보존 증명
- 표현 품질과 선별 정확도 간의 상호작용 분석

**③ Persistent Homology 기반 확장**

0차원 연결 성분(connected components)을 넘어:
- 1차원 루프, 2차원 공동(cavity) 등 고차 위상 정보 활용
- Differentiable topological loss (Chen et al., AISTATS 2019)와의 결합
- 위상적 데이터 분석(TDA)의 최신 알고리즘 적용

**④ 다중 클래스 이론 확장**

현재 이진 분류 이론을 다중 클래스로 일반화:
- 노이즈 전이 행렬 $T \in \mathbb{R}^{C \times C}$에 대한 일반 분석
- 클래스 불균형(class imbalance)과의 상호작용 연구

**⑤ 계산 효율성 개선**

대규모 데이터셋에서의 적용을 위해:
- 근사 KNN 알고리즘(FAISS, HNSW 등) 활용
- 그래프 샘플링 기반 LCC 근사
- 분산 컴퓨팅 환경에서의 KNN 그래프 구성

**⑥ 실세계 노이즈 패턴 대응**

실제 환경에서는 노이즈가 클래스/특징에 의존적(feature-dependent)일 수 있음:
- Instance-dependent noise에 대한 이론/알고리즘 확장
- 편향된 어노테이터 모델(annotator bias model)과의 통합

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래는 제가 훈련 데이터 기반으로 알고 있는 내용이며, 논문 PDF에 직접 인용되지 않은 2020년 이후 연구들입니다. 일부 세부 수치는 제 학습 데이터의 한계로 부정확할 수 있으므로, 원문 확인을 권장합니다.

### 5.1 주요 후속 연구 동향

| 연구 | 방법론 | TopoFilter 대비 특징 |
|------|--------|---------------------|
| DivideMix (Li et al., ICLR 2020) | GMM 기반 클린/노이즈 분리 + MixUp + 반지도 학습 | 더 높은 성능이나 복잡한 파이프라인, 이론 보장 없음 |
| CORES² (Cheng et al., NeurIPS 2021) | 클래스별 센트로이드 거리 기반 선별 | 공간 정보 활용하나 위상 정보 미사용 |
| PES (Liu et al., NeurIPS 2022) | 앙상블 기반 반복적 레이블 수정 | 다중 모델 필요, 계산 비용 높음 |
| SOP (Liu et al., ICML 2022) | 과최적화 방지 정규화 | 데이터 선별 없이 손실 설계만으로 접근 |

### 5.2 DivideMix (Li et al., ICLR 2020)와의 비교

DivideMix는 다음 파이프라인을 사용합니다:
1. GMM으로 손실 분포 모델링 → 클린/노이즈 확률 추정
2. 클린 데이터로 지도 학습 + 노이즈 데이터로 반지도 학습(MixMatch)
3. 두 네트워크 교차 학습

**TopoFilter 대비 장점**: CIFAR-10에서 더 높은 절대 성능  
**TopoFilter 대비 단점**: 이론적 보장 없음, 더 복잡한 구현, 두 네트워크 필요

### 5.3 위상학적 방법의 최신 발전

- **TopoSemiSeg** 등: TopoFilter의 위상학적 아이디어를 반지도 분할(semi-supervised segmentation)로 확장하는 시도들이 등장
- **Persistent Homology 기반 손실**: Chen et al. (AISTATS 2019)에서 시작된 흐름이 레이블 노이즈 문제에도 점진적으로 적용되는 추세

### 5.4 종합 비교 관점

TopoFilter는 다음 관점에서 여전히 차별화됩니다:

```
이론적 보장:  TopoFilter ★★★★  vs  DivideMix ★★  vs  Co-teaching ★
단일 모델:    TopoFilter ★★★★  vs  Co-teaching ★★  vs  DivideMix ★★
강건성:       TopoFilter ★★★★  vs  대부분의 방법 ★★~★★★
극고 노이즈:  TopoFilter ★★   vs  DivideMix ★★★★
```

---

## 참고 자료

**주요 참고 논문 (논문 내 인용)**

- Wu, P., Zheng, S., Goswami, M., Metaxas, D., & Chen, C. (2020). **A Topological Filter for Learning with Label Noise**. *NeurIPS 2020*. arXiv:2012.04835
- Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding deep learning requires rethinking generalization. *ICLR 2017*.
- Patrini, G., et al. (2017). Making deep neural networks robust to label noise: A loss correction approach. *CVPR 2017*.
- Han, B., et al. (2018). Co-teaching: Robust training of deep neural networks with extremely noisy labels. *NeurIPS 2018*.
- Chaudhuri, K. & Dasgupta, S. (2010). Rates of convergence for the cluster tree. *NeurIPS 2010*.
- Chen, C., Ni, X., Bai, Q., & Wang, Y. (2019). A topological regularizer for classifiers via persistent homology. *AISTATS 2019*.
- Edelsbrunner, H. & Harer, J. (2010). *Computational Topology: An Introduction*. AMS.
- Yi, K. & Wu, J. (2019). PENCIL: Probabilistic end-to-end noise correction. *CVPR 2019*.
- Jiang, L., et al. (2018). MentorNet. *ICML 2018*.
- Lee, K., et al. (2019). RoG: Robust inference via generative classifiers. *ICML 2019*.

**2020년 이후 관련 연구 (비교 분석 참조)**

- Li, J., Socher, R., & Hoi, S. C. H. (2020). DivideMix: Learning with noisy labels as semi-supervised learning. *ICLR 2020*.
- Zheng, S., Wu, P., Goswami, A., Goswami, M., Metaxas, D., & Chen, C. (2020). Error-bounded correction of noisy labels. *ICML 2020*.
