# Lost in Tokenization: Fundamental Trade-offs in Graph Tokenization for Transformers

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 **그래프 토크나이제이션(graph tokenization)이 단순한 전처리 선택이 아니라, 트랜스포머의 표현력(expressivity)을 결정하는 근본적인 구성 요소**라는 것입니다. 동일한 트랜스포머 아키텍처라도 어떤 토크나이제이션을 선택하느냐에 따라 풀 수 있는 문제의 종류와 필요한 레이어 깊이가 근본적으로 달라집니다.

### 주요 기여 (5가지)

| 기여 | 내용 |
|------|------|
| ① 이론적 형식화 | 그래프 토크나이제이션을 계산 모델의 일부로 공식 정의 |
| ② 깊이 분리 증명 | Adjacency, Spectral, Random-walk 간 깊이 분리(depth separation) 증명 |
| ③ 스펙트럼 취약성 | Spectral 토크나이제이션의 ill-conditioning 및 절삭(truncation)의 취약성 증명 |
| ④ 변환 불가능성 | 제한된 깊이에서 토크나이제이션 간 변환이 일반적으로 불가능함을 증명 |
| ⑤ 실험적 검증 | 합성 및 실제 데이터셋에서 이론적 예측을 실험으로 확인 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2-1. 해결하고자 하는 문제

**그래프 트랜스포머에서 토크나이제이션이 표현력에 미치는 영향이 체계적으로 연구된 바 없다는 문제**를 다룹니다. 기존 연구들은 Spectral, Random-walk, Adjacency 기반 인코딩을 경험적 설계 선택으로만 다뤘으나, 이 논문은 이를 계산적 표현력의 축으로 공식화합니다.

구체적 질문:
- 서로 다른 토크나이제이션이 동일한 태스크에 대해 서로 다른 계산 복잡도를 유발하는가?
- 특정 토크나이제이션이 내재적으로 정보 손실(lossy)한가?
- 트랜스포머가 불리한 토크나이제이션을 내부적으로 더 유리한 것으로 변환할 수 있는가?

---

### 2-2. 세 가지 토크나이제이션 정의 (수식 포함)

#### (a) Spectral Tokenization (스펙트럼 토크나이제이션)

그래프 라플라시안의 고유분해를 활용합니다:

$$\mathbf{L} = \mathbf{D} - \mathbf{A} = \mathbf{U}\mathbf{\Lambda}\mathbf{U}^\top$$

각 노드 $v$에 대한 토큰:

$$\mathbf{x}_v = \left(u_1(v), \ldots, u_n(v), \lambda_1, \ldots, \lambda_n\right) \in \mathbb{R}^{2n}$$

- **전체 스펙트럼**: $\mathcal{P}_{\text{Lap}}(G) \in \mathbb{R}^{n \times 2n}$, lossless
- **절삭 버전** ($k \leq n$ 사용):

$$\mathbf{x}_v = \left[u_1(v), \ldots, u_k(v), \lambda_1, \ldots, \lambda_k\right] \in \mathbb{R}^{2k}$$

$\mathcal{P}_{\text{Lap},k}(G) \in \mathbb{R}^{n \times 2k}$, lossy

#### (b) Random-Walk Tokenization (랜덤 워크 토크나이제이션)

전이 행렬 $\mathbf{P} = \mathbf{D}^{-1}\mathbf{A}$로부터 각 노드의 귀환 확률을 토큰으로 사용:

$$\mathbf{x}_v = \left((\mathbf{P}^1)_{vv}, \ldots, (\mathbf{P}^{t(n)})_{vv}\right) \in \mathbb{R}^{t(n)}$$

- 전처리 비용: $T_{\text{pre}}(n) = \mathcal{O}(t(n)|E|)$
- **임의의 $t(n)$에 대해 lossy** (Theorem 2 참조)

#### (c) Adjacency Tokenization (인접 행렬 토크나이제이션)

각 노드 $v$를 인접 행렬 행으로 표현:

$$\mathcal{P}_{\text{Adj}}(G) = \begin{bmatrix}\mathbf{A}_1^\top & \cdots & \mathbf{A}_n^\top\end{bmatrix} \in \mathbb{R}^{n \times n}$$

- Lossless, 전처리 비용 $T_{\text{pre}}(n) = \mathcal{O}(|E|)$
- 절삭 변형: $\mathcal{P}\_{\text{adj,tr}}(G) = \mathbf{A}\mathbf{R} \in \mathbb{R}^{n \times d_{\text{tr}}}$ (랜덤 행렬 $\mathbf{R} \in \mathbb{R}^{n \times d_{\text{tr}}}$)
- **순열 동변환(permutation-equivariant)이 아님**

---

### 2-3. 트랜스포머 모델 구조

$L$ 레이어, 은닉 차원 $m$, $H$ 어텐션 헤드를 가진 트랜스포머:

$$Z^{(\ell)} = X^{(\ell-1)} + \text{MHA}(X^{(\ell-1)}), \quad X^{(\ell)} = Z^{(\ell)} + \text{MLP}(Z^{(\ell)})$$

각 어텐션 헤드 $h \in [H]$:

$$\text{head}_h(X) = \text{softmax}\!\left(\frac{X W_Q^h (X W_K^h)^\top}{\sqrt{d_h}}\right) X W_V^h$$

여기서 $W_Q^h, W_K^h, W_V^h \in \mathbb{R}^{m \times d_h}$, $d_h = m/H$.

입력: $X^{(0)} = \mathcal{P}\_n(G) \in \mathbb{R}^{n \times d_{\text{tok}}}$ → 선형 투영 → $\mathbb{R}^{n \times m}$

---

### 2-4. 핵심 이론적 결과

#### **Theorem 1**: Adjacency의 $k$-닫힌 보행 탐지 하한

> $\text{TC}^0 \subsetneq \text{NC}^1$ 가정 하에, Adjacency 토크나이제이션을 사용하는 트랜스포머는 $k$-closed-walk detection을 풀기 위해 $\Omega(\log k)$ 깊이가 필요하다.

- **반면**, Random-walk 토크나이제이션은 $\mathcal{O}(1)$ 깊이로 해결 가능 (Theorem 6)

**증명 직관**: $k$-분할 그래프에서 에지 순회를 순열군 $S_5$의 합성으로 환원. $S_5$의 단어 문제는 $\text{NC}^1$-complete이므로, 상수 깊이 트랜스포머($\text{TC}^0$)로 해결 불가.

---

#### **Theorem 2**: 플래너리(Planarity) 결정 불가

> 임의 길이 $t(n)$의 Random-walk 토크나이제이션은 그래프 평면성(planarity)을 결정하기 불충분하다. (모델 깊이/너비에 무관)

**증명 직관**: Godsil-McKay(GM) 전환을 통해 동일한 Random-walk 분포를 가지는 평면 그래프 $G$와 비평면 그래프 $G'$를 구성.

$$\left(D^{-1}A'\right)^m = \left(D^{-1}QAQ\right)^m = Q\left(D^{-1}A\right)^m Q$$

$$\left[Q(D^{-1}A)^m Q\right]_{ii} = \left[(D^{-1}A)^m\right]_{ii}$$

따라서 두 그래프의 Random-walk 토큰이 동일.

---

#### **Theorem 3**: 절삭(Truncation)에서의 삼각형 계산 실패

> 삼각형 수를 계산하는 트랜스포머 $T$에 대해:
> 1. **Laplacian**: $k < n-1$개의 고유값만 사용하면 삼각형 계산 불가
> 2. **Adjacency**: 은닉 차원 $m$, 정밀도 $p$, 헤드 수 $H$, 깊이 $L$에서:
>    - Residual 연결 有: $mpHL = \Omega(n)$
>    - Residual 연결 無: $mpH = \Omega(n)$

삼각형 수는 $\frac{1}{6}\text{Tr}(A^3)$으로 계산되며, 단 하나의 고유값을 제거해도 계산이 불가능해집니다.

---

#### **Theorem 4**: 연결성에 대한 $\Omega(\log n)$ 깊이 하한

> $\text{TC}^0 \subsetneq \text{L}$ 가정 하에, 전체 $\Theta(n)$ Adjacency 행이 주어져도 그래프 연결성을 결정하려면 $\Omega(\log n)$ 깊이가 필요하다.

**반면**, Spectral 토크나이제이션은 $L=1$ 깊이로 연결성 분류 가능 (실험 Figure 2a).

---

#### **Theorem 5**: Laplacian의 지역 엣지 예측 ill-conditioning

> 최대 차수 $d_{\max}$를 가진 그래프에서 Laplacian 토크나이제이션으로 엣지 $(u,v)$ 존재를 예측하는 1-레이어 트랜스포머의 파라미터는 다음을 만족해야 한다:

$$\text{Lip}_{\text{MLP}} \cdot \|W_V\|_2 (1 + \gamma) \geq d_{\max}$$

여기서 $\gamma = \|W_{QK}\|_2 \|X\|_2^2$ (최대 로짓 에너지).

**최적화 딜레마**: Laplacian 토큰의 노름이 그래프 크기에 따라 스케일링되어 ( $\|X\|_2^2 = \Omega(n^2)$ ), Softmax 포화(saturation) 또는 가중치 폭발(explosion) 중 하나가 필연적으로 발생합니다.

**깊은 트랜스포머로의 확장** (Remark 1): $L$ 레이어 트랜스포머에서 레이어당 요구사항:

$$\|W_V\|_2(1 + \gamma) \geq \Omega\left(d_{\max}^{1/L}\right)$$

---

### 2-5. 토크나이제이션 간 변환 불가능성 요약

```
Truncated tokenizations
        ↘ Impossible (Theorem 3)
Adjacency (P_Adj) ←→ Laplacian (P_Lap)
    |  Ω(log n) (Theorem 4)        |  Ill-cond. (Theorem 5)
    |  Ω(log k) (Theorem 1)         |  Impossible (Theorem 2)
    ↘                              ↙
        Random Walk (P_RW)
        Impossible (Theorem 2)
```

---

### 2-6. 성능 향상 및 한계

#### 성능 향상 (실험 결과, Table 1)

| 태스크 유형 | 최적 토크나이제이션 | 이유 |
|------------|------------------|------|
| MaxClq., TopoOrd. (로컬 제약) | Adjacency (Padded) | 정확한 엣지 제약 직접 노출 |
| Tox21 (글로벌 구조) | Full Laplacian | 글로벌 그래프 기하 포착 |
| HIV, MaxClq., TopoOrd. | **Combined** (최고 성능) | 다중 구조적 시각 활용 |

**핵심 발견**: 단일 토크나이제이션이 모든 태스크에서 지배적이지 않으며, 보완적 토크나이제이션 결합이 성능을 개선합니다.

#### 합성 실험 검증

**Figure 2a** (연결성 분류): $\mathcal{P}\_{\text{Lap}}$은 $L=1$에서 모든 그래프 크기( $n \in \{16,32,...,256\}$ )에서 완벽 해결. $\mathcal{P}_{\text{Adj}}$는 $n \geq 128$에서 랜덤 성능으로 붕괴.

**Figure 2b** (RW 귀환 확률 예측): Adjacency 토크나이제이션의 정규화 오류가 $k=2$에서 $k=8$로 증가할 때 거의 두 배 증가 (깊이 격차 3% → 14%), $\Omega(\log k)$ 하한과 일치.

#### 한계

1. **노드 수준 토크나이제이션에 한정**: 엣지 수준 토크나이제이션은 미포함
2. **복잡도 이론적 가정 의존**: $\text{TC}^0 \subsetneq \text{NC}^1$, $\text{TC}^0 \subsetneq \text{L}$ 등 미증명 가정
3. **비선형 트랜스포머의 변환 가능성**: 선형 어텐션/MLP에서는 정확한 하한이 증명되나, 비선형 트랜스포머가 정밀도나 너비 비용으로 이 병목을 우회할 수 있는지 미해결
4. **엣지 특성 미활용**: 실험 모델이 엣지 특성을 사용하지 않아 엣지 특성 활용 모델 대비 성능 제한

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 토크나이제이션이 일반화에 미치는 영향

이 논문은 일반화 성능과 관련하여 여러 중요한 통찰을 제공합니다:

#### (a) 표현력과 일반화의 트레이드오프

- **Lossless 토크나이제이션** (Full Adjacency, Full Laplacian)은 표현력이 높지만, 파라미터 노름 요구사항이 증가하여 **최적화 어려움과 과적합 위험**이 높아집니다.

- Theorem 5에서 보듯, Laplacian 토크나이제이션으로 로컬 엣지를 예측하려면:

$$\text{Lip}_{\text{MLP}} \cdot \|W_V\|_2 (1 + \gamma) \geq d_{\max}$$

파라미터 노름이 $d_{\max}$에 선형 스케일링되어, 고차수 노드가 많은 그래프에서 **일반화 성능이 저하**됩니다.

#### (b) ill-conditioning의 일반화 영향

Laplacian 토크나이제이션에서 로컬 태스크를 위한 ill-conditioning은 **최적화 경로를 불안정**하게 만들어:
- Softmax 포화 → **그래디언트 소실** → 학습 부족 → 일반화 실패
- 또는 가중치 폭발 → **과적합** → 테스트 성능 저하

#### (c) Random-walk 토크나이제이션의 일반화 한계

Random-walk 토크나이제이션은 Theorem 2에 의해 **평면성과 같은 전역 위상적 특성을 결코 학습할 수 없습니다**. 이는 훈련 데이터가 아무리 많아도 극복할 수 없는 **구조적 일반화 실패**입니다.

$$\left[(D^{-1}A')^m\right]_{ii} = \left[(D^{-1}A)^m\right]_{ii} \quad \forall i, m$$

두 위상적으로 다른 그래프 $G$, $G'$가 동일한 토큰을 생성하므로, 어떤 모델도 이를 구분하여 일반화할 수 없습니다.

### 3-2. 일반화 성능 향상을 위한 핵심 제안: 보완적 토크나이제이션 결합

논문의 핵심 실용적 발견은 **보완적 토크나이제이션의 결합이 일반화 성능을 향상**시킨다는 것입니다:

$$\mathcal{P}_{\text{Comb}}(G) = \left[\mathcal{P}_{\text{Lap}}(G) \| \mathcal{P}_{\text{RW}}(G) \| \mathcal{P}_{\text{Adj}}(G)\right] \in \mathbb{R}^{n \times (3n-1)}$$

**이론적 근거**: 각 토크나이제이션이 서로 다른 구조적 정보를 노출하므로:
- Adjacency: 로컬 연결성 → 엣지 예측, 클리크 탐지에 유리
- Laplacian: 전역 위상 → 연결성, 분자 특성 예측에 유리
- Random-walk: 확산 통계 → 워크 기반 태스크에 유리

결합 모델은 이 세 가지 구조적 신호를 모두 활용하여 **다양한 태스크에서 더 강한 일반화**를 달성합니다.

### 3-3. 일반화와 깊이의 관계

깊이 분리 결과는 일반화에 중요한 함의를 가집니다:

- **부적절한 토크나이제이션 + 충분하지 않은 깊이** = 해당 태스크에서 체계적 일반화 실패
- **적절한 토크나이제이션** = 얕은 모델도 강한 일반화 가능

예시: Laplacian 토크나이제이션을 사용하면 $L=1$로 연결성 태스크에서 완벽한 일반화 가능. Adjacency 토크나이제이션은 $L=7$로도 $n \geq 128$에서 일반화 실패.

이는 **모델 깊이만 증가시키는 스케일링 전략이 잘못된 토크나이제이션을 선택하면 무효**임을 의미합니다.

### 3-4. 절삭(Truncation)과 일반화

절삭은 특히 **분포 외(Out-of-Distribution) 일반화**에 심각한 위협:

- 훈련 그래프에 없던 새로운 위상 구조가 테스트에 나타날 때, 절삭된 Laplacian은 해당 고유값을 놓칠 가능성이 있음
- 따라서 **절삭 차원의 선택이 일반화 성능에 직접적 영향**

Theorem 3에 따르면, Laplacian에서 단 하나의 고유값 누락으로도 삼각형 계산이 불가능해집니다. 이는 **도메인 외 일반화에서 절삭 토크나이제이션의 취약성**을 시사합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4-1. 향후 연구에 미치는 영향

#### (a) 그래프 트랜스포머 설계 패러다임 전환

이 논문은 그래프 트랜스포머 연구에서 **"어떤 아키텍처를 쓸 것인가"에서 "어떤 토크나이제이션을 쓸 것인가"로 연구 초점을 전환**시킵니다. 향후:

- 태스크 특성 분석 → 토크나이제이션 선택 → 아키텍처 설계의 파이프라인이 표준화될 것
- 자동화된 토크나이제이션 선택(AutoML 관점)에 대한 연구 촉진

#### (b) 이론적 프레임워크 확장

- **회로 복잡도와 그래프 학습의 연결**: $\text{TC}^0$, $\text{NC}^1$, $\text{L}$ 복잡도 클래스를 그래프 태스크 분류에 활용하는 방향으로 확장
- **엣지 수준 토크나이제이션**: 논문이 노드 수준에 한정되므로, 분자 및 관계형 도메인의 엣지 수준 분석이 후속 연구 과제
- **위상적 데이터 분석(TDA)과의 연결**: Persistent homology 기반 토크나이제이션과의 비교 연구

#### (c) 실용적 엔지니어링 가이드라인

표 형태의 명확한 가이드라인 제공:

| 태스크 유형 | 권장 토크나이제이션 |
|------------|------------------|
| 로컬 구조 (엣지 예측, 클리크) | Adjacency |
| 전역 구조 (연결성, 분자 특성) | Laplacian |
| 확산 기반 태스크 | Random-walk |
| 태스크 불명확 또는 복합적 | Combined |

#### (d) 대규모 언어 모델의 그래프 이해

Fatemi et al. (2023)의 "Talk like a graph" 방향과 연계하여, LLM이 그래프를 처리할 때 어떤 텍스트 기반 표현(=토크나이제이션의 일종)을 선택하느냐가 이해 능력을 결정한다는 함의를 제공합니다.

---

### 4-2. 향후 연구 시 고려할 점

#### (a) 복잡도 이론적 가정의 조건부 성격

모든 하한은 **$\text{TC}^0 \subsetneq \text{NC}^1$, $\text{TC}^0 \subsetneq \text{L}$** 등 미증명 가정에 의존합니다. 이 가정들이 성립하지 않는 엣지 케이스에 대한 별도 분석이 필요합니다.

#### (b) 비선형 트랜스포머의 변환 능력

논문은 선형 어텐션/MLP에서의 정확한 하한을 제공하지만, **비선형 트랜스포머가 충분한 너비나 정밀도로 이 병목을 우회할 수 있는지**는 미해결 문제입니다. 향후 연구:

- 비선형 활성화 함수의 역할 분석
- 수치 정밀도(precision)와 표현력의 트레이드오프 정량화

#### (c) 동적/헤테로지니어스 그래프로의 확장

현재 논문은 정적 동질 그래프(static homogeneous graphs)에 집중. **시간에 따라 변하는 동적 그래프**나 **이종 그래프(heterogeneous graphs)**에서의 토크나이제이션 트레이드오프는 별도 분석 필요.

#### (d) 최적화 관점의 토크나이제이션 연구

Theorem 5는 Laplacian 토크나이제이션이 **표현력은 충분하지만 최적화하기 어렵다**는 것을 보입니다. 이는 **"표현력이 충분한 토크나이제이션이라도 훈련 가능성(trainability)이 다를 수 있다"**는 새로운 연구 방향을 열어줍니다:

- 토크나이제이션별 손실 경관(loss landscape) 분석
- 정규화 기법이 ill-conditioning을 완화할 수 있는지 연구
- 적응형 고유값 정규화(adaptive eigenvalue normalization) 개발

#### (e) 태스크 인식 자동 토크나이제이션 선택

현재는 인간이 태스크 특성을 분석하여 토크나이제이션을 선택합니다. 향후:

- **Meta-learning** 기반 자동 토크나이제이션 선택
- **학습 가능한 토크나이제이션** (learnable tokenization) - 현재 비학습 맵으로 정의된 것을 학습 가능하게 확장
- **다중 토크나이제이션의 동적 가중 결합** (attention over tokenizations)

#### (f) 순열 동변환 문제

Adjacency 토크나이제이션은 순열 동변환이 아닙니다. 순열 불변/동변환 모델이 요구되는 태스크에서 Adjacency 토크나이제이션의 정규화 방법 연구가 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 주요 내용 | 본 논문과의 관계 |
|------|---------|----------------|
| **Graphormer** (Ying et al., NeurIPS 2021) | 노드 중심성, 공간 인코딩을 어텐션 바이어스로 주입 | 고정된 구조 인코딩 사용; 토크나이제이션 선택의 이론적 분석 부재 |
| **SAN** (Kreuzer et al., NeurIPS 2021) | 스펙트럼 어텐션으로 LPE(Laplacian PE) 학습 | Laplacian 토크나이제이션 활용; 본 논문이 이의 ill-conditioning을 이론적으로 분석 |
| **GraphGPS** (Rampášek et al., NeurIPS 2022) | MPNN + Transformer 혼합; RWPE, LapPE 사용 | Random-walk와 Laplacian PE 혼합; 본 논문이 두 토크나이제이션의 보완성을 이론적으로 정당화 |
| **GRIT** (Ma et al., ICML 2023) | 메시지 패싱 없이 랜덤 워크 유도 귀납적 바이어스 | Random-walk 특성의 효용성 활용; 본 논문이 이의 이론적 한계(lossy, planarity 불결정) 규명 |
| **SignNet/BasisNet** (Lim et al., ICLR 2023) | 스펙트럼의 부호/기저 불변성 학습 | Laplacian 토크나이제이션의 대칭성 문제 해결 시도; 본 논문의 ill-conditioning과 연관 |
| **Sanford et al.** (NeurIPS 2023, 2024a,b) | 트랜스포머의 깊이-너비 트레이드오프, 그래프 알고리즘 추론 | 본 논문의 깊이 하한 분석의 직접적 기반; 본 논문은 여기에 토크나이제이션 축을 추가 |
| **Yehudai et al.** (2025, 2026) | 그래프 태스크에서 트랜스포머의 깊이-너비 트레이드오프 | 본 논문 저자들의 직접적 선행 연구; Adjacency 토크나이제이션의 경쟁력 실증 |
| **GraphBench** (Stoll et al., 2026) | 차세대 그래프 학습 벤치마킹 | 본 논문의 실험 평가에 사용된 데이터셋 제공 |
| **Graph-BERT** (Zhang et al., 2020) | 어텐션만으로 그래프 표현 학습 | 토크나이제이션 선택의 영향을 이론적으로 다루지 않음; 본 논문이 이 공백 보완 |

---

## 참고 자료

- **주 논문**: Bechler-Speicher, M., Yehudai, G., Harari, G., Sanford, C., Globerson, A., & Bruna, J. (2026). "Lost in Tokenization: Fundamental Trade-offs in Graph Tokenization for Transformers." arXiv:2605.22471v1.

- **인용 문헌** (논문 내 참조):
  - Rampášek et al. (2022). "Recipe for a general, powerful, scalable graph transformer." NeurIPS 2022.
  - Ying et al. (2021). "Do transformers really perform badly for graph representation?" NeurIPS 2021.
  - Kreuzer et al. (2021). "Rethinking graph transformers with spectral attention." NeurIPS 2021.
  - Ma et al. (2023). "Graph inductive biases in transformers without message passing." ICML 2023.
  - Lim et al. (2023). "Sign and basis invariant networks for spectral graph representation learning." ICLR 2023.
  - Sanford et al. (2023). "Representational strengths and limitations of transformers." NeurIPS 2023.
  - Sanford et al. (2024a). "Transformers, parallel computation, and logarithmic depth." arXiv:2402.09268.
  - Sanford et al. (2024b). "Understanding transformer reasoning capabilities via graph algorithms." arXiv:2405.18512.
  - Yehudai et al. (2025). "Depth vs width tradeoffs in graph transformers." arXiv:2503.01805.
  - Yehudai et al. (2026). "Depth-width tradeoffs for transformers on graph tasks." NeurIPS 2026.
  - Merrill et al. (2022). "Saturated transformers are constant-depth threshold circuits." TACL.
  - Merrill & Sabharwal (2023). "The parallelism tradeoff: Limitations of log-precision transformers." TACL.
  - Godsil & McKay (1982). "Constructing cospectral graphs." Aequationes Mathematicae.
  - Reingold (2008). "Undirected connectivity in log-space." JACM.
  - Hu et al. (2020). "Open graph benchmark." NeurIPS 2020.
  - Stoll et al. (2026). "GraphBench: Next-generation graph learning benchmarking." arXiv:2512.04475.
  - Xu et al. (2019). "How powerful are graph neural networks?" ICLR 2019.
  - Zaheer et al. (2017). "Deep sets." NeurIPS 2017.
  - Fatemi et al. (2023). "Talk like a graph: Encoding graphs for large language models." arXiv:2310.04560.
  - Vollmer (1999). "Introduction to Circuit Complexity: A Uniform Approach." Springer.
  - Brouwer & Haemers (2012). "Spectra of Graphs." Springer.
