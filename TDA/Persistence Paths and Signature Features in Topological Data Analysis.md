# Persistence Paths and Signature Features in Topological Data Analysis

**저자:** Ilya Chevyrev, Vidit Nanda, Harald Oberhauser (University of Oxford)
**출판:** arXiv:1806.00381v2, 2018년 12월

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
Persistent homology에서 생성되는 **바코드(barcode)**를 통계적 학습에 적합한 형태로 변환하기 위해, 바코드를 먼저 벡터 공간의 **경로(path)**로 변환한 후, 해당 경로의 **path signature**를 계산하여 텐서 대수(tensor algebra)에 값을 갖는 **feature map**을 구성하는 2단계 방법론을 제안한다.

### 주요 기여
1. **Persistence path embedding** $\iota_\bullet : \mathbf{Bar} \to \mathbf{BV}(V)$의 체계적 제안 (landscape, envelope, Betti, Euler, naive 등 다양한 임베딩)
2. Path signature $\mathrm{S}$와 결합한 feature map $\Phi_\bullet = \mathrm{S} \circ \iota_\bullet$의 **보편성(universality)**, **특성성(characteristicness)**, **안정성(stability)** 이론적 증명
3. 커널화(kernelized) 및 비커널화(unkernelized) 버전 모두 제공하여 다양한 학습기 활용 가능
4. 세 가지 분류 벤치마크(Orbits, Textures, Shapes)에서 **state-of-the-art 성능** 달성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

Persistent homology의 출력인 바코드 공간 $\mathbf{Bar}$는 **비선형 메트릭 공간**이다. 대부분의 확장 가능한 학습 알고리즘은 선형 방법에 의존하므로, 바코드를 직접 통계적 학습에 사용하기 어렵다. 구체적으로:

- 바코드는 구간(interval)들의 **다중집합(multiset)**으로, 고정 차원의 벡터 공간에 자연스럽게 놓이지 않음
- 기존 feature map(persistence landscape, persistence image 등)은 보편성이나 특성성에 대한 이론적 보장이 불충분하거나, 효율적 계산이 어려움
- 안정성(stability), 판별력(discriminative power), 계산 가능성(computability) 세 축을 동시에 최적화하는 단일 feature map의 부재

### 2.2 제안하는 방법

#### 전체 구조: 2단계 파이프라인

$$\mathbf{Bar} \xrightarrow{\iota_\bullet} \mathbf{BV}(V) \xrightarrow{\mathrm{S}} \mathbf{T}(V)$$

여기서:
- $\mathbf{Bar}$: 바코드 공간 (유한 구간들의 다중집합)
- $\mathbf{BV}(V)$: 벡터 공간 $V$ 위의 유계 변동(bounded variation) 경로 공간
- $\mathbf{T}(V) = \prod_{m \geq 0} V^{\otimes m}$: $V$의 텐서 대수

최종 feature map: $\Phi_\bullet = \mathrm{S} \circ \iota_\bullet : \mathbf{Bar} \to \mathbf{T}(V)$

#### Step 1: Persistence Path Embedding $\iota_\bullet$

**① 적분 랜드스케이프 임베딩 (Integrated Landscape Embedding) $\iota_{\text{iL}}$:**

바코드 $B$의 persistence landscape $\Lambda^B(k,t) = \lambda_k^B(t)$를 정의한 후 적분:

$$[\iota_{\text{iL}}(B)]_k(t) = \int_{-\infty}^{t} \lambda_k(s) \, \mathrm{d}s$$

여기서 landscape 함수는:

$$\Lambda^B(k,t) = \sup\{s \geq 0 \mid \beta^{t-s,t+s} \geq k\}$$

$\beta^{t-s,t+s}$는 구간 $[t-s, t+s]$를 포함하는 바코드 구간의 수이다.

**안정성 (Theorem 3):** 합성 사상

$$\mathbf{Met} \xrightarrow{\mathrm{PH}_i} \mathbf{Bar} \xrightarrow{\Lambda^\bullet} L^\infty(\mathbb{N} \times \mathbb{R}) \xrightarrow{\mathcal{I}} \mathbf{BV}(\ell^\infty)$$

는 모든 $i \geq 0$에 대해 **1-Lipschitz**이다. 여기서 $\mathbf{Met}$에는 Gromov-Hausdorff 거리, $\mathbf{BV}(\ell^\infty)$에는 1-Hölder 노름이 부여된다.

**② 포락선 임베딩 (Envelope Embedding) $\iota_{\mathrm{E}}$:**

바코드의 구간 $\{[b_i, d_i)\}_{i=1}^m$을 길이 내림차순으로 정렬한 후 상·하 포락선을 구성:

```math
[\iota_{\mathrm{E}}(B)](t) = (\ell_B(t), u_B(t)) \in \mathbb{R}^2
```

$u_B$: 상위 포락선 (사망 시간 $d_i$의 선형 보간), $\ell_B$: 하위 포락선 (탄생 시간 $b_i$의 선형 보간). 항상 **2차원** 경로를 생성하여 계산 효율이 높다.

**③ 베티 임베딩 (Betti Embedding) $\iota_\beta$:**

$$\iota_\beta(B)(t_j) = (\beta_0^{t_j}, \beta_1^{t_j}, \ldots, \beta_{n-1}^{t_j})$$

여기서 $\beta_i^{t_j} = \dim H_i(\mathrm{K}(t_j); \mathbb{F})$는 스케일 $t_j$에서의 $i$차 베티 수이다.

**④ 오일러 임베딩 (Euler Embedding) $\iota_\chi$:**

$$\iota_\chi(B)(t_j) = \sum_{i=0}^{n} (-1)^i \beta_i^{t_j}$$

호몰로지 계산 없이 단체(simplex) 수의 교대합으로 직접 계산 가능하다.

#### Step 2: Path Signature $\mathrm{S}$

**정의 (Definition 4.1):**

$$\mathrm{S} : \mathbf{BV}(V) \to \mathbf{T}(V), \quad x \mapsto (\mathrm{S}_0(x), \mathrm{S}_1(x), \ldots)$$

여기서 $\mathrm{S}_0(x) = 1$이고:

$$\mathrm{S}_m(x) := \int_{0 < t_1 < \cdots < t_m < T} \mathrm{d}x(t_1) \otimes \mathrm{d}x(t_2) \otimes \cdots \otimes \mathrm{d}x(t_m) \in V^{\otimes m}$$

$V = \mathbb{R}^n$인 경우:

$$\mathrm{S}^{i_1, \ldots, i_m}(x) = \int_{0 < t_1 < \cdots < t_m < T} \mathrm{d}x^{i_1}(t_1) \, \mathrm{d}x^{i_2}(t_2) \cdots \mathrm{d}x^{i_m}(t_m)$$

$m$차 항 $\mathrm{S}_m(x)$는 $n^m$개의 실수로 해석된다.

**셔플 항등식 (Shuffle Identity, Lemma 4.4):** $\ell_i \in (V')^{\otimes m_i}$에 대해:

$$\langle \mathrm{S}(x), \ell_1 \rangle \langle \mathrm{S}(x), \ell_2 \rangle = \langle \mathrm{S}(x), \ell_3 \rangle$$

여기서 $\ell_3 \in (V')^{\otimes(m_1 + m_2)}$는 $\ell_1$과 $\ell_2$의 shuffle product이다. 이는 signature의 선형 함수가 **곱셈에 닫혀 있음**을 의미한다.

**단사성 (Theorem 4):** $\mathrm{S}(x) = \mathrm{S}(y)$ $\iff$ $x$와 $y$가 tree-like equivalent.

#### 커널화

$V$가 힐베르트 공간일 때, 바코드 커널:

$$k_\bullet(B, B') := \langle \Phi_\bullet(B), \Phi_\bullet(B') \rangle$$

Király & Oberhauser (2016)의 알고리즘으로 수준- $M$ 근사를 $O(l^2 c^{2M})$ 시간, $O(l^2)$ 메모리로 계산 가능 ($l$: 경로의 시간 점 수, $c$: $V$에서의 내적 비용).

#### 하이퍼파라미터 $\pi = \{M, \tau, \Delta, \varphi\}$

- $M \in \mathbb{N}$: signature 절단 수준
- $\tau \in \{0, 1\}$: 시간 증강 (경로를 $(t, x(t))$로 확장)
- $\Delta \in V^l$: lag 벡터 ($x(t)$를 $(x(t), x(\max(t-\Delta_1, 0)), \ldots)$로 확장)
- $\varphi : V \to W$: 비선형 변환 (예: RBF 커널의 feature map)

### 2.3 이론적 성질 (Theorem 5)

$\Phi : \mathbf{Bar}/\iota \to \mathbf{T}(V)$, $B \mapsto \mathrm{S} \circ \iota(B)$에 대해, 컴팩트 부분집합 $K \subset \mathbf{Bar}/\iota$에서:

**(1) 보편성 (Universal):** 임의의 연속함수 $f : K \to \mathbb{R}$과 $\epsilon > 0$에 대해, $\ell \in \bigoplus_{m \geq 0} (V')^{\otimes m}$가 존재하여:

$$\sup_{B \in K} |f(B) - \langle \Phi(B), \ell \rangle| < \epsilon$$

**(2) 특성성 (Characteristic):** Borel 확률 측도의 집합 $\mathcal{M}$에 대해:

$$\mathcal{M} \to \mathbf{T}(V), \quad \mu \mapsto \mathbb{E}_{B \sim \mu}[\Phi(B)]$$

는 **단사(injective)**이다.

**(3) 커널화:** $V$가 힐베르트 공간이면, $k(B,B') = \langle \Phi(B), \Phi(B') \rangle$는 유계·연속이며 $C(K, \mathbb{R})$에 대해 보편적이고 Borel 확률 측도에 대해 특성적이다.

증명은 Stone–Weierstrass 정리와 signature의 연속성·점 분리 성질에 기반한다.

### 2.4 실험 성능

| Method | Textures | Orbits | Shapes |
|--------|----------|--------|--------|
| $k_{\text{SW}}$ | $96.8 \pm 1.0$ | $94.6 \pm 1.3$ | $95.8 \pm 1.6$ |
| $\Phi_{\text{PI}}$ | $93.7 \pm 1.0$ | $\mathbf{99.86 \pm 0.21}$ | $90.3 \pm 2.3$ |
| $k_\beta$ | $\mathbf{97.8 \pm 0.2}$ | NA | $93.0 \pm 3.0$ |
| $\Phi_\beta$ | $96.6 \pm 0.6$ | $97.7 \pm 0.8$ | $\mathbf{98.1 \pm 0.7}$ |
| $\Phi_\chi$ | $92.9 \pm 0.7$ | $98.8 \pm 0.6$ | $98.0 \pm 1.1$ |

- **Textures:** $k_\beta$가 $97.8\%$로 최고 (커널화 형태)
- **Shapes:** $\Phi_\beta$가 $98.1\%$로 state-of-the-art (feature 형태)
- **Orbits:** $\Phi_{\text{PI}}$가 $99.86\%$로 최고이나 $\Phi_E$, $\Phi_\chi$도 $98\%$ 이상으로 경쟁적

### 2.5 한계

1. **안정성-판별력-계산가능성의 트릴레마:** 세 축을 동시에 최적화하는 단일 임베딩은 존재하지 않을 가능성이 높음. 안정성은 긴 구간에 민감하고, 판별력은 짧은 구간에도 정보가 있을 수 있으므로 상충.

2. **차원 폭발:** $V = \mathbb{R}^n$에서 수준 $M$까지 절단된 signature는 $O(n^M)$개의 좌표를 가짐. $\iota_{\text{iL}}$처럼 고차원 경로를 생성하는 임베딩은 직접 feature map 계산이 불가능.

3. **컴팩트 집합 제한:** 보편성과 특성성 보장이 **컴팩트 부분집합** $K$에서만 성립. 비컴팩트 경우는 적분가능성 조건이 추가로 필요.

4. **불안정 임베딩:** envelope, Betti, Euler 임베딩은 bottleneck 거리에 대해 안정적이지 않음 (간단한 반례 존재).

5. **벤치마크 범위 제한:** 실험이 세 데이터셋에 한정되며, 대규모 데이터나 고차원 점구름에 대한 확장성은 미검증.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 일반화 보장

**보편성(Universality)**은 일반화 성능의 핵심 이론적 기초이다. Theorem 5(1)에 의해, 컴팩트 집합 $K$ 위의 **임의의 연속 함수**를 $\Phi(B)$의 선형 함수로 균일하게 근사할 수 있다:

$$\sup_{B \in K} |f(B) - \langle \Phi(B), \ell \rangle| < \epsilon$$

이는 feature space $\mathbf{T}(V)$가 **충분히 풍부한 표현력**을 가짐을 의미하며, 적절한 정규화와 결합하면 과적합 없이 복잡한 함수를 학습할 수 있음을 시사한다.

**특성성(Characteristicness)**은 $\mu \mapsto \mathbb{E}_{B \sim \mu}[\Phi(B)]$의 단사성으로, 서로 다른 데이터 분포를 feature space에서 구별할 수 있음을 보장한다. 이는 **two-sample test**, **hypothesis testing** 등 분포 수준의 일반화에 핵심적이다.

### 3.2 하이퍼파라미터를 통한 일반화 제어

1. **절단 수준 $M$:** 낮은 $M$은 저차 통계량만 포착하여 과적합을 방지하고, 높은 $M$은 표현력을 증가시킴. 실험에서 커널 방법은 $M=2\sim3$, feature 방법은 $M=4\sim8$에서 최적 — 가우시안 커널 비선형성이 저차에서 이미 충분한 정보를 포착하기 때문으로 추정.

2. **임베딩 선택의 유연성:** 교차검증으로 최적 $\iota_\bullet$를 선택할 수 있어, 데이터의 위상적 특성에 맞춤형 일반화 가능:
   - 안정성 중시 → $\iota_{\text{iL}}$ (1-Lipschitz)
   - 계산 효율 중시 → $\iota_\chi$ (호몰로지 계산 불필요)
   - 판별력 중시 → $\iota_{\text{E}}$ (2차원 경로)

3. **시간 증강 $\tau$:** $x(t) \to (t, x(t))$로 확장하면 tree-like equivalence를 제거하여 signature가 **완전 단사**가 됨. 이는 정보 손실 없는 일반화를 보장.

4. **비선형 변환 $\varphi$:** RBF 커널 등을 통해 경로를 고차원/무한차원으로 리프팅하면, 낮은 signature 수준에서도 충분한 비선형성을 확보하여 일반화 성능 향상.

### 3.3 커널/비커널 이중 접근의 일반화 이점

- **커널 방법:** RKHS의 정규화 이론(Tikhonov regularization)에 의한 일반화 보장, Nyström 근사로 대규모 데이터 처리 가능
- **비커널 방법:** 랜덤 포레스트, 신경망 등 더 일반적인 분류기 사용 가능. Tikhonov 이외의 정규화(예: $L_1$, dropout)를 통해 과적합 방지

### 3.4 포락선 임베딩의 계층적 일반화

절단 포락선 $\iota_{\mathrm{E}}^N$에서 $N$이 작으면 가장 길고 안정적인 구간만 사용하여 **노이즈에 강건한 일반화**를 달성. $N$이 클 때 성능이 떨어지면, 신호가 안정적 구간에 집중되어 있다는 증거로 해석 가능.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구 영향

1. **TDA와 path analysis의 연결:** 지속적 호몰로지를 경로로 해석하는 관점은 **확률론적 분석(rough path theory)**과 **위상적 데이터 분석**을 연결하는 새로운 다리를 구축.

2. **feature engineering의 체계화:** 바코드에 대한 다양한 feature map 구성을 안정성·판별력·계산가능성의 세 축으로 체계적으로 분류하는 프레임워크 제공.

3. **커널 학습과 TDA의 통합:** signature kernel은 이후 시계열 학습, 그래프 학습 등 다양한 구조적 데이터에 확장되어, TDA를 넘어선 광범위한 응용 가능성을 열었음.

4. **비컴팩트 확장 문제:** 컴팩트 집합에서의 보편성/특성성 보장을 비컴팩트로 확장하는 것은 moment problem과 연결되며, 확률론의 깊은 문제와 접속.

### 4.2 향후 연구 시 고려할 점

1. **확장성(Scalability):** 대규모 점구름($n > 10^4$)에서 바코드 계산과 signature 계산의 병목 해소 필요. Low-rank 알고리즘의 실질적 구현과 벤치마킹이 중요.

2. **다차원 persistence:** 이 논문은 1-parameter persistence에 한정. Multi-parameter persistence로의 확장은 바코드의 구조적 복잡성(quiver representation의 wild type 문제)으로 인해 비자명한 연구 과제.

3. **딥러닝과의 통합:** Signature를 신경망의 입력 layer로 사용하거나, 미분 가능한 persistence path embedding을 end-to-end 학습하는 방법론 개발.

4. **임베딩 자동 선택:** 데이터에 최적인 $\iota_\bullet$를 자동으로 학습하는 meta-learning 또는 neural architecture search 접근법.

5. **비컴팩트 설정의 이론 확장:** 적분가능성 조건하에서의 특성성 보장 (Chevyrev & Lyons, 2016의 결과 활용).

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 Signature Kernel (Kiraly & Oberhauser → Salvi, Cass, Foster, Lyons, 2021)

**논문:** "The Signature Kernel is the solution of a Goursat PDE" (Salvi et al., *SIAM Journal on Mathematics of Data Science*, 2021)

- Chevyrev et al.의 signature 커널 $k(B,B') = \langle \mathrm{S}(x), \mathrm{S}(y) \rangle$의 계산을 **Goursat PDE의 해**로 재정식화하여, 절단(truncation) 없이 **전체 signature 커널**을 효율적으로 계산
- 절단 수준 $M$ 선택 문제 해소 → 일반화 성능 향상
- 이 논문의 커널화 접근의 직접적 발전

### 5.2 PersLay (Carrière, Chazal, Ike, Lacombe, Royer, Umeda, 2020)

**논문:** "PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological Signatures" (*AISTATS*, 2020)

- Persistence diagram을 **신경망 layer**로 변환하는 프레임워크
- 다양한 기존 feature map(persistence image, landscape, Betti curve 등)을 **학습 가능한 매개변수**로 통합
- Chevyrev et al.의 수동적 임베딩 선택($\iota_\bullet$의 교차검증)을 **end-to-end 학습**으로 대체
- **비교:** PersLay는 더 유연하지만, 보편성·특성성 같은 이론적 보장이 부족

### 5.3 Persistence Weighted Gaussian Kernel (Kusano, Fukumizu, Hiraoka, 2016 → 확장 2020+)

- 가중 가우시안 커널과 persistence diagram의 결합
- Chevyrev et al.의 signature 기반 커널과 보완적: signature는 **경로의 순서 정보**를 포착하지만, 가우시안 커널은 **개별 점의 위치 정보**를 강조

### 5.4 PLLay & Topological Layers (Kim, Kim, Memoli, 2020)

**논문:** "PLLay: Efficient Topological Layer based on Persistent Landscapes" (*NeurIPS*, 2020)

- Persistence landscape를 미분 가능한 신경망 층으로 구현
- Chevyrev et al.의 landscape 임베딩 $\iota_{\text{iL}}$와 직접 관련되지만, **역전파(backpropagation)**를 통한 학습이 가능
- **비교:** Chevyrev et al.은 landscape 후 signature를 사용하지만, PLLay는 landscape 자체를 미분 가능하게 만들어 신경망 학습에 통합

### 5.5 Persformer (Reinauer, Rébillat, Gerin, 2022)

**논문:** "Persformer: A Transformer Architecture for Topological Machine Learning" (*arXiv:2112.15210*)

- Transformer 아키텍처를 persistence diagram에 직접 적용
- Self-attention 메커니즘이 바코드 구간 간의 **상호작용(interaction)**을 자동 학습
- **비교:** Chevyrev et al.의 signature가 포착하는 **순서 정보 및 고차 상호작용**을 attention 메커니즘이 대체. 이론적 보장은 약하지만 실용적 성능이 우수한 경향

### 5.6 Topological Transformer (Piekenbrock & Doran, 2023)

- Persistence diagram에 특화된 transformer 구조로, set-structured 입력의 순열 불변성 처리
- Chevyrev et al.이 정렬(sorting)을 통해 해결한 순열 문제를 attention으로 자연스럽게 처리

### 5.7 비교 종합

| 방법 | 이론적 보장 | 학습 가능성 | 계산 효율 | 일반화 |
|------|-----------|-----------|-----------|--------|
| **Chevyrev et al. (2018)** | ✅ 보편성, 특성성, 안정성 | ❌ 임베딩 수동 선택 | △ ($O(n^M)$ 좌표) | ✅ 교차검증으로 제어 |
| **Signature Kernel (2021)** | ✅ PDE 기반 정확 계산 | ❌ 커널만 | ✅ 절단 불필요 | ✅ |
| **PersLay (2020)** | △ 제한적 | ✅ end-to-end | ✅ | ✅ 데이터 적응적 |
| **PLLay (2020)** | △ landscape 한정 | ✅ 미분 가능 | ✅ | ✅ |
| **Persformer (2022)** | ❌ | ✅ Transformer | ✅ | ✅ attention 기반 |

---

## 참고자료

1. **Chevyrev, I., Nanda, V., & Oberhauser, H.** "Persistence paths and signature features in topological data analysis." arXiv:1806.00381v2, 2018. *(본 논문)*

2. **Bubenik, P.** "Statistical topological data analysis using persistence landscapes." *Journal of Machine Learning Research*, 16:77–102, 2015.

3. **Carrière, M., Cuturi, M., & Oudot, S.** "Sliced Wasserstein kernel for persistence diagrams." *ICML*, 2017.

4. **Adams, H. et al.** "Persistence images: A stable vector representation of persistent homology." *JMLR*, 18(1):218–252, 2017.

5. **Király, F. J. & Oberhauser, H.** "Kernels for sequentially ordered data." arXiv:1601.08169, 2016.

6. **Boedihardjo, H., Geng, X., Lyons, T., & Yang, D.** "The signature of a rough path: uniqueness." *Advances in Mathematics*, 293:720–737, 2016.

7. **Salvi, C., Cass, T., Foster, J., Lyons, T., & Yang, W.** "The Signature Kernel is the solution of a Goursat PDE." *SIAM Journal on Mathematics of Data Science*, 3(3):873–899, 2021.

8. **Carrière, M., Chazal, F., Ike, Y., Lacombe, T., Royer, M., & Umeda, Y.** "PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological Signatures." *AISTATS*, 2020.

9. **Kim, K., Kim, J., & Memoli, F.** "PLLay: Efficient Topological Layer based on Persistent Landscapes." *NeurIPS*, 2020.

10. **Reinauer, R., Rébillat, M., & Gerin, L.** "Persformer: A Transformer Architecture for Topological Machine Learning." arXiv:2112.15210, 2022.

11. **Chazal, F., de Silva, V., Glisse, M., & Oudot, S.** *Structure and stability of persistence modules.* Springer, 2016.

12. **Chevyrev, I. & Oberhauser, H.** "Signature moments to characterize laws of stochastic processes." arXiv:1810.10971, 2018.

13. **Simon-Gabriel, C.-J. & Schölkopf, B.** "Kernel distribution embeddings: Universal kernels, characteristic kernels and kernel metrics on distributions." *JMLR*, 19(44):1–29, 2018.

14. **Cohen-Steiner, D., Edelsbrunner, H., & Harer, J.** "Stability of persistence diagrams." *Discrete and Computational Geometry*, 37(1):103–120, 2007.

15. **Lyons, T. J., Caruana, M., & Lévy, T.** *Differential equations driven by rough paths.* Springer, 2007.
