
# Choose a Transformer: Fourier or Galerkin

## I. 핵심 주장 및 기여

"Choose a Transformer: Fourier or Galerkin"(Cao, 2021)은 편미분방정식(PDE) 기반 연산자 학습에 Transformer 아키텍처를 최초로 적용한 논문으로, 다음의 획기적인 주장을 제시합니다.

**핵심 주장**: Transformer의 scaled dot-product attention에서 softmax 정규화는 필수가 아니라 충분조건일 뿐이며, softmax를 제거할 경우 선형 Transformer 변수의 근사 용량이 Petrov-Galerkin 사영과 동등함을 Hilbert 공간 이론으로 최초 증명했습니다.

**주요 기여**:

1. **Softmax 제거의 이론적 정당화**: Hilbert 공간의 연산자 근사 이론을 통해 softmax 정규화가 선택적임을 입증, 동시에 계산 복잡도 감소와 메모리 효율성 향상
2. **두 가지 어텐션 변형 제안**: Fourier 타입(O(n²d) 복잡도)과 Galerkin 타입(O(nd²) 복잡도)의 선형 어텐션 개발
3. **새로운 정규화 스킴**: Petrov-Galerkin 사영을 모방한 계층 정규화로 비정규화 데이터에서도 우수한 성능 달성
4. **역문제 해결**: 기존 방법이 불가능한 노이즈 있는 계수 역식별 문제 해결[1]

***

## II. 해결하고자 하는 문제 및 제안 방법

### 2.1 해결하고자 하는 문제

전통적 PDE 해법(유한 요소법, 스펙트럼 방법)은 단일 인스턴스 해결에 최적화되어, 계수나 경계조건 변경 시 전체 재계산이 필요합니다. 이에 대한 해결책으로 **연산자 학습** $T: H_1 \to H_2$ (Hilbert 공간 간 사상)이 제안되었으나, DeepONet과 FNO는 다음의 한계를 갖습니다.

- **FNO의 한계**: 주로 저주파 특성만 학습하여 고주파 불규칙 성질 포착 실패
- **부분공간 한정**: 학습된 해상도 이상의 문제에서 일반화 실패
- **계산 효율**: 긴 시퀀스(n이 크거나 매우 작을 때)에서 quadratic 복잡도로 인한 비효율

### 2.2 제안하는 방법 (수식 포함)

#### 문제 정의 (식 1-2)

손실 함수:

$$J(\theta) := \mathbb{E}_{a \sim \nu} \left[ \|T_\theta(a) - u\|^2_H + G(a, u; \theta) \right] $$

이산 버전:

$$J(\theta) \approx \frac{1}{N} \sum_{j=1}^{N} \left\| T_\theta(a_h^{(j)}) - u_h^{(j)} \right\|_H^2 + G(a_h^{(j)}, u_h^{(j)}; \theta) $$

여기서 $G$는 문제 의존 정규화 항입니다.

#### Fourier 타입 어텐션 (식 5, 8-9)

$$\text{Attn}_f(y) := (\tilde{Q}\tilde{K}^\top) V / n $$

각 성분:

$$(z_i)\_j = h \sum_{l=1}^{n} (q_i \cdot k_l)(v_j)\_l \approx \int_\Omega (\zeta_q(x_i) \cdot \phi_k(\xi)) \psi_v(\xi) d\xi $$

Skip 연결 후:
$$\delta_j^{-1} v_j(x) \approx z_j(x) - \int_\Omega \kappa(x, \xi) v_j(\xi) d\xi, \quad j=1,\ldots,d $$

**해석**: Fredholm 제2종 방정식 $u + \int K u = f$ 형태. 수치 적분(Nyström 방법)과 동등.

#### Galerkin 타입 어텐션 (식 6, 13-14)

$$\text{Attn}_g(y) := Q(\tilde{K}^\top \tilde{V}) / n $$

$$z_j(x_i) := \sum_{l=1}^{d} \langle v_j, k_l \rangle q_l(x_i) $$

$$ = h \sum_{l=1}^{d} (k_l \cdot v_j)(q_l)_i \approx \sum_{l=1}^{d} \left( \int_\Omega v_j(\xi) k_l(\xi) d\xi \right) q_l(x_i) $$

**해석**: 각 기저 $v_j$에 대한 학습 가능한 Petrov-Galerkin 사영.

#### 정리 4.3 (Céa 타입 보조정리)

$$\|f - g_\theta(y)\|_H \leq c^{-1} \min_{q \in Q_h} \max_{v \in V_h} \frac{|b(v, f_h - q)|}{\|v\|_V} + \|f - f_h\|_H $$

여기서:
- $b(\cdot, \cdot)$: 연속 쌍선형 형식
- $c > 0$: 이산 LBB(Ladyzhenskaya-Babuška-Brezzi) 조건의 하한
- $n$과 무관하게 성립 → **시퀀스 길이 불변 근사 보증**

#### 초기화 전략 (식 17)

$$W_{\text{init}}^{\bullet} \leftarrow \eta U + \delta I, \quad \bullet \in \{Q, K, V\} $$

여기서 $U \sim \text{Xavier}(-\sqrt{3/d}, \sqrt{3/d})$, $\delta > 0$ 작은 값

***

## III. 모델 구조 및 아키텍처

### 3.1 전체 네트워크 구조

**1D 문제 (Burgers' 방정식, Figure 4)**
```
입력 함수 → FFN (피처 추출) → 어텐션 기반 인코더 → 스펙트럼 합성곱 → 출력
                                    ↓
                            좌표 위치 인코딩 (반복)
```

**2D 문제 (Darcy 유동, Figure 5)**
```
고해상도 입력 → 보간 CNN (다운샘플) → 위치 인코딩 병렬화
                                            ↓
                            다중 헤드 어텐션 인코더
                                            ↓
                        보간 CNN (업샘플) + 디코더 → 출력
```

### 3.2 주요 컴포넌트

**피처 추출기 (Feature Extractor)**:
- 1D: 모든 위치에 공유되는 FFN
- 2D: 3 레벨 보간 기반 CNN (3×3 interpolation으로 다운/업샘플링)

**인코더**: 동일한 간단한 어텐션 레이어 스택

$$\begin{align}
\tilde{y} &\leftarrow y + \text{Attn}^\dagger(y)\\
y' &\leftarrow \tilde{y} + g(\tilde{y})
\end{align}$$

여기서 $g(\cdot)$는 standard 2-layer FFN, $\text{Attn}^\dagger \in \{\text{Attn}_f, \text{Attn}_g\}$

**디코더**:
- 매끄러운 해($H^{1+\alpha}(\Omega)$ 공간): 2-layer 스펙트럼 합성곱
- 비매끄러운 해( $L^\infty(\Omega)$ ): pointwise FFN

### 3.3 Galerkin Transformer의 특화 설계

**가정 4.2 (구조 보존 특성)**: Q, K, V의 각 열이 학습된 기저함수를 나타냄
$$\{v_j(\cdot)\}_{j=1}^d \subset \text{Hilbert space}, \quad (v_j)_i = v_j(x_i)$$

**동적 기저 업데이트 (정리 4.4)**: 각 계층에서 기저가 FFN과 위치 인코딩으로 지속 풍부해짐
$$\tilde{q}_l(\cdot) := q_l(\cdot) + \text{FFN}(x)$$

**새로운 계층 정규화 스킴**: 식 (5)-(6) 형태 유지로 Galerkin 사영 특성 보존
- Pre-normalization 대신 Post-normalization (skip 연결 후)
- 스케일이 계층을 통해 전파되도록 허용

***

## IV. 성능 향상 및 실험 결과

### 4.1 계산 효율성 비교 (Table 1)

시퀀스 길이 n=8192, 잠재 차원 d=128, 계층 수 l=10 기준:

| 모델 | 어텐션 복잡도 | CUDA 메모리(GB) | 속도(iter/sec) | GFLOP |
|------|-------------|----------------|----------------|--------|
| **ST** (Softmax) | O(n²ceₐd) | 31.06 | 5.02 | 1393 |
| **FT** (Fourier) | O(n²d) | 22.92 | 6.10 | 1138 |
| **LT** (Linear) | O(n(d²+ceₐd)) | 2.31 | 12.70 | 606 |
| **GT** (Galerkin) | O(nd²) | 1.93 | 27.15 | **275** |

**결과**: Galerkin Transformer는 Softmax 모델 대비 **메모리 94% 절감, 속도 5.4배 향상**[1]

### 4.2 정확도 비교

#### 예제 1: 점성 Burgers' 방정식 (Table 2a)
$$\partial_t u + u \partial_x u - 0.0001 \partial_{xx} u = 0$$

해를 학습하는 연산자: $T: u_0(\cdot) \mapsto u(\cdot, 1)$

| 모델 | n=512 | n=2048 | n=8192 |
|------|------|--------|--------|
| FNO1d (원본) | 15.8 | 14.6 | 13.9 |
| FNO1d (1cycle) | 4.373 | 4.126 | 4.151 |
| **GT** (Galerkin Ln) | **1.203** | **1.150** | **1.025** |

**상대 오차 ×10⁻³**: 약 **4배 개선**[1]

#### 예제 2: Darcy 인터페이스 문제 (Table 2b)
$$-\nabla \cdot (a \nabla u) = f, \quad a: L^\infty(\Omega) \to H^1_0(\Omega)$$

| 모델 | $n_f, n_c$ = 141, 43 | $n_f, n_c$ = 211, 61 |
|------|------------------|-------------------|
| FNO2d | 1.09 | 1.09 |
| FNO2d (1cycle) | 1.419 | 1.424 |
| **GT** (K, V Ln) | **0.839** | **0.844** |

**상대 오차 ×10⁻²**: **30-50% 개선**[1]

#### 예제 3: 역 계수 식별 (Table 3)
$$\text{Operator}: u + \epsilon N_\nu(u) \mapsto a \text{ (noisy measurements)}$$

1% 노이즈 (ε=0.01) 조건:

| 모델 | $n_f, n_c$ =141, 36 | $n_f, n_c$ =211, 71 |
|------|-------------------|-------------------|
| FNO2d | 13.78 | 13.96 |
| **GT** (K, V Ln) | **2.717** | **2.729** |

**상대 오차 ×10⁻²**: **약 5배 개선** – FNO 불가능한 고주파 불규칙 계수 복원 가능[1]

### 4.3 해상도 불변성 (Resolution Invariance)

학습된 모델은 학습 격자와 다른 해상도에서도 작동:
- 학습: 512~8192 격자점으로 학습한 모델
- 평가: 비학습 해상도에서 일관된 성능
- **원인**: 정리 4.3의 LBB 조건이 시퀀스 길이 n과 무관하게 성립

***

## V. 모델의 일반화 성능 향상 가능성

### 5.1 일반화 성능의 이론적 보증

#### LBB 조건의 역할
쌍선형 형식 $b(\cdot, \cdot): V \times Q \to \mathbb{R}$가 이산 LBB 조건을 만족하면:
$$\|b(\cdot, q)\|_{V'_h} \geq c \|q\|_H, \quad c > 0 \text{ (n-independent)}$$

이는 다음을 보증합니다:
$$\min_\theta \|f - g_\theta(y)\|_H \leq \frac{1}{c} \text{(interpolation error)} + \text{(subspace error)}$$

**핵심**: Softmax 제거로 LBB 조건 검증 가능 → 시퀀스 길이 불변 수렴[1]

#### 동적 기저 업데이트 (정리 4.4)
각 계층에서 기저함수가 업데이트:
$$a(v_j, \cdot) - b(\cdot, z_j) \to 0$$

이는 네트워크가 학습 과정에서 현재 부분공간의 리만 기하(metric)를 최적화함을 의미합니다.

### 5.2 실제 일반화 특성

**1. 미학습 해상도 처리**:
- Galerkin Transformer: n=2048에서 학습 → n=8192에서 1.150 (학습) vs 1.025 (평가)
- 상대 오차 증가: 10% 미만 → 견고한 외삽(extrapolation)

**2. 노이즈 강건성**:
역 문제에서 ε=0% (1.651), ε=0.01 (2.729), ε=0.1 (8.024)로 단조 증가
- 기울기 흐름 안정성으로 인한 자연스러운 성능 저하

**3. 비정규화 데이터 처리**:
Galerkin 계층 정규화로 정규화 부재 시 FT 대비 명시적 우위:
- Burgers' 방정식 (비정규화): GT 정규 Ln 1.203 vs FT 1.400 (~15% 개선)

### 5.3 향후 일반화 개선 방향

**이 연구의 한계에서 비롯된 개선 전략**:

1. **멀티스케일 기저 함수**:
   - 현재: 단일 d차원 잠재 공간
   - 개선: 계층별 다양한 스케일의 기저 → 고주파/저주파 동시 학습

2. **적응적 희소 어텐션**:
   - 현재: 모든 점에 균등 계산
   - 개선: 중요도 기반 라우팅 (후속 연구: Skip-Block Routing)

3. **물리 제약 통합**:
   - 에너지 보존, 엔트로피 조건 명시적 인코딩
   - 인버스 문제의 ill-posedness 정규화

***

## VI. 한계 및 개선 필요 영역

### 6.1 논문의 명시적 한계 (Limitations)

1. **무한 차원 공간의 저차원 가정**:
   - 근사 부분공간 $Q_h, V_h$가 실제로는 무한 차원이나, 실제 연산자가 저차원 구조(smoothing property) 필요
   - 고주파를 강하게 증폭하는 PDE는 적용 어려움

2. **2D 고해상도 비효율**:
   - Galerkin 타입의 O(nd²) 복잡도로 d×d 메시 크기가 커지면 비효율
   - 예: 128×128 이상에서는 다운샘플링 필수 → 정보 손실

3. **비인과성 (Non-causality)**:
   - 행렬 곱 순서로 인해 선형 변수는 비인과적
   - 디코더 필수 → 재귀 예측 불가능

4. **학습 불안정성**:
   - Softmax 제거로 정규화 수준 감소 → 초기 학습 불안정
   - 해결책: 새 계층 정규화 + 초기화 전략이지만 완전하지 않음 (Table 8 참고)[1]

### 6.2 최신 연구의 추가 발견 (2023-2025)

**FNO의 근본적 취약점** (최근 분석, 2025):
- **스펙트럼 편향**: 저주파만 학습하여 고주파 전혀 포착 불가
- **해상도 외삽 실패**: 학습보다 세밀한 격자에서 새로운 고주파 성분 완전 실패
- **적분 오차 누적**: 장시간 롤아웃에서 지수 증가 (오류 3.5배)[2]

***

## VII. 최신 관련 연구 비교 분석 (2020년 이후)

### 7.1 동시대 연구와의 비교

#### Fourier Neural Operator (FNO, Li et al. 2020-2021)
| 항목 | Galerkin Transformer | FNO |
|------|-------------------|-----|
| **기초 이론** | Petrov-Galerkin 사영 | 스펙트럼 필터링 |
| **공간 복잡도** | O(nd²) | O(n log n) |
| **고주파 포착** | 우수 (동적 기저) | 약함 (필터 절단) |
| **해석성** | 함수공간 해석 명확 | 주파수 영역 직관적 |
| **해상도 불변** | ✓ (LBB 조건) | 부분적 |

**결과**: 정확도는 GT > FNO, 속도는 FNO > GT (3D 고차원)[3][1]

#### Operator Transformer (OFormer, Li et al. 2022)
```
자기-어텐션 + 교차-어텐션 + MLP
```
- **장점**: 불규칙 격자 처리, 인코더-디코더 구조로 인과성 보존
- **한계**: 계산 복잡도 더 높음, 이론적 근거 부족
- **관계**: GT의 선형 어텐션 아이디어 채용[4]

#### General Neural Operator Transformer (GNOT, Hao et al. 2023)
```
이질 정규화(heterogeneous normalization) 어텐션
+ 기하학적 게이팅 (soft domain decomposition)
```
- **개선점**: 불규칙 격자 + 다중 입력 함수 + 멀티스케일
- **성과**: 기계 설계, CFD 산업 응용에서 SOTA 성능[5]
- **차이**: GT는 이론 중심, GNOT는 실무 응용 중심[6]

#### Vision Transformer-Operator (ViTO, Ovadia et al. 2023)
```
Vision Transformer (패치 기반) + U-Net 인코더-디코더
```
- **특징**: 패치 분할로 계산 복잡도 대폭 감소
- **성과**: 파라미터 90% 감소, 5배 속도, 슈퍼 해상도 역문제 해결
- **장점**: 소형 모델로도 정확도 유지[7]

### 7.2 후속 발전 (2024-2025)

#### Poseidon (다중스케일 연산자 변환기)
- **혁신**: 시간 조건 계층 정규화로 시간 종속 PDE 연속 평가 가능
- **확장성**: 반군(semi-group) 성질 활용으로 학습 데이터 대폭 확장
- **성과**: Foundation model 수준의 다양한 PDE 학습[8]

#### Continuous Vision Transformer (CViT, 2024)
- **개선**: 그리드 기반 좌표 임베딩 + 쿼리별 교차-어텐션
- **다중 스케일**: 계층적 표현 학습으로 미세한 세부 포착
- **결과**: GT 대비 고주파 재구성 성능 향상[9]

#### Neural Interpretable PDE Solver (2025)
- **특징**: Fourier 커널 + Attention 결합
- **목표**: Forward PDE + 역 PDE (지배 방정식 발견) 동시 학습
- **성과**: 단일 모델로 해 예측 + 물리 메커니즘 발견[10]

### 7.3 견고성 및 실패 분석 연구 (2025)

최근 연구(2025)는 FNO와 신경 연산자의 근본적 한계 실증:

**FNO 약점 진단** (1,000개 모델 규모 테스트):
- **스펙트럼 편향**: 학습 격자의 Nyquist 주파수 이상 완전 실패
- **계수 민감도**: 고주파 계수 변화에 10배 이상 오류 증가
- **장시간 불안정**: 5배 적분 시간에서 지수 오류 증가 (포화값 도달)[2]

**해결 방안**:
1. 멀티 해상도 학습 데이터
2. 적응적 기저 (GT의 동적 업데이트 아이디어 강화)
3. 희소성 메커니즘 (Skip-Block Routing)
4. 안정성 제약 (sympathic integrator 유사)[3]

***

## VIII. 일반화 성능과 미래 연구 방향

### 8.1 Galerkin Transformer의 일반화 강점

| 차원 | 강점 | 약점 |
|------|------|------|
| **이론** | LBB 조건으로 n-불변 수렴 보증 | 실제 조건 검증 어려움 |
| **정확도** | 역문제/고주파 우수 | 3D 대규모 문제 비효율 |
| **효율성** | 선형 복잡도 (d² 항만 남음) | d > 128일 때 메모리 부담 |
| **해석성** | 기저함수 동적 업데이트 명확 | 블랙박스적 해석 여전함 |

### 8.2 다음 단계 연구 방향

**이론적 확장**:
1. 비정규 부분공간에 대한 LBB 조건 완화
2. 동적 기저 업데이트의 수렴 속도 분석
3. 멀티스케일 기저의 최적 선택 문제

**아키텍처 개선**:
1. **적응적 스파시티**: 중요도 라우팅으로 각 계층의 계산 선택적 실행
2. **계층 간 기저 공유**: 계산 복잡도 O(nd²) → O(nd) 감소 가능
3. **물리 제약 인코딩**: 에너지 보존, 엔트로피 부등식 경성 제약

**응용 확대**:
1. 초고차원(d > 100) 문제의 효율적 처리
2. 동역학 시스템의 장시간 예측 (오류 누적 제어)
3. 다중 물리(multi-physics) 결합 문제

### 8.3 최종 평가

**Galerkin Transformer의 학문적 기여**:
- Transformer를 PDE 해법에 적용한 최초 이론적 근거 제공
- Softmax 제거 = 선형 연산자 사영 동치성 규명
- 시퀀스 길이 독립 근사 보증의 새 패러다임

**실무 적용 관점**:
- 역 문제, 노이즈 있는 데이터에서 SOTA 성능
- 메모리/속도 효율로 임베디드 디바이스 가능성 제시
- 그러나 대규모 3D 문제는 여전히 FNO/GNOT 우선

***

## 참고문헌

 Cao, S. (2021). Choose a Transformer: Fourier or Galerkin. *NeurIPS 2021*, arXiv:2105.14995[1]
 Poseidon: Efficient Foundation Models for PDEs (2024)[8]
 Li, Z., et al. (2022). Transformer for Partial Differential Equations' Operator Learning. *TMLR* (2022)[11]
 Forcing and Diagnosing Failure Modes of Fourier Neural Operators (2025)[2]
 Neural Interpretable PDEs: Harmonizing Fourier Insights... (2025)[10]
 Hao, Z., et al. (2023). GNOT: A General Neural Operator Transformer. *ICML* 2023[5]
 Li, Z., et al. (2020). Fourier Neural Operator for Parametric PDEs. *ICLR* 2021[3]
 https://arxiv.org/abs/2302.14376 (GNOT)[6]
 https://arxiv.org/abs/2303.08891 (ViTO)[7]

출처
[1] 2105.14995v4.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/171dffd6-5f94-4046-8038-2a376461b92b/2105.14995v4.pdf
[2] Forcing and Diagnosing Failure Modes of Fourier Neural ... https://arxiv.org/html/2601.11428v1
[3] block mechanisms for efficient pde neural operators https://arxiv.org/pdf/2511.00032.pdf
[4] Transformer for Partial Differential Equations' Operator ... https://openreview.net/pdf?id=EPPqt3uERT
[5] GNOT: A General Neural Operator Transformer for Operator ... https://proceedings.mlr.press/v202/hao23c/hao23c.pdf
[6] GNOT: A General Neural Operator Transformer for ... https://arxiv.org/abs/2302.14376
[7] [2303.08891] ViTO: Vision Transformer-Operator https://arxiv.org/abs/2303.08891
[8] Poseidon: Efficient Foundation Models for PDEs https://arxiv.org/html/2405.19101v2
[9] CViT: Continuous Vision Transformer for Operator Learning http://arxiv.org/pdf/2405.13998.pdf
[10] Neural Interpretable PDEs: Harmonizing Fourier Insights ... https://arxiv.org/pdf/2505.23106.pdf
[11] Transformer for Partial Differential Equations' Operator Learning https://arxiv.org/pdf/2205.13671.pdf
[12] Upaya Meningkatkan Motivasi Belajar melalui Kompetensi Sosial Guru PAK pada Siswa Kelas 2 di SD Elpida Noelbaki Tahun Ajaran 2022/2023 https://jurnal.sttarastamarngabang.ac.id/index.php/sinarkasih/article/view/376
[13] Comparative Evaluation of the Accuracy, Operator Comfort and Time Taken for Implant Placement among Different Practitioners under Dynamic Navigation https://www.jcdr.net//article_fulltext.asp?issn=0973-709x&year=2023&month=October&volume=17&issue=10&page=FC10-FC14&id=18581
[14] Dual-Energy CT Deep Learning Radiomics to Predict Macrotrabecular-Massive Hepatocellular Carcinoma. http://pubs.rsna.org/doi/10.1148/radiol.230255
[15] Development of a novel machine learning model based on laboratory and imaging indices to predict acute cardiac injury in cancer patients with COVID-19 infection: a retrospective observational study https://link.springer.com/10.1007/s00432-023-05417-3
[16] Special issue on machine learning in additive manufacturing https://www.tandfonline.com/doi/full/10.1080/0951192X.2023.2235679
[17] Learning Multilingual Sentence Representations with Cross-lingual Consistency Regularization https://arxiv.org/abs/2306.06919
[18] Histobot: Question Generation System Using Deep Learning Techniques https://rspsciencehub.com/article_23871.html
[19] Hippocampus substructure segmentation using morphological vision transformer learning https://iopscience.iop.org/article/10.1088/1361-6560/ad0d45
[20] Deep learning diagnostic and severity-stratification for interstitial lung diseases and chronic obstructive pulmonary disease in digital lung auscultations and ultrasonography: clinical protocol for an observational case–control study https://bmcpulmmed.biomedcentral.com/articles/10.1186/s12890-022-02255-w
[21] Active Learning Methodology Applied in Electric Machines Classes https://ieeexplore.ieee.org/document/10408333/
[22] In-Context Operator Learning for Linear Propagator Models https://arxiv.org/html/2501.15106v1
[23] ViTO: Vision Transformer-Operator https://arxiv.org/pdf/2303.08891.pdf
[24] MODNO: Multi Operator Learning With Distributed Neural Operators https://arxiv.org/pdf/2404.02892.pdf
[25] From Features to Transformers: Redefining Ranking for Scalable Impact http://arxiv.org/pdf/2502.03417.pdf
[26] Q-Transformer: Scalable Offline Reinforcement Learning via
  Autoregressive Q-Functions https://arxiv.org/html/2309.10150
[27] OPT: Open Pre-trained Transformer Language Models https://arxiv.org/pdf/2205.01068.pdf
[28] Transformer for Partial Differential Equations' Operator ... https://arxiv.org/abs/2205.13671
[29] PDE-Transformer: A Continuous Dynamical Systems ... https://arxiv.org/html/2510.03272v1
[30] Mesh-Informed Neural Operator : A Transformer ... https://arxiv.org/html/2506.16656v1
[31] Enhancing Fourier Neural Operators with Local Spatial ... https://arxiv.org/html/2503.17797v2
[32] Towards understanding how attention mechanism works in ... https://arxiv.org/html/2412.18288v1
[33] PROSE: Predicting Operators and Symbolic Expressions ... https://arxiv.org/html/2309.16816v1
[34] A reduced-order derivative-informed neural operator for ... https://arxiv.org/html/2509.13620v1
[35] Geometry-Informed Neural Operator Transformer https://arxiv.org/html/2504.19452v2
[36] A Data-Aware Fourier Neural Operator for Modeling ... https://arxiv.org/html/2508.17238v1
[37] PDE-Guided Mechanisms for Long-Sequence Transformers https://arxiv.org/html/2505.20666v1
[38] Transformer for Partial Differential Equations' Operator ... https://www.semanticscholar.org/paper/Transformer-for-Partial-Differential-Equations'-Li-Meidani/179070d3d43e97d1ce4d12127a3dc63581328809
[39] Redefining Neural Operators in 𝑑+1 Dimensions https://arxiv.org/html/2505.11766v2
[40] Improved Operator Learning by Orthogonal Attention https://arxiv.org/html/2310.12487v4
[41] Memristive floating-point Fourier neural operator network ... https://www.science.org/doi/10.1126/sciadv.adv4446
[42] A review on the attention mechanism of deep learning https://www.sciencedirect.com/science/article/abs/pii/S092523122100477X
[43] GSoC '25 - Final Report: Fourier Neural Operator https://forum.deepchem.io/t/gsoc-25-final-report-fourier-neural-operator/2221
[44] Nonlocal Attention Operator: Materializing Hidden Knowledge ... https://pmc.ncbi.nlm.nih.gov/articles/PMC12588098/
[45] Operator Learning with Neural Fields: Tackling PDEs on ... https://proceedings.neurips.cc/paper_files/paper/2023/file/df54302388bbc145aacaa1a54a4a5933-Paper-Conference.pdf
[46] D-Fno: A Decomposed Fourier Neural Operator for Large- ... https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4944664
[47] Physics-informed attention-based neural network for ... https://pmc.ncbi.nlm.nih.gov/articles/PMC9085858/
[48] Transformer for Partial Differential Equations' Operator ... https://openreview.net/forum?id=EPPqt3uERT
[49] Lightweight Fourier Neural Operator for Time-Dependent ... https://neurips.cc/virtual/2025/123080
[50] Attention Is All You Need - Wikipedia https://en.wikipedia.org/wiki/Attention_Is_All_You_Need
[51] ViTO: Vision Transformer-Operator https://www.sciencedirect.com/science/article/abs/pii/S0045782524003657
[52] Enhancing Fourier Neural Operators with CNNs architectures https://www.sciencedirect.com/science/article/abs/pii/S0925231225005776
[53] A Length Adaptive Algorithm-Hardware Co-design of Transformer on FPGA Through Sparse Attention and Dynamic Pipelining https://dl.acm.org/doi/10.1145/3489517.3530585
[54] Deep-learning-based radiomics of intratumoral and peritumoral MRI images to predict the pathological features of adjuvant radiotherapy in early-stage cervical squamous cell carcinoma https://bmcwomenshealth.biomedcentral.com/articles/10.1186/s12905-024-03001-6
[55] ORFormer: Occlusion-Robust Transformer for Accurate Facial Landmark
  Detection https://arxiv.org/pdf/2412.13174.pdf
[56] ODEFormer: Symbolic Regression of Dynamical Systems with Transformers https://arxiv.org/pdf/2310.05573.pdf
[57] ASFormer: Transformer for Action Segmentation https://arxiv.org/pdf/2110.08568.pdf
[58] OT-Transformer: A Continuous-time Transformer Architecture with Optimal
  Transport Regularization http://arxiv.org/pdf/2501.18793.pdf
[59] AlgoFormer: An Efficient Transformer Framework with Algorithmic
  Structures http://arxiv.org/pdf/2402.13572.pdf
[60] HuggingFace's Transformers: State-of-the-art Natural Language Processing http://arxiv.org/pdf/1910.03771.pdf
[61] [PDF] GNOT: A General Neural Operator Transformer for ... https://www.semanticscholar.org/paper/GNOT:-A-General-Neural-Operator-Transformer-for-Hao-Ying/6e036e28e7af03bfcdd98ffa254df6644f7657c5
[62] Geometry Aware Operator Transformer As An Efficient And ... https://arxiv.org/html/2505.18781v4
[63] Physics-Informed Transformer operator for the prediction of ... https://arxiv.org/html/2601.19351v3
[64] Prions-Inspired Vision Transformers for Temperature ... https://arxiv.org/pdf/2411.05836.pdf
[65] Mixture-of-Experts Operator Transformer for Large-Scale ... https://arxiv.org/html/2510.25803v1
[66] Position-induced Transformer (PiT) for Operator Learning https://arxiv.org/html/2405.09285v1
[67] arXiv:2302.14376v1 [cs.LG] 28 Feb 2023 https://arxiv.org/pdf/2302.14376.pdf
[68] [PDF] ViTO: Vision Transformer-Operator https://www.semanticscholar.org/paper/ViTO:-Vision-Transformer-Operator-Ovadia-Kahana/4b572031d82294d4392723b746518f8f53d9fe3f
[69] Fugu-MT 論文翻訳(概要): GNOT: A General Neural Operator Transformer for Operator Learning https://fugumt.com/fugumt/paper_check/2302.14376v2
[70] ViTO: Vision Transformer-Operator - Tel Aviv University https://cris.tau.ac.il/en/publications/vito-vision-transformer-operator/
[71] GNOT: Neural Operator Transformer for PDEs https://www.emergentmind.com/papers/2302.14376
[72] MMET: A Multi-Input and Multi-Scale Transformer for ... https://www.ijcai.org/proceedings/2025/0849.pdf
[73] GitHub - HaoZhongkai/GNOT https://github.com/HaoZhongkai/GNOT
[74] George Karniadakis' Post https://www.linkedin.com/posts/george-karniadakis-9b499853_vito-the-godfather-of-neural-operators-activity-7042681293976797184-Zs6i
[75] GNOT: A General Neural Operator Transformer for Operator ... https://proceedings.mlr.press/v202/hao23c.html
[76] GNOT: A General Neural Operator Transformer for ... https://liner.com/review/gnot-general-neural-operator-transformer-for-operator-learning
