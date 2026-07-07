# High-accuracy sampling for diffusion models and log-concave distributions

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 질문
스코어 함수(log-density의 그래디언트) 평가만을 사용하여 목표 정확도 $\delta$에 대해 $\mathrm{polylog}(1/\delta)$ 반복 복잡도를 달성할 수 있는가?

### 주요 기여

**① 지수적 복잡도 개선 (Exponential Improvement)**

이전 최고 결과들이 $\tilde{O}(d/\delta)$ 또는 $\tilde{O}(d/\delta^{1/2})$의 쿼리 복잡도를 가졌던 것과 달리, 이 논문은:

$$\text{쿼리 수} = \tilde{O}\!\left(\mathsf{d}_\star \cdot \log^3\!\left(\frac{d + M_2^2}{\delta^2}\right)\right)$$

을 달성한다. 여기서 $\mathsf{d}\_\star$는 데이터의 **내재적 차원(intrinsic dimension)**, $M_2^2 = \mathbb{E}\_{X_0 \sim p_\mathrm{data}}\|X_0\|^2$이다.

**② First-Order Rejection Sampling (FORS)**

스코어(1차 정보)만으로 기각 샘플링을 시뮬레이션하는 새로운 메타 알고리즘 제안.

**③ log-concave 분포에 대한 최초의 $\mathrm{polylog}(1/\delta)$ 샘플러**

밀도 평가 없이 그래디언트 평가만으로 log-concave 분포의 고정밀 샘플링 달성.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**기존 문제점:**
- DDPM은 $\Omega(1/\delta)$의 복잡도 하한이 알려져 있음 (Jiao et al., 2025)
- 고차 이산화(higher-order discretization) 방법들은 복잡도가 $C\_p \cdot d^{1+1/p}/\delta^{1/p}$ 수준으로 여전히 poly $(1/\delta)$ 의존성 존재
- 밀도 평가를 활용한 방법(Huang et al., 2024; Wainwright, 2025)은 실제 확산 모델과 호환되지 않음

**목표:**
$$D(p_\mathrm{data}, \hat{p}) \leq \delta + C_\mathrm{apx} \cdot \varepsilon_\mathrm{score}$$
를 $\mathrm{polylog}(1/\delta)$ 스텝으로 달성하되, 스코어 평가만 사용.

---

### 2.2 제안하는 방법: FORS (First-Order Rejection Sampling)

#### 핵심 아이디어: Bernoulli Factory

밀도 값 $f(x)$ 없이 $b \sim \mathrm{Ber}(c \cdot e^{-f(x)})$를 생성하는 문제를 다룬다.

미적분 항등식:
$$f(x) = \mathbb{E}_{y \sim \mathrm{Unif}([0,x])}\left[x f'(y)\right]$$

이를 이용한 테일러 전개:
$$e^{\mathbb{E}W_1} = e \cdot \mathbb{E}\!\left[\prod_{j=1}^{J}\frac{1+W_j}{2}\right], \quad J \sim \mathrm{Poisson}(2)$$

#### Algorithm 1 (FORS)

**입력:** 파라미터 $B>0$, 제안 분포 $q$, 추정량 분포 $(\mathcal{W}\_x)_{x \in \mathbb{R}^d}$ (지지가 $[-B,B]$)

```
for i = 1, 2, 3, ...
  x ~ q 샘플
  J ~ Poisson(2B) 샘플
  W_1, ..., W_J ~ W_x 샘플 (i.i.d.)
  확률 prod_{j=1}^J (B + W_j)/(2B) 로 x 출력
end for
```

**Theorem 3.1 (FORS 보장):**

Algorithm 1의 출력 분포는 $\hat{p}(x) \propto q(x) e^{\mathbb{E}[W_1|x]}$이며, $W_j$의 총 샘플 수는 확률 $1-\delta$ 이상으로:
$$O\!\left(B e^{2B}(T + \log(1/\delta))\right)$$
이다. ($T$: FORS 호출 횟수)

---

#### Gaussian Tilt 샘플링

핵심 서브루틴: 다음 형태의 Gaussian tilt에서 샘플링

$$\nu(x) \propto \exp\!\left(-f(x) - \frac{\|x - x_0\|^2}{2\eta}\right) \tag{7}$$

1차 테일러 전개로 제안 분포를 선택:
$$q = \mathcal{N}\!\left(x_0 - \eta \nabla f(x_+),\, \eta I\right)$$

경로 적분 기반 추정량 (경로 함수 Eq. (11)):
$$\gamma_{z,r}(x) = a_r x + (1-a_r)\hat{x} + b_r z, \quad a_r = \sin(\pi r/2),\; b_r = \cos(\pi r/2) \tag{11}$$

추정량:

$$W_{r,z,x} := \langle \dot{\gamma}_{z,r}(x),\, \nabla f(x_+) - \nabla f(\gamma_{z,r}(x))\rangle \tag{10}$$

**Theorem 3.3:** $\nabla f$가 Hölder 연속 (지수 $s \in [0,1]$, 상수 $\beta_s$)이면:

$$\eta^{-1} \gg \left(\beta_s^2 d^s \log(1/\delta) + \frac{s\beta_s^2}{d^{1-s}}\log^2(1/\delta)\right)^{1/(1+s)}$$

조건 하에서 $D_{\chi^2}(\nu \| \hat{\nu}) \leq \delta^2$.

---

### 2.3 확산 모델 샘플링

#### DDPM 순방향 프로세스

$$X_0 \sim p_\mathrm{data},\quad X_{k+1} \sim \mathcal{N}(\alpha_k X_k,\, \alpha_k^2 \eta_k I) \tag{3}$$

Tweedie 항등식에 의한 진짜 스코어:
$$\mathsf{s}_k^\star(x) = \nabla \log p_k(x) = \frac{1}{\sigma_k^2}\mathbb{E}[\bar{\alpha}_k X_0 - X_k \mid X_k = x] \tag{5}$$

역방향 전이 커널:
$$\rho_k(x \mid x') \propto_x p_k(x)\exp\!\left(-\frac{\|x - \alpha_k^{-1}x'\|^2}{2\eta_k}\right) \tag{4}$$

#### 내재적 차원 (Intrinsic Dimension)

**Definition 4.1:**

$$\mathsf{d}_\star := \dim_{\sigma_0^2/\alpha_0^2}(p_\mathrm{data}) = 1 \vee \inf_{r \geq 0}\left[\log \mathcal{N}(p_\mathrm{data};\, r) + \frac{r^2}{\sigma^2}\right] \wedge d$$

- 데이터가 $k$차원 매니폴드에 지지되면 $\mathsf{d}_\star = \tilde{O}(k)$
- 지지가 $N$점이면 $\mathsf{d}_\star \leq \log N$
- 항상 $\mathsf{d}_\star \leq d$

#### Algorithm 2 (역방향 확산 샘플링)

제안 분포:

$$\bar{\rho}_k(\cdot \mid X_{k+1}) = \mathcal{N}\!\left(\alpha_k^{-1}X_{k+1} + \alpha_k \eta_k \mathsf{s}_{k+1}(X_{k+1}),\; \bar{\eta}_k I\right), \quad \frac{1}{\bar{\eta}_k} = \frac{1}{\eta_k} + \frac{1}{\sigma_k^2} \tag{12}$$

**Theorem 4.3:** 다음 조건 하에서:

$$\frac{\sigma_k^2}{\eta_k} \gg \mathsf{d}_\star \log(1/\delta) + \log^2(1/\delta) \tag{16}$$

$$D_\mathrm{KL}(p_1 \| \hat{p}_1) \lesssim D_\mathrm{KL}(p_K \| \hat{p}_K) + K\delta + \sum_{k=1}^{K} \eta_k \varepsilon_{k,\mathrm{score}}^2$$

**Corollary 4.4:** Variance-preserving 설정에서 총 쿼리 수:

$$K \leq O\!\left((\mathsf{d}_\star + \log(\kappa/\delta))\log^2(\mathsf{d}_\star \kappa/\delta)\right), \quad \kappa = M_2^2/\sigma_0^2 + 1$$

Bounded Lipschitz 메트릭에서의 최종 복잡도:

$$\boxed{\mathsf{d}_\star \cdot \log^3\!\left(\frac{d + M_2^2}{\delta^2}\right)}$$

---

#### 비균일 Lipschitz 조건 하의 개선

**Assumption 4.5 (비균일 Lipschitz, 연산자 노름):**

$$\mathbb{P}_{Y_\tau \sim q_\tau}\!\left(\|\nabla m_\tau(Y_\tau)\|_\mathrm{op} > L_{\mathrm{op},\delta}\right) \leq \frac{\delta}{\mathsf{d}_\star^5} \tag{17}$$

**Assumption 4.6 (Frobenius 노름):**

$$\mathbb{P}_{Y_\tau \sim q_\tau}\!\left(\|\nabla m_\tau(Y_\tau)\|_F > L_{F,\delta}\right) \leq \frac{\delta}{\mathsf{d}_\star^5} \tag{18}$$

**Proposition 4.7:** Assumption 4.5가 $L_{\mathrm{op},\delta}$로 성립하면:

$$L_{F,\delta} \leq C\sqrt{L_{\mathrm{op},\delta/2}(\mathsf{d}_\star + \log(1/\delta))}$$

**Theorem 4.9** 하의 복잡도:

$$\boxed{L_{F,\delta} \cdot \log^3\!\left(\frac{d + M_2^2}{\delta^2}\right)}$$

**Proposition 4.10**의 최종 복잡도 (연산자 노름 기준):

```math
\min\!\left\{\sqrt{d L_\mathrm{op}},\; \mathsf{d}_\star^{2/3} L_\mathrm{op}^{1/3}\right\} \cdot \mathrm{polylog}\!\left(\frac{\mathsf{d}_\star + M + M_2^2}{\delta^2}\right)
```

---

### 2.4 Log-concave 샘플링

**Proximal Sampler** (Algorithm 3)와 FORS를 결합:

$$\widetilde{\mu}(x,y) \propto \exp\!\left(-f(x) - \frac{1}{2\eta}\|y-x\|^2\right)$$

RGO(Restricted Gaussian Oracle):
$$\mathrm{RGO}_{f,\eta,y}(x) \propto_x \exp\!\left(-f(x) - \frac{1}{2\eta}\|y-x\|^2\right)$$

이는 정확히 Gaussian tilt (식 (7))의 형태이므로 FORS로 구현 가능.

**결과 (smooth case $s=1$, log-Sobolev 조건):**
$$N = \tilde{O}\!\left(\kappa d^{1/2} \log^{3/2}\!\frac{\mathcal{R}}{\varepsilon^2} + \kappa\log^2\!\frac{\mathcal{R}}{\varepsilon^2}\right) \text{ 쿼리}$$
$$\kappa = C_\mathrm{LSI}(\mu)\beta_1, \quad \mathcal{R} = \log(1 + D_{\chi^2}(\mu_0 \| \mu))$$

**Lipschitz case ($s=0$, Poincaré 조건):**
$$N = \tilde{O}\!\left(C_\mathrm{PI}\beta_0^2 \log(1/\varepsilon)\log(\chi^2/\varepsilon^2)\right) \text{ 쿼리}$$

---

### 2.5 모델 구조 요약

```
[데이터 분포 p_data]
        ↓ 순방향 확산 (DDPM forward, Eq. 3)
[노이즈 분포 p_K ≈ N(0, σ²_K I)]
        ↓ Algorithm 2 (역방향 샘플링)
[각 스텝: FORS를 이용한 역방향 커널 근사]
        ↓ 내부: Gaussian tilt 샘플링 (Theorem 3.3)
        ↓ 경로 적분 추정량 Eq.(10)-(11)
        ↓ Bernoulli Factory (Algorithm 1)
[출력 분포 p̂_1 ≈ p_data (early stopped)]
```

---

### 2.6 성능 향상 비교

| 방법 | 복잡도 | 스코어만 사용 | 조건 |
|------|--------|--------------|------|
| DDPM (Chen et al. 2023c) | $\tilde{O}(d/\delta^2)$ | ✓ | 유계 지지 |
| Li and Cai (2024) | $\tilde{O}(d/\delta^{1/2})$ | ✓ | 최소 가정 |
| Li and Yan (2025); Jain and Zhang (2026) | $\tilde{O}(d/\delta)$ | ✓ | 최소 가정 |
| Huang et al. (2024) | $\tilde{O}(d^2\log(1/\delta))$ | ✗ (밀도 필요) | Lipschitz |
| Wainwright (2025) | $\tilde{O}(\sqrt{d}\log^3(1/\delta))$ | ✗ (밀도 필요) | Lipschitz |
| **본 논문 (FORS)** | $\tilde{O}(\mathsf{d}_\star\log^3(1/\delta))$ | **✓** | **2차 모멘트만** |

---

### 2.7 한계

1. **이론 논문**: 구현 및 실험적 검증은 미래 연구로 남겨짐
2. **스코어 오차의 선형 의존성**: $\sum_k \eta_k \varepsilon_{k,\mathrm{score}}^2$ 항이 성능에 영향
3. **근사 근접 연산자 필요**: log-concave 적용 시 proximal oracle 접근 가정
4. **Gatmiry et al. (2026) 비교**: 동시 연구는 $\tilde{O}((R/\sigma)^2 \log^2(1/\delta))$를 달성하지만 sub-exponential 스코어 오차 및 Lipschitz 스코어 가정 필요 (본 논문은 $L^2$ 스코어 오차만 요구)
5. **$\mathrm{polylog}$ 인수 개선 여지**: 현재 $\log^3$ 인수가 최적인지 불명확

---

## 3. 일반화 성능 향상 가능성

### 3.1 내재적 차원을 통한 차원 적응성

**핵심 통찰**: 복잡도가 임베딩 차원 $d$가 아닌 내재적 차원 $\mathsf{d}_\star$에 의존함.

$$\mathsf{d}_\star \leq d, \quad \mathsf{d}_\star = \tilde{O}(k) \text{ (k차원 매니폴드의 경우)}$$

이는 다음을 의미한다:
- 실제 데이터(이미지, 텍스트)가 저차원 매니폴드에 분포한다는 **매니폴드 가설**과 정합
- 이론적 복잡도가 실제 데이터의 내재적 복잡도를 반영

**일반화와의 연결**: 모델이 데이터의 내재적 구조를 더 잘 포착할수록 $\mathsf{d}_\star$가 작아지고, 샘플링 정확도가 향상됨. 이는 스코어 함수 학습 품질이 직접 샘플링 품질로 이어짐을 보장한다:

$$D_\mathrm{BL}(p_\mathrm{data}, \hat{p}_1)^2 \lesssim \delta^2 + \sum_{k=1}^{K} \eta_k \varepsilon_{k,\mathrm{score}}^2$$

### 3.2 최소 데이터 가정에서의 보편성

2차 모멘트 $M_2^2 < \infty$만을 가정하므로:
- 분포 형태에 무관한 일반적 보장
- 다양한 데이터 도메인에서의 이론적 일반화 가능성

### 3.3 스코어 오차 강건성

$$D_\mathrm{KL}(p_1 \| \hat{p}_1) \lesssim D_\mathrm{KL}(p_K \| \hat{p}_K) + K\delta + \sum_{k=1}^{K} \eta_k \varepsilon_{k,\mathrm{score}}^2$$

- 스코어 오차가 **평균 제곱 오차** ($L^2$)로 측정되어 실제 신경망 학습과 직접 연결
- $C_\mathrm{apx} = O(1)$로 근사 인수가 상수, 스코어 오차의 영향이 선형적으로 제어됨

### 3.4 비균일 Lipschitz 조건과 분포 적응

Gaussian 혼합 모델의 경우:
$$p_\mathrm{data} = \sum_{h=1}^{H} p_h \mathcal{N}(\mu_h, \sigma_h^2 I)$$
이면 $L_{\mathrm{op},\delta} \leq O(\log H \cdot \log(d/\delta))$가 성립하여 "거의 차원 자유" 복잡도 달성.

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 4.1 연구에 미치는 영향

**① 확산 모델 이론의 패러다임 전환**

DDPM의 $\Omega(1/\delta)$ 하한이 알려진 상황에서, 알고리즘 변경을 통해 $\mathrm{polylog}(1/\delta)$ 복잡도를 달성함으로써 "고차 이산화 vs. 알고리즘 설계"라는 새로운 연구 방향을 제시.

**② Bernoulli Factory의 샘플링 이론 적용**

기존에는 SDE 시뮬레이션에 제한적으로 사용되던 Bernoulli Factory 기법을 확산 모델과 log-concave 샘플링에 체계적으로 적용하는 새로운 연구 방향.

**③ 내재적 차원 분석의 표준화**

$\mathsf{d}_\star$의 정의와 분석이 향후 확산 모델 연구의 표준 도구가 될 가능성.

**④ log-concave 샘플링과 확산 모델의 통합**

두 분야를 통합하는 단일 프레임워크 제공 → 상호 발전 촉진.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 주요 기여 | 복잡도 | 비고 |
|------|------|----------|--------|------|
| Chen et al. (2023c) | 2023 | DDPM 이론적 보장 최초 확립 | $\tilde{O}(d/\delta^2)$ | 유계 지지 필요 |
| Benton et al. (2024) | 2024 | Stochastic localization 활용 | $\tilde{O}(d/\delta^2)$ | 최소 가정 |
| Li & Yan (2024) | 2024 | 내재적 차원 개념 도입 | $\tilde{O}(d/\delta^2)$ | NeurIPS 2024 |
| Li & Cai (2024) | 2024 | $1/\delta^{1/2}$ 가속 | $\tilde{O}(d/\delta^{1/2})$ | arXiv |
| Conforti et al. (2025) | 2025 | KL 수렴 보장 | $\tilde{O}(d/\delta^2)$ | SIAM |
| Huang et al. (2025a) | 2025 | 고차 ODE 솔버 | $C_p d^{1+1/p}/\delta^{1/p}$ | sub-poly but not polylog |
| Li & Yan (2025) | 2025 | $O(d/T)$ 이론 | $\tilde{O}(d/\delta)$ | ICLR 2025 |
| Jain & Zhang (2026) | 2026 | KL sharp 수렴 | $\tilde{O}(d/\delta)$ | ICLR 2026 |
| Gatmiry et al. (2026) | 2026 | ODE 가속, high-accuracy | $\tilde{O}((R/\sigma)^2\log^2(1/\delta))$ | 동시 연구, 강한 가정 |
| **본 논문** | **2026** | **FORS, polylog 복잡도** | $\tilde{O}(\mathsf{d}_\star \log^3(1/\delta))$ | **스코어만, 최소 가정** |

**Gatmiry et al. (2026)과의 상세 비교:**

| 항목 | 본 논문 | Gatmiry et al. (2026) |
|------|---------|----------------------|
| 스코어 오차 조건 | $L^2$ 오차 | Sub-exponential tail |
| 데이터 가정 | 2차 모멘트 | Gaussian convolution |
| 복잡도 | $\tilde{O}(\mathsf{d}_\star\log^3(1/\delta))$ | $\tilde{O}((R/\sigma)^2\log^2(1/\delta))$ |
| 적용 범위 | 확산 + log-concave | 확산 모델 (ODE 기반) |
| 스코어 Lipschitz 가정 | 불필요 | 필요 |

---

### 4.3 향후 연구 시 고려할 점

**① 알고리즘 구현 및 실험**
- 논문 자체가 "primarily theoretical"임을 인정
- FORS의 실제 샘플링 효율 (Poisson 난수 생성 오버헤드 등) 검증 필요
- 기존 DDPM 대비 실제 이미지 생성 품질 비교 실험

**② $\log^3$ 인수 개선**

$$\tilde{O}(\mathsf{d}_\star \log^3(1/\delta)) \xrightarrow{?} \tilde{O}(\mathsf{d}_\star \log(1/\delta))$$

현재 $\log^3$ 인수가 분석의 한계인지 본질적 하한인지 불명확.

**③ 스코어 오차 조건의 최적성**
- $L^2$ 스코어 오차에서 $\sum_k \eta_k \varepsilon_{k,\mathrm{score}}^2$ 의존성의 최적성 분석
- 스코어 학습 복잡도와의 엔드-투-엔드 분석 필요

**④ 조건 수 의존성 제거**
- log-concave 적용에서 $\kappa = C_\mathrm{LSI}\beta_1$ 의존성 개선
- KLS 추측(Klartag, 2023의 $\log d$ 바운드 활용) 등 최신 이소페리메트릭 결과와의 결합

**⑤ 근접 연산자(Proximal Oracle) 의존성**
- log-concave 샘플링에서 근접 연산자에 대한 의존성 제거 또는 약화
- 1차 정보만으로 근접 연산자를 근사하는 방법 연구

**⑥ 연속 시간 극한 분석**
- 이산 알고리즘의 연속 시간 대응물 분석
- 확률 흐름 ODE(probability flow ODE)와의 연결

**⑦ 조건부 생성 및 가이던스**
- Classifier-free guidance 등 조건부 생성 시나리오에서의 FORS 적용 및 보장

---

## 참고 자료

**주 논문:**
- Fan Chen, Sinho Chewi, Constantinos Daskalakis, Alexander Rakhlin. *"High-accuracy sampling for diffusion models and log-concave distributions."* arXiv:2602.01338v2, April 2026.

**논문 내 인용 주요 참고문헌:**
- Chen et al. (2023c). *"Sampling is as easy as learning the score."* ICLR 2023.
- Benton et al. (2024). *"Nearly d-linear convergence bounds for diffusion models via stochastic localization."* ICLR 2024.
- Li and Yan (2024). *"Adapting to unknown low-dimensional structures in score-based diffusion models."* NeurIPS 2024.
- Li and Cai (2024). *"Provable acceleration for diffusion models under minimal assumptions."* arXiv:2410.23285.
- Li and Yan (2025). *"O(d/T) convergence theory for diffusion probabilistic models under minimal assumptions."* ICLR 2025.
- Jain and Zhang (2026). *"A sharp KL convergence analysis for diffusion models under minimal assumptions."* ICLR 2026.
- Gatmiry, Chen, and Salim (2026). *"High-accuracy and dimension-free sampling with diffusions."* arXiv:2601.10708.
- Lee, Shen, and Tian (2021). *"Structured logconcave sampling with a restricted Gaussian oracle."* COLT 2021.
- Fan, Yuan, and Chen (2023). *"Improved dimension dependence of a proximal algorithm for sampling."* COLT 2023.
- Altschuler and Chewi (2024). *"Faster high-accuracy log-concave sampling via algorithmic warm starts."* J. ACM.
- Huang et al. (2024). *"Reverse transition kernel."* NeurIPS 2024.
- Wainwright (2025). *"Score-based sampling without diffusions."* arXiv:2512.24152.
- Jiao, Zhou, and Li (2025). *"Optimal convergence analysis of DDPM for general distributions."* arXiv:2510.27562.
- Keane and O'Brien (1994). *"A Bernoulli factory."* ACM TOMACS.
- Klartag (2023). *"Logarithmic bounds for isoperimetry and slices of convex sets."* arXiv:2303.14938.
- Chewi (2026). *"Log-concave sampling."* (book draft) https://chewisinho.github.io/
