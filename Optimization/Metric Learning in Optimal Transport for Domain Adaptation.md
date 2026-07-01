# Metric Learning in Optimal Transport for Domain Adaptation

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문(Kerdoncuff et al., IJCAI-20)의 핵심 주장은 다음과 같습니다:

> **"최적 수송(Optimal Transport, OT)에서 기본적으로 사용되는 유클리드 거리(Euclidean distance)는 도메인 적응(Domain Adaptation, DA)을 위한 최적의 거리 측도가 아니며, 마할라노비스 거리(Mahalanobis distance)를 학습함으로써 더 나은 수송 계획(transportation plan)을 얻고 타겟 도메인에서의 오류를 줄일 수 있다."**

### 주요 기여 (Two-fold)

| 기여 | 내용 |
|------|------|
| **이론적 기여** | 여러 Wasserstein 거리를 포함하는 타겟 오류의 일반화 경계(generalization bound) 도출 |
| **알고리즘적 기여** | 이 이론을 바탕으로 Mahalanobis 거리를 최적화하는 **MLOT** 알고리즘 설계 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**비지도 도메인 적응(Unsupervised Domain Adaptation)** 문제를 다룹니다:

- 레이블이 있는 소스 도메인 $\mu_s$에서 학습한 모델을 레이블이 없는 타겟 도메인 $\mu_t$에 적용
- 기존 OT 기반 DA(OTDA)는 유클리드 거리를 비용 함수로 사용 → 도메인 구조를 제대로 반영하지 못함
- **핵심 문제**: OT의 ground metric(비용 함수)이 도메인 적응에 최적화되어 있지 않음

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 기본 Wasserstein 거리 정의

$p$-Wasserstein 거리는 다음과 같이 정의됩니다:

$$\mathcal{W}_p^p(\mu_s, \mu_t) = \min_{\gamma \in \Pi(\mu_s, \mu_t)} \int_{\mathcal{X} \times \mathcal{X}} c(x_s, x_t)^p \, d\gamma(x_s, x_t) \tag{1}$$

경험적 분포에서 이산적으로 표현하면:

$$\mathcal{W}_p^p(\hat{\mu}_s, \hat{\mu}_t) = \min_{\gamma \in \hat{\Pi}(\hat{\mu}_s, \hat{\mu}_t)} \sum_{i=1}^{m_s} \sum_{j=1}^{m_t} \|x_s^i - x_t^j\|_2^p \, \gamma_{ij} = \min_{\gamma \in \hat{\Pi}(\hat{\mu}_s, \hat{\mu}_t)} \langle \gamma, C^p \rangle \tag{2}$$

엔트로피 정규화를 추가하면 (Sinkhorn-Knopp 알고리즘 적용 가능):

$$\mathcal{W}_p^p(\hat{\mu}_s, \hat{\mu}_t) \approx \left\langle \underset{\gamma \in \hat{\Pi}(\hat{\mu}_s, \hat{\mu}_t)}{\arg\min} \langle \gamma, C^p \rangle - \lambda_e \Omega_e(\gamma), \; C^p \right\rangle \tag{3}$$

여기서 $\Omega_e(\gamma) = -\sum_{i=1}^{m_s} \sum_{j=1}^{m_t} \gamma_{ij} \log(\gamma_{ij})$ (Shannon 엔트로피)

#### Step 2: OTDA의 클래스 정규화

$$\min_{\gamma \in \hat{\Pi}(\hat{\mu}_s, \hat{\mu}_t)} \langle \gamma, C^p \rangle - \lambda_e \Omega_e(\gamma) + \lambda_c \Omega_c(\gamma) \tag{4}$$

여기서 $\Omega_c(\gamma) = \sum_{j=1}^{m_t} \sum_{cl=1}^{c} \|\gamma(\mathcal{I}_{cl}, j)\|_2$ (다른 클래스 소스 포인트가 같은 타겟 위치로 수송되는 것을 방지)

#### Step 3: 마할라노비스 거리 정의

$L \in \mathbb{R}^{k \times n}$, $M = L^T L$ (PSD 행렬)일 때:

$$D_M^2(x_s^i, x_s^j) = (x_s^i - x_s^j)^T M (x_s^i - x_s^j) = \|L(x_s^i - x_s^j)\|_2^2 \tag{5}$$

#### Step 4: PCA와 Wasserstein 거리의 관계 (Theorem 1)

**[핵심 이론]** PCA는 Wasserstein 거리 의미에서 차원 축소의 최적 방법임을 증명:

```math
\underset{g \in \mathcal{G}_d}{\arg\min} \; \mathcal{W}_2^2(\hat{\mu}, g_\# \hat{\mu}) = V^T V
```

여기서 $V$는 공분산 행렬의 상위 $d$개 고유벡터로 구성된 행렬, $\mathcal{G}_d = \{g: \mathbb{R}^n \to \mathbb{R}^n \mid \text{Dim}(\text{Im}(g)) \leq d\}$

**증명 요약**: 임의의 최적 $g^\*$에 대해 $g^\*$를 $\tilde{g}^\*$로 재정렬하면 $\gamma_{\tilde{g}^*} = \frac{1}{m}I_m$이 되고:

```math
\min_{g \in \mathcal{G}_d} \mathcal{W}_2^2(\hat{\mu}, g_\# \hat{\mu}) = \frac{1}{m}\sum_i \|x^i - \tilde{g}^*(x^i)\|_2^2 \geq \frac{1}{m}\sum_i \|x^i - V^T V x^i\|_2^2
```

PCA가 이 하한을 달성하므로 $V^TV$가 최적해입니다.

#### Step 5: 타겟 오류의 일반화 경계 (Theorem 2)

**[핵심 이론]** $g_s, g_t: \mathcal{X} \to \mathcal{X}$에 대해 $\forall h \in \mathcal{H}$:

```math
\epsilon_t(h) \leq \epsilon_s(h) + 2K\left[\mathcal{W}_2(g_{s\#}\hat{\mu}_s, g_{t\#}\hat{\mu}_t)\right]
```

```math
+ 2K\left[\mathcal{W}_2(\hat{\mu}_s, g_{s\#}\hat{\mu}_s) + \mathcal{W}_2(g_{t\#}\hat{\mu}_t, \hat{\mu}_t)\right]
```

$$+ 2K\left[\mathcal{W}_2(\mu_s, \hat{\mu}_s) + \mathcal{W}_2(\hat{\mu}_t, \mu_t)\right] + \lambda \tag{9}$$

여기서:
- $K$: 가설 $h$의 Lipschitz 상수
- $\lambda$: 이상적 가설 $h^*$의 결합 오류
- **첫 번째 항**: 학습된 부분공간 간 Wasserstein 거리 (MLOT가 최소화)
- **두 번째 항**: PCA로 최소화 가능 (Theorem 1 활용)
- **세 번째 항**: 샘플 복잡도 항 (데이터 크기에 의존)

#### Step 6: MLOT 최적화 문제

이론을 바탕으로 설계된 핵심 최적화 문제:

$$\min_{L_s \in \mathbb{R}^{n \times n}, \gamma \in \hat{\Pi}(\hat{\mu}_s, \hat{\mu}_t)} \langle \gamma, C^2(L_s, L_t) \rangle - \lambda_e \Omega_e(\gamma) + \lambda_c \Omega_{cl}(\gamma) + \lambda_l \Omega_l(L_s) \tag{10}$$

여기서:
- $C^2(L_s, L_t)_{ij} = \|L_s x_s^i - L_t x_t^j\|_2^2$ (마할라노비스 기반 비용 행렬)
- $\Omega_l(L_s)$: 메트릭 학습 정규화 항 (예: LMNN)
- $\lambda_e, \lambda_c, \lambda_l$: 각 항의 균형을 제어하는 정규화 파라미터

---

### 2.3 모델 구조 (MLOT 알고리즘)

```
Algorithm 1: MLOT
입력: η (그래디언트 스텝), Xs, Xt, Ys

1: Vs = PCA(Xs), Vt = PCA(Xt)
2: Ls = Vs^T Vs,  Lt = Vt^T Vt      ← Theorem 1로 초기화
3: for i = 1 to N do
4:   γ = argmin_{γ∈Π} ⟨γ, C²(Ls,Lt)⟩ - λe·Ωe(γ) + λc·Ωcl(γ)   ← OT 최적화
5:   Ls = Ls - η∇_{Ls}(⟨γ, C²(Ls,Lt)⟩ + λl·Ωl(Ls))              ← 메트릭 학습
6: end for
7: X̃s = γ Lt Xt                     ← 수송된 소스 포인트 계산
8: classifier = classifier_method(X̃s, Ys)
9: Ŷt = classifier(Xt)
10: return Ŷt
```

**워크플로우**:
$$X_s \xrightarrow{\text{ML with PCA}} L_s X_s \xrightarrow{\text{Optimal Transport}} L_t X_t \xleftarrow{\text{PCA}} X_t$$

**핵심 설계 선택**:
- $L_s$만 학습 (소스 레이블 활용, trivial solution 방지)
- $L_t = V_t^T V_t$ 고정 (Theorem 1 기반)
- 교대 최적화: $\gamma$와 $L_s$를 번갈아 업데이트

---

### 2.4 성능 향상 및 한계

#### 성능 향상

| 데이터셋 | OTDA | OTDAp | MLOT | 개선폭 |
|---------|------|-------|------|--------|
| Office-Caltech SURF AVG | 48.2 | 48.8 | **49.7** | +1.5 |
| Office-Caltech DeCAF6 AVG | 83.2 | 82.6 | **84.7** | +1.8 |
| Office31 AVG | 65.3 | 65.2 | **66.2** | +0.9 |
| **전체 평균** | 65.6 | 65.6 | **67.0** | +1.4 |

- 12개 SURF DA 서브문제 중 **8회** OTDA 능가
- 어떤 파라미터 조합에서도 MLOT가 OTDA보다 항상 우수

#### 한계

1. **하이퍼파라미터 민감성**: $\lambda_e, \lambda_c, \lambda_l, d, N$ 등 5개의 하이퍼파라미터 튜닝 필요
2. **비볼록 최적화**: 문제가 비볼록(non-convex)하여 초기화에 민감 (PCA 초기화로 완화)
3. **비지도 교차검증의 어려움**: 타겟 레이블 없이 최적 파라미터 선택이 어려움
4. **심층학습과의 통합 미흡**: PyTorch 버전을 구현했으나, 사전 추출된 특징에서 성능 향상 없음
5. **확장성**: 대규모 데이터셋에서 OT의 초입방체(supercubic) 복잡도 문제
6. **$L_s$만 학습**: 비대칭 설계로 인한 표현력 제한

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 일반화 경계 분석

Theorem 2의 경계식을 항별로 분해하여 분석합니다:

```math
\epsilon_t(h) \leq \underbrace{\epsilon_s(h)}_{\text{소스 위험}} + \underbrace{2K \mathcal{W}_2(g_{s\#}\hat{\mu}_s, g_{t\#}\hat{\mu}_t)}_{\textbf{(A) MLOT가 최소화}} + \underbrace{2K[\mathcal{W}_2(\hat{\mu}_s, g_{s\#}\hat{\mu}_s) + \mathcal{W}_2(g_{t\#}\hat{\mu}_t, \hat{\mu}_t)]}_{\textbf{(B) PCA로 최소화}} + \underbrace{2K[\mathcal{W}_2(\mu_s, \hat{\mu}_s) + \mathcal{W}_2(\hat{\mu}_t, \mu_t)]}_{\textbf{(C) 샘플 수로 제어}} + \underbrace{\lambda}_{\textbf{(D) 적응 가능성}}
```

### 3.2 각 항의 일반화 기여

**항 (A)**: MLOT의 메트릭 학습이 직접 최소화  
→ 학습된 $L_s$가 소스/타겟 부분공간 간 분포 차이를 줄임

**항 (B)**: Theorem 1이 보장  
→ PCA가 Wasserstein 의미에서 최적 차원 축소임을 수학적으로 증명  
→ 

```math
\mathcal{W}_2(\hat{\mu}_s, g_{s\#}\hat{\mu}_s)
```

와 

```math
\mathcal{W}_2(g_{t\#}\hat{\mu}_t, \hat{\mu}_t)
```
동시 최소화

**항 (C)**: [Bolley et al., 2007]의 Theorem 2.1로 제어 가능  
→ 샘플 수 $m_s, m_t$가 증가할수록 이 항은 감소

**항 (D)**: $\lambda$는 적응 가능성(adaptability)을 나타냄  
→ [Redko et al., 2019a]의 이론으로 분석 가능

### 3.3 일반화 성능 향상의 메커니즘

```
유클리드 거리 (고정)          마할라노비스 거리 (학습)
       ↓                              ↓
도메인 구조 무시              도메인별 기하학 반영
       ↓                              ↓
불량한 수송 계획              최적화된 수송 계획
       ↓                              ↓
큰 W₂(g_s#μ̂_s, g_t#μ̂_t)    작은 W₂(g_s#μ̂_s, g_t#μ̂_t)
       ↓                              ↓
높은 타겟 오류               낮은 타겟 오류 (일반화 향상)
```

### 3.4 실험적 일반화 증거

파라미터 강건성 실험에서 **어떤 $(\lambda_e, \lambda_c)$ 조합에서도 MLOT가 OTDA보다 항상 양의 개선**을 보임:

$$\Delta_{acc}(\text{MLOT} - \text{OTDA}) > 0, \quad \forall (\lambda_e, \lambda_c) \in \text{실험 범위}$$

이는 메트릭 학습이 특정 파라미터 설정에 과적합된 것이 아닌, 구조적으로 우월함을 시사합니다.

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 4.1 향후 연구에 미치는 영향

#### 이론적 영향

1. **통합 이론 프레임워크**: OT, 메트릭 학습, DA를 단일 이론으로 연결한 최초의 시도 → 이후 연구의 이론적 토대 제공
2. **PCA-Wasserstein 등가성**: Theorem 1은 차원 축소와 OT의 연결고리를 제시 → 다른 차원 축소 기법(e.g., Autoencoder)과 OT의 연결 연구 촉진
3. **파라미터화된 Wasserstein 거리**: Ground metric을 학습 가능한 파라미터로 보는 관점 → 다양한 응용에서 적응형 OT 연구 확대

#### 알고리즘적 영향

1. **딥러닝과의 통합 가능성**: 미분 가능한 MLOT 구현 → End-to-end 학습 연구의 기반
2. **메트릭 학습 알고리즘의 플러그인 구조**: $\Omega_l(L_s)$를 교체 가능하도록 설계 → 다양한 메트릭 학습 방법론 적용 가능
3. **교대 최적화 패러다임**: 수송 계획과 메트릭을 교대 업데이트하는 방식 → 이후 연구들이 광범위하게 채택

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 비교는 제공된 논문의 내용과 일반적인 연구 흐름을 기반으로 작성하였으나, 개별 논문의 구체적 수치는 직접 검증이 필요합니다.

### 5.1 MLOT와 후속 연구 비교표

| 방법 | 연도 | Ground Metric | 이론적 보장 | DA 유형 | 딥러닝 통합 |
|------|------|--------------|------------|---------|------------|
| **MLOT** | 2020 | Mahalanobis (학습) | ✅ (Wasserstein 경계) | 비지도 | 부분적 |
| OTDA | 2017 | 유클리드 (고정) | 부분적 | 비지도 | ❌ |
| JDOT | 2017 | 유클리드 (고정) | ❌ | 비지도 | ❌ |
| DeepJDOT | 2018 | 피처 공간 학습 | ❌ | 비지도 | ✅ |
| SRW (Paty & Cuturi) | 2019 | 부분공간 투영 | 부분적 | - | ❌ |

### 5.2 주요 후속 연구 동향

**[참고: 아래는 연구 트렌드 기반 분석이며, 개별 논문 수치의 정확성은 직접 확인 필요]**

#### (1) 적응형 Ground Metric 연구 심화

MLOT 이후 ground metric 최적화 연구가 활성화되었으며, 다음과 같은 방향으로 발전:

- **Scalable Optimal Transport** (2021~): 대규모 데이터에서 OT를 효율적으로 계산
- **Robust OT**: 이상치(outlier)에 강건한 ground metric 설계
- **Fused Gromov-Wasserstein**: 구조적 정보를 고려한 거리 학습

#### (2) 딥러닝과 OT의 완전한 통합

MLOT의 한계(딥러닝 통합 미흡)를 극복한 연구들:

- **DASPOT**: 딥 피처와 OT를 end-to-end로 결합
- **CGDM** (2021): 조건부 Wasserstein 거리를 활용한 DA

#### (3) 이론적 정교화

- **[Redko et al., 2019a]** → 다중 소스 DA에서의 적응 가능성 이론 확장
- Wasserstein 경계의 타이트니스(tightness) 연구

### 5.3 MLOT의 한계와 최신 연구의 해결책

| MLOT의 한계 | 후속 연구의 접근 |
|-----------|--------------|
| 하이퍼파라미터 5개 | 자동화된 하이퍼파라미터 최적화 (AutoML) |
| 비볼록 최적화 | 볼록 완화(convex relaxation) 또는 더 나은 초기화 |
| 딥러닝 통합 미흡 | End-to-end 학습 프레임워크 |
| OT의 계산 복잡도 | Sliced Wasserstein, Mini-batch OT |
| $L_s$만 학습 | 양방향 메트릭 학습 |

---

## 6. 앞으로 연구 시 고려할 점

### 6.1 이론적 측면

1. **경계의 타이트니스 분석**: Theorem 2의 경계가 얼마나 타이트한지 분석 필요
   $$\epsilon_t(h) \leq \text{경계} \implies \text{실제 gap 분석}$$

2. **비선형 메트릭 확장**: 마할라노비스(선형)를 넘어 커널 기반 비선형 메트릭으로 일반화

3. **다중 소스 DA**: 복수의 소스 도메인에서 Theorem 2 확장

4. **부분 적응(Partial DA)**: 소스와 타겟의 클래스 집합이 다른 경우 이론 확장

### 6.2 알고리즘적 측면

1. **하이퍼파라미터 자동 선택**: 
   - 현재의 pseudo-label 기반 교차검증 개선
   - 메타학습(meta-learning) 기반 자동 파라미터 선택

2. **양방향 메트릭 학습**:
   - $L_s$와 $L_t$ 동시 학습 시 trivial solution 방지 메커니즘 연구
   
3. **계산 효율성**:
   - Mini-batch OT 적용으로 대규모 데이터셋 확장
   - Sliced Wasserstein 거리 활용

4. **딥러닝 통합**:
   ```
   MLOT(사전 추출 피처) → MLOT(end-to-end 딥러닝)
   ```
   미분 가능한 OT 레이어와 메트릭 학습을 신경망에 통합

### 6.3 실용적 측면

1. **클래스 불균형**: 소스/타겟 도메인의 클래스 분포가 다른 경우 처리
2. **개방형 DA(Open-set DA)**: 타겟에만 존재하는 새로운 클래스 처리
3. **지속적 적응(Continual DA)**: 시간에 따라 변화하는 타겟 분포 처리

---

## 참고자료

**주 논문**
- Kerdoncuff, T., Emonet, R., & Sebban, M. (2020). *Metric Learning in Optimal Transport for Domain Adaptation*. IJCAI-20, pp. 2162-2168.

**논문 내 인용 참고자료**
- Courty, N., Flamary, R., Tuia, D., & Rakotomamonjy, A. (2017). *Optimal transport for domain adaptation*. PAMI.
- Courty, N., Flamary, R., Habrard, A., & Rakotomamonjy, A. (2017). *Joint distribution optimal transportation for domain adaptation*. NIPS.
- Redko, I., Habrard, A., & Sebban, M. (2017). *Theoretical analysis of domain adaptation with optimal transport*. ECML PKDD.
- Paty, F.P., & Cuturi, M. (2019). *Subspace robust wasserstein distances*. arXiv:1901.08949.
- Fernando, B., Habrard, A., Sebban, M., & Tuytelaars, T. (2013). *Unsupervised visual domain adaptation using subspace alignment*. ICCV.
- Cuturi, M. (2013). *Sinkhorn distances: Lightspeed computation of optimal transport*. NIPS.
- Villani, C. (2008). *Optimal transport: old and new*. Springer.
- Bellet, A., Habrard, A., & Sebban, M. (2015). *Metric learning*. Synthesis Lectures on AI and ML.
- Ben-David, S., Blitzer, J., Crammer, K., & Pereira, F. (2007). *Analysis of representations for domain adaptation*. NIPS.
- Bolley, F., Guillin, A., & Villani, C. (2007). *Quantitative concentration inequalities for empirical measures on non-compact spaces*. PTRF.
- Redko, I., Morvant, E., Habrard, A., Sebban, M., & Bennani, Y. (2019). *Advances in Domain Adaptation Theory*. Elsevier.
- Weinberger, K.Q., & Saul, L.K. (2009). *Distance metric learning for large margin nearest neighbor classification*. JMLR.

**코드 저장소**
- https://github.com/Hv0nnus/MLOT
