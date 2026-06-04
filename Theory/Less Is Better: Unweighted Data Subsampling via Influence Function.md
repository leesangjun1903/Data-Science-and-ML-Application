# Less Is Better: Unweighted Data Subsampling via Influence Function

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
> **"적은 데이터로 더 나은 모델을 만들 수 있다"**

기존의 서브샘플링 방법들은 대부분 **가중치 기반(weighted)** 방법으로, 서브셋 모델이 전체 데이터 모델(full-set-model)을 **넘어설 수 없다**는 이론적 한계를 가지고 있었습니다. 이 논문은 **비가중치(unweighted) + 영향 함수(Influence Function, IF)** 기반의 서브샘플링 방법인 **UIDS(Unweighted Influence Data Subsampling)**를 제안하여, 서브셋 모델이 전체 데이터 모델보다 **우수할 수 있음**을 이론적으로 증명하고 실험적으로 검증합니다.

### 주요 기여 (4가지)

| 기여 | 내용 |
|------|------|
| ① 이론적 증명 | 서브셋 모델이 전체 모델을 능가할 수 있음을 증명 (Lemma 1, 2) |
| ② 확률적 샘플링 분석 | χ²-divergence ball 위의 worst-case risk 분석 및 신뢰도 대리 지표 제안 |
| ③ 효율적 구현 | Hessian-free mixed Preconditioned Conjugate Gradient (PCG) 방법 사용 |
| ④ 다양한 실험 검증 | 텍스트, 이미지, CTR 예측 등 다양한 태스크에서 우수성 확인 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

**ERM(Empirical Risk Minimization)의 3가지 한계:**

1. **분포 이동(Distribution Shift):** 학습 분포 $P(x,y) = p_{train}(x)p(y|x)$와 테스트 분포 $Q(x,y) = q_{test}(x)p(y|x)$ 간의 불일치
2. **노이즈 데이터:** 레이블 노이즈나 데이터 오염이 모델 성능을 저하
3. **계산 부담:** 대규모 데이터셋에서의 학습 비용

**기존 가중치 서브샘플링의 한계:**

$$\mathcal{R}_w = \frac{1}{n} \sum_{i, O_i=1} \frac{1}{\pi_i} l_i(\theta) $$

가중치 방식의 기댓값:

$$\mathbb{E}_O(\mathcal{R}_w) = \frac{1}{n} \sum_{i=1}^{n} \mathbb{E}_{O_i}\left(O_i \times \frac{1}{\pi_i} l_i(\theta)\right) = \frac{1}{n} \sum_{i=1}^{n} l_i(\theta) $$

이 식은 서브셋 리스크의 기댓값이 **전체 세트 경험적 리스크와 동일**하므로, 이론적으로 서브셋 모델이 전체 모델을 능가할 수 없습니다. 또한 $\frac{1}{\pi_i}$ 가중치로 인한 **높은 분산 문제**도 존재합니다.

---

### 2-2. 제안하는 방법 (UIDS)

#### ① 비가중치 서브샘플링 목적함수

$$\mathcal{R}_{uw} = \frac{1}{|\{i, O_i=1\}|} \sum_{i, O_i=1} l_i(\theta) $$

가중치 항 $\frac{1}{\pi_i}$를 제거하여:
- 기존 학습 절차를 수정할 필요 없음
- 높은 분산 문제 해결
- 서브셋 모델이 전체 모델을 능가할 가능성 열림

#### ② 영향 함수(Influence Function)

파라미터에 대한 영향:

$$\psi_\theta(z_i) \triangleq \frac{d\hat{\theta}_\epsilon}{d\epsilon}\bigg|_{\epsilon=0} = -H_{\hat{\theta}}^{-1} \nabla_\theta l_i(\hat{\theta}) $$

테스트 리스크에 대한 영향:

$$\phi(z_i \sim P, z_j \sim Q') \triangleq \frac{dl_j(\hat{\theta}_\epsilon)}{d\epsilon}\bigg|_{\epsilon=0} = -\nabla_\theta l_j(\hat{\theta})^\top H_{\hat{\theta}}^{-1} \nabla_\theta l_i(\hat{\theta}) $$

여기서 $H_{\hat{\theta}} \triangleq \frac{1}{n}\sum_{i=1}^{n} \nabla^2_\theta l_i(\hat{\theta})$는 Hessian 행렬

전체 테스트 분포 $Q'$에 대한 훈련 샘플 $z_i$의 영향:

$$\phi_i(\hat{\theta}) \triangleq \sum_{j=1}^{m} \phi(z_i, z_j, \hat{\theta})$$

#### ③ 서브셋이 더 나은 이유: 핵심 이론

$n$개의 perturbation $\vec{\epsilon} = (\epsilon_1, \epsilon_2, ..., \epsilon_n)^\top$ 하에서 테스트 리스크 변화:

$$\mathcal{R}_{\hat{\theta}_\epsilon}(Q') - \mathcal{R}_{\hat{\theta}}(Q') \approx \frac{1}{m} \sum_{i=1}^{n} \epsilon_i \phi_i(\hat{\theta}) $$

**Lemma 1:** 훈련 분포 $P$ 위에서 영향 함수의 기댓값은 0:

$$\mathbb{E}_P(\phi) \approx \frac{1}{n} \sum_{i=1}^{n} \phi_i(\hat{\theta}) = 0 $$

**Lemma 2 (핵심):** $\epsilon$과 $\phi$가 음의 상관관계를 가지면 서브셋 모델이 전체 모델보다 우수:

$$\text{Cov}(\phi, \epsilon) \leq 0 $$

**증명 스케치:**
$$\mathbb{E}(\phi \times \epsilon) = \mathbb{E}(\phi) \times \mathbb{E}(\epsilon) + \text{Cov}(\phi, \epsilon) = 0 + \text{Cov}(\phi, \epsilon) \leq 0$$

따라서 $\mathcal{R}\_{\hat{\theta}\_\epsilon}(Q') \leq \mathcal{R}_{\hat{\theta}}(Q')$

이를 위해 $\epsilon(\phi)$를 **감소 함수(decreasing function)**로 설계합니다.

#### ④ 결정론적 vs 확률적 샘플링

**결정론적 샘플링(Data Dropout):**

$$\pi_i^* = \begin{cases} 1 & \phi_i \leq 0 \\ 0 & \phi_i > 0 \end{cases} $$

이 방법은 특정 $Q'$에 과도하게 확신(overly confident)하여 out-of-sample 일반화 성능이 저하됩니다.

#### ⑤ 확률적 샘플링 함수

**Linear Sampling:**

$$\pi_i = n\epsilon(\phi_i) + 1 = \max\{0, \min\{1, -\alpha\phi_i\}\} $$

**Sigmoid Sampling:**

$$\pi_i = n\epsilon(\phi_i) + 1 = \frac{1}{1 + e^{\frac{\alpha\phi_i}{\max(\{\phi_i\}) - \min(\{\phi_i\})}}} $$

여기서 $\alpha \in \mathbb{R}^+$는 신뢰도(confidence degree)를 조절하는 하이퍼파라미터

---

### 2-3. Worst-Case Risk 분석 (분포 강건성)

$\chi^2$-divergence 분포 공(ball):

$$\mathcal{Q} = \{Q \mid Q \ll P, D_{\chi^2}(Q \| P) \leq \delta, \delta \geq 0\}$$

$$D_{\chi^2}(Q\|P) = \mathbb{E}_P\left[\frac{1}{2}\left(1 - \frac{dQ}{dP}\right)^2\right] $$

Worst-case risk:

$$\mathcal{R}_{\hat{\theta}_\epsilon}(\mathcal{Q}) \triangleq \sup_{Q \in \mathcal{Q}} \{\mathbb{E}_Q[l(\hat{\theta}_\epsilon; Z)]\} $$

쌍대 형식(Duchi and Namkoong 2018):

```math
\mathcal{R}_{\hat{\theta}_\epsilon}(\eta) = \inf_{\eta \in \mathbb{R}} \left\{\sqrt{2\delta+1} \times \mathbb{E}_P[(l(\hat{\theta}_\epsilon; Z) - \eta)^2_+]^{\frac{1}{2}} + \eta\right\}
```

**Theorem 3:** $\epsilon(\phi)$가 $\sigma$-bounded gradient를 가지면, worst-case risk $\mathcal{R}\_{\hat{\theta}_\epsilon}(\eta^*)$는 영향 함수 벡터 $\vec{\phi}$에 대해 Lipschitz 연속:

$$\|\nabla_{\vec{\phi}} \mathcal{R}_{\hat{\theta}_\epsilon}(\eta^*)\| \leq \sigma \frac{\sqrt{2\delta+1}}{n} \times \sqrt{\sum_{i=1}^{n} \phi_i^2}$$

Lipschitz 상수: $\xi = \mathcal{O}\left(\sigma\frac{\sqrt{2\delta+1}}{n}\right)$

**Theorem 4 (신뢰도 대리 지표):** $\Gamma(\vec{\phi}) = \|\hat{\theta}_\epsilon - \hat{\theta}\|^2$는 Lipschitz 연속이며 Lipschitz 상수 $\xi = \mathcal{O}(\sigma\tau)$, $\tau = \frac{1}{n}$일 때 $\xi = \mathcal{O}\left(\frac{\sigma}{n}\right)$

**결론:** Data Dropout은 $\sigma \to \infty$ (불연속점에서 unbounded gradient)이므로 worst-case risk가 급격히 변동하는 반면, 확률적 샘플링은 $\alpha$(=σ) 조정을 통해 신뢰도를 제어 가능합니다.

---

### 2-4. 모델 구조 및 구현

#### UIDS 프레임워크 (Fig. 2)

```
전체 데이터셋
    ↓ (a) Full-set model 학습
    θ̂ = argmin_θ (1/n) Σ l(z_i, θ)
    ↓ (b) 영향 함수 계산
    φ⃗ = (φ(z₁,θ̂), φ(z₂,θ̂), ..., φ(zₙ,θ̂))
    ↓ (c) 샘플링 확률 계산
    π⃗ = (π(φ₁), π(φ₂), ..., π(φₙ))
    ↓ (d) Subset model 학습
    θ̃ = argmin_θ (1/|{i,Oᵢ=1}|) Σ l(zᵢ, θ)
```

#### 효율적 Hessian 계산

Mixed Preconditioned Conjugate Gradient (PCG):

$$\bar{M} = \alpha \times \text{diag}(H_\theta) + (1-\alpha) \times I $$

로지스틱 회귀에서 Hessian 대각 원소:

$$(H_\theta)_{kk} = 1 + C\sum_{i=1}^{n}(\hat{y}_i - y_i)x_{ik}^2 $$

---

### 2-5. 성능 향상 및 한계

#### 성능 향상 (Table 2, 샘플링 비율 95%)

| 데이터셋 | Full set | Random | OptLR | Dropout | Lin-UIDS | **Sig-UIDS** |
|---------|---------|--------|-------|---------|---------|------------|
| breast-cancer | 0.0914 | 0.0944 | 0.0934 | **0.0785** | 0.0873 | **0.0803** |
| Avazu-app | 0.3449 | 0.3449 | 0.3450 | 0.3576 | **0.3446** | **0.3446** |
| Avazu-site | 0.4499 | 0.4499 | 0.4505 | 0.5736 | 0.4490 | **0.4486** |
| Company (~100M) | 0.1955 | 0.1956 | 0.1958 | 0.1964 | **0.1952** | 0.1953 |

- **Sig-UIDS**: 14개 데이터셋 중 5개에서 최우수
- **Lin-UIDS & Dropout**: 각 4개에서 최우수
- **Dropout**: 이질적(heterogeneous) 대규모 데이터셋에서 실패
- **OptLR**: 가중치 분산으로 인해 심각한 성능 저하

#### 노이즈 레이블 실험 (40% 레이블 반전, Fig. 5)
UIDS 방법이 노이즈 환경에서 더욱 큰 우위를 보임

#### 효율성 (Table 3)
- 대부분 데이터셋: IF 계산 **1분 이내**
- 대규모 희소 데이터셋 (Avazu, ~25M 샘플): **10분 이내**

#### 한계점

1. **비볼록(Non-convex) 모델 미검증:** 실험이 주로 로지스틱 회귀(볼록)에 집중
2. **Full-set 사전 학습 필요:** IF 계산을 위해 전체 데이터로 먼저 모델을 학습해야 함 (2단계 프로세스)
3. **검증 세트 의존성:** 샘플링이 특정 검증 세트 $Q'$에 의존하므로 검증 세트 선택이 중요
4. **이론적 가정의 제한:** IF의 선형 근사는 작은 perturbation에서만 유효
5. **확장성 이슈:** Hessian 계산이 초대규모 딥러닝 모델에서는 여전히 비용이 클 수 있음

---

## 3. 모델 일반화 성능 향상 가능성 (중점 분석)

### 3-1. 일반화 성능 향상의 이론적 근거

UIDS가 일반화 성능을 향상시키는 핵심 메커니즘은 다음 세 가지입니다:

#### (A) 유해 샘플 제거를 통한 일반화

Lemma 2에 의해 $\phi_i > 0$인 샘플(테스트 리스크를 증가시키는 샘플)을 낮은 확률로 샘플링함으로써, 이러한 **유해 샘플들이 모델 파라미터에 미치는 부정적 영향을 감소**시킵니다.

$$\tilde{\theta} = \arg\min_\theta \frac{1}{|\{i, O_i=1\}|} \sum_{i, O_i=1} l_i(\theta)$$

이 서브셋 모델 $\tilde{\theta}$는 노이즈 레이블이나 분포 이동으로 인한 유해 샘플들의 영향을 받지 않아 **더 나은 일반화 파라미터**를 획득합니다.

#### (B) 분포 강건성(Distributional Robustness) 제어

Theorem 3의 worst-case risk 분석:
- **Dropout (결정론적):** $\sigma \to \infty$로 worst-case risk가 불안정 → 특정 $Q'$에서만 잘 작동
- **Sig/Lin-UIDS (확률적):** $\sigma = \alpha$ 조정으로 $\chi^2$-divergence ball 내 모든 분포에서 안정적

이는 실제 배포 환경에서 학습/테스트 분포가 완전히 일치하지 않을 때 **실용적 일반화 성능 향상**을 의미합니다.

#### (C) 신뢰도 대리 지표를 통한 과적합 방지

$$\Gamma(\vec{\phi}) = \|\hat{\theta}_\epsilon - \hat{\theta}\|^2$$

이 지표는 모델이 특정 검증 분포 $Q'$에 얼마나 과도하게 최적화되었는지를 측정합니다. 실험(Fig. 7)에서 Dropout은 샘플링 비율 감소에 따라 $\Gamma$가 급증하는 반면, Sig-UIDS는 Random sampling과 유사한 낮은 $\Gamma$ 값을 유지합니다.

### 3-2. 실험적 일반화 성능 검증: Tr-Va-Te 설정

논문은 일반화 능력을 엄밀하게 평가하기 위해 **Tr-Va-Te 3분할** 설정을 사용합니다:

- **Va (Validation):** IF 계산 시 기준 테스트 분포로 활용
- **Te (Test):** 서브셋 모델의 out-of-sample 일반화 성능 평가

Fig. 6에서 Data Dropout은 Va logloss는 매우 낮지만 **Te logloss가 급격히 악화**됩니다. 반면 Sig-UIDS는 두 세트에서 모두 안정적인 성능을 보입니다.

### 3-3. 노이즈 환경에서의 일반화 (Fig. 5)

40%의 레이블을 반전시킨 실험에서 UIDS는 노이즈 샘플들을 자동으로 낮은 샘플링 확률로 처리합니다. 이는 **데이터 정제 없이도 노이즈에 강건한 모델**을 학습할 수 있음을 시사합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4-1. 연구에 미치는 영향

#### (A) 데이터 중심 AI(Data-Centric AI) 패러다임 강화
이 논문은 "더 많은 데이터가 항상 더 좋다"는 통념에 도전하며, **데이터 품질이 양보다 중요**할 수 있음을 이론적으로 뒷받침합니다. 이는 데이터 선택/정제 연구의 이론적 기반을 강화합니다.

#### (B) 영향 함수 응용 연구 확장
IF를 단순한 모델 설명 도구에서 **데이터 선택 최적화 도구**로 활용하는 새로운 방향을 제시합니다.

#### (C) 분포 강건성과 서브샘플링의 통합
$\chi^2$-divergence를 이용한 worst-case risk 분석은 **분포 강건 최적화(DRO)**와 서브샘플링을 연결하는 이론적 다리 역할을 합니다.

#### (D) 지속 학습(Continual Learning) / 코어셋(Coreset) 연구와의 연계
노이즈 제거와 정보 밀도 높은 서브셋 선택 아이디어는 코어셋 선택, 액티브 러닝, 커리큘럼 러닝 등의 연구에 영향을 미칩니다.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

#### ① Data-Centric AI / Dataset Pruning 연구

| 연구 | 방법 | UIDS와의 관계 |
|------|------|--------------|
| **Forgetting Events** (Toneva et al., 2019 / 이후 인용 증가) | 학습 중 예측이 변하는 샘플 추적 | UIDS의 유해 샘플 제거와 유사한 철학 |
| **Dataset Distillation / Condensation** (Zhao et al., 2021; Kim et al., 2022) | 원본 데이터 분포를 작은 합성 데이터셋으로 증류 | UIDS와 달리 원본 샘플을 변형; 더 공격적인 압축 가능 |
| **Beyond neural scaling laws** (Sorscher et al., 2022, NeurIPS) | 데이터 프루닝이 파워 법칙을 개선할 수 있음을 증명 | UIDS의 "less is better" 주장을 대규모에서 재확인 |

> **Sorscher et al. (2022)** "Beyond neural scaling laws: beating power law scaling via data pruning," *NeurIPS 2022*

#### ② 영향 함수 개선 연구

| 연구 | 방법 | UIDS와의 관계 |
|------|------|--------------|
| **FastIF** (Guo et al., 2021) | IF 계산 속도 개선 | UIDS의 PCG 방법을 보완하는 더 빠른 대안 |
| **Influence Function 재현성 연구** (Basu et al., 2021) | 비볼록 모델에서 IF 신뢰성 문제 제기 | UIDS의 비볼록 시나리오 한계를 실증적으로 확인 |
| **DataInf** (Kwon et al., 2023) | 대규모 언어 모델에서 효율적 IF 계산 | UIDS를 LLM 스케일로 확장하는 방향 제시 |

> **Basu et al. (2021)** "Influence Functions in Deep Learning Are Fragile," *ICLR 2021*

#### ③ 분포 강건 최적화(DRO) 연구

| 연구 | 방법 | UIDS와의 관계 |
|------|------|--------------|
| **Group DRO** (Sagawa et al., 2020, ICLR) | 그룹별 최악 케이스 리스크 최소화 | UIDS의 χ²-divergence ball 분석과 이론적 연계 |
| **Just Train Twice (JTT)** (Liu et al., 2021, ICML) | 잘못 분류된 샘플 업샘플링 | UIDS의 영향 기반 선택과 반대 방향(업샘플링) |

> **Sagawa et al. (2020)** "Distributionally Robust Neural Networks for Group Shifts," *ICLR 2020*
> **Liu et al. (2021)** "Just Train Twice: Improving Group Robustness without Training Group Information," *ICML 2021*

#### ④ 대규모 언어 모델(LLM) 데이터 선택 연구

| 연구 | 방법 | UIDS와의 관계 |
|------|------|--------------|
| **LIMA** (Zhou et al., 2023) | 1000개 고품질 데이터로 LLM 파인튜닝 | "less is better" 원리의 LLM 적용 |
| **DSIR** (Xie et al., 2023) | 중요도 재가중치로 프리트레이닝 데이터 선택 | UIDS의 영향 기반 선택을 LLM 사전학습으로 확장 |
| **Instruction Mining** (Cao et al., 2023) | 명령어 파인튜닝 데이터 품질 필터링 | UIDS 철학의 LLM 적용 |

> **Zhou et al. (2023)** "LIMA: Less Is More for Alignment," *NeurIPS 2023*

---

### 4-3. 앞으로 연구 시 고려할 점

#### 🔴 단기적 고려사항

1. **비볼록 모델(Deep Neural Networks)로의 확장**
   - Basu et al. (2021)이 지적한 비볼록 모델에서 IF 신뢰성 문제 해결 필요
   - 스토캐스틱 IF 추정 방법(예: EK-FAC, DataInf)과의 결합 검토

2. **검증 세트 선택 자동화**
   - 현재 Va 세트에 과도하게 의존; 검증 세트 품질이 서브샘플링 품질에 직접적 영향
   - Meta-learning 기반 자동 Va 구성 방법 연구 필요

3. **샘플링 비율의 이론적 최적화**
   - 현재 샘플링 비율은 경험적으로 설정(예: 95%); 이론적 최적 비율 결정 방법 필요

#### 🟡 중장기적 고려사항

4. **LLM 사전학습/파인튜닝 데이터 선택**
   - LIMA, DSIR 등의 연구와 UIDS를 결합하여 LLM 규모에서의 유효성 검증
   - IF 계산 비용을 DataInf 등으로 대체하여 확장성 확보

5. **연속 학습(Continual Learning) / 온라인 설정으로 확장**
   - 현재는 오프라인 배치 방식; 실시간 데이터 스트림에서의 점진적 서브샘플링 연구

6. **공정성(Fairness) 관점 통합**
   - IF 기반 샘플 제거가 특정 인구통계 그룹을 과소표현할 위험성 분석
   - Group DRO와의 결합으로 공정하고 강건한 서브샘플링 프레임워크 개발

7. **다중 검증 분포에 대한 견고성**
   - 단일 Va 세트 의존에서 벗어나 여러 분포를 동시에 고려하는 프레임워크로 발전

#### 🟢 방법론적 개선사항

8. **IF 추정 정확도와 샘플링 품질 간의 트레이드오프 분석**
   - 근사 오차가 서브셋 모델 성능에 미치는 영향 이론화

9. **분류 외 다른 태스크(생성, 회귀, 순위 학습)로의 확장**

10. **샘플링 함수의 이론적 최적 형태 탐색**
    - Linear, Sigmoid 외 더 최적화된 $\pi(\phi)$ 함수 설계 (예: 볼록 함수 제약 하의 최적화)

---

## 참고자료

**주요 참고 논문 (논문 내 인용):**
- Wang et al. (2020). "Less Is Better: Unweighted Data Subsampling via Influence Function." *AAAI 2020*. arXiv:1912.01321v3
- Koh, P.W. & Liang, P. (2017). "Understanding black-box predictions via influence functions." *ICML 2017*
- Ting, D. & Brochu, E. (2018). "Optimal subsampling with influence functions." *NeurIPS 2018*
- Duchi, J.C. & Namkoong, H. (2018). "Learning models with uniform performance via distributionally robust optimization." arXiv:1810.08750
- Wang, T., Huan, J. & Li, B. (2018). "Data dropout: Optimizing training data for convolutional neural networks." *ICTAI 2018*

**비교 분석에 활용된 2020년 이후 연구:**
- Basu, S. et al. (2021). "Influence Functions in Deep Learning Are Fragile." *ICLR 2021*
- Sagawa, S. et al. (2020). "Distributionally Robust Neural Networks for Group Shifts: On the Importance of Spurious Correlations." *ICLR 2020*
- Liu, E.Z. et al. (2021). "Just Train Twice: Improving Group Robustness without Training Group Information." *ICML 2021*
- Sorscher, B. et al. (2022). "Beyond neural scaling laws: beating power law scaling via data pruning." *NeurIPS 2022*
- Zhou, C. et al. (2023). "LIMA: Less Is More for Alignment." *NeurIPS 2023*

> ⚠️ **주의:** 2020년 이후 연구 비교 분석 부분은 해당 논문 원문에 포함된 내용이 아니며, 논문의 주제와 관련된 후속 연구들을 저의 지식 범위 내에서 분석한 것입니다. 일부 세부 사항은 확인이 필요할 수 있습니다.
