# Differentially Private Optimal Transport: Application to Domain Adaptation 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
이 논문은 **최적 수송(Optimal Transport, OT)**을 이용한 도메인 적응(Domain Adaptation) 과정에서 소스/타겟 데이터의 프라이버시를 보호하는 최초의 차분 프라이버시(Differential Privacy) 기반 프레임워크를 제안합니다.

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| **DPOT** | 최초의 차분 프라이버시 최적 수송 알고리즘 |
| **DPDA** | 최초의 완전한 차분 프라이버시 도메인 적응 알고리즘 |
| **이론적 보장** | $(\varepsilon, \delta)$-DP 보장 및 Wasserstein 거리 근사 오차 분석 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**기존 OT 기반 도메인 적응의 문제점:**
- 소스-타겟 도메인 간 데이터를 공유해야 커플링 행렬(coupling matrix)을 계산 가능
- 의료 기록, 개인 정보 등 민감한 데이터가 포함될 경우 데이터 공유 자체가 불가능
- 기존 프라이버시 보존 방법들 (Wang et al., 2018; Guo et al., 2018)은 공개 보조 데이터셋 필요 또는 완전히 레이블된 타겟 데이터 요구 → 비현실적 가정

**목표:**
$$\text{Privacy} + \text{Optimal Transport} + \text{Domain Adaptation을 동시에 달성}$$

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) 차분 프라이버시 정의

$$\mathbb{P}(\mathcal{M}(X) \in \mathcal{O}) \leq e^{\varepsilon} \mathbb{P}(\mathcal{M}(X') \in \mathcal{O}) + \delta$$

- $\mathcal{M}$: 랜덤화 메커니즘
- $X, X'$: 단 하나의 원소만 다른 두 데이터셋
- $(\varepsilon, \delta)$: 프라이버시 예산

#### (B) 표준 OT 문제 (Kantorovich 공식)

$$W(\mu_1, \mu_2) = \min_{\gamma \in \Pi} \int_{\Omega_s \times \Omega_t} c(x^s, x^t) d\gamma(x^s, x^t)$$

경험적 측도로 이산화하면:

$$\gamma_0 = \underset{\gamma \in \Pi}{\arg\min} \langle \gamma, C \rangle_F$$

여기서 $C$는 소스-타겟 간 비용 행렬(유클리드 거리 행렬), $\langle \cdot, \cdot \rangle_F$는 Frobenius 내적.

#### (C) Johnson-Lindenstrauss (JL) 변환 (Theorem 1)

$v_1, \ldots, v_n \in \mathbb{R}^k$에 대해 $\mathcal{N}(0, \frac{1}{\ell})^{k \times \ell}$ 랜덤 행렬 $M$을 이용하면, 확률 $1 - \frac{2}{\exp(\ell\eta^2/8)}$ 이상으로 다음이 성립:

$$1 - \eta \leq \frac{\|v_i^T M - v_j^T M\|_2^2}{\|v_i - v_j\|_2^2} \leq 1 + \eta, \quad \eta \in [0, 0.5]$$

즉, 랜덤 투영이 쌍별 거리를 $(1 \pm \eta)$ 비율로 보존함.

#### (D) DPOT 알고리즘 핵심 수식

**Step 3**: 노이즈가 추가된 비용 행렬 계산:

$$\tilde{C} = c(\tilde{X}_s + \Delta, \tilde{X}_t) - \ell\sigma^2$$

- $\tilde{X}_s = X_s M$: 소스 데이터의 랜덤 투영
- $\tilde{X}_t = X_t M$: 타겟 데이터의 랜덤 투영
- $\Delta \sim \mathcal{N}(0, \sigma)^{k \times \ell}$: 프라이버시 노이즈
- $\ell\sigma^2$: $\Delta$에 의한 편향 보정

**DPOT 프라이버시 보장 (Theorem 2):**

$$\text{Algorithm 1은 임의의 } \varepsilon, \delta > 0 \text{ 및 } \sigma \geq w\frac{\sqrt{2(\ln(\frac{1}{2\delta})+\varepsilon)}}{\varepsilon} \text{에 대해 } (\varepsilon, \delta)\text{-DP}$$

여기서 $w = \max_{1 \leq i \leq k} \left(\sum_{j=1}^{\ell} M_{ij}^2\right)^{1/2}$는 $M$의 $\ell_2$-norm sensitivity.

#### (E) DPDA: 정규화된 커플링 행렬 계산

$$\tilde{\gamma}_0 = \underset{\gamma \in \Pi}{\arg\min} \langle \gamma, \tilde{C} \rangle_F + \lambda_e R_e(\gamma) + \lambda_g R_g(\gamma)$$

**엔트로픽 정규화 (Cuturi, 2013):**

$$R_e(\gamma) = -\sum_{i,j} \gamma_{ij}(\log \gamma_{ij} - 1)$$

**그룹 Lasso 정규화 (Courty et al., 2017b):**

$$R_g(\gamma) = \sum_j \sum_s \|\gamma(I_s, j)\|_2$$

- $I_s$: 레이블 $s$에 해당하는 소스 샘플 인덱스
- 동일 레이블의 소스 샘플에서 타겟 샘플로 질량이 이동하도록 유도

#### (F) 바리센트릭 매핑 (Private Barycentric Mapping)

$$\hat{x}_i^s = \underset{x \in \mathbb{R}^k}{\arg\min} \sum_j \tilde{\gamma}_0[i,j] c(x, x_j^t)$$

행렬 형태로 표현:

$$\hat{X}_s = n_s \tilde{\gamma}_0 X_t$$

소스 원본 데이터 $X_s$에 의존하지 않으므로 **프라이버시가 자동 보장**.

#### (G) 소스 레이블 전송 (Histogram Query)

$$v(Y_s) + \text{Lap}\left(\frac{1}{\varepsilon'}\right)^q$$

라플라스 노이즈 추가로 $(\varepsilon', 0)$-DP 보장.

#### (H) DPDA 전체 프라이버시 보장 (Theorem 3, Composition Theorem)

$$\text{Algorithm 2는 } (\varepsilon + \varepsilon', \delta)\text{-DP}$$

---

### 2.3 모델 구조

```
[Source Domain]          [Target Domain]
   (Xs, Ys)                   Xt
      |                        |
  JL 변환 M              JL 변환 M
  노이즈 Δ 추가          (동일한 M 사용)
      |                        |
  X̃s + Δ 공개  -------→  C̃ 계산 (편향 보정)
                               |
                         γ̃₀ 최적화 (식 3)
                               |
                    X̂s = ns·γ̃₀·Xt (바리센트릭)
                               |
                    (X̂s, Ỹs)로 분류기 학습
                               |
                          Target Model
```

---

### 2.4 성능 결과

#### Office-Caltech (비지도, 평균 정확도 %)

| 방법 | 평균 |
|------|------|
| PATE | 80.2 |
| **DPDA** | **91.2** |
| OTDA (비프라이버시) | 93.0 |
| ADDA (비프라이버시) | 92.3 |

#### VisDA (비지도, 평균 정확도 %)

| 방법 | 평균 |
|------|------|
| PATE | 60.6 |
| **DPDA** | **68.8** |
| OTDA | 69.1 |
| ADDA | 67.3 |

> **주목할 점**: DPDA는 프라이버시를 보장하면서도 비프라이버시 방법인 ADDA보다 VisDA에서 1.5점 높은 성능을 달성.

#### Office-Home 반지도 설정

| 방법 | 평균 |
|------|------|
| **DPDA (반지도)** | **60.0** |
| OTDA (반지도) | 61.2 |
| DPDA (비지도) | 54.8 |
| OTDA (비지도) | 57.9 |

반지도에서 DPDA와 OTDA의 격차가 **3점 → 1점으로 감소** (일반화 성능 향상의 핵심 증거).

#### DPOT Wasserstein 거리 근사 오차

| 설정 | $\varepsilon$ | 오차 (Err) |
|------|--------------|------------|
| 전체 배치 | 10 | 8% |
| 전체 배치 | 4 | 22% |
| 미니배치 ($\sigma_w=0.7$) | 7 | 7% |
| 미니배치 ($\sigma_w=1.0$) | 5 | 10% |

---

### 2.5 한계점

1. **프라이버시-정확도 트레이드오프**: $\varepsilon$이 작을수록 노이즈 $\sigma$가 커지고 Wasserstein 거리 근사 오차가 증가 (Office-Caltech에서 $\varepsilon=4$이면 오차 22%)
2. **소규모 데이터셋 취약성**: 샘플 수가 적은 도메인(DSLR, Webcam: 150~200개)에서는 높은 $\varepsilon$ 필요 ($\varepsilon=20$)
3. **계산 복잡도**: JL 변환 및 OT 최적화의 추가 연산 비용
4. **단방향 프라이버시**: 현재 알고리즘은 소스 데이터 보호에 초점; 타겟 데이터의 완전한 양방향 프라이버시 보장은 추가 분석 필요
5. **고차원 데이터**: 특징 차원 $k$가 매우 크면 JL 투영 후 차원 $\ell = k/10$ 설정에서도 계산 비용이 높아질 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 근거: Wasserstein 거리 기반 일반화 경계

Redko et al. (2017)에 따르면 Wasserstein 거리를 이용한 타겟 도메인의 오류 상계는:

$$\varepsilon_T(h) \leq \varepsilon_S(h) + W(\mu_S, \mu_T) + \lambda^*$$

- $\varepsilon_T(h)$: 타겟 도메인 오류
- $\varepsilon_S(h)$: 소스 도메인 오류
- $W(\mu_S, \mu_T)$: Wasserstein 거리 (분포 간 불일치)
- $\lambda^*$: 이상적 공동 분류기 오류 (도메인 적응 불가능성 항)

DPOT가 $\tilde{W}(X_s, X_t) \approx W(X_s, X_t)$를 보장하므로, **프라이버시를 보장하면서도 이론적 일반화 경계를 유지**할 수 있음.

### 3.2 실험적 증거: 반지도 학습에서의 일반화 향상

Office-Home 반지도 설정에서 클래스당 레이블 타겟 샘플 1개 허용 시:

$$\text{Gap}(\text{DPDA vs OTDA}): \underbrace{57.9 - 54.8}_{\text{비지도}} = 3.1\text{점} \rightarrow \underbrace{61.2 - 60.0}_{\text{반지도}} = 1.2\text{점}$$

이는 **소량의 타겟 레이블 정보가 DPDA의 일반화 성능을 OTDA 수준으로 빠르게 끌어올림**을 의미. 즉 DPDA가 레이블 정보를 더 효율적으로 활용하는 경향이 있음.

### 3.3 일반화 향상 메커니즘 분석

```
1. 그룹 Lasso 정규화 → 레이블 일관성 강제
   → 동일 클래스 샘플 간 질량 이동 우선
   → 더 의미론적(semantic)인 정렬

2. 엔트로픽 정규화 → 커플링 행렬 평활화
   → 과적합 방지, 더 분산된 매핑
   → 일반화 성능에 기여

3. 미니배치 서브샘플링 → 프라이버시 증폭
   → 더 낮은 ε에서 좋은 Wasserstein 근사
   → 프라이버시-유틸리티 트레이드오프 개선
```

### 3.4 한계와 잠재적 향상 방향

- **현재**: JL 변환의 무작위성이 경우에 따라 유용한 기하학적 구조를 파괴할 수 있음
- **향후 가능성**: 적응적 DP 노이즈 주입(데이터 민감도 기반) 또는 프라이버시 보장 딥 피처 추출기와의 결합으로 일반화 성능 추가 향상 가능

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### 즉각적 영향
1. **프라이버시 보존 전이 학습의 체계화**: OT + DP의 결합이라는 새로운 연구 방향 개척
2. **의료·금융 도메인 적용 가능성**: 환자 데이터나 금융 거래 데이터의 안전한 도메인 간 지식 전이
3. **분산 학습 프라이버시**: 연합학습(Federated Learning)에서 OT 기반 분포 정렬의 프라이버시 보장에 직접 응용 가능

#### 장기적 영향
1. **프라이버시 보존 생성 모델**: DP + Wasserstein GAN 결합 연구의 이론적 토대 제공
2. **멀티소스 도메인 적응**: 여러 소스 도메인의 프라이버시를 동시에 보장하는 OT 확장
3. **공정성(Fairness)과 프라이버시의 결합**: OT 기반 공정성 제약과 DP의 결합

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 연구들은 제 학습 데이터(2020~2021년경까지) 기반 지식으로, 일부 세부 내용(특히 정확한 수치)은 직접 논문 검색을 통해 확인하시기 바랍니다.

### 5.1 관련 연구 흐름

#### (1) 프라이버시 보존 OT 확장 연구 방향

| 연구 방향 | 내용 | 이 논문 대비 발전 |
|-----------|------|------------------|
| Federated OT | 분산 환경에서 OT 계산 | 다중 당사자 시나리오 |
| DP + Sliced Wasserstein | 더 효율적인 1D 투영 기반 OT | 계산 복잡도 감소 |
| DP + Sinkhorn | Sinkhorn 알고리즘에 DP 적용 | 수렴 속도 개선 |

#### (2) 도메인 적응의 발전

**딥러닝 기반 OT 도메인 적응 (2020~):**
- **DeepJDOT** (Damodaran et al.): 딥 피처 공간에서 직접 OT 수행
- **JUMBOT** (Fatras et al., 2021): 미니배치 OT의 이론적 분석 강화

이 논문의 DPDA는 얕은 피처(DeCAF, NASNet 추출 피처)에 OT를 적용하는 반면, 최신 연구들은 **end-to-end 딥러닝 파이프라인 내에서 OT를 직접 최적화**하는 방향으로 발전.

#### (3) 프라이버시 예산 효율화

| 방법 | 프라이버시 증폭 메커니즘 |
|------|-------------------------|
| 이 논문 | 미니배치 서브샘플링 |
| Balle et al. (2018) | 쿠플링 기반 tight한 서브샘플링 분석 |
| Mironov et al. (2017) | Rényi DP → 더 tight한 구성 |

**Rényi Differential Privacy (RDP)** 적용 시 이 논문의 Composition Theorem보다 더 tight한 프라이버시 예산 추적 가능.

### 5.2 한계 극복을 위한 향후 연구 방향

```
1. DP + 딥러닝 특징 추출 통합
   → 현재: 사전 학습된 피처 사용 후 DPOT 적용
   → 개선: DP-SGD로 특징 추출기 자체를 프라이버시 보장하며 학습

2. Rényi DP 또는 f-DP 적용
   → 현재: (ε,δ)-DP Composition
   → 개선: 더 tight한 프라이버시 예산 계산으로 유틸리티 향상

3. 적응적 노이즈 메커니즘
   → 현재: 고정 σ_w 사용
   → 개선: 데이터 민감도에 따른 적응적 노이즈 → 정확도 향상

4. 멀티소스 DPDA
   → 현재: 단일 소스-타겟 쌍
   → 개선: 여러 소스 도메인의 프라이버시 동시 보장

5. 연합학습(Federated Learning)과의 결합
   → 클라이언트별 로컬 OT + 중앙 집계 프라이버시 보장
```

---

## 참고 자료

### 논문에서 직접 인용된 문헌
1. **LeTien, Habrard, Sebban (2019)**. "Differentially Private Optimal Transport: Application to Domain Adaptation." *IJCAI-19*, pp. 2852-2858.
2. **Dwork et al. (2006)**. "Calibrating noise to sensitivity in private data analysis." *TCC*, Springer.
3. **Johnson & Lindenstrauss (1984)**. "Extensions of Lipschitz mappings into a Hilbert space." *Contemporary Mathematics*.
4. **Kenthapadi et al. (2013)**. "Privacy via the Johnson-Lindenstrauss transform." *Journal of Privacy and Confidentiality*, 5(1).
5. **Courty et al. (2017b)**. "Optimal transport for domain adaptation." *IEEE TPAMI*, 39(9):1853-1865.
6. **Courty et al. (2017a)**. "Joint distribution optimal transportation for domain adaptation." *NIPS*.
7. **Redko et al. (2017)**. "Theoretical analysis of domain adaptation with optimal transport." *ECML-PKDD*.
8. **Abadi et al. (2016)**. "Deep learning with differential privacy." *CCS*, ACM.
9. **Papernot et al. (2017)**. "Semi-supervised knowledge transfer for deep learning from private training data." *ICLR*.
10. **Cuturi (2013)**. "Sinkhorn distances: Lightspeed computation of optimal transport." *NIPS*.
11. **Blocki et al. (2012)**. "The Johnson-Lindenstrauss transform itself preserves differential privacy." *FOCS*, IEEE.
12. **Balle et al. (2018)**. "Privacy amplification by subsampling." *NIPS*.
13. **Tzeng et al. (2017)**. "Adversarial discriminative domain adaptation." *CVPR*.
14. **Perrot et al. (2016)**. "Mapping estimation for discrete optimal transport." *NIPS*.
15. **Dwork et al. (2014)**. "The algorithmic foundations of differential privacy." *Foundations and Trends in Theoretical Computer Science*, 9(3-4).
