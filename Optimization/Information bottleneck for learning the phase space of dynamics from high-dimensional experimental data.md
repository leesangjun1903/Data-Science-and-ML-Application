# Information Bottleneck for Learning the Phase Space of Dynamics from High-Dimensional Experimental Data

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **DySIB (Dynamical Symmetric Information Bottleneck)** 를 제안하여, 고차원 실험 데이터(예: 비디오)로부터 감독(supervision) 없이 물리 시스템의 위상 공간(phase space) 좌표를 자동으로 학습할 수 있음을 보입니다.

핵심 아이디어는 다음과 같습니다:

> **"과거의 압축된 잠재 표현이 미래를 최대한 잘 예측하도록 강제하면, 그 잠재 표현은 시스템의 동역학적 상태 변수가 된다."**

### 주요 기여

| 기여 | 설명 |
|---|---|
| **DySIB 프레임워크 제안** | DVSIB를 동역학 시스템에 확장한 새로운 자기지도(self-supervised) 학습 방법 |
| **잠재 공간 내 예측** | 데이터 공간 재구성 없이 순수하게 잠재 공간에서만 예측 수행 |
| **자기일관적 하이퍼파라미터 선택** | $k_z$, $n_F$ 를 데이터로부터 자동 결정 |
| **실험 검증** | 실제 진자 비디오에서 위상 공간 위상(topology)·기하(geometry)·차원 복원 성공 |
| **$\delta$-predictor 도입** | 물리적 미분 구조(differential structure)를 아키텍처에 귀납적 편향으로 내재화 |

---

## 2. 자세한 분석

### 2-1. 해결하고자 하는 문제

물리 시스템은 소수의 상태 변수로 기술되지만, 실험 관측은 고차원(예: 비디오 픽셀)입니다. 기존 접근법의 한계는 다음과 같습니다:

- **오토인코더 기반 방법**: 재구성(reconstruction)에 필요한 정보 ≠ 동역학에 필요한 정보
- **데이터 공간 예측 방법 (GPT류)**: 물리적 잠재 변수가 아닌 다음 프레임을 예측 (뉴턴 법칙은 다음 프레임이 아닌 잠재 변수의 미래를 기술함)
- **기존 잠재 예측 방법**: 물리적 미분 구조 미반영, 저차원·해석 가능 표현에 집중하지 않음

따라서 **비지도·완전 자기지도적으로 잠재 공간에서 예측 가능한 저차원 표현을 학습**하는 방법이 필요합니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (a) Information Bottleneck 배경

표준 IB 손실함수:

$$\mathcal{L}_{\text{IB}} = I(X; Z) - \beta\, I(Z; Y) \tag{1}$$

Symmetric IB (SIB):

$$\mathcal{L}_{\text{SIB}} = I(X; Z_X) + I(Y; Z_Y) - \beta\, I(Z_X; Z_Y) \tag{3}$$

#### (b) DySIB의 변분 인코더

과거 윈도우 $X$와 미래 윈도우 $Y$를 다음과 같이 정의:

$$X = \{F_t, F_{t+1}, \ldots, F_{t+n_F-1}\}, \quad Y = \{F_{t+n_s}, \ldots, F_{t+n_s+n_F-1}\} \tag{6}$$

변분 인코더는 대각 가우시안으로 파라미터화:

$$q(z_x | x) = \mathcal{N}\!\left(\mu(x),\, \text{diag}(\exp(\ell(x)))\right) \tag{7}$$

인코더 압축항 (KL divergence):

$$\tilde{I}^E(X; Z_X) = \frac{1}{B}\sum_{i=1}^{B} D_{\text{KL}}\!\left(q(z_x | x_i) \,\|\, \mathcal{N}(0, I)\right) \tag{14}$$

#### (c) $\delta$-predictor

물리적 미분 구조를 반영하여, 과거 잠재 변수의 작은 잔차 업데이트로 미래를 예측:

$$z_y^{\text{pred}} = z_x + \mu_\delta(z_x) \tag{9}$$

$$r(z_y | z_x) = \mathcal{N}\!\left(z_y^{\text{pred}},\, \text{diag}(\exp(\ell_\delta(z_x)))\right) \tag{10}$$

#### (d) InfoNCE 추정량 (예측 상호정보 하한)

배치 내 크리틱 함수:

$$s(z_{x,i}, z_{y,j}) = \log r(z_{y,j} | z_{x,i}) = -\frac{1}{2}\sum_{d=1}^{k_z}\left[\frac{(z_{y,j,d} - z_{y,i,d}^{\text{pred}})^2}{\exp(\ell_{\delta,d}(z_{x,i}))} + \ell_{\delta,d}(z_{x,i}) + \log 2\pi\right] \tag{11}$$

$$\tilde{I}_{\text{NCE}}(Z_X; Z_Y) = \frac{1}{B}\sum_{i=1}^{B} \log\!\left(\frac{e^{s(z_{x,i}, z_{y,i})}}{\frac{1}{B}\sum_{j=1}^{B} e^{s(z_{x,i}, z_{y,j})}}\right) \tag{12}$$

#### (e) DySIB 최종 손실함수

$$\mathcal{L}_{\text{DySIB}} = \tilde{I}^E(X; Z_X) + \tilde{I}^E(Y; Z_Y) - \beta\, \tilde{I}_{\text{NCE}}(Z_X; Z_Y) \tag{13}$$

---

### 2-3. 모델 구조

```
비디오 프레임 시퀀스
        ↓
공유 프레임 인코더 Φ (3-layer MLP, 256 hidden, ReLU)
        ↓
지연 임베딩 연결 (delayed embedding)
        ↓
Ψ_μ, Ψ_ℓ (선형 헤드) → μ(x), ℓ(x)
        ↓
재파라미터화 트릭 → Z_X ~ q(z|x)
        ↓
δ-predictor (3-layer MLP, 64 hidden, ReLU)
Z_X → Z_X + μ_δ(Z_X)  [잔차 업데이트]
        ↓
InfoNCE 추정기 → I(Z_X; Z_Y) 최대화
```

**주요 설계 원칙:**
- **공유 인코더**: 과거·미래에 동일한 인코더 사용 → 시간 이동 불변성(time-translation invariance)
- **잠재 공간 내 예측 전용**: 원본 데이터 재구성 없음 (JEPA 접근법과 유사)
- **β = 100** 고정: 예측 항이 지배적이고 KL은 약한 정규화로만 작동

---

### 2-4. 성능 향상

| 지표 | 결과 |
|---|---|
| 잠재 차원 $k_z = 2$ 선택 | MI 포화 기준으로 자동 선택, 진자의 실제 자유도와 일치 |
| 시간 윈도우 $n_F = 2$ 선택 | 각속도 추정에 최소 2프레임 필요, 자동 검출 |
| $\theta$ RMSE | ~ $1.6^\circ$ (RF-linear probe, $k_z=2$ ) |
| $\omega$ RMSE | 소수 수백 degrees/s 수준 (256 videos에서 수렴) |
| 표본 효율성 | 각도 복원에 ~4–8개 비디오, 각속도 복원에 ~256개 비디오 |
| 오버파라미터화 ($k_z=8$) | 유효 차원은 여전히 2, 시드 간 분산 크게 감소 |

---

### 2-5. 한계

논문이 명시적으로 인정하는 한계:

1. **검증 시스템의 단순성**: 진자는 저잡음, 결정론적, 저차원 시스템 — 카오스, 강한 노이즈, 다중 스케일 시스템에서의 성능은 미검증
2. **표현의 비유일성 (gauge freedom)**: 잠재 공간은 매끄러운 역변환 하에서 불변 → 두 런의 잠재 공간을 직접 비교하기 어려움 (비선형 ICA의 비식별성 문제와 동일)
3. **아키텍처 단순성**: 단순 MLP 사용 — 복잡한 시스템에는 CNN, Transformer 등이 필요할 수 있음
4. **InfoNCE 상한**: 배치 크기 $B$에 의해 $\log B$로 제한됨 (B=1024이면 약 10 bits)
5. **양적 정확도 미보장**: 위상 구조와 위상학적 구조는 복원되지만, 주기나 보존량의 정량적 정확도는 보장하지 않음
6. **$\delta$-predictor의 가정**: 작은 미분적 업데이트 가정 → 동역학이 미분방정식으로 자연스럽게 기술되지 않는 시스템에는 부적합할 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 표본 효율성 (Sample Efficiency)

논문은 잠재 공간 기반 예측이 데이터 공간 기반 생성 모델보다 **표본 효율적**임을 이론적·실험적으로 지지합니다:

$$\tilde{I}_{\text{NCE}}(Z_X; Z_Y) \sim \log_2 N$$

MI가 $\log_2 N$으로 증가한다는 실험 결과는, **새로운 초기 조건(궤적)의 수**가 증가할수록 정보가 증가함을 의미합니다. 이는 프레임 쌍의 총 수가 아닌, 독립적인 초기 조건의 다양성이 중요함을 시사합니다.

### 3-2. 오버파라미터화 보틀넥의 효과

$k_z > k_z^*$ (실제 자유도보다 큰 잠재 차원)를 사용하면:

- **유효 차원은 변하지 않음** (TwoNN, Levina-Bickel 추정기로 확인: $d_{\text{eff}} \approx 2$)
- **시드 간 분산이 크게 감소** → 훈련 안정성 향상
- **probe 성능이 동등하거나 향상** ($k_z=8$에서 RF-linear probe가 MLP probe만큼 좋거나 더 좋음)

이는 실용적인 일반화 전략을 제공합니다:

> **" $k_z$를 과도하게 크게 설정한 후, 내재 차원을 추정하여 실제 자유도를 읽어낸다."**

### 3-3. 시간 이동 불변성의 귀납적 편향

공유 인코더(shared encoder) 사용은:
- 파라미터 수를 줄여 과적합 방지
- 시간에 따른 일반화 능력 향상

### 3-4. $\delta$-predictor의 물리적 귀납적 편향

잔차 업데이트 구조:
$$z_y^{\text{pred}} = z_x + \mu_\delta(z_x)$$

이는 ResNet의 잔차 연결과 유사하여 훈련을 안정화하고, 물리적으로 타당한 (연속적이고 미분 가능한) 궤적을 학습하도록 강제합니다. 이 편향은 작은 데이터에서도 물리적으로 의미 있는 표현을 학습하는 데 기여합니다.

### 3-5. 일반화 가능성의 한계와 향후 방향

일반화 성능을 더 향상시키기 위해 논문이 제안하는 방향:

1. **더 강한 노이즈, 카오스 시스템**으로 확장
2. **심볼릭 회귀(symbolic regression)와 결합**: 학습된 잠재 좌표에서 운동 방정식 자동 발견
3. **정규화 흐름(normalizing flow)** 도입으로 표준 좌표계 학습
4. **더 깊은 비선형 헤드** ($\Psi_\mu$, $\Psi_\ell$) 사용으로 복잡한 동역학 처리

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 앞으로의 연구에 미치는 영향

#### (A) 물리 과학에서의 AI 활용

DySIB는 **랑다우 프로그램의 데이터 기반 대응물**로서 위치합니다:

> "랑다우는 깨진 대칭으로부터 질서 매개변수를 찾는다. DySIB는 자신의 미래를 예측한다는 요건으로부터 질서 매개변수를 찾는다."

이는 대칭성이나 보존법칙을 a priori로 알 수 없는 시스템(유전자 조절 네트워크, 신경과학, 동물 행동 등)에 적용 가능한 일반적 프레임워크를 제공합니다.

#### (B) 자기지도 표현 학습 커뮤니티에의 영향

- JEPA (Joint Embedding Predictive Architecture) 계열 방법들과의 연결을 강화
- 물리적 해석 가능성을 자기지도 학습의 평가 기준으로 도입
- 잠재 차원의 자기일관적 결정 방법을 제공

#### (C) 신경과학 및 복잡계 연구

- 뉴런 집단 활동에서 저차원 동역학 변수 추출에 직접 응용 가능
- 동물 행동 데이터에서 위상 공간 구조 발견에 활용 가능

### 4-2. 앞으로 연구 시 고려할 점

#### 이론적 측면

1. **비선형 ICA의 비식별성 문제 해결**: 잠재 공간의 표준형(canonical form)을 정의하는 방법 개발 필요. 논문은 전이 연산자의 선형 부분을 대각화하는 정규 모드 좌표계를 후보로 제시
2. **InfoNCE 상한 극복**: 배치 크기에 의존하지 않는 MI 추정량 사용 또는 더 큰 배치 활용
3. **이론적 보장 강화**: 어떤 조건에서 DySIB가 진정한 충분 통계량을 복원하는지 증명

#### 실용적 측면

4. **확장성**: 더 복잡한 인코더 아키텍처 (CNN, Vision Transformer) 도입으로 고해상도, 복잡한 장면 처리
5. **카오스 및 확률적 시스템**: 결정론적 진자 이상의 시스템에서 검증 필요
6. **다중 스케일 분리**: 시간 스케일이 분리된 시스템에서의 적용성 연구
7. **심볼릭 통합**: 학습된 잠재 좌표를 SINDy(Brunton et al., 2016) 등의 심볼릭 회귀와 결합
8. **부분 관측(partial observability)**: 상태 변수 일부만 관측 가능한 경우의 처리

#### 평가 방법론

9. **비지도 검증 지표**: 실제 물리 변수를 모를 때의 평가 방법 개발 (out-of-distribution 예측, 내적 일관성 등)
10. **다른 동역학 시스템과의 벤치마크**: 로렌츠 어트랙터, 반응-확산 시스템 등에서의 체계적 비교

---

## 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 접근법 | DySIB와의 차이 | 출처 |
|---|---|---|---|
| **SINDy (Brunton et al.)** | 희소 회귀로 방정식 발견 | 상태 변수를 이미 알아야 함 | PNAS 2016 |
| **Chen et al. (2022)** | 자동인코더로 숨겨진 변수 발견 | 재구성 기반, 동역학과 무관한 정보 포함 | Nature Computational Science 2022 [29] |
| **Schmitt et al. (2023/2024)** | 정보이론적 차원 축소 | 대칭 압축 없음, $\delta$-predictor 없음 | arXiv:2312.06608 [44,45] |
| **JEPA (LeCun 계열)** | 잠재 공간 예측 | 해석 가능한 저차원 표현 목표 아님 | arXiv:2511.08544 [39] |
| **I-JEPA (Assran et al.)** | 이미지 기반 JEPA | 물리적 해석 가능성 미강조 | CVPR 2023 [69] |
| **CPC (van den Oord et al.)** | 대조적 예측 코딩 | 인코더 비공유, δ-predictor 없음 | arXiv:1807.03748 [43] |
| **DVSIB (Abdelaleem et al.)** | 심층 변분 SIB 프레임워크 | DySIB의 기반 프레임워크 | JMLR 2025 [47] |
| **Martini & Nemenman (2024)** | 일반화된 SIB와 데이터 효율성 | 이론적 기반 제공 | Neural Computation 2024 [41] |
| **Meng & Bouchard (2024)** | 직교 확률적 선형 혼합 모델 | 신경 집단 활동 특화 | PLOS Computational Biology 2024 [46] |

### 핵심 차별점

DySIB가 기존 방법들과 구별되는 가장 중요한 특징:

1. **재구성 없는 순수 잠재 공간 예측** + **대칭 압축**의 결합
2. **물리적 귀납적 편향** ($\delta$-predictor)의 아키텍처 내재화
3. **하이퍼파라미터의 완전한 자기일관적 결정** ($k_z$, $n_F$ 모두 데이터로 선택)
4. **실제 실험 데이터**에서의 검증 (합성 데이터 아님)

---

## 참고자료

**본 논문:**
- Martini, K.M., Abdelaleem, E., Gulati, P., & Nemenman, I. (2026). "Information bottleneck for learning the phase space of dynamics from high-dimensional experimental data." arXiv:2604.24662v2 [physics.data-an]

**논문 내 주요 인용 문헌:**
- [47] Abdelaleem, E., Nemenman, I., & Martini, K.M. (2025). "Deep variational multivariate information bottleneck—a framework for variational losses." *Journal of Machine Learning Research*, 26, 1.
- [48] Tishby, N., Pereira, F.C., & Bialek, W. (1999). "The information bottleneck method." *37th Annual Allerton Conference*.
- [43] van den Oord, A., Li, Y., & Vinyals, O. (2018). "Representation learning with contrastive predictive coding." arXiv:1807.03748.
- [29] Chen, B., et al. (2022). "Automated discovery of fundamental variables hidden in experimental data." *Nature Computational Science*, 2, 433.
- [44] Schmitt, M.S., et al. (2023). "Information theory for dimensionality reduction in dynamical systems." arXiv:2312.06608.
- [39] Balestriero, R. & LeCun, Y. (2025). "LeJEPA: Provable and scalable self-supervised learning without the heuristics." arXiv:2511.08544.
- [41] Martini, K.M. & Nemenman, I. (2024). "Data efficiency, dimensionality reduction, and the generalized symmetric information bottleneck." *Neural Computation*, 36, 1353.
- [20] Brunton, S.L., Proctor, J.L., & Kutz, J.N. (2016). "Discovering governing equations from data by sparse identification of nonlinear dynamical systems." *PNAS*, 113, 3932.
- [69] Assran, M., et al. (2023). "Self-supervised learning from images with a joint-embedding predictive architecture." *CVPR 2023*.
- [54] Kingma, D.P. & Welling, M. (2014). "Auto-encoding variational bayes." *ICLR 2014*. arXiv:1312.6114.
- [59] Gulati, P., et al. (2026). "Mutual information and task-relevant latent dimensionality." *ICLR 2026 Workshop*. arXiv:2602.08105.
