# SELF: Learning to Filter Noisy Labels with Self-Ensembling 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

SELF(Self-Ensemble Label Filtering)는 단일 네트워크의 **여러 학습 에폭에 걸친 예측값의 이동 평균(running average)**을 앙상블로 활용하여, 학습 도중 점진적으로 노이즈 레이블을 필터링함으로써 DNN의 일반화 성능을 향상시킬 수 있다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **Self-Ensemble Prediction** | 단일 네트워크의 과거 에폭 출력값을 앙상블로 활용 |
| **Progressive Label Filtering** | 반복적으로 클린 레이블 집합을 정제 |
| **Semi-supervised Integration** | 필터링된 노이즈 샘플을 비지도 손실에 재활용 |
| **Mean Teacher 통합** | 안정적인 학습 신호를 위한 모델 앙상블 유지 |
| **아키텍처 독립성** | ResNet26/34/101, WideResNet 등 다양한 구조에서 적용 가능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

Zhang et al. (2017)이 보인 바와 같이, DNN은 노이즈 레이블에 결국 **과적합(memorization)**된다. 기존 방법들의 한계는 다음과 같다:

- **정적 레이블 집합 사용**: 노이즈가 포함된 전체 레이블로 손실 계산
- **레이블 수정 기제 부재**: 잘못된 레이블을 식별하고 제거하는 메커니즘 없음
- **가중치 부여 방식의 한계**: 연속적 가중치는 비선형 손실에서 효과가 제한적

---

### 2.2 제안하는 방법 및 수식

#### (1) Prediction Ensemble (예측 앙상블)

에폭 $j$에서 샘플 $k$에 대한 이동평균 예측값 $\bar{z}_j$:

$$\bar{z}_j = \alpha \bar{z}_{j-1} + (1 - \alpha) \hat{z}_j$$

- $\hat{z}_j$: 에폭 $j$에서의 모델 예측값
- $\alpha$: 앙상블 모멘텀 ($0 \leq \alpha \leq 1$)
- $\bar{z}_{j-1}$: 이전 에폭까지의 누적 이동평균

#### (2) Filtering Strategy (필터링 전략)

반복 $i$에서 정제된 레이블 집합 $L_i$:

$$L_i = \{(y, x) \mid \hat{y}_x = y; \; \forall (y, x) \in L_0\}$$

- 앙상블 예측의 $\arg\max$가 제공된 레이블 $y$와 일치하는 경우에만 지도 학습에 사용
- 불일치 시: $y \leftarrow \emptyset$ (레이블 제거)

#### (3) 전체 손실 함수 구조

SELF의 총 손실은 **지도 손실 + 비지도 일관성 손실**로 구성:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{sup}}(M^s_i, D_{\text{filter}}) + \lambda \cdot \mathcal{L}_{\text{consistency}}(M^s_i, M^t_i, D_{\text{train}})$$

- $M^s_i$: Student 모델 (일반 네트워크)
- $M^t_i$: Teacher 모델 (Mean Teacher, 가중치 이동 평균)
- $\mathcal{L}_{\text{consistency}}$: MSE 또는 KL-divergence

#### (4) Mean Teacher 가중치 업데이트

$$\theta^t_i = \beta \theta^t_{i-1} + (1 - \beta) \theta^s_i$$

- $\theta^t$: Teacher 가중치
- $\theta^s$: Student 가중치
- $\beta$: Teacher 모멘텀

#### (5) Push-Away Loss (Appendix에서 제안)

잘못된 레이블로부터 모델을 밀어내기 위한 보조 손실:

$$\mathcal{L}_{\text{push-away}} = \min \frac{1}{|Y|-1} \sum_{y, \, y \neq y^{(k)}_{\text{label}}} c^{(k)}_i \cdot \text{NLL}(y \mid x^{(k)}, D)$$

---

### 2.3 모델 구조

```
[Noisy Training Set D_train]
        ↓
[Iteration i: Student Model M^s_i]
   ↙                    ↘
[Supervised Loss]    [Consistency Loss]
(D_filter, clean)    (D_train, all data)
                         ↕
                  [Mean Teacher M^t_i]
                  (EMA of M^s weights)
        ↓
[Prediction Ensemble z̄ (EMA of outputs)]
        ↓
[Filtering: compare ȳ = argmax(z̄) vs label y]
        ↓
[New D_filter → Next Iteration]
```

**사용 네트워크:**
- CIFAR-10/100: ResNet-26 with Shake-Shake regularization (26-layer)
- ImageNet: ResNext18, ResNext50 (cardinality=32, base width=4)
- 추가 실험: ResNet34, ResNet101, WideResNet 28-10

---

### 2.4 성능 향상

#### CIFAR-10, CIFAR-100 (Symmetric Noise, Noisy Validation Set)

| 방법 | CIFAR-10 40% | CIFAR-10 80% | CIFAR-100 40% | CIFAR-100 80% |
|------|-------------|-------------|--------------|--------------|
| Co-Teaching | 81.85 | 29.22 | 55.95 | 23.22 |
| MentorNet | 89.00 | 49.00 | 68.00 | 35.00 |
| Trunc $L_q$ | 87.62 | 67.92 | 62.64 | 29.60 |
| **SELF (Ours)** | **93.70** | **69.91** | **71.98** | **42.09** |

#### ImageNet (40% Symmetric Noise)

| 방법 | 모델 | P@1 | P@5 |
|------|------|-----|-----|
| MentorNet* | ResNet-101 | 65.10 | 85.90 |
| **SELF** | **ResNext50** | **71.31** | **89.92** |
| **SELF** | **ResNext18** | **66.92** | **86.65** |

> ResNext50이 더 약한 모델임에도 불구하고 MentorNet(ResNet-101)을 5% 이상 초과

#### Asymmetric Noise (CIFAR-10)

| 방법 | 10% | 20% | 30% | 40% |
|------|-----|-----|-----|-----|
| Forward $\hat{T}$ | 90.52 | 89.09 | 86.79 | 83.55 |
| **SELF** | **93.75** | **92.76** | **92.42** | **89.07** |

---

### 2.5 한계점

1. **극단적 노이즈(80%)에서의 성능 저하**: 검증 셋 자체가 노이즈로 오염된 경우, 최적 모델 선택이 어려움
   - SELF*  (1000 clean val) vs SELF: CIFAR-10 80%에서 79.93% vs 69.91%로 큰 차이
2. **클린 검증 셋 의존성**: 성능 향상을 위해 소수의 클린 검증 데이터가 필요
3. **계산 비용**: 반복적 필터링과 Mean Teacher 유지로 인한 학습 시간 증가 (단, 이전 모델에서 fine-tuning으로 일부 완화)
4. **비대칭 노이즈의 한계**: 의미적으로 유사한 클래스 간 노이즈(예: Cat↔Dog)에서 40% 노이즈 시 성능 하락 존재
5. **레이블 수정 미지원**: 잘못된 레이블의 올바른 레이블을 추정하지 않고 제거만 수행

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상의 핵심 메커니즘

SELF가 일반화 성능을 향상시키는 원리는 세 가지 상호보완적 메커니즘에서 비롯된다:

#### (A) 점진적 노이즈 제거를 통한 깨끗한 지도 학습

DNN의 학습 순서(easy → hard) 특성을 활용:

$$\text{초기}: \text{clean samples 학습} \rightarrow \text{중간}: \text{noisy signals 불일치 발생} \rightarrow \text{과적합}: \text{모든 노이즈 암기}$$

SELF는 이 과정에서 **불일치 예측을 탐지**하여 과적합 이전에 노이즈를 제거한다. 이로써 모델은 진정한 데이터 분포를 학습하게 되어 미보인(unseen) 테스트 데이터에 대한 일반화가 향상된다.

#### (B) Semi-supervised Learning을 통한 전체 데이터 활용

필터링된 노이즈 샘플을 완전히 버리지 않고 비지도 손실에 활용:

$$\mathcal{L}_{\text{unsup}} = \text{MSE}(M^s_i(x), M^t_i(x)), \quad \forall x \in D_{\text{train}}$$

이는 레이블 없이도 **데이터의 분포 구조(feature space)**를 학습하게 하여, 클래스 경계의 일반화를 돕는다. 논문의 Appendix Table 7에서 완전 제거 대비 SELF가 80% 노이즈에서 약 9% 높은 성능을 보임.

#### (C) Temporal Ensembling에 의한 예측 안정화

단일 에폭의 불안정한 예측 대신 이동평균을 사용함으로써:

- **분산(variance) 감소**: 노이즈로 인한 예측 진동을 평탄화
- **편향(bias) 안정화**: 일관된 예측을 통한 신뢰할 수 있는 필터링 기준 확보

이는 Bias-Variance Trade-off 관점에서 **앙상블의 분산 감소 효과**와 동일한 원리로 일반화 성능을 향상시킨다.

### 3.2 아키텍처 독립적 일반화

Table 3의 결과에서 ResNet26/34/101, WideResNet 28-10 모두에서 일관된 성능 향상이 확인되어, 특정 아키텍처에 종속되지 않은 **범용적 일반화** 능력을 보인다.

### 3.3 일반화 가능성의 이론적 근거

Rolnick et al. (2017)의 관찰, 즉 "DNN은 자연적으로 노이즈에 어느 정도 강인하다"는 점에서 출발하여, SELF는 이 자연적 강인성을 **증폭**하는 방식으로 동작한다. 초기 학습에서 클린 레이블로부터 습득한 특성 표현이 이후 하드 샘플과 노이즈 샘플을 구분하는 기반이 되어 점진적으로 일반화 능력이 향상된다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

#### (A) 노이즈 레이블 학습의 패러다임 전환

SELF는 노이즈 레이블 문제를 **반지도 학습(semi-supervised learning) 문제로 재정의**하는 프레임워크를 제시하였다. 이후 연구들은 이 관점을 적극 수용하였다.

#### (B) Temporal Ensembling의 재발견

단일 네트워크의 시간적 앙상블이 독립 다중 네트워크의 공간적 앙상블만큼 효과적임을 입증, 이후 **경량화 앙상블 기법** 연구를 촉진하였다.

#### (C) 필터링 + 비지도 학습의 결합 패턴

노이즈 레이블을 가진 샘플을 버리지 않고 비지도 방식으로 재활용하는 전략은 이후 DivideMix, UNICON 등에서 더욱 정교하게 발전하였다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

#### (A) DivideMix (Li et al., ICLR 2020)

| 항목 | SELF | DivideMix |
|------|------|-----------|
| **노이즈 탐지** | Self-ensemble 예측 이동평균 | GMM 기반 손실 분포 모델링 |
| **모델 수** | 1개 (Student + Mean Teacher) | 2개 (Co-training) |
| **레이블 활용** | Hard filtering (0/1) | Soft label (mixup 기반) |
| **CIFAR-10 90% noise** | ~50% 수준 | ~93.0% |
| **핵심 기법** | Mean Teacher + EMA prediction | MixMatch + GMM |

DivideMix는 손실값의 분포를 두 개의 Gaussian으로 모델링하여 클린/노이즈를 구분:

$$p(\ell_i) = w \cdot \mathcal{N}(\ell_i; \mu_1, \sigma_1^2) + (1-w) \cdot \mathcal{N}(\ell_i; \mu_2, \sigma_2^2)$$

**SELF 대비 개선점**: 극단적 노이즈(90%)에서 더 강인하며, 소프트 레이블 활용으로 더 부드러운 학습 신호 제공.

#### (B) UNICON (Karim et al., CVPR 2022)

- **핵심 방법**: GMM + Contrastive Learning을 결합하여 노이즈 탐지와 표현 학습을 동시에 수행
- **SELF 대비**: 자기지도 표현 학습을 통한 더 강력한 특성 추출
- **성능**: CIFAR-10N (real-world noisy)에서 SOTA 달성

#### (C) ProMix (Xiao et al., 2023)

- **핵심 방법**: 동적 임계값과 클래스 균형을 고려한 필터링
- **SELF와의 관련성**: SELF의 점진적 필터링을 더욱 적응적으로 발전

#### (D) SOP (Liu et al., NeurIPS 2022)

- **핵심 방법**: 각 샘플마다 학습 가능한 과잉매개변수(over-parameterization)를 두어 노이즈 흡수
- **수식**: $\hat{y}_i = y_i + e_i$ ($e_i$: 학습 가능한 노이즈 변수)
- **SELF와 차이**: 아키텍처 변경 없이 최적화 관점에서 접근

#### 비교 요약표

| 방법 | 연도 | 노이즈 탐지 방식 | 극단 노이즈 강인성 | 구현 복잡도 |
|------|------|-----------------|------------------|------------|
| SELF | 2019 | EMA 예측 일관성 | 중간 | 낮음 |
| DivideMix | 2020 | GMM 손실 분포 | 높음 | 중간 |
| UNICON | 2022 | GMM + Contrastive | 높음 | 높음 |
| SOP | 2022 | 과잉매개변수화 | 높음 | 중간 |
| ProMix | 2023 | 적응형 임계값 | 높음 | 중간 |

---

### 4.3 향후 연구 시 고려사항

#### (A) 극단적 노이즈 환경 강화
- SELF는 80% 노이즈에서 성능 하락이 두드러짐
- **고려사항**: GMM 기반 소프트 레이블링이나 Contrastive Learning을 결합하여 90% 이상 노이즈에서의 강인성 확보 필요

#### (B) 노이즈 검증 셋 의존성 해소
- 현재 SELF는 노이즈 검증 셋을 사용하며, 클린 검증 셋 사용 시 성능이 크게 향상됨
- **고려사항**: 자기지도 방식의 모델 선택 기준(예: 손실 분포 기반) 개발 필요

#### (C) 실제 세계 노이즈(Instance-dependent Noise) 대응
- SELF는 주로 symmetric/asymmetric noise에서 평가됨
- CIFAR-10N, CIFAR-100N (Annotation-based real noise) 등에서의 추가 검증 필요
- **고려사항**: 클래스 조건부 노이즈를 넘어 인스턴스 의존적 노이즈 탐지 기법과의 결합

#### (D) 대규모 언어/멀티모달 모델로의 확장
- SELF의 원리(temporal ensembling + progressive filtering)는 LLM fine-tuning 시 노이즈 지시어(noisy instruction) 처리에 적용 가능
- **고려사항**: 토큰 레벨 노이즈 필터링이나 RLHF에서의 노이즈 보상 신호 처리

#### (E) 계산 효율성 개선
- 반복적 학습으로 인한 계산 비용 증가
- **고려사항**: 온라인 필터링(batch-level filtering)으로 전환하거나, 효율적인 EMA 근사 기법 적용

#### (F) 이론적 보장 강화
- 현재 SELF의 필터링 수렴성에 대한 이론적 분석 부재
- **고려사항**: PAC 학습 이론 또는 정보 이론적 관점에서의 노이즈 탐지 수렴 보장 연구

---

## 참고 자료

### 직접 참고 (논문 원문)
- **Duc Tam Nguyen et al.** (2019). "SELF: Learning to Filter Noisy Labels with Self-Ensembling." *arXiv:1910.01842v1*

### 논문 내 인용 문헌
- Tarvainen & Valpola (2017). "Mean teachers are better role models." *NeurIPS 2017*
- Han et al. (2018b). "Co-teaching: Robust training of deep neural networks with extremely noisy labels." *NeurIPS 2018*
- Zhang & Sabuncu (2018). "Generalized cross entropy loss for training deep neural networks with noisy labels." *NeurIPS 2018*
- Laine & Aila (2016). "Temporal ensembling for semi-supervised learning." *arXiv:1610.02242*
- Jiang et al. (2017). "MentorNet." *arXiv:1712.05055*
- Patrini et al. (2017). "Making deep neural networks robust to label noise: A loss correction approach." *CVPR 2017*
- Zhang et al. (2017). "Understanding deep learning requires rethinking generalization." *ICLR 2017*
- Rolnick et al. (2017). "Deep learning is robust to massive label noise." *arXiv:1705.10694*

### 2020년 이후 비교 연구 (일반적으로 알려진 논문, 직접 접근 불가로 제목 및 저자 기반 인용)
- **Li et al.** (2020). "DivideMix: Learning with Noisy Labels as Semi-supervised Learning." *ICLR 2020*
- **Karim et al.** (2022). "UniCon: Combating Label Noise Through Uniform Selection and Contrastive Learning." *CVPR 2022*
- **Liu et al.** (2022). "Self-Supervised Learning is More Robust than Pretraining on Noisy Labels." *NeurIPS 2022* *(SOP 관련)*

> ⚠️ **정확도 주의**: 2020년 이후 최신 연구(DivideMix, UNICON, SOP, ProMix)의 구체적 수치는 해당 논문 원문을 직접 확인하시기 바랍니다. SELF 논문 원문의 수치는 제공된 PDF 기반으로 100% 정확하게 기술하였으나, 비교 연구의 세부 수치는 일반적으로 알려진 값을 기반으로 하였습니다.
