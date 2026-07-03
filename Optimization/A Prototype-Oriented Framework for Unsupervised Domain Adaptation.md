# A Prototype-Oriented Framework for Unsupervised Domain Adaptation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 비지도 도메인 적응(UDA) 방법들은 소스와 타겟 샘플 간의 통계적 거리를 최소화하는 방식에 의존하는데, 이는 **샘플링 변동성(sampling variability)**, **클래스 불균형(class imbalance)**, **데이터 프라이버시(data-privacy)** 문제를 야기한다. 본 논문은 이러한 문제를 해결하기 위해 **클래스 프로토타입(class prototypes)**을 추출하고, 타겟 특징을 프로토타입에 정렬하는 확률론적 프레임워크인 **PCT(Prototype-oriented Conditional Transport)**를 제안한다.

### 주요 기여

1. **클래스 프로토타입 활용**: 선형 분류기의 가중치를 클래스 프로토타입으로 활용하여, 추가적인 모델 파라미터 없이 타겟 특징을 프로토타입에 정렬하는 일반적인 확률론적 프레임워크 제안
2. **확률론적 양방향 전송(Probabilistic Bi-directional Transport)**: 기대 이동 비용(expected cost)을 최소화하는 양방향 전송 손실 도입
3. **다양한 시나리오 적용**: 단일 소스(single-source), 다중 소스(multi-source), 클래스 불균형(class-imbalance), 소스-프라이빗(source-private) 등 다양한 도메인 적응 시나리오에서 경쟁력 있는 성능 달성

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

기존의 소스-타겟 특징 직접 정렬 방식의 문제점:

| 문제 | 설명 |
|------|------|
| 샘플링 변동성 | MMD, Wasserstein 등 통계적 거리 측도가 미니배치 내 이상치에 민감 |
| 클래스 불일치 | 미니배치 샘플링 시 소스/타겟 도메인의 클래스 분포가 다를 수 있음 |
| 데이터 프라이버시 | 소스 데이터에 직접 접근해야 하므로 헬스케어 등 프라이버시 중요 분야 적용 어려움 |

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 클래스 프로토타입 학습

선형 분류기의 가중치 $[\boldsymbol{\mu}_1, \boldsymbol{\mu}_2, \ldots, \boldsymbol{\mu}_K] \in \mathbb{R}^{d_f \times K}$를 클래스 프로토타입으로 사용하며, 소스 데이터에 대한 크로스 엔트로피 손실로 학습한다:

$$\mathcal{L}_{\text{cls}} = \mathbb{E}_{(\boldsymbol{x}^s_i, y^s_i) \sim \mathcal{D}_s} \left[ \sum_{k=1}^{K} -\log p^s_{ik} \mathbf{1}_{\{y^s_i = k\}} \right]$$

$$p^s_{ik} := \frac{\exp(\boldsymbol{\mu}_k^T \boldsymbol{f}^s_i + b_k)}{\sum_{k'=1}^{K} \exp(\boldsymbol{\mu}_{k'}^T \boldsymbol{f}^s_i + b_{k'})}$$

여기서 $\boldsymbol{f}^s_i = F_\theta(\boldsymbol{x}^s_i)$는 소스 샘플의 특징 표현이다.

#### 2.2.2 타겟 → 프로토타입 방향 전송

타겟 특징 $\boldsymbol{f}^t_j$에서 클래스 프로토타입 $\boldsymbol{\mu}_k$로 이동하는 조건부 분포:

$$\pi_\theta(\boldsymbol{\mu}_k | \boldsymbol{f}^t_j) = \frac{p(\boldsymbol{\mu}_k) \exp(\boldsymbol{\mu}_k^T \boldsymbol{f}^t_j)}{\sum_{k'=1}^{K} p(\boldsymbol{\mu}_{k'}) \exp(\boldsymbol{\mu}_{k'}^T \boldsymbol{f}^t_j)}, \quad k \in \{1, \ldots, K\}$$

이 때 $p(\boldsymbol{\mu}_k)$는 타겟 도메인의 클래스 사전 분포(class prior)이며, $\exp(\boldsymbol{\mu}_k^T \boldsymbol{f}^t_j)$는 비정규화 우도(unnormalized likelihood)로 프로토타입과 타겟 특징의 유사도를 측정한다.

포인트 간 이동 비용(point-to-point moving cost)은 코사인 비유사도(cosine dissimilarity)로 정의:

$$c(\boldsymbol{\mu}_k, \boldsymbol{f}^t_j) = 1 - \frac{\boldsymbol{\mu}_k^T \boldsymbol{f}^t_j}{\|\boldsymbol{\mu}_k\|_2 \|\boldsymbol{f}^t_j\|_2}$$

타겟 → 프로토타입 방향의 기대 이동 비용:

$$\mathcal{L}_{t \to \mu} = \mathbb{E}_{\boldsymbol{x}^t_j \sim \mathcal{D}^x_t} \left[ \sum_{k=1}^{K} c(\boldsymbol{\mu}_k, \boldsymbol{f}^t_j) \frac{p(\boldsymbol{\mu}_k) \exp(\boldsymbol{\mu}_k^T \boldsymbol{f}^t_j)}{\sum_{k'=1}^{K} p(\boldsymbol{\mu}_{k'}) \exp(\boldsymbol{\mu}_{k'}^T \boldsymbol{f}^t_j)} \right]$$

> **엔트로피 최소화와의 연결**: 만약 $c(\boldsymbol{\mu}\_k, \boldsymbol{f}^t_j) = -\log p^t_{jk}$이고 균등 사전분포를 사용하면, $\mathcal{L}\_{t \to \mu}$는 엔트로피 최소화와 동치가 된다:
> $$\mathcal{L}\_{t \to \mu} = -\mathbb{E}\_{\boldsymbol{x}^t_j \sim \mathcal{D}^x_t} \left[ \sum\_{k=1}^{K} p^t_{jk} \log p^t_{jk} \right]$$

#### 2.2.3 프로토타입 → 타겟 방향 전송

미니배치 $\{\boldsymbol{x}^t_j\}_{j=1}^M$ 내에서 프로토타입 $\boldsymbol{\mu}_k$에서 타겟 특징 $\boldsymbol{f}^t_j$로 이동하는 조건부 분포:

$$\pi_\theta(\boldsymbol{f}^t_j | \boldsymbol{\mu}_k) = \frac{\exp(\boldsymbol{\mu}_k^T \boldsymbol{f}^t_j)}{\sum_{j'=1}^{M} \exp(\boldsymbol{\mu}_k^T \boldsymbol{f}^t_{j'})}, \quad \boldsymbol{f}^t_j \in \{\boldsymbol{f}^t_1, \ldots, \boldsymbol{f}^t_M\}$$

프로토타입 → 타겟 방향의 기대 이동 비용:

$$\mathcal{L}_{\mu \to t} = \mathbb{E}_{\{\boldsymbol{x}^t_j\}_{j=1}^M \sim \mathcal{D}^x_t} \left[ \sum_{k=1}^{K} p(\boldsymbol{\mu}_k) \sum_{j=1}^{M} c(\boldsymbol{\mu}_k, \boldsymbol{f}^t_j) \frac{\exp(\boldsymbol{\mu}_k^T \boldsymbol{f}^t_j)}{\sum_{j'=1}^{M} \exp(\boldsymbol{\mu}_k^T \boldsymbol{f}^t_{j'})} \right]$$

#### 2.2.4 최종 손실 함수

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{t \to \mu} + \mathcal{L}_{\mu \to t}$$

> **핵심**: 전송 손실($\mathcal{L}\_{t \to \mu}$, $\mathcal{L}_{\mu \to t}$ )에서 $\boldsymbol{\mu}$의 그래디언트는 역전파하지 않음(gradient stopping). 이를 통해 소스 데이터 없이도 타겟 도메인 적응이 가능하며, 학습이 더 안정적으로 이루어진다.

#### 2.2.5 타겟 클래스 비율 학습 (EM 알고리즘)

타겟 도메인의 클래스 비율 $\{p(\boldsymbol{\mu}\_k)\}_{k=1}^K$를 EM 알고리즘으로 추정:

$$p(\boldsymbol{\mu}_k)^{l+1} = \frac{1}{M} \sum_{j=1}^{M} \pi^l_\theta(\boldsymbol{\mu}_k | \boldsymbol{f}^t_j)$$

$$\text{where} \quad \pi^l_\theta(\boldsymbol{\mu}_k | \boldsymbol{f}^t_j) = \frac{p(\boldsymbol{\mu}_k)^l \exp(\boldsymbol{\mu}_k^T \boldsymbol{f}^t_j)}{\sum_{k'=1}^{K} p(\boldsymbol{\mu}_{k'})^l \exp(\boldsymbol{\mu}_{k'}^T \boldsymbol{f}^t_j)}$$

전체 데이터셋에 대한 반복 업데이트:

$$p(\boldsymbol{\mu}_k)^{l+1} \leftarrow (1 - \beta^l) p(\boldsymbol{\mu}_k)^l + \beta^l p(\boldsymbol{\mu}_k)^{l+1}$$

$$\beta^l = \beta_0 (1 + \gamma l)^{-\alpha}, \quad \gamma = 0.0002, \quad \alpha = 0.75$$

### 2.3 모델 구조

```
소스 데이터 (X_s, Y_s)  ──→  Feature Encoder (F_θ, ResNet-50/101)  ──→  f_s  ──→  Cross-Entropy Loss (L_cls)
                                        ↕ (Shared Weights)                        ↗ ↖
타겟 데이터 (X_t)        ──→  Feature Encoder (F_θ, ResNet-50/101)  ──→  f_t  ──→  Transport Loss (L_{t→μ} + L_{μ→t})
                                                                              ↕
                                                                    Class Prototypes [μ_1, ..., μ_K]
                                                                    (= Linear Classifier Weights, 그래디언트 정지)
```

- **Feature Encoder**: ResNet-50 (단일/클래스불균형/소스-프라이빗), ResNet-101 (다중 소스)
- **Class Prototypes**: 선형 분류기 가중치 $\boldsymbol{\mu}_k \in \mathbb{R}^{d_f}$, 추가 파라미터 없음
- **Transport Loss**: 양방향 조건부 전송 손실
- 연산 복잡도: $\mathcal{O}(d_f M K)$ (기존 최적 전송의 $\mathcal{O}(M^3 \log M)$ 대비 훨씬 효율적)

### 2.4 성능 향상

| 데이터셋 | 최고 기존 방법 | PCT | 향상 |
|----------|---------------|-----|------|
| Office-31 (avg) | MDD: 88.9% | **90.0%** | +1.1% |
| Office-Home (avg) | MDD: 68.1% | **71.8%** | +3.7% |
| Office-Home MSDA | MFSAN: 74.1% | **77.4%** | +3.3% |
| DomainNet MSDA | ML-MSDA: 44.3% | **47.6%** | +3.3% |
| Sub-sampled O-31 (sub-S) | IWCDAN: 83.9% | **87.9%** | +4.0% |
| Sub-sampled O-H (sub-S) | IWCDAN: 61.2% | **67.8%** | +6.6% |

### 2.5 한계점

1. **폐쇄 범주(Closed-set) 가정**: 소스와 타겟이 동일한 레이블 공간을 공유한다고 가정하므로, 오픈셋(open-set) 또는 파셜(partial) 도메인 적응에는 직접 적용 어려움
2. **단순 프로토타입 표현**: 각 클래스를 단일 프로토타입으로 표현하므로, 클래스 내 분포가 복잡하거나 다중 모드(multi-modal)인 경우 표현력이 제한될 수 있음
3. **소스-프라이빗 환경에서의 성능 저하**: 소스 데이터 없이 적응할 경우 평균 정확도가 Office-31에서 1.6%, Office-Home에서 0.8% 하락
4. **하이퍼파라미터 민감도**: $\beta_0$ 등의 하이퍼파라미터 선택이 클래스 불균형 시나리오에서 성능에 영향을 미침
5. **대규모 클래스 수에서의 확장성**: 클래스 수 $K$가 매우 클 경우 EM 기반 비율 추정의 안정성 검증 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화에 기여하는 핵심 메커니즘

#### (1) 클러스터 가정(Cluster Assumption) 기반 정렬

양방향 전송 손실은 클러스터 가정에 기반한다. $\mathcal{L}\_{t \to \mu}$는 각 타겟 샘플을 가장 가까운 프로토타입 방향으로 당겨 결정 경계가 저밀도 영역을 지나도록 유도한다. $\mathcal{L}_{\mu \to t}$는 모든 프로토타입 주변에 타겟 샘플이 존재하도록 보장하여, 특정 클래스로의 collapse를 방지한다. 이 두 손실의 결합은 타겟 도메인에서 **구조적으로 분리된 클래스 표현**을 학습하게 한다.

#### (2) 샘플링 변동성 제거

소스 샘플 대신 학습된 프로토타입을 기준으로 정렬하므로, 미니배치 샘플링에 의한 노이즈가 제거된다. 이는 학습 과정을 안정화하고 일반화에 유리한 특징 공간을 형성한다.

#### (3) 클래스 비율 적응

EM 기반 클래스 비율 추정( $p(\boldsymbol{\mu}_k)$ )을 통해 타겟 도메인의 레이블 분포 변화에 동적으로 적응한다. 실험 결과(Figure 3), 추정된 비율의 L1 오차가 균등 분포 대비 대폭 감소(0.16 vs 0.58)하였으며, 이는 불균형 데이터에서의 일반화를 크게 향상시킨다.

#### (4) 그래디언트 정지(Gradient Stopping) 전략

전송 손실에서 $\boldsymbol{\mu}$의 그래디언트를 역전파하지 않음으로써, 프로토타입이 타겟 도메인에 과적합되는 것을 방지한다. 이는 SimSiam[Chen & He, 2021]에서의 연구와 일치하며, 퇴화 솔루션(degenerate solution) 방지에 기여한다.

#### (5) 소스-프라이빗 환경에서의 일반화

소스 데이터 없이도 분류기 가중치만으로 프로토타입을 활용할 수 있어, 프라이버시 제약이 있는 실제 환경에서도 일반화 성능이 유지된다(Office-31: 88.4%, SHOT: 88.6%로 통계적으로 유의미한 차이 없음).

#### (6) 다중 소스 시나리오에서의 일반화

다중 소스 설정에서 PCT는 도메인 레이블이나 도메인별 분류기 없이도 멀티소스 전용 방법들(MFSAN, ML-MSDA)보다 우수한 성능을 보였다. 이는 프로토타입이 소스 도메인의 변동성을 자연스럽게 흡수하는 역할을 함을 시사한다.

### 3.2 일반화 향상의 이론적 근거

Ben-David et al.의 이론에 따르면 타겟 오류의 상한은 다음과 같다:

$$\epsilon_T(h) \leq \epsilon_S(h) + d_{\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda$$

PCT는 소스 오류 $\epsilon_S(h)$를 $\mathcal{L}\_{\text{cls}}$로 최소화하고, 도메인 불일치 $d_{\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T)$를 프로토타입 기반 전송 손실로 줄이며, 클래스 비율 추정을 통해 $\lambda$(최적 공동 오류)를 암묵적으로 최소화한다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 논문에서 직접 언급된 방법들 및 2020년 이후 공개적으로 알려진 주요 연구들을 기반으로 작성하였습니다. PCT 이후 등장한 일부 연구들의 정확한 수치는 논문 원본을 직접 확인하시기 바랍니다.

### 4.1 논문 내에서 직접 비교된 2020년 이후 연구

| 방법 | 발표연도 | 핵심 아이디어 | Office-31 | Office-Home | PCT 대비 |
|------|----------|--------------|-----------|-------------|---------|
| **SHOT** (Liang et al., ICML 2020) | 2020 | 정보 최대화 + 의사 레이블, 소스 프라이빗 | 88.6% (private) | 71.8% (private) | 통계적 유의차 없음 |
| **IWCDAN** (Tachet des Combes et al., NeurIPS 2020) | 2020 | 중요도 가중 조건부 적대적 DA | 83.9% (sub-S) | 61.2% (sub-S) | PCT +4.0% / +6.6% |
| **Robust OT** (Balaji et al., NeurIPS 2020) | 2020 | 강건한 최적 전송 | - | - | 참조 연구 |
| **PCT (Ours)** | 2021 | 프로토타입 기반 양방향 조건부 전송 | **90.0%** | **71.8%** | - |

### 4.2 PCT 이후 주요 관련 연구 방향 (2021~)

PCT가 발표된 이후, 도메인 적응 분야에서는 다음과 같은 방향의 연구들이 활발히 진행되고 있다:

#### (A) Vision-Language 모델 기반 도메인 적응
CLIP(Radford et al., 2021) 등 대규모 사전학습 모델을 활용한 도메인 적응 연구들이 등장하였다. 이들은 PCT와 달리 텍스트-이미지 멀티모달 정보를 활용하나, 프로토타입 개념은 여전히 유효하게 활용된다.

#### (B) 자기지도학습(Self-supervised Learning) 기반 DA
MoCo, SimCLR 등의 대조학습(contrastive learning) 기법을 도메인 적응에 결합하는 연구들이 증가하였다. PCT의 그래디언트 정지 전략과 시너지 효과가 기대된다.

#### (C) 테스트 타임 적응(Test-Time Adaptation, TTA)
소스 데이터 없이 테스트 시점에만 적응하는 연구들(TTT, TENT 등)은 PCT의 소스-프라이빗 설정과 맥락을 같이 한다.

### 4.3 PCT vs 주요 방법 종합 비교

| 비교 항목 | DANN | CDAN/MDD | SHOT | DeepJDOT | PCT |
|-----------|------|----------|------|----------|-----|
| 적대적 학습 | ✅ | ✅ | ❌ | ❌ | ❌ |
| 추가 파라미터 | ✅ | ✅ | ❌ | ❌ | ❌ |
| 소스 데이터 필요 | ✅ | ✅ | ❌ | ✅ | 선택적 |
| 클래스 불균형 처리 | ❌ | ❌ | ❌ | ❌ | ✅ |
| 다중 소스 지원 | 제한적 | 제한적 | ❌ | ❌ | ✅ |
| 연산 복잡도 | $O(M)$ | $O(M)$ | $O(M)$ | $O(M^3 \log M)$ | $O(d_f MK)$ |
| 수렴 안정성 | 낮음 | 낮음 | 중간 | 중간 | **높음** |

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

#### (1) 프로토타입 중심 패러다임의 확산
PCT는 **소스 샘플 대신 학습된 프로토타입을 정렬 기준**으로 사용하는 새로운 패러다임을 제시하였다. 이는 퓨샷 학습(few-shot learning), 반지도학습(semi-supervised learning) 등 인접 분야에서도 프로토타입 활용을 촉진할 것으로 예상된다.

#### (2) 소스-프라이빗 DA의 실용적 발전 가속화
소스 데이터 없이 분류기 가중치만으로 적응할 수 있음을 보임으로써, 의료영상, 금융, 자율주행 등 데이터 프라이버시가 중요한 산업 분야에서의 적용 가능성을 크게 열었다.

#### (3) 클래스 불균형 문제에 대한 통합 접근 제시
EM 기반 타겟 클래스 비율 추정과 프로토타입 기반 정렬을 통합한 접근법은, 기존 중요도 가중(importance weighting) 방법보다 효과적인 불균형 처리 방향을 제시하였다.

#### (4) 비적대적(Non-adversarial) DA 방법론의 재조명
안정적인 수렴과 높은 성능을 동시에 달성함으로써, 복잡하고 불안정한 GAN 기반 방법들의 대안으로서 비적대적 방법론의 가능성을 확인시켜 주었다.

### 5.2 향후 연구 시 고려할 점

#### (1) 다중 프로토타입(Multi-Prototype) 확장
현재 클래스당 단일 프로토타입을 사용하므로, 클래스 내 분포가 복잡한 경우(multi-modal) 성능이 저하될 수 있다. 클래스당 복수의 프로토타입 또는 가우시안 혼합 모델을 활용한 확률론적 프로토타입 표현을 연구할 필요가 있다.

#### (2) 오픈셋/파셜 도메인 적응으로의 확장
폐쇄 범주 가정을 완화하여, 타겟 도메인에만 존재하는 클래스 또는 소스 도메인 클래스의 일부만 타겟에 존재하는 시나리오에서의 적용 방법을 연구해야 한다.

#### (3) 대규모 사전학습 모델(Foundation Models)과의 통합
ViT, CLIP 등 대규모 사전학습 모델의 강력한 특징 표현과 PCT의 프로토타입 정렬 메커니즘을 통합할 경우, 도메인 간 일반화 성능의 추가적인 향상이 기대된다.

#### (4) 이론적 보장 강화
현재 논문은 실험적 검증에 집중되어 있으며, 양방향 전송 손실의 수렴성 및 일반화 오류 경계에 대한 엄밀한 이론적 분석이 부족하다. Ben-David et al.의 이론 프레임워크를 확장하여 PCT의 수렴 조건과 일반화 경계를 분석하는 후속 연구가 필요하다.

#### (5) 동적 프로토타입 업데이트 메커니즘
현재는 학습이 진행되면서 프로토타입(= 분류기 가중치)이 소스 데이터에만 의존하여 업데이트된다. 타겟 도메인의 정보를 프로토타입 업데이트에 점진적으로 반영하는 동적 메커니즘(예: EMA 기반 메모리 뱅크)을 연구할 수 있다.

#### (6) 연속적 도메인 적응(Continual DA)
단일 도메인 전이가 아닌, 시간이 지남에 따라 도메인이 변화하는 환경에서 PCT의 프로토타입을 어떻게 효율적으로 갱신할 수 있는지 연구가 필요하다.

#### (7) 세그멘테이션/검출 등 다른 비전 태스크로의 확장
현재 분류 태스크에 집중된 PCT를 의미론적 분할(semantic segmentation), 객체 탐지(object detection) 등 더 복잡한 태스크에 적용할 때, 픽셀/영역 단위의 프로토타입 정의와 전송 손실 설계가 추가적으로 고려되어야 한다.

---

## 참고자료

**주요 참고 논문 (논문 내 직접 인용)**:

1. **Tanwisuth et al. (2021)** - "A Prototype-Oriented Framework for Unsupervised Domain Adaptation" - NeurIPS 2021 *(본 논문)*
2. **Ganin & Lempitsky (2015)** - "Unsupervised Domain Adaptation by Backpropagation" - ICML 2015
3. **Long et al. (2017)** - "Deep Transfer Learning with Joint Adaptation Networks" - ICML 2017
4. **Long et al. (2018)** - "Conditional Adversarial Domain Adaptation" - NeurIPS 2018
5. **Zhang et al. (2019)** - "Bridging Theory and Algorithm for Domain Adaptation (MDD)" - ICML 2019
6. **Liang et al. (2020)** - "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation (SHOT)" - ICML 2020
7. **Tachet des Combes et al. (2020)** - "Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift" - NeurIPS 2020
8. **Damodaran et al. (2018)** - "DeepJDOT: Deep Joint Distribution Optimal Transport for Unsupervised Domain Adaptation"
9. **Zheng & Zhou (2020)** - "Comparing Probability Distributions with Conditional Transport" - arXiv:2012.14100
10. **Saerens et al. (2002)** - "Adjusting the Outputs of a Classifier to New A Priori Probabilities: A Simple Procedure" - Neural Computation
11. **Ben-David et al. (2010)** - "A Theory of Learning from Different Domains" - Machine Learning
12. **Chen & He (2021)** - "Exploring Simple Siamese Representation Learning" - CVPR 2021
13. **Peng et al. (2019)** - "Moment Matching for Multi-Source Domain Adaptation (DomainNet)" - ICCV 2019
14. **Saito et al. (2019)** - "Semi-Supervised Domain Adaptation via Minimax Entropy" - ICCV 2019
15. **Pan et al. (2019)** - "Transferrable Prototypical Networks for Unsupervised Domain Adaptation (TPN)" - CVPR 2019
