# LoopFM: Learning frOm HistOrical RePresentations of Foundation Model for Recommendation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

LoopFM은 **지식 증류(Knowledge Distillation, KD)의 대역폭 병목(bandwidth bottleneck) 문제**를 해결하기 위해 제안된 프레임워크입니다. 기존 KD는 대형 Foundation Model(FM)의 지식을 단일 스칼라 예측값으로 압축하여 소형 Vertical Model(VM)에 전달하는데, 이는 FM이 커질수록 **전달 비율(Transfer Ratio, TR)**이 감소하는 근본적 한계를 가집니다.

$$\text{TR} = \frac{\Delta\text{NE}_\text{VM}}{\Delta\text{NE}_\text{FM}}$$

LoopFM은 FM의 중간 레이어 임베딩(intermediate embeddings)을 **구조화된 입력 피처(structured input features)**—구체적으로 사용자별 시간순 시퀀스—로 변환하여 VM에 직접 공급함으로써, 스칼라 KD 채널과 **상보적(complementary)**인 고대역폭 임베딩 채널을 추가합니다. 핵심은 **과거(historical)** 임베딩만 사용하므로, 서빙 시점에 실시간 FM 추론이 불필요하다는 점입니다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| **모듈형 프레임워크** | 추출(Extraction) → 압축(Compression) → 구조화(Structuring) 3단계, 각 단계 독립 구성 가능 |
| **이론적 분석** | 정보 이득 분해, 전달 비율 하한, 시퀀스 길이 단조성 정리 |
| **실증 검증** | 3개 공개 벤치마크 + 산업 규모 조 단위 파라미터 FM 시스템에서 검증 |
| **생산 배포** | Y1H1 +0.5%, Y1H2 +1.03% 및 +1.22% 광고 전환율 향상 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

산업용 추천 시스템은 두 계층 구조로 운영됩니다:
- **FM (Foundation Model)**: 수조 개 파라미터, 오프라인 학습, 풍부한 크로스도메인 피처 사용
- **VM (Vertical Model)**: 수백만 개 파라미터, 엄격한 지연시간 제약 하에 실시간 서빙

기존 외부 KD는 FM의 스칼라 예측 $\hat{y}^F$를 소프트 레이블로 전달하나, FM이 클수록:

1. **대역폭 병목**: 단일 스칼라는 FM이 학습한 풍부한 크로스도메인 피처, 다층 상호작용 패턴을 표현 불가
2. **피처 격차(feature gap)**: FM은 VM보다 훨씬 많은 피처를 학습하나, 스칼라 KD는 이를 전달 불가
3. **TR 감소**: 교사-학생 용량 격차가 클수록 KD 효율 저하 (Cho & Hariharan, 2019)

### 2.2 제안 방법 및 수식

#### Stage 1: 임베딩 추출 (Embedding Extraction)

FM의 레이어 $l_1, \ldots, l_M$ 중 $K$개를 선택하여 각 예제 $(u, a)$에 대한 원시 임베딩을 연결:

$$\mathbf{e}^{(i)} = [\mathbf{h}^{(i)}_{l_1}; \ldots; \mathbf{h}^{(i)}_{l_K}] \in \mathbb{R}^D, \quad D = \sum_{k=1}^{K} d_{l_k}$$

#### Stage 2: 압축 (Compression)

오토인코더를 FM과 공동 학습하여 고차원 임베딩을 압축:

$$\mathbf{z}^{(i)} = f_{\text{enc}}(\mathbf{e}^{(i)}), \quad \hat{\mathbf{e}}^{(i)} = f_{\text{dec}}(\mathbf{z}^{(i)}), \quad \mathcal{L}_{\text{AE}} = \|\mathbf{e}^{(i)} - \hat{\mathbf{e}}^{(i)}\|_2^2 $$

- Stop-gradient로 오토인코더 그래디언트가 FM 백본에 흐르지 않도록 보장
- 인코더 마지막 레이어에 $\tanh$ 활성화 → 출력을 $[-1, 1]$로 제한 → INT4 양자화 용이

**Matryoshka 압축**: 임의의 접두사 $\mathbf{z}_{1:d'}$가 유효한 표현이 되도록 다중 목표 차원 $\mathcal{D}$에 대해 손실 합산:

$$\mathcal{L}_{\text{MAE}} = \sum_{d' \in \mathcal{D}} \|\mathbf{e} - f_{\text{dec}}^{(d')}(\mathbf{z}_{1:d'})\|_2^2$$

**INT4 양자화**:
$$z_{\text{quant}} = \text{round}(z \cdot 8).\text{clamp}(-8, 7) / 8$$
→ FP16 대비 4× 저장 공간 절감

#### Stage 3: 구조화 (Structuring)

사용자 키(user-keyed) 시간순 시퀀스 생성:

$$\mathbf{S}_k = [\mathbf{z}_{k,t_1}, \mathbf{z}_{k,t_2}, \ldots, \mathbf{z}_{k,t_L}], \quad t_L < \cdots < t_2 < t_1 < t_{\text{cur}} $$

- 현재 서빙 중인 샘플은 **제외** → 실시간 FM 추론 불필요
- 보존 윈도우(e.g., 30일), 최근 $L$개(e.g., 200개) 항목으로 절단

#### VM-side 통합

VM의 학습 손실:

$$\mathcal{L} = \mathcal{L}_{\text{task}}(\hat{y}^V, y) + \lambda \cdot \mathcal{L}_{\text{KD}}(\hat{y}^V, \hat{y}^F), \quad \hat{y}^V = g(\mathbf{x}_{\text{VM}}, \mathbf{S}_u; \Theta_V)$$

### 2.3 모델 구조

```
FM (Trillion-param)
├── Layer 1 → sg → ─┐
├── Layer 2 → sg → ─┤  concat → [e^(i)]  → Autoencoder → [z^(i)]
├── ...    → sg → ─┤                         ↓ (INT4 quantization)
└── Layer N → sg → ─┘                    Group by user_id, sort by time
                                              ↓
                                    S_u = [z_{u,t1}, z_{u,t2}, ..., z_{u,tL}]
                                              ↓
                                    VM (Seq Encoder: attention/pooling)
                                              ↓
                               Concat with other features → Interaction layers → Prediction
                                    (+ KD soft label from FM)
```

### 2.4 성능 향상

#### 공개 벤치마크

| 데이터셋 | 피처 수 | AUC 향상 (avg) | 범위 |
|----------|---------|---------------|------|
| TaobaoAd | 22 | +6.4% | +6.1~6.6% |
| KuaiVideo | 9 | +1.0% | +0.6~1.6% |
| Amazon Electronics | 6 | +0.5% | +0.02~1.14% |

- TaobaoAd, DeepFM VM 기준: KD(0.5980) → KD+LoopFM(0.6344), +6.09% relative
- 5 seeds 통계 검정: $p < 0.001$ (paired t-test)

#### 산업 시스템

- KD 대비 TR 약 **2배** 증가
- Y1H1: +0.5% 광고 전환율
- Y1H2: +1.03%, +1.22% (두 개별 출시)

### 2.5 한계

1. **스토리지**: 대규모 LoopFM 시퀀스는 상당한 저장 공간 필요
2. **콜드스타트**: 신규 사용자는 임베딩 히스토리 부재
3. **압축 손실**: 고차원 임베딩 압축 시 정보 손실 불가피
4. **추론 지연**: 시퀀스 피처가 VM 추론 비용 증가
5. **스케일 격차**: 공개 벤치마크(~6M 파라미터) 결과가 조 단위 파라미터 설정에 완전히 일반화되지 않을 수 있음
6. **이론 분석의 약점**: 경계값이 모집단 수준의 상호 정보와 베이즈 위험으로 표현되며, 유한 샘플 효과, 최적화 문제, 모델 용량 제한 미반영

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 일반화 보장: 이득 분해 정리

**Theorem 1 (정보 이득 분해)**:

$$\mathcal{I}_{\text{LoopFM}}(\text{FM}_k) = \underbrace{\mathcal{I}_{\text{temporal}}}_{\text{사용자 행동 이력}} + \underbrace{\mathcal{I}_{\text{cross},k}}_{\text{피처 격차 이득}} - \underbrace{\mathcal{I}_{\text{residual},k}}_{\text{압축 손실}} $$

여기서:
- $\mathcal{I}\_{\text{temporal}} := I(\mathbf{H}\_u; y \mid \mathbf{x}^{(t)}_{\text{VM}}) \geq 0$: FM과 독립적인 사용자 히스토리의 예측 정보
- $\mathcal{I}\_{\text{cross},k} := I(\mathbf{S}^{(k)}\_u; y \mid \mathbf{x}^{(t)}\_{\text{VM}}, \mathbf{H}\_u) \geq 0$: FM의 추가 피처 $\mathbf{x}_{\text{extra}}$에서 오는 정보
- $\mathcal{I}\_{\text{residual},k} := I(\mathbf{H}\_u; y \mid \mathbf{x}^{(t)}_{\text{VM}}, \mathbf{S}^{(k)}_u) \geq 0$: 압축 비용

데이터 처리 부등식에 의해:

$$\mathcal{I}_{\text{cross},k} \leq I(\mathbf{x}^{(t_1:t_L)}_{\text{extra},k}; y \mid \mathbf{x}^{(t)}_{\text{VM}}, \mathbf{H}_u) =: \mathcal{I}_{\text{feature-raw},k}$$

### 3.2 파이프라인 품질과 보존 파라미터

압축 잔차 $\mathcal{I}_{\text{residual},k}$의 파이프라인 분해 (Proposition 7):

$$\mathcal{I}_{\text{residual},k} \leq \underbrace{\ell_{\text{repr},k}(p_k)}_{\text{FM 표현 잔차}} + \underbrace{\ell_{\text{AE},k}(d)}_{\text{AE 압축 손실}} + \underbrace{\ell_{\text{Q},k}(b_k)}_{\text{양자화 손실}} $$

두 보존 파라미터 정의:
- $\tau_k \geq 0$: 시간적 파이프라인 손실 비율, $\mathcal{I}\_{\text{residual},k} \leq \tau_k \cdot \mathcal{I}_{\text{temporal}}$
- $\eta_k \in [0,1]$: 크로스-플랫폼 파이프라인 손실 비율, $\mathcal{I}\_{\text{cross},k} = (1-\eta_k) \cdot \mathcal{I}_{\text{feature-raw},k}$

**이득 샌드위치 (Corollary 10)**:

$$\underbrace{(1-\tau_k)\mathcal{I}_{\text{temporal}} + (1-\eta_k)\mathcal{I}_{\text{feature-raw},k}}_{\text{하한}} \leq \mathcal{I}_{\text{LoopFM}}(\text{FM}_k) \leq \underbrace{\mathcal{I}_{\text{temporal}} + (1-\eta_k)\mathcal{I}_{\text{feature-raw},k}}_{\text{상한}}$$

### 3.3 전달 비율 분석: Theorem 2

두 FM 세대 비교: FM₁(구) vs FM₂(신, 더 많은 피처)

```math
\Delta_{\text{teacher}} := R_{\text{ach}}(\text{FM}_1) - R_{\text{ach}}(\text{FM}_2) = \underbrace{R^*(\text{FM}_1) - R^*(\text{FM}_2)}_{\Delta_{\text{feat}} \geq 0} + \underbrace{\epsilon_{\text{over}}(p_1,m_1) - \epsilon_{\text{over}}(p_2,m_2)}_{\Delta_{\text{param}}}
```

$$\text{TR}_{\text{LoopFM}} := \frac{\Delta_{\text{LoopFM}}}{\Delta_{\text{teacher}}} $$

**초기 출시 시 하한 (Eq. 7)**:

$$\text{TR}_{\text{LoopFM}} \geq \frac{(1-\tau_2)\mathcal{I}_{\text{temporal}} + (1-\eta_2)\mathcal{I}_{\text{feature-raw},2}}{\Delta_{\text{teacher}}} $$

**양성 전달 시 (Assumption A3 하에서) (Eq. 8)**:

$$\text{TR}_{\text{LoopFM}} \geq \frac{-\tau_2 \mathcal{I}_{\text{temporal}} + (1-\eta_1)\kappa^{\text{hist}}_{\text{gap}}\delta}{\bar{\kappa}_{\text{gap}}\delta + \bar{\kappa}_{\text{over}}\xi_1 - \underline{\kappa}_{\text{over}}\xi_2} =: \text{TR}_{\text{LB}}(\delta) $$

여기서 $\delta := m_2 - m_1$ (피처 격차), $\xi_k := m_k/n + n/(p_k - m_k)$

### 3.4 피처 격차 단조성: Corollary 4

**Corollary 4**: Assumption A3 하에서 $\text{TR}_{\text{LB}}(\delta)$는 $\delta \geq 0$에 대해 단조 증가하며:

$$\text{TR}_{\text{LB}}(\delta) \xrightarrow{\delta \to \infty} \frac{(1-\eta_1)\kappa^{\text{hist}}_{\text{gap}}}{\bar{\kappa}_{\text{gap}}} > 0$$

**일반화 함의**: FM과 VM 간의 피처 격차 $\delta$가 클수록 LoopFM의 이득이 단조 증가합니다. 즉, VM이 학습할 수 없는 크로스도메인 피처가 많은 FM일수록 일반화 이점이 커집니다.

### 3.5 시퀀스 길이 단조성: Theorem 5

**Theorem 5**: LoopFM 정보 이득 $\mathcal{I}\_{\text{LoopFM},k}(L) := I(\mathbf{S}^{(k,L)}\_u; y \mid \mathbf{x}^{(t)}_{\text{VM}})$은 $L$에 대해 단조 비감소:

$$\mathcal{I}_{\text{LoopFM},k}(L+1) = \mathcal{I}_{\text{LoopFM},k}(L) + \underbrace{I\left(\mathbf{z}_{k,t_{L+1}}; y \mid \mathbf{x}^{(t)}_{\text{VM}}, \mathbf{S}^{(k,L)}_u\right)}_{\delta_{\text{LoopFM},k}(L) \geq 0} $$

상한: $\mathcal{I}^\*\_{\text{LoopFM},k} \leq H(y \mid \mathbf{x}^{(t)}_{\text{VM}}) \leq \log 2$ → 수렴 보장

### 3.6 일반화 성능 향상을 위한 실험적 근거

| 설정 | 일반화 함의 |
|------|------------|
| VM 용량이 작을수록 LoopFM 이득 증가 (Table 4) | FM-VM 용량 격차가 클수록 일반화 향상 |
| FM 용량 5단계(1.9M~31.5M)에서 AUC 격차 < 0.001 (Table 10) | 소형 FM도 강한 일반화 이득 제공 |
| 6개 VM 아키텍처 모두에서 일관된 개선 (Table 1) | 아키텍처 독립적 일반화 |
| KuaiVideo 피처 부족 시 이득 감소 | 피처 격차 이론과 일치 |
| 임베딩 차원 d=8~128 AUC 격차 0.0008 (Table 5) | 압축에 강건한 일반화 |
| Sum-pool도 강한 이득 (Table 3) | 단순 집계도 충분한 일반화력 보유 |

#### FM 체크포인트 신선도와 일반화

체크포인트를 고정(fixed)하면 임베딩 공간이 일관되어 시퀀스 인코더의 학습이 용이:
- 고정 체크포인트: centroid drift ≈ 0.000006~0.000017 → AUC 0.6344
- 매 split 업데이트: centroid drift ≈ 0.013~0.021 (~1000× 증가) → AUC 0.6311

**일반화 시사점**: 임베딩 공간의 일관성(consistency)이 개별 임베딩의 신선도(freshness)보다 일반화에 더 중요합니다.

#### 음성 전달(Negative Transfer) 방지

Assumption A3(파이프라인 품질 비악화)가 위배될 때 $\eta_2 > \eta_1$이면 음성 전달 가능:

$$\tau_1\mathcal{I}_{\text{temporal}} + (1-\eta_2)\mathcal{I}_{\text{feature-raw},2} - (1-\eta_1)\mathcal{I}_{\text{feature-raw},1} < 0$$

A3 하에서는 이 조건이 절대 만족되지 않아 음성 전달 방지 → **일반화 안정성 보장**

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 앞으로의 연구에 미치는 영향

#### 패러다임 전환: 스칼라 → 임베딩 전달

LoopFM은 FM-VM 지식 전달의 패러다임을 "단일 예측값 전달"에서 "풍부한 임베딩 시퀀스 전달"로 전환시킵니다. KD와 LoopFM이 **거의 직교적(orthogonal)인 지식**을 전달함을 실증함으로써, 향후 **다채널 지식 전달(multi-channel knowledge transfer)**이 FM-VM 시스템의 기본 설계 패턴이 될 수 있음을 시사합니다.

#### 이론적 틀의 확장성

이득 분해 이론(Theorem 1)과 전달 비율 분석(Theorem 2)은 추천 시스템에 국한되지 않고, **FM-VM 두 계층 구조를 가진 모든 대규모 ML 시스템**에 적용 가능합니다. 특히:
- Transfer ratio 지표($\text{TR} = \Delta\text{NE}\_\text{VM}/\Delta\text{NE}_\text{FM}$)는 FM 업그레이드 효과를 정량화하는 표준 지표로 활용 가능
- 피처 격차 $\delta$와 TR의 단조 관계는 FM 설계 시 어떤 새로운 피처가 VM에 가장 유익한지 안내

#### Self-LoopFM 가능성

FM이 자신의 과거 임베딩을 소비하는 **자기 개선 루프(self-improving loop)**는 대형 언어 모델의 자기 학습(self-play, constitutional AI)과 유사한 새로운 연구 방향을 제시합니다.

#### 시퀀스 모델링과 FM 표현의 융합

LoopFM은 사용자 행동 시퀀스 모델링(DIN, DIEN, SASRec, BERT4Rec 등)과 FM 표현 학습을 처음으로 체계적으로 융합합니다. 이는 향후 추천 시스템에서 **FM 임베딩 시퀀스를 1등 시민(first-class citizen) 피처**로 취급하는 연구 흐름을 촉진할 것으로 예상됩니다.

### 4.2 앞으로 연구 시 고려할 점

#### (A) 최적 레이어 선택 문제

현재는 경험적으로 얕은 레이어(Hidden-0)가 더 나은 성능을 보이나, 깊은 레이어일수록 학습된 상호작용 정보가 풍부합니다. 이 **깊이-압축 트레이드오프(depth-compression tradeoff)**를 이론적으로 최적화하는 연구가 필요합니다.

수식으로 표현하면, 최적 레이어 $l^*$는:

$$l^* = \arg\max_{l} \left[ I(h_l; y \mid \mathbf{x}^{(t)}_{\text{VM}}) - \ell_{\text{AE}}(d, l) \right]$$

#### (B) 임베딩 드리프트 관리

체크포인트 신선도 실험에서 드러난 임베딩 드리프트 문제는 실제 배포에서 중요합니다. 연구 방향:
- **앵커 학습(Anchored Training)**: 임베딩 공간 고정
- **투영 정렬(Projection Alignment)**: 새 체크포인트 공간 → 이전 공간으로의 선형 매핑 학습
- **EMA 기반 체크포인트 보간**: 신선도와 일관성 균형
- 드리프트 임계값($\approx 0.01$) 자동 탐지 메커니즘

#### (C) 콜드스타트 문제 해결

새 사용자에게 히스토리 임베딩이 없는 문제는 특히 성장하는 플랫폼에서 중요합니다:
- FM 임베딩의 인구통계학적/컨텍스트 기반 초기화
- 메타 학습(meta-learning) 기반 cold-start 특화 임베딩 생성
- 콜드스타트 전환 경계(cold-to-warm transition) 학습

#### (D) 그래프 기반 구조화 탐구

현재 시퀀스 구조에서 **사용자-아이템 이분 그래프(bipartite graph)**로의 확장:

$$\mathbf{S}_k \to \mathcal{G}_{\text{FM}} = (\mathcal{V}_{\text{user}}, \mathcal{V}_{\text{item}}, \mathcal{E}_{\text{FM}})$$

여기서 엣지에 FM 임베딩이 담기면 GNN 기반 VM 인코더가 더 풍부한 구조적 지식 활용 가능

#### (E) 벡터 양자화(VQ) 기반 압축

현재 INT4 스칼라 양자화 대신 **RQ-VAE** 등 벡터 공간 양자화로 압축 손실($\ell_Q$) 최소화:
$$\mathbf{z} \approx \sum_{j=1}^{J} \mathbf{c}_{j, \sigma_j(\mathbf{z})}$$

#### (F) 크로스도메인 전달

FM이 유기적 콘텐츠 도메인에서 학습하고, VM이 광고 도메인에서 해당 임베딩을 소비하는 시나리오:
- 도메인 간 피처 스키마 정렬 불필요 (임베딩이 불투명 피처로 소비되므로)
- 크로스도메인 트랜스퍼 비율 이론 확장 필요

#### (G) 유한 샘플 효과 반영

현재 이론은 모집단 수준 상호 정보(population-level MI)와 베이즈 위험으로 표현되어 유한 샘플 효과를 반영하지 않습니다:
- PAC-베이즈 프레임워크를 활용한 유한 샘플 보장
- 최적화 경로(optimization trajectory)가 TR에 미치는 영향 분석

---

## 5. 2020년 이후 최신 연구 비교 분석

### 5.1 지식 증류 관련 연구 비교

| 연구 | 방법 | LoopFM과의 차이 |
|------|------|----------------|
| **Hinton et al. (2015)** [KD 원본] | 소프트 레이블 전달 | LoopFM: 임베딩 시퀀스 전달로 대역폭 확장 |
| **FitNets (Romero et al., 2015)** | 은닉 활성화 매칭 (보조 손실) | LoopFM: 손실 최소화 아닌 피처로 직접 소비 |
| **CRD (Tian et al., 2020)** | 대조적 표현 증류, MI 최대화 | 현재 샘플 표현만, LoopFM은 역사적 시퀀스 |
| **Privileged Features Distillation (Xu et al., 2020)** | 학습 시간에만 사용 가능한 피처 활용 | LoopFM: 서빙 시에도 사용 가능한 임베딩 |
| **External KD (Liang et al., 2025)** | FM과 VM 학습 분리, 스칼라 소프트 레이블 | LoopFM: 이를 보완하는 임베딩 채널 추가 |
| **Kang et al. (2024)** | 이기종 모델 간 증류 | 스칼라 중심, LoopFM의 시퀀스 접근과 상보적 |
| **Cui et al. (2024)** | LLM 임베딩 → 경량 순차 모델 | 현재 샘플 중심, LoopFM은 역사 시퀀스 |
| **Khani et al. (2024)** | 온라인 랭킹 KD의 숨겨진 도전 분석 | LoopFM의 문제 의식과 일치 |

### 5.2 시퀀스 모델링 관련 연구 비교

| 연구 | 방법 | LoopFM과의 차이 |
|------|------|----------------|
| **DIN (Zhou et al., 2018)** | 어텐션 기반 행동 히스토리 | 원시 ID 피처 시퀀스, LoopFM은 FM 임베딩 시퀀스 |
| **DIEN (Zhou et al., 2019)** | 관심 진화 네트워크 | 원시 신호 모델링, LoopFM은 FM-enriched 표현 |
| **SASRec (Kang & McAuley, 2018)** | 자기 어텐션 순차 추천 | 단일 도메인 원시 ID, LoopFM은 크로스도메인 FM 임베딩 |
| **BERT4Rec (Sun et al., 2019)** | 양방향 Transformer 순차 추천 | 단일 도메인, LoopFM은 FM의 크로스도메인 지식 전달 |
| **SIM (Pi et al., 2020)** | 평생 시퀀스 모델링 | 원시 ID 기반, LoopFM은 FM 중간 표현 기반 |

### 5.3 FM 표현 및 대규모 시스템 관련 연구 비교

| 연구 | 방법 | LoopFM과의 차이 |
|------|------|----------------|
| **PinSage (Ying et al., 2018)** | 아이템 수준 그래프 임베딩 | 엔티티 수준, LoopFM은 상호작용 수준 임베딩 |
| **Wukong (Zhang et al., 2024a)** | 대규모 추천의 스케일링 법칙 | FM 용량 스케일링, LoopFM은 VM으로의 전달 효율성 |
| **Kunlun (Hou et al., 2026)** | 통합 아키텍처 기반 스케일링 법칙 | FM 설계, LoopFM은 FM→VM 전달 |
| **LLaTTE (Xiong et al., 2026)** | 다단계 시퀀스 모델링 스케일링 | 서빙 효율을 위한 비동기 캐싱 사용 |
| **IAT (Li et al., 2026)** [동시 연구] | 역사적 상호작용 표현 압축-시퀀스 파이프라인 | LoopFM과 유사한 고수준 아이디어이나, LoopFM은 FM→VM TR 향상 동기 + 이론적 분석 포함 |

### 5.4 종합 비교 포지셔닝

```
                    스칼라 전달 ←─────────────────→ 임베딩 전달
                         │                               │
현재 샘플 ──── KD, FitNets, CRD ────────── Current-Emb-as-Feature
                                               (실시간 FM 추론 필요)
                         │                               │
역사 시퀀스 ─── DIN/DIEN/SASRec (원시 ID) ──────── LoopFM ★
                                               (실시간 FM 추론 불필요)
```

LoopFM은 **(1) 역사적 임베딩 시퀀스** + **(2) 실시간 추론 불필요** + **(3) FM-VM 아키텍처 비결합**의 교차점에서 독특한 위치를 점유합니다.

---

## 참고 자료

**주요 참고 논문 (논문 내 인용 기준)**:

1. **LoopFM 본 논문**: Jiang, S. et al. (2026). "LoopFM: Learning frOm HistOrical RePresentations of Foundation Model for Recommendation." arXiv:2605.29280v1
2. Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the knowledge in a neural network." arXiv:1503.02531
3. Cho, J. H., & Hariharan, B. (2019). "On the efficacy of knowledge distillation." ICCV
4. Romero, A. et al. (2015). "FitNets: Hints for thin deep nets." ICLR
5. Tian, Y., Krishnan, D., & Isola, P. (2020). "Contrastive representation distillation." ICLR
6. Liang, M. et al. (2025). "External large foundation model: How to efficiently serve trillions of parameters for online ads recommendation." arXiv:2502.17494
7. Kusupati, A. et al. (2022). "Matryoshka representation learning." NeurIPS
8. Bartlett, P. L. et al. (2020). "Benign overfitting in linear regression." PNAS
9. Jacot, A. et al. (2018). "Neural tangent kernel: Convergence and generalization in neural networks." NeurIPS
10. Zhou, G. et al. (2018). "Deep interest network for click-through rate prediction." KDD (DIN)
11. Zhou, G. et al. (2019). "Deep interest evolution network for click-through rate prediction." AAAI (DIEN)
12. Kang, W. C., & McAuley, J. (2018). "Self-attentive sequential recommendation." ICDM (SASRec)
13. Sun, F. et al. (2019). "BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformers." CIKM
14. Pi, Q. et al. (2020). "Search-based user interest modeling with lifelong sequential behavior data." CIKM (SIM)
15. Xu, C. et al. (2020). "Privileged features distillation at Taobao recommendations." KDD
16. Yang, S. et al. (2022). "Toward understanding privileged features distillation in learning-to-rank." NeurIPS
17. Cui, Y. et al. (2024). "Distillation matters: Empowering sequential recommenders to match the performance of large language model." arXiv:2405.00338
18. Kang, S. et al. (2024). "Unbiased, effective, and efficient distillation from heterogeneous models for recommender systems." ACM TORS
19. Khani, N. et al. (2024). "Bridging the gap: Unpacking the hidden challenges in knowledge distillation for online ranking systems." RecSys
20. Li, X. et al. (2026). "IAT: Instance-as-token compression for historical user sequence modeling." arXiv:2604.08933
21. Wang, K., Muthukumar, V., & Thrampoulidis, C. (2023). "Benign overfitting in multiclass classification." NeurIPS
22. Zhang, B. et al. (2024a). "Wukong: Towards a scaling law for large-scale recommendation." ICML
23. Guo, H. et al. (2017). "DeepFM: a factorization-machine based neural network for CTR prediction." IJCAI
24. Tishby, N., & Zaslavsky, N. (2015). "Deep learning and the information bottleneck principle." arXiv:1503.02406
25. Lee, D. et al. (2022). "Autoregressive image generation using residual quantization." CVPR (RQ-VAE)
26. Zhu, J. et al. (2022). "BARS: towards open benchmarking for recommender systems." SIGIR

> **정확도 주의**: 본 답변은 제공된 PDF 원문(arXiv:2605.29280v1)에 근거하여 작성되었습니다. 2026년 5월 28일 게재된 최신 논문으로, 일부 인용 논문(Hou et al., 2026; Li et al., 2026 등)은 아직 게재 전 preprint 상태일 수 있습니다.
