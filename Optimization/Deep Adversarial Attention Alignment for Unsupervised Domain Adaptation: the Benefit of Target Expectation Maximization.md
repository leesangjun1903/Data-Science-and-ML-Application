# Deep Adversarial Attention Alignment for Unsupervised Domain Adaptation: the Benefit of Target Expectation Maximization

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문(ECCV 2018)은 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)에서 기존 방법들이 **완전 연결층(FC Layer)의 고수준 표현만 정렬**하는 한계를 지적하며, **합성곱층(Convolutional Layer)의 어텐션 메커니즘**까지 정렬하는 새로운 프레임워크를 제안한다.

> **핵심 가정:** 이미지의 판별적(discriminative) 영역은 도메인 스타일 변화에 상대적으로 불변(invariant)하다.

### 두 가지 주요 기여

| 기여 | 내용 |
|------|------|
| **1. 적대적 어텐션 정렬 (Adversarial Attention Alignment)** | CycleGAN으로 도메인 간 데이터 쌍을 생성하고, 소스 네트워크의 모든 합성곱층 어텐션 맵을 타겟 네트워크가 모방하도록 강제 |
| **2. EM 기반 타겟 학습 (Target Expectation Maximization)** | 레이블 없는 타겟 데이터에 대해 카테고리 사후 분포를 추정하여 의사 레이블(pseudo-label) 오류 누적 문제를 해소 |

**성능:** Office-31 벤치마크에서 당시 SOTA 대비 평균 **+2.6%** 향상, 어려운 전이 태스크 $D \rightarrow A$에서 **+5.1%** 향상

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 문제 1: 합성곱층의 도메인 불일치 무시

기존 UDA 방법(DAN, JAN, RevGrad 등)은 FC 레이어의 표현만 정렬한다. 그러나:

- FC층의 그래디언트가 깊은 합성곱층까지 역전파될 때 **소실(vanishing)/폭발(explosion)** 문제 발생
- 도메인 불일치는 합성곱층에서부터 시작될 수 있으므로 **네트워크 후단(tail)만 수정하는 것은 비효율적**
- 소스 네트워크를 타겟 도메인에 직접 적용 시 어텐션 메커니즘이 **배경이나 비판별적 영역에 집중**하는 현상 발생 (Fig. 1 참조)

#### 문제 2: 의사 레이블의 오류 누적

기존 pseudo-label 방법들은 타겟 네트워크가 예측한 레이블로 네트워크를 반복적으로 업데이트하므로 **초기 오류가 누적**된다.

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 적대적 데이터 쌍 생성 (CycleGAN)

GAN 손실:

$$\mathcal{L}^{GAN}(G^{ST}, D^T, X^S, X^T) = \mathbb{E}_{x^T}[\log D^T(x^T)] + \mathbb{E}_{x^S}[1 - \log D^T(G^{ST}(x^S))]$$

사이클 일관성 손실:

$$\mathcal{L}^{cyc}(G^{ST}, G^{TS}) = \mathbb{E}_{x^S}[\|G^{TS}(G^{ST}(x^S)) - x^S\|_1] + \mathbb{E}_{x^T}[\|G^{ST}(G^{TS}(x^T)) - x^T\|_1]$$

CycleGAN 전체 목적함수:

$$\mathcal{L}^{cyc}(G, F, D_X, D_Y) = \mathcal{L}^{GAN}(G^{ST}, D^T, X^S, X^T) + \mathcal{L}^{GAN}(G^{TS}, D^S, X^T, X^S) + \lambda\mathcal{L}^{cyc}(G^{ST}, G^{TS})$$

#### Step 2: 어텐션 맵 정의

레이어 $l$에서의 입력 $x$에 대한 어텐션 맵:

$$A_l(x) = \sum_c |F_{l,c}(x)|^2$$

여기서 $F_{l,c}(x)$는 레이어 $l$의 $c$번째 채널 피처맵 (element-wise 연산)

#### Step 3: 어텐션 정렬 손실 (Attention Alignment Loss)

```math
\mathcal{L}^{AT} = \sum_l \left\{ \sum_i \left\| \frac{A_l^S(x_i^S)}{\|A_l^S(x_i^S)\|_2} - \frac{A_l^T(x_i^S)}{\|A_l^T(x_i^S)\|_2} \right\|_2 + \sum_j \left\| \frac{A_l^S(x_j^S)}{\|A_l^S(x_j^S)\|_2} - \frac{A_l^T(\tilde{x}_j^T)}{\|A_l^T(\tilde{x}_j^T)\|_2} \right\|_2 \right.
```

```math
\left. + \sum_m \left\| \frac{A_l^S(\tilde{x}_m^S)}{\|A_l^S(\tilde{x}_m^S)\|_2} - \frac{A_l^T(\tilde{x}_m^S)}{\|A_l^T(\tilde{x}_m^S)\|_2} \right\|_2 + \sum_n \left\| \frac{A_l^S(\tilde{x}_n^S)}{\|A_l^S(\tilde{x}_n^S)\|_2} - \frac{A_l^T(x_n^T)}{\|A_l^T(x_n^T)\|_2} \right\|_2 \right\}
```

- $x^S, x^T$: 실제 소스/타겟 도메인 데이터
- $\tilde{x}^T = G^{ST}(x^S)$: 합성 타겟 데이터
- $\tilde{x}^S = G^{TS}(x^T)$: 합성 소스 데이터
- $A_l^S, A_l^T$: 소스/타겟 네트워크의 레이어 $l$ 어텐션 맵

#### Step 4: EM 알고리즘을 통한 타겟 학습

**소스 데이터에 대한 교차 엔트로피 손실:**

$$\mathcal{L}^{CE} = -\left[\sum_i \log p_\theta(y_i^S | x_i^S) + \sum_j \log p_\theta(y_j^S | \tilde{x}_j^T)\right]$$

**타겟 데이터 로그-우도 최대화 목표:**

$$\sum_i \log p_\theta(x_i^T)$$

**(E-step)** 사후 레이블 분포 추정:

$$p_{\theta_{t-1}}(z|x) = \frac{p_{\theta_{t-1}}(x|z)p(z)}{\sum_z p_{\theta_{t-1}}(x|z)p(z)}$$

$p(z)$를 균등 분포로 가정하면: $p_{\theta_{t-1}}(z|x) = \alpha p_{\theta_{t-1}}(x|z)$

**(M-step)** 사후 분포 기반 하한 최대화:

$$\sum_z p_{\theta_{t-1}}(z|x) \log p_{\theta_t}(z|x)$$

**EM 손실 함수:**

```math
\mathcal{L}^{EM} = -\left\{ \sum_i \sum_{z_i} p_{\theta^{post}}(z_i|x_i^T) \log p_\theta(z_i|x_i^T) + \sum_j \sum_{z_j} p_{\theta^{post}}(z_j|x_j^T) \log p_\theta(z_j|\tilde{x}_j^S) \right\}
```

**EM 안정화를 위한 3가지 수정:**

| 수정 | 내용 |
|------|------|
| **A) 비동기 업데이트** | 독립 네트워크 $M^{post}$로 $p(z\|x)$ 추정, $N$ 스텝마다 동기화 |
| **B) 노이즈 필터링** | $\max_z p(z\|x) < p_t$ 인 샘플 학습에서 제외 |
| **C) 학습률 재초기화** | $M^{post}$ 업데이트 후 학습률 스케줄 재초기화 |

#### Step 5: 전체 목적함수

$$\min_\theta \mathcal{L}^{full} = \mathcal{L}^{CE} + \mathcal{L}^{EM} + \beta \mathcal{L}^{AT}$$

여기서 $\beta$는 어텐션 정렬 페널티 강도를 조절하는 하이퍼파라미터

---

### 2.3 모델 구조

```
[CycleGAN]
  Source Domain ──G^{ST}──> Synthetic Target
  Target Domain ──G^{TS}──> Synthetic Source

[Source Network] (고정됨)
  ImageNet 사전학습 ResNet-50 → Source 도메인 파인튜닝 → 고정

[Target Network] (학습됨)
  ResNet-50 (소스 네트워크와 동일 합성곱 구조)
  ↑
  입력: 실제 소스 + 합성 타겟 + 실제 타겟 + 합성 소스

[M^{post} 네트워크]
  타겟 네트워크와 동일 구조
  N 스텝마다 타겟 네트워크 파라미터와 동기화
  → p(z|x^T) 추정에 사용

[손실 함수]
  L^CE: 레이블 있는 데이터(소스 + 합성 타겟)
  L^EM: 레이블 없는 데이터(타겟 + 합성 소스)
  L^AT: 모든 합성곱층 어텐션 정렬 (첫 번째 레이어 제외)
```

---

### 2.4 성능 향상

#### Office-31 결과 (ResNet-50 기반)

| 방법 | A→W | D→W | W→D | A→D | D→A | W→A | **평균** |
|------|-----|-----|-----|-----|-----|-----|---------|
| ResNet-50 (소스만) | 68.4 | 96.7 | 99.3 | 68.9 | 62.5 | 60.7 | 76.1 |
| RevGrad | 82.0 | 96.9 | 99.1 | 79.7 | 68.2 | 67.4 | 82.2 |
| JAN-A | 86.0 | 96.7 | 99.7 | 85.1 | 69.2 | 70.7 | 84.6 |
| **Ours (w $\mathcal{L}^{AT}$)** | **86.8** | **99.3** | **100** | **88.8** | **74.3** | **73.9** | **87.2** |

#### MNIST → MNIST-M 결과

| 방법 | 정확도 (%) |
|------|-----------|
| RevGrad | 81.5 |
| DSN | 83.2 |
| ADA | 85.9 |
| **Ours (w $\mathcal{L}^{AT}$)** | **95.6** |
| PixelDA | 98.2 |

#### 어텐션 측정 지표 비교 (Table 4)

| 측정 지표 | 평균 |
|-----------|------|
| $L_1$-norm | 매우 낮음 |
| MMD | 74.9 |
| JMMD | 78.1 |
| **Ours ($L_2$)** | **80.9** |

---

### 2.5 한계

1. **CycleGAN 의존성:** 이미지 변환 품질이 전체 성능에 직접적 영향. CycleGAN의 변환 실패 시 어텐션 정렬이 잘못된 방향으로 유도될 수 있음
2. **계산 비용:** CycleGAN 사전 학습 + 소스 네트워크 학습 + 타겟 네트워크 학습 + $M^{post}$ 관리로 **3단계 학습 파이프라인** 필요
3. **소규모 데이터셋 한계:** PixelDA(98.2%)에 비해 MNIST→MNIST-M에서 성능이 낮음 (95.6%)
4. **가정의 한계:** "판별적 영역은 도메인에 불변"이라는 가정이 모든 도메인 쌍에서 성립하지 않을 수 있음 (e.g., 의료 영상 ↔ 자연 이미지)
5. **EM 하이퍼파라미터 민감성:** 임계값 $p_t$, 동기화 주기 $N$ 등 추가적인 하이퍼파라미터 튜닝 필요
6. **Office-31만 평가:** 보다 대규모의 어려운 벤치마크(Office-Home, DomainNet 등)에서의 검증 부재

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 가능케 하는 핵심 메커니즘

#### (1) 다층 어텐션 정렬의 일반화 효과

기존 방법들이 FC층의 **의미적(semantic) 표현**만 정렬하는 것과 달리, 본 논문은 모든 합성곱층의 **구조적(structural) 어텐션**을 정렬함으로써:

$$\mathcal{L}^{AT} = \sum_{\text{모든 레이어 } l} \text{(소스-타겟 어텐션 거리)}$$

- **저수준 → 고수준**까지 계층적으로 도메인 불변 표현을 학습
- 그래디언트가 네트워크 전체에 직접 전달되어 합성곱층이 **직접적으로 업데이트**됨
- Fig. 1에서 보듯, 어텐션 정렬 후 타겟 네트워크의 어텐션이 **레이블 있는 타겟으로 학습한 것보다 더 좋은** 판별적 영역 집중을 보임

#### (2) EM의 일반화 기여

단순 pseudo-label 방식 대신 **카테고리 분포(soft label)**를 사용:

$$\mathcal{L}^{EM} = -\sum_i \sum_{z_i} \underbrace{p_{\theta^{post}}(z_i|x_i^T)}_{\text{소프트 타겟}} \log p_\theta(z_i|x_i^T)$$

- 이는 **지식 증류(Knowledge Distillation)**와 유사한 효과로 과적합을 방지
- 비동기 업데이트($M^{post}$)로 확률 분포가 급격히 변하지 않아 **안정적 수렴** 보장
- 노이즈 필터링으로 **불확실한 샘플 제거** → 일반화 성능 유지

#### (3) 데이터 증강을 통한 일반화

CycleGAN으로 생성된 합성 데이터($\tilde{x}^T, \tilde{x}^S$)를 활용:
- 실제 소스($x^S$) + 합성 타겟($\tilde{x}^T$) + 실제 타겟($x^T$) + 합성 소스($\tilde{x}^S$)의 **4가지 데이터 타입** 혼합 학습
- 다양한 도메인 간 이미지 대응 쌍이 타겟 네트워크의 **도메인 불변 특징 학습** 촉진

#### (4) 어텐션 정렬과 EM의 상호 보완적 일반화

```
어텐션 정렬 → 네트워크가 의미있는 영역에 집중하도록 정규화
      ↓
EM 학습 → 더 정확한 레이블 분포 추정 가능
      ↓  
더 나은 어텐션 가이드 → 더 나은 EM 학습 (선순환)
```

- **어텐션 정렬**: 판별적 정보에 집중하도록 **구조적 정규화** 제공
- **EM**: 타겟 도메인 통계에 맞는 **데이터 기반 정규화** 제공
- 두 메커니즘의 협력으로 국소 최적(local optima)에 빠지는 것을 방지

### 3.2 일반화 성능의 실험적 증거

- **어려운 태스크에서 더 큰 향상**: $D \rightarrow A$ (+5.1%), $W \rightarrow A$ (+3.2%) — 도메인 차이가 클수록 합성곱층 정렬의 중요성 증가
- $\mathcal{L}^{AT}$ 추가 시 수렴 속도 향상 (Fig. 4) — 더 나은 최적화 경로 = 더 나은 일반화
- 어텐션 정렬 후 타겟 도메인 이미지의 어텐션 시각화가 지도 학습 모델보다 우수

### 3.3 일반화의 잠재적 한계와 개선 방향

| 한계 | 개선 방향 |
|------|-----------|
| 단일 소스 도메인 가정 | 멀티 소스 도메인 어텐션 앙상블 |
| 클래스 불균형 처리 부재 | 클래스 조건부 어텐션 정렬 |
| 정적 소스 네트워크 | 적응형 소스 네트워크 업데이트 |

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

#### (1) 합성곱층 정렬의 패러다임 전환

본 논문은 UDA에서 **"어디서 정렬할 것인가"**에 대한 패러다임을 FC층 중심에서 **합성곱층 포함 전체 네트워크**로 전환하는 데 기여했다. 이는 이후 다음 연구들에 영감을 제공:

- **TransNorm (Wang et al., 2019)**: 배치 정규화 통계량 전이
- **ATDOC (Liu et al., 2021)**: 어텐션 기반 타겟 도메인 최적화
- **CDTrans (Xu et al., 2021)**: 트랜스포머 기반 교차 도메인 어텐션

#### (2) GAN+분류 통합 프레임워크의 선구자

CycleGAN을 활용한 **이미지 수준 도메인 브리지** + **특징 수준 정렬**의 결합은 이후 표준적 접근법이 됨:
- **UNIT, CycADA** 등의 후속 연구 방향 제시

#### (3) 소프트 레이블 기반 UDA 학습의 정착

EM을 통한 카테고리 분포 추정은 이후 **엔트로피 최소화(entropy minimization)**, **자기 훈련(self-training with soft labels)** 방법들의 이론적 기반이 됨

### 4.2 앞으로 연구 시 고려할 점

#### (1) 트랜스포머 시대의 어텐션 정렬

ViT(Vision Transformer), DINO, CLIP 등 **셀프-어텐션 기반 모델**의 등장으로:
- CNN의 공간적 어텐션 맵 대신 **멀티헤드 셀프-어텐션 행렬** 정렬 연구 필요
- 트랜스포머의 어텐션은 이미 학습된 구조적 정보를 포함하므로 다른 정렬 전략 필요

#### (2) CycleGAN 의존성 탈피

- 최신 diffusion 모델 기반 도메인 변환 활용 가능성
- 이미지 변환 없이 **특징 공간에서의 대응 관계** 직접 학습 (e.g., OT(Optimal Transport) 기반)

#### (3) 레이블 효율적 확장

- Few-shot DA, Source-free DA 환경에서의 적용성 검토
- **Source-free UDA**: 소스 데이터 없이 소스 네트워크만으로 어텐션 가이드 가능 여부

#### (4) 대규모 벤치마크 검증 필요

Office-31(31클래스, 4K 이미지)를 넘어:
- **DomainNet** (345클래스, 600K 이미지)
- **VisDA** (합성→실제 대규모 전이)
에서의 확장성 검증 필요

#### (5) 이론적 보장 강화

- 어텐션 정렬과 도메인 차이 감소 간의 **이론적 상계(upper bound)** 분석 필요
- Ben-David et al.의 도메인 적응 이론과의 연결

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **중요 고지:** 아래 2020년 이후 연구들에 대한 상세 내용은 제 학습 데이터 기준(2023년 초)으로 알려진 정보이며, 일부 세부 수치는 원 논문 직접 확인을 권장합니다.

### 5.1 주요 후속 연구 흐름

#### 흐름 1: 트랜스포머 기반 UDA

| 논문 | 핵심 아이디어 | 본 논문과의 관계 |
|------|--------------|----------------|
| **CDTrans** (Xu et al., ICLR 2022) | 크로스-도메인 트랜스포머 어텐션으로 도메인 간 특징 정렬 | 본 논문의 어텐션 정렬을 트랜스포머로 확장 |
| **TVT** (Yang et al., 2022) | 트랜스포머 비전 어텐션 전이 | 어텐션 전이 아이디어 계승 |
| **SWD** | 슬라이스 와서스타인 거리 기반 정렬 | 분포 거리 측정 개선 |

#### 흐름 2: 소스-프리 UDA (Source-Free DA)

본 논문과 달리 **소스 데이터 없이** 소스 모델만으로 적응:

| 논문 | 핵심 아이디어 |
|------|--------------|
| **SHOT** (Liang et al., ICML 2020) | 소스 가설 전이, 정보 최대화 |
| **G-SFDA** (Yang et al., 2021) | 그래프 기반 소스-프리 적응 |

**비교:** 본 논문은 소스 네트워크가 고정되어 있지만 소스 데이터를 직접 사용한다는 점에서 소스-프리 방법과 차별화

#### 흐름 3: 엔트로피 최소화 및 자기 학습

$$\mathcal{L}_{ent} = -\sum_x \sum_k p_\theta(k|x) \log p_\theta(k|x)$$

| 논문 | 핵심 아이디어 | 본 논문과의 차이 |
|------|--------------|----------------|
| **ATDOC** (Liu et al., 2021) | 어텐션 기반 지역 클러스터링 | 클러스터 구조 명시적 활용 |
| **NRC** (Yang et al., NeurIPS 2021) | 이웃 관계 기반 클러스터링 | 샘플 간 관계 명시적 모델링 |

**비교:** 본 논문의 EM이 소프트 레이블을 활용하는 것과 달리, 이후 연구들은 **클러스터 구조**를 더 명시적으로 활용

#### 흐름 4: 대비 학습 기반 UDA

| 논문 | 핵심 아이디어 |
|------|--------------|
| **CDCL** (Wang et al., 2021) | 교차 도메인 대비 학습 |
| **PMTrans** (Zhu et al., ECCV 2022) | 패치 매칭 기반 트랜스포머 |

### 5.2 벤치마크 성능 비교 (Office-31, ResNet-50 기준)

| 방법 | 연도 | 평균 정확도 | 핵심 기법 |
|------|------|------------|----------|
| **본 논문 (DAAA)** | 2018 | 87.2% | 어텐션 정렬 + EM |
| CDAN (Long et al.) | 2018 | 87.7% | 조건부 적대적 학습 |
| BSP (Chen et al.) | 2019 | 88.0% | 배치 스펙트럼 페널티 |
| MDD (Zhang et al.) | 2019 | 88.9% | 마진 분산 불일치 |
| ATDOC (Liu et al.) | 2021 | ~90%+ | 어텐션 기반 클러스터링 |
| CDTrans (Xu et al.) | 2022 | ~90%+ | 트랜스포머 어텐션 |

> ⚠️ 2020년 이후 수치들은 각 논문의 공식 발표 결과를 직접 확인하시기 바랍니다. 백본, 데이터 분할, 평가 프로토콜에 따라 수치가 달라질 수 있습니다.

### 5.3 본 논문의 한계가 해소된 부분과 여전히 열린 문제

| 본 논문의 한계 | 이후 연구에서의 해소 | 여전히 열린 문제 |
|--------------|-------------------|----------------|
| CycleGAN 의존성 | Feature-level 대응 학습 (OT 기반) | Diffusion 모델 활용 가능성 |
| FC층 정렬 부재 | 전체 네트워크 정렬 일반화 | 최적 정렬 레이어 자동 탐색 |
| Office-31만 검증 | DomainNet 등 대규모 검증 | 수백만 클래스 환경 |
| 이론적 보장 부재 | PAC 학습 이론 기반 분석 일부 | 어텐션 정렬의 이론적 보장 |

---

## 참고 자료

**본 논문 (직접 참조):**
- Kang, G., Zheng, L., Yan, Y., & Yang, Y. (2018). "Deep Adversarial Attention Alignment for Unsupervised Domain Adaptation: the Benefit of Target Expectation Maximization." *ECCV 2018*. (제공된 PDF 원문)

**논문 내 참조 문헌 (원문 Reference List 기반):**
- [7] Ganin, Y., & Lempitsky, V. (2015). "Unsupervised domain adaptation by backpropagation." *ICML 2015*.
- [16] Long, M., et al. (2015). "Learning transferable features with deep adaptation networks." *ICML 2015*.
- [17] Long, M., et al. (2017). "Deep transfer learning with joint adaptation networks." *ICML 2017*.
- [28] Zagoruyko, S., & Komodakis, N. (2017). "Paying more attention to attention." *ICLR 2017*.
- [33] Zhu, J.Y., et al. (2017). "Unpaired image-to-image translation using cycle-consistent adversarial networks." *ICCV 2017*.
- [2] Bousmalis, K., et al. (2017). "Unsupervised pixel-level domain adaptation with generative adversarial networks." *CVPR 2017*.
- [3] Bousmalis, K., et al. (2016). "Domain separation networks." *NeurIPS 2016*.
- [8] Haeusser, P., et al. (2017). "Associative domain adaptation." *ICCV 2017*.
- [9] He, K., et al. (2016). "Deep residual learning for image recognition." *CVPR 2016*.
- [20] Selvaraju, R.R., et al. (2017). "Grad-CAM." *ICCV 2017*.

**2020년 이후 관련 연구 (일반 지식 기반, 직접 확인 권장):**
- Liang, J., et al. (2020). "Do we really need to access the source data? Source hypothesis transfer for unsupervised domain adaptation." *ICML 2020*.
- Xu, T., et al. (2022). "CDTrans: Cross-domain transformer for unsupervised domain adaptation." *ICLR 2022*.
- Liu, Y., et al. (2021). "Cycle self-training for domain adaptation." *NeurIPS 2021*.
