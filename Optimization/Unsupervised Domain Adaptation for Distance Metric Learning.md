# Unsupervised Domain Adaptation for Distance Metric Learning

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **소스 도메인과 타겟 도메인의 레이블 공간이 서로 겹치지 않는(disjoint label spaces)** 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA) 문제를 다룹니다. 기존의 두 가지 주류 접근법인 도메인 불일치 감소 학습(Domain Discrepancy Reduction Learning)과 반지도 학습(Semi-supervised Learning)은 모두 소스와 타겟 도메인이 **공통 레이블 공간을 공유한다는 가정** 하에 설계되어, 이 시나리오에는 적용할 수 없습니다.

논문의 핵심 주장은:

> **분류(Classification) 태스크를 검증(Verification) 태스크로 재정의함으로써, 비겹침 레이블 공간을 가진 두 도메인 간의 거리 메트릭 적응(Cross-Domain Distance Metric Adaptation, CD2MA)이 가능하다.**

### 주요 기여 (4가지)

| 기여 | 설명 |
|------|------|
| **이론적 일반화** | Ben-David et al. (2007)의 도메인 적응 이론을 검증 태스크로 확장, 타겟 도메인 검증 손실의 일반화 경계(generalization bound) 도출 |
| **Feature Transfer Network (FTN)** | 도메인 분리와 정렬을 동시에 수행하는 새로운 네트워크 아키텍처 제안 |
| **Multi-Class Entropy Minimization (MCEM)** | 비레이블 타겟 도메인에서 HDBSCAN 클러스터링 기반 엔트로피 최소화 손실 제안 |
| **실용적 응용** | 민족 간 얼굴 인식(cross-ethnicity face recognition)에서 인종 편향 극복 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**Cross-Domain Distance Metric Adaptation (CD2MA)**

기존 UDA의 한계:

$$\text{기존 UDA} \begin{cases} \text{Domain Divergence Reduction} & \Rightarrow \mathcal{Y}_S = \mathcal{Y}_T \text{ 필요} \\ \text{Semi-supervised Learning} & \Rightarrow \mathcal{Y}_S = \mathcal{Y}_T \text{ 필요} \end{cases}$$

본 논문이 해결하는 문제: $\mathcal{Y}_S \cap \mathcal{Y}_T = \emptyset$ (완전 비겹침 레이블 공간)

**구체적 예시:** Caucasian 얼굴 데이터(레이블 있음, 소스) → African-American 얼굴 데이터(레이블 없음, 타겟)

---

### 2.2 이론적 배경: 검증 태스크로의 재정의

**핵심 아이디어:** 분류 문제를 이진 검증(Binary Verification) 문제로 변환

$$\text{입력: }(x_1, x_2) \rightarrow \text{레이블: } \begin{cases} 1 & \text{같은 클래스} \\ 0 & \text{다른 클래스} \end{cases}$$

이 변환을 통해 Ben-David et al. (2007)의 Theorem 1을 검증 태스크에 직접 적용할 수 있습니다:

$$\epsilon_T(h) \leq \hat{\epsilon}_S(h) + \sqrt{\frac{4}{m}\left(d\log\frac{2m}{d} + d + \log\frac{4}{\delta}\right)} + d_\mathcal{H}(\tilde{\mu}^S, \tilde{\mu}^T) + \lambda$$

여기서:
- $\epsilon_T(h)$: 타겟 도메인 검증 손실
- $\hat{\epsilon}_S(h)$: 소스 도메인 경험적 검증 손실
- $d_\mathcal{H}(\tilde{\mu}^S, \tilde{\mu}^T)$: 소스-타겟 도메인 간 변분 거리(variational distance)
- $\lambda$: 이상적 가설의 결합 오차
- $m$: 샘플 수, $d$: VC 차원

**이 bound의 의미:** 소스 도메인 검증 손실을 줄이고 $d_\mathcal{H}(\tilde{\mu}^S, \tilde{\mu}^T)$를 최소화하면 타겟 도메인 검증 성능이 보장됩니다.

---

### 2.3 제안 방법 및 수식

#### Feature Transfer Network (FTN) 구성요소

FTN은 3가지 주요 모듈로 구성됩니다:

$$\text{FTN} = \underbrace{f: \mathcal{X} \rightarrow \mathcal{Z}}_{\text{Gen (feature generator)}} + \underbrace{g: \mathcal{Z} \rightarrow \mathcal{Z}}_{\text{Tx (feature transfer)}} + \underbrace{D_1, D_2}_{\text{Domain discriminators}}$$

#### (1) 검증 손실 (Verification Objective)

소스 도메인 쌍에 대한 검증 손실:

$$\mathcal{L}_{\text{vrf}}(f) = \mathbb{E}_{(x_1,x_2)\in\mathcal{X}_S\times\mathcal{X}_S}\left[y_{12}\log h_f(f_1,f_2) + (1-y_{12})\log(1-h_f(f_1,f_2))\right] $$

$$\mathcal{L}_{\text{vrf}}(g) = \mathbb{E}_{(x_1,x_2)\in\mathcal{X}_S\times\mathcal{X}_S}\left[y_{12}\log h_g(g_1,g_2) + (1-y_{12})\log(1-h_g(g_1,g_2))\right] $$

여기서 비파라메트릭 분류기: $h_f = \sigma(f_1^\top f_2)$, $h_g = \sigma(g_1^\top g_2)$, $\sigma(a) = \frac{1}{1+\exp(-a)}$

#### (2) 도메인 적대적 손실 (Domain Adversarial Objective)

$D_1$은 $f(\mathcal{X}_T)$와 $g(f(\mathcal{X}_S))$를 구별하도록 학습:

$$\mathcal{L}_{D_1} = \mathbb{E}_{x\in\mathcal{X}_S}\log D_1(g) + \mathbb{E}_{x\in\mathcal{X}_T}\log(1-D_1(f)), \quad \mathcal{L}_{\text{adv}} = \mathbb{E}_{x\in\mathcal{X}_T}\log D_1(f) $$

> **Note:** $g(f(x)) = f(x)$ (항등 매핑)이면 표준 DANN과 동일

#### (3) 도메인 분리 손실 (Domain Separation Objective)

$D_2$를 통해 소스와 타겟 표현 공간을 명시적으로 분리:

$$\mathcal{L}_{\text{sep}} = \mathbb{E}_{x\in\mathcal{X}_S}\log D_2(f) + \frac{1}{2}\left[\mathbb{E}_{x\in\mathcal{X}_S}\log(1-D_2(g)) + \mathbb{E}_{x\in\mathcal{X}_T}\log(1-D_2(f))\right] $$

#### (4) 전체 학습 목표 (Training Objectives)

$$\mathcal{L}_f = \frac{1}{2}\left[\mathcal{L}_{\text{vrf}}(g) + \mathcal{L}_{\text{vrf}}(f)\right] + \lambda_1\mathcal{L}_{\text{adv}} + \lambda_2\mathcal{L}_{\text{sep}}, \quad \mathcal{L}_g = \mathcal{L}_{\text{vrf}}(g) + \lambda_2\mathbb{E}_{\mathcal{X}_S}\log(1-D_2(g)) $$

#### (5) 특징 재구성 손실 (Feature Reconstruction Loss)

모드 붕괴(mode collapse) 방지를 위한 정규화:

$$\mathcal{L}_{\text{recon}} = -\left[\lambda_3\mathbb{E}_{x\in\mathcal{X}_S}\|f(x)-f_{\text{ref}}(x)\|_2^2 + \lambda_4\mathbb{E}_{x\in\mathcal{X}_T}\|f(x)-f_{\text{ref}}(x)\|_2^2\right] $$

#### (6) N-pair 손실 (N-pair Loss)

검증 손실을 N-pair 손실로 대체하여 더 강력한 메트릭 학습:

$$\mathcal{L}_N(f) = \mathbb{E}_{\{x_n, x_n^+\}_{n=1}^N, x_n, x_n^+\in\mathcal{X}_S}\left[\sum_{n=1}^N\log p_n(f)\right], \quad p_n(f) = \frac{\exp(f(x_n)^\top f(x_n^+))}{\sum_{k=1}^N\exp(f(x_n)^\top f(x_k^+))} $$

N-pair 손실을 적용한 최종 FTN 학습 목표:

$$\mathcal{L}_f = \frac{1}{2}\left[\mathcal{L}_N(g) + \mathcal{L}_N(f)\right] + \lambda_1\mathcal{L}_{\text{adv}} + \lambda_2\mathcal{L}_{\text{sep}} + \mathcal{L}_{\text{recon}}, \quad \mathcal{L}_g = \mathcal{L}_N(g) + \lambda_2\mathbb{E}_{\mathcal{X}_S}\log(1-D_2(g)) $$

---

### 2.4 MCEM (Multi-Class Entropy Minimization)

비지도 타겟 도메인에서의 엔트로피 최소화:

**기본 검증 엔트로피 최소화:**

$$\mathcal{L}_{\text{vrf}}^{\text{ent}}(f) = \mathbb{E}_{x_i, x_j\in\mathcal{X}_T}\left[p_{ij}\log p_{ij} + (1-p_{ij})\log(1-p_{ij})\right] $$

**MCEM (HDBSCAN 클러스터링 + N-pair 구조):**

```math
\mathcal{L}_N^{\text{ent}}(f) = \mathbb{E}_{\{x_n,x_n^+\}_{n=1}^N, x_n, x_n^+\in\mathcal{X}_T}\left[\sum_{n=1}^N\left\{\sum_{m=1}^N p_{nm}\log p_{nm}\right\}\right], \quad p_{nm}(f) = \frac{\exp(f_n^\top f_m^+)}{\sum_{k=1}^N\exp(f_n^\top f_k^+)}
```

HDBSCAN 클러스터링으로 의사 레이블(pseudo-label)을 생성한 후, 같은 클러스터 내 예시를 positive pair로 사용합니다.

---

### 2.5 모델 구조 (Model Architecture)

```
입력 이미지 (xs, xt)
     │
     ▼
┌──────────────────────────────────────────┐
│  Gen (f): Feature Generation Module     │
│  - 소스/타겟 → 구별 가능한 표현 공간 매핑  │
│  - 38-layer ResNet (얼굴 실험)           │
│  - CNN + FC (digit 실험)                │
└────────────┬─────────────┬──────────────┘
             │             │
          fs (소스)      ft (타겟)
             │
             ▼
┌──────────────────────────┐
│  Tx (g): Feature Transfer│
│  - MLP + Residual        │
│  - fs → g(fs)            │
└───────────┬──────────────┘
            │ g(fs)
            ▼
    ┌───────────────┐     ┌───────────────┐
    │  D1 (Ladv)    │     │  D2 (Lsep)   │
    │ g(fs) vs ft   │     │ fs vs ft, g(fs)│
    └───────────────┘     └───────────────┘
    
검증 손실: Lvrf 또는 LN을 f(xs) 쌍과 g(f(xs)) 쌍에 적용
```

---

### 2.6 성능 향상

#### CEF 데이터셋 (Cross-Ethnicity Faces) 결과

| 모델 | AA 검증(%) | EA 검증(%) | AA 식별(%) | EA 식별(%) |
|------|-----------|-----------|-----------|-----------|
| $\text{Sup}^C$ (하한) | 92.24 | 93.41 | 69.64 | 76.37 |
| DANN | 95.37 | 96.36 | 74.88 | 79.39 |
| **FTN** | **95.62** | **96.64** | **75.35** | **80.69** |
| DANN+MCEM | 96.36 | 97.34 | 80.30 | 83.07 |
| **FTN+MCEM** | **96.76** | **97.40** | **80.75** | **83.71** |
| $\text{Sup}^{C,A,E}$ (상한) | 97.16 | 97.05 | 84.02 | 84.38 |

#### DANN 대비 FTN의 핵심 이점

DANN은 도메인 정렬 과정에서 서로 다른 민족의 identity를 혼동:

| 모델 | CAU vs. AA, EA (Cross-domain) |
|------|-------------------------------|
| $\text{Sup}^C$ | 91.67 |
| DANN | 89.91 (기준보다 **낮음**) |
| **FTN** | **92.29** |

**DANN은 cross-domain 식별 정확도가 오히려 저하**되는 문제를 FTN이 해결합니다.

---

### 2.7 한계점

1. **이론적 한계:** 이론적 일반화 경계는 **within-domain verification만 보장**하며, cross-domain verification에는 이론적 보장이 없음
2. **클러스터링 의존성:** MCEM의 성능이 HDBSCAN 클러스터링 품질에 의존 (클러스터 품질이 낮으면 성능 저하)
3. **하이퍼파라미터 민감성:** $\lambda_1, \lambda_2, \lambda_3, \lambda_4$ 등 다수의 하이퍼파라미터 조정 필요
4. **도메인 이진 가정:** Cross-domain 검증 시 소스와 타겟 도메인의 샘플은 **무조건 다른 클래스**라는 강한 가정 필요
5. **확장성:** 단일 소스 → 단일 타겟 구조로, 다중 소스/타겟 시나리오에 대한 확장이 직접적이지 않음
6. **사설 데이터셋:** CEF 데이터셋이 공개되지 않아 재현이 어려움

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 일반화 경계 분석

논문은 검증 태스크로의 변환을 통해 다음 일반화 경계를 제공합니다:

$$\epsilon_T(h) \leq \hat{\epsilon}_S(h) + \underbrace{\sqrt{\frac{4}{m}\left(d\log\frac{2m}{d} + d + \log\frac{4}{\delta}\right)}}_{\text{샘플 복잡도 항}} + \underbrace{d_\mathcal{H}(\tilde{\mu}^S, \tilde{\mu}^T)}_{\text{도메인 거리}} + \lambda$$

**일반화 성능을 높이기 위한 3가지 전략:**

#### 전략 1: 도메인 거리 $d_\mathcal{H}(\tilde{\mu}^S, \tilde{\mu}^T)$ 감소
- $\mathcal{L}_{\text{adv}}$를 통해 $g(f(\mathcal{X}_S))$와 $f(\mathcal{X}_T)$ 분포를 정렬
- 단, 완전 정렬 시 cross-domain 판별력 손실 → $\mathcal{L}_{\text{sep}}$으로 균형 유지

#### 전략 2: 소스 도메인 검증 손실 $\hat{\epsilon}_S(h)$ 최소화
- N-pair 손실로 소스 도메인에서 강력한 판별 메트릭 학습
- 이 판별력이 도메인 적응을 통해 타겟으로 전이

#### 전략 3: 타겟 도메인 엔트로피 최소화 (MCEM)
- 비레이블 타겟 데이터의 클러스터 구조를 활용
- 타겟 도메인 자체의 판별력을 직접 강화

### 3.2 일반화 성능 향상의 실증적 증거

**HDBSCAN 클러스터링 품질 (F-score):**

| 방법 | AA F-score | EA F-score |
|------|-----------|-----------|
| Source example 기반 | 0.22 | 0.25 |
| Source center 기반 | 0.58 | 0.16 |
| HDBSCAN ($\text{Sup}^C$) | 93.99 | 89.95 |
| **HDBSCAN (FTN)** | **96.31** | **96.34** |
| HDBSCAN (FTN+MCEM) | 96.79 | 96.49 |

FTN이 학습한 특징 공간이 타겟 도메인 내 클러스터 구조를 더 잘 반영함을 보여주며, 이는 **반복적 레이블링(iterative labeling)**으로 연결될 수 있는 잠재력입니다.

### 3.3 일반화를 저해하는 요인

1. **특징 재구성 손실과 도메인 적응 손실의 트레이드오프:** $\lambda_3, \lambda_4$가 너무 크면 적응이 이루어지지 않고, 너무 작으면 모드 붕괴 발생
2. **도메인 분리 가정의 강도:** 소스와 타겟이 매우 다른 경우 분리가 쉽지만, 유사한 경우 $\mathcal{L}_{\text{sep}}$이 과도하게 작용할 수 있음

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

#### (1) 비겹침 레이블 도메인 적응의 이론적 토대 마련
검증 태스크로의 재정의는 이후 **Open-Set Domain Adaptation**, **Zero-Shot Domain Adaptation** 연구의 이론적 기반을 제공합니다.

#### (2) 메트릭 러닝 + 도메인 적응의 융합 연구 촉진
FTN 이후, 다음과 같은 연구 방향이 활성화되었습니다:
- Contrastive Learning 기반 도메인 적응
- Self-supervised 표현 학습과 도메인 적응의 결합

#### (3) AI 공정성(Fairness) 연구의 기술적 기반
민족 간 얼굴 인식 편향 해소는 **AI Fairness** 연구의 기술적 해결책으로 제시되어, 이후 더 광범위한 인구통계학적 편향 해소 연구에 영감을 줍니다.

#### (4) 자동 레이블링 파이프라인
FTN + HDBSCAN의 반복적 사용은 **반자동 데이터 레이블링** 시스템으로 발전 가능성을 시사합니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의:** 아래의 연구들은 제가 학습한 지식을 기반으로 작성하였으며, 본 논문의 PDF 외 외부 출처를 직접 검색하거나 확인할 수 없는 상황입니다. 논문명과 주요 내용은 학습 데이터 기반이므로, 정확한 내용은 반드시 원문을 통해 확인하시기 바랍니다.

#### 비교 분석 표

| 연구 방향 | 본 논문 (FTN, ICLR 2019) | 2020년 이후 관련 연구 |
|----------|--------------------------|----------------------|
| **핵심 문제** | 비겹침 레이블 공간 UDA | 동일 문제 + 더 넓은 도메인 격차 |
| **도메인 정렬** | 도메인 적대적 학습 | Optimal Transport, Contrastive DA |
| **타겟 레이블 활용** | HDBSCAN 클러스터링 | Self-supervised learning, MoCo, SimCLR 기반 |
| **이론적 보장** | 검증 기반 일반화 경계 | PAC-Bayes, Information-theoretic bounds |
| **응용 분야** | 얼굴 인식 | 의료 영상, 자율주행, NLP |

#### 주요 연구 트렌드 (2020년 이후)

**① Contrastive Learning 기반 도메인 적응**

FTN의 N-pair 손실과 유사한 개념이 SimCLR, MoCo 등의 자기지도 대조 학습(Self-supervised Contrastive Learning)과 결합되어, 타겟 도메인에서도 더 강력한 표현을 학습하는 방향으로 발전했습니다. 이는 MCEM의 확장으로 볼 수 있습니다.

$$\mathcal{L}_{\text{contrastive}} = -\log\frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N}\mathbf{1}_{[k\neq i]}\exp(\text{sim}(z_i, z_k)/\tau)}$$

**② Open-Set / Universal Domain Adaptation**

FTN의 비겹침 레이블 공간 설정은 이후 Universal Domain Adaptation (UniDA) 연구로 이어집니다. 소스에만 있는 클래스, 타겟에만 있는 클래스, 공유 클래스를 모두 처리하는 더 일반적인 설정입니다.

**③ Prompt-based / Foundation Model 기반 DA**

2022년 이후 CLIP 등 대형 멀티모달 모델을 활용한 도메인 적응 연구가 등장했습니다. 소수의 소스 레이블만으로도 강력한 적응이 가능하며, FTN의 전이 학습 철학과 유사합니다.

**④ Fairness-aware Domain Adaptation**

FTN의 민족 편향 해소 응용은 이후 더 광범위한 **공정성 인식 도메인 적응(Fairness-aware DA)** 연구로 확장되었습니다.

---

### 4.3 향후 연구 시 고려해야 할 점

#### (1) 이론적 측면
- **Cross-domain verification의 이론적 보장 부재** 문제를 해결해야 합니다. 현재 이론은 within-domain만 커버
- 더 tight한 일반화 경계 도출 필요 (현재 경계는 다소 느슨함)

#### (2) 방법론적 측면
- **클러스터링 품질 의존성 탈피:** HDBSCAN 대신 학습 가능한 클러스터링 모듈 도입 고려
- **반복 학습(Iterative Training):** FTN → 클러스터링 → MCEM → FTN 반복을 공식화
- **다중 타겟 도메인:** 단일 타겟이 아닌 여러 타겟 도메인 동시 적응

#### (3) 공정성 및 사회적 측면
- 민족 외 성별, 연령, 조명 등 다양한 속성에 대한 편향 해소로 확장
- 데이터 수집 및 레이블링 과정의 편향 자체를 고려

#### (4) 실용적 측면
- **하이퍼파라미터 자동화:** $\lambda_1, \lambda_2, \lambda_3, \lambda_4$ 자동 조정 메커니즘
- **계산 효율:** 대형 모델 기반 도메인 적응 시 효율적인 파인튜닝 전략 필요
- **도메인 레이블 가용성:** 실제로는 어떤 샘플이 어느 도메인인지 모를 수 있음

---

## 참고 자료

**직접 참조한 원문:**
- **Sohn, K., Shang, W., Yu, X., & Chandraker, M. (2019). "Unsupervised Domain Adaptation for Distance Metric Learning." *ICLR 2019.*** (본 답변의 주 참조 문서)

**원문에서 인용된 주요 참고문헌:**
- Ben-David, S., Blitzer, J., Crammer, K., & Pereira, F. (2007). "Analysis of representations for domain adaptation." *NIPS 2007.*
- Ganin, Y., et al. (2016). "Domain-adversarial training of neural networks." *JMLR, 17(59):1–35.*
- Sohn, K. (2016). "Improved deep metric learning with multi-class N-pair loss objective." *NIPS 2016.*
- Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). "Density-based clustering based on hierarchical density estimates." *PAKDD 2013.*
- Long, M., Zhu, H., Wang, J., & Jordan, M. I. (2016). "Unsupervised domain adaptation with residual transfer networks." *NIPS 2016.*
- Grandvalet, Y., & Bengio, Y. (2005). "Semi-supervised learning by entropy minimization." *NIPS 2005.*
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep residual learning for image recognition." *CVPR 2016.*
- Buolamwini, J., & Gebru, T. (2018). "Gender shades: Intersectional accuracy disparities in commercial gender classification." *FAT* 2018.*

> **면책 고지:** 2020년 이후 최신 연구 비교 분석 부분은 제 학습 데이터 기반으로 작성되었으며, 직접적인 원문 접근이 불가능하므로 세부 내용에 오류가 있을 수 있습니다. 반드시 원문을 통해 검증하시기 바랍니다.
