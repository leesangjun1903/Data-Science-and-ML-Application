# Domain-Specific Batch Normalization for Unsupervised Domain Adaptation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문의 핵심 주장은 다음과 같습니다: **소스 도메인과 타겟 도메인이 배치 정규화(Batch Normalization, BN) 통계량을 공유하는 것은 도메인 시프트(domain shift)가 존재할 때 부적절하며**, 각 도메인에 전용 BN 레이어(DSBN)를 할당함으로써 도메인 불변(domain-invariant) 표현 학습을 더 효과적으로 수행할 수 있다는 것입니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **DSBN 모듈 제안** | 도메인별 BN 브랜치를 통해 도메인 특화 정보를 분리 |
| **2단계 학습 프레임워크** | 슈도 레이블 생성 → 멀티태스크 분류 학습 |
| **다중 소스 도메인 확장** | DSBN의 브랜치 수를 늘려 자연스럽게 확장 |
| **State-of-the-art 달성** | Office-31, VisDA-C 벤치마크에서 당시 최고 성능 달성 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

**도메인 시프트(Domain Shift) 문제**: 소스 도메인($\mathcal{X}_S$, 레이블 있음)과 타겟 도메인($\mathcal{X}_T$, 레이블 없음)의 데이터 분포 차이로 인해 소스 도메인에서 학습한 모델이 타겟 도메인에서 성능이 저하되는 문제.

기존 방법들의 한계:
- 소스·타겟 도메인이 **전체 네트워크(BN 포함)를 공유**하므로, 도메인 특화 정보가 BN 통계량에 혼재
- BN의 평균($\mu$)과 분산($\sigma^2$)이 두 도메인의 혼합 통계량을 반영하여 각 도메인의 특성을 제대로 포착 불가

### 2-2. 제안 방법 (수식 포함)

#### (A) 기존 Batch Normalization (BN)

BN 레이어는 미니배치 내 활성화를 정규화합니다:

$$\text{BN}(\mathbf{x}[i,j,n]; \gamma, \beta) = \gamma \cdot \hat{\mathbf{x}}[i,j,n] + \beta $$

$$\hat{\mathbf{x}}[i,j,n] = \frac{\mathbf{x}[i,j,n] - \mu}{\sqrt{\sigma^2 + \epsilon}} $$

$$\mu = \frac{\sum_n \sum_{i,j} \mathbf{x}[i,j,n]}{N \cdot H \cdot W} $$

$$\sigma^2 = \frac{\sum_n \sum_{i,j} (\mathbf{x}[i,j,n] - \mu)^2}{N \cdot H \cdot W} $$

학습 중 지수 이동 평균으로 전체 통계량을 추정:

$$\bar{\mu}^{t+1} = (1-\alpha)\bar{\mu}^t + \alpha\mu^t $$

$$\left(\bar{\sigma}^{t+1}\right)^2 = (1-\alpha)\left(\bar{\sigma}^t\right)^2 + \alpha\left(\sigma^t\right)^2 $$

#### (B) Domain-Specific Batch Normalization (DSBN)

DSBN은 도메인 레이블 $d \in \{S, T\}$에 따라 도메인별 어파인 파라미터 $\gamma_d$, $\beta_d$를 할당합니다:

$$\text{DSBN}_d(\mathbf{x}_d[i,j,n]; \gamma_d, \beta_d) = \gamma_d \cdot \hat{\mathbf{x}}_d[i,j,n] + \beta_d $$

$$\hat{\mathbf{x}}_d[i,j,n] = \frac{\mathbf{x}_d[i,j,n] - \mu_d}{\sqrt{\sigma_d^2 + \epsilon}} $$

$$\mu_d = \frac{\sum_n \sum_{i,j} \mathbf{x}_d[i,j,n]}{N \cdot H \cdot W} $$

$$\sigma_d^2 = \frac{\sum_n \sum_{i,j} (\mathbf{x}_d[i,j,n] - \mu_d)^2}{N \cdot H \cdot W} $$

도메인별 지수 이동 평균:

$$\bar{\mu}_d^{t+1} = (1-\alpha)\bar{\mu}_d^t + \alpha\mu_d^t $$

$$\left(\bar{\sigma}_d^{t+1}\right)^2 = (1-\alpha)\left(\bar{\sigma}_d^t\right)^2 + \alpha\left(\sigma_d^t\right)^2 $$

#### (C) 2단계 학습 프레임워크

**Stage 1: 초기 슈도 레이블 생성**

MSTN의 전체 손실 함수:

$$\mathcal{L} = \mathcal{L}_{\text{cls}}(\mathcal{X}_S) + \lambda \mathcal{L}_{\text{da}}(\mathcal{X}_S, \mathcal{X}_T) + \lambda \mathcal{L}_{\text{sm}}(\mathcal{X}_S, \mathcal{X}_T) $$

CPUA의 클래스 가중치:

$$w_S(x, y) = \frac{\max_{y'} p_S(y')}{p_S(y)} $$

$$w_T(x) = \frac{\max_{y'} \tilde{p}_T(y')}{\tilde{p}_T(\tilde{y}(x))} $$

**Stage 2: 슈도 레이블 기반 자기 학습**

최종 손실 함수:

$$\mathcal{L} = \mathcal{L}_{\text{cls}}(\mathcal{X}_S) + \mathcal{L}_{\text{cls}}^{\text{pseudo}}(\mathcal{X}_T) $$

$$\mathcal{L}_{\text{cls}}(\mathcal{X}_S) = \sum_{(x,y) \in \mathcal{X}_S} \ell(F_S^2(x), y) $$

$$\mathcal{L}_{\text{cls}}^{\text{pseudo}}(\mathcal{X}_T) = \sum_{x \in \mathcal{X}_T} \ell(F_T^2(x), y') $$

슈도 레이블 점진적 정제:

```math
y' = \underset{c \in \mathcal{C}}{\arg\max} \left\{ (1-\lambda) F_T^1(x)[c] + \lambda F_T^2(x)[c] \right\}
```

$$\lambda = \frac{2}{1 + \exp(-\gamma \cdot p)} - 1, \quad \gamma = 10$$

**다중 소스 도메인 확장:**

$$\mathcal{L} = \frac{1}{|\mathcal{D}_S|} \sum_{i}^{|\mathcal{D}_S|} \left( \mathcal{L}_{\text{cls}}(\mathcal{X}_{S_i}) + \mathcal{L}_{\text{align}}(\mathcal{X}_{S_i}, \mathcal{X}_T) \right) $$

### 2-3. 모델 구조

```
[입력 이미지 (소스/타겟)]
        ↓
[공유 Convolutional Layer]
        ↓
   ┌────┴────┐
[BN(S)]   [BN(T)]   ← DSBN (도메인별 분리)
   └────┬────┘
        ↓
[공유 Fully Connected Layer]
        ↓
  [도메인별 분류기]
   FS(·)   FT(·)
```

- **공유 파라미터**: BN을 제외한 모든 CNN 가중치 (도메인 불변 표현 학습)
- **분리 파라미터**: 각 도메인의 $\gamma_d$, $\beta_d$, $\mu_d$, $\sigma_d^2$ (도메인 특화 정보 포착)
- **백본**: VisDA-C → ResNet-101, Office-31/Office-Home → ResNet-50

### 2-4. 성능 향상

| 데이터셋 | 기준 모델 | DSBN 적용 전 | DSBN Stage 1 | DSBN Stage 1+2 |
|----------|-----------|-------------|--------------|----------------|
| VisDA-C | MSTN | 65.0% | 72.3% | **80.2%** |
| VisDA-C | CPUA | 66.6% | 71.9% | 76.2% |
| Office-31 | MSTN | 86.5% | 87.8% | **88.3%** |
| Office-31 | CPUA | 86.4% | 87.5% | **88.3%** |

특히 어려운 클래스(knife, person, skate, truck)에서 성능 향상이 두드러짐.

### 2-5. 한계

1. **레이블 노이즈 누적**: 슈도 레이블의 오류가 반복 학습 시 축적될 가능성
2. **도메인 레이블 요구**: 테스트/학습 시 도메인 정보가 사전에 주어져야 함 (완전한 미지 도메인에 대한 즉각 적용 어려움)
3. **배치 크기 민감성**: DSBN은 도메인별 미니배치를 별도 구성해야 하므로, 작은 배치에서 통계량이 불안정할 수 있음
4. **단일 타겟 도메인 가정**: 여러 타겟 도메인에 동시 적응하는 시나리오에 대한 검토 미흡
5. **복잡도 증가**: 도메인 수 증가 시 BN 파라미터가 선형적으로 증가

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. DSBN이 일반화에 기여하는 메커니즘

**핵심 원리**: 도메인 특화 정보를 BN에 격리시킴으로써, 나머지 네트워크 파라미터가 **순수하게 도메인 불변(class-discriminative) 특징**에 집중하도록 유도합니다.

$$\underbrace{F_d(x)}_{\text{도메인 특화 네트워크}} = \underbrace{W}_{\text{공유 가중치 (도메인 불변)}} \circ \underbrace{\text{DSBN}_d}_{\text{도메인 특화 정규화}}$$

이는 다음과 같은 귀납적 편향(inductive bias)을 제공합니다:

- **도메인 불변 표현**: BN 이외의 모든 파라미터가 두 도메인에서 공유되므로, 도메인 간 공통 특성(클래스 관련 특징)을 더 잘 포착
- **도메인 특화 정규화**: 각 도메인의 실제 통계적 특성($\mu_d$, $\sigma_d^2$)에 맞게 활성화를 정규화하여, 도메인 시프트로 인한 내부 공변량 시프트(internal covariate shift)를 도메인별로 제거

### 3-2. t-SNE 분석을 통한 일반화 검증

논문의 Figure 4에서 BN 대비 DSBN을 사용했을 때, 같은 클래스의 소스·타겟 도메인 샘플들이 특징 공간에서 더 가깝게 정렬됨을 시각적으로 확인. 이는 DSBN이 클래스 판별 특징을 더 효과적으로 학습함을 의미.

### 3-3. 반복 학습(Iterative Learning)을 통한 일반화 강화

| 단계 | Stage 1 | Stage 2 Iter 1 | Iter 2 | Iter 3 | Iter 4 |
|------|---------|----------------|--------|--------|--------|
| VisDA-C 정확도(%) | 72.3 | 80.2 | 81.4 | 82.2 | **82.7** |

슈도 레이블의 품질이 반복 학습을 통해 개선되면서 타겟 도메인에 대한 일반화 성능이 지속적으로 향상됨.

### 3-4. 다중 소스 도메인에서의 일반화

Office-31의 `W, D → A` 태스크에서:
- BN (Separate): 69.9%
- **DSBN (Separate): 75.6%** (+5.7%p)

다수의 소스 도메인을 효과적으로 활용할 때 DSBN의 일반화 이점이 더욱 두드러짐.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4-1. 연구에 미치는 영향

**① 정규화 레이어의 도메인 적응 핵심 역할 부각**

DSBN은 BN의 어파인 파라미터와 통계량이 도메인 정보를 암묵적으로 인코딩한다는 사실을 실험적으로 입증하여, 이후 연구에서 정규화 레이어를 도메인 적응의 핵심 구성 요소로 다루는 방향을 제시했습니다.

**② 플러그인(Plug-in) 방식의 모듈화 설계**

기존 DA 네트워크에 BN만 DSBN으로 교체하면 적용 가능한 범용적 설계는, 이후 연구에서도 모듈식 적응 구성 요소를 설계하는 방법론적 전례가 되었습니다.

**③ 슈도 레이블 + 자기 학습의 결합**

점진적 슈도 레이블 정제(Eq. 23)와 반복 자기 학습의 결합은 이후 **Self-Training 기반 DA 연구**에 영향을 미쳤습니다.

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

#### (A) Transferable Normalization (TransNorm, 2019/2020)

**Wang et al., "Transferable Normalization: Towards Improving Transferability of Deep Neural Networks," NeurIPS 2019**

- DSBN과 유사하게 도메인별 정규화를 활용하지만, 채널별로 **전이 가능한 정도(transferability)**를 동적으로 가중하여 적응
- DSBN이 모든 채널에 동일한 도메인 분리를 적용하는 반면, TransNorm은 채널 선택적으로 적용

$$\text{TransNorm}: \hat{x} = \frac{x - \mu_{\text{mix}}}{\sigma_{\text{mix}}}, \quad \mu_{\text{mix}} = w\mu_S + (1-w)\mu_T$$

#### (B) Domain Adaptation with Source Data-free (SHOT, 2020)

**Liang et al., "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation," ICML 2020**

- 소스 데이터에 접근 없이 타겟 도메인만으로 적응 (Source-free DA)
- DSBN의 가정(소스 도메인 데이터 접근 가능)과 달리, **개인정보 보호** 관점에서 현실적 제약을 해결하려는 방향
- **DSBN의 한계**: 소스 데이터를 학습 시 항상 필요로 한다는 점에서 Source-free 시나리오에 직접 적용 불가

#### (C) Domain Generalization with Normalization (2021~)

**Fan et al., "Adversarially Adaptive Normalization for Single Domain Generalization," CVPR 2021**

- 단일 소스로 여러 미지 도메인에 일반화
- DSBN의 아이디어를 도메인 일반화(Domain Generalization)로 확장하는 시도
- BN 통계량 교란(perturbation)을 통해 도메인 다양성을 증강(augment)

#### (D) Test-Time Adaptation (TTA, 2020~)

**Wang et al., "Tent: Fully Test-Time Adaptation by Entropy Minimization," ICLR 2021**

- 테스트 시점에 BN 통계량을 타겟 도메인 배치로 동적 업데이트 (AdaBN의 발전형)
- DSBN이 학습 시 도메인 레이블을 요구하는 것과 달리, **레이블 없이 테스트 시점 적응** 가능
- 엔트로피 최소화를 통해 BN의 어파인 파라미터 업데이트:

$$\mathcal{L}_{\text{Tent}} = -\sum_c \hat{p}_c \log \hat{p}_c$$

#### (E) Normalization을 넘어선 Transformer 기반 DA (2021~)

**Xu et al., "CDTrans: Cross-Domain Transformer for Unsupervised Domain Adaptation," ICLR 2022**

- Transformer 아키텍처는 BN 대신 Layer Normalization(LN)을 사용하므로, DSBN 기반 방법론을 직접 적용하기 어려움
- **향후 과제**: Vision Transformer에서의 도메인 특화 LN 설계 필요

#### 비교 요약표

| 방법 | 소스 데이터 필요 | 도메인 레이블 필요 | 정규화 분리 | 확장성 |
|------|----------------|------------------|------------|--------|
| DSBN (본 논문) | ✅ | ✅ | ✅ | 다중 소스 지원 |
| AdaBN | ✅ | ✅ | 부분적 (통계량만) | 제한적 |
| TransNorm | ✅ | ✅ | 채널 선택적 | 제한적 |
| SHOT | ❌ | ❌ | ❌ | Source-free |
| Tent (TTA) | ❌ | ❌ | 테스트 시점 | 온라인 적응 |

### 4-3. 향후 연구 시 고려 사항

**① Source-Free 시나리오 확장**

- DSBN은 소스 데이터 접근을 전제하므로, 개인정보/저작권 문제가 있는 실제 응용에서는 제약
- 소스 모델의 BN 통계량만 저장하여 타겟 적응에 활용하는 방향 연구 필요

**② Vision Transformer (ViT) 적용**

- ViT는 Layer Normalization을 사용하므로, **도메인 특화 Layer Normalization (DSLN)** 설계가 필요
- 어텐션 메커니즘에서의 도메인 특화 요소 탐구 필요

**③ 슈도 레이블 노이즈 처리 개선**

- 현재의 점진적 가중($\lambda$ 스케줄링)은 단순한 휴리스틱
- 신뢰도 기반 샘플 선택(confidence-based filtering), Mixup, 대조 학습(contrastive learning) 결합으로 노이즈 레이블 내성 강화 가능

**④ 온라인/연속 도메인 적응**

- 현재 프레임워크는 오프라인(offline) 학습 기반
- 실시간으로 변하는 도메인에 DSBN을 적응시키는 Online DSBN 연구 필요

**⑤ 배치 크기 독립적 정규화**

- 소규모 배치에서의 통계량 불안정 문제 해결을 위한 Group Normalization 또는 Instance Normalization과의 결합 가능성 탐구

**⑥ 이론적 분석 보강**

- DSBN이 도메인 간 분포 차이를 어느 정도 줄이는지에 대한 이론적 상한(generalization bound) 분석 필요
- 도메인 불변 표현의 품질을 측정하는 정보 이론적 지표(예: 상호 정보량) 활용

---

## 참고 자료

1. **Chang, W.-G., You, T., Seo, S., Kwak, S., & Han, B. (2019).** "Domain-Specific Batch Normalization for Unsupervised Domain Adaptation." *arXiv:1906.03950* [cs.LG]. *(본 논문)*

2. **Wang, X., Jin, Y., Long, M., Wang, J., & Jordan, M. I. (2019).** "Transferable Normalization: Towards Improving Transferability of Deep Neural Networks." *NeurIPS 2019.*

3. **Liang, J., Hu, D., & Feng, J. (2020).** "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation." *ICML 2020.*

4. **Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021).** "Tent: Fully Test-Time Adaptation by Entropy Minimization." *ICLR 2021.*

5. **Fan, X., Wang, Q., Ke, J., Yang, F., Gong, B., & Zhou, M. (2021).** "Adversarially Adaptive Normalization for Single Domain Generalization." *CVPR 2021.*

6. **Xu, T., Chen, W., Wang, P., Wang, F., Li, H., & Jin, R. (2022).** "CDTrans: Cross-Domain Transformer for Unsupervised Domain Adaptation." *ICLR 2022.*

7. **Ioffe, S., & Szegedy, C. (2015).** "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *ICML 2015.* *(BN 원논문)*

8. **Li, Y., Wang, N., Shi, J., Hou, X., & Liu, J. (2018).** "Adaptive Batch Normalization for practical domain adaptation." *Pattern Recognition, 80:109–117.* *(AdaBN)*

> **⚠️ 정확도 관련 고지**: 2020년 이후 비교 연구 부분(TransNorm, SHOT, Tent, CDTrans 등)은 논문 원문을 직접 참조하였으나, 각 방법의 세부 수식 및 성능 수치는 해당 논문 원문을 재확인하시기를 권장합니다. 본 답변에서 인용한 비교 분석 내용은 각 논문의 공개된 arXiv 버전 및 학회 발표 자료를 기반으로 작성하였습니다.
