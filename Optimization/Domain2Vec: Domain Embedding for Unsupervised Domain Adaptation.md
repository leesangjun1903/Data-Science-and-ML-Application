# Domain2Vec: Domain Embedding for Unsupervised Domain Adaptation 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Domain2Vec은 **도메인을 벡터 공간의 원소로 표현(embedding)** 하여, 다수의 도메인 간 자연스러운 거리(domain distance)를 측정하고 이를 비지도 도메인 적응(UDA)에 활용할 수 있다고 주장합니다.

핵심 명제는 다음과 같습니다:

> 도메인 임베딩 공간에서의 거리가 작을수록, 해당 소스 도메인으로 학습한 모델의 타겟 도메인 성능이 높다 (음의 상관관계).

이를 실험적으로 검증한 Pearson 상관계수(PCC)는 **-0.774**로 강한 음의 상관관계를 보였습니다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **(i) 모델 제안** | 특징 분리(feature disentanglement) + Gram Matrix의 결합 학습을 통한 Deep Domain Embedding 모델 Domain2Vec 제안 |
| **(ii) 벤치마크 구축** | TinyDA (54도메인, ~100만 장, MNIST 스타일) 및 DomainBank (56도메인, 339,772장, 다양한 CV 데이터셋) 구축 |
| **(iii) 광범위한 실험** | 멀티소스 DA, 오픈셋 DA, 파셜 DA 등 다양한 시나리오에서 SOTA 대비 성능 향상 입증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 UDA의 두 가지 핵심 한계:

1. **제한된 도메인 수**: 대부분의 기존 UDA는 1개 소스 → 1개 타겟의 단순 설정만 고려
2. **도메인 간 관계 무시**: 소스-타겟 간 자연스러운 거리나 관계를 고려하지 않고 단순히 분포를 정렬

현실에서는 **수십~수백 개의 도메인**이 존재하며, 새로운 타겟 도메인이 등장할 때 **어떤 소스 도메인을 선택**할지가 중요한 문제입니다.

---

### 2.2 제안하는 방법 (수식 포함)

Domain2Vec은 두 구성요소로 이루어집니다:

#### (A) 특징 분리 (Feature Disentanglement)

**목표:** 잠재 표현 $f_G = G(x)$를 도메인 특화 특징 $f_{ds}$와 카테고리 특화 특징 $f_{cs}$로 분리

**카테고리 분리 (Category Disentanglement):**

$$\mathcal{L}_{ce}^{class} = -\sum_{i=1}^{N} \mathbb{E}_{(x,y_c)\sim\hat{\mathcal{D}}_i} \sum_{k=1}^{K} \mathbf{1}[k=y_c]\log(C(f_{cs})) \tag{1}$$

$$\mathcal{L}_{ent}^{class} = -\sum_{i=1}^{N} \frac{1}{n_i} \sum_{j=1}^{n_i} \log DC(f_{cs}) \tag{2}$$

> 식 (1)은 카테고리 분류기 $C$가 $f_{cs}$로 클래스를 정확히 예측하도록 학습, 식 (2)는 도메인 분류기 $DC$를 속이도록 적대 학습하여 $f_{cs}$에서 도메인 정보를 제거

**도메인 분리 (Domain Disentanglement):**

$$\mathcal{L}_{ce}^{domain} = -\mathbb{E}_{(x,y_d)\sim\hat{\mathcal{D}}} \sum_{k=1}^{N} \mathbf{1}[k=y_d]\log(DC(f_{ds})) \tag{3}$$

$$\mathcal{L}_{ent}^{domain} = -\sum_{i=1}^{N} \frac{1}{n_i} \sum_{j=1}^{n_i} \log C(f_{ds}) \tag{4}$$

> 식 (3)은 도메인 분류기가 $f_{ds}$로 도메인을 정확히 식별, 식 (4)는 카테고리 분류기를 속여 $f_{ds}$에서 카테고리 정보를 제거

**특징 재구성 (Feature Reconstruction):**

$$\mathcal{L}_{rec} = \|\hat{f}_G - f_G\|_F^2 + KL(q(z|f_G)\|p(z)) \tag{5}$$

> 분리 과정에서 손실되는 정보를 방지하기 위해 재구성기 $R$이 $(f_{ds}, f_{cs})$로부터 원래 $f_G$를 복원. 두 번째 항은 KL 발산으로 잠재 표현을 사전 분포 $p(z) = \mathcal{N}(0, I)$에 가깝게 유지

**상호정보 최소화 (Mutual Information Minimization):**

$$I(f_{ds}; f_{cs}) = \int_{\mathcal{P}\times\mathcal{Q}} \log \frac{d\mathbb{P}_{\mathcal{PQ}}}{d\mathbb{P}_{\mathcal{P}} \otimes \mathbb{P}_{\mathcal{Q}}} d\mathbb{P}_{\mathcal{PQ}} \tag{6}$$

MINE(Mutual Information Neural Estimator)을 활용한 Monte-Carlo 근사:

$$I(\mathcal{P}, \mathcal{Q}) = \frac{1}{n}\sum_{i=1}^{n} T(p,q,\theta) - \log\left(\frac{1}{n}\sum_{i=1}^{n} e^{T(p,q',\theta)}\right) \tag{7}$$

> $(p,q)$는 결합분포에서, $q'$는 주변분포 $\mathbb{P}_{\mathcal{Q}}$에서 샘플링. $T(p,q,\theta)$는 상호정보를 추정하는 신경망

#### (B) 깊은 도메인 임베딩 (Deep Domain Embedding)

**Gram Matrix 계산:**

$$\mathcal{G}_{ij} = \sum_k F_{ik} F_{jk} \tag{8}$$

> $F$는 특징 추출기의 은닉 합성곱 층의 벡터화된 특징맵. $\mathcal{G} \in \mathbb{R}^{n\times n}$은 스타일/텍스처 정보를 포착

**최종 임베딩:**

$$\text{Domain Embedding}(\hat{\mathcal{D}}) = \text{concat}\left(P_{\hat{\mathcal{D}}},\ \text{diag}(\overline{\mathcal{G}})\right)$$

$$P_{\hat{\mathcal{D}}} = \frac{1}{n_i}\sum_j f_{ds}^j$$

> 도메인 프로토타입 $P_{\hat{\mathcal{D}}}$ (도메인 특화 특징의 평균)와 평균 Gram Matrix의 대각 원소를 결합하여 최종 도메인 벡터 생성

#### (C) 전체 최적화 목적함수

$$\mathcal{L} = w_1 \mathcal{L}^{class} + w_2 \mathcal{L}^{domain} + w_3 \mathcal{L}_{rec} + w_4 I(f_{ds}, f_{cs}) \tag{9}$$

$$\mathcal{L}^{class} = \mathcal{L}_{ce}^{class} + \alpha \mathcal{L}_{ent}^{class}, \quad \mathcal{L}^{domain} = \mathcal{L}_{ce}^{domain} + \alpha \mathcal{L}_{ent}^{domain}$$

---

### 2.3 모델 구조

```
입력 이미지 x
    ↓
[Feature Generator G] → f_G (잠재 표현)
    ↓
[Disentangler D]
  ├─→ f_ds (도메인 특화)  ─→ [Domain Classifier DC] ←─ 적대 학습
  └─→ f_cs (카테고리 특화) ─→ [Category Classifier C] ←─ 적대 학습
    ↓
[Reconstructor R] : (f_ds, f_cs) → f̂_G
[MINE] : I(f_ds, f_cs) 최소화
    ↓
[Gram Matrix G] : 합성곱 층 활성화로부터 계산
    ↓
최종 임베딩 = concat(P_D̂, diag(G)) → PCA + t-SNE 차원 축소
```

---

### 2.4 성능 향상

| 실험 설정 | 비교 대상 최고 성능 | Domain2Vec | 향상 |
|-----------|-------------------|-----------|------|
| MSDA (TinyDA, D2V-α) | DCTN 48.1% | **48.5%** | +0.4% |
| MSDA (TinyDA, D2V-β) | DCTN 48.1% | **49.7%** | +1.6% |
| Openset DA (DomainBank) | AODA 71.3% | **73.8%** | +2.5% |
| Partial DA (DomainBank) | PADA 64.6% | **65.5%** | +0.9% |

**어블레이션 스터디 결과 (TinyDA, MSDA 평균 정확도):**

| 모델 | 평균 정확도 |
|------|------------|
| D2V (전체) | 44.9% |
| D2V w/o Gram Matrix | 43.5% (-1.4%) |
| D2V w/o Mutual Info | 44.0% (-0.9%) |

---

### 2.5 한계점

논문에서 명시적으로 언급된 한계 및 분석을 통해 도출된 한계:

1. **계산 비용**: Gram Matrix 계산은 특징 차원에 따라 메모리 비용이 $O(n^2)$으로 증가. 논문에서는 부대각, 주대각, 초대각만 사용하는 근사로 해결하지만 여전히 비용이 큼
2. **하이퍼파라미터 민감성**: $w_1, w_2, w_3, w_4, \alpha$ 등 다수의 가중치 파라미터 조정이 필요
3. **소스 도메인 레이블 필요**: 완전한 비지도 학습이 아닌, 도메인 분리를 위해 도메인 레이블이 필요
4. **확장성**: 56개 도메인의 DomainBank에서 모든 (소스, 타겟) 조합 탐색이 불가능하여 일부 조합만 실험
5. **텍스트·비전 외 모달리티**: 시각 도메인에 특화된 Gram Matrix 기반 접근법으로, 다른 모달리티로의 일반화는 검증되지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상의 핵심 메커니즘

Domain2Vec의 일반화 향상은 세 가지 메커니즘에서 비롯됩니다:

**① 도메인 거리 기반 소스 선택 (Source Selection/Weighting)**

새로운 타겟 도메인 $\hat{\mathcal{D}}_t$가 주어졌을 때, 각 소스 도메인 $\hat{\mathcal{D}}_s^{(i)}$와의 임베딩 거리를 계산하여 가중치를 부여합니다:

$$w_i \propto \exp\left(-\frac{\|\Phi(\hat{\mathcal{D}}_t) - \Phi(\hat{\mathcal{D}}_s^{(i)})\|_2^2}{\sigma^2}\right)$$

이를 통해 **관련성 없는 소스 도메인의 부정적 전이(negative transfer)** 를 방지하고, 관련성 높은 소스 도메인의 기여도를 높여 타겟에서의 일반화 성능을 향상시킵니다.

**② 특징 분리를 통한 도메인 불변 표현 학습**

$f_{cs}$에서 도메인 정보를 제거하는 적대 학습은 **도메인 불변(domain-invariant) 특징** 을 추출하여 새로운 도메인으로의 일반화를 가능하게 합니다. 상호정보 최소화 $\min I(f_{ds}; f_{cs})$는 두 특징 간의 정보 누출을 방지하여 분리의 품질을 높입니다.

**③ Gram Matrix의 스타일 표현 (다중 스케일 도메인 표현)**

Gram Matrix는 이미지의 텍스처와 스타일 정보를 **다중 스케일, 위치 독립적(stationary)** 으로 포착합니다. 이는 도메인의 본질적인 시각적 특성을 안정적으로 인코딩하여, 개별 이미지 특성에 과적합하지 않고 도메인 수준의 일반적 표현을 제공합니다.

**④ 멀티소스 도메인 적응으로의 적용**

PCC = -0.774의 강한 음의 상관관계는 임베딩 거리가 실제 전이 가능성을 잘 예측함을 보여줍니다. 이를 통해:

$$\mathcal{L}_{MSDA} = \sum_{i=1}^{K} w_i \cdot \mathcal{L}_{alignment}(\hat{\mathcal{D}}_s^{(i)}, \hat{\mathcal{D}}_t)$$

형태로 **거리 가중 멀티소스 적응** 이 가능하며, 이는 단순 멀티소스 대비 일반화 성능을 향상시킵니다 (Domain2Vec-β: 49.7% vs M³SDA: 46.8%).

### 3.2 일반화 관련 실험적 증거

- **TinyDA PCC = -0.774**: 임베딩 거리와 크로스 도메인 성능 간의 강한 음의 상관관계 → 임베딩이 일반화 가능성을 예측 가능
- **DomainBank 클러스터링**: 건물, 얼굴 등 의미적으로 유사한 도메인이 자동으로 클러스터를 형성 → 의미적 일반화 가능성 내재
- **오픈셋 DA 성능 향상**: 타겟에 알려지지 않은 카테고리가 있는 상황에서도 +2.5% 향상 → 미지의 카테고리에 대한 일반화 향상

---

## 4. 앞으로의 연구에 미치는 영향과 고려사항

### 4.1 연구에 미치는 영향

**① 도메인 표현 학습의 패러다임 전환**

기존의 UDA가 "소스와 타겟을 어떻게 정렬할 것인가"에 집중했다면, Domain2Vec은 "도메인 자체를 어떻게 표현할 것인가"라는 새로운 연구 방향을 제시합니다. 이는 **메타러닝(meta-learning), 연속학습(continual learning), 도메인 일반화(domain generalization)** 등 다양한 분야에 영향을 미칩니다.

**② 대규모 멀티도메인 벤치마크의 필요성 인식**

TinyDA (54도메인)와 DomainBank (56도메인)는 기존 벤치마크(Office-31: 3도메인, DomainNet: 6도메인) 대비 훨씬 다양한 도메인을 포함하여, **현실적 다도메인 시나리오** 연구를 촉진합니다.

**③ 소스 도메인 선택 문제의 공식화**

실용적인 AI 시스템에서 새로운 타겟 도메인이 등장할 때 어떤 기존 데이터를 활용할지의 문제를 형식화하여, **자동화된 도메인 선택(automated domain selection)** 연구의 기반을 마련합니다.

**④ 도메인 지식 그래프(Domain Knowledge Graph)**

도메인 간 관계를 그래프로 표현하는 아이디어는 **그래프 신경망(GNN) 기반 도메인 적응** 연구로 발전할 수 있는 기반을 제공합니다.

---

### 4.2 앞으로 연구 시 고려할 점

**① 임베딩 방법의 개선**

- Gram Matrix의 대각 원소만 사용하는 근사의 정보 손실 문제 → 효율적인 전체 Gram Matrix 활용 방법 연구
- Vision Transformer(ViT) 기반 특징에 대한 적용 가능성 검토 (Gram Matrix는 CNN 특화)
- 자기지도학습(self-supervised learning) 기반 특징으로의 확장

**② 도메인 레이블 의존성 제거**

현재 도메인 분리를 위해 도메인 레이블이 필요 → 완전 비지도 방식의 도메인 임베딩 연구 필요

**③ 동적 도메인 환경으로의 확장**

- 시간이 지남에 따라 도메인이 변화하는 **연속 도메인 적응(continuous domain adaptation)**
- 새로운 도메인이 점진적으로 추가되는 **점진적 학습(incremental learning)** 환경 대응

**④ 멀티모달 도메인으로의 확장**

텍스트, 오디오, 포인트 클라우드 등 비시각적 도메인에 대한 임베딩 방법 연구

**⑤ 이론적 보장의 강화**

Ben-David et al.의 $\mathcal{H}\Delta\mathcal{H}$-divergence와 같은 이론적 프레임워크와 Domain2Vec 임베딩 거리 간의 관계를 이론적으로 규명

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교 분석에서 언급하는 2020년 이후 논문들은 제가 직접 원문을 검색하여 확인한 것이 아니며, 학습 데이터 기반의 지식으로 작성된 것입니다. 일부 세부 수치나 방법론적 설명이 부정확할 수 있으므로, 반드시 원문을 직접 확인하시기 바랍니다.

### 5.1 Domain2Vec 이후의 연구 흐름

| 연구 방향 | 대표 연구 (참고용) | Domain2Vec 대비 차이점 |
|-----------|----------|----------------------|
| **Transformer 기반 도메인 적응** | CDTrans (Xu et al., 2021), TVT (Yang et al., 2023) | CNN 기반 Gram Matrix → Attention 메커니즘으로 도메인 관계 포착 |
| **소스 없는 도메인 적응 (SFDA)** | SHOT (Liang et al., ICML 2020) | 소스 데이터 없이 타겟 도메인만으로 적응; 도메인 임베딩 불필요 |
| **도메인 일반화** | DomainBed (Gulrajani & Lopez-Paz, ICLR 2021) | 단일 통합 벤치마크에서 다양한 방법 비교 평가 |
| **프롬프트 기반 DA** | DAPL (Ge et al., 2022) | CLIP 등 대형 사전학습 모델의 프롬프트 튜닝으로 도메인 적응 |

### 5.2 주요 차별점 및 Domain2Vec의 한계가 드러나는 지점

**SHOT (Liang et al., ICML 2020, "Do We Really Need to Access the Source Data?"):**
- 소스 데이터 없이 모델 가중치만으로 타겟 도메인 적응 → 도메인 레이블 및 소스 데이터 접근 자체가 불필요
- Domain2Vec은 소스 도메인 데이터와 레이블 모두 필요 → 프라이버시/접근성 측면에서 불리

**DomainBed (Gulrajani & Lopez-Paz, ICLR 2021, "In Search of Lost Domain Generalization"):**
- 엄격한 하이퍼파라미터 선택 프로토콜 하에서 많은 복잡한 방법들이 단순 ERM(Empirical Risk Minimization) 대비 개선이 미미함을 보임
- Domain2Vec의 다수 하이퍼파라미터($w_1, w_2, w_3, w_4, \alpha$)는 이러한 관점에서 위험 요소

**CDTrans (Xu et al., ICCV 2021):**
- Cross-Attention을 통해 소스-타겟 쌍 간의 관계를 직접 학습
- Gram Matrix보다 더 풍부한 도메인 간 의존성 포착 가능

### 5.3 Domain2Vec이 여전히 유효한 방향

- **다수 소스 도메인 선택 문제**: 대부분의 SFDA 방법은 어떤 소스를 선택할지의 문제를 다루지 않음
- **도메인 지식 그래프 활용**: 도메인 관계를 명시적으로 그래프로 표현하는 아이디어는 GNN 기반 DA 연구 (e.g., GVRT, 2022)로 발전
- **대규모 멀티도메인 벤치마크**: TinyDA와 DomainBank는 여전히 다도메인 연구의 중요한 테스트베드

---

## 참고자료

**주요 참고 문헌 (논문 내 인용 기준):**

- **[본 논문]** Peng, X., Li, Y., Saenko, K., "Domain2Vec: Domain Embedding for Unsupervised Domain Adaptation" (제공된 PDF)
- Belghazi et al., "Mutual Information Neural Estimation (MINE)," ICML 2018
- Gatys et al., "A Neural Algorithm of Artistic Style," arXiv:1508.06576 (2015)
- Peng et al., "Domain Agnostic Learning with Disentangled Representations," arXiv:1904.12347 (2019)
- Peng et al., "Moment Matching for Multi-source Domain Adaptation (M³SDA)," ICCV 2019
- Xu et al., "Deep Cocktail Network (DCTN)," CVPR 2018
- Ben-David et al., "A Theory of Learning from Different Domains," Machine Learning, 2010
- Achille et al., "Task2Vec: Task Embedding for Meta-Learning," ICCV 2019
- Cao et al., "Partial Adversarial Domain Adaptation (PADA)," ECCV 2018
- Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation (DANN)," ICML 2015
- Tzeng et al., "Adversarial Discriminative Domain Adaptation (ADDA)," CVPR 2017

**2020년 이후 비교 연구 (지식 기반, 원문 확인 필요):**
- Liang et al., "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation (SHOT)," ICML 2020
- Gulrajani & Lopez-Paz, "In Search of Lost Domain Generalization (DomainBed)," ICLR 2021
- Xu et al., "CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation," ICCV 2021
