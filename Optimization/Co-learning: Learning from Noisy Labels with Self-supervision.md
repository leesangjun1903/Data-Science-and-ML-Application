# Co-learning: Learning from Noisy Labels with Self-supervision 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 **노이즈 레이블(noisy labels)** 문제를 해결하기 위해 **자기지도학습(self-supervised learning)** 을 지도학습과 협력적으로 결합하는 새로운 패러다임인 **Co-learning**을 제안합니다.

> **핵심 직관**: 자기지도학습은 레이블 없이 동작하므로, 노이즈 레이블의 부정적 영향을 받지 않는다. 이를 지도학습과 협력시키면 노이즈에 강건한 표현 학습이 가능하다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **Co-learning 프레임워크** | 단일 공유 인코더 + 두 개의 독립적 헤드(분류기 헤드, 투영 헤드) 구조 제안 |
| **구조적 유사성 손실** | 두 헤드의 출력 간 관계 구조를 보존하는 새로운 손실 함수 도입 |
| **사전 지식 불필요** | 노이즈율, 데이터 분포, 깨끗한 검증 셋 없이 동작 |
| **종단간(End-to-End) 학습** | 단계적 학습 없이 하나의 통합된 파이프라인으로 학습 |
| **성능 우월성** | CIFAR-10, CIFAR-100, Animal-10N, Food-101N에서 SOTA 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

딥러닝 모델은 **노이즈 레이블에 과적합(memorization effect)** 되는 경향이 있습니다.

$$\Theta_{t+1} = \Theta_t - \eta \nabla \left( \frac{1}{|\mathcal{B}_t|} \sum_{(x, \hat{y}) \in \mathcal{B}_t} \mathcal{L}(F(x; \Theta_t), \hat{y}) \right) \tag{3}$$

위 식에서 $\hat{y}$이 노이즈 레이블이면, 경사 방향이 잘못되어 모델의 일반화 성능이 저하됩니다.

기존 방법들의 한계:
- **Decoupling, Co-teaching, JoCoR**: 두 개의 동일 구조 네트워크를 사용 → 랜덤 초기화에 의존한 제한적 다양성
- **소손실 트릭(small-loss trick)**: 노이즈율 추정 등 사전 지식 필요, 고손실 샘플의 잠재 정보 무시
- **비종단간 방법(DivideMix 등)**: 시간·공간 복잡도 증가

### 2.2 제안하는 방법 (수식 포함)

#### 전체 손실 함수

$$\mathcal{L} = \mathcal{L}_{sup} + \mathcal{L}_{int} + \mathcal{L}_{str} \tag{4}$$

---

#### (1) 지도 학습 손실 (Supervised Loss) — MixUp 적용

먼저 MixUp 증강으로 새로운 샘플을 생성합니다:

$$\bar{x}^{(i)} = \lambda \tilde{x}^{(i)} + (1-\lambda)\tilde{x}^{(m(i))} \tag{6}$$

$$\bar{y}^{(i)} = \lambda \hat{y}^{(i)} + (1-\lambda)\hat{y}^{(m(i))} \tag{7}$$

여기서 $\lambda \sim \text{Beta}(\alpha, \alpha)$이며, $m(i)$는 같은 미니배치 내에서 랜덤하게 선택된 샘플 인덱스입니다.

MixUp이 적용된 지도 학습 손실:

$$\mathcal{L}_{sup} = -\sum_{i=1}^{N} \left[ \lambda \bar{y}^{(i)} \log(\tilde{y}^{(i)}) + (1-\lambda)\bar{y}^{(i)} \log(\tilde{y}^{(m(i))}) \right] \tag{8}$$

> **MixUp의 역할**: 인코더의 수렴 속도를 늦추어 지도학습이 노이즈 레이블에 빠르게 과적합되는 것을 방지합니다.

---

#### (2) 내재적 유사성 손실 (Intrinsic Similarity Loss) — InfoNCE 기반

같은 이미지 $x^{(i)}$에서 두 가지 강한 변환 $\tilde{x}_2^{(i)}, \tilde{x}_3^{(i)}$을 통해 얻은 투영 벡터 $v_2^{(i)}, v_3^{(i)}$ 사이의 유사성을 최대화합니다.

페어별 유사도:

$$D(v_a, v_b) = \frac{v_a^T v_b}{\|v_a\| \|v_b\|} \tag{내부 정의}$$

InfoNCE 기반 대조 손실:

$$\ell(v_a^{(i)}, v_b^{(i)}) = -\log \frac{\exp(D(v_a^{(i)}, v_b^{(i)})/\tau)}{\sum_{j=1, i \neq j}^{N} \sum_{t_i, t_j \in \{2,3\}} \exp(D(v_{t_i}^{(i)}, v_{t_j}^{(j)})/\tau)} \tag{9}$$

여기서 $\tau$는 temperature 파라미터입니다.

$$\mathcal{L}_{int} = \sum_{i=1}^{N} \ell(v_2^{(i)}, v_3^{(i)}) + \ell(v_3^{(i)}, v_2^{(i)}) \tag{10}$$

---

#### (3) 구조적 유사성 손실 (Structural Similarity Loss)

분류기 헤드 $g(\cdot)$의 출력 $\tilde{y}$과 투영 헤드 $h(\cdot)$의 출력 $v$ 사이의 **쌍별 관계 구조**가 유사하도록 강제합니다.

유클리드 거리를 유사도 메트릭으로 변환:

$$p(d) = C_\sigma \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{d-\mu}{\sigma}\right)^2} \tag{11}$$

여기서 $C_\sigma = \sigma\sqrt{2\pi}$는 정규화 상수, $\mu=0$, $\sigma=0.5$로 설정합니다.

KL-발산 기반 구조적 유사성 손실:

$$\mathcal{L}_{str} = \sum_{i \neq j} p(d(v^{(i)}, v^{(j)})) \log \frac{p(d(v^{(i)}, v^{(j)}))}{p(d(\tilde{y}^{(i)}, \tilde{y}^{(j)}))} \tag{12}$$

> **구조적 유사성 손실의 역할**: 자기지도학습 헤드가 학습한 노이즈-독립적 구조를 분류기 헤드에 전달하여, 노이즈 레이블로 인한 편향을 정규화합니다.

### 2.3 모델 구조

```
입력 이미지 x
    │
    ├──[약한 증강 T']──→ x̃₁
    ├──[강한 증강 T ]──→ x̃₂  
    └──[강한 증강 T ]──→ x̃₃
         │
         ▼
    [공유 인코더 f; θ₁]
         │
    u₁, u₂, u₃ (표현 벡터)
         │
    ┌────┴────────────────┐
    │                     │
    ▼                     ▼
[분류기 헤드 g; θ₂]   [투영 헤드 h; θ₃]
    │                     │
    ỹ (예측값)           v₂, v₃ (투영값)
    │                     │
    └──── 구조적 유사성 ──┘
    
손실: L_sup(ỹ, ŷ) + L_int(v₂,v₃) + L_str(v, ỹ)
```

**기존 방법과의 차이점 비교:**

| 방법 | Agreement | Small-loss | Double Classifiers | Cross Update |
|------|-----------|------------|-------------------|--------------|
| Co-teaching | ✓ | ✓ | ✓ | ✓ |
| Co-teaching+ | ✓ | ✓ | ✓ | ✓ |
| JoCoR | ✓ | ✓ | ✓ | ✗ |
| **Co-learning** | ✓ | **✗** | **✗** | **✗** |

### 2.4 성능 향상

**CIFAR-10 결과 (%):**

| 노이즈 설정 | Standard | Co-teaching | JoCoR | **Co-learning** |
|------------|----------|-------------|-------|-----------------|
| Sym-20% | 84.81 | 90.29 | 90.43 | **92.21** |
| Sym-50% | 61.49 | 63.45 | 66.00 | **84.49** |
| Sym-80% | 28.98 | 28.03 | 29.19 | **61.20** |
| Asym-40% | 76.30 | 74.25 | 73.95 | **81.42** |

**CIFAR-100 결과 (%):**

| 노이즈 설정 | Standard | Co-teaching | JoCoR | **Co-learning** |
|------------|----------|-------------|-------|-----------------|
| Sym-20% | 57.79 | 64.28 | 62.29 | **66.58** |
| Sym-50% | 33.75 | 32.62 | 30.19 | **54.54** |
| Sym-80% | 8.64 | 6.65 | 6.84 | **35.45** |

**실제 노이즈 데이터셋:**
- Animal-10N: Co-learning **82.95%** (best), **82.18%** (last) — 최고 성능
- Food-101N: Co-learning **87.57%** (best), **86.56%** (last) — 최고 성능

### 2.5 한계점

1. **이론적 근거 부족**: 논문 자체에서 "향후 이론적 기반과 일반화 분석을 탐구할 것"이라고 명시
2. **학습 시간**: 자기지도학습의 특성상 수렴에 긴 학습 시간 필요 (MixUp으로 부분 완화)
3. **하이퍼파라미터 민감도**: $\tau$ (temperature), $\alpha$ (MixUp Beta 파라미터), $\sigma$ (유사도 메트릭) 설정에 따른 민감도
4. **비전 도메인 중심**: 이미지 분류에 초점, 자연어처리나 그래프 등 타 도메인 적용 가능성 미검증
5. **소손실 샘플 미활용 vs. 미검증**: 소손실 트릭을 사용하지 않아 단순화되었지만, 일부 경우 정보 손실 가능성 존재
6. **클래스 불균형 미고려**: 노이즈 유형이 symmetric/asymmetric에 한정

---

## 3. 모델의 일반화 성능 향상과 관련된 심층 분석

### 3.1 일반화 성능 향상 메커니즘

Co-learning의 일반화 성능 향상은 세 가지 상호보완적 메커니즘에서 비롯됩니다.

#### 메커니즘 1: 자기지도학습을 통한 레이블-독립적 표현 학습

내재적 유사성 손실 $\mathcal{L}_{int}$는 레이블과 무관하게 데이터 자체의 고유한 특성을 학습합니다:

$$\mathcal{L}_{int} = \sum_{i=1}^{N} \ell(v_2^{(i)}, v_3^{(i)}) + \ell(v_3^{(i)}, v_2^{(i)})$$

이 손실은 **노이즈 레이블에 전혀 의존하지 않으므로**, 인코더가 노이즈에 오염되지 않은 특징을 학습하도록 유도합니다. 결과적으로 공유 인코더는 레이블 의존적 정보와 특징 의존적 정보를 모두 반영한 표현을 학습하게 됩니다.

#### 메커니즘 2: 구조적 유사성을 통한 정규화 효과

$\mathcal{L}_{str}$은 분류기 헤드와 투영 헤드 간의 관계 구조를 일치시킵니다. 이는 일종의 **지식 증류(Knowledge Distillation)** 효과로 볼 수 있으며:

$$\mathcal{L}_{str} = \sum_{i \neq j} p(d(v^{(i)}, v^{(j)})) \log \frac{p(d(v^{(i)}, v^{(j)}))}{p(d(\tilde{y}^{(i)}, \tilde{y}^{(j)}))}$$

- 자기지도학습으로 학습된 노이즈-독립적 표현 구조를 분류기에 전달
- 분류기가 노이즈 레이블에 과적합되는 것을 방지하는 **암묵적 정규화** 역할 수행
- 두 뷰(view) 간의 상호 제약이 모델이 진짜 데이터 구조를 학습하도록 강제

#### 메커니즘 3: MixUp 증강을 통한 과적합 억제

$$\bar{x}^{(i)} = \lambda \tilde{x}^{(i)} + (1-\lambda)\tilde{x}^{(m(i))}, \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$

MixUp은 두 가지 방식으로 일반화에 기여합니다:
1. **수렴 속도 조절**: 지도학습이 자기지도학습보다 훨씬 빠르게 수렴하는 불균형을 해소
2. **결정 경계 평활화(decision boundary smoothing)**: 선형 보간을 통해 모델이 날카로운 결정 경계 대신 부드러운 경계를 학습하도록 유도

### 3.2 Ablation Study에서 확인된 일반화 기여도

논문의 절제 연구(Ablation Study, Figure 8)는 각 구성 요소의 일반화 기여를 명확히 보여줍니다:

| 구성 | Sym-20% | Sym-50% | 일반화 특성 |
|------|---------|---------|------------|
| CE only | 과적합 발생 | 심각한 과적합 | 낮음 |
| CE + $\mathcal{L}_{int}$ | 소폭 향상 | 여전히 과적합 | 제한적 향상 |
| CE + MixUp | 과적합 억제 | 개선 | 중간 |
| CE + $\mathcal{L}_{int}$ + MixUp | 안정적 | 양호 | 높음 |
| **Co-learning (전체)** | **최고** | **최고** | **가장 높음** |

특히 Sym-80%에서 Co-learning이 61.20%를 달성한 반면, 次선인 Co-teaching+는 30.37%에 그쳤는데, 이는 고노이즈 환경에서의 극적인 일반화 성능 향상을 보여줍니다.

### 3.3 고노이즈 환경에서의 일반화 우월성

기존의 소손실 트릭 기반 방법들은 **고노이즈 환경에서 사용 가능한 깨끗한 샘플이 매우 적어지므로** 성능이 급격히 저하됩니다. 반면 Co-learning은:

- 모든 샘플을 학습에 활용 (소손실 샘플만 선택하지 않음)
- 자기지도학습을 통해 고노이즈 환경에서도 의미 있는 특징 학습 가능
- 이로 인해 Sym-80%와 같은 극단적 노이즈 환경에서 월등한 일반화 성능 달성

---

## 4. 미래 연구에 미치는 영향 및 고려사항

### 4.1 미래 연구에 미치는 영향

#### 영향 1: 자기지도학습과 노이즈 학습의 결합 패러다임 정립
Co-learning은 자기지도학습을 노이즈 레이블 학습의 보조 도구로 명시적으로 활용하는 선구적 연구입니다. 이는 "레이블 없이 학습하는 방법(SSL)이 레이블 품질 문제를 해결할 수 있다"는 새로운 연구 방향성을 제시합니다.

#### 영향 2: 단일 인코더 + 다중 헤드 구조의 확산
두 개의 독립 네트워크 대신 **공유 인코더 + 다중 헤드** 구조는 계산 효율성과 다양성을 동시에 확보하는 설계 원칙을 제시합니다. 이는 향후 멀티태스크 학습, 도메인 적응 등 다양한 분야에 영향을 줄 수 있습니다.

#### 영향 3: 구조적 유사성 손실의 범용성
$\mathcal{L}_{str}$으로 표현된 관계 구조 보존 손실은 지식 증류, 메트릭 학습, 표현 학습 등 여러 분야에 응용 가능한 일반적인 원리를 제시합니다.

#### 영향 4: 사전 지식 불필요 방법론의 가치 증명
노이즈율, 깨끗한 검증 셋 없이도 경쟁력 있는 성능을 달성함으로써, **실제 환경 적용 가능성**을 높이는 연구 방향을 강화합니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

제공된 논문에서 직접 언급된 관련 연구들과, Co-learning과의 비교를 중심으로 분석합니다.

#### 논문 내 인용된 2020년 이후 연구

| 논문 | 핵심 방법 | Co-learning과의 관계 |
|------|-----------|---------------------|
| **DivideMix** (Li et al., ICLR 2020) | GMM으로 깨끗한/노이즈 샘플 분리 후 MixMatch 적용 | 비종단간, 사전 지식 필요 vs. Co-learning은 종단간, 사전 지식 불필요 |
| **BYOL** (Grill et al., NeurIPS 2020) | 모멘텀 인코더로 붕괴 없이 SSL | Co-learning의 SSL 구성 요소로 BYOL 계열 대체 가능성 |
| **SimSiam** (Chen & He, CVPR 2021) | Stop-gradient로 붕괴 방지 | Co-learning의 투영 헤드를 SimSiam 방식으로 개선 가능 |
| **SwAV** (Caron et al., NeurIPS 2020) | 온라인 클러스터링 기반 SSL | Co-learning의 $\mathcal{L}_{int}$을 SwAV로 대체 시 추가 개선 가능성 |
| **APL** (Ma et al., ICML 2020) | 정규화된 손실 함수 조합 | 손실 설계 vs. 프레임워크 설계 — 상호 보완 가능 |

#### 주요 후속 연구 방향 (논문 외부 정보 포함, 확인 가능한 범위 내)

다음은 Co-learning 이후 활발히 연구된 방향들로, **논문에서 직접 언급되지는 않지만** 관련성이 높은 흐름입니다:

1. **Foundation Model 활용**: CLIP 등 대규모 사전학습 모델의 특징을 노이즈 학습에 활용하는 연구 방향
2. **그래프 기반 노이즈 학습**: 샘플 간 관계를 그래프로 모델링하여 노이즈 레이블 처리
3. **확산 모델(Diffusion Model) 결합**: 생성 모델을 활용한 데이터 증강과 노이즈 학습의 결합

> ⚠️ **주의**: 위 3개 항목은 Co-learning 논문(2021) 이후의 일반적인 연구 흐름으로, 특정 논문과의 정확한 수치 비교는 해당 논문들을 직접 확인해야 합니다.

### 4.3 향후 연구 시 고려할 점

#### 고려사항 1: 이론적 일반화 경계 분석
논문 저자들도 인정하듯이, Co-learning의 이론적 기반이 부족합니다. 향후 연구에서는:
- 노이즈율 $\eta$에 따른 일반화 오류 상한(generalization error bound) 도출
- PAC-Learning 프레임워크 내에서의 분석
- $\mathcal{L}_{str}$의 정규화 효과에 대한 이론적 분석

#### 고려사항 2: 다양한 SSL 방법론과의 결합 최적화

$$\mathcal{L}_{int}^{(\text{improved})} = \text{SimSiam}(\cdot) \text{ or } \text{BYOL}(\cdot) \text{ or } \text{SwAV}(\cdot)$$

투영 헤드의 SSL 손실로 최신 SSL 방법(SimSiam, BYOL 등)을 사용하면 추가 성능 향상 가능성이 있습니다.

#### 고려사항 3: 비전 이외 도메인으로의 확장
- **자연어처리**: 텍스트 노이즈 레이블 학습에서 언어 모델 기반 SSL 활용
- **그래프 학습**: 그래프 구조의 자기지도학습과 결합
- **의료 영상**: 전문가 어노테이션 노이즈 처리에 적용

#### 고려사항 4: 구조적 유사성 손실의 개선

현재의 가우시안 기반 유사도 메트릭:
$$p(d) = C_\sigma \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{d-\mu}{\sigma}\right)^2}$$

이를 데이터에 적응적으로 학습하는 방식으로 개선하거나, 다른 커널 함수(Cauchy, Student-t 등)와 비교 연구가 필요합니다.

#### 고려사항 5: 동적 가중치 조정

현재 $\mathcal{L} = \mathcal{L}\_{sup} + \mathcal{L}\_{int} + \mathcal{L}_{str}$에서 각 손실의 가중치가 1:1:1로 고정되어 있습니다. 학습 단계에 따라 동적으로 조정하는 커리큘럼 학습(curriculum learning) 방식 도입을 고려할 수 있습니다:

$$\mathcal{L} = \alpha(t)\mathcal{L}_{sup} + \beta(t)\mathcal{L}_{int} + \gamma(t)\mathcal{L}_{str}$$

여기서 $\alpha(t), \beta(t), \gamma(t)$는 학습 epoch $t$에 따라 변화하는 가중치입니다.

#### 고려사항 6: 노이즈 유형 다양화
- 현재 실험은 symmetric/asymmetric 노이즈에 집중
- 인스턴스 의존 노이즈(instance-dependent noise), 클래스 불균형 노이즈 등 더 현실적인 노이즈 유형에 대한 검증 필요

#### 고려사항 7: 대규모 데이터셋 확장성
- ImageNet 규모의 데이터셋에서의 성능 검증 필요
- 배치 크기, 메모리 요구사항 등 실용적 확장성 분석

---

## 참고자료 및 출처

**주 논문:**
- Tan, C., Xia, J., Wu, L., & Li, S. Z. (2021). **Co-learning: Learning from Noisy Labels with Self-supervision**. *Proceedings of the 29th ACM International Conference on Multimedia (MM '21)*. https://doi.org/10.1145/3474085.3475622
- arXiv: https://arxiv.org/abs/2108.04063

**논문 내 주요 인용 참고문헌:**
- Han, B., et al. (2018). Co-teaching: Robust training of deep neural networks with extremely noisy labels. *NeurIPS 2018*.
- Wei, H., et al. (2020). Combating noisy labels by agreement: A joint training method with co-regularization. *CVPR 2020*.
- Li, J., Socher, R., & Hoi, S. C. H. (2020). DivideMix: Learning with Noisy Labels as Semi-supervised Learning. *ICLR 2020*.
- Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations (SimCLR). *ICML 2020*.
- Grill, J. B., et al. (2020). Bootstrap Your Own Latent (BYOL). *NeurIPS 2020*.
- Chen, X., & He, K. (2021). Exploring Simple Siamese Representation Learning (SimSiam). *CVPR 2021*.
- Caron, M., et al. (2020). Unsupervised learning of visual features by contrasting cluster assignments (SwAV). *NeurIPS 2020*.
- Ma, X., et al. (2020). Normalized Loss Functions for Deep Learning with Noisy Labels (APL). *ICML 2020*.
- Zhang, H., et al. (2018). mixup: Beyond Empirical Risk Minimization. *ICLR 2018*.
- van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding (InfoNCE). *arXiv:1807.03748*.
