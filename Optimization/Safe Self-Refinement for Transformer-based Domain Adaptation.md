# Safe Self-Refinement for Transformer-based Domain Adaptation (SSRT) 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

SSRT(Safe Self-Refinement for Transformer-based Domain Adaptation)는 두 가지 핵심 문제를 동시에 해결합니다:

1. **표현력(Representation)**: CNN 기반 백본(ResNet)의 한계를 극복하기 위해 Vision Transformer(ViT)를 UDA에 도입
2. **안전한 학습(Safe Training)**: 대규모 도메인 갭에서의 모델 붕괴(collapse)를 방지하는 Safe Self-Refinement 전략 제안

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| ViT 기반 UDA | UDA에 Vision Transformer 백본을 최초로 본격 탐구 |
| Multi-layer Perturbation | 잠재 토큰 시퀀스 공간에서 다중 레이어 perturbation 적용 |
| Bi-directional Self-Refinement | 양방향 KL divergence 기반 자기 정제 손실 |
| Safe Training Mechanism | 다양성(diversity) 모니터링 기반 적응적 학습 설정 조정 |
| SOTA 성능 달성 | Office-Home 85.43%, VisDA-2017 88.76%, DomainNet 45.2% |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**Unsupervised Domain Adaptation(UDA)**의 근본 과제는 레이블이 있는 소스 도메인의 지식을 레이블이 없는 타겟 도메인으로 이전하는 것입니다. 구체적인 문제점:

- **표현력 한계**: 기존 CNN(ResNet) 백본은 DomainNet 같은 대규모·대도메인 갭 데이터셋에서 33.3% 수준에 그침
- **모델 붕괴 위험**: 대규모 도메인 갭에서 self-training 기반 방법들이 noisy supervision으로 인해 모델이 붕괴되는 현상 발생
- **하이퍼파라미터 민감성**: 태스크마다 최적 학습 설정이 달라, 고정된 하이퍼파라미터로는 일부 태스크에서 실패

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 전체 최적화 목표

$$\min_{f,g}\max_{d} \mathcal{L} = \mathcal{L}_{\mathrm{CE}} - \mathcal{L}_{\mathrm{d}} + \beta \mathcal{L}_{\mathrm{tgt}} $$

- $\mathcal{L}_{\mathrm{CE}}$: 소스 도메인 교차 엔트로피 손실
- $\mathcal{L}_{\mathrm{d}}$: 도메인 적대적 손실(Domain Adversarial Loss)

$$\mathcal{L}_{\mathrm{d}} = -\mathbb{E}_{\boldsymbol{x}\sim \mathcal{D}_s}\big[\log d(f(\boldsymbol{x}))\big] - \mathbb{E}_{x\sim \mathcal{D}_t}\big[\log(1-d(f(\boldsymbol{x})))\big]$$

- $\mathcal{L}\_{\mathrm{tgt}}$: 타겟 도메인 손실 (본 논문에서 $\mathcal{L}_{\mathrm{SR}}$로 구체화)
- $\beta$: 트레이드오프 파라미터

#### 2.2.2 Multi-layer Perturbation

타겟 도메인 이미지 $\boldsymbol{x}$의 $l$-번째 트랜스포머 블록 입력 토큰 시퀀스 $b^l_x$에 대해, 임의로 선택된 다른 타겟 이미지 $\boldsymbol{x}_r$의 토큰 시퀀스를 오프셋으로 활용:

$$\tilde{b}^{l}_x = b^{l}_x + \alpha [b^{l}_{x_r} - b^{l}_x]_{\times} $$

- $\alpha$: perturbation 스칼라
- $[\cdot]_{\times}$: 그래디언트 역전파 차단(stop-gradient)
- $b^l_x$를 통해서는 그래디언트가 흐름 → 모델 파라미터 업데이트에 기여

**핵심**: 레이어 $l \in \{0, 4, 8\}$ 중 랜덤하게 하나를 선택하여 perturbation 적용 → 다중 레이어에 동시 정규화 효과

#### 2.2.3 Bi-directional Self-Refinement Loss

KL Divergence:

$$D_{\mathrm{KL}}(\boldsymbol{p}_t \| \boldsymbol{p}_s) = \sum_{i} \boldsymbol{p}_t[i] \log\frac{\boldsymbol{p}_t[i]}{\boldsymbol{p}_s[i]} $$

양방향 자기 정제 손실:

```math
\mathcal{L}_{\mathrm{SR}} = \mathbb{E}_{\mathcal{B}_t\sim \mathcal{D}_t} \Big\{\omega\, \mathbb{E}_{\boldsymbol{x}\sim F[\mathcal{B}_t;\boldsymbol{p}]} D_{\mathrm{KL}}(\boldsymbol{p}_x \| \tilde{\boldsymbol{p}}_x) + (1-\omega)\, \mathbb{E}_{\boldsymbol{x}\sim F[\mathcal{B}_t;\tilde{\boldsymbol{p}}]} D_{\mathrm{KL}}(\tilde{\boldsymbol{p}}_x \| \boldsymbol{p}_x)\Big\}
```

- $\omega \sim \mathcal{B}(0.5)$: 베르누이 분포에서 샘플링한 확률 변수
- $\boldsymbol{p}_x$: 원본 입력의 예측 확률 벡터
- $\tilde{\boldsymbol{p}}_x$: perturbation된 입력의 예측 확률 벡터

Confidence Filter:

$$F[\mathcal{D}; \boldsymbol{p}] = \{\boldsymbol{x} \in \mathcal{D} \mid \max(\boldsymbol{p}_x) > \epsilon\} $$

- $\epsilon$: 신뢰도 임계값 (실험에서 $\epsilon = 0.4$ 사용)

**양방향 그래디언트 역전파**의 중요성: Teacher 확률과 Student 확률 모두에 그래디언트를 역전파 → 단일 확률로 인한 과도한 그래디언트 방지

#### 2.2.4 Safe Training: 적응적 스칼라

$$r(t) = \begin{cases} \sin\!\left(\dfrac{\pi}{2T_r}(t - t_r)\right) & \text{if } t - t_r < T_r \\ 1.0 & \text{otherwise} \end{cases} $$

- $\alpha_r = r\alpha$, $\beta_r = r\beta$로 perturbation 강도와 손실 가중치를 동적으로 조정
- 모델 붕괴 감지 시 $t_r$을 현재 스텝으로 리셋, 마지막 스냅샷으로 모델 복원

다양성 측정:

$$\mathrm{div}(t; \mathcal{B}_t) = \mathrm{unique\_labels}(h(\mathcal{B}_t)) $$

---

### 2.3 모델 구조

```
입력 이미지
    ↓
Patch Embedding (Conv + Position Embedding)
    ↓
Transformer Block 0  ← 랜덤 오프셋 추가 (Multi-layer perturbation)
    ↓
Transformer Block 4  ← 랜덤 오프셋 추가
    ↓
Transformer Block 8  ← 랜덤 오프셋 추가
    ↓
...
    ↓
Classifier Head (Dropout 포함)
    ↓
Softmax → 클래스 예측
```

**두 브랜치 구조**:
- **원본 브랜치**: $b^l_x \rightarrow p_x$
- **Perturbed 브랜치**: $\tilde{b}^l_x \rightarrow \tilde{p}_x$
- 두 브랜치는 파라미터 공유, Confidence Filter를 통해 신뢰도 높은 예측만 사용
- **Domain Discriminator**가 추가로 존재 (적대적 적응)

**백본**: ViT-B/16, ViT-S/16 (ImageNet 사전학습)
**하이퍼파라미터**: $\alpha = 0.3$, $\beta = 0.2$, $\epsilon = 0.4$, $T = 1000$, $L = 4$

---

### 2.4 성능 향상 및 한계

#### 성능 향상

| 벤치마크 | SSRT-B | 이전 SOTA (CNN) | 개선폭 |
|---|---|---|---|
| Office-Home | **85.43%** | SHOT 71.8% | +13.6%p |
| VisDA-2017 | **88.76%** | CDTrans 88.4% | +0.36%p |
| DomainNet | **45.2%** | MDD+SCDA 33.3% | +11.9%p |
| Office-31 | **93.5%** | TVT 93.8% | 유사 수준 |

- Baseline-B(ViT + 적대적 적응) 대비 Office-Home +4.38%, VisDA +3.53%, DomainNet +6.7% 향상
- 특히 도메인 갭이 큰 `qdr` 타겟 태스크에서 29.3%로 타 방법 대비 압도적 우위

#### 한계

- DomainNet의 45.2%는 여전히 포화(saturation)와 거리가 멀며, 성능 향상 여지가 많음
- **단일 소스 도메인** 설정만 고려; 다중 소스 도메인 활용 미탐구
- 타겟 도메인에 대한 **메타 지식(meta knowledge)** 미반영
- Safe Training의 다양성 감지에 **거짓 양성(false-positive)** 발생 가능성
- 하이퍼파라미터($T$, $L$, $\alpha$, $\beta$, $\epsilon$)가 여전히 존재하며, 새로운 도메인에서의 일반화 보장 불명확

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 ViT 백본의 전이 가능성(Transferability)

ViT의 글로벌 자기-어텐션(global self-attention)과 대규모 ImageNet 사전학습은 CNN보다 **도메인 불변(domain-invariant) 특징 표현**을 자연스럽게 학습합니다. 논문에서 단순 ViT-B + 적대적 적응만으로 DomainNet 38.5%를 달성, ResNet-101 SOTA(33.3%)를 이미 초과합니다.

### 3.2 Multi-layer Perturbation의 일반화 기여

잠재 공간(latent space)에서의 perturbation은 단순 입력 증강보다 더 강력한 정규화를 제공합니다:

$$\tilde{b}^{l}_x = b^{l}_x + \alpha [b^{l}_{x_r} - b^{l}_x]_{\times}$$

- 다른 타겟 이미지의 토큰 시퀀스를 오프셋으로 사용함으로써 **타겟 도메인 데이터 매니폴드 내에서 의미론적으로 유효한 방향**으로 perturbation 가능
- 레이어 $l \in \{0, 4, 8\}$ 중 랜덤 선택으로 **여러 레이어에 동시 정규화** → 특정 레이어 의존성 감소

Table 5에서 확인:
- `SSRT-B (raw)` (입력 이미지 perturbation): DomainNet 44.2%
- `SSRT-B` (잠재 공간 perturbation): DomainNet **45.2%**

### 3.3 Bi-directional Supervision의 안정성 기여

양방향 KL divergence는 단방향 대비 **더 강건한 정규화**를 제공합니다. 단방향 ($\omega=0$ 또는 $\omega=1$)에서는 confidence threshold가 클수록 성능 저하, 특정 태스크에서 모델 붕괴가 발생하는 반면, 양방향은 두 손실의 보완적 특성으로 이런 문제를 완화합니다(Table 6).

### 3.4 Safe Training의 역할

Safe Training은 **도메인 갭이 매우 큰 태스크에서의 일반화 실패를 구조적으로 방지**합니다:

- `clp→qdr` 태스크에서 Safe Training 미적용 시 약 10k iteration 후 정확도 0.3%로 붕괴
- Safe Training 적용 시 붕괴 감지 후 스냅샷 복원 및 $r$ 리셋으로 안정적 학습 유지

이는 다양한 UDA 태스크에 걸친 **적응적 일반화(adaptive generalization)** 능력을 크게 향상시킵니다.

### 3.5 Confidence Filter의 역할

$$F[\mathcal{D}; \boldsymbol{p}] = \{\boldsymbol{x} \in \mathcal{D} \mid \max(\boldsymbol{p}_x) > \epsilon\}$$

신뢰도가 낮은 noisy pseudo-supervision을 필터링함으로써 **잘못된 정제(refinement)로 인한 일반화 성능 저하 방지**.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 향후 연구에 미치는 영향

#### (1) Vision Transformer의 UDA 표준화 촉진

SSRT는 ViT를 UDA에 본격 도입한 초기 연구 중 하나로, 이후 트랜스포머 기반 DA 연구의 기준점이 됩니다. CDTrans, TVT와 함께 **트랜스포머 기반 UDA 패러다임 전환**을 선도했습니다.

#### (2) 잠재 공간 perturbation의 새로운 방향

입력 이미지가 아닌 **트랜스포머 토큰 시퀀스 공간에서의 perturbation**이라는 새로운 아이디어는 semi-supervised learning, domain generalization 등에서도 응용 가능합니다.

#### (3) 모델 붕괴 감지와 적응적 학습의 중요성 부각

Safe Training 메커니즘은 **훈련 중 다양성 측정을 통한 자동 학습 설정 조정**의 실용성을 입증, 향후 다양한 도메인 적응 방법론에서 유사한 접근이 활용될 것으로 예상됩니다.

#### (4) 대규모 벤치마크에서의 한계 노출

DomainNet 45.2%는 당시 SOTA이나 여전히 포화와 거리가 멀어, **대규모 멀티 도메인 적응** 문제의 난이도를 재조명했습니다.

### 4.2 향후 연구 시 고려할 점

#### (1) 다중 소스 도메인 확장

논문 자체가 언급한 한계로, 여러 소스 도메인 정보를 통합하는 **Multi-source UDA** 확장이 중요합니다. DomainNet의 경우 6개 도메인 간 모든 조합 이용 시 성능 향상 가능성이 높습니다.

#### (2) 더 강력한 사전학습 모델 활용

- CLIP, DINO, MAE 등 대규모 자기지도학습(self-supervised) 사전학습 모델을 백본으로 활용 시 전이 가능성 대폭 향상 기대
- 최근 연구들(예: PMTrans, SPA+, 등)이 이 방향을 적극 탐구

#### (3) Perturbation 전략 고도화

- 현재 랜덤 오프셋 방식은 단순하지만 효과적; **적대적 perturbation(Adversarial Perturbation)**, **학습 가능한 perturbation** 등으로 발전 가능
- 어텐션 맵 기반의 의미론적으로 중요한 토큰에만 선택적 perturbation 적용 고려

#### (4) Safe Training의 일반화

- 현재 다양성 측정으로 고유 예측 레이블 수를 사용하는데, 보다 정교한 분포 측정(예: 엔트로피, Jensen-Shannon Divergence 등) 적용 연구 필요
- 연속적 모니터링 비용 절감을 위한 효율적 구현 방안 탐구

#### (5) 목표 도메인 메타 지식 활용

논문이 언급하지 못한 **target domain prior knowledge**(예: 도메인 레이블 없는 클러스터 구조, 시각적 통계 정보 등)를 Self-Refinement에 통합하는 연구 필요

#### (6) 이론적 기반 강화

Safe Training의 다양성-정확도 상관관계, 양방향 KL divergence의 수렴 보장 등에 대한 **이론적 분석**이 부족하므로, 향후 이론적 근거 마련이 필요합니다.

---

## 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 발표 | 백본 | 핵심 아이디어 | DomainNet | Office-Home | VisDA |
|---|---|---|---|---|---|---|
| SHOT (Liang et al.) | ICML 2020 | ResNet | 소스 가설 전이, 정보 극대화 | - | 71.8% | 82.9% |
| CDTrans (Xu et al.) | arXiv 2021 | DeiT-B | 크로스 도메인 어텐션 | 37.0% | 80.5% | 88.4% |
| TVT (Yang et al.) | arXiv 2021 | ViT-B | 전이 가능 자기-어텐션 모듈 | - | 83.56% | 83.92% |
| **SSRT (Sun et al.)** | **CVPR 2022** | **ViT-B** | **Safe Self-Refinement + ViT** | **45.2%** | **85.43%** | **88.76%** |

**CDTrans와의 차별점**: CDTrans는 소스-타겟 이미지 쌍 간 크로스 어텐션을 사용하는 반면, SSRT는 타겟 도메인 데이터와 그 perturbed 버전 쌍을 사용하여 **의미론적 클래스 보장** 및 **안전한 학습** 구현

**TVT와의 차별점**: TVT는 전이 가능 멀티헤드 자기-어텐션 모듈 설계에 집중하는 반면, SSRT는 **잠재 공간 perturbation + 안전 메커니즘** 조합으로 특히 도메인 갭이 큰 경우에 더 강건

---

## 참고 자료

- **논문 원문**: Tao Sun, Cheng Lu, Tianshuo Zhang, Haibin Ling. "Safe Self-Refinement for Transformer-based Domain Adaptation." *CVPR 2022*, pp. 7191–7200.
- **관련 논문 (논문 내 인용)**:
  - Dosovitskiy et al., "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale." *ICLR 2021* [3]
  - Liang et al., "Do We Really Need to Access the Source Data?" *ICML 2020* [12]
  - Xu et al., "CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation." *arXiv 2021* [39]
  - Yang et al., "TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation." *arXiv 2021* [40]
  - Long et al., "Conditional Adversarial Domain Adaptation." *NeurIPS 2018* [14]
  - Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation." *ICML 2015* [4]
  - Peng et al., "Moment Matching for Multi-source Domain Adaptation." *ICCV 2019* [19]
  - Sohn et al., "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence." *NeurIPS 2020* [25]
- **코드**: https://github.com/tsun/SSRT
