
# RWOT: Reliable Weighted Optimal Transport for Unsupervised Domain Adaptation 

> **논문 정보:**
> - **제목:** Reliable Weighted Optimal Transport for Unsupervised Domain Adaptation
> - **저자:** Renjun Xu, Pelen Liu, Liyan Wang, Chao Chen, Jindong Wang
> - **학회:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 4394–4403
> - **공개 접근:** [CVPR 2020 Open Access](https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_Reliable_Weighted_Optimal_Transport_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.html) | [Microsoft Research](https://www.microsoft.com/en-us/research/publication/reliable-weighted-optimal-transport-for-unsupervised-domain-adaptation/) | [Semantic Scholar](https://www.semanticscholar.org/paper/Reliable-Weighted-Optimal-Transport-for-Domain-Xu-Liu/cceadc490b6b141ff93b0f00b42212e64647b00d)

---

## 1. 핵심 주장 및 주요 기여 (요약)

Optimal Transport(OT)는 소스 도메인과 타겟 도메인의 표현을 정렬하는 데 유망한 메트릭이지만, 기존 OT 기반 연구들은 대부분 도메인 내부(intra-domain) 구조를 무시하여 단순한 쌍별(pair-wise) 매칭만을 달성한다.

특히, 클러스터의 경계 근방이나 해당 클래스 중심에서 멀리 떨어진 타겟 샘플들은 소스 도메인으로부터 학습된 결정 경계(decision boundary)에 의해 오분류되기 쉽다.

이 문제를 해결하기 위해, RWOT는 다음 **두 가지 핵심 기여**를 제안한다:

① **Shrinking Subspace Reliability (SSR):** 공간적 프로토타입 정보(spatial prototypical information)와 도메인 내부 구조를 활용하여, 도메인 간 샘플 수준의 도메인 불일치를 동적으로 측정한다.

② **Weighted Optimal Transport 전략:** SSR 기반의 가중 최적 수송 전략으로 정밀한 쌍별 OT를 수행하여 타겟 도메인에서 결정 경계 근처 샘플들의 부정적 전이(negative transfer)를 줄인다. 또한, **판별적 센트로이드 클러스터링 전략(discriminative centroid clustering)**을 통해 전이 특징(transfer features)을 학습한다.

철저한 평가 결과, RWOT는 표준 도메인 적응 벤치마크에서 기존 최신 방법들을 능가한다.

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

딥 신경망은 대규모 레이블 데이터에서 강력한 표현을 학습할 수 있지만, 입력 분포의 변화에 항상 잘 일반화되지는 않는다. 도메인 적응 알고리즘은 도메인 이동(domain shift)으로 인한 성능 저하를 보완하기 위해 제안되어 왔다.

기존 OT 기반 UDA 방법들의 문제는 두 가지로 정리된다:

| 문제 | 설명 |
|---|---|
| **도메인 내부 구조 무시** | 클러스터 구조, 클래스 중심 정보를 OT 비용 행렬에 반영하지 않음 |
| **경계 샘플의 부정적 전이** | 결정 경계 근처 타겟 샘플들이 소스 분류기에 의해 오분류됨 |

이는 결정 경계 근방에 분포한 "하드 정렬(hard-aligned)" 타겟 샘플들이 부정적 전이를 유발하는 비지도 도메인 적응의 전형적인 예시다.

---

### 2-2. 제안 방법 및 수식

#### (A) 전체 최적화 목표

RWOT의 전체 손실 함수는 세 가지 손실의 가중 합으로 구성된다:

$$\mathcal{L} = \mathcal{L}_{cls} + \alpha \mathcal{L}_{p} + \beta \mathcal{L}_{g}$$

여기서 $G_f$는 피처 생성기(feature generator), $G_y$는 적응형 분류기(adaptive classifier)이며, $\mathcal{L}\_g$는 SSR 기반 가중 최적 수송 손실, $\mathcal{L}\_p$는 판별적 센트로이드 손실(discriminative centroid loss), $\mathcal{L}_{cls}$는 표준 교차 엔트로피 손실(cross-entropy loss)이고, $\alpha$와 $\beta$는 하이퍼파라미터다.

#### (B) Optimal Transport 기본 공식

표준 OT 문제는 소스 $\mu_s$와 타겟 $\mu_t$ 사이의 Wasserstein 거리를 최소화하는 수송 계획(transport plan) $\gamma^*$를 구하는 것이다:

$$W(\mu_s, \mu_t) = \min_{\gamma \in \Pi(\mu_s, \mu_t)} \int_{\mathcal{X} \times \mathcal{Y}} c(x_s, x_t) \, d\gamma(x_s, x_t)$$

RWOT는 이를 **가중(weighted)** 형태로 확장한다. 수송 비용 행렬 $Q$는 SSR에 의해 동적으로 구성된다:

$$Q = \lambda \cdot M + (1 - \lambda) \cdot D$$

- $M$: 도메인 내부 구조(intra-domain structure)에 기반한 비용 행렬
- $D$: 공간적 프로토타입 정보(spatial prototypical information)에 기반한 비용 행렬
- $\lambda$: 두 정보의 기여도를 균형 있게 조절하는 동적 가중치

이 SSR 비용 행렬 $Q$는 훈련 과정에서 공간적 프로토타입 정보와 도메인 내부 구조의 기여를 동적으로 균형 잡도록 설계되었다. 초기 단계에서는 소스 도메인에서 학습된 결정 경계가 타겟 샘플 분류에 신뢰성이 낮고, 점차 결정 경계가 타겟 샘플의 신뢰할 수 있는 intra-domain 구조를 획득하여 더 나은 정렬을 달성한다.

#### (C) Weighted OT 손실 $\mathcal{L}_g$

$$\mathcal{L}_g = \sum_{i,j} \gamma^*_{ij} \cdot Q_{ij} + \eta \cdot H(\gamma)$$

여기서 $H(\gamma) = -\sum_{i,j} \gamma_{ij} \log \gamma_{ij}$는 엔트로피 정규화 항(Sinkhorn 알고리즘 적용)이고, $\eta > 0$는 정규화 강도이다.

#### (D) Discriminative Centroid Loss $\mathcal{L}_p$

클래스 내 컴팩트함(intra-class compactness)과 클래스 간 분리(inter-class separability)를 위한 센트로이드 손실:

$$\mathcal{L}_p = \sum_{k=1}^{K} \sum_{x_s \in \mathcal{C}_k} \| f(x_s) - c_k \|^2$$

- $c_k$: $k$번째 클래스의 공유 클래스 센트로이드(shared class center)
- $f(\cdot)$: 피처 생성기 $G_f$의 출력

#### (E) Shrinking Subspace Reliability (SSR) — 신뢰도 계산

타겟 샘플 $x_t$에 대한 신뢰도 점수는 클래스 센트로이드와의 거리를 기반으로 계산되어, 신뢰도가 낮은 샘플(경계 근처 샘플)에 낮은 가중치를 부여한다:

$$r(x_t) = \exp\left(-\frac{\|f(x_t) - c_{\hat{y}}\|^2}{\sigma^2}\right)$$

RWOT는 SSR을 통해 공간적 프로토타입 정보와 도메인 내부 구조를 활용하여 도메인 간 샘플 수준의 도메인 불일치를 동적으로 측정한다.

이를 통해 수송 계획에서 신뢰도 높은 샘플에는 높은 가중치를, 신뢰도 낮은 경계 샘플에는 낮은 가중치를 부여한다:

$$\gamma^* = \arg\min_{\gamma \in \Pi} \sum_{i,j} \gamma_{ij} \cdot r(x_t^j) \cdot Q_{ij}$$

---

### 2-3. 모델 구조

RWOT 아키텍처는 피처 생성기 $G_f$, 적응형 분류기 $G_y$, SSR 기반 가중 최적 수송 손실 $\mathcal{L}\_g$, 판별적 센트로이드 손실 $\mathcal{L}\_p$, 표준 교차 엔트로피 손실 $\mathcal{L}_{cls}$로 구성된다.

```
[Source Domain] ──► [Feature Generator Gf] ──► [Adaptive Classifier Gy] ──► Lcls
[Target Domain] ──► [Feature Generator Gf] ──►  SSR 계산
                                              │
                       ┌──────────────────────┘
                       │
                  ┌────▼────────────────────┐
                  │  Q = λM + (1-λ)D (SSR) │
                  └────────────┬───────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Weighted OT (Lg)   │   ← Sinkhorn 알고리즘
                    └──────────┬──────────┘
                               │
              ┌────────────────▼──────────────────┐
              │  Discriminative Centroid Loss (Lp) │
              └───────────────────────────────────┘
              
Total Loss: L = Lcls + αLp + βLg
```

RWOT는 SSR과 판별적 센트로이드 손실을 통해 공간적 프로토타입 정보와 intra-domain 구조를 활용하며, 소스와 타겟 도메인 모두에서 intra-class 컴팩트함과 inter-class 분리성을 달성한다.

---

### 2-4. 성능 향상

Ablation Study를 통해 가중 최적 수송 전략과 판별적 센트로이드 손실의 개별 기여를 분리하여 Office-31 및 Digits 데이터셋에서 DeepJDOT 및 RWOT의 세 가지 변형 모델과 비교 평가하였다.

벤치마크 평가 요약:

| 벤치마크 데이터셋 | 설명 |
|---|---|
| **Office-31** | Amazon(A), DSLR(D), Webcam(W), 31개 클래스 |
| **Office-Home** | 4개 도메인(Art, Clipart, Product, Real-World), 65개 클래스 |
| **VisDA-2017** | 합성 → 실제 이미지, 12개 클래스, 28만 장 |
| **Digits** | SVHN→MNIST 등 숫자 도메인 적응 |

RWOT-C(SSR 없음)와 RWOT의 모든 변형 모델이 각 실험에서 DeepJDOT를 유의미하게 능가하며, RWOT는 A→D 태스크에서 31개의 클러스터를 명확한 경계로 달성한다.

RWOT는 세 가지 변형 모델 및 DeepJDOT 대비 큰 폭으로 성능이 향상되어, 두 전략(SSR + centroid loss)의 상호 보완적 효과를 검증한다.

---

## 3. 모델의 일반화 성능 향상 가능성

RWOT가 일반화 성능을 향상시키는 핵심 메커니즘은 다음 세 가지이다.

### 3-1. 동적 신뢰도 측정 (SSR)

SSR은 공간적 프로토타입 정보와 intra-domain 구조를 활용하여 도메인 간 샘플 수준의 도메인 불일치를 동적으로 측정하며, 이에 기반한 가중 OT 전략으로 정밀한 쌍별 수송을 달성함으로써 결정 경계 근처 샘플의 부정적 전이를 감소시킨다.

이를 통해 훈련 과정 전반에 걸쳐 신뢰도 정보가 점진적으로 업데이트되므로, **다양한 도메인 갭(domain gap) 크기에 대해 적응적으로 동작**할 수 있다.

### 3-2. Intra-class 컴팩트함과 Inter-class 분리성

RWOT는 소스 및 타겟 도메인 모두에서 intra-class 컴팩트함과 inter-class 분리성을 달성한다.

이 두 가지 속성은 타겟 도메인의 미지 샘플에 대한 분류 경계가 더 명확해지도록 하여, 새로운 분포에서도 일반화 성능을 높인다.

### 3-3. 부정적 전이 억제

RWOT는 공간적 프로토타입 정보와 intra-domain 구조를 활용하여 결정 경계 근처에 분포한 타겟 샘플들에 의한 부정적 전이를 감소시킨다.

신뢰도가 낮은 샘플(경계 샘플)을 OT 계획에서 하향 가중함으로써, 도메인 정렬 시 노이즈 역할을 하는 샘플들의 영향을 효과적으로 억제한다.

### 3-4. Shrinking Subspace의 의미

"Shrinking(수축)" 개념은 훈련 초기에는 넓은 서브스페이스(공간적 프로토타입 정보 위주)를 사용하고, 훈련이 진행됨에 따라 점차 신뢰할 수 있는 intra-domain 구조로 수렴하는 **curriculum learning**적 특성을 가진다. 이는 과도한 정렬로 인한 과적합(overfitting)을 방지하고, 보다 안정적인 학습 궤적을 형성하여 일반화에 유리하다.

---

## 4. 미래 연구에 미치는 영향 및 고려할 점

### 4-1. 앞으로의 연구에 미치는 영향

#### ① 신뢰도 기반 가중치의 확산
Li et al. (2020)은 네트워크 분류기의 예측 피드백을 활용하여 수송 비용을 재가중하는 Enhanced Transport Distance(ETD)를 제안했으며, RWOT는 intra-domain 구조를 활용하여 도메인 불일치를 동적으로 최적화하고 가중 방식으로 OT를 수행한다. RWOT 이후, 예측 피드백 + 신뢰도 결합 방식이 다양한 후속 연구에서 채택되었다.

#### ② OT 기반 UDA 연구 흐름 촉진
TIDOT(2021)은 OT 프레임워크 하에서 교사 모방 학습(imitation learning)을 제안하고, NWD(2022)는 핵 노름(nuclear norm) Wasserstein 거리를 도입하여 판별적 정렬(discriminative alignment)을 달성한다.

#### ③ 프로토타입 기반 OT로의 발전
COT(Liu et al., 2023)는 소스와 타겟 클러스터 센터 간 OT를 활용하여 도메인 정렬과 클래스 불균형 완화를 동시에 달성한다.

#### ④ 계층적·비균형 OT로의 확장
DeepHOT(Xu et al., 2022)은 전역 도메인 분포와 지역 이미지 구조를 정렬하기 위해 이미지 수준 OT를 도메인 수준 비용 계산에 통합하며, tractability를 위해 슬라이스 Wasserstein 거리와 비균형 OT(unbalanced OT)를 사용한다.

DDW-OT(Decomposed-Distance Weighted OT)는 UDA를 위한 더 나은 적응을 위해 제안되었으며, 또 다른 연구는 판별적 클래스 인식 정보를 완전히 활용하는 새로운 비균형 최적 수송(UOT) 전략으로 정밀한 쌍별 매칭을 달성한다.

#### ⑤ 비전-언어 모델로의 확장
DeepJDOT 및 관련 딥 OT 접근법들은 딥 잠재 공간에서 공동 표현 학습과 OT 기반 정렬을 위한 신경망을 통합하며(Damodaran et al., 2018; Jiang et al., 2023), 이는 CLIP 등 대형 비전-언어 모델에까지 확장되고 있다.

---

### 4-2. 앞으로 연구 시 고려할 점

#### 📌 한계 및 개선 방향

| 한계 | 설명 | 개선 방향 |
|---|---|---|
| **계산 비용** | OT 계획 계산은 $O(n^2)$ 또는 $O(n^3)$ 복잡도 | 미니배치 OT, 계층적 OT 활용 |
| **이진 도메인 가정** | 소스-타겟 1:1 구조만 가정 | 다중 소스 도메인 확장 필요 |
| **클래스 불균형 미고려** | 신뢰도가 클래스 불균형에 취약 | Unbalanced OT 또는 PPOT 결합 |
| **레이블 부재 타겟** | 타겟 신뢰도 계산이 불안정 | 반지도학습 또는 자기지도학습 결합 |
| **하이퍼파라미터 민감성** | $\alpha$, $\beta$, $\lambda$ 등 민감 | 자동 하이퍼파라미터 탐색 필요 |

#### 📌 향후 연구 아이디어

1. **Source-Free DA와 결합:** 소스 데이터 없이도 신뢰도를 동적으로 추정하는 방향
2. **부분 도메인 적응(Partial DA):** 타겟 도메인에만 존재하는 클래스에 대한 처리
3. **대규모 비전-언어 모델 결합:** CLIP, DINO 등의 프리트레인 특징 + RWOT 결합
4. **의료 영상·시계열 데이터 적용:** 비전 외 도메인으로의 일반화 검증
5. **이론적 보장 강화:** 현재 SSR의 신뢰도 추정에 대한 엄밀한 수렴 이론 부재

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

OT 기반 방법들은 추가적이거나 여분의 타겟 피처를 가진 도메인으로 확장되었으며, 고정 차원 케이스와 유사한 이론적 경계를 갖는 의사 레이블링 및 신중하게 구조화된 비용 함수를 사용한다.

| 방법 | 연도 | 핵심 아이디어 | RWOT 대비 특징 |
|---|---|---|---|
| **ETD** (Li et al.) | CVPR 2020 | 분류기 예측 피드백 → 수송 비용 재가중 | 신뢰도 계산 방식 차이 |
| **TIDOT** (Nguyen et al.) | IJCAI 2021 | 교사 모방 학습 + OT | 교사-학생 프레임워크 도입 |
| **Unbalanced Mini-batch OT** (Fatras et al.) | ICML 2021 | 비균형 미니배치 OT | 클래스 불균형 명시적 처리 |
| **DDW-OT** | - | 분해된 거리 가중 OT | 수송 비용 분해 방식 차이 |
| **NWD** (Chen et al.) | 2022 | 핵 노름 Wasserstein 거리 | Discriminator-free 방식 |
| **DeepHOT** (Xu et al.) | 2022 | 계층적 OT (이미지+도메인 수준) | 다중 스케일 정렬 |
| **COT** (Liu et al.) | 2023 | 클러스터 센터 간 OT | 클래스 불균형 완화 |
| **PPOT** (2024) | 2024 | 프로토타입 기반 부분 OT | UniDA (범용 DA) 지원 |
| **P²OT** (2024) | 2024 | 점진적 부분 OT | 불균형 클러스터링 |

이러한 OT 기반 UDA 프레임워크들은 숫자 인식(MNIST, USPS, SVHN), Office-31, Office-Home, VisDA, 텍스트, 의료 영상, 오디오 등 다양한 벤치마크에서 종합적으로 검증되었으며, 보고된 정확도는 지속적으로 최신 방법들과 동등하거나 이를 초과하고 있다.

---

## 📚 참고 자료 (출처)

1. **[주 논문]** Xu, R., Liu, P., Wang, L., Chen, C., & Wang, J. (2020). *Reliable Weighted Optimal Transport for Unsupervised Domain Adaptation*. **CVPR 2020**, pp. 4394–4403.
   - [CVPR Open Access](https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_Reliable_Weighted_Optimal_Transport_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.html)
   - [IEEE Xplore](https://ieeexplore.ieee.org/document/9156476/)
   - [Microsoft Research](https://www.microsoft.com/en-us/research/publication/reliable-weighted-optimal-transport-for-unsupervised-domain-adaptation/)
   - [Semantic Scholar (PDF)](https://www.semanticscholar.org/paper/Reliable-Weighted-Optimal-Transport-for-Domain-Xu-Liu/cceadc490b6b141ff93b0f00b42212e64647b00d)
   - [ResearchGate](https://www.researchgate.net/publication/343461109_Reliable_Weighted_Optimal_Transport_for_Unsupervised_Domain_Adaptation)

2. **[관련 연구]** Nguyen, T. et al. (2021). *TIDOT: A Teacher Imitation Learning Approach for Domain Adaptation with Optimal Transport*. **IJCAI 2021**. [ResearchGate](https://www.researchgate.net/publication/353832739_TIDOT_A_Teacher_Imitation_Learning_Approach_for_Domain_Adaptation_with_Optimal_Transport)

3. **[최신 후속 연구]** *Unsupervised Domain Adaptation via Optimal Prototypes Transport*. **Expert Systems with Applications**, 2025. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417425003665)

4. **[OT-UDA 종합 분석]** *Optimal Transport-based Unsupervised Domain Adaptation* (Survey). [EmergentMind](https://www.emergentmind.com/topics/optimal-transport-based-unsupervised-domain-adaptation)

5. **[논문 발표 영상]** PaperTalk, CVPR 2020 Virtual Conference. [PaperTalk](https://papertalk.org/papertalks/15011)

> ⚠️ **주의:** 본 답변의 수식 중 SSR 비용 행렬 $Q = \lambda M + (1-\lambda)D$ 및 신뢰도 점수 $r(x_t)$의 구체적 형태는 논문의 공개 abstract와 부분적으로 공개된 PDF를 기반으로 재구성하였으며, 수식의 정확한 파라미터 표기는 원문 논문을 직접 확인하여 검증하시기를 권장드립니다.

# Reliable Weighted Optimal Transport for Unsupervised Domain Adaptation

### 1. 핵심 주장 및 주요 기여

**Reliable Weighted Optimal Transport (RWOT)**는 비지도 도메인 적응 문제를 해결하기 위한 혁신적인 방법으로, 다음 세 가지 핵심 기여를 제시합니다.[1]

첫째, **Shrinking Subspace Reliability (SSR)**를 도입하여 공간적 원형 정보(spatial prototypical information)와 도메인 내 구조를 활용해 표본 수준의 도메인 불일치를 동적으로 측정합니다. 이는 기존 최적 운송 방법이 간과했던 도메인 내 구조를 고려합니다.

둘째, **가중 최적 운송 전략**을 제안하여 정밀한 표본 단위 매칭을 달성합니다. 이는 결정 경계 근처의 표본들로 인한 음의 전이(negative transfer)를 감소시킵니다.

셋째, **판별적 중심 손실 함수**를 통해 깊은 판별 특징을 학습하여 클래스 내 밀집성(intra-class compactness)과 클래스 간 분리성(inter-class separability)을 동시에 달성합니다.[1]

***

### 2. 문제 정의, 제안 방법, 모델 구조 및 성능

#### 2.1 해결하고자 하는 문제

기존의 최적 운송 기반 도메인 적응 방법들은 다음과 같은 한계를 가집니다:[1]

- **조잡한 표본 쌍 매칭**: 각 표본을 전체로만 고려하여 표본 내 특성을 무시합니다
- **도메인 내 구조 미활용**: 클래스 중심으로부터 거리가 먼 표본들이 오분류됩니다
- **음의 전이**: 결정 경계 근처의 모호한 표본들로 인해 성능이 저하됩니다

#### 2.2 제안 방법 - 수식

**Shrinking Subspace Reliability (SSR)**

공간적 원형 정보 행렬 D는 다음과 같이 정의됩니다:[1]

$$
D(i, k) = \frac{e^{-d(G_f(x_i^t), c_k^s)}}{\sum_{m=1}^{C} e^{-d(G_f(x_i^t), c_m^s)}}
$$

여기서 $$d(G_f(x_i^t), c_k^s)$$는 다중 커널 공식으로 정의됩니다:[1]

$$
d(G_f(x_i^t), c_k^s) = K(c_k^s, c_k^s) - 2K(G_f(x_i^t), c_k^s) + K(G_f(x_i^t), G_f(x_i^t))
$$

다중 커널 K는 조합 계수 제약을 만족합니다:[1]

```math
K = \left\{ K = \sum_{u=1}^{m} \beta_u K_u : \sum_{u=1}^{m} \beta_u = 1, \beta_u \geq 0 \right\}
```

도메인 내 구조 정보를 나타내는 확률 주석 행렬 M은:[1]

$$
M(i, k) = P\left( y = k \left| \text{Softmax}\left( \frac{G_y(G_f(x_i^t))}{\tau} \right) \right. \right)
$$

SSR 비용 행렬 Q의 핵심 공식은:[1]

$$
Q(i, k) = \frac{d_A(k)D(i, k) + (2 - d_A(k))M(i, k)}{\sum_{m=1}^{C} \left[ d_A(m)D(i, m) + (2 - d_A(m))M(i, m) \right]}
$$

여기서 $$d_A(k)$$는 A-거리로, 선형 SVM 분류기의 오류 $$\epsilon(h_k)$$를 기반으로 합니다:[1]

$$
d_A(k)(D_k^s, D_k^t) = 2(1 - 2\epsilon(h_k))
$$

A-거리의 중요한 특성은 훈련 초기에는 D(i,k)에 더 가중치를 두고, 훈련이 진행되면서 M(i,k)로 점진적 전환합니다.

**가중 최적 운송**

가중 칸토로비치 문제의 이산 형식은:[1]

$$
\gamma^* = \arg \min_{\gamma \in \mathcal{X}(D_s, D_t)} \langle \gamma, Z \rangle_F = \arg \min_{\gamma \in \mathcal{X}(D_s, D_t)} \langle \gamma, R \cdot C \rangle_F
$$

적응적 비용 함수 행렬 Z는:[1]

$$
Z(i, j) = \left\| G_f(x_i^s) - G_f(x_j^t) \right\|^2 \cdot (1 - Q(j, y_i^s))
$$

가중 최적 운송 손실함수는:[1]

$$
\mathcal{L}_g = \sum_{i,j} \gamma_{i,j}^* \left( \left\| G_f(x_i^t) - G_f(x_j^s) \right\|^2 + F_1(\text{Softmax}(G_y(G_f(x_i^t))), y_j^s) \right)
$$

**판별적 중심 손실**

판별적 중심 손실 함수는:[1]

$$
\mathcal{L}_p = \sum_{i=1}^{n} \left\| G_f(x_i^s) - c_{y_i^s}^s \right\|_2^2 + \sum_{k=1}^{C} \sum_{i=1}^{n} Q(i,k) \left\| G_f(x_i^t) - c_k^s \right\|_2^2 + \lambda \sum_{\substack{k_1, k_2 = 1 \\ k_1 \neq k_2}}^{C} \max(0, \nu - \left\| c_{k_1}^s - c_{k_2}^s \right\|_2^2)
$$

여기서 클래스 중심은:[1]

$$
c_k^s = \frac{1}{S} \sum_{i=1}^{N_b} G_f(x_i^s) \varphi(y_i^s, k)
$$

**전체 목적함수**

최종 훈련 목표는:[1]

$$
\min_{G_y, G_f} \mathcal{L}_{cls} + \alpha \mathcal{L}_p + \beta \mathcal{L}_g
$$

여기서 $$\alpha$$와 $$\beta$$는 각각 판별적 중심 손실과 가중 최적 운송 전략의 기여도를 조절하는 하이퍼파라미터입니다.

#### 2.3 모델 구조

RWOT의 구조는 **두 스트림 시아미즈 CNN 아키텍처**를 기반으로 합니다:[1]

- **특징 생성기 $$G_f$$**: ResNet-50 백본 네트워크로 깊은 특징 표현 학습
- **적응형 분류기 $$G_y$$**: 소스 및 타겟 도메인의 공유 분류기
- **병목층(Bottleneck Layer)**: 의미론적 및 공간적 정보 인코딩

훈련 프로세스는 다음과 같습니다:[1]

1. 소스 데이터로부터 클래스 중심 계산
2. 공간적 원형 정보 행렬 D 계산
3. 소프트맥스 온도 조정을 통한 확률 주석 행렬 M 계산
4. SSR 비용 행렬 Q 업데이트
5. 세 가지 손실함수 계산 및 역전파

#### 2.4 성능 향상

**Office-31 데이터셋에서의 성능 비교**[1]

| 방법 | A→W | A→D | D→W | W→D | D→A | W→A | 평균 |
|------|-----|-----|-----|-----|-----|-----|------|
| ResNet | 70.0 | 65.5 | 96.1 | 99.3 | 62.8 | 60.5 | 75.7 |
| DANN | 81.5 | 74.3 | 97.1 | 99.6 | 65.5 | 63.2 | 80.2 |
| DeepJDOT | 88.9 | 88.2 | 98.5 | 99.6 | 72.1 | 70.1 | 86.2 |
| CDAN | 94.1 | 92.9 | 98.6 | 100.0 | 69.3 | 71.0 | 87.7 |
| **RWOT** | **95.1** | **94.5** | **99.5** | **100.0** | **77.5** | **77.9** | **90.8** |

**VisDA-2017 데이터셋에서의 성능**[1]

RWOT는 84.0%의 평균 정확도를 달성하였으며, DeepJDOT(77.4%)와 TPN(80.4%)을 크게 상회합니다. 특히 큰 도메인 갭으로 인해 다른 방법들이 일부 클래스에서 낮은 성능을 보일 때도 RWOT는 안정적인 성능을 유지합니다.

**Digits 데이터셋에서의 성능**[1]

| 방법 | SVHN→MNIST | MNIST→USPS | USPS→MNIST | 평균 |
|------|------------|-----------|-----------|------|
| DeepJDOT | 96.1±0.3 | 96.3±0.5 | 96.7±0.2 | 96.4 |
| **RWOT** | **98.8±0.1** | **98.5±0.2** | **97.5±0.2** | **98.3** |

#### 2.5 한계

논문의 주요 한계들은 다음과 같습니다:[1]

- **계산 복잡도**: A-거리 계산을 위한 선형 SVM 훈련으로 인한 추가 계산 비용
- **하이퍼파라미터 민감성**: α, β, τ, λ, ν 등 다양한 하이퍼파라미터의 튜닝 필요
- **클래스 중심 계산**: 이상적으로는 모든 표본을 사용해야 하지만 실무적으로는 배치 표본만 사용
- **오픈셋 도메인 적응 미지원**: 타겟 도메인에서 소스 도메인에 없는 클래스 처리 불가
- **제한된 모달리티**: 이미지 기반의 시각 작업에 중점으로 다른 모달리티 확장 제한

***

### 3. 일반화 성능 향상 관련 분석

#### 3.1 일반화 메커니즘

RWOT의 일반화 성능 향상은 다음과 같은 메커니즘에 기반합니다:[1]

**도메인 불일치 감소**

A-거리 분석 결과, RWOT는 최대 불일치(Max-A-Distance)와 평균 불일치(Avg-A-Distance) 모두에서 다른 방법들보다 훨씬 낮은 값을 달성합니다. 이는 소스와 타겟 도메인 간의 분포 갭을 효과적으로 감소시킵니다.[1]

**표현 가능성 향상**

t-SNE 시각화에서 RWOT는 정확히 31개의 클러스터(Office-31의 클래스 수)를 형성하며, 명확한 경계를 가진 특징 표현을 학습합니다. 반면 DeepJDOT는 클래스별 특징이 혼재되어 있습니다.[1]

**수렴 성능**

RWOT는 다른 방법들보다 빠른 수렴을 보이며, 수렴 과정에서 더 낮은 테스트 오류를 유지합니다. 이는 도메인 내 구조를 고려한 설계의 효과를 보여줍니다.[1]

#### 3.2 어려운 전이 작업에 대한 강건성

RWOT는 **하드 전이 작업(hard transfer tasks)**에서 특히 두드러진 성능 향상을 보입니다:[1]

- **A→D 작업**: 94.5% (DeepJDOT 88.2%에서 +6.3%)
- **D→A 작업**: 77.5% (DeepJDOT 72.1%에서 +5.4%)
- **SVHN→MNIST**: 98.8% (DeepJDOT 96.1%에서 +2.7%)

이러한 개선은 SSR이 결정 경계 근처의 어려운 표본들을 더 신뢰성 있게 처리하기 때문입니다.

#### 3.3 절제 연구(Ablation Study)

RWOT 변형들의 성능 분석을 통해 각 컴포넌트의 기여도를 확인할 수 있습니다:[1]

- **RWOT-M** (M만 사용): 기본 가중 최적 운송 성능 제공
- **RWOT-D** (D만 사용): 더 나은 성능 달성
- **RWOT-C** (판별적 중심 손실 제외): 기본 DeepJDOT 능력 향상
- **RWOT** (전체): 최고 성능 달성

이는 SSR의 동적 가중치 조정과 판별적 중심 손실의 상호작용이 일반화 성능을 크게 향상시킨다는 것을 입증합니다.[1]

***

### 4. 논문의 영향 및 향후 연구 고려사항

#### 4.1 향후 연구에 미치는 영향

**신뢰성 기반 적응 패러다임의 확산**

RWOT는 도메인 적응에서 표본 수준의 신뢰도를 고려한 새로운 관점을 제시했습니다. 최근 2024-2025년 연구들에서 이러한 패러다임이 확장되고 있습니다:[2][3]

- **다중 소스 도메인 적응**: 2025년 분배 강건 학습(Distributionally Robust Learning) 연구는 RWOT의 신뢰도 개념을 여러 소스 도메인에 확장하고 있습니다.[3]

- **소스 자유 도메인 적응**: 소스 데이터 접근이 불가능한 상황에서도 신뢰성 메커니즘을 적용하려는 시도가 진행 중입니다.[4]

**최적 운송 이론의 고도화**

RWOT의 가중 최적 운송 전략은 최적 운송 기반 도메인 적응의 발전을 주도하고 있습니다:[5]

- 저순위 제약을 가진 강건한 최적 운송(2024년 발표) - 노이즈가 있는 소스 데이터 처리
- 부분 최적 운송을 통한 오픈셋 도메인 적응 - RWOT의 개념을 미지 클래스 처리로 확장

**일반화 성능 향상 방향**

RWOT가 제시한 도메인 내 구조 활용과 표본 수준 신뢰도 개념은 다음과 같이 확장되고 있습니다:[6][7][8]

- **도메인 일반화(Domain Generalization)**: 타겟 데이터 접근 없이 여러 소스 도메인으로부터 일반화된 모델 학습
- **지속적 도메인 적응(Continual Domain Adaptation)**: 시간 경과에 따른 지속적인 도메인 변화에 적응
- **점 클라우드 도메인 적응**: RWOT의 원리를 3D 포인트 클라우드 분석에 적용 (2024년 ECCV 발표)

#### 4.2 향후 연구 시 고려할 점

**1. 계산 효율성 개선**

RWOT의 A-거리 계산 및 동적 가중치 조정은 계산 비용이 높습니다. 향후 연구는:[1]

- 더 효율적인 신뢰도 측정 방법 개발
- 근사 기법을 통한 계산 복잡도 감소
- 온라인 학습 방식으로의 전환

**2. 오픈셋 도메인 적응으로의 확장**

현재 RWOT는 폐쇄형 도메인 적응(closed-set)만 지원합니다:[9]

- 타겟 도메인의 미지 클래스 처리를 위한 신뢰도 메커니즘 확장
- 부분 최적 운송(Partial Optimal Transport) 개념과의 결합 필요

**3. 멀티모달 도메인 적응**

최근 비전-언어 모델의 발전과 함께:[8]

- 텍스트 임베딩을 활용한 도메인 불변 특징 학습
- 크로스 모달 신뢰도 평가 메커니즘 개발
- 언어 기반 도메인 설명을 통한 적응

**4. 강건성 향상**

노이즈가 있는 라벨이나 특징에 대한 대응:[5]

- 저순위 제약을 통한 부식된 부분공간 구조 복구
- 신뢰도 기반의 노이즈 필터링 메커니즘
- 대적 학습과의 결합

**5. 도메인 일반화와의 통합**

RWOT의 신뢰도 개념을 도메인 일반화에 적용:[6]

- 메타러닝을 통한 신뢰도 학습
- 다중 소스 도메인으로부터의 도메인 불변 신뢰도 추정
- 미지 도메인에 대한 신뢰도 외삽(extrapolation) 능력

**6. 이론적 분석 강화**

현재 RWOT는 경험적 성능 검증에 중점을 두고 있습니다:[10]

- 일반화 오류 한계(Generalization Error Bounds) 분석
- SSR과 최적 운송 전략의 이론적 수렴성 증명
- 신뢰도 기반 적응의 수렴 속도 분석

***

### 결론

**Reliable Weighted Optimal Transport**는 비지도 도메인 적응 분야에서 **표본 수준의 신뢰도를 고려한 혁신적 접근법**을 제시했습니다. 공간적 원형 정보와 도메인 내 구조를 동적으로 활용하는 SSR, 그리고 이를 기반으로 한 가중 최적 운송 전략은 특히 **어려운 전이 작업에서 음의 전이를 크게 감소**시켰습니다.

향후 연구는 이 신뢰도 패러다임을 **계산 효율성, 오픈셋 적응, 멀티모달 학습, 강건성, 도메인 일반화** 등으로 확장하는 방향으로 진행될 것으로 예상됩니다. 특히 최근 비전-언어 모델의 발전과 함께 텍스트 기반 신뢰도 평가, 다중 소스 강건 적응, 그리고 지속적 도메인 적응 시나리오에서의 응용이 주요 연구 방향이 될 것입니다.

***

**참고**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1fe1e2f0-ad24-4f06-8bba-29d53483bc7f/Xu_Reliable_Weighted_Optimal_Transport_for_Unsupervised_Domain_Adaptation_CVPR_2020_paper.pdf)
[2](https://arxiv.org/pdf/2208.07422.pdf)
[3](https://www.mdpi.com/1099-4300/27/4/426)
[4](https://arxiv.org/pdf/2309.02211.pdf)
[5](https://arxiv.org/pdf/2110.12024.pdf)
[6](https://arxiv.org/pdf/2303.15833.pdf)
[7](https://arxiv.org/html/2410.15811v2)
[8](https://arxiv.org/html/2502.06272v1)
[9](http://arxiv.org/pdf/2303.03770.pdf)
[10](https://openreview.net/forum?id=ewgLuvnEw6)
[11](https://bridges.monash.edu/articles/thesis/Optimal_Transport_Theory_for_Domain_Adaptation/25016954)
[12](https://www.meegle.com/en_us/topics/transfer-learning/transfer-learning-for-domain-generalization)
[13](https://github.com/xiaoyao3302/PCFEA)
[14](https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2024.124344)
[15](https://dgresearch.github.io)
[16](https://arxiv.org/abs/2507.07125)
[17](https://www.ijcai.org/proceedings/2020/0352.pdf)
[18](https://arxiv.org/abs/1812.11806)
[19](https://www.sciencedirect.com/science/article/pii/S1569843225001268)
