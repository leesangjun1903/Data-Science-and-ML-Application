# Self-adaptive Re-weighted Adversarial Domain Adaptation (SRDA)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
기존 적대적 도메인 적응(Adversarial Domain Adaptation) 방법들은 **주변 분포(marginal distribution)만을 고려**하여 과소 전이(under transfer) 또는 부정 전이(negative transfer)를 초래한다. 이 논문은 **조건부 분포(conditional distribution) 관점**에서 도메인 정렬을 강화하는 자기 적응적 재가중치(self-adaptive re-weighted) 방법을 제안한다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **엔트로피 기반 재가중치** | 조건부 엔트로피를 활용하여 잘 정렬된 샘플의 가중치는 낮추고, 부족하게 정렬된 샘플의 가중치는 높임 |
| **클래스 수준 정렬** | 소스 샘플 + 의사 레이블(pseudo-label) 타겟 샘플을 이용한 Triplet Loss 적용 |
| **공동 훈련(co-training)** | 도메인 수준 + 클래스 수준 정렬을 동시에 수행하여 도메인 불변(domain-invariant)이면서 클래스 판별적(class-discriminative)인 표현 학습 |
| **이론적 보장** | Ben-David 정리를 기반으로 이상적인 소스-타겟 가설의 결합 오차가 낮음을 보임 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)** 에서:

1. **주변 분포만 고려**: 기존 DANN, CDAN 등의 방법이 $P(X_s)$와 $P(X_t)$의 차이만 줄이고, $P(Y|X)$의 차이(조건부 분포)를 무시
2. **과소 전이(under transfer)**: 도메인 정렬이 충분히 이루어지지 않아 타겟 도메인에서 분류 성능 저하
3. **부정 전이(negative transfer)**: 잘못된 샘플을 과도하게 정렬하여 성능이 오히려 하락
4. **의사 레이블의 부정확성**: 도메인 편향(domain bias)으로 인해 의사 레이블이 항상 정확하지 않음

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) 기본 도메인 적대적 네트워크 (Baseline)

소스 분류 손실:

$$\mathcal{L}^s_{task}(\theta_f, \theta_y) = \frac{1}{n_s} \sum_{x_i \in \mathcal{D}_s} \mathcal{L}_y(G_y(f(x_i)), y^s_i) \tag{1a}$$

도메인 판별 손실:

$$\mathcal{L}_D(\theta_f, \theta_y, \theta_d) = -\frac{1}{n_s + n_t} \sum_{x_i \in (\mathcal{D}_s \cup \mathcal{D}_t)} \mathcal{L}_d(G_d(f(x_i)), d_i) \tag{1b}$$

여기서 $\theta_f$는 특징 추출기, $\theta_d$는 도메인 판별기, $\theta_y$는 소스 분류기의 파라미터이며 $d_i$는 샘플 $x_i$의 도메인 레이블.

#### (2) 자기 적응적 재가중치 적대적 손실 (핵심 기여)

조건부 엔트로피를 활용한 재가중치 적대적 손실:

$$\mathcal{L}_{adv}(\theta_f, \theta_d) = -\frac{1}{n_s + n_t} \sum_{x_i \in (\mathcal{D}_s \cup \mathcal{D}_t)} (1 + \mathcal{H}_p)\mathcal{L}_d(G_d(f(x_i)), d_i) \tag{2}$$

$$\text{where} \quad \mathcal{H}_p = -\frac{1}{C} \sum^C_{c=1} p_c \log(p_c)$$

- $C$: 클래스 수
- $p_c$: 샘플이 클래스 $c$로 예측될 확률
- $\mathcal{H}_p$가 낮을수록 → **잘 정렬된 샘플** → 가중치 감소
- $\mathcal{H}_p$가 높을수록 → **잘못 정렬된 샘플** → 가중치 증가

**직관적 해석**: $(1 + \mathcal{H}_p)$ 항이 불확실성이 높은(poorly aligned) 샘플에 더 강한 적대적 힘을 부여한다.

#### (3) 엔트로피 최소화 손실

타겟 샘플의 판별력 향상:

$$\mathcal{L}_h(\theta_f, \theta_y) = -\frac{1}{n_t} \sum_{x_i \in \mathcal{D}_t} p_t \log(p_t) \tag{3}$$

#### (4) 클래스 수준 정렬: Triplet Loss

혼동 도메인(confusing domain)에서 앵커 $x_a$, 포지티브 $x_p$, 네거티브 $x_n$에 대해:

$$\mathcal{L}_{tri}(\theta_f) = \sum_{\substack{x_i \in (\mathcal{D}_s \cup \mathcal{D}_t) \\ y_a = y_p, y_n}} [m + d_{a,p} - d_{a,n}]_+ \tag{4}$$

- $m$: 마진 (실험에서 0.3으로 설정)
- $d_{a,p}$: 앵커-포지티브 간 거리
- $d_{a,n}$: 앵커-네거티브 간 거리
- $[\cdot]_+$: hinge function

#### (5) 전체 학습 손실

$$\mathcal{L} = \mathcal{L}^s_{task} + \mathcal{L}_{adv} + \mathcal{L}_h + \mathcal{L}_{tri} \tag{5}$$

#### (6) 최적화 목표

$$(\hat{\theta}_f, \hat{\theta}_y) = \arg\min_{\theta_f, \theta_y} \mathcal{L}(\theta_f, \theta_y, \theta_d)$$

$$\hat{\theta}_d = \arg\max_{\theta_d} \mathcal{L}(\theta_f, \theta_y, \theta_d) \tag{6}$$

---

### 2.3 모델 구조

```
소스 데이터 (X_s) ──────────────────────────────────────────────────────────────────┐
                    ↓                                                               │
타겟 데이터 (X_t) → [특징 추출기 f(·)] → [가중치 계산 (엔트로피 기반)] → [도메인 판별기 G_d(·)]
                         ↓                                                         │
                  [소스 분류기 G_y(·)] → 소스 분류 손실 + 엔트로피 최소화 손실       │
                         ↓                                                         │
                  의사 레이블 생성 (threshold T=0.9 이상만 선택)                    │
                         ↓                                                         │
                  [Triplet Loss] (소스 샘플 + 의사 레이블 타겟 샘플 쌍 구성)        │
```

**백본**: ResNet-50 (사전학습), 손글씨 데이터셋은 LeNet  
**의사 레이블 선택 임계값**: $T = 0.9$ (고신뢰도 샘플만 선택)  
**Triplet Loss 마진**: $m = 0.3$

---

### 2.4 성능 향상

#### Office-31 데이터셋 (ResNet-50)

| 방법 | A→W | D→W | W→D | A→D | D→A | W→A | **Avg.** |
|------|-----|-----|-----|-----|-----|-----|----------|
| DANN | 82.0 | 96.9 | 99.1 | 79.7 | 68.2 | 67.4 | 82.2 |
| CDAN | 94.1 | 98.6 | 100.0 | 92.9 | 71.0 | 69.3 | 87.7 |
| TADA | 94.3 | 98.7 | 99.8 | 91.6 | 72.9 | 73.0 | 88.4 |
| SymNet | 90.8 | 98.8 | 100.0 | 93.9 | 74.6 | 72.5 | 88.4 |
| **Ours** | **95.2** | 98.6 | **100.0** | 91.7 | **74.5** | **73.7** | **89.0** |

#### Office-Home 데이터셋 (ResNet-50)

| 방법 | Avg. |
|------|------|
| CDAN | 65.8 |
| SAFN | 67.3 |
| TADA | 67.6 |
| SymNet | 67.6 |
| **Ours** | **68.9** |

#### Ablation Study (Office-31)

| 변형 | A→W | W→D | A→D | W→A | Avg. |
|------|-----|-----|-----|-----|------|
| ResNet-50 | 68.4 | 99.3 | 68.9 | 60.7 | 74.3 |
| DANN | 82.0 | 99.1 | 79.7 | 67.4 | 82.1 |
| DANN (Em) | 89.8 | 100.0 | 90.1 | 69.0 | 87.2 |
| DANN (Em+ $\mathcal{H}_p$ ) | 92.3 | 100.0 | 91.1 | 71.9 | 88.8 |
| DANN (Em+ $\mathcal{L}_{tri}$ ) | 93.8 | 99.8 | 91.7 | 72.6 | 89.5 |
| **Ours** | **95.2** | **100.0** | **91.7** | **73.7** | **90.2** |

---

### 2.5 한계점

1. **고정 임계값(T=0.9)**: 의사 레이블 선택 임계값이 데이터셋에 관계없이 고정되어 있어 유연성 부족
2. **Triplet Loss의 쌍 구성 비용**: 소스-타겟 쌍 구성이 계산 비용을 증가시킴
3. **극단적 도메인 차이에 취약**: 도메인 간 격차가 매우 큰 경우 의사 레이블의 정확도가 낮아질 수 있음
4. **단순 엔트로피 정규화**: $(1 + \mathcal{H}_p)$ 형태의 선형 재가중치는 비선형적 전이 특성을 완전히 반영하지 못할 수 있음
5. **멀티소스 도메인 미지원**: 단일 소스-타겟 쌍에만 적용 가능

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Ben-David 정리 기반 이론적 분석

Ben-David 정리에 따르면, 타겟 도메인의 기대 오차는 다음과 같이 상한이 정해진다:

$$\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2} d_{\mathcal{H} \Delta \mathcal{H}}(D_S, D_T) + \lambda^* \tag{*}$$

여기서 $\lambda^* = \min_{h \in \mathcal{H}}[\epsilon_S(h) + \epsilon_T(h)]$는 **이상적인 소스-타겟 가설의 결합 오차**.

SRDA는 두 가지 방향으로 이 상한을 줄인다:
- **$d_{\mathcal{H}\Delta\mathcal{H}}$ 감소**: 재가중치 적대적 훈련으로 도메인 분포 차이 감소
- **$\lambda^*$ 감소**: Triplet Loss를 통한 클래스 수준 정렬로 결합 오차 최소화

### 3.2 일반화 성능 향상 메커니즘

#### A. 엔트로피 기반 재가중치의 일반화 기여
- **잘 정렬된 샘플(낮은 엔트로피)**: 가중치 감소 → 이미 효과적으로 전이된 특징에 불필요한 압력 방지
- **부족하게 정렬된 샘플(높은 엔트로피)**: 가중치 증가 → 어려운 전이 케이스에 집중적 학습

이는 자원 할당의 효율성을 높여 **다양한 도메인 변화에 강건한 특징**을 학습하게 한다.

#### B. 공동 분포 (Joint Distribution) 정렬
기존 방법: $P(X_s) \approx P(X_t)$ (주변 분포만)

SRDA: $P(X_s, Y_s) \approx P(X_t, Y_t)$ (공동 분포)

공동 분포 정렬을 통해 **클래스 구조가 도메인 간 일관**되게 유지되어, 새로운 타겟 도메인에 대한 일반화 능력이 향상된다.

#### C. 클래스 판별적 특징 학습 (Triplet Loss)
- **Intra-class compactness**: 같은 클래스 샘플들을 특징 공간에서 가깝게 배치
- **Inter-class separability**: 다른 클래스 샘플들을 멀리 배치

이를 통해 학습된 특징은 소스-타겟 도메인 양쪽에서 **클래스 경계를 더 명확히 형성**하여, 미지의 타겟 샘플에 대한 일반화 성능이 향상된다.

#### D. A-distance 분석 (정량적 일반화 지표)

$$d_{\mathcal{A}} = 2(1 - 2\epsilon)$$

실험에서 A→W 태스크의 A-distance:
- ResNet: 높음 (약 1.7)
- DANN: 중간 (약 1.3)
- **Ours: 낮음 (약 0.9)**

A-distance가 낮을수록 도메인 분포가 가까워져 일반화 성능이 향상됨이 정량적으로 확인된다.

#### E. t-SNE 시각화를 통한 일반화 근거
논문의 Figure 5에서 확인된 바:
- ResNet: 소스-타겟 도메인 특징이 분리됨
- DANN: 도메인 수준 정렬 개선되었으나 클래스 경계 불명확
- **Ours**: 도메인 간 정렬 + 클래스 내 응집성 + 클래스 간 분리성 모두 개선

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

#### A. 재가중치 패러다임의 확장 가능성
SRDA의 **엔트로피 기반 재가중치** 아이디어는 다음 연구 방향에 영향을 미쳤다:
- **부분 도메인 적응(Partial DA)**: 소스-타겟 간 클래스 집합이 다른 경우에도 전이 가능성을 측정하는 재가중치 메커니즘 적용
- **개방 집합 도메인 적응(Open-set DA)**: 미지 클래스 샘플에 낮은 가중치 부여로 부정 전이 방지

#### B. 의사 레이블 + 메트릭 러닝 결합 패러다임
Triplet Loss와 의사 레이블의 결합 전략은 이후 **자기 지도 학습(self-supervised learning)** 기반 도메인 적응 연구의 선구적 접근법으로 기능할 수 있다.

#### C. 이론적 기여의 파급력
Ben-David 정리와의 연결을 통해 실용적 방법론에 **이론적 보장**을 제공하는 연구 방향성을 제시한다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의**: 아래 비교는 논문 원문에 언급된 방법들과 일반적으로 알려진 2020년 이후 연구 트렌드를 기반으로 작성하였으며, 개별 수치의 정확성은 해당 논문을 직접 확인하시기 바랍니다.

#### 2020년 이후 주요 연구 트렌드와 비교

| 연구 방향 | SRDA와의 관계 | 한계/차이점 |
|-----------|---------------|-------------|
| **Transformer 기반 DA** (TVT, CDTrans 등) | SRDA의 적대적 구조와 달리 attention 메커니즘 활용 | SRDA는 CNN 백본 의존적 |
| **자기 지도 학습 기반 DA** (MCC, SHOT 등) | SRDA의 엔트로피 최소화와 유사한 방향 | SRDA는 소스 데이터 필요 |
| **소스 프리(Source-free) DA** | 소스 데이터 없이 적응 | SRDA는 소스 데이터 접근 필요 |
| **프롬프트 기반 DA** (CLIP 활용) | 대규모 사전학습 모델 활용 | SRDA는 소규모 ResNet-50 |

**특히 주목할 2020년 이후 연구들:**

- **SHOT (ICML 2020)**: 소스-프리 도메인 적응으로, 의사 레이블 + 정보 최대화를 결합 → SRDA의 엔트로피 활용 아이디어와 연결됨
- **MCC (ECCV 2020)**: 클래스 혼동 행렬을 활용한 조건부 분포 정렬 → SRDA의 조건부 분포 관점과 유사

### 4.3 앞으로 연구 시 고려할 점

#### A. 동적 임계값 메커니즘
현재 고정된 의사 레이블 임계값($T=0.9$)을 **학습 과정에서 동적으로 조절**하는 커리큘럼 학습 전략 도입 필요:

$$T(t) = T_0 + (T_{max} - T_0) \cdot \frac{t}{T_{total}}$$

#### B. 더 정교한 재가중치 함수 설계
선형적 엔트로피 가중치 $(1 + \mathcal{H}_p)$를 넘어서 **비선형 적응적 가중치** 탐구:

$$\omega(x) = \sigma(\alpha \cdot \mathcal{H}_p(x) + \beta)$$

여기서 $\alpha, \beta$는 학습 가능한 파라미터.

#### C. 멀티소스 도메인 적응으로의 확장
단일 소스에서 다중 소스(Multi-source DA)로 확장 시, 각 소스 도메인별로 다른 재가중치 적용:

$$\mathcal{L}_{adv} = -\sum_{k=1}^{K} \frac{1}{n_k + n_t} \sum_{x_i} \omega_k(x_i) \mathcal{L}_d(G_d(f(x_i)), d_i)$$

#### D. 대규모 사전학습 모델과의 통합
ViT, CLIP 등 대규모 사전학습 모델에 SRDA의 재가중치 메커니즘을 통합하여 **제로샷/퓨샷 도메인 적응** 성능 향상 탐구.

#### E. 계산 효율성 개선
Triplet Loss의 쌍 구성 비용을 줄이기 위한 **효율적 샘플링 전략** (hard negative mining, proxy-based metric learning 등) 도입.

#### F. 공정성(Fairness) 관점 도입
재가중치 메커니즘이 특정 하위 그룹에 편향될 수 있으므로, **공정한 도메인 적응** 관점에서의 재가중치 설계 필요.

---

## 참고자료

- **Wang, S., & Zhang, L. (2020).** Self-adaptive Re-weighted Adversarial Domain Adaptation. *arXiv:2006.00223v2*
- **Ben-David, S., et al. (2010).** A theory of learning from different domains. *Machine Learning, 79*
- **Ganin, Y., et al. (2016).** Domain-adversarial training of neural networks. *JMLR*
- **Long, M., et al. (2018).** Conditional adversarial domain adaptation. *NeurIPS*
- **Saito, K., et al. (2018).** Maximum classifier discrepancy for unsupervised domain adaptation. *CVPR*
- **Pan, S. J., & Yang, Q. (2010).** A survey on transfer learning. *IEEE TKDE*
- **Schroff, F., et al. (2015).** FaceNet: A unified embedding for face recognition and clustering. *CVPR*
- **Xu, R., et al. (2019).** Larger norm more transferable: An adaptive feature norm approach. *ICCV*
- **Zhang, Y., et al. (2019).** Domain-symmetric networks for adversarial domain adaptation. *CVPR*
- **Wang, X., et al. (2019).** Transferable attention for domain adaptation. *AAAI*
