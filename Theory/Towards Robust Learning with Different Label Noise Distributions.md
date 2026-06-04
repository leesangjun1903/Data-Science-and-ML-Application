
# Towards Robust Learning with Different Label Noise Distributions

> **논문 정보**
> - **저자**: Diego Ortego, Eric Arazo, Paul Albert, Noel E. O'Connor, Kevin McGuinness (Dublin City University, Insight Centre for Data Analytics)
> - **arXiv**: 1912.08741 (2019년 12월 제출, 2020년 7월 최종 업데이트)
> - **게재**: IEEE (ICPR 2020)
> - **코드**: https://git.io/JJ0PV

---

## 1. 🔑 핵심 주장과 주요 기여 요약

### 핵심 주장

노이즈 레이블은 레이블링 과정에서 불가피하게 발생하며 CNN의 성능 저하를 막기 위해 이를 탐지하는 것이 중요하다. 노이즈 레이블을 폐기하면 유해한 암기(memorization)를 피할 수 있으며, 해당 이미지 콘텐츠는 반지도 학습(SSL) 설정에서 여전히 활용 가능하다.

클린 샘플은 보통 **small loss trick**(손실이 낮다는 특성)으로 식별되지만, **다양한 노이즈 분포에 따라 이 트릭의 적용이 복잡해짐**을 보이고, 모든 이미지를 지속적으로 재레이블링(relabeling)하여 다중 분포에 대해 판별력 있는 손실(discriminative loss)을 드러내는 방법을 제안한다.

### 주요 기여 (4가지)


1. **다양한 노이즈 분포를 연구할 수 있는 프레임워크** 제공
2. **노이즈 분포에 독립적인(agnostic) 레이블 노이즈 탐지 방법** — 기존 방법 대비 실질적 성능 향상
3. **표현 학습에서의 레이블 노이즈 암기 효과 연구** — 중간 특징(intermediate features)의 판별력이 영향받지 않음을 보임
4. **DRPL(Distribution Robust Pseudo-Labeling)** — 다양한 노이즈 분포 및 실세계 노이즈에서 우수한 성능 입증


---

## 2. 🧩 문제 정의 · 제안 방법 · 모델 구조 · 성능 · 한계

### 2-1. 해결하고자 하는 문제

클린 샘플을 식별하는 small loss trick은 "낮은 손실 = 클린 샘플"이라는 가정에 기반하지만, **다양한 노이즈 분포에서는 이 트릭의 적용이 단순하지 않다.**

구체적으로는 아래 세 가지 노이즈 분포 유형을 고려한다:

| 노이즈 유형 | 설명 |
|---|---|
| **Symmetric (Uniform) Noise** | 모든 클래스로 균등하게 잘못 레이블링 |
| **Asymmetric (Structured) Noise** | 유사한 클래스 간의 체계적 오류 |
| **Out-of-Distribution (OOD) Noise** | 완전히 다른 분포의 이미지가 삽입됨 |

특히 **비균일 OOD 노이즈(non-uniform out-of-distribution noise)**가 실세계 노이즈와 가장 유사하다는 것을 ImageNet32/64 기반 실험으로 밝혔다.

---

### 2-2. 제안 방법: DRPL (Distribution Robust Pseudo-Labeling)

#### ① 연속적 재레이블링 (Continuous Relabeling)

Small loss trick의 핵심 문제는 **OOD 노이즈 샘플이 낮은 손실을 가져 클린 샘플로 오탐지**된다는 점이다. 이를 해결하기 위해 훈련 도중 모든 샘플을 지속적으로 재레이블링하여 **노이즈 분포에 독립적인 손실 분포**를 형성한다.

재레이블링은 현재 모델의 예측 확률을 이용하며, 샘플 $x_i$의 pseudo-label $\tilde{y}_i$는 다음과 같이 정의된다:

$$\tilde{y}_i = \text{argmax}_{c} \, p(y = c \mid x_i; \theta)$$

또는 소프트 레이블(soft label)로 표현하면:

$$\tilde{p}_i = \text{sharpen}\left(\frac{1}{K}\sum_{k=1}^{K} p(y \mid \text{Aug}_k(x_i); \theta)\right)$$

여기서 $K$는 augmentation 횟수, $\text{sharpen}(\cdot)$은 온도 파라미터 $T$를 이용한 sharpening 함수:

$$\text{sharpen}(p, T)_c = \frac{p_c^{1/T}}{\sum_{c'} p_{c'}^{1/T}}$$

---

#### ② 클린-노이즈 분리: GMM 기반 탐지 (1단계 SSL)

재레이블링 후 손실 분포에 **Gaussian Mixture Model (GMM)**을 적합(fit)시켜 클린/노이즈 샘플을 분리한다. 각 샘플 $x_i$의 클린 확률 $w_i$는 다음과 같이 추정된다:

$$w_i = p\left(g = \text{clean} \mid \ell_i; \phi_\text{GMM}\right)$$

GMM의 2-성분 혼합 모델:

$$p(\ell) = \pi_{\text{clean}} \cdot \mathcal{N}(\ell \mid \mu_{\text{clean}}, \sigma_{\text{clean}}^2) + \pi_{\text{noisy}} \cdot \mathcal{N}(\ell \mid \mu_{\text{noisy}}, \sigma_{\text{noisy}}^2)$$

- $\ell_i$: 재레이블링 이후 샘플 $x_i$의 cross-entropy loss
- $\pi_{\text{clean}}, \pi_{\text{noisy}}$: 각 성분의 혼합 가중치
- $w_i \geq \tau$ 이면 클린, 그 미만이면 노이즈로 분류

---

#### ③ 두 번의 SSL 적용 (Two-stage SSL)

SSL은 **두 번 적용**된다: 첫 번째는 클린-노이즈 탐지 성능 향상을 위해, 두 번째는 최종 모델 훈련을 위해 사용된다.

**전체 학습 손실 함수:**

클린 샘플 집합 $\mathcal{X}$과 노이즈 샘플 집합 $\mathcal{U}$에 대해 MixMatch 기반 손실을 적용한다:

$$\mathcal{L} = \mathcal{L}_{\mathcal{X}} + \lambda_u \cdot \mathcal{L}_{\mathcal{U}}$$

$$\mathcal{L}_{\mathcal{X}} = \frac{1}{|\mathcal{X}|} \sum_{(x_i, \tilde{y}_i) \in \mathcal{X}} H(\tilde{y}_i, p(y \mid x_i; \theta))$$

$$\mathcal{L}_{\mathcal{U}} = \frac{1}{|\mathcal{U}|} \sum_{x_j \in \mathcal{U}} \left\| \hat{q}_j - p(y \mid x_j; \theta) \right\|_2^2$$

여기서:
- $H(\cdot, \cdot)$: cross-entropy loss
- $\hat{q}_j$: 노이즈 샘플에 대한 pseudo-label (모델 예측 기반)
- $\lambda_u$: 비지도 손실 가중치 (하이퍼파라미터)

---

### 2-3. 모델 구조

전체 DRPL 파이프라인은 다음과 같이 구성된다:

```
[초기 CNN 훈련]
       ↓
[연속적 재레이블링 (모델 예측 기반 pseudo-label 생성)]
       ↓
[GMM으로 손실 분포 피팅 → 클린/노이즈 분리]
       ↓
[1차 SSL: 클린-노이즈 탐지 정밀도 향상]
       ↓
[개선된 클린/노이즈 분리 재수행]
       ↓
[2차 SSL (MixMatch 기반): 최종 모델 훈련]
       ↓
[최종 분류 모델]
```

- 백본(Backbone): **PreAct ResNet-18** (CIFAR), **ResNet-50** (ImageNet32/64, WebVision)
- Semi-supervised 방법: **MixMatch** (Berthelot et al., 2019) 기반

---

### 2-4. 성능 향상

CIFAR-10/100, ImageNet32/64 및 WebVision(실세계 노이즈) 실험에서 DRPL이 당시 최신 기법(state-of-the-art) 대비 **실질적인 성능 향상**을 보였다.

SSL이 오라클(oracle) 사용 시 다른 대안들을 능가하며, 5개 데이터셋에 걸쳐 DRPL의 실질적 성능 향상을 입증하였다.

특히 OOD 노이즈 시나리오에서 기존 small loss trick 기반 방법들의 실패 사례를 극복하는 데 효과적이었다.

---

### 2-5. 한계점

1. **계산 비용**: SSL을 두 번 적용하는 2단계 파이프라인은 단일 훈련 대비 높은 연산 비용 발생
2. **하이퍼파라미터 민감성**: GMM 임계값 $\tau$, MixMatch의 $\lambda_u$, sharpening 온도 $T$ 등 다수의 하이퍼파라미터 튜닝 필요
3. **노이즈율이 매우 높은 경우(>80%)**: GMM의 클린-노이즈 분리가 불안정해질 수 있음
4. **인스턴스 의존적 노이즈(instance-dependent noise)** 시나리오에 대한 검증 부족

---

## 3. 🚀 모델의 일반화 성능 향상 가능성

### (1) 표현 학습의 견고성

핵심 발견 중 하나는 **중간 특징(intermediate features)이 레이블 노이즈 손상에 대부분 영향받지 않는다**는 것이다. 이는 다음을 의미한다:

- 노이즈 레이블이 있더라도 CNN의 중간 레이어는 유용한 시각 표현을 학습함
- 따라서 노이즈 탐지 → 반지도 학습 파이프라인이 **표현 품질을 유지하면서 분류 성능을 향상**시킬 수 있음

### (2) 연속적 재레이블링의 일반화 기여

연속적 재레이블링은 다음 방식으로 일반화 성능에 기여한다:

$$\tilde{y}_i^{(t)} \leftarrow p(y \mid x_i; \theta^{(t)}) \quad \text{(에폭 } t \text{마다 갱신)}$$

- **동적 레이블 교정**: 훈련이 진행될수록 모델이 더 정확한 pseudo-label을 생성 → 점진적 품질 향상
- **과적합 방지**: 노이즈 레이블 암기 대신 실제 데이터 분포에 맞는 레이블로 수정

### (3) 비지도 데이터 활용을 통한 일반화

노이즈 레이블을 폐기함으로써 유해한 암기를 방지하고, 관련 이미지 콘텐츠는 반지도 학습 설정에서 계속 활용될 수 있다. 이를 통해:

- 노이즈 샘플을 **완전히 버리지 않고** unlabeled data로 재활용
- MixMatch의 consistency regularization으로 **결정 경계의 평탄화(flat decision boundary)** 유도 → 일반화 향상

### (4) OOD 노이즈 처리 능력

비균일 OOD 노이즈가 실세계 노이즈를 더 잘 모사한다는 발견은, 실제 웹 크롤링 데이터나 크라우드소싱 데이터에서 수집된 훈련셋에 직접 적용 가능한 일반화 가능성을 보여준다.

---

## 4. 🔭 미래 연구에 미치는 영향 및 고려 사항

### 4-1. 앞으로의 연구에 미치는 영향

| 영향 | 설명 |
|---|---|
| **노이즈 분포 다양성 인식** | 단일 noise type 가정 탈피 → 혼합 노이즈 시나리오 연구 촉진 |
| **SSL + 노이즈 학습 통합** | 반지도 학습 기반 노이즈 처리의 표준 파이프라인 정립 |
| **재레이블링 패러다임** | 정적 레이블 대신 동적 pseudo-label 갱신 전략의 주류화 |
| **표현 학습 분석** | 중간 특징의 노이즈 견고성 → 사전학습 모델 활용 연구로 확장 |

### 4-2. 2020년 이후 관련 연구 비교 분석

#### 🔹 DivideMix (Li et al., ICLR 2020)

DivideMix는 두 개의 분기 네트워크를 동시에 훈련하고, 각 네트워크가 GMM을 이용해 다른 네트워크를 위한 클린/노이즈 분리를 수행하는 **co-divide** 전략을 통해 확증 편향(confirmation bias)을 방지한다.

DivideMix는 CIFAR-10, CIFAR-100, Clothing1M, WebVision 등 모든 벤치마크에서 일관되게 최신 기법을 능가하며, CIFAR-100의 80~90% 대칭 노이즈에서 차순위 대비 약 10% 향상을 달성했다.

> **DRPL vs DivideMix**: DRPL이 다양한 노이즈 분포 연구 프레임워크와 재레이블링 전략을 제시했다면, DivideMix는 co-training 기반 확증 편향 방지에 집중하며 더 높은 성능을 달성했다.

#### 🔹 PLReMix (2024)

2차원 GMM을 통해 의미 정보와 모델 출력을 동시에 활용하여 클린/노이즈 샘플을 구분하고, PLR 손실과 반지도 손실을 동시에 적용한다. 이는 DRPL의 1차원 GMM 기반 접근을 고차원으로 확장한 발전으로 볼 수 있다.

#### 🔹 주요 후속 연구 트렌드 (2020~2024)

이후 연구들은 위상학적 필터(topological filter), 인스턴스 의존적 노이즈(instance-dependent noise), 샘플 선별(sample selection), 정규화 손실(normalized loss functions) 등 다양한 방향으로 발전하였다.

| 연구 | 핵심 기여 | DRPL과의 차별점 |
|---|---|---|
| **DivideMix** (ICLR 2020) | Co-training + GMM + MixMatch | 이중 네트워크로 확증 편향 방지 |
| **ELR** (NeurIPS 2020) | Early Learning Regularization | 암기 방지를 위한 정규화 항 추가 |
| **C2D** (CVPR 2021) | 대조 학습 + DivideMix 초기화 | 자기지도 사전학습으로 표현 향상 |
| **SOP** (ICML 2022) | 노이즈 레이블을 최적화 변수로 처리 | 레이블 자체를 학습 가능 파라미터화 |
| **PLReMix** (arXiv 2024) | 대조 표현 + 2D GMM | 의미론적 클러스터 정보 통합 |

---

### 4-3. 앞으로 연구 시 고려해야 할 점

1. **인스턴스 의존적 노이즈(Instance-Dependent Noise) 처리**
   - DRPL은 클래스 조건부 노이즈를 주로 다루나, 실세계에서는 샘플별로 다른 노이즈율이 발생함
   - 향후 연구는 샘플의 특성에 따라 동적으로 변하는 노이즈 전이 행렬 추정이 필요

2. **대규모 데이터셋 확장성**
   - 2단계 SSL 파이프라인의 계산 복잡도를 줄이는 효율적인 구현 필요 (예: 경량화된 GMM, 배치 단위 갱신)

3. **자기지도/대조 학습 통합**
   - 일부 접근법들은 자기지도 사전학습된 표현 인코더를 대조 학습을 통해 활용하려 시도한다. DRPL의 파이프라인과 대조 학습을 결합하면 표현 품질과 노이즈 탐지 정확도를 동시에 향상시킬 수 있음

4. **레이블 노이즈 + 클래스 불균형 동시 처리**
   - 대부분의 기존 노이즈 강건 방법들은 클래스 불균형을 효과적으로 처리하지 못한다. 현실 데이터에서는 노이즈와 불균형이 동시에 존재하는 경우가 많으므로, 두 문제를 통합적으로 다루는 방법론 개발이 필요

5. **하이퍼파라미터 자동화**
   - GMM 임계값 $\tau$, sharpening 온도 $T$ 등의 자동 탐색(AutoML/NAS 기반) 전략 도입

6. **멀티모달 및 NLP 도메인 확장**
   - 텍스트 데이터의 특징은 더 복잡하고 동적이며, 노이즈 레이블은 NLP 모델의 의미 포착 능력을 저해하여 예측 시 정확도와 일반화에 영향을 미친다. 이미지 도메인에서 검증된 DRPL 아이디어를 NLP로 확장하는 연구가 필요

---

## 📚 참고 자료 및 출처

| # | 제목 / 출처 | URL |
|---|---|---|
| 1 | **Towards Robust Learning with Different Label Noise Distributions** (Ortego et al., 2019/2020) — arXiv | https://arxiv.org/abs/1912.08741 |
| 2 | 동 논문 PDF (arXiv) | https://arxiv.org/pdf/1912.08741 |
| 3 | IEEE Xplore 게재 버전 | https://ieeexplore.ieee.org/document/9412747 |
| 4 | GitHub 공식 코드 (DiegoOrtego/LabelNoiseDRPL) | https://github.com/DiegoOrtego/LabelNoiseDRPL |
| 5 | **DivideMix: Learning with Noisy Labels as Semi-supervised Learning** (Li et al., ICLR 2020) | https://arxiv.org/pdf/2002.07394 |
| 6 | DivideMix OpenReview | https://openreview.net/forum?id=HJgExaVtwr |
| 7 | **PLReMix: Combating Noisy Labels with Pseudo-Label Relaxed Contrastive Representation Learning** (2024) | https://arxiv.org/abs/2402.17589 |
| 8 | **A Survey of Label-noise Representation Learning** (Han et al., 2020) | https://www.researchgate.net/publication/345654678 |
| 9 | **Awesome-Noisy-Labels** (GitHub Survey, songhwanjun) | https://github.com/songhwanjun/Awesome-Noisy-Labels |
| 10 | **Advances-in-Label-Noise-Learning** (GitHub, weijiaheng) | https://github.com/weijiaheng/Advances-in-Label-Noise-Learning |
| 11 | **Mitigating Memorization of Noisy Labels by Clipping the Model Prediction** (LogitClip, ICML 2023) | https://proceedings.mlr.press/v202/wei23e/wei23e.pdf |
| 12 | **A survey of label-noise deep learning for medical image analysis** (ScienceDirect, 2024) | https://www.sciencedirect.com/science/article/abs/pii/S1361841524000914 |
| 13 | **AlphaXiv 논문 페이지** | https://www.alphaxiv.org/resources/1912.08741v3 |

> ⚠️ **정확도 관련 안내**: 본 논문의 구체적 수치(정확한 성능 수치, 세부 수식의 일부 파라미터 정의)는 공개된 arXiv 버전 및 IEEE 버전을 기반으로 작성되었습니다. 논문 전문에서만 확인 가능한 일부 실험 세부 수치는 PDF 직접 확인을 권장합니다.
