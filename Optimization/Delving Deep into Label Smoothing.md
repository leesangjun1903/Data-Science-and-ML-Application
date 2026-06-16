# Delving Deep into Label Smoothing

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

기존 Label Smoothing(LS)은 소프트 레이블 생성 시 **비목표 클래스 간의 관계를 무시하고 균일한 확률**을 부여한다. 예를 들어, 'cat' 이미지에 대해 'dog'와 'automobile'에 동일한 확률을 부여하는 것은 불합리하다. 본 논문은 **모델의 중간 예측(intermediate prediction) 통계**를 활용해 클래스 간 유사도를 반영한 더 합리적인 소프트 레이블을 생성할 수 있다고 주장한다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **OLS(Online Label Smoothing) 제안** | 훈련 과정에서 동적으로 업데이트되는 클래스 수준의 소프트 레이블 생성 |
| **클래스 간 관계 반영** | 균일 분포 대신 모델 예측 통계 기반의 비균일 분포 사용 |
| **노이즈 레이블 강건성** | 별도 설계 없이도 노이즈 레이블에 대한 강건성 향상 |
| **적대적 공격 강건성** | 클래스 내(intra-class) 제약으로 결정 경계 근처의 샘플 수 감소 |
| **플러그인 모듈** | 다양한 모델, 데이터셋, 태스크에 용이하게 적용 가능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**기존 LS의 한계:**

기존 Label Smoothing(Szegedy et al., 2016)의 소프트 레이블은 아래와 같이 정의된다:

$$q'(k|\boldsymbol{x}_i) = (1 - \varepsilon)q(k|\boldsymbol{x}_i) + \frac{\varepsilon}{K} \tag{2}$$

여기서 $\varepsilon$은 스무딩 파라미터(보통 0.1), $K$는 클래스 수이다.

**문제점:** 비목표 클래스에 **고정된 균일한 확률** $\frac{\varepsilon}{K}$를 부여하므로, 클래스 간의 의미론적(semantic) 유사도를 전혀 반영하지 못한다. Müller et al. (2019, "When does label smoothing help?")도 이 점을 지적한 바 있다.

### 2.2 제안 방법 (수식 포함)

#### (a) 기본 손실 함수

**Hard Label 기반 Cross-Entropy:**

$$\mathcal{L}_{hard} = -\sum_{k=1}^{K} q(k|\boldsymbol{x}_i) \log p(k|\boldsymbol{x}_i) = -\log p(k=y_i|\boldsymbol{x}_i) \tag{1}$$

#### (b) OLS의 소프트 손실 함수

$t$번째 에폭에서 이전 에폭의 소프트 레이블 $S^{t-1}_{y_i}$을 이용해 훈련:

$$\mathcal{L}_{soft} = -\sum_{k=1}^{K} S^{t-1}_{y_i, k} \cdot \log p(k|\boldsymbol{x}_i) \tag{3}$$

#### (c) 최종 훈련 손실

$$\mathcal{L} = \alpha \mathcal{L}_{hard} + (1 - \alpha) \mathcal{L}_{soft} \tag{4}$$

여기서 $\alpha = 0.5$일 때 최적 성능을 달성한다 (Ablation Study 결과).

#### (d) 소프트 레이블 업데이트

입력 샘플 $(\boldsymbol{x}_i, y_i)$가 **정확히 분류된 경우**에만 소프트 레이블 행렬 $S^t$를 업데이트:

$$S^t_{y_i, k} = S^t_{y_i, k} + p(k|\boldsymbol{x}_i) \tag{5}$$

에폭 종료 후 컬럼별 정규화:

$$S^t_{y_i, k} \leftarrow \frac{S^t_{y_i, k}}{\sum_{l=1}^{K} S^t_{y_i, l}} \tag{6}$$

#### (e) 소프트 레이블의 해석

$t-1$ 에폭의 소프트 레이블은 다음과 같이 해석된다:

$$S^{t-1}_{y_i, k} = \frac{1}{N} \sum_{j=1}^{N} p^{t-1}(k|\boldsymbol{x}_j) \tag{7}$$

이를 $\mathcal{L}_{soft}$에 대입하면:

$$\mathcal{L}_{soft} = -\frac{1}{N} \sum_{j=1}^{N} \sum_{k=1}^{K} p^{t-1}(k|\boldsymbol{x}_j) \cdot \log p(k|\boldsymbol{x}_i) \tag{8}$$

이 수식은 **같은 클래스의 모든 올바르게 분류된 샘플들**이 현재 샘플에 제약을 가하여, 클래스 내 응집도(intra-class cohesion)를 높임을 의미한다.

### 2.3 모델 구조 및 알고리즘

```
Algorithm 1: OLS Pipeline
─────────────────────────────────────────────────────
Input: Dataset D_train = {(x_i, y_i)}, model f_θ, epochs T
Initialize: S⁰ = (1/K)I  [균일 분포로 초기화]

for t = 1 to T:
    Initialize: Sᵗ = 0
    for each mini-batch B:
        1. Forward pass → {p(k|x_i)}
        2. Compute L = αL_hard + (1-α)L_soft  [Eq.4]
        3. Backward & update θ
        4. if correctly classified:
               Sᵗ_{y_i} ← Sᵗ_{y_i} + f(θ, x_i)  [Eq.5]
    Normalize Sᵗ column-wise  [Eq.6]
─────────────────────────────────────────────────────
```

**핵심 구조적 특징:**
- $S^t \in \mathbb{R}^{K \times K}$: 각 컬럼이 한 클래스의 소프트 레이블
- **클래스 수준(class-level)** 소프트 레이블 (샘플 수준이 아님)
- 첫 에폭: 균일 분포 $S^0 = \frac{1}{K}I$ 사용 (수렴 안정성 보장)
- 교사 모델, 특수 아키텍처, 추가 순전파 불필요

### 2.4 성능 향상 결과

#### CIFAR-100

| Method | ResNet-56 Top-1 Err(%) | ResNeXt29-2x64d Top-1 Err(%) |
|---|---|---|
| Hard Label | 26.81 | 20.92 |
| LS | 26.37 | 20.34 |
| **OLS (Ours)** | **25.24** | **18.81** |

- OLS가 ResNet-56 대비 **1.57%**, ResNeXt29-2x64d 대비 **2.11%** 향상

#### ImageNet

| Model | Hard Label | LS | OLS |
|---|---|---|---|
| ResNet-50 Top-1 Err(%) | 23.68 | 22.82 | **22.28** |
| ResNet-101 Top-1 Err(%) | 21.87 | 21.27 | **20.85** |

#### 객체 검출 (PASCAL VOC, YOLO)

| Method | mAP(%) |
|---|---|
| Hard Label | 81.6 |
| LS | 82.3 |
| **OLS** | **82.7** |

### 2.5 한계점

논문에서 명시적으로 언급된 한계 및 추론 가능한 한계:

1. **초기 에폭 의존성:** 첫 에폭에는 균일 분포를 사용하므로, 초기 학습이 불안정할 경우 소프트 레이블의 품질이 영향받을 수 있음
2. **메모리 오버헤드:** $K \times K$ 크기의 소프트 레이블 행렬 $S^t$ 유지 필요 (클래스 수가 매우 클 경우 문제)
3. **업데이트 주기의 민감성:** 업데이트 주기가 1에폭보다 길어지면 성능이 급격히 하락
4. **정확 분류 샘플만 활용:** 초기 훈련 시 정확히 분류되는 샘플 수가 적어 소프트 레이블 품질이 낮을 수 있음
5. **하이퍼파라미터 $\alpha$ 튜닝:** 데이터셋/모델에 따른 최적 $\alpha$ 탐색 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상 메커니즘

OLS가 일반화 성능을 높이는 핵심 메커니즘은 **클래스 내 제약(intra-class constraints)** 이다. 수식 (8)에서 보듯이, 같은 클래스의 모든 올바르게 분류된 샘플들이 현재 샘플의 학습에 제약을 가한다. 이를 통해:

$$\mathcal{L}_{soft} = -\frac{1}{N}\sum_{j=1}^{N}\sum_{k=1}^{K} p^{t-1}(k|\boldsymbol{x}_j) \cdot \log p(k|\boldsymbol{x}_i)$$

**동일 클래스 샘플들의 표현이 서로 가까워지도록 유도** → 과적합 방지 → 일반화 향상

### 3.2 t-SNE 시각화

논문의 Figure 3에서 ResNet-56 (CIFAR-100)의 penultimate layer 표현을 t-SNE로 시각화한 결과:

| 방법 | 클래스 간 구분 | 클래스 내 응집도 |
|---|---|---|
| Hard Label | 낮음 | 낮음 |
| LS | 중간 | 중간 |
| **OLS** | **높음** | **높음** |

### 3.3 모델 캘리브레이션 (Expected Calibration Error)

과적합의 대표적 증상인 **과신뢰(over-confidence)**를 ECE(Expected Calibration Error)로 측정:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|$$

| Method | ResNet-56 ECE | ResNet-110 ECE |
|---|---|---|
| Hard Label | 11.37 | 13.14 |
| LS | 3.35 | 2.32 |
| **OLS** | **2.85** | **2.05** |

OLS가 LS보다 낮은 ECE를 달성 → **더 잘 보정된(calibrated) 모델** → 일반화 성능 향상

### 3.4 노이즈 레이블에서의 일반화

노이즈율 40%에서:

| Method | Top-1 Error(%) |
|---|---|
| Hard Label | 47.07 |
| LS | 43.99 |
| **OLS** | **38.86** |

OLS는 **잘못된 레이블에 대한 과적합을 억제**하여 일반화 성능을 향상시킨다. Figure 5에서 OLS가 노이즈 샘플에 더 높은 에러율(즉, 덜 맞춤)을 보이면서도 테스트 에러는 낮음을 확인.

### 3.5 인간 불확실성과의 일치성 (CIFAR-10H)

CIFAR-10H(50명 이상의 인간 투표로 생성된 소프트 레이블)와 모델 예측 분포의 KL Divergence 비교:

| Method | CIFAR-10 Top-1 Err(%) | CIFAR-10H KL Divergence |
|---|---|---|
| Hard Label | 7.18 | 0.2974 |
| LS | 6.81 | 0.1866 |
| **OLS** | **6.46** | **0.1399** |

OLS의 소프트 레이블이 **인간의 불확실성 분포에 가장 근접** → 더 현실적인 클래스 관계 학습 → 일반화 향상

---

## 4. 앞으로의 연구에 미치는 영향과 고려 사항

### 4.1 앞으로의 연구에 미치는 영향

#### (a) 레이블 정규화 패러다임의 확장
OLS는 정적(static) 소프트 레이블에서 **동적(dynamic) 소프트 레이블**로의 전환을 제시하여, 이후 연구들이 훈련 과정에서 레이블을 지속적으로 개선하는 방향으로 발전하도록 영향을 미쳤다.

#### (b) Knowledge Distillation과의 통합
OLS는 교사 모델 없이도 KD의 혜택을 얻을 수 있음을 보여줌으로써, **Self-KD 연구의 새로운 방향**을 제시했다.

#### (c) 노이즈 레이블 학습과의 연계
별도로 설계하지 않았음에도 노이즈에 강건함을 보여주어, **레이블 정제(label correction)와 정규화를 결합**하는 연구 방향을 열었다.

#### (d) 다양한 태스크로의 확장
객체 검출(YOLO + PASCAL VOC)에서의 성능 향상은 OLS가 분류를 넘어 **다양한 컴퓨터 비전 태스크**에 적용 가능함을 시사한다.

### 4.2 앞으로 연구 시 고려할 점

#### (a) 대규모 클래스 환경
$K \times K$ 행렬 유지의 메모리 비용 → **수천~수만 클래스 환경(예: Open-Vocabulary)에서의 효율적 구현** 연구 필요

#### (b) 초기 훈련 불안정 문제
초기 에폭의 소프트 레이블 품질 저하 → **Curriculum Learning**이나 **Warm-up 전략**과의 결합 탐색

#### (c) 클래스 불균형(Class Imbalance)
OLS는 정확히 분류된 샘플만 집계하므로, **소수 클래스의 소프트 레이블 품질이 저하될 가능성** 존재 → 샘플 가중치 보정 연구 필요

#### (d) Transformer 기반 모델로의 적용
ViT, Swin Transformer 등 최신 아키텍처에서의 OLS 효과 체계적 검증 필요 (논문은 주로 CNN 기반 실험)

#### (e) 멀티태스크 및 멀티레이블 학습
하나의 이미지가 여러 레이블을 가지는 환경에서의 OLS 확장 방법 탐색

#### (f) 소프트 레이블의 신뢰도 측정
모든 에폭에서 동일 가중치로 소프트 레이블을 업데이트하므로, **에폭별 신뢰도를 고려한 가중 업데이트** 전략 연구

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 Tf-KD (Yuan et al., CVPR 2020)
**"Revisiting Knowledge Distillation via Label Smoothing Regularization"**

| 항목 | Tf-KD $_{reg}$ | OLS |
|---|---|---|
| 소프트 레이블 방식 | 수동 설계 분포 | 모델 예측 통계 기반 |
| 클래스 간 관계 반영 | ✗ (비목표는 균일) | ✓ (비균일) |
| 교사 모델 필요 여부 | ✗ | ✗ |
| Fine-grained 평균 Top-1 개선 | +0.44% (Hard 대비) | **+2.00% (Hard 대비)** |

Tf-KD의 비목표 클래스 분포:

$$u(k) = \begin{cases} a & \text{if } k = c \\ \frac{1-a}{K-1} & \text{if } k \neq c \end{cases} \tag{9}$$

**평가:** Tf-KD는 목표 클래스 정보를 추가했지만 비목표 클래스 간 관계는 여전히 균일 → OLS가 fine-grained 분류에서 명확한 우위

### 5.2 DeiT (Touvron et al., ICML 2021)
**"Training data-efficient image transformers & distillation through attention"**

- ViT 훈련에 Label Smoothing($\varepsilon=0.1$)을 기본 구성요소로 포함
- 토큰 기반 Knowledge Distillation 도입
- **OLS와의 관계:** DeiT의 LS 부분을 OLS로 대체 가능한지에 대한 연구 공백 존재
- Transformer 환경에서 동적 소프트 레이블의 효과 검증 필요

### 5.3 CKDF (Chen et al., 이후 연구들)
OLS가 제시한 **클래스 수준 소프트 레이블** 개념은 이후 다양한 Self-KD 연구에서 참조되었으며, 특히 클래스 프로토타입(class prototype) 기반 정규화 연구로 이어졌다.

### 5.4 비교 요약표

| 연구 | 연도 | 소프트 레이블 방식 | 클래스 관계 | 동적 업데이트 | 교사 필요 |
|---|---|---|---|---|---|
| LS (Szegedy et al.) | 2016 | 균일 분포 혼합 | ✗ | ✗ | ✗ |
| Tf-KD (Yuan et al.) | 2020 | 수동 설계 | 부분적 | ✗ | ✗ |
| **OLS (Zhang et al.)** | **2021** | **모델 예측 통계** | **✓** | **✓** | **✗** |
| DeiT (Touvron et al.) | 2021 | LS + Distillation | 부분적 | ✗ | ✓ |

---

## 참고자료

1. **Chang-Bin Zhang et al.** (2021). "Delving Deep into Label Smoothing." *IEEE Transactions on Image Processing* (TIP). (본 논문 원문, 제공된 PDF)

2. **Szegedy, C. et al.** (2016). "Rethinking the inception architecture for computer vision." *CVPR 2016.* (Label Smoothing 원저)

3. **Müller, R., Kornblith, S., & Hinton, G.** (2019). "When does label smoothing help?" *NeurIPS 2019.*

4. **Hinton, G., Vinyals, O., & Dean, J.** (2015). "Distilling the knowledge in a neural network." *NeurIPS Workshop 2015.*

5. **Yuan, L. et al.** (2020). "Revisiting knowledge distillation via label smoothing regularization." *CVPR 2020.*

6. **Touvron, H. et al.** (2021). "Training data-efficient image transformers & distillation through attention." *ICML 2021.*

7. **Zhang, L. et al.** (2019). "Be your own teacher: Improve the performance of convolutional neural networks via self distillation (BYOT)." *ICCV 2019.*

8. **Peterson, J.C. et al.** (2019). "Human uncertainty makes classification more robust." *ICCV 2019.* (CIFAR-10H 데이터셋)

9. **Guo, C. et al.** (2017). "On calibration of modern neural networks." *ICML 2017.* (ECE 측정 방법론)

10. **Wang, Y. et al.** (2019). "Symmetric cross entropy for robust learning with noisy labels." *ICCV 2019.*

> **주의:** Touvron et al. (DeiT)와 OLS의 직접 비교 실험은 원 논문에 포함되어 있지 않으며, 해당 부분은 공개된 각 논문의 내용을 바탕으로 분석한 것입니다. OLS가 Transformer 기반 모델에서 갖는 효과에 대한 체계적 실험 결과는 현재까지 제한적이므로, 이 부분의 주장은 추론에 기반한 것임을 명시합니다.
