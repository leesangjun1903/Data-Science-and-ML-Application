# ExpertNet: Adversarial Learning and Recovery Against Noisy Labels 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 연구들은 노이즈 레이블의 **부정적 영향을 제거(distill/filter out)** 하는 데 초점을 맞추었으나, ExpertNet은 패러다임을 전환하여 **노이즈 레이블 자체를 학습의 보조 특징(auxiliary feature)으로 활용**한다는 점이 핵심 주장입니다.

> *"Turning noisy labels into learning features"* — 더러운 레이블을 학습 이점으로 전환

### 주요 기여

| 기여 | 설명 |
|------|------|
| 새로운 프레임워크 | Amateur + Expert 이중 네트워크 구조 제안 |
| 패러다임 전환 | 노이즈 레이블 제거 → 노이즈 레이블 활용 |
| 데이터 효율성 | 20~50% 훈련 데이터로 기존 SOTA 수준 달성 |
| 실세계 적용성 | Clothing1M 실세계 노이즈에서 13%p 이상 성능 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 설정:**

훈련 데이터셋 $\mathcal{D} = \{(\boldsymbol{x}_1, \boldsymbol{y}_1, \boldsymbol{t}_1), (\boldsymbol{x}_2, \boldsymbol{y}_2, \boldsymbol{t}_2), \ldots, (\boldsymbol{x}_N, \boldsymbol{y}_N, \boldsymbol{t}_N)\}$

여기서:
- $\boldsymbol{x}_i$: $i$번째 이미지 샘플
- $\boldsymbol{t}_i \in \{0,1\}^K$: Ground truth 레이블 (K 클래스)
- $\boldsymbol{y}_i \in \{0,1\}^K$: 노이즈가 포함된 주어진 레이블

**핵심 문제:** 딥러닝 모델의 **암기 효과(memorization effect)** 로 인해, 노이즈 레이블 하에서 성능이 급격히 저하됨. 예: AlexNet의 CIFAR-10 분류 정확도가 노이즈 레이블 존재 시 77% → 10%로 하락.

**기존 접근의 한계:**
- D2L, Co-teaching, Forward, Bootstrap 등은 모두 $(\boldsymbol{x}_i, \boldsymbol{t}_i)$ 또는 수정된 손실함수만 사용
- 노이즈 레이블 $\boldsymbol{y}_i$가 가진 **보조 정보(auxiliary information)** 를 버림

**ExpertNet의 접근:** 훈련 시: $(\boldsymbol{x}_i, \boldsymbol{y}_i, \boldsymbol{t}_i)$ 모두 사용 / 추론 시: $(\boldsymbol{x}_i, \boldsymbol{y}_i)$ 사용

---

### 2.2 제안 방법 및 수식

#### 소프트맥스 변환 (정보 전달 기반)

Amateur의 출력층은 소프트맥스 변환을 사용:

$$\sigma(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad z_j \in [0,1], \quad \sum_{j=1}^{K} z_j = 1$$

이 확률 분포가 단순 예측 클래스보다 **모델의 확신도(confidence)** 에 대한 더 풍부한 정보를 포함하기 때문에, Expert의 입력으로 사용됨.

#### Amateur 손실 함수

$$l^{\mathcal{A}} = \min_{\boldsymbol{\phi}} \sum_{i=1}^{N} \mathcal{L}\left(g^{\mathcal{E}}\left(\langle g^{\mathcal{A}}(\boldsymbol{x}_i), \boldsymbol{y}_i \rangle; \boldsymbol{\omega}\right),\ g^{\mathcal{A}}(\boldsymbol{x}_i; \boldsymbol{\phi})\right) \tag{1}$$

- Amateur는 **Expert가 보정한 레이블** $\hat{\boldsymbol{y}}^{\mathcal{E}}$에 가깝도록 학습
- $\boldsymbol{\phi}$: Amateur의 파라미터
- $\langle \cdot, \cdot \rangle$: 벡터 연결(concatenation) 함수

#### Expert 손실 함수

$$l^{\mathcal{E}} = \min_{\boldsymbol{\omega}} \sum_{i=1}^{N} \mathcal{L}\left(\boldsymbol{t}_i,\ g^{\mathcal{E}}\left(\langle g^{\mathcal{A}}(\boldsymbol{x}_i), \boldsymbol{y}_i \rangle; \boldsymbol{\omega}\right)\right) \tag{2}$$

- Expert는 **Ground Truth** $\boldsymbol{t}_i$에 가깝도록 학습
- $\boldsymbol{\omega}$: Expert의 파라미터

두 손실 함수 모두 **교차 엔트로피 손실(cross-entropy loss)** 사용:

$$\mathcal{L}(\boldsymbol{p}, \boldsymbol{q}) = -\sum_{k=1}^{K} p_k \log q_k$$

#### 교대 최소화 학습 (Alternating Minimization)

```
Algorithm 1: Training ExpertNet
Input: Training set D = {x, y, t}
Output: Trained Amateur A and Expert E

1. Initialize A and E with random φ and ω
2. for training iteration do
3.   for each batch B{x, y, t} from D do
4.     ŷ^A := Predict label probabilities of x by A        [Amateur 예측]
5.     z  := concatenate <ŷ^A, y>                          [벡터 연결]
6.     Train E with pair (z, t) updating ω                 [Expert 업데이트]
7.     ŷ^E := Predict corrected label probabilities by E    [Expert 보정]
8.     Train A with pair (x, ŷ^E) updating φ              [Amateur 업데이트]
9.   end
10. end
```

---

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────────────────┐
│                      ExpertNet 구조                          │
│                                                             │
│  이미지 x ──→ [Amateur A (CNN)] ──→ Softmax 출력 ŷ^A       │
│                                          │                  │
│  노이즈 레이블 y ────────────────────────┤                  │
│                                     concatenate             │
│                                          │                  │
│                                          ↓                  │
│                              [Expert E (MLP)] ──→ ŷ^E       │
│                                          │                  │
│                              Ground Truth t (훈련시에만)     │
│                                                             │
│  추론 시: (x, y) → Amateur → Expert → 최종 예측             │
└─────────────────────────────────────────────────────────────┘
```

**Amateur (A):**
- CIFAR-10/100: 12-layer CNN (3×[Conv+Conv+Pooling] + FC + Softmax), ReLU 활성화
- Clothing1M: ResNet50, 224×224 입력
- SGD, momentum=0.9, weight decay= $10^{-4}$ , lr=0.01

**Expert (E):**
- 4-layer Feed-forward MLP
- 은닉층: 512×Leaky ReLU, 512×Leaky ReLU
- 출력층: Sigmoid
- 입력: $\langle \hat{\boldsymbol{y}}^{\mathcal{A}}, \boldsymbol{y} \rangle$ (Amateur 소프트맥스 출력 + 노이즈 레이블 연결)

---

### 2.4 성능 향상

#### CIFAR-10 (100% 훈련 데이터)

| 노이즈 비율 | ExpertNet (Expert) | D2L | Co-teaching | Bootstrap | Forward |
|:-----------:|:------------------:|:---:|:-----------:|:---------:|:-------:|
| 20% | **89.23%** | 84.75% | 82.45% | 81.80% | 83.11% |
| 30% | **88.30%** | 82.45% | 80.29% | 77.14% | 81.68% |
| 40% | **84.36%** | 80.69% | 77.28% | 72.44% | 78.12% |
| 50% | **80.73%** | 78.94% | 74.47% | 70.14% | 76.23% |

#### CIFAR-100 (100% 훈련 데이터)

| 노이즈 비율 | ExpertNet (Expert) | D2L | Co-teaching | Bootstrap | Forward |
|:-----------:|:------------------:|:---:|:-----------:|:---------:|:-------:|
| 20% | **86.72%** | 55.70% | 52.74% | 52.58% | 59.87% |
| 30% | **79.92%** | 51.13% | 45.68% | 44.99% | 54.18% |
| 40% | **73.87%** | 49.50% | 41.87% | 40.11% | 49.44% |
| 50% | **66.11%** | 43.56% | 35.89% | 39.84% | 46.06% |

**→ CIFAR-100에서 최대 30%p 이상의 절대적 정확도 향상**

#### Clothing1M (실세계 노이즈, 39.5%)

| 훈련 데이터 | ExpertNet | D2L | Co-teaching | Forward | Bootstrap |
|:-----------:|:---------:|:---:|:-----------:|:-------:|:---------:|
| 50% | **69.83%** | 49.05% | 50.11% | 51.26% | 48.94% |
| 100% | **83.42%** | 69.43% | 69.92% | 70.04% | 68.77% |

---

### 2.5 한계점

논문에서 명시적으로 언급된 한계 및 추론 가능한 한계:

1. **Ground Truth 필요성:** 훈련 시 $\boldsymbol{t}_i$ (ground truth)가 반드시 필요 → 완전한 비지도 설정에는 적용 불가
2. **노이즈 분포 가정:** 대칭적 무작위 노이즈(symmetric random noise)를 가정하며, 비대칭·구조적 노이즈에 대한 분석 미흡
3. **노이즈 증가에 취약:** CIFAR-100에서 노이즈 비율 증가 시 정확도 감소폭이 큼 ("the same does not hold for increasing noise levels")
4. **Expert의 단순한 구조:** 4-layer MLP로 복잡한 노이즈 패턴 모델링의 한계 가능성
5. **추론 시 노이즈 레이블 필요:** 추론 단계에서도 $\boldsymbol{y}_i$가 필요 → 레이블이 없는 순수 이미지 분류 불가

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 데이터 효율성을 통한 일반화

ExpertNet의 가장 두드러진 일반화 특성은 **적은 훈련 데이터로도 높은 성능**을 달성하는 것입니다:

$$\text{CIFAR-100, 20\% 훈련 데이터} \rightarrow \text{ExpertNet: } 80.74\% \text{ vs. D2L(100\%): } 55.70\%$$

이는 노이즈 레이블 $\boldsymbol{y}_i$가 **정규화 효과(regularization effect)** 와 유사한 역할을 함을 시사합니다.

### 3.2 보조 특징으로서의 노이즈 레이블

Expert의 입력:

$$\boldsymbol{z}_i = \langle g^{\mathcal{A}}(\boldsymbol{x}_i; \boldsymbol{\phi}),\ \boldsymbol{y}_i \rangle \in \mathbb{R}^{2K}$$

이 설계는 두 개의 독립적 오류 소스를 결합하는 **앙상블 효과**를 유도합니다:
- Amateur의 오류: 모델 불완전성에 기인
- 노이즈 레이블의 오류: 레이블 노이즈에 기인

Expert는 이 두 오류 패턴의 **차이(disagreement)** 를 학습하여 정확한 클래스를 추론합니다.

### 3.3 다양한 노이즈 비율에 대한 강건성

CIFAR-10에서 노이즈 비율 20%~50% 전 구간에서 일관된 우수성을 보이며:

$$\Delta_{\text{accuracy}} = \text{ExpertNet} - \text{best competitor} \in [1.79\%, 11.92\%]$$

### 3.4 실세계 노이즈 패턴 학습

Clothing1M에서 Expert가 **실세계 노이즈 혼동 행렬(noise confusion matrix)** 패턴을 효과적으로 학습하여 일반화함을 보임. 50% 데이터만으로도 다른 방법의 100% 데이터 수준 달성.

### 3.5 일반화 성능 향상의 메커니즘

```
일반화 향상 메커니즘:
1. Expert의 피드백 → Amateur의 과적합 방지 (implicit regularization)
2. 노이즈 패턴 명시적 학습 → 다양한 노이즈 환경 적응
3. 교대 최소화 → 두 네트워크의 상호 보완적 오류 수정
4. 소프트맥스 확률 활용 → 불확실성 정보 보존
```

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

**패러다임 전환의 영향:**
ExpertNet은 노이즈 레이블 연구의 패러다임을 "노이즈 제거"에서 "노이즈 활용"으로 전환시키는 중요한 이정표입니다. 이는 다음 연구 방향을 촉발합니다:

1. **레이블 정보 활용 극대화:** 불완전한 데이터도 학습 신호로 활용하는 연구 가속화
2. **이중 네트워크 패러다임:** Amateur-Expert와 유사한 Teacher-Student 구조의 확장
3. **보조 정보 융합:** 이미지 외의 메타데이터(태그, 설명 등)를 활용하는 멀티모달 학습
4. **의료 영상 분야 적용:** 여러 전문가의 레이블이 공존하는 환경에서의 적용 가능성

### 4.2 향후 연구 시 고려사항

**① Ground Truth 의존성 완화:**

완전한 실세계 적용을 위해서는 일부 클린 레이블만으로 Expert를 학습하는 **준지도학습(semi-supervised)** 확장이 필요합니다. 예: Hendrycks et al. [6]의 trusted data 접근법과 결합.

**② 비대칭 노이즈(Asymmetric Noise) 처리:**

현재 대칭 노이즈 $P(\tilde{y}=j|y=k) = \frac{\epsilon}{K-1}$ 만 다루지만, 실세계에서는 클래스 간 혼동이 비대칭적으로 발생:

$$P(\tilde{y}=j|y=k) \neq P(\tilde{y}=k|y=j)$$

**③ 동적 노이즈 패턴 적응:**

훈련 과정에서 노이즈 패턴이 변화하는 **비정상(non-stationary)** 환경에 대한 적응 메커니즘 연구 필요.

**④ Expert 구조의 고도화:**

단순 MLP 대신 **Transformer** 또는 **Graph Neural Network** 를 활용하여 클래스 간 관계를 명시적으로 모델링:

$$\text{Expert}(\boldsymbol{z}) = \text{Transformer}(\langle g^{\mathcal{A}}(\boldsymbol{x}), \boldsymbol{y} \rangle)$$

**⑤ 추론 시 레이블 불필요 설정 확장:**

추론 시 $\boldsymbol{y}_i$가 없는 경우에도 적용 가능하도록, Amateur만으로도 Expert 수준 성능을 달성하는 **지식 증류(knowledge distillation)** 적용.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래 최신 연구들은 제가 훈련 데이터를 기반으로 알고 있는 일반적 지식을 활용한 것이며, 논문 원문을 직접 확인하지 않았으므로 세부 수치에 오류가 있을 수 있습니다. 개요 수준의 비교만 제시합니다.

### 5.1 주요 후속 연구 동향

| 연구 | 핵심 아이디어 | ExpertNet과의 관계 |
|------|--------------|-------------------|
| **DivideMix** (Li et al., ICLR 2020) | GMM으로 클린/노이즈 샘플 분리 + MixMatch | 노이즈 제거 패러다임, ExpertNet과 상호보완 가능 |
| **SELF** (Nguyen et al., NeurIPS 2020) | 자기 앙상블을 통한 노이즈 레이블 보정 | Expert의 보정 역할과 유사한 동기 |
| **Jo-SRC** (Yao et al., CVPR 2021) | 두 뷰의 일관성으로 클린 샘플 선택 | Co-teaching 계열의 발전 |
| **CORES** (Cheng et al., ICML 2021) | 샘플별 신뢰도 점수 추정 | Expert의 레이블 보정과 보완적 |
| **NoiseRank** (Sharma et al., 2020) | 그래프 기반 노이즈 레이블 순위화 | ExpertNet의 노이즈 활용 아이디어 연장 가능 |

### 5.2 ExpertNet의 차별성

```
ExpertNet의 독창성:
┌─────────────────────────────────────────────────────┐
│  기존 2020년 이후 연구들: 노이즈 "제거" 또는 "회피"  │
│  ExpertNet: 노이즈를 "특징"으로 "활용"               │
│                                                     │
│  공통점: 이중 네트워크 구조 (두 모델의 상호작용)      │
│  차이점: 노이즈 레이블을 입력으로 명시적 사용        │
└─────────────────────────────────────────────────────┘
```

### 5.3 ExpertNet이 현재 연구 트렌드에서 가지는 위치

**한계 재확인:** DivideMix 등 최신 연구들은 클린 레이블 없이도 높은 성능을 달성하는 반면, ExpertNet은 훈련 시 ground truth 레이블이 필요하다는 전제 조건이 다릅니다. 따라서 **완전히 동일한 설정에서의 비교는 공정하지 않을 수 있습니다.**

**연구 융합 방향:** ExpertNet의 "노이즈 활용" 아이디어와 DivideMix의 "반지도학습" 아이디어를 결합하면:

$$\text{향후 연구} = \text{ExpertNet의 노이즈 활용} + \text{DivideMix의 준지도 학습}$$

이는 ground truth 없이도 노이즈 패턴을 학습하는 더 실용적인 프레임워크로 발전 가능합니다.

---

## 참고 자료

1. **논문 원문:** Ghiassi, A., Birke, R., Han, R., & Chen, L. Y. (2020). *ExpertNet: Adversarial Learning and Recovery Against Noisy Labels*. arXiv:2007.05305v2.
2. **비교 연구 [D2L]:** Wang, Y., Ma, X., Houle, M. E., Xia, S. T., & Bailey, J. (2018). *Dimensionality-driven learning with noisy labels*. ICML 2018.
3. **비교 연구 [Co-teaching]:** Han, B., et al. (2018). *Co-teaching: Robust training of deep neural networks with extremely noisy labels*. NeurIPS 2018.
4. **비교 연구 [Forward]:** Patrini, G., et al. (2017). *Making deep neural networks robust to label noise: A loss correction approach*. CVPR 2017.
5. **비교 연구 [Bootstrap]:** Reed, S. E., et al. (2015). *Training deep neural networks on noisy labels with bootstrapping*. ICLR Workshop 2015.
6. **후속 연구 [DivideMix]:** Li, J., Socher, R., & Hoi, S. C. (2020). *DivideMix: Learning with noisy labels as semi-supervised learning*. ICLR 2020. (arXiv:2002.07394)
7. **데이터셋 [Clothing1M]:** Xiao, T., Xia, T., Yang, Y., Huang, C., & Wang, X. (2015). *Learning from massive noisy labeled data for image classification*. CVPR 2015.
8. **memorization 효과:** Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). *Understanding deep learning requires rethinking generalization*. ICLR 2017.
