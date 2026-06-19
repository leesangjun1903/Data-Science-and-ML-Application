# Decoupling Representation and Classifier for Long-Tailed Recognition

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 **롱테일 인식(long-tailed recognition)에서 표현 학습(representation learning)과 분류기 학습(classifier learning)을 분리(decouple)하는 것이 효과적**이라는 것입니다. 특히 두 가지 놀라운 발견을 제시합니다:

1. **데이터 불균형은 고품질 표현 학습에 방해가 되지 않는다** — 가장 단순한 인스턴스 균형 샘플링(instance-balanced sampling)으로 학습된 표현이 가장 일반화 성능이 높다.
2. **분류기만 조정해도 강력한 롱테일 인식 성능을 달성할 수 있다** — 복잡한 손실 함수나 메모리 모듈 없이도 분류기 재조정만으로 SOTA를 달성한다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| 디커플링 프레임워크 제안 | 표현 학습과 분류기 학습을 체계적으로 분리 |
| 샘플링 전략 분석 | 4가지 샘플링 전략의 효과를 분리 평가 |
| 분류기 재조정 방법 3종 제안 | cRT, NCM, $\tau$-normalized (+ LWS) |
| SOTA 달성 | ImageNet-LT, Places-LT, iNaturalist 2018에서 새로운 최고 성능 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

현실 세계의 데이터는 **롱테일 분포(long-tailed distribution)**를 따릅니다. 즉, 일부 클래스(head class)에는 데이터가 풍부하고, 나머지 클래스(tail class)에는 데이터가 매우 적습니다.

기존 접근법들(손실 재가중치, 데이터 재샘플링, 헤드→테일 전이학습)은 **표현 학습과 분류기 학습을 동시에** 수행하므로, 성능 향상이 어디서 비롯되는지 불명확했습니다. 논문은 이를 명확히 분리하여 분석합니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### ▶ 샘플링 전략 통합 수식

클래스 $j$에서 데이터 포인트를 샘플링할 확률 $p_j$:

$$p_j = \frac{n_j^q}{\sum_{i=1}^{C} n_i^q} \tag{1}$$

- $n_j$: 클래스 $j$의 훈련 샘플 수
- $C$: 전체 클래스 수
- $q \in [0, 1]$: 샘플링 전략을 제어하는 파라미터

| $q$ 값 | 샘플링 전략 |
|---|---|
| $q = 1$ | Instance-balanced sampling (자연 분포 유지) |
| $q = 0$ | Class-balanced sampling (각 클래스 동등) |
| $q = 1/2$ | Square-root sampling (절충안) |

#### ▶ Progressive-balanced Sampling

에폭 $t$에서의 샘플링 확률:

$$p_j^{\text{PB}}(t) = \left(1 - \frac{t}{T}\right) p_j^{\text{IB}} + \frac{t}{T} p_j^{\text{CB}} \tag{2}$$

- 훈련 초반에는 인스턴스 균형 샘플링, 후반으로 갈수록 클래스 균형 샘플링으로 점진적 전환

#### ▶ 분류기 재조정 방법

**① Classifier Re-Training (cRT)**

고정된 표현 위에서 분류기 가중치 $\mathbf{W}$와 $\mathbf{b}$를 클래스 균형 샘플링으로 재초기화 후 재훈련:

$$\hat{y} = \arg\max \left(\mathbf{W}^\top f(x; \theta) + \mathbf{b}\right)$$

여기서 $\theta$는 고정되고 $\mathbf{W}, \mathbf{b}$만 업데이트됩니다.

---

**② Nearest Class Mean Classifier (NCM)**

각 클래스의 평균 특징 벡터를 계산 후 최근접 이웃 탐색:

$$\hat{y} = \arg\min_{j} \, d\!\left(f(x;\theta),\, \mu_j\right)$$

여기서 $\mu_j = \frac{1}{n_j}\sum_{x: y=j} f(x;\theta)$이며, 코사인 유사도 또는 L2 거리를 사용합니다.

---

**③ $\tau$-Normalized Classifier ($\tau$-normalized)**

공동 훈련 후 분류기 가중치 노름이 클래스 크기에 비례하는 문제를 해결하기 위한 정규화:

$$\widetilde{w}_i = \frac{w_i}{\|w_i\|^\tau} \tag{3}$$

- $\tau = 1$: 표준 L2 정규화
- $\tau = 0$: 스케일링 없음
- $\tau \in (0, 1)$: 부드러운 균형 조정 (실험적으로 결정)

정규화 후 최종 로짓: $\hat{y} = \widetilde{\mathbf{W}}^\top f(x;\theta)$

---

**④ Learnable Weight Scaling (LWS)**

$\tau$-normalization의 스케일 인수를 학습 가능한 파라미터로 확장:

$$\widetilde{w}_i = f_i \cdot w_i, \quad \text{where} \quad f_i = \frac{1}{\|w_i\|^\tau} \tag{4}$$

$f_i$를 클래스 균형 샘플링으로 학습하되, 표현과 분류기 방향은 고정합니다.

---

#### ▶ 비교를 위한 손실 함수 (부록 수식)

**Focal Loss:**

$$\mathcal{L}_{\text{focal}} := (1 - h_i)^\gamma \mathcal{L}_{CE} = -(1 - h_i)^\gamma \log(h_i) \tag{5}$$

**LDAM Loss:**

$$\mathcal{L}_{\text{LDAM}} := -\log \frac{e^{\hat{y}_j - \Delta_j}}{e^{\hat{y}_j - \Delta_j} + \sum_{c \neq j} e^{\hat{y}_c - \Delta_c}} \tag{6}$$

- $\Delta_j$: 클래스 인식 마진으로 $n_j^{1/4}$에 반비례

---

### 2-3. 모델 구조

논문의 모델 구조는 **2단계 디커플링 파이프라인**입니다:

```
[Stage 1] 표현 학습
  입력 데이터 → CNN Backbone (ResNet/ResNeXt)
               → Instance-balanced sampling으로 훈련
               → 특징 벡터 z = f(x; θ) 추출

[Stage 2] 분류기 재조정 (θ 고정)
  특징 벡터 z → 분류기 선택
              ├─ cRT (클래스 균형 재훈련)
              ├─ NCM (비파라메트릭 최근접 평균)
              ├─ τ-normalized (가중치 정규화)
              └─ LWS (학습 가능한 스케일링)
              → 최종 예측 ỹ
```

**백본 네트워크**: ResNet-{10, 50, 101, 152}, ResNeXt-{50, 101, 152}(32×4d)

**훈련 설정**:
- 옵티마이저: SGD (momentum=0.9)
- 배치 크기: 512
- 학습률: 코사인 스케줄 (0.2 → 0)
- Stage 1: 90~200 에폭
- Stage 2 (cRT): 10 에폭

---

### 2-4. 성능 향상

#### ImageNet-LT (ResNeXt-50 기준)

| 방법 | Many | Medium | Few | All |
|---|---|---|---|---|
| Joint (Instance-balanced) | 65.9 | 37.5 | 7.7 | 44.4 |
| OLTR* | - | - | - | 37.7 |
| NCM | 56.6 | 45.3 | 28.1 | 47.3 |
| cRT | 61.8 | **46.2** | 27.4 | 49.6 |
| $\tau$-normalized | 59.1 | 46.9 | 30.7 | 49.4 |
| **LWS** | 60.2 | 47.2 | 30.3 | **49.9** |

#### iNaturalist 2018 (ResNet-152, 200 에폭)

| 방법 | All |
|---|---|
| LDAM+DRW | 68.0 |
| cRT | 71.2 |
| **$\tau$-normalized** | **72.5** |

#### Places-LT (ResNet-152, ImageNet pretrained)

| 방법 | Many | Medium | Few | All |
|---|---|---|---|---|
| OLTR | 44.7 | 37.0 | 25.3 | 35.9 |
| Joint | 45.7 | 27.3 | 8.2 | 30.2 |
| **$\tau$-normalized** | 37.8 | **40.7** | **31.8** | **37.9** |

---

### 2-5. 한계

1. **$\tau$ 하이퍼파라미터 의존성**: $\tau$-normalized는 교차 검증으로 $\tau$를 결정해야 합니다 (단, 훈련 세트에서도 탐색 가능함을 부록에서 확인).
2. **2단계 훈련의 추가 비용**: cRT와 LWS는 추가적인 재훈련 단계가 필요합니다.
3. **테일 클래스의 절대 정확도 한계**: Few-shot 클래스에서의 성능은 여전히 Many-shot 대비 낮습니다.
4. **검증 세트 필요성**: 일부 방법에서 하이퍼파라미터 튜닝에 검증 세트가 필요합니다.
5. **오픈셋 시나리오 미검토**: 훈련 중 보지 못한 새로운 클래스에 대한 일반화는 다루지 않습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 핵심 발견: 인스턴스 균형 샘플링의 우월한 일반화

논문의 가장 중요한 발견 중 하나는 **인스턴스 균형 샘플링(Instance-balanced sampling, $q=1$)으로 학습된 표현이 가장 높은 일반화 성능을 보인다**는 점입니다.

Figure 1의 결과에서, 디커플링된 분류기(NCM, cRT, $\tau$-normalized)와 조합할 때 인스턴스 균형 샘플링으로 학습된 표현이 클래스 균형, 제곱근, 점진적 균형 샘플링보다 전반적으로 우수한 성능을 보입니다:

$$p_j^{\text{IB}} = \frac{n_j}{\sum_{i=1}^{C} n_i} \quad (q=1 \text{ 적용})$$

### 3-2. 왜 인스턴스 균형 샘플링이 일반화에 유리한가?

**① 표현의 다양성 보존**: 자연 분포를 유지함으로써 head 클래스의 풍부한 데이터에서 강건하고 다양한 특징을 학습합니다. 이 특징은 tail 클래스에도 전이될 수 있습니다.

**② 과적합 방지**: 클래스 균형 샘플링은 tail 클래스의 소수 샘플을 반복 노출시켜 과적합을 유발할 수 있습니다. 인스턴스 균형 샘플링은 이를 자연스럽게 회피합니다.

**③ 결정 경계의 분리**: 표현의 품질은 보존하면서, 불균형한 결정 경계만을 분류기 재조정 단계에서 수정합니다.

### 3-3. 분류기 가중치 노름과 일반화의 관계

논문 Figure 2(left)에서 보여주듯이:

- **공동 훈련 후** (파란 선): $\|w_j\|$가 $n_j$와 양의 상관관계 → head 클래스에 과도하게 편향된 결정 경계
- **cRT 후** (초록 선): 가중치 노름이 균형화됨
- **$\tau$-normalized 후** (금색 선): 부드럽게 균형화됨

$$\widetilde{w}_i = \frac{w_i}{\|w_i\|^\tau}$$

이 정규화는 추가 훈련 없이도 결정 경계를 균형화하여 tail 클래스의 분류 성능을 대폭 향상시킵니다.

### 3-4. NCM의 일반화 우수성

NCM은 클래스 평균 특징 벡터를 활용하여 **프로토타입 기반 분류**를 수행합니다. 코사인 유사도를 사용할 경우 가중치 노름의 불균형 문제를 내재적으로 해결하며, 추가 훈련 없이도 강력한 성능을 보입니다. 이는 학습된 표현 공간 자체의 일반화 능력을 직접적으로 반영합니다.

### 3-5. iNaturalist에서의 균형 성능

$\tau$-normalized 분류기를 200 에폭 훈련 후 적용했을 때:
- Many-shot: 71.1%, Medium-shot: 68.9%, Few-shot: **69.3%** (ResNet-50)

이는 many/medium/few 클래스 간 정확도가 거의 균등해지는 놀라운 결과로, 높은 일반화 성능의 증거입니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 연구에 미치는 영향

#### ① 롱테일 인식의 새로운 패러다임 확립
이 논문은 롱테일 인식 연구에서 **"표현 학습"과 "분류기 학습"을 분리하여 분석하는 방법론적 기준**을 제시했습니다. 이후 많은 연구들이 이 프레임워크를 기반으로 발전했습니다.

#### ② 자기지도 학습과의 결합 가능성
인스턴스 균형 샘플링이 최적의 표현을 학습한다는 발견은, **자기지도 학습(SSL: Self-Supervised Learning)** 방식의 사전학습이 롱테일 인식에 효과적일 수 있음을 시사합니다. 실제로 이후 연구들(예: BCL, PaCo 등)이 대조 학습과 결합하여 성능을 향상시켰습니다.

#### ③ 실용적인 파이프라인 제공
복잡한 메모리 모듈이나 정교한 손실 설계 없이도 경쟁력 있는 성능을 달성할 수 있는 **간단하고 재현 가능한 베이스라인**을 제공했습니다.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

#### ① MiSLAS (CVPR 2021)
- **제목**: "Improving Calibration for Long-Tailed Recognition" (Zhong et al., 2021)
- **핵심**: 디커플링 프레임워크에서 **믹스업(Mixup)**을 통한 데이터 증강과 레이블 스무딩을 결합
- **관계**: 본 논문의 2단계 학습을 직접 채택하고, 표현 품질을 추가로 향상

#### ② PaCo (ICCV 2021)
- **제목**: "Parametric Contrastive Learning" (Cui et al., 2021)
- **핵심**: 파라메트릭 대조 학습으로 표현 학습과 분류기 학습을 동시에 개선
- **관계**: 인스턴스 균형 샘플링의 우수성을 대조 학습 관점에서 재해석

#### ③ BALLAD (NeurIPS 2021)
- **핵심**: 지식 증류를 활용하여 균형 잡힌 분류기로부터 표현을 개선
- **관계**: 디커플링을 더 동적인 방식으로 확장

#### ④ BCL (CVPR 2022)
- **제목**: "Balanced Contrastive Learning for Long-Tailed Visual Recognition" (Zhu et al., 2022)
- **핵심**: 대조 학습에서 클래스 균형 손실과 인스턴스 균형 손실을 결합
- **관계**: 디커플링 아이디어를 대조 학습 프레임워크에 적용

#### ⑤ VL-LTR (ECCV 2022)
- **핵심**: CLIP 등 비전-언어 모델의 사전지식을 롱테일 인식에 활용
- **관계**: 강력한 사전학습 표현이 롱테일 인식에 효과적임을 본 논문의 통찰과 연결

#### 연구 트렌드 비교표

| 연구 | 표현 학습 | 분류기 학습 | 주요 기법 |
|---|---|---|---|
| **본 논문 (2020)** | Instance-balanced | cRT / $\tau$-norm / NCM | 디커플링 |
| MiSLAS (2021) | Mixup + Instance-balanced | 레이블 스무딩 + cRT | 보정 강화 |
| PaCo (2021) | 파라메트릭 대조 학습 | 통합 | 대조 학습 |
| BCL (2022) | 균형 대조 학습 | 통합 | 이중 균형 |
| VL-LTR (2022) | CLIP 기반 | 파인튜닝 | 비전-언어 |

---

### 4-3. 앞으로 연구 시 고려할 점

#### ① 표현 학습의 추가 개선 여지
현재 본 논문은 **지도 학습(supervised learning)** 방식의 표현 학습을 사용합니다. 자기지도 학습(SSL), 대조 학습, 비전-언어 사전학습을 활용하면 더 일반화된 표현을 학습할 수 있습니다.

#### ② 동적 디커플링
고정된 2단계 파이프라인 대신, 훈련 중 동적으로 표현-분류기 관계를 조정하는 방법을 탐구할 수 있습니다.

#### ③ 오픈셋 및 분포 외 데이터 처리
본 논문은 **닫힌 집합(closed-set)** 시나리오만을 다룹니다. 실제 환경에서는 오픈셋 인식이나 분포 외(out-of-distribution) 데이터 처리가 중요합니다.

#### ④ 다른 태스크로의 확장
이미지 분류 외에 객체 탐지(LVIS 등), 세그멘테이션, NLP 등 다른 태스크에서의 디커플링 효과 검증이 필요합니다.

#### ⑤ 이론적 기반 강화
왜 인스턴스 균형 샘플링이 가장 일반화 가능한 표현을 학습하는지에 대한 **이론적 설명**이 부족합니다. PAC 학습 이론이나 일반화 경계(generalization bound) 관점의 분석이 필요합니다.

#### ⑥ 계산 효율성
대규모 데이터셋에서 2단계 훈련의 계산 비용을 줄이는 효율적인 방법이 필요합니다.

---

## 참고 자료

1. **주 논문**: Kang, B., Xie, S., Rohrbach, M., Yan, Z., Gordo, A., Feng, J., & Kalantidis, Y. (2020). *Decoupling Representation and Classifier for Long-Tailed Recognition*. ICLR 2020. arXiv:1910.09217v2

2. **비교 논문**:
   - Cao, K., Wei, C., Gaidon, A., Arechiga, N., & Ma, T. (2019). *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss*. NeurIPS 2019.
   - Liu, Z., et al. (2019). *Large-Scale Long-Tailed Recognition in an Open World (OLTR)*. CVPR 2019.
   - Cui, Y., et al. (2019). *Class-Balanced Loss Based on Effective Number of Samples*. CVPR 2019.
   - Lin, T.-Y., et al. (2017). *Focal Loss for Dense Object Detection*. ICCV 2017.

3. **2020년 이후 후속 연구**:
   - Zhong, Z., et al. (2021). *Improving Calibration for Long-Tailed Recognition (MiSLAS)*. CVPR 2021.
   - Cui, J., et al. (2021). *Parametric Contrastive Learning (PaCo)*. ICCV 2021.
   - Zhu, J., et al. (2022). *Balanced Contrastive Learning for Long-Tailed Visual Recognition (BCL)*. CVPR 2022.

4. **GitHub 코드**: https://github.com/facebookresearch/classifier-balancing
