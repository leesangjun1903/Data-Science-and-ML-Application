# Long-Tail Learning via Logit Adjustment

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Menon et al., 2020)의 핵심 주장은 **라벨 빈도(class prior)에 기반한 로짓 조정(logit adjustment)** 이 롱테일 학습 문제를 해결하는 데 있어 통계적으로 가장 견고한 방법이라는 것입니다.

기존의 두 가지 주요 접근법인:
- **사후 가중치 정규화(post-hoc weight normalisation)**: 최적화기(optimizer) 선택에 민감
- **손실 함수 수정(loss modification, 예: LDAM, Equalised Loss)**: Fisher 일관성(Fisher consistency) 결여

...의 한계를 지적하고, **균형 오차(balanced error)** 최소화에 대해 **Fisher 일관성을 보장하는** 두 가지 로짓 조정 기법을 제안합니다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| ① 사후 로짓 조정 | 학습된 모델의 로짓에 클래스 사전확률 기반 오프셋을 사후 적용 |
| ② 로짓 조정 손실 함수 | 학습 중 소프트맥스 크로스엔트로피에 로짓 조정을 내장 |
| ③ 이론적 근거 제공 | Theorem 1을 통한 Fisher 일관성 증명 |
| ④ 통합 프레임워크 | 쌍별 마진 손실(pairwise margin loss)로 기존 방법들을 특수 케이스로 포괄 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

현실 분류 문제에서 레이블 분포 $\mathbb{P}(y)$가 **롱테일(long-tailed)** 구조를 가질 때 두 가지 문제가 발생합니다:

**문제 1: 희소 클래스(tail class)에서의 일반화 실패**
- 소수 샘플로는 충분한 학습이 어려움

**문제 2: 지배 클래스(dominant class)로의 편향**
- 표준 ERM은 다수 클래스에 과도하게 편향됨
- 단순 다수결 분류기도 낮은 분류 오차(misclassification error)를 달성 가능

따라서 논문은 **균형 오차(balanced error, BER)** 를 목표 지표로 설정합니다:

```math
\text{BER}(f) \doteq \frac{1}{L} \sum_{y \in [L]} \mathbb{P}_{x|y}\!\left(y \notin \text{argmax}_{y' \in \mathcal{Y}} f_{y'}(x)\right)
```

이는 각 클래스별 오차율을 균등하게 평균한 값으로, 클래스 불균형에 강인합니다.

---

### 2.2 기존 방법의 한계

#### 한계 1: 가중치 정규화의 불안정성

가중치 정규화 방법(Kang et al., 2020)은 예측 시 다음을 사용합니다:

$$\text{argmax}_{y \in [L]} \frac{f_y(x)}{\nu_y^\tau}, \quad \nu_y = \|w_y\|_2 $$

이 방법은 $\|w_y\|_2 \propto \mathbb{P}(y)$를 가정하지만, **Adam 옵티마이저 사용 시 이 상관관계가 성립하지 않음**이 실험으로 확인됩니다. 또한 로짓이 음수인 경우 정규화가 올바른 순서를 보장하지 못합니다(Fisher 비일관성).

#### 한계 2: 기존 손실 함수 수정의 Fisher 비일관성

**적응형 마진 손실(Cao et al., 2019)**:

$$\ell(y, f(x)) = \log\!\left[1 + \sum_{y' \neq y} e^{\delta_y} \cdot e^{f_{y'}(x) - f_y(x)}\right], \quad \delta_y \propto \mathbb{P}(y)^{-1/4} $$

**균등화 손실(Tan et al., 2020)**:

$$\ell(y, f(x)) = \log\!\left[1 + \sum_{y' \neq y} e^{\delta_{y'}} \cdot e^{f_{y'}(x) - f_y(x)}\right] $$

두 방법 모두 **균형 오차에 대해 Fisher 일관성을 만족하지 못함** — 즉, 충분한 데이터에서도 최적 해에 수렴한다는 이론적 보장이 없습니다.

---

### 2.3 제안 방법 및 수식

#### 통계적 관점에서의 출발점

균형 오차를 위한 베이즈 최적 예측기는 다음과 같습니다:

$$\text{argmax}_{y \in [L]} f^*_y(x) = \text{argmax}_{y \in [L]} \mathbb{P}^{\text{bal}}(y|x) = \text{argmax}_{y \in [L]} \mathbb{P}(x|y) $$

여기서 $\mathbb{P}^{\text{bal}}(y|x) \propto \mathbb{P}(y|x)/\mathbb{P}(y)$입니다. 클래스 확률이 $\mathbb{P}(y|x) \propto \exp(s^*_y(x))$라고 가정하면:

$$\text{argmax}_{y \in [L]} \mathbb{P}^{\text{bal}}(y|x) = \text{argmax}_{y \in [L]} \left[s^*_y(x) - \ln \mathbb{P}(y)\right] $$

이 식이 **두 가지 로짓 조정 방법**의 이론적 근거가 됩니다.

---

#### 방법 1: 사후(Post-hoc) 로짓 조정

표준 소프트맥스 크로스엔트로피로 학습된 모델의 로짓을 사후에 조정합니다:

$$\text{argmax}_{y \in [L]} \exp(w_y^\top \Phi(x)) / \pi_y^\tau = \text{argmax}_{y \in [L]} f_y(x) - \tau \cdot \log \pi_y $$

- $\pi_y$: 훈련 데이터에서 추정된 클래스 사전확률 $\mathbb{P}(y)$
- $\tau > 0$: 온도 스케일링 파라미터 (캘리브레이션 조정)
- $\tau = 1$일 때: 식 (8)의 직접 구현

**핵심 장점**: 가중치 정규화와 달리 **덧셈(additive)** 연산을 수행하므로, 로짓이 음수여도 올바른 순서를 유지합니다.

---

#### 방법 2: 로짓 조정 소프트맥스 크로스엔트로피 손실

학습 중에 로짓 조정을 손실 함수에 내장합니다:

$$\ell(y, f(x)) = -\log \frac{e^{f_y(x) + \tau \cdot \log \pi_y}}{\sum_{y' \in [L]} e^{f_{y'}(x) + \tau \cdot \log \pi_{y'}}} = \log\!\left[1 + \sum_{y' \neq y} \left(\frac{\pi_{y'}}{\pi_y}\right)^\tau \cdot e^{(f_{y'}(x) - f_y(x))}\right] $$

이 손실은 **쌍별 마진 손실(pairwise margin loss)** 의 특수 케이스로 볼 수 있습니다:

$$\ell(y, f(x)) = \alpha_y \cdot \log\!\left[1 + \sum_{y' \neq y} e^{\Delta_{yy'}} \cdot e^{(f_{y'}(x) - f_y(x))}\right] $$

$\tau = 1$일 때: $\alpha_y = 1$, $\Delta_{yy'} = \log\!\left(\frac{\pi_{y'}}{\pi_y}\right)$

이는 **희소 양성 클래스($\pi_y \sim 0$)와 지배 음성 클래스($\pi_{y'} \sim 1$) 사이의 마진**을 크게 확대합니다.

---

#### Theorem 1: Fisher 일관성 조건

**Theorem 1.** 임의의 $\delta \in \mathbb{R}^L_+$에 대해, 쌍별 마진 손실 (11)이 다음 조건을 만족할 때 균형 오차에 대해 Fisher 일관성을 가집니다:

$$\alpha_y = \frac{\delta_y}{\pi_y}, \qquad \Delta_{yy'} = \log\!\left(\frac{\delta_{y'}}{\delta_y}\right)$$

$\delta_y = \pi_y$로 설정하면 $\alpha_y = 1$, $\Delta_{yy'} = \log(\pi_{y'}/\pi_y)$가 되어 **로짓 조정 손실(10)의 일관성이 증명**됩니다.

기존 방법들과의 비교:

| 방법 | $\alpha_y$ | $\Delta_{yy'}$ | Fisher 일관성 |
|------|-----------|----------------|--------------|
| 균형 손실 | $1/\pi_y$ | $0$ | ✓ (마진 없음, 분리 문제에서 비효율) |
| 적응형 마진 (Cao et al.) | $1$ | $\pi_y^{-1/4}$ | ✗ |
| 균등화 손실 (Tan et al.) | $1$ | $\log F(\pi_{y'})$ | ✗ |
| **로짓 조정 손실 (본 논문)** | $1$ | $\log(\pi_{y'}/\pi_y)$ | **✓** |

---

### 2.4 모델 구조

본 논문은 **새로운 아키텍처를 제안하지 않습니다**. 대신 표준 분류 네트워크에 손실 함수 또는 예측 단계만 수정합니다:

```
[입력 x]
    ↓
[특징 추출기 Φ(x)]  ← ResNet-32 (CIFAR) 또는 ResNet-50/152 (ImageNet, iNaturalist)
    ↓
[선형 분류기 w_y^T Φ(x)]
    ↓
[로짓 조정: f_y(x) + τ·log π_y]  ← 학습 중(방법 2) 또는 사후(방법 1)
    ↓
[예측: argmax_y f_y(x)]
```

---

### 2.5 성능 향상

#### 합성 데이터 실험

$\mathbb{P}(y=+1) = 5\%$인 불균형 이진 분류에서, 로짓 조정 손실만이 베이즈 최적 분류기에 근접합니다.

#### 실제 데이터 실험 결과 (균형 오차 %, 낮을수록 좋음)

| 방법 | CIFAR-10-LT | CIFAR-100-LT | ImageNet-LT | iNaturalist |
|------|------------|-------------|------------|------------|
| ERM | 27.16 | 61.64 | 53.11 | 38.66 |
| 가중치 정규화 ($\tau^*$) | 21.50 | 58.76 | 49.37 | 34.10 |
| 적응형 (Cao et al.) | 26.65 | 60.40 | 52.15 | 35.42 |
| 균등화 (Tan et al.) | 26.02 | 57.26 | 54.02 | 38.37 |
| **LA 사후 ($\tau=1$)** | **22.60** | **58.24** | **49.66** | **33.98** |
| **LA 손실 ($\tau=1$)** | **22.33** | **56.11** | **48.89** | **33.64** |

*LA = Logit Adjustment*

#### ResNet-152 + 200에폭 (iNaturalist)

| 방법 | 균형 오차 |
|------|----------|
| LA 손실 + 적응형 마진 조합 | **28.02%** |

---

### 2.6 한계

논문 자체에서 인정하거나 분석을 통해 도출되는 한계는 다음과 같습니다:

1. **클래스 사전확률 추정 의존성**: $\pi_y$를 훈련 데이터의 경험적 빈도로 추정하므로, 훈련/테스트 분포가 다를 경우 성능 저하 가능
2. **온도 파라미터 $\tau$ 튜닝 필요**: 최적 $\tau$는 데이터셋마다 다르며 추가 검증 비용 발생
3. **분리 가능한 학습 문제에서 균형 손실의 한계**: Byrd & Lipton(2019)이 지적했듯, 손실 가중치만으로는 분리 가능 설정에서 효과가 제한적
4. **데이터 증강과의 결합 미탐색**: 논문에서 향후 연구로 남겨둠
5. **Step 프로파일 CIFAR-100에서 가중치 정규화에 소폭 열위**: $\tau=1$ 고정 시 Step-100 CIFAR-100-LT에서 가중치 정규화(55.19)가 소폭 우세

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Fisher 일관성과 일반화의 관계

Fisher 일관성은 **무한 샘플 극한에서 손실 최소화가 베이즈 최적 예측기에 수렴함을 보장**합니다. 이는 일반화의 필요 조건으로, 기존 방법들이 이를 결여할 때 발생하는 구조적 편향을 제거합니다.

수식으로 표현하면, 로짓 조정 손실의 최소화는:

$$f^*(x) = \text{argmax}_{y} \left[\log \eta_y(x) - \log \pi_y\right] = \text{argmax}_{y} \frac{\eta_y(x)}{\pi_y}$$

여기서 $\eta_y(x) = \mathbb{P}(y|x)$이며, 이는 균형 오차를 위한 진정한 베이즈 최적 해입니다.

### 3.2 상대적 마진과 일반화

로짓 조정 손실에서 $\Delta_{yy'} = \log(\pi_{y'}/\pi_y)$는 **희소 클래스(양성)와 지배 클래스(음성) 사이의 상대적 마진**을 확대합니다. 마진 이론에 따르면 이는 해당 클래스 쌍에 대한 일반화 경계를 개선합니다.

직관적으로:
- $\pi_y \ll 1$ (희소 클래스): $\log(\pi_{y'}/\pi_y) \gg 0$ → 희소 클래스가 지배 클래스를 누르지 않도록 큰 마진 강제
- $\pi_y \approx \pi_{y'}$: $\Delta_{yy'} \approx 0$ → 유사 빈도 클래스 간 불필요한 마진 추가 없음

### 3.3 캘리브레이션과 일반화

사후 로짓 조정에서 $\tau \neq 1$는 모델의 **확률 캘리브레이션(calibration)** 불량을 보정합니다. 잘 캘리브레이션된 예측기는 일반적으로 더 나은 일반화 성능을 보입니다. 논문은 온도 스케일링을 통한 캘리브레이션 후 로짓 조정 적용을 권장합니다.

### 3.4 실험적 증거

- **Figure 4**: 로짓 조정 손실이 **모든 빈도 그룹(Many/Medium/Few)에서** 일관된 성능 향상을 보임
- **표 4의 ResNet-152 + 200에폭 결과**: 더 복잡한 아키텍처와 더 긴 학습에서도 로짓 조정이 일관적으로 유리 → 표현력 향상 시 이점이 더욱 두드러짐
- **적응형 마진과의 결합**: $\Delta_{yy'} = \log\frac{\pi_{y'}}{\pi_y} + \frac{1}{\pi_y^{1/4}}$로 두 마진을 결합할 때 iNaturalist에서 28.02%로 추가 개선 → 상보적 일반화 이득

### 3.5 DRW(Deferred Re-Weighting)와의 결합 가능성

Cao et al.(2019)의 DRW 기법은 초기 표준 학습 후 재가중치를 적용하는 방식으로, 표현 학습과 분류기 조정을 분리합니다. 논문은 로짓 조정과 DRW 결합이 추가적인 일반화 이득을 줄 가능성을 시사합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### 이론적 영향
1. **통합 프레임워크 제공**: 쌍별 마진 손실 (11)이 기존 방법들을 특수 케이스로 포괄하는 통합 이론을 제시, 새로운 손실 설계의 기준점 마련
2. **Fisher 일관성의 중요성 부각**: 롱테일 학습에서 통계적 일관성이 실증적 성능과 강하게 연결됨을 보임
3. **최적화기 불변성 요구**: 가중치 정규화의 최적화기 의존성 문제를 지적, 손실 수준에서의 해결책으로 연구 방향 전환

#### 실용적 영향
1. **구현 단순성**: 기존 학습 파이프라인에 최소한의 수정(로짓에 $\tau \cdot \log \pi_y$ 추가)으로 적용 가능
2. **표현 학습과 분류기의 분리**: 사후 로짓 조정의 강력한 성능이 Zhang et al.(2019)의 표현-분류기 분리 학습 패러다임을 지지
3. **비전 이외의 도메인 확장**: 자연어처리, 추천 시스템 등 클래스 불균형이 존재하는 모든 분류 문제에 적용 가능

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래는 본 논문 이후 등장한 주요 관련 연구들입니다. 단, 저는 2021년까지의 공개 논문에 대해 신뢰도 높은 정보를 가지고 있으며, 그 이후 논문들에 대해서는 알려진 범위 내에서 기술합니다.

#### ① LADE (Disentangling Label Distribution for Long-tailed Recognition, Hong et al., CVPR 2021)

**핵심 아이디어**: 훈련 분포와 테스트 분포가 다를 수 있음을 명시적으로 모델링. 레이블 분포 불일치(label distribution shift)를 고려한 보정.

**로짓 조정과의 관계**: 로짓 조정은 훈련 분포의 $\pi_y$를 고정적으로 사용하지만, LADE는 테스트 시 다른 분포를 상정. **로짓 조정이 훈련/테스트 분포 일치를 가정**한다는 한계를 보완.

#### ② DisAlign (Distribution Alignment, Zhang et al., CVPR 2021)

**핵심 아이디어**: 학습 가능한 적응형 캘리브레이션 함수를 통해 분포 정렬. 고정된 $\tau$가 아닌 데이터로부터 학습된 스케일링.

**로짓 조정과의 관계**: 로짓 조정의 $\tau$ 선택 문제를 데이터 기반으로 자동화. 로짓 조정의 직접적 확장으로 볼 수 있음.

#### ③ PaCo (Parametric Contrastive Learning, Cui et al., ICCV 2021)

**핵심 아이디어**: 지도 대조 학습(supervised contrastive learning)에 파라메트릭 클래스 대표 벡터를 결합하여 롱테일 표현 학습 개선.

**로짓 조정과의 관계**: PaCo는 표현 학습 단계를 개선하며, 분류기 단계에서 로짓 조정과 결합 가능. **표현 학습 + 로짓 조정의 상보적 결합**을 시사.

#### ④ MiSLAS (Improving Calibration for Long-Tail Recognition, Zhong et al., CVPR 2021)

**핵심 아이디어**: 레이블 스무딩(label smoothing)과 혼합(mixup)을 통한 캘리브레이션 개선, 단계별 학습 전략.

**로짓 조정과의 관계**: 로짓 조정이 $\tau$ 선택에 캘리브레이션 품질에 의존한다는 점에서, MiSLAS의 캘리브레이션 개선이 로짓 조정의 효과를 증폭시킬 수 있음.

#### ⑤ VPT/Prompt 기반 방법들 (2022~)

**핵심 아이디어**: 사전학습된 대규모 모델(ViT 등)에 프롬프트를 추가하여 롱테일 문제를 해결.

**로짓 조정과의 관계**: 강력한 표현력을 가진 기반 모델 위에서 로짓 조정을 사후 적용하면 추가 이득이 기대됨. 그러나 사전학습 데이터의 분포가 이미 불균형하거나 미지인 경우 $\pi_y$ 추정의 어려움이 있음.

#### 비교 요약표

| 논문 | 접근 방식 | 로짓 조정과의 관계 | Fisher 일관성 |
|------|----------|-------------------|--------------|
| Logit Adjustment (본 논문) | 손실/사후 조정 | 기준 | ✓ |
| LADE (Hong et al., 2021) | 분포 불일치 보정 | 확장 | 분포 설정 의존 |
| DisAlign (Zhang et al., 2021) | 학습형 캘리브레이션 | 자동화된 확장 | 부분적 |
| PaCo (Cui et al., 2021) | 대조 학습 | 상보적 결합 | 해당 없음 |
| MiSLAS (Zhong et al., 2021) | 캘리브레이션 개선 | 선행 조건 개선 | 해당 없음 |

---

### 4.3 앞으로 연구 시 고려할 점

#### 이론적 고려사항

1. **비정상(non-stationary) 분포에서의 일관성**
   - 현재 이론은 훈련/테스트 분포가 동일함을 가정
   - 분포 이동(distribution shift) 하에서의 로짓 조정의 행동 분석 필요
   - 수식: 테스트 시 사전확률 $\pi_y^{\text{test}} \neq \pi_y^{\text{train}}$인 경우 조정 값 재설계 필요

2. **다중 레이블 및 계층적 분류로의 확장**
   - 현재는 단일 레이블 분류에 국한
   - 객체 탐지, 이미지 세그멘테이션 등 복잡한 구조에서의 $\pi_y$ 정의 문제

3. **노이즈 레이블(noisy label) 환경에서의 강인성**
   - 롱테일 데이터에서는 레이블 노이즈가 흔히 발생
   - Fisher 일관성이 레이블 노이즈 하에서 어떻게 수정되어야 하는지 연구 필요

#### 실용적 고려사항

4. **온라인/점진적 학습(online/incremental learning)과의 결합**
   - 새로운 클래스가 순차적으로 추가되는 환경에서 $\pi_y$의 동적 업데이트 방법

5. **$\tau$ 자동 선택**
   - 현재는 검증 셋에서 수동으로 튜닝
   - 메타 학습 또는 베이지안 최적화를 통한 자동화

6. **대규모 사전학습 모델(LLM, ViT)과의 통합**
   - GPT, CLIP, ViT 등을 파인튜닝할 때 로짓 조정 적용 전략
   - 사전학습 데이터의 클래스 분포 $\pi_y^{\text{pretrain}}$과 다운스트림 $\pi_y^{\text{finetune}}$ 간의 상호작용

7. **자기지도 학습(self-supervised learning)과의 결합**
   - SimCLR, MoCo 등으로 학습된 표현에 로짓 조정을 적용할 때의 최적 전략

8. **공정성(fairness) 연구와의 연결**
   - 소수 그룹에 대한 균등한 성능 보장 문제는 롱테일 학습과 구조적으로 유사
   - 균형 오차 최소화가 공정성 지표(예: equalized odds)와 어떻게 연결되는지 탐구

---

## 참고문헌 및 출처

**주요 논문 (본 문서에서 직접 인용)**

- **Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S. (2020).** *Long-Tail Learning via Logit Adjustment.* arXiv:2007.07314 [cs.LG]. (본 분석의 주요 대상 논문)
- **Kang, B., Xie, S., Rohrbach, M., Yan, Z., Gordo, A., Feng, J., & Kalantidis, Y. (2020).** *Decoupling Representation and Classifier for Long-Tailed Recognition.* ICLR 2020.
- **Cao, K., Wei, C., Gaidon, A., Arechiga, N., & Ma, T. (2019).** *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss.* NeurIPS 2019.
- **Tan, J., Wang, C., Li, B., Li, Q., Ouyang, W., Yin, C., & Yan, J. (2020).** *Equalization Loss for Long-Tailed Object Recognition.* CVPR 2020.
- **Byrd, J., & Lipton, Z. C. (2019).** *What is the Effect of Importance Weighting in Deep Learning?* ICML 2019.
- **Lin, Y. (2004).** *A note on margin-based loss functions in classification.* Statistics & Probability Letters.
- **Bartlett, P. L., Jordan, M. I., & McAuliffe, J. D. (2006).** *Convexity, classification, and risk bounds.* JASA.

**2020년 이후 비교 연구 (공개 논문 기반)**

- **Hong, Y., Han, S., Choi, K., Seo, S., Kim, B., & Chang, B. (2021).** *Disentangling Label Distribution for Long-tailed Visual Recognition.* CVPR 2021.
- **Zhang, S., Li, Z., Yan, S., He, X., & Sun, J. (2021).** *Distribution Alignment: A Unified Framework for Long-tail Visual Recognition.* CVPR 2021.
- **Cui, J., Zhong, Z., Liu, S., Yu, B., & Jia, J. (2021).** *Parametric Contrastive Learning.* ICCV 2021.
- **Zhong, Z., Cui, J., Liu, S., & Jia, J. (2021).** *Improving Calibration for Long-Tail Recognition.* CVPR 2021.
