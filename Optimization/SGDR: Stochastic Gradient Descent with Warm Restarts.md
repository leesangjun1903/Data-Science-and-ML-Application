# SGDR: Stochastic Gradient Descent with Warm Restarts

---

## 1. 핵심 주장과 주요 기여 (요약)

**핵심 주장:** 심층 신경망 학습 시 SGD의 학습률(learning rate)을 주기적으로 재설정(warm restart)하면서 코사인 어닐링(cosine annealing)으로 감소시키는 간단한 스케줄링 기법이, 기존의 고정 간격 학습률 감소(step decay) 방식보다 **빠른 수렴(anytime performance)**과 **더 나은 일반화 성능**을 동시에 달성할 수 있다.

**주요 기여:**
1. **Cosine annealing with warm restarts** 학습률 스케줄 제안 — 기존 step decay 대비 하이퍼파라미터가 적고 구현이 단순함
2. CIFAR-10 (3.14%), CIFAR-100 (16.21%)에서 당시 새로운 **state-of-the-art** 달성
3. 학습 과정 중 restart 직전 생성되는 **snapshot 모델들을 앙상블**로 활용하여 추가 비용 없이 성능 향상 (Huang et al., 2016a의 후속 연구와 결합)
4. EEG 데이터셋, 다운샘플된 ImageNet 등 **다양한 도메인**에서의 효과 검증
5. 기존 학습률 스케줄 대비 **2~4배 빠르게** 동등하거나 더 나은 성능 도달

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

심층 신경망 학습의 핵심 병목은 **학습률 스케줄링**이다. 기존 방식은 학습률을 고정 상수로 시작한 뒤, 사전에 정해진 에포크(예: 60, 120, 160)에서 고정 비율로 감소시키는 **step decay** 방식을 사용했다. 이 방식에는 다음과 같은 문제가 있다:

- **사전에 총 학습 에포크 수를 정해야** 하며, 학습률 감소 시점을 수동으로 설정해야 함
- **Anytime performance가 나쁨** — 학습 초기에는 좋은 모델을 빠르게 얻기 어려움
- **과적합(overfitting) 경향** — 학습 후반에 학습률이 지나치게 작아지면 training loss만 줄어들고 test error는 증가하는 현상 발생
- 손실 함수 지형(loss landscape)의 다양한 지역 최적해(local optima)를 탐색하기 어려움

### 2.2 제안하는 방법 (수식 포함)

#### 기본 SGD with Momentum

$$\boldsymbol{v}_{t+1} = \mu_t \boldsymbol{v}_t - \eta_t \nabla f_t(\boldsymbol{x}_t)$$

$$\boldsymbol{x}_{t+1} = \boldsymbol{x}_t + \boldsymbol{v}_{t+1}$$

여기서 $\boldsymbol{v}_t$는 속도 벡터(초기값 $\boldsymbol{0}$), $\eta_t$는 학습률, $\mu_t$는 모멘텀 계수이다.

#### Cosine Annealing with Warm Restarts (SGDR 핵심 수식)

$i$번째 실행(run) 내에서, 각 배치(batch)마다 학습률을 다음과 같이 코사인 어닐링으로 감소시킨다:

$$\eta_t = \eta_{\min}^{i} + \frac{1}{2}\left(\eta_{\max}^{i} - \eta_{\min}^{i}\right)\left(1 + \cos\left(\frac{T_{cur}}{T_i}\pi\right)\right)$$

| 기호 | 의미 |
|------|------|
| $\eta_{\max}^{i}$ | $i$번째 실행에서의 최대 학습률 |
| $\eta_{\min}^{i}$ | $i$번째 실행에서의 최소 학습률 |
| $T_{cur}$ | 마지막 재시작(restart) 이후 경과한 에포크 수 (배치 단위로 갱신되므로 0.1, 0.2 등 이산화된 값 가능) |
| $T_i$ | $i$번째 실행의 총 에포크 수 |

**동작 원리:**
- $T_{cur} = 0$일 때: $\cos(0) = 1$이므로 $\eta_t = \eta_{\max}^{i}$ (최대 학습률)
- $T_{cur} = T_i$일 때: $\cos(\pi) = -1$이므로 $\eta_t = \eta_{\min}^{i}$ (최소 학습률)
- 한 주기가 끝나면 학습률을 다시 $\eta_{\max}^{i}$로 올려 **warm restart**를 수행

#### 주기 증가 전략 (Increasing Period)

Anytime performance를 향상시키기 위해, 초기에는 짧은 주기 $T_0$로 시작하고 매 restart마다 $T_{mult}$ 배로 주기를 늘린다:

$$T_i = T_0 \cdot T_{mult}^{i}$$

예를 들어, $T_0 = 10, T_{mult} = 2$이면 주기가 10 → 20 → 40 → 80 → ... 에포크로 증가한다.

#### Warm Restart의 핵심 특징
- 파라미터 $\boldsymbol{x}_t$를 **초기화하지 않고** 이전 값을 유지한 채 학습률만 증가시킴 (cold restart가 아닌 **warm** restart)
- 모멘텀 등 이전에 축적된 정보를 일정 부분 활용 가능
- 재시작 직후 일시적으로 성능이 악화되지만, 이후 빠르게 새로운 (잠재적으로 더 나은) 지역 최적해로 수렴

### 2.3 모델 구조

논문 자체는 새로운 모델 아키텍처를 제안하지 않는다. 대신 기존의 **Wide Residual Networks (WRN)**(Zagoruyko & Komodakis, 2016)을 실험 모델로 사용한다:

- **WRN- $d$ - $k$ **: 깊이(depth) $d$, 너비 배수(widening factor) $k$
- 주요 실험 모델: **WRN-28-10** (36.5M 파라미터), **WRN-28-20** (145.8M 파라미터)
- 훈련 설정: SGD with momentum (μ=0.9), weight decay 0.0005, 미니배치 128, 초기 학습률 $\eta_0 = 0.05$

### 2.4 성능 향상

#### 단일 모델 결과 (Table 1 기준)

| 설정 | CIFAR-10 | CIFAR-100 |
|------|----------|-----------|
| WRN-28-10 Default ($\eta_0=0.05$) | 4.13% | 20.21% |
| WRN-28-10 SGDR ($T_0=10, T_{mult}=2$) | 4.03% | 19.58% |
| WRN-28-10 SGDR ($T_0=200, T_{mult}=1$) | 3.86% | 19.98% |
| WRN-28-20 SGDR ($T_0=10, T_{mult}=2$) | **3.74%** | **18.70%** |

#### 앙상블 결과 (Table 2 기준)

| 설정 | CIFAR-10 | CIFAR-100 |
|------|----------|-----------|
| 1 run, 1 snapshot | 4.03% | 19.57% |
| 1 run, 3 snapshots | 3.51% | 17.75% |
| 3 runs, 3 snapshots/run | 3.25% | 16.64% |
| 16 runs, 3 snapshots/run | **3.14%** | **16.21%** |

#### Anytime Performance
- SGDR($T_0=1, T_{mult}=2$ 또는 $T_0=10, T_{mult}=2$)은 기존 step decay 대비 **2~4배 적은 에포크**로 동등하거나 더 나은 test error 도달
- 특히 CIFAR-100에서 WRN-28-20 + SGDR은 50 에포크만에 19% 미만의 test error를 달성 (기존 WRN-28-10은 200 에포크에도 19.5% 이상)

### 2.5 한계

1. **이론적 분석 부재**: SGDR의 일반화 성능 향상에 대한 이론적 설명이 없으며, 순수히 경험적(empirical) 결과에 의존함
2. **하이퍼파라미터 추가**: $T_0$, $T_{mult}$, $\eta_{\max}^{i}$, $\eta_{\min}^{i}$ 등 새로운 하이퍼파라미터가 도입됨 (저자들은 단순화를 위해 $\eta_{\max}^{i}$, $\eta_{\min}^{i}$를 고정하지만, 최적 설정은 문제에 따라 다를 수 있음)
3. **대규모 데이터셋 검증 부족**: 본 논문에서 full-scale ImageNet에 대한 실험이 없으며, 다운샘플된 32×32 ImageNet만 예비 실험으로 제시
4. **다중 모달리티 주장의 부재**: 저자들 스스로 "multi-modality 관련 효과를 관찰했다고 주장하지 않는다"라고 명시
5. **Adam, AdaDelta 등 다른 옵티마이저와의 결합이 미탐구**: SGD에만 적용되었으며, 다른 적응적 옵티마이저와의 결합은 향후 과제로 남김

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 과적합 방지 효과

논문의 Figure 7에서 핵심적인 관찰이 제시된다:

- **기존 step decay 스케줄**: 약 120 에포크 이후 training loss는 계속 감소하지만 test cross-entropy loss와 test error가 **증가** → 명확한 **과적합(overfitting)** 징후
- **SGDR**: 매우 경미한(mild) 과적합만 관찰됨

이는 SGDR이 학습률을 주기적으로 증가시킴으로써 다음과 같은 **암묵적 정규화(implicit regularization)** 효과를 제공하기 때문으로 해석된다:

1. **Sharp minima로부터의 탈출**: 학습률이 갑자기 증가하면, 현재의 좁고 깊은(sharp) 지역 최적해에서 벗어나 **더 넓고 평탄한(flat) 지역 최적해**로 이동할 가능성이 높아진다. 일반적으로 flat minima는 더 나은 일반화 성능과 관련이 있다는 연구들이 있다 (Keskar et al., 2017; Hochreiter & Schmidhuber, 1997).

2. **학습률 범위 스캐닝**: SGDR은 각 주기에서 $\eta_{\max}$부터 $\eta_{\min}$까지 학습률을 연속적으로 스캔하므로, 초기 학습률 선택의 문제를 완화한다. 저자들은 이를 다운샘플된 ImageNet 실험에서 관찰하며, "SGDR reduces the problem of improper selection of the [initial learning rate] by scanning/annealing from the initial learning rate to 0"이라고 기술한다.

3. **코사인 어닐링의 부드러운 감소**: Step decay 방식의 급격한 학습률 감소와 달리, 코사인 함수 기반의 부드러운 감소는 학습 과정에서 더 안정적인 수렴을 유도한다.

### 3.2 앙상블을 통한 일반화 향상

SGDR의 각 restart 직전에 생성되는 snapshot 모델들은 **서로 다른 지역 최적해(local optima)**에 수렴한 모델들이다. 이러한 다양성(diversity)은 앙상블 학습에서 핵심적인 요소이며:

- 단일 모델 대비 앙상블로 **0.5~3.4% 포인트의 추가 성능 향상** 달성
- 동일 에포크에서 인접한 snapshot (예: 148, 149, 150)은 다양성이 부족하여 앙상블 효과가 없었으나, restart 직전의 snapshot (30, 70, 150)은 충분한 다양성을 제공
- Huang et al. (2016a)의 "Snapshot Ensembles: Train 1, Get M for Free"와 자연스럽게 결합

### 3.3 더 큰 모델 학습 가능성

SGDR의 빠른 수렴은 동일 시간 내 **더 큰 모델을 학습**할 수 있는 기회를 제공한다:
- WRN-28-20 (145.8M)은 WRN-28-10 (36.5M) 대비 3~4배 더 많은 연산이 필요하지만, SGDR의 aggressive schedule 덕분에 WRN-28-10의 200 에포크와 동일한 시간 내에 **더 나은 성능** 달성 가능
- 이를 통해 모델 용량 증가에 의한 표현력 향상과 SGDR에 의한 정규화 효과가 결합되어 일반화 성능이 향상됨

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (1) 학습률 스케줄링 패러다임 전환
SGDR 이후, cosine annealing은 심층 학습의 **사실상 표준(de facto standard)** 학습률 스케줄이 되었다. PyTorch의 `CosineAnnealingLR`, `CosineAnnealingWarmRestarts` 등의 구현이 표준 라이브러리에 포함되어 있으며, 대부분의 최신 연구에서 채택되고 있다.

#### (2) AdamW의 탄생
본 논문의 저자인 Loshchilov & Hutter는 이 연구를 확장하여 **AdamW** (Loshchilov & Hutter, 2019, "Decoupled Weight Decay Regularization")를 제안했다. AdamW는 Adam 옵티마이저에서 weight decay와 L2 정규화를 분리(decouple)하고, cosine annealing with warm restarts를 결합한 방법으로, 현재 **Transformer 계열 모델 학습의 표준 옵티마이저**가 되었다.

#### (3) Snapshot Ensemble 및 Stochastic Weight Averaging
- **Snapshot Ensembles** (Huang et al., 2017): SGDR의 cosine annealing 주기를 활용하여 무료 앙상블 구축
- **Stochastic Weight Averaging (SWA)** (Izmailov et al., 2018): cyclical/constant 학습률로 생성된 모델들의 가중치를 평균하여 일반화 성능을 향상

#### (4) 대규모 사전학습 모델
GPT, BERT, ViT 등의 대규모 모델 학습에서 cosine annealing schedule은 거의 표준으로 사용되며, 특히 warm-up 구간과 결합한 **linear warm-up + cosine decay** 패턴이 보편적이다.

### 4.2 앞으로 연구 시 고려할 점

1. **Warm restart 주기의 자동 결정**: $T_0$와 $T_{mult}$의 최적 값은 문제에 따라 다르므로, 이를 자동으로 결정하는 적응적(adaptive) 방법이 필요
2. **학습률 범위의 적응적 조정**: 매 restart마다 $\eta_{\max}^{i}$와 $\eta_{\min}^{i}$를 감소시키는 것이 유효할 수 있으며, 이에 대한 체계적 연구 필요
3. **Loss landscape 분석과의 연결**: 왜 warm restart가 일반화를 향상시키는지에 대한 이론적 분석 (flat minima, loss surface geometry 관점)
4. **대규모 분산 학습과의 호환성**: 다수의 GPU/TPU에서의 분산 학습 시 warm restart의 동기화 및 배치 크기와의 상호작용 고려
5. **다른 옵티마이저와의 결합**: SGD 외에 Adam, LAMB, AdaFactor 등과의 결합 시 warm restart의 효과 및 최적 설정 탐구

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 후속 연구 비교표

| 연구 | 연도 | 핵심 아이디어 | SGDR과의 관계 |
|------|------|-------------|-------------|
| **Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW)** | 2019 (ICLR) | Adam에서 weight decay를 분리하고 cosine annealing 적용 | SGDR의 직접적 확장; warm restart를 Adam에 결합 |
| **Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization" (SWA)** | 2018 (UAI) | 학습 후반부의 cyclical LR로 탐색한 가중치들을 평균 | SGDR의 cyclical LR을 가중치 평균에 활용 |
| **Gotmare et al., "A Closer Look at Deep Learning Heuristics: Learning rate restarts, Warmup and Distillation"** | 2019 (ICLR) | Warm restart의 효과 분석: warm restart 없이도 cosine annealing 자체가 유효 | SGDR의 restart 대비 단일 cosine decay의 효과를 심층 분석 |
| **Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)** | 2020 (ICML) | 대조 학습에서 cosine decay schedule 사용 | SGDR의 cosine schedule을 자기지도 학습에 적용 |
| **Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT)** | 2021 (ICLR) | Vision Transformer 학습 시 linear warmup + cosine decay 표준 사용 | SGDR의 cosine schedule이 Transformer 학습의 표준으로 확립 |
| **Touvron et al., "Training data-efficient image transformers & distillation through attention" (DeiT)** | 2021 (ICML) | ViT 효율적 학습에서 cosine schedule + repeated augmentation 사용 | Cosine schedule의 보편적 채택 확인 |
| **Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time"** | 2022 (ICML) | 여러 fine-tuned 모델의 가중치를 평균하여 성능 향상 | SGDR의 snapshot ensemble 아이디어를 fine-tuning 시나리오에 확장 |
| **Li et al., "Exponential Moving Average of Weights in Deep Learning" (EMA)** | 2023 | 학습 중 파라미터의 지수 이동 평균이 일반화 향상 | SGDR의 snapshot 기반 앙상블과 유사한 동기; 가중치 공간에서의 평균화 |
| **Defazio & Mishchenko, "Learning-Rate-Free Learning by D-Adaptation"** | 2023 (ICML) | 학습률을 자동으로 결정하는 옵티마이저 | SGDR이 남긴 "학습률 선택 문제"를 근본적으로 해결하려는 시도 |

### 5.2 심층 비교 분석

#### (1) SWA vs. SGDR
**SWA (Stochastic Weight Averaging)** (Izmailov et al., 2018)는 SGDR의 아이디어를 발전시켜, 학습 후반부에 cyclical 또는 constant 높은 학습률로 탐색한 여러 모델의 **가중치를 직접 평균**한다.

- SGDR은 snapshot 모델들의 **prediction(softmax output)을 앙상블**하지만, SWA는 **가중치 자체를 평균**
- SWA는 더 넓은(flat) 최적해로 수렴함이 이론적/실험적으로 확인됨
- SWA는 추론 시 **단일 모델**만 필요하므로 앙상블 대비 추론 비용이 없음
- SWAG (SWA-Gaussian, Maddox et al., 2019)으로 확장되어 불확실성 추정에도 활용

#### (2) Cosine Schedule의 보편화
2020년 이후 거의 모든 주요 연구에서 **linear warmup + cosine decay**가 표준 스케줄로 사용된다:

$$\eta_t = \begin{cases} \eta_{\max} \cdot \frac{t}{T_{warmup}} & \text{if } t < T_{warmup} \\ \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - T_{warmup}}{T_{total} - T_{warmup}} \pi\right)\right) & \text{if } t \geq T_{warmup} \end{cases}$$

이는 SGDR의 cosine annealing에 warm-up을 추가한 형태로, BERT (Devlin et al., 2019), GPT-3 (Brown et al., 2020), ViT (Dosovitskiy et al., 2021) 등에서 표준적으로 사용된다.

#### (3) AdamW의 지배적 위치
Loshchilov & Hutter가 SGDR의 후속으로 제안한 **AdamW**는:
- Adam의 적응적 학습률 + SGDR의 cosine annealing + 분리된 weight decay를 결합
- 2020년 이후 Transformer 기반 모델(BERT, GPT, ViT 등)의 **사실상 표준 옵티마이저**
- PyTorch `torch.optim.AdamW`로 기본 제공

#### (4) 학습률-free 옵티마이저의 등장
2023년 이후, 학습률 자체를 자동으로 결정하려는 연구들이 등장:
- **D-Adaptation** (Defazio & Mishchenko, 2023): 학습률을 $D$(초기점과 최적해 사이 거리)의 추정을 통해 자동 결정
- **Prodigy** (Mishchenko & Defazio, 2023): D-Adaptation의 개선 버전
- 이러한 연구들은 SGDR이 남긴 "최적 $\eta_{\max}$ 선택 문제"를 근본적으로 해결하고자 함

#### (5) Model Soups (2022)와 Snapshot Ensemble의 발전
Wortsman et al. (2022)의 **Model Soups**는:
- SGDR의 snapshot ensemble 아이디어를 fine-tuning 시나리오로 확장
- 동일 사전학습 모델에서 서로 다른 하이퍼파라미터로 fine-tuning한 모델들의 **가중치를 평균**
- 추론 비용 증가 없이 성능 향상 (SGDR의 앙상블은 추론 비용이 M배 증가)
- CLIP 모델에서 ImageNet 정확도 향상 검증

---

## 참고 자료

1. **Loshchilov, I. & Hutter, F.** "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR 2017.* arXiv:1608.03983v5.
2. **Loshchilov, I. & Hutter, F.** "Decoupled Weight Decay Regularization." *ICLR 2019.* arXiv:1711.05101.
3. **Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J.E. & Weinberger, K.Q.** "Snapshot Ensembles: Train 1, Get M for Free." *ICLR 2017.* arXiv:1704.00109.
4. **Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D. & Wilson, A.G.** "Averaging Weights Leads to Wider Optima and Better Generalization." *UAI 2018.* arXiv:1803.05407.
5. **Chen, T., Kornblith, S., Norouzi, M. & Hinton, G.** "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)." *ICML 2020.* arXiv:2002.05709.
6. **Dosovitskiy, A. et al.** "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)." *ICLR 2021.* arXiv:2010.11929.
7. **Touvron, H. et al.** "Training data-efficient image transformers & distillation through attention (DeiT)." *ICML 2021.* arXiv:2012.12877.
8. **Wortsman, M. et al.** "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time." *ICML 2022.* arXiv:2203.05482.
9. **Defazio, A. & Mishchenko, K.** "Learning-Rate-Free Learning by D-Adaptation." *ICML 2023.* arXiv:2301.07733.
10. **Gotmare, A., Keskar, N.S., Xiong, C. & Socher, R.** "A Closer Look at Deep Learning Heuristics: Learning rate restarts, Warmup and Distillation." *ICLR 2019.* arXiv:1810.13243.
11. **Keskar, N.S., Mudigere, D., Nocedal, J., Smelyanskiy, M. & Tang, P.T.P.** "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima." *ICLR 2017.* arXiv:1609.04836.
12. **Zagoruyko, S. & Komodakis, N.** "Wide Residual Networks." *BMVC 2016.* arXiv:1605.07146.
13. **Smith, L.N.** "Cyclical Learning Rates for Training Neural Networks." *WACV 2017.* arXiv:1506.01186.
14. **Maddox, W.J., Izmailov, P., Garipov, T., Vetrov, D.P. & Wilson, A.G.** "A Simple Baseline for Bayesian Uncertainty in Deep Learning (SWAG)." *NeurIPS 2019.* arXiv:1902.02476.
