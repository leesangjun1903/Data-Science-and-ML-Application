# Class-Balanced Loss Based on Effective Number of Samples

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

기존의 클래스 불균형 해결 방법(역빈도 재가중치 등)은 **데이터 간 중복(overlap)을 고려하지 않는다**는 근본적 한계가 있다. 데이터 수가 증가할수록 **새로운 샘플이 추가적으로 기여하는 정보량은 감소(diminishing marginal benefit)** 하므로, 단순 샘플 수 대신 **유효 샘플 수(Effective Number of Samples)** 를 기반으로 손실을 재가중해야 한다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| 이론적 프레임워크 | 랜덤 커버링(random covering) 이론 기반으로 데이터 중복을 정량화 |
| 유효 샘플 수 공식 | $E_n = \frac{1 - \beta^n}{1 - \beta}$ 형태의 닫힌 해(closed-form) 도출 |
| 범용 손실 함수 설계 | Softmax CE, Sigmoid CE, Focal Loss에 모두 적용 가능한 클래스 균형 항 제안 |
| 실험적 검증 | Long-tailed CIFAR, iNaturalist 2017/2018, ILSVRC 2012에서 성능 향상 확인 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

실제 세계의 데이터셋은 **롱테일 분포(long-tailed distribution)** 를 가진다. 즉, 소수의 클래스가 대부분의 데이터를 차지하고, 다수의 클래스는 극소수의 샘플만 보유한다.

기존 접근법의 한계:
- **재샘플링(re-sampling)**: 오버샘플링은 과적합, 언더샘플링은 정보 손실
- **역빈도 재가중치(inverse class frequency)**: 극단적 불균형 시 성능 저하

$$\alpha_i \propto \frac{1}{n_i} \quad \text{(역빈도 재가중치, 기존 방법)}$$

이 방법은 데이터 간 중복을 전혀 고려하지 않는다는 치명적 결함이 있다.

---

### 2.2 제안 방법: 유효 샘플 수와 클래스 균형 손실

#### (1) 데이터 샘플링을 랜덤 커버링으로 모델링

- 클래스의 모든 가능한 데이터 집합 $\mathcal{S}$의 부피를 $N$으로 정의
- 각 샘플은 단위 부피 1을 가지며, 다른 샘플과 중복될 수 있음
- 새 샘플이 기존 샘플과 중복될 확률: $p = E_{n-1}/N$

#### (2) 유효 샘플 수 공식 (Proposition 1)

귀납법으로 증명된 유효 샘플 수:

$$\boxed{E_n = \frac{1 - \beta^n}{1 - \beta}}, \quad \beta = \frac{N-1}{N} \in [0, 1)$$

**점화식 유도**:

$$E_n = p \cdot E_{n-1} + (1-p)(E_{n-1} + 1) = 1 + \frac{N-1}{N} E_{n-1} \tag{1}$$

귀납 가정 $E_{n-1} = \frac{1-\beta^{n-1}}{1-\beta}$ 을 대입하면:

$$E_n = 1 + \beta \cdot \frac{1-\beta^{n-1}}{1-\beta} = \frac{1-\beta^n}{1-\beta} \tag{2}$$

**기하급수적 해석**:

$$E_n = \sum_{j=1}^{n} \beta^{j-1} \tag{3}$$

즉, $j$번째 샘플은 $\beta^{j-1}$만큼 유효 샘플 수에 기여하며, 중복이 증가할수록 기여도가 지수적으로 감소한다.

**점근적 특성 (Implication 1)**:

$$\lim_{\beta \to 0} E_n = 1 \quad (N=1, \text{ 모든 샘플이 동일한 프로토타입})$$

$$\lim_{\beta \to 1} E_n = n \quad (N \to \infty, \text{ 중복 없음, 유효 수 = 실제 수})$$

#### (3) 클래스 균형 손실 (Class-Balanced Loss)

클래스 $y$에 대한 CB 손실:

$$\text{CB}(\mathbf{p}, y) = \frac{1}{E_{n_y}} \mathcal{L}(\mathbf{p}, y) = \frac{1-\beta}{1-\beta^{n_y}} \mathcal{L}(\mathbf{p}, y) \tag{6}$$

여기서 $n_y$는 정답 클래스 $y$의 학습 샘플 수이다.

**CB Softmax Cross-Entropy Loss**:

$$\text{CB}_{\text{softmax}}(\mathbf{z}, y) = -\frac{1-\beta}{1-\beta^{n_y}} \log\left(\frac{\exp(z_y)}{\sum_{j=1}^{C} \exp(z_j)}\right) \tag{8}$$

**CB Sigmoid Cross-Entropy Loss**:

$$\text{CB}_{\text{sigmoid}}(\mathbf{z}, y) = -\frac{1-\beta}{1-\beta^{n_y}} \sum_{i=1}^{C} \log\left(\frac{1}{1+\exp(-z_i^t)}\right) \tag{11}$$

단, $z_i^t = z_i$ if $i = y$, else $-z_i$.

**CB Focal Loss**:

$$\text{CB}_{\text{focal}}(\mathbf{z}, y) = -\frac{1-\beta}{1-\beta^{n_y}} \sum_{i=1}^{C} (1 - p_i^t)^\gamma \log(p_i^t) \tag{13}$$

---

### 2.3 모델 구조

이 논문은 **새로운 아키텍처를 제안하지 않는다**. 대신 기존 손실 함수에 클래스 균형 가중치 항을 추가하는 방식이다.

- **백본**: ResNet-32 (CIFAR), ResNet-50/101/152 (대규모 데이터셋)
- **학습 프레임워크**: TensorFlow, SGD with momentum
- **Sigmoid 기반 손실의 편향 초기화**:

$$b = -\log\left(\frac{1-\pi}{\pi}\right), \quad \pi = \frac{1}{C}$$

---

### 2.4 성능 향상

#### Long-Tailed CIFAR (Error Rate %, ResNet-32)

| 손실 함수 | CIFAR-10 (불균형 200) | CIFAR-100 (불균형 200) |
|-----------|----------------------|----------------------|
| Softmax | 34.32 | 65.16 |
| Sigmoid | 34.51 | 64.39 |
| Focal ($\gamma$=0.5) | 36.00 | 65.00 |
| **CB Loss** | **31.11** | **63.77** |

#### 대규모 데이터셋 (Top-1 Error %, ResNet-50)

| 데이터셋 | Softmax | CB Focal | 개선 |
|----------|---------|----------|------|
| iNaturalist 2017 | 45.38 | 41.92 | **-3.46%p** |
| iNaturalist 2018 | 42.86 | 38.88 | **-3.98%p** |
| ILSVRC 2012 | 23.92 | 22.71 | **-1.21%p** |

---

### 2.5 한계점

1. **$\beta$ 하이퍼파라미터 튜닝 필요**: 데이터셋 특성(세분화 정도)에 따라 최적 $\beta$ 값이 상이하며, 교차검증이 필요하다.
2. **데이터 분포 가정의 단순성**: 부분 중복(partial overlap)을 고려하지 않는 단순화된 커버링 모델을 사용한다.
3. **클래스 내 분포 동질성 가정**: 클래스 내의 샘플이 균일하게 분포한다고 암묵적으로 가정한다.
4. **표현 학습의 불균형 미해결**: 손실 재가중만으로는 특징 표현(feature representation) 수준의 불균형을 완전히 해소하기 어렵다.
5. **단일 하이퍼파라미터 $N$**: 모든 클래스에 동일한 $N$을 적용하여 클래스 간 개별 특성을 반영하지 못한다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화에 기여하는 핵심 메커니즘

#### (a) 과소표현 클래스의 학습 개선

유효 샘플 수 기반 재가중치는 tail 클래스에 더 높은 가중치를 부여하되, **역빈도보다 완만하게 조정**한다:

$$\beta \to 1: \quad \frac{1-\beta}{1-\beta^{n_y}} \approx \frac{1}{n_y} \quad \text{(역빈도와 동일)}$$

$$\beta \to 0: \quad \frac{1-\beta}{1-\beta^{n_y}} \to 1 \quad \text{(재가중 없음)}$$

$\beta$를 적절히 설정하면 **두 극단 사이의 최적 균형점**을 찾을 수 있어 tail 클래스 일반화가 향상된다.

#### (b) 모델-비종속(model-agnostic) 특성

CB 손실은 모델 구조와 무관하게 적용 가능하여, 더 강력한 백본(예: ViT, Swin Transformer)과 결합 시 추가적인 일반화 향상을 기대할 수 있다.

#### (c) Sigmoid 기반 손실의 일반화 이점

논문에서 Sigmoid CE의 장점으로 언급:
- 클래스 간 상호배타성 가정을 완화 → 실세계 데이터의 모호한 경계 처리 우수
- 단일 레이블과 다중 레이블 분류를 통합 → 범용적 일반화

#### (d) Focal Loss와의 시너지

CB Focal Loss는 다음 두 가지 일반화 기제를 동시에 활용한다:
- **CB 항**: 클래스 수준 불균형 보정 → 클래스 간 일반화
- **Focal 항 $(1-p_i^t)^\gamma$**: 샘플 난이도 기반 집중 학습 → 클래스 내 일반화

#### (e) 균형 잡힌 결정 경계

극단적 재가중치(역빈도)는 head 클래스의 결정 경계를 과도하게 축소시켜 오히려 일반화를 해친다. CB 손실의 완만한 재가중치는 이를 방지한다.

### 3.2 일반화 한계

- **$\beta$ 민감도**: 세분류 데이터셋(fine-grained, CIFAR-100)에서 큰 $\beta$는 성능을 저하시킬 수 있어 일반화 신뢰도가 떨어진다.
- **표현 공간 불균형**: 손실 재가중만으로는 head 클래스에 편향된 특징 공간 자체의 구조를 바꾸지 못한다.

---

## 4. 향후 연구에 미치는 영향과 고려사항

### 4.1 연구에 미치는 영향

#### (a) 롱테일 학습의 이론적 기반 제공

유효 샘플 수라는 개념은 **데이터 다양성을 정량화하는 이론적 틀**을 제공하여, 이후 연구들이 더 정교한 데이터 가치 정량화 방법을 탐색하는 계기가 되었다.

#### (b) 손실 함수 설계 패러다임 전환

단순 역빈도에서 **정보 이론 및 기하학적 관점**의 재가중치 설계로 패러다임이 전환되었다.

#### (c) 광범위한 응용 가능성

의료 영상 진단, 자율주행 희귀 상황 인식, 자연어처리의 희귀 어휘 등 **롱테일 분포가 보편적인 실응용 분야**에 직접 적용 가능한 기반을 마련했다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 후속 연구 비교

#### (a) LDAM-DRW (NeurIPS 2020)
**Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"**

$$\mathcal{L}_{\text{LDAM}} = -\log \frac{e^{z_y - \Delta_y}}{e^{z_y - \Delta_y} + \sum_{j \neq y} e^{z_j}}, \quad \Delta_j = \frac{C}{n_j^{1/4}}$$

| 관점 | CB Loss | LDAM |
|------|---------|------|
| 접근 방식 | 손실 재가중 | 마진(margin) 조정 |
| 이론적 근거 | 유효 샘플 수 | PAC 학습 이론 기반 일반화 경계 |
| tail 클래스 처리 | 가중치 증가 | 결정 경계 마진 증가 |
| 표현 학습 | 간접 영향 | 직접 영향 |

#### (b) BBN (CVPR 2020)
**Zhou et al., "BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition"**

- **주요 아이디어**: 일반 학습 브랜치와 재균형 브랜치를 병렬로 운영하며, 훈련이 진행될수록 재균형 브랜치의 비중을 증가시키는 누적 학습
- **CB Loss 대비 장점**: 표현 학습과 분류기 학습을 분리하여 각각 최적화
- **한계**: 모델 구조가 복잡해짐

#### (c) Decoupling (ICLR 2020)
**Kang et al., "Decoupling Representation and Classifier for Long-Tailed Recognition"**

- **핵심 발견**: 표현 학습은 인스턴스 균형 샘플링으로, 분류기 학습은 클래스 균형 샘플링으로 분리하면 성능이 크게 향상된다.
- **CB Loss와의 관계**: CB Loss를 분류기 재조정 단계에서 활용 가능함을 시사
- **시사점**: CB Loss의 한계(표현 학습 불균형 미해결)를 명시적으로 보완

#### (d) MiSLAS (CVPR 2021)
**Zhong et al., "Improving Calibration for Long-Tailed Recognition"**

- 믹스업(Mixup) 증강과 레이블 스무딩(label smoothing)을 결합하여 롱테일 문제에서 모델 보정(calibration) 개선
- CB Loss가 보정 문제를 충분히 해결하지 못함을 보완

#### (e) Logit Adjustment (ICLR 2021)
**Menon et al., "Long-tail learning via logit adjustment"**

$$\hat{y} = \arg\max_y \left[ f_y(\mathbf{x}) + \tau \log \pi_y \right]$$

- **이론적 강점**: 베이즈 최적 분류기와의 연결성을 엄밀히 증명
- **CB Loss 대비**: 사후 보정(post-hoc correction)으로도 적용 가능하여 더 유연함

#### (f) PaCo (ICCV 2021)
**Cui et al., "Parametric Contrastive Learning"**

- 대조학습(contrastive learning)에 클래스 균형 개념을 통합
- CB Loss의 지도 학습 한계를 자기지도 학습 영역으로 확장

### 5.2 종합 비교표

| 방법 | 연도 | 핵심 아이디어 | CB Loss 대비 장점 | 한계 |
|------|------|--------------|-----------------|------|
| CB Loss | 2019 | 유효 샘플 수 재가중 | 이론적 기반, 범용성 | 표현 학습 미해결 |
| LDAM | 2020 | 마진 조정 | 더 강한 이론 보장 | 구현 복잡 |
| Decoupling | 2020 | 표현/분류기 분리 | 표현 학습 개선 | 2단계 학습 필요 |
| Logit Adj. | 2021 | 로짓 사후 조정 | 베이즈 최적성 | 클래스 사전확률 필요 |
| PaCo | 2021 | 파라메트릭 대조학습 | 자기지도와 통합 | 계산 비용 증가 |

---

## 6. 앞으로 연구 시 고려할 점

### 6.1 방법론적 고려사항

1. **적응형 $\beta$ 추정**: 고정 $\beta$ 대신 각 클래스의 데이터 다양성에 따라 $\beta$를 적응적으로 추정하는 방법 개발
2. **표현 학습과의 통합**: Decoupling 프레임워크와 결합하여 표현 학습과 분류기 학습을 동시에 최적화
3. **대조 학습과의 결합**: SimCLR, MoCo 등 자기지도 학습 패러다임에서 유효 샘플 수 개념 적용
4. **부분 중복 모델링**: 현재의 이진적 중복 가정(전체 내부 또는 전체 외부)을 연속적 중복으로 확장

### 6.2 응용 시 고려사항

1. **도메인별 $N$ 추정**: 의료, 자연어 등 도메인별 데이터 프로토타입 수를 사전 지식으로 활용
2. **동적 불균형 환경**: 온라인 학습이나 클래스 증가 환경에서의 실시간 적응 메커니즘 개발
3. **노이즈 레이블 강건성**: tail 클래스는 노이즈 레이블 비율이 높을 수 있으므로 CB Loss와 강건 학습의 결합 필요

### 6.3 평가 지표 고려사항

단순 전체 정확도 대신 **tail 클래스 정확도, 클래스별 F1 점수, ECE(Expected Calibration Error)** 등을 종합적으로 평가해야 한다.

---

## 참고자료

1. **본 논문**: Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019). *Class-balanced loss based on effective number of samples*. CVPR 2019.
2. Cao, K., et al. (2019). *Learning imbalanced datasets with label-distribution-aware margin loss*. NeurIPS 2019.
3. Kang, B., et al. (2020). *Decoupling representation and classifier for long-tailed recognition*. ICLR 2020.
4. Zhou, B., et al. (2020). *BBN: Bilateral-branch network with cumulative learning for long-tailed visual recognition*. CVPR 2020.
5. Menon, A. K., et al. (2021). *Long-tail learning via logit adjustment*. ICLR 2021.
6. Zhong, Z., et al. (2021). *Improving calibration for long-tailed recognition*. CVPR 2021.
7. Cui, J., et al. (2021). *Parametric contrastive learning*. ICCV 2021.
8. Lin, T. Y., et al. (2017). *Focal loss for dense object detection*. ICCV 2017 / PAMI 2018.
9. He, K., et al. (2016). *Deep residual learning for image recognition*. CVPR 2016.
10. GitHub 코드: https://github.com/richardaecn/class-balanced-loss
