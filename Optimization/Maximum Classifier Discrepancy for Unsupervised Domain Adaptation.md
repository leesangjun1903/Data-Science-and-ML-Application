# Maximum Classifier Discrepancy (MCD) for Unsupervised Domain Adaptation 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

기존의 도메인 적응(Domain Adaptation, DA) 방법들은 **도메인 분류기(domain classifier)** 를 이용하여 소스와 타겟 분포를 맞추는 방식을 사용한다. 그러나 이 방식은 두 가지 근본적인 문제를 갖는다:

1. **클래스 경계를 고려하지 않음**: 도메인 분류기는 단순히 소스/타겟 구분만 학습하여, 생성된 피처가 클래스 경계 근방의 모호한 영역에 위치할 수 있음.
2. **완전한 분포 매칭의 어려움**: 각 도메인 고유의 특성으로 인해 분포를 완전히 일치시키는 것은 비현실적임.

이 논문은 **두 개의 태스크 특화 분류기(task-specific classifiers) 간의 불일치(discrepancy)를 최대화/최소화하는 적대적 학습**을 통해 위 문제를 해결하고자 한다.

### 주요 기여

- **새로운 적대적 학습 프레임워크 제안**: 도메인 분류기 대신 태스크 특화 분류기를 판별자로 활용
- **이론적 근거 제시**: Ben-David et al.의 $\mathcal{H}\Delta\mathcal{H}$-distance 이론과의 연결성 확립
- **다양한 태스크에서 SOTA 달성**: 숫자 분류, 객체 분류, 시맨틱 세그멘테이션에서 우수한 성능 검증

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

비지도 도메인 적응(UDA: Unsupervised Domain Adaptation) 환경에서:

- **소스 도메인**: 레이블이 있는 데이터 $\{X_s, Y_s\}$
- **타겟 도메인**: 레이블이 없는 데이터 $X_t$

기존 GAN 기반 방법(DANN 등)은 도메인 분류기를 속이도록 피처를 생성하지만, 이 과정에서 타겟 샘플이 소스 분포의 **서포트(support) 밖**에 위치해도 탐지하지 못함. 즉, 클래스 경계 근방에 모호한 피처가 생성되는 문제가 발생한다.

### 2-2. 제안 방법 및 수식

#### 모델 구성요소

- $G$: 피처 생성기(Feature Generator)
- $F_1, F_2$: 두 개의 태스크 특화 분류기
- $p_1(\mathbf{y}|\mathbf{x}),\ p_2(\mathbf{y}|\mathbf{x})$: 각 분류기의 $K$-차원 소프트맥스 출력

#### Discrepancy Loss (불일치 손실)

두 분류기 출력 간의 L1 거리를 불일치 척도로 사용:

$$d(p_1, p_2) = \frac{1}{K} \sum_{k=1}^{K} |p_{1k} - p_{2k}| \tag{1}$$

#### 3단계 학습 절차

**Step A: 소스 샘플에 대한 분류 손실 최소화**

$$\min_{G, F_1, F_2} \mathcal{L}(X_s, Y_s) \tag{2}$$

$$\mathcal{L}(X_s, Y_s) = -\mathbb{E}_{(\mathbf{x}_s, y_s) \sim (X_s, Y_s)} \sum_{k=1}^{K} \mathbf{1}_{[k=y_s]} \log p(\mathbf{y}|\mathbf{x}_s) \tag{3}$$

**Step B: 분류기가 타겟에 대한 불일치를 최대화 (G 고정)**

$$\min_{F_1, F_2} \mathcal{L}(X_s, Y_s) - \mathcal{L}_{\text{adv}}(X_t) \tag{4}$$

$$\mathcal{L}_{\text{adv}}(X_t) = \mathbb{E}_{\mathbf{x}_t \sim X_t}[d(p_1(\mathbf{y}|\mathbf{x}_t), p_2(\mathbf{y}|\mathbf{x}_t))] \tag{5}$$

**Step C: 생성기가 타겟에 대한 불일치를 최소화 ($F_1, F_2$ 고정)**

$$\min_{G} \mathcal{L}_{\text{adv}}(X_t) \tag{6}$$

이 세 단계를 반복하며 생성기가 타겟 피처를 소스 서포트 내부에 위치시키도록 학습한다. 전체 목표는 다음의 minimax 문제로 요약:

$$\min_{G} \max_{F_1, F_2} \mathbb{E}_{\mathbf{x} \sim \mathcal{T}} \mathbf{I}[F_1 \circ G(\mathbf{x}) \neq F_2 \circ G(\mathbf{x})] \tag{9}$$

### 2-3. 이론적 근거 (Ben-David et al.)

Ben-David et al.의 도메인 적응 오류 상한 이론:

$$\forall h \in \mathcal{H},\quad R_{\mathcal{T}}(h) \leq R_{\mathcal{S}}(h) + \frac{1}{2} d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{S}, \mathcal{T}) + \lambda \tag{7}$$

여기서:

$$d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{S}, \mathcal{T}) = 2 \sup_{(h,h') \in \mathcal{H}^2} \left| \mathbb{E}_{\mathbf{x} \sim \mathcal{S}} \mathbf{I}[h(\mathbf{x}) \neq h'(\mathbf{x})] - \mathbb{E}_{\mathbf{x} \sim \mathcal{T}} \mathbf{I}[h(\mathbf{x}) \neq h'(\mathbf{x})] \right|$$

소스 샘플에 대한 두 분류기의 불일치가 낮다고 가정하면($h$, $h'$이 소스를 잘 분류), $d_{\mathcal{H}\Delta\mathcal{H}}$는 근사적으로:

$$\sup_{(h,h') \in \mathcal{H}^2} \mathbb{E}_{\mathbf{x} \sim \mathcal{T}} \mathbf{I}[h(\mathbf{x}) \neq h'(\mathbf{x})]$$

즉, **타겟 샘플에 대한 두 분류기의 불일치를 최소화하는 것이 타겟 오류 상한을 줄이는 것과 직결**됨을 이론적으로 보인다.

### 2-4. 모델 구조

```
입력 이미지 (소스/타겟)
        ↓
   Feature Generator G
   (공유 피처 추출기; CNN 백본)
        ↓
  ┌─────────────────┐
  F₁ (분류기 1)    F₂ (분류기 2)
  └─────────────────┘
        ↓
  Discrepancy Loss 계산
  d(p₁(y|xₜ), p₂(y|xₜ))
```

- 분류 태스크: 숫자 분류 → DANN 논문의 CNN 구조 사용
- 객체 분류: ResNet-101 (ImageNet 사전학습)
- 시맨틱 세그멘테이션: VGG-16 기반 FCN8s, DRN-D-105

### 2-5. 성능 향상

| 실험 설정 | Source Only | DANN | MCD (n=4) |
|---|---|---|---|
| SVHN→MNIST | 67.1% | 71.1% | **96.2%** |
| SYN SIGNS→GTSRB | 85.1% | 88.7% | **94.4%** |
| MNIST→USPS | 76.7% | 77.1% | **94.2%** |
| VisDA (mAcc) | 52.4% | 57.4% | **71.9%** |
| GTA5→Cityscapes (mIoU, DRN) | 22.2% | 32.8% | **39.7%** |

### 2-6. 한계

1. **두 분류기의 다양성 보장 문제**: $F_1$과 $F_2$의 충분한 다양성이 확보되지 않으면 불일치 신호가 약해질 수 있음
2. **하이퍼파라미터 $n$ 선택**: Step C의 반복 횟수 $n$이 성능에 민감하며, UDA 설정에서 검증 세트 없이 최적값 선택이 어려움
3. **두 분류기 구조의 단순성**: 단 2개의 분류기만 사용하여 불일치 신호의 표현력이 제한될 수 있음
4. **완전 비지도 설정에서의 하이퍼파라미터 튜닝 어려움**: 논문에서도 일부 방법(ATDA)이 소수의 레이블된 타겟 샘플을 사용하면 더 나은 결과를 보임을 인정
5. **대규모 클래스 수 확장성**: 클래스 수 $K$가 매우 클 경우 discrepancy 측정의 안정성이 불확실

---

## 3. 일반화 성능 향상 가능성

### 3-1. 결정 경계 인식 기반 정렬

기존 방법들이 분포 자체를 맞추는 데 초점을 맞춘 것과 달리, MCD는 **클래스 결정 경계를 고려한 정렬**을 수행한다. 타겟 샘플을 소스 분포의 서포트 내부로 유도함으로써:

$$\text{Target features} \xrightarrow{\text{Generator}} \text{Source support 내부로 이동}$$

이는 타겟 도메인에서도 클래스 구분이 명확한 피처를 생성하게 하여, 단순한 분포 매칭보다 **판별력(discriminability)이 높은 표현**을 학습시킨다.

### 3-2. $\mathcal{H}\Delta\mathcal{H}$-distance 최소화와 일반화

이론적으로, MCD는 기존 방법들이 최소화하는 $d_\mathcal{H}(\mathcal{S}, \mathcal{T})$ (H-distance)보다 더 타이트한 상한인 $d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{S}, \mathcal{T})$를 최소화하는 방향으로 설계되어 있다. 즉:

$$d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{S}, \mathcal{T}) \leq d_{\mathcal{H}}(\mathcal{S}, \mathcal{T})$$

이는 **타겟 오류 상한이 더 tight하게 줄어드는 것**을 의미하며, 이론적으로 더 강한 일반화 보장을 제공한다.

### 3-3. 다양한 태스크로의 일반화

이 방법은 특정 아키텍처에 종속되지 않고:
- 이미지 분류 (digit, object)
- 시맨틱 세그멘테이션

에 모두 적용 가능함을 실험적으로 증명하여, 프레임워크 자체의 범용성(generalizability)을 입증하였다.

### 3-4. 모호한 샘플 처리를 통한 일반화

소스 서포트 밖의 타겟 샘플을 두 분류기의 불일치로 탐지하고, 생성기가 이를 서포트 내부로 이동시키는 과정은 타겟 도메인에서 **모호한 경계 근방 샘플**에 대한 분류 신뢰도를 높여 일반화 성능을 향상시킨다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려 사항

### 4-1. 연구에 미치는 영향

**① 태스크 특화 판별자 패러다임의 확산**

MCD 이후 도메인 분류기 대신 태스크 특화 구조를 판별자로 사용하는 연구가 크게 증가하였다. 분류기의 불일치를 활용하는 아이디어는 이후 다양한 변형으로 발전되었다.

**② 이론-실험 연결의 모범 사례**

$\mathcal{H}\Delta\mathcal{H}$-distance와의 명확한 연결은, 이후 도메인 적응 이론 연구에서 알고리즘 설계의 이론적 근거를 제시하는 방식의 표준적 사례가 되었다.

**③ 반지도 및 소수 레이블 설정으로의 확장**

MCD의 프레임워크는 소수의 레이블된 타겟 샘플이 주어지는 준지도(semi-supervised) DA나, 소스-프리(source-free) DA 연구에서도 참조 구조로 활용되었다.

**④ 다중 분류기 앙상블 DA 연구의 기초**

두 분류기 구조는 이후 3개 이상의 분류기를 사용하거나, 분류기 다양성을 다른 방식으로 보장하는 연구들의 출발점이 되었다.

### 4-2. 앞으로 연구 시 고려할 점

**① 분류기 다양성 보장 메커니즘**

현재 방법은 초기화 차이에만 의존하여 $F_1$, $F_2$의 다양성을 확보한다. 더 명시적인 다양성 보장 메커니즘 (예: 드롭아웃, 다른 아키텍처, orthogonality 제약)의 도입이 필요하다.

**② 클래스 불균형 문제**

소스/타겟 도메인 간 클래스 분포가 다를 경우(partial DA, open-set DA), 단순 discrepancy 최대화가 오히려 부정적 이전(negative transfer)을 초래할 수 있다.

**③ 대규모 도메인 격차 대응**

매우 큰 도메인 격차(예: 합성→실사)에서는 피처 정렬만으로 충분하지 않을 수 있으며, 이미지 수준 변환(image translation)과의 결합을 고려해야 한다.

**④ 소스 프리(Source-Free) DA 설정으로의 확장**

실제 환경에서는 학습 시 소스 데이터에 접근할 수 없는 경우가 많다. MCD를 소스 프리 설정에 적용하기 위한 연구가 필요하다.

**⑤ 하이퍼파라미터 자동화**

$n$(Step C 반복 횟수)의 최적화를 위한 자동화된 방법 또는 적응적 스케줄링이 실용적 적용에 중요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5-1. Cycle Self-Training (CST) — Kim et al., NeurIPS 2020

- **관련성**: MCD의 분류기 불일치 아이디어를 의사 레이블(pseudo-label) 생성에 활용
- **차이점**: 두 분류기의 동의(agreement)한 타겟 샘플에만 의사 레이블을 부여하는 자기 학습(self-training) 방식으로 확장

### 5-2. Transferable Query Selection for Active Domain Adaptation — Fu et al., CVPR 2021

- **관련성**: 능동 학습(active learning)과 DA를 결합하며, 분류기 불일치를 불확실성 지표로 활용
- **차이점**: 완전 비지도가 아닌 소수 레이블 선택 전략에 MCD 아이디어 적용

### 5-3. Source-Free Domain Adaptation (SHOT) — Liang et al., ICML 2020

- **관련성**: MCD의 태스크 특화 분류기 활용 아이디어를 계승하되, 소스 데이터 없이 DA 수행
- **차이점**: 소스 모델의 분류기를 고정하고 피처 추출기만 업데이트; 정보 극대화(information maximization)와 의사 레이블 결합
- MCD 대비 소스 데이터 프라이버시 문제 해결

### 5-4. Generalized Domain Adaptation (GDA) 및 Open-Set DA

- MCD는 클로즈드-셋(closed-set) 가정에 기반하지만, 이후 연구들은 타겟에 소스에 없는 클래스가 존재하는 오픈-셋 DA(open-set DA)를 고려
- **Universal Domain Adaptation (UAN) — You et al., CVPR 2019**: 공통 클래스와 비공통 클래스를 구분하는 불확실성 지표로 분류기 불일치 활용

### 5-5. 비교 정리표

| 논문 | 분류기 역할 | 소스 데이터 필요 | 레이블 가정 | 주요 발전 |
|---|---|---|---|---|
| MCD (CVPR 2018) | 판별자 | ✅ | 클로즈드-셋 | 기초 프레임워크 |
| SHOT (ICML 2020) | 고정 분류기 | ❌ | 클로즈드-셋 | 소스 프리 DA |
| CST (NeurIPS 2020) | 의사 레이블 필터 | ✅ | 클로즈드-셋 | 자기 학습 결합 |
| UAN (CVPR 2019) | 불확실성 지표 | ✅ | 오픈-셋 | 범용 DA |

### 5-6. 종합 평가

MCD는 2018년 CVPR에서 제안된 이후, **태스크 특화 분류기를 판별자로 활용하는 패러다임**을 도메인 적응 연구에 확립시켰다. 2020년 이후 연구들은 MCD의 핵심 아이디어를 계승하면서도:

- 소스 데이터 의존성 제거 (소스 프리 DA)
- 오픈-셋/범용 설정 확장
- 자기 학습과의 결합
- 트랜스포머(ViT) 백본 도입

등의 방향으로 발전하고 있으며, MCD는 여전히 이 분야의 핵심 기준선(baseline)으로 광범위하게 인용되고 있다.

---

## 참고 자료

**주요 논문 (본문에서 직접 인용)**
- Saito, K., Watanabe, K., Ushiku, Y., & Harada, T. (2018). **Maximum Classifier Discrepancy for Unsupervised Domain Adaptation**. *CVPR 2018*. (제공된 PDF)
- Ben-David, S., et al. (2010). A theory of learning from different domains. *Machine Learning*, 79(1-2):151–175.
- Ben-David, S., et al. (2007). Analysis of representations for domain adaptation. *NIPS 2007*.
- Ganin, Y., & Lempitsky, V. (2014). Unsupervised domain adaptation by backpropagation. *ICML 2014*.
- Ganin, Y., et al. (2016). Domain-adversarial training of neural networks. *JMLR*, 17(59):1–35.

**2020년 이후 비교 참고 논문**
- Liang, J., et al. (2020). Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation. *ICML 2020*.
- Kim, S., et al. (2020). Attract, Perturb, and Explore: Learning a Feature Alignment Network for Semi-supervised Domain Adaptation. *ECCV 2020*.
- You, K., et al. (2019). Universal Domain Adaptation. *CVPR 2019*.

> **주의**: 2020년 이후 최신 연구 비교 부분은 논문 원문에 직접 포함된 내용이 아니므로, 해당 논문들의 제목과 학회 정보를 기반으로 기술하였으며, 세부 수치는 확인된 범위 내에서만 기재하였습니다.
