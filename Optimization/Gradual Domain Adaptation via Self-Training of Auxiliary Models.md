# Gradual Domain Adaptation via Self-Training of Auxiliary Models

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문은 **소스 도메인과 타겟 도메인 간의 격차(domain gap)가 클수록 기존 도메인 적응(DA) 방법의 성능이 급격히 저하**된다는 경험적 분석에서 출발합니다. 이를 해결하기 위해 **진화하는 중간 도메인(evolving intermediate domains)** 을 자동으로 구성하고, 각 중간 도메인에 대한 보조 모델(auxiliary model)을 자기 학습(self-training)으로 학습하는 **AuxSelfTrain** 방법을 제안합니다.

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| **AuxSelfTrain 제안** | 중간 도메인을 위한 보조 모델을 자기 학습으로 학습하여 도메인 격차를 점진적으로 극복 |
| **효율적인 샘플 선택 전략** | 예측 확률 기반의 타겟/소스 샘플 선택 전략을 통해 중간 도메인 구성 |
| **암묵적 앙상블(Implicit Ensemble)** | 예측 불확실성 품질 향상을 위한 암묵적 앙상블 기반 향상 지표 도입 및 SSDA로의 확장 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**문제 정의:**
- $n_s$개의 레이블된 소스 데이터 $\mathcal{S} = \{x_i^s, y_i^s\}\_{i=1}^{n_s}$와 $n_t$개의 레이블 없는 타겟 데이터 $\mathcal{T} = \{x_i^t\}_{i=1}^{n_t}$가 존재
- 도메인 간 격차가 클수록 소스 모델의 타겟 성능이 급격히 저하됨 (Figure 1)
- 실제 벤치마크 데이터셋에서는 중간 도메인이 주어지지 않음

**핵심 관찰:** Rotating MNIST 실험에서, 도메인 거리가 커질수록 소스 모델의 최대 예측 확률(maximum prediction probability)이 감소함을 경험적으로 확인

$$\text{도메인 거리} \uparrow \;\Rightarrow\; \text{최대 예측 확률} \downarrow \;\Rightarrow\; \text{타겟 성능} \downarrow$$

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 소스 모델 훈련 (Source Model Training)

$$\boldsymbol{f}_s = \arg\min_{\boldsymbol{f} \in \mathcal{F}} \mathbb{E}_{\{x^s, y^s\} \in \mathcal{S}} \mathcal{L}(\boldsymbol{f}(x^s), y^s) \tag{1}$$

여기서 $\mathcal{L}(\boldsymbol{f}(x), y) = -\log f_y(x)$는 cross-entropy loss

#### Step 2: 기본 자기 학습 (Standard Self-Training)

소스 모델 $\boldsymbol{f}\_s$를 이용해 타겟 샘플에 pseudo label $o(\boldsymbol{f}\_s(x^t)) = \arg\max_k f_{s,k}(x^t)$ 부여:

$$\boldsymbol{f}_t = \arg\min_{\boldsymbol{f} \in \mathcal{F}} \mathbb{E}_{x^t \in \mathcal{T}} \mathcal{L}(\boldsymbol{f}(x^t), o(\boldsymbol{f}_s(x^t))) \tag{2}$$

> **문제점:** Kumar et al. [24]에 따르면, $P$와 $Q$ 간의 도메인 불일치가 클수록 $\boldsymbol{f}_t$의 타겟 리스크가 증가

#### Step 3: 중간 도메인 구성 (Intermediate Domain Construction)

$M+1$개의 중간 데이터셋 $\{\mathcal{M}\_m\}_{m=0}^{M}$ 도입:
- $\mathcal{M}_0 = \mathcal{S}$ (소스 도메인)
- $\mathcal{M}_M = \mathcal{T}$ (타겟 도메인)
- $\mathcal{M}_m$: 타겟 데이터의 $\frac{m}{M}$ 비율 + 소스 데이터의 $\frac{M-m}{M}$ 비율

**타겟 샘플 선택 지표:**

$$c_t(x_i^t, \boldsymbol{f}_m) = \max(\boldsymbol{f}_m(x_i^t)) \tag{3}$$

$$v_i^t = \begin{cases} 1, & c_t(x_i^t, \boldsymbol{f}_m) \in TOP_{\frac{(m+1)n_t}{M}}(\mathcal{C}_t) \\ 0, & \text{Otherwise} \end{cases} \tag{4}$$

**소스 샘플 선택을 위한 타겟 프로토타입 분류기:**

$$\boldsymbol{p}_k = \frac{\sum_{i=1}^{n_t} \mathbf{1}_{o(\boldsymbol{f}_m(x_i^t))=k} \phi_m(x_i^t)}{\sum_{i=1}^{n_t} \mathbf{1}_{o(\boldsymbol{f}_m(x_i^t))=k}} \tag{5}$$

$$c_s(x_i^s, \boldsymbol{f}_m) = \frac{\exp(1 + \|\phi_m(x_i^s) - \boldsymbol{p}_{y_i^s}\|_2)^{-1}}{\sum_{k=1}^{K} \exp(1 + \|\phi_m(x_i^s) - \boldsymbol{p}_k\|_2)^{-1}} \tag{6}$$

$$v_i^s = \begin{cases} 1, & c_s(x_i^s, \boldsymbol{f}_m) \in TOP_{\frac{(M-m-1)n_s}{M}}(\mathcal{C}_s) \\ 0, & \text{Otherwise} \end{cases} \tag{7}$$

#### Step 4: 보조 모델 자기 학습 (Auxiliary Model via Self-Training)

$$\boldsymbol{f}_{m+1} = \arg\min_{\boldsymbol{f} \in \mathcal{F}} \frac{1}{n_s}\sum_{i=1}^{n_s} v_i^s \mathcal{L}(\boldsymbol{f}(x_i^s), y_i^s) + \frac{1}{n_t}\sum_{i=1}^{n_t} v_i^t \mathcal{L}(\boldsymbol{f}(\mathcal{A}(x_i^t)), o(\boldsymbol{f}_m(x_i^t))) \tag{8}$$

여기서 $\mathcal{A}(\cdot)$는 고급 데이터 증강(RandAugment [8])

#### Step 5: 암묵적 앙상블 기반 향상 지표 (Implicit Ensemble)

$$\tilde{\boldsymbol{f}}_m(x^t) = (\boldsymbol{f}_m(x^t) + \hat{\boldsymbol{f}}_m(x^t) + \bar{\boldsymbol{f}}_m(x^t))/3 \tag{9}$$

- $\boldsymbol{f}_m$: 기본 보조 모델
- $\hat{\boldsymbol{f}}_m$: 클러스터링 기반 분류기 (k-means 활용)

$$\hat{f}_{m,k}(x^t) = \frac{\exp(1 + \|\phi(x^t) - O_k^t\|_2)^{-1}}{\sum_{k'=1}^{K} \exp(1 + \|\phi(x^t) - O_{k'}^t\|_2)^{-1}} \tag{10}$$

- $\bar{\boldsymbol{f}}_m$: 레이블 전파(label propagation) 기반 분류기

$$\sum_{i=1}^{n_{st}} \|\bar{\boldsymbol{f}}_m(x_i^{st}) - y_i^{st}\|^2 + \lambda \sum_{i,j}^{n_{st}} a_{ij} \left\|\frac{\bar{\boldsymbol{f}}_m(x_i^{st})}{\sqrt{d_{ii}}} - \frac{\bar{\boldsymbol{f}}_m(x_j^{st})}{\sqrt{d_{jj}}}\right\|^2 \tag{11}$$

---

### 2.3 모델 구조

```
[소스 데이터 S] → [소스 모델 f_0 = f_s]
        ↓
[중간 도메인 M_1 구성] (소스 (M-1)/M + 타겟 1/M, 선택 전략 적용)
        ↓
[암묵적 앙상블 f̃_0 계산] → [보조 모델 f_1 학습] (자기 학습, 식 8)
        ↓
[중간 도메인 M_2 구성] (소스 (M-2)/M + 타겟 2/M)
        ↓
        ...
        ↓
[중간 도메인 M_M = T]
        ↓
[최종 모델 f_M] → [타겟 도메인 테스트]
```

**백본 네트워크:** ResNet-50, ResNet-34, VGG-16 (ImageNet 사전 학습)
- 특징 추출기 $\phi_m: \mathcal{X} \rightarrow \mathcal{Z}$ (마지막 FC 레이어 제거)
- 분류기 $\varphi_m: \mathcal{Z} \rightarrow [0,1]^K$ (새로운 FC 레이어, $K$ 뉴런)

**Algorithm 1: AuxSelfTrain**
```
Input: S, T, M
1: f_0 ← f_s (소스 모델 초기화)
2: for m = 1 to M do
3:   f̃_{m-1} ← 암묵적 앙상블 (식 9)
4:   M_m ← 식 (4), (7)로 중간 도메인 구성
5:   f_m ← 식 (8)로 보조 모델 학습
6: end for
7: return f_M (타겟 도메인 테스트용)
```

---

### 2.4 성능 향상

#### VisDA-2017 (Syn. → Real, ResNet-50)

| 방법 | 정확도 (%) |
|------|-----------|
| Source Only | 45.6 |
| DANN | 55.0 |
| MCD | 69.8 |
| MDD | 74.6 |
| GSDA | 81.5 |
| RWOT | 84.0 |
| **AuxSelfTrain** | **85.2** |

#### OfficeHome DA (ResNet-50, 평균)

| 방법 | 평균 (%) |
|------|---------|
| Source Only | 46.1 |
| DANN | 57.6 |
| MDD | 68.1 |
| SRDC | 71.3 |
| **AuxSelfTrain** | **72.5** |

#### DomainNet SSDA (ResNet-34, 3-shot 평균)

| 방법 | 평균 (%) |
|------|---------|
| MME | 68.9 |
| BiAT | 69.7 |
| Kim et al. | 71.7 |
| HDAN | 71.3 |
| **AuxSelfTrain** | **76.4** |

---

### 2.5 한계

1. **중간 도메인 수 $M$의 민감성:** $M$이 너무 작으면 성능이 저하되고, 너무 크면 계산 비용이 증가 (경험적으로 $M=50$ 권장)
2. **최대 예측 확률의 신뢰성 문제:** DNN이 훈련 데이터에서 멀리 떨어진 입력에 대해서도 높은 확률을 출력할 수 있음 [33] → 암묵적 앙상블로 부분적 보완
3. **이론적 분석의 부재:** 도메인 이동과 최대 예측 확률의 상관관계에 대한 이론적 근거가 아직 미완성
4. **계산 효율:** 소스 모델 훈련 대비 약 2배의 반복 횟수 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상의 핵심 메커니즘

**점진적 도메인 적응의 이론적 근거:**

Kumar et al. [24] (Theorem 3.2)에 의하면:

> *"두 도메인 간의 분포 발산(distribution divergence)이 충분히 작고, 한 도메인에서 신뢰할 수 있는 모델이 존재하면, 자기 학습을 통해 다른 도메인에서도 신뢰할 수 있는 모델을 얻을 수 있다."*

이를 수식으로 표현하면, 소스 리스크 $\epsilon_s$와 도메인 발산 $d_\mathcal{H}(P, Q)$에 대해:

$$\epsilon_T(f) \leq \epsilon_S(f) + d_{\mathcal{H}\Delta\mathcal{H}}(P, Q) + \lambda^* \tag{Ben-David et al., 2010}$$

AuxSelfTrain은 연속적인 중간 도메인 $\mathcal{M}\_m$, $\mathcal{M}\_{m+1}$ 사이의 $d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{M}\_m, \mathcal{M}_{m+1})$을 최소화함으로써 각 단계에서의 자기 학습 신뢰성을 보장합니다.

### 3.2 타겟 특징 판별력(Feature Discriminability) 향상

논문의 Figure 5(a)에서 AuxSelfTrain은 두 가지 특징 판별력 지표에서 ResNet-50 기준선 대비 향상:
- $\max J(W)$ 증가 → 클래스 간 분리성 향상
- 평균 오류율 감소 → 클래스 내 응집성 향상

이는 분포 불일치 최소화 방법(DANN)이 오히려 타겟 특징 판별력을 **저하**시키는 문제 [5]와 대조되는 중요한 장점입니다.

### 3.3 암묵적 앙상블의 일반화 기여

세 가지 서로 다른 알고리즘(신경망 분류기, k-means 클러스터링, 레이블 전파)의 앙상블은:
- 예측 불확실성의 품질을 도메인 이동에 대해 향상
- 과적합(overconfident prediction) 문제 완화
- 샘플 선택의 정확도 향상 → pseudo label 노이즈 감소

### 3.4 고급 데이터 증강의 기여

RandAugment [8]를 DA에 최초로 적용하여:
- OfficeHome: $69.7\% \rightarrow 72.5\%$ (+2.8%)
- VisDA-2017: $77.5\% \rightarrow 85.2\%$ (+7.7%)

이는 데이터 증강이 자기 학습 기반 DA에서 일반화 성능에 매우 중요함을 시사합니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

**① 점진적 도메인 적응 패러다임 강화**

기존의 단일 단계 도메인 적응(one-shot DA)에서 벗어나, **다단계 점진적 적응**이 대규모 도메인 격차 극복에 더 효과적임을 실험적으로 증명합니다. 이는 향후 연속적 도메인 적응(continual DA), 온라인 DA 연구의 이론적·실험적 기반을 제공합니다.

**② 자기 학습과 중간 도메인의 결합 중요성 부각**

Kumar et al. [24]의 이론과 결합하여, "작은 도메인 격차에서의 자기 학습"이라는 원칙이 실제 벤치마크에서도 유효함을 보여줍니다. 이는 자기 학습 기반 DA 연구의 설계 원칙으로 자리잡을 가능성이 있습니다.

**③ SSDA로의 자연스러운 확장**

소스+레이블 타겟 데이터를 사전 주어진 중간 도메인으로 처리하는 접근 방식은, 레이블 효율적인(label-efficient) 학습 연구에 중요한 방법론적 시사점을 줍니다.

**④ 샘플 선택과 불확실성 추정의 중요성**

예측 확률을 도메인 이동 지표로 활용하는 아이디어는 능동 학습(active learning), 분포 외(out-of-distribution) 탐지 연구와 연결됩니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래는 본 논문과 관련된 2020년 이후 주요 연구들의 비교입니다. 단, 본 논문이 제출된 시점(2021년 6월) 이후의 연구에 대해서는 제가 직접 확인한 내용과 논문 내 인용 정보를 기반으로 서술하며, **2021년 이후 논문에 대한 구체적 수치는 직접 확인한 논문이 아닌 경우 제시하지 않습니다.**

#### 논문 내 인용된 2020년 이후 주요 관련 연구

| 연구 | 핵심 방법 | AuxSelfTrain과의 관계 |
|------|-----------|----------------------|
| **Kumar et al., ICML 2020** [24]: *Understanding Self-Training for Gradual Domain Adaptation* | 점진적 DA에서 자기 학습의 이론적 보장 제시 | AuxSelfTrain의 이론적 근거 제공; AuxSelfTrain은 이를 실제로 구현 |
| **Mei et al., ECCV 2020** [32]: *Instance Adaptive Self-Training for UDA* | 인스턴스 적응적 자기 학습 | 중간 도메인 없이 직접 자기 학습 적용 → 대형 도메인 격차에서 취약 |
| **Cui et al., CVPR 2020** [10]: *Gradually Vanishing Bridge for Adversarial DA* | GAN 기반 중간 도메인 생성 | 생성 모델 의존 vs. AuxSelfTrain의 데이터 혼합 방식 (더 단순하고 효율적) |
| **Saito et al., ICCV 2019** [39]: *MME (Minimax Entropy for SSDA)* | 적대적 엔트로피 최소화 | SSDA 기준선; AuxSelfTrain이 OfficeHome/DomainNet에서 일관되게 우월 |
| **Tang et al., CVPR 2020** [45]: *SRDC* | 구조적 정규화 깊은 클러스터링 | OfficeHome 평균 71.3% vs. AuxSelfTrain 72.5% |

#### 방법론적 차별성 비교

```
[도메인 불일치 최소화 방법 (DANN, MDD)]
  → 전역적 특징 정렬, 큰 격차에서 판별력 저하 문제

[생성 모델 기반 중간 도메인 (DLOW, GVB)]
  → 복잡한 학습, 생성 품질에 의존

[직접 자기 학습 (SRDC, Instance-adaptive ST)]
  → 큰 격차에서 pseudo label 오류 누적

[AuxSelfTrain (본 논문)]
  → 점진적 중간 도메인 + 선택적 자기 학습
  → 이론적 보장 + 실용적 구현
  → DA와 SSDA 통합 프레임워크
```

---

### 4.3 앞으로 연구 시 고려할 점

**① 이론적 기반 강화**

논문 스스로 인정하듯, 도메인 이동과 최대 예측 확률의 상관관계에 대한 이론적 분석이 미완성입니다. 향후 연구에서는 이를 수학적으로 엄밀하게 증명하고, 중간 도메인 구성의 최적성 조건을 이론화할 필요가 있습니다.

**② 샘플 선택 지표의 고도화**

현재의 최대 예측 확률은 DNN의 과확신(overconfidence) 문제로 인해 신뢰성이 제한됩니다. 다음과 같은 방향을 고려할 수 있습니다:

$$\text{개선 방향: Bayesian 불확실성 추정, Energy-based OOD 탐지, Conformal Prediction}$$

**③ 중간 도메인 구성 방식의 다양화**

- 단순 데이터 혼합(mixing)이 아닌, **Pixel-level Mixup** [51], **StyleGAN 기반 변환** 등
- **연속 학습(Continual Learning)** 관점에서의 중간 도메인 동적 관리

**④ 확장성 고려**

- 현재는 단일 소스-타겟 쌍; **다중 소스 도메인(multi-source DA)** 으로의 확장 필요
- 대규모 언어 모델(LLM), 비전-언어 모델(VLM) 시대에서의 적용 가능성
- 클래스 불균형, 레이블 노이즈가 있는 실제 환경에서의 강건성

**⑤ 계산 효율성**

$M=50$의 중간 도메인은 반복적인 특징 추출 및 k-means 클러스터링을 요구합니다. 경량화된 구현이나 **온라인 방식**으로의 전환을 검토해야 합니다.

**⑥ 공정한 비교 실험 설계**

2020년 이후 사전 학습 모델(ViT, CLIP 등)의 발전으로 기준선(baseline)이 크게 향상되었습니다. 향후 연구에서는 강력한 사전 학습 표현과의 시너지를 탐색해야 합니다.

---

## 참고 자료

**본 답변에서 직접 참조한 논문:**

1. **Zhang, Y., Deng, B., Jia, K., & Zhang, L. (2021).** *Gradual Domain Adaptation via Self-Training of Auxiliary Models.* arXiv:2106.09890v1 [cs.LG]. *(본 분석의 주 대상 논문)*

2. **Kumar, A., Ma, T., & Liang, P. (2020).** *Understanding Self-Training for Gradual Domain Adaptation.* ICML 2020. *(논문 내 [24]로 인용)*

3. **Ben-David, S., Blitzer, J., Crammer, K., & Pereira, F. (2007).** *Analysis of Representations for Domain Adaptation.* NeurIPS 2007. *(논문 내 [2]로 인용)*

4. **Ben-David, S. et al. (2010).** *A Theory of Learning from Different Domains.* Machine Learning. *(논문 내 [1]로 인용)*

5. **Saito, K. et al. (2019).** *Semi-Supervised Domain Adaptation via Minimax Entropy.* ICCV 2019. *(논문 내 [39]로 인용)*

6. **Ganin, Y. et al. (2016).** *Domain-Adversarial Training of Neural Networks.* JMLR. *(논문 내 [15]로 인용)*

7. **Cubuk, E.D. et al. (2020).** *RandAugment.* CVPR Workshops. *(논문 내 [8]로 인용)*

8. **Nguyen, A., Yosinski, J., & Clune, J. (2015).** *Deep Neural Networks are Easily Fooled.* CVPR 2015. *(논문 내 [33]으로 인용)*

9. **Lakshminarayanan, B. et al. (2016).** *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* arXiv:1612.01474. *(논문 내 [25]로 인용)*

10. **Chopra, S., Balakrishnan, S., & Gopalan, R. (2013).** *DLID: Deep Learning for Domain Adaptation by Interpolating between Domains.* ICML Workshop. *(논문 내 [7]로 인용)*

> **⚠️ 주의:** 2021년 이후 발표된 후속 연구들(예: SHOT++, T3A, 등)에 대한 구체적인 수치 비교는 본 논문의 제출 시점(2021년 6월)에서 해당 연구들이 존재하지 않았거나 직접 확인이 어렵기 때문에, 확실하게 확인된 내용만 서술하였습니다.
