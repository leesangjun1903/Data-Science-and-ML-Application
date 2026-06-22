# Minority-Oriented Vicinity Expansion with Attentive Aggregation for Video Long-Tailed Recognition

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문은 **Video Long-Tailed Recognition(VLTR)**이라는 문제를 체계적으로 정의하고, 기존 방법들이 간과했던 세 가지 핵심 도전과제를 동시에 해결하는 통합 프레임워크를 제안합니다. 실세계 온라인 비디오 플랫폼의 데이터는 자연스럽게 롱테일 분포를 따르며, 이를 효과적으로 학습하기 위해서는 (1) 태스크 비관련 피처 문제, (2) 약한 레이블(Video-level) 문제, (3) 클래스 불균형으로 인한 편향 학습 문제를 **동시에** 다뤄야 한다고 주장합니다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **VLTR 문제 정의** | 세 가지 도전과제를 체계적으로 정리하고 연구 방향 제시 |
| **학습 가능한 피처 집계기** | Self-attentive + Codebook-attentive Aggregator를 통한 태스크 관련 표현 생성 |
| **MOVE 제안** | Minority-Oriented Vicinity Expansion: 동적 외삽 + 보정된 내삽으로 소수 클래스 분포 확장 |
| **새로운 벤치마크** | Imbalanced-MiniKinetics200 제안으로 다양한 불균형 시나리오 평가 가능 |
| **성능** | VideoLT에서 이전 SOTA 대비 헤드 클래스 18%, 테일 클래스 58% 상대적 성능 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

논문은 VLTR의 세 가지 핵심 도전과제를 다음과 같이 정의합니다:

**도전 1: 태스크 비관련 피처 (Task-irrelevant Features)**
- 대규모 비디오 재학습은 비현실적 → pretrained network 사용 불가피
- 사전학습 데이터와 목표 도메인 간 도메인 갭 존재

**도전 2: 약한 레이블 (Weakly-labeled Problem)**  
- 프레임 단위 레이블 취득은 비용 과다
- 비디오 레벨 레이블만 가용하여 snippet-level 감독이 불가

**도전 3: 편향 학습 (Biased Training)**  
- 롱테일 분포로 인해 결정 경계가 소수 클래스 쪽으로 편향
- 소수 클래스는 데이터 부족으로 과적합 발생

---

### 2.2 제안 방법 및 수식

#### Phase 1: 학습 가능한 피처 집계기 (Learnable Feature Aggregators)

**Self-Attentive Aggregator (PSA)**

기본 Self-Attention은 다음과 같이 정의됩니다:

$$\hat{\mathbf{x}}^i = \text{Softmax}\left(QK^\top / \sqrt{d}\right)V \tag{1}$$

여기서 $d$는 피처 차원입니다. 논문은 이를 **Prototypical Self-Attention(PSA)**으로 변형하여, $Q$에 $\text{Pool}(\mathbf{x}^i)$를 사용합니다. 이를 통해 전역 프로토타입이 프레임 단위의 로컬 단서로 보완됩니다.

**Codebook-Attentive Aggregator (NetVLAD 기반)**

NetVLAD의 VLAD 표현:

$$\tilde{\mathbf{x}}^i_{c,k} = \sum_{t=1}^{T} \rho_k(\mathbf{x}^i_t)(\mathbf{x}^i_{t,c} - \boldsymbol{\mu}_{k,c}) \tag{2}$$

여기서 $\rho_k$는 softmax로 미분 가능하게 정의:

$$\rho_k(\mathbf{x}^i_t) = \frac{e^{w_k^\top \mathbf{x}^i_t + b_k}}{\sum_{k'=1}^{K} e^{w_{k'}^\top \mathbf{x}^i_t + b_{k'}}} \tag{3}$$

**최종 집계 피처:**

$$\mathbf{z}^i = f_\theta(\hat{\mathbf{x}}^i) \oplus f_\phi(\tilde{\mathbf{x}}^i) \tag{4}$$

여기서 $\oplus$는 concatenation, $f_\theta(\cdot)$와 $f_\phi(\cdot)$는 차원 축소를 위한 FC 레이어입니다.

---

#### Phase 2: MOVE (Minority-Oriented Vicinity Expansion)

**기반 이론: Vicinal Risk Minimization (VRM)**

$$R_{vic}(h) = \frac{1}{N}\sum_{i=1}^{N} \int \ell(h(\mathbf{z}), \mathbf{y}^i) dP_{\mathbf{z}^i, \mathbf{y}^i}(\mathbf{z}, \mathbf{y}) \tag{5}$$

$$= \frac{1}{N}\sum_{i=1}^{N} \ell(h(\tilde{\mathbf{z}}^i), \tilde{\mathbf{y}}^i) \tag{6}$$

**Tail-weighted Criterion:**

$$\boldsymbol{\tau}_s = (q_s - q_{\min}) / (q_{\max} - q_{\min}) \tag{7}$$

$q_s$는 $s$번째 클래스의 샘플 수, $q_{\min}$, $q_{\max}$는 최솟값·최댓값입니다. **$\tau_s$가 작을수록 소수 클래스**를 의미합니다.

**Dynamic Frame Sampler (DFS):**

$$\mathbf{m}_{s,t} = \mathbb{1}_{[t \in \mathbf{I}_s]} \tag{8}$$

$$\mathbf{I}_s \subseteq \mathbf{T}, \quad |\mathbf{I}_s| \sim \mathcal{U}\left(\max(\lfloor \boldsymbol{\tau}_s \times T \rfloor, \sigma), T\right) \tag{9}$$

소수 클래스일수록 $|\mathbf{I}_s|$가 작아져 더 다양한 마스크가 생성됩니다.

**Minority-Oriented Dynamic Extrapolation:**

Codebook-attentive aggregator에 DFS 마스크 적용:

$$\tilde{\mathbf{x}}^i_{c,k} = \sum_{t=1}^{T} \mathbf{m}_{y^i, t} \cdot \rho_k(\mathbf{x}^i_t)(\mathbf{x}^i_{t,c} - \boldsymbol{\mu}^i_{k,c}) \tag{10}$$

외삽 분포 $p_{ex}$:

$$p_{ex}(\hat{\mathbf{z}}, \hat{\mathbf{y}} | \mathbf{u}^i, \mathbf{v}^i, \mathbf{y}^i) = \mathbb{E}_\omega \left[ \delta(\hat{\mathbf{z}} = \omega \mathbf{u}^i + (1-\omega)\mathbf{v}^i, \hat{\mathbf{y}} = \mathbf{y}^i) \right] \tag{11}$$

여기서 $\omega \sim \text{Beta}(\alpha, \alpha) + 1$, 범위는 $[1, 2]$로 클래스 경계를 벗어나지 않도록 제한됩니다.

**Minority-Oriented Calibrated Interpolation:**

$$p_{in}(\tilde{\mathbf{z}}, \tilde{\mathbf{y}} | (\hat{\mathbf{z}}^i, \hat{\mathbf{y}}^i) \sim p_{ex}) = \frac{1}{N}\sum_{j=1}^{N} \mathbb{E}_\lambda \left[ \delta\left(\tilde{\mathbf{z}} = \lambda\hat{\mathbf{z}}^i + (1-\lambda)\hat{\mathbf{z}}^j,\right.\right.$$

$$\left.\left. \tilde{\mathbf{y}}_s = \min_{1 \le s \le S}\left(\left(\lambda\hat{\mathbf{y}}^i_s + (1-\lambda)\hat{\mathbf{y}}^j_s\right) \times ((1-\boldsymbol{\tau}_s) + \gamma),\ 1\right)\right) \right] \tag{12}$$

여기서 $(1 - \tau_s)$가 소수 클래스에 더 큰 가중치를 부여합니다. $\gamma$는 레이블이 0이 되는 것을 방지하는 스무딩 편향, $\lambda \sim \text{Beta}(\alpha, \alpha)$입니다.

---

### 2.3 모델 구조

```
입력 피처 x (pretrained network 출력)
       ↓
   DFS (Dynamic Frame Sampler)
   ↙              ↘
Self-attentive    Codebook-attentive
Aggregator(PSA)   Aggregator(NetVLAD)
   ↘              ↙
    Concatenation → z
       ↓
  Dynamic Extrapolation (p_ex)
       ↓
  Calibrated Interpolation (p_in)
       ↓
   Classifier → 예측
```

- **추론 시**: 학습 가능한 피처 집계기 + 분류기만 사용
- **학습 시**: MOVE (DFS + 외삽 + 내삽) 추가 적용

---

### 2.4 성능 향상

#### VideoLT 결과 (ResNet-50 기준)

| 방법 | All AP | Head AP | Medium AP | Tail AP |
|------|--------|---------|-----------|---------|
| Baseline (no agg) | 0.499 | 0.675 | 0.553 | 0.376 |
| Framestack | 0.516 | 0.683 | 0.569 | 0.397 |
| **Ours** | **0.705** | **0.804** | **0.742** | **0.626** |

- 헤드 클래스: 약 18% 상대적 향상
- 테일 클래스: 약 58% 상대적 향상

#### 어블레이션 연구 (ResNet-101)

| 설정 | S-A | C-A | 외삽 | 내삽 | All | Tail |
|------|-----|-----|------|------|-----|------|
| (a) | - | - | - | - | 0.516 | 0.396 |
| (b) | ✓ | - | - | - | 0.681 | 0.595 |
| (c) | - | ✓ | - | - | 0.677 | 0.593 |
| (d) | ✓ | ✓ | - | - | 0.690 | 0.601 |
| (e) | ✓ | ✓ | ✓ | - | 0.710 | 0.632 |
| (f) | ✓ | ✓ | - | ✓ | 0.712 | 0.632 |
| **(g)** | ✓ | ✓ | ✓ | ✓ | **0.719** | **0.644** |

---

### 2.5 한계점

1. **Feature-level 방법의 한계**: Pretrained backbone의 피처를 직접 수정하지 않고 집계기로 보정하는 방식이므로, 근본적인 도메인 갭 해소에는 한계가 있습니다.

2. **하이퍼파라미터 민감성**: $\alpha$, $\gamma$, $\sigma$ 등 다양한 하이퍼파라미터 조정이 필요합니다.

3. **외삽의 불안정성**: 외삽 자체가 고불확실성 샘플 생성 위험이 있으며, 이를 $\omega \in [1,2]$로 제한하지만 완전한 해결책은 아닙니다.

4. **테일 클래스 편향**: 극단적인 불균형(imbalance ratio 0.01)에서도 성능 향상이 있지만, 매우 희귀한 클래스에서의 효과는 여전히 제한적입니다.

5. **계산 비용**: 두 개의 집계기를 병렬 운영하고 외삽·내삽 과정이 추가되어 학습 시간이 증가합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 위한 핵심 메커니즘

**① Vicinal Risk Minimization 기반 일반화**

VRM은 경험적 리스크 최소화(ERM)의 한계인 데이터 분포 편향을 완화합니다. 소수 클래스의 vicinal 분포를 확장함으로써 모델이 실제 데이터 분포를 더 잘 근사할 수 있습니다:

$$R_{vic}(h) = \frac{1}{N}\sum_{i=1}^{N} \ell(h(\tilde{\mathbf{z}}^i), \tilde{\mathbf{y}}^i)$$

이는 단순 ERM보다 더 넓은 분포를 커버하므로 미지의 데이터에 대한 **일반화 성능이 향상**됩니다.

**② 결정 경계 보정을 통한 일반화**

논문의 Figure 4에서 확인할 수 있듯이, MOVE 적용 전(Setting d)에는 테일 클래스로 갈수록 원본 샘플과 내삽 샘플 간 신뢰도 차이가 증가하여 결정 경계가 왜곡됩니다. MOVE 적용 후(Setting g)에는 이 차이가 모든 클래스에서 균등하게 낮아져 **더 선형적이고 일반화된 결정 경계**가 형성됩니다.

**③ 동적 외삽을 통한 데이터 다양성 확보**

$$|\mathbf{I}_s| \sim \mathcal{U}\left(\max(\lfloor \boldsymbol{\tau}_s \times T \rfloor, \sigma), T\right)$$

소수 클래스에 대해 더 다양한 마스크 $\mathbf{m}_s$가 생성되어 특정 패턴에 대한 과적합 억제 효과가 있습니다.

**④ Calibrated Interpolation의 Label Smoothing 효과**

$(1 - \tau_s) + \gamma$ 항은 소수 클래스에 큰 가중치를 부여하면서도 레이블 스무딩 효과를 가져 **과신(overconfidence) 방지**에 기여합니다.

### 3.2 다양한 시나리오에서의 일반화 검증

Imbalanced-MiniKinetics200에서 불균형 비율(0.01~0.1)을 달리했을 때 일관된 성능 향상이 관찰되었습니다:

| 불균형 비율 | Baseline AP | **Ours AP** |
|------------|-------------|-------------|
| 0.01 | 0.559 | **0.570** |
| 0.02 | 0.595 | **0.609** |
| 0.05 | 0.633 | **0.646** |
| 0.10 | 0.662 | **0.675** |

이는 MOVE가 다양한 불균형 정도에서 **강건한 일반화 성능**을 보임을 입증합니다.

### 3.3 Domain Generalization 관점

- Self-attentive aggregator는 **인트라-비디오 관계**(클래스 내부 시간적 관계)를 학습
- Codebook-attentive aggregator는 **인터-비디오 관계**(전역 코드북 기반 다른 비디오와의 관계)를 학습

두 집계기의 보완적 설계는 단일 모달리티나 단순 특징에 의존하지 않아 **다양한 도메인에서의 적용 가능성**을 높입니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

**① VLTR의 연구 방향 체계화**

기존 이미지 롱테일 인식 연구를 비디오 도메인에 단순 적용하는 것의 한계를 명확히 제시하고, **비디오 특화 세 가지 도전과제**를 정의함으로써 후속 연구의 기준점을 제공합니다.

**② 약한 레이블 환경에서의 표현 학습**

피처 집계를 통해 레이블 수준과 피처 수준을 일치시키는 접근법은 **비디오 이해의 다양한 약지도 학습 문제**에 응용 가능합니다 (예: 비디오 질문 답변, 이상 탐지 등).

**③ Vicinal Risk Minimization의 비디오 도메인 확장**

VRM을 비디오 롱테일 문제에 맞게 수정한 MOVE는 향후 **다양한 모달리티(오디오, 텍스트, 멀티모달)**에서의 불균형 학습 연구에 영감을 줍니다.

**④ 새로운 벤치마크 기여**

Imbalanced-MiniKinetics200의 제안은 다양한 불균형 시나리오에서 방법론을 평가할 수 있는 유연한 도구를 제공하여 **향후 VLTR 연구의 표준 벤치마크**로 활용될 가능성이 높습니다.

---

### 4.2 앞으로의 연구 시 고려할 점

**① End-to-End 학습 가능성 탐색**

현재 접근법은 Pretrained backbone을 고정하고 집계기만 학습합니다. 향후 연구에서는 **효율적인 파인튜닝(예: LoRA, Adapter)을 결합한 End-to-End 학습**을 통해 도메인 갭을 더욱 효과적으로 줄일 수 있습니다.

**② 멀티모달 VLTR로 확장**

비디오는 시각 정보 외에도 오디오, 텍스트(자막) 등 다양한 모달리티를 포함합니다. **CLIP, VideoCLIP 등 멀티모달 모델과의 결합**을 통해 소수 클래스의 표현을 더욱 풍부하게 만들 수 있습니다.

**③ 자기지도 학습(Self-supervised Learning)과의 융합**

최근 MAE-Video, VideoMAE 등 자기지도 사전학습 모델이 발전하고 있습니다. **자기지도 학습으로 더 나은 초기 피처를 획득**한 뒤 MOVE를 적용하면 롱테일 인식 성능이 더욱 향상될 것으로 예상됩니다.

**④ Class-incremental Learning과의 결합**

실세계에서는 새로운 클래스가 지속적으로 등장합니다. **점진적 학습(Continual/Incremental Learning)**과 VLTR을 결합하면, 새롭게 등장하는 소수 클래스를 효과적으로 학습하면서 기존 클래스의 성능을 유지하는 연구로 발전할 수 있습니다.

**⑤ Generative Model을 활용한 소수 클래스 증강**

DFS 기반 외삽보다 더 고품질의 소수 클래스 샘플 생성을 위해 **Diffusion Model이나 GAN을 활용한 비디오 데이터 증강**을 탐색할 수 있습니다.

**⑥ Transformer 기반 집계기 확장**

PSA는 Transformer의 Self-Attention을 변형한 것이나, **Video Swin Transformer, TimeSformer** 등 최신 비디오 Transformer 구조를 집계기로 활용하면 더욱 강력한 시공간 표현이 가능합니다.

**⑦ 실용적인 불균형 비율 측정**

실세계 데이터에서 불균형 비율은 동적으로 변합니다. **온라인 학습 환경**에서 클래스 빈도를 실시간으로 추정하고 $\tau$를 동적으로 업데이트하는 방법론이 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 방법 | 적용 도메인 | VLTR 특화 | 주요 차이점 |
|------|------|----------|------------|----------|------------|
| **LDAM** (Cao et al.) | 2019 | Label-distribution-aware margin loss | 이미지 | ✗ | 분류기 재조정, 비디오 약한 레이블 미고려 |
| **Decoupling** (Kang et al.) | 2020 | 표현·분류기 분리 학습 | 이미지 | ✗ | 2단계 학습, 비디오 시간 특성 미반영 |
| **Mixup** (Zhang et al.) | 2018 | 선형 보간 기반 데이터 증강 | 이미지 | ✗ | 클래스 빈도 고려 없는 단순 내삽 |
| **RIDE** (Wang et al.) | 2021 | 다중 전문가 앙상블 | 이미지 | ✗ | 추론 비용 증가 |
| **VideoLT + Framestack** (Zhang et al.) | 2021 | 클래스별 다른 프레임 수 스택 | **비디오** | ✓ | 스니펫 레벨 처리, 약한 레이블 문제 미해결 |
| **PaCo** (Cui et al.) | 2021 | Parametric contrastive learning | 이미지 | ✗ | 대조 학습 기반, 비디오 미적용 |
| **BALLAD** (Ma et al.) | 2021 | 양방향 학습 | 이미지 | ✗ | 이미지 전용 |
| **MOVE (본 논문)** | 2022 | PSA + NetVLAD + VRM 기반 데이터 확장 | **비디오** | ✓ | 세 가지 도전 동시 해결, 소수 지향 분포 확장 |

### 주목할 만한 차이점 분석

**Framestack vs MOVE:**

- Framestack: 스니펫 레벨에서 프레임 스택 → 약한 레이블 문제 미해결
- MOVE: 비디오 레벨 집계 → 레이블 수준 불일치 해결, MOVE로 분포 확장

**Decoupling vs MOVE:**

- Decoupling: 표현 학습과 분류기 학습을 단계적으로 분리 → 비디오 약한 레이블 환경에 부적합
- MOVE: 집계기를 통한 동시적 표현 개선과 MOVE를 통한 경계 보정

**Mixup vs Calibrated Interpolation:**

일반 Mixup:

$$\tilde{z} = \lambda z^i + (1-\lambda) z^j, \quad \tilde{y} = \lambda y^i + (1-\lambda) y^j$$

MOVE의 Calibrated Interpolation:

$$\tilde{y}_s = \min\left((\lambda\hat{y}^i_s + (1-\lambda)\hat{y}^j_s) \times ((1-\tau_s) + \gamma),\ 1\right)$$

핵심 차이: $\tau_s$를 통해 **클래스 빈도를 보간 공간 할당에 반영** → 소수 클래스에 더 넓은 feature space 할당

---

## 참고 자료 및 출처

### 논문 원문 (주 참고 자료)

1. **Moon, W., Seong, H. S., & Heo, J. P. (2022).** "Minority-Oriented Vicinity Expansion with Attentive Aggregation for Video Long-Tailed Recognition." *arXiv:2211.13471v1 [cs.CV]* (AAAI 2023). [제공된 PDF 원문]
   - GitHub: https://github.com/wjun0830/MOVE

### 논문 내 인용된 핵심 참고문헌

2. **Zhang, X., et al. (2021b).** "VideoLT: Large-scale Long-tailed Video Recognition." *ICCV 2021*, 7960–7969.

3. **Chapelle, O., et al. (2000).** "Vicinal risk minimization." *NeurIPS 2000*, 13.

4. **Vaswani, A., et al. (2017).** "Attention is all you need." *NeurIPS 2017*, 30.

5. **Arandjelovic, R., et al. (2016).** "NetVLAD: CNN architecture for weakly supervised place recognition." *CVPR 2016*, 5297–5307.

6. **Cao, K., et al. (2019).** "Learning imbalanced datasets with label-distribution-aware margin loss." *NeurIPS 2019*, 32.

7. **Kang, B., et al. (2020).** "Decoupling Representation and Classifier for Long-Tailed Recognition." *ICLR 2020*.

8. **Zhang, H., et al. (2018).** "mixup: Beyond empirical risk minimization." *ICLR 2018*.

9. **Tan, J., et al. (2020).** "Equalization loss for long-tailed object recognition." *CVPR 2020*.

10. **Cui, Y., et al. (2019).** "Class-balanced loss based on effective number of samples." *CVPR 2019*, 9268–9277.

11. **Wang, X., et al. (2021).** "Long-tailed Recognition by Routing Diverse Distribution-Aware Experts." *ICLR 2021*.

12. **Lin, J., Gan, C., & Han, S. (2019).** "TSM: Temporal shift module for efficient video understanding." *ICCV 2019*, 7083–7093.

13. **Dosovitskiy, A., et al. (2022).** "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2022*.

14. **Park, S., et al. (2021).** "Influence-balanced loss for imbalanced visual classification." *ICCV 2021*, 735–744.

15. **Zhong, Z., et al. (2021a).** "Improving calibration for long-tailed recognition." *CVPR 2021*, 16489–16498.

---

> **정확도 관련 고지**: 본 답변은 제공된 PDF 원문(arXiv:2211.13471v1)을 기반으로 작성되었습니다. 2020년 이후 최신 연구 비교 표의 일부 상세 수치는 원문에 직접 포함되지 않은 내용이 있으며, 이 경우 원문의 Related Work 섹션과 참고문헌을 토대로 기술하였습니다. 확인되지 않은 외부 논문의 상세 성능 수치는 의도적으로 기재하지 않았습니다.
