# Improving Unsupervised Image Clustering With Robust Learning

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문은 기존 비지도 이미지 클러스터링 모델들이 공통적으로 겪는 두 가지 문제를 지적합니다:

1. **오분류(Faulty Predictions)**: 대안적 학습 목표(alternative objectives)로 인한 잘못된 예측
2. **과신뢰(Overconfident Results)**: 엔트로피 기반 균형화로 인한 지나치게 높은 예측 신뢰도

이를 해결하기 위해 **Robust Learning** 개념을 비지도 클러스터링에 최초로 체계적으로 적용한 **RUC(Robust learning for Unsupervised Clustering)** 모델을 제안합니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| 핵심 아이디어 | 기존 클러스터링 결과를 **노이즈가 포함된 pseudo-label**로 취급하여 robust learning 적용 |
| 구조적 유연성 | 어떤 클러스터링 모델에도 **애드온(add-on) 모듈**로 활용 가능 |
| 성능 향상 | SCAN+RUC: CIFAR-10(90.3%), CIFAR-20(54.3%), STL-10(86.7%), ImageNet-50(78.5%) |
| 추가 효과 | 모델 **신뢰도 보정(calibration)** 개선 및 **적대적 노이즈 강건성** 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 비지도 클러스터링의 핵심 문제:

$$\text{Overconfident Prediction: } \max(\mathbf{y}) \approx 1 \text{ (저엔트로피 예측)}$$

초기 학습 단계에서 불확실한 샘플이 잘못된 클러스터에 할당되면, 이후 모델이 점점 더 잘못된 분류에 과신뢰하는 **노이즈 누적 현상**이 발생합니다. 이는 기존 방법들이 사용하는 엔트로피 기반 균형화(entropy-based balancing)의 부작용입니다.

### 2.2 제안하는 방법 (수식 포함)

#### Phase 1: 클린 샘플 추출 (Clean Sample Extraction)

학습 데이터를 클린셋 $\mathcal{X}$와 언클린셋 $\mathcal{U}$로 분리합니다:

$$\mathcal{D} = \mathcal{X} \cup \mathcal{U}$$

**전략 1 - Confidence-based:**

$$\mathcal{X} = \{(\mathbf{x}, \mathbf{y}) \in \mathcal{D} \mid \max(\mathbf{y}) > \tau_1\}$$

- $\tau_1 = 0.99$로 설정하여 가장 전형적인 샘플만 선택

**전략 2 - Metric-based:**

$$\mathbf{y}' = k\text{-NN}(h_\psi(\mathbf{x}))$$

$$\mathcal{X} = \{(\mathbf{x}, \mathbf{y}) \in \mathcal{D} \mid \arg\max(\mathbf{y}) = \arg\max(\mathbf{y}')\}$$

- SimCLR 등 비지도 임베딩 모델 $h_\psi$의 k-NN 분류 결과와 pseudo-label 비교

**전략 3 - Hybrid:**
- 두 전략 모두 클린으로 판단한 샘플만 $\mathcal{X}$에 포함

#### Phase 2: Robust Learning으로 재학습

**MixUp 증강:**

$$\lambda \sim \text{Beta}(\alpha, \alpha)$$

$$\lambda' = \max(\lambda, 1-\lambda)$$

$$\mathbf{x}' = \lambda'\mathbf{x}_1 + (1-\lambda')\mathbf{x}_2$$

$$\mathbf{y}' = \lambda'\mathbf{y}_1 + (1-\lambda')\mathbf{y}_2$$

**Label Smoothing:**

$$\tilde{\mathbf{y}} = (1-\epsilon) \cdot \mathbf{y} + \frac{\epsilon}{(C-1)} \cdot (\mathbf{1} - \mathbf{y})$$

여기서 $C$는 클래스 수, $\epsilon \sim \text{Uniform}(0,1)$

**손실 함수:**

$$\hat{\mathcal{X}}, \hat{\mathcal{U}} = \text{MixMatch}(\mathcal{X}, \mathcal{U})$$

$$\mathcal{L}_{\hat{\mathcal{X}}} = \frac{1}{|\hat{\mathcal{X}}|} \sum_{\hat{\mathbf{x}}, \hat{\mathbf{y}} \in \hat{\mathcal{X}}} H(\hat{\mathbf{y}}, f_\theta(\hat{\mathbf{x}}))$$

$$\mathcal{L}_{\hat{\mathcal{U}}} = \frac{1}{|\hat{\mathcal{U}}|} \sum_{\hat{\mathbf{u}}, \hat{\mathbf{q}} \in \hat{\mathcal{U}}} ||\hat{\mathbf{q}} - f_\theta(\hat{\mathbf{u}})||_2^2$$

$$\mathcal{L}_{\mathcal{X}^s} = \frac{1}{|\mathcal{X}|} \sum_{\mathbf{x}, \tilde{\mathbf{y}} \in \mathcal{X}} H(\tilde{\mathbf{y}}, f_\theta(\phi_A(\mathbf{x})))$$

$$\mathcal{L}(\theta; \mathcal{X}, \hat{\mathcal{X}}, \hat{\mathcal{U}}) = \mathcal{L}_{\mathcal{X}^s} + \mathcal{L}_{\hat{\mathcal{X}}} + \lambda_U \mathcal{L}_{\hat{\mathcal{U}}}$$

**Co-refinement (Co-training):**

$$\bar{\mathbf{y}} = (1 - w^{(2)}) \cdot \mathbf{y} + w^{(2)} \cdot f_{\theta^{(2)}}(\mathbf{x})$$

$$\bar{\mathbf{y}} = \text{Sharpen}(\bar{\mathbf{y}}, T)$$

$$\bar{\mathbf{q}} = \frac{1}{2M} \sum_m \left(f_{\theta^{(1)}}(\mathbf{u}_m) + f_{\theta^{(2)}}(\mathbf{u}_m)\right)$$

$$\bar{\mathbf{q}} = \text{Sharpen}(\bar{\mathbf{q}}, T)$$

**Co-refurbishing (언클린 샘플 재활용):**

$$\mathbf{p} = f_{\theta^{(k)}}(\mathbf{u}), \quad k = \arg\max_{k'}\left(\max(f_{\theta^{(k')}}(\mathbf{u}))\right)$$

$$\mathcal{X} \leftarrow \mathcal{X} \cup \{(\mathbf{u}, \mathbf{1}_\mathbf{p}) \mid \max(\mathbf{p}) > \tau_2\}$$

**신뢰도 보정 지표 (ECE):**

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|$$

**적대적 공격 (FGSM, BIM):**

$$\mathbf{x}^{adv} = \mathbf{x} + \epsilon \cdot \text{sgn}(\nabla_\mathbf{x} J(\theta, \mathbf{x}, \mathbf{y}))$$

$$\mathbf{x}^{adv}_i = \text{clip}_{\mathbf{x},\epsilon}(\mathbf{x}^{adv}_{i-1} + \epsilon \cdot \text{sgn}(\nabla_{\mathbf{x}^{adv}_{i-1}} J(\theta, \mathbf{x}^{adv}_{i-1}, \mathbf{y})))$$

### 2.3 모델 구조

```
[비지도 클러스터링 모델 (SCAN/TSUC)]
            ↓ pseudo-labels
[샘플링 전략 (Confidence/Metric/Hybrid)]
      ↓                    ↓
  클린셋 X              언클린셋 U
      ↓                    ↓
[Co-refinement] ← 두 네트워크 fθ(1), fθ(2) 교차 학습
      ↓
[MixMatch + Label Smoothing + Strong Augmentation]
      ↓
[Co-refurbishing: U → X (조건 충족 시)]
      ↓
[최종 클러스터링 결과]
```

- **백본**: ResNet18
- **학습**: 200 epochs, SGD (momentum=0.9, weight decay=0.0005)
- **배치 크기**: STL-10: 100, CIFAR: 200

### 2.4 성능 향상

| 데이터셋 | SCAN (기준) | SCAN+RUC (Best) | 향상폭 |
|---------|------------|-----------------|--------|
| CIFAR-10 | 88.7% | **90.3%** | +1.6pp |
| CIFAR-20 | 50.6% | **54.3%** | +3.7pp |
| STL-10 | 81.4% | **86.7%** | **+5.3pp** |
| ImageNet-50 | 76.8% | **78.5%** | +1.7pp |

**Ablation Study (STL-10 기준):**

| 설정 | Last Acc | Best Acc |
|------|----------|----------|
| RUC 전체 | **86.7** | **86.8** |
| co-training 제거 | 86.2 | 86.4 |
| label smoothing 제거 | 85.5 | 85.8 |
| MixMatch만 사용 | 85.2 | 85.4 |

**신뢰도 보정:**
- SCAN ECE: 18.5 → SCAN+RUC ECE: **17.0** (STL-10 기준)

### 2.5 한계점

1. **초기 pseudo-label 품질 의존성**: 기반 클러스터링 모델의 품질이 낮을수록 RUC의 효과가 제한적
2. **하이퍼파라미터 민감성**: $\tau_1 = 0.99$와 같이 신중한 임계값 설정이 필요
3. **계산 비용**: 두 개의 네트워크를 병렬로 학습하는 co-training 구조로 인해 단일 네트워크 대비 약 2배의 계산 자원 필요 (실제로는 4×TITAN Xp에서 12시간 이내)
4. **샘플링 전략의 한계**: Hybrid 전략이 항상 최선이 아님 — 상황에 따라 Metric 또는 Confidence 전략이 더 높은 성능을 보일 수 있음
5. **도메인 한계**: 의료 이미지, 위성 이미지 등 특수 도메인에서의 검증 부재

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상 메커니즘

RUC의 일반화 성능 향상은 다음 세 가지 메커니즘으로 설명됩니다:

#### (1) Label Smoothing을 통한 과적합 방지

$$\tilde{\mathbf{y}} = (1-\epsilon) \cdot \mathbf{y} + \frac{\epsilon}{C-1} \cdot (\mathbf{1} - \mathbf{y})$$

- 노이즈 레이블에 대한 과신뢰를 방지하여 **모델이 더 일반적인 특징을 학습**
- 소프트 레이블로 인해 클래스 경계 근처의 샘플들에 대한 일반화 향상

#### (2) MixUp 기반 정규화

$$\mathbf{x}' = \lambda'\mathbf{x}_1 + (1-\lambda')\mathbf{x}_2, \quad \lambda' = \max(\lambda, 1-\lambda)$$

- MixUp 보간으로 생성된 가상 샘플이 **메모리제이션을 어렵게** 만들어 일반화 향상
- 논문에서 직접 인용: "a large amount of extra virtual examples from MixUp interpolation makes memorization hard to achieve"

#### (3) Co-training을 통한 편향 감소

두 네트워크 $f_{\theta^{(1)}}, f_{\theta^{(2)}}$의 서로 다른 초기화와 학습 경로로 인해:
- 샘플링 편향(sampling bias) 누적 방지
- 앙상블 예측을 통한 더 안정적인 pseudo-label 생성
- 실험적으로 CIFAR-20에서 co-training 유무에 따라 정확도 차이가 36.3% → 39.6%로 나타남

#### (4) 적대적 노이즈 강건성

FGSM과 BIM 공격 실험에서 SCAN+RUC가 다른 모든 기준 모델보다 우수한 성능 유지:
- Label smoothing이 적대적 그래디언트의 크기를 줄임
- 이는 **분포 외(out-of-distribution) 샘플에 대한 일반화** 향상을 의미

#### (5) 애드온 모듈의 범용성

- TSUC, SCAN 두 가지 다른 클러스터링 기반 모델에 모두 적용 시 일관된 성능 향상
- 이는 RUC가 특정 모델에 과적합되지 않은 **범용적 일반화 능력**을 가짐을 시사

### 3.2 일반화 한계와 개선 가능성

**현재 한계:**
- 검증 데이터셋이 CIFAR-10/20, STL-10, ImageNet-50으로 제한됨
- 클래스 수가 적은(10~50개) 데이터셋 위주의 검증

**개선 가능성:**
- 더 강력한 사전학습 표현(ViT, DINO 등) 활용 시 일반화 성능 추가 향상 기대
- 더 다양한 샘플링 전략(예: 불확실성 기반, 다양성 기반) 적용

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

#### (1) 방법론적 패러다임 전환
- 비지도 클러스터링과 robust learning의 결합이라는 **새로운 연구 방향** 제시
- pseudo-label을 "완벽한 레이블"이 아닌 "노이즈가 포함된 레이블"로 취급하는 관점의 일반화 가능성

#### (2) 애드온 모듈 연구의 활성화
- 기존 모델을 대체하는 것이 아닌 **보완하는 방식**의 연구 촉진
- 이후 GCC, NNM, TCL 등 다양한 클러스터링 모델과의 결합 연구를 촉발

#### (3) 신뢰도 보정 연구
- ECE를 클러스터링 평가지표로 활용하는 관행 정립
- 비지도 학습에서의 **캘리브레이션 연구** 활성화

#### (4) 자기지도 학습과의 연계
- RUC가 SimCLR, MoCo 등 자기지도 학습 표현에 의존함으로써 향후 **더 강력한 표현 학습 모델과의 시너지** 연구 유도

### 4.2 향후 연구 시 고려할 점

| 고려 항목 | 현재 상태 | 개선 방향 |
|-----------|-----------|-----------|
| **표현 학습 백본** | ResNet18 + SimCLR | ViT, DINO, MAE 등 강력한 사전학습 모델 활용 |
| **샘플링 전략** | Confidence/Metric/Hybrid | 학습 기반 적응형 샘플링 전략 |
| **스케일 확장** | ImageNet-50까지 검증 | 전체 ImageNet-1K, iNaturalist 등 |
| **클러스터 수 자동화** | 클러스터 수 고정 | 자동 클러스터 수 결정 메커니즘 |
| **준지도 학습과 통합** | 순수 비지도 | 소수의 레이블 활용 시 성능 향상 탐구 |
| **계산 효율성** | 2개 네트워크 병렬 학습 | 지식 증류(knowledge distillation) 기반 경량화 |
| **다양한 도메인 적용** | 자연 이미지 중심 | 의료·위성·음성 등 특수 도메인 적용 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | CIFAR-10 | CIFAR-20 | STL-10 | 주요 특징 |
|------|------|----------|----------|--------|-----------|
| **SCAN** (Van Gansbeke et al., ECCV 2020) | 최근접 이웃 기반 클러스터링 | 88.7% | 50.6% | 81.4% | 다단계 정제 |
| **RUC** (Park et al., 2021) | Robust Learning + SCAN | **90.3%** | **54.3%** | **86.7%** | 본 논문 |
| **GCC** (Zhong et al., CVPR 2021) | 그래프 대조 학습 | 85.6% | 47.2% | 78.8% | 그래프 구조 활용 |
| **NNM** (Dang et al., ICCV 2021) | 최근접 이웃 매칭 | 90.5% | 52.9% | 86.5% | 메모리 뱅크 활용 |
| **TCL** (Huang et al., CVPR 2022) | 삼각 상호 학습 | 91.4% | 55.9% | 88.9% | 앙상블 기반 정제 |
| **TWIST** (Zhan et al., CVPR 2022) | 양방향 자기지도 | 92.3% | 57.4% | 90.2% | 트랜스포머 활용 |

> **⚠️ 주의**: 위 표의 GCC, NNM, TCL, TWIST 수치는 각 논문의 reported 결과에 기반하나, 실험 설정(backbone, 전처리 등)이 완전히 동일하지 않을 수 있습니다. 정확한 비교를 위해서는 각 논문의 원본을 참조하시기 바랍니다.

### 주요 트렌드 비교

```
2020: SCAN (다단계 정제) → 2021: RUC (Robust Learning 결합)
                        → 2021: GCC (그래프 대조학습)
2022: TCL (앙상블 삼각학습) → 2022: TWIST (Transformer 기반)
2023~: DINO/iBOT 기반 표현 + 클러스터링 통합 연구
```

**RUC의 차별성:**
- 기존 연구들이 새로운 클러스터링 아키텍처를 설계하는 반면, RUC는 **기존 모델을 개선하는 범용 모듈**이라는 독특한 위치
- Robust Learning의 체계적 적용은 이후 연구들이 pseudo-label 노이즈 처리를 명시적으로 고려하는 계기 제공

---

## 참고자료 및 출처

**주요 참고 논문 (본 논문 내 인용 기준):**

1. **본 논문**: Park, S. et al. "Improving Unsupervised Image Clustering With Robust Learning." arXiv:2012.11150v2, 2021.
2. **SCAN**: Van Gansbeke, W. et al. "SCAN: Learning to classify images without labels." ECCV 2020.
3. **MixMatch**: Berthelot, D. et al. "MixMatch: A holistic approach to semi-supervised learning." NeurIPS 2019.
4. **DivideMix**: Li, J. et al. "Dividemix: Learning with noisy labels as semi-supervised learning." ICLR 2020.
5. **SimCLR**: Chen, T. et al. "A simple framework for contrastive learning of visual representations." ICML 2020.
6. **IIC**: Ji, X. et al. "Invariant information clustering for unsupervised image classification and segmentation." ICCV 2019.
7. **Co-teaching**: Han, B. et al. "Co-teaching: Robust training of deep neural networks with extremely noisy labels." NeurIPS 2018.
8. **Label Smoothing**: Lukasik, M. et al. "Does label smoothing mitigate label noise?" ICML 2020.
9. **TSUC**: Han, S. et al. "Mitigating embedding and class assignment mismatch in unsupervised image classification." ECCV 2020.
10. **ResNet**: He, K. et al. "Deep residual learning for image recognition." CVPR 2016.
11. **RandAugment**: Cubuk, E.D. et al. "Randaugment: Practical automated data augmentation." CVPR Workshops 2020.
12. **ECE**: Guo, C. et al. "On calibration of modern neural networks." ICML 2017.
13. **GitHub 코드**: https://github.com/deu30303/RUC
