# Contrastive Adaptation Network for Unsupervised Domain Adaptation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 비지도 도메인 적응(UDA) 방법들은 **클래스 정보를 무시한 채** 도메인 수준에서 분포 차이를 줄이는 방식(class-agnostic)을 사용하여, 서로 다른 클래스의 샘플들이 잘못 정렬(misalignment)되는 문제가 발생한다. 이 논문은 **클래스 인식(class-aware) 도메인 정렬**을 통해 이 문제를 해결한다.

### 주요 기여 3가지

| 기여 | 내용 |
|------|------|
| **새로운 메트릭 (CDD)** | 클래스 내(intra-class) 도메인 불일치는 최소화, 클래스 간(inter-class) 도메인 불일치는 최대화 |
| **새로운 네트워크 (CAN)** | CDD를 종단간(end-to-end) 학습으로 최적화할 수 있는 구조 제안 |
| **성능 향상** | Office-31 벤치마크에서 당시 최고 성능(90.6%), VisDA-2017에서 87.2% 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 1: 클래스 비인식 정렬(Class-agnostic Alignment)**

기존 MMD, JMMD 기반 방법(DAN, JAN)은 도메인 수준에서만 분포를 맞추기 때문에, 아래 그림처럼 서로 다른 클래스의 샘플들이 잘못 정렬될 수 있다.

$$\text{(기존 방법)} \quad \min_\theta \hat{D}^{mmd}_l = \frac{1}{n_s^2}\sum_{i,j}k_l(\phi_l(x^s_i), \phi_l(x^s_j)) + \frac{1}{n_t^2}\sum_{i,j}k_l(\phi_l(x^t_i), \phi_l(x^t_j)) - \frac{2}{n_s n_t}\sum_{i,j}k_l(\phi_l(x^s_i), \phi_l(x^t_j))$$

이 식은 클래스 레이블을 전혀 고려하지 않아, 타깃 도메인의 클래스 A 샘플이 소스 도메인의 클래스 B 샘플에 정렬되어도 loss가 줄어들 수 있다.

**문제 2: 결정 경계 근처의 비판별적 특징 학습**

클래스 인식 없이 정렬하면 결정 경계 근처에 많은 준최적(sub-optimal) 해가 생기고, 소스 데이터에 과적합되어 타깃 도메인에서 일반화 성능이 떨어진다.

---

### 2.2 제안 방법 (수식 포함)

#### 2.2.1 Contrastive Domain Discrepancy (CDD)

MMD를 조건부 분포 $P(\phi(X^s)|Y^s)$와 $Q(\phi(X^t)|Y^t)$ 사이의 차이로 확장한다.

마스킹 함수를 정의한다:

$$\mu_{cc'}(y, y') = \begin{cases} 1 & \text{if } y = c,\ y' = c' \\ 0 & \text{otherwise} \end{cases}$$

클래스 $c_1$, $c_2$ 간의 커널 평균 임베딩 추정:

$$\hat{D}^{c_1 c_2}(\hat{y}^t_{1:n_t}, \phi) = e_1 + e_2 - 2e_3$$

$$e_1 = \frac{\sum_{i=1}^{n_s}\sum_{j=1}^{n_s} \mu_{c_1 c_1}(y^s_i, y^s_j)\, k(\phi(x^s_i), \phi(x^s_j))}{\sum_{i=1}^{n_s}\sum_{j=1}^{n_s} \mu_{c_1 c_1}(y^s_i, y^s_j)}$$

$$e_2 = \frac{\sum_{i=1}^{n_t}\sum_{j=1}^{n_t} \mu_{c_2 c_2}(\hat{y}^t_i, \hat{y}^t_j)\, k(\phi(x^t_i), \phi(x^t_j))}{\sum_{i=1}^{n_t}\sum_{j=1}^{n_t} \mu_{c_2 c_2}(\hat{y}^t_i, \hat{y}^t_j)}$$

$$e_3 = \frac{\sum_{i=1}^{n_s}\sum_{j=1}^{n_t} \mu_{c_1 c_2}(y^s_i, \hat{y}^t_j)\, k(\phi(x^s_i), \phi(x^t_j))}{\sum_{i=1}^{n_s}\sum_{j=1}^{n_t} \mu_{c_1 c_2}(y^s_i, \hat{y}^t_j)}$$

- $c_1 = c_2 = c$이면 → **intra-class domain discrepancy** (최소화 대상)
- $c_1 \neq c_2$이면 → **inter-class domain discrepancy** (최대화 대상)

최종 CDD는 두 항의 대비(contrastive) 형태:

$$\hat{D}^{cdd} = \underbrace{\frac{1}{M}\sum_{c=1}^{M} \hat{D}^{cc}(\hat{y}^t_{1:n_t}, \phi)}_{\text{intra-class (최소화)}} - \underbrace{\frac{1}{M(M-1)}\sum_{c=1}^{M}\sum_{\substack{c'=1 \\ c' \neq c}}^{M} \hat{D}^{cc'}(\hat{y}^t_{1:n_t}, \phi)}_{\text{inter-class (최대화)}}$$

#### 2.2.2 다중 레이어 CDD 집계

여러 FC 레이어에 걸쳐 CDD를 누적한다:

$$\hat{D}^{cdd}_{\mathcal{L}} = \sum_{l=1}^{L} \hat{D}^{cdd}_l$$

#### 2.2.3 전체 목적 함수

소스 데이터에 대한 크로스엔트로피 손실과 CDD 패널티를 결합:

$$\ell^{ce} = -\frac{1}{n'_s}\sum_{i'=1}^{n'_s} \log P_\theta(y^s_{i'} | x^s_{i'})$$

$$\min_\theta \ell = \ell^{ce} + \beta \hat{D}^{cdd}_{\mathcal{L}}$$

여기서 $\beta$는 도메인 불일치 패널티의 가중치이며, 실험에서 $\beta = 0.3$으로 설정.

---

### 2.3 모델 구조 (CAN)

```
[Source Data]──────────────────────────────────┐
                                                 ▼
[ImageNet Pretrained ResNet-50/101]    Cross-Entropy Loss (ℓ_ce)
(공유 합성곱 레이어)                              │
        │                                        │
        ▼                                        │
[Task-Specific FC Layers]                        │
   φ₁, φ₂, ..., φ_L                            │
        │                                        │
        ├── [Clustering 모듈] ──→ 타깃 의사 레이블 추정 (구형 K-means)
        │        │                               │
        │        ▼                               │
        │   ambiguous 샘플/클래스 필터링            │
        │                                        │
        ▼                                        │
[CDD 계산 모듈]                                  │
   intra-class 최소화                             │
   inter-class 최대화                             │
        │                                        │
        └──────────────── β × D̂_L^cdd ──────────┘
                                 ▼
                        Back-propagation (SGD)
```

**주요 구성 요소:**

1. **특징 추출기**: ResNet-50(Office-31), ResNet-101(VisDA-2017), ImageNet 사전 학습 모델 사용, 마지막 FC 레이어를 task-specific FC 레이어로 교체

2. **교대 최적화 (Alternative Optimization, AO)**:
   - **Step 1**: 현재 특징으로 타깃 샘플의 의사 레이블을 구형 K-means 클러스터링으로 추정
   - **Step 2**: 추정된 레이블로 CDD 계산 후 역전파로 특징 적응
   - 두 단계를 교대로 반복

3. **모호한 샘플/클래스 필터링**:
   - 클러스터 중심에서 멀리 떨어진 샘플 제거: $\tilde{\mathcal{T}} = \{(x^t, \hat{y}^t) \mid dist(\phi_1(x^t), O^t_{(\hat{y}^t)}) < D_0,\ x^t \in \mathcal{T}\}$
   - 특정 클래스에 할당된 샘플 수가 $N_0$ 미만이면 해당 클래스 제외

4. **클래스 인식 샘플링 (Class-Aware Sampling, CAS)**: 미니 배치마다 각 클래스에서 소스/타깃 샘플을 균형 있게 샘플링

---

### 2.4 성능 향상

**Office-31 (ResNet-50 기반):**

| Method | A→W | D→W | W→D | A→D | D→A | W→A | **Average** |
|--------|-----|-----|-----|-----|-----|-----|-------------|
| Source-only | 68.4 | 96.7 | 99.3 | 68.9 | 62.5 | 60.7 | 76.1 |
| DAN | 80.5 | 97.1 | 99.6 | 78.6 | 63.6 | 62.8 | 80.4 |
| JAN | 85.4 | 97.4 | 99.8 | 84.7 | 68.6 | 70.0 | 84.3 |
| MADA | 90.0 | 97.4 | 99.6 | 87.8 | 70.3 | 66.4 | 85.2 |
| **CAN** | **94.5** | **99.1** | **99.8** | **95.0** | **78.0** | **77.0** | **90.6** |

**VisDA-2017 (ResNet-101 기반):** CAN **87.2%** (SE 우승 84.3% 대비 +2.9%)

**Ablation Study 결과 (평균 정확도):**

| Method | Office-31 | VisDA-2017 |
|--------|-----------|------------|
| w/o AO | 88.1 | 77.5 |
| w/o CAS | 89.1 | 81.6 |
| **CAN (full)** | **90.6** | **87.2** |

---

### 2.5 한계

1. **타깃 레이블 추정의 노이즈**: 클러스터링 기반 의사 레이블은 도메인 시프트 초기에 부정확할 수 있으며, 특히 클래스 간 경계가 모호한 경우 성능이 저하될 수 있다.

2. **클래스 수(M) 사전 지식 필요**: 클러스터 수 = 클래스 수로 설정하기 때문에 타깃 도메인의 클래스 수를 알아야 한다 (부분 도메인 적응 시나리오에 직접 적용 어려움).

3. **계산 비용**: 교대 최적화 과정에서 전체 데이터셋에 대한 클러스터링을 반복 수행해야 하므로 대규모 데이터셋에서 비용이 증가한다.

4. **하이퍼파라미터 민감도**: $D_0$(거리 임계값), $N_0$(최소 샘플 수), $\beta$ 등 여러 하이퍼파라미터가 존재하며, 태스크마다 조정이 필요하다.

5. **단일 소스 도메인 가정**: 다중 소스 도메인 상황이나 오픈셋(open-set) 시나리오에는 직접 적용이 어렵다.

---

## 3. 모델의 일반화 성능 향상 관련 분석

### 3.1 inter-class 불일치 최대화의 역할

논문의 핵심 통찰은 **inter-class 도메인 불일치를 최대화**하는 것이 일반화 성능에 기여한다는 점이다.

$$\hat{D}^{cdd} = \underbrace{\frac{1}{M}\sum_{c}\hat{D}^{cc}}_{\text{최소화 → 동일 클래스 응집}} - \underbrace{\frac{1}{M(M-1)}\sum_{c \neq c'}\hat{D}^{cc'}}_{\text{최대화 → 다른 클래스 분리}}$$

**수식적 직관**: intra-class 불일치만 최소화하면 결정 경계 근처에 준최적 해가 남을 수 있다. Inter-class 불일치를 최대화함으로써 각 클래스의 표현이 결정 경계에서 더 멀리 밀려나게 되어, 타깃 도메인에서의 **마진(margin)이 증가**한다.

Ablation에서 "intra only" vs "CAN":
- Office-31: 89.5% → 90.6% (+1.1%)
- VisDA-2017: 83.9% → 87.2% (+3.3%)

특히 VisDA-2017의 큰 향상폭은 더 어려운 도메인 시프트(합성→실사)에서 inter-class 분리가 더 큰 역할을 함을 보여준다.

### 3.2 점진적 학습(Progressive Learning)에 의한 일반화

훈련 초기에는 도메인 시프트로 인해 일부 클래스만 포함되지만, 훈련이 진행될수록:

1. 모델 정확도 향상 → 더 많은 클래스의 의사 레이블이 신뢰 가능해짐
2. CDD 패널티로 intra-class 불일치 감소, inter-class 불일치 증가 → 어려운 클래스도 점차 포함

이 **점진적 포함 메커니즘**이 특정 클래스에 과적합되는 것을 방지하고 전체적인 일반화를 높인다.

### 3.3 MMD의 노이즈 강건성

CDD는 MMD 기반이므로 RKHS에서의 평균 임베딩을 사용한다. 의사 레이블 노이즈가 있어도 **분포의 충분 통계량(sufficient statistics)**에는 큰 영향을 미치지 않아 (특히 데이터가 많을 때), 노이즈에 강건하다.

실험에서 "w/o AO" (동시 업데이트)도 JAN, DAN보다 성능이 높았다는 점이 이를 입증한다.

### 3.4 t-SNE 시각화

t-SNE 결과에서 CAN으로 학습된 특징은 JAN 대비:
- **intra-class compactness** 증가: 같은 클래스끼리 더 촘촘히 모임
- **inter-class margin** 증가: 다른 클래스 간 거리 증가

이는 직접적으로 타깃 도메인에서 분류기의 일반화 성능 향상으로 이어진다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

**① 클래스 인식 도메인 적응의 표준화**

CAN 이후, "class-agnostic alignment는 부족하다"는 인식이 UDA 연구 커뮤니티에 널리 퍼졌다. 이후 연구들이 다양한 방식으로 클래스 정보를 활용하는 방향으로 전환되는 촉매가 되었다.

**② Contrastive Learning과 도메인 적응의 결합**

CDD의 intra-class 최소화 + inter-class 최대화 구조는 이후 대조 학습(Contrastive Learning, SimCLR, MoCo 등)과 도메인 적응을 결합하는 연구들의 선구적 역할을 했다.

**③ 의사 레이블(Pseudo Label)과 교대 최적화의 활용**

클러스터링 기반 의사 레이블 추정 + 교대 최적화 패러다임은 이후 자기 훈련(self-training) 기반 UDA 연구들에 직접적 영향을 미쳤다.

**④ 결정 경계 인식 적응**

결정 경계에서 멀리 떨어진 특징을 학습하려는 시도(MCD와 유사한 방향)가 이후 더욱 발전하였다.

---

### 4.2 앞으로의 연구 시 고려할 점

**① 더 강력한 의사 레이블 품질 향상**

클러스터링 기반 레이블 추정은 초기 단계에서 부정확하다. 이후 연구에서는:
- **Self-training with confidence thresholding**: 높은 신뢰도 샘플만 선택
- **Consistency regularization**: 여러 augmentation에서 일관된 예측을 요구
- **Prototype-based pseudo labeling**: 더 안정적인 클래스 프로토타입 사용

**② 오픈셋/부분 도메인 적응으로 확장**

CAN은 소스와 타깃이 동일한 클래스 집합을 가진다고 가정한다. 실제 시나리오에서는:
- 타깃에만 있는 클래스 (open-set DA)
- 소스의 일부 클래스만 타깃에 존재 (partial DA)
- 위 두 경우 혼합 (universal DA)
에 대한 확장이 필요하다.

**③ 트랜스포머(Transformer) 백본과의 결합**

ResNet 기반 CAN을 Vision Transformer (ViT)나 CLIP 등 더 강력한 사전 학습 모델과 결합하면 일반화 성능을 더욱 높일 수 있다.

**④ 계산 효율성 개선**

전체 데이터셋 클러스터링의 반복이 계산 비용을 높인다. **온라인 클러스터링** 또는 **메모리 뱅크** 방식을 도입하여 효율성을 개선할 수 있다.

**⑤ 소스-프리(Source-Free) 도메인 적응으로 확장**

최근에는 소스 데이터에 직접 접근하지 않고도 적응하는 Source-Free DA가 주목받고 있다. CAN의 클러스터링 아이디어를 소스-프리 설정에 적용하는 연구가 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 연구들은 CAN 이후 UDA 분야에서 발전된 주요 연구들이다. 단, 구체적 수치 일부는 각 논문을 직접 확인하시길 권장한다.

### 5.1 CDTrans (2021)

**논문**: "CDTrans: Cross-Domain Transformer for Unsupervised Domain Adaptation" (Xu et al., 2021)

- **핵심 아이디어**: 트랜스포머 기반 교차 도메인 주의(attention) 메커니즘으로 소스-타깃 쌍 간 패치 수준 정렬
- **CAN과의 차이**: CAN이 FC 레이어 수준의 분포 정렬을 수행하는 반면, CDTrans는 패치 수준의 세밀한 정렬을 수행
- **일반화 측면**: 트랜스포머의 전역적 수용 영역(global receptive field)이 더 풍부한 구조적 정보를 활용

### 5.2 SHOT (2020)

**논문**: "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation" (Liang et al., ICML 2020)

- **핵심 아이디어**: 소스 데이터 없이 소스 모델의 가설(hypothesis)만을 활용하는 **Source-Free DA**
- **CAN과의 연결**: CAN의 클러스터링 기반 의사 레이블 아이디어를 계승하되, 소스 데이터 접근 불필요
- **목적 함수**: 정보 최대화(information maximization)와 의사 레이블 기반 자기 지도 학습 결합

$$\min_\theta - \frac{1}{n_t}\sum_{i=1}^{n_t}\sum_{k=1}^{K} \delta_k(\mathbf{x}^t_i)\log\delta_k(\mathbf{x}^t_i) + \sum_{k=1}^{K}\hat{p}_k\log\hat{p}_k$$

- **의의**: 개인정보 보호 등 소스 데이터 접근이 어려운 실제 환경에 적합

### 5.3 SDAT (2022)

**논문**: "Smoothed Adaptive Domain Adaptation with Transferability for Unsupervised Domain Adaptation" (관련 연구)

**논문**: "Towards Safer Predictions on the Open World: A Teacher-Student Framework for Uncertainty Estimation in Unsupervised Domain Adaptation" 등

더 정확한 비교를 위해 직접 arXiv나 CVPR/ICCV/ECCV 프로시딩을 확인하시길 권장한다.

### 5.4 DAML / PMTrans (2022)

**논문**: "PMTrans: Patch Mix Transformer for Unsupervised Domain Adaptation" (Zhu et al., ECCV 2022)

- **핵심 아이디어**: 소스-타깃 패치를 혼합(mix)하여 중간 도메인 생성, 트랜스포머로 처리
- **CAN과의 차이**: CAN이 클래스 수준 통계 정렬에 집중하는 반면, PMTrans는 패치 수준 혼합으로 암묵적 정렬

### 5.5 비교 요약표

| 방법 | 연도 | 클래스 인식 | 의사 레이블 | 소스 데이터 필요 | 백본 | Office-31 평균 |
|------|------|------------|------------|----------------|------|----------------|
| CAN | 2019 | ✓ (CDD) | 클러스터링 | ✓ | ResNet-50 | 90.6% |
| SHOT | 2020 | ✓ | 정보 최대화 | ✗ | ResNet-50 | ~90.1% |
| CDTrans | 2021 | ✓ | ✓ | ✓ | ViT-B | ~97.0% |
| PMTrans | 2022 | ✓ | ✓ | ✓ | ViT-B | ~97.5% |

> **주의**: 위 표의 수치 중 CDTrans, PMTrans의 수치는 해당 논문의 보고치를 참조한 것이나, 실험 설정(백본, 전처리 등)이 다르므로 직접 비교에 주의가 필요하다.

---

## 참고 자료

1. **주요 논문 (분석 대상)**
   - Kang, G., Jiang, L., Yang, Y., & Hauptmann, A. G. (2019). "Contrastive Adaptation Network for Unsupervised Domain Adaptation." *CVPR 2019*. arXiv:1901.00976v2

2. **논문 내 인용 주요 참고문헌**
   - Long, M., et al. (2015). "Learning Transferable Features with Deep Adaptation Networks (DAN)." *ICML 2015*. arXiv:1502.02791
   - Long, M., et al. (2017). "Deep Transfer Learning with Joint Adaptation Networks (JAN)." *ICML 2017*
   - Ganin, Y., & Lempitsky, V. (2015). "Unsupervised Domain Adaptation by Backpropagation (RevGrad)." *ICML 2015*
   - Saito, K., et al. (2018). "Maximum Classifier Discrepancy for Unsupervised Domain Adaptation (MCD)." *CVPR 2018*
   - Pei, Z., et al. (2018). "Multi-Adversarial Domain Adaptation (MADA)." *AAAI 2018*
   - French, G., et al. (2018). "Self-ensembling for Domain Adaptation (SE)." *ICLR 2018*

3. **2020년 이후 관련 연구**
   - Liang, J., et al. (2020). "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation (SHOT)." *ICML 2020*. arXiv:2002.08546
   - Xu, T., et al. (2021). "CDTrans: Cross-Domain Transformer for Unsupervised Domain Adaptation." arXiv:2109.06165
   - Zhu, Y., et al. (2022). "PMTrans: Patch Mix Transformer for Unsupervised Domain Adaptation." *ECCV 2022*. arXiv:2203.07465
