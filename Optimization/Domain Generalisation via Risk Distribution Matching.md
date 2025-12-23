# Domain Generalisation via Risk Distribution Matching

### 1. 핵심 요약

**"Domain Generalisation via Risk Distribution Matching"** (Nguyen et al., WACV 2024)는 도메인 일반화(Domain Generalization, DG) 문제를 새로운 관점에서 접근합니다. 기존 방법들이 고차원 특성(representation) 또는 그래디언트 분포의 정렬에 집중했다면, 본 논문은 **스칼라 위험도(scalar risk) 분포의 정렬**을 통해 도메인 불변성을 달성합니다.

**핵심 주장**:
1. 위험도 분포는 도메인 간 차이를 효과적으로 나타낼 수 있다
2. 훈련 도메인 간 위험도 분포의 발산을 최소화하면 강건한 도메인 불변성을 획득할 수 있다
3. 최악의 경우 도메인과 집계된 분포만 정렬하면 계산 효율을 유지하면서도 성능을 향상할 수 있다

***

### 2. 해결하는 문제

#### 2.1 기존 도메인 생성화 방법의 한계

기존 접근법들은 두 가지 주요 문제에 직면합니다:

**표현 기반 정렬 (Representation Matching)**
- CORAL, MMD-based methods는 고차원 특성 공간에서 분포 정렬 추구[1][2]
- **고차원의 저주(curse of dimensionality)**: 고차원 공간에서 통계량 추정이 불안정
- 데이터 희소성으로 인한 분포 차이 추정의 신뢰성 저하

**그래디언트 기반 정렬 (Gradient Matching)**
- Fish, Fishr: 그래디언트 분포 간의 분산 최소화[3][4]
- 고차원 그래디언트 공간에서의 동일한 계산 복잡도 문제

**인과기반 방법의 제한**
- IRM: 모든 환경에서 예측자 불변성 강제 (실제로는 달성 어려움)[5]
- VREx, EQRM: 평균 위험도(mean risk)만 고려하여 전체 분포 특성 무시[6][7]

#### 2.2 논문의 동기

PACS 데이터셋 검증(Figure 1):
- Art 도메인: 낮은 위험도 값 집중
- Photo 도메인: 더 넓고 높은 위험도 분포

이는 **위험도 분포가 도메인 복잡도의 신뢰할 수 있는 지표**임을 시사합니다.

***

### 3. 제안 방법론: Risk Distribution Matching

#### 3.1 기본 개념

도메인 $e$에 대한 위험도 분포 $\mathcal{T}_e$를 정의:

$$\mathcal{T}_e = \text{위험도 } \{R_i^e = \ell(f(x_i^e), y_i^e) : (x_i^e, y_i^e) \in \mathcal{D}_e\} \text{의 분포}$$

도메인 전체의 집계된 위험도 분포:

$$\bar{\mathcal{T}} = \frac{1}{m}\sum_{e=1}^m \mathcal{T}_e$$

#### 3.2 Kernel Mean Embedding

위험도 분포를 재현 커널 힐버트 공간(RKHS)에 임베딩:

$$\mu_{\mathcal{T}_e} = \mathbb{E}_{R_e \sim \mathcal{T}_e}[\phi(R_e)] \in \mathcal{H}$$

여기서 $\phi$는 kernel function $k$에 의한 암묵적 특성 맵입니다.

**정리 1**: Characteristic kernel $k$를 사용하면:

$$V_H(\mathcal{T}_1, ..., \mathcal{T}_m) = 0 \iff \mathcal{T}_1 = \mathcal{T}_2 = ... = \mathcal{T}_m = \bar{\mathcal{T}}$$

본 논문은 RBF kernel을 사용합니다:

$$k(x, x') = \exp\left(-\frac{1}{2\sigma^2}\|x - x'\|^2\right)$$

#### 3.3 분포 분산 메트릭

분포 분산을 RKHS에서의 노름으로 정의:

$$V_H(\mathcal{T}_1, ..., \mathcal{T}_m) = \frac{1}{m}\sum_{e=1}^m \|\mu_{\mathcal{T}_e} - \bar{\mu}_{\mathcal{T}}\|^2_H$$

#### 3.4 최대 평균 불일치(MMD) 계산

분포 간 거리를 MMD로 측정:

$$\text{MMD}^2(\mathcal{T}_e, \bar{\mathcal{T}}) = \mathbb{E}_{R_e, R_e' \sim \mathcal{T}_e}[k(R_e, R_e')] - 2\mathbb{E}_{R_e \sim \mathcal{T}_e, R_f \sim \bar{\mathcal{T}}}[k(R_e, R_f)] + \mathbb{E}_{R_f, R_f' \sim \bar{\mathcal{T}}}[k(R_f, R_f')]$$

**손실 함수**:

$$L_{\text{RDM}} = L_{\text{ERM}} + \lambda \sum_{e=1}^m \text{MMD}^2(\mathcal{T}_e, \bar{\mathcal{T}})$$

이는 모든 moment를 정렬함을 의미합니다 (RBF kernel의 특성).

#### 3.5 효율성 개선: 최악의 경우 도메인 근사

**문제**: 도메인 수 $m$이 증가하면 MMD 계산이 $O(m)$으로 증가

**해결책**: 최악의 경우 도메인만 고려:

$$\hat{L}_{\text{RDM}} = \text{MMD}^2(\mathcal{T}_w, \bar{\mathcal{T}}), \quad w = \arg\max_{e \in E} R_e$$

**수학적 근거** (상한 성질):

$$L_{\text{RDM}} = \frac{1}{m}\sum_{e=1}^m \text{MMD}^2(\mathcal{T}_e, \bar{\mathcal{T}}) \leq \frac{1}{m}\sum_{e=1}^m \text{MMD}^2(\mathcal{T}_w, \bar{\mathcal{T}}) = \text{MMD}^2(\mathcal{T}_w, \bar{\mathcal{T}}) = \hat{L}_{\text{RDM}}$$

**이중 이점**:
1. **계산 효율**: 모든 도메인 쌍의 MMD 계산이 아닌 단일 쌍의 계산만 필요 ( $O(m) \to O(1)$ )
2. **강건성 향상**: 최악의 경우 도메인에 집중하여 극단적 시나리오에 대한 견고성 증진

***

### 4. 모델 구조 및 훈련

#### 4.1 아키텍처

```
Input Image (224×224×3)
        ↓
ResNet-50 Feature Extractor (ImageNet 사전학습)
        ↓
Feature Vector (2048-d)
        ↓
Classifier (fc layer) → Softmax output (K classes)
        ↓
Batch의 각 샘플의 Loss 계산
        ↓
Risk Distribution (배치 내 loss 값들의 분포)
        ↓
MMD 거리 계산 및 분포 정렬
```

#### 4.2 훈련 절차

1. **사전학습 단계 (ERM)**:
   - ERM 손실로 1500-2400 iterations 사전학습
   - 기본 분류 능력 확보

2. **RDM 최적화 단계**:
   - 결합된 손실: $L_{\text{final}} = L_{\text{ERM}} + \lambda \cdot \hat{L}_{\text{RDM}}$
   - Adam optimizer (learning rate: 1.5×10⁻⁵)
   - 5,000 iterations 훈련

#### 4.3 핵심 하이퍼파라미터

| 파라미터 | 범위 | 최적값 | 설명 |
|---------|------|-------|-----|
| Matching coefficient ($\lambda$) | [0.1, 10.0] | 5.0 | ERM과 RDM 손실의 균형 |
| Batch size | 30-100 | 70-88 | Risk distribution 추정 정확도 |
| RBF kernel bandwidth ($\sigma$) | Multiple | Auto | 0.0001-1000 범위의 모든 값 사용 (커널 트릭) |
| Pre-training iterations | 800-2700 | 1500 | ERM 사전학습 기간 |

***

### 5. 성능 향상 분석

#### 5.1 ColoredMNIST 결과 (합성 데이터셋)

ColoredMNIST는 색상이 숫자 형태와 독립적이지만 훈련 데이터에서 상관되는 도메인 강화 설정입니다.

| 알고리즘 | 무작위 초기화 | ERM 사전학습 | Oracle |
|---------|-----------|----------|--------|
| ERM | 27.9±1.5 | 27.9±1.5 | - |
| GroupDRO | 27.3±0.9 | 29.0±1.1 | - |
| IRM | 52.5±2.4 | 69.7±0.9 | - |
| VREx | 55.2±4.0 | 71.6±0.5 | - |
| EQRM | 53.4±1.7 | 71.4±0.4 | - |
| CORAL | 55.3±2.8 | 65.6±1.1 | - |
| MMD | 54.6±3.2 | 66.4±1.7 | - |
| **RDM** | **56.3±1.5** | **72.4±1.0** | **72.1±0.7** |

**해석**:
- RDM은 Oracle (완벽한 불변성, grayscale 학습)에 가장 가깝게 수렴
- 전체 위험도 분포 고려 → VREx (평균만) 대비 +0.8% 향상
- Spurious feature(색상) 완전 제거 달성

#### 5.2 DomainBed 벤치마크 (5개 대규모 데이터셋)

**VLCS** (Caltech-101, LabelMe, SUN09, VOC2007):

| 알고리즘 | 정확도 | 표준편차 |
|---------|-------|---------|
| ERM | 77.5 | ±0.4 |
| Mixup | 77.4 | ±0.6 |
| MLDG | 77.2 | ±0.4 |
| GroupDRO | 76.7 | ±0.6 |
| IRM | 78.5 | ±0.5 |
| VREx | 78.3 | ±0.2 |
| EQRM | 77.8 | ±0.6 |
| Fish | 77.8 | ±0.3 |
| Fishr | 77.8 | ±0.1 |
| CORAL | 78.8 | ±0.6 |
| **RDM** | **78.4** | **±0.4** |

**PACS** (Photo, Art-painting, Cartoon, Sketch):

| 알고리즘 | 정확도 | 표준편차 |
|---------|-------|---------|
| ERM | 85.5 | ±0.2 |
| Mixup | 84.6 | ±0.6 |
| CORAL | 86.2 | ±0.3 |
| MMD | 84.6 | ±0.5 |
| **RDM** | **87.2** | **±0.7** |

- **평균 개선**: 모든 벤치마크에서 +0.5~1.9% 향상
- **DomainNet에서 가장 큰 성과**: +2.5% (43.4% vs 40.9% ERM)
- **강건성**: 모든 벤치마크에서 상위 성능 유지

#### 5.3 계산 효율성 비교 (DomainNet)

| 방법 | 학습 시간(초) | 메모리 사용(GiB) | 정확도(%) | 시간 절감 |
|------|-----------|------------|---------|----------|
| Fish | 11,502 | 5.26 | 42.7 | baseline |
| CORAL | 11,504 | 17.00 | 41.5 | 비슷 |
| RDM ($L_{\text{RDM}}$) | 9,854 | 16.94 | 43.1 | 14% 절감 |
| **RDM ($\hat{L}_{\text{RDM}}$)** | **7,749** | **16.23** | **43.4** | **33% 절감** |

**성과**:
- 근사 방법( $\hat{L}\_{\text{RDM}}$ )이 전체 분산( $L_{\text{RDM}}$ )보다 우수한 성능
- Fish 대비 정확도 +0.7%이면서 33% 더 빠른 학습
- CORAL 대비 메모리 효율적

***

### 6. 성능 향상의 원리

#### 6.1 위험도 분포 정렬의 효과 (시각화)

ColoredMNIST에서의 위험도 분포 히스토그램:

**ERM의 문제점**:
- 훈련 도메인(90% red, 80% red): 매우 낮은 위험도 집중
- 테스트 도메인(10% red): 훨씬 높은 위험도 분포
- **원인**: 색상이라는 Spurious feature에 과적합

**RDM의 해결책**:
- 모든 도메인에서 일관된, 넓은 위험도 분포
- 배치 내 다양한 위험도 값들의 분포 정렬
- **결과**: 도메인 불변 특성 학습, 일반화 향상

#### 6.2 최악의 경우 도메인 근사의 정당성

**상한 관계**:

$$L_{\text{RDM}} \leq \hat{L}_{\text{RDM}} \leq \frac{1}{m} \text{(최대 bound)}$$

그림 3a 분석:
- $\hat{L}\_{\text{RDM}}$은 $L_{\text{RDM}}$보다 항상 크거나 같음 (상한 성질 확인)
- 실제로는 두 값의 차이가 작음 (5-10% 정도)
- 최악의 경우 도메인이 다른 도메인들의 분포 차이를 대표할 수 있음을 시사

**Robustness 향상**:
- 최악의 경우 도메인에 집중 → 극단적 시나리오 처리 능력 강화
- 이는 분포외(OOD) 데이터에 더 견고해지는 효과

***

### 7. 논문의 주요 한계

#### 7.1 OfficeHome 성능 저하

| 알고리즘 | A | C | P | R | 평균 |
|---------|---|---|---|---|-----|
| ERM | 61.3 | 52.4 | 75.8 | 76.6 | 66.5 |
| CORAL | 65.3 | 54.4 | 76.5 | 78.4 | 68.7 |
| **RDM** | **61.1** | **55.1** | **75.7** | **77.3** | **67.3** |

**저하 원인**:
- 클래스당 평균 240개 샘플 (다른 벤치마크는 1,400+ 샘플)
- **제한된 샘플로 인한 위험도 분포의 다양성 부족**
- Training/validation 매칭 손실 간 큰 격차 (Figure 5)
  - 훈련 손실: 빠르게 최소값으로 수렴
  - 검증 손실: 일관되게 높음 (불안정)
- **원인**: 제한된 데이터로 안정적인 위험도 분포를 학습하지 못함

**해결책** (미래 연구):
- 적응적 배치 크기 조정
- 위험도 분포 평활화(smoothing)
- 메타-학습과 결합하여 적응성 강화

#### 7.2 배치 크기 의존성

**필요한 배치 크기**:
- PACS, VLCS, OfficeHome: 70-100
- TerraIncognita, DomainNet: 30-60
- 큰 배치 크기가 정확한 위험도 분포 추정에 필수적

**문제점**:
- GPU 메모리 제약이 있는 환경에서 제한적
- 모바일/엣지 디바이스 배포 어려움

#### 7.3 하이퍼파라미터 민감도

- Matching coefficient $\lambda$: 데이터셋마다 [0.1, 10.0] 범위 내 조정 필요
- RBF kernel bandwidth: 자동 선택하지만, 특정 데이터셋에서 수동 튜닝 가능

***

### 8. 2020년 이후 관련 최신 연구 비교 분석

#### 8.1 도메인 정렬 기반 방법의 진화

**CORAL 계열** (2016 → 2025)

| 방법 | 주요 아이디어 | 한계 | RDM과의 관계 |
|------|-----------|------|-----------|
| CORAL (2016) | 2차 통계량 정렬 (mean, covariance) | 고차 moment 무시 | 모든 moment 고려 |
| Deep CORAL (2016) | 다층 특성에 적용 | 고차원 문제 | Scalar로 차원 축소 |
| MMD-AAE (2018) | MMD로 모든 moment 정렬 | 고차원 representation | Risk distribution으로 간단화 |

**성능 비교** (PACS):
- CORAL: 86.2% → **RDM: 87.2%** (+1.0%)

#### 8.2 인과기반 방법 (2019-2023)

**IRM (Arjovsky et al., 2019)**[8]
- 목표: 모든 환경에서 최적 분류기 불변
- 방식: Bi-level 최적화로 불변 예측자 강제
- 한계: 비선형 체제에서 spurious feature 의존 가능

**VREx (Krueger et al., 2020)**[9]
- 목표: Risk 분산 최소화
- 공식: $R_{\text{VREx}} = \beta \cdot \text{Var}(\{R_1, ..., R_m\}) + \sum R_e$
- 한계: **평균값(1차 moment)만 고려**

**RDM과의 비교**:

$$\text{VREx: 평균값만} \quad \Rightarrow \quad \text{RDM: 전체 분포} \quad \Rightarrow \quad \text{더 강력한 불변성}$$

ColoredMNIST에서 **RDM +0.8% 향상** (72.4% vs 71.6%)

**EQRM (Eastwood et al., 2022)**[6]
- 목표: Quantile risk 최소화
- 방식: 각 도메인의 위험도 분위수(quantile) 정렬
- RDM: 전체 분포 모양 고려 (더 세밀한 정렬)

#### 8.3 최신 방향 (2023-2025)

**Vision Transformer 기반 DG**
- CLIP, DINO 활용으로 semantic feature 강화
- RDM은 backbone 독립적으로 적용 가능
- 예상: ViT + RDM 조합이 강력할 것으로 예상

**Single Domain Generalization (SDG)**[10]
- 단일 도메인으로 다중 도메인 가정
- 데이터 증강 + 불변성
- RDM의 위험도 분포 개념을 SDG에 적용 가능

**Federated Domain Generalization (FedDG)**[3]
- 분산 환경에서 도메인 불변성
- RDM의 $O(1)$ 복잡도가 통신 효율성 측면에서 장점
- "최악 도메인" 선택이 지역 노드에서 계산 가능

**Temporal Domain Generalization (TDG)**[11]
- 시간 흐름에 따라 진화하는 도메인 분포
- Weight averaging으로 temporal experts 결합
- RDM의 확률 분포 정렬 원리를 시계열에 확장 가능

**분포 정렬의 새로운 메트릭** (2025)

| 메트릭 | 특성 | 계산 복잡도 | RDM과의 비교 |
|--------|------|-----------|-----------|
| Wasserstein | Optimal transport 기반, 해석 용이 | 높음 | MMD보다 느리지만 해석성 있음 |
| Stein Discrepancy | Score function만 필요, 저데이터 강함 | 중간 | Complementary: 배치 작을 때 Stein 고려 |
| KL Divergence | 정보이론 기반 | 높음 (밀도 추정 필요) | RDM은 밀도 추정 불필요 |

#### 8.4 구체적 최신 논문 비교 (2024-2025)

**CbDA (2025): Contrastive-Based Data Augmentation**[12]
```
Contrastive loss (sample level)
            +
Category별 분포 통계 (domain level)
            ↓
RDM과 상호보완적 (동일 배치 내에서 적용 가능)
```

**FOUND (2025): Fourier-based von Mises Distribution for SDG**[10]
- Von Mises-Fisher distribution으로 directional features 모델링
- CLIP 기반 semantic 보존
- RDM과 결합: Risk distribution + semantic CLIP features

**TTA-FedDG (2025): Test-Time Adaptation for Federated DG**[13]
- 테스트 시점의 적응
- RDM은 학습 시점 최적화에 집중하므로 complementary

**Information-Theoretic Analysis (2025)**[14]
- Information theory로 분포 정렬의 일반화 경계 분석
- **RDM의 이론적 근거 강화 가능**:
  - Why does distribution matching help? → Mutual information reduction
  - Risk distribution이 특히 효과적인 이유 → Information bottleneck 관점

#### 8.5 요약: 방법 진화 타임라인

```
2016: CORAL (2차 통계량)
  ↓
2018: MMD-based (모든 moment, but 고차원)
  ↓
2019: IRM (불변 예측자, but 이론적 갭)
  ↓
2020: VREx (risk 분산, but 평균값만)
  ↓
2022: EQRM (quantile risk)
  ↓
2023: RDM ← 모든 moment (scalar risk로 간단화)
  ↓
2024-2025: ViT+RDM, Federated RDM, Semantic+RDM
```

***

### 9. 앞으로의 연구에서 고려할 점

#### 9.1 이론적 강화

**1. 일반화 경계(Generalization Bounds)**

RDM이 OOD 오차를 얼마나 줄이는지 정량화:

$$\mathbb{E}_{\text{test}}[\text{error}] \leq \mathbb{E}_{\text{train}}[\text{error}] + O(d_H(\mathcal{T}_1, ..., \mathcal{T}_m) + \ldots)$$

여기서 $d_H$는 RKHS에서의 거리.

**2. 최악의 경우 도메인 근사의 정당성**

- 현재: 실증적 성능만 제시
- 필요: Theoretical guarantee가 $w = \arg\max R_e$일 때 성립하는 조건 명시

**3. Convergence 분석**

- 이중-레벨 최적화의 수렴 속도
- Kernel bandwidth 자동 선택의 이론적 근거

#### 9.2 방법론적 확장

**1. 적응적 최악 도메인 선택**
```python
# 현재
w = argmax(mean_risk)

# 개선안
w = select_worst_k_domains(k=adaptive)
# 초기: k=1, 훈련 진행 → k 증가
```

**2. 다층 위험도 분포**
- Sample level: $\mathcal{T}_e^{\text{sample}}$ (현재)
- Class level: $\mathcal{T}_{e,c}$ (클래스별)
- Domain level: 계층적 분포 정렬

**3. 자동 배치 크기 조정**
```python
# 위험도 분포 표본 분산 감시
if sample_variance(risks) > threshold:
    increase_batch_size()
```

#### 9.3 실제 응용 확대

**1. Unsupervised/Semi-supervised DG**
- 레이블 없는 시나리오에서 위험도 정의 (예: reconstruction loss)

**2. Single Source DG (SDG)**
- 단일 도메인 → 합성 도메인 생성
- 합성 도메인의 위험도 분포 활용

**3. 의료 영상 (Medical Imaging)**
- 다중 센터(hospital) 간 일반화
- 스캐너 종류(CT, MRI, Ultrasound) 간 이동성

**4. 자율 주행 (Autonomous Driving)**
- 날씨 조건(晴雨雪) 간 이동성
- 시간대(주간/야간) 간 이동성
- RDM의 효율성이 실시간 처리에 중요

**5. 객체 검출/세분화**
- 분류 외에 bounding box 위험도, segmentation 위험도 정렬

#### 9.4 관련 기술과의 시너지

**IRM + RDM**:
```
인과 불변성 (IRM) + 위험도 분포 불변성 (RDM)
→ 더 강력한 domain invariance
```

**CORAL + RDM**:
```
특성 분포 정렬 (CORAL) + 위험도 분포 정렬 (RDM)
→ 다층 수준의 정렬
```

**Data Augmentation + RDM**:
```
증강 데이터의 위험도 분포 분석
→ 어떤 증강이 위험도를 균형있게 하는지 파악
```

**Self-Supervised + RDM**:
```
Contrastive loss (representation level)
        +
Risk distribution matching (prediction level)
```

#### 9.5 배치 크기 의존성 해결

**핵심 문제**: 작은 배치에서 위험도 분포 추정 불안정

**해결 방안**:
1. **위험도 분포 평활화**: Kernel density estimation으로 부드러운 분포 추정
2. **메타-학습 적용**: Few-shot 배치에서 학습할 수 있는 적응형 RDM
3. **경험적 누적**: 배치를 넘어 이동 평균(running average) 사용

```python
# Running estimate of risk distribution
risk_buffer = collections.deque(maxlen=10000)
while training:
    batch_risks = model(batch).loss.detach()
    risk_buffer.extend(batch_risks)
    # Use risk_buffer for distribution matching
```

#### 9.6 Interpretability 향상

**현재 한계**: 어떤 도메인이 정렬을 주도하는지 명확하지 않음

**개선 방향**:
```python
# 도메인별 MMD 기여도 분석
domain_contributions = {}
for e in range(m):
    domain_contributions[e] = MMD(T_e, T_bar).item()
    
# Attention mechanism으로 도메인 가중치
# 시간에 따른 attention 변화 시각화
```

***

### 10. 결론: 종합 평가

#### 10.1 RDM의 강점

| 측면 | 강점 | 영향 |
|------|------|------|
| **효율성** | O(m) → O(1) 복잡도 감소 | 대규모 도메인 수 처리 가능 |
| **성능** | 최신 SOTA 방법 대비 우수 | +0.5~2.5% 정확도 향상 |
| **해석성** | Scalar risk의 직관성 | 실무진의 이해 용이 |
| **확장성** | 다양한 도메인 생성화 시나리오 | 의료, 자율주행 등 적용 |
| **단순성** | 구현 간단, 하이퍼파라미터 적음 | 재현성 높음 |

#### 10.2 RDM의 한계 및 극복 방안

| 한계 | 원인 | 극복 방안 |
|------|------|---------|
| 배치 크기 의존성 | 정확한 분포 추정 필요 | 경험적 누적, kernel smoothing |
| OfficeHome 저하 | 제한된 샘플 수 | 메타-학습, 데이터 증강 시너지 |
| 이론적 갭 | 경험적 성공만 제시 | 일반화 경계 유도 |
| 테스트 시점 적응 불가 | 학습 시점 최적화만 | TTA와 결합 |

#### 10.3 향후 연구의 우선순위

**단기** (1-2년):
1. 배치 크기 의존성 완화 연구
2. OfficeHome 같은 저샘플 환경에서의 성능 개선
3. 의료 영상 등 실제 응용 검증

**중기** (2-4년):
1. Vision Transformer와의 시너지 탐구
2. Federated DG에서의 효율성 활용
3. 일반화 경계 이론 개발

**장기** (4년 이상):
1. Foundation models (CLIP, DINO)과의 통합
2. 다모달(multimodal) 도메인 생성화
3. 연속적(continual) 도메인 이동 처리

#### 10.4 학문적 기여도

RDM은 도메인 생성화 연구에 다음과 같은 기여를 제시합니다:

1. **패러다임 전환**: 고차원 정렬 → 스칼라 위험도 정렬로 단순화
2. **효율성-성능 트레이드오프 해결**: 계산 복잡도 감소와 동시에 성능 향상
3. **직관적 접근**: 위험도 분포의 물리적 해석이 명확
4. **확장 가능성**: 다양한 DG 시나리오에 적용 가능한 기본 원리 제공

***

### 참고 문헌 (주요 인용 논문)

***
[1](https://arxiv.org/pdf/2309.01343.pdf)
[2](https://arxiv.org/pdf/1607.01719.pdf)
[3](https://arxiv.org/abs/2410.11267)
[4](https://ieeexplore.ieee.org/document/10651307/)
[5](https://link.springer.com/10.1007/s00044-025-03407-3)
[6](https://arxiv.org/pdf/2102.08604.pdf)
[7](https://pmc.ncbi.nlm.nih.gov/articles/PMC12286383/)
[8](https://arxiv.org/pdf/1907.02893.pdf)
[9](https://arxiv.org/pdf/2003.00688.pdf)
[10](https://www.semanticscholar.org/paper/8f67505e183c423dfcb4e25b8beca825c8272013)
[11](https://arxiv.org/abs/2509.26045)
[12](https://ieeexplore.ieee.org/document/10601637/)
[13](https://ojs.aaai.org/index.php/AAAI/article/view/34053)
[14](https://ieeexplore.ieee.org/document/10844895/)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a1ec3c9e-c598-4396-b8f1-0ef569c73e06/2310.18598v1.pdf)
[16](https://ieeexplore.ieee.org/document/11148050/)
[17](https://ieeexplore.ieee.org/document/10965929/)
[18](https://revistaft.com.br/static-liquefaction-in-tailings-and-fine-ores-2020-2025-mechanisms-assessment-methods-modeling-and-management-guidelines/)
[19](https://ieeexplore.ieee.org/document/11076159/)
[20](https://arxiv.org/abs/2509.15791)
[21](http://arxiv.org/pdf/2110.04545.pdf)
[22](https://arxiv.org/pdf/2401.08464.pdf)
[23](http://arxiv.org/pdf/2302.06874.pdf)
[24](https://arxiv.org/pdf/2302.02350.pdf)
[25](https://arxiv.org/pdf/2211.04393.pdf)
[26](http://arxiv.org/pdf/2411.02920.pdf)
[27](http://arxiv.org/pdf/2208.00898.pdf)
[28](https://www.sciencedirect.com/science/article/abs/pii/S0031320325003383)
[29](https://proceedings.neurips.cc/paper_files/paper/2022/file/0b5eb45a22ff33956c043dd271f244ea-Paper-Conference.pdf)
[30](https://pmc.ncbi.nlm.nih.gov/articles/PMC12025361/)
[31](https://www.scitepress.org/Papers/2025/133003/133003.pdf)
[32](https://arxiv.org/abs/2510.03540)
[33](https://openaccess.thecvf.com/content/WACV2024/papers/Nguyen_Domain_Generalisation_via_Risk_Distribution_Matching_WACV_2024_paper.pdf)
[34](https://arxiv.org/abs/2208.08661v1)
[35](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0320300)
[36](https://arxiv.org/html/2507.16406v1)
[37](https://arxiv.org/html/2502.03587v4)
[38](https://arxiv.org/html/2509.12845v1)
[39](https://arxiv.org/html/2502.05593v2)
[40](https://arxiv.org/html/2511.06056v1)
[41](https://arxiv.org/abs/2206.11646)
[42](https://arxiv.org/html/2510.04441v1)
[43](https://arxiv.org/html/2505.18906v2)
[44](https://www.sciencedirect.com/science/article/abs/pii/S0031320323007264)
[45](https://arxiv.org/abs/2007.02931v4)
[46](https://ieeexplore.ieee.org/document/10678236/)
[47](https://ieeexplore.ieee.org/document/10093034/)
[48](https://ieeexplore.ieee.org/document/10571357/)
[49](https://ieeexplore.ieee.org/document/10655401/)
[50](https://dl.acm.org/doi/10.1145/3659953)
[51](https://ieeexplore.ieee.org/document/10695100/)
[52](https://ieeexplore.ieee.org/document/10750436/)
[53](https://aclanthology.org/2023.acl-long.696)
[54](https://arxiv.org/pdf/1612.01939.pdf)
[55](https://arxiv.org/pdf/2411.12913.pdf)
[56](https://arxiv.org/pdf/2110.09410.pdf)
[57](https://arxiv.org/pdf/2306.07266.pdf)
[58](http://arxiv.org/pdf/2210.02655.pdf)
[59](https://arxiv.org/pdf/2303.18031.pdf)
[60](https://www.emergentmind.com/topics/invariant-risk-minimization-irm)
[61](https://openaccess.thecvf.com/content/ICCV2025W/PHAROS-AFE-AIMI/papers/Yuan_Multi-Source_Covid-19_Detection_via_Variance_Risk_Extrapolation_ICCVW_2025_paper.pdf)
[62](https://openaccess.thecvf.com/content/CVPR2022/papers/Galstyan_Failure_Modes_of_Domain_Generalization_Algorithms_CVPR_2022_paper.pdf)
[63](https://hrilab.tufts.edu/publications/thuan2023mlsp.pdf)
[64](https://liner.com/review/outofdistribution-generalization-via-risk-extrapolation-rex)
[65](https://dmqa.korea.ac.kr/activity/seminar/415)
[66](https://proceedings.neurips.cc/paper_files/paper/2022/file/91b482312a0845ed86e244adbd9935e4-Paper-Conference.pdf)
[67](https://arxiv.org/abs/2003.00688)
[68](https://arxiv.org/abs/1607.01719)
[69](https://arxiv.org/pdf/2007.01434.pdf)
[70](https://arxiv.org/pdf/2103.03097.pdf)
[71](https://arxiv.org/abs/2407.05765)
[72](https://arxiv.org/abs/2506.23208)
[73](https://bayesgroup.github.io/bmml_sem/2019/Kodryan_Invariant%20Risk%20Minimization.pdf)
[74](https://www.ijcai.org/proceedings/2021/0628.pdf)
[75](https://openaccess.thecvf.com/content/CVPR2022/papers/Lin_Bayesian_Invariant_Risk_Minimization_CVPR_2022_paper.pdf)
[76](https://arxiv.org/abs/2006.07544)
