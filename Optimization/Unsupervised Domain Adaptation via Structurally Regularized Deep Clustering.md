# Unsupervised Domain Adaptation via Structurally Regularized Deep Clustering

## 📌 참고 자료

- **주요 논문**: Tang, H., Chen, K., & Jia, K. (2020). *Unsupervised Domain Adaptation via Structurally Regularized Deep Clustering*. arXiv:2003.08607v1 [cs.CV], CVPR 2020.
- GitHub 코드: https://github.com/huitangtang/SRDC-CVPR2020

> ⚠️ **주의**: 2020년 이후 최신 연구 비교 분석 섹션에서 언급되는 일부 논문(NRC, SHOT, CDTrans 등)은 본 제공 PDF에 포함되지 않은 외부 문헌입니다. 해당 부분은 공개된 arXiv 논문 정보를 기반으로 작성하되, 불확실한 수치는 명시합니다.

---

## 1. 핵심 주장과 주요 기여 요약

### 🎯 핵심 주장

기존 UDA(비지도 도메인 적응) 방법들이 **도메인 정렬(domain alignment)** 전략을 사용하는 반면, 이 전략은 타겟 도메인의 **내재적 판별 구조(intrinsic discrimination)를 손상시킬 위험**이 있다. SRDC는 명시적 도메인 정렬 없이, **구조적 도메인 유사성 가정** 하에 타겟 데이터를 직접 클러스터링하여 판별 구조를 uncovering하는 방법을 제안한다.

### 🏆 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| 새로운 패러다임 제안 | 도메인 정렬 대신 **판별적 클러스터링으로 내재적 타겟 판별 구조 발굴** |
| 기술적 프레임워크 | KL 발산 최소화 기반 딥 클러스터링 + 소스 구조적 정규화 |
| 추가 기법 | 중간 특징 공간 클러스터링 + 소스 샘플 소프트 선택 |
| 성과 | 3개 UDA 벤치마크에서 **명시적 도메인 정렬 없이** 기존 모든 방법 초과 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**기존 방법의 한계:**

```
기존 UDA (Transferring Strategy):
Source 데이터 → 도메인 정렬 학습 → Target에 적용
                     ↓
         타겟 도메인 내재적 판별 구조 손상 위험
```

논문은 두 가지 핵심 가정을 제시한다:

- **Domain-wise discrimination**: 각 도메인 내에 판별적 클러스터 구조가 존재
- **Class-wise closeness**: 동일 클래스에 해당하는 두 도메인의 클러스터가 기하학적으로 근접

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 딥 판별적 타겟 클러스터링 (Deep Discriminative Target Clustering)

**보조 분포 업데이트 (Auxiliary Distribution Update):**

$$q^t_{i,k} = \frac{p^t_{i,k} / \left(\sum_{i'=1}^{n_t} p^t_{i',k}\right)^{\frac{1}{2}}}{\sum_{k'=1}^{K} p^t_{i,k'} / \left(\sum_{i'=1}^{n_t} p^t_{i',k'}\right)^{\frac{1}{2}}} \tag{2}$$

**네트워크 업데이트 (Network Update):**

$$\min_{\theta, \vartheta} -\frac{1}{n_t} \sum_{i=1}^{n_t} \sum_{k=1}^{K} q^t_{i,k} \log p^t_{i,k} \tag{3}$$

**소프트 클러스터 할당 확률 (특징 공간 $\mathcal{Z}$에서):**

$$\tilde{p}^t_{i,k} = \frac{\exp\left((1 + \|\boldsymbol{z}^t_i - \boldsymbol{\mu}_k\|^2)^{-1}\right)}{\sum_{k'=1}^{K} \exp\left((1 + \|\boldsymbol{z}^t_i - \boldsymbol{\mu}_{k'}\|^2)^{-1}\right)} \tag{4}$$

**전체 딥 판별적 타겟 클러스터링 목적 함수:**

$$\min_{Q^t, \tilde{Q}^t, \{\theta, \vartheta\}, \{\mu_k\}_{k=1}^K} \mathcal{L}^t_{\text{SRDC}} = \mathcal{L}^t_{f \circ \varphi} + \mathcal{L}^t_{\varphi} \tag{6}$$

여기서:

$$\min_{Q^t, \{\theta, \vartheta\}} \mathcal{L}^t_{f \circ \varphi} = \text{KL}(Q^t \| P^t) + \sum_{k=1}^{K} \varrho^t_k \log \varrho^t_k \tag{1}$$

$$\min_{\tilde{Q}^t, \theta, \{\mu_k\}_{k=1}^K} \mathcal{L}^t_{\varphi} = \text{KL}(\tilde{Q}^t \| \tilde{P}^t) + \sum_{k=1}^{K} \tilde{\varrho}^t_k \log \tilde{\varrho}^t_k \tag{5}$$

- $\varrho^t_k = \frac{1}{n_t}\sum_{i=1}^{n_t} q^t_{i,k}$: 클러스터 크기 균형 항 (균등 분포 장려)

#### Step 2: 구조적 소스 정규화 (Structural Source Regularization)

$$\min_{\theta, \vartheta} \mathcal{L}^s_{f \circ \varphi} = -\frac{1}{n_s} \sum_{j=1}^{n_s} \sum_{k=1}^{K} \mathbb{I}[k = y^s_j] \log p^s_{j,k} \tag{7}$$

$$\min_{\theta, \{\mu_k\}_{k=1}^K} \mathcal{L}^s_{\varphi} = -\frac{1}{n_s} \sum_{j=1}^{n_s} \sum_{k=1}^{K} \mathbb{I}[k = y^s_j] \log \tilde{p}^s_{j,k} \tag{8}$$

$$\min_{\{\theta, \vartheta\}, \{\mu_k\}_{k=1}^K} \mathcal{L}^s_{\text{SRDC}} = \mathcal{L}^s_{f \circ \varphi} + \mathcal{L}^s_{\varphi} \tag{10}$$

#### Step 3: 소스 샘플 소프트 선택 (Soft Source Sample Selection)

**코사인 유사도 기반 가중치:**

$$w^s(\boldsymbol{x}^s) = \frac{1}{2}\left(1 + \frac{{\boldsymbol{c}^t_{y^s}}^\top \boldsymbol{x}^s}{\|\boldsymbol{c}^t_{y^s}\| \|\boldsymbol{x}^s\|}\right) \in [0, 1] \tag{12}$$

**가중치 적용 소스 학습:**

$$\mathcal{L}^s_{f \circ \varphi}(\cdot; \{w^s_j\}_{j=1}^{n_s}) = -\frac{1}{n_s} \sum_{j=1}^{n_s} w^s_j \sum_{k=1}^{K} \mathbb{I}[k = y^s_j] \log p^s_{j,k} \tag{13}$$

#### 최종 목적 함수 (Final Objective):

$$\min_{Q^t, \tilde{Q}^t, \{\theta, \vartheta\}, \{\mu_k\}_{k=1}^K} \mathcal{L}_{\text{SRDC}} = \mathcal{L}^t_{\text{SRDC}} + \lambda \mathcal{L}^s_{\text{SRDC}} \tag{11}$$

- $\lambda$: 페널티 파라미터 (훈련 과정에서 $\lambda_p = 2(1 + \exp(-\gamma p))^{-1} - 1$으로 점진적 증가, $\gamma = 10$)

### 2.3 모델 구조

```
입력 이미지
    ↓
[ResNet-50 백본] (ImageNet 사전학습)
    ↓
[Bottleneck FC Layer] (2048 → 512)  ← 특징 공간 Z에서 클러스터링 수행
    ↓
[Task-specific FC Layer] (512 → K)
    ↓
[Softmax] → 예측 확률 벡터 p ∈ [0,1]^K
```

**구체적 구현 사항:**
- 기반 네트워크: ImageNet 사전학습 ResNet-50
- FC 레이어: $2048 \rightarrow 512 \rightarrow K$ (2단계)
- 학습률 스케줄: $\eta_p = \eta_0(1 + \alpha p)^{-\beta}$ ($\eta_0=0.001, \alpha=10, \beta=0.75$)
- 배치 크기: 64, 훈련 에폭: 200
- 클러스터 센터 $\{\mu_k\}$: 매 에폭 시작 시 K-means로 재초기화

### 2.4 성능 향상

#### Office-31 결과 (ResNet-50)

| Method | A→W | A→D | D→A | W→A | **Avg** |
|--------|-----|-----|-----|-----|---------|
| Source Model | 77.8 | 82.1 | 64.5 | 66.1 | 81.1 |
| DANN [16] | 81.7 | 83.9 | 66.4 | 66.0 | 82.6 |
| CDAN+E [35] | 94.1 | 92.9 | 71.0 | 69.3 | 87.7 |
| CAN [27] | 94.5 | 95.0 | 78.0 | 77.0 | 90.6 |
| **SRDC** | **95.7** | **95.8** | **76.7** | **77.1** | **90.8** |

#### Ablation Study (Office-31)

| Method | A→W | A→D | D→A | W→A | Avg |
|--------|-----|-----|-----|-----|-----|
| Source Model | 77.8 | 82.1 | 64.5 | 66.1 | 72.6 |
| w/o 구조적 소스 정규화 | 87.3 | 92.1 | 73.9 | 75.0 | 82.1 |
| w/o 특징 판별화 | 94.2 | 94.3 | 74.3 | 75.5 | 84.6 |
| w/o 소프트 샘플 선택 | 94.8 | 94.6 | 74.6 | 75.7 | 84.9 |
| **SRDC (full)** | **95.7** | **95.8** | **76.7** | **77.1** | **86.3** |

#### ImageCLEF-DA 및 Office-Home

| 데이터셋 | SRDC Avg | 이전 최고 성능 |
|---------|---------|------------|
| ImageCLEF-DA | **90.9%** | SymNets: 89.9% |
| Office-Home | **71.3%** | MDD: 68.1% |

### 2.5 한계점

1. **도메인 간 차이가 매우 큰 경우**: Office-Home에서 일부 태스크(예: Ar→Cl=52.3%)는 여전히 낮은 성능
2. **클러스터 수 K 사전 지정**: 공유 레이블 공간 크기를 알아야 함 → 실제 응용에서 제약
3. **초기화 민감성**: K-means 초기화에 따른 성능 변동 가능성
4. **대규모 도메인 불일치**: 구조적 유사성 가정이 강한 도메인 시프트에서는 성립하지 않을 수 있음
5. **계산 복잡도**: 매 에폭마다 K-means 재실행 및 가중치 계산 필요
6. **단일 소스 도메인**: 멀티 소스 시나리오로의 확장이 직접적으로 논의되지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능을 높이는 핵심 메커니즘

#### (a) Inductive UDA vs Transductive UDA 관점

논문은 **inductive UDA**의 중요성을 강조한다. 도메인 정렬 전략은 oracle target classifier(타겟 레이블로 직접 학습한 분류기)로부터 멀어질 위험이 있는 반면, SRDC는 타겟 데이터의 내재적 구조를 보존하여 **더 oracle classifier에 근접한 모델** 학습이 가능하다.

**Table 2 (inductive UDA 비교):**

| Method | A→W | A→D | D→A | W→A | Avg |
|--------|-----|-----|-----|-----|-----|
| DANN | 80.8 | 82.4 | 66.0 | 64.6 | 73.5 |
| MCD | 86.5 | 86.7 | 72.4 | 70.9 | 79.1 |
| **SRDC** | **91.9** | **91.6** | **75.6** | **75.7** | **83.7** |
| Oracle Model | 98.8 | 97.6 | 87.8 | 87.8 | 93.0 |

SRDC는 Oracle Model과의 격차를 줄이는 데 있어 기존 방법보다 **유의미하게 우수**함을 보인다.

#### (b) 클러스터 균형 항의 역할

$$\sum_{k=1}^{K} \varrho^t_k \log \varrho^t_k$$

이 항은 **엔트로피 최대화**를 통해 모든 클래스에 균등하게 타겟 샘플을 배분하도록 유도하여, 특정 클래스에 편향되는 퇴화 솔루션(degenerate solution)을 방지한다.

#### (c) 소프트 소스 샘플 선택의 일반화 기여

$$w^s(\boldsymbol{x}^s) = \frac{1}{2}\left(1 + \frac{{\boldsymbol{c}^t_{y^s}}^\top \boldsymbol{x}^s}{\|\boldsymbol{c}^t_{y^s}\| \|\boldsymbol{x}^s\|}\right)$$

- 타겟 클러스터 중심과 소스 샘플 간의 코사인 유사도로 가중치 부여
- **도메인 유사 소스 샘플에 더 높은 가중치** → 부정적 전이(negative transfer) 완화
- 가중치 범위 (0.5~1.0): 합리적 범위 내에서 점진적 선택 → 급격한 분포 변화 방지

#### (d) 중간 특징 공간 클러스터링

Bottleneck 공간에서의 추가적인 클러스터링($\mathcal{L}^t_\varphi$)은:
- 거리 기반 소프트 할당으로 특징 공간의 **기하학적 구조** 보존
- 클러스터 센터 $\{\mu_k\}$를 소스+타겟 공동으로 초기화하여 **구조적 유사성 활용**

#### (e) 수렴 특성

Figure 5 분석:
- SRDC는 Source Model 대비 **빠르고 안정적인 수렴** 달성
- 훈련 초기 단계부터 테스트 에러 감소가 명확 → **최적화 안정성 향상**

---

## 4. 연구에 미치는 영향 및 앞으로의 고려 사항

### 4.1 앞으로의 연구에 미치는 영향

#### (A) 패러다임 전환 측면
SRDC는 UDA에서 **"정렬(Align)"에서 "발굴(Uncover)"로의 패러다임 전환**을 명확히 제시하였다. 이는 이후 연구들이 도메인 정렬의 부작용을 의식하고 클러스터링 기반 접근법을 더 적극적으로 채택하는 계기가 되었다.

#### (B) 이론적 기여
- 구조적 도메인 유사성(domain-wise discrimination + class-wise closeness)의 명시적 정의는 이후 UDA 이론 연구의 기반이 됨
- Inductive vs Transductive UDA의 차이를 명확히 구분하여 **실용적 UDA 평가 기준** 제시

#### (C) 방법론적 기여
- 소스 레이블을 보조 분포로 활용하는 **단순하고 우아한 joint training 전략**
- 이후 다양한 pseudo-label 기반 방법 발전에 영감 제공

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ 아래 비교 분석에 언급되는 논문들은 SRDC 논문 PDF에 포함되지 않은 외부 문헌입니다. arXiv 공개 정보를 기반으로 작성되었으며, 정확한 수치는 해당 논문을 직접 확인하시기 바랍니다.

#### 2020년 이후 주요 UDA 연구 동향

| 연구 방향 | 대표 논문 | SRDC와의 관계 |
|-----------|-----------|--------------|
| Source-free UDA | SHOT (ICML 2020) | 소스 데이터 없이 타겟만 적응 → SRDC보다 더 제약적 환경 |
| Self-training + Pseudo label | NRC (NeurIPS 2021) | SRDC의 클러스터링 아이디어 확장 |
| Transformer 기반 UDA | CDTrans (ICCV 2021) | 더 강력한 백본 활용 |
| Multi-source UDA | 다수 연구 | SRDC의 단일 소스 한계 극복 |

**SHOT (Liang et al., ICML 2020)**:
- 소스 모델을 고정하고 타겟 데이터만으로 적응
- 정보 최대화 + pseudo label 사용
- SRDC와 유사한 "정렬 없이 타겟 판별 구조 발굴" 방향성

**NRC (Yang et al., NeurIPS 2021)**:
- 이웃 클러스터링(Neighborhood Reciprocal) 활용
- SRDC의 클러스터링 접근법을 더 발전시킨 형태

#### 발전 방향 정리

```
SRDC (2020)
    ↓ 영향
Source-free UDA → 소스 데이터 의존성 제거
    ↓
Self-supervised + Clustering → 대규모 사전학습 모델 활용
    ↓
Foundation Model 기반 UDA (현재)
```

### 4.3 앞으로 연구 시 고려할 점

#### (A) 기술적 개선 방향

1. **더 강력한 백본 활용**
   - ViT(Vision Transformer), CLIP 등 대규모 사전학습 모델과의 결합
   - 더 풍부한 특징 표현이 구조적 유사성 가정 충족에 유리할 수 있음

2. **동적 클러스터 수 결정**
   - 고정된 $K$가 아닌 데이터 적응적 클러스터 수 결정 메커니즘 연구 필요
   - 오픈셋(open-set) UDA 시나리오 확장 가능성

3. **소스 없는(Source-free) 환경으로의 확장**
   - 현재 SRDC는 소스 데이터 접근을 전제하나, 프라이버시 보호 관점에서 소스 없는 설정이 중요해짐
   - 구조적 정규화를 소스 없는 환경에서 어떻게 대체할지 연구 필요

4. **멀티 소스 도메인 시나리오**
   - 여러 소스 도메인의 구조적 유사성을 어떻게 통합할지 고려

5. **클러스터링 초기화 안정성**
   - K-means 초기화의 불안정성 해소를 위한 더 robust한 초기화 전략

#### (B) 이론적 연구 방향

1. **구조적 도메인 유사성의 정량적 측정**
   - 현재 "가정"으로만 제시된 구조적 유사성을 **수치적으로 측정·검증**하는 방법론 개발

2. **클러스터링 기반 UDA의 일반화 오차 이론**

$$\varepsilon_T(h) \leq \varepsilon_S(h) + d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda^*$$

기존 Ben-David et al.의 UDA 이론에서, 클러스터링 기반 방법이 어떻게 $d_{\mathcal{H}\Delta\mathcal{H}}$ 항을 효과적으로 감소시키는지 이론적 분석 필요

3. **도메인 유사성과 성능 사이의 관계 규명**
   - 어느 정도의 도메인 시프트까지 SRDC의 가정이 유효한지 정량적 분석

#### (C) 실용적 고려사항

1. **레이블 공간 불일치(partial/open-set UDA)**: SRDC는 완전한 공유 레이블 공간을 가정하나, 실제로는 부분적 공유만 가능한 경우가 많음

2. **계산 효율성**: 대규모 데이터셋에서의 K-means 및 가중치 계산 비용 최적화

3. **온라인/연속 학습**: 실제 배포 환경에서 타겟 데이터가 스트리밍으로 들어오는 경우의 적용 방안

4. **다양한 모달리티**: 텍스트, 오디오 등 비전 외 도메인으로의 SRDC 확장

---

## 📊 종합 요약

```
SRDC의 핵심 가치
├── 철학적: 정렬(Align) → 발굴(Uncover) 패러다임 전환
├── 기술적: 구조적 정규화 + KL 발산 최소화 클러스터링
├── 실용적: 명시적 도메인 정렬 없이 SOTA 달성
└── 이론적: Inductive UDA에서의 일반화 우수성 입증
```

SRDC는 **"도메인 정렬이 반드시 최선인가?"** 라는 근본적 질문을 제기함으로써, UDA 분야의 연구 방향 다양화에 중요한 기여를 했으며, 향후 source-free UDA, foundation model 기반 UDA 등 다양한 연구의 이론적·방법론적 토대가 되고 있다.
