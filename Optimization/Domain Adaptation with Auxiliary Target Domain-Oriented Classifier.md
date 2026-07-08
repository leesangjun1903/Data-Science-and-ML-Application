# Domain Adaptation with Auxiliary Target Domain-Oriented Classifier (ATDOC) 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 Domain Adaptation(DA) 방법들이 소스 데이터로 학습된 분류기(classifier)의 편향(bias) 문제를 무시한다는 점을 지적하며, **타겟 도메인 전용 보조 분류기(Auxiliary Target Domain-Oriented Classifier, ATDOC)**를 통해 의사 레이블(pseudo-label)의 품질을 향상시켜 도메인 적응 성능을 개선할 수 있다고 주장합니다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **(i) 새로운 프레임워크 제안** | 분류기 편향 문제를 해결하는 ATDOC 프레임워크 |
| **(ii) 비파라메트릭 분류기** | 추가 네트워크 파라미터 없이 메모리 뱅크 기반 두 가지 분류기 개발 |
| **(iii) 광범위한 적용성** | UDA, SSDA, PDA, 희소 레이블 SSL 등 다양한 설정에서 SOTA 달성 |
| **(iv) 플러그인 호환성** | CDAN 등 기존 도메인 정렬 방법과 결합 가능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**핵심 문제: 소스 편향된 분류기(Source-biased Classifier)**

기존 의사 레이블링(pseudo-labeling) 방식은 소스 데이터로 학습된 분류기 $F$를 이용해 타겟 도메인의 레이블을 예측합니다:

$$\hat{y}_i = \arg\max_k p_{i,k}, \quad i = 1, 2, \cdots, N_{tu}$$

$$\mathcal{L}_{pl} = -\frac{\alpha}{N_{tu}} \sum_{i=1}^{N_{tu}} \log p_{i,\hat{y}_i} \tag{1}$$

이 방식의 문제점:
- 분류기가 소스 데이터에 편향되어 타겟 도메인에서 낮은 품질의 의사 레이블 생성
- 도메인 이동(domain shift)을 무시
- 오류 누적(error accumulation) 발생

### 2.2 제안하는 방법

저자들은 먼저 신뢰도 가중치를 반영한 개선된 의사 레이블링을 기반으로:

$$\hat{y}_i = \arg\max_k p_{i,k}, \quad i = 1, 2, \cdots, N_t$$

$$\mathcal{L}^{ours}_{pl} = -\frac{\lambda}{N_{tu}} \sum_{i=1}^{N_{tu}} p_{i,\hat{y}_i} \log p_{i,\hat{y}_i} \tag{2}$$

여기서 $\lambda$는 선형 스케줄러로 조정됩니다.

---

#### 2.2.1 ATDOC-NC: 최근접 중심 분류기 (Nearest Centroid Classifier)

**메모리 뱅크 업데이트 (EMA 방식):**

$$c_j = \sum_{i \in B_t} \mathbb{1}_{[j=\hat{y}_i]} G(x^t_i) / \sum_{i \in B_t} \mathbb{1}_{[j=\hat{y}_i]}$$

$$c^m_j = \gamma c_j + (1-\gamma) c^m_j, \quad m = 1, 2, \cdots, K \tag{3}$$

- $B_t$: 타겟 도메인의 미니배치 인덱스 집합
- $\gamma = 0.1$: 스무딩 파라미터

**의사 레이블 생성 (코사인 거리 기반):**

$$\hat{y}_i = \arg\min_{j=1}^{K} d\left(G(x^t_i), c^m_j\right), \quad i = 1, 2, \cdots, N_t \tag{4}$$

**NC 손실함수:**

$$\mathcal{L}_{nc} = -\frac{\lambda}{N_{tu}} \sum_{i=1}^{N_{tu}} \log p_{i,\hat{y}_i} \tag{5}$$

---

#### 2.2.2 ATDOC-NA: 이웃 집계 (Neighborhood Aggregation)

**예측값 샤프닝 및 클래스 균형화:**

$$\check{p}^m_{i,k} = p^{1/T}_{i,k} / \sum_i p^{1/T}_{i,k} \tag{6}$$

- $T = 0.5$: 온도 파라미터 (샤프닝 강도 조절)

**이웃 집계를 통한 새로운 예측:**

$$\hat{q}_i = \frac{1}{m} \sum_{j \neq i, j \in \mathcal{N}_i} \check{p}_j \tag{7}$$

- $\mathcal{N}_i$: 메모리 모듈에서 $x^t_i$에 대한 $m$개 최근접 이웃 인덱스 집합

**신뢰도 가중 교차 엔트로피 손실:**

$$\mathcal{L}_{na} = -\frac{\lambda}{N_{tu}} \sum_{i=1}^{N_{tu}} \hat{q}_{i,\hat{y}_i} \log p_{i,\hat{y}_i} \tag{8}$$

**최종 목적함수:**

$$\mathcal{L} = \mathcal{L}^s_{lsr}(\mathcal{D}_s) + \mathcal{L}^t_{lsr}(\mathcal{D}_{tl}) + \mathcal{L}_{nc/na}(\mathcal{D}_{tu}) \tag{9}$$

- $\mathcal{L}^s_{lsr}$, $\mathcal{L}^t_{lsr}$: 레이블 스무딩 정규화가 적용된 교차 엔트로피 손실

---

### 2.3 모델 구조

```
[소스 데이터 + 레이블] ─────────────────────────┐
                                                  ▼
[타겟 데이터 (레이블 없음)] → Feature Extractor G → Classifier F → Standard CE Loss
                    │                                          ↑
                    │         ┌─────────────────────────────┐ │
                    └────────→│     Memory Bank              │ │
                              │  (features + predictions)   │ │
                              └─────────────────────────────┘ │
                                         │                     │
                              ┌──────────▼──────────┐         │
                              │  NC: 클래스 중심 업데이트│         │
                              │  NA: 이웃 집계       │─────────┘
                              └─────────────────────┘
                              Target-oriented Pseudo Labels + Confidence Weights
```

---

### 2.4 성능 향상

#### Office-31 (ResNet-50)

| 방법 | Avg. |
|------|------|
| ResNet-50 (기준) | 76.5% |
| Pseudo-labeling | 84.7% |
| BNM | 89.2% |
| **ATDOC-NA** | **89.7%** |
| CDAN+E + ATDOC-NA | **90.4%** |

#### VisDA-C (ResNet-101)

| 방법 | Mean |
|------|------|
| ResNet-101 (기준) | 49.1% |
| SHOT | 82.9% |
| **MixMatch + ATDOC-NA** | **86.3%** |

#### Office-Home (ResNet-50)

| 방법 | Avg. |
|------|------|
| SHOT | 71.8% |
| **CDAN+E + ATDOC-NA** | **73.2%** |

### 2.5 한계점

1. **메모리 효율성**: 전체 타겟 샘플의 피처와 예측값을 저장하는 대용량 메모리 뱅크 필요
2. **폐쇄 집합 설정**: 주로 소스와 타겟이 동일한 클래스를 공유하는 closed-set 설정에 집중
3. **초기 편향**: 학습 초기에는 여전히 소스 데이터에 편향된 상태에서 출발
4. **하이퍼파라미터 민감성**: $T$, $m$, $\lambda$ 등 여러 하이퍼파라미터 조정 필요
5. **밀집 예측 태스크 미적용**: 시맨틱 세그멘테이션 등 픽셀 레벨 태스크에 직접 적용 어려움

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 가능하게 하는 핵심 메커니즘

#### (A) 타겟 도메인 구조 활용

기존 방법이 인스턴스 단위 예측만 고려하는 것과 달리, ATDOC는 **전역 구조(global structure)**를 학습합니다:

- **NC**: 클래스 중심(centroid)을 통한 전역 클래스 구조 포착
- **NA**: 이웃 집계를 통한 로컬 매니폴드 구조 포착

이는 타겟 도메인의 데이터 분포를 더 잘 반영하므로, 소스 도메인에 편향된 분류기보다 일반화 성능이 높습니다.

#### (B) 밀도 기반 신뢰도 가중치

$$\hat{q}_{i,\hat{y}_i} \propto \text{neighborhood density}$$

고밀도 영역에 있는 샘플에 더 높은 가중치를 부여함으로써, 노이즈가 많은 경계 영역 샘플의 영향을 줄입니다.

#### (C) 예측 샤프닝과 클래스 균형화

$$\check{p}^m_{i,k} = p^{1/T}_{i,k} / \sum_i p^{1/T}_{i,k}$$

- **샤프닝**: 예측의 확신도를 높여 의사 레이블의 정확도 향상
- **클래스 균형화**: 특정 클래스로의 붕괴(degenerate solution) 방지 → 다양성(diversity) 확보

#### (D) 비파라메트릭 분류기

추가 파라미터 없이 메모리 뱅크만으로 구현되므로, 과적합(overfitting) 위험이 낮고 다양한 아키텍처에 범용적으로 적용 가능합니다.

#### (E) 플러그인 호환성

$$\mathcal{L}_{total} = \mathcal{L}_{domain\_alignment} + \mathcal{L}_{nc/na}$$

CDAN 등 기존 방법에 추가적으로 결합하여 상호 보완적으로 성능을 향상시킵니다.

### 3.2 일반화 성능의 실험적 근거

**희소 레이블 SSL (도메인 이동 없음)에서도 성능 향상:**

| 방법 | Office-Home Avg. | DomainNet-126 Avg. |
|------|------------------|--------------------|
| BNM | 63.2% | 49.4% |
| MCC | 64.6% | 51.2% |
| **ATDOC-NA** | **65.0%** | **57.2%** |

이는 ATDOC의 일반화 능력이 도메인 이동 상황에만 국한되지 않음을 보여줍니다.

**t-SNE 시각화**: Source-only 및 Pseudo-labeling 대비 ATDOC-NA가 두 도메인의 피처를 더 잘 정렬함을 확인.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (A) 분류기 편향 문제의 재조명
이 논문은 도메인 적응 연구에서 **피처 정렬(feature alignment)** 위주의 접근에서 벗어나, **분류기 편향 해소**라는 새로운 관점을 제시했습니다. 이는 이후 연구들(예: SHOT, NRC 등)이 타겟 전용 학습 전략에 집중하도록 영향을 미쳤습니다.

#### (B) 메모리 뱅크 + 의사 레이블링의 결합
메모리 뱅크를 통한 전역 구조 학습과 의사 레이블링을 결합한 프레임워크는 이후 **대조 학습(contrastive learning) 기반 DA** 연구의 기초가 되었습니다.

#### (C) 범용 SSL 기술의 DA 적용
ATDOC가 도메인 이동이 없는 순수 SSL 설정에서도 효과적임을 보였으므로, **SSL과 DA의 통합 프레임워크** 연구를 촉진했습니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법론 | 핵심 아이디어 | ATDOC와의 관계 |
|------|--------|--------------|----------------|
| **SHOT** (ICML 2020, Liang et al.) | 소스 가설 전이 | 소스 데이터 없이 타겟 특화 피처 추출기 학습 | ATDOC의 타겟 중심 학습 철학 공유 |
| **NRC** (NeurIPS 2021, Yang et al.) | 이웃 상호 클러스터링 | 상호 최근접 이웃 기반 의사 레이블 정제 | ATDOC-NA의 이웃 집계 개념 확장 |
| **DaC** (CVPR 2022) | 분리 및 보정 | 소스-타겟 분리 후 타겟 분포 보정 | 분류기 편향 해소 문제 공유 |
| **ViDA** (ICLR 2023) | Vision Transformer 기반 DA | ViT의 피처를 활용한 도메인 적응 | ATDOC를 ViT 백본으로 확장 가능성 |
| **PLUE** (ECCV 2022) | 불확실성 기반 의사 레이블 | 의사 레이블의 불확실성 정량화 | ATDOC의 신뢰도 가중치 개념 정교화 |

> **주의**: 위 비교 표의 일부 연구(DaC, ViDA, PLUE 등)는 논문에 직접 인용된 것이 아니므로, 상세 내용 확인 시 원문 참조를 권장합니다.

### 4.3 앞으로 연구 시 고려할 점

#### (A) 오픈셋 및 유니버설 DA로의 확장
현재 ATDOC는 closed-set 설정에 최적화되어 있습니다. 소스와 타겟의 클래스가 다른 **open-set DA** 또는 **universal DA** 설정에서는 메모리 뱅크의 클래스 중심이 잘못 초기화될 위험이 있으므로, 이를 해결하는 메커니즘이 필요합니다.

#### (B) 대규모 데이터셋 및 모델에서의 메모리 효율성
ATDOC-NA는 전체 타겟 샘플의 피처를 저장해야 하므로, 수백만 개의 샘플을 다루는 대규모 설정에서는 메모리 비용이 문제가 됩니다. **동적 메모리 관리** 또는 **압축 표현** 기법 연구가 필요합니다.

#### (C) Vision Transformer(ViT) 백본과의 결합
논문은 ResNet을 주로 사용했으나, ViT 기반 모델에서 ATDOC의 메모리 뱅크가 어떻게 동작하는지 탐구할 필요가 있습니다. ViT의 [CLS] 토큰 피처를 중심 업데이트에 활용하는 방안을 고려할 수 있습니다.

#### (D) 노이즈 레이블 강건성
초기 의사 레이블의 품질이 낮을 경우 메모리 뱅크가 오염될 수 있습니다. **노이즈 레이블 학습(noisy label learning)** 기법과의 결합을 통해 강건성을 높이는 연구가 필요합니다.

#### (E) 연속 도메인 적응 (Continual DA)
단일 소스-타겟 쌍이 아닌, **연속적으로 변화하는 도메인**에서 ATDOC의 메모리 뱅크를 점진적으로 업데이트하는 방법을 연구할 필요가 있습니다.

#### (F) 이론적 보장
현재 ATDOC는 실험적 검증에 의존합니다. 타겟 지향 분류기가 도메인 갭을 줄인다는 **이론적 수렴 보장**을 제공하는 연구가 향후 중요할 것입니다.

---

## 참고자료

**주요 참고자료 (논문 원문 기준):**

1. **Liang, J., Hu, D., & Feng, J. (2021).** "Domain Adaptation with Auxiliary Target Domain-Oriented Classifier." *arXiv:2007.04171v5*
2. **Lee, D.-H. (2013).** "Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks." *ICML Workshop*
3. **Long, M. et al. (2018).** "Conditional adversarial domain adaptation." *NeurIPS*
4. **Cui, S. et al. (2020).** "Towards discriminability and diversity: Batch nuclear-norm maximization." *CVPR*
5. **Jin, Y. et al. (2020).** "Minimum class confusion for versatile domain adaptation." *ECCV*
6. **Berthelot, D. et al. (2019).** "MixMatch: A holistic approach to semi-supervised learning." *NeurIPS*
7. **Liang, J. et al. (2020).** "Do we really need to access the source data? SHOT." *ICML*
8. **Saito, K. et al. (2019).** "Semi-supervised domain adaptation via minimax entropy." *ICCV*

**GitHub 공식 코드:**
- https://github.com/tim-learn/ATDOC
