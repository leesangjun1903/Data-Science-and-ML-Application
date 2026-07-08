# Dynamic Domain Adaptation for Efficient Inference

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

"Dynamic Domain Adaptation for Efficient Inference" (Li et al., 2021, arXiv:2103.16403)는 다음의 핵심 문제를 제기합니다:

> **기존 도메인 적응(DA) 방법들은 정확도는 높지만, 실시간 응용(스마트폰, 웨어러블 기기 등)에서 요구되는 빠른 추론(Fast Inference)을 보장하지 못한다.**

이를 해결하기 위해 **Dynamic Domain Adaptation (DDA)** 프레임워크를 제안하며, 다음 두 목표를 동시에 달성하고자 합니다:

1. **효율적 타겟 추론** (저자원 시나리오)
2. **우수한 크로스 도메인 일반화** (도메인 적응 성능 유지)

### 주요 기여 (3가지)

| 기여 | 설명 |
|------|------|
| **DDA 프레임워크** | 다중 출구(multi-exit) 적응 네트워크에 도메인 혼동 손실을 통합 |
| **신뢰도 점수 학습 전략** | 다수 분류기의 예측 일관성을 활용한 정확한 의사 라벨 생성 |
| **클래스 균형 자기 학습 전략** | 예측 다양성을 유지하면서 모든 분류기를 점진적으로 적응 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 구조:**

$$P_s \neq P_t$$

소스 도메인 $D_s = \{(x_i^s, y_i^s)\}\_{i=1}^{N_s}$와 타겟 도메인 $D_t = \{x_j^t\}_{j=1}^{N_t}$는 서로 다른 분포를 따르며(도메인 시프트), 기존 방법들은 다음 두 문제를 동시에 해결하지 못합니다:

- **문제 1:** 정적(Static) 네트워크 기반 DA 방법 → 추론 비용 절감 불가
- **문제 2:** 경량화(Adaptive Inference) 모델 → 도메인 시프트에 취약, 일반화 성능 저하

특히, 다중 출구 네트워크에서 각 출구(early/late exit)별로 특징의 전이 가능성(transferability)이 다르기 때문에, 단순히 모든 출구에 도메인 혼동 손실을 적용하면 판별력(discriminability)이 저하됩니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 소스 지도 학습 (Source Supervised Learning)

모든 $K$개의 분류기에 대해 소스 도메인의 경험적 위험 최소화를 수행합니다:

$$\mathcal{L}_s = \frac{1}{N_s} \sum_{i=1}^{N_s} \sum_{k=1}^{K} \mathcal{E}(f_k(x_i^s; \theta_k), y_i^s) \tag{1}$$

여기서 $\mathcal{E}(\cdot, \cdot)$는 크로스엔트로피 손실이고, $f_k(x_i^s; \theta_k)$는 $k$번째 분류기의 확률 출력입니다.

#### Step 2: 도메인 혼동 학습 (Domain Confusion Learning)

이진 도메인 판별기를 활용한 적대적 도메인 손실:

$$\mathcal{L}_d = \frac{1}{N_s} \sum_{x \in D_s} \sum_{k=1}^{K} [\log D_k(F_k(x; \theta_k))] + \frac{1}{N_t} \sum_{x \in D_t} \sum_{k=1}^{K} [\log(1 - D_k(F_k(x; \theta_k)))] \tag{2}$$

여기서 $D_k(\cdot)$는 $k$번째 도메인 판별기이며, $F_k(x; \theta_k)$는 $k$번째 분류기 이전의 특징 표현입니다.

#### Step 3: 신뢰도 점수 학습 (Target Confidence Score Learning)

타겟 샘플 $x_j^t$에 대해 전체 분류기의 평균 예측값:

$$\bar{p}_j^t = \frac{1}{K} \sum_{k=1}^{K} f_k(x_j^t; \theta_k)$$

각 분류기의 예측과 평균 예측 간의 **코사인 유사도**를 기반으로 신뢰도 점수 계산:

$$v_j = \max(\bar{p}_j^t) \sum_{k=1}^{K} \frac{f_k(x_j^t; \theta_k) \cdot \bar{p}_j^t}{|f_k(x_j^t; \theta_k)||\bar{p}_j^t|} \tag{3}$$

- $\max(\bar{p}_j^t)$: 모든 분류기가 혼동될 경우(균등 분포)에 신뢰도가 높게 나오는 문제를 방지하는 스케일링 항
- 높은 코사인 유사도 → 높은 신뢰도 → 의사 라벨 신뢰 가능

#### Step 4: 클래스 균형 자기 학습 (Class-balanced Self-training)

클래스 $c$에 대한 클래스별 신뢰도 점수:

$$e_c = \frac{1}{N_t^c} \sum_{x_j^t \in \hat{D}_t^c} v_j \tag{4}$$

클래스 $c$의 샘플 선택 임계값 (선형 비례):

$$\lambda_c = N_t \times \mu \frac{e_c}{\sum_{i=1}^{C} e_i} \tag{5}$$

여기서 $\mu$는 전체 타겟 데이터 중 자기 학습에 사용할 비율을 결정하는 제어 인수입니다.

샘플 선택 결정 함수:

$$I_j^t = \begin{cases} 1, & \text{if } |U_c| < \lambda_c \text{ and } \hat{y}_{(v_j)}^t = c \\ 0, & \text{otherwise} \end{cases} \tag{6}$$

자기 학습 손실 (각 출구에 다른 샘플 할당):

$$\mathcal{L}_t = \frac{1}{|U|} \sum_{x_j^t \in U} \mathcal{E}(f_{k_j}(x_j^t; \theta_{k_j}), \hat{y}_j^t) \tag{7}$$

#### 최종 목적 함수 (Overall Objective)

$$\mathcal{L} = \mathcal{L}_s + \alpha \mathcal{L}_d + \beta \mathcal{L}_t \tag{8}$$

$\alpha$와 $\beta$는 균형 파라미터이며, 논문에서 $\alpha = \beta = 1.0$으로 설정했을 때 안정적인 성능을 보였습니다.

---

### 2.3 모델 구조

```
[소스/타겟 입력]
      ↓
[Block 1] → [Classifier f_1] → Exit 1 (Easy samples)
      ↓
[Block 2] → [Classifier f_2] → Exit 2
      ↓
     ...
      ↓
[Block K] → [Classifier f_K] → Exit K (Hard samples)
      ↓
[도메인 혼동 손실 L_d: 각 출구별 적용]
      ↓
[신뢰도 점수 계산 → 의사 라벨 생성]
      ↓
[클래스 균형 자기 학습 L_t]
```

**백본 네트워크:** MSDNet (Multi-Scale Dense Network)
- `DDA(S4)`: MSDNet(S4) - 블록당 4개 합성곱 레이어, 5개 분류기 출구
- `DDA(S7)`: MSDNet(S7) - 블록당 7개 합성곱 레이어, 5개 분류기 출구

**평가 설정:**
- **Anytime Prediction:** 임의의 시점에 예측 가능
- **Budgeted Classification:** 고정 계산 예산 내 자원 할당

---

### 2.4 성능 향상

| 비교 항목 | 성능 향상 |
|----------|-----------|
| DDA(S4)+DANN vs. ResNet50+DANN (Office31) | +2.1% 정확도, **4× FLOPs 절감** |
| DDA(S7)+CDAN vs. ResNet152+CDAN (VisDA-2017) | +4.5% 정확도, **3.6× FLOPs 절감** |
| DDA vs. MSDNet+DA (Office31-DANN) | +5.5% 평균 정확도 |
| DDA vs. MSDNet+DA (VisDA2017-CDAN) | +5.6% 평균 정확도 |
| DDA(S4)+DANN vs. 신뢰도 핸드크래프트(>0.9) | 69.4% vs. 63.1% (5th exit) |
| DDA(S4)+DANN (Office31, Budgeted) | 87.3% @ 0.6×10⁹ MUL-ADD, REDA 대비 +6.5% |

DomainNet 전체 30개 태스크에서 DDA(S7)+DANN이 ResNet152+DANN 대비 **평균 3.5%** 향상을 달성했습니다.

---

### 2.5 한계점

논문에서 명시적으로 또는 구조적으로 파악되는 한계:

1. **백본 의존성:** MSDNet을 주 백본으로 사용하지만, Vision Transformer(ViT) 등 최신 아키텍처와의 통합 가능성은 실험적으로 검증되지 않음
2. **의사 라벨 노이즈:** 초기 학습 단계에서 신뢰도 점수가 부정확할 수 있으며, 노이즈 누적(error accumulation) 위험 존재
3. **멀티 소스 도메인 미지원:** 단일 소스 도메인 가정에 한정
4. **하이퍼파라미터 $\mu$ 고정:** $\mu = 80\%$로 고정했으나, 태스크별 최적값이 다를 수 있음
5. **도메인 갭이 매우 큰 경우:** DomainNet의 Quickdraw(qdr) 도메인과 같이 도메인 갭이 극단적으로 클 때 성능 향상 폭이 제한적

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상의 핵심 메커니즘

DDA의 일반화 성능 향상은 세 가지 상호 보완적 메커니즘에서 비롯됩니다:

#### (1) 다중 분류기 신뢰도 집계를 통한 의사 라벨 품질 향상

단일 분류기 의존 방식 대비, 모든 $K$개 분류기의 예측을 종합함으로써:

$$\bar{p}_j^t = \frac{1}{K} \sum_{k=1}^{K} f_k(x_j^t; \theta_k)$$

이는 **앙상블 효과(Ensemble Effect)**를 내재적으로 활용하여, 단일 분류기의 편향(bias)을 감소시킵니다. 특히 각 출구가 서로 다른 receptive field를 가지므로, 다양한 스케일의 특징 정보가 통합됩니다.

#### (2) 클래스 균형 전략을 통한 결정 경계 정규화

수식 (5)의 $\lambda_c$ 설계는 전이 가능성이 낮은 클래스(낮은 $e_c$)도 자기 학습에 포함시켜:

$$\lambda_c = N_t \times \mu \frac{e_c}{\sum_{i=1}^{C} e_i}$$

- 특정 쉬운 클래스에 과적합(overfitting) 방지
- 어려운 클래스의 결정 경계를 점진적으로 개선
- **예측 다양성(Prediction Diversity)** 유지 → 각 출구에 서로 다른 샘플을 할당함으로써 분류기 간 상호 보완성 증가

논문의 ablation study에서 클래스 균형 전략을 제거(DDA w/o CB)하거나 대체(DDA w/ sub-CB)하면 성능이 저하됨이 확인됩니다.

#### (3) 다중 스케일 특징 정렬의 점진적 개선

초기에는 도메인 혼동 손실 $\mathcal{L}_d$만으로 각 출구별 독립적 특징 정렬을 수행하고, 이후 신뢰도 높은 의사 라벨을 활용한 $\mathcal{L}_t$로 보완합니다. 이는:

$$\mathcal{L} = \mathcal{L}_s + \alpha \mathcal{L}_d + \beta \mathcal{L}_t$$

- **전이 가능성과 판별력의 균형:** $\mathcal{L}_d$가 전이 가능성을 높이고, $\mathcal{L}_t$가 타겟 도메인 판별력을 강화
- DomainNet에서의 **귀납적 학습(Inductive Learning)** 설정에서 DDA(S7)+DANN이 ResNet152+DANN 대비 모든 하위 태스크에서 우월 → 단순 암기가 아닌 **강건한 결정 경계 학습** 시사

### 3.2 일반화 가능성의 추가 분석

**transferability analysis (Figure 4a)에서의 발견:**

특정 출구에만 DA를 적용하면 해당 출구의 성능만 향상되고 다른 출구에는 영향이 없음이 확인되었습니다. 이는:

- 각 출구가 **독립적인 도메인 적응 능력**을 보유함을 의미
- 모든 출구에 DA를 적용하는 DDA의 설계가 **전체적인 일반화 성능 극대화**에 기여함을 정당화

**일반화 향상 가능성이 특히 높은 시나리오:**
- 도메인 갭이 중간 수준인 경우 (매우 작거나 매우 크지 않은 경우)
- 클래스 수가 많고 클래스 간 전이 가능성 차이가 클 경우 (DomainNet의 345 클래스)
- 계산 예산 제약이 있는 실세계 배포 환경

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

#### (1) 효율적 DA 연구 방향 제시
DDA는 **"정확도 vs. 효율성"의 트레이드오프를 DA 맥락에서 명시적으로 다룬 초기 연구** 중 하나로, 엣지 AI 환경에서의 도메인 적응 연구의 기반이 됩니다.

#### (2) 다중 출구 아키텍처 + DA의 결합 패러다임 확립
정적 네트워크 중심이었던 DA 연구에 동적 추론(Dynamic Inference) 관점을 도입함으로써, 향후 연구들이 네트워크 깊이별 전이 가능성 차이를 고려하도록 유도합니다.

#### (3) 의사 라벨링의 다중 분류기 활용
단일 분류기 기반 의사 라벨 생성의 한계를 지적하고, 다중 분류기의 예측 일관성을 활용하는 새로운 신뢰도 측정 방식을 제시하여 semi-supervised learning 및 self-training 연구에 영향을 줄 수 있습니다.

### 4.2 향후 연구 시 고려할 점

#### (1) 더 강력한 백본 아키텍처와의 통합
- Vision Transformer(ViT), Swin Transformer 등과의 통합 필요
- 논문은 MSDNet에 특화되어 있으나, DDA 자체는 "orthogonal to other adaptive inference models"라고 주장 → 실험적 검증 필요

#### (2) 노이즈에 강건한 의사 라벨링 방법 개발
- 초기 학습 단계에서 신뢰도 점수가 부정확할 수 있음
- **불확실성 추정(Uncertainty Estimation)** 기법(예: MC-Dropout, Evidential Deep Learning)과의 결합 고려

#### (3) 멀티 소스/멀티 타겟 도메인 확장
- 현재는 단일 소스-단일 타겟 설정에 한정
- Multi-source DA, Universal DA, Open-set DA로의 확장 필요

#### (4) 온라인/지속적 도메인 적응 (Continual DA)
- 실세계에서는 타겟 도메인이 시간에 따라 변화함
- DDA의 자기 학습 전략을 온라인 학습 환경에 적용하는 연구 필요

#### (5) 이론적 일반화 경계 분석
- 논문은 실험적 검증에 집중하지만, 다중 출구 DA에 대한 이론적 일반화 경계(Generalization Bound) 분석이 부재
- MDD[56]에서 제시한 margin disparity discrepancy 기반 이론과의 연계 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 비교는 DDA 논문(2021년 3월)의 참고문헌 및 관련 분야의 공개 연구를 기반으로 합니다. **단, DDA 발표 이후의 연구(2021년 이후)에 대한 구체적 수치 비교는 본 논문에서 직접 인용되지 않았으므로, 공개적으로 알려진 연구 방향을 중심으로 서술합니다.**

### 5.1 DDA 논문에서 직접 비교된 관련 연구 (2020년 이후)

| 방법 | 발표 | 핵심 아이디어 | DDA와의 차이 |
|------|------|--------------|-------------|
| **REDA** (Jiang et al., ACM-MM 2020) | 2020 | MSDNet + 지식 증류 + 마지막 출구만 DA | DDA는 모든 출구에 DA 적용, 양방향 지식 공유 |
| **BSP** (Chen et al., ICML 2019) | 2019 | 배치 스펙트럼 페널티로 전이 가능성↔판별력 균형 | DDA는 효율적 추론 추가 |
| **CDAN** (Long et al., NeurIPS 2018) | 2018 | 조건부 적대적 DA | DDA는 CDAN을 $\mathcal{L}_d$로 대체 가능 (플러그인 구조) |

### 5.2 DDA 이후의 관련 연구 방향 (공개 문헌 기반)

다음 연구들은 DDA와 유사한 문제를 다루지만, 이 분석은 **DDA 논문 자체에서 인용되지 않은 내용**으로 확실성이 제한됩니다:

- **Test-Time Adaptation (TTA) 연구** (예: TENT, TTT++): 추론 시점에서 모델 적응 → DDA의 효율적 추론 목표와 방향성 일치하나 레이블 없는 타겟 적응 방식에서 차별화
- **Prompt Tuning 기반 DA** (대형 언어모델 맥락): 파라미터 효율적 적응 방법으로 DDA와 상호 보완적 연구 가능성

> ⚠️ **주의:** DDA 발표(2021년 3월) 이후의 구체적인 벤치마크 비교 수치는 본 논문에 포함되지 않으므로, 해당 수치를 확인하려면 최신 DA 벤치마크 논문 또는 Papers with Code (https://paperswithcode.com/task/domain-adaptation)를 직접 참조하시기 바랍니다.

---

## 참고 자료

**주요 참고 자료 (논문 내 인용 기준):**

1. **Li, S., Zhang, J., Ma, W., Liu, C. H., & Li, W. (2021).** *Dynamic Domain Adaptation for Efficient Inference.* arXiv:2103.16403v1. (본 분석의 주 대상 논문)
2. **Huang, G., Chen, D., Li, T., Wu, F., van der Maaten, L., & Weinberger, K. (2018).** *Multi-scale dense networks for resource efficient image classification.* ICLR 2018. [MSDNet 백본]
3. **Ganin, Y., & Lempitsky, V. (2015).** *Unsupervised domain adaptation by backpropagation.* ICML 2015. [DANN]
4. **Long, M., Cao, Z., Wang, J., & Jordan, M. I. (2018).** *Conditional adversarial domain adaptation.* NeurIPS 2018. [CDAN]
5. **Chen, X., Wang, S., Long, M., & Wang, J. (2019).** *Transferability vs. discriminability: Batch spectral penalization for adversarial domain adaptation.* ICML 2019. [BSP]
6. **Jiang, J., Wang, X., Long, M., & Wang, J. (2020).** *Resource efficient domain adaptation.* ACM-MM 2020. [REDA]
7. **Zou, Y., Yu, Z., Kumar, B. V. K. V., & Wang, J. (2018).** *Unsupervised domain adaptation for semantic segmentation via class-balanced self-training.* ECCV 2018.
8. **Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014).** *How transferable are features in deep neural networks?* NeurIPS 2014.
9. **Pan, S. J., & Yang, Q. (2010).** *A survey on transfer learning.* TKDE, 22(10), 1345-1359.
10. **Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., & Wang, B. (2019).** *Moment matching for multi-source domain adaptation.* ICCV 2019. [DomainNet 데이터셋]
