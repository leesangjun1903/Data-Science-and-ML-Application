# Identifying Mislabeled Data using the Area Under the Margin Ranking

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

딥 뉴럴 네트워크 훈련 시 **잘못 레이블된(mislabeled) 데이터**는 과적합을 유발하고 일반화 성능을 저하시킨다. 이 논문은 훈련 동역학(training dynamics)만을 관찰하여 mislabeled 샘플을 자동으로 식별·제거함으로써 모델 성능을 향상시킬 수 있다고 주장한다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **AUM 통계량 도입** | 훈련 중 margin의 누적 평균으로 clean/noisy 샘플 구별 |
| **Threshold Sample 기법** | 추가 클래스(c+1)에 의도적으로 잘못 레이블된 샘플을 넣어 AUM 임계값 자동 학습 |
| **Plug-and-Play 호환성** | 기존 분류 모델 구조 변경 없이 적용 가능 (`pip install aum`) |
| **실증적 성능 향상** | WebVision50에서 17% 데이터 제거 → 테스트 오류 1.6% 절대 감소, CIFAR100에서 13% 제거 → 1.2% 감소 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

- **과파라미터화된 딥 네트워크**는 무작위로 할당된 레이블조차 완전히 암기(memorize)할 수 있음 (Zhang et al., 2017)
- 웹 스크래핑, 크라우드소싱 등으로 수집된 대규모 데이터셋에는 필연적으로 mislabeled 샘플 포함
- MNIST, ImageNet 등 정제된 데이터셋조차 유해한 샘플 존재
- 기존 방법들(robust loss function, 다단계 파이프라인)은 복잡하거나 특정 노이즈 모델 가정에 의존

### 2.2 제안하는 방법 및 수식

#### Step 1: Margin 정의

훈련 epoch $t$에서 샘플 $(\mathbf{x}, y)$의 margin은:

$$M^{(t)}(\mathbf{x}, y) = \underbrace{z_y^{(t)}(\mathbf{x})}_{\text{assigned logit}} - \underbrace{\max_{i \neq y} z_i^{(t)}(\mathbf{x})}_{\text{largest other logit}} \tag{1}$$

- $z_y^{(t)}(\mathbf{x}) \in \mathbb{R}^c$: epoch $t$에서의 pre-softmax logit 벡터
- **양의 margin**: 올바른 예측 (assigned logit > 다른 모든 logit)
- **음의 margin**: 잘못된 예측

#### Step 2: AUM(Area Under the Margin) 정의

전체 $T$ epoch에 걸친 margin의 평균:

$$\text{AUM}(\mathbf{x}, y) = \frac{1}{T} \sum_{t=1}^{T} M^{(t)}(\mathbf{x}, y) \tag{2}$$

**직관:** mislabeled 샘플(예: BIRD를 DOG로 레이블)은 다른 올바르게 레이블된 BIRD 샘플들로부터의 gradient 업데이트로 인해 DOG logit이 지속적으로 낮게 유지 → 작고 음수인 AUM

#### Step 3: Threshold Sample 생성

$N$개의 $c$-class 훈련 데이터셋에서:

$$\mathcal{D}'_{\text{train}} = \{(\mathbf{x}, c+1) : \mathbf{x} \in \mathcal{D}_{\text{THR}}\} \cup (\mathcal{D}_{\text{train}} \setminus \mathcal{D}_{\text{THR}})$$

- $|\mathcal{D}_{\text{THR}}| = N/(c+1)$개의 샘플을 무작위 선택하여 새 클래스 $c+1$로 재할당
- $c+1$ 클래스는 실제로 존재하지 않으므로 네트워크는 memorization을 통해서만 해당 logit 증가 가능
- → threshold sample의 AUM은 mislabeled 샘플과 유사하게 낮음

#### Step 4: 임계값 결정 및 식별

$$\alpha = 99\text{th percentile of } \{\text{AUM}(\mathbf{x}, c+1) : \mathbf{x} \in \mathcal{D}_{\text{THR}}\}$$

mislabeled 샘플 식별:

$$\{(\mathbf{x}, y) \in (\mathcal{D}_{\text{train}} \setminus \mathcal{D}_{\text{THR}}) : \text{AUM}_{\mathbf{x},y} \leq \alpha\}$$

#### 전체 알고리즘 절차

```
1. DTHR 생성 (N/(c+1)개 샘플, 레이블 = c+1)
2. D'train = {(x, c+1): x∈DTHR} ∪ (Dtrain \ DTHR) 구성
3. D'train으로 첫 번째 learning rate drop까지 훈련, 모든 샘플의 AUM 측정
4. α = 99번째 백분위 threshold sample AUM 계산
5. AUM ≤ α인 샘플을 mislabeled로 식별 후 제거
6. 나머지 DTHR 샘플에 대해 절차 반복 (다른 threshold set 사용)
7. 정제된 데이터셋으로 최종 모델 훈련
```

### 2.3 모델 구조

- **모델 자체**: 기존 분류 네트워크 그대로 사용 (ResNet-32, ResNet-50 등)
- **유일한 구조적 변경**: 출력층에 뉴런 1개 추가 (class $c+1$ 대응)
- **훈련 중 추가 작업**: 매 epoch마다 각 샘플의 logit 기록 → AUM 누적 계산

### 2.4 성능 향상

#### 합성 노이즈 데이터셋 (균일 노이즈, ResNet-32)

| Dataset | Noise | Standard | AUM | Oracle |
|---------|-------|----------|-----|--------|
| CIFAR10 | 20% | 25.0% | **9.8%** | 9.0% |
| CIFAR10 | 40% | 43.3% | **12.5%** | 9.7% |
| CIFAR100 | 20% | 50.4% | **34.5%** | 35.5% |
| CIFAR100 | 40% | 62.5% | **38.7%** | 39.0% |

> CIFAR100 20%/40% 노이즈에서 Oracle 성능을 **초과** (mislabeled뿐 아니라 ambiguous 샘플도 제거)

#### 실세계 데이터셋

| Dataset | Standard Error | AUM Error | 제거 비율 |
|---------|---------------|-----------|---------|
| WebVision50 | 21.4% | **19.8%** | 17.8% |
| Clothing100K | 35.8% | **33.5%** | 16.7% |
| CIFAR100 | 33.0% | **31.8%** | 13.0% |

### 2.5 한계점

1. **비대칭 노이즈(Asymmetric Noise) 취약성**: 모든 BIRD가 DOG로만 mislabeled되는 경우, threshold sample의 AUM이 mislabeled sample의 AUM보다 더 낮아져 recall이 급격히 저하
2. **대규모 데이터셋**: 데이터가 많을수록 네트워크가 더 많은 훈련이 필요해 mislabeled 식별이 어려워짐 (Clothing1M 전체 vs. 100K subset 비교에서 확인)
3. **두 번의 훈련 필요**: AUM 계산용 + 최종 훈련용으로 총 약 1.5~2배의 계산 비용

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Margin과 일반화의 이론적 연결

논문은 margin 기반 일반화 이론(Bartlett et al., 1997; Bartlett et al., 2017)에 근거한다. Spectrally-normalized margin bound에 따르면:

$$\text{Generalization Gap} \leq \mathcal{O}\left(\frac{\|W\|_F \cdot \prod_l \sigma_l}{\gamma \sqrt{m}}\right)$$

여기서 $\gamma$는 margin이다. **큰 margin → 더 나은 일반화 보장**이라는 원리를 데이터 선별에 활용.

### 3.2 AUM이 일반화를 향상시키는 메커니즘

#### (a) Memorization 감소

```
Mislabeled sample 제거
→ 네트워크가 sample-specific 필터 학습 불필요
→ 과적합 방지
→ 일반화 향상
```

#### (b) Oracle 성능 초과 현상

CIFAR100 (20%, 40% noise)에서 AUM이 Oracle보다 낮은 테스트 오류를 기록:

- Oracle: 정확히 mislabeled만 제거
- AUM: **ambiguous(경계선상의) 올바르게 레이블된 샘플도 제거**
- → 이러한 ambiguous 샘플들이 오히려 generalization에 해롭다는 실증적 증거

이는 AUM이 단순한 mislabeled 탐지 도구를 넘어 **데이터 품질 기반 일반화 향상 도구**임을 시사한다.

#### (c) AUM 랭킹에 따른 데이터 제거 최적점

![Figure 5 개념 설명]

$$\text{Best Test Error} = \underset{\text{threshold}}{\arg\min} \text{ Test Error} \approx \text{99th percentile of threshold AUM}$$

- AUM 순서로 데이터 제거 시 뚜렷한 최적점 존재 (random 제거는 단조 증가)
- 이 최적점이 threshold sample의 99번째 백분위와 일치 → 자동 임계값 설정의 타당성 확인

#### (d) 아키텍처 독립성 → 데이터셋 고유 특성 포착

Spearman's 상관계수 비교:

| 지표 | 아키텍처 간 상관 |
|------|--------------|
| **AUM** | **> 98%** |
| Margin (단일 epoch) | ~75% |
| Training Loss | ~75% |
| Validation Loss | ~40% |

AUM이 모델 의존적 변동이 아닌 **데이터셋 고유의 특성**을 포착한다는 것을 의미 → 다양한 모델에서 일관된 일반화 향상 기대 가능

#### (e) Double Descent 현상과의 관계

논문은 AUM이 자연적으로 노이즈가 있는 데이터셋에서의 **double descent 현상**을 완화할 수 있는지를 미래 연구 방향으로 제시. Mislabeled 데이터 제거 → interpolation threshold 변화 → double descent 구조 자체가 변할 수 있음.

---

## 4. 연구에 미치는 영향 및 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### (a) 데이터 중심 AI(Data-Centric AI) 패러다임 강화

AUM은 "모델 아키텍처 개선"이 아닌 **"데이터 품질 개선"**을 통한 성능 향상의 강력한 사례를 제시. Andrew Ng이 주창하는 Data-Centric AI 관점을 실증적으로 뒷받침하며, 이후 연구들이 데이터 curation을 모델 개발의 핵심 단계로 포함하도록 유도.

#### (b) 훈련 동역학(Training Dynamics) 연구 촉진

Dataset Cartography (Swayamdipta et al., 2020, EMNLP)와 같이 훈련 과정에서의 신호를 이용한 데이터 분석 연구 흐름과 맥을 같이 하며, 이 분야의 후속 연구를 자극:

- **Curriculum Learning**: AUM 기반 쉬운 샘플 → 어려운 샘플 순서 학습
- **Active Learning**: AUM 낮은 샘플 우선 재레이블링 요청
- **Semi-supervised Learning**: 제거된 샘플을 unlabeled data로 활용

#### (c) Foundation Model 사전훈련 데이터 품질 향상

대규모 웹 크롤링 데이터(LAION 등)로 훈련되는 CLIP, GPT, LLaMA 등의 Foundation Model에서 사전훈련 데이터 정제에 AUM 원리 적용 가능. 최근 **LAION-5B의 품질 문제**가 지적되는 상황에서 scalable한 데이터 정제 방법으로서의 가치 증대.

#### (d) LLM Fine-tuning 데이터 품질 관리

RLHF(Reinforcement Learning from Human Feedback)에서 사람 평가자의 레이블 오류 탐지, Instruction Tuning 데이터 품질 향상 등에 AUM 원리 적용 가능성.

### 4.2 앞으로 연구 시 고려할 점

#### ⚠️ 방법론적 고려사항

**1. 비대칭 노이즈 처리 개선**

현재 AUM은 systematic/asymmetric noise에 취약. 다음을 고려해야 함:

$$\text{Improved threshold} = f(\text{noise type}, \text{class distribution}, \text{AUM distribution})$$

- 클래스별 AUM 분포를 독립적으로 모델링
- 노이즈 유형을 먼저 추정 후 threshold 적응적 조정

**2. 대규모 데이터셋에서의 확장성**

- AUM 계산을 위한 추가 훈련 비용: 데이터셋이 클수록 비용 증가
- **Mini-batch AUM**: 전체 데이터를 한 번에 처리하지 않고 스트리밍 방식으로 AUM 추정
- **Early AUM**: 초기 몇 epoch만으로도 충분한 신호 확보 가능 여부 탐색

**3. Continual Learning / Online 설정으로의 확장**

데이터가 스트림으로 들어오는 환경에서 실시간 AUM 계산 및 데이터 필터링 방법 개발 필요.

**4. 다중 모달 데이터로의 확장**

텍스트-이미지 쌍(예: CLIP 훈련 데이터), 음성-텍스트 쌍 등에서 AUM 원리 적용 시 모달리티별 margin 정의 필요.

**5. 편향(Bias) 문제 주의**

논문 자체가 언급하듯, 자동 식별 절차는 기존 데이터셋의 편향을 증폭시킬 위험 존재:

$$P(\text{flagged} | \text{minority group}) > P(\text{flagged} | \text{majority group})$$

소수 클래스나 희귀 특징을 가진 올바른 샘플이 AUM 낮음 → mislabeled로 오분류 → 데이터 다양성 감소.

**6. 레이블 효율성과의 결합**

$$\text{Semi-supervised AUM} = \text{AUM identification} + \text{pseudo-labeling of removed samples}$$

제거된 샘플을 완전히 버리지 않고 semi-supervised 또는 self-supervised 방식으로 재활용하는 전략 탐색.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 연구들 중 일부는 논문에서 직접 인용된 것(2020년 이전/당시)과 제 훈련 데이터 기반 지식을 구분하여 제시합니다. 2020년 이후 연구에 대해서는 제 지식 컷오프(2024년 초) 기준으로 서술하되, 논문 내 직접 언급된 내용과 명확히 구분합니다.

### 5.1 논문에서 직접 비교한 동시대 연구 (2020년)

| 방법 | 핵심 아이디어 | AUM 대비 장단점 |
|------|-------------|----------------|
| **DivideMix** (Li et al., ICLR 2020) | GMM으로 clean/noisy 분리 후 semi-supervised 학습 | 더 복잡한 파이프라인, semi-SSL의 이점 포함 |
| **SELF** (Nguyen et al., ICLR 2020) | Self-ensembling으로 noisy 레이블 필터링 | 앙상블 필요 → 계산 비용 증가 |
| **Confident Learning** (Northcutt et al., 2019→JAIR 2021) | 클래스별 confident joint를 추정하여 노이즈 행렬 학습 | AUM과 유사하게 margin 활용, 보완적 관계 |
| **Dataset Cartography** (Swayamdipta et al., EMNLP 2020) | 훈련 동역학으로 easy/hard/ambiguous 샘플 지도 작성 | 분류는 하나 자동 제거는 미제공 |

### 5.2 AUM 이후 발전된 연구 방향 (2021~2024, 제 지식 기반)

**[주의: 아래 내용은 확실성이 100%가 아닐 수 있으며, 논문 원문에서 직접 확인되지 않은 정보입니다]**

#### (a) Confident Learning의 발전 (Cleanlab)

- Northcutt et al.의 Confident Learning이 **Cleanlab** 도구로 발전
- AUM과 달리 노이즈 행렬을 명시적으로 추정, 클래스 간 혼동 패턴 파악 가능
- AUM은 훈련 동역학 기반, Confident Learning은 모델 출력 확률 기반으로 **상호 보완적**

#### (b) 대규모 데이터 정제로의 확장

- **DataComp** (Gadre et al., 2023): CLIP 훈련을 위한 대규모 데이터 필터링 벤치마크
- AUM 원리와 유사하게 모델 기반 필터링을 체계화
- 규모의 차이: AUM은 ~100K 이미지, DataComp는 수십억 이미지

#### (c) LLM 데이터 정제로의 적용

- Instruction tuning 데이터에서 품질 낮은 샘플 식별에 유사한 훈련 동역학 활용 연구들 등장
- **LIMA** (Zhou et al., 2023): 소량의 고품질 데이터만으로도 강력한 LLM 훈련 가능 → AUM의 "데이터 품질이 양보다 중요" 주장과 일맥상통

#### (d) Foundation Model을 활용한 레이블 품질 평가

- CLIP 등 사전훈련 모델의 임베딩을 이용해 레이블 일관성 점수 계산
- AUM과 달리 별도 훈련 없이 적용 가능하나, 도메인 특화 데이터에서는 AUM이 유리

### 5.3 종합 비교표

| 방법 | 연도 | 노이즈 모델 가정 | 신뢰 데이터 필요 | 계산 비용 | 비대칭 노이즈 |
|------|------|---------------|--------------|---------|-------------|
| **AUM (본 논문)** | 2020 | 불필요 | 불필요 | 중간 (2x 훈련) | 취약 |
| DivideMix | 2020 | 불필요 | 불필요 | 높음 | 보통 |
| Confident Learning | 2021 | 불필요 | 불필요 | 낮음 | 보통 |
| INCV | 2019 | 불필요 | 불필요 | 높음 | 보통 |
| Co-teaching | 2018 | 노이즈율 필요 | 불필요 | 중간 | 강함 |

---

## 참고자료

**주요 참고 문헌 (논문 내 인용)**

1. **Pleiss, G., Zhang, T., Elenberg, E., & Weinberger, K. Q. (2020).** "Identifying Mislabeled Data using the Area Under the Margin Ranking." *NeurIPS 2020*. arXiv:2001.10528v4

2. **Li, J., Socher, R., & Hoi, S. C. (2020).** "DivideMix: Learning with Noisy Labels as Semi-Supervised Learning." *ICLR 2020*.

3. **Northcutt, C. G., Jiang, L., & Chuang, I. L. (2019/2021).** "Confident Learning: Estimating Uncertainty in Dataset Labels." arXiv:1911.00068.

4. **Swayamdipta, S., et al. (2020).** "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics." *EMNLP 2020*.

5. **Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017).** "Understanding Deep Learning Requires Rethinking Generalization." *ICLR 2017*.

6. **Bartlett, P. L., Foster, D. J., & Telgarsky, M. J. (2017).** "Spectrally-Normalized Margin Bounds for Neural Networks." *NeurIPS 2017*.

7. **Han, B., et al. (2018).** "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels." *NeurIPS 2018*.

8. **Arazo, E., et al. (2019).** "Unsupervised Label Noise Modeling and Loss Correction." *ICML 2019*.

9. **Chen, P., et al. (2019).** "Understanding and Utilizing Deep Neural Networks Trained with Noisy Labels (INCV)." *ICML 2019*.

10. **Nakkiran, P., et al. (2020).** "Deep Double Descent: Where Bigger Models and More Data Hurt." *ICLR 2020*.
