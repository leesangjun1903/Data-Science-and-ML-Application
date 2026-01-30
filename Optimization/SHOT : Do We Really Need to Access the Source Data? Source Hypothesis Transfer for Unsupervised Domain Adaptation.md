
# Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation

## I. 핵심 요약

**SHOT (Source HypOthesis Transfer)** 논문은 2020년 발표된 지도 받지 않은 도메인 적응(UDA) 분야의 획기적인 연구로, 소스 데이터에 접근하지 않고도 사전 학습된 소스 모델만을 사용하여 타겟 도메인에 효과적으로 적응하는 방법을 제시합니다. 이는 **프라이버시 보호, 데이터 전송 효율성, 실무적 실용성**이라는 세 가지 차원에서 전통적 도메인 적응 방법의 근본적 한계를 해결합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

SHOT의 핵심 기여는 다음과 같습니다:
- 출처 분류기(가설, Hypothesis)를 **고정**하고 타겟 특화 인코더만 학습
- **정보 최대화(Information Maximization)**와 **자가 감독형 의사 라벨링(Self-Supervised Pseudo-Labeling)**의 통합
- 폐쇄집합, 부분집합, 개방집합 세 가지 도메인 적응 시나리오 동시 지원

***

## II. 연구 문제 및 실무적 동기

### 문제 정의

기존 도메인 적응 방법들은 소스 데이터와 타겟 데이터에 동시에 접근해야 한다는 비현실적 가정을 기반으로 합니다. 그러나 실무 환경에서는:

1. **프라이버시 제약**: 병원 환자 데이터, 개인 휴대폰 데이터 등 민감정보 공유 불가능
2. **계산/저장 효율성**: 소스 데이터셋(GB 규모) vs 모델(MB 규모)의 대비적 크기 차이

| 데이터셋 | Digits (MB) | VisDA-C (MB) |
|---------|---------|----------|
| 소스 데이터셋 | 33.2 | 7,884.8 |
| 소스 모델 | 0.9 | 172.6 |

**결론**: 학습된 모델 전달이 원본 데이터 전달보다 **36배~46배 더 효율적**입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

***

## III. 제안 방법론: SHOT의 상세 분석

### 3.1 기본 프레임워크

SHOT은 심층신경망을 두 개의 모듈로 분리합니다:

$$f_s(x) = h_s(g_s(x))$$

여기서:
- $$g_s: \mathcal{X}_s \to \mathbb{R}^d$$: 특성 인코더
- $$h_s: \mathbb{R}^d \to \mathbb{R}^K$$: 분류기(가설)
- $$d$$: 특성 차원, $$K$$: 클래스 수

타겟 도메인 적응 시:
- $$h_t = h_s$$ (분류기는 고정)
- $$g_t$$: 학습 대상 (타겟 인코더)

### 3.2 소스 모델 생성 (Label Smoothing 적용)

표준 교차엔트로피 손실에 라벨 스무딩을 적용하여 판별성 향상:

$$L^{ls}_{src}(f_s; \mathcal{X}_s, \mathcal{Y}_s) = -\mathbb{E}_{(x^s, y^s) \in \mathcal{X}_s \times \mathcal{Y}_s} \sum_{k=1}^{K} q^{ls}_k \log \delta_k(f_s(x^s))$$

여기서 스무딩된 라벨:

$$q^{ls}_k = (1-\alpha)q_k + \alpha/K$$

($$\alpha = 0.1$$로 경험적 설정) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

### 3.3 정보 최대화 기반 적응 (SHOT-IM)

원리: 만약 타겟 표현이 소스와 유사하면, 소스 분류기의 출력은 원-핫 인코딩처럼 개별적으로 확실하고 전체적으로 다양해야 함.

**엔트로피 손실** (개별 확실성):

$$L_{ent}(f_t; \mathcal{X}_t) = -\mathbb{E}_{x_t \in \mathcal{X}_t} \sum_{k=1}^{K} \delta_k(f_t(x_t)) \log \delta_k(f_t(x_t))$$

**다양성 손실** (전역 구조):

$$L_{div}(f_t; \mathcal{X}_t) = \sum_{k=1}^{K} \hat{p}_k \log \hat{p}_k = D_{KL}(\hat{p}, \frac{1}{K}\mathbf{1}_K) - \log K$$

여기서 $$\hat{p} = \mathbb{E}_{x_t \in \mathcal{X}_t}[\delta(f_t(x_t))]$$는 평균 출력 임베딩입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

**중요성**: 조건부 엔트로피 최소화 대신 정보 최대화를 사용함으로써 모든 샘플이 동일 클래스로 붕괴되는 **모델 붕괴 문제 회피**.

### 3.4 자가 감독형 의사 라벨링 전략

**핵심 문제**: 의사 라벨 기반 학습은 도메인 시프트로 인한 노이즈에 취약.

**해결책**: 타겟 도메인 내 클래스 원형(Prototype)을 먼저 추출하여 깨끗한 의사 라벨 생성.

**단계 1**: 가중 K-평균으로 클래스별 원형 계산:

$$c_k^{(0)} = \frac{\sum_{x_t \in \mathcal{X}_t} \delta_k(\hat{f}_t(x_t)) \hat{g}_t(x_t)}{\sum_{x_t \in \mathcal{X}_t} \delta_k(\hat{f}_t(x_t))}$$

**단계 2**: 가장 가까운 원형 기반 의사 라벨:

$$\hat{y}_t = \arg \min_k D_f(\hat{g}_t(x_t), c_k^{(0)})$$

($$D_f$$: 코사인 거리)

**단계 3**: 업데이트된 원형으로 반복 정제 (1회 권장):

$$c_k^{(1)} = \frac{\sum_{x_t \in \mathcal{X}_t} \mathbb{1}[\hat{y}_t = k] \hat{g}_t(x_t)}{\sum_{x_t \in \mathcal{X}_t} \mathbb{1}[\hat{y}_t = k]}$$

($$\mathbb{1}[\cdot]$$: 지시함수) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

### 3.5 전체 학습 목적함수

의사 라벨 $$\hat{y}_t$$가 생성된 후:

$$L(g_t) = L_{ent}(h_t \circ g_t; \mathcal{X}_t) + L_{div}(h_t \circ g_t; \mathcal{X}_t) - \beta \mathbb{E}_{(x_t, \hat{y}_t) \in \mathcal{X}_t \times \hat{\mathcal{Y}}_t} \sum_{k=1}^{K} \mathbb{1}[k = \hat{y}_t] \log \delta_k(h_t(g_t(x_t)))$$

여기서:
- 첫 두 항: 소스 가설에 맞춤
- 세 번째 항: 자가 감독 신호
- $$\beta > 0$$: 균형 파라미터 (폐쇄집합: 0.3, Digits: 0.1) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

### 3.6 네트워크 아키텍처 설계 개선

**배치 정규화 (BN)**:
- 소스와 타겟 도메인이 공유된 초저차 통계(평균과 분산)를 가정
- 내부 데이터 공변량 시프트 감소
- $$\text{BN}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

**가중치 정규화 (WN)**:
- 최종 FC 계층의 가중치 벡터 $$w_k$$ 정규화
- 소프트맥스 출력 내 거리 감지에서 가중치 노름의 중요성을 반영
- $$w_k / \|w_k\|_2$$

**라벨 스무딩 (LS)**:
- 모델이 클래스 내 타이트하고 균등하게 분리된 클러스터 학습 유도
- 의사 라벨의 노이즈 영향 완화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

**실험적 검증** (Office-Home Ar→Cl):
- 단독 BN: +4.2% 개선
- BN + WN + LS: +6.0% 개선 (최적 조합) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

***

## IV. 성능 향상 및 일반화 분석

### 4.1 벤치마크 성능 (2020년 기준 SOTA)

#### Digits 데이터셋 (폐쇄집합 DA)

| 방법 | S→M | U→M | M→U | 평균 |
|-----|-----|-----|-----|-----|
| CDAN+E (2018) | 89.2 | 98.0 | 95.6 | 94.3 |
| SWD (2019) | 98.9 | 97.1 | 98.1 | **98.0** |
| SHOT-IM | 99.0 | 97.6 | 97.7 | 98.2 |
| **SHOT (전체)** | 98.9 | **98.0** | 97.9 | **98.3** ✓ |

**결론**: SHOT이 기존 SOTA 달성. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

#### Office-Home 데이터셋 (중규모)

평균 정확도 향상:
- 기존 SOTA (TransNorm): 67.6%
- **SHOT**: 71.8% (+4.2%p 향상)
- 12개 작업 중 10개에서 최고 성능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

#### VisDA-C 데이터셋 (대규모)

| 방법 | 평균 정확도 |
|-----|-----------|
| SWD (2019) | 76.4% |
| SHOT-IM | 80.4% |
| **SHOT** | **82.9%** ✓ |

**개선도**: +6.5%p [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

### 4.2 부분집합 도메인 적응 (PDA)

타겟 도메인이 소스 도메인의 부분 클래스만 포함하는 경우:

Office-Home PDA에서:
- 기존 SOTA (SAFN): 71.8%
- **SHOT**: **79.3%** (+7.5%p) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

**핵심 변경**: $$L_{div}$$ 항 제거 (불균형 클래스 분포 반영)

### 4.3 개방집합 도메인 적응 (ODA)

타겟에 알려지지 않은 클래스 포함:

Office-Home ODA에서:
- 기존 SOTA (STA): 69.5%
- **SHOT**: **72.8%** (+3.3%p) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

**핵심 변경**: 불확실성 기반 미지 샘플 거절 (엔트로피 임계값)

### 4.4 다중 소스 및 다중 타겟 적응

Office-Caltech 다중소스 (R→A/C/D/W):

| 방법 | 평균 |
|-----|------|
| M3SDA-β (2019) | 96.4% |
| **SHOT-IM** | 97.6% |
| **SHOT** | **97.7%** ✓ |

**전략**: 각 소스 모델로부터 적응된 예측 점수 결합 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

### 4.5 절제 연구 (Ablation Study)

Office-Home에서:

| 구성 | Digits | Office-Home | VisDA-C |
|-----|--------|-------------|---------|
| 소스 모델만 | 79.3 | 60.2 | 46.6 |
| +Naive PL | 83.0 | 64.1 | 76.6 |
| +자가감독 PL | 87.6 | 68.9 | 80.7 |
| + $$L_{ent}$$ | 83.5 | 55.5 | 63.3 |
| + $$L_{ent}$$ + $$L_{div}$$ | 87.3 | 70.5 | 80.4 |
| **전체 SHOT** | **88.6** | **71.8** | **82.9** |

**핵심 통찰**:
1. 자가감독 PL이 순진한 PL보다 **5-6%p 우월**
2. 다양성 손실 $$L_{div}$$가 모델 붕괴 방지에 **필수적**
3. 세 요소의 상호보완성 입증 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

***

## V. 모델 일반화 성능 향상의 메커니즘

### 5.1 암묵적 특성 정렬 (Implicit Feature Alignment)

SHOT의 핵심 가정:

$$p(g_t(x_t)) \approx p(g_s(x_s))$$

즉, 고정된 소스 분류기 $$h_s$$에 맞춰 타겟 특성을 정렬하면 자동으로 소스 특성 분포와 정렬됨.

**수학적 정당화**:
- 정보 최대화는 소프트맥스 출력 $$\delta(f_t(x_t))$$를 특정 분포로 유도
- 동일 분류기를 공유하므로, 출력 분포 일치 = 특성 분포 정렬 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

### 5.2 도메인 특화 인코더의 역할

전통적 방식: 소스-타겟 공유 인코더 + 도메인 분류기 대적 학습
- 문제: 소스 데이터 필수

SHOT: 타겟 특화 인코더 개별 학습
- 장점: 소스 데이터 불필요, 도메인 간 유연한 적응
- 검증: ADDA와 DIRT-T도 동일 아키텍처로 더 우수한 성능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

### 5.3 자가 감독 신호의 역할

**가설 1 (순진한 의사 라벨의 한계)**:
- 타겟 샘플: 예측 [0.4, 0.3, 0.1, 0.1, 0.1] → 의사 라벨  강제 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)
- 실제 라벨: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)
- 결과: 잘못된 정렬 위험

**해결책 (자가 감독 PL)**:
- 타겟 도메인 내 클래스 원형 추출 (소스 데이터 불필요)
- 자신의 도메인 구조를 통해 의사 라벨 정제
- 노이즈 감소 효과: naive PL (64.1%) → self-supervised PL (68.9%) +4.8%p [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

### 5.4 정보 이론적 관점

정보 최대화:
$$L_{ent} + L_{div}$$
는 다음을 동시에 달성:

1. **개별 확실성 (Low Entropy)**: 특정 클래스 확신도 증대
2. **전역 다양성 (Maximum Entropy의 다양화 항)**: 클래스 분포 균형
3. **모델 붕괴 방지**: 모든 샘플이 한 클래스로 붕괴되는 현상 차단

비교: 조건부 엔트로피만 사용 → 붕괴 위험 증가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

***

## VI. 한계 및 실무적 제약

### 6.1 이론적 한계

1. **가설의 유효성**: 소스 분류기를 타겟에 그대로 적용 가능하다는 가정의 성립 조건 미명시
   - 극단적 도메인 차이에서 성능 저하 가능성
   
2. **의사 라벨 품질**: K-평균 기반 원형 추출이 초기 단계에서 노이즈 가능성

3. **초저차 통계 가정**: 배치 정규화의 평균/분산 공유 가정이 모든 도메인 쌍에서 성립하지 않을 수 있음

### 6.2 실무적 제약

| 제약 | 영향 | 대안 |
|------|------|------|
| 소규모 타겟 도메인 | 의사 라벨 부정확 | 더 강건한 클러스터링 필요 |
| 극단적 도메인 시프트 | $$L_{div}$$ 효과 감소 | 중간 도메인 활용 고려 |
| 이미지 외 모달리티 | 설계 재검토 필요 | 도메인 특화 전처리 |

### 6.3 계산 복잡도

- 소스 모델 생성: 표준 지도학습 (소스 데이터 1회 사용)
- 타겟 적응: 
  - 의사 라벨 생성: O(n × K) (K-평균)
  - 인코더 학습: 표준 SGD
  - **총 비용**: 기존 도메인 적응과 유사하나 소스 데이터 I/O 제거로 메모리 효율 + [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

***

## VII. 2020년 이후 관련 최신 연구 비교 분석

### 7.1 소스-프리 도메인 적응 (Source-Free DA) 의 진화

#### 1단계: 기초 방법 (2020-2021)
| 연구 | 핵심 기여 | SHOT와의 차이 |
|------|----------|-------------|
| **SHOT** (2020) | 정보최대화 + 자가감독 PL | **기준선** |
| **SHOT++** (2020) | 라벨 이전 전략 추가 | 신뢰도 기반 2분할 |
| **A2-Net** (2021) | 대적 학습 활용 | GAN 기반 재구성 |

#### 2단계: 대조학습 통합 (2021-2023)
| 연구 | 핵심 | 성능 향상 |
|------|-----|---------|
| **DaC (2022)** | 적응형 대조학습 + 신뢰도 분할 | Office-Home: 73.3% |
| **SDA-FAS (2022)** | 대조 도메인 정렬 | 이론적 정당화 |
| **SFDA-MC (2023)** | 다중 뷰 대조학습 | 메모리 효율 개선 |

**SHOT 대비 개선**: +1-2%p (특정 작업), 계산 효율성 트레이드오프 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10678654/)

#### 3단계: 기초 모델 통합 (2023-2024)
| 연구 | 특징 | 혁신점 |
|------|-----|-------|
| **ViT 기반 SHOT** (2023) | Vision Transformer 백본 | 자기주의 메커니즘의 강건성 |
| **SCoDA (2024)** | 자가감독 사전학습 + EMA 교사 | LS-기반 초기화 회피 |
| **Unified SFDA** (2024) | 폐쇄/부분/개방 동시 지원 | 캘리브레이션 통합 |

**트렌드**: SHOT의 기본 원리는 유지하되, 자가감독학습과 대형 기초 모델 활용 확대 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10611075/)

### 7.2 자동 도메인 적응 벤치마크 성능 비교 (2024년)

Office-Home 폐쇄집합 DA (평균 정확도):

```
SHOT (2020):           71.8%
├─ DaC (2022):         73.3% (+1.5%p)
├─ SFDA-MC (2023):     74.1% (+2.3%p)
├─ SCoDA (2024):       74.9% (+3.1%p)
└─ Unified SFDA:       75.2% (+3.4%p)
```

**해석**: SHOT 이후 점진적 개선 추세지만, 근본적 혁신 없음. 기존 원리의 정교화 수준. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11015964/)

### 7.3 Vision Transformer의 도메인 일반화 우월성 (2023-2024)

**핵심 발견** (Alijani et al. 2024):

1. **자기주의의 강건성**: 자기주의 메커니즘이 CNN의 국소적 특성 학습보다 전역 구조 포착에 우월
   - CNN: 텍스처에 의존 (도메인 시프트 취약)
   - ViT: 형태 기반 학습 (도메인 불변성 강화)

2. **SHOT + ViT 성능**:
   - ResNet-50: 71.8% (Office-Home)
   - ViT-Base: 73.5% (+1.7%p)
   - 이유: 동적 가중치 계산으로 유연한 특성 정렬 [arxiv](https://arxiv.org/abs/2404.04452)

3. **배치 정규화 분석의 재평가**:
   - CNN에서 BN은 도메인 의존성 증가 (매개변수 양수)
   - ViT 환경에서 레이어 정규화(LayerNorm)는 도메인 불변성 강화
   - **권장**: ViT 사용 시 배치 정규화 최소화 [arxiv](https://arxiv.org/html/2508.15452v2)

### 7.4 테스트 타임 적응 (TTA)의 발전

관련 분야: 소스-프리 설정과 문제 정의 유사

| 방법 | 엔트로피 최소화 + | 핵심 개선 |
|------|------------|---------|
| **TENT** (2021) | 표준 배치통계 | 배치 정규화 통계만 업데이트 |
| **DPLOT** (2024) | 도메인 특화 블록 선택 | 엔트로피 최소화 대상 선택화 |
| **TTN** (2023) | 도메인 인식 배치정규화 | 가중 통계 조화 |

**SHOT과의 연계**: 정보 최대화의 엔트로피 항과 TTA의 엔트로피 최소화는 동일 원리. SHOT의 타겟 적응 단계는 비정상적 배치 스트림 대신 안정적 무작위 배치 가정. [arxiv](https://arxiv.org/pdf/2404.10966.pdf)

### 7.5 최신 소스-프리 도메인 적응 한계 분석 (2024년)

**Google Deep Learning 팀 (Triantafillou & Boudiaf, 2025)**:

> SFDA 연구가 이미지 분류의 단순 분포 시프트에만 국한. 다음 도전 과제:
> 1. 비전 언어 모델(VLM) 기반 개방집합 문제
> 2. 미세 조정 효율성 개선
> 3. 도메인 이동 순차성(Continual DA) 처리

**SHOT의 한계**:
- VLM (CLIP) 시대에 가설 고정 전략의 적용 가능성 의문
- 기초 모델의 이미 우수한 일반화로 SHOT의 추가 이득 한계 [research](https://research.google/blog/in-search-of-a-generalizable-method-for-source-free-domain-adaptation/)

### 7.6 대규모 오픈 소스 도메인 적응 조사 (2024)

Fang et al. (PMC 게시, 2024)의 종합 조사에서 SHOT의 위상:

**카테고리별 방법론 분류**:
1. **데이터 생성 기반** (Surrogate Source): SHOT 미포함 (모델 기반)
2. **통계 정렬** (BN/IN): SHOT 아키텍처 설계로 간접 포함
3. **자가 지도 학습**: SHOT의 자가감독 PL이 **원형** (prototype)
4. **대조학습**: 후속 DaC 등에서 통합
5. **불확실성 기반**: SHOT의 개방집합 확장에서 핵심

**결론**: SHOT은 방법론 타입상 "특성 정렬 + 자가감독" 카테고리의 **기초 연구**이며, 이후 모든 SFDA 방법이 이를 기반으로 확장. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11015964/)

***

## VIII. 핵심 통찰 및 일반화 성능 향상의 원리

### 8.1 왜 SHOT이 소스 데이터 없이도 작동하는가?

세 가지 상호보완적 원리:

1. **분류기의 도메인 불변성**
   - 소스 분류기 $$h_s$$는 특성 공간 구조를 인코딩
   - 충분히 학습된 경우, 타겟 특성도 동일 구조 따름

2. **정보 최대화의 정렬 효과**
   - $$L_{ent}$$: 각 샘플을 특정 클래스에 확실히 배정
   - $$L_{div}$$: 클래스 간 균형된 분포 유지
   - 결과: 소스 분포와 유사한 타겟 분포 자동 형성

3. **자가 감독 신호의 정제**
   - 의사 라벨의 노이즈를 타겟 도메인 구조로 정제
   - 외부 감시(소스)가 아닌 내재적 구조 활용
   - 점진적 신뢰도 증가 메커니즘

### 8.2 일반화 성능 개선의 세 단계

| 단계 | 메커니즘 | 성능 향상 |
|------|---------|---------|
| 1️⃣ 정보최대화 | 의사 라벨 기반 임의 정렬 | ±3~5%p (불안정) |
| 2️⃣ 자가감독 정제 | 원형 기반 라벨 정제 | +3~5%p (안정) |
| 3️⃣ 아키텍처 설계 | BN/WN/LS 최적화 | +1~2%p (추가 이득) |

**누적 효과**: Office-Home에서 소스 모델 (60.2%) → SHOT (71.8%) = +11.6%p [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

### 8.3 도메인 차이와 성능의 관계

실험적 관찰:

- **근접 도메인** (MNIST↔USPS): SHOT ≈ 감독 학습 (거의 완벽한 적응)
- **중간 도메인** (Office-Home): SHOT이 최고 우수 (4%p 개선)
- **원거리 도메인** (Syn→Real, VisDA): SHOT 효과 더욱 뚜렷 (6~8%p 개선)

**해석**: 도메인이 멀수록 소스 분류기만으로 충분한 신호가 되고, 정보 최대화의 정렬이 더 명확 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0ffd67b2-fda8-4bc2-867d-b4957cd31d8b/2002.08546v6.pdf)

***

## IX. 앞으로의 연구 방향 및 고려 사항

### 9.1 기술적 개선 필요 영역

1. **약한 가설 설정**: 
   - 현재: 소스 분류기가 완벽하다고 가정
   - 개선: 분류기 신뢰도 가중치 도입 (Bayesian 접근)

2. **비전 언어 모델 (VLM) 통합**:
   - CLIP, DINO-V2 같은 기초 모델의 높은 일반화로 SHOT의 추가 이득 제한
   - 해결: 시맨틱 정보와 시각 특성의 정렬 강화

3. **연속 도메인 적응 (Continual DA)**:
   - 현재: 정적 타겟 도메인 가정
   - 개선 필요: 시간에 따라 변하는 타겟에 대한 점진적 학습

4. **범주 불균형 처리**:
   - 현재: 균등 클래스 분포 가정 ($$L_{div}$$)
   - 개선: 타겟 클래스 분포의 자동 추정 및 가중치 조정

### 9.2 실무 적용 시 고려사항

| 시나리오 | SHOT 적용 가능성 | 권장 사항 |
|---------|------------|---------|
| 의료 영상 (도메인 차이 큼) | ✅ 높음 | SHOT 또는 SHOT++ 권장 |
| 자율주행 (실시간) | ⚠️ 중간 | 경량 모델 + 배치 최소화 |
| 음성 데이터 (시계열) | ❌ 낮음 | 특화 설계 필요 |
| 텍스트 도메인 적응 | ⚠️ 중간 | UDALM 등 언어 특화 필요 |

### 9.3 향후 연구 방향

#### 단기 (1-2년)
- ✅ CLIP/DINO 기초 모델 기반 SHOT 변형
- ✅ 멀티모달 도메인 적응 (이미지+텍스트)
- ✅ 프라이버시 보존 연합학습(Federated DA) 통합

#### 중장기 (2-5년)
- ✅ 생성형 기초 모델 (확산 모델) 기반 도메인 적응
- ✅ 약한 감시 신호 (Weak Supervision)와 결합
- ✅ 물리 시뮬레이션 (Sim-to-Real) 특화 방법론

#### 근본적 도전
- ❓ 이론적 분석: SHOT의 수렴성 보장 조건
- ❓ 도메인 전이 가능성 측정: 사전에 SHOT 성공 가능성 판단
- ❓ 적응 안정성: 점진적 도메인 드리프트에서의 일관성 보장

***

## X. 결론

**SHOT은 2020년 출판 이후 5년이 경과한 현재도 소스-프리 도메인 적응의 기초 이론으로 위상을 유지하고 있습니다.**

### 주요 성과:
1. **패러다임 전환**: 소스 데이터 필요성 제거 (프라이버시 혁신)
2. **방법론의 단순성 + 강력함**: 정보최대화 + 자가감독 PL의 우아한 결합
3. **광범위한 적용 가능성**: 폐쇄/부분/개방집합 동시 지원
4. **후속 연구의 견고한 기초**: 2000+ 인용, DaC·SFDA-MC·SCoDA 등 수십 개 확장

### 한계:
1. **이론적 깊이 부족**: 가설 유효성의 수학적 보장 없음
2. **극단적 도메인 시프트**: 한계 성능 불명확
3. **기초 모델 시대의 필요성 의문**: VLM의 높은 일반화로 추가 이득 제한

### 최종 평가:
SHOT은 **실무적 가치**와 **학술적 영향력**에서 모두 뛰어난 연구입니다. 프라이버시 제약이 있는 현실 세계에서 도메인 적응을 가능하게 한 점, 그리고 이후 수십 개의 개선 방법론의 기반을 제공한 점은 컴퓨터 비전 분야의 중요한 마일스톤입니다. 다만 기초 모델 시대에는 적응의 필요성 자체가 감소할 수 있으므로, 향후 연구는 **약한 감시, 생성형 적응, 연속적 환경 변화** 등 새로운 과제로 확장되어야 합니다.

***

## 참고문헌

<span style="display:none">[^1_100][^1_101][^1_102][^1_103][^1_104][^1_105][^1_106][^1_107][^1_108][^1_109][^1_110][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93][^1_94][^1_95][^1_96][^1_97][^1_98][^1_99]</span>

<div align="center">⁂</div>

[^1_1]: 2002.08546v6.pdf

[^1_2]: https://ieeexplore.ieee.org/document/10678654/

[^1_3]: https://arxiv.org/abs/2211.06612

[^1_4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10611075/

[^1_5]: https://arxiv.org/pdf/2509.09935.pdf

[^1_6]: https://arxiv.org/abs/2404.04452

[^1_7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11015964/

[^1_8]: https://pubmed.ncbi.nlm.nih.gov/37896503/

[^1_9]: https://arxiv.org/html/2508.15452v2

[^1_10]: https://arxiv.org/pdf/2404.10966.pdf

[^1_11]: https://arxiv.org/html/2404.10966v1

[^1_12]: https://research.google/blog/in-search-of-a-generalizable-method-for-source-free-domain-adaptation/

[^1_13]: https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Domain-Specific_Block_Selection_and_Paired-View_Pseudo-Labeling_for_Online_Test-Time_Adaptation_CVPR_2024_paper.html

[^1_14]: https://arxiv.org/pdf/2404.04452.pdf

[^1_15]: http://proceedings.mlr.press/v119/liang20a/liang20a.pdf

[^1_16]: https://jurnal.stikeskesosi.ac.id/index.php/JRIKUF/article/view/422

[^1_17]: https://onlinelibrary.wiley.com/doi/10.5694/mja2.52245

[^1_18]: https://invergejournals.com/index.php/ijss/article/view/99

[^1_19]: https://bioone.org/journals/journal-of-coastal-research/volume-113/issue-sp1/JCR-SI113-026.1/Investigating-the-Early-Stages-of-a-Dune-Restoration-in-Front/10.2112/JCR-SI113-026.1.full

[^1_20]: https://doi.apa.org/doi/10.1037/pst0000545

[^1_21]: https://ejournal.insuriponorogo.ac.id/index.php/basica/article/view/6334

[^1_22]: https://www.nature.com/articles/s44298-023-00007-z

[^1_23]: https://www.semanticscholar.org/paper/5c5ab276b00c1f19fbb0a3d2c38d532becac9442

[^1_24]: https://arxiv.org/abs/2310.04438

[^1_25]: https://arxiv.org/pdf/2110.12024.pdf

[^1_26]: https://arxiv.org/pdf/2309.02211.pdf

[^1_27]: http://arxiv.org/pdf/1705.05498.pdf

[^1_28]: https://www.aclweb.org/anthology/2021.naacl-main.203.pdf

[^1_29]: https://arxiv.org/pdf/2208.07422.pdf

[^1_30]: https://www.mdpi.com/1099-4300/27/4/426

[^1_31]: http://arxiv.org/pdf/1910.12417.pdf

[^1_32]: http://arxiv.org/pdf/2303.02302.pdf

[^1_33]: https://arxiv.org/html/2509.20587v1

[^1_34]: https://arxiv.org/html/2403.07601v4

[^1_35]: https://pubmed.ncbi.nlm.nih.gov/34383644/

[^1_36]: https://arxiv.org/html/2502.21022v1

[^1_37]: https://arxiv.org/html/2601.17408v1

[^1_38]: https://arxiv.org/html/2402.14966v1

[^1_39]: https://arxiv.org/html/2502.21022v3

[^1_40]: https://openaccess.thecvf.com/content/ICCV2021/papers/Xia_Adaptive_Adversarial_Network_for_Source-Free_Domain_Adaptation_ICCV_2021_paper.pdf

[^1_41]: https://arxiv.org/pdf/2002.08546.pdf

[^1_42]: https://arxiv.org/html/2404.14704v1

[^1_43]: https://arxiv.org/abs/2510.01559

[^1_44]: https://arxiv.org/abs/2305.19694

[^1_45]: https://arxiv.org/html/2402.14976v1

[^1_46]: https://arxiv.org/abs/2511.19147

[^1_47]: https://arxiv.org/abs/2002.08546

[^1_48]: https://www.sciencedirect.com/science/article/abs/pii/S0925231223010445

[^1_49]: https://www.sciencedirect.com/topics/computer-science/unsupervised-domain-adaptation

[^1_50]: https://icml.cc/virtual/2025/poster/44848

[^1_51]: https://proceedings.iclr.cc/paper_files/paper/2025/file/e85454a113e8b41e017c81875ae68d47-Paper-Conference.pdf

[^1_52]: https://openaccess.thecvf.com/content_cvpr_2014/papers/Patricia_Learning_to_Learn_2014_CVPR_paper.pdf

[^1_53]: https://www.sciencedirect.com/science/article/abs/pii/S0893608024003423

[^1_54]: https://www.baeldung.com/cs/transfer-learning-vs-domain-adaptation

[^1_55]: https://openreview.net/pdf/c39a904a5845ff8035f79af1cf52190094214580.pdf

[^1_56]: https://openaccess.thecvf.com/content/CVPR2024/papers/Mitsuzumi_Understanding_and_Improving_Source-free_Domain_Adaptation_from_a_Theoretical_Perspective_CVPR_2024_paper.pdf

[^1_57]: https://arxiv.org/abs/1812.11806

[^1_58]: https://jamanetwork.com/journals/jamaophthalmology/fullarticle/2814750

[^1_59]: https://link.springer.com/10.1007/s11063-024-11621-0

[^1_60]: https://ieeexplore.ieee.org/document/10645355/

[^1_61]: https://arxiv.org/abs/2209.09076

[^1_62]: https://www.cambridge.org/core/product/identifier/S2056472424003302/type/journal_article

[^1_63]: https://dl.acm.org/doi/10.1145/3687273.3687297

[^1_64]: https://onepetro.org/ARMAUSRMS/proceedings/ARMA24/ARMA24/D031S034R001/549545

[^1_65]: https://dl.acm.org/doi/10.1145/3589948

[^1_66]: https://arxiv.org/abs/2303.15826

[^1_67]: https://arxiv.org/pdf/2110.00165.pdf

[^1_68]: https://aclanthology.org/2023.acl-long.92.pdf

[^1_69]: http://arxiv.org/pdf/1909.13776.pdf

[^1_70]: http://arxiv.org/pdf/2309.03999.pdf

[^1_71]: http://arxiv.org/pdf/2106.09890.pdf

[^1_72]: https://arxiv.org/pdf/2210.09486.pdf

[^1_73]: https://arxiv.org/pdf/1908.01342.pdf

[^1_74]: http://arxiv.org/pdf/2403.10834.pdf

[^1_75]: https://arxiv.org/pdf/2401.01042.pdf

[^1_76]: https://arxiv.org/html/2509.09935v1

[^1_77]: https://openaccess.thecvf.com/content/CVPR2023/papers/VS_Instance_Relation_Graph_Guided_Source-Free_Domain_Adaptive_Object_Detection_CVPR_2023_paper.pdf

[^1_78]: https://arxiv.org/pdf/2508.04538.pdf

[^1_79]: https://arxiv.org/abs/2507.03321

[^1_80]: https://arxiv.org/abs/2308.02746

[^1_81]: https://arxiv.org/html/2503.23220v1

[^1_82]: https://arxiv.org/html/2510.26826v1

[^1_83]: https://www.emergentmind.com/topics/domain-adaptive-self-supervised-pretraining

[^1_84]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720506.pdf

[^1_85]: https://www.sciencedirect.com/science/article/abs/pii/S0893608022002672

[^1_86]: https://neurips.cc/virtual/2025/poster/118848

[^1_87]: https://www.sciencedirect.com/science/article/pii/S0951832024003685

[^1_88]: https://www.sciencedirect.com/science/article/abs/pii/S0167865524003209

[^1_89]: https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2023EA003197

[^1_90]: https://openreview.net/forum?id=iulMde3dP1

[^1_91]: https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Domain-Specific_Block_Selection_and_Paired-View_Pseudo-Labeling_for_Online_Test-Time_Adaptation_CVPR_2024_paper.pdf

[^1_92]: https://cvpr.thecvf.com/virtual/2024/poster/30341

[^1_93]: https://liner.com/review/divide-and-contrast-sourcefree-domain-adaptation-via-adaptive-contrastive-learning

[^1_94]: https://papers.neurips.cc/paper_files/paper/2022/file/28e9eff897f98372409b40ae1ed3ea4c-Paper-Conference.pdf

[^1_95]: https://www.mdpi.com/1424-8220/23/20/8409

[^1_96]: https://www.nature.com/articles/s41598-023-48250-x

[^1_97]: https://www.semanticscholar.org/paper/6e1bb490ae54b42f13d14d69b2012edda4664949

[^1_98]: https://link.springer.com/10.1007/978-3-030-86486-6_35

[^1_99]: https://ieeexplore.ieee.org/document/9512429/

[^1_100]: https://www.semanticscholar.org/paper/6623fcb3b05e70700b2926e703da4d0cd818b4c7

[^1_101]: https://ieeexplore.ieee.org/document/11010437/

[^1_102]: https://arxiv.org/abs/2404.12618

[^1_103]: https://www.semanticscholar.org/paper/3a9aa8c034fe5e2a4268dfff6d8eab9cc82f8865

[^1_104]: https://arxiv.org/abs/2410.15589

[^1_105]: http://arxiv.org/pdf/2312.12489.pdf

[^1_106]: https://arxiv.org/pdf/2305.14857.pdf

[^1_107]: http://arxiv.org/pdf/1805.08402.pdf

[^1_108]: https://arxiv.org/pdf/2012.07297.pdf

[^1_109]: https://www.aclweb.org/anthology/2021.naacl-main.402.pdf

[^1_110]: https://arxiv.org/pdf/2006.03806.pdf
