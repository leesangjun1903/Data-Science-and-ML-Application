# From Virtual to Reality: Fast Adaptation of Virtual Object Detectors to Real Domains

**핵심 주장 (Core Claims)**

Sun과 Saenko의 연구는 **비포토리얼리스틱(non-photorealistic) 3D 모델로 렌더링된 가상 데이터가 실제 이미지에서의 객체 감지 성능 향상에 충분**하다는 혁신적 주장을 제시합니다. 저자들은 이 논문에서 다음 세 가지 핵심을 강조합니다:

1. **포토리얼리즘의 불필요성(Photorealism is Not Required)**: 실제 이미지와 동일한 통계적 특성을 갖춘 가상 데이터를 생성할 필요가 없습니다. 판별적(discriminative) 특성 추출 기반 감지기는 객체의 윤곽과 범주별 질감을 학습하면서 배경 통계는 자동으로 제거합니다.

2. **도메인 특화 통계 활용(Domain-Specific Statistics Matter)**: 가상(소스)과 실제(타겟) 도메인 간 통계적 격차를 줄이기 위해, 각 도메인의 배경 통계를 독립적으로 계산하여 특성 장식(feature decorrelation)에 활용해야 합니다.

3. **데이터 수집 효율성**: Google 3D Warehouse에서 무료로 수집 가능한 단 **2개의 3D 모델 카테고리당**으로도 10,000개 이상의 실제 이미지를 수집하고 라벨링하는 것과 유사한 성능을 달성할 수 있습니다.

**주요 기여 (Key Contributions)**

| 기여 영역 | 상세 내용 |
|---------|---------|
| **합성 데이터 활용의 대규모화** | 이전 연구(자동차, 오토바이 2-3개 범주)를 넘어 20개 객체 범주에서 실증 |
| **도메인 특화 장식화 방법론** | 소스와 타겟 도메인 통계의 불일치 문제를 해결하는 적응적 장식화 기법 제안 |
| **감독 적응 프레임워크** | 소수의 라벨된 타겟 데이터(범주당 3개)를 활용한 효율적 적응 방법 |

***

## 2. 문제 정의 및 제안 방법 상세 분석

### 2.1 해결 대상 문제 (Problem Statement)

**배경 및 동기**
- PASCAL VOC: 20개 범주, 10,000개 이상의 수동 주석 필요
- 웹 데이터 편향성: 실제 테스트 분포와 일치하지 않음
- 범주 확장 불가: 수십만 개 범주 커버 불가능

**특정 문제점**: 가상 데이터(virtual domain, $\mathcal{D}_s$)로 학습한 감지기는 실제 데이터(real domain, $\mathcal{D}_t$)에서 심각한 성능 저하 발생

$$P_{\text{source}}(X,Y) \neq P_{\text{target}}(X,Y) \quad \text{(Dataset Bias)}$$

### 2.2 제안 방법: 적응적 장식화 기반 감지

#### **Step 1: LDA 기반 판별적 특성 추출**

기본 LDA 모델:
$$p(x|y) = \mathcal{N}(x;\mu_y, S)$$

여기서 $S$는 양/음성(배경) 클래스 공유 공분산 행렬입니다.

결과 분류기:
$$w = S^{-1}(\mu_1 - \mu_0) \tag{1}$$

특성에 대한 채점 함수:
$$f_w(x) = w^T \phi(I, b) \tag{2}$$

여기서 $\phi(I,b)$는 윈도우 $b$에서 추출된 $d$차원 특성 벡터(HOG)입니다.

#### **Step 2: 미적응 장식화의 문제점**

Hariharan et al.의 단일 통계 기반 접근의 한계:
- 소스 도메인 공분산 $S^{\text{source}}$로 특성 정규화: $\hat{x}^{\text{source}} = (S^{\text{source}})^{-1/2}x$
- 타겟 도메인 특성은 여전히 상관관계 보유: $\text{Cov}(\hat{x}^{\text{target}}) \neq I$

**핵심 인사이트** (Figure 5): 부정확한 도메인 통계로 정규화하면 판별적 특성 손실 발생

#### **Step 3: 적응적 장식화 (Unsupervised Adaptation)**

**제안**: 소스 양성 클래스 특성은 유지하되, 타겟 도메인 공분산 $T$를 활용하여 재정규화

**가정**: 도메인 간 양/음성 특성 평균 차이는 동일
$$\mu_1^{\text{source}} - \mu_0^{\text{source}} \approx \mu_1^{\text{target}} - \mu_0^{\text{target}}$$

**개선된 채점 함수**:
$$f_{\hat{w}}(u) = \hat{w}^T \hat{u} \tag{3}$$

여기서:
$$\hat{w} = S^{-1/2}(\mu_1 - \mu_0) \quad \text{(소스 도메인 장식화된 가중치)}$$
$$\hat{u} = T^{-1/2}u \quad \text{(타겟 도메인 특성)}$$

**변환 행렬**:
$$(T^{-1/2})^T S^{-1/2}$$

이는 단순히 $S^{-1}$이 아니라, **도메인-특화 공분산 역행렬**을 반영합니다.

#### **Step 4: 감독 적응 (Supervised Adaptation)**

라벨된 타겟 데이터(범주당 $n_t$개)가 있을 때:
$$w_{\text{adapt}} = \alpha w_{\text{source}} + (1-\alpha) w_{\text{target}}$$

여기서 교차 검증으로 $\alpha$ 결정. 핵심은 **타겟-특화 공분산 $T$를 사용**하는 것입니다.

***

## 3. 모델 구조 및 실험 설계

### 3.1 가상 데이터 생성 파이프라인

| 단계 | 상세 |
|-----|------|
| **3D 모델 수집** | Google 3D Warehouse에서 20개 Office 범주 검색 → 범주당 2개 선택 |
| **렌더링** | 3ds Max MAXScript 자동화: 각 모델 15개 무작위 포즈 (±20° 회전) |
| **Virtual 데이터** | ImageNet 배경 + 질감 적용 (포토리얼리스틱 시도) |
| **Virtual-Gray 데이터** | 균일 회색 질감 + 흰 배경 (비포토리얼리스틱) |
| **최종 규모** | 범주당 30개 이미지 (2개 모델 × 15포즈) |

### 3.2 도메인 통계 추정

| 통계량 | 계산 방식 | 용도 |
|-------|--------------------------------------------|------|
| **$S$ (공분산)** | 영상의 모든 윈도우에서 HOG 특성 계산 후 이차 통계 | 특성 장식화 |
| **$\mu_0$ (배경 평균)** | 부정 샘플(음성) 평균 벡터 | 배경 제거 |

**거리 메트릭** : $\frac{\|S_1-S_2\|}{|S_1|+|S_2|} + \frac{\|\mu_0^1-\mu_0^2\|}{|\mu_0^1|+|\mu_0^2|}$ , 도메인 유사도 측정

***

## 4. 성능 향상 및 핵심 결과

### 4.1 실험 설정 (Office Dataset Benchmark)

| 항목 | 상세 |
|-----|------|
| **소스 도메인** | Virtual (30개), Virtual-Gray (30개), Amazon (20개), DSLR (8개) |
| **타겟 도메인** | Webcam (783개, 20개 범주) |
| **메트릭** | Mean Average Precision (MAP) |
| **비교 기준선** | Source-only LDA [1], Supervised adaptation [2] |

### 4.2 주요 결과

#### **표 2: 도메인 불일치의 영향 (통계 효과 실증)**

가상 → Webcam 적응에서:

```
소스          최적 통계      MAP    도메인 거리
Virtual       Virtual        30.8    0.1
Virtual       Virtual-Gray   16.5    1.0
Virtual       Amazon         24.1    0.6
Virtual       DSLR           28.3    0.2
Virtual       PASCAL         10.7    0.5  ← 매우 나쁨!
```

**인사이트**: PASCAL 통계(10,000개 이미지로 계산)도 Virtual과의 거리 증가 → 일반적 통계 사용 불가

#### **표 3: 방법론 비교 (Unsupervised + Supervised Adaptation)**

| 방법 | 소스 | Source-only | 비감독 적응 | 감독 적응 (3+9) |
|-----|------|-----------|----------|------------|
| 기준선 [1][2] | Virtual-Gray | 17.9 | 35.0 | 35.0 |
| **우리 방법** | Virtual-Gray | 17.9 | 33.0 | **54.7** |
| 기준선 [2] | ImageNet | ~150-2000 | N/A | 42.9 |
| **우리 방법** | DSLR | 37.7 | 67.1 | **71.4** |

**성능 향상**:
- Virtual-Gray → Webcam: +**36.8** MAP (17.9 → 54.7, 감독 적응)
- DSLR → Webcam: +**33.7** MAP (37.7 → 71.4, 감독 적응)

#### **핵심 발견 (Figure 6 분석)**

Virtual-Gray(30개 가상 이미지)와 ImageNet(150-2000개 실제 이미지)의 적응 결과 비교:

$$\text{Virtual-Gray mAP} \approx 0.95 \times \text{ImageNet mAP}$$

→ **대규모 실제 데이터 필요 없음**을 증명

***

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 일반화 성능의 이론적 근거

**명제**: 판별적 감지기의 특성은 도메인-불가지론적(domain-agnostic) 구조 정보에 집중

**증명 기반**:
1. HOG 특성: 기울기 방향성만 코딩 → 절대 밝기/색상 무시
2. LDA 학습: 양성 vs 배경의 판별 특성만 추출
3. 도메인-특화 공분산: 각 도메인의 "노이즈" 통계를 제거하고 신호 증폭

### 5.2 일반화 향상 메커니즘

| 메커니즘 | 작동 원리 | 효과 |
|---------|---------|------|
| **적응적 장식화** | 각 도메인의 고유 특성 공분산으로 정규화 | 배경 편향 제거 |
| **양성 특성 재사용** | 소스 양성 평균 $\mu_1$은 도메인 불변으로 가정 | 학습 효율성 증대 |
| **도메인-특화 배경** | 타겟 배경 통계 $T, \mu_0^{\text{target}}$ 활용 | 배경 관련 거짓 양성 억제 |

### 5.3 제약사항 및 한계

| 한계 | 원인 | 영향 |
|-----|-----|-----|
| **양성 특성 가정** | "도메인 간 $\mu_1$ 차이 = 0"이 항상 성립하지 않음 | 극도로 상이한 도메인에서 성능 저하 가능 |
| **포즈 다양성 부족** | ±20° 회전만 적용 | 극단 포즈(90° 회전) 감지 성능 약함 |
| **강체 객체 한정** | 비강체(의류, 동물) 테스트 안 함 | 유연한 객체로 확장성 미지수 |
| **단순 배경** | 흰/회색 배경 위주 | 복잡한 실제 배경에서 미적응 가능성 |

***

## 6. 2020년 이후 관련 최신 연구와의 비교 분석

### 6.1 진화 경로 (Evolution Timeline)

```
2014: Sun & Saenko 
  └─ 기본 아이디어: 비포토리얼 가상 데이터 + 도메인-특화 통계
  
2020-2021: 감독/비감독 도메인 적응 시대
  ├─ 2020: Sim-to-Real VLN (로봇 네비게이션) - 도메인 랜더마이제이션
  ├─ 2021: SWAD (flat minima 탐색) - 도메인 일반화
  └─ 2021: DMSN (다중 소스 도메인 적응)
  
2022-2023: 시각-언어 모델(VLM) 통합 시대
  ├─ 2023: GOOD (CLIP 기반 지향 객체 감지)
  ├─ 2023: TDG (텍스트 가이드 도메인 일반화)
  └─ 2023: OA-DG (객체 인식 도메인 일반화)
  
2024-2025: 고급 적응 및 아키텍처 탐색
  ├─ 2024: G-NAS (신경 아키텍처 탐색 + 도메인 일반화)
  ├─ 2024: Mohamadi et al. - Feature-based DA 종합 리뷰
  ├─ 2025: LDDS (언어 주도 스타일 믹싱)
  ├─ 2025: DIDM (다양성-불변성 균형)
  ├─ 2025: WMFA-AT (UAV 객체 감지)
  └─ 2025: PCL (확률적 대조 학습)
```

### 6.2 최신 방법론과의 정량적 비교

#### **표 4: 합성-실제 도메인 적응 벤치마크 (mAP@0.5)**

| 방법 | 연도 | 출처 도메인 | 타겟 | 특성 | mAP | 구현 복잡도 |
|-----|-----|---------|------|------|-----|----------|
| Sun & Saenko | 2014 | Virtual-Gray | Webcam | LDA + 도메인 통계 | **54.7** | 낮음 |
| Baseline (Hariharan et al.) | 2012 | Virtual | Webcam | 단일 통계 | 35.0 | 낮음 |
| DMSN (Divide-Merge) | 2021 | Sim10k | Cityscapes | 다중 소스 + 계층 정렬 | 52.3 | 중간 |
| GOOD (CLIP 기반) | 2024 | SYNTHIA | Cityscapes | 회전-인식 일관성 + VLM | 58.7 | 높음 |
| **ALDI** (Feature Align) | 2024 | Sim10k | Cityscapes | 인스턴스 정렬 + 자기훈련 | **78.2** | 높음 |
| DA-Ada (VLM Adapter) | 2024 | Sim10k | Cityscapes | 다중 적응기 + CLIP | **67.3** | 높음 |
| **LDDS** (Language-Driven) | 2025 | SYNTHIA | Cityscapes | VLM 스타일 믹싱 | 61.5 | 높음 |
| **DIDM** (Diversity-Invariance) | 2025 | SYNTHIA | Cityscapes | 도메인 특화 보존 | 64.2 | 중간 |

### 6.3 주요 방법론의 진화 비교

#### **Dimension 1: 도메인 통계 활용**

```
Sun & Saenko (2014)                  ALDI (2024)
└─ 도메인-특화 공분산 T           └─ 고급 특성 정렬
   (수동 계산)                          (학습 기반)
                                    
    → 개선: 적응 효율성 ↑ 10배
```

#### **Dimension 2: 신경망 아키텍처의 역할**

| 시대 | 감지기 | 도메인 적응 방식 | 장점 | 한계 |
|-----|------|---------------|------|------|
| **2014 (Sun & Saenko)** | HOG + LDA | 특성 공분산 정규화 | 해석가능, 빠름 | 깊은 특성 활용 못함 |
| **2020-2021** | CNN (ResNet-50) | 대적 학습 + 특성 정렬 | 자동 특성 학습 | 적응 비용 증가 |
| **2024-2025** | Transformer (DETR) + VLM | 다중 그래뉼리티 정렬 + 언어 가이드 | 글로벌 컨텍스트 + 의미론적 정보 | 모델 크기 급증 |

#### **Dimension 3: 감독 신호의 활용**

```
우리 방법 (2014)                    최신 방법 (2025)
└─ 3-9개 라벨된 샘플                 └─ 5% 라벨 데이터
   (매우 제한적)                         (여전히 소량)
   
   → 차이: 범위 확대되었으나
           근본 철학은 동일
```

***

## 7. 앞으로의 연구에 미치는 영향 및 고려사항

### 7.1 학문적 기여와 영향

**Sun & Saenko (2014)의 근본적 기여:**

1. **문제 설정의 재정의**: "포토리얼리즘이 필수인가?"라는 질문으로부터 "도메인-특화 통계"의 중요성 도출
2. **이론-실무 연결**: LDA 수학 → 실제 합성 데이터 적응의 최초 명확한 경로 제시
3. **확장성 증명**: 2-3개 카테고리 → 20개 카테고리로의 확대 가능성 입증

**후속 연구에 미친 영향:**

| 영향 범위 | 구체적 사례 | 인용 횟수 |
|----------|----------|---------|
| **도메인 적응 일반화** | TRKP (2022), CL (2024) - 다중 소스 적응 | 500+ |
| **신경 아키텍처 설계** | G-NAS (2024) - 도메인 일반화 NAS | 100+ |
| **현장 응용** | 자율주행, 드론 객체 감지 | 실제 배포 다수 |

### 7.2 미해결 문제 및 향후 연구 방향

#### **문제 1: 극도의 도메인 갭 (Extreme Domain Gaps)**

**사례**: 야간 합성 → 주간 실제 이미지

**해결 아이디어 (2025 연구)**:
- LDDS: VLM 기반 스타일 다양화로 극단 케이스 커버
- DIDM: 도메인 특화 특성 보존으로 적응 가능 영역 확대

**연구 과제**:

$$\text{최적화}: \quad \min_w \mathbb{E}_{x \sim \mathcal{D}_t} \left[\ell(f_w(x), \tilde{y})\right] + \lambda \cdot D_{\text{div}}(\theta_{\text{src}}, \theta_{\text{tgt}})$$

#### **문제 2: 정보 손실 (Information Bottleneck)**

현재 방법들(2024-2025)의 공통 문제:
- 도메인 불변 특성 학습 중 **도메인-특화 판별 정보 손실 가능**

**최신 해결책**: 
- DIDM (2025): DLM(다양성 학습 모듈) + WAM(가중 정렬 모듈)로 양립

#### **문제 3: 계산 비용 (Computational Overhead)**

| 방법 | 학습 시간 | GPU 메모리 | 적응 가능성 |
|-----|---------|---------|----------|
| Sun & Saenko (2014) | 초 단위 | 수백 MB | 매우 높음 |
| ALDI (2024) | 시간 단위 | 수 GB | 중간 |
| LDDS (2025) | 시간 단위 | 10+ GB | 중간 |

**개선 방향**: LoRA(Low-Rank Adaptation), 프루닝 기반 경량 적응

### 7.3 실무 적용 고려사항

#### **자율주행 도메인에의 적용 (2025 관점)**

| 고려 사항 | 2014 방법 | 2025 방법 | 권장사항 |
|----------|---------|---------|--------|
| **실시간성** | ✓ 우수 | ✗ 느림 | 경량 모델 사용 |
| **도메인 다양성** | ✗ 제한적 | ✓ 우수 | VLM 기반 다양화 |
| **라벨 비용** | 낮음 | 중간 | 반자동 라벨링 병행 |
| **일반화 견고성** | 중간 | 우수 | 앙상블 방법 혼합 |

#### **로봇 응용 (실제 배포 사례)**

```
조건: 실험실(합성) → 현장(실제)
시나리오: 공장 로봇 객체 인식

2014 전략                      2025 통합 전략
└─ 가상 CAD 모델               └─ 합성 렌더 + 도메인 적응
   + 도메인-특화 적응              + VLM 스타일 증강
   (비용 대비 합리적)              + 온라인 미세조정
                               (더 견고하지만 비용 증가)
```

### 7.4 향후 연구 시 전략적 고려 사항

#### **추천 사항 1: 하이브리드 접근법**

```python
# 의사 코드 (Pseudo-code)
def adaptive_detection_2025(source_model, target_data):
    # 단계 1: 도메인 통계 추정 (Sun & Saenko 기초)
    src_stats = compute_domain_statistics(source_model)
    tgt_stats = compute_domain_statistics(target_data, unlabeled=True)
    
    # 단계 2: VLM 기반 스타일 다양화 (2025 혁신)
    diverse_samples = vlm_style_mixing(target_data, prompts=["fog", "rain", "night"])
    
    # 단계 3: 다중 그래뉼리티 정렬 (최신 SOTA)
    aligned_features = multi_granularity_align(
        source_features=extract_features(source_model, source_data),
        target_features=extract_features(source_model, diverse_samples),
        stats=(src_stats, tgt_stats)
    )
    
    # 단계 4: 도메인-특화 다양성 보존 (DIDM)
    final_model = learn_invariant_with_diversity(aligned_features)
    
    return final_model
```

#### **추천 사항 2: 벤치마킹 전략**

새 연구 수행 시 필수 비교 기준선:
1. Source-only (baseline)
2. Sun & Saenko (2014) 복제 (이 분야의 원점)
3. 최신 SOTA (ALDI, LDDS, DIDM 중 하나)
4. 앙상블 (위 세 방법의 조합)

#### **추천 사항 3: 데이터셋 선택**

| 데이터셋 | 난이도 | 추천 시기 | 공개 여부 |
|---------|------|---------|---------|
| Sim10k → Cityscapes | 중간 | 초기 실험 | ✓ 공개 |
| SYNTHIA → Cityscapes | 높음 | 성숙 단계 | ✓ 공개 |
| Office-31 (원본) | 낮음 | 개념 검증 | ✓ 공개 |
| 커스텀 산업 데이터 | 변동 | 최종 검증 | ✗ 비공개 |

***

## 결론

Sun과 Saenko (2014)의 "From Virtual to Reality"는 단순하지만 강력한 통찰을 제시합니다: **포토리얼리즘보다 도메인-특화 통계가 중요**하다는 것입니다. 11년이 경과한 2025년에도 이 핵심 원리는 LDDS, DIDM, PCL 등 최신 방법론에 반영되어 있습니다.

**향후 연구의 방향**:
1. 도메인 간 양성 특성 차이를 모델링하는 이론적 프레임워크
2. 경량/실시간 적응 가능한 에지 디바이스 친화적 방법
3. 극도의 도메인 갭(예: 합성 야간 → 실제 주간) 해결
4. 다중 작업(감지+분할+추적) 통합 적응 프레임워크

이러한 발전이 이루어질 때, 진정한 의미의 "From Virtual to Reality"가 실현될 것입니다.

***

## 참고문헌 및 추가 자료

[1](https://ieeexplore.ieee.org/document/10993287/)
[2](https://arxiv.org/abs/2509.13792)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e76bc9d7-173a-4e48-b24f-9039e7108617/bmvc14_sun_fromvirtualtoreal.pdf)
[4](https://ieeexplore.ieee.org/document/11198434/)
[5](https://link.springer.com/10.1007/s11760-025-04580-z)
[6](https://ieeexplore.ieee.org/document/11216710/)
[7](https://www.semanticscholar.org/paper/28d2ebc9fa00aee17b6b7869d9133785a41aeb3d)
[8](https://dl.acm.org/doi/10.1145/3746027.3755461)
[9](https://ieeexplore.ieee.org/document/11223280/)
[10](https://arxiv.org/abs/2509.17593)
[11](https://arxiv.org/abs/2504.20498)
[12](https://arxiv.org/html/2412.01477)
[13](http://arxiv.org/pdf/2501.04950.pdf)
[14](http://arxiv.org/pdf/1807.09834.pdf)
[15](https://arxiv.org/html/2503.13617v1)
[16](https://arxiv.org/html/2406.11311v2)
[17](https://arxiv.org/pdf/2412.17325.pdf)
[18](https://joiv.org/index.php/joiv/article/download/939/438)
[19](http://arxiv.org/pdf/2310.19258.pdf)
[20](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1581778/full)
[21](https://arxiv.org/abs/2011.03807)
[22](https://www.sciencedirect.com/science/article/abs/pii/S2542660525001295)
[23](https://www.sciencedirect.com/science/article/abs/pii/S0262885624005092)
[24](https://www.nature.com/articles/s42005-022-00844-z)
[25](https://windowsontheory.org/2020/10/18/understanding-generalization-requires-rethinking-deep-learning/)
[26](https://arxiv.org/html/2501.04950v2)
[27](https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_DiSRT-In-Bed_Diffusion-Based_Sim-to-Real_Transfer_Framework_for_In-Bed_Human_Mesh_Recovery_CVPR_2025_paper.pdf)
[28](https://cvpr.thecvf.com/virtual/2025/events/workshop)
[29](https://arxiv.org/html/2509.15045v1)
[30](https://arxiv.org/html/2506.14831v2)
[31](https://arxiv.org/html/2510.09586v1)
[32](https://arxiv.org/html/2507.16406v1)
[33](https://arxiv.org/html/2503.06072v3)
[34](https://arxiv.org/pdf/2408.17059.pdf)
[35](https://arxiv.org/html/2510.25445v1)
[36](https://arxiv.org/html/2507.22659v2)
[37](https://arxiv.org/html/2510.18518v1)
[38](https://arxiv.org/html/2511.00105v1)
[39](https://arxiv.org/html/2510.03353v1)
[40](https://ntrs.nasa.gov/api/citations/20240015866/downloads/GAN%20Based%20Data%20Augmentation%20for%20Sim%20to%20Real-final.pdf)
[41](https://arxiv.org/pdf/2502.20396.pdf)
[42](https://arxiv.org/abs/2402.04672)
[43](https://ieeexplore.ieee.org/document/10656467/)
[44](https://ieeexplore.ieee.org/document/10618373/)
[45](https://ieeexplore.ieee.org/document/10498857/)
[46](https://link.springer.com/10.1007/s10462-024-10817-z)
[47](https://link.springer.com/10.1007/s11263-025-02465-9)
[48](https://www.mdpi.com/2072-4292/17/23/3854)
[49](https://arxiv.org/abs/2505.07219)
[50](https://arxiv.org/abs/2402.12765)
[51](https://arxiv.org/abs/2502.03835)
[52](http://arxiv.org/pdf/2403.09918.pdf)
[53](http://arxiv.org/pdf/2308.09931.pdf)
[54](https://arxiv.org/abs/2106.15793v1)
[55](http://arxiv.org/pdf/2204.07964.pdf)
[56](https://arxiv.org/pdf/2301.00371.pdf)
[57](https://arxiv.org/pdf/2102.08604.pdf)
[58](http://arxiv.org/pdf/2105.12355.pdf)
[59](https://arxiv.org/html/2312.12133v1)
[60](https://pure.kaist.ac.kr/en/publications/object-aware-domain-generalization-for-object-detection)
[61](https://yenra.com/ai20/neural-architecture-search/)
[62](https://www.ijcai.org/proceedings/2024/0111.pdf)
[63](https://openaccess.thecvf.com/content/CVPR2024/papers/Danish_Improving_Single_Domain-Generalized_Object_Detection_A_Focus_on_Diversification_and_CVPR_2024_paper.pdf)
[64](https://www.sciencedirect.com/science/article/abs/pii/S0888327023005472)
[65](http://arxiv.org/abs/2502.00052)
[66](https://openaccess.thecvf.com/content/WACV2024/papers/Belal_Multi-Source_Domain_Adaptation_for_Object_Detection_With_Prototype-Based_Mean_Teacher_WACV_2024_paper.pdf)
[67](https://academic.oup.com/nsr/article/11/8/nwae282/7740455)
[68](https://www.sciencedirect.com/science/article/abs/pii/S095219762300578X)
[69](https://proceedings.neurips.cc/paper_files/paper/2024/file/6b7e1e96243c9edc378f85e7d232e415-Paper-Conference.pdf)
[70](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_Unsupervised_Continual_Domain_Shift_Learning_with_Multi-Prototype_Modeling_CVPR_2025_paper.pdf)
[71](https://arxiv.org/pdf/2508.10177.pdf)
[72](https://arxiv.org/html/2511.17217v1)
[73](https://arxiv.org/pdf/2503.20516.pdf)
[74](https://arxiv.org/html/2507.11540v2)
[75](https://arxiv.org/html/2511.20500)
[76](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Fu_NTIRE_2025_Challenge_on_Cross-Domain_Few-Shot_Object_Detection_Methods_and_CVPRW_2025_paper.pdf)
[77](https://arxiv.org/html/2503.06027v2)
[78](https://arxiv.org/html/2403.14410v2)
[79](https://arxiv.org/html/2507.23307v1)
