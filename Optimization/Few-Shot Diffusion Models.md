# Few-Shot Diffusion Models

### 1. 핵심 주장과 주요 기여

**Few-Shot Diffusion Models (FSDM)**는 **생성 확산 모델(Denoising Diffusion Probabilistic Models, DDPM)**의 강력한 표현 능력을 활용하여, 사전 학습 중 보지 못한 새로운 클래스의 샘플을 소수의 예시(5개 샘플 수준)로부터 생성할 수 있는 혁신적인 프레임워크입니다.[1]

논문의 핵심 주장은 다음과 같습니다:

1. **조건부 계층적 모델링**: 확산 모델의 매개변수 공유와 매개변수 자유 추론 과정이 few-shot 생성에 이상적인 속성이며, 집합 수준의 조건화를 통해 표현력 있는 적응 메커니즘을 구현할 수 있다는 것[1]

2. **Learnable Attentive Conditioning (LAC)**: 입력 집합을 패치 기반으로 처리하고, 표본-수준 변수와 집합-수준 변수 간의 교차 주의(cross-attention)를 통해 DDPM을 조건화하는 새로운 방법론[1]

3. **일반화 성능 향상**: 비조건부 및 조건부 DDPM 기준선 대비 데이터 효율성, 샘플 품질, 다양성 측면에서 우수한 성능을 보이며, 교차 데이터셋 전이(transfer) 능력까지 시연[1]

***

### 2. 해결하고자 하는 문제와 제안 방법

#### 2.1 문제 정의

**해결 대상 문제**:
- 기계 학습 시스템은 전통적으로 대량의 데이터를 필요로 하는 반면, 인간은 극소량의 예시로부터 새로운 개념을 학습할 수 있음[1]
- 생성형 잠재변수 모델(VAE, 자기회귀 모델 등)에서의 few-shot 적응은 매우 도전적임[1]
- 기존 조건화 메커니즘들(FiLM, 단순 평균화 등)은 복잡한 새로운 객체 클래스 생성에 실패함[1]

#### 2.2 제안 방법: FSDM 프레임워크

**수학적 정식화**:

FSDM의 생성 모델은 다음과 같이 정의됩니다:

$$p_\theta(x_{0:T}|X) = p_\theta(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t, c), \quad c = h_\phi(X)$$

여기서:
- $X = \{x_s\}_{s=1}^S$: 지원 집합(support set)
- $c$: 집합 $X$로부터 생성된 컨텍스트 표현
- $h_\phi$: Vision Transformer 기반 컨텍스트 인코더

**손실 함수**:

조건부 per-layer 손실은:

$$L^c_{t-1,\epsilon} = \mathbb{E}_{q(\epsilon)} \left[ \|\epsilon_\theta(x_t, c) - \epsilon\|_2^2 \right]$$

여기서 $x_t(x_0, \epsilon) = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$

전체 조건부 ELBO는:

$$L_{FSDM} = L^c_0 + \sum_{t=2}^{T} L^c_{t-1} + L^c_T$$

#### 2.3 모델 구조

**1) 컨텍스트 네트워크 ($h_\phi$)**:

Set Vision Transformer (sViT)를 사용:
- 입력 집합 $X$를 겹치지 않는 패치로 분할
- 공유된 위치 인코딩을 사용하여 패치들을 트랜스포머 인코더에 입력
- 벡터 형태 $c \in \mathbb{R}^d$ 또는 토큰 집합 형태 $c \in \mathbb{R}^{N \times d}$로 출력

**2) 조건화 메커니즘**: 두 가지 방식 제시

| 조건화 방식 | 컨텍스트 형태 | 공식 | 설명 |
|-----------|-----------|------|------|
| **FiLM** | 벡터 ($c \in \mathbb{R}^d$) | $u = m(c) \odot u + b(c)$ | 모든 계층에서 학습 가능한 변환 적용 |
| **LAC** | 토큰 ($c \in \mathbb{R}^{N \times d}$) | $\(u=\text{att}(u,\{c_{pp=1}^{N_{p}}\})\)$ | 교차 주의를 통한 정보 융합 |

**LAC의 핵심**:
- 패치 단위 집계: $c_p = \frac{1}{N_s} \sum_{s=1}^{N_s} c_s^p$
- 교차 주의 적용으로 모든 샘플의 정보를 효율적으로 통합
- 집합 크기 변화에 강건함

**3) 변분 FSDM (Variational FSDM)**:

대안 공식화로, 컨텍스트 $c$를 잠재변수로 모델링:

$$p_\theta(X_{0:T}, c) = p_\theta(c) \prod_{s=1}^{S} p_\theta(x^{(s)}_{0:T}|c)$$

$$q_\phi(X_{1:T}, c|X_0) = q_\phi(c|X_0) \prod_{s=1}^{S} q(x^{(s)}_{1:T}|x^{(s)}_0, c)$$

ELBO:

$$L_{VFSDM} = \mathbb{E}_{q_\phi(c|X_0)}[L_{FSDM}] + \text{KL}[q_\phi(c|X_0) \| p_\theta(c)]$$

***

### 3. 성능 향상 및 한계

#### 3.1 성능 향상 분석

**정량적 결과** (표 2, CIFAR100 데이터셋):

| 메트릭 | 지표 | DDPM | cDDPM | sDDPM | FSDM-s | **FSDM** |
|--------|------|------|-------|-------|--------|----------|
| 기본 성능 | $L_\epsilon$ (In) | 6.92 | 6.58 | 6.70 | 5.81 | **5.56** |
| | $L_\epsilon$ (Out) | 8.14 | 8.08 | 8.17 | 7.72 | **6.88** |
| 이미지 품질 | FID (Out) | 62.84 | 38.50 | 45.50 | 40.71 | **35.07** |
| 공간 관계 | sFID (Out) | 28.91 | 22.21 | 29.87 | 22.12 | **20.95** |
| 다양성 | Recall (Out) | 0.40 | 0.46 | 0.46 | 0.44 | **0.53** |

**miniImageNet 전이 결과**:
- CIFAR100에서 학습, miniImageNet에서 테스트 시 FID: 39.55 (ILVR: 53.12, DDPM: 63.13)[1]

**수렴 속도 개선** (그림 2):
- FSDM은 동등한 성능을 달성하기까지 필요한 학습 단계 수가 현저히 적음
- 패치 기반 입력 집합 정보 조건화는 학습 수렴을 가속화[1]

#### 3.2 주요 성능 향상의 원인

**1) 데이터 효율성**: 
- 컨텍스트 네트워크가 입력 집합에서 글로벌 정보를 효율적으로 추출
- 제한된 샘플로부터 더 나은 표현 학습 가능[1]

**2) 표현력 있는 조건화**:
- LAC의 교차 주의 메커니즘이 집합 내 여러 샘플의 정보를 동적으로 융합
- 단순 평균화(기준선)보다 훨씬 더 풍부한 조건 정보 전달[1]

**3) 계층 특화 학습**:
- sViT에서 각 계층 $t$에 따라 다른 수준의 상세함으로 컨텍스트 학습 가능
- 거친 단계에서는 대략적 정보, 미세 단계에서는 세부 정보 습득[1]

#### 3.3 명시적 한계

**1) Out-of-Distribution 성능 저하**:
- 학습 중 보지 못한 새로운 클래스 생성 시 성능 감소는 필연적
- CIFAR100에서 학습 후 miniImageNet에서 테스트 시 FID 급증(39.55 → 그 이상)[1]

**2) 입력 독립적 vs 입력 종속적 컨텍스트 간 트레이드오프**:
- 입력 독립적 컨텍스트: 분포 내 샘플 다양성 우수, 분포 외 성능 저하
- 입력 종속적 컨텍스트: 분포 외 조건화 품질 우수, 분포 내 다양성 감소[1]
- 최종 선택: 분포 외 성능 우선 (입력 종속적)

**3) 변분 FSDM의 한계**:
- 초기 실험에서 학습이 더 도전적이며 성능 저하
- 컨텍스트에 대한 매개변수화된 인코더가 최적화 어려움[1]

**4) 집합 크기 의존성**:
- 극소수(1-2개) 샘플로는 충분한 정보 추출 어려움
- 5개 이상의 샘플이 안정적인 성능 보장[1]

**5) 모달리티 한정**:
- 현재 시각 데이터 중심으로 설계
- 다른 모달리티(텍스트, 오디오)로의 확장 미흡[1]

***

### 4. 모델의 일반화 성능 향상 가능성

#### 4.1 현재 일반화 능력 분석

**현재 강점**:
- **도메인 내 일반화** (In-Distribution): 기존 클래스에서 매우 높은 성능
- **제한된 도메인 외 일반화** (Out-of-Distribution): 단계적 평가에서 괜찮은 성능
  - CIFAR100 → CIFAR100mix: 새 클래스지만 같은 카테고리 범위
  - Recall@Out = 0.53 (기준선 0.40-0.46 대비)[1]

**현재 한계**:
- **극도의 도메인 외 일반화**: CIFAR100 (일반 객체) → miniImageNet (동물 중심) 시 큰 성능 저하
- **카테고리 간 전이**: 학습된 개념의 근본적 다름 (예: 과일 학습 후 동물 생성) 시 어려움[1]

#### 4.2 일반화 성능 향상 가능성

**긍정적 가능성**:

1. **대규모 사전학습 데이터 활용**
   - 더 광범위한 클래스 스펙트럼에 노출된 DDPM 백본 사용
   - 일반화 능력 증대[1]

2. **향상된 컨텍스트 인코딩 전략**
   - 현재: 평균 집계 및 패치 수준 어텐션
   - 가능성: 계층적 컨텍스트 표현, 샘플-샘플 상호작용 명시적 모델링
   - 잠재 공간에서의 더 표현력 있는 벡터 학습[1]

3. **메타 학습 통합**
   - MAML(Model-Agnostic Meta-Learning) 같은 기울기 기반 메타 학습 결합[2][3][4][5]
   - 일반화 가능한 초기화 학습[3]

4. **적응적 조건화 메커니즘**
   - 고정된 LAC 대신 동적 어텐션 가중치 학습
   - 각 샘플에 대해 최적의 컨텍스트 수준 조정[1]

5. **정규화 기법 강화**
   - Contrastive learning을 통한 표현 학습[6]
   - 배포 시프트에 강건한 특성 습득

**부정적 고려사항**:

1. **근본적 한계**:
   - 확산 모델은 학습 분포의 밀도 추정에 최적화됨
   - 극도로 다른 분포에 대한 외삽은 본질적으로 어려움[1]

2. **Few-shot의 정보 부족**:
   - 5개 샘플로부터 완전한 클래스 분포 학습 불가능
   - 통계적 한계 존재[1]

3. **모듈 상호작용의 복잡성**:
   - 컨텍스트 네트워크와 생성 모델의 공동 최적화의 어려움
   - 과적합 위험성[1]

#### 4.3 이론적 근거

**FSDM이 일반화할 수 있는 이유**:[1]
- **계층적 구조**: 매개변수 공유는 학습 용량을 제한하면서도 계층적 추상화 제공
- **매개변수 자유 후향 과정**: 추가 매개변수 도입 없이 테스트 시간 적응 가능
- **집합 기반 컨텍스트화**: 개별 샘플이 아닌 집합 레벨의 정보가 분포 수준의 특성 포착[1]

***

### 5. 최신 연구 탐색 (2020년 이후)

#### 5.1 관련 주요 연구

**A. 조건부 확산 모델 발전**

| 연도 | 논문/방법 | 기여 |
|------|---------|------|
| 2021 | D2C (Diffusion-Decoding)[7] | Few-shot 조건부 생성을 위한 초기 확산 기반 접근 |
| 2023 | DifFSS[8] | 확산 모델을 Few-shot 의미 분할에 처음 적용 |
| 2023 | MetaDiff[3] | 조건부 확산으로 메타 학습의 기울기 강하 과정을 모델링 |
| 2024 | DiffiT[9] | Vision Transformer 기반 확산 생성 모델 제안 |
| 2024 | CDM (Conditional Distribution Modelling)[10] | 잠재 공간 분포 모델링으로 few-shot 생성 개선 |

**B. Few-shot 학습과 확산 모델의 결합**

- **Meta-DM (2023)**: Few-shot 학습을 위한 데이터 처리 모듈로 확산 모델 활용. 기존 FSL 방법에 플러그인 가능한 범용 모듈[11]
- **MetaDiff (2023)**: 기울기 강하를 확산 과정으로 재해석. 이계 미분 계산 제거로 메모리 효율성 향상[3]
- **Meta-Learning Without Data (2024)**: 무조건 확산 모델과 메타 학습을 결합하여 라벨 없는 학습 가능[2]

**C. Meta-Learning 최신 동향**

- **UniDense (2024)**: Mixture-of-Experts 아키텍처와 메타 라우터를 활용한 범용 few-shot 밀집 예측[12]
- **Few-shot Learner Parameterization by Diffusion Time-steps (2024)**: 시간 단계별로 확산 모델의 다양한 특성 추출, LoRA로 효율적 적응[13]

**D. Transfer Learning과 일반화**

- **Self-supervised ViT for Domain Generalization (2024)**: 자감독 ViT가 히스토패톨로지 영상에서 우수한 도메인 일반화 능력 보유[14]
- **FDS (Feedback-guided Domain Synthesis, 2025)**: 확산 모델을 이용한 도메인 혼합으로 SOTA 도메인 일반화 성능 달성[15]
- **Meta-Unlearning (2025)**: 확산 모델이 재학습 시도에 대한 저항력을 메타 학습으로 강화[16]

#### 5.2 Vision Transformer 기술 발전[17]

**ViT의 확산 모델 통합**:
- 패치 기반 처리의 우수성 확인
- 자기주의와 교차주의의 효과적 활용
- 다중 헤드 어텐션으로 다양한 수준의 특성 학습

**최신 효율화 기법**:
- Swin Transformer: 이동된 윈도우로 계산 복잡도 감소[9]
- DeiT(Data-efficient Image Transformers): 작은 데이터셋에서 효과적 학습[9]
- CompactViT: 모바일 및 엣지 장치 배포[9]

#### 5.3 주의 메커니즘의 역할

**Attention in Diffusion Models (2025)**:[18]
- 자기주의: 모드 내 공간 종속성 포착
- 교차주의: 모달리티 간 특성 정렬 및 퓨전
- 어텐션 스코어 기반 가이던스: 세밀한 제어 가능[18]

**Learnable Attentive Conditioning 맥락**:
- FSDM의 LAC는 교차주의를 few-shot 컨텍스트에 특화시킨 사례
- 최근 연구들이 어텐션의 역할 강조하는 추세[18]

#### 5.4 Out-of-Distribution 일반화 연구

**Domain Generalization 접근**:[15]
- Stable Diffusion 기반 다중 소스 도메인 혼합
- PACS: 4.5% 정확도 향상 (SOTA 달성)[15]

**Transfer Learning 메커니즘**:[19]
- 사전학습 전략이 OOD 성능에 미치는 영향
- 미세조정 프로토콜의 중요성[19]

***

### 6. 미래 연구에 미치는 영향과 고려사항

#### 6.1 논문이 미칠 연구 영향

**즉각적 영향**:

1. **Few-shot 생성 모델의 새로운 방향 제시**
   - 확산 모델이 few-shot 시나리오에서 경쟁력 있음을 입증
   - 기존 GAN, 자기회귀 모델 중심의 관점 전환[1]

2. **집합 기반 조건화의 중요성**
   - 단순한 클래스 라벨이 아닌 샘플 집합 자체를 조건으로 활용하는 패러다임
   - 메타 학습과의 자연스러운 통합 가능성[1]

3. **ViT + 확산 모델 결합의 선례**
   - 패치 기반 처리가 계산 효율성과 표현력의 균형점임을 보여줌
   - 이후 연구에서 광범위하게 채택 (DiffiT, FDS 등)[9]

**장기적 영향**:

1. **범용 생성 모델로의 진화**
   - FSDM 프레임워크를 다른 모달리티(텍스트, 3D, 멀티모달)로 확장
   - 통합 생성 모델의 토대 제공[1]

2. **메타 학습과 생성 모델의 깊은 통합**
   - 현재: 별도의 영역 → 향후: 유기적 결합
   - MetaDiff, Meta-DM 등의 후속 연구 촉발[11][2][3]

3. **산업 응용의 확대**
   - 의료 영상 (제한된 이미지로부터의 합성)
   - 개인화된 콘텐츠 생성
   - 데이터 증강[20][21][22]

#### 6.2 향후 연구 시 고려할 점

**기술적 개선 방향**:

1. **더 강력한 컨텍스트 인코더**
   ```
   현재: sViT + 평균 집계 + 교차주의
   제안: 
   - 그래프 신경망으로 샘플 간 관계 명시
   - 적응적 패치 크기 조정
   - 시간 의존적 컨텍스트 강화
   ```

2. **메타 학습 통합 강화**
   - MAML의 이계 미분 문제 → 확산 기반 메타 최적화로 해결[4][3]
   - 온라인 적응 시나리오 고려

3. **계산 효율성**
   - 현재: 200K 반복 학습 (대규모 리소스 필요)
   - 개선: 매개변수 효율적 미세조정 (LoRA, 어댑터)[13]
   - 증류 기법 활용으로 샘플링 가속화

4. **불확실성 정량화**
   - 현재: 점 추정 기반 조건화
   - 제안: 확률적 컨텍스트 학습으로 신뢰도 추정[3]

5. **크로스 모달 일반화**
   - 텍스트 또는 스케치로부터의 few-shot 생성
   - 이미지-텍스트 정합 개선

**평가 및 벤치마크 측면**:

1. **더 엄격한 OOD 평가**
   - 현재: 같은 데이터셋 내 새 클래스
   - 제안: 완전히 다른 도메인, 카테고리 간 전이
   - 극단적 few-shot (1-2개 샘플) 평가[1]

2. **인간 평가 추가**
   - FID/sFID만으로는 부족
   - 다양성(diversity), 일관성(coherence) 인간 평가[1]

3. **계산 복잡도 측정**
   - 메모리 사용량, 샘플링 시간 문서화
   - 실제 응용 가능성 평가[1]

**이론적 분석**:

1. **일반화 한계 분석**
   - PAC-Bayes 이론 적용[23]
   - 샘플 복잡도 이론적 하한 도출
   - 집합 크기와 생성 품질의 관계 분석

2. **컨텍스트 충분성**
   - $n$개 샘플로부터 클래스 분포를 얼마나 정확히 복원할 수 있는가?
   - 차원의 저주 문제 분석[1]

3. **확산 과정의 내재적 구조**
   - 서로 다른 시간 단계에서 학습되는 특성 분석[13]
   - 초기 단계 vs 후기 단계의 역할[24]

**실무적 고려사항**:

1. **데이터 편향 문제**
   - Few-shot 학습 시 소수 샘플의 편향이 생성 모델에 미치는 영향
   - 가중치 조정, 데이터 증강 전략[1]

2. **모달리티별 최적화**
   - 이미지만 아닌 3D, 점 구름, 그래프 등에 대한 구체적 방법론
   - 각 모달리티의 특성에 맞는 패치 정의[1]

3. **개인정보 보호**
   - Few-shot 샘플로부터의 과적합으로 인한 데이터 유출 위험
   - 차분 프라이버시 기법 적용 필요[1]

4. **배포 및 실시간 처리**
   - 모바일/엣지 장치에서의 추론
   - 경량 모델 개발 (사전 학습된 소형 DDPM)[1]

#### 6.3 병렬 진행 중인 관련 연구 방향

**최근 3-4년 (2023-2025) 주요 동향**:

1. **양자 기계학습**: Quantum Diffusion Models을 활용한 few-shot 학습으로 3-way 10-shot에서 SOTA 달성[25]

2. **원격 감지 응용**: DIG-FSOD로 원격 감지 이미지의 few-shot 객체 탐지 향상[20]

3. **이상 탐지**: DualAnoDiff로 few-shot 이상 이미지 생성 및 마스크 생성[21]

4. **신호 처리**: PPG 신호의 few-shot 생성을 위한 가이드 확산 모델[22]

5. **강화학습 적응**: 확산 모델 손실 기반 RL로 미세조정 개선[26][27]

***

### 결론

**Few-Shot Diffusion Models**은 확산 모델의 계층적, 매개변수 공유 구조와 Vision Transformer의 패치 기반 유연성을 결합하여, **극소량의 예시(5개 샘플)로부터 새로운 클래스 생성**이라는 도전 과제를 해결한 중요한 연구입니다.[1]

**핵심 기여**:
- **새로운 프레임워크**: 조건부 DDPM의 few-shot 적응성을 체계적으로 탐구한 최초의 포괄적 연구[1]
- **혁신적 조건화 방식**: LAC를 통한 교차주의 기반 집합 처리로 단순 평균화 대비 현저한 성능 향상[1]
- **광범위한 평가**: 분포 내외 성능, 전이 학습, 샘플링 시간 등 다각적 평가로 방법론의 우수성 입증[1]

**앞으로의 방향**:
- 메타 학습 기법과의 깊은 통합으로 더욱 강력한 일반화 능력 확보
- 다중 모달리티 지원으로 범용 생성 모델로의 진화
- 이론적 일반화 한계 분석으로 근본적 이해 심화
- 실제 응용(의료 영상, 개인화 생성)으로의 적용 확대

이 논문은 **생성형 모델의 few-shot 적응**이 가능함을 보였으며, 이는 데이터 부족 환경에서의 AI 시스템 개발에 새로운 가능성을 열어주었습니다.[2][11][3][1]

***

### 참고 문헌 요약

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0c05cd02-9d3f-4f14-b1ec-927dadfdfccc/2205.15463v1.pdf)
[2](https://ieeexplore.ieee.org/document/10587268/)
[3](https://arxiv.org/abs/2307.16424)
[4](https://arxiv.org/pdf/2307.16424.pdf)
[5](https://arxiv.org/pdf/2305.08092.pdf)
[6](https://www.nature.com/articles/s41598-024-61040-3)
[7](https://proceedings.neurips.cc/paper/2021/file/682e0e796084e163c5ca053dd8573b0c-Paper.pdf)
[8](https://arxiv.org/abs/2307.00773)
[9](https://arxiv.org/abs/2312.02139)
[10](http://arxiv.org/pdf/2404.16556.pdf)
[11](https://arxiv.org/abs/2305.08092)
[12](https://dl.acm.org/doi/10.1145/3664647.3680831)
[13](https://openaccess.thecvf.com/content/CVPR2024/papers/Yue_Few-shot_Learner_Parameterization_by_Diffusion_Time-steps_CVPR_2024_paper.pdf)
[14](https://arxiv.org/abs/2407.02900)
[15](https://openaccess.thecvf.com/content/WACV2025/papers/Noori_FDS_Feedback-Guided_Domain_Synthesis_with_Multi-Source_Conditional_Diffusion_Models_for_WACV_2025_paper.pdf)
[16](https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_Meta-Unlearning_on_Diffusion_Models_Preventing_Relearning_Unlearned_Concepts_ICCV_2025_paper.pdf)
[17](https://blog.roboflow.com/vision-transformers/)
[18](https://arxiv.org/html/2504.03738v1)
[19](https://proceedings.neurips.cc/paper_files/paper/2022/file/2f5acc925919209370a3af4eac5cad4a-Paper-Conference.pdf)
[20](https://arxiv.org/html/2511.18031v1)
[21](https://openaccess.thecvf.com/content/CVPR2025/papers/Jin_Dual-Interrelated_Diffusion_Model_for_Few-Shot_Anomaly_Image_Generation_CVPR_2025_paper.pdf)
[22](https://yonsei.elsevierpure.com/en/publications/few-shot-ppg-signal-generation-via-guided-diffusion-models-2/)
[23](https://openreview.net/forum?id=JrraNaaZm5)
[24](https://arxiv.org/abs/2403.01633)
[25](https://www.merl.com/publications/docs/TR2025-025.pdf)
[26](https://ieeexplore.ieee.org/document/10658223/)
[27](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[28](http://medrxiv.org/lookup/doi/10.1101/2024.12.13.24319008)
[29](https://arxiv.org/abs/2507.02686)
[30](https://dialogue-conf.org/wp-content/uploads/2025/06/RossyaykinP.105.pdf)
[31](https://link.springer.com/10.1007/s00330-025-11871-z)
[32](https://www.semanticscholar.org/paper/6c708659768e470f63d06f791ff8420e7ff0feac)
[33](https://arxiv.org/abs/2503.21330)
[34](https://arxiv.org/abs/2509.11446)
[35](http://www.sor-journal.org/index.php/sor/article/view/18)
[36](http://medrxiv.org/lookup/doi/10.1101/2024.08.11.24311828)
[37](https://pubs.aip.org/pof/article/37/11/117120/3371493/Fine-structure-investigation-of-turbulence-induced)
[38](http://arxiv.org/pdf/2305.15798.pdf)
[39](https://arxiv.org/html/2503.06674v1)
[40](https://arxiv.org/pdf/2311.16353.pdf)
[41](https://arxiv.org/html/2406.03146v1)
[42](https://arxiv.org/pdf/2402.03017.pdf)
[43](https://arxiv.org/abs/2305.10722)
[44](https://proceedings.neurips.cc/paper_files/paper/2024/file/0c1124bd3be769dacf491d92d499c7d8-Paper-Conference.pdf)
[45](https://arxiv.org/abs/2509.16447)
[46](https://arxiv.org/abs/2205.15463)
[47](https://arxiv.org/abs/2411.12874)
[48](http://pubs.rsna.org/doi/10.1148/radiol.240153)
[49](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13035/3023596/Generative-EO-IR-multi-scale-vision-transformer-for-improved-object/10.1117/12.3023596.full)
[50](https://www.mdpi.com/2076-3417/15/12/6622)
[51](https://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0012938500004508)
[52](https://ieeexplore.ieee.org/document/10440990/)
[53](https://onlinelibrary.wiley.com/doi/10.1002/ima.22979)
[54](https://ieeexplore.ieee.org/document/11094593/)
[55](https://www.mdpi.com/2073-431X/13/12/305)
[56](https://arxiv.org/pdf/2312.09251.pdf)
[57](https://arxiv.org/abs/2303.12208)
[58](https://arxiv.org/pdf/2107.06263.pdf)
[59](https://www.mdpi.com/1424-8220/23/7/3447/pdf?version=1680001445)
[60](https://arxiv.org/html/2403.09394v1)
[61](http://arxiv.org/pdf/2408.15178.pdf)
[62](https://arxiv.org/html/2408.14131v2)
[63](https://www.emergentmind.com/topics/conditional-denoising-diffusion-probabilistic-models-ddpms)
[64](https://openreview.net/pdf?id=2E6OK8cSoB)
[65](https://www.v7labs.com/blog/vision-transformer-guide)
[66](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820730.pdf)
[67](https://sayak.dev/posts/attn-diffusion.html)
[68](https://www.sciencedirect.com/science/article/pii/S1474034625006123)
[69](https://academic.oup.com/bjr/article/97/1155/535/7529222)
[70](https://bmcpsychiatry.biomedcentral.com/articles/10.1186/s12888-024-06116-0)
[71](https://academic.oup.com/ehjdh/article/6/1/7/7845948)
[72](http://arxiv.org/pdf/2410.13201.pdf)
[73](https://aclanthology.org/2023.acl-long.248.pdf)
[74](https://arxiv.org/pdf/2206.03992.pdf)
[75](https://arxiv.org/html/2405.00984v2)
[76](https://arxiv.org/pdf/2305.18455.pdf)
[77](https://arxiv.org/pdf/2405.16560.pdf)
[78](https://www.sciencedirect.com/science/article/abs/pii/S0950705125012201)
[79](https://academic.oup.com/bib/article/26/3/bbaf294/8176475)
[80](https://www.sciencedirect.com/science/article/abs/pii/S0925231225003601)
[81](https://pubmed.ncbi.nlm.nih.gov/33381839/)
[82](https://learnprompting.org/docs/basics/few_shot)
[83](https://dl.acm.org/doi/10.1609/aaai.v38i15.29608)
