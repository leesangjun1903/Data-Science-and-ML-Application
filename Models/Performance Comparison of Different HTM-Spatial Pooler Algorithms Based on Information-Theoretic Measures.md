# Performance Comparison of Different HTM-Spatial Pooler Algorithms Based on Information-Theoretic Measures
### 1. 핵심 주장 및 주요 기여 요약
본 논문(Sanati et al., 2024)은 Hierarchical Temporal Memory(HTM)의 Spatial Pooler(SP) 알고리즘 성능을 비교하기 위한 정보이론적 프레임워크를 제안한다. 이 연구의 가장 중요한 주장은 **정보이론적 측도가 기존 통계적 방법보다 우월하며, 로그함수 부스팅을 적용한 학습형 SP 알고리즘이 최적의 성능을 제공한다**는 것이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

논문의 주요 기여는 다음과 같다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

1. **정보이론적 평가 프레임워크 구축**: Rényi 상호정보(Mutual Information), Rényi 발산(Divergence), Henze-Penrose 발산을 사용하여 SP 알고리즘의 입출력 유사성과 차이를 정량화
2. **네 가지 SP 알고리즘 비교**: 로그 부스팅, 지수 부스팅, 부스팅 없음, 학습 없음의 성능 평가
3. **최적 알고리즘 식별**: 학습 + 로그함수 부스팅이 모든 데이터셋에서 가장 효과적임을 증명
4. **현대 딥러닝과의 비교**: HTM이 LSTM, GRU, OS-ELM보다 온라인 학습과 패턴 변화 적응에서 우수함을 입증

### 2. 해결하고자 하는 문제, 제안 방법 및 모델 구조
#### 2.1 해결하고자 하는 문제

기존 HTM 연구는 SP 알고리즘이 유사한 입력에 대해 유사한 희소분산표현(SDR)을 생성한다는 것을 직관적으로 주장했으나, **엄밀한 수학적 증명과 정확한 정량적 측도가 부족했다**. 특히: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

- SP 알고리즘의 입출력 정보 보존도를 정량화하는 방법이 없었음
- 다양한 SP 변형 알고리즘 간의 성능 차이를 객관적으로 비교할 수 있는 프레임워크 부재
- 데이터 표현의 품질이 하위 학습 계층의 성능에 미치는 영향을 체계적으로 분석하지 못함

#### 2.2 제안하는 방법 및 핵심 수식

논문은 정보이론적 측도를 기반으로 하는 혁신적인 평가 프레임워크를 제안한다. 

**Rényi 발산 (입출력 차이 측정)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

$$D_\alpha(P(X)||P(Y)) = \frac{1}{\alpha-1}\log_2\left(\sum_{x,y}P(X)\left(\frac{P(X)}{P(Y)}\right)^{\alpha-1}\right)$$

가우스 분포를 사용한 근사:

$$D_\alpha \approx \frac{1}{\alpha-1}\log_2\left(\frac{1}{N}\sum_{j=1}^{N}\left(\frac{G_\delta(X_j-\mu)}{G_\delta(Y_j-\mu)}\right)^{\alpha-1}\right)$$

여기서 $G_\delta(x-x_i) = \frac{1}{\sqrt{2\pi\delta_1}}\exp\left(-\frac{(X-X_i)^2}{2\delta_1^2}\right)$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

**Henze-Penrose 발산**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

$$D_{HP}(P(X)||P(Y)) \approx \frac{1}{4ab}\left(\sum_{j=1}^{N}\frac{\left(aG_\delta(X_j-\mu)-bG_\delta(Y_j-\mu)\right)^2}{aG_\delta(X_j-\mu)+bG_\delta(Y_j-\mu)}-(a-b)^2\right)$$

**Rényi 상호정보 (입출력 유사성 측정)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

$$I_\alpha(X:Y) = \frac{1}{\alpha-1}\log_2\left(\frac{1}{N^2}\sum_{j=1}^{N}\left(\frac{G_\delta(X_j-\mu, Y_j-\mu)}{G_\delta(X_j-\mu)G_\delta(Y_j-\mu)}\right)^{\alpha-1}\right)$$

다변량 가우스 분포:

$$f(x,y) = \frac{1}{2\pi\delta_x\delta_y\sqrt{1-\rho^2}}\exp\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_x)^2}{\delta_x^2}+\frac{(y-\mu_y)^2}{\delta_y^2}-\frac{2\rho(x-\mu_x)(y-\mu_y)}{\delta_x\delta_y}\right]\right)$$

**Rényi 엔트로피 (컬럼 효율성 평가)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

$$H_\alpha(Y) = \frac{1}{1-\alpha}\log_2\left(\frac{1}{N^\alpha}\sum_{j=1}^{N}\left(\sum_{i=1}^{N}\frac{1}{\sqrt{2\pi\delta_1}}\exp\left(-\frac{(Y_j-Y_i)^2}{2\delta_1^2}\right)\right)^{\alpha-1}\right)$$

#### 2.3 모델 구조

HTM은 네 개의 주요 컴포넌트로 구성된다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

1. **인코더**: 수치 입력 데이터를 이진 벡터로 변환
2. **공간 풀러(SP)**: 이진 입력을 희소분산표현(SDR)으로 인코딩 (본 연구의 주요 대상)
3. **시간 메모리(TM)**: SDR 시계열의 패턴을 학습하고 미래 입력 예측
4. **분류기**: SDR을 최종 출력값으로 변환

SP의 내부 구조는 네 가지 단계로 구성된다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

- **초기화**: 각 컬럼에 대한 잠재 연결 풀 설정, 영구성 값(permanence) 할당
- **중복값 계산**: 입력 데이터가 컬럼의 잠재 풀과 얼마나 유사한지 계산
- **억제(Inhibition)**: 최고의 중복값을 가진 컬럼을 활성 컬럼으로 선택
- **학습**: 활성 컬럼의 시냅스 연결 강도를 조정하여 알고리즘 적응

### 3. 성능 향상 및 한계
#### 3.1 성능 향상 결과

**MNIST 데이터셋 (전역 억제 모드)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

| SP 알고리즘 | Rényi Div | HP Div | Renyi MI | 정확도 |
|-----------|-----------|--------|----------|---------|
| 로그 부스팅 + 학습 | 0.1310 | 0.1409 | 0.9557 | 95.34% |
| 지수 부스팅 + 학습 | 0.1419 | 0.1498 | 0.9442 | 94.28% |
| 부스팅 없음 + 학습 | 0.2010 | 0.2106 | 0.8873 | 88.53% |
| 학습 없음 | 0.2388 | 0.2467 | 0.8576 | 88.39% |

**로그 부스팅의 장점**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)
- Rényi 발산 18.5% 감소 (로그 vs 지수 부스팅)
- Rényi 상호정보 1.2% 증가
- 분류 정확도 1.06% 향상
- 모든 컬럼의 활용도 증가 (Rényi 엔트로피 0.79→0.98, 24% 개선)

**NYC 택시 데이터셋**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

|  | Renyi Div | RMSE | MAPE |
|-----------|-----------|--------|----------|
| 로그 부스팅 + 학습 | 0.1721 | 0.2594 | 0.1049 |
| 학습 없음 | 0.2566 | 0.3541 | 0.1423 |
#### 3.2 일반화 성능 향상 메커니즘

논문의 결과는 **정보이론적 측도와 전통적 성능 지표 간의 높은 상관관계**를 보여준다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

- Rényi 발산 감소 ↔ 상호정보 증가 (역상관)
- 상호정보 증가 ↔ 분류 정확도 향상
- 낮은 발산 ↔ 예측 오차 감소

**노이즈 강건성**: SP 알고리즘은 입력 데이터에 최대 40%의 노이즈가 추가되어도 출력이 거의 변하지 않는 강력한 노이즈 강건성을 시연했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

**고정 희소성의 역할**: 2%의 고정 희소성은 다양한 입력 희소성(2~20%)에 대해 일관된 출력 희소성을 유지하여, 하위 시간 메모리 컴포넌트가 안정적으로 패턴을 학습할 수 있게 함. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

#### 3.3 모델의 한계

논문의 주요 한계점은 다음과 같다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

1. **가우스 분포 가정**: 실제 데이터가 항상 가우스 분포를 따르지 않으므로 근사 오차 발생 가능
2. **이진 입력 제한**: 연속값 입력 데이터에는 직접 적용 불가
3. **계산 복잡도**: 정보이론적 측도 계산에 추가 연산 오버헤드
4. **제한된 데이터셋**: MNIST, Fashion-MNIST, NYC 택시, HotGym 네 가지만 평가
5. **이론적 수렴 분석 부족**: 제안 방법의 수렴 속도나 수렴 조건에 대한 형식적 증명 미흡

### 4. 모델의 일반화 성능 향상 가능성
#### 4.1 일반화 성능의 개념적 이해

논문의 정보이론적 프레임워크는 **입출력 정보의 보존도와 모델의 일반화 능력이 본질적으로 연결되어 있다**는 것을 시사한다. 높은 Rényi 상호정보는 다음을 의미한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

- SP가 입력 패턴의 본질적 정보를 유지함
- SDR 표현이 의미 있는 특징을 인코딩함
- 학습 계층이 일관된 패턴을 추출할 수 있음

#### 4.2 일반화 성능 향상의 실증적 증거

**HTM의 온라인 학습 우수성**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

| 알고리즘 | Renyi Div (패턴 변화 전) | Renyi Div (패턴 변화 후) | MI 변화 | 적응성 |
|---------|----------|----------|---------|---------|
| HTM | 0.18 | 0.23 | 0.82→0.79 (-3.7%) | 최고 |
| LSTM 6000 | 0.28 | 0.33 | 0.76→0.73 (-3.9%) | 중간 |
| LSTM 3000 | 0.32 | 0.38 | 0.73→0.71 (-2.7%) | 중간 |
| GRU | 0.37 | 0.41 | 0.70→0.68 (-2.9%) | 낮음 |
| OS-ELM | 0.40 | 0.49 | 0.64→0.62 (-3.1%) | 최저 |

HTM은 패턴 변화 후 발산이 27.8% 증가한 반면, OS-ELM은 22.5% 증가하여, **HTM의 일반화 성능이 급격한 데이터 분포 변화에도 더 안정적임**을 보여준다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)
#### 4.3 일반화 개선을 위한 메커니즘

**1. 적응적 희소성 조정** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)
- 현재: 고정 2% 희소성
- 개선 방향: 입력 데이터의 특성에 따라 동적으로 희소성 조정
- 예상 효과: 다양한 도메인 특성에 대한 적응성 향상

**2. 정보이론적 손실함수** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)
- 현재 접근: 사후 평가 도구로서의 정보이론적 측도
- 개선 방향: 훈련 중 직접적인 목적함수로 Rényi MI 최대화, 발산 최소화
- 수식:
$$\mathcal{L} = -\lambda_1 I_\alpha(X:Y) + \lambda_2 D_\alpha(P(X)||P(Y))$$

**3. 전이 학습 응용** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)
- 다른 도메인의 사전학습 SP 표현을 초기값으로 사용
- 최소한의 미세조정으로 새로운 작업에 적응

**4. 앙상블 방법** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)
- 여러 부스팅 함수를 가진 SP 알고리즘 조합
- 다양한 파라미터 설정의 강점을 결합

### 5. 논문이 앞으로의 연구에 미치는 영향 및 고려사항
#### 5.1 학문적 영향

본 논문은 **정보이론적 프레임워크가 생물학적으로 영감 받은 알고리즘의 성능 평가에 강력한 도구임**을 입증함으로써 다음과 같은 영향을 미친다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

1. **새로운 알고리즘 설계 패러다임**: Rényi MI/발산을 명시적 목적함수로 사용하는 알고리즘 개발
2. **평가 방법론의 혁신**: 통계적 메트릭 이외의 정보이론적 측도의 중요성 강조
3. **생물학적 제약의 수학적 정당화**: HTM의 설계 선택(고정 희소성, 학습 규칙)의 정보이론적 기초 제공

#### 5.2 2020년 이후 최신 연구와의 비교 분석

**A. HTM 확장 및 개선 연구** [downloads.hindawi](https://downloads.hindawi.com/journals/cin/2021/6680833.pdf)

| 연도 | 연구 | 주요 기여 |
|------|------|----------|
| 2020 | Sanati et al. 초기 연구 | 수정된 정보 병목 관계 제안 |
| 2021 | 빠른 공간 풀링 알고리즘 | 미니컬럼 자동지명을 통한 훈련 시간 단축 |
| 2022 | 활성화 강도 기반 HTM | 미세한 컬럼 정보 활용으로 표현력 향상 |
| 2024 | 본 논문 | 정보이론적 비교 프레임워크 확립 |
| 2025 | 하드웨어 가속 HTM | 7.55~10.10배 추론 속도 향상 |

**B. 정보이론적 학습 프레임워크의 발전** [mdpi](https://www.mdpi.com/1999-4893/16/9/450)

최근 3년간 정보이론적 접근은 다음과 같이 진화했다:

- **2023**: 프라이버시-보존, 해석 가능성-전이성 간 균형 [mdpi](https://www.mdpi.com/1999-4893/16/9/450)
- **2023**: 양자 학습에서의 정보이론적 경계 [arxiv](https://arxiv.org/abs/2311.05529)
- **2022**: Rényi 발산을 이용한 심층 상호 학습 개선 [arxiv](https://arxiv.org/abs/2209.05732)
- **2024**: 기계학습의 정보이론적 기초 통합 프레임워크 [arxiv](https://arxiv.org/pdf/2407.12288.pdf)

**C. 희소분산표현(SDR) 연구의 최신 동향** [emergentmind](https://www.emergentmind.com/topics/sparse-distributed-representations)

| 영역 | 진전 |
|------|------|
| 수학적 성질 | 조합론적 용량의 정확한 계산, 초지수적 오류 감소 증명 |
| 변수 바인딩 | 블록-로컬 순환 회선의 우수성 입증 |
| 응용 분야 | 언어 처리, 강화학습, 분산 시스템 확대 |
| 에너지 효율 | 스파이킹 뉴런 모델에서 혁신적 에너지 절감 |

#### 5.3 향후 연구 시 고려사항

**1. 이론적 분석의 강화** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

- Rényi 발산이 최소화될 때 일반화 오류의 명시적 상한선 유도
- SGD 수렴 속도의 정보이론적 특성화
- 정보 병목 원리와 SP 학습의 연결 고찰

**2. 실무적 응용 확대** [arxiv](https://arxiv.org/abs/2504.03746v1)

- 신경형 하드웨어(SpiNNaker, TrueNorth 등)에서의 최적화된 구현
- 엣지 컴퓨팅 환경에서의 계산 복잡도 분석
- IoT 센서 데이터 스트림에서의 실시간 처리 검증

**3. 현대 딥러닝과의 통합** [arxiv](https://arxiv.org/html/2504.03746v1)

- Vision Transformer와의 비교: 계산 효율 vs 정확도 트레이드오프
- 강화학습 프레임워크와의 통합 (HTMRL 발전)
- 그래프 신경망과의 하이브리드 아키텍처

**4. 데이터 도메인의 확장** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

- 비이진 연속 입력에 대한 정보이론적 측도 확장
- 고차원 이미지 데이터(ImageNet 규모)에서의 성능 평가
- 자연 언어 처리에서의 적용 가능성 탐색

**5. 정보이론적 프레임워크의 정교화** [frontiersin](https://www.frontiersin.org/articles/10.3389/fncom.2023.1140782/full)

- 더 정밀한 정보이론적 측도 개발 (정보 누수, 불확실성 정량화)
- 다변량 데이터에 대한 고차 의존성 캡처
- 비정상 환경에서의 적응적 측도

### 6. 결론
Sanati et al. (2024)의 연구는 **정보이론적 측도가 신경망 알고리즘의 성능 평가와 설계에 강력한 도구임**을 명확히 보여준다. 특히 Rényi 상호정보와 발산을 사용한 SP 알고리즘 비교는 다음과 같은 중요한 통찰을 제공한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/fc424112-7b96-42ca-8b3d-8d052998d5a0/s11063-024-11546-8.pdf)

1. **최적 알고리즘**: 로그함수 부스팅과 학습을 결합한 SP가 정보 보존과 컬럼 활용성에서 모두 우수함
2. **일반화의 원리**: 정보이론적 유사성이 높을수록 서로 다른 데이터셋에서 일관된 성능 유지
3. **온라인 학습의 우월성**: HTM의 패턴 변화에 대한 빠른 적응 능력이 LSTM 등보다 우수함

이 논문이 미치는 영향은 단순히 HTM 알고리즘 최적화를 넘어 **생물학적으로 제약된 신경망의 설계 원칙을 정보이론적으로 정당화하는 프레임워크**를 제공한다는 점에서 의미가 있다. 향후 연구는 이러한 정보이론적 기초를 바탕으로 더욱 효율적이고 일반화 가능한 신경망 아키텍처를 개발할 수 있을 것으로 예상된다. [arxiv](https://arxiv.org/abs/2504.03746v1)

***

**참고문헌**

<span style="display:none">[^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90]</span>

<div align="center">⁂</div>

[^1_1]: s11063-024-11546-8.pdf

[^1_2]: https://downloads.hindawi.com/journals/cin/2021/6680833.pdf

[^1_3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8803450/

[^1_4]: https://www.mdpi.com/1999-4893/16/9/450

[^1_5]: https://arxiv.org/abs/2209.05732

[^1_6]: https://arxiv.org/pdf/2407.12288.pdf

[^1_7]: https://arxiv.org/abs/2311.05529

[^1_8]: https://www.emergentmind.com/topics/sparse-distributed-representations

[^1_9]: https://pubmed.ncbi.nlm.nih.gov/34478381/

[^1_10]: https://arxiv.org/abs/2504.03746v1

[^1_11]: https://arxiv.org/html/2504.03746v1

[^1_12]: https://www.frontiersin.org/articles/10.3389/fncom.2023.1140782/full

[^1_13]: https://arxiv.org/abs/2203.00246

[^1_14]: https://dl.acm.org/doi/10.1145/3416921.3416940

[^1_15]: https://www.mdpi.com/2076-3417/10/7/2596

[^1_16]: https://ieeexplore.ieee.org/document/9294802/

[^1_17]: https://ieeexplore.ieee.org/document/9096053/

[^1_18]: https://www.mdpi.com/1424-8220/20/6/1646

[^1_19]: https://www.infocommunications.hu/2020_2_6

[^1_20]: https://ieeexplore.ieee.org/document/9260106/

[^1_21]: https://www.semanticscholar.org/paper/654313123043ef23ed48552713caac05eed037ed

[^1_22]: https://www.techrxiv.org/doi/full/10.36227/techrxiv.12404393.v1

[^1_23]: https://dl.acm.org/doi/10.1145/3393822.3432317

[^1_24]: https://arxiv.org/pdf/1611.02792.pdf

[^1_25]: https://www.mdpi.com/1424-8220/23/4/2087/pdf?version=1676276941

[^1_26]: https://arxiv.org/ftp/arxiv/papers/1512/1512.05463.pdf

[^1_27]: https://arxiv.org/abs/1808.05839

[^1_28]: http://arxiv.org/pdf/2405.06067.pdf

[^1_29]: http://arxiv.org/pdf/2111.03456.pdf

[^1_30]: https://en.wikipedia.org/wiki/Hierarchical_temporal_memory

[^1_31]: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00111/full

[^1_32]: https://pubs.aip.org/aip/acp/article/3270/1/020058/3343775/Spatial-pyramid-pooling-SPP-NET-compared-with

[^1_33]: https://www.numenta.com/assets/pdf/biological-and-machine-intelligence/BaMI-SDR.pdf

[^1_34]: https://www.sciencedirect.com/science/article/pii/S1877050920302465

[^1_35]: https://github.com/ddobric/neocortexapi/blob/master/source/Documentation/SpatialPooler.md

[^1_36]: https://www.cortical.io/science/sparse-distributed-representations/

[^1_37]: https://www.numenta.com/resources/research-publications/papers/hierarchical-temporal-memory-white-paper/

[^1_38]: https://www.nature.com/articles/s41598-024-51258-6

[^1_39]: https://www.linkedin.com/pulse/sparse-distributed-representations-harnessing-power-sparsity-n-6qkzc

[^1_40]: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2023.1140782/full

[^1_41]: https://www.sciencedirect.com/science/article/pii/S2211675325000053

[^1_42]: https://seanpedersen.github.io/posts/sparse-distributed-representations

[^1_43]: http://arxiv.org/abs/2209.14583

[^1_44]: http://arxiv.org/abs/1710.07829

[^1_45]: https://arxiv.org/abs/2504.03746

[^1_46]: https://pubmed.ncbi.nlm.nih.gov/27171856/

[^1_47]: https://pdfs.semanticscholar.org/dd52/92227caa8c1fab8a99de6a214b73ce0bf973.pdf

[^1_48]: https://arxiv.org/pdf/2112.14820.pdf

[^1_49]: https://pubmed.ncbi.nlm.nih.gov/35125669/

[^1_50]: https://arxiv.org/html/2601.02845v1

[^1_51]: https://pubmed.ncbi.nlm.nih.gov/36532804/

[^1_52]: https://arxiv.org/ftp/arxiv/papers/1710/1710.07829.pdf

[^1_53]: https://arxiv.org/pdf/1601.06116.pdf

[^1_54]: https://arxiv.org/abs/2205.12718

[^1_55]: https://arxiv.org/pdf/1806.04704.pdf

[^1_56]: https://arxiv.org/pdf/2209.14583.pdf

[^1_57]: https://www.semanticscholar.org/paper/9d768a04b3b80dc723dc94f16f4463521e1c1ded

[^1_58]: https://www.mdpi.com/1099-4300/28/1/108

[^1_59]: https://www.semanticscholar.org/paper/a2008509a2833326cf6ed7d19e1fb87ed85cb26a

[^1_60]: https://dl.acm.org/doi/10.1145/3611019

[^1_61]: https://arxiv.org/abs/2311.08309

[^1_62]: https://ieeexplore.ieee.org/document/10206951/

[^1_63]: https://genescells.ru/2313-1829/article/view/623517

[^1_64]: http://arxiv.org/pdf/2109.14595.pdf

[^1_65]: https://www.arxiv.org/pdf/1501.04309.pdf

[^1_66]: http://arxiv.org/pdf/2502.19183.pdf

[^1_67]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11284159/

[^1_68]: https://arxiv.org/pdf/2210.00881.pdf

[^1_69]: http://arxiv.org/pdf/2305.11042.pdf

[^1_70]: http://arxiv.org/pdf/2405.20452.pdf

[^1_71]: https://www.ijcai.org/proceedings/2021/0633.pdf

[^1_72]: https://www.sciencedirect.com/topics/computer-science/generalization-performance

[^1_73]: https://arxiv.org/html/2209.05732

[^1_74]: https://www.nature.com/articles/s41467-024-48069-8

[^1_75]: https://www.sciencedirect.com/science/article/pii/S0960077925017370

[^1_76]: https://proceedings.neurips.cc/paper_files/paper/2024/file/bdcfa850adac4a1088153881282ca972-Paper-Conference.pdf

[^1_77]: https://www.nature.com/articles/s42005-024-01837-w

[^1_78]: https://engineering.tamu.edu/news/2024/05/information-theoretic-measures-in-machine-learning.html

[^1_79]: https://openreview.net/pdf?id=KC2MViQASx

[^1_80]: https://arxiv.org/html/2209.01610v3

[^1_81]: https://arxiv.org/abs/2407.12288

[^1_82]: https://par.nsf.gov/servlets/purl/10356225

[^1_83]: https://www.sciencedirect.com/science/article/abs/pii/S0925231224014723

[^1_84]: https://arxiv.org/pdf/2505.06978.pdf

[^1_85]: https://www.arxiv.org/pdf/2405.00423v4.pdf

[^1_86]: https://arxiv.org/html/2406.16992v1

[^1_87]: https://arxiv.org/pdf/2309.08297.pdf

[^1_88]: https://www.arxiv.org/pdf/2405.00423v2.pdf

[^1_89]: https://arxiv.org/html/2507.09500v1

[^1_90]: https://arxiv.org/pdf/2601.14053.pdf
