# The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions 

### 1. 핵심 주장 및 주요 기여

**CalMS21 데이터셋의 핵심 주장**[1]

The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions (CalMS21)의 핵심 주장은 **다중 에이전트 행동 모델링 분야에서 자동화된 행동 분류(automated behavior classification)의 발전을 가속화하기 위해서는 대규모 벤치마크 데이터셋이 필수적**이라는 것입니다. 이 논문은 신경과학 분야에서 전통적인 수동 주석 기반 행동 분석의 비효율성을 해결하기 위해 설계되었습니다.[1]

**주요 기여**[1]

- **대규모 주석 데이터셋**: 600만 프레임의 라벨이 없는 추적 자세 데이터와 100만 프레임 이상의 추적 자세 및 프레임 레벨 행동 주석 데이터 제공
- **세 가지 평가 설정**: (1) 단일 주석자로 주석된 대규모 데이터셋에서의 학습, (2) 주석자 간 차이 학습을 위한 스타일 전이, (3) 제한된 학습 데이터로 새로운 행동 학습
- **포괄적 벤치마크**: 다양한 신경망 아키텍처(완전 연결, LSTM, 자기주의, 1D 합성곱)와 최고 성능 방법들에 대한 기준선 평가

***

### 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

**해결하고자 하는 문제**[1]

CalMS21 논문이 해결하고자 하는 핵심 문제는 세 가지입니다:

1. **수동 주석의 비효율성**: 전통적으로 행동 분석은 훈련된 전문가가 프레임 단위로 수행하는 비용 높고 시간 소모적인 작업이었습니다. 초당 30프레임(30Hz)으로 1시간 영상을 분석하려면 약 6시간의 수동 노동이 필요합니다.[1]

2. **일반화 능력 부족**: 기존 행동 분류 모델들은 특정 주석자의 주석 방식에 의존하여 다른 주석자의 데이터로 일반화되지 못합니다. 이는 서로 다른 실험실과 연구자 간의 행동 정의 차이 문제(annotator variability)를 야기합니다.[1]

3. **새로운 행동 학습의 제한**: 연구자들이 특정 관심 행동을 새롭게 학습해야 할 때, 충분한 학습 데이터 없이 정확한 분류 모델을 구축하기 어렵습니다. 이는 소수 샘플 학습(few-shot learning) 문제입니다.[1]

**제안하는 방법**[1]

논문은 **시퀀스-투-시퀀스(sequence-to-sequence) 학습 프레임워크**를 기반으로 합니다. 모델은 28차원의 동적 체포 데이터(7개 keypoints × 2마우스 × 2차원)를 입력받아 각 프레임의 행동 레이블을 예측합니다.[1]

**모델 구조**[1]

기본 모델 아키텍처는 다음과 같습니다:

| 아키텍처 | 특징 | 장점 |
|---------|------|------|
| 완전 연결(Fully Connected) | 간단한 구조 | 빠른 학습 |
| 1D 합성곱(1D Conv Net) | 시간적 특성 추출 | **최고 성능** |
| LSTM | 장기 의존성 학습 | 순차 정보 활용 |
| 자기주의(Self-Attention) | 다중 단계 의존성 | 해석 가능성 |

**성능 결과**[1]

| Task | 모델 | 평균 F1 점수 | MAP |
|------|------|---------|-----|
| Task 1 (고전 분류) | Baseline | 0.793 ± 0.011 | 0.856 ± 0.010 |
| Task 1 | Baseline + Task Programming | 0.829 ± 0.004 | 0.889 ± 0.004 |
| Task 1 | MABe 2021 Top-1 | 0.864 ± 0.011 | 0.914 ± 0.009 |
| Task 2 (주석 스타일 전이) | Baseline | 0.754 ± 0.005 | 0.813 ± 0.003 |
| Task 2 | MABe 2021 Top-1 | 0.809 ± 0.015 | 0.857 ± 0.007 |
| Task 3 (새로운 행동) | Baseline | 0.338 ± 0.004 | 0.317 ± 0.005 |
| Task 3 | MABe 2021 Top-1 | 0.363 ± 0.020 | 0.352 ± 0.023 |

**최고 성능 모델 구조 (MABe 2021 Top-1)**[1]

최고 성능을 달성한 모델은 공간-시간 그래프 합성곱(MS-G3D, Multi-Scale Spatial-Temporal Graph Convolution)을 기반으로 합니다:

- **전처리 단계**: 자아 중심 표현(egocentric representation)으로 변환하고, 키포인트 간 거리, 속도, 각도 기반 특성 계산
- **특성 입력**: 모든 좌표의 쌍별 거리의 PCA 임베딩
- **모델 구조**: 
  - Embedder Network: 인과관계가 없는(non-causal) 1D 합성곱 잔차 블록(residual blocks)
  - Contexter Network: 인과관계가 있는(causal) 1D 합성곱 잔차 블록
  - 최종 분류: 다중 선형 분류 헤드(multiple linear classification heads)
- **손실 함수**: 정규화된 온도 스케일 교차 엔트로피 손실(normalized temperature-scaled cross-entropy loss)을 비라벨 샘플에, 범주형 교차 엔트로피 손실을 라벨 샘플에 적용
- **데이터 증강**: 회전 기반 증강 적용[1]

**한계 및 성능 이슈**[1]

1. **행동 전이 경계에서의 오류**: 모델의 주요 예측 오류가 행동 전이 경계(behavior transition boundaries)에서 집중되었습니다. 이는 부분적으로 인간 주석자의 노이즈에서 비롯될 수 있습니다.[1]

2. **Task 3의 낮은 성능**: 새로운 행동 분류에서 평균 F1 점수가 0.363에 불과하여, 제한된 학습 데이터로 인한 과제의 어려움을 드러냅니다.[1]

3. **클래스 불균형**: Task 3에서 특정 행동의 발생이 매우 드물어(예: Intromission 0.9%, Attack 1.5%) 가중 교차 엔트로피 손실을 사용하여 대응했지만, 여전히 성능 저하가 있습니다.[1]

4. **행동 지속 시간의 영향**: 짧은 행동 발화(short behavior bouts)가 낮은 분류 성능과 상관관계를 보였습니다.[1]

---

### 3. 일반화 성능 향상 가능성

**현재 일반화 능력 분석**[1]

논문은 세 가지 다양한 설정을 통해 일반화 성능을 명시적으로 다룹니다:

1. **Task 2 - 주석자 스타일 전이**: 다섯 명의 서로 다른 주석자 데이터에 대해 모델을 적응시키는 문제입니다. 학습된 주석자 임베딩(annotator embedding)이 주석 방식의 유사성을 포착하여 일반화를 개선합니다.[1]

2. **Task 3 - 새로운 행동 학습**: 7개의 라벨이 지정되지 않은 행동에 대해 매우 제한된 학습 데이터(행동별 단일 비디오)로 새로운 행동 분류기를 학습합니다. 이는 미세 조정된 샷 학습(few-shot learning)의 핵심 과제입니다.[1]

**일반화 향상을 위한 제안 기법**[1]

1. **라벨이 없는 데이터 활용 (Task Programming)**[1]
   - 600만 프레임의 라벨이 없는 자세 데이터를 사용하여 자기 지도 특성 학습(self-supervised feature learning)
   - Task Programming 특성을 기본선에 연결하면 Task 1에서 F1 점수가 0.793에서 0.829로 향상
   - Task 3에서는 0.338에서 0.328로의 변화 (일부 개선)

2. **전이 학습(Transfer Learning)**[1]
   - Task 1에서 사전 학습된 모델을 Task 2와 Task 3의 시작점으로 사용
   - 이를 통해 적응 학습을 위한 초기 특성 표현을 제공

3. **다중 작업 학습(Multi-Task Learning)**[1]
   - MABe Challenge 최고 성능 모델은 세 작업 모두에 대해 공유 매개변수로 단일 모델 학습
   - 마지막 선형 분류 레이어만 작업별로 분리

4. **클래스 불균형 처리**[1]
   - 가중 교차 엔트로피 손실을 적용하여 드문 행동의 가중치를 증가

**최근 연구 기반 향상 방향**[2][3]

최근 동물 행동 분석 분야의 연구는 다음과 같은 향상 방향을 제시합니다:

1. **기초 모델(Foundation Models) 활용**[3]
   - 2024년 연구에 따르면, 대규모 인터넷 데이터로 사전 학습된 비디오 기초 모델(VideoPrism)이 동물 행동 분류에 탁월한 성능을 보임
   - 종 간 일반화, 행동 간 일반화, 다양한 실험 패러다임에서 최소한의 작업별 미세 조정만으로도 효과적
   - 이는 CalMS21의 일반화 문제를 크게 개선할 수 있는 가능성을 시사

2. **자기 지도 학습(Self-Supervised Learning)**[4]
   - SELFEE 같은 자기 지도 특성 추출 방법이 동물 행동의 시간적 역학을 효과적으로 포착
   - 라벨이 없는 데이터에서 구별되는 특성을 학습하여 일반화 능력 향상

3. **그래프 신경망 기반 접근**[1]
   - MS-G3D 모델이 CalMS21에서 최고 성능을 달성
   - 키포인트 간 공간-시간 관계를 그래프 구조로 모델링하여 일반화 능력 증진

4. **메타 학습(Meta-Learning)**[5]
   - Model-Agnostic Meta-Learning (MAML) 기반 프레임워크가 새로운 작업으로의 빠른 적응 가능
   - 소수 샘플 학습 시나리오에서 우수한 성능

***

### 4. 논문의 연구 영향 및 향후 고려사항

**논문의 기여 및 영향**[6][1]

CalMS21 데이터셋과 벤치마크는 다음과 같은 중요한 영향을 미쳤습니다:

1. **커뮤니티 표준 수립**: NeurIPS 2021 Datasets and Benchmarks 트랙에서 발표되어 동물 행동 분석 분야의 벤치마크 데이터셋 표준으로 자리잡음.[1]

2. **MARS(Mouse Action Recognition System)과의 시너지**: CalMS21은 MARS 포즈 추정 시스템과 함께 제공되어, 엔드-투-엔드 행동 분석 파이프라인 구축을 가능하게 함.[6][1]

3. **다양한 응용 분야로의 확장**: 자율주행차, 스포츠 분석, 비디오 게임 등 다중 에이전트 행동 모델링이 필요한 여러 분야에 기여할 수 있는 기초 제공.[1]

4. **신경과학 연구 가속화**: 고처리량 행동 스크리닝을 통해 신경회로 매핑, 약물 개발, 질병 및 장애 연구를 가속화.[1]

**향후 연구 시 고려할 점 (최신 연구 기반)**

1. **기초 모델 통합**[3]
   - 비디오 기초 모델(Video Foundation Models)의 활용이 주목할 만한 경향
   - 제목 특정 미세 조정 없이도 다양한 종(flies, birds, mammals)과 행동 타입에서 일반화
   - CalMS21 재평가 시 기초 모델 기준선 추가 필요

2. **크로스-데이터셋 일반화 연구**[7]
   - ChimpBehave 같은 최근 데이터셋들이 도메인 적응(domain adaptation) 및 크로스-데이터셋 일반화 연구를 중시
   - CalMS21 데이터를 다른 종/환경의 데이터셋과 결합한 일반화 연구 필요

3. **라벨 효율성 개선**[8]
   - Few-shot learning 기반 인간 행동 인식 모델이 급속도로 발전 중
   - 적응형 라벨링 전략과 능동 학습(active learning) 기법 적용 고려

4. **시간 정보의 활용**[9]
   - 모델-불가지론적 메타 학습(MAML) 등 메타 학습 접근법으로 빠른 적응 능력 향상
   - 시계열 분석을 통한 행동 간 과도 현상 개선

5. **해석 가능성 및 신뢰성**[10]
   - 전이 학습 과정에서의 특성 공간 시각화(feature space visualization)와 DeepDream 분석으로 모델 의사결정 과정 이해 필요
   - 민감도 함수(sensitivity functions) 등을 통한 주의 영역 파악

6. **멀티모달 학습**[1]
   - 논문에서 제공하는 영상 데이터를 활용한 엔드-투-엔드 비디오 기반 학습
   - 포즈 데이터와 원본 영상의 결합이 성능을 추가로 향상시킬 수 있는지 검증

7. **주석자 간 일관성 문제**[1]
   - Task 2의 주석자 스타일 전이 연구를 통해 서로 다른 실험실 간 행동 정의 표준화 추진
   - 크라우드소싱된 라벨을 전문가 라벨 스타일로 변환하는 기법 개발

8. **희귀 행동 감지**[11][1]
   - Task 3에서 드문 행동의 낮은 성능 문제 해결을 위한 언더샘플링/오버샘플링 전략
   - 이상 탐지(anomaly detection) 관점에서의 접근

**최신 기술 트렌드와의 맥락**[12][13]

2020-2024년 동물 행동 분석 분야의 발전을 보면:

- 자기 지도 학습이 라벨이 없는 데이터를 효율적으로 활용하는 핵심 기법으로 부상
- 신경망 표현의 기하학적 구조(geometric properties)가 소수 샷 학습 성능을 예측할 수 있다는 이론적 발전
- 동물 행동 분석이 신경회로와의 연결을 통해 뇌 과학의 근본적인 질문에 답하는 도구로 진화

이러한 최근 연구 동향들은 CalMS21의 강점(다양한 데이터 크기, 여러 작업 설정)과 결합될 때, 동물 행동 분석 분야의 진정한 돌파구를 만들 수 있을 것으로 예상됩니다.

***

### 결론

CalMS21 데이터셋은 동물 행동 분석 분야에서 **"컴퓨터 비전과 기계 학습의 표준화된 벤치마크"로서의 역할**을 확립했습니다. 특히 세 가지 구분된 작업 설정을 통해 일반화, 소수 샷 학습, 도메인 적응 등 현대 기계 학습의 중요한 과제들을 직시합니다.[1]

향후 이 데이터셋의 활용이 극대화되려면, **기초 모델 기반 접근법의 통합**, **크로스-데이터셋 일반화 연구 강화**, 그리고 **주석자 간 일관성 개선을 통한 산업 표준화**가 필수적입니다. 특히 자율주행차, 생태 모니터링, 임상 신경학 등 다양한 응용 분야에서의 활용성을 높이기 위해서는, 본 연구에서 제기된 일반화 문제를 최신 기술과 결합하여 해결하는 것이 향후의 중요한 과제가 될 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bef1730d-5a11-4cc1-871f-f52346e12c1d/NeurIPS-Datasets-and-Benchmarks-2021-the-multi-agent-behavior-dataset-mouse-dyadic-social-interactions-Paper-round1.pdf)
[2](https://arxiv.org/abs/2409.15383)
[3](https://www.biorxiv.org/content/10.1101/2024.07.30.605655v1.full-text)
[4](https://elifesciences.org/articles/76218)
[5](https://ieeexplore.ieee.org/document/10382174/)
[6](https://elifesciences.org/articles/63720)
[7](https://arxiv.org/html/2405.20025v1)
[8](https://www.sciencedirect.com/science/article/abs/pii/S0747563223003898)
[9](https://pmc.ncbi.nlm.nih.gov/articles/PMC11654173/)
[10](https://ieeexplore.ieee.org/document/10960173/)
[11](https://www.ingentaconnect.com/content/10.3397/IN_2024_4043)
[12](https://pmc.ncbi.nlm.nih.gov/articles/PMC7780298/)
[13](https://pmc.ncbi.nlm.nih.gov/articles/PMC10949956/)
[14](https://academic.oup.com/biomethods/article/doi/10.1093/biomethods/bpae080/7903126)
[15](https://www.semanticscholar.org/paper/ddd8734267db682f7925516a59de4785b424f622)
[16](https://aacrjournals.org/clincancerres/article/30/21_Supplement/PR019/749437/Abstract-PR019-Transfer-learning-for-accurate)
[17](https://www.ewadirect.com/proceedings/ace/article/view/16710)
[18](http://www.emerald.com/ilt/article/77/2/211-218/1239749)
[19](http://biorxiv.org/lookup/doi/10.1101/2024.10.15.616846)
[20](https://www.science.org/doi/pdf/10.1126/sciadv.adf8068?download=true)
[21](https://arxiv.org/pdf/2111.12295.pdf)
[22](https://pmc.ncbi.nlm.nih.gov/articles/PMC10088092/)
[23](https://www.pnas.org/doi/10.1073/pnas.1515982112)
[24](https://www.pnas.org/doi/10.1073/pnas.2200800119)
[25](https://www.sciencedirect.com/science/article/pii/S0957417425019499)
[26](https://arxiv.org/pdf/2301.01047.pdf)
[27](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.14294)
[28](https://pubmed.ncbi.nlm.nih.gov/34048344/)
[29](https://revistas.usal.es/cinco/index.php/2255-2863/article/view/31638/30838)
