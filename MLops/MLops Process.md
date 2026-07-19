데이터 과학자에게 ML 파이프라인이 "모델을 만드는 코드의 흐름"이라면, MLOps는 "그 코드가 살아 움직이는 공장을 짓는 일"입니다. 파이프라인 구축 경험을 바탕으로 MLOps의 본질을 명확히 설명해보고자 합니다.

------------------------------
## 1. MLOps 정의가 왜 필요한가?

많은 데이터 과학자가 .ipynb 파일이나 단일 파이프라인 스크립트를 완성하면 일이 끝났다고 생각합니다.  
하지만 실제 비즈니스 환경에서는 모델을 배포한 순간부터 문제가 시작됩니다. [1] 

데이터는 계속 변하고(Data Drift), 모델의 예측력은 시간이 지나며 떨어집니다(Model Decay).  
MLOps의 정의를 정확히 이해해야 내가 만든 파이프라인이 실험실을 벗어나 실제 서비스에서 '지속 가능하게' 작동하는 구조를 설계할 수 있습니다.

## 2. MLOps의 핵심 메커니즘 (정의~증명)

### 정의 (Definition)

MLOps = f(ML Pipeline) × CI/CD/CM

MLOps는 데이터 수집부터 모델 배포까지의 ML 파이프라인(코드/데이터/모델)에 소프트웨어 공학의 지속적 통합(CI), 지속적 배포(CD), 지속적 모니터링(CM)을 결합하여, 시스템을 자동화하고 안정적으로 운영하는 문화이자 엔지니어링 체계입니다.

### 예시 (Example)
추천 시스템 파이프라인을 구축한 상황을 가정해 봅시다.

* MLOps 환경: 사용자의 클릭 로그 데이터가 유입되면, 시스템이 자동으로 데이터 드리프트를 감지합니다. 성능이 기준치 이하로 떨어지면 자동으로 학습 파이프라인이 트리거되어 새 모델을 만들고, A/B 테스트를 거쳐 안전하게 배포됩니다. 데이터 과학자는 이 과정에 개입하지 않고 모니터링 대시보드만 확인합니다. [2] 

### 반례 (Counter-Example)
ML 파이프라인은 훌륭하지만 MLOps가 없는 상황입니다.

* 전형적인 실패 사례: 주피터 노트북으로 완벽한 추천 파이프라인 코드를 짰습니다. 배포를 위해 Flask API로 감싸 서버에 올렸습니다. 한 달 뒤, 트렌드가 바뀌어 모델 성능이 폭락했습니다. 데이터 과학자는 다시 과거 데이터를 수동으로 긁어모아, 노트북을 켜고, 하이퍼파라미터를 튜닝한 뒤, 엔지니어에게 파일(.pkl)을 넘겨주며 재배포를 부탁합니다. [3, 4] 

### 정리 (Synthesis)

* ML 파이프라인: 데이터 → 전처리 → 학습 → 평가를 수행하는 단방향 선형 프로세스 (Static)
* MLOps: 파이프라인 전체를 유기적으로 연결하여 무한히 순환시키는 지속 가능한 피드백 루프 체계 (Dynamic)

### 증명 개요 (Proof Outline)

명제: "ML 파이프라인의 완성은 안정적인 서비스 운영을 보장하지 않는다."

* 가정: 완벽한 ML 파이프라인 P가 존재하여 시간 t₀에 정확도 95%를 달성함.
* 변수: 현실 세계의 데이터 분포 D는 시간 t에 따라 $D_t$로 변화함 ($D_{t_0} \neq D_{t_1}$).
* 현상: 고정된 파이프라인 P로 학습된 모델은 t₁ 시점에 입력된 $D_{t_1}$에 대해 오차가 증가함 (Concept Drift 발생).
* 결론: 따라서 시간에 따른 성능 저하를 막기 위해서는 P의 외부에서 $D_t$를 감시하고 P를 재실행(Retraining)하는 상위 제어 시스템(MLOps)이 필수적으로 요구됨.

------------------------------

## 3. ML 파이프라인 vs MLOps 차이점 비교
두 개념의 경계를 데이터 과학자의 언어로 직관적으로 비교합니다.

| 비교 항목 | ML 파이프라인 (ML Pipeline) | MLOps |
|---|---|---|
| 핵심 질문 | "이 데이터로 어떻게 모델을 만들 것인가?" | "이 모델을 어떻게 365일 지탱할 것인가?" |
| 중심 객체 | 데이터(Data), 알고리즘(Algorithm) | 시스템(System), 인프라(Infrastructure) |
| 작업 단위 | 단일 실험 수행 및 유효성 검증 | 파이프라인 자체의 버전 관리 및 자동화 |
| 종료 시점 | 만족하는 성능의 모델 파일(.pkl) 생성 | 서비스가 종료되거나 완전히 폐기될 때 |
| 실패 정의 | Loss가 줄어들지 않거나 과적합 발생 | API 서빙 지연(Latency), 서버 다운, 데이터 누수 |

------------------------------
## 4. 흔한 오해 (Common Misconceptions)

* 오해 1: "쿠버네티스(Kubernetes)나 MLflow를 쓰면 MLOps를 하는 것이다?"
* 진실: 도구는 수단일 뿐입니다. 자동화된 피드백 루프와 협업 프로세스가 없다면, 무겁고 비싼 도구를 다루는 '인프라 삽질'에 불과합니다. 엑셀과 파이썬 스크립트만으로도 자동 재학습 루프를 만들었다면 그것이 MLOps의 시작입니다.
* 오해 2: "MLOps는 데이터 엔지니어나 DevOps 담당자만 하는 일이다?"
* 진실: 모델의 성능 하락 요인(Drift)을 정의하고, 어떤 시점에 재학습을 해야 하는지(Trigger Condition) 결정하는 것은 데이터 과학자의 몫입니다. 데이터 과학자가 MLOps를 모르면 파이프라인 설계 단계에서 '지속 가능성'을 고려하지 못합니다.
* 오해 3: "파이프라인이 Airflow로 스케줄링되어 있으면 MLOps 레벨이다?"
* 진실: 매일 밤 12시에 단순히 배치를 돌리는 것은 기초적인 단계(Level 0~1)입니다. 진정한 MLOps는 모델의 '실제 운영 성능'을 실시간 감시하고, 이상 징후에 따라 동적으로 반응하는 체계를 뜻합니다.

------------------------------
## 5. 개념 의존성 지도 (Conceptual Dependency Map)
MLOps를 이해하기 위해 선행되어야 하는 개념과 그 확장의 흐름입니다. 화살표(→)는 "을(를) 알아야 다음 단계로 나아갈 수 있음"을 뜻합니다.

```
[데이터 파이프라인] (ETL, Feature Engineering)
       │
       ▼
[ML 파이프라인] (Modeling, Validation, Artifact 생성)
       │
       ├──────────────────────────────┐
       ▼                              ▼
[CI / CD (소프트웨어 공학)]      [지속적 모니터링 (CM)]
(Code, Data, Model 버전 관리)   (Data/Concept Drift 감지)
       │                              │
       └──────────────┬───────────────┘
                      │
                      ▼
               🌱 [MLOps 완성]
       (자동화된 지속적 재학습 & 배포)

------------------------------
```

[1] [https://docs.cloud.google.com](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?hl=ko)
[2] [https://www.jaenung.net](https://www.jaenung.net/tree/25990)
[3] [https://www.backend.ai](https://www.backend.ai/ko/blog/2022-01-MLOps)
[4] [https://gguguk.github.io](https://gguguk.github.io/posts/MLOps/)
[5] [https://www.makinarocks.ai](https://www.makinarocks.ai/blog/mlops-the-foundation-for-scalable-and-reliable-ai/)

본 분석은 2026년 7월 기준 공개된 학술 논문(arXiv, ScienceDirect), 글로벌 테크 기업(Google Cloud, Microsoft Azure)의 공식 아키텍처 가이드라인, 그리고 2020년 이후 정립된 최신 MLOps 연구 문헌을 기반으로 수행되었습니다.  
데이터 과학자가 겪는 'MLOps 단계별 과정의 직관적·엄밀한 이해 부족' 문제를 해결하기 위해 엔지니어링 표준 프레임워크를 적용하여 고해상도로 검증한 결과입니다. [1, 2, 3, 4] 

------------------------------

## 1. 문제의 재정의 (Problem Redefinition)

* 현재 상태(As-Is): 대상자는 데이터 과학 파이프라인(실험용 스크립트, 데이터 전처리, 모델 모델링 및 평가)의 선형 구축 방식은 직관적으로 알고 있으나, 이를 서비스 환경에서 '지속 가능하고 자동화된 비선형적 루프(MLOps)'로 전환하는 구체적 아키텍처 단계와 트리거 작동 메커니즘을 명확히 식별하지 못하고 있음.
* 목표 상태(To-Be): 데이터 과학 지식을 기반으로 MLOps의 단계별 상호작용(의존성)과 진화 단계(Maturity Levels)를 엄밀히 분리하고, 실제 현업에 즉시 적용할 수 있는 엔지니어링 전이 모델을 확립함. [4] 

------------------------------
## 2. 필요한 가정 및 부족한 정보 (Assumptions & Missing Information)

### 가정한 사항 (Assumptions)

* 대상자는 이미 코드 버전 관리(Git)와 로컬 환경에서의 머신러닝 개발 라이프사이클(Scikit-learn, PyTorch 등)에 숙련되어 있음.
* 가장 표준적인 정형/비정형 비즈니스 실시간 예측 인프라를 타깃으로 삼음.

## 부족한 정보 (Missing Information)

* 인프라 종속성: 인하우스 온프레미스 쿠버네티스 환경 기반인지, 글로벌 클라우드 CSP 환경([Google Cloud Vertex AI](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning), [Azure ML](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/mlops-maturity-model-with-azure-machine-learning/3520625))을 채택 중인지에 대한 구체적 제약 조건이 누락됨. [1, 5] 
* 비즈니스 유형: 데이터 스트리밍 속도(실시간 Online vs 배치 Batch)에 따라 MLOps 아키텍처 구축의 우선순위 단계가 달라지나, 타깃 도메인이 명시되지 않음. [3] 

------------------------------

## 3. 가능한 접근법 비교 및 선택 (Approach Comparison & Selection)
단계별 과정을 완벽하게 규명하기 위해 다음 세 가지 접근론을 검토하였습니다.

| 분석 접근법 | 특징 | 장점 | 단점 |
|---|---|---|---|
| A. 오픈소스 도구 중심 접근 (MLflow, Kubeflow 기반) | 특정 오픈소스 도구 스택의 API 단계에 맞추어 과정을 분류함. | 즉각적인 실무 코드 구현이 가능함. | 도구 의존성이 높아 개념의 본질이 희석되고 트렌드 변화에 취약함. |
| B. 학술적 생애주기 접근 (Lifecycle Reference Models) | 데이터 공학, 알고리즘, 운영계의 결합을 7-8단계 이론으로 분리. | 매우 학술적이며 프로세스의 경계가 엄격함. | 데이터 과학자가 한눈에 아키텍처 흐름을 연동하기 복잡함. |
| C. 프레임워크 성숙도 모델 접근 (Google/Microsoft 융합식 표준) | 자동화 수준(Level 0, 1, 2)에 맞춘 단계별 컴포넌트 확장 방식. | 현재 조직의 수준 진단과 함께 단계별 필요 요소가 유기적으로 증명됨. | 초기 인프라 개념의 이해가 요구됨. |


* 선택된 접근법: 'C. 프레임워크 성숙도 모델 접근'을 메인 프레임으로 삼고, 'B'의 컴포넌트 생애주기를 상세 결합하는 하이브리드 방식을 채택합니다. 데이터 과학자 입장에서 자신의 파이프라인이 어떤 단계를 거쳐 완전 자동화(Level 2)로 전이되는지 증명하는 데 가장 적합하기 때문입니다. [1] 

------------------------------
## 4. 결론을 지지하는 논리와 반례 검토 (Logic & Counter-Examples)

### 핵심 결론

MLOps의 단계별 과정은 '정적 파이프라인의 오케스트레이션(Level 1)'을 거쳐 'CI/CD 파이프라인 자체의 자동화(Level 2)' 단계로 전전(Evolution)해야만 완성된다. [1] 

### 지지 논리
Google Cloud Architecture Framework (2024년 8월 업데이트 버전) 및 최신 MLOps 실무 가이드라인에 따르면, MLOps는 단순히 모델 배포 자동화가 아닌 "파이프라인 자체를 자동으로 빌드하고 배포하는 시스템"으로 정의됩니다. 모델의 성능 저하는 데이터의 동적 변화(Data/Concept Drift)로 인해 필연적으로 유발되므로, 수동 개입 없는 재학습 파이프라인 구축(Level 1)과 이를 지속해서 통합·배포하는 환경(Level 2)의 단계적 정립이 당위성을 갖습니다. [1] 

### 반례 가능성 및 검토

* 반례 주장: "LLM(대형 언어 모델) 시대나 파인튜닝 비용이 수억 원에 달하는 무거운 모델의 경우, Level 1~2 수준의 자동 재학습 루프(CT)를 돌리는 것이 오히려 비용 효율성 측면에서 악영향을 준다." [3] 
* 반례 기각/수용 분석: 이 반례는 타당합니다. 따라서 2024-2026년 최신 Microsoft AI 워크로드 아키텍처 문헌(GenAIOps 확장형 가이드라인)에서는 모델의 크기와 아키텍처 특성에 따라 '자동 재학습(CT)' 대신 '실시간 검색 증강 생성(RAG) 파이프라인의 데이터 저장소(Vector DB) 업데이트 및 프롬프트 가드레일 모니터링'으로 MLOps 3단계 세부 프로세스를 대체 정의하고 있습니다. 즉, 본질인 '피드백 루프 자동화'는 유지되되 세부 물리 단계는 모델 특성에 맞춰 변형되어야 합니다. [3, 6] 

------------------------------
## 5. "2020년 이후 관련 최신 연구 비교 분석" (Research Review)

* Google Cloud Whitepaper & Architecture Docs (2020 ~ 2024.08 개정):
* MLOps를 세 수준으로 엄격하게 격리함. Level 0(수동 프로세스), Level 1(ML 파이프라인 자동화/지속적 학습), Level 2(CI/CD 파이프라인 자동화).
   * 핵심 컴포넌트로 Feature Store, Model Registry, ML Metadata Store의 유기적 결합 단계를 명시함. [1, 7, 8, 9] 
* Microsoft Azure MLOps Maturity Model (2022 ~ 2024.11 최신화):
* 조직적/인프라 관점에서 0단계(No DevOps)부터 4단계(Full MLOps)까지로 세분화하여 표현함.
   * Google이 기술 인프라 파이프라인 중심이라면, Azure 프레임워크는 데이터 팀과 운영 팀 간의 거버넌스(Governance), 모델 보안(Security), 그리고 2024년 이후 부각된 GenAIOps(생성형 AI 운영)와의 결합 단계를 강조하여 충돌하기보다 상호 보완적인 관점을 제시함. [3, 5, 6, 10] 
* 학술 문헌 분석 (Faubel et al., 2025 / ArXiv, 2025-2026):
* 최신 연구([Navigating MLOps, 2025](https://arxiv.org/html/2503.15577v1))에 따르면 MLOps 단계 구축 실패의 70% 이상이 기술 도구의 부재가 아닌, '인프라 복잡도의 추상화 실패'와 '실시간 관측 가능성(Observability) 체계의 미비'에서 발생함을 정량적/정성적으로 규명함. [8, 11, 12] 

------------------------------
## 6. 결과: MLOps 3대 핵심 진화 단계 및 세부 컴포넌트 과정
데이터 과학자가 알고 있는 단일 파이프라인이 MLOps 표준 프레임워크 내에서 어떻게 고도화되는지 보여주는 최종 종합 결과입니다. [1] 

### 1단계: MLOps Level 0 (수동 파이프라인 및 모델 배포) [10] 

* 특징: 모든 단계가 수동(Manual)으로 이루어짐. 데이터 과학자가 분석용 노트북에서 스크립트를 실행해 파이프라인을 구동함. [8] 
* 상세 과정:
1. 데이터 추출 및 전처리 (수동 스크립트)
   2. 모델 학습 및 검증 (.ipynb 실행)
   3. 생성된 모델 아티팩트(예: .pkl)를 엔지니어에게 수동 전달
   4. REST API 등의 형태로 서빙 환경에 정적 배포
* 한계: 데이터와 모델 간의 버전 추적이 어렵고, 모니터링이 부재하여 성능 저하(Decay)에 즉각 대응 불가. [8, 9] 

### 2단계: MLOps Level 1 (ML 파이프라인 자동화 - 지속적 학습, CT)

* 특징: 새로운 데이터가 유입될 때 모델을 자동으로 재학습(Continuous Training)하는 파이프라인 오케스트레이션 단계. [3] 
* 상세 과정:
1. 자동화된 데이터 트리거: 새 데이터 유입 혹은 모니터링 시스템의 유효성 검증 경보 발령.
   2. 오케스트레이션 실행: Apache Airflow, Kubeflow Pipelines 같은 도구가 선형 ML 파이프라인의 각 컴포넌트(전처리->학습->평가)를 순차적으로 자동 실행.
   3. Feature Store 연동: 피처의 정합성과 재사용성을 보장하기 위해 중앙 저장소에서 데이터를 호출.
   4. Model Registry 등록: 검증을 통과한 우수한 성능의 모델 아티팩트를 메타데이터(학습 일시, 하이퍼파라미터 등)와 함께 중앙 저장소에 자동 등록. [1, 3, 5] 

### 3단계: MLOps Level 2 (CI/CD 파이프라인 자동화) [1] 

* 특징: 모델뿐만 아니라, "ML 파이프라인 코드 자체"를 안전하고 지속해서 테스트하여 운영 환경에 배포하는 자동화의 최종 단계. [1] 
* 상세 과정:

1. 지속적 통합 (CI): 데이터 과학자가 파이프라인 코드를 수정하여 레포지토리에 푸시하면, 데이터 유효성 테스트 및 유닛 테스트가 자동으로 실행됨.

2. 지속적 배포 (CD): CI를 통과한 새로운 파이프라인 패키지가 대상 환경(Target Infrastructure)에 자동으로 배포되어 새로운 '지속적 학습(CT) 루프'를 구성함.

3. 지속적 모니터링 (CM): 프로덕션 환경의 실시간 로그를 수집하여 데이터 예측 값의 분포 변화, 시스템 레이턴시(Latency)를 추적하고 Level 1의 CT 트리거로 신호를 전송함. [1, 3, 13] 

------------------------------
## 7. 검증 (Verification)

* 사실 검증 (Fact Check): Google Cloud 공식 아키텍처 문서 표준 가이던스에 명시된 MLOps 컴포넌트 구성 요소(Feature Store, Model Registry, ML Metadata Store, Pipeline Orchestrator)의 배치 순서와 기능적 정의가 완벽히 일치함을 대조·검증 완료하였습니다. [1] 
* 계산 및 코드 실행 여부: 본 분석은 순수 엔지니어링 개념 아키텍처 설계 프레임워크 및 방법론을 다루므로 수치 계산 및 소스 코드 빌드 테스트는 수행하지 않았습니다(해당 사항 없음).

------------------------------
## 8. 한계 (Limitations)

* 구체적 도구 명세의 부재: 본 아키텍처 표준 프레임워크는 벤더 중립적(Vendor-agnostic) 관점에서 기술되었기 때문에, 기업의 내부 예산, 기존 레거시 인프라 상황에 따른 최적의 오픈소스/클라우드 솔루션 조합(예: BentoML vs Triton, MLflow vs Weights & Biases)의 구체적인 비용 최적화 산식은 제공하지 못합니다. [14] 
* 에지(Edge) 인프라 예외: 클라우드 인프라가 아닌 디바이스 내부(On-device AI)에 모델을 최적화하여 배포하는 특수 MLOps 단계를 커버하기에는 경량화(Quantization) 및 하드웨어 컴파일 단계에 대한 설명이 간소화되어 있습니다.

------------------------------
## 9. 다음 의사결정 (Next Decisions)
조직의 MLOps 전환을 위해 가장 먼저 결정해야 할 구체적 액션 아이템입니다.

* 현재 아키텍처 성숙도 자가 진단: 현재 운영 중인 모델 파이프라인이 완벽한 Level 0에 머물러 있는지, 혹은 Airflow 등으로 배치 자동화가 되어 있는 Level 1의 초기 단계인지를 명확히 규정해야 합니다.
* 핵심 지표(Trigger Metric) 수립: Level 1(지속적 학습)로 넘어가기 위해 모델의 재학습을 유도할 '데이터 드리프트(PSI, KL-Divergence)'의 임계치 기준과 모니터링 주기를 통계적으로 산출하는 작업을 데이터 과학 팀 내에서 선제적으로 논의 및 의결해야 합니다. [9, 13] 


[1] [https://docs.cloud.google.com](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
[2] [https://www.sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S2444569X24001768)
[3] [https://www.smiit.de](https://www.smiit.de/en/blog/mlops-with-microsoft-azure/)
[4] [https://www.databricks.com](https://www.databricks.com/blog/what-is-mlops)
[5] [https://techcommunity.microsoft.com](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/mlops-maturity-model-with-azure-machine-learning/3520625)
[6] [https://learn.microsoft.com](https://learn.microsoft.com/en-us/azure/well-architected/ai/mlops-genaiops)
[7] [https://services.google.com](https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf)
[8] [https://arxiv.org](https://arxiv.org/html/2503.15577v1)
[9] [https://www.zenml.io](https://www.zenml.io/blog/everything-you-ever-wanted-to-know-about-mlops-maturity-models)
[10] [https://stephenweixu.com](https://stephenweixu.com/blog/an-introduction-to-mlops-maturity-level)
[11] [https://arxiv.org](https://arxiv.org/pdf/2510.20590)
[12] [https://arxiv.org](https://arxiv.org/html/2406.09737v1)
[13] [https://dev.to](https://dev.to/apprecode/mlops-workflow-from-definition-to-production-ready-pipelines-1lbl)
[14] [https://www.youtube.com](https://www.youtube.com/watch?v=lZnSq61e0wk)

