MLOps를 체계적으로 설명하기 위해 권위 있는 정의와 단계별 구분을 확인하겠습니다.

# MLOps를 데이터 과학의 언어로 정의하기

당신은 이미 "데이터 → 전처리 → 피처엔지니어링 → 학습 → 평가" 로 이어지는 **ML 파이프라인**을 구축할 줄 안다. 이 답변은 그 파이프라인이라는 개념 위에 MLOps라는 상위 구조를 수학적 정의처럼 얹는 방식으로 설명한다.

---

## 0. 왜 이 개념이 필요한가 (동기)

데이터 과학자가 오프라인 홀드아웃 데이터셋에서 좋은 성능의 모델을 학습시키는 것은 사실 전체 문제의 일부에 불과하다. 데이터 과학자는 자신의 사용 사례에 관련된 학습 데이터가 주어지면 오프라인 홀드아웃 데이터셋에서 예측 성능을 갖는 ML 모델을 구현하고 학습시킬 수 있다. 그러나 진짜 도전은 ML 모델을 만드는 것이 아니라 통합된 ML 시스템을 구축하고 이를 프로덕션에서 지속적으로 운영하는 것이다. 즉, 파이프라인은 "모델을 만드는 함수" $f_\theta$ 를 만드는 절차이지만, 실제 세계에서는 $\theta$가 만들어진 그 시점의 데이터 분포 $P(X,Y)$가 시간에 따라 변한다는 문제(covariate/label shift)가 남는다.

이 문제를 방치하면 발생하는 위험은 명확하다. 수동 프로세스는 ML 생애주기에서 오류와 불일치를 유발하여 모델의 정확도와 신뢰성에 영향을 줄 수 있고, 모델과 데이터셋이 커질수록 확장이 어려워지며, 개발과 배포 속도가 느려지고, 데이터 과학자·엔지니어·운영팀 간 협업이 어려워져 사일로와 소통 단절이 생긴다. MLOps는 바로 이 "파이프라인 이후"의 공백—배포, 재학습, 모니터링, 협업—을 구조적으로 메우기 위해 필요하다.

---

## 1. 정의 (Definition)

> **MLOps** := ML 시스템 개발(Dev)과 ML 시스템 운영(Ops)을 통합하는 것을 목표로 하는 ML 엔지니어링 문화이자 실천 방법론이다. MLOps를 실천한다는 것은 통합(integration), 테스트, 릴리스, 배포, 인프라 관리를 포함한 ML 시스템 구축의 모든 단계에서 자동화와 모니터링을 지향한다는 것을 의미한다.

이를 데이터 과학의 언어로 다시 쓰면:

- ML 파이프라인이 함수 $f: D \to M$ (데이터 $D$에서 모델 $M$을 산출하는 사상)이라면,
- MLOps는 이 함수 $f$ 자체를 **버전 관리·검증·자동 트리거·모니터링이 가능한 하나의 재현 가능한 시스템**으로 승격시키고, 그 산출물 $M$을 서빙 시스템에 안전하게 배포·감시·재학습하는 **상위 제어 루프(control loop)**다.

핵심 구성요소는 세 가지 "연속성(Continuous-)" 속성으로 요약된다:

- 지속적 배포(CD)는 단일 소프트웨어 패키지나 서비스에 관한 것이 아니라, 다른 서비스(모델 예측 서비스)를 자동으로 배포해야 하는 시스템(ML 학습 파이프라인)에 관한 것이다.
- 지속적 학습(CT)은 ML 시스템에 고유한 새로운 속성으로, 테스트와 서빙을 위한 후보 모델을 자동으로 재학습하는 것과 관련이 있다.
- 지속적 모니터링(CM)은 프로덕션 시스템의 오류를 잡아내는 것뿐 아니라, 프로덕션 추론 데이터와 비즈니스 성과와 연결된 모델 성능 지표를 모니터링하는 것이다.

---

## 2. 예시 (Example)

가상의 추천 시스템을 생각해보자.

1. 데이터 과학자가 파이프라인(전처리→피처→학습→평가)을 만들어 오프라인 AUC 0.85 모델을 얻음.
2. 이 파이프라인 자체가 **컨테이너화되어 Git에 버전관리**되고, 매일 새 클릭 로그가 들어오면 **자동으로 트리거**되어 재학습(CT)됨. 재학습으로 최신 정보를 학습한 새로운 후보 모델이 나오면, 이는 중요한 품질 게이트를 거쳐야 한다. 단순히 재학습한다고 더 나은 모델이 되는 것은 아니기 때문에, 새 모델은 두 모델 모두 본 적 없는 홀드아웃 테스트셋으로 현재 배포된 모델과 엄격히 비교되어야 하며, 통계적으로 유의한 개선이 없으면 파이프라인은 중단되어 열등한 모델이 프로덕션에 올라가는 것을 막는다.
3. 통과한 모델은 버전 관리되어 모델 레지스트리에 저장되고, 이 레지스트리는 학습된 모든 모델의 중앙 인벤토리 역할을 하며 CD 파이프라인이 이를 픽업해 배포한다.
4. 배포 후에도 실시간 클릭률(CTR) 지표를 모니터링(CM)하다가 성능이 임계값 이하로 떨어지면 모니터링 시스템이 모델 성능 지표(정확도나 F1-score 등)가 사전 정의된 임계값 아래로 떨어졌음을 감지하는 성능 기반 트리거가 재학습 파이프라인을 다시 발동시킨다.

이 전체 순환 구조(파이프라인 + 트리거 + 검증 + 레지스트리 + 모니터링 + CD)가 MLOps다. 파이프라인은 이 중 "1번" 하나의 컴포넌트에 불과하다.

---

## 3. 반례 (Counterexample) — 이것은 MLOps가 아니다

Jupyter 노트북에서 데이터 과학자가 매주 수동으로 스크립트를 실행해 모델을 재학습시키고, 결과 pickle 파일을 엔지니어에게 이메일로 넘겨 수동 배포하는 경우를 생각해보자. 이것이 바로 "MLOps 레벨 0"이다.

레벨 0 프로세스의 특징은: 수동적이고 스크립트 주도적이며 대화형인 프로세스로, 데이터 분석, 데이터 준비, 모델 학습, 검증을 포함한 모든 단계가 수동이다. 또한 데이터 과학자는 학습된 모델을 아티팩트로 넘기고 엔지니어링 팀이 API 인프라에 배포하는 방식으로, 모델을 만드는 사람과 배포하는 사람이 분리되어 있다.

여기서는 파이프라인(전처리→학습→평가)은 분명 존재하지만, 자동 트리거·CI/CD·모니터링·재학습 루프가 전혀 없다. 즉 **"파이프라인이 있다" ≠ "MLOps를 한다"**는 것을 보여주는 반례다.

---

## 4. 정리 (Theorem)

> **정리**: MLOps는 ML 파이프라인을 필요조건으로 포함하지만, ML 파이프라인의 존재만으로는 MLOps가 성립하기에 충분하지 않다.
>
> $$\text{MLOps} \;\Rightarrow\; \exists\, \text{ML Pipeline}, \qquad \text{ML Pipeline} \;\not\Rightarrow\; \text{MLOps}$$

즉 ML 파이프라인은 MLOps의 **부분집합(subset)**이지, 동의어가 아니다.

---

## 5. 증명 개요 (Proof Sketch)

이 정리는 Google의 MLOps 성숙도 단계로 귀납적으로 증명할 수 있다.

**Level 0 (수동)**: 위 반례에서 보였듯, 파이프라인은 존재하나 CT/CD/CM이 전무. 따라서 파이프라인이 있어도 MLOps 정의(자동화+모니터링을 전 단계에 지향)를 만족하지 못함 → 명제의 역이 거짓임을 보임.

**Level 1 (파이프라인 자동화)**: 레벨 1의 목표는 ML 파이프라인을 자동화하여 모델의 지속적 학습을 수행하는 것이며, 이를 통해 모델 예측 서비스의 지속적 배포를 달성한다. 프로덕션에서 새 데이터로 모델을 재학습하는 과정을 자동화하려면, 파이프라인에 자동화된 데이터 및 모델 검증 단계, 파이프라인 트리거, 메타데이터 관리를 도입해야 한다. 여기서 배포 대상 자체가 바뀐다: 레벨 0에서는 학습된 모델을 예측 서비스로 프로덕션에 배포하지만, 레벨 1에서는 학습된 모델을 예측 서비스로 제공하기 위해 자동으로 반복 실행되는 전체 학습 파이프라인을 배포한다. 즉 파이프라인이 "1회성 산출 도구"에서 "지속 운영되는 시스템의 구성요소"로 승격됨 — MLOps 방향으로의 필요조건 충족 단계.

**Level 2 (CI/CD 파이프라인 통합)**: CI/CD 파이프라인 자동화까지 더해지는 단계로, 파이프라인 코드 자체의 테스트·빌드·배포까지 자동화되며, 소스 코드 관리, 테스트·빌드 서비스, 배포 서비스, 모델 레지스트리, 피처 스토어, 메타데이터 스토어, 파이프라인 오케스트레이터 같은 구성요소가 결합해야 비로소 정의상의 "모든 단계에서 자동화와 모니터링"이 충족된다.

∎ 따라서 파이프라인 자동화(CT)는 MLOps의 필요조건이지만, CI/CD·모니터링·레지스트리·거버넌스가 결합되어야 MLOps 정의를 완전히 만족한다. 이는 성숙도가 0→1→2로 갈수록 파이프라인이라는 부분집합 위에 계속 구조가 덧붙는 것으로 증명된다.

---

## 6. MLOps vs ML 파이프라인 — 차이 명확화

| 구분 | ML 파이프라인 | MLOps |
|---|---|---|
| 범위 | 데이터→모델 산출까지의 **오프라인 실험/학습 절차** | 파이프라인을 포함한 **전체 프로덕션 생애주기** (개발+운영) |
| 목적 | 재현 가능한 학습 결과 생성 | 모델을 신뢰성 있고 효율적으로 배포·유지하는 것, 프로덕션 환경에서 신뢰성·확장성·유지보수성을 보장하는 것 |
| 트리거 | 사람이 실행 명령 | 스케줄 기반 또는 성능 저하 감지 기반의 자동 트리거 |
| 산출물 관리 | 로컬 파일/노트북 결과 | 모델 레지스트리, 메타데이터 스토어, 피처 스토어 |
| 배포 대상 | (없음, 혹은 1회성 모델) | 자동·반복 실행되는 학습 파이프라인 자체가 배포 대상 |
| 실패 시 대응 | 사람이 재확인 | 품질 게이트 통과 실패 시 자동으로 파이프라인 중단 |
| 조직적 측면 | 데이터 과학자 단독 작업 가능 | 데이터 과학자와 ML 엔지니어가 DevOps 원칙을 ML 시스템에 적용하는 협업 문화 |

---

## 7. 흔한 오해 정리

**오해 1: "MLOps = 모델을 Docker/API로 배포하는 도구"**
틀렸다. MLOps는 도구가 아니라 ML 시스템 구축의 모든 단계(통합, 테스트, 릴리스, 배포, 인프라 관리)에서 자동화와 모니터링을 지향하는 엔지니어링 문화이자 실천법이다. 배포는 그 일부일 뿐이다.

**오해 2: "파이프라인을 자동화(스케줄러로 크론잡)하면 MLOps 끝"**
이는 Level 1에 불과하다. 현재 MLOps 모범사례 발전 단계에서 레벨 2는 여전히 다소 사변적이지만, 파이프라인 자체를 빠르고 안정적으로 갱신하기 위한 아키텍처로, 데이터 과학자와 ML 엔지니어가 피처 엔지니어링, 모델 아키텍처, 하이퍼파라미터에 대한 새로운 아이디어를 빠르게 탐색할 수 있도록 견고한 자동화된 CI/CD가 필요하다. 모니터링·CI/CD 없이는 미완성이다.

**오해 3: "MLflow/Kubeflow 같은 툴만 도입하면 MLOps가 된다"**
실제로는 도구를 많이 쓸수록 성숙도가 올라간다고 믿는 경향이 있지만, 검증 실무자들은 인프라·도구가 프레임워크 단계와 일치하지 않는 경우가 많다고 지적한다. 문화·프로세스·조직 정렬이 툴보다 우선이다.

**오해 4: "성숙도는 한 번에 하나의 레벨로만 고정된다"**
실제로 조직은 동시에 여러 레벨의 특성을 보이는 경우가 많으며, 레벨 간 진행은 이산적 단계가 아니라 능력이 겹치고 점진적으로 발전하는 연속체로 다뤄야 한다.

**오해 5: "MLOps 성숙도는 단일 사다리 하나로 측정된다"**
MLOps 성숙도는 단일 사다리가 아니라 데이터, 모델, 코드라는 여러 축에 걸친 성숙도의 조합이며, 대부분의 팀은 세 축 모두에서 같은 레벨에 있지 않다.

**오해 6: "MLOps는 그냥 ML에 적용한 DevOps다, 특별할 게 없다"**
CI/CD는 ML 시스템에 고유한 문제를 완전히 다루지 못한다. 전통적 소프트웨어와 달리 ML 모델의 성능은 정적이지 않고 마주치는 데이터가 변함에 따라 저하될 수 있으며, 이를 해결하는 지속적 학습(CT)이 필요하다는 점이 DevOps와의 결정적 차이다.

---

## 8. 개념 의존성 지도

```
                     DevOps 원칙 (CI/CD 문화)
                              │
                              ▼
                     ML 시스템에 적용 ── (당신이 아는) 데이터 사이언스 ML 파이프라인
                              │                     │
                              ▼                     ▼
                        ┌───────────── MLOps ─────────────┐
                        │   (Dev + Ops 통합, 문화/실천)      │
                        └──────┬──────────┬──────────┬─────┘
                               ▼          ▼          ▼
                             CI/CD        CT         CM
                        (파이프라인 코드   (자동 재학습   (프로덕션 성능·
                         테스트/빌드/배포)  루프 = 당신의   데이터 드리프트
                               │          파이프라인이    모니터링)
                               │          핵심 구성요소)      │
                               ▼               │             ▼
                        모델 레지스트리 ◄───────┘        트리거(스케줄/
                        피처 스토어                        성능저하 기반)
                        메타데이터 스토어                       │
                               │                              ▼
                               └──────────────► CT 파이프라인 재실행 (순환)
                                        │
                                        ▼
                              MLOps 성숙도 (Level 0→1→2, 다차원: 데이터/모델/코드)
```

**핵심 의존 관계 요약**: `ML 파이프라인` ⊂ `CT` ⊂ `MLOps`, 그리고 `MLOps` = `CI` + `CD` + `CT` + `CM` + (레지스트리/피처스토어/메타데이터 인프라) + (문화·조직).

---

## 참고 자료
1. Google Cloud, *"MLOps: Continuous delivery and automation pipelines in machine learning"* — https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
2. Google Cloud, *"What is MLOps?"* — https://cloud.google.com/discover/what-is-mlops
3. Google Cloud Blog, *"MLOps aims to unify ML system development"* — https://cloud.google.com/blog/products/ai-machine-learning/key-requirements-for-an-mlops-foundation
4. AWS, *"What is MLOps? - Machine Learning Operations Explained"* — https://aws.amazon.com/what-is/mlops/
5. Microsoft Learn, *"MLOps Maturity Model - Azure Architecture Center"* — https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model
6. apxml, *"What is Continuous Training (CT) in MLOps?"* — https://apxml.com/courses/introduction-to-mlops/chapter-4-automation-and-cicd-for-ml/continuous-training
7. Databricks Community, *"Object-Oriented MLOps Maturity: How Databricks Accelerates Your Journey to Production"* — https://community.databricks.com/t5/technical-blog/object-oriented-mlops-maturity-how-databricks-accelerates-your/ba-p/148348
8. Sulaiman Shamasna, *"The Maturity of an MLOps process"* (Medium) — https://medium.com/@sulaiman.shamasna/the-maturity-of-an-mlops-process-e408c576bdfb
9. ScienceDirect, *"An empirical guide to MLOps adoption: Framework, maturity model and taxonomy"* — https://www.sciencedirect.com/science/article/pii/S0950584925000643
10. arXiv, *"Navigating MLOps: Insights into Maturity, Lifecycle, Tools, and Careers"* — https://arxiv.org/pdf/2503.15577

MLOps 단계별 과정에 대한 2020년 이후 연구를 정확히 비교하기 위해 추가로 학술 자료와 공식 문서를 검증하겠습니다.

# MLOps 단계별 과정 문제 해결 보고서

---

## 1. 문제의 정확한 재정의

**원 문제**: "MLOps의 단계별 과정을 정확히 이해하지 못함"

이를 검증 가능한 형태로 재정의하면 다음과 같다:

> **재정의**: (a) 학술 문헌(2020년 이후)에서 MLOps의 "단계(stage)"가 서로 다른 의미로 사용되고 있다 — ① **라이프사이클 단계**(데이터→학습→배포→모니터링의 워크플로우 단계)와 ② **성숙도 단계**(조직이 Level 0에서 Level 2로 발전하는 능력 단계)라는 두 가지 축이 혼재되어 있다. (b) 현재 학계에 이 둘을 통일하는 단일 표준이 존재하는지, 존재한다면 무엇인지 확인이 필요하다.

이 재정의가 중요한 이유는, 실제로 MLOps 채택은 증가했지만 산업, 학계, 조직마다 제안된 MLOps 라이프사이클 프레임워크와 성숙도 모델이 서로 달라 표준 채택 관행에 대한 혼란이 초래되었다는 점, 그리고 포괄적인 MLOps 라이프사이클 프레임워크가 무엇인지에 대해 산업 간 합의가 없다는 점이 2025년 논문에서도 명시적으로 지적되고 있기 때문이다. 즉 "이해하지 못함"의 상당 부분은 학습자의 문제가 아니라 **분야 자체의 미표준화(non-standardization)** 때문일 가능성이 있다 — 이는 사실(fact)이며 추측이 아니다.

---

## 2. 필요한 가정과 부족한 정보 명시

**가정**:
- A1: "단계별 과정"은 (i) 라이프사이클 워크플로우 단계, (ii) 조직 성숙도 단계 두 가지 모두를 포함하는 것으로 해석한다.
- A2: 2026년 7월 현재까지 arXiv/IEEE/ACM 등에 공개된, 검색으로 확인 가능한 자료만을 근거로 삼는다.
- A3: "가장 많이 인용되는 학술적 정의"를 우선 기준으로 삼되, Google/Microsoft/AWS의 공식 문서도 실무 표준으로서 병기한다.

**부족한 정보 (확인 불가 사항)**:
- 2026년 상반기 시점의 정확한 인용수 순위는 검색으로 실시간 확인하지 못했다 — 이 부분은 추측하지 않는다.
- LLMOps로의 확장이 "MLOps 단계"의 공식 표준을 대체했는지 여부는 단일 논문(Navigating MLOps, 2025)의 주장 외에 교차 검증된 컨센서스를 찾지 못했다 — 아래에서 한계로 명시한다.

---

## 3. 가능한 접근법 비교와 선택

MLOps 단계를 정의하는 학술·산업 접근법은 최소 4가지 계열로 분류된다.

| 접근법 | 대표 문헌 (발행일 명시) | 초점 | 장점 | 한계 |
|---|---|---|---|---|
| **A. 워크플로우 단계 모델** | Kreuzberger, Kühl, Hirschl, *arXiv:2205.02302* (2022년 5월 제출) → *IEEE Access* 정식본 (2023) | 파이프라인 구성요소(CI/CD, 피처스토어, 모델레지스트리 등) | 27편의 동료심사 논문과 8건의 인터뷰를 종합한 체계적 문헌고찰 기반 | "단계"가 아니라 "컴포넌트/역할" 중심이라 순차적 단계로 보기 어려움 |
| **B. 성숙도 단계 모델 (학술)** | John, Olsson, Bosch, *IEEE SEAA 2021* (2021년 9월) | 조직이 MLOps를 도입하며 거치는 단계 | 3개 실제 기업 사례로 검증 | 임베디드 시스템 3개사 한정, 일반화 제한 |
| **C. 성숙도 단계 모델 (산업)** | Google Cloud *Practitioners Guide to MLOps* (2021년 5월); Microsoft *MLOps with Azure ML* (2021년 8월); AWS Whitepaper (2020년 12월, 2022년 6월 개정) | Level 0/1/2 자동화 수준 | 실무 도구와 직결, 널리 인용됨 | 회사마다 Level 정의가 조금씩 다름(반례 가능성 있음) |
| **D. 통합/메타 리뷰** | *MLOps Spanning Whole ML Life Cycle: A Survey*, arXiv:2304.07296 (2023년 4월); *A Mapping Study*, arXiv:2409.19416; *Navigating MLOps*, arXiv:2503.15577 (2025년 3월) | A/B/C를 종합한 "survey of surveys" | 여러 정의의 공통분모 추출 | 메타 리뷰 자체도 서로 다른 통합안을 제시 → 완전한 컨센서스 아님 |

**선택**: 학습자가 이미 파이프라인 구축 능력을 갖췄다는 전제 하에, **접근법 A(워크플로우 단계, Kreuzberger et al.)를 "무엇을 자동화하는가"의 기준으로, 접근법 C(성숙도 단계, Google/Microsoft/AWS)를 "얼마나 자동화되었는가"의 기준으로 병행 사용**하는 것을 선택한다. 이유: A는 가장 엄밀한 체계적 문헌고찰(27편 동료심사 논문, 이후 ACM/IEEE Access 게재)에 기반하고, C는 실무에서 가장 널리 참조되는 표준이기 때문이다. B와 D는 보조 검증 자료로만 사용한다.

---

## 4. 논리 전개와 반례 가능성 검토

### 4.1 라이프사이클 단계 (접근법 A + D)

Kreuzberger 등의 정의에 기반한 MLOps는 데이터 획득, 데이터 준비, 모델 학습, 테스트, 평가, 배포, 지속적 모니터링을 포함하는 ML 모델의 전체 생애주기를 관리하는 포괄적 프레임워크로 제시된다.

2023년 발표된 별도의 8단계 서베이(Zhengxin et al.)는 기존 문헌과 산업 모범사례를 종합하여 8단계로 구성된 MLOps 기반 모델을 제시하며, 첫 단계는 모델 요구사항(Model Requirement) 정의라고 밝힌다.

2025년 JMIR 의료 분야 스코핑 리뷰는 19개 연구를 분석해 MLOps 워크플로우가 (1) 데이터 추출, (2) 데이터 준비 및 엔지니어링, (3) 모델 학습, (4) ML 지표 측정 및 모델 평가, (5) 모델 검증 및 프로덕션 테스트, (6) 모델 서빙 및 배포, (7) 지속적 모니터링, (8) 지속 학습(continual learning)의 8단계로 구성됨을 확인했다.

**반례 가능성 검토**: 위 세 자료(A: 컴포넌트 중심, D-1: 8단계 요구사항 시작, D-2: 8단계 데이터 시작)는 **단계 수는 유사(8개 내외)하나 시작점과 세분화 기준이 다르다** — 이는 "8단계"가 보편 표준이 아니라 저자마다 재구성한 결과임을 보여주는 반례다. 즉, "MLOps는 정확히 N단계"라는 명제는 **거짓**이며, 옳은 명제는 "여러 문헌에서 대략 데이터→학습→검증→배포→모니터링→재학습의 흐름이 공통적으로 나타나지만 정확한 경계와 개수는 저자마다 다르다"이다.

### 4.2 성숙도 단계 (접근법 C + B)

산업계에서는 AWS가 "MLOps: Emerging Trends in Data, Code, and Infrastructure" 백서를 2022년 6월에 발표했으며, 이는 훨씬 단순한 생애주기를 정의하고, Microsoft 역시 2021년 8월 "MLOps with Azure Machine Learning" 백서에서 ML 생애주기와 MLOps 워크플로우를 정의했으며, AWS와 유사하게 단순하고 자명한 구조를 보인다.

학술 성숙도 모델의 경우, John, Olsson, Bosch(2021)는 문헌 검토를 기반으로 ML 모델의 지속적 개발에 관여하는 활동을 상술하는 MLOps 프레임워크를 도출하고, 기업들이 MLOps 실천을 발전시키며 거치는 서로 다른 단계를 개략화하는 성숙도 모델을 제시했으며, 이 프레임워크를 3개 임베디드 시스템 사례 기업에서 검증했다. 이후 후속 연구(John, Gillblad, Olsson, Bosch, 2023)는 이를 확장하여 Ad hoc부터 Kaizen까지 기업이 MLOps를 채택하며 거치는 전형적 단계를 개략화하는 성숙도 모델을 제안하고, 성숙도 모델의 각 단계와 연관된 5개 차원을 식별했다.

**반례 가능성 검토**: Google/Microsoft/AWS의 "Level 0/1/2" 구조와 John et al.의 "Ad hoc→Kaizen" 구조는 **명칭과 단계 수가 다르다** (전자는 3단계 고정, 후자는 5차원 기반 다단계). 따라서 "MLOps 성숙도는 Level 0/1/2로 정의된다"는 명제는 Google Cloud 문서에 국한된 것이며, **보편적 학술 표준이 아니다** — 이것이 핵심 반례다.

### 4.3 정리 (종합)

> **잠정 정리**: MLOps의 "단계"는 (i) 워크플로우 차원과 (ii) 성숙도 차원이라는 서로 직교하는 두 축으로 나뉘며, 각 축 내에서도 문헌 간 완전한 합의가 존재하지 않는다. 다만 워크플로우 축에서는 "데이터 준비 → 모델 학습/평가 → 배포/서빙 → 모니터링 → (재학습으로 순환)"이라는 5개 대분류가 검토한 모든 자료에서 공통적으로 나타난다.

이 정리는 서로 다른 MLOps 라이프사이클 프레임워크와 성숙도 모델이 산업, 학계, 조직에 의해 제안되어 표준 채택 관행에 혼란을 초래했다는 2025년 논문의 명시적 진술과 정합적이다.

---

## 5. 사실 / 계산 / 코드 실행 여부 구분

| 구분 | 해당 내용 | 상태 |
|---|---|---|
| **사실 (웹 검색으로 검증됨)** | Kreuzberger et al. 논문의 존재, 발행일(arXiv 2022년 5월, IEEE Access 2023년), John et al. 2021/2023 논문 존재 및 발행일, Google/Microsoft/AWS 백서 발행일, JMIR 2025 8단계 워크플로우 | 검증 완료, 인용 표시함 |
| **계산** | 해당 없음 (수치 계산이 필요한 문제가 아님) | — |
| **코드 실행** | 수행하지 않음 | 요청되지 않았고, 실행하지 않았음을 명시함 |
| **추측성 내용 (배제함)** | "2026년 현재 표준으로 확정된 단일 MLOps 단계 모델" 여부 | 검색으로 확인된 단일 표준을 찾지 못했으므로 존재한다고 서술하지 않음 |

---

## 6. 결론 — 결과 / 검증 / 한계 / 다음 의사결정 분리

### 결과 (Result)
MLOps의 "단계별 과정"은 **단일하게 확정된 표준이 아니라 두 개의 축**으로 이해해야 한다.
- **축 1 (워크플로우)**: 데이터 준비 → 모델 학습/평가 → 검증/테스트 → 배포/서빙 → 모니터링 → 재학습(트리거)으로의 순환. 이는 Kreuzberger et al.(2022/2023), Zhengxin et al.(2023), JMIR 스코핑리뷰(2025) 세 독립 문헌에서 세부 개수는 다르지만 공통 골격으로 확인됨.
- **축 2 (성숙도)**: 조직이 수동(Level 0/Ad hoc)에서 완전 자동화(Level 2/Kaizen)로 발전하는 단계. Google/Microsoft/AWS의 산업 표준과 John et al.의 학술 모델이 각각 존재하나 명칭·단계 수가 일치하지 않음.

### 검증 (Verification)
- 위 주장들은 모두 arXiv 원문, IEEE Access/SEAA 학회 논문, JMIR 동료심사 저널, Google/Microsoft/AWS 공식 백서를 근거로 인라인 인용했다.
- 서로 다른 문헌 간 **충돌**(단계 수, 명칭 불일치)을 은폐하지 않고 4절에서 명시적으로 반례로 제시했다.

### 한계 (Limitations)
- 2026년 7월 시점 기준 arXiv에서 확인된 최신 리뷰(*Navigating MLOps*, 2025년 3월; *A Systematic Review of MLOps Tools*, 2026년 3월 버전)조차 "통합 표준"을 제안하는 시도일 뿐, 이것이 학계 전반에서 채택된 최종 컨센서스인지는 추가 인용 데이터(피인용수 등)로 교차검증하지 못했다.
- LLMOps로의 확장이 기존 MLOps 단계 정의를 어떻게 개정하는지는 이번 검색 범위에서 충분히 확인되지 않았다 — 이 부분은 확신 없이 서술하지 않았다.
- 산업 백서(Google/Microsoft/AWS)의 최신(2026년) 개정판 존재 여부는 이번 검색에서 재확인하지 못했으며, 확인된 최신 버전은 각각 2021년 5월(Google), 2021년 8월(Microsoft), 2022년 6월(AWS)이다.

### 다음 의사결정 (Next Decision Point)
1. 만약 목표가 **실무 도입**이라면 → Google Cloud *Practitioners Guide to MLOps* (Level 0/1/2)를 1차 참조로 채택할 것을 권고.
2. 만약 목표가 **학술적 엄밀성**이라면 → Kreuzberger, Kühl, Hirschl (IEEE Access, 2023)의 컴포넌트/역할 기반 아키텍처를 1차 참조로 채택할 것을 권고.
3. 두 축(워크플로우 vs 성숙도)을 혼동하지 않도록, 다음 학습 단계에서는 "우리 조직이 어느 축의 어느 단계를 묻고 있는가"를 먼저 명확히 하는 것이 필요하다.

---

## 참고 자료 (전체 출처)

1. Kreuzberger, D., Kühl, N., & Hirschl, S. *Machine Learning Operations (MLOps): Overview, Definition, and Architecture*. arXiv:2205.02302 (2022년 5월 제출) / *IEEE Access*, 11, 31866–31879 (2023). https://arxiv.org/pdf/2205.02302
2. John, M. M., Olsson, H. H., & Bosch, J. *Towards MLOps: A Framework and Maturity Model*. 2021 47th Euromicro Conference on Software Engineering and Advanced Applications (SEAA), pp. 1–8 (2021년 9월). https://ieeexplore.ieee.org/document/9582569/
3. John, M. M., Gillblad, D., Olsson, H. H., & Bosch, J. *Advancing MLOps from Ad hoc to Kaizen*. 2023 SEAA. https://www.computer.org/csdl/proceedings-article/seaa/2023/423500a094/1TlXH9Ty1mE
4. Zhengxin, F. et al. *MLOps Spanning Whole Machine Learning Life Cycle: A Survey*. arXiv:2304.07296 (2023년 4월). https://arxiv.org/abs/2304.07296
5. *Maturity Framework for Operationalizing Machine Learning Applications in Health Care: Scoping Review*. Journal of Medical Internet Research, 2025;27:e66559 (2025). https://www.jmir.org/2025/1/e66559
6. *Machine Learning Operations: A Mapping Study*. arXiv:2409.19416 (2024년 9월). https://arxiv.org/pdf/2409.19416
7. *Navigating MLOps: Insights into Maturity, Lifecycle, Tools, and Careers*. arXiv:2503.15577 (2025년 3월). https://arxiv.org/html/2503.15577v1
8. *A Systematic Review of MLOps Tools: Tool Adoption, Lifecycle Coverage, and Critical Insights*. arXiv:2604.16371. https://arxiv.org/html/2604.16371
9. *Machine Learning Operations: A Survey on MLOps Tool Support* (Hewage & Meedeniya). arXiv:2202.10169 (2022년 2월). https://arxiv.org/pdf/2202.10169
10. *An analysis of the challenges in the adoption of MLOps*. ScienceDirect (2024). https://www.sciencedirect.com/science/article/pii/S2444569X24001768
11. Google Cloud. *Practitioners Guide to MLOps: A framework for continuous delivery and automation of machine learning* (백서, 2021년 5월).
12. Microsoft. *MLOps with Azure Machine Learning* (백서, 2021년 8월).
13. AWS. *MLOps: Emerging Trends in Data, Code, and Infrastructure* (백서, 2022년 6월) 및 *MLOps: Continuous Delivery for Machine Learning on AWS* (백서, 2020년 12월).
14. ml4devs. *Survey of Machine Learning Lifecycle*. https://www.ml4devs.com/en/articles/mlops-survey-of-machine-learning-life-cycle/

MLOps 단계별 과정을 더 세밀하게 작성하기 위해 각 단계의 구체적 내용을 재확인하겠습니다.필요한 세부 정보를 좀 더 확인하겠습니다.

# MLOps "단계별 과정" 상세 분석 (검증판)

MLOps의 "단계"는 **① 워크플로우 라이프사이클 단계**와 **② 조직 성숙도 단계**라는 서로 다른 두 축으로 구성된다. 이번 답변에서는 각 축을 실제 문헌·공식문서에 근거해 세부 단계까지 풀어서 설명한다.

---

## 축 1. 워크플로우 라이프사이클 단계 (기능적 컴포넌트 기준)

Kreuzberger, Kühl, Hirschl (IEEE Access, 2023)의 체계적 문헌고찰은 27편의 동료심사 논문과 인터뷰 연구를 결합해 이 논문은 지정된 기술 세트로 ML 제품을 자동화하고 운영하려는 ML 연구자와 실무자에게 지침을 제공하며 MLOps에 대한 포괄적 정의를 제시한다. 이 논문은 방법론으로 문헌고찰, 도구 리뷰, 인터뷰 연구를 사용했다.

이 아키텍처는 순차적 "단계"보다는 **역할과 기능 컴포넌트**로 구성되며, 실무 정리(Bhashkar Bhaskar 등)에서는 이를 다음과 같이 단계화한다:

1. **비즈니스 문제 분석 (Business problem analysis)**
2. **데이터셋 피처 및 저장 (Dataset features and storage)**
3. **ML 분석 방법론 (ML analytical methodology)** — 모델 학습
4. **파이프라인 CI 컴포넌트**
5. **파이프라인 CD 컴포넌트**
6. **자동 ML 트리거링 (Automated ML triggering)**
7. **모델 레지스트리 저장 (Model registry storage)**
8. **모니터링 및 성능 (Monitoring and performance)**
9. **프로덕션 ML 서비스 (Production ML service)**

이를 뒷받침하는 4대 원칙은 다음과 같이 명확히 구분된다: 지속적 통합(CI)은 코드, 데이터, 모델의 지속적 테스트와 검증 단계이고, 지속적 배포(CD)는 자동으로 다른 모델 예측 서비스를 배포하는 ML 학습 파이프라인의 전달 단계이며, 지속적 학습(CT)은 재배포를 위해 자동으로 ML 모델을 재학습하는 단계이고, 지속적 모니터링(CM)은 프로덕션 데이터와 모델 성능 지표를 모니터링하는 단계이다.

**핵심 저장소(Stores) 컴포넌트** (MLHOps 논문, arXiv:2305.02474 기준):
- 원시 데이터 저장소: 원시 데이터 저장소는 처리 전 데이터가 최초로 수집·저장되는 스테이징 영역 역할을 하는 원본 미가공 형태의 데이터를 저장하는 중앙 저장소이다.
- 피처 스토어: 피처 스토어는 ML 모델에 사용되는 피처를 저장·관리·공유하는 중앙 온라인 저장소로, 원시 데이터를 처리해 얻은 피처를 실시간 서빙에 제공한다.
- ML 메타데이터 스토어: ML 메타데이터 스토어는 파이프라인 컴포넌트 정보와 그 실행 내역을 포함한 ML 파이프라인 관련 메타데이터를 기록·조회하는 역할을 한다.

---

## 축 2. 성숙도 단계 (자동화 수준 기준) — 여기서 문헌 간 명확한 불일치 존재

### 2-A. Google Cloud 모델 (3단계, 2021년 5월 백서 / 2024년 8월 개정 문서)

**Level 0 — 수동 프로세스**
레벨 0 프로세스의 특징은 데이터 분석, 데이터 준비, 모델 학습, 검증을 포함한 모든 단계가 수동인 스크립트 기반의 대화형 프로세스라는 점이며, 최첨단 모델을 만들 수 있는 데이터 과학자와 ML 연구자가 있어도 모델 구축·배포 프로세스가 전적으로 수동인 것이 일반적이다.

**Level 1 — ML 파이프라인 자동화**
레벨 1 설정의 특징은 다음과 같다: 실험 단계들이 오케스트레이션되어 단계 간 전환이 자동화되며 빠른 실험 반복과 전체 파이프라인의 프로덕션 이관 준비성이 향상되고, 프로덕션 내 모델의 지속적 학습(CT)은 다음 절에서 다룰 라이브 파이프라인 트리거를 기반으로 새로운 데이터를 사용해 자동으로 학습되며, 실험-운영 대칭성은 개발/실험 환경에서 사용한 파이프라인 구현이 사전운영·운영 환경에서도 동일하게 사용되는 것이고, 컴포넌트와 파이프라인을 위한 코드가 모듈화되어 재사용·조합·공유가 가능하다.

**Level 2 — CI/CD 파이프라인 자동화**
Level 2에서는 파이프라인 코드 자체가 자동으로 빌드·테스트·배포된다. 이 단계의 근거로 실무 요약에서는 견고하고 자동화된 CI/CD 시스템이 파이프라인을 신속하고 신뢰성 있게 프로덕션에 갱신하는 데 필수적이며, 이를 통해 데이터 과학자가 피처 엔지니어링, 모델 아키텍처, 하이퍼파라미터에 관한 새로운 아이디어를 신속히 실험할 수 있다고 설명한다.

이 3단계 모델의 한계도 명시적으로 지적된다: 완전 수동 레벨에서 레벨 1("ML 파이프라인 자동화")로의 도약은 상당히 크며, 이는 Google 프레임워크를 진단 도구로 사용하기 어렵게 만드는데 왜냐하면 여러 면에서 MLOps를 하고 있거나 하고 있지 않거나 둘 중 하나일 뿐 그 사이의 전환 단계가 거의 없기 때문이다.

### 2-B. Microsoft Azure 모델 (5단계, 2020년 1월 최초 공개 / 2026년 6월 최신 갱신)

Microsoft의 모델은 Google보다 세분화되어 있다. Microsoft 모델은 가장 세밀하게 단계 진행을 구분하며, 조직의 MLOps 도입과 성숙도 정도를 이해하기 위해 사용할 수 있는 5개의 개별 레벨(0부터 시작)을 제시한다.

- **Level 0 — No MLOps**: Google의 레벨 0에 해당하며 모든 것이 수동이고 자동화가 전혀 없으며, 프로세스가 스크립트 기반·수동일 뿐 아니라 모델 개발·배포의 여러 부분을 담당하는 팀들이 서로 사일로화되어 있다.
- **Level 1 — DevOps but no MLOps**: 자동화의 시작 단계로, 릴리스는 여전히 "고통스럽지만" 빌드와 테스트는 자동화되고 제한적인 피드백이 도입된다.
- **Level 2 — Automated Training**
- **Level 3 — Automated Model Deployment**: 모델 릴리스가 CD 파이프라인을 통해 관리된다.
- **Level 4 — Full MLOps Automated Operations**: 이 레벨에서는 지속적 모델 개선을 위한 모니터링과 재학습이 도입되며, 정의된 지표에 의해 자동으로 또는 사람의 판단을 거쳐 재학습 작업이 트리거되고, 새 모델을 기존 추론 엔드포인트에 배포할 때는 블루-그린 배포(및 미러링 트래픽)를 사용해 기존 프로덕션 환경에 영향을 주지 않고 트래픽을 제어할 수 있다.

가장 최신(2026년 6월 갱신) Microsoft Learn 문서는 다음과 같이 명시한다: 실제로는 조직이 동시에 하나 이상의 레벨 특성을 보이는 경우가 많으며, 레벨 간 전환은 이산적이고 고립된 단계가 아니라 능력이 겹치고 점진적으로 발전하는 연속체로 다뤄야 한다. 또한 이 문서는 생성형 AI 운영(GenAIOps)이 MLOps 성숙도 레벨을 대체하는 것이 아니라 보완하는 추가 역량을 도입한다고 명시해, LLM 시대에도 이 5단계 골격이 유지되고 있음을 확인해준다.

**참고 - 충돌 사항**: 일부 자료(GigaOm 모델)는 Microsoft와 마찬가지로 CMMI에서 영감받아 Level 0부터 4까지 5단계를 정의하지만, 각 레벨을 전략, 아키텍처, 모델링, 프로세스, 관리라는 5개 범주로 설명한다는 점에서 Microsoft의 순수 기술 중심 5단계와는 분류축이 다르다.

### 2-C. 학술 성숙도 모델 (John, Gillblad, Olsson, Bosch — SEAA 2023 / Information and Software Technology 2025)

이 모델은 산업 백서와 별개로 **7개 기업 다중 사례 연구**를 통해 도출되었다는 점에서 검증 방식이 다르다. 실증 연구를 바탕으로, 기업들이 MLOps를 채택하며 거치는 전형적 단계를 Ad hoc부터 Kaizen까지 개략화한 성숙도 모델을 제안하고, 각 성숙도 모델 단계와 연관된 5개 차원을 식별했다.

2025년 확장판(Information and Software Technology, 183호)은 이를 다음과 같이 구체화한다: MLOps 성숙도 모델은 (a) Ad hoc, (b) DataOps, (c) Manual MLOps, (d) Automated MLOps, (e) Kaizen MLOps의 5단계로 구성되며, 이 단계들은 데이터, 모델, 배포, 운영&인프라 등 5개 차원에 걸쳐 있다.

각 단계의 성격은 다음과 같이 설명된다:
- **Manual MLOps 단계**는 기업이 모델 개발·배포·운영 프로세스의 표준화로 전환하는 단계이며, 이해관계자·팀과의 정기적 소통·협업이 이루어진다.
- **Automated MLOps 단계**로 진행하면서 기업들은 모델과 데이터 관련 모든 프로세스에 자동화를 도입하기 시작한다.
- **Kaizen MLOps 단계**에서는 데이터, 모델, 배포, 운영, 인프라, 역할 관련 모든 프로세스가 정제·최적화되는 전 차원에 걸친 지속적 개선 마인드셋을 추구한다.

별도 논문(arXiv:2501.08402)은 이 모델을 인용하며, 이 모델에서 실험 추적과 모델 모니터링은 3단계 성숙도(즉 Manual MLOps)에 도달하기 위한 필수 요소라고 설명한다 — 이는 산업계 Google/Microsoft 모델이 모니터링을 주로 최상위 레벨(Level 2 또는 Level 4)에 배치하는 것과 대조적으로, **학술 모델은 모니터링을 상대적으로 이른 단계(3단계 중)에 필수 요건으로 위치시킨다는 차이**를 보여준다. 이는 4절에서 언급했던 "산업계 vs 학계 정의 불일치"의 구체적 반례다.

---

## 종합 비교표 (성숙도 축)

| 모델 | 발행/최신개정 | 단계 수 | 최상위 단계 특징 |
|---|---|---|---|
| Google Cloud | 2021년 5월 백서 / 2024년 8월 문서 갱신 | 3 (Level 0–2) | CI/CD 파이프라인 자동화 |
| Microsoft Azure | 2020년 1월 최초 / 2026년 6월 갱신 | 5 (Level 0–4) | 자동 재학습 + 블루-그린 배포 |
| GigaOm | — | 5 (Level 0–4) | 전략·거버넌스까지 포함한 조직적 최적화 |
| John et al. (학술) | SEAA 2023 / IST 2025 | 5 (Ad hoc→Kaizen) | 5개 차원 전반의 지속적 개선 |

세 산업 모델과 한 학술 모델 모두 "낮은 단계에서는 ML 실천이 부재하고, 최고 단계에서는 MLOps 운영이 자동화된다"는 점에서는 수렴하지만, 그 사이 단계를 무엇으로 채우는지는 프레임워크마다 다르며 초점의 각도가 다를 뿐이라는 점이 여러 비교 자료에서 공통적으로 지적된다 — 이는 이전 답변에서 제기한 "완전한 컨센서스 부재" 명제를 재확인한다.

---

## 사실 / 계산 / 검증 구분

| 구분 | 내용 |
|---|---|
| **사실 (검증됨)** | Kreuzberger et al. 아키텍처 컴포넌트, Google 3단계 정의(원문 인용), Microsoft 5단계 정의(공식문서+블로그 교차확인), John et al. 5단계/5차원 학술모델(원논문+2025 확장판 교차확인) |
| **계산** | 해당 없음 |
| **코드 실행** | 수행하지 않음 (검색 결과 중 예시 코드 스니펫이 있었으나 이를 실행하거나 검증하지 않았음을 명시) |
| **추측 배제** | GigaOm 모델의 세부 사항은 원문 백서에 직접 접근하지 못해 2차 출처(블로그)로만 확인했으므로 확신도를 낮춰 표기함 |

---

## 한계

- Microsoft의 Level 3/4 구분에 대해서는 실무자 커뮤니티(Microsoft Q&A)에서도 CI/CD 경계가 불명확하다는 질문이 제기되어 있어, 이 부분은 공식 문서만으로 완전히 해소되지 않는 모호성이 남아있다.
- GigaOm 모델은 원저작물(연구 리포트)에 직접 접근하지 못했으며, 2차 블로그 자료에 근거했으므로 신뢰도가 상대적으로 낮다.
- LLMOps/GenAIOps가 이 성숙도 단계를 어떻게 구체적으로 확장하는지는 Microsoft 문서에서 "보완한다"는 언급만 확인했고, 세부 단계 정의까지는 검증하지 못했다.

---

## 참고 자료 (전체 출처)

1. Kreuzberger, D., Kühl, N., & Hirschl, S. *Machine Learning Operations (MLOps): Overview, Definition, and Architecture*. IEEE Access, 11, 31866–31879 (2023). https://ieeexplore.ieee.org/document/10081336
2. Bhashkar Bhaskar. *MLOps on GCP — Understand basic ML Workflow Management up-to Production-Ready*. Medium (2021년 10월). https://bhashkarkunal.medium.com/mlops-on-gcp-understand-basic-ml-workflow-management-up-to-production-ready-1c8b2119b62f
3. *MLHOps: Machine Learning for Healthcare Operations*. arXiv:2305.02474. https://arxiv.org/pdf/2305.02474
4. Google Cloud. *MLOps: Continuous delivery and automation pipelines in machine learning*. Cloud Architecture Center (최종 갱신 2024년 8월 28일). https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
5. ZenML Blog. *Everything you ever wanted to know about MLOps maturity models* (2022년 3월). https://www.zenml.io/blog/everything-you-ever-wanted-to-know-about-mlops-maturity-models
6. Microsoft Learn. *MLOps Maturity Model - Azure Architecture Center* (최종 갱신 2026년 6월 3일). https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model
7. Microsoft Community Hub. *MLOps Maturity Model with Azure Machine Learning* (2022년 7월). https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/mlops-maturity-model-with-azure-machine-learning/3520625
8. Kiroframe Blog. *MLOps maturity levels: the most well-known models* (2025년 12월). https://kiroframe.com/mlops-maturity-levels-the-most-well-known-models/
9. John, M. M., Olsson, H. H., & Bosch, J. *Towards MLOps: A Framework and Maturity Model*. 2021 47th Euromicro SEAA (2021년 9월). https://ieeexplore.ieee.org/document/9582569/
10. John, M. M., Gillblad, D., Olsson, H. H., & Bosch, J. *Advancing MLOps from Ad hoc to Kaizen*. 2023 49th Euromicro SEAA, pp. 94–101 (2023년 9월). https://www.computer.org/csdl/proceedings-article/seaa/2023/423500a094/1TlXH9Ty1mE
11. John, M. M., Olsson, H. H., & Bosch, J. *An empirical guide to MLOps adoption: Framework, maturity model and taxonomy*. Information and Software Technology, 183, 107725 (2025년 3월). https://www.sciencedirect.com/science/article/pii/S0950584925000643 / 전문 PDF: https://www.diva-portal.org/smash/get/diva2:1952784/FULLTEXT01.pdf
12. *Addressing Quality Challenges in Deep Learning: The Role of MLOps and Domain Knowledge*. arXiv:2501.08402 (2025년 1월). https://arxiv.org/pdf/2501.08402
13. *Navigating MLOps: Insights into Maturity, Lifecycle, Tools, and Careers*. arXiv:2503.15577 (2025년 3월). https://arxiv.org/pdf/2503.15577
