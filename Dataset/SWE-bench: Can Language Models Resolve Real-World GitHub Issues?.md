# SWE-bench: Can Language Models Resolve Real-World GitHub Issues?

### 1. 핵심 주장과 주요 기여

**SWE-bench** 논문의 핵심 주장은 기존 코드 생성 벤치마크(HumanEval 등)들이 **자체 포함적이고 단순한 문제**에 집중하여 실제 소프트웨어 엔지니어링의 복잡성을 반영하지 못한다는 것입니다. 이 논문은 실제 GitHub 이슈와 이를 해결하는 PR(Pull Request)을 연결한 **2,294개의 실제 소프트웨어 엔지니어링 문제**로 구성된 평가 프레임워크를 도입하여, 언어모델이 **대규모 저장소(repository)에서 다중 파일 수정을 통해 실제 문제를 해결할 수 있는 능력**을 평가하고자 합니다.[1]

**주요 기여:**
- **현실적 벤치마크 구성**: 12개 인기 Python 패키지에서 수집한 2,294개의 실제 이슈 및 해결책
- **견고한 평가 프레임워크**: 단위 테스트를 통한 실행 기반 검증 (Execution-based evaluation)
- **지속 업데이트 가능성**: 새로운 GitHub 이슈를 통한 연속적 벤치마크 확장
- **오픈 모델 데이터 공개**: 19,000개 이슈-PR 쌍으로 구성된 훈련 데이터셋(SWE-bench-train) 및 **SWE-Llama** 미세조정 모델 공개[1]

***

### 2. 해결하고자 하는 문제, 제안 방법, 모델 구조 및 성능

#### 2.1 핵심 문제

**기존 벤치마크의 한계:**[1]
- 자체 포함적이고 분리된 문제(HumanEval: 함수 생성)에만 초점
- 실제 소프트웨어 엔지니어링의 복잡성 미반영
  - 대규모 저장소(평균 3,010개 파일, 438,000줄 코드)의 탐색
  - 다중 파일, 함수, 클래스 간의 조직된 변경 필요
  - 컨텍스트 길이 한계
  - 코드베이스 레벨의 복잡한 추론

#### 2.2 과제 설정(Task Formulation)

**입력:** 이슈 텍스트 설명과 완전한 코드베이스

**출력:** Patch 형식의 코드 수정사항 (Unix diff 형식)

**평가 기준:**
- Patch 적용 성공 여부
- 생성된 패치를 코드베이스에 적용 후 모든 단위/시스템 테스트 통과 여부
- 평가 지표: **해결된 이슈의 백분율** (% of resolved issues)[1]

#### 2.3 제안된 방법론

**벤치마크 구성 3단계 파이프라인:**[1]

```math
\text{Stage I: Repository Selection} \xrightarrow{90,000 \text{ PRs}}
```

```math
\text{Stage II: Attribute Filtering} \xrightarrow{11,407 \text{ candidates}}
```
```math
\text{Stage III: Execution Filtering} \xrightarrow{2,294  \text{final instances}}
```

**Stage I - 저장소 선택 및 데이터 수집:**
- 12개 인기 Python 저장소에서 약 90,000개 PR 수집
- 인기 저장소 선택 이유: 유지보수 품질 높음, 명확한 기여 가이드라인, 우수한 테스트 커버리지

**Stage II - 속성 기반 필터링:**
필터링 조건:
1. PR이 특정 GitHub 이슈 해결 (Resolves an issue)
2. PR이 테스트 파일에 변경 추가 (Contributes tests)

**Stage III - 실행 기반 필터링:**
필터링 조건:
1. **Fail-to-Pass (F2P) 테스트** 존재: PR 적용 후 실패 → 성공으로 변하는 테스트 최소 1개
2. 설치 및 런타임 에러 없음

#### 2.4 입력 구성 및 검색 기법

**컨텍스트 관련 문제 해결:** 코드베이스는 평균 438,000줄(438K 토큰)로 모델의 컨텍스트 윈도우를 초과하므로, 관련 파일 선택이 중요합니다.[1]

**두 가지 검색 방식 비교:**

| 검색 방식 | 특징 | 성능 |
|---------|------|------|
| **Sparse Retrieval (BM25)** | BM25 기반 의미론적 검색으로 관련 파일 선택 | 현실적이나 성능 제한 |
| **Oracle Retrieval** | 참조 PR이 실제 편집한 파일 제공 | 분석 목적의 상한(upper bound) |

**검색 성능 분석:**[1]
- BM25 (27K 토큰 한계): 오라클 파일의 평균 44.41% 회상 (Avg recall)
- 40% 이상의 이슈에서 BM25는 오라클 파일의 상위 집합 검색 (subset recall)
- 거의 절반의 이슈에서 오라클 파일을 전혀 검색하지 못함

**BM25 vs 오라클 회상 메트릭:**[1]

```math
\text{Recall}_{\text{Any}} = \frac{\text{instances where BM25 retrieves ANY oracle file}}{\text{Total instances}}
```

- Any (최소 1개 파일): 51.27% (27K 토큰)
- All (모든 파일): 39.83% (27K 토큰)

#### 2.5 모델 구조

**평가 대상 모델:**[1]

| 모델 | 컨텍스트 윈도우 | 커버리지 |
|------|----------------|---------|
| ChatGPT-3.5 | 16,385 | 58.1% |
| GPT-4 | 32,768 | 84.1% |
| Claude 2 | 100,000 | 96.4% |
| SWE-Llama 7b | 100,000 | 94.8% |
| SWE-Llama 13b | 100,000 | 94.8% |

**SWE-Llama 미세조정 설정:**[1]

SWE-Llama는 CodeLlama 모델을 19,000개 이슈-PR 쌍으로 감독 학습 미세조정합니다.

**LoRA (Low-Rank Adaptation) 설정:**

$$\text{Weight} = W_0 + \alpha \cdot A \cdot B^T$$

여기서:
- $W_0$: 원래 가중치
- $\alpha$: 스케일링 계수
- $A, B$: 저순위 행렬 ($r=16$)

**하이퍼파라미터:**[1]
- LoRA rank ($r$): 16
- Dropout: 0.05
- Learning rate: $6 \times 10^{-4}$
- Batch size: 32 sequences/gradient step
- Max epochs: 4
- 최대 시퀀스 길이: 30,000 토큰
- 유효 훈련 데이터: 10,000 이슈 (30K 토큰 이상 제외)

#### 2.6 성능 결과

**BM25 검색 기반 결과:**[1]

| 모델 | 해결률 (%) | Patch 적용률 (%) |
|------|----------|-----------------|
| Claude 2 | 1.97 | 43.07 |
| Claude 3 Opus | 3.79 | 46.56 |
| SWE-Llama 13b | 0.70 | 53.62 |
| SWE-Llama 7b | 0.70 | 51.74 |
| GPT-4-turbo | 1.31 | 26.90 |
| ChatGPT-3.5 | 0.17 | 26.33 |

**오라클 검색 기반 결과:**

$$\text{Resolution Rate}_{\text{Oracle}} = \frac{\text{Instances with all F2P tests passed}}{\text{Total instances}} \times 100$$

- Claude 2: 4.8%
- SWE-Llama 13b: 3.97%
- ChatGPT-3.5: 0.52%

#### 2.7 핵심 성능 분석

**1. 컨텍스트 길이와의 상관관계:**[1]

모델 성능은 **입력 컨텍스트 길이가 증가함에 따라 감소**합니다:

$$\text{Performance}_{Claude2} \propto \frac{1}{\text{Context Length}}$$

- 13K 토큰: 1.96% 해결률
- 27K 토큰: 1.87% 해결률 (↓ 4.6%)
- 50K 토큰: 1.22% 해결률 (↓ 37.8%)

**이유:** 모델이 많은 관련 없는 코드에 주의를 산만해하며, 편집할 문제 코드의 위치 파악(localization)에 어려움

**2. 오라클 축약 (Oracle-collapsed) 설정 개선:**[1]

편집되지 않은 코드를 제거하되 15줄 버퍼 제공:

$$\text{Performance}_{\text{Oracle-collapsed}} = 1.33 \times \text{Performance}_{\text{Oracle}}$$

- Claude 2: 4.8% → 5.93% (성능 향상 23.5%)
- GPT-4: 1.3% → 3.4% (성능 향상 161%)

**3. Patch 생성 크기 분석:**[1]

모델 생성 패치는 참조 솔루션보다 **훨씬 짧고 단순합니다:**

$$\text{Avg Model Patch Length} = 0.48 \times \text{Avg Gold Patch Length}$$

| 메트릭 | Claude 2 (모델) | Claude 2 (정답) |
|--------|----------------|-----------------|
| 총 라인 | 19.6 | 44.1 |
| 추가 라인 | 4.2 | 12.0 |
| 제거 라인 | 1.9 | 5.8 |
| 편집 함수 수 | 1.1 | 2.1 |
| 편집 파일 수 | 1.0 | 1.2 |

**4. 시간별 성능 분석:**[1]

이슈 생성 시간에 따른 성능은 유의미한 차이 없음:
- 2023년 이전: 4.87% (Claude 2)
- 2023년 이후: 4.23% (Claude 2)

→ 훈련 데이터 오염(data contamination)의 영향 제한적

**5. 적용 성공 vs 해결 성공:**[1]

**Fail-to-Pass, Pass-to-Pass 분류:**

성공적으로 적용된 패치의 6가지 상태:

| 상태 | F2P 테스트 | P2P 테스트 | 설명 |
|------|----------|-----------|------|
| Resolved | ✓ All | ✓ All | 완전 해결 |
| Breaking Resolved | ✓ All | ✗ Some | 이슈는 해결했으나 기존 기능 손상 |
| Partially Resolved | ✗ Some | ✓ All | 이슈 부분 해결 |
| Work in Progress | ✗ Some | ✗ Some | 진행 중 상태 |
| No-Op | ✗ All | ✓ All | 아무 변화 없음 (60-70%) |
| Regression | ✗ Some | ✗ Some | 기존 기능 손상 (30-40%) |

대부분의 실패 사례는 **No-Op** (모델 생성 패치가 아무 작용하지 않음)

***

### 3. 일반화 성능 향상 가능성

#### 3.1 원본 논문의 일반화 문제 분석

**핵심 한계:**[1]

1. **컨텍스트 지역화(Localization) 실패:**
   - 모델이 편집할 정확한 위치를 파악하지 못함
   - 장 컨텍스트 내에서 "바늘 찾기" 현상

2. **원시적 코드 생성:**
   - 기존 제3자 라이브러리나 코드베이스 활용 부족
   - 코드 스타일, 관례 무시
   - 대규모 컨텍스트를 고려한 포괄적 해결책 미제시

3. **Patch 형식 생성 어려움:**
   - 모델은 전체 파일 재생성보다 patch 형식이 나음
   - 그러나 patch 형식도 정확한 생성 어려움

#### 3.2 최신 연구 기반 일반화 개선 방법 (2024-2025)

**최신 성능 진전:**[2][3][4][5][6]

2023년부터 2024년 사이에 **놀라운 성능 향상**이 관찰되었습니다:

```math
\text{Performance Improvement}_{2023 \to 2024} = \frac{71.7\% - 4.4\%}{4.4\%} = \boxed{1527\%}
```

최신 모델들의 성능:[3][4][5][6][2]

| 접근 방식 | 해결률 | 발표 | 주요 기법 |
|---------|------|------|---------|
| Claude 2 (원본) | 1.97% | 2023 | - |
| SWE-RL (Llama 3 70B) | **41.0%** | 2025 | 강화학습 + 소프트웨어 진화 |
| SWE-Gym + Verifiers | 32.0-26.0% | 2024 | 훈련 환경 + 검증기 |
| Lingma SWE-GPT 72B | 30.20% | 2024 | 개발 프로세스 중심 |
| Test-Time Compute (32B) | 46% | 2025 | 추론 시간 스케일링 |
| ReSAT (작은 모델) | - | 2024 | 저장소 구조 인식 훈련 |

**1. 강화학습 기반 개선 (SWE-RL):**[6]

$$L_{\text{RL}} = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau) - V_\phi(\tau)]$$

- **소프트웨어 진화 기록** 활용: 코드 스냅샷, 코드 변경, 이슈, PR
- **Llama 3-SWE-RL-70B**: SWE-bench Verified에서 **41.0% 해결률** (최고 성능)
- 훈련 프로세스를 통한 **일반화 능력 향상**

**2. 추론 시간 컴퓨트 스케일링:**[2]

$$\text{Performance}_{32B} = \text{Performance}_{7B} + f(\text{Test-Time Compute})$$

- 32B 모델이 **46% 해결률** 달성
- "더 크게 생각하되, 더 길게 생각하라 (Think Longer, Not Larger)" 패러다임
- 테스트 실행을 통한 피드백 루프에서 **중간 결정 지점**에 집중

**3. 저장소 구조 인식 훈련 (ReSAT):**[3]

$$L = L_{\text{localization}} + L_{\text{editing}}$$

- **버그 지역화 능력** 향상: 저장소 내 문제 위치 파악
- **코드 편집 훈련 데이터**: 컨텍스트 기반 코드 편집 능력
- **작은 모델도 효과**: 작은 언어모델(SLM)의 일반화 성능 향상

**4. 훈련 환경 개선 (SWE-Gym):**[5]

$$\text{Performance}_{\text{trained}} = \text{Performance}_{\text{baseline}} + 19\%$$

- 실행 가능한 런타임 환경 제공
- 에이전트 궤적(agent trajectories) 샘플링
- **검증기(Verifier)** 훈련으로 추론 시간 스케일링

**5. 개발 프로세스 중심 학습 (Lingma SWE-GPT):**[4]

$$P_{\text{resolve}} = f(\text{Issue} \to \text{Edit} \to \text{Test} \to \text{Verify})$$

- 동적 상호작용 및 반복적 문제 해결 모델링
- **22.76% 상대 성능 향상** (vs. Llama 3.1 405B)
- 개발 프로세스 시뮬레이션

#### 3.3 에이전트 기반 접근법의 돌파

최신 연구에서 **에이전트 기반 방법**이 단순 시퀀스-투-시퀀스 접근법을 크게 능가합니다:[7]

```math
\text{Agent-based Performance} = \boxed{55\%} \text{ (SWE-Bench Lite)}
```

**다양성 강화 지능 (Diversity Empowers Intelligence, DEI):**[7]

- 개방형 소스 SWE 에이전트 위원회: 최대 개별 해결률 27.3% → **34.3%** (25% 향상)
- 최고 성능 그룹: **55% 해결률** (대부분의 폐쇄형 솔루션 능가)
- **협력 AI 시스템**이 단일 모델 능력 초과

#### 3.4 일반화 성능 향상의 핵심 요인

| 요인 | 작용 메커니즘 | 성능 영향 |
|------|-------------|---------|
| **컨텍스트 압축** | 관련 없는 코드 제거, 편집 지역 집중 | +161% (GPT-4) |
| **강화학습** | 보상 신호를 통한 반복 개선 | +41% (Llama 3 70B) |
| **저장소 구조 인식** | 파일 간 의존성 이해 | SLM 개선 |
| **에이전트 아키텍처** | 반복적 탐색, 피드백, 수정 | +25-55% |
| **추론 시간 스케일링** | 테스트 실행 기반 검증 | 46% (32B) |
| **개발 프로세스 모델링** | 현실적 소프트웨어 개발 시뮬레이션 | 30%+ |

***

### 4. 한계와 향후 연구 고려사항

#### 4.1 원본 논문의 명시적 한계

**1. 언어 제약:**[1]
- 현재 Python만 포함
- 다른 프로그래밍 언어 확장 필요

**2. 평가 방법의 한계:**[1]
- 실행 기반 테스트만으로 부족 가능
- 코드 품질, 효율성, 가독성 등 주관적 측면 미포함
- 자동화된 테스트만으로는 신뢰성 보장 부족

**3. 모델 선택 제약:**[1]
- 장 컨텍스트 처리 모델만 사용 가능
- 당시 기준 코드라마 기반 모델에 제한

#### 4.2 최신 비판과 신뢰성 문제 (2024-2025)

**1. SWE-Bench+ 분석: 데이터 품질 문제:**[8]

$$\text{Solution Leakage Rate} = 60.83\%$$

- **60.83%의 해결 이슈가 "솔루션 누출" 포함**: 
  - 이슈 리포트에서 직접 제시
  - 댓글에서 간접 힌트
- **47.93%가 약한 테스트**로 인해 잘못된 "해결" 표시
- 실제 모델 능력 평가의 신뢰성 의문

**2. 메모리화 vs. 일반화 (SWE-Bench Illusion):**[9]

메모리화 분석 결과:

$$\text{Accuracy}_{\text{SWE Verified}} > \text{Accuracy}_{\text{SWE Full}} > \text{Accuracy}_{\text{SWE Extra}}$$

- **SWE Verified**: 34.9%
- **SWE Full**: 28.7%
- **SWE Extra**: 18.2% (외부 벤치마크 수준)

**시사점:** 모델이 **큐레이션된 표준 솔루션을 메모리화**했을 가능성 → 진정한 일반화 능력 의문

**3. 데이터 오염 추적 필요성:**[10][9]

최신 연구(SWE-Rebench)는 다음을 강조:

$$\text{Evaluation Reliability} \propto \text{Contamination Tracking} + \text{Temporal Controls}$$

- 모델 발표 날짜 전 수집된 데이터만 평가
- 저장소 간 검증으로 일반화 능력 검증
- 벤치마크 특화 최적화 vs. 실제 능력 구분

#### 4.3 향후 연구 시 고려사항

**1. 멀티언어 확장 및 도메인 다양화:**
- Java, C++, Go, Rust 등 주요 언어 추가
- 웹 개발, 데이터 과학 등 도메인별 벤치마크

**2. 메모리화 방지 메커니즘:**
- 시간 기반 필터링 강화
- 저장소 외(out-of-distribution) 테스트 세트
- 명시적 학습 데이터 오염 추적

**3. 평가 지표 고도화:**
- 단순 성공/실패를 넘어 **코드 품질 메트릭**:
  - 순환 복잡도 (Cyclomatic Complexity)
  - Halstead 메트릭
  - 유지보수성 지수
- 인간 검증 추가

**4. 에이전트 아키텍처 표준화:**
- 추론 경로 투명성 요구
- Scaffolding 정규화
- 공정한 비교 가능한 평가 프레임워크

**5. 지속적 벤치마크 업데이트:**
- 실시간 GitHub 이슈 수집
- 모델 발표 후 새로운 이슈 우선순위
- 동적 데이터셋 관리

***

### 5. SWE-bench의 학술적 영향과 의의

#### 5.1 벤치마크로서의 역할 강화

SWE-bench는 **소프트웨어 엔지니어링 AI 분야의 표준 평가 프레임워크**로 자리잡았습니다:[11]

$$\text{Industry Adoption} = \{\text{OpenAI, Anthropic, Google, Meta}\}$$

2023년부터 2024년 사이 **1,527%의 성능 향상**으로, 매년 혁신적 돌파구 제시[11]

#### 5.2 연구 방향의 개선

**패러다임 전환:**

| 이전 | 현재 |
|-----|------|
| 함수 수준 코드 생성 | 저장소 수준 문제 해결 |
| 폐쇄형 문제 (자체 포함) | 개방형 문제 (현실 기반) |
| 정적 테스트 | 동적 실행 검증 |
| 단순 시퀀스-투-시퀀스 | 에이전트 기반 상호작용 |

#### 5.3 미래 전망

**다음 세대 도전 과제:**

1. **SWE-Bench Extended (SWEE-Bench)**: 수백 개 저장소 포함[12]
2. **애플리케이션 벤치마크 (SWA-Bench)**: 라이브러리 대신 애플리케이션 중심[12]
3. **분포 외 테스트**: 메모리화 저항성 강화[9]
4. **프로세스 기반 평가**: 단순 결과 뿐 아니라 문제 해결 프로세스 평가

***

### 결론

**SWE-bench**는 단순한 벤치마크를 넘어 **실제 소프트웨어 엔지니어링의 복잡성을 반영한 평가 표준**으로서 AI 연구의 방향을 재정의했습니다. 원본 논문의 발견(최고 모델도 1.97% 해결률)은 초기에 차담스러웠으나, 이는 **실제 문제 해결의 어려움과 기존 접근의 한계**를 명확히 했습니다.

최근 1년간의 진전은 강화학습, 에이전트 아키텍처, 추론 시간 스케일링, 개발 프로세스 모델링 등을 통해 **일반화 성능을 획기적으로 향상**시켰으나, 여전히 **메모리화, 데이터 오염, 약한 테스트 등의 신뢰성 문제**가 제기되고 있습니다.

향후 연구는 이러한 한계를 극복하면서 동시에 **멀티언어, 도메인 다양화, 코드 품질 평가, 투명한 에이전트 아키텍처** 등으로 확장될 것으로 예상되며, 이는 진정한 의미의 **자율 소프트웨어 엔지니어링 AI** 실현을 위한 필수 발판이 될 것입니다.

***

## 참고 자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3ad5b1ff-1251-40a3-89b4-de1d08c01876/2310.06770v3.pdf)
[2](https://arxiv.org/html/2503.23803v2)
[3](http://arxiv.org/pdf/2412.19031.pdf)
[4](http://arxiv.org/pdf/2411.00622.pdf)
[5](https://arxiv.org/pdf/2412.21139.pdf)
[6](https://arxiv.org/pdf/2502.18449.pdf)
[7](https://arxiv.org/html/2408.07060v1)
[8](https://openreview.net/forum?id=R40rS2afQ3)
[9](https://arxiv.org/html/2506.12286v3)
[10](https://nebius.com/blog/posts/introducing-swe-rebench)
[11](https://hai.stanford.edu/ai-index/2025-ai-index-report/technical-performance)
[12](https://arxiv.org/pdf/2503.07701.pdf)
[13](http://arxiv.org/pdf/2410.06992.pdf)
[14](https://openai.com/index/introducing-swe-bench-verified/)
[15](https://pli.princeton.edu/blog/2023/swe-bench-can-language-models-resolve-real-world-github-issues)
[16](https://arxiv.org/abs/2310.06770)
[17](https://arxiv.org/html/2506.17208v2)
[18](https://www.reddit.com/r/LocalLLaMA/comments/176fuod/swebench_can_language_models_resolve_realworld/)
[19](https://www.vals.ai/benchmarks/swebench-2025-09-24)
