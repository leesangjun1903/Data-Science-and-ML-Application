# TradingAgents: Multi-Agents LLM Financial Trading Framework 

---

## 1. 핵심 주장 및 주요 기여 (간결 요약)

### 핵심 주장

TradingAgents는 실제 트레이딩 회사의 조직 구조를 모방한 **다중 에이전트 LLM 기반 금융 트레이딩 프레임워크**를 제안한다. 기존 단일 에이전트 시스템 또는 비조직적 다중 에이전트 시스템의 한계를 극복하고, 전문화된 역할 분담과 구조화된 커뮤니케이션 프로토콜을 통해 금융 거래 성능을 향상시킬 수 있다는 것이 핵심 주장이다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **현실적 조직 모델링** | 실제 트레이딩 펌 구조(애널리스트, 리서처, 트레이더, 리스크 매니저)를 LLM 에이전트로 구현 |
| **구조화된 커뮤니케이션 프로토콜** | 자연어와 구조화 문서를 혼합하여 "전화 게임 효과(telephone effect)" 방지 |
| **다관점 변증법적 토론** | Bull/Bear 연구원 및 Aggressive/Neutral/Conservative 리스크 에이전트 간의 토론 도입 |
| **멀티모달 데이터 통합** | 주가, 뉴스, 소셜미디어, 재무제표, 기술지표 60개 등을 동시 분석 |
| **설명 가능성** | ReAct 프롬프팅 기반으로 모든 결정에 자연어 근거 제시 |
| **백본 LLM 교체 가능성** | GPU 없이 API만으로 운영 가능, 모델 교체 용이 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

논문은 기존 LLM 기반 금융 에이전트의 두 가지 핵심 한계를 지적한다.

**문제 1: 현실적 조직 모델링 부재 (Lack of Realistic Organizational Modeling)**
- 기존 프레임워크는 특정 태스크에만 집중하거나, 에이전트들이 독립적으로 데이터를 수집하는 방식에 그침
- 실제 트레이딩 펌의 협업 워크플로우를 재현하지 못함

**문제 2: 비효율적 커뮤니케이션 인터페이스 (Inefficient Communication Interfaces)**
- 자연어 메시지 히스토리만 사용하면 대화가 길어질수록 초기 정보가 손실·왜곡되는 **"전화 게임 효과"** 발생
- 비구조적 정보 풀 방식은 에이전트 간 데이터의 관계적 무결성(relational integrity)을 훼손

### 2.2 제안 방법 및 수식

#### 평가 지표 수식

**누적 수익률 (Cumulative Return, CR):**

$$\text{CR} = \left(\frac{V_{\text{end}} - V_{\text{start}}}{V_{\text{start}}}\right) \times 100\%$$

여기서 $V_{\text{end}}$는 시뮬레이션 종료 시점 포트폴리오 가치, $V_{\text{start}}$는 초기 포트폴리오 가치.

**연환산 수익률 (Annualized Return, AR):**

$$\text{AR} = \left(\left(\frac{V_{\text{end}}}{V_{\text{start}}}\right)^{\frac{1}{N}} - 1\right) \times 100\%$$

여기서 $N$은 시뮬레이션 기간(년 단위).

**샤프 비율 (Sharpe Ratio, SR):**

$$\text{SR} = \frac{\bar{R} - R_f}{\sigma}$$

여기서 $\bar{R}$은 포트폴리오 평균 수익률, $R_f$는 무위험 수익률(3개월 국채 수익률), $\sigma$는 수익률의 표준편차.

**최대 낙폭 (Maximum Drawdown, MDD):**

$$\text{MDD} = \max_{t \in [0,T]} \left(\frac{\text{Peak}_t - \text{Trough}_t}{\text{Peak}_t}\right) \times 100\%$$

#### 핵심 방법론

**① ReAct 프롬프팅 프레임워크 (Yao et al., 2023)**
모든 에이전트는 Reasoning(추론)과 Acting(행동)을 결합한 ReAct 방식으로 동작. 환경 상태를 공유하며 맥락에 맞는 행동(리서치, 거래, 토론, 리스크 관리)을 수행.

**② 하이브리드 커뮤니케이션 프로토콜**
- **구조화 문서**: 애널리스트 보고서, 트레이더 결정 보고서 → 제어·명확성·추론 목적
- **자연어 대화**: 에이전트 간 토론(Researcher Team, Risk Management Team) → 다양한 관점 통합

**③ 백본 LLM 전략적 선택**
- 빠른 추론 모델(`gpt-4o-mini`, `gpt-4o`): 요약, 데이터 수집, 표형 데이터 변환 등 저심도 태스크
- 심층 추론 모델(`o1-preview`): 의사결정, 증거 기반 보고서 작성, 데이터 분석 등 고심도 태스크

### 2.3 모델 구조 (5계층 조직 구조)

```
[데이터 소스]
시장가격 | 소셜미디어 | 뉴스 | 재무제표
           ↓
[I. 애널리스트 팀] ──────────────────────────────────
  • 기술적 분석가 (Technical Analyst)
    - MACD, RSI, Bollinger Bands 등 60개 기술지표 분석
  • 감성 분석가 (Sentiment Analyst)
    - Reddit, X/Twitter 소셜미디어 감성 점수 산출
  • 뉴스 분석가 (News Analyst)
    - Bloomberg, Yahoo, EODHD, FinnHub 뉴스 분석
  • 기본적 분석가 (Fundamental Analyst)
    - 재무제표, 이익 보고서, 내부자 거래 분석
           ↓ (구조화 보고서)
[II. 리서처 팀] ──────────────────────────────────────
  • 강세 연구원 (Bullish Researcher)
  ↔ [n라운드 자연어 토론]
  • 약세 연구원 (Bearish Researcher)
  → 진행자(Facilitator)가 토론 결과를 구조화 항목으로 기록
           ↓
[III. 트레이더] ──────────────────────────────────────
  • 분석 보고서 및 연구 결과 종합
  • 매수/매도/보유 신호 생성 + 거래 비율 결정
  • 포트폴리오 조정
           ↓ (거래 제안서)
[IV. 리스크 관리 팀] ─────────────────────────────────
  • 공격적 분석가 (Aggressive/Risky Analyst)
  • 중립 분석가 (Neutral Analyst)
  ↔ [n라운드 자연어 토론]
  • 보수적 분석가 (Conservative/Safe Analyst)
  → 진행자가 결론 구조화 기록
           ↓
[V. 펀드 매니저] ─────────────────────────────────────
  • 최종 거래 승인 및 실행
  • 리스크 조정 반영
           ↓
[거래 실행]
```

### 2.4 성능 향상

실험은 2024년 1월 1일 ~ 3월 29일, AAPL·GOOGL·AMZN 3종목에 대해 5개 기준 전략과 비교.

**성능 비교 결과 (Table 1 기반):**

| 전략 | AAPL CR(%) | AAPL SR | AAPL MDD(%) |
|---|---|---|---|
| Buy & Hold | -5.23 | -1.29 | 11.90 |
| MACD | -1.49 | -0.81 | 4.53 |
| KDJ & RSI | 2.05 | 1.64 | 1.09 |
| ZMR | 0.57 | 0.17 | 0.86 |
| SMA | -3.20 | -1.72 | 3.67 |
| **TradingAgents** | **26.62** | **8.21** | **0.91** |
| **개선폭** | **+24.57%p** | **+6.57** | **–** |

- **AAPL**: 누적수익률 +26.62%, 연환산 수익률 +30.5%, SR=8.21, MDD=0.91%
- **GOOGL**: 누적수익률 +24.36%, SR=6.39
- **AMZN**: 누적수익률 +23.21%, SR=5.60
- 전략 중 최고 대비 최소 **+6.1%p** 이상의 누적수익률 개선

**샤프 비율 해석 주의:** 논문 자체적으로 SR이 매우 높게 나온 이유로, 해당 3개월 기간에 TradingAgents의 낙폭(pullback)이 거의 없었기 때문임을 인정하며 정직하게 보고.

### 2.5 한계

논문에서 명시적·암묵적으로 언급된 한계:

1. **단기 백테스트**: 3개월(분기) 데이터만 사용 → 장기 일반화 불확실
2. **높은 연산 비용**: 예측 1건당 LLM 호출 11회 + 툴 호출 20회 이상 → 실시간 적용 어려움
3. **소수 종목**: 기술주 5개(AAPL, NVDA, MSFT, META, GOOGL)에 한정 → 다양한 섹터·시장 검증 필요
4. **SR 신뢰성**: 극히 낮은 MDD로 인한 SR 과대평가 가능성 존재
5. **시장 영향 미고려**: 대규모 자본 적용 시 시장 충격(market impact) 미반영
6. **라이브 트레이딩 미검증**: 백테스트 환경에서만 검증, 실제 거래 환경 적용 미완
7. **LLM 환각(hallucination) 위험**: 완전히 제거되지 않음

---

## 3. 모델의 일반화 성능 향상 가능성

일반화 성능은 TradingAgents의 핵심 강점이자 동시에 향후 연구의 핵심 과제다. 논문의 내용과 구조적 특성을 분석하면 다음과 같다.

### 3.1 일반화를 지지하는 구조적 강점

#### (1) 도메인-불가지론적 멀티모달 정보 통합
TradingAgents는 특정 기술지표나 패턴에 의존하는 규칙 기반 시스템과 달리, 다양한 정보 소스를 LLM이 유연하게 해석한다.

- 기술적 지표(60개), 뉴스, 소셜미디어, 재무제표를 통합
- LLM은 미리 정의된 패턴이 아닌 컨텍스트 기반 추론으로 결정 → **새로운 시장 환경에도 유연하게 대응 가능**

논문에서 AAPL 사례를 명시적으로 언급:

> *"Notably, on \$AAPL stock—a particularly challenging case due to market volatility during the testing period—traditional methods struggled, as their patterns failed to generalize to this situation. In contrast, TradingAgents excelled under these adverse conditions, achieving returns exceeding 26% within months."*

이는 전통적 패턴 매칭 방식의 일반화 실패와 대비하여 TradingAgents의 비선형적 추론 능력이 새로운 시장 환경에서 더 강건함을 보여준다.

#### (2) 백본 LLM 교체 가능성 (Backbone Exchangeability)

논문은 "**seamless exchangeability of backbone models**"를 명시:

> *"This adaptability supports the integration of improved reasoning models or finance-tuned models customized for specific tasks."*

이는 다음을 의미한다:
- 더 강력한 추론 모델(예: GPT-5, Claude 3.5 등) 출시 시 즉시 성능 향상 가능
- 금융 특화 파인튜닝 모델(FinGPT, BloombergGPT 등)로 교체하여 도메인 특화 일반화 개선 가능
- 로컬 호스팅 오픈소스 모델(LLaMA, Qwen 등) 적용으로 비용 절감 및 다양한 언어권 시장 적용 가능

#### (3) 변증법적 토론 메커니즘의 일반화 효과

Bull/Bear 연구원과 Aggressive/Neutral/Conservative 리스크 에이전트 간 다관점 토론은 다음과 같은 일반화 효과를 낸다:

- **단일 관점 편향 감소**: 다양한 시장 상황(상승장, 하락장, 횡보장)에 대한 대응력 향상
- **과적합 방지**: 특정 시장 조건에 편향된 결정을 토론을 통해 균형화
- Du et al. (2023)의 연구에 따르면, 다중 에이전트 토론은 팩추얼리티(factuality)와 추론 능력을 향상시킴

#### (4) AMZN·GOOGL·AAPL 3종목 일관된 성능

논문 부록에서 AMZN, GOOGL 결과를 제시하며 이를 명시:

> *"By including detailed analyses for AMZN and GOOGL, we aim to demonstrate the versatility of our approach in diverse market environments, thereby reinforcing the overall effectiveness and **generalizability** of our methodology."*

세 종목 모두에서 일관된 우월성 → 종목 간 일반화 능력의 초기 증거.

### 3.2 일반화 향상을 위한 미래 방향 (논문 기반 + 추론)

#### (A) 더 넓은 시장·섹터 적용

현재는 미국 대형 기술주 5종목에 한정. 일반화 향상을 위해:
- **중소형주**: 정보 비대칭이 큰 환경에서 LLM의 텍스트 분석 능력이 더욱 빛날 수 있음
- **비미국 시장**: 한국, 일본, 유럽 시장에서도 뉴스·소셜미디어 분석 적용 가능
- **암호화폐·채권·파생상품**: 멀티모달 정보 분석 능력의 다른 자산군 적용

#### (B) 장기 백테스트 검증

3개월 → 1년 이상으로 확장 시:
- 베어마켓(2022년 기술주 하락장), 크래시(2020년 코로나), 횡보장 등 다양한 장세 검증
- SR의 신뢰성 향상 (단기 낙폭 부재에 의한 편향 제거)

#### (C) 금융 특화 모델 통합

$$\text{일반화 성능} \propto f(\text{LLM}_{\text{백본}} \times \text{데이터}_{\text{품질}} \times \text{에이전트}_{\text{설계}})$$

BloombergGPT, FinGPT 등 금융 도메인 파인튜닝 모델을 특정 에이전트(기본적 분석가, 뉴스 분석가 등)의 백본으로 활용하면 도메인 특화 일반화 성능 향상 가능.

#### (D) 메모리 및 반성(Reflection) 메커니즘 추가

FinMem(Yu et al., 2023), FinAgent(Zhang et al., 2024b)에서 제안된 계층적 메모리 구조를 통합하면:
- 과거 거래 결과를 기억하고 전략을 자동으로 조정
- 특정 시장 조건에서의 실패 경험을 반성(self-reflection)하여 과적합 방지

---

## 4. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### (1) 조직 시뮬레이션 패러다임의 확립

TradingAgents는 단순 LLM + 금융 데이터 파이프라인을 넘어, **실제 조직의 의사결정 구조를 LLM 멀티에이전트로 시뮬레이션**하는 패러다임을 금융 AI 분야에 정착시킬 가능성이 높다. 이는 다음 분야로 확장될 수 있다:
- 포트폴리오 매니지먼트 조직 시뮬레이션
- M&A 의사결정 시뮬레이션
- 거시경제 정책 영향 분석

#### (2) 변증법적 AI 토론 방법론의 금융 적용 기준 마련

Bull/Bear 연구원 토론 구조는 **사회과학의 변증법적 추론**을 AI 시스템에 접목한 사례로, 추후 연구에서 다음을 탐구하는 기준점이 될 것이다:
- 토론 라운드 수($n$)의 최적화
- 토론 에이전트 수와 다양성이 성능에 미치는 영향
- 진행자(Facilitator) 에이전트의 편향 제어 방법

#### (3) 설명 가능한 AI(XAI) 트레이딩의 실용화 촉진

ReAct 기반의 완전한 거래 로그 공개는 금융 규제 환경에서 중요한 의미를 갖는다. EU AI Act, SEC 알고리즘 트레이딩 규제 등 금융 AI의 설명 가능성 요구에 부합하는 프레임워크로 활용될 수 있다.

#### (4) 오픈소스 생태계 기여

GitHub 코드 공개(`https://github.com/TauricResearch/TradingAgents`)를 통해:
- 학술 연구의 재현성(reproducibility) 향상
- 다양한 시장·자산군으로의 확장 연구 가속화
- 교육용 금융 AI 실습 플랫폼으로 활용 가능

### 4.2 앞으로 연구 시 고려할 점

#### ① 더 엄격한 통계적 검증

**현재 한계:** 3개월 단일 기간의 단일 시드 실험

**고려 사항:**
- 복수의 시장 조건(bull/bear/sideways market)에 걸친 장기 백테스트 필요
- 통계적 유의성 검정(예: Diebold-Mariano test for forecast accuracy)
- Walk-forward 검증 및 Monte Carlo 시뮬레이션 도입

#### ② 거래 비용 및 시장 영향 모델링

현재 실험은 **슬리피지(slippage), 거래 수수료, 시장 충격(market impact)**을 충분히 고려하지 않는다. 실제 환경에서는:

$$\text{실제 수익률} = \text{이론 수익률} - \text{거래 비용} - \text{시장 충격}$$

대규모 자본 적용 시 전략의 알파(alpha)가 감소할 수 있으며, 이를 정량적으로 모델링해야 한다.

#### ③ LLM 환각(Hallucination) 제어 메커니즘

금융 의사결정에서 LLM의 환각은 치명적 손실을 초래할 수 있다. 향후 연구에서는:
- 사실 검증(fact-checking) 에이전트 추가
- 소스 귀속(source attribution) 강화
- 자기 반성(self-reflection) 메커니즘(Ji et al., 2023) 통합

#### ④ 에이전트 토론 최적화

토론 라운드 수($n$)가 성능에 미치는 영향에 대한 **체계적인 ablation study** 필요:
- $n$이 너무 적으면 다양한 관점 수렴 부족
- $n$이 너무 많으면 계산 비용 과다 및 정보 손실 위험
- 동적 $n$ 결정 메커니즘(예: 합의 도달 시 조기 종료) 연구

#### ⑤ 실시간 트레이딩 환경 적용

논문도 미래 과제로 제시한 **라이브 트레이딩(live trading) 환경 배포**에서는:
- 레이턴시(latency) 문제: LLM 추론 시간 vs. 시장 가격 변동 속도
- API 비용 최적화: 덜 중요한 태스크에 더 작은 모델 활용
- 장애 복구(fault tolerance) 메커니즘 설계

#### ⑥ 다언어·다시장 확장

현재 영어 기반 정보에 특화. 한국어, 일본어, 중국어 등 비영어권 시장 적용 시:
- 다언어 LLM(예: Qwen, Baichuan)의 현지 뉴스·소셜미디어 분석 능력 평가
- 문화적 투자 심리 차이를 반영하는 Sentiment Analyst 설계

#### ⑦ 에이전트 역할 확장

논문이 제안하는 미래 방향인 "expanding agent roles"와 관련하여:
- **퀀트 모델 에이전트**: 통계적 팩터 모델 생성 (AlphaGPT 방식)
- **매크로 이코노미스트 에이전트**: 금리, 환율, GDP 등 거시경제 분석 전담
- **포트폴리오 최적화 에이전트**: 마코위츠 최적화, Black-Litterman 모델 적용

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구명 | 연도 | 방법론 | 강점 | 약점 | TradingAgents 대비 |
|---|---|---|---|---|---|
| **BloombergGPT** (Wu et al.) | 2023 | 금융 코퍼스 사전학습 | 금융 NLP 태스크 특화 | 직접 거래 의사결정 불가 | TA는 LLM을 의사결정 에이전트로 활용 |
| **FinGPT** (Yang et al.) | 2023 | LoRA 파인튜닝 | 저비용 도메인 적응 | 멀티에이전트 협업 부재 | TA는 협업 구조 강점 |
| **FinMem** (Yu et al.) | 2023 | 계층적 메모리 + 캐릭터 설계 | 장기 기억 기반 개선 | 단일 에이전트 | TA는 조직적 다중 에이전트 |
| **TradingGPT** (Li et al.) | 2023 | 계층적 메모리 + 다양한 캐릭터 | 다중 캐릭터 구현 | 감성 분류 중심 | TA는 실제 거래 실행까지 통합 |
| **SEP** (Koa et al.) | 2024 | RL + 메모리·반성 | 설명 가능한 예측 | 단일 에이전트 RL | TA는 규칙 기반 학습 없이 즉시 적용 |
| **FinAgent** (Zhang et al.) | 2024 | 멀티모달 + 계층 메모리 | 다양한 데이터 통합 | 조직 구조 부재 | TA는 역할 특화 팀 구조 추가 |
| **FinRobot** (Yang et al.) | 2024 | 오픈소스 멀티에이전트 플랫폼 | 플러그인 구조 | 거래 특화 토론 부재 | TA는 Bull/Bear 변증법 강점 |
| **FinCon** (Yu et al.) | 2024 | 개념적 언어 강화 멀티에이전트 | 개념 학습 기반 개선 | 복잡한 학습 파이프라인 | TA는 학습 없이 제로샷 적용 |
| **MetaGPT** (Hong et al.) | 2024 | 구조화 통신 프로토콜 | 소프트웨어 개발 성공 | 금융 도메인 미적용 | TA는 MetaGPT 아이디어를 금융에 적용 |
| **AlphaGPT** (Wang et al.) | 2023 | Human-in-the-loop 알파 마이닝 | 알파 팩터 자동 생성 | 직접 거래 결정 불가 | TA는 종단간(end-to-end) 거래 프레임워크 |
| **LLMFactor** (Wang et al.) | 2024 | 프롬프트로 수익 예측 팩터 추출 | 설명 가능한 팩터 | 단일 LLM | TA는 다중 에이전트 협업 |

### 비교 분석 종합

```
발전 궤적:
단일 LLM(뉴스 분석) → 파인튜닝 LLM → 단일 에이전트(메모리/RL) →
다중 에이전트(독립 수집) → TradingAgents(조직 시뮬레이션 + 구조화 통신)
```

**TradingAgents의 차별점:**
1. **조직 완결성**: 분석 → 연구 → 거래 → 리스크 → 승인의 5단계 전 과정 통합
2. **변증법 토론**: Bull/Bear 및 리스크 스펙트럼 토론 시스템화
3. **하이브리드 통신**: 구조화 문서 + 자연어 토론의 최적 조합
4. **교체 가능 백본**: 특정 LLM에 종속되지 않는 유연한 구조

---

## 참고 자료

**주요 논문 (본 논문 및 인용 문헌):**

1. **Xiao, Y., Sun, E., Luo, D., & Wang, W. (2025).** TradingAgents: Multi-Agents LLM Financial Trading Framework. *arXiv:2412.20138v7*. https://arxiv.org/abs/2412.20138

2. **Yao, S., et al. (2023).** ReAct: Synergizing Reasoning and Acting in Language Models. https://arxiv.org/abs/2210.03629

3. **Hong, S., et al. (2024).** MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework. https://arxiv.org/abs/2308.00352

4. **Du, Y., et al. (2023).** Improving Factuality and Reasoning in Language Models through Multiagent Debate. https://arxiv.org/abs/2305.14325

5. **Wu, S., et al. (2023).** BloombergGPT: A Large Language Model for Finance. https://arxiv.org/abs/2303.17564

6. **Yang, H., et al. (2023).** FinGPT: Open-Source Financial Large Language Models. https://arxiv.org/abs/2306.06031

7. **Yu, Y., et al. (2023).** FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design. https://arxiv.org/abs/2311.13743

8. **Li, Y., et al. (2023).** TradingGPT: Multi-Agent System with Layered Memory and Distinct Characters for Enhanced Financial Trading Performance. https://arxiv.org/abs/2309.03736

9. **Koa, K.J., et al. (2024).** Learning to Generate Explainable Stock Predictions Using Self-Reflective Large Language Models. https://dl.acm.org/doi/10.1145/3589334.3645611

10. **Zhang, W., et al. (2024).** A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist. https://arxiv.org/abs/2402.18485

11. **Yang, H., et al. (2024).** FinRobot: An Open-Source AI Agent Platform for Financial Applications Using Large Language Models. https://arxiv.org/abs/2405.14767

12. **Yu, Y., et al. (2024).** FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making. *arXiv:2407.06567*.

13. **Wang, S., et al. (2024).** QuantAgent: Seeking Holy Grail in Trading by Self-Improving Large Language Model. https://arxiv.org/abs/2402.03755

14. **Park, J.S., et al. (2023).** Generative Agents: Interactive Simulacra of Human Behavior. https://arxiv.org/abs/2304.03442

15. **Ji, Z., et al. (2023).** Towards Mitigating Hallucination in Large Language Models via Self-Reflection. https://arxiv.org/abs/2310.06271

16. **Lopez-Lira, A. & Tang, Y. (2023).** Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models. https://arxiv.org/abs/2304.07619

17. **OpenAI (2024).** Learning to Reason with LLMs – OpenAI o1 Model. https://openai.com/index/learning-to-reason-with-llms/

18. **Xie, Q., et al. (2023).** PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance. https://arxiv.org/abs/2306.05443

19. **Wang, M., et al. (2024).** LLMFactor: Extracting Profitable Factors through Prompts for Explainable Stock Movement Prediction. https://arxiv.org/abs/2406.10811

20. **Schulman, J., et al. (2017).** Proximal Policy Optimization Algorithms. https://arxiv.org/abs/1707.06347

> **정확도 관련 주의사항:** 본 답변은 제공된 논문 PDF(arXiv:2412.20138v7)를 직접 참조하여 작성하였습니다. 2020년 이후 비교 연구 섹션의 일부 내용(특히 각 연구의 정량적 성능 수치)은 논문 원문에 인용된 참고문헌 수준에서 서술하였으며, 인용 논문 전체를 직접 검토하지 않은 부분에 대해서는 논문 내 서술에 근거하였습니다.
