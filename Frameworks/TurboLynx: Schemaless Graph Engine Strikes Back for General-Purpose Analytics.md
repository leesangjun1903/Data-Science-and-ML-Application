# TurboLynx: Schemaless Graph Engine Strikes Back for General-Purpose Analytics

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

기존 Graph Database Management System(GDBMS)들이 복잡한 분석 쿼리(group-by, aggregation 등)에서 성능 병목을 겪는 근본 원인은 **스키마리스(schemaless) 처리가 스토리지, 쿼리 처리, 최적화 레이어 전반에 걸쳐 핵심 설계 요구사항으로 다뤄지지 않았기 때문**이라고 주장한다.

이를 해결하기 위해 TurboLynx는 **스키마리스 속성을 시스템의 모든 레이어에 통합적으로 적용한 최초의 범용 그래프 분석 엔진**을 제안한다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| 문제 진단 | 스키마리스 처리 부재가 스토리지·실행·최적화 성능 저하의 주요 원인임을 체계적으로 분석 |
| TurboLynx 프로토타입 | 스토리지~최적화 전 레이어를 아우르는 완전한 기능의 시스템 구현 (약 57k LOC 신규 코드) |
| 포괄적 성능 평가 | LDBC SNB Interactive, TPC-H, DBpedia 벤치마크에서 최대 **183.9×** 성능 향상 실증 |
| 오픈소스 공개 | [https://github.com/postechdblab/TurboLynx](https://github.com/postechdblab/TurboLynx) |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

논문은 기존 스키마리스 GDBMS(Neo4j, Memgraph 등)가 분석 쿼리에서 느린 세 가지 구체적 원인을 식별한다:

**① 스토리지 문제: Row-based 레이아웃**
- Neo4j, Memgraph는 각 튜플에 스키마 정보를 직접 내장(row-based)
- 쿼리 시 튜플별 스키마 파싱(per-tuple schema interpretation) 오버헤드 발생

**② 쿼리 처리 문제: 벡터화 불가**
- 다양한 스키마의 데이터가 혼재하므로 Volcano-style 처리에 의존
- SIMD 기반 벡터화 활용이 어려움

**③ 쿼리 최적화 문제: 미성숙한 옵티마이저**
- Neo4j는 동등성(equality), 범위(range) 기반 단순 휴리스틱 선택도 사용
- Orca, Calcite 수준의 최적화 룰의 약 20% 수준만 보유
- 속성 수준 통계(attribute-level statistics) 부재

### 2.2 제안 방법 및 수식

#### (A) Cost-based Graphlet Chunking (CGC) — 스토리지 레이어

**핵심 아이디어**: 속성 집합이 유사한 노드/엣지를 **graphlet**이라는 단위로 묶어 컬럼 형식으로 저장

**정의**: 두 graphlet $gl_i, gl_j \in \mathcal{H}$ 간의 비용-인식 유사도(cost-aware similarity):

$$casim(gl_i, gl_j, \mathcal{H}) = c(\mathcal{H}) - c(\mathcal{H}')$$

여기서 $\mathcal{H}' = (\mathcal{H} - \{gl_i, gl_j\}) \cup \{gl_i \oplus gl_j\}$ 이며 $\oplus$는 두 graphlet의 병합 연산이다.

전체 비용 함수 $c(\mathcal{H})$:

$$c(\mathcal{H}) = C_{sch} \cdot |\mathcal{H}| + C_{null} \cdot \sum_{gl \in \mathcal{H}} \Gamma(gl) + C_{vec} \cdot \sum_{gl \in \mathcal{H}} \Psi(|gl|)$$

각 항의 의미:
- $C_{sch} \cdot |\mathcal{H}|$: graphlet 수 (스키마 수) 페널티
- $C_{null} \cdot \sum_{gl \in \mathcal{H}} \Gamma(gl)$: null 엔트리 수 페널티 ($\Gamma$는 null 값 카운터)
- $C_{vec} \cdot \sum_{gl \in \mathcal{H}} \Psi(|gl|)$: 벡터화 미활용 오버헤드, $\Psi(|gl|) = \frac{\kappa}{|gl|}$ (단, $|gl| < \kappa$일 때, $\kappa = 1024$)

이를 전개하면:

$$casim(gl_i, gl_j, \mathcal{H}) = C_{sch} + C_{null} \cdot (\Gamma(gl_i) + \Gamma(gl_j) - \Gamma(gl_i \oplus gl_j)) + C_{vec} \cdot (\Psi(|gl_i|) + \Psi(|gl_j|) - \Psi(|gl_i| + |gl_j|))$$

$casim(gl_i, gl_j, \mathcal{H}) > 0$ 이면 병합이 유익하다고 판단한다.

실험에서 사용된 기본 가중치: $C_{sch} = 100, \; C_{null} = 0.3, \; C_{vec} = 10000$

#### (B) Schema Index (SI) — 스토리지 레이어

속성 이름을 키로, 해당 속성을 포함하는 graphlet ID 목록을 값으로 갖는 역색인(inverted index):

$$SI(\mathcal{H}, a) = \{i \mid \exists gl \in \mathcal{H} \; s.t. \; a \in sch(gl) \land i = id(gl)\}$$

여러 속성 $(a_1, a_2, ..., a_n)$을 동시에 조회할 때는 집합 교차(set intersection) 사용:

$$\bigcap_{i=1}^{n} SI(\mathcal{H}, a_i)$$

#### (C) Shared Schema Row Format (SSRF) — 쿼리 처리 레이어

**문제**: 바이너리 조인 시 중간 스키마 폭발(schema bloating). 예를 들어 `MATCH (a:A)-[]->(b:B)-[]->(c:C)`의 경우 최악 $|\mathcal{H}(A)| \times |\mathcal{H}(B)| \times |\mathcal{H}(C)|$개의 스키마 조합 발생.

**해결책**: 스키마 정의를 튜플과 분리하여 별도 `schema infos` 테이블에 저장.
- `OffsetArr`: TupleStore 내 각 튜플의 바이트 오프셋
- `total_size`: 해당 스키마의 튜플 크기
- `offset_infos[i]`: i번째 속성의 오프셋 (값이 $-1$이면 null)

이를 통해 sparse한 컬럼을 null로 채우지 않고, 동일 스키마 튜플들이 하나의 메타데이터 엔트리를 공유하게 한다.

#### (D) Graphlet Early Merge (GEM) — 최적화 레이어

**문제**: `PushJoinBelowUnionAll` 룰 적용 시 플랜 탐색 공간이 지수적으로 증가.

**해결책**: 조인 열거(join enumeration) 이전에 graphlet을 가상 graphlet으로 조기 병합:

1. graphlet을 랜덤으로 그룹화하여 가상 graphlet 형성 (기본 2개 그룹)
2. `PushJoinBelowUnionAll` 룰로 논리적 동치 플랜 열거 및 서브셋 평가
3. Greedy Operator Ordering을 이용해 각 대안 플랜의 최적 조인 순서 결정 및 비용 계산

### 2.3 모델 구조

```
[데이터 입력 (PGM)]
        ↓
[Graphlet Manager]
  - Cost-based Graphlet Chunking (CGC)
  - Schema Index (SI) 구축
  - CSR Index 구축
        ↓
[Storage Manager] → Graph-Native Storage (Columnar Graphlets)
        ↓
[Graph Query Optimizer] ← Catalog Manager (통계, 히스토그램)
  - Orca 기반
  - Graphlet Early Merge (GEM)
  - Graph-aware operators, rules, cost models
        ↓
[Vectorized Graph Query Processor]
  - Unified Schema + Validity Vector
  - Shared Schema Row Format (SSRF)
  - AdjIdxJoin, VarlenJoin, ShortestPath 등
```

**구현**: DuckDB (표현식 평가기, 내장 함수, Vector/DataChunk 자료구조) + Orca (최적화기) 재활용, 신규 57k LOC 추가 (총 246k LOC)

### 2.4 성능 향상

| 벤치마크 | 비교 대상 | 최대 향상 배율 | 평균 향상 배율 |
|---|---|---|---|
| LDBC SNB Interactive SF100 | DuckPGQ | **183.9×** | 29.78× |
| LDBC SNB Interactive SF100 | Neo4j | 10.47× | — |
| LDBC SNB Interactive SF100 | Kuzu | 106.89× | — |
| LDBC SNB Interactive SF100 | Umbra (최고 경쟁자) | 7.74× | — |
| TPC-H SF10 | Memgraph | 44.63× | — |
| DBpedia | Neo4j | 86.14× | 27.37× |
| DBpedia | Kuzu | 18.88× | — |

**구성 요소별 효과**:
- CGC (scan with selection): 최대 **5319×** 향상
- SSRF: 최대 **2.6×** 향상
- GEM: 실행 시간 **26.7%** 감소 (컴파일 오버헤드 6.2% 증가)

### 2.5 한계

1. **배치 업데이트만 지원**: 인터리브 쿼리와 업데이트 동시 처리(OLTP 스타일) 미지원
2. **트랜잭션 보장 미구현**: Delta Lake처럼 트랜잭션 기능 확장 가능성은 언급하나 현재 범위 밖
3. **카디널리티 추정 한계**: 경로 표현식 및 최단 경로 쿼리에 대해 정확한 카디널리티 추정이 어려워 휴리스틱(hop count × one-hop 카디널리티)에 의존
4. **단일 스레드 평가**: 모든 실험이 단일 스레드로 수행되어 멀티스레드/분산 환경에서의 성능은 미검증
5. **GEM의 비결정론적 탐색**: 랜덤 그룹화 + 타임아웃 기반 휴리스틱으로 최적 플랜 보장 불가
6. **가중치 민감도**: CGC의 $C_{sch}, C_{null}, C_{vec}$ 값이 경험적으로 설정되어 데이터셋 특성에 따라 튜닝 필요
7. **GraphScope, Memgraph의 DBpedia 데이터 로딩 실패**: 일부 비교 실험 누락

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 스키마풀/스키마리스 양쪽 워크로드 처리

TurboLynx의 가장 주목할 일반화 강점은 **사전 정의된 스키마(schemaful)와 스키마리스(schemaless) 데이터 모두에서 우수한 성능**을 발휘한다는 점이다.

- **LDBC SNB / TPC-H**: 고정 스키마 데이터 → 기존 RDBMS/GDBMS 대비 우위 확인
- **DBpedia**: 2,796개의 고유 속성, 282,764개의 고유 속성 집합을 가진 실세계 지식 그래프 → 스키마 다양성이 극단적인 환경에서도 최대 86.14× 향상

이는 CGC의 비용 기반 클러스터링이 데이터 분포에 적응적으로 동작함을 시사한다. 균일한 데이터에서는 단일 또는 소수의 graphlet으로 수렴하고, 이질적인 데이터에서는 최적의 granularity로 분할된다.

### 3.2 CGC의 적응적 일반화

CGC의 비용 함수는 세 가지 상충 요소를 동시에 고려한다:

$$c(\mathcal{H}) = C_{sch} \cdot |\mathcal{H}| + C_{null} \cdot \sum_{gl \in \mathcal{H}} \Gamma(gl) + C_{vec} \cdot \sum_{gl \in \mathcal{H}} \Psi(|gl|)$$

이 설계는 데이터 분포에 관계없이 다음과 같이 일반화한다:

| 데이터 특성 | CGC 동작 | 결과 |
|---|---|---|
| 매우 균일한 스키마 | graphlet 병합 촉진 ($C_{null}$ 페널티 낮음) | 단일/소수 graphlet → 높은 벡터화 효율 |
| 매우 이질적인 스키마 | 과도한 병합 억제 ($C_{null}$ 페널티 높음) | 다수 graphlet, null 최소화 |
| 소규모 graphlet 다수 | $\Psi(gl) = \kappa(gl)$ 페널티로 병합 유도 | 벡터화 파이프라인 활용 극대화 |

### 3.3 스키마 인덱스를 통한 레이블-비의존적 쿼리 일반화

Schema Index(SI)는 레이블 없이 속성 이름만으로 쿼리하는 **label-agnostic** 쿼리를 효율적으로 처리한다. 이는 실세계에서 사용자가 그래프의 스키마를 완전히 알지 못하는 상황—즉, 오픈 데이터, 진화하는 지식 그래프(evolving knowledge graph)—에서 일반화 성능을 높인다.

DBpedia 실험에서 TurboLynx가 Q14(속성 기반 필터)에서 Neo4j 대비 **7,377×** 성능 향상을 달성한 것은 SI의 일반화 능력을 직접적으로 보여준다.

### 3.4 SSRF의 멀티-홉 쿼리 일반화

SSRF는 홉 수 증가에 따른 스키마 폭발 문제를 억제한다. 실험에서 1~5홉 쿼리에 걸쳐 일관되게 최대 2.1× 이상의 성능 우위를 유지했으며, 다속성 반환 시 최대 2.6× 향상을 보였다. 이는 복잡한 그래프 탐색 패턴에서의 일반화 안정성을 의미한다.

### 3.5 GEM의 플랜 다양성을 통한 일반화

GEM은 graphlet을 그룹화하여 각 그룹에 대해 다른 조인 순서를 탐색한다. DBpedia 논문에서 언급된 바와 같이, 비디오 게임 도메인처럼 **인기도·시간·도메인에 따라 스키마 분포가 달라지는 경우**, 각 graphlet에 최적화된 조인 순서 선택이 일반화 성능에 기여한다.

### 3.6 증분 업데이트 시 일반화 (VP-CGC)

VP-CGC(벡터화 페널티 기반 트리거) 전략은 업데이트 누적으로 인한 소규모 graphlet 증가를 자동으로 탐지하고 CGC 재계산을 트리거한다. 이는 **동적으로 변화하는 데이터에 대한 시간적 일반화**를 제공한다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 향후 연구에 미치는 영향

#### ① 스키마리스 처리의 1급 시민화 (First-class Schemaless)
TurboLynx는 스키마리스 처리를 사후에 추가하는 것이 아니라 **설계 초기부터 핵심 요구사항**으로 삼아야 함을 실증했다. 이후 그래프 DB 연구에서 스키마 유연성과 분석 성능의 동시 달성이 중요한 연구 주제로 부상할 것이다.

#### ② Graphlet 개념의 확장 가능성
- 분산 그래프 처리: graphlet을 분산 노드의 기본 파티션 단위로 활용
- 스트리밍 그래프: 실시간 graphlet 재조직화 알고리즘 연구
- 그래프 신경망(GNN): 이질적 속성을 가진 그래프에서 graphlet 기반 미니배치 구성

#### ③ 스키마리스 특화 카디널리티 추정
논문이 경로 쿼리와 최단 경로 쿼리의 카디널리티 추정을 휴리스틱에 의존하는 한계를 인정했다. 이는 **스키마리스 환경에서의 학습 기반(learned) 카디널리티 추정** 연구를 촉진할 것이다.

#### ④ ISO/GQL 표준과의 연계
논문이 Cypher와 ISO/GQL의 연관성을 언급한 점에서, TurboLynx의 설계가 GQL 표준 구현의 참조 아키텍처로 활용될 수 있다.

#### ⑤ RDBMS 옵티마이저의 그래프 적응
Orca를 그래프-aware하게 확장한 접근법은 PostgreSQL, DuckDB 등 기성 RDBMS의 옵티마이저를 그래프 쿼리에 적응시키는 연구의 청사진을 제공한다.

### 4.2 향후 연구 시 고려할 점

#### ① 다중 스레드 및 분산 환경
현재 단일 스레드 평가만 수행되었다. 멀티코어/분산 환경에서 graphlet 병렬 스캔, CSR 인덱스의 캐시 지역성, GEM의 병렬 플랜 탐색 등에 대한 연구가 필요하다.

#### ② 트랜잭션 지원
배치 업데이트만 지원하는 현재 구조를 넘어, Delta Lake 방식의 트랜잭션 보장 확장이 필요하다. 특히 graphlet 재구성(CGC 재계산) 중 쿼리 가용성(availability) 유지가 과제다.

#### ③ 가중치 자동 튜닝
$C_{sch}, C_{null}, C_{vec}$ 가중치의 경험적 설정 의존성을 극복하기 위해, 워크로드 피드백(workload feedback) 또는 강화학습 기반 자동 튜닝 메커니즘 연구가 요구된다.

#### ④ GNN/ML 워크로드와의 통합
지식 그래프 임베딩, GNN 학습 등 ML 워크로드가 그래프 DB와 통합되는 추세에서, TurboLynx의 graphlet 구조가 특징 추출(feature extraction) 파이프라인과 어떻게 결합될 수 있는지 검토해야 한다.

#### ⑤ 카디널리티 추정의 고도화
경로 쿼리 및 재귀 쿼리에서의 정확한 카디널리티 추정은 미해결 과제다. 학습 기반 추정기(Learned Cardinality Estimator)를 graphlet 통계와 결합하는 방향을 고려해야 한다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 비교는 논문 본문에서 직접 참조하거나 평가에 포함된 시스템들을 중심으로 정리한다.

| 시스템/연구 | 발표 연도 | 스토리지 | 스키마 | 그래프 특화 | 주요 특징 | TurboLynx 대비 |
|---|---|---|---|---|---|---|
| **Kuzu** [CIDR 2023] | 2023 | Columnar | 사전정의 단일 | WCOJ, 팩터화 실행 | 그래프 쿼리 가속화, worst-case optimal join | 최대 106.89× 느림 (복잡 쿼리 최적화 부족) |
| **DuckPGQ** [CIDR 2023] | 2023 | Columnar | 사전정의 단일 | SQL/PGQ 확장 | DuckDB 기반 속성 그래프 쿼리 | 최대 183.9× 느림 (스키마리스 미지원) |
| **GRainDB** [CIDR 2022] | 2022 | Columnar | 사전정의 단일 | 인접성 인덱스 | DuckDB 내부 수정, 그래프 조인 연산자 | 스키마리스 PGM 미지원 |
| **GraphScope** [VLDB 2021] | 2021 | Columnar | 단일 스키마 | 분산 그래프 처리 | 빅 그래프 통합 처리 엔진 | 스키마 유연성 없음, 쿼리 지원 부분적 |
| **Umbra** [VLDB 2020~] | 2020~ | Columnar | 사전정의 단일 | HTAP, 코드 생성 | 데이터 중심 코드 생성, Diamond hardened join | LDBC SF100에서 7.74× 느림 |
| **Diamond Hardened Joins** [VLDB 2024] | 2024 | — | — | Lookup+Expand 분리 | 다이아몬드 형태 조인 폭발 방지 | TurboLynx와 상호 보완적 |
| **JSON Tiles** [SIGMOD 2021] | 2021 | 컬럼(부분) | 반구조화 | 없음 | JSON 분석 가속화 | 그래프 탐색 미지원, 순수 분석에 특화 |
| **GMMSchema** [EDBT 2022] | 2022 | — | 스키마 발견 | 없음 | 계층적 클러스터링으로 스키마 추론 | CGC 대비 6% 빠른 컴파일, 실행 성능 열위 |
| **DiscoPG** [VLDB 2022] | 2022 | — | 스키마 발견 | 없음 | 스키마 발견 및 탐색 도구 | TurboLynx와 상호 보완(CGC 입력으로 활용 가능) |
| **μSlope** [OSDI 2024] | 2024 | 컬럼(per-schema) | 스키마별 분리 | 없음 | 로그 데이터 반구조화 처리 | SA 방식과 유사, 소규모 graphlet 문제 동일 |

### 핵심 비교 관점 요약

**스키마 유연성 vs 성능 트레이드오프**:
- Kuzu, DuckPGQ, GRainDB는 사전 정의된 스키마 강제 → 스키마리스 환경에서 null 폭발 또는 다중 테이블 관리 문제
- Neo4j, Memgraph는 스키마리스 지원하나 row-based → 분석 성능 저하
- **TurboLynx만이 스키마리스 + 컬럼형 + 고성능 분석을 동시에 달성**

**최적화 성숙도**:
- Kuzu: 자체 옵티마이저 (초기 단계) → 복잡 쿼리에서 필터 푸시다운 실패
- TurboLynx: 성숙한 Orca 옵티마이저를 그래프-aware하게 확장 → 유리한 출발점

---

## 참고 자료

**논문 원문**
- Taesung Lee, Jaehyun Ha, Byungchul Tak, Wook-Shin Han. **"TurboLynx: Schemaless Graph Engine Strikes Back for General-Purpose Analytics."** *Proceedings of the VLDB Endowment*, Vol. 19, No. 6, pp. 1250–1263, 2026. DOI: 10.14778/3797919.3797932

**논문 내 직접 인용 참고문헌 (본 답변에 관련된 주요 항목)**
- [13] Boncz et al. "MonetDB/X100: HyperPipelining Query Execution." CIDR 2005.
- [15] Bonifati et al. "Hierarchical clustering for property graph schema discovery." EDBT 2022.
- [22] Durner et al. "JSON Tiles: Fast analytics on semi-structured data." SIGMOD 2021.
- [25] Fegaras. "A New Heuristic for Optimizing Large Queries." DEXA 1998.
- [26] Feng et al. "Kùzu graph database management system." CIDR 2023.
- [40] Jin et al. "GRainDB: A Relational-core Graph-Relational DBMS." CIDR 2022.
- [53] Raasveldt and Mühleisen. "DuckDB: an embeddable analytical database." SIGMOD 2019.
- [60] Soliman et al. "Orca: a modular query optimizer architecture for big data." SIGMOD 2014.
- [62] Sun et al. "SQLGraph: An efficient relational-based property graph store." SIGMOD 2015.
- [64] ten Wolde et al. "DuckPGQ: Efficient Property Graph Queries in an analytical RDBMS." CIDR 2023.
- [69] Wang et al. "μSlope: High Compression and Fast Search on Semi-Structured Logs." OSDI 2024.
- [12] Birler et al. "Robust join processing with diamond hardened joins." PVLDB 2024.

**소스코드**
- TurboLynx GitHub: [https://github.com/postechdblab/TurboLynx](https://github.com/postechdblab/TurboLynx)

> **⚠️ 정확도 주의**: 본 답변은 제공된 논문 PDF 전문을 직접 분석하여 작성되었습니다. 논문에 명시되지 않은 외부 최신 연구(2020년 이후)와의 비교는 논문 내 참고문헌 및 평가 섹션에 등장한 시스템들로 한정하였으며, 논문 밖의 추가 연구에 대한 독립적 주장은 포함하지 않았습니다.
