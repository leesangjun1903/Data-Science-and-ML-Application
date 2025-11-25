# ColabFold: making protein folding accessible to all

### 1. 핵심 주장과 주요 기여

ColabFold의 핵심 주장은 **복잡한 계산 자원이 없는 연구자들도 고정확도의 단백질 구조 예측에 접근할 수 있어야 한다**는 것입니다. 이를 위해 ColabFold는 MMseqs2 기반의 40-60배 빠른 상동성 검색과 최적화된 모델 활용을 결합하여, 단일 GPU 서버에서 **하루에 약 1,000개의 단백질 구조를 예측**할 수 있는 환경을 제공합니다.[1]

주요 기여는 다음과 같습니다:

- **계산 속도 획기적 개선**: MSA 생성 속도 40-60배, 배치 예측 약 90배 가속화로 AlphaFold2 대비 평균 5배 빠른 단일 예측 달성[1]

- **접근성 혁명**: Google Colaboratory와의 통합으로 완전히 무료이고 강력한 GPU에 접근 가능한 플랫폼 제공

- **정확도 유지**: CASP14 벤치마크에서 AlphaFold2와 동등한 성능(TM-score 0.887-0.888)을 유지하면서도 속도 획기적 향상[1]

- **복합체 예측 능력 확장**: 단일 단백질뿐 아니라 호모머(homo-oligomeric)와 헤테로머(hetero-oligomeric) 복합체 예측 지원[1]

- **혁신적 데이터베이스 구축**: 메타게놈 시퀀스를 통합한 ColabFoldDB로 진핵생물 단백질 다양성 향상[1]

### 2. 해결하는 문제와 제안하는 방법

#### 2.1 문제 정의

AlphaFold2의 혁신에도 불구하고 두 가지 심각한 제약이 존재했습니다:[1]

1. **MSA 생성의 계산 병목**: 민감한 상동성 탐지(HMMer, HHblits)는 **단일 단백질 당 수 시간의 처리 시간**이 필요하고, 2TB 이상의 저장공간 필요[1]

2. **자원 접근성 불평등**: GPU RAM과 저장공간이 제한된 연구자들은 AlphaFold2 사용 불가능

#### 2.2 제안하는 방법

**MMseqs2 기반 신속 MSA 생성**

ColabFold는 다음 수식으로 표현되는 **3단계 반복 검색 워크플로우** 도입:

$$E_{\text{MSA}} = \sum_{i=1}^{3} \text{search}(Q, DB_i, P_i)$$

여기서:
- $$E_{\text{MSA}}$$: 전체 MSA 생성 수행도
- $$Q$$: 쿼리 시퀀스
- $$DB_i$$: i번째 데이터베이스 (UniRef30 → BFD/MGnify 또는 ColabFoldDB)
- $$P_i$$: i번째 검색에서 생성된 프로필[1]

각 반복 단계는 이전 단계의 프로필을 입력으로 사용하여 **10배의 가속화** 달성 (UniRef30의 2,930만 시퀀스만 검색, 모든 UniRef100의 2억 7,750만 시퀀스 대비)[1]

**다양성 인식 필터링 (Diversity-Aware Filtering)**

새로운 필터링 메커니즘:

```math
\text{MSA}_{\text{filtered}} = \arg\max_{\text{MSA}} \left\{ \text{coverage} \times \text{diversity} - \lambda \times \text{size} \right\}
```

구체적으로, 시퀀스 정체성(sequence identity) 버킷을 기반으로 한 4단계 필터링:
- 1단계: UniRef30 클러스터 확장 중 개별 클러스터 당 최대 시퀀스 정체성 95% 필터
- 2단계: 정렬 후 쿼리 스캔 점수(qsc) ≥ 0.8 임계값 적용
- 3단계: MSA 구성 중 5개 정체성 버킷 $$[0.0-0.2], (0.2-0.4], (0.4-0.6], (0.6-0.8], (0.8-1.0]$$에서 각각 최대 3,000개 서열 유지[1]

이를 통해 **충분한 MSA 다양성을 유지하면서도 메모리 요구사항을 90% 이상 감소**시킵니다.

**확장된 환경 데이터베이스 (ColabFoldDB)**

$$\text{ColabFoldDB} = \text{BFD/MGnify} \cup \{\text{SMAG, MetaEuk, TOPAZ, MGV, GPD, MetaClust}\}$$

BFD와 MGnify의 **중복성 제거** 방식:

```math
\text{cluster assignment} = \begin{cases} \text{BFD cluster} & \text{if } \text{ID} > 30\% \text{ and } \text{cov} > 90\% \\ \text{new cluster} & \text{otherwise} \end{cases}
```

결과적으로 2.5억 개에서 5.13억 개 시퀀스로 확장되었으며, 각 클러스터에서 **상위 10개 다양 서열만 보존**하여 최종 RAM 요구사항을 2.5배 감소(517GB → 84GB)[1]

#### 2.3 모델 구조

ColabFold는 **3개의 주요 구성 요소**로 구성:[1]

1. **MMseqs2 기반 상동성 검색 서버**: UniRef100, PDB70, 환경 시퀀스 세트에 대한 고속 정렬

2. **Python 라이브러리**: 검색 서버와 통신, 구조 추론을 위한 입력 특징 준비, 결과 시각화, 명령줄 인터페이스 구현

3. **Jupyter 노트북**: 기본, 고급, 배치 사용을 위한 다양한 인터페이스[1]

**단백질 복합체 예측 모드**

AlphaFold2의 상대 위치 인코딩의 한계(|i−j| ≥ 32에서 동일한 인코딩)를 활용한 **잔기 지수(residue index) 조작**:

$$\text{pos}_{\text{rel}} = \min(|i - j|, 32)$$

서로 다른 단백질 사이의 거리를 32 이상으로 유지하면 독립적인 폴리펩타이드 체인으로 인식:

$$\text{residue index offset} > 32 \Rightarrow \text{separate chains}$$

호모올리고머의 경우 각 성분마다 MSA를 복사하고, 헤테로올리고머의 경우 분류학 정체성을 기반으로 서열을 짝지음 방식 적용[1]

#### 2.4 성능 향상 메커니즘

**재컴파일 회피 최적화**:

전체 런타임 $$T_{\text{total}}$$:
$$T_{\text{total}} = T_{\text{compile}} + \sum_{i=1}^{n} (T_{\text{inference}, i} + T_{\text{recycle}, i})$$

ColabFold는:
- 5개 모델 중 1개만 컴파일, 나머지는 가중치 교체: $$\Delta T_{\text{compile}} = -7$$ min (템플릿 없음), $$-5$$ min (템플릿 있음)
- 배치 처리에서 입력 시퀀스를 길이로 정렬하고 10% 여백으로 패딩하여 재컴파일 회피
- 결과적으로 배치 예측에서 **약 90배 가속화** 달성[1]

**조기 종료 기준 (Early Stop)**:

$$\text{Stop} = \begin{cases} \text{true} & \text{if } \max(\text{pLDDT}) \geq \tau_1 \text{ or } \text{TM-score} \geq \tau_2 \\ \text{false} & \text{otherwise} \end{cases}$$

pLDDT ≥ 85 임계값으로 설정 시, M. jannaschii 단백질체 1,762개 예측을 **48시간 내**에 완료하면서도 평균 pLDDT 손실 최소화 (AlphaFold2: 89.75 vs ColabFold Stop≥85: 88.78)[1]

**재사이클 횟수 최적화**:

기본값 3회에서 12회로 증가 시 CASP14 벤치마크의 평균 TM-score **0.887에서 0.898로 향상** (MSA 정보 부족한 타겟 개선)[1]

### 3. 성능 평가

#### 3.1 정확도 비교

**CASP14 타겟에 대한 평가**

자유 모델링 (Free-Modeling) 타겟의 평균 TM-score:

| 방법 | 평균 TM-score |
|------|--------------|
| ColabFold-AlphaFold2-BFD/MGnify | 0.826 |
| ColabFold-AlphaFold2-ColabFoldDB | 0.818 |
| AlphaFold2 | 0.79 |
| AlphaFold-Colab | 0.744 |
| ColabFold-RoseTTAFold | 0.62 |

전체 CASP14 타겟의 TM-score:

| 방법 | 평균 TM-score |
|------|--------------|
| ColabFold-AlphaFold2-BFD/MGnify | 0.887 |
| ColabFold-AlphaFold2-ColabFoldDB | 0.886 |
| AlphaFold2 | 0.888 |
| ColabFold-RoseTTAFold | 0.754 |

**핵심 통찰**: ColabFold는 AlphaFold2와 **통계적으로 동등한 정확도**를 달성하면서도 **5배 빠른 속도** 제공[1]

#### 3.2 복합체 예측 성능

ClusPro 데이터셋의 17개 타겟에서 DockQ 점수:

| 예측 모드 | 평균 성능 |
|----------|----------|
| ColabFold-AlphaFold-Multimer | 최고 정확도 |
| ColabFold-Residue-Index | 일부 타겟에서 더 우수 |

예시 사례:[1]
- Homo-six-mer 복합체 성공 예측
- 3개 단백질로 구성된 D-메티오닌 수송 시스템 정확 모델링
- 인간 핵공 복합체(120 MDa) 구조 결정을 위한 cryo-EM 보조[1]

### 4. 일반화 성능 향상 가능성

#### 4.1 현재 한계

**MSA 깊이에 대한 의존성**

AlphaFold2처럼 ColabFold도 MSA 질의 영향을 받습니다:

$$\text{Accuracy} = f(\text{MSA depth}, \text{MSA diversity}, \text{sequence complexity})$$

- **고아 단백질 (Orphan proteins)**: 상동 서열이 적은 단백질의 경우 성능 저하
- **설계 단백질 (Designed proteins)**: 자연에 존재하지 않아 MSA 정보 전무

#### 4.2 최신 연구 기반 개선 방향

**1. 언어 모델 기반 접근법 (MSA-Free Methods)**

ESMFold와 같은 **단백질 언어 모델(PLM) 기반 방법**이 등장:[2][3]

$$\text{ESMFold}: \text{Sequence} \rightarrow \text{Embedding}_{\text{PLM}} \rightarrow \text{Structure}$$

- **장점**: 상동 서열 부족한 단백질에 더 우수한 일반화 성능
- **성능**: AlphaFold2와 비교하여 **~40초의 빠른 예측** 유지하면서 고아 단백질 성능 개선[4]
- **한계**: 구조적 상세도(atomic-level 정확도)에서 여전히 MSA 기반 방법이 우수

**2. 디퓨전 기반 생성 모델**

FoldingDiff와 같은 **확산 모델(Diffusion Model)** 기반 접근법:[5]

$$x_t = x_{t-1} - \nabla_x \log p(x_{t-1} | \text{conditions})$$

- 단백질 백본을 **내부 각도(inter-residue angles)** 로 표현하여 기하학적 불변성 확보
- 회전 및 평행이동 불변 표현으로 **복잡한 동등 변환 네트워크 제거**
- 자연 단백질 분포와 **높은 일관성** 달성

**3. AlphaFold3의 아키텍처 혁신**[6]

2024년 발표된 AlphaFold3의 **확산 기반 아키텍처**:

$$\text{AF3}: \text{joint representation} \rightarrow \text{diffusion module} \rightarrow \text{atomic coordinates}$$

- **MSA 의존성 감소**: MSA가 주변 역할로 격하되어 **MSA 깊이가 얕아도 성능 유지**
- **멀티모달 예측**: 단백질, DNA/RNA, 리간드, 이온 등 모든 생물분자 통일 예측
- **우수한 일반화**: 항체-항원 복합체에서 기존 방법 대비 **상당한 성능 향상**[7][6]

**4. 하이브리드 접근법: AF3Complex**

최근 연구(2025)에서 AF3의 MSA 생성 알고리즘을 특화시킨 **AF3Complex** 제안:[8]

$$\text{AF3Complex} = \text{AF3} + \text{specialized MSA} + \text{interface-focused confidence}$$

- ClusPro 데이터셋에서 **평균 DockQ 0.735** (AlphaFold3: 0.718)
- 펩타이드 및 항체-항원 복합체에서 **상태 최고 성능(SOTA)** 달성

#### 4.3 논문의 맥락에서 향상 전략

ColabFold 기반 일반화 성능 향상을 위한 3가지 전략:

**전략 1: 다중 재사이클을 통한 내부 반복**

$$\text{Accuracy}_{\text{final}} = \lim_{R \to \infty} \text{Accuracy}(R \text{ recycles})$$

- 기본값 3회에서 12회로 증가하면 **약 1.2% 정확도 향상** (0.887 → 0.898)
- 특히 **MSA 정보 부족한 타겟에서 효과적**[1]

**전략 2: 환경 데이터베이스 확대를 통한 MSA 품질 향상**

$$\text{MSA}_{\text{quality}} \propto \sum \text{metagenomic sources}$$

ColabFoldDB의 메타게놈 데이터 통합으로 **Pfam < 30멤버 도메인에서 더 다양한 MSA 생성**[1]

**전략 3: 스토캐스틱 예측을 통한 불확실성 샘플링**

$$\text{Ensemble Predictions} = \{\text{pred}_i | i \in [1, N], \text{seed}_i \text{ varies}\}$$

ColabFold의 `is_training` 옵션으로 드롭아웃 활성화 및 여러 랜덤 시드 반복으로:
- **구조 다양성 샘플링** 가능
- 모델의 **불확실성 정량화** 실현
- 메타모르픽 단백질의 **대체 구조 예측** 가능 (AF-cluster 방법)[9]

### 5. 한계

#### 5.1 내재적 한계

1. **MSA 의존성**: ColabFold도 여전히 MSA 깊이에 강하게 의존
   - 단독 서열 예측 성능 현저히 저하
   
2. **복잡한 인터페이스**: 단백질-핵산 상호작용 정확도 여전히 제한적[6]

3. **동역학 정보 부재**: 정적 구조만 예측 가능, 동적 앙상블 정보 제한적

4. **멤브레인 단백질**: 막 삽입 부위 정확도 낮음 (진핵생물에서 특히 문제)

#### 5.2 실용적 한계

1. **템플릿 정보 제외**: 벤치마크에서 템플릿 미사용으로 인한 성능 저하

2. **유연한 영역**: 고도로 유연한 루프 또는 disordered 영역 정확도 낮음

3. **계산 자원**: Google Colaboratory의 제한된 가용성 (간헐적 중단, 세션 타임아웃)

### 6. 앞으로의 연구에 미치는 영향

#### 6.1 이미 미친 영향

**1. 구조 생물학의 민주화**

- 2022년 발표 이후 **8,500회 이상의 인용**[10]
- 저자원 국가의 연구자들도 최고 수준의 구조 예측 수행 가능
- 플라즈모디움(말라리아 기생충)부터 인간 핵공 복합체까지 **다양한 생물학적 문제 해결**[1]

**2. 단백질 엔지니어링 가속화**

ColabFold-DiffDock 프레임워크에서 **단백질-리간드 결합 친화성 예측** 성능 향상 (32% 개선 DAVIS 데이터셋)[11]

**3. 메타게노믹 단백질 구조 대규모 예측**

ESMFold와 결합 시 **6억 1,700만개 메타게놈 단백질** 구조 예측 가능 (ESM Metagenomic Atlas)[12]

#### 6.2 향후 연구 방향

**1. 멀티모달 예측의 표준화**

AlphaFold3의 성공으로 **단일 통합 프레임워크** 내에서:
- 단백질-단백질 상호작용
- 단백질-핵산 상호작용
- 화학 리간드 결합
- 동역학 정보 포함

의 **완전 통합 예측**이 미래 표준[6]

**2. 구조 기반 약물 발견 가속화**

디퓨전 모델 기반의 **역폴딩(inverse folding)** 기술과의 결합:

$$\text{Design Process}: \text{Target Structure} \rightarrow^{\text{inverse}} \text{Sequences} \rightarrow^{\text{forward}} \text{Validation}$$

이를 통해 약물 개발 시간 **10배 이상 단축** 가능[13]

**3. 동적 구조 앙상블 예측**

**AF-cluster 방법**과 같이 **대체 단백질 구조(metamorphic proteins)** 예측:[9]
- 메타모르픽 단백질의 여러 기능적 상태 동시 포착
- 단백질 역학의 **근본적 이해 제고**

**4. 언어 모델 기반 접근의 발전**

- **ESM3** 같은 차세대 단백질 언어 모델 통합
- MSA 완전 제거로 **고아/설계 단백질 성능 획기적 개선**
- 이론적으로는 **수백 배의 계산 효율** 달성 가능

**5. 분산 컴퓨팅과의 결합**

ColabFold의 배치 처리 능력과 분산 시스템 결합으로:
- 전체 유기체 단백질체 **병렬 예측 표준화**
- **AlphaFoldDB 같은 자원**을 조직 규모로 구축 가능

### 7. 향후 연구 시 고려할 점

#### 7.1 방법론적 고려사항

**1. 평가 벤치마크 선택**

- CASP 데이터만이 아닌 **매년 새로운 데이터셋** 사용
- 자유 모델링 타겟과 템플릿 기반 타겟의 **차별화된 평가**
- 유연한 영역의 정확도를 별도 측정

**2. MSA 전략의 재검토**

현재 ColabFoldDB가 **진핵생물 프로테인에 최적화**되어 있으므로:
- 원핵생물, 고세균, 바이러스 특화 데이터베이스 구축 필요
- 자극적 환경(hot spring, 심해) 적응 단백질의 MSA 품질 향상

**3. 아키텍처 선택의 명확화**

- **언어 모델 기반** (ESMFold류) vs **구조 기반** (AlphaFold류) 선택 기준
- 각 방법의 계산 비용-정확도 트레이드오프 정량화

#### 7.2 적용 시 주의사항

**1. 신뢰도 메트릭의 해석**

- pLDDT > 85 ≠ **100% 신뢰도**
- 특히 인터페이스 영역에서 **PAE(Predicted Aligned Error) 동시 검토 필수**
- 유연한 루프: pLDDT가 낮더라도 동적으로 중요할 수 있음

**2. 실험적 검증의 계획**

- ColabFold 예측 구조의 **변형 가설 검증** 필수
- SAXS, NMR 등 동적 실험과의 **통합 분석 권장**

**3. 복합체 예측의 한계 인식**

- AlphaFold3도 여전히 복잡한 멀티-체인 시스템에서 **10-15% 에러율**[14]
- 특히 약한 상호작용이나 transient 복합체에서 신뢰도 낮음

#### 7.3 계산 자원 최적화

**1. Google Colab 의존성 감소**

- LocalColabFold 사용으로 **자체 하드웨어 통제 가능**
- 단 M1/M2 Mac에서는 GPU 성능 제한 (약 10배 느림)

**2. 배치 처리 최적화**

$$\text{Efficiency} = \frac{\text{structures/hour}}{\text{GPU cost/hour}}$$

- 조기 종료 (pLDDT ≥ 85) 설정으로 **약 75% 처리 시간 단축** 가능

**3. 데이터베이스 업데이트 전략**

ColabFoldDB는 지속적 업데이트 필요:
- 신규 메타게놈 데이터 통합 주기 6개월 권장
- 중복성 제거 알고리즘의 정기적 최적화

### 결론

ColabFold는 단순한 성능 최적화를 넘어 **단백질 구조 예측의 민주화**를 실현한 획기적 도구입니다. 40-60배의 속도 향상을 통해 고급 연구를 저자원 환경에서 가능하게 함으로써, 구조 생물학 분야에서 **진정한 패러다임 전환**을 일으켰습니다.

최근의 AlphaFold3, ESMFold, 디퓨전 기반 생성 모델 등의 등장은 ColabFold가 열어둔 **개방형 구조 예측의 경로를 더욱 확대**하고 있습니다. 향후 연구는 다음 세 가지에 집중해야 합니다:

1. **동역학 정보 통합**: 정적 구조에서 동적 앙상블로의 진화
2. **복합체 예측의 정확화**: 특히 약한 상호작용과 transient 복합체
3. **일반화 성능 극대화**: 언어 모델과 구조 기반 방법의 하이브리드 활용

이러한 방향의 발전이 이루어진다면, 구조 생물학은 실시간 약물 설계, 맞춤형 단백질 엔지니어링, 합성 생물학의 완전 자동화 시대로 진입할 것입니다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ebcf6819-bd92-45ad-abbb-5b27e855530f/s41592-022-01488-1.pdf)
[2](https://pmc.ncbi.nlm.nih.gov/articles/PMC9184281/)
[3](http://arxiv.org/pdf/2405.20313.pdf)
[4](https://pubmed.ncbi.nlm.nih.gov/37321965/)
[5](https://www.nature.com/articles/s41467-024-45051-2)
[6](https://www.nature.com/articles/s41586-024-07487-w)
[7](https://pmc.ncbi.nlm.nih.gov/articles/PMC11168924/)
[8](https://academic.oup.com/bioinformatics/article/41/8/btaf432/8220020)
[9](https://pmc.ncbi.nlm.nih.gov/articles/PMC10705582/)
[10](https://www.nature.com/articles/s41592-022-01488-1)
[11](https://pmc.ncbi.nlm.nih.gov/articles/PMC11977223/)
[12](https://pubs.acs.org/doi/10.1021/acs.jctc.4c01585)
[13](http://arxiv.org/pdf/2209.12643v2.pdf)
[14](https://pmc.ncbi.nlm.nih.gov/articles/PMC11351016/)
[15](http://arxiv.org/pdf/2209.15611.pdf)
[16](https://pmc.ncbi.nlm.nih.gov/articles/PMC11236705/)
[17](https://pmc.ncbi.nlm.nih.gov/articles/PMC10769378/)
[18](https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btab881/42377860/btab881.pdf)
[19](https://www.sciencedirect.com/science/article/abs/pii/S0010482525001921)
[20](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.3c01324)
[21](https://www.nature.com/articles/s42256-023-00721-6)
[22](https://arxiv.org/html/2509.18480v2)
[23](https://academic.oup.com/bioinformatics/article/38/22/5007/6709341)
[24](https://www.biorxiv.org/content/10.1101/2025.09.28.679101v1.full-text)
[25](https://academic.oup.com/nar/advance-article-pdf/doi/10.1093/nar/gkad1011/52777135/gkad1011.pdf)
[26](https://pmc.ncbi.nlm.nih.gov/articles/PMC11634763/)
[27](https://arxiv.org/pdf/2406.03979v1.pdf)
[28](https://pmc.ncbi.nlm.nih.gov/articles/PMC10767828/)
[29](http://arxiv.org/pdf/2408.16975.pdf)
[30](http://arxiv.org/pdf/2406.03979.pdf)
[31](https://process-mining.tistory.com/222)
[32](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00976)
[33](https://pmc.ncbi.nlm.nih.gov/articles/PMC11948316/)
[34](https://pmc.ncbi.nlm.nih.gov/articles/PMC12550075/)
[35](https://www.sciencedirect.com/science/article/pii/S2001037025004180)
