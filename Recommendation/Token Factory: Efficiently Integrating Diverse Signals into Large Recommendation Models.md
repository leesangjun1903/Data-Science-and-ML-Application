# Token Factory: Efficiently Integrating Diverse Signals into Large Recommendation Models

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

"Token Factory"는 Large Recommendation Models(LRMs)에서 기존 전통적 신호(dense, sparse, sequence features)를 **"soft token"** 으로 변환하여 Transformer 기반 추천 모델에 효율적으로 통합하는 프레임워크입니다. 기존의 텍스트화(textualization) 방식이 야기하는 **프롬프트 길이 폭발, 메모리 오버헤드, 높은 계산 비용** 문제를 해결합니다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **전통적 신호의 중요성 입증** | Dense/Sparse features가 LRM 성능에 핵심적임을 실험으로 증명 |
| **Token Factory 아키텍처 제안** | 이종(heterogeneous) 특성을 soft token으로 변환하는 새로운 모달리티 도입 |
| **오프라인/온라인 검증** | 랭킹 및 생성적 검색 태스크에서 동등하거나 우수한 성능 달성 + 200% 학습 속도 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**기존 방식의 한계:**

LRM(PLUM 등)은 전통적 신호를 텍스트로 변환(textualize)하여 처리합니다. 예를 들어, 영상 추천에서 하나의 시청 기록 아이템이 다음과 같이 표현됩니다:

```
A100 B100 C100 D100 E100 F100 G100 H100 5.00% 50.00s 2.5h | ...
region PH | user 32 years female | device small ANDROID | video baby shark...
```

- SID 8토큰 + dense feature 텍스트 토큰 = **아이템당 12토큰**
- 200개 시청 기록 → 총 1,536토큰의 긴 프롬프트
- 메모리 및 계산 비용 폭발적 증가
- 오래된 시청 기록은 truncation으로 손실

### 2.2 제안 방법 및 수식

#### Token Maker 핵심 수식

**Step 1: 입력 특성 정의**

$$F_{input} = [f_1; f_2; \ldots; f_n]$$

여기서 $f_i$는 Token Maker에 입력되는 $i$번째 원시(raw) 입력 특성입니다.

**Step 2: 특성 변환 및 연결(concatenation)**

$$E_{input} = \text{Concat}(t_1(f_1), t_2(f_2), \ldots, t_n(f_n))$$

여기서 $t_i$는 특성 $f_i$에 적용되는 변환 함수(정규화, 임베딩 룩업 등)입니다.

**Step 3: Soft Token 생성**

$$T_{output} = G(E_{input})$$

여기서:
- $T_{output}$: $N$개의 출력 soft token, 각각 $d_{model}$ 차원
- $G$: $E_{input}$을 $N \times d_{model}$ 임베딩 벡터로 변환하는 미분 가능한 함수(MLP 또는 복잡한 신경망)
- $G$는 LRM과 **end-to-end로 공동 학습(co-trained)**

**MLP 압축 수식:**

입력 시퀀스 배열 형태: $[\text{batch size}, N, \text{token dim}]$

MLP 레이어 $[N, \ldots, M]$을 시퀀스 차원에 적용 시:

$$\text{압축 비율} = \frac{M}{N}$$

**Attention Pooling 압축:**

매 $K$개 아이템마다 경량 Transformer + Attention Pooling 적용:

$$\text{압축 비율} = \frac{1}{K}$$

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────┐
│              Large Recommendation Model          │
│                (Transformer 기반)                │
└────────────────────┬────────────────────────────┘
                     │ Input Prompt
     ┌───────────────┼───────────────┐
     │               │               │
┌────▼────┐    ┌─────▼─────┐   ┌────▼────────┐
│  WH     │    │  Query    │   │ Candidate   │
│  Token  │    │  Feature  │   │  Feature    │
│  Maker  │    │  Token    │   │  Token      │
│         │    │  Maker    │   │  Maker      │
└────┬────┘    └─────┬─────┘   └─────┬───────┘
     │               │               │
     ▼               ▼               ▼
 WH Tokens     Query Tokens    Candidate Tokens
(시청기록→      (유저특성→        (영상특성→
 soft token)    soft token)      soft token)
     │               │               │
     └───────────────┼───────────────┘
                     ▼
        [watch_history]<emb>...<emb>[end_watch_history]
        <emb>...<emb> video baby shark <emb>...<emb>
```

**세 가지 Token Maker:**

| Token Maker | 입력 | 역할 |
|-------------|------|------|
| **WH Token Maker** | 시청 기록 시퀀스 (SID, 채널명, 시청 시간 등) | 시청 이력 → soft token 시퀀스 |
| **Query Token Maker** | 유저 레벨 특성 (dense, sparse) | 쿼리 특성 → soft token |
| **Candidate Token Maker** | 영상 레벨 특성 | 후보 아이템 특성 → soft token |

**핵심 설계 특징:**
- **Prefix Caching**: Query/User 레벨 soft token은 사전 계산 후 캐싱 가능 → 추론 지연 감소
- **Deterministic Length**: 출력 토큰 수 $N$을 미리 고정 → 프롬프트 길이 결정론적 제어
- **텍스트 토큰과의 혼용**: 영상 제목 등 텍스트 토큰과 soft token을 인터리빙(interleaving)

### 2.4 성능 향상

#### 랭킹 태스크 (CTR 예측)

**실험 설정:**
- 모델: 110M MoE Gemini encoder 기반 PLUM
- Baseline: 1,536 토큰 (아이템당 12토큰), 200개 시청 기록
- Treatment: 480 토큰 (아이템당 1 soft token), 200개 시청 기록

| 지표 | 결과 |
|------|------|
| ROC AUC | 1.5M 스텝 이후 baseline과 동등 수준 도달 |
| 학습 속도 | **약 200% 향상** (프롬프트 길이가 baseline의 30% 수준) |
| 배치 크기 증가 시 | Baseline 대비 **초기부터 우수한 AUC** 달성 |

#### 생성적 검색 태스크 (다음 영상 SID 예측)

**실험 설정:**
- 모델: 210M MoE Gemini decoder 기반 PLUM
- Baseline: 768 토큰 (아이템당 5토큰)
- Treatment: 256 토큰 (아이템당 1 soft token), 200개 시청 기록

| 지표 | 향상 |
|------|------|
| Offline Recall@10 | **+2.0%** |
| Unique Impressions (온라인) | **+16.8%** |
| Unique Impressions (1일 신규 영상) | **+67.1%** |
| Satisfied Watchers | **+0.04%** |
| Satisfied Watch Time | **+0.05%** |

#### Ablation Study 결과 (AUC 비교)

| 설정 | 설명 | 상대적 성능 |
|------|------|------------|
| Baseline (Token Factory) | Soft token + 전체 특성 | **최고** |
| WH_SID | Textual SID + 전체 특성 | Baseline과 유사 |
| NO_FEAT | Soft token, dense/sparse 제거 | 하락 |
| WH_SID_NO_FEAT | Textual SID + 특성 제거 | 더 하락 |
| NO_FEAT_STRICT | Soft token, WH dense/sparse도 제거 | 최저 |

**더 긴 시청 기록 (500개):** 압축 비율 10% 적용 시 AUC **+0.08%** 향상

### 2.5 한계

논문에서 명시적으로 언급된 한계 및 유추 가능한 한계:

1. **초기 학습 지연**: Token Maker의 새롭게 초기화된 파라미터(임베딩 테이블 등)가 LLM 시맨틱 공간에 정렬되는 데 시간이 필요 → 초기 AUC가 baseline보다 낮음 (1.5M 스텝 이후 수렴)

2. **도메인 특화성**: YouTube 영상 추천 시스템에서 검증되었으며, 타 도메인(전자상거래, 음악 등)으로의 일반화 여부는 별도 검증 필요

3. **$N$ 값의 경험적 결정**: 출력 soft token 수 $N$이 경험적으로 결정되며, 자동 최적화 방법론이 부재

4. **해석 가능성 부재**: Soft token은 인간이 읽을 수 없는 연속 벡터로, 모델 디버깅이 어려움

5. **다중 플랫폼 검증 부재**: 단일 추천 플랫폼(YouTube Homepage)에서만 온라인 실험

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 촉진하는 메커니즘

#### (1) 더 풍부한 특성 통합으로 인한 일반화 향상

Token Factory는 기존 텍스트화 방식에서 **context window 제한으로 인해 잘렸던(truncated) 특성들을 더 많이 포함**할 수 있습니다:

$$\text{기존 방식: } \text{프롬프트} = \text{truncate}\left(\sum_{i=1}^{H} 12 \cdot \text{tokens}_i, \text{budget}=1536\right)$$

```math
\text{Token Factory: } \text{프롬프트} = \sum_{i=1}^{H} 1 \cdot \text{soft token}_i, \quad H \leq 480
```

즉, 동일한 context window에서 더 많은 시청 기록(예: 200 → 500)을 처리 가능하며, 이는 **user preference의 장기 패턴(long-term pattern) 학습**을 가능하게 합니다.

#### (2) 신규 콘텐츠(Cold-Start) 일반화

온라인 실험에서 **1일 신규 영상에 대한 Unique Impressions이 +67.1%** 증가했습니다. 이는 Token Factory가 SID(Semantic ID) 기반 정보 외에도 **영상의 메타데이터(채널, 제목) 및 컨텍스트 특성을 더 풍부하게 인코딩**하기 때문입니다:

$$E_{\text{item}} = \text{Concat}(t_{\text{SID}}(f_{\text{SID}}), t_{\text{channel}}(f_{\text{channel}}), t_{\text{dense}}(f_{\text{watch time}}), \ldots)$$

SID가 없거나 희소한 신규 아이템에 대해 나머지 특성들이 보완적 역할을 수행하여 **cold-start 일반화**가 향상됩니다.

#### (3) Attention 분산으로 인한 일반화

Appendix A의 attention 시각화 분석에서:
- **Textual SID 모델**: 소수의 토큰에 attention 집중 (sparse, redundant)
- **Soft Token 모델**: attention이 시퀀스 전반에 걸쳐 분산 (dense, distributed)

$$\text{Attention Entropy}_{\text{soft}} > \text{Attention Entropy}_{\text{textual SID}}$$

Attention이 고르게 분산되면 다양한 head와 layer가 **상호보완적 신호를 병렬로 학습**하게 되어, 특정 패턴에 과적합(overfitting)하지 않고 더 넓은 패턴을 포착합니다.

#### (4) End-to-End 학습에 의한 표현 정렬(Representation Alignment)

$$G: E_{input} \rightarrow T_{output} \in \mathbb{R}^{N \times d_{model}}$$

$G$가 LRM과 함께 end-to-end로 학습되므로, soft token은 LLM의 **사전 학습된 언어 이해 능력과 정렬**됩니다. 이는 텍스트 이해로 학습된 일반 표현 공간을 활용하여 추천 태스크로의 **전이(transfer) 효율**을 높입니다.

#### (5) 모달리티 융합을 통한 일반화

Token Factory는 dense, sparse, sequence 특성을 **통합된 임베딩 공간으로 융합**합니다. 이는 단일 모달리티에 의존하는 과적합을 방지하고, 특정 특성이 노이즈이거나 결측일 때 다른 특성이 보완할 수 있는 **강건성(robustness)**을 제공합니다.

### 3.2 일반화 한계 및 주의사항

| 요인 | 영향 |
|------|------|
| **도메인 의존적 특성 설계** | Token Maker 구성이 영상 추천에 최적화되어 있어 타 도메인 적용 시 재설계 필요 |
| **Cold-Start for Token Makers** | Token Maker 파라미터 자체는 처음부터 학습 → 데이터 부족 시 일반화 어려움 |
| **압축으로 인한 정보 손실** | $N/M$ 압축 비율이 높을수록 미세한 패턴 손실 가능 |

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (1) Soft Token 패러다임의 확산

Token Factory는 추천 시스템에서 **"feature-as-token" 패러다임**을 확립합니다. 텍스트, 이미지, 오디오 외에 **추천 특성(recommendation features)을 하나의 독립적 모달리티**로 처리하는 연구 방향을 제시합니다.

#### (2) 멀티모달 LRM 연구 촉진

Soft token과 텍스트 토큰의 혼용(interleaving) 아이디어는 이미지, 오디오, 그래프 구조 등 **다양한 비텍스트 정보를 LRM에 통합하는 연구**로 확장될 수 있습니다.

#### (3) 추천 시스템의 효율화 연구

프롬프트 압축을 통해 동일한 context window에서 더 많은 정보를 처리하는 아이디어는 **long-context recommendation** 연구에 영향을 줄 것입니다. 특히 사용자의 수년간 상호작용 기록을 처리하는 lifelong recommendation 연구와 연계됩니다.

#### (4) 산업-학계 연구 격차 해소

YouTube 규모의 프로덕션 환경에서 검증된 결과는 학계 연구자들에게 **실제 배포 환경에서의 요구사항(latency, memory, throughput)**을 명확히 이해하는 데 기여합니다.

### 4.2 향후 연구 시 고려할 점

#### (1) 최적 $N$ 값의 자동 결정

현재 논문에서는 $N$을 경험적으로 결정합니다. 향후 연구에서는:

$$N^* = \arg\min_{N} \mathcal{L}(\theta, N) + \lambda \cdot \text{Complexity}(N)$$

와 같이 **Neural Architecture Search(NAS)** 또는 **학습 기반 토큰 할당(learned token budget allocation)**을 통해 최적 $N$을 자동으로 결정하는 연구가 필요합니다.

#### (2) 도메인 적응(Domain Adaptation) 연구

전자상거래, 뉴스, 음악 등 다양한 도메인에서의 Token Factory 적용 가능성 검증이 필요합니다. 특히 **cross-domain recommendation**에서 소스 도메인에서 학습된 Token Maker를 타겟 도메인으로 전이하는 방법론 연구가 중요합니다.

#### (3) Token Maker의 초기화 전략

초기 학습 지연 문제를 해결하기 위한 **사전 학습된 임베딩을 활용한 Token Maker 초기화** 연구가 필요합니다:

$$G_{\text{init}} \leftarrow \text{pretrained LEM weights}$$

Large Embedding Models(LEM)의 가중치로 Token Maker를 초기화하면 수렴 속도를 높일 수 있습니다.

#### (4) 프라이버시 보존 학습(Federated Learning)과의 결합

Soft token은 원시 특성을 직접 노출하지 않고 임베딩으로 변환하므로, **연합 학습(Federated Learning)** 환경에서의 프라이버시 보존 추천과의 결합 연구가 의미있을 것입니다.

#### (5) 동적 시퀀스 압축 전략

현재 MLP 압축과 Attention Pooling 두 가지 고정 전략만 제시됩니다. 사용자/아이템의 특성에 따라 **동적으로 압축 비율을 조정하는 adaptive compression** 연구가 필요합니다:

$$K_{\text{dynamic}} = f(\text{user activity level}, \text{sequence diversity})$$

#### (6) 해석 가능성(Interpretability) 연구

Soft token은 연속 벡터로 해석이 불가능합니다. **Concept Bottleneck Models** 또는 **Sparse Autoencoder**를 활용하여 soft token의 의미를 해석하는 연구가 중요합니다.

#### (7) 다중 태스크 학습(Multi-task Learning)과의 결합

랭킹과 검색을 각각 별도로 실험했지만, 두 태스크를 동시에 최적화하는 **통합 Token Factory 학습 프레임워크** 연구가 필요합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 핵심 방법 | Token Factory와의 관계 | 차이점 |
|------|------|-----------|------------------------|--------|
| **HSTU** (Zhai et al., ICML 2024) | 2024 | Hierarchical Sequential Transducer; 아키텍처 최적화로 스케일링 | 유사한 LRM 기반 추천 목표 | 특성 통합보다 아키텍처 효율화에 집중; 전통적 신호 텍스트화 여전히 사용 |
| **PLUM** (He et al., arXiv 2025) | 2025 | Semantic ID + 사전 학습 LM 활용 생성적 추천 | Token Factory가 PLUM을 기반 모델로 사용 | Soft token 부재; 텍스트화 방식으로 프롬프트 길이 문제 존재 |
| **TokenRec** (Qu et al., IEEE TKDE 2025) | 2025 | Masked 유저/아이템 표현을 양자화하여 이산 토큰 생성 | ID 토큰화 전략 비교 대상 | 이산 토큰(discrete)으로 연속 특성 손실; soft token 미사용 |
| **LONGER** (Chai et al., RecSys 2025) | 2025 | Global token + $K$개 아이템 병합으로 시퀀스 압축 | Attention Pooling 압축 아이디어의 영감 | 시퀀스 아이템 수 감소에만 집중; 아이템별 이종 특성 융합 미지원 |
| **SEATER** (Si et al., SIGIR-AP 2024) | 2024 | 의미론적 트리 구조 식별자 + 대조 학습으로 아이템 표현 | 아이템 ID 표현 방식 비교 | 보조 특성(dense/sparse) 통합 미지원 |
| **HyMiRec** (Zhou et al., arXiv 2025) | 2025 | 아이템(제목+콘텐츠)을 SID-like 코드(3개)로 압축 | 유사한 토큰 압축 목표 | 콘텐츠 기반 압축에 한정; 유저-아이템 상호작용 특성 미활용 |
| **GenRec** (Zou et al., SIRIG 2026) | 2026 | 선형 투영으로 긴 행동 시퀀스를 다중 토큰으로 인코딩 | 유사한 시퀀스 압축 목표 | 아이템별 개별 특성(dense/sparse) 통합 미지원 |
| **TokenMixer-Large** (Jiang et al., arXiv 2026) | 2026 | 대규모 랭킹 모델 스케일링 | Token Maker 아이디어 활용 명시 (논문에서 [7] 인용) | 랭킹 특화; 검색 태스크 미적용 |
| **RankMixer** (Zhu et al., CIKM 2025) | 2025 | 산업용 랭킹 모델 스케일링 | Token Maker 아이디어 활용 명시 (논문에서 [16] 인용) | 랭킹 특화 |
| **OneRec** (Zhou et al., arXiv 2025) | 2025 | 계층적 K-means 클러스터링으로 시퀀스 압축 | 유사한 시퀀스 압축 목표 | 클러스터링 기반 이산화 → 특성 정보 손실 가능 |
| **OnePiece** (Dai et al., arXiv 2025) | 2025 | Context Engineering + Reasoning for Cascade Ranking | LRM 기반 추천의 발전 방향 공유 | Token Factory의 효율화 접근과 상호보완적 |

### 비교 분석 종합

```
특성 통합 풍부도
        ↑
        │  ● Token Factory (본 논문)
        │     (Dense+Sparse+Seq 모두 통합)
        │
        │        ● HyMiRec
        │           (콘텐츠 기반)
        │  ● PLUM
        │     (SID+텍스트)
        │        ● LONGER
        │           (시퀀스 압축)
        │  ● HSTU
        │     (아키텍처 효율화)
        └──────────────────────────→
              프롬프트 효율성
```

**Token Factory의 차별점:**
1. **Dense + Sparse + Sequence** 세 가지 특성 유형을 **모두** soft token으로 통합
2. **프롬프트 길이를 결정론적으로 제어** (다른 방법들은 특성 수에 따라 가변)
3. **Prefix Caching** 지원으로 서빙 효율 극대화
4. **Production-scale 온라인 실험** 결과 제시 (많은 논문이 오프라인 평가에 그침)

---

## 참고 자료

**본 답변에서 직접 참조한 논문:**

1. **Chen, X., Wang, S.-C., et al. (2026). "Token Factory: Efficiently Integrating Diverse Signals into Large Recommendation Models."** arXiv preprint arXiv:2606.19635v2. *(제공된 PDF 원문)*

**논문 내 인용 참고문헌 (핵심):**

2. He, R., et al. (2025). "PLUM: Adapting Pre-trained Language Models for Industrial-Scale Generative Recommendations." arXiv:2510.07784

3. Zhai, J., et al. (2024). "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations." ICML 2024.

4. Singh, A., et al. (2024). "Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations." RecSys 2024.

5. Chai, Z., et al. (2025). "LONGER: Scaling up Long Sequence Modeling in Industrial Recommenders." RecSys 2025.

6. Qu, H., et al. (2025). "TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendations." IEEE TKDE 2025.

7. Si, Z., et al. (2024). "SEATER: Generative Retrieval with Semantic Tree-Structured Identifiers and Contrastive Learning." SIGIR-AP 2024.

8. Zhou, J., et al. (2025). "HyMiRec: A Hybrid Multi-interest Learning Framework for LLM-based Sequential Recommendation." arXiv:2510.13738

9. Jiang, Y., et al. (2026). "TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders." arXiv:2602.06563

10. Zhu, J., et al. (2025). "RankMixer: Scaling Up Ranking Models in Industrial Recommenders." CIKM 2025.

11. Zhou, G., et al. (2025). "OneRec Technical Report." arXiv:2506.13695

12. Zou, Y., et al. (2026). "GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation." SIRIG 2026.

> **⚠️ 정확도 관련 고지:** 본 답변은 제공된 PDF 원문(arXiv:2606.19635v2)을 1차 출처로 하며, 비교 분석에 사용된 관련 연구 정보는 논문 내 참고문헌 목록을 기반으로 작성되었습니다. 일부 비교 연구의 세부 내용은 해당 논문을 직접 열람하지 않고 인용 정보만으로 기술하였으므로, 정밀한 비교 분석을 위해서는 각 논문 원문 확인을 권장합니다.
