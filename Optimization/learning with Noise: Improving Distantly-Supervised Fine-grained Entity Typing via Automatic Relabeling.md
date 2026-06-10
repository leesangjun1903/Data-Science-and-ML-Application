# Learning with Noise: Improving Distantly-Supervised Fine-grained Entity Typing via Automatic Relabeling

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
본 논문은 **원거리 감독(Distant Supervision)** 방식으로 구축된 Fine-grained Entity Typing(FET) 학습 데이터의 **노이즈 레이블 문제**를 해결하기 위해, 별도의 외부 감독 없이 훈련 과정에서 자동으로 레이블을 재추정하는 **확률적 자동 재레이블링(Probabilistic Automatic Relabeling, AR)** 방법을 제안한다.

### 주요 기여 (3가지)
| 기여 | 내용 |
|------|------|
| **통합 처리 프레임워크** | 기존의 'clean/noisy' 분리 처리 방식을 탈피, 모든 샘플을 균일하게 처리 |
| **Pseudo-truth 분포 추정** | 각 샘플의 pseudo-truth 레이블 분포 $\tilde{p}$를 훈련 가능한 파라미터로 공동 최적화 |
| **전제 조건 불필요** | 인간이 레이블링한 클린 데이터나 "정답이 반드시 원거리 감독 레이블에 존재한다"는 강한 가정 불필요 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**원거리 감독의 노이즈 문제:**
- 동일한 엔티티 멘션(예: "Amazon")이 문맥에 무관하게 동일한 타입 집합으로 레이블링됨
- 기존 연구의 두 가지 한계:
  1. **한계 1:** 'clean(단일 레이블)'과 'noisy(다중 레이블)' 샘플을 분리 처리 → 단일 레이블도 오류 가능성 존재
  2. **한계 2:** "원거리 감독 레이블 집합에 반드시 정답이 존재한다"는 과도하게 강한 가정에 의존

**기존 방법론의 문제점:**
```
단일 레이블 샘플 → "clean"으로 가정 (실제로는 false-positive 존재)
다중 레이블 샘플 → "noisy"로 분리 처리 (유용한 정보 손실 가능)
외부 클린 데이터 의존 → 실용성 저하
```

---

### 2.2 제안 방법 및 수식

#### 모델 전체 목적 함수

기존 FET 최적화 문제:

$$\theta^* = \arg\min_{\theta} \mathcal{L}(m, c, y;\, \theta) \tag{3}$$

자동 재레이블링 모듈 적용 시 최적화 문제:

```math
\theta^*, \tilde{p}^* = \arg\min_{\theta, \tilde{p}} \mathcal{L}(m, c, y;\, \theta, \tilde{p})
```

---

#### (1) 기본 Cross-Entropy 손실

$$p(y_i | m_i, c_i) = \text{softmax}(W r_i + b) \tag{1}$$

$$\mathcal{L}_{ce}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} y_i \log\left(p(y_i | m_i, c_i;\, \theta)\right) \tag{2}$$

---

#### (2) KL-Divergence 손실 (핵심 손실)

각 샘플에 연속 레이블 분포 $\tilde{p}\_i = \{\tilde{p}\_{ij} : \tilde{p}\_{ij} \in [0,1],\, \sum_j \tilde{p}_{ij} = 1\}$ 를 부여하고, 예측 분포 $p$와의 KL 발산을 최소화:

$$\mathcal{L}_{kl} = \frac{1}{N} \sum_{i=1}^{N} \text{KL}(\tilde{p}_i \| p(y_i | m_i, c_i;\, \theta)) = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{|T|} \tilde{p}_{ij} \log\left(\frac{\tilde{p}_{ij}}{p_j(y_i | m_i, c_i;\, \theta)}\right) \tag{5}$$

---

#### (3) 원거리 레이블 제약 손실 (Distant Label Constraint)

$\tilde{p}$가 원본 노이즈 레이블 $y$에서 너무 멀어지는 것을 방지:

$$\mathcal{L}_d = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{|T|} \tilde{p}_{ij} \log y_{ij} \tag{6}$$

---

#### (4) 분포 첨예화 제약 손실 (Distribution Sharpen Constraint)

예측 분포가 0 또는 1에 가까워지도록 엔트로피를 최소화 (FET의 단일 레이블 가정 반영):

$$\mathcal{L}_s = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{|T|} p_j(y_i | m_i, c_i;\, \theta) \log p_j(y_i | m_i, c_i;\, \theta) \tag{7}$$

---

#### (5) 최종 통합 손실 함수

$$\mathcal{L}_{ar} = \beta \cdot \mathcal{L}_{ce} + \gamma \cdot \mathcal{L}_{kl} + \omega \cdot \mathcal{L}_d + \delta \cdot \mathcal{L}_s \tag{8}$$

여기서 $\beta, \gamma, \omega, \delta$는 하이퍼파라미터.

---

### 2.3 모델 구조 (NFETC-AR)

```
┌─────────────────────────────────────────────────────────┐
│                    Feature Encoder                      │
│  ┌──────────────────┐    ┌──────────────────────────┐   │
│  │ Mention Encoder  │    │   Context Encoder        │   │
│  │ - AVG Encoder    │    │   - Bi-LSTM              │   │
│  │ - LSTM Encoder   │    │   - Word-level Attention │   │
│  └──────────────────┘    └──────────────────────────┘   │
│           r_mi                      r_ci                │
│                  r_i = [r_mi, r_ci]                     │
│                       FC → Softmax → p(m,c;θ)           │
└─────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────┐
│              Automatic Relabeling Module                 │
│  p(m,c;θ) ←── KL(p̃ ∥ p) ───→ p̃ (Trainable Param)    │
│                    ↑                                    │
│         L_d (distant label constraint)                  │
│         L_s (sharpen constraint)                        │
│                    ↓                                    │
│           ŷ = argmax(p̃)  [Pseudo-Truth Label]          │
└─────────────────────────────────────────────────────────┘
```

**3단계 훈련 전략:**

| 단계 | 내용 | 손실 함수 |
|------|------|----------|
| Phase 1: 예비 훈련 | 원본 노이즈 레이블로 기본 모델 학습 (warm-up) | $\mathcal{L}_{ce}$ |
| Phase 2: 자동 재레이블링 | $\theta$와 $\tilde{p}$ 공동 최적화, 초기화: $\tilde{p} = \text{softmax}(y)$ | $\mathcal{L}_{ar}$ |
| Phase 3: 파인튜닝 | $\hat{y} = \arg\max(\tilde{p})$로 one-hot 변환 후 파인튜닝 | $\mathcal{L}_{ce}$ |

---

### 2.4 성능 향상

#### 전체 성능 비교 (Table 1 기반)

| 모델 | Wiki Strict Acc | OntoNotes Strict Acc | BBN Strict Acc |
|------|-----------------|---------------------|----------------|
| NFETC $_{hier}$ | 68.9 | 60.2 | 73.9 |
| NFETC-CLSC $_{hier}$ | - | 62.8 | 73.0 |
| **NFETC-AR $_{hier}$ ** | **70.1** | **64.0** | **74.9** |

- Wiki: 68.9 → 70.1 (+1.2)
- OntoNotes: 60.2 → 64.0 (+3.8)
- BBN: 73.9 → 74.9 (+1.0)

#### Ablation Study (OntoNotes, Strict Acc 기준)

| 설정 | Acc | Macro F1 | Micro F1 |
|------|-----|----------|----------|
| NFETC-AR $_{hier}$ (full) | **64.0** | **78.8** | **73.0** |
| w/o $\mathcal{L}_{kl}$ | 61.2 | 76.6 | 70.4 |
| w/o noisy label init | 55.0 | 67.1 | 60.3 |
| w/o $\mathcal{L}_{ce}$ | 61.1 | 76.1 | 69.9 |
| w/o AR | 60.2 | 76.4 | 70.2 |

**핵심 발견:** $\mathcal{L}_{kl}$ 제거 시 가장 큰 성능 하락, noisy label 초기화 제거 시 Acc 55.0으로 급락

---

### 2.5 한계점

1. **백본 의존성:** BiLSTM 기반 NFETC 인코더에 실험이 한정됨 (BERT 등 Transformer 기반 검증 미흡)
2. **하이퍼파라미터 민감성:** $\beta, \gamma, \omega, \delta, e_1, e_2, e_3$ 등 다수의 하이퍼파라미터 튜닝 필요
3. **확인 편향(Confirmation Bias) 위험:** KL 손실만 사용할 경우 모델 예측 오류가 $\tilde{p}$에 누적될 수 있음
4. **BBN 극단 노이즈 상황:** 클린 데이터 95% 제거 시 NFETC-CLSC 대비 성능 열위
5. **메모리 비용:** $\tilde{p} \in \mathbb{R}^{N \times |T|}$ 전체 학습 데이터에 대한 분포 파라미터 저장 필요
6. **단일 레이블 가정:** FET 태스크가 반드시 단일 정답을 갖는다는 가정이 항상 성립하지 않을 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 노이즈 강건성 실험 결과

논문의 Figure 4에서 클린 데이터를 75%~95% 제거했을 때의 성능을 비교:

- **OntoNotes:** 모든 제거 비율에서 NFETC-AR이 NFETC, NFETC-CLSC를 일관되게 상회
- **BBN:** 대부분의 경우 NFETC-CLSC와 동등하거나 우수 (95% 제거 시 NFETC-CLSC가 소폭 우위)

이는 NFETC-AR이 **실제 환경의 고노이즈 시나리오에서도 일반화 성능을 유지**함을 시사한다.

### 3.2 일반화를 높이는 메커니즘 분석

#### (a) 소프트 레이블 학습의 일반화 효과

표준 Cross-Entropy 손실은 one-hot 레이블을 직접 학습하므로 과적합(overfitting)에 취약:

$$\mathcal{L}_{ce}^{\text{noisy}} = -\frac{1}{N}\sum_{i=1}^N y_i^{\text{noisy}} \log p(y_i|m_i,c_i;\theta)$$

반면, pseudo-truth 분포 $\tilde{p}$는 **소프트 레이블(Soft Label) 효과**를 제공하여 레이블 스무딩과 유사한 정규화 역할을 수행:

$$\mathcal{L}_{kl} = \frac{1}{N}\sum_{i=1}^N \sum_{j=1}^{|T|} \tilde{p}_{ij} \log \frac{\tilde{p}_{ij}}{p_j(\cdot)}$$

이는 Knowledge Distillation의 소프트 타겟 효과와 유사하게 **일반화 성능을 향상**시킨다.

#### (b) $\mathcal{L}_s$ (첨예화 제약)의 역할

예측 분포의 엔트로피를 최소화함으로써 모델이 **명확한 결정 경계**를 학습하도록 유도:

$$\mathcal{L}_s = -\frac{1}{N}\sum_{i=1}^N \sum_{j=1}^{|T|} p_j \log p_j$$

이 정규화 항은 과도한 분포 평탄화를 방지하고, 실제 정답 레이블 분포에 가까운 sharp한 $\tilde{p}$를 학습하게 함.

#### (c) 3단계 훈련의 커리큘럼 학습 효과

Phase 1(노이즈 데이터 학습) → Phase 2(소프트 재레이블링) → Phase 3(하드 재레이블 파인튜닝)의 점진적 학습은 **커리큘럼 학습(Curriculum Learning)**과 유사한 효과를 제공하여 일반화에 기여한다.

#### (d) 백본 독립성

논문은 결론에서 "**our proposed method is independent of the backbone network**"를 명시하여, BERT, RoBERTa 등 더 강력한 인코더와 결합 시 추가적인 일반화 성능 향상이 기대됨.

#### (e) 재레이블 통계가 보여주는 일반화 근거

OntoNotes에서 27.52%의 훈련 샘플이 재레이블되었으며:
- Multi-to-one-in (정답이 원래 집합 내): 97.49%
- One-to-one-out (단일 레이블도 오수정 가능): 2.39%
- Multi-to-one-out (원래 집합 외로 수정): 0.12%

이는 모델이 **컨텍스트 정보를 활용하여 실제 의미론적으로 적절한 레이블을 복원**할 수 있음을 보여주며, 이러한 노이즈 정제 능력이 테스트 셋 일반화의 핵심 원동력임.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 영향

#### (a) NLP 노이즈 학습 패러다임 변화
기존의 'clean/noisy 분리 처리' 패러다임에서 **'통합 처리 + 자기 지도 재레이블링'** 패러다임으로의 전환을 제시함. 이는 레이블 노이즈가 존재하는 모든 NLP 태스크에 적용 가능한 일반 원리를 제공한다.

#### (b) 다른 NLP 태스크로의 확장 가능성
논문이 명시적으로 언급한 확장 방향:
- **관계 추출(Relation Extraction)**
- **어휘 기반 NER(Lexicon-based Named Entity Recognition)**

유사하게 적용 가능한 분야:
- 원거리 감독 텍스트 분류
- 약지도 감성 분석
- 노이즈가 많은 의료/법률 도메인 NLP

#### (c) 소프트 레이블 학습 연구 촉진
$\tilde{p}$를 훈련 가능한 파라미터로 취급하는 아이디어는 **레이블 분포 학습(Label Distribution Learning)** 및 **지식 증류(Knowledge Distillation)** 연구와의 교차점을 형성한다.

---

### 4.2 향후 연구 시 고려해야 할 점

#### (a) 대규모 사전학습 모델과의 통합
BERT, RoBERTa, LUKE 등 Transformer 기반 인코더와 AR 모듈의 결합 효과 검증이 필요. 특히 BERT의 맥락적 표현이 $\tilde{p}$ 추정 품질을 얼마나 향상시키는지 분석해야 함.

#### (b) 확인 편향 완화 전략
KL 손실이 모델의 잘못된 예측을 $\tilde{p}$에 전파하는 확인 편향 문제를 더 체계적으로 해결해야 함:
- 앙상블 모델을 통한 다양한 예측 집계
- 신뢰도 기반 가중치 조정
- Co-training 기반 접근법

#### (c) 타입 계층 구조의 명시적 활용
현재 AR 모듈은 타입 계층을 간접적으로만 반영함. 온톨로지 기반 계층 제약을 $\tilde{p}$ 추정에 직접 통합하면 성능 향상 기대 가능.

#### (d) 메모리 효율화
$\tilde{p} \in \mathbb{R}^{N \times |T|}$ 전체 저장의 메모리 비용 문제 → 온라인 갱신 방식이나 클러스터링 기반 공유 분포 접근법 탐색 필요.

#### (e) 동적 레이블 재레이블링
현재는 Phase 2 종료 후 $\hat{y}$를 고정하여 Phase 3에 사용. 훈련 중 동적으로 레이블을 갱신하는 **온라인 재레이블링** 전략의 효과 탐색 필요.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래 연구들은 본 논문 이후 FET/노이즈 레이블 관련 연구들로, 제가 직접 접근한 논문 PDF를 기반으로 하지 않고 제 학습 데이터(~2024년 초)에 기반한 정보입니다. 일부 세부 수치는 부정확할 수 있으므로 원문 확인을 권장합니다.

### 5.1 주요 후속 연구 흐름

#### (a) BERT 기반 FET 연구

**"Fine-grained Entity Typing with Hierarchical Inference" (Onoe et al., EMNLP 2021)**
- BERT 기반 인코더 사용
- 타입 계층 간 의존성을 명시적으로 모델링
- 본 논문 대비: 더 강력한 인코더 사용, 노이즈 처리 방식은 다름

**"An Empirical Study on Multiple Information Sources for Zero-Shot Fine-Grained Entity Typing" (Ding et al., EMNLP 2021)**
- Zero-shot 시나리오에서의 FET 연구
- 본 논문의 한계였던 새로운 타입에 대한 일반화 문제 접근

#### (b) 노이즈 레이블 학습의 발전

**"Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations" (Wei et al., ICLR 2022)**
- 실제 인간 주석의 노이즈 패턴 분석
- 인스턴스 의존적 노이즈 처리 강조
- 본 논문의 균일한 노이즈 가정에 대한 도전

**"Robust Training under Label Noise by Over-parameterization" (Liu and Guo, ICML 2022)**
- 과파라미터화를 통한 노이즈 강건성
- 본 논문의 $\tilde{p}$ 파라미터화 아이디어와 일부 연결됨

#### (c) 대비 학습 기반 FET

**"Contrastive Self-Supervised Learning for Graph Classification" 계열 연구**
- FET에서 대비 학습을 통한 타입 표현 개선
- 노이즈에 강건한 표현 학습

### 5.2 본 논문 vs. 최신 연구 비교표

| 항목 | NFETC-AR (본 논문, 2020) | 최신 BERT 기반 FET (2021~) | 노이즈 레이블 학습 최신 (2022~) |
|------|--------------------------|---------------------------|--------------------------------|
| **인코더** | BiLSTM | BERT/RoBERTa | 다양 |
| **노이즈 처리** | 확률적 재레이블링 | 외부 지식/대비 학습 | 인스턴스 의존적 |
| **외부 감독** | 불필요 | 일부 필요 | 일부 필요 |
| **계층 활용** | 간접적 | 명시적 | 태스크 의존 |
| **확장성** | 중간 | 높음 | 높음 |
| **노이즈 가정** | 컨텍스트 의존적 | 다양 | 인스턴스 의존적 |

### 5.3 본 논문의 한계가 이후 어떻게 극복되었는가

1. **BiLSTM → BERT 전환:** LUKE (Yamada et al., 2020), BERT-FET 계열이 인코더 성능을 크게 향상
2. **단순 KL 기반 재레이블 → 더 정교한 노이즈 모델링:** 인스턴스 의존적 노이즈 행렬, 메타 학습 기반 노이즈 처리 등이 등장
3. **단일 도메인 → 다도메인/제로샷 일반화:** 후속 연구들이 도메인 전이 및 새로운 타입 일반화 문제를 다룸

---

## 참고 자료

- **본 논문:** Haoyu Zhang et al., "Learning with Noise: Improving Distantly-Supervised Fine-grained Entity Typing via Automatic Relabeling," *Proceedings of IJCAI-2020*, pp. 3808–3815. (제공된 PDF)
- Xu and Barbosa (2018), "Neural Fine-Grained Entity Type Classification with Hierarchy-Aware Loss," *NAACL-HLT*
- Chen et al. (2019), "Improving Distantly-Supervised Entity Typing with Compact Latent Space Clustering," *NAACL-HLT*
- Wu et al. (2019), "Modeling Noisy Hierarchical Types in Fine-grained Entity Typing," *IJCAI*
- Onoe and Durrett (2019), "Learning to Denoise Distantly-Labeled Data for Entity Typing," *NAACL-HLT*
- Ren et al. (2016), "AFET: Automatic Fine-Grained Entity Typing by Hierarchical Partial-Label Embedding," *EMNLP*
- Mintz et al. (2009), "Distant Supervision for Relation Extraction without Labeled Data," *ACL-IJCNLP*

> **정확도 관련 고지:** 섹션 5(최신 연구 비교)의 일부 논문 세부 사항(특히 2021년 이후)은 제 학습 데이터 기반이며, 직접 접근한 PDF가 없어 세부 수치의 정확성을 100% 보장할 수 없습니다. 인용 전 원문 확인을 강력히 권장합니다.
