# Semi-Supervised Graph Imbalanced Regression (SGIR)

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

**SGIR(Semi-supervised framework for Graph Imbalanced Regression)**은 그래프 회귀 태스크에서 레이블 불균형(label imbalance) 문제를 준지도학습(semi-supervised learning) 방식으로 해결하는 최초의 프레임워크입니다.

기존 연구들은 다음 한계를 가집니다:
- 불균형 학습(imbalanced learning) 연구 → 주로 **분류(classification)** 문제에 집중
- 준지도학습 연구 → 그래프 데이터의 **연속적 레이블 불균형** 미고려
- 그래프 속성 예측 연구 → 레이블 불균형 문제 미해결

SGIR는 이 세 가지를 **동시에** 해결하는 최초의 시도입니다 (논문 Table 7 참조).

### 주요 기여 요약

| 기여 | 내용 |
|------|------|
| 새로운 문제 정의 | 그래프 불균형 회귀(Graph Imbalanced Regression) 태스크 |
| 회귀 신뢰도 측정 | GRation: 그래프 합리화 기반 예측 신뢰도 |
| 역방향 샘플링 | 소수 레이블 범위의 pseudo-label 우선 선택 |
| Label-Anchored Mixup | 잠재 공간에서의 목표 레이블 기반 데이터 증강 |
| 이론적 보장 | 일반화 오차 경계를 분류에서 회귀로 확장 |

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**그래프 불균형 회귀 문제**:

$$\frac{\max\{\mu_i\}}{\min\{\mu_i\}} \gg 1$$

위 조건이 성립할 때 레이블 불균형이 존재합니다. 레이블 공간을 $C$개의 구간 $[b_0, b_1), [b_1, b_2), \ldots, [b_{C-1}, b_C)$으로 분할하고, 각 구간의 빈도 집합 $\{\mu_i\}_{i=1}^{C}$를 구성할 때, 대부분의 구간에서 데이터가 심각하게 편중됩니다.

구체적인 문제 설정:
- **레이블 데이터**: $\mathcal{G}\_{\text{imb}} = \{(G_i, y_i)\}\_{i=1}^{n_{\text{imb}}}$ (불균형 분포)
- **비레이블 데이터**: $\mathcal{G}\_{\text{unlbl}} = \{G_j\}\_{j=n_{\text{imb}}+1}^{n_{\text{imb}}+n_{\text{unlbl}}}$ (대용량)
- **목표**: 인코더 $g: G \to \mathbf{h} \in \mathbb{R}^d$, 디코더 $f: \mathbf{h} \to \hat{y} \in \mathbb{R}$를 전체 레이블 공간에서 균등하게 잘 동작하도록 훈련

### 2.2 제안 방법 (수식 포함)

#### (1) 회귀 신뢰도 측정 (GRation)

배치 내 $i$번째 그래프 $G_i$를 rationale 서브그래프 $G_i^{(r)}$와 environment 서브그래프 $G_i^{(e)}$로 분리합니다.

같은 배치의 $j$번째 그래프의 environment를 결합한 $G_{(i,j)} = G_i^{(r)} \cup G_j^{(e)}$에 대해 신뢰도를 정의합니다:

```math
\sigma_i = \frac{1}{\text{Var}\left\{f(g(G_{(i,j)}))\right\}_{j=1,2,\ldots,B}}
```

GREA 모델을 사용해 잠재 공간에서 직접 계산:

```math
\sigma_i = \frac{1}{\text{Var}\left\{f(\mathbf{h}_i^{(r)} + \mathbf{h}_j^{(e)})\right\}_{j=1,2,\ldots,B}}
```

- $\sigma_i \geq \tau$인 경우만 고품질 예측으로 선택 $\rightarrow$ $\mathcal{G}_{\text{conf}}$ 구성

#### (2) 역방향 샘플링 (Reverse Sampling)

빈도 집합 $\{\mu_i\}\_{i=1}^C$를 역순으로 재정렬한 새로운 빈도 집합 $\{\mu_i'\}_{i=1}^C$를 정의합니다. $\mu_i$가 $\{\mu\}$에서 $k$번째로 크면, $\mu_i'$는 $k$번째로 작은 값입니다.

샘플링 비율:

$$p_i = \frac{\mu_i'}{\max\{\mu_1, \mu_2, \ldots, \mu_C\}} $$

- 소수 레이블 영역($\mu_i$가 작음)일수록 $p_i$가 커짐 → 더 많이 샘플링

#### (3) Label-Anchored Mixup

레이블 구간의 대표 표현 행렬 계산:

$$\mathbf{Z} = \text{norm}(\mathbf{M}) \cdot \mathbf{H} $$

- $\mathbf{M} \in \{0,1\}^{C \times n_{\text{imb}}}$: 레이블-구간 지시 행렬
- $\mathbf{H} \in \mathbb{R}^{n_{\text{imb}} \times d}$: GNN 인코더로 얻은 그래프 표현 행렬
- $\mathbf{z}_i$: $i$번째 구간의 대표 표현, $a_i$: 구간 중심 레이블

Mixup 연산 ($n_i \propto p_i$개의 실제 그래프 선택):

$$\begin{cases} \tilde{\mathbf{h}}_{(i,j)} = \lambda \cdot \mathbf{z}_i + (1-\lambda) \cdot \mathbf{h}_j \\ \tilde{y}_{(i,j)} = \lambda \cdot a_i + (1-\lambda) \cdot y_j \end{cases} $$

- $\lambda = \max(\lambda', 1-\lambda')$, $\lambda' \sim \text{Beta}(1, \beta)$
- $\lambda$가 1에 가깝게 설계 → 증강 데이터의 레이블이 앵커 $a_i$에 가까움

#### (4) 전체 최적화 손실 함수

$$\mathcal{L} = \sum_{(G,y) \in \mathcal{G}_{\text{imb}} \cup \mathcal{G}_{\text{conf}}} \ell_{\text{imb+conf}}(G, y) + \sum_{(\mathbf{h}, y) \in \mathcal{H}_{\text{aug}}} \ell_{\text{aug}}(\mathbf{h}, y)$$

- $\ell_{\text{imb+conf}} = \text{MAE}(f(g(G)), y)$
- $\ell_{\text{aug}} = \text{MAE}(f(\mathbf{h}), y)$

### 2.3 모델 구조

```
[비레이블 그래프 G_unlbl]
        ↓
   GNN 인코더 g(·): GIN
        ↓
  GREA 기반 신뢰도 σ_i 계산
        ↓
   역방향 샘플링 (p_i)
        ↓
   G_conf (신뢰 pseudo-label 집합)
        ↓
Label-Anchored Mixup → H_aug
        ↓
균형 학습 데이터: G_imb ∪ G_conf ∪ H_aug
        ↓
  디코더 f(·): 3-layer MLP
        ↓
  예측값 ŷ
        ↓
   [반복 Self-training]
```

**주요 컴포넌트:**
- **인코더**: Graph Isomorphism Network (GIN)
- **디코더**: 3층 MLP
- **기반 모델**: GREA (Graph Rationalization with Environment-based Augmentations)

### 2.4 성능 향상

주요 결과 (논문 Table 2, 3 기준):

| 데이터셋 | 비교 대상 (최고 베이스라인) | 전체 MAE 개선 | Few-shot MAE 개선 |
|---------|------------------------|-------------|-----------------|
| Mol-FreeSolv | GREA (0.642) | 0.563 (**12.3%↓**) | 0.777 (**30.3%↓**) |
| Mol-Lipo | LDS (0.468) | 0.432 (**9.1%↓**) | 0.515 (**6.5%↓**) |
| Mol-ESOL | GREA (0.497) | 0.457 (**8.1%↓**) | 0.604 (**7.4%↓**) |
| Plym-Oxygen | RankSim (165.7) | 150.9 (**8.9%↓**) | 382.8 (**9.0%↓**) |
| Superpixel-Age | RankSim (14.464) | 13.787 (**4.7%↓**) | 20.687 (**5.6%↓**) |

**핵심 관찰**: SGIR는 well-represented 영역의 성능을 희생하지 않으면서 few-shot 영역의 성능을 동시에 향상시킵니다. 기존 방법들(LDS, BMSE, RankSim)은 특정 영역 개선 시 다른 영역 성능이 저하됩니다.

### 2.5 한계

논문에서 명시적으로 인정한 한계 및 추론 가능한 한계:

1. **도메인 갭(Domain Gap) 의존성**: 비레이블 그래프가 레이블 데이터와 분포가 크게 다를 경우 pseudo-label 품질 저하. 분자(133,015개) 데이터는 성능 개선이 크나, 폴리머(13,114개) 데이터는 상대적으로 작은 개선 폭

2. **GREA 의존성**: 신뢰도 측정이 GREA 모델 구조에 특화되어 있어, 다른 GNN 아키텍처로의 일반화 시 수정 필요

3. **계산 비용**: Self-training 반복 + 배치 내 rationale/environment 분리로 인한 추가적 계산 부담

4. **이론적 한계**: Mixup 증강 데이터가 해당 구간의 조건부 분포 $\mathcal{P}\_{[b_i, b_{i+1}]}$로부터 i.i.d. 샘플임을 엄밀히 보장하지 못함 (논문 Appendix B.3에서 자체 인정)

5. **하이퍼파라미터 민감도**: 구간 수 $C$, 신뢰도 임계값 $\tau$, Beta 분포 파라미터 $\beta$ 등 조정 필요

---

## 3. 일반화 성능 향상 가능성

### 3.1 이론적 일반화 경계

논문은 분류의 margin-based 일반화 경계를 회귀로 확장한 Theorem 4.1을 제시합니다:

$$\mathcal{E}_{[b_i, b_{i+1})}[f] \lesssim \frac{1}{\gamma_{[b_i, b_{i+1})}} \sqrt{\frac{C(\mathcal{F})}{n_{[b_i, b_{i+1})}}} + \sqrt{\frac{\log\log_2(1/\gamma_{[b_i, b_{i+1})}) + \log(1/\delta)}{n_{[b_i, b_{i+1})}}} $$

- $\gamma_{[b_i, b_{i+1})}$: $i$번째 구간의 학습 마진 (최소 마진)
- $n_{[b_i, b_{i+1})}$: 해당 구간의 학습 예제 수
- $C(\mathcal{F})$: 가설 클래스 $\mathcal{F}$의 복잡도 (Rademacher complexity)

전체 균형 분포 오차:

$$\mathcal{E}_{\text{bal}}[f] \lesssim \frac{1}{C}\sum_{i=1}^{C} \mathcal{E}_{[b_i, b_{i+1})}[f]$$

**일반화 향상 메커니즘**:
- $n_{[b_i, b_{i+1})}$ 증가: pseudo-label과 mixup 증강을 통해 소수 레이블 구간의 샘플 수를 늘림 → 오차 경계 감소
- 구간 간 균형화: 모든 $i$에 대해 $n_{[b_i, b_{i+1})}$를 균등하게 만들어 $\mathcal{E}_{\text{bal}}$을 최소화
- 기존 분류 기반 접근법과 달리, **마진 조작 없이** 샘플 수 증가로 안전하게 경계 감소

### 3.2 실험적 일반화 증거

**다도메인 일반화**: 분자(Mol-*), 폴리머(Plym-*), 이미지 슈퍼픽셀(Superpixel-Age) 세 도메인에서 일관된 성능 향상 확인

**Self-training 반복 효과** (Figure 4): 반복이 증가할수록 전체 레이블 범위와 few-shot 영역 모두에서 MAE가 점진적으로 감소. 모델 편향과 데이터 품질이 상호 강화(mutual enhancement) 구조를 형성

**Well-represented 영역 보호**: 기존 방법(LDS는 Mol-FreeSolv에서 many-shot 영역 MAE -29% 감소)과 달리, SGIR는 well-represented 영역 성능을 유지하면서 few-shot 개선

### 3.3 일반화 향상의 추가 가능성

1. **더 나은 GNN 백본**: GIN 대신 더 표현력 높은 GNN (예: Graph Transformer)으로 인코더 교체 시 성능 향상 기대

2. **더 많은 비레이블 데이터**: 폴리머 데이터의 비레이블 셋이 분자보다 적어 성능 개선폭이 작음 → 더 많은 비레이블 폴리머 데이터 수집 시 개선 기대

3. **적응적 구간 분할**: 고정된 균일 구간 대신 데이터 분포에 적응적인 구간 설정으로 경계 최적화 가능

4. **Dropout 대안**: 논문에서 GRation의 대안으로 Dropout이 효과적임을 보임 → 그래프 외 이미지, 텍스트 도메인의 회귀 불균형 문제로 확장 가능

---

## 4. 미래 연구에 미치는 영향과 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### 새로운 연구 방향 개척

1. **그래프 불균형 회귀의 벤치마크 확립**: 7개 데이터셋과 평가 프로토콜(many/medium/few-shot 분류)이 후속 연구의 표준 벤치마크로 활용될 수 있음

2. **회귀 신뢰도 측정의 새로운 패러다임**: GRation 방식의 환경 서브그래프 기반 불확실성 측정은 능동 학습(active learning), 분포 외 탐지(OOD detection) 등으로 확장 가능

3. **준지도 그래프 학습의 확장**: 기존 준지도 그래프 학습이 주로 노드 분류에 집중되었으나, SGIR는 그래프 레벨 회귀로의 확장 가능성을 실증

4. **과학 데이터(약물, 소재) ML에의 기여**: 레이블 불균형이 만성적인 분자/폴리머 속성 예측 분야에서 직접적 활용 가능

#### 관련 분야에 미치는 영향

- **분류→회귀 이론 확장**: Theorem 4.1의 framework는 다른 연속 레이블 학습 이론 발전에 기여
- **Self-training 재설계**: 분류 중심의 CReST, DARP 등이 회귀 태스크로 재설계되는 데 참고 모델 제공
- **데이터 효율적 학습**: 소규모 레이블+대규모 비레이블 구조는 의료, 환경 데이터 등 레이블 획득 비용이 높은 도메인에 광범위하게 적용 가능

### 4.2 향후 연구 시 고려할 점

#### 방법론적 측면

1. **더 정교한 신뢰도 측정**
   - GRation은 GREA 구조에 의존적 → 범용적 신뢰도 측정 방법 개발 필요
   - Conformal Prediction 기반의 이론적으로 보장된 신뢰 구간 활용 검토
   - 에너지 기반 모델(EBM)이나 베이지안 딥러닝과의 결합 가능성

2. **Label-Anchored Mixup의 개선**
   - 현재 잠재 공간에서의 선형 보간 → 비선형 보간(geodesic mixup, Riemannian manifold 기반)으로 더 현실적인 증강 데이터 생성 가능
   - Diffusion 모델을 활용한 그래프 구조 수준의 생성적 증강 (논문 [24]에서 일부 탐색)
   - C-Mixup [53]과의 결합: 연속 레이블을 고려한 쌍 선택 전략과 SGIR의 불균형 해소 전략 통합

3. **구간 설정의 자동화**
   - 현재 $C$를 하이퍼파라미터로 수동 설정 → 정보 이론적 기준(MDL, 엔트로피 최소화)으로 자동 설정
   - 비균일(non-uniform) 구간 설정 탐색

4. **도메인 적응 통합**
   - 레이블/비레이블 데이터 간 도메인 갭 처리를 위한 명시적 도메인 적응 메커니즘 필요
   - Sample selection bias 이론 [논문 참조 10, 39]과 SGIR의 결합

#### 이론적 측면

5. **구간 없는 이론 개발**
   - 현재 이론은 연속 레이블을 $C$개 구간으로 이산화 → 연속 레이블 공간에서 직접 작동하는 일반화 경계 개발 필요
   - 혼합 회귀 모델(mixture regressor models)을 활용한 covariate shift 이론과의 연결

6. **Mixup 증강의 i.i.d. 보장**
   - 논문 자체적으로 인정한 한계: 증강 데이터가 조건부 분포의 i.i.d. 샘플임을 보장하지 못함 → 엄밀한 이론적 분석 필요

#### 실용적 측면

7. **계산 효율성**
   - 대규모 비레이블 데이터셋(146,129개)에 대한 반복적 pseudo-labeling의 계산 비용 → 효율적 배치 선택 전략 필요
   - 점진적(online) self-training으로의 전환 고려

8. **다른 도메인으로의 확장**
   - 의료 이미지의 연속적 진단 수치 예측
   - 교통 네트워크의 속도/밀도 예측
   - 소셜 네트워크의 영향력(continuous) 예측
   - 이때 GRation의 Dropout 대안이 유용

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 불균형 회귀 관련 연구

| 연구 | 연도 | 핵심 방법 | SGIR와의 관계 |
|------|------|----------|-------------|
| **DIR (Yang et al., ICML 2021)** [논문 52] | 2021 | LDS(Label Distribution Smoothing), FDS(Feature Distribution Smoothing), Focal Loss | SGIR의 주요 비교 대상; 레이블 분포 평활화로 소수 레이블 개선 but 다수 레이블 성능 저하 |
| **BMSE (Ren et al., CVPR 2022)** [논문 35] | 2022 | Balanced MSE: 레이블 사전 분포 기반 손실 함수 재보정 | 비그래프 데이터; 준지도학습 미활용; SGIR에 비해 few-shot 영역 개선 제한적 |
| **RankSim (Gong et al., ICML 2022)** [논문 13] | 2022 | 잠재 공간에서 레이블 유사도와 표현 유사도 정렬 정규화 | 준지도학습 미활용; SGIR와 상호 보완적 (정규화 + 데이터 균형화 결합 가능) |
| **C-Mixup (Yao et al., NeurIPS 2022)** [논문 53] | 2022 | 연속 레이블 기반 쌍 샘플링: 가까운 레이블 쌍을 높은 확률로 선택 | 불균형 미고려; 오히려 다수 레이블 쌍이 더 자주 선택되어 불균형 악화 가능 |

### 5.2 준지도 그래프 학습 관련 연구

| 연구 | 연도 | 핵심 방법 | SGIR와의 관계 |
|------|------|----------|-------------|
| **InfoGraph (Sun et al., ICLR 2020)** [논문 40] | 2020 | 상호 정보 최대화 기반 비레이블 그래프 표현 학습 | SGIR의 비교 대상; 불균형 미고려로 성능 열세 |
| **GREA (Liu et al., KDD 2022)** [논문 25] | 2022 | 환경 기반 증강으로 합리적 서브그래프 식별 | SGIR의 기반 모델; SGIR는 GREA의 신뢰도 개념을 pseudo-label 품질 측정으로 확장 |
| **Data-Centric from Unlabeled Graphs with Diffusion (Liu et al., 2023)** [논문 24] | 2023 | Diffusion 모델로 비레이블 그래프에서 데이터 중심 학습 | SGIR의 저자들이 진행한 후속 연구; 더 고품질의 그래프 생성 가능성 |

### 5.3 불균형 준지도 분류 관련 연구 (SGIR가 회귀로 확장한 방법론)

| 연구 | 연도 | 핵심 방법 | SGIR와의 관계 |
|------|------|----------|-------------|
| **DARP (Kim et al., NeurIPS 2020)** [논문 20] | 2020 | 진짜 클래스 분포 추정으로 pseudo-label 정제 | SGIR의 역방향 샘플링은 이를 회귀로 확장 |
| **CReST (Wei et al., CVPR 2021)** [논문 47] | 2021 | 소수 클래스에 더 많은 pseudo-label 선택 | SGIR 역방향 샘플링의 직접적 inspiration |
| **DASO (Oh et al., CVPR 2022)** [논문 32] | 2022 | 분포 인식 시맨틱 pseudo-label | SGIR와 동일 계열 but 분류에 한정 |

### 5.4 Mixup 관련 최신 이론 연구

| 연구 | 연도 | 핵심 내용 | SGIR와의 관계 |
|------|------|----------|-------------|
| **Zhang et al., ICLR 2021** [논문 55] | 2021 | Mixup이 강건성과 일반화에 미치는 이론적 분석 | SGIR의 mixup 설계에 이론적 근거 제공 |
| **Liu et al., ICLR 2023** [논문 26] | 2023 | 과도한 Mixup 학습이 일반화를 해칠 수 있음 경고 | SGIR에서 $\lambda$를 1에 가깝게 설계한 것은 이 위험을 부분적으로 회피 |

### 5.5 종합 비교 포지셔닝

```
준지도 학습
    ↑
InfoGraph --- SGIR (그래프+회귀+준지도+불균형)
    |              ↑
    |         GREA (그래프+회귀+감독)
    ↓
분류 기반 ← DARP, CReST, DASO (준지도+불균형+분류)
    
불균형 회귀
    ↑
DIR, BMSE, RankSim (감독+비그래프+불균형+회귀)
    ↓
비그래프 데이터
```

SGIR는 준지도 학습 × 그래프 데이터 × 불균형 × 회귀의 교차점을 최초로 채운 연구입니다.

---

## 참고 자료

**1차 출처 (논문 PDF)**
- Gang Liu, Tong Zhao, Eric Inae, Tengfei Luo, Meng Jiang. *"Semi-Supervised Graph Imbalanced Regression"*. KDD 2023. arXiv:2305.12087v1

**논문 내 핵심 참고 문헌**
- [25] Gang Liu et al. *"Graph Rationalization with Environment-based Augmentations"*. KDD 2022.
- [52] Yuzhe Yang et al. *"Delving into Deep Imbalanced Regression"*. ICML 2021.
- [13] Yu Gong et al. *"RankSim: Ranking Similarity Regularization for Deep Imbalanced Regression"*. ICML 2022.
- [35] Jiawei Ren et al. *"Balanced MSE for Imbalanced Visual Regression"*. CVPR 2022.
- [47] Chen Wei et al. *"CReST: A Class-Rebalancing Self-Training Framework for Imbalanced Semi-Supervised Learning"*. CVPR 2021.
- [53] Huaxiu Yao et al. *"C-Mixup: Improving Generalization in Regression"*. NeurIPS 2022.
- [40] Fan-Yun Sun et al. *"InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization"*. ICLR 2020.
- [51] Keyulu Xu et al. *"How Powerful are Graph Neural Networks?"*. ICLR 2019.
- [7] Kaidi Cao et al. *"Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss"*. NeurIPS 2019.
- [24] Gang Liu et al. *"Data-Centric Learning from Unlabeled Graphs with Diffusion Model"*. arXiv:2303.10108, 2023.
- [26] Zixuan Liu et al. *"Over-Training with Mixup May Hurt Generalization"*. ICLR 2023.
- [55] Linjun Zhang et al. *"How Does Mixup Help With Robustness and Generalization?"*. ICLR 2021.

> **주의사항**: 본 답변은 제공된 논문 PDF(arXiv:2305.12087v1)를 직접 분석하여 작성되었습니다. 2020년 이후 관련 연구 비교 분석 부분은 논문 내 인용 문헌을 기반으로 하였으며, 논문 출판 이후(2023년 5월 이후)의 후속 연구는 포함되지 않았습니다.
