
# Feature Importance Ranking for Deep Learning

## 1. 논문의 핵심 주장 및 기여도 요약

**논문명**: "Feature Importance Ranking for Deep Learning"  
**저자**: Maksymilian A. Wojtas, Ke Chen (University of Manchester)  
**발표**: NeurIPS 2020 (인용 수: 228회)

본 논문은 딥러닝 모델에서 **모집단 수준의 특성 중요도 순위(Population-wise Feature Importance Ranking, FIR)** 문제를 해결하는 혁신적인 접근법을 제시한다. 핵심 기여도는 다음과 같다:

1. **이중망 아키텍처 제안**: 연산자 네트(operator net)와 선택자 네트(selector net)로 구성된 새로운 신경망 구조
2. **교대학습 알고리즘 개발**: 두 네트워크를 공동으로 훈련하는 혁신적인 학습 방식
3. **조합 최적화 해결**: 확률적 국소탐색(stochastic local search) 절차를 통합하여 NP-하드 문제 극복
4. **성능 우월성**: 기존 DFS, AvGrad, FS 등의 심층학습 방법 및 LASSO, 랜덤포레스트 등의 고전적 방법들을 능가

***

## 2. 해결하고자 하는 문제 상세 설명

### 2.1 문제 정의

특성 중요도 순위 문제는 다음과 같이 수학적으로 정의된다:

$$m^\*, \text{Score}(m^*) = \arg\max_{m \in M} \sum_{x \in X} Q(x, m)$$

여기서:
- $D = \{X, Y\}$: 훈련 데이터셋
- $m \in M$: $d$차원 이진 마스크 벡터 ($m_0 s, s \leq d$, $|M| = \binom{d}{s}$)
- $Q(x,m)$: 특성 부분집합 $x \odot m$에 기반한 연산자의 인스턴스 수준 성능
- $m^*$: 최적 특성 부분집합의 지시자
- $\text{Score}(m^*)$: 선택된 특성들의 중요도 점수

### 2.2 핵심 문제점

1. **조합 최적화의 복잡성**: $d$개 특성 중 $s$개를 선택하는 경우의 수는 $\binom{d}{s}$로 지수적으로 증가
   - $d=100, s=50$인 경우: $\approx 10^{29}$ 가지 조합

2. **딥러닝의 비선형성**: 특성과 목표 변수 간의 복잡한 함수적 종속성을 정확히 파악하기 어려움

3. **기존 방법들의 한계**:
   - **정규화 기반 방법(DFS)**: 높은 계산 비용, 그래디언트 소실 문제, 선형 모델에만 이론적 정당성
   - **탐욕 탐색(Forward Selection)**: 부분최적 결과, 극도로 높은 계산 복잡도
   - **평균 입력 그래디언트(AvGrad)**: 인스턴스 수준 설명을 모집단 수준으로 단순 집계하여 일관성 부재

***

## 3. 제안 방법론 상세 설명

### 3.1 이중망 아키텍처

![논문 그림 1]에서 보듯이, 제안된 모델은 다음 두 개의 신경망으로 구성:

**1) 연산자 네트 $f_O(x,m|\theta)$**
- 주어진 특성 부분집합 $m$에서 지도학습 작업 수행
- 다층퍼셉트론(MLP) 또는 합성곱 신경망(CNN)으로 구현
- 개별 특성 부분집합에 대한 성능 평가

**2) 선택자 네트 $f_S(m|\phi)$**
- 연산자의 성능 피드백을 기반으로 최적 특성 부분집합 학습
- 서로 다른 특성 부분집합 후보에 대한 연산자의 평균 성능 예측
- MLP로 구현된 회귀 모델: $f_S: M \rightarrow \mathbb{R}$

### 3.2 학습 알고리즘: 교대학습 방식

#### **Phase I: 초기 연산자 학습 (탐색)**

$$\mathcal{L}\_O(D, M_1) = \frac{1}{|M|} \sum_{m \in M_1} \sum_{(x,y) \in D} l(x \odot m, y)$$

- 무작위 특성 부분집합 $M_1$에서 여러 에포크 동안 연산자 훈련
- 선택자 네트의 초기 훈련 데이터 생성

#### **Phase II-A: 선택자 학습**

선택자는 다음 가중 손실함수로 훈련:

$$\mathcal{L}_S(M) = \frac{1}{|M|} \sum_{m \in M} w_m \left(\sum_{(x,y) \in D} l(x \odot m, y)\right)^2$$

여기서 

```math
w_m = \begin{cases} 10 & m = m_{t,\text{best}} \\ 5 & m = m_{t+1,\text{opt}} \\ 1 & \text{otherwise} \end{cases}
```

**최적 부분집합 생성 절차** (3단계 검증):

$$m_0 = \left[\frac{1}{2}, \frac{1}{2}, \ldots, \frac{1}{2}\right]$$

$$\nabla m_0 = \frac{\partial f_S(m)}{\partial m}\Big|_{m=m_0}$$

$$m_{\text{opt}}, m_{\overline{\text{opt}}} = \text{argsort}(\nabla m_0, s)$$

i) **검증 1**: 선택된 특성의 기울기 재계산 및 음수 기울기 특성 치환
ii) **검증 2**: 최소 기울기 특성과 최대 기울기 미선택 특성 간 교환
iii) **최적성 보장**: $f_S(m_{\text{opt}}) \geq f_S(m_{\text{opt}}')$ 확인 시까지 반복

**특성 부분집합 후보 생성** (탐색-활용 전략):

$$M_{t+1,1} = \text{RandomMask}(d, s, |M_{t+1,1}|)$$ (탐색 - 무작위 부분집합)

$$M_{t+1,2} = \{m_{t,\text{best}}, m_{t+1,\text{opt}}, \text{Perturb}(m_{t+1,\text{opt}}, s_p), \ldots\}$$ (활용 - 입력 그래디언트 기반)

여기서 $\text{Perturb}(m_{\text{opt}}, s_p)$는 $s_p$개 요소를 무작위로 반전시킴:

$$\text{Perturb}(m_{\text{opt}}, s_p): 1 \rightarrow 0 \text{ or } 0 \rightarrow 1$$

#### **Phase II-B: 연산자 학습**

$$\theta_{t+1} \leftarrow \theta_t - \eta_\theta \nabla_\theta \mathcal{L}_O(D, M_{t+1})$$

선택자로부터 제공된 후보 $M_{t+1}$에서 연산자 업데이트

### 3.3 모델 구조

**연산자 네트 구현의 특수성**:

선택된 특성 $x \odot m$과 마스크 $m$을 연결하여 입력:

$$\text{Input}_{\text{actual}} = [x \odot m; m]$$

- **목적**: 특성값이 0인 경우와 마스크된 특성을 구별
- 입력 차원: $d$ → $2d$로 확장

**하이퍼파라미터**:
- 연산자: 20-60-30-20 (합성곱층) 또는 유사 MLP 구조
- 선택자: 10-100-50-10 구조
- 최적화: Adam + SGD, 조기 종료 사용

***

## 4. 성능 향상 및 한계

### 4.1 성능 향상

#### **종합 벤치마크 결과**

| 데이터셋 | 방법 | 정확도/MSE | FIR 품질 | 특성 수 |
|---------|------|----------|---------|--------|
| **XOR 분류** | 제안방법 | 98.8%↑ | 완벽 | 5/10 |
| | DFS | 97.4% | 불완전 | 212 |
| | AvGrad | 99.3% | 불완전 | 784 |
| **비선형 회귀** | 제안방법 | MSE: 최소 | 완벽 | 5/10 |
| | RF | MSE: 높음 | 완벽 | - |
| **Glass (9feat)** | 제안방법 | **성능↑** | - | 우수 |
| | CCM | 유사성능 | - | 비교 |
| **Vowel (10feat)** | 제안방법 | **성능↑** | - | 우수 |
| **MNIST (784pix)** | 제안방법 | 99.31%±0.08 | 의미있음 | 85/784 |

#### **일반화 성능 개선**

| 데이터셋 | 훈련 손실 | 검증 손실 | 테스트 손실 | 일반화갭 |
|---------|---------|---------|-----------|---------|
| 합성 회귀 | ↓ 지속감소 | ↓ 지속감소 | ↓ 최소 | 우수 |
| MNIST | ↓ 급격히 감소 | ↓ 감소 | ↓ **테스트>검증** | **매우 우수** |
| Yale | ↑ 증가 | ↑ 증가 | **정확도 유지** | 이상현상* |

*Yale 데이터셋의 이상현상: 공변량 시프트와 제한된 훈련 데이터 때문에 탐색이 많아짐

### 4.2 일반화 성능 향상 메커니즘

#### **1) 다양한 특성 부분집합 학습**

$$\text{학습 예제 수} = |D| \times |M| \gg |D|$$

- 각 훈련 배치에서 여러 특성 부분집합에 대해 학습
- 부분집합 크기 $s < d$이므로 실질적인 특성 감소

#### **2) 암묵적 정규화 효과**

- **Dropout의 변형**: 임의 입력 노드 드롭아웃과 유사
- 다양한 특성 조합에 노출되면서 모델 견고성 증가
- 과적합 감소 및 일반화 성능 향상

#### **3) 확률적 탐색의 노이즈 효과**

$$M_{t+1,1} \cup \text{Perturb}(m_{\text{opt}}, s_p)$$

- 무작위 탐색과 가우드 탐색의 균형
- 국소최적값에서 탈출 가능
- 모델의 안정성과 견고성 증가

***

## 5. 한계 및 제약사항

### 5.1 계산 복잡도

**Phase I**: $E_1$ 에포크 동안 $|M_1|$개 마스크로 훈련
**Phase II**: $|M_{t+1,1}| + |M_{t+1,2}|$개 후보로 교대 훈련

- **시간 복잡도**: $O(E_{\text{total}} \times |D| \times |M|)$ (높음)
- **Yale 데이터셋**: 약 1,100초 소요
- **비교**: RF(2.5초), RFE(90초), DFS(35초)

### 5.2 확장성 문제

1. **고차원 데이터**: 
   - TOX-171 (5,784 특성 × 109 샘플/fold) 에서 CCM에 비해 우월하지 않음
   - 깊은 학습은 충분한 훈련 데이터 필요

2. **수렴 보장 부재**:
   - 교대학습이 국소최적값에 수렴하는 이론적 보증 없음
   - 실증적 관찰상 일반적으로 수렴하지만 보장 불가

### 5.3 모델 아키텍처 의존성

- 연산자로 사용되는 신경망 구조에 크게 의존
- 특성 크기별로 최적 아키텍처 다름
- 하이퍼파라미터 튜닝 필요

### 5.4 학습 행동의 이상 현상

Yale 얼굴 데이터셋에서 관찰:
- 검증 손실 증가 → 검증 정확도 유지
- 이유: 공변량 시프트(covariate shift) + 제한된 훈련 데이터
- 조기 종료 조건 결정의 어려움

***

## 6. 최신 연구 비교 분석 (2020-2025)

### 6.1 주요 최신 기술 동향

#### **1) SHAP 기반 설명 가능 AI (2021-2025)**

**특성**: 게임 이론 기반 Shapley 값 활용

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}[f(S \cup \{i\}) - f(S)]$$

**장점**:
- 이론적 기반 견고함
- 전역 특성 중요도 제공
- 지역 설명도 가능

**한계**:
- 계산 비용 높음 ($2^d$ 개 부분집합 필요)
- 특성 독립성 가정 부정확
- 불안정성 문제 존재 [arxiv](https://arxiv.org/pdf/2503.23111.pdf)

**최근 개선**: 
- "Extended Support"를 고려한 수정된 SHAP [arxiv](https://arxiv.org/pdf/2503.23111.pdf)
- KernelSHAP의 이론적 보증 제공 [arxiv](https://arxiv.org/pdf/2503.23111.pdf)
- LLM을 활용한 해석성 개선 [arxiv](https://arxiv.org/pdf/2409.00079.pdf)

#### **2) 어텐션 메커니즘 기반 특성 선택 (2021-2025)**

**Sequential Attention** :
- Attention 가중치를 특성 중요도 프록시로 사용
- OMP(Orthogonal Matching Pursuit) 알고리즘과 이론적 연결

$$w_i = \text{softmax}(\text{attention}(\text{query}, \text{key}_i))$$

**장점**:
- 모델 아키텍처에 내재된 투명성
- 계산 효율적
- 장거리 의존성 포착

**한계**:
- Attention 가중치가 항상 특성 중요도를 반영하지 않음
- 해석 어려움

#### **3) 기울기 기반 특성 중요도 (2022-2025)**

**Gradient-based Feature Importance**:
- Saliency map 기반 접근
- 입력에 대한 출력 그래디언트 활용

$$\text{FI}_i = \left|\frac{\partial f(x)}{\partial x_i}\right| \text{ or } \sum_j \left(\frac{\partial f(x)}{\partial x_i}\right)^2$$

**Weight Profile 분석**: [nature](https://www.nature.com/articles/s41598-024-72640-4)
- 쌍별 가중층의 그래디언트 변화 추적
- 훈련 중 계층별 중요도 변화 분석

**최근 발전**:
- Gradient-based Causal Feature Selection (GCFS) [ijcai](https://www.ijcai.org/proceedings/2025/0636.pdf)
- AutoEncoder + 미분 가능 특성 선택

#### **4) 최신 신경망 기반 특성 선택 (2024-2025)**

**GFSNetwork**: [arxiv](https://arxiv.org/pdf/2503.13304.pdf)
- Temperature-controlled Gumbel-Sigmoid 샘플링
- 자동 및 미분 가능 특성 선택
- 선형 계산 오버헤드: $O(d)$ (상수)

$$s_i = \text{Gumbel-Sigmoid}(\log \alpha_i, \tau)$$

**장점**:
- 확장성 우수 (고차원 데이터)
- 특성 수 감소: 평균 47.83%
- 정확도 개선: 63.33% 모델에서 향상

**FeatureX**: [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417425012977)
- 특성 중요도 분석이 통합된 설명 가능 특성 선택
- SHAP과 비교 시 우수한 성능

#### **5) 앙상블 기반 안정성 개선 (2022)**

**Ensembling improves stability**: [arxiv](https://arxiv.org/pdf/2210.00604.pdf)
- 심층 학습 특성 중요도의 고유한 불안정성 확인
- 여러 실행에서 앙상블 방식으로 결합

$$\text{FI}_{\text{ensemble}} = \frac{1}{N} \sum_{k=1}^N \text{FI}^{(k)}$$

**발견**: 동일한 훈련 조건에서도 특성 선택 결과 변동
**해결책**: 앙상블 방식으로 안정성 98% 향상

***

### 6.2 비교 분석 표

| 방법 | 발표년 | 계산복잡도 | 확장성 | 일반화성 | 해석성 | 안정성 |
|-----|------|---------|------|--------|------|------|
| **본 논문 (FIR)** | 2020 | 높음 | 중간 | **우수** | 중간 | 중간 |
| SHAP | 2017 | 매우높음 | 낮음 | 좋음 | **우수** | 문제있음 [arxiv](https://arxiv.org/pdf/2503.23111.pdf) |
| GFSNetwork | 2025 | **낮음** | **우수** | 좋음 | 중간 | 우수 |
| FeatureX | 2025 | 중간 | 중간 | 좋음 | **우수** | 중간 |
| Sequential Attention | 2024 | 낮음 | 우수 | 좋음 | 중간 | 중간 |
| Gradient-based GCFS | 2025 | 중간 | 우수 | 우수 | 중간 | 좋음 |

***

## 7. 일반화 성능 향상과 관련된 심화 분석

### 7.1 데이터 대표성과 특성 선택의 관계

**가설**: 최적 특성 부분집합 학습이 일반화 성능을 향상시키는 메커니즘

$$\text{Risk}_{\text{train}}(m^*) < \text{Risk}_{\text{test}}(m^*) \quad \text{(통상적)}$$

하지만 본 논문에서:

$$\text{Risk}_{\text{test}}(m^*) < \text{Risk}_{\text{valid}}(m^*) \quad \text{(MNIST 데이터)}$$

**설명**:
1. **효과적인 특성 선택**: 불필요한 특성 제거로 인한 편향 감소
2. **정규화 효과**: 여러 부분집합에 대한 노출 = 암묵적 $L_1$ 정규화
3. **모델 용량 감소**: $d \rightarrow s$ 특성으로 모델 복잡도 감소 = VC 차원 감소

### 7.2 일반화 오류 한계

딥러닝에서 일반화 오류는:

$$\epsilon_{\text{gen}} \leq \epsilon_{\text{train}} + \mathcal{O}\left(\sqrt{\frac{d \log(d/\delta)}{n}}\right)$$

특성 선택 후:

$$\epsilon_{\text{gen}}' \leq \epsilon_{\text{train}} + \mathcal{O}\left(\sqrt{\frac{s \log(s/\delta)}{n}}\right)$$

$s \ll d$이면 $\epsilon_{\text{gen}}' \ll \epsilon_{\text{gen}}$

### 7.3 실증적 발견

**MNIST 부분집합**:
- 기울기 정보를 통한 픽셀 선택
- 숫자 3과 8 구별에 핵심적 픽셀만 선택 → 명확한 시각화

**Enhanced-Promoter DNA 데이터**:
- DNA 서열 특성 선택에서 생물학적으로 의미있는 결과
- 기존 DFS 보다 일관성 있는 FIR 점수

***

## 8. 향후 연구에 미치는 영향 및 고려 사항

### 8.1 학문적 기여도

#### **1) 특성 선택 분야의 패러다임 전환**

**이전**: 필터-래퍼-임베딩 방식 분류
**이후**: 신경망 기반 자동 최적화 + 설명 가능성 통합

#### **2) 조합 최적화 해결 방식**

- 순수 탐색 vs 탐욕 탐색의 이분법 극복
- **탐색-활용 균형**이 조합 최적화의 핵심임을 입증

#### **3) 이중망 아키텍처의 보편성**

- 감독 특성 선택 이외에 비감독, 그룹 기반 FIR로 확장 가능
- 진화 알고리즘과의 연결고리 발견

***

### 8.2 실무 적용 시 고려 사항

#### **1) 계산 비용 vs 성능 트레이드오프**

| 상황 | 추천 방법 |
|-----|---------|
| 높은 정확도 필수 (의료/금융) | 본 논문 + 더 나은 최적화 |
| 빠른 추론 필수 (엣지 컴퓨팅) | GFSNetwork [arxiv](https://arxiv.org/pdf/2503.13304.pdf) |
| 설명성 최우선 (규제) | SHAP + FeatureX [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417425012977) |
| 대규모 데이터 (빅데이터) | Sequential Attention |

#### **2) 데이터 특성별 적용 전략**

**작은 데이터셋** ($n < 1000$):
- 본 논문 방법 적용 가능
- 과적합 주의, 조기 종료 필수

**중간 규모** ($1000 < n < 10^6$, $d < 1000$):
- GFSNetwork 추천
- 계산 효율성 + 일반화 성능 우수

**대규모 데이터** ($n > 10^6$, $d > 10000$):
- 기울기 기반 GCFS [ijcai](https://www.ijcai.org/proceedings/2025/0636.pdf)
- 또는 attention 기반 방법

**고차원 데이터** ($d > 10000$):
- 사전 차원 축소 필수
- PCA/UMAP 후 이중망 적용

***

### 8.3 향후 개선 방향

#### **1) 계산 복잡도 감소**

현재: $O(E_{\text{total}} \times |M|)$ (높음)

개선 방안:
- **Efficient networks** (EfficientNet) 활용 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12465982/)
- **분산 훈련**: 여러 마스크를 병렬 처리
- **가우드 탐색 개선**: 더 적은 후보로 동일 성능

#### **2) 이론적 수렴 분석**

현재: 경험적 수렴만 확인
향후: 다음 분석 필요:
- 교대학습 수렴성 보증
- 근처 최적성(approximate optimality) 한계 도출

$$f(m^*) \geq (1-\epsilon) \cdot \max_{m} f(m)$$

#### **3) 안정성 개선**

현재 문제: 여러 실행에서 특성 선택 변동
해결책:
- 앙상블 방식 통합 [arxiv](https://arxiv.org/pdf/2210.00604.pdf)
- 확률적 초기화 제한

#### **4) 설명 가능성 강화**

현재: 선택된 특성의 기울기만 이용
개선:
- SHAP과 결합된 하이브리드 접근
- Counterfactual explanation 추가
- 특성 상호작용 분석

#### **5) 비지도 학습 확장**

현재: 감독 학습에만 적용
향후:
- Autoencoder 기반 비지도 FIR
- 클러스터링 작업에서 핵심 특성 발견

***

### 8.4 최신 연구와의 시너지

#### **1) Vision Transformer (ViT) 시대의 특성 선택**

**도전**: 이미지에서 "특성"의 정의 애매
**기회**: Patch-based 토큰화 → 자연스러운 특성 단위

$$\text{논문 방법} + \text{ViT Attention} = \text{강력한 설명 가능 AI}$$

#### **2) 인과관계 발견과의 통합**

**GCFS (Causal)**  + **본 논문 (구조)**: [ijcai](https://www.ijcai.org/proceedings/2025/0636.pdf)
- 특성 선택 + 인과 관계 동시 파악
- 단순 상관성이 아닌 진정한 인과 관계 식별

$$\text{인과 그래프} \times \text{신경망 최적화}$$

#### **3) 설명 가능 AI (XAI)의 표준화**

현재 분열:
- SHAP (전역), LIME (지역), Attention (구조적)

미래:
- 통합 프레임워크로 발전
- 본 논문의 이중망 구조가 메타레벨 설명성 제공 가능

***

## 9. 결론: 학문적 가치와 실무 적용 전망

### 9.1 핵심 성과 재검토

| 차원 | 평가 |
|-----|------|
| **문제 정의** | 명확하고 중요한 문제 해결 |
| **방법론** | 참신한 이중망 + 교대학습 |
| **성능** | 여러 벤치마크에서 우월 [semanticscholar](https://www.semanticscholar.org/paper/235be0aa43cc1cf92b28926e0d34ef4b155053e4) |
| **일반화** | 암묵적 정규화로 일반화 성능 향상 |
| **복잡도** | 계산 비용 높음 (한계) |
| **적용 범위** | 광범위하지만 대규모에서 제한 |

### 9.2 2020-2025년 발전 맥락

| 기간 | 트렌드 | 본 논문의 위상 |
|-----|------|-------------|
| **2020** | 딥러닝 해석성 초기 | **선구적 작업** |
| **2021-2022** | SHAP, LIME 확산 | 보완적 역할 |
| **2023-2024** | Attention, Causal AI | **통합 가능** |
| **2025** | XAI 표준화 추진 중 | **참고 대상** |

### 9.3 최종 권고사항

**학술 연구자**: 
- 인용 가치 높음 (228회)
- 향후 개선 아이디어 풍부
- 특히 계산 복잡도 개선, 비지도 학습 확장 가능

**산업 실무자**:
- 소규모~중규모 데이터: **직접 적용 가능**
- 대규모 데이터: **GFSNetwork 우선 고려**
- 규제 대상 산업(의료/금융): **SHAP 하이브리드** 추천

**향후 연구 기획**:
- 본 논문 + GFSNetwork = 새로운 하이브리드 (효율성 + 성능)
- 본 논문 + GCFS = 인과-신경망 통합 (해석성 + 인과성)
- 본 논문 + Attention-XAI = 강화된 설명 가능성

***

**주요 인용**:
: NeurIPS 2020 원본 논문 및 성능 비교 [semanticscholar](https://www.semanticscholar.org/paper/235be0aa43cc1cf92b28926e0d34ef4b155053e4)



[50-68]: SHAP/LIME 발전 관련 최신 연구

<span style="display:none">[^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93][^1_94]</span>

<div align="center">⁂</div>

[^1_1]: https://arxiv.org/pdf/2503.23111.pdf

[^1_2]: https://arxiv.org/pdf/2409.00079.pdf

[^1_3]: https://www.nature.com/articles/s41598-024-72640-4

[^1_4]: https://www.ijcai.org/proceedings/2025/0636.pdf

[^1_5]: https://arxiv.org/pdf/2503.13304.pdf

[^1_6]: https://www.sciencedirect.com/science/article/abs/pii/S0957417425012977

[^1_7]: https://arxiv.org/pdf/2210.00604.pdf

[^1_8]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12465982/

[^1_9]: https://www.semanticscholar.org/paper/235be0aa43cc1cf92b28926e0d34ef4b155053e4

[^1_10]: https://proceedings.neurips.cc/paper/2020/file/36ac8e558ac7690b6f44e2cb5ef93322-Paper.pdf

[^1_11]: 2010.08973v1.pdf

[^1_12]: https://onlinelibrary.wiley.com/doi/10.1002/for.70051

[^1_13]: https://www.semanticscholar.org/paper/751d8117f53d4060dcefb184e2c7af598cc71a45

[^1_14]: https://lifescienceglobal.com/pms/index.php/ijsmr/article/view/10136

[^1_15]: https://www.frontiersin.org/articles/10.3389/feduc.2025.1689205/full

[^1_16]: https://ieeexplore.ieee.org/document/11112144/

[^1_17]: https://ijain.org/index.php/IJAIN/article/view/2091

[^1_18]: https://bmjopen.bmj.com/lookup/doi/10.1136/bmjopen-2025-098878

[^1_19]: https://www.tandfonline.com/doi/full/10.1080/10549811.2025.2589352

[^1_20]: https://jisem-journal.com/index.php/journal/article/view/8374

[^1_21]: https://arxiv.org/pdf/1710.05649.pdf

[^1_22]: http://arxiv.org/pdf/2502.03417.pdf

[^1_23]: http://arxiv.org/pdf/2410.23772.pdf

[^1_24]: http://arxiv.org/pdf/2401.15800.pdf

[^1_25]: https://arxiv.org/pdf/2308.00549.pdf

[^1_26]: https://peerj.com/articles/cs-310.pdf

[^1_27]: https://arxiv.org/html/2412.16188v1

[^1_28]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12252469/

[^1_29]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11735761/

[^1_30]: https://superagi.com/mastering-explainable-ai-in-2025-a-beginners-guide-to-transparent-and-interpretable-models/

[^1_31]: https://dasoldasol.github.io/deep learning/추천시스템/feature importance/feature selection/paper_review-recsys-fir/

[^1_32]: https://www.nature.com/articles/s41598-024-82583-5

[^1_33]: https://proceedings.neurips.cc/paper/2020/hash/36ac8e558ac7690b6f44e2cb5ef93322-Abstract.html

[^1_34]: https://harrieo.github.io/files/2024-ecir-xltr.pdf

[^1_35]: https://www.nitorinfotech.com/blog/explainable-ai-in-2025-navigating-trust-and-agency-in-a-dynamic-landscape/

[^1_36]: https://dl.acm.org/doi/10.5555/3495724.3496153

[^1_37]: https://onlinelibrary.wiley.com/doi/10.1002/ett.4970

[^1_38]: https://arxiv.org/html/2502.04695v1

[^1_39]: https://arxiv.org/html/2510.09586v1

[^1_40]: https://arxiv.org/pdf/2510.14669.pdf

[^1_41]: https://www.biorxiv.org/content/10.1101/2025.02.27.640573v1.full.pdf

[^1_42]: https://arxiv.org/html/2601.07235v1

[^1_43]: https://arxiv.org/html/2506.04788v1

[^1_44]: https://www.arxiv.org/pdf/2510.05120.pdf

[^1_45]: https://arxiv.org/html/2601.05095v1

[^1_46]: https://arxiv.org/pdf/2311.08760.pdf

[^1_47]: https://arxiv.org/pdf/2507.07344.pdf

[^1_48]: https://arxiv.org/html/2503.15237v2

[^1_49]: https://arxiv.org/html/2601.08401v1

[^1_50]: https://arxiv.org/html/2510.05120v1

[^1_51]: https://arxiv.org/pdf/2508.20130.pdf

[^1_52]: https://arxiv.org/html/2507.22659v1

[^1_53]: https://arxiv.org/html/2506.04133

[^1_54]: https://ieeexplore.ieee.org/document/10974933/

[^1_55]: https://ieeexplore.ieee.org/document/11315251/

[^1_56]: https://isjem.com/download/interpretable-deep-neural-networks-using-shap-and-lime-for-decision-making-in-smart-home-automation/

[^1_57]: https://dx.plos.org/10.1371/journal.pone.0326587

[^1_58]: https://www.indjcst.com/archives/paper-details?paperid=221\&papertitle=multi-task-deep-learning-with-shap-explainability-for-personalized-nutrition-prediction

[^1_59]: https://ieeexplore.ieee.org/document/11249263/

[^1_60]: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/sfw2/5455082

[^1_61]: https://link.springer.com/10.1007/s00170-025-16954-1

[^1_62]: https://arxiv.org/abs/2511.06282

[^1_63]: https://ieeexplore.ieee.org/document/11083552/

[^1_64]: https://arxiv.org/pdf/2211.14797.pdf

[^1_65]: http://arxiv.org/pdf/2405.10008.pdf

[^1_66]: https://arxiv.org/pdf/2403.08428.pdf

[^1_67]: http://arxiv.org/pdf/2405.00076.pdf

[^1_68]: https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202400304

[^1_69]: https://arxiv.org/pdf/2110.02484v1.pdf

[^1_70]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9464364/

[^1_71]: https://drpress.org/ojs/index.php/ajst/article/view/23543

[^1_72]: https://www.nature.com/articles/s41598-025-19545-y

[^1_73]: https://www.sciencedirect.com/science/article/abs/pii/S092523122100477X

[^1_74]: https://www.sciencedirect.com/science/article/abs/pii/S1389128622002754

[^1_75]: https://blog.naver.com/kcc_press/223792201210

[^1_76]: https://www.sciencedirect.com/science/article/abs/pii/S0022169423001427

[^1_77]: https://www.sciencedirect.com/science/article/pii/S0952197625014599

[^1_78]: https://openreview.net/forum?id=omzijInU1T

[^1_79]: https://dl.acm.org/doi/10.24963/ijcai.2025/636

[^1_80]: https://www.nature.com/articles/s41598-025-98763-w

[^1_81]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12680242/

[^1_82]: https://arxiv.org/html/2512.07241v1

[^1_83]: https://www.arxiv.org/pdf/2502.17361v1.pdf

[^1_84]: https://arxiv.org/html/2512.10913v1

[^1_85]: https://www.arxiv.org/pdf/2601.03181.pdf

[^1_86]: https://arxiv.org/pdf/2504.20571.pdf

[^1_87]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0332779

[^1_88]: https://arxiv.org/html/2406.10322v5

[^1_89]: https://arxiv.org/html/2601.07235v2

[^1_90]: https://www.arxiv.org/pdf/2512.07241.pdf

[^1_91]: https://arxiv.org/pdf/2412.19985.pdf

[^1_92]: https://arxiv.org/html/2507.20202v1

[^1_93]: https://arxiv.org/pdf/2407.03257.pdf

[^1_94]: https://www.semanticscholar.org/paper/AFS:-An-Attention-based-mechanism-for-Supervised-Gui-Ge/59712d0a0cc255e545ea27b2f4dc733ea709756a

