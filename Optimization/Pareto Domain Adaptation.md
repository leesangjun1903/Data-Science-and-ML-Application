# Pareto Domain Adaptation (ParetoDA) 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

기존 Domain Adaptation(DA) 방법들은 소스 분류 손실 $\mathcal{L}_S$와 도메인 정렬 손실 $\mathcal{L}_D$를 선형 결합하여 최적화하지만, **두 목적함수의 그래디언트 방향이 도메인 이동(domain shift)으로 인해 충돌**할 수 있다. 이 경우 선형 최적화는 하나의 목적함수를 희생하는 제한된 해(restricted solution)만 탐색하며, 비볼록(non-convex) Pareto 전선의 일부에만 접근 가능하다. 이를 해결하기 위해 **그래디언트 기반 다목적 최적화 관점**에서 DA를 재설계한 ParetoDA를 제안한다.

### 주요 기여

| 기여 항목 | 내용 |
|----------|------|
| 문제 재정의 | 기존 선형 결합 방식의 한계를 Pareto 최적 관점에서 분석 |
| TCM 손실 | 타겟 분류를 모방하는 대리 손실(surrogate loss) 설계 |
| 예측 정제 메커니즘 | 베이즈 정리를 활용한 타겟 예측 정확도 향상 |
| 동적 선호 메커니즘 | held-out 타겟 데이터의 그래디언트로 동적 최적화 유도 |
| 이론적 보장 | held-out 데이터 과적합 불가 이론 증명 |
| 플러그인 설계 | 기존 DA 방법에 범용적으로 적용 가능 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존 DA의 표준 최적화 공식은 아래와 같다.

**불일치 기반(Discrepancy-based):**

$$\min_{\theta, \phi_c} \mathcal{L}_S + \mathcal{L}_D \tag{1}$$

**적대적 학습 기반(Adversarial-based):**

$$\min_{\theta, \phi_c} \mathcal{L}_S + \mathcal{L}_D, \quad \max_{\phi_d} \mathcal{L}_D \tag{2}$$

이 방식의 문제점:

1. **그래디언트 충돌**: $\mathcal{L}_S$와 $\mathcal{L}_D$의 그래디언트가 반대 방향으로 작용할 때, 선형 결합은 한 목적함수를 손상시키며 최적화됨
2. **비볼록 Pareto 전선 접근 불가**: 선형 가중치 결합은 Pareto 전선의 볼록 부분에만 접근 가능 (Boyd & Vandenberghe, 2014)
3. **목표 불일치**: $\mathcal{L}_S$나 $\mathcal{L}_D$ 모두 실제 목표인 타겟 분류 손실 $\mathcal{L}_T^*$에 직접 대응하지 않음
4. **하이퍼파라미터 민감성**: 가중치 $\lambda$ 변화에 따라 해가 Pareto 전선에서 크게 이동

---

### 2.2 제안 방법 및 수식

#### 2.2.1 타겟 예측 정제 메커니즘 (Target-Prediction Refining Mechanism)

타겟 도메인의 레이블이 없으므로, **베이즈 정리**를 이용해 도메인 레이블을 조건부 정보로 활용하여 예측 확률을 정제한다.

$$p(y=k \mid z=d, \boldsymbol{x}, \boldsymbol{\theta}, \boldsymbol{\phi}_c) = \frac{p(z=d \mid \boldsymbol{x}, \boldsymbol{v}_k)\, p(y=k \mid \boldsymbol{x}, \boldsymbol{\theta}, \boldsymbol{\phi}_c)}{\sum_{k'} p(z=d \mid \boldsymbol{x}, \boldsymbol{v}_{k'})\, p(y=k' \mid \boldsymbol{x}, \boldsymbol{\theta}, \boldsymbol{\phi}_c)} = \rho_{k|d} \tag{3}$$

- $z$: 도메인 레이블 ($d=0$: 소스, $d=1$: 타겟)
- $\boldsymbol{v}_k$: $k$번째 클래스별 도메인 판별기 파라미터
- $\rho_{k|d}$: 정제된 예측 확률

클래스별 판별기 학습 목적함수:

$$\min_{\boldsymbol{v}_1, \ldots, \boldsymbol{v}_K} -\sum_{k=1}^{K} \sum_{d=0}^{1} s_{k|d}\, \mathcal{I}(z=d) \log p(z=d \mid \boldsymbol{x}, \boldsymbol{v}_k) \tag{4}$$

여기서 $s_{k|d}$는 소스 도메인에선 하드 레이블 $\mathcal{I}(y=k)$, 타겟 도메인에선 소프트 레이블 $\rho_{k|d}$.

#### 2.2.2 타겟 분류 모방(TCM) 손실

정제된 예측 $\rho_{k|d}$를 활용하여 정보 최대화(Information Maximization) 손실로 타겟 분류를 모방:

$$\mathcal{L}_T = \sum_{k=1}^{K} \hat{\rho}_{k|1} \log \hat{\rho}_{k|1} - \mathbb{E}_{\boldsymbol{x} \in D_t} \sum_{k=1}^{K} \rho_{k|1} \log \rho_{k|1} \tag{5}$$

- 첫 번째 항: 전역 다양성(global diversity) 극대화 (예측의 균등 분포 유도)
- 두 번째 항: 개별 확실성(individual certainty) 극대화 (조건부 엔트로피 최소화)
- $\hat{\rho}\_{k|1} = \mathbb{E}\_{\boldsymbol{x} \in D_t} \rho_{k|1}$: 타겟 도메인 전체에 대한 $k$번째 클래스 평균 확률

#### 2.2.3 동적 선호 메커니즘 (Dynamic Preference Mechanism)

매 최적화 스텝에서 업데이트 방향 $\boldsymbol{d}$를 세 손실의 그래디언트의 볼록 결합으로 구성:

$$\boldsymbol{d} = G\boldsymbol{w}, \quad G = [\nabla_{\boldsymbol{\theta}} \mathcal{L}_S,\; \nabla_{\boldsymbol{\theta}} \mathcal{L}_D,\; \nabla_{\boldsymbol{\theta}} \mathcal{L}_T], \quad \boldsymbol{w} \in \mathcal{S}^m$$

held-out 타겟 데이터에서의 TCM 손실 $\mathcal{L}_{Val}$의 그래디언트를 동적 가이던스로 사용:

$$\hat{\boldsymbol{g}}_v = \nabla_{\boldsymbol{\theta}} \mathcal{L}_{Val}$$

최적 가중치 $\boldsymbol{w}^*$를 선형 프로그래밍(LP)으로 탐색:

$$\boldsymbol{w}^* = \arg\max_{\boldsymbol{w} \in \mathcal{S}^m} (G\boldsymbol{w})^T \left(\mathcal{I}(\mathcal{L}_{Val} > 0)\hat{\boldsymbol{g}}_v + \mathcal{I}(\mathcal{L}_{Val}=0)G\mathbf{1}/m\right)$$

$$\text{s.t.} \quad (G\boldsymbol{w})^T \boldsymbol{g}_j \geq \mathcal{I}(J \neq \emptyset)(\hat{\boldsymbol{g}}_v^T \boldsymbol{g}_j), \quad \forall j \in \bar{J} - J^* \tag{6}$$

$$\quad\quad (G\boldsymbol{w})^T \boldsymbol{g}_j \geq 0, \quad \forall j \in J^*$$

- $J = \{j \mid \hat{\boldsymbol{g}}_v^T \boldsymbol{g}_j > 0\}$: 가이던스와 일치하는 그래디언트 집합
- $\bar{J} = \{j \mid \hat{\boldsymbol{g}}_v^T \boldsymbol{g}_j \leq 0\}$: 가이던스와 불일치하는 그래디언트 집합
- $J^\* = \{j \mid \hat{\boldsymbol{g}}\_v^T \boldsymbol{g}\_j = \max_{j'} \hat{\boldsymbol{g}}\_v^T \boldsymbol{g}_{j'}\}$

#### 2.2.4 과적합 방지 이론적 보장 (Theorem 1)

**Theorem 1**: $\boldsymbol{w}^\*$가 식 (6)의 해이고 $\boldsymbol{d}^* = G\boldsymbol{w}^*$일 때:

- $\mathcal{L}_{Val} = 0$이면: $(\boldsymbol{d}^*)^T \boldsymbol{g}_j \geq 0, \; \forall j \in \{1,2,3\}$ (순수 하강 모드)
- $\mathcal{L}_{Val} > 0$이고 $\gamma^* = (\boldsymbol{d}^*)^T \hat{\boldsymbol{g}}_v > 0$이면: 가이던스 하강 모드 활성화
- $\mathcal{L}_{Val} > 0$이고 $\gamma^* \leq 0$이면: 여전히 $(\boldsymbol{d}^*)^T \boldsymbol{g}_j \geq 0$ 유지

즉, held-out 데이터는 **안내(guide)** 역할만 하며 **과적합되지 않음**이 이론적으로 보장된다.

#### 2.2.5 Ben-David 이론과의 연결

Ben-David et al.의 타겟 오류 상한:

$$\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2} d_{H\Delta H}(D_s, D_t) + \lambda \tag{9}$$

ParetoDA는 그래디언트 기반 협력 최적화로 $\epsilon_S(h)$와 $d_{H\Delta H}(D_s, D_t)$를 동시에 감소시키고, TCM 손실로 결합 오류 $\lambda$까지 줄임으로써 **상한 전체를 효과적으로 감소**시킨다.

---

### 2.3 모델 구조

```
입력 데이터 (소스 + 타겟)
        ↓
[공유 특징 추출기 θ]
   ↙         ↘
[분류기 φ_c]  [도메인 판별기 φ_d]  [클래스별 판별기 v_1,...,v_K]
   ↓               ↓                        ↓
  L_S              L_D              정제된 예측 ρ_{k|d}
                                           ↓
                                        L_T (TCM 손실)
                                           ↓
                              held-out 타겟에서 L_Val 계산
                                           ↓
                              LP로 최적 w* 탐색 → d* = Gw*
                                           ↓
                              θ 업데이트: θ^{t+1} = θ^t - η d*
```

---

### 2.4 성능 향상

| 데이터셋 | 기본 방법 | 기본 성능 | +ParetoDA | 향상폭 |
|---------|---------|---------|-----------|-------|
| Office-31 | DANN | 82.2% | **90.2%** | +8.0% |
| Office-31 | CDAN | 87.7% | **90.4%** | +2.7% |
| Office-31 | MDD | 88.9% | **90.1%** | +1.2% |
| Office-Home | DANN | 57.6% | **69.4%** | +11.8% |
| Office-Home | CDAN | 65.8% | **70.6%** | +4.8% |
| VisDA-2017 | DANN | 57.4% | **82.4%** | +25.0% |
| VisDA-2017 | CDAN | 73.9% | **83.2%** | +9.3% |
| GTA5→Cityscapes | AdvEnt | 43.8 mIoU | **46.1 mIoU** | +2.3 |
| GTA5→Cityscapes | CBST | 45.2 mIoU | **48.1 mIoU** | +2.9 |

---

### 2.5 한계점

1. **전통적 DA 설정에 집중**: 논문 스스로 "전통적 DA 설정에 집중하며 다른 DA 변형으로의 일반화는 추가 탐색 필요"라고 명시
2. **held-out 데이터 분할 필요**: 타겟 데이터의 10%를 검증용으로 분리하여 학습 데이터 감소
3. **계산 오버헤드**: LP 풀이 및 다중 그래디언트 계산으로 추가 연산 발생 (단, $d \gg m$이므로 전체 복잡도는 $O(d)$ )
4. **클래스별 판별기**: $K$개의 추가 MLP 네트워크 필요로 파라미터 증가
5. **자기 학습(self-training) 기반 한계**: TCM 손실이 타겟 예측에 의존하므로 초기 예측 품질이 낮을 경우 성능 저하 가능

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상의 세 가지 메커니즘

#### (1) Pareto 최적화를 통한 동시 최적화

기존 방법은 $\mathcal{L}_S$를 낮추는 과정에서 $\mathcal{L}_D$가 증가하거나 그 반대가 발생할 수 있다. ParetoDA는 모든 목적함수가 동시에 감소하거나 적어도 손상되지 않는 방향 $\boldsymbol{d}^*$를 찾음으로써:

$$\epsilon_T(h) \leq \underbrace{\epsilon_S(h)}_{\mathcal{L}_S \downarrow} + \frac{1}{2}\underbrace{d_{H\Delta H}(D_s, D_t)}_{\mathcal{L}_D \downarrow} + \underbrace{\lambda}_{\mathcal{L}_T \downarrow}$$

**세 항을 동시에 감소**시켜 일반화 상한을 효과적으로 낮춘다.

#### (2) 베이즈 예측 정제를 통한 의사 레이블 품질 향상

기존 방법: $p(y=k \mid \boldsymbol{x}, \boldsymbol{\theta}, \boldsymbol{\phi}_c)$ (도메인 정보 미활용)

ParetoDA: $\rho_{k|1} = p(y=k \mid z=1, \boldsymbol{x}, \boldsymbol{\theta})$ (도메인 레이블을 조건부 정보로 활용)

이는 타겟 도메인 전용 예측 확률을 정확히 계산하여 의사 레이블(pseudo-label)의 품질을 향상시키고, 이를 통해 TCM 손실의 신뢰도를 높인다.

#### (3) Held-out 데이터를 통한 일반화 유도

- 학습 데이터가 아닌 **held-out 검증 세트**에서 $\mathcal{L}_{Val}$을 계산하여 최적화 방향 유도
- 이는 **일반화 성능을 직접적으로 반영**하는 신호로 작용
- Theorem 1에 의해 과적합 없이 안내 역할만 수행

#### (4) 객관적 척도 민감도 강인성

$\lambda_0, \lambda_1 \in \{0.1, 0.5, 1.0, 1.5\}$를 변화시킨 민감도 분석(Fig. 3(b))에서 ParetoDA는 목적함수 스케일 변화에 강인함을 보였으며, 이는 다양한 데이터셋과 방법에 걸친 **범용적 일반화 가능성**을 시사한다.

#### (5) 다양한 백본에서의 일반화

| 백본 | DANN | +ParetoDA |
|-----|------|-----------|
| ResNet-50 | 67.4% | **76.3%** |
| ResNet-101 | 74.5% | **77.5%** |
| ResNet-152 | 76.1% | **79.0%** |

더 깊은 네트워크에서도 일관된 성능 향상으로 **아키텍처 독립적 일반화**를 확인.

---

## 4. 미래 연구에 미치는 영향과 고려할 점

### 4.1 미래 연구에 미치는 영향

#### (1) 다목적 최적화 패러다임의 확장

ParetoDA는 DA 문제에서 **다목적 그래디언트 최적화를 처음으로 체계적으로 적용**한 연구로, 유사한 목적함수 충돌이 발생하는 다음 분야로 확장 가능:

- **반지도 학습(Semi-supervised Learning)**: 지도 손실과 일관성 손실 간 충돌
- **연속 학습(Continual Learning)**: 새 태스크와 이전 태스크 손실 간 충돌
- **멀티태스크 학습(Multi-task Learning)**: 태스크 간 그래디언트 충돌
- **페더레이티드 러닝(Federated Learning)**: 클라이언트 간 목표 충돌

#### (2) 소스 없는 도메인 적응(Source-free DA) 연구 방향

ParetoDA는 소스 데이터를 활용하지만, TCM 손실의 아이디어는 소스 데이터 없이 타겟 도메인만으로 최적화하는 **Source-free DA** 연구에도 영향을 줄 수 있다.

#### (3) 테스트 시간 적응(Test-Time Adaptation, TTA) 연구

동적 선호 메커니즘의 held-out 데이터 활용 아이디어는 **TTA** 연구에서 온라인 최적화 방향 결정에 응용 가능하다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | ParetoDA와의 관계 | 주요 차이점 |
|-----|------|-----------------|-----------|
| **MetaAlign (Wei et al., 2021, CoRR)** | 메타 학습으로 도메인 정렬과 분류 조정 | 비교 기준선 (논문 내 포함) | 메타 학습 기반 vs. Pareto 그래디언트 기반 |
| **SHOT (Liang et al., ICML 2020)** | 소스 없는 DA, 정보 최대화 | TCM 손실과 유사한 IM 손실 사용 | Source-free 설정에서의 적용 |
| **MIC (Hoyer et al., CVPR 2023)** | 마스킹 기반 이미지 일관성 DA | 아키텍처 수준 혁신 | 최적화 스킴 고려 없음 |
| **T3A (Iwasawa & Matsuo, NeurIPS 2021)** | 테스트 시간 분류기 조정 | TTA 관점의 유사 목표 | 추론 시간에만 적용 |
| **CAiDA (Dong et al., 2022)** | 클래스 인식 적응 | 클래스별 정렬 | Pareto 최적화 미포함 |

#### 특이 관찰: ParetoDA 이후 연구 트렌드

1. **그래디언트 기반 다목적 최적화의 DA 적용 증가**: ParetoDA를 기점으로 MGDA, EPO 등 다목적 최적화를 DA에 적용하는 연구가 증가
2. **의사 레이블 정제**: 베이즈 추론 기반 정제 아이디어가 후속 연구에서 확장
3. **검증 기반 최적화 유도**: 레이블 없는 검증 세트를 활용한 최적화 방향 결정은 TTA 연구로 연결

---

### 4.3 앞으로 연구 시 고려할 점

#### (1) 더 다양한 DA 변형으로의 확장
- **부분 도메인 적응(Partial DA)**: 타겟이 소스 클래스의 부분 집합인 경우
- **열린 집합 도메인 적응(Open-set DA)**: 타겟에 미지 클래스 포함
- **다중 소스 DA(Multi-source DA)**: 여러 소스 도메인이 존재하는 경우

#### (2) 초기 예측 품질 의존성 해결
TCM 손실은 초기 예측 품질에 민감하므로, **웜업(warm-up) 전략**이나 **커리큘럼 학습(curriculum learning)**을 통한 점진적 정제 메커니즘 연구가 필요하다.

#### (3) 다중 도메인 및 지속적 도메인 이동
단일 소스-타겟 쌍을 넘어, **연속적으로 변화하는 도메인**에서 Pareto 최적 해가 어떻게 변화하는지 동적 추적 연구가 필요하다.

#### (4) 이론적 수렴 보장 강화
현재 Theorem 1은 held-out 과적합 방지만 보장하며, **전역 Pareto 최적해로의 수렴 속도**에 대한 이론적 분석이 추가로 필요하다.

#### (5) 자기 지도 학습(Self-supervised Learning)과의 결합
사전 학습된 대형 모델(ViT, CLIP 등)을 활용할 때 Pareto 최적화가 어떻게 작동하는지 연구가 필요하다. 특히 **프롬프트 튜닝(prompt tuning)** 기반 DA에서의 적용 가능성을 탐색할 필요가 있다.

#### (6) held-out 분할 비율 최적화
현재 고정된 90%/10% 분할이 최적인지, **적응적 분할 비율**이 성능에 미치는 영향을 체계적으로 분석할 필요가 있다.

---

## 참고 자료 (출처)

1. **주 논문**: Fangrui Lv, Jian Liang, et al., *"Pareto Domain Adaptation"*, NeurIPS 2021. GitHub: https://github.com/BIT-DA/ParetoDA

2. **인용 논문 (논문 내 참조)**:
   - Ben-David et al., *"A Theory of Learning from Different Domains"*, MLJ 2010
   - Boyd & Vandenberghe, *"Convex Optimization"*, Cambridge University Press, 2014
   - Ganin & Lempitsky, *"Unsupervised Domain Adaptation by Backpropagation"* (DANN), ICML 2015
   - Long et al., *"Conditional Adversarial Domain Adaptation"* (CDAN), NeurIPS 2018
   - Mahapatra & Rajan, *"Multi-task Learning with User Preferences: Gradient Descent with Controlled Ascent in Pareto Optimization"* (EPO), ICML 2020
   - Lin et al., *"Pareto Multi-Task Learning"* (PMTL), NeurIPS 2019
   - Désidéri, *"Multiple-Gradient Descent Algorithm (MGDA) for Multiobjective Optimization"*, Comptes Rendus Mathematique, 2012
   - Hu et al., *"Learning Discrete Representations via Information Maximizing Self-Augmented Training"* (IM Loss), ICML 2017
   - Zhang et al., *"Bridging Theory and Algorithm for Domain Adaptation"* (MDD), ICML 2019
