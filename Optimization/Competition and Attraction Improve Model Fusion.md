
# Competition and Attraction Improve Model Fusion

> **논문 정보**
> - **제목**: Competition and Attraction Improve Model Fusion
> - **저자**: João P. Abrantes, Robert Tjarko Lange, Yujin Tang (Sakana AI)
> - **발표**: GECCO'25 (Genetic and Evolutionary Computation Conference 2025, Málaga, Spain), **Best Paper Runner-up**
> - **arXiv**: [arxiv.org/abs/2508.16204](https://arxiv.org/abs/2508.16204)
> - **코드**: [github.com/SakanaAI/natural_niches](https://github.com/SakanaAI/natural_niches)
> - **ACM DOI**: 10.1145/3712256.3726329

---

## 1. 📌 핵심 주장과 주요 기여 (요약)

Model merging은 여러 머신러닝 모델의 전문 지식을 단일 모델에 통합하는 강력한 기술이다. 그러나 기존 방법들은 모델 파라미터를 고정된 그룹으로 수동 분할해야 하며, 이는 잠재적 조합 탐색을 제한하고 성능을 저하시킨다.

이를 극복하기 위해 이 논문은 **M2N2 (Model Merging of Natural Niches)** 를 제안한다. M2N2는 진화 알고리즘으로, 세 가지 핵심 특성을 갖는다: (1) 더 넓은 파라미터 조합을 탐색하기 위한 병합 경계의 동적 조정, (2) 자연계 자원 경쟁에서 영감을 받은 다양성 보존 메커니즘, (3) 가장 유망한 모델 쌍을 식별하기 위한 휴리스틱 기반 Attraction 메트릭.

### 주요 기여 3가지

| 기여 | 설명 |
|------|------|
| 🌿 **Evolving Split-Points** | 고정 레이어 경계 대신 동적 분할점 탐색 |
| 🐠 **Competition for Diversity** | 자원 경쟁을 통한 다양한 전문화 모델 풀 유지 |
| 💏 **Attraction-based Mate Selection** | 상호 보완적 강점을 가진 모델 쌍 선택 |

이 결과들은 모델 머징이 처음으로 완전히 scratch에서 모델을 진화시키는 데 사용될 수 있음을 보여준다.

---

## 2. 🔬 상세 분석

### 2-1. 해결하고자 하는 문제

기존 Model merging은 초기에 시드 모델들을 결합하기 위한 계수를 수동으로 조정하는 방식에 의존했으며, 이는 직관에 의존하고 상당한 시행착오가 필요했다. 최근 진화 알고리즘이 최적 계수를 자동으로 탐색함으로써 이 과정을 간소화했지만, 여전히 하나의 수동 단계가 남아 있다: 개발자들은 병합 전에 모델 파라미터를 고정된 집합으로 그룹화해야 하며, 이는 잠재적 조합의 탐색 공간을 제한한다.

기존 모델 머징은 파라미터 그룹화를 아키텍처 레이어 수준에서 고정하고, 해당 블록들 사이의 보간 계수만을 탐색한다. 이 제약은 가능한 파라미터 재조합의 범위를 제한하며, 특히 유용한 "재조합 구조"가 레이어 경계를 가로질러 존재하는 경우 병합 효율을 저하시킬 수 있다.

---

### 2-2. 제안 방법 및 수식

#### 🔹 (1) Evolving Split-Points (동적 병합 경계)

M2N2는 사전 정의된 정적 경계(예: 고정 레이어) 대신 병합을 위한 "split-points"를 동적으로 진화시킨다. 이는 전체 염색체 대신 가변 길이의 DNA 단편을 교환하는 것처럼, 파라미터 조합에 대한 훨씬 더 유연하고 강력한 탐색을 가능하게 한다.

구체적으로 고정 레이어로 파라미터를 그룹화하는 대신, 유연한 "split point"와 "mixing ratio"를 사용하여 모델을 분할·결합한다. 예를 들어, 알고리즘이 Model A의 특정 레이어에서 파라미터의 30%와 Model B의 동일 레이어 파라미터 70%를 병합할 수 있다.

M2N2는 유연한 split point를 사용하여 한 번에 두 모델을 반복적으로 병합하며, 진화하는 모델 아카이브를 유지한다. 세대 수가 증가함에 따라 M2N2는 점진적으로 더 넓은 경계 및 계수 집합을 탐색하여 유익한 경우 점점 더 복잡한 조합을 가능하게 한다.

---

#### 🔹 (2) Competition (자원 경쟁을 통한 다양성 보존)

효과적인 모델 머징은 고성능 모델들의 다양한 집단 유지에 의존한다. 수작업 다양성 메트릭에 의존하는 대신, M2N2는 진화 생물학의 암묵적 fitness sharing에서 영감을 받은 자원 경쟁 메커니즘을 사용한다. 각 데이터 포인트는 제한된 자원으로 취급되며, 모델이 데이터 포인트에서 도출하는 fitness는 집단 내 상대적 성능에 비례한다. 이 접근법은 자연스럽게 표현 부족 niche에서 뛰어난 모델의 발견과 보존을 장려하며, crossover 연산의 다양성 감소 효과에 대응한다.

**Competition을 통한 수정된 최적화 목표 (수식):**

$$\theta^* = \arg\max_{\theta} \sum_{j=1}^{N} \frac{s(x_j \mid \theta)}{z_j + \epsilon} c_j$$

$$z_j = \sum_{k=1}^{P} s(x_j \mid \theta_k)$$

$z_j$는 집단 내 모든 모델의 해당 데이터 포인트에 대한 점수의 합이고, $\epsilon$은 작은 상수이다. 많은 모델이 잘 수행하는 데이터 포인트는 각 모델의 fitness에 덜 기여하므로, 특화 및 다양성을 촉진한다 — 모델들은 서로 다른 "자원"을 추구해야 한다.

여기서:
- $s(x_j \mid \theta)$: 파라미터 $\theta$를 가진 모델의 데이터 포인트 $x_j$에 대한 점수
- $c_j$: 데이터 포인트 $x_j$의 용량(capacity)
- $P$: 아카이브 크기
- $\epsilon$: 영 나눗셈 방지를 위한 작은 수

---

#### 🔹 (3) Attraction (보완적 강점 기반 부모 쌍 선택)

Crossover 연산은, 특히 대형 모델의 경우 계산 비용이 많이 든다. M2N2는 상호 보완적 강점을 가진 부모 쌍을 선택하는 Attraction 휴리스틱을 도입하여 유익한 병합의 가능성을 최대화한다.

**Attraction Score 수식:**

$$g(\theta_A, \theta_B) = \sum_{j=1}^{N} \frac{c_j}{z_j + \epsilon} \cdot \max\!\left(s(x_j \mid \theta_B) - s(x_j \mid \theta_A),\ 0\right)$$

첫 번째 부모가 선택된 후(예: fitness 가중 룰렛 선택), 두 번째 부모는 "attraction" 점수를 기반으로 선택된다. 이 점수는 병합을 위한 상호 보완적 전문성을 우선시하며, 이 메커니즘은 전문화된 기술을 결합할 가능성이 높은 병합을 체계적으로 구성한다.

이 수식은 **모델 A가 약한 데이터 포인트에서 모델 B가 강할수록** 높은 attraction 점수를 부여한다. 즉, "내가 못하는 것을 잘하는" 파트너를 선택하는 자연 선택의 짝짓기를 모방한다.

---

### 2-3. 모델 구조

프로세스는 시드 모델들의 "아카이브"로 시작한다. 각 단계에서 M2N2는 아카이브에서 두 모델을 선택하고, mixing ratio와 split point를 결정하여 병합한다. 결과 모델이 우수하면 더 약한 모델을 대체하며 아카이브에 추가된다.

**알고리즘 흐름 요약:**

```
초기화: 시드 모델 아카이브 구성
반복 (각 세대):
  1. 첫 번째 부모 θ_A 선택 (fitness 기반 룰렛)
  2. 두 번째 부모 θ_B 선택 (attraction score g(θ_A, θ_B) 기반)
  3. 랜덤 split-point s와 mixing ratio α 샘플링
  4. SLERP 등으로 θ_child 생성
  5. θ_child 평가 → fitness 우수 시 아카이브에 추가
  6. split-point 범위 점진적 확장 (Curriculum)
```

M2N2는 split-point(예: "Model A의 전반부 + Model B의 후반부")를 랜덤 선택하고 SLERP로 보간한다. 세대를 거치면서 필요에 따라 복잡성을 점진적으로 확장한다.

---

### 2-4. 성능 향상

실험 결과는 처음으로 모델 머징이 완전히 scratch에서 모델을 진화시키는 데 사용될 수 있음을 보여준다. 구체적으로 M2N2를 MNIST 분류기 진화에 적용하여 CMA-ES에 필적하는 성능을 달성하면서 계산 효율이 더 우수했다.

실험 결과 M2N2가 가장 높은 점수를 달성했다. Attraction과 split-point 기술 모두 중요한 역할을 하며, split-point가 약간 더 중요하다. 동일한 횟수의 평가와 동일한 SLERP 병합 방법을 사용하여 비교했다. Math와 Agentic 기술을 결합할 때 CMA-ES는 낮은 점수를 기록했는데, 이는 최적화 과정에 병합 경계를 포함할 필요성을 강조한다.

**벤치마크 비교 (LLM 실험):**

M2N2를 수학 전문 LLM과 에이전틱 전문 LLM을 병합하는 데 사용했을 때, 수학(GSM8k)과 웹 쇼핑(WebShop) 작업 모두에서 뛰어난 성능을 보이며 다른 방법들을 크게 능가했다. 유연한 split-point가 이 결과에 결정적이었다.

**이미지 생성 실험:**

텍스트-이미지 모델에 M2N2를 적용했을 때, 일본어 프롬프트에만 적응시킨 여러 모델을 병합한 결과 모델이 일본어에서 향상될 뿐만 아니라 강력한 영어 능력도 유지했다 — 이는 catastrophic forgetting으로 고통받을 수 있는 fine-tuning 대비 핵심 이점이다.

---

### 2-5. 한계점

GA는 competition이 없는 극단적 경우로, M2N2에서 competition을 점진적으로 줄이면 더 나쁜 해에 더 일찍 수렴한다는 것을 관찰했다.

논문 및 관련 분석에서 파악된 주요 한계:

1. **계산 비용**: mate selection은 유전 알고리즘의 탐구가 부족한 측면으로, crossover(병합)의 계산 비용이 증가할수록 점점 더 중요해진다.
2. **아키텍처 동질성**: 부모 모델의 상업적 라이선스, 아키텍처 호환성, 크기에 대한 고려가 필요하다.
3. **Fitness 함수 의존성**: 피트니스 함수에 의해 명시적으로 최적화되지 않은 중요한 모델 특성을 보존하는 견고성과 다양성을 보여주었다는 점에서 fitness 함수 설계가 결과에 큰 영향을 미침.
4. **스케일 확장성**: 매우 대형 모델(70B+)에서의 반복적 병합 비용 문제.

---

## 3. 🌟 모델의 일반화 성능 향상 가능성

이 논문에서 일반화 성능 향상과 관련하여 가장 중요한 발견은 다음과 같다.

### 3-1. Catastrophic Forgetting 방지

텍스트-이미지 모델에 M2N2를 적용했을 때, 여러 모델을 일본어 프롬프트에만 적응시켜 병합했다. 결과 모델은 일본어에서 향상될 뿐만 아니라 강력한 영어 능력도 유지했다 — 이는 catastrophic forgetting으로 고통받을 수 있는 fine-tuning 대비 핵심 이점이다.

### 3-2. Fitness 함수 외부 능력 보존

M2N2는 fitness 함수에 의해 명시적으로 최적화되지 않은 중요한 모델 능력들을 보존하며, 이는 견고성과 범용성을 강조한다. 이는 일반화 성능의 핵심 지표로, 특정 작업에 과적합하지 않고 더 넓은 능력을 유지한다는 의미이다.

### 3-3. 다양성 기반 일반화

진화적 모델 병합 중 행동적·기능적 다양성 유지는 두 가지 이유로 중요하다: (1) 다양한 집단은 상호 보완적 강점을 통합할 수 있는 고성능 병합 모델 형성을 가능하게 하고, (2) 최대 fitness의 복제본들만 병합하면 빠르게 정체되고 niche 특화가 상실된다. M2N2는 생태적 자원 경쟁에서 영감을 받은 다양성 보존 목적 함수를 통합한다.

### 3-4. 다중 도메인 일반주의 모델 가능성

M2N2 접근법은 catastrophic forgetting, 높은 계산 오버헤드, 잠재적 태스크 특화 기술 손실 없이 전문화된 전문성을 통합하는 범용 모델 구축을 위한 확장 가능한 경로를 제공한다는 것이 그럴듯한 함의이다. 그 진화적 특성, 즉 exploitation(fitness 기반 파트너 선택)과 exploration(동적 경계 및 자원 경쟁)의 균형은 고도로 유능하고 다목적인 AI 시스템 합성의 열린 과제를 해결하는 데 적합하다.

---

## 4. 🚀 앞으로의 연구에 미치는 영향 및 고려사항

### 4-1. 연구에 미치는 영향

**① 진화 알고리즘 + 모델 병합의 새로운 패러다임**

M2N2는 competition, attraction, split point를 사용한 모델 fusion의 세 가지 핵심 컴포넌트를 도입하는 새로운 진화적 접근법이다. Ablation 연구를 통해 이 컴포넌트들이 모델 머징을 크게 향상시키고 crossover 연산을 사용하는 다른 진화 알고리즘에도 적용될 수 있음을 보여준다.

**② 오픈소스 생태계 활용 가능성**

오픈소스 생성 모델의 확산으로 특정 도메인이나 작업에 특화된 방대한 생태계가 만들어졌다. 모델 머징은 원본 훈련 데이터 접근이 제한되거나 이질적 목표로 훈련된 모델들을 결합할 때 전통적인 fine-tuning의 실용적 대안으로 부상했다.

**③ Scratch부터의 진화 가능성 증명**

M2N2는 랜덤 초기화에서 성능 있는 분류기를 진화시키는 최초의 모델 병합 접근법으로, CMA-ES에 필적하는 테스트 정확도를 달성하지만 계산 비용은 크게 절감된다. M2N2는 다양성 보존이 없는 GA와 대조적으로 훈련 전반에 걸쳐 높은 훈련 커버리지와 집단 엔트로피를 유지한다.

**④ 집단 지능 AI 생태계로의 방향 제시**

이 자연 영감 접근법은 집단 지능을 기반으로 AI의 새로운 기반을 찾으려는 Sakana AI의 사명의 핵심이다. 단일 모놀리식 모델을 확장하는 대신, 다양한 특화 모델들의 생태계가 공진화하고 협력하며 결합하여 더 적응적이고 견고하며 창의적인 AI로 이어지는 미래를 구상한다.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 핵심 아이디어 | 한계 | M2N2와의 차이 |
|------|--------------|------|--------------|
| **Weight Averaging** (2022) | 모델 가중치의 단순 평균 | 다양한 목표로 훈련된 모델 간 성능 저하 | M2N2는 동적 경계와 경쟁 메커니즘으로 극복 |
| **Task Arithmetic** (2023) | fine-tuned 가중치 빼기로 task vector 생성, 산술 연산 | 파라미터 간섭 문제 발생 | M2N2는 파라미터 공간 구조 자체를 진화 |
| **TIES-Merging** (2023) | 파라미터 부호 충돌 해소 및 중요 파라미터 보존 | 전역 고정 하이퍼파라미터 의존 | M2N2는 동적 탐색으로 경계 문제 없음 |
| **DARE** (2024) | 델타 파라미터 랜덤 드롭 및 rescaling | 고정 확률 드롭의 한계 | M2N2는 경쟁 기반으로 자연스러운 다양성 보존 |
| **Evolutionary Model Merging** (Akiba et al., 2024, *Nature MI*) | CMA-ES로 최적 병합 레시피 자동 탐색 | **고정 파라미터 그룹화** 필요 | M2N2는 이 고정 경계 문제를 split-point로 해결 |
| **M2N2** (2025, **본 논문**) | Competition + Attraction + Dynamic Split-Point | 아키텍처 호환성, 대형 모델 확장 비용 | 세 메커니즘의 시너지로 SOTA 달성 |

Task Arithmetic는 fine-tuned 가중치에서 pre-trained 가중치를 빼서 task vector를 구성하고, 산술 연산을 통해 병합 모델의 동작을 조종하는 방법이다. 가중치 보간 기반 방법의 핵심 문제는 파라미터 간섭을 무시하여 성능 저하를 초래한다는 점이다.

TIES-MERGING은 파라미터 중복성 최소화와 부호 충돌 해소에 초점을 맞춘다. TIES-MERGING은 중요하지 않은 파라미터를 삭제하고 부호 충돌을 해소하여 모델 병합의 간섭을 줄인다. 먼저 낮은 크기의 파라미터를 제거하고, 모델 전체에서 지배적인 방향을 선택하여 충돌하는 파라미터 부호를 해소한 후, 정렬된 파라미터만 병합한다.

DARE는 크기 trimming과 부호 선택을 통한 충돌 해소를 추가하고, TIES-Merging은 랜덤 드롭과 rescaling을 통해 task vector를 희소화한다. 그러나 모든 산술 방법들은 전역 고정 하이퍼파라미터에 의존하며, 작은 변동도 정확도와 출력 길이에 큰 변화를 일으킬 수 있다.

---

### 4-3. 미래 연구 시 고려사항

1. **Mate Selection의 이론적 심화**
   상호 보완적 강점을 기반으로 한 모델 쌍 구성은 효율성과 최종 모델 성능 모두를 향상시킨다. Mate selection은 유전 알고리즘의 탐구가 부족한 측면이지만, crossover 비용이 증가할수록 점점 더 중요해진다. M2N2는 이 요소의 중요성을 강조하고 이 분야의 추가 연구를 장려한다.

2. **이종 아키텍처 병합으로의 확장**
   M2N2는 동적 병합 경계, 자원 경쟁, attraction 휴리스틱을 통해 효과적으로 모델을 퓨전한다. 이 방법론은 다양한 작업으로 훈련된 모델의 gradient-free 통합을 가능하게 하면서 다양성과 견고성을 보존한다. 향후 서로 다른 아키텍처 간 병합 가능성 탐색이 필요하다.

3. **Fitness 함수 설계의 중요성**
   모델을 scratch에서 훈련하는 경우 split-point와 attraction score는 최소한의 영향을 미친다. 그러나 사전 훈련 모델에서 시작하는 경우 split-point는 결정적으로 중요해지며, attraction은 훈련 과정 전반에 걸쳐 성능을 크게 향상시킨다.

4. **다목적 최적화와의 결합**
   단일 fitness 함수를 넘어 다중 목표(정확도, 효율성, 공정성, 안전성)를 동시에 최적화하는 연구 방향이 유망하다.

5. **자원 경쟁 메커니즘의 적용 범위 확장**
   M2N2의 핵심 이점으로는 gradient 불필요, 훈련 데이터 의존성 없음, 최소한의 망각이 포함된다. 이러한 특성은 데이터 프라이버시가 중요한 federated learning, 지속 학습(continual learning) 등에 응용 가능하다.

---

## 📚 참고 자료 및 출처

| 번호 | 자료명 | 출처 |
|------|--------|------|
| 1 | **Competition and Attraction Improve Model Fusion** (주 논문) | [arxiv.org/abs/2508.16204](https://arxiv.org/abs/2508.16204) |
| 2 | **GECCO'25 ACM 공식 발표** | [dl.acm.org/doi/10.1145/3712256.3726329](https://dl.acm.org/doi/10.1145/3712256.3726329) |
| 3 | **Sakana AI 공식 블로그** | [sakana.ai/m2n2/](https://sakana.ai/m2n2/) |
| 4 | **GitHub 코드 저장소** | [github.com/SakanaAI/natural_niches](https://github.com/SakanaAI/natural_niches) |
| 5 | **ResearchGate 논문 PDF** | [researchgate.net/publication/394921233](https://www.researchgate.net/publication/394921233_Competition_and_Attraction_Improve_Model_Fusion) |
| 6 | **Emergent Mind - M2N2 분석** | [emergentmind.com/papers/2508.16204](https://www.emergentmind.com/papers/2508.16204) |
| 7 | **Evolutionary Optimization of Model Merging Recipes** (Akiba et al., 2024) | [nature.com/articles/s42256-024-00975-8](https://www.nature.com/articles/s42256-024-00975-8) |
| 8 | **From Task-Specific Models to Unified Systems: A Review of Model Merging Approaches** (2025) | [arxiv.org/html/2503.08998v1](https://arxiv.org/html/2503.08998v1) |
| 9 | **EMR-MERGING: Tuning-Free High-Performance Model Merging** (NeurIPS 2024) | [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2024/file/dda5cac5272a9bcd4bc73d90bc725ef1-Paper-Conference.pdf) |
| 10 | **Multi-objective Evolutionary Merging Enables Efficient Reasoning Models** (2026) | [arxiv.org/html/2604.06465](https://arxiv.org/html/2604.06465) |
| 11 | **Sakana AI Twitter/X 공식 발표** | [x.com/SakanaAILabs](https://x.com/SakanaAILabs/status/1959799343088857233) |
| 12 | **Greeden Blog - M2N2 In-Depth Analysis** | [blog.greeden.me](https://blog.greeden.me/en/2025/09/02/definitive-guide-in-depth-analysis-of-sakana-ais-m2n2) |

> ⚠️ **정확도 주의**: 본 답변의 수식 및 실험 결과는 공개된 논문(arXiv 2508.16204) 및 공식 코드 저장소를 기반으로 합니다. 논문 내부의 전체 수식 표기는 HTML 버전([arxiv.org/html/2508.16204v1](https://arxiv.org/html/2508.16204v1))에서 직접 확인하시기 바랍니다.
