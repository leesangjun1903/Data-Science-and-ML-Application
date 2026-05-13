
# Monte Carlo Tree Diffusion for System 2 Planning

> **논문 정보**
> - **저자**: Jaesik Yoon, Hyeonseo Cho, Doojin Baek, Yoshua Bengio, Sungjin Ahn
> - **게재**: ICML 2025 (Spotlight), *Proceedings of Machine Learning Research*, Vol. 267, pp. 72618–72640
> - **arXiv**: [2502.07202](https://arxiv.org/abs/2502.07202)
> - **공식 프로젝트 페이지**: [https://sites.google.com/view/mctd-s2planning/home](https://sites.google.com/view/mctd-s2planning/home)
> - **공식 GitHub**: [https://github.com/ahn-ml/mctd](https://github.com/ahn-ml/mctd)

---

## 1. 핵심 주장과 주요 기여 요약

확산 모델(Diffusion Models)은 최근 계획(Planning)을 위한 강력한 도구로 부상했으나, 추론 시간 연산(inference-time computation) 증가에 따라 성능이 자연스럽게 향상되는 MCTS와 달리, 표준 확산 기반 플래너는 확장성(scalability)에서 제한적이었다.

이 논문은 **Monte Carlo Tree Diffusion (MCTD)**라는 새로운 프레임워크를 제안하며, 확산 모델의 생성 능력과 MCTS의 적응적 탐색 능력을 통합한다. 핵심 아이디어는 디노이징(denoising) 과정을 **트리 구조 프로세스**로 재개념화하여 부분적으로 디노이즈된 계획을 반복적으로 평가, 가지치기, 정제하는 것이다. 유망한 궤적을 선택적으로 확장하면서 최적이 아닌 브랜치도 재방문하여 개선할 수 있도록 설계되었다.

### 주요 기여 3가지

MCTD는 트리 구조 계획을 가능하게 하는 세 가지 핵심 개념으로 구성된다: **(1) Denoising as Tree-Rollout**, **(2) Guidance Levels as Meta-Actions**, **(3) Jumpy Denoising as Simulation**. 이 혁신들이 전통적인 트리 탐색 방법과 확산 기반 계획의 간격을 잇는 MCTD의 토대를 이룬다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 🔴 2-1. 해결하고자 하는 문제

기존 MCTS는 추론 시간 연산이 증가할수록 성능이 자연스럽게 향상되지만, 표준 확산 기반 플래너는 이러한 확장성 측면에서 제한적이었다.

또한 표준 MCTS에서는 액션 공간이 크거나 연속 액션(continuous actions)을 다룰 때 트리 구성과 탐색이 계산 비용이 매우 크거나 사실상 불가능하다는 문제가 있었다.

기존 확산 기반 플래너들(예: Diffuser)은 전진 동역학 모델 없이 일련의 디노이징 단계를 통해 전체 궤적을 전체적으로 생성하지만, 장기 의존성 모델링 불량 등 핵심 한계를 지니고 있었다.

---

### 🟢 2-2. 제안하는 방법 (수식 포함)

#### (A) Denoising as Tree-Rollout (트리 롤아웃으로서의 디노이징)

표준 확산 모델과 달리, MCTD는 궤적 $\mathbf{x}$를 $S$개의 서브플랜으로 분할한다: $\mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_S]$. 각 서브플랜 $\mathbf{x}_s$에 별도의 디노이징 스케줄을 부여하여 앞 구간은 빠르게, 뒷 구간은 느리게 처리한다. 이로써 미래는 이미 정제된 과거에 기반하여 디노이즈되는 인과적(causal), 반-자기회귀적(semi-autoregressive) 과정이 된다.

이를 수식으로 표현하면 다음과 같다:

$$p(\mathbf{x}) \approx \prod_{s=1}^S p(\mathbf{x}_s \mid \mathbf{x}_{1:s-1})$$

이 과정은 미래가 이미 정제된 과거를 기반으로 디노이즈되어 인과적이고 반-자기회귀적이 되면서도, 동시에 Diffuser의 전역적으로 일관된 전체적 생성 이점도 유지한다.

#### (B) Guidance Levels as Meta-Actions (메타-액션으로서의 가이던스 레벨)

MCTD는 연속 액션 공간에서의 MCTS 한계를 극복하기 위해 **메타-액션(meta-actions)** 개념을 도입한다. 메타-액션은 알고리즘이 어떻게 탐색하거나 활용할지를 조정하는 이산적 결정이며, 이 논문에서는 디노이징 과정의 가이던스 레벨로 구현된다.

두 가지 가이던스 레벨을 예시로 들면: $\text{NO GUIDE}$는 오프라인 데이터로부터 학습한 사전 분포 $p(\mathbf{x})$에서의 샘플링(탐색적 행동), $\text{GUIDE}$는 목표 지향 분포 $p_g(\mathbf{x})$(예: 분류기 안내 확산)에서의 샘플링(활용적 행동)에 해당한다.

이를 공식화하면:

$$\mathbf{g}_s \in \{\text{NO GUIDE},\ \text{GUIDE}\}, \quad s = 1, \dots, S$$

MCTS의 노드 선택에는 **UCB (Upper Confidence Bound)** 기준이 사용된다:

$$\text{UCB}(v) = \bar{r}(v) + C \cdot \sqrt{\frac{\ln N(v_{\text{parent}})}{N(v)}}$$

여기서 $\bar{r}(v)$는 노드 $v$의 평균 보상, $N(v)$는 방문 횟수, $C$는 탐색 상수이다.

#### (C) Jumpy Denoising as Simulation (시뮬레이션으로서의 점프 디노이징)

트리 롤아웃 디노이징 프로세스가 $s$번째 서브플랜까지 진행되면, 나머지 단계는 매 $C$ 스텝을 건너뜀으로써 빠르게 디노이즈된다:

$$\tilde{\mathbf{x}}_{s+1:S} \sim p(\mathbf{x}_{s+1:S} \mid \mathbf{x}_{1:s},\ \mathbf{g})$$

이로써 전체 궤적 $\tilde{\mathbf{x}} = (\mathbf{x}\_{1:s}, \tilde{\mathbf{x}}_{s+1:S})$이 생성되고, 보상 함수 $r(\tilde{\mathbf{x}})$로 평가된다. 이 빠른 디노이징은 더 큰 근사 오차를 유발할 수 있지만, 계산 효율이 매우 높아 시뮬레이션 단계에 적합하다.

---

### 🔵 2-3. 모델 구조 (MCTS 4단계)

MCTD의 라운드는 표준 MCTS의 4단계—**선택(Selection)**, **확장(Expansion)**, **시뮬레이션(Simulation)**, **역전파(Backpropagation)**—로 구성된다. 각 노드는 부분적으로 디노이즈된 서브-궤적에 해당하고, 엣지는 이진 가이던스 레벨(0 = 가이던스 없음, 1 = 가이던스 있음)로 레이블링된다. 새로운 노드가 확장되면 "점프 디노이징"이 수행되어 값이 빠르게 추정되고, 트리의 경로를 따라 역전파된다.

시뮬레이션 단계 후 완성된 계획을 평가하여 얻은 보상이 루트까지의 모든 부모 노드 값 추정치를 업데이트하도록 역전파된다. 이 역전파 과정에서 메타-액션 기반 가이던스 스케줄도 함께 업데이트되어, 트리가 미래 반복에서 탐색-활용 균형을 동적으로 조정할 수 있게 된다.

MCTD에서는 각 서브플랜(시간적으로 확장된 상태)을 개별 타임스텝 대신 **트리의 단일 노드**로 표현한다. 이 높은 추상 수준에서 동작함으로써 트리 탐색이 더 효율적이고 확장 가능하게 된다.

---

### 🟡 2-4. 성능 향상

MCTD는 Offline Goal-conditioned RL Benchmark (OGBench)의 pointmaze와 antmaze 태스크(중간, 대형, 초대형 미로)에서 평가되었다. 에이전트는 지정된 목표 영역에 도달하면 보상을 받으며, 해당 데이터셋은 장기 궤적으로 구성된다. MCTD는 특히 가장 큰 맵(giant)에서 거의 완벽한 성능을 보이며 다른 방법들을 큰 폭으로 능가했다.

Diffuser와 Diffusion Forcing가 성공적인 궤적 계획을 생성하지 못하는 반면, MCTD는 계획을 적응적으로 정제함으로써 성공한다.

장기 작업(long-horizon tasks)에 대한 실험 결과, MCTD는 확산 기반 기준선보다 우수하며, 추론 시간 연산이 증가할수록 더 높은 품질의 솔루션을 산출한다.

---

### 🔴 2-5. 한계

한계점으로는 복잡한 시나리오에서의 높은 계산 요구량과 실시간 응용에서의 잠재적 확장성 문제가 있으며, 다양한 실세계 시나리오에서의 확장성 향상 및 테스트에 대한 추가 연구가 필요하다.

MCTD는 트리 탐색의 순차적 특성과 반복적 디노이징 비용으로 인해 상당한 계산 오버헤드가 발생한다.

예를 들어, Diffuser-Random Search는 다중 샘플을 병렬 배치로 디노이즈할 수 있으나, 이는 MCTD의 트리 구조 방식에서는 직접적으로 가능하지 않다.

또한 점프 디노이징 과정은 더 큰 근사 오차를 유발할 수 있다는 본질적 한계를 가진다.

---

## 3. 모델의 일반화 성능 향상 가능성

MCTD는 트리 탐색 기반 디노이징 방식을 통해 생성 다양성을 넓히고, 테스트 시간에 복잡한 행동 계획 태스크에서 성능을 향상시킨다.

기존 확산 접근법이 전체 행동 시퀀스를 한 번에 생성하는 것과 달리, MCTD는 트리 구조 탐색을 통해 점진적으로 계획을 구축하여 다양한 가능성을 탐색하고, 결과를 평가하며, 유망한 경로를 정제한다. 이를 통해 보다 사려 깊고 전략적인 계획이 가능해진다.

주목할 점은 MCTD가 **미지의 시나리오에 대한 더 나은 일반화(generalization to unseen scenarios)** 를 달성했다는 것이다.

확산 기반 플래너의 실패 사례, 특히 재계획 없는 Diffusion Forcing의 경우, 반복적 교정 부재가 궤적 붕괴로 이어짐을 보여준다. 이는 복잡하고 부분 관측 가능한 환경에서 강건한 궤적 생성을 위한 재계획의 필요성을 간접적으로 강조하며, 이것이 MCTD의 일반화 강점과 직결된다.

특히 일반화 관점에서 중요한 구조적 특징은:

- **반-자기회귀적 구조**: 표준 Diffuser와 달리 각 서브플랜에 별도의 디노이징 페이스를 부여함으로써, 미래가 이미 정제된 과거를 기반으로 디노이즈되는 인과적 특성을 가진다.
- **동적 탐색-활용 균형**: MCTD(또는 Best-of-N)는 각 단계에서 가장 적절한 가이던스 레벨을 동적으로 선택하고, 이 메커니즘이 안내 함수의 영향을 적응적으로 조절하여 더 효과적이고 다양한 로컬 계획을 생성한다.

---

## 4. 미래 연구에 미치는 영향과 고려할 점

### 🔮 4-1. 연구 영향

MCTD는 계산 시간이 주어질수록 일관되게 더 나은 솔루션을 찾는 확장성을 갖추어 복잡한 문제에 이상적이며, 생성 AI와 전략적 추론을 연결하여 AI 시스템의 장기 행동 계획 방식을 발전시킨다.

이 논문은 이미 여러 후속 연구를 촉발하였다:

1. **C-MCTD (Compositional MCTD)**: C-MCTD는 서브플랜 내부가 아닌 플랜 간 추론을 가능하게 하기 위해, 스티칭 기반 트리 확장으로 개별 확산 생성 계획들을 더 길고 일관된 계획으로 연결하는 추론 시간 스케일링 프레임워크이다.

2. **Fast-MCTD**: MCTD의 강점을 유지하면서 속도와 확장성을 크게 향상시키는 Fast-MCTD를 제안하며, 병렬 롤아웃을 가능하게 하는 Parallel MCTD와 궤적 조대화를 통한 Sparse MCTD 두 가지 기법을 통합한다.

3. **단백질 설계 응용 (MCTD-ME)**: MCTD-ME는 마스크 확산 모델과 트리 탐색을 통합하여 다중 토큰 계획과 다중 전문가 가이던스 하의 효율적 탐색을 가능하게 하며, 자기회귀 플래너와 달리 생체물리적 충실도 향상 확산 디노이징을 롤아웃 엔진으로 사용한다.

### 🧭 4-2. 향후 연구 시 고려할 점

향후 연구 방향으로는 **적응형 컴퓨팅 할당(adaptive compute allocation)**, **학습 기반 메타-액션 선택(learning-based meta-action selection)**, **보상 형성(reward shaping)**을 통한 성능 향상이 제시되어 있으며, 이는 더 확장 가능하고 유연한 System 2 계획을 위한 방향성을 제시한다.

추가적으로 고려해야 할 연구 방향:

| 고려사항 | 설명 |
|---|---|
| **계산 효율화** | MCTD는 트리 탐색의 순차적 특성과 반복적 디노이징 비용으로 인해 상당한 계산 오버헤드가 발생한다는 점에서, 병렬화 및 희소화 전략이 필수적이다. |
| **보상 함수 설계** | 트리 탐색의 품질은 보상 함수의 정확도에 크게 의존하므로, 보상 오차에 대한 강건성 확보가 필요하다. |
| **실세계 적용** | 복잡한 시나리오에서의 높은 계산 요구량과 실시간 응용에서의 확장성 문제를 해소하기 위한 하드웨어 친화적 설계가 요구된다. |
| **부분 관측 환경** | 복잡하고 부분 관측 가능한 환경에서 강건한 궤적 생성을 위한 재계획의 필요성이 확인되었으므로, 부분 관측 설정에서의 MCTD 확장이 중요하다. |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 기여 | MCTD와의 차이점 |
|---|---|---|---|
| **Diffuser** (Janner et al.) | 2022 | 확산 모델로 전체 궤적 생성 | 추론 시간 확장성 없음 |
| **Decision Diffuser / DD** (Ajay et al.) | 2022 | 조건부 생성 모델링 기반 의사결정 | 단일 샘플링, 탐색 없음 |
| **Simple Hierarchical Planning (HD)** (Chen et al.) | 2024 | 계층적 확산 계획 | OOD 태스크에서의 일반화 능력 향상 입증되었으나 명시적 탐색 없음 |
| **Diffusion Forcing** (Chen et al.) | 2024 | 다음 토큰 예측 + 전체 시퀀스 확산 결합 | MCTD에 비해 적응적 정제 불가 |
| **Monte Carlo Guidance (MCG)** (Chen et al.) | 2024 | 다중 서브플랜 평균 가이던스 신호 | MCG는 기대 보상을 향한 계획을 장려하지만 명시적 탐색 메커니즘을 구현하지 않는다 |
| **DiffuserLite** | 2024 | 실시간 확산 계획 (122Hz) | 계산 속도에 중점, 탐색보다 효율 우선 |
| **MCTD** (Yoon et al.) | **2025** | MCTS + 확산 통합, 추론 시간 확장 | **본 논문** |
| **C-MCTD** (후속) | 2025 | 플랜 수준 트리 확장으로 장기 계획 | 스티칭 기반 확장으로 고립된 계획 생성의 한계 극복 |
| **MCTD-ME** (단백질 설계) | 2025 | 다중 전문가 가이던스로 단백질 시퀀스 설계 | 마스크 확산 + 다중 전문가 앙상블로 MCTD를 생물학 도메인에 확장 |
| **Fast-MCTD** | 2025 | 병렬·희소 계획으로 100배 속도 향상 | MCTD의 강점을 유지하면서 병렬 롤아웃 및 궤적 조대화를 도입 |

---

## 📚 참고 자료 (출처)

1. **arXiv 논문 원문**: Yoon, J. et al. (2025). *Monte Carlo Tree Diffusion for System 2 Planning*. arXiv:2502.07202. https://arxiv.org/abs/2502.07202
2. **ICML 2025 공식 게재본**: Proceedings of Machine Learning Research, Vol. 267, pp. 72618–72640. https://proceedings.mlr.press/v267/yoon25a.html
3. **공식 프로젝트 페이지 (MCTD)**: https://sites.google.com/view/mctd-s2planning/home
4. **저자 개인 페이지**: https://jaesikyoon.com/mctd-page/
5. **공식 GitHub 코드**: https://github.com/ahn-ml/mctd
6. **OpenReview (ICML 2025)**: https://openreview.net/forum?id=XrCbBdycDc
7. **ICML 포스터**: https://icml.cc/virtual/2025/poster/44944
8. **후속 연구 - C-MCTD**: arXiv:2510.21361. https://arxiv.org/html/2510.21361
9. **후속 연구 - Fast-MCTD**: arXiv:2506.09498. https://arxiv.org/abs/2506.09498
10. **후속 연구 - MCTD-ME (단백질 설계)**: arXiv:2509.15796. https://arxiv.org/abs/2509.15796
11. **관련 연구 - Simple Hierarchical Planning with Diffusion (HD)**: ICLR 2024. https://arxiv.org/html/2401.02644v1
12. **관련 연구 - DiffuserLite**: NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/dd6a47bc0aad6f34aa5e77706d90cdc4-Paper-Conference.pdf
13. **관련 연구 - Diffusion Model for Planning: A Systematic Literature Review**: arXiv:2408.10266. https://arxiv.org/pdf/2408.10266
14. **Consensus 논문 요약**: https://consensus.app/papers/monte-carlo-tree-diffusion-for-system-2-planning-baek-ahn/
