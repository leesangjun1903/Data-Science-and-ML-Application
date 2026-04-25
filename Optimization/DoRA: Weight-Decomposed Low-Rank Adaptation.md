
# DoRA: Weight-Decomposed Low-Rank Adaptation

---

## 📌 1. 핵심 주장 및 주요 기여 (간결 요약)

DoRA는 ICML 2024 Oral 논문으로, FT(Full Fine-Tuning)와 LoRA 사이의 근본적 차이를 규명하기 위한 새로운 가중치 분해(weight decomposition) 분석을 제안하고, 그 결과를 바탕으로 Weight-Decomposed Low-Rank Adaptation(DoRA)를 제안합니다.

### 세 가지 핵심 기여

DoRA는 다음 세 가지 핵심 지점에 기여합니다:
1. **LoRA vs. 완전 파인튜닝 분석**: 가중치 행렬을 크기(magnitude)와 방향(direction)으로 분해하여 LoRA와 Full Fine-Tuning의 주요 차이를 밝힙니다.
2. **크기와 방향의 분리 최적화**: DoRA는 크기와 방향을 독립적으로 최적화하여 LoRA의 낮은 자원 요구를 유지하면서 Full Fine-Tuning에 가까운 결과를 달성합니다.
3. **다양한 방법과의 결합 평가**: DoRA는 LoRA 외에도 VeRA, QLoRA와의 결합 가능성을 탐구합니다.

---

## 📌 2. 문제 정의 / 제안 방법 (수식) / 모델 구조 / 성능 / 한계

---

### 2-1. 해결하고자 하는 문제

PEFT(파라미터 효율적 파인튜닝) 방법 중 LoRA와 그 변형들은 추가적인 추론 비용 없이 파인튜닝이 가능하여 널리 활용되고 있습니다. 그러나 이 방법들과 완전 파인튜닝(FT) 사이에는 여전히 정확도 격차(accuracy gap)가 존재합니다.

DoRA 개발의 동기는 LoRA와 완전 파인튜닝의 학습 패턴을 분석·비교하는 것에서 출발합니다. 저자들은 LoRA가 크기(magnitude)와 방향(direction) 업데이트를 비례적으로 증가 또는 감소시킬 뿐, 미세한 방향 변화만 일으키는 것이 불가능하다는 것을 발견했습니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### ① LoRA의 기존 가중치 업데이트 공식

LoRA에서는 사전학습 가중치 $W_0 \in \mathbb{R}^{d \times k}$에 저랭크 행렬의 곱 $\Delta W = BA$ 를 더하여 업데이트합니다:

$$W' = W_0 + \Delta W = W_0 + BA$$

여기서 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ 이며 $r \ll \min(d, k)$입니다.

#### ② DoRA의 가중치 분해 (Weight Decomposition)

DoRA는 두 단계로 설명됩니다. 첫 번째 단계는 사전학습 가중치 행렬을 크기 벡터(magnitude vector) $m$과 방향 행렬(directional matrix) $V$로 분해하는 것입니다. 두 번째 단계는 LoRA를 방향 행렬 $V$에 적용하고 크기 벡터 $m$을 별도로 학습하는 것입니다.

가중치 행렬 $W$를 아래와 같이 분해합니다:

$$W = m \cdot \frac{V}{\|V\|_c}$$

여기서:
- $m \in \mathbb{R}^{1 \times k}$: 각 열(column)의 크기를 나타내는 벡터 (학습 가능)
- $V \in \mathbb{R}^{d \times k}$: 방향 행렬 (초기에는 고정, LoRA로 업데이트)
- $\|\cdot\|_c$: 각 열 벡터에 대한 벡터 노름(column-wise norm)

DoRA는 사전학습 가중치 $W_0$으로 초기화되며, 초기화 시 $m = \|W_0\|_c$, $V = W_0$으로 설정됩니다. 이후 $V$는 고정(frozen)되고 $m$은 학습 가능한 벡터가 됩니다. 방향 성분은 LoRA를 통해 업데이트됩니다.

#### ③ DoRA의 최종 파인튜닝 공식

$$W' = m \cdot \frac{V + \Delta V}{\|V + \Delta V\|_c} = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}$$

여기서 $\Delta V = BA$ 는 LoRA 방식의 저랭크 업데이트입니다.

DoRA는 LoRA를 구체적으로 방향 성분의 업데이트에만 사용하며, 크기 성분은 직접 파인튜닝합니다. 이 접근법은 완전 파인튜닝의 학습 능력을 더욱 가깝게 모방하는 것을 목표로 합니다.

#### ④ 학습 비용 절감을 위한 수정

학습 비용을 줄이기 위해 $\|V + \Delta V\|_c$를 상수(constant)로 처리하여 그래디언트 그래프에서 분리(detach)합니다. 이는 $\|V + \Delta V\|_c$가 $\Delta V$의 업데이트를 동적으로 반영하지만, 역전파 시 어떠한 그래디언트도 수신하지 않음을 의미합니다.

---

### 2-3. 모델 구조

DoRA의 전체적인 개요는 사전학습 가중치를 크기(magnitude)와 방향(direction) 성분으로 분해하여 파인튜닝하되, 특히 방향 성분을 LoRA로 효율적으로 업데이트하는 구조를 취합니다.

**DoRA의 Forward Pass 흐름:**

```
입력 x
  ↓
기본 선형 변환: F.linear(x, W)
  ↓
LoRA 업데이트: new_weight_v = W + (B @ A) * scaling
  ↓
정규화 스케일 계산: norm_scale = m / ‖new_weight_v‖ (detach)
  ↓
최종 출력 = norm_scale * (W_forward + LoRA_forward)
```

DoRA는 추론 전 분해된 크기와 방향 성분을 사전학습 가중치에 병합할 수 있으므로, 추가적인 추론 지연(latency)이 발생하지 않습니다.

DoRA와 가중치 정규화(weight normalization)의 주요 차이점은 학습 방식에 있습니다. 가중치 정규화는 두 성분 모두를 처음부터 학습하여 초기화에 민감하지만, DoRA는 두 성분 모두 사전학습 가중치로 초기화하므로 초기화 문제를 방지합니다.

---

### 2-4. 성능 향상

DoRA는 LoRA의 학습 능력과 학습 안정성을 향상시키면서도 추가적인 추론 오버헤드를 방지합니다. DoRA는 LLaMA, LLaVA, VL-BART의 파인튜닝에서 상식 추론, 시각 명령 튜닝, 이미지/비디오-텍스트 이해 등 다양한 다운스트림 태스크에서 LoRA를 일관되게 능가합니다.

DoRA는 다양한 LLM 및 VLM 태스크에서 LoRA를 일관되게 능가하며, 예를 들어 상식 추론에서 Llama 7B/13B 대비 각각 +3.7/+1.0의 성능 향상을 보입니다.

8개의 서로 다른 추론 데이터셋과 4개의 백본 모델로 평가했을 때, DoRA는 모든 태스크에서 LoRA를 일관되게 능가했습니다. 특히 절반의 랭크 크기(DoRA†)를 사용하여 절반의 학습 가능한 파라미터로도 DoRA는 LoRA 대비 성능 우위를 유지했습니다.

훈련 샘플 수를 1000개로 줄였을 때에도 DoRA와 DVoRA는 LoRA와 VeRA 대비 각각 0.29, 0.22의 우위를 유지했습니다. 이는 우리의 방법이 훈련 샘플 수에 관계없이 지속적으로 성능을 향상시킴을 보여줍니다.

DoRA는 낮은 랭크 설정에서도 강건하며, 더 낮은 파라미터 예산에서 LoRA 대비 더 높은 정확도를 유지합니다. 또한 실제 생성 AI 환경에서 DoRA는 LoRA(85.5%) 및 RAG(81.2%)보다 높은 정확도(90.1%)를 달성합니다.

---

### 2-5. 한계점

1. **추가 메모리 사용**: DoRA에서 저랭크 적응이 방향 성분으로 재지향되므로, 저랭크 업데이트의 그래디언트가 $W'$의 그래디언트와 달라집니다. 이 차이로 인해 역전파 시 추가 메모리가 필요합니다.

2. **과적합 위험성**: DoRA는 크기 벡터 또는 행렬이라는 추가 파라미터를 도입하므로, 약간의 과적합 위험성이 증가할 수 있습니다.

3. **하이퍼파라미터 민감성**: DoRA로 파인튜닝할 때 LoRA 설정을 사용하면 대부분의 경우 더 나은 결과를 얻을 수 있지만, LoRA 대비 최적 성능을 위해 하이퍼파라미터 조정이 필요합니다. 특히 LoRA보다 약간 낮은 학습률로 시작하는 것이 권장됩니다.

4. **수렴 속도**: LoRA는 DoRA보다 더 빠르게 수렴하는 경향이 있어, LoRA에서 과적합을 유발하는 파라미터 설정이 DoRA에서는 잘 동작할 수 있습니다.

---

## 📌 3. 일반화 성능 향상 가능성

DoRA의 일반화 성능 향상 가능성은 여러 측면에서 확인됩니다.

### 3-1. 크기-방향 분리 학습의 일반화 효과

저자들은 완전 파인튜닝이 자연스럽게 음의 기울기(negative slope)를 보이는 이유를 설명합니다. 사전학습 모델 가중치는 이미 다운스트림 태스크에 관련된 광범위한 지식을 포함하고 있어, 크기나 방향 중 하나만 크게 변화시켜도 효과적인 적응이 가능합니다. 이것이 DoRA가 달성하는 것과 정확히 일치하며, LoRA보다 완전 파인튜닝에 더 가까운 동작을 보여줍니다.

LoRA는 무엇을 변경할지(what to change)를 학습하는 반면, DoRA는 무엇을 변경하고 얼마나 강하게 변경할지(what to change and how strongly)를 학습합니다.

### 3-2. 저데이터 환경에서의 일반화

LoRA와 VeRA 대비 0.3, 0.33의 마진 차이를 보이며, 샘플 수를 1000개로 줄이더라도 DoRA와 DVoRA는 성능 우위를 유지합니다. 이는 훈련 샘플 양에 관계없이 방법이 일관되게 성능을 향상시킴을 보여줍니다.

### 3-3. 저랭크 환경에서의 강건성

DoRA 품질은 특히 낮은 랭크에서 LoRA보다 우수합니다. 예를 들어 랭크 8에서 DoRA와 LoRA의 품질 차이는 랭크 32나 64에서의 차이보다 훨씬 더 유의미합니다.

### 3-4. 다중 도메인 일반화

DoRA의 성능 향상은 언어, 비전, 다중 도메인 벤치마크 전반에 걸쳐 확장됩니다.

DoRA는 특히 모델이 전문 용어, 작성 스타일 또는 지식 도메인에 적응해야 하는 도메인 적응 시나리오에서 성공적입니다. DoRA의 명시적 크기 제어는 중요한 수정이 필요한 레이어의 업데이트를 더욱 효과적으로 스케일링할 수 있게 합니다.

### 3-5. 다운스트림 태스크 성능

DoRA는 LoRA와 FT 모두에 대해 우수한 성능을 보이며, LoRA 대비 평균 0.7%, FT 대비 1.1% 향상을 달성합니다.

---

## 📌 4. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 핵심 아이디어 | DoRA와의 차이점 |
|------|------|-------------|--------------|
| **LoRA** | 2021 | 저랭크 행렬로 가중치 변화 근사 | 크기/방향 분리 없음 |
| **AdaLoRA** | 2023 | SVD 기반 동적 랭크 할당 | 랭크 적응적 조정, 방향 분리 없음 |
| **QLoRA** | 2023 | 4-bit 양자화 + LoRA | 메모리 효율 중심, 크기 분리 없음 |
| **VeRA** | 2023 | 공유 랜덤 행렬 + 스케일링 벡터 | 매우 적은 파라미터, 표현력 제한 |
| **DoRA** | 2024 | 크기+방향 분해, LoRA로 방향 업데이트 | 학습 능력/안정성 향상 |
| **DVoRA** | 2024 | DoRA + VeRA 결합 | 파라미터 효율 극대화 |
| **QDoRA** | 2024 | QLoRA + DoRA 결합 | 소비자용 GPU에서 DoRA 적용 |

AdaLoRA(Zhang et al., 2023)는 각 레이어의 중요도에 따라 랭크 예산을 동적으로 할당합니다. SVD 기반으로 어떤 가중치 행렬이 더 높은 랭크 업데이트에서 이점을 얻는지 식별하며, 중요 레이어에는 더 높은 랭크를, 덜 민감한 레이어에는 더 낮은 랭크를 할당합니다. AdaLoRA는 제한된 예산에서 기본 LoRA보다 우수한 성능을 보입니다.

DoRA가 LoRA와 FT 사이의 간극을 좁히면서, DoRA가 QLoRA 프레임워크 내에서 LoRA의 정확도를 향상시킬 수 있는지 탐색하는 것이 자연스러웠습니다. 최근 Answer.AI 팀과의 협업으로 QLoRA의 LoRA 성분을 DoRA로 대체한 QDoRA 프로젝트가 진행되었으며, QDoRA는 Llama 2와 Llama 3 모두에서 FT와 QLoRA를 능가했습니다.

DoRAN은 DoRA의 새로운 변형으로, 훈련을 더욱 안정화하고 DoRA의 샘플 효율성을 높이기 위해 설계되었으며, LoRA, DoRA 및 기타 PEFT 기준선을 일관되게 능가합니다.

DeLoRA(Decoupled Low-rank Adaptation)는 학습 가능한 저랭크 행렬을 정규화하고 스케일링하여 각도 학습(angular learning)을 적응 강도(adaptation strength)로부터 효과적으로 분리하는 새로운 파인튜닝 방법으로, 성능을 저해하지 않으면서 강건성을 향상시킵니다.

DoRA(Liu et al., 2024), MiSS, AdaLoRA(Zhang et al., 2023b)와 같은 구조적 변형들은 DoRA가 RLVR(Reinforcement Learning with Verifiable Rewards) 환경에서 표준 LoRA를 능가하며 우수한 추론 정확도를 달성함을 보여줍니다.

---

## 📌 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 연구에 미치는 영향

**① 가중치 분해 패러다임의 확산**

DoRA의 핵심 혁신인 각 가중치 행렬을 독립적인 크기와 방향 성분으로 명시적으로 분해하는 방법은 더욱 세밀한 적응과 향상된 학습 안정성을 가능하게 합니다. DoRA와 그 확장들은 자연어, 멀티모달, 바이오메디컬 태스크 전반에 걸쳐 뛰어난 성능을 보여주었으며, PEFT 전략 설계에서 빠른 발전을 이끌어냈습니다.

**② LoRA 생태계의 기본 대체재 가능성**

NVIDIA Research Taiwan과 NVIDIA Learning and Perception Research Group이 개발한 DoRA는 LoRA의 기본 대체재(default replacement)가 될 수 있으며, 추가적인 추론 오버헤드 없이 LoRA의 학습 능력과 안정성을 향상시킵니다.

DoRA는 현재 HuggingFace PEFT 패키지에서 지원되며, LoRAConfig의 `use_dora` 인수를 True로 설정하는 것만으로 간단히 적용할 수 있습니다.

**③ 다양한 LoRA 변형과의 결합 확장**

다수의 DoRA 파생 연구들이 핵심 원칙을 확장하고 있습니다: Dynamic Rank DoRA는 성분 중요도 기반으로 런타임 가지치기와 할당을 수행하고, BoRA는 행과 열에 독립적인 학습 가능한 스케일링을 적용하며, EDoRA와 DuDe는 SVD 기반 초기화를 사용하여 학습 가능한 파라미터 수를 크게 줄입니다.

**④ 멀티모달 및 확산 모델로의 확장**

DoRA는 압축 인식 LLM 및 텍스트-이미지 생성을 포함한 기타 태스크에서도 적용 가능성이 입증되었습니다.

---

### 5-2. 향후 연구 시 고려할 점

**① 적응적 랭크 + 크기-방향 분리의 결합**

LoRA는 모델의 도메인 특화 지식 기억 및 다운스트림 태스크 일반화 능력을 제한할 수 있습니다. 핵심 한계는 모든 레이어에 고정된 랭크를 사용하는 것으로, 이는 레이어마다 모델 적응에 기여하는 정도가 다르다는 사실을 무시합니다. 이 균일한 할당은 학습 가능한 파라미터의 비효율적 사용으로 이어질 수 있습니다.

DoRA가 AdaLoRA의 동적 랭크 할당과 결합된다면, 레이어별로 최적의 랭크와 크기-방향 분리를 동시에 달성하는 연구가 기대됩니다.

**② 양자화 환경에서의 최적화**

메모리 요구를 줄이기 위해 QLoRA는 사전학습 모델을 4-bit으로 양자화하고 얼어붙은 저비트 백본 위에서 LoRA를 파인튜닝합니다. DoRA가 LoRA와 FT 사이의 간극을 좁히므로, DoRA가 QLoRA 프레임워크 내에서 LoRA의 정확도를 향상시킬 수 있는지 탐색이 필요합니다.

**③ 과적합 위험성 대비 정규화 전략**

DoRA 파인튜닝 시 LoRA보다 약간 낮은 학습률로 시작하는 것을 권장하며, LoRA 설정의 절반 랭크로 시작해도 종종 LoRA에 비견되거나 우월한 정확도를 달성할 수 있습니다.

DoRA의 추가 파라미터로 인한 과적합 위험을 억제할 수 있는 체계적인 정규화 전략 연구가 필요합니다.

**④ 다양한 모달리티와 아키텍처로의 확장**

EDoRA는 EEG 기반 BCI 응용을 위한 파라미터 효율적 전이 학습을 가능하게 하며, Zero-shot HOI 탐지에서 가중치 분해 저랭크 분해는 이전 VLM 적응 방법들을 크게 능가합니다.

**⑤ RLVR(강화학습 기반 추론) 환경에서의 적용**

대규모 경험적 분석은 표준 LoRA의 기본 채택에 도전합니다. 표준 LoRA는 RLVR에서 최적이 아니며, DoRA 같은 구조적 변형들이 지속적으로 우수한 추론 정확도를 달성합니다.

---

## 📚 참고 자료 (출처)

1. **원본 논문 (arXiv)**: Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., & Chen, M.-H. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation*. arXiv:2402.09353. https://arxiv.org/abs/2402.09353

2. **ICML 2024 공식 게재본**: Proceedings of the 41st ICML, PMLR 235:32100–32121, 2024. https://proceedings.mlr.press/v235/liu24bn.html

3. **NVIDIA Research 공식 발표**: *DoRA: Weight-Decomposed Low-Rank Adaptation*. NVIDIA Research Publication (2024-07). https://research.nvidia.com/publication/2024-07_dora-weight-decomposed-low-rank-adaptation

4. **NVIDIA Technical Blog**: *Introducing DoRA, a High-Performing Alternative to LoRA for Fine-Tuning*. https://developer.nvidia.com/blog/introducing-dora-a-high-performing-alternative-to-lora-for-fine-tuning/

5. **GitHub 공식 구현 (NVlabs)**: https://github.com/NVlabs/DoRA

6. **DoRA 프로젝트 페이지**: https://nbasyl.github.io/DoRA-project-page/

7. **HuggingFace Paper 페이지**: https://huggingface.co/papers/2402.09353

8. **Sebastian Raschka 튜토리얼**: *Improving LoRA: Implementing Weight-Decomposed Low-Rank Adaptation (DoRA) from Scratch*. https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch

9. **Towards AI 해설**: *DoRA Explained: Next Evolution of LoRA?* https://towardsai.net/p/l/dora-explained-next-evolution-of-lora

10. **Emergent Mind 연구 요약**: *Weight-Decomposed Low-Rank Adaptation*. https://www.emergentmind.com/topics/weight-decomposed-low-rank-adaptation-dora

11. **Moonlight 문헌 리뷰**: *[Literature Review] DoRA: Weight-Decomposed Low-Rank Adaptation*. https://www.themoonlight.io/en/review/dora-weight-decomposed-low-rank-adaptation

12. **Michael Brenndoerfer 가이드**: *PEFT Beyond LoRA: Advanced Parameter-Efficient Fine-Tuning Techniques*. https://mbrenndoerfer.com/writing/peft-beyond-lora-advanced-parameter-efficient-finetuning-techniques

13. **ElaLoRA 관련 연구 (arXiv)**: *ElaLoRA: Elastic & Learnable Low-Rank Adaptation for Efficient Model Fine-Tuning*. arXiv:2504.00254. https://arxiv.org/html/2504.00254v1

14. **RLVR PEFT 평가 연구**: *Evaluating Parameter Efficient Methods for RLVR*. arXiv:2512.23165. https://arxiv.org/pdf/2512.23165

15. **ACL 2024 Dynamic DoRA**: Mao et al. (2024). *DoRA: Enhancing Parameter-Efficient Fine-Tuning with Dynamic Rank Distribution*. ACL 2024. https://aclanthology.org/2024.acl-long.626/

16. **Semantic Scholar**: https://www.semanticscholar.org/paper/DoRA:-Weight-Decomposed-Low-Rank-Adaptation-Liu-Wang/da053e2a4ba1b244940c8f2cad5dcdf0d730f85f

# DoRA: Weight-Decomposed Low-Rank Adaptation

---

## 📌 참고 자료 (출처)

> **주 논문:**
> - Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., & Chen, M.-H. (2024). **DoRA: Weight-Decomposed Low-Rank Adaptation**. *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, PMLR 235. arXiv:2402.09353v6.
>
> **관련 비교 논문 (논문 내 인용 기반):**
> - Hu et al. (2022). **LoRA: Low-Rank Adaptation of Large Language Models**. ICLR 2022.
> - Kopiczko et al. (2024). **VeRA: Vector-based Random Matrix Adaptation**. ICLR 2024.
> - Dettmers et al. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs**. NeurIPS 2023.
> - Houlsby et al. (2019). **Parameter-Efficient Transfer Learning for NLP**. ICML 2019.
> - Salimans & Kingma (2016). **Weight Normalization**. NeurIPS 2016.
> - Zhang et al. (2023). **AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning**. ICLR 2023.
> - He et al. (2021). **Towards a Unified View of Parameter-Efficient Transfer Learning**. ICLR 2021.
> - Sung et al. (2022). **VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks**. CVPR 2022.
> - Liu et al. (2023a). **Visual Instruction Tuning (LLaVA)**. NeurIPS 2023.
> - Touvron et al. (2023). **LLaMA: Open and Efficient Foundation Language Models**. arXiv:2302.13971.

---

## 1. 핵심 주장과 주요 기여 요약

### 1.1 핵심 주장

DoRA는 사전학습된 가중치(pre-trained weight)를 **크기(magnitude)** 와 **방향(direction)** 두 성분으로 분해하여 파인튜닝함으로써, LoRA와 Full Fine-Tuning(FT) 사이의 **학습 능력 격차(capacity gap)** 를 체계적으로 해소할 수 있다고 주장합니다.

핵심 통찰은 다음과 같습니다:

> *"LoRA는 magnitude 변화와 direction 변화 사이에 강한 양(+)의 상관관계(r=0.83)를 보이는 반면, FT는 음(-)의 상관관계(r=-0.62)를 보인다. DoRA는 이 패턴을 FT에 가깝게 재현(r=-0.31)함으로써 더 세밀한 학습이 가능하다."*

### 1.2 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **① 새로운 분석 도구** | Weight Decomposition Analysis: FT와 LoRA의 학습 패턴 차이를 magnitude/direction 관점에서 최초로 정량 분석 |
| **② DoRA 방법 제안** | 추가 추론 비용 없이 FT에 근접한 학습 능력을 달성하는 새로운 PEFT 방법 |
| **③ 광범위한 검증** | NLP(LLaMA 계열), Vision-Language(LLaVA, VL-BART), Text-to-Image(SDXL) 등 다양한 도메인에서 LoRA 대비 일관된 성능 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**배경:**
대규모 사전학습 모델(LLM, LVLM)을 특정 태스크에 적응시키기 위해 Full Fine-Tuning(FT)을 수행하면 막대한 컴퓨팅 비용이 발생합니다. LoRA는 이를 해결하지만, FT와의 **정확도 격차**가 여전히 존재하며, 기존 연구들은 이를 단순히 "훈련 가능 파라미터 수의 부족" 때문이라고만 설명해 왔습니다.

**DoRA가 새롭게 발견한 문제:**
LoRA의 학습 패턴 자체가 FT와 본질적으로 다르다는 것입니다:

- **LoRA**: $\Delta D$ (방향 변화)와 $\Delta M$ (크기 변화) 사이 **양의 상관관계** → magnitude와 direction을 동시에 학습해야 해서 최적화가 복잡
- **FT**: **음의 상관관계** → 큰 방향 변화 시 작은 크기 변화, 또는 그 반대가 가능 → 더 세밀하고 효율적인 적응

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: Weight Decomposition (가중치 분해)

임의의 가중치 행렬 $W \in \mathbb{R}^{d \times k}$를 다음과 같이 분해합니다:

$$W = m \frac{V}{\|V\|_c} = \|W\|_c \frac{W}{\|W\|_c} $$

- $m \in \mathbb{R}^{1 \times k}$: **magnitude vector** (각 열 벡터의 크기)
- $V \in \mathbb{R}^{d \times k}$: **directional matrix** (방향 행렬)
- $\|\cdot\|_c$: 행렬의 각 열(column) 방향 벡터 노름(vector-wise norm)
- $V/\|V\|_c$의 각 열은 단위 벡터(unit vector)가 됩니다.

#### Step 2: LoRA 수식 (비교 기준)

$$W' = W_0 + \Delta W = W_0 + \underline{B}\underline{A} $$

- $W_0 \in \mathbb{R}^{d \times k}$: 동결된 사전학습 가중치
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$: 훈련 가능한 저랭크 행렬 ( $r \ll \min(d,k)$ )
- 밑줄: 훈련 가능 파라미터

#### Step 3: DoRA 공식 (핵심)

$$\boxed{W' = \underline{m} \frac{V + \Delta V}{\|V + \Delta V\|_c} = \underline{m} \frac{W_0 + \underline{B}\underline{A}}{\|W_0 + \underline{B}\underline{A}\|_c}} $$

- $\underline{m}$: 훈련 가능한 magnitude vector (초기값: $\|W_0\|_c$)
- $V = W_0$: 동결된 방향 행렬 (초기화)
- $\Delta V = \underline{B}\underline{A}$: LoRA로 학습되는 방향성 업데이트
- $B$, $A$: LoRA와 동일한 초기화 방식 적용 → 초기에 $W' = W_0$ 보장

**훈련 가능 파라미터:** $m$ (크기, $1 \times k$) + $B$, $A$ (방향, LoRA)

#### Step 4: 분석을 위한 크기/방향 변화 측정 수식

FT 가중치의 magnitude 변화:

$$\Delta M^t_{\text{FT}} = \frac{\sum_{n=1}^{k} |m^{n,t}_{\text{FT}} - m^n_0|}{k} $$

FT 가중치의 direction 변화:

$$\Delta D^t_{\text{FT}} = \frac{\sum_{n=1}^{k}(1 - \cos(V^{n,t}_{\text{FT}}, W^n_0))}{k} $$

#### Step 5: DoRA의 그래디언트 분석

Loss $\mathcal{L}$에 대한 DoRA의 그래디언트 (Eq. 5로부터 유도):

$$\nabla_{V'}\mathcal{L} = \frac{m}{\|V'\|_c}\left(I - \frac{V'V'^T}{\|V'\|^2_c}\right)\nabla_{W'}\mathcal{L} $$

$$\nabla_m \mathcal{L} = \frac{\nabla_{W'}\mathcal{L} \cdot V'}{\|V'\|_c} $$

**해석:**
- Eq. (6): 그래디언트가 $m/\|V'\|_c$로 스케일링되고 현재 가중치 방향으로부터 투영(projection)됨 → 그래디언트 공분산 행렬이 단위행렬에 가까워져 최적화 유리
- $V' = V + \Delta V$이므로 $\nabla_{V'}\mathcal{L} = \nabla_{\Delta V}\mathcal{L}$ → LoRA 학습 안정성 향상

#### Step 6: 훈련 메모리 절감 (실용적 수정)

$\|V + \Delta V\|_c$를 그래디언트 그래프에서 분리(detach)하여 상수 $C$로 처리:

$$\nabla_{V'}\mathcal{L} = \frac{m}{C}\nabla_{W'}\mathcal{L} \quad \text{where } C = \|V'\|_c $$

→ LLaMA 파인튜닝 시 GPU 메모리 **24.4% 절감**, VL-BART 시 **12.4% 절감**, 정확도 손실은 무시할 수 있는 수준 (LLaMA: 0.2%, VL-BART: 0%)

---

### 2.3 모델 구조

DoRA의 구조적 특징을 정리하면 다음과 같습니다:

```
[사전학습 가중치 W₀]
        ↓ 분해 (Decompose)
┌─────────────────────────────────┐
│  magnitude m = ||W₀||_c  [훈련가능] │
│  direction V = W₀         [동결]   │
└─────────────────────────────────┘
        ↓ 방향 업데이트 (LoRA)
┌─────────────────────────────────┐
│  ΔV = BA  (B∈R^{d×r}, A∈R^{r×k}) [훈련가능] │
└─────────────────────────────────┘
        ↓ 병합 (Merge)
[추론 가중치] W' = m · (W₀ + BA) / ||W₀ + BA||_c
```

**핵심 특징:**
- **추론 시 오버헤드 없음**: 학습 후 $m$, $B$, $A$를 $W_0$에 병합 → 원본 모델과 동일한 구조
- **LoRA와 호환**: $\Delta V$ 부분을 VeRA 등 다른 LoRA 변형으로 대체 가능 (DVoRA)
- **QLoRA와 호환**: QDoRA로 확장 가능 (4-bit 양자화 기반)

---

### 2.4 성능 향상

#### 상식 추론 (Commonsense Reasoning) - LLaMA 계열

| 모델 | 방법 | 파라미터(%) | 평균 정확도 | LoRA 대비 향상 |
|------|------|------------|------------|---------------|
| LLaMA-7B | LoRA | 0.83 | 74.7 | - |
| LLaMA-7B | **DoRA** | 0.84 | **78.4** | **+3.7%** |
| LLaMA-7B | DoRA† (rank/2) | 0.43 | 77.5 | +2.8% |
| LLaMA-13B | LoRA | 0.67 | 80.5 | - |
| LLaMA-13B | **DoRA** | 0.68 | **81.5** | **+1.0%** |
| LLaMA2-7B | LoRA | 0.83 | 77.6 | - |
| LLaMA2-7B | **DoRA** | 0.84 | **79.7** (DoRA†: **80.5**) | **+2.9%** |
| LLaMA3-8B | LoRA | 0.70 | 80.8 | - |
| LLaMA3-8B | **DoRA** | 0.71 | **85.2** | **+4.4%** |

#### 이미지/비디오-텍스트 이해 (VL-BART)

| 태스크 | FT | LoRA | DoRA | DoRA vs LoRA |
|--------|-----|------|------|-------------|
| Image-Text Avg. | 77.3 | 76.5 | **77.4** | **+0.9%** |
| Video-Text Avg. | 87.5 | 83.5 | **85.4** | **+1.9%** |

#### 시각적 지시 튜닝 (LLaVA-1.5-7B)

| 방법 | 파라미터(%) | 평균 점수 |
|------|------------|----------|
| FT | 100 | 66.5 |
| LoRA | 4.61 | 66.9 |
| **DoRA** | 4.63 | **67.6** |

#### VeRA와의 호환성 (MT-Bench, LLaMA2-7B)

| 방법 | 파라미터(%) | MT-Bench 점수 |
|------|------------|--------------|
| VeRA | 0.02 | 5.5 |
| **DVoRA** | 0.04 | **6.0** |
| LoRA | 2.31 | 5.7 |
| **DoRA** | 2.33 | **6.0** |

---

### 2.5 한계

논문에서 명시적으로 또는 암묵적으로 확인되는 한계점들:

1. **분석 범위의 제한**: Weight Decomposition Analysis가 주로 self-attention의 query/value 행렬에 집중되어 있으며, MLP 레이어 등 다른 구성요소에 대한 분석이 부족

2. **음성(audio) 도메인 미검증**: 논문 결론부에서 직접 언급 — *"we wish to explore the generalizability of DoRA in domains beyond language and vision, particularly in the field of audio."*

3. **추가 하이퍼파라미터**: magnitude vector $m$을 위한 학습률 조정이 필요할 수 있으며, LoRA 대비 약간의 추가 튜닝이 필요

4. **메모리 오버헤드**: 수정 없이 사용 시 역전파 시 추가 메모리 필요 (수정 적용 시 해소되나 근사치 도입)

5. **FT가 이미 LoRA보다 열등한 경우**: LLaVA 실험에서 FT가 LoRA보다 낮은 점수를 보이는 상황에서 DoRA의 개선폭이 제한적 (과적합 억제 측면에서는 오히려 강점일 수 있으나 이론적 설명 보완 필요)

6. **높은 랭크에서의 성능 저하**: 표 15에서 DoRA(r=64)의 HellaSwag 정확도가 40.7%로 급락하는 현상이 관찰되어, 특정 조건에서의 안정성 문제 존재

---

## 3. 일반화 성능 향상 가능성 (중점 분석)

### 3.1 왜 DoRA가 더 나은 일반화를 달성하는가?

DoRA의 일반화 성능 향상은 다음 세 가지 메커니즘으로 설명됩니다:

#### (1) 사전학습 지식의 보존 (Pre-trained Knowledge Preservation)

DoRA로 파인튜닝된 가중치는 LoRA 대비 사전학습 가중치와의 편차가 **magnitude와 direction 모두에서** 훨씬 작습니다 (Figure 3, Figure 8 참조).

이는 다음 가설을 지지합니다:

> *"a robust foundation model does not require significant alterations for effective downstream adaptation"*

사전학습된 가중치가 이미 풍부한 일반 지식을 담고 있으므로, 소폭의 정밀한 조정만으로 충분하며, 이것이 오히려 더 나은 일반화로 이어집니다.

#### (2) 세밀한 크기/방향 독립 제어 (Decoupled Magnitude-Direction Control)

LoRA의 근본적 문제: magnitude와 direction 업데이트가 **결합(coupled)** 되어 있어 미세 조정이 어렵습니다.

DoRA의 핵심 이점: 두 성분을 **독립적으로** 최적화함으로써:

- **크게 방향만 바꾸고 크기는 유지** (예: 의미적 변화가 큰 태스크)
- **크기만 조정하고 방향은 유지** (예: 스케일 조정만 필요한 태스크)

이 유연성이 FT의 학습 패턴을 모방하며, 더 나은 태스크 적응과 일반화를 가능하게 합니다.

#### (3) 데이터 효율성 (Data Efficiency)

DoRA와 DVoRA는 **훈련 데이터 크기가 작을 때도** LoRA/VeRA보다 일관되게 우수한 성능을 보입니다 (Figure 4, 9):

| 훈련 샘플 수 | DoRA vs LoRA | DVoRA vs VeRA |
|------------|-------------|--------------|
| 1,000 | **+0.29** | **+0.22** |
| 4,000 | +0.27 | +0.28 |
| 7,000 | **+0.30** | **+0.33** |
| 10,000 | +0.30 | +0.50 |

이는 DoRA가 **제한된 데이터 환경에서도** 효과적으로 사전학습 지식을 활용한다는 것을 의미하며, 실제 산업 응용에서의 일반화 성능 우위를 뒷받침합니다.

#### (4) 랭크 견고성 (Rank Robustness)

낮은 랭크에서 DoRA의 일반화 성능 우위가 더욱 두드러집니다:

$$\text{DoRA}(r=8) = 77.9\% \quad \text{vs} \quad \text{LoRA}(r=8) = 40.7\%$$

$$\text{DoRA}(r=4) = 61.9\% \quad \text{vs} \quad \text{LoRA}(r=4) = 39.5\%$$

즉, DoRA는 **극히 적은 파라미터로도 의미 있는 일반화 성능**을 유지합니다. 이는 파라미터 효율성과 일반화 사이의 트레이드오프를 LoRA보다 훨씬 유리한 방향으로 이동시킵니다.

#### (5) 과적합 억제 (Overfitting Suppression)

LLaVA 실험에서 FT는 과적합으로 인해 LoRA보다 낮은 성능(66.5 vs 66.9)을 보이지만, DoRA는 67.6으로 두 방법 모두를 상회합니다. 이는 DoRA의 가중치 분해 구조가 불필요한 방향 변화를 억제하여 **정규화(regularization) 효과**를 갖는다는 것을 시사합니다.

#### (6) QDoRA: 메모리 제약 환경에서의 일반화

QDoRA(DoRA + QLoRA)는 LLaMA3-8B에서 Orca-Math 100k 샘플 파인튜닝 시:

$$\text{QDoRA} = 0.56 > \text{Full FT} = 0.51 > \text{QLoRA} = 0.32$$

4-bit 양자화 환경에서도 Full FT를 능가하는 일반화 성능을 보이며, 이는 DoRA의 가중치 분해 원리가 양자화된 모델에서도 일반화 성능 향상에 기여함을 의미합니다.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 PEFT 방법론 비교 표

| 방법 | 연도 | 핵심 아이디어 | 추론 오버헤드 | 파라미터 효율 | 일반화 성능 |
|------|------|-------------|-------------|-------------|------------|
| **Adapter** (Houlsby et al.) | 2019 | 레이어 간 추가 모듈 삽입 | **있음** | 중간 | 중간 |
| **Prefix-Tuning** (Li & Liang) | 2021 | 소프트 토큰 추가 | **있음** | 높음 | 초기화 민감 |
| **LoRA** (Hu et al.) | 2022 | 저랭크 행렬로 가중치 변화 근사 | **없음** | 높음 | 중간 |
| **AdaLoRA** (Zhang et al.) | 2023 | SVD 기반 동적 랭크 할당 | **없음** | 높음 | 중간-높음 |
| **VeRA** (Kopiczko et al.) | 2024 | 공유 랜덤 행렬 + 스케일링 벡터 | **없음** | **매우 높음** | 중간 |
| **DoRA** (Liu et al.) | 2024 | magnitude/direction 분해 + LoRA | **없음** | 높음 | **높음** |

### 4.2 DoRA vs LoRA: 이론적 차이

$$\text{LoRA: } W' = W_0 + BA$$

$$\text{DoRA: } W' = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}$$

LoRA는 가중치 변화를 **덧셈적(additive)** 으로 모델링하는 반면, DoRA는 **정규화된 방향 + 크기 스케일링**으로 모델링합니다. 이 구조적 차이가 Weight Normalization(Salimans & Kingma, 2016)의 최적화 이점을 파인튜닝에 이식합니다.

### 4.3 AdaLoRA와의 비교

AdaLoRA (Zhang et al., 2023)는 SVD를 통해 중요도에 따라 랭크를 동적으로 할당합니다:

$$W = P \Lambda Q^T$$

여기서 $\Lambda$의 작은 특이값은 제거하여 파라미터를 절약합니다. 반면 DoRA는 랭크를 고정하되 가중치 분해 방식을 바꿔 학습 패턴 자체를 개선하는 직교적 접근입니다. 두 방법은 상호 보완적으로 결합될 가능성이 있습니다.

### 4.4 VeRA와의 호환 (DVoRA)

$$\text{DVoRA: } W' = m \cdot \frac{W_0 + \Lambda_b B \Lambda_d A}{\|W_0 + \Lambda_b B \Lambda_d A\|_c}$$

VeRA의 공유 랜덤 행렬 $B$, $A$에 학습 가능한 스케일링 벡터 $\Lambda_b$, $\Lambda_d$를 적용한 것을 DoRA의 방향 업데이트로 사용합니다. 이로써 **0.04%의 파라미터만으로** LoRA(2.31%)와 동등한 성능 달성이 가능합니다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

#### (1) PEFT 패러다임의 재정립
DoRA는 "파라미터 수가 부족하기 때문에 LoRA가 FT보다 열등하다"는 기존 통념을 **학습 패턴의 질적 차이** 문제로 재프레임하였습니다. 이는 향후 PEFT 연구에서 단순히 훈련 가능 파라미터를 늘리는 방향이 아닌, **학습 역학(learning dynamics)** 자체를 개선하는 방향으로의 패러다임 전환을 촉진할 것입니다.

#### (2) 가중치 분해 분석 프레임워크의 확산
Weight Decomposition Analysis는 다른 PEFT 방법들(Prefix-Tuning, Adapter 등)의 학습 패턴을 분석하는 범용 도구로 활용될 수 있습니다. 이는 PEFT 방법의 이론적 이해를 심화하는 데 기여할 것입니다.

#### (3) QDoRA의 민주화 가능성
QDoRA는 소비자용 GPU에서 수십억 파라미터 모델을 파인튜닝할 수 있게 하며, 이는 오픈소스 커뮤니티와 학계의 연구 접근성을 크게 향상시킬 것입니다.

#### (4) 멀티모달 모델 파인튜닝의 표준화
LLaVA, VL-BART 등 다양한 멀티모달 모델에서의 성능 향상 입증은, DoRA가 멀티모달 파인튜닝의 **de facto 표준**으로 자리잡을 가능성을 시사합니다.

#### (5) 생성 모델로의 확장
SDXL DreamBooth 실험에서의 우수한 개인화 성능은, **텍스트-이미지 생성 모델** 파인튜닝 분야에서도 DoRA가 중요한 역할을 할 것임을 예고합니다.

---

### 5.2 앞으로 연구 시 고려할 점

#### (1) 음성(Audio) 도메인 확장 검증
논문 저자들이 명시한 미래 방향입니다. Whisper, EnCodec 등 음성 모델에서의 DoRA 적용 가능성과 magnitude/direction 분해 패턴이 텍스트/이미지와 유사한지 검증이 필요합니다.

#### (2) 높은 랭크에서의 불안정성 분석
$r=64$ 설정에서 DoRA의 HellaSwag 정확도가 40.7%로 급락하는 현상(표 15)은 해결되지 않은 문제입니다. 최적 랭크 선택 기준이나 적응적 랭크 조정(AdaLoRA와의 결합) 연구가 필요합니다.

$$\text{Consider: DoRA + AdaLoRA} \rightarrow \text{Adaptive rank DoRA}$$

#### (3) Magnitude 성분의 정규화 전략
현재 $m$은 단순히 훈련 가능한 벡터로 설정되어 있습니다. L1/L2 정규화나 스파스성(sparsity) 유도를 통해 더 효율적인 magnitude 학습이 가능한지 탐구할 수 있습니다.

#### (4) 레이어별 선택적 적용 (Tuning Granularity)
표 6에서 보듯, 모든 레이어에 동일하게 DoRA를 적용하는 것이 최적이 아닐 수 있습니다. 어떤 레이어에 magnitude만, 어떤 레이어에 direction도 업데이트할지를 자동으로 결정하는 **적응적 적용 방법** 개발이 중요합니다.

#### (5) 상관관계 이론의 정밀화
DoRA가 FT의 음의 $\Delta D$ - $\Delta M$ 상관관계를 완전히 재현하지는 못합니다 (-0.31 vs -0.62). 이 격차를 좁히기 위한 이론적 분석과 개선 방향 탐구가 필요합니다.

#### (6) 연속 학습(Continual Learning) 및 재난적 망각
DoRA의 가중치 분해 구조가 연속 학습 시나리오에서 재난적 망각(catastrophic forgetting)을 얼마나 억제하는지 체계적인 분석이 부족합니다. 이는 특히 다중 태스크 적응 연구에서 중요한 주제입니다.

#### (7) 이론적 수렴 보장 (Convergence Guarantee)
현재 DoRA의 이론적 분석은 주로 그래디언트 해석에 머물러 있습니다. 수렴 속도나 최적해 도달 가능성에 대한 엄밀한 이론적 보장을 제시하는 연구가 필요합니다.

#### (8) 다른 아키텍처로의 확장
현재 DoRA는 주로 Transformer 기반 모델에 적용됩니다. CNN, State Space Model (Mamba 등), GNN 등 다른 아키텍처에서의 적용 가능성과 효과 검증이 필요합니다.

---

## 요약 다이어그램

```
[FT 학습 패턴]          [LoRA 학습 패턴]       [DoRA 학습 패턴]
ΔM ↑ → ΔD ↓           ΔM ↑ → ΔD ↑           ΔM ↑ → ΔD ↓
(음의 상관, -0.62)      (양의 상관, +0.83)      (음의 상관, -0.31)
    ↓                      ↓                      ↓
세밀한 적응 가능         결합된 업데이트          FT에 근접한 패턴
우수한 일반화            제한적 학습능력          개선된 일반화
```

DoRA는 Weight Normalization의 최적화 이점을 파인튜닝에 이식함으로써, 추가 추론 비용 없이 LoRA의 학습 패턴을 FT에 근접하게 만드는 **원리적으로 타당하고 실용적으로 검증된** PEFT 방법입니다. 이 논문은 PEFT 분야에서 "왜 특정 방법이 더 잘 작동하는가"에 대한 깊은 통찰을 제공하며, 향후 PEFT 연구의 중요한 이론적·실용적 토대가 될 것입니다.
