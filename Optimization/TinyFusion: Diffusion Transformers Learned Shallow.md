# TinyFusion: Diffusion Transformers Learned Shallow

## 1. 핵심 주장과 주요 기여 요약

본 논문(Fang, Li, Ma, Wang, NUS, arXiv:2412.01199)은 사전학습된 Diffusion Transformer의 **깊이(depth)를 학습 가능한 방식으로 가지치기(pruning)** 하는 방법인 **TinyFusion**을 제안합니다. 핵심 주장은 다음과 같이 요약됩니다.

- **기존 가설의 한계 지적**: "가지치기 직후 보정 손실(calibration loss)이 작을수록 좋다"는 손실 최소화 원칙이 Diffusion Transformer에는 적합하지 않음을 100,000개 무작위 샘플 실험으로 입증.
- **회복 가능성(Recoverability) 중심의 새 패러다임**: 즉각적인 손실 최소화가 아니라, **fine-tuning 이후 성능**을 직접 모델링하고 최적화.
- **미분 가능한 샘플링(Gumbel-Softmax) + LoRA 기반 weight update** 의 결합으로, pruning과 fine-tuning이라는 비미분 가능한 두 과정을 종단간(end-to-end)으로 학습.
- **범용성**: DiT, MAR, SiT 세 가지 서로 다른 Diffusion Transformer 아키텍처에 동일한 프레임워크가 적용됨.
- **효율성**: DiT-XL 기준 사전학습 비용의 **7% 미만**으로 절반 깊이 모델(D14)을 만들어 **2× 속도 향상**과 **FID 2.86** 달성.

---

## 2. 상세 분석: 문제 · 방법 · 구조 · 성능 · 한계

### 2.1 해결하고자 하는 문제

Diffusion Transformer는 이미지·영상 생성에서 표준 아키텍처로 자리 잡았으나, 막대한 파라미터 수로 인해 추론 비용이 큽니다. 기존 압축 연구는 두 갈래입니다.

- **너비(width) pruning / sparsity** (Diff-Pruning, SparseDM 등): GPU 같은 병렬 장치에서는 실제 속도 이득이 제한적(50% 압축에서 약 1.6×).
- **깊이(depth) pruning** (ShortGPT, Flux-Lite, BK-SDM 등): 압축률에 거의 비례하는 선형 가속(50% 압축 시 약 2×) 가능. 그러나 **휴리스틱 중요도 기준** 또는 **수작업 스킴**에 의존하며 fine-tuning 후 성능 보장이 약함.

저자가 제기한 핵심 의문은 *"보정 손실 최소화가 정말로 최적의 가지치기 지표인가?"* 이며, 실험적으로 **No**임을 보였습니다(Table 3 참조).

### 2.2 제안 방법 (수식 포함)

**(1) 깊이 pruning의 기본 정식화.** $L$개 레이어를 가진 Transformer에서 이진 마스크 $\mathbf{m} = [m_1, m_2, \dots, m_L]^\top$로 레이어를 제거합니다.

$$x_{i+1} = m_i \, \phi_i(x_i) + (1 - m_i)\, x_i = \begin{cases} \phi_i(x_i), & \text{if } m_i = 1 \\ x_i, & \text{otherwise} \end{cases}$$

**(2) 기존 손실 최소화 vs. 회복 가능성 최적화.** 종래 방식은
$$\min_{\mathbf{m}} \; \mathbb{E}_x \left[ \mathcal{L}(x, \Phi, \mathbf{m}) \right]$$
이지만, 본 논문은 **fine-tuning 이후 성능**을 직접 목적함수에 내재화합니다.

$$\min_{\mathbf{m}} \; \min_{\Delta\Phi} \; \mathbb{E}_x \left[ \mathcal{L}(x, \Phi + \Delta\Phi, \mathbf{m}) \right]$$

여기서 $\Delta\Phi$는 fine-tuning에 해당하는 가중치 업데이트입니다.

**(3) 확률적 관점 + 국소 N:M 구조.** 모델을 $K$개의 비중첩 블록 $\Phi = [\Phi_1, \dots, \Phi_K]^\top$으로 분할하고, 각 블록에서 $M$개 중 $N$개를 유지하는 N:M 스킴을 사용합니다. 각 블록의 마스크는 독립 카테고리 분포를 따른다고 가정하여

$$p(\mathbf{m}) = p(\mathbf{m}_1)\cdot p(\mathbf{m}_2) \cdots p(\mathbf{m}_K)$$

예를 들어 2:3 스킴이면 후보 집합은 $\hat{\mathbf{m}}^{2:3} = [[1,1,0],[1,0,1],[0,1,1]]$이고 각 블록은 확률 $[p_{k1}, p_{k2}, p_{k3}]$를 가집니다.

**(4) Gumbel-Softmax 미분가능 샘플링.** Straight-Through Estimator(STE)와 Gumbel 노이즈 $g_i \sim \text{Gumbel}(0,1)$로 다음과 같이 샘플링합니다.

$$y = \text{one-hot}\!\left( \frac{\exp\!\left((g_i + \log p_i)/\tau\right)}{\sum_j \exp\!\left((g_j + \log p_j)/\tau\right)} \right)$$

샘플링된 인덱스 $y$로부터 마스크는

$$\mathbf{m} = y^\top \hat{\mathbf{m}}$$

로 얻어집니다. $\tau$는 온도 스케줄러로 점진적 감소.

**(5) LoRA 기반 회복 가능성 추정.** 각 선형 가중치 $\mathbf{W}$에 대해

$$\mathbf{W}_{\text{fine-tuned}} = \mathbf{W} + \alpha \Delta\mathbf{W} = \mathbf{W} + \alpha \mathbf{B}\mathbf{A}$$

LoRA를 사용하면 학습 파라미터가 전체 fine-tuning 대비 **약 0.9%** 수준으로 줄어 탐색이 효율적이며, 1:2 스킴에서 LoRA(FID 33.39)가 Full fine-tuning(35.77)보다도 우수했습니다.

**(6) 최종 종단간 목적함수.**

$$\min_{\{p(\mathbf{m}_k)\}} \; \min_{\Delta\Phi} \; \mathbb{E}_{x,\,\{\mathbf{m}_k \sim p(\mathbf{m}_k)\}} \! \left[ \mathcal{L}(x, \Phi + \Delta\Phi, \{\mathbf{m}_k\}) \right]$$

학습 후에는 각 블록에서 가장 높은 확률의 패턴을 채택하고 $\Delta\Phi$는 폐기, 일반 fine-tuning(또는 KD)으로 회복합니다.

**(7) Masked Knowledge Distillation.** 회복 단계에서는 다음 손실을 사용합니다.

$$\mathcal{L} = \alpha_{\text{KD}} \cdot \mathcal{L}_{\text{KD}} + \alpha_{\text{Diff}} \cdot \mathcal{L}_{\text{Diff}} + \beta \cdot \mathcal{L}_{\text{Rep}}$$

은닉 상태에는 "massive activations" 문제가 있어, 임계 조건 $|x - \mu_x| < k\sigma_x$ ($k = 2, 4$)을 만족하는 활성치만 사용해 표현을 정렬합니다. 이 단순한 마스킹만으로 RepKD가 발산(NaN)에서 FID 3.73으로 회복되는 것이 보고됩니다.

### 2.3 모델 구조

- **분할 단위**: 인접한 $M$개 레이어를 하나의 local block으로 묶음(논문 추천 1:2 또는 2:4).
- **샘플링 모듈**: 블록별 카테고리 분포 → Gumbel-Softmax → 이진 마스크.
- **회복 모듈**: 모든 레이어에 LoRA 어댑터를 부착해 공동 최적화.
- **추론 시**: 마스크 0인 레이어를 통째로 제거 → 실제 깊이 감소 → 직렬 연산 깊이 감소.

### 2.4 성능 향상

DiT-XL/2 기준 주요 결과(Table 1):

- **TinyDiT-D14 (KD, 500K steps)**: FID **2.86**, IS 234.50, 처리량 13.54 it/s — 원본 DiT-XL/2(FID 2.27, 6.91 it/s)와 비교해 약 0.6 FID 손실로 **약 2× 가속**.
- **TinyDiT-D14 (KD, 100K steps만)**: FID 3.73 — ShortGPT(22.28), Flux-Lite(25.92), Sensitivity Analysis(21.15) 같은 손실 최소화 기반 baseline 대비 **수 배 우수**.
- **TinyDiT-D7 (KD)**: FID 5.87, 26.81 it/s, 173M 파라미터.
- **선형 가속 입증**: Figure 4에서 깊이 pruning은 압축률에 비례한 선형 속도 향상 곡선에 거의 일치(50% → 2×, 75% → 4×).

다른 아키텍처(Table 2):

- **TinyMAR-D16**: FID **2.28** (40 epochs, 원본의 10%) — 24-블록 MAR-Base(FID 2.31)를 능가.
- **TinySiT-D14**: FID **3.02** (100 epochs, 원본의 7%).

### 2.5 한계 (논문 명시)

1. **클래스-조건 ImageNet 256×256 생성에 국한** — 텍스트→이미지 시나리오에 대한 체계적 분석 부재.
2. **블록 단위 제거**에 한정 — Attention과 MLP를 분리해 더 세밀하게 가지치기하는 전략은 미탐구.
3. **N:M 스킴이 클수록 탐색 공간이 폭발** — 7:14 스킴(3,432개 후보)은 1 epoch 내 안정적 수렴이 어려워 1:2 또는 2:4를 권장.
4. **Massive activation 문제**가 지식 증류 안정성에 치명적이어서 별도 마스킹이 필수.

---

## 3. 모델의 일반화 성능 향상 가능성 (집중 분석)

이 논문이 일반화 측면에서 유의미한 이유를 다음 네 가지로 정리할 수 있습니다.

**(i) 아키텍처 비종속성.** TinyFusion은 레이어 출력 $\phi_i(x_i)$에 게이트 $m_i$를 곱하는 매우 추상적인 인터페이스만 가정하므로, **잔차 연결(residual)** 을 가진 어떤 Transformer에도 형식상 그대로 적용됩니다. 실제로 논문은
- DiT (denoising score matching, AdaLN),
- SiT (flow-based interpolant),
- MAR (autoregressive + diffusion loss, bidirectional attention)

세 가지 학습 패러다임이 모두 다른 모델에 동일 코드로 작동함을 보였습니다. 이는 후속 연구가 Stable Diffusion 3, FLUX 같은 **MMDiT, 멀티모달 DiT**로 확장할 여지를 강하게 시사합니다 (실제로 PPCL 등 후속 연구가 이 방향으로 갔습니다).

**(ii) 회복 가능성 모델링이 일반화의 핵심.** 손실 최소화 기반 지표는 도메인·아키텍처마다 별도 설계가 필요했지만, "fine-tuning을 LoRA로 시뮬레이션"하는 방식은 데이터 분포·손실 형태에 무관해 새 아키텍처로 옮길 때 **하이퍼파라미터 재설계가 거의 불필요**합니다.

**(iii) N:M 국소 패턴이 탐색 공간을 일반화 친화적으로 줄임.** 글로벌 마스크($\binom{28}{14} = 40{,}116{,}600$ 후보)는 탐색이 사실상 불가능하지만, 블록 단위로 분해하면 각 블록은 $\binom{M}{N}$ 후보로 줄어 **데이터·모델 크기에 따라 적응적으로 조절** 가능합니다.

**(iv) PEFT(LoRA)와의 자연스러운 결합.** LoRA는 이미 LLM/DiT 모두에서 사실상 표준이 된 PEFT 기법이라, TinyFusion 파이프라인은 **사전학습된 LoRA 어댑터를 기반으로 한 adapter-aware pruning**으로 자연 확장될 가능성이 큽니다.

**일반화의 잠재적 약점도 함께 지적할 필요가 있습니다.**
- 텍스트→이미지에서는 cross-attention과 prompt embedding의 영향이 강해, 단순 잔차 게이팅만으로는 의미 정보 손실이 발생할 수 있습니다 (HierarchicalPrune 등 후속 연구가 "초반 블록 = 의미 / 후반 블록 = 디테일"의 위치별 민감도를 별도 처리하는 이유).
- Massive activation 마스킹의 임계값 $k$가 모델별로 달라질 수 있어, 거대 MMDiT(8B+)에 그대로 옮길 경우 추가 보정이 필요합니다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 영향

- **"Recoverability"라는 새 평가축 제시.** 이전 pruning 문헌이 거의 전적으로 의존하던 *post-prune calibration loss* 패러다임에 정면으로 의문을 제기하고, 실험적으로 반증한 점은 향후 모든 generative 모델 압축 연구가 베이스라인을 다시 검토하게 만든 의미 있는 기여입니다.
- **CVPR 2025 Highlight 채택**으로 후속 연구의 표준 비교 대상이 됨 (HierarchicalPrune, PPCL 등 2025–2026 논문이 모두 TinyFusion을 핵심 baseline으로 사용).
- **LoRA를 "fine-tuning 시뮬레이터"로 활용**하는 메타 아이디어는 quantization-aware training, distillation-aware NAS 등 다른 압축 분야에도 이식 가능성이 큼.

### 4.2 향후 연구 시 고려할 점

1. **블록 단위 가정의 완화**: Attention head, MLP, AdaLN modulation을 **이종(heterogeneous)** 단위로 따로 다루는 fine-grained 변형. 논문 본문에서도 한계로 명시.
2. **Timestep-wise / Block-wise 동적 pruning**: 후속 연구 LazyDiT, Dynamic-DiT는 정적 제거가 아닌 **timestep마다 다른 깊이**를 사용하는 방향으로 발전 — TinyFusion의 정적 가정과 결합하면 추가 가속 가능.
3. **Position-aware pruning**: HierarchicalPrune(arXiv:2508.04663)은 "초반 블록은 의미 정보, 후반 블록은 디테일"이라는 hierarchy를 활용. TinyFusion의 학습 가능 분포에 위치 사전(prior)을 도입하면 더 안정적일 가능성.
4. **대규모 MMDiT(FLUX, SD3.5, Qwen-Image) 확장**: PPCL 보고에 따르면 8B 규모에서 TinyFusion의 평균 성능 저하가 13.80%로, 작은 모델 대비 격차가 커집니다. 대형 멀티모달 모델용 별도 안정화 기법이 필요.
5. **Massive activation의 근본적 해결**: 단순 임계 마스킹 대신, attention sink·outlier-aware quantization 연구와의 통합.
6. **Pruning–Quantization–Distillation 공동 최적화**: 본 연구는 KD까지만 통합. INT4 quantization과 동시에 학습하는 joint pipeline은 미개척 영역.
7. **이론적 분석 부재**: "왜 회복 가능성이 calibration loss보다 좋은 지표인가"에 대한 이론적 보장이 없음 — generalization bound, NTK 관점 분석이 후속 과제.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연도 | 방법 | 대상 | 핵심 아이디어 | TinyFusion과의 차이 |
|---|---|---|---|---|
| 2023 | **Diff-Pruning** (Fang et al., NeurIPS 2023) | U-Net 기반 DM | $\Delta \mathcal{L} \approx \sum_t \nabla_W \mathcal{L}_t \cdot \Delta W$ — pruned timesteps에 대한 Taylor 전개로 너비 pruning | 너비 중심, GPU 가속 1.6× 한정 |
| 2023 | **BK-SDM** (Kim et al., ICML W) | Stable Diffusion U-Net | 휴리스틱 블록 제거 + 증류 | 수작업 스킴, 학습 불가능 |
| 2024 | **ShortGPT** (Men et al.) | LLM | Block Influence 점수 $\text{BI}\_i = 1 - \mathbb{E}\left[\frac{x_i^\top x_{i+1}}{\|x_i\|\|x_{i+1}\|}\right]$ | 정적 중요도, calibration loss 의존 |
| 2024 | **LD-Pruner** (Castells et al., CVPR W) | Latent DM | task-agnostic operator score | 너비 pruning, 회복성 미고려 |
| 2024 | **Flux-Lite** (Daniel Verdu) | FLUX.1-dev | MSE 기반 layer similarity로 절반 제거 + 증류 | calibration MSE 최소화, 학습 가능 분포 없음 |
| 2024 | **SparseDM** (Wang et al.) | DM | STE 기반 50% sparsity | 비구조적, 실속도 이득 제한 |
| 2024 | **LAPTOP-Diff** (Zhang et al.) | SDXL/SDM | output-loss 기반 layer pruning + normalized distillation | U-Net 한정, additive 가정 |
| 2024 | **LazyDiT** (arXiv:2412.12444) | DiT | timestep 간 redundancy 활용한 lazy 재사용 | 동적 추론, 깊이 자체는 유지 |
| **2024.12** | **TinyFusion** (본 논문, CVPR 2025 Highlight) | DiT/MAR/SiT | $\min_{p(\mathbf{m})}\min_{\Delta\Phi}\mathbb{E}[\mathcal{L}(\Phi+\Delta\Phi,\mathbf{m})]$ — Gumbel-Softmax + LoRA | **회복가능성 직접 최적화, 학습 가능** |
| 2025 (ICLR) | **Dynamic-DiT** | DiT | timestep·patch별 동적 너비 | TinyFusion의 정적 깊이 vs. 동적 너비 — 보완적 |
| 2025 | **HierarchicalPrune** (arXiv:2508.04663) | SD3.5 Large Turbo | HPP + PWP + SGDistill + INT4 | 위치 사전 + 양자화 통합, 대규모 MMDiT |
| 2025 | **PPCL** (arXiv:2511.16156) | MMDiT (FLUX, Qwen-Image) | linear probing + 1차 미분 trend로 redundant 구간 식별 + plug-and-play 증류 | 8B 모델에서 TinyFusion(13.80% 저하) 대비 4.03% 저하 — 대형 모델로 확장 |
| 2025 | **EntPruner** (arXiv:2511.21122) | DiT/SiT | 엔트로피 기반 layer 분포 편차 분석 | 적응적 stage-wise pruning |

**핵심 정성 비교:**

- TinyFusion 이전 방법들은 거의 전부 $\min_{\mathbf{m}} \mathbb{E}[\mathcal{L}(\Phi, \mathbf{m})]$ 형태의 즉시 손실 최소화에 머물렀습니다. TinyFusion은 inner $\min_{\Delta\Phi}$ 항을 추가해 이 패러다임을 바꿉니다.
- 후속(2025–) 연구들은 TinyFusion을 표준 비교 baseline으로 채택하면서, 주로 **(a) 대규모 MMDiT 확장, (b) 위치별 민감도 활용, (c) 양자화·증류와의 공동 최적화** 방향으로 발전 중입니다.
- 직교적 흐름인 LazyDiT/Dynamic-DiT는 "정적 깊이 제거" 대신 "동적 사용"을 주장 — TinyFusion과 결합 가능한 보완재.

---

## 참고 자료

1. **본 논문**: Fang, Li, Ma, Wang. *TinyFusion: Diffusion Transformers Learned Shallow.* arXiv:2412.01199v1, 2024. https://arxiv.org/abs/2412.01199
2. **공식 코드**: https://github.com/VainF/TinyFusion (CVPR 2025 Highlight)
3. **Diff-Pruning**: Fang et al., *Structural Pruning for Diffusion Models*, NeurIPS 2023. arXiv:2305.10924
4. **BK-SDM**: Kim et al., *BK-SDM: Architecturally Compressed Stable Diffusion*, ICML Workshop 2023.
5. **ShortGPT**: Men et al., arXiv:2403.03853, 2024.
6. **LD-Pruner**: Castells et al., CVPR Workshop 2024.
7. **LAPTOP-Diff**: Zhang et al., arXiv:2404.11098, 2024.
8. **SparseDM**: Wang et al., arXiv:2404.10445, 2024.
9. **LazyDiT**: arXiv:2412.12444, 2024.
10. **Dynamic Diffusion Transformer**: ICLR 2025.
11. **HierarchicalPrune**: arXiv:2508.04663, 2025.
12. **PPCL (Pluggable Pruning with Contiguous Layer Distillation)**: arXiv:2511.16156, 2025.
13. **EntPruner**: arXiv:2511.21122, 2025.
14. **LoRA**: Hu et al., ICLR 2022.
15. **Gumbel-Softmax**: Jang, Gu, Poole, arXiv:1611.01144, 2016.
16. **Massive Activations**: Sun et al., arXiv:2402.17762, 2024.
17. **DiT**: Peebles & Xie, ICCV 2023.
18. **MAR**: Li et al., arXiv:2406.11838, 2024.
19. **SiT**: Ma et al., arXiv:2401.08740, 2024.

> 참고: 본 분석에서 논문 본문 내용(수식, 표 수치, 한계 명시 등)은 업로드된 PDF에 직접 근거하며, 후속 연구(HierarchicalPrune, PPCL, EntPruner, LazyDiT, Dynamic-DiT 등)와의 비교는 위 출처들의 공개 초록·도입부에서 확인 가능한 사항만 인용했습니다. PPCL이 보고한 "8B 규모에서 TinyFusion 13.80% 평균 성능 저하"와 같은 정량 수치는 PPCL 논문의 자체 측정이므로 절대치 해석에는 주의가 필요합니다.
