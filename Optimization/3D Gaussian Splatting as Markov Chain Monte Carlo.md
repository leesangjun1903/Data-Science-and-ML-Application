# 3D Gaussian Splatting as Markov Chain Monte Carlo

논문의 본문과 부록, 그리고 최근 후속 연구들을 종합하여 분석한 결과를 정리합니다.

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 **3D Gaussian Splatting(3DGS)의 학습 과정을 결정론적 휴리스틱이 아닌 확률 분포로부터의 MCMC 샘플링으로 재해석**한 것이 핵심입니다.

주요 기여를 압축하면 다음과 같습니다.

- **3DGS와 MCMC의 연결성 규명**: 기존 3DGS의 가우시안 업데이트 식이 노이즈 항만 추가하면 SGLD(Stochastic Gradient Langevin Dynamics) 업데이트와 동일해진다는 점을 수식적으로 보임.
- **휴리스틱 제거**: clone, split, prune, opacity reset 같은 수동 조정 규칙들을 "샘플 확률을 보존하는 결정론적 상태 전이(state transition)"로 대체.
- **재배치(relocalization) 전략**: 1D 슬라이스 기반의 새로운 cloning 식을 유도하여 $P(g^{new}) \approx P(g^{old})$를 만족하도록 함.
- **L1 정규화**: opacity와 covariance 양쪽에 정규화를 가해 가우시안 개수를 효율적으로 절감.
- **초기화 강건성**: SfM 점군 없이 random 초기화로도 성능이 거의 동일하게 유지.

---

## 2. 문제 정의 · 제안 방법 · 모델 구조 · 성능 · 한계

### 2-1. 해결하고자 하는 문제

기존 3DGS는 가우시안 배치를 위해 **Adaptive Density Control (ADC)** 라는 정교하게 튜닝된 휴리스틱에 의존합니다. 그 결과:

- **초기화 의존성** — 좋은 SfM 점군이 없으면 품질이 급격히 저하됨.
- **하이퍼파라미터 민감성** — clone/split threshold, opacity reset 시점 등 다수의 수동 튜닝 필요.
- **메모리/연산 예산 통제 어려움** — 학습 후 가우시안 개수를 사전에 예측 불가.
- **장면 일반화 실패** — 어떤 장면에서는 sub-optimal한 가우시안 배치로 quality 저하.

### 2-2. 제안 방법: 3DGS를 MCMC로 재정의

**(a) 표준 3DGS 업데이트 식**

$$g \leftarrow g - \lambda_{lr} \cdot \nabla_g \mathbb{E}_{I \sim \mathcal{I}}[\mathcal{L}_{total}(g; I)]$$

**(b) SGLD 일반 형태**

$$g \leftarrow g + a \cdot \nabla_g \log P(g) + b \cdot \epsilon$$

저자는 두 식을 비교하여, 우도 분포를 다음과 같이 정의하면 두 식이 본질적으로 동치임을 보입니다.

$$\mathcal{G} = P \propto \exp(-\mathcal{L}_{total})$$

즉, **3DGS 학습은 이미 암묵적으로 MCMC 샘플링을 수행하고 있었으며, 단지 탐색(exploration)을 위한 노이즈 항이 빠져 있었을 뿐**이라는 통찰입니다.

**(c) SGLD 기반 새로운 업데이트 식**

$$g \leftarrow g - \lambda_{lr} \cdot \nabla_g \mathbb{E}_{I \sim \mathcal{I}}[\mathcal{L}_{total}(g; I)] + \lambda_{noise} \cdot \epsilon$$

**(d) 노이즈 설계** — 위치 $\mu$에만, 공분산과 opacity에 의존적으로 추가:

$$\epsilon_\mu = \lambda_{lr} \cdot \sigma\big(-k(t-o)\big) \cdot \Sigma \eta, \quad \epsilon = [\epsilon_\mu, 0]$$

여기서 $\eta \sim \mathcal{N}(0, I)$, $\sigma$는 sigmoid, $k=100$, $t=0.005$. opacity가 충분히 높은 "잘 학습된" 가우시안에는 노이즈 영향이 줄어들고, opacity가 낮은 "탐색 중인" 가우시안에 큰 noise가 가해집니다.

**(e) Cloning을 확률 보존 상태 전이로 재정의**

기존 3DGS의 cloning은 가우시안의 합성 결과(rasterization)를 크게 변형시켜 $P(g^{new}) \neq P(g^{old})$가 됩니다. 저자는 1D slice 기반 sliced Wasserstein 아이디어로 다음 update 식을 유도합니다.

$$\mu^{new}_{1,\dots,N} = \mu^{old}_N$$

$$o^{new}_{1,\dots,N} = 1 - \sqrt[N]{1 - o^{old}_N}$$

$$\Sigma^{new}_{1,\dots,N} = \left(o^{old}_N\right)^2 \left( \sum_{i=1}^N \sum_{k=0}^{i-1} \binom{i-1}{k}(-1)^k \frac{(o^{new}_N)^{k+1}}{\sqrt{k+1}} \right)^{-2} \Sigma^{old}_N$$

이 식은 cloning 전후의 적분(즉, 임의 슬라이스에서의 누적 contribution)이 보존되도록 하므로, MCMC chain을 깨지 않고 가우시안 개수를 변경할 수 있습니다. **이는 [4] (Bulò et al., ECCV 2024) 의 center-corrected cloning이 중심 픽셀만 보존하던 한계를 넘어 가우시안 전체 형태를 보존한다**는 점에서 본질적으로 다릅니다 (Figure 1 참조).

**(f) 효율성 정규화** — opacity와 covariance eigenvalue에 L1 정규화:

$$\mathcal{L}_{total} = (1-\lambda_{D\text{-}SSIM}) \mathcal{L}_1 + \lambda_{D\text{-}SSIM} \mathcal{L}_{D\text{-}SSIM} + \lambda_o \sum_i |o_i|_1 + \lambda_\Sigma \sum_{ij} \big| \sqrt{eig_j(\Sigma_i)} \big|_1$$

### 2-3. 모델 구조

**렌더링 표현 자체는 기존 3DGS와 동일**합니다. 즉, 추론(inference) 시점에는 같은 raster pipeline을 사용하므로 속도가 동일합니다. 학습 파이프라인만 다음과 같이 바뀝니다.

1. 초기화 (random 또는 SfM)
2. SGLD 업데이트 (기존 gradient + noise term)
3. 100 iter마다 dead Gaussian → live Gaussian으로 multinomial 샘플링 기반 재배치
4. L1 정규화로 불필요 가우시안 자연 소멸
5. 5%씩 점진적으로 live 가우시안 수 증가

### 2-4. 성능 향상

논문 Table 1, Table 5 결과 요약 (동일 가우시안 수 기준):

| 데이터셋 | 3DGS (Random) | Ours (Random) | 3DGS (SfM) | Ours (SfM) |
|---|---|---|---|---|
| MipNeRF 360 | 27.89 / 0.84 / 0.26 | **29.72 / 0.89 / 0.19** | 29.30 / 0.88 / 0.21 | **29.89 / 0.90 / 0.19** |
| Tank & Temples | 21.93 / 0.79 / 0.27 | **24.21 / 0.86 / 0.19** | 23.67 / 0.84 / 0.22 | **24.29 / 0.86 / 0.19** |
| OMMO | 28.24 / 0.88 / 0.24 | **29.31 / 0.90 / 0.20** | 28.83 / 0.89 / 0.22 | **29.52 / 0.91 / 0.20** |

특히 **MipNeRF 360에서 PSNR 29.72는 NeRF 계열 백본을 처음으로 능가한 3DGS 결과**이며, 저자는 이를 결론에서 강조합니다. 또한 가우시안 예산을 제한했을 때 3DGS와의 성능 격차가 더 벌어지며 (Fig. 3), 3× → 1× camera extent로 초기화 영역을 줄이는 robustness 실험에서 3DGS는 27.89 → 22.72로 급락하는 반면 본 방법은 29.72 → 29.64로 거의 변화가 없습니다 (Table 2).

### 2-5. 한계 (논문에서 명시한 부분)

- **3DGS의 모델링 한계 자체는 그대로 상속**: aliasing, reflection 모델링은 여전히 [48] (Mip-Splatting), [44] (Deferred Reflection) 같은 별도 기법이 필요.
- **재배치가 정확한 분포 보존이 아닌 근사** ( $P(g^{new}) \approx P(g^{old})$ )이므로 100 iter마다 적용하는 보수적 스케줄 필요.
- **동일 PSNR 도달까지의 학습 시간이 약간 더 길 수 있음** (단, 같은 학습량 대비 quality는 더 높음, Table 4).
- **가우시안 총 수를 사전에 고정**하는 한계는 그대로 남아 있음 — 후속 연구 MH-3DGS에서 비판되는 지점.

---

## 3. 일반화 성능 향상 가능성 (집중 분석)

이 논문이 일반화 측면에서 가지는 의미는 다음과 같이 정리할 수 있습니다.

**(1) 초기화 의존성 제거 → "장면 다양성"에 대한 일반화**

기존 3DGS는 SfM 점군 품질이 곧 reconstruction 품질이었습니다. 텍스쳐가 적거나 반사가 강한 장면, 야외 대규모 장면(OMMO)에서 SfM이 실패하면 3DGS도 실패합니다. 본 논문은 SGLD 노이즈가 **명시적인 탐색(exploration) 메커니즘**을 제공하므로, 초기 점이 부정확하거나 sparse해도 가우시안들이 scene support로 자율적으로 이동합니다. Table 2의 robustness 실험이 이를 직접 증명합니다.

**(2) 휴리스틱 의존 제거 → "도메인 전이"에 대한 일반화**

ADC의 threshold들은 Mip-NeRF 360 같은 일반적 풍경에 맞춰 튜닝된 값입니다. 동적 장면(4DGS), 의료 영상, 인간 아바타, 위성 영상 등 다른 도메인에서는 이 threshold가 최적이 아닐 가능성이 큽니다. **MCMC 프레임워크는 도메인에 무관하게 "가우시안 = 분포에서의 샘플"이라는 단일 원리만 사용**하므로 도메인 전이 시 재튜닝 부담이 적습니다.

**(3) 다른 3DGS 확장과 직교적(orthogonal)**

저자가 부록 C에서 명시한 대로, 본 방법은 기존의 representation 자체를 변경하지 않으므로 Mip-Splatting (anti-aliasing), Deferred Reflection (반사), 4DGS (동적 장면), HUGS (아바타) 등 모든 후속 확장과 결합 가능합니다. 실제로 후속 연구들은 본 논문을 **"better training framework"** 로 받아들이고 있으며, 예를 들어 *AuGS* 등 NeRF를 능가하는 후속 연구들은 "3DGS-MCMC framework 위에서 실험"한다고 명시합니다. (출처: [3D Gaussian Splatting as Markov Chain Monte Carlo | Request PDF, ResearchGate](https://www.researchgate.net/publication/397201252))

**(4) 한계로 남은 일반화 이슈**

- 가우시안 총 수를 사전에 고정해야 하므로 **장면 복잡도가 매우 다양한 환경**에서는 여전히 사전 예산 결정 부담이 남음.
- Sparse view, few-shot 시나리오에서는 단순한 SGLD 탐색으로는 부족할 수 있고, 별도의 prior가 필요.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4-1. 학계에 미친 영향 (실제 관찰된 흐름)

- **NeurIPS 2024 Spotlight** 발표 논문으로, 출판 직후부터 3DGS 학습 프레임워크의 새로운 표준 중 하나로 자리 잡았습니다. (출처: [NeurIPS Poster - 3D Gaussian Splatting as Markov Chain Monte Carlo](https://neurips.cc/virtual/2024/poster/94984), [GitHub - ubc-vision/3dgs-mcmc](https://github.com/ubc-vision/3dgs-mcmc))
- 오픈소스 라이브러리 `gsplat` (Ye et al., 2024)이 본 방법을 통합하여 학습 시간 20%, 메모리 65% 감소를 독립적으로 검증.
- 후속 연구의 **공통 baseline 또는 building block**으로 사용됨 — Metropolis-Hastings 3DGS, SteepGS, ImprovedGS+, Compact Relightable 3DGS 등.

### 4-2. 향후 연구 시 고려할 점

**(1) 가우시안 수의 적응적 조정**
본 방법은 가우시안 총 수를 사전에 정해야 합니다. 가우시안 업데이트를 SGLD로 재정의하면서 휴리스틱을 상태 전이로 대체했지만, 여전히 전체 가우시안 수를 사전에 고정한다는 점이 장면 복잡도 변동에 대한 적응성을 제한한다는 비판이 후속 연구에서 제기되고 있습니다 (출처: Hyunjin Kim et al., *Metropolis-Hastings Sampling for 3D Gaussian Reconstruction*, arXiv:2506.12945). 향후 연구는 acceptance ratio 기반의 진정한 가역 점프 MCMC(RJMCMC)로 확장하는 방향이 유망합니다.

**(2) Error-aware sampling과의 결합**
저자도 본문에서 명시했듯, [4]의 error-based densification은 본 방법과 직교적이며 결합 가능합니다. *Pixel-GS, Perceptual-GS, Edge-Aware Score* 같은 인지/오류 기반 기법들과의 통합은 자연스러운 다음 단계입니다 (출처: [Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering, arXiv:2508.12313](https://arxiv.org/html/2508.12313)).

**(3) 노이즈 스케줄러 설계**
논문 ablation에서 exponential 스케줄러는 PSNR 24.21, linear는 17.64, [30]의 스케줄러는 22.46으로 큰 차이를 보였습니다. 노이즈 스케줄링 자체가 별도의 연구 주제가 될 수 있으며, **장면별 adaptive 노이즈** 설계는 미해결 과제입니다.

**(4) 이론적 보강**
$P(g^{new}) \approx P(g^{old})$는 근사이며, 정확한 detailed balance 보장이 없습니다. 이로 인해 100-iter 주기로 보수적으로 적용해야 합니다. 정확한 acceptance 단계를 도입한 Metropolis–Hastings 변형(MH-3DGS) 같은 시도가 이미 등장했으며, 이론적 정합성을 강화하면서 수렴 속도를 높이는 방향이 활발히 연구 중입니다.

**(5) 도메인 확장**
저자가 broader impact에서 언급한 대로 3D 생성, 동적 장면, 인간 아바타로의 확장이 자연스럽습니다. 단, 생성 모델과 결합 시 **noise 항이 generative noise와 entangle되는 문제**를 주의해야 합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연도/논문 | 핵심 아이디어 | 본 논문과의 관계 |
|---|---|---|
| **NeRF (Mildenhall et al., ECCV 2020)** | implicit neural field + volume rendering | 본 논문이 비교 baseline으로 사용; 3DGS-MCMC가 MipNeRF 360에서 NeRF 계열을 처음 능가 |
| **3DGS (Kerbl et al., SIGGRAPH 2023)** | 3D 가우시안 + α-blending raster | **본 논문의 직접적 base**; ADC 휴리스틱을 본 논문이 대체 |
| **Soft Mining for NeRF (Kheradmand et al., CVPR 2024) [20]** | NeRF 학습 가속을 위해 SGLD 사용 | 동일 저자의 선행 연구; SGLD를 sample selection에 사용 vs 본 논문은 representation 자체에 적용 |
| **Revising Densification (Bulò et al., ECCV 2024) [4]** | center-corrected cloning, error-based densification | 가장 가까운 동시 연구; 중심 보존만 → 본 논문은 슬라이스 적분 보존(Fig. 1) |
| **Mini-Splatting (Fang & Wang, ECCV 2024)** | constrained 가우시안 수 | 가우시안 효율성 측면에서 상호 보완적 |
| **DUSt3R (Wang et al., CVPR 2024) [37], InstantSplat [9]** | dense geometry estimator로 초기화 개선 | 초기화 측 접근; 본 논문은 학습 측에서 초기화 의존성 자체를 제거 |
| **gsplat library (Ye et al., 2024) [45]** | 오픈소스 3DGS 라이브러리 | 본 방법을 통합하여 20% 학습시간 / 65% 메모리 절감 독립 검증 |
| **SteepGS (Wang et al., CVPR 2025)** | splitting matrix로 split 결정 | densification의 또 다른 원리적 접근; MCMC와 상보적 |
| **Pixel-GS (Zhang et al., ECCV 2024)** | pixel-aware gradient 기반 density control | error 기반 densification 흐름 |
| **MH-3DGS (Kim et al., 2025, arXiv:2506.12945)** | Metropolis-Hastings acceptance ratio 도입 | 본 논문의 SGLD가 가우시안 총 수를 사전 고정하는 한계를 극복하고자 closed-form Bayesian posterior와 photometric surrogate를 acceptance ratio로 결합한 후속 연구 |
| **Compact Relightable 3DGS (SIGGRAPH Asia 2025)** | MCMC 기반 가우시안 필터링 + gradient-aware light sampling으로 86% 가우시안 감소, 60배 학습 가속 | 본 논문의 MCMC 프레임워크를 relighting으로 확장 |
| **AuGS / Augmented 3DGS (2025)** | 3DGS-MCMC를 학습 프레임워크로 사용하여 NeRF 계열을 능가하는 rendering quality 달성 | 본 논문을 building block으로 활용 |
| **Perceptual-GS (Zhou & Ni, 2025)** | scene-adaptive perceptual densification | error-driven densification 계열 |

이 흐름을 보면 본 논문은 **"3DGS densification 연구의 분기점"** 역할을 합니다. 이전에는 다양한 휴리스틱 변형이 산발적으로 제안되었으나, 본 논문 이후 연구들은 **확률적 프레임워크 (MCMC, Bayesian posterior, MH sampling)** 위에서 논의를 전개하는 경향이 뚜렷합니다.

---

## 참고 자료 (출처)

1. Kheradmand, S. et al. *3D Gaussian Splatting as Markov Chain Monte Carlo*. NeurIPS 2024 (Spotlight). arXiv:2404.09591v3 — 본 분석의 1차 자료 (사용자가 업로드한 PDF)
2. arXiv 페이지: https://arxiv.org/abs/2404.09591
3. 공식 프로젝트 페이지: https://ubc-vision.github.io/3dgs-mcmc/
4. 공식 GitHub 저장소: https://github.com/ubc-vision/3dgs-mcmc
5. NeurIPS 2024 Poster page: https://neurips.cc/virtual/2024/poster/94984
6. Kim, H. et al. *Metropolis-Hastings Sampling for 3D Gaussian Reconstruction*. arXiv:2506.12945 — 후속 비교 연구
7. Deng, X. et al. *Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering*. arXiv:2508.12313
8. *Markov Chain Monte Carlo-Guided Compact 3D Gaussian Splatting for Relightable Rendering*, SIGGRAPH Asia 2025 Technical Communications: https://dl.acm.org/doi/10.1145/3757376.3771401
9. Bernhard Kerbl, *The Impact and Outlook of 3D Gaussian Splatting*, arXiv:2510.26694
10. Bulò, S.R., Porzi, L., Kontschieder, P. *Revising densification in gaussian splatting*. ECCV 2024
11. Ye, V. et al. *gsplat: An Open-Source Library for Gaussian Splatting*. arXiv (2024)
12. ResearchGate citation analysis: https://www.researchgate.net/publication/397201252

---

**부기**: 위 분석은 사용자가 업로드한 논문 PDF 본문(서론·방법·실험·부록 A의 수식 유도 포함)에 직접 근거하여 작성되었으며, 후속 연구 비교 부분은 web search 결과를 참조하여 보충했습니다. 수식은 모두 본 논문의 식 (1)~(10)과 부록 A의 식 (11)~(18)을 따릅니다. 후속 연구 인용 시 정확한 수치(예: 86% 감소, 60배 가속, 20%/65% 절감 등)는 위 출처의 원문을 직접 인용하였으며, 임의로 보강한 부분은 없습니다.
