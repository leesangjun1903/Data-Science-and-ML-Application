# Domain Separation Networks (DSN)

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

기존 도메인 적응(Domain Adaptation) 연구들은 소스-타겟 도메인 간 **공유 표현(shared representation)** 만을 학습하는 데 집중하였으나, 이는 각 도메인 고유의 특성을 무시한다는 문제가 있습니다. DSN은 다음 가설을 제안합니다:

> **"각 도메인에 고유한 정보(private)를 명시적으로 모델링하면, 도메인 불변 특징(domain-invariant features)을 더 효과적으로 추출할 수 있다."**

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **Private-Shared 분리 구조** | 표현 공간을 도메인 공유(shared) + 도메인 고유(private) 서브스페이스로 명시 분리 |
| **Difference Loss** | Soft 직교성 제약으로 shared/private 표현의 독립성 보장 |
| **Scale-Invariant 재구성 손실** | 색상·밝기 절대값이 아닌 픽셀 쌍 간 차이를 페널티로 부여 |
| **해석 가능성** | Private/Shared 표현을 시각화하여 도메인 적응 과정 해석 가능 |
| **SOTA 성능** | 4개 벤치마크(MNIST→MNIST-M, Synth→SVHN, SVHN→MNIST, Synth Signs→GTSRB)에서 당시 최고 성능 달성 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**문제 배경:**
- 합성 데이터(synthetic data)로 학습한 모델은 실제 데이터(real data)에서 성능이 저하되는 **도메인 시프트(domain shift)** 문제가 존재
- 기존 방법(DANN, MMD 등)은 shared representation만 정렬하므로, shared 표현이 각 도메인 특유의 노이즈에 오염될 위험이 있음 (Salzmann et al. [24]의 지적)

**설정:**
- 소스 도메인: 레이블이 있는 데이터 $\mathbf{X}_S = \{(\mathbf{x}_i^s, \mathbf{y}_i^s)\}\_{i=0}^{N_s}$, $\mathbf{x}_i^s \sim \mathcal{D}_S$
- 타겟 도메인: 레이블이 없는 데이터 $\mathbf{X}^t = \{\mathbf{x}_i^t\}\_{i=0}^{N_t}$, $\mathbf{x}_i^t \sim \mathcal{D}_T$
- 목표: 소스에서 학습한 분류기가 타겟 도메인에서도 일반화

---

### 2.2 제안하는 방법 (수식 포함)

#### 전체 손실 함수

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \alpha \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{difference}} + \gamma \mathcal{L}_{\text{similarity}} $$

$\alpha, \beta, \gamma$는 각 손실 항의 가중치 하이퍼파라미터입니다.

---

#### (1) Task Loss (분류 손실)

소스 도메인에만 적용되는 교차 엔트로피 손실:

$$\mathcal{L}_{\text{task}} = -\sum_{i=0}^{N_s} \mathbf{y}_i^s \cdot \log \hat{\mathbf{y}}_i^s $$

여기서 $\hat{\mathbf{y}}_i^s = G(E_c(\mathbf{x}_i^s))$, $\mathbf{y}_i^s$는 one-hot 레이블.

---

#### (2) Reconstruction Loss (재구성 손실)

양 도메인 모두에 적용되는 **Scale-Invariant MSE**:

$$\mathcal{L}_{\text{recon}} = \sum_{i=1}^{N_s} \mathcal{L}_{\text{si mse}}(\mathbf{x}_i^s, \hat{\mathbf{x}}_i^s) + \sum_{i=1}^{N_t} \mathcal{L}_{\text{si mse}}(\mathbf{x}_i^t, \hat{\mathbf{x}}_i^t) $$

$$\mathcal{L}_{\text{si mse}}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{k}\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 - \frac{1}{k^2}\left([\mathbf{x} - \hat{\mathbf{x}}] \cdot \mathbf{1}_k\right)^2 $$

- $k$: 입력 픽셀 수, $\mathbf{1}_k$: 길이 $k$의 모두 1인 벡터
- 일반 MSE 대비 절대적 밝기/색상이 아닌 **픽셀 쌍 간 상대적 차이**를 페널티로 부여 → 전체적인 형태 학습에 유리

---

#### (3) Difference Loss (직교성 손실)

Private 표현과 Shared 표현 간의 직교성을 강제하는 **Soft Subspace Orthogonality Constraint**:

$$\mathcal{L}_{\text{difference}} = \left\|\mathbf{H}_c^{s\top} \mathbf{H}_p^s\right\|_F^2 + \left\|\mathbf{H}_c^{t\top} \mathbf{H}_p^t\right\|_F^2 $$

- $\mathbf{H}_c^s$, $\mathbf{H}_c^t$: 소스/타겟 shared 표현 행렬 (행 = $\mathbf{h}_c^s = E_c(\mathbf{x}^s)$, $\mathbf{h}_c^t = E_c(\mathbf{x}^t)$ )
- $\mathbf{H}_p^s$, $\mathbf{H}_p^t$: 소스/타겟 private 표현 행렬 (행 = $\mathbf{h}_p^s = E_p^s(\mathbf{x}^s)$, $\mathbf{h}_p^t = E_p^t(\mathbf{x}^t)$ )
- $\|\cdot\|_F^2$: Frobenius norm의 제곱
- 행렬은 zero mean, unit $\ell_2$ norm으로 정규화

---

#### (4) Similarity Loss (두 가지 옵션)

**옵션 A: DANN 기반 (Gradient Reversal Layer)**

```math
\mathcal{L}_{\text{similarity}}^{\text{DANN}} = \sum_{i=0}^{N_s + N_t} \left\{ d_i \log \hat{d}_i + (1-d_i)\log(1-\hat{d}_i) \right\}
```

- $d_i \in \{0,1\}$: 도메인 레이블
- GRL을 통해 도메인 분류기는 최대화, Shared Encoder $\theta_c$는 최소화 (minimax 최적화)

**옵션 B: MMD 기반**

$$\mathcal{L}_{\text{similarity}}^{\text{MMD}} = \frac{1}{(N_s)^2} \sum_{i,j=0}^{N_s} \kappa(\mathbf{h}_{ci}^s, \mathbf{h}_{cj}^s) - \frac{2}{N_s N_t} \sum_{i,j=0}^{N_s, N_t} \kappa(\mathbf{h}_{ci}^s, \mathbf{h}_{cj}^t) + \frac{1}{(N_t)^2} \sum_{i,j=0}^{N_t} \kappa(\mathbf{h}_{ci}^t, \mathbf{h}_{cj}^t) $$

- $\kappa(\cdot, \cdot)$: PSD 커널 함수 (다중 RBF 커널의 선형 결합 사용)

```math
\kappa(\mathbf{x}_i, \mathbf{x}_j) = \sum_n \eta_n \exp\left\{-\frac{1}{2\sigma_n}\|\mathbf{x}_i - \mathbf{x}_j\|^2\right\}
```

---

#### 추론(Inference)

$$\hat{\mathbf{x}} = D(E_c(\mathbf{x}) + E_p(\mathbf{x})), \quad \hat{\mathbf{y}} = G(E_c(\mathbf{x}))$$

- 재구성: shared + private 표현의 합으로 복원
- 분류: **오직 shared 표현만** 사용

---

### 2.3 모델 구조

```
[Source Image x^s] ──► [Private Source Encoder E_p^s] ──► h_p^s ──┐
[Source Image x^s] ──► [Shared Encoder E_c] ──────────► h_c^s ──┼──► [Shared Decoder D] ──► x̂^s (L_recon)
                                                                   │
[Target Image x^t] ──► [Private Target Encoder E_p^t] ──► h_p^t ──┐
[Target Image x^t] ──► [Shared Encoder E_c] ──────────► h_c^t ──┼──► [Shared Decoder D] ──► x̂^t (L_recon)

                         h_c^s ─────────────────────────────────────────► [Classifier G] ──► ŷ^s (L_task)

                         h_c^s ◄──────── L_similarity ────────────► h_c^t
                         h_c^s & h_p^s ◄── L_difference (직교성 제약)
                         h_c^t & h_p^t ◄── L_difference (직교성 제약)
```

**구성 요소:**

| 구성 요소 | 역할 | 파라미터 |
|-----------|------|----------|
| $E_c(\mathbf{x}; \theta_c)$ | 공유 인코더 (shared encoder) | $\theta_c$ |
| $E_p^s(\mathbf{x}; \theta_p^s)$, $E_p^t(\mathbf{x}; \theta_p^t)$ | 도메인별 private 인코더 | $\theta_p$ |
| $D(\mathbf{h}; \theta_d)$ | 공유 디코더 | $\theta_d$ |
| $G(\mathbf{h}; \theta_g)$ | 분류기 (task-specific) | $\theta_g$ |

---

### 2.4 성능 향상

**Table 1: 분류 정확도 (%) 비교**

| 모델 | MNIST→MNIST-M | Synth→SVHN | SVHN→MNIST | Synth Signs→GTSRB |
|------|--------------|------------|------------|-------------------|
| Source-only | 56.6 | 86.7 | 59.2 | 85.1 |
| CORAL | 57.7 | 85.2 | 63.1 | 86.9 |
| MMD | 76.9 | 88.0 | 71.1 | 91.1 |
| DANN | 77.4 | 90.3 | 70.7 | 92.9 |
| **DSN w/ MMD (ours)** | **80.5** | **88.5** | **72.2** | **92.6** |
| **DSN w/ DANN (ours)** | **83.2** | **91.2** | **82.7** | **93.1** |
| Target-only | 98.7 | 92.4 | 99.5 | 99.8 |

**Table 2: Synth Objects → LINEMOD (3D 포즈 추정)**

| 방법 | 분류 정확도 | 평균 각도 오차 |
|------|------------|--------------|
| Source-only | 47.33% | 89.2° |
| DANN | 99.90% | 56.58° |
| **DSN w/ DANN** | **100.00%** | **53.27°** |

**Ablation Study (Table 3):**
- $\mathcal{L}_{\text{difference}}$ 제거 시 일관된 성능 저하 (예: MNIST→MNIST-M: 83.23 → 80.26)
- Scale-invariant MSE 대신 일반 L2 사용 시 성능 저하 (예: 83.23 → 80.42)

---

### 2.5 한계점

1. **하이퍼파라미터 민감도**: $\alpha, \beta, \gamma$ 조합, MMD 커널 대역폭 $\sigma_n$ 등 튜닝이 어렵고, 비지도 검증 방법론이 미확립
2. **타겟 레이블 일부 사용**: 하이퍼파라미터 검증을 위해 소량의 타겟 레이블을 사용 (완전한 비지도 설정과 괴리)
3. **도메인 쌍 제한**: 두 도메인 간 적응에 초점, 다중 소스/타겟 도메인 확장 미흡
4. **저수준 차이 가정**: 소스-타겟 간 차이가 주로 저수준(noise, illumination, color)에 국한된다고 가정 → 의미론적(semantic) 차이가 큰 경우 적용 제한
5. **평가 데이터셋 한계**: Office, Caltech-256 등의 고수준 변화 데이터셋 제외 (방법론적 한계보다는 데이터셋 적합성 이슈)
6. **GAN 기반 이미지 번역 미활용**: 당시에는 GAN 기반 도메인 변환 방법과의 결합이 시도되지 않음

---

## 3. 모델의 일반화 성능 향상 가능성 (심층 분석)

### 3.1 일반화 향상의 핵심 메커니즘

DSN의 일반화 성능 향상은 다음 세 가지 메커니즘에서 비롯됩니다:

#### (A) Shared Space의 순도(Purity) 보장

$$\mathcal{L}_{\text{difference}} = \left\|\mathbf{H}_c^{s\top} \mathbf{H}_p^s\right\|_F^2 + \left\|\mathbf{H}_c^{t\top} \mathbf{H}_p^t\right\|_F^2$$

기존 방법은 shared space를 도메인 불변으로 만들려 하지만, 도메인 고유 노이즈가 공유 표현에 섞여들 수 있습니다. Difference Loss는 **shared와 private 서브스페이스를 직교화**함으로써 분류기가 순수하게 도메인 불변 정보만 입력받도록 보장합니다. 이는 Salzmann et al.(2010)의 Factorized Orthogonal Latent Spaces 이론에 근거합니다.

#### (B) 재구성을 통한 표현의 풍부성 유지

Private encoder가 재구성 손실 없이 존재하면 trivial solution(항등 함수, 또는 무의미한 표현)에 수렴할 수 있습니다. $\mathcal{L}_{\text{recon}}$은:
- Private encoder가 도메인 고유의 의미 있는 정보를 실제로 포착하도록 강제
- Shared encoder 역시 재구성에 기여하므로 충분한 semantic 정보를 유지

#### (C) 분류기의 입력 정화

$$\hat{\mathbf{y}} = G(E_c(\mathbf{x}))$$

분류기는 오직 $\mathbf{h}\_c$만 입력받으므로, $\mathcal{L}_{\text{difference}}$에 의해 정화된 shared 표현만 사용. 이론적으로 Ben-David et al.(2010)의 타겟 도메인 오차 상한:

$$\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2} d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda$$

에서 도메인 불일치 항 $d_{\mathcal{H}\Delta\mathcal{H}}$를 줄이는 것이 핵심인데, DSN은 shared space를 더 순수하게 유지함으로써 이 항을 효과적으로 감소시킵니다.

### 3.2 일반화 성능 향상의 실증적 증거

- **SVHN→MNIST**: 기존 최고 DANN(70.7%) 대비 DSN w/ DANN이 **82.7%**로 12%p 향상 → 도메인 차이가 클수록 Private-Shared 분리의 효과가 두드러짐
- **Ablation**: $\mathcal{L}_{\text{difference}}$ 제거 시 모든 시나리오에서 일관된 성능 저하 → 직교성 제약이 일반화의 핵심 요소임을 실증

### 3.3 일반화 한계와 확장 가능성

- **현재 한계**: 저수준 도메인 차이(noise, illumination)에 주로 효과적. 의미론적으로 큰 차이(예: 객체 자세 분포가 매우 다른 경우)에서는 제한적
- **확장 가능성**: 
  - NLP, 오디오 등 이미지 외 도메인에도 Private-Shared 분리 원리 적용 가능
  - 다중 도메인으로 확장 시 각 도메인별 private encoder + 하나의 shared encoder 구조로 자연스럽게 확장 가능

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 후속 연구에 미친 영향

#### (A) 표현 분리(Disentanglement) 패러다임 정착

DSN은 **도메인 적응에서 표현을 명시적으로 분리**하는 패러다임을 확립했습니다. 이는 이후:
- **CDANN (Conditional Domain Adversarial Networks)**: 클래스 조건부 도메인 정렬로 확장
- **Domain Randomization + Adaptation**: Private space 개념이 스타일 변환 연구로 이어짐
- **Disentangled VAE 기반 도메인 적응**: Beta-VAE, Factor-VAE 등과 결합

#### (B) GAN 기반 도메인 적응으로의 연결

DSN의 재구성 손실 아이디어는 CycleGAN(Zhu et al., 2017), UNIT(Liu et al., 2017) 등의 이미지-이미지 변환 연구와 자연스럽게 결합되었습니다.

#### (C) 다중 도메인 일반화(Domain Generalization)로의 확장

Private-Shared 분리는 이후 **여러 소스 도메인에서 공통 표현을 추출**하는 Domain Generalization 연구에 직접적인 영향을 줌.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> **주의:** 아래 언급되는 2020년 이후 연구들은 제가 훈련 데이터를 기반으로 기술한 것으로, 논문 원문을 직접 확인하지 않은 부분이 있습니다. 핵심 개념과 방향성은 정확하나, 세부 수치나 세부 방법론은 원문을 확인하시기 바랍니다.

#### (1) SHOT (Liang et al., ICML 2020)
- **"Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation"**
- DSN과의 차이: 소스 데이터에 접근하지 않고 소스 모델의 가중치만으로 도메인 적응
- DSN은 소스 데이터가 필요한 반면, SHOT은 **데이터 프라이버시** 문제를 해결
- DSN의 shared encoder 개념을 발전시켜 소스 모델의 feature extractor를 가설(hypothesis)로 활용

#### (2) MDD (Zhang et al., ICML 2019) / MCC (Jin et al., ECCV 2020)
- Margin Disparity Discrepancy / Minimum Class Confusion
- DSN이 표현 수준에서 도메인 정렬을 수행한 것과 달리, **예측 수준**에서 클래스 조건부 정렬을 강조
- DSN의 한계였던 "클래스 조건 무시 문제"를 보완

#### (3) DANN → CDAN (Long et al., NeurIPS 2018)
- **"Conditional Adversarial Domain Adaptation"**
- DSN이 shared encoder에 비조건부 DANN을 적용한 것을 발전시켜, **클래스 예측과 특징을 결합한 조건부 적대적 훈련** 도입
- multilinear conditioning: $\mathbf{h} \otimes \hat{\mathbf{y}}$를 도메인 분류기 입력으로 사용

#### (4) Domain Generalization 연구들 (2020년 이후)
- **SWAD (Cha et al., NeurIPS 2021)**: Sharpness-Aware Minimization을 통한 flat minima 탐색으로 도메인 일반화
- **DomainBed Benchmark (Gulrajani & Lopez-Paz, ICLR 2021)**: 도메인 일반화 벤치마크 통합 - DSN류 방법들의 실제 성능이 단순 ERM 대비 크게 우월하지 않다는 도전적 결과 제시
  - 이는 DSN이 타겟 레이블 일부를 하이퍼파라미터 튜닝에 사용했다는 점과 관련 있음

#### (5) Transformer 기반 도메인 적응 (2021~)
- **CDTrans (Xu et al., 2021)**: Cross-Domain Transformer
- **TVT (Yang et al., 2023)**: Transformers의 self-attention이 자연스럽게 도메인 불변 특징을 포착
- DSN의 CNN 기반 구조의 한계를 Transformer 아키텍처로 극복 시도

#### (6) Source-Free & Privacy-Preserving Domain Adaptation
- DSN이 소스 데이터를 항상 필요로 하는 한계를 극복하는 방향으로 발전
- **LAME (Boudiaf et al., NeurIPS 2022)**: 타겟 도메인에서의 Laplacian Adjustment만으로 적응

**비교 요약표:**

| 연구 | 주요 아이디어 | DSN 대비 발전점 |
|------|-------------|----------------|
| CDAN (2018) | 조건부 적대 훈련 | 클래스 조건 무시 문제 해결 |
| SHOT (2020) | 소스 가설 전이 | 소스 데이터 불필요 |
| DomainBed (2021) | 공정한 벤치마크 | 방법론의 공정 비교 틀 제공 |
| SWAD (2021) | Flat minima 탐색 | 일반화 이론적 근거 강화 |
| CDTrans (2021) | Transformer 기반 | 더 강력한 표현 학습 |

---

### 4.3 앞으로 연구 시 고려할 점

#### (A) 방법론적 고려사항

1. **클래스 조건부 정렬**: DSN의 비조건부(class-agnostic) 도메인 정렬은 클래스 간 표현 혼동을 유발할 수 있음. 클래스 레이블 정보를 활용한 조건부 정렬 필요
   
2. **하이퍼파라미터의 완전 비지도 최적화**: 현재 소량 타겟 레이블을 사용하는 한계 → 비지도 검증 지표(clustering quality, entropy minimization 등) 개발 필요

3. **확장성**: Private encoder가 각 도메인마다 별도로 존재하므로, 도메인 수 증가 시 파라미터 수 선형 증가. 공유 가능한 private encoder 구조 탐색 필요

4. **고수준 도메인 차이 처리**: 현재 모델은 저수준 통계 차이 가정. 객체 유형, 레이아웃 등의 semantic shift에 대한 robustness 강화 필요

#### (B) 이론적 고려사항

5. **직교성 제약의 이론적 보장**: Soft orthogonality constraint가 실제로 정보 분리를 보장하는지에 대한 이론적 분석 부족. Information-theoretic 프레임워크(Mutual Information 최소화 등)로 보완 가능

6. **일반화 오차 상한의 tight한 분석**: Ben-David et al.의 이론을 DSN의 Private-Shared 구조에 맞게 정제된 오차 상한 도출 필요

#### (C) 실용적 고려사항

7. **프라이버시 보존**: 소스 데이터 전체에 접근이 필요한 DSN은 의료, 금융 등의 데이터 프라이버시 규제 환경에서 사용 제한. Source-free 버전으로의 발전 필요

8. **Foundation Model과의 결합**: CLIP, ViT 등의 대규모 사전학습 모델을 Shared Encoder의 초기화로 활용하면 일반화 성능 대폭 향상 가능

9. **다중 모달리티 확장**: 이미지-텍스트, 이미지-LiDAR 등 멀티모달 도메인 적응에서 Private-Shared 분리 원리의 적용 가능성 탐색

---

## 참고 자료

**논문 원문:**
- Bousmalis, K., Trigeorgis, G., Silberman, N., Krishnan, D., & Erhan, D. (2016). **"Domain Separation Networks."** *Advances in Neural Information Processing Systems (NIPS 2016)*, Barcelona, Spain.

**논문 내 인용 문헌 (주요):**
- Ganin, Y. et al. (2016). "Domain-Adversarial Training of Neural Networks." *JMLR*, 17(59):1–35. [8]
- Salzmann, M. et al. (2010). "Factorized orthogonal latent spaces." *AISTATS*, pp. 701–708. [24]
- Ben-David, S. et al. (2010). "A theory of learning from different domains." *Machine Learning*, 79(1-2):151–175. [4]
- Long, M. & Wang, J. (2015). "Learning transferable features with deep adaptation networks." *ICML*. [17]
- Gretton, A. et al. (2012). "A Kernel Two-Sample Test." *JMLR*, pp. 723–773. [11]
- Sun, B., Feng, J., & Saenko, K. (2016). "Return of frustratingly easy domain adaptation." *AAAI*. [26]

**2020년 이후 관련 연구 (개념 참조):**
- Liang, J. et al. (2020). "Do We Really Need to Access the Source Data?" *ICML 2020*.
- Gulrajani, I. & Lopez-Paz, D. (2021). "In Search of Lost Domain Generalization." *ICLR 2021*.
- Cha, J. et al. (2021). "SWAD: Domain Generalization by Seeking Flat Minima." *NeurIPS 2021*.
- Long, M. et al. (2018). "Conditional Adversarial Domain Adaptation." *NeurIPS 2018*.
