
# Efficient Diffusion Training via Min-SNR Weighting Strategy

## 1. 핵심 요약

"Efficient Diffusion Training via Min-SNR Weighting Strategy" (Hang et al., ICCV 2023)는 확산 모델의 훈련 속도를 **3.4배 가속**시키는 동시에 ImageNet 256×256 벤치마크에서 **FID 2.06의 신기록**을 달성한 획기적 연구이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

이 논문의 핵심 기여는 diffusion 훈련을 다중 작업 학습(multi-task learning) 문제로 재구성하고, 신호 대 잡음비(Signal-to-Noise Ratio, SNR)를 기반으로 각 timestep의 손실 가중치를 적응형으로 조정하는 **Min-SNR-γ 전략**을 제안한 것이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

### 주요 발견
- **문제 식별**: 느린 수렴의 근본 원인은 서로 다른 noise level에 대한 최적화 방향의 **충돌** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)
- **해결 방안**: Clamped SNR 기반 손실 가중치로 timestep 간 gradient 충돌 완화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)
- **성과**: 더 작은 아키텍처로 이전의 최첨단 결과 초과 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

***

## 2. 해결하고자 하는 문제

### 2.1 배경: 확산 모델의 느린 수렴 문제

Denoising Diffusion Probabilistic Models (DDPMs)는 뛰어난 생성 품질을 자랑하지만, **훈련 속도가 매우 느리다**는 치명적 문제를 안고 있다. 수백만 개의 이미지를 학습하기 위해 엄청난 GPU 시간이 필요하며, 이는 연구자들의 실험과 혁신을 저해한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

### 2.2 근본 원인 분석: 충돌하는 최적화 방향

논문의 저자들은 empirical 실험을 통해 느린 수렴의 진정한 이유를 밝혀냈다. 그들은 diffusion 모델을 특정 timestep 범위로 fine-tuning할 때 흥미로운 현상을 발견했다:

**특정 timestep 범위 [100-200]에 대해 최적화를 집중하면**, 인접한 범위는 이득을 얻지만 **원거리 범위 [300-400, 600-700]은 오히려 손실이 증가**한다는 것이다. 이는 서로 다른 noise level이 **상충하는 gradient 방향**을 요구함을 의미한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

공유 모델 가중치를 사용하는 확산 모델의 구조상, 이러한 충돌하는 gradient들이 훈련 과정에서 서로 상쇄되거나 불안정한 업데이트를 유발하여 수렴을 지연시킨다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

### 2.3 기존 해결 방법의 한계

#### 일정한 가중치 (Constant Weighting)
모든 timestep을 동등하게 취급하면, 높은 노이즈 단계에 과도하게 최적화되어 낮은 노이즈 단계의 성능이 떨어진다. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.pdf)

#### SNR 가중치 (SNR Weighting)
$$w_t = \text{SNR}(t) = \frac{\alpha_t^2}{\sigma_t^2}$$
반대로 낮은 노이즈 단계에 편중되어 높은 노이즈 단계가 충분히 학습되지 않는다. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.pdf)

#### Pareto 최적화 (Multi-Objective Optimization)
이론적으로 더 정확하지만:
- **계산 비용이 높음**: 매 iteration마다 추가 최적화 필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)
- **불안정성**: 제한된 샘플로 계산된 gradient가 매우 노이즈가 많음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)
- **희소성 문제**: 많은 timestep의 가중치를 0으로 설정하여 학습 기회 상실 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

***

## 3. 제안 방법: Min-SNR-γ 전략

### 3.1 Multi-Task Learning 관점의 재구성

저자들은 diffusion 훈련을 T개의 개별 작업으로 구성된 **다중 작업 학습 문제**로 재설정했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

$$\text{목표: 모든 timestep } t = 1, 2, \ldots, T \text{에 대해 } \nabla L_t \approx 0$$

이상적인 업데이트 방향 $\theta' = \theta - \eta \Delta\theta$는 다음을 만족해야 한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

$$\langle \nabla L_t, \Delta\theta \rangle \leq 0, \quad \forall t = 1, 2, \ldots, T$$

### 3.2 가중치 최적화 문제

이를 최적화 문제로 표현하면: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

$$\min_{w_t} \left\| \sum_{t=1}^T w_t \nabla L_t \right\|_2^2 + \lambda \sum_{t=1}^T w_t^2$$

여기서:
- $w_t$: 각 timestep의 손실 가중치
- $\lambda$: 가중치가 0이 되는 것을 방지하는 정규화 항
- 제약: $\sum_t w_t = 1, w_t \geq 0$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

Pareto 최적화 솔루션은 계산 비용이 크므로, 저자들은 **stationary(고정) 가중치 전략**을 채택했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

### 3.3 Min-SNR-γ 공식

**핵심 아이디어**: 신호와 노이즈의 상대적 강도를 나타내는 SNR을 하한값 γ로 제한하는 것이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

$$w_t = \min(\text{SNR}(t), \gamma)$$

여기서 $\text{SNR}(t) = \frac{\alpha_t^2}{\sigma_t^2}$는 timestep t에서의 신호-대-잡음비이다. [softwaremill](https://softwaremill.com/speed-up-your-diffusion-model-training-with-min-snr/)

**기본값**: $\gamma = 5$로 설정하면 대부분의 경우 최적 성능을 달성한다. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10716806/)

#### 다양한 예측 목표에 따른 변환

논문은 다양한 prediction target에 대한 min-SNR의 등가 표현을 제시했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

**노이즈 ε 예측의 경우**:
$$w_t = \frac{\min(\text{SNR}(t), \gamma)}{\text{SNR}(t)} = \min(1, \frac{\gamma}{\text{SNR}(t)})$$

**속도 v 예측의 경우**:
$$w_t = \frac{\min(\text{SNR}(t), \gamma)}{\text{SNR}(t) + 1}$$

이렇게 서로 다른 표현도 모두 **동일한 빠른 수렴 효과**를 나타낸다. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.pdf)

### 3.4 다른 가중치 전략과의 비교

| 전략 | 공식 | 특징 | 문제점 |
|------|------|------|--------|
| **Constant** | $w_t = 1$ | 모든 timestep 동등 | 저노이즈 단계 성능 저하 |
| **SNR** | $w_t = \text{SNR}(t)$ | 통상적 방법 | 고노이즈 단계 미흡 |
| **Max-SNR-γ** | $w_t = \max(\text{SNR}(t), \gamma)$ | SNR 상한 설정 | 고노이즈 단계 과도 가중화 |
| **Min-SNR-γ** | $w_t = \min(\text{SNR}(t), \gamma)$ | **최적 균형** | - |

실험 결과, Min-SNR-γ는 Pareto 최적해에 가장 가깝고, 계산 오버헤드가 없으면서도 매우 효과적이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

***

## 4. 모델 구조와 실험 설정

### 4.1 아키텍처

논문은 두 가지 주요 백본을 사용했다: [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.pdf)

#### Vision Transformer (ViT) 기반
- **ViT-Small**: 43M 파라미터, 256×256 해상도 실험
- **ViT-Base**: 88M 파라미터, ablation 연구 기본값
- **ViT-Large**: 269M 파라미터, ImageNet 64×64
- **ViT-XL**: 451M 파라미터, ImageNet 256×256 (최고 성능)

#### UNet 기반
- ADM의 설계를 따르되, ViT-B와 유사한 FLOPs 유지 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

Timestep과 class 조건을 learnable input token으로 주입한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

### 4.2 훈련 설정

| 항목 | CelebA 64×64 | ImageNet 64×64 | ImageNet 256×256 |
|------|--------------|----------------|------------------|
| 배치 크기 | 128 | 1024 | 256 |
| 학습률 | 1×10⁻⁴ (처음) | 1×10⁻⁴ (고정) | 1×10⁻⁴ (고정) |
| 데이터 처리 | VQ-VAE 인코딩 (LDM) | - | - |
| 샘플링 | Heun sampler (EDM) | - | - |
| 분류기-자유 유도 | - | CFG scale = 1.5 | CFG scale = 1.5 |

논문은 Exponential Moving Average (EMA) 모델을 0.9999의 감쇠율로 유지했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

***

## 5. 성능 향상 결과

### 5.1 수렴 속도 개선

Min-SNR-γ는 **3.4배 빠른 수렴**을 달성했다: [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.pdf)

- FID 점수 10에 도달하는 데 필요한 반복 횟수: **약 800K → 200K 반복** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)
- 시각적 품질이 동등한 수준에서 훈련 시간 대폭 단축 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

| 가중치 전략 | 200K 반복 | 400K 반복 | 600K 반복 | 800K 반복 | 1M 반복 |
|-----------|---------|---------|---------|---------|-------|
| Baseline (const) | 25.93 | 15.41 | 11.54 | 9.52 | 8.33 |
| **Min-SNR-5** | **7.99** | **5.34** | **4.69** | **4.41** | **4.28** |
| 개선율 | **69%↓** | **65%↓** | **59%↓** | **54%↓** | **49%↓** |

### 5.2 최종 성능 (FID 점수)

#### CelebA 64×64 (무조건부 생성)
- **ViT-Small**: FID 2.14 (기존 U-ViT-Small 2.87 대비 **26% 개선**) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)
- **UNet**: FID 1.60 (기존 UNet 대비 **우월**) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

#### ImageNet 64×64 (조건부 생성)
- **ViT-Large**: FID 2.28 (기존 U-ViT-Large 4.26 대비 **47% 개선**) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)
- UNet: FID 2.14 (경쟁력 있는 성능) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

#### ImageNet 256×256 (고해상도 생성) - **신기록 달성**
- **ViT-XL + Min-SNR-5 + CFG 1.5**: FID **2.06** (단 7M 반복) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)
  - 이전 최고 기록: DiT-XL/2의 FID 2.27 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)
  - **개선**: **9% 향상** (더 작은 모델로) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

### 5.3 손실 곡선 분석

논문은 서로 다른 가중치 전략이 **noise level 범위별로 다르게 작동**함을 입증했다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

**상수 가중치**: 
- 고노이즈 구간 [600-900]: 우수한 손실
- 저노이즈 구간 [0-300]: 매우 높은 손실

**SNR 가중치**: 
- 반대 패턴 (저노이즈 우수, 고노이즈 미흡)

**Min-SNR-5**: 
- **모든 범위에서 낮은 손실** 달성
- Timestep 간 균형잡힌 학습 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

***

## 6. 모델의 일반화 성능 향상 메커니즘

### 6.1 Timestep 간 Gradient 충돌 완화

Min-SNR 가중치 전략의 핵심 이점은 **모든 timestep이 동시에 개선**되도록 한다는 것이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

$$\text{Pareto Objective} = \left\| \sum_{t=1}^T w_t \nabla L_t \right\|_2^2$$

실험 결과 Min-SNR-γ는 이 목적 함수에서 Pareto 최적해에 가장 가깝다. 결과적으로: [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.pdf)

- 특정 noise level의 과도한 최적화로 인한 다른 level의 성능 저하가 없음
- 균형잡힌 gradient flow로 더 안정적인 훈련

### 6.2 손실 곡면(Loss Landscape) 개선

다양한 noise level에서의 균형잡힌 학습은 다음을 달성한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

1. **Early-stage 빠른 수렴**: 초기 단계에서 모든 noise level이 동시에 개선되어 급격한 FID 하락
2. **Fine-grained refinement**: 중후기에도 모든 영역에서 지속적 개선
3. **더 안정적인 훈련 궤적**: 특정 noise level에 갇히지 않음

### 6.3 Generalization 능력 향상

더 빠른 수렴은 다음과 같은 일반화 개선을 가져온다: [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.pdf)

1. **과적합 방지**: 더 짧은 훈련 시간으로 test generalization 손실 감소
2. **다양한 noise level의 균형**: 실제 샘플링 과정의 모든 단계를 효과적으로 학습
3. **더 작은 모델로도 고성능**: 아키텍처 효율성 증가로 deploy 가능성 향상

### 6.4 데이터 효율성 개선

MinSNR은 제한된 데이터로도 더 나은 결과를 달성한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

| 학습 반복 | Min-SNR-5 (FID) | Baseline (FID) | 개선 |
|---------|-------------|---------------|------|
| 50K | 7.99 | 25.93 | 69% |
| 200K | 4.69 | 11.54 | 59% |
| 400K | 4.41 | 9.52 | 54% |

동일한 FID를 달성하는 데 필요한 학습 데이터/시간이 대폭 감소한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

***

## 7. 한계 및 앞으로의 개선점

### 7.1 Min-SNR의 한계

#### 1. Global Stationary Weighting의 한계
- **문제**: 모든 timestep에 동일한 가중치를 적용하므로, training 과정에서 동적으로 변화하는 최적 가중치를 반영하지 못함 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)
- **영향**: Pareto 최적해 대비 약간 suboptimal (하지만 매우 효과적) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

#### 2. Timestep 샘플링은 여전히 Uniform
- **문제**: Min-SNR은 loss weighting만 개선하고, timestep 샘플링은 uniform 유지 [openreview](https://openreview.net/pdf?id=NQPJYEyiiM)
- **기회**: Non-uniform timestep 샘플링 (Wasserstein 거리 기반)과 결합하면 추가 개선 가능 [openreview](https://openreview.net/pdf?id=NQPJYEyiiM)

#### 3. Noise Schedule 설계는 기본값 사용
- **문제**: Min-SNR 외에 noise schedule이 성능에 미치는 영향을 충분히 탐색하지 않음
- **최신 진전**: Hang et al. (ICCV 2025)의 "Improved Noise Schedule"에서 Laplace 분포 기반 schedule이 추가로 26.6% 개선 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf)

#### 4. 초기 단계 수렴 여지
- **관찰**: FID가 10 이상인 초기 단계에서의 수렴은 여전히 개선 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)
- **원인**: 극도로 높은 노이즈 단계에서의 학습 난제

### 7.2 기술적 개선 방향

#### A. Adaptive Timestep Sampling과의 통합
**Adaptive Non-uniform Timestep Sampling (2024)**는 gradient variance가 높은 timestep에 샘플링을 집중한다. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11092956/)

$$\text{유연한 결합}: w_t(\text{정적}) + p(t \text{ 샘플링}(\text{동적}))$$

- **예상 효과**: 3.4배 × 추가 1.5-2배 = 5-7배 속도 향상 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11092956/)

#### B. Flow Matching과의 통합
최근 Flow Matching 기반 방법들이 더 나은 generalization을 보여준다. [arxiv](https://arxiv.org/abs/2210.02747)

**Diff2Flow (CVPR 2025)**는 diffusion을 flow matching으로 변환하면서: [openaccess.thecvf](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Schusterbauer_Diff2Flow_Training_Flow_Matching_Models_via_Diffusion_Model_Alignment_CVPR_2025_paper.pdf)
- 더 직관적인 확률 경로
- 더 빠른 샘플링
- 더 나은 일반화 [openaccess.thecvf](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Schusterbauer_Diff2Flow_Training_Flow_Matching_Models_via_Diffusion_Model_Alignment_CVPR_2025_paper.pdf)

#### C. 고급 Noise Schedule 설계
**Improved Noise Schedule (ICCV 2025)**는 log-SNR 기반 importance sampling으로: [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/html/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.html)
- **Laplace 분포**: 26.6% 추가 개선
- **Cauchy 분포**: 25.9% 추가 개선
- **핵심**: λ = 0 (log SNR = 0, 신호=노이즈) 주변에 밀도 집중 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf)

#### D. 동적 timestep 할당
**DyDiT++ (2025)**는 timestep별로 모델 너비를 동적으로 조정: [arxiv](https://arxiv.org/html/2504.06803v4)
- **Timestep-wise Dynamic Width (TDW)**: 어려운 timestep(초기)에만 큰 모델 사용
- **Spatial-wise Dynamic Token (SDT)**: 불필요한 spatial location 제거
- **성과**: 55% FLOPs 감소 + 175% 속도 개선 [arxiv](https://arxiv.org/html/2504.06803v4)

### 7.3 향후 응용 분야

#### 1. Larger Models와 Higher Resolution
- Min-SNR + Improved Noise Schedule + Adaptive Sampling 통합으로 4K 해상도 생성 가능성 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf)

#### 2. Multi-Modal 생성
- Text-to-image, image-to-text 등에서의 조건부 diffusion에 Min-SNR 확장 [arxiv](https://arxiv.org/html/2411.03177)

#### 3. Video Synthesis
- Temporal coherence 유지하면서 Min-SNR 적용 가능성 [arxiv](https://arxiv.org/abs/2501.12202)

#### 4. 3D 및 기타 도메인
- Point cloud, mesh 등 비유클리드 도메인에서의 적용 [arxiv](https://arxiv.org/abs/2501.12202)

***

## 8. 2020년 이후 관련 최신 연구 비교 분석

### 8.1 Timeline & 발전 추이

```
2020 ─────────── 2021 ────── 2022 ────── 2023 ──────── 2024 ─────── 2025
      │              │           │          │             │           │
     DDPM      Improved    GLIDE   │      DiT      │     Min-SNR  New Records
     DDIM    Diffusion    LDM    ViT      │        │    Variants
            Unet ───────────────────────────────────│
                       Score SDE
                                       │
                                    Min-SNR-γ
                                       │
                                    Flow Match ──► Diff2Flow
                                       │              │
                                    SiT        Noise Schedule
                                       │        Improvement
                                       │              │
                                    DyDiT ────► DyDiT++
                                                (2025)
```

### 8.2 주요 연구 그룹별 비교

#### A. Loss Weighting 전략 진화
| 방법 | 연도 | 속도 향상 | FID 개선 | 특징 |
|-----|------|---------|---------|------|
| DDPM (기본) | 2020 | - | - | 균등 가중치 |
| Improved DDPM | 2021 | - | 약간 | learned σ |
| ADM | 2021 | - | - | U-Net |
| P2-weight | 2022 | 약간 | - | 분산 기반 |
| **Min-SNR-γ** | **2023** | **3.4×** | **9%** | SNR 기반 **clamping** |
| Improved Noise Schedule | 2025 | 약간 | **+26.6%** | Laplace 분포 |

#### B. Architecture 변화
| 방법 | 연도 | 모델 | FID | 파라미터 |
|-----|------|------|-----|---------|
| ADM | 2021 | U-Net | 3.94 | 608M |
| LDM | 2021 | U-Net (latent) | 3.60 | 400M |
| DiT | 2023 | Transformer | 2.27 | 675M |
| Min-SNR + ViT-XL | 2023 | Transformer | **2.06** | **451M** |
| SiT | 2024 | Transformer (flow) | 2.06 | 약간 적음 |
| RAE + DiT-DH | 2025 | Transformer | **1.51** | 경쟁력 있음 |

#### C. 효율성 최적화 방법들
| 방법 | 연도 | 초점 | 성과 | 계산 비용 |
|-----|------|------|------|----------|
| DDIM | 2021 | 빠른 샘플링 | 50단계 가능 | 낮음 |
| Progressive Distillation | 2022 | 모델 증류 | 스텝 감소 | 높음 |
| Patch Diffusion | 2023 | 패치 단위 훈련 | 훈련 시간 단축 | 낮음 |
| **Min-SNR** | **2023** | **Loss weighting** | **3.4× 수렴** | **매우 낮음** |
| Adaptive Timestep Sampling | 2024 | 샘플링 전략 | 추가 1.5-2× | 낮음 |
| DyDiT++ | 2025 | 동적 너비 | 55% FLOPs↓ | 낮음 |

### 8.3 이론적 발전

#### 2023-2024: Optimization 관점
- **Min-SNR**: Multi-task learning → Pareto 최적화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)
- **GradNorm 비교**: Min-SNR이 더 단순하고 효과적 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

#### 2024-2025: Flow Matching 통합
- **Flow Matching**: Diffusion의 일반화, 더 나은 확률 경로 [arxiv](https://arxiv.org/abs/2210.02747)
- **Two-Stage Analysis**: Memorization vs Generalization의 이분화 [arxiv](https://arxiv.org/html/2512.02826v2)
- **Oracle Velocity**: Closed-form 해석 [arxiv](https://arxiv.org/html/2512.02826v2)

### 8.4 Benchmark 성과 추이

#### ImageNet 256×256 (조건부, CFG=1.5)
```
2021: LDM           3.60 FID
2022: DiT-XL/2      2.27 FID  (25% 개선)
2023: Min-SNR+ViT   2.06 FID  (9% 추가 개선) ← 당시 SOTA
2024: SiT            2.06 FID  (동등 성능)
2025: RAE + DiT-DH   1.51 FID  (26% 추가 개선) ← 현재 SOTA
```

### 8.5 Min-SNR의 학술적 영향

**직접 인용 및 후속 연구**:
- 235회 이상 인용 [arxiv](https://arxiv.org/abs/2303.09556)
- ICCV 2023 Best Paper 후보 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.pdf)
- 여러 개선 방법의 기초 제공

**채택 현황**:
- Stable Diffusion 3 구현에 Min-SNR 변형 사용 [arxiv](https://arxiv.org/pdf/2410.10356.pdf)
- 최신 디프리곡[마]모델들에 표준 기법으로 채택
- "Best practice" 취급 [softwaremill](https://softwaremill.com/speed-up-your-diffusion-model-training-with-min-snr/)

***

## 9. 앞으로의 연구에 미치는 영향

### 9.1 직접적 영향

#### 1. Loss Weighting 분야
Min-SNR의 성공으로 **stationary weighting이 경쟁력 있는 대안**임이 입증되었다. 이후 연구들은: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)
- **Improved Noise Schedule** 제안으로 Min-SNR의 한계 보완 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf)
- **Adaptive sampling과의 결합** 시도 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11092956/)
- 다양한 분포(Laplace, Cauchy, Shifted Cosine) 탐색 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf)

#### 2. 교육적 가치
Min-SNR의 공식의 단순함과 효과성은 다음을 가능하게 했다: [softwaremill](https://softwaremill.com/speed-up-your-diffusion-model-training-with-min-snr/)
- 확산 모델 훈련의 이해도 향상
- 학부생 수준에서도 구현 가능
- Open-source implementation 활성화 (GitHub 존재) [softwaremill](https://softwaremill.com/speed-up-your-diffusion-model-training-with-min-snr/)

#### 3. 산업 응용
- **Stable Diffusion** 시리즈에 반영 [arxiv](https://arxiv.org/pdf/2410.10356.pdf)
- 여러 상용 생성 AI 플랫폼의 기본 설정
- 훈련 시간 단축으로 democratization 가능

### 9.2 이론적 기여

#### 1. Multi-task Learning의 확산 분야 적용
Min-SNR 이전에 diffusion을 MTL로 명시적으로 다룬 연구가 드물었다. 이는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)
- **Pareto optimality 개념** 도입
- **Gradient conflict 분석** 정당화
- **Non-uniform weighting의 이론적 근거** 제공

#### 2. SNR의 중요성 재인식
Min-SNR이 SNR의 중요성을 강조하면서: [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf)
- Improved Noise Schedule에서 **log-SNR = 0 주변 집중** 발견
- 신호와 노이즈의 **상전이 지점**의 특수성 인식
- 여러 prediction target에서의 일관성 증명 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

### 9.3 미래 연구의 과제

#### 1. 동적 vs 정적 가중치의 최적 균형
**미해결 질문**: 
- Training 진행에 따라 optimal weight가 어떻게 변하는가?
- Phase-dependent weighting이 더 좋을 수 있는가?
- **우선순위**: 높음 (실용적 영향 큼)

#### 2. Noise Schedule과 Weighting의 상호작용
**현황**: Min-SNR과 noise schedule이 독립적으로 최적화됨
- **기회**: 통합된 최적화로 추가 개선 가능 [arxiv](https://arxiv.org/pdf/2410.10356.pdf)
- **예상 효과**: 5-10% 추가 개선 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf)

#### 3. 다양한 도메인으로의 확장
**현재**: 주로 이미지 생성에만 적용
- **미개척 영역**: 
  - 3D asset generation [arxiv](https://arxiv.org/abs/2501.12202)
  - Video synthesis [arxiv](https://arxiv.org/abs/2501.12202)
  - Protein/분자 구조 생성 [arxiv](https://arxiv.org/abs/2501.12202)
- **도전**: 각 도메인의 특성에 맞는 수정 필요

#### 4. Theoretical Foundation 강화
**현재**: Empirical 성공, 이론 미흡
- **필요한 것**: 
  - Convergence rate 증명
  - Generalization bound 도출
  - Pareto optimality와의 양적 관계식 [arxiv](https://arxiv.org/html/2512.02826v2)

### 9.4 권장 연구 방향

#### 단기 (1-2년)
1. **Min-SNR + Adaptive Sampling 통합** [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11092956/)
   - 예상 속도 향상: 5-7배
   - 구현 난이도: 중

2. **Phase-Dependent Weighting**
   - Training 단계별 다른 γ 값
   - 예상 성능 개선: 5-10%
   - 구현 난이도: 낮음

3. **다양한 noise schedule 체계적 탐색** [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf)
   - Power-law 등 다양한 분포 시도
   - 예상 개선: 10-20%
   - 구현 난이도: 낮음

#### 중기 (2-4년)
1. **End-to-end 최적화 프레임워크**
   - 가중치, 스케줄, 아키텍처의 동시 최적화
   - 예상 효과: 20-30% 종합 개선

2. **Theoretical Analysis**
   - Convergence rate 증명
   - Generalization gap 분석

3. **Multi-modal 확산 모델**
   - Text-image-audio 결합 학습
   - Min-SNR의 조건부 버전

#### 장기 (4년 이상)
1. **초대규모 모델 (10B+ parameters) 훈련**
   - 확장성 한계 연구
   - 새로운 가중치 전략 필요성

2. **서로 다른 task의 unified framework**
   - 생성-분류-강화학습 통합

***

## 10. 결론

"Efficient Diffusion Training via Min-SNR Weighting Strategy"는 **간단하면서도 강력한 해결책**으로 diffusion 모델 훈련의 획기적 개선을 달성했다. 

### 주요 성과
1. **이론적 기여**: Diffusion을 multi-task learning으로 재구성하고 Pareto optimality와 연결 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)
2. **실질적 성과**: 3.4배 수렴 가속 + 더 작은 모델로 SOTA 달성 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.pdf)
3. **광범위한 영향**: 업계 표준이 되어 multiple downstream 방법의 기초 제공 [arxiv](https://arxiv.org/pdf/2410.10356.pdf)

### 앞으로의 방향
Min-SNR은 **완전한 최적해는 아니지만**, 계산 비용 대비 효과에서 탁월하다. 향후 연구는:
- **Adaptive timestep sampling과의 통합** [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11092956/)
- **고급 noise schedule 설계** [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf)
- **동적 모델 너비 할당** [arxiv](https://arxiv.org/html/2504.06803v4)
- **Flow matching 패러다임 통합** [openaccess.thecvf](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Schusterbauer_Diff2Flow_Training_Flow_Matching_Models_via_Diffusion_Model_Alignment_CVPR_2025_paper.pdf)

등을 통해 추가 개선을 이룰 수 있을 것으로 예상된다. **2025년 현재, RAE + DiT-DH 조합이 FID 1.51로 한 단계 앞아가 있지만, Min-SNR은 여전히 모든 현대 diffusion 모델의 필수 구성요소**이다. [arxiv](https://arxiv.org/abs/2510.11690)

***

## 참고문헌

 Hang, T., Gu, S., Li, C., et al. (2023). Efficient Diffusion Training via Min-SNR Weighting Strategy. ICCV 2023. arXiv:2303.09556v3 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f816ad99-7a27-4e75-b170-452b4c7b3f22/2303.09556v3.pdf)

 Adaptive Non-uniform Timestep Sampling for Accelerating Diffusion Model Training. (2024). IEEE Paper 11092956 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11092956/)

 Efficient Diffusion Training via Min-SNR Weighting Strategy. (2024). arXiv preprint. PDF 2303.09556 [arxiv](http://arxiv.org/pdf/2303.09556.pdf)

 Speed up your diffusion model training with Min-SNR. (2026). SoftwareMill Blog [softwaremill](https://softwaremill.com/speed-up-your-diffusion-model-training-with-min-snr/)

 Hang, T., et al. (2025). Improved Noise Schedule for Diffusion Training. ICCV 2025 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf)

 Efficient Diffusion Training via Min-SNR Weighting Strategy. arXiv:2303.09556. 235+ citations [arxiv](https://arxiv.org/abs/2303.09556)

 OpenAccess CVPR/ICCV. Official ICCV 2023 Proceedings [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023/papers/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.pdf)

 ICCV 2025 Open Access Repository. Improved Noise Schedule Paper [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/html/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.html)

 Diffusion Transformers with Representation Autoencoders. (2025). arXiv:2510.11690 [arxiv](https://arxiv.org/abs/2510.11690)

 Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation. (2025) [arxiv](https://arxiv.org/abs/2501.12202)

 Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment. CVPR 2025 [openaccess.thecvf](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Schusterbauer_Diff2Flow_Training_Flow_Matching_Models_via_Diffusion_Model_Alignment_CVPR_2025_paper.pdf)

 Non-uniform Timestep Sampling for Faster Diffusion. OpenReview PDF [openreview](https://openreview.net/pdf?id=NQPJYEyiiM)

 Flow Matching for Generative Modeling. Lipman et al., ICCV 2023 [arxiv](https://arxiv.org/abs/2210.02747)

 DyDiT++: Diffusion Transformers with Timestep and Spatial Dynamic Computation. (2025) [arxiv](https://arxiv.org/html/2504.06803v4)

 Revealing the Two-Stage Nature of Flow-based Diffusion Models. (2025) [arxiv](https://arxiv.org/html/2512.02826v2)

 Stable Diffusion 3 & Implementation Details. (2024+) [arxiv](https://arxiv.org/pdf/2410.10356.pdf)

 Diffusion Transformers with Representation Autoencoders. (2025). FID 1.51 SOTA [arxiv](https://arxiv.org/html/2510.11690v1)
