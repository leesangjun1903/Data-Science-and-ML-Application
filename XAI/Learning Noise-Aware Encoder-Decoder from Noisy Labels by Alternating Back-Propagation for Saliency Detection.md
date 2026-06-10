# Learning Noise-Aware Encoder-Decoder from Noisy Labels by Alternating Back-Propagation for Saliency Detection

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문은 **비지도 핸드크래프트 기반 방법으로 생성된 노이즈 레이블(noisy labels)로부터 깨끗한 현저성(saliency) 예측기를 분리(disentangle)하여 학습**할 수 있다는 것을 주장합니다. 핵심 아이디어는 노이즈 레이블 $Y$를 깨끗한 현저성 맵 $S$와 노이즈 맵 $\Delta$의 합으로 분해하는 잠재 변수 모델(latent variable model)을 구성하고, 이를 Alternating Back-Propagation(ABP) 알고리즘으로 학습하는 것입니다.

### 주요 기여

1. **Noise-Aware Encoder-Decoder 프레임워크 제안**: 노이즈 레이블에서 깨끗한 현저성 예측기를 분리하는 새로운 잠재 변수 모델
2. **ABP 알고리즘의 확장 적용**: 추가적인 보조 모델 없이 데이터 가능도(data likelihood)를 직접 최대화
3. **Edge-Aware Smoothness Loss 활용**: 자명해(trivial solution) 수렴 방지를 위한 정규화
4. **비지도 현저성 검출에서 SOTA 달성**: 여러 벤치마크에서 기존 비지도/약지도 방법 대비 최고 성능, 일부 완전 지도 방법과 비견되는 성능

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

딥러닝 기반 현저성 검출 모델은 대규모 픽셀 수준의 정확한 레이블링을 필요로 하지만, 이는 **비용과 시간이 매우 많이** 소요됩니다. 기존 비지도 방법들은 핸드크래프트 기반 알고리즘(RBD, MR, GS 등)으로 생성된 노이즈 레이블을 그대로 사용하여 딥 모델을 학습하므로, 노이즈에 취약한 문제가 있습니다.

**핵심 문제**: 노이즈 레이블로부터 어떻게 깨끗한 현저성 예측기를 학습할 것인가?

---

### 2.2 제안하는 방법 (수식 포함)

#### 모델 정의

$$S = f_1(X; \theta_1) \tag{1}$$

$$\Delta = f_2(Z; \theta_2), \quad Z \sim \mathcal{N}(0, I_d) \tag{2}$$

$$Y = S + \Delta + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I_D) \tag{3}$$

- $f_1$: VGG16 기반 인코더-디코더 (깨끗한 현저성 맵 $S$ 예측)
- $f_2$: 가우시안 잠재 벡터 $Z$로부터 노이즈 $\Delta$ 생성하는 top-down 생성 네트워크
- $\epsilon$: 관측 잔차 (Gaussian residual)

노이즈 $\Delta$는 $Z$가 가우시안이더라도 $f_2$의 비선형 변환으로 인해 임의의 구조적 분포를 가질 수 있습니다.

#### 최대 가능도 추정 (MLE)

$$\mathcal{L}(\theta) = \sum_{i=1}^{n} \log p_\theta(Y_i | X_i) \tag{목적함수}$$

Log-likelihood의 기울기:

$$\frac{\partial}{\partial \theta} \log p_\theta(Y|X) = \mathbb{E}_{p_\theta(Z|Y,X)} \left[ \frac{\partial}{\partial \theta} \log p_\theta(Y, Z|X) \right] \tag{4}$$

사후 분포 $p_\theta(Z|Y,X)$는 분석적으로 다루기 어렵(intractable)하므로, **Langevin Dynamics**를 이용한 MCMC 샘플링을 사용합니다.

#### Inferential Back-Propagation (Langevin Dynamics)

$$Z_{t+1} = Z_t + \frac{s^2}{2} \left[ \frac{\partial}{\partial Z} \log p_\theta(Y, Z_t | X) \right] + s\mathcal{N}(0, I_d) \tag{5}$$

$$\frac{\partial}{\partial Z} \log p_\theta(Y, Z|X) = \frac{1}{\sigma^2}(Y - f(X, Z; \theta)) \frac{\partial}{\partial Z} f(X, Z) - Z \tag{6}$$

#### Learning Back-Propagation (파라미터 업데이트)

$$\frac{\partial}{\partial \theta} \mathcal{L}(\theta) \approx \sum_{i=1}^{n} \frac{1}{\sigma^2} (Y_i - f(X_i, Z_i; \theta)) \frac{\partial}{\partial \theta} f(X_i, Z_i) \tag{7}$$

#### Edge-Aware Smoothness Loss (정규화)

$$l_s(X, S) = \sum_{u,v} \sum_{d \in x,y} \Psi\left(|\partial_d S_{u,v}| e^{-\alpha|\partial_d X_{u,v}|}\right) \tag{8}$$

- $\Psi(s) = \sqrt{s^2 + 1e^{-6}}$: Charbonnier penalty
- $(u,v)$: 픽셀 좌표
- $d$: $x$, $y$ 방향 편미분
- 실험 설정: $\lambda = 0.7$, $\alpha = 10$

최종 목적함수:

$$\hat{\theta} = \arg\max_\theta \left[ \mathcal{L}(\theta) - \lambda \cdot l_s(X, S; \theta) \right]$$

---

### 2.3 모델 구조

```
[Training Phase]
입력 이미지 X ──► [Encoder-Decoder f₁(θ₁)] ──► 깨끗한 현저성 S
                                                        │
가우시안 벡터 Z ──► [Noise Generator f₂(θ₂)] ──► 노이즈 Δ
                                                        │
                              Y = S + Δ + ε ◄──────────┘
                              (노이즈 레이블과 비교)

[Test Phase]
입력 이미지 X ──► [Encoder-Decoder f₁(θ₁)] ──► 최종 현저성 맵 S
```

#### 인코더-디코더 ($f_1$, Saliency Predictor)
- **백본**: VGG16-Net (ImageNet 사전학습 가중치로 초기화)
- 각 컨볼루션 그룹의 마지막 레이어: $s_1, s_2, ..., s_5$
- $1\times1$ 컨볼루션으로 채널 차원 32로 축소 → $s'_m$
- **Residual Channel Attention (RCA)** 모듈로 고/저수준 특징 융합
  - Squeeze-and-Excitation 연산
  - 2x bilinear interpolation 업샘플링
- $3\times3$ 컨볼루션으로 최종 1채널 현저성 맵 출력

#### 노이즈 생성기 ($f_2$, Noise Generator)
- 4개의 연속된 Deconvolutional layers
- Batch Normalization + ReLU 레이어
- 마지막: **tanh** 활성화 → 노이즈 범위 $[-1, 1]$
- 잠재 변수 차원: $d = 8$

#### 학습 세부사항
- 이미지 크기: $352 \times 352$
- Adam optimizer, 학습률 $\gamma = 0.0001$
- 최대 에폭 $K = 20$
- Langevin steps $l = 6$, step size $s = 0.3$, $\sigma = 0.1$
- 배치 크기: 10

---

### 2.4 성능 향상

#### 비지도/약지도 방법 대비

| 데이터셋 | 방법 | $S_\alpha \uparrow$ | $F_\beta \uparrow$ | $E_\xi \uparrow$ | $\mathcal{M} \downarrow$ |
|---------|------|------|------|------|------|
| DUTS | MNL [53] | .8128 | .7249 | .8525 | .0749 |
| DUTS | **Ours** | **.8276** | **.7467** | **.8592** | **.0601** |
| HKU-IS | MNL [53] | .8602 | .8196 | .8579 | .0650 |
| HKU-IS | **Ours** | **.8901** | **.8782** | **.9191** | **.0428** |

- 비지도/약지도 방법 중 **전 벤치마크에서 최고 성능** 달성
- DUTS 기준: S-measure 약 **2% 향상**, F-measure 약 **4% 향상** (MNL 대비)
- 일부 완전 지도 방법(NLDF, DGRL)과 **비견되는 성능**

#### Ablation Study 결과

| 모델 구성 | $S_\alpha$ (DUTS) | $F_\beta$ (DUTS) |
|----------|----------|----------|
| $f_1$ only | .644 | .453 |
| $f_1$ + $l_s$ | .668 | .519 |
| $f$ + $l_c$ (대체 손실) | .813 | .725 |
| **Full Model** | **.828** | **.747** |

---

### 2.5 한계점

1. **계산 비용**: Langevin Dynamics의 반복적 추론 과정이 추가되어 학습 시간 증가 (8시간/RTX GPU)
2. **노이즈 레이블 품질 의존성**: 핸드크래프트 방법(RBD, MR, GS)의 품질에 어느 정도 의존 (단, 논문에서는 단일 노이즈 레이블에도 견고성 입증)
3. **완전 지도 방법과의 성능 격차**: 최신 완전 지도 방법(SCRN, BASNet 등)과는 여전히 성능 차이 존재
4. **단일 이미지 입력 가정**: 이미지당 단일 노이즈 레이블을 사용하는 경우에도 잘 작동하지만, 여러 노이즈 레이블 활용 가능성이 충분히 탐색되지 않음
5. **백본 제한**: VGG16에 한정되어 있어 최신 강력한 백본(ResNet, Transformer 등) 적용 시 추가 성능 향상 가능성 존재

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 논문에서 확인된 일반화 성능

#### (a) 다양한 노이즈 소스에 대한 견고성

논문은 단일/복수 노이즈 레이블, 다양한 품질의 노이즈 생성기(RBD, MR, GS)에 대한 실험을 수행했습니다:

$$\text{f-RBD}: S_\alpha=.824, \quad \text{f-MR}: S_\alpha=.814, \quad \text{f-GS}: S_\alpha=.787 \quad \text{(DUTS 기준)}$$

이는 원본 노이즈 레이블 대비 각각:
$$\text{RBD}: .644 \rightarrow .824, \quad \text{MR}: .620 \rightarrow .814, \quad \text{GS}: .619 \rightarrow .787$$

노이즈 품질에 무관하게 일관되게 크게 향상되어, **노이즈 소스의 다양성에 강건함**을 보여줍니다.

#### (b) 완전 지도 레이블로의 확장 (Clean Label 적용)

깨끗한 레이블을 사용할 때 ($f^*$ 실험):
- 노이즈 생성기가 zero noise를 출력하도록 학습됨
- $f^* > f_1^*$: 노이즈 처리 전략이 인간 어노테이션의 불완전성에도 효과적임을 시사

```math
f^*: S_\alpha=.861 > f_1^*: S_\alpha=.840 \quad \text{(DUTS)}
```

이는 **완전 지도 학습에서도 일반화 성능 향상**이 가능함을 의미합니다.

#### (c) 완전 지도 모델의 부스팅 전략으로의 활용

BASNet의 출력을 노이즈 레이블로 사용한 f-BAS 실험:
$$\text{f-BAS}: S_\alpha=.870 \geq \text{BASNet}: S_\alpha=.8657 \quad \text{(DUTS)}$$

이는 **기존 완전 지도 모델의 성능을 추가적으로 향상**시킬 수 있음을 보여주어, 방법론의 범용성(generalizability)을 증명합니다.

#### (d) VAE 대비 ABP의 일반화 우위

VAE 기반 추론(cVAE)과 ABP 비교:
$$\text{cVAE}: S_\alpha=.771 < \text{Ours (ABP)}: S_\alpha=.828 \quad \text{(DUTS)}$$

ABP는 근사 추론 모델( $p_\phi$ )에 의존하지 않으므로, **사후 분포 근사 오차( $\text{KL}(p_\phi \| p_\theta)$ )가 없어 더 정확한 기울기 추정**이 가능합니다. 이는 일반화 성능 향상의 이론적 근거입니다.

### 3.2 일반화 성능 향상 잠재력

#### (a) 노이즈 모델의 유연성

$f_2$가 신경망으로 파라미터화되어 있어:

$$\Delta = f_2(Z; \theta_2), \quad Z \sim \mathcal{N}(0, I_d)$$

**임의의 구조적 노이즈 분포를 근사**할 수 있습니다. 이는 가우시안 노이즈 가정에 한정된 기존 방법보다 더 넓은 범위의 노이즈 유형에 적용 가능합니다.

#### (b) Edge-Aware Smoothness Loss의 정규화 효과

$$l_s(X, S) = \sum_{u,v} \sum_{d \in x,y} \Psi\left(|\partial_d S_{u,v}| e^{-\alpha|\partial_d X_{u,v}|}\right)$$

이 손실은 이미지 구조($X$의 엣지)와 예측 현저성 맵($S$)의 구조를 연결하는 **image-guided 정규화**로 작동합니다. 이는 오버피팅 방지와 함께, 훈련 데이터에 없는 새로운 이미지 구조에 대한 일반화를 촉진합니다.

#### (c) 다양한 도메인으로의 확장 가능성

- **의료 이미지**: 정확한 픽셀 레이블이 비싸고, 알고리즘 기반 노이즈 레이블 활용 가능
- **RGB-D 현저성**: 깊이 정보를 추가 입력으로 확장 가능
- **비디오 현저성**: 시간적 노이즈 생성기 설계로 확장 가능

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

#### (a) 노이즈 레이블 학습 패러다임의 전환

기존 방법들이 노이즈 레이블을 **필터링하거나 수정**하는 방향이었다면, 본 논문은 노이즈를 **명시적으로 모델링하고 분리**하는 새로운 패러다임을 제시합니다. 이는 현저성 검출뿐만 아니라 노이즈 레이블이 존재하는 **다양한 밀집 예측(dense prediction) 태스크**에 영향을 미칠 수 있습니다.

#### (b) ABP의 적용 범위 확장

ABP 알고리즘을 현저성 검출이라는 특수한 태스크에 적용하여 그 효용성을 입증함으로써, **다른 컴퓨터 비전 태스크**(깊이 추정, 시맨틱 분할 등)에서의 ABP 적용 연구를 촉진할 수 있습니다.

#### (c) 약지도/비지도 학습의 실용적 가능성 제시

완전 지도 방법과 비견되는 성능을 보여줌으로써, **레이블링 비용 절감과 성능 유지 간의 균형**을 가능성이 있음을 시사합니다.

---

### 4.2 향후 연구 시 고려할 점

#### (a) 더 강력한 백본 아키텍처 탐색

VGG16 대신 ResNet, EfficientNet, Vision Transformer(ViT) 등의 백본을 적용할 때:
- ABP의 Langevin Dynamics와 Transformer 기반 아키텍처의 호환성 검토
- 더 큰 모델에서의 수렴 안정성 확인 필요

#### (b) Langevin Dynamics 효율화

현재 학습 시 각 이미지마다 $l=6$ 스텝의 반복 추론이 필요합니다. 향후 연구에서:
- **Stochastic Gradient Langevin Dynamics(SGLD)** 적용을 통한 학습 속도 향상
- **MCMC mixing time** 단축 방안 연구
- 병렬화 가능한 추론 구조 설계

#### (c) 노이즈 생성기의 조건부 설계

현재 노이즈 생성기 $f_2(Z; \theta_2)$는 이미지 $X$를 조건으로 하지 않습니다. 향후에는:

$$\Delta = f_2(Z, X; \theta_2)$$

와 같이 **이미지 조건부 노이즈 생성기**를 설계함으로써, 이미지 특성에 맞는 맞춤형 노이즈 표현이 가능할 수 있습니다.

#### (d) 다중 모달 노이즈 레이블

RGB-D 데이터, 비디오 시퀀스 등의 멀티모달 정보를 노이즈 레이블로 활용하거나, 여러 알고리즘에서 생성된 복수의 노이즈 레이블을 결합하는 앙상블 전략 연구가 필요합니다.

#### (e) 이론적 수렴 보장 강화

현재 ABP 알고리즘의 수렴성은 경험적으로 입증되었으나, Langevin Dynamics의 **혼합 시간(mixing time)과 수렴 조건**에 대한 이론적 분석이 부족합니다. 특히 non-convex 최적화 환경에서의 이론적 보장이 필요합니다.

#### (f) 최신 노이즈 학습 기법과의 통합

- **Co-training/Co-teaching** 전략과의 결합 (다수의 노이즈 레이블 처리)
- **Meta-learning** 기반 노이즈 적응적 학습률 조정
- **Curriculum Learning** 전략을 ABP 프레임워크에 통합

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 논문들 중 일부는 본 논문의 reference에 포함된 논문(2020년 이전 포함)이거나, 제가 학습 데이터 기반으로 알고 있는 연구들입니다. 2020년 이후 최신 논문에 대한 세부 수치는 제 학습 데이터의 한계로 인해 100% 정확성을 보장하기 어려운 부분이 있으므로, **본 논문의 내용과 직접 비교 가능한 수치만 제시**하고 나머지는 방향성 수준에서 기술합니다.

### 5.1 비지도/약지도 현저성 검출 분야의 연구 흐름

#### Self-supervised 방향
- **DeepUSPS** (Nguyen et al., NeurIPS 2019): 자기 지도 학습으로 노이즈 레이블을 정제 후 pseudo label로 활용 → 본 논문과 달리 노이즈를 명시적으로 모델링하지 않음

#### Transformer 기반 비지도 현저성
2020년 이후 Vision Transformer(ViT)의 등장으로:
- Self-attention 메커니즘의 attention map을 현저성 맵의 초기 noisy label로 활용하는 연구 방향 등장
- 본 논문의 프레임워크에 Transformer 백본을 적용하면 추가적인 성능 향상 가능성 존재

### 5.2 핵심 비교 관점

| 관점 | 본 논문 (Zhang et al., 2020) | 후속 연구 방향 |
|------|------------------------------|----------------|
| 노이즈 모델링 | 명시적 잠재 변수 모델 | Diffusion model 기반 노이즈 모델링 탐색 |
| 추론 방법 | Langevin Dynamics (MCMC) | Flow-based 추론, Normalizing Flows |
| 백본 | VGG16 | ResNet, Swin Transformer |
| 레이블 소스 | 핸드크래프트 알고리즘 | Foundation model(SAM 등) 출력 활용 |
| 정규화 | Edge-aware smoothness | Contrastive learning 기반 정규화 |

### 5.3 본 논문의 위상

2020년 기준으로 본 논문은 비지도 현저성 검출 분야에서 **노이즈를 잠재 변수로 모델링한 최초의 체계적인 시도** 중 하나로, 이후 노이즈 레이블 기반 밀집 예측 연구의 방법론적 기반을 제공했습니다.

---

## 참고자료 (출처)

**주요 참고 논문 (본 논문의 Reference 기반)**:

1. **Zhang, J., Xie, J., Barnes, N.** (2020). "Learning Noise-Aware Encoder-Decoder from Noisy Labels by Alternating Back-Propagation for Saliency Detection." *arXiv:2007.12211v1*
2. **Han, T., Lu, Y., Zhu, S.C., Wu, Y.N.** (2017). "Alternating back-propagation for generator network." *AAAI 2017* [본 논문의 ABP 기반]
3. **Kingma, D., Welling, M.** (2014). "Auto-encoding variational bayes." *ICLR 2014* [VAE 비교 기반]
4. **Zhang, J., Zhang, T., Dai, Y., Harandi, M., Hartley, R.** (2018). "Deep unsupervised saliency detection: A multiple noisy labeling perspective." *CVPR 2018* [MNL 비교 대상]
5. **Neal, R.M.** (2010). "MCMC using Hamiltonian dynamics." *Handbook of Markov Chain Monte Carlo* [Langevin Dynamics 기반]
6. **Xie, J., Gao, R., Nijkamp, E., Zhu, S.C., Wu, Y.N.** (2020). "Representation learning: A statistical perspective." *Annual Review of Statistics and Its Application 7* [ABP 이론적 배경]
7. **Wang, Y., et al.** (2018). "Occlusion aware unsupervised learning of optical flow." *CVPR 2018* [Edge-aware smoothness loss 기반]
8. **Nguyen, D.T., et al.** (2019). "DeepUSPS: Deep robust unsupervised saliency prediction with self-supervision." *NeurIPS 2019*
9. **Simonyan, K., Zisserman, A.** (2014). "Very deep convolutional networks for large-scale image recognition." *CoRR abs/1409.1556* [VGG16 백본]
10. **Zhang, Y., Li, K., et al.** (2018). "Image super-resolution using very deep residual channel attention networks." *ECCV 2018* [RCA 모듈 기반]
