# The Theory of Diffraction Tomography

**참고 논문:**
- Müller, P., Schürmann, M., & Guck, J. (2016). *The Theory of Diffraction Tomography*. arXiv:1507.00466v3 [q-bio.QM].

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
이 논문은 **회절 단층촬영(Diffraction Tomography, DT)의 포괄적 이론 리뷰**로, 고전적 단층촬영(Optical Projection Tomography, OPT)이 빛의 파동적 성질을 무시함으로써 발생하는 한계를 극복하고자 한다. 가시광선 파장 영역에서 단일 세포와 같은 생물학적 샘플을 이미징할 때, 회절 효과를 명시적으로 모델링함으로써 굴절률 분포 재구성의 정확도를 향상시킬 수 있음을 이론적으로 정립한다.

### 주요 기여
1. **통일된 표기법 확립**: 기존 문헌(Wolf, Devaney, Kak & Slaney 등)에서 각기 다른 표기법을 사용하던 회절 단층촬영 이론을 단일하고 일관된 표기 체계로 정리.
2. **파동방정식으로부터의 완전한 이론 유도**: 헬름홀츠 방정식 → Born/Rytov 근사 → 푸리에 회절 정리 → 역전파(Backpropagation) 알고리즘으로 이어지는 단계적 유도.
3. **2D 및 3D 역전파 알고리즘의 상세 구현 기술**: 수치적 구현에 필요한 FFT 기반 알고리즘을 명시적으로 제시.
4. **Rytov 근사의 유효성 분석**: Born 근사보다 더 넓은 적용 범위를 갖는 Rytov 근사의 수학적 조건 도출.
5. **고전 단층촬영과의 연결**: Rytov 근사 기반 회절 단층촬영이 단파장 극한($\lambda \to 0$)에서 고전적 푸리에 슬라이스 정리로 수렴함을 증명.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

**고전 단층촬영의 근본적 한계**: 가시광선을 이용한 생물학적 세포 이미징 시, 빛의 파장이 세포 내 구조물의 크기와 비슷해 회절 효과가 무시될 수 없다. 기존의 **광학 투영 단층촬영(OPT)**은 빛이 직선으로 전파된다는 가정 하에 **역투영(Backprojection)** 알고리즘을 사용하며, 이는 회절에 의한 파면 변형을 무시한다.

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 파동방정식과 헬름홀츠 방정식

시간-독립적 파동 전파는 헬름홀츠 방정식으로 기술된다:

$$\left(\nabla^2 + k(\mathbf{r})^2\right) u(\mathbf{r}) = 0 \tag{2.2}$$

여기서 $k(\mathbf{r}) = k_m \frac{n(\mathbf{r})}{n_m}$, $k_m = \frac{2\pi n_m}{\lambda}$.

산란 포텐셜 $f(\mathbf{r})$을 도입하면 비동차 헬름홀츠 방정식이 된다:

$$\left(\nabla^2 + k_m^2\right) u(\mathbf{r}) = -f(\mathbf{r}) u(\mathbf{r}) \tag{2.8}$$

$$f(\mathbf{r}) = k_m^2 \left[\left(\frac{n(\mathbf{r})}{n_m}\right)^2 - 1\right] \tag{2.9}$$

Green's 함수 (3D):

$$G(\mathbf{r} - \mathbf{r}') = \frac{\exp(ik_m |\mathbf{r} - \mathbf{r}'|)}{4\pi |\mathbf{r} - \mathbf{r}'|} \tag{2.11}$$

---

#### Step 2: Born 근사

전체 산란파 $u(\mathbf{r}) = u_0(\mathbf{r}) + u_s(\mathbf{r})$에서, **Born 근사**는 내부 필드를 입사파로 대체한다:

$$u_B(\mathbf{r}) = \int d^3r' \, G(\mathbf{r} - \mathbf{r}') f(\mathbf{r}') u_0(\mathbf{r}') \tag{3.6}$$

**Born 근사 유효 조건**: 광학 경로차가 파장보다 훨씬 작아야 한다:

$$s(n_s - n_m) \ll \lambda \tag{3.15}$$

즉, 광학적으로 얇은 샘플에만 적용 가능하다.

---

#### Step 3: Rytov 근사

산란파를 복소 위상으로 표현한다:

$$u(\mathbf{r}) = \exp(\varphi(\mathbf{r})), \quad \varphi(\mathbf{r}) = \varphi_0(\mathbf{r}) + \varphi_s(\mathbf{r}) \tag{3.18}$$

Rytov 위상 $\varphi_R(\mathbf{r})$은 Born 근사 결과로부터:

$$\varphi_R(\mathbf{r}) = \frac{\int d^3r' \, G(\mathbf{r} - \mathbf{r}') f(\mathbf{r}') u_0(\mathbf{r}')}{u_0(\mathbf{r})} = \frac{u_B(\mathbf{r})}{u_0(\mathbf{r})} \tag{3.36, 3.37}$$

**Rytov 근사 유효 조건**: 절대 위상 변화가 아닌 **굴절률의 기울기**에 의존:

$$|\nabla n(\mathbf{r})| \ll \frac{\sqrt{2 n_m |n(\mathbf{r}) - n_m|}}{s}, \quad s > \lambda \tag{3.52}$$

이로 인해 Rytov 근사는 광학적으로 두꺼운 샘플에도 적용 가능하다.

---

#### Step 4: 푸리에 회절 정리 (Fourier Diffraction Theorem)

**2D 경우** (Eq. 4.24):

$$\hat{U}_{B,\phi_0}(k_{Dx}) = \frac{ia_0}{k_m}\sqrt{\frac{\pi}{2}} \frac{1}{M} \hat{F}(k_m(\mathbf{s} - \mathbf{s}_0)) \exp(ik_m M l_D) \tag{4.23}$$

역으로 풀면:

$$\hat{F}(k_m(\mathbf{s} - \mathbf{s}_0)) = -\sqrt{\frac{2}{\pi}} \frac{ik_m}{a_0} M \hat{U}_{B,\phi_0}(k_{Dx}) \exp(-ik_m M l_D) \tag{4.24}$$

여기서 $M = \sqrt{1 - (k_{Dx}/k_m)^2}$.

**3D 경우** (Eq. 5.19):

$$\hat{F}(\mathbf{k}_D - k_m \mathbf{s}_0) = -\sqrt{\frac{2}{\pi}} \frac{iMk_m}{a_0} \exp(-ik_m M l_D) \hat{U}_{B,\phi_0}(\mathbf{k}_D) \tag{5.19}$$

여기서 $M = \frac{1}{k_m}\sqrt{k_m^2 - k_{Dx}^2 - k_{Dy}^2}$.

**핵심 차이점**: 고전 단층촬영에서 데이터는 푸리에 공간의 **직선**에 분포하지만, 회절 단층촬영에서는 **반원호(2D)** 또는 **반구면(3D)**에 분포한다.

---

#### Step 5: 역전파 알고리즘 (Backpropagation Algorithm)

**3D 역전파 알고리즘** (Eq. 5.87):

$$f(\mathbf{r}) = \frac{-ik_m}{(2\pi)^2 a_0} \int_0^{2\pi} d\phi_0 \int_{-k_m}^{k_m} dk_{Dx} \int_{-k_m}^{k_m} dk_{Dy} \, |k_{Dx}| \hat{U}_{B,\phi_0}(k_{Dx}, k_{Dy}) \exp(-ik_m M l_D) \exp[i(k_{Dx} \mathbf{t}_\perp + k_m(M-1)\mathbf{s}_0)\mathbf{r}]$$

여기서:

$$k_m M = \sqrt{k_m^2 - k_{Dx}^2 - k_{Dy}^2} \tag{5.88}$$

$$\mathbf{t}_\perp = \left(\cos\phi_0, \frac{k_{Dy}}{k_{Dx}}, \sin\phi_0\right)^\top, \quad \mathbf{s}_0 = (-\sin\phi_0, 0, \cos\phi_0)^\top$$

**수치 구현** (FFT 기반, Eq. 6.23):

```math
f(\mathbf{r}) = \frac{-ik_m}{2\pi} \sum_{j=1}^{N} \Delta\phi_0 \times D_{-\phi_j} \left\{ \text{FFT}^{-1}_{2D} \left\{ \frac{\hat{U}_{B,\phi_j}(k_{Dx}, k_{Dy})}{u_0(l_D)} |k_{Dx}| \exp[ik_m(M-1)(z_{\phi_j} - l_D)] \right\} \right\} \tag{6.23}
```

---

### 2.3 모델 구조

```
입사 평면파 u₀(r)
        ↓
    [산란 샘플]  n(r) 분포
        ↓
검출기에서 u(r) = u₀(r) + uₛ(r) 측정
        ↓
  [위상 측정 - DHM 등]
        ↓
Born 또는 Rytov 근사로 uB(r) 계산
        ↓
1D/2D Fourier 변환 → Û_B,φ₀(k_D)
        ↓
푸리에 회절 정리 적용:
F̂(k_D - kₘs₀) ∝ M · Û_B,φ₀(k_D) · exp(-ikₘMl_D)
        ↓
역전파 알고리즘 (FFT⁻¹ + 회전 + 누적)
        ↓
굴절률 분포 f(r) 재구성
```

**푸리에 공간 분포**: 각 투영각 $\phi_0$마다 데이터는 반구면(반지름 $k_m$)에 배치되며, $\phi_0$를 $0 \to 2\pi$ 회전 시 **호른 토러스(horn torus)** 형태로 채워진다.

---

### 2.4 성능 향상

논문의 수치 시뮬레이션 결과(Mie 이론 기반 정확해와 비교):

| 방법 | 조건 | 결과 |
|------|------|------|
| 역투영 (OPT) | 회절 무시 | 굴절률 경계 흐릿함, 정량 오차 큼 |
| Born + 역전파 | 250 projections | 굴절률 분포 복구 실패 (광학적 두꺼운 샘플) |
| **Rytov + 역전파** | **50 projections** | **정확해에 근접** |
| Rytov + 역전파 | 250 projections | 정확해와 거의 일치 |

- **2D 시뮬레이션** (Figure 4.2): $n_m = 1.333$, $\epsilon_n = 0.006$, 반지름 $30\lambda$ 실린더에서 Rytov 근사가 Born 근사보다 월등히 우수.
- **3D 시뮬레이션** (Figure 5.5): $n_m = 1.0$, $\epsilon_n = 0.006$, 반지름 $14\lambda$ 구에서 Rytov 근사의 x축 방향 재구성은 정확해와 거의 일치.

**해상도 향상**: 회절 단층촬영은 데이터가 반구면에 분포하므로 최대 주파수가 $\sqrt{2} k_m$에 달한다:

$$f_{\max} = \sqrt{2} \cdot \frac{k_m}{2\pi} = \frac{\sqrt{2} n_m}{\lambda} \tag{5.21}$$

즉, 이론적 광학 해상도가 $\lambda/(\sqrt{2} n_m)$까지 향상된다.

---

### 2.5 한계

1. **Born 근사의 엄격한 유효 범위**: $s(n_s - n_m) \ll \lambda$ 조건 → 광학적으로 두꺼운 생물학적 샘플에 부적합.
2. **Rytov 근사의 한계**: 굴절률 기울기가 작아야 하며, 굴절률 불연속면(날카로운 경계)에서 정확도 저하.
3. **단축 회전으로 인한 Missing Apple Core 문제**: 한 축을 중심으로만 회전 시, 푸리에 공간에서 극 방향 데이터 부재 → 회전축 방향으로 방향성 블러링 발생 (Figure 5.4, 5.5의 y축 방향 아티팩트).
4. **360° 데이터 필요**: 역전파 알고리즘은 $0 \sim 2\pi$ 전 방위 데이터를 요구 (역투영은 $0 \sim \pi$로 충분).
5. **스칼라 파동 근사**: 벡터 전자기장의 편광 효과 무시.
6. **위상 언래핑 필요**: Rytov 근사 구현 시 $2\pi$ 모듈러스 문제 처리 필요.
7. **계산 복잡도**: 3D 역전파는 각 투영마다 2D FFT + 3D 체적 회전 및 누적이 필요해 고전 역투영 대비 계산 비용이 높다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Rytov 근사의 일반화 우월성

논문에서 가장 중요하게 다루는 일반화 관련 내용은 **Born 대비 Rytov 근사의 넓은 유효 범위**이다.

- **Born 근사**: 절대 위상 변화 $\Delta\Phi \ll 2\pi$, 즉 $s(n_s - n_m) \ll \lambda$ 필요. 샘플 두께와 굴절률 차이 모두에 제약.
- **Rytov 근사**: 위상 기울기 조건:

$$\frac{|d\varphi_s(\mathbf{r})|}{2\pi} \ll \frac{\sqrt{2n_m|\epsilon_n(\mathbf{r})|}}{|\mathbf{r}|/\lambda} \tag{3.48}$$

이는 **절대 위상 변화가 아니라 굴절률의 공간적 기울기**에 의존하므로, 두꺼운 샘플이라도 내부 굴절률 변화가 완만하면 유효하다. 생물학적 세포(느린 굴절률 변화)에 특히 적합하다.

### 3.2 Rytov 근사의 단파장 극한에서의 수렴

논문은 Rytov 근사 기반 회절 단층촬영이 고전 단층촬영의 **상위 호환**임을 수학적으로 증명한다 (Section 5.3):

$$\lambda \to 0 \Rightarrow k_m \to \infty \Rightarrow M^* \to 1$$

이 극한에서 푸리에 회절 정리가 **푸리에 슬라이스 정리**로 수렴한다:

$$\int_{-A}^{+A} dz \, f(\mathbf{r}) = -2ik_m \varphi_{R,\phi_0}(\mathbf{r}_D) \tag{5.39}$$

이는 투영 데이터가 선적분으로 환원됨을 의미하며, 이 논문의 방법이 **더 일반적인 프레임워크**임을 보여준다.

### 3.3 데이터 요구량과 일반화

- Rytov 근사는 **더 적은 투영 수**로도 양질의 재구성 가능 (50 projections vs 250 projections, Figure 4.1e vs 4.1f).
- 이는 데이터가 희소한 실험 환경에서 더 robust한 일반화를 의미한다.

### 3.4 다른 물리 도메인으로의 일반화

논문 결론부에서 명시적으로 언급:
> "The reconstruction methods described are applicable not only to optical diffraction tomography but also to **ultrasonic diffraction tomography** whose underlying principle is the scalar wave equation."

스칼라 파동방정식을 기반으로 하는 모든 도메인(광학, 초음파, 전자기파 등)에 이론이 적용 가능하다.

### 3.5 Missing Apple Core 문제와 일반화 한계

단축 회전으로 인한 데이터 누락은 일반화 성능 저하의 주요 원인이다. 이를 해결하기 위해 논문은 정규화 기법([24]–[27])을 제안하지만, 상세한 구현은 범위 외로 남긴다. 이 부분이 추후 연구에서 가장 중요한 일반화 개선 방향이다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 영향

1. **기준 이론 문서로서의 역할**: 통일된 표기법과 단계적 유도는 후속 연구자들이 회절 단층촬영을 구현하고 확장하는 데 필수적인 기준점을 제공한다. 실제로 ODTbrain [21]과 같은 오픈소스 라이브러리의 이론적 기반이 된다.

2. **생물 이미징 분야 영향**: 단일 세포의 3D 굴절률 분포를 비침습적으로 측정하는 도구로서 세포 생물학, 병리학 연구에 직접적 영향. 세포의 건조 질량, 형태, 내부 구조 정량화 가능.

3. **역산란 문제(Inverse Scattering)와의 연결**: 이 논문의 이론은 보다 일반적인 역산란 문제의 특수 경우로, 비선형 역산란 방법(Distorted Born Iterative Method, DBIM 등)의 선형 근사 기반을 제공한다.

4. **딥러닝 기반 이미징과의 융합**: 회절 단층촬영의 물리적 순방향 모델을 딥러닝의 사전 정보(physics-informed prior)로 활용하는 연구에 이론적 토대 제공.

### 4.2 앞으로 연구 시 고려할 점

1. **Missing Apple Core 문제 해결**: 단축 회전의 한계를 극복하기 위한 다축 회전, 복수 조명 방향, 압축 센싱(Compressed Sensing) 기반 재구성 연구 필요.

2. **비선형 역산란으로의 확장**: Born/Rytov 근사의 선형 한계를 넘어서는 Distorted Born Iterative Method(DBIM) 또는 완전 파동방정식 기반 역산란(Full-Wave Inversion) 방법론 필요.

3. **흡수성 샘플 처리**: 복소 굴절률 $\tilde{n} = n + i\kappa$ (허수부 존재) 처리를 위한 이론 확장. 현재 이론은 주로 실수 굴절률 가정.

4. **벡터 파동 효과**: 스칼라 근사의 한계 극복을 위한 편광 효과(vectorial diffraction) 포함.

5. **위상 측정 기술과의 통합**: Digital Holographic Microscopy(DHM) 외에 Quantitative Phase Imaging(QPI), 간섭계 측정 오차, 노이즈 모델을 이론에 통합.

6. **계산 효율화**: 3D 역전파의 계산 복잡도($O(N^3 \log N)$ per projection)를 줄이기 위한 병렬화, GPU 가속, 근사 알고리즘 연구.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> **주의**: 아래 연구들은 제가 학습 데이터 기반으로 알고 있는 내용이며, 해당 논문 원문을 직접 확인하지 못했습니다. 내용의 정확성이 100% 보장되지 않을 수 있으므로 참조 시 원문 확인을 권장합니다.

### 5.1 딥러닝 기반 회절 단층촬영

**방향**: Müller et al.의 선형 역전파 알고리즘의 한계(Born/Rytov 근사)를 딥러닝으로 보완.

대표적 접근:
- **Physics-Informed Neural Networks (PINNs)**: 헬름홀츠 방정식을 손실 함수에 포함하여 비선형 산란 효과를 학습. Müller et al.의 순방향 모델(Forward Model)을 데이터 생성기로 활용.
- **Unrolled Optimization Networks**: 역전파 알고리즘의 반복적 수렴 과정을 신경망 레이어로 표현, 적은 투영 수에서도 Missing Cone 문제 완화.

**Müller et al. 대비**: 선형 근사 한계 극복 가능하나, 학습 데이터 의존성 및 일반화 불확실성 존재.

### 5.2 비선형 역산란 방법

**방향**: Born/Rytov 선형 근사를 넘어서는 완전 비선형 역산란.

- **Distorted Born Iterative Method (DBIM)**: 각 반복에서 현재 추정 굴절률 분포를 기반으로 선형화하여 반복적 업데이트. Müller et al.의 푸리에 회절 정리를 각 반복의 내부 솔버로 사용.
- **Contrast Source Inversion (CSI)**: 산란 포텐셜과 내부 전기장을 동시에 최적화.

**Müller et al. 대비**: 더 높은 굴절률 대비(contrast) 샘플에도 적용 가능하나, 계산 비용이 수십~수백 배 증가.

### 5.3 Multi-angle Illumination 및 Synthetic Aperture

**방향**: 단축 회전의 Missing Apple Core 문제를 다방향 조명으로 해결.

- 복수의 조명 방향을 사용해 푸리에 공간의 빈 영역(missing cone)을 채우는 방법. Müller et al.이 명시한 아티팩트([23] Vertu et al.) 해결 시도.

### 5.4 Intensity-only (위상 없이) 회절 단층촬영

**방향**: DHM과 같은 위상 측정 장비 없이 강도(intensity)만으로 굴절률 재구성.

- Müller et al.은 명시적으로 위상 측정(DHM 등)을 요구했으나, 최근 연구는 복수 초점면 강도 이미지(Transport of Intensity Equation, TIE 등)에서 위상을 추출하여 DT에 적용.

---

### 비교 요약표

| 항목 | Müller et al. (2016) | 2020년 이후 연구 |
|------|---------------------|-----------------|
| 산란 모델 | Born/Rytov (1차 선형 근사) | 비선형 역산란, 딥러닝 |
| Missing cone 처리 | 정규화 언급 (미상세) | 딥러닝, 다축 조명, CS |
| 계산 방법 | FFT 기반 해석적 역전파 | GPU 가속, 신경망 |
| 위상 요구 | 필수 (DHM) | Intensity-only 방법 등장 |
| 이론적 기반 | 완전한 해석적 유도 | Müller et al. 이론 위에 구축 |
| 적용 범위 | 소~중간 굴절률 대비 | 높은 굴절률 대비까지 확장 |

---

## 참고 자료 (논문 내 인용 문헌)

1. Wolf, E. (1969). *Three-dimensional structure determination of semi-transparent objects from holographic data*. Optics Communication 1.4.
2. Devaney, A.J. (1981). *Inverse-scattering theory within the Rytov approximation*. Optics Letters 6.8.
3. Devaney, A.J. (1982). *A filtered backpropagation algorithm for diffraction tomography*. Ultrasonic Imaging 4.4.
4. Kak, A.C. & Slaney, M. (2001). *Principles of Computerized Tomographic Imaging*. SIAM.
5. Müller, P., Schürmann, M., & Guck, J. (2015). *ODTbrain: a Python library for full-view, dense diffraction tomography*. BMC Bioinformatics 16.1.
6. Vertu, S. et al. (2009). *Diffraction microtomography with sample rotation: influence of a missing apple core in the recorded frequency space*. Central European Journal of Physics 7.1.
7. Chen, B. & Stamnes, J.J. (1998). *Validity of Diffraction Tomography Based on the First Born and the First Rytov Approximations*. Applied Optics 37.14.
8. Slaney, M., Kak, A.C., & Larsen, L.E. (1984). *Limitations of Imaging with First-Order Diffraction Tomography*. IEEE Transactions on Microwave Theory and Techniques 32.8.
9. Sung, Y. et al. (2009). *Optical diffraction tomography for high resolution live cell imaging*. Optics Express 17.1.
10. **본 논문**: Müller, P., Schürmann, M., & Guck, J. (2016). *The Theory of Diffraction Tomography*. arXiv:1507.00466v3.
