
# 3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes

> **논문 정보**
> - **저자**: Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Riccardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja Fidler, Nicholas Sharp, Zan Gojcic (NVIDIA Toronto AI Lab)
> - **발표**: SIGGRAPH Asia 2024 (Journal Track), *ACM Transactions on Graphics*, Vol. 43, No. 6, Article 232
> - **arXiv**: [2407.07090](https://arxiv.org/abs/2407.07090)
> - **프로젝트 페이지**: [gaussiantracer.github.io](https://gaussiantracer.github.io/)
> - **코드**: [github.com/nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 배경 및 핵심 주장

3D Gaussian Splatting(3DGS)과 같은 파티클 기반 방사 필드 표현은 복잡한 장면의 재구성과 렌더링에서 큰 성공을 거두었다. 그러나 기존 방법의 대부분은 파티클을 래스터화(rasterization)를 통해 렌더링하며, 이를 화면 공간 타일에 투영하여 정렬된 순서로 처리한다.

이 논문의 핵심 주장은: **래스터화 기반 접근 방식의 구조적 한계를 극복하기 위해, Gaussian 파티클 씬에 대한 고성능 GPU 레이 트레이싱을 도입하는 것**이 가능하며, 이를 통해 기존 방법과 동등하거나 우수한 품질을 실시간으로 달성할 수 있다는 것이다.

이 연구의 목표는 전역 조명(global illumination)이나 역 조명(inverse lighting) 문제에 대한 end-to-end 해결책을 제시하는 것이 아니라, 미래 연구를 위한 핵심 알고리즘 구성 요소인 **빠른 미분 가능 레이 트레이서(fast differentiable ray tracer)**를 제공하는 것이다.

### 1.2 주요 기여 (Contributions)

GPU 가속 Gaussian 파티클 레이 트레이서를 커스텀 설계하였으며, k-buffer 기반 히트 마칭(hits-based marching)을 통해 정렬된 교차점을 수집하고, 바운딩 메시 프록시(bounding mesh proxies)를 활용하여 빠른 레이-삼각형 교차를 가능하게 하며, 역전파(backward pass)를 지원하여 최적화를 가능하게 한다.

| 기여 항목 | 설명 |
|---|---|
| **GPU 가속 레이 트레이서** | 반투명 파티클을 위한 특수화된 렌더링 알고리즘 |
| **개선된 최적화 파이프라인** | 레이 트레이싱 기반 파티클 방사 필드를 위한 학습 파이프라인 |
| **일반화된 커널 함수** | 파티클 히트 수를 대폭 감소시키는 일반화 가우시안 커널 제안 |
| **다양한 응용** | 왜곡 카메라, 롤링 셔터, 이차 광선 효과, 확률적 샘플링 |

---

## 2. 상세 분석: 해결 문제 / 제안 방법 / 모델 구조 / 성능 / 한계

### 2.1 해결하고자 하는 문제

**래스터화의 구조적 한계:**

래스터화는 반사, 굴절, 그림자와 같은 현상을 처리하는 이차 광선(secondary rays)을 효율적으로 시뮬레이션할 수 없다. 또한 컴퓨터 비전 학습에서 일반적으로 사용되는 확률적 레이 샘플링도 지원하지 않는다. 이전 연구들도 이러한 기능의 필요성을 인식했지만, 제한적인 트릭이나 우회 방법에 의존할 수밖에 없었다.

래스터화는 완벽한 핀홀(pinhole) 카메라를 요구하므로 어안(fisheye) 렌즈를 사용하기 어렵다. 또한 반사, 굴절, 그림자 같은 현상을 처리하는 이차 광선을 효율적으로 시뮬레이션할 수 없다.

**기존 레이 트레이싱 방법의 한계:**

Gaussian 씬의 효율적인 레이 트레이싱은 아직 해결되지 않은 문제이다. 반투명 파티클을 위해 특별히 설계된 기존 알고리즘들조차도, 불균일하게 분포되고 밀집된 파티클의 대규모 수로 인해 효과적이지 못하다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 Gaussian 파티클 표현

3D Gaussian 파티클 $i$는 다음 파라미터로 정의된다:

$$\mathcal{G}_i = \{\boldsymbol{\mu}_i,\ \boldsymbol{\Sigma}_i,\ o_i,\ \mathbf{c}_i(\mathbf{d})\}$$

- $\boldsymbol{\mu}_i \in \mathbb{R}^3$: 파티클 중심 위치
- $\boldsymbol{\Sigma}_i \in \mathbb{R}^{3\times3}$: 공분산 행렬 (형태/방향 결정)
- $o_i \in [0, 1]$: 불투명도(opacity)
- $\mathbf{c}_i(\mathbf{d})$: 뷰 방향 $\mathbf{d}$에 따른 색상 (구면 조화 함수, SH 기반)

표준 Gaussian 커널 함수:

$$G(\mathbf{x}) = \exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

**일반화 Gaussian 커널 (Generalized Gaussian Kernel, 차수 $p$):**

논문은 기본 Gaussian 표현의 개선점으로 파티클 히트 수를 크게 줄이는 **일반화 커널 함수**의 활용을 제안한다.

$$G_p(\mathbf{x}) = \exp\!\left(-\frac{1}{2}\left[(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right]^p\right),\quad p \geq 1$$

$p=1$이면 표준 Gaussian, $p>1$이면 더 날카로운 감쇠(sharper falloff)로 인해 교차 파티클 수가 감소한다.

Gaussian 커널과 차수 2의 일반화 Gaussian 커널에 대한 평균 히트 수와 성능을 비교한 결과, 성능은 파티클 수보다 히트 수에 따라 결정됨이 재확인되었다. 이는 날카로운 범위가 히트 수를 줄이기 때문에 일반화 Gaussian 커널의 속도 향상 원인을 설명한다.

#### 2.2.2 볼륨 렌더링 수식 (Alpha Compositing)

레이 $\mathbf{r}(t) = \mathbf{r}_o + t\mathbf{r}_d$를 따라 렌더링되는 픽셀의 색상:

$$\mathbf{C}(\mathbf{r}) = \sum_{i=1}^{N} \mathbf{c}_i \cdot \alpha_i \cdot \prod_{j=1}^{i-1}(1 - \alpha_j)$$

각 파티클의 알파 값은:

$$\alpha_i = o_i \cdot G_p\!\left(\mathbf{r}_o + t_{\alpha,i}\,\mathbf{r}_d\right)$$

여기서 $t_{\alpha,i}$는 레이가 파티클 $i$에서 최대 응답을 가지는 깊이 파라미터로, 파티클 중심까지의 레이 최근접점:

$$t_{\alpha,i} = \frac{(\boldsymbol{\mu}_i - \mathbf{r}_o)^\top \boldsymbol{\Sigma}_i^{-1} \mathbf{r}_d}{\mathbf{r}_d^\top \boldsymbol{\Sigma}_i^{-1} \mathbf{r}_d}$$

누적 투과율(transmittance):

$$T_i = \prod_{j < i} (1 - \alpha_j)$$

렌더링은 $T_i < \epsilon$ (임계값 이하)가 되면 조기 종료(early termination)된다.

#### 2.2.3 BVH 기반 레이 트레이싱 파이프라인

렌더링은 네 가지 핵심 단계로 진행된다: ① 카메라 파라미터 기반 레이 생성, ② BVH 순회를 통한 교차 Gaussian 식별, ③ 깊이 기반 교차 Gaussian 정렬, ④ 최종 픽셀 색상을 계산하는 알파 블렌딩.

**바운딩 메시 프록시 (Icosahedron Mesh Proxy):**

파티클은 먼저 정이십면체(icosahedron) 메시로 바운딩된다. 여러 형태를 테스트한 결과, 늘어난 정이십면체 메시가 가장 우수한 결과를 제공하였다. 이 메시는 파티클을 단단하게 바운딩하면서도 하드웨어 가속 레이-삼각형 교차를 활용하여 거짓 양성(false-positive) 교차를 크게 줄인다. 바운딩 프록시의 크기는 파티클 불투명도에 따라 적응적으로 제한되어, 기여도가 낮은 파티클의 처리를 줄이고 효율성을 높인다.

**k-buffer 히트 기반 마칭:**

주어진 3D 파티클 집합에 대해 먼저 해당 바운딩 프리미티브를 구성하고 BVH에 삽입한다. 각 레이를 따라 수신되는 복사(radiance)를 계산하기 위해, 레이를 BVH에 대해 추적하여 다음 $k$개의 파티클을 가져온다. 그런 다음 교차된 파티클의 응답을 계산하고 복사를 누적한다. 모든 파티클이 평가되거나 투과율이 사전 정의된 임계값을 충족할 때까지 이 과정을 반복하고 최종 렌더링을 반환한다.

---

### 2.3 모델 구조

```
입력: 멀티뷰 이미지 + 카메라 파라미터 (핀홀/어안/롤링셔터 등 다양한 카메라 모델 지원)
        ↓
[1단계] 파티클 초기화
  - COLMAP 등 SfM 포인트 클라우드로부터 초기 Gaussian 위치 설정
        ↓
[2단계] BVH 구성
  - 각 Gaussian 파티클을 늘어난 정이십면체(icosahedron) 메시로 바운딩
  - GPU 가속 BVH(Bounding Volume Hierarchy) 구성
        ↓
[3단계] 레이 트레이싱 렌더링
  - 픽셀마다 레이 발사 (임의 카메라 모델 지원)
  - BVH 순회 → k-buffer 히트 수집 → 깊이 정렬 → Alpha Compositing
        ↓
[4단계] 역전파 (Backward Pass)
  - 렌더링 손실 L = L_photometric + L_reg 계산
  - Gaussian 파라미터 {μ, Σ, o, c} 업데이트
  - Adaptive Densification (3D 세계 공간 기울기 기반)
        ↓
[5단계] 반복 및 BVH 재구성
  - 파티클 파라미터 업데이트 시마다 BVH 재구성
  - 수렴까지 반복
```

파티클 씬 피팅을 위해 Kerbl et al.(2023)의 최적화 방식(pruning, cloning, splitting 포함)을 채택한다. 한 가지 중요한 변경 사항이 있는데, 기존 3DGS는 화면 공간 그래디언트를 클로닝 및 분할 기준으로 사용하지만, 본 연구의 더 일반화된 설정에서는 화면 공간 그래디언트가 사용 가능하지도 않고 의미 있지도 않으므로, **3D 월드 공간 그래디언트**를 동일한 목적으로 사용한다.

---

### 2.4 성능 향상

표준 멀티뷰 벤치마크에서 레이 트레이싱은 Kerbl et al.(2023)의 3DGS 래스터라이저와 거의 동등하거나 우수한 품질에 근접하며, 실시간 렌더링 프레임 속도를 유지한다. 더 중요하게는, 그림자 및 반사와 같은 이차 광선 효과, 고왜곡 및 롤링 셔터가 있는 카메라로부터의 렌더링, 확률적으로 샘플링된 레이를 이용한 학습 등 레이 트레이싱으로 쉽고 효율적으로 가능한 다양한 새로운 기술들을 시연한다.

**왜곡 카메라에서의 품질 향상:**

ZipNeRF 데이터셋의 두 장면에 대해 왜곡 해제된(pinhole) 뷰와 원본 fisheye 카메라로 모델을 학습시킨 결과, 원본 왜곡 뷰로 학습하는 것이 더 높은 품질의 출력을 생성함이 명확하게 나타났다.

**속도 비교 (기존 연구 GRTX와의 비교):**

레이 트레이싱 기반 Gaussian 렌더링은 래스터화에 비해 평균 약 3.04배 느리며, 3DGS는 추가 최적화로 더 높은 성능을 달성할 수 있다.

---

### 2.5 한계 (Limitations)

3DGRT는 전용 레이 트레이싱 하드웨어(RT 코어)를 필요로 하며, 3DGS보다 느리다.

동적 씬에서 3DGRT를 사용할 때 잠재적인 문제가 있다. BVH는 학습 중에 정기적으로 재구성되어야 하므로 더 높은 계산 비용이 발생한다.

| 한계 항목 | 상세 내용 |
|---|---|
| **하드웨어 의존성** | NVIDIA RT 코어 탑재 GPU(RTX 시리즈) 필수 |
| **속도** | 래스터화 대비 약 3배 느린 렌더링 속도 |
| **동적 씬** | BVH 재구성 비용으로 인한 동적 씬 처리 어려움 |
| **글로벌 일루미네이션** | 전역 조명/역 조명의 end-to-end 해결책은 제공하지 않음 |
| **메모리** | BVH 구조로 인한 추가 메모리 사용 |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 왜곡 카메라 모델 지원 (핵심 일반화 능력)

래스터화와 달리, 레이 트레이싱은 장면 감독을 위해 완벽한 핀홀 카메라를 필요로 하지 않는다. 대신 레이 트레이싱은 다른 왜곡 카메라 모델을 사용한 학습을 지원한다.

래스터화 기반 접근 방식과 달리, 레이 트레이싱 기반 방법은 왜곡된 어안 렌즈와 같은 복잡한 카메라 모델을 자연스럽게 지원한다. 이를 일반 원근(perspective) 카메라 등 다른 카메라 모델로 재렌더링할 수 있어 보이지 않는 참조에 대한 높은 재구성 품질을 달성한다. 레이 트레이싱은 또한 센서 운동으로 인한 롤링 셔터 왜곡과 같은 시간 종속적 효과를 자연스럽게 보상한다.

### 3.2 파티클 표현의 일반화

장면을 파티클로 표현하는 기본 접근 방식은 Gaussian 커널에 국한되지 않으며, 최근 연구에서는 이미 여러 자연스러운 일반화가 제안되었다. 본 연구의 레이 트레이싱 방식과 그 이점 및 응용은 파티클 기반 씬 표현으로 더 폭넓게 일반화된다.

### 3.3 확률적 레이 샘플링 (Stochastic Ray Sampling)

효율적인 레이 트레이싱은 거울, 굴절, 그림자와 같은 이차 광선 효과, 롤링 셔터 효과가 있는 고왜곡 카메라, 그리고 확률적 레이 샘플링까지 다양한 고급 기술에 문을 열어 준다.

확률적 레이 샘플링은 다음의 Monte Carlo 적분 형태로 훈련 손실을 구성할 수 있게 한다:

$$\mathcal{L} = \mathbb{E}_{\mathbf{r} \sim \mathcal{D}}\left[\left\|\mathbf{C}(\mathbf{r}) - \mathbf{C}^*(\mathbf{r})\right\|^2\right]$$

이를 통해 이미지 전체 픽셀을 사용하지 않고 랜덤 샘플링된 레이만으로도 학습이 가능해져 다양한 뷰 분포나 도메인에 대한 일반화가 용이해진다.

### 3.4 이차 광선 효과를 통한 역 렌더링 일반화

효율적인 레이 트레이싱은 거울, 굴절, 그림자와 같은 이차 광선 효과와 같은 고급 기술을 가능하게 한다. 또한 롤링 셔터 효과가 있는 고왜곡 카메라를 지원하고 확률적 레이 샘플링까지 지원한다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

#### 4.1.1 후속 연구로의 연결

3DGRT의 한계를 완화하기 위해 3DGUT가 제안되었으며, 이는 래스터화 프레임워크 내에서 왜곡 카메라 및 이차 광선을 지원한다. 3DGRT와 3DGUT의 렌더링 공식을 정렬함으로써 **3DGRUT**라는 하이브리드 접근 방식이 도입되었다. 이 기술은 래스터화를 통해 기본 레이를, 레이 트레이싱을 통해 이차 광선을 렌더링하여 두 방법의 장점을 결합한다.

이 흐름은 아래와 같이 정리된다:

```
3DGS (2023) → 래스터화만 지원
    ↓
3DGRT (2024) → 레이 트레이싱 도입, 왜곡 카메라/이차 광선 지원
    ↓
3DGUT (CVPR 2025, Oral) → 래스터화 프레임워크 내에서 동일 기능
    ↓
3DGRUT → 래스터화(Primary) + 레이 트레이싱(Secondary) 하이브리드
```

#### 4.1.2 역 렌더링 / 재조명 연구에의 영향

그림자 및 반사와 같은 이차 광선 효과, 고왜곡 및 롤링 셔터 카메라로부터의 렌더링, 확률적으로 샘플링된 레이를 이용한 학습 등 레이 트레이싱으로 쉽고 효율적으로 가능한 다양한 새로운 기술들을 시연한다.

#### 4.1.3 자율주행 / 로보틱스 적용

주목할 만한 응용 분야 중 하나는 자율주행이다. 제안된 레이 트레이싱 알고리즘의 복잡한 카메라 모델 처리 능력은 자율주행 시나리오의 시뮬레이션 및 테스트에 매우 유용하다.

---

### 4.2 앞으로 연구 시 고려할 점

| 고려 사항 | 설명 |
|---|---|
| **하드웨어 독립성 확보** | RT 코어 없는 환경에서도 동작 가능한 소프트웨어 레이 트레이서 또는 Vulkan 기반 구현 필요 |
| **동적 씬 처리** | BVH 재구성 비용 절감을 위한 동적/증분적 BVH 업데이트 알고리즘 연구 |
| **전역 조명 통합** | 논문 자체가 밝히듯, 글로벌 일루미네이션과의 통합은 여전히 미해결 문제 |
| **메모리 효율화** | BVH 구조 + 파티클 저장에 따른 메모리 증가 문제 |
| **속도-품질 트레이드오프** | 래스터화 대비 3배 느린 속도 문제 해결 (GRTX 등 후속 연구 참조) |
| **도메인 일반화** | 실내/실외/대형 씬/동적 씬에 대한 일관된 성능 검증 |
| **역 렌더링 통합** | 재조명(relighting), 재질 분해(material decomposition)와의 결합 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

3DGS는 렌더 시간이나 메모리 사용량 감소(Niedermayr et al., 2023; Fan et al., 2023; Papantonakis et al., 2024), 표면 표현 개선(Guédon & Lepetit, 2023; Huang et al., 2024), 대규모 씬 지원(Ren et al., 2024; Kerbl et al., 2024) 등 수많은 후속 연구를 촉발했다.

| 방법 | 연도 | 렌더링 방식 | 왜곡 카메라 | 이차 광선 | 주요 특징 |
|---|---|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | 볼륨 레이캐스팅 | ✅ | ✅ (원리상) | 암시적 표현, 느린 렌더링 |
| **3DGS** (Kerbl et al.) | 2023 | 래스터화 | ❌ | ❌ | 실시간, 명시적 표현 |
| **2DGS** (Huang et al.) | 2024 | 래스터화 | ❌ | ❌ | 2D 서피스 기반, 기하 정확도 향상 |
| **RadSplat** (Niemeyer et al.) | 2024 | 래스터화 | ❌ | ❌ | 900+ FPS, NeRF 지식 증류 |
| **3DGRT (본 논문)** | 2024 | **레이 트레이싱** | ✅ | ✅ | 빠른 미분 가능 레이 트레이서 |
| **3DGUT** (Wu et al.) | 2025 | 래스터화 (UT 기반) | ✅ | 제한적 | RT 코어 없이도 왜곡 카메라 지원 |
| **3DGRUT** | 2025 | **하이브리드** | ✅ | ✅ | 기본 레이: 래스터화 + 이차 레이: RT |
| **GRTX** (Fang et al.) | 2026 | 레이 트레이싱 | ✅ | ✅ | BVH 최적화로 평균 4.36× 속도 향상 |

BVH 순회가 실행 시간을 지배하며, 정렬 및 블렌딩의 기여는 미미하다. 3DGS는 2D 투영 후 어떤 픽셀이 Gaussian과 교차하는지 직접 식별할 수 있는 반면, 3DGRT는 교차 프리미티브를 찾기 위해 각 레이에 대해 루트에서 리프 노드까지 포인터 체이싱을 수행해야 한다. Bonsai와 같이 특정 영역에 수많은 소형 Gaussian이 집중된 씬에서는 이 영역을 통과하는 레이의 순회 시간이 증가하여 성능 격차가 더 커진다.

---

## 참고 자료 및 출처

1. **[논문 원문 (arXiv)]** Nicolas Moenne-Loccoz et al., "3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes," arXiv:2407.07090, 2024. https://arxiv.org/abs/2407.07090
2. **[ACM TOG 공식 출판]** 3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes, *ACM Transactions on Graphics*, Vol. 43, No. 6, Article 232. https://dl.acm.org/doi/10.1145/3687934
3. **[프로젝트 페이지]** GaussianTracer.github.io, NVIDIA Toronto AI Lab. https://gaussiantracer.github.io/
4. **[NVIDIA 공식 페이지]** NVIDIA Spatial Intelligence Lab - 3DGRT. https://research.nvidia.com/labs/toronto-ai/3DGRT/
5. **[공식 코드 저장소]** nv-tlabs/3dgrut, GitHub. https://github.com/nv-tlabs/3dgrut
6. **[HuggingFace 논문 페이지]** Paper page - 3D Gaussian Ray Tracing. https://huggingface.co/papers/2407.07090
7. **[Semantic Scholar]** 3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes. https://www.semanticscholar.org/paper/3D-Gaussian-Ray-Tracing
8. **[ResearchGate]** 3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes. https://www.researchgate.net/publication/382111339
9. **[후속 연구: GRTX]** "GRTX: Efficient Ray Tracing for 3D Gaussian-Based Rendering," arXiv:2601.20429, 2026. https://arxiv.org/abs/2601.20429
10. **[후속 연구: 3DGUT]** Qi Wu et al., "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting," CVPR 2025 (Oral).
11. **[관련 리뷰]** Radiancefields.com - 3D Gaussian Ray Tracing. https://radiancefields.com/3d-gaussian-ray-tracing
12. **[3DGS 관련 서베이]** Chen G, Wang W., "A Survey on 3D Gaussian Splatting," *ACM Computing Surveys*, 2026. https://dl.acm.org/doi/10.1145/3807511
