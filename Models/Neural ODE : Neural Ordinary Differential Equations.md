
# Neural Ordinary Differential Equation

> **논문 정보**: Chen, R.T.Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). *Neural Ordinary Differential Equations*. NeurIPS 2018 (Best Paper Award). arXiv:1806.07366

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 새로운 딥 뉴럴 네트워크 모델군을 제안하며, 이산적인 은닉층(hidden layer)의 시퀀스를 명시하는 대신, 은닉 상태(hidden state)의 **도함수**를 뉴럴 네트워크로 매개변수화(parameterize)한다. 네트워크의 출력은 블랙박스 미분방정식 솔버(black-box ODE solver)를 통해 계산된다.

이러한 연속 깊이(continuous-depth) 모델은 **상수 메모리 비용**, 입력별 적응적 평가 전략, 그리고 수치적 정밀도와 속도 간의 명시적 트레이드오프를 제공하며, 연속 깊이 잔차 네트워크(ResNet)와 연속 시간 잠재 변수 모델(latent variable model)에서 이러한 속성을 실증했다.

**주요 기여 3가지:**

| 기여 | 설명 |
|------|------|
| ① 연속 깊이 네트워크 | ResNet의 이산 레이어를 ODE의 연속 역학으로 대체 |
| ② Adjoint 감도 분석 방법 | $O(1)$ 메모리 복잡도의 역전파 알고리즘 제안 |
| ③ 연속 정규화 흐름(CNF) | Normalizing Flow를 자유 형식 연속 역학으로 확장 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 RNN이나 ResNet과 같은 순차적 블록 구조의 네트워크는 다음과 같이 표현된다:

$$h_{t+1} = h_t + f(h_t, \theta_t)$$

논문의 핵심 질문은 "이러한 네트워크에서 스텝 크기를 점진적으로 줄이면 (즉, 레이어 수를 무한히 늘리면) 더 나은 결과를 얻을 수 있는가?"이다. 스텝 크기를 무한소로 줄이면 결국 미분방정식(ODE)에 도달하게 된다.

기존 이산 네트워크의 한계:
- **고정된 깊이**: 레이어 수가 미리 결정됨
- **메모리 비용**: 역전파 시 모든 중간 활성값 저장 필요 → $O(L)$
- **유연성 부족**: 입력별 적응적 계산 불가

### 2.2 제안하는 방법 (수식 포함)

#### (a) Neural ODE의 핵심 정의

은닉 상태의 도함수를 뉴럴 네트워크 $f$로 매개변수화한다:

$$\frac{d\mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t, \theta)$$

여기서:
- $\mathbf{h}(t) \in \mathbb{R}^d$: 시간 $t$에서의 은닉 상태
- $f$: 뉴럴 네트워크 (파라미터 $\theta$)
- $\theta$: 학습 가능한 가중치

출력은 블랙박스 미분방정식 솔버를 통해 계산된다:

$$\mathbf{h}(t_1) = \mathbf{h}(t_0) + \int_{t_0}^{t_1} f(\mathbf{h}(t), t, \theta) \, dt = \text{ODESolve}(\mathbf{h}(t_0), f, t_0, t_1, \theta)$$

#### (b) Adjoint 감도 분석 방법 (역전파)

손실 함수 $L(\mathbf{h}(t_1))$에 대해, **adjoint state** $\mathbf{a}(t)$를 정의한다:

$$\mathbf{a}(t) = -\frac{\partial L}{\partial \mathbf{h}(t)}$$

$\frac{\partial L}{\partial \mathbf{h}(t_0)}$ (역전파에 필요한 기울기)는 **증강 ODE(augmented ODE)**를 시간 역방향으로 풀어 계산할 수 있다.

Adjoint 역학:

$$\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)^T \frac{\partial f(\mathbf{h}(t), t, \theta)}{\partial \mathbf{h}}$$

파라미터에 대한 기울기:

$$\frac{dL}{d\theta} = -\int_{t_1}^{t_0} \mathbf{a}(t)^T \frac{\partial f(\mathbf{h}(t), t, \theta)}{\partial \theta} \, dt$$

이 방법의 메모리 복잡도는 **$O(1)$** (상태 차원에 대해)로, 기존 역전파의 $O(L)$와 대비된다.

#### (c) 연속 정규화 흐름 (Continuous Normalizing Flows, CNF)

기존 Normalizing Flow에서 로그 확률의 변화는 이산적 야코비안 행렬식을 요구하지만, Neural ODE 프레임워크에서는 **instantaneous change of variables** 공식을 사용한다:

$$\frac{\partial \log p(\mathbf{h}(t))}{\partial t} = -\text{Tr}\left(\frac{\partial f}{\partial \mathbf{h}(t)}\right)$$

이를 통해 야코비안 행렬식 계산의 복잡도가 $O(d^3)$에서 $O(d)$로 감소한다.

### 2.3 모델 구조

```
입력 x ──→ [Encoder/Downsampling] ──→ h(t₀)
                                        │
                                        ▼
                              ┌─────────────────┐
                              │  ODE Block       │
                              │  dh/dt = f(h,t,θ)│
                              │  (ODE Solver)    │
                              └─────────────────┘
                                        │
                                        ▼
                                      h(t₁)
                                        │
                              ┌─────────────────┐
                              │  Output Layer    │
                              │  (FC / Softmax)  │
                              └─────────────────┘
                                        │
                                        ▼
                                     출력 y
```

논문에서 검증한 세 가지 응용:

| 응용 | 구조 | 데이터셋 |
|------|------|----------|
| 지도 학습 (분류) | ODE-Net (연속 깊이 ResNet) | MNIST |
| 생성 모델 | CNF (Continuous Normalizing Flow) | 2D density |
| 시계열 모델 | Latent ODE (VAE + Neural ODE) | Spiral data |

### 2.4 성능 향상

MNIST 분류 실험에서의 비교:

| 모델 | 파라미터 수 | 메모리 | Test Error |
|------|------------|--------|------------|
| ResNet | ~0.60M | $O(L)$ | ~0.41% |
| RK-Net | ~0.50M | $O(L)$ | ~0.47% |
| **ODE-Net** | **~0.22M** | **$O(1)$** | **~0.42%** |

연속 깊이 모델은 상수 메모리 비용을 가지며, 각 입력에 맞게 평가 전략을 적응시키고, 수치적 정밀도를 속도와 명시적으로 교환할 수 있다.

### 2.5 한계점

Neural ODE의 중요한 과제는 그래디언트 역전파 시 과도한 메모리 비용이다. Chen et al.이 제안한 역방향 ODE 풀이 방법은 (i) ReLU/비-ReLU 활성화 함수 및 일반 합성곱 연산자에 대해 수치적으로 불안정할 수 있으며, (ii) 작은 시간 스텝에서 불일치 기울기로 인해 훈련 발산을 유발할 수 있다.

Neural ODE는 입력 공간의 위상(topology)을 보존하는 표현을 학습하며, 이는 Neural ODE가 표현할 수 없는 함수가 존재함을 의미한다. 이러한 한계를 해결하기 위해 Augmented Neural ODE가 도입되었으며, 이는 더 표현력이 높고, 경험적으로 더 안정적이며, 더 잘 일반화되고, 계산 비용이 낮다.

또한 근본적인 문제로, ODE의 해는 초기 조건에 의해 결정되며, 이후 관측값을 기반으로 궤적을 조정하는 메커니즘이 없다.

추가적인 한계점:
- **학습 속도**: Adjoint 방법을 통한 훈련은 ODE를 수치적으로 풀어야 하므로 이산 모델에 비해 느리다.
- **Stiff ODE**: 경직(stiff) ODE 시스템은 과학/공학 분야에서 광범위하지만, 표준 Neural ODE 접근법은 이를 학습하는 데 어려움을 겪으며, 이것이 Neural ODE의 광범위한 채택에 대한 주요 장벽이다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 일반화 한계(Generalization Bound)

Marion (2023)은 연속 시간 파라미터를 갖는 ODE의 넓은 계열에 대해 Lipschitz 기반 일반화 한계(generalization bound)를 도출하였다. Neural ODE와 심층 잔차 네트워크 간의 유사성을 활용하여, 연속적인 가중치 행렬 간의 차이 크기가 일반화 성능에 영향을 미침을 수치적으로 보여주었다.

일반화 오차의 상한:

$$\mathcal{E}_{\text{gen}} \leq \widetilde{O}\left(\frac{1}{\sqrt{n}} \exp\left(\int_0^T \text{Lip}(f(\cdot, t, \theta)) \, dt\right)\right)$$

여기서 $\text{Lip}(f)$는 $f$의 Lipschitz 상수, $n$은 샘플 수, $T$는 적분 시간이다.

### 3.2 일반화 향상 전략

#### (a) 물리적 사전지식과 대칭 정규화

물리적 사전지식(priors)과 대칭 정규화는 특히 과학 및 공학 분야에서 out-of-sample 성능에 필수적이다.

모듈형 힘 분해(modular force decomposition)는 구조적 사전지식(에너지 보존, 대칭, 소산 한계)을 강제하여 해석 가능성과 장기 안정성을 향상시킨다. Lie 대칭 분석을 통해 식별된 불변량(invariants)은 손실에 패널티로 부과되어, 학습된 ODE가 핵심 보존 관계를 준수하도록 보장하며, 이는 수치적 안정성과 일반화를 향상시킨다.

#### (b) 피드백 뉴럴 네트워크 (Feedback Neural Networks)

일반화 문제는 Neural ODE의 연속 시간 예측 작업 응용을 크게 제한한다. 이를 해결하기 위해 이전 과제의 정확도를 훼손하지 않으면서 일반화를 개선하는 새로운 네트워크 아키텍처가 제안되었다. 피드백 뉴럴 네트워크는 학습된 잠재 역학을 유연하게 보정하여 일반화를 향상시키는 2자유도(two-DOF) 아키텍처이다.

피드백 Neural ODE의 수학적 구조:

$$\frac{d\mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t, \theta) + K(t)(\hat{\mathbf{h}}(t) - \mathbf{h}(t))$$

여기서 $K(t)$는 피드백 이득(gain), $\hat{\mathbf{h}}(t)$는 관측된/목표 상태이다.

피드백 뉴럴 네트워크는 불규칙한 물체의 궤적 예측에서 모델 기반 및 학습 기반 방법 모두를 유의미하게 능가하였으며, 0.5초 후 물체 위치를 정확히 예측하는 뛰어난 실시간 적응 능력을 보여주었다.

#### (c) Augmented Neural ODE

Augmented Neural ODE는 Neural ODE보다 더 표현력이 높은 모델이며, 경험적으로 더 안정적이고, 더 잘 일반화되며, 계산 비용이 낮다.

$$\frac{d}{dt}\begin{bmatrix} \mathbf{h}(t) \\ \mathbf{a}(t) \end{bmatrix} = f\left(\begin{bmatrix} \mathbf{h}(t) \\ \mathbf{a}(t) \end{bmatrix}, t, \theta\right), \quad \mathbf{a}(t_0) = \mathbf{0}$$

여기서 $\mathbf{a}(t) \in \mathbb{R}^p$는 증강 차원(augmented dimensions)이다.

#### (d) 연속 의존성 기반 PINN (cd-PINN)

ODE 해의 초기값 및 파라미터에 대한 연속 의존성 정보를 추가 통합하여 PINN을 비자명하게 확장하였다. cd-PINN은 뉴럴 연산자와 Meta-PINN의 장점을 통합하며, 소수의 레이블된 데이터만으로 새로운 초기값과 파라미터에 대해 미세 조정 없이 ODE를 풀 수 있다. 미학습 조건에서의 cd-PINN 정확도는 일반적으로 PINN보다 1-3 자릿수 높다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

Neural Differential Equations(NDE)는 뉴럴 네트워크를 활용한 연속 시간 모델링의 패러다임 전환을 가져왔다. Neural ODE에 관한 핵심 연구(Chen et al., 2018)는 연속 시간 은닉 상태 진화를 도입하여, Neural Controlled Differential Equations(NCDE)(Kidger et al., 2020), Neural Stochastic Differential Equations(NSDE) 등 수많은 후속 발전을 촉발시켰다.

ODE는 단순한 물리 시스템부터 유체 흐름, 화학 반응, 양자 진동기까지 공학 과학 전반에 편재한다. Neural ODE는 뉴럴 네트워크로 매개변수화된 ODE로서, 단순 분류 작업 이후 희소 데이터로부터 다물리 시스템의 비선형 역학 학습, 비선형 시스템의 최적 제어, 의료 영상, 불규칙 시계열의 실시간 처리 등으로 빠르게 응용이 확장되었다.

### 4.2 향후 연구 시 고려할 점

| 고려 사항 | 설명 |
|-----------|------|
| **솔버-아키텍처 결합** | 응용 역학과 수치 적분 방법 간의 매칭이 안정성과 성능에 결정적이다. |
| **Stiff 시스템** | 단일 스텝 암묵적(implicit) 방법에 기반한 접근법이 Neural ODE의 경직성 처리를 가능하게 하며, 이는 더 넓은 범위의 과학 문제에의 활용을 위한 핵심이다. |
| **불확실성 정량화** | Neural SDE는 브라운 운동 항을 통해 불확실성과 노이즈를 모델링하여, 실세계 현상에서 무작위성이 중요한 역할을 하는 보다 강건한 모델링을 가능하게 한다. |
| **관측값 통합** | ODE 해가 초기 조건에 의해 결정되어 후속 관측값 기반 궤적 조정이 불가능한 문제를 Controlled Differential Equation의 수학을 통해 해결할 수 있으며, Neural CDE 모델은 부분 관측, 불규칙 샘플링된 다변량 시계열에 직접 적용 가능하다. |
| **스케일링** | NP-ODE는 바닐라 NP 디코더 대비 파라미터 수를 4배 감소시키며, adjoint 및 체크포인팅 방법은 거의 일정한 메모리로 깊은 ODE 스택 훈련을 가능하게 한다. |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 후속 연구 계보

```
Neural ODE (2018, Chen et al.)
    │
    ├── Augmented Neural ODE (2019, Dupont et al.)
    │       └── SONODE (2020, Norcliffe et al.)
    │
    ├── FFJORD / CNF (2019, Grathwohl et al.)
    │       └── Score-based SDE (2021, Song et al.)
    │
    ├── Neural CDE (2020, Kidger et al.)
    │       ├── Neural Rough DE (2021, Morrill et al.)
    │       └── ANCDE (2024, Jhin et al.)
    │
    ├── Neural SDE (2020-2024)
    │       ├── Latent SDE (2020, Li et al.)
    │       └── Stable Neural SDE (2024, Oh et al.)
    │
    ├── 일반화 이론
    │       ├── Generalization Bounds (2023, Marion)
    │       └── Feedback Neural ODE (2024, ICLR)
    │
    └── 훈련 효율성
            ├── ANODE (2019, Gholami et al.)
            ├── ACA (2021, Zhuang et al.)
            └── Gauß-Legendre (2023, Norcliffe et al.)
```

### 5.2 상세 비교

| 모델 (연도) | 핵심 수식 | Neural ODE 대비 개선점 | 한계 |
|------------|----------|----------------------|------|
| **Neural CDE** (2020) | $d\mathbf{h} = f(\mathbf{h}) \, dX(t)$ | 부분 관측, 불규칙 시계열에 직접 적용 가능하며 메모리 효율적 adjoint 역전파를 관측 간에도 사용 가능 | 경로 보간 방법 선택 민감 |
| **Neural SDE** (2020+) | $d\mathbf{h} = f \, dt + g \, dW_t$ | 그래디언트 계산, 잠재 공간의 변분 추론, 불확실성 정량화에 초점을 맞춘 Neural ODE의 확장 | 특수화된 솔버 필요 |
| **SONODE** (2020) | $\ddot{\mathbf{x}} = f(\mathbf{x}, \dot{\mathbf{x}}, t, \theta)$ | 2차 역학을 기술하고 NODE의 교차 궤적 문제를 해결하며, ANODE와 달리 고유 해를 제공 | ANODE의 특수한 경우로 볼 수 있어 제약적 |
| **Feedback Neural ODE** (2024) | $\dot{\mathbf{h}} = f(\mathbf{h}, \theta) + K(\hat{\mathbf{h}}-\mathbf{h})$ | 실시간 피드백을 통합하여 연속 시간 예측에서의 일반화를 크게 향상, 기존 Neural ODE의 보이지 않는 시나리오에 대한 일반화 한계를 해결 | 피드백 신호 설계에 의존 |
| **GTSONO** (2024) | Min-Max DDP 최적화 | 자연 훈련 하에 공격된 이미지에 대해 더 정확하고 확신있는 예측을 생성, 기존 적대적 훈련 방법에도 적응하여 강건성 향상 | 대규모 아키텍처로의 확장 필요 |
| **Generalization Bounds** (2023) | Lipschitz 기반 경계 | Neural ODE를 위한 일반화 한계를 Lipschitz 인수로 도출, 심층 잔차 네트워크에도 적용 가능 | 이론적 분석 위주 |
| **Gauß-Legendre Adj.** (2023) | 구적법 기반 adjoint | ODE 기반 방법보다 빠르게 적분을 풀면서 메모리 효율적인 Gauß-Legendre 구적법으로 adjoint 방법을 가속화 | 특정 문제 유형에 한정 |
| **DualDynamics** (2025) | 명시적 + 암묵적 | 명시적 시간 진화를 위한 Neural ODE와 잠재 상태 전이를 위한 학습 가능한 암묵적 업데이트를 통합하여 해석 가능성과 강건성 간의 균형 달성 | 최신 연구로 검증 범위 한정 |

### 5.3 Score-based SDE와의 연결 (생성 모델 관점)

Neural ODE의 CNF 프레임워크는 현대 확산 모델(Diffusion Models)의 이론적 기반이 되었다. Song et al. (2021)의 **Score-based SDE** 프레임워크는 다음과 같다:

$$d\mathbf{x} = f(\mathbf{x}, t) \, dt + g(t) \, d\mathbf{w}$$

역방향(생성):

$$d\mathbf{x} = \left[f(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt + g(t) \, d\bar{\mathbf{w}}$$

이 프레임워크에서 "데이터에서 노이즈를 만드는 것은 쉽고, 노이즈에서 데이터를 만드는 것이 생성 모델링"이다. SDE를 통해 복잡한 데이터 분포를 알려진 사전 분포로 부드럽게 변환하고, 대응하는 역방향 SDE가 사전 분포를 다시 데이터 분포로 변환한다.

---

## 6. 종합 결론

Neural ODE는 딥러닝과 미분방정식의 접점에서 근본적인 패러다임을 제시한 연구이다. Neural ODE 모듈은 이제 과학적 기계 학습, 연산자 학습, 불확실성 정량화, 연속 시간 모델링의 핵심 기본 요소(primitive)를 구성한다.

일반화 성능 향상을 위해서는:
1. **물리적 사전지식 통합**: 보존 법칙, 대칭성, 에너지 구조 활용
2. **증강된 상태 공간**: Augmented/Second-order ODE를 통한 표현력 확장
3. **피드백 메커니즘**: 실시간 관측값을 통한 적응적 보정
4. **확률적 확장**: Neural SDE를 통한 불확실성 모델링

향후에는 이산 아키텍처와의 하이브리드 통합, 확장 가능한 물리 기반 대리 모델(surrogate), 그리고 실세계 과학 컴퓨팅에서의 강건하고 해석 가능한 배포에서의 발전이 예상된다.

---

## 참고문헌 및 출처

1. **Chen, R.T.Q., Rubanova, Y., Bettencourt, J., Duvenaud, D.** (2018). *Neural Ordinary Differential Equations*. NeurIPS 2018. arXiv:1806.07366
2. **Marion, P.** (2023). *Generalization bounds for neural ordinary differential equations and deep residual networks*. arXiv:2305.06648
3. **Dupont, E., Doucet, A., Teh, Y.W.** (2019). *Augmented Neural ODEs*. arXiv:1904.01681
4. **Gholami, A., Keutzer, K., Biros, G.** (2019). *ANODE: Unconditionally Accurate Memory-Efficient Gradients for Neural ODEs*. IJCAI-19
5. **Kidger, P., Morrill, J., Foster, J., Lyons, T.** (2020). *Neural Controlled Differential Equations for Irregular Time Series*. NeurIPS 2020. arXiv:2005.08926
6. **Norcliffe, A., et al.** (2020). *On Second Order Behaviour in Augmented Neural ODEs*. NeurIPS 2020
7. **Grathwohl, W., Chen, R.T.Q., Bettencourt, J., et al.** (2019). *FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models*. ICLR 2019
8. **Song, Y., et al.** (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021
9. **Li, X., et al.** (2020). *Scalable Gradients for Stochastic Differential Equations*. AISTATS 2020
10. **Oh, S., et al.** (2024). *Stable Neural Stochastic Differential Equations*. ICLR 2024
11. **Feedback Favors the Generalization of Neural ODEs** (2024). ICLR 2024. arXiv:2410.10253
12. **GTSONO: A Robust Differential Neural ODE Optimizer** (2024). ICLR 2024
13. **Norcliffe, A., et al.** (2023). *Faster Training of Neural ODEs Using Gauß-Legendre Quadrature*. TMLR 2023
14. **Comprehensive Review of Neural Differential Equations for Time Series Analysis** (2025). IJCAI-25. arXiv:2502.09885
15. **Geometric Neural ODEs: From Manifolds to Lie Groups** (2025). PMC
16. **A guide to neural ordinary differential equations** (2025). ScienceDirect
17. **Improving generalization ability of deep-learning-based ODE solvers using continuous dependence** (2025). npj AI
18. **Jhin, S.Y., et al.** (2024). *Attentive Neural Controlled Differential Equations*. KAIS
19. **Oh, S., et al.** (2025). *DualDynamics*. 불규칙 시계열을 위한 명시적-암묵적 프레임워크

# Neural Ordinary Differential Equations

## 1. 핵심 주장 및 주요 기여

**Neural Ordinary Differential Equations (Neural ODE)**는 2018년 Chen et al.이 제시한 혁신적 프레임워크로, NeurIPS Best Paper에 선정되었습니다. 이 논문의 핵심 통찰은 잔차 신경망(ResNet)이 상미분방정식(ODE)의 오일러 이산화로 해석될 수 있다는 관찰에서 출발합니다. 이를 극한까지 발전시켜, 명시적 레이어 정의 대신 신경망으로 매개변수화된 ODE를 직접 정의하고, 검은상자 ODE solver로 네트워크의 출력을 계산하는 방식을 제안합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

**주요 기여는 다음과 같습니다:**

1. **메모리 효율성**: 기존 깊은 신경망은 모든 중간 활성화를 저장해야 하므로 메모리 비용이 깊이에 선형적으로 증가합니다. Neural ODE는 수반 민감도 방법(adjoint sensitivity method)을 활용하여 **상수 메모리 비용**으로 역전파를 수행하며, forward pass의 중간값을 저장하지 않습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

2. **적응형 계산**: ODE solver의 적응형 스텝 크기 조절 메커니즘을 활용하여, 각 입력의 복잡도에 따라 자동으로 평가 전략을 조정합니다. 이는 모형 평가 비용을 문제의 복잡도에 맞춥니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

3. **정밀도-속도 트레이드오프**: 허용 오차(tolerance) 조정으로 훈련 및 추론 단계에서 정밀도와 속도를 명시적으로 교환할 수 있습니다.

4. **Continuous Normalizing Flows**: 기존 정규화 흐름의 변수 변환 공식에서 log-determinant 계산으로 인한 입방 시간 복잡도를 **선형 시간**으로 단축하는 "순간변수변환 정리"를 도입합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

***

## 2. 해결하고자 하는 문제 및 제안 방법

### 2.1 문제 정의

기존 깊은 신경망 아키텍처는 다음의 근본적 제약을 가집니다:

- **메모리 병목**: 역전파 시 모든 중간 활성화 저장으로 깊이에 선형적 메모리 증가
- **고정 계산**: 모든 입력에 동일한 계산량 소요 (불필요한 낭비)
- **표현력 제약**: 이산 레이어로 연속 변환 근사 시 비효율성

### 2.2 제안 방법론

**기본 ODE 공식:**

$$\frac{dh(t)}{dt} = f(h(t), t, \theta)$$

여기서 $h(t) \in \mathbb{R}^D$는 숨겨진 상태, $f$는 신경망으로 매개변수화된 벡터장, $\theta$는 학습 가능한 매개변수입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

**손실 함수의 기울기 계산:**

손실 $L$에 대해, 수반 상태 $a(t) = \frac{\partial L}{\partial z(t)}$는 다음 역방향 ODE를 만족합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

$$\frac{da(t)}{dt} = -a(t)^T \frac{\partial f(z(t), t, \theta)}{\partial z}$$

이 ODE를 시간 역방향으로 풀어 기울기를 계산합니다. 매개변수에 대한 기울기는:

$$\frac{\partial L}{\partial \theta} = -\int_{t_1}^{t_0} a(t)^T \frac{\partial f(z(t), t, \theta)}{\partial \theta} dt$$

**Algorithm 1: 역모드 미분**
```
입력: θ, t_0, t_1, z(t_1), ∂L/∂z(t_1)
s_0 = [z(t_1), ∂L/∂z(t_1), 0]
증강 동역학: f_aug = [f(z,t,θ), -a^T ∂f/∂z, -a^T ∂f/∂θ]
증강 ODE 해결: ODESolve(f_aug, s_0, t_1, t_0)
반환: ∂L/∂z(t_0), ∂L/∂θ, ∂L/∂t_0
```

이 접근법은 ODE solver를 **검은상자**로 취급하므로, 임의의 solver에 적용 가능합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

### 2.3 Continuous Normalizing Flows

정규화 흐름에서 변수 변환 시 로그-판별식 계산은 일반적으로 $\mathcal{O}(d^3)$ 비용을 요구합니다. Chen et al.은 다음 **순간변수변환 정리**를 증명합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

**정리 1**: $z(t)$가 $\frac{dz}{dt} = f(z(t), t)$를 따르고, $f$가 $z$에서 균등 Lipschitz 연속이며 $t$에서 연속이면, 로그 확률의 변화는:

$$\frac{d\log p(z(t))}{dt} = \text{tr}\left(\frac{\partial f}{\partial z(t)}\right)$$

**증명 스케치**: 변수 변환 공식의 이산 형태에서 시작하여, $\Delta t \to 0$ 극한을 취하면:

$$\log p(z(t+\Delta t)) - \log p(z(t)) = \log\left|\det\left(\frac{\partial z(t+\Delta t)}{\partial z(t)}\right)\right|$$

$$= \log\left|\det\left(I + \Delta t \frac{\partial f}{\partial z} + O(\Delta t^2)\right)\right|$$

야코비 행렬식의 미분 성질(Jacobi's formula)과 행렬식의 미분 공식을 이용하면:

$$\log\left|\det(I + \varepsilon J)\right| \approx \varepsilon \text{tr}(J) + O(\varepsilon^2)$$

따라서:

$$\frac{d\log p(z(t))}{dt} = \text{tr}\left(\frac{\partial f}{\partial z(t)}\right)$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

이로 인해 Jacobian 결정식 계산이 **trace** 연산으로 단순화되며, $M$개의 숨겨진 유닛을 가진 흐름의 계산 비용은 $\mathcal{O}(M^3)$에서 $\mathcal{O}(M)$으로 감소합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

***

## 3. 모델 구조

### 3.1 Supervised Learning (ODE-Net)

분류 작업에서 기본 구조:

1. 입력층 $h_0 = x$ (크기 $D$)
2. 다운샘플링 블록들
3. **ODE 블록**: $h_T = \text{ODESolve}(h_0, f, t_0, t_1)$
4. 분류 헤드

실험 결과는 Table 1 및 Figure 3에서 확인됩니다. MNIST에서:

| 모델 | 테스트 오차 | 매개변수 | 메모리 | 시간 |
|------|-----------|--------|--------|------|
| 1-Layer MLP | 1.60% | - | - | - |
| ResNet | 0.41% | 0.60M | OL | OL |
| RK-Net | 0.47% | 0.22M | OL | OL |
| **ODE-Net** | **0.42%** | **0.22M** | **O(1)** | **OL** | [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

메모리가 상수 비용으로 유지되며, ODE solver의 함수 평가 횟수(NFE)는 훈련 중 증가하여 모델이 점진적으로 더 복잡한 동역학을 학습함을 시사합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

### 3.2 Continuous Normalizing Flows (CNF)

생성 모델로서의 구조:

$$z_0 \sim p_0(z) = \mathcal{N}(0, I)$$

$$z_t = \text{ODESolve}(z_0, f, t=0, t=1; \theta)$$

$$\log p_1(z_1) = \log p_0(z_0) - \int_0^1 \text{tr}\left(\frac{\partial f}{\partial z_t}\right) dt$$

**게이팅 메커니즘**: 시간 의존 동역학을 더욱 표현력 있게 만들기 위해:

$$\frac{dz}{dt} = \sum_{n=1}^M \lambda_n(t) f_n(z)$$

여기서 $\lambda_n(t) \in $은 신경망으로 학습됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

### 3.3 잠재 변수 시계열 모델 (Latent ODE)

불규칙하게 샘플링된 시계열 $\{(t_i, x_{t_i})\}$에 대해:

1. **인코더**: RNN이 $q_\phi(z_0|x_{t_1}, \ldots, x_{t_N})$을 학습
2. **ODE 동역학**: $z_t = \text{ODESolve}(z_0, f, t_0, \ldots, t_M)$
3. **디코더**: $p(x_{t_i}|z_{t_i})$ 정의

**VAE 학습 목표**:

$$\mathcal{L} = \sum_{i=1}^N \log p(x_{t_i}|z_{t_i}) + \text{KL}(q_\phi(z_0|x_{t_i})||p(z_0))$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

***

## 4. 성능 향상 및 한계

### 4.1 성능 향상

**메모리 효율성**:
- Forward pass에서 중간 활성화 미저장
- Backward pass에서 동시 ODE 풀이로 메모리 상수 유지
- 깊이에 무관한 메모리 복잡도

**계산 효율성**:
- Backward pass의 함수 평가 횟수는 forward pass의 약 50% [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)
- 이는 adjoint sensitivity method가 ODE solver의 내부 연산을 통한 직접 역전파보다 우월함을 의시합니다.

**정밀도 제어**:
- Figure 3의 결과: 허용 오차 조정으로 NFE와 계산 시간 간 3배 트레이드오프 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

**시계열 예측**:
- Table 2에서 Latent ODE는 RNN보다 현저히 낮은 RMSE 달성
- 불규칙 샘플링에서 30개 포인트로 30→100 외삽 시 RMSE: 0.1642 vs 0.3937 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

### 4.2 한계

**Topology 보존 문제** (Dupont et al., 2019):
- Neural ODE는 입력 공간의 위상(topology)을 보존하는 표현을 학습합니다.
- 이는 교차하는 궤적을 표현하는 함수가 존재하지 않음을 의미합니다. [arxiv](https://arxiv.org/abs/1904.01681)

**Expressiveness 제약**:
- 동일한 손실을 달성하는 여러 흐름이 존재 가능
- 더 복잡한 함수는 더 많은 평가 단계 필요 → 계산 비용 증가

**Minibatching의 어려움**:
- 배치 요소를 결합하면 ODE 차원이 $D \times K$로 증가
- 모든 배치 요소에서 오차 제어 시 함수 평가 횟수 최대 K배 증가 가능
- 실제로는 현저한 증가 관찰되지 않지만 여전히 고려 사항 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

**허용 오차 설정**:
- Forward/backward pass 모두에서 수동으로 허용 오차 설정 필요
- 부적절한 설정은 훈련 불안정 유발 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

**수치적 불안정성**:
- Forward 궤적을 역시간에 재구성 시 추가 수치 오차 가능
- Checkpointing으로 완화 가능하지만 메모리 증가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5988af8-82e0-49ff-ade8-eea72e699a09/1806.07366v5.pdf)

***

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 Augmented Neural ODE (2019)

**핵심 개선**: Neural ODE의 표현력 부족을 **차원 증강**으로 해결합니다. [arxiv](https://arxiv.org/abs/1904.01681)

기본 NODE는 다음 제약을 가집니다:
- 입력 공간의 위상 보존으로 인해 표현 불가능한 함수 존재

**ANODE 해법**:

$$\frac{da(t)}{dt} = f(a(t), t; \theta), \quad a(t) = [z(t), z_{\text{aug}}(t)]$$

여기서 $z_{\text{aug}}(t)$는 보조 변수입니다. [arxiv](https://arxiv.org/abs/1904.01681)

**효과**:
- 표현력 증가로 더 빠른 수렴
- 더 낮은 일반화 오차
- 더 낮은 계산 비용 (동일 성능에서)
- 안정적인 훈련 동역학 [dsba.snu.ac](https://dsba.snu.ac.kr/?kboard_content_redirect=1781)

### 5.2 Feedback Neural Networks (2024)

**피드백 기반 동역학 수정**: [arxiv](https://arxiv.org/html/2410.10253v1)

$$\frac{dz(t)}{dt} = f(z(t), t; \theta) + K(y_t - \hat{y}_t)$$

여기서 $K$는 피드백 루프의 게인, $y_t$는 실제 관찰, $\hat{y}_t$는 예측입니다.

**일반화 개선**:
- 실시간 피드백 메커니즘으로 학습된 동역학 보정
- 미지 시나리오에 더 robust한 성능
- 기존 작업의 정확도 손실 없음 [arxiv](https://arxiv.org/html/2410.10253v1)

### 5.3 Polynomial Neural ODE (2022)

**해석 가능성과 외삽(Extrapolation)**: [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10076068/)

다항 신경망을 ODE 내부에 도입하여:
- 훈련 영역 밖의 예측 성능 향상
- 기호적 회귀(symbolic regression) 가능
- 물리적 동역학에 더 적합

### 5.4 물리 정보 기반 Neural ODE

**제약 통합으로 일반화 개선**:
- PHOENIX 모델: Hill-Langmuir kinetics와 생물학적 선행 지식 통합 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10055646/)
- 생물 시스템에서의 해석성과 확장성 향상

### 5.5 해석적으로 가역인 ODE Solver (2025)

**Efficient, Accurate and Stable Gradients for Neural ODEs**: [arxiv](http://arxiv.org/pdf/2410.11648.pdf)

역전파의 수치적 정확도와 메모리 효율성을 동시에 확보:
- 해석적으로 가역인 solver는 역전파 오차 제거
- DtO(Discretize-Then-Optimize) 방식으로 더 정확한 기울기 [arxiv](http://arxiv.org/pdf/2410.11648.pdf)

***

## 6. 2020년 이후 관련 최신 연구 비교 분석

### 6.1 역전파 효율성 개선

| 논문 | 연도 | 기여 | 성능 개선 |
|------|------|------|---------|
| Adaptive Checkpoint Adjoint | 2020 | 역전파 정확도 향상 | 불안정성 감소 |
| "Hey, that's not an ODE" | 2020 | Seminorm 기반 가속 [arxiv](https://arxiv.org/abs/2009.09457v1) | 40-62% NFE 감소 |
| Efficient Gradients for Neural ODEs | 2025 | 해석적 가역성 [arxiv](http://arxiv.org/pdf/2410.11648.pdf) | 정확도↑, 메모리↓ |

Seminorm 기반 접근은 adjoint 방정식 풀이 시 매개변수 채널의 오차 요구를 낮춤으로써 ODE solver의 불필요한 단계 거부를 줄입니다. [arxiv](https://arxiv.org/abs/2009.09457v1)

### 6.2 표현력 및 안정성

| 논문 | 연도 | 초점 | 결과 |
|------|------|------|------|
| Augmented Neural ODE [arxiv](https://arxiv.org/abs/1904.01681) | 2019 | 차원 증강 | 1.75%, 1.16%, 0.94% 개선 |
| Polynomial Neural ODE [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10076068/) | 2022 | 해석성 | 외삽 성능↑ |
| Stiff ODE 처리 [arxiv](https://arxiv.org/abs/2408.06073) | 2024 | 경직 문제 | 안정성↑ |
| ANODEV2 [arxiv](https://arxiv.org/pdf/1906.04596.pdf) | - | 매개변수 진화 | 일반화 성능↑ |

### 6.3 시계열 및 생성 모델

| 논문 | 연도 | 응용 | 특징 |
|------|------|------|------|
| BrainODE [arxiv](http://arxiv.org/pdf/2405.00077.pdf) | 2024 | 뇌신호 | 불규칙 샘플링, 임상 예측 |
| OT-Flow [arxiv](http://arxiv.org/pdf/2006.00104.pdf) | 2020 | 정규화 흐름 | 최적수송 정규화, 8x 가속 |
| Flow Matching | 2024 | 생성 모델 | 확률 경로 학습 |

### 6.4 물리 제약 통합

**최신 트렌드**: Physics-Informed Neural ODE 급증

- PHOENIX (2023): 생물 동역학에 생물학적 제약 통합 [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC10055646/)
- PINN 하이브리드 (2024-2025): Physics-Informed Normalizing Flows
- Neural-ODE Hybrid Block Method (2025): 고차 ODE와 수치 안정성 [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/40069216/)

### 6.5 실제 응용의 확장

**의료**: Neural ODE 기반 시계열 예측으로 임상 결과 예측

**지구과학**: 해수면 온도 계절 예측에 적용

**자율주행** (2025): 불확실성 모델링으로 30% 안전성 향상 [jaenung](https://www.jaenung.net/tree/15906)

***

## 7. 향후 연구에 미치는 영향

### 7.1 이론적 발전

**일반화 경계**: 연속시간 매개변수를 가진 ODE의 일반화 경계 도출 [semanticscholar](https://www.semanticscholar.org/paper/Neural-Ordinary-Differential-Equations-Chen-Rubanova/449310e3538b08b43227d660227dfd2875c3c3c1)
- Lipschitz 기반 논증으로 표본 복잡도 분석
- ResNet과의 이론적 연결 강화

**암묵적 정규화**: Gradient flow 하에서 ResNet의 Neural ODE로의 수렴 [semanticscholar](https://www.semanticscholar.org/paper/Neural-Ordinary-Differential-Equations-Chen-Rubanova/449310e3538b08b43227d660227dfd2875c3c3c1)
- 선형 폭 과파라미터화에서의 수렴 증명

### 7.2 기술적 개선 방향

**1. 해석적 가역성의 활용**:
- DtO(Discretize-Then-Optimize) 방식의 확대 적용
- Algebraically reversible solvers의 설계 및 분석

**2. 고차 ODE 확장**:
- Neural-ODE Hybrid Block Method 활용으로 안정성 개선
- 경직(stiff) 문제 해결

**3. 적응형 허용 오차**:
- 동적 허용 오차 설정으로 훈련 안정성 향상
- 문제별 자동 튜닝 메커니즘

### 7.3 응용 분야 확장

**다중 물리(Multiphysics) 문제**:
- Physics-Informed 접근으로 여러 동역학 동시 모델링
- 기후, 재료과학, 생물 시스템에 적용

**불확실성 정량화**:
- Neural SDE를 통한 확률적 동역학 모델링
- Bayesian Neural ODE의 발전

**해석 가능성**:
- Symbolic Neural ODE (AAAI 2025): 의미 있는 기호식 도출 [liner](https://liner.com/ko/review/symbolic-neural-ordinary-differential-equations)
- Vector field 시각화를 통한 직관적 이해

### 7.4 연산 최적화

**메모리-계산 트레이드오프**:
- Checkpointing 전략의 정교한 설계
- Reversible solver와 비가역 solver의 하이브리드 활용

**분산 학습**:
- Large-scale 응용을 위한 병렬화 기법
- ODE 병렬 풀이(PARAREAL 등) 적분

***

## 8. 향후 연구 시 고려할 점

### 8.1 이론적 고려사항

1. **일반화 한계 분석**: Neural ODE의 표현 클래스가 어떤 함수족을 포함하는지 특성화
   - 위상 불변성으로 인한 근본적 제약
   - Augmentation 전략의 최적성

2. **수치해석적 안정성**: ODE solver 선택이 일반화에 미치는 영향
   - Implicit vs Explicit solver의 수렴 특성
   - 적응형 스텝 크기 조절 메커니즘의 정규화 효과

### 8.2 실무적 고려사항

1. **하이퍼파라미터 튜닝**:
   - 허용 오차 설정의 자동화
   - 초기 조건 민감도 해소

2. **계산 비용**:
   - 실시간 응용에서의 추론 속도 보장
   - 메모리-계산 트레이드오프 최적화

3. **데이터 특성에 맞는 모델 선택**:
   - 규칙 샘플링 vs 불규칙 샘플링
   - 결정론적 vs 확률적 동역학

### 8.3 검증 및 해석성

1. **외삽 성능**: 훈련 영역 밖 예측 능력 검증
   - 물리 제약 기반 정규화의 필요성
   - Polynomial ODE 등 해석 가능 변형의 활용

2. **인과성 분석**: 개입(intervention)에 대한 모델 반응
   - IMODE (2020) 등 개입 모델링 기법
   - 인과 추론과의 통합

3. **불확실성 정량화**: Bayesian 접근 및 앙상블 방법

***

## 결론

Neural ODE는 심층 학습에서 **연속-깊이 모델링의 패러다임 전환**을 제시합니다. 원본 논문의 메모리 효율성, 적응형 계산, Continuous Normalizing Flow는 기본적이면서도 심원한 기여입니다.

2020년 이후의 발전은 다음을 중심으로 전개되었습니다:

1. **일반화 성능 개선**: Augmented Neural ODE, Feedback Networks, 물리 제약 통합
2. **수치적 안정성 강화**: Reversible solvers, Seminorm 기반 가속, 고차 방법
3. **응용 확장**: 의료, 지구과학, 자율주행 등 실제 시계열 예측
4. **해석성 강화**: Symbolic Neural ODE, Physics-Informed 접근

향후 연구는 **이론적 경계 강화**, **표현력 확장**, **실 규모 응용**에 초점을 맞춰야 합니다. 특히 물리 제약 통합과 불확실성 정량화는 과학 응용에서의 신뢰도를 높일 것으로 기대됩니다.

***

## 참고문헌

<span style="display:none">[^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82]</span>

<div align="center">⁂</div>

[^1_1]: 1806.07366v5.pdf

[^1_2]: https://arxiv.org/abs/1904.01681

[^1_3]: https://dsba.snu.ac.kr/?kboard_content_redirect=1781

[^1_4]: https://arxiv.org/html/2410.10253v1

[^1_5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10076068/

[^1_6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10055646/

[^1_7]: http://arxiv.org/pdf/2410.11648.pdf

[^1_8]: https://arxiv.org/abs/2009.09457v1

[^1_9]: https://arxiv.org/abs/2408.06073

[^1_10]: https://arxiv.org/pdf/1906.04596.pdf

[^1_11]: http://arxiv.org/pdf/2405.00077.pdf

[^1_12]: http://arxiv.org/pdf/2006.00104.pdf

[^1_13]: https://pubmed.ncbi.nlm.nih.gov/40069216/

[^1_14]: https://www.jaenung.net/tree/15906

[^1_15]: https://www.semanticscholar.org/paper/Neural-Ordinary-Differential-Equations-Chen-Rubanova/449310e3538b08b43227d660227dfd2875c3c3c1

[^1_16]: https://liner.com/ko/review/symbolic-neural-ordinary-differential-equations

[^1_17]: https://arxiv.org/pdf/2206.11120.pdf

[^1_18]: https://arxiv.org/html/2408.06073v1

[^1_19]: https://arxiv.org/html/2106.12430v2

[^1_20]: https://kimjy99.github.io/논문리뷰/neural-ode/

[^1_21]: https://www.youtube.com/watch?v=UegW1cIRee4

[^1_22]: https://e-opr.org/articles/xml/vR4W/

[^1_23]: https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10501091

[^1_24]: https://www.themoonlight.io/ko/review/high-order-expansion-of-neural-ordinary-differential-equations-flows

[^1_25]: https://liner.com/ko/review/feedback-favors-the-generalization-of-neural-odes

[^1_26]: https://komes2025.co.kr/default/img/hblock/content/poster/p019.pdf

[^1_27]: https://lilys.ai/ko/notes/452737

[^1_28]: https://blog.outta.ai/280

[^1_29]: https://www.youtube.com/watch?v=-tYqABNgFY8

[^1_30]: https://ppta.or.kr/webzine/2022_11/a1.html

[^1_31]: https://kimjy99.github.io/논문리뷰/instaflow/

[^1_32]: https://arxiv.org/html/2510.09685v1

[^1_33]: https://arxiv.org/html/2410.10174v1

[^1_34]: http://arxiv.org/abs/1806.07366

[^1_35]: https://arxiv.org/pdf/2503.03129.pdf

[^1_36]: http://www.arxiv.org/abs/2010.08304

[^1_37]: https://arxiv.org/pdf/2507.19036.pdf

[^1_38]: https://arxiv.org/pdf/2510.09685.pdf

[^1_39]: https://arxiv.org/html/2507.19036v2

[^1_40]: https://arxiv.org/html/2502.09885v1

[^1_41]: https://www.arxiv.org/pdf/2508.04799.pdf

[^1_42]: https://arxiv.org/abs/2410.01911

[^1_43]: https://epubs.siam.org/doi/10.1137/21M140078X

[^1_44]: https://arxiv.org/abs/2307.10711

[^1_45]: https://ieeexplore.ieee.org/document/10700017/

[^1_46]: https://arxiv.org/abs/2209.06886

[^1_47]: https://www.semanticscholar.org/paper/53c0b27ed37f7f96870c711e6803543f765498e9

[^1_48]: https://arc.aiaa.org/doi/10.2514/1.J064455

[^1_49]: https://saemobilus.sae.org/papers/drag-reduction-study-vehicle-shape-optimization-using-gradient-based-adjoint-method-2024-01-2528

[^1_50]: https://ieeexplore.ieee.org/document/11132551/

[^1_51]: https://www.ingentaconnect.com/content/10.3397/IN_2022_0321

[^1_52]: http://arxiv.org/abs/1912.07696

[^1_53]: https://arxiv.org/pdf/2009.09457v1.pdf

[^1_54]: http://arxiv.org/pdf/2109.00067.pdf

[^1_55]: https://arxiv.org/pdf/2402.15141.pdf

[^1_56]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8299461/

[^1_57]: http://arxiv.org/pdf/2410.01911.pdf

[^1_58]: http://arxiv.org/pdf/1606.04406v3.pdf

[^1_59]: https://epubs.siam.org/doi/pdf/10.1137/17M1144532

[^1_60]: https://www.reddit.com/r/MachineLearning/comments/v2phwb/d_adjoint_sensitivity_method_vs_reverse_mode/

[^1_61]: https://bayesian-bacteria.tistory.com/4

[^1_62]: https://www.youtube.com/watch?v=AxJX5eiUVTI

[^1_63]: https://velog.io/@whitecl1031/Continuous-Normalizing-Flows-wl9genml

[^1_64]: https://horizon.kias.re.kr/29873/

[^1_65]: https://velog.io/@guts4/Neural-Ordinary-Differential-Equations-논문리뷰

[^1_66]: https://chan4im.tistory.com/268

[^1_67]: https://junhan-ai.tistory.com/295

[^1_68]: https://blog.outta.ai/211

[^1_69]: https://data-newbie.tistory.com/1011

[^1_70]: https://www.themoonlight.io/ko/review/meshodenet-a-graph-informed-neural-ordinary-differential-equation-neural-network-for-simulating-mesh-based-physical-systems

[^1_71]: https://seastar105.tistory.com/176

[^1_72]: http://arxiv.org/pdf/2402.15141.pdf

[^1_73]: https://www.arxiv.org/pdf/2410.01911.pdf

[^1_74]: http://arxiv.org/pdf/2309.15139.pdf

[^1_75]: https://arxiv.org/pdf/2002.02798.pdf

[^1_76]: https://arxiv.org/abs/2404.00551

[^1_77]: https://arxiv.org/pdf/2505.02019.pdf

[^1_78]: http://arxiv.org/pdf/2009.09457.pdf

[^1_79]: https://arxiv.org/abs/2309.15139

[^1_80]: https://arxiv.org/html/2403.02224v1

[^1_81]: https://arxiv.org/abs/2309.15139v1

[^1_82]: https://arxiv.org/html/2507.19036v1


 OT-Flow: Fast and Accurate Continuous Normalizing Flows. (2020). [arxiv](http://arxiv.org/pdf/2006.00104.pdf)

 "Hey, that's not an ODE": Faster ODE Adjoints via Seminorms. (2020). [arxiv](https://arxiv.org/abs/2009.09457v1)
