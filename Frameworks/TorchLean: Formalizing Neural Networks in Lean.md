
# TorchLean: Formalizing Neural Networks in Lean 

> **논문 정보:**
> - **제목:** TorchLean: Formalizing Neural Networks in Lean
> - **저자:** Robert Joseph George, Jennifer Cruden, Xiangru Zhong, Huan Zhang, Anima Anandkumar
> - **arXiv ID:** arXiv:2602.22631 (2026년 2월 27일 공개)
> - **프로젝트 페이지:** [torchlean.org](http://torchlean.org) / [leandojo.org/torchlean.html](https://leandojo.org/torchlean.html)
> - **GitHub:** [lean-dojo/TorchLean](https://github.com/lean-dojo/TorchLean)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

신경망이 안전·임무 필수 파이프라인에 점점 더 많이 배포되고 있지만, 많은 검증 및 분석 결과가 모델을 정의하고 실행하는 프로그래밍 환경 밖에서 생성된다. 이러한 분리는 실행된 네트워크와 분석 산출물 사이에 **시맨틱 갭(semantic gap)**을 만들어 내며, 연산자 시맨틱, 텐서 레이아웃, 전처리, 부동소수점 코너 케이스 등의 암묵적 규약에 보장이 의존하게 된다.

TorchLean의 핵심 주장은 이 갭을 제거하자는 것입니다. TorchLean은 Lean 4 정리 증명기 안에서 학습된 모델을 **실행과 검증 모두에 공유되는 단 하나의 정밀한 시맨틱을 가진 일급(first-class) 수학적 객체**로 취급하는 프레임워크다.

### 1.2 주요 기여

TorchLean은 다음을 통합한다: **(1)** PyTorch 스타일의 검증된 API — 모델 및 학습 루프 정의, eager 실행, op-tagged 계산 그래프 IR로 낮추는 compiled 모드; **(2)** 명시적 Float32 시맨틱 — 실행 가능한 IEEE-754 binary32 커널(IEEE32Exec)과 수치적 가정 및 신뢰 경계를 명시하는 proof-relevant 반올림 모델; **(3)** 네이티브 IBP 및 CROWN/LiRPA 스타일 bound propagation과 인증서 검사를 통한 검증.

TorchLean은 정확 및 유한 정밀도 텐서 시맨틱, 검증된 역방향 미분, 구간 및 어파인 bound propagation, CROWN/LiRPA 스타일 인증서 검사, import/export 워크플로, FFI 경계를 통한 CUDA 기반 실행을 지원한다. 또한 어텐션/FlashAttention, 상태공간 시퀀스 모델, 확산 및 샘플링 프로세스, 확률 커널, 강화학습 목적함수와 MDP, 마스크드 오토인코딩·JEPA·분산/상관 기반 anti-collapse 손실 같은 자기지도 목적함수에 대한 시맨틱 레이어도 포함한다.

TorchLean은 분류기를 위한 **인증된 강건성 인증서**, **PINN 스타일 과학 모델**의 물리 정보 잔차/도함수 바운드, **신경 컨트롤러**의 제어 지향 안전성/안정성 검사라는 세 가지 구체적인 엔드투엔드 사용 사례와, 보편 근사 정리를 포함한 기계화된 이론적 결과를 통해 검증한다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

신경망은 과학적·안전 필수·임무 필수 파이프라인에 점점 더 많이 배포되고 있지만, 검증과 분석은 모델을 정의하고 실행하는 프로그래밍 환경 밖에서 수행되는 경우가 많다. 이로 인해 실행된 네트워크와 분석 산출물 사이에 **시맨틱 갭**이 생기며, 연산자 시맨틱, 텐서 레이아웃, 전처리, 부동소수점 동작, 그래프 변환, 가속 커널, 외부 인증서에 관한 암묵적 규약에 보장이 의존할 수 있다.

광범위하게 보면 문헌은 (i) Reluplex·Marabou처럼 검증 문제를 충족가능성/제약 쿼리로 인코딩하는 솔버 기반 방법, (ii) IBP·CROWN/DeepPoly 등 추상 해석/완화 방법으로 나뉘는데, 이러한 도구들은 종종 내보낸 산출물(ONNX/TorchScript/커스텀 IR)에서 작동하므로 **export/해석 단계에서 추가적인 신뢰 경계를 상속**한다.

### 2.2 제안하는 방법 및 수식

#### (a) Interval Bound Propagation (IBP)

IBP는 입력 $\ell$ - $\infty$ 퍼터베이션 $\|\delta\|_\infty \le \varepsilon$ 하에서 각 레이어의 출력 구간 $[\mathbf{l}^{(k)}, \mathbf{u}^{(k)}]$를 레이어별로 전파합니다.

선형 레이어 $\mathbf{y} = W\mathbf{x} + \mathbf{b}$에 대해, $W = W^+ - W^-$ (양수/음수 부분 분해) 형태로:

$$\mathbf{l}^{(k+1)} = W^+\mathbf{l}^{(k)} + W^-\mathbf{u}^{(k)} + \mathbf{b}$$

$$\mathbf{u}^{(k+1)} = W^+\mathbf{u}^{(k)} + W^-\mathbf{l}^{(k)} + \mathbf{b}$$

IBP는 Linear, Conv2d, BatchNorm, MaxPool, AvgPool 등 **모든 레이어 타입**에 대해 구현된다.

#### (b) CROWN / LiRPA 스타일 선형 바운드

CROWN은 비선형 활성화 함수(예: ReLU)를 선형으로 완화하여 네트워크 출력에 대해 더 정밀한(tight) 선형 바운드를 계산합니다. ReLU 뉴런 $y = \max(0, x)$에 대해 입력 구간 $[l, u]$에 따라:

- **Dead** ($u \le 0$): $y = 0$
- **Alive** ($l \ge 0$): $y = x$
- **Unstable** ($l < 0 < u$): 상하한을 각각 선형 함수로 완화

$$\underline{a}x \le y \le \bar{a}x + \bar{b}$$

네트워크 최종 출력에 대한 선형 바운드:

$$f(\mathbf{x}) \ge \mathbf{A}\mathbf{x} + \mathbf{b}_{lower}, \quad f(\mathbf{x}) \le \mathbf{\hat{A}}\mathbf{x} + \mathbf{b}_{upper}$$

CROWN 검증은 IBP보다 더 tight한 바운드를 제공한다. 실험 결과, IBP의 전체 출력 너비(0.284)보다 CROWN의 출력 너비(0.200)가 더 작아 더 정밀한 검증이 가능하다.

#### (c) $\alpha, \beta$-CROWN (Branch-and-Bound 포함)

파라미터화된 CROWN 분석, 즉 $\alpha$-CROWN은 신경망 검증을 위한 실용적으로 성공적인 bound propagation 방법으로 부상했다.

$\alpha$-CROWN에서는 불안정 ReLU 뉴런 각각에 대해 하한 완화 기울기 $\alpha_i \in [0, 1]$를 최적화:

$$\max_{\alpha} \min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x}) \ge \max_{\alpha} \left( \mathbf{A}(\alpha)\mathbf{x}_0 + b_{lower}(\alpha) - \|\mathbf{A}(\alpha)\|_1 \varepsilon \right)$$

$\beta$-CROWN에서는 분기(branching)와 바운드 타이트닝을 결합하여 완전 검증(complete verification)을 수행합니다.

#### (d) IEEE-754 Float32 시맨틱

TorchLean은 실행 가능한 IEEE-754 binary32 커널(IEEE32Exec)과 **수치적 가정과 신뢰 경계를 명시하는 proof-relevant 반올림 모델**을 통한 명시적 Float32 시맨틱을 제공한다.

#### (e) 공유 IR (SSA/DAG 계산 그래프)

TorchLean은 eager 및 compiled 모드 모두 공유 op-tagged SSA/DAG 계산 그래프 IR로 낮추는 PyTorch 스타일 검증된 API를 통합한다.

### 2.3 모델 구조

프레임워크 구조는 다음과 같이 구성된다: **Runtime/** (IEEE-754 시맨틱 — Float32.lean, Arith.lean, Semantics.lean), **Frontend/** (PyTorch 스타일 API — Tensor.lean, Layers.lean, Graph.lean, Execution.lean), **Verification/** (IBP.lean, Crown.lean, AlphaBetaCrown.lean, Robustness.lean, Certificate.lean).

실행 가능한 최신 ML 예시로는 GPT-2 스타일 텍스트, Mamba/SSM 시퀀스 모델, 확산 모델, ResNet, ViT, MAE/자기지도 학습, 강화학습이 포함된다.

TorchLean은 소형 신뢰 커널과 확장 가능한 자동화 기능을 중심으로 설계된 프로그래밍 언어이자 대화형 정리 증명기(ITP)인 Lean 4 위에 구축되었다. Lean은 신경망의 속성을 단순히 기술하는 것이 아니라, 모델 정의·컴파일·실행·검증을 단일 형식 환경 내에서 구현하는 전체 파이프라인을 위해 적합하다.

### 2.4 성능 향상

TorchLean은 **인증된 강건성 인증서(분류기)**, **PINN 스타일 과학 모델**의 물리 정보 잔차/도함수 바운드, **신경 컨트롤러**의 제어 지향 안전성/안정성 검사, 그리고 **보편 근사 정리를 포함한 기계화된 이론적 결과**에 대해 엔드투엔드로 검증된다.

Lean은 주장된 정리에 충분한 형상(shape)·그래프·구간·어파인·스키마 의무를 검사하거나, 생성자 자체가 커널 외부에 있을 때 명명된 가정을 기록한다. 따라서 성공적인 검증 결과는 Lean 측 그래프 시맨틱에 관한 진술이며, 모든 외부 런타임이나 익스포터가 검증되었다는 암묵적 주장이 아니다.

### 2.5 한계

논문의 한계로 명시적으로 확인된 것은 다음과 같습니다:

1. 검증 결과는 **Lean 내부 그래프 시맨틱에 한정**되며, 외부 런타임(예: CUDA 커널, ONNX 인터프리터 등)의 올바름을 포함하지 않는다. 즉 외부 FFI 경계 바깥의 신뢰 경계는 여전히 명시적으로 남는다.

2. VNN-COMP 규약에 따라 각 벤치마크 인스턴스는 ONNX 네트워크와 VNN-LIB 속성 명세를 쌍으로 사용하며, Python export 단계가 추가로 필요하다.

3. 대규모 현대 신경망(수십억 파라미터 수준)에 대한 검증 확장성은 아직 공개된 수치적 성능 결과가 제한적입니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 인증된 강건성을 통한 일반화

TorchLean이 제공하는 **certified robustness**는 일반화 성능과 직결됩니다. IBP와 CROWN 기반의 certified training은 단순한 경험적 정확도를 넘어, 입력 퍼터베이션에 대한 수학적으로 보장된 강건성을 제공합니다.

$$\forall \mathbf{x}' \in \mathcal{B}_\infty(\mathbf{x}, \varepsilon): \arg\max_k f_k(\mathbf{x}') = \arg\max_k f_k(\mathbf{x})$$

이 인증은 테스트 데이터 분포 변화에 대한 **worst-case 일반화 보장**을 의미합니다.

신경망이 오류가 물리적·사회적 비용을 유발하는 시스템에 점점 더 내재화됨에 따라, 머신러닝 커뮤니티는 경험적 정확도를 넘어, **퍼터베이션에 대한 강건성, 하드 안전/안정성 제약 충족, 네트워크 계산에서 도출된 양에 대한 보수적 바운드**라는 보장에 더 큰 비중을 두고 있다. 이를 위해 Reluplex·Marabou 같은 SMT/MILP 기반 solver 접근법과, IBP·CROWN·싱 등의 bound propagation 기반 방법을 아우르는 활발한 검증 생태계가 형성되었다.

### 3.2 PINN 및 과학 ML에서의 일반화

물리 정보 잔차/도함수 바운드는 PINN 스타일 과학 모델에서 **물리 법칙을 통한 일반화 제약**을 형식적으로 보장한다. 즉, 물리 방정식의 잔차 $\mathcal{R}(u_\theta)$에 대해:

$$\|\mathcal{R}(u_\theta)\|_\infty \le \delta$$

를 형식적으로 증명함으로써 학습 데이터 외 영역에서도 모델의 물리 일관성(physical consistency)을 보장하며, 이는 **분포 외(out-of-distribution) 일반화**에 직결됩니다.

### 3.3 Lyapunov 안정성과 제어 시스템의 일반화

Lyapunov 스타일 신경 컨트롤러 검증을 통해, 학습된 제어 정책이 새로운 초기 조건 및 상태 공간 영역에서도 안정성을 보장할 수 있습니다:

$$V(\mathbf{x}) > 0, \quad \dot{V}(\mathbf{x}) < 0 \quad \forall \mathbf{x} \in \mathcal{D}$$

이는 강화학습 에이전트의 **상태 공간 전반에 걸친 일반화 성능 보장**으로 해석됩니다.

### 3.4 자기지도 학습의 anti-collapse 손실

TorchLean은 분산/상관 기반 **anti-collapse 손실(VICReg 스타일)**을 포함한 자기지도 목적함수를 지원한다. 이는 표현 공간의 붕괴를 방지하여 새로운 데이터 분포에서의 일반화를 구조적으로 뒷받침합니다.

### 3.5 보편 근사 정리의 기계화

보편 근사 정리를 포함한 기계화된 이론적 결과는 신경망의 이론적 표현력이 형식적으로 증명되었음을 의미하며, 충분한 규모의 네트워크가 주어진 함수 클래스를 근사할 수 있음을 보장합니다. 이는 일반화 이론의 기초를 형식화하는 첫 단계로 의미가 있습니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 미치는 영향

#### (a) 형식 검증과 딥러닝의 통합 패러다임 전환

TorchLean이 제공하는 시맨틱 우선(semantics-first) 인프라는 학습 가능 시스템의 완전 형식적·엔드투엔드 검증을 위한 기반이 된다. 이는 AI 안전성 연구에서 **검증 가능성이 설계 단계부터 내재된 모델 개발 패러다임**으로의 전환을 촉진할 것입니다.

#### (b) 안전 필수 AI 시스템에의 직접 적용

IBP, CROWN/LiRPA, 강건성 인증서, PINN, Lyapunov 스타일 컨트롤러, VNN-COMP 스타일 산출물, 3D 기하 인증서에 대한 검증 워크플로가 갖춰짐으로써, 자율주행·의료·항공우주 등 안전 필수 도메인에서 신경망 채택의 법적·기술적 기준을 충족하는 데 기여할 것입니다.

#### (c) 검증 생태계의 신뢰 경계 축소

기존 도구들은 내보낸 산출물(ONNX/TorchScript/커스텀 IR)에서 작동하여 export/해석 단계에서 추가적인 신뢰 경계를 상속한다. TorchLean은 이 경계를 단일 형식 환경 내로 통합함으로써 **신뢰 연쇄(chain of trust)**를 획기적으로 단축시킵니다.

#### (d) 관련 연구 비교 (2020년 이후)

| 도구/연구 | 방식 | 환경 | 공식 증명 | 확장성 |
|---|---|---|---|---|
| **TorchLean (2026)** | IBP + CROWN + $\alpha,\beta$-CROWN | Lean 4 | ✅ (Lean 커널) | 중간 |
| **$\alpha,\beta$-CROWN (2021~2025)** | Branch-and-Bound + CROWN | Python/CUDA | ❌ | 높음 |
| **auto_LiRPA (2020~)** | LiRPA bound propagation | Python | ❌ | 높음 |
| **Luna (2026)** | $\alpha$-CROWN | C++ | ❌ | 높음 |
| **Marabou (2019~2024)** | SMT/MILP | C++ | ❌ | 중간 |
| **Isabelle/HOL (Brucker, 2023)** | Feedforward NN 증명 | Isabelle | ✅ | 낮음 |

$\alpha$ - $\beta$ -CROWN은 VNN-COMP 2021, 2022, 2023, 2024, 2025에서 우승한 신경망 검증기로, 효율적이고 확장 가능하며 GPU 가속을 지원한다. 그러나 이는 형식 증명을 제공하지 않는 반면, TorchLean은 Lean 커널 내에서 수학적 보장을 제공합니다.

기존 $\alpha$-CROWN 구현은 Python에 한정되어 있어 기존 DNN 검증기 및 장기 프로덕션 시스템과의 통합이 복잡하다는 한계가 있으며, 이에 C++ 구현인 Luna가 제안되었다. TorchLean은 이와 달리 Lean 4 내 형식 증명과 실행 가능성을 동시에 달성합니다.

### 4.2 앞으로 연구 시 고려할 점

#### (a) 확장성 문제
대규모 모델(GPT-4, LLaMA 등 수십억 파라미터)에 대한 형식 검증은 현재 Lean의 커널 검사 속도 및 메모리 제약으로 인해 실현 어렵습니다. **계층적 추상화(compositional verification)** 및 **증명 스케치(proof sketching)** 방법론 개발이 필요합니다.

#### (b) 부동소수점 시맨틱의 정밀도
IEEE-754 binary32 커널을 통한 명시적 Float32 시맨틱이 제공되지만, GPU의 실제 연산(예: cuBLAS, cuDNN의 연산 순서 변경, mixed precision)과 Lean 내 시맨틱 사이의 완전한 동치 증명은 여전히 열린 문제입니다.

#### (c) 검증 자동화 수준 향상
현재 검증은 사용자가 명시적으로 속성을 기술해야 합니다. 향후에는 **자동 속성 추출(automated property extraction)** 및 **LLM 기반 증명 자동화**와의 통합이 중요한 연구 방향이 될 것입니다.

#### (d) 훈련과 검증의 공동 최적화 (Certified Training)
IBP 대비 CROWN/LiRPA의 선형 바운드 완화를 활용한 검증된 오류율 향상은 **certified training** 연구와 직접 연결됩니다. TorchLean 환경 내에서 IBP·CROWN 바운드를 손실 함수에 직접 통합하는 certified training 루프의 형식화가 필요합니다.

#### (e) 새로운 아키텍처 지원
GPT-2 스타일, Mamba/SSM, 확산 모델, ResNet, ViT, MAE/자기지도 학습, 강화학습 예시가 제공되지만, 이들 아키텍처 전반에 대한 **형식 검증 정리(formal theorems)** 확장이 후속 연구의 핵심 과제입니다.

#### (f) 분산 외 일반화(OOD Generalization) 형식화
현재 인증은 주로 $\ell$ - $\infty$ 퍼터베이션에 집중됩니다. 향후 **자연적 분포 이동(natural distribution shift)**, **도메인 적응(domain adaptation)**에 대한 형식적 일반화 보장 연구가 필요합니다.

---

## 📚 참고 자료 및 출처

| # | 출처 | URL |
|---|---|---|
| 1 | **arXiv 논문 (주 논문)** | https://arxiv.org/abs/2602.22631 |
| 2 | **arXiv PDF** | https://arxiv.org/pdf/2602.22631 |
| 3 | **arXiv HTML (전문)** | https://arxiv.org/html/2602.22631v1 |
| 4 | **공식 프로젝트 페이지** | http://torchlean.org |
| 5 | **LeanDojo TorchLean 소개** | https://leandojo.org/torchlean.html |
| 6 | **GitHub (lean-dojo/TorchLean)** | https://github.com/lean-dojo/TorchLean |
| 7 | **Cool Papers** | https://papers.cool/arxiv/2602.22631 |
| 8 | **alphaXiv** | https://www.alphaxiv.org/resources/2602.22631 |
| 9 | **NASA ADS** | https://ui.adsabs.harvard.edu/abs/2026arXiv260222631G/abstract |
| 10 | **Luna Bound Propagator (arXiv:2603.23878)** | https://arxiv.org/pdf/2603.23878 |
| 11 | **GCP-CROWN (arXiv:2208.05740)** | https://arxiv.org/pdf/2208.05740 |
| 12 | **GitHub Topics: robustness-verification** | https://github.com/topics/robustness-verification |
| 13 | **GitHub: nktkt/leanx (IBP/CROWN 구현 참고)** | https://github.com/nktkt/leanx |

> ⚠️ **정확성 고지:** 본 답변은 arXiv에 공개된 논문 원문(2602.22631)과 공식 프로젝트 페이지, GitHub 저장소를 기반으로 작성되었습니다. 수식 중 IBP/CROWN의 일반적 형태는 해당 분야의 표준 문헌(Gowal et al. 2018, Zhang et al. 2018)에 기반하였으며, 논문에 명시적으로 적힌 수식과 미세한 표기 차이가 있을 수 있습니다. 논문 전문을 직접 확인하시기를 권장합니다.
