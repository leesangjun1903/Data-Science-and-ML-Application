# FP8 is All You Need (Part 1): Debunking Hardware FP64 as the HPC Holy Grail

---

## 1. 핵심 주장과 주요 기여 (간결 요약)

### 핵심 테제

본 논문은 **"FP8 텐서 코어 행렬 곱셈이 배정밀도(FP64) 과학 계산의 유일한 계산 프리미티브"**라고 주장한다. 즉, 네이티브 FP64 실리콘은 더 이상 하드웨어 요구사항이 아니라, FP8 프리미티브 위에서의 합성(composition)을 통해 도출되는 정밀도 보장(derived accuracy guarantee)으로 격하된다.

### 배경: FP64의 붕괴

NVIDIA Blackwell Ultra(B300)에서 FP64 처리량은 ~1.2 TFLOPS로 사실상 제거되었고, FP8은 5 PFLOPS에 달한다. 이 비율은 3800:1이다. 이에 따른 두 가지 결과:

**결과 1**: 메모리-바운드 커널의 리지 포인트가

$$\text{Ridge} = \frac{P_{\text{FP64}}}{B_{\text{mem}}} = \frac{1.3 \text{ TFLOPS}}{8 \text{ TB/s}} = 0.16 \text{ FLOPS/Byte}$$

로 붕괴되어, 모든 표준 HPC 커널이 compute-bound로 전락.

**결과 2**: FP8 텐서 코어가 전통적 HPC 워크로드에서 완전히 유휴(idle) 상태.

### 주요 기여 6가지

1. **"FP8 is all you need" 테제**: 모든 FP64 HPC 커널이 FP8 MMA의 시퀀스로 환원됨
2. **계층적 합성 구조 (L0–L4)**: FP8 → Ozaki II → Berkeley Dwarfs → Solver Kernels → Applications
3. **반증 가능한 주장 + TME 모델**: Tensor–Memory Equilibrium 모델로 테제를 검증
4. **레지스터-레벨 퓨전 메커니즘**: $\beta \to 1$을 달성하는 구체적 방법
5. **교차 아키텍처 정량적 검증**: B300, Rubin R200, H100 기준 성능 비교
6. **분석-검증 분업 명확화**: 본 논문은 분석(Part 1), 후속 논문에서 실제 측정

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능/한계

### 2.1 해결하고자 하는 문제

AI 가속기의 FP64 처리량 붕괴로 인해:
- 메모리-바운드 과학 커널이 compute-bound로 전락
- HBM 대역폭이 물리적으로 존재하지만 소비 불가
- FP8 텐서 코어는 전통 HPC 코드에서 휴면 상태 → **Dark Silicon 문제**

### 2.2 제안 방법: Ozaki Scheme II (CRT 기반)

#### Ozaki Scheme I (맨티사 슬라이싱)

$$A = \sum_{p=1}^{S_A} A^{(p)}, \quad B = \sum_{q=1}^{S_B} B^{(q)} \tag{1}$$

$$C \approx \sum_{p=1}^{S_A} \sum_{q=1}^{S_B} A^{(p)} B^{(q)} \tag{2}$$

비용이 $\Theta(S^2)$로 이차적으로 증가하는 단점이 있음.

#### 슬라이스 카운트 제약

$$2b + \lceil \log_2 k \rceil \leq w_{\text{acc}} \tag{3}$$

여기서 $w_{\text{acc}}$는 누산기 유효숫자 비트 (FP32: 24, INT32: 31). FP8(E4M3)의 경우 실효 입력 맨티사가 ~4비트이므로, $k=1024$에서 $S \approx 14$개의 슬라이스가 필요.

#### Ozaki Scheme II: CRT 기반 (핵심 방법)

**Phase 1: 정수 스케일링**

$$\tilde{A} = \lfloor DA \rceil \in \mathbb{Z}^{m \times k}, \quad \tilde{B} = \lfloor BE \rceil \in \mathbb{Z}^{k \times n} \tag{4}$$

**Phase 2: 모듈러 GEMM**

서로소인 moduli $m_1 < m_2 < \cdots < m_r$를 선택하여:

$$M = \prod_{i=1}^{r} m_i > 2 \cdot \max_{ij} |(\tilde{A}\tilde{B})_{ij}| \tag{5}$$

$$C^{(i)} = \left(\tilde{A} \bmod m_i\right)\left(\tilde{B} \bmod m_i\right) \bmod m_i \tag{6}$$

**Phase 3: Garner 알고리즘 (CRT 재구성)**

$$v_1 = C^{(1)}_{ij}$$

$$v_k = \left(C^{(k)}_{ij} - \sum_{j=1}^{k-1} v_j \prod_{\ell=1}^{j-1} m_\ell\right) \cdot \left(\prod_{\ell=1}^{k-1} m_\ell\right)^{-1} \pmod{m_k}, \quad k \geq 2 \tag{7}$$

비용이 $r$개의 저정밀도 GEMM + $O(r^2)$ 원소별 재구성으로 **선형(linear) 스케일링** 달성.

#### FP8 기반 비용 구조

FP8 Ozaki II에서 각 modulus당 Karatsuba 구조로 인해 **3배의 FP8 MMA + 1회 최대 크기 추정 패스**:

$$\alpha = 3r + 1$$

$r=12$에서 $\alpha = 37$.

### 2.3 TME(Tensor–Memory Equilibrium) 모델

#### 네이티브 FP64 실행 시간 (기존 Roofline)

$$T_{\text{nat}} = \max\left(\frac{W}{P_{\text{FP64}}}, \frac{Q}{B_{\text{mem}}}\right) + L_{\text{mem}} \tag{8}$$

#### 에뮬레이션 실행 시간 (TME 확장)

$$T_{\text{emu}} = \max\left(\frac{\alpha W}{P_{\text{low}}}, \frac{\beta Q}{B_{\text{mem}}}\right) + \gamma n_{\text{out}} \tag{9}$$

세 가지 에뮬레이션 파라미터:
- $\alpha$: FP64 FMA당 저정밀도 MMA 수 ($= 3r+1 = 37$ at $r=12$)
- $\beta \geq 1$: 대역폭 승수 (레지스터 퓨전 시 $\beta=1$, 미퓨전 시 $\beta=r$)
- $\gamma \geq 0$: 출력 원소당 Garner 재구성 지연

#### Case A: 네이티브 compute-bound → 에뮬레이션 memory-bound

$$\frac{T_{\text{nat}}}{T_{\text{emu}}} = \frac{W/P_{\text{FP64}}}{Q/B_{\text{mem}}} = \frac{I \cdot B_{\text{mem}}}{P_{\text{FP64}}} \tag{12}$$

B300에서 7-point stencil ($I=0.5$): 속도향상 $= 0.5 \times 8 / 1.3 \approx 3.1\times$

#### Case B: 양쪽 모두 memory-bound

$T_{\text{emu}}/T_{\text{nat}} \to \beta$ → 레지스터-레벨 퓨전($\beta=1$)이 필수조건

#### Case C: 양쪽 모두 compute-bound (dense GEMM)

속도향상 $= \rho/\alpha$, B300에서 FP8 ceiling:

$$\frac{P_{\text{FP8}}}{\alpha} = \frac{5000}{37} \approx 135 \text{ TFLOPS} \approx 104\times \text{ over native FP64}$$

### 2.4 메모리-바운드 커널 구현 전략

레지스터-레벨 퓨전의 핵심 단계:
1. HBM → 공유 메모리로 FP64 타일 로드
2. 레지스터 내에서 $r$개 residue plane 계산 (HBM 접촉 없음)
3. $r$회 FP8 MMA 수행
4. Garner 알고리즘으로 재구성 후 FP64 결과 저장

**전략 1: Batched GEMV** ($Y = AX$, batch $B \approx 8$)

$$v_{\text{tile}} = c \cdot U_{\text{im2col}}, \quad U_{\text{im2col}} \in \mathbb{R}^{7 \times N_{\text{tile}}} \tag{14}$$

목표 처리량: ~32 TFLOPS on B300 (vs. 1.3 TFLOPS native)

**전략 2: 7-point Stencil (im2col)**

$$v_{ijk} = c_0 u_{ijk} + c_1(u_{i\pm1,j,k} + u_{i,j\pm1,k} + u_{i,j,k\pm1}) \tag{13}$$

im2col 변환으로 계수 벡터를 행렬 곱으로 변환하여 텐서 코어 활용.

**전략 3: SpMV (Blocked-Ellpack)**
- 블록 열 너비 $bw$로 비정형 희소 행렬을 블록-밀도 GEMM으로 변환
- 패딩 낭비는 FP8 연산으로 처리 (FP64 대비 ~10,000배 저렴)

### 2.5 성능 향상 (정량적)

**표: B300에서 Ozaki II/FP8의 네이티브 FP64 대비 속도향상**

| 워크로드 | OI (FLOPS/B) | B300 속도향상 |
|---------|-------------|-------------|
| Dense GEMM | ≥50 | ~104× |
| Batched GEMV (B=8) | ~4 | ~24× |
| Batched GEMV (B=2) | ~1.5 | ~9.2× |
| 7-point Stencil | ~0.5 | ~3.1× |
| SpMV | ~0.2 | ~1.2× |

**H100 기준 상대 성능 (Ozaki II)**

- B300 native FP64: Dense GEMM에서 **0.02× H100** (50배 퇴보)
- B300 Ozaki II: Dense GEMM에서 **2.02× H100** (오히려 개선)
- Rubin Ozaki II: 모든 메모리-바운드 워크로드에서 **6.57× H100** (HBM4 22TB/s 비율 그대로 반영)

### 2.6 한계점

1. **$\beta=1$은 이상치**: 레지스터 파일 경쟁으로 실제 $\beta > 1$ 가능성 존재
2. **희소 구조 일부 적용 불가**: 극도로 불규칙한 스파시티 패턴
3. **FFT**: 별도 Kulisch 재구성 경로 필요 (동반 논문 [24]에서 다룸)
4. **검증 부재**: 본 논문은 순수 분석 논문 (실제 하드웨어 측정은 후속 작업)
5. **ADP 폴백 빈도**: 실제 조건에서 native FP64로의 폴백 빈도 미확인

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

이 논문은 머신러닝 모델의 일반화를 직접 다루지 않지만, **과학 계산의 정밀도 일반화**와 **HPC 커널 분류법의 완전성**이라는 두 가지 의미에서 일반화를 논의한다.

### 3.1 Berkeley Dwarf 분류법을 통한 완전성 보장

논문의 핵심 일반화 전략은 Berkeley Dwarfs [3,6]의 **폐쇄성(closedness)** 활용이다:

$$\forall \text{kernel} \in \text{Berkeley Dwarfs (L2)} \Rightarrow \exists \text{ reduction to FP8 MMA (L0) via Ozaki II (L1)}$$

$$\Rightarrow \forall \text{composition} \in \text{L3, L4} \Rightarrow \exists \text{ reduction to L0}$$

이 합성 논리는 특정 커널의 샘플링이 아닌 **전체 과학 계산 공간에 대한 보장**을 제공한다.

**Dwarf 클래스별 일반화 검증 요약 (표 6 기반)**:

| Dwarf 클래스 | 환원 경로 | B300 판정 |
|------------|---------|---------|
| 밀집 선형대수 | GEMM → Ozaki II | 메모리 루프 달성 |
| 희소 선형대수 | SpMV/SpMM → Ozaki II | 메모리 루프 달성 |
| 스펙트럼 방법 | 3-D FFT → Kulisch | 메모리 루프 달성 |
| 구조화 격자 | Stencil/im2col → Ozaki II | 메모리 루프 달성 |
| 비구조화 격자 | FEM SpMV → Ozaki II | 메모리 루프/코너 케이스 |
| BLAS-1 리덕션 | FP32+Kahan | 바인딩 아님 (<5%) |
| 격자 QCD | Dirac → Stencil/GEMM | 메모리 루프 달성 |
| Monte Carlo | — | 코너 케이스 (§9) |

### 3.2 에러 경계의 일반화

Ozaki II는 **임의의 조건수를 가진 입력에 대해 증명 가능한 오차 경계**를 제공:

$$\text{componentwise relative error} \leq u_{\text{FP64}} + \delta_{\text{rounding}} \approx 2^{-53}$$

실측 오차는 $2 \sim 10 \cdot u_{\text{FP64}}$ 수준으로, 이는 native FP64 DGEMM과 동등.

**ADP(Automatic Dynamic Precision) 메커니즘**을 통해:
- 런타임에 슬라이스 수 $r$을 입력 동적 범위에 따라 자동 조정
- ESC(Exponent-Span-Capacity) 추정기로 ill-conditioned 입력 감지
- worst-case에서도 10% 미만의 오버헤드 ([42] Schwarz et al. 실측)

### 3.3 아키텍처 간 일반화

**TME 모델의 파라미터화**는 특정 하드웨어에 종속되지 않음:

$$T_{\text{emu}} = \max\left(\frac{(3r+1)W}{P_{\text{low}}}, \frac{\beta Q}{B_{\text{mem}}}\right) + \gamma n_{\text{out}}$$

이 모델은 FP8이 지배적인 B300뿐만 아니라:
- FP8:FP64 = 30:1인 H100에서 INT8이 우월한 기판임을 정확히 예측
- FP8:FP64 = 113:1인 B200에서 FP8의 우월성 (~8.6×) 예측
- 미래 아키텍처에도 $\rho = P_{\text{low}}/P_{\text{FP64}}$만 업데이트하면 적용 가능

### 3.4 희소 커널에서의 일반화 (Appendix F의 핵심 논리)

**반직관적 결과**: FP8 유닛이 낮은 활용률로 동작해도 $\beta \approx 1$ 유지 가능.

요구 FP8 처리율 (SpMV 기준):

$$\text{Required FP8 rate} = \alpha \cdot \frac{B_{\text{mem}}}{2} \cdot 2 = 37 \times 4 \text{ TB/s} \approx 296 \text{ TFLOPS} = 5.9\% \text{ of peak}$$

최악의 경우 (m16 n8 k32 MMA에서 1/8 레인 사용):

$$\text{Derated peak} \approx 625 \text{ TFLOPS} \gg 296 \text{ TFLOPS (required)}$$

따라서 SpMV는 텐서 코어 활용률이 매우 낮아도 여전히 memory-bound를 유지.

### 3.5 한계: 일반화가 실패하는 지점

1. **잠재력-바운드 코드** (f 범주): 가속기 빔 역학, 지오다이나모 시뮬레이션 등 - FP64 처리량이 아닌 커널 런치 레이턴시가 병목
2. **확률론적 방법** (g 범주): FCI-QMC, AFQMC - 극단적 동적 범위
3. **비정규 희소 구조**: Blocked-ELL 패딩 비율 $\rho_{\text{pad}} \gg 1$인 경우 $\beta > 1$로 퇴화

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 HPC 하드웨어 설계에 대한 영향

본 논문이 제시하는 **4-floor 코디자인 규칙** (동반 논문 [24] 기반):

향후 post-FP64 GPU는 다음 조건 중 하나를 만족해야 함:

$$\eta_{\text{FP64-vec}} \geq 1.56 B_{\text{mem}} \quad \text{(안전 네이티브 목표)}$$

또는 다음 두 조건을 동시에:

$$\eta_{\text{INT32-vec}} \geq 8.25 B_{\text{mem}} \quad \text{(Kulisch 서브플로어)}$$
$$\eta_{\text{FP8}} \geq 170 B_{\text{mem}} \quad \text{(FP8 텐서코어 플로어)}$$

이 설계 경계를 위반하면(FP64와 INT32를 동시에 삭제하면) 기존 탈출 경로가 모두 봉쇄됨.

**AMD MI430X와의 경쟁** [49]: AMD는 native FP64 강화 전략을 선택하고 있어, 두 접근법의 실증적 검증이 필요.

### 4.2 수치 소프트웨어 커뮤니티에 대한 영향

**cuBLAS 통합** (2025년 10월 [35]): NVIDIA가 Ozaki-style 에뮬레이션을 cuBLAS에 공식 통합함으로써 이 기술이 연구 호기심에서 전략적 인프라로 전환됨.

**GEMMul8 오픈소스 라이브러리** [44]: INT8/FP8 기반 GEMM 에뮬레이션의 참조 구현 제공.

### 4.3 앞으로 연구 시 고려할 점

**즉시 해결 필요한 문제들**:

1. **실제 $\beta$ 측정**: 생산 타일 크기에서 레지스터 퓨전의 실제 효율성 측정
   - 레지스터 파일 경쟁 모델: $\beta(r, T_k, T_m, T_n, \text{regs})$의 정확한 파라미터화 필요

2. **Kulisch FFT 커널 구현**: 현재 어떤 프로덕션 라이브러리에도 존재하지 않음

3. **ADP 폴백 빈도**: 실제 과학 워크로드에서 native FP64로 폴백하는 빈도 실측

4. **2:4 구조적 희소성과 Ozaki II의 결합**: 잔차 평면 레벨에서 희소성 마스크 적용 시 모듈러 리덕션이 무효화될 수 있음 — **현재 미해결 문제**

**중기 연구 방향**:

5. **비-GEMM 핫스팟 처리**: 정수 지배적 정렬(sort), 스캔, 원자적 그래프 순회의 $(\alpha, \beta, \gamma)$ 분석

6. **일반화 성능 검증**: 역-조건 행렬, NaN/Inf 처리, 비트-재현성 등 IEEE-FP64 완전 의미론의 에뮬레이션

7. **극도로 불규칙한 희소 구조**: Monte Carlo/입자 코드, 동적 연결성을 가진 코드에 대한 에뮬레이션 전략

**장기 연구 방향**:

8. **FP4/NVFP4 기반 Ozaki III**: FP8이 FP16을 대체했듯이, NVFP4 시대에 대응하는 다음 세대 에뮬레이션 스킴

9. **AI-보조 커널 생성**: 논문이 언급하듯 에뮬레이션 구현 작업(잔차 분해, 텐서-코어 MMA, Garner 재구성의 각 커널 타일 형태로의 번역)은 AI 코딩 보조도구에 적합 → 수년이 아닌 수개월 내 구현 가능성

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 핵심 방법 | 정밀도 | 한계 | 본 논문과의 관계 |
|-----|---------|------|------|--------------|
| **Markidis et al., 2018** [23] | FP16 텐서코어로 FP32 정밀도 | FP32 | FP64 미지원 | 선구자 작업 |
| **Mukunoki et al., ISC 2020** [27] | INT8/FP16 기반 Ozaki I DGEMM | FP64 | $\Theta(S^2)$ 비용 | Ozaki I의 현대적 적용 |
| **Ootomo, Ozaki, Yokota, 2024** [38] | INT8 텐서코어 DGEMM | FP64 | INT8 감소 추세 | INT8 기반 선행 연구 |
| **Ozaki, Uchino, Imamura, 2025** [40] | **CRT 기반 Ozaki II** (INT8) | FP64 | FP8 직접 불가 | 본 논문의 핵심 기반 |
| **Uchino, Ozaki, Imamura, 2026** [46] | **FP8 양자화 트릭** → Ozaki II on FP8 | FP64 | 2026년 최신 | 본 논문이 활용하는 FP8 적응 |
| **Schwarz et al., 2025** [42] | **ADP + ESC**: 동적 정밀도 조정 + 검증 | FP64 | 실측 연구 | 본 논문의 에러 분석 지원 |
| **TCStencil, Liu et al., 2022** [20] | im2col → 텐서코어 스텐실 | FP32/FP16 | 정밀도 미보장 | 본 논문이 FP64로 확장 |
| **SPTCStencil, Gu et al., 2025** [11] | 2:4 희소 텐서코어 스텐실 | FP32/FP16 | 정밀도 미보장 | 향후 Ozaki II 결합 가능성 |
| **SparStencil, Li et al., SC25** [19] | 구조적 희소성 변환 스텐실 | FP32/FP16 | 정밀도 미보장 | 동일 방향, 정밀도 미해결 |
| **Haidar et al., SC18** [12] | 반복 세련화 + FP16 텐서코어 | FP64 (솔버) | bare GEMM 미보장 | Ozaki II의 대안적 접근 |
| **Dongarra et al., 2026** [9] | AI 하드웨어 시대 HPC 전략 | — | 분석적 | 본 논문과 상보적 |
| **DeepSeek-V3, 2024** [7] | FP8 혼합정밀도 LLM 훈련 | BF16/FP8 | AI 워크로드 | FP8의 AI 지배성 증거 |
| **NVIDIA NVFP4, 2025** [31] | 4비트 사전훈련 | FP4 | AI 워크로드 | FP4 시대 도래 예고 |

### 차별화된 기여점

본 논문은 기존 연구들과 달리:
- **단일 커널 최적화가 아닌 전체 과학 계산의 프리미티브 레벨 이론** 제시
- **Berkeley Dwarfs 전체에 대한 완전성 감사(completeness audit)** 수행
- **메모리-바운드 커널에서의 Ozaki II 수익성** 최초 체계적 분석
- **TME 모델이라는 반증 가능한 분석 도구** 개발

---

## 참고 자료

본 답변은 다음 자료를 기반으로 작성됨:

- **Matsuoka, S. (2026).** "FP8 is All You Need (Part 1): Debunking Hardware FP64 as the HPC Holy Grail." *arXiv:2606.06510v2 [cs.AR]*, 13 Jun 2026.
- **Ozaki, K., Uchino, Y., and Imamura, T. (2025).** "Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix multiplication using an integer modular technique." [40]
- **Uchino, Y., Ozaki, K., and Imamura, T. (2026).** "Double-precision matrix multiplication emulation via Ozaki-II scheme with FP8 quantization." [46]
- **Schwarz, A. et al. (2025).** "Guaranteed DGEMM accuracy while using reduced precision tensor cores through extensions of the Ozaki scheme." *arXiv:2511.13778*. [42]
- **Williams, S., Waterman, A., and Patterson, D. (2009).** "Roofline: An insightful visual performance model for multicore architectures." *Communications of the ACM*, 52(4):65–76. [47]
- **Asanović, K. et al. (2006).** "The landscape of parallel computing research: A view from Berkeley." *Technical Report UCB/EECS-2006-183*. [3]
- **NVIDIA Corporation. (2025).** "NVIDIA Blackwell Ultra GPU Datasheet." [33]
- **NVIDIA Corporation. (2026).** "Inside the NVIDIA Vera Rubin platform." *NVIDIA Developer Blog*. [37]
- **Mukunoki, D. (2025).** "DGEMM without FP64 arithmetic: Using FP64 emulation and FP8 tensor cores with Ozaki scheme." [26]
- **Uchino, Y. (2025).** "GEMMul8: GEMM emulation using INT8/FP8 matrix engines based on the Ozaki Scheme II." *GitHub repository, RIKEN-RCCS*. [44]
- **Dongarra, J., Reed, D., and Gannon, D. (2026).** "Ride the wave: Adapting scientific computing to the AI hardware era." [9]
