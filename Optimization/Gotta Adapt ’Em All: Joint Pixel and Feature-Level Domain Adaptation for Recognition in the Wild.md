# Gotta Adapt ’Em All: Joint Pixel and Feature-Level Domain Adaptation for Recognition in the Wild

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문(CVPR 2019)은 **비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)** 에서 픽셀 수준(pixel-level)과 특징 수준(feature-level) 적응을 **동시에(jointly)** 수행함으로써, 각 방법의 상호보완적 장점을 활용해 더 높은 성능을 달성할 수 있다고 주장합니다.

- **특징 수준 DA**: 반지도 학습(Semi-Supervised Learning, SSL) 통찰 활용 → 분류 인식 가능한 도메인 불변 표현 학습
- **픽셀 수준 DA**: 컴퓨터 비전의 도메인 특화 통찰(3D 기하학, 속성 기반 이미지 합성) 활용 → 명시적 변환 요소 처리

### 주요 기여

| 기여 | 설명 |
|------|------|
| 새로운 UDA 프레임워크 | 픽셀↔특징 다중 의미 레벨 적응 |
| DANN-CA | 분류기와 판별기의 공동 파라미터화 |
| AC-CGAN | 속성 조건부 CycleGAN으로 다양한 조명 조건 이미지 생성 |
| KFNet | 키포인트 기반 Appearance Flow로 시점 합성의 실제 이미지 일반화 |
| 새 실험 프로토콜 | 웹→감시카메라 차량 인식 (CompCars 데이터셋) |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**핵심 문제**: 레이블된 소스 도메인(웹 이미지)에서 레이블 없는 타겟 도메인(감시카메라 이미지)으로 지식을 전이할 때 발생하는 **도메인 갭(domain gap)**.

구체적 변화 요인:
- **명시적(nameable) 요인**: 시점(pose), 조명(lighting) → 픽셀 수준 DA로 처리
- **암묵적(unspecified) 요인**: 날씨, 카메라 특성 등 → 특징 수준 DA로 처리

기존 방법의 한계:
- 특징 수준 DA (예: DANN): 도메인 특화 통찰 주입 어려움, mode collapse 문제
- 픽셀 수준 DA (예: CycleGAN): 고차원 최적화 문제, 단일 출력 스타일만 생성

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) 특징 수준 DA: DANN-CA (Classification-Aware DANN)

**기존 DANN의 목적 함수**:

$$\max_{\theta_c} \{\mathcal{L}_C = \mathbb{E}_{\mathcal{X}_S} \log C(f, y)\} \tag{1}$$

$$\max_{\theta_d} \{\mathcal{L}_D = \mathbb{E}_{\mathcal{X}_S} \log(1-D(f)) + \mathbb{E}_{\mathcal{X}_T} \log D(f)\} \tag{2}$$

$$\max_{\theta_f} \{\mathcal{L}_F = \mathcal{L}_C + \lambda \mathbb{E}_{\mathcal{X}_T} \log(1-D(f))\} \tag{3}$$

**제안하는 DANN-CA의 목적 함수** (분류기와 판별기 공동 파라미터화):

$$\max_{\theta_c} \{\bar{\mathcal{L}}_C = \mathbb{E}_{\mathcal{X}_S} \log \bar{C}(y) + \mathbb{E}_{\mathcal{X}_T} \log \bar{C}(N+1)\} \tag{4}$$

$$\max_{\theta_f} \{\bar{\mathcal{L}}_F = \mathbb{E}_{\mathcal{X}_S} \log \bar{C}(y|\mathcal{Y}) + \lambda \mathbb{E}_{\mathcal{X}_T} \log(1-\bar{C}(N+1))\} \tag{5}$$

여기서 조건부 점수 함수는:

$$\bar{C}(y|\mathcal{Y}) = \frac{\bar{C}(y)}{1-\bar{C}(N+1)}, \quad \forall y \leq N, \quad \bar{C}(N+1|\mathcal{Y}) = 0 \tag{6}$$

**핵심 아이디어**: 분류기를 $(N+1)$-way 출력으로 확장하여, $(N+1)$번째 출력이 타겟 도메인 판별 역할을 동시에 수행. 이를 통해 판별기가 클래스 모드(mode)를 자연스럽게 인식.

**기울기 분석** (DANN vs DANN-CA 비교):

DANN의 adversarial 기울기:

$$\frac{\partial \log(1-D(f))}{\partial f} = -D(f) w_d \tag{7}$$

DANN-CA의 adversarial 기울기:

$$\frac{\partial \log(1-\bar{C}(N+1))}{\partial f} = -\bar{C}(N+1)\left(w_{N+1} - \sum_{y=1}^{N} w_y \bar{C}(y|\mathcal{Y})\right) \tag{8}$$

> **해석**: DANN은 모든 타겟 샘플에 동일한 방향의 기울기를 적용(mode collapse 유발), 반면 DANN-CA는 각 타겟 샘플의 조건부 클래스 확률 $\bar{C}(y|\mathcal{Y})$에 따라 개인화된 기울기를 제공하여 분류 가능한 영역으로 유도.

**MCD와의 관계**: KL divergence를 통해 두 분류 분포를 정의하면:

$$p_1(y|x_t) = \bar{C}(y|\mathcal{Y}), \quad p_2(y|x_t) = \bar{C}(y), \quad y \leq N+1 \tag{9}$$

$$-\text{KL}(p_1 \| p_2) = \log(1-\bar{C}(N+1)) \tag{10}$$

이는 식 (5)의 adversarial loss와 동치 → DANN, MCD, SSL 알고리즘의 통합적 관점 제공.

---

#### (B) 픽셀 수준 DA

##### B-1. 광도 변환: AC-CGAN (Attribute-Conditioned CycleGAN)

속성 집합 $\mathcal{A}$ (예: {낮, 밤})에 대해 생성기 $G: \mathcal{X}_S \times \mathcal{A} \rightarrow \mathcal{X}_T$를 학습:

$$\max_{\theta_{d_a}} \{\mathcal{L}_{D_a} = \mathbb{E}_{\mathcal{X}_{T_a}} \log D_a(x) + \mathbb{E}_{\mathcal{X}_S} \log(1-D_a(G(x,a)))\} \tag{11}$$

$$\max_{\theta_g} \{\mathcal{L}_G = \mathbb{E}_{\mathcal{X}_S} \mathbb{E}_{\mathcal{A}} \log D_a(G(x,a))\} \tag{12}$$

Cycle consistency loss:

$$\mathbb{E}_{\mathcal{X}_S} \|F(G(x,a),a)-x\|_1 + \mathbb{E}_{\mathcal{X}_{T_a}} \|G(F(x,a),a)-x\|_1 \tag{13}$$

> 기존 CycleGAN [Zhu et al., ICCV 2017]은 단일 스타일만 출력하는 한계가 있으나, AC-CGAN은 속성 조건 변수 $a$를 통해 **다양한 조명 조건의 이미지를 동시에 생성** 가능.

##### B-2. 시점 변환: KFNet (Keypoint-based Appearance Flow Network)

Appearance Flow 기반 이미지 합성 [Zhou et al., ECCV 2016]:

$$I_p^{i,j} = \sum_{(h,w) \in \mathcal{N}} I_s^{h,w}(1-|F_y^{i,j}-h|)(1-|F_x^{i,j}-w|) \tag{14}$$

KFNet 학습 목적 함수 (AFNet으로부터 지식 증류):

$$\min\{\mathcal{L} = \|F_{\text{kpt}} - F_{\text{pix}}\|_1 + \lambda \|I_p(F_{\text{kpt}}, I_s) - I_t\|_1\} \tag{15}$$

> **핵심**: RGB 이미지 대신 **2D 키포인트**를 입력으로 사용하여 합성(synthetic)→실제(real) 도메인 일반화 문제를 해결. 키포인트는 렌더링 이미지와 실제 이미지 간 도메인 불변(domain-invariant) 표현.

---

### 2-3. 모델 구조

```
[전체 파이프라인]

웹 이미지 (레이블 있음)
    │
    ├─→ [픽셀 수준 DA]
    │       ├─ §4.2 KFNet: 시점 변환 (10°~30° elevation)
    │       └─ §4.1 AC-CGAN: 조명 변환 (낮/밤)
    │           └─→ 합성 이미지 (확장된 소스 도메인)
    │
    └─→ [특징 수준 DA: DANN-CA]
            ├─ CNN (ResNet-18) 공유 특징 추출기
            ├─ (N+1)-way 분류기 (소스: 1~N, 타겟: N+1)
            └─→ 감시카메라 이미지 (레이블 없음)

최종 출력: 감시카메라 이미지 분류
```

**구체적 아키텍처**:
- 특징 추출기: ResNet-18 (512-dim 특징)
- 분류기: 선형 레이어 (512→431/432)
- DANN 판별기: 3-layer MLP (512→320→320→1)
- AC-CGAN: PatchGAN 판별기 + UNet 생성기

---

### 2-4. 성능 향상

**CompCars 감시카메라 차량 인식 (메인 결과)**:

| 방법 | SV 전체 | 낮 | 밤 |
|------|---------|-----|-----|
| 베이스라인 (웹 전용) | 54.98% | 72.67% | 19.87% |
| CyCADA [Hoffman et al.] | 64.82% | 76.35% | 41.93% |
| DANN-CA (특징만) | 75.83% | 76.73% | 74.05% |
| MKF+AC-CGAN (픽셀만) | 79.71% | 84.10% | 70.99% |
| **MKF+AC-CGAN+DANN-CA (제안)** | **84.20%** | **85.77%** | **81.10%** |
| 지도학습 상한 | 98.63% | 98.92% | 98.05% |

→ 베이스라인 대비 오류율 **64.9% 감소**

**표준 UDA 벤치마크 (DANN-CA 단독)**:

| 방법 | M→MM | S→S | S→M | M→S | S→G |
|------|------|-----|-----|-----|-----|
| DANN | 98.00 | 92.24 | 88.70 | 82.30 | 97.38 |
| DANN-CA | **98.03** | **94.47** | **96.23** | **87.48** | **98.70** |

**Office-31 (ResNet-50)**:

| 방법 | A→W | D→W | W→D | A→D | D→A | W→A |
|------|-----|-----|-----|-----|-----|-----|
| DANN | 85.97 | 96.87 | 97.94 | 84.12 | 67.63 | 66.78 |
| DANN-CA | **91.35** | **98.24** | **99.48** | **89.94** | **69.63** | **68.76** |

---

### 2-5. 한계점

1. **부가 주석(side annotation) 필요**: 픽셀 수준 DA는 완전한 비지도 방식이 아님. 속성(낮/밤 레이블)과 3D CAD 모델이 필요하여 실제 적용 범위가 제한됨.

2. **도메인 특화성**: 실험이 주로 차량 인식에 집중. 일반적인 객체 범주로의 확장성은 충분히 검증되지 않음.

3. **복잡한 파이프라인**: KFNet + AC-CGAN + DANN-CA의 순차적 학습으로 학습 복잡도가 높고, 각 모듈 간 최적화 조율이 어려움.

4. **모델 선택 문제**: 타겟 도메인의 소수 레이블(클래스당 약 5개)을 사용한 지도적 모델 선택에 의존 → 순수 비지도 설정과 괴리.

5. **픽셀 수준 DA의 적응 유연성 한계**: 명시적으로 정의되지 않은 변환 요소(예: 복잡한 날씨 패턴)는 처리 어려움.

6. **Mode Coverage**: 특징 수준 DA(DANN-CA) 단독 시 29.6개 클래스 누락(미배정), 공동 프레임워크에서도 10.4개 누락.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화를 위한 핵심 설계 요소

**① KFNet의 도메인 불변 표현**

합성 이미지로 학습된 AFNet이 실제 이미지에서 성능 저하를 보이는 문제를 해결하기 위해, RGB 픽셀 대신 **2D 키포인트**를 입력으로 활용:

$$\min\{\mathcal{L} = \|F_{\text{kpt}} - F_{\text{pix}}\|_1 + \lambda \|I_p(F_{\text{kpt}}, I_s) - I_t\|_1\}$$

키포인트는 렌더링 도메인과 실제 도메인 모두에서 강건하게 추출 가능 [Li et al., CVPR 2017]. 실험 결과:
- AFNet (합성→실제): 왜곡된 이미지 생성
- KFNet (합성→실제): 정확한 시점 변환 유지
- L1 재구성 오류: KFNet 0.072 vs AFNet 0.071 (합성 데이터 기준 동등 성능)
- 인식 정확도: M3(59.73%) → M4(61.55%) 향상

**② DANN-CA의 클래스 조건부 기울기**

분류 경계(classification boundary)를 인식하는 판별기를 통해 타겟 샘플을 소스 도메인의 **분류 가능한 영역(classifiable regions)** 으로 유도:

$$\frac{\partial \log(1-\bar{C}(N+1))}{\partial f} = -\bar{C}(N+1)\left(w_{N+1} - \sum_{y=1}^{N} w_y \bar{C}(y|\mathcal{Y})\right)$$

이는 각 타겟 샘플의 현재 분류 확률에 따라 개인화된 기울기를 제공하므로, 다양한 클래스 분포를 가진 타겟 도메인에도 일반화 가능.

**③ 픽셀+특징 공동 적응의 상호보완성**

| 구성 | 장점 | 단점 |
|------|------|------|
| 픽셀 수준만 (M9) | Mode coverage 우수 (누락 2개), 학습 안정성 | 적응 유연성 제한 |
| 특징 수준만 (M11) | 유연한 도메인 정렬 | Mode collapse (누락 29.6개), 학습 불안정 |
| **공동 (M14)** | **양쪽 장점 결합** | 복잡한 학습 파이프라인 |

픽셀 수준 DA가 특징 수준 DA의 mode collapse를 보완:
- M11: 29.6±1.1개 클래스 누락
- M14: 10.4±0.6개 클래스 누락 (64.9% 감소)

**④ AC-CGAN의 연속적 조명 보간**

잠재 코드의 연속적 보간을 통해 다양한 조명 조건을 생성함으로써, 훈련 시 보지 못한 조명 조건에도 일반화 가능:
- CycleGAN (M6) → AC-CGAN (M7): 야간 성능 39.12% → 45.66% (+6.54%)
- MKF+CycleGAN (M8) → MKF+AC-CGAN (M9): 50.68% → 70.99% (+20.31%)

### 3-2. 일반화 한계와 극복 방향

| 한계 | 극복 가능성 |
|------|------------|
| 특정 도메인(차량)에 특화 | 일반 객체로 확장 시 도메인 특화 키포인트 추출기 필요 |
| 합성 데이터 의존 | NeRF 등 현대적 3D 합성 기술 활용 가능 |
| 부가 속성 주석 필요 | 자동 속성 발견(attribute discovery) 방법 결합 가능 |

---

## 4. 연구 영향 및 앞으로 고려할 점

### 4-1. 앞으로의 연구에 미치는 영향

**① 다중 레벨 DA의 패러다임 정립**

이 논문은 픽셀과 특징 수준 적응이 **상호보완적**임을 실증적으로 보여줌으로써, 단일 레벨 적응의 한계를 극복하는 연구 방향을 제시했습니다. CyCADA [Hoffman et al., 2018]의 단순 결합(CycleGAN+DANN)을 넘어 각 레벨에 특화된 통찰을 주입하는 방법론적 기여를 했습니다.

**② SSL과 UDA의 연결**

DANN-CA가 MCD [Saito et al., CVPR 2018], 일관성 기반 SSL 알고리즘 [Laine & Aila, ICLR 2017; Tarvainen & Valpola, NeurIPS 2017]과 통합적으로 이해될 수 있음을 보임으로써, UDA 이론의 통합적 관점을 제공했습니다.

**③ 도메인 브릿지(domain bridge)로서의 중간 표현 활용**

2D 키포인트를 합성↔실제 도메인 간 불변 표현으로 사용하는 아이디어는, 이후 **중간 표현을 활용한 도메인 적응** 연구들에 영향을 미쳤습니다.

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

> **주의**: 아래 비교는 논문에서 직접 인용된 내용이 아니며, 본 논문(CVPR 2019) 이후 발표된 관련 연구들에 대한 분석입니다. 개별 수치의 정확성은 해당 논문을 직접 확인하시기 바랍니다.

#### (A) 특징 수준 DA의 발전

**SHOT (ICML 2020)** - "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation" [Liang et al.]:
- 소스 데이터 없이 타겟 도메인에서만 적응 가능한 **소스-프리(source-free) UDA** 제안
- 본 논문의 DANN-CA가 소스 데이터를 필요로 하는 한계를 극복
- 정보 최대화(information maximization)와 pseudo-label 기반 자기 지도 학습 결합

**DAPL (NeurIPS 2022)** - "Domain Adaptation via Prompt Learning":
- CLIP 등 대규모 사전학습 모델을 활용한 프롬프트 기반 DA
- 본 논문의 수작업 특징 설계 대신 **자동 표현 학습** 방향으로 발전

#### (B) 픽셀 수준 DA의 발전

**DACS (WACV 2021)** - "Domain Adaptation via Cross-Domain Mixed Sampling":
- MixUp 기반 도메인 혼합으로 픽셀 수준 적응
- 본 논문의 CycleGAN 기반 접근보다 간단하면서도 경쟁력 있는 성능

**GAN 기반 방법의 한계 인식**:
- 이후 연구들은 GAN 학습의 불안정성과 계산 비용 문제로 인해 **Diffusion Model** 기반 도메인 변환으로 이동하는 추세

#### (C) 대규모 사전학습 모델 활용

**CDTrans (ICLR 2022)** - "CDTrans: Cross-Domain Transformer for Unsupervised Domain Adaptation":
- Vision Transformer(ViT)를 UDA에 적용
- 본 논문의 ResNet-18 기반 접근 대비 **사전학습 표현의 강점** 활용

**PMTrans (ECCV 2022)** - "Patch Mix Transformer for Unsupervised Domain Adaptation":
- 패치 수준 혼합을 통한 도메인 정렬
- 본 논문의 이미지 수준 변환을 패치 수준으로 세분화

#### (D) Source-Free & Test-Time Adaptation (TTA)

본 논문은 타겟 도메인 데이터에 접근 가능한 설정을 가정하나, 최근 연구들은 더 현실적인 설정으로 이동:

- **T3A (NeurIPS 2020)**: 테스트 시간 적응으로 배포 환경 변화 대응
- **Tent (ICLR 2021)**: 엔트로피 최소화 기반 테스트 시간 적응
- 이는 본 논문의 **엔트로피 최소화와 DANN의 관계**에 대한 이론적 분석이 TTA 연구에 기초를 제공

#### 비교 요약 테이블

| 관점 | 본 논문 (CVPR 2019) | 2020년 이후 연구 |
|------|---------------------|-----------------|
| 적응 레벨 | 픽셀+특징 공동 | 특징 중심 + 사전학습 모델 활용 |
| 픽셀 변환 | GAN 기반 | Diffusion, MixUp 등 |
| 특징 추출기 | ResNet-18/50 | ViT, CLIP 등 대규모 모델 |
| 소스 데이터 | 필요 | Source-free 방향으로 진화 |
| 주석 요구 | 속성 레이블 필요 | 완전 비지도 방향 |
| 이론적 기반 | DANN+SSL 연결 | 정보 이론, 인과 추론 |

---

### 4-3. 앞으로 연구 시 고려할 점

**① 대규모 사전학습 모델과의 결합**

CLIP, DINO 등 대규모 사전학습 모델은 이미 풍부한 도메인 불변 표현을 학습했을 가능성이 높습니다. 본 논문의 DANN-CA를 이러한 모델에 적용하거나, 프롬프트 엔지니어링으로 속성 조건을 대체하는 방향을 고려해야 합니다.

**② Source-Free 및 Test-Time Adaptation 확장**

실제 배포 환경에서는 소스 데이터 접근이 불가능한 경우가 많습니다. DANN-CA의 분류 인식 적응을 소스 프리 설정으로 확장하는 연구가 필요합니다.

**③ 픽셀 수준 변환의 Diffusion Model 대체**

GAN 기반 AC-CGAN은 학습 불안정성 문제를 내포합니다. Stable Diffusion 등 Diffusion Model을 활용한 고품질 도메인 변환으로 대체하면 더 안정적이고 다양한 이미지 생성이 가능합니다.

**④ 자동 속성 발견(Attribute Discovery)**

현재 속성(낮/밤)은 수동으로 정의됩니다. 자동으로 도메인 간 차이를 나타내는 속성을 발견하는 비지도 속성 발견 방법 [예: DIVA]과 결합하면 일반화 가능성이 높아집니다.

**⑤ 이론적 일반화 한계 분석**

Ben-David et al.의 도메인 적응 이론을 바탕으로, 본 논문의 다중 레벨 적응 프레임워크에 대한 **이론적 오류 한계(error bound)** 분석이 필요합니다. 특히 픽셀 변환 후 특징 수준 적응의 연쇄 오류 전파를 수식적으로 규명해야 합니다.

**⑥ 계산 효율성**

KFNet + AC-CGAN + DANN-CA의 파이프라인은 계산 비용이 높습니다. 경량화된 단일 통합 모델 설계(예: 엔드-투-엔드 학습)를 고려해야 합니다.

**⑦ 다중 타겟 도메인(Multi-Target DA)**

본 논문은 단일 소스→단일 타겟 설정이나, 실제 환경에서는 다양한 타겟 도메인이 존재합니다. Universal Domain Adaptation (UniDA) 또는 Multi-Target DA로의 확장이 필요합니다.

---

## 참고자료

**본 논문**:
- Tran, L., Sohn, K., Yu, X., Liu, X., & Chandraker, M. (2019). "Gotta Adapt 'Em All: Joint Pixel and Feature-Level Domain Adaptation for Recognition in the Wild." *CVPR 2019*, pp. 2672-2681.

**논문 내 핵심 참조 문헌**:
- [11] Ganin, Y., et al. "Domain-adversarial training of neural networks." *JMLR*, 2016.
- [18] Hoffman, J., et al. "CyCADA: Cycle-consistent adversarial domain adaptation." *arXiv:1711.03213*, 2017.
- [39] Saito, K., et al. "Maximum classifier discrepancy for unsupervised domain adaptation." *CVPR*, 2018.
- [40] Salimans, T., et al. "Improved techniques for training GANs." *NeurIPS*, 2016.
- [62] Yang, L., et al. "A large-scale car dataset for fine-grained categorization and verification." *CVPR*, 2015. (CompCars)
- [65] Zhou, T., et al. "View synthesis by appearance flow." *ECCV*, 2016.
- [66] Zhu, J.-Y., et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." *ICCV*, 2017.
- [25] Li, C., et al. "Deep supervision with shape concepts for occlusion-aware 3D object parsing." *CVPR*, 2017.

**2020년 이후 비교 연구** (직접 분석에 활용):
- Liang, J., et al. "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation." *ICML*, 2020.
- Wang, D., et al. "Tent: Fully Test-Time Adaptation by Entropy Minimization." *ICLR*, 2021.
- Yang, L., et al. "Transformer-Based Source-Free Domain Adaptation." *arXiv*, 2021.
- Xu, T., et al. "CDTrans: Cross-Domain Transformer for Unsupervised Domain Adaptation." *ICLR*, 2022.
