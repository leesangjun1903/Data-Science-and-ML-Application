# Bi-Directional Generation for Unsupervised Domain Adaptation (BDG)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
기존의 단방향(mono-directional) 도메인 적응 방법들은 다음과 같은 한계를 가짐:
- **단방향 생성**: 소스→타겟 방향만 고려하여 타겟 도메인 정보를 충분히 활용하지 못함
- **도메인 레벨 정렬**: 클래스 레벨의 의미론적 구조(class-level semantic structure)를 무시
- **데이터 불균형**: 소스와 타겟 도메인의 샘플 수 불균형 시 성능 저하

BDG는 **양방향 생성(Bi-Directional Generation)** 을 통해 두 개의 중간 도메인(intermediate domain)을 보간(interpolate)하여 도메인 갭을 줄이는 동시에 클래스 레벨 구조를 보존함.

### 주요 기여 (3가지)
1. **양방향 크로스 도메인 생성 모듈**: $G_s: X_s \to X_t$, $G_t: X_t \to X_s$ 두 개의 독립 생성기를 통해 중간 도메인 $F_t$, $F_s$를 합성
2. **이중 일관성 분류기(Dual Consistent Classifiers)**: $C_s$와 $C_t$ 두 분류기의 예측 일관성을 극대화하는 $\mathcal{L}_{con}$ 도입
3. **클래스 수준 MMD 정렬**: 전역(global) MMD와 클래스별(class-wise) MMD를 결합하여 클래스 레벨 의미 정보 보존

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**문제 정의**: 레이블이 있는 소스 도메인 $X_s = \{x_s\}\_{i=1}^{N}$ (레이블 $y_s$ 포함)과 레이블 없는 타겟 도메인 $X_t = \{x_t\}\_{j=1}^{M}$이 주어졌을 때, $p_{data}(x_t) \neq p_{data}(x_s)$ 인 상황에서 타겟 도메인 샘플을 정확히 분류하는 것.

**기존 방법의 문제점**:

| 문제 | 기존 방법 | BDG 해결책 |
|------|-----------|------------|
| 단방향 적응 | DAN, DANN 등 | 양방향 생성기 $G_s$, $G_t$ |
| 도메인 레벨만 정렬 | MMD 기반 방법 | 클래스별 MMD 추가 |
| 분류기 불일치 | MCD의 발산 문제 | $\mathcal{L}_{con}$으로 직접 최소화 |

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) 소스 브랜치 양방향 생성 손실

$$\mathcal{L}^s_{GAN}(X_s) = \mathcal{L}_{dis_s} + \mathcal{L}_{cls_s}$$

$$\mathcal{L}_{dis_s} = \mathbb{E}[\log C_s(X_s)] + \mathbb{E}[\log(1 - C_s(G_s(X_s)))]$$

$$\mathcal{L}_{cls_s} = \mathbb{E}[\log C_s(X_s, Y_s)] + \mathbb{E}[\log C_s(G_s(X_s), Y_s)] \tag{1}$$

- $\mathcal{L}_{dis_s}$: 실제 소스 샘플 $X_s$와 생성된 중간 도메인 $F_t = G_s(X_s)$ 간의 판별 손실
- $\mathcal{L}_{cls_s}$: 소스 분류기 $C_s$의 분류 손실

#### (2) 타겟 브랜치 양방향 생성 손실

$$\mathcal{L}^t_{GAN}(X_s, X_t) = \mathcal{L}_{dis_t} + \mathcal{L}_{cls_t}$$

$$\mathcal{L}_{dis_t} = \mathbb{E}[\log C_t(G_s(X_s))] + \mathbb{E}[\log(1 - C_t(G_t(X_t)))]$$

$$\mathcal{L}_{cls_t} = \mathbb{E}[\log C_t(F_t, Y_s)] + \mathbb{E}[\log C_t(F_s, \hat{Y}_t)] \tag{2}$$

- 타겟 브랜치에서는 $C_t$가 어떤 도메인에서 왔는지를 구별하는 **도메인 혼동(domain confusion)** 역할 수행
- $\hat{Y}_t = C_0(X_t)$: 사전 훈련된 분류기 $C_0$로 생성한 **의사 레이블(pseudo label)**

#### (3) 클래스별 MMD 손실

$$\mathcal{L}^{s/t}_{MMD} = \mathcal{L}^{s/t}_{gMMD} + \frac{1}{C}\mathcal{L}^{s/t}_{cMMD} \tag{3}$$

**전역 MMD** (소스 브랜치):

$$\mathcal{L}^s_{gMMD} = \left\| \frac{1}{n_s}\sum_{x_s \in X_s} G_s(x_s) - \frac{1}{n_t}\sum_{x_t \in X_t} x_t \right\|_2 \tag{4}$$

**클래스별 MMD** (소스 브랜치):

$$\mathcal{L}^s_{cMMD} = \sum_{c}^{C} \left\| \frac{1}{n^c_s}\sum_{x_s \in X^c_s} G_s(x_s) - \frac{1}{n^c_t}\sum_{x_t \in X^c_t} x_t \right\|_2 \tag{5}$$

- $n^c_s$, $n^c_t$: 클래스 $c$에 속하는 소스/타겟 도메인 샘플 수
- 타겟 브랜치에서도 동일한 방식으로 $\mathcal{L}^t_{gMMD}$, $\mathcal{L}^t_{cMMD}$ 계산

#### (4) 이중 일관성 분류기 손실

$$\mathcal{L}_{con} = \| C_t(F_s) - C_s(F_s) \|_1 \tag{6}$$

- $C_s$와 $C_t$가 동일한 입력 $F_s$에 대해 유사한 예측을 내도록 강제
- L1 노름을 통해 두 분류기의 출력 확률 분포 간 차이를 직접 최소화

#### (5) 전체 목적 함수

$$\mathcal{L} = \mathcal{L}^s_{GAN} + \mathcal{L}^t_{GAN} + \lambda(\mathcal{L}^s_{MMD} + \mathcal{L}^t_{MMD}) + \gamma\mathcal{L}_{con} \tag{7}$$

- $\lambda$, $\gamma$: 각 항의 상대적 중요도를 조절하는 하이퍼파라미터 (논문에서는 $\lambda = \gamma = 1$로 설정)

---

### 2.3 모델 구조

```
[소스 도메인 Xs] ──→ [Source Generator Gs] ──→ Ft (중간 도메인)
                                                      ↓
[타겟 도메인 Xt] ──→ [Target Generator Gt] ──→ Fs (중간 도메인)
                                                      ↓
                    Cs (Xs, Ft 입력) ←──── [MMD Module M]
                    Ct (Ft, Fs 입력) ←──── [Consistency Module D]
```

**주요 구성 요소**:

| 구성 요소 | 역할 |
|-----------|------|
| $G_s: X_s \to X_t$ | 소스 샘플을 타겟 분포의 중간 도메인 $F_t$로 변환 |
| $G_t: X_t \to X_s$ | 타겟 샘플을 소스 분포의 중간 도메인 $F_s$로 변환 |
| $C_s$ | $X_s$와 $F_t$를 입력으로 분류 |
| $C_t$ | $F_t$와 $F_s$를 입력으로 분류 |
| $C_0$ | 소스 도메인으로 사전 훈련, 타겟의 의사 레이블 생성 |
| MMD Module | $\mathcal{L}_{MMD}$ 계산 |
| Consistency Module | $\mathcal{L}_{con}$ 계산 |

**백본**: ResNet-50 (ImageNet 사전 훈련), 마지막 FC 레이어 제거 후 파인튜닝

**최적화 3단계**:

- **Step A**: $C_0$ 훈련 → 의사 레이블 $\hat{Y}_t = C_0(X_t)$ 생성

$$\min_{C_0} \mathcal{L}(C_0, X_s) = \mathbb{E}[\log C_0(X_s)] \tag{8}$$

- **Step B**: 생성기 고정 → $C_s$, $C_t$ 훈련

$$\min_{C_s, C_t} \mathcal{L}^s_{GAN} + \mathcal{L}^t_{GAN} + \gamma\mathcal{L}_{con} \tag{9}$$

- **Step C**: 분류기 고정 → 생성기 훈련 (전체 목적 함수 식(7) 사용)

---

### 2.4 성능 향상

**Office-31 데이터셋 결과** (Table 1 기준):

| 방법 | A→W | D→W | W→D | A→D | D→A | W→A | **평균** |
|------|-----|-----|-----|-----|-----|-----|---------|
| SymNets | 90.8 | 98.8 | 100.0 | 93.9 | 74.6 | 72.5 | 88.4 |
| TADA | 94.3 | 98.7 | 99.8 | 91.6 | 72.9 | 73.0 | 88.4 |
| **BDG** | 93.6 | **99.0** | **100.0** | 93.6 | 73.2 | 72.0 | **88.5** |

**Office-Home 데이터셋 결과** (Table 2 기준):

| 방법 | Avg |
|------|-----|
| TADA | 67.6 |
| SymNets | 67.6 |
| **BDG** | **68.7** |

- Cl→Ar (+6.2%), Pr→Ar (+5.4%) 등 어려운 태스크에서 특히 큰 향상
- D→A 태스크에서는 기존 최고 방법(SymNets 74.6%)보다 소폭 낮음 (73.2%)

**절제 연구(Ablation Study)** (Table 3):

| 변형 | Office-31 | Office-Home |
|------|-----------|-------------|
| Variant 1 (단방향, MMD 없음) | 81.1 | 61.0 |
| Variant 2 (단방향 + MMD) | 87.4 | 67.8 |
| Variant 3 (양방향 + 이중 분류기) | 82.9 | 63.2 |
| Variant 4 ( $\mathcal{L}\_{GAN}$ + $\mathcal{L}_{con}$ ) | 82.5 | 63.5 |
| Variant 5 ( $\mathcal{L}\_{GAN}$ + $\mathcal{L}_{MMD}$ ) | 88.3 | 68.2 |
| **BDG (전체)** | **88.5** | **68.7** |

---

### 2.5 한계점

1. **D→A 성능**: 소스(DSLR, 498장)가 타겟(Amazon, 2817장)보다 적을 때 일부 성능이 기존 방법 대비 소폭 낮음
2. **의사 레이블 노이즈**: $C_0$의 성능에 의존하는 의사 레이블의 정확도가 전체 성능에 영향
3. **계산 비용**: 두 개의 독립 생성기와 분류기를 사용하여 파라미터 및 계산량 증가
4. **하이퍼파라미터 민감성**: $\lambda$, $\gamma$ 값에 따라 데이터셋별로 최적 성능이 다르게 나타남
5. **벤치마크 제한**: Office-31, Office-Home에만 평가 → 더 다양한 도메인 시나리오 검증 필요

---

## 3. 모델의 일반화 성능 향상 가능성

BDG의 일반화 성능 향상과 관련된 핵심 메커니즘은 다음과 같음:

### 3.1 양방향 데이터 증강을 통한 일반화

$$F_t = G_s(X_s), \quad F_s = G_t(X_t)$$

두 중간 도메인 $F_t$, $F_s$가 **추가적인 훈련 샘플** 역할을 하여 분류기 훈련에 활용되는 데이터 다양성을 증가시킴. 이는 특히 소스 도메인 샘플 수가 타겟보다 적을 때 효과적임.

### 3.2 클래스 레벨 의미 구조 보존

클래스별 MMD ($\mathcal{L}^{s/t}_{cMMD}$)는 단순히 도메인 전체의 분포를 맞추는 것을 넘어, **각 클래스별로** 소스와 타겟 특징 분포를 정렬함. 이를 통해:
- 클래스 경계(class boundary)가 더 명확하게 유지됨
- 부정적 전이(negative transfer) 방지
- t-SNE 시각화 결과, BDG의 특징이 31개의 명확한 클러스터를 형성함

### 3.3 일관성 손실의 정규화 효과

$$\mathcal{L}_{con} = \| C_t(F_s) - C_s(F_s) \|_1$$

이 손실은 **앙상블 효과**와 유사하게 작동함:
- 서로 다른 도메인 샘플로 훈련된 두 분류기의 예측을 일치시킴
- 훈련 과정에서 발산(divergence) 문제 방지 (MCD와 달리 분류기 적대적 학습 없음)
- Figure 4에서 확인: $\mathcal{L}\_{MMD}$와 $\mathcal{L}_{con}$ 감소와 함께 정확도가 안정적으로 증가

### 3.4 불균형 도메인에서의 일반화

Office-Home처럼 65개 클래스, ~15,500장의 대용량 데이터셋에서:
- 양방향 생성으로 두 도메인 모두의 정보를 충분히 활용
- 클래스별 MMD에서 특정 클래스의 샘플이 비어있는(blank) 매칭 오류 방지
- **6% 이상 향상**: Cl→Ar, Pr→Ar 등 어려운 태스크

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

**① 양방향 생성 패러다임 확립**

BDG는 단방향에서 양방향으로의 전환이 도메인 적응에서 중요함을 실증적으로 보여줌. 이후 연구들이 대칭적/양방향 구조를 더 적극적으로 탐색하는 데 기여함.

**② 클래스 인식 도메인 정렬의 중요성 재확인**

클래스 레벨 MMD의 효과를 ablation study로 명확히 입증함으로써, 이후 연구들이 단순 도메인 레벨 정렬을 넘어 클래스 조건부 정렬을 설계하는 방향을 제시함.

**③ 일관성 기반 정규화의 가능성**

$\mathcal{L}_{con}$의 도입은 이후 연구에서 **상호 학습(mutual learning)**, **교사-학생(teacher-student)** 프레임워크와의 결합 가능성을 시사함.

**④ 의사 레이블 활용의 체계화**

$C_0$를 통한 의사 레이블 생성이 타겟 도메인 정보 활용의 핵심임을 보여주어, 이후 **자기지도학습(self-supervised)** 및 **반지도학습** 기반 DA 연구에 영감을 줌.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래는 BDG 이후 발표된 주요 UDA 연구들과의 비교임. 단, 아래 연구들의 세부 수치는 본 논문(BDG) 원문에 포함되어 있지 않으므로, **제가 직접 알고 있는 범위 내에서만** 기술하며, 불확실한 수치는 제시하지 않겠음.

#### 주요 후속 연구 방향

**① 트랜스포머 기반 도메인 적응**

- **CDTrans** (Xu et al., 2021, ICLR 2022): Vision Transformer를 UDA에 적용. Self-attention을 통한 크로스 도메인 특징 정렬. BDG의 CNN 기반 생성 방식 대비 더 강력한 전역 문맥 모델링 가능.
- **TVT** (Yang et al., 2021): 트랜스포머의 토큰 수준에서 도메인 정렬 수행.

**② 자기지도학습과의 결합**

- **MCC** (Jin et al., 2020, ECCV 2020): Minimum Class Confusion을 통해 타겟 도메인의 클래스 경계를 명확히 함. BDG의 $\mathcal{L}_{con}$과 유사한 동기를 가지나 다른 접근법.

**③ 소스 없는(Source-Free) 도메인 적응**

- BDG는 소스 데이터에 직접 접근해야 하지만, 프라이버시 등의 이유로 소스 데이터에 접근 불가한 시나리오를 다루는 **Source-Free DA** 연구가 2020년 이후 급증함. BDG의 의사 레이블 아이디어가 이 방향으로 확장될 수 있음.

**④ 프롬프트 튜닝 기반 적응**

- **CLIP 기반 방법들** (2022~): 대규모 사전 훈련 모델의 등장으로 도메인 적응의 패러다임 자체가 변화. BDG처럼 생성적 접근이 필요하지 않을 수 있음.

#### BDG와 후속 연구 비교 요약

| 측면 | BDG (2020) | 2020년 이후 연구 동향 |
|------|-----------|----------------------|
| 백본 | ResNet-50 | ViT, CLIP 등 대형 모델 |
| 데이터 증강 | GAN 기반 이미지 생성 | 자기지도학습, 대조학습 |
| 레이블 활용 | 의사 레이블 | 점진적 의사 레이블 정제 |
| 도메인 접근 | 소스+타겟 동시 필요 | Source-Free, Test-Time 적응 |
| 정렬 전략 | MMD + GAN | 최적 운반(Optimal Transport), 대조 손실 |

---

### 4.3 앞으로 연구 시 고려할 점

**① 더 강건한 의사 레이블 전략 필요**

현재 $C_0$의 품질에 의존하는 구조는 소스와 타겟 간 도메인 갭이 클 경우 의사 레이블 오류가 누적될 수 있음. **점진적 자기 훈련(progressive self-training)** 이나 **신뢰도 임계값 기반 필터링** 도입 필요.

**② 트랜스포머와의 결합**

BDG의 양방향 생성 아이디어를 Vision Transformer의 크로스 어텐션 메커니즘과 결합하면 더 세밀한 특징 정렬이 가능할 것으로 기대됨.

**③ Source-Free 시나리오로의 확장**

소스 데이터 없이 저장된 모델만으로 양방향 생성을 구현하는 방법 탐구. 예: 모델 기반 데이터 증류(data distillation)를 활용한 소스 도메인 근사.

**④ 더 다양한 벤치마크에서의 검증**

Office-31, Office-Home 외에 DomainNet(345 클래스), VisDA 등 대규모/다중 도메인 벤치마크에서의 검증이 필요함.

**⑤ 계산 효율성 개선**

두 생성기와 두 분류기를 동시에 최적화하는 구조의 계산 비용을 줄이기 위한 **지식 증류(knowledge distillation)** 또는 **경량화** 연구 필요.

**⑥ 오픈셋(Open-Set) 도메인 적응으로 확장**

타겟 도메인에 소스에 없는 미지(unknown) 클래스가 존재하는 현실적 시나리오에 BDG를 적용하기 위한 이상치 탐지 메커니즘 추가 검토 필요.

---

## 참고 자료

- **본 논문 (주요 출처)**: Guanglei Yang, Haifeng Xia, Mingli Ding, Zhengming Ding. "Bi-Directional Generation for Unsupervised Domain Adaptation." AAAI 2020. arXiv:2002.04869v1.
- Saito et al. "Maximum Classifier Discrepancy for Unsupervised Domain Adaptation." CVPR 2018.
- Zhang et al. "Domain-Symmetric Networks for Adversarial Domain Adaptation." CVPR 2019.
- Wang et al. "Transferable Attention for Domain Adaptation." AAAI 2019.
- Kang et al. "Contrastive Adaptation Network for Unsupervised Domain Adaptation." CVPR 2019.
- Gretton et al. "A Kernel Two-Sample Test." JMLR 2012.
- Zhu et al. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." ICCV 2017.
- Ganin et al. "Domain-Adversarial Training of Neural Networks." JMLR 2016.
- Sankaranarayanan et al. "Generate to Adapt: Aligning Domains using Generative Adversarial Networks." CVPR 2018.

> **⚠️ 주의**: 2020년 이후 최신 연구(CDTrans, TVT, MCC 등)의 구체적 수치 비교는 해당 논문 원문을 직접 확인하지 않았으므로, 수치 없이 연구 방향만 서술하였습니다. 정확한 비교를 위해서는 각 논문 원문을 참조하시기 바랍니다.
