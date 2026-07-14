# Confidence Regularized Self-Training (CRST) 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

기존 Self-Training 기반 비지도 도메인 적응(UDA)에서 사용되는 **하드(hard) 의사 레이블(pseudo-label)** 은 노이즈를 포함할 수 있으며, 이를 ground truth처럼 취급하면 **과잉 확신(overconfident)** 으로 인한 오류 전파(error propagation)가 발생한다. 이를 해결하기 위해 **신뢰도 정규화(Confidence Regularization)** 를 Self-Training에 통합한 CRST 프레임워크를 제안한다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| Continuous CBST 일반화 | 의사 레이블의 feasible space를 one-hot → probability simplex로 확장 |
| CRST-LR (Label Regularization) | 소프트 의사 레이블 생성 (LRENT 정규화기) |
| CRST-MR (Model Regularization) | 출력 평활화 정규화기 도입 (MRL2, MRENT, MRKLD) |
| 이론적 분석 | RCML(Regularized Classification Maximum Likelihood) 및 CEM과의 등가성 증명, 수렴성 증명 |
| 실험적 검증 | VisDA17, Office-31, GTA5→Cityscapes, SYNTHIA→Cityscapes에서 SOTA 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

Self-Training의 핵심 메커니즘은 **엔트로피 최소화(entropy minimization)** 로, 네트워크 출력을 하드 의사 레이블처럼 뾰족하게(sharp) 만드는 것이다. 그러나 다음의 두 가지 문제가 발생한다:

1. **의사 레이블 노이즈**: 예측이 틀렸음에도 100% 확신을 부여 → 오류 누적(error propagation)
2. **레이블 모호성(Label Ambiguity)**: 자연 이미지에서 여러 클래스가 동시에 의미를 가질 수 있는데, 하나의 클래스만을 강제하면 학습 저하

### 2.2 제안 방법 및 수식

#### (a) 기반: Continuous CBST

기존 CBST를 의사 레이블의 feasible space를 확장하여 연속화한다:

$$\min_{\mathbf{w}, \hat{\mathbf{Y}}_T} \mathcal{L}_{CB}(\mathbf{w}, \hat{\mathbf{Y}}) = -\sum_{s \in S}\sum_{k=1}^{K} y_s^{(k)} \log p(k|\mathbf{x}_s; \mathbf{w}) - \sum_{t \in T}\sum_{k=1}^{K} \hat{y}_t^{(k)} \log \frac{p(k|\mathbf{x}_t; \mathbf{w})}{\lambda_k}$$

$$\text{s.t.} \quad \hat{y}_t \in \Delta^{K-1} \cup \{\mathbf{0}\}, \quad \forall t $$

의사 레이블 생성의 closed-form 해:

```math
\hat{y}_t^{(k)*} = \begin{cases} 1, & \text{if } k = \arg\max_c \left\{\frac{p(c|\mathbf{x}_t; \mathbf{w})}{\lambda_c}\right\} \text{ and } p(k|\mathbf{x}_t; \mathbf{w}) > \lambda_k \\ 0, & \text{otherwise} \end{cases}
```

#### (b) CRST 일반 형태

$$\min_{\mathbf{w}, \hat{\mathbf{Y}}_T} \mathcal{L}_{CR}(\mathbf{w}, \hat{\mathbf{Y}}_T) = \mathcal{L}_{CB}(\mathbf{w}, \hat{\mathbf{Y}}_T) + \alpha \mathcal{R}_C(\mathbf{w}, \hat{\mathbf{Y}}_T)$$

$$= -\sum_{s \in S}\sum_{k=1}^{K} y_s^{(k)} \log p(k|\mathbf{x}_s; \mathbf{w}) - \sum_{t \in T}\left[\sum_{k=1}^{K} \hat{y}_t^{(k)} \log \frac{p(k|\mathbf{x}_t; \mathbf{w})}{\lambda_k} - \alpha r_c(\mathbf{w}, \hat{y}_t)\right]$$

$$\text{s.t.} \quad \hat{y}_t \in \Delta^{(K-1)} \cup \{\mathbf{0}\}, \quad \forall t $$

여기서 $\mathcal{R}\_C(\mathbf{w}, \hat{\mathbf{Y}}\_T) = \sum_{t \in T} r_c(\mathbf{w}, \hat{y}_t)$ 이고 $\alpha \geq 0$은 정규화 가중치이다.

#### (c) CRST-LR: 레이블 정규화 (LRENT)

의사 레이블 생성 시 엔트로피 정규화를 추가:

$$\min_{\hat{\mathbf{Y}}_T} -\sum_{t \in T}\left[\sum_{k=1}^{K} \hat{y}_t^{(k)} \log \frac{p(k|\mathbf{x}_t; \mathbf{w})}{\lambda_k} - \alpha r_c(\hat{y}_t)\right] \quad \text{s.t.} \quad \hat{y}_t \in \Delta^{(K-1)} \cup \{\mathbf{0}\} $$

LRENT 정규화기의 closed-form 소프트 의사 레이블 해 (KKT 조건 이용):

$$\hat{y}_t^{(i)\dagger} = \frac{\left(\frac{p(i|\mathbf{x}_t)}{\lambda_k}\right)^{\frac{1}{\alpha}}}{\sum_{k=1}^{K}\left(\frac{p(k|\mathbf{x}_t)}{\lambda_k}\right)^{\frac{1}{\alpha}}} $$

이는 **온도를 가진 Softmax(Softmax with temperature)** 와 동일함을 증명한다:

$$p(i) = \frac{e^{z_i/\alpha}}{\sum_{k=1}^{K} e^{z_k/\alpha}} $$

- $\alpha \to \infty$: 균등 분포(최대 불확실성)
- $\alpha = 1$: 원래 softmax
- $\alpha \to 0$: one-hot 벡터

#### (d) CRST-MR: 모델 정규화

네트워크 재학습 시 출력 평활화 항을 추가:

$$\min_{\mathbf{w}} -\sum_{s \in S}\sum_{k=1}^{K} y_s^{(k)} \log p(k|\mathbf{x}_s; \mathbf{w}) - \sum_{t \in T}\left[\sum_{k=1}^{K} \hat{y}_t^{(k)} \log p(k|\mathbf{x}_t; \mathbf{w}) - \alpha r_c(p(\mathbf{x}_t; \mathbf{w}))\right] $$

세 가지 모델 정규화기:

| 정규화기 | 정의 | Softmax logit $z_i$에 대한 기울기 |
|----------|------|-------------------------------|
| **MRL2** | $\sum_{k=1}^{K} p(k\|\mathbf{x}_t)^2$ | $2\sum\_{k=1}^{K} p^2(k\|\mathbf{x}\_t)[\delta_{ki} - p(i\|\mathbf{x}_t)]$ |
| **MRENT** | $\sum_{k=1}^{K} p(k\|\mathbf{x}_t)\log p(k\|\mathbf{x}_t)$ | $p(i\|\mathbf{x}\_t)[\log p(i\|\mathbf{x}_t) + H(p(\mathbf{x}_t))]$ |
| **MRKLD** | $-\sum_{k=1}^{K} \frac{1}{K}\log p(k\|\mathbf{x}_t)$ | $p(i\|\mathbf{x}_t) - \frac{1}{K}$ |

**MRKLD의 닫힌 형태 최소화 해**:

$$p^{*(k)} = \frac{y^{(k)} + \frac{\alpha}{K}}{1 + \alpha}$$

즉, 음성 클래스에 대해 균등하게 확률을 분배하는 **레이블 평활화(label smoothing)** 와 동일하다.

**Proposition 4**: MRKLD 정규화 Self-Training은 $\epsilon = \frac{K\alpha - \alpha}{K + K\alpha}$로 균등 평활화된 pseudo-label Self-Training과 동치이다.

### 2.3 모델 구조

CRST는 특정 백본 네트워크를 새로 제안하지 않고, 기존 모델 위에 **정규화된 Self-Training 알고리즘 프레임워크**를 적용한다.

**학습 파이프라인 (교대 최적화)**:

```
Round 1, 2, 3, ...
├── Step a) 의사 레이블 생성 (Pseudo-label Generation)
│   ├── CRST-LR: 소프트 의사 레이블 생성 (LRENT 기반)
│   └── CRST-MR: 하드 의사 레이블 생성 (CBST와 동일)
└── Step b) 네트워크 재학습 (Network Retraining)
    ├── CRST-LR: 소프트 의사 레이블로 Cross-Entropy 학습
    └── CRST-MR: 하드 의사 레이블 + 출력 평활화 정규화기
```

**백본 구성**:
- 이미지 분류: ResNet-101 (VisDA17), ResNet-50 (Office-31)
- 시맨틱 분할: DeepLabv2, Wide ResNet-38

### 2.4 성능 향상 및 한계

#### 성능 향상

**VisDA17 (Synthetic→Real 이미지 분류)**:

| 방법 | Mean Accuracy |
|------|--------------|
| CBST | 76.4 ± 0.9 |
| MRKLD | 77.9 ± 0.5 |
| LRENT | 76.6 ± 0.9 |
| **MRKLD+LRENT** | **78.1 ± 0.2** |
| SimNet-Res152 | 72.9 (더 강력한 백본) |

**GTA5→Cityscapes (DeepLabv2)**:

| 방법 | mIoU |
|------|------|
| CBST | 45.9 |
| MRKLD | **47.1** |
| MRKLD-SP-MST (ResNet-38) | **49.8** |

**신뢰도 분석 ($C_{TP}/C_{FP}$ 비율)**:

MRKLD와 LRENT는 CBST 대비 False Positive의 신뢰도를 효과적으로 낮추면서 True Positive 대비 False Positive의 비율($C_{TP}/C_{FP}$)을 개선한다:

| 방법 | Mean $C_{TP}/C_{FP}$ |
|------|---------------------|
| CBST | 1.19 |
| MRKLD | **1.27** |
| LRENT | 1.25 |

#### 한계점

1. **LR의 저장 비용**: CRST-LR은 데이터셋 수준의 소프트 의사 레이블을 저장해야 하므로, 시맨틱 분할과 같이 레이블이 큰 경우 추가적인 I/O 비용 발생
2. **하이퍼파라미터 민감도**: $\alpha$ (정규화 가중치), $p$ (선택 비율) 등 여러 하이퍼파라미터 조정 필요 (MR+LR 결합 시 더욱 복잡)
3. **일부 정규화기의 성능 저하**: MRL2, MRENT는 일부 태스크에서 CBST보다 낮거나 유사한 성능
4. **순수 도메인 적응에만 검증**: 다른 분포 이동 시나리오(예: 연속 도메인 이동)에서의 효과 불확실
5. **클래스 순위 정보 손실**: MRKLD 계열 MR은 음성 클래스에 균등 확률을 부여하여 클래스 간 순위 정보가 손실됨

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 과잉 엔트로피 최소화 방지를 통한 일반화

Self-Training의 핵심 문제는 **무한 엔트로피 최소화(infinite entropy minimization)** 로, 모델이 목표 도메인의 특정 클래스에 과잉 확신을 가지게 된다. CRST는 이를 정규화 항을 통해 방지한다.

$$\mathcal{L}_{CR} = \mathcal{L}_{CB} + \alpha \mathcal{R}_C$$

정규화 항 $\mathcal{R}_C$는 모델 출력의 엔트로피를 일정 수준 유지시켜 **과적합을 방지**하고 일반화 성능을 향상시킨다.

### 3.2 소프트 의사 레이블의 정보 이론적 관점

LRENT 소프트 의사 레이블은 온도 $\alpha$를 통해 정보를 여러 클래스에 분산시킨다:

$$\hat{y}_t^{(i)\dagger} = \frac{\left(\frac{p(i|\mathbf{x}_t)}{\lambda_k}\right)^{1/\alpha}}{\sum_{k=1}^{K}\left(\frac{p(k|\mathbf{x}_t)}{\lambda_k}\right)^{1/\alpha}}$$

- $\alpha > 1$: 분포를 부드럽게 → 레이블 모호성 처리 가능
- $\alpha < 1$: 분포를 날카롭게 → 하드 레이블 방향

**소프트 레이블의 일반화 기여**: 잘못된 의사 레이블의 손실 기여를 줄이고, 경계 샘플(boundary sample)에서의 학습 신호를 완화함으로써 과적합을 방지한다.

### 3.3 Confusion Matrix 분석을 통한 일반화 확인

논문의 혼동 행렬(Figure 8) 분석에서 MRKLD+LRENT는 유사 클래스 간의 혼동(예: "person vs. horse", "motor vs. bike")을 효과적으로 감소시켜, 더 나은 도메인 일반화를 보여준다.

### 3.4 특징 공간 정렬과 일반화

Figure 7의 t-SNE 시각화에서 MRKLD+LRENT는 CBST 대비 **더 명확한 클래스별 클러스터 구분**을 보이며, 이는 정규화가 특징 공간의 도메인 불변 표현 학습에 기여함을 의미한다.

### 3.5 RCML 관점에서의 이론적 일반화 보장

CRST는 **정규화된 분류 최대 우도(Regularized Classification Maximum Likelihood, RCML)** 로 해석된다:

$$\max_{\mathbf{w}, \hat{\mathbf{Y}}_T} \log \tilde{\mathcal{L}}_C + \mathcal{R}_C$$

정규화 항 $\mathcal{R}_C$는 기존 통계 학습 이론에서의 정규화와 동일한 역할을 하며, **VC 차원 기반의 일반화 오차 경계**를 낮추는 효과를 기대할 수 있다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려사항

### 4.1 연구에 미치는 영향

#### (a) UDA/SSL 분야에서의 파급 효과

CRST는 Self-Training과 정규화를 체계적으로 연결한 최초의 프레임워크 중 하나로, 다음 연구들에 직접적 영향을 미쳤다:

- **노이즈 레이블 학습(Noisy Label Learning)**: 소프트 의사 레이블 아이디어는 노이즈 레이블 연구에서 활발히 활용
- **Teacher-Student 기반 방법**: Mean Teacher와의 통합 가능성을 제시
- **지식 증류(Knowledge Distillation)**: 온도 스케일링과 소프트 레이블의 연결은 증류 기반 UDA 연구의 이론적 기반 제공

#### (b) 이론적 기여의 영향

CEM(Classification Expectation Maximization)과의 연결은 Self-Training을 확률론적 EM 알고리즘으로 해석하는 후속 연구의 기반이 됨.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래는 제공된 논문(CRST, arXiv:1908.09822, 2020)의 내용과 일반적으로 알려진 관련 연구 방향을 비교한 것이다. **단, 2020년 이후 개별 논문의 구체적 수치나 방법은 해당 논문을 직접 확인하지 못했으므로, 연구 방향 수준에서만 기술한다.**

| 연구 방향 | 대표 방법 | CRST와의 비교 |
|-----------|-----------|--------------|
| **Noisy Pseudo-label 처리** | DivideMix (Li et al., 2020), SHOT (Liang et al., 2020) | CRST는 정규화로 접근; 이후 연구는 노이즈 분리 모델 명시적 사용 |
| **Teacher-Student UDA** | Mean Teacher + UDA 통합 연구들 | CRST는 단일 네트워크; Teacher-Student는 일관성 정규화 추가 |
| **Source-free DA** | SHOT, 3C-GAN 계열 | CRST는 소스 데이터 접근 가정; Source-free는 더 실용적 설정 |
| **Transformer 기반 UDA** | CDTrans, SWD+ViT 계열 | CRST는 CNN 기반; Transformer의 강력한 특징 추출 활용 미흡 |
| **Semi-supervised + DA 통합** | FlexMatch, FreeMatch | CRST의 임계값($\lambda_k$) 자동화 아이디어와 유사 |

### 4.3 앞으로 연구 시 고려해야 할 사항

#### ① 동적 정규화 가중치 ($\alpha$) 조정

현재 CRST는 고정된 $\alpha$ 값을 사용하지만, Self-Training 라운드가 진행될수록 의사 레이블의 품질이 향상되므로 **$\alpha$를 동적으로 감소**시키는 스케줄링 전략이 필요하다:

$$\alpha_t = \alpha_0 \cdot \gamma^t, \quad \gamma < 1$$

#### ② Source-Free 설정으로의 확장

실제 응용에서는 소스 데이터에 접근하기 어려운 경우가 많다. CRST의 정규화 아이디어를 **소스 데이터 없이** 적용하는 연구가 필요하다.

#### ③ Transformer 기반 백본과의 통합

Vision Transformer(ViT)는 CNN보다 강력한 도메인 불변 특징을 학습하는 경향이 있으나, Self-Training과의 결합에서 정규화 전략이 달라질 수 있다.

#### ④ 다중 소스 도메인 적응으로의 확장

CRST는 단일 소스-타겟 쌍만 고려한다. **다중 소스 도메인(Multi-source DA)** 에서의 클래스 균형 전략과 정규화 방법 설계가 필요하다.

#### ⑤ 의사 레이블 품질 평가 메트릭 개발

현재는 $C_{TP}/C_{FP}$ 비율을 사용하지만, 더 정교한 **의사 레이블 신뢰도 평가 지표** 개발이 필요하다.

#### ⑥ 클래스 불균형(Class Imbalance) 문제

$\lambda_k$의 클래스 균형 전략은 클래스 불균형을 일부 완화하지만, **롱테일 분포(long-tail distribution)** 에서의 성능은 별도 연구가 필요하다.

#### ⑦ 불확실성 정량화(Uncertainty Quantification)와의 통합

베이지안 딥러닝 기반의 불확실성 추정과 CRST를 결합하면, 더 정교한 의사 레이블 선택 전략을 설계할 수 있다:

$$\hat{y}_t^* \text{ 선택 기준} = f\left(\frac{p(k|\mathbf{x}_t)}{\lambda_k}, \sigma^2(k|\mathbf{x}_t)\right)$$

---

## 참고 자료

- **주 논문**: Yang Zou, Zhiding Yu, Xiaofeng Liu, B.V.K. Vijaya Kumar, Jinsong Wang. "Confidence Regularized Self-Training." arXiv:1908.09822v3 [cs.CV], 15 Jul 2020. (ICCV 2019 게재)
- **GitHub 코드**: https://github.com/yzou2/CRST
- **기반 논문**: Yang Zou et al. "Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training." ECCV 2018. (CBST)
- **참조 이론**: Amini & Gallinari. "Semi-supervised Logistic Regression." ECAI 2002. (CEM 기반)
- **연결 이론**: Hinton et al. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531 (Knowledge Distillation/Temperature Softmax)
- **비교 방법**: Tsai et al. "Learning to Adapt Structured Output Space for Semantic Segmentation." CVPR 2018. (AdaptSegNet); Vu et al. "AdvEnt: Adversarial Entropy Minimization." CVPR 2019.

> **주의**: 2020년 이후 최신 연구와의 구체적 수치 비교는 해당 개별 논문을 직접 확인하지 못하였으므로, 연구 방향 수준에서만 기술하였습니다.
