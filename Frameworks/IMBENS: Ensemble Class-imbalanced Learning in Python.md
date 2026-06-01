# IMBENS: Ensemble Class-imbalanced Learning in Python 

---

## ⚠️ 사전 고지

본 논문(arXiv:2111.12776v2)은 **소프트웨어 툴박스 논문**으로, 새로운 알고리즘이나 수식을 제안하는 연구 논문이 아닙니다. 따라서 **독자적인 핵심 수식이나 새로운 모델 아키텍처를 직접 제시하지 않습니다.** 수식 관련 내용은 imbens가 구현한 기존 알고리즘들의 원리를 바탕으로 설명하며, 이를 명확히 구분하겠습니다.

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

> **"앙상블 학습(Ensemble Learning)은 클래스 불균형 문제를 해결하는 가장 강력한 접근법 중 하나이지만, 이를 통합적으로 지원하는 표준 오픈소스 패키지가 없다."**

기존 패키지인 `imbalanced-learn`(imblearn)과 `smote-variants`는 단순 리샘플링/재가중치 방법에 집중하고 있어, **앙상블 불균형 학습(EIL: Ensemble Imbalanced Learning)** 전반을 커버하지 못합니다. imbens는 이 공백을 채우는 것을 목표로 합니다.

### 1.2 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **통합 구현** | 14개의 EIL 알고리즘을 단일 API로 구현 |
| **확장 기능** | 커스터마이즈 가능한 리샘플링 스케줄러, verbose 로깅 |
| **scikit-learn 호환** | `fit`, `predict`, `predict_proba` 인터페이스 준수 |
| **시각화 도구** | `ImbalancedEnsembleVisualizer` 제공 |
| **확장 용이성** | 상속/다형성 기반 설계로 새 알고리즘 추가 용이 |
| **품질 보증** | 96% 코드 커버리지, CircleCI 통합, PEP8 준수 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

#### 클래스 불균형 문제 (Class Imbalance Problem)

클래스 불균형은 분류 문제에서 각 클래스의 샘플 수가 현저히 다른 상황을 의미합니다. 불균형 비율(Imbalance Ratio, IR)은 다음과 같이 정의됩니다:

$$IR = \frac{|S_{maj}|}{|S_{min}|}$$

여기서 $|S_{maj}|$는 다수 클래스 샘플 수, $|S_{min}|$은 소수 클래스 샘플 수입니다.

표준 머신러닝 알고리즘은 **전체 정확도(Global Accuracy)**를 최적화하도록 설계되어 있어:

$$\text{Global Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

이는 소수 클래스를 무시하는 편향된 모델을 생성합니다. 불균형 학습에서는 아래와 같은 지표가 더 적합합니다:

$$\text{Balanced Accuracy} = \frac{1}{K}\sum_{k=1}^{K} \frac{\text{올바르게 분류된 클래스 } k \text{ 샘플 수}}{|\text{클래스 } k \text{ 전체 샘플}|}$$

$$\text{Macro F1} = \frac{1}{K}\sum_{k=1}^{K} \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$

$$\text{G-mean} = \left(\prod_{k=1}^{K} \text{Recall}_k\right)^{1/K}$$

#### 기존 패키지의 한계

```
imbalanced-learn (imblearn)    → 단순 리샘플링 위주, EIL 지원 부족
smote-variants                 → SMOTE 변형에만 특화
개별 논문 구현                  → 비표준화, 재현성 부족
```

### 2.2 제안하는 방법 (구현된 알고리즘 원리 포함)

imbens는 새로운 알고리즘을 제안하지 않고, **기존 EIL 알고리즘 14개를 표준화하여 구현**합니다. 핵심 알고리즘들의 원리는 다음과 같습니다:

#### (A) 언더샘플링 기반 앙상블

**SelfPacedEnsemble (SPE, Liu et al., 2020)**:

훈련 데이터를 난이도 기반으로 구간화(binning)하여 불균형 처리:

$$P(\text{샘플 } x_i \text{ 선택}) \propto \frac{1}{\hat{f}(x_i)}$$

여기서 $\hat{f}(x_i)$는 현재 앙상블이 예측한 $x_i$의 확신도(confidence)입니다. 즉, 앙상블이 쉽게 분류하는 샘플은 덜 선택되고, 어려운 샘플은 더 많이 선택됩니다.

**EasyEnsemble (Liu et al., 2009)**:

다수 클래스를 $T$번 독립적으로 언더샘플링하여 $T$개의 균형 서브셋 생성:

$$\mathcal{D}_{t} = S_{min} \cup \tilde{S}_{maj}^{(t)}, \quad |\tilde{S}_{maj}^{(t)}| = |S_{min}|, \quad t = 1, \ldots, T$$

최종 예측: $H(x) = \arg\max_c \sum_{t=1}^{T} \mathbb{1}[h_t(x) = c]$

**BalancedRandomForest (Chen et al., 2004)**:

각 트리 학습 시 부트스트랩 샘플링 단계에서 소수 클래스와 동일한 수의 다수 클래스 샘플을 추출:

$$\tilde{\mathcal{D}}_{boot}^{(t)} = \text{Bootstrap}(S_{min}) \cup \text{Undersample}(S_{maj}, |S_{min}|)$$

#### (B) 오버샘플링 기반 앙상블

**SmoteBoost (Chawla et al., 2003)**:

AdaBoost의 각 반복(iteration) $t$에서 SMOTE를 적용하여 소수 클래스 오버샘플링:

SMOTE 합성 샘플 생성:

$$x_{syn} = x_i + \lambda \cdot (x_{nn} - x_i), \quad \lambda \sim \mathcal{U}(0, 1)$$

여기서 $x_{nn}$은 $x_i$의 $k$-최근접 이웃 중 소수 클래스 샘플입니다.

AdaBoost 가중치 업데이트:

$$w_i^{(t+1)} = w_i^{(t)} \cdot \exp\left(-\alpha_t \cdot y_i \cdot h_t(x_i)\right)$$

$$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$

#### (C) 재가중치(Cost-Sensitive) 기반 앙상블

**AdaCost (Fan et al., 1999)**:

오분류 비용 행렬 $C$를 도입하여 부스팅 가중치 조정:

$$w_i^{(t+1)} = w_i^{(t)} \cdot \exp\left(-\alpha_t \cdot y_i \cdot h_t(x_i) \cdot \beta(c_i)\right)$$

여기서 $c_i$는 샘플 $x_i$의 오분류 비용, $\beta(\cdot)$는 비용 조정 함수입니다:

$$\beta(c_i) = \begin{cases} c_i & \text{if } h_t(x_i) \neq y_i \\ 0.5 & \text{otherwise} \end{cases}$$

오분류 비용 행렬의 일반적 형태:

$$C = \begin{pmatrix} 0 & C_{FP} \\ C_{FN} & 0 \end{pmatrix}$$

소수 클래스에 대해 $C_{FN} \gg C_{FP}$로 설정하여 소수 클래스 오분류에 더 큰 패널티를 부여합니다.

**RusBoost (Seiffert et al., 2010)**:

AdaBoost의 각 반복에서 랜덤 언더샘플링(RUS)을 적용:

$$\mathcal{D}_t = \text{RUS}(\mathcal{D}, w^{(t)}) = \{(x_i, y_i) : y_i \in S_{min}\} \cup \text{Random}(\{(x_i, y_i) : y_i \in S_{maj}\}, w^{(t)})$$

#### (D) 커스터마이즈 가능한 리샘플링 스케줄러 (imbens 신기능)

imbens의 핵심 확장 기능 중 하나인 `balancing_schedule`은 훈련 반복에 따라 리샘플링 비율을 동적으로 조정합니다:

$$n_{target}^{(t)} = \text{schedule}(t, T, n_{min}, n_{maj})$$

예를 들어 선형 스케줄러:

$$n_{target}^{(t)} = n_{min} + \frac{t}{T-1}(n_{maj} - n_{min})$$

초기 반복에는 강한 균형화, 후반 반복에는 원본 분포에 가깝게 학습하는 전략을 구현할 수 있습니다.

### 2.3 모델 구조

```
imbens 패키지 구조
├── imbens.ensemble/
│   ├── BaseEnsemble (기반 클래스)
│   │   ├── ResampleBoostClassifier (리샘플링+부스팅)
│   │   │   ├── SelfPacedEnsembleClassifier
│   │   │   ├── RusBoostClassifier
│   │   │   ├── SmoteBoostClassifier
│   │   │   └── ...
│   │   ├── ResampleBaggingClassifier (리샘플링+배깅)
│   │   │   ├── BalancedRandomForestClassifier
│   │   │   ├── EasyEnsembleClassifier
│   │   │   └── ...
│   │   └── ReweightBoostClassifier (재가중치+부스팅)
│   │       ├── AdaCostClassifier
│   │       └── ...
├── imbens.samplers/       (리샘플링 모듈 - imblearn 연동)
├── imbens.datasets/       (합성 불균형 데이터 생성)
├── imbens.utils/          (evaluate_print 등 유틸리티)
└── imbens.visualizer/     (ImbalancedEnsembleVisualizer)
```

**통합 인터페이스:**
```python
clf = ensemble.XxxClassifier(
    estimator=DecisionTreeClassifier(),  # base learner
    n_estimators=10                       # 앙상블 크기
)
clf.fit(X_train, y_train,
    balancing_schedule='uniform',         # 리샘플링 스케줄러
    eval_datasets={'val': (X_val, y_val)}, # 검증 데이터
    eval_metrics={'balanced_acc': ...},    # 평가 지표
    train_verbose=True                     # 로깅
)
```

### 2.4 성능 향상

논문에서 데모로 제시한 SelfPacedEnsemble의 결과:

```
SPE balanced Acc: 0.972 | macro Fscore: 0.886 | macro Gmean: 0.972
(데이터: 200샘플, 불균형 비율 9:1)
```

논문이 **소프트웨어 논문**이므로 체계적인 벤치마크 실험 결과를 직접 제시하지는 않습니다. 그러나 각 알고리즘의 원 논문에서 보고된 성능 개선은 다음과 같습니다:

| 알고리즘 | 원 논문 성능 개선 (요약) |
|----------|--------------------------|
| SelfPacedEnsemble | EasyEnsemble 대비 G-mean 평균 3-5% 향상 (Liu et al., 2020) |
| EasyEnsemble | 단순 AdaBoost 대비 AUC 유의미한 향상 (Liu et al., 2009) |
| BalancedRandomForest | 표준 RF 대비 소수 클래스 Recall 대폭 향상 (Chen et al., 2004) |
| SmoteBoost | AdaBoost 단독 대비 소수 클래스 F1 향상 (Chawla et al., 2003) |

### 2.5 한계

| 한계 | 설명 |
|------|------|
| **소프트웨어 논문의 한계** | 새로운 알고리즘 제안 없음, 체계적 비교 실험 부재 |
| **구현 범위** | v0.2.0 기준 14개 알고리즘, 진화 알고리즘/메타러닝/하이브리드 기반 EIL 미포함 |
| **딥러닝 미지원** | scikit-learn 스타일에 집중, 딥러닝 기반 불균형 학습 제외 |
| **다중 레이블 미지원** | 이진/다중 클래스에 집중, 다중 레이블 불균형 미지원 |
| **이론적 분석 부재** | 각 알고리즘의 수렴 보장, 편향-분산 트레이드오프 분석 없음 |
| **대규모 데이터 확장성** | 분산 처리 미지원 |

---

## 3. 모델의 일반화 성능 향상 가능성

imbens가 일반화 성능 향상에 기여하는 메커니즘을 중점적으로 분석합니다.

### 3.1 앙상블 학습의 편향-분산 분해

앙상블 예측의 기대 오차는 다음과 같이 분해됩니다:

$$\mathbb{E}[(y - H(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

$$\text{Variance}(H) = \frac{1}{T^2}\sum_{t=1}^{T}\text{Var}(h_t) + \frac{2}{T^2}\sum_{t < t'}\text{Cov}(h_t, h_{t'})$$

앙상블은 다수의 **다양한(diverse)** base learner를 결합하여 분산을 줄입니다. 불균형 학습에서는 각 반복마다 **다른 서브셋**을 사용하여 다양성을 확보합니다.

### 3.2 클래스 불균형이 일반화에 미치는 영향

표준 ERM(Empirical Risk Minimization)은:

$$\hat{\theta} = \arg\min_\theta \frac{1}{N}\sum_{i=1}^{N} \ell(h_\theta(x_i), y_i)$$

불균형 상황에서 다수 클래스가 손실을 지배하여 소수 클래스에 대한 일반화 실패. 이를 해결하기 위한 **재가중치 손실함수**:

$$\hat{\theta} = \arg\min_\theta \frac{1}{N}\sum_{i=1}^{N} w_i \cdot \ell(h_\theta(x_i), y_i)$$

여기서 $w_i = \frac{1/|S_k|}{1/K \sum_{k'} 1/|S_{k'}|}$ (클래스 $k$에 속한 샘플의 가중치)

### 3.3 일반화 성능 향상 메커니즘

#### (1) 커스터마이즈 리샘플링 스케줄러
훈련 초반에 강한 균형화(높은 일반화 bias 감소)로 소수 클래스 특성을 학습하고, 후반에 점진적으로 실제 분포에 가깝게 조정:

$$\text{Schedule}(t) = \left\lfloor n_{min} + \left(\frac{t}{T}\right)^\gamma (n_{maj} - n_{min}) \right\rfloor, \quad \gamma > 0$$

$\gamma$를 조정하여 스케줄링 곡선 제어 가능. 이는 커리큘럼 학습(Curriculum Learning)과 유사한 효과를 줍니다.

#### (2) SelfPacedEnsemble의 난이도 기반 샘플링

경계 영역(decision boundary) 근방의 어려운 샘플에 집중함으로써, 모델이 **판별력 있는 특징(discriminative features)**을 학습:

$$\mathcal{L}_{SPE} = \mathbb{E}_{x \sim P_{hard}}[\ell(h(x), y)]$$

여기서 $P_{hard}$는 현재 앙상블의 불확실성이 높은 샘플에 더 높은 확률을 부여하는 분포입니다.

#### (3) 앙상블의 다양성을 통한 과적합 방지

다양한 리샘플링 서브셋으로 학습된 base learner들 간의 낮은 상관관계:

$$\text{Corr}(h_t, h_{t'}) \approx 0 \Rightarrow \text{Var}(H) \approx \frac{1}{T}\bar{\sigma}^2$$

즉, 앙상블 크기 $T$가 증가할수록 분산이 $1/T$로 감소하여 일반화 성능 향상.

#### (4) 다중 평가 지표를 통한 모델 선택

```python
eval_metrics={
    'balanced_acc': (balanced_accuracy_score, {}),
    'macro_f1': (f1_score, {'average': 'macro'}),
    'g_mean': (geometric_mean_score, {})
}
```

훈련 중 다양한 불균형 특화 지표로 모니터링하여, **전체 정확도**에 과적합되지 않는 모델 선택 가능.

### 3.4 일반화 성능의 이론적 보장 (한계)

imbens 논문 자체는 이론적 분석을 제공하지 않습니다. 관련 이론적 분석은 원 알고리즘 논문에서 찾아야 하며, 예를 들어:

**AdaBoost의 일반화 오차 경계** (원 AdaBoost 이론):

$$\mathbb{P}_{(x,y)\sim D}[H(x) \neq y] \leq \exp\left(-2\sum_{t=1}^{T}(\gamma_t)^2\right)$$

여기서 $\gamma_t = \frac{1}{2} - \epsilon_t$는 $t$번째 weak learner의 edge입니다. 그러나 이는 균형 분포 가정 하의 결과이며, 불균형 상황에서는 직접 적용에 주의가 필요합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (A) 재현성 및 공정한 비교 기반 마련
표준화된 API로 인해 새로운 EIL 알고리즘 제안 시 공정한 기준선(baseline) 비교가 용이해집니다. 연구자들이 자신의 알고리즘을 imbens의 14개 메서드와 즉시 비교할 수 있습니다.

#### (B) 알고리즘 결합 연구 가속화
상속/다형성 기반 설계로 새로운 리샘플러와 앙상블 방식의 결합 실험이 간소화됩니다:
```python
# 새로운 리샘플러를 기존 앙상블 프레임워크에 결합
class MyNewSampler(BaseSampler): ...
class MyNewEnsemble(ResampleBoostClassifier):
    def __init__(self):
        super().__init__(sampler=MyNewSampler())
```

#### (C) AutoML과 연계
하이퍼파라미터 최적화(HPO) 프레임워크와 결합하여 EIL 알고리즘 자동 선택 연구 가능성:
- 리샘플링 전략 $\in$ {US, OS, RW}
- 앙상블 방식 $\in$ {Boosting, Bagging}
- 스케줄러 파라미터 $\gamma$ 자동 최적화

#### (D) 도메인 특화 응용 연구
의료 진단, 사기 탐지, 이상 탐지 등 실제 불균형 문제에 EIL을 적용하는 응용 연구의 진입 장벽 낮춤.

### 4.2 앞으로 연구 시 고려할 점

| 고려 사항 | 설명 | 권장 방향 |
|-----------|------|-----------|
| **딥러닝 통합** | imbens는 scikit-learn 기반으로 신경망 적용 제한 | PyTorch/TensorFlow와의 연동 브릿지 개발 |
| **스트리밍 데이터** | 정적 데이터셋 가정, 온라인 학습 미지원 | Online EIL 알고리즘 연구 필요 |
| **다중 레이블 불균형** | 이진/다중 클래스에 한정 | 다중 레이블 EIL 확장 연구 |
| **설명 가능성(XAI)** | 앙상블의 블랙박스 특성 | SHAP, LIME과의 통합으로 해석 가능성 제공 |
| **데이터 증강과의 결합** | 현재 SMOTE 계열에 한정 | GAN 기반 오버샘플링(e.g., CTGAN)과 결합 |
| **비용 행렬 자동 설정** | 현재 수동 설정 필요 | 도메인 지식 기반 자동 비용 추정 연구 |
| **분산 처리** | 단일 머신 기반 | Spark/Dask 기반 대규모 EIL 연구 |
| **클래스 불균형의 동적 변화** | 정적 불균형 가정 | Concept drift 환경의 EIL 연구 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

⚠️ **주의:** 아래 연구들은 본 논문 제출 이후(2021~2023) 관련 분야의 주요 연구들을 제 학습 데이터 기준으로 제시합니다. 일부 세부 수치는 확인이 필요할 수 있습니다.

### 5.1 주요 최신 연구 비교

| 연구 | 방법 | imbens 대비 차별점 |
|------|------|--------------------|
| **MESA (Liu et al., 2021)** | Meta-sampler + SPE | 메타러닝으로 최적 리샘플링 전략 자동 학습 |
| **KRNN (Zhang et al., 2022)** | K-nearest neighbor 기반 과대표집 + 앙상블 | 지역 불균형 밀도 고려 |
| **DIVE (He et al., 2021)** | 다양성 강화 앙상블 | base learner 다양성 명시적 최대화 |
| **ImGAGN (Ding et al., 2021)** | GAN 기반 그래프 불균형 학습 | 그래프 구조 데이터의 불균형 처리 |
| **SMOTE-NC 확장** | 범주형+연속형 혼합 데이터 | imbens가 지원하지 않는 혼합 타입 처리 |

### 5.2 연구 트렌드 분석

```
2020년 이후 EIL 연구 트렌드

[1] 딥러닝 기반 EIL
    └─ MixUp, CutMix를 불균형 학습에 적용
    └─ GAN 기반 소수 클래스 합성

[2] 메타러닝/AutoML 기반 전략 선택
    └─ MESA: 과거 학습 경험으로 리샘플링 전략 학습

[3] 그래프/비정형 데이터의 불균형
    └─ 노드 분류의 클래스 불균형 (GraphSMOTE 등)

[4] 연합 학습(Federated Learning) + 불균형
    └─ 분산 환경의 데이터 불균형 처리

[5] 설명 가능한 불균형 학습
    └─ 어떤 특징이 소수 클래스를 정의하는지 설명
```

### 5.3 imbens와 최신 연구의 관계

```
imbens (2021)          최신 연구 방향
───────────────────    ────────────────────────────
sklearn 기반 EIL   →   딥러닝 프레임워크 통합 필요
정적 리샘플러       →   메타러닝 기반 동적 선택 (MESA)
수동 비용 행렬      →   자동 비용 추정
단순 SMOTE 계열     →   GAN/VAE 기반 합성 (더 현실적)
표형 데이터 집중    →   그래프, 텍스트, 시계열 불균형
```

---

## 참고 자료 (출처)

### 직접 참조 (제공된 PDF)
1. **Liu, Z., Kang, J., Tong, H., Chang, Y. (2023).** "IMBENS: Ensemble Class-imbalanced Learning in Python." arXiv:2111.12776v2.

### 논문 내 인용 참고문헌 (논문 원문에서 확인됨)
2. **Liu, Z., Cao, W., Gao, Z., et al. (2020).** "Self-paced ensemble for highly imbalanced massive data classification." ICDE 2020, pp. 841-852.
3. **Liu, X.-Y., Wu, J., Zhou, Z.-H. (2009).** "Exploratory undersampling for class-imbalance learning." IEEE Trans. SMC-B, 39(2):539-550.
4. **Chen, C., Liaw, A., Breiman, L. (2004).** "Using random forest to learn imbalanced data." UC Berkeley Tech Report.
5. **Chawla, N.V., Lazarevic, A., Hall, L.O., Bowyer, K.W. (2003).** "SMOTEBoost: Improving prediction of the minority class in boosting." ECML/PKDD 2003.
6. **Fan, W., Stolfo, S.J., Zhang, J., Chan, P.K. (1999).** "AdaCost: Misclassification cost-sensitive boosting." ICML 1999.
7. **Seiffert, C., Khoshgoftaar, T.M., Van Hulse, J., Napolitano, A. (2010).** "RUSBoost: A hybrid approach to alleviating class imbalance." IEEE Trans. SMC-A, 40(1):185-197.
8. **Galar, M., Fernandez, A., Barrenechea, E., et al. (2012).** "A review on ensembles for the class imbalance problem." IEEE Trans. SMC-C, 42(4):463-484.
9. **Lemaître, G., Nogueira, F., Aridas, C.K. (2017).** "Imbalanced-learn: A python toolbox to tackle the curse of imbalanced datasets." JMLR, 18(17):1-5.
10. **He, H., Garcia, E.A. (2008).** "Learning from imbalanced data." IEEE Trans. TKDE, (9):1263-1284.
11. **Pedregosa, F., et al. (2011).** "Scikit-learn: Machine learning in Python." JMLR, 12:2825-2830.
12. **Dong, X., Yu, Z., Cao, W., et al. (2020).** "A survey on ensemble learning." Frontiers of Computer Science, 14:241-258.

### 추가 참고 (일반 지식 기반)
13. **Haixiang, G., et al. (2017).** "Learning from class-imbalanced data: Review of methods and applications." Expert Systems with Applications, 73:220-239.
14. **GitHub Repository:** https://github.com/ZhiningLiu1998/imbalanced-ensemble
15. **Documentation:** https://imbalanced-ensemble.readthedocs.io

---

> **최종 정확도 고지:** 본 답변의 §5 (2020년 이후 연구 비교)의 일부 세부 내용(MESA, DIVE, KRNN 등)은 제 학습 데이터를 바탕으로 작성하였으며, 논문 PDF에서 직접 확인되지 않은 내용입니다. 해당 부분은 참고용으로만 활용하시고, 실제 연구 시 원 논문을 반드시 확인하시기 바랍니다.
