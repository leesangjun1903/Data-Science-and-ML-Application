
# LTF: A Label Transformation Framework for Correcting Label Shift

> **📌 논문 정보**
> - **저자**: Jiaxian Guo, Mingming Gong, Tongliang Liu, Kun Zhang, Dacheng Tao
> - **학회**: ICML 2020 (Proceedings of the 37th International Conference on Machine Learning, PMLR 119:3843–3853)
> - **GitHub**: [CR-Gjx/LTF-Label-Transformation-Framework](https://github.com/CR-Gjx/LTF-Label-Transformation-Framework)
> - **공식 페이퍼**: [proceedings.mlr.press/v119/guo20d.html](https://proceedings.mlr.press/v119/guo20d.html)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

Distribution shift는 현실 문제에 딥러닝 모델을 배포할 때 주요 장애물이다. LTF는 특히 **Label Shift** — 레이블의 주변 분포 $P_Y$는 변하지만, 조건부 분포 $P_{X|Y}$는 변하지 않는 형태의 분포 이동 — 에 초점을 맞춘다.

기존의 대부분의 방법들은 소스·타겟 도메인 간 레이블 분포의 밀도 비율(density ratio)을 밀도 매칭(density matching)으로 추정하는데, 이는 대규모 데이터에서 계산적으로 비현실적이거나 이산형 레이블에만 제한된다.

### 🌟 주요 기여

LTF는 $P_Y$의 이동과 조건부 분포 $P_{X|Y}$를 신경망으로 **암묵적으로(implicitly)** 모델링하는 엔드투엔드 Label Transformation Framework이다. 딥 네트워크의 유연성 덕분에, 이 프레임워크는 연속형, 이산형, 다차원 레이블을 **통합된 방식**으로 처리할 수 있으며 대용량 데이터로 확장 가능하다.

특히 이미지와 같은 고차원 $X$에서는 $X$ 내의 불필요한 정보가 추정 정확도를 심각하게 저하시키는데, 이를 해결하기 위해 $Y$와 무관한 정보를 제거한 **저차원 피처 공간**에서 분포를 매칭하는 방법을 제안한다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

**Label Shift (= Target Shift)** 문제를 공식적으로 정의하면:

$$P_S(X, Y) \neq P_T(X, Y)$$

단, 다음 조건을 만족:

$$P_S(X \mid Y) = P_T(X \mid Y), \quad P_S(Y) \neq P_T(Y)$$

즉, 타겟 변수 $Y$의 주변 분포 $P_Y$는 변하지만, 피처의 조건부 분포 $P_{X|Y}$는 변하지 않는다.

이때 타겟 도메인에서의 예측 분포는 다음과 같이 분해된다:

$$P^T_X(x) = \int P^T_{X|Y}(x \mid y) \, P^T_Y(y) \, dy = \int P^S_{X|Y}(x \mid y) \, P^T_Y(y) \, dy$$

타겟 도메인에서는 레이블 $Y$가 관측되지 않기 때문에, $P^T_Y$를 직접 알기 어렵다는 것이 핵심 난제이다.

### 2-2. 제안 방법 (수식 포함)

LTF는 다음 두 핵심 모듈로 구성된다:

#### (1) Label Transformation Model (LT)

소스 도메인의 레이블 분포 $P^S_Y$를 타겟 도메인의 레이블 분포 $P^R_Y$로 변환하는 **변환 함수 $T$**를 학습:

$$P^R_Y = T(P^S_Y; \theta_T)$$

딥 네트워크의 유연성 덕분에 서로 다른 변환 모델 $T$를 설계하여 이산형, 연속형, 다차원 레이블을 처리할 수 있다.

변환된 레이블로부터 새로운 데이터 샘플을 생성하기 위해, 다음과 같은 **함수적 모델(functional model)**을 사용:

$$X = G(Y, \epsilon), \quad \epsilon \sim P_\epsilon$$

여기서 $G$는 Label Influence Recovery Network이며, $\epsilon$은 독립 노이즈이다.

#### (2) Label Influence Recovery Network (G)

변환된 레이블을 Label Influence Recovery Network $G$에 통과시켜, 조건부 분포 $P_{X|Y}$를 암묵적으로 모델링하고 분포 $P^R_Y$를 따르는 샘플을 생성한다.

생성된 샘플의 분포와 타겟 도메인 분포의 **매칭 목적함수(matching objective)**:

$$\min_{\theta_T} \, d\!\left(P^R_X,\; P^T_X\right)$$

여기서 $d(\cdot, \cdot)$은 두 분포 간의 거리(예: MMD, Wasserstein, 또는 Adversarial Loss).

#### (3) 저차원 피처 공간에서의 분포 매칭

이미지 등 고차원 $X$에서는 $Y$와 무관한 불필요한 정보가 추정 정확도를 심각하게 저하시키므로, 생성 모델이 암시하는 분포와 타겟 도메인 분포를 $Y$와 무관한 정보를 제거한 **저차원 피처 공간**에서 매칭하도록 제안한다.

즉, 피처 추출기 $\phi$를 통해:

$$\min_{\theta_T, \phi} \, d\!\left(\phi(P^R_X),\; \phi(P^T_X)\right)$$

이 과정은 $Y$와 관련된 정보만을 보존하는 표현 공간을 학습하도록 유도한다.

### 2-3. 모델 구조

LTF의 전체 파이프라인은 다음과 같이 요약된다:

```
[Source Labels P^S_Y]
       ↓
[Label Transformation Model T (θ_T)]  → 변환된 레이블 P^R_Y
       ↓
[Label Influence Recovery Network G]  → 생성 샘플 (분포 P^R_X)
       ↓
[Feature Extractor φ]                 → 저차원 피처 공간
       ↓
[Distribution Matching]               ↔ [Target Domain P^T_X]
       ↓
[P^T_Y 추정 완료] → [Classifier 재보정]
```

- 실험은 Fashion-MNIST, MNIST, CIFAR-10 데이터셋을 PyTorch 기반으로 구현되었다.
- 실험 설정으로는 Random Dirichlet Shift(디리클레 분포 기반 랜덤 레이블 분포 생성)와 Tweak-One Shift(대규모 레이블 확률 정량화 평가)가 사용되었다.
- Minority-Class Shift는 소규모 레이블 확률 정량화를 평가하기 위해 사용되었으며, 한 클래스의 비율을 $[0.5, 0.6, 0.7, 0.8, 0.9]$로 설정하고 나머지 클래스는 균등하게 분배하였다.

### 2-4. 성능 향상

이론적·실증적 연구 모두에서 기존 방법 대비 LTF의 우수성이 입증되었다.

기존 비교 대상 방법들:
- **BBSL (Black Box Shift Learning)**: BBSL과 RLLS는 레이블 시프트 처리의 최신 기법으로 부상했으나, 두 방법 모두 중요도 가중치(importance weights)로 모델을 재학습해야 한다.
- **RLLS**: RLLS는 특히 소샘플 및 대형 시프트 환경에서 기존 방법 대비 분류 정확도를 향상시킨다.

LTF는 기존 방법들과 달리:
1. 이산·연속·다차원 레이블을 통합 처리
2. 대규모 데이터에 확장 가능
3. 별도의 밀도비 추정 없이 end-to-end 학습

### 2-5. 한계점

논문에서 인정되거나 구조적으로 유추되는 한계는 다음과 같다:

| 한계 | 설명 |
|---|---|
| **생성 모델 의존성** | 사전 학습된 생성 모델(G)이 필요하며, 생성 품질이 전체 성능에 영향을 줌 |
| **Label Shift 가정** | $P_{X \mid Y}$가 완전히 불변이라는 강한 가정에 의존 |
| **고차원 데이터 추가 설계 필요** | 고차원 $X$(예: 이미지)에서 불필요한 정보가 추정 정확도를 심각하게 저하시키므로, 저차원 피처 공간 매칭이라는 추가적인 설계가 필요하다. |
| **온라인/비정상 환경 미지원** | 정적 레이블 분포 이동만을 가정하며, 시간에 따라 변하는 레이블 분포는 다루지 않음 |

---

## 3. 모델의 일반화 성능 향상 가능성

LTF의 일반화 성능 향상은 크게 세 가지 메커니즘에 기반한다:

### 3-1. 타겟 도메인 분포를 반영한 재보정(Recalibration)

레이블 시프트가 발생하면 소스 분포로 학습된 모델의 예측 확률은 타겟 도메인에서 왜곡된다. LTF는 $P^T_Y$를 추정하여 분류기의 출력을 재보정함으로써, 타겟 도메인에서의 일반화 성능을 향상시킨다:

$$P^T(Y=k \mid X=x) \propto P^S(Y=k \mid X=x) \cdot \frac{P^T(Y=k)}{P^S(Y=k)}$$

### 3-2. 저차원 피처 공간에서의 불변 표현 학습

고차원 입력 $X$에서의 불필요한 정보 문제를 해결하기 위해, $Y$와 관련된 정보만을 담은 저차원 피처 공간에서 분포를 매칭한다. 이는 일반화에 유리한 **불변 표현(invariant representation)**을 자연스럽게 학습하도록 유도한다.

### 3-3. 연속·다차원 레이블 지원으로 인한 광범위한 적용 가능성

딥 네트워크의 유연성 덕분에, 이 프레임워크는 연속형, 이산형, 다차원 레이블을 통합된 방식으로 처리하며 대규모 데이터에도 확장 가능하다. 이는 실제 응용 환경(회귀, 분류, 멀티레이블 등)에서 일반화 성능을 높이는 기반이 된다.

---

## 4. 이 논문이 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 4-1. 후속 연구에 미치는 영향

#### 온라인 레이블 시프트 적응 연구 촉진

터미널 기기의 소형화 및 스트리밍 데이터의 확산으로, 데이터를 순차적으로 처리하고 각 샘플이 도착하는 즉시 모델 파라미터를 즉시 조정할 수 있는 온라인 레이블 시프트 추정 방법에 대한 수요가 급격히 증가하고 있다. 2021년 Wu et al.은 온라인 경사 하강법(OGD)과 Follow the History(FTH) 방법을 활용하여 온라인 레이블 시프트 추정을 최초로 다루었다.

이를 기반으로 Bai et al.은 편향 없는 위험 추정량을 도입하고 온라인 레이블 시프트 추정에 대한 이론적 보장을 제공하여, 이 연구 방향의 이론적 토대를 강화하였다.

#### Bayesian 및 EM 기반 방법으로의 확장

EM 기반 방법들이 재조명되어 CIFAR-10, MNIST 등의 벤치마크에서 BBSE 및 RLLS를 능가하는 성능을 보였다. 그러나 기존 기법들은 상대적으로 소규모 데이터셋에서 평가되었으며, 레이블 시프트 추정에서 클래스 불균형 훈련 데이터의 영향을 거의 다루지 않았다.

최근에는 고차원 디리클레 분포로 클래스 사전 확률을 모델링하고 EM 기반 오프라인 최적화를 통해 Maximum A Posterior Label Shift(MAPLS)를 구현하는 Bayesian 프레임워크도 제안되었다.

#### 효율적인 레이블 시프트 적응 방법(ELSA 등)

LTF에 이어, ELSA(Efficient Label Shift Adaptation)라는 새로운 방법이 제안되었는데, 적응 가중치를 선형 시스템 풀이로 추정하여 사후 예측 보정 없이도 최신 추정 성능을 달성하며 계산 효율성을 높였다.

#### Long-tailed Recognition 및 교정(Calibration) 연구와의 연계

최근 연구들은 전통적인 i.i.d. 교정 방법으로 교정된 모델들이 데이터셋 이동 하에서 교정 성능을 잃는다는 것을 보였다. LTF의 분포 매칭 아이디어는 레이블 시프트 하의 교정 연구에도 직접적인 영향을 주고 있다.

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 핵심 아이디어 | LTF 대비 특징 |
|---|---|---|---|
| **BBSL** (Lipton et al.) | 2018 | Black-box 예측기 기반 밀도비 추정 | 이산 레이블 한정, 재학습 필요 |
| **RLLS** (Azizzadenesheli et al.) | 2019 | 정규화된 밀도비 추정, 일반화 보장 | 차원 독립적 일반화 한계를 도출하며, 클래스 수 $k$에 대해 $k\log(k)$ 배 가중치 추정 오차를 개선한다. |
| **MLLS** (Alexandari et al.) | 2020 | 최대 우도 + 편향 보정 교정 | 재학습 불필요, 교정 모델 필요 |
| **LTF** (Guo et al.) | 2020 | 생성 모델 기반 레이블 변환, end-to-end | 연속·이산·다차원 통합, 확장성 우수 |
| **Online Label Shift** (Wu et al.) | 2021 | OGD/FTH 기반 온라인 적응 | 스트리밍 데이터 처리 |
| **ELSA** (Tian et al.) | 2023 | 반모수 모델 기반 선형 시스템 풀이 | 사후 보정 없이 계산 효율적으로 최신 성능 달성 |
| **LaSCal** (Popordanoska et al.) | 2024 | 레이블 시프트 하의 교정 오차 추정기 | 타겟 레이블 없이 교정 가능 |

### 4-3. 앞으로 연구 시 고려할 점

1. **Label Shift 가정의 완화**: 레이블 시프트 가정 자체가 현실에서는 지나치게 단순화된 경우가 있으며, 레이블 $y$ 외에 관측 불가능한 속성 $z$에도 $x$가 의존하는 설정도 연구할 필요가 있다.

2. **온라인·비정상 환경 대응**: 최근에는 적응 문제를 온라인 회귀로 환원하고 최적의 동적 후회(dynamic regret)를 보장하는 알고리즘이 개발되고 있으며, 이론적 최적성과 강력한 실증적 성능을 동시에 달성하는 방향이 중요하다.

3. **대규모·클래스 불균형 데이터셋에서의 평가**: 기존 기법들은 상대적으로 소규모 데이터셋에서 평가되었으며, 클래스 불균형 훈련 데이터의 영향이 레이블 시프트 추정에 미치는 영향은 충분히 연구되지 않았다.

4. **생성 모델 품질과의 결합**: LTF의 성능은 사전 학습된 생성 모델(G)의 품질에 크게 의존하므로, 최신 생성 모델(Diffusion Model, VAE 등)과의 결합을 고려할 필요가 있다.

5. **교정(Calibration)과의 통합**: 레이블 시프트 환경에 특화된 교정 오차 추정기를 훈련 목적함수로 활용하여 사후 교정 및 학습 가능한 교정 방법에 적용하는 연구가 유망하다.

---

## 📚 참고 자료 및 출처

1. **LTF 논문 (공식)**: Guo et al., *LTF: A Label Transformation Framework for Correcting Label Shift*, ICML 2020. https://proceedings.mlr.press/v119/guo20d.html
2. **ICML 2020 포스터 페이지**: https://icml.cc/virtual/2020/poster/5962
3. **Papers with Code**: https://paperswithcode.com/paper/ltf-a-label-transformation-framework-for
4. **Semantic Scholar**: https://www.semanticscholar.org/paper/LTF:-A-Label-Transformation-Framework-for-Label-Guo-Gong/994a4f9a75ca10da4f9a0f4a3a03fd622aeeea41
5. **GitHub (공식 코드)**: https://github.com/CR-Gjx/LTF-Label-Transformation-Framework
6. **RLLS 논문**: Azizzadenesheli et al., *Regularized Learning for Domain Adaptation under Label Shifts*, ICLR 2019. https://arxiv.org/pdf/1903.09734
7. **ELSA 논문**: Tian et al., *ELSA: Efficient Label Shift Adaptation through the Lens of Semiparametric Models*, Semantic Scholar. https://www.semanticscholar.org/paper/ELSA
8. **MLLS 논문**: Alexandari et al., *Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation*, arXiv 2020. https://arxiv.org/abs/1901.06852
9. **LaSCal 논문**: Popordanoska et al., *LaSCal: Label-Shift Calibration without target labels*, NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/783c5986e1d6112cb4688d9b2105609a-Paper-Conference.pdf
10. **Bayesian Online Label Shift**: *Bayesian-based Online Label Shift Estimation with Dynamic Dirichlet Priors*, arXiv 2025. https://arxiv.org/html/2511.18615v1
11. **Label Shift Correction for Long-Tailed Recognition**: *Learning Label Shift Correction for Test-Agnostic Long-Tailed Recognition*, ICML 2024. https://palm.seu.edu.cn/weit/paper/ICML2024_LSC.pdf

> ⚠️ **정확도 참고**: 본 답변에서 LTF 논문의 상세 수식(특히 $G$의 구체적 functional form 및 실험 결과 수치)은 웹 검색으로 확인된 범위 내에서만 제시하였습니다. 논문 원문 PDF에서 직접 수식 세부 내용을 확인하실 것을 강력히 권장합니다.
