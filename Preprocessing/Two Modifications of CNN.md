# Two Modifications of CNN

---

## ⚠️ 사전 고지

본 논문은 현대의 딥러닝 기반 CNN(Convolutional Neural Network)과 **무관**합니다. 여기서 CNN은 **Condensed Nearest Neighbor** (압축 최근접 이웃) 규칙을 의미하며, 1976년 IEEE Transactions on Systems, Man, and Cybernetics에 게재된 패턴 인식 관련 논문입니다.

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

> CNN(Condensed Nearest Neighbor)은 샘플을 **무작위로** 처리하기 때문에, (a) 불필요한 내부 샘플을 보유하고, (b) 결정 경계 근방의 샘플 대신 내부 샘플을 선택하는 문제가 있다. 이를 해결하기 위해 **경계 근방 점만을 활용**하는 두 가지 개선 방법(Method 1, Method 2)을 제안한다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **문제 정의** | CNN의 무작위 샘플 선택으로 인한 비효율성 명확히 식별 |
| **Method 1** | 반대 클래스 최근접 이웃 기반 경계점 탐색 알고리즘 제안 |
| **Method 2** | 클래스 간 경계 쌍(pair)을 명시적으로 탐색하는 알고리즘 제안 |
| **이론적 증명** | Method 2가 D의 모든 점을 올바르게 분류함을 수학적으로 증명 |
| **실험적 검증** | 기존 방법(CNN, Ritter et al. 방법)과 비교 실험 수행 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**원래 CNN 알고리즘의 문제점:**

원래 Hart(1968)의 CNN은 설계 집합 $D$에서 부분집합 $E \subseteq D$를 구성하여, $E$로 최근접 이웃(NN) 분류를 수행해도 $D$를 사용한 것과 거의 동일한 성능을 유지하는 것이 목표입니다.

그러나 CNN은:
- **문제 (a):** 무작위 처리로 인해 결정 경계와 무관한 **내부 샘플**이 $E$에 포함됨 → $E$가 필요 이상으로 커짐
- **문제 (b):** $D$에서 경계점이 아닌 샘플이 $E$에서는 경계점으로 작동하게 되어 **경계 이동(boundary shift)** 발생

수식으로 표현하면, 이상적인 $E$는:

$$E^* = \arg\min_{E \subseteq D} |E| \quad \text{subject to} \quad \forall x \in D,\ \text{NN}(x, E) = \text{class}(x)$$

그러나 CNN은 이 최적해를 보장하지 못하며, Tomek은 이를 개선하는 두 방법을 제안합니다.

---

### 2.2 제안 방법

#### 🔹 Method 1: 반대 클래스 최근접 이웃 기반 경계점 탐색

**핵심 아이디어:**

임의의 점 $x \in D$에 대해, 반대 클래스에서의 최근접 이웃을 다음과 같이 정의합니다:

$$y = nno(x) = \arg\min_{v \in D,\ \text{class}(v) \neq \text{class}(x)} \text{dist}(x, v)$$

$y$는 결정 경계 근방에 위치할 가능성이 높으므로, 이를 초기 $E$의 후보로 사용합니다.

**알고리즘 흐름:**

```
a) pass = 1
b) x ∈ D 무작위 선택, y = nno(x) 계산
   D(1) = D - {y}, E = {y}, F ≠ ∅
c) D(pass+1) = ∅, count = 0
d) x ∈ D(pass) 무작위 선택, E로 NN 분류
e) 분류 일치 → D(pass+1) = D(pass+1) ∪ {x}
   불일치 → x ∈ F이면: E = E ∪ {x}, F = F - {x}
            x ∉ F이면: F로 분류
              일치 → E = E ∪ {x}, F = F - {x}
              불일치 → z = nno(x), z ∈ D(pass)를 F에 추가
                       u = argmin_{v ∈ A} dist(v, z) 탐색
                       (A = {w ∈ D(pass) | dist(x,w) < dist(x,z), class(w) = class(x)})
                       E = E ∪ {u}
f~h) CNN과 동일
```

특히 내부 탐색에서 사용되는 점 $u$의 선택 기준:

$$u = \arg\min_{v \in A} \text{dist}(v, z), \quad A = \{w \in D(\text{pass}) \mid \text{dist}(x, w) < \text{dist}(x, z),\ \text{class}(w) = \text{class}(x)\}$$

**보장:** Method 1은 $E$가 $D$의 모든 점을 올바르게 분류함을 보장하며, $E$는 경계점만을 포함합니다.

---

#### 🔹 Method 2: 경계 쌍(Boundary Pair) 탐색

**핵심 아이디어:**

클래스 1의 샘플 $x(i),\ i=1,\ldots,N$과 클래스 2의 샘플 $y(j),\ j=1,\ldots,M$에 대해, 두 클래스 간 **경계 쌍 집합** $C$를 다음 조건으로 구성합니다:

중간점 $z$를 정의:

$$z = 0.5 \cdot (x(I) + y(J))$$

쌍 $(x(I), y(J))$가 $C$에 포함되는 조건 (Fig. 4 플로우차트 기반):

$$\forall x(I_1) \in D:\ \text{dist}(x(I_1), z) \geq \text{dist}(x(I), z)$$

$$\forall y(J_1) \in D:\ \text{dist}(y(J_1), z) \geq \text{dist}(y(J), z)$$

즉, $z$에 대해 $x(I)$가 같은 클래스 내에서 가장 가깝고, $y(J)$가 반대 클래스 내에서 가장 가까운 쌍만 $C$에 포함시킵니다.

**중요한 특성:**
- $C$의 선택은 **순서 독립적(order independent)** → 재현성 보장
- CNN과 달리 무작위성이 없음

**정리 (Appendix):**

> **Theorem:** Method 2로 생성된 부분집합 $C$를 사용한 NN 규칙으로 $D$의 모든 점이 올바르게 분류된다.

**증명 요약:**

$x_1 \in D - C_1$이 잘못 분류된다고 가정합니다. $x_1$의 $C$에서의 최근접 이웃 $y \in C$ (반대 클래스)가 존재한다고 하면, 중간점:

$$z = 0.5 \cdot (x_1 + y)$$

$x_1 \notin C$이므로, $z$에 더 가까운 점 $x_2 \in D$가 존재해야 합니다:

$$\text{dist}(x_2, z) < \text{dist}(x_1, z)$$

이후 귀납법(induction)과 $D$의 유한성(finiteness)으로 모순을 도출하여 증명 완성.

---

### 2.3 모델 구조

본 논문은 딥러닝 모델이 아닌 **인스턴스 기반 학습(instance-based learning)** 프레임워크에서의 데이터 전처리 알고리즘입니다. 전체 구조:

```
원본 설계 집합 D
       ↓
[전처리: Method 1 또는 Method 2]
       ↓
압축된 설계 집합 E (또는 C)
       ↓
NN 분류기 적용
       ↓
분류 결과
```

---

### 2.4 성능 향상 및 한계

#### 성능 향상

Fig. 6의 실험 결과 (400개 샘플, 2클래스, 균일분포):

| 방법 | 특징 |
|------|------|
| **Method 1** | CNN보다 더 작은 $E$, 경계 근방 점만 보유 |
| **Method 2** | 경계 쌍 명시적 탐색, 순서 독립적, 더 정확한 경계 표현 |
| **CNN (Hart)** | 내부점 혼입, 불필요한 경계 이동 발생 |
| **Ritter et al.** | 경계점 탐색 시도하나 본 방법보다 열등 |

**공통 향상점:**
1. 결과 설계 집합 $E$의 크기가 더 작음
2. 보유된 경계점이 실제 결정 경계에 더 가까움

#### 한계

1. **Method 1의 불완전성:** 모든 바람직한 경계점이 $E$에 포함된다고 보장하지 못함 (Fig. 2)
2. **계산 복잡도:** 반대 클래스 최근접 이웃 탐색으로 인해 원래 CNN보다 계산량 증가
3. **소규모 실험:** 단일 예시(400샘플, 2차원, 2클래스)만으로 검증 → 고차원·다중 클래스 일반화 미검증
4. **이론적 최적성 미보장:** $E$가 최소 크기라는 보장 없음
5. **노이즈 민감성:** 경계 근방 점에 집중하므로 노이즈 데이터에 취약할 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상 메커니즘

**경계점 집중의 이론적 이점:**

NN 분류기의 일반화 오류는 베이즈 오류 $e^*$와 다음 관계가 있습니다 (Cover & Hart, 1967):

```math
e^* \leq e_{NN} \leq 2e^*\left(1 - e^*\right)
```

Tomek의 방법은 $E$를 경계점으로 구성함으로써, NN 분류기가 결정 경계를 더 정확하게 근사하게 합니다. 이는 다음을 의미합니다:

$$P(\text{error} \mid E_{\text{Tomek}}) \leq P(\text{error} \mid E_{\text{CNN}})$$

(경험적 주장; 엄밀한 이론적 증명은 논문에 없음)

### 3.2 Wilson 편집(Editing)과의 시너지

논문에서 언급된 Wilson(1972)의 편집 방법과의 결합:

$$D \xrightarrow{\text{Wilson Editing}} D_{\text{edited}} \xrightarrow{\text{Method 1 or 2}} E_{\text{final}}$$

Wilson 편집으로 경계 "오염" 샘플(wrong-side samples)을 제거한 후 Tomek 방법을 적용하면, 더 "깨끗한" 경계 집합을 얻어 일반화 성능이 크게 향상됩니다. 논문은 이를 명시적으로 언급합니다:

> *"the edited design set is much cleaner than the original one"*

### 3.3 Method 2의 구조적 일반화 기여

Method 2가 생성하는 경계 쌍 집합 $C$는:

$$C = \{(x(I), y(J)) \mid x(I),\ y(J) \text{가 상호 중간점에서 최근접}\}$$

이는 단순히 분류 집합을 줄이는 것을 넘어, **결정 경계의 구조적 정보**를 포함합니다. 논문은 이것이 구간별 선형 분류기(piecewise-linear classifier) 개발에 활용될 수 있다고 제안합니다:

> *"This information might be very useful in the development of more powerful methods of classification by piecewise-linear classifiers."*

이는 현대의 SVM(Support Vector Machine)에서 서포트 벡터(support vector)의 역할과 개념적으로 유사합니다.

### 3.4 과적합 감소 가능성

내부 샘플 제거는 모델이 학습 데이터의 내부 구조(노이즈 포함)에 과적합되는 것을 방지합니다. 수식으로:

$$\text{Generalization Gap} = \mathbb{E}[L_{\text{test}}] - \mathbb{E}[L_{\text{train}}]$$

$E$의 크기가 작고 경계 집중적일수록, NN 규칙의 효과적 복잡도가 감소하여 일반화 갭이 줄어들 가능성이 있습니다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### 🔸 단기적 영향 (1976~2000년대)

1. **데이터 압축/편집 연구의 기반:** Tomek Link의 개념은 이후 데이터 전처리 분야의 표준 도구로 발전
2. **불균형 데이터 처리:** "Tomek Links" 개념은 다수 클래스 샘플 제거를 통한 클래스 불균형 해소 기법으로 재해석됨
3. **프로토타입 선택(prototype selection):** Method 1, 2는 현대 프로토타입 선택 알고리즘의 선구적 아이디어 제공

#### 🔸 장기적 영향

**Tomek Links의 재발견:**

Tomek이 제안한 경계 쌍의 개념은 훗날 **"Tomek Links"** 로 명명되어, 특히 **불균형 학습(imbalanced learning)** 분야에서 널리 사용됩니다:

$$\text{Tomek Link: } (x, y) \text{ s.t. } \nexists z: \text{dist}(x,z) < \text{dist}(x,y) \text{ or } \text{dist}(y,z) < \text{dist}(x,y)$$

이는 현대의 SMOTE+Tomek, ENN+Tomek 등 하이브리드 기법의 기반이 됩니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 불균형 데이터 처리에서의 Tomek Links 활용

| 연구 방향 | Tomek(1976) 대비 발전 |
|-----------|----------------------|
| **SMOTE+Tomek** | 소수 클래스 오버샘플링 + Tomek Links로 경계 정리 |
| **딥러닝 기반 프로토타입 선택** | 신경망으로 경계점 표현 학습 |
| **능동 학습(Active Learning)** | 경계 근방 불확실 샘플 선택 (Tomek 아이디어의 확장) |

### 5.2 현대적 관점에서의 비교

#### 능동 학습과의 연결

현대 능동 학습에서 경계 근방 샘플 선택:

$$x^* = \arg\min_{x \in D_{\text{unlabeled}}} \max_{y} P(y \mid x)$$

이는 Tomek Method 1의 $nno(x)$ 기반 경계점 탐색과 개념적으로 동일한 철학을 공유합니다.

#### 코어셋(Coreset) 방법과의 비교

현대 코어셋 연구 (Sener & Savarese, 2018):

$$\min_{S \subseteq D} \max_{x \in D} \min_{s \in S} \text{dist}(x, s)$$

Tomek의 $E$ 구성은 이 최적화 문제의 경계점 집중 버전으로 해석 가능합니다.

#### 그래프 기반 방법과의 비교

현대의 그래프 신경망 기반 데이터 압축 방법들은 Tomek의 쌍(pair) 기반 접근과 구조적으로 유사한 **엣지 기반 경계 표현**을 활용합니다.

### 5.3 한계와 현대적 보완

| Tomek(1976) 한계 | 2020년 이후 해결 방향 |
|-----------------|----------------------|
| 저차원·소규모 실험 | 고차원 벤치마크에서 Tomek Links 재검증 |
| 이진 분류 중심 | 다중 클래스 Tomek Links 확장 연구 |
| 계산 복잡도 $O(n^2)$ | 근사 최근접 이웃(ANN) 알고리즘으로 가속 |
| 노이즈 취약성 | 강건한 거리 메트릭 학습과 결합 |

---

## 6. 향후 연구 시 고려할 점

### 6.1 알고리즘 측면

1. **고차원 확장:** 고차원에서 거리 기반 방법의 "차원의 저주" 문제 해결 필요
   $$\lim_{d \to \infty} \frac{\max_{x,y} \text{dist}(x,y) - \min_{x,y} \text{dist}(x,y)}{\min_{x,y} \text{dist}(x,y)} \to 0$$

2. **계산 효율화:** 현재 $O(n^2)$ 복잡도를 ANN(Approximate Nearest Neighbor) 등으로 $O(n \log n)$ 수준으로 감소

3. **비유클리드 거리:** 도메인 특화 거리 메트릭 학습과의 결합

### 6.2 일반화 측면

4. **이론적 보장 강화:** PAC 학습 프레임워크에서 $E$의 크기와 일반화 오류의 관계를 엄밀히 분석

5. **분포 변화(Distribution Shift) 강건성:** $D$와 실제 테스트 분포 간의 차이를 고려한 경계점 선택

6. **딥러닝 특징 공간에서의 적용:** 원 입력 공간 대신 딥러닝 임베딩 공간에서 Tomek-like 전처리 수행

### 6.3 응용 측면

7. **연속 학습(Continual Learning):** 새로운 데이터가 추가될 때 $E$를 동적으로 업데이트하는 방법

8. **공정성(Fairness):** 경계점 선택이 특정 인구통계 그룹에 편향되지 않도록 보장

---

## 참고 자료

### 논문 내 인용 문헌
1. **Hart, P. E.** (1968). "The condensed nearest neighbor rule." *IEEE Transactions on Information Theory*, IT-14, pp. 515–516.
2. **Duda, R. O. & Hart, P. E.** (1973). *Pattern Classification and Scene Analysis*. John Wiley & Sons.
3. **Ritter, G. L. et al.** (1975). "An algorithm for a selective nearest neighbor decision rule." *IEEE Transactions on Information Theory*, IT-21, pp. 665–669.
4. **Gates, G. W.** (1972). "The reduced nearest neighbor decision rule." *IEEE Transactions on Information Theory*, IT-18, pp. 431–433.
5. **Wilson, D. L.** (1972). "Asymptotic properties of nearest neighbor rules using edited data." *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-2, pp. 408–421.
6. **Tomek, I.** (1976). "An experiment with the edited nearest neighbor rule." *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-6, pp. 448–452.

### 주요 분석 대상 논문
- **Tomek, I.** (1976). "Two Modifications of CNN." *IEEE Transactions on Systems, Man, and Cybernetics*, November 1976, pp. 769–772.

### 현대 연구 맥락 참고
- **Cover, T. & Hart, P.** (1967). "Nearest neighbor pattern classification." *IEEE Transactions on Information Theory*, 13(1), pp. 21–27.
- **Sener, O. & Savarese, S.** (2018). "Active learning for convolutional neural networks: A core-set approach." *ICLR 2018*.
- **He, H. & Garcia, E. A.** (2009). "Learning from Imbalanced Data." *IEEE Transactions on Knowledge and Data Engineering*, 21(9), pp. 1263–1284. *(Tomek Links의 현대적 활용 맥락)*

> **⚠️ 정확도 고지:** 2020년 이후 Tomek Links를 직접 다루는 최신 논문의 구체적 인용 정보는 제가 확인할 수 없어 제목을 명시하지 않았습니다. 위 현대 연구 비교는 개념적 연결성에 기반한 분석임을 밝힙니다. 정확한 최신 문헌은 Google Scholar에서 "Tomek Links imbalanced learning 2020-2024"로 검색하실 것을 권장합니다.
