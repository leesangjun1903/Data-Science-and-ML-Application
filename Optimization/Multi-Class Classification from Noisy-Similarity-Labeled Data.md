
# Multi-Class Classification from Noisy-Similarity-Labeled Data 

> **논문 정보**
> - **제목**: Multi-Class Classification from Noisy-Similarity-Labeled Data
> - **저자**: Songhua Wu\*, Xiaobo Xia\*, Tongliang Liu, Bo Han, Mingming Gong, Nannan Wang, Haifeng Liu, Gang Niu
> - **소속**: University of Sydney, Xidian University, University of Melbourne, RIKEN, HKBU 외
> - **공개**: arXiv:2002.06508 (2020년 2월)

---

## 1. 핵심 주장 및 주요 기여 요약

유사도 레이블(similarity label)은 두 인스턴스가 같은 클래스에 속하는지를 나타내는 반면, 클래스 레이블(class label)은 인스턴스의 정확한 클래스를 알려 준다.

클래스 레이블 없이도 유사도 레이블이 붙은 쌍(pairwise) 데이터로부터 메타 분류 학습(meta classification learning)을 통해 다중 클래스 분류기를 학습할 수 있다. 그러나 유사도 레이블은 클래스 레이블보다 정보량이 적기 때문에 노이즈(noise)에 더 취약하다.

딥 뉴럴 네트워크는 노이즈가 있는 데이터를 쉽게 기억(memorize)하여 분류에서 과적합(overfitting)이 발생할 수 있다.

**주요 기여 세 가지:**

| 번호 | 기여 내용 |
|------|-----------|
| ① | **노이즈 전이 행렬(Noise Transition Matrix)** 기반 노이즈 모델링 |
| ② | **노이즈 데이터만으로** 전이 행렬 추정 및 노이즈-프리 레이블 할당 시스템 구축 |
| ③ | 분류기 일반화에 대한 **이론적 보장(generalization guarantee)** 제공 |

이 논문은 노이즈가 있는 유사도 레이블 데이터만으로 학습하는 방법을 제안하며, 노이즈를 모델링하기 위해 깨끗한 데이터와 노이즈 데이터 사이의 클래스 사후 확률(class-posterior probability)을 연결하는 노이즈 전이 행렬을 사용한다. 노이즈 데이터만으로 전이 행렬을 추정하여 노이즈 없는 클래스 레이블을 할당하는 학습 시스템을 구축하고, 이 방법의 일반화 이론을 수립한다.

---

## 2. 해결 문제 · 제안 방법 · 모델 구조 · 성능 및 한계

### 2-1. 해결하고자 하는 문제

실제 세계의 분류 응용에서 명시적인 클래스 레이블을 가진 데이터셋은 개인정보 민감성(예: 정치, 종교 등) 이유로 수집이 어려운 경우가 많다. 이에 반해 유사도 레이블은 명시적 정보를 드러내지 않아 프라이버시를 자연스럽게 보호한다.

그러나 유사도 레이블 기반 분류(SU classification)에는 명확한 한계가 있다: 응답자가 진실되게 답하지 않고 다른 사람에게 호의적으로 보이는 방식으로 대답할 수 있다. 따라서 실제로 비유사한 쌍이 유사하다고 레이블되는 경우가 발생하며, 이는 분류 성능을 크게 저하시킨다.

---

### 2-2. 제안 방법 (수식 포함)

#### 핵심 프레임워크: 노이즈 전이 행렬

**클래스 사후 확률과 유사도 사후 확률의 관계** (meta classification learning, Hsu et al., 2019 기반):

두 인스턴스 $X_i$, $X_j$에 대해 분류기 $f$가 범주 분포(categorical distribution)를 출력할 때, 예측된 **클린(clean) 유사도 사후 확률**은 다음과 같이 정의된다:

$$\hat{S}_{ij} = f(X_i)^\top f(X_j)$$

$\hat{S}_{ij}$는 두 범주 분포 간의 내적(inner product)으로 예측된 클린 유사도 사후 확률을 나타내며, $f(X_i)^\top f(X_j)$는 두 분포가 얼마나 유사한지를 측정한다.

**노이즈 유사도 사후 확률과 클린 유사도 사후 확률의 관계:**

노이즈 유사도 사후 확률 $P(\bar{H}\_{ij}|X_i, X_j)$와 클린 유사도 사후 확률 $P(H_{ij}|X_i, X_j)$는 다음 관계를 만족한다:

$$P(\bar{H}_{ij}|X_i, X_j) = T_s^\top P(H_{ij}|X_i, X_j)$$

따라서 유사도 전이 행렬(similarity transition matrix)을 이용하여 예측된 노이즈 유사도 사후 확률로부터 클린 유사도 사후 확률을 추론할 수 있다.

**클래스 전이 행렬로부터 유사도 전이 행렬 계산** (Class2Simi 및 관련 연구에서도 공유되는 핵심 수식):

유사도 전이 행렬의 원소는 클린 유사도 레이블 $H$가 노이즈 유사도 레이블 $\bar{H}$로 뒤집힐 확률, 즉 $T_{s,mn} := P(\bar{H} = n | H = m)$으로 정의되며, 유사도 전이 행렬의 차원은 항상 $2 \times 2$이다.

클래스 전이 행렬 $T_c$가 주어졌을 때, 유사도 전이 행렬 $T_s$의 원소는 아래와 같이 계산된다:

$$T_{s,00} = \frac{c^2 - c - \left(\sum_j \left(\sum_i T_{c,ij}\right)^2 - \|T_c\|_{\mathrm{Fro}}^2\right)}{c^2 - c}$$

$$T_{s,01} = \frac{\sum_j \left(\sum_i T_{c,ij}\right)^2 - \|T_c\|_{\mathrm{Fro}}^2}{c^2 - c}$$

$$T_{s,10} = \frac{c - \|T_c\|_{\mathrm{Fro}}^2}{c}, \quad T_{s,11} = \frac{\|T_c\|_{\mathrm{Fro}}^2}{c}$$

\*(여기서 $c$는 클래스 수, $\|\cdot\|_{\mathrm{Fro}}$는 Frobenius norm)*

---

### 2-3. 모델 구조

논문이 제안하는 학습 시스템은 다음 단계로 구성된다:

```
[입력: 노이즈 유사도 레이블 쌍 데이터]
        ↓
[Step 1] 노이즈 전이 행렬 T_s 추정
  - 노이즈 데이터만으로 앵커 포인트(anchor points) 방법론 등을 통해 추정
        ↓
[Step 2] 노이즈 보정 손실 함수 (loss correction)
  - T_s를 활용하여 noisy label에 의한 편향 제거
        ↓
[Step 3] 딥 뉴럴 네트워크 기반 분류기 f 학습
  - 인스턴스에 대해 노이즈 없는 클래스 레이블 할당
        ↓
[출력: 다중 클래스 분류기]
```

노이즈를 모델링하기 위해 노이즈 전이 행렬을 활용하여 클린 데이터와 노이즈 데이터 간의 클래스 사후 확률을 연결하고, 노이즈 데이터만으로 전이 행렬을 추정하여 인스턴스에 대해 노이즈 없는 클래스 레이블을 할당하는 새로운 학습 시스템을 구축한다.

---

### 2-4. 성능 향상

벤치마크 시뮬레이션 및 실제 노이즈 레이블 데이터셋에서의 실험 결과는 제안된 방법이 최신 방법(state-of-the-art)에 비해 우월함을 보여 준다.

---

### 2-5. 한계

논문 자체에서 명시적으로 밝힌 한계는 검색된 정보만으로는 구체적으로 확인하기 어렵습니다. 다만, 문헌 전반에서 공통적으로 지적되는 한계는 다음과 같습니다:

1. **노이즈 모델 가정**: 클래스 의존적(class-dependent) 노이즈 가정에 기반하며, 인스턴스 의존적(instance-dependent) 노이즈에는 적용이 제한될 수 있음.
2. **전이 행렬 추정 오류**: 노이즈 전이 행렬 추정이 부정확할 경우 성능 저하 가능.
3. **페어 생성의 계산 비용**: 인스턴스를 쌍(pair)으로 구성하므로 데이터 수가 많을수록 계산 비용이 증가.

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문은 제안된 방법이 분류기 학습에 어떻게 일반화되는지를 이론적으로 정당화한다.

**일반화 오차 경계(Generalization Error Bound):**

전이 행렬이 주어진 상황에서, 훈련 오류가 작고 훈련 샘플 크기가 크면 노이즈 유사도 사후 확률의 표현에 대한 기대 위험 $R(\hat{f})$도 작아짐을 Theorem 3이 함의한다.

전이 행렬이 잘 추정되면, 클린 유사도 사후 확률뿐만 아니라 클린 클래스에 대한 분류기도 올바르게 학습할 수 있다.

이를 형식화하면:

$$R(\hat{f}) \leq \hat{R}(\hat{f}) + O\left(\frac{M \cdot \prod_{l=1}^{d} M_l}{\sqrt{n}}\right)$$

\*(여기서 $\hat{R}$은 경험적 위험, $n$은 샘플 수, $M_l$은 각 레이어의 Frobenius norm 제약, $M$은 손실 함수의 상한)*

**일반화 성능 향상을 가능하게 하는 핵심 메커니즘:**

1. **노이즈율 감소**: 유사도 레이블 자체가 노이즈에 강인하기 때문에, 클래스 레이블에서 유사도 레이블로 변환할 경우 노이즈율이 감소함을 이론적으로 증명할 수 있다.

2. **페어와이즈 학습의 장점**: 이 변환을 통해 노이즈율의 감소가 이론적으로 보장되며, 노이즈 데이터 포인트로부터 사전 학습된 DNN이 노이즈가 있는 데이터 쌍으로부터 클린 클래스 레이블을 예측할 수 있게 된다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4-1. 연구에 미치는 영향

이 논문은 **"유사도 레이블 공간(similarity label space)에서 노이즈를 다루는"** 새로운 패러다임을 제시하여 이후 연구들에 직접적인 영향을 미쳤습니다.

#### ✅ 직접적 후속 연구: Class2Simi (ICML 2021)

Class2Simi는 노이즈 레이블 학습에 대한 새로운 관점을 제안하며, 노이즈 클래스 레이블을 가진 훈련 예시들을 노이즈 유사도 레이블을 가진 쌍(pair)으로 변환하고, 노이즈 유사도 레이블로 직접 강건한 분류기를 학습하는 딥러닝 프레임워크를 제안한다.

Class2Simi는 유사도 노이즈 전이 행렬을 추정하는 방법을 제공하고, 클래스 노이즈 전이 행렬이 부정확하게 추정되더라도 유도된 유사도 노이즈 전이 행렬은 여전히 잘 작동함을 보인다.

#### ✅ 유사도 기반 노이즈 탐지 연구 (2025)

최근 연구들은 유사성 관점에서의 이론적 분석을 제공하며, 잘못 레이블된 데이터 포인트의 마지막 특징 레이어와 실제 클래스 데이터 포인트 간의 유사도가 다른 클래스보다 크다는 이유를 밝히고 있다.

#### ✅ 멀티레이블 전이 행렬 연구 (NeurIPS 2022)

이후 연구들은 다중 레이블 전이 행렬 추정기에 대한 추정 오차 경계(estimation error bound)와 실제 전이 행렬을 이용한 통계적으로 일관된 알고리즘의 일반화 오차 경계(generalization error bound)를 이론적으로 도출하고 있다.

#### ✅ 유사도+비유사도 레이블 연구 (ECML 2021)

노이즈 유사(S) 및 비유사(D) 레이블 쌍에서 분류기를 학습하는 방법을 연구하며, 두 가지 현실적인 노이즈 모델하에서 상세히 분석하고 두 가지 알고리즘을 제안한다.

---

### 4-2. 미래 연구 시 고려할 점

#### 🔬 이론 측면
1. **인스턴스 의존 노이즈(Instance-dependent Noise) 확장**: 본 논문은 클래스 의존 노이즈를 가정한다. 실제 환경에서는 인스턴스별로 노이즈 확률이 다를 수 있으므로, 인스턴스 의존 전이 행렬로의 확장이 중요한 연구 방향이다.
2. **전이 행렬 추정 오차의 영향 분석**: 전이 행렬 추정 오차가 최종 분류기 성능에 미치는 영향을 정량화하는 이론적 분석이 필요하다.
3. **비균형 데이터(Class Imbalance) 설정**: Theorem은 불균형 클래스 설정으로 쉽게 확장될 수 있으며, 각 $T_{c,ij}$에 계수 $n_i$ (i번째 클래스의 인스턴스 수)를 곱함으로써 조정 가능하다.

#### 🔬 실용 측면
1. **확장성(Scalability)**: 페어와이즈 데이터 생성은 $O(n^2)$ 복잡도를 가지므로, 대규모 데이터셋에 대한 효율적인 샘플링 전략 연구가 필요하다.
2. **자기 지도 학습(Self-Supervised Learning)과의 결합**: 페어와이즈 방식이 지도 메트릭 학습과 비지도 대조 학습에서 강력한 잠재력을 보이고 있으며, 이는 자연스럽게 노이즈 레이블 문제와 결합될 수 있음을 시사한다.
3. **LLM 시대의 노이즈 레이블**: 데이터셋과 대형 언어 모델이 전례 없는 속도로 커지면서 지도 학습 데이터의 품질을 개선하기 위한 자동화 도구의 필요성이 증가하고 있다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도/학회 | 핵심 아이디어 | 본 논문과의 관계 |
|------|-----------|--------------|-----------------|
| **Multi-Class Classif. from Noisy-Sim. Data** (Wu et al.) | 2020, arXiv | 노이즈 유사도 레이블에서 다중 클래스 분류, 전이 행렬 추정 | **본 논문** |
| **Class2Simi** (Wu et al.) | 2021, ICML | 클래스 레이블 → 유사도 레이블 변환으로 노이즈율 감소 보장 | 역방향 접근: 클래스→유사도 변환으로 확장 |
| **Learning from Noisy Similar & Dissimilar Data** | 2021, ECML | S/D 쌍 레이블 노이즈 처리, 두 가지 노이즈 모델 분석 | 유사도+비유사도 레이블로 확장 |
| **Learning from Noisy Pairwise Similarity & Unlabeled Data** (JMLR 2022) | 2022, JMLR | noisy similar 쌍 + 비레이블 데이터(nSU) 활용 | 비레이블 데이터를 추가 활용하는 확장 |
| **Multi-Label Noise Transition Matrix** | 2022, NeurIPS | 다중 레이블 설정으로 전이 행렬 일반화 | 전이 행렬 추정 아이디어를 멀티레이블로 확장 |
| **Detecting & Rectifying Noisy Labels** | 2025, arXiv | 유사도 기반 노이즈 탐지 및 보정 자동화 | 유사도 기반 노이즈 처리 흐름의 최신 응용 |

Class2Simi는 클래스 레이블에서 유사도 레이블로 변환 시 유사도 레이블 자체가 노이즈에 강인하기 때문에 노이즈율이 감소함을 이론적으로 증명한다는 점에서, 본 논문의 핵심 통찰을 역방향으로 활용한 중요한 후속 연구이다.

---

## 📚 참고 자료 출처

1. **본 논문 원문**: Wu et al., *"Multi-Class Classification from Noisy-Similarity-Labeled Data"*, arXiv:2002.06508, 2020. — https://arxiv.org/abs/2002.06508
2. **DeepAI 논문 페이지**: https://deepai.org/publication/multi-class-classification-from-noisy-similarity-labeled-data
3. **Class2Simi (후속 연구)**: Wu et al., *"Class2Simi: A Noise Reduction Perspective on Learning with Noisy Labels"*, ICML 2021. — https://arxiv.org/pdf/2006.07831 / https://icml.cc/virtual/2021/spotlight/9642
4. **Learning from Noisy Similar and Dissimilar Data**: arXiv:2002.00995 / Springer ECML 2021. — https://arxiv.org/pdf/2002.00995 / https://link.springer.com/chapter/10.1007/978-3-030-86520-7_15
5. **Learning from Noisy Pairwise Similarity and Unlabeled Data**: JMLR 2022. — https://www.jmlr.org/papers/v23/21-0946.html
6. **Multi-Label Noise Transition Matrix Estimation**: NeurIPS 2022 Spotlight. — https://arxiv.org/html/2309.12706
7. **Detecting and Rectifying Noisy Labels (Similarity-Based)**: arXiv 2025. — https://arxiv.org/html/2509.23964v2
8. **Awesome-Learning-with-Label-Noise (GitHub 큐레이션 목록)**: https://github.com/subeeshvasu/Awesome-Learning-with-Label-Noise
9. **Learning from Similarity-Confidence Data**: arXiv:2102.06879 — https://arxiv.org/pdf/2102.06879
10. **Instance-Dependent Label-Noise under Structural Causal Models**: NSF PAR — https://par.nsf.gov/servlets/purl/10380992

> ⚠️ **정확도 안내**: 논문의 상세 수식 일부(특히 손실 함수 구체식 및 실험 수치)는 공개된 PDF 원문 전체를 직접 확인하는 것을 권장드립니다. 수식 중 일반화 오차 경계의 구체적 형태는 Class2Simi(ICML 2021)의 Theorem 3에서 인용하였으며, 원본 논문 arXiv:2002.06508과 관련 내용이 공유됩니다.
