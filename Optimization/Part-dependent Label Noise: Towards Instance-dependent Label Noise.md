# Part-dependent Label Noise: Towards Instance-dependent Label Noise

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 다음과 같습니다:

> **인간은 객체를 부분(parts)으로 분해하여 인식하므로, 어노테이터도 인스턴스 전체가 아닌 부분에 기반하여 레이블을 부여한다. 따라서 실제 레이블 노이즈는 인스턴스 의존적(instance-dependent)이 아니라 부분 의존적(part-dependent)으로 근사할 수 있다.**

### 주요 기여 (Contributions)

1. **새로운 노이즈 모델 제안**: Part-dependent Label Noise (PDN) 모델 - CCN과 IDN 사이의 "중간" 모델
2. **수학적으로 학습 가능한 전이 행렬 근사법**: NMF 기반 부품 분해 + 앵커 포인트 활용
3. **심리학/생리학적 근거에 기반한 이론적 정당화**: 인간 인지 과학에서 영감을 받은 가정 설정
4. **우수한 실험 성능**: 50% 노이즈율에서 CIFAR-10 기준 최고 기준선 대비 약 10% 정확도 향상

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능

### 2.1 해결하고자 하는 문제

#### 레이블 노이즈의 세 가지 기존 모델

| 모델 | 정의 | 한계 |
|------|------|------|
| RCN (Random Classification Noise) | 일정한 확률로 무작위 플립 | 너무 단순 |
| CCN (Class-Conditional Noise) | 클래스에 따라 플립률 결정 | 인스턴스별 차이 무시 |
| IDN (Instance-Dependent Noise) | 인스턴스마다 개별 플립률 | **비식별성(non-identifiability) 문제** |

기존 IDN 모델의 핵심 문제:

$$T_{ij}(\boldsymbol{x}) = \Pr(\bar{Y} = j \mid Y = i, X = \boldsymbol{x})$$

위 전이 행렬 $T(\boldsymbol{x})$는 각 인스턴스마다 다른 행렬을 추정해야 하므로, **노이즈 데이터만으로는 비식별(ill-posed)** 문제가 발생합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 부분 기반 표현 학습 (NMF)

데이터 행렬 $\boldsymbol{X} = [\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n] \in \mathbb{R}^{d \times n}$에 대해 다음 최적화 문제를 풀어 부분 행렬 $W$와 결합 파라미터 $\boldsymbol{h}(\boldsymbol{x}_i)$를 학습합니다:

$$\min_{W \in \mathbb{R}^{d \times r},\ \boldsymbol{h}(\boldsymbol{x}_i) \in \mathbb{R}^r_+,\ \|\boldsymbol{h}(\boldsymbol{x}_i)\|_1=1} \sum_{i=1}^{n} \|\boldsymbol{x}_i - W\boldsymbol{h}(\boldsymbol{x}_i)\|_2^2 \tag{1}$$

- $W$의 각 열 $W_{\cdot j}$: $j$번째 부분(part)
- $\boldsymbol{h}(\boldsymbol{x}_i)$: 인스턴스 $\boldsymbol{x}_i$를 재구성하기 위한 결합 가중치
- $\|\boldsymbol{h}(\boldsymbol{x}_i)\|_1 = 1$: 유효한 전이 행렬 조건 보장

#### Step 2: 인스턴스 의존적 전이 행렬 근사

핵심 아이디어: **인스턴스 재구성 파라미터 = 전이 행렬 결합 파라미터** (공유 가중치)

$$T(\boldsymbol{x}) \approx \sum_{j=1}^{r} h_j(\boldsymbol{x}) P^j \tag{2}$$

- $P^j \in [0,1]^{c \times c}$: $j$번째 부분에 대한 전이 행렬
- $r$: 부분의 개수
- $\|\boldsymbol{h}(\boldsymbol{x})\|_1 = 1$ 제약으로 결합 행렬도 유효한 전이 행렬이 됨

#### Step 3: 앵커 포인트를 활용한 전이 행렬 학습

앵커 포인트 $\boldsymbol{x}^i$ (클래스 $i$에 확실히 속하는 데이터, 즉 $\Pr(Y=i \mid X=\boldsymbol{x}^i) = 1$):

$$\Pr(\bar{Y} = j \mid X = \boldsymbol{x}^i) = \sum_{k=1}^{c} \Pr(\bar{Y}=j \mid Y=k, X=\boldsymbol{x}^i)\Pr(Y=k \mid X=\boldsymbol{x}^i) = T_{ij}(\boldsymbol{x}^i) \tag{3}$$

클래스 $i$의 앵커 포인트 $(\boldsymbol{x}^i_1, \ldots, \boldsymbol{x}^i_k)$ ($k \geq r$)를 이용하여 부분 전이 행렬 최적화:

$$\min_{P^1, \ldots, P^r \in [0,1]^{c \times c}} \sum_{i=1}^{c} \sum_{l=1}^{k} \left\| T_{i\cdot}(\boldsymbol{x}^i_l) - \sum_{j=1}^{r} h_j(\boldsymbol{x}^i_l) P^j_{i\cdot} \right\|_2^2$$
$$\text{s.t.} \quad \|P^j_{i\cdot}\|_1 = 1, \quad i \in \{1, \ldots, c\},\ j \in \{1, \ldots, r\} \tag{4}$$

#### Step 4: 슬랙 변수를 이용한 전이 행렬 수정

앵커 포인트 가정이 완전하지 않을 때를 대비해 슬랙 변수 $\Delta T$를 도입:

$$\bar{h}'(\boldsymbol{x}) = \arg\max_{i \in \{1,2,\ldots,c\}} \left((T(\boldsymbol{x}) + \Delta T)^\top g\right)_i(\boldsymbol{x}) \tag{11}$$

#### 학습 목적 함수들

**PTD-F (Forward 방식)**:

$$\bar{R}_n(\bar{h}) = \frac{1}{n} \sum_{i=1}^{n} \ell(\bar{h}(\boldsymbol{x}_i), \bar{y}_i) \tag{9}$$

**PTD-R (Reweighting 방식)**:

$$\bar{R}_n(f, \bar{h}) = \frac{1}{n} \sum_{i=1}^{n} \frac{g_{\bar{y}_i}(\boldsymbol{x}_i)}{\bar{h}_{\bar{y}_i}(\boldsymbol{x}_i)} \ell(f(\boldsymbol{x}_i), \bar{y}_i) \tag{10}$$

---

### 2.3 모델 구조 (Algorithm 1)

```
입력: 노이즈 훈련 데이터 Dt, 노이즈 검증 데이터 Dv

Step 1: Dt, Dv로 딥 모델 훈련
Step 2: 훈련된 딥 네트워크로 딥 표현(deep representation) 추출
Step 3: Eq.(1) 최적화 → 부분(W)과 결합 파라미터(h) 학습
Step 4: 앵커 포인트 기반 Eq.(3) → 인스턴스 의존 전이 행렬의 각 행 학습
Step 5: Eq.(4) 최적화 → 부분 의존 전이 행렬 P^1,...,P^r 학습
Step 6: Eq.(2) → 각 인스턴스의 인스턴스 의존 전이 행렬 T(x) 계산

출력: T(x)
```

**네트워크 구조:**
- F-MNIST: ResNet-18
- SVHN, CIFAR-10: ResNet-34
- NEWS: 3개 합성곱 레이어 + FC 레이어
- Clothing1M: ResNet-50 (ImageNet 사전 훈련)

---

### 2.4 성능 향상

#### 합성 노이즈 데이터셋 결과 (주요 비교)

| 데이터셋 | 노이즈율 | 최고 기준선 | PTD-R-V | 향상폭 |
|---------|---------|-----------|---------|-------|
| CIFAR-10 | 50% | ~44% (Joint) | **53.98%** | **~+10%** |
| CIFAR-10 | 40% | ~54.75% (Joint) | **58.62%** | ~+3.87% |
| SVHN | 50% | ~49.02% (T-Revision) | **58.09%** | ~+9% |
| F-MNIST | 50% | ~68.99% (T-Revision) | **75.96%** | ~+7% |
| NEWS | 50% | ~59.29% (T-Revision) | **62.77%** | ~+3.5% |

#### 실세계 노이즈 데이터셋 결과 (Clothing1M)

| 방법 | 정확도 |
|------|-------|
| CE | 68.88% |
| Joint | 70.88% |
| T-Revision | 70.97% |
| **PTD-R-V** | **71.67%** |

---

### 2.5 한계점

1. **앵커 포인트 의존성**: 각 클래스당 최소 $r$개의 앵커 포인트 필요. 앵커 포인트가 부정확하면 전이 행렬 추정 오류 발생
2. **공유 파라미터 가정의 강도**: 인스턴스 재구성 파라미터와 전이 행렬 결합 파라미터를 동일하게 가정하는 것은 강한 가정
3. **계산 복잡도**: 각 인스턴스마다 전이 행렬을 계산해야 하므로 클래스 수와 부분 수가 커질수록 연산 비용 증가
4. **딥 표현의 시각화 한계**: RGB 이미지(SVHN, CIFAR-10 등)에서 학습된 부분의 직접 시각화가 어려움
5. **결합 파라미터에 슬랙 변수 미도입**: 논문에서 미래 작업으로 언급

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상의 핵심 메커니즘

#### (a) 노이즈 모델의 정밀도 향상 → 일반화 기여

기존 CCN 방식의 클래스 의존 전이 행렬은 모든 인스턴스에 동일한 노이즈율을 가정합니다:

$$\Pr(\bar{Y} = j \mid Y = i, X = \boldsymbol{x}) = \Pr(\bar{Y} = j \mid Y = i) \quad \text{(CCN)}$$

반면 PDN은 인스턴스별 전이 행렬을 근사하므로, 개별 인스턴스의 노이즈 특성을 더 정확히 반영:

$$T(\boldsymbol{x}) \approx \sum_{j=1}^{r} h_j(\boldsymbol{x}) P^j \quad \text{(PDN)}$$

**정밀한 노이즈 모델링 → 손실 함수 보정 정확도 향상 → 클린 분포에 가까운 분류기 학습 → 일반화 성능 향상**

#### (b) 전이 행렬의 유효성 보장

$\|\boldsymbol{h}(\boldsymbol{x})\|_1 = 1$ 및 $h_j(\boldsymbol{x}) \geq 0$ 제약은 결합된 전이 행렬이 항상 유효한 확률 행렬임을 보장합니다. 이는 클린 사후확률과 노이즈 사후확률 사이의 일관된 연결을 유지하여 **분류기 일관성(classifier consistency)** 을 확보합니다.

#### (c) 슬랙 변수를 통한 강건성

슬랙 변수 $\Delta T$는 부분 의존 전이 행렬의 부정확한 추정을 보완하여 모델이 더 유연하게 실제 노이즈에 적응할 수 있게 합니다:

$$T_{\text{eff}}(\boldsymbol{x}) = T(\boldsymbol{x}) + \Delta T$$

이를 통해 과적합(overfitting) 위험을 줄이고 일반화를 개선합니다.

#### (d) 높은 노이즈율에서의 강건성

실험 결과를 보면, 노이즈율이 높아질수록 PTD 방법의 이점이 더 커집니다. 이는 **심한 노이즈 환경에서도 올바른 결정 경계를 학습**할 수 있음을 시사합니다.

논문의 ablation study (Figure 2)에서 확인된 사실:
- 부분 의존 전이 행렬의 근사 오차가 클래스 의존 방법보다 현저히 낮음 ($\ell_1$ 노름 기준)
- 부분의 수 $r$에 무감(insensitive)하여 하이퍼파라미터 튜닝 없이도 안정적 성능

#### (e) 딥 표현 활용

NMF를 원시 데이터가 아닌 **딥 네트워크의 표현 공간**에 적용함으로써, 더 의미론적(semantic)으로 일관된 부분을 추출합니다. 이는 노이즈 전이 행렬의 품질을 높여 일반화에 기여합니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

#### (a) 이론적 영향

1. **새로운 레이블 노이즈 분류 체계 제시**: PDN은 CCN과 IDN 사이의 중간 모델로서, 향후 연구자들이 더 세밀한 노이즈 계층을 탐색하는 기반 제공
2. **인지과학과 머신러닝의 융합**: 인간 인지 메커니즘에서 노이즈 모델 설계 영감을 얻는 새로운 연구 방향 제시
3. **전이 행렬 추정 방법론 발전**: 부분 기반 분해를 통한 고차원 전이 행렬 추정 패러다임 확립

#### (b) 실용적 영향

- **대규모 실세계 노이즈 데이터셋 활용 가능성 증대**: Clothing1M 등에서의 성능 향상으로 실제 산업 적용 가능성 제고
- **클린 데이터 없이도 작동**: 클린 검증 데이터 없이 순수 노이즈 데이터만으로 학습 가능 (실용성 ↑)

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래는 논문에서 직접 인용되었거나, PDN과 비교 가능한 동시대 및 이후 연구들입니다:

#### PDN 논문 내에서 비교된 2020년 연구들

| 논문 | 방법 | 특징 | PDN과의 관계 |
|------|------|------|-------------|
| **DivideMix** (Li et al., ICLR 2020) | 반지도 학습 기반 | GMM으로 클린/노이즈 분리 | PDN과 상호 보완 가능 |
| **Peer Loss** (Liu & Guo, ICML 2020) | 노이즈율 무지식 손실 | 노이즈율 추정 불필요 | PDN과 다른 접근 |
| **SIGUA** (Han et al., ICML 2020) | 잊기(forgetting) 활용 | 점진적 레이블 정제 | PDN의 전이 행렬 방식과 다름 |
| **Dual-T** (Yao et al., NeurIPS 2020) | 이중 전이 행렬 | 추정 오차 감소 | PDN과 결합 가능성 높음 |
| **Cheng et al.** (ICML 2020) | 경계 있는 IDN | 노이즈율 상한 가정 | PDN의 이론적 비교 대상 |

#### 2020년 이후 주목할 만한 관련 연구 (논문에서 직접 인용하지 않은 최신 연구)

> ⚠️ **주의**: 아래 2020년 이후 연구들은 제공된 논문 PDF에 직접 인용되지 않았으며, 필자의 지식에 기반합니다. 정확도에 다소 불확실성이 있을 수 있습니다.

| 논문 | 학회/연도 | 핵심 아이디어 | PDN과의 비교 |
|------|----------|------------|------------|
| **CORES²** (Cheng et al.) | ICML 2021 | 확신 점수 기반 IDN 학습 | PDN의 앵커 포인트와 유사한 신뢰 데이터 활용 |
| **Instance-dependent Label Noise** (Berthon et al.) | ICML 2021 | 신뢰 점수로 IDN 가능하게 | PDN과 직접 경쟁 관계 |
| **Robust Early-Learning** | - | 초기 학습의 강건성 | PDN의 딥 표현 학습과 연관 |

---

### 4.3 앞으로 연구 시 고려할 점

#### (a) 가정의 검증 및 완화

1. **공유 파라미터 가정 완화**:
   - 인스턴스 재구성과 전이 행렬 재구성의 파라미터가 동일하다는 가정은 편의상 설정된 것
   - 두 파라미터를 독립적으로 학습하되, 정규화를 통해 유사성을 유도하는 방법 연구 필요

2. **앵커 포인트 의존성 해소**:
   - 완전히 앵커 포인트가 없는 환경에서의 PDN 학습 방법론 개발
   - Self-training 또는 준지도학습 기법과의 결합

#### (b) 방법론적 확장

3. **부분 학습의 다양화**:
   - NMF 외에 **희소 오토인코더**, **베타-VAE** 등 딥 생성 모델 기반 부분 추출
   - 주의 메커니즘(Attention)을 활용한 동적 부분 가중치 학습

4. **다중 모달리티 확장**:
   - 텍스트, 이미지, 오디오 등 다중 모달 데이터에서의 PDN 적용
   - 부분의 개념을 멀티모달 맥락으로 일반화

5. **슬랙 변수 개선**:
   - 결합 파라미터에도 슬랙 변수 도입 (논문에서 미래 작업으로 언급)
   - 베이지안 접근을 통한 불확실성 정량화

#### (c) 이론적 발전

6. **일반화 오차 경계 (Generalization Bound) 분석**:
   - 현재 논문은 경험적 결과 중심. 이론적 일반화 보장 제공 필요
   - 부분 수 $r$, 앵커 포인트 수, 노이즈율 $\tau$에 따른 바운드 도출

7. **PDN의 식별성(Identifiability) 이론 강화**:
   - 어떤 조건 하에서 PDN 모델이 완전히 식별 가능한지 이론적 분석

#### (d) 실용적 고려사항

8. **계산 효율성**:
   - 대규모 데이터셋에서 각 인스턴스별 전이 행렬 계산의 확장성(scalability) 문제
   - 클러스터링 기반 근사 또는 그래프 신경망 활용

9. **도메인 적응 및 연합 학습 (Federated Learning)**:
   - 도메인 시프트 환경에서 PDN 모델의 강건성 연구
   - 분산 학습 환경에서의 노이즈 레이블 처리

10. **자기 지도 학습(Self-supervised Learning)과의 통합**:
    - 대규모 사전 훈련 모델(CLIP, DINO 등)에서 추출한 표현으로 부분 학습 품질 향상

---

## 참고 자료

**주요 참고 논문 (논문 PDF에서 직접 인용된 문헌들):**

1. Xia, X., Liu, T., Han, B., et al. **"Part-dependent Label Noise: Towards Instance-dependent Label Noise"**. *NeurIPS 2020*. arXiv:2006.07836v2
2. Patrini, G., et al. **"Making deep neural networks robust to label noise: A loss correction approach"**. *CVPR 2017*. [Forward 방법]
3. Liu, T., & Tao, D. **"Classification with noisy labels by importance reweighting"**. *IEEE TPAMI 2016*. [Reweight 방법]
4. Xia, X., et al. **"Are anchor points really indispensable in label-noise learning?"**. *NeurIPS 2019*. [T-Revision]
5. Han, B., et al. **"Co-teaching: Robust training of deep neural networks with extremely noisy labels"**. *NeurIPS 2018*.
6. Lee, D.D., & Seung, H.S. **"Learning the parts of objects by non-negative matrix factorization"**. *Nature 1999*.
7. Cheng, J., et al. **"Learning with bounded instance-and label-dependent label noise"**. *ICML 2020*.
8. Berthon, A., et al. **"Confidence scores make instance-dependent label-noise learning possible"**. arXiv:2001.03772, 2020.
9. Li, J., et al. **"DivideMix: Learning with noisy labels as semi-supervised learning"**. *ICLR 2020*.
10. Yao, Y., et al. **"Dual T: Reducing estimation error for transition matrix in label-noise learning"**. *NeurIPS 2020*.
