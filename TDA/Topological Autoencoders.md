# Topological Autoencoders

## 1. 논문의 핵심 주장과 주요 기여

Moor et al.의 “Topological Autoencoders(TopoAE)”는 오토인코더의 잠재공간(latent space)이 입력 데이터의 위상 구조(특히 연결성, 구멍, 중첩된 매니폴드 등)를 가능한 한 보존하도록 학습시키는 새로운 위상 손실(topological loss)을 제안한다.[^1][^2]

핵심 주장은 다음과 같다:[^3][^1]

- 입력 공간과 잠재공간에 대해 각각 퍼시스턴트 호몰로지(persistent homology)를 계산하고, 두 공간의 "위상적으로 중요한" 거리(edge)들을 정렬하여 손실로 사용하면, 잠재코드가 데이터 매니폴드의 다중 스케일 연결 구조를 잘 보존하게 만들 수 있다.
- 이 손실은 미분 가능하도록 구성되며, 미니배치 수준의 퍼시스턴트 호몰로지 근사에 대해 안정성 보장을 제공한다.
- 합성 SPHERES 데이터셋(고차원 중첩 구형 매니폴드)과 실제 이미지 데이터셋(MNIST, Fashion‑MNIST, CIFAR‑10)에서 TopoAE는 낮은 재구성 오차를 유지하면서도, 기존 차원 축소 기법보다 위상 구조를 더 잘 보존하는 잠재 표현을 학습한다.[^1]

주요 기여를 정리하면:[^3][^1]

- 입력–잠재 공간 위상 정렬을 위한 새로운 위상 손실 $\(L_t\)$ 제안(0차 퍼시스턴트 호몰로지 기반, 향후 고차원 확장 가능성을 논의).
- 미니배치 서브샘플링 하에서 퍼시스턴스 다이어그램 근사 안정성에 대한 이론적 결과(하우스도르프 거리와 병목거리 사이의 확률적 상계).
- 합성/실세계 데이터에 대한 정성·정량 실험을 통해, TopoAE가 밀도 보존(KL divergence), 연속성(continuity), 신뢰도(trustworthiness) 지표에서 경쟁력 있는 성능을 보이며, 특히 복잡한 매니폴드(중첩 구) 구조를 유일하게 올바르게 재현함을 보여줌.

[^1]
[^3]

## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 문제 설정

기존 오토인코더 및 대부분의 차원 축소 기법은 국소 거리나 분포를 잘 보존하더라도, 데이터 매니폴드의 전역 위상 구조(연결 성분 개수, 구멍 수, 중첩 구조 등)를 보존한다는 보장은 없다. 예를 들어, t‑SNE나 UMAP은 시각적으로 잘 분리된 군집을 만들지만, 실제로는 중첩된 매니폴드의 포개짐이나 구멍의 존재를 찢어 버릴 수 있다.[^1]

논문이 겨냥하는 핵심 문제는 다음과 같다:[^4][^1]

- 비선형 차원 축소와 표현 학습에서, 잠재공간이 데이터의 위상 구조를 파괴하지 않고 보존하도록 제약을 거는 일반적인 방법이 부족하다.
- 퍼시스턴트 호몰로지를 통한 위상 특성은 본질적으로 이산적이어서, 이를 통해 얻은 특성에 대해 역전파를 수행하기 어렵다(미분 불가능성 문제).

[^1]
[^4]

### 2.2 퍼시스턴트 호몰로지와 Vietoris–Rips 복합체

데이터 포인트 집합 $\(X = \{x_1,\dots,x_n\} \subset \mathbb{R}^d\)$ 와 거리함수 $\(\mathrm{dist}\)$ 에 대해, 스케일 $\(\epsilon\)$ 에서의 Vietoris–Rips 복합체 $\(R_\epsilon(X)\)$ 는 모든 점 집합 $\(\{x_{i_0},\dots,x_{i_k}\}\)$ 가 $\(\mathrm{dist}(x_{i_p},x_{i_q}) \le \epsilon\)$ 을 만족하면 단체로 포함하는 단순체 복합체이다.[^1]

스케일 $\(\epsilon\)$ 을 증가시키며 얻는 필트레이션 $\(R_{\epsilon_0}(X) \subseteq R_{\epsilon_1}(X) \subseteq \dots\)$ 에서, d차 호몰로지 군 $\(H_d(R_\epsilon(X))\)$ 의 생성과 소멸 시점을 기록한 것이 d차 퍼시스턴스 다이어그램 $\(D_d\)$ 이다.[^1]

병목거리(bottleneck distance)는 두 퍼시스턴스 다이어그램 $\(D, D'\)$ 사이의 거리를

$$
 d_B(D, D') = \inf_{\eta : D \to D'} \sup_{x \in D} \lVert x - \eta(x) \rVert_\infty
$$

와 같이 정의하고, 이는 데이터에 대한 작은 섭동에 대해 안정적이다는 것이 알려져 있다.[^1]

[^1]

### 2.3 위상 손실의 기본 아이디어

오토인코더 $\(h \circ g\)$ 에서, 인코더 $\(g : \mathcal{X} \to \mathcal{Z}\)$ , 디코더 $\(h : \mathcal{Z} \to \mathcal{X}\)$ 를 두고, 미니배치 $\(X\)$ 에 대한 잠재 코드 $\(Z = g(X)\)$ 를 얻는다. 데이터 공간과 잠재공간 각각에서 Vietoris–Rips 복합체를 만들고 0차 퍼시스턴트 호몰로지를 계산하여, 위상적으로 중요한 엣지 인덱스 집합(퍼시스턴스 페어링) $\(\pi_X, \pi_Z\)$ 와 이에 대응하는 거리 벡터 $\(A_X[\pi_X], A_Z[\pi_Z]\)$ 를 얻는다.[^1]

핵심 관찰은 다음과 같다:[^1]

- 0차 호몰로지의 퍼시스턴스 페어링은 사실상 최소신장트리(MST)의 엣지 집합과 일치하며, 이는 "연결 성분들이 합쳐지는" 중요한 거리들을 의미한다.
- 퍼시스턴트 호몰로지 계산은, 거리 행렬의 특정 원소(위상적으로 중요한 엣지)를 선택하는 연산으로 볼 수 있으며, 고유한 거리(심볼릭 섭동으로 보장)라는 가정 하에 이 선택은 국소적으로 상수이므로, 선택된 거리 값은 인코더 파라미터에 대해 미분 가능하다.

이를 이용해, 데이터 공간과 잠재공간 간에 선택된 엣지 거리들을 정렬(alignment)하는 손실을 정의한다.[^1]

[^1]

### 2.4 제안된 최종 목적함수와 수식

기본 오토인코더의 재구성 손실은

$$
 L_r(X, h(g(X))) = \frac{1}{m} \sum_{i=1}^m \lVert x_i - h(g(x_i)) \rVert^2
$$

(예: MSE)로 두고, 여기에 위상 손실 $\(L_t\)$ 을 가중합으로 추가한다.[^1]

전체 손실:

$$
 L = L_r(X, h(g(X))) + \lambda L_t(A_X, A_Z, \pi_X, \pi_Z),
$$

여기서 $\(\lambda \in \mathbb{R}\)$ 는 위상 정규화 강도를 제어하는 하이퍼파라미터이다.[^1]

위상 손실은 두 개의 방향성 성분으로 분해된다:[^1]

- 데이터→잠재 공간 정렬 손실

$$
 L_{X \to Z} := \frac{1}{2} \big\| A_X[\pi_X] - A_Z[\pi_X] \big\|_2^2,
$$

- 잠재→데이터 공간 정렬 손실

$$
 L_{Z \to X} := \frac{1}{2} \big\| A_Z[\pi_Z] - A_X[\pi_Z] \big\|_2^2,
$$

따라서 최종 위상 손실은

$$
 L_t = L_{X \to Z} + L_{Z \to X}
$$

으로 정의된다.[^1]

여기서 $\(A_X[\pi_X]\)$ 는 데이터 공간 거리행렬 $\(A_X\)$ 에서, 퍼시스턴스 페어링 $\(\pi_X\)$ 에 해당하는 엣지의 거리들을 모아 만든 벡터(크기 $\(|\pi_X|\))$ 이며, $\(A_Z[\pi_X]\)$ 는 동일한 엣지 인덱스를 잠재공간 거리행렬 $\(A_Z\)$ 에 적용해 얻은 거리 벡터이다[^1]. 잠재공간에서도 마찬가지다.

이 손실의 직관은 다음과 같다:[^1]

- 데이터 공간에서 위상적으로 중요한 엣지(퍼시스턴스 페어링에 의해 선택된 거리)가 잠재공간에서도 유사한 거리 관계를 유지하도록 강제한다 $(\(L_{X \to Z}\))$ .
- 반대로, 잠재공간에서 위상적으로 중요한 엣지가 데이터 공간에서도 일치하도록 강제하여, 잠재공간에서 새롭게 생기는 위상 구조가 원 데이터와 동조되도록 한다 $(\(L_{Z \to X}\))$ .

[^1]

### 2.5 미분 가능성 및 그라디언트

인코더 파라미터 $\(\theta\)$ 에 대해, $\(L_{X \to Z}\)$ 의 그라디언트는

$$
 \rho := A_X[\pi_X] - A_Z[\pi_X] \in \mathbb{R}^{|\pi_X|}
$$

로 두고,

$$
 \frac{\partial L_{X \to Z}}{\partial \theta}
 = - \rho^\top \left( \frac{\partial A_Z[\pi_X]}{\partial \theta} \right)
 = - \sum_{i=1}^{|\pi_X|} \rho_i 
   \frac{\partial A_Z[\pi_X]_i}{\partial \theta}
$$

과 같이 표현된다.[^1]

- 데이터 공간 거리 $\(A_X\)$ 는 인코더와 무관하므로 $\(\partial A_X/\partial \theta = 0\)$ 이다.
- 퍼시스턴스 페어링(어떤 엣지가 선택되는지)은 거리값이 고유하다는 가정 하에서, 거리가 연속적으로 약간 변할 때 국소적으로 변하지 않으므로(안정성 정리), 각 업데이트 스텝에서 손실은 인코더 파라미터에 대해 미분 가능하다고 볼 수 있다.[^1]

이는 퍼시스턴스 다이어그램 전체를 대상으로 한 복잡한 최적화(예: Wasserstein 거리 최소화)를 피하면서도, 위상적으로 의미 있는 엣지 집합에 대해 손실을 정의하여 역전파를 가능하게 만드는 핵심 기술이다.[^1]

[^1]

## 3. 모델 구조와 학습 설정

### 3.1 기본 오토인코더 구조

TopoAE 자체는 특정 네트워크 구조에 종속되지 않지만, 논문에서는 실험을 위해 다음과 같은 단순한 구조를 사용한다:[^1]

- 합성 SPHERES 데이터셋: 2개의 은닉층(각 32 유닛)을 가진 MLP 인코더–디코더, 잠재 차원 2.
- MNIST / Fashion‑MNIST / CIFAR‑10: Deep Autoencoder(Hinton & Salakhutdinov 스타일)를 변형한 MLP 구조로, 인코더는 1000–500–250–2, 디코더는 2–250–500–1000 형태, ReLU + BatchNorm, 출력층은 $\(\tanh\)$ 를 사용하여 입력 스케일(−1~1)에 맞춤.[^1]

모든 오토인코더는 Adam 옵티마이저와 weight decay $\(10^{-5}\)$ 로 학습하고, 재구성 손실은 MSE를 사용한다.[^1]

[^1]

### 3.2 미니배치 기반 퍼시스턴트 호몰로지

정확한 Vietoris–Rips 복합체와 퍼시스턴트 호몰로지 계산은 $\(O(m^2 \alpha(m^2))\)$ 이상의 복잡도를 갖기 때문에, 전체 데이터셋에 대해 반복 수행하기 어렵다. 논문은 다음 전략을 사용한다:[^1]

- 학습은 미니배치 크기 $\(m\)$ (예: 16–128)에서 진행하며, 각 미니배치에 대해 데이터·잠재공간의 거리행렬 $(\(m \times m\))$ 을 계산하고 퍼시스턴트 호몰로지를 구한다.
- 이때 얻은 퍼시스턴스 다이어그램과 페어링을 통해 위상 손실을 계산하고 역전파한다.

미니배치 기반 근사에 대해, 원 전체 데이터 $\(X\)$ 의 퍼시스턴스 다이어그램 $\(D_X\)$ 와 서브샘플 $\(X^{(m)}\)$ 의 다이어그램 $\(D_{X^{(m)}}\)$ 사이의 병목거리 $\(d_B(D_X, D_{X^{(m)}})\)$ 를 하우스도르프 거리 상계로 제한하는 정리를 제시한다.[^1]

정리 1 (요지):

$$
 \mathbb{P}\big( d_B(D_X, D_{X^{(m)}}) > \epsilon \big)
 \le \mathbb{P}\big( d_H(X, X^{(m)}) > 2\epsilon \big),
$$

여기서 $\(d_H\)$ 는 하우스도르프 거리이다.[^1]

또한, 거리 행렬의 통계적 모델 가정 하에서 $\(\mathbb{E}[d_H(X, X^{(m)})]\)$ 에 대한 상계를 제시해, $\(m \to n\)$ 일 때 0으로 수렴함을 보인다. 이는 미니배치 퍼시스턴트 호몰로지가 전체 데이터의 위상 구조를 적절히 근사함을 시사한다.[^1]

[^1]

## 4. 성능 향상, 한계, 일반화 성능 관점

### 4.1 정량적 성능: 위상·기하 품질과 재구성 오차

TopoAE는 다음 지표들에서 기존 차원 축소·오토인코더와 비교된다:[^1]

- KL divergence 기반 밀도 보존 지표 $\(\mathrm{KL}_\sigma\)$ : 입력·잠재공간에서 Gaussian kernel 기반 density estimator $\(f^\sigma_X, f^\sigma_Z\)$ 를 만든 뒤, $\(\mathrm{KL}(f^\sigma_X \Vert f^\sigma_Z)\)$ 를 계산한다.
- 비선형 차원축소(NLDR) 품질 지표: 거리 행렬 기반
  - $\(\ell\)$ -RMSE (입력·잠재 거리 분포 간 RMSE)
  - $\(\ell\)$ -MRRE (mean relative rank error)
  - $\(\ell\)$ -Trust, $\(\ell\)$ -Cont (최근접 이웃 보존 정도)
- Data MSE: 재구성 오차(오토인코더 계열에만 해당).

핵심 관찰은:[^1]

- SPHERES 데이터셋에서, TopoAE는 $\(\mathrm{KL}\_{0.01}, \mathrm{KL}\_{0.1}, \mathrm{KL}\_1\)$ 에서 모든 방법 중 최상 또는 준최상 성능을 보이며, 특히 $\(\mathrm{KL}_{0.1}\)$ 를 하이퍼파라미터 탐색 목표로 최소화할 때 밀도 보존이 가장 우수하다.
- 시각적으로, SPHERES에서 유일하게 중첩된 큰 구와 작은 구들의 nested 관계를 정확하게 재현하며, t‑SNE·UMAP은 큰 구를 찢어 여러 조각으로 흩어버린다.[^1]
- Fashion‑MNIST, MNIST, CIFAR‑10에서도, TopoAE는 PCA/AE/UMAP 등과 유사하거나 더 나은 $\(\mathrm{KL}_\sigma\)$ 및 연속성 값을 보이면서, 재구성 오차(Data MSE)는 표준 AE와 거의 동일한 수준(차이는 매우 작음)을 유지한다.[^1]

이는 위상 손실이 재구성 품질을 크게 해치지 않으면서도, 잠재공간의 전역 구조를 더 잘 정렬한다는 것을 의미한다.

[^1]

### 4.2 정성적 성능: 잠재공간 시각화와 해석 가능성

시각화 결과에서, TopoAE의 잠재공간은 다음 특성을 보인다:[^1]

- SPHERES: 중첩된 구형 매니폴드가 2차원 잠재공간에서도 중첩 구조를 유지한다(큰 외곽 구와 내부 구들의 관계가 잘 보존됨). 다른 기법은 외곽 구를 찢거나 내부 구들을 뭉개서 위상을 파괴한다.
- Fashion‑MNIST: TopoAE는 UMAP과 유사하게 클래스들이 의미 있는 곡면 구조를 따른 채 배치되며, 단순히 완전히 분리된 점 구름이 아니라 연결된 매니폴드 상에 클래스들이 배치된다는 점에서 더 해석 가능한 구조를 제공한다.
- MNIST: t‑SNE/UMAP이 클래스간 관계를 무시하고 강하게 분리하는 반면, TopoAE는 PCA와 비슷하게 클래스 간 위상적 관계(예: 4와 9의 유사성)를 어느 정도 유지하면서도 비선형 구조를 포착한다.

이러한 구조는 다운스트림 작업(클러스터링, 이상치 탐지, 순서형 구조 파악 등)에서 일반화 성능과 해석 가능성 향상에 기여할 수 있다.[^5][^6]

[^1]
[^5]
[^6]

### 4.3 한계와 이론적 제약

저자들은 다음과 같은 한계를 명시하거나 논의한다:[^1]

- 0차 퍼시스턴트 호몰로지에 제한: 구현은 주로 0차(연결 성분) 위상 정보에 집중하며, 1차 이상(사이클, 구멍 등)을 사용하면 계산 복잡도가 기하급수적으로 증가한다.
- 손실이 0이라고 해서 두 퍼시스턴스 다이어그램이 완전히 동일하다는 보장은 없다(역은 성립하지 않을 수 있음). 실험에서는 문제를 관찰하지 못했지만, 이론적 정밀 분석은 향후 과제로 남겨 둔다.[^7][^1]
- 미니배치 기반 위상 근사는 대규모 데이터셋에서의 전역 위상을 정밀히 반영하지 못할 수 있으며, 배치 크기가 작을수록 근사가 거칠어진다.
- 퍼시스턴트 호몰로지 계산의 시간·메모리 비용 때문에, batch size와 위상 차원을 크게 늘리기 어렵고, 고차원 위상 구조(예: $\(H_1, H_2\)$ )까지 포함한 상용 수준 모델로의 확장은 별도의 근사화 기법(예: TopoAE++, 근사 PH, 병렬화)이 필요하다.[^8][^7]

[^1]
[^8]
[^7]

### 4.4 일반화 성능 향상 가능성과 이론적 연결점

TopoAE 논문 자체는 일반화 오차의 엄밀한 상계나 통계적 학습 이론을 제시하지는 않지만, 위상 보존이 일반화에 긍정적일 수 있는 여러 근거를 제공한다:[^9][^1]

- 매니폴드 가설 하에서, 좋은 표현은 원 데이터 매니폴드와 위상 동형(homeomorphic)에 가까운 잠재 매니폴드를 형성해야 하며, TopoAE는 이를 직접적인 손실로 구현한 사례이다.
- TopoAE의 잠재공간은 t‑SNE처럼 과도하게 군집을 찢는 대신, 매니폴드의 연결 구조를 보존하여, 학습 데이터 주변 이외의 영역에서도 더 온전한 매니폴드 구조를 유지한다. 이는 새로운 샘플에 대한 내삽/보간 성질을 향상시켜 일반화에 기여할 수 있다.[^10][^5]

동시에, 후속 이론 연구들은 autoencoder 일반화와 위상 제약 사이의 연결을 부분적으로 뒷받침한다:

- Chart Autoencoders(Deep nonparametric estimation of intrinsic data structures)는 적절한 네트워크 구조와 매니폴드 가정 하에서, 제곱 일반화 오차가 $\(n^{-2/(d+2)} \log^4 n\)$ 속도로 수렴함을 보여주며, 이 수렴율은 매니폴드의 내재 차원 $\(d\)$ 에만 의존하고 주변 차원에는 거의 의존하지 않음을 보인다. 이는 매니폴드 구조(따라서 위상)를 잘 추정하는 autoencoder가 빠른 일반화 수렴을 이룰 수 있음을 시사한다.[^11][^12]
- Geometry Regularized Autoencoders 및 관련 연구는 퍼시스턴트 호몰로지 기반 위상 손실이, 잠재공간에서 데이터 매니폴드를 더 잘 보존하여 오버피팅을 줄이고, 데이터 의존적 정규화로 작용함을 보고한다.[^13]
- "Ensuring Topological Data-Structure Preservation under Autoencoder Compression"은 Gauss–Legendre 노드를 이용한 야코비안 정규화로, 오토인코더가 초기 데이터 매니폴드를 위상 동형으로 재매핑(일대일 재임베딩)하도록 보장하며, 이는 위상 보존이 일반화된 표현을 낳을 수 있음을 이론적으로 뒷받침한다.[^14]

정리하면, TopoAE 자체는 일반화 오차에 대한 직접적 이론은 없지만, 위상 보존이라는 강한 구조적 정규화가 잠재공간을 더 의미 있는 매니폴드로 만들고, 이는 매니폴드 기반 일반화 이론들과 잘 맞물린다.[^11][^14][^13]

[^11]
[^14]
[^13]

## 5. 2020년 이후 관련 최신 연구와 TopoAE의 위치

TopoAE 이후, 위상 정보와 퍼시스턴트 호몰로지를 autoencoder 및 딥러닝 정규화에 사용하는 연구가 크게 확장되었다.

### 5.1 Topological Autoencoders의 직접적인 후속·확장 연구

- Topological Autoencoders++ (TopoAE++)는 TopoAE 손실의 이론 분석을 통해, 0차 퍼시스턴트 호몰로지( $\(\mathrm{PH}^0\)$ )에 대해서는 $\(L_t = 0\)$ 이면 입력·잠재공간의 퍼시스턴스 페어링이 동일함을 보이고, 1차 이상으로의 순진한 확장이 이 성질을 잃는다는 반례를 제시한다. 이를 바탕으로 1차 퍼시스턴트 호몰로지(사이클)를 정확히 보존하는 새로운 손실(“cascade distortion”)을 도입하여, 고차원 데이터의 순환 패턴을 보다 정확히 시각화하는 TopoAE++를 제안한다.[^8][^7]
- 여러 응용 연구(예: TopoReformer, adversarial purification, anomaly detection, COVID‑19 전파 패턴 매핑 등)가 TopoAE 또는 유사한 위상 손실을 방어·탐지·시각화 모듈로 사용하여, 구조 보존이 견고성과 일반화에 도움이 됨을 보인다.[^15][^16][^17][^6]

[^8]
[^7]
[^15]
[^16]
[^17]

### 5.2 위상 정규화와 일반화 성능

- Homological Regularization for Autoencoders는 0차 퍼시스턴트 호몰로지 기반 정규화를 도입하여, 잠재공간이 단위 볼 전체를 고르게 채우도록 유도하고, 이는 VAE류 정규화 없이도 생성 품질을 크게 개선하면서 잠재 표현 품질을 유지함을 보인다. 이 역시 위상 손실이 일반화 가능한 잠재 구조 형성에 도움을 준다는 실증적 증거다.[^18]
- Topology-aware autoencoders for anomaly detection는 위상 priors(구형, 곱공간, 사영공간 등)를 잠재공간에 부여하여 고에너지 물리 이벤트의 이상치 탐지 성능을 향상시키며, 위상 제약이 스푸리어스 재구성 에러를 줄이고 진짜 구조적 이상을 강조함을 보여준다.[^10][^17]
- "Ensuring Topological Data-Structure Preservation under Autoencoder Compression"은 데이터 독립적 위상 보존 정규화(야코비안 샘플링)를 통해, autoencoder가 데이터 매니폴드를 위상적으로 보존하는 재임베딩을 학습하고, 다양한 실제 데이터(MRI 등)에서 신뢰도 높은 저차원 표현을 얻을 수 있음을 시연한다.[^14]

이들 연구는 TopoAE의 아이디어(위상적 구조를 정규화로 사용)를 일반화하여, 잠재공간 구조 제어와 일반화 개선에 활용하고 있다.

[^10]
[^17]
[^14]
[^18]

### 5.3 기타 관련 흐름: 지오메트리·위상 정규화

- Geometry Regularized Autoencoders 등은 리만 기하와 퍼시스턴트 호몰로지 기반 지오메트리 손실을 도입하여, 잠재공간의 곡률과 위상 특성을 제어함으로써, 표현의 일반화와 견고성을 향상시키는 방법을 제안한다.[^13]
- Local distance preserving autoencoders는 연속 kNN 그래프 기반의 로컬 거리 보존 손실을 사용해, 모든 스케일의 위상 특징을 동시에 포착하는 autoencoder 변형을 제안한다. 이는 TopoAE와는 다르게 PH를 직접 쓰지 않지만, "위상·기하 구조 보존"이라는 동일한 목표를 갖는다.[^19]
- 최근 설문 논문들은 "representation regularization or topology‑preserving learning" 범주에서, TopoAE를 대표적인 위상 정규화 기법으로 언급하고, 다양한 NLP·비전·과학응용에서의 활용을 정리한다.[^9][^5]

[^5]
[^9]
[^19]
[^13]

## 6. TopoAE 관점에서 본 일반화 성능 향상 가능성

### 6.1 구조적 정규화로서의 위상 손실

TopoAE의 위상 손실은, 매개변수 공간 상에서 다음과 같은 일반적 정규화 효과를 가진다고 해석할 수 있다:[^14][^1]

- 잠재공간에서 임의의 비위상적 변형(예: 클래스 사이를 불연속적으로 잘라 분리, 매니폴드 찢기)을 억제하고, 입력 매니폴드와 위상 동형에 가까운 잠재 매니폴드만 허용한다.
- 이는 재구성 손실만 최소화할 때 생길 수 있는 병적인 해(예: 극단적으로 꼬인 잠재공간, 단일 지점으로의 붕괴 등)를 배제하는 역할을 하며, 효과적으로 함수 공간의 가용 영역을 줄여 일반화를 돕는 데이터 의존적 정규화이다.

또한, TopoAE의 잠재공간은

- 고밀도 영역 주변에서 국소 거리와 연결 구조를 잘 보존하고,
- 저밀도 영역에서 불필요한 왜곡(예: 과도한 군집 분리)을 피하는 경향을 보여,

새로운 샘플에 대한 보간적 일반화(특히 manifold interior)에서 더 좋은 거동을 할 가능성이 크다.[^10][^14][^1]

[^1]
[^10]
[^14]

### 6.2 이론적 일반화 연구와의 정합성

앞서 인용한 CAE/차트‑오토인코더, Gauss–Legendre 정규화, 지오메트리 정규화 결과들은, 적절한 구조적 제약 아래의 autoencoder가 빠른 일반화 수렴률과 강한 잡음 제거 능력을 보인다는 점을 이론적·실험적으로 보여준다.[^11][^13][^14]

TopoAE는 이러한 구조적 제약의 한 구체적 사례로 볼 수 있으며, 다음과 같은 연구 방향이 의미가 있다:

- TopoAE의 위상 손실을 포함한 autoencoder에 대해, 매니폴드 가설 하에서의 일반화 오차 상계를 도출(예: 병목거리 기준 위상 근사 오차와 재구성 오차를 동시에 포함하는 bound).
- TopoAE++와 같이 1차 이상 퍼시스턴스까지 포함할 때, 위상 보존 강도가 일반화 성능(예: 다운스트림 분류·클러스터링)과 어떻게 상관되는지 체계적으로 분석.

[^11]
[^7]
[^13]

## 7. 향후 연구에 미치는 영향과 연구 시 고려할 점

### 7.1 TopoAE가 연 분야

TopoAE는 다음과 같은 방향에서 후속 연구의 기초가 되었다:[^5][^9][^7]

- **위상 인식 표현 학습**: 딥러닝 모델의 잠재 표현에 위상 priors(연결성, 구멍 개수, 중첩 구조 등)를 명시적으로 부여하는 연구 흐름을 촉발하였다.
- **퍼시스턴트 호몰로지의 미분 가능 사용**: 퍼시스턴스 페어링을 이용해, PH 결과를 거리 행렬의 특정 원소 선택 연산으로 해석하고, 이를 통한 역전파가 가능함을 보여 이후 다양한 differentiable PH 손실(TopoAE++, RTD‑AE, Homological Regularization 등)의 설계에 참고점이 되었다.[^20][^18][^8]
- **위상 기반 견고성·해석 가능성 연구**: adversarial defense(TopoReformer), anomaly detection, GAN 비교(Geometry score 등)에서, 위상 제약과 잠재공간 구조 분석을 결합하는 연구가 활발해졌다.[^21][^15][^10]

[^20]
[^5]
[^15]
[^9]
[^21]
[^8]
[^7]
[^18]

### 7.2 앞으로 연구 시 고려할 점

TopoAE 및 후속 연구를 바탕으로, 향후 연구에서 고려해야 할 주요 포인트는 다음과 같다.

1. **위상 차원 선택과 계산 복잡도**  
   - 0차 위상(연결성)만으로도 상당한 구조 정보를 얻을 수 있지만, 실제 매니폴드는 종종 1차(사이클), 2차(공동) 구조를 갖는다.  
   - TopoAE++처럼 부분 차원( $\(\mathrm{PH}^1\)$ )에 특화된 효율적 손실 설계나, 근사 PH·병렬화·서브샘플링 이론을 활용해 실용적인 스케일로 확장할 필요가 있다.[^22][^8][^7]

2. **손실 설계: 단순 다이어그램 거리 vs. 엣지 기반 정렬**  
   - 단순히 퍼시스턴스 다이어그램 간 Wasserstein/병목거리를 최소화하면, 엣지 매칭이 모호해지고 그라디언트 정보가 약해질 수 있다.[^18][^1]
   - TopoAE가 택한 “퍼시스턴스 페어링→엣지 거리 정렬” 접근은 강한 그라디언트를 제공하지만, 모든 위상적 상황에서 완전한 보존을 보장하지는 않는다.  
   - TopoAE++의 분석처럼, 어떤 손실이 어떤 위상 차원에서 completeness를 갖는지 이론적 분석이 중요하다.[^8][^7]

3. **데이터스케일·도메인 특화 설계**  
   - 자연영상, 과학데이터(HEP, 생명과학, 시공간 데이터 등)마다 매니폴드의 위상 특성이 다르므로, 어떤 차수의 퍼시스턴트 호몰로지가 중요한지, 어떤 거리/유사도(픽셀 L2, perceptual, domain-specific metric)를 써야 하는지 도메인 별 분석이 필요하다.[^23][^24][^10]

4. **일반화·견고성 평가 프로토콜 정립**  
   - TopoAE류 모델의 일반화 이점을 명확히 보이려면, 단순 재구성/시각화 외에, 다운스트림 작업(분류, 클러스터링, 이상 탐지, 전이학습)에서의 성능 및 adversarial/노이즈 견고성 평가가 필요하다.[^15][^10][^14]
   - 위상 보존 강도( $\(\lambda\$ ), 손실 형태)와 일반화 성능 사이의 trade‑off를 체계적으로 탐색하는 연구가 요구된다.

5. **이론과 구현 간의 간극 축소**  
   - 매니폴드 가설, 차트‑오토인코더, Gauss–Legendre 정규화 등에서 얻어진 일반화·위상 보존 이론을, TopoAE/TopoAE++류의 실질적인 구현·하이퍼파라미터와 연결하는 작업이 중요하다.[^12][^11][^13][^14]
   - 예를 들어, batch size, 서브샘플링 전략, 거리 척도 선택이 병목거리/하우스도르프 거리 상계와 어떤 상관을 갖는지 정량화할 수 있다.

[^11]
[^10]
[^8]
[^23]
[^24]
[^7]
[^14]
[^22]
[^12]
[^13]

## 8. 요약

- TopoAE는 퍼시스턴트 호몰로지를 활용해, 입력–잠재공간 사이의 위상적으로 중요한 거리들을 정렬하는 미분 가능 위상 손실을 제안함으로써, 차원 축소와 표현 학습에서 데이터 매니폴드의 전역 위상 구조를 보존하려는 첫 대표적 시도 중 하나이다.[^4][^1]
- 이 손실은 재구성 품질을 유지하면서, SPHERES와 같은 복잡 매니폴드에서 기존 기법이 실패하는 전역 위상 구조를 성공적으로 재현하며, 실제 데이터에서도 밀도·연속성·해석 가능성 측면에서 경쟁력 있는 성능을 보인다.[^1]
- 2020년 이후의 여러 후속 연구(TopoAE++, homological regularization, topology‑aware AE, Gauss–Legendre 정규화 등)는 TopoAE의 아이디어를 확장·일반화하여, 위상 보존을 통해 잠재공간 구조를 제어하고 일반화와 견고성을 향상시키는 방향으로 발전하고 있다.[^17][^18][^13][^8][^14]
- 향후 연구에서는, (i) 고차원 위상까지 포함하는 효율적 손실 설계, (ii) 위상 손실과 일반화 오차 사이의 이론적 연결, (iii) 도메인 특화 위상 priors와의 결합, (iv) 다운스트림 작업에서의 체계적 벤치마킹이 중요하다.

---

## References

1. [1906.00722v5.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/836aaf87-0cfa-45f7-864c-7db1489c08ce/1906.00722v5.pdf?AWSAccessKeyId=ASIA2F3EMEYESDNHVL5M&Signature=9ZIpWLpVLlJEhTJqCsTzfjuJGCw%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEC8aCXVzLWVhc3QtMSJHMEUCIBNQK1ZBIvTtjTAjiix566qk1TzqNekb7n5%2F2OBOaQI9AiEAhKvgHyAenhMhsfXO8ZElmGqvIORuBsB0WG%2BFo7owwsIq%2FAQI%2BP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDErIKPeYzs2ZTFu%2FXyrQBDOfpnW1oFsYCRrdbKD%2F3CjHIkpFbEORWcr0hHiwPWjUjQPS5%2FSoNKoeJcHayT9NYvWJjSa5ODBkYsY5akSRftPQAIaC%2BmgOog0jRdf4vtjP51MZCuoCbEkLzQGUu0FrTRwEZoq4bA3Oab%2BEYcOXYZOLnWdrFWtlWlLEdqLKPvHCt1SaUWhMKPjFYTc4HOWc6sAOcpKWnZh%2FUDIKWpWbC6rttNDQUV0V2zevHFh7nmM0274Rw41dByZtf7gohmCCT%2FpAOb4doRS4qf7tWcCQIYmKYSKaeNRx7sNzHsGBwdkvOFIZGY6GLWVvTH8tFlR%2FT7qYnGK5mMYbjSVAfKTZbcaYLUcAABstaUE9PyaS3AoycdYe4Y3INoKaP385B%2FcD%2F6Mhez6ubiWCcxYl8i5VtQdJnZCF4GFd86p7qgz9eYfZfX0tobE%2Fan%2B9oJAPdbvNE%2FcIyLJ9b0nc1x5Jvqn%2FGTO0eQ1ydrDbc5RqL7G7f%2F98H1t0m90OatCCYGYvfZP%2FRbI48ZzWX8I%2B7muDCC3RNTPCOfuy6%2F6larHG8TjTd85ECOi%2Be5V5%2BMEXf29%2B83Wf8%2BA0qw3q7jXDa%2FM4k%2Br62TUfHnx2uoBdf1zWHcZ07lU002d7u7oTlsenm07PZYw3mGxRUTl0YWOp%2BQrdCwxrFY%2F6%2FRE5e2RDCqRoPk2npueooKJvjnNQ3CyLNe%2FX%2FV8KslQwbR4pjK6peaDa3twk54AduEGHjRBH0A2U6dOe9iuzEt245%2Fejle9knQSNII%2Ft9IkUluE8V%2FxrJE6eugomwecwrZWvzQY6mAFv808EIIY1fukPOb988bFKQrgOwOWVmio%2Bqi3PgSnpSU8F1DJ%2B5vq%2B87%2BDGZZGkKOvKIQ9rc0uswkYyMO1rIEgotoTFk9IzU1cd0UH1j0wg9ew4hxQEKZz4yk5G99ksO3RxhWmvGXUubA3CHKdZxblgsVAaePe2qOgbkk%2BqyT%2BgqZrRvb53%2FGJQ3Aog7ZM%2BhC71fIiL0KWZA%3D%3D&Expires=1772870420) - Topological Autoencoders
Michael Moor † 1 2 Max Horn † 1 2 Bastian Rieck ‡ 1 2 Karsten Borgwardt ‡ 1...

2. [[1906.00722] Topological Autoencoders - arXiv](https://arxiv.org/abs/1906.00722) - We propose a novel approach for preserving topological structures of the input space in latent repre...

3. [Topological Autoencoders](http://arxiv.org/pdf/1906.00722.pdf) - We propose a novel approach for preserving topological structures of the
input space in latent repre...

4. [Topological Autoencoders](https://www.semanticscholar.org/paper/ed69978f1594a4e2b9dccfc950490fa1df817ae8) - We propose a novel approach for preserving topological structures of the input space in latent repre...

5. [A Survey of Topological Data Analysis Applications in NLP - arXiv](https://arxiv.org/html/2411.10298v4) - Representation regularization or topology-preserving learning: Adding topological losses (or using t...

6. [[PDF] Mitigating Adversarial Attacks Using Topological Purification in OCR ...](https://arxiv.org/pdf/2511.15807.pdf) - Topological Autoencoders. Topology-preserving autoencoders (Moor et al. 2020) incor- porate persiste...

7. [Topological Autoencoders++: Fast and Accurate Cycle-Aware Dimensionality
  Reduction](https://arxiv.org/html/2502.20215v1) - This paper presents a novel topology-aware dimensionality reduction approach
aiming at accurately vi...

8. [Topological Autoencoders++: Fast and Accurate Cycle-Aware Dimensionality Reduction](https://ieeexplore.ieee.org/document/11301037/) - This paper presents a novel topology-aware dimensionality reduction approach aiming at accurately vi...

9. [Challenges and Opportunities in Topological Deep Learning - arXiv](https://arxiv.org/html/2402.08871v1) - Topological autoencoders. In Proceedings of the 37th International ... Journal of Machine Learning R...

10. [Enhancing anomaly detection with topology-aware autoencoders](https://arxiv.org/html/2502.10163v1) - Our results show that autoencoders with topological priors significantly improve anomaly separation ...

11. [[PDF] Generalization Error and Robustness - arXiv.org](https://arxiv.org/pdf/2303.09863.pdf) - Autoencoder is a special designed deep learning method to effectively learn low-dimensional features...

12. [Deep nonparametric estimation of intrinsic data structures by chart ...](https://www.sciencedirect.com/science/article/abs/pii/S1063520323000891) - Our paper establishes statistical guarantees on the generalization error of chart autoencoders, and ...

13. [Geometry Regularized Autoencoders - PMC - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC10339657/) - Persistent homology [49], [50] was introduced as a means to identify the topological signature of ma...

14. [[2309.08228] Ensuring Topological Data-Structure Preservation ...](https://arxiv.org/abs/2309.08228) - Abstract:We formulate a data independent latent space regularisation constraint for general unsuperv...

15. [Mitigating Adversarial Attacks Using Topological Purification in OCR ...](https://arxiv.org/html/2511.15807v1) - Topological purification: A Topological Autoencoder is employed for the purification of adversarial ...

16. [Generating Similarity Map for COVID-19 Transmission Dynamics with Topological Autoencoder](https://www.semanticscholar.org/paper/2439614328da116439f51ad2113eff15d76cd842) - At the beginning of 2020 the world has seen the initial outbreak of COVID-19, a disease caused by SA...

17. [Enhancing anomaly detection with topology-aware autoencoders](http://arxiv.org/pdf/2502.10163.pdf) - ...momentum conservation. We construct
autoencoders with spherical ($S^n$), product ($S^2 \otimes S^...

18. [Homological Regularization for Autoencoders](https://bhoagsbargrill.com/latex/public/hom-ae.pdf)

19. [Local distance preserving autoencoders using continuous kNN graphs](https://icml.cc/virtual/2022/21072) - In this paper, we introduce several auto-encoder models that preserve local distances in the latent ...

20. [[PDF] arXiv:2505.01694v1 [cs.CV] 3 May 2025](https://arxiv.org/pdf/2505.01694.pdf) - Persistent Homology (PH) is a central technique within. Topological Data Analysis (TDA) designed to ...

21. [Cover Learning for Large-Scale Topology Representation - arXiv.org](https://arxiv.org/html/2503.09767v2) - (2020) ↑ Moor, M., Horn, M., Rieck, B., and Borgwardt, K. Topological autoencoders. In International...

22. [Towards Scalable Topological Regularizers - arXiv.org](https://arxiv.org/html/2501.14641v1) - The application of persistent homology in machine learning has been enabled by theoretical studies i...

23. [Analyzing crises in global financial indices using Recurrent Neural Network based Autoencoder](https://dx.plos.org/10.1371/journal.pone.0326947) - In this study, we present a novel approach to analyzing financial crises of the global stock market ...

24. [Spatial Transcriptome Uncovers the Mouse Lung Architectures and Functions](https://www.frontiersin.org/articles/10.3389/fgene.2022.858808/full) - Diseases leading to lung structural and functional disruption pose a major threat to human health (T...

