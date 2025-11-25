# Bagging Predictors

## 1. 핵심 주장과 주요 기여 요약

Leo Breiman의 1996년 논문 **"Bagging Predictors"**는 앙상블 학습(Ensemble Learning)의 근본 원리를 정립한 기념비적 연구로, 현대 머신러닝의 초석을 마련했다.[1]

**핵심 주장:** Bagging(Bootstrap Aggregating)은 학습 세트의 부트스트랩 복제본을 생성하여 여러 예측기를 훈련시키고, 이를 집계(회귀: 평균, 분류: 투표)함으로써 예측 정확도를 크게 향상시킬 수 있다. 가장 중요한 요소는 **예측 방법의 불안정성(Instability)**이다 - 학습 세트의 작은 변화가 예측기에 큰 변화를 야기하는 불안정한 방법(의사결정 트리, 신경망, 선형 회귀의 변수 선택)에서 배깅이 가장 효과적이다.[1]

**주요 기여:**
- 부트스트랩 샘플링과 예측기 집계를 통한 **분산(Variance) 감소 메커니즘** 이론화
- 분류 트리에서 6%~77%의 오분류율 감소, 회귀 트리에서 21%~46%의 평균제곱오차 감소 실증
- 안정적/불안정적 학습 알고리즘에 대한 배깅 효과의 이론적 설명 제시
- 학습 세트를 테스트 세트로 활용할 수 있는 이론적 근거 제공

***

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

머신러닝에서 단일 학습 세트 $$L = \{(y_n, x_n), n = 1, ..., N\}$$로 훈련된 예측기 $$\phi(x, L)$$는 학습 데이터의 특성에 과도하게 의존하여 **높은 분산**을 보일 수 있다. 특히 의사결정 트리, 신경망 등 불안정한 알고리즘은 학습 세트의 미세한 변화에도 예측기가 크게 달라지는 문제가 있다.[1]

Breiman은 다음과 같은 질문을 제기했다: "여러 개의 독립적인 학습 세트 $$\{L_k\}$$가 주어졌을 때, 단일 예측기보다 더 나은 예측기를 어떻게 구축할 수 있는가?"[1]

### 2.2 제안 방법: Bootstrap Aggregating (Bagging)

#### 수학적 정의

**수치 예측(회귀)의 경우:**

여러 학습 세트가 주어졌을 때, 집계된 예측기는 다음과 같이 정의된다:

$$\phi_A(x) = E_L[\phi(x, L)]$$

여기서 $$E_L$$은 학습 세트 $$L$$에 대한 기댓값을 나타낸다.[1]

실제로는 단일 학습 세트만 주어지므로, 부트스트랩 샘플 $$\{L^{(B)}\}$$을 생성하여 배깅 예측기를 구성한다:

$$\phi_B(x) = \text{av}_B \phi(x, L^{(B)})$$

**분류의 경우:**

각 부트스트랩 예측기 $$\phi(x, L^{(B)})$$가 클래스를 예측하면, 다수결 투표로 최종 클래스를 결정한다:

$$\phi_B(x) = \arg\max_j N_j$$

여기서 

```math
N_j = \#\{k; \phi(x, L_k) = j\}
```

이다.[1]

#### 분산 감소의 이론적 근거

고정된 입력 $$x$$와 출력 $$y$$에 대해 다음 부등식이 성립한다:

$$E_L(y - \phi(x, L))^2 \geq (y - \phi_A(x))^2$$

이는 $$[E_L\phi(x, L)]^2 \leq E_L\phi^2(x, L)$$ 부등식을 적용한 결과이다.[1]

핵심 통찰: **예측기 $$\phi(x, L)$$의 가변성이 클수록(불안정할수록) 양 변의 차이가 커지고, 집계를 통한 개선 효과가 증대된다**.[1]

### 2.3 알고리즘 구조

배깅의 실행 절차는 다음과 같다:[1]

1. **부트스트랩 샘플링:** 원본 학습 세트 $$L$$에서 복원 추출로 $$N$$개의 샘플을 추출하여 $$L^{(B)}$$ 생성 (각 샘플은 원본에서 0회, 1회, 2회 이상 등장 가능)

2. **개별 예측기 훈련:** 각 $$L^{(B)}$$에서 예측기 $$\phi_1(x), ..., \phi_B(x)$$ 훈련

3. **집계:**
   - 회귀: 모든 예측기의 평균
   - 분류: 다수결 투표

4. **가지치기(Pruning):** 원본 학습 세트 $$L$$을 테스트 세트로 활용하여 최적의 가지치기된 서브트리 선택

### 2.4 실험 결과 및 성능 향상

#### 분류 트리 실험 결과:[1]

| 데이터셋 | 단일 트리 오류율(%) | 배깅 오류율(%) | 감소율 |
|---------|-------------------|--------------|-------|
| Waveform | 29.1 | 19.3 | 34% |
| Heart | 4.9 | 2.8 | 43% |
| Breast Cancer | 5.9 | 3.7 | 37% |
| Ionosphere | 11.2 | 7.9 | 29% |
| Diabetes | 25.3 | 23.9 | 6% |
| Glass | 30.4 | 23.6 | 22% |

#### 대규모 데이터셋(Statlog Project) 결과:[1]

| 데이터셋 | 단일 트리 오류율(%) | 배깅 오류율(%) | 감소율 |
|---------|-------------------|--------------|-------|
| Letters | 12.6 | 6.4 | 49% |
| Satellite | 14.8 | 10.3 | 30% |
| Shuttle | 0.062 | 0.014 | 77% |
| DNA | 6.2 | 5.0 | 19% |

Statlog 프로젝트의 22개 분류기 중 배깅 트리는 **평균 순위 1.8**로 최고 성능을 기록했다 (2위는 6.3).[1]

#### 회귀 트리 실험 결과:[1]

| 데이터셋 | 단일 트리 MSE | 배깅 MSE | 감소율 |
|---------|-------------|---------|-------|
| Boston Housing | 20.0 | 11.6 | 42% |
| Ozone | 23.9 | 18.8 | 21% |
| Friedman #1 | 11.4 | 6.1 | 46% |
| Friedman #2 | 31,100 | 22,100 | 29% |
| Friedman #3 | 0.0403 | 0.0242 | 40% |

***

## 3. 일반화 성능 향상의 핵심 메커니즘

### 3.1 분산 감소를 통한 일반화

배깅의 핵심 강점은 **예측 분산을 줄이면서 편향을 크게 증가시키지 않는 것**이다. 기존 편향-분산 분해(Bias-Variance Decomposition)에 따르면:[2][3]

$$\text{Expected Error} = \text{Noise}^2 + \text{Bias}^2 + \text{Variance}$$

배깅은 여러 모델의 예측을 평균화하여 분산 항을 효과적으로 감소시킨다. 특히 **낮은 편향, 높은 분산**을 가진 모델(의사결정 트리 등)에서 최대 효과를 발휘한다.[3][4]

### 3.2 불안정성과 배깅 효과의 관계

Breiman은 다음과 같은 핵심 발견을 보고했다:[1]

- **불안정한 절차(Unstable Procedures):** 배깅이 크게 효과적 (신경망, 의사결정 트리, 변수 선택)
- **안정적 절차(Stable Procedures):** 배깅이 효과 없거나 오히려 성능 저하 (k-최근접 이웃)

실험에서 k-최근접 이웃 분류기에 배깅을 적용한 결과, 오류율 변화가 전혀 없었다. 이는 안정적 방법에서 부트스트랩 샘플로 인한 예측 변화가 매우 작기 때문이다.[1]

### 3.3 교차점(Cross-over Point) 현상

선형 회귀 시뮬레이션에서 Breiman은 중요한 현상을 발견했다:[1]

- 변수 수가 적을 때: 불안정성이 높아 배깅이 효과적
- 변수 수가 증가할 때: 안정성이 높아져 배깅 효과 감소
- 특정 지점 이후: 배깅된 예측기가 비배깅 예측기보다 오히려 성능 저하

이는 $$\phi_B = \phi_A(x, P_L)$$이 부트스트랩 분포 $$P_L$$에서 집계되지만, 실제 데이터는 원래 분포 $$P$$에서 추출되기 때문이다. 안정적 절차에서는 $$\phi_A(x, P_L)$$이 $$\phi_A(x, P) \approx \phi(x, L)$$보다 정확도가 떨어진다.[1]

### 3.4 분류에서의 순서 정확성(Order-Correctness)

분류 문제에서 예측기 $$\phi$$가 입력 $$x$$에서 **순서 정확(order-correct)**하다는 것은 다음을 의미한다:[1]

$$\arg\max_j Q(j|x) = \arg\max_j P(j|x)$$

여기서 $$Q(j|x) = P(\phi(x, L) = j)$$이고 $$P(j|x)$$는 입력 $$x$$가 클래스 $$j$$를 생성할 확률이다.

**핵심 통찰:** 예측기가 대부분의 입력에서 순서 정확하다면, 집계를 통해 **거의 최적(near-optimal)** 분류기로 변환될 수 있다. 그러나 수치 예측과 달리, 분류에서는 나쁜 예측기가 더 나빠질 수도 있다.[1]

***

## 4. 한계점

### 4.1 안정적 모델에서의 비효과성

배깅은 **높은 편향을 가진 안정적 모델**(선형 회귀, k-최근접 이웃)에서는 성능 향상이 제한적이거나 오히려 성능이 저하될 수 있다. 이미 낮은 분산을 가진 모델에서는 배깅의 분산 감소 효과가 미미하다.[5][6]

### 4.2 해석 가능성 상실

단일 의사결정 트리는 직관적이고 해석 가능하지만, 배깅된 앙상블은 **"블랙박스"**가 되어 해석이 어렵다. Breiman 자신도 "얻는 것은 증가된 정확도이고, 잃는 것은 단순하고 해석 가능한 구조"라고 인정했다.[7][8][1]

### 4.3 계산 비용 증가

여러 모델을 훈련해야 하므로 계산 시간과 메모리 사용량이 증가한다. 다만 배깅은 **병렬화에 매우 적합**하여, 각 부트스트랩 모델이 독립적으로 훈련될 수 있다.[5][1]

### 4.4 최적 성능 한계

데이터가 이미 달성 가능한 최소 오류율에 근접한 경우, 배깅으로도 큰 개선을 기대하기 어렵다. Diabetes 데이터셋에서 6%의 적은 감소가 이를 보여준다.[1]

### 4.5 소규모 데이터셋 문제

데이터셋이 작으면 부트스트랩 샘플 간 중복이 많아져 모델 다양성이 감소하고, 일반화보다 과적합이 발생할 수 있다.[5]

***

## 5. 연구 영향 및 향후 연구 고려사항

### 5.1 현대 머신러닝에 미친 영향

Breiman의 배깅 논문은 **41,000회 이상 인용**되며 앙상블 학습의 토대를 확립했다. 이 연구는 다음의 핵심 발전으로 이어졌다:[9]

**Random Forest (2001):** Breiman은 배깅을 확장하여 각 분할에서 **무작위 특성 선택**을 추가한 Random Forest를 개발했다. 최근 연구에 따르면 Random Forest는 배깅 대비 편향도 함께 감소시킬 수 있으며, 특히 높은 신호 대 잡음비(SNR) 환경에서 우수한 성능을 보인다.[10][11]

**딥러닝 앙상블:** 배깅 원리는 Deep Ensembles로 확장되어 불확실성 추정과 분포 외 탐지에 활용되고 있다. 2024년 연구에서 배깅 기반 딥러닝이 작물 분할 작업에서 기존 방법 대비 IoU를 40% 향상시켰다.[12][13][14]

**진화적 배깅:** 기존 고정 bag 대신 진화 알고리즘으로 bag 내용을 반복적으로 개선하는 **Evolutionary Bagging** 기법이 제안되었다.[15][16]

### 5.2 향후 연구 고려사항

**해석 가능성과 성능의 균형:** 최근 연구들은 트리 앙상블을 **일반화 가산 모델(GAM)** 형태로 변환하여 해석 가능성을 확보하면서 성능을 유지하는 방법을 제시하고 있다. eXplainable Random Forest(XRF)와 같이 훈련 단계에서 해석 가능성 제약을 통합하는 접근도 등장했다.[17][8][18]

**편향 완화:** 2024년 연구에서 배깅/부스팅 모델의 손실 함수에 정규화 항을 추가하여 예측 편향을 절반으로 줄이면서 정확도를 유지하는 방법이 제안되었다.[19]

**효율적 앙상블 학습:** Packed-Ensembles, LightTS 등 경량화된 앙상블 기법이 하드웨어 제약 환경에서도 배깅의 이점을 활용할 수 있도록 발전하고 있다.[20][12]

**Stratified Sampling 통합:** 기존 무작위 샘플링 대신 **층화 샘플링**을 배깅에 통합하여 예측 정확도와 안정성을 동시에 향상시키는 ssBlending 알고리즘이 제안되었다.[21]

**SHAP 기반 모델 해석:** 앙상블 모델의 예측을 SHAP(SHapley Additive exPlanations)을 통해 특성별 기여도로 분해하여 투명성을 확보하는 연구가 활발히 진행 중이다.[22][23]

***

## 결론

Leo Breiman의 "Bagging Predictors"는 단순하면서도 강력한 아이디어—**부트스트랩 복제와 예측 집계**—로 머신러닝의 패러다임을 전환시켰다. 이 논문의 핵심 기여는 불안정한 학습 알고리즘의 **분산을 효과적으로 감소**시켜 일반화 성능을 향상시킬 수 있음을 이론적, 실증적으로 입증한 것이다. 안정적 모델에서의 제한된 효과와 해석 가능성 상실이라는 한계가 있지만, 이후 Random Forest, 딥러닝 앙상블, 그리고 설명 가능한 AI(XAI) 연구로 이어지며 그 영향력은 지속적으로 확대되고 있다. 현대 연구자들은 배깅의 원리를 기반으로 **정확도, 해석 가능성, 계산 효율성**의 균형을 추구하는 방향으로 연구를 발전시키고 있다.[11][24][17][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/bbfea155-37fd-4008-83ec-db9c821c6fce/BF00058655.pdf)
[2](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
[3](https://www.cs.cornell.edu/courses/cs578/2005fa/CS578.bagging.boosting.lecture.pdf)
[4](https://web.engr.oregonstate.edu/~tgd/classes/534/slides/part9.pdf)
[5](https://www.tencentcloud.com/techpedia/113009)
[6](https://massedcompute.com/faq-answers/?question=What+are+the+advantages+and+disadvantages+of+bagging+algorithms+in+machine+learning%3F)
[7](http://arxiv.org/pdf/2302.07580.pdf)
[8](https://arxiv.org/abs/2410.19098)
[9](https://scholar.google.com/citations?user=mXSv_1UAAAAJ&hl=en)
[10](https://keylabs.ai/blog/random-forest-ensemble-learning-technique/)
[11](http://www.jmlr.org/papers/volume26/24-0255/24-0255.pdf)
[12](https://arxiv.org/pdf/2210.09184.pdf)
[13](https://pmc.ncbi.nlm.nih.gov/articles/PMC9668167/)
[14](https://pmc.ncbi.nlm.nih.gov/articles/PMC11174727/)
[15](https://arxiv.org/pdf/2208.02400.pdf)
[16](https://www.sciencedirect.com/science/article/abs/pii/S0925231222010414)
[17](https://arxiv.org/pdf/2410.19098.pdf)
[18](https://ceur-ws.org/Vol-3765/Camera_Ready_Paper-10.pdf)
[19](https://www.nature.com/articles/s41598-024-68907-5)
[20](https://arxiv.org/pdf/2302.12721.pdf)
[21](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf002/8030212)
[22](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0312124)
[23](https://www.nature.com/articles/s41598-025-97547-6)
[24](https://www.dremio.com/wiki/bagging-and-boosting/)
[25](https://www.mdpi.com/1099-4300/19/10/520/pdf?version=1506602662)
[26](https://arxiv.org/abs/2407.10574)
[27](https://arxiv.org/pdf/2105.02569.pdf)
[28](http://arxiv.org/pdf/2409.12849.pdf)
[29](https://www.ijnrd.org/papers/IJNRD2411095.pdf)
[30](https://scikit-learn.org/stable/modules/ensemble.html)
[31](https://github.com/Pudding2159/Bagging-Algorithm-for-Neural-Networks)
[32](https://www.nature.com/articles/s41598-025-15971-0)
[33](https://hanseokhyeon.tistory.com/entry/Chapter-7-Ensemble-Learning-and-Random-Forests)
[34](https://code-b.dev/blog/bagging-machine-learning)
[35](https://www.sciencedirect.com/science/article/pii/S1319157823000228)
[36](https://www.sciencedirect.com/science/article/pii/S1094996805700591)
[37](https://dl.acm.org/doi/10.1145/3718740)
[38](https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/)
[39](https://ieeexplore.ieee.org/document/10324161/)
[40](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.3166)
[41](https://www.ibm.com/think/topics/ensemble-learning)
[42](https://arxiv.org/html/2403.15766v1)
[43](https://www.tandfonline.com/doi/full/10.1080/10095020.2024.2311261)
[44](https://clairedavid.github.io/ml_in_hep/week1/BDTs_forest.html)
[45](https://arxiv.org/pdf/2402.12668.pdf)
[46](https://arxiv.org/pdf/1905.12787.pdf)
[47](https://pmc.ncbi.nlm.nih.gov/articles/PMC2367370/)
[48](http://arxiv.org/pdf/2011.03321.pdf)
[49](https://pmc.ncbi.nlm.nih.gov/articles/PMC11310502/)
[50](https://arxiv.org/html/2405.15403)
[51](https://arxiv.org/pdf/1908.02718.pdf)
[52](https://arxiv.org/pdf/2002.11328.pdf)
[53](https://www.diva-portal.org/smash/get/diva2:1669623/FULLTEXT01.pdf)
[54](https://towardsdatascience.com/bagging-on-low-variance-models-38d3c70259db/)
[55](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1996-ML-Breiman-Bagging%20Predictors.pdf)
[56](https://www.datacamp.com/tutorial/what-bagging-in-machine-learning-a-guide-with-examples)
[57](https://journament.com/biblio/112872)
[58](https://uniathena.com/understanding-bias-variance-tradeoff-balance-model-performance)
[59](https://www.ibm.com/think/topics/bagging)
[60](https://www.semanticscholar.org/paper/Bagging-Predictors-Breiman/d1ee87290fa827f1217b8fa2bccb3485da1a300e)
[61](https://towardsdatascience.com/strength-in-numbers-ensembling-models-with-bagging-and-boosting/)
[62](https://arxiv.org/pdf/1101.0917.pdf)
[63](https://zephyrus1111.tistory.com/238)
[64](https://blog.naver.com/angryking/221200181588)
[65](https://www.geeksforgeeks.org/machine-learning/bagging-vs-boosting-in-machine-learning/)
[66](https://arxiv.org/html/2312.12715v2)
[67](http://arxiv.org/pdf/2401.17200.pdf)
[68](https://arxiv.org/html/2306.06193)
[69](https://www.mdpi.com/2504-4990/6/2/38/pdf?version=1712731580)
[70](https://arxiv.org/pdf/2402.02933.pdf)
[71](http://arxiv.org/pdf/2312.06255.pdf)
[72](https://luisgalarraga.de/docs/xai_mdd24.pdf)
[73](https://aiquizzes.com/questions/68.html)
[74](https://www.sciencedirect.com/science/article/pii/S259000562500013X)
[75](https://pmc.ncbi.nlm.nih.gov/articles/PMC11424291/)
[76](https://arxiv.org/abs/2507.03884)
[77](https://ieeexplore.ieee.org/abstract/document/10955903/)
[78](https://www.sciencedirect.com/science/article/pii/S026840122200072X)
[79](https://dl.acm.org/doi/10.1007/978-3-031-77367-9_26)
[80](https://ieeexplore.ieee.org/document/9533601)
[81](https://stat.ethz.ch/Manuscripts/buhlmann/breiman-rememb.pdf)
[82](https://arxiv.org/html/2305.02012v3)
[83](https://www.biorxiv.org/content/10.1101/2024.02.18.580860v2.full.pdf)
[84](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1493)
