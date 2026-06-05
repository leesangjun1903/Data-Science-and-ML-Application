
# Self-Training With Noisy Student Improves ImageNet Classification 

> **논문 정보**
> - 저자: Qizhe Xie, Minh-Thang Luong, Eduard Hovy, Quoc V. Le (Google Brain / CMU)
> - 발표: CVPR 2020
> - arXiv: [1911.04252](https://arxiv.org/abs/1911.04252)

---

## 1. 🔑 핵심 주장 및 주요 기여 요약

이 논문은 **Noisy Student Training**이라는 반지도 학습(semi-supervised learning) 접근법을 제시하며, 레이블된 데이터가 충분한 경우에도 효과적으로 작동한다는 점을 강조합니다.

**핵심 주장:**

Noisy Student Training은 ImageNet에서 **88.4% top-1 정확도**를 달성하였으며, 이는 35억 개의 약한 레이블 Instagram 이미지를 필요로 하는 당시 최고 성능 모델보다 **2.0% 높은** 수치입니다.

강건성(robustness) 테스트셋에서도 탁월한 성능을 보여, ImageNet-A의 top-1 정확도를 61.0%에서 **83.7%**로 향상시키고, ImageNet-C의 평균 손상 오류(mean corruption error)를 45.7에서 **28.3**으로 감소시키며, ImageNet-P의 평균 플립율(mean flip rate)을 27.8에서 **12.2**로 낮췄습니다.

**주요 기여:**

Noisy Student Training은 **동일하거나 더 큰 학생 모델(student model)** 사용과 **학습 중 노이즈 주입**이라는 두 가지 핵심 개념을 결합하여 self-training 및 distillation의 아이디어를 확장합니다.

이 접근법의 핵심 개선점은 두 가지로, **학생 모델에 노이즈를 추가**하는 것과 **학생 모델이 교사 모델보다 크거나 동일한 크기**를 갖도록 하는 것입니다.

---

## 2. 🔬 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상, 한계

### 2.1 해결하고자 하는 문제

당시 최고 수준의 비전 모델들은 대규모 레이블 이미지 데이터에 의존하는 지도 학습(supervised learning)으로 학습되었습니다. 레이블된 이미지만을 사용함으로써, 훨씬 많은 양의 **비레이블 이미지(unlabeled images)**를 활용하여 정확도와 강건성을 향상시킬 수 있는 기회를 놓치고 있었습니다.

---

### 2.2 제안하는 방법 (알고리즘 + 수식)

논문이 제안하는 Noisy Student Training은 아래 4단계로 구성됩니다.

**(1) 레이블 데이터로 교사(teacher) 분류기를 학습, (2) 훨씬 큰 비레이블 데이터셋에 대해 pseudo label 추론, (3) 노이즈를 추가하며 결합된 데이터셋으로 더 큰 학생(student) 분류기를 학습**, (4) 해당 과정을 반복합니다.

**핵심 목적 함수(Loss Function):**

학생 모델 $\theta^s$는 레이블 이미지와 비레이블 이미지 모두에 대한 교차 엔트로피 손실의 합을 최소화하도록 학습되며, 이때 학생 모델에는 **노이즈(noise)가 주입**됩니다:

$$
\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} \ell\!\left(y_i,\, f^{\text{noised}}(x_i, \theta^s)\right) + \frac{1}{m} \sum_{i=1}^{m} \ell\!\left(\tilde{y}_i,\, f^{\text{noised}}(\tilde{x}_i, \theta^s)\right)
$$

여기서:
- $n$: 레이블 이미지 수, $m$: 비레이블 이미지 수
- $y_i$: 실제 레이블, $\tilde{y}_i$: 교사가 생성한 pseudo label
- $x_i$: 레이블 이미지, $\tilde{x}_i$: 비레이블 이미지
- $f^{\text{noised}}$: 노이즈가 적용된 학생 모델의 예측 함수
- $\ell$: 교차 엔트로피 손실 함수

**Pseudo Label 유형:**

Pseudo label은 **soft label**(연속 분포) 또는 **hard label**(one-hot 분포)로 생성될 수 있으며, 두 방식 모두 실험에서 효과적이었습니다. 특히, **soft pseudo label은 도메인이 다른(out-of-domain) 비레이블 데이터에 더 효과적**인 것으로 나타났습니다.

**노이즈의 종류:**

학생 모델에 노이즈를 주입하기 위해, **RandAugment**와 같은 입력 노이즈(input noise)와 **Dropout**, **Stochastic Depth**와 같은 모델 노이즈(model noise)를 훈련 중에 적용합니다.

**교사와 학생의 역할 구분 (중요한 비대칭성):**

이 설정에서 핵심 요소는 **학생 모델 학습 시에는 노이즈를 사용하지만, 교사 모델이 pseudo label을 생성할 때에는 노이즈를 사용하지 않는다**는 것입니다. 이것이 바로 "Noisy Student" 방법의 핵심입니다.

**데이터 필터링 및 균형 조정:**

Noisy Student Training은 데이터 필터링과 밸런싱이라는 추가적인 기법으로 성능이 향상됩니다. 구체적으로, **교사 모델이 낮은 신뢰도로 예측한 이미지들은 도메인 외 이미지일 가능성이 높으므로 필터링**합니다.

신뢰도가 0.3 미만인 이미지는 필터링하며, **각 클래스별 비레이블 이미지 수를 균등하게 조정**하는데, 이는 ImageNet의 모든 클래스가 유사한 수의 레이블 이미지를 가지기 때문입니다.

---

### 2.3 모델 구조

ImageNet에서는 먼저 **EfficientNet** 모델을 레이블 이미지로 학습시켜 교사 모델로 활용하여 3억 장의 비레이블 이미지에 대한 pseudo label을 생성하고, 이후 더 큰 EfficientNet을 학생 모델로 레이블 + pseudo label 이미지의 조합으로 학습시킵니다. 이 과정을 학생 모델을 교사 모델로 교체하여 반복합니다.

**반복적 학습 순서 (Iterative Training):**

구체적인 반복 순서는 다음과 같습니다:
1. **EfficientNet-B7**을 교사이자 학생으로 사용하여 성능을 향상시킵니다.
2. 향상된 EfficientNet-B7을 교사로, **EfficientNet-L0**을 학생으로 사용합니다.
3. EfficientNet-L0을 교사로, 더 넓은 **EfficientNet-L1**을 학생으로 사용합니다.
4. EfficientNet-L1을 교사로, 가장 큰 모델인 **EfficientNet-L2**를 학생으로 사용합니다.

**EfficientNet-L2 특징:**

EfficientNet-L2는 EfficientNet-B7보다 더 넓고 깊지만 더 낮은 해상도를 사용하며, 이는 대량의 비레이블 이미지에 맞는 파라미터 용량을 갖습니다. 모델 크기가 크기 때문에 학습 시간은 EfficientNet-B7 대비 약 **5배** 소요됩니다.

가장 큰 모델인 EfficientNet-L2는 비레이블 배치 크기가 레이블 배치 크기의 14배인 설정에서 **2048코어를 가진 Cloud TPU v3 Pod**에서 6일이 소요됩니다.

---

### 2.4 성능 향상

| 벤치마크 | 기존 SOTA | Noisy Student | 향상 |
|---|---|---|---|
| ImageNet Top-1 | 86.4% | **88.4%** | +2.0% |
| ImageNet-A Top-1 | 61.0% | **83.7%** | +22.7% |
| ImageNet-C mCE | 45.7 | **28.3** | −17.4 |
| ImageNet-P mFR | 27.8 | **12.2** | −15.6 |

반복 학습을 통해 1차 반복 후 87.6%, 2차 반복 후 88.1%로 성능이 향상되었으며, 마지막 반복에서 비레이블 배치 크기 비율을 늘려 최종 **88.4%**의 성능을 달성했습니다.

**적대적 강건성:**

Noisy Student Training은 적대적 강건성(adversarial robustness)을 위해 최적화되지 않았음에도 불구하고, EfficientNet-L2의 정확도를 **1.1%에서 4.4%까지 향상**시켰습니다.

---

### 2.5 한계

논문 및 관련 연구들에서 지적된 주요 한계점은 다음과 같습니다:

1. **막대한 계산 비용**: EfficientNet-L2 학습에는 2048코어를 가진 Cloud TPU v3 Pod에서 6일이 소요되며, 이는 일반적인 연구 환경에서 재현하기 매우 어렵습니다.

2. **대규모 비레이블 데이터 필요**: 비레이블 데이터로 약 **3억 장의 JFT 데이터셋** 이미지를 활용했으며, 해당 이미지의 레이블은 무시하고 비레이블 데이터로 처리했습니다. 이러한 규모의 데이터는 접근이 제한적입니다.

3. **극단적 레이블 부족 시나리오에서의 취약성**: 극소수 레이블 환경(예: SVHN에서 레이블 40개)에서는 pseudo label 노이즈와 모델 초기화에 대한 민감도가 증폭되어 성능이 저하될 수 있습니다.

4. **누적 오류(Error Accumulation)**: 반복 학습 과정에서 초기 교사 모델의 오류가 pseudo label을 통해 학생 모델에 전파될 수 있으며, 이는 self-training의 본질적인 한계입니다.

---

## 3. 🌐 모델의 일반화 성능 향상 가능성

일반화 성능 향상은 이 논문의 가장 중요한 기여 중 하나입니다.

### 3.1 노이즈가 일반화에 미치는 메커니즘

**노이즈는 학생이 교사보다 더 나은 성능을 달성하는 데 중요한 역할을 합니다.** 추가된 노이즈는 레이블 데이터와 비레이블 데이터 모두에서 모델의 결정 경계(decision-making frontier)를 부드럽게 만드는 복합적 효과(compound effect)를 갖습니다.

**각 노이즈 유형별 일반화 메커니즘:**

**데이터 증강 노이즈(Data Augmentation Noise)**의 경우, 예를 들어 이미지를 평행이동(translation)해도 동일한 카테고리를 예측해야 한다는 **불변성(invariant constraint)**을 학생이 학습하도록 강제하며, 이는 학생 모델이 더 어려운 이미지에 대해서도 올바른 예측을 하도록 합니다.

**Dropout과 Stochastic Depth**를 노이즈로 사용하면, 교사는 추론 시(pseudo label 생성 시) **앙상블(ensemble)**처럼 작동하는 반면 학생은 단일 모델로 작동하게 됩니다. 즉, **학생은 더 강력한 앙상블 모델을 모방**하도록 강제됩니다.

### 3.2 강건성(Robustness) 측면에서의 일반화

다양한 분포 변화(distribution shift)에 대한 강건성 테스트에서 ImageNet-A top-1 정확도를 61.0%에서 83.7%로 향상시키고, ImageNet-C 평균 손상 오류를 45.7에서 28.3으로 감소시키며, ImageNet-P 평균 플립율을 27.8에서 12.2로 줄이는 탁월한 성능을 보였습니다. 이는 단순한 정확도 향상을 넘어 **실제 세계의 다양한 조건에서도 높은 일반화 성능**을 갖는다는 것을 의미합니다.

### 3.3 Teacher-Student 비대칭 구조와 일반화

일반화를 위한 핵심 설계 원칙으로, **학생 모델을 교사보다 크게** 만들어 더 큰 데이터셋을 더 잘 학습할 수 있도록 하고, **학생에 노이즈를 추가**하여 노이즈가 있는 학생이 pseudo label로부터 더 어렵게 학습하도록 강제합니다.

### 3.4 적은 레이블 데이터에서의 일반화

이 방법은 레이블 데이터가 충분한 경우에도 효과적으로 작동하는 반지도 학습 접근법이지만, 레이블이 적은 환경에서도 비레이블 데이터를 활용함으로써 일반화를 극적으로 향상시킬 수 있는 잠재력을 가집니다.

---

## 4. 🚀 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 이 논문이 후속 연구에 미친 영향

**① Semi-Supervised Learning 패러다임 강화**

이 논문은 Teacher-Student 기반의 self-training이 대규모 이미지 분류에 실질적으로 유효하다는 것을 명확히 보여주었으며, 이후 다양한 반지도 학습 연구의 기반이 되었습니다. FixMatch, UDA, FlexMatch, Semi-ViT 등의 이후 SSL 방법들은 모두 Noisy Student와 같이 임계값(threshold) 기법을 활용하는 방법들과 비교되며 발전해 왔습니다.

**② FixMatch (2020)와의 연관성**

FixMatch는 Noisy Student의 일관성 정규화(consistency regularization) 아이디어를 계승하면서 약한 증강(weak augmentation)으로 생성된 pseudo label을 강한 증강(strong augmentation)에 적용하는 방식으로 발전시켰습니다. FixMatch의 비지도 손실 항은 다음과 같이 정의됩니다:

$$\frac{1}{\mu B} \sum_{b=1}^{\mu B} \mathbb{1}\!\left(\max\!\left(p_m(y|\omega(u_b))\right) > \tau\right) H\!\left(\hat{p}_m\!\left(y|\omega(u_b)\right), p_m\!\left(y|\Omega(u_b)\right)\right)$$

여기서 $\Omega$는 강한 증강 함수이고 $\omega$는 약한 증강 함수입니다. 이때 사전 정의된 임계값 $\tau$는 모든 클래스에 동일하게 적용됩니다.

**③ FlexMatch (NeurIPS 2021) — Noisy Student의 한계 극복**

FlexMatch(Zhang et al., NeurIPS 2021)는 FixMatch가 대부분의 SSL 벤치마크에서 최고 성능을 달성했지만, 다른 현대 SSL 알고리즘들처럼 모든 클래스에 동일한 사전 정의 임계값을 사용함으로써 **서로 다른 클래스의 학습 상태와 난이도를 고려하지 못한다**는 문제점을 지적했습니다.

이를 해결하기 위해 **Curriculum Pseudo Labeling(CPL)**을 제안했는데, 이는 모델의 학습 상태에 따라 비레이블 데이터를 활용하는 커리큘럼 학습 방식이며, **각 시간 단계마다 서로 다른 클래스에 대한 임계값을 유동적으로 조정**하여 유익한 비레이블 데이터와 pseudo label이 통과되도록 합니다.

FlexMatch는 클래스당 4개의 레이블만 존재할 때 CIFAR-100과 STL-10에서 FixMatch 대비 각각 13.96%, 18.96%의 오류율 감소를 달성했으며, FixMatch 대비 **학습 시간의 1/5만으로도 더 나은 성능을 달성**했습니다.

**④ Self-Supervised Learning (SSL)과의 시너지**

Noisy Student 이후, DINO, SimCLRv2, MoCo v3 등 자기 지도 학습(self-supervised learning) 방법들과 Teacher-Student 구조가 결합된 연구들이 활발하게 진행되었습니다. SimCLRv2를 사용한 ImageNet에서 ResNet101의 경우 비자명한 일반화 경계(non-vacuous generalization bound)가 48%까지 달성되었습니다.

---

### 4.2 앞으로 연구 시 고려할 점

**① 계산 효율성과 접근성**

Noisy Student가 가진 가장 큰 장벽은 막대한 컴퓨팅 자원입니다. EfficientNet-L2 학습에는 2048코어의 Cloud TPU v3 Pod에서 6일이 걸립니다. 향후 연구는 동일한 원칙을 유지하면서도 **자원 효율적인(resource-efficient) 대안**을 개발하는 것이 중요합니다.

**② Pseudo Label 품질 관리**

극단적으로 레이블이 적은 환경에서는 pseudo label 노이즈와 모델 초기화에 대한 민감도가 증폭되어 성능이 저하됩니다. 따라서 pseudo label의 신뢰도를 동적으로 측정하고 조정하는 메커니즘 연구가 필요합니다.

**③ 클래스 불균형 문제**

Noisy Student는 신뢰도가 낮은 이미지 필터링과 클래스별 데이터 균형 조정이 필요합니다. 이는 ImageNet의 모든 클래스가 유사한 수의 레이블 이미지를 가지기 때문입니다. 실세계 데이터는 심각한 클래스 불균형을 가지는 경우가 많아, 이에 대한 robust한 처리 방법 연구가 필요합니다.

**④ 도메인 특화 적용**

Noisy Student 원칙은 이미지 분류를 넘어 다양한 도메인으로 확장될 잠재력이 있습니다. 의료 영상, 자율주행, 음성 인식 등 레이블 획득이 어려운 분야에서의 적용 가능성을 탐색해야 합니다.

**⑤ Vision Transformer(ViT)와의 결합**

EfficientNet 기반이었던 Noisy Student를 Vision Transformer(ViT), Swin Transformer와 같은 최신 아키텍처에 적용하거나, DINO, MAE(Masked Autoencoders)와 같은 자기 지도 학습과 결합하는 방향으로 발전시키는 것이 중요한 연구 방향입니다.

**⑥ 오류 누적(Error Accumulation) 방지**

반복적 self-training에서 초기 교사의 오류가 누적될 수 있습니다. 실제로, 교사 모델에 노이즈를 추가하면 pseudo label의 품질이 떨어지고 정확도가 낮아지므로, **강력하고 노이즈 없는(unnoised) 교사 모델을 유지하는 것의 중요성**이 입증되었습니다. 따라서 반복 횟수와 교사 품질을 체계적으로 관리하는 전략이 필요합니다.

---

## 📊 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 핵심 아이디어 | Noisy Student 대비 차이점 | 주요 성과 |
|---|---|---|---|
| **FixMatch** (2020) | 약한 증강 → pseudo label, 강한 증강 → 학습 | 고정 임계값 $\tau$ 사용 | CIFAR-10 등 SSL 벤치마크 SOTA |
| **FlexMatch** (NeurIPS 2021) | 클래스별 동적 임계값 (CPL) | 클래스별 학습 상태 반영 | CIFAR-100, STL-10에서 FixMatch 대비 오류율 13~19% 감소 |
| **SimCLRv2** (2020) | 대조 학습 기반 SSL | Pseudo label 불필요, 표현 학습 중심 | ImageNet 79.8% top-1 (semi-supervised) |
| **DINO** (2021) | Self-distillation with no labels | 레이블 없이 ViT 기반 teacher-student | ViT-S/8에서 강력한 표현 학습 |
| **DST** (2022) | Debiased Self-Training | 확증 편향(confirmation bias) 제거 | FixMatch, FlexMatch 대비 8~10% 추가 향상 |
| **Semi-ViT** (2022) | ViT + SSL | Transformer 아키텍처에 semi-supervised 적용 | ImageNet 반지도 학습 SOTA |

FlexMatch의 CPL은 별도의 파라미터나 추가 연산 없이 모델의 학습 상태에 따라 비레이블 데이터를 활용하는 커리큘럼 학습 방식이며, **각 시간 단계마다 서로 다른 클래스에 대한 임계값을 유동적으로 조정**합니다.

---

## 📚 참고 자료

1. **[논문 원문]** Xie, Q., Luong, M.-T., Hovy, E., & Le, Q. V. (2020). *Self-Training With Noisy Student Improves ImageNet Classification.* CVPR 2020. arXiv:1911.04252. [https://arxiv.org/abs/1911.04252](https://arxiv.org/abs/1911.04252)

2. **[공식 구현]** Google Research. *Noisy Student Training GitHub Repository.* [https://github.com/google-research/noisystudent](https://github.com/google-research/noisystudent)

3. **[CVPR 2020 공식 페이지]** *Self-Training With Noisy Student Improves ImageNet Classification.* Computer Vision Foundation. [https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.html](https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.html)

4. **[IEEE Xplore]** IEEE Conference Publication, CVPR 2020. [https://ieeexplore.ieee.org/document/9156610/](https://ieeexplore.ieee.org/document/9156610/)

5. **[HuggingFace timm 문서]** *Noisy Student (EfficientNet).* [https://huggingface.co/docs/timm/models/noisy-student](https://huggingface.co/docs/timm/models/noisy-student)

6. **[리뷰 블로그]** Tsang, S.-H. (2022). *Review — Noisy Student: Self-training with Noisy Student improves ImageNet classification.* Medium. [https://sh-tsang.medium.com/review-noisy-student](https://sh-tsang.medium.com/review-noisy-student-self-training-with-noisy-student-improves-imagenet-classification-2e4e0acb7358)

7. **[관련 튜토리얼]** Nain, A. *Self-training with Noisy Student.* Medium. [https://medium.com/@nainaakash012/self-training-with-noisy-student-f33640edbab2](https://medium.com/@nainaakash012/self-training-with-noisy-student-f33640edbab2)

8. **[Semi-Supervised Learning 서베이]** Weng, L. (2021). *Learning with not Enough Data Part 1: Semi-Supervised Learning.* Lil'Log. [https://lilianweng.github.io/posts/2021-12-05-semi-supervised/](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/)

9. **[FlexMatch]** Zhang, B. et al. (2021). *FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling.* NeurIPS 2021. arXiv:2110.08263. [https://arxiv.org/abs/2110.08263](https://arxiv.org/abs/2110.08263)

10. **[Pseudo-Label 리뷰]** *A Review of Pseudo Labeling for Semi-Supervised Learning.* arXiv:2408.07221. [https://arxiv.org/pdf/2408.07221](https://arxiv.org/pdf/2408.07221)

11. **[SST]** *SST: Self-training with self-adaptive thresholding for semi-supervised learning.* ScienceDirect. [https://www.sciencedirect.com/science/article/abs/pii/S0306457325000998](https://www.sciencedirect.com/science/article/abs/pii/S0306457325000998)
