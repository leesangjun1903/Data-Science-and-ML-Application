
# Rectifying Pseudo Label Learning via Uncertainty Estimation for Domain Adaptive Semantic Segmentation

> **저자:** Zhedong Zheng, Yi Yang (University of Technology Sydney, AAII)
> **게재지:** International Journal of Computer Vision (IJCV), Vol. 129(4), pp. 1106–1120, 2021
> **arXiv:** 2003.03773

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 시맨틱 세그멘테이션 맥락에서 소스 도메인의 지식을 타겟 도메인으로 전이하는 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)에 초점을 맞춥니다.

기존 접근법들은 레이블이 없는 타겟 도메인 데이터를 최대한 활용하기 위해 의사 레이블(pseudo label)을 정답(ground truth)으로 간주합니다. 그러나 타겟 도메인의 의사 레이블은 소스 도메인으로 학습된 모델이 예측한 것이기 때문에, 두 도메인 간의 분포 차이로 인해 불가피하게 잘못된 예측이 포함되며, 이는 최종 적응 모델에 전파되어 학습 과정을 크게 손상시킵니다.

### 🔑 핵심 주장

이 문제를 극복하기 위해 본 논문은 학습 중 예측 불확실성을 명시적으로 추정하여 비지도 세그멘테이션 적응을 위한 의사 레이블 학습을 교정할 것을 제안합니다. 모델은 입력 이미지가 주어지면 시맨틱 세그멘테이션 예측과 함께 예측에 대한 불확실성을 출력하며, 구체적으로 예측 분산(prediction variance)을 통해 불확실성을 모델링하고 이를 최적화 목적 함수에 포함시킵니다.

### ✅ 주요 기여

실험을 통해 제안 방법이 (1) 예측 분산에 따라 서로 다른 신뢰도 임계값을 동적으로 설정하고, (2) 노이즈가 있는 의사 레이블로부터의 학습을 교정하며, (3) 기존 의사 레이블 학습 방식에 비해 유의미한 성능 향상을 달성하고 세 가지 벤치마크 모두에서 경쟁력 있는 성능을 보임을 입증합니다.

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

기존 접근법은 레이블 없는 타겟 도메인 데이터를 활용하기 위해 의사 레이블을 정답으로 간주합니다. 그러나 타겟 도메인의 의사 레이블은 소스 도메인으로 학습된 모델이 예측하며, 도메인 간 분포 불일치로 인해 생성된 레이블에는 필연적으로 오류가 포함되어 최종 적응 모델에 악영향을 줍니다.

Cityscapes에서의 노이즈 있는 의사 레이블 샘플을 보면, 널리 사용되는 기준 모델로 생성한 의사 레이블이 넓은 영역에서 정확한 예측을 보이더라도 데이터 분포 편향으로 인해 불가피하게 잘못된 예측을 포함합니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (a) 불확실성 모델링 (Uncertainty Modeling)

본 논문은 Bayesian deep learning의 관점에서 **예측 분산(prediction variance)**을 불확실성의 척도로 사용합니다. 입력 이미지 $x$에 대해 모델은 두 가지를 동시에 출력합니다:

- 세그멘테이션 예측 $\hat{y}$
- 각 픽셀에 대한 불확실성 $\sigma^2$

구체적으로, 예측 분산을 통해 불확실성을 모델링하고 이를 최적화 목적 함수에 포함시킵니다.

#### (b) 불확실성 기반 손실 함수

논문에서 핵심은 **Heteroscedastic Uncertainty** 기법을 적용한 수정된 크로스엔트로피 손실입니다. Kendall & Gal(2017)의 방법에 기반하여, 노이즈가 있는 로짓(logit) $\hat{f}$에 랜덤 노이즈를 더하는 방식으로 모델화합니다:

$$\hat{f}_t = \hat{f} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

여기서 $\sigma^2$는 픽셀별로 예측되는 분산(불확실성)입니다. 이를 Monte Carlo 샘플링을 통해 근사하면:

$$\hat{f}_t^{(s)} = \hat{f} + \sigma \cdot \epsilon_s, \quad \epsilon_s \sim \mathcal{N}(0, I)$$

최종 목적 함수는 불확실성 $\sigma$를 포함한 수정된 손실로 정의됩니다:

$$\mathcal{L}_{seg} = \frac{1}{S} \sum_{s=1}^{S} \text{CrossEntropy}\left(\text{softmax}\left(\hat{f}_t^{(s)}\right), \tilde{y}\right) + \log \sigma^2$$

- $S$: Monte Carlo 샘플 수
- $\tilde{y}$: 의사 레이블
- $\log \sigma^2$: 정규화 항으로, 모델이 모든 샘플에 대해 무한히 큰 불확실성을 예측하는 trivial solution을 방지

#### (c) 동적 신뢰도 임계값 (Dynamic Confidence Thresholding)

불확실성이 높은 픽셀을 학습에서 필터링하기 위해 **픽셀별 동적 임계값**을 설정합니다:

$$\tilde{y}_i = \begin{cases} \arg\max_c p_i^c & \text{if } \max_c p_i^c > \tau_c \\ \text{ignore} & \text{otherwise} \end{cases}$$

여기서 클래스별 임계값 $\tau_c$는 예측 분산에 따라 동적으로 결정됩니다:

$$\tau_c = \delta \cdot \left(1 - \frac{\sigma_c^2}{\sum_{c'} \sigma_{c'}^2}\right)$$

불확실성이 높은 클래스일수록 임계값을 낮춰 더 많은 샘플을 필터링하고, 불확실성이 낮은 클래스는 더 높은 임계값으로 정교하게 학습합니다.

#### (d) 전체 손실 함수

$$\mathcal{L}_{total} = \mathcal{L}_{src} + \lambda_{adv} \mathcal{L}_{adv} + \lambda_{seg} \mathcal{L}_{seg}^{target}$$

- $\mathcal{L}_{src}$: 소스 도메인 지도 학습 손실
- $\mathcal{L}_{adv}$: 적대적 정렬 손실 (adversarial alignment)
- $\mathcal{L}_{seg}^{target}$: 불확실성 기반 의사 레이블 학습 손실

---

### 2-3. 모델 구조

모델은 크게 세 모듈로 구성됩니다:

| 모듈 | 설명 |
|------|------|
| **Segmentation Network** | DeepLab 계열 backbone (ResNet-101 + ASPP), 세그멘테이션 예측 출력 |
| **Uncertainty Estimation Head** | 분산 $\sigma^2$를 픽셀별로 예측하는 별도 헤드 |
| **Dynamic Pseudo Label Generator** | 불확실성에 따라 신뢰도 임계값을 동적으로 조정하여 의사 레이블 생성 |

입력 이미지가 주어지면 모델은 시맨틱 세그멘테이션 예측과 함께 예측에 대한 불확실성을 동시에 출력합니다.

---

### 2-4. 성능 향상

세 가지 벤치마크(GTA5→Cityscapes, SYNTHIA→Cityscapes, Cityscapes→Oxford RobotCar)에서 평가하였으며, 실험을 통해 제안 방법이 예측 분산에 따른 동적 신뢰도 임계값 설정, 노이즈 있는 의사 레이블 교정, 그리고 기존 의사 레이블 학습 방식 대비 유의미한 성능 향상을 달성함을 입증합니다.

벤치마크별 대표 성능 (mIoU, %):

| 벤치마크 | 기존 Pseudo Label | 제안 방법 (Seg-Uncert) |
|----------|------------------|----------------------|
| GTA5 → Cityscapes | ~44–46 | **~49** |
| SYNTHIA → Cityscapes | ~44–46 | **~48** |
| Cityscapes → Oxford RobotCar | baseline 대비 향상 | 경쟁력 있는 성능 |

> ⚠️ 정확한 수치는 논문 원문 Table을 직접 확인하시기 바랍니다.

### 2-5. 한계

- **계산 비용 증가**: Monte Carlo 샘플링을 통한 불확실성 추정은 단순 pseudo label 방식 대비 추론 및 학습 비용 증가
- **단일 모델 불확실성**: Teacher-Student 구조가 아닌 단일 모델에서 추정하므로, 불확실성의 보정(calibration) 신뢰도가 제한적
- **Transformer 미사용**: DeepLabV2/ResNet 기반으로 ViT/Transformer 시대의 최신 방법들과의 성능 격차 존재
- **멀티 소스 도메인 미지원**: 단일 소스 → 단일 타겟에 특화되어 다중 소스 시나리오 미적용

---

## 3. 모델의 일반화 성능 향상 가능성

불확실성 추정은 의사 레이블을 사용하는 도메인 적응 모델의 중요한 도구로, 의사 레이블의 품질을 향상시킬 뿐만 아니라 모델이 더 효과적으로 샘플 선택 및 탐색과 활용의 균형을 잡도록 돕습니다.

일반화 성능 향상 가능성을 구체적으로 살펴보면:

#### (1) 동적 임계값의 일반화 기여
기존 고정 임계값 방식은 클래스 불균형 및 도메인 이동 정도에 따라 부적절한 필터링을 야기합니다. 본 방법의 **클래스별 동적 임계값**은 다양한 타겟 도메인에서도 안정적으로 작동할 수 있어 일반화 가능성이 높습니다.

#### (2) 불확실성의 정규화 효과
$\log \sigma^2$ 정규화 항은 모델이 쉬운 패턴에 과적합하는 것을 방지하고, 타겟 도메인의 다양한 분포에서도 견고한 특성 학습을 유도합니다.

#### (3) Cross-City 벤치마크 검증
제안 방법은 두 가지 합성-실제 세그멘테이션 벤치마크(GTA5→Cityscapes, SYNTHIA→Cityscapes)뿐 아니라 도시 간 벤치마크(Cityscapes→Oxford RobotCar)에서도 평가됩니다. 이는 단순 합성→실제 시나리오를 넘어 현실적인 도메인 이동에서도 일반화 능력이 있음을 시사합니다.

#### (4) 후속 연구에서의 일반화 확인
실험 결과, 불확실성을 고려하면 특히 정상 조건 도메인에서 성능이 향상되며, 역조건에서는 불확실성이 더 높게 나타나지만 이러한 조건에서의 결과는 정상 조건만큼 두드러지지 않습니다.

자기 학습(self-training) 방법들은 신뢰도 추정, 일관성 정규화, 또는 엔트로피 최소화를 통해 세그멘테이션 성능을 향상시키며, 의사 레이블을 활용하여 클래스 인식 정렬(class-aware alignment)을 가능하게 합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4-1. 연구에 미치는 영향

노이즈 있는 의사 레이블에 대응하기 위한 전략으로 신뢰도 임계값, 일관성 정규화, 도메인 믹스업, 의사 레이블 교정 등이 등장하였으며, 자기 학습 방법들은 UDA의 주요 방법론으로 부상하였습니다.

이 논문은 다음 연구들에 직접적인 영감을 제공하였습니다:

| 후속 연구 | 핵심 기여 | 관계 |
|-----------|-----------|------|
| **Uncertainty-aware Pseudo Label Refinery** (Wang et al., ICCV 2021) | 불확실성 기반 정제를 더욱 정교화 | 직접 후속 |
| **DAFormer** (Hoyer et al., CVPR 2022) | Transformer + Self-training UDA | 자기 학습 패러다임 계승 |
| **HRDA** (Hoyer et al., ECCV 2022) | 멀티스케일 고해상도 UDA | DAFormer의 확장 |
| **ProDA** (Zhang et al., CVPR 2021) | Prototype-based Pseudo Label Denoising | 의사 레이블 노이즈 제거 연구 |

자기 학습 방법들은 안정성과 우수한 성능으로 UDA의 지배적인 방법론으로 자리 잡았습니다. DAFormer는 Transformer 아키텍처와 자기 학습을 UDA에 최초로 통합한 방법이며, HRDA는 스케일 어텐션 모듈을 통해 고품질과 저품질 이미지를 활용하여 세그멘테이션 결과를 향상시킵니다.

### 4-2. 2020년 이후 최신 연구 비교 분석

최근 UDA 연구들은 적대적 학습과 자기 학습의 두 가지 방법을 통해 도메인 불변 지식을 학습하려 합니다. 적대적 학습은 도메인 판별기를 속이는 방식으로 소스와 타겟 도메인 간 전역 분포를 정렬하지만, 타겟 도메인에서 서로 다른 카테고리 간 특성 분리 가능성을 보장하지 못합니다.

| 방법 | 연도 | 핵심 아이디어 | GTA5→City mIoU |
|------|------|--------------|----------------|
| **Seg-Uncert (본 논문)** | 2020 | 예측 분산 기반 불확실성 교정 | ~49 |
| **IAST** (ECCV 2020) | 2020 | 인스턴스 적응적 자기 학습 | ~51 |
| **DAFormer** (CVPR 2022) | 2022 | Transformer + Self-training | ~68 |
| **HRDA** (ECCV 2022) | 2022 | 멀티스케일 context-aware UDA | ~73 |

DAFormer는 UDA에 Transformer 아키텍처와 자기 학습을 최초로 통합한 방법으로, 안정적인 학습 과정과 소스 도메인 과적합 방지를 위한 세 가지 간단하지만 효과적인 학습 전략을 채택합니다.

### 4-3. 앞으로 연구 시 고려할 점

#### 🔬 방법론적 발전 방향

1. **Transformer 기반 불확실성 추정**
   - 본 논문은 CNN(DeepLab) 기반으로 설계됨. ViT/Swin Transformer의 attention map을 활용한 불확실성 추정으로 확장 필요
   - DAFormer·HRDA에 불확실성 교정을 결합하면 시너지 기대

2. **Source-Free Domain Adaptation과의 결합**
   - 소스 데이터 없이 타겟 도메인만으로 적응하는 **Source-Free UDA** 시나리오에서 불확실성 기반 교정의 중요성이 더 커짐
   - 프라이버시 보호 및 실용적 배포 관점에서 중요

3. **Open-Vocabulary / Foundation Model 시대**
   - SAM, CLIP 등 foundation model이 등장한 현재, 대규모 사전학습 모델의 불확실성 추정과 결합한 도메인 적응 연구 필요

4. **멀티 소스 / 멀티 타겟 도메인**
   최근 연구들은 주변 확률 분포 정렬만으로는 충분하지 않으며, 조건부 확률 분포 정렬도 지식 이전에 마찬가지로 중요하다는 점을 지적합니다. 이를 고려한 멀티 도메인 불확실성 모델링 연구가 필요합니다.

5. **불확실성 보정(Calibration) 연구**
   - 예측된 불확실성이 실제 오류와 얼마나 일치하는지(calibration)에 대한 정량적 평가 필요
   - ECE(Expected Calibration Error) 등 보정 지표를 함께 보고하는 연구 필요

6. **온라인 학습 / 연속 학습**
   - 새로운 도메인 데이터가 순차적으로 도입되는 환경에서 불확실성 기반 의사 레이블 교정의 적용 연구

#### ⚠️ 주의해야 할 점

- **Monte Carlo Dropout의 계산 비용**: 실시간 자율주행 등 응용에서 추론 지연 최소화 방안 필요
- **Confirmation Bias**: 자기 학습의 고질적 문제인 확증 편향(잘못된 의사 레이블이 반복 강화)을 불확실성만으로 완전히 해결하기 어려움 — Teacher-Student, Contrastive Learning 등 보완 전략 병행 필요
- **벤치마크 편향**: GTA5/SYNTHIA→Cityscapes 벤치마크에서의 포화 현상으로 인해, 의료 영상, 위성 영상 등 더 다양한 도메인 쌍에서의 검증 필요

---

## 📚 참고 자료 (출처)

1. **arXiv 원문**: Zhedong Zheng, Yi Yang, "Rectifying Pseudo Label Learning via Uncertainty Estimation for Domain Adaptive Semantic Segmentation," arXiv:2003.03773 — https://arxiv.org/abs/2003.03773

2. **IJCV 게재본 (Springer)**: International Journal of Computer Vision, Vol. 129(4), pp. 1106–1120, 2021 — https://link.springer.com/article/10.1007/s11263-020-01395-y

3. **DeepAI 논문 페이지**: https://deepai.org/publication/rectifying-pseudo-label-learning-via-uncertainty-estimation-for-domain-adaptive-semantic-segmentation

4. **NASA ADS Abstract**: https://ui.adsabs.harvard.edu/abs/2020arXiv200303773Z/abstract

5. **ResearchGate PDF**: https://www.researchgate.net/publication/339814420

6. **An Uncertainty-aware Domain Adaptive Semantic Segmentation Framework** (Autonomous Intelligent Systems, 2024) — https://link.springer.com/article/10.1007/s43684-024-00070-0

7. **Pseudo Labels for Unsupervised Domain Adaptation: A Review** (MDPI Electronics, 2023) — https://www.mdpi.com/2079-9292/12/15/3325

8. **DAFormer GitHub**: https://github.com/lhoyer/DAFormer

9. **HRDA GitHub / ECCV 2022 Supplementary**: https://github.com/lhoyer/HRDA

10. **Pseudolabel guided pixels contrast for domain adaptive semantic segmentation** (Scientific Reports / PMC, 2025) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11685941/

11. **Unsupervised Domain Adaptation for Semantic Segmentation with Pseudo Label Self-Refinement** (arXiv:2310.16979) — https://arxiv.org/html/2310.16979v2
