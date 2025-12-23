# An Open-Set Domain Adaptation Framework for Hyperspectral Image Classification With Pixel-Aware Weighting and Decoupled Alignment

***

### 1. 핵심 주장 및 주요 기여 요약

이 논문은 하이퍼스펙트럼 이미지(HSI) 분류 시 타겟 도메인에 **'알 수 없는 클래스(Unknown Class)'**가 존재하는 **개방형 도메인 적응(OSDA)** 환경에서 발생하는 두 가지 핵심 문제를 해결합니다.

1.  **부정적 전이(Negative Transfer) 방지:** 기존 방식이 미지의 클래스를 억지로 아는 클래스로 분류하려다 성능이 떨어지는 문제를 해결하기 위해, 아는 클래스와 모르는 클래스의 특징을 분리하여 정렬하는 **Decoupled Dual Alignment (DDA)** 전략을 제안합니다.
2.  **공간적 특징 편향(Feature Bias) 제거:** HSI 패치 내에서 중심 픽셀과 다른 라벨을 가진 이웃 픽셀(노이즈)이 학습을 방해하는 것을 막기 위해, 픽셀 단위로 신뢰도를 측정하여 가중치를 부여하는 **Pixel-Aware Adaptive Weight Learning (PAWL)** 모듈을 도입합니다.

**핵심 기여:** 픽셀 수준의 불확실성 제어(PAWL)와 클래스별 분리 정렬(DDA)을 결합하여, 개방형 환경에서 '아는 클래스 분류 정확도'와 '미지 클래스 탐지 능력'을 동시에 극대화했습니다.

***

### 2. 논문 상세 분석

#### 2.1 해결하고자 하는 문제 (Problem Definition)
*   **Open-Set 문제:** 실제 원격 탐사 환경에서는 학습 데이터(Source)에는 없지만 테스트 데이터(Target)에는 존재하는 새로운 지물(예: 새로운 건물 유형)이 등장합니다. 기존의 폐쇄형(Closed-set) DA는 이를 강제로 기존 클래스로 매핑하여 **부정적 전이**를 일으킵니다.
*   **이웃 픽셀의 불확실성:** 딥러닝 기반 HSI 분류는 주로 패치(Patch) 단위로 입력을 받습니다. 이때 패치의 라벨은 '중심 픽셀'을 따르지만, 패치 가장자리에는 다른 클래스(Heterogeneous pixels)나 미지의 클래스가 섞여 있어 특징 추출 시 **노이즈로 작용**하고 모델의 일반화 성능을 저해합니다.

#### 2.2 제안하는 방법 (Proposed Method)

이 논문은 **PWDA (Pixel-Aware Weighting and Decoupled Alignment)** 프레임워크를 제안하며, 두 가지 핵심 모듈로 구성됩니다.

**(1) Pixel-Aware Adaptive Weight Learning (PAWL)**
소스 도메인 패치 내부의 픽셀들을 3가지로 분류하고, 서로 다른 가중치를 부여하여 학습합니다.
*   **$P_{Lk}$ (Neighborhood Homogeneous):** 중심 픽셀과 라벨이 같은 픽셀 → 학습 **강화**.
*   **$P_{Nlk}$ (Neighborhood Heterogeneous):** 중심 픽셀과 라벨이 다른 픽셀 → 학습에서 **배제** (가중치 0).
*   **$P_{Un}$ (Neighborhood Unknown):** 신뢰도가 낮은 미지 픽셀 → 미지 클래스 탐지 학습에 활용.

이를 위한 가중치 손실 함수 $L_{PAWL}$은 다음과 같습니다:

$$
L_{PAWL} = w_{Lk}L_{ce}^s + w_{Un}L_{Un}
$$

여기서 $w_{Lk}$와 $w_{Un}$은 각 픽셀 그룹의 비율에 따라 적응적으로 계산되는 가중치입니다.

**(2) Decoupled Dual Alignment (DDA)**
'아는 클래스'와 '미지 클래스'의 특징을 분리(Decouple)하여 각각 정렬합니다. 이는 미지 클래스가 아는 클래스의 분포로 잘못 매핑되는 것을 막습니다.
*   **Known Alignment ($L_{align}^{kn}$):** 소스의 아는 클래스 특징과 타겟의 아는 클래스(로 추정되는) 특징을 정렬합니다.
*   **Unknown Alignment ($L_{align}^{Unk}$):** 소스의 잠재적 미지 픽셀($P_{Un}$)과 타겟의 미지 클래스 특징을 정렬합니다.

전체 목적 함수는 다음과 같습니다:

$$
L_{total} = L_{PAWL} + \lambda_{align} (L_{align}^{kn} + L_{align}^{Unk}) + \lambda_{ucl} L_{ucl}
$$

($L_{ucl}$: 타겟 도메인에서 미지 클래스를 효과적으로 분리하기 위한 Unknown Class Constraint Loss)

#### 2.3 모델 구조 (Model Structure)
*   **Feature Extractor:**
    *   **Spectral:** 1x1 Conv 및 Residual block을 사용하여 스펙트럼 특징 추출.
    *   **Spatial:** ResNet50의 중간 레이어(Layer 1~4)를 활용하여 공간 특징 추출.
    *   이 두 특징을 결합(Concatenation)하여 최종 Spectral-Spatial 특징을 생성합니다.
*   **Classifier:** $K+1$개의 클래스(K개의 아는 클래스 + 1개의 미지 클래스)를 분류하는 Softmax 분류기.

#### 2.4 성능 향상 및 한계
*   **성능:** Pavia University/Center 및 Houston 2013/2018 데이터셋 실험에서 기존 SOTA 모델(MTS, UADAL, ANNA) 대비 **HOS(Harmonic Mean of OS* and Unk)** 지표가 획기적으로 향상되었습니다. 특히 Houston 데이터셋에서는 HOS가 **8.5% 이상 향상**되었습니다.
*   **한계:**
    *   픽셀 단위 분류를 수행하므로 패치 크기가 커질수록 연산량이 증가할 수 있습니다.
    *   $P_{Nlk}$(이질적 픽셀)를 완전히 배제하는 방식이 일부 유용한 경계 정보를 손실할 가능성이 있습니다.

***

### 3. 모델의 일반화 성능 향상 가능성 분석

이 논문의 가장 큰 강점은 **"데이터의 순도(Purity)를 높여 일반화를 달성한다"**는 점입니다.

1.  **노이즈 필터링을 통한 강건성 확보:** 기존 CNN 기반 HSI 분류는 패치 내의 모든 픽셀 정보를 뭉뚱그려 학습했습니다. 반면, PAWL은 **"도움이 되는 픽셀($P_{Lk}$)"**과 **"방해가 되는 픽셀($P_{Nlk}$)"**을 명확히 구분합니다. 훈련 데이터에서 노이즈(이질적 픽셀)를 스스로 걸러내고 학습하므로, 테스트 시 깨끗한 특징(Clean Feature)만을 추출하여 타겟 도메인에서도 높은 일반화 성능을 보입니다.
2.  **이중 정렬(Dual Alignment)을 통한 경계 명확화:** 일반화 성능 저하의 주범인 'Negative Transfer(미지 클래스를 억지로 분류함)'를 DDA로 차단합니다. 아는 것은 아는 것끼리, 모르는 것은 모르는 것끼리 매칭함으로써, 모델이 **"모르는 것을 모른다고 답하는 능력"**이 향상되어 전체적인 신뢰도가 높아집니다.

***

### 4. 향후 연구 영향 및 고려사항

#### 향후 연구에 미치는 영향
*   **Pixel-Level Attention의 중요성 부각:** 단순히 이미지 패치 전체를 입력으로 쓰는 것을 넘어, 패치 내부의 구성 요소(Semantic Parts)를 따지는 미세적 접근이 HSI 도메인 적응의 새로운 트렌드가 될 것입니다.
*   **Unknown Class 활용의 재발견:** 미지 클래스를 단순히 '제거해야 할 대상'이 아니라, 소스 도메인 내의 '잠재적 미지 픽셀($P_{Un}$)'을 통해 능동적으로 학습할 수 있는 대상으로 패러다임을 전환했습니다.

#### 연구 시 고려할 점
*   **동적 임계값 설정:** PAWL에서 픽셀을 구분할 때 모델의 예측 확률에 의존하는데, 초기 학습 단계에서는 이 확률이 부정확할 수 있습니다. 학습 진행에 따른 **Curriculum Learning**이나 **Dynamic Thresholding** 기법의 도입을 고려해야 합니다.
*   **Source-Free 확장성:** 현재는 소스 데이터가 필요하지만, 실제 보안 문제 등으로 소스 데이터를 접근할 수 없는 **Source-Free OSDA** 환경에서도 이 픽셀별 가중치 아이디어가 유효할지 검증이 필요합니다.

***

### 5. 2020년 이후 관련 최신 연구 비교 분석

2020년 이후 발표된 주요 Open-Set Domain Adaptation (OSDA) 연구들과 비교했을 때, 이 논문의 위치는 다음과 같습니다.

| 연구 (연도) | 주요 방법론 (Key Method) | 특징 및 차이점 | 본 논문(PWDA)의 우위 |
| :--- | :--- | :--- | :--- |
| **MTS** (2024) [1] | **Mutual-to-Separate Framework:** 아는 클래스와 모르는 클래스를 상호 분리하여 학습 | 클래스 간 경계 최적화에 집중하나, 입력 데이터 자체의 **공간적 노이즈(패치 내 이질성)** 처리에는 취약함 | **PAWL 모듈**을 통해 입력 단계에서부터 노이즈를 제거하여 더 깨끗한 특징 학습 가능 |
| **UADAL** (2022) [2] | **Entropy-weighted Adversarial Learning:** 엔트로피를 이용해 미지 샘플을 탐지하고 정렬에서 제외 | 엔트로피만으로는 경계선에 있는 샘플(Hard Sample)과 미지 샘플을 구분하기 어려움 | 엔트로피 대신 **픽셀별 예측 일관성**을 사용하여 더 정교하게 미지 픽셀을 탐지함 |
| **ANNA** (2023) [3] | **Adjustment and Alignment:** 인과 추론(Causal Inference) 기반의 편향 제거 | 일반 이미지(VisDA 등)에 최적화되어 있어, HSI 특유의 **스펙트럼-공간적 복잡성**을 충분히 반영하지 못함 | HSI의 **Patch-based 구조**에 특화된 가중치 학습으로 HSI 데이터셋에서 더 높은 성능 달성 |
| **OSBP** (2019) [4] | **Backpropagation:** 미지 클래스를 위한 별도 경계 설정 | 초기 OSDA 모델로, 미지 클래스 탐지 성능(Unk)이 낮아 전체 정확도(HOS)가 떨어지는 경향 | **Dual Alignment**를 통해 Known/Unknown 모두에서 균형 잡힌 높은 성능 달성 |

**종합 평가:** 최신 연구들이 주로 손실 함수(Loss function)나 네트워크 구조 변경을 통해 도메인 적응을 시도하는 반면, 이 논문은 **HSI 데이터의 본질적 특성(패치 내 라벨 불일치)**에 집중하여 전처리나 입력 단의 개선 없이도 **데이터 해석 방식(Weighting)**만으로 성능을 끌어올렸다는 점에서 차별화된 가치를 가집니다.

[1](https://ieeexplore.ieee.org/document/10466372/)
[2](https://ieeexplore.ieee.org/document/10919145/)
[3](https://ieeexplore.ieee.org/document/10640978/)
[4](https://ieeexplore.ieee.org/document/10980119/)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/08fa4f4d-1b09-4ce1-af58-6f8aac105bd2/An_Open-Set_Domain_Adaptation_Framework_for_Hyperspectral_Image_Classification_With_Pixel-Aware_Weighting_and_Decoupled_Alignment.pdf)
[6](https://ieeexplore.ieee.org/document/10418259/)
[7](https://linkinghub.elsevier.com/retrieve/pii/S0924271624000248)
[8](https://ieeexplore.ieee.org/document/10985914/)
[9](https://ieeexplore.ieee.org/document/10227336/)
[10](https://ieeexplore.ieee.org/document/10530296/)
[11](https://library.imaging.org/ei/articles/36/15/COIMG-162)
[12](http://arxiv.org/pdf/2107.02067v1)
[13](https://arxiv.org/pdf/1904.05200.pdf)
[14](http://arxiv.org/pdf/2411.07392.pdf)
[15](http://arxiv.org/pdf/2309.08964.pdf)
[16](https://www.mdpi.com/2072-4292/12/7/1054/pdf)
[17](http://arxiv.org/pdf/2412.13036.pdf)
[18](https://arxiv.org/pdf/1805.12277.pdf)
[19](https://arxiv.org/html/2411.12558v1)
[20](https://arxiv.org/html/2506.09460v2)
[21](https://pure.kaist.ac.kr/en/publications/unknown-aware-domain-adversarial-learning-for-open-set-domain-ada/)
[22](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Adjustment_and_Alignment_for_Unbiased_Open_Set_Domain_Adaptation_CVPR_2023_paper.pdf)
[23](https://colab.ws/articles/10.1109%2Ftgrs.2024.3441617)
[24](https://www.sciencedirect.com/science/article/pii/S2949715924000544)
[25](https://www.sciencedirect.com/science/article/abs/pii/S0167865524003659)
[26](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Adjustment_and_Alignment_for_Unbiased_Open_Set_Domain_Adaptation_CVPR_2023_paper.html)
[27](https://www.semanticscholar.org/paper/Domain-Adaptation-in-Remote-Sensing-Image-A-Survey-Peng-Huang/cfe1160ea0ab025577580c5cd89770d9863fe0b3)
[28](https://discovery.researcher.life/download/article/d19ce22107fb3a52853281faca60fa30/full-text)
[29](https://openaccess.thecvf.com/content/ICCV2021/papers/Awais_Adversarial_Robustness_for_Unsupervised_Domain_Adaptation_ICCV_2021_paper.pdf)
[30](https://openaccess.thecvf.com/content/CVPR2024W/MAT/papers/Jahan_Unknown_Sample_Discovery_for_Source_Free_Open_Set_Domain_Adaptation_CVPRW_2024_paper.pdf)
[31](https://www.arxiv.org/pdf/2512.08989.pdf)
[32](https://arxiv.org/pdf/2506.09460.pdf)
[33](https://openaccess.thecvf.com/content/CVPR2024/papers/Wan_Unveiling_the_Unknown_Unleashing_the_Power_of_Unknown_to_Known_CVPR_2024_paper.pdf)
[34](https://arxiv.org/pdf/2502.15163.pdf)
[35](https://arxiv.org/html/2506.09460v1)
[36](https://arxiv.org/abs/2206.07551)
[37](https://levir.buaa.edu.cn/static/pdfs/2022_jun_zhang_an.pdf)
[38](https://pmc.ncbi.nlm.nih.gov/articles/PMC11085882/)
