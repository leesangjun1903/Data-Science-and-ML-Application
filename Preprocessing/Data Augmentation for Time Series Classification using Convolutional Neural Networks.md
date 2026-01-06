# Data Augmentation for Time Series Classification using Convolutional Neural Networks

### 요약 및 핵심 기여

Le Guennec et al.(2016)의 논문은 시계열 분류(Time Series Classification, TSC) 분야에서 CNN의 실용성과 데이터 증강의 중요성을 처음으로 체계적으로 입증한 선구적 연구입니다. 핵심 주장은 **CNN이 시계열 분류에 효과적이지만, 소규모 데이터셋에서 과적합 문제에 직면한다**는 것이며, 이를 해결하기 위해 **두 가지 데이터 증강 전략(Window Slicing, Window Warping)**과 **반지도 학습(Dataset Mixing)**을 제안합니다.[1]

***

### 해결하고자 하는 문제 및 제안하는 방법

#### 1. 문제 정의

당시 시계열 분류 분야는 다음의 모순을 안고 있었습니다:[1]

- **CNN의 강점**: 이미지 분류에서 입증된 강력한 특징 추출 능력
- **CNN의 약점**: 수백만 개의 매개변수를 학습하기 위해 대규모 훈련 데이터 필요
- **TSC의 현실**: UCR 벤치마크의 128개 데이터셋 중 12개만이 1,000개 이상의 훈련 샘플 보유

#### 2. 제안하는 방법론

##### 모델 구조 (t-leNet)

논문에서 제안한 시계열 특화 CNN은 LeNet 기반으로 다음과 같이 설계됩니다:[1]

$$\text{입력 시계열} \xrightarrow{5 \text{ filters, size } 5} \text{Conv1} \xrightarrow{\text{MaxPool}(2)} \xrightarrow{20 \text{ filters, size } 5} \text{Conv2} \xrightarrow{\text{MaxPool}(4)} \text{FC 분류층}$$

이 아키텍처는 계산 효율과 과적합 방지 사이의 균형을 추구합니다.

##### 데이터 증강 기법

**가. Window Slicing (WS)**[1]

원본 시계열에서 연속적인 슬라이스를 추출하여 동일한 클래스로 할당합니다:

$$\text{슬라이스}_{i} = x_{[i:i+W]}, \quad y_{\text{슬라이스}_{i}} = y_{\text{원본}}$$

여기서 $W$는 슬라이스 크기(보통 원본의 90%)입니다. 테스트 시에는 각 슬라이스의 분류 결과에 다수결 투표를 적용합니다.

**나. Window Warping (WW)**[1]

시계열의 무작위 부분을 시간 축에서 비선형적으로 변환합니다:

$$x'_t = x_{\tau(t)}, \quad \tau(t) = t \cdot \alpha, \quad \alpha \in \{\frac{1}{2}, 2\}$$

이 기법의 장점은 **시계열 고유의 시간 왜곡을 합리적으로 모방**하면서도 클래스 의미를 보존한다는 것입니다.

**다. Dataset Mixing (DM) - 반지도 학습**[1]

여러 데이터셋으로부터 Convolution 필터를 비지도 방식으로 사전학습합니다:

$$\text{Convolution 필터} = \text{비지도 학습}(D_1 \cup D_2 \cup \cdots \cup D_n)$$

이후 특정 데이터셋별로 지도 학습 분류층만 재훈련하므로:
- Vanishing gradient 문제 해소
- 더 나은 매개변수 공간 탐색 가능
- 전이 학습의 원초적 형태

***

### 성능 향상 및 실험 결과

#### 주요 실험 결과 분석

| 평가 지표 | Window Slicing | Window Warping |
|----------|----------------|-----------------|
| 승률 | 47 wins | 49 wins |
| 패율 | 28 losses | 26 losses |
| p-값 | **0.0003** ✓ | 0.0700 ~ |
| 통계적 유의성 | 5% 수준에서 유의 | 한계적 |

UCR 벤치마크의 7개 데이터 유형(Device, ECG, Image, Motion, Sensor, Simulated, Spectro)에 걸쳐 **WS는 일관된 성능 향상**을 보였으나, **WW는 Image outlines 데이터셋에서 역효과**를 나타냈습니다. 이는 **도메인별 특이성(domain specificity)**을 시사합니다.[1]

#### 모델별 성능 비교 (선정 데이터셋)

| Dataset | PROP<br>(기존 SOTA) | t-leNet-WS<br>(기본) | t-leNet-WS+DM<br>(SVM) | 향상도 |
|---------|-------|--------|---------|--------|
| ChlorineCon. | 0.360 | 0.188 | **0.129** | ↓ 64.2% |
| ECGFiveDays | 0.178 | 0.001 | **0.002** | ↓ 98.9% |
| Gun_Point | 0.007 | 0.007 | **0.006** | ↓ 14.3% |

특히 **Dataset Mixing의 효과는 SVM 분류기 조합에서 가장 두드러졌습니다**.[1]

***

### 모델의 한계점

1. **아키텍처 제한성**
   - 2개 Convolution 레이어로 제한 (깊은 네트워크 미탐색)
   - 잔차 연결(Residual connections) 미적용
   - Global Average Pooling 부재

2. **데이터 증강의 선택적 효과**
   - WW의 일관성 부족 (Image outlines 데이터에서 악화)
   - 특정 데이터셋에서만 유효한 기법

3. **Dataset Mixing의 명확한 개선 부족**
   - PROP 앙상블 대비 여전히 낮은 성능
   - 다중 데이터셋 활용의 효과 제한적

4. **하이퍼파라미터 튜닝 부재**
   - 슬라이스 크기 최적화 미실행
   - 워핑 비율 고정 (1/2, 2)

***

### 일반화 성능 향상 가능성

#### 논문의 접근

이 논문은 **일반화 성능 향상의 두 가지 경로**를 제시합니다:

**경로 1: 데이터 증강을 통한 훈련셋 확대**
$$\text{과적합 위험} \propto \frac{\text{모델 매개변수 수}}{\text{훈련 샘플 수}}$$

WS를 적용하면 훈련셋을 원본의 약 4배로 확대 가능하며, 이는 과적합 위험을 상당히 감소시킵니다.

**경로 2: 특징 공간의 강건화**
Dataset Mixing으로 학습한 Convolution 필터는 여러 데이터셋의 공통 특징을 포착하므로, 특정 데이터셋 내 노이즈나 이상치에 덜 민감해집니다.

#### 2020년 이후 발전

후속 연구들은 다음과 같은 방식으로 일반화 성능을 더욱 향상시켰습니다:[2][3]

1. **배치 정규화 & 드롭아웃**: Vanishing gradient 해결 및 정규화
2. **Residual Learning**: 더 깊은 네트워크 학습 가능 (ResNet, 2016→현재)
3. **Inception 아키텍처**: 다중 스케일 특징 동시 추출 (InceptionTime, 2020)
4. **주의 메커니즘**: 시간 단계별 중요도 동적 가중치 (Attention, 2021+)
5. **자동 증강 정책**: 데이터 기반 최적 증강 방식 학습 (AutoAugment, 2023+)

***

### 앞으로의 연구에 미치는 영향과 고려할 점

#### 학술적 기여도

이 논문은 다음 세 가지 측면에서 후속 연구의 토대를 마련했습니다:

**1. Window Slicing/Warping의 표준화**
- 2021년 Iwana & Uchida의 경험적 조사에서 **Window Warping은 ResNet에서 최고 성능** 기법으로 재평가됨[2]
- 2024년 Gao et al. 조사에서 **Scaling + Window Warping 조합**이 가장 효과적인 것으로 확인[3]

**2. Dataset Mixing 개념의 발전**
- Transfer Learning의 초기 실험적 근거 제공
- Few-shot Learning, Domain Adaptation의 이론적 선행 연구

**3. UCR 벤치마크의 확립**
- 이후 모든 TSC 논문의 평가 기준이 됨 (지금까지 128개 → 최근 500개+ 데이터셋)

#### 향후 연구 시 고려할 점

**1. 도메인 특이성 인식**
- 데이터 유형(ECG, 센서, 신호 등)에 따라 최적 증강 기법이 다름
- 개별 증강 방법이 아닌 **조합적 접근** 필요

**2. 아키텍처 혁신**
```
t-leNet (2016): 기본 CNN
  ↓
ResNet (2016): 잔차 학습
  ↓
InceptionTime (2020): 다중 스케일 필터
  ↓
InceptionResNet + Attention (2024+): 하이브리드 구조
```

**3. 데이터 불균형 해결**
- SMOTE 기반 기법: 소수 클래스 과잉샘플링
- Focal Loss: 분류 어려움 샘플에 가중치 부여

**4. 자동화된 증강 정책**
- AutoAugment, RandAugment: 강화학습을 통한 최적 정책 탐색
- 온라인 증강: 훈련 중 동적 증강 적용

**5. 강건성과 해석 가능성**
- Adversarial Robustness: 미세한 입력 변화에 강함
- SHAP/LIME: 모델 결정 과정 설명 가능성

***

### 2020년 이후 관련 최신 연구 비교 분석

#### [A] 구조적 진화: InceptionTime vs InceptionResNet

**InceptionTime (Fawaz et al., 2020)**[4]
- Inception-v4 모듈 5개의 앙상블
- UCR 벤치마크에서 HIVE-COTE와 동등 성능
- 훨씬 빠른 학습 (HIVE-COTE는 O(N²·T⁴), InceptionTime은 거의 선형)
- 논문의 t-leNet 대비 **40-60배 깊은 구조**

**InceptionResNet (2024-2025)**[5]
- Inception 모듈 + ResNet 잔차 학습 통합
- UCR-85 벤치마크에서 InceptionFCN 대비 49/85 데이터셋에서 우수
- **깊은 네트워크의 Vanishing Gradient 문제 완전 해결**

#### [B] 데이터 증강 방법론의 진화

| 시기 | 방법 | 특징 | 성능 향상 |
|------|------|------|----------|
| **2016년** (원본 논문) | WS, WW | 수동 설계 | ±30% |
| **2021년** (Iwana & Uchida) | 12개 기법 경험적 평가 | 도메인별 추천 | ±40% |
| **2024년** (Gao et al.) | 60+개 기법 분류체계 | 자동화된 정책 탐색 | ±50% |

가장 주목할 점은 **단순 변환 기반 방법(WS)에서 생성 모델(GAN, VAE) 기반 방법으로의 전환**입니다.[3]

#### [C] 주의 메커니즘의 도입

최신 연구들은 **CNN의 지역적 특징 추출과 Attention의 전역적 의존성 모델링을 결합**합니다:[6][7]

$$\text{특징} = \text{CNN}(\text{입력}) \xrightarrow{\text{Attention}} \text{가중 조합된 특징}$$

구체적으로:
- **Temporal Attention**: 시간 단계별 중요도 학습
- **Channel Attention**: 변수 간 상호작용 캡처
- **Cross Attention**: 다중 시계열 간 관계 모델링

***

### 결론

Le Guennec et al.(2016)의 논문은 **소규모 시계열 데이터셋에서 CNN을 실용적으로 적용하는 최초의 체계적 시도**였습니다. Window Warping과 Dataset Mixing이라는 두 가지 혁신적 아이디어는 이후 8년간의 연구에서 지속적으로 검증 및 개선되었으며, 2020년대 들어 InceptionTime, Attention 메커니즘, 자동화된 증강 정책 등의 발전으로 수렴되었습니다.

향후 연구자들이 고려해야 할 핵심은:
1. **도메인 특화 증강**: 일괄적 방법이 아닌 데이터 특성별 맞춤형 접근
2. **깊은 아키텍처**: 잔차 학습과 주의 메커니즘의 적극 활용
3. **해석 가능성**: 모델 성능뿐 아닌 의사결정 과정의 투명성
4. **자동화**: 수작업 하이퍼파라미터 튜닝 최소화

***

### 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/6b65d18b-cc03-481a-a575-8f607a5b8842/AALTD16_paper_9.pdf)
[2](https://ieeexplore.ieee.org/document/10764737/)
[3](https://ieeexplore.ieee.org/document/11182667/)
[4](https://arxiv.org/abs/2507.12645)
[5](https://ieeexplore.ieee.org/document/10979940/)
[6](https://ieeexplore.ieee.org/document/11011741/)
[7](https://www.mdpi.com/2673-7426/5/4/61)
[8](https://ieeexplore.ieee.org/document/9412812/)
[9](https://ieeexplore.ieee.org/document/9243174/)
[10](https://ieeexplore.ieee.org/document/10946155/)
[11](https://ieeexplore.ieee.org/document/11136527/)
[12](https://arxiv.org/pdf/2311.03194.pdf)
[13](https://arxiv.org/pdf/2309.04732.pdf)
[14](http://arxiv.org/pdf/2310.10060.pdf)
[15](http://arxiv.org/pdf/2405.00319.pdf)
[16](http://arxiv.org/pdf/2404.16918.pdf)
[17](https://arxiv.org/pdf/1808.02455.pdf)
[18](https://pmc.ncbi.nlm.nih.gov/articles/PMC8282049/)
[19](https://onlinelibrary.wiley.com/doi/10.1002/eng2.12589)
[20](https://germain-forestier.info/publis/bigdata2022.pdf)
[21](https://www.sciencedirect.com/science/article/abs/pii/S1568494622009942)
[22](https://aaltd16.irisa.fr/files/2016/08/AALTD16_paper_9.pdf)
[23](https://www.sciencedirect.com/org/science/article/pii/S1546221825008872)
[24](https://www.emergentmind.com/topics/temporal-convolutional-networks-tcns)
[25](https://www.ijcai.org/proceedings/2021/0631.pdf)
[26](https://arxiv.org/abs/2010.00567)
[27](https://www.sciencedirect.com/science/article/abs/pii/S0952197623014252)
[28](https://stackoverflow.com/questions/65095078/time-series-classification-using-cnn)
[29](https://dl.acm.org/doi/10.1145/3649448)
[30](https://www.nature.com/articles/s41598-023-38465-3)
[31](https://maxime-devanne.com/publis/pialla_AALTD2022.pdf)
[32](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-025-50947-3)
[33](https://arxiv.org/html/2505.00302v1)
[34](https://arxiv.org/html/2310.10060v5)
[35](https://royalsocietypublishing.org/rsta/article/379/2194/20200209/41189/Time-series-forecasting-with-deep-learning-a)
[36](https://dl.acm.org/doi/abs/10.1145/3723890.3723911)
[37](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0254841)
[38](https://arxiv.org/html/2302.02515v2)
[39](https://arxiv.org/pdf/2506.13201.pdf)
[40](https://arxiv.org/pdf/2310.15978.pdf)
[41](https://arxiv.org/html/2411.04669v1)
[42](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0324294)
[43](https://arxiv.org/html/2501.13392v2)
[44](https://www.semanticscholar.org/paper/dd80f082b2fd3dc2fb82c31d9cb21390becbc46a)
[45](https://arxiv.org/pdf/2408.17059.pdf)
[46](https://arxiv.org/html/2409.02869v1)
[47](https://arxiv.org/html/2512.06630v1)
[48](https://arxiv.org/html/2511.03799v1)
[49](https://arxiv.org/html/2507.06009v1)
[50](https://pdfs.semanticscholar.org/f2b2/95e3b893f409d28e0784a0e1040e74951819.pdf)
[51](https://arxiv.org/html/2502.10721v1)
[52](https://arxiv.org/html/2408.15737v1)
[53](https://arxiv.org/html/2506.14831v2)
[54](https://pubmed.ncbi.nlm.nih.gov/33679210/)
[55](https://arxiv.org/html/2412.17452v1)
[56](https://arxiv.org/html/2510.07041v1)
[57](https://arxiv.org/html/2511.13237v1)
[58](https://www.sciencedirect.com/science/article/pii/S0010482523012908)
[59](https://arxiv.org/html/2310.10060v6)
[60](https://www.sciencedirect.com/science/article/abs/pii/S0957417424004019)
[61](https://www.academia.edu/115600447/Data_Augmentation_for_Time_Series_Classification_using_Convolutional_Neural_Networks)
[62](https://www.semanticscholar.org/paper/Data-Augmentation-for-Time-Series-Classification-Pialla-Devanne/89ae84f75f28e5beb90525d8cd8b06cacad79411)
[63](https://openreview.net/forum?id=vpJMJerXHU)
[64](http://www.cinc.org/archives/2020/pdf/CinC2020-349.pdf)
[65](https://dl.acm.org/doi/10.1145/3410530.3414348)
[66](https://essd.copernicus.org/articles/13/2753/2021/)
[67](https://ieeexplore.ieee.org/document/9313442/)
[68](https://www.semanticscholar.org/paper/57f56a1f3702d5145f26c5280dac1c2ebaa140bd)
[69](https://hess.copernicus.org/preprints/hess-2019-638/hess-2019-638-RC1.pdf)
[70](https://www.mdpi.com/1660-4601/17/14/4979)
[71](https://www.mdpi.com/2072-4292/12/19/3196)
[72](https://ieeexplore.ieee.org/document/9311041/)
[73](https://www.semanticscholar.org/paper/971602286459a3fee502456403af65e6c008ccae)
[74](https://arxiv.org/pdf/1909.04939.pdf)
[75](https://arxiv.org/pdf/2403.18687.pdf)
[76](https://arxiv.org/pdf/1910.13051.pdf)
[77](http://arxiv.org/pdf/2303.17809.pdf)
[78](http://arxiv.org/pdf/2406.14456.pdf)
[79](https://arxiv.org/pdf/2403.12371.pdf)
[80](https://arxiv.org/pdf/2306.10084v2.pdf)
[81](https://maxime-devanne.com/delegation/publis/dmkd2020.pdf)
[82](https://journals.sagepub.com/doi/abs/10.1177/17483026251348851)
[83](https://francis-press.com/uploads/papers/bXyGEjgc51ousDKPXHBj4rJocCJnrggjOtPCRN0p.pdf)
[84](https://www.worldscientific.com/doi/pdf/10.1142/S2196888824500234)
[85](https://www.sciencedirect.com/science/article/abs/pii/S0020025523000968)
[86](https://arxiv.org/abs/1909.04939)
[87](https://www.modsimworld.org/papers/2020/MODSIM_2020_paper_53_.pdf)
[88](https://www.ijcai.org/proceedings/2020/0277.pdf)
[89](https://github.com/hfawaz/InceptionTime)
[90](https://stackoverflow.com/questions/49337897/how-to-adapt-resnet-to-time-series-data)
[91](https://www.sciencedirect.com/science/article/abs/pii/S0360835223006915)
[92](https://dl.acm.org/doi/abs/10.1007/s10618-020-00710-y)
[93](https://arxiv.org/abs/1611.06455)
[94](https://pytorchtime.com/docs/stable/tutorials/classification_tutorial.html)
[95](http://arxiv.org/pdf/1611.06455.pdf)
[96](https://jisoo0-0.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/2023/05/28/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Temporal-Convolutional-Attention-Neural-Networks-for-Time-Series-Forecasting.html)
[97](https://dl.acm.org/doi/10.1007/s10618-020-00710-y)
[98](https://www.sciencedirect.com/science/article/abs/pii/S156849462100747X)
[99](https://www.semanticscholar.org/paper/InceptionTime:-Finding-AlexNet-for-time-series-Fawaz-Lucas/0ddd01d67b0056cadc0845e0e020fee944afa916)
[100](https://arxiv.org/pdf/1611.06455.pdf)
[101](https://arxiv.org/html/2510.06840v1)
[102](https://arxiv.org/abs/2510.08060)
[103](https://arxiv.org/html/2510.20222v1)
[104](https://arxiv.org/pdf/2406.06518.pdf)
[105](https://arxiv.org/html/2411.05793v1)
[106](https://arxiv.org/pdf/2210.07713.pdf)
[107](https://arxiv.org/pdf/2402.05370.pdf)
[108](https://arxiv.org/html/2311.14534v2)
[109](https://arxiv.org/html/2510.08060v1)
[110](https://arxiv.org/html/2312.06220v1)
[111](https://arxiv.org/html/2409.02869v2)
[112](https://arxiv.org/html/2404.06966v1)
[113](https://pmc.ncbi.nlm.nih.gov/articles/PMC11024260/)
[114](https://www.ijcai.org/proceedings/2025/0619.pdf)
[115](https://www.nature.com/articles/s41598-025-05877-2)
