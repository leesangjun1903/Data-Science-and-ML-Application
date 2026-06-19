
# Influence-Balanced Loss for Imbalanced Visual Classification 

> **논문 정보**
> - **저자**: Seulki Park, Jongin Lim, Younghan Jeon, Jin Young Choi (서울대학교)
> - **학회**: ICCV 2021 (Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 735–744)
> - **arXiv**: https://arxiv.org/abs/2110.02444
> - **공식 코드**: https://github.com/pseulki/IB-Loss

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 불균형 데이터 학습(imbalanced data learning)의 문제를 해결하기 위한 밸런싱 학습 방법을 제안하며, 특히 결정 경계(decision boundary)의 과적합(overfitting)을 유발하는 샘플의 영향력을 완화하는 새로운 손실 함수를 유도한다.

제안된 손실 함수는 다양한 종류의 불균형 학습 방법의 성능을 효율적으로 향상시키며, 여러 벤치마크 데이터셋 실험에서 최신(state-of-the-art) cost-sensitive 손실 방법들을 능가하는 유효성을 입증한다.

또한, 제안된 손실 함수는 특정 task, 모델, 학습 방법에 국한되지 않으므로, 리샘플링(resampling), 메타러닝(meta-learning), cost-sensitive 학습 등 다른 최신 방법들과 쉽게 결합할 수 있다.

**주요 기여 요약**

| 항목 | 내용 |
|------|------|
| 핵심 아이디어 | Influence Function 기반의 샘플별 가중치 부여 |
| 핵심 손실 함수 | IB Loss (Influence-Balanced Loss) |
| 학습 전략 | 2단계 학습 (Normal Training → IB Fine-tuning) |
| 적용 범위 | Task/모델/학습 방법에 무관하게 범용 적용 가능 |
| 발표 학회 | ICCV 2021 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

딥뉴럴네트워크(DNN)가 다양한 컴퓨터 비전 분야에서 큰 성과를 이뤘음에도 불구하고, 실제 데이터셋은 극도로 불균형한 경우가 많으며, 이로 인해 모델이 다수 클래스(majority class)에 과적합되어 소수 클래스(minority class)에서 성능이 저하되는 문제가 발생한다.

기존의 cost-sensitive 재가중치(re-weighting) 방법들은 클래스 내에서 균일한 가중치를 부여하거나, 주로 다수 클래스에 집중된 '어려운 예제(hard example)'에 과적합되어 복잡하고 편향된 결정 경계를 형성하는 문제가 있었다.

이 연구는 DNN이 고도로 불균형한 데이터로 학습될 때, 결정 경계에 영향을 미치는 샘플을 식별하고 과적합을 유발하는 샘플의 가중치를 낮추는 방법을 개발하는 것을 목표로 한다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) Influence Function 기반 이론적 동기

논문은 **Influence Function** $I(x; w)$를 이용하여 각 학습 샘플이 모델의 결정 경계에 미치는 영향력을 측정한다.

$I(x; w)$는 역헤시안(inverse Hessian) 계산이 필요한 벡터로 직접 사용하기가 거의 불가능하므로, 이를 단순하지만 효과적인 IB 가중치 계수(influence-balanced weighting factor)로 수정하여 문제를 해결한다.

#### (2) IB 가중치 계수 유도

Cross-Entropy 손실 함수는 $L(y, f(x,w)) = -\sum_k^K y_k \log f_k$로 표현되며, 여기서 $y_k$는 정답 레이블, $f_k$는 모델 출력이다. 본 논문은 결정 경계의 과적합에 관심이 있으므로, 심층 신경망의 마지막 완전 연결(FC) 레이어에서의 변화에 집중하며, $h = [h_1, \dots, h_L]^T$를 FC 레이어의 입력 피처 벡터로 정의한다.

이때 $w_{kl}$에 대한 손실의 기울기는 다음과 같이 계산된다:

$$\frac{\partial}{\partial w_{kl}} L(y, f(x, w)) = (f_k - y_k) h_l \tag{3}$$

이 결과는 sigmoid 함수를 사용하는 cross-entropy 손실이나 MSE 손실에 대해서도 동일하게 성립한다.

그 결과, IB 가중치 계수는 다음과 같이 유도된다:

$$\mathcal{IB}(x; w) = \sum_{k}^{K} \sum_{l}^{L} |(f_k - y_k) h_l|$$
$$= \sum_{k}^{K} |(f_k - y_k)| \sum_{l}^{L} |h_l|$$
$$= \|f(x, w) - y\|_1 \cdot \|h\|_1 \tag{4}$$

이 값의 역수를 재가중치 계수로 사용하여, 결정 경계에 큰 영향력을 갖는 샘플의 가중치를 낮춰 불균형 데이터 학습을 개선한다.

#### (3) 최종 IB Loss 수식

최종적으로 **Influence-Balanced Loss**는 다음과 같이 정의된다:

$$L_{IB}(y, f(x, w)) = \frac{L(y, f(x, w))}{\|f(x, w) - y\|_1 \cdot \|h\|_1} \tag{5}$$

제안된 influence-balanced 항은 결정 경계가 영향력이 큰 다수 클래스 샘플에 과적합되지 않도록 제약한다.

#### (4) 클래스별 재가중치 포함 전체 손실

실제 적용 시, IB 손실은 클래스별 재가중치 항 $\lambda_k$와 결합되어 다음과 같이 표현된다:

$$L_{IB}(x, y, \Theta) = \lambda_k \frac{L(x, y, \Theta)}{\mathcal{IB}(x, \Theta)} \tag{6}$$

여기서 $\lambda_k = \gamma n_k^{-1} / \sum_{i=1}^K n_i^{-1}$는 클래스별 재가중치 항이고, $n_i$는 $k$번째 클래스의 샘플 수이며, $\gamma$는 하이퍼파라미터이다.

---

### 2.3 모델 구조 및 학습 전략

IB 방법은 두 단계의 학습 단계를 가진다: (i) Cross-Entropy 손실을 최소화하는 일반 분류 학습(normal classification training), (ii) IB 손실 $L_{IB}$를 최소화하여 모델을 파인튜닝하는 밸런싱 학습(balancing training).

정상 학습 단계의 학습 에포크 $T_1$을 조정하여 최적 전환 시점을 결정하며, 학습 손실이 수렴하는 시점을 전환점으로 설정할 때 가장 좋은 성능을 달성할 수 있다.

**2단계 학습 파이프라인 요약:**

```
[Phase 1] 일반 학습 (Normal Training)
  - 손실 함수: Cross-Entropy Loss (CE)
  - 목적: 표현(representation) 학습
  - 에포크: T1 (학습 손실 수렴 시까지)

        ↓ (전환점: 학습 손실 수렴)

[Phase 2] 밸런싱 파인튜닝 (IB Fine-Tuning)
  - 손실 함수: L_IB (Influence-Balanced Loss)
  - 목적: 결정 경계 보정 (majority 과적합 억제)
  - 에포크: T_total - T1
```

**적용 대상**: 제안된 IB 손실은 어떤 종류의 불균형 학습 방법의 성능도 효율적으로 향상시킨다.

- 특정 backbone architecture에 종속되지 않음
- ResNet-32, ResNet-50 등 다양한 구조에 적용 실험

---

### 2.4 성능 향상

IB 손실은 CIFAR-10과 같은 불균형 데이터셋에서 클래스별 정확도, 특히 소수 클래스 정확도를 크게 향상시키며, Focal Loss, CB Loss, LDAM Loss 등 최신 cost-sensitive 방법들을 능가한다.

예를 들어, 불균형 비율(imbalance ratio) 50의 long-tailed CIFAR-10에서 'truck' 소수 클래스(학습 샘플 100개)에 대해 IB Loss는 81.1%의 정확도를 달성하며, 기준선(baseline)의 52.0%에 비해 큰 폭의 향상을 보인다.

일부 데이터셋에서는 IB 손실 단독 사용만으로도 최고 성능을 달성할 수 있으며, 이는 결정 경계 과적합에 책임 있는 샘플의 영향력을 균형 있게 하는 것이 모델 강건성에 효과적임을 시사한다. 다른 방법들과 결합 시 다수의 데이터셋에서 정확도를 추가로 향상시키며, 이는 과적합을 유발하는 영향력 있는 샘플을 낮추는 방법이 다른 기법에도 이점을 줄 수 있음을 나타낸다.

**벤치마크 성능 비교 요약 (CIFAR-10, ResNet-32)**:

| 방법 | Imbalance ratio 100 | Imbalance ratio 50 |
|------|--------------------|--------------------|
| Baseline (ERM) | ~70.4% | ~74.8% |
| Focal Loss | ~70.4% | ~76.7% |
| CB Loss | ~74.6% | ~79.3% |
| LDAM-DRW | ~77.0% | ~81.0% |
| **IB Loss (Ours)** | **~79.6%** | **~82.0%** |

> ※ 위 수치는 논문 Table 4 기준 대략값이며, 정확한 수치는 논문 원문 참고 권장.

---

### 2.5 한계점

1. **하이퍼파라미터 $T_1$ 민감성**: 정상 학습에서 IB 파인튜닝으로 전환하는 최적 시점 $T_1$을 학습 손실 수렴 시점으로 설정해야 최고 성능이 도출되므로, 데이터셋마다 최적 $T_1$ 탐색이 필요하다.

2. **Inverse Hessian 근사**: $I(x; w)$는 역헤시안 계산이 필요한 벡터로 직접 사용하기가 거의 불가능하므로 이를 $L_1$ norm으로 근사하는데, 이 근사가 모든 경우에 완전하지 않을 수 있다.

3. **클래스별 재가중치 의존성**: 클래스 빈도(class frequency) 기반 $\lambda_k$ 항을 도입하므로, 클래스 분포 정보가 필요하다.

4. **FC 레이어 중심 분석**: 결정 경계의 과적합에 집중하여 심층 신경망의 마지막 FC 레이어의 변화에만 초점을 맞추므로, 중간 레이어의 표현 학습 동역학까지 포괄하지 못한다.

---

## 3. 모델의 일반화 성능 향상 가능성

제안 방법의 핵심은 과적합된 결정 경계(overfitted decision boundary)에 큰 영향력을 갖는 샘플(light blue × 샘플)의 가중치를 낮춰 더 매끄러운 결정 경계를 만드는 것이다.

**일반화 성능 향상의 핵심 메커니즘:**

1. **과적합 억제를 통한 일반화 향상**
   결정 경계 과적합에 책임 있는 샘플의 영향력을 균형 있게 조정하는 것이 모델의 강건성(robustness)에 효과적이다.

2. **범용성을 통한 일반화 가능성 확대**
   손실 함수가 특정 task, 모델, 학습 방법에 제한되지 않으므로, 리샘플링, 메타러닝, cost-sensitive 학습 등 다양한 불균형 학습 방법과 쉽게 결합이 가능하다.

3. **2단계 학습에서의 일반화 기여**
   2단계 학습 방법은 재균형화(rebalancing) 전략의 적용을 두 번째 단계로 미루고, 더 작은 학습률을 사용하여 특징 추출기(feature extractor)가 추출한 특징 위에서 분류기가 더 나은 결정 경계를 획득하도록 한다.

4. **소수 클래스 일반화 개선**
   IB 손실은 클래스 불균형 상황에서 영향력 함수(influence function) 기반의 샘플별 가중치 부여를 통해 결정 경계의 과적합을 완화하도록 설계되어 있다.

5. **결합 학습에서의 시너지**
   다른 불균형 학습 방법과 결합 시 다수의 데이터셋에서 추가적인 정확도 향상을 달성하며, 과적합을 유발하는 영향력 있는 샘플을 낮추는 것이 다른 방법에도 이점을 제공할 수 있다.

---

## 4. 연구에 미치는 영향 및 향후 연구 시 고려할 점

### 4.1 향후 연구에 미치는 영향

**① Influence Function의 실용적 확장 개척**

Park et al.은 influence function을 학습 스킴(learning scheme)에 최초로 적용한 연구 중 하나로, long-tailed 분류에서 학습 중 influence function을 활용한 IB 손실을 설계하였다. 이는 이후 영향력 기반 학습 방법론의 토대를 마련하였다.

**② 태양 플레어 예측 등 타 도메인 확장**

IB 손실은 클래스 불균형 상황에서의 결정 경계 과적합 완화를 위한 설계로, 이후 태양 플레어 예측과 같은 전문 도메인에도 적용되며 그 범용성이 검증되었다.

**③ 2단계 학습 패러다임의 강화**

2단계 학습 방법은 재균형화 전략의 두 번째 단계 적용을 통해 디커플링 학습을 달성하고 모델의 일반화 성능을 향상시킬 수 있지만, 두 가지 충돌하는 재균형화 전략을 결합하면 모델 성능이 저하될 수 있다는 점도 밝혀져, 이 분야의 후속 연구들이 더 정교한 전략을 탐구하도록 자극하였다.

**④ 비전 언어 모델(VLM) 기반 불균형 학습으로의 연계**

손실 함수 엔지니어링(loss function engineering)은 학습 중 균형 잡힌 기울기를 얻는 것을 목표로 하며, 손실 재가중치와 로짓 조정을 포함하고, 다양한 클래스/인스턴스에 대해 손실 가중치를 조정하여 더 균형 잡힌 분포를 달성한다. IB 손실은 이러한 흐름에서 중요한 참조 기법으로 계속 인용되고 있다.

---

### 4.2 향후 연구 시 고려할 점

#### (A) 방법론적 확장 방향

| 고려 사항 | 내용 |
|-----------|------|
| **중간 레이어 영향력** | FC 레이어에만 집중하지 않고, 중간 레이어의 영향력도 분석하는 방향 |
| **동적 $T_1$ 결정** | 학습 손실 수렴 기준 외에 자동화된 전환 시점 탐색 방법 연구 |
| **VLM/Foundation Model 결합** | CLIP 등 대규모 사전학습 모델에 IB 손실 적용 가능성 탐구 |
| **Noisy Label 환경** | 레이블 노이즈와 클래스 불균형이 동시에 존재하는 실제 시나리오 적용 |

#### (B) 2020년 이후 관련 최신 연구 비교 분석

불균형 데이터셋으로 학습할 때 기울기와 손실이 다수 클래스 쪽으로 편향되는 문제가 있으며, 현재의 완화 방법들은 손실 함수 수정, 학습 샘플 재샘플링 및 증강, 2단계 학습·앙상블·표현 학습을 통한 모듈 개선 등을 포함한다.

| 연구 | 핵심 방법 | IB Loss와의 비교 |
|------|----------|----------------|
| **LDAM (Cao et al., 2019)** | 소수 클래스에 더 큰 마진 부여 | IB Loss는 샘플별 가중치 기반으로 보완 가능 |
| **CB Loss (Cui et al., 2019)** | 유효 샘플 수 기반 클래스 가중치 | IB Loss는 인스턴스 레벨로 세밀화 |
| **Balanced Meta-Softmax (Ren et al., 2020)** | Softmax 출력의 사후 조정 | IB Loss와 결합 시 추가 향상 가능 |
| **Balanced MSE (Ren et al., 2022)** | 회귀 불균형으로 확장 | IB 아이디어를 회귀 영역으로 일반화 |
| **VS Loss (Kini et al., 2021)** | 로짓에 가감 인수 모두 적용 | 이론적 근거 강화; IB와 상호 보완 |
| **VLM 기반 접근 (2023~)** | CLIP 등 대규모 모델 활용 | IB Loss를 파인튜닝 단계에 결합 가능성 |

클래스 재균형화와 정보 강화 방법 외에도 최근에는 표현 학습, 분류기 설계, 디커플링 학습(decoupled training)으로 모델을 개선하는 방법도 탐구되고 있으며, 이는 모델 구조, 손실 함수, 학습 전략, 데이터 증강 기법에 대상별 수정을 가하는 것을 포함한다.

#### (C) 종합적인 후속 연구 방향 제안

1. **Influence Function의 계층별 확장**: 현재 FC 레이어 중심 분석을 전체 네트워크 레이어로 확장
2. **자기 지도 학습(SSL)과의 결합**: 레이블 없는 데이터 활용 시 IB 가중치 적용
3. **다중 도메인 불균형**: 도메인 시프트(domain shift)와 클래스 불균형이 동시에 존재하는 환경 대응
4. **LLM/Diffusion 기반 데이터 증강과의 결합**: 샘플의 신뢰도 점수, 클래스 빈도, 모델 가중치에 대한 영향력 등 다양한 특성에 기반하여 샘플별 가중치를 적응적으로 변경하는 연구들이 이어지고 있으므로, 합성 데이터 증강 후 IB 손실 적용의 시너지 탐구

---

## 📚 참고자료 및 출처

1. **[주 논문]** Park, S., Lim, J., Jeon, Y., & Choi, J. Y. (2021). *Influence-Balanced Loss for Imbalanced Visual Classification*. ICCV 2021, pp. 735–744.
   - arXiv: https://arxiv.org/abs/2110.02444
   - CVF Open Access: https://openaccess.thecvf.com/content/ICCV2021/papers/Park_Influence-Balanced_Loss_for_Imbalanced_Visual_Classification_ICCV_2021_paper.pdf

2. **[관련 리뷰]** Liner.com — *[Quick Review] Influence-Balanced Loss for Imbalanced Visual Classification*: https://liner.com/review/influencebalanced-loss-for-imbalanced-visual-classification

3. **[인용 분석]** *Effective Decision Boundary Learning for Class Incremental Learning* (2023), arXiv:2301.05180

4. **[비교 연구]** *Revisiting Long-tailed Image Classification: Survey and Benchmarks with New Evaluation Metrics* (2023), arXiv:2302.01507

5. **[응용 연구]** *FLARE-SSM: Deep State Space Models with Influence-Balanced Loss for 72-Hour Solar Flare Prediction* (2025), arXiv:2509.09988

6. **[서베이]** *A survey on imbalanced learning: latest research, applications and future directions*, Artificial Intelligence Review, Springer (2024): https://link.springer.com/article/10.1007/s10462-024-10759-6

7. **[서베이]** *Tackling class imbalance in computer vision: a contemporary review*, Artificial Intelligence Review, Springer (2023): https://link.springer.com/article/10.1007/s10462-023-10557-6

8. **[비교 연구]** *Long-Tailed Classification with Gradual Balanced Loss and Adaptive Feature Generation*, arXiv:2203.00452

9. **[비교 연구]** *Training Over a Distribution of Hyperparameters for Enhanced Performance and Adaptability on Imbalanced Classification*, arXiv:2410.03588

10. **[비교 연구]** *Exploring Vision-Language Models for Imbalanced Learning* (2023), arXiv:2304.01457

11. **[확장 연구]** Ren, J. et al. (2022). *Balanced MSE for Imbalanced Visual Regression*. CVPR 2022. arXiv:2203.16427
