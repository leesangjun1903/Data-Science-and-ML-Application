# Learning Transferable Features with Deep Adaptation Networks (DAN) 

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

딥 신경망의 특징(feature)은 네트워크의 하위 레이어에서는 일반적(general)이지만, 상위 레이어로 갈수록 특정 태스크에 종속적(task-specific)으로 변한다. 이로 인해 도메인 간 전이(transferability)가 급격히 저하된다. DAN은 이 문제를 **다중 레이어에서 동시에 도메인 분포를 정렬**함으로써 해결한다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| **다층 적응 (Multi-layer Adaptation)** | fc6~fc8의 모든 태스크 특화 레이어에서 동시에 도메인 정렬 수행 |
| **다중 커널 MMD (MK-MMD)** | 단일 커널 한계를 극복하여 고차·저차 통계량 동시 매칭 |
| **선형 시간 추정** | $O(n^2)$ → $O(n)$으로 복잡도 감소, 편향 없는(unbiased) 추정 |
| **이론적 보장** | 도메인 적응 이론 기반으로 타깃 위험의 상한 감소를 증명 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**도메인 적응(Domain Adaptation)** 에서는 레이블이 있는 소스 도메인 $\mathcal{D}_s = \{(\mathbf{x}_i^s, y_i^s)\}\_{i=1}^{n_s}$와 레이블이 없는 타깃 도메인 $\mathcal{D}_t = \{\mathbf{x}_j^t\}\_{j=1}^{n_t}$가 존재하며, 두 도메인의 확률분포 $p$와 $q$가 서로 다르다.

핵심 문제:
- **상위 레이어의 낮은 전이성**: Yosinski et al. (2014)에 의하면 conv4~fc8로 갈수록 특징의 전이성이 급감
- **도메인 편향(Dataset Bias)**: 깊은 특징일수록 도메인 간 분포 차이가 오히려 커질 수 있음
- **기존 DDC의 한계**: 단일 레이어(fc7)에서 단일 커널 MMD만 적용 → 불충분한 적응

---

### 2-2. 제안 방법 (수식 포함)

#### (1) MK-MMD (Multiple Kernel Maximum Mean Discrepancy)

두 분포 $p$, $q$ 사이의 거리를 재현 커널 힐베르트 공간(RKHS, $\mathcal{H}_k$)에서 다음과 같이 정의한다:

$$d_k^2(p, q) \triangleq \left\| \mathbf{E}_p[\phi(\mathbf{x}^s)] - \mathbf{E}_q[\phi(\mathbf{x}^t)] \right\|_{\mathcal{H}_k}^2 \tag{1}$$

$p = q$인 경우 $d_k^2(p, q) = 0$이 성립한다 (특성 커널의 특성).

다중 커널은 $m$개의 PSD 커널 $\{k_u\}$의 볼록 조합으로 정의된다:

```math
\mathcal{K} \triangleq \left\{ k = \sum_{u=1}^{m} \beta_u k_u : \sum_{u=1}^{m} \beta_u = 1,\ \beta_u \geq 0,\ \forall u \right\}
```

#### (2) DAN 학습 목적 함수

CNN 경험적 위험(cross-entropy loss)에 MK-MMD 기반 다층 정규화 항을 추가한다:

$$\min_{\Theta} \frac{1}{n_a} \sum_{i=1}^{n_a} J\left(\theta(\mathbf{x}_i^a), y_i^a\right) + \lambda \sum_{\ell=l_1}^{l_2} d_k^2\left(\mathcal{D}_s^\ell, \mathcal{D}_t^\ell\right) \tag{4}$$

- $J$: 크로스 엔트로피 손실 함수
- $\lambda > 0$: 페널티 파라미터
- $l_1 = 6$, $l_2 = 8$ (fc6, fc7, fc8 레이어에 적용)
- $\mathcal{D}_\*^\ell = \{h_i^{*\ell}\}$: $\ell$번째 레이어의 은닉 표현

#### (3) 선형 시간 MK-MMD 추정 (Unbiased Estimate)

쿼드-튜플 $\mathbf{z}\_i \triangleq (\mathbf{x}\_{2i-1}^s, \mathbf{x}\_{2i}^s, \mathbf{x}\_{2i-1}^t, \mathbf{x}_{2i}^t)$에 대해:

$$g_k(\mathbf{z}_i) \triangleq k(\mathbf{x}_{2i-1}^s, \mathbf{x}_{2i}^s) + k(\mathbf{x}_{2i-1}^t, \mathbf{x}_{2i}^t) - k(\mathbf{x}_{2i-1}^s, \mathbf{x}_{2i}^t) - k(\mathbf{x}_{2i}^s, \mathbf{x}_{2i-1}^t)$$

$$d_k^2(p, q) = \frac{2}{n_s} \sum_{i=1}^{n_s/2} g_k(\mathbf{z}_i) \quad \Rightarrow \quad O(n) \text{ 복잡도}$$

#### (4) 미니배치 SGD 그래디언트

$$\nabla_{\Theta^\ell} = \frac{\partial J(\mathbf{z}_i)}{\partial \Theta^\ell} + \lambda \frac{\partial g_k(\mathbf{z}_i^\ell)}{\partial \Theta^\ell} \tag{5}$$

가우시안 커널에 대한 구체적 그래디언트:

$$\frac{\partial k(\mathbf{h}_{2i-1}^{s\ell}, \mathbf{h}_{2i}^{t\ell})}{\partial \mathbf{W}^\ell} = -\sum_{u=1}^{m} \frac{2\beta_u}{\gamma_u} k_u\left(\mathbf{h}_{2i-1}^{s\ell}, \mathbf{h}_{2i}^{t\ell}\right) \times \left(\mathbf{h}_{2i-1}^{s\ell} - \mathbf{h}_{2i}^{t\ell}\right) \times \left(\mathbb{I}\left[\mathbf{h}_{2i-1}^{s(\ell-1)}\right] - \mathbb{I}\left[\mathbf{h}_{2i}^{t(\ell-1)}\right]\right)^T \tag{6}$$

#### (5) 최적 커널 파라미터 $\beta$ 학습

검정력 최대화 및 2종 오류 최소화를 위한 최적화:

$$\max_{k \in \mathcal{K}} d_k^2\left(\mathcal{D}_s^\ell, \mathcal{D}_t^\ell\right) \sigma_k^{-2} \tag{7}$$

여기서 $\sigma_k^2 = \mathbf{E}\_{\mathbf{z}} g_k^2(\mathbf{z}) - [\mathbf{E}_{\mathbf{z}} g_k(\mathbf{z})]^2$ 이다. 이는 다음의 이차 프로그래밍(QP)으로 환원된다:

$$\min_{\mathbf{d}^T \boldsymbol{\beta} = 1, \boldsymbol{\beta} \geq 0} \boldsymbol{\beta}^T (\mathbf{Q} + \varepsilon \mathbf{I}) \boldsymbol{\beta} \tag{8}$$

DAN의 전체 목적 함수는 미니맥스(min-max) 문제이다:

$$\min_{\Theta} \max_{\mathcal{K}} d_k^2\left(\mathcal{D}_s^\ell, \mathcal{D}_t^\ell\right) \sigma_k^{-2}$$

---

### 2-3. 모델 구조

```
입력 이미지
    │
[conv1 ~ conv3] ── 동결(freeze): 일반적 특징, 전이 가능
    │
[conv4 ~ conv5] ── 파인튜닝(fine-tune): 약간 도메인 편향
    │
[fc6] ──────────── MK-MMD 적용 (도메인 정렬)
    │
[fc7] ──────────── MK-MMD 적용 (도메인 정렬)
    │
[fc8] ──────────── MK-MMD 적용 + 분류기 출력
    │
소스 출력 / 타깃 출력
```

- 사전학습 모델: ImageNet으로 학습된 **AlexNet** (Krizhevsky et al., 2012)
- 구현 프레임워크: **Caffe** (Jia et al., 2014)
- conv1~conv3: 동결(일반 특징 보존)
- conv4~conv5: 파인튜닝(약한 도메인 편향 보정)
- fc6~fc8: MK-MMD로 도메인 분포 정렬

---

### 2-4. 성능 향상

**Office-31 비지도 적응 (Table 1)**

| 방법 | A→W | D→W | W→D | A→D | D→A | W→A | 평균 |
|------|-----|-----|-----|-----|-----|-----|------|
| TCA | 21.5 | 50.1 | 58.4 | 11.4 | 8.0 | 14.6 | 27.3 |
| GFK | 19.7 | 49.7 | 63.1 | 10.6 | 7.9 | 15.8 | 27.8 |
| CNN | 61.6 | 95.4 | 99.0 | 63.8 | 51.1 | 49.8 | 70.1 |
| DDC | 61.8 | 95.0 | 98.5 | 64.4 | 52.1 | 52.2 | 70.6 |
| **DAN** | **68.5** | **96.0** | **99.0** | **67.0** | **54.0** | **53.1** | **72.9** |

**Office-10 + Caltech-10 비지도 적응 (Table 2)**: DAN 평균 87.3% vs. DDC 84.6%

---

### 2-5. 한계

1. **AlexNet 의존성**: 당시 AlexNet 기반 설계로, 더 강력한 백본(ResNet, ViT 등)으로의 직접 확장은 논문에서 논의되지 않음
2. **주변 분포만 정렬**: MK-MMD는 주변 분포(marginal distribution) $p(\mathbf{x})$를 매칭하며, 조건부 분포 $p(y|\mathbf{x})$의 불일치는 완전히 해결하지 못함
3. **계산 비용**: $\beta$ 학습의 QP 문제는 $O(m^2 n)$ 비용 소요
4. **레이어 범위의 수동 설정**: $l_1=6, l_2=8$로 고정하여 태스크별 최적 레이어 범위를 자동 결정하지 못함
5. **합성곱 레이어 미적응**: conv1~conv3은 동결되어 도메인 편향이 클 경우 대응이 제한적
6. **단순 도메인 쌍**: 멀티소스 도메인 적응이나 연속적 도메인 시프트 시나리오는 다루지 않음

---

## 3. 일반화 성능 향상 가능성

### 3-1. 이론적 보장 (Theorem 1)

Ben-David et al. (2007, 2010)의 도메인 적응 이론에 기반한 타깃 위험 상한:

$$\epsilon_t(\theta) \leq \epsilon_s(\theta) + 2d_k(p, q) + C \tag{9}$$

- $\epsilon_s(\theta)$: 소스 위험 (학습 데이터로 최소화)
- $d_k(p, q)$: MK-MMD (명시적으로 최소화)
- $C$: 가설 공간 복잡도 + 이상적 가설의 위험 (상수항)

**DAN은 위 상한의 두 번째 항 $2d_k(p, q)$를 직접 최소화함으로써 타깃 위험의 상한을 줄인다.**

H-발산(H-divergence)과 MK-MMD의 관계:

$$d_{\mathcal{H}}(p, q) \leq \hat{d}_{\mathcal{H}}(\mathcal{D}_s, \mathcal{D}_t) + C_1 \leq 2(1 + d_k(p, q)) + C_1 \tag{11}$$

따라서 MK-MMD 감소 → H-발산 감소 → 타깃 위험 감소로 이어지는 이론적 연결고리가 성립한다.

### 3-2. 실험적 근거

- **A-Distance 분석**: DAN 특징의 $\hat{d}_A = 2(1-2\epsilon)$이 CNN 특징보다 작아 도메인 구분이 어려워짐 → 도메인 불변 특징 획득 확인
- **t-SNE 시각화**: DAN 특징은 소스-타깃 간 카테고리 클러스터가 잘 정렬되어, 소스 분류기를 타깃에 직접 적용 가능
- **파라미터 민감도**: $\lambda$가 적절한 범위에서 성능 향상 → 분류 손실과 정렬 손실의 균형이 일반화에 중요

### 3-3. 일반화 향상의 메커니즘

| 메커니즘 | 설명 |
|---------|------|
| 다층 적응 | 여러 추상화 수준에서 분포 정렬 → 주변/조건부 분포 동시 근사 |
| 다중 커널 | 고차 통계량까지 매칭 → 분포 차이를 더 세밀하게 포착 |
| 편향 없는 추정 | 선형 시간 추정 → 큰 배치 사용 가능 → 안정적 학습 |
| 파인튜닝 + 사전학습 | ImageNet의 일반 표현 활용 → 소스 위험 $\epsilon_s$ 낮춤 |

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4-1. 후속 연구에 미친 영향

DAN은 이후 도메인 적응 연구의 표준적 접근법을 확립하였으며, 다음 연구들의 기반이 되었다:

| 후속 연구 | 핵심 발전 |
|---------|---------|
| **DANN** (Ganin et al., 2016) | Gradient Reversal Layer로 적대적 도메인 정렬 |
| **JAN** (Long et al., 2017) | 결합 분포(joint distribution)의 MMD 매칭 |
| **CDAN** (Long et al., 2018) | 조건부 도메인 적응 (분류기 예측 활용) |
| **MDD** (Zhang et al., 2019) | 마진 분산 발산으로 대체 |

### 4-2. 2020년 이후 최신 연구 비교 분석

#### (A) CDAN (Conditional Domain Adversarial Networks, Long et al., 2018 → 2020년대 활발히 인용)

DAN이 주변 분포만 정렬하는 것과 달리, **조건부 분포** $p(y|\mathbf{x})$까지 정렬:

$$\text{조건부 MMD} = d_k(p(\mathbf{x}^s, \hat{y}^s), q(\mathbf{x}^t, \hat{y}^t))$$

#### (B) TVT (Transferable Vision Transformer, Yang et al., 2021)

ViT(Vision Transformer) 기반 도메인 적응으로, AlexNet 기반 DAN 대비 백본 한계를 극복. Office-31에서 **92.4%** 평균 정확도 달성(DAN: 72.9%).

#### (C) PMTrans (Patch Mix Transformer, Zhu et al., 2022)

Patch 수준의 도메인 혼합으로 세밀한 분포 정렬 수행.

#### (D) 비교 분석 표

| 특성 | DAN (2015) | DANN (2016) | CDAN (2018) | TVT (2021) |
|------|-----------|------------|------------|------------|
| 정렬 방법 | MK-MMD | 적대적 학습 | 조건부 적대적 | 어텐션 + 적대적 |
| 분포 | 주변 분포 | 주변 분포 | 조건부 분포 | 패치 수준 |
| 백본 | AlexNet | AlexNet | ResNet | ViT |
| Office-31 평균 | 72.9% | 73.0% | 87.7% | 92.4% |
| 이론 보장 | ✅ | 부분적 | 부분적 | 제한적 |

> **주의**: 위 정확도 수치 중 TVT(2021)의 92.4%는 공개된 논문("TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation", Yang et al., WACV 2023)에 근거하나, 정확한 수치는 원문을 직접 확인하시기 바랍니다.

### 4-3. 향후 연구 시 고려할 점

#### ① 백본 현대화
- AlexNet → ResNet, ViT, CLIP 등의 강력한 사전학습 모델로 대체 시 MK-MMD 적응 레이어 재설계 필요
- Transformer 기반 구조에서 어느 레이어에 정렬을 적용할지 재검토

#### ② 조건부 분포 정렬
- DAN의 주변 분포 $p(\mathbf{x})$ 정렬에서 벗어나 $p(y|\mathbf{x})$까지 명시적으로 정렬하는 방법 탐색
- 의사 레이블(pseudo-label) 기반의 조건부 정렬이 성능 향상에 핵심적

#### ③ 다중 소스/타깃 도메인
- 단일 소스-타깃 쌍을 넘어 멀티소스, 오픈셋, 부분적 도메인 적응으로 확장 필요

#### ④ 커널 방법의 확장성
- MK-MMD는 여전히 강력하지만, 대규모 데이터셋에서 적대적 방법 대비 표현력이 제한될 수 있음
- 신경 접선 커널(Neural Tangent Kernel)이나 학습 가능한 커널 탐구

#### ⑤ 이론과 실험의 간극
- Theorem 1의 상한($C$ 상수)이 실용적으로 너무 느슨할 수 있으므로, 더 타이트한 일반화 경계 연구 필요

#### ⑥ 레이블 효율성과 Few-shot 결합
- 타깃 도메인에 극소량의 레이블만 존재하는 few-shot DA 시나리오에서의 MK-MMD 적용 방안

#### ⑦ 공정성 및 부정적 전이 방지
- 소스-타깃 분포 차이가 클 경우 발생하는 **부정적 전이(negative transfer)** 탐지 및 방지 메커니즘 연구

---

## 참고 자료

**주요 참고 논문 (원문 PDF 기반)**:
- **Long, M., Cao, Y., Wang, J., & Jordan, M. I. (2015)**. "Learning Transferable Features with Deep Adaptation Networks." *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, JMLR: W&CP Volume 37. *(제공된 PDF)*

**논문 내 인용 문헌**:
- Ben-David, S., Blitzer, J., Crammer, K., & Pereira, F. (2007). "Analysis of representations for domain adaptation." *NIPS*.
- Ben-David, S., et al. (2010). "A theory of learning from different domains." *Machine Learning*, 79(1-2):151–175.
- Gretton, A., et al. (2012a). "A kernel two-sample test." *JMLR*, 13:723–773.
- Gretton, A., et al. (2012b). "Optimal kernel choice for large-scale two-sample tests." *NIPS*.
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). "How transferable are features in deep neural networks?" *NIPS*.
- Tzeng, E., et al. (2014). "Deep domain confusion: Maximizing for domain invariance." *arXiv:1412.3474*.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet classification with deep convolutional neural networks." *NIPS*.

**후속 연구 (2020년 이후 비교 분석 참고)**:
- Long, M., et al. (2018). "Conditional Adversarial Domain Adaptation." *NeurIPS*.
- Yang, J., et al. (2023). "TVT: Transferable Vision Transformer for Unsupervised Domain Adaptation." *WACV 2023*.

> ⚠️ **정확도 관련 주의**: 2020년 이후 최신 연구의 구체적 수치(특히 TVT 등)는 해당 원문 논문을 직접 확인하시기를 권장합니다. 본 답변에서 제시한 수치는 공개된 논문에 기반하나, 정확한 실험 설정(프로토콜 차이 등)에 따라 수치가 달라질 수 있습니다.
