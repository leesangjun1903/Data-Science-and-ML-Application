# Margin-aware Adversarial Domain Adaptation with Optimal Transport

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)** 문제에 대해 세 가지 핵심 개념인 **대마진 분리(Large Margin Separation)**, **적대적 학습(Adversarial Learning)**, **최적 수송(Optimal Transport, OT)** 을 통합하는 새로운 이론적 분석을 제시합니다.

기존 연구들이 타깃 도메인의 **오분류율(misclassification rate)** 을 제어하는 데 집중했다면, 이 논문은 더 엄격한 기준인 **타깃 도메인의 마진 위반율(target margin violation rate)** 에 대한 상한을 제시하는 것이 핵심 주장입니다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **이론적 기여 1** | 타깃 마진 위반율에 대한 새로운 상한 도출; Ben-David et al. (2010)을 특수 케이스로 포함 |
| **이론적 기여 2** | 정렬 항을 태스크 의존적 OT 거리로 상한 설정; 이 거리가 Wasserstein 거리보다 더 tight함을 증명 |
| **알고리즘적 기여** | 이론에서 도출된 최초의 OT 기반 적대적 DA 알고리즘(MADAOT) 제안 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**비지도 도메인 적응(UDA)** 에서 소스 도메인 $\mathcal{S}$의 레이블 데이터로 학습된 모델이 타깃 도메인 $\mathcal{T}$(레이블 없음)에서도 잘 동작하도록 하는 것입니다.

기존 이론의 한계:
- **Ben-David et al. (2010)**: 0-1 손실(오분류율)만 다루며, $\mathcal{H}\Delta\mathcal{H}$-발산 사용
- **Mansour et al. (2009)**: 삼각 부등식을 만족하는 임의 손실 함수로 일반화했지만, 마진 위반 손실은 삼각 부등식 불만족
- **Zhang et al. (2019)**: 마진을 도입했지만, 여전히 **타깃 도메인의 오분류율**을 바운딩

→ 본 논문은 이보다 더 엄격한 **타깃 마진 위반율**을 직접 바운딩하는 이론 제시

### 2.2 핵심 수식 및 이론

#### 손실 함수 정의

$$l^{\rho,\beta}(t) := \begin{cases} 1 - \frac{(t-\rho)}{\beta}, & \text{if } \rho \leq t \leq \beta + \rho \\ [t < \rho], & \text{otherwise} \end{cases}$$

여기서 $1 > \rho, \beta > 0$이며, 다음 부등식이 성립합니다:

$$l^{\rho,0}(t) = [t < \rho] \leq l^{\rho,\beta}(t) \leq l^{\rho+\beta,0}(t) = [t < \rho+\beta] \tag{1}$$

#### 분류 리스크 정의

도메인 $\mathcal{P}$에서의 마진 위반율:

$$\epsilon_{\mathcal{P}}^{\rho,\beta}(h) := \underset{\mathbf{x},y \sim \mathcal{P}}{\mathbb{P}}\left[l^{\rho,\beta}(yh(\mathbf{x}))\right]$$

- $\beta=0$일 때: $\rho$-마진 위반율
- $\rho=\beta=0$일 때: 0-1 오분류율

#### Theorem 1 (핵심 이론): 타깃 마진 위반율 상한

$\rho, \beta, \alpha > 0$이고 $\rho + \beta < \alpha < 1$일 때, 임의의 $h \in \mathcal{H}$에 대해:

$$\epsilon_{\mathcal{T}}^{\rho,0}(h) \leq \epsilon_{\mathcal{S}}^{\frac{\rho+\beta}{\alpha},0}(h) + d_{h,\mathcal{H}'}^{\rho,\beta}(\mathcal{D}_\mathcal{S}, \mathcal{D}_\mathcal{T}) + \lambda_\alpha \tag{Thm. 1}$$

여기서 **정렬 항(alignment term)**:

$$d_{h,\mathcal{H}'}^{\rho,\beta}(\mathcal{D}_\mathcal{S}, \mathcal{D}_\mathcal{T}) := \sup_{h' \in \mathcal{H}'} \left| \epsilon_\mathcal{S}^{\rho,\beta}(h, h') - \epsilon_\mathcal{T}^{\rho,\beta}(h, h') \right|$$

**비추정 항(non-estimable term)**:

$$\lambda_\alpha := \inf_{f \in \mathcal{H}'} \left[ \epsilon_\mathcal{T}^{0,0}(f) + \epsilon_\mathcal{S}^{0,0}(f) + \underset{\mathbf{x} \sim \mathcal{D}_\mathcal{S}}{\mathbb{P}}[|f| < \alpha] \right]$$

#### Proposition 1: OT 기반 볼록 정렬 상한

$$d_{h,\mathcal{H}'}^{\rho,\beta}(\mathcal{D}_\mathcal{S}, \mathcal{D}_\mathcal{T}) \leq \frac{1}{\beta} \inf_{\mathcal{D} \in \Pi} \Delta_{\mathcal{H}'}(h, \mathcal{D})$$

여기서:

$$\Delta_{\mathcal{H}'}(h, \mathcal{D}) := \sup_{h' \in \mathcal{H}'} \underset{\mathbf{x}_s, \mathbf{x}_t \sim \mathcal{D}}{\mathbb{E}}\left[|hh'(\mathbf{x}_s) - hh'(\mathbf{x}_t)|\right]$$

#### Proposition 2: 타깃 리스크에 대한 OT 바운드

$$\epsilon_{\mathcal{T}}^{\rho,0}(h) \leq \epsilon_{\mathcal{S}}^{\frac{\rho+\beta}{\alpha},0}(h) + \frac{1}{\beta}\inf_{\mathcal{D} \in \Pi}\Delta_{\mathcal{H}'}(h, \mathcal{D}) + \lambda_\alpha \tag{4}$$

#### Proposition 3: Wasserstein 거리와의 관계

$c: \mathcal{X} \times \mathcal{X} \to \mathbb{R}_+$가 메트릭이고, $\mathcal{H}, \mathcal{H}'$의 모든 가설이 $L$-Lipschitz 연속이면:

$$\sup_{h \in \mathcal{H}} \inf_{\mathcal{D} \in \Pi} \Delta_{\mathcal{H}'}(h, \mathcal{D}) \leq 2L \cdot W_1(\mathcal{D}_\mathcal{S}, \mathcal{D}_\mathcal{T})$$

여기서 Wasserstein 거리:

$$W_1(\mathcal{D}_\mathcal{S}, \mathcal{D}_\mathcal{T}) := \inf_{\mathcal{D} \in \Pi} \underset{\mathbf{x}_s, \mathbf{x}_t \sim \mathcal{D}}{\mathbb{E}}[c(\mathbf{x}_s, \mathbf{x}_t)] \tag{5}$$

이는 본 논문의 OT 정렬 항이 Wasserstein 거리보다 **더 tight한 바운드**임을 의미합니다.

### 2.3 모델 구조 및 알고리즘 (MADAOT)

#### 최적화 목표 (일반형)

$$\min_{\substack{h \in \mathcal{H} \\ \mathcal{D} \in \Pi}} \underset{\mathbf{x},y \sim \mathcal{S}}{\mathbb{E}}\left[(\rho' - y \cdot h(\mathbf{x}))_+\right] + \frac{1}{\beta}\Delta_{\mathcal{H}'}(h, \mathcal{D}) \tag{6}$$

여기서 $\rho' = \frac{\rho+\beta}{\alpha}$이고, $(\cdot)_+$는 힌지 손실의 양의 부분입니다.

#### 선형 분류기 특화 볼록 최적화 (Proposition 4)

$\mathcal{H}$를 $\ell_2$ 유계 선형 분류기 공간, $\mathcal{H}'$를 $\ell_1$ 유계 선형 분류기 공간으로 설정하면:

$$\min_{\substack{\mathbf{w} \in \mathbb{R}^n \\ \mathcal{D} \in \hat{\Pi}}} \underset{\mathbf{x},y \sim \mathcal{S}}{\mathbb{E}}\left[l(y \cdot \mathbf{w}^T \mathbf{x})\right] + \delta \left\| \underset{\mathbf{x}_s, \mathbf{x}_t \sim \mathcal{D}}{\mathbb{E}}\left[|\mathbf{D}_{st}\mathbf{w}|\right] \right\|_\infty + \zeta\|\mathbf{w}\|_2^2 \tag{7}$$

여기서 $\mathbf{D}_{st} = \mathbf{x}_s\mathbf{x}_s^T - \mathbf{x}_t\mathbf{x}_t^T$이고, $\delta, \zeta > 0$은 하이퍼파라미터입니다.

#### 이산 경험적 비용 함수 (식 9)

$$\frac{1}{m}\sum_{1 \leq i \leq m} l(y_{s,i} \cdot \mathbf{w}^T \mathbf{x}_{s,i}) + \delta \left\| \sum_{\substack{1 \leq i \leq m \\ 1 \leq j \leq n}} \gamma_{ij}|\mathbf{D}_{ij}\mathbf{w}| \right\|_\infty + \zeta\|\mathbf{w}\|_2^2 \tag{9}$$

#### 최적화 절차

블록 좌표 하강법(Block Coordinate Descent) 사용:
1. **$\mathbf{w}$ 최적화 (고정 $\boldsymbol{\Gamma}$)**: L-BFGS 준-뉴턴 방법
2. **$\boldsymbol{\Gamma}$ 최적화 (고정 $\mathbf{w}$)**: Minimax 알고리즘 (Blankenship & Falk, 1976)

유사성 공간(Similarity-induced Space) 활용 (식 8):

$$\Psi(\mathbf{x}) = (K(\mathbf{x}, \tilde{\mathbf{x}}_1), \ldots, K(\mathbf{x}, \tilde{\mathbf{x}}_L))$$

### 2.4 성능 향상 및 한계

#### 성능 결과

**Moons Toy Dataset (회전 각도별 정확도 %)**

| 각도 | 10° | 20° | 30° | 40° | 50° | 70° | 90° |
|------|-----|-----|-----|-----|-----|-----|-----|
| SVM | 100 | 89.6 | 76 | 68.8 | 60 | 26.6 | 17.2 |
| OT-GL | 100 | 100 | 100 | 98.7 | 80.4 | 62.2 | 49.2 |
| JDOT | 98.9 | 95.5 | 90.6 | 86.5 | 81.5 | 70.5 | 60 |
| **MADAOT** | **99.5** | **99.3** | **99.6** | **99.6** | **98.9** | **77** | **64.1** |

**Amazon Reviews Dataset**: 12개 태스크 중 8개에서 최고 성능, 2개에서 2위 달성

#### 한계점

1. **이진 분류 한정**: 현재 이론적 분석이 이진 분류만 다루며, 다중 클래스로의 확장이 필요
2. **Shallow 구조**: 선형 분류기 기반으로 딥러닝 특징 추출의 이점 미활용
3. **하이퍼파라미터 민감성**: $\delta, \zeta$ 튜닝에 타깃 레이블 사용(비지도 설정의 순수성 일부 훼손)
4. **계산 복잡도**: 고차원 데이터에서 OT 계산 비용이 높음 (표본 복잡도가 차원에 지수적으로 증가)
5. **집중 불등식 부재**: 데이터 의존적 OT 항에 대한 집중 불등식이 아직 확립되지 않음

---

## 3. 모델 일반화 성능 향상과의 관련성

### 3.1 마진 위반율이 일반화에 미치는 영향

이 논문의 핵심 통찰 중 하나는 **소스 도메인에서의 대마진 분리가 도메인 적응의 성공 가능성을 높인다**는 것입니다.

$$\lambda_\alpha \leq \min_{f \in \mathcal{H}'} \left[\epsilon_\mathcal{S}^{\alpha,0}(f) + \epsilon_\mathcal{T}^{\alpha,0}(f)\right]$$

이 비추정 항에서 $\alpha$가 클수록(소스에서 마진이 크면):
- $\underset{\mathbf{x} \sim \mathcal{D}\_\mathcal{S}}{\mathbb{P}}[|f_\alpha(\mathbf{x})| < \alpha]$가 작아짐
- 소스 오차 항 $\epsilon_\mathcal{S}^{\frac{\rho+\beta}{\alpha},0}(h)$의 마진이 $\frac{\rho+\beta}{\alpha}$로 감소
- 좋은 소스 분류기의 공간이 커져 소스-타깃 모두에 좋은 가설 발견 가능성 증가

**직관적 해석**: 소스에서 충분한 마진으로 분리된 분류기는 타깃에서도 마진 위반 없이 일반화될 가능성이 높습니다.

### 3.2 태스크 의존적 OT 항의 일반화 효과

기존 Wasserstein 거리 기반 정렬은 **태스크와 무관하게** 두 도메인의 분포 거리를 최소화합니다. 반면, 본 논문의 정렬 항:

$$\Delta_{\mathcal{H}'}(h, \mathcal{D}) = \sup_{h' \in \mathcal{H}'} \underset{\mathbf{x}_s, \mathbf{x}_t \sim \mathcal{D}}{\mathbb{E}}\left[|hh'(\mathbf{x}_s) - hh'(\mathbf{x}_t)|\right]$$

는 **현재 분류기 $h$와 가설 공간 $\mathcal{H}'$에 의존**합니다. 이 태스크 의존성이 일반화에 기여하는 이유:

- 분류에 **무관한 특징들의 불필요한 정렬을 방지**
- 고차원에서 Wasserstein 거리의 차원의 저주 문제를 완화 (minimax 공식화)
- $\sup_{h \in \mathcal{H}} \inf_{\mathcal{D} \in \Pi} \Delta_{\mathcal{H}'}(h, \mathcal{D}) \leq 2L \cdot W_1(\mathcal{D}\_\mathcal{S}, \mathcal{D}_\mathcal{T})$이므로, 기존보다 tight한 바운드 → 더 정확한 일반화 추정

### 3.3 유사성 유도 공간에서의 일반화

$(\epsilon, \gamma, \tau)$-good 유사성 함수 이론(Balcan et al., 2008)에 의해, 유사성 공간에서 $\ell_1$ 유계 선형 분류기가 존재함이 보장됩니다. 이는 이상적 결합 가설(ideal joint hypothesis)의 존재를 이론적으로 뒷받침하며, 두 도메인 모두에서 낮은 오차를 갖는 분류기 탐색에 대한 이론적 근거를 제공합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

#### (1) 이론적 영향
- **마진 위반율 바운딩 패러다임**: 기존의 오분류율 중심 이론에서 마진 위반율 중심으로의 이론적 전환 촉진
- **태스크 의존적 도메인 발산 개념**: 이후 연구에서 분류기와 결합된 도메인 거리 측정의 중요성 부각
- **이론과 알고리즘의 연계**: 이론에서 알고리즘을 직접 도출하는 원칙적 접근법의 중요성 재확인

#### (2) 알고리즘적 영향
- **OT와 적대적 학습의 결합**: 이후 연구에서 OT 기반 적대적 DA 알고리즘의 확장 촉진
- **Shallow DA의 재평가**: 선형 분류기만으로도 딥러닝 기반 방법과 경쟁력 있는 성능 가능성 제시
- **볼록 최적화 기반 DA**: 수렴 보장이 있는 볼록 공식화의 실용적 가치 입증

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 최신 연구 비교는 제가 학습한 지식 범위 내에서 서술하며, 논문 원문에서 직접 인용된 것이 아닙니다. 구체적 수치나 세부 내용의 정확성에 대해 100% 확신하기 어려운 부분이 있음을 먼저 밝힙니다. 따라서 핵심 사실 관계 중심으로 제한적으로 서술합니다.

| 연구 방향 | 관련 연구 | MADAOT와의 비교 |
|-----------|-----------|-----------------|
| **깊은 OT 기반 DA** | DeepJDOT (Damodaran et al., 2018 후속), OT-DA with deep features | MADAOT는 shallow이지만 OT의 태스크 의존성 측면에서 이론적 선구자 역할 |
| **마진 이론 일반화** | 다중 클래스 마진 바운드 연구들 | MADAOT는 이진 분류에 한정; 다중 클래스 확장 필요성 제기 |
| **정규화된 OT (Sinkhorn)** | Flamary et al. (2021), POT library | MADAOT의 OT 계산 효율성 개선에 Sinkhorn 정규화 적용 가능 |
| **적대적 학습 이론** | Zhao et al. (2019) 이후 후속 연구들 | 부정적 전이(negative transfer) 문제를 MADAOT 이론에서 고려 필요 |

구체적으로 확인 가능한 관련 연구:
- **"Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation" (Liang et al., ICML 2020)**: 소스 데이터 없는 DA를 다루며, MADAOT의 소스 데이터 의존성과 대비
- **"Reliable Weighted Optimal Transport for Unsupervised Domain Adaptation" (Xu et al., CVPR 2020)**: 가중 OT를 통한 DA, 태스크 의존성 측면에서 MADAOT와 유사한 방향

### 4.3 향후 연구 시 고려할 점

#### (1) 이론적 확장
- **다중 클래스 마진 위반율 바운드**: 현재 이진 분류에 제한된 이론을 $K$-클래스로 일반화
- **집중 불등식 확립**: $\Delta_{\mathcal{H}'}(h, \mathcal{D})$에 대한 집중 불등식으로 유한 샘플 보장 확립
- **비추정 항 $\lambda_\alpha$의 추정 가능성**: 부분적 타깃 레이블 활용 가능한 준지도 설정에서의 분석

#### (2) 알고리즘적 개선
- **딥러닝 확장**: Theorem 1의 비볼록 정렬 항을 딥 적대적 방법으로 직접 최대화
  
  가능한 목적함수 형태:
  $$\min_{\theta_h} \max\_{\theta_{h'}} \underset{\mathbf{x}\_s, \mathbf{x}\_t \sim \mathcal{D}}{\mathbb{E}}\left[|h_{\theta_h} h'\_{\theta_{h'}}(\mathbf{x}\_s) - h_{\theta_h} h'\_{\theta_{h'}}(\mathbf{x}\_t)|\right]$$

- **Sinkhorn 정규화 적용**: OT 계산의 효율성 향상을 위해 엔트로피 정규화 OT 도입
- **하이퍼파라미터 자동 튜닝**: 타깃 레이블 없이 $\delta, \zeta$ 선택을 위한 이론적 근거 마련

#### (3) 실용적 고려사항
- **고차원 데이터**: OT의 차원의 저주 문제 해결을 위한 슬라이스된 Wasserstein 거리나 부분공간 기반 접근
- **부정적 전이(Negative Transfer) 방지**: 도메인 간 거리가 매우 클 때 적응이 오히려 해가 되는 경우에 대한 이론적 분석
- **의료 영상 등 고비용 레이블 분야**: 논문이 언급한 의료 분야 적용을 위한 레이블 효율적 방법 개발

---

## 참고자료

**주요 참고 논문 (논문 원문 내 인용)**:
- **Dhouib, S., Redko, I., Lartizien, C. (2020)**. "Margin-aware Adversarial Domain Adaptation with Optimal Transport." *Proceedings of the 37th ICML*, PMLR 119. (본 논문)
- Ben-David, S., et al. (2010). "A theory of learning from different domains." *Machine Learning*, 79(1):151–175.
- Mansour, Y., Mohri, M., Rostamizadeh, A. (2009). "Domain Adaptation: Learning Bounds and Algorithms." arXiv:0902.3430.
- Zhang, Y., Liu, T., Long, M., Jordan, M. (2019). "Bridging Theory and Algorithm for Domain Adaptation." *ICML*, pp. 7404–7413.
- Courty, N., Flamary, R., Habrard, A., Rakotomamonjy, A. (2017). "Joint Distribution Optimal Transportation for Domain Adaptation." arXiv:1705.08848.
- Courty, N., et al. (2015). "Optimal Transport for Domain Adaptation." arXiv:1507.00504.
- Redko, I., Habrard, A., Sebban, M. (2016). "Theoretical Analysis of Domain Adaptation with Optimal Transport." arXiv:1610.04420.
- Ganin, Y., et al. (2016). "Domain-Adversarial Training of Neural Networks." *JMLR*, 17(59):1–35.
- Balcan, M.-F., Blum, A., Srebro, N. (2008). "Improved guarantees for learning via similarity functions." *Computer Science Department*, p. 126.
- Santambrogio, F. (2016). *Optimal Transport for Applied Mathematicians*. Springer.
- Blitzer, J., Dredze, M., Pereira, F. (2007). "Biographies, Bollywood, Boomboxes and Blenders: Domain Adaptation for Sentiment Classification." *ACL*, pp. 187–205.
