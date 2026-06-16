# Divergence Optimization for Noisy Universal Domain Adaptation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **Noisy Universal Domain Adaptation (Noisy UniDA)** 라는 새로운 현실적 설정을 제안합니다. 기존 UniDA 방법들이 소스 도메인의 **완전히 깨끗한 레이블**을 가정하는 한계를 극복하고자, 소스 도메인의 노이즈 레이블, 소스 프라이빗 클래스, 타겟 프라이빗 클래스를 **동시에** 처리하는 통합 프레임워크를 제안합니다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **새로운 설정 정의** | Noisy UniDA: 노이즈 레이블 + Partial DA + Open-set DA를 모두 포함 |
| **Divergence Optimization Framework** | 두 분류기의 출력 발산을 활용한 통합 해결 |
| **Joint Divergence 제안** | 타겟 프라이빗 샘플 탐지를 위한 새로운 발산 척도 |
| **포괄적 실험 검증** | Office, OfficeHome, VisDA 데이터셋에서 SOTA 달성 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**Noisy UniDA**의 세 가지 핵심 문제:

$$\text{Noisy UniDA} = \underbrace{\text{Noisy Labels}}_{\text{소스 노이즈}} + \underbrace{C \subset C_s}_{\text{Source Private}} + \underbrace{C \subset C_t}_{\text{Target Private}}$$

- **소스 노이즈**: $\exists\{x_s, y_s\}, y_s \neq y_s^{GT}$ (소스 레이블 오염)
- **소스 프라이빗 클래스**: $C \subset C_s$, $\overline{C_s} = C_s \setminus C$ (소스에만 존재하는 클래스)
- **타겟 프라이빗 클래스**: $C \subset C_t$, $\overline{C_t} = C_t \setminus C$ (타겟에만 존재하는 클래스)

기존 방법들의 한계:

| 방법 | 노이즈 레이블 | Partial DA | Open-set DA |
|------|:---:|:---:|:---:|
| DANN | ✗ | ✗ | ✗ |
| TCL | ✓ | ✗ | ✗ |
| ETN | ✗ | ✓ | ✗ |
| STA | ✗ | ✗ | ✓ |
| UAN | ✗ | ✓ | ✓ |
| DANCE | ✗ | ✓ | ✓ |
| **Proposed** | **✓** | **✓** | **✓** |

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 Symmetric KL Divergence (소스 샘플 처리)

두 분류기 $F_1$, $F_2$의 출력 간 대칭 KL 발산:

$$\mathcal{L}_{SKLD}(D_s) = \frac{1}{N}\sum_{i=1}^{N} D_{KL}(\boldsymbol{p}_1 \| \boldsymbol{p}_2) + \frac{1}{N}\sum_{i=1}^{N} D_{KL}(\boldsymbol{p}_2 \| \boldsymbol{p}_1)$$

여기서:

$$D_{KL}(\boldsymbol{p}_1 \| \boldsymbol{p}_2) = \sum_{k=1}^{|C_s|} p_1^k(y|x_s^i) \log \frac{p_1^k(y|x_s^i)}{p_2^k(y|x_s^i)}$$

$$D_{KL}(\boldsymbol{p}_2 \| \boldsymbol{p}_1) = \sum_{k=1}^{|C_s|} p_2^k(y|x_s^i) \log \frac{p_2^k(y|x_s^i)}{p_1^k(y|x_s^i)}$$

#### 2.2.2 Joint Divergence (타겟 샘플 처리)

$\mathcal{L}_{SKLD}$를 분해하면:

$$D_{KL}(\boldsymbol{p}_1 \| \boldsymbol{p}_2) = -H(\boldsymbol{p}_1(y|x_t)) + H(\boldsymbol{p}_1(y|x_t), \boldsymbol{p}_2(y|x_t))$$

따라서:

$$\mathcal{L}_{SKLD}(D_t) = \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}_{crs}(D_t) - \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}_{ent}(D_t)$$

타겟 프라이빗 샘플은 **높은 엔트로피**를 가져야 하는데, $\mathcal{L}_{SKLD}$에서 엔트로피 항이 음수로 작용하는 문제를 해결하기 위해 **Joint Divergence** 도입:

$$\mathcal{L}_{JD}(D_t) = \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}_{crs}(D_t) + \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}_{ent}(D_t)$$

여기서:
- $\mathcal{L}_{crs}(D_t) = H(\boldsymbol{p}_1(y|x_t), \boldsymbol{p}_2(y|x_t)) + H(\boldsymbol{p}_2(y|x_t), \boldsymbol{p}_1(y|x_t))$
- $\mathcal{L}_{ent}(D_t) = H(\boldsymbol{p}_1(y|x_t)) + H(\boldsymbol{p}_2(y|x_t))$

#### 2.2.3 소스 손실 함수

지도 학습 손실:

$$\mathcal{L}_{sup}(D_s) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{|C_s|} y_s^i \log p_1^k(y|x_s^i) - \frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{|C_s|} y_s^i \log p_2^k(y|x_s^i)$$

소스 손실 (발산 포함):

$$\mathcal{L}_s(D_s) = \mathcal{L}_{sup}(D_s) + \lambda \mathcal{L}_{SKLD}(D_s), \quad \lambda = 0.1$$

소형 손실 샘플 선택 (노이즈 필터링):

$$D_s' = \arg\min_{D': |D'| \geq \alpha|D_s|} \mathcal{L}_s(D_s)$$

#### 2.2.4 타겟 발산 분리 손실

임계값 $\delta = \log|C_s|$와 마진 $m=1$을 이용한 발산 분리:

$$\mathcal{L}_t(D_t) = \tilde{\mathcal{L}}_{JD}(D_t) = \frac{1}{N}\sum_{i=1}^{N}\tilde{\mathcal{L}}_{crs}(D_t) + \frac{1}{N}\sum_{i=1}^{N}\tilde{\mathcal{L}}_{ent}(D_t)$$

$$\tilde{\mathcal{L}}_{crs}(D_t) = \begin{cases} -|\mathcal{L}_{crs}(x_t) - \delta| & \text{if } |\mathcal{L}_{crs}(x_t) - \delta| > m \\ 0 & \text{otherwise} \end{cases}$$

$$\tilde{\mathcal{L}}_{ent}(D_t) = \begin{cases} -|\mathcal{L}_{ent}(x_t) - \delta| & \text{if } |\mathcal{L}_{ent}(x_t) - \delta| > m \\ 0 & \text{otherwise} \end{cases}$$

---

### 2.3 모델 구조

```
입력 (x_s 또는 x_t)
        ↓
  [Feature Generator G]  ← ResNet-50 (avg-pooling 이전까지)
        ↓
   ┌────┴────┐
[F1: FC Layer]  [F2: FC Layer]   ← 서로 다른 초기화
   ↓              ↓
 p1(y|x)        p2(y|x)         ← softmax, |C_s|-차원
```

**핵심 설계 원리:**
- $G$: 공통 특징 추출기
- $F_1$, $F_2$: 독립적으로 초기화된 두 분류기
- 동일 미니배치로 학습하지만 다른 결정 경계 형성

#### 훈련 절차 (4단계 반복)

**Step A-1**: 소형 손실 샘플 선택 후 전체 네트워크 업데이트
$$\min_{G, F_1, F_2} \mathcal{L}_s(D_s')$$

**Step A-2**: 타겟 프라이빗 샘플 탐지 및 분리
$$\min_{G, F_1, F_2} \mathcal{L}_t(D_t)$$

**Step B**: 분류기를 discriminator로 활용 (발산 최대화)

$$\min_{F_1, F_2} \mathcal{L}_s(D_s') - \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}_{crs}(D_t)$$

**Step C**: 생성기 업데이트 (발산 최소화, $n=4$회 반복)
$$\min_{G} \frac{1}{N}\sum_{i=1}^{N}\mathcal{L}_{crs}(D_t')$$

**추론 시 타겟 프라이빗 판별:**

$$\mathcal{L}_{crs}(x) > \delta \Rightarrow \text{target private}$$

---

### 2.4 성능 향상 및 한계

#### 성능 향상

| 데이터셋 | 노이즈 유형 | 최고 기존 방법 | 본 논문 | 향상폭 |
|---------|-----------|-------------|--------|--------|
| Office | P20 | DANCEsel (86.07) | **91.22** | +5.15% |
| Office | P45 | DANCEsel (56.32) | **62.49** | +6.17% |
| Office | S45 | DANCEsel (79.82) | **87.92** | +8.10% |
| OfficeHome | P45 | DANCEsel (48.45) | **51.93** | +3.48% |
| VisDA | S20 | DANCEsel (62.97) | **70.53** | +7.56% |

#### 한계

1. **하이퍼파라미터 민감성**: $\alpha, \lambda, \delta, m, n$ 총 5개 하이퍼파라미터 존재
2. **노이즈율 사전 지식 불필요하나** $\alpha$ 설정이 성능에 영향
3. **클린 데이터 대비 성능 격차**: 노이즈가 0일 때 DANCE와 유사한 수준에 그침
4. **이진 분류 한계**: 타겟 프라이빗을 단일 "unknown" 클래스로 통합하여 세분류 불가
5. **대규모 도메인** (OfficeHome P45: 51.93%)에서의 성능 개선 여지 존재

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 위한 핵심 메커니즘

#### (1) 노이즈 레이블 정제를 통한 일반화

두 분류기는 **Co-training 원리**에 기반하여, 노이즈 샘플에 대해 서로 다른 예측을 출력합니다:

$$\text{Clean sample} \Rightarrow \mathcal{L}_{SKLD} \approx 0 \quad \text{(두 분류기 일치)}$$
$$\text{Noisy sample} \Rightarrow \mathcal{L}_{SKLD} \gg 0 \quad \text{(두 분류기 불일치)}$$

이를 통해 **깨끗한 샘플로만 학습**하여 일반화 성능이 향상됩니다.

#### (2) 부분 도메인 정렬을 통한 일반화

기존 방법처럼 전체 분포를 정렬하지 않고, **공통 클래스 샘플만** 선택적으로 정렬:

$$D_t' = \{x_t^i : x_t^i \in D_t, \mathcal{L}_{crs}(x_t^i) < \delta - m\}$$

이 부분 정렬 전략은 **Negative Transfer를 방지**하여 일반화를 향상시킵니다.

#### (3) Joint Divergence의 일반화 기여

$$\mathcal{L}_{JD} = \mathcal{L}_{crs} + \mathcal{L}_{ent}$$

- $\mathcal{L}_{crs}$: 두 분류기 불일치 측정
- $\mathcal{L}_{ent}$: 각 분류기의 불확실성 측정

두 항의 합산으로 타겟 프라이빗 샘플의 **고엔트로피 특성**을 정확히 포착하여, 불필요한 샘플이 정렬 과정에 개입하지 않도록 합니다.

#### (4) 노이즈 수준별 일반화 강건성

실험 결과, 대칭 노이즈 45% (S45)에서 다른 방법들의 성능이 급격히 하락하는 반면:

$$\text{DANCE (S45, Office)} = 56.02\% \xrightarrow{\text{Ours}} 87.92\%$$

이는 **극단적 노이즈 환경**에서도 일반화가 유지됨을 시사합니다.

#### (5) 다양한 Noisy UniDA 설정 일반화

$|\overline{C_t}| = 0$ (Partial DA) → ETN과 유사, $|\overline{C_t}| = 21$ (Open-set DA) → DANCE와 유사, 그 중간 영역에서는 압도적 우위를 보여 **설정 불가지론적 일반화** 능력을 입증합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (1) 새로운 벤치마크 설정 확립
Noisy UniDA는 실제 세계의 도메인 적응 문제를 보다 현실적으로 반영하는 설정으로, 향후 이 설정을 표준 벤치마크로 채택하는 연구가 증가할 것으로 예상됩니다.

#### (2) Co-training 철학의 도메인 적응 확장
두 분류기의 발산을 활용하는 아이디어는 **자기 지도 학습(Self-supervised Learning), 연속 학습(Continual Learning), 페더레이션 학습(Federated Learning)** 등 다양한 분야로 확장 가능합니다.

#### (3) 노이즈 레이블 + 도메인 적응의 통합 연구 활성화
기존에 별개로 연구되던 두 분야를 통합한 최초의 포괄적 프레임워크로서, 후속 연구의 기초를 제공합니다.

### 4.2 향후 연구 시 고려사항

#### (1) 하이퍼파라미터 자동화
현재 $\delta = \log|C_s|$로 고정되어 있으나, 실제 환경에서는 $|C_s|$가 불명확할 수 있습니다. **Meta-learning 또는 Bayesian Optimization**을 통한 자동 하이퍼파라미터 탐색이 필요합니다.

#### (2) 타겟 프라이빗 클래스의 세분류
현재 모든 타겟 프라이빗 샘플을 단일 "unknown" 클래스로 처리하나, **Open-world Recognition** 관점에서 이를 세분화하는 연구가 필요합니다.

#### (3) 노이즈 유형의 다양화
현재는 pair flipping과 symmetric flipping만 고려하지만, **Instance-dependent Noise**, **Feature-dependent Noise** 등 더 현실적인 노이즈 모델 적용이 필요합니다.

#### (4) 소스 데이터 없는 설정 (Source-free DA)으로의 확장
최근 **Source-free Universal Domain Adaptation** 연구가 부상하고 있으며, 소스 데이터 접근이 불가능한 환경에서의 Noisy UniDA 해결이 중요한 연구 방향입니다.

#### (5) 대규모 언어/멀티모달 모델과의 결합
**CLIP, ALIGN** 등 대규모 사전 학습 모델을 특징 추출기로 활용하면 Noisy UniDA의 성능을 크게 향상시킬 수 있을 것으로 기대됩니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

본 논문(2021)과 직접 비교 및 연관된 2020년 이후 주요 연구:

### 5.1 논문 내 직접 비교 방법

| 방법 | 연도 | 핵심 아이디어 | Noisy UniDA 대비 한계 |
|------|------|------------|---------------------|
| **DANCE** (NeurIPS 2020) | 2020 | 자기지도 학습 + 이웃 클러스터링으로 UniDA | 노이즈 레이블 처리 불가 |
| **UAN** (CVPR 2019) | 2019 | 중요도 가중치로 UniDA | 노이즈 레이블 처리 불가, 노이즈 시 성능 급락 |
| **TCL** (AAAI 2019) | 2019 | Small-loss trick + DANN | Partial/Open-set 처리 불가 |

### 5.2 2020년 이후 관련 연구 동향

> **주의**: 아래 연구들은 본 논문(arXiv 2021)의 참고문헌에 포함되지 않은 후속 연구들로, 제가 확인한 범위 내에서만 기술하며, 세부 수치나 방법론에 대해 100% 확신이 없는 부분은 명시합니다.

#### Source-free Universal DA 방향
- 소스 데이터 없이 UniDA를 해결하려는 연구들이 2021-2022년에 등장하였으나, 이를 Noisy UniDA와 결합한 연구는 아직 초기 단계입니다.

#### 대규모 사전학습 모델 활용 방향
- Vision-Language 모델(CLIP 등)을 활용한 도메인 적응 연구에서 노이즈 레이블 문제를 함께 다루는 연구가 증가하고 있습니다.

---

## 참고자료

**본 논문 (주요 출처)**
- Yu, Q., Hashimoto, A., & Ushiku, Y. (2021). *Divergence Optimization for Noisy Universal Domain Adaptation*. arXiv:2104.00246v1 [cs.CV].

**논문 내 주요 인용 문헌**
- You, K., Long, M., Cao, Z., Wang, J., & Jordan, M. I. (2019). *Universal Domain Adaptation*. CVPR 2019.
- Saito, K., Kim, D., Sclaroff, S., & Saenko, K. (2020). *Universal Domain Adaptation through Self-Supervision (DANCE)*. NeurIPS 2020.
- Han, B., et al. (2018). *Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels*. NeurIPS 2018.
- Saito, K., Watanabe, K., Ushiku, Y., & Harada, T. (2018). *Maximum Classifier Discrepancy for Unsupervised Domain Adaptation (MCD)*. CVPR 2018.
- Shu, Y., Cao, Z., Long, M., & Wang, J. (2019). *Transferable Curriculum for Weakly-Supervised Domain Adaptation (TCL)*. AAAI 2019.
- Blum, A., & Mitchell, T. (1998). *Combining Labeled and Unlabeled Data with Co-training*. COLT 1998.
- Wei, H., Feng, L., Chen, X., & An, B. (2020). *Combating Noisy Labels by Agreement: A Joint Training Method with Co-regularization*. CVPR 2020.
- Ganin, Y., & Lempitsky, V. (2015). *Unsupervised Domain Adaptation by Backpropagation (DANN)*. ICML 2015.
