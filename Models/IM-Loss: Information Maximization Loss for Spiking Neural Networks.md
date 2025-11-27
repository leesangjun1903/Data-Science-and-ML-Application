# IM-Loss: Information Maximization Loss for Spiking Neural Networks

### **1. 핵심 주장 및 주요 기여 요약**

이 NeurIPS 2022 논문은 스파이킹 신경망(SNNs)의 근본적인 문제인 **정보 손실**을 정보 이론 관점에서 해결합니다. SNNs는 0/1 바이너리 스파이크를 사용하여 높은 에너지 효율성을 달성하지만, 연속값을 이산 스파이크로 양자화하는 과정에서 심각한 정보 손실이 발생합니다.[1]

**주요 기여**:
- **IM-Loss**: 정보 엔트로피를 직접 최대화하는 손실 함수 제안
- **Evolutionary Surrogate Gradients (ESG)**: 학습 단계에 따라 동적으로 진화하는 기울도 제공
- **정규화 기법 불필요**: 추가 연산 없이 정규화 효과 달성
- **깊은 SNN 직접 학습**: 워밍-스타트 없이 처음부터 훈련 가능[1]

***

### **2. 해결하는 문제 및 제안하는 해결책**

#### **A. 정보 손실 문제**

스파이크 활성화 함수로 인한 미분 불가능성:

$$\frac{\partial o}{\partial u} = \begin{cases} \infty, & u = V_{th} \\ 0, & \text{otherwise} \end{cases}$$

**IM-Loss 해결책**: 상호 정보 최대화 원리로부터 유도된 손실 함수

$$L_{IM} = \sum_{l=0}^{L} (\bar{U}_l - V_{th})^2$$

여기서 $\bar{U}_l$는 계층 $l$의 평균 막전위입니다. 이는 스파이크 분포가 균등할 때($p(0)=p(1)=0.5$) 정보 손실이 최소화되는 정보 이론 원리에 기반합니다.[1]

#### **B. 기울도 소실 문제**

**고정된 대체 기울기의 한계**: 초기 학습에는 불충분, 후기 학습에는 부정확

**ESG 해결책**: 진화적 점근 함수를 통한 동적 기울도

$$\sigma_i(x) = \frac{1}{2}\tanh(K_i(x - V_{th})) + \frac{1}{2}$$

$$K_i = 10^{\frac{i}{N}100(K_{max} - K_{min}) + K_{min}}$$

초기 $K_i \approx 1$ (강한 업데이트), 후기 $K_i \approx 10$ (정확한 기울도)[1]

#### **C. 전체 손실 함수**

$$L_{Total} = L_{CE} + \lambda L_{IM}$$

여기서 $\lambda = 2$로 교차 엔트로피 손실과 정보 최대화 손실을 균형[1]

***

### **3. 모델 구조 및 성능**

#### **핵심 아키텍처**

Leaky Integrate-and-Fire (LIF) 뉴런 모델:

$$u_i^l(t) = \text{decay} \cdot u_i^l(t-1) + (1-\text{decay})(1-o_i^{l-1}(t)) + I_i^l(t)$$

$$o_i^l(t) = \begin{cases} 1, & u_i^l(t) \geq V_{th} \\ 0, & \text{otherwise} \end{cases}$$

역전파는 ESG를 통한 대체 기울도 사용[1]

#### **성능 향상 (타임스텝 효율성)**

| 데이터셋 | 아키텍처 | 타임스텝 | 정확도 | 개선폭 |
|---------|---------|---------|------|--------|
| CIFAR-10 | ResNet-19 | 6 | 95.49% | +1.33% |
| CIFAR-10 | ResNet-19 | 2 | 93.85% | +1.51% (vs 6T baseline) |
| ImageNet | ResNet-34 | 6 | 67.43% | +3.71% |
| CIFAR-10-DVS | ResNet-19 | 10 | 72.60% | +4.80% |

**의의**: 훨씬 적은 타임스텝으로 높은 정확도 → 에너지 효율성 극적 향상[2][1]

#### **절제 실험**

ResNet-19 (CIFAR-10, T=4) 기준:
- 정규화 없음: 91.23%
- + IM-Loss: 94.29% (+3.06%)
- + ESG: 94.44% (+3.21%)
- + IM-Loss + ESG: 94.64% (+3.41%)
- + IM-Loss + ESG + tdBN: 95.40% (+4.17%)[1]

***

### **4. 모델의 한계**

#### **A. 기술적 한계**

1. **선택적 제약**: Residual 블록 내 계층에는 IM-Loss 미적용 (작은 섭동 학습과의 충돌)
2. **하이퍼파라미터**: $\lambda$, $K_{min}$, $K_{max}$ 고정값 → 최적값은 데이터셋 의존적
3. **타임스텝 의존성**: T≤2 일 때 성능 저하 (충분한 시공간 정보 부족)

#### **B. 일반화 성능 제약**

1. **아키텍처 편차**: ResNet vs VGG 간 개선폭 상이
2. **데이터셋 특이성**: 정적(CIFAR-10) vs 동적(CIFAR-10-DVS) 데이터에 따라 최적화 전략 다름
3. **Residual 블록 예외**: 모든 계층에 균등하게 적용되지 않음

#### **C. 비교 연구의 한계**

- **Dspike**: ImageNet에서 68.19% (본 방법 67.43% < Dspike) - 유한 차분 사용하여 더 정확하지만 계산량 훨씬 많음[1]

***

### **5. 일반화 성능 향상 메커니즘**

#### **정보 이론적 기초**

정보 이론에서 일반화 오차 상한:

$$\text{Generalization} \leq L_{\text{train}}(\theta) + I(Y; \theta | X)$$

**IM-Loss의 기여**:
- $H(O)$ 최대화로 스파이크 정보 용량 증가
- 제한된 정보 범위 내에서 더 효율적 학습
- 균등한 스파이크 분포로 신호 소실 방지[1]

#### **스파이크 비율 안정화**

그림 2 분석:
- 에포크 0 (미학습): 후속 계층의 스파이크 비율 → 0 (훈련 불가능)
- 에포크 1 (IM-Loss): 모든 계층에서 ~50% (안정적 훈련)
- 에포크 1000 (완료): 제한된 계층 ~50% 유지[1]

#### **ESG의 편향-분산 트레이드오프**

- **초기**: 낮은 $K_i$ → 큰 가중치 변화 → 높은 편향, 낮은 분산
- **중간**: 중간 $K_i$ → 균형잡힌 업데이트
- **후기**: 높은 $K_i$ → 미세 조정 → 낮은 편향, 낮은 분산[1]

***

### **6. 향후 연구에 미치는 영향 (2024-2025 최신 연구 기반)**

#### **A. 대규모 SNNs 확장 (SpikeLLM, ICLR 2025)**[3]

- **목표**: 70억 파라미터 스파이킹 대규모 언어모델
- **개선**: 일반화된 적분-발화(GIF) 뉴런, 스파이크 길이 압축 (T → log₂T bits)
- **의미**: IM-Loss 원리를 대규모 모델에 확장 가능성 입증

#### **B. 시공간 유연성 (Mixed Time-step Training, ICLR 2025)**[4]

- **혁신**: 다양한 타임스텝에서 일반화 가능한 SNNs
- **성과**: 완전 이벤트 드리빈 칩 배포 시 손실 없음
- **IM-Loss 연관**: 다양한 시간 구조에서도 정보 흐름 최적화 필요

#### **C. 스파이크 타이밍 정보 활용 (Beyond Rate Coding, 2025)**[5]

- **기존**: 발화율(firing rate)만 인코딩
- **새로운**: 스파이크 타이밍 정보도 활용 가능
- **확장**: IM-Loss 개념의 시간 영역 확대

#### **D. 객체 탐지 응용 (Bidirectional Dynamic Threshold SNN, 2025)**[6]

- **문제**: 객체 탐지에서 정보 손실 더욱 심각
- **해결**: 적응 임계값과 정보 최대화 손실 결합
- **검증**: IM-Loss 원리가 다른 도메인에도 적용 가능

***

### **7. 향후 연구 시 고려할 점**

#### **1. 이론적 기초 강화**

- IM-Loss와 일반화 오차 간 정량적 관계 규명
- 다양한 신경망 구조에 대한 보편성 정리
- IM-Loss 최적값의 계층-의존성 분석

#### **2. 적응형 하이퍼파라미터**

```
기존: λ = 2 (고정)
개선: λ(l,t) = f(깊이, 타임스텝, 데이터셋)

기존: K_min, K_max 고정
개선: 계층별 동적 설정
```

#### **3. 다양한 도메인 확장**

- ✓ 이미지 분류 (달성)
- ? 객체 탐지 (진행 중)
- ? 시계열 예측 (필요)
- ? 자연어 처리 (필요 - SpikeLLM 진행 중)

#### **4. 강건성 및 일반화 심화**

- 분포 외(OOD) 데이터 성능
- 적대적 공격(Adversarial) 강건성
- 클래스 불균형 데이터셋 (Long-tail)

#### **5. 신경형태 하드웨어 최적화**

- Loihi, TrueNorth 칩 위에서의 실제 구현
- 에너지-정확도 트레이드오프 분석
- 하드웨어 특성별 파라미터 최적화

#### **6. 통합 최적화 프레임워크**

$$L = \alpha L_{CE} + \beta L_{IM} + \gamma L_{sparsity} + \delta L_{complexity} + \epsilon L_{robustness}$$

***

## 결론

**IM-Loss**는 정보 이론을 SNNs에 직접 적용하여 문제의 근본 원인을 파악한 혁신적 연구입니다. 2024-2025년 최신 연구들이 이를 기반으로 대규모 모델, 시공간 유연성, 다중 도메인 확장을 추진 중이며, 향후 에너지 효율적 AI와 신경형태 컴퓨팅 분야에서 획기적 진전을 초래할 것으로 예상됩니다.[7][8][3][4][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c8310f87-b0af-4b97-a5e8-a47070b4e031/NeurIPS-2022-im-loss-information-maximization-loss-for-spiking-neural-networks-Paper-Conference.pdf)
[2](https://papers.neurips.cc/paper_files/paper/2022/file/010c5ba0cafc743fece8be02e7adb8dd-Paper-Conference.pdf)
[3](https://proceedings.iclr.cc/paper_files/paper/2025/file/510e7d39fce008a3e31de54b8f5be9ac-Paper-Conference.pdf)
[4](https://openreview.net/forum?id=9HsfTgflT7)
[5](https://arxiv.org/html/2507.16043v2)
[6](https://pmc.ncbi.nlm.nih.gov/articles/PMC12498527/)
[7](http://arxiv.org/pdf/2406.03287.pdf)
[8](https://arxiv.org/pdf/2409.02111.pdf)
[9](https://arxiv.org/abs/2411.01663)
[10](https://dergipark.org.tr/en/doi/10.12995/bilig.8402)
[11](https://www.semanticscholar.org/paper/696419f0fef87e5dc871013dfee93525cf7ddc80)
[12](https://aca.pensoft.net/article/151406/)
[13](https://www.jidc.org/index.php/journal/article/view/19790)
[14](https://onlinelibrary.wiley.com/doi/10.1155/tbed/7480710)
[15](https://invergejournals.com/index.php/ijss/article/view/117)
[16](https://www.semanticscholar.org/paper/6c708659768e470f63d06f791ff8420e7ff0feac)
[17](https://www.dovepress.com/development-and-validation-of-a-neonatal-hypothermia-prediction-model--peer-reviewed-fulltext-article-JMDH)
[18](https://onepetro.org/ARMAUSRMS/proceedings/ARMA24/ARMA24/D031S034R001/549545)
[19](https://arxiv.org/pdf/2401.10843.pdf)
[20](https://arxiv.org/pdf/2303.10780.pdf)
[21](https://arxiv.org/pdf/2208.01204.pdf)
[22](https://arxiv.org/pdf/2309.04426.pdf)
[23](https://pmc.ncbi.nlm.nih.gov/articles/PMC7339963/)
[24](http://arxiv.org/pdf/2409.01564.pdf)
[25](https://openreview.net/forum?id=Jw34v_84m2b)
[26](https://www.ijcai.org/proceedings/2023/335)
[27](https://www.nature.com/articles/s41467-024-51110-5)
[28](https://proceedings.neurips.cc/paper/2021/file/c4ca4238a0b923820dcc509a6f75849b-Paper.pdf)
[29](https://www.ijcai.org/proceedings/2025/0157.pdf)
