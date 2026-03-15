# PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological Signatures

## 종합 분석 보고서

---

## 1. 핵심 주장과 주요 기여 요약

PersLay 논문은 두 가지 핵심 기여를 제시한다:

**첫째**, 그래프 위에서 **Heat Kernel Signature (HKS)**를 이용한 **확장 퍼시스턴스 다이어그램(Extended Persistence Diagram)**이라는 새로운 위상학적 서명(topological signature)을 도입하고, 이 서명이 입력 그래프와 확산 파라미터 $t$ 모두에 대해 **안정성(stability)**을 가짐을 이론적으로 증명한다.

**둘째**, 퍼시스턴스 다이어그램을 신경망에 입력할 수 있도록 **학습 가능한 벡터화 레이어**인 **PersLay**를 제안한다. 이 레이어는 Deep Sets 프레임워크를 확장하여 기존 문헌의 대부분의 벡터화 기법(Persistence Landscape, Persistence Image, Sliced Wasserstein Kernel 등)을 **특수한 경우**로 포함하는 일반적이고 유연한 프레임워크이다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

퍼시스턴스 다이어그램은 위상적 데이터 분석(TDA)의 핵심 기술자이나, 다음과 같은 본질적 문제를 가진다:

1. **비-힐베르트 공간 문제**: 퍼시스턴스 다이어그램의 메트릭 공간은 힐베르트 공간이 아니므로, 대부분의 ML 알고리즘에 직접 입력할 수 없다.
2. **가변 크기**: 서로 다른 다이어그램은 서로 다른 수의 점을 가지며, 덧셈·스칼라 곱 등 기본 연산이 정의되지 않는다.
3. **기존 벡터화 방법의 한계**: 기존 벡터화 방법들은 학습 가능한 파라미터가 적어 특정 작업에 최적화하기 어렵고, 커널 방법은 $O(n^2)$ 이상의 연산/메모리 비용이 요구되어 대규모 데이터셋에서 비실용적이다.
4. **일반 퍼시스턴스의 정보 손실**: 그래프에서 일반(ordinary) 퍼시스턴스를 사용하면 루프나 전체 연결 컴포넌트에 대해 무한 좌표가 발생하여 정보 손실이 불가피하다.

### 2.2 제안하는 방법

#### 2.2.1 확장 퍼시스턴스 다이어그램과 HKS

그래프 $G = (V, E)$의 정규화 그래프 라플라시안 $L_w = I - D^{-1/2}AD^{-1/2}$의 고유값 $\lambda_1, \ldots, \lambda_n$과 고유벡터 $\psi_1, \ldots, \psi_n$으로부터 **Heat Kernel Signature**를 정의한다:

$$\text{hks}_{G,t}(v) = \sum_{k=1}^{n} \exp(-t\lambda_k)\psi_k(v)^2$$

이 함수를 이용하여 sublevel set과 superlevel set 모두를 고려하는 **확장 퍼시스턴스 다이어그램** $\text{Dg}(G, t)$를 구성한다. 확장 퍼시스턴스는 그래프의 네 가지 위상학적 특징을 포착한다:
- $\text{Ord}_0$: 하향 가지(downwards branches)
- $\text{Rel}_1$: 상향 가지(upwards branches)
- $\text{Ext}_0^+$: 연결 컴포넌트(connected components)
- $\text{Ext}_1^-$: 루프(loops)

**안정성 정리 (Theorem 2.2)**: 그래프 $G$에 대한 라플라시안 $L_w$와 섭동된 그래프 $G'$의 라플라시안 $\tilde{L}_w = L_w + W$에 대해:

$$d_B(\text{Dg}(G, t),\, \text{Dg}(G', t)) \leqslant C(G, t)\|W\|_F$$

여기서 $C(G, t) > 0$은 $t$와 $L_w$의 스펙트럼에만 의존하는 상수이다.

**파라미터 $t$에 대한 안정성 (Theorem 2.3)**:

$$d_B(\text{Dg}(G, t),\, \text{Dg}(G, t')) \leqslant 2|t - t'|$$

즉, 다이어그램은 $t$에 대해 2-Lipschitz 연속이다.

#### 2.2.2 PersLay 레이어

PersLay는 다음과 같이 정의된다:

$$\text{PersLay}(\text{Dg}) := \mathbf{op}\left(\{w(p) \cdot \phi(p)\}_{p \in \text{Dg}}\right)$$

여기서:
- $\mathbf{op}$: 순열 불변(permutation invariant) 연산 (sum, max, min, $k$-th largest 등)
- $w: \mathbb{R}^2 \to \mathbb{R}$: 학습 가능한 가중 함수
- $\phi: \mathbb{R}^2 \to \mathbb{R}^q$: 점 변환 함수(point transformation)

세 가지 **점 변환 함수**를 제안한다:

**1) Triangle point transformation** $\phi_\Lambda$:

$$\Lambda_p(t) = \max\{0,\, y - |t - x|\}, \quad p = (x, y)$$

$$\phi_\Lambda(p) = \left[\Lambda_p(t_1), \Lambda_p(t_2), \ldots, \Lambda_p(t_q)\right]^T$$

**2) Gaussian point transformation** $\phi_\Gamma$:

$$\Gamma_p(t) = \exp\left(-\frac{\|p - t\|_2^2}{2\sigma^2}\right)$$

$$\phi_\Gamma(p) = \left[\Gamma_p(t_1), \Gamma_p(t_2), \ldots, \Gamma_p(t_q)\right]^T$$

**3) Line point transformation** $\phi_L$:

$$L_\Delta(p) = \langle p, e_\Delta \rangle + b_\Delta$$

$$\phi_L(p) = \left[L_{\Delta_1}(p), L_{\Delta_2}(p), \ldots, L_{\Delta_q}(p)\right]^T$$

이 프레임워크는 기존 방법들을 **특수 경우**로 포함한다:

| 기존 방법 | PersLay 설정 |
|---------|-----------|
| Persistence Landscape [Bub15] | $\phi = \phi_\Lambda$, $\mathbf{op} = k\text{-th largest}$, $w = 1$ |
| Persistence Silhouette [CFL+15] | $\phi = \phi_\Lambda$, $\mathbf{op} = \text{sum}$, 임의 $w$ |
| Persistence Surface [AEK+17] | $\phi = \phi_\Gamma$, $\mathbf{op} = \text{sum}$, 임의 $w$ |
| Sliced Wasserstein Kernel [CCO17] | $\phi = \phi_L$, $\mathbf{op} = k\text{-th largest}$, $w = 1$ |

#### 2.2.3 PersLay의 연속성 조건

$\mathbf{op} = \text{sum}$인 경우, 맵 $\text{Dg} \mapsto \sum_{p \in \text{Dg}} \phi(p)$가 메트릭 $d_s$ ($s \geqslant 1$)에 대해 연속이 되기 위한 **필요충분조건**은:

$$\phi(p) = \varphi(p)\|p - \Delta\|^s$$

여기서 $\|p - \Delta\|$는 점 $p$에서 대각선 $\Delta = \{(x,x) : x \in \mathbb{R}\}$까지의 거리이고, $\varphi$는 연속이고 유계인 함수이다. 특히 $s = 1$이고 $\varphi$가 1-Lipschitz일 때 안정성이 성립한다:

$$\left\|\sum_{p \in \text{Dg}_1} \phi(p) - \sum_{p' \in \text{Dg}_2} \phi(p')\right\|_\infty \leqslant d_1(\text{Dg}_1, \text{Dg}_2)$$

### 2.3 모델 구조

전체 아키텍처는 **2-layer 네트워크**로 설계된다 (Figure 3):

1. **PersLay 레이어**: 각 그래프를 여러 유형($\text{Ord}_0$, $\text{Rel}_1$, $\text{Ext}_0^+$, $\text{Ext}_1^-$)의 확장 퍼시스턴스 다이어그램으로 인코딩하고, 각 다이어그램을 독립적인 PersLay 인스턴스로 처리한 후 결과를 **연결(concatenate)**한다.
2. **Fully-connected 레이어**: 배치 정규화 후 완전 연결층을 통해 분류 출력을 생성한다.

가중 함수 $w$는 단위 정사각형을 $N \times N$ 그리드로 이산화하여 각 셀의 가중치 $w_{i,j}$를 학습 파라미터로 설정한다 ($N = 10$ 또는 $20$).

### 2.4 성능 향상

**ORBIT 데이터셋 (동적 시스템)**:

| 데이터셋 | PSS-K | PWG-K | SW-K | PF-K | **PersLay** |
|---------|-------|-------|------|------|-----------|
| ORBIT5K | 72.38(±2.4) | 76.63(±0.7) | 83.6(±0.9) | 85.9(±0.8) | **87.7(±1.0)** |
| ORBIT100K | — | — | — | — | **89.2(±0.3)** |

커널 방법으로는 처리 불가능한 100K 규모에서도 PersLay는 효과적으로 동작한다.

**그래프 분류 벤치마크**: 매우 단순한 2-layer 아키텍처임에도 REDDIT5K, REDDIT12K, COX2, DHFR, MUTAG, PROTEINS 등 다수의 데이터셋에서 기존 SOTA 방법(RetGK, FGSD, GCNN, GIN)과 **경쟁적인 성능**을 달성한다.

**Ablation study (Table 7)**: 확장 퍼시스턴스가 일반 퍼시스턴스보다 모든 데이터셋에서 우수하며, 위상 특징과 스펙트럴 특징의 결합이 각각 단독 사용보다 높은 성능을 보인다.

### 2.5 한계

1. **NCI 데이터셋에서의 낮은 성능**: 위상 정보가 해당 데이터셋에 대해 판별적이지 않아 다른 방법 대비 성능이 크게 떨어진다 (NCI1: 73.5 vs RetGK 84.5).
2. **하이퍼파라미터 민감성**: 점 변환 함수 $\phi$의 선택, 그리드 크기 $N$, 확산 파라미터 $t$ 등 하이퍼파라미터가 성능에 영향을 미치며, 데이터셋마다 최적 설정이 다르다.
3. **단순한 후단 아키텍처**: 의도적으로 2-layer 아키텍처를 사용하여 PersLay 자체의 기여를 부각시켰으나, 더 복잡한 아키텍처와의 결합 가능성은 충분히 탐구되지 않았다.
4. **파라미터 $t$ 최적화의 비효율성**: $t$를 학습 가능 파라미터로 설정하면 매 에폭마다 전체 퍼시스턴스 다이어그램을 재계산해야 하므로 실행 시간이 크게 증가하며, 성능 향상은 미미하다.
5. **연속성 조건의 제약**: 벡터화의 연속성을 보장하려면 대각선 근처의 점에 작은 가중치를 부여해야 하나, 이것이 항상 학습 과제에 적합한 것은 아니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 일반화 근거

PersLay의 일반화 성능은 다음과 같은 이론적 기반에 의해 지지된다:

**안정성 보장**: Theorem 2.2와 2.3에 의해, 입력 그래프의 작은 섭동이나 파라미터 $t$의 변화가 출력 퍼시스턴스 다이어그램에 연속적이고 제한적인 변화만을 야기한다. 이는 학습된 표현이 입력의 노이즈에 강건하다는 것을 의미하며, 일반화 성능의 핵심 전제조건이다.

$$d_B(\text{Dg}(G, t),\, \text{Dg}(G', t)) \leqslant C(G, t)\|W\|_F$$

이 부등식은 입력 공간에서의 작은 변화가 특징 공간에서도 제한된 변화를 유발함을 보장하여, 과적합 위험을 줄인다.

### 3.2 유연한 프레임워크를 통한 일반화

PersLay의 일반화 가능성은 세 가지 축에서 향상될 수 있다:

1. **점 변환 함수 $\phi$의 적응적 선택**: $\phi = \alpha_\Lambda \phi_\Lambda + \alpha_\Gamma \phi_\Gamma + \alpha_L \phi_L$ (여기서 $\alpha_\Lambda + \alpha_\Gamma + \alpha_L = 1$, $\alpha_i \geq 0$)로 설정하여 데이터에 맞게 최적 변환을 학습할 수 있다.

2. **학습 가능한 가중 함수 $w$의 정규화**: 그리드 크기 증가에 따른 과적합 경향(Table 6: MUTAG에서 $50 \times 50$ 그리드 시 train 94.1%, test 87.7%)을 방지하기 위한 적절한 정규화가 필요하다.

3. **다중 스케일 통합**: 서로 다른 $t$ 값에서 계산된 다이어그램을 결합함으로써 ($t = 0.1$과 $t = 10$), 지역적/전역적 위상 구조를 동시에 포착하여 일반화를 향상시킨다.

### 3.3 스케일러빌리티와 일반화의 관계

ORBIT5K (87.7%) → ORBIT100K (89.2%)에서 관찰되듯이, 데이터 규모 증가에 따라 분류 정확도가 향상된다. 이는 커널 방법과 달리 PersLay가 대규모 데이터셋에서도 효율적으로 학습하여 **더 나은 일반화**를 달성할 수 있음을 보여준다.

### 3.4 확장 퍼시스턴스의 일반화 기여

Table 7의 ablation study에서 확장 퍼시스턴스가 일반 퍼시스턴스 대비 일관되게 높은 성능을 보인다 (예: MUTAG 85.1 vs 70.2, REDDIT5K 55.0 vs 52.5). 이는 루프와 연결 컴포넌트에 대한 완전한 정보가 모델의 **표현력과 일반화 능력** 모두를 향상시킴을 의미한다.

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 4.1 연구에 미치는 영향

1. **TDA와 딥러닝의 교차점 확립**: PersLay는 퍼시스턴스 다이어그램을 신경망에 통합하는 표준적 방법론을 제시하여, TDA 기반 특징을 다양한 딥러닝 파이프라인에 플러그인할 수 있는 길을 열었다.

2. **벡터화 방법의 통합 프레임워크**: 기존의 분산된 벡터화 기법들을 하나의 프레임워크로 통합함으로써, 연구자들이 하이퍼파라미터 수준에서 벡터화 방법을 선택하고 비교할 수 있게 되었다.

3. **확장 퍼시스턴스의 ML 도입**: 머신러닝 맥락에서 확장 퍼시스턴스를 최초로 활용하여, 무한 좌표 문제를 해결하고 더 풍부한 위상 정보를 활용할 수 있음을 입증했다.

4. **미분 가능한 위상 특징 계산**: 필트레이션 파라미터 $\theta \mapsto \text{Dg}(G, f_\theta)$의 최적화 가능성을 시사하여, end-to-end 학습에서 위상 특징의 역할을 확대했다.

### 4.2 향후 연구 시 고려할 점

1. **더 복잡한 후단 아키텍처와의 결합**: GNN, Transformer 등과 PersLay를 결합하여 위상 정보와 구조 정보를 동시에 활용하는 하이브리드 모델 탐구가 필요하다.

2. **자동 하이퍼파라미터 탐색**: $\phi$, $\mathbf{op}$, $w$의 그리드 크기 등을 자동으로 선택하는 메타러닝 또는 NAS 기반 접근법이 유용할 것이다.

3. **위상 특징이 유효하지 않은 도메인**: NCI 데이터셋에서의 결과가 보여주듯, 위상 정보가 모든 문제에 판별적인 것은 아니므로, 위상 특징의 유효성을 사전에 평가하는 방법이 필요하다.

4. **End-to-end 미분 가능 파이프라인**: 필트레이션 함수 자체를 학습하는 방향 ([BGND+19, LOT19])과의 결합을 통해 완전한 end-to-end 학습이 가능한 시스템 구축이 중요하다.

5. **이론적 표현력 분석**: PersLay가 근사할 수 있는 함수 클래스에 대한 보편 근사 정리(universal approximation theorem)를 확립하는 것이 남은 과제이다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 PLLay (Kim et al., NeurIPS 2020)

- **논문**: "PLLay: Efficient Topological Layer based on Persistent Landscapes"
- **핵심**: Persistence Landscape를 미분 가능하게 구현하여 역전파를 가능하게 한다.
- **PersLay와의 비교**: PersLay가 일반적 프레임워크를 제공하는 반면, PLLay는 Persistence Landscape에 특화되어 더 효율적인 구현을 제공한다. 그러나 PersLay의 유연성(다양한 $\phi$ 선택)은 PLLay에서 지원되지 않는다.

### 5.2 Topology Layer (Brüel-Gabrielsson et al., AISTATS 2020)

- **논문**: "A Topology Layer for Machine Learning"
- **핵심**: 퍼시스턴스 다이어그램의 계산 자체를 미분 가능하게 만들어 end-to-end 학습을 가능하게 한다.
- **PersLay와의 비교**: PersLay는 이미 계산된 다이어그램을 처리하는 반면, Topology Layer는 입력 데이터로부터 다이어그램 생성까지 미분 가능하게 하여 필트레이션 함수 자체를 학습할 수 있다. 두 접근법은 상호 보완적이다.

### 5.3 TOGL (Horn et al., ICML 2022)

- **논문**: "Topological Graph Neural Networks"
- **핵심**: GNN 메시지 패싱에 위상 정보를 직접 통합하는 레이어를 제안한다.
- **PersLay와의 비교**: PersLay가 GNN과 별도의 채널로 위상 정보를 처리하는 반면, TOGL은 GNN 아키텍처 내부에 위상 정보를 내재화한다. TOGL은 여러 그래프 분류 벤치마크에서 PersLay보다 개선된 결과를 보고한다.

### 5.4 GFL (Hofer et al., ICLR 2020)

- **논문**: "Graph Filtration Learning"
- **핵심**: 학습 가능한 노드 함수를 통해 그래프 필트레이션을 최적화하고, 그로부터 퍼시스턴스 다이어그램을 생성한다.
- **PersLay와의 비교**: PersLay가 HKS와 같은 고정된(또는 제한적으로 학습 가능한) 필트레이션을 사용하는 반면, GFL은 필트레이션 함수 자체를 데이터로부터 학습한다. 이는 PersLay가 Section 4에서 시사한 "파라미터 $t$ 최적화"의 본격적 확장으로 볼 수 있다.

### 5.5 Persformer (Reinauer et al., 2022)

- **논문**: "Persformer: A Transformer Architecture for Topological Machine Learning"
- **핵심**: Transformer의 self-attention 메커니즘을 퍼시스턴스 다이어그램에 적용한다.
- **PersLay와의 비교**: PersLay의 순열 불변 연산($\mathbf{op}$)이 고정된 집계(sum, max 등)인 반면, Persformer는 attention 기반의 적응적 집계를 사용하여 다이어그램 점들 간의 상호작용을 더 풍부하게 모델링한다.

### 5.6 비교 요약표

| 방법 | 연도 | 필트레이션 학습 | 벡터화 유연성 | End-to-end | GNN 통합 |
|------|------|-------------|-----------|-----------|---------|
| **PersLay** | 2020 | 제한적 ($t$ 최적화) | 매우 높음 | 부분적 | 별도 채널 |
| Topology Layer | 2020 | ✓ | 제한적 | ✓ | 별도 |
| PLLay | 2020 | ✗ | Landscape 한정 | 부분적 | 별도 |
| GFL | 2020 | ✓ | 중간 | ✓ | 별도 |
| TOGL | 2022 | ✓ | 중간 | ✓ | **내재화** |
| Persformer | 2022 | ✗ | Attention 기반 | 부분적 | 별도 |

---

## 참고자료

1. Carrière, M., Chazal, F., Ike, Y., Lacombe, T., Royer, M., & Umeda, Y. (2020). "PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological Signatures." *arXiv:1904.09378v4* [stat.ML]. (본 논문 원문)
2. Adams, H. et al. (2017). "Persistence images: a stable vector representation of persistent homology." *Journal of Machine Learning Research*, 18(8).
3. Bubenik, P. (2015). "Statistical topological data analysis using persistence landscapes." *Journal of Machine Learning Research*, 16(77).
4. Zaheer, M. et al. (2017). "Deep Sets." *Advances in Neural Information Processing Systems*.
5. Hofer, C., Kwitt, R., & Niethammer, M. (2019). "Learning representations of persistence barcodes." *Journal of Machine Learning Research*, 20(126).
6. Brüel-Gabrielsson, R. et al. (2020). "A Topology Layer for Machine Learning." *AISTATS 2020*. arXiv:1905.12200.
7. Hofer, C. et al. (2020). "Graph Filtration Learning." *ICML 2020*. arXiv:1905.10996.
8. Horn, M. et al. (2022). "Topological Graph Neural Networks." *ICML 2022*. arXiv:2102.07835.
9. Kim, K. et al. (2020). "PLLay: Efficient Topological Layer based on Persistent Landscapes." *NeurIPS 2020*. arXiv:2002.02778.
10. Reinauer, R. et al. (2022). "Persformer: A Transformer Architecture for Topological Machine Learning." arXiv:2112.15210.
11. Divol, V. & Lacombe, T. (2019). "Understanding the topology and the geometry of the persistence diagram space via optimal partial transport." arXiv:1901.03048.
12. Leygonie, J., Oudot, S., & Tillmann, U. (2019). "A framework for differential calculus on persistence barcodes." arXiv:1910.00960.
13. Hu, N., Rustamov, R., & Guibas, L. (2014). "Stable and informative spectral signatures for graph matching." *CVPR 2014*.
14. Chazal, F., de Silva, V., Glisse, M., & Oudot, S. (2016). *The Structure and Stability of Persistence Modules*. Springer.
15. GitHub Repository: https://github.com/MathieuCarriere/perslay
