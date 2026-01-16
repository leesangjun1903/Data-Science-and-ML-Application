# Feature Selection in the Contrastive Analysis Setting

## 1. 핵심 주장과 주요 기여 (간결 요약)

이 논문의 핵심 주장은 다음 한 줄로 요약할 수 있다.  
**“타깃–백그라운드 데이터가 주어지는 contrastive analysis(CA) 환경에서, ‘흥미로운(salient) 변화’만 잘 잡아내는 feature subset을 선택하는 것이 중요하며, 이를 위해 2단계 학습과 정보이론적 분석에 기반한 Contrastive Feature Selection(CFS)을 제안한다.”** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

주요 기여는 네 가지다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

1. **문제 정의**: 기존 연구들이 다루지 않았던 “contrastive analysis 환경에서의 feature selection 문제”를 명확한 그래픽 모델과 latent 변수(s, z)를 통해 정식화.
2. **CFS 알고리즘 제안**:  
   - (1단계) 백그라운드 데이터로 “무관한 요인 z”를 요약하는 표현 $b$를 학습하고  
   - (2단계) 타깃 데이터에서 $b$로 설명되지 않는 잔여 변동을 가장 잘 설명하는 feature subset $S$를 end-to-end로 최적화하는 2단계 neural feature selection 기법을 제안.
3. **정보이론적 분석**: 두 단계의 학습 목표가 상호정보량 $I(a;s)$ (선택된 표현 $a$와 salient 변수 $s$ 사이 정보)를 **하한에서 극대화한다**는 정리를 증명하여, CFS가 “흥미로운 요인에 집중하는 표현”을 학습함을 이론적으로 뒷받침. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
4. **실험 검증**:  
   - Grassy MNIST(반합성)와 4개 생물의학 데이터셋에서, CFS가  
     - 최신 **완전 비지도 feature selection**(Concrete AE, DUFS)  
     - **완전 지도 feature selection**(STG, LassoNet의 단순 CA 적응)  
     보다 일관되게 높은 downstream 성능을 보임을 입증. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
   - 특히, CA 설정에 특화되지 않은 기존 방법들이 “잡음(z) 요인”에 치우친 feature를 고르는 반면, CFS는 타깃 특이적인(s-specific) 변동을 잘 포착함을 시각적·정량적으로 보여준다.

***

## 2. 해결하려는 문제, 제안 방법, 모델 구조, 성능 및 한계 (상세)

### 2.1 문제 설정: CA 환경의 feature selection

관측 벡터 $x \in \mathbb{R}^d$ 는 두 개의 잠재 요인으로부터 생성된다고 가정한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

- **salient 변수** $s$: 우리가 알고 싶은 “흥미로운” 변동 (예: 질병 상태, 암 아형, 치료 반응 등)
- **background 변수** $z$: 분석 목적과 무관하거나 방해가 되는 요인 (예: 성별, 나이, batch effect, 기술적 잡음 등)

데이터 생성 과정:

$$
p(x) = \int p(x \mid s, z)\,p(s, z)\,ds\,dz
$$

CA 환경에서는 두 개의 비라벨 데이터셋만 주어진다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

- 타깃 데이터셋 $D_t$: $x \sim p(x)$, $s$와 $z$가 **둘 다** 변동
- 백그라운드 데이터셋 $D_b$: $x \sim p(x\mid s = s')$, 즉 $s$는 고정, **$z$만** 변동

목표는 **고정된 feature 개수 $k$에 대해, 가장 정보량이 높은 subset $S \subset [d]$를 고르는 것**이다.

- notation: $S \subset [d] = \{1,\dots,d\}$, $|S|=k$, $x_S = (x_i)_{i\in S}$
- 이상적인 목표:

$$
  x_S \text{가 } s \text{와 동등한 수준의 정보를 담도록, 즉 } I(x_S; s) \text{ 최대화.}
  $$

하지만 $s$에 대한 라벨은 없으므로, 직접 $I(x_S; s)$를 최적화할 수 없다. 대신 **타깃/백그라운드 쌍이 제공하는 “약한 감독(weak supervision)”**을 활용해야 한다는 점이 핵심 난제이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

***

### 2.2 직관: “무관한 요인”을 먼저 설명하고 나머지를 salient로 보는 전략

아이디어는 다음과 같다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

1. 만약 $z$를 완전히 요약한 변수 $b$를 안다면, $x$의 대부분 변동은 $b$로 설명된다.
2. **$b$로 설명되지 않는 나머지 변동**은 $s$에 관련된 salient 요인일 가능성이 크다.
3. 그러므로 $b$와 함께 $x$를 재구성했을 때, 재구성 오차를 가장 많이 줄여주는 feature subset $S$가 **“salient 변동을 포착하는 feature 집합”**일 것이다.

즉, “ $b$ 가 설명하지 못한 잔차를 최대한 잘 설명하는 feature를 골라라”가 CFS의 설계 철학이다.

***

### 2.3 수식: CFS의 최적화 목적

#### (1) 이상적인 contrastive feature selection 목적

$b \in \mathbb{R}^\ell$이 주어졌다고 가정하자. $b$와 $k$개의 선택된 feature $x_S$를 이용해 $x$를 재구성하는 함수 $f: \mathbb{R}^{\ell + k} \to \mathbb{R}^d$를 학습한다고 하면, 이상적인 목적은

$$
\min_{\theta,\;|S| = k}\;
\mathbb{E}_{x \sim D_t}\bigl\|f\bigl(b, x_S;\theta\bigr) - x\bigr\|_2^2
$$

이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

- 이 목적은 **타깃 데이터에서만** 정의되며, “ $b$ 와 $S$가 함께 타깃 데이터의 변동을 얼마나 잘 설명하는지”를 측정한다.
- 최적의 $S$는
  - $b$만으로는 설명되지 않는,  
  - 타깃 데이터에 **특이적으로(enriched)** 나타나는 변동을 포착하는 feature subset이 된다.

#### (2) STG를 이용한 연속 근사

$S$는 이산 집합이므로 직접 최적화가 어렵다. 논문은 **Stochastic Gates(STG)**를 이용해 differentiable하게 근사한다. [semanticscholar](https://www.semanticscholar.org/paper/Contrastive-Principal-Component-Analysis-Abid-Bagaria/b8658b08a14a55470c12413a8c762d46b7f29351)

각 feature $i$에 대해 gate 변수 $G_i \in$ 를 두고, [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
STG 샘플링은

$$
G_i = \max(0, \min(1, \mu_i + \zeta)),\quad \zeta \sim \mathcal{N}(0,\sigma^2)
$$

로 정의된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

- $\mu_i$는 학습 가능한 파라미터,
- $G_i \approx 1$이면 feature $i$는 “켜짐(on)”, $G_i \approx 0$이면 “꺼짐(off)”으로 해석.

이제 $x_S$ 대신 $x \odot G$ (Hadamard product)를 쓰고, 열려 있는 gate 수를 $\ell_0$ 유사 노름으로 정규화하면, 연속 완화된 목적은

$$
\min_{\mu,\theta}\;
\mathbb{E}_{x \sim D_t}\bigl\|f\bigl(b, x \odot G;\theta\bigr) - x\bigr\|_2^2
\;+\;
\lambda \sum_{i=1}^d \Phi\!\left(\frac{\mu_i}{\sigma}\right)
$$

이 된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

- $\Phi$는 표준 정규 CDF, $\lambda$는 sparsity를 조절하는 하이퍼파라미터.
- $\sum_i \Phi(\mu_i/\sigma)$는 **열린 gate(선택된 feature)의 개수**에 대한 연속 근사이다. [semanticscholar](https://www.semanticscholar.org/paper/Contrastive-Principal-Component-Analysis-Abid-Bagaria/b8658b08a14a55470c12413a8c762d46b7f29351)

#### (3) 백그라운드 표현 $b$ 학습

이제 $b$를 어떻게 얻을까? 논문은 **백그라운드 데이터에 autoencoder를 학습**해서 $z$의 변동을 요약하는 representation $b=g(x;\phi)$를 얻는다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

- encoder: $g:\mathbb{R}^d \to \mathbb{R}^\ell$
- decoder: $h:\mathbb{R}^\ell \to \mathbb{R}^d$

목적 함수는

$$
\min_{\phi,\eta}\;
\mathbb{E}_{x \sim D_b}\bigl\|h\bigl(g(x;\phi);\eta\bigr) - x\bigr\|_2^2
$$

이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

- 이 단계는 **백그라운드 데이터만** 사용하므로, $b=g(x)$는 원칙적으로 “ $z$ 에 의한 변동”만을 학습하도록 유도된다.

***

### 2.4 모델 구조 및 2단계 학습 절차

CFS의 전체 구조는 다음과 같은 **두 단계 최적화**로 구성된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

1. **1단계 (background autoencoder 학습)**  
   - 입력: 백그라운드 데이터 $x \sim D_b$  
   - 네트워크: $x \xrightarrow{g} b \xrightarrow{h} \hat{x}$  
   - 목적: 식 (4)를 최소화하여 $b$가 **무관한 변동(z)**을 요약하도록 학습.
   - 학습 후, encoder $g$의 파라미터 $\phi$를 **고정(freeze)**.

2. **2단계 (contrastive feature selection)**  
   - 입력: 타깃 데이터 $x \sim D_t$  
   - 고정된 encoder $g$로부터 $b = g(x)$를 얻음.  
   - STG selector layer를 통해 $x \odot G$를 생성.  
   - $(b, x \odot G)$를 입력으로 하는 재구성 네트워크 $f$로 $x$를 재구성.  
   - 목적: 식 (3)을 최소화하여  
     - 재구성 오차를 줄이면서,  
     - 가능한 적은 수의 gate(=feature)만 열도록 유도.

구조적으로 보면, **“백그라운드 인코더 $g$ + feature gate layer(STG) + 재구성 네트워크 $f$”**의 세 모듈로 이루어진 autoencoder 계열 모델이며, 학습 단계가 분리(pretrain+fine-tune)되었다고 이해할 수 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

***

### 2.5 성능 향상: 실험 결과 개요

논문은 Grassy MNIST(반합성)와 4개 생물의학 데이터셋에서 성능을 평가한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

1. **Grassy MNIST (반합성)** [link.springer](http://link.springer.com/10.1007/978-981-15-5566-4_31)
   - 타깃: “잔디 배경 + 손글씨 숫자” 이미지  
   - 백그라운드: “잔디만” 이미지  
   - 진짜로 중요한 요인: “숫자 모양” (salient $s$)  
   - 실험:
     - $k=20$개 픽셀 선택 시, CFS는 **숫자가 있는 중앙 영역** 중심으로 픽셀을 선택하는 반면,  
       - Concrete AE, DUFS(비지도)와 STG, LassoNet(지도) 등은 주변 잡음 영역이나 부분적인 영역만 커버하는 등, 타깃 특이 변동을 제대로 포착하지 못함. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
     - 선택된 픽셀만으로 랜덤 포레스트를 학습해 숫자(0–9) 분류를 수행하면,  
       - **동일한 $k$에서 CFS가 항상 더 높은 정확도**를 보인다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

2. **4개 실제 생물의학 데이터셋** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
   - Mice Protein, Epithelial Cell infection, AML treatment, ECCITE-seq perturbation [arxiv](https://arxiv.org/abs/2303.08068)
   - 공통 구조:
     - 타깃: 특정 질병/치료/감염 조건의 샘플
     - 백그라운드: 건강한 컨트롤 또는 비처리 샘플
     - 평가:  
       - CFS로 선택한 feature만 사용해서 랜덤 포레스트 / XGBoost / MLP로 downstream 클래스를 예측.
   - 결과:
     - CFS(Pretrained, Stop-Gradient 변형 모두)가  
       - **비지도 FS (CAE, DUFS)**보다 항상 좋거나 비슷하고,  
       - **지도 FS (STG, LassoNet을 타깃 vs 백그라운드 이진 분류로 학습)**보다도 대부분 조건에서 우수. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
     - Joint training(CFS-Joint)은 이론적 분석에서 예측한 대로, salient 정보가 $b$로 “새어(leak)”가서 성능이 떨어지는 경우가 많다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

3. **배경 차원 $l$과 백그라운드 샘플 수에 대한 민감도**  
   - Grassy MNIST에서 $l$을 다양하게 바꾸어도 CFS 성능은 크게 변하지 않아 **robust**함을 보임. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
   - Mice Protein에서 백그라운드 샘플 수를 늘리면 CFS 성능이 초기에는 향상되다가 어느 시점에서 포화되며,  
     - 백그라운드 인코더를 **random init로 고정**하면 성능이 크게 떨어져, “**좋은 $b$ 학습이 필수**”임을 보여준다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

***

### 2.6 한계와 실무적 제약

1. **CA 설정 자체의 제약**  
   - 타깃–백그라운드 두 데이터셋이 필요하며,  
   - 백그라운드가 “ $s$ 는 고정, $z$만 변동”이라는 가정을 어느 정도 만족해야 한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
   - 실제 응용에서 이 가정이 깨질 경우, $b$에 salient 정보가 섞이고, CFS가 원래 의도대로 동작하지 않을 수 있다.

2. **$s$와 $z$의 독립성 가정**  
   - 정보이론 분석(다음 섹션)의 정리는 $I(s;z)\approx 0$ 등의 “약한 독립성” 가정 하에서 성립한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
   - 현실에서는 인구학적 변수와 질병 상태가 상관 있는 경우가 많아, 이 가정이 깨지면 이론적 보장은 약해진다.

3. **하이퍼파라미터 선택**  
   - STG의 $\lambda$는 사실상 “k개 feature를 남기도록” tuning해야 하며, [semanticscholar](https://www.semanticscholar.org/paper/Contrastive-Principal-Component-Analysis-Abid-Bagaria/b8658b08a14a55470c12413a8c762d46b7f29351)
   - $k$와 $\lambda$를 어떻게 선택할지에 대한 자동화된 기준은 제공되지 않는다.

4. **계산 비용과 복잡성**  
   - 두 단계 학습(백그라운드 AE + 타깃 CFS)이 필요하고,  
   - 각 단계에 neural network를 사용하므로 데이터가 매우 큰 경우 비용이 만만치 않다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

***

## 3. 정보이론 분석과 “일반화 성능 향상 가능성”

### 3.1 두 단계 학습과 상호정보량 하한

논문은 두 representation $a$와 $b$를 고려한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

- $b$: 1단계에서 학습한 백그라운드 표현 (z 관련)
- $a$: 2단계에서 학습되는 “salient 표현” (실제로는 $x_S$ 또는 그 함수)

목표는 ** $a$ 가 $s$ 와 최대한 동등한 정보를 갖도록**, 즉 $I(a;s)$ 를 크게 만드는 것이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

이를 위해 다음과 같은 가정을 둔다.

- (가정 1) $s$와 $z$는 거의 독립이고, $x$는 $(s,z)$로 잘 설명되며, 각 latent는 $x$ (다른 latent)  에서 잘 복원 가능:

$$
  I(s;z)\le \epsilon,\quad
  H(x\mid s,z)\le \epsilon,\quad
  H(s\mid x,z)\le \epsilon,\quad
  H(z\mid x,s)\le \epsilon
$$

- (가정 2) 학습된 표현 $a,b$를 conditioning해도 $s,z$의 “거의 독립성”이 유지된다:

$$
  I(s;z\mid b)\le \epsilon,\quad
  I(s;z\mid a)\le \epsilon
  $$

이때, 논문의 핵심 정리는 다음과 같다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

> **정리 1.** 위 가정 하에서,  
> 1단계에서 $I(b;x\mid s)$를 최대화하고,  
> 2단계에서 $I(a;x\mid b)$를 최대화하면  
> 
> $$
> I(a;x\mid b) + I(b;x\mid s) - H(z) - 4\epsilon \;\le\; I(a;s).
> $$
> 
> 즉, 두 단계의 정보량 목적 합이 **진짜 목표 $I(a;s)$의 하한**이 된다.

해석:

- 1단계에서 $b$는 “ $s$가 고정된 상태에서 $x$에 대한 정보”를 최대화하므로, $z$에 대한 정보를 최대한 많이 담게 된다.
- 2단계에서 $a$는 “ $b$를 알고 있을 때 $x$에 대한 추가 정보”를 최대화하므로, $b$가 놓친 나머지 요인(= $s$)을 회수하게 된다.
- 따라서 $a$와 $s$의 상호정보량 $I(a;s)$는 **적어도** 두 단계 목표 합에서 $H(z)+4\epsilon$을 뺀 값 이상이 된다.

이는 CFS가 **구체적으로 $I(a;s)$를 직접 최적화하진 않지만**, 실제로는 그 하한을 최대화하는 구조를 가지고 있음을 의미하고, “salient 요인에 대한 표현력”이 좋음을 이론적으로 뒷받침한다.

***

### 3.2 다른 학습전략과의 대비: joint training, 완전 비지도 학습

#### (1) Joint training (contrastive VAE류)와의 비교

기존 contrastive VAE류는 대체로 $a$와 $b$를 **동시에(joint)** 학습하면서 두 데이터셋(타깃+백그라운드)을 한꺼번에 모델링한다. [semanticscholar](https://www.semanticscholar.org/paper/0ea94e5fd3071496a42c153bb5a856eb509acbe2)

이 경우 타깃 데이터에 대한 목적은 $I(a,b;x)$를 최대화하는 꼴이 된다. 논문은 이 때의 $I(a;s)$ 하한이

$$
I(a,b;x) + I(b;x\mid s) - I(b;x) - H(z) - 4\epsilon \;\le\; I(a;s)
$$

으로 변형된다는 것을 보인다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

- joint 학습에서는 $I(a,b;x)=I(b;x)+I(a;x\mid b)$를 **둘 다** 키우려 하므로, $I(b;x)$가 너무 커져 버릴 수 있다.
- 그러면 salient 정보를 설명해야 하는 $a$가 할 일이 줄어들고, 실제로 **salient 정보가 $b$ 안으로 “새어(leak)” 들어가는 현상**이 발생한다. [biorxiv](https://www.biorxiv.org/content/10.1101/2021.12.21.473757v3.full-text)
- 실험에서도 CFS-Joint 변형이 “backbone은 비슷하지만 두 단계가 아니라 joint로 학습”하는 경우, 성능이 떨어지는 것을 보여준다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

요약하면, **두 단계 분리 학습이 joint 학습보다 $I(a;s)$를 더 잘 키우는 방향으로 작동할 가능성이 크다**는 이론적·실증적 근거를 제시한다.

#### (2) 완전 비지도 feature selection과의 비교

Concrete AE, DUFS 등 비지도 FS 방법은 라벨도, 타깃–백그라운드 쌍도 사용하지 않고, 단순히 $I(a;x)$ (또는 reconstruction 정확도)를 최대화하는 방향으로 표현 $a$를 학습한다. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9178272/)

논문은 이 경우 다음과 같은 결과를 보인다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

> **정리 2.** $I(a;x)$를 최대화하는 단일 표현 $a$를 학습한다면

> $$
> I(a;x) - H(x) + I(x;s) \;\le\; I(a;s) \\
> I(a;x) - H(x) + I(x;z) \;\le\; I(a;z)
> $$
> 
> 이 동시에 성립한다.

- 만약 데이터에서 $z$가 $s$보다 훨씬 큰 분산과 변동을 차지한다면 (실제 omics 데이터에서 batch effect, 기술 잡음 등이 그렇다): [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12456736/)
  - $I(x;z) > I(x;s)$ 이고,
  - 따라서 **$I(a;z)$에 대한 하한이 $I(a;s)$에 대한 하한보다 크다.**
- 즉, 비지도 FS는 **잡음/무관 요인(z)에 더 잘 적응하는 표현**을 학습하는 경향을 갖게 되고,  
  - 이것이 Grassy MNIST에서 주변 grass 픽셀을 고르는 현상으로 나타난다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

이 결과는 CFS가 CA 특화 정보(타깃–백그라운드 구조)를 활용해서 **“잘못된 요인(z)”이 아닌 “관심 요인(s)”으로 일반화**하도록 유도한다는 점에서, **일반화 성능 측면의 장점**을 이론적으로 뒷받침한다.

***

### 3.3 일반화 성능 관점에서의 해석

실질적인 일반화 성능(새로운 샘플, 새로운 downstream 태스크로의 전이)을 고려하면, CFS의 구조는 다음과 같은 이점을 가진다.

1. **Nuisance 요인 제거를 통한 분산 감소**  
   - $z$에 의한 variation을 $b$가 흡수하고, feature subset $S$는 $s$에 특화되도록 유도되므로,  
   - $S$ 위에서 학습되는 downstream 모델은 **덜 복잡한 decision boundary**를 학습하게 된다 → 분산 감소, 일반화 향상.

2. **정보량 관점에서의 충분성**  
   - 정리 1에 따르면 $I(a;s)$가 큰 값을 갖게 되어, $a$ (또는 $x_S$)는 $s$의 **충분한 통계(sufficient statistic)에 가까운 역할**을 한다. [arxiv](https://arxiv.org/abs/2108.09159)
   - 이는 다양한 supervised 태스크(질병 subtype 분류, 치료 반응 예측 등)에서 **라벨이 제한적이어도 안정적으로 학습**할 수 있는 잠재력을 시사한다.

3. **여러 downstream 모델에 대한 robust한 성능**  
   - 동일한 feature subset으로 Random Forest, XGBoost, MLP 등 서로 다른 모델에 대해 consistently 좋은 성능을 보였다는 실험 결과는,  
   - CFS가 특정 모델 구조에 overfit되지 않고 **표현 자체의 일반적 품질**을 높인다는 증거로 해석할 수 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

요약하면, CFS는 **“CA 구조를 활용한 representation learning + feature selection”**을 통해,  
- 무관한 요인(z)을 제거하고,  
- 관심 요인(s)에 대한 정보량을 극대화함으로써,  
**다양한 downstream 태스크에서의 일반화 성능 향상 가능성**을 이론과 실험으로 함께 보여준다.

***

## 4. 2020년 이후 관련 최신 연구와의 비교 분석

CFS는 “contrastive analysis + feature selection”이라는 상당히 특화된 조합을 다루지만, 인접 영역에서 2020년 이후 활발한 연구들이 진행되었다. 여기서는 (1) contrastive latent variable models, (2) dynamic feature selection & MI, (3) contrastive/disentangled representation 학습 관점에서 비교한다.

### 4.1 Contrastive latent variable models (CA 표현 학습)

1. **Contrastive latent variable modeling for sequencing (CPLVM/CGLVM)** [projecteuclid](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-16/issue-3/Contrastive-latent-variable-modeling-with-application-to-case-control-sequencing/10.1214/21-AOAS1534.full)
   - Annals of Applied Statistics(2021)에서 제안된 contrastive Poisson latent variable model (CPLVM)은  
     - 카운트 데이터(특히 RNA-seq)를 위한 **확률론적 contrastive 모형**이다. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12456736/)
     - 타깃/백그라운드 조건 간의 shared vs target-specific latent factor를 분리해 case–control 간 transcriptional 변화 구조를 요약한다.
   - 그러나 **feature selection이 아닌, 저차원 latent representation**을 산출하며,  
     - 중요 gene set은 주로 latent loading 해석을 통해 간접적으로 얻는다.

2. **ContrastiveVI (contrastive variational inference)** [rna-seqblog](https://www.rna-seqblog.com/contrastivevi-isolating-salient-variations-of-interest-in-single-cell-data/)
   - Weinberger et al., Nature Methods 2023. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/37550579/)
   - single-cell RNA-seq에서 타깃(처리) vs 백그라운드(컨트롤)를 위한 deep generative model.  
   - 구조적으로,
     - **shared latent space**: 타깃/백그라운드 공통 변동,
     - **salient latent space**: 타깃 특이 변동을 캡처하도록 encoder를 분리해 학습. [rna-seqblog](https://www.rna-seqblog.com/contrastivevi-isolating-salient-variations-of-interest-in-single-cell-data/)
   - downstream으로 visualization, clustering, DE testing 등 CA 전용 분석을 지원하며, scvi-tools에 구현되어 있다. [scvi-tools.readthedocs](https://scvi-tools.readthedocs.io/en/stable/user_guide/models/contrastivevi.html)
   - CFS와 공통점:
     - 둘 다 CA 설정에서 **타깃 특이 variation을 분리/강조**하려 한다. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/37550579/)
     - 둘 다 latent 변수 분리 및 leakage 방지에 주의를 기울인다 (contrastiveVI도 leakage 방지 정규화 도입). [biorxiv](https://www.biorxiv.org/content/10.1101/2024.01.05.574421v1.full-text)
   - 차이점:
     - ContrastiveVI는 **연속 latent 표현에 초점**, CFS는 **명시적 feature subset**을 산출.
     - ContrastiveVI는 joint training 기반 VAE 구조로, regularization을 통해 leakage를 제어하는 반면, [biorxiv](https://www.biorxiv.org/content/10.1101/2024.01.05.574421v1.full-text)
       CFS는 **2단계 학습 구조와 정보이론 분석**으로 leakage 문제를 정면으로 다룬다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

3. **Moment-Matching contrastive VAE (MM-cVAE)** [semanticscholar](https://www.semanticscholar.org/paper/Contrastive-latent-variable-modeling-with-to-Jones-Townes/193a1781329c5b936e75e1c13ed15c81bee434dc)
   - Severson et al.의 contrastive VAE류의 한계를 지적하고, maximum mean discrepancy(MMD)를 사용해  
     - “shared latent”와 “salient latent” 간 분포를 명시적으로 분리하는 모델. [semanticscholar](https://www.semanticscholar.org/paper/0ea94e5fd3071496a42c153bb5a856eb509acbe2)
   - CFS 논문에서 인용하듯, joint 학습 구조에서 latent 간 정보 누출이 발생할 수 있음을 강조한다. [biorxiv](https://www.biorxiv.org/content/10.1101/2021.12.21.473757v3.full-text)

4. **후속 contrastive/다중 그룹 VAE 계열**  
   - multiGroupVI, spVIPES(spatial/private VAE for multi-group data), Multi-ContrastiveVAE(single-cell 이미지 perturbation) 등은 [biorxiv](https://www.biorxiv.org/content/10.1101/2023.11.07.565957v1.full.pdf)
     - 여러 그룹의 데이터에서 shared vs private variation을 disentangle하려는 시도이다. [biorxiv](https://www.biorxiv.org/content/10.1101/2023.11.28.569094v1)
   - 이들 역시 **representation 수준 disentangling**이 중심이며,  
     - CFS처럼 **“극소수 feature subset”을 직접 선택**하지는 않는다. [biorxiv](https://www.biorxiv.org/content/10.1101/2023.11.07.565957v1.full.pdf)

**정리:**  
contrastiveVI, CPLVM 등은 “CA 환경에서 saliency를 분리하는 representation 학습”의 강력한 도구이지만,  
- feature 측면에서는 **“어떤 gene/feature가 salient variation을 담당하는지”를 직접적으로, 소수 subset 단위로 제공하지 못한다.**  
CFS는 이 틈을 메우는, **CA 특화 neural feature selection**이라는 점에서 독자적인 위치를 가진다. [semanticscholar](https://www.semanticscholar.org/paper/Contrastive-latent-variable-modeling-with-to-Jones-Townes/193a1781329c5b936e75e1c13ed15c81bee434dc)

***

### 4.2 Dynamic feature selection & mutual information (MI 기반 FS)

CFS의 정보이론 분석은 최근의 **MI 기반 feature selection** 흐름과도 긴밀히 연결된다.

1. **Learning to Maximize Mutual Information for Dynamic Feature Selection (Covert et al., ICML 2023)** [proceedings.mlr](https://proceedings.mlr.press/v202/covert23a/covert23a.pdf)
   - Dynamic Feature Selection(DFS) 문제에서 **조건부 상호정보량(conditional MI)**을 greedy하게 최대화하는 정책을 제안. [arxiv](https://arxiv.org/abs/2301.00557)
   - 핵심 아이디어:
     - 현재까지 관측된 feature subset $x_S$에 대해,  
       다음 feature $i$는

$$
       i^\* = \arg\max_i I(x_i; y \mid x_S)
       $$

를 만족하도록 선택한다. [proceedings.mlr](https://proceedings.mlr.press/v202/covert23a/covert23a.pdf)
   - 실제 데이터 분포 $p(x,y)$를 알 수 없으므로, amortized optimization을 통해 **MI를 근사 예측**하는 모델을 학습. [proceedings.iclr](https://proceedings.iclr.cc/paper_files/paper/2024/file/9682490bedc064aba8aac1ab3f703507-Paper-Conference.pdf)
   - 결과적으로, 여러 DFS/정적 FS 방법보다 우수한 성능을 보인다. [arxiv](https://arxiv.org/pdf/2301.00557.pdf)

2. **Conditional MI estimation & DFS 후속 연구** [arxiv](https://arxiv.org/pdf/2508.02566.pdf)
   - ICLR 2024 등에서 DFS를 위한 CMI 추정 개선, rule-based DFS 등 다양한 확장이 제안되었다. [arxiv](https://arxiv.org/abs/2306.03301)

**CFS와의 관계:**

- 공통점:
  - 둘 다 **상호정보량을 명시적인 설계 기준**으로 삼는다. [proceedings.mlr](https://proceedings.mlr.press/v202/covert23a/covert23a.pdf)
  - Covert et al.은 $I(x_i; y\mid x_S)$를 직접 최대화하려 하고, [proceedings.mlr](https://proceedings.mlr.press/v202/covert23a/covert23a.pdf)
    CFS는 $I(a;x\mid b)$, $I(b;x\mid s)$를 통해 간접적으로 $I(a;s)$를 최대화한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
- 차이점:
  - DFS 계열은 **완전 지도(supervised)** 환경에서 $y$ 라벨에 직접 최적화하는 반면, [arxiv](https://arxiv.org/pdf/2301.00557.pdf)
    CFS는 **라벨 없는 contrastive 데이터(target vs background)**만 활용해, latent 요인 $s$에 관한 정보를 극대화한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
  - DFS는 **샘플별로 동적인 feature 시퀀스**를 선택하는 것이 목적이고, [arxiv](https://arxiv.org/pdf/2508.02566.pdf)
    CFS는 **고정된 feature subset $S$**를 CA 해석과 downstream 분석의 공유 자원으로 제공한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

요약하면, CFS는 “CA 환경에서의 비지도(weakly supervised) MI 기반 feature selection”으로,  
- “지도 DFS + MI” 연구들과 개념적으로 연결되나,
- 데이터 구조(contrastive)와 목적(일반적 CA 표현 및 해석)을 달리한다.

***

### 4.3 기타 contrastive/disentangled latent 모델 (텍스트, 멀티모달 등)

- Contrastive latent variable models for neural text generation, ContrastVAE 등은 [proceedings.mlr](https://proceedings.mlr.press/v180/teng22a/teng22a.pdf)
  - VAE latent space에 contrastive learning을 추가해 텍스트 생성 품질과 latent 표현의 의미론적 구조를 향상시키지만,
  - CA 설정(타깃/백그라운드 쌍)이 아니라, 인스턴스 간 pairwise contrast를 사용한다. [yangliangwei.github](https://yangliangwei.github.io/publication/cikm2022/CIKM2022.pdf)

- 단일세포 perturbation 분석에서 SC-VAE(supervised contrastive VAE), CINEMA-OT, istructured-contrastiveVI 등은 [nature](https://www.nature.com/articles/s41592-023-02040-5)
  - contrastiveVI를 확장하거나, 최적 수송/독립성 기준(HSIC) 등을 사용해 perturbation 효과 disentangling을 강화한다. [nature](https://www.nature.com/articles/s41592-023-02040-5)
  - 마찬가지로 **feature selection이 아닌 표현 학습**이 중심이다.

***

### 4.4 요약 비교 표

| 방법 | 설정 | 산출물 | 학습 전략 | 특징 |
|------|------|--------|----------|------|
| **CFS** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf) | CA (타깃/백그라운드), 비라벨 | 고정된 feature subset $S$ | 2단계 (백그라운드 AE → 타깃 CFS), MI 하한 분석 | CA에 특화된 neural feature selection, leakage 방지 |
| **ContrastiveVI** [rna-seqblog](https://www.rna-seqblog.com/contrastivevi-isolating-salient-variations-of-interest-in-single-cell-data/) | CA, 비라벨(single-cell) | shared & salient latent space | joint VAE + 정규화 | 표현 학습, 다양한 downstream 분석 지원 |
| **CPLVM/CGLVM** [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12456736/) | CA, count data | latent factor & loading | Bayesian GLM 계열 | case–control RNA-seq 차이를 latent로 요약 |
| **MM-cVAE 등 contrastive VAE류** [semanticscholar](https://www.semanticscholar.org/paper/0ea94e5fd3071496a42c153bb5a856eb509acbe2) | CA | latent 표현 | joint VAE + MMD/정규화 | leakage 완화, 표현 품질 개선 |
| **MI-DFS(Covert et al.)** [proceedings.mlr](https://proceedings.mlr.press/v202/covert23a/covert23a.pdf) | 지도(라벨 존재) | 샘플별 동적 feature 시퀀스 | CMI 기반 greedy policy 학습 | 예측 정확도 vs feature 비용 trade-off 최적화 |

***

## 5. 앞으로의 연구에 미치는 영향과 향후 고려할 점

### 5.1 연구적·실무적 영향

1. **CA를 위한 “표현 학습 + feature selection”의 통합 프레임워크 제시**  
   - 이전 CA 연구는 대부분 latent representation에 초점을 두었고, [direct.mit](https://direct.mit.edu/neco/article/32/10/1901-1935/95614)
     feature selection은 supervised/unsupervised 환경에서만 논의되었다. [arxiv](https://arxiv.org/html/2601.07666v1)
   - CFS는 **CA 구조를 전제로 한 feature selection 문제를 독립된 연구 대상으로 정립**하고,  
     - 정보이론 분석,  
     - 뉴럴 네트워크 기반 구현,  
     - 실제 생물학 도메인 적용까지 연결함으로써,  
     이 영역에 대한 후속 연구의 출발점 역할을 한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)

2. **“leakage 문제”에 대한 일반적 통찰 제공**  
   - CA뿐 아니라, shared/private latent를 분리하는 모든 representation learning에서  
     - joint 학습 시 shared latent가 salient 정보를 흡수하는 문제가 발생할 수 있음을 이론과 실험으로 명확히 보여주었다. [semanticscholar](https://www.semanticscholar.org/paper/0ea94e5fd3071496a42c153bb5a856eb509acbe2)
   - 이는 contrastiveVI, spVIPES, SC-VAE 같은 후속 모델들도 정규화·훈련 전략 설계 시 반드시 고려해야 할 포인트이다. [biorxiv](https://www.biorxiv.org/content/10.1101/2023.11.28.569094v1)

3. **실제 생물의학 데이터에서의 해석 가능하고 비용 효율적인 feature 설계**  
   - 유전자 발현 데이터에서 “소수의 marker gene set”을 뽑아 이후 여러 실험/진단에 재사용한다는 목적은 매우 실용적이다. [biorxiv](https://www.biorxiv.org/content/10.1101/2025.11.19.689125v1.full-text)
   - CFS는 **타깃 특이 변동을 잘 포착하는 gene subset**을 제공하여,  
     - 실험 비용 절감,  
     - 해석 가능성 향상,  
     - data-efficient supervised 학습을 동시에 지원할 수 있다.

***

### 5.2 향후 연구 방향 및 고려할 점

1. **multi-target / multi-background CA로의 확장**  
   - 현재 CFS는 “하나의 타깃 vs 하나의 백그라운드” 구조를 전제로 한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
   - 실제 실험에서는 여러 treatment, 여러 control group이 존재할 수 있으므로,
     - **다중 그룹 간 contrastive feature selection** (예: multiGroupVI/spVIPES 스타일 + CFS)을 탐색할 수 있다. [biorxiv](https://www.biorxiv.org/content/10.1101/2023.11.07.565957v1.full.pdf)

2. **부분적 감독 정보의 통합 (semi-supervised CA FS)**  
   - 어떤 경우에는 $s$와 관련된 일부 라벨(예: 일부 subtype, 일부 time point)이 주어진다.
   - contrastiveVI + supervised contrastive(SC-VAE)와 유사하게, [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/37550579/)
     - **라벨 정보를 CFS의 두 번째 단계에 통합**해 $I(x_S; s)$를 더 직접적으로 키우는 hybrid FS가 가능하다.

3. **동적 feature selection과의 결합**  
   - MI-DFS 계열 연구와 결합하면, [proceedings.iclr](https://proceedings.iclr.cc/paper_files/paper/2024/file/9682490bedc064aba8aac1ab3f703507-Paper-Conference.pdf)
     - 먼저 CFS로 “CA에 좋은 global subset”을 고르고,  
     - 그 위에서 DFS를 수행해 “샘플별 추가적인 feature acquisition”을 설계하는 2계층 전략을 상상할 수 있다.
   - 고비용 임상 검사용 feature 설계 등에서 유용할 수 있다.

4. **공정성(fairness)/인과성(causality) 관점의 해석**  
   - $z$는 종종 성별, 인종, 사회경제적 요인을 포함하며, 이는 **공정성 문제와 직접 연결**된다.
   - CFS는 본질적으로 “ $z$에 의한 변동을 제거하고 $s$에 집중”하는 구조이므로,  
     - 인과적 representation 학습이나 [arxiv](https://arxiv.org/pdf/2406.13966.pdf)
     - fairness-aware representation 학습과 결합하여,  
       **인과적으로 타당한 feature subset**을 찾는 방향으로 확장될 수 있다.

5. **대규모 사전학습 표현과의 결합**  
   - 최근 single-cell, 이미지, 텍스트 등에서 foundation model/large model 기반 표현이 보편화되고 있다. [arxiv](https://arxiv.org/pdf/2402.06223.pdf)
   - 이들 사전학습 encoder를 고정하고,
     - CFS를 **feature-level이 아닌, latent feature-level에서 작동**하도록 일반화하는 것도 가능하다.
   - 이 경우,  
     - “대규모 사전학습 표현에서 CA에 특화된 소수 차원을 선택하는 CFS”라는 형태로,  
       대규모 모델과 도메인 특화 분석 사이를 이어주는 역할을 할 수 있다.

6. **이론적 가정 완화 및 일반화 성능에 대한 formal bound**  
   - 현재 정보이론 분석은 $s,z$의 거의 독립성, $x$가 $(s,z)$로 충분히 잘 설명된다는 가정에 의존한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
   - 실제에서는 이 가정이 깨지는 경우가 많으므로,  
     - **부분 의존성 하에서의 CFS 성질**,  
     - 선택된 feature subset이 downstream risk(일반화 오차)에 미치는 영향에 대한 **more direct generalization bound** 연구가 필요하다.

***

## 6. 정리

- 이 논문은 contrastive analysis 환경에서 **salient variation을 잘 반영하는 feature subset**을 선택하는 문제를 명시적으로 제기하고, 이를 위한 CFS 알고리즘과 정보이론 분석, 실증 평가를 제시한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
- CFS는
  - 백그라운드 데이터로 nuisance variation을 요약하는 표현 $b$를 먼저 학습하고,
  - 그 위에서 잔여 변동을 가장 잘 설명하는 feature subset을 neural feature selection(STG)으로 찾는 **2단계 구조**를 취한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
- 정보이론적으로, 이 두 단계의 mutual information 목적 합이 **salient 변수 $s$와 표현 $a$ 사이 상호정보량 $I(a;s)$의 하한을 최대화**함을 보여,  
  - CFS가 **흥미로운 요인에 대한 일반화 성능**을 높이는 메커니즘을 이론적으로 설명한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
- 실험적으로, Grassy MNIST와 네 개의 실제 생물의학 데이터셋에서  
  - 최신 비지도/지도 feature selection 방법보다 **일관되게 더 나은 downstream 성능**과  
  - 타깃 특이 변동을 더 잘 포착하는 feature 선택 패턴을 보였다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/902ca97e-88e4-4eba-a606-9c2130dcd82b/2310.18531v1.pdf)
- 2020년 이후의 contrastive latent variable 모델들(contrastiveVI, CPLVM, MM-cVAE 등)과 MI 기반 dynamic feature selection 연구(Covert et al.)는  
  - CFS와 문제의식(“중요한 변동에 집중하라”)을 공유하지만,  
  - 대부분 latent representation에 초점을 두거나 supervised 설정에 머무른다. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12456736/)
- 따라서 CFS는 **CA 환경에서의 비지도/약지도 feature selection**이라는 독자적인 위치를 갖고 있으며,  
  - 향후 multi-group CA, semi-supervised FS, DFS와의 결합, 인과/공정성 통합 등 다양한 방향으로 확장될 수 있다.
 
<span style="display:none">[^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66]</span>

<div align="center">⁂</div>

[^1_1]: 2310.18531v1.pdf

[^1_2]: https://www.semanticscholar.org/paper/Contrastive-Principal-Component-Analysis-Abid-Bagaria/b8658b08a14a55470c12413a8c762d46b7f29351

[^1_3]: http://link.springer.com/10.1007/978-981-15-5566-4_31

[^1_4]: https://arxiv.org/abs/2303.08068

[^1_5]: https://www.biorxiv.org/content/10.1101/2025.11.19.689125v1.full-text

[^1_6]: https://arxiv.org/html/2411.08072v1

[^1_7]: http://arxiv.org/pdf/2211.07723.pdf

[^1_8]: https://www.semanticscholar.org/paper/0ea94e5fd3071496a42c153bb5a856eb509acbe2

[^1_9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12456736/

[^1_10]: https://direct.mit.edu/neco/article/32/10/1901-1935/95614

[^1_11]: https://arxiv.org/pdf/2111.03040.pdf

[^1_12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7286535/

[^1_13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5976774/

[^1_14]: https://www.biorxiv.org/content/10.1101/2021.12.21.473757v3.full-text

[^1_15]: https://ieeexplore.ieee.org/document/9178272/

[^1_16]: https://arxiv.org/html/2601.07666v1

[^1_17]: https://arxiv.org/abs/1808.10868v1

[^1_18]: https://arxiv.org/abs/2108.09159

[^1_19]: https://projecteuclid.org/journals/annals-of-applied-statistics/volume-16/issue-3/Contrastive-latent-variable-modeling-with-application-to-case-control-sequencing/10.1214/21-AOAS1534.full

[^1_20]: https://www.semanticscholar.org/paper/Contrastive-latent-variable-modeling-with-to-Jones-Townes/193a1781329c5b936e75e1c13ed15c81bee434dc

[^1_21]: https://www.rna-seqblog.com/contrastivevi-isolating-salient-variations-of-interest-in-single-cell-data/

[^1_22]: https://pubmed.ncbi.nlm.nih.gov/37550579/

[^1_23]: https://scvi-tools.readthedocs.io/en/stable/user_guide/models/contrastivevi.html

[^1_24]: https://news.cs.washington.edu/2023/08/09/distinctions-with-a-difference-allen-school-researchers-unveil-contrastivevi-a-deep-generative-model-for-gleaning-additional-insights-from-single-cell-datasets/

[^1_25]: https://openreview.net/forum?id=cLVtw2uAEe5

[^1_26]: https://www.biorxiv.org/content/10.1101/2024.01.05.574421v1.full-text

[^1_27]: https://www.biorxiv.org/content/10.1101/2023.11.07.565957v1.full.pdf

[^1_28]: https://www.biorxiv.org/content/10.1101/2023.11.28.569094v1

[^1_29]: https://proceedings.mlr.press/v202/covert23a/covert23a.pdf

[^1_30]: https://proceedings.iclr.cc/paper_files/paper/2024/file/9682490bedc064aba8aac1ab3f703507-Paper-Conference.pdf

[^1_31]: https://arxiv.org/abs/2301.00557

[^1_32]: https://www.semanticscholar.org/paper/74635efeaf024733c8a031de6b8427b128a11a4a

[^1_33]: https://arxiv.org/pdf/2301.00557.pdf

[^1_34]: https://arxiv.org/pdf/2508.02566.pdf

[^1_35]: https://arxiv.org/abs/2306.03301

[^1_36]: https://proceedings.mlr.press/v180/teng22a/teng22a.pdf

[^1_37]: https://yangliangwei.github.io/publication/cikm2022/CIKM2022.pdf

[^1_38]: https://github.com/zeeeyang/contrastive_vae

[^1_39]: https://www.nature.com/articles/s41592-023-02040-5

[^1_40]: https://www.biorxiv.org/content/10.1101/2023.10.06.561320v1.full.pdf

[^1_41]: https://liner.com/review/contrastvae-contrastive-variational-autoencoder-for-sequential-recommendation

[^1_42]: https://www.semanticscholar.org/paper/258dfbbba3aebdb3b8e78f0921a273bdde53c576

[^1_43]: https://arxiv.org/pdf/2406.13966.pdf

[^1_44]: https://arxiv.org/pdf/1705.08821.pdf

[^1_45]: https://arxiv.org/pdf/2402.06223.pdf

[^1_46]: https://arxiv.org/html/2510.14190v1

[^1_47]: https://arxiv.org/html/2510.11847v1

[^1_48]: https://www.semanticscholar.org/paper/32de305601607f4f52b333bdec76f63bca2c8d26

[^1_49]: https://osf.io/u2zdv_v1

[^1_50]: https://projecteuclid.org/journals/annals-of-statistics/volume-50/issue-6/Half-trek-criterion-for-identifiability-of-latent-variable-models/10.1214/22-AOS2221.full

[^1_51]: https://journals.sagepub.com/doi/10.1177/1094428119872531

[^1_52]: https://www.annualreviews.org/doi/10.1146/annurev-statistics-040220-091910

[^1_53]: https://osf.io/qm7kj_v1

[^1_54]: https://arxiv.org/abs/2211.02218

[^1_55]: https://ieeexplore.ieee.org/document/9746739/

[^1_56]: https://arxiv.org/html/2408.07908v2

[^1_57]: https://arxiv.org/pdf/2311.04056.pdf

[^1_58]: http://arxiv.org/pdf/1706.08137.pdf

[^1_59]: https://arxiv.org/pdf/2306.12841.pdf

[^1_60]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7997863/

[^1_61]: https://www.broadinstitute.org/talks/talk-tbd-6

[^1_62]: https://arxiv.org/html/2306.03301

[^1_63]: https://github.com/iancovert/dynamic-selection

[^1_64]: https://docs.scvi-tools.org/en/1.3.3/user_guide/models/contrastivevi.html

[^1_65]: https://arxiv.org/pdf/2212.07183.pdf

[^1_66]: https://openreview.net/pdf?id=Oju2Qu9jvn
