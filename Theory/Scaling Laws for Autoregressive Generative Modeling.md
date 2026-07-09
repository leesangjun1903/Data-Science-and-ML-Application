# Scaling Laws for Autoregressive Generative Modeling

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Henighan et al., 2020, arXiv:2010.14701)은 **자기회귀적(autoregressive) Transformer 모델의 Cross-Entropy Loss가 모델 크기($N$), 계산량($C$), 데이터셋 크기($D$)에 따라 예측 가능한 거듭제곱 법칙(Power Law)을 따른다**는 것을 4개의 도메인(이미지 생성, 비디오 모델링, 멀티모달 이미지↔텍스트, 수학 문제 풀이)에 걸쳐 실증적으로 증명합니다.

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **범용성 입증** | 언어 모델에 국한되었던 스케일링 법칙을 이미지·비디오·멀티모달·수학 도메인으로 확장 |
| **정보이론적 해석** | Loss를 엔트로피($S$)와 KL 발산($D_{KL}$)으로 분해하는 프레임워크 제시 |
| **최적 모델 크기 예측** | 주어진 계산 예산에서 최적 모델 크기를 거듭제곱 법칙으로 예측 |
| **다운스트림 성능 연결** | 생성 모델 Loss 개선이 분류 등 하위 태스크 성능 향상으로 이어짐을 증명 |
| **멀티모달 정보이론** | 이미지-텍스트 간 상호정보량(Mutual Information) 스케일링 법칙 식별 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

기존 스케일링 법칙 연구(Kaplan et al., 2020, "Scaling Laws for Neural Language Models")는 **언어 모델에 한정**되어 있었습니다. 이 논문은 다음 질문들에 답하고자 합니다:

- 스케일링 법칙이 **모든 데이터 모달리티**에 적용되는가?
- **Loss 개선이 다운스트림 태스크 성능**으로 전환되는가?
- 모델 성능이 포화되는 시점을 **사전에 예측**할 수 있는가?
- **환원 불가능한 손실(Irreducible Loss)**이 실제로 데이터 분포의 엔트로피를 나타내는가?

### 2.2 제안하는 방법 및 핵심 수식

#### (1) 중심 스케일링 법칙 (Equation 1.1)

$$L(x) = L_{\infty} + \left(\frac{x_0}{x}\right)^{\alpha_x}$$

- $L_{\infty}$: **환원 불가능한 손실(Irreducible Loss)** — 데이터 분포의 엔트로피 추정값
- $\left(\frac{x_0}{x}\right)^{\alpha_x}$: **환원 가능한 손실(Reducible Loss)** — KL 발산 추정값
- $x$: 모델 크기 $N$, 계산량 $C$, 또는 데이터셋 크기 $D$
- $\alpha_x$: 도메인 및 변수에 따른 스케일링 지수

#### (2) 정보이론적 해석 (Equation 1.2)

$$L_{\infty} \approx S(\text{True}) \quad \text{("Irreducible Loss")}$$

$$\left(\frac{x_0}{x}\right)^{\alpha_x} \approx D_{KL}(\text{True}\|\text{Model}) \quad \text{("Reducible Loss")}$$

**해석:** Cross-Entropy Loss는 다음과 같이 분해됩니다:

$$\mathbb{E}_{x \sim P}\left[\log \frac{1}{Q(x)}\right] = D_{KL}(P\|Q) + S(P)$$

여기서 $P$는 진짜 데이터 분포, $Q$는 모델 분포입니다. 즉:
- $L_{\infty}$은 데이터의 **진짜 엔트로피**
- Reducible Loss는 모델이 아직 **학습하지 못한 정보량** (KL 발산)

#### (3) 최적 모델 크기 (Equation 1.4)

$$N_{\text{opt}} \propto C^{\beta}, \quad \beta \approx 0.7 \text{ (모든 도메인에서 유사)}$$

이로부터 Compute-Optimal 학습에서 데이터셋 크기와 모델 크기의 관계:

$$D \propto C^{1-\beta} \propto N^{\frac{1-\beta}{\beta}} \approx N^{0.4}$$

이는 **데이터셋 크기가 모델 크기보다 훨씬 느리게 증가해도 된다**는 반직관적 결론입니다.

#### (4) 계산량 추정

$$C \equiv 6NE, \quad E = SB$$

- $N$: non-embedding 파라미터 수
- $E$: 학습 중 처리한 총 토큰 수
- $S$: 파라미터 업데이트 횟수, $B$: 배치 크기(토큰 수)

#### (5) 멀티모달 정보이득 (Equation 1.3)

$$\text{Infogain} \equiv \frac{I(\text{text}, \text{image})}{L(\text{text})}$$

$$I(\text{text}, \text{image}), \; \text{Infogain} \approx \lambda \log\left(\frac{N}{N_c}\right) \quad \text{(Equation 4.1)}$$

이 비율은 반드시 $[0, 1]$ 구간에 속하며, 멀티모달 모델 성능의 상한을 제공합니다.

### 2.3 모델 구조

모든 도메인에서 **Decoder-Only Transformer**를 동일하게 사용:

| 설정 항목 | 언어/멀티모달 | 수학/이미지/비디오 |
|-----------|--------------|-----------------|
| FC 레이어 크기 | $4d_{\text{model}}$ | $d_{\text{model}}$ |
| Attention 레이어 크기 | $d_{\text{model}}$ | $d_{\text{model}}/4$ |
| 최적 Aspect Ratio ($d_{\text{model}}/n_{\text{layer}}$) | $\sim 100$ | $\sim 5{-}10$ |
| Attention 유형 | Dense | Sparse (이미지/비디오) |

**핵심 발견:** 이미지·수학 모델은 언어 모델보다 **10배 이상 깊은(deeper) 구조**가 최적입니다.

이미지 인코딩:
- 픽셀 수준: $8\times8$, $16\times16$, $32\times32$ (RGB, 토큰 수 $= 3R^2$)
- VQ-VAE 인코딩: $64\times64$ 이미지를 $16\times16$ 또는 $32\times32$ VQ 코드로 압축

### 2.4 성능 향상

**도메인별 스케일링 지수 요약 (Table 1에서 발췌):**

| 도메인 | $\alpha_N$ (모델 크기) | $\alpha_C$ (계산량) | $\beta$ (최적 크기) |
|--------|----------------------|-------------------|-------------------|
| 이미지 8×8 | 0.24 | 0.19 | 0.64 |
| 이미지 16×16 | 0.22 | 0.16 | 0.75 |
| 이미지 32×32 | 0.13 | 0.10 | 0.65 |
| 비디오 VQ 16×16 | 0.24 | 0.14 | 0.71 |
| 수학(외삽) | 0.16 | 0.17 | 0.69 |
| 언어 | 0.070 | 0.048 | 0.73 |

**핵심 결과:**
- 10억 파라미터 모델이 $8\times8$ 해상도 YFCC100M 이미지 분포를 거의 완벽하게 모델링 (Reducible Loss $\approx$ 수 nats/image)
- ImageNet 32×32 분류: 사전학습 후 파인튜닝 시 모델 크기에 따라 순수 거듭제곱 법칙 추종 (지수 0.089~0.105)

### 2.5 한계

1. **이론적 설명 부재:** 왜 이러한 거듭제곱 법칙이 성립하는지 이론적 근거가 없음
2. **데이터셋 크기 vs 계산량 스케일링의 불일치(Section 6):** $L(D)$와 $L(C)$ 트렌드를 외삽하면 어느 시점에 모순이 발생함 — 해결책 불명확
3. **수렴 미달:** 가장 큰 모델들은 완전히 수렴하지 않아 $L(N)$ 해석에 주의 필요
4. **수학 외삽 한계:** 모델 크기 증가 자체가 '강한 일반화(strong generalization)'에 직접적인 이점을 제공하지 않음
5. **자연어의 엔트로피 추정 불가:** 현재 규모의 언어 모델로는 자연어의 $L_{\infty}$ 추정 불가

---

## 3. 모델의 일반화 성능 향상 가능성 (심층 분석)

### 3.1 일반화의 두 가지 의미

논문은 **일반화를 두 층위**에서 다룹니다:

**① 표준적 의미 (Train → Test):** 테스트 손실이 훈련 손실을 얼마나 잘 따라가는가

**② 야심찬 의미 (분포 외삽):** 훈련 분포 밖의 더 어려운 문제로 성능을 일반화할 수 있는가

### 3.2 생성 모델 → 다운스트림 태스크 일반화 (핵심 발견)

**ImageNet 분류 파인튜닝 실험 (Section 3.4)의 결론:**

$$\text{Classification Loss}(N) \propto \left(\frac{N}{1.72 \times 10^{10}}\right)^{-0.105}$$

$$\text{Error Rate}(N) \propto \left(\frac{N}{2.09 \times 10^3}\right)^{-0.089}$$

**중요한 함의:**
- 생성 손실이 $L_{\infty}$에 근접하여 **포화(bending)되는 것처럼 보여도**, 분류 성능은 **계속 순수 거듭제곱 법칙으로 향상**
- 이는 "'마지막 몇 비트(last few bits)'에 중요한 의미 정보가 있다"는 것을 의미
- **사전학습이 고효율 정규화기(regularizer)로 작용**: 처음부터 학습한 모델은 큰 모델에서 오버피팅하지만, 사전학습 모델은 지속 향상

> **결론:** 생성 Loss의 포화가 표현 품질의 포화를 의미하지 않습니다. 스케일링은 여전히 의미 있는 표현 향상을 제공합니다.

### 3.3 수학 문제의 분포 외 일반화 (Section 5)

수학 문제 풀이 실험에서 훈련 분포 밖 난이도 $s > 10$에 대한 외삽:

$$L_{\text{extrapolate}}(N) = 0.28 + \left(\frac{N}{1.1 \times 10^4}\right)^{-0.16}$$

**핵심 발견 (Figure 13):**
- 외삽 성능은 **주로 훈련 분포에서의 성능을 통해** 모델 크기에 의존
- 동일한 훈련 손실을 달성하는 다른 크기의 모델들은 외삽 테스트에서 **거의 동일한 성능**을 보임
- 즉, **모델 크기 자체는 '강한 일반화' 능력에 직접적 이점을 주지 않음** — 훈련 분포에서의 성능 향상을 통해 간접적으로 기여

**수식적 표현:**
$$L_{\text{test}}(\text{난이도}) \approx f(L_{\text{train}}) \quad \text{(모델 크기와는 독립적)}$$

### 3.4 분포 간 일반화 (Appendix E.2)

YFCC100M으로 학습한 모델을 ImageNet에서 평가하면:

$$L(\text{ImageNet}) = D_{KL}(\text{ImageNet}\|\text{YFCC100M}) + S(\text{ImageNet})$$

Figure 32에서 ImageNet Loss도 거듭제곱 법칙을 따름이 확인 → **오프-분포(off-distribution)에서도 스케일링 법칙의 일반화 성능 확인**

### 3.5 멀티모달 일반화 및 Infogain 포화

$$\text{Infogain} < 1 \quad \text{(이론적 상한)}$$

$$\text{Infogain}(N) \approx 0.015 \log\left(\frac{N}{5.03 \times 10^5}\right) \quad \text{(Text-to-Image)}$$

10억 파라미터 모델의 Infogain $\approx 10\%$ → **20%에 도달하려면 $N \approx 3 \times 10^{12}$ 파라미터 필요**

이는 현재 모델들의 멀티모달 일반화 능력이 여전히 매우 제한적임을 보여줍니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

#### (1) 패러다임 전환: 아키텍처 중심 → 스케일 중심

스케일링 법칙은 연구 초점을 **"어떤 아키텍처가 좋은가?"에서 "얼마나 크게 스케일해야 하는가?"**로 이동시킵니다. GPT-4, Gemini 등 대형 모델 개발의 이론적 토대를 제공했습니다.

#### (2) 자원 배분 최적화

$$N_{\text{opt}} \propto C^{0.7}, \quad D \propto N^{0.4}$$

이 공식은 **Compute-Optimal 학습**(Chinchilla, Hoffmann et al., 2022)의 직접적인 선행 연구로 작용했습니다.

#### (3) 멀티모달 AI 로드맵 제시

Infogain 스케일링 법칙은 현재 멀티모달 모델의 한계를 정량화하고, 목표 모델 크기를 예측하는 도구를 제공합니다.

#### (4) 다운스트림 태스크 예측 가능성

생성 손실과 다운스트림 성능의 연결은 **사전학습 모델 선택의 과학적 기준**을 제공합니다.

### 4.2 앞으로 연구 시 고려할 점

#### (1) 스케일링 법칙의 붕괴 시점 탐구 ⚠️

논문 Section 6에서 지적한 $L(D)$와 $L(C)$ 불일치 문제는 여전히 미해결입니다:

$$C(D) \approx (5 \times 10^{-42}) D^{3.9}$$

이 외삽의 교차 지점에서 어떤 트렌드가 먼저 붕괴하는지 검증이 필요합니다.

#### (2) 이론적 기반 마련

지수 $\alpha_x$와 $\beta \approx 0.7$의 보편성에 대한 이론적 설명이 없습니다. 데이터 다양체 차원과의 연결(Sharma & Kaplan, 2020), Neural Tangent Kernel 이론 등과의 결합 연구가 필요합니다.

#### (3) 데이터 품질과 스케일링의 관계

본 논문은 데이터 **양**에 집중했지만, **데이터 품질** (다양성, 정제 수준)이 스케일링 지수에 미치는 영향은 충분히 탐구되지 않았습니다.

#### (4) Compute-Optimal 학습 실증 검증

$D \propto N^{0.4}$의 sub-linear 스케일링은 논문 저자들도 인정하듯 실제로 $D \ll N$인 regime에서 아직 검증되지 않았습니다.

#### (5) 강한 일반화(Strong Generalization) 달성 방법

수학 외삽 실험에서 모델 크기가 직접적인 '강한 일반화'에 기여하지 못함이 밝혀졌습니다. 이를 달성하기 위한 **귀납적 편향(inductive bias)**, 사고 연쇄(Chain-of-Thought) 등의 추가적인 방법론 연구가 필요합니다.

#### (6) 아키텍처 다양성 검토

본 논문은 Transformer에만 집중합니다. State Space Model (Mamba, 2023), Mixture-of-Experts 등 새로운 아키텍처에서 동일한 스케일링 법칙이 성립하는지 검토가 필요합니다.

#### (7) 환경적 비용 고려

$N_{\text{opt}} \propto C^{0.7}$이 제시하는 "더 큰 모델을 더 짧게 학습"이라는 방향은 환경적 비용과의 트레이드오프를 고려해야 합니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 Chinchilla (Hoffmann et al., 2022)와의 비교

| 항목 | Henighan et al. (2020) | Hoffmann et al. "Chinchilla" (2022) |
|------|------------------------|--------------------------------------|
| 최적 모델/데이터 비율 | $D \propto N^{0.4}$ (데이터 sub-linear) | $N \propto D$ (동일 비율 증가 권장) |
| 분석 방법 | 고정 FLOPs에서 최적 N | 더 엄밀한 등등 비용 분석 |
| 핵심 결론 | 더 큰 모델, 짧은 학습 | **균형 있는** 모델/데이터 스케일링 |

**Chinchilla**는 본 논문의 $D \propto N^{0.4}$ 추정을 수정하여, **모델과 데이터를 동등하게 스케일**해야 함을 보였습니다. GPT-4, LLaMA 등은 Chinchilla 법칙을 반영했습니다.

> **참고:** Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." arXiv:2203.15556

### 5.2 GPT-4 Technical Report (OpenAI, 2023)

GPT-4는 소규모 모델 실험에서 대규모 모델 성능을 **예측**하는 데 스케일링 법칙을 활용했음을 공개했습니다. 본 논문의 실용적 적용 사례입니다.

> **참고:** OpenAI. (2023). "GPT-4 Technical Report." arXiv:2303.08774

### 5.3 DALL-E / DALL-E 2 / Stable Diffusion (2021-2022)

본 논문의 멀티모달 스케일링 연구는 DALL-E (Ramesh et al., 2021), DALL-E 2 (Ramesh et al., 2022) 개발의 이론적 기반이 되었습니다. 특히 텍스트-이미지 상호정보량 분석이 직접적으로 연결됩니다.

> **참고:** Ramesh, A., et al. (2021). "Zero-Shot Text-to-Image Generation." arXiv:2102.12092

### 5.4 Emergent Abilities (Wei et al., 2022)

$$\text{성능} = \begin{cases} \text{무작위 수준} & N < N_{\text{threshold}} \\ \text{급격한 향상} & N \geq N_{\text{threshold}} \end{cases}$$

Wei et al.은 스케일링 법칙의 **부드러운 Power-Law**와 달리, 일부 능력이 특정 모델 크기에서 **갑자기 출현(emergence)**함을 발견했습니다. 이는 본 논문의 연속적 스케일링 관점과 부분적으로 긴장 관계를 이룹니다.

> **참고:** Wei, J., et al. (2022). "Emergent Abilities of Large Language Models." arXiv:2206.07682

다만, Schaeffer et al. (2023)은 이 출현 현상이 비선형 평가 메트릭의 artifact일 수 있음을 주장합니다.

> **참고:** Schaeffer, R., et al. (2023). "Are Emergent Abilities of Large Language Models a Mirage?" arXiv:2304.15004

### 5.5 Mamba (Gu & Dao, 2023) - 새로운 아키텍처 도전

State Space Model 기반 Mamba는 Transformer 대비 선형 복잡도를 달성하면서도 경쟁적 성능을 보였습니다. 이는 본 논문이 전제한 "Transformer가 보편적 아키텍처"라는 가정에 새로운 시각을 제시합니다.

> **참고:** Gu, A., & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752

### 5.6 비교 요약표

| 논문 | 연도 | 본 논문과의 관계 | 핵심 차이 |
|------|------|----------------|-----------|
| Chinchilla (Hoffmann et al.) | 2022 | 스케일링 법칙 수정 | 최적 데이터/모델 비율 재정립 |
| GPT-4 Technical Report | 2023 | 실용적 적용 | 소규모 실험으로 대규모 예측 |
| Emergent Abilities (Wei et al.) | 2022 | 부분적 반론 | 비연속적 능력 출현 |
| Mamba (Gu & Dao) | 2023 | 아키텍처 도전 | Transformer 대안 제시 |
| DALL-E (Ramesh et al.) | 2021 | 직접 계승 | 멀티모달 생성 모델 구현 |

---

## 참고자료

**주요 참고 문헌 (본 논문 및 직접 인용):**

1. **Henighan, T., Kaplan, J., Katz, M., et al.** (2020). "Scaling Laws for Autoregressive Generative Modeling." *arXiv:2010.14701*

2. **Kaplan, J., McCandlish, S., Henighan, T., et al.** (2020). "Scaling Laws for Neural Language Models." *arXiv:2001.08361*

3. **Brown, T. B., Mann, B., Ryder, N., et al.** (2020). "Language Models are Few-Shot Learners." *arXiv:2005.14165*

4. **Vaswani, A., et al.** (2017). "Attention is All You Need." *NeurIPS 2017*

5. **van den Oord, A., Vinyals, O., & Kavukcuoglu, K.** (2018). "Neural Discrete Representation Learning (VQ-VAE)." *arXiv:1711.00937*

6. **Saxton, D., Grefenstette, E., Hill, F., & Kohli, P.** (2019). "Analysing Mathematical Reasoning Abilities of Neural Models." *arXiv:1904.01557*

**2020년 이후 비교 분석 참고 문헌:**

7. **Hoffmann, J., et al.** (2022). "Training Compute-Optimal Large Language Models (Chinchilla)." *arXiv:2203.15556*

8. **OpenAI.** (2023). "GPT-4 Technical Report." *arXiv:2303.08774*

9. **Wei, J., et al.** (2022). "Emergent Abilities of Large Language Models." *arXiv:2206.07682*

10. **Ramesh, A., et al.** (2021). "Zero-Shot Text-to-Image Generation (DALL-E)." *arXiv:2102.12092*

11. **Gu, A., & Dao, T.** (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *arXiv:2312.00752*

12. **Schaeffer, R., et al.** (2023). "Are Emergent Abilities of Large Language Models a Mirage?" *arXiv:2304.15004*

13. **Sharma, U., & Kaplan, J.** (2020). "A Neural Scaling Law from the Dimension of the Data Manifold." *arXiv:2004.10802*
