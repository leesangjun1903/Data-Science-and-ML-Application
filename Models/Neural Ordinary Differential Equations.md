# Neural Ordinary Differential Equations

## 주요 주장 및 기여 (간결 요약)
Neural Ordinary Differential Equations(ODE-Net) 논문은 **딥 신경망의 연속적 깊이 모델**을 제안하며, 다음 네 가지 핵심 기여를 제시한다.  
1. **연속적 깊이 파라미터화**: ResNet의 불연속 계층 대신, 상태의 미분방정식 $$\frac{dh(t)}{dt} = f(h(t), t, \theta) $$ 으로 숨겨진 표현을 정의하여 연속 깊이 모델을 제시.  
2. **상수 메모리 역전파**: 어드조인트 민감도 해법(adj oint sensitivity method)을 활용해 ODE 솔버를 블랙박스로 취급하고, 메모리 사용을 입력·출력 여유공간만으로 상수화.  
3. **적응적 계산 비용 제어**: 기존 ODE 솔버의 오차 감시 및 적응적 단계 조정 능력을 활용하여 입력 복잡도에 따라 함수 평가 횟수(“가상 계층 수”)를 조절.  
4. **연속 정규화 흐름(CNF)**: 연속적 변화 공식(Instantaneous Change of Variables, $$\partial_t \log p(z(t)) = -\mathrm{tr}\bigl(\partial_z f(z(t),t)\bigr)$$)을 유도하여, 역행렬 대신 트레이스 연산만으로 밀도 변화를 계산하는 효율적 가역 생성 모델 제안.  

***

## 1. 해결하고자 하는 문제  
딥 네트워크에서  
- 계층(layer) 수가 많아질수록 메모리 사용량이 선형으로 증가하며,  
- 고정된 불연속 계층 설계로 인해 계산량 조절이 불가능하고,  
- 정상적인 정규화 흐름(normalizing flow)은 야코비안 행렬식 계산 비용이 $$O(d^3)$$에 달함.  

이를 해결하기 위해 **모델의 깊이(depth)를 연속 변수(time)로 재정의**하고, ODE 솔버의 특성을 활용하여 메모리·계산 비용을 효율화하고 생성 모델 확장성을 높이고자 한다.  

***

## 2. 제안하는 방법  
### 2.1 연속 깊이 네트워크  
기존 Residual Network가  

$$
h_{t+1} = h_t + f(h_t, \theta_t)
$$

로 계층별 불연속 변환을 정의했다면, ODE-Net은  

$$
\frac{dh(t)}{dt} = f(h(t), t, \theta)
$$

라는 연속 미분방정식으로 숨겨진 상태를 모델링한다. 입력 $$h(0)$$에서 출력 $$h(T)$$까지의 변환은 ODE 솔버(예: Runge-Kutta, Adams 등)로  

$$
h(T) = \mathrm{ODESolve}\bigl(h(0),\,f,\,0,\,T,\,\theta\bigr)
$$

을 계산함으로써 얻는다.

### 2.2 상수 메모리 역전파  
손실 $$L(h(T))$$에 대한 파라미터 $$\theta$$ 및 초기 상태 $$h(0)$$의 그래디언트는 **어드조인트 상태** $$a(t) = \frac{\partial L}{\partial h(t)}$$ 를 도입하여, 다음 연립 ODE를 뒤로 적분(backward integration)함으로써 계산한다:  

$$
\frac{da(t)}{dt} = -\,a(t)^{\mathsf{T}}\,\frac{\partial f(h(t),t,\theta)}{\partial h}\,, 
\quad
\frac{dL}{d\theta} = -\int_{T}^{0}a(t)^{\mathsf{T}}\frac{\partial f(h(t),t,\theta)}{\partial \theta}\,dt.
$$

이로써 **앞방향 연산 상태를 저장하지 않고** 역전파가 가능해져, 메모리 사용량이 계층 수 $$L$$와 무관하게 상수 $$O(1)$$가 된다.

### 2.3 적응적 계산  
모델 평가 시 ODE 솔버의 **오차 허용치(tolerance)** 를 조절하여 함수 $$f$$ 평가 횟수(NFE, number of function evaluations)를 동적으로 결정할 수 있다. 이로 인해 **속도-정확도(trade-off)** 조정이 가능하다.

### 2.4 연속 정규화 흐름(CNF)  
비가역적 불연속 흐름에서 야코비안 행렬식(det $$\partial f/\partial z$$ ) 계산 비용은 $$O(d^3)$$이지만, 연속 구조에서는  

$$
\frac{\partial}{\partial t}\log p(z(t)) = -\,\mathrm{tr}\Bigl(\tfrac{\partial f(z(t),t)}{\partial z}\Bigr)
$$

이라는 **즉시 변화 공식**만으로 밀도 변화를 계산할 수 있어 비용이 $$O(d^2)$$에서 $$O(d)$$로 감소한다.

***

## 3. 실험 및 성능 향상  
- **MNIST 분류**: 6개 ResNet 블록을 가지는 ODE-Net은 0.42% 테스트 오류로 ResNet(0.41%)과 동등한 성능을 보이며, 파라미터 수 및 메모리 사용량이 감소.[1]
- **정규화 흐름**: 동일한 표현력(capacity) 하에서 CNF는 수렴 속도가 빠르며, planar flow 대비 KL 손실이 낮음.[1]
- **불규칙 시계열 예측**: 잠재 ODE 모델(latent ODE)은 RNN 대비 예측 RMSE를 크게 개선(예: 관측 30점 기준 0.1642 vs 0.3937).[1]

***

## 4. 한계 및 고려사항  
1. **오차 허용치 설정**: 전·후방 통합 시 각각의 tolerance 값을 수동 조정해야 하며, 잘못 설정 시 성능 저하 가능.  
2. **배치 처리**: 미니배치 통합은 상태 차원 수를 배치 크기만큼 늘려 통합해야 하므로, 특정 경우 평가 횟수가 배치별 개별 통합보다 증가할 수 있음.  
3. **수치 오차 누적**: 역전파 시 궤적 재통합으로 인한 오차가 축적될 수 있으며, 심각한 경우 체크포인팅(checkpointing)이 필요.  
4. **모델 특성 제약**: PICARD 정리 조건(Lipschitz 연속성, 무균주성)을 만족해야 해, 특정 비선형 활성화 함수나 무한 가중치 상황에서 해(unique solution)가 보장되지 않을 수 있음.

***

 Chen et al., “Neural Ordinary Differential Equations,” NeurIPS 2018.[1]

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/3a24616a-dfd4-403b-b075-1cea59eb6f9a/1806.07366v5.pdf
