# PyTorch Distributed: Experiences on Accelerating Data Parallel Training

### 1. 논문의 핵심 주장과 주요 기여

**"PyTorch Distributed: Experiences on Accelerating Data Parallel Training"** 논문은 Facebook AI에서 개발한 PyTorch의 분산 데이터 병렬 모듈(DistributedDataParallel, DDP)의 설계, 구현 및 성능 평가를 다룹니다. 논문의 핵심 주장은 분산 딥러닝 훈련에서 **계산과 통신 간의 미묘한 의존성을 최적화함으로써 선형에 가까운 확장성(near-linear scalability)**을 달성할 수 있다는 것입니다.[1]

주요 기여는 다음과 같습니다:[1]

1. **산업 수준 분산 훈련 솔루션의 공개**: 널리 채택되는 분산 훈련 프레임워크의 설계와 구현 과정을 상세히 공개
2. **실제 구현상 주의사항 강조**: 기존 연구에서 간과된 복수 그래프(pluralized graphs) 문제 등 실제 배포 시 발생하는 문제들 식별
3. **성능 최적화 경험 공유**: 내부 팀과 오픈소스 커뮤니티를 통해 수집한 성능 튜닝 경험 공유

***

### 2. 해결하고자 하는 문제와 핵심 과제

#### 2.1 근본적인 문제

분산 데이터 병렬 훈련에서 모든 모델 복제본이 일관된 상태를 유지하려면 **각 반복마다 모든 프로세스가 동일한 기울기를 사용**해야 합니다. 하지만 이를 효율적으로 구현하는 것은 다음 세 가지 핵심 과제를 제시합니다:[1]

1. **수학적 동치성(Mathematical Equivalence)**: 분산 훈련의 결과가 단일 기계에서의 로컬 훈련과 동일해야 함
2. **비침투적이고 차단 가능한 API(Non-intrusive and Interceptive API)**: 기존 로컬 훈련 코드를 최소한의 수정으로 분산 훈련으로 전환할 수 있어야 함
3. **높은 성능(High Performance)**: 계산과 통신의 미묘한 의존성을 탐색하여 효율적인 훈련 처리량 구현

#### 2.2 매개변수 평균화의 한계

논문은 기존 매개변수 평균화 방식의 두 가지 근본적인 문제를 지적합니다:[1]

$$\text{Parameter Averaging Issue 1: 수학적 부동등성}$$

매개변수 평균화는 로컬 훈련과 다른 결과를 생성할 수 있습니다. 이는 최적화기가 과거 로컬 기울기 값에 의존할 때 특히 심각합니다(예: 모멘텀). 서로 다른 모델 복제본이 서로 다른 기울기를 보면 최적화기의 상태가 점진적으로 발산하여 충돌하는 경사 하강 방향을 초래할 수 있습니다.[1]

$$\text{Parameter Averaging Issue 2: 계산-통신 겹침 불가능}$$

매개변수 평균화는 계산(역전파)과 통신(평균 계산)을 겹치지 않을 수 있는 비겹침 단계로 조직합니다. 따라서 한 유형의 리소스는 항상 유휴 상태입니다.[1]

***

### 3. 제안하는 방법 및 기울기 감소 알고리즘

#### 3.1 기울기 동기화 기반 접근

PyTorch DDP는 매개변수 대신 **기울기를 동기화**하는 데이터 병렬화를 채택합니다. 이를 통해:[1]

- 각 반복마다 모든 모델 복제본이 동일한 기울기를 받음
- 독립적인 최적화기가 로컬 모델 복제본을 같은 상태로 유지
- 수학적 동치성 보장

#### 3.2 기울기 버킷팅(Gradient Bucketing)

$$\text{Problem: 작은 텐서에서 AllReduce 비효율}$$

그림 2에서 보듯이, AllReduce 작업은 작은 텐서에서 효율이 떨어집니다. NCCL에서 60M 개의 torch.float32 매개변수를 AllReduce할 때:[1]

$$T_{AllReduce} \propto \frac{1}{n_{params\_per\_reduce}}$$

여기서 $$n_{params\_per\_reduce}$$는 AllReduce당 매개변수 수입니다.[1]

$$\text{해결책: 기울기 버킷팅}$$

대신 작은 기울기를 버킷으로 모아 한 번의 AllReduce로 처리합니다:[1]

$$B = \{g_{i_1}, g_{i_2}, \ldots, g_{i_k}\}$$

여기서 각 버킷 $$B$$는 최대 크기 제한 $$c$$를 가집니다:

$$|B| \leq c$$

최적의 버킷 크기를 결정하려면 기울기 준비 시간과 AllReduce 시간의 균형이 필요합니다.[1]

#### 3.3 계산과 통신 겹침

$$\text{핵심 전략: 비동기 AllReduce 실행}$$

각 버킷의 모든 기울기가 준비되면 즉시 AllReduce를 시작하되, 백워드 패스가 완료될 때까지 기다리지 않습니다:[1]

$$\text{AllReduce}_i \text{ 시작 조건}: \forall g \in B_i, \quad g \text{ ready}$$

이를 구현하기 위해 PyTorch는 **자동 그래드 후킹(autograd post-hook)**을 사용합니다.[1]

$$\text{문제점: 기울기 준비 순서 불일치}$$

그림 3에서 보듯이, 동적으로 구성된 자동 그래드 그래프로 인해 서로 다른 프로세스에서 기울기 준비 순서가 다를 수 있습니다. AllReduce 내용이 불일치하면 잘못된 감소 또는 프로그램 충돌이 발생합니다.[1]

$$\text{해결책: 역순 정렬 사용}$$

PyTorch v1.5는 모델 매개변수의 역순을 버킷팅 순서로 사용합니다. 계층이 호출되는 순서와 동일하게 등록되면, 역순이 백워드 패스에서의 대략적인 기울기 계산 순서를 나타낼 것이라 가정합니다:[1]

```math
\text{bucket\_order} = \text{reverse}(\text{model.parameters}())
```

#### 3.4 미사용 매개변수 처리

$$\text{문제: 동적 그래프에서 일부 기울기 건너뜀}$$

일부 반복에서는 훈련 그래프의 서브셋만 포함될 수 있으며, 어떤 기울기는 건너뛰어질 수 있습니다. 이 경우 예상된 후킹 신호가 없어 백워드 패스가 멈춤(hang)으로 인해 교착 상태(deadlock)에 빠질 수 있습니다.[1]

$$\text{해결책: 자동 그래프 순회를 통한 미리 표시}$$

DDP는 포워드 패스의 출력 텐서에서 자동 그래프를 순회하여 참여하는 모든 매개변수를 찾습니다. 참여하지 않는 매개변수는 포워드 패스 끝에서 미리 준비됨으로 표시됩니다:[1]

$$P_{participating} = \{p : p \text{ in autograd graph from output tensors}\}$$

```math
\forall p \notin P_{participating}, \quad \text{mark\_ready}(p)
```

#### 3.5 Algorithm 1: DistributedDataParallel 의사코드

```
Algorithm 1: DistributedDataParallel

Input: 프로세스 순위 r, 버킷 크기 제한 c, 로컬 모델 net

Function constructor(net):
    if r = 0 then
        net 상태를 다른 프로세스에 브로드캐스트
    init 버킷, 매개변수를 역순으로 버킷에 할당
    for p in net.parameters() do
        acc ← p.grad accumulator
        acc → add post hook(autograd hook)

Function forward(inp):
    out = net(inp)
    out에서 자동 그래드 그래프를 순회하여 미사용 매개변수를 준비됨으로 표시
    return out

Function autograd hook(param index):
    bucket bi와 버킷 오프셋을 param index로 얻음
    매개변수 var를 param index로 얻음
    view ← bi.narrow(offset, var.size())
    view.copy(var.grad)
    if all grads in bi are ready then
        mark bi as ready
        순서대로 준비된 버킷에서 AllReduce 시작
    if all buckets are ready then
        모든 AllReduce 작업 완료 대기
```


#### 3.6 기울기 누적 및 no_sync 모드

큰 배치를 처리하기 위해 여러 마이크로배치에서 기울기를 누적할 수 있습니다. 이를 지원하기 위해 `no_sync()` 컨텍스트 매니저가 제공됩니다:[1]

```python
ddp = DistributedDataParallel(net)
with ddp.no_sync():
    for inp, exp in zip(inputs, expected_outputs):
        # 동기화 없음, 기울기 누적
        loss_fn(ddp(inp), exp).backward()
# 기울기 동기화
loss_fn(ddp(another_inp), another_exp).backward()
opt.step()
```

이 모드에서 모든 DDP 후킹은 비활성화되고, 컨텍스트 밖의 첫 번째 백워드 패스에서 누적된 기울기를 한 번에 동기화합니다.[1]

***

### 4. 모델 구조와 아키텍처

#### 4.1 DDP 구성 요소

DDP는 세 가지 주요 계층으로 구성됩니다:[1]

```
┌─────────────────────────────────────────┐
│   Python API 프론트엔드                │
│ (forward, no_sync context manager)      │
├─────────────────────────────────────────┤
│   C++ 기울기 감소 핵심 알고리즘        │
│ (버킷팅, 후킹, AllReduce 조율)        │
├─────────────────────────────────────────┤
│   c10d 집합 통신 라이브러리            │
│ (NCCL, Gloo, MPI)                      │
└─────────────────────────────────────────┘
```


#### 4.2 AllReduce 원시 연산

AllReduce는 분산 훈련의 주요 통신 연산입니다:[1]

```math
\text{AllReduce}(x_1, x_2, \ldots, x_n, \text{op}) \rightarrow \left\{\sum_{i=1}^{n} x_i \otimes \text{op}, \ldots, \sum_{i=1}^{n} x_n \otimes \text{op}\right\}
```

모든 참여 프로세스가:
1. 동일한 크기의 텐서를 제공
2. 주어진 산술 연산(합, 곱, 최소, 최대)을 집합적으로 적용
3. 동일한 결과를 받음

NCCL과 Gloo는 링 기반 AllReduce 또는 트리 기반 AllReduce 등 효율적인 알고리즘을 구현합니다.[1]

#### 4.3 Python 프론트엔드 특징

**구성 가능한 매개변수:**[1]

- `process_group`: AllReduce를 실행할 프로세스 그룹 지정
- `bucket_cap_mb`: AllReduce 버킷 크기 제어
- `find_unused_parameters`: 미사용 매개변수 자동 감지 여부

**모델 버퍼 처리:**[1]

배치 정규화 같은 계층의 상태(러닝 분산, 러닝 평균)는 순위 0 프로세스에서 브로드캐스트됩니다.

***

### 5. 성능 향상 분석

#### 5.1 지연 시간 분석

32개 GPU를 통한 ResNet50 및 BERT 모델의 훈련 반복별 지연 시간 분해:[1]

**ResNet50 (NCCL):**
- 포워드 패스: ~15%
- 백워드 패스 (계산): ~50%
- 백워드 패스 (AllReduce): ~35%
- 최적화기 스텝: ~5%

계산과 통신 겹침을 통해 **38.0% 속도 향상** 달성[1]

**BERT (NCCL):**
- 포워드 패스: ~10%
- 백워드 패스 (계산): ~40%
- 백워드 패스 (AllReduce): ~45%
- 최적화기 스텝: ~5%

계산과 통신 겹침을 통해 **35.2% 속도 향상** 달성[1]

#### 5.2 버킷 크기의 영향

최적 버킷 크기는 모델과 통신 백엔드에 따라 달라집니다:[1]

**ResNet50:**
- NCCL: 10-25 MB (최적)
- Gloo: 5 MB (최적)

**BERT:**
- NCCL: 50 MB (최적, 모델 크기 증가로 인한 상향)
- Gloo: 5 MB (최적)

버킷 크기를 부적절히 설정하면 **2배 이상의 성능 저하** 발생 가능[1]

#### 5.3 확장성 측정

256개 GPU에서의 성능:[1]

**ResNet50:**
- NCCL: 100% 지연 시간 증가 (256 × 50% = 128배 확장 인자)
- Gloo: 3배 지연 시간 증가

**BERT:**
- NCCL: 3배 이상 지연 시간 증가
- Gloo: 6배 지연 시간 증가

적절히 구성하면 **near-linear scalability** 달성 가능[1]

#### 5.4 동기화 건너뛰기의 효과

기울기 동기화를 매 n번 반복마다 수행:[1]

**ResNet50 (256 GPUs, NCCL):**
- 매 1회: 기본 성능
- 매 2회: 15% 속도 향상
- 매 4회: 25% 속도 향상
- **매 8회: 38% 속도 향상**

**수렴 속도 분석:**
- 배치 크기 8: 동기화 건너뛰기 후에도 무시할 수 있는 수렴 저하
- 배치 크기 256: 큰 배치와 동기화 건너뛰기 결합 시 최종 손실 악화 (더 작은 학습률 필요)[1]

***

### 6. 모델의 일반화 성능 향상 가능성

#### 6.1 일반화 성능에 미치는 영향

DDP는 기울기 동기화를 통해 **훈련 과정의 수렴 속도와 최종 모델 성능에 영향**을 미칩니다:[1]

$$L_{\text{distributed}}(\theta) = \frac{1}{m} \sum_{p=1}^{m} \sum_{i \in D_p} \ell(f_\theta(x_i), y_i)$$

여기서 $$D_p$$는 프로세스 p의 데이터 부분집합입니다. 모든 프로세스가 동일한 기울기를 사용하므로:[1]

$$\nabla L_{\text{distributed}} = \text{AllReduce}\left(\frac{1}{|D_p|} \sum_{i \in D_p} \nabla \ell(f_\theta(x_i), y_i)\right)$$

#### 6.2 손실 함수 평면도와 최적화 안정성

논문에서는 명시적으로 언급되지 않지만, 기울기 버킷팅과 겹침은 **손실 함수의 평면도**에 영향을 미칠 수 있습니다:[1]

**Gradient Centralization의 영향 (최신 연구):**

2020년 이후 관련 연구에서 제안된 Gradient Centralization (GC)는 다음을 보여줍니다:[2]

$$\hat{g}_t = g_t - \mathbb{E}[g_t]$$

이 기법을 적용하면:
- 손실 함수의 Lipschitzness 개선
- 더 평탄한 최소값 도달 (generalization 개선)[2]

#### 6.3 배치 크기와 학습률의 관계

논문의 실험에서 배치 크기와 동기화 건너뛰기의 상호작용이 일반화에 영향을 미칩니다:[1]

**작은 배치 (크기 8):**
$$L_{\text{final}} \approx L_{\text{baseline}} + \epsilon_1, \quad |\epsilon_1| \approx 0$$

동기화 건너뛰기가 수렴에 무시할 수 있는 영향

**큰 배치 (크기 256):**
$$L_{\text{final}} \approx L_{\text{baseline}} + \epsilon_2, \quad |\epsilon_2| > 0$$

학습률을 감소시켜야 부정적 영향 완화 (더 큰 유효 배치에 대한 적응 필요)[1]

#### 6.4 2020년 이후 일반화 성능 관련 최신 연구

**Communication-Efficient Distributed Training for Collaborative Flat Optima Recovery (2025):**

플랫 최소값 가설(flat minima hypothesis)을 기반으로, DPPF (Distributed Pull-Push Force) 알고리즘은:[3]

$$\min_w L(w) \text{ subject to} \quad \text{Sharpness}(w) \leq \epsilon$$

이를 통해:
- 통신 효율성 유지 ながら 일반화 성능 향상
- 로컬 기울기 방법과 동기식 기울기 평균화보다 우수한 성능[3]

**Pseudo-Asynchronous Local SGD (2025):**

PALSGD는 통신 빈도를 줄이면서 모델 일관성 유지:[4]

$$\text{동기화 간격} = T_{\text{local}} \times \text{(표준 Local SGD보다 길음)}$$

이 방법은:
- 표준 Local SGD와 유사한 수렴 속도 유지
- 통신 횟수 크게 감소
- 더 나은 일반화 성능 달성[4]

**DreamDDP: Layer-wise Scheduled Partial Synchronization (2025):**

계층별 선택적 동기화를 통해:[5]

$$\text{AllReduce}_i = \begin{cases} \text{실행} & \text{if layer } i \text{ 중요} \\ \text{건너뜀} & \text{if layer } i \text{ 덜 중요} \end{cases}$$

- ResNet-18/50, GPT-2, Llama-2에서 성능 향상 달성[5]

***

### 7. 한계와 개선 기회

#### 7.1 DDP의 주요 한계

**1. 기울기 준비 순서 불예측성:**[1]

동적 그래프 구성으로 인해 정확한 버킷팅 순서를 구성 시점에 결정할 수 없습니다. 현재 구현은 휴리스틱(모델 매개변수의 역순)에만 의존합니다.[1]

**2. 고정 버킷 할당의 비효율성:**[1]

- 레이어 드롭핑 같은 기술에서 일부 매개변수가 건너뛰어져도 버킷 할당이 변경되지 않음
- 불필요한 통신 오버헤드 증가

**3. 계층 간 불균형 통신:**[1]

- 작은 모델과 대형 모델 간 통신 오버헤드의 차이 (모델 크기에 따라 다양함)
- 모든 배치 크기에 최적화된 단일 버킷 크기 없음

#### 7.2 논문에서 제시한 미래 개선 방향

**1. 기울기 순서 예측 (Gradient Order Prediction):**[1]

기울기 준비 순서를 자동 그래드 후킹으로 추적하고, 버킷 할당을 동적으로 업데이트합니다.

```math
\text{traced\_order}_t = \text{autograd\_hook\_trace}(\text{backward pass}_t)
```

```math
\text{update\_bucket\_mapping}(\text{traced\_order}_t) \quad \text{if } \text{traced\_order}_t \neq \text{cached\_order}
```

*단점*: 추적 오버헤드, 여러 반복 간 불일치 처리의 복잡성[1]

**2. 레이어 드롭핑 (Layer Dropping):**[1]

훈련 중 계층을 무작위로 건너뜀으로써 과적합 방지 및 훈련 가속화. DDP는 이를 지원해야 하지만, 고정 매개변수-버킷 매핑의 제약이 있습니다.

*해결책*:
- 개별 매개변수가 아닌 계층 수준의 버킷팅
- 모든 프로세스가 동일한 난수 시드로 동기화[1]

**3. 기울기 압축 (Gradient Compression):**[1]

기울기 통신량 감소를 위해 적응형 압축을 적용합니다.

$$g_{\text{compressed}} = \text{compress}(g, \text{precision}_{\text{adaptive}}(||g||))$$

예: 1-bit 확률적 기울기 하강 (1-bit SGD)[1]

#### 7.3 2020년 이후 최신 개선 기법

**Gradient Compression 기법 (2021-2025):**[6]

기울기 전송 크기를 줄이는 여러 기법들이 제안되었습니다:

$$\text{Compression Ratio} = \frac{||g||}{||g_{\text{compressed}}||} = 10-100\times$$

- Sparse-Top-K 선택
- Quantization (8-bit 또는 16-bit)
- Low-rank 근사[6]

**Themis: 네트워크 대역폭 인식 스케줄링 (2022):**[7]

다차원 네트워크에서 특정 차원의 대역폭이 충분하지 않을 때, 스케줄링 전략을 동적으로 조정합니다.[7]

**Communication-Efficient Local SGD 변형 (2024-2025):**

Local SGD를 확장하여 통신 빈도를 줄이면서도 수렴 속도 유지:

$$\text{Global Update} = \frac{1}{m} \sum_{p=1}^{m} \theta_p^{(T)}$$

여기서 $$T$$는 동기화 간격입니다.[4]

***

### 8. 논문의 앞으로의 영향과 연구 시 고려 사항

#### 8.1 학계 및 산업에 미친 영향

**PyTorch DDP의 광범위한 채택:**[1]

2020년 5월-6월 Facebook 내부 연구: 해당 기간 GPU 시간의 **60% 이상**이 PyTorch DDP를 사용하는 음성, 비전, 모바일 비전, 번역 등 다양한 애플리케이션에 소비되었습니다.[1]

**후속 프레임워크 영향:**

- TensorFlow v2.2: 유사한 계산-통신 겹침 기법을 Multi Worker Mirrored Strategy로 도입[1]
- Horovod: PyTorch DDP와 유사한 API 제공[1]

#### 8.2 분산 훈련 연구의 확장

논문에서 강조한 기술들이 이후 연구의 기초가 됨:[1]

1. **GradientFlow (2019):** 버킷팅 + 선택적 동기화 건너뛰기 결합[1]
2. **ByteScheduler (2019):** 프레임워크 독립적인 통신 스케줄링[1]
3. **PACE (2020):** 최적 통신 스케줄 계산 및 AllReduce 분할[1]

#### 8.3 다중 병렬화 전략 통합

논문 이후 하이브리드 병렬화 연구가 활성화됨:[1]

- **Mesh-TensorFlow (2018):** 데이터 병렬화 + 모델 병렬화 결합
- **ZeRO (2020):** 매개변수, 기울기, 최적화기 상태 분할을 통한 초대형 모델 훈련
- **PipeDream (2019):** 파이프라인 병렬화 + 데이터 병렬화
- **Parallax (2019):** 희소성 인식 하이브리드 병렬화[1]

#### 8.4 향후 연구 시 고려할 점

**1. 이질성 있는 환경에서의 최적화:**

현실의 분산 클러스터에서 기기 간 네트워크 대역폭, 계산 속도, 메모리가 다양합니다:[8][9]

```math
\text{고려사항}: \text{adaptively\_select\_bucket\_size}(B_p, N_p, \text{speedup}_p)
```

여기서 $$B_p$$는 프로세스 p의 대역폭, $$N_p$$는 계산 속도, $$\text{speedup}_p$$는 가속기 성능입니다.[9][8]

**2. 저대역폭 시나리오 (Federated Learning):**

가장자리 기기의 제한된 대역폭에 대응:[10]

$$\text{Communication Cost} \ll \text{Computation Cost}$$

이 경우, 더 적극적인 기울기 압축과 덜 빈 동기화가 필요합니다.[10]

**3. 동적 모델 구조:**

Transformer 기반 LLM 훈련에서 변하는 계산 그래프에 대응:[11]

- Transformer 특화 기울기 압축 (TAGC)[11]
- 계층별 선택적 동기화 (DreamDDP)[5]

**4. 일반화 성능 보장:**

**플랫 최소값 가설(Flat Minima Hypothesis):**

더 넓은 최소값을 찾는 훈련이 더 나은 일반화를 제공합니다:[3]

$$\text{Generalization Gap} \propto \text{Loss Curvature at Minima}$$

따라서 지표:
$$\text{Sharpness} = \lambda_{\max}(\mathcal{H}(\theta))$$

를 최소화하는 훈련 방법이 일반화 성능을 개선합니다.[3]

**5. 통신 효율성과 정확도의 트레이드오프:**

기울기 압축, 기울기 건너뛰기, 동기화 간격 조정 시 다음을 고려:

$$\text{Accuracy} = f(\text{Communication Cost}, \text{Compression Ratio}, \text{Sync Interval})$$

최적 균형점 찾기가 핵심입니다.[6]

***

### 9. 2020년 이후 관련 최신 연구 현황

#### 9.1 통신 효율성 개선 (2021-2025)

| 기법 | 연도 | 주요 개선사항 | 성능 |
|------|------|-------------|------|
| Learned Gradient Compression[6] | 2021 | 적응형 압축률 | 통신 50-70% 감소 |
| Themis[7] | 2022 | 네트워크 대역폭 인식 스케줄링 | 멀티차원 네트워크에서 최적화 |
| AB-Training[12] | 2024 | Low-rank 표현 + 독립 훈련 그룹 | 네트워크 트래픽 70% 감소 |
| Caesar[13] | 2025 | 저편차 모델/기울기 압축 | Federated Learning에서 적응형 압축 |
| SEPARATE[14] | 2025 | Low-rank 기울기 투영 | GPT-2 훈련 2배 가속 |

#### 9.2 확장 가능한 분산 훈련 (2023-2025)

| 방법 | 핵심 특징 | 결과 |
|------|---------|------|
| DreamDDP[5] | 계층별 선택적 동기화 | 32 GPU에서 성능 향상 확인 |
| PALSGD[4] | 의사-비동기 Local SGD | 통신 빈도 ↓ 수렴 속도 ↑ |
| DPPF (Flat Optima)[3] | 플랫 최소값 추구 | 일반화 성능 ↑ |
| PredTOP[15] | 기울기 Transformer 기반 레이턴시 예측 | 3D 병렬화에서 최적 전략 자동 선택 |
| TAGC[11] | Transformer 특화 기울기 압축 | LLM 훈련에서 통신 오버헤드 감소 |

#### 9.3 Federated Learning 및 프라이버시 (2024-2025)

**FDA (Federated Dynamic Averaging, 2025):**[10]

조건부 동기화 프로토콜:
- 워커 간 모델 분산 초과 시에만 동기화
- 통신 비용을 수개 자릿수 감소

$$\text{Sync if } \text{Var}(w_1, w_2, \ldots, w_m) > \tau$$

[10]

#### 9.4 대규모 모델 훈련 (2024-2025)

**Zero-Redundancy Optimization (ZeRO) 확장:**

ZeRO를 기반으로 한 최신 연구:
- 매개변수, 기울기, 옵티마이저 상태 분할
- 초대형 LLM 훈련 지원
- PyTorch FSDP (Fully Sharded Data Parallel)로 구현[16][1]

**양자화 기반 분산 훈련 (2023-2024):**[16]

대형 모델 분산 훈련에서:
$$g_{\text{quantized}} = \text{quantize}(g, \text{bits} \in \{4, 8\})$$

- 가중치 정량화로 통신 4-8배 감소
- 수렴 속도 이론적 보장 제공[16]

#### 9.5 HPC 환경에서의 3D 병렬화 (2025)

**Commshift 최적화:**[9]

XLA 컴파일러에서 3D 병렬화 (데이터 + 텐서 + 파이프라인):
- 특정 통신 명령을 계산 단계 사이에 이동
- 통신 대기 시간 영향 감소
- GPT-J 훈련에서 27% 처리량 개선[9]

***

### 10. 종합 결론

**PyTorch Distributed: Experiences on Accelerating Data Parallel Training** 논문은 분산 딥러닝 훈련의 **실제 구현과 성능 최적화의 핵심**을 제시합니다.[1]

#### 핵심 기여:

1. **기울기 버킷팅**: 작은 기울기 텐서의 통신 비효율성 해결
2. **계산-통신 겹침**: 백워드 패스 중 AllReduce 비동기 실행으로 최대 38% 속도 향상
3. **동적 그래프 대응**: 자동 그래드 그래프 순회를 통한 미사용 매개변수 안전 처리
4. **근-선형 확장성**: 256 GPU에서 적절한 구성으로 달성 가능함을 실증

#### 남은 과제:

- **이질성 환경 적응**: 기기 간 성능 편차 극복
- **극도의 통신 효율성**: Federated Learning 환경에 대응
- **초대형 모델 훈련**: 모델 병렬화와의 통합
- **일반화 성능 보증**: 플랫 최소값 추구 및 정규화 기법

#### 미래 방향:

2020년 이후의 연구는 다음 방향으로 진행되고 있습니다:

1. **적응형 최적화**: 기울기 압축, 동기화 간격, 버킷 크기 동적 조정
2. **계층별/연산별 특화**: Transformer, RNN 등 모델 유형에 맞춘 최적화
3. **이론적 보증**: 수렴 속도 및 일반화 한계에 대한 형식적 분석
4. **프레임워크 통합**: TensorFlow, JAX 등 다양한 프레임워크에 기법 확산

이 논문은 **단순한 기술 리포트를 넘어 분산 훈련의 산업 표준 확립**에 기여했으며, 이후 5년간의 활발한 연구 활동의 기초가 되었습니다.

***

### 참고 문헌 및 인용

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/cb264463-a681-4e73-8a97-90663427ed78/2006.15704v1.pdf)
[2](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460613.pdf)
[3](https://arxiv.org/abs/2507.20424)
[4](https://arxiv.org/abs/2504.18454)
[5](https://arxiv.org/html/2502.11058v1)
[6](https://arxiv.org/pdf/2103.08870.pdf)
[7](https://arxiv.org/pdf/2110.04478.pdf)
[8](https://dl.acm.org/doi/10.1145/3731599.3767571)
[9](https://ieeexplore.ieee.org/document/11044826/)
[10](https://www.openproceedings.org/2025/conf/edbt/paper-113.pdf)
[11](https://euromlsys.eu/pdf/euromlsys25-19.pdf)
[12](http://arxiv.org/pdf/2405.01067.pdf)
[13](https://arxiv.org/html/2412.19989v1)
[14](https://openreview.net/forum?id=8HuLgtjqOD)
[15](https://ieeexplore.ieee.org/document/11078491/)
[16](https://proceedings.mlr.press/v202/markov23a/markov23a.pdf)
[17](https://ieeexplore.ieee.org/document/11136599/)
[18](https://www.tandfonline.com/doi/full/10.1080/17538947.2025.2542913)
[19](https://www.semanticscholar.org/paper/0d1b320136ea6390320be2573733e87cf42e239c)
[20](https://ieeexplore.ieee.org/document/11162203/)
[21](https://etasr.com/index.php/ETASR/article/view/12485)
[22](https://arxiv.org/pdf/2211.16648.pdf)
[23](http://arxiv.org/pdf/2306.08423.pdf)
[24](http://arxiv.org/pdf/2411.05614.pdf)
[25](https://arxiv.org/pdf/1712.02679.pdf)
[26](https://www.acceldata.io/blog/how-distributed-data-parallel-transforms-deep-learning)
[27](https://theaisummer.com/distributed-training/)
[28](https://arxiv.org/pdf/2109.13049.pdf)
[29](https://arxiv.org/html/2503.23186v1)
[30](https://ieeexplore.ieee.org/document/10542408/)
[31](https://www.nature.com/articles/s41598-021-98794-z)
[32](https://github.com/Yonghongwei/Gradient-Centralization)
[33](https://im.pusan.ac.kr/bbs/cse/2614/1717491/artclView.do)
[34](https://ieeexplore.ieee.org/document/11175132/)
[35](https://etasr.com/index.php/ETASR/article/view/13201)
[36](https://dl.acm.org/doi/10.1145/3676642.3736399)
[37](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025JB031756)
[38](https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.4c01086)
[39](https://ieeexplore.ieee.org/document/11017171/)
[40](http://eartharxiv.org/repository/view/2152/)
[41](https://www.semanticscholar.org/paper/d228773902989bb9e04d36dea82acd14974715c3)
[42](https://arxiv.org/abs/2305.11665)
[43](https://arxiv.org/pdf/2111.04949.pdf)
[44](http://arxiv.org/pdf/2204.03230.pdf)
[45](https://arxiv.org/pdf/2111.05426.pdf)
[46](https://arxiv.org/pdf/2103.04303.pdf)
[47](http://arxiv.org/pdf/2406.08115.pdf)
[48](https://arxiv.org/pdf/2403.10616.pdf)
[49](https://arxiv.org/pdf/1710.05468.pdf)
[50](https://pure.kaist.ac.kr/en/publications/generalization-capability-of-deep-learning)
[51](https://yy-ko.github.io/assets/files/CIKM21-aladdin-paper.pdf)
[52](https://www.cs.purdue.edu/homes/lintan/publications/d3-tse25icse25.pdf)
[53](https://arxiv.org/pdf/2409.17836.pdf)
[54](https://pmc.ncbi.nlm.nih.gov/articles/PMC12331544/)
[55](https://3dvar.com/Chen2025Can.pdf)
[56](https://arxiv.org/pdf/2412.12156.pdf)
