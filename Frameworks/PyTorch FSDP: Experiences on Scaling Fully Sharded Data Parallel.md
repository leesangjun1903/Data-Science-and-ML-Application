
# PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel

## 1. 핵심 주장과 주요 기여

**"PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"** 논문은 Meta AI에서 개발한 업계 수준의 대규모 모델 훈련 솔루션을 제시합니다. FSDP의 핵심 주장은 다음과 같습니다.

**주요 기여:**

1. **기술적 민주화**: 대규모 모델 훈련 기술을 소수의 고급 사용자와 산업 리더에게만 한정되어 있던 상황을 개선하여, 더 넓은 커뮤니티가 접근 가능한 솔루션 제공

2. **비침습적 사용자 경험**: 기존 PyTorch의 분산 훈련 기능(DistributedDataParallel)과 유사한 API를 제공하면서도, 단일 GPU에 맞지 않는 대규모 모델 훈련 지원

3. **업계 수준의 구현**: Tensor 구현, 디스패처 시스템, CUDA 메모리 캐싱 할당자 등 PyTorch 핵심 컴포넌트와 긴밀하게 공동 설계

4. **확장성**: 512개의 80GB A100 GPU를 활용한 실험에서 DDP 대비 유사한 성능을 유지하면서도 훨씬 더 큰 모델(175B 파라미터) 지원

5. **선형적 확장성**: TFLOPS 기준으로 근선형(near-linear) 확장성 달성, GPU 증가에 따른 성능 저하 최소화

***

## 2. 해결하고자 하는 문제

### 2.1 기술적 문제

FSDP가 해결하는 주요 문제들은:

**메모리 한계**: 기존 분산 훈련 방식(DDP)은 모든 GPU에 모델 복제본을 유지해야 하므로, 단일 GPU 메모리 용량을 초과하는 모델을 훈련할 수 없습니다. 예를 들어, 10억 개 이상의 파라미터를 가진 모델을 40GB GPU에서 훈련하려면 메모리 부족(Out-of-Memory) 오류 발생.

**통신 오버헤드**: 모델 샤딩 시 AllGather와 ReduceScatter 같은 집단 통신(collective communication) 작업으로 인한 통신 오버헤드가 훈련 성능을 저하시킴.

**사용자 경험**: 기존 기술들은 특정 모델 아키텍처에 밀접하게 연결되어 있거나, 기계학습 프레임워크의 내부 인터페이스에 의존하여 유지보수가 어려움.

**하드웨어 이질성**: 현대적 GPU 클러스터는 호스트 내 고대역폭 네트워크와 호스트 간 저대역폭 네트워크가 혼합된 fat-tree 네트워크 토폴로지를 채택하고 있어, 이를 활용한 최적화 필요.

### 2.2 실제 적용 장애물

- 사용자가 모델을 초기화하기 전에 전체 모델을 GPU에 로드해야 함
- 초기화 로직이 비공유(unshard) 파라미터에 의존할 경우 샤딩 불가능
- 메모리 조각화(fragmentation)로 인한 성능 저하

***

## 3. 제안하는 방법론

### 3.1 FSDP 알고리즘 개요

FSDP는 Zero-Redundancy Optimizer(ZeRO)에서 영감을 받았지만, PyTorch에 맞게 개정된 설계를 제시합니다. 기본 워크플로는 다음과 같습니다.

**Forward Pass:**
1. 각 FSDP 유닛의 파라미터에 대해 AllGather 실행
2. 비공유(unsharded) 파라미터로 로컬 계산 수행
3. 계산 후 즉시 받은 샤드 메모리 해제

**Backward Pass:**
1. AllGather로 파라미터 복구
2. 자동 미분(autograd) 엔진으로 그래디언트 계산
3. ReduceScatter로 그래디언트 감소 및 샤딩

**메모리 요구사항:**
$$M_{\text{peak}} = O\left(\frac{\Psi}{F} + \max_{i=1}^{N}\psi_i\right)$$

여기서:
- $$\Psi$$: 전체 모델 파라미터 개수
- $$F$$: 샤딩 팩터(파라미터가 샤딩되는 GPU 개수)
- $$\psi_i$$: 각 FSDP 유닛의 파라미터 개수

### 3.2 핵심 기술: FlatParameter

FSDP는 각 FSDP 유닛의 모든 파라미터를 1D 텐서인 **FlatParameter**로 변환합니다.

**구성 방식:**

$$\text{FlatParameter} = \text{Concat}(p_1, p_2, \ldots, p_n) + \text{Padding}$$

여기서 패딩은 최대 $$F-1$$개 요소까지만 필요합니다.

**이점:**
- 통신 효율 최대화: NCCL 라이브러리의 AllGather Base는 짝수 입력 크기 요구
- 불필요한 복사 제거: 입출력 텐서가 정확한 데이터 레이아웃 유지
- 일반성: 원본 파라미터의 임의의 형태 지원

**예시 (16개 GPU에서 4×3 Linear 레이어):**
- 각 GPU는 FlatParameter에서 1개 요소만 보유
- 마지막 GPU는 패딩된 값 유지

### 3.3 샤딩 전략

FSDP는 **샤딩 팩터** $$F$$를 통해 세 가지 전략 제공:

#### 3.3.1 완전 샤딩(Full Sharding, F = W)
- 각 GPU가 전체 모델의 1/W만 보유
- 메모리 사용량 최소: $$O(\Psi/F + \max_i\psi_i)$$
- 통신 오버헤드 최대: 1.5배(DDP 대비)

$$\text{Cross-host traffic} = \frac{3M}{W} - \frac{1}{W}$$

여기서 $$M$$은 모델 크기입니다.

#### 3.3.2 하이브리드 샤딩(Hybrid Sharding, 1 < F < W)

그래디언트 감소를 분해:

$$\sum_{r=1}^{W} g_r = \sum_{i=1}^{W/F} \sum_{r \in S_i} g_r$$

여기서:
- $$S_i$$: 샤딩 그룹들
- $$R_j$$: 복제 그룹들

**Cross-host traffic:**

$$\text{Traffic} = \frac{2M}{W} \cdot \frac{W-1}{G}$$

($$G$$: 호스트당 GPU 개수)

이는 완전 복제 $$2M(W-1)/W$$와 완전 샤딩 $$3M(W-1)/W$$의 중간값.

#### 3.3.3 완전 복제(Full Replication, F = 1)
- 표준 DDP처럼 작동
- 모든 GPU에서 모든 파라미터 복제

### 3.4 지연 초기화(Deferred Initialization)

대규모 모델 초기화 문제 해결:

1. **"더미(Fake)" 디바이스에 모델 생성**: 실제 메모리 할당 없이 모델 구조 정의
2. **초기화 작업 기록**: 텐서 초기화 중 수행된 모든 작업 기록
3. **유닛별 재생**: GPU 디바이스로 이동 시 기록된 작업 재생

이 방법으로 사용자는 수정 없이 제3자 라이브러리의 모델 사용 가능.

### 3.5 통신 최적화 기법

#### 3.5.1 통신과 계산 겹침(Overlapping)

전형적인 async 패턴과 달리, FSDP는 별도의 CUDA 스트림 사용:

```python
# 기본 스트림(계산)과 독립적으로 
# NCCL 스트림(통신)에서 AllGather 실행
ProcessGroupNCCL_stream.all_gather(...)
```

**타이밍 도표:**
- CPU 스레드가 다음 계산을 예측하고 AllGather 발행
- GPU 계산 완료 후 AllGather 시작
- AllGather와 다음 계산 겹침(Overlap)

#### 3.5.2 역방향 사전 페치(Backward Prefetching)

역방향 패스에서 ReduceScatter와 AllGather가 순차 실행되므로:

$$\text{ReduceScatter}_{i} \to \text{AllGather}_{i+1}$$

이를 개선하기 위해 AllGather를 먼저 발행:

$$\text{ReduceScatter}_{i} \parallel \text{AllGather}_{i+1}$$

**구현**: 정방향 패스의 모듈 실행 순서를 역방향 실행 순서의 프록시로 기록.

#### 3.5.3 정방향 사전 페치(Forward Prefetching)

정적 계산 그래프의 경우, 이전 반복의 정방향 순서로 다음 AllGather 사전 발행.

#### 3.5.4 그래디언트 누적

통신 포함/미포함 두 가지 변형:
- **통신 포함**: 모든 반복에서 그래디언트 감소(표준)
- **통신 미포함**: 특정 반복에서만 감소 → 메모리 증가, 통신 감소

### 3.6 메모리 관리: Rate Limiter

PyTorch의 CUDA 캐싱 할당자 문제:
- CPU 스레드가 GPU 실행보다 빨리 진행할 때, AllGather 출력 텐서가 할당될 수 없음
- 캐싱 할당자는 블록 재사용 불가능 → cudaMalloc retry 강제

**해결책**: Rate Limiter로 CPU 스레드 차단, 최대 2개 inflight AllGather 유지.

$$\text{AllGather}_{\text{max}} = 2$$

이는 겹침을 유지하면서 메모리 재사용 보장.

### 3.7 혼합 정밀도(Mixed Precision) 최적화

표준 혼합 정밀도는 메모리 오버헤드 증가:
$$K_{\text{full}}\Psi \to (K_{\text{low}} + K_{\text{full}})\Psi$$

FSDP는 비공유 파라미터만 저정밀도 사용:

$$M_{\text{peak}}' = \frac{K_{\text{full}}}{F}\sum_i\psi_i + K_{\text{low}}\max_i\psi_i$$

기존 대비 감소량:
$$\Delta M = K_{\text{full}}\max_i\psi_i - K_{\text{low}}\max_i\psi_i = (K_{\text{full}} - K_{\text{low}})\max_i\psi_i$$

***

## 4. 모델 구조 및 구현

### 4.1 PyTorch Autograd 통합

FSDP는 PyTorch의 autograd 엔진을 직접 활용:

1. **정방향 패스 전**: 원본 파라미터를 비공유 FlatParameter의 뷰로 설정
   ```python
   params = torch.split(flat_param_unsharded, ...)
   ```

2. **역방향 패스**: Autograd가 자동으로 FlatParameter 그래디언트 계산
   
3. **그래디언트 후킹**: FlatParameter의 AccumulateGrad 함수에 후킹 등록
   - 그래디언트 누적 완료 시 ReduceScatter 즉시 실행
   - torch.Tensor.register_hook()을 통해 정확한 시점 제어

### 4.2 모듈 래핑 API

두 가지 방식 지원:

**방식 1: FullyShardedDataParallel 래퍼**
```python
model = FullyShardedDataParallel(model, auto_wrap_policy=...)
```
- 하위 모듈을 FSDP 유닛으로 교체
- 모델 구조 변경

**방식 2: fully_shard 함수 어노테이터**
```python
fully_shard(model)
```
- nn.Module의 전후 후킹을 통한 FSDP 로직 설치
- 모델 구조 보존, 파라미터 정규화 이름 유지

### 4.3 FlatParameter 관리

**FlatParamHandle 클래스**:
- 개별 FlatParameter 관리 담당
- AllGather/ReduceScatter 시점 결정
- 메모리 할당/해제 조율

**FSDP 유닛 경계**:
- 정적 nn.Module 구조 활용
- 모듈 주석(annotation) 기반 구성
- 또는 동적 실행 순서 감지를 통한 재구성

### 4.4 세 가지 초기화 옵션

**옵션 1: 지연 초기화 (권장)**
- 메모리 효율 최대
- 모델 코드 수정 불필요
- 동적 특성 지원(반복 간 모듈 실행 순서 변화)

**옵션 2: GPU에서 비공유 초기화**
- 조건: 초기화는 훈련보다 메모리 요구량 적음
- 제한: 모델이 단일 GPU에 맞아야 함

**옵션 3: CPU에서 비공유 초기화**
- 가장 큰 모델 지원
- 단점: CPU 메모리 대역폭 제한으로 속도 저하

***

## 5. 성능 향상

### 5.1 모델 크기별 성능

| 모델 크기 | DDP(TFLOPS/GPU) | FSDP-완전 샤딩(TFLOPS/GPU) | FSDP-하이브리드(TFLOPS/GPU) | 메모리 상태 |
|-----------|------------------|------------------------|------------------------|-----------|
| 611M | 15.18 | 15.28 | 14.61 | ✓ 모두 가능 |
| 2.28B | 27.40 | 27.70 | 25.76 | ✓ 모두 가능 |
| 11.3B | OOM | 148.48 | 145.81 | ✓ FSDP만 |

DDP는 2.28B를 초과하면 메모리 부족 발생.[1]

### 5.2 GPU 확장성 (GPT-175B, BF16 활성화)

| GPU 수 | Batch 1 (TFLOPS/GPU) | Batch 2 (TFLOPS/GPU) |
|-------|---------------------|---------------------|
| 128 | 173 | 186 |
| 256 | 172 | 185 |
| 512 | 171 | 184 |

**근선형 확장**: 128-512 GPU 범위에서 ~1% 성능 저하만 관찰.[1]

### 5.3 역방향 사전 페치 효과

GPT-175B에서:
- 사전 페치 미적용: 기본 성능
- 사전 페치 적용: **~18% 속도 향상**[1]

### 5.4 Rate Limiter의 효과

메모리 재조각화(malloc retry) 가능성에 따라 변동:

| 모델 | 머신 수 | Rate Limit 미적용 | Rate Limit 적용 | 개선도 |
|------|--------|-----------------|----------------|-------|
| RegNet-9B | 2 | 14.81s | 14.80s | 0% (불필요) |
| RegNet-9B | 4 | 21.70s | 21.81s | -0.5% (해로움) |
| T5-11B | 2 | 8.36s | 5.02s | **40% ↑** |
| T5-11B | 4 | 5.02s | 4.23s | **16% ↑** |

Rate Limiter 활용 여부는 CUDA malloc retry 통계로 판단.[1]

### 5.5 대규모 모델 훈련

**DHEN 추천 모델 (768B sparse + 550M dense)**:
- 32-512 GPU 범위에서 테스트
- 완전 샤딩 (RAF): 메모리 최소, QPS 저하
- 하이브리드 샤딩 (NRAF): 중간 메모리, 높은 QPS
- GPU 증가 시 피크 메모리 일관되게 감소[1]

**T5-11B (512 GPU)**:
- 메모리 용량 이하에서 안정적 작동
- 메모리 조각화 위험 낮음
- GPU 증가에 따라 7% TFLOPS 저하 (통신 지배적)[1]

***

## 6. 모델 일반화 성능 향상 가능성

### 6.1 배치 크기와 일반화의 트레이드오프

**핵심 문제**: 대규모 분산 훈련은 데이터 병렬로 인해 큰 배치 크기 필요
$$B_{\text{global}} = B_{\text{local}} \times N_{\text{GPU}}$$

**일반화 갭(Generalization Gap)**: 큰 배치는 그래디언트 노이즈 감소 → 수렴하는 해의 예각성(sharpness) 증가 → 테스트 성능 저하.[2]

### 6.2 FSDP가 제공하는 개선 기회

#### 6.2.1 유연한 배치 크기 스케줄

적응형 배치 크기 스케줄(DDP-Norm)을 FSDP와 결합:
- 소배치로 시작: 좋은 일반화
- 점진적 증가: 훈련 효율성 개선
- 결과: 작은 배치의 일반화 + 큰 배치의 효율성 결합[2]

**수학적 표현**:
$$B_k = B_0 \cdot e^{\eta \cdot k}$$

여기서 $$\eta$$는 배치 크기 증가율.

#### 6.2.2 통신 오버헤드 감소가 미치는 영향

FSDP의 효율성 개선(통신 겹침, 사전 페치)은:
1. 더 작은 로컬 배치로도 충분한 처리량 가능
2. 결과적으로 더 작은 전역 배치 사용 가능
3. 일반화 성능 개선

#### 6.2.3 혼합 정밀도와 정규화

FSDP의 네이티브 혼합 정밀도:
- BF16 연산으로 계산 가속
- FP32 마스터 사본 유지로 수렴성 보장
- 그래디언트 스케일링 안정성 확보[1]

### 6.3 제한사항 및 고려사항

#### 6.3.1 수학적 동치성(Mathematical Equivalence) 부족

**문제**: FSDP는 샤딩된 파라미터에 대해 최적화 수행
- Adam 같은 적응형 옵티마이저의 벡터 노름 계산이 영향 받음
- 근사 2차 옵티마이저(approximate second-order optimizers)와 호환성 문제
- 글로벌 상태 의존 최적화 불가능[1]

**영향**: 일부 최적화 기술이 정확히 동작하지 않을 수 있음

#### 6.3.2 공유 파라미터 처리

**문제**: 여러 모듈에서 사용되는 공유 파라미터
- 각 사용처에서 재비공유(unsharded) 필요
- 복잡한 모듈 구조에서 처리 어려움
- 임시 메모리 증가

**권장**: 공유 파라미터를 최하위 공통 조상 FSDP 유닛에 배치

#### 6.3.3 동적 모델 호환성

**개선점**: FSDP는 동적 구조 지원
- 반복 간 모듈 실행 순서 변화 감지
- 사전 페치 호환성 유지

하지만 급격한 변화는 성능 저하 가능

### 6.4 실증적 증거

**IBM 연구**: PyTorch FSDP를 사용한 7B 모델 훈련[3]
- 선택적 활성화 체크포인팅: 10% 처리량 향상
- 90% 이상의 계산-통신 겹침 달성
- 메모리 효율과 성능의 균형 유지

**Meta 실험**: T5-11B에서 배치 크기 변화에 따른 동작[1]
- B=8: 안정적 수렴, 7% TFLOPS 저하는 순수 통신 병목
- B=16: 높은 처리량, 수렴 안정성 유지
- 메모리 압박 없이 배치 크기 선택 가능 → 일반화 최적화 여지

***

## 7. 한계와 제약사항

### 7.1 통신 병목 현상

**약한 스케일링(Weak Scaling)**: GPU 수 증가, 배치 크기 비례 증가

FSDP에서는 AllGather와 ReduceScatter의 통신량이 GPU 수에 비례:

$$\text{Comm}_{\text{total}} = O(M \cdot W)$$

반면 계산량은:
$$\text{Compute}_{\text{total}} = O(M \cdot W)$$

따라서 노드 간 통신이 지배적이면 통신 오버헤드 증가.

**512 GPU에서 T5-11B 관찰**: 8 GPU 대비 7% 성능 저하[1]

### 7.2 메모리 단편화

**문제**: 여러 CUDA 스트림 사용 시 캐싱 할당자 효율성 저하

**영향**: 특히 GPU 메모리 거의 가득 참 상태에서 악화

**예시**: GPT-175B, 128 GPU, B=1
- 백워드 패스가 전체 반복 시간의 85.56% 차지
- 정상적으로는 67% 정도
- 메모리 조각화로 인한 성능 저하[1]

### 7.3 벤더 의존성

모듈 래핑 및 autograd 후킹은 PyTorch 구현 세부사항에 의존:
- PyTorch 업데이트로 인한 호환성 깨짐 위험
- 다른 프레임워크로 직접 이식 불가능

### 7.4 하드웨어 요구사항

**최적 조건**:
- 고대역폭 클러스터: ≥200 Gbps 노드 간 네트워크
- 충분한 GPU 메모리: 모든 FSDP 유닛의 비공유 형태 할당 가능

**열악한 환경**: 저대역폭 클러스터에서는 통신 오버헤드로 TFLOPS 심각 저하

### 7.5 최적화 문제

**Optimizer State Partition의 한계**:
- 벡터 노름 기반 적응형 최적화와 호환성 문제
- 2차 모멘트 기반 최적화에서 정확성 손상 가능

**현재 상태**: Adam 같은 1차 적응형 옵티마이저는 비교적 안정적, 복잡한 최적화는 신중한 테스트 필요

***

## 8. 2020년 이후 관련 최신 연구 현황

### 8.1 FSDP 후속 기술

#### SimpleFSDP (2024)

**혁신**: torch.compile을 활용한 FSDP 단순화[4]

**개선 사항**:
- 구현 단순화: 기존 FSDP 복잡도 감소
- 컴파일러 기반 최적화: TorchInductor 백엔드에서 IR 노드 버킷팅 및 재순서화
- 계산-통신 중첩 향상: torch.compile의 전체 그래프 추적으로 최적화 기회 발굴

**주요 기술**:
- 매개변수화(Parameterizations) 활용
- 선택적 활성화 체크포인팅
- DTensor를 통한 분산 텐서 표현

**성능**: Llama 3 모델에서 기존 FSDP 대비 개선된 확장성

#### Zero++ (최신)

**목표**: 저대역폭 클러스터에서 ZeRO 한계 극복[5]

**핵심 기술**:
- 저정밀도 AllGather (8/6 비트 사전학습)
- 데이터 리매핑(Data Remapping)
- 저정밀도 그래디언트 평균화 (4/2 비트 미세조정)

**성능**: 
- 통신량 4배 감소
- 384 GPU 규모에서 2.16배 처리량 향상
- RLHF 훈련 3.3배 가속[5]

**정확성**: 13B 모델 사전학습, 30B 미세조정에서 기존 ZeRO와 동등 정확도[5]

### 8.2 메모리 최적화 연구

#### Mist (2025)

**포커스**: 메모리-병렬성 공동 최적화[6]

**approach**: 다양한 병렬화 기법(데이터, 텐서, 파이프라인)과 메모리 최적화(활성화 체크포인팅, 중복 제거, 오프로딩) 조합 최적화

#### COMET (2024)

**방법론**: 클러스터-워크로드 공동 설계[7]

**기여**:
- 분산 훈련 전체 스택 분석
- 병렬화 전략과 클러스터 자원의 영향 평가
- 설계 공간 탐색 도구 제공

### 8.3 일반화 성능 연구

#### 배치 크기의 일반화 영향 (2025)

**핵심 발견**:[2]
- 큰 배치 훈련은 일반화 성능 저하 (일반화 갭)
- 동적 배치 크기 스케줄로 개선: 소배치의 일반화 + 대배치의 효율성

**공식**:
$$L_{\text{gen}}(B) = \alpha \cdot \ln(B) + \beta$$

적응형 배치 스케줄로 최적점 탐색 가능.

#### 효율성 스펙트럼 (2025)

**포괄적 분석**: 대규모 언어 모델 효율성의 다중 측면[8]

- **예산 효율(Budget Efficiency)**: 스케일링 법칙
- **데이터 효율(Data Efficiency)**: 필터링, 활동적 학습, 커리큘럼 학습
- **아키텍처 효율**: 효율적 어텐션, 희소 모델링
- **훈련 효율**: 혼합 정밀도, 병렬화, 메모리 최적화
- **추론 효율**: 가지치기, 지식 증류, 양자화

### 8.4 통신 효율 최근 연구

#### 양자화 분산 훈련 (2023)

**도전**: FSDP에 직접 압축 기법 적용 어려움 (가중치 통신은 모델 일관성 영향)[9]

**해결책**: FSDP 특화 양자화 기법 개발 필요

#### 대역폭 제한 환경 (2025)

**발견**: 대역폭이 메모리보다 더 제한적 요인[10]

$$\text{Throughput} = \min\left(\frac{\text{Compute}}{\text{필요 연산}}, \frac{\text{Bandwidth}}{\text{통신량}}\right)$$

고대역폭(≥200 Gbps) 네트워크 없으면 통신이 병목.

### 8.5 통합 시스템 연구

#### Colossal-AI (2023)

**통합 인터페이스 제공**[11]
- 데이터 병렬화 (DDP, FSDP)
- 파이프라인 병렬화
- 텐서 병렬화
- 시퀀스 병렬화
- ZeRO와 통합

**성능**: 기본 시스템 대비 2.76배 훈련 가속

#### DistTrain (2024)

**포커스**: 멀티모달 LLM의 이질성 처리[12]

- 모델 이질성: 다양한 모달리티의 서로 다른 구조
- 데이터 이질성: 모달 간 데이터 양 불균형

### 8.6 연구 트렌드 분석

**2020-2022**: FSDP/ZeRO 기본 알고리즘 정립
- 메모리 효율성 핵심 포커스
- 100B 모델 훈련 가능성 입증

**2023-2024**: 실무 최적화 및 일반화
- 통신 최적화 심화 (SimpleFSDP, ZeRO++)
- 배치 크기 최적화와 일반화 연관성 규명
- 멀티모달 및 이질 환경 확장

**2025 현재**: AI 시스템의 총체적 효율성
- 메모리-대역폭-연산의 균형 분석
- 동적 최적화 및 적응형 스케줄
- 저정밀도 및 양자화 통합

***

## 9. 향후 연구 시 고려사항

### 9.1 학술적 개선 방향

#### 9.1.1 수학적 동치성 보장

**현재 문제**: 샤딩된 파라미터로 인한 최적화 계산 부정확성

**연구 방향**:
- 샤딩 인식(Sharding-aware) 옵티마이저 설계
- 글로벌 상태 의존 최적화의 분산 버전 개발
- 2차 모멘트 기반 방법의 정확한 구현

**목표**: 단일 GPU 훈련과 완전히 동등한 수렴 특성

#### 9.1.2 공유 파라미터 자동 처리

**현재 제약**: 사용자가 공유 파라미터 FSDP 유닛 경계 수동 설정

**개선 기회**:
- 공유 파라미터 자동 감지
- 최적 경계 자동 결정 알고리즘
- 임시 메모리 오버헤드 최소화

#### 9.1.3 일반화 성능 이론

**연구 주제**:
$$\text{Generalization Error} = f(\text{Batch Size}, \text{Learning Rate}, \text{Parallelism Strategy})$$

**목표**: 이 함수의 명시적 형태 규명, 최적 하이퍼파라미터 도출

#### 9.1.4 혼합 병렬화 최적화

현재는 FSDP + 텐서 병렬화를 2D 메시에서 조합:
- 3D/4D 병렬화의 자동 최적화
- 메시 토폴로지와 모델 구조의 자동 매칭
- 동적 병렬화 전략 전환

### 9.2 시스템 최적화

#### 9.2.1 네트워크 인식 최적화

**한계**: 현재 FSDP는 정적 토폴로지 가정

**개선**:
- 동적 네트워크 토폴로지 감지
- 실시간 대역폭 모니터링
- 통신 전략 자동 조정

#### 9.2.2 저대역폭 환경 최적화

**필요성**: 엣지 학습, 지리적 분산 데이터센터

**기술**:
- 더 공격적인 양자화 (2-4비트)
- 그래디언트 압축 기법 통합
- 비동기 업데이트 방식 재검토

#### 9.2.3 메모리 할당자 개선

**문제**: CUDA 캐싱 할당자의 멀티스트림 비효율

**방안**:
- 할당자 내 스트림별 분리 완화
- FSDP 특화 할당자 개발
- 메모리 사전 할당 및 풀 관리

#### 9.2.4 CPU 오프로딩 고급화

**현재 한계**: CPU 오프로딩은 CPU 대역폭 병목

**개선 방향**:
- 비동기 CPU ↔ GPU 전송
- 압축 통신
- 다중 계층 메모리 활용

### 9.3 응용 연구

#### 9.3.1 특수 아키텍처 최적화

**영역별 특화**:
- **Transformer**: 어텐션 메커니즘 특화
- **CNN**: 합성곱 레이어 최적화
- **GNN**: 그래프 구조 활용
- **하이브리드 모델**: 서로 다른 모달리티 처리

#### 9.3.2 미세조정(Fine-tuning) 최적화

**연구 가능성**:
- 사전학습 모델의 FSDP 미세조정 최적화
- LoRA, QLoRA 등 PEFT와 FSDP 결합
- 불균형 배치와 일반화

#### 9.3.3 온디바이스 학습

**새로운 영역**: 엣지 디바이스, 연합학습, 프라이버시 보존 학습

**도전**:
- 극도로 제한된 메모리/대역폭
- 이질적 하드웨어
- 점증적 학습

### 9.4 벤치마킹 및 평가

#### 9.4.1 종합 평가 프레임워크

**필요성**: FSDP, DeepSpeed ZeRO, Megatron-LM 등 공정한 비교

**요소**:
- 모델 크기, 배치 크기 변화에 따른 성능
- 일반화 성능 (검증 손실)
- 메모리 사용 패턴
- 전력 소비

#### 9.4.2 장기 훈련 안정성

**중요성**: 수주 단위의 대규모 훈련에서 나타나는 문제

- 메모리 누수 감지
- 수렴성 모니터링
- 자동 복구 메커니즘

#### 9.4.3 실제 응용 사례 축적

**필요**: 다양한 모델, 데이터셋, 클러스터 환경에서의 실측 데이터

### 9.5 이론적 진전

#### 9.5.1 확장성 이론

**기존 한계**: 근선형 확장성의 이론적 한계

**연구 방향**:
$$\text{MFU}(P, N) = \frac{\text{Actual Throughput}}{\text{Peak FP32 Throughput}}$$

이 함수의 상한(upper bound) 규명.

#### 9.5.2 통신-계산 트레이드오프

**수학적 모델**:
$$\text{Total Time} = T_{\text{compute}} + T_{\text{comm}} - T_{\text{overlap}}$$

이를 최소화하는 최적 샤딩 팩터, 배치 크기, 통신 패턴 규명.

#### 9.5.3 대역폭 한계의 근본 분석

**문제**: 현재 대역폭이 성능의 주요 제약

**분석**:
$$\text{Bandwidth Requirement} = \frac{3M \cdot \text{Token}}{T_{\text{target}}}$$

대규모 모델의 필요 대역폭이 기술 한계를 넘는 시점 예측.

***

## 10. 결론

PyTorch FSDP는 대규모 모델 훈련을 민주화하면서 산업 수준의 확장성을 제공하는 근본적 기술입니다. **핵심 성과**는 단순한 API와 강력한 성능의 조화입니다.

**2025년 현황**: FSDP는 진화하는 생태계의 중심
- 컴파일러 기반 단순화(SimpleFSDP)
- 통신 효율 혁신(ZeRO++)
- 동적 최적화 및 적응형 전략

**향후 10년의 과제**:
1. **수학적 엄밀성**: 샤딩된 최적화의 정확한 이론화
2. **대역폭 혁신**: 통신 효율의 근본적 개선
3. **자동화**: 사용자 개입 최소화로 접근성 극대화
4. **일반화**: 단일 기술에서 범용 분산 학습 플랫폼으로

FSDP는 단순한 구현 기술이 아니라, **분산 학습의 미래를 규정하는 사상**으로서 AI 시스템의 진화를 주도할 것으로 예상됩니다.

***

## 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/766ef900-4405-4807-a6af-68cd33d15a18/2304.11277v2.pdf)
[2](https://arxiv.org/html/2412.21124v1)
[3](https://research.ibm.com/blog/pytorch-fsdp)
[4](https://arxiv.org/html/2411.00284v2)
[5](https://openreview.net/pdf?id=gx2BT0a9MQ)
[6](http://arxiv.org/pdf/2503.19050.pdf)
[7](https://arxiv.org/pdf/2211.16648.pdf)
[8](https://arxiv.org/html/2312.00678v2)
[9](https://arxiv.org/pdf/2302.02390.pdf)
[10](https://arxiv.org/pdf/2504.03655.pdf)
[11](https://arxiv.org/pdf/2110.14883.pdf)
[12](https://arxiv.org/pdf/2408.04275.pdf)
[13](https://arxiv.org/html/2412.07210v2)
[14](https://arxiv.org/pdf/2207.11912.pdf)
[15](https://www.markiiisys.com/blog/ml-dl-model-multi-node-distributed-training-strategies-primer/)
[16](https://www.computeontario.ca/training-colloquia/too-big-to-train-large-model-training-in-pytorch-with-fully-sharded-data-parallel)
[17](https://aclanthology.org/2025.acl-long.1493.pdf)
[18](https://www.edge-ai-vision.com/2024/05/fully-sharded-data-parallelism-fsdp/)
[19](https://www.vldb.org/pvldb/vol16/p3848-huang.pdf)
[20](https://www.emergentmind.com/topics/fully-sharded-data-parallel-fsdp)
[21](https://epoch.ai/blog/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data)
[22](https://arxiv.org/html/2502.11058v1)
[23](https://arxiv.org/pdf/2407.20018.pdf)
[24](https://arxiv.org/html/2503.00813v1)
[25](https://arxiv.org/pdf/2301.02691.pdf)
[26](https://arxiv.org/html/2405.17267)
[27](https://arxiv.org/pdf/2311.11822.pdf)
[28](https://research-explorer.ista.ac.at/download/17490/17492/Thesis_final_version_pdfa2.pdf)
[29](https://aclanthology.org/2024.findings-emnlp.686.pdf)
[30](https://www.deepspeed.ai/tutorials/zero/)
[31](https://openreview.net/pdf?id=p7jQEf3wlh)
[32](https://openreview.net/forum?id=PJNhZoCjLh)
[33](https://disml2024.github.io/disml-workshop-2024/assets/8_945276_86359389_Group8_DISML_Project_Report.pdf)
