# Particle Swarm Optimization
입자 군집 최적화(Particle Swarm Optimization, PSO)는 새 떼나 물고기 떼와 같은 생태계의 군집 행동을 모방한 확률적 최적화 알고리즘입니다.  
1995년 James Kennedy와 Russell Eberhart에 의해 제안되었습니다. 

PSO의 핵심은 여러 개의 입자(Particle)들이 탐색 공간을 날아다니며 최적의 위치(해)를 찾는 것입니다. 각 입자는 다음 두 가지 정보를 바탕으로 이동합니다. 
- pBest (Personal Best): 자기 자신이 지금까지 경험한 가장 좋은 위치.
- gBest (Global Best): 전체 군집이 지금까지 발견한 가장 좋은 위치.

각 입자는 매 반복(Iteration)마다 자신의 속도(Velocity)와 위치(Position)를 업데이트합니다.  
- 속도 업데이트: 기존 속도에 '자기 경험(pBest 방향)'과 '전체 경험(gBest 방향)'을 더해 새로운 방향을 결정합니다.  
이때 관성(Inertia), 인지적 능력(Self-confidence), 사회적 능력(Swarm-confidence) 가중치를 조절하여 탐색 성능을 높입니다.  
- 위치 업데이트: 수정된 속도만큼 현재 위치에서 이동합니다.

## 알고리즘 프로세스 요약
- 입자들의 초기 위치와 속도를 무작위로 설정합니다.
- 각 입자의 현재 위치에서 목적 함수(Fitness Function) 값을 계산합니다.
- 현재 값이 개별 최적값(pBest)보다 좋으면 갱신합니다.
- 전체 입자 중 가장 좋은 값을 전체 최적값(gBest)으로 갱신합니다.
- 위치와 속도를 업데이트하고, 종료 조건(목표값 도달 또는 반복 횟수 초과)까지 반복합니다.

