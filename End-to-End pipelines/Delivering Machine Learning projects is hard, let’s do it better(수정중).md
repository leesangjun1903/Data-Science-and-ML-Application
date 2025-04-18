# Delivering Machine Learning projects is hard, let’s do it better

## Machine Learning Powered System
![](https://wikidocs.net/images/page/183718/A01_03_ML_pow_SYS.png)

![](https://wikidocs.net/images/page/183718/Fig_A01_03.png)

민첩한 사고의 추가 진화는 "Devops"(Ebert 2016)의 아이디어입니다.  
개발자(dev)와 소프트웨어를 운영하는 지원 팀(ops) 사이에 가교를 구축하려는 시도입니다.  
Devops 아이디어를 이끄는 통찰력은 운영 팀이 조직의 다른 어떤 부분보다 소프트웨어와 더 많이 접촉하는 전문 사용자 그룹이라는 것입니다.  
사용 중인 소프트웨어에 대한 주요 장벽은 생산 환경과 현실에 대한 개발 팀의 이해 사이의 불일치 비용입니다. 이 비용은 소프트웨어 제공이라는 목표를 달성하려는 개발팀과 무결점 비즈니스 연속성의 목표를 달성하려는 운영팀 모두에서 부담합니다.

그림 1.3은 신속한 적응형 소프트웨어 개발을 지원하는 Devops 프로젝트의 주요 활동을 보여줍니다.  
Devops 팀은 소프트웨어 개발 및 제공 프로세스를 중심으로 자동화를 개발합니다. 이를 통해 프로젝트가 성숙해짐에 따라 개발 자체에 점점 더 집중할 수 있습니다.  
프로젝트 활동으로의 정보 흐름은 나중에 개발 주기에서 소프트웨어를 변경하는 비용과 위험을 줄임으로써 촉진됩니다. 일반적으로 이것은 (프로젝트 후반에) 사용자와 이해 관계자가 실제로 무엇을 할 것인지, 어떻게 가치를 창출할 것인지를 깨닫는 때이므로 유연성을 갖는 것은 제공되는 소프트웨어의 품질에 불균형한 영향을 미칩니다.  
Devops는 소프트웨어 운영에 개발 팀을 참여시키고 운영 팀을 개발 수명 주기로 가져옵니다. Devops는 릴리스의 주기를 프로덕션으로 크게 늘리고 각 릴리스의 범위를 크게 줄임으로써 소프트웨어 프로젝트에서 위험과 비용의 균형을 변경하려고 합니다.

## Deep learning End-to-End pipelines

프로젝트 설정은 당면한 문제에 대해 가능한 한 많은 세부 사항을 파악하는 프로세스로 설명됩니다.  
이를 수행하는 방법은 기술 인터뷰의 토론 측면에서 표현되며 정보의 출처는 면접관으로 간주됩니다.  
목표, 사용자 경험, 성능 제약, 평가, 개인화 및 프로젝트 제약(사람, 컴퓨팅 성능 및 인프라)은 고려해야 할 중요한 요소로 식별됩니다.
데이터 파이프라인 요소는 개인 정보 보호 및 편향, 저장, 전처리 및 가용성을 고려합니다.  
모델링은 모델선택, 학습, 디버깅, 하이퍼파라미터 튜닝&스케일링(많은 학습데이터를 커버한다는 의미)의 관점에서 고려됩니다.  

![](https://wikidocs.net/images/page/183718/End2End_Pipelines_simplified.png)

- Simplipied

![](https://wikidocs.net/images/page/183718/End2End_Pipelines_A.png)
![](https://wikidocs.net/images/page/183718/End2End_Pipelines_B.png)

- Mid Scale

# Summary
지난 10년 동안 데이터와 컴퓨팅이 폭발적으로 증가하면서 ML은 중요한 기술이 되었습니다.  
성공적으로 ML 프로젝트를 제공하는 것과 프로젝트가 제공될 때 미칠 수 있는 부정적인 영향 측면에서 문제가 있었습니다.  
기계 학습 프로젝트는 복잡한 데이터에 의존하고 팀이 데이터에서 생성된 모델을 생성하고 관리해야 하며 사용자 및 이해 관계자의 요구 사항에 따라 신중하게 조정되어야 한다는 점에서 다릅니다.  
성공적인 ML 프로젝트는 요구 사항 및 데이터에서 위험을 제거하고 비기능적 및 기능적 요구 사항을 캡처하며 모델을 처리하고 평가하는 기능을 개발합니다.  
프로젝트는 나쁜 결과를 피하기 위해 수명 주기 전반에 걸쳐 사회 및 이해 관계자의 요구에 맞춰야 합니다.  
Agile 소프트웨어 개발 및 Devops 커뮤니티에서 아이디어를 빌려 이를 수행할 수 있습니다.
