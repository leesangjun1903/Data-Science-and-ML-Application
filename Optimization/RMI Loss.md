픽셀 단위의 loss와는 다르게, RMI는 하나의 픽셀을 표현할 때 다른 픽셀들도 이 픽셀을 표현하기 위해 사용한다.

그리고 이미지에서 각각의 픽셀을 위해 픽셀 간의 관계 정보를 담은 다차원 포인트를 인코딩한다.

그 이미지는 이 고차원 포인트의 다차원 분포에 캐스팅된다.

따라서 예측값과 실제값은 mutual information을 극대화함으로써 고차 일관성을 달성할 수 있다. 

게다가, MI의 실제값이 계산하기 어려움으로, MI의 하한을 계산하고 MI의 실제값을 최대화하기 위해 하한을 최대화한다.

따라서 RMI는 학습 과정에서 적은 추가적인 연산량을 요구한다 그리고 테스트 과정에서 오버헤드가 없다.

# Reference
- https://nolja98.tistory.com/313
- https://github.com/ZJULearning/RMI/blob/10a40cdbeb58bdd1bd7125fde73b48b12f9452c7/losses/rmi/rmi.py
