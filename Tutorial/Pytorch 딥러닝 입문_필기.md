# 데이터의 분리(Splitting Data)

- from sklearn.model_selection import train_test_split
- X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 파이토치 패키지의 기본 구성
- torch.autograd
- torch.nn
- torch.optim
- torch.utils.data
- torch.onnx : ONNX(Open Neural Network Exchange)의 포맷으로 모델을 익스포트(export)할 때 사용합니다. ONNX는 서로 다른 딥 러닝 프레임워크 간에 모델을 공유할 때 사용하는 포맷입니다.

# 선형회귀, 자동미분
- optimizer = optim.SGD([W, b], lr=0.01)
- optimizer.zero_grad() : 미분을 통해 얻은 기울기를 이전에 계산된 기울기 값에 누적 방지
- cost.backward() : 해당 수식의 대한 기울기를 계산
- optimizer.step()

# nn.Module
파이토치에서는 선형 회귀 모델이 nn.Linear()라는 함수로, 또 평균 제곱오차가 nn.functional.mse_loss()라는 함수로 구현되어져 있습니다.
- torch.optim.SGD(model.parameters(), lr=0.01)

```
if epoch % 100 == 0:
    - 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
```

- 선형회귀 모델
```
class LinearRegressionModel(nn.Module): # torch.nn.Module을 상속받는 파이썬 클래스
    def __init__(self): #
        super().__init__()
        self.linear = nn.Linear(1, 1) # 단순 선형 회귀이므로 input_dim=1, output_dim=1.

    def forward(self, x):
        return self.linear(x)
```

- 다중 선형회귀
```
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1) # 다중 선형 회귀이므로 input_dim=3, output_dim=1.

    def forward(self, x):
        return self.linear(x)

model = MultivariateLinearRegressionModel()
```

# 미니 배치와 데이터 로더(Mini Batch and DataLoader)

# nn.Module과 클래스로 구현하는 로지스틱 회귀

# 
