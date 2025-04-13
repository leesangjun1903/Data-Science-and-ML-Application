# Supervised learning
Let’s start by talking about a few examples of supervised learning prob- lems. Suppose we have a dataset giving the living areas and prices of 47 houses from Portland, Oregon:

먼저 지도 학습 문제의 몇 가지 예에 대해 이야기해 보겠습니다. 오리건주 포틀랜드에 있는 47채의 주택 거주 지역과 가격을 제공하는 데이터 세트가 있다고 가정해 보겠습니다:

We can plot this data:

이 데이터를 플롯할 수 있습니다:

Given data like this, how can we learn to predict the prices of other houses in Portland, as a function of the size of their living areas? To establish notation for future use, we’ll use x(i) to denote the “input” variables (living area in this example), also called input features, and y(i) to denote the “output” or target variable that we are trying to predict (price). A pair (x(i),y(i)) is called a training example, and the dataset that we’ll be using to learn "a list of n training examples {(x(i),y(i)); i= 1,...,n}" is called a training set. Note that the superscript “(i)” in the notation is simply an index into the training set, and has nothing to do with exponentiation. We will also use Xdenote the space of input values, and Y the space of output values. In this example, X= Y= R.

이와 같은 데이터가 주어졌을 때, 포틀랜드의 다른 주택들의 주거 지역 크기에 따라 가격을 예측하는 방법을 어떻게 배울 수 있을까요?  
향후 사용을 위한 표기법을 확립하기 위해 입력 특징이라고도 불리는 "입력" 변수(이 예제에서는 생활 지역)를 나타내는 x(i)와 예측하려는 "출력" 또는 목표 변수(가격)를 나타내는 y(i)를 사용할 것입니다.  
한 쌍의 (x(i),y(i)를 훈련 예제라고 하며, "n개의 훈련 예제 목록 $x^(i),y^(i)$ i= 1,...,n"을 학습하는 데 사용할 데이터셋을 훈련 집합이라고 합니다.  
표기법의 위첨자 "(i)"는 단순히 훈련 집합의 인덱스일 뿐이며, 지수화와는 아무런 관련이 없습니다.  
또한 입력 값의 공간을 X로 표기하고 출력 값의 공간을 Y로 표기할 것입니다. 이 예제에서는 $X= Y= R$로 표기합니다.

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function h: X→Yso that h(x) is a “good” predictor for the corresponding value of y. For historical reasons, this function h is called a hypothesis. Seen pictorially, the process is therefore like this:

지도 학습 문제를 조금 더 형식적으로 설명하자면, 우리의 목표는 훈련 세트가 주어졌을 때 함수 h: X→Y를 학습하여 h(x)가 해당 y 값에 대한 "좋은" 예측 변수가 되도록 하는 것입니다.  
역사적인 이유로, 이 함수 h는 가설이라고 불립니다. 따라서 그림으로 볼 때, 이 과정은 다음과 같습니다:

<img width="405" alt="스크린샷 2025-04-13 오후 9 57 33" src="https://github.com/user-attachments/assets/8bc24b6a-0c7d-4287-8e95-e943b9f7727a" />

When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.

우리가 예측하려는 목표 변수가 연속적인 경우, 예를 들어 주택 예시와 같이 학습 문제를 회귀 문제라고 부릅니다.  
y가 소수의 이산 값(예: 거주 지역이 주어졌을 때 주택이 주택인지 아파트인지 예측하고자 했을 때)만을 가질 수 있는 경우를 분류 문제라고 합니다.

# Linear regression
To make our housing example more interesting, let’s consider a slightly richer dataset in which we also know the number of bedrooms in each house:

주택 사례를 더 흥미롭게 만들기 위해 각 집의 침실 수를 알고 있는 약간 더 풍부한 데이터 세트를 고려해 보겠습니다:

Here, the x’s are two-dimensional vectors in R2. For instance, x(i)_1 is the living area of the i-th house in the training set, and x(i)_2 is its number of bedrooms. (In general, when designing a learning problem, it will be up to you to decide what features to choose, so if you are out in Portland gathering housing data, you might also decide to include other features such as whether each house has a fireplace, the number of bathrooms, and so on. We’ll say more about feature selection later, but for now let’s take the features as given.)

여기서 x는 R2의 2차원 벡터입니다. 예를 들어, $x^{(i)}_1$은 훈련 세트에서 i번째 집의 거실 면적이고, $x^{(i)}_2$는 침실의 수입니다.  
(일반적으로 학습 문제를 설계할 때 어떤 특징을 선택할지는 여러분의 결정에 달려 있으므로, 주택 데이터를 수집하는 포틀랜드에 있는 경우 각 집에 벽난로가 있는지, 욕실 수 등과 같은 다른 특징도 포함하기로 결정할 수 있습니다. 특징 선택에 대해서는 나중에 더 말씀드리겠지만, 지금은 주어진 특징을 고려해 보겠습니다.)

To perform supervised learning, we must decide how we’re going to rep- resent functions/hypotheses h in a computer. As an initial choice, let’s say we decide to approximate y as a linear function of x:

지도 학습을 수행하려면 컴퓨터에서 함수/가설 h를 어떻게 표현할지 결정해야 합니다. 초기 선택으로 y를 x의 선형 함수로 근사하기로 결정한다고 가정해 보겠습니다:

```math
h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2
```

Here, the θi’s are the parameters (also called weights) parameterizing the space of linear functions mapping from X to Y. When there is no risk of confusion, we will drop the θ subscript in hθ(x), and write it more simply as h(x). To simplify our notation, we also introduce the convention of letting x0 = 1 (this is the intercept term), so that

여기서 θ i는 X에서 Y로 매핑되는 선형 함수의 공간을 매개변수화하는 매개변수(가중치라고도 함)입니다.  
혼동의 위험이 없을 때는 $h_θ(x)$에 θ 첨자를 생략하고, 이를 $h(x)$로 더 간단하게 작성합니다.  
표기법을 단순화하기 위해 $x_0 = 1$ (이것이 절편 항)을 허용하는 관례도 도입합니다.

```math
h(x) = \sum^d_{i=1} \theta_ix_i = \theta^Tx
```

where on the right-hand side above we are viewing θ and x both as vectors, and here d is the number of input variables (not counting x0). Now, given a training set, how do we pick, or learn, the parameters θ? One reasonable method seems to be to make h(x) close to y, at least for the training examples we have. To formalize this, we will define a function that measures, for each value of the θ’s, how close the h(x(i))’s are to the corresponding y(i)’s. We define the cost function:

위 오른쪽에서는 θ와 x를 모두 벡터로 보고 있으며, 여기서 d는 입력 변수의 수(x0을 세지 않음)입니다.  
이제 주어진 훈련 집합에서 매개변수 θ를 어떻게 선택하거나 학습할 수 있을까요? 한 가지 합리적인 방법은 적어도 훈련 예제에 대해 h(x)를 y에 가깝게 만드는 것입니다.  
이를 공식화하기 위해 θ의 각 값에 대해 h(x(i))가 해당 y(i)에 얼마나 가까운지 측정하는 함수를 정의할 것입니다. 우리는 비용 함수를 정의합니다:  

```math
J(\theta) = \frac{1}{2}\sum^n_{i=1} (h_\theta(x^(i))-y^(i))^2
```
If you’ve seen linear regression before, you may recognize this as the familiar least-squares cost function that gives rise to the ordinary least squares regression model. Whether or not you have seen it previously, let’s keep going, and we’ll eventually show this to be a special case of a much broader family of algorithms.

이전에 선형 회귀를 본 적이 있다면 일반 최소자승 회귀 모델을 탄생시키는 익숙한 최소자승 비용 함수로 인식할 수 있습니다. 이전에 보셨든 아니든 계속 진행해 봅시다.  
결국 훨씬 더 광범위한 알고리즘 계열의 특별한 경우임을 보여드리겠습니다.

## 1.1 LMS algorithm
We want to choose θ so as to minimize J(θ). To do so, let’s use a search algorithm that starts with some “initial guess” for θ, and that repeatedly changes θ to make J(θ) smaller, until hopefully we converge to a value of θ that minimizes J(θ). Specifically, let’s consider the gradient descent algorithm, which starts with some initial θ, and repeatedly performs the update:

J(θ)를 최소화하기 위해 θ을 선택하고자 합니다. 이를 위해 θ에 대한 "초기 추측"으로 시작하여 θ을 반복적으로 변경하여 J(θ)를 작게 만드는 검색 알고리즘을 사용해 보겠습니다.  
이 알고리즘은 J(θ)를 최소화하는 θ 값으로 수렴하기를 바랍니다. 구체적으로, 초기 θ로 시작하여 업데이트를 반복적으로 수행하는 경사 하강 알고리즘을 고려해 보겠습니다:

```math
\theta_j \approx \theta_j\alpha \frac{\partial}{\partial\theta_j}J(θ)
```

(This update is simultaneously performed for all values of j = 0,...,d.) Here, α is called the learning rate. This is a very natural algorithm that repeatedly takes a step in the direction of steepest decrease of J. In order to implement this algorithm, we have to work out what is the partial derivative term on the right hand side. Let’s first work it out for the case of if we have only one training example (x,y), so that we can neglect the sum in the definition of J. We have:

(이 업데이트는 j = 0,...,d의 모든 값에 대해 동시에 수행됩니다.) 여기서 α를 학습률이라고 합니다.  
이 알고리즘은 J가 가장 가파르게 감소하는 방향으로 한 걸음씩 나아가는 매우 자연스러운 알고리즘입니다. 이 알고리즘을 구현하기 위해서는 오른쪽의 편미분 항이 무엇인지 알아내야 합니다.  
먼저 J의 정의에서 합을 무시할 수 있도록 훈련 예제(x,y)가 하나뿐인 경우에 대해 알아봅시다:

```math
\frac{\partial}{\partial\theta_j}J(θ) = \frac{\partial}{\partial\theta_j}\frac{1}{2} (h_\theta(x)- y)^2 = (h_\theta(x) - y)x_j
```

For a single training example, this gives the update rule:

단일 훈련 예제의 경우, 업데이트 규칙이 제공됩니다:

```math
\theta_j \approx \theta_j + \alpha(y - h_\theta(x))x_j
```
We use the notation “a := b” to denote an operation (in a computer program) in which we set the value of a variable a to be equal to the value of b. In other words, this operation overwrites awith the value of b. In contrast, we will write “a= b” when we are asserting a statement of fact, that the value of a is equal to the value of b.

컴퓨터 프로그램에서 변수 a의 값을 b의 값과 같게 설정하는 작업을 나타내기 위해 "a := b"라는 표기법을 사용합니다.  
즉, 이 작업은 a를 b의 값으로 덮어씁니다. 반면에, a의 값이 b의 값과 같다는 사실을 주장할 때는 "a= b"라고 쓸 것입니다.

The rule is called the LMS update rule (LMS stands for “least mean squares”), and is also known as the Widrow-Hoﬀ learning rule. This rule has several properties that seem natural and intuitive. For instance, the magnitude of the update is proportional to the error term (y(i)−hθ(x(i))); thus, for in- stance, if we are encountering a training example on which our prediction nearly matches the actual value of y(i), then we find that there is little need to change the parameters; in contrast, a larger change to the parameters will be made if our prediction hθ(x(i)) has a large error (i.e., if it is very far from y(i)).

이 규칙은 LMS 업데이트 규칙(LMS는 "최소 평균 제곱"을 의미함)이라고 하며, Widrow-Hoﬀ 학습 규칙이라고도 합니다.  
이 규칙은 자연스럽고 직관적으로 보이는 몇 가지 속성을 가지고 있습니다.  
예를 들어, 업데이트의 크기는 오류 항 $(y - h_\theta(x))$ 에 비례합니다.  
따라서 현재 상태에서 우리의 예측이 실제 y 값과 거의 일치하는 훈련 예제를 접하게 되면, 매개변수를 변경할 필요가 거의 없다는 것을 알게 됩니다.  
반대로, 우리의 예측 $h_\theta(x)$가 큰 오류(즉, y보다 먼)를 가질 경우 매개변수에 대한 더 큰 변경이 이루어질 것입니다.

단일 훈련 예제만 있을 때의 LMS 규칙을 도출했습니다. 두 개 이상의 예제로 구성된 훈련 세트에 대해 이 방법을 수정하는 두 가지 방법이 있습니다.  
첫 번째는 다음 알고리즘으로 대체하는 것입니다:

<img width="765" alt="스크린샷 2025-04-13 오후 10 41 02" src="https://github.com/user-attachments/assets/0bc35e25-4e7f-4a4d-8ea0-edb65780d112" />

By grouping the updates of the coordinates into an update of the vector θ, we can rewrite update (1.1) in a slightly more succinct way:

좌표의 업데이트를 벡터 θ의 업데이트로 그룹화하면 업데이트 (1.1)을 조금 더 간결하게 다시 쓸 수 있습니다:

<img width="375" alt="스크린샷 2025-04-13 오후 10 55 21" src="https://github.com/user-attachments/assets/faca7dd9-f4aa-4cfe-b7c2-4465d4fedb0b" />

The reader can easily verify that the quantity in the summation in the update rule above is just ∂J(θ)/∂θj (for the original definition of J). So, this is simply gradient descent on the original cost function J. This method looks at every example in the entire training set on every step, and is called batch gradient descent. Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus gradient descent always converges (assuming the learning rate α is not too large) to the global minimum. Indeed, J is a convex quadratic function. Here is an example of gradient descent as it is run to minimize a quadratic function.

독자는 위의 업데이트 규칙에서 요약된 양이 ∂J(θ)/∂θj(원래 J의 정의에 대해)라는 것을 쉽게 확인할 수 있습니다.  
따라서 이것은 단순히 원래 비용 함수 J에 대한 경사 하강법입니다. 이 방법은 모든 단계에서 전체 훈련 세트의 모든 예제를 살펴보며 배치 경사 하강법이라고 합니다.  
경사 하강법은 일반적으로 국소 최소값에 취약할 수 있지만, 선형 회귀에 대해 여기서 제안한 최적화 문제는 전역 최적값이 하나뿐이고 다른 국소적인 최적값이 없으므로 경사 하강법은 항상 전역 최소값으로 수렴합니다(학습률 α가 너무 크지 않다고 가정).  
실제로 J는 볼록 이차 함수입니다. 다음은 이차 함수를 최소화하기 위해 실행되는 경사 하강법의 예입니다.

<img width="470" alt="스크린샷 2025-04-13 오후 10 57 39" src="https://github.com/user-attachments/assets/06da31b4-c512-404f-ba92-8356717db052" />

The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at (48,30). The x’s in the figure (joined by straight lines) mark the successive values of θ that gradient descent went through. When we run batch gradient descent to fit θ on our previous dataset, to learn to predict housing price as a function of living area, we obtain θ0 = 71.27, θ1 = 0.1345. If we plot hθ(x) as a function of x (area), along with the training data, we obtain the following figure:

위에 표시된 타원은 이차 함수의 등고선입니다.  
또한 (48,30)에서 초기화된 경사 하강법을 사용한 궤적도 보여줍니다.  
그림의 x(직선으로 이어지는)는 경사 하강법이 겪은 연속적인 θ 값을 나타냅니다.  
이전 데이터셋에서 θ에 맞추기 위해 배치 경사 하강법을 실행하면 생활 면적에 따른 주택 가격 예측 방법을 배우기 위해 θ_0 = 71.27, θ_1 = 0.1345가 됩니다.  
훈련 데이터와 함께 h(x)를 x(면적)의 함수로 표시하면 다음과 같은 그림을 얻을 수 있습니다:

<img width="470" alt="스크린샷 2025-04-13 오후 11 00 42" src="https://github.com/user-attachments/assets/76db1c6b-b9ba-4f19-84a9-1a06bd22d442" />

If the number of bedrooms were included as one of the input features as well, we get θ0 = 89.60,θ1 = 0.1392, θ2 =−8.738. The above results were obtained with batch gradient descent. There is an alternative to batch gradient descent that also works very well. Consider the following algorithm:

침실 수가 입력 기능 중 하나로 포함된 경우 θ0 = 89.60, θ1 = 0.1392, θ2 =-8.738이 됩니다.  
위의 결과는 배치 경사 하강법을 사용하여 얻었습니다. 배치 경사 하강법에 대한 대안도 매우 잘 작동합니다. 다음 알고리즘을 고려해 보세요:

<img width="766" alt="스크린샷 2025-04-13 오후 11 01 30" src="https://github.com/user-attachments/assets/d453595e-8f6d-455f-999f-cc90ab4fcacb" />

By grouping the updates of the coordinates into an update of the vector θ, we can rewrite update (1.2) in a slightly more succinct way:

좌표의 업데이트를 벡터 θ의 업데이트로 그룹화하면 업데이트(1.2)를 조금 더 간결하게 다시 쓸 수 있습니다:

<img width="331" alt="스크린샷 2025-04-13 오후 11 02 51" src="https://github.com/user-attachments/assets/4b8c7cc3-904e-459a-90c0-caf6d20c37a2" />

In this algorithm, we repeatedly run through the training set, and each time we encounter a training example, we update the parameters according to the gradient of the error with respect to that single training example only. This algorithm is called stochastic gradient descent (also incremental gradient descent). Whereas batch gradient descent has to scan through the entire training set before taking a single step "a costly operation if n is large" stochastic gradient descent can start making progress right away, and continues to make progress with each example it looks at. Often, stochastic gradient descent gets θ “close” to the minimum much faster than batch gradient descent. (Note however that it may never “converge” to the minimum, and the parameters θ will keep oscillating around the minimum of J(θ); but in practice most of the values near the minimum will be reasonably good approximations to the true minimum.2) For these reasons, particularly when the training set is large, stochastic gradient descent is often preferred over batch gradient descent.

이 알고리즘에서는 훈련 집합을 반복적으로 실행하며, 훈련 예제를 만날 때마다 해당 단일 훈련 예제에 대해서만 오차의 기울기에 따라 매개변수를 업데이트합니다.  
이 알고리즘을 확률적 경사 하강법(또는 증분 경사 하강법)이라고 합니다. 배치 경사 하강법은 한 단계 "n이 클 경우 비용이 많이 드는 작업"을 수행하기 전에 전체 훈련 집합을 스캔해야 하는 반면, 확률적 경사 하강법은 즉시 진행을 시작할 수 있으며, 각 예제를 살펴볼 때마다 계속 진행됩니다.  
종종 확률적 경사 하강법은 배치 경사 하강법보다 훨씬 빠르게 최소값에 "가까운" θ을 얻습니다. (그러나 최소값에 "수렴"하지 않을 수 있으며, 매개변수 θ은 J(θ)의 최소값 근처에서 계속 진동할 것입니다. 그러나 실제로는 최소값 근처의 대부분의 값이 실제 최소값에 대해 상당히 좋은 근사치가 될 것입니다.2)  
이러한 이유로 인해 특히 훈련 집합이 클 때 배치 경사 하강법보다 확률적 경사 하강법을 선호하는 경우가 많습니다.

By slowly letting the learning rate α decrease to zero as the algorithm runs, it is also possible to ensure that the parameters will converge to the global minimum rather than merely oscillate around the minimum.

알고리즘이 실행됨에 따라 학습률 α를 천천히 0으로 낮추면 매개변수가 단순히 최소값을 중심으로 진동하는 것이 아니라 전역 최소값에 수렴하도록 보장할 수도 있습니다.

## 1.2 The normal equations
Gradient descent gives one way of minimizing J. Let’s discuss a second way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm. In this method, we will minimize J by explicitly taking its derivatives with respect to the θj’s, and setting them to zero. To enable us to do this without having to write reams of algebra and pages full of matrices of derivatives, let’s introduce some notation for doing calculus with matrices.

경사 하강법은 J를 최소화하는 한 가지 방법을 제공합니다. 이번에는 반복 알고리즘에 의존하지 않고 명시적으로 최소화를 수행하는 두 번째 방법에 대해 논의해 보겠습니다.  
이 방법에서는 $θ_j$에 대한 도함수를 명시적으로 취하고 0으로 설정하여 J를 최소화합니다. 대수의 양과 도함수 행렬로 가득 찬 페이지를 작성하지 않고도 이를 수행할 수 있도록 행렬을 사용한 미적분을 수행하기 위한 몇 가지 표기법을 소개해 보겠습니다.

### 1.2.1 Matrix derivatives
For a function f : Rn×d →R mapping from n-by-d matrices to the real numbers, we define the derivative of f with respect to A to be:

함수 $f : R^{n×d} → R$ 매핑을 n*d 행렬에서 실수로 변환할 때, 우리는 f의 A에 대한 미분을 다음과 같이 정의합니다:

<img width="363" alt="스크린샷 2025-04-13 오후 11 07 28" src="https://github.com/user-attachments/assets/936cb2f5-e119-4e36-95be-61a0ce508a58" />

Thus, the gradient ∇Af(A) is itself an n-by-d matrix, whose (i,j)-element is ∂f/∂Aij.

따라서, 기울기 ∇Af(A)는 (i,j)-원소가 ∂f/∂Aij인 n-by-d 행렬 자체입니다.

For example, suppose A= A11 A12 A21 A22 is a 2-by-2 matrix, and the function f : R2×2 →R is given by

예를 들어, A= [A11 A12 A21 A22]가 2x2 행렬이고 함수 $f : R^{2×2} → R$은 다음과 같이 주어졌다고 가정합니다.

<img width="338" alt="스크린샷 2025-04-13 오후 11 09 18" src="https://github.com/user-attachments/assets/9f749b30-a681-4247-9420-308e32a4f6e2" />

Here, Aij denotes the (i,j) entry of the matrix A. We then have

여기서 Aij는 행렬 A의 (i,j) 항목을 나타냅니다. 그런 다음

<img width="338" alt="스크린샷 2025-04-13 오후 11 09 53" src="https://github.com/user-attachments/assets/bc7cc901-7f32-4d42-82c6-bcb7ba4498c1" />

을 얻을 수 있습니다.

### 1.2.2 Least squares revisited
Armed with the tools of matrix derivatives, let us now proceed to find in closed-form the value of θ that minimizes J(θ). We begin by re-writing J in matrix-vectorial notation. Given a training set, define the design matrix X to be the n-by-dmatrix (actually n-by-d + 1, if we include the intercept term) that contains the training examples’ input values in its rows:

행렬 도함수 도구를 사용하여 이제 J(θ)를 최소화하는 θ의 값을 닫힌 형태로 찾아보겠습니다. 먼저 행렬-벡터 표기법으로 J를 다시 작성하는 것으로 시작합니다.  
훈련 집합이 주어졌을 때, 설계 행렬 X를 행에 훈련 예제의 입력 값을 포함하는 n-by-d-matrix (실제로 n-by-d + 1, 인터셉트 항을 포함하면)로 정의합니다:

<img width="272" alt="스크린샷 2025-04-13 오후 11 11 25" src="https://github.com/user-attachments/assets/d9878855-aa8a-4cc1-919c-068db1c6e0de" />

Also, let ⃗y be the n-dimensional vector containing all the target values from the training set:

또한, ⃗y를 훈련 세트의 모든 목표 값을 포함하는 n차원 벡터라고 가정해 보겠습니다:

<img width="183" alt="스크린샷 2025-04-13 오후 11 12 17" src="https://github.com/user-attachments/assets/3e047e83-f21d-4a4d-b38e-e3e48e85afa1" />

Now, since hθ(x(i)) = (x(i))Tθ, we can easily verify that

이제 h θ(x(i) = (x(i)T θ)이므로 쉽게 확인할 수 있습니다

<img width="410" alt="스크린샷 2025-04-13 오후 11 13 57" src="https://github.com/user-attachments/assets/6aef5a08-5109-4348-8f32-25fe2ca44c25" />

Thus, using the fact that for a vector z, we have that

따라서 벡터 z에 대해 다음과 같은 사실을 사용하여

<img width="525" alt="스크린샷 2025-04-13 오후 11 14 51" src="https://github.com/user-attachments/assets/cc9166e5-1c6d-4297-b8e7-d659ae10de62" />

Finally, to minimize J, let’s find its derivatives with respect to θ. Hence,

마지막으로, J를 최소화하기 위해 θ에 대한 그 도함수를 구합시다. 따라서,

<img width="618" alt="스크린샷 2025-04-13 오후 11 18 11" src="https://github.com/user-attachments/assets/d1d09230-f41b-4e4c-a392-98637df9be29" />

In the third step, we used the fact that aTb= bTa, and in the fifth step used the facts ∇xbTx= b and ∇xxTAx= 2Ax for symmetric matrix A (for more details, see Section 4.3 of “Linear Algebra Review and Reference”). To minimize J, we set its derivatives to zero, and obtain the normal equations:

세 번째 단계에서는 aTb= bTa라는 사실을 사용했고, 다섯 번째 단계에서는 대칭 행렬 A에 대해 ∇xbTx= b와 ∇xxTAx= 2Ax라는 사실을 사용했습니다(자세한 내용은 "선형 대수 검토 및 참조" 섹션 4.3을 참조하십시오).  
J를 최소화하기 위해 그 도함수를 0으로 설정하고 정규 방정식을 얻습니다:

```math
X^TX\theta = X^T ⃗y
```

Thus, the value of θ that minimizes J(θ) is given in closed form by the equation

따라서 J(θ)를 최소화하는 θ의 값은 다음 방정식에 의해 닫힌 형태로 주어집니다

<img width="232" alt="스크린샷 2025-04-13 오후 11 20 56" src="https://github.com/user-attachments/assets/9212dca2-5180-4a6f-adb2-2d802a057964" />

Note that in the above step, we are implicitly assuming that XTX is an invertible matrix. This can be checked before calculating the inverse. If either the number of linearly independent examples is fewer than the number of features, or if the features are not linearly independent, then XTX will not be invertible. Even in such cases, it is possible to “fix” the situation with additional techniques, which we skip here for the sake of simplicty.

위 단계에서는 XTX가 가역 행렬이라고 암묵적으로 가정합니다. 이는 역행렬을 계산하기 전에 확인할 수 있습니다.  
선형적으로 독립적인 예제의 수가 특징의 수보다 적거나 특징이 선형적으로 독립적이지 않은 경우 XTX는 가역 행렬이 아닙니다.  
이러한 경우에도 단순성을 위해 여기서는 생략하는 추가 기법으로 상황을 "고정"할 수 있습니다.

## 1.3 Probabilistic interpretation
When faced with a regression problem, why might linear regression, and specifically why might the least-squares cost function J, be a reasonable choice? In this section, we will give a set of probabilistic assumptions, under which least-squares regression is derived as a very natural algorithm. Let us assume that the target variables and the inputs are related via the equation

회귀 문제에 직면했을 때 선형 회귀가 합리적인 선택이 될 수 있는 이유는 무엇이며, 특히 최소제곱 비용 함수 J가 합리적인 선택이 될 수 있는 이유는 무엇일까요?  
이 섹션에서는 최소제곱 회귀가 매우 자연스러운 알고리즘으로 도출되는 일련의 확률적 가정을 제시하겠습니다.  
목표 변수와 입력이 방정식을 통해 관련이 있다고 가정해 보겠습니다

<img width="232" alt="스크린샷 2025-04-13 오후 11 22 50" src="https://github.com/user-attachments/assets/1feb29dd-5840-4433-9295-356d8118e3b3" />

where ϵ(i) is an error term that captures either unmodeled eﬀects (such as if there are some features very pertinent to predicting housing price, but that we’d left out of the regression), or random noise. Let us further assume that the ϵ(i) are distributed IID (independently and identically distributed) according to a Gaussian distribution (also called a Normal distribution) with mean zero and some variance σ2. We can write this assumption as “ϵ(i) ∼ N(0,σ2).” I.e., the density of ϵ(i) is given by

여기서 ϵ(i)는 주택 가격 예측과 매우 관련이 있지만 회귀에서 제외한 특징이 있는 경우와 같이 모델링되지 않은 효과 또는 무작위 노이즈를 포착하는 오류 항입니다.  
또한 ϵ(i)가 평균이 0이고 분산이 $σ^2$인 가우시안 분포(또는 정규 분포라고도 함)에 따라 독립적이고 동일하게 분포된 IID라고 가정해 보겠습니다.  
이 가정을 "ϵ(i) ~ N(0, σ2)"라고 쓸 수 있습니다. 즉, ϵ(i)의 밀도는 다음과 같이 주어집니다

<img width="663" alt="스크린샷 2025-04-13 오후 11 26 15" src="https://github.com/user-attachments/assets/91ae99a0-4a97-4141-9c33-abe9819689a3" />

The notation “p(y(i)|x(i); θ)” indicates that this is the distribution of y(i) given x(i) and parameterized by θ. Note that we should not condition on θ (“p(y(i)|x(i),θ)”), since θ is not a random variable. We can also write the distribution of y(i) as y(i) |x(i); θ∼N(θTx(i),σ2). Given X (the design matrix, which contains all the x(i)’s) and θ, what is the distribution of the y(i)’s? The probability of the data is given by p(⃗y|X; θ). This quantity is typically viewed a function of ⃗y(and perhaps X), for a fixed value of θ. When we wish to explicitly view this as a function of θ, we will instead call it the likelihood function:

표기법 "p(y(i)|x(i); θ)"는 x(i)가 주어졌을 때 θ에 의해 매개변수화된 y(i)의 분포임을 나타냅니다.  
θ는 랜덤 변수가 아니기 때문에 θ ("p(y(i))|x(i), θ)") 를 조건으로 해서는 안 됩니다.  
또한 y(i)의 분포를 y(i) |x(i); θ~N(θ^Tx(i), σ2)으로 쓸 수도 있습니다.  
X(모든 x(i)와 θ를 포함하는 설계 행렬)가 주어졌을 때, y(i)의 분포는 무엇인가요?  
데이터의 확률은 p(⃗ y|X; θ)로 주어집니다. 이 양은 일반적으로 고정된 θ 값에 대해 ⃗ y(및 아마도 X)의 함수로 간주됩니다.  
이를 θ의 함수로 명시적으로 보고자 할 때, 대신 우도 함수라고 부르겠습니다:

<img width="330" alt="스크린샷 2025-04-13 오후 11 32 14" src="https://github.com/user-attachments/assets/38714814-888a-4016-b3a6-8419466cf85a" />

Note that by the independence assumption on the ϵ(i)’s (and hence also the y(i)’s given the x(i)’s), this can also be written

ϵ(i)에 대한 독립성 가정(따라서 y(i)도 x(i)로 주어짐)에 따라 다음과 같이 쓸 수 있습니다

<img width="497" alt="스크린샷 2025-04-13 오후 11 33 28" src="https://github.com/user-attachments/assets/c331dcec-5a32-4193-a826-5687215eca04" />

Now, given this probabilistic model relating the y(i)’s and the x(i)’s, what is a reasonable way of choosing our best guess of the parameters θ? The principal of maximum likelihood says that we should choose θ so as to make the data as high probability as possible. I.e., we should choose θ to maximize L(θ).

이제 y(i)와 x(i)를 연결하는 확률 모델을 고려할 때, 매개변수 θ을 가장 잘 맞추는 합리적인 방법은 무엇일까요?  
최대 우도의 원리는 데이터를 가능한 한 높은 확률로 만들기 위해 θ을 선택해야 한다고 말합니다. 즉, L(θ)을 최대화하기 위해 θ을 선택해야 합니다.

Instead of maximizing L(θ), we can also maximize any strictly increasing function of L(θ). In particular, the derivations will be a bit simpler if we instead maximize the log likelihood ℓ(θ):

L(θ)을 최대화하는 대신 L(θ)의 엄격하게 증가하는 함수를 최대화할 수도 있습니다.  
특히 로그 우도 ℓ(θ)을 최대화하면 도출이 조금 더 간단해집니다:

<img width="538" alt="스크린샷 2025-04-13 오후 11 35 51" src="https://github.com/user-attachments/assets/2332e5f1-f716-450c-b466-5b35aa28a554" />

Hence, maximizing ℓ(θ) gives the same answer as minimizing

따라서 ℓ(θ)을 최대화하는 것은 아래를 최소화하는 것과 동일한 답을 제공합니다

<img width="244" alt="스크린샷 2025-04-13 오후 11 37 03" src="https://github.com/user-attachments/assets/75f06420-d069-43e7-9909-74487ce2b5d4" />

which we recognize to be J(θ), our original least-squares cost function. To summarize: Under the previous probabilistic assumptions on the data, least-squares regression corresponds to finding the maximum likelihood esti- mate of θ. This is thus one set of assumptions under which least-squares re- gression can be justified as a very natural method that’s just doing maximum likelihood estimation. (Note however that the probabilistic assumptions are by no means necessary for least-squares to be a perfectly good and rational procedure, and there may—and indeed there are—other natural assumptions that can also be used to justify it.) Note also that, in our previous discussion, our final choice of θ did not depend on what was σ2, and indeed we’d have arrived at the same result even if σ2 were unknown. We will use this fact again later, when we talk about the exponential family and generalized linear models.

우리가 J(θ)라고 인식하는 것은 원래 최소자승 비용 함수입니다.  
요약하자면: 데이터에 대한 이전의 확률적 가정 하에서 최소자승 회귀는 θ의 최대 우도 추정치를 찾는 것과 일치합니다.  
따라서 최소자승 회귀가 최대 우도 추정을 수행하는 매우 자연스러운 방법으로 정당화될 수 있는 가정 세트입니다.  
(그러나 최소자승이 완벽하게 좋고 합리적인 절차가 되기 위해 확률적 가정이 반드시 필요한 것은 아니며, 이를 정당화하는 데 사용할 수 있는 다른 자연스러운 가정도 "그리고 실제로 존재할 수 있습니다.")  
또한 이전 논의에서 θ의 최종 선택은 $σ^2$에 의존하지 않았으며, 실제로 $σ^2$가 알려지지 않았더라도 동일한 결과에 도달했을 것입니다.  
지수 가족 및 일반화된 선형 모델에 대해 이야기할 때 이 사실을 나중에 다시 사용하겠습니다.

## 1.4 Locally weighted linear regression (optional reading)
Consider the problem of predicting y from x∈R. The leftmost figure below shows the result of fitting a y= θ0 + θ1x to a dataset. We see that the data doesn’t really lie on straight line, and so the fit is not very good.

$x ∈ R$에서 y를 예측하는 문제를 생각해 보세요.  
아래 가장 왼쪽 그림은 $y = θ_0 + θ_1x$ 를 데이터셋에 맞춘 결과를 보여줍니다.  
데이터가 실제로 직선에 위치하지 않아서 적합도가 그다지 좋지 않다는 것을 알 수 있습니다.

<img width="773" alt="스크린샷 2025-04-13 오후 11 40 31" src="https://github.com/user-attachments/assets/8177cbe1-4b0d-4846-b9d7-a873ab894edd" />

Instead, if we had added an extra feature x2, and fit y= θ0 + θ1x+ θ2x2 , then we obtain a slightly better fit to the data. (See middle figure) Naively, it might seem that the more features we add, the better. However, there is also a danger in adding too many features: The rightmost figure is the result of fitting a 5-th order polynomial y= 5 j=0 θjxj. We see that even though the fitted curve passes through the data perfectly, we would not expect this to be a very good predictor of, say, housing prices (y) for diﬀerent living areas (x). Without formally defining what these terms mean, we’ll say the figure on the left shows an instance of underfitting—in which the data clearly shows structure not captured by the model—and the figure on the right is an example of overfitting. (Later in this class, when we talk about learning theory we’ll formalize some of these notions, and also define more carefully just what it means for a hypothesis to be good or bad.)

대신에, 만약 우리가 추가적인 특징 $x^2$를 추가하고 $y=θ_0 + θ_1x+ θ_2x^2$를 맞췄다면, 데이터에 약간 더 잘 맞는 결과를 얻을 수 있을 것입니다. (가운데 그림 참조)  
순진하게도, 더 많은 특징을 추가할수록 더 좋은 것처럼 보일 수 있습니다.  
그러나 너무 많은 특징을 추가하는 데에는 위험도 있습니다: 가장 오른쪽 그림은 5차 다항식 $y= \sum^5_{j=0} θ_jx^j$를 맞춘 결과입니다.  
적합 곡선이 데이터를 완벽하게 통과하더라도, 이것이 다양한 생활 지역(x)의 주택 가격(y)을 예측하는 데 매우 적합하지 않을 것이라는 것을 알 수 있습니다.  
이 용어들이 무엇을 의미하는지 공식적으로 정의하지 않고, 왼쪽의 그림은 데이터가 모델에 포착되지 않은 구조를 명확하게 보여주는 과소적합 사례이고, 오른쪽의 그림은 과잉적합의 예라고 할 수 있습니다. (이 수업 후반에 학습 이론에 대해 이야기할 때, 우리는 이러한 개념들 중 일부를 공식화하고, 가설이 좋거나 나쁘다는 것이 무엇을 의미하는지 더 신중하게 정의할 것입니다.)

As discussed previously, and as shown in the example above, the choice of features is important to ensuring good performance of a learning algorithm. (When we talk about model selection, we’ll also see algorithms for automat- ically choosing a good set of features.) In this section, let us briefly talk about the locally weighted linear regression (LWR) algorithm which, assum- ing there is suﬃcient training data, makes the choice of features less critical. This treatment will be brief, since you’ll get a chance to explore some of the properties of the LWR algorithm yourself in the homework.

앞서 논의한 바와 위의 예시에서 볼 수 있듯이 학습 알고리즘의 우수한 성능을 보장하기 위해서는 특징 선택이 중요합니다. (모델 선택에 대해 이야기할 때, 좋은 특징 집합을 자동으로 선택하는 알고리즘도 살펴보겠습니다.)  
이 섹션에서는 충분한 학습 데이터가 있다고 가정하면 특징 선택의 중요성이 낮아지는 국소 가중 선형 회귀(LWR) 알고리즘에 대해 간략히 설명하겠습니다. 

In the original linear regression algorithm, to make a prediction at a query point x (i.e., to evaluate h(x)), we would:

원래 선형 회귀 알고리즘에서 쿼리 지점 x에서 예측을 수행하기 위해 (즉, h(x))를 평가하기 위해), 우리는 이렇게 할 것입니다:

<img width="422" alt="스크린샷 2025-04-13 오후 11 44 43" src="https://github.com/user-attachments/assets/6a16e76e-b20c-430d-8249-eb75217c9667" />

In contrast, the locally weighted linear regression algorithm does the following:

반면에, 국소 가중 선형 회귀 알고리즘은 다음과 같은 작업을 수행합니다:

<img width="470" alt="스크린샷 2025-04-13 오후 11 45 14" src="https://github.com/user-attachments/assets/6b7d4a57-fb8d-4a30-a303-7480e78b4787" />

Here, the w(i)’s are non-negative valued weights. Intuitively, if w(i) is large for a particular value of i, then in picking θ, we’ll try hard to make (y(i)− θTx(i))2 small. If w(i) is small, then the (y(i)−θTx(i))2 error term will be pretty much ignored in the fit.

여기서 w(i)는 음수가 아닌 값의 가중치입니다.  
직관적으로 특정 i 값에 대해 w(i)가 크면 θ를 선택할 때 (y(i)-θ^Tx(i)^2를 작게 만들기 위해 열심히 노력할 것입니다.  
w(i)가 작으면 적합도에서 (y(i)-θ^Tx(i)^2 오차 항은 거의 무시됩니다.

A fairly standard choice for the weights is

가중치에 대한 비교적 표준적인 선택은 다음과 같습니다

<img width="303" alt="스크린샷 2025-04-13 오후 11 47 05" src="https://github.com/user-attachments/assets/0c0da4ac-46ad-40ec-8372-b1b435de8d6a" />

If xis vector-valued, this is generalized to be w(i) = exp(−(x(i)−x)T(x(i)−x)/(2τ2)), or w(i) = exp(−(x(i)−x)TΣ−1(x(i)−x)/(2τ2)), for an appropriate choice of τ or Σ.

x가 벡터 값인 경우 이는 w(i) = exp(−(x(i)−x)T(x(i)−x)/(2τ2)) 또는 w(i) = exp(−(x(i)−x)TΣ−1(x(i)−x)/(2τ2))로 일반화되어 적절한 τ 또는 Σ를 선택할 수 있습니다.

Note that the weights depend on the particular point xat which we’re trying to evaluate x. Moreover, if |x(i)−x|is small, then w(i) is close to 1; and if |x(i)−x|is large, then w(i) is small. Hence, θ is chosen giving a much higher “weight” to the (errors on) training examples close to the query point x. (Note also that while the formula for the weights takes a form that is cosmetically similar to the density of a Gaussian distribution, the w(i)’s do not directly have anything to do with Gaussians, and in particular the w(i) are not random variables, normally distributed or otherwise.) The parameter τ controls how quickly the weight of a training example falls oﬀ with distance of its x(i) from the query point x; τ is called the bandwidth parameter, and is also something that you’ll get to experiment with in your homework.

가중치는 x를 평가하려는 특정 지점 x에 따라 달라집니다.  
또한 |x(i)-x|가 작으면 w(i)는 1에 가깝고, |x(i)-x|가 크면 w(i)는 작습니다.  
따라서 쿼리 지점 x에 가까운 (오류에 대한) 훈련 예제에 훨씬 더 높은 "가중치"를 부여하는 θ을 선택합니다. (또한 가중치 공식은 가우시안 분포의 밀도와 미용적으로 유사한 형태를 취하지만, w(i)는 가우시안 분포와 직접적으로 관련이 없으며, 특히 w(i)는 정규 분포든 그렇지 않든 무작위 변수가 아닙니다.)  
매개변수 τ는 훈련 예제의 가중치가 쿼리 지점 x로부터 x(i) 거리에 따라 얼마나 빨리 감소하는지를 제어합니다. τ는 대역폭 매개변수라고 불리며, 숙제에서 실험할 수 있는 요소이기도 합니다.

Locally weighted linear regression is the first example we’re seeing of a non-parametric algorithm. The (unweighted) linear regression algorithm that we saw earlier is known as a parametric learning algorithm, because it has a fixed, finite number of parameters (the θi’s), which are fit to the data. Once we’ve fit the θi’s and stored them away, we no longer need to keep the training data around to make future predictions. In contrast, to make predictions using locally weighted linear regression, we need to keep the entire training set around. The term “non-parametric” (roughly) refers to the fact that the amount of stuﬀ we need to keep in order to represent the hypothesis h grows linearly with the size of the training set.

국소 가중 선형 회귀는 비모수 알고리즘의 첫 번째 예입니다. 앞서 살펴본 (가중치가 없는) 선형 회귀 알고리즘은 데이터에 맞는 고정된 유한한 수의 매개변수(θi)를 가지고 있기 때문에 파라메트릭 학습 알고리즘으로 알려져 있습니다.  
$θ_i$를 맞추고 저장한 후에는 더 이상 훈련 데이터를 유지하여 미래 예측을 할 필요가 없습니다.  

반면에 국소 가중 선형 회귀를 사용하여 예측을 하려면 전체 훈련 세트를 유지해야 합니다.  
"비모수적"이라는 용어는 (대략적으로) 우리가 가설 h를 나타내기 위해 지켜야 할 것들의 양이 훈련 세트의 크기에 따라 선형적으로 증가한다는 사실을 의미합니다.

```
non-parametric model은 데이터가 특정 분포를 따른다는 가정이 없기 때문에 우리가 학습에 따라 튜닝해야 할 파라미터가 명확하게 정해져 있지 않은 것이다.
그러므로 non-parametric model은 우리에게 data에 대한 사전 지식이 전혀 없을 때 유용하게 사용될 수 있다.

Parametric model: Linear regression, Logistic regression, Bayesian inference, Neural network(CNN, RNN 등) 등. 모델이 학습해야 하는 것이 명확히 정해져 있기 때문에 속도가 빠르고, 모델을 이해하기가 쉽다는 장점이 있다. 하지만 데이터의 분포가 특정한 분포를 따른다는 가정을 해야 하기 때문에 flexibility가 낮고, 간단한 문제를 푸는 데에 더 적합하다는 단점을 가진다. 
Non-parametric model: Decision tree, Random forest, K-nearest neighbor classifier 등. 데이터가 특정한 분포를 따른다는 가정을 하지 않기 때문에 더 flexible하다는 장점이 있다. 하지만 속도가 느린 경우가 많고, 더 큰 데이터를 필요로 하는 경우가 있으며 모델이 왜 그런 형태가 되었는지에 대한 명확한 설명을 하기가 쉽지 않다. 
https://process-mining.tistory.com/131
```
