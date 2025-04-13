# Python NumPy Library for Data Analysis
NumPy (Numerical Python) is a Python library for data science and numerical computing. Many advanced data science and machine learning libraries require data to be in the form of NumPy arrays before it can be processed. In this chapter, you are going to learn some of the most commonly used functionalities of the NumPy array. NumPy comes prebuilt with Anaconda’s distribution of Python. Or else, you can install NumPy with the following pip command in a terminal or a command prompt: pip install numpy

NumPy(Numerical Python)는 데이터 과학 및 수치적 컴퓨팅을 위한 파이썬 라이브러리입니다.  
많은 고급 데이터 과학 및 머신 러닝 라이브러리는 데이터를 처리하기 전에 NumPy 배열 형태여야 합니다.  
이 장에서는 NumPy 배열에서 가장 일반적으로 사용되는 몇 가지 기능을 배우게 될 것입니다.  
NumPy는 Anaconda의 Python 배포판과 함께 미리 구축되어 있습니다.  
또는 터미널이나 명령 프롬프트에 다음과 같은 pip 명령어로 NumPy를 설치할 수 있습니다: pip install numpy

## 3.1. Advantages of NumPy Library

## 3.2. Creating NumPy Arrays
Depending upon the type of data you need inside your NumPy array, different methods can be used to create a NumPy array.

NumPy 배열에 필요한 데이터 유형에 따라 NumPy 배열을 생성하는 데 다양한 방법을 사용할 수 있습니다.

### 3.2.1. Using Array Methods
To create a NumPy array, you can pass a list to the array() method of the NumPy module as shown below:

NumPy 배열을 만들려면 아래와 같이 NumPy 모듈의 array()  메서드에 목록을 전달할 수 있습니다:

```
np.array(nums_list)

np.array([row1, row2, row3])
```

### 3.2.2. Using Arrange Method
With the arrange () method, you can create a NumPy array that contains a range of integers. The first parameter to the arrange method is the lower bound, and the second parameter is the upper bound. The lower bound is included in the array. However, the upper bound is not included.

arrange () 메서드를 사용하면 정수 범위를 포함하는 NumPy 배열을 만들 수 있습니다. 배열 메서드의 첫 번째 매개변수는 하한이고 두 번째 매개변수는 상한입니다. 하한은 배열에 포함됩니다. 그러나 상한선은 포함되지 않습니다.

```
np.arrange(5,11) : 5,6,7,8,9,10
```

You can also specify the step as a third parameter in the arrange() function. A step defines the distance between two consecutive points in the array. The following script creates a NumPy array from 5 to 11 with a step size of 2.

arrange() 함수에서 단계를 세 번째 매개변수로 지정할 수도 있습니다. 단계는 배열에서 연속적인 두 점 사이의 거리를 정의합니다. 

```
np.arange(5,12,2) : 5,7,9,11
```

### 3.2.3. Using Ones Method
The ones() method can be used to create a NumPy array of all ones. 

one() 메서드를 사용하여 모든 one의 NumPy 배열을 만들 수 있습니다.

You can create a 2-dimensional array of all ones by passing the number of rows and columns as the first and second parameters of the ones() method

행과 열의 수를 one() 메서드의 첫 번째 및 두 번째 매개변수로 전달하여 모든 one의 2차원 배열을 만들 수 있습니다.

### 3.2.4. Using Zeros Method
The zeros() method can be used to create a NumPy array of all zeros.

zeros() 방법을 사용하여 모든 영점의 NumPy 배열을 만들 수 있습니다.

### 3.2.5. using Eyes Method
The eye() method is used to create an identity matrix in the form of a 2-dimensional numPy array.

eye() 방법은 2차원 numPy 배열 형태의 항등 행렬을 만드는 데 사용됩니다.

### 3.2.6. Using Random Method
The random.rand() function from the NumPy module can be used to create a NumPy array with uniform distribution.

NumPy 모듈의 random.rand() 함수를 사용하여 균일한 분포를 가진 NumPy 배열을 만들 수 있습니다.

The random.randn() function from the NumPy module can be used to create a NumPy array with normal distribution

NumPy 모듈의 random.randn() 함수를 사용하여 정규 분포를 가진 NumPy 배열을 만들 수 있습니다.

Finally, the random.randint() function from the NumPy module can be used to create a NumPy array with random integers between a certain range. The first parameter to the randint() function specifies the lower bound, the second parameter specifies the upper bound, while the last parameter specifies the number of random integers to generate between the range.

마지막으로, NumPy 모듈의 random.randint() 함수를 사용하여 특정 범위 사이의 임의 정수를 갖는 NumPy 배열을 만들 수 있습니다.  
randint() 함수의 첫 번째 매개변수는 하한을 지정하고, 두 번째 매개변수는 상한을 지정하며, 마지막 매개변수는 범위 사이에 생성할 임의 정수의 수를 지정합니다.


## 3.3. Reshaping NumPy Arrays
A NumPy array can be reshaped using the reshape() function. It is important to mention that the product of the rows and columns in the reshaped array must be equal to the product of rows and columns in the original array. For instance, in the following example, the original array contains four rows and six columns, i.e., 4 x 6 = 24. The reshaped array contains three rows and eight columns, i.e., 3 x 8 = 24.

NumPy 배열은 reshape() 함수를 사용하여 재구성할 수 있습니다.  
재구성된 배열의 행과 열의 곱은 원래 배열의 행과 열의 곱과 같아야 한다는 점을 언급하는 것이 중요합니다.  
예를 들어, 원래 배열은 4개의 행과 6개의 열, 즉 4 x 6 = 24를 포함합니다. 재구성된 배열은 3개의 행과 8개의 열, 즉 3 x 8 = 24를 포함합니다.

```
행렬.reshape(3,8)
```

## 3.4. Array Indexing And Slicing
NumPy arrays can be indexed and sliced. Slicing an array means dividing an array into multiple parts. NumPy arrays are indexed just like normal lists. Indexes in NumPy arrays start from 0, which means that the first item of a NumPy array is stored at the 0th index. The following script creates a simple NumPy array of the first 10 positive integers.

NumPy 배열은 인덱싱하고 슬라이스할 수 있습니다. 배열을 슬라이스한다는 것은 배열을 여러 부분으로 나누는 것을 의미합니다.  
NumPy 배열은 일반 목록과 마찬가지로 인덱싱됩니다. NumPy 배열의 인덱스는 0부터 시작하므로 NumPy 배열의 첫 번째 항목이 0번째 인덱스에 저장됩니다.  

To slice an array, you have to pass the lower index, followed by a colon and the upper index. The items from the lower index (inclusive) to the upper index (exclusive) will be filtered. 

배열을 자르려면 하한을 통과한 다음 콜론(:) 값과 위쪽 인덱스, 상한을 포함하여 통과해야 합니다. 아래 인덱스(포함)에서 위쪽 인덱스(독점)까지의 항목이 필터링됩니다. 

if you specify only the upper bound, all the items from the first index to the upper bound are returned. similarly, if you specify only the lower bound, all the items from the lower bound to the last item of the array are returned.

상한만 지정하면 첫 번째 인덱스부터 상한까지 모든 항목이 반환됩니다. 마찬가지로 하한만 지정하면 하한부터 배열의 마지막 항목까지 모든 항목이 반환됩니다.

Array slicing can also be applied on a 2-dimensional array. To do so, you have to apply slicing on arrays and columns separately. A comma separates the rows and columns slicing. In the following script, the rows from the first and second index are returned, While all the columns returned. You can see the first two complete rows in the output.

배열 슬라이싱은 2차원 배열에도 적용할 수 있습니다. 이를 위해서는 배열과 열에 슬라이싱을 별도로 적용해야 합니다.  
쉼표(,)는 행과 열 슬라이싱을 구분합니다. 

## 3.5. NumPy for Arithmetic Operations
### 3.5.1. Finding Square Roots
The sqrt() function is used to find the square roots of all the elements in a list as shown below:

sqrt() 함수는 목록에 있는 모든 요소의 제곱근을 찾는 데 사용됩니다:

### 3.5.2. Finding Logs
The log() function is used to find the logs of all the elements in a list as shown below:

log() 함수는 목록에 있는 모든 요소의 로그를 찾는 데 사용됩니다:

### 3.5.3. Finding Exponents
```
np.exp(nums)
```

### 3.5.4. Finding Sine and Cosine
```
np.sin(nums), np.cos(nums)
```

## 3.6. NumPy for Linear Algebra Operations
### 3.6.1. Finding Matrix Dot Product
To find a matrix dot product, you can use the dot() function. To find the dot product, the number of columns in the first matrix must match the number of rows in the second matrix.

행렬 도트 곱을 찾으려면 dot() : np.dot(A,B) 함수를 사용할 수 있습니다. 도트 곱을 찾으려면 첫 번째 행렬의 열 수가 두 번째 행렬의 행 수와 일치해야 합니다.

### 3.6.2. Element-wise Matrix Multiplication
In addition to finding the dot product of two matrices, you can element-wise multiply two matrices. To do so, you can use the multiply() function. The dimensions of the two matrices must match.

두 행렬의 내적을 구하는 것 외에도 요소별로 두 행렬을 곱할 수 있습니다. 이를 위해 multiply() : np.multiply() 함수를 사용할 수 있습니다. 두 행렬의 차원이 일치해야 합니다.

### 3.6.3. Finding Matrix Inverse
You find the inverse of a matrix via the linalg.inv() function as shown below:

다음과 같이 linalg.inv() : np.linalg.inv(nums_2d) 함수를 통해 행렬의 역행렬을 구합니다:

### 3.6.4. Finding Matrix Determinant
linalg.det() function : np.linalg.det(nums_2d)

### 3.6.5. Finding Matrix Trace
trace() function : np.trace(nums_2d)
