# Foundations
이 장의 목적은 신경망이 어떻게 작동하는지 이해하는 데 필수적인 몇 가지 기본 정신 모델을 설명하는 것입니다.  
구체적으로 중첩된 수학 함수와 그 도함수에 대해 다룹니다.  
가장 간단한 구성 요소에서 벗어나 구성 함수의 '사슬'로 구성된 복잡한 함수를 만들 수 있음을 보여주고, 이러한 함수 중 하나가 여러 입력을 받는 행렬 곱셈인 경우에도 함수의 입력에 대한 출력의 도함수를 계산합니다.  
이 과정이 어떻게 작동하는지 이해하는 것은 신경망을 이해하는 데 필수적이며, 기술적으로는 2장이 되어서야 다룰 것입니다.

이러한 신경망의 기본 구성 요소들을 중심으로 방향을 잡으면서, 우리가 소개하는 각 개념을 세 가지 관점에서 체계적으로 설명하겠습니다:
• 수학, 방정식 또는 방정식의 형태로
• 코드, 가능한 한 적은 추가 구문(Python을 이상적인 선택으로 만듭니다)
• 코딩 인터뷰 중 흰색 ‐ 보드에 어떤 종류의 그림을 그릴지 설명하는 다이어그램입니다

이 세 가지 관점의 관점에서 서로 다른 개념을 이해하는 방법을 설명하기 위해 매우 간단한 구성 요소부터 시작하겠습니다.  
첫 번째 구성 요소는 간단하지만 비판적인 개념인 함수입니다.

## Functions
함수란 무엇이며 어떻게 설명할 수 있을까요? 신경망과 마찬가지로 함수를 설명하는 방법에는 여러 가지가 있지만 개별적으로 완전한 그림을 그리는 방법은 없습니다.  

## Math
우리가 임의로 f1과 f2라고 부르는 함수들은 숫자 x를 입력으로 받아 $x^2$ (첫 번째 경우) 또는 max(x, 0) (두 번째 경우)로 변환합니다.

## Diagrams
함수를 묘사하는 한 가지 방법은 다음과 같습니다:  
1. x-y 평면을 그립니다(여기서 x는 가로축을 나타내고 y는 세로축을 나타냅니다).  
2. 점들을 여러 개 그려보세요. 여기서 점들의 x좌표는 특정 범위에서 함수의 입력값(보통 균등하게 간격을 두고)이고, y좌표는 해당 범위에서 함수의 출력값입니다.  
3. 이 표시된 점들을 연결합니다.

함수는 입력에 어떤 일이 일어나는지에 대한 자체 내부 규칙이 있는 미니팩터처럼 숫자를 입력으로 받아 출력으로 숫자를 생성하는 상자라고 생각할 수 있습니다.

<img width="761" alt="스크린샷 2025-04-15 오후 9 13 24" src="https://github.com/user-attachments/assets/133779f2-8675-43e3-ab09-7ee3a1edf019" />

## Code
마지막으로 코드를 사용하여 이러한 함수를 설명할 수 있습니다. 그 전에 함수를 작성할 파이썬 라이브러리에 대해 조금 말씀드려야 합니다: NumPy.

Deep Learning from Scratch_1.ipynb 에서 진행합니다.

### Code caveat #1: NumPy
NumPy는 빠른 숫자 계산을 위해 널리 사용되는 파이썬 라이브러리로, 내부 데이터는 대부분 C로 작성됩니다.  
간단히 말해, 신경망에서 다루는 데이터는 항상 거의 항상 1차원, 2차원, 3차원 또는 4차원, 특히 2차원 또는 3차원 배열로 유지됩니다.  

### Code caveat #2: Type-checked functions

## Derivatives
한 지점에서 함수의 미분은 해당 지점의 입력에 대한 함수 출력의 "변화율"이라고 간단히 말하는 것으로 시작하겠습니다.

$$ \frac{df}{dx}(a) = 
\lim_{\Delta \to 0} \frac{{f \left( {a + \Delta } \right) - f\left( a - \Delta \right)}}{2 * \Delta } $$

### Diagrams

함수 f의 데카르트 표현에 단순히 접선을 그린다면, 한 지점 a에서 f의 도함수는 이 선의 기울기일 뿐입니다. 
 
 이전 부분의 수학적 설명과 마찬가지로 실제로 이 선의 기울기를 계산할 수 있는 두 가지 방법이 있습니다. 
  
  첫 번째는 미적분을 사용하여 실제로 극한을 계산하는 것입니다.  
  두 번째는 f를 -0.001과 +0.001로 연결하는 선의 기울기를 구하는 것입니다. 

## Nested Functions
수학적 관례에 따라 f1과 f2라고 부르는 두 개의 함수가 있으면 함수 중 하나의 출력이 다음 함수의 입력이 되어 "둘을 함께 묶을 수 있다"는 뜻입니다.

<img width="761" alt="스크린샷 2025-04-15 오후 9 44 17" src="https://github.com/user-attachments/assets/50c1145a-5cc2-4476-b8fe-7548ad2dbf48" />

### Math
$f_2(f_1(x))=y$

### Another Diagram
f1*f2

또한 미적분학의 정리에 따르면 "대부분 미분 가능한" 함수로 구성된 합성 함수는 그 자체로 대부분 미분 가능합니다!  
따라서 f1f2를 도함수를 계산할 수 있는 또 다른 함수로 생각할 수 있으며, 합성 함수의 도함수를 계산하는 것은 딥러닝 모델을 학습하는 데 필수적입니다.  
그러나 이 합성 함수의 도함수를 구성 함수의 도함수로 계산하려면 공식이 필요합니다. 다음으로 다룰 내용입니다.

## The Chain Rule
연쇄 법칙은 합성 함수의 도함수를 계산할 수 있게 해주는 수학적 정리입니다.  
딥러닝 모델은 수학적으로 합성 함수이며, 다음 몇 장에서 볼 수 있듯이 그 도함수에 대한 추론은 이를 학습하는 데 필수적입니다.

<img width="269" alt="스크린샷 2025-04-15 오후 9 54 41" src="https://github.com/user-attachments/assets/38a46a45-371d-4737-b6b9-22bc1374ab41" />

여기서 u는 단순히 함수에 대한 입력을 나타내는 더미 변수입니다.

## A Slightly Longer Example
<img width="424" alt="스크린샷 2025-04-15 오후 10 01 24" src="https://github.com/user-attachments/assets/a70dea54-f05e-4252-bde6-0aeb6feeac42" />

## Functions with Multiple Inputs
딥러닝에서 다루는 함수에는 종종 하나의 입력만 있는 것이 아닙니다. 대신 특정 단계에서 여러 입력이 합산되거나 곱해지거나 다른 방식으로 결합됩니다.  
앞으로 살펴보겠지만, 입력에 대한 이러한 함수들의 출력의 도함수를 계산하는 것은 여전히 문제가 되지 않습니다: 두 개의 입력이 합산된 다음 다른 함수를 통해 입력되는 매우 간단한 시나리오를 고려해 봅시다.

1단계에서는 x와 y를 더하는 함수를 통해 입력합니다.

<img width="169" alt="스크린샷 2025-04-15 오후 10 09 21" src="https://github.com/user-attachments/assets/89870ef5-6d14-4f0c-b6f1-4db20be5b768" />

2단계는 함수 σ를 통해 a를 입력하는 것입니다(σ는 시그모이드, 정사각형 함수, 또는 이름이 s로 시작하지 않는 함수와 같은 모든 연속 함수일 수 있습니다).

<img width="84" alt="스크린샷 2025-04-15 오후 10 09 38" src="https://github.com/user-attachments/assets/eab04884-6d92-4883-ba29-f796f48dcb5b" />

우리는 동등하게 전체 함수를 f로 표시하고 쓸 수 있습니다:

<img width="154" alt="스크린샷 2025-04-15 오후 10 10 45" src="https://github.com/user-attachments/assets/a8d93922-7f61-410e-bc01-b1c4517b49a2" />

<img width="767" alt="스크린샷 2025-04-15 오후 10 10 59" src="https://github.com/user-attachments/assets/a73a3ae2-4e7c-423c-9ac4-2edc3266a68f" />

## Derivatives of Functions with Multiple Inputs
개념적으로, 우리는 단순히 하나의 입력이 있는 함수의 경우와 동일한 작업을 수행합니다: 계산 그래프를 통해 각 구성 함수의 도함수를 "뒤로 이동"한 다음 결과를 곱하여 총 도함수를 얻습니다. 이는 그림 1-13에 나와 있습니다.

<img width="767" alt="스크린샷 2025-04-15 오후 10 12 48" src="https://github.com/user-attachments/assets/0be01848-b109-4209-a6fb-2eefe5a4b97c" />

## Functions with Multiple Vector Inputs
딥러닝에서 우리의 목표는 일부 데이터에 모델을 맞추는 것입니다. 더 정확히 말하면, 이는 데이터에서 관찰한 것, 즉 함수의 입력이 될 것, 즉 함수의 출력이 될 일부 원하는 예측에 가능한 한 최적의 방식으로 매핑하는 수학적 함수를 찾고자 한다는 것을 의미합니다.  
결과적으로 이러한 관찰은 일반적으로 행을 관찰로, 각 열을 해당 관찰의 숫자 특징으로 하는 행렬로 인코딩될 것입니다.  
이에 대해서는 다음 장에서 더 자세히 다룰 것이며, 현재로서는 점곱과 행렬 곱셈을 포함하는 복소수 함수의 도함수에 대해 추론하는 것이 필수적입니다. 

신경망에서 단일 데이터 포인트 또는 "관찰"을 나타내는 일반적인 방법은 n개의 특징이 있는 행으로, 각 특징은 단순히 x1, x2 등의 숫자로 xn까지 표시됩니다:

## Creating New Features from Existing Features
신경망에서 가장 일반적인 단일 작업은 이러한 특징들의 "가중합"을 형성하는 것일 수 있습니다. 가중합은 특정 특징을 강조하고 다른 특징들을 비강조화할 수 있으며, 따라서 새로운 특징으로 생각될 수 있습니다.  
이를 수학적으로 표현하는 간결한 방법은 이 관찰의 내적곱으로, 특징들과 동일한 길이의 "가중치" 집합인 w1, w2 등을 wn까지 포함하는 것입니다.

<img width="767" alt="스크린샷 2025-04-15 오후 10 17 40" src="https://github.com/user-attachments/assets/770d3f06-fdec-445e-8dfc-2322ab771f3b" />

## Derivatives of Functions with Multiple Vector Inputs

<img width="767" alt="스크린샷 2025-04-15 오후 10 18 38" src="https://github.com/user-attachments/assets/93c72235-2091-465e-9d0d-d90181eb0a23" />

## Vector Functions and Their Derivatives: One Step Further
우리의 함수가 벡터 X와 W를 사용하여 이전 섹션에서 설명한 내적을 수행한 다음 함수 σ를 통해 벡터를 공급한다고 가정해 보겠습니다.  
이전과 동일한 목표를 표현하겠지만, 새로운 언어로 X와 W에 대한 이 새로운 함수의 출력의 기울기를 계산하고자 합니다.

<img width="767" alt="스크린샷 2025-04-15 오후 10 21 36" src="https://github.com/user-attachments/assets/4b2f756c-d64d-4995-a3f3-dc9cc856c70f" />

## Vector Functions and Their Derivatives: The Backward Pass
그림 1-19에 표시된 이 함수의 역방향 통과 다이어그램은 이전 예제와 유사하며 수학보다 더 높은 수준입니다.  
행렬 곱셈 결과에서 평가된 σ 함수의 도함수를 기반으로 한 곱셈을 하나 더 추가하기만 하면 됩니다.

<img width="767" alt="스크린샷 2025-04-15 오후 10 22 57" src="https://github.com/user-attachments/assets/8cae3277-53e2-461e-aefa-15bc39ac4bc6" />

## Computational Graph with Two 2D Matrix Inputs
간단한 예제를 자세히 살펴보고 2D 행렬의 곱셈이 1D 벡터의 점곱이 아니라 관련된 경우에도 이 장 내내 사용한 추론이 여전히 수학적으로 의미가 있으며 실제로 코딩하기 매우 쉽다는 것을 보여드리겠습니다. 

<img width="204" alt="스크린샷 2025-04-15 오후 10 32 58" src="https://github.com/user-attachments/assets/30e8f2fb-4f94-4f1f-9287-397cb91f1279" />

<img width="686" alt="스크린샷 2025-04-15 오후 10 33 12" src="https://github.com/user-attachments/assets/e8d328bb-5f4c-428b-8908-1ff43ad43cf6" />

<img width="747" alt="스크린샷 2025-04-15 오후 10 33 26" src="https://github.com/user-attachments/assets/c7b3e77f-58c0-4a4e-b78d-408cd5ba175b" />

<img width="686" alt="스크린샷 2025-04-15 오후 10 33 40" src="https://github.com/user-attachments/assets/9c177c78-842c-4665-8485-0bcb03dc46e0" />

<img width="344" alt="스크린샷 2025-04-15 오후 10 34 07" src="https://github.com/user-attachments/assets/cf1fb54f-bce5-4dd2-ac3a-d33d47bf8dad" />

<img width="760" alt="스크린샷 2025-04-15 오후 10 34 24" src="https://github.com/user-attachments/assets/6337d8bb-8f5c-4018-bd35-51c7ed0bedd4" />

## The Fun Part: The Backward Pass
<img width="760" alt="스크린샷 2025-04-15 오후 10 37 51" src="https://github.com/user-attachments/assets/79379bcb-3621-480d-a2bd-e33e304527bc" />

먼저 이를 직접 계산할 수 있다는 점에 유의하세요. 값 L은 실제로 x11, x12 등의 함수이며, x33까지입니다. 하지만 이는 복잡해 보입니다.  
복잡한 함수의 도함수를 간단한 조각으로 분해하고 각 조각을 계산한 다음 결과를 곱하기만 하면 된다는 것이 체인 규칙의 핵심이 아니었을까요?  
실제로 이러한 사실이 이러한 것들을 쉽게 코딩할 수 있게 해주었습니다: 전진 패스를 한 걸음 한 걸음씩 진행하면서 결과를 저장한 다음 그 결과를 사용하여 후진 패스에 필요한 모든 도함수를 평가했습니다.  
이 접근 방식은 행렬이 관련된 경우에만 작동한다는 것을 보여드리겠습니다. 자세히 살펴보겠습니다. L을 λ σ ν X, W로 체인 규칙으로 쓸 수 있습니다: . 이것이 정규 함수라면 그냥 쓸 것입니다

<img width="327" alt="스크린샷 2025-04-15 오후 10 43 07" src="https://github.com/user-attachments/assets/6a08d2de-818f-4736-aae4-22f6cb50be95" />

<img width="135" alt="스크린샷 2025-04-15 오후 10 43 24" src="https://github.com/user-attachments/assets/d52aaa84-07f6-4444-8970-410ee0110e2d" />

<img width="56" alt="스크린샷 2025-04-15 오후 10 43 47" src="https://github.com/user-attachments/assets/f8a9271f-a2d0-48c6-9976-9695bbf5a348" />

<img width="218" alt="스크린샷 2025-04-15 오후 10 43 36" src="https://github.com/user-attachments/assets/bd7b30d6-9765-4123-ab53-3a2bbe37958c" />

<img width="618" alt="스크린샷 2025-04-15 오후 10 44 47" src="https://github.com/user-attachments/assets/5a980389-d78c-4b98-a99d-892d12b969c5" />

<img width="467" alt="스크린샷 2025-04-15 오후 10 47 46" src="https://github.com/user-attachments/assets/a95626bc-5fbd-463b-b7b5-d4412cf153c1" />

<img width="279" alt="스크린샷 2025-04-15 오후 10 47 34" src="https://github.com/user-attachments/assets/1cb12bbf-3a09-4a62-b605-79c78207887c" />

### Describing these gradients visually




