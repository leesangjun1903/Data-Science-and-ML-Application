# Python Crash Course
If you are familiar with the basic concepts of the Python programming language, you can skip this chapter. For those who are absolute beginners to Python, this section provides a very brief overview of some of the most basic concepts of Python. Python is a very vast programming language, and this section is by no means a substitute for a complete Python book. However, if you want to see how various operations and commands are executed in Python, you are welcome to follow along the rest of this section.

파이썬 프로그래밍 언어의 기본 개념에 익숙하다면 이 장을 건너뛰어도 됩니다.  
파이썬을 완전히 처음 접하는 분들을 위해 이 섹션에서는 파이썬의 가장 기본적인 개념 중 일부에 대해 간략하게 설명합니다.  
파이썬은 매우 방대한 프로그래밍 언어이며, 이 섹션은 완전한 파이썬 책을 대체할 수 있는 것은 아닙니다.  
하지만 파이썬에서 다양한 연산과 명령이 어떻게 실행되는지 확인하고 싶다면 이 섹션의 나머지 부분을 따르셔도 됩니다.

## 2.1. Writing Your First Program
### Script 1: print() Hello World!

## 2.2. Python Variables and Data Types
Data types in a programming language refer to the type of data that the language is capable of processing. The following are the major data types supported by Python: 
a. Strings b. Integers c. Floating Point Numbers d. Booleans

프로그래밍 언어에서 데이터 유형은 해당 언어가 처리할 수 있는 데이터 유형을 의미합니다.  
다음은 Python에서 지원하는 주요 데이터 유형입니다:  
a. Strings b. Integers c. Floating Point Numbers d. Booleans e. Lists f. Tuples g. Dictionaries

A variable is an alias for the memory address where actual data is stored. The data or the values stored at a memory address can be accessed and updated via the variable name. Unlike other programming languages like C++, Java, and C#, Python is loosely typed, which means that you don’t have to define the data type while creating a variable. Rather, the type of data is evaluated at runtime. The following example demonstrates how to create different data types and how to store them in their corresponding variables. The script also prints the type of the variables via the type() function.

변수는 실제 데이터가 저장되는 메모리 주소의 별칭입니다.  
메모리 주소에 저장된 데이터 또는 값은 변수 이름을 통해 액세스하고 업데이트할 수 있습니다.  
C++, Java, C#과 같은 다른 프로그래밍 언어와 달리 파이썬은 느슨하게 입력되므로 변수를 만들 때 데이터 유형을 정의할 필요가 없습니다. 대신 런타임에 데이터 유형을 평가합니다.  
다음 예제는 다양한 데이터 유형을 생성하는 방법과 해당 변수에 저장하는 방법을 보여줍니다. 스크립트는 또한 type() 함수를 통해 변수 유형을 출력합니다.

```
print (type(days))

<class ‘str’>, int, float, bool, list, tuple, dict
```

## 2.3. Python Operators
Python programming language contains the following types of operators: 
a. Arithmetic Operators b. Logical Operators c. Comparison Operators d. Assignment Operators e. Membership Operators  
Let’s briefly review each of these types of operators.

파이썬 프로그래밍 언어에는 다음과 같은 유형의 연산자가 포함되어 있습니다:   
a. 산술 연산자 b. 논리 연산자 c. 비교 연산자 d. 할당 연산자 e. 멤버십 연산자  
이러한 유형의 연산자 각각에 대해 간략하게 살펴보겠습니다.

<img width="908" alt="스크린샷 2025-04-12 오후 3 27 46" src="https://github.com/user-attachments/assets/eb9d40ab-75c4-42fd-8979-5d17f10a52f8" />

### Logical Operators
Logical operators are used to perform logical AND, OR, and NOT operations in Python. The following table summarizes the logical operators. Here, X is True, and Y is False.

논리 연산자는 파이썬에서 논리적 AND, OR, NOT 연산을 수행하는 데 사용됩니다. 여기서 X는 참이고 Y는 거짓입니다.

<img width="908" alt="스크린샷 2025-04-12 오후 3 29 46" src="https://github.com/user-attachments/assets/f78ab113-b260-4e32-8be2-c8d31a4df6d7" />

### Comparison Operators
Comparison operators, as the name suggests, are used to compare two or more than two operands. Depending upon the relation between the operands, comparison operators return Boolean values. The following table summarizes comparison operators in Python. Here, X is 20, and Y is 35.

이름에서 알 수 있듯이 비교 연산자는 두 개 이상의 피연산자를 비교하는 데 사용됩니다.  
피연산자 간의 관계에 따라 비교 연산자는 부울 값을 반환합니다. 다음 표는 파이썬으로 비교 연산자를 요약한 것입니다. 여기서 X는 20이고 Y는 35입니다.

<img width="908" alt="스크린샷 2025-04-12 오후 3 31 36" src="https://github.com/user-attachments/assets/9af993d0-05a8-4fc9-817d-f41ba88a9003" />

### Assignment Operators
Assignment operators are used to assign values to variables. The following table summarizes the assignment operators.

할당 연산자는 변수에 값을 할당하는 데 사용됩니다. 다음 표는 할당 연산자를 요약한 것입니다.
<img width="908" alt="스크린샷 2025-04-12 오후 3 33 31" src="https://github.com/user-attachments/assets/e2a9158a-0f81-496c-ae1b-7b77421c6cb9" />

<img width="908" alt="스크린샷 2025-04-12 오후 3 33 46" src="https://github.com/user-attachments/assets/f13e20c5-d7a3-4fce-a515-bcea3f8b3784" />

### Membership Operators
Membership operators are used to find if an item is a member of a collection of items or not. There are two types of membership operators: the in operator and the not in operator. The following script shows the in operator in action.

멤버십 연산자는 아이템이 아이템 컬렉션의 멤버인지 아닌지를 찾는 데 사용됩니다.  
멤버십 연산자에는 두 가지 유형이 있습니다:  
in operator and the not in operator. 

```
1. days = (“Sunday”
,
“Monday”
,
“Saturday” )
2. print (‘Xunday’ not in days)
,
“Tuesday”
,
“Wednesday”
,
“Thursday”
,
“Friday”
Output:
True
```

## 2.4. Conditional Statements
Conditional statements in Python are used to implement conditional logic in Python. Conditional statements help you decide whether to execute a certain code block or not. There are three main types of conditional statements in Python: a. If statement b. If-else statement c. If-elif statement

파이썬의 조건문은 파이썬에서 조건 논리를 구현하는 데 사용됩니다.  
조건문은 특정 코드 블록을 실행할지 여부를 결정하는 데 도움이 됩니다. 파이썬에는 세 가지 주요 유형의 조건문이 있습니다:  
a. if 문 b. if-else 문 c. if-elif 문

## 2.5. Iteration Statements
Iteration statements, also known as loops, are used to iteratively execute a certain piece of code. There are two main types of iteration statements in Python: a. For loop b. While Loop

반복문(루프라고도 함)은 특정 코드 조각을 반복적으로 실행하는 데 사용됩니다.  
파이썬에는 두 가지 주요 유형의 반복문이 있습니다:  
a. For loop b. While Loop

### For Loop
The for loop is used to iteratively execute a piece of code for a certain number of times. You should typically use for loop when you know the exact number of iterations or repetitions for which you want to run your code. A for loop iterates over a collection of items. In the following example, we create a collection of five integers using the range() method. Next, a for loop iterates five times and prints each integer in the collection.

for 루프는 코드 조각을 일정 횟수 동안 반복적으로 실행하는 데 사용됩니다.  
일반적으로 코드를 실행하려는 반복 횟수나 반복 횟수를 정확히 알 때 for 루프를 사용해야 합니다.  
for 루프는 항목 모음에서 반복됩니다. 

### While Loop
The while loop keeps executing a certain piece of code unless the evaluation condition becomes false. For instance, the while loop in the following script keeps executing unless the variable c becomes greater than 10.

while 루프는 평가 조건이 거짓이 되지 않는 한 특정 코드 조각을 계속 실행합니다.  

## 2.6. Functions

## 2.7. Objects and Classes

## 2.8. Data Science and Machine Learning Libraries



