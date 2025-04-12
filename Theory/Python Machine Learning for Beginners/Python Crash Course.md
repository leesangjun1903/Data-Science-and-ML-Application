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


