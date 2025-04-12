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
In any programming language, functions are used to implement the piece of code that is required to be executed numerous times at different locations in the code. In such cases, instead of writing long pieces of codes again and again, you can simply define a function that contains the piece of code, and then you can call the function wherever you want in the code.

모든 프로그래밍 언어에서 함수는 코드의 여러 위치에서 여러 번 실행해야 하는 코드 조각을 구현하는 데 사용됩니다.  
이러한 경우 긴 코드 조각을 반복해서 쓰는 대신 코드 조각을 포함하는 함수를 정의한 다음 코드의 원하는 위치에 함수를 호출하기만 하면 됩니다.

To create a function in Python, the def keyword is used, followed by the name of the function and opening and closing parenthesis. Once a function is defined, you have to call it in order to execute the code inside a function body. To call a function, you simply have to specify the name of the function, followed by opening and closing parenthesis. In the following script, we create a function named myfunc, which prints a simple statement on the console using the print() method.

파이썬에서 함수를 만들려면 def 키워드를 사용하고 함수 이름과 열고 닫는 괄호를 붙입니다.  
함수가 정의되면 함수 본문 내에서 코드를 실행하려면 함수를 호출해야 합니다.  
함수를 호출하려면 함수의 이름을 지정한 다음 괄호를 열고 닫기만 하면 됩니다.  
다음 스크립트에서는 print() 메서드를 사용하여 콘솔에 간단한 문을 인쇄하는 myfunc이라는 함수를 만듭니다.

You can also pass values to a function. The values are passed inside the parenthesis of the function call. However, you must specify the parameter name in the function definition, too. In the following script, we define a function named myfuncparam() . The function accepts one parameter, i.e., num. The value passed in the parenthesis of the function call will be stored in this num variable and will be printed by the print() method inside the myfuncparam() method.

값을 함수에 전달할 수도 있습니다. 값은 함수 호출의 괄호 안에 전달됩니다.  
그러나 함수 정의에서도 매개변수 이름을 지정해야 합니다.  
다음 스크립트에서는 함수 myfuncparam()이라는 함수를 정의합니다. 함수는 하나의 매개변수, 즉 num을 허용합니다.  
함수 호출의 괄호 안에 전달된 값은 이 num 변수에 저장되며 myfuncaram() 메서드 내의 print() 메서드에 의해 인쇄됩니다.

```
1. def myfuncparam(num):
2. 3. print (“This is a function with parameter value: “+num )
4. ### function call
5. myfuncparam(“Parameter 1” )
```

Finally, a function can also return values to the function call. To do so, you simply have to use the return keyword, followed by the value that you want to return. In the following script, the myreturnfunc() function returns a string value to the calling function.

마지막으로 함수는 함수 호출에 값을 반환할 수도 있습니다. 이를 위해 반환 키워드를 사용한 다음 반환하려는 값을 반환하기만 하면 됩니다.

## 2.7. Objects and Classes
Python supports object-oriented programming (OOP). In OOP, any entity that can perform some function and have some attributes is implemented in the form of an object. For instance, a car can be implemented as an object since a car has some attributes such as price, color, model, and can perform some functions such as drive car, change gear, stop car, etc. Similarly, a fruit can also be implemented as an object since a fruit has a price, name, and you can eat a fruit, grow a fruit, and perform functions with a fruit. To create an object, you first have to define a class. For instance, in the following example, a class Fruit has been defined. The class has two attributes, name and price, and one method, eat_fruit(). Next, we create an object f of class Fruit and then call the eat_fruit() method from the f object. We also access the name and price attributes of the f object and print them on the console.

파이썬은 객체 지향 프로그래밍(OOP)을 지원합니다. OOP에서는 어떤 기능을 수행할 수 있고 어떤 속성을 가질 수 있는 모든 엔티티가 객체의 형태로 구현됩니다.  
예를 들어, 자동차는 가격, 색상, 모델 등의 속성을 가지고 있고 드라이브카, 기어 변경, 스톱카 등의 일부 기능을 수행할 수 있기 때문에 자동차를 객체로 구현할 수 있습니다.  
마찬가지로 과일에는 가격, 이름이 있고 과일을 먹고, 과일을 재배하고, 과일과 함께 기능을 수행할 수 있기 때문에 과일도 객체로 구현할 수 있습니다.  

객체를 만들려면 먼저 클래스를 정의해야 합니다.  
예를 들어, 다음 예제에서는 클래스 과일이 정의되었습니다. 클래스에는 이름과 가격이라는 두 가지 속성과 하나의 메서드인 eat_fruit()이 있습니다.  
다음으로 클래스 과일의 객체 f를 생성한 다음 f 객체에서 eat_fruit() 메서드를 호출합니다. 또한 f 객체의 이름과 가격 속성에 액세스하여 콘솔에 출력합니다.

```
1. class Fruit:
2.
3. name = “apple”
4. price = 10
5.
6. def eat_fruit(self):
print (“Fruit has been eaten”)
7. 8.
9.
10. f = Fruit()
11. f.eat_fruit()
12. print (f.name)
13. print (f.price)
```

A class in Python can have a special method called a constructor. The name of the constructor method in Python is __ init __(). The constructor is called whenever an object of a class is created. Look at the following example to see the constructor in action.

파이썬의 클래스에는 생성자라는 특별한 메서드가 있을 수 있습니다. 파이썬의 생성자 메서드 이름은 __init __()입니다.  
클래스의 객체가 생성될 때마다 생성자가 호출됩니다. 다음 예제를 보면 생성자가 실제로 작동하는지 확인할 수 있습니다.

```
def init __ __(self, fruit _ name, fruit _price):
Fruit.name = fruit name
Fruit.price = fruit _price
```

## 2.8. Data Science and Machine Learning Libraries



