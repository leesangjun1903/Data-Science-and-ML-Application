# Introduction and Environment Set Up
Data science libraries exist in various programming languages. However, you will be using Python programming language for data science and machine learning since Python is flexible, easy to learn, and offers the most advanced data science and machine learning libraries. Furthermore, Python has a huge data science community where you can take help from whenever you want. In this chapter, you will see how to set up the Python environment needed to run various data science and machine learning libraries. The chapter also contains a crash Python course for absolute beginners in Python. Finally, the different data science and machine learning libraries that we are going to study in this book have been discussed. The chapter ends with a simple exercise.

데이터 과학 라이브러리는 다양한 프로그래밍 언어로 존재합니다.  
하지만 파이썬은 유연하고 학습하기 쉬우며 가장 진보된 데이터 과학 및 머신 러닝 라이브러리를 제공하기 때문에 데이터 과학과 머신 러닝을 위해 파이썬 프로그래밍 언어를 사용하게 됩니다.  
또한 파이썬에는 원할 때 언제든지 도움을 받을 수 있는 방대한 데이터 과학 커뮤니티가 있습니다.  
이 장에서는 다양한 데이터 과학 및 머신 러닝 라이브러리를 실행하는 데 필요한 파이썬 환경을 설정하는 방법을 살펴보겠습니다.  
이 장에서는 파이썬의 절대 초보자를 위한 crash Python course 도 포함되어 있습니다.  
마지막으로 이 책에서 공부할 다양한 데이터 과학 및 머신 러닝 라이브러리에 대해 설명합니다. 이 장은 간단한 연습으로 마무리됩니다.

## 1.1. Difference between Data Science and Machine Learning
Data science and machine learning are terms that are often interchangeably used. However, the two terms are different. Data science is a subject area of that uses scientific approaches and mathematical techniques such as statistics to draw out meaning and insights from data. According to Dr. Thomas Miller from Northwestern University, data science is “a combination of information technology, modeling and business management.” Machine learning, on the other hand, is an approach that consists of mathematical algorithms that enable computers to make decisions without being explicitly performed. Rather, machine learning algorithms learn from data, and then based on the insights from the dataset, make decisions without human input. In this book, you will learn both Data Science and Machine Learning. In the first five chapters, you will study the concepts required to store, analyze, and visualize the datasets. From the 6th chapter onwards, different types of machine learning concepts are explained.

데이터 과학과 머신 러닝은 종종 혼용되어 사용되는 용어입니다. 그러나 두 용어는 다릅니다.  
데이터 과학은 데이터에서 의미와 인사이트를 도출하기 위해 통계와 같은 과학적 접근 방식과 수학적 기법을 사용하는 주제 분야입니다.  
노스웨스턴 대학교의 토마스 밀러 박사에 따르면 데이터 과학은 "정보 기술, 모델링 및 비즈니스 관리의 조합"이라고 합니다.  
반면 머신 러닝은 컴퓨터가 명시적으로 수행하지 않고도 의사 결정을 내릴 수 있도록 하는 수학적 알고리즘으로 구성된 접근 방식입니다.  
오히려 머신 러닝 알고리즘은 데이터를 통해 학습한 다음 데이터셋의 인사이트를 기반으로 사람의 입력 없이 의사 결정을 내립니다.  
이 책에서는 데이터 과학과 머신 러닝을 모두 배우게 됩니다. 처음 다섯 장에서는 데이터셋을 저장, 분석, 시각화하는 데 필요한 개념을 공부합니다. 6장부터는 다양한 유형의 머신 러닝 개념에 대해 설명합니다.

## 1.2. Steps in Learning Data Science and Machine Learning

### 1. 데이터 과학과 머신 러닝의 본질 알아보기   
1. Know What Data Science and Machine Learning Is All About 
Before you delve deep into developing data science and machine learning applications, you have to know what the field of data science and machine learning is, what you can do with that, and what are some of the best tools and libraries that you can use. The first chapter of the book answers these questions.

데이터 과학과 머신 러닝 응용 프로그램 개발에 깊이 몰두하기 전에 데이터 과학과 머신 러닝 분야가 무엇인지, 이를 통해 무엇을 할 수 있는지, 사용할 수 있는 최고의 도구와 라이브러리는 무엇인지 알아야 합니다.  
이 책의 첫 장에서는 이러한 질문에 답합니다.

### 2. 프로그래밍 언어 배우기
2. Learn a Programming Language 
If you wish to be a data science and machine learning expert, you have to learn programming. There is no working around this fact. Though there are several cloud- based machine learning platforms like Amazon Sage Maker and Azure ML Studio where you can create data science applications without writing a single line of code. However, to get fine-grained control over your applications, you will need to learn programming. And though you can program natural language applications in any programming language, I would recommend that you learn Python programming language. Python is one of the most routinely used libraries for data science and machine learning with myriads of basic and advanced data science and ML libraries. In addition, many data science applications are based on deep learning and machine learning techniques. Again, Python is the language that provides easy to use libraries for deep learning and machine learning. In short, learn Python. Chapter 2 contains a crash course for absolute beginners in Python.

데이터 과학 및 머신 러닝 전문가가 되고 싶다면 프로그래밍을 배워야 합니다. 이 사실을 해결할 방법은 없습니다.  
Amazon Sage Maker 및 Azure ML Studio와 같은 클라우드 기반 머신 러닝 플랫폼이 하나의 코드를 작성하지 않고도 데이터 과학 애플리케이션을 만들 수 있습니다.  
하지만 애플리케이션에 대한 세밀한 제어를 얻으려면 프로그래밍을 배워야 합니다.  
그리고 모든 프로그래밍 언어로 자연어 애플리케이션을 프로그래밍할 수 있지만 Python 프로그래밍 언어를 배우는 것이 좋습니다.  
Python은 수많은 기본 및 고급 데이터 과학 및 ML 라이브러리를 갖춘 데이터 과학 및 머신 러닝에 가장 일반적으로 사용되는 라이브러리 중 하나입니다.  
또한 많은 데이터 과학 애플리케이션이 딥러닝과 머신 러닝 기술을 기반으로 합니다.  
다시 말해, Python은 딥러닝과 머신러닝에 라이브러리를 쉽게 사용할 수 있는 언어입니다. 즉, Python을 학습합니다.  
2장에는 Python의 절대 초보자를 위한 크래시 코스가 포함되어 있습니다.

### 3. 기본부터 시작하기 
3. Start with the Basics 
Start with very basic data science applications. I would rather recommend that you should not start developing data science applications right away. Start with basic mathematical and numerical operations like computing dot products and matrix multiplication, etc. Chapter 3 of this book explains how to use the NumPy library for basic data science and machine learning tasks. You should also know how to import data into your application and how to visualize it. Chapters 4 and 5 of this book explain the task of data analysis and visualization. After that, you should know how to visualize and preprocess data.

매우 기본적인 데이터 과학 애플리케이션부터 시작하세요.  
데이터 과학 애플리케이션을 바로 개발하지 않는 것이 좋습니다.  
점 곱셈 및 행렬 곱셈과 같은 기본적인 수학적 및 수치적 연산부터 시작하세요.  
이 책의 3장에서는 기본 데이터 과학 및 머신 러닝 작업에 NumPy 라이브러리를 사용하는 방법을 설명합니다.  
또한 애플리케이션으로 데이터를 가져오는 방법과 이를 시각화하는 방법도 알아야 합니다.  
이 책의 4장과 5장에서는 데이터 분석 및 시각화 작업에 대해 설명합니다.  
그 후에는 데이터를 시각화하고 전처리하는 방법을 알아야 합니다.

### 4. 머신 러닝 및 딥 러닝 알고리즘 배우기
4. Learn Machine Learning and Deep Learning Algorithms
Data Science, machine learning, and deep learning go hand in hand. Therefore, you have to learn machine learning and deep learning algorithms. Among machine learning, start with the supervised learning techniques. Supervised machine learning algorithms are chiefly divided into two types, i.e., regression and classification. Chapter 6 of this book explains regression algorithms, while chapter 7 explains classification algorithms. Chapter 8 explains unsupervised machine learning, while chapter 9 briefly reviews deep learning techniques. Finally, the 10th Chapter explains how to reduce feature (dimensions) set to improve the performance of machine learning applications.

데이터 과학, 머신 러닝, 딥 러닝은 함께합니다. 따라서 머신 러닝과 딥 러닝 알고리즘을 학습해야 합니다.  
머신 러닝 중에서 지도 학습 기법부터 시작하세요. 지도 학습 알고리즘은 주로 회귀와 분류의 두 가지 유형으로 나뉩니다.  
이 책의 6장에서는 회귀 알고리즘에 대해 설명하고, 7장에서는 분류 알고리즘에 대해 설명합니다.  
8장에서는 비지도 학습 머신 러닝에 대해 설명하고, 9장에서는 딥 러닝 기법에 대해 간략하게 살펴봅니다.  
마지막으로 10장에서는 머신 러닝 애플리케이션의 성능을 향상시키기 위해 설정된 특징(차원)을 줄이는 방법에 대해 설명합니다.

### 5. 데이터 과학 애플리케이션 개발  
5. Develop Data Science Applications  
Once you are familiar with basic machine learning and deep learning algorithms, you are good to go for developing data science applications. Data science applications can be of different types, i.e., predicting house prices, recognizing images, classifying text, etc. Being a beginner, you should try to develop versatile data science applications, and later, when you find your area of interest, e.g., natural language processing or image recognition, delve deep into that. It is important to mention that this book provides a very generic introduction to data science, and you will see applications of data science to structured data, textual data, and image data. However, this book is not dedicated to any specific data science field.

기본적인 머신러닝과 딥러닝 알고리즘에 익숙해지면 데이터 과학 응용 프로그램을 개발하는 것이 좋습니다.  
데이터 과학 응용 프로그램은 집값 예측, 이미지 인식, 텍스트 분류 등 다양한 유형이 있을 수 있습니다.  
초보자라면 다재다능한 데이터 과학 응용 프로그램을 개발하려고 노력해야 하며, 나중에 자연어 처리나 이미지 인식과 같은 관심 분야를 찾으면 그 부분을 깊이 탐구해야 합니다.  
이 책은 데이터 과학에 대한 매우 일반적인 소개를 제공하며, 구조화된 데이터, 텍스트 데이터, 이미지 데이터에 대한 데이터 과학의 응용 사례를 살펴볼 수 있다는 점을 언급하는 것이 중요합니다.  
그러나 이 책은 특정 데이터 과학 분야에 국한되지 않습니다.

### 6. 데이터 과학 애플리케이션 배포  
6. Deploying Data Science Applications  
To put a data science or machine learning application into production so that anyone can use it, you need to deploy it to production. There are several ways to deploy data science applications. You can use dedicated servers containing REST APIs that can be used to call various functionalities in your data science application. To deploy such applications, you need to learn Python Flask, Docker, or similar web technology. In addition to that, you can also deploy your applications using Amazon W eb Services or any other cloud-based deployment platform. To be an expert data science and machine learning practitioner, you need to perform the aforementioned 6 steps in an iterative manner. The more you practice, the better you will get at NLP.

누구나 사용할 수 있도록 데이터 과학 또는 머신 러닝 애플리케이션을 프로덕션에 배포하려면 이를 프로덕션에 배포해야 합니다.  
데이터 과학 애플리케이션을 배포하는 방법에는 여러 가지가 있습니다.  
데이터 과학 애플리케이션에서 다양한 기능을 호출하는 데 사용할 수 있는 REST API가 포함된 전용 서버를 사용할 수 있습니다.  
이러한 애플리케이션을 배포하려면 Python Flask, Docker 또는 유사한 웹 기술을 배워야 합니다.  
그 외에도 Amazon Web Services 또는 기타 클라우드 기반 배포 플랫폼을 사용하여 애플리케이션을 배포할 수도 있습니다.  
데이터 과학 및 머신 러닝 전문가가 되려면 앞서 언급한 6단계를 반복적으로 수행해야 합니다. 연습을 많이 할수록 NLP에서 더 나은 결과를 얻을 수 있습니다.

## 1.3. Environment Setup
### 1.3.1. Windows Setup
### 1.3.2. Mac Setup
### 1.3.3. Linux Setup
### 1.3.4. Using Google Colab Cloud Environment


