# Introduction


Machine learning is about extracting knowledge from data. It is a research field at the intersection of statistics, artificial intelligence, and computer science and is also known as predictive analytics or statistical learning. The application of machine learning methods has in recent years become ubiquitous in everyday life. From auto‐ matic recommendations of which movies to watch, to what food to order or which products to buy, to personalized online radio and recognizing your friends in your photos, many modern websites and devices have machine learning algorithms at their core. When you look at a complex website like Facebook, Amazon, or Netflix, it is very likely that every part of the site contains multiple machine learning models. Outside of commercial applications, machine learning has had a tremendous influence on the way data-driven research is done today. The tools introduced in this book have been applied to diverse scientific problems such as understanding stars, finding distant planets, discovering new particles, analyzing DNA sequences, and providing personalized cancer treatments. Your application doesn’t need to be as large-scale or world-changing as these examples in order to benefit from machine learning, though. In this chapter, we will explain why machine learning has become so popular and discuss what kinds of problems can be solved using machine learning. Then, we will show you how to build your first machine learning model, introducing important concepts along the way.

머신 러닝은 데이터에서 지식을 추출하는 것입니다. 통계, 인공지능, 컴퓨터 과학이 교차하는 연구 분야이며 예측 분석 또는 통계 학습이라고도 합니다.  
최근 몇 년 동안 머신 러닝 방법의 적용은 일상 생활에서 보편화되었습니다.  
어떤 영화를 시청할지, 어떤 음식을 주문할지 또는 어떤 제품을 구매할지, 개인화된 온라인 라디오와 사진 속 친구를 인식할지에 이르기까지 많은 현대 웹사이트와 기기의 핵심에는 머신 러닝 알고리즘이 있습니다.  
페이스북, 아마존, 넷플릭스와 같은 복잡한 웹사이트를 보면 사이트의 모든 부분에 여러 머신 러닝 모델이 포함되어 있을 가능성이 매우 높습니다.  
상업적 응용 분야 외에도 머신 러닝은 오늘날 데이터 기반 연구가 수행되는 방식에 엄청난 영향을 미쳤습니다.  
이 책에서 소개한 도구는 별 이해, 먼 행성 발견, 새로운 입자 발견, DNA 서열 분석, 개인화된 암 치료 제공 등 다양한 과학적 문제에 적용되었습니다.  
하지만 당신의 응용 프로그램이 머신 러닝의 혜택을 누리기 위해 이러한 예시처럼 대규모 또는 세계를 변화시킬 필요는 없습니다.  
이 장에서는 머신 러닝이 대중화된 이유를 설명하고 머신 러닝을 사용하여 해결할 수 있는 문제의 종류에 대해 설명합니다.  
그런 다음 중요한 개념을 소개하면서 첫 번째 머신 러닝 모델을 구축하는 방법을 보여드리겠습니다.

## Why Machine Learning?
In the early days of “intelligent” applications, many systems used handcoded rules of “if ” and “else” decisions to process data or adjust to user input. Think of a spam filter whose job is to move the appropriate incoming email messages to a spam folder. You could make up a blacklist of words that would result in an email being marked as spam. This would be an example of using an expert-designed rule system to design an “intelligent” application. Manually crafting decision rules is feasible for some applica‐ tions, particularly those in which humans have a good understanding of the process to model. However, using handcoded rules to make decisions has two major disadvantages:

"지능형" 애플리케이션 초기에는 많은 시스템이 데이터를 처리하거나 사용자 입력에 적응하기 위해 "만약" 및 "다른" 결정의 수갑을 채운 규칙을 사용했습니다.  
적절한 수신 이메일 메시지를 스팸 폴더로 옮기는 것이 임무인 스팸 필터를 생각해 보세요. 이메일이 스팸으로 표시되는 단어의 블랙리스트를 작성할 수도 있습니다.  
이는 전문가가 설계한 규칙 시스템을 사용하여 "지능형" 애플리케이션을 설계하는 예가 될 수 있습니다.  
일부 애플리케이션, 특히 인간이 모델링하는 과정을 잘 이해하고 있는 경우 의사 결정 규칙을 수동으로 만드는 것이 가능합니다.  
그러나 수갑을 채운 규칙을 사용하여 의사 결정을 내리는 데는 두 가지 주요 단점이 있습니다:

• The logic required to make a decision is specific to a single domain and task. Changing the task even slightly might require a rewrite of the whole system. 
• Designing rules requires a deep understanding of how a decision should be made by a human expert.

• 결정을 내리는 데 필요한 논리는 단일 도메인과 작업에만 적용됩니다. 작업을 조금이라도 변경하려면 전체 시스템을 다시 작성해야 할 수도 있습니다. 
• 규칙을 설계하려면 전문가가 어떻게 결정을 내려야 하는지에 대한 깊은 이해가 필요합니다.

One example of where this handcoded approach will fail is in detecting faces in images. Today, every smartphone can detect a face in an image. However, face detection was an unsolved problem until as recently as 2001. The main problem is that the way in which pixels (which make up an image in a computer) are “perceived” by the computer is very different from how humans perceive a face. This difference in representation makes it basically impossible for a human to come up with a good set of rules to describe what constitutes a face in a digital image. Using machine learning, however, simply presenting a program with a large collection of images of faces is enough for an algorithm to determine what characteristics are needed to identify a face.

이 수갑으로 채워진 접근 방식이 실패할 수 있는 한 가지 예는 이미지에서 얼굴을 감지하는 것입니다.  
오늘날 모든 스마트폰은 이미지에서 얼굴을 감지할 수 있습니다. 그러나 얼굴 감지는 2001년까지만 해도 해결되지 않은 문제였습니다.  
가장 큰 문제는 컴퓨터에서 이미지를 구성하는 픽셀이 컴퓨터에 의해 '인식'되는 방식이 인간이 얼굴을 인식하는 방식과 매우 다르다는 것입니다.  
이러한 표현의 차이로 인해 인간은 디지털 이미지에서 얼굴을 구성하는 요소를 설명하기 위한 좋은 규칙을 생각해내는 것이 기본적으로 불가능합니다.  
그러나 머신 러닝을 사용하여 단순히 얼굴 이미지를 대량으로 수집한 프로그램을 제시하는 것만으로도 알고리즘은 얼굴을 식별하는 데 필요한 특성을 파악할 수 있습니다.

### Problems Machine Learning Can Solve
지도 학습과 비지도 학습 작업 모두에서 컴퓨터가 이해할 수 있는 입력 데이터를 표현하는 것이 중요합니다.  
종종 데이터를 표로 생각하는 것이 도움이 됩니다. 추론하고자 하는 각 데이터 포인트(각 이메일, 각 고객, 각 거래)는 행이며, 해당 데이터 포인트를 설명하는 각 속성(예: 고객의 나이 또는 거래 금액 또는 위치)은 열입니다.  
사용자의 나이, 성별, 계정을 만든 시기, 온라인 상점에서 구매한 빈도 등을 기준으로 사용자를 설명할 수 있습니다.  

종양의 이미지는 각 픽셀의 그레이스케일 값으로 설명하거나 종양의 크기, 모양, 색상을 사용하여 설명할 수도 있습니다.  
여기서 각 엔티티 또는 행은 머신 러닝에서 샘플(또는 데이터 포인트)로 알려져 있으며, 이러한 엔티티를 설명하는 열은 특징이라고 합니다.  
이 책 후반부에서는 특징 추출 또는 특징 엔지니어링이라고 하는 데이터의 좋은 ‐ 표현을 구축하는 주제에 대해 자세히 설명하겠습니다.  
그러나 어떤 머신 러닝 알고리즘도 정보가 없는 데이터에 대해 예측을 할 수 없다는 점을 염두에 두어야 합니다.  
예를 들어 환자의 성별만 있는 유일한 특징이라면 어떤 알고리즘도 환자의 성별을 미리 예측할 수 없습니다.  
이 정보는 단순히 데이터에 포함되지 않습니다. 환자의 이름이 포함된 다른 특징을 추가하면 사람의 이름으로 성별을 구분할 수 있는 경우가 많기 때문에 훨씬 더 좋은 운을 얻을 수 있습니다.

### Knowing Your Task and Knowing Your Data
머신 러닝 프로세스에서 가장 중요한 부분은 작업 중인 데이터와 해결하고자 하는 작업과 데이터가 어떻게 관련되어 있는지 이해하는 것일 수 있습니다.  
알고리즘을 무작위로 선택하여 데이터를 던지는 것은 효과적이지 않습니다.  
모델을 구축하기 전에 데이터셋에서 무슨 일이 일어나고 있는지 이해하는 것이 필요합니다.  
각 알고리즘은 어떤 종류의 데이터와 어떤 문제 설정에 가장 적합한지에 따라 다릅니다. 

머신 러닝 솔루션을 구축하는 동안 다음 질문에 답하거나 적어도 염두에 두어야 합니다: 
• 제가 어떤 질문에 답하려고 하나요? 수집된 데이터가 그 질문에 답할 수 있을까요?  
• 제 질문을 기계 학습 문제로 표현하는 가장 좋은 방법은 무엇인가요?  
• 제가 해결하고자 하는 문제를 표현할 수 있을 만큼 충분한 데이터를 수집했나요? - 데이터의 어떤 특징을 추출했으며, 이를 통해 올바른 예측이 가능할까요?  
• 지원서의 성공률을 어떻게 측정할 수 있을까요?  
• 머신 러닝 솔루션이 제 연구나 비즈니스 제품의 다른 부분과 어떻게 상호작용할까요?  

더 큰 맥락에서 머신 러닝의 알고리즘과 방법은 특정 문제를 해결하기 위한 더 큰 프로세스의 한 부분일 뿐이며, 항상 큰 그림을 염두에 두는 것이 좋습니다.  
많은 사람들이 복잡한 머신 러닝 솔루션을 구축하는 데 많은 시간을 할애하지만 올바른 문제를 해결하지 못한다는 사실을 알게 됩니다.  
이 책에서처럼 머신 러닝의 기술적 측면에 깊이 들어가면 궁극적인 목표를 놓치기 쉽습니다.  
여기에 나열된 질문에 대해서는 자세히 설명하지 않겠지만, 머신 러닝 모델을 구축할 때 명시적이든 암묵적이든 사용자가 취할 수 있는 모든 가정을 염두에 두시기 바랍니다.

## Why Python?
파이썬은 많은 데이터 과학 응용 프로그램에서 링구아 프랑카(모국어가 다른 사람들이 상호 이해를 위하여 만들어 사용하는 언어.)가 되었습니다.  
범용 프로그래밍 언어의 힘과 MATLAB 또는 R과 같은 도메인별 스크립팅 언어의 사용 편의성을 결합한 것입니다.  
파이썬에는 데이터 로딩, 시각화, 통계, 자연어 처리, 이미지 처리 등을 위한 라이브러리가 있습니다.  
이 방대한 도구 상자는 데이터 과학자들에게 다양한 범용 및 특수 목적 기능을 제공합니다.  
파이썬 사용의 주요 장점 중 하나는 터미널이나 Jupyter 노트북과 같은 다른 도구를 사용하여 코드와 직접 상호 작용할 수 있는 능력입니다.  
머신 러닝과 데이터 분석은 데이터가 분석을 주도하는 근본적인 정신적 반복 과정입니다.  
이러한 과정에서 빠른 반복과 쉬운 상호 작용을 가능하게 하는 도구가 필수적입니다.  
범용 프로그래밍 언어인 파이썬은 복잡한 그래픽 사용자 인터페이스(GUI)와 웹 서비스를 생성하고 기존 시스템에 통합할 수 있도록 합니다.

### scikit-learn
### Installing scikit-learn
$ pip install numpy scipy matplotlib ipython scikit-learn pandas

## Essential Libraries and Tools
### Jupyter Notebook
### NumPy
### SciPy
SciPy is a collection of functions for scientific computing in Python. It provides, among other functionality, advanced linear algebra routines, mathematical function optimization, signal processing, special mathematical functions, and statistical distri‐ butions. scikit-learn draws from SciPy’s collection of functions for implementing its algorithms. The most important part of SciPy for us is scipy.sparse: this provides sparse matrices, which are another representation that is used for data in scikit- learn. Sparse matrices are used whenever we want to store a 2D array that contains mostly zeros:.

SciPy는 파이썬의 과학 컴퓨팅을 위한 함수 모음입니다.  
이 함수 모음은 고급 선형 대수 루틴, 수학 함수 최적화, 신호 처리, 특수 수학 함수 및 통계 분포를 제공합니다.  
scikit-learn은 SciPy의 알고리즘 구현을 위한 함수 모음에서 파생됩니다.  
SciPy에서 가장 중요한 부분은 scipy.sparse입니다: 이는 scikit-learn의 데이터에 사용되는 또 다른 표현인 희소 행렬을 제공합니다.  
희소 행렬은 대부분 0을 포함하는 2D 배열을 저장하고 싶을 때마다 사용됩니다.

```
# Convert the NumPy array to a SciPy sparse matrix in CSR format
sparse_matrix = sparse.csr_matrix(eye)
```
Usually it is not possible to create dense representations of sparse data (as they would not fit into memory), so we need to create sparse representations directly. Here is a way to create the same sparse matrix as before, using the COO format:

일반적으로 희소 데이터의 조밀한 표현(메모리에 맞지 않기 때문에)을 만드는 것은 불가능하므로 직접 희소 표현을 만들어야 합니다.  
COO 형식을 사용하여 이전과 동일한 희소 행렬을 만드는 방법은 다음과 같습니다:

```
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
```

### matplotlib
Jupyter 노트북 내부에서 작업할 때는 %matplotlib 노트북과 %matplotlib inline 명령어를 사용하여 브라우저에서 직접 그림을 표시할 수 있습니다.  
대화형 환경을 제공하는 %matplotlib 노트북을 사용하는 것이 좋습니다(이 책을 제작하기 위해 %matplotlib 인라인을 사용하고 있지만).

### pandas
판다가 제공하는 또 다른 유용한 도구는 SQL, 엑셀 파일, 쉼표로 구분된 값(CSV) 파일과 같은 다양한 파일 형식과 데이터 ‐ 데이터베이스에서 가져올 수 있다는 점입니다. 

### mglearn
이 라이브러리는 이 책을 위해 작성한 유틸리티 함수 라이브러리로, 플롯 및 데이터 로딩에 대한 세부 정보로 코드 목록을 복잡하게 만들지 않도록 합니다.  
관심이 있으시면 저장소에서 모든 함수를 찾을 수 있지만 mglearn 모듈의 세부 정보는 이 책의 자료에 크게 중요하지 않습니다.  
코드에서 mglearn을 요청하는 부분을 보면 일반적으로 그림을 빨리 그리거나 흥미로운 데이터를 손에 넣을 수 있는 방법입니다.

### Python 2 Versus Python 3

### Versions Used in this Book
이러한 버전을 정확하게 맞추는 것은 중요하지 않지만, 우리가 사용한 것만큼 최신 버전의 scikit-learn 이 있어야 합니다. 이제 모든 것을 설정했으니 머신러닝의 첫 번째 응용 프로그램에 대해 살펴보겠습니다.

## A First Application: Classifying Iris Species
Introduction to Machine Learning with Python_1.ipynb 로 진행합니다.

이 섹션에서는 간단한 머신러닝 애플리케이션을 통해 첫 번째 모델을 만들겠습니다.  
이 과정에서 몇 가지 핵심 개념과 용어를 소개합니다.  
취미 식물학자가 자신이 발견한 홍채 꽃의 종을 구별하는 데 관심이 있다고 가정해 보겠습니다.  
그녀는 각 붓꽃과 관련된 꽃잎의 길이와 너비, 꽃받침의 길이와 너비 등 몇 가지 측정값을 수집했으며, 모두 센티미터 단위로 측정했습니다(그림 1-2 참조).  

<img width="761" alt="스크린샷 2025-04-15 오후 4 04 00" src="https://github.com/user-attachments/assets/167f26fe-b9e3-4156-89b1-4c7b6f3a62ad" />

또한 전문 식물학자가 이전에 세토사, 버시컬러 또는 버기니카 종에 속하는 것으로 확인한 일부 붓꽃의 측정값도 가지고 있습니다.  
이러한 측정값에 대해 각 붓꽃이 어떤 종에 속하는지 확신할 수 있습니다.  
우리의 취미 식물학자가 야생에서 만날 수 있는 종은 이것뿐이라고 가정해 봅시다.  
우리의 목표는 새로운 붓꽃의 종을 예측할 수 있도록 종이 알려진 이 붓꽃의 측정값을 통해 학습할 수 있는 머신러닝 모델을 구축하는 것입니다.

올바른 붓꽃 종을 알 수 있는 측정값이 있기 때문에 이것은 지도 학습 문제입니다.  
이 문제에서는 여러 옵션 중 하나(붓꽃 종)를 예측하고자 합니다.  
이것은 분류 문제의 한 예입니다. 가능한 출력(다른 종류의 붓꽃)을 클래스라고 합니다.  
데이터셋의 모든 붓꽃은 세 가지 클래스 중 하나에 속하므로 이 문제는 세 가지 클래스 분류 문제입니다.  
단일 데이터 포인트(붓꽃)에 대해 원하는 출력은 이 꽃의 종입니다.  
특정 데이터 포인트에 대해 이 꽃이 속한 종을 레이블이라고 합니다.

### Meet the Data

### Measuring Success: Training and Testing Data
우리는 이 데이터를 바탕으로 새로운 측정 세트의 홍채 종을 예측할 수 있는 기계 학습 모델을 구축하고자 합니다.  
하지만 새로운 측정에 모델을 적용하기 전에 실제로 작동하는지, 즉 예측을 신뢰해야 하는지 알아야 합니다.  
안타깝게도 모델을 구축하는 데 사용한 데이터를 사용하여 평가할 수는 없습니다.  
이는 우리 모델이 항상 전체 훈련 세트를 단순히 기억할 수 있기 때문에 훈련 세트의 어느 지점에서든 항상 올바른 레이블을 예측할 수 있기 때문입니다.  
이 "기억"은 우리 모델이 잘 일반화될지 여부(즉, 새로운 데이터에서도 좋은 성능을 발휘할지 여부)를 나타내는 것이 아닙니다.  
모델의 성능을 평가하기 위해 우리는 레이블이 있는 새로운 데이터(이전에 보지 못한 데이터)를 보여줍니다.  
이는 일반적으로 수집한 레이블 데이터(여기서 150개의 꽃 측정값)를 두 부분으로 나누는 방식으로 이루어집니다.  
데이터의 한 부분은 기계 학습 모델을 구축하는 데 사용되며, 이를 훈련 데이터 또는 훈련 세트라고 합니다.  
나머지 데이터는 테스트 데이터, 테스트 세트 또는 홀드아웃 세트라고 합니다.  
scikit-learn는 데이터셋을 셔플하고 분할하는 함수인 train_test_split 함수를 포함합니다.  
이 함수는 데이터의 행의 75%를 훈련 세트로 추출하고, 나머지 레이블과 함께 테스트 세트로 선언합니다.  
훈련과 테스트 세트에 각각 얼마나 많은 데이터를 넣을지 결정하는 것은 임의의 수이지만, 데이터의 25%를 포함하는 테스트 세트를 사용하는 것이 좋은 경험 법칙입니다.  
scikit-learn에서 데이터는 일반적으로 대문자 X로 표시되며, 레이블은 소문자 Y로 표시됩니다.  
이는 수학의 표준 공식 f(x)=y에서 영감을 받았으며, 여기서 x는 함수의 입력이고 y는 출력입니다.  
수학의 더 많은 관례에 따라 데이터가 2차원 배열(행렬)이기 때문에 대문자 X를 사용하고, 대상이 1차원 배열(벡터)이기 때문에 소문자 Y를 사용합니다.  

분할하기 전에 train_test_split 함수는 의사 난수 생성기를 사용하여 데이터셋을 셔플합니다.  
마지막 25%의 데이터를 테스트 세트로 가져간다면, 데이터 포인트는 레이블에 따라 정렬되므로 모든 데이터 포인트에는 레이블 2가 남게 됩니다(앞서 보여준 iris['target' 출력 참조).  
세 가지 클래스 중 하나만 포함된 테스트 세트를 사용하면 모델이 얼마나 잘 일반화되는지 알 수 없으므로, 테스트 데이터에 모든 클래스의 데이터가 포함되어 있는지 확인하기 위해 데이터를 셔플합니다.  
동일한 함수를 여러 번 실행해도 동일한 출력을 얻을 수 있도록 하기 위해, 우리는 무작위_상태 매개변수를 사용하여 고정된 시드를 가진 의사 난수 생성기를 제공합니다.  
이렇게 하면 결과가 결정적으로 변하므로 이 선은 항상 동일한 결과를 얻게 됩니다. 이 책에서 무작위 절차를 사용할 때 우리는 항상 무작위_상태를 이 방식으로 고정할 것입니다.

### First Things First: Look at Your Data
머신러닝 모델을 구축하기 전에 데이터를 검사하고, 머신러닝 없이도 작업을 쉽게 해결할 수 있는지, 원하는 정보가 데이터에 포함되어 있지 않은지 확인하는 것이 좋은 생각인 경우가 많습니다.  
또한 데이터를 검사하는 것은 이상과 특이점을 찾는 좋은 방법입니다. 예를 들어, 일부 붓꽃은 센티미터가 아닌 인치를 사용하여 측정한 것일 수 있습니다.  
현실 세계에서는 데이터의 불일치와 예상치 못한 측정이 매우 흔합니다.  
데이터를 검사하는 가장 좋은 방법 중 하나는 데이터를 시각화하는 것입니다.  
이를 수행하는 한 가지 방법은 산점도를 사용하는 것입니다.  
데이터의 산점도는 하나의 특징을 x축을 따라, 다른 특징을 y축을 따라 배치하고 각 데이터 포인트에 점을 그립니다.  
안타깝게도 컴퓨터 화면에는 2차원만 있어 한 번에 두 개(또는 세 개)의 특징만 플롯할 수 있습니다.  
이렇게 하면 세 개 이상의 특징이 있는 데이터셋을 플롯하기가 어렵습니다.  
이 문제를 해결하는 한 가지 방법은 가능한 모든 특징 쌍을 살펴보는 쌍 플롯을 만드는 것입니다.  
여기에 있는 네 개와 같이 특징의 수가 적다면 이는 상당히 합리적입니다.  
그러나 쌍 플롯은 모든 특징의 상호 작용을 한 번에 보여주지 않으므로 이러한 방식으로 시각화할 때 데이터의 흥미로운 측면이 드러나지 않을 수 있다는 점을 염두에 두어야 합니다.

### Building Your First Model: k-Nearest Neighbors
이제 실제 머신 러닝 모델을 구축하기 시작할 수 있습니다.  
사이킷러닝에는 사용할 수 있는 분류 알고리즘이 많이 있습니다.  
여기서는 이해하기 쉬운 k-최근접 이웃 분류기를 사용하겠습니다.  
이 모델을 구축하는 것은 학습 세트만 저장하는 것으로 구성됩니다.  
새 데이터 포인트를 예측하기 위해 알고리즘은 학습 세트에서 새 포인트에 가장 가까운 포인트를 찾습니다.  
그런 다음 이 학습 포인트의 레이블을 새 데이터 포인트에 할당합니다.

k-최근접 이웃의 k는 새로운 데이터 포인트에 가장 가까운 이웃만 사용하는 대신 학습에서 고정된 수의 이웃(예: 가장 가까운 세 개 또는 다섯 개의 이웃)을 고려할 수 있음을 의미합니다.  
그런 다음 이러한 이웃 중 다수 클래스를 사용하여 예측할 수 있습니다.  
이에 대해서는 2장에서 더 자세히 설명하겠지만, 지금은 단일 이웃만 사용하겠습니다.

### Making Predictions

### Evaluating the Model

## Summary and Outlook
이 장에서 배운 내용을 요약해 보겠습니다. 기계 학습과 그 응용에 대한 간단한 소개로 시작하여 지도 학습과 비지도 학습의 구분에 대해 논의하고 이 책에서 사용할 도구에 대해 개요를 제공했습니다.  
그런 다음 꽃의 물리적 측정을 사용하여 특정 꽃이 어떤 종류의 붓꽃에 속하는지 예측하는 작업을 공식화했습니다.  
우리는 올바른 종으로 전문가가 주석을 달아 모델을 구축한 측정 데이터셋을 사용하여 이를 지도 학습 과제로 만들었습니다.  
세 가지 가능한 종, 즉 세트사, 버시컬러, 또는 버기니카가 있어 이 작업이 세 가지 클래스 분류 문제가 되었습니다.  
가능한 종을 분류 문제에서 클래스라고 하고, 단일 홍채의 종을 레이블이라고 합니다. 붓꽃 데이터셋은 두 개의 NumPy 배열로 구성됩니다: 하나는 scikit-learn에서 X라고 불리는 데이터를 포함하는 배열이고, 다른 하나는 y라고 불리는 정확하거나 원하는 출력을 포함하는 배열입니다.  
배열 X는 데이터 포인트당 하나의 행과 특징당 하나의 열을 가진 2차원 특징 배열입니다.  
배열 Y는 1차원 배열로, 각 샘플에 대해 하나의 클래스 레이블, 0에서 2까지의 정수를 포함합니다.  
우리는 데이터셋을 훈련 세트와 테스트 세트로 나누어 모델이 이전에 보지 못한 새로운 데이터에 얼마나 잘 일반화될 수 있는지 평가했습니다.  
우리는 훈련 세트에서 가장 가까운 이웃을 고려하여 새로운 데이터 포인트에 대한 예측을 수행하는 k-최근접 이웃 분류 알고리즘을 선택했습니다.  
이는 모델을 구축하는 알고리즘과 모델을 사용하여 예측을 수행하는 알고리즘을 포함하는 KNeighborsClassifier 클래스에 구현되었습니다.  
우리는 클래스를 인스턴스화하여 매개변수를 설정했습니다.  
그런 다음 적합 방법을 호출하여 훈련 데이터(X_train)와 훈련 출력(y_train)을 매개변수로 전달하여 모델을 구축했습니다.  
우리는 모델의 정확도를 계산하는 점수 방법을 사용하여 모델을 평가했습니다.  
우리는 점수 방법을 테스트 세트 데이터와 테스트 세트 라벨에 적용한 결과, 우리 모델이 약 97% 정확하다는 것을 발견했습니다.  
이는 테스트 세트에서 97%의 시간 동안 정확하다는 것을 의미합니다.  
이를 통해 우리는 모델을 새로운 데이터(예: 새로운 꽃 측정)에 적용할 수 있는 자신감을 얻었고, 모델이 약 97%의 시간 동안 정확할 것이라고 신뢰할 수 있었습니다.  



