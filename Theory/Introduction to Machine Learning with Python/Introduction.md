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

### Knowing Your Task and Knowing Your Data

## Why Python?
