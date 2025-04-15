# Pretrained networks
We closed our first chapter promising to unveil amazing things in this chapter, and now it’s time to deliver. Computer vision is certainly one of the fields that have been most impacted by the advent of deep learning, for a variety of reasons. The need to classify or interpret the content of natural images existed, very large data- sets became available, and new constructs such as convolutional layers were invented and could be run quickly on GPUs with unprecedented accuracy. All of these factors combined with the internet giants’ desire to understand pictures taken by millions of users with their mobile devices and managed on said giants’ platforms. Quite the perfect storm. We are going to learn how to use the work of the best researchers in the field by downloading and running very interesting models that have already been trained on open, large-scale datasets. We closed our first chapter promising to unveil amazing things in this chapter, and now it’s time to deliver. Computer vision is certainly one of the fields that have been most impacted by the advent of deep learning, for a variety of reasons. The need to classify or interpret the content of natural images existed, very large data- sets became available, and new constructs such as convolutional layers were invented and could be run quickly on GPUs with unprecedented accuracy. All of these factors combined with the internet giants’ desire to understand pictures taken by millions of users with their mobile devices and managed on said giants’ platforms. Quite the perfect storm. We are going to learn how to use the work of the best researchers in the field by downloading and running very interesting models that have already been trained on open, large-scale datasets. We can think of a pretrained neural network as similar to a program that takes inputs and generates outputs. The behavior of such a program is dictated by the architecture of the neural network and by the examples it saw during training, in terms of desired input-output pairs, or desired properties that the output should satisfy. Using an off-the-shelf model can be a quick way to jump-start a deep learning project, since it draws on expertise from the researchers who designed the model, as well as the computation time that went into training the weights.

이미 대규모 오픈 데이터셋에서 학습된 매우 흥미로운 모델을 다운로드하고 실행하여 해당 분야 최고의 연구자들의 연구를 활용하는 방법을 배울 것입니다.  
사전 학습된 신경망은 입력을 받아 출력을 생성하는 프로그램과 유사하다고 생각할 수 있습니다.  
이러한 프로그램의 동작은 신경망의 아키텍처와 학습 중 본 예제에 따라 원하는 입력-출력 쌍 또는 출력이 만족해야 하는 원하는 속성 측면에서 결정됩니다.  
기성 모델을 사용하는 것은 딥러닝 프로젝트를 빠르게 시작하는 방법이 될 수 있습니다. 이는 모델을 설계한 연구자들의 전문 지식과 가중치 학습에 소요된 계산 시간을 활용하기 때문입니다.

이 장에서는 이미지의 내용에 따라 레이블을 지정할 수 있는 모델, 실제 이미지에서 새로운 이미지를 제작할 수 있는 모델, 적절한 영어 문장을 사용하여 이미지의 내용을 설명할 수 있는 모델 등 세 가지 인기 있는 사전 학습 모델을 살펴봅니다.  
우리는 이러한 사전 학습된 모델을 PyTorch에서 로드하고 실행하는 방법을 배우고, 우리가 논의할 사전 학습된 모델과 같은 PyTorch 모델을 균일한 인터페이스를 통해 쉽게 사용할 수 있는 도구 세트인 PyTorch Hub를 소개할 것입니다. 

## 2.1 A pretrained network that recognizes the subject of an image
As our first foray into deep learning, we’ll run a state-of-the-art deep neural network that was pretrained on an object-recognition task. There are many pretrained networks that can be accessed through source code repositories. It is common for researchers to publish their source code along with their papers, and often the code comes with weights that were obtained by training a model on a reference dataset. Using one of these models could enable us to, for example, equip our next web ser- vice with image-recognition capabilities with very little effort. The pretrained network we’ll explore here was trained on a subset of the ImageNet dataset (http://imagenet.stanford.edu). ImageNet is a very large dataset of over 14 mil- lion images maintained by Stanford University. All of the images are labeled with a hier- archy of nouns that come from the WordNet dataset (http://wordnet.princeton.edu), which is in turn a large lexical database of the English language.

딥러닝에 대한 첫 번째 도전으로, 객체 인식 작업에 대해 사전 학습된 최첨단 딥 뉴럴 네트워크를 실행할 것입니다. 소스 코드 저장소를 통해 접근할 수 있는 사전 학습된 네트워크가 많이 있습니다.

The ImageNet dataset, like several other public datasets, has its origin in academic competitions. Competitions have traditionally been some of the main playing fields where researchers at institutions and companies regularly challenge each other. Among others, the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) has gained popularity since its inception in 2010. This particular competition is based on a few tasks, which can vary each year, such as image classification (telling what object categories the image contains), object localization (identifying objects’ position in images), object detection (identifying and labeling objects in images), scene classifica- tion (classifying a situation in an image), and scene parsing (segmenting an image into regions associated with semantic categories, such as cow, house, cheese, hat). In particular, the image-classification task consists of taking an input image and produc- ing a list of 5 labels out of 1,000 total categories, ranked by confidence, describing the content of the image. The training set for ILSVRC consists of 1.2 million images labeled with one of 1,000 nouns (for example, “dog”), referred to as the class of the image. In this sense, we will use the terms label and class interchangeably. We can take a peek at images from ImageNet in figure 2.1.

ImageNet 데이터셋은 다른 여러 공개 데이터셋과 마찬가지로 학술 대회에서 기원을 두고 있습니다.  
대회는 전통적으로 기관과 기업의 연구자들이 정기적으로 서로 도전하는 주요 분야 중 하나였습니다. 그 중에서도 ImageNet 대규모 시각 인식 챌린지(ILSVRC)는 2010년에 시작된 이래로 인기를 얻고 있습니다.  
이 대회는 이미지 분류(이미지에 포함된 객체 카테고리를 알려줌), 객체 위치 파악(이미지 내 객체의 위치 식별), 객체 감지(이미지 내 객체 식별 및 라벨링), 장면 분류-이온(이미지 내 상황 분류), 장면 구문 분석(이미지를 소, 집, 치즈, 모자와 같은 의미 카테고리와 관련된 영역으로 분할함) 등 매년 달라질 수 있는 몇 가지 과제를 기반으로 합니다.  
특히, 이미지 분류 작업은 입력 이미지를 가져와서 총 1,000개의 카테고리 중 5개의 라벨을 신뢰도에 따라 순위를 매기고 이미지의 내용을 설명하는 것으로 구성됩니다.  
ILSVRC의 훈련 세트는 이미지 클래스라고 불리는 1,000개의 명사(예: "개") 중 하나로 라벨링된 120만 개의 이미지로 구성됩니다.  
이러한 의미에서 라벨과 클래스라는 용어를 혼용하여 사용할 것입니다. 그림 2.1에서 ImageNet의 이미지를 엿볼 수 있습니다.

We are going to end up being able to take our own images and feed them into our pretrained model, as pictured in figure 2.2. This will result in a list of predicted labels for that image, which we can then examine to see what the model thinks our image is. Some images will have predictions that are accurate, and others will not! The input image will first be preprocessed into an instance of the multidimen- sional array class torch.Tensor. It is an RGB image with height and width, so this ten- sor will have three dimensions: the three color channels, and two spatial image dimensions of a specific size. (We’ll get into the details of what a tensor is in chapter 3, but for now, think of it as being like a vector or matrix of floating-point numbers.) Our model will take that processed input image and pass it into the pretrained network to obtain scores for each class. The highest score corresponds to the most likely class according to the weights. Each class is then mapped one-to-one onto a class label. That output is contained in a torch.Tensor with 1,000 elements, each representing the score associated with that class. Before we can do all that, we’ll need to get the network itself, take a peek under the hood to see how it’s structured, and learn about how to prepare our data before the model can use it.

우리는 결국 그림 2.2와 같이 자신의 이미지를 가져와서 사전 학습된 모델에 입력할 수 있게 될 것입니다.  
이렇게 하면 해당 이미지에 대한 예측 레이블 목록이 표시되며, 이를 통해 모델이 우리의 이미지를 어떻게 생각하는지 살펴볼 수 있습니다.  
일부 이미지에는 정확한 예측이 있고 다른 이미지에는 그렇지 않은 예측이 있을 수 있습니다!  
입력 이미지는 먼저 다차원 분할 배열 클래스 torch.Tensor의 인스턴스로 전처리됩니다.  
이 이미지는 높이와 너비를 가진 RGB 이미지이므로 이 10개의 소르는 세 가지 차원, 즉 세 가지 색상 채널과 특정 크기의 두 가지 공간 이미지 차원을 갖습니다. (텐서가 무엇인지에 대한 자세한 내용은 3장에서 설명하겠지만, 지금은 부동 소수점 숫자의 벡터 또는 행렬과 같다고 생각하세요.)  
우리 모델은 처리된 입력 이미지를 가져와 사전 학습된 네트워크에 전달하여 각 클래스에 대한 점수를 얻습니다.  
가장 높은 점수는 가중치에 따라 가장 가능성이 높은 클래스에 해당합니다. 그런 다음 각 클래스는 클래스 레이블에 일대일로 매핑됩니다.  
출력은 각각 해당 클래스와 관련된 점수를 나타내는 1,000개의 요소로 구성된 torch.Tensor에 포함됩니다.  
이 모든 작업을 수행하기 전에 네트워크 자체를 가져와 후드 아래를 들여다보고 모델이 사용하기 전에 데이터를 준비하는 방법을 배워야 합니다.

### 2.1.1 Obtaining a pretrained network for image recognition
As discussed, we will now equip ourselves with a network trained on ImageNet. To do so, we’ll take a look at the TorchVision project (https://github.com/pytorch/vision), which contains a few of the best-performing neural network architectures for com- puter vision, such as AlexNet (http://mng.bz/lo6z), ResNet (https://arxiv.org/pdf/ 1512.03385.pdf), and Inception v3 (https://arxiv.org/pdf/1512.00567.pdf). It also has easy access to datasets like ImageNet and other utilities for getting up to speed with computer vision applications in PyTorch. We’ll dive into some of these further along in the book. For now, let’s load up and run two networks: first AlexNet, one of the early breakthrough networks for image recognition; and then a residual network, ResNet for short, which won the ImageNet classification, detection, and localization competitions, among others, in 2015. If you didn’t get PyTorch up and running in chapter 1, now is a good time to do that. The predefined models can be found in torchvision.models

또한 ImageNet 및 PyTorch의 컴퓨터 비전 애플리케이션 속도를 높이기 위한 기타 유틸리티와 같은 데이터셋에 쉽게 접근할 수 있습니다.  
이 중 일부는 책에서 더 자세히 다룰 것입니다.  
지금은 두 가지 네트워크를 로드하여 실행해 보겠습니다: 먼저 이미지 인식을 위한 초기 혁신 네트워크 중 하나인 AlexNet을 시작으로, 2015년 ImageNet 분류, 탐지 및 현지화 대회 등에서 우승한 ResNet을 줄여서 소개합니다.

대문자 이름은 여러 인기 있는 모델을 구현하는 파이썬 클래스를 의미합니다.  
이들은 아키텍처, 즉 입력과 출력 사이에 발생하는 연산의 배열에서 다릅니다.  
소문자 이름은 이러한 클래스에서 인스턴스화된 모델을 반환하는 편의 함수로, 때로는 다른 매개변수 집합을 사용하기도 합니다.  
예를 들어, resnet101은 101개의 레이어로 ResNet 인스턴스를 반환하고, resnet18은 18개의 레이어로 구성됩니다. 이제 AlexNet으로 눈을 돌립니다.


### 2.1.2 AlexNet
The AlexNet architecture won the 2012 ILSVRC by a large margin, with a top-5 test error rate (that is, the correct label must be in the top 5 predictions) of 15.4%. By comparison, the second-best submission, which wasn’t based on a deep network, trailed at 26.2%. This was a defining moment in the history of computer vision: the moment when the community started to realize the potential of deep learning for vision tasks. That leap was followed by constant improvement, with more modern architectures and training methods getting top-5 error rates as low as 3%.

AlexNet 아키텍처는 2012년 ILSVRC에서 15.4%의 큰 차이로 상위 5개의 테스트 오류율(즉, 올바른 레이블이 상위 5개의 예측에 있어야 함)로 우승했습니다. 

먼저 각 블록은 여러 곱셈과 덧셈, 그리고 5장에서 발견할 출력의 다른 함수들로 구성되어 있습니다. 하나 이상의 이미지를 입력으로 받아 다른 이미지를 출력으로 생성하는 필터라고 생각할 수 있습니다.  
이렇게 하는 방식은 학습 중에 관찰한 예제와 원하는 출력에 따라 결정됩니다.

<img width="777" alt="스크린샷 2025-04-15 오후 1 15 15" src="https://github.com/user-attachments/assets/4d24d8c8-4d9e-4530-b1fb-1a9182c1ca5d" />

그림 2.3에서 입력 이미지는 왼쪽에서 들어와 다섯 개의 필터 스택을 통과하여 각각 여러 개의 출력 이미지를 생성합니다.  
각 필터가 끝나면 주석이 달린 대로 이미지의 크기가 줄어듭니다. 마지막 필터 스택에서 생성된 이미지는 4,096개의 요소로 구성된 1D 벡터로 배치되어 각 출력 클래스당 하나씩 1,000개의 출력 확률을 생성하도록 분류됩니다.  
입력 이미지에서 AlexNet 아키텍처를 실행하기 위해 AlexNet 클래스의 인스턴스를 만들 수 있습니다. 

At this point, alexnet is an object that can run the AlexNet architecture. It’s not essential for us to understand the details of this architecture for now. For the time being, AlexNet is just an opaque object that can be called like a function. By providing alexnet with some precisely sized input data (we’ll see shortly what this input data should be), we will run a forward pass through the network. That is, the input will run through the first set of neurons, whose outputs will be fed to the next set of neurons, all the way to the final output. Practically speaking, assuming we have an input object of the right type, we can run the forward pass with output = alexnet(input). But if we did that, we would be feeding data through the whole network to produce … garbage! That’s because the network is uninitialized: its weights, the numbers by which inputs are added and multiplied, have not been trained on anything—the network itself is a blank (or rather, random) slate. We’d need to either train it from scratch or load weights from prior training, which we’ll do now. To this end, let’s go back to the models module. We learned that the uppercase names correspond to classes that implement popular architectures for computer vision. The lowercase names, on the other hand, are functions that instantiate models with predefined numbers of layers and units and optionally download and load pre- trained weights into them. Note that there’s nothing essential about using one of these functions: they just make it convenient to instantiate the model with a number of layers and units that matches how the pretrained networks were built.

우리는 네트워크를 통해 순방향 패스를 실행할 것입니다. 즉, 입력은 첫 번째 뉴런 세트를 통해 실행되며, 뉴런의 출력은 다음 뉴런 세트에 공급되어 최종 출력까지 전달됩니다.  
실제로 적절한 유형의 입력 객체가 있다고 가정하면 출력 = alexnet(입력)으로 순방향 패스를 실행할 수 있습니다. 하지만 그렇게 한다면 전체 네트워크를 통해 데이터를 공급하여 ... 쓰레기를 생성할 수 있습니다!  
네트워크가 초기화되지 않았기 때문입니다: 가중치, 즉 입력을 더하고 곱하는 숫자는 아무것도 학습되지 않았기 때문에 네트워크 자체는 빈(또는 무작위) 슬레이트입니다.

이제 처음부터 다시 훈련하거나 이전 훈련에서 가중치를 로드해야 합니다.  
이를 위해 모델 모듈로 돌아가 보겠습니다. 대문자 이름은 컴퓨터 비전을 위한 인기 있는 아키텍처를 구현하는 클래스에 해당한다는 것을 배웠습니다.  
반면 소문자 이름은 미리 정의된 수의 레이어와 단위로 모델을 인스턴스화하고 선택적으로 사전 훈련된 가중치를 다운로드하여 로드하는 함수입니다.  
이러한 함수 중 하나를 사용하는 데 필수적인 것은 아니며, 사전 훈련된 네트워크가 구축된 방식과 일치하는 여러 레이어와 단위로 모델을 인스턴스화하는 것이 편리하기 때문입니다.

### 2.1.3 ResNet
Using the resnet101 function, we’ll now instantiate a 101-layer convolutional neural network. Just to put things in perspective, before the advent of residual networks in 2015, achieving stable training at such depths was considered extremely hard. Resid- ual networks pulled a trick that made it possible, and by doing so, beat several bench- marks in one sweep that year. Let’s create an instance of the network now. We’ll pass an argument that will instruct the function to download the weights of resnet101 trained on the ImageNet dataset, with 1.2 million images and 1,000 categories:

이제 resnet101 함수를 사용하여 101층 합성곱 신경망을 구현해 보겠습니다.  
참고로, 2015년 잔차 신경망이 등장하기 전에는 이러한 깊이에서 안정적인 훈련을 달성하는 것이 매우 어려웠습니다.  
잔차 신경망은 이를 가능하게 하는 트릭을 사용했고, 이를 통해 그해 한 번의 스윕으로 여러 벤치마크를 달성했습니다.  
이제 네트워크의 인스턴스를 만들어 보겠습니다. 120만 개의 이미지와 1,000개의 카테고리로 구성된 ImageNet 데이터셋에서 훈련된 resnet101의 가중치를 다운로드하도록 함수에 지시하는 논거를 제시하겠습니다:

While we’re staring at the download progress, we can take a minute to appreciate that resnet101 sports 44.5 million parameters—that’s a lot of parameters to optimize automatically!

다운로드 진행 상황을 주시하는 동안, resnet101 4450만 개의 매개변수를 파악하는 데 시간을 할애할 수 있습니다. 이는 자동으로 최적화하기 위한 많은 매개변수입니다!

### 2.1.4 Ready, set, almost run
좋아요, 방금 무엇을 얻었나요? 궁금하기 때문에 resnet101이 어떻게 생겼는지 살펴보겠습니다. 반환된 모델의 값을 출력하면 됩니다.  
이렇게 하면 2.3에서 보았던 것과 같은 종류의 정보를 텍스트로 표현하여 네트워크 구조에 대한 세부 정보를 얻을 수 있습니다.  
지금은 정보 과부하가 걸리겠지만, 책을 진행하면서 이 코드가 우리에게 무엇을 알려주는지 이해하는 능력을 높일 수 있습니다:

What we are seeing here is modules, one per line. Note that they have nothing in common with Python modules: they are individual operations, the building blocks of a neural network. They are also called layers in other deep learning frameworks. If we scroll down, we’ll see a lot of Bottleneck modules repeating one after the other (101 of them!), containing convolutions and other modules. That’s the anat- omy of a typical deep neural network for computer vision: a more or less sequential cascade of filters and nonlinear functions, ending with a layer (fc) producing scores for each of the 1,000 output classes (out_features). The resnet variable can be called like a function, taking as input one or more images and producing an equal number of scores for each of the 1,000 ImageNet classes. Before we can do that, however, we have to preprocess the input images so they are the right size and so that their values (colors) sit roughly in the same numerical range. In order to do that, the torchvision module provides transforms, which allow us to quickly define pipelines of basic preprocessing functions:

여기서 우리가 보고 있는 모듈은 한 줄당 하나씩입니다.  
파이썬 모듈과 공통점이 없다는 점에 유의하세요: 이들은 신경망의 구성 요소인 개별 연산입니다. 다른 딥러닝 프레임워크에서는 레이어라고도 합니다.  
아래로 스크롤하면 컨볼루션 및 기타 모듈을 포함하는 많은 병목 모듈이 차례로 반복되는 것을 볼 수 있습니다(그 중 101개!).  
이것이 바로 컴퓨터 비전을 위한 일반적인 딥러닝 신경망의 해부학적 구조입니다: 필터와 비선형 함수의 다소 순차적인 연쇄로, 레이어(fc)가 1,000개의 출력 클래스(out_feature) 각각에 대한 점수를 생성하는 것으로 끝납니다.  
resnet 변수는 함수와 같이 하나 이상의 이미지를 입력으로 받아 1,000개의 ImageNet 클래스 각각에 대해 동일한 수의 점수를 생성하는 것입니다.  
하지만 이를 수행하기 전에 입력 이미지가 적절한 크기이고 값(색상)이 대략 동일한 수치 범위에 있도록 전처리해야 합니다.  
이를 위해 토치비전 모듈은 변환을 제공하여 기본 전처리 함수의 파이프라인을 빠르게 정의할 수 있게 합니다:

In this case, we defined a preprocess function that will scale the input image to 256 × 256, crop the image to 224 × 224 around the center, transform it to a tensor (a PyTorch multidimensional array: in this case, a 3D array with color, height, and width), and normalize its RGB (red, green, blue) components so that they have defined means and standard deviations. These need to match what was presented to the network during training, if we want the network to produce meaningful answers. We’ll go into more depth about transforms when we dive into making our own image- recognition models in section 7.1.3. We can now grab a picture of our favorite dog (say, bobby.jpg from the GitHub repo), preprocess it, and then see what ResNet thinks of it. We can start by loading an image from the local filesystem using Pillow (https://pillow.readthedocs.io/en/stable), an image-manipulation module for Python:

이 경우, 우리는 입력 이미지를 256 × 256으로 스케일링하고, 이미지를 중심을 중심으로 224 × 224로 잘라 텐서(PyTorch 다차원 배열: 이 경우 색상, 높이, 너비를 가진 3D 배열)로 변환하고, RGB(적색, 녹색, 파란색) 구성 요소를 정규화하여 평균과 표준 편차를 정의하는 전처리 함수를 정의했습니다.  
네트워크가 의미 있는 답변을 제공하려면 이 함수들이 훈련 중에 네트워크에 제시된 내용과 일치해야 합니다.  
섹션 7.1.3에서 우리만의 이미지 인식 모델을 만들 때 변환에 대해 더 자세히 설명하겠습니다.  
이제 GitHub 레포에서 좋아하는 강아지(예: bobby.g jp)의 사진을 가져와서 전처리한 다음 ResNet이 어떻게 생각하는지 확인할 수 있습니다.  
Pillow(https://pillow.readthedocs.io/en/stable), 에서 Python용 이미지 manip 모듈을 사용하여 로컬 파일 시스템에서 이미지를 로드하는 것부터 시작할 수 있습니다:

If we were following along from a Jupyter Notebook, we would do the following to see the picture inline (it would be shown where the <PIL.JpegImagePlugin… is in the following):

Jupyter 노트북을 따라가면 다음과 같이 사진을 인라인으로 볼 수 있습니다(<PIL.JpegImagePlugin...>의 위치가 다음과 같이 표시됩니다):
```
<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720 at
0x1B1601360B8>
```

Next, we can pass the image through our preprocessing pipeline:

전처리 진행하고,

Then we can reshape, crop, and normalize the input tensor in a way that the network expects. We’ll understand more of this in the next two chapters; hold tight for now:

그런 다음 네트워크가 예상하는 방식으로 입력 텐서를 재구성하고, 자르며, 정규화할 수 있습니다. 이에 대해서는 다음 두 장에서 자세히 설명하겠지만, 지금은 꽉 잡고 있습니다:

### 2.1.5 Run!
The process of running a trained model on new data is called inference in deep learn- ing circles. In order to do inference, we need to put the network in eval mode:

새로운 데이터에 대해 학습된 모델을 실행하는 과정을 딥러닝 서클에서 추론이라고 합니다.  
추론을 수행하려면 네트워크를 평가 모드로 전환해야 합니다:

If we forget to do that, some pretrained models, like batch normalization and dropout, will not produce meaningful answers, just because of the way they work internally. Now that eval has been set, we’re ready for inference:

이를 잊어버리면 배치 정규화 및 드롭아웃과 같은 일부 사전 학습된 모델은 내부적으로 작동하는 방식 때문에 의미 있는 답변을 제공하지 못합니다.  
이제 평가가 설정되었으므로 추론할 준비가 되었습니다:

A staggering set of operations involving 44.5 million parameters has just happened, producing a vector of 1,000 scores, one per ImageNet class. That didn’t take long, did it? We now need to find out the label of the class that received the highest score. This will tell us what the model saw in the image. If the label matches how a human would describe the image, that’s great! It means everything is working. If not, then either some- thing went wrong during training, or the image is so different from what the model expects that the model can’t process it properly, or there’s some other similar issue. To see the list of predicted labels, we will load a text file listing the labels in the same order they were presented to the network during training, and then we will pick out the label at the index that produced the highest score from the network. Almost all models meant for image recognition have output in a form similar to what we’re about to work with. Let’s load the file containing the 1,000 labels for the ImageNet dataset classes:

4,450만 개의 매개변수와 관련된 놀라운 연산 세트가 방금 발생하여 ImageNet 클래스당 하나씩 1,000개의 점수 벡터를 생성했습니다. 그리 오래 걸리지 않았죠?  
이제 가장 높은 점수를 받은 클래스의 레이블을 찾아야 합니다. 그러면 모델이 이미지에서 무엇을 보았는지 알 수 있습니다.  
레이블이 사람이 이미지를 설명하는 방식과 일치한다면 정말 좋은 일입니다! 모든 것이 작동한다는 뜻입니다.  
그렇지 않다면 훈련 중에 문제가 발생했거나 모델이 기대하는 것과 이미지가 너무 다르거나 모델이 제대로 처리할 수 없는 다른 유사한 문제가 발생한 것입니다.  
예측된 레이블 목록을 보려면 훈련 중에 네트워크에 제시된 레이블과 동일한 순서로 나열된 텍스트 파일을 로드한 다음 네트워크에서 가장 높은 점수를 생성한 인덱스의 레이블을 선택합니다.  
이미지 인식을 위한 거의 모든 모델은 우리가 작업하려는 것과 유사한 형태로 출력됩니다. ImageNet 데이터셋 클래스에 사용할 1,000개의 레이블이 포함된 파일을 로드해 보겠습니다:

At this point, we need to determine the index corresponding to the maximum score in the out tensor we obtained previously. We can do that using the max function in PyTorch, which outputs the maximum value in a tensor as well as the indices where that maximum value occurred:

이 시점에서 이전에 얻은 아웃 텐서의 최대 점수에 해당하는 인덱스를 결정해야 합니다.  
텐서의 최대값과 해당 최대값이 발생한 인덱스를 출력하는 PyTorch의 최대값 함수를 사용하여 이를 수행할 수 있습니다:

We can now use the index to access the label. Here, index is not a plain Python num- ber, but a one-element, one-dimensional tensor (specifically, tensor([207])), so we need to get the actual numerical value to use as an index into our labels list using index[0]. We also use torch.nn.functional.softmax (http://mng.bz/BYnq) to nor- malize our outputs to the range [0, 1], and divide by the sum. That gives us something roughly akin to the confidence that the model has in its prediction. In this case, the model is 96% certain that it knows what it’s looking at is a golden retriever:

이제 인덱스를 사용하여 레이블에 액세스할 수 있습니다. 여기서 인덱스는 단순한 파이썬 숫자가 아니라 1차원 element, 1차원 텐서(specif로 텐서([207))이므로 인덱스[0]를 사용하여 레이블 목록에 인덱스로 사용할 실제 숫자 값을 가져와야 합니다.  
또한 torch.n.functional.softmax(http://mng.bz/BYnq) 을 사용하여 출력을 [0, 1] 범위로 정규화하고 합으로 나눕니다.  
이는 모델이 예측에 대해 가지고 있는 신뢰도와 거의 유사한 것을 제공합니다. 이 경우 모델이 보고 있는 것이 골든 리트리버라는 것을 96% 확신할 수 있습니다:

Uh oh, who’s a good boy? Since the model produced scores, we can also find out what the second best, third best, and so on were. To do this, we can use the sort function, which sorts the values in ascending or descending order and also provides the indices of the sorted values in the original array:

모델이 점수를 산출했기 때문에 차선책, 차선책 등이 무엇인지 알아낼 수 있습니다.  
이를 위해 값을 오름차순 또는 내림차순으로 정렬하고 원본 배열의 정렬된 값의 인덱스도 제공하는 정렬 함수를 사용할 수 있습니다:

We see that the first four are dogs (redbone is a breed; who knew?), after which things start to get funny. The fifth answer, “tennis ball,” is probably because there are enough pictures of tennis balls with dogs nearby that the model is essentially saying, “There’s a 0.1% chance that I’ve completely misunderstood what a tennis ball is.” This is a great example of the fundamental differences in how humans and neural networks view the world, as well as how easy it is for strange, subtle biases to sneak into our data. Time to play! We can go ahead and interrogate our network with random images and see what it comes up with. How successful the network will be will largely depend on whether the subjects were well represented in the training set. If we present an image containing a subject outside the training set, it’s quite possible that the network will come up with a wrong answer with pretty high confidence. It’s useful to experi- ment and get a feel for how a model reacts to unseen data. We’ve just run a network that won an image-classification competition in 2015. It learned to recognize our dog from examples of dogs, together with a ton of other real-world subjects. We’ll now see how different architectures can achieve other kinds of tasks, starting with image generation.

처음 네 가지는 개(레드본은 품종인데 누가 알았나요?)이고 그 후에는 재미있어지기 시작합니다.  
다섯 번째 대답인 '테니스공'은 아마도 모델이 "테니스공이 무엇인지 완전히 오해했을 확률은 0.1%입니다."라고 말할 정도로 근처에 개가 있는 테니스공 사진이 충분하기 때문일 것입니다.  
이는 인간과 신경망이 세상을 바라보는 방식의 근본적인 차이와 이상하고 미묘한 편견이 데이터에 몰래 침투하는 것이 얼마나 쉬운지를 잘 보여주는 좋은 예입니다.  
플레이할 시간입니다! 무작위 이미지로 네트워크를 조사하여 어떤 결과가 나오는지 확인할 수 있습니다.  
네트워크가 얼마나 성공적으로 작동할지는 피험자가 훈련 세트에 잘 표현되었는지에 따라 크게 달라집니다.  
훈련 세트 외부에 피험자가 포함된 이미지를 제시하면 네트워크가 꽤 높은 신뢰도로 오답을 내놓을 가능성이 높습니다.  
보이지 않는 데이터에 대한 모델의 반응을 직접 체험하고 느끼는 것이 유용합니다.  
우리는 2015년에 이미지 분류 대회에서 우승한 네트워크를 운영했습니다.  
개의 예시와 수많은 다른 실제 피험자로부터 우리 개를 인식하는 법을 배웠습니다.  
이제 이미지 생성부터 다양한 아키텍처가 다른 종류의 작업을 어떻게 수행할 수 있는지 살펴보겠습니다.

## 2.2 A pretrained model that fakes it until it makes it
### 2.2.1 The GAN game
GAN은 생성적 적대 신경망을 의미하며, 여기서 적대적이란 두 네트워크가 서로를 능가하기 위해 경쟁하는 것을 의미합니다.

우리의 가장 중요한 목표는 가짜로 인식할 수 없는 이미지 클래스의 합성 예제를 생성하는 것입니다.  
합법적인 예제와 혼합될 때 숙련된 검사관은 어떤 것이 진짜인지, 어떤 것이 위조품인지 판단하는 데 어려움을 겪을 수 있습니다.  
생성기 네트워크는 우리 시나리오에서 화가의 역할을 맡아 임의의 입력에서 시작하여 현실적으로 보이는 이미지를 생성하는 임무를 맡습니다.  
판별기 네트워크는 비도덕적인 미술 검사관으로, 주어진 이미지가 생성기에 의해 조작되었는지 아니면 실제 이미지 세트에 속하는지를 구분해야 합니다.  
이 두 네트워크 설계는 대부분의 딥러닝 아키텍처에서는 비정형적이지만, GAN 게임을 구현하는 데 사용될 경우 놀라운 결과를 초래할 수 있습니다.

그림 2.5는 무슨 일이 일어나고 있는지 대략적인 그림을 보여줍니다.

<img width="929" alt="스크린샷 2025-04-15 오후 1 57 04" src="https://github.com/user-attachments/assets/cbf93878-be6e-419e-9988-43fca65a155e" />

### 2.2.2 CycleGAN
An interesting evolution of this concept is the CycleGAN. A CycleGAN can turn images of one domain into images of another domain (and back), without the need for us to explicitly provide matching pairs in the training set. In figure 2.6, we have a CycleGAN workflow for the task of turning a photo of a horse into a zebra, and vice versa. Note that there are two separate generator net- works, as well as two distinct discriminators.

이 개념의 흥미로운 발전 중 하나는 CycleGAN입니다. CycleGAN은 훈련 세트에서 일치하는 쌍을 명시적으로 제공할 필요 없이 한 도메인의 이미지를 다른 도메인의 이미지(그리고 뒤로)로 변환할 수 있습니다.  
그림 2.6에서는 말의 사진을 얼룩말로 변환하는 작업을 위한 CycleGAN 워크플로우를 보여줍니다. 두 개의 별도 생성기 네트워크 작업과 두 개의 구별되는 판별기가 있다는 점에 유의하세요.

<img width="929" alt="스크린샷 2025-04-15 오후 1 58 25" src="https://github.com/user-attachments/assets/f0110b39-3f8a-4c6f-8cf2-160cc3315387" />

As the figure shows, the first generator learns to produce an image conforming to a target distribution (zebras, in this case) starting from an image belonging to a different distribution (horses), so that the discriminator can’t tell if the image produced from a horse photo is actually a genuine picture of a zebra or not. At the same time—and here’s where the Cycle prefix in the acronym comes in—the resulting fake zebra is sent through a different generator going the other way (zebra to horse, in our case), to be judged by another discriminator on the other side. Creating such a cycle stabilizes the training process considerably, which addresses one of the original issues with GANs. The fun part is that at this point, we don’t need matched horse/zebra pairs as ground truths (good luck getting them to match poses!). It’s enough to start from a collection of unrelated horse images and zebra photos for the generators to learn their task, going beyond a purely supervised setting. The implications of this model go even further than this: the generator learns how to selectively change the appearance of objects in the scene without supervision about what’s what. There’s no signal indicating that manes are manes and legs are legs, but they get translated to something that lines up with the anatomy of the other animal.

그림에서 볼 수 있듯이, 첫 번째 생성기는 다른 분포(말)에 속하는 이미지에서 시작하여 목표 분포(제브라, 이 경우에는 제브라)에 맞는 이미지를 생성하는 방법을 배웁니다.  
이를 통해 판별자는 말 사진에서 생성된 이미지가 실제로 진짜 제브라인지 아닌지 알 수 없게 됩니다.  
동시에, 약어의 사이클 접두사가 여기에 포함됩니다. 결과적으로 가짜 제브라는 다른 생성기를 통해 다른 방향(제브라에서 말로, 우리의 경우에는 제브라에서 말로)으로 보내져 반대편의 다른 판별기에 의해 판단됩니다.  
이러한 사이클을 생성하면 훈련 과정이 상당히 안정화되어 GAN의 원래 문제 중 하나를 해결할 수 있습니다.  
재미있는 점은 이 시점에서 말/제브라 쌍을 실제 진실로 맞출 필요가 없다는 것입니다(포즈를 맞추게 되길 행운을 빕니다!).  
생성기가 자신의 임무를 학습하기 위해서는 관련 없는 말 이미지와 제브라 사진 모음에서 시작하는 것만으로도 충분합니다.  
이 모델의 의미는 이보다 더 발전합니다: 생성기는 무엇이 무엇인지에 대한 감독 없이 장면 속 물체의 모양을 선택적으로 변경하는 방법을 학습합니다.  
갈기는 갈기이고 다리는 다리라는 신호는 없지만, 다른 동물의 해부학과 일치하는 무언가로 번역됩니다.

### 2.2.3 A network that turns horses into zebras
CycleGAN 네트워크는 ImageNet 데이터셋에서 추출한 (관련 없는) 말 이미지와 얼룩말 이미지 데이터셋을 기반으로 훈련되었습니다.  
네트워크는 하나 이상의 말 이미지를 가져와서 모두 얼룩말로 변환하고 나머지 이미지는 가능한 한 수정하지 않은 채로 만드는 방법을 배웁니다.  
인류는 지난 몇 천 년 동안 말을 얼룩말로 변환하는 도구에 대해 숨을 쉬지 않았지만, 이 작업은 이러한 아키텍처가 복잡한 실제 프로세스를 원격 감독을 통해 모델링할 수 있는 능력을 보여줍니다.

미리 훈련된 CycleGAN을 사용하면 네트워크(이 경우 생성기)가 어떻게 구현되는지 한 걸음 더 가까워지고 살펴볼 수 있는 기회를 얻을 수 있습니다. 우리는 오랜 친구인 ResNet을 사용할 것입니다. 

netG 모델이 생성되었지만, 여기에는 무작위 가중치가 포함되어 있습니다.  
우리는 이전에 horse2 zebra 데이터셋에서 사전 학습된 생성기 모델을 실행할 것이라고 언급했습니다.  
이 생성기 모델은 각각 말과 얼룩말의 두 세트인 1068개와 1335개의 이미지를 포함하고 있습니다.  
데이터셋은 http://mng.bz/8pKP 에서 확인할 수 있습니다.  
모델의 가중치는 .pth 파일에 저장되었으며, 이는 모델의 텐서 매개변수에 대한 피클 파일에 불과합니다.  
우리는 이를 모델의 load_state_dict 방법을 사용하여 ResNetGenerator에 로드할 수 있습니다:


At this point, netG has acquired all the knowledge it achieved during training. Note that this is fully equivalent to what happened when we loaded resnet101 from torch- vision in section 2.1.3; but the torchvision.resnet101 function hid the loading from us. Let’s put the network in eval mode, as we did for resnet101:

resnet101에서와 마찬가지로 네트워크를 평가 모드로 설정해 보겠습니다:

이전에 했던 것처럼 모델을 출력하면 실제로는 상당히 응축되어 있다는 것을 알 수 있습니다.  
이미지를 가져와서 픽셀을 보고 그 안에 있는 하나 이상의 말을 인식하고, 픽셀의 값을 개별적으로 수정하여 나오는 것이 신뢰할 수 있는 얼룩말처럼 보이도록 합니다.  
출력물(또는 소스 코드에서는)에서 얼룩말과 같은 것을 인식하지 못할 것입니다: 그 안에는 얼룩말과 같은 것이 없기 때문입니다.  
네트워크는 저울로 구성되어 있으며, 무게에 주스가 들어 있습니다. 우리는 말의 무작위 이미지를 로드하고 발전기가 무엇을 생성하는지 확인할 준비가 되어 있습니다. 먼저 PIL과 토치비전을 가져와야 합니다:

Then we define a few input transformations to make sure data enters the network with the right shape and size:

그런 다음 데이터가 올바른 모양과 크기로 네트워크에 진입할 수 있도록 몇 가지 입력 변환을 정의합니다:

학습 과정이 인간이 수만 마리의 말을 묘사하거나 수천 마리의 얼룩말 줄무늬를 인간이 직접 포토샵한 직접적인 감독을 거치지 않았다는 점을 반복하고 있습니다. 

## 2.3 A pretrained network that describes scenes
자연어와 관련된 모델을 직접 경험하기 위해, 우리는 Ruotian Luo.2에서 아낌없이 제공하는 사전 훈련된 이미지 캡셔닝 모델을 사용할 것입니다.  
이 모델은 Andrej Karpathy의 NeuralTalk2 모델을 구현한 것입니다.  
자연스러운 이미지와 함께 제시되면, 그림 2.9와 같이 영어로 장면을 설명하는 캡션을 생성합니다.  

<img width="943" alt="스크린샷 2025-04-15 오후 2 21 07" src="https://github.com/user-attachments/assets/f1ad2f97-85b2-4648-ae8b-1b558ce0270b" />

이 모델은 대규모 이미지 데이터셋과 쌍을 이루는 문장 설명에 대해 훈련됩니다: 예를 들어, "Tabby 고양이는 나무 테이블에 기대어 있고, 한쪽 발은 레이저 마우스에, 다른 쪽 발은 검은색 노트북에 있습니다."3  
이 캡션 모델은 두 개의 연결된 반쪽을 가지고 있습니다. 모델의 전반부는 장면의 "설명적인" 수치 표현(Tabby 고양이, 레이저 마우스, 발)을 생성하는 방법을 학습하는 네트워크로, 이를 후반부에 입력으로 받아들입니다.  
후반부는 이러한 수치 설명을 결합하여 일관된 문장을 생성하는 순환 신경망입니다. 모델의 두 절반은 이미지 캡션 쌍에 대해 함께 훈련됩니다.  
모델의 후반부는 후속 전진 패스에서 출력(개별 단어)을 생성하기 때문에 순환이라고 불립니다.  
각 전진 패스에 대한 입력에는 이전 전진 패스의 출력이 포함됩니다.  
이는 문장이나 일반적으로 시퀀스를 다룰 때 예상할 수 있듯이, 이전에 생성된 단어에 다음 단어의 의존성을 생성합니다.

```
3. Deep Visual-Semantic Alignments for Generating Image Descriptions
https://m.blog.naver.com/data_flow/221693091492
```
### 2.3.1 NeuralTalk2
```
https://github.com/deep-learning-with-pytorch/ImageCaptioning.pytorch?tab=readme-ov-file#generate-image-captions
```

## 2.4 Torch Hub
PyTorch 1.0은 Torch Hub의 도입을 목격했는데, 이는 저자들이 사전 학습된 가중치가 있든 없든 GitHub에 모델을 게시하고 PyTorch가 이해하는 인터페이스를 통해 노출할 수 있는 메커니즘입니다.  
이를 통해 제3자로부터 사전 학습된 모델을 로드하는 것이 TorchVision 모델을 로드하는 것만큼 쉬워집니다.  
저자가 Torch Hub 메커니즘을 통해 모델을 게시하는 데 필요한 것은 GitHub 저장소의 루트 디렉토리에 hubconf.py 이라는 파일을 배치하는 것뿐입니다. 파일의 구조는 매우 간단합니다:

흥미로운 사전 학습된 모델을 찾는 과정에서 이제 hubconf.py 을 포함한 GitHub 리포지토리를 검색할 수 있으며, Torch.hub 모듈을 사용하여 로드할 수 있다는 것을 바로 알 수 있습니다.

<img width="943" alt="스크린샷 2025-04-15 오후 2 36 10" src="https://github.com/user-attachments/assets/1bdbb1dd-50e6-4318-b969-a106d65d9288" />

이렇게 하면 파이토치/비전 레포의 마스터 브랜치 스냅샷과 가중치를 로컬 디렉토리(홈 디렉토리의 .torch/hub 기본값)에 다운로드하고, 인스턴스화된 모델을 반환하는 resnet18 진입점 함수를 실행할 수 있습니다.

Torch Hub는 이 글을 쓰는 시점에서 꽤 새로운 모델이며, 이렇게 공개된 모델은 몇 가지에 불과합니다.  
Google에서 "github.com hubconf.py "을 검색하면 이를 확인할 수 있습니다. 앞으로 더 많은 저자들이 이 채널을 통해 모델을 공유함에 따라 목록이 더 커질 것으로 기대됩니다.

## 2.5 Conclusion
이 책은 완전한 PyTorch API를 다루거나 딥러닝 아키텍처를 검토하는 것에 중점을 두지 않고, 이러한 빌딩 블록에 대한 실무 지식을 구축할 것입니다.  
이렇게 하면 탄탄한 기초 위에 뛰어난 온라인 문서 작성 및 저장소를 소비할 수 있을 것입니다.  
다음 장을 시작으로, PyTorch를 사용하여 이 장에서 설명한 것처럼 컴퓨터 기술을 처음부터 가르칠 수 있는 여정을 시작하겠습니다.  
또한 사전 학습된 네트워크에서 시작하여 새로운 데이터에 대해 처음부터 시작하지 않고 미세 조정하는 것이 우리가 가진 데이터 포인트가 많지 않을 때 문제를 해결하는 효과적인 방법임을 배울 것입니다. 

