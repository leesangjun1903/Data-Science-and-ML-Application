# Introducing deep learning and the PyTorch Library
This chapter covers  How deep learning changes our approach to machine learning  Understanding why PyTorch is a good fit for deep learning  Examining a typical deep learning project  The hardware you’ll need to follow along with the examples

이 장에서는 다음을 다룹니다  
 딥러닝이 머신러닝에 대한 우리의 접근 방식을 어떻게 변화시키는지  
 PyTorch가 딥러닝에 적합한 이유 이해하기  
 일반적인 딥러닝 프로젝트 살펴보기  
 예제와 함께 따라야 할 하드웨어

The poorly defined term artificial intelligence covers a set of disciplines that have been subjected to a tremendous amount of research, scrutiny, confusion, fantastical hype, and sci-fi fearmongering. Reality is, of course, far more sanguine. It would be disingenuous to assert that today’s machines are learning to “think” in any human sense of the word. Rather, we’ve discovered a general class of algorithms that are able to approximate complicated, nonlinear processes very, very effectively, which we can use to automate tasks that were previously limited to humans. For example, at https://talktotransformer.com, a language model called GPT-2 can generate coherent paragraphs of text one word at a time. When we fed it this very paragraph, it produced the following:

오늘날의 기계가 인간의 어떤 의미에서든 '생각'하는 법을 배우고 있다고 주장하는 것은 무의미할 것입니다.  
오히려 복잡하고 비선형적인 프로세스를 매우 효과적으로 근사할 수 있는 일반적인 알고리즘 클래스를 발견했으며, 이를 통해 이전에는 인간에게만 국한되었던 작업을 자동화할 수 있습니다.  
예를 들어, https://talktotransformer.com 에서 GPT-2라는 언어 모델은 한 번에 한 단어씩 일관된 텍스트 단락을 생성할 수 있습니다. 바로 이 단락을 입력했을 때 다음과 같은 결과가 나왔습니다:

Next we’re going to feed in a list of phrases from a corpus of email addresses, and see if the program can parse the lists as sentences. Again, this is much more complicated and far more complex than the search at the beginning of this post, but hopefully helps you understand the basics of constructing sentence structures in various programming languages.

다음으로 이메일 주소 코퍼스에서 구문 목록을 입력하고 프로그램이 목록을 문장으로 구문 분석할 수 있는지 확인해 보겠습니다.  
다시 말하지만, 이 글은 이 글의 시작 부분에서 검색하는 것보다 훨씬 더 복잡하고 복잡하지만 다양한 프로그래밍 언어로 문장 구조를 구성하는 기본을 이해하는 데 도움이 되기를 바랍니다.

That general class of algorithms we’re talking about falls under the AI subcategory of deep learning, which deals with training mathematical entities named deep neural net- works by presenting instructive examples. Deep learning uses large amounts of data to approximate complex functions whose inputs and outputs are far apart, like an input image and, as output, a line of text describing the input; or a written script as input and a natural-sounding voice reciting the script as output; or, even more simply, associating an image of a golden retriever with a flag that tells us “Yes, a golden retriever is present.” This kind of capability allows us to create programs with functionality that was, until very recently, exclusively the domain of human beings.

우리가 이야기하는 일반적인 알고리즘 클래스는 딥러닝의 AI 하위 범주에 속하며, 이는 딥 뉴럴 네트 워크라는 수학적 실체를 훈련시켜 교훈적인 예제를 제시하는 것을 다룹니다.  
딥러닝은 입력 이미지와 출력, 입력을 설명하는 텍스트 라인, 또는 입력으로 작성된 스크립트와 출력으로 스크립트를 낭송하는 자연스러운 음성, 또는 더 간단히 말해 골든 리트리버의 이미지를 "예, 골든 리트리버가 존재합니다"라고 말하는 플래그와 연관시키는 등 입력과 출력이 멀리 떨어져 있는 복잡한 함수를 근사하기 위해 대량의 데이터를 사용합니다.  
이러한 기능을 통해 최근까지 인간의 영역이었던 기능을 갖춘 프로그램을 만들 수 있습니다.

## 1.1 The deep learning revolution
To appreciate the paradigm shift ushered in by this deep learning approach, let’s take a step back for a bit of perspective. Until the last decade, the broader class of systems that fell under the label machine learning relied heavily on feature engineering. Features are transformations on input data that facilitate a downstream algorithm, like a classifier, to produce correct outcomes on new data. Feature engineering consists of coming up with the right transformations so that the downstream algorithm can solve a task. For instance, in order to tell ones from zeros in images of handwritten digits, we would come up with a set of filters to estimate the direction of edges over the image, and then train a classifier to predict the correct digit given a distribution of edge directions. Another useful feature could be the number of enclosed holes, as seen in a zero, an eight, and, particularly, loopy twos.

지난 10년 동안 머신러닝이라는 레이블에 속하는 광범위한 시스템은 피처 엔지니어링에 크게 의존했습니다.  
피처는 분류기와 같은 다운스트림 알고리즘이 새로운 데이터에서 올바른 결과를 도출할 수 있도록 하는 입력 데이터의 변환입니다.  
피처 엔지니어링은 다운스트림 알고리즘이 작업을 해결할 수 있도록 올바른 변환을 고안하는 것으로 구성됩니다.  
예를 들어, 손으로 쓴 숫자 이미지에서 0과 1을 구분하기 위해 이미지 위의 엣지 방향을 추정하는 필터 세트를 고안한 다음 엣지 방향 분포에 따라 올바른 숫자를 예측하는 분류기를 훈련시킵니다.  
또 다른 유용한 기능은 0, 8, 특히 루프형 2에서 볼 수 있듯이 둘러싸인 구멍의 수입니다.

Deep learning, on the other hand, deals with finding such representations auto- matically, from raw data, in order to successfully perform a task. In the ones versus zeros example, filters would be refined during training by iteratively looking at pairs of examples and target labels. This is not to say that feature engineering has no place with deep learning; we often need to inject some form of prior knowledge in a learn- ing system. However, the ability of a neural network to ingest data and extract useful representations on the basis of examples is what makes deep learning so powerful. The focus of deep learning practitioners is not so much on handcrafting those repre- sentations, but on operating on a mathematical entity so that it discovers representa- tions from the training data autonomously. Often, these automatically created features are better than those that are handcrafted! As with many disruptive technolo- gies, this fact has led to a change in perspective. On the left side of figure 1.1, we see a practitioner busy defining engineering fea- tures and feeding them to a learning algorithm; the results on the task will be as good as the features the practitioner engineers. On the right, with deep learning, the raw data is fed to an algorithm that extracts hierarchical features automatically, guided by the optimization of its own performance on the task; the results will be as good as the ability of the practitioner to drive the algorithm toward its goal.

반면에 딥러닝은 작업을 성공적으로 수행하기 위해 원시 데이터에서 이러한 표현을 자동으로 찾는 것을 다룹니다.  
1과 0의 예제에서 필터는 예제와 대상 레이블 쌍을 반복적으로 살펴봄으로써 훈련 중에 정제됩니다.  
그렇다고 해서 feature engineering 이 딥러닝에 적합하지 않다는 뜻은 아니며, 학습 시스템에 어떤 형태로든 사전 지식을 주입해야 하는 경우가 많습니다.  
그러나 딥러닝을 강력하게 만드는 것은 신경망이 데이터를 수집하고 예제를 기반으로 유용한 표현을 추출할 수 있는 능력입니다.  
딥러닝 실무자들의 초점은 이러한 표현을 수작업으로 만드는 것이 아니라 수학적 실체에서 작동하여 훈련 데이터에서 표현을 자율적으로 발견하는 데 있습니다.  
종종 이러한 자동으로 생성된 특징이 수작업으로 만들어진 특징보다 더 나은 경우가 많습니다!  
많은 혁신적인 기술과 마찬가지로 이러한 사실은 관점의 변화를 가져왔습니다.  
그림 1.1의 왼쪽에는 엔지니어링 특징을 정의하고 학습 알고리즘에 제공하느라 바쁜 실무자를 볼 수 있는데, 작업의 결과는 실무자 엔지니어의 특징만큼이나 좋을 것입니다.  
오른쪽에는 딥러닝을 통해 원시 데이터가 작업에 대한 자체 성능 최적화에 따라 계층적 특징을 자동으로 추출하는 알고리즘에 입력되며, 결과는 실무자가 알고리즘을 목표로 이끄는 능력만큼이나 좋을 것입니다.

<img width="815" alt="스크린샷 2025-04-15 오전 10 46 08" src="https://github.com/user-attachments/assets/288e111c-875f-4988-a4c0-d79e86420ff9" />

Starting from the right side in figure 1.1, we already get a glimpse of what we need to execute successful deep learning: 
 We need a way to ingest whatever data we have at hand. 
 We somehow need to define the deep learning machine. 
 We must have an automated way, training, to obtain useful representations and make the machine produce desired outputs.
This leaves us with taking a closer look at this training thing we keep talking about. During training, we use a criterion, a real-valued function of model outputs and reference data, to provide a numerical score for the discrepancy between the desired and actual output of our model (by convention, a lower score is typically better). Training consists of driving the criterion toward lower and lower scores by incrementally modifying our deep learning machine until it achieves low scores, even on data not seen during training.

그림 1.1의 오른쪽에서 시작하여, 우리는 이미 성공적인 딥러닝을 실행하기 위해 필요한 것들을 엿볼 수 있습니다:  
 우리는 당면한 모든 데이터를 수집할 방법이 필요합니다.  
 어떻게든 딥러닝 머신을 정의해야 합니다.  
 우리는 유용한 표현을 얻고 기계가 원하는 출력을 낼 수 있도록 자동화된 방법과 훈련을 제공해야 합니다.  
이렇게 하면 우리가 계속 이야기하는 이 훈련에 대해 자세히 살펴볼 수 있습니다.  
훈련 중에는 모델 출력과 참조 데이터의 실수 값 함수인 기준을 사용하여 원하는 모델 출력과 실제 출력 간의 불일치에 대한 수치 점수를 제공합니다(일반적으로 점수가 낮을수록 더 좋습니다).  
훈련은 훈련 중에 보지 못한 데이터에서도 낮은 점수를 얻을 때까지 딥러닝 머신을 점진적으로 수정하여 기준을 점점 더 낮은 점수로 유도하는 것으로 구성됩니다.

## 1.2 PyTorch for deep learning
PyTorch is a library for Python programs that facilitates building deep learning projects. It emphasizes flexibility and allows deep learning models to be expressed in idiomatic Python. This approachability and ease of use found early adopters in the research community, and in the years since its first release, it has grown into one of the most prominent deep learning tools across a broad range of applications. As Python does for programming, PyTorch provides an excellent introduction to deep learning. At the same time, PyTorch has been proven to be fully qualified for use in professional contexts for real-world, high-profile work. We believe that PyTorch’s clear syntax, streamlined API, and easy debugging make it an excellent choice for introducing deep learning. We highly recommend studying PyTorch for your first deep learning library. Whether it ought to be the last deep learning library you learn is a decision we leave up to you. At its core, the deep learning machine in figure 1.1 is a rather complex mathematical function mapping inputs to an output. To facilitate expressing this function, PyTorch provides a core data structure, the tensor, which is a multidimensional array that shares many similarities with NumPy arrays. Around that foundation, PyTorch comes with features to perform accelerated mathematical operations on dedicated hardware, which makes it convenient to design neural network architectures and train them on individual machines or parallel computing resources.

Python이 프로그래밍에 사용하는 것과 마찬가지로 PyTorch는 딥러닝에 대한 훌륭한 입문서를 제공합니다.  
동시에 PyTorch는 실제 세계에서 주목받는 작업을 위해 전문적인 맥락에서 사용할 수 있는 충분한 자격을 갖춘 것으로 입증되었습니다.  
우리는 PyTorch의 명확한 구문, 간소화된 API, 쉬운 디버깅 덕분에 딥러닝을 도입하는 데 탁월한 선택이라고 믿습니다.  
첫 번째 딥러닝 라이브러리를 위해 PyTorch를 적극적으로 연구할 것을 권장합니다.  
이 라이브러리가 마지막 딥러닝 라이브러리여야 하는지 여부는 우리가 결정할 사항입니다.  
그 핵심에는 그림 1.1의 딥러닝 머신이 입력을 출력으로 매핑하는 다소 복잡한 수학 함수입니다.  
이 함수를 표현하기 위해 PyTorch는 NumPy 배열과 많은 유사점을 공유하는 다차원 배열인 텐서라는 핵심 데이터 구조를 제공합니다.  
그 기반을 바탕으로 PyTorch는 전용 하드웨어에서 가속화된 수학 연산을 수행할 수 있는 기능을 갖추고 있어 신경망 아키텍처를 설계하고 개별 머신이나 병렬 컴퓨팅 리소스에서 훈련하기에 편리합니다.

Deep Learning with PyTorch is organized in three distinct parts. Part 1 covers the foundations, examining in detail the facilities PyTorch offers to put the sketch of deep learning in figure 1.1 into action with code. Part 2 walks you through an end-to-end project involving medical imaging: finding and classifying tumors in CT scans, building on the basic concepts introduced in part 1, and adding more advanced topics. The short part 3 rounds off the book with a tour of what PyTorch offers for deploying deep learning models to production.

PyTorch를 통한 딥러닝은 세 가지 부분으로 구성되어 있습니다.  
1부에서는 PyTorch가 그림 1.1의 딥러닝 스케치를 코드로 구현하기 위해 제공하는 시설을 자세히 살펴봅니다.  
2부에서는 의료 영상과 관련된 엔드투엔드 프로젝트를 안내합니다: CT 스캔에서 종양을 찾아 분류하고, 1부에서 소개한 기본 개념을 바탕으로 더 고급 주제를 추가합니다.  
책의 짧은 3부는 PyTorch가 딥러닝 모델을 프로덕션에 배포하는 데 제공하는 내용을 둘러보는 것으로 마무리됩니다.

Deep learning is a huge space. In this book, we will be covering a tiny part of that space: specifically, using PyTorch for smaller-scope classification and segmentation projects, with image processing of 2D and 3D datasets used for most of the motivating examples. This book focuses on practical PyTorch, with the aim of covering enough ground to allow you to solve real-world machine learning problems, such as in vision, with deep learning or explore new models as they pop up in research literature. Most, if not all, of the latest publications related to deep learning research can be found in the arXiV public preprint repository, hosted at https://arxiv.org. 2

딥러닝은 거대한 공간입니다. 이 책에서는 이러한 공간의 작은 부분을 다룰 것입니다.  
구체적으로, PyTorch를 사용하여 소규모 분류 및 분할 프로젝트를 수행하고, 대부분의 동기 부여 예제에서 2D 및 3D 데이터셋의 이미지 처리를 수행합니다.  
이 책은 실제 PyTorch에 초점을 맞추고 있으며, 시각과 같은 실제 머신러닝 문제를 딥러닝으로 해결하거나 연구 문헌에 등장하는 새로운 모델을 탐구할 수 있는 충분한 기반을 제공하는 것을 목표로 합니다.  
대부분의 경우, 전부는 아니더라도, 딥러닝 연구와 관련된 최신 출판물은 https://arxiv.org 에서 호스팅되는 arXiV 공개 사전 인쇄 저장소에서 찾을 수 있습니다. 2

## 1.3 Why PyTorch?
PyTorch는 숫자, 벡터, 행렬 또는 배열을 일반적으로 저장할 수 있는 데이터 유형인 텐서를 제공합니다. 또한, PyTorch는 이를 작동시키기 위한 기능도 제공합니다.

PyTorch also has a compelling story for the transition from research and development into production. While it was initially focused on research workflows, PyTorch has been equipped with a high-performance C++ runtime that can be used to deploy models for inference without relying on Python, and can be used for designing and training models in C++. It has also grown bindings to other languages and an inter- face for deploying to mobile devices. These features allow us to take advantage of PyTorch’s flexibility and at the same time take our applications where a full Python runtime would be hard to get or would impose expensive overhead. Of course, claims of ease of use and high performance are trivial to make. We hope that by the time you are in the thick of this book, you’ll agree with us that our claims here are well founded.

처음에는 연구 워크플로우에 중점을 두었지만, PyTorch는 Python에 의존하지 않고 추론 모델을 배포하는 데 사용할 수 있는 고성능 C++ 런타임을 갖추고 있으며, C++에서 모델을 설계하고 훈련하는 데 사용할 수 있습니다.

### 1.3.1 The deep learning competitive landscape

## 1.4 An overview of how PyTorch supports deep learning projects
PyTorch는 무엇보다도 딥러닝 라이브러리이며, 신경망을 구축하고 훈련하는 데 필요한 모든 구성 요소를 제공합니다.

The core PyTorch modules for building neural networks are located in torch.nn, which provides common neural network layers and other architectural components. Fully connected layers, convolutional layers, activation functions, and loss functions can all be found here (we’ll go into more detail about what all that means as we go through the rest of this book). These components can be used to build and initialize the untrained model we see in the center of figure 1.2. In order to train our model, we need a few additional things: a source of training data, an optimizer to adapt the model to the training data, and a way to get the model and data to the hardware that will actually be performing the calculations needed for training the model.

신경망을 구축하기 위한 핵심 PyTorch 모듈은 공통 신경망 계층 및 기타 아키텍처 구성 요소를 제공하는 torch.nn에 위치해 있습니다.  
완전 연결 계층, 컨볼루션 계층, 활성화 함수 및 손실 함수는 모두 여기에서 확인할 수 있습니다(이 책의 나머지 부분을 살펴보면서 이 모든 것이 무엇을 의미하는지 자세히 설명하겠습니다).  
이러한 구성 요소는 그림 1.2의 중앙에서 볼 수 있는 훈련되지 않은 모델을 구축하고 초기화하는 데 사용할 수 있습니다.  
모델을 훈련시키기 위해서는 몇 가지 추가 사항이 필요합니다: 훈련 데이터의 출처, 훈련 데이터에 모델을 적응시키기 위한 최적화기, 그리고 실제로 모델 훈련에 필요한 계산을 수행할 하드웨어에 모델과 데이터를 가져오는 방법.

<img width="815" alt="스크린샷 2025-04-15 오전 11 09 48" src="https://github.com/user-attachments/assets/8e217ce9-c034-49b3-ab90-c7dc72df94d1" />

At left in figure 1.2, we see that quite a bit of data processing is needed before the training data even reaches our model.4 First we need to physically get the data, most often from some sort of storage as the data source. Then we need to convert each sample from our data into a something PyTorch can actually handle: tensors. This bridge between our custom data (in whatever format it might be) and a standardized PyTorch tensor is the Dataset class PyTorch provides in torch.utils.data. As this process is wildly different from one problem to the next, we will have to implement this data sourcing ourselves. We will look in detail at how to represent various type of data we might want to work with as tensors in chapter 4. As data storage is often slow, in particular due to access latency, we want to parallelize data loading. But as the many things Python is well loved for do not include easy, efficient, parallel processing, we will need multiple processes to load our data, in order to assemble them into batches: tensors that encompass several samples. This is rather elaborate; but as it is also relatively generic, PyTorch readily provides all that magic in the DataLoader class. Its instances can spawn child processes to load data from a dataset in the background so that it’s ready and waiting for the training loop as soon as the loop can use it. We will meet and use Dataset and DataLoader in chapter 7.

그림 1.2의 왼쪽을 보면, 훈련 데이터가 모델에 도달하기도 전에 상당히 많은 데이터 처리가 필요하다는 것을 알 수 있습니다.4  
그리고 그것은 실제 프로젝트에서 꽤 큰 비중을 차지할 수 있는 전처리가 아니라 즉석에서 이루어지는 데이터 준비일 뿐입니다.  

먼저, 데이터를 물리적으로 가져와야 합니다. 주로 데이터 소스로서의 일종의 저장소에서 데이터를 가져옵니다.  
그런 다음 데이터에서 각 샘플을 PyTorch가 실제로 처리할 수 있는 텐서로 변환해야 합니다. 이는 사용자 지정 데이터(어떤 형식이든 상관없이)와 표준화된 PyTorch 텐서 사이의 다리 역할을 합니다.  
PyTorch는 torch.utils.data에서 제공하는 데이터 클래스입니다. 이 프로세스는 문제마다 크게 다르기 때문에, 우리는 이 데이터 소싱을 직접 구현해야 합니다.  
4장에서는 텐서로 작업하고자 할 다양한 유형의 데이터를 어떻게 표현할지 자세히 살펴보겠습니다.  

특히 접근 지연으로 인해 데이터 저장 속도가 느리기 때문에 데이터 로딩을 병렬화하고자 합니다.  
하지만 Python이 사랑하는 많은 것들이 쉽고 효율적이며 병렬 처리를 포함하지 않기 때문에, 데이터를 배치로 조립하기 위해서는 여러 프로세스가 필요합니다:  
여러 샘플을 포함하는 텐서입니다. 이는 다소 정교하지만, 비교적 일반적이기 때문에 PyTorch는 DataLoader 클래스에서 이러한 모든 마법을 쉽게 제공합니다.  
이 인스턴스는 하위 프로세스를 생성하여 백그라운드의 데이터셋에서 데이터를 로드하고 루프가 사용할 수 있는 즉시 학습 루프를 대기할 수 있도록 할 수 있습니다.  
우리는 7장에서 Dataset과 DataLoader를 만나 사용할 것입니다.

With the mechanism for getting batches of samples in place, we can turn to the training loop itself at the center of figure 1.2. Typically, the training loop is implemented as a standard Python for loop. In the simplest case, the model runs the required calculations on the local CPU or a single GPU, and once the training loop has the data, computation can start immediately. Chances are this will be your basic setup, too, and it’s the one we’ll assume in this book. At each step in the training loop, we evaluate our model on the samples we got from the data loader. We then compare the outputs of our model to the desired output (the targets) using some criterion or loss function. Just as it offers the components from which to build our model, PyTorch also has a variety of loss functions at our dis- posal. They, too, are provided in torch.nn. After we have compared our actual outputs to the ideal with the loss functions, we need to push the model a little to move its outputs to better resemble the target. As mentioned earlier, this is where the PyTorch autograd engine comes in; but we also need an optimizer doing the updates, and that is what PyTorch offers us in torch.optim. We will start looking at training loops with loss functions and optimizers in chapter 5 and then hone our skills in chapters 6 through 8 before embarking on our big project in part 2. It’s increasingly common to use more elaborate hardware like multiple GPUs or multiple machines that contribute their resources to training a large model, as seen in the bottom center of figure 1.2. In those cases, torch.nn.parallel.Distributed-DataParallel and the torch.distributed submodule can be employed to use the additional hardware.

샘플 배치를 준비하는 메커니즘을 사용하면 그림 1.2의 중심에 있는 훈련 루프 자체로 전환할 수 있습니다.  
일반적으로 훈련 루프는 루프를 위한 표준 Python으로 구현됩니다. 가장 간단한 경우 모델은 로컬 CPU 또는 단일 GPU에서 필요한 계산을 실행하며, 훈련 루프가 데이터를 확보하면 즉시 계산을 시작할 수 있습니다.  
이것이 기본 설정일 가능성이 높으며, 이 책에서 가정할 내용입니다.  
훈련 루프의 각 단계에서 우리는 데이터 로더에서 얻은 샘플에 대해 모델을 평가합니다.  
그런 다음 기준 또는 손실 함수를 사용하여 모델의 출력을 원하는 출력(목표)과 비교합니다.  
모델을 구축하는 데 필요한 구성 요소를 제공하는 것처럼, PyTorch도 우리의 디스포지션에서 다양한 손실 함수를 제공합니다. 이들 역시 torch.nn.  
실제 출력과 손실 함수를 이상과 비교한 후, 모델이 목표와 더 잘 유사하도록 출력을 조금 더 밀어내야 합니다.  
앞서 언급했듯이, 여기서 PyTorch 자동 제어 엔진이 등장하지만 업데이트를 수행하는 최적화기도 필요하며, PyTorch가 torch.optim에서 제공하는 것이 바로 그것입니다.  
5장에서는 손실 함수와 최적화기를 사용하는 훈련 루프를 살펴보고, 6장에서 8장까지는 우리의 기술을 연마한 후 2장에서는 큰 프로젝트를 시작할 것입니다.  
그림 1.2의 하단 중앙에서 볼 수 있듯이, 여러 GPU나 여러 기계와 같은 더 정교한 하드웨어를 사용하여 대규모 모델을 훈련시키는 것이 점점 더 일반화되고 있습니다.  
이러한 경우, torch.n.parallelaral.distributed-DataParallel과 torch.distributed 서브모듈을 사용하여 추가 하드웨어를 사용할 수 있습니다.

The training loop might be the most unexciting yet most time-consuming part of a deep learning project. At the end of it, we are rewarded with a model whose parameters have been optimized on our task: the trained model depicted to the right of the training loop in the figure. Having a model to solve a task is great, but in order for it to be useful, we must put it where the work is needed. This deployment part of the process, depicted on the right in figure 1.2, may involve putting the model on a server or exporting it to load it to a cloud engine, as shown in the figure. Or we might integrate it with a larger application, or run it on a phone. One particular step of the deployment exercise can be to export the model. As mentioned earlier, PyTorch defaults to an immediate execution model (eager mode). Whenever an instruction involving PyTorch is executed by the Python interpreter, the corresponding operation is immediately carried out by the underlying C++ or CUDA implementation. As more instructions operate on tensors, more operations are executed by the backend implementation. PyTorch also provides a way to compile models ahead of time through TorchScript. Using TorchScript, PyTorch can serialize a model into a set of instructions that can be invoked independently from Python: say, from C++ programs or on mobile devices. We can think about it as a virtual machine with a limited instruction set, specific to tensor operations. This allows us to export our model, either as TorchScript to be used with the PyTorch runtime, or in a standardized format called ONNX. These features are at the basis of the production deployment capabilities of PyTorch. We’ll cover this in chapter 15.

훈련 루프는 딥러닝 프로젝트에서 가장 흥미롭지는 않지만 시간이 많이 소요되는 부분일 수 있습니다.  
마지막으로, 우리는 PyTorch와 관련된 명령어가 Python 인터프리터에 의해 실행될 때마다 해당 작업에 최적화된 모델, 즉 훈련 루프의 오른쪽에 묘사된 훈련된 모델로 보상을 받습니다.  
작업을 해결하기 위해 모델을 갖는 것도 좋지만, 이 모델이 유용하려면 작업이 필요한 곳에 배치해야 합니다.  
이 프로세스의 배포 부분은 그림 1.2의 오른쪽에 묘사된 것처럼 서버에 배치하거나 클라우드 엔진에 로드하기 위해 모델을 내보내는 것을 포함할 수 있습니다.  
또는 더 큰 애플리케이션과 통합하거나 휴대폰에서 실행할 수도 있습니다.  
배포 연습의 특정 단계 중 하나는 모델을 내보내는 것입니다.  
앞서 언급했듯이 PyTorch는 즉시 실행 모델(eager mode)로 기본 설정됩니다. PyTorch와 관련된 명령어가 Python 인터프리터에 의해 실행될 때마다 해당 작업은 기본 C++ 또는 CUDA 구현에 의해 즉시 수행됩니다.  
텐서에서 더 많은 명령어가 작동할수록 백엔드 구현에 의해 더 많은 작업이 실행됩니다. PyTorch는 TorchScript를 통해 모델을 미리 컴파일할 수 있는 방법도 제공합니다.  
TorchScript를 사용하면 Python과 독립적으로 호출할 수 있는 명령어 집합으로 모델을 직렬화할 수 있습니다: 예를 들어, C+ 프로그램이나 모바일 장치에서 호출할 수 있습니다.  
텐서 연산에 특화된 명령어 집합이 제한된 가상 머신으로 생각할 수 있습니다. 이를 통해 PyTorch 런타임에 사용할 TorchScript 또는 ONNX라는 표준화된 형식으로 모델을 내보낼 수 있습니다.  
이러한 기능은 PyTorch의 프로덕션 배포 기능의 기초가 됩니다. 이에 대해서는 15장에서 다룰 것입니다.

## 1.5 Hardware and software requirements
This book will require coding and running tasks that involve heavy numerical computing, such as multiplication of large numbers of matrices. As it turns out, running a pretrained network on new data is within the capabilities of any recent laptop or per- sonal computer. Even taking a pretrained network and retraining a small portion of it to specialize it on a new dataset doesn’t necessarily require specialized hardware. You can follow along with everything we do in part 1 of this book using a standard personal computer or laptop. However, we anticipate that completing a full training run for the more advanced examples in part 2 will require a CUDA-capable GPU. The default parameters used in part 2 assume a GPU with 8 GB of RAM (we suggest an NVIDIA GTX 1070 or better), but those can be adjusted if your hardware has less RAM available. To be clear: such hardware is not mandatory if you’re willing to wait, but running on a GPU cuts training time by at least an order of magnitude (and usually it’s 40–50x faster). Taken indi- vidually, the operations required to compute parameter updates are fast (from fractions of a second to a few seconds) on modern hardware like a typical laptop CPU. The issue is that training involves running these operations over and over, many, many times, incrementally updating the network parameters to minimize the training error. Moderately large networks can take hours to days to train from scratch on large, real-world datasets on workstations equipped with a good GPU. That time can be reduced by using multiple GPUs on the same machine, and even further on clusters of machines equipped with multiple GPUs. These setups are less prohibitive to access than it sounds, thanks to the offerings of cloud computing providers. DAWNBench (https://dawn.cs.stanford.edu/benchmark/index.html) is an interesting initiative from Stanford University aimed at providing benchmarks on training time and cloud computing costs related to common deep learning tasks on publicly available datasets. So, if there’s a GPU around by the time you reach part 2, then great. Otherwise, we suggest checking out the offerings from the various cloud platforms, many of which offer GPU-enabled Jupyter Notebooks with PyTorch preinstalled, often with a free quota. Goo- gle Colaboratory (https://colab.research.google.com) is a great place to start.

표준 개인용 컴퓨터나 노트북을 사용하여 이 책의 1부에서 설명하는 모든 작업을 따라할 수 있습니다.  
그러나 2부에서 더 고급 예제에 대한 전체 훈련을 완료하려면 CUDA 지원 GPU가 필요할 것으로 예상됩니다.  
2부에서 사용되는 기본 매개변수는 8GB의 RAM을 가진 GPU를 가정하지만(NVIDIA GTX 1070 이상을 제안합니다), 하드웨어의 RAM 사용량이 적을 경우 이를 조정할 수 있습니다.  
명확하게 말하자면, 이러한 하드웨어는 기다릴 의향이 있다면 필수는 아니지만 GPU에서 실행하면 훈련 시간이 최소 한 자릿수 이상 단축됩니다(보통 40~50배 빠릅니다).  
개별적으로 보면, 파라미터 업데이트를 계산하는 데 필요한 작업은 일반적인 노트북 CPU와 같은 최신 하드웨어에서 몇 초 단위로 빠릅니다.  
문제는 이러한 작업을 여러 번 반복하고 점진적으로 네트워크 매개변수를 업데이트하여 훈련 오류를 최소화하는 훈련이 필요하다는 것입니다.  
중간 규모의 네트워크는 좋은 GPU가 장착된 워크스테이션에서 대규모 실제 데이터셋을 처음부터 다시 훈련하는 데 며칠이 걸릴 수 있습니다.  
이 시간은 동일한 머신에서 여러 GPU를 사용하거나 여러 GPU가 장착된 머신 클러스터에서도 단축될 수 있습니다.  
클라우드 컴퓨팅 제공업체의 제공 덕분에 이러한 설정에 접근하는 것이 생각보다 덜 금지됩니다.

DAWNBench 는 공개적으로 이용 가능한 데이터셋에서 일반적인 딥러닝 작업과 관련된 훈련 시간과 클라우드 컴퓨팅 비용에 대한 벤치마크를 제공하는 것을 목표로 하는 스탠포드 대학교의 흥미로운 이니셔티브입니다.

For installation information, please see the Get Started guide on the official PyTorch website (https://pytorch.org/get-started/locally). We suggest that Windows users install with Anaconda or Miniconda (https://www.anaconda.com/distribution or https://docs.conda.io/en/latest/miniconda.html). Other operating systems like Linux typically have a wider variety of workable options, with Pip being the most com- mon package manager for Python. We provide a requirements.txt file that pip can use to install dependencies. Of course, experienced users are free to install packages in the way that is most compatible with your preferred development environment. Part 2 has some nontrivial download bandwidth and disk space requirements as well. The raw data needed for the cancer-detection project in part 2 is about 60 GB to download, and when uncompressed it requires about 120 GB of space. The com- pressed data can be removed after decompressing it. In addition, due to caching some of the data for performance reasons, another 80 GB will be needed while training. You will need a total of 200 GB (at minimum) of free disk space on the system that will be used for training. While it is possible to use network storage for this, there might be training speed penalties if the network access is slower than local disk. Preferably you will have space on a local SSD to store the data for fast retrieval.

Linux와 같은 다른 운영 체제는 일반적으로 더 다양한 실행 가능한 옵션을 제공하며, Pip은 Python에서 가장 일반적인 패키지 관리자입니다.  
Pip이 종속성을 설치하는 데 사용할 수 있는 요구 사항.txt 파일을 제공합니다.  
물론 숙련된 사용자는 선호하는 개발 환경과 가장 호환되는 방식으로 패키지를 자유롭게 설치할 수 있습니다.  
파트 2에는 몇 가지 비자명한 다운로드 대역폭과 디스크 공간 요구 사항도 있습니다. 파트 2의 암 detection 프로젝트에 필요한 원시 데이터는 다운로드하는 데 약 60GB이며, 압축 해제 시 약 120GB의 공간이 필요합니다.  
압축 해제된 데이터는 압축 해제 후 제거할 수 있습니다. 또한 성능상의 이유로 일부 데이터를 캐싱하기 때문에 학습 중에 80GB가 추가로 필요합니다. 학습에 사용할 시스템에는 총 200GB(최소)의 여유 디스크 공간이 필요합니다.  
이를 위해 네트워크 스토리지를 사용할 수는 있지만, 네트워크 액세스 속도가 로컬 디스크보다 느릴 경우 학습 속도 페널티가 발생할 수 있습니다. 빠른 검색을 위해 로컬 SSD에 데이터를 저장할 공간이 있는 것이 좋습니다.

### 1.5.1 Using Jupyter Notebooks


