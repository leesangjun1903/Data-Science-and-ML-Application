# Positional encoding
Transformer model을 살펴보면, positional encoding이 적용된다. 

positional encoding에 대해 이야기 하려면, transformer 모델의 특징에 대해 먼저 알아보아야 한다. transformer 모델은 기존의 자연어 처리에 사용되는 LSTM, RNN 기반의 모델과 달리 '시간적 연속성'을 모델의 핵심부에서 다루지 않는다. transformer 모델의 핵심은 attention 함수인데, 쉽게 말하면 "John is good at playing soccer and he want to be a soccer player." 라는 문장에서 he를 attention 함수를 통과시키면 John에 대한 attention 값이 가장 높게 나오도록 하는 것이다. 여기서 attention 함수가 들어간 layer에서는 구조적으로 시간적 연속성이 없이 입력값을 다루게 된다. 반면,
RNN 류의 모델은 모델의 구조 자체가 시간적 연속성을 보장하게 된다. (cell과 cell들이 병렬적으로 연결된 구조)
 
attention 함수가 이러한 구조를 통해 얻는 이점은 데이터가 통과하는 layer의 수를 줄일 수 있어 연산에서의 이득과, RNN 류 모델의 학습 과정에서 발생하는 기울기 소실/폭발 등에서 자유롭다는 것이다. 
 
positional encoding이 왜 필요한가 하면, 그럼에도 어순은 언어를 이해하는 데 중요한 역할을 하기에 이 정보에 대한 처리가 필요하다. 따라서 이 논문의 저자가 채택한 방식은 attention layer에 들어가기 전에 입력값으로 주어질 단어 vector 안에 positional encoding 정보, 즉, 단어의 위치 정보를 포함시키고자 하는 것이다.

# Reference
https://skyjwoo.tistory.com/entry/positional-encoding이란-무엇인가 [jeongstudy:티스토리]
