# Grad CAM

class c에 대한 점수 y_c (before the softmax)을 각 원소로 미분합니다. (back propagation 하듯이 말이죠.)  
이 미분값은 각 Feature map의 원소가 특정 class에 주는 영향력이 됩니다.  
각 feature map에 포함된 모든 원소의 미분값을 모두 더하여 neuron importance weight, a를 구하면, 이 a는 해당 feature map이 특정 class에 주는 영향력이 됩니다.

# Reference
- https://arxiv.org/abs/1610.02391
- https://wikidocs.net/135874
