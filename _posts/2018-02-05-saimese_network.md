---
title: Saimese Neural Networks for One-shot Image Recognition(샴 네트워크)
description: 딥러닝에서 네트워크를 학습시킬 때, 매우 많은 트레이닝 데이터가 필요합니다. 이러한 단점을 극복하여 한 레이블 당 하나의 이미지만 있어도 분류할 수 있게 학습시키는게 one-shot learning입니다.
category: Deep Learning
tags:
- one-shot learning
- CNN
- transfer learning
---

# Saimese networks for one-shot learning 

## Introduction

딥러닝이 이렇게 급부상하게 된 가장 큰 이유는, 매우 큰 차원수를 가진 데이터(고화질 이미지, 자연어 등)를 효과적으로 쉽게 처리할 수 있기 때문입니다. 그러나 또한 가장 큰 단점으로, 연산량이 매우 크다는 것과 대량의 트레이닝 데이터가 필요하다는 점을 들 수 있습니다. 

![Imgur](https://i.imgur.com/TBtqdxy.png)

칼에 대해 아무런 지식이 없는 사람이라도 왼쪽 칼 사진 한 장만 보면 칼이라는 개념(concept)을 바로 학습할 수 있습니다. 그 이후로는 단 한번도 본 적 없는 특이한 칼들(오른쪽 세 개의 칼 사진)도 "칼"이라는 사실을 바로 알아냅니다. 하지만 일반적인 딥러닝 네트워크들은 한 클래스를 학습하기 위해 수백장, 아니 수천장이 넘는 사진이 필요합니다.  데이터의 클래스가 뭔지 하나씩 전부 레이블을 달아주는게 쉽지 않은 일인 만큼(어마어마한 돈이 들게 마련이죠) 사람과 같은 학습 능력을 가진다는건 매우 중요합니다.  



이를 해결하기 위한 방법 중 하나가 one-shot learning(원샷러닝)입니다.  이름이 one-shot인 이유는 레이블 당 단 하나의 트레이닝 이미지가 존재하더라도 네트워크가 빠르게 학습할 수 있기 때문입니다. 실제 세계에는 레이블이 없는 데이터가 굉장히 많은 만큼 매우 유용한 기법이겠죠? 원샷러닝을 정의해보면 아래와 같이 표현할 수 있습니다.

***The act of learning to generalize from one or a few number of training examples per class***



## N-way one-shot learning

### Dataset

이제 원샷러닝을 학습하기에 앞서, 사용 데이터에 대해 먼저 알아보겠습니다. 원샷러닝을 연구하기 좋은 데이터로, Omniglot dataset이 있습니다. 50가지의 언어에 대해 총 1623개의 문자가 20명의 다른 사람에게서 쓰였습니다. 즉, 총 20*1623개의 데이터가 존재합니다. 그중에서는 무려 한국어도 있습니다! 아래 그림에 보면 ㅚ와 ㅃ가 있는 것을 확인할 수 있습니다.

![Imgur](https://i.imgur.com/guMYeU4.png)

이 연구에서는 30개의 언어를 통해 학습시키고 20개의 언어로 one-shot task를 테스트하였습니다. 트레이닝 언어는 964개의 알파벳, 테스트 언어는 총 659개의 알파벳이 있다고 합니다.



### Evaluation

이번에는 평가 지표에 대해 알아보도록 하겠습니다.  원샷러닝은 학습 후 평가할 때 분류할 클래스의 갯수(n개)에 따라 n-way one-shot task라고 합니다. 예를 들어 아래 사진에서 분류할 클래스가 9개라면 9-way one-shot task가 됩니다. 만약 클래스가 25개라면 25-way one-shot task가 되겠죠? 

![Imgur](https://i.imgur.com/AM2P06O.png)

네트워크의 학습이 끝나고 나면 [테스트 이미지, support set]형태를 만들어서 테스트를 진행합니다. Support set에는 한 개만 같은 클래스의 이미지를 넣고 나머지에는 다른 클래스의 이미지를 집어넣습니다. 만약 랜덤으로 예측하면 확률은 $1/n$이 됩니다.  위 그림에선 랜덤 예측의 경우 9-way one-shot task는 11%, 25-way one-shot task는 5%가 될 것입니다. 



## One-shot learning Baseline

### 1-Nearest Neighbor

원샷러닝의 가장 간단한 베이스라인으로 1-Nearest Neighbor를 사용합니다. 일단 클래스마다 하나의 example밖에 없기 때문아 $k$를 1로 설정해줍니다. 정말 간단하게, 단순히 테스트 이미지에서 유클리디언 거리(euclidean distance)로 가장 가까운 학습 이미지와 가까운 것을 선택하는 방법입니다. 수식으로 표현하면 아래와 같습니다.

$$C(\hat { x } )=arg\min _{ c\in S }{ \left\| \hat { x } -{ x }_{ c } \right\|  } $$

20-way one-shot task에서 1-NN 알고리즘으로는 28%의 정확도를 보였다고합니다. 사실 28%가 되게 낮아보이는 수치이긴 하지만 랜덤 예측이 5%인걸 감안하면 상당히 높은 수치가 나왔다고 볼 수 있습니다. 

![Imgur](https://i.imgur.com/HPlw4Yr.png)



### HBPL(Hierarchical Bayesian Program Learning)

두 번째 베이스라인으로는 HBPL을 사용합니다. HBPL은 기존 방법과 다르게 문자를 그릴 때의 획 정보를 더 사용하여 글자를 만드는 생성 모델(generative model)을 만들었습니다. 베이지안 룰을 사용하여 한 획 그어질 때마다 앞으로 어떻게 획이 그려질지 사전 분포(prior distribution)를 사용하여 글자를 생성합니다. 이 모델을 사용하여 20-way one-shot task에서 무려 95.2%라는 인간에 근접하는 정확도를 보여주게 됩니다. 하지만 이 모델에는 다음과 같은 단점이 있습니다.

- Raw pixel이 아닌 획에 대한 정보를 사용한다(실제 세계에는 없는 데이터)
- 획에 대한 생멍 모델을 학습한다. 이는 HBPL은 학습시킬 때 더 복잡한 데이터가 필요하며 

하지만 실제로 우리가 원샷러닝을 사용한다면, 획에 대한 정보가 있을 리가 없습니다. 오히려 레이블을 하나씩 달아주는 것보다 획 정보 얻는 것이 더 어려워 보이네요.

![Imgur](https://i.imgur.com/tO4WAT5.png)



 생성모델에 대한 개념은 [이전 포스트](https://jayhey.github.io/semi-supervised%20learning/2017/12/08/semisupervised_generative_models/)를 참고해주시면 되겠습니다. ~~링크된 포스트는 베이지안 생성 모델이 아닌 일반적인 생성 모델에 대한 개념입니다!~~



## Deep networks for one shot learning?

여기까지 읽으면 이런 생각이 들 수도 있습니다. '어째서 vanila cross entropy loss softmax으로는 원샷러닝을 할 수 없는것일까?' 답은 과적합(overfitting)때문입니다. 일반적인 딥러닝 네트워크는 one-shot이 아닌 hundred-shot learning조차 overfitting에서 벗어나기 쉽지 않습니다. 너무나도 많은(몇백만개에 이르는) 파라미터를 가지고 있기 때문인데, 이는 굉장히 심도있는 학습을 가능하게 해주지만 반대로 원샷러닝에는 도움이 되지 않습니다.  그 많은 파라미터를 전부 움직이게 만드려면 소수의 데이터로는 어림도 없기 때문입니다.  아래 그림의 네트워크가 "고양이"에 대한 학습을 전혀 시켜놓지 않은 상태라고 해봅시다. 이 네트워크는 이미 다른 클래스에 과적합되어있는 상태이기 때문에 달랑 한 장의 고양이 사진으로는 고양이 클래스에 대한 학습이 불가능합니다.

![Imgur](https://i.imgur.com/oz1hHso.png)

사실 생각해보면 사람에게는 "one-shot learn"이 매우 쉽습니다. 수년, 수십년의 시간 동안 비슷한 물체들을 보면서 익숙해져 있기 때문입니다. 이런 면에서는 고작 정규분포를 따르는 랜덤 웨이트로 시작하는 뉴럴네트워크에게서 원샷러닝을 기대한다는 것도 어려운 일일 것입니다. 이러한 이유 때문에 원샷러닝 관련된 논문들이 *knowledge transfer*의 개념을 사용합니다.



뉴럴네트워크는 고차원 데이터(특히 이미지)에서 중요한 feature를 잘 잡아냅니다. 따라서 만약 네트워크를 학습시킬 때 one-shot task와 유사한 데이터로 학습을 시킨다면(예를 들면 언어에서 히브리어로 학습을 시키고 테스트는 한국어로) 딥러닝 자체의 수많은 파라미터를 조절할 필요 없이 학습 과정에서 추출한 유용한 feature들을 잘 활용할 수 있을 것입니다.  이 논문에서는 이렇게 유용하고 robust한 feature를 학습하는 모델을 만드는 것을 목표로 합니다. 아래 그림처럼 verification tasks(학습)에서는 feature를 학습하고, one-shot task(테스트)에서는 맞는 분류를 찾습니다.

![Imgur](https://i.imgur.com/MX3OB4l.png)



## Network architecture

논문에서는 뉴럴네트워크에 두 개의 이미지를 주고 같은 카테고리인지 아닌지 판단하게 학습을 시킵니다. 그리고 학습이 끝나면 테스트 이미지를 네트워크에 집어넣고 support set 중에서 어떤 이미지가 가장 같은 카테고리에 속하는지 맞춰보게 합니다.  이를 위해 샴 네트워크(siamese neural network)는 두 개의 이미지를 각각 하나의 네트워크에 집어넣어 최종적으로 두 이미지가 같은 클래스인지에 대한 확률값이 나오게 설계되어있습니다.

<div>샴 넷의 가장 큰 특징은 대칭성(symmetry)을 가진다는 점입니다.  두 이미지 ${x}_{1}$과 ${x}_{2}$가 있다고 했을 때, ${ x }_{ 1 }\circ { x }_{ 2 }$는 두 이미지가 같은 클래스에 속해 있다는 뜻입니다. 그렇다면 ${ x }_{ 1 }\circ { x }_{ 2 }$과 ${ x }_{ 2}\circ { x }_{ 1 }$는 항상 같은 결과가 나와야합니다. 즉 샴 넷의 결과물인 $p({ x }_{ 1 }\circ { x }_{ 2 })$는 $p({ x }_{ 2 }\circ { x }_{ 1 })$와 항상 같아야합니다. 이러한 대칭성이 매우 중요한 이유는, 샴 넷의 가장 큰 특징 중 하나가 distance metric을 사용하기 때문입니다. </div>



그림으로 확인 해보도록 하겠습니다. 

![Imgur](https://i.imgur.com/ikqax68.png)

위 그림이 샴 넷의 두 네트워크중 한 개의 모습을 보여주고 있습니다. 우리가 일반적으로 알고있는 conv, relu, pooling 등을 사용합니다. 만약 위 그림처럼 그냥 그대로 네트워크를 만든다면 단순 이진분류기 모델밖에 되지 않습니다. 



![Imgur](https://i.imgur.com/HLQqD3A.png)

샴 넷은 위와 같은 구조를 가집니다. 각 네트워크를 거쳐 나온 4096개의 feature를 가지고 $L1$ norm(absolute distance)을 구합니다. 아래 식을 보면 ${ \alpha  }_{ j }$는 트레이닝 과정에서 에서 자동으로 학습되는 파라미터로 각 피쳐에 대한 거리값에 곱해줍니다. $\sigma $는 시그모이드 활성함수 입니다. 

$$\sigma (\sum _{ j }^{  }{ { \alpha  }_{ j }\left| { h }_{ 1,L-1 }^{ (j) }-{ h }^{ (j) }_{ 2,L-1 } \right|  }) $$

이 값을 다시 fully connected layer를 거쳐 시그모이드 함수로 단일 값을 받아냅니다. 1이면 같은 클래스, 0이면 다른 클래스가 되는 것이죠.  여기 있는 모든 fully connected layer가 파라미터 갯수의 96%를 차지한다고 하네요. 여기서 나온 예측값과 타겟에 대하여 binary cross entropy를 이용한 로스를 구합니다. 식은 아래와 같습니다.

$$L(x_1,x_2,t) = t \cdot log(p(x_1 \circ x_2)) + (1 - t) \cdot log(1-p(x_1 \circ x_2)) + \lambda \cdot ||w||_2$$

학습에서 사용하는 손실함수(loss function)가 위와 같다면, 실제 one-shot task를 할 때는 다음과 같은 방식으로 진행합니다. test image를 한 네트워크에 집어넣고, 다른 네트워크에는 support set에 있는 이미지를 하나씩 집어넣습니다. 여기서 가장 큰 확률값을 가지는 이미지를 같은 클래스로 분류하는 것입니다. 식은 아래와 같습니다.

 $$C(\hat { x } ,S)=\underset { c }{ { argmax } } P(\hat { x } \circ x_{ c }),x_{ c }\in S$$

케라스로 간단하게 모델 구조를 짜면 아래와 같습니다. [출처는 이곳입니다.](https://sorenbouma.github.io/blog/oneshot/) 

```python
def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)

def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

input_shape = (105, 105, 1)
left_input = Input(input_shape)
right_input = Input(input_shape)
#build convnet to use in each siamese 'leg'
convnet = Sequential()
convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(7,7),activation='relu',
                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(Flatten())
convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))
#encode each of the two inputs into a vector with the convnet
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
#merge two encoded inputs with the l1 distance between them
L1_distance = lambda x: K.abs(x[0]-x[1])
both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(both)
siamese_net = Model(input=[left_input,right_input],output=prediction)

optimizer = Adam(0.00006)
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
```





## Results

학습률(learning rate)이라던가 epoch수 등등은 그냥 넘어가도록 하겠습니다. ~~자세한 내용은 논문을 참고해주세요~~

![Imgur](https://i.imgur.com/f4mmvus.png)

테스트 정확도를 봐도 굉장히 좋은 결과를 얻은 것을 확인할 수 있습니다.  왼쪽은 Omniglot 데이터셋을 대상으로 one-shot performace를 여러 모델로 측정한 결과입니다. 사람과 비교해도 큰 차이가 나지 않는 92%의 정확도를 보여줍니다. 오른쪽은 MNIST 데이터셋을 대상으로 1-NN과 비교실험한 결과인데, 70.3%라는 준수한 성능을 보여주고 있습니다.




> Reference
>* Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "Siamese neural networks for one-shot image recognition." ICML Deep Learning Workshop. Vol. 2. 2015.
>* https://sorenbouma.github.io/blog/oneshot/
