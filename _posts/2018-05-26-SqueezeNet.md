---
title: SQEEZENET(모델 압축)
description: 딥러닝의 발달로 분류 정확도가 매우 좋아졌습니다. 하지만 반면급부로 모델의 복잡도는 그만큼 증가하게 됩니다. 이에 따라 모델의 파라미터수와 크기를 줄이면서 정확도를 최대한 유지하고자 하는 연구들도 발전하게됩니다.
category: Deep Learning
tags:
- CNN
- model compression
---


# SQUEEZENET(ALEXNET-LEVEL ACCURACY WITH 50X FEWER PARAMETERS AND <0.5MB MODEL SIZE)

여태까지 딥러닝(Deep learning), 특히 CNN(Convolutional Neural Networks) 관련된 연구들의 경우 정확도를 증가시키는데 초점을 맞추었습니다. 그러나 같은 성능(정확도)를 보여준다면, 모델이 작으면 작을수록 큰 이점을 가지게됩니다. 일단 학습 성능을 줄일 수 있으며 모바일과 같이 헤비하지 않은 기기들에도 쉽게 적용이 가능합니다. 딥러닝 모델들이 좋은 성능이 나오지만 반대로 그만큼 많은 수의 파라미터도 학습이 필요하다는 것을 생각하면 실제 세계(real world)에서는 이러한 장점 또한 매우 중요합니다.

## Introduction and Motivation

먼저, 동일한 성능을 가지는 모델이 정확도가 더 낮다면 다음과 같은 장점을 가질 수 있습니다.

- More efficient distributed training 
    * 병렬학습 때 굉장히 큰 효율이 납니다. (less communication -> fast training)
- Less overhead when exporting new models to clients
    * 자율주행과 같이 실시간으로 서버와 소통해야 하는 시스템의 경우, 매우 좋습니다. 데이터 전송 자체가 크지 않기 때문에 서버 과부하도 적게 걸리고 더욱 더 업데이트를 자주 할 수 있게 됩니다.
- Feasible FPGA and embedded deployment
    * FPGA(일종의 반도체 소자)는 보통 10MB 이하의 휘발성 메모리를 가지고 있습니다. 작은 모델은 직접적으로 FPGA에 모델을 심을 수 있으며, 이는 다른 기관을 통해 inference할 경우 생기는 병목현상(bottleneck)이 없어집니다. 또한 ASIC(Application-Specific Integrated Circuits)에 직접적으로 CNN을 배치할 수 있게됩니다.

사실 위에 나열한 것처럼 장점들을 자세히 설명하지 않더라도, 동일 성능이라는 가정 하에 모델이 작으면 작을수록 큰 이점이 생긴다는 것은 당연한 사실입니다. 논문 저자들은 이렇게 알렉스넷(AlexNet)과 비슷한 성능을 내면서도 파라미터는 50배나 줄고 모델 사이즈는 0.5MB밖에 안되는 SQUEEZENET(스퀴즈넷)이라는 네트워크 구조를 제시합니다.

## Related work

### Model Compression

모델 압축같은 경우는 사전 학습된 모델(pretrained model)에 SVD(Singular Vector Decomposition - 특이값분해)를 적용하여 적용하는 방식도 있었습니다. 또한 사전학습된 모델을 조금씩 잘라내면서 특정 값 이하로 내려가는 파라미터들을 전부 0으로 바꿔버려 sparse matrix를 생성하는 방식을 사용하는 방법도 있었습니다. Deep compression이라는 논문도 있습니다.

<div align='center'><a href="https://imgur.com/U6GiQMR"><img src="https://i.imgur.com/U6GiQMR.png" title='Deep compression'width="400px" /></a></div>

### CNN Microarchitecture & Macroarchitecture

CNN의 구성 요소인 convolution은 얀 리쿤이 처음 제안한 이후로 거의 25년 가까이 쓰이고 있습니다. Convolution filter는 보통 height, width, channel 이렇게 3차원으로 이루어져 있습니다. VGG 이전에는 5x5과 11x11 등 다양한 필터 사이즈를 사용했지만, 이후로는 거의 3x3 사이즈를 많이 사용하게됩니다. 이 논문에서는 하나의 모델을 구성하고 있는 특정 부분 등을 ***CNN microarchitecture***이라고 부릅니다.

모델의 부분을 CNN microarchitecture라고 한다면 input부터 끝까지 모델의 end-to-end를 ***CNN macroarchitecture***이라고 합니다. 이러한 CNN macroarchitecture로는 VGGNet, ResNet, DenseNet 등 여러 모델들이 존재합니다.

## SQUEEZENET: Preserving accuracy with few parameters

이제 네트워크가 최대한 파라미터를 적게 가질 수 있도록 하는 전략, Fire module이라는 모델 전체를 구성하는 block 그리고 스퀴즈넷(SqueezeNet)을 디자인하는 전략에 대해 설명하도록 하겠습니다.

### Architectural design strategies

1. Replace 3x3 filters with 1x1 filters
    * 모든 3x3 convolution filter를 1x1 필터로 교체합니다. 이는 1x1 필터가 3x3 필터에 비해 9배나 더 적은 파라미터를 가지고 있기 때문입니다.
2. Decrease the number of input channels to 3x3 filters
    * 만약 모델의 레이어(layer) 하나가 전부 3x3 필터로 구성되어있다면 파라미터의 총 수는 (input channel) x (number of filters) x (3x3)개와 같습니다. 따라서 3x3 필터 자체의 수를 줄이고 이에 더해 3x3으로 들어가는 input channel의 수도 줄여아합니다. 논문에서는 squeeze layer를 사용하여 input channel -> 3x3 filter로의 수를 줄여버립니다.
3. Downsample late in the network so that convolution layers have large activation
    * Downsampling part를 네트워크 후반부에 집중시키는 방법도 사용합니다. 보통 downsampe은 max(or average) pooling 또는 필터 자체의 stride를 높이는 방식으로 이미지의 spatial resolution을 줄이게됩니다. 이렇게 줄여서 한 번에 필터가 볼 수 있는 영역을 좁히면서 해당 이미지의 정보를 압축시키는 것입니다. (이에 대한 좀 더 자세한 설명은 [CapsNet -1 포스트](https://jayhey.github.io/deep%20learning/2017/11/28/CapsNet_1/) pooling 파트를 참고하시면 좋을 것 같습니다) 논문의 저자들은 모든 조건이 동등하다는 가정하에서 큰 activation map을 가지고 있을수록 성능이 더 높다는 것에서 영감을 얻었습니다. 따라서 스퀴즈넷에서는 네트워크 후반부에 downsample을 넣는 방법을 취합니다.

일단 1번과 2번은 CNN 전체 파라미터 수를 줄이면서 정확도를 최대한 보존하는 것에 초점을 맞췄습니다. 3번의 경우는 파라미터 수가 제한된 상황에서 정확도를 최대화 시키는 방식입니다. 이제 이렇게 3가지 전략을 적용시키는 fire module에 대하여 알아보도록 하겠습니다.

### The Fire Module

Fire module은 총 두 가지 layer로 이루어져있습니다. 먼저 *squeeze convolution layer*는 1x1 필터로만 이루어져 있습니다. 여기를 통과하면 다음으로는 *expand convolution layer*를 거치게 됩니다. 아래 그림을 보시면 이해가 더 쉬울 것 같습니다.

![Imgur](https://i.imgur.com/RZXOZDJ.png)

<div>Fire module에는 총 세 가지 하이퍼파라미터(${s}_{1x1}$, ${e}_{1x1}$, ${e}_{3x3}$)가 있습니다. 먼저 ${s}_{1x1}$는 squeeze layer에서 1x1 필터의 총 갯수입니다. ${e}_{1x1}$는 expand layer에서의 1x1 필터 총 갯수이며 ${e}_{3x3}$는 expand layer에서의 3x3 필터의 총 갯수입니다. Fire module을 만들 때는 ${s}_{1x1}$의 값을 ${e}_{1x1} +{e}_{3x3}$보다 더 작게 설정해주었습니다. 이는 위에 설명한 2번 전략처럼 3x3 필터로 들어가는 input channel의 수를 제한할 수 있게합니다. </div>

### SqueezeNet Architecture

스퀴즈넷의 전체적인 구조는 다음과 같습니다. 모든 스퀴즈넷은 전부 1개의 Convolution filter를 거치고 나서 max pooling이 이어집니다. 그 이후 총 8개의 fire module로 이루어져 있고 마지막에 convolution filter를 거치고 GAP(Global Average Pooling)로 마무리가 됩니다. Pooling layer를 conv1, fire4, fire8, conv10 이후에 배치하며 3번 전략(activation map을 크게)을 취했다고 합니다.~~(사실 이건 downsample late in the network가 아니라 그냥 적게 썼다는 표현이 더 맞는 것 같긴 합니다)~~

<div align="center"><a href="https://imgur.com/zEPVkzW"><img src="https://i.imgur.com/zEPVkzW.png"/></a></div>


기타 다른 디테일은 다음과 같습니다.
- 1x1과 3x3 필터의 output activation이 같기 때문에 expand module의 3x3 필터로 들어가는 데이터에 1-pixel짜리 zero padding 추가
- Squeeze와 expand layer 모두 ReLU 적용
- Fire9 module 이후로 dropout 50% 적용
- 초기 학습률 0.04로 설정 후 점차 감소시킴


## Evaluation of SqueezeNet

일단 논문 제목에도 나와있지만, 알렉스넷과 비교했을 때 파라미터 수를 확 줄였으면서도 성능이 비슷하다고 했으므로 실험 결과도 알렉스넷과 비교를 합니다. 실험은 이미지넷 데이터셋을 사용했습니다. 

![Imgur](https://i.imgur.com/OcwRopR.png)

위 표에서 왼쪽 CNN architecture와 compression approach를 보시면 됩니다. 단순 SqueezeNet만 사용했을 때는 50배 가까이 모델 사이즈가 줄어들었습니다. 게다가 기존 AlexNet의 top-1 & top-5 accuray에 근접하거나 뛰어넘는 모습을 보여줍니다.

여기에 더해 uncompressed 된 32bit의 데이터 타입을 사용한 생짜 SqueezeNet과 deep compression을 적용한 8bit, 6bit짜리 데이터 타입을 사용한 결과도 매우 놀랍습니다. 최고의 결과물은 모델 사이즈가 510배까지 줄어들었으며 성능도 큰 차이가 나지 않습니다. 이는 SqueezeNet 또한 모델 압축에 굉장히 유연하다는 뜻입니다. 

또한 기존의 파라미터가 많은 VGG나 AlexNet같은 모델들 뿐만 아니라 이미 컴팩트한 모델도 압축할 수 있다는 것을 보여주었습니다. 

![Imgur](https://i.imgur.com/qeSP0si.png)

## CNN Microarchitecture Design Space Exploration

SqueezeNet이 모델 압축에서 좋은 성능을 보여주긴 했지만, 모델의 세세한 부분에는 여전히 개선의 여지가 많이 남아있습니다. 여기서는 *microarchitectural exploration*(모델 세부 구조 탐색)과 *macroarchitectural exploration*(모델 전체 구조 탐색)에 대해 알아보도록 하겠습니다. 정확도를 올리는 것에 초점을 두지 않고 CNN의 구조가 모델의 크기와 정확도에 어떤 영향을 끼치는지 알아보기 위해 진행하였다고 합니다.

### CNN Microarchitectural Metaparameters

<div>Fire module은 하이퍼파라미터 ${s}_{1x1}$, ${e}_{1x1}$, ${e}_{3x3}$로 구성되어 있습니다. SqueezeNet은 총 8개의 fire module이 있으므로 총 24개의 하이퍼파라미터를 가지게됩니다. 논문에서는 24개의 파라미터를 전부 통제하는 파라미터를 metaparameter(메타 파라미터)라고 지명하였습니다. 먼저 ${base}_{e}$는 가장 첫 번째 fire module의 expand filter의 수 입니다. 이후로는 매번 $freq$ fire module들이 ${incr}_{e}$만큼씩 expand filter의 수가 증가합니다. Expand layer 내부에서 1x1과 3x3 필터의 총 수는 $e_i$입니다. 여기에 expand layer 내부에서 3x3 필터의 비율을 결정해주는 ${pct}_{3x3}$이 추가됩니다. 이는 전체 fire module이 공유하는 파라미터입니다. 그리고 SR(Squeeze Ratio)이라는 메타파라미터가 있습니다. SR은 0과 1사이의 값을 가지며 expand layer filter의 전체 갯수에 곱해주어 squeeze layer filter의 전체 갯수를 결정해줍니다. 수식으로 정리하면 아래와 같습니다. 전체적으로 메타 파라미터는 다음과 같이 설정하였습니다 : ${base}_{e}=128$, ${incr}_{e}=128$, ${freq}=2$, ${SR}=0.125$</div>

$${ e }_{ i }={ base }_{ e }+\left( { incr }_{ e }\times \left[ \frac { i }{ freq }  \right]  \right) $$

$$ { e }_{ i }={ e }_{ i,1x1 }+{ e }_{ i,3x3 } $$

$$ { e }_{ i,3x3 }={ e }_{ i }\times { pct }_{ 3x3 } $$

$$ { e }_{ i,1x1 }=SR\times{e}_{i}$$


![Imgur](https://i.imgur.com/KJ36oZ8.png)

### SR(Squeeze Ratio)

여기서 SR은 squeeze layer에서 expand layer로 들어가는 input channel의 수를 줄여주는 역할을 합니다. 논문에서는 SR을 [0.125, 1.0]의 범위에서 조금씩 바꿔봐며 실험을 진행합니다. 위 그림을 보시면 SR이 0.125롤 넘어가도 모델 사이즈가 조금씩 커지긴 하지만 성능이 더 좋아지는 것을 확인할 수 있습니다. 거의 0.75일 때 최고점을 찍고 그 이후로는 정확도가 올라가지는 않습니다. 다른 메타파라미터는 다음과 같이 설정했습니다. 

${base}_{e}=128$, ${incr}_{e}=128$, ${pct}_{3x3}=0.5%


### Trading off 1x1 and 3x3 filters

다음으로는 3x3 필터의 비율을 정해주는 ${pct}_{3x3}$에 대한 실험이 진행됩니다. 위 그림 (b)에서 ${pct}_{3x3}$의 값을 0.01부터 0.99까지 증가시켰습니다. (a)와 (b)에서 모델 사이즈가 13MB일 때가 같은 모델입니다. ${pct}_{3x3}$가 0.5를 넘어가는 순간부터 모델 사이즈만 커지고 정확도는 증가하지 않습니다.

## CNN macroarchitecture Design Space Exploration

모델의 세부 부분들에 대한 최적화가 끝났으니 이제 모델 전체 구조 탐색에 대해 다음과 같이 총 세가지 모델에 대한 실험을 진행하였습니다.

- Vanilla SqueezeNet
- SqueezeNet with simple bypass connections between some Fire modules
- SqueezeNet with complex bypass connections between the remaining Fire modules.

<div align="center"><a href="https://imgur.com/zEPVkzW"><img src="https://i.imgur.com/zEPVkzW.png"/></a></div>

(저 위에도 있는 그림이지만) 위 그림이 세 가지 CNN macroarchitecture입니다. 그림을 보면 바로 알 수 있지만 bypass connection은 ResNet에서 쓰인 skip connection과 같은 개념입니다. fire4에 들어가는 input channel들은 fire2와 fire3의 elementwise addition이라고 보시면 됩니다. 그림의 가운데를 보면, fire module을 1개 이상 건너뛰지 않고 bypass connection이 연결되어 있습니다. 이는 fire module의 input과 output 모두 같은 수의 channel을 가지고 있어야하기 때문입니다. 이러한 한계점이 있기 때문에, 논문의 저자들은 "complex bypass connection"이라는 개념을 추가합니다. Complex가 복잡하단 뜻이지만 단순히 1x1짜리 convolution을 거쳐줘서 채널의 숫자를 맞춰주는 것입니다. 이렇게하면 각 fire module의 output channel 수가 달라도 숫자를 맞춰서 elementwise addition을 해줄 수 있습니다.

Fire module의 구조를 보면, squeeze layer가 expand layer보다 필터 수가 더 적습니다. 이는 중요한 정보가 layer 사이의 병목에서 사라질 수 있다는 단점이 있습니다. 그러나 bypass connection을 추가하면 중요한 정보도 손실 없이 쉽게 흘러갈 수 있게 만들 수 있습니다.

Microarchitecture를 전부 고정시킨 상태에서 실험한 결과는 아래와 같습니다. 저는 논문을 처음 봤을 때 당연히 complex bypass 버젼이 더 높은 성능이 나올 줄 알았는데 simple bypass 버젼이 더 좋은 성능을 보여주고 있습니다. (혹시 이유를 아시는분은 댓글 부탁드려요...)

## Conclusions

논문에서는 기존 AlexNet과 비슷한 성능을 내면서 용량은 엄청나게 줄인 네트워크를 만들었습니다. 사실 딥러닝 네트워크 구조 관련 논문을 많이 읽다보면, 정확도에만 신경이 쓰이기 마련입니다. 특히나 요즘처럼 마지막단에 fully connected layer를 넣지 않는 네트워크들은 확실히 파라미터 수가 예전보단 적으면서도 높은 성능을 보여주고 있습니다.

하지만 실제 세계에는 모델 사이즈가 작은게 굉장히 중요한 분야가 많이 있습니다. 이런 분야에서는 모델의 성능도 좋지만 확실히 모델이 얼마나 무거운지도 굉장히 중요하다고 합니다. 논문에서도 자율주행이나 반도체 소자 등 여러 분야에 관한 언급이 있었습니다. 파라미터 최적화 관점에서 굉장히 재밌게 읽을 수 있었습니다.

![Imgur](https://i.imgur.com/M8GXcAv.png)

> Reference
>
> * Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., & Keutzer, K. (2016). SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size. arXiv preprint arXiv:1602.07360.