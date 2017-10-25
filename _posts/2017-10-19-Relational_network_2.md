---
title: A simple neural network module for relational reasoning - 2
description: 이번 포스트에서는 RN(Relation Network)이 어떤 방식으로 관계 추론에서 큰 강점을 가지고 있는지 네트워크 구조를 통해서 살펴보도록 하겠습니다. 네트워크 자체는 MLP로 매우 단순하지만 적용하는 object set를 잘 지정해서 놀라운 효과를 거두게 됩니다. 사람보다 더 뛰어난(super-human) 성능을 어떻게 만들어 냈는지 확인해 봅시다.
category: Deep Learning
tags:
- network architecture
- experiments
---

# Tasks

먼저 RN에 사용한 데이터셋에 대한 설명이 필요할 것 같습니다. 기존 이미지 분류 문제 등에서는 이미지넷 데이터셋, CIFAR 등을 벤치마크용 데이터로 사용했습니다. 하지만 관계성 추론 문제는 이러한 벤치마크 데이터가 없기 때문에 총 3가지의 데이터셋을 만들어서 사용하게됩니다. ~~필요하면 데이터를 만들어내는 딥마인드...역시 돈이 많습니다.~~

## CLEVR

![Imgur](https://i.imgur.com/Y4ISsY3.png)

기존 visual QA 데이터셋들이 모호성이 있고 언어학적으로 굉장히 편향되어 있다는 점 때문에 만든 데이터셋입니다. 3D 이미지로 렌더링 된 도형들로 이루어져 있습니다. 이 데이터셋의 가장 큰 특징으로는 많은 질문들이 실제 세계에서도 관련성이 있다는 것입니다. QA부문에서 높은 성능을 봅여주던 네트워크들이 이 데이터셋에서는 제대로 된 성능을 내지 못한다는 것을 그 이유로 들고 있습니다. 

CLEVR는 pixel, description 두 가지 버전이 있습니다. Pixel 버전에서는 이미지가 2D 픽셀 형식으로 나타나있고, description 버전은 그림에 나온 상황이 행렬로 정리되어 있습니다. 그림에 나온 도형이 어느 위치에 있고 무슨 모양이며 색깔과 재질은 무엇인지 등으로 나타냈습니다.


## Sort-of-CLEVR

<div align="center"><a href="https://imgur.com/DzqU4qP"><img src="https://i.imgur.com/DzqU4qP.png"/></a></div>

RN이 관계성 추론에 얼마나 더 도움이 되는가 알아보기 위해 관계성이 있는 질문과 없는 질문으로 나누어져 있습니다. 각 그림은 6개의 물체(사각형 또는 원)로 이루어져 있으며 6개의 색깔을 넣어주었습니다. 질문들은 자연어 처리 자체의 어려움을 고려하여 최대한 통일성을 줘서 쉽고 간결하게 만들었습니다.

## bAbI

<div align="center"><a href="https://imgur.com/K3xebJy"><img src="https://i.imgur.com/K3xebJy.png" width="450px"/></a></div>

이 데이터셋은 완전 텍스트로 이루어져있습니다. 추론, 유도, 갯수세기 등 20여가지 태스크가 있으며 각 질문은 하나의 "facts"를 이루고 있습니다. 이게 무슨 말이냐면 "Sandra picked up the football"과 "Sandra went to the office"라는 문장이 있다면 질문은 "Where is the football?"이 되는 것이죠. 

## Dynamic physical systems

<div align="center"><a href="https://imgur.com/INGWItu"><img src="https://i.imgur.com/INGWItu.png"/></a></div>

MuJoCo라는 물리엔진을 사용하여 공을 랜덤으로 이리저리 움직이게 만듭니다. 어떤 공들은 랜덤으로 튀지만 어떤 공들은 보이지 않는 환경에 의해 이리저리 움직입니다. 여기서 태스크가 두 개로 나뉘는데 첫 번째 태스크는 공들 간 관계를 모델이 학습하게 만들어서 관계가 있다면 링크를 이어주는 것입니다. 두 번째로는 연속된 프레임에 따라 공들의 색과 좌표가 바뀌는 것을 보고 "시스템"이 몇 개 인지 세는 태스크입니다. 여기서 "시스템"이란 몇 개의 독립적인 시스템으로 공들이 뭉쳐있는가를 말한다고 할 수 있습니다. 이 두 태스크 모두 공들의 움직임과 위치를 가지고 관계를 추론해내는 과정이 필요합니다. 

이해가 잘 안가신다면 아래 유튜브 링크를 확인하시면 바로 이해가 되실겁니다.

[![Video Label](https://img.youtube.com/vi/FDF6-NGv38c/0.jpg)](https://youtu.be/FDF6-NGv38c)

# Relation Network 적용하기

RN(Relation Network)은 다른 네트워크와는 다르게 원본 데이터를 처음부터 입력하지 않습니다. [이전 포스트](https://jayhey.github.io/deep%20learning/2017/10/19/Relational_network_1/)에서 이미 설명했듯이, 여러 object set에 적용을 합니다. 자연어나 이미지에 직접적으로 적용하는게 아니라는게 다른 네트워크와의 차이점 중 하나라고 할 수 있습니다. LSTM이나 CNN의 결과물과 같은 비정형화 된 입력값들(unstructured input)을 object로 만들어서 RN에 넣게 됩니다. 

![Imgur](https://i.imgur.com/cx52f6w.png)

## Dealing with pixels

먼저 128x128사이즈의 이미지를 4개의 콘볼루션(Convolution) 레이어에 통과시킵니다. 마지막 레이어를 거치면 $d\times d$사이즈의 $k$개의 피쳐맵이 나옵니다. 여기서 어떤 오브젝트가 어떤 의미를 가지고 있는지 알 수 없으므로 오브젝트 전체 관계를 각각 쌍으로 만들어서 입력해줍니다. 좀 더 자세히 말하자면, $d\times d\times k$ 사이즈의 피쳐맵 중에서 $1\times 1\times k$를 하나의 오브젝트로 해 주고 모든 오브젝트들의 2개 조합을 만듭니다. 이런 식으로 조합을 만들면 각 오브젝트쌍(object pair) 간의 어떠한 관계를 표현할 수 있습니다.

## Conditioning RNs with question embeddings

오브젝트와 오브젝트 사이의 관계는 사실 질문이 어떤가에 따라서 완전히 달라질 수 있습니다. 만약 질감에 관한 질문이 나온다면 질감에 관련된 오브젝트 쌍들이 더 큰 의미를 가지게 됩니다. 따라서 RN $ { g } _ { \theta  } $에 질문을 집어넣어 다음과 같이 수정합니다.

$$a={ f }_{ \phi  }(\sum _{ i,j }^{  }{ { g }_{ \theta  }({ o }_{ i },{ o }_{ j },q)). } $$

제일 마지막에 $q$가 들어가게 됩니다. 질문에 들어가는 단어들은 전부 lookup table에 집어넣고 LSTM에 들어가는 입력값으로 사용됩니다. LSTM의 산출물을 $q$에 그대로 집어넣습니다.

## Dealing with state descriptions

CLEVR 데이티셋의 description 버젼은 RN에 그대로 집어넣습니다. 질문은 그대로 LSTM을 사용합니다.

## Dealing with natuural language

![Imgur](https://i.imgur.com/haUmUTR.png)

bAbI 데이터셋은 텍스트로만 이루어져있기에 조금 다른 과정을 거칩니다. 먼저 서포트 문장들 중 질문 바로 전에 나오는 문장들을 따로 지정해놓습니다. 그리고 이 문장들을 서포트 문장 세트 안에서 상대적인 위치를 지정해주고 단어별로 LSTM을 집어넣습니다. 질문 문장은 visual QA에서와 마찬가지로 똑같이 LSTM을 거쳐줍니다. 

## Model configuration details

모델의 세세한 디테일은 아래와 같습니다.
- 4 conv layerss with each 24 kernels, ReLU, batch normalization
- 128 unit LSTM for question processing
- 32 unit work-lookup embeddings

$ { g } _ { \theta  } $는 4개 층으로 이루어진 MLP이고 각 층마자 256개의 히든 노드와 ReLU로 이루어져있습니다. ${ f }_{ \phi  }$는 3개 층이며 각각 256, 256(dropout 50%) 그리고 29개의 히든 노드와 ReLU로 이루어져 있습니다. 

#Results & Conclusions

<div align="center"><a href="https://imgur.com/UG7rTBP"><img src="https://i.imgur.com/UG7rTBP.png"/></a></div>

결과는 놀랍게도 인간을 뛰어넘는(super-human) 성능을 보여줍니다. 사실 이 데이터셋을 사람에게 실험했을 때 우리가 수능풀듯이 엄청 열심히(...)풀지는 않았기에 사람의 정확도가 그렇게 높지 않은 것 같기도 하지만 결론적으로 기존 방법론들보다 월등한 성능을 보여주게 됩니다. ~~(사실 논문 한 번 써보면 연구 결과물이 매우 좋다는걸 강조하는것도 중요한 것 중 하나라는걸 알 수 있습니다)~~

![Imgur](https://i.imgur.com/ynpBaJX.png)

이 논문의 가장 컨 contribution은 이렇게 간단히 모델 뒤에 붙이는 것만으로 관계성을 추론하는 능력을 비약적으로 향상시킬 수 있는 방법을 찾아냈다는 것에 있는 것 같습니다. 이전까지 사람이 월등하게 잘 한다고 생각하던 문제들을 전혀 복잡하지 않은 방법으로도 비슷한 수준까지 끌어올릴 수 있다는게 참 신기하네요. Deepmind의 저력을 느낄 수 있는 논문이기도 했습니다. 앞으로 여기서 더 발전해서 훨씬 복잡한 문제들을 척척 풀어내는 그런 모델이 나오는 것도 시간문제일 듯 합니다.
