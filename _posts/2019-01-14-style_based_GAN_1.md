---
title: A Style-Based Generator Architecture for GANs - 1
description: 2018년 12월에 나온 GAN의 generator 구조 관련 논문입니다. 기존 GAN의 generator(생성기)들의 한계점을 극복하고 한단계 더 나아갈 수 있는 방향을 제시하였습니다. 생성기의 구조를 어떤 식으로 바꿨는지 살펴보도록 하겠습니다.
category: Deep Learning
tags:
- GAN
- CNN
---

시작하기에 앞서 이 포스트는 Deep paper 스터디 조원인 [정지원님의 발표영상](https://youtu.be/TWzEbMrH59o)을 참고하였음을 밝힙니다.

# Style-Based Generator for GANs - overview

Nvidia에서 2018년 12월에 올린 논문입니다. GAN의 generator(생성기)들의 한계점을 극복하고 한단계 더 나아갈 수 있는 방향을 제시하였습니다. 기존 GAN의 구조와는 다르게 거꾸로 생각하여 손쉬운 방법으로도 결과물에 대한 체리피킹 없이도 놀라운 결과를 보여주고 있습니다.

## Introduction

작년 즈음 PG-GAN 공부를 하다가 궁금해서 직접 학습 후 이미지를 생성해본 적이 있습니다. 풀 데이터셋으로 학습시키기에는 연구실의 gtx 1080ti 한 장으로 조금 벅찬 감이 있어서, 네트워크 규모도 줄이고 데이터도 줄여가며 딱 일주일 정도 컴퓨터를 혹사시켰습니다. 그러나 아래 그림처럼 결과가 생각보다 좋지 않아 실망했던 기억이 납니다.

![Imgur](https://i.imgur.com/En88nT9.png)

PG-GAN 논문에서 제시한 왼쪽 그림 만큼 잘 뽑힌 사진이 어느정도 나오긴 했지만..오른쪽에 보이는 8개의 사진이 직접 뽑은 이상한 사진의 예시입니다. GAN 논문들을 보면 대체적으로 왼쪽 사진처럼 잘 나온 그림들을 체리피킹하여 제시합니다. 그러나 실제로 모델을 학습시켜보면 저런 식으로 어색한 사진들도 많이 생성됩니다. 이는 latent space $z$를 학습 이미지들에 최대한 맞추려고 하다보니 분포가 조금 뭉개져서 그렇습니다. 아래 영상의 1분 51초부터 보시면 PG-GAN의 latent space interpolation 결과를 확인할 수 있습니다. Interpolation 과정에서 상당히 어색한 이미지들이 나오는 것이 보입니다.

<div align='center'>
<iframe width="560" height="315" src="https://www.youtube.com/embed/XOxxPcy5Gr4?start=112" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

이번 포스트에서 설명드릴 style-based generator는 이런 문제점을 해결할 수 있는 간단하면서도 효과적인 방법을 제시합니다. 방법론 설명에 앞서 이 논문의 contribution은 다음과 같습니다.

- 어떠한 GAN 구조에도 바로 적용할 수 있는 generator 구조를 제안하였습니다. 또한 여기서는 discriminator(분류기)를 수정하거나 손실함수를 바꾸는 등 어떠한 변경점도 없는 generator만 관련된 논문입니다.
- 이미지 합성을 하는 과정에서 이미지의 전체적인 스타일(머리 색, 성별, 인종 등)과 세세한 부분(머리카락의 세세한 위치, 곱슬거림 정도 등)까지 조정할 수 있습니다.
- Latent space의 interpolation quality(disentanglement)를 측정할 수 있는 measure 제안하였습니다.
  - Perceptual path length
  - Linear separability
- CelebA-HQ보다 더 고화질이면서도 훨씬 다양한 종류의 사람 얼굴을 포함하고 있는 Flickr-Faces_HQ(FFHQ)데이터셋을 공개하였습니다.


## Style-based generator

![Imgur](https://i.imgur.com/TcrMZ4O.png)

네트워크의 전체적인 구조와 결과물은 위 그림과 같습니다. 왼쪽 그림에서 (a)가 기존 GAN의 generator 구조입니다. latent space $z$가 직접 네트워크에 들어가서 바로 변형이 되고 있습니다. 그러나 style-based generator의 경우 $z$가 직접 convolution으로 들어가지 않고 mapping network를 통과합니다. 이후 변형된 $w$를 **이미 학습 된 텐서에 스타일을 입히는 방법**을 사용하였습니다. 간단하지만 발상의 전환입니다. 여기서 이미 학습 된 텐서라는건 학습 데이터들의 style이 하나도 가미되지 않은 평균적인 얼굴을 표현하는 텐서라고 볼 수 있습니다.
 
### Mapping network

<img src="https://i.imgur.com/htVvbjC.png" title="Mapping network(그림 출처: Rani Horev's blog)" />


기존 GAN의 방식대로라면 latent space가 학습 데이터의 분포를 직접적으로 따라가게 됩니다. 하지만 이렇게 시행할 경우 필연적으로 위에서 언급했던 문제점인 interpolation이 부자연스럽게 되는 features entanglement 현상이 발생합니다. 예를들어 서양인 80%, 동양인 20%로 이루어진 이미지 데이터를 가지고 generator를 학습한다면 latent space 또한 편중되어 있는 데이터의 분포를 따라갈 수 밖에 없습니다. 하지만 또다른 비선형 mapping network $f$를 통과시킨다면, 학습 데이터의 분포를 따라갈 필요가 없으면서도 feature들 사이의 편중된 상관관계를 줄여줄 수 있습니다. 이러한 이유로 $z$를 비선형 함수인 mapping network $f$를 통과시켜 나온 $w$를 사용합니다. $f$는 8개의 FC layer로 이루어진 단순한 인공신경망입니다. $z$와 $w$는 512차원으로 같습니다.

여기에 Adaptive Instance Normalization(AdaIN)을 적용하기 위한 추가적인 연산(그림에서 A)과 noise를 더해주기 위한 추가적인 연산(그림에서 B)를 거치면 generator 구조가 완성됩니다. 

### AdaIN for styling

<div align='center'>
<img src="https://i.imgur.com/GPpL1yH.png" title="Adaptive Instance Normalization 적용 방법(그림 출처: Rani Horev's blog)" />
</div>

이제 Mapping network $f$를 통과한 $w$를 사용하여 해당 텐서에 스타일을 입힐 차례입니다. 그림의 Synthesis Network는 처음 4x4x512짜리 텐서로 시작해서 1024x1024x3으로 끝나는 8개의 레이어로 이루어져 있습니다. 해당 레이어마다 upsampling과 convolution operation이 끝난 뒷부분에 AdaIN을 적용합니다. 

<!-- <div align='center'>
<img src="https://i.imgur.com/2UxWWYN.png" title="Affine transformation" />
</div> -->


다만, $w$는 512개로 AdaIN을 적용하기엔 채널 개수와 사이즈가 다르기 때문에 이곳에서 아핀 변환(affine transformation)을 적용합니다. 


(이 부분이 지금 굉장히 헷갈리는 부분입니다. 단순히 $2n$개 만큼 output을 주는 FC layer를 사용해서 결과물을 $n$개의 scale과 $n$개의 bias에 사용했는지 아니면 다른 방법을 사용했는지 확실하지가 않습니다. 아핀 변환이라는 말을 굳이 쓴 이유를 잘 모르겠습니다...코드가 공개된다면 보고 업데이트 할 수 있도록 하겠습니다. 혹시 아시는 분 있으면 댓글 부탁드려요!)


$$AdaIN\left( { x }_{ i },y \right) ={ y }_{ s,i }\frac { { x }_{ i }-\mu ({ x }_{ i }) }{ \sigma \left( { x }_{ i } \right)  } +{ y }_{ b,i }$$

$2 \times n$개의 파라미터를 가지고 이제 기존 채널에 스타일을 입힐 차례입니다. Convolution output 각 채널들을 먼저 정규화 시킨 후 앞서 구한 스타일 함수 $y$를 각 채널 별로 적용합니다. 이렇게 되면 해당 채널들에 원하는 스타일을 입힐 수 있습니다.


### Add noise for stochastic variation

<img src="https://i.imgur.com/151Q1lx.png" title="노이즈 더해주기(그림 출처: Rani Horev's blog)" />

AdaIN이 이미지의 큼직한 부분(style)들을 바꾸는 방법이라면, 이미지의 세세한 부분을 바꾸기 위해(stochastic variation) noise를 더하는 방법을 사용하였습니다. 큼직한 스타일이 인종, 성별 등과 같이 눈에 띄게 큰 부분이라면 세세한 부분은 짧은 수염, 머리가 흩날리는 정도, 주름살의 위치 등을 말합니다. 적용은 위의 AdaIN 방식과 똑같은 방법을 사용하였으며 랜덤한 가우시안 노이즈를 각 채널별로 집어넣는 방식을 사용하였습니다. 


## 정리하면...

이번 포스트에서는 네트워크의 구조에 대해 알아보았습니다. 특징을 간략하게 정리하면 다음과 같습니다.

- $z$를 집어넣어 다이렉트로 이미지를 생성하는 것이 아니라 비선형 mapping function을 거친 $w$를 먼저 구함
- 유동적인 $z$ 대신 학습된 고정 텐서(쉽게 생각해서 얼굴 사진을 예로 들면, 모든 데이터의 평균적인 얼굴)에 $w$를 사용하여 큼직하게 스타일을 더해줌
- 생성되는 이미지에 가우시안 노이즈를 더하여 세세한 부분을 수정

자세한 실험결과, 어째서 $z$가 아닌 $w$를 쓰는게 더 좋은 결과가 나왔는지 그리고 기타 다양한 trick에 대한 설명은 다음 포스트에 이어서 쓰도록 하겠습니다.


> Reference
> * [Karras, T., Laine, S., & Aila, T. (2018). A Style-Based Generator Architecture for Generative Adversarial Networks. arXiv preprint arXiv:1812.04948.](https://arxiv.org/pdf/1812.04948.pdf)
> * [Rani Horev's blog](https://towardsdatascience.com/explained-a-style-based-generator-architecture-for-gans-generating-and-tuning-realistic-6cb2be0f431) 
> * [PR-131 youtube (발표자 : 정지원)](https://youtu.be/TWzEbMrH59o) 


