---
title: 혼합 가우시안 밀도 추정법(Mixture of Gaussian Density Estimation)
description: 밀도 기반 이상치 탐지법(Density based novelty detection) 중에서 여러개의 가우시안 분포를 따른다는 가정 하에 탐지하는 혼합 가우시안 밀도 추정법(Mixture of Gaussian density estimation)에 대해 알아보도록 하겠습니다. 
category: Novelty Detection
tags:
- density based novelty detection
- novelty detection
---


시작하기 전에 앞서 이 글은 고려대학교 강필성 교수님의 Business Analytics 강의를 정리했음을 밝힙니다.


# Mixture of Gaussian Density Estimation

[저번 포스트](https://jayhey.github.io/novelty%20detection/2017/11/02/Novelty_detection_Gaussian/)에서 설명했던 가우시안 밀도 추정법(Gaussian density estimation)이 데이터가 하나의 가우시안 분포를 따르고 convex하다는 가정을 하고 있습니다. 하지만 단 하나의 가우시안 분포만으로 설명하기에는 데이터가 더 복잡하다면 더 좋은 추정법이 있습니다. 그게 바로 가우시안 혼합 모델(Mixture of Gaussian density estimation)입니다. 

<div align ="center"><a href="https://imgur.com/Z33QhxM"><img src="https://i.imgur.com/Z33QhxM.png" /></a></div>

위 그림은 특정 데이터가 4개의 혼합 가우시안 분포를 따른다고 가정한 케이스 입니다. 만약 1개의 가우시안 분포로 추정한다고 하면 혼합 가우시안 분포보다 더 설명력이 낮은 분포가 그려질 것을 알 수 있습니다.

혼합 가우시안 분포의 특징을 정리하면 다음과 같습니다.

- 여러 개의 가우시안 분포로 확장
- 정규 분포들의 선형 조합으로 만들어짐
- 단일 가우시안 분포보다 작은 bias를 가지고 있음. 그러나 학습에 더 많은 데이터가 필요.

## Components of MoG(Mixture of Gaussian)

이상치가 아닌 일반 데이터일 확률 $p(x)$는 다음과 같이 구할 수 있습니다.

$$p(x|\lambda )=\sum _{ m=1 }^{ M }{ { w }_{ m }g(x|{ \mu  }_{ m },{\sum} _{ m }  } )$$

${ \mu  }_{ m }$와$\sum _{ m }$는 파라미터입니다. 그리고 $M$은 총 몇개의 혼합 가우시안 분포를 따를지 사용자가 지정해주어야 하는 하이퍼 파라미터입니다. 시그마 바로 옆에 있는 ${w}_{m}$은 각 가우시안 분포의 가중치를 뜻합니다 

혼합 가우시안 모델의 분포식은 다음과 같습니다.

$$g(x|{ \mu  }_{ m },{ \sum   }_{ m })=\frac { 1 }{ { (2\pi ) }^{ d/2 }{ |{ \sum   }_{ m }| }^{ 1/2 } } exp[\frac { 1 }{ 2 } (x-{ \mu  }_{ m })^{ T }{ { \sum   } }_{ m }^{ -1 }(x-{ \mu  }_{ m })]$$


$$\lambda =\left\{ { w }_{ m },{ \mu  }_{ m },{ \sum   }_{ m } \right\} ,m=1,\cdots ,M$$

기존 가우시안 모델과는 다른점으로 $m$번째 가우시안 모델에 대한 분포가 추가된 것을 확인할 수 있습니다. 

## Expectation-Maximization Algorithm(기댓값 최대화 알고리즘)

그렇다면 혼합 가우시안 모델은 어떤 방식으로 분포를 추정할까요? 단일 가우시안 분포는 convex하기 때문에 해가 정확하게 딱 정해져 있습니다. 그래서 쉽게 최대우도추정법(Maximum likelihood estimation)으로 최적값을 찾아낼 수 있었습니다. 하지만 혼합 가우시안 모델은 convex하지 않아 정해진 최적값을 한 번에 찾을 수 없기 때문에 휴리스틱 기법들로 풀어나가야 합니다. 

![Imgur](https://i.imgur.com/4POAWm8.gif)

그 중 EM알고리즘(Expectaion-Maximination Algorithm)을 사용하여 추정할 수 있습니다. 이 알고리즘은 매개변수에 관한 추정값으로 log-likelihood의 기댓값을 계산하는 expectation 단계와 이 기댓값을 최대화하는 maximization 단계를 번갈아가며 적용합니다. 위 그림은 실제로 휴리스틱 알고리즘으로 분포를 추정해 나가는 모습입니다.


![Imgur](https://i.imgur.com/Mqq71HD.png)


EM 알고리즘을 혼합 가우시안 모델에 적용하게 되면, E-Step에서는 각 개체 기준 modal에 속할 확률를 합니다. 그리고 M-Step에서는 전체 dataset 관점(${ \mu  }_{ m }$,$\sum _{ m }$)에서 각 modal의 가중치을 구합니다. 

이 과정에 대한 증명은 [위키피디아](https://ko.wikipedia.org/w/index.php?title=%EA%B8%B0%EB%8C%93%EA%B0%92_%EC%B5%9C%EB%8C%80%ED%99%94_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98&action=edit&section=13)에 자세히 나와있습니다. 그렇게 구한 Expectaion과 Maximization 식을 정리해보면 다음과 같습니다.

- Expectation

$$p(m|{ x }_{ i },\lambda )=\frac { { w }_{ m }g({ x }_{ t }|{ \mu  }_{ m },{ m }_{ m }) }{ \sum _{ k=1 }^{ M }{ { w }_{ k }g({ x }_{ t }|{ \mu  }_{ k },{ m }_{ k }) }  } $$


- Maximization(차례대로 각 모달 가중치, 평균, 분산)

$${ w }_{ m }^{ (new) }=\frac { 1 }{ N } \sum _{ i=1 }^{ N }{ p(m|{ x }_{ i },\lambda ) } $$

$${ \mu  }_{ m }^{ (new) }=\frac { p(m|{ x }_{ i },\lambda ){ x }_{ i } }{ \sum _{ i=1 }^{ N }{ p(m|{ x }_{ i },\lambda ) }  } $$

$${ \sigma  }_{ m }^{ 2(new) }=\frac { \sum _{ i=1 }^{ N }{ p(m|{ x }_{ i },\lambda ){ x }_{ i }^{ 2 } }  }{ \sum _{ i=1 }^{ N }{ p(m|{ x }_{ i },\lambda ) }  } -{ \mu  }_{ m }^{ 2(new) } $$


## Covariance matrix type

단일 가우시안 밀도 추정에도 공분산 행렬에 대한 조건이 다른 것처럼 혼합 가우시안 모델에서도 마찬가지로 총 3가지의 공분산 행렬 종류가 있습니다.

### Spherical

$${ \sigma  }^{ 2 }=\frac { 1 }{ d } \sum _{ i=1 }^{ d }{ { \sigma  }^{ 2 } } ,\quad \sum  ={ \sigma  }^{ 2 }\left[ \begin{matrix} 1 & \cdots  & 0 \\ \vdots  & \ddots  & \vdots  \\ 0 & \cdots  & 1 \end{matrix} \right] $$

<div align="center">
<a href="https://imgur.com/qz0SpKd"><img src="https://i.imgur.com/qz0SpKd.png" width=350 /></a></div>

Spherical 공분산 행렬을 사용하면 역시나 마찬가지로 조금 덜 정밀하다는 단점이 있습니다. 하지만 계산 복잡도가 줄어드는 단점이 있습니다. 싱글 가우시안 밀도 추정에서는 단일 분포이므로 쉽게 빠른 시간 안에 추정이 가능하지만 혼합 모델은 시간이 더 걸릴 수 있는 관계로 데이터가 크면 계산 복잡도가 중요 고려 요소에 들어갈 수도 있습니다.

### Diagonal

$$\sum  =\left[ \begin{matrix} { { \sigma  } }_{ 1 }^{ 2 } & \cdots  & 0 \\ \vdots  & \ddots  & \vdots  \\ 0 & \cdots  & { { \sigma  } }_{ d }^{ 2 } \end{matrix} \right] $$

<div align="center"><a href="https://imgur.com/dao2A51"><img src="https://i.imgur.com/dao2A51.png" width=350 /></a></div>

Spherical보다는 더 정밀하고 마찬가지로 full 공분산 행렬을 사용하는 것에 비해 계산복잡도가 더 낮습니다.

### Full

$$\sum  =\left[ \begin{matrix} { \sigma  }_{ 11 } & \cdots  & { \sigma  }_{ 11 } \\ \vdots  & \ddots  & \vdots  \\ { \sigma  }_{ d1 } & \cdots  & { \sigma  }_{ dd } \end{matrix} \right] $$


<div align="center"><a href="https://imgur.com/dDkxgW5"><img src="https://i.imgur.com/dDkxgW5.png" width=350" /></a></div>

가장 정밀하지만 그만큼 계산 복잡도도 증가하게 됩니다. 또한 변수가 많아지면 복잡해지면서 공분산 행렬이 singular matrix가 될 수 있다는 리스크가 있습니다.