---
title: 가우시안 밀도 추정법(Gaussian density estimation)
description: 이상치 탐지 기법에도 여러가지 방법이 있습니다. 그 중에서 밀도 기반 이상치 탐지법(Density based novelty detection)을 앞으로 몇 개 소개해 드릴텐데요, 가장 먼저 가우시안 밀도 추정법(Gaussian density estimation)에 대해 설명드리겠습니다.
category: Novelty Detection
tags:
- density based novelty detection
- novelty detection
---


시작하기 전에 앞서 이 글은 고려대학교 강필성 교수님의 Business Analytics 강의를 정리했음을 밝힙니다.


# Density-based Novelty Detection

## Purpose

밀도 기반 이상치 탐지법이란 데이터의 분포를 사용하여 이상치를 찾아내는 방법입니다. 기존에 존재하는 데이터 분포를 사용하여 데이터에 맞는 분포를 추정을 합니다. 이를 통해 분포를 추정하면 기존 데이터 중에서도 이상치를 찾아낼 수 있고, 새로운 데이터가 들어와도 그대로 추정한 분포에 집어넣으면 이상치인지 확인할 수 있습니다.


![Imgur](https://i.imgur.com/YcugOBe.png)

위 그림을 예시로 들어보면 낮은 밀도 부분에 있는 데이터(빨간 부분)는 이상치입니다.
결론적으로 이 방법론들의 목적은 **데이터를 사용하여 추정한 분포를 가지고 이상치를 탐지하는 것**이라고 할 수 있습니다. 기존 데이터로 분포를 "학습" 시키고 새로운 데이터가 들어와도 적용이 가능한 점을 보면 지도학습(supervised learning)의 일종이라고 볼 수도 있습니다.

앞으로 총 5개의 밀도 기반 이상치 탐지 기법들을 살펴보도록 하겠습니다.

- Gaussian Density Estimation
- Mixture of Gaussian Density Estimation
- Kernel-density Estimation
- Parzen Window Density Estimation
- Local Outlier Factors(LOF)

# Gaussian Density Estimation

먼저 살펴볼 방법론은 가우시안 밀도 추정입니다. 이는 데이터가 하나의 정규 분포를 따른다고 가정하고 사용하는 방법론입니다.

![Imgur](https://i.imgur.com/o0wdkoW.png)

$$p(x)=\frac { 1 }{ { (2\pi ) }^{ d/2 }{ |\sum  | }^{ 1/2 } } exp[\frac { 1 }{ 2 } (x-\mu )^{ T }{ \sum   }^{ -1 }(x-\mu )]$$
$$\mu =\frac { 1 }{ 2 } \sum _{ { x }_{ i }\in X^{ + } }^{  }{ { x }_{ i } } (mean\quad vector)$$
$$\sum  =\frac { 1 }{ 2 } \sum _{ { x }_{ i }\in X^{ + } }^{  }{ ({ x }_{ i }-\mu )({ x }_{ i }-\mu )^{ T } }, \quad (covariance\quad matrix)$$

위 식에서 ${ X }^{ + }$는 정상 영역인 부분을 뜻합니다. 

## Advantages

가우시안 밀도 추정은 두 가지 장점이 있습니다. 먼저 데이터를 스케일링하는데 있어서 민감하지 않습니다. 

$$\frac { 1 }{ 2 } (x-\mu )^{ T }{ \sum   }^{ -1 }(x-\mu )$$

확률값 $p(x)$를 구하는 식에서 exp 뒤에 있는 부분이 마할라노비스 거리를 구하는 식입니다. 이는 데이터의 각 변수 별 분산을 고려해서 거리를 구하게 해줍니다. 즉, 데이터를 스케일링하지 않아도 어차피 분산까지 고려한다는 말과 같습니다.

두 번째로 식 자체가 주어져 있기 때문에 추가적인 분석이 가능합니다. $p(X)$를 미분을 하는 등 여러 방법을 사용해서 최적의 임계값을 분석적으로 계산이 가능할 수 있습니다.

## Maximum likelihood estimation

고등학교에서는 정규분포에 대해 언급하긴 하지만 왜 이 정규분포의 평균이 $\mu$이고 분산이 ${ \sigma  }^{ 2 }$인지는 알려주지 않습니다. 교육과정이 아니기에 스리슬쩍 넘어가는게 보통입니다. 이제 최대 우도 추정법(Maximum Likelihood Estimation)으로 이를 증명해보겠습니다.

<div align="center">
<a href="https://imgur.com/gofjkS6"><img src="https://i.imgur.com/gofjkS6.png" width="450px"/></a></div>

위에서 파란색과 빨간색 정규분포가 있다고 가정하고 x자는 데이터의 위치라고 해봅시다. 그렇다면 데이터가 파란색 분포를 따른다고 할 때와 주황색 분포를 따른다고 할 때 언제 더 설득력이 있을까요? 주황색 분포를 따른다고 할 떄가 더 설득력이 있다고 할 수 있습니다. 그림에서 x자의 데이터를 주황색 정규분포 식에 집어넣으면 $p(x)$값이 나옵니다. **이 값들을 곱했을 때 가장 최대가 되는 분포가 가장 맞는 정규분포라고 할 수 있습니다.** 이를 식으로 표현하면 다음과 같습니다. 1차원 데이터 기준으로 설명하도록 하겠습니다.

$$L=\prod _{ i=1 }^{ N }{ P({ x }_{ i }|\mu ,{ \sigma  }^{ 2 }) } =\prod _{ i=1 }^{ N }{ \frac { 1 }{ \sqrt { 2\pi  } \sigma  } exp(-\frac { ({ x }_{ i }-\mu )^{ 2 } }{ 2{ \sigma  }^{ 2 } } ) } $$
$$\log { L } =-\frac { 1 }{ 2 } \sum _{ i=1 }^{ N }{ \frac { ({ x }_{ i }-\mu )^{ 2 } }{ { \sigma  }^{ 2 } }  } -\frac { N }{ 2 } \log { (2\pi { \sigma  }^{ 2 }) } $$


이제 식을 더 쉽게 다루기 위해서 $\gamma =\frac { 1 }{ { \sigma  }^{ 2 } } $로 치환해서 식을 더 써보도록 하겠습니다.


$$\log { L } =-\frac { 1 }{ 2 } \sum _{ i=1 }^{ N }{ \gamma ({ x }_{ i }-\mu )^{ 2 } } -\frac { N }{ 2 } \log { (2\pi ) } +\frac { N }{ 2 } log(\gamma )$$

$$\frac { \partial log{ L } }{ \partial \mu  } =\gamma \sum _{ i=1 }^{ N }{ ({ x }_{ i }-\mu ) } =0$$
$$ \rightarrow \quad \mu =\frac { 1 }{ N } \sum _{ i=1 }^{ N }{ { x }_{ i } } $$

$$\frac { \partial log{ L } }{ \partial \gamma  } =-\frac { 1 }{ 2 } \sum _{ i=1 }^{ N }{ ({ x }_{ i }-\mu )^{ 2 } } +\frac { N }{ 2\gamma  } =0\quad $$
$$\rightarrow \quad { \sigma  }^{ 2 }=\frac { 1 }{ N } \sum _{ i=1 }^{ N }{ { (x }_{ i }-\mu )^{ 2 } } $$

우리가 아는 식이 드디어 나왔습니다! $\mu$와 $ { \sigma  }^{ 2 }$이 최대 우도 추정법으로 추정한 정규분포의 평균과 분산이라는게 증명됬습니다. 

결국 핵심은 $log-likelihood$가 convex이므로 1차 도함수로 최적값을 찾을 수 있다는 것입니다.


## covariance matrix type

가우시안 밀도 추정에서 사용하는 식은 다음과 같습니다.

$$\mu =\frac { 1 }{ N } \sum _{ i=1 }^{ N }{ { x }_{ i } } ,\quad \sum  =\frac { 1 }{ N } \sum _{ i=1 }^{ N }{ { (x }_{ i }-\mu ){ { (x }_{ i }-\mu ) }^{ T } } $$

위 식에서 $\sum$는 원래대로라면 공분산 행렬 자체를 써야하지만, 여러가지 문제점이 발생할 수 있기에 다음과 같은 방법으로 공분산 행렬을 바꿔서 적용하게 됩니다. 



### Spherical

$${ \sigma  }^{ 2 }=\frac { 1 }{ d } \sum _{ i=1 }^{ d }{ { \sigma  }^{ 2 } } ,\quad \sum  ={ \sigma  }^{ 2 }\left[ \begin{matrix} 1 & \cdots  & 0 \\ \vdots  & \ddots  & \vdots  \\ 0 & \cdots  & 1 \end{matrix} \right] $$

각 변수 별 분산을 전부 평균으로 놓고 단위행렬에다가 곱해서 분산으로 사용합니다. 이렇게 추정하면 diagonal matrix가 대각선이 전부 같은 값을 가지기 때문에 항상 정규분포 모양이 원이 나오게 됩니다. 
오른쪽 등고선 그래프를 보면 전부 원으로 나오는 것을 확인할 수 있습니다.

![Imgur](https://i.imgur.com/vZWtNIk.png)

### Diagonal

$$\sum  =\left[ \begin{matrix} { { \sigma  } }_{ 1 }^{ 2 } & \cdots  & 0 \\ \vdots  & \ddots  & \vdots  \\ 0 & \cdots  & { { \sigma  } }_{ d }^{ 2 } \end{matrix} \right] $$

이번에는 diagonal type입니다. Spherical보다 좀 더 엄격하게 보는 방법입니다. 축이 틀어지지는 않지만(각 변수별로 수직) 값이 다르기 때문에 축 길이는 다르게 됩니다.


![Imgur](https://i.imgur.com/VAIKKYd.png)

### Full

$$\sum  =\left[ \begin{matrix} { \sigma  }_{ 11 } & \cdots  & { \sigma  }_{ 11 } \\ \vdots  & \ddots  & \vdots  \\ { \sigma  }_{ d1 } & \cdots  & { \sigma  }_{ dd } \end{matrix} \right] $$

Full type은 공분산을 전부 고려합니다. Diagonal만 쓰지 않고 전부 쓰게되면 축이 기울어지게 됩니다. 여기까지 설명을 보시면 왜 굳이 무조건 full을 안쓰고 다른 타입의 공분산 행렬을 쓰는지 이해가 안가시는 분들도 계실 겁니다. 이는 변수가 너무 많아지면 공분산 행렬이 singular matrix가 되어서 역행렬을 구하지 못하게 되기 때문입니다. 이런 위험성이 있기 때문에 보통은 spherical을 많이 사용하며 실제 성능도 괜찮게 나옵니다.

![Imgur](https://i.imgur.com/ic7e1Nx.png)




