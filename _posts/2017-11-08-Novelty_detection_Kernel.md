---
title: 커널 밀도 추정(Kernel density estimation) - Parzen window density estimation
description: 다른 밀도 추정법들이 데이터가 특정 분포를 따른다는 가정 하에 추정하는 방법이었습니다. 하지만 이번에 설명할 커널 밀도 추정은 데이터가 특정 분포를 따르지 않는다는 가정 하에 밀도를 추정하는 방법입니다. 커널 밀도 추정의 기본적인 개념을 알아보고 대표적인 그 중 파젠 윈도우 밀도 추정(Parzen window density estimation)에 대해 알아보겠습니다.
category: Novelty Detection
tags:
- density based novelty detection
- non-parametric
---


시작하기 전에 앞서 이 글은 고려대학교 강필성 교수님의 Business Analytics 강의를 정리했음을 밝힙니다.


# Kernel-density Estimation

이전 포스트에서 다뤘던 [가우시안 분포 추정](https://jayhey.github.io/novelty%20detection/2017/11/02/Novelty_detection_Gaussian/) 그리고 [가우시안 혼합 모델](https://jayhey.github.io/novelty%20detection/2017/11/03/Novelty_detection_MOG/)은 데이터가 가우시안 분포(Gaussian distribution)을 따른다는 가정 하에 데이터의 분포를 찾아가는 방법이었습니다. 하지만 이번에 다룰 커널 밀도 추정(kernel density estimation)은 데이터가 특정 분포를 따르지 않는다는 가정 하에서 밀도를 추정해 나가는 방법입니다. 아래 그림을 보면 가우시안 분포처럼 일정한 분포가 아닌 완전히 다른 분포를 추정해 낸 것을 알 수 있습니다.

![Imgur](https://i.imgur.com/dggz7Q5.png)

위 그림은 3D로 나타낸 경우이고, 1-D 데이터에 대한 예시는 아래와 같습니다. 회색 음영 부분이 실제 정답 분포입니다. 커널 종류를 바꿔감에 따라서 다른 모양을 추정하는 것도 확인할 수 있습니다.

<div align="center"><a href="https://imgur.com/E95iRvE"><img src="https://i.imgur.com/E95iRvE.png" width=500 /></a></div>


## 자세히 살펴보기


만약 $p({ x })$의 분포에서 region R 사이에서 벡터 $x$가 추출 될 확률을 $P$라고 하면 다음과 같이 나타낼 수 있습니다. 

$$P=\int _{ R }^{  }{ p({ x }^{ \prime  })dx^{ \prime  } } $$

여기서 $p(x)$는 어떠한 확률 밀도 함수입니다. 이게 적분이 된다는 말은 범위 R 내에서는 $p({ x }^{ \prime  })$가 연속인 밀도 함수라는 말입니다. 

이제 $N$ 개의 벡터 $\{ { x }^{ 1 },{ x }^{ 2 },... ,{ x }^{ n }\} $가 있다고 해봅시다.
이 $N$개의 벡터 중 $k$개가 공간 R에 들어갈 확률을 구하면 이항분포를 따르므로 다음과 같습니다.

$$P(k)=\left( \begin{matrix} N \\ k \end{matrix} \right) { P }^{ k }(1-P)^{ N-k }$$

이를 통해 평균과 분산을 알 수 있습니다.

$$E(k)=NP\rightarrow E\left[ \frac { k }{ N }  \right] =P$$

$$Var(k)=NP(1-P)\rightarrow Var\left[ \frac { k }{ N }  \right] =\frac { P(1-P) }{ N } $$

N이 무한대($\infty $)로 가면 이 분포가 점점 더 뾰족(sharp)하게 됩니다. 사실상 분산이 0에 가까워지므로 $E()$를 제거해도 거의 똑같은 값을 구할 수 있습니다. 

$$ P\cong \frac { k }{ N }  $$

그리고 이전 적분 식에서 region $R$을 굉장히 작게 만들면 $P$를 상수처럼 취급할 수 있습니다. 

$$P=\int _{ R }^{  }{ P({ x }^{ \prime  })d{ x }^{ \prime  } } \cong p(x)V$$

$V$는 region $R$을 통해 근사시킨 넓이입니다. 사실상 영역의 넓이는 사각형의 넓이를 구하는 것과 같다고 볼 수 있습니다. 이제 이전 식을 전부 합치면 다음과 같은 결과가 나옵니다.

$$ P=\int _{ R }^{  }{ P({ x }^{ \prime  })d{ x }^{ \prime  } } \cong p(x)V=\frac { k }{ N } ,\quad p(x)=\frac { k }{ NV } $$

이태까지 살펴본 식들을 보면 다음과 같은 조건이 필요하다는 것을 알 수 있습니다.

- R이 충분히 작아야 함(적분 결과가 특정 상수값이 될 수 있을 정도로) 
- $N$개 벡터 중에서 $K$개가 $R$ 안에 들어갈 수 있을 만큼 충분히 커야 함

이는 서로 반대되는 조건이라서 어렵긴 하지만 이론적으로 만족시켜야 합니다. 그리고 위 식에서 $K$와 $V$를 어떻게 조정하냐에 따라서 커널 밀도 추정이 되거나 K-최근접 이웃 밀도 추정(K-nearest neighbor density estimation)이 결정됩니다. $K$를 고정시키고 $V$를 결정하면 K-nn 밀도 추정이 되는 것이고 $V$를 고정시키고 $K$를 결정하면 커널 밀도 추정이 됩니다.


## Parzen window density estimation

데이터가 $d$차원의 공간에 있다고 가정해봅시다. 위에서 설명했던 영역 $R$은 매우 작은 크기이며 그 부피를 식으로 나타내면 다음과 같습니다. 

$${ V }_{ n }={ h }_{ n }^{ d }$$

<div align="center"><a href="https://imgur.com/R4TAg8r"><img src="https://i.imgur.com/R4TAg8r.png" /></a></div>

예를 들어 위와 같은 3차원의 큐브가 있다고 치면, ${V}={h}^{3}$이 됩니다. 그리고 이 큐브의 정 중앙에 $x$가 있다고 해봅시다. 이 때 우리가 하고싶은건 이에 대한 확률 밀도를 결정하고자 하는 것입니다. 

만약 이 입방체 안에 샘플이 있을 때 갯수를 세는 식을 만들어 보면

$$K({ u })=\left\{ \begin{array}{lll} 1, & |u_{ i }|\le 1/2, & i=1,...,D \\ 0, & otherwise & \;  \end{array} \right. \qquad $$


$$k=\sum _{ i=1 }^{ N }{ K(\frac { { x }^{ i }-x }{ h } ) } $$


이렇게 표현할 수 있습니다. $k$가 입방체 안에 있는 샘플의 갯수가 되는 것입니다. 이 말은 $x$가 주어졌을 때 이를 중심으로 입방체 안에 몇 개의 샘플이 있는지 확인하는 것과 같습니다. 어렵게 생각할 필요 없이 주어진 점 $x$를 기준으로 각 차원으로 $h/2$안에 존재하는 모든 샘플의 갯수를 셉니다. 위와 같은 함수를 커널 함수(kernel method)의 일종이며 파젠 윈도우(Parzen window)라고 합니다. 

이제 $p(x)$에 그대로 대입해보겠습니다.

$$p(x)=\frac { 1 }{ N{ h }^{ d } } \sum _{ i=1 }^{ N }{ K(\frac { { x }^{ i }-x }{ h } ) } $$

이게 기존 파젠 윈도우 밀도 추정(Parzen window density estimation)을 통해 추정한 데이터의 확률 밀도 함수 입니다.

### Smoothing

하지만 위에서 제시한 $K(U)$함수에는 단점이 있습니다. 이는 영역 안이면 무조건 1, 밖이면 무조건 0을 준다는 점입니다. 큐브의 가장자리 영역에서는 결국 불연속적인 값을 가질 수 밖에 없습니다. 즉, discrete한 밀도 표현이 문제가 되고 연속형 밀도 추정에 부적합하다는 것입니다.

그래고 스무딩(smoothing)이라는 기법을 사용합니다. 

![Imgur](https://i.imgur.com/PBicTz0.png)

이렇게 표현하면 가장자리에서 연속적인 값도 표현할 수 있습니다. 예를 들어서 가우시안 커널을 사용하면 $p(x)$가 다음과 같이 바뀌게 됩니다.

$$p({\bf x}) = \frac{1}{N}\sum_{n=1}^N\frac{1}{(2\pi h^2)^{D/2}}\exp\left\{-\frac{\|{\bf x}-{\bf x}_n\|^2}{2h^2}\right\} $$

쉽게 생각해서 관찰한 각 데이터 별로 하나의 가우시안 분포가 만들어지고 최종적으로 이를 선형 결합하면 알고자 하는 데이터 분포가 됩니다. 

사용자가 이 커널을 어떻게 지정하느냐에 따라 최종적으로 추정한 분포의 모습이 바뀌게 됩니다. 가우시안 커널을 사용할 수도 있고, 유니폼을 사용할 수도 있으니 상황에 맞는 적절한 커널을 선택하면 되겠습니다. [위키피디아 커널](https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use)을 보면 여러 커널들에 대한 설명이 나와 있습니다. 


<div align="center"><a href="https://imgur.com/JoFpY65"><img src="https://i.imgur.com/JoFpY65.png" width=500 /></a></div>

### Smoothing parameter h

위에 있는 3차원의 길이가 $h$인 정육면체 그림을 보시면, 이 값을 어떻게 지정하느냐에 따라서 분포의 최종적인 모형이 달라질 수 있다는 것이 예상되실 겁니다. $h$를 큰 값으로 잡게 된다면 굉장히 밀도 있는 완만한 분포가 추정되고, $h$를 작은 값으로 지정해준다면 뾰족뾰족한 분포로 추정되게 됩니다. 이를 smoothing parameter 또는 bandwidth라고 하는데 잘못 지정할 경우 밀도가 과도하게 부드러워져서 데이터의 구조를 덮어버리는 oversmooth 현상이나 분석하기 어려울 정도로 엉성하게 피크가 형성되는 undersmooth 현상이 일어날 수 있습니다.

![Imgur](https://i.imgur.com/iumNcnR.png)

