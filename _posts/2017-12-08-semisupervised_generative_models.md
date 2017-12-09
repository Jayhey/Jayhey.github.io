---
title: 생성 모델(Generative model)
description: 준지도학습(semi-supervised learning) 방법중에서 가장 간단한 셀프 트레이닝(self training)에 대하여 알아보도록 하겠습니다. 높은 확률값이 나오면 가중치를 주는 간단한 방식으로 손쉽게 모델의 성능 향상을 꾀할 수 있는 테크닉입니다.
category: Semi-supervised Learning
tags: 
- machine learning
- semi supervised learning
---

이 글은 고려대학교 강필성 교수님의 Business Analytics 강의를 정리했음을 밝힙니다.

# Generative models

## Discriminative vs. Generative

분류(Classification) 문제는 크게 Generative model과 discriminative model로 나눌 수 있습니다. 조금 생소한 부분이라고 할 수도 있지만 우리가 잘 알고있는 머신러닝 방법들 중 다중회귀분석, 나이브 베이지안 분류기 등 클래스를 분류하는데 집중하는 모델들이 discriminative model입니다. 반면에 generative model은 우도(likelihood)나 사후 확률(posterior probability)를 사용하여 분류 경계선(decision boundary)를 만듭니다. [이상치 탐지 기법 중 밀도 기반 추정법](https://jayhey.github.io/category/#/Novelty%20Detection)들이 대표적인 generative model이라고 할 수 있습니다.

<div align = "center"><a href="https://imgur.com/fVlrzkU"><img src="https://i.imgur.com/fVlrzkU.png" width="450px" /></a></div>

위 그림에서 식을 보면 generative model은 결합확률 분포 자체를 찾는다는 걸 알 수 있습니다. 반면에 discriminative model은 직접적으로 모델을 생성합니다. [이전 포스트에서 다뤘던 셀프 트레이닝(self training)](https://jayhey.github.io/semi-supervised%20learning/2017/12/07/semisupervised_self_training/)은 discriminative model에 적용할 수 있는 테크닉이라고 할 수 있겠습니다. **분류 측면에서 정리하자면 discriminative model은 클래스의 차이점에 주목하는 것이고 generative model은 클래스의 분포에 주목한다고 보면 이해하기 쉽습니다.**

## Examples

![Imgur](https://i.imgur.com/6OpEUyd.png)



위 그림의 데이터가 가우시안 분포를 따른다고 가정해봅시다. 그럼 어떤 decision boundary를 가지게 될까요? ([가우시안 분포에 대한 설명은 이전 포스트를 참고하시면 됩니다](https://jayhey.github.io/novelty%20detection/2017/11/02/Novelty_detection_Gaussian/))


![Imgur](https://i.imgur.com/etE6BAC.png)

왼쪽이 레이블 데이터로 추정한 가우시안 분포이고 오른쪽이 언레이블 데이터를 추가한 모습입니다. 모든 데이터를 뿌려도 레이블 데이터가 뭘로 뽑히냐에 따라서 분포가 완전히 바뀌게 된다는 것을 알 수 있습니다. 단순 레이블 데이터만 가지고 가우시안 분포를 추정하면 사실 언레이블 데이터도 같이 적용하기에는 무리가 있어보입니다.

![Imgur](https://i.imgur.com/691Ixqc.png)

언레이블 데이터를 같이 적용해서 오른쪽 그림과 같은 분포가 나온다면 어느정도 납득할 수 있습니다. 

<div align="center"><a href="https://imgur.com/44vOpnl"><img src="https://i.imgur.com/44vOpnl.png"/></a></div>

이 둘이 이렇게 다른 이유는 왼쪽의 경우 순수 레이블 데이터만 이용하여 분포를 추정했고, 오른쪽은 언레이블 데이터까지 포함하여 분포를 추정했기 때문입니다. 이제 어떻게 언레이블 데이터를 포함하여 분포를 추정하는지 알아봅시다.

## Generative model for semi-supervised learning

준지도학습에서 쓰이는 generative model에 대하여 알아보겠습니다. 우리가 최종적으로 구하고 싶은건 레이블이 달려있지 않은 언레이블 데이터까지 포함해서 분포를 추정하는 것입니다. 즉, 구하고자 하는것을 식으로 나타내면 아래와 같습니다.

$$p({ X }_{ l },{ y }_{ l },{ X }_{ u }|\theta )=\sum _{ { y }_{ u } }^{  }{ p({ X }_{ l },{ y }_{ l },{ X }_{ u },{ y }_{ u }|\theta ) } $$

이를 구하는 방법에는 최대 우도 추정법(Maximum Likelihood Estimation), 최대 사후 확률(Maximum a posteriori), 베이지안(Bayesian) 등이 있습니다. 그 중에서 가우시안 혼합 모델을 사용한 준지도학습에 대해 알아보도록 하겠습니다.

### Gaussian Mixture Mordel

[이전 포스트](https://jayhey.github.io/novelty%20detection/2017/11/02/Novelty_detection_Gaussian/)에서 다룬 내용인데, 최대 우도 추정법을 통해서 GMM을 구해봅시다. 먼저 레이블 데이터만 사용하면 아래와 같이 식을 세울 수 있습니다.

$$p({ X }_{ l },{ y }_{ l }|\theta )=\sum _{ i=1 }^{ l }{ logp(y_{ i }|\theta )p({ x }_{ i }|{ y }_{ i },\theta ) } $$


하지만 위에서도 설명했듯이 레이블 데이터가 적은 경우 이는 불충분합니다. 그래서 식을 추가해주면 아래와 같습니다.

$$p({ X }_{ l },{ y }_{ l }|\theta )=\sum _{ i=1 }^{ l }{ logp(y_{ i }|\theta )p({ x }_{ i }|{ y }_{ i },\theta ) } +\sum _{ i=l+1 }^{ l+u }{ log(\sum _{ y=1 }^{ 2 }{ logp(y|\theta )p({ x }_{ i }|{ y },\theta ) } ) }  $$

뒤에 붙은 언레이블 데이터 식은 $y$를 알 수 없기 때문에 최대우도추정법을 통한 추정이 어렵습니다. 이럴 때 EM알고리즘(Expectation-Maximization algorithm)을 사용하여 로컬 옵티멈(local optimum)을 찾아낼 수 있습니다.

### The EM algorithm for GMM

EM 알고리즘의 순서는 다음과 같습니다.  

먼저 레이블 달린 데이터를 사용한 최대 우도 추정법으로 $\theta =\{ w,\mu ,\Sigma \} $를 구합니다. 그 다음으로는 E-Step(Expectation step)으로 추정한 $\theta$값을 가지고 언레이블 데이터의 확률값을 구합니다. 여기서 확률값이란 것은 단순 이진분류 문제에서 클래스1이나 2에 속할 확률을 말합니다. 소프트 클러스터링(soft clustering)의 결과물과 같다고 보시면 될 것 같습니다. 이렇게 구한 값들로 M-step(Maximization step)을 실시하여 $\theta$값을 업데이트합니다. 그리고 다시 E-step으로 가서 계속 이 과정을 반복하면 최소한 local optimum에 수렴한다는 것을 보장할 수 있습니다. 이 과정이 어떻게 보면 self-training의 특수한 방식처럼 보일수도 있겠네요.

![Imgur](https://i.imgur.com/P82kTl4.png)

EM 알고리즘의 핵심은 사후 확률을 최대화 하는 것(maximize posterior probability)입니다. 물론 EM 알고리즘 말고도 다른 방법(variational approximation, direct optimization 등)을 사용해도 $\theta$값을 구하는데는 전혀 문제가 없습니다. 

## Advantages and disadvantages

### Advantages

Generative model의 장점은 다음과 같습니다.

- 매우 명확하면서 지금까지 계속 꾸준히 연구해온 탄탄한 확률방법론들을 사용합니다.
- 당연한 말이지만, 모델이 올바른 방향으로 학습된다면 굉장히 효과적입니다.

### Disadvantages

- 모델이 얼마나 잘 만들어졌는지 확인하기가 어렵습니다.
- EM 알고리즘으로 local optimum까지는 갈 수 있지만 좋지 않은 local optimum일 수도 있습니다.
- 만약 generative model이 잘못됬다면 잘못된 결과가 산출될 수 있습니다. (예시 아래그림)

![Imgur](https://i.imgur.com/yTwySgy.png)

여기서 첫번째 그림이 올바른 레이블 모양이고 두번째가 generative model로 산출한 분포, 세번째가 실제로 나와야 할 분포입니다. 우도를 최대화 시켰음에도 불구하고 실제 레이블을 따라 분포를 잡지 못한 모습입니다. 최대우도추정법으로도 클래스를 제대로 나누지 못한것을 확인할 수 있습니다. ~~하지만 저건 반박용 억지 데이터라는 생각이 드는건 어쩔 수 없나봅니다. 세상에 저런식으로 레이블 된 데이터가 있다면 그 데이터가 잘못된게 아닐까요..?~~


> Reference
> Zhu, Xiaojin. "Semi-supervised learning tutorial." International Conference on Machine Learning (ICML). 2007.
> Zhu, Xiaojin, and Andrew B. Goldberg. "Introduction to semi-supervised learning." Synthesis lectures on artificial intelligence and machine learning 3.1 (2009): 1-130.
