---
title: Semi-supervised Learning(준지도학습) - Overview
description: 머신 러닝(machine learning) 방법론들 중 교사 학습(supervised learning) 모델들은 정답 데이터가 꼭 있어야만 학습이 가능하다는 단점이 있습니다. 이러한 단점을 극복하기 위해 준지도 학습(semi-supervised learning)에 대해 살펴보도록 하겠습니다.
category: Semi-supervised Learning
tags: 
- overview
- machine learning
- semi supervised learning
---

이 카테고리에서 앞으로 다룰 준지도학습(semi-supervised learning)기법들에 대해 자세히 알아보기 전 과연 준지도 학습이란 무엇인가에 대해 간단히 살펴보도록 하겠습니다.

이 글은 고려대학교 강필성 교수님의 Business Analytics 강의를 정리했음을 밝힙니다.

# Semi-supervised Learning(준지도 학습)?

## Machine Learning Categories

통계학과나 산업공학과 등 전공 과목으로 데이터 마이닝(data mining)을 수강하신 분들은 아시겠지만 지도학습(supervised learning)과 비지도학습(unsupervised learning)에 대해서 가장 먼저 배우게 됩니다.  지도학습은 관측치 하나마다 정답 레이블이 달려 있는 데이터셋을 가지고 모델을 학습시킵니다. 대표적인 모델로는 다중회귀분석, 로지스틱 회귀분석, 인공신경망 등등이 있습니다. 반면 비지도학습은 정답 레이블이 달려 있지 않은 데이터를 대상으로도 사용할 수 있으며, 모델 스스로 학습한다는 특징을 가지고 있습니다. 클러스터링이 비지도학습 방법론 중에서는 가장 유명합니다. 

지도학습과 비지도학습의 특징은 [Novelty Detection(이상치 탐지) - Overview 포스트](https://jayhey.github.io/novelty%20detection/2017/10/18/Novelty_detection_overview/)에서 이미 정리해두었으니 읽어보시면 될 것 같습니다.



![Imgur](https://i.imgur.com/Ygm4iOs.png)

## Backgrounds



지도학습 기반의 모델들을 학습시키기 위해서는 정답 레이블이 달려있는 데이터셋이 필요합니다. 사실 말이 쉽지만 대량의 데이터  모두 정답 레이블이 달려있기가 쉽지 않습니다. 이걸 사람이 눈으로 보고 하나씩 달자니 또 너무 많습니다. 그리고 인건비 또한 무시 못할정도로 많이 들기 마련입니다. 단순 반복작업에 인건비도 많이들지만, 영상 데이터같이 특정 도메인의 경우는 전문가만 이런 일을 할 수가 있습니다. 의사들이 아니면 MRI 사진을 보고 암인지 아닌지 판단할 수는 없으니까요. 그리고 매우 비싼 기계가 있어야만 레이블을 달 수 있는 경우도 있습니다. ~~마지막으로 대학원생들이 휴가중일 수도 있습니다…~~아래 그림을 보시면 중국어를 포스태깅해서 트리를 만든 데이터입니다. 4천 문장에다 레이블을 다는데 **무려 2년**이 걸렸습니다. 그런데도 데이터가 많다고 할 수 없는 4000개밖에 만들지 못했습니다. 

<div align ="center"><a href="https://imgur.com/ZLCEVm6"><img src="https://i.imgur.com/ZLCEVm6.png" width="400px" /></a></div>

또 다른 예시를 들어보도록 하겠습니다. "eclipse"라는 키워드로 구글 검색을 하면 우리가 흔히 생각하는 일식 현상을 보여주는 사진이 나옵니다. 하지만 eclipse라는 차종도 같이 검색되어버립니다. 우리가 진짜 원하는 사진이 만약 일식 현상 "ecllipse"라면 사진을 하나하나씩 다 보고 이게 일식 현상인지 자동차인지 체크를 해줘야하는 일이 생깁니다.



![Imgur](https://i.imgur.com/RMzq5Lt.png)



이렇게 많은 데이터를 눈으로 직접 찾으려면 너무 많은 시간과 돈이 필요하기 마련입니다. 어떻게 지도학습 성능을 올리는데 위와 같이 레이블이 달려있지 않은 데이터를 활용할 수 있을까? 그래서 준지도학습(semi-supervised learning)이 등장하게 됩니다.



## Purpose

준지도학습의 목표는 간단합니다. 레이블이 달려있는 데이터와 레이블이 달려있지 않은 데이터를 동시에 사용해서 더 좋은 모델을 만들자는겁니다.  왼쪽 클러스터링 데이터를 보시면 (b)에 있는 작은 초록색 점들이 unlabeled data입니다. 레이블 된 데이터만 가지고 클러스터링을 시켜도 초록색 점들이 알아서 잘 나뉘는 것을 볼 수 있습니다. 오른쪽 그림은 서포트 벡터 머신(Support Vector Machine)을 학습시키는데 10개의 레이블 데이터와 1400개의 언레이블 데이터를 사용했을 경우 에러 차이를 나타냅니다. 무려 140배의 데이터를 사용했는데도 성능이 좋아진 것을 확인할 수 있습니다. 그런데 네번째 막대를 보시면 큰 차이가 없는것이 보입니다. 준지도학습은 앙상블 기법들과는 다르게 성능을 보장(guarantee)할 수는 없습니다. 앙상블처럼 수식으로 여러 개의 모델이 더 좋다고 증명할 수 없다는 말입니다. 그러나 요즘은 컴퓨팅 파워도 좋은데 시도해보지 않을 이유는 없겠죠?

![Imgur](https://i.imgur.com/QUIux6S.png)

### Why would unlabeled data be useful at all?

그러면 언제 언레이블 데이터들이 학습에 도움을 줄 수 있을까요? 언레이블 데이터들의 분포가 만약 균등하다면 지도학습에 전혀 도움이 되지 않을 수 있습니다.  왼쪽 그림에서 빨간 점들의 분포는(물론 극단적인 경우지만) 완전히 균등합니다. 저런 분포를 가지고 있는 언레이블 데이터를 아무리 더해줘도 기존 모델의 성능을 향상시킬 수는 없습니다. 반대로 오른쪽 그림처럼 군집 형태라면 학습에 도움이 될 수 있습니다.

![Imgur](https://i.imgur.com/mAMdPAC.png)



사실 현실 세계의 데이터들은 많이들 클러스터 가정을 만족하는 것으로 보입니다. 그래서 최소한 준지도학습을 사용하면 손해볼 일은 없다고 할 수 있겠습니다. **"Always"는** 아니더라도**"almost"**정도로 생각하시면 될 것 같습니다.** 



### Notations

준지도 학습에서 쓰이는 notation들을 정리하면 다음과 같습니다.

- Input instance $x$, label $y$
- Learner $f:X\rightarrow \Upsilon $
- Labeled data $({ X }_{ l },{ y }_{ l })=\{ ({ x }_{ 1:l },{ y }_{ 1:l })\} $
- Unlabeled data ${ X }_{ u }=\{ ({ X }_{ l+1:n })\} $, available during training
- Usually $l <<N $
- Test data ${ X }_{ test }=\{ ({ x }_{ n+1: })\} $, not available during training



다음 포스트부터는 본격적으로 준지도학습 기법들에 대해 알아보도록 하겠습니다

> Reference
> Zhu, Xiaojin. "Semi-supervised learning tutorial." International Conference on Machine Learning (ICML). 2007.
