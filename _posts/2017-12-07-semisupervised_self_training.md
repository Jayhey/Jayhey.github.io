---
title: 셀프 트레이닝(self training) 
description: 준지도학습(semi-supervised learning) 방법중에서 가장 간단한 셀프 트레이닝(self training)에 대하여 알아보도록 하겠습니다. 높은 확률값이 나오면 가중치를 주는 간단한 방식으로 손쉽게 모델의 성능 향상을 꾀할 수 있는 테크닉입니다.
category: Semi-supervised Learning
tags: 
- machine learning
- semi supervised learning
---




이 글은 고려대학교 강필성 교수님의 Business Analytics 강의를 정리했음을 밝힙니다.

# 셀프 트레이닝(Self training)

셀프 트레이닝의 원리는 매우 간단합니다. 바로 **높은 확률값이 나오는 데이터** 위주로 가져가겠다는 겁니다. 예를들어 로지스틱 회귀분석(logistic regression) 결과 한 데이터에 대한 1일 확률값이 0.95가 나온 경우가 있고 0.55가 나온 경우가 있다면, 0.95로 예측한 데이터를 가져간다는 뜻입니다.

## Procedure

간단한 셀프 트레이닝의 알고리즘은 다음과 같습니다.

- 레이블이 달린 데이터로 모델을 학습시킵니다.
- 이 모델을 가지고 레이블이 달리지 않은 데이터를 예측합니다.
- 이중에서 가장 확률값이 높은 데이터들만 레이블 데이터로 다시 가져갑니다.
- 위 과정을 계속 반복합니다.

이런 방식으로 알고리즘을 반복하면 점점 더 모델이 정확해질 수 있다는게 셀프 트레이닝의 핵심 개념입니다. 과정을 그림으로 나타내면 다음과 같습니다.

<div align = "center"><a href="https://imgur.com/puPDnOX"><img src="https://i.imgur.com/puPDnOX.png" /></a></div>

## Image Categorization

<div align="center"><a href="https://imgur.com/uchE1Cq"><img src="https://i.imgur.com/uchE1Cq.png"  /></a></div>

이미지 분류 예시를 들어보겠습니다. 먼저 eclipse를 검색하면 나오는 두 개의 이미지가 있습니다. 하나는 일식 현상을 보여주는 그림이고 하나는 자동차 종류 eclipse입니다. 이 두 개의 이미지를 나이브 베이즈 분류기로 학습시킵니다. 그리고나서 레이블이 달려있지 않은 데이터들을 $logp(y=Astronomy|x)$값을 기준으로 정렬합니다. 

<div align="center"><a href="https://imgur.com/Pf8RP7V"><img src="https://i.imgur.com/Pf8RP7V.png"/></a></div>

여기서 가장 신뢰성 있는(확률값이 높게 나온) 이미지들을 정분류된 레이블 데이터라고 생각하고 다시 모델을 학습시키면 됩니다. 이런 방식을 계속 반복하면 점점 더 모델이 정확해지게 됩니다.

## Propagating 1-Nearest Neighbor

![Imgur](https://i.imgur.com/a5Ofvq2.png)

k-nearest neighbor 모델에서 k에 1을 넣고도 셀프 트레이닝이 가능합니다. 먼저 레이블 데이터에서 **가장 가까운 1개의** 언레이블 데이터를 집습니다. 그리고 이 언레이블 데이터를 레이블 데이터와 같은 클래스로 설정을 합니다. 이 작업을 언레이블 데이터가 없어질 때까지 계속 반복하면 됩니다. 아래 그림에서 초록색이 언레이블 데이터, 빨간색과 파란색이 각 클래스별 레이블 데이터입니다. 이터레이션을 거칠수록 점점 제대로 분류를 하는 것을 볼 수 있습니다.

<div align="center"><a href="https://imgur.com/nWdYT5h"><img src="https://i.imgur.com/nWdYT5h.png" /></a></div>

하지만 이 방법에도 단점이 있으니, 만약 밑에 그림처럼 잘못된 부분에 아웃라이어가 한 개 포함되어 있어도 잘못된 결과가 나올 수 있습니다. (사실 아웃라이어가 아니고 노이즈 느낌이라고 수업시간에 교수님이 말씀하셨지만...)

<div align="center"><a href="https://imgur.com/9PzsDpU"><img src="https://i.imgur.com/9PzsDpU.png"/></a></div>

## Advantages & Disadvantages

정리하자면 셀프 트레이닝의 장점은 다음과 같습니다.
- 가장 간단한 준지도학습(semi-supervised learning)이다.
- 어떤 알고리즘이라고 적용 가능하다(wrapper method).
- NLP 같은 분야에서 종종 쓰인다.

단점은 다음과 같습니다.
- 초반에 잘못 저지른 실수(early mistakes)가 잘못된 길로 인도할 수 있다.
- 완전히 수렴(convergence)한다고 딱 말할 수 없다(Cannot say too much in terms of convergence)

다음 포스트에서는 generative model에 대해서 알아보도록 하겠습니다. 

> Reference
>* Zhu, Xiaojin. "Semi-supervised learning tutorial." International Conference on Machine Learning (ICML). 2007.
>* Zhu, Xiaojin, and Andrew B. Goldberg. "Introduction to semi-supervised learning." Synthesis lectures on artificial intelligence and machine learning 3.1 (2009): 1-130.
