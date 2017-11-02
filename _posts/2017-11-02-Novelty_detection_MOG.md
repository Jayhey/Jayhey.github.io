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



ㅇㅇ