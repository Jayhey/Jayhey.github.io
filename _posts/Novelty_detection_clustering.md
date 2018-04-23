---
title: 군집화 기반 이상치 탐지(clustering based Novelty Detection)
description: 거리 기반 이상치 탐지 기법 중 가장 기본적은 군집화 기반 접근법에 대하여 알아보겠습니다.
category: Novelty Detection
tags:
- distance based novelty detection
---

시작하기 전에 앞서 이 글은 고려대학교 강필성 교수님의 Business Analytics 강의를 정리했음을 밝힙니다.



# Clustering based Approach

## K-means clustering

군집화 기반 이상치 탐지는 매우 간단합니다. 각 데이터에서 가장 가까운 군집의 중심까지 거리를 novelty score로 계산합니다.  군집화는 k-means clustering 알고리즘을 기준으로 수식은 아래와 같습니다.  여기서 k는 군집의 갯수로, 사용자가 정해주는 하이퍼 파라미터입니다.



$$ X={ C }_{ 1 }{ \cup C }_{ 2 }{ ...\cup C }_{ K },\quad { C }_{ i }\cap { C }_{ j }=\phi $$
$$ arg\min _{ c }{ \sum _{ i=1 }^{ K }{ \sum _{ { x }_{ j }\in { C }_{ i } }^{  }{ { \left\| { x }_{ j }-{ c }_{ i } \right\|  }^{ 2 }}}} $$




> Reference
>
> > 