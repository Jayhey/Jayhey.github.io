---
title: k-최근접이웃 기반 이상치 탐지(k-NN based Novelty Detection)
description: 이전까지는 밀도 기반 이상치 탐지 기법들에 알아보았다면 이번에는 거리 기반 이상치 탐지 기법 중 가장 기본적은 k-근접이웃 기반 이상치 탐지에 대하여 알아보겠습니다.
category: Novelty Detection
tags:
- distance based novelty detection
- non-parametric
---



시작하기 전에 앞서 이 글은 고려대학교 강필성 교수님의 Business Analytics 강의를 정리했음을 밝힙니다.



# K-NN based Approach

k-최근접 이웃 기반 접근법은 단순합니다. 바로 각 데이터에 대한 novelty score를 k개의 근접 이웃까지 거리를 이용하여 계산하는 것입니다.  만약 이웃까지의 거리를 계산해봤더니 값이 다른 데이터들에 비해 크다면, 이는 이상치일 확률이 높다고 할 수 있습니다. 아래 그림에서 파란색 데이터는 일반 데이터들에 비해 k개의 이웃까지의 거리가 매우 멉니다. 



![Imgur](https://i.imgur.com/LN8cn0e.png)



## Distance information

그렇다면 거리를 구해서 단순히 평균을 낼 것인지, 최댓값으로 해줄 것인지 정해줘야합니다. Novelty score를 정해주는데 다음과 같은 방식이 있습니다.

- Maximum distance to the k-th nearest neighbor (주변에서 가장 먼 이웃과의 거리)

$${ d }_{ max }^{ k }=\kappa (x)=\left\| x-{ z }_{ k }(x) \right\| $$

- Average distance to the k-nearest neighbors (거리의 평균)

$${ d }_{ avg }^{ k }=\gamma (x)=\frac { 1 }{ k } \sum _{ j=1 }^{ k }{ \left\| x-{ z }_{ j }(x) \right\|  } $$

- Distance to the mean of the k-nearest neighbors (이웃들 간 centroid를 구해서 거리 구함)

$${ d }_{ mean }^{ k }=\delta (x)=\left\| x-\frac { 1 }{ k } \sum _{ j=1 }^{ k }{ { z }_{ j }(x) }  \right\| $$



그림으로 표현하면 아래와 같습니다.



![Imgur](https://i.imgur.com/j0c4cXl.png)

Mean 값으로 구할 경우, 세 번째 그림처럼 경계선 위에 있는 (데이터들 사이에 있지 않은) 데이터들이 novelty로 포착 될 확률이 높게 나옵니다. 



## Counter example

아래 그림은 동그라미(circle)과 삼각형(triangle)에 대한 novelty score를 구한 예시입니다. 동그라미가 일반적으로 생각하는 novelty이고 삼각형이 일반 데이터입니다. 

![Imgur](https://i.imgur.com/5lKPfP2.png)

![Imgur](https://i.imgur.com/dU4KdNC.png)

k-NN 기법을 사용하여 novelty score를 계산해본 결과  왼쪽 그림 A의 경우 average distance로만 제대로 잡는 것을 확인할 수 있습니다. 오른쪽 B의 경우엔 어떤 distance 정보로도 novelty를 잡아내지 못하고 있습니다. 



## Consider additional factor

따라서 기존 k-NN 계산 방법에 한 가지 항을 더 추가해줍니다. 바로 이웃들의 convex hull 까지 거리도 고려해주는 것입니다. 

![Imgur](https://i.imgur.com/54LzHep.png)

Convex hull까지의 거리는 위 그림을 보면 이해하기 쉽습니다. 이웃들끼리 연결했을 때 그 안에 있으면 거리가 0, 그 밖에 있으면 거리가 0 이상이 됩니다. 즉, 이웃들과의 convex combination과의 거리를 계산하겠다는 뜻입니다. 수식으로 나타내면 아래와 같습니다. ${d}^{k}_{c\_hull}$가 convex combination까지의 거리를 뜻합니다.



$$\min _{ w }{ ({ d }^{ k }_{ { c\_ hull } }(x))^{ 2 } } ={ \left\| { x }_{ new }-\sum _{ j=1 }^{ k }{ { w }_{ i }{ z }_{ j }(x) }  \right\|  }^{ 2 }$$

$$s.t.\sum _{ i=1 }^{ k }{ { w }_{ i } } =1,{ w }_{ i }\ge 0,\forall i $$



이제 ${d}^{k}_{avg}$와 위에서 구한 ${d}^{k}_{c\_hull}$와 를 정리하여 조합하면 아래와 같습니다.

${ d }^{ k }_{ avg }=\frac { 1 }{ k } \sum _{ j=1 }^{ k }{ \left\| x-{ z }_{ j }(x) \right\|  } $

$${ d }^{ k }_{ c\_ hull }={ \left\| { x }-\sum _{ j=1 }^{ k }{ { w }_{ i }{ z }_{ j }(x) }  \right\|  }$$

$${ d }_{ k }^{ hybrid }={ d }_{ k }^{ avg }\times \left( \frac { 2 }{ 1+exp(-{ d }_{ c\_ hull }^{ k }) }  \right) $$

위 식을 잘 살펴보면 알겠지만, convex hull distance에 비례하게 0과 2 사이의 값을 ${ d }_{ k }^{ avg }$에 곱해줍니다. 즉, convex 안에 있으면 penalty를 주지 않고 최대 2배 까지 패널티를 주겠다느 말과 같습니다. 최종적으로 구한 값들은 아래와 같습니다.

![Imgur](https://i.imgur.com/5lKPfP2.png)

![Imgur](https://i.imgur.com/h7FB1fd.png)



${ d }_{ k }^{ hybrid }$를 사용하여 novelty score를 구하면 제대로 삼각형 데이터를 novelty로 잡는 것을 확인할 수 있습니다.



![Imgur](https://i.imgur.com/gsVzVPa.png)



위 그림은 여러가지 distance information을 사용했을 때 novelty boundary를 보여주고 있습니다. (d)를 보면 average distance를 사용했을때도 가운데에 구멍이 조그맣게 생긴 것을 확인할 수 있습니다. (e) 또한 너저분한 boundary가 생성됩니다. 하지만 convex hull distance까지 같이 활용한 (f)는 굉장히 깔끔한 boundary를 보여줍니다. 







> Reference
>
> > Kang, Pilsung, and Sungzoon Cho. "A hybrid novelty score and its use in keystroke dynamics-based user authentication." *Pattern recognition* 42.11 (2009): 3115-3127.