---
title: 로컬 아웃라이어 팩터(Local Outlier Factors)
description: 다른 밀도 추정법들이 데이터가 특정 분포를 따른다는 가정 하에 추정하는 방법이었습니다. 하지만 이번에 설명할 커널 밀도 추정은 데이터가 특정 분포를 따르지 않는다는 가정 하에 밀도를 추정하는 방법입니다. 커널 밀도 추정의 기본적인 개념을 알아보고 대표적인 그 중 파젠 윈도우 밀도 추정(Parzen window density estimation)에 대해 알아보겠습니다.
category: Novelty Detection
tags:
- density based novelty detection
- non-parametric
---


시작하기 전에 앞서 이 글은 고려대학교 강필성 교수님의 Business Analytics 강의를 정리했음을 밝힙니다.


# Local Outlier Factors(LOF)

<div align="center"><a href="https://imgur.com/u5Jy8zs"><img src="https://i.imgur.com/u5Jy8zs.png" width="500" /></a></div>


위 그림에서 O1은 이상치라는걸 바로 알 수 있지만 O2는 이상치라고 말할 수 있을까요? 얼핏 보기엔 아닌 것 같지만 근처에 빡빡하게 밀집해 있는 데이터가 있으므로 이상치라고 볼 수도 있을 것입니다. LOF(Local outlier factors)관점에서는 O2도 이상치입니다. 밀집 지역에서 밀도 관점으로 봤을 때 급격한 감소가 이루어지기 때문입니다. 즉, LOF는 데이터가 가지는 상대적인 밀도까지 고려한 이상치 탐지 기법입니다.

## 수식 살펴보기

### k-distance

그래서 LOF에서는 $k-distance$라는 개념을 사용합니다. $k-distance(A)$는 $A$로부터 k번째 근접 이웃까지의 거리를 뜻합니다. 그리고 $k-distance$안에 들어오는 오브젝트의 집합을 ${N}_{k}(A)$라고 정의합니다. 쉽게 얘기하면 $k-distance$보다 작거나 같은 거리를 가지는 수 입니다. 

![Imgur](https://i.imgur.com/SRJyXZv.png)

표로 정리하면 위와 같습니다. 표 안의 숫자는 현재 오브젝트에서 다른 오브젝트 까지의 거리입니다.

### reachability distance

$${ reachability-distance }_{ k }(A,B)=max\left\{ k-distance(B),dist(A,B) \right\} $$

다음은 reachability distance입니다. 막상 수식을 보면 복잡하게 보이지만 간단히 말해서 A와 B까지의 거리 그리고 k-distance중 큰 값을 사용하면 됩니다. k-distance 안에 들어오는 object들은 전부 원 위로 밀어내고 원 밖은 그대로 거리 값을 사용하는게 reachability distance입니다. 만약 항상 k-distance를 사용한다면, LOF가 아닌 Simplified-LOF가 됩니다.

![Imgur](https://i.imgur.com/9HaaED5.png)

위 그림에서도 A를 기준으로 봤을 때, B와 C까지의 reachability-distance는 원 밖으로 밀어내서 3-distance(A)와 같아집니다. 그리고 D의 경우는 원 밖에 있으니까 그대로 거리값을 사용하게 됩니다.

### local reachability density

오브젝트 A에 대한 local reachability density는 다음과 같이 구할 수 있습니다.

$$lrd_{ k }(A)=\frac { |{ N }_{ k }(p)| }{ \sum _{ O\in { N }_{ k }(A) } reachability-distance_{ k }(A,B) } $$

분자는 k-distance 안의 개체 수이고 분모는 A에서 다른 오브젝트들까지의 reachability-distance의 입니다. A로부터 다른 오브젝트들 까지의 reachability distance들의 평균 식을 거꾸로 뒤집은 것과 같습니다.
- Case 1 : 만약 A가 밀도가 높은(dense area) 지역에 위치한다면 분모가 작아질 것이고 이에 따라 ${lrd}_{k}(A)$값이 커지게 됩니다.
- Case 2 : 반대로 A가 밀도가 높지 않은(sparse area) 지역에 위치한다면 분모가 커지게 되고 이에 따라 ${lrd}_{k}(A)$값도 작아지게 됩니다.

![Imgur](https://i.imgur.com/6rJQgYl.png)

### local outlier factor

$${ LOF }_{ k }(A)=\frac { \sum _{ B\in { N }_{ k }(A) }^{  }{ \frac { lrd_{ k }(B) }{ lrd_{ k }(A) }  }  }{ |{ N }_{ k }(A)| } =\frac { \frac { 1 }{ lrd_{ k }(A) } \sum _{ B\in { N }_{ k }(A) }^{  }{ lrd_{ k }(B) }  }{ |{ N }_{ k }(A)| }  $$


위 식에서 분자를 보면 local reachability density의 평균을 local reachability density로 나눴다는것을 확인할 수 있습니다. 여기서 산출된 LOF score는 결국 A가 얼마나 이상치인가를 나타내는 정도라고 볼 수 있습니다. 

쉽게 이해하기 위해서 아래 그림을 예시로 들어보겠습니다. 파란색 점이 A이고 초록색 점이 B입니다. ${ LOF }(A)$값이 크다는 것은, 초록색 점들의 lrd(${lrd}(B)$)가 높고 파란색 점의 lrd(${lrd}(A)$)가 낮다는 말과 같습니다. 즉, 초록색 점들이 밀도가 높은 지역에, 파란색 점은 밀도가 낮은 지역에 위치한다는 말입니다.

정리해보자면 파란색 점이 밀도가 낮은 지역에 있을수록, 초록색 점들이 밀도가 높은 지역에 위치할수록 파란색 점의 LOF값은 커지게 됩니다. 쉬운 이해를 위해 표도 같이 첨부하겠습니다.

![Imgur](https://i.imgur.com/yrAXN5B.png)

![Imgur](https://i.imgur.com/gUGFvhB.png)

# Conclusion

LOF의 장점으로는, 굉장히 밀집된 클러스터에서 조금만 떨어져 있어도 이상치로 탐지해준다는 점입니다. 아래 그림에서 숫자들이 LOF 스코어를 나타내고 있습니다. 굉장히 뺵빽한 곳에 가까운 이상치들은 확실히 더 높은 LOF값을 가지는 것을 확인할 수 있습니다. 

<div align="center"><a href="https://imgur.com/3i0Wl0x"><img src="https://i.imgur.com/3i0Wl0x.png" width="500" /></a></div>

단점이라면 이상치라고 판단하는 기준을 어디에 잡아야할지 정해줘야 한다는 것입니다. 위 그림의 경우 2차원 데이터라서 쉽게 시각적인 도움을 받을 수 있지만 차원이 늘어나면 판단하기 골치아프게 됩니다. 어떤 데이터셋에서는 1.1이라는 값이 이상치이지만 어떤 데이터셋에서는 2라는 값을 가지고 있어도 정상 데이터일 수 있습니다. 