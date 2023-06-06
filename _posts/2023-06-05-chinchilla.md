---
title: Training Compute-Optimal Large Language Models
description: LLM들의 파라미터 수가 늘어나면서 학습에 대한 정량적인 분석이 나오게 됩니다. 엄청난 양의 실험을 진행하면서 기존 LLM 모델들이 과소적합 되어있다는 점을 증명한 논문에 대해 알아보겠습니다.
category: Deep Learning
tags:
- NLP
- LLM
---


요새 LLM이 많이 나오고는 있었는데 가져다 쓰는 것에만 익숙해지다 보니 많은 반성을 하게되었습니다. 대학원생 때만 하더라도 논문 열심히 읽고 정리하는게 나름 일상화 된 루틴 중 하나였는데 회사를 다니기 시작하면서 많이 나태해진 것 같다는 생각을 하며, 유명했던 논문 위주로 하나씩 포스팅을 작성해보려 합니다. 오늘 리뷰해볼 논문은 'chinchilla' 라는 이름으로도 유명한 'Training Compute-Optimal Large Language Models'입니다.

<div align="center">
<img src="https://i.imgur.com/pVIEfD3.png" title="chinchilla" width="500"/>
</div>

## Introduction

LLM들의 파라미터 수가 엄청나게 늘어나면서 컴퓨팅 능력에 대한 요구사항은 더더욱 늘어나게 되었습니다. 모델 파라미터 수 뿐만 아니라 학습 step 수 또한 엄청나게 늘어나게 되었습니다. 일단 기본적으로 domain specific task 들을 푸는게 아닌 범용 모델인 만큼 어마무시한 양의 token을 학습이 필요하게 된 셈이죠. 실제 클라우드로 GPU를 가동해보신 분은 아시겠지만 가격이 엄청나게 비쌉니다. 그렇다고 기업 입장에서 온프렘으로 GPU 수십장을 구비하는 것도 상당히 비용이 많이 드는 일입니다.

일단 자원이 제한되어 있다는 가정 하에 큰 모델을 사용하게 되면 학습을 많이 할 수 없고, 그렇다고 작은 모델을 사용하자니 토큰 수를 다 커버할 수 없습니다. 이런 제한된 자원에서 최적의 성능을 달성하기 위한 조합을 학습하는게 'compute-optimal training' 이라고 할 수 있습니다.(최적의 trade-off 지점을 찾기)

## Estimating the optimal parameter/training tokens allocation

위에서 언급한 '제한된 자원'은 쉽게 이야기 해서 FLOPS(Floating Point Operations per Second)라고 말할 수 있습니다. 딥러닝에서 가장 널리 사용되는 성능 측정 단위이며, 초당 수행되는 부동 소수점 연산의 양 입니다. 즉, FLOPS가 높을 수록 더 많은 연산을 할 수 있는 자원이 있다고 생각하면 될 것 같습니다. 수식으로 나타내면 아래와 같습니다. 

<div align="center">
<img src="https://i.imgur.com/k7O98qS.png" title="chinchilla" width="500"/>
</div>

- 𝐿(𝑁,𝐷): number of model parameters 𝑁, and the number of training tokens 𝐷.
- 𝐶 : computational budget 
- minimizing 𝐿 under the constraint FLOPs(𝑁,𝐷)=𝐶:
- 𝑁𝑜𝑝𝑡(𝐶), 𝐷𝑜𝑝𝑡(𝐶): optimal allocation of a computational budget 𝐶.

### Approaches

논문에서는 parameter ↔ token 사이의 trade-off 최적점을 찾기 위해 다음 3가지 접근 방식을 시도해봤다고 합니다. 논문에 매우 자세한 appendix까지 첨부되어 있습니다.

- Approach 1. Fix model sizes and vary number of training tokens (모델 사이즈를 고정하고 학습 토큰 수를 변경)
- Approach 2. IsoFLOP profiles (FLOPS를 고정하고 모델 사이즈를 변경)
- Approach 3. Fitting a parametric loss function (loss 함수 fitting)


### Parameter/Tokens Allocation Results

<div align="center">
<img src="https://i.imgur.com/Z38mf4L.png" title="chinchilla" width="700"/>
</div>

다양한 FLOPS에서 실험을 진행한 결과입니다(approach 2). 왼쪽에 있는 2차함수처럼 생긴 곡선들이 동일한 FLOPS에서 이루어진 여러가지 실험입니다. (같은 색 = 동일한 FLOPS)
곡선마다 최저점을 나타내는 부분이 최적화된 모델 크기와 스텝이라고 생각하시면 됩니다. 전체적으로 FLOPS 수가 많아질수록 최저점에서늬 Training loss가 줄어드는 것을 확인할 수 있습니다. 최저점을 지나 다시 오른다는 것(파라미터가 커지는데도 불구하고 loss가 커진다는 것)은 학습 step 수가 아직 부족하다는 뜻이라고 볼 수 있습니다. 진한 검은색이 될수록 loss가 전체적으로 작아지면서 더 성능이 좋은 모델이 나오는 모습도 볼 수 있습니다.

가운데와 우측에 있는 회귀선은, 왼쪽부터 각각 최적의 모델 파라미터 수 그리고 학습 step 수와 회귀선을 시각화 한 것입니다. 
둘 다 FLOPS가 증가할수록 최적의 파라미터 수와 학습 step 수는 증가하는 모습입니다. 사실 여기까지 써놓고 보면 그래서 파라미터 수와 step 수를 어떻게 조합하는데? 라는 의문이 들게 됩니다. 논문에서는 해당 조합식을 구하는 방법으로 'Chinchilla Scaling Law'를 제안하고 있습니다.

### Chinchilla Scaling Law

<div align="center">
<img src="https://i.imgur.com/krcon0p.png" title="chinchilla" width="700"/>
</div>


<div align="center">
<img src="https://i.imgur.com/S9ULsIn.png" title="chinchilla" width="700"/>
</div>

이게 최적의 모델 파라미터 수와 학습 수를 수식으로 나타낸 것입니다(approach 3). 왼쪽부터 차례대로 최적의 모델 파라미터 수, 최적의 학습 step 수 입니다. 실험 결과, closed form 형식으로 답을 구해보면 𝑎 = 0.46 그리고 𝑏 = 0.54 일 때 가장 최적의 값이라고 하네요. (아래 Table 2 참고)

<div align="center">
<img src="https://i.imgur.com/0MKIMIp.png" title="chinchilla" width="700"/>
</div>

논문에서는 Gopher 모델 학습 FLOPS와 동일한 자원으로 Chinchilla scailing law를 따르는 모델을 완전 공개하고 학습 방법까지 상세히 적어두었습니다. 동일한 데이터로 학습 시켰음에도, Chincilla 방식으로 학습시킨 모델이 크기는 1/4 정도로 줄어들었고 성능 또한 좋게 나옵니다. 이를 통해서 기존 LLM들은 상당히 과소적합 되어있으며 충분한 학습이 더 필요하다는 것을 증명하였습니다. 같은 자원을 가지고 더 적은 모델을 만들어 완전히 학습시킨 모델을 만들 수 있음을 보여준 셈입니다. 내용이 너무 많아 여기 다 옮기지는 못했지만 논문에 있는 수많은 첨부자료를 통해 다양한 task에서 기존 모델들보다 더 좋은 성능을 보이는 것도 증명하였습니다.

## 정리하며
처음엔 저자가 왜이렇게 많지? 싶었는데 논문 내용을 자세히 살펴볼수록 납득할 수 밖에 없었습니다. 이정도로 많은 실험을 진행했다는게 존경스러울 따름입니다. 논문을 써보신 분들은 아시겠지만 표 하나 완성하는데도 엄청난 시간이 걸리는데 정말 다방면으로 무지막지한 실험을 진행해서 결과를 뽑아 냈습니다. 

급격하게 발전하는 LLM 분야에서 '최적화'라는 관점에서 발전 방향을 잘 집어준 논문이었던 것 같습니다. 이름이 어째서 친칠라인지 봤더니 몸집에 비해서 먹는 양이 매우 많아서 그렇다고 하네요. 상대적으로 적은 파라미터 수로도 더 강력한 성능을 보여줄 수 있는 방법을 제안한다는 점에서 참 직관적인 이름을 잘 지은 것 같습니다.




> Reference
> * [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf)







