---
title: Recurrent World Models Facilitate Policy Evolution(World Models) - 2
description: 2018 Nips oral presentation으로 발표된 구글 브레인 논문입니다. 뇌인지과학 개념을 적용하여 강화학습 과제를 해결하였습니다. 적용한 방식도 굉장히 훌륭하지만 결과도 해당 task들의 SOTA를 찍어버린 놀라운 결과를 보여주고 있습니다. 실험 결과와 해석 그리고 논문에서 제안하는 'dream' 방법에 대한 포스트입니다.
category: Deep Learning
tags:
- RNN
- VAE
- CNN
- model-based RL
---

# World Models - Experiments

트랙 위에서 자동차를 운전하는 CarRacing-v0과 몬스터들이 쏘는 파이어볼을 피하는 VizDoom 총 두 가지 환경(environment)에서 실험을 진행하였습니다. 

## Car Racing Experiment

첫번째로는 CarRacing-v0데이터 입니다. 매 trial마다 랜덤하게 트랙이 생성되고, top-down 형식의 시점입니다. 정해진 시간 안에 자동차가 최대한 회색의 트랙을 오랫동안 밟을수록 높은 reward를 받습니다. Agent(자동차)의 action벡터는 좌/우회전, 가속 그리고 브레이크로 이루어진 3차원의 벡터입니다.

![Imgur](https://i.imgur.com/O5yXUw5.png)

먼저 V를 학습하기 위해 만 개의 environment를 수집하였습니다(만 개의 트랙이라 보시면 될 것 같습니다). 그리고 agent가 랜덤하게 탐색하며 여기서 관측되는 observation(프레임)을 사용하여 VAE를 학습시켰습니다. 아래 그림은 학습이 완료된 VAE로 다시 이미지를 재구성한 모습입니다. ([논문 저자의 블로그](https://worldmodels.github.io/)를 들어가시면 아래 슬라이드를 움직여 동적으로 어떻게 변화하는지 살펴볼 수 있습니다. 비단 아래 그림 뿐만 아니라 다른 그림들도 마찬가지입니다.)

![Imgur](https://i.imgur.com/3BUtJlr.png)

<div>
이후에는 V를 통해 구한 ${z}_{t}$와 랜덤 액션 벡터 ${a}_{t}$를 사용하여 MDN-RNN을 학습하였습니다. $P({ z }_{ t+1 }|{ a }_{ t },{ z }_{ t },{ h }_{ t })$ 이 수식을 보면 MDN-RNN을 학습시키는데 필요한 요건이 갖추어진 것을 확인할 수 있습니다.
</div>
V와 M만 가지고는 단순히 프레임을 압축하고 다음 프레임을 예측하는 것 밖에 못하기 때문에 실제 reward를 구할 수 없습니다. Reward에 직접적으로 영향을 끼치는 C는 파라미터 수가 867개로 매우 적기 때문에 CMA-ES를 사용하여 최적화 시킵니다. 전체적인 과정은 다음과 같습니다.

1. Collect 10,000 rollouts from a random policy
2. Train VAE(V) to encode each frame into a latent vector $z \in \mathcal{R}^{32}$
3. <div>Train MDN-RNN(M) to model $P({ z }_{ t+1 }|{ a }_{ t },{ z }_{ t },{ h }_{ t })$</div>
4. Evolve Controller(C) to maximize the expected cumulative reward of a rollout



### V Model Only

사실 observation의 정보를 잘 담아낼 수 있는 representation만 가지고 있다면 agent를 학습시키는 것은 어려운 일이 아닙니다. 이전 강화학습 연구들도 이러한 방식으로 학습을 진행한 경우가 많았습니다. 추가적인 실험을 위해 논문에서도 M을 사용하지 않고 V만 사용하여 학습을 진행합니다. 수식으로 나타내면 아래와 같습니다.

$${ a }_{ t }={ W }_{ c }{ z }_{ t } +{ b }_{ c }$$

<div align='center'>
<img src="https://i.imgur.com/JU5uftV.gif" title="V만 가지고 학습" />
</div>

학습을 잘 하기는 하지만, 엄청 비틀거리면서 운행하며 급코너 구간에서는 트랙을 벗어나는 모습을 볼 수 있습니다. 100번의 실험을 진행하였을 때 $632\pm 251$점을 얻었으며 이는 OpenAI Gym leaderboard의 다른 방법들과 비슷한 성능입니다. Single layer인 C에 추가적으로 hidden layer를 추가하여 파라미터 수를 증가시키면 $788\pm 141$점으로 향상되기는 하였지만 여전히 아쉬운 면이 있습니다.


### Full World Model (V and M)

이번에는 V와 M모두 사용한 결과물입니다. V를 사용하면 현재 어떤 상황인지는 알 수 있지만 V와 M 모두 사용하면 현재 상황 뿐만 아니라 미래 상황에 대한 정보까지 C에 입력할 수 있습니다. 

<div align='center'>
<img src="https://i.imgur.com/6nHdyc0.gif" title="V와 M 보두 사용하여 학습" />
</div>

확실히 비틀거리는 경우도 훨씬 덜하고 급커브도 부드럽게 돌면서 안정적인 운행을 하는 것을 확인할 수 있습니다. MDN-RNN의 장점, 즉 미래에 대한 확률 분포(${h}_{t}$) 덕분에 안정적인 action decision을 얻을 수 있습니다.

![Imgur](https://i.imgur.com/C4Ghok2.png)

Full model을 사용하였더니 $906\pm21$점이라는 매우 높은 SOTA급 득점 기록을 세울 수 있었다고 합니다. 해당 실험 환경에서 평균 906점으로 평균 900점을 넘겨서 과제를 완전히 풀었습니다! 기존 Deep RL 방법론들은 입력으로 들어오는 프레임들에 edge-detection등과 같은 전처리 기법과 여러 프레임을 하나로 쌓는 추가적인 과정이 필요합니다. 그러나 논문에서 제안하는 방법론은 직접적으로 RGB 픽셀 값들을 VAE에 집어넣어 효율적으로 ${z}_{t}$를 뽑아낼 수 있습니다. 

### Car Racing Dreams

해당 방법론의 MDN-RNN은 매 프레임마다 다음 시점 프레임의 latent vector ${z}_{t+1}$를 예측합니다. 그렇다면 새로 관측되는 프레임을 encode하지 않고 예측한 다음 시점의 latent vector를 지속적으로사용한다면 어떤 결과가 나올까요? 이를 'dream'이라는 표현을 사용하였습니다.

(홈페이지)

$\tau$값을 변화시키는 것에 따라 생성되는 dream environment도 바뀜을 확인할 수 있습니다. 값을 높이면 uncertainty가 커져 더 추상적이고 뭉개지는 결과가 나옵니다. 그렇다면 이렇게 dream environment만 가지고 agent를 학습한 뒤에 실제 environment로 policy를 적용하면 잘 할수 있을까? 하는 궁금증이 생기게 됩니다.
주어진 프레임 하나만 가지고 계속 다음 장면을 예측한다는 발상이 상당히 놀라웠습니다. 처음 읽었을 때는 이미지 트레이닝과 유사하다는 생각이 들었습니다. 'Dream'속에서 연습하다 실제 환경에 적용해 보면 과연 어떤 결과가 나올까요?



## VizDoom Experiment: Learning inside of a Dream

서론에서 실제 사람들이 세상을 바라볼 때 전부를 똑같이 인식하는것이 아닌 뇌 속의 'model'을 통해 추상화된 representation만 바라본다는 언급을 했습니다. 이 가정이 맞다면 방금 이야기 하였듯이 dream environment로 학습한 모델이 실제 세계에서도 잘 작동해야합니다.

![Imgur](https://i.imgur.com/uKhmk5T.png)

학습 데이터는 위와 같습니다. 하나의 rollout당 최대 60초(2100 time steps)까지 진행되는 환경이며 agent는 적이 발사하는 총알? fireball?을 좌 우로 움직이며 피해서 살아남아야 합니다. 20초(750 time steps)까지 살아남을 경우 생존하였다고 판단합니다(~~실제로 사람이 직접 해봐도 생각보다 어려운 게임입니다~~). 

### Procedure

<div>CarRacing 실험과 비슷하지만 몇 가지 다른 점이 있습니다. 이전 실험에서는 다음 latent vector ${z}_{t}$만 예측해도 된다면, VizDoom에서는 agent가 사망할 수도 있기 때문에 다음 프레임에서 agent의 생존 여부 ${d}_{t}$도 예측하게 하였습니다. </div>

학습은 전부 latent space environment에서만 진행됩니다. 실제 environment와 동일한 구조를 가지고 있기 때문에 학습 완료 후에는 inference가 가능합니다. 전체적인 과정은 다음과 같습니다.

1. Collect 10,000 rollouts from a random policy.
2. Train VAE (V) to encode each frame into a latent vector $ z \in \mathcal{R}^{64}$, and use $V$ to convert the images collected from (1) into the latent space representation.
3. <div>Train MDN-RNN (M) to model $P({ z }_{ t+1 },{d}_{t+1}|{ a }_{ t },{ z }_{ t },{ h }_{ t })$</div>
4. Evolve Controller (C) to maximize the expected survival time inside the virtual environment.
5. Use learned policy from (4) on actual Gym environment.

<div align='center'><img src="https://i.imgur.com/DZU4TNr.gif" title="학습이 완료된 agent" /></div>


### Training Inside of the Dream

M에서 예측한 latent vector를 V를 사용하여 decode한 결과는 아래와 같습니다. 가상으로 꿈 속에서 학습을 진행합니다.

<div align="center"><img src="https://i.imgur.com/fD6wLGP.png" title="원문 홈페이지에서 직접 조작할 수 있습니다" width="400" /></div>

RNN 모델은 게임 환경을 완전히 모사할 수 있게 학습이 됩니다. 무작위로 생성된 게임 환경의 이미지 데이터만 가지고 게임의 주요한 특징을 잘 잡아낼 수 있습니다. 예를 들어 agent가 왼쪽으로 움직이는 action을 선택한다면 M이 예측하는 dream environment 또한 agent를 왼쪽으로 움직이고 내부 환경을 그에 맞춰 변경시킵니다. 만약 fireball이 날아오는 상황이라면 해당 환경 또한 일관된 방향으로 날아오게끔 표현합니다. 즉, 미래를 일관성있게 잘 예측합니다. $\tau$값을 바꾸면 불확실성을 많이 추가하여 실제 게임 환경보다 훨씬 어렵게 만들 수도 있습니다. 실험 결과 적당히 높은 uncertainty가 dream environment의 불완전한 부분에서 agent가 쉽게 학습하는것을 방지하여 더 좋은 결과가 나왔다고 합니다.

<div align='center'><img src="https://i.imgur.com/U3Zs6fC.gif" title="<좌 : 실제 environment frame, 우 : decoder로 reconstruct된 frame" /></div>

위 gif파일을 보시면 encode 되는 프레임과는 다르게 다시 reconstruct되는 프레임은 약간 부정확한 것을 확인할 수 있습니다. 특히 몬스터 숫자들이 부정확하게 표현되는데, agent의 생존에 중요한 벽이나 파이어볼은 제대로 잡아내고 있습니다.

## Cheating the World Model

<div align='center'><img src="https://i.imgur.com/pQxYTfE.png" title="스트리트 파이터에서 와리가리(?) 중인 모습" /></div>

오락실 좀 가본 분들이라면 위 사진이 무엇인지 바로 아시겠지만, 게임 디자이너가 의도하지 않은 일명 얍삽이(?)입니다. 논문에서도 실험 초기에는 이러한 'generative policy' 현상이 발견되었다고 합니다. 특정 위치에서 약간씩 왔다갔다 하는 경우 파이어볼이 생성되지 않는 경우도 있고, agent가 특정 방향으로 이동할 경우 파이어볼이 사라져 버리는 경우도 있었습니다.

<div align='center'><img src="https://i.imgur.com/q0OiKWJ.gif" title="특정 위치에서 비비고 있으면(?) 파이어볼이 날아오지 않습니다." /></div>

이는 M이 사실은 environment의 probabilistic model을 단지 '근사'하기 때문이라고 합니다. 실제 게임 화면에서는 있어야 할 반대쪽 몬스터들이, dream에서는 없어져버리기도 합니다. 모든 정보를 완벽히 가져갈 수 없는 한계라고도 할 수 있습니다.

또한 C의 수식을 보면 M의 hidden state ${h}$ 전부에 접근할 수 있으므로, agent가 게임 엔진의 모든 환경을 이해하게 됩니다. 이러한 상황에서는 C가 adversarial policy를 쉽게 찾아낼 수 있습니다. 더 쉽게 얘기하자면, 실제 학습 데이터의 분포(training distribution)에서 멀어져 버리게 됩니다. 이러한 이유 때문에 실제 관측 데이터가 들어오게 되면 agent가 제대로 행동하지 못하는 결과를 보여줍니다.

논문에서는 MDN-RNN이 일반적인 RNN보다 adversarial policy에 강점을 가진다고 말하고 있습니다. 사실 VAE의 latent space는 단일 가우시안 분포를 따릅니다. 따라서 MDN이 조금 과하다고 생각할 수는 있지만 MDN은 environments가 random discrete events(파이어볼을 쏠 것인가 말 것인가)일 경우 효과적입니다. 프레임 한 장을 encode할 때는 하나의 가우시안 분포로도 충분하지만 MDN-RNN은 random discrete한 복잡한 미래를 예측할 때 효과적이라고 말할 수 있습니다.

$\tau=0.1$일 때는 거의 deterministic LSTM과 다르지 않은 결과물이 나옵니다. Agent가 무슨 행동을 하던지 mode collapse에 빠져 파이어볼이 날아오지 않습니다. $\tau$값을 증가시키면 조금 나아지기는 하지만 때때로 갑자기 파이어볼이 없어지는 경우가 있습니다. $\tau$값에 따른 실험 결과는 아래 표와 같습니다.

![Imgur](https://i.imgur.com/kzZDhFe.png)

$\tau$값이 너무 높아지만 오히려 agent가 너무 빨리 죽어버려서 학습을 제대로 하지 못하는 상황이 발생하는 것을 확인할 수 있습니다.

## Iterative Training Procedure

논문에서 진행한 실험들은 상대적으로 간단하고 쉬운 task이기 때문에 random policy를 통해 나온 데이터로 괜찮은 model을 생성할 수 있었습니다. 만약 더 정교한 task의 경우에는 어떤 방식으로 접근하면 될까요? 이런 경우에는 다음과 같은 iterative training procedure이 필요하게 됩니다.

1. Initialize M, C with random model parameters.
2. Rollout to actual environment NN times. Agent may learn during rollouts. Save all actions ${a}_{t}$ and observations ${x}_{t}$ during rollouts to storage device.
3. <div>Train $M$ to model $P\left( { x }_{ t+1 },{ r }_{ t+1 },{ a }_{ t+1 },{ d }_{ t+1 }|{ x }_{ t },{ a }_{ t },{ h }_{ t } \right) $ and train $C$ to optimize expected rewards inside of $M$</div>
4. Go back to (2) if task has not been completed.

이미 이전 두 번의 실험에서 한 번의 loop만으로 간단한 task를 해결하는데 충분하는 것을 확인하였습니다. 복잡한 task에서는 2번에서 4번까지 반복하는 것으로 world model을 개선할 수 있습니다. <div>달라진 점이 있다면 ${r}_{t+1}$과 ${a}_{t+1}$까지 맞추는 방향으로 M을 학습합니다.</div> 다음 시점의 reward, action까지 model 학습에 쓰이는 이유는 사실 단순하게 생각할 수 있습니다. Model이 완전히 skill을 체득하게 되면, controller가 더 어려운 skill을 습득하는데 집중할 수 있기 때문입니다.


논문에서 제시한 두 번의 실험을 예시로 들었을 때, M은 다음 프레임에 대한 확률 분포를 알려줍니다. 만약 제대로 하지 못한다면 전혀 익숙하지 않은 프레임을 테스트 과정에서 겪고 있다는 뜻입니다. 이럴 경우 새로운 데이터를 더 모아서 model 성능을 향상시킬 수 있습니다. 

쉽게 아래 그림에서 슈퍼마리오를 예로 들어보겠습니다. 둘 다 같은 게임이지만 왼쪽 그림의 스테이지를 가지고 학습한 마리오 agent는 땅에서만 제대로 동작합니다. 오른쪽 그림처럼 수중 스테이지는 추가적인 데이터가 필요하게 됩니다.

![Imgur](https://i.imgur.com/lsUP3Qu.png)

아래 그림은 iterative training procedure을 축약해서 보여준 것과 같습니다. Model이 skill을 습득하게 되면 long-term memory에 들어가게 됩니다. 피아노를 처음 배울 때는 악보와 손가락을 번갈아가면서 보게 되지만 익숙해지면 손가락을 쳐다보지 않고 바로 건반을 칠 수 있는 것과 같습니다. 

![Imgur](https://i.imgur.com/dkvSJX7.png)


## 마치며 

오랜만에 정말 재미있게 읽은 논문입니다. 구글 브레인에서 나오는 논문들이 인지과학과는 연관이 있는 경우가 많은데, 이번 논문도 특히 그렇습니다. 뭔가 진짜 사람처럼 생각하는 AI를 만들고자 하는 커다란 목표가 느껴지는 연구였습니다. 제가 올린 포스트들은 결국 원 저자 블로그 요약 + 제 생각을 정리해둔 것이니 저자 블로그도 꼭 다시 읽어보시길 권해드립니다.

> Reference
> * [Ha, D., & Schmidhuber, J. (2018). World Models. arXiv preprint arXiv:1803.10122.](https://arxiv.org/abs/1803.10122)
> * [worldmodel blog](https://worldmodels.github.io/) 