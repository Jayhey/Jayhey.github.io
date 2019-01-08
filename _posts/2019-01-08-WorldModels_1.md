---
title: Recurrent World Models Facilitate Policy Evolution(World Models) - 1
description: 2018 Nips oral presentation으로 발표된 구글 브레인 논문입니다. 뇌인지과학 개념을 적용하여 강화학습 과제를 해결하였습니다. 적용한 방식도 굉장히 훌륭하지만 결과도 해당 task들의 SOTA를 찍어버린 놀라운 결과를 보여주고 있습니다.
category: Deep Learning
tags:
- MDN-RNN
- VAE
- CNN
- model-based RL
---

# World Models

구글 브레인에서 뇌인지과학 개념을 적용한 놀라운 강화학습 모델을 제안하였습니다. 답보 상태에 있었던 Open AI 과제를 쉽게 풀어버리는 대단한 결과를 보여주었습니다. 2018년 3월에 처음 나온 논문인데 이제서야 읽고 정말 대단하다는 생각이 들었습니다. 실험 결과 뿐만 아니라 실험에 적용한 'dream'이라는 방식도 상당히 놀랍습니다. 

## Introduction

<div align="center"><a href="https://imgur.com/Dc2Bx2t"><img src="https://i.imgur.com/Dc2Bx2t.png" title="source: imgur.com" width='500'/></a></div>

System dynamics의 창시자인 Jay Wright Forrester는 mental model에 관하여 다음과 같은 말을 했다고 합니다.

“The image of the world around us, which we carry in our head, is just a model. Nobody in his head imagines all the world, government or country. He has only selected concepts, and relationships between them, and uses those to represent the real system.”  
  
쉽게 말해, 사람들이 세상을 바라보고 이해할 때 세상 전체를 온전하게 받아들이는게 아니라 추상화 된 컨셉, 관계 등을 이용한다는 것입니다. 즉, 어떤 실제 사물을 바라보고 할 때 이를 'model'을 통해 걸러진 정보들을 가지고 이해합니다. 
  
하루하루 어마어마한 양의 정보가 들어오면 뇌(model)은 이러한 정보들의 spatial & temporal한 추상적인 부분들을 사용합니다. 

![What we see is based on our brain's prediction of the future](https://i.imgur.com/C6kuKhC.png)


이렇게 들어온 정보들을 세상을 바라보는 시점 뿐만 아니라 어떤 'action'을 행할지 예측하는 모델 개념으로도 이해할 수 있습니다. 위험한 상황에 처했을 때 사람들은 의식적으로 다음 행동을 계획하지 않고 즉각적으로 반응합니다.  


야구를 예로 들었을 때, 타자들은 야구 방망이를 어느 방향으로 휘두를지 밀리세컨드 안에 결정합니다. 이는 시각적인 정보가 뇌까지 들어오는 속도보다 짧습니다. 프로 야구선수들 같은 경우는 이런 식으로 예측하는 모델들과 근육들이 잘 훈련되었다고 할 수 있겠습니다.

복싱을 예로 들었을 때, 복싱 선수들은 어느 방향으로 움직여 상대의 주먹을 피할지 극도로 짧은 순간 내에 결정합니다. 이는 시각 정보가 뇌까지 들어오는 속도보다 훨씬 짧습니다. 프로 복싱 선수들 같은 경우는 이런 식으로 예측하는 'model'과 그에 반응하는 근육들(controller)들이 잘 훈련되었다고 할 수 있겠습니다(~~inference time이 짧습니다~~).


<div align='center'><a href="https://imgur.com/1ccoloG"><img src="https://i.imgur.com/1ccoloG.gif" title="무하마드 알리의 반사신경" /></a></div>

강화학습 agent 관점에서 RNN은 과거, 현재를 잘 표현할 뿐만 아니라 미래에 대한 정보를 잘 예측할 수 있음이 알려져 있습니다. 이는 모델이 위에서 언급한 spatial & temporal representation을 잘 표현한다는 말과 같습니다. 그러나 이렇게 파라미터가 많은 모델을 학습하고 바로 agent로 사용할 경우 credit assignment problem이 발생할 수 있습니다. 따라서 보통 일반적인 강화학습들은 파라미터 수가 적은 model-free RL을 사용합니다. 


![Imgur](https://i.imgur.com/lpDY9Yx.png)

Credit assignment problem란 쉽게 말해 어떤 경우에 얼마나 보상을 해야하는지를 말합니다. 체스 경기를 예로 들어보겠습니다. 체스에서의 reward는 전체 경기에서 이기고 지고 딱 한 번 입니다. 그러나 이에 영향을 끼치는 수는 굉장히 많습니다. 그렇다면 과연 어떤 수에 얼마만큼의 영향을 주어야 할까요? 이전의 말을 움직였던 수들이 어떤 전략적인 성과를 얻어냈으며 어느정도 규모인지 믿기가 쉽지 않습니다.

이러한 문제를 피하기 위해 논문에서는 RNN을 model로 활용하는 model-based RL 방법론을 제안합니다. 모델의 효과적인 학습을 위해 크게 large model과 smaller controler model로 나누었습니다. Large model은 다음 장면의 추상화된 정보들을 맞추는데 초점을 두었습니다. Controller model의 경우에는 small search space에서 효율적으로 credit assignment problem을 잘 해결하는 것에 초점을 맞추었습니다. 

대부분의 model-based RL에서는 실제 environment를 가지고 학습을 진행합니다. 여기서는 서론에서 이야기 했듯이 실제 environment를 large model을 통해 generate 되는 representation으로 대체하여 학습하는 실험도 진행하였습니다. 

## Agent Model

![Imgur](https://i.imgur.com/3PWEREN.png)

모델은 크게 다음과 같이 세 부분으로 구성되어 있습니다.

- Vision(V) : VAE(Variational Auto Encoder)
- Memory(M) : MDN-RNN(Mixture Density Networks - Recurrent Neural Network)
- Controller(C) : Simple single layer linear model


### VAE (V) Model

![Imgur](https://i.imgur.com/RVK37xN.png)

먼저 environment의 고차원 데이터를 encode시키는 VAE입니다. 입력 데이터는 영상의 프레임이며 상당히 고차원으로 이루어져 있습니다. 이를 controller에 바로 연결하기에는 꽤나 무리가 있기 때문에 vAE를 거쳐 저차원으로 압축된 latent vector $z$를 사용합니다. 논문에는 자세히 나와있지는 않지만 VAE를 사용한 이유로는 encode되는 latent vector가 가우시안 분포를 따르기 때문에 좀 더 실험 환경에 맞지 않을까 생각합니다. 또한 아래 설명할 MDN-RNN의 결과물인 다음 장면도 여러 개의 가우시안 분포를 따르므로 더 맞는 가정이 될 것입니다.


### MDN-RNN (M) Model

VAE가 입력 프레임을 압축시킨다면, MDN-RNN(이하 M)은 시간이 지남에 따라 어떤 상황이 발생하는지 예측합니다. 이렇게 sequential한 데이터를 사용하여 미래를 예측할 때는 RNN계열 네트워크들이 가장 효율적입니다. 논문에서는 RNN에 추가적으로 MDN(mixture density network)개념을 적용합니다. 이를 통해 RNN의 output으로 단순히 deterministic한 $z$값이 아닌 stochastic한 probability density function $p(z)$를 알 수 있습니다. 이렇게 MDN을 사용하게 되면 실제 복잡한 environment를 정해진 deterministic한 값이 아닌 stochastic한 값으로 표현할 수 있습니다. RNN이 하는 역할은 sequential한 다음 장면을 예측하는 것이기 때문에 확률적으로 표현한다면 실제 세계를 좀 더 설명력 있는 방식으로 설명할 수 있습니다.

![Imgur](https://i.imgur.com/vQr8THN.png)

MDN-RNN을 사용하면 V를 통해 encode된 $z$를 사용하여 다음 latent vector ${z}_{t+1}$의 probability distribution을 알아낼 수 있습니다. 수식으로 정리하면 다음과 같습니다.

$$P({ z }_{ t+1 }|{ a }_{ t },{ z }_{ t },{ h }_{ t })$$

- ${a}_{t}$ : action
- ${z}_{t}$ : latent vector
- ${h}_{t}$ : hidden state

여기에 추가적으로 temperature parameter $\tau $를 추가하여 모델의 uncertainty를 조절하였습니다. uncertainty가 높으면 높을수록 예측하는 프레임의 변동성이 높아지게 됩니다. (근데 어떤 방식으로 들어가는지 수식적으로 나온 부분이 없어서 적용 방식은 잘 모르겠습니다)

#### MDN - RNN

MDN의 기본 가정은 결과값이 여러 개의 가우시안 분포에서 확률적으로 존재하는 것입니다. RNN에 앞서 간단한 2중 FC layer로 하나의 값을 예측하는 MDN을 파이토치 코드로 나타내면 아래와 같습니다. 
(출처 및 자세한 내용은 [hadmaru's github](https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb)를 참고하시면 될 것 같습니다. 이러한 방식으로 이상치 탐지를 하는 방법 중 하나인 **[혼합 가우시안 밀도 추정법을 설명한 다른 포스트](https://jayhey.github.io/novelty%20detection/2017/11/03/Novelty_detection_MOG/)**를 읽어보셔도 좋을 것 같습니다)

```python
class MDN(nn.Module):
    def __init__(self, n_hidden, n_gaussians):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)  

    def forward(self, x):
        z_h = self.z_h(x)
        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return pi, sigma, mu
     
```
모든 결과물에 대해 더해서 1이 되어야 하므로 pi에 softmax 값을 취해줍니다. sigma에 torch.exp을 씌워주는 것은 분산은 음수가 될 수 없기 때문에 무조건 0보다 큰 값이 나오게 합니다. 구하고자 하는 것은 가우시안 분포를 따르는 아래 식과 코드를 따르는 결과물입니다. 

$$N\left( \mu ,\sigma  \right) \left( x \right) =\frac { 1 }{ \sigma \sqrt { 2\pi  }  } exp(-\frac { { \left( x-\mu  \right)  }^{ 2 } }{ 2{ \sigma  }^{ 2 } } )$$

```python
oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi)

def gaussian_distribution(y, mu, sigma):
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * oneDivSqrtTwoPI
```

Loss 함수는 아래와 같습니다. 단순 MSE loss를 사용해서는 확률 분포를 학습시킬 수 없습니다. 따라서 아래와 같이 크로스 엔트로피를 사용하여 최소화합니다.

$$E=-\log { \sum _{ i=1 }^{ m } p(c=i\mid x)N(y;\mu ^{ i },\sigma ^{ i }) } $$

```python
def mdn_loss_fn(pi, sigma, mu, y):
    result = gaussian_distribution(y, mu, sigma) * pi
    result = torch.sum(result, dim=1)
    result = -torch.log(result)
    return torch.mean(result)
```

위 코드는 단순 FC layer를 예시로 들었습니다. 논문의 MDN-RNN 결과물인 ${z}_{t+1}$의 경우 latent vector를 이루는 하나의 feature마다 $m$개의 가우시안 분포를 따르게 됩니다.

### Controller (C) Model

Controller는 environment가 지속적으로 변하는 상황에서 cumulative reward의 기대값을 최대화 하기 위한 action을 정합니다. 논문에서는 V와 M과 따로 학습을 시키기 위하여 일부러 최대한 파라미터 수가 적으면서도 간단한 모델을 만들었습니다. 
수식으로 나타내면 아래와 같습니다. $\left[ { z }_{ t }{ h }_{ t } \right]$는 concatenate vector이며 전체적으로 simple single layer linear model이라고 할 수 있습니다.

$${ a }_{ t }={ W }_{ c }\left[ { z }_{ t }{ h }_{ t } \right] +{ b }_{ c }$$

- ${a}_{t}$: action vector
- ${W}_{c}$: weight matrix
- ${b}_{c}$: bias vector

논문에서는 유전 알고리즘 중 하나인 CMA-ES(Covatiance-Matrix Adaptation Evolution Strategy)를 사용하여 모델 회적화를 진행합니다. 이 알고리즘은 유동적으로 search space를 조절하는 장점을 가지고 있습니다. 파라미터 수가 수천개 정도로 적은 모델에서 효과적이므로 많이 사용되는데, CPU를 사용하여 파라미터를 학습시켰다고 합니다. 


### 전체적인 구조

종합적으로 정리하면 아래 다이어그램과 같이 나타낼 수 있습니다.

![Imgur](https://i.imgur.com/mz10c8H.png)

<div>Observation(프레임)이 들어오면 ${z}_{t}$를 생성합니다. ${z}_{t}$와 M의 hidden state ${h}_{t}$를 C에 집어넣고 action vector ${a}_{t}$를 계산합니다. 세 값 모두 합쳐 future latent vector ${z}_{t+1}$과 다음 프레임에 쓰일 hidden state ${h}_{t+1}$을 계산합니다. 슈도 코드로 나타내면 아래와 같습니다.</div>

```python
def rollout(controller):
  ''' env, rnn, vae are '''
  ''' global variables  '''
  obs = env.reset()
  h = rnn.initial_state()
  done = False
  cumulative_reward = 0
  while not done:
    z = vae.encode(obs)
    a = controller.action([z, h])
    obs, reward, done = env.step(a)
    cumulative_reward += reward
    h = rnn.forward([a, z, h])
  return cumulative_reward
```

Controller가 매우 간단한 모델이기 때문에 practical한 장점이 있다고 합니다. VAE와 MDN-RNN은 커다란 모델이므로 GPU를 사용하여 모델 학습 및 inference가 가능합니다. 따라서 많은 weight들이 필요한 역할은 V와 M에서 수집한 데이터를 가지고 unsupervised하게 전부 끝내버릴 수 있습니다. 실제 test를 진행할 때, 학습이 끝난 V와 M을 GPU 세션 위에 올려놓고 inference만 진행합니다. 닭 잡는데 소 잡는 칼을 사용할 필요가 없는 것처럼, 파라미터 수가 적은 C는 inference로 나온 값들을 CPU를 사용하여 agent의 action을 결정합니다.

## 정리하면...

이번 포스트에서는 World Models의 전체적인 구조에 대해 살펴봤습니다. 사실 **[논문 저자의 블로그 포스팅](https://worldmodels.github.io)**내용을 한글로 쉽게 풀어 쓰고 제 의견을 덧붙인 부분이라 틀린 내용이 있을 수도 있습니다. 혹시나 의견 있으시면 댓글 남겨주시면 감사하겠습니다. 다음 포스트에서는 실험 결과와 이 논문의 하이라이트 중 하나인 'dream'에 대하여 포스팅 하도록 하겠습니다. 



> Reference
> * Ha, D., & Schmidhuber, J. (2018). World Models. arXiv preprint arXiv:1803.10122.
> * worldmodel blog : https://worldmodels.github.io/
> * hadmaru's github : https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb
