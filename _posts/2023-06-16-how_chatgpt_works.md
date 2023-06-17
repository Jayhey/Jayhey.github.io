---
title: ChatGPT가 작동하는 방법
description: ChatGPT 등장 전 LLM들이 가지고 있던 한계와 그 한계를 극복하기 위해 ChatGPT를 학습하는데 어떤 방법을 사용했는지 살펴보겠습니다.
category: Deep Learning
tags:
- ChatGPT
- LLM
---

# Introduction

ChatGPT가 나오면서 가장 충격이었던 부분은, 정말 자연스럽게 텍스트를 "생성"할 수 있었다는 점 입니다. 사실 ChatGPT도 다른 방식으로 학습하기는 했지만 결국 GPT 계열의 transformer를 쌓아 올린 모델일 뿐인데, 어떻게 이런 일이 가능했을까요? 

OpenAI에서는 supervised learning과 reinforcement learning을 조합하여 ChatGPT를 fine-tuning 했다고 밝히고 있습니다. 이 중에서 가장 중요하게 봐야 할 부분은 reinforcement learning 부분입니다. Reinforcement Learning from Human Feedback (이하 RLHF) 라고 불리는 이 방식은, ChatGPT가 물흐르듯이 텍스트를 자연스럽게 생성할 수 있는 원동력이라고 할 수 있겠습니다. 특히 민감한 사안들에 대한 굉장히 부적절한 표현들이(ex. 인종, 종교 등) 생성되지 않도록 하는데 큰 역학을 했다고 합니다.


## Capability vs. Alignment

머신러닝에서 **capability**와 **alignment**는 얼핏 비슷하다고 볼 수 있지만 서로 다른 상당히 중요한 개념입니다. 

머신러닝 모델의 **capability**는 해당 모델이 특정 작업을 얼마나 잘 수행할 수 있는지를 나타냅니다. 만약 모델이 classification task에서 주어진 데이터에따라 분류를 매우 잘 한다면 이는 capability가 높은 모델이라고 보시면 됩니다. 생성에서도 마찬가지고 모델이 "잘" 생성한다면 이 또한 capability가 높다고 할 수 있습니다. 이는 모델의 구조, 학습 데이터의 품질과 양, 학습 알고리즘 등에 의해 결정될 수 있습니다. 우리는 일반적으로 capability가 높은 모델을 성능이 좋다고 평가합니다.

**Alignment**는 머신러닝 시스템의 목표 또는 의도와 사람의 의도 또는 사회적 가치 간의 "일치"를 말합니다. 모델의 결과물이 나왔을 때 정말 사람이 원하는 결과가 나왔는지 여부를 의미한다고 할 수 있습니다. Alignment가 좋지 않다면, 비록 모델 입장에서 학습된 데이터만 가지고 봤을 때는 맞는 말처럼 보이더라도 윤리적으로 옳지 못할 수도 있습니다. 


<div align="center">
<img src="https://i.imgur.com/eP2LyBS.png" title="chatgpt" width="500"/>
</div>

요약하면, "capability"는 모델의 자체적인 성능이고, "alignment"은 모델의 목표와 사람의 의도 간의 일치를 의미라고 보면 됩니다. 

거대 언어 모델들을 학습하는데 수없이 많은 데이터가 필요하다 보니, 실제로 학습에 쓰이는 데이터에 부적절한 표현이 들어갈 수 있습니다. 이런 데이터를 사용하여 일반적인 random masking 방식으로 모델 파라미터를 학습한다면 비록 high capability는 확보 될지라도, low alignment 결과가 나올 수 있게 됩니다. 예를 들어 "이슬람교를 믿는 사람들이 폭력적인 이유가 있어?" 라는 질문이 들어왔을 때 low alignment 모델을 사용하게 되면 아무런 필터링 없이 부적절한 답변이 나올 수 있게 되는 셈이죠.

실제로 Instruct GPT 등장 전, 많은 모델들이 다양한 데이터로 학습되었지만 사람들이 원하는 답변을 제대로 생성하지 못하는 모습을 많이 보여주었습니다. 사람이 문장을 이어서 쓸 때 어떤 문장을 써야할지 다양한 배경 지식과 상황을 가지고 유추한다면 이런 모델들은 그냥 단순히 "확률"이 높은 문장들만 작성하는 것과 같습니다. 목적 함수가 다음 단어가 나오기 좋은 확률을 최대화 하는 수식이기 때문이겠죠? 이런 현상들은 아래 문제점들을 유발하게 됩니다. 
 
- Low alignment : 위에서도 언급했던 alignment 이슈입니다. 사람들이 진짜 원하는 답변을 생성할 확률이 낮아지게 되는 셈입니다.
- Hallucinations : 진짜 사실이 아닌 내용을 작성하거나 말도 안되는 문장들이 나오게 됩니다. 잘못된 데이터에 기반하여 현실과 일치하지 않는 정보를 생성하거나 예측할 수 있습니다. 
- Lack of interpretability : 모델이 어떤 이유로 이러한 문장을 생성했는지 인간이 이해하기가 어렵습니다. 정확한 인과관계 파악이 어려운 셈입니다.
- Generating biased or toxic output : 인간의 보편적인 가치에 반하는 반사회적인 답변이 나올 수 있습니다. 

## 기존 LLM 모델 학습법의 장단점

Transformer 기반의 모델들이 쏟아지면서 가장 많이 사용했던 학습 방법은 next token prediction 그리고 masked language modeling 입니다. 이 방법들이 인기 있는있는 이유는 아래와 같습니다.

- 문맥 이해: 모델이 문맥을 이해하고 예측하는 능력을 강화하는데 유용합니다. 모델은 텍스트 내에서 빈칸으로 표시된 부분의 내용을 추론하여 완성하는 구조이므로 문장에서 단어나 구문의 위치에 따른 의미와 관계를 파악하기 용이하죠.
- 비지도 학습: 학습 데이터를 생성하기 위해 레이블이 필요하지 않으며, 대규모의 텍스트 데이터만 있으면 바로 적용이 가능합니다. Pre-training 관점에서 굉장히 유용하다고 할 수 있습니다. Wikipedia, 뉴스 기사 등 방대한 데이터에 적용이 가능합니다.
- 정보 복원: 모델이 텍스트의 누락된 정보를 복원하는 능력을 향상시킵니다. 이는 자연어 자체를 이해하는데 도움이 될 뿐만 아니라 질의응답, 번역 등 다양한 분야에서 매우 유용합니다. 모델이 누락된 단어를 올바르게 예측할 수 있다면, 실제 데이터에서 유사한 상황에서도 좋은 예측을 할 수 있을 확률이 높아지게 됩니다. 

하지만 이런 학습법은 역시 위에서 얘기한 misalignment 현상을 유발할 수 있다는 것입니다. 이런 현상이 발생하는 이유는 아래 두 측면에서 설명이 가능합니다.

- Biases in training data : Masked Language Modeling은 입력 텍스트에서 일부 단어만 마스킹 하고 예측하도록 하는 방식입니다. 이때 가려진 단어를 맞추기 위해서는 학습 데이터에 해당 단어에 대한 적절한 문맥과 정보가 충분히 제공되어야 합니다. 그러나 실제 학습 데이터는 주로 인터넷의 대규모 텍스트 코퍼스로부터 수집됩니다. 이는 다양한 주제와 문체를 포함하지만, 학습 데이터 자체가 상당히 biased 될 수 있습니다. 예를 들어 특정 정치적 편향성을 가진 커뮤니티 데이터를 가지고 학습하면 misalignment 현상이 발생하기 쉽습니다.
- Limitations in generalization : 모델이 누락된 단어를 예측하는 작업에 초점을 맞추면서 문제가 발생하게 됩니다. 이러한 작업 자체는 언어의 통계적인 구조를 파악하는데 큰 도움이 될 수는 있지만 진짜 중요한 것이 무엇인지 확실하게 파악하는 것이 어려워집니다. 좀 더 고차원적인 언어의 표현 방식이나 문장의 뜻을 파악하는게 어렵다는 뜻입니다. 

이런 단점들을 해결하기 위해 RLHF를 활용한 Instruct GPT가 등장하게 됩니다. ChatGPT는 이 방법을 사용하기는 했으되, 약간 변형해서 학습했다고 밝히고 있습니다. (자세한 내용은 비공개)

## RLHF

RLHF는 다음과 같은 하위 개념을 정의하여 "사용자의 입력에 편향적이지 않고 안전하고 유용하게 반응"하는 모델을 만들고자 했습니다.
- Helpful : 사용자가 해결하려는 task에 도움이 되는 결과물을 만들어야 합니다.
- Truthfulness : 잘못된 정보나 사용자가 잘못 해석할 수 있는 생성은 피해야 합니다. 
- Harmless : 사회 및 개인에게 물리적, 정신적 악영향을 미치지 않아야 합니다. 

사실 위와 같은 개념을 수학적인 objective function으로 정의하고 모델을 학습시킨다는것은 상당히 어려운 작업입니다. 따라서 LLM이 생성한 문장들을 인간이 직접 평가할 수 있게 process를 정의한 것이 RLHF입니다.



<div align="center">
<img src="https://i.imgur.com/omm7iv3.png" title="chatgpt" width="800"/>
</div>

RLHF는 총 3가지 단계로 나뉘어져 있습니다. 

### Step 1: Supervised Fine-Tuning model

줄여서 SFT 라고 부르며 이 단계에서는 데이터를 수집하고 모델을 학습하게 됩니다. Prompt dataset에서 데이터를 가져와서, 사람들에게 가장 알맞은 답변을 작성하게 합니다. ChatGPT 기준으로는 사람들이 직접 작성한 prompt 그리고 OpenAI API request 이렇게 두 가지 prompt가 사용되었습니다. 

이 데이터를 사용하여 이제 모델을 fine-tuning 하는 과정을 거치게 되는데요, Instruct GPT 에서는 6B 사이즈의 GPT-3 모델을 사용하였고 ChatGPT에서는 GPT-3.5 모델을 fine-tuning 했다고 합니다.

<div align="center">
<img src="https://i.imgur.com/c1C84xU.png" title="chatgpt" width="800"/>
</div>

이 과정을 거치게 되면 모델이 어느정도 사람에 맞춰 align된다고 볼 수 있습니다. 상당한 양의 고품질 데이터를 확보가 생명이라고 보시면 될 것 같습니다.

 
### Step 2: Reward model training

그 다음으로는 fine-tuning 된 모델을 사용하여 여러 개의 output을 추출하고, 해당 output을 평가하는 reward model(RM)을 생성하게 됩니다. 단계적으로 설명하자면 먼저 학습된 모델에서 동일한 prompt를 대상으로 4~9개 사이의 output을 만들어냅니다. 이후에 사람들이 직접 이 output을 평가를 하게 되는데요, 좋은 답변에는 높은 점수를 주고 좋지 못한 답변에는 낮은 점수를 부여합니다. 그리고 마지막으로 이렇게 부여된 점수를 가지고 답변의 적절성을 판단하는 reward model을 학습시킵니다. 해당 모델은 문장들이 주어졌을 때 어떤 문장이 가장 적절한지 판단해주는 역할을 합니다.

<div align="center">
<img src="https://i.imgur.com/OknTSSJ.png" title="chatgpt" width="800"/>
</div>

Data augmentation 관점에서도 좋은 접근법인 것 같습니다. 사람이 직접 처음부터 모든 답변을 하나씩 만드는 것보다, 적절하게 align 된 모델의 output 순위를 매기는 것이 더 효율적인 접근 방식입니다.

### Step 3: Fine-tuning with Reinforcement Learning

다음으로는 이제 Proximal Policy Optimization (PPO) 방법론을 통해 SFT 모델을 다시 학습하게 됩니다. PPO는 강화학습 중에서 on-policy 알고리즘 중 하나로 에이전트가 훈련하는 동안 사용하는 policy와 동일한 policy를 평가하고 개선하는 알고리즘입니다. 즉, 에이전트는 현재 정책을 따라 환경과 상호작용하면서 학습을 진행합니다. DQN과 같이 off-policy 알고리즘과는 다르게 지속적으로 에이전트가 취하는 행동에 따라 업데이트가 이루어지는게 특징입니다. 

<div align="center">
<img src="https://i.imgur.com/AY8URKF.png" title="chatgpt" width="800"/>
</div>

먼저 파라미터가 freeze 된 SFT 모델과 freeze 되지 않은 SFT 모델을 준비합니다. 동일한 prompt를 넣었을 때 나오는 모델들의 output token들에 대한 분포를 가지고 KL divergence를 계산합니다. KL divergence를 구할 때 freeze 된 모델의 ouptut이 있기 때문에 너무 엉뚱한 결과값이 나와서 과하게 policy가 업데이트 되는것을 막을 수 있습니다.

Freeze 되지 않은 모델의 output y를 가지고 reword model을 통과시키면, ${r}_{θ}$(preferbility)를 산출할 수 있습니다. preferbility와 앞서 구한 KL divergence를 가지고 policy를 업데이트할 수 있는 loss를 구하게 됩니다. 해당 loss로 SFT 모델을 업데이트 한 뒤 동일한 과정을 반복하게 됩니다.

Step 3를 전체적으로 정리해보자면 reward model에서 나오는 "선호도"에 대한 reward를 SFT 모델(policy)에 적용하는 과정이라고 할 수 있습니다. 이 과정에서는 사람이 필요하지 않고 계속 반복이 가능하게 됩니다.

## RLHF의 단점?

이렇게 학습시킨 ChatGPT가 매우 좋은 성능을 보여주고는 있지만 사실 상당히 많은 단점이 존재하는 것도 사실입니다. 학습시키는 모델이 굉장히 주관적인 요소들에 의해 편향 될 수도 있는데요, 데이터를 만드는 labeler들의 주관이 매우 강하게 적용된다는 것입니다. Labeler 뿐만 아니라 모델을 학습시키는 연구원들이 labeler들에게 주는 가이드라인 자체도 문제가 될 수 있습니다. 또한 SFT모델을 학습시키는 prompt를 어떻게 샘플링하냐에 따라 모델 결과가 달라질 수도 있겠죠. Reward model을 학습시킬 때 순위를 매기는 사람의 주관도 강하게 반영될 수 있습니다. 그렇기 때문에 아래와 같은 기사들을 종종 발견할 수 있습니다.


> [오픈AI, 챗GPT "편향되거나 극단적으로 공격적이거나" 인정...이를 보완하는 새로운 버전 몇 달 안에 출시할 것!](https://www.aitimes.kr/news/articleView.html?idxno=27383)
> [전지전능한 챗GPT의 정체.. 알고보니 '젊은 백인 남성'?[양철민의 아알못]](https://www.sedaily.com/NewsView/29N42CHK1C)





> Reference
> * [How ChatGPT actually works](https://www.assemblyai.com/blog/how-chatgpt-actually-works/)
> * [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)






