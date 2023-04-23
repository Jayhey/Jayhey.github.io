---
title: LangChain이란?
description: LLM(ChatGPT 등)을 사용하여 다양한 application을 개발할 수 있는 프레임워크인 랭체인(LangChain)에 대해 알아보겠습니다
category: Deep Learning
tags:
- ChatGPT
- LangChain
---

# Langchain

## Langchain이란 무엇일까?

OpenAI가 2022년 11월 30일 ChatGPT를 발표하고나서 NLP쪽을 조금이라도 알고 계신 분들이라면 다들 엄청난 충격을 받으셨을 겁니다. 그야말로 기술이 세상을 바꿀 수 있다는 확신이 들게 만들어준 순간이기도 했습니다. 
이전 언어 모델들이 학습한 데이터를 기반으로 한정된 작업을 수행했다면, ChatGPT는 완성적인 대화형 AI의 모습을 보여주고 있습니다. 앞뒤 문맥 파악, 이전 대화 기억, 전문적인 지식에 대한 답변 등이전까지 볼 수 없는 수준의 능력을 가지고 있는 셈입니다. 심지어 시, 소설, 입사지원서 등 스스로 생각하지 못했던 부분에서 신뢰도가 높은 답변을 보여주고 있습니다.

그렇다면 ChatGPT를 활용한 다양한 어플리케이션을 개발할 수 있는 프레임워크에 대한 수요도 분명히 생길 것입니다. 이를 해결해줄 수 있는게 langchain이라고 할 수 있습니다.
Langchain을 한 문장으로 정의하면 다음과 같습니다.

- 다양한 언어 모델을 기반으로 한 어플리케이션을 개발하기 위한 프레임워크

이전 대화를 기억하고 문맥을 파악할 수 있는 기능을 가지고 있기 때문에 이를 활용하기 좋은 프레임워크만 있다면 다양한 방식으로 활용할 수 있습니다.
예전 GPT 기반 LLM 모델들의 경우 하나의 single prompt를 이해하는데는 문제가 없었습니다.
예를 들어 다음 문장들은 기존 LLM에서 문제 없이 답변하던 내용들입니다.

- bake me a cake
- bake me a vanilla cake with chocolate frosting

그런데 다음과 같은 문장들이 들어오면 어떨까요?

- give me the ingredients you need to bake a cake and the steps to bake a cake?

<div align="center">
<img src="https://imgur.com/Y6CxLJm.png" title="baking step" width="700"/>
</div>

사용자가 수동으로 각 단계를 정해주고 실행하는건 매우 번거로운 일입니다. 대신 langchain을 활용한다면 해당 단계들을 하나의 context로 포함시키고 이전 단계의 output을 자연스럽게 다음 단계로 넘기는 구조를 짜는게 훨씬 간단해질 수 있습니다.
ChatGPT가 원하는 답변을 줄 수 있도록 다양한 prompt를 사용해본 경험이 있으실 겁니다. 간단하게 이야기 한다면, 이런 prompt를 잘 엮고 제공하는 tool들을 사용하여 원하는 결과를 나올 수 있게 도와주는 프레임워크라고 볼 수 있습니다.



## Langchain을 쓰는 이유?

위에서도 이야기했듯이, 하나의 prompt만 가지고 답을 구하는건 이미 기존 LLM에서도 충분히 좋은 성능을 낼 수 있었습니다. 예를 들면 문서를 "요약"한다던지, 유사한 문서를 "검색"한다던지 이런 기능들은 이미 충분히 좋은 모델들이 많이 나와있습니다. 하지만 진짜 사람이 생각하는 것처럼 "추론"을 하는 관점에서는 약점이 많았습니다.


Langchain을 사용한다면, agent를 사용하여 "문제"를 "추론"하고 여러 개의 작은 sub-task로 분할하는게 가능합니다. 각 단계마다 context를 유지하기 위해 어떤 도구를 사용해야하는지 결정하고 memory, prompt 등 다양한 기능들을 사용하여 원하는 결과를 만들어내게 할 수 있습니다.

아주 간단한 예시를 들어보자면, 마이크로소프트에서 최근 copilot 기능 발표를 한 적이 있습니다. 아래 사진에서처럼 특정 문서를 주고, 제안서를 작성하라는 지시를 내리면 자연스럽게 해당 문서를 생성해주게 됩니다.


<div align="center">
<img src="https://i.imgur.com/28F6Ifo.png" title="copilot" width="700"/>
</div>

Langchain을 이용한다면 이런 복잡한 task를 쉽게 만들어줄 수 있는 기능을 직접 구현할 수 있습니다. 

아마도 근시일 내에 MS Office를 사용한다면 위 사진과 같은 것처럼 복잡한 기능 뿐만 아니라 다양한 문서 기능들도 언어 기반으로 지시내릴 수 있지 않을까요? 수백수천개의 버튼을 찾아서 클릭하는 것보다 "페이지 양식을 -문서1-과 동일하게 해줘" 처럼 자연어로 문서 서식을 변경하는게 훨씬 쉬울 것입니다.

## Langhain 사용 예시

가장 재미있었던 사용 사례는 Generative Agents 프로젝트입니다. 현재 [페이퍼](https://arxiv.org/abs/2304.03442)가 이미 등록되어 있으며  링크에서 데모를 확인할 수 있습니다. 
각 캐릭터들에게 인간 행동을 시뮬레이션 하게끔 만들어 놓습니다. 각자 직업이 있어서 누군가는 요리를 하고 출근하기도 하며, 예술가는 그림을 그리고 작가는 글을 씁니다. 우리가 매일 하는 것처럼 다음 날 뭐할지 계획하기도 하고 지난 시간을 반성하기도 합니다. 심지어 이를 기록하여 나중에는 더 높은 수준으로 에이전트에 반영합니다. 실제로 각 에이전트들이 아예 새로운 사회적 행동을 하는 것을 관찰할 수 있다고 합니다.
아래 데모페이지에서는 에이전트들이 어떤 행동을 했는지, 어떤 대화를 했는지 모두 기록되고 있는 것을 확인할 수 있습니다.

- [Generative Agents: Interactive Simulacra of Human Behavior](https://reverie.herokuapp.com/arXiv_Demo/
)

<div align="center">
<img src="https://i.imgur.com/A3ePCRe.png" title="Generative agents" width="700"/>
</div>



다음으로는 [Einblick Prompt](https://www.einblick.ai/prompt/)입니다. 
데이터 분석가들이 복잡한 코드를 짤 필요 없이 하나의 문장만 입력한다면 원하는 결과가 나올 수 있게 도와주는 툴이라고 볼 수 있습니다. 
아래 GIF를 보면 실제 파이썬 코드나 SQL 쿼리를 복잡하게 짤 필요 없이 간단하게 문장 하나만으로도 쉽게 원하는 결과가 나오는걸 볼 수 있습니다.
데이터를 plot할지, SQL로 원하는 결과를 가져와야 할지 스스로 결정하고 우선순위를 정하는 것까지 모두 langchain에서 제공하는 tool을 이용해서 만들었다고 합니다.

<div align="center">
<img src="https://sanity-media.einblick.ai/images/1xvnv7n3/production/664193915cde6d7c8a52ebb9851c8cac19a8f2b5-800x450.gif?q=75&fit=max&auto=format&dpr=2" title="Einblick" width="700"/>
</div>

위에 두 예시는 복잡해보이는 agent지만, 개인이 사용하기 좋은 맞춤형 agent도 langchain을 사용할 수 있다면 100줄~200줄 사이의 코드로 코드로 쉽게 만들 수 있습니다. 실제 langchain 공식 문서에서도 [BabyAGI 라는 간단한 비서 agent를 만드는 가이드](https://python.langchain.com/en/latest/use_cases/agents/baby_agi.html)가 예시로 나와있는데, NLP에 대한 지식이 조금만 있더라도 쉽게 이해 되는 코드를 볼 수 있습니다.
다음 포스트에서는 langchain의 기본 구조와 사용 방법 등에 대해 알아보도록 하겠습니다.



> Reference
> * [https://www.einblick.ai/blog/what-is-langchain-why-use-it/#what's-so-exciting-about-langchain](https://www.einblick.ai/blog/what-is-langchain-why-use-it/#what's-so-exciting-about-langchain)



