---
title: Langchain 훑어보기
description: Langchain의 기본 구성요소와 사용 방법에 대해 알아보겠습니다
category: Deep Learning
tags:
- ChatGPT
- LangChain
---

# Langchain 훑어보기

## Langchain의 구성 요소

지난 포스트에서 langchain이란 "LLM을 기반으로 한 어플리케이션을 개발하기 위한 framework"라고 정의 내렸습니다. 랭체인 공식 문서를 살펴보면 다음과 같은 개발 목표에 대해 설명하고 있습니다.

- 가장 잘 만들어지고 강력한 어플리케이션의 특징?
  - 단순 API만 가지고 LLM을 호출하지 않음
  - Be data-aware : 언어 모델을 다른 데이터 소스에 연결할 수 있음
  - Be agentic : 언어 모델이 다른 환경과 상호작용 할 수 있음

위 조건들을 잘 지킬 수 있는 framework를 만드는게 랭체인의 개발 목적이라고 할 수 있습니다. 

langchain에서는 'component'와 'user-case specific chain'라는 두 가지 개념이 있습니다. Component가 모여서 하나의 user-case specific chain이 된다고 보시면 될 것 같습니다.
Component는 langchain에서 LLM과 함께 작동하는데 필요한 기능을 제공하며 총 7가지가 있습니다.


<div align="center">
<img src="https://i.imgur.com/0p6uBPD.png" title="baking step" width="700"/>
</div>


## Component

### Schema

먼저 schema입니다. langchain의 input과 output이 될 수 있으며 가장 기본적인 입출력 형태라고 보시면 됩니다.
- Text : 가장 기본적으로 언어 모델과 소통하는 방식입니다. 파이썬의 str()과 같습니다.
- ChatMessages : 사용자와 모델이 서로 상호 작용하는 형태입니다. 아래와 같이 총 3가지로 이루어져 있습니다.
  - SystemChatMessage : 모델에게 사람이 지시하는 메시지
  - HumanChatMessage : 사람이 입력하는 메시지
  - AIChatMessage : AI의 출력 메시지
- Examples : 모델의 입/출력 쌍을 나타내는 방식입니다.
- Document : 데이터와 메타데이터를 하나로 표현하는 방식입니다.
  - page_content : 데이터의 내용
  - metadata : 데이터의 메타 정보를 나타냄


ChatMessage는 아래와 같이 모델의 응답을 강제(?)해서 다음 응답을 유도할 수 있습니다. 
```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

chat = ChatOpenAI()
chat(
  [ 
    SystemMessage(content="당신은 신혼여행 휴양지를 소개시켜주는 AI 봇입니다."),
    HumanMessage(content="1월 신혼여행지는 어디 가면 좋을까??"), 
    AIMessage(content="칸문을 추천 드립니다."), 
    HumanMessage(content="칸쿤은 이미 가봤어. 다른 곳은 없어?")
  ]
)

# AIMessage(content='1월에 신혼여행을 계획 중이시라면, 다음과 같은 여행지를 추천해드립니다.\n\n1. 뉴질랜드의 퀸스타운(Queenstown): 야외 활동이 풍부하고 아름다운 자연 경관으로 유명한 지역입니다. 스카이 다이빙, 스카이 스윙, 스노우보드 등 다양한 액티비티를 즐길 수 있습니다.\n\n2. 태국의 코 사무이(Koh Samui): 아름다운 해변과 푸른 바다, 그리고 현지 문화를 경험할 수 있는 다양한 투어가 준비되어 있습니다. 더불어 태국 요리 체험도 추천드립니다.', additional_kwargs={})
```



Document schema는 아래와 같이 표현됩니다. 위에서 나온 응답을 하나의 Document로 처리해서 다른 input으로 활용할 수도 있습니다.

```python
from langchain.schema import Document

Document(page_content="1월에 신혼여행을 계획 중이시라면, 다음과 같은 여행지를 추천해드립니다.\n\n1. 뉴질랜드의 퀸스타운(Queenstown): 야외 활동이 풍부하고 아름다운 자연 경관으로 유명한 지역입니다. 스카이 다이빙, 스카이 스윙, 스노우보드 등 다양한 액티비티를 즐길 수 있습니다.\n\n2. 태국의 코 사무이(Koh Samui): 아름다운 해변과 푸른 바다, 그리고 현지 문화를 경험할 수 있는 다양한 투어가 준비되어 있습니다. 더불어 태국 요리 체험도 추천드립니다.",
metadata={
    'my_document_id' : 234234,
    'my_document_source' : "The LangChain Papers",
    'my_document_create_time' : 1680013019
})
```

### Models

모델은 아래와 같이 input과 output이 정해져 있습니다. Text, ChatMessages 두 가지 형태가 입력될 수 있고 output으로는 Text, ChatMessages, Vector 세 가지 형태가 나올 수 있습니다. OpenAI api에서 기본적으로 임베딩 모델도 
같이 제공하기 때문에 텍스트의 임베딩 값도 여러 방식으로 저장해서 사용이 가능합니다.

<div align="center">
<img src="https://i.imgur.com/NsitoAJ.png" title="baking step" width="700"/>
</div>


### Prompt
명령어도 양식을 지정해서 입력 가능합니다. 일반적으로 **PromptTemplate**을 사용하며 아래와 같이 사용 가능합니다.
f-string을 사용하지 않고 템플릿을 설정해놓고 입력으로 사용할 수 있습니다. 

```python
from langchain import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(max_tokens=1024)
prompt = PromptTemplate(
    template="""
    [예시 문장]
    - 치킨은 살 안쪄요. 살은 내가 쪄요.
    - 치킨을 맛있게 먹는 101가지 방법. 101번 먹는다.

    [음식 정보]
    - {food} : {description}

    위 정보는 모두 음식 '{food}'에 관련된 내용입니다. [예시 문장]은 '치킨'을 가지고 만든 마케팅 문구입니다.
    당신은 음식회사 마케터입니다. 다양한 음식 중 하나를 선택해야하는 고객에게 '{food}' 추천 메시지를 전송해야 합니다. 
    [예시 문장]과 [음식 정보]를 참고하여 다음 조건을 만족하면서 '{food}'을 권유하는 짧은 메시지 5개를 리스트 형식으로 생성해주세요. 리스트는 '-' 단위로 구분되어야 합니다.
    - 문장의 길이는 42자 이하로 작성
    - 메시지를 받는 사람은 배달음식을 주로 시킴
    ...
    - 고객이 흥미를 느낄 수 있도록 발랄한 어투로 작성 
    """,
    input_variables=["food", "description"],
)

_input = prompt.format(food="pizza", description=description["pizza"])
output = llm(_input)
```

위 방식처럼 단순히 template도 지정 가능하지만, **OutputParser**를 사용하면 원하는 형태의 output을 받게 지정할 수도 있습니다. 아래 예시에서는 list 형태로 받고 싶을 때 사용하는 함수를 썼지만, json 같이 원하는 형태로도 반환 받을 수 있습니다.

```python
from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="""
    [예시 문장]
    - 치킨은 살 안쪄요. 살은 내가 쪄요.
    - 치킨을 맛있게 먹는 101가지 방법. 101번 먹는다.

    [음식 정보]
    - {food} : {description}

    위 정보는 모두 음식 '{food}'에 관련된 내용입니다. [예시 문장]은 '치킨'을 가지고 만든 마케팅 문구입니다.
    당신은 음식회사 마케터입니다. 다양한 음식 중 하나를 선택해야하는 고객에게 '{food}' 추천 메시지를 전송해야 합니다. 
    [예시 문장]과 [음식 정보]를 참고하여 다음 조건을 만족하면서 '{food}'을 권유하는 짧은 메시지 5개를 리스트 형식으로 생성해주세요. 리스트는 '-' 단위로 구분되어야 합니다.
    - 문장의 길이는 42자 이하로 작성
    - 메시지를 받는 사람은 배달음식을 주로 시킴
    ...
    - 고객이 흥미를 느낄 수 있도록 발랄한 어투로 작성 

    {format_instructions}
    """,
    input_variables=["food", "description"],
    partial_variables={"format_instructions": format_instructions}
)

_input = prompt.format(food="pizza", description=description["pizza"])
output = llm(_input)

# format_instructions 값 : Your response should be a list of comma separated values, eg: `foo, bar, baz` 
```

다만 실제로 위 prompt를 생성해서 만들어보면 아래 코드의 "partial_variables"부분이 특정한 API를 사용한게 아니라 단순히 지시문을 가져왔다는 것을 확인할 수 있습니다. 저도 직접 실행해보기 전까진 OpenAI의 특정 api를 활용하는 줄 알았는데 말 그대로
그냥 정해진 지시문을 붙여 넣어 주는 역할을 하고 있었습니다.

### Index

Index는 input으로 들어온 문서를 구조화하고 모델과 상호작용하는데 사용합니다. 

- Document Loaders
  - 다양한 형태의 문서들을 텍스트 모음으로 로드함
  - CSV, Bigquery Table, PowerPoint, Facebook chat, Slack, HTML 등 지원 (공식 문서 지원 loader 확인)

```python
# text 데이터 불러오기 예시
from langchain.document_loaders import TextLoader
loader = TextLoader('../state_of_the_union.txt', encoding='utf8')
```

- Text Splitters
  - 문서가 너무 길어서 LLM에 한 번에 입력이 어려운 경우 사용
  - 긴 텍스트를 처리하기 위해 여러 조각으로 나누어 줌 
  - 텍스트의 분할/결합/생성 등이 조건에 따라 이루어짐

```python
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
```


- Vector Stores
  - 텍스트를 임베딩한 벡터 저장/관리/검색을 지원하는 저장소
  - Elastic search, Opensearch, FAISS, Milvus 등 다양한 검색 엔진 사용 가능 
아래 코드는 벡터 유사도 검색 라이브러리인 FAISS 예시입니다.

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
```


- Retrievers
  - 문서와 언어 모델을 결합해주는 역할
  - 문자열을 받아서 document 목록을 반환
  - vectorstore, ChatGPT Plugin 등 다양한 retriever 존재 

아래 코드는 위에서 설정헀던 FAISS db를 retirever로 설정한 예시입니다. "what did he say about ketanji brown jackson" 문장과 가장 가까운 문서를 FAISS를 통해 DB에서 찾아줄 수 있습니다.

```python
retriever = db.as_retriever()
docs = retriever.get_relevant_documents("what did he say about ketanji brown jackson")
```

### Memory

메모리는 LLM이 데이터를 저장하고 검색할 수 있게 도와줍니다. 메모리를 사용하면 단기, 장기 기억이 가능하며 연속적인 채팅이 가능합니다. Chain이나 Agent가 일종의 '기억'을 갖는 것처럼 만들어 준다고 할 수 있습니다. 다만 메모리라는 개념 자체가 기존 채팅 기록들을 prompt에 넣어 제공하도록 구성되어 있습니다. (새로운 기능을 사용하는게 아님!) 따라서 많은 내용을 기억시키기는 어려우며 단순 prompt 확장과 같은 개념이라고 보셔도 될 것 같습니다.

대표적으로 ChatMessageHistory가 있으며 아래처럼 사람과 AI의 대화를 지정해서 저장해놓을 수 있습니다.

```python
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()
history.add_user_message("hi!")
history.add_ai_message("whats up?")

history.messages
# [HumanMessage(content='hi!', additional_kwargs={}), AIMessage(content='whats up?', additional_kwargs={})]
```

### Chain
랭체인에서 가장 핵심적인 개념 중 하나인 chain입니다. Chain의 뜻 그대로 데이터 -> LLM1 -> LLM2 ... -> plugin -> output 등 다양한 방식으로 사용자가 원하는대로 모듈들을 엮을 수 있습니다. 


<div align="center">
<img src="https://i.imgur.com/DyHUWvm.png" title="baking step" width="500"/>
</div>

아래 코드는 하나의 LLM을 가지고 순차적인 실행을 하는 방법을 보여줍니다. SimpleSequentialChain이라는 모듈로, 특정 Text가 들어오면 주어진 순서대로 작동하는 모습을 보여줍니다. 아래 코드는 "제목"을 입력하면 첫 번째 단계에서 먼저 "제목"에 맞춰 특정 연극에 대한 내용을 창작하고 두 번째 단계에서 해당 연극에 대한 리뷰를 작성해주는 단계를 밟습니다. 즉, 연극의 제목만 입력하면 리뷰를 받아볼 수 있는 chain입니다.

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# This is an LLMChain to write a synopsis given a title of a play.
llm = OpenAI(temperature=.7)
template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

Title: {title}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

# This is an LLMChain to write a review of a play given a synopsis.
llm = OpenAI(temperature=.7)
template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template)


# This is the overall chain where we run these two chains in sequence.
from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain], verbose=True)

review = overall_chain.run("Tragedy at sunset on the beach")
```

단순히 이렇게 순서를 정해주는 chain도 있고, 다른 방식으로 적용도 가능합니다. 한 번에 처리가 어려운 긴 문서가 들어왔을 때는 load_summarize_chain을 사용해서 chunk별로 나눈 요약을 합칠 수도 있습니다. 아래 코드에서는 text_splitter를 사용하여 들어온 긴 텍스트를 잘게 나눠주고 Document로 지정을 해줍니다. 그리고 load_summarize_chain을 사용하여 텍스트들을 한 번에 나눠주는걸 확인할 수 있습니다.


```python
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

llm = OpenAI(temperature=0)

text_splitter = CharacterTextSplitter()

with open("../../state_of_the_union.txt") as f:
    state_of_the_union = f.read()
texts = text_splitter.split_text(state_of_the_union)

docs = [Document(page_content=t) for t in texts[:3]]


chain = load_summarize_chain(llm, chain_type="map_reduce")
chain.run(docs)
```

### Agent

Agent 또한 chain과 더불어 랭체인에서 가장 중요한 개념 중 하나라고 볼 수 있습니다. 다양한 tool을 사용하여 사용자의 입력에 따라 상황에 맞게 다른 결과를 반환받을 수 있습니다. 이제까지 설명했던 component들이 모두 LLM과 text와의 상호 작용 위주였다면, agent는 외부 다른 리소스와 상호작용이 가능하게끔 도와줍니다. ChatGPT가 2021년 까지의 데이터만 가지고 학습된만큼 최신 데이터를 실시간으로 검색해서 사용하고 싶다거나, 특정 API를 손쉽게 사용하고 싶다면 agnet를 활용하여 개발이 가능합니다.

기본적으로 agent에는 다양한 tool들이 있습니다. 구글 검색 결과를 활용할 수 있는 serpapi 부터 날씨 정보를 알 수 있는 openweather api 등 수많은 기능들을 제공하며 지금 현재도 계속 추가되고 있습니다. 

먼저 정해진 tool을 사용하느 간단한 코드 예시입니다. 여기서는 구글 검색(serpapi)과 수학 계산용 모델(llm-math)를 사용하게 됩니다. 재미있는건 tool을 준다고 다 사용하는건 아니고 적절한 tool을 ChatGPT 스스로 **선택**한다는 점입니다. 기존의 프로그래밍 프레임워크들이라면 개발자가 우선순위를 하나씩 지정했지만 여기서는 툴의 description을 읽고 스스로 모델이 어떤걸 사용해야할지 결정을 합니다.

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
# 적합한 툴을 줄 수 있음
tools = load_tools(["serpapi", "llm-math"], llm=llm) #툴을 사용하는데 있어서 필요한 LLM

# AgentType : 주어진 툴에서 뭐가 최적인지 고르는 방식
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True) 

agent.run("가수 RAIN의 아내가 누구야? 그녀의 나이에 0.37을 곱하면?")

# 실행 과정 
""" 
> Entering new AgentExecutor chain...
 RAIN의 아내를 찾아야 하고, 그녀의 나이를 곱해야 한다.
Action: Search
Action Input: RAIN's wife
Observation: Kim Tae-hee
Thought: Kim Tae-hee의 나이를 찾아야 한다.
Action: Search
Action Input: Kim Tae-hee age
Observation: 43 years
Thought: 43에 0.37을 곱해야 한다.
Action: Calculator
Action Input: 43 * 0.37
Observation: Answer: 15.91
"""
```

아래 코드처럼 도구들에 대한 커스터마이징이 필요하다면 아래처럼 따로 지정할 수도 있습니다. 모델이 어떤 도구를 사용하는지 판단하는 문장은 description 파라미터이므로 해당 설명을 바꿔준다면 어떤 도구를 사용할지 판단하는 기준을 바꿀 수도 있습니다. 위의 코드에서 AgentType의 **ZERO_SHOT_REACT_DESCRIPTION**이 도구의 판별 방법 중 하나입니다. 이 방법 말고도 react_docstore, self-ask-with-search 등등 다양한 방법들을 사용할 수도 있습니다. 


```python
# 도구 사용법
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, Tool, initialize_agent, tool
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool


llm = ChatOpenAI(temperature=0)
search =SerpAPIWrapper()

# 아래처럼 커스텀 가능 
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
]
```

우선순위를 정하고 싶다면 description에 한 줄 추가하면 됩니다. 아래 코드는 음악 검색이 들어오면 이 tool을 쓰라는 description을 추가만 해줬을 뿐인데 모델이 스스로 판단해서 음악 검색 관련 문구가 들어오면 해당 함수를 실행하는 것을 확인할 수 있습니다.

```python
# 도구 간 우선순위 설정하기
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Music Search",
        # Mock Function
        func=lambda x: "'All I Want For Christmas Is You' by Mariah Carey.", 
        description="A Music search engine. Use this more than the normal search if the question is about Music, like 'who is the singer of yesterday?' or 'what is the most popular song in 2022?'",
    )
]

agent = initialize_agent(tools, OpenAI(temperature=0), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("what is the most famous song of christmas?")

# 지정해놓은 Mock Function 실행함
"""
> Entering new AgentExecutor chain...
 I should search for a popular christmas song
Action: Music Search
Action Input: most famous christmas song
Observation: 'All I Want For Christmas Is You' by Mariah Carey.
Thought: I now know the final answer
Final Answer: 'All I Want For Christmas Is You' by Mariah Carey.
"""
```

Toolkit이라는 json, pandas dataframe 등을 자체적으로 다루는 방법도 있습니다. 이 뿐만 아니라 python toolkit, SQL db toolkit 등 다양한 툴킷을 제공하고 있습니다.


```python
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd

df = pd.read_csv('titanic.csv')

agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
agent.run("how many rows are there?")

"""
> Entering new AgentExecutor chain...
Thought: I need to count the number of rows
Action: python_repl_ast
Action Input: len(df)
Observation: 891
Thought: I now know the final answer
Final Answer: There are 891 rows in the dataframe.

> Finished chain.
"""
```


## Custom Agent with PlugIn Retrieval

위의 Component들을 활용해서 ChatGPT plugin retrieval을 만들 수 있습니다. 
먼저 아래와 같이 다양한 플러그인을 정의합니다. 해당 플러그인들은 OpenAI 에서 지정한 양식으로 만들어져 있으며 위에서 tool을 사용하는 것처럼 description이 모두 지정되어 있습니다. (URL 눌러서 내용 확인 가능)

아래 코드는 [공식 문서](https://python.langchain.com/en/latest/use_cases/agents/custom_agent_with_plugin_retrieval.html) 예시입니다. 
```python
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent_toolkits import NLAToolkit
from langchain.tools.plugin import AIPlugin
import re


llm = OpenAI(temperature=0)

urls = [
    "https://datasette.io/.well-known/ai-plugin.json",
    "https://api.speak.com/.well-known/ai-plugin.json",
    "https://www.wolframalpha.com/.well-known/ai-plugin.json",
    "https://www.zapier.com/.well-known/ai-plugin.json",
    "https://www.klarna.com/.well-known/ai-plugin.json",
    "https://www.joinmilo.com/.well-known/ai-plugin.json",
    "https://slack.com/.well-known/ai-plugin.json",
    "https://schooldigger.com/.well-known/ai-plugin.json",
]

AI_PLUGINS = [AIPlugin.from_url(url) for url in urls]
```
이제 위 셀에서 정의한 plugin들의 description 값을 모두 벡터로 변환하여 임베딩 합니다. 이렇게 도구가 많을 경우 similarity search를 사용하여 주어진 문장에 가장 적합한 tool이 뭔지 찾을 수 있습니다. 아래 코드에서는 FAISS를 사용하여 임베딩 하는 방법을 보여주고 있습니다. (description text -> Document -> Embedding -> FAISS)


```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

embeddings = OpenAIEmbeddings()
docs = [
    Document(page_content=plugin.description_for_model, 
             metadata={"plugin_name": plugin.name_for_model}
            )
    for plugin in AI_PLUGINS
]
vector_store = FAISS.from_documents(docs, embeddings)
toolkits_dict = {plugin.name_for_model: 
                 NLAToolkit.from_llm_and_ai_plugin(llm, plugin) 
                 for plugin in AI_PLUGINS}
```

그리고 해당 벡터 스코어에서 embedding vector 기반으로 유사도가 가장 높은 plugin을 찾는 함수를 만들어줍니다.
해당 함수를 실행하면 구입 관련 plugin이 가장 첫 번째로 나오는 것을 확인할 수 있습니다.
```python

# 임베딩 벡터 기반으로 tools 우선순위 서치
retriever = vector_store.as_retriever()

def get_tools(query):
    # Get documents, which contain the Plugins to use
    docs = retriever.get_relevant_documents(query)
    # Get the toolkits, one for each plugin
    tool_kits = [toolkits_dict[d.metadata["plugin_name"]] for d in docs]
    # Get the tools: a separate NLAChain for each endpoint
    tools = []
    for tk in tool_kits:
        tools.extend(tk.nla_tools)
    return tools

tools = get_tools("what shirts can i buy?")

# 구입 관련 plugin을 보여줌 
print(tools[0].name)
# Open_AI_Klarna_product_Api.productsUsingGET'
```

Prompt template & OutputParser는 너무 길어서 공식 문서를 참고해주시면 좋을 것 같습니다. prompt와 output parser가 지정되었다면 chain과 agent를 지정해주고 executor를 통해서 문장을 입력해주면 됩니다. agent_executor는 미리 지정해둔 형식대로 필요한 plugin을 찾고 해당 플러그인을 LLM과 결합하여 사용하게 됩니다.


```python
# prompt template, output parser 지정
# ...

llm = OpenAI(temperature=0)

llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

agent_executor.run("what shirts can i buy?")

"""
> Entering new AgentExecutor chain...
Thought: I need to find a product API
Action: Open_AI_Klarna_product_Api.productsUsingGET
Action Input: shirts

Observation:I found 10 shirts from the API response. They range in price from $9.99 to $450.00 and come in a variety of materials, colors, and patterns. I now know what shirts I can buy
Final Answer: Arg, I found 10 shirts from the API response. They range in price from $9.99 to $450.00 and come in a variety of materials, colors, and patterns.

> Finished chain.
"""

```

이 외에도 공식 문서를 살펴보면, 다양한 use-case가 많이 있으니 참고 해주시면 좋을 것 같습니다.
 

## 정리하며
이번 포스트에서는 langchain component 들에 대해서 간단히 알아봤습니다. ChatGPT를 사용하면서 아쉬웠던 부분이나 필요한 부분들을 다양한 방식으로 제공하는 것을 확인할 수 있었습니다. 상당히 재미있던 것은 langchain 내에서 제공하는 함수들이 대부분 "자연어"로 ChatGPT를 구성할 수 있게 코드가 구성되어 있다는 점이었습니다. 기존 프레임워크들처럼 복잡한 함수로 이루어진게 아닌 메인 ChatGPT 모델 한 개만 두고 자연어로만 외부 확장이 가능하도록 개발되어 있다고 볼 수 있겠네요. 
현재 버전이 0.0.153인데 거의 2~3일에 0.0.001씩 오르고 있습니다. 그만큼 업데이트가 빠르고 아직 불완전한 프레임워크라고도 볼 수 있는데요, 마치 텐서플로우 초기 버전을 보는 듯한 느낌이 들기도 합니다. 공식 문서도 설명이 조금 부족하다는 생각이 드는데, 사용이 많이 활성화되고 contribute도 많은 만큼 조금만 시간이 지나면 대세가 되지 않을까 생각이 듭니다.

> Reference
> * [Langchain 공식 문서](https://python.langchain.com/en/latest/index.html#)



