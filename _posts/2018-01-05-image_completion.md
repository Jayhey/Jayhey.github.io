---
title: Globally and Locally Consistent Image Completion(이미지 복구)
description: GAN을 활용한 많은 연구들이 있지만 그 중에 굉장히 흥미로운 연구 결과가 있습니다. 이미지의 특정 부분을 지워도 자동으로 원복해주는 딥러닝을 활용한 image completion 기법입니다. 
category: Deep Learning
tags: 
- CNN
- GAN
- image completion
---

시작하기에 앞서 아래 자료들은 서덕성 석사과정의 deeplearning weekly 스터디 발표 자료에서 가져왔음을 밝힙니다.

GAN(Generative Adversarial Network)을 활용한 연구들이 굉장히 많이 생겨났습니다. "창작"이라는 기계가 할 수 없었던 일들도 척척 해내는 걸 보고 사람들이 많은 관심을 가지게 되었기 때문이죠. GAN으로 재미있는 결과물들을 낼 수는 있지만 활용 측면에서는 사실 다른 기법들보다 조금 떨어지는게 사실입니다. 그래서 예전보다는 조금 사그라들기는 했지만 이번에 매우 재미있는 논문이 새로 나왔습니다. 바로 Globally and Locally Consistent Image Completion이라는 GAN을 활용한 이미지 복구입니다. 


# Globally and Locally Consistent Image Completion
 
## Introduction

Image completion에는 여러 종류가 있는데 사물 지우기(Object removal), 장면 완성시키기(scene completion), 얼굴 완성시키기(face completion) 등이 있습니다. 전부 풀기 힘든 문제인데, 이는 texture pattern을 완전히 완성시키기 어렵기도 하고 어떤 물체의 구조적인 특징이나 주변 사물과의 연관관계 등 고려해야할게 한두가지가 아니기 때문입니다. 논문에서는 이미지의 전체 맥락을 고려하면서도 지역적인 부분까지 고려하는 이미지 복구 방법론을 제시합니다. 

네트워크는 크게 세 부분으로 나뉘며 CNN과 GAN을 사용하였습니다. 

- Completion network : CNN이며 이미지를 복구시킵니다.
- Global context discriminator : 이미지의 전체적인(global) 맥락을 파악합니다.
- Local context discriminator : 이미지의 세부적인(local) 부분을 판단합니다.


이 논문의 contribution을 요약해보자면 다음과 같습니다. 

- A high performance network model that can complete arbitrary missing regions
- A globally and locally consistent adversarial training approach for image completion
- Results of applying our approach to specific datasets for more challenging image completion

유튜브 영상을 보시면 상당히 뛰어난 성능을 보여주는 것을 알 수 있습니다.

<iframe width="1903" height="764" src="https://www.youtube.com/embed/qzIbDf_GfAk" frameborder="0" gesture="media" allow="encrypted-media" allowfullscreen></iframe>

## Related work

이전부터 이미지 복구 관련 연구들은 꾸준히 있었습니다. difussion-based, patch-based, cnn-based approach 그리고 context encoder입니다. 예전 방법들에 대해 간단히 살펴보도록 하겠습니다.

### Difussion-based image synthesis

Difussion-based 방식은 비어 있는 부분 바로 근처에 있는 픽셀들을 그대로 가져오는 방식입니다. 아래 그림처럼 오래돼서 바랜 사진같은 경우 적용할 수 있는 방법이지만, 스크래치처럼 매우 작고 좁은 부분이 유실되었을 때만 사용할 수 있는 한계가 있습니다.

![Imgur](https://i.imgur.com/RTNbAMP.png)

### Patch-based image synthesis

Patch-based 방식은 difussion-based 방식과는 다르게 크게 비어있는 부분도 채울 수 있습니다. 소스 이미지에서 유사한 부분을 찾아 메꾸는 방식입니다. 이러한 방식을 사용하기 위해서 optimal patch search 방법들이 많이 연구되었습니다. 하지만 만약 소스 이미지에 없는 특이한 오브젝트들을 채워야 하는 경우라면, 이 방법으로는 불가능합니다. 아래 그림을 예시로 들어보겠습니다. 아이가 자고있는 모습인데(비행기 내부로 보입니다) 비행기 창문이나 벽 등은 잘 복구했지만 헤드폰과 얼굴 앞쪽 붕 떠있는 부분은 제대로 구현하지 못했습니다. 

![Imgur](https://i.imgur.com/VE4bDx7.png)

### CNN-based image synthesis

다음으로는 CNN-based 방식입니다. Difussion-based와 patch-based 방식이 결국 소스이미지만 이용해서 빈 공간을 채워넣는 방식을 썼다는 한계점이 있습니다. 위에서처럼 어색한 헤드폰이 완성되는것도 이러한 이유 떄문입니다. 그래서 CNN 기반 방법론도 등장했지만 이는 매우 작고 얇은 mask에만 잘 적용된다는 단점이 있습니다. 하지만 좋은 성능을 내줌에도 불구하고 계산량이 매우 많다는 단점이 있습니다. 

![Imgur](https://i.imgur.com/0Oa63Lj.png)

### Context encoder

CNN-based 방식에서 확장한 context encoder방식도 있습니다. 작은 mask에만 적용 가능하다는 단점을 극복하여 큰 mask에도 적용 가능하게 만들었습니다. GAN 기반이며 adversarial loss를 사용합니다. 또한 channel-wise fully-connected layer를 사용했다는 특징이 있습니다.

![Imgur](https://i.imgur.com/tVmv4O4.png)

결국 기존 방법들과 비교하면 아래와 같습니다. 논문에서는 제안된 방법론이 이미지 사이즈도 자유롭게 적용 가능하고 local consistency, sementic 그리고 novel object까지 구현이 가능하다고 말하고 있습니다. 그럼 이제 어떻게 네트워크 구조를 구성했는지 살펴보도록 하겠습니다.

![Imgur](https://i.imgur.com/WZHm4kk.png)


## Proposed method

먼저 네트워크 구조를 보면 아래와 같습니다. 원본 이미지와 mask를 completion network에 입력합니다. 여기서 completion network는 이름 그대로 이미지를 완성시켜주는 네트워크입니다. 여기에 두 개의 discriminator network가 있는데, global discriminator와 local discriminator입니다. 이 네트워크들은 복구시키는 이미지를 자연스럽게 만들어주는 역할을 합니다. 즉, completion network는 discriminator network를 진짜인지 가짜인지 속이는(fool) 역할을 하고, discriminator network는 복구된 이미지의 진위 여부를 파악하는 것입니다. GAN의 전체적인 개념을 따라가는 모습을 보여주고 있습니다.

![Imgur](https://i.imgur.com/i6vOVos.png)

### Dilated convolution

이 논문에서 제안한 네트워크 구조에서 특이한 점이라면 completion network에 일반적인 convolution layer가 아닌 dilated convolution layer를 사용했다는 점입니다. Dilated convolution은 일반적인 convolution과 파라미터 수와 연산량은 같으면서도 훨씬 더 넓은 부분을 한 번에 볼 수 있다는 장점이 있습니다. 논문의 저자들은 아래 오른쪽 그림을 예시로 들면서 일반적인 convolution을 쓰면 입력된 이미지의 비어있는 부분이 큰 부분을 보지 못한다고 말하고 있습니다. 하지만 dilated convolution을 사용하면 가장 오른쪽 그림처럼, 비어있는 부분 가운데에 있는 픽셀이라도 전체적으로 다 커버를 할 수 있게 됩니다.


![Imgur](https://i.imgur.com/PenSs3O.png)


<div>수식은 아래와 같습니다. ${ k }_{ h }$와 ${ k }_{ h }$는 각각 커널의 높이와 넓이입니다. $\eta$는 dilation factor로 convolution을 얼마나 더 dilate(팽창) 시킬 것인지 정해주는 하이퍼 파라미터입니다. $\eta$가 1이면 일반적인 convolution 커널과 똑같다고 생각하면 됩니다.</div>

$${ y }_{ u,v }=\sigma (b+\sum _{ i=-{ k }_{ h }^{ ' } }^{ { k }_{ h }^{ ' } }{ \sum _{ j=-{ k_{ w }^{ ' } } }^{ { k }_{ w }^{ ' } }{ { W }_{ { k }_{ h }^{ ' }+i,{ k }_{ w }^{ ' }+j }{ x }_{ u+\eta i,v+\eta j } }  } ),$$

$${ k }_{ h }^{ ' }=\frac { { k }_{ h }-1 }{ 2 } ,\quad { k }_{ w }^{ ' }=\frac { { k }_{ w }-1 }{ 2 } ,$$

### Completion nework

![Imgur](https://i.imgur.com/4Hh0nS9.png)

Completion network에 들어가는 입력값으로는 RGB채널의 빈 공간이 있는 이미지와 이 부분을 마스킹한 바이너리 채널 이미지 총 2개가 들어갑니다. 네트워크 구조 그림에서 Image+Mask 부분이 입력값입니다. 아래 코드가 mask를 만드는 부분이니 한 번 보시면 이해가 쉽게 되실겁니다.

```python
def get_points():
    points = []
    mask = []
    for i in range(BATCH_SIZE):
        x1, y1 = np.random.randint(0, IMAGE_SIZE - LOCAL_SIZE + 1, 2)
        x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
        points.append([x1, y1, x2, y2])

        w, h = np.random.randint(HOLE_MIN, HOLE_MAX + 1, 2)
        p1 = x1 + np.random.randint(0, LOCAL_SIZE - w)
        q1 = y1 + np.random.randint(0, LOCAL_SIZE - h)
        p2 = p1 + w
        q2 = q1 + h
        
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        m[q1:q2 + 1, p1:p2 + 1] = 1
        mask.append(m)


    return np.array(points), np.array(mask)
```

이 네트워크의 목적 자체가 이미지의 복구이기 때문에 빈 공간이 아닌 부분은 네트워크를 통과해도 여전히 똑같은 모습을 유지합니다. 전반적인 구조는 인코더, 디코더와 비슷한 모습을 보입니다. 디코더 부분에서 deconvolution layer를 통과하면서 이미지가 복구되는 과정을 거칩니다. 특이할 만한 점이라면 다른 네트워크에서는 pooling layer를 사용하여 해상도를 줄이는데, 이 네트워크에서는 strided convolution만 사용하여 이미지의 해상도 1/4만큼 낮춥니다. 이는 이미지 빈 공간에 non-blurred texture를 생성하는데 중요한 역할을 합니다. 그리고 그림에서 주황색으로 색칠된 layer가 dilated convolution을 사용합니다. Image completion에서 이미지의 전체적인 context가 사실적으로 복구하는데 매우 중요하기 때문에 dilated convolution으로 넓은 부분을 한 번에 보면 더 좋은 결과가 나오게 됩니다. 

$$L(x,{ M }_{ c })=||{ M }_{ c }\odot (C(x,{ M }_{ c })-x)||^{ 2 }$$

<div>Completion region mask에 대한 mse loss는 위 식과 같으며 $\odot$는 pixelwise multiplication을 뜻하며 $|| \cdot ||$는 Euclidean norm입니다. ${ M }_{ c }$는 completion region mask를 뜻합니다.  </div>

### Context Discriminators

![Imgur](https://i.imgur.com/PLLzQcW.png)

다음은 discriminator입니다. 크게 global과 local discriminator로 나뉘고 각각 이미지 전체 그리고 빈 부분을 집어넣습니다. 즉 global discriminator는 이미지의 전체적인 context를 구분하는 역할을 합니다. Local discriminator는 global discriminator와 구조는 거의 비슷하지만, 들어가는 이미지의 사이즈만 조금 다릅니다. 여기서는 빈 공간을 잘 복구했는지 판단하는 기능을 수행합니다.  두 discriminator를 통과하면 각각 1024차원의 벡터가 나오고 총 2048차원의 벡터로 합쳐서(텐서플로우에선 tf.concat이겠죠?) 이게 잘 복구가 되었는지 판단합니다.

전체적인 네트워크의 구조를 표로 정리하면 다음과 같습니다.

![Imgur](https://i.imgur.com/l45qMds.png)

<div>Context discriminator의 d_loss(GAN)는 아래와 같습니다. ${M}_{d}$는 random mask입니다.

$$\min _{ C }{ \max _{ D }{ E[logD(x,{ M }_{ d })+log(1-D(C(x,{ M }_{ c }),{ M }_{ c })] }  } $$

이제 위에서 구한 mse loss와 합치면 최종 loss는 아래와 같습니다.

$$\min _{ C }{ \max _{ D }{ E[L(x,{ M }_{ c })+\alpha logD(x,{ M }_{ d })+\alpha log(1-D(C(x,{ M }_{ c }),{ M }_{ c })] }  } $$


<div align="center"><a href="https://imgur.com/nF0mQUU"><img src="https://i.imgur.com/nF0mQUU.png" /></a></div>

알고리즘 순서는 정리하면 위와 같습니다. 학습은 completion network를 먼저 진행하고 일정 횟수가 지나면 그 이후 discriminator를 학습시킵니다. 이후 discriminator를 학습시킨 후, 일정 트레이닝 횟수를 넘기면 completion network를 다시 학습시키는 구조를 가지고 있습니다. 

## Results

8,097,967개의 이미지를 사용하여 학습하였으며 여기선 Places2 dataset을 사용했다고 합니다. completion network는 9만번, discriminator는 1만번의 이터레이션을 거쳤으며 4개의 K80을 장착한 기계 하나가 대략 학습에 무려 두달이나 걸렸습니다. 다양한 기법들과 복구 비교를 한 모습을 보면 굉장히 뛰어난 것을 볼 수 있습니다.

<div align="center"><a href="https://imgur.com/e8wimga"><img src="https://i.imgur.com/e8wimga.png" /></a></div>

아래 왼쪽 사진은 첫 번째 두 행은 잘 복구되었지만 마지막 두 행은 오히려 잘 안된 것도 확인할 수 있습니다. 특히 원숭이 같은 경우는 확연히 좋지 못한 성능을 보입니다. 하지만 하늘 복구작업을 보면 후처리까지 입힐 경우 거의 원본과 유사한 모습을 보입니다. 마찬가지로 object removal에서도 굉장히 좋은 성능을 보여주고 있습니다.

<div align="center"><a href="https://imgur.com/QDpA31h"><img src="https://i.imgur.com/QDpA31h.png" /></a></div>

로스를 어떻게 지정했는가에 따라 결과가 달라지는 실험 결과도 있습니다.

![Imgur](https://i.imgur.com/bco9dbC.png)

따로 얼굴 데이터셋(CelebA dataset)을 사용하여 학습시킨 모델로 나름 자연스러운 사람 얼굴 복구도 가능합니다. 중간에 조금 어색한 부분도 있긴 하지만, 어느정도 괜찮은 복구 성능을 보여주고 있습니다.

![Imgur](https://i.imgur.com/IfVcgy5.png)


더 많은 실험 결과는 [여기서](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/extra.html#comp)확인할 수 있습니다.

논문을 읽어보니 image completion 분야에서 굉장히 큰 성과가 나온 듯 합니다. 상당히 인위적이고 rule-base 적인 기존 방법들을 벗어나서 그림의 복잡한 맥락 파악 등을 매우 자연스럽게 해서 실험 결과를 처음 봤을때는 엄청 놀랐습니다. 학습에 약 2달이나 걸렸다는데 실험 설계를 해낸 대학원생들 고생이 여기까지 느껴지는 인상깊은 논문이었습니다.

> Reference
>*High-Resolution Image Inpainting using Multi-Scale Neural Patch Synthesis, [Yang et al. 2017]
>*Context Encoders: Feature Learning by Inpainting [Pathak et al., 2016]
>*https://github.com/tadax/glcic
