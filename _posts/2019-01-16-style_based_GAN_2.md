---
title: A Style-Based Generator Architecture for GANs - 2
description: 2018년 12월에 나온 GAN의 generator 구조 관련 논문입니다. 기존 GAN의 generator(생성기)들의 한계점을 극복하고 한단계 더 나아갈 수 있는 방향을 제시하였습니다. 실험 결과와 어떻게 GAN의 한계를 극복했는지 그 배경에 대해 알아보도록 하겠습니다.

category: Deep Learning
tags:
- GAN
- CNN
---

시작하기에 앞서 이 포스트는 Deep paper 스터디 조원인 [정지원님의 발표영상](https://youtu.be/TWzEbMrH59o)을 참고하였음을 밝힙니다.

# Style-Based Generator for GANs - experiments

이번 포스트에서는 크게 실험 결과와 적용 가능한 다양한 트릭들 그리고 왜 이렇게 현실적인, 어색하지 않은 이미지가 생성되는지 그 이유에 대해 알아보도록 하겠습니다.

## Experiments

<div align = 'center'>
<img src="https://i.imgur.com/JuPYoHR.png" title="'uncurated image'와 생성기의 구조" />
</div>

논문을 읽었을 때 가장 놀랐던 부분입니다. "Uncurated image"라고 쓰여있는데, 이는 잘 나온 결과만 선별적으로 골라서 제시한 것이 아니라 딱히 검수를 거치지 않고 생성된 이미지를 뜻합니다. 하나도 어색하지 않게 이미지가 잘 생성된 것을 확인할 수 있습니다.

### Quality of generated images

아래 표는 실험  결과입니다. GAN 성능 지표 중 하나인 FID(Frechet inception distance)를 사용했습니다. A는 우리가 알고 있는 PG-GAN이고 B는 PG-GAN에 추가적인 방법을 사용하여 성능을 향상시킨 모델입니다. C는 mapping network와 AdaIN을 사용한 결과이며 상당히 성능이 향상된 것을 확인할 수 있습니다. D는 기존 방식처럼 $z$를 집어넣는게 아닌 학습된 고정 텐서를 사용하였습니다. E는 여기에 노이즈까지 더한 결과, F는 하나의 생성되는 이미지에 스타일을 다양하게 집어 넣어 regularization 효과를 추가한 결과입니다.

<div align='center'>
<img src="https://i.imgur.com/2sUmnKM.png" title="실험 결과" width ="500"/>
</div>

데이터셋은 PG-GAN에서 사용했던 CelebA-HQ 그리고 Nvidia에서 자체적으로 만든 FFHQ를 사용했습니다. 다만 손실 함수의 경우 Celeba-HQ는 WGAN-GP를 사용하였고 FFGQ는 non-saturating loss with R1 regularization을 사용했다고 합니다. 논문에서는 "생성기의 구조"에 집중한다며 따로 새로운 손실 함수에 대한 제안은 하지 않았다고 언급하고 있습니다.

#### Truncation trick in $W$

실제로 학습 데이터의 분포를 고려하면, density가 낮은 부분의 경우 학습 후 표현이 잘 되질 않습니다. 즉, 생성기가 제대로 학습을 하지 못합니다. 이러한 부분을 방지하기 위하여 쓰는 방법이 truncation trick입니다. 이는 학습 중에 적용하는게 아닌 학습이 완료된 네트워크의 input을 제어하는 방법입니다. 보통 이전 GAN 모델에선 truncation trick은 $z$에 바로 적용하였습니다. 

$$\bar { w } ={ { \mathbb{E}} }_{ z\sim  P(z) }\left[ f(z) \right] $$

$$ { w }^{ ' }=\overline { w } +\psi \left( w-\bar { w }  \right) $$

여기서는 mapping network의 결과물인 $w$의 space $W$에 적용하는 방법을 사용하였습니다. 먼저 위 식의 $\bar { w }$를 구하고 ${ w }^{ ' }$로 truncate를 해줍니다. $\psi$는 항상 1보다 작은 값입니다. $\psi$를 1로 설정하면 truncation trick을 사용하지 않은 경우와 같습니다. 동영상에도 나오는 부분이지만, $\psi$를 0으로 지정하면 모두 다 같은 얼굴이 나오게 됩니다. 이 얼굴이 $\bar{w}$ 즉, 평균의 얼굴이라고 할 수 있습니다.

포스트 시작 부분에 있는 생성된 이미지들은 truncation trick을 적용한 결과물이고, 실험 결과 테이블에 나와 있는 FID 값은 truncation trick을 적용하지 않은 결과물입니다.


### Style mixing

<div>
여기에 추가적으로 mixing regularization 방법을 적용했다고 합니다. 원래대로라면 하나의 latent code $z$만 가지고 학습을 해야하는데 네트워크 정규화 효과를 노리기 위하여 여러 개의 $z$를 사용하였습니다. 예를 들어 2개를 사용한다고 하면 ${z}_{1}$과 ${z}_{2}$를 mapping network에 통과시켜 ${w}_{1}$과 ${w}_{2}$를 만듭니다. 이후에 
${w}_{1}$을 전체 layer에 적용을 시킨 뒤, ${w}_{2}$는 랜덤하게 layer를 골라 적용을 시킵니다. 이런 방법을 사용하면 다양한 스타일이 섞여 네트워크의 regularization 효과를 얻을 수 있습니다. 실험 결과는 아래 표와 같습니다.
</div>


<div align='center'>
<img src="https://i.imgur.com/uKW7hZ7.png" title="style mixing" width="500" />
</div>


백문이 불여일견이라고 아래 영상을 보시면 어떤 식으로 스타일이 적용 되는지 볼 수 있습니다. 영상에서는 각 레이어에서 어떤 레이어에 어떤 스타일을 입히느냐에 따라 생성되는 이미지가 달라지는 것을 확인할 수 있습니다. 영상에서도 확인할 수 있지만, 재미있게도 각 레이어마다 표현하는 스타일이 전부 다릅니다.

- 4x4부터 8x8까지의 레이어는 포즈, 얼굴의 전체적인 모양, 안경, 머리색 등이 바뀝니다.
- 16x16부터 32x32까지는 머리 스타일, 눈을 떴는가 감았는가 등이 바뀝니다.
- 64x64부터 1024x1024까지는 색의 배열과 미세 구조(microstructure)에 영향을 줍니다.



<iframe width="560" height="315" src="https://www.youtube.com/embed/bIVU8UuHPKI?start=26" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<img src="https://i.imgur.com/1iiSOKs.png" title="style mixing 과정(출처 : PR-131)" />

스타일을 섞는 방법은 위와 같습니다. 전부 다른 $z$값을 가지고 $w$를 만든 뒤 해당하는 레이어에 스타일을 입혀주면 위와 같은 결과물이 나옵니다.

### Stochastic variation

사람의 얼굴을 봤을 때 머리스타일, 안경 유무, 인종 등은 큼직큼직한 부분이라고 할 수 있습니다. 그렇다면 모공, 수염 자국, 머리카락의 흐트러진 정도 등은 어떤 식으로 바꿀 수 있을까요? 이전 포스트에서 설명했지만 convolution output에 단순히 elementwise로 노이즈 값을 더해주면 방금 언급한 세세한 부분들에서 변화가 옵니다.

아래 그림에서 왼쪽 사진을 보면, 사진을 그대로 두고 노이즈 값을 다르게 주면 머리카락이 조금씩 바뀌는 것이 보입니다(b). (c)는 100장의 사진을 그대로 두고 노이즈 값을 다르게 했을 때 표준편차의 모습입니다. 오른쪽 그림의 흑인 여자아이를 기준으로 보면, 왼쪽이 노이즈를 모든 레이어에 줬을 때고 오른쪽이 노이즈를 전혀 주지 않았을 때 입니다. 마찬가지로 위의 Nvidia 유튜브 영상에서 동적으로 변화하는 모습을 확인할 수 있습니다.

<div align='center'>
<img src="https://i.imgur.com/8EYZNBI.png" title="Effect of noise(출처 : PR-131)" />
</div>

## Disentanglement studies

"Disentanglment"의 뜻은 영어사전으로 직역하면 "얽힌 것을 푸는 것"입니다. GAN에서 말하는 disentanglement란 latent space 가 선형적인 구조를 가지게 되어서, 하나의 팩터를 움직였을 때 정해진 하나의 특성이 바뀌게 만들고자 하는 것입니다. 예를 들어 $z$의 특정한 값을 바꿨을 때 생성되는 이미지의 하나의 특성(성별, 머리카락의 길이, 바라보는 방향 등)만 영향을 주게 되면 disentanglement라고 합니다. 

근데 이전에 사용하던 방법은 고정된 분포를 따르는 latent space $Z$를 바로 학습 데이터의 분포에 끼워맞추고자 했습니다. 따라서 실제 데이터가 아래 그림의 (a)와 같은 분포를 가지고 있더라도 (b)처럼 $Z$의 분포에 맞게 억지로 끼워맞춰지는 방향으로 따라하게 됩니다. 이렇기 떄문에 제가 이전 포스트의 첫 부분에 보여드린 사람이 보기에도 이상한 그림들이 생성되는 것이죠. 

그런데 비선형 mapping funcion을 거치게 되면 굳이 $W$가 latent space처럼 고정된 분포를 따를 필요가 없습니다. 따라서 학습 데이터의 분포와 비슷하게 알아서 (c)처럼 $W$ 자체 분포가 변형되게 학습됩니다. 논문에서는 이렇게 disentanglement 정도를 학습할 수 있는 두 가지 평가 지표를 제안하고 있습니다.

<div align='center'>
<img src="https://i.imgur.com/rUHIkb2.png"  width='500'/>
</div>

### Perceptual path length

먼저 perceptual path length입니다. 이 지표의 가정은, 만약 latent space가 disentanglement 하다면 $z$의 값이 아주 약간 변화했을 때 큰 차이가 없어야 한다는 것입니다. 만약 entanglement한 경우, 약간의 변화에도 다양한 특성들이 더 많이 변화해야 합니다. 이는 다양한 피쳐들이 서로 얽혀있기 때문이라고 해석할 수 있습니다.

<div>
과정은 다음과 같습니다. 미리 학습된 VGG16 모델에 ${z}_{1}$과 ${z}_{2}$를 가지고 생성된 이미지를 넣어 임베딩을 시킵니다. 여기서 임베딩 된 피쳐들을 가지고 perceptual difference를 아래 식과 같이 구하게 됩니다. 다만 조금 더 정확한 결과를 위해 실제 측정할 때는 이미지의 얼굴 부분만 크롭해서 사용했다고 합니다.
</div>

$${ l }_{ Z }=\mathbb{E}\left[ \frac { 1 }{ { \epsilon  }^{ 2 } } d\left( G\left( slerp\left( { z }_{ 1 },{ z }_{ 2 };t \right)  \right) ,G\left( slerp\left( { z }_{ 1 },{ z }_{ 2 };t+\epsilon  \right)  \right)  \right)  \right] $$

위 식은 $Z$에서 slerp(spherical interpolation operation)을 수행합니다. 여기서 $t$는 0과 1사이의 유니폼 분포를 따릅니다. 아래 왼쪽 그림을 보시면 interpolation을 간략하게 나타내고 있습니다. 검은색 동그라미가 각각 서로 다른 $z$이고 $t$와 $1-t$만큼 interpolation을 합니다. 다만 $z$는 일반적으로 가우시안 분포를 따르는 경우가 많습니다. 따라서 slerp을 사용합니다.

$${ l }_{ W }=\mathbb{E}\left[ \frac { 1 }{ { \epsilon  }^{ 2 } } d\left( G\left( lerp\left( f({ z }_{ 1 }),f({ z }_{ 2 });t \right) ,lerp\left( f({ z }_{ 1 }),f({ z }_{ 2 });t+\epsilon  \right)  \right)  \right)  \right] $$

위 식은 $W$일 때 사용합니다. 달라지는 부분은 $W$ space에서 $slerp$이 아닌 $lerp$(linear interpolation)을 수행하는 것입니다. 이는 이미 mapping function을 거쳤기 때문에 normalized 되지 않았기 때문입니다. 
또 하나의 차이점으로는 $W$에서의 perceptual path를 구할 때는 $t$가 0 또는 1의 값으로 고정합니다. 그냥 똑같이 유니폼 분포를 따르게 한다면 아래 오른쪽 그림의 노란색 원 처럼 실제 존재하지 않는 벡터 부분이 interpolation 됩니다. 이러면 $W$에서 구한 path가 아무래도 $Z$에서 구한 path보다는 더 불리한 수치가 나오기 마련입니다. 따라서 0 또는 1의 값으로 $w$ 근처의 값만 가지고 계산하게 하였습니다. 수치는 총 10만 개의 샘플을 뽑아 산출 하였으며 $Z$와 $W$ 모두 full path, end path를 전부 계산하였습니다. 정리하면 아래와 같습니다.

- <div>Full path :  $t\sim U(0,1)$ </div>
- <div>End path : $t\in \{ 0,1\} $</div>

<div align='center'>
<img src="https://i.imgur.com/ymAEXb7.png" title="interpolation" width='500'/>
</div>

### Linear seperability

만약 제대로 latent space가 dientangle 하다면 특성을 나누는 정확한 방향 벡터를 찾아낼 수 있다는 개념에서 출발한 평가 지표입니다. 선형적인 초평면(linear hyperplane)을 가지고 이를 나눌 수 있다면 이는 latent space가 얽혀 있지 않다고 할 수 있습니다. 이를 위해 기존 celebA-HQ 데이터 셋의 사진 특성에 대한 40개의 변수를 이진 분류할 수 있는 40개의 auxiliary classification network를 학습했다고 합니다.

이후 생성기로 생성한 20만장의 이미지를 auxiliary network에 집어넣어 분류를 시킵니다. 다만 완전히 다 맞출 수 없기 때문에 신뢰도가 높은 상위 50%의 이미지만 가지고 갑니다.

이후 linear SVM을 사용하여 분류를 시킵니다. 입력 변수는 생성된 이미지의 원래 $z$와 $w$값입니다. Linear SVM으로 분류가 잘 된다면 이는 linear hyperplane만으로도 특성을 잘 잡아낼 수 있다는 말과 같습니다. 측정 지표는 conditional cross entropy $H(Y|X)$를 사용했으며 $X$는 SVM으로 예측한 레이블, $Y$는 auxiliary network로 분류한 레이블 입니다. 최종 평가 지표는 아래 식과 같습니다.

$$ exp\left( \sum _{ i }^{  }{ H\left( { Y }_{ i }|{ X }_{ i } \right)  }  \right) $$

<div align='center'>
<img src="https://i.imgur.com/m4Um4Z2.png" title="Seperability score" />
</div>

실험 결과는 위 표와 같습니다. 확실히 기존 생성기들보다 mapping network를 거치게 만든 생성기가 훨씬 더 disentangle 하다는 것을 알 수 있습니다.


## 그 외 다른 결과들

<div align='center'>
<img src="https://i.imgur.com/1GrW9za.png" title="침대와 자동차 이미지 생성" width="600"/>
</div>

다른 결과물들도 상당히 깔금하게 잘 나옵니다. 영상에서 interpolation 과정을 보여주는데 확실히 이전 GAN들보다 자연스러운 것을 확인할 수 있습니다. 


## 정리하며

GAN이 생성하는 어색한 부분을 복잡하지 않은 방법으로 풀어버린 재미있는 논문이었습니다. 사실 GAN을 공부할 때마다 GAN 자체가 implicit한 방법으로 분포를 맞추다 보니 아무리 노력해도 어색한 부분이 고쳐지지 않을 것이라는 생각이 많이 들었습니다. 그러나 이번 실험 결과물들을 보니 앞으로 여기서 제안한 생성기의 구조를 베이스로 GAN 연구가 지속되지 않을까 하는 생각이 듭니다.


> Reference
> * [Karras, T., Laine, S., & Aila, T. (2018). A Style-Based Generator Architecture for Generative Adversarial Networks. arXiv preprint arXiv:1812.04948.](https://arxiv.org/pdf/1812.04948.pdf)
> * [Rani Horev's blog](https://towardsdatascience.com/explained-a-style-based-generator-architecture-for-gans-generating-and-tuning-realistic-6cb2be0f431) 
> * [PR-131 youtube (발표자 : 정지원)](https://youtu.be/TWzEbMrH59o) 


