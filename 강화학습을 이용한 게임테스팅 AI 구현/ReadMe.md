학사논문으로, Pysc2를 환경으로 한 강화학습으로 환경의 변화에 따라 Ai가 변화의 영향을 확인하기위한 프로젝트입니다

pytorch를 사용하였고 환경으론 Pysc2의 CollectMineralShards를 사용하였습니다

![image](https://github.com/gray-spade/Portfolio/assets/52790712/5d5abb80-886b-411d-ac30-625f5143a567)

pysc2에대한 자세한정보는 https://github.com/zeuseyera/pysc2-krfmf 를 확인해주세요

강화학습의 환경에서 요구되는 정보는 3가지가 있으며 각각 관측(Observation),행동(Action),보상(Reward)입니다

이 환경에선 아래와같이 정의 되어 있습니다
![image](https://github.com/gray-spade/Portfolio/assets/52790712/239850af-373b-42c9-8438-b1761c34ea75)



Pysc2에서 제공하는Observation은 
![image](https://github.com/gray-spade/Portfolio/assets/52790712/e8b7fbf5-7d67-494c-8b7f-dba09e5a7a0a)

이렇게 구성되어 있는데 이번 프로젝트에서 필요한건 

![image](https://github.com/gray-spade/Portfolio/assets/52790712/345ae690-68f0-4339-9783-c366ceed8f08)

이 3개 뿐임으로 이3개만 따로 가져와서 학습에 사용합니다.
각각 

선택된 유닛정보

모든 유닛의 정보

마지막으로 한 행동정보

를 담고있습니다.

학습 알고리즘은 DDQN을 사용하였으며

신경망은 63-512-128-24의 형태로 구성되어있습니다.

이와같이 학습을 진행하면 아래와 같은 모습을 보여주게 됩니다.

https://github.com/gray-spade/Portfolio/assets/52790712/8683503b-7a72-4527-8081-b6b1c38138a2

학습 결과를 그래프로 나타내면

![image](https://github.com/gray-spade/Portfolio/assets/52790712/49261e93-fe22-466e-bb68-6c288aa2af70)

와 같이 보상이 점점 늘어나며 안정적이게 변화하는걸 확인할수있고

환경을 아래와 같이 변화 하였을때

![image](https://github.com/gray-spade/Portfolio/assets/52790712/84bd56b1-75dd-4d69-a702-cb835e91d88d)

아래와 같은 차이를 보이는것을 확인할수있습니다

![image](https://github.com/gray-spade/Portfolio/assets/52790712/374b74fc-be67-45f9-8d42-7c6bd1579098)


![image](https://github.com/gray-spade/Portfolio/assets/52790712/6e305423-8290-43ee-b795-882ae56a4a59)


이를통하여 변인이 미치는 영향을 직접 사람이 실험하지 않더라도 미리 테스트해볼수있다는 점을 확인이 가능합니다



