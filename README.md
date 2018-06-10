# 2018 PKU Data Warehouse

### 一、对XML文件的预处理：HandleXML.ipynb

**包括**：

——（1）将其中的类似'&ouml ;'的字符串删掉，这是unicode遗留问题

——（2）提取出1998-2007年间的（会议、作者）信息，写入到新文件 data/data.txt。每行的格式为：会议$作者1;作者2;.....作者n;



### 二、RankCLus算法：Rankclus.ipynb

#### Step 0: Initialization.

第一步，初始化。将会议随机分配到K个类别中。K 设定为15.



***



#### Step 1: Ranking for each cluster

在每个类别中，计算作者与会议的条件排名。排名方法有**Simple Ranking**和**Authority Ranking**两种

##### **————Simple Ranking**————

计算公式如下图:

<img src="pic/1.jpg" width=5></img>

其中$\vec{r}_{X}(x)$和$\vec{r}_{Y}(y)$分别是会议和作者在这个分类情况下的排名.



##### **————Authority Ranking————**

**Authority Ranking**的原理是基于这样一个直觉：**高排名的作者倾向于向高排名的会议投稿，高排名的会议更容易吸引高排名的人**。所以$\vec{r}_{X}(x)$和$\vec{r}_{Y}(y)$其实是相互影响的。下图是会议、作者排名的计算公式(包含归一化的步骤)：

<img src="pic/2.jpg" />

同时我们也认为，作者之间的排名会相互影响，比如说，相互之间有合作（共同发表论文）的作者会提高相互之间的排名。 <img src="pic/3.jpg" />



${aplha}​$ 是自定义的一个参数，代表${作者—作者}​$和${会议-作者}​$之间影响权重。



***



##### Step 2: Estimation of the mixture model component co-efficients



***



##### Step 3: Cluster adjustment



***



##### Repeat Steps 1, 2 and 3



### 三、输出文件

##### 1、output/simple_rankclus_confer.csv, output/simple_rankclus_author.csv

用 simpleranking 排名算法得到的每个类别排名前十的会议/作者

