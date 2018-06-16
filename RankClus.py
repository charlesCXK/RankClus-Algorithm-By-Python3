# coding: utf-8

import random
import heapq
import time
import multiprocessing
import numpy as np

from multiprocessing import Process
from collections import defaultdict


'''
读入data文件，构建网络(作者-会议、会议-作者、作者-作者)
'''
def BuildNet(filename):
    con_dict = defaultdict(int)
    tmp_aut_dict = defaultdict(int)
    aut_dict = defaultdict(int)

    with open('data/data.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    '''
    统计所有会议的信息以及所有作者的信息。
    '''
    for line in lines:
        line = line.strip()
        conference = line.split('$')[0]    # 会议信息
        authors = line.split('$')[1].split(';')[:-1]    # 作者(可能大于1)

        con_dict[conference] = 1
        for aut in authors:
            tmp_aut_dict[aut] += 1    # 统计作者频率
    # 取最大的 20000 个作者
    for author in heapq.nlargest(20000, tmp_aut_dict.items(), lambda x: x[1]):
        aut_dict[author[0]] = author[1]
        
    '''
    构建三种字典
    '''
    C_A, A_C, A_A = {}, {}, {}
    for key, value in con_dict.items():
        C_A[key] = defaultdict(int)    # 作者-会议字典
    for key, value in aut_dict.items():
        A_A[key] = defaultdict(int)
        A_C[key] = defaultdict(int)
        
    for line in lines:
        line = line.strip()
        conference = line.split('$')[0]    # 会议信息
        authors = line.split('$')[1].split(';')[:-1]    # 作者(可能大于1)
        
        for aut in authors:
            if aut not in aut_dict:    # 这个作者频率太低
                continue
            C_A[conference][aut] += 1
            A_C[aut][conference] += 1
            for aut2 in authors:
                if aut2 not in aut_dict:    # 这个作者频率太低
                    continue
                if aut != aut2:
                    A_A[aut][aut2] += 1
                    A_A[aut2][aut] += 1
    
    return C_A, A_C, A_A


'''
Step 0: Initialization
'''

'''
将会议初始化成K个类
'''
def init_cluster(CAnet, K):
    cluster = [[] for i in range(K)]    # 每个类一个列表
    for k in CAnet.keys():
        randnum = random.randint(0, K-1)
        cluster[randnum].append(k)
    return cluster


'''
Step 1: Ranking for each cluster
'''

'''
实现 Simple Ranking
'''
def SimpleRanking(confer_author, author_confer, cluster):
    #print("Ranking...")
    rank_confer, rank_author, rank_allconfer = defaultdict(float), defaultdict(float), defaultdict(float)
    
    # 作者排名初始化
    for author in author_confer:
        rank_author[author] = 1.0 / len(author_confer)
    # 统计会议和作者的排名
    con_totsum = 0
    aut_totsum = 0
    for con in cluster:# 只计算这个类里的会议
        rank_confer[con] = 0
        for key, value in confer_author[con].items():    # 所有指向这个会议的作者求和
            rank_confer[con] += value
            con_totsum += value

            rank_author[key] += value    # 作者在这个会议发表的论文数量
            aut_totsum += value
    for key, value in rank_confer.items():    # 对排名得分归一化
        rank_confer[key] /= float(con_totsum)
    for key, value in rank_author.items():    # 对排名得分归一化
        rank_author[key] /= aut_totsum
    
    # 最后计算这个类别条件下，所有会议的rank。将会议排名再乘上作者排名————是否要归一化？经过输出发现，第一轮的时候貌似不同类别总和差的不是很大
    golbal_sum = 0
    for confer in confer_author:
        for author,value in confer_author[confer].items():
            rank_allconfer[confer] += value * rank_author[author]
            golbal_sum +=  value * rank_author[author]
    for confer in rank_allconfer:
        rank_allconfer[confer] /= golbal_sum
        
    return rank_confer, rank_author, rank_allconfer

'''
实现 Authority Ranking
'''
def AuthorityRanking(confer_author, author_confer, author_author, cluster, alpha):
    #print("Ranking...")
    rank_confer, rank_author, rank_allconfer = defaultdict(float), defaultdict(float), defaultdict(float)
    
    # 记录类别中论文的数量以及所有作者的数量
    tot_confer = len(cluster)
    tot_author = len(author_confer)
    # 对会议、作者排名进行初始化
    for confer in cluster:
        rank_confer[confer] = 1.0 / tot_confer
    for author,value in author_confer.items():
        rank_author[author] = 1.0 / tot_author
    
    # 循环更新 conference 和 author 的rank值
    iternum = 0    # 迭代次数
    
    while iternum < 5:
        iternum += 1    # 迭代次数
        #print("iter num: {}".format(iternum))
        
        # --------先计算rank_confer--------
        confer_sum = 0
        newrank_confer = rank_confer.copy()
        for confer in cluster:
            rankc = 0
            for author in confer_author[confer]:
                rankc += confer_author[confer][author] * rank_author[author]
            newrank_confer[confer] = rankc
            confer_sum += rankc
        
        # confer排名归一化
        for confer in rank_confer:
            rank_confer[confer] = newrank_confer[confer] / confer_sum

        # --------计算author_rank--------
        save_rankc = rank_author.copy()    # 上一轮迭代的作者排名
        author_sum = 0
        
        for author in save_rankc:
            acsum = 0    # 作者与会议之间的影响
            for confer in author_confer[author]:
                acsum += author_confer[author][confer] * rank_confer[confer]
            aasum = 0    # 作者之间相互影响
            for a2 in author_author[author]:
                aasum += author_author[author][a2] * save_rankc[a2]
            rank_author[author] = alpha*acsum + (1-alpha)*aasum
            author_sum += alpha*acsum + (1-alpha)*aasum
        # author排名归一化
        for author in rank_author:
            rank_author[author] /= author_sum
            

        # 最后计算这个类别条件下，所有会议的rank。
        golbal_sum = 0
        for confer in confer_author:
            for author in confer_author[confer]:
                rank_allconfer[confer] += confer_author[confer][author] * rank_author[author]
                golbal_sum +=  confer_author[confer][author] * rank_author[author]
        for confer in rank_allconfer:
            rank_allconfer[confer] /= golbal_sum
        
        
    
    return rank_confer, rank_author, rank_allconfer


'''
Step 2: Get new attributes for objects and cluster
'''

'''
用EM算法估计参数
'''
def EM(confer_author, rank_confer, rank_author, cluster, K, iternum=5):
    pai = {}    # 每个会议有 K 维向量，代表属于第 i 个类别的概率
    
    '''
    用每个类别的 confe-author 链接数来初始化 P(z=k)
    '''
    pz = [0 for k in range(K)]
    tot_pz = 0
    for i in range(K):
        for confer in cluster[i]:    # 每个类别的会议
            for author in confer_author[confer]:
                pz[i] += confer_author[confer][author]
    for ele in pz:
        tot_pz += ele;
    pz = [ele/tot_pz for ele in pz]

    # 迭代更新
    for iterk in range(iternum):
        '''
        更新 P(z=k)，通过  P(z=k | Xi,Yi,theta)
        '''
        print("——calculate P(z = k) ...")
        # 计算 p(z = k|yj,xi,Θ0)
        
        pca = {}
        
        for confer in confer_author:
            pca[confer] = {}
            for author in confer_author[confer]:
                pca[confer][author] = {}
                sum_pca = 0    # 对每个类进行平均
                for k in range(K):
                    pca[confer][author][k] = rank_confer[k][confer] * rank_author[k][author] * pz[k]
                    sum_pca += pca[confer][author][k]
                for k in range(K):
                    pca[confer][author][k] /= sum_pca
                
        new_pz = [0 for k in range(K)]

        for k in range(K):
            son = 0    # 分子
            tot_pz = 0    # 用于求和归一化
            for confer in confer_author:
                for author in confer_author[confer]:
                    Pz_condition = pca[confer][author][k]
                    son += Pz_condition * confer_author[confer][author]    # 会议-作者权重 乘上 z=k 的条件概率
                    tot_pz += confer_author[confer][author]
            new_pz[k] = son / tot_pz
        
        for k in range(K):
            pz[k] = new_pz[k]
            
    '''
    计算 π(i,k)
    '''
    print("——calculate π ...")
    for confer in confer_author:
        pai[confer] = np.zeros(K)
        sum_this_confer = 0.000001
        for k in range(K):
            pai[confer][k] = rank_confer[k][confer] * pz[k]
            sum_this_confer += pai[confer][k]
        # 归一化
        pai[confer] /= sum_this_confer
    return pai, pz


'''
Step 3: Adjust each object
'''

'''
计算向量间距离 1-余弦相似度
'''
def myconsine(x, y):
    dist = 1 - np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return dist


'''
重新划分类别
'''
def adjust_class(confer_author, cluster, feature, K):
    '''
    计算每个类的聚类中心
    '''
    S = [np.zeros(K) for k in range(K)]
    

    for i in range(K):
        tot_sum = 0
        for confer in cluster[i]:
            S[i] += feature[confer]    # 这个会议的特征向量
            tot_sum += 1
        S[i] /= tot_sum    # 归一化
    '''
    根据聚类中心重新划分类别
    '''
    new_cluster = [[] for k in range(K)]
    for confer in confer_author:
        label = -1
        now_min = 99999999
        for k in range(K):
            dist = myconsine(feature[confer], S[k])    # 会议 confer 跟第 k 个类别的距离
            if dist < now_min:
                now_min = dist
                label = k
        # 将这个会议分配到一个新的类别
        new_cluster[label].append(confer)
    return new_cluster

'''
判段是否有类别是空的，若有空的，返回 False
'''
def check_cluster(cluster):
    for i in range(len(cluster)):
        if len(cluster[i]) == 0:
            return False
    return True


'''
Main Function
'''

'''
在多进程中调用的函数
'''
def poolrank(rank_confer, rank_author, k):
    print("start ranking of cluster {}".format(k))
    if RankAlgorithm == SimpleRanking:
        tmp, rank_author[k], rank_confer[k] = RankAlgorithm(CAnet, ACnet, Cluster[k])
    else:
        tmp, rank_author[k], rank_confer[k] = RankAlgorithm(CAnet, ACnet, AAnet, Cluster[k], alpha)


if __name__ == '__main__':
	K = 15    # 类别数目
	alpha = 0.95
	delta = 0.000001    # 判断是否收敛
	iter_num = 30    # 迭代次数
	now_iter = 0
	RankAlgorithm = AuthorityRanking    # AuthorityRanking

	# 多进程
	manager = multiprocessing.Manager()
	        
	'''获取几种网络'''
	print("Build the net of authors and conferences ...")
	CAnet, ACnet, AAnet = BuildNet('data/data.txt')

	'''
	开始计时
	'''
	main_start = time.time()

	'''初始化类别'''
	print("Init the cluster ...")
	Cluster = init_cluster(CAnet, K)

	rank_confer= manager.list([{} for k in range(K)])    # 加上 manager.list，可以在子进程修改全局变量
	rank_author= manager.list([{} for k in range(K)])

	'''
	开始迭代算法
	'''
	while now_iter < iter_num:
	    # 检查是否有类为空
	    if not check_cluster(Cluster):
	        print("Some cluster has been reduced to ZERO, exit ...")
	        now_iter = 0
	        Cluster = init_cluster(CAnet, K)
	    else:
	        now_iter += 1
	    # 创建进程池
	    pool = multiprocessing.Pool(processes=K)

	    print("********************第{}次迭代：********************".format(now_iter))
	    print("start ranking ...")
	        
	    process = []    # 进程列表
	    for k in range(K):
	        # 用分好的类别计算相对排名
	        pool.apply_async(poolrank, (rank_confer, rank_author, k))

	    pool.close()
	    pool.join()
	    
	    # 将数据恢复成正常(单进程)模式
	    rank_confer = list(rank_confer)
	    rank_author = list(rank_author)
	        
	    print("start EM algorithm ...")
	    pai,pz = EM(CAnet, rank_confer, rank_author, Cluster, K, iternum=5)
	    print("adjust the cluster ...")
	    Cluster = adjust_class(CAnet, Cluster, pai, K)

	'''
	结束计时
	'''
	main_end = time.time()
	print("**********************************************************************")
	print("Time of RankCLus Algorithm using AuthorityRanking is {} s".format(main_end-main_start))
	print("**********************************************************************")

	'''
	根据最后的分类再计算一次排名，得出排名最靠前的10个
	'''
	inrank_confer = [{} for k in range(K)]
	for k in range(K):
	    print("********************************************")
	    print("start ranking of cluster {}".format(k))
	    # 用分好的类别计算相对排名
	    if RankAlgorithm == SimpleRanking:
	        inrank_confer[k], rank_author[k], rank_confer[k] = RankAlgorithm(CAnet, ACnet, Cluster[k])
	    else:
	        inrank_confer[k], rank_author[k], rank_confer[k] = RankAlgorithm(CAnet, ACnet, AAnet, Cluster[k], alpha)
	 
	    # heapq堆算法，取最大的10个依次输出
	    print("Conference information ...")
	    for confer in heapq.nlargest(10, inrank_confer[k].items(), lambda x: x[1]):
	        print(str(confer[0]), end='\t')
        print('\n' * 3)
	    print("Author information ...")
	    for author in heapq.nlargest(10, rank_author[k].items(), lambda x: x[1]):
	        print(str(author[0]), end='\t')
        print('\n' * 3)