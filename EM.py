import random
import math
import matplotlib.pyplot as plt

#详细的代码逻辑已在文档中给出，因而本代码文件不包含过多注释

final_p = []
final_q = []
final_r = []
final_pi1 = []
final_pi2 = []
final_pi3 = []
dist_item = []
#指定单轮硬币投掷次数
M = 100
#生成折线图
def gen_picture(paras):
    length = len(final_p)
    x = list(range(1,length+1))

    plt.figure()
    ax1 = plt.subplot(1,2,1)
    plt.plot(x, final_pi1, color='blue', label='pi1-data')
    plt.plot(x, final_pi2, color='orange',label='pi2-data')
    plt.plot(x, final_pi2, color='green',label='pi3-data')
    axes1 = plt.gca()
    left, right = axes1.get_xlim()
    axes1.hlines(y=paras[0], colors='blue', xmin=left, xmax=right, linestyles='dashed')
    axes1.hlines(y=paras[1], colors='orange', xmin=left, xmax=right, linestyles='dashed')
    data =1-paras[0]-paras[1]
    axes1.hlines(y=data, colors='green', xmin=left, xmax=right, linestyles='dashed')
    plt.title('Parameters Iteration')
    # plt.rcParams['font.sans-serif']=['SimHei']
    plt.xlabel('Iteration')
    plt.ylabel('Value of Parameter')

    plt.legend()

    ax2 = plt.subplot(1,2,2)
    plt.plot(x,final_p,color='blue',label='p-data')
    plt.plot(x,final_q, color='orange', label='q-data')
    plt.plot(x,final_r,color='green', label='r-data')
    axes = plt.gca()
    left, right = axes.get_xlim()
    axes.hlines(y=paras[2], colors='blue', xmin=left, xmax=right, linestyles='dashed')
    axes.hlines(y=paras[3], colors='orange', xmin=left, xmax=right, linestyles='dashed')
    axes.hlines(y=paras[4], colors='green', xmin=left, xmax=right, linestyles='dashed')

    plt.title('Parameters Iteration')
    #plt.rcParams['font.sans-serif']=['SimHei']
    plt.xlabel('Iteration')
    plt.ylabel('Value of Parameter')
    plt.legend()
    plt.show()


# 基于给定参数生成測试的样本
def generate_dataset(pi1,pi2,p,q,r,N):
    sequences = []
    for i in range(N):
        choice_seed = random.random()
        sequence = []
        if choice_seed >=0 and choice_seed <pi1:
            for i in range(M):
                status = random.random()
                if status >=0 and status < p:
                    sequence.append(1)
                else:
                    sequence.append(0)
            sequences.append(sequence)
        if choice_seed >=pi1 and choice_seed < pi1+pi2:
            for i in range(M):
                status = random.random()
                if status >=0 and status < q:
                    sequence.append(1)
                else:
                    sequence.append(0)
            sequences.append(sequence)
        if choice_seed >= pi1+pi2 and choice_seed < 1:
            for i in range(M):
                status = random.random()
                if status >= 0 and status < r:
                    sequence.append(1)
                else:
                    sequence.append(0)
            sequences.append(sequence)
    return sequences


def e_step(pi1,pi2,p,q,r,x):
    """
    e步计算的单次迭代
    :param pi1: 下一次迭代开始的 pi1
    :param pi2: 下一次迭代开始的 pi2
    :param p:  下一次迭代开始的 p
    :param q:  下一次迭代开始的 q
    :param x: 观察数据
    :return:
    """
    pi3 = 1- pi1-pi2
    #求mu_a
    mu_a = []
    mu_b = []
    mu_c = []
    for xi in x:
        # 求mu_a
        a_item = pi1 * math.pow(p, sum(xi)) * math.pow(1 - p, M - sum(xi)) / \
        float(pi1 * math.pow(p, sum(xi)) * math.pow(1 - p, M - sum(xi)) + pi2 * math.pow(q, sum(xi)) * math.pow(1 - q, M- sum(xi))+ pi3 * math.pow(r, sum(xi)) * math.pow(1 - r, M - sum(xi)))
        mu_a.append(a_item)

        #求mu_b
        b_item = pi2 * math.pow(q, sum(xi)) * math.pow(1 - q, M - sum(xi)) / \
        float(pi1 * math.pow(p, sum(xi)) * math.pow(1 - p, M - sum(xi)) +
            pi2 * math.pow(q, sum(xi)) * math.pow(1 - q, M - sum(xi)) + pi3 * math.pow(r, sum(xi)) * math.pow(1 - r, M - sum(xi)))
        mu_b.append(b_item)

        #求mu_c
        c_item = pi3 * math.pow(r, sum(xi)) * math.pow(1 - r, M - sum(xi)) / \
        float(pi1 * math.pow(p, sum(xi)) * math.pow(1 - p, M - sum(xi)) +
            pi2 * math.pow(q, sum(xi)) * math.pow(1 - q, M - sum(xi)) + pi3 * math.pow(r, sum(xi)) * math.pow(1 - r, M - sum(xi)))
        mu_c.append(c_item)
    return mu_a,mu_b,mu_c

#更新参数
def m_step(mu_a,mu_b,mu_c,x):
    """
     m步计算
    :param u:  e步计算的后验概率mu
    :param x:  观察数据
    :return:
    """
    new_pi1 = sum(mu_a)/len(mu_a)
    new_pi2 = sum(mu_b)/len(mu_b)

    new_p = sum([mu_a[i] * sum(x[i]) for i in range(len(mu_a))]) / sum(mu_a[i] * M for i in range(len(mu_a)))
    new_q = sum([mu_b[j]* sum(x[j]) for j in range(len(mu_b))]) / sum(mu_b[j] * M for j in range(len(mu_b)))
    new_r = sum([mu_c[h] * sum(x[h]) for h in range(len(mu_c))]) / sum(mu_c[h] * M for h in range(len(mu_c)))

    return [new_pi1, new_pi2, new_p,new_q,new_r]


#em算法总入口
def em(observed_x, start_pi1,start_pi2,start_p, start_q,start_r, iter_num):
    """
    :param observed_x:  观察数据
    :param start_pi1:  下一次迭代开始的pi1
    :param start_pi2:  下一次迭代开始的pi2
    :param start_p:  下一次迭代开始的p
    :param start_q:  下一次迭代开始的q
    :param start_q:  下一次迭代开始的r
    :param iter_num:  迭代次数
    :return:
    """
    old_vector =[start_pi1,start_pi2,start_p, start_q,start_r]
    for i in range(iter_num):
        mu_a,mu_b,mu_c =e_step(start_pi1, start_pi2, start_p, start_q,start_r,observed_x)
        print ("第"+str(i+1)+"轮：","参数：", [start_pi1,start_pi2,start_p,start_q,start_r])
        final_p.append(start_p)
        final_q.append(start_q)
        final_r.append(start_r)
        final_pi1.append(start_pi1)
        final_pi2.append(start_pi2)
        final_pi3.append(1 - start_pi1 - start_pi2)
        if [start_pi1,start_pi2,start_p,start_q,start_r] == m_step(mu_a,mu_b,mu_c, observed_x):
            new_vector = [start_pi1, start_pi2, start_p, start_q, start_r]
            dist_item.append(distance(old_vector,new_vector))
            break
        else:
            [start_pi1,start_pi2,start_p,start_q,start_r]=m_step(mu_a,mu_b,mu_c,observed_x)
            new_vector =  [start_pi1,start_pi2,start_p,start_q,start_r]
            dist_item.append(distance(old_vector, new_vector))

if __name__ =="__main__":
    #生成样本序列长度
    N = 1000
    #允许的最大迭代次数
    iter = 50
    #设置用于生成样本的对照参数
    paras = [_pi1, _pi2, _p, _q, _r] = [0.3, 0.4, 0.6, 0.4, 0.7]
    # 以此为依据，生成样本集
    x = generate_dataset(_pi1, _pi2, _p, _q, _r, N)

    # 设置输入em算法进行初轮迭代的参数pi1，pi2,p,q,r参数1
    [pi1, pi2, p, q, r] = [0.2,0.4,0.4,0.3,0.6]

    # 将参数与样本数据集送入em算法
    em(x,pi1,pi2,p,q,r,iter)

    #生成迭代效果展示的可视化折线图
    gen_picture(paras)







