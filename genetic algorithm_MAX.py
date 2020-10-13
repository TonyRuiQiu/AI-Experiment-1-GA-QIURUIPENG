import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
利用GA计算最大值
'''

# 参数设置
DIM_SIZE = 24  # 给一个维度分配的长度
X_BOUND = [-4, 4]  # x维度的上下界
Y_BOUND = [-4, 4]  # y维度的上下界
POP_SIZE = 200  # 种群中个体的数量
CROSSOVER_RATE = 0.8  # 交叉概率
MUTATION_RATE = 0.005  # 变异概率
GENERATION_NUMBERS = 150  # 迭代次数


# 计算适应度
def get_fitness(population):
    x, y = dna_to_decimal(population)
    calcResu = func(x, y)
    return (calcResu - np.min(calcResu)+1e-8)  # 整体移动适应度的范围（防止出现负数)；+1是为了防止出现当每个个体适应度相等时fitness.sum()为0的情况


# 将二进制DNA转化为十进制数
def dna_to_decimal(population):
    x_pop = population[:, 1::2]  # 奇数列数字的组合表示x
    y_pop = population[:, ::2]  # 偶数列数字的组合表示y
    # 转化为对应范围的十进制数—--先转化为十进制数，然后按比例缩放到0-1的范围，最后按比例缩放到上下界之间
    x = x_pop.dot(2 ** np.arange(DIM_SIZE)[::-1]) / float(2 ** DIM_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DIM_SIZE)[::-1]) / float(2 ** DIM_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y


# 选择
def select(population, fitness):
    # 选出个体对应的下标
    index = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness/(fitness.sum()))
    return population[index]


# 交叉并变异
def crossover_and_mutation(population, CROSSOVER_RATE):
    new_population = []
    for father in population:
        child = father
        if np.random.rand() < CROSSOVER_RATE:  # 以一定的概率发生交叉
            mother = population[np.random.randint(POP_SIZE)]  # 随机选择一个个体进行交叉
            cross_points = np.random.randint(0, DIM_SIZE * 2)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]
        mutation(child, MUTATION_RATE)  # 每个后代根据一定的机率发生变异
        new_population.append(child)  # 得到最终的后代

    return new_population

# 变异
def mutation(child, MUTATION_RATE):
    if np.random.rand() < MUTATION_RATE:
        muPoint = np.random.randint(0, DIM_SIZE * 2)  # 随机产生一个变异基因的位置
        child[muPoint] = child[muPoint] ^ 1  # 将变异点的二进制为反转

# 打印结果
def show_result(pop):
    fitness = get_fitness(pop)  # 获取种群各个体的适应度
    max_fitness_index = np.argmax(fitness)  # 获取最大适应度个体的下标
    x, y = dna_to_decimal(pop)
    print("最优的基因型：", pop[max_fitness_index])  # 根据下标找出最优基因
    print("(x, y): ", (x[max_fitness_index], y[max_fitness_index]))  # 输出最优x, y
    print("z: ", func(x[max_fitness_index], y[max_fitness_index]))  # 输出最优z


# 所需计算的函数
def func(x, y):
    return (((y-x)/5) * np.exp(np.sin(x+y)) * np.exp(np.sin(x-y)) + (x + y)/2)


if __name__ == "__main__":
    # 二维可视化
    # 画出函数的图像（作为背景）
    figure = plt.figure()
    axes = Axes3D(figure)
    plt.ion()
    X = np.linspace(*X_BOUND, 100)
    Y = np.linspace(*Y_BOUND, 100)
    X, Y = np.meshgrid(X, Y)
    Z = func(X, Y)
    axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    axes.set_zlim(-10, 10)
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    plt.pause(2)
    plt.show()
    
    # 遗传算法开始
    # 初始化种群--产生(POP_SIZE, DIM_SIZE*2)的矩阵，每个位子的数为0或1
    population = np.random.randint(2, size=(POP_SIZE, DIM_SIZE * 2)) 
    # 迭代
    for generation in range(GENERATION_NUMBERS):
        x, y = dna_to_decimal(population)
        # 绘制每一代种群的图像
        if 'scatter_flag' in locals():
            scatter_flag.remove()  # 去除上一代种群的图像
        scatter_flag = axes.scatter(x, y, func(x, y), c='black', marker='o');
        plt.show()
        plt.pause(0.1)
        # 计算种群中各个体的适应度
        fitness = get_fitness(population)
        # 按适应度选择种群
        population = select(population, fitness)
        # 对种群进的个体中行交叉和变异，产生新的种群
        population = np.array(crossover_and_mutation(population, CROSSOVER_RATE))
    # 打印出最后的计算结果
    show_result(population)
    plt.ioff()
    plt.show()