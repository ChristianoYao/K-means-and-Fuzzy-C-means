import numpy as np

def _EuclideanDistances(sample, centroids):
    """
    计算样本和聚类中心的欧几里得距离，通过公式(a-b)^2 = a^2 + b^2 - 2ab

    参数
    ----------
    sample : array, size(n, p)
        用户样本矩阵
    centroids : array, size(k, p)
        聚类中心矩阵(如果输入的是1d array在输入前用reshape转换成二维数组)

    返回
    ----------
    ED : array, size(n, k)
        样本与聚类中心的距离矩阵
    """
    if centroids.ndim == 1:
        centroids = centroids.reshape(1, len(centroids))
    vecProd = np.dot(sample, centroids.T)
    SqA = sample ** 2
    sumSqA = np.sum(SqA, axis=1, keepdims=True)
    sumSqAEx = np.tile(sumSqA, (1, vecProd.shape[1]))  # 将(n, 1)的向量sumSqA拓展成(n, k)的矩阵

    SqB = centroids ** 2
    sumSqB = np.sum(SqB, axis=1).reshape(1, -1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))  # 将(1, k)的向量sumSqB扩展成(n, k)的矩阵
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)

    return ED


def _initialize_center(X, k, init_state='K_Means++'):
    """
    初始化聚类中心矩阵

    参数
    ----------
    X : array, size(n, p)
        用户样本矩阵
    k : int
        簇个数
    init_state : string ['K_Means++', 'random']
        选择初始化聚类中心矩阵的方式

    返回
    ----------
    C : 2d array, size(k, p)
        初始化后的聚类中心矩阵
    label : 1d array, size(n, )
        初始化矩阵下的样本标签
    D: 2d array, size(n, k)
        初始化后的样本点与聚类中心的欧几里得距离矩阵
    """
    n, p = X.shape

    if init_state == 'random':
        index = random.sample([i for i in range(n)], k)
        C = X[index]
        D = _EuclideanDistances(X, C)
        label = np.argmin(D, axis=1)

    elif init_state == 'K_Means++':
        # 初始化聚类中心矩阵
        C = np.zeros((k, p))
        # index向量
        index_vec = [i for i in range(n)]
        # 随机选一个点作为第一个聚类中心
        index0 = np.random.randint(0, n)
        C[0] = X[index0]
        # 求出欧几里得距离矩阵
        D = _EuclideanDistances(X, C[0])
        for i in range(2, k + 1):
            d_min = np.min(D, axis=1)
            prob = d_min / np.sum(d_min)
            index = np.random.choice(index_vec, size=1, p=prob)[0]
            C[i - 1] = X[index]
            if i == k:
                D = _EuclideanDistances(X, C)
                break

            # 求出欧几里得距离矩阵
            D = _EuclideanDistances(X, C[:i])

        label = np.argmin(D, axis=1)

    return C, label, D


def K_Means(X, k=2, error=0.001, T=100, random_state=None, init_state='K_Means++'):
    """
    模糊C均值聚类算法

    参数
    ----------
    X : 2d array, size (n, p)
        用户样本矩阵.  n是用户样本个数; p是特征维度
    k : int
        簇的个数
    error : float
        停止条件; J - J_old <= error
    T : int
        迭代次数
    random_state : int
        随机观测种子，如果提供了则观测结果可固定，否则随机
    init_state : string ['K_Means++', 'random']
        选择初始化聚类中心矩阵的方式

    返回
    -------
    C : 2d array, size (k, p)
        聚类中心矩阵
    label : 1d array, size(n, )
        样本标签向量
    """
    if random_state:
        np.random.seed(seed=random_state)

    n, p = X.shape
    C, label, D = _initialize_center(X, k, init_state)
    J = np.sum(np.min(D ** 2, axis=1))

    for t in range(T):
        J_old = J
        # 重新计算簇中心
        for i in range(k):
            index = np.where(label == i)[0]
            C[i] = X[index].mean(axis=0)

        D = _EuclideanDistances(X, C)
        J = np.sum(np.min(D ** 2, axis=1))
        if np.abs(J - J_old) < error:
            print('迭代次数为：', t + 1)
            break
        label = np.argmin(D, axis=1)

    label = np.argmin(D, axis=1)

    return C, label
