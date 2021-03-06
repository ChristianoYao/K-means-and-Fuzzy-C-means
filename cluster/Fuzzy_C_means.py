import numpy as np

def _initialize_U0(numbers, clusters):
    """
    初始化隶属度矩阵U0

    参数
    ----------
    numbers : int
        用户样本的个数
    clusters : int
        簇的个数

    返回
    ----------
    U0 : array, size(numbers, clusters)
        初始化后的隶属矩阵
    """
    U = np.random.rand(numbers, clusters)
    U0 = U / np.sum(U, axis=1, keepdims=True)

    return U0


def _EuclideanDistances(sample, centroids):
    """
    计算样本和聚类中心的欧几里得距离，通过公式(a-b)^2 = a^2 + b^2 - 2ab

    参数
    ----------
    sample : 2d array, size(n, p)
        用户样本矩阵
    centroids : 2d array, size(k, p)
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


def C_Means(X, k=5, m=2, error=0.001, T=1000, random_state=False):
    """
    模糊C均值聚类算法

    参数
    ----------
    X : 2d array, size (n, p)
        用户样本矩阵.  n是用户样本个数; p是特征维度
    k : int
        簇的个数
    m : float
        模糊化因子，是一个超参数
    error : float
        停止条件; np.max(np.abs(U - U_old)) < error
    T : int
        迭代次数
    random_state : int
        随机观测种子，如果提供了则观测结果可固定，否则随机

    返回
    -------
    C : 2d array, size (k, p)
        聚类中心矩阵
    U : 2d array, (n, k)
        隶属度矩阵
    """
    if random_state:
        np.random.seed(seed=random_state)
    n = X.shape[0]  # 用户个数
    p = X.shape[1]  # 特征维度
    U = _initialize_U0(n, k)

    for t in range(T):
        # 拷贝U用于停止迭代的判断
        U_old = U.copy()
        # 计算聚类中心矩阵
        C = np.dot((U ** m).T, X) / np.sum(U ** 2, axis=0, keepdims=True).T
        # 更新隶属度矩阵U
        d = _EuclideanDistances(X, C)  # 计算X与C的欧几里得距离矩阵
        d = d ** (2 / (m - 1))
        U = (d * (1 / d).sum(axis=1).reshape(n, 1))
        U = 1 / U
        if np.max(np.abs(U - U_old)) <= error:
            print('迭代次数为: ', t + 1)
            break

    return C, U
