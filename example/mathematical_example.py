import numpy as np

# ---------------- 基础四个算例 ---------------- #
def rosenbrock(x):
    x1, x2 = x[0], x[1]
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2


def griewank(x):
    d = len(x)
    sum_term = np.sum(x ** 2 / 4000)
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
    return sum_term - prod_term + 1


def schwefel(x):
    d = len(x)
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def himmelblau(x):
    x1, x2 = x[0], x[1]
    return (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2


# ---------------- TwoDOF 算例 ---------------- #
# 测量数据
data1 = np.array([0.3860, 0.3922, 0.4157, 0.3592, 0.3615])  # λ1
data2 = np.array([2.3614, 2.5877, 2.7070, 2.3875, 2.7272])  # λ2
data3 = np.array([1.6824, 1.7110, 1.5788, 1.5872, 1.6188])  # φ12/φ11

# 噪声标准差
sig1 = 0.05 * 0.382   # case1,3
sig2 = 0.05 * 2.618   # case2
sig_phi = 0.05 * 1.618  # case3


def eigen_system(theta):
    """给定参数 θ1, θ2 计算特征值和模态比"""
    k1, k2 = theta
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    M = np.eye(2)
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(M) @ K)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    phi_ratio = eigvecs[1, 0] / eigvecs[0, 0]
    return eigvals, phi_ratio


def two_dof_case1(x):
    """Case 1: 只用第一个特征值"""
    eigvals, _ = eigen_system(x)
    return 0.5 * np.sum((eigvals[0] - data1) ** 2 / sig1 ** 2)


def two_dof_case2(x):
    """Case 2: 用两个特征值"""
    eigvals, _ = eigen_system(x)
    return (0.5 * np.sum((eigvals[0] - data1) ** 2 / sig1 ** 2) +
            0.5 * np.sum((eigvals[1] - data2) ** 2 / sig2 ** 2))


def two_dof_case3(x):
    """Case 3: 第一个特征值 + 模态比"""
    eigvals, phi_ratio = eigen_system(x)
    return (0.5 * np.sum((eigvals[0] - data1) ** 2 / sig1 ** 2) +
            0.5 * np.sum((phi_ratio - data3) ** 2 / sig_phi ** 2))
