"""
------------------------------------------------------------------------------------------------------------------------
BearGangLab - 过渡马尔可夫链蒙特卡洛 (T-MCMC)
------------------------------------------------------------------------------------------------------------------------
先验分布和似然函数定义

参考文献:
'J. Ching and Y.-C. Chen. Transitional Markov Chain Monte Carlo Method for Bayesian
Model Updating, Model Class Selection, and Model Averaging. Journal of Engineering
Mechanics, 133(7):816-832, 2007'

作者: BoMin Wang
日期: 20240324
------------------------------------------------------------------------------------------------------------------------
"""
import numpy as np
from scipy import stats


class PriorDistribution:
    """
    先验分布类

    支持的分布类型:
    - 'normal': 正态分布
    - 'uniform': 均匀分布
    - 'lognormal': 对数正态分布
    - 'gamma': 伽马分布
    - 'beta': Beta 分布
    - 'exponential': 指数分布
    """

    def __init__(self, distribution_type: str, **params):
        """
        初始化先验分布

        参数:
        ----------
        distribution_type : str
            分布类型，可选 ['normal', 'uniform', 'lognormal', 'gamma', 'beta', 'exponential']
        **params : dict
            分布参数
            - normal: mean, std
            - uniform: low, high
            - lognormal: mean, std (对数空间的均值和标准差)
            - gamma: shape, scale
            - beta: alpha, beta
            - exponential: scale
        """
        self.distribution_type = distribution_type
        self.params = params

        # 根据分布类型创建 scipy.stats 分布对象
        if distribution_type == 'normal':
            mean = params.get('mean', 0.0)
            std = params.get('std', 1.0)
            self.dist = stats.norm(loc=mean, scale=std)

        elif distribution_type == 'uniform':
            low = params.get('low', 0.0)
            high = params.get('high', 1.0)
            self.dist = stats.uniform(loc=low, scale=high-low)

        elif distribution_type == 'lognormal':
            mean = params.get('mean', 0.0)
            std = params.get('std', 1.0)
            self.dist = stats.lognorm(s=std, scale=np.exp(mean))

        elif distribution_type == 'gamma':
            shape = params.get('shape', 2.0)
            scale = params.get('scale', 1.0)
            self.dist = stats.gamma(a=shape, scale=scale)

        elif distribution_type == 'beta':
            alpha = params.get('alpha', 2.0)
            beta = params.get('beta', 2.0)
            self.dist = stats.beta(a=alpha, b=beta)

        elif distribution_type == 'exponential':
            scale = params.get('scale', 1.0)
            self.dist = stats.expon(scale=scale)

        else:
            raise ValueError(f"不支持的分布类型: {distribution_type}")

    def sample(self, size):
        """
        从先验分布中生成随机样本

        参数:
        ----------
        size : int or tuple
            样本形状

        返回:
        ----------
        samples : np.ndarray
            随机样本
        """
        return self.dist.rvs(size=size)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        计算概率密度函数值

        参数:
        ----------
        x : np.ndarray
            输入点

        返回:
        ----------
        pdf_values : np.ndarray
            概率密度值
        """
        return self.dist.pdf(x)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        计算对数概率密度函数值

        参数:
        ----------
        x : np.ndarray
            输入点

        返回:
        ----------
        log_pdf_values : np.ndarray
            对数概率密度值
        """
        return self.dist.logpdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        计算累积分布函数值

        参数:
        ----------
        x : np.ndarray
            输入点

        返回:
        ----------
        cdf_values : np.ndarray
            累积分布值
        """
        return self.dist.cdf(x)

    def __repr__(self):
        """返回分布的字符串表示"""
        return f"PriorDistribution(type='{self.distribution_type}', params={self.params})"


class JointPrior:
    """
    多维联合先验分布类

    用于处理多个独立的先验分布组成的联合分布
    """

    def __init__(self, priors: list):
        """
        初始化联合先验分布

        参数:
        ----------
        priors : list of PriorDistribution
            各个维度的先验分布列表
        """
        self.priors = priors
        self.n_dim = len(priors)

    def sample(self, size: int) -> np.ndarray:
        """
        从联合先验分布中生成样本

        参数:
        ----------
        size : int
            样本数量

        返回:
        ----------
        samples : np.ndarray, shape (size, n_dim)
            联合样本
        """
        samples = np.zeros((size, self.n_dim))
        for i, prior in enumerate(self.priors):
            samples[:, i] = prior.sample(size)
        return samples

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        计算联合对数概率密度（假设各维度独立）

        参数:
        ----------
        x : np.ndarray, shape (n_samples, n_dim)
            输入样本

        返回:
        ----------
        log_pdf : np.ndarray, shape (n_samples, 1)
            对数概率密度值
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        log_pdf = np.zeros((x.shape[0], 1))
        for i, prior in enumerate(self.priors):
            log_pdf += prior.log_pdf(x[:, i]).reshape(-1, 1)

        return log_pdf

    def __repr__(self):
        """返回联合分布的字符串表示"""
        prior_strs = [f"  Dim {i}: {prior}" for i, prior in enumerate(self.priors)]
        return f"JointPrior({self.n_dim} dimensions):\n" + "\n".join(prior_strs)


class LikelihoodFunction:
    """
    似然函数类

    用于封装用户定义的似然函数
    """

    def __init__(self, func: callable):
        """
        初始化似然函数

        参数:
        ----------
        func : callable
            似然函数，接受参数 theta (np.ndarray) 并返回似然值
            注意：应该返回似然值本身，而不是对数似然
        """
        self.func = func

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        """
        计算似然值

        参数:
        ----------
        theta : np.ndarray
            参数值

        返回:
        ----------
        likelihood : np.ndarray
            似然值
        """
        return self.func(theta)

    def log_likelihood(self, theta: np.ndarray) -> np.ndarray:
        """
        计算对数似然值

        参数:
        ----------
        theta : np.ndarray
            参数值

        返回:
        ----------
        log_likelihood : np.ndarray
            对数似然值
        """
        likelihood = self.func(theta)
        # 避免 log(0)
        return np.log(likelihood + 1e-300)

    def __repr__(self):
        """返回似然函数的字符串表示"""
        return f"LikelihoodFunction(func={self.func.__name__})"


class GaussianLikelihood(LikelihoodFunction):
    """
    高斯似然函数

    假设观测数据服从高斯分布：y ~ N(model(theta), sigma^2)
    """

    def __init__(self, observed_data: np.ndarray, model_func: callable, sigma: float):
        """
        初始化高斯似然函数

        参数:
        ----------
        observed_data : np.ndarray
            观测数据
        model_func : callable
            模型函数，接受参数 theta 并返回模型预测值
        sigma : float
            观测噪声标准差
        """
        self.observed_data = observed_data
        self.model_func = model_func
        self.sigma = sigma
        self.n_obs = len(observed_data) if observed_data.ndim == 1 else observed_data.shape[0]

        # 创建似然函数
        def likelihood_func(theta):
            """计算高斯似然"""
            if theta.ndim == 1:
                theta = theta.reshape(1, -1)

            n_samples = theta.shape[0]
            likelihoods = np.zeros(n_samples)

            for i in range(n_samples):
                prediction = self.model_func(theta[i])
                residual = self.observed_data - prediction

                # 对数似然（为了数值稳定性）
                log_lik = -0.5 * np.sum(residual**2) / (self.sigma**2)
                log_lik -= self.n_obs * np.log(self.sigma * np.sqrt(2 * np.pi))

                likelihoods[i] = np.exp(log_lik)

            # 归一化以提高数值稳定性
            return likelihoods / (likelihoods.max() + 1e-300)

        super().__init__(likelihood_func)

    def __repr__(self):
        """返回高斯似然的字符串表示"""
        return f"GaussianLikelihood(n_obs={self.n_obs}, sigma={self.sigma})"
