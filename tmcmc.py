"""
------------------------------------------------------------------------------------------------------------------------
BearGangLab - 过渡马尔可夫链蒙特卡洛 (T-MCMC)
------------------------------------------------------------------------------------------------------------------------
T-MCMC 主算法实现

过渡马尔可夫链蒙特卡洛是一种序贯蒙特卡洛方法，通过引入退火参数 φ ∈ [0, 1] 来实现从先验到后验的平滑过渡：
- φ = 0: 采样自先验分布
- φ = 1: 采样自后验分布
- 0 < φ < 1: 中间过渡分布

参考文献:
'J. Ching and Y.-C. Chen. Transitional Markov Chain Monte Carlo Method for Bayesian
Model Updating, Model Class Selection, and Model Averaging. Journal of Engineering
Mechanics, 133(7):816-832, 2007'

作者: BoMin Wang
日期: 20240324
------------------------------------------------------------------------------------------------------------------------
"""
import os
# 解决 OpenMP 和 MKL 相关问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # OpenMP 库冲突
os.environ['OMP_NUM_THREADS'] = '4'  # KMeans 内存泄漏（Windows + MKL）

import numpy as np
from typing import Callable, Tuple, Dict, Optional
from utils import PriorDistribution, LikelihoodFunction, JointPrior

# 可选：支持 GMM 提议分布
try:
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ========================================================================================================
# T-MCMC 内部计算函数
# ========================================================================================================

def _calculate_plausible_weights(log_likelihood: np.ndarray, delta: float) -> np.ndarray:
    """
    计算可信权重

    基于似然值和退火参数增量计算粒子权重：
    w_i ∝ exp(Δφ * log L_i)

    参数:
    ----------
    log_likelihood : np.ndarray, shape (n_samples, 1)
        当前样本的对数似然值
    delta : float
        退火参数增量 (φ_new - φ_old)

    返回:
    ----------
    weights : np.ndarray, shape (n_samples, 1)
        归一化的权重
    """
    n_samples = log_likelihood.shape[0]

    # 数值稳定性：减去最大值
    log_weights = delta * (log_likelihood - log_likelihood.max())
    weights = np.exp(log_weights).reshape(n_samples, 1)

    # 归一化
    weights = weights / np.sum(weights)

    return weights


def _calculate_next_annealing_parameter(log_likelihood: np.ndarray,
                                       current_phi: float,
                                       target_cov: float = 1.0) -> float:
    """
    使用二分法计算下一个退火参数

    通过二分法搜索使得权重的变异系数 (COV) 达到目标值的退火参数。
    变异系数定义为: COV = std(weights) / mean(weights)

    参数:
    ----------
    log_likelihood : np.ndarray, shape (n_samples, 1)
        对数似然值
    current_phi : float
        当前退火参数
    target_cov : float, default=1.0
        目标变异系数

    返回:
    ----------
    next_phi : float
        下一个退火参数
    """
    phi_min = current_phi
    phi_max = 2.0  # 给二分法更大的搜索空间，最终会被限制在 1.0

    next_phi = current_phi

    while phi_max - phi_min > 1e-8:
        # 二分法中点
        next_phi = 0.5 * (phi_min + phi_max)
        delta = next_phi - current_phi

        # 计算权重
        weights = _calculate_plausible_weights(log_likelihood, delta)

        # 计算变异系数
        cov = np.std(weights) / np.mean(weights)

        # 调整搜索区间
        if cov > target_cov:
            phi_max = next_phi
        else:
            phi_min = next_phi

    # 确保不超过 1.0
    return min(next_phi, 1.0)


def _resample_particles(samples: np.ndarray,
                        weights: np.ndarray,
                        n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于权重的多项式重采样

    使用多项式分布对粒子进行重采样，避免粒子退化。

    参数:
    ----------
    samples : np.ndarray, shape (n_samples, n_dim)
        当前样本
    weights : np.ndarray, shape (n_samples, 1)
        归一化权重
    n_samples : int
        重采样数量

    返回:
    ----------
    resampled_samples : np.ndarray, shape (n_samples, n_dim)
        重采样后的样本
    resampled_weights : np.ndarray, shape (n_samples, 1)
        重采样后的权重
    """
    weights_flat = weights.flatten()

    # 多项式重采样
    indices = np.random.choice(len(samples), size=n_samples, replace=True, p=weights_flat)

    resampled_samples = samples[indices]
    resampled_weights = weights[indices]

    return resampled_samples, resampled_weights


def _calculate_weighted_covariance(samples: np.ndarray,
                                   weights: np.ndarray,
                                   beta: float) -> np.ndarray:
    """
    计算加权协方差矩阵

    计算样本的加权协方差矩阵，并乘以缩放因子 β²。
    用于构造 Metropolis-Hastings 算法的提议分布。

    参数:
    ----------
    samples : np.ndarray, shape (n_samples, n_dim)
        样本
    weights : np.ndarray, shape (n_samples, 1)
        权重
    beta : float
        提议分布缩放因子

    返回:
    ----------
    covariance : np.ndarray, shape (n_dim, n_dim)
        加权协方差矩阵
    """
    # 使用 numpy 的加权协方差函数
    weights_flat = weights.flatten()
    covariance = np.cov(samples.T, aweights=weights_flat, ddof=0)

    # 应用缩放因子
    covariance = (beta ** 2) * covariance

    # 确保协方差矩阵是二维的（处理一维情况）
    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)

    # 确保正定性：添加小的对角项
    n_dim = covariance.shape[0]
    covariance += np.eye(n_dim) * 1e-8

    return covariance


def _calculate_effective_sample_size(weights: np.ndarray) -> float:
    """
    计算有效样本大小 (ESS)

    ESS 衡量重采样后样本的有效性：
    ESS = 1 / Σ(w_i²)

    参数:
    ----------
    weights : np.ndarray, shape (n_samples, 1)
        归一化权重

    返回:
    ----------
    ess : float
        有效样本大小
    """
    return 1.0 / np.sum(weights ** 2)


def _adaptive_beta(acceptance_rate: float) -> float:
    """
    基于接受率自适应调整 beta

    根据 MH 算法的接受率动态调整提议分布的缩放因子。
    公式: β = 1/9 + (8/9) * acceptance_rate

    参数:
    ----------
    acceptance_rate : float
        Metropolis-Hastings 接受率

    返回:
    ----------
    beta : float
        新的缩放因子
    """
    return 1.0 / 9.0 + (8.0 / 9.0) * acceptance_rate


def _fit_gmm_proposal(samples: np.ndarray, max_components: int = 5) -> Optional[object]:
    """
    使用 BIC 准则拟合 GMM 提议分布

    参数:
    ----------
    samples : np.ndarray
        当前样本
    max_components : int
        最大高斯分量数

    返回:
    ----------
    gmm : GaussianMixture or None
        拟合的 GMM 模型，如果 sklearn 不可用则返回 None
    """
    if not SKLEARN_AVAILABLE:
        return None

    best_gmm = None
    best_bic = np.inf

    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(samples)
        bic = gmm.bic(samples)

        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    return best_gmm


def _metropolis_hastings_step(current_samples: np.ndarray,
                              log_likelihood_func: Callable,
                              log_prior_func: Callable,
                              phi: float,
                              covariance: np.ndarray,
                              n_steps: int,
                              proposal_type: str = 'gaussian',
                              max_gmm_components: int = 5) -> Tuple[np.ndarray, float]:
    """
    Metropolis-Hastings 扰动步骤

    对每个粒子执行 MH 采样以增加样本多样性。

    参数:
    ----------
    current_samples : np.ndarray, shape (n_samples, n_dim)
        当前样本位置
    log_likelihood_func : callable
        对数似然函数
    log_prior_func : callable
        对数先验函数
    phi : float
        当前退火参数
    covariance : np.ndarray, shape (n_dim, n_dim)
        提议分布协方差矩阵
    n_steps : int
        MH 步数
    proposal_type : str, default='gaussian'
        提议分布类型: 'gaussian' 或 'gmm'
    max_gmm_components : int, default=5
        GMM 最大分量数

    返回:
    ----------
    samples : np.ndarray, shape (n_samples, n_dim)
        扰动后的样本
    acceptance_rate : float
        接受率
    """
    n_samples, n_dim = current_samples.shape

    # 计算当前样本的对数后验
    log_likelihood_current = log_likelihood_func(current_samples)
    log_prior_current = log_prior_func(current_samples)
    log_posterior_current = log_prior_current + phi * log_likelihood_current

    # 如果使用 GMM 提议，则拟合一次
    gmm = None
    if proposal_type == 'gmm':
        gmm = _fit_gmm_proposal(current_samples, max_gmm_components)
        if gmm is None:
            proposal_type = 'gaussian'  # 降级为高斯提议

    # 接受计数
    n_accepts = 0
    samples = current_samples.copy()

    for step in range(n_steps):
        # 生成提议
        if proposal_type == 'gmm' and gmm is not None:
            proposals, _ = gmm.sample(n_samples)
        else:
            # 高斯提议
            proposals = np.zeros_like(samples)
            for i in range(n_samples):
                proposals[i] = np.random.multivariate_normal(samples[i], covariance)

        # 计算提议的对数后验
        log_likelihood_proposal = log_likelihood_func(proposals)
        log_prior_proposal = log_prior_func(proposals)
        log_posterior_proposal = log_prior_proposal + phi * log_likelihood_proposal

        # 计算接受概率（对数空间）
        log_alpha = log_posterior_proposal - log_posterior_current

        # 接受/拒绝
        log_u = np.log(np.random.uniform(size=(n_samples, 1)))
        accept_mask = (log_alpha > log_u).flatten()

        # 更新接受的样本
        samples[accept_mask] = proposals[accept_mask]
        log_posterior_current[accept_mask] = log_posterior_proposal[accept_mask]

        n_accepts += np.sum(accept_mask)

    acceptance_rate = n_accepts / (n_steps * n_samples)
    return samples, acceptance_rate


# ========================================================================================================
# T-MCMC 主类
# ========================================================================================================

class TransitionalMCMC:
    """
    过渡马尔可夫链蒙特卡洛采样器

    实现 T-MCMC 算法进行贝叶斯推断，通过一系列中间分布从先验平滑过渡到后验。
    """

    def __init__(self,
                 initial_beta: float = 0.2,
                 target_cov: float = 1.0,
                 adapt_beta: bool = True,
                 proposal_type: str = 'gaussian',
                 verbose: bool = False):
        """
        初始化 T-MCMC 采样器

        参数:
        ----------
        initial_beta : float, default=0.2
            提议分布协方差的初始缩放因子
        target_cov : float, default=1.0
            退火参数选择的目标变异系数
        adapt_beta : bool, default=True
            是否根据接受率自适应调整 beta
        proposal_type : str, default='gaussian'
            提议分布类型: 'gaussian' 或 'gmm'
        verbose : bool, default=False
            是否打印详细信息（默认只显示进度条）
        """
        # 采样器参数
        self.initial_beta = initial_beta
        self.target_cov = target_cov
        self.adapt_beta = adapt_beta
        self.proposal_type = proposal_type
        self.verbose = verbose

        # 内部状态
        self.phi = 0.0  # 退火参数
        self.beta = initial_beta
        self.n_samples = None
        self.n_dim = None
        self.n_stages = 0

        # 似然和先验函数
        self._log_likelihood_func = None
        self._log_prior_func = None

        # 采样历史
        self.samples_history = None
        self.phi_history = [0.0]
        self.beta_history = [initial_beta]
        self.acceptance_rate_history = []
        self.ess_history = []

    def _log_likelihood(self, theta: np.ndarray) -> np.ndarray:
        """内部：计算对数似然并调整形状"""
        log_lik = self._log_likelihood_func(theta)
        if log_lik.ndim == 1:
            log_lik = log_lik.reshape(-1, 1)
        return log_lik

    def _log_prior(self, theta: np.ndarray) -> np.ndarray:
        """内部：计算对数先验并调整形状"""
        log_prior = self._log_prior_func(theta)
        if log_prior.ndim == 1:
            log_prior = log_prior.reshape(-1, 1)
        return log_prior

    def sample(self,
               likelihood: LikelihoodFunction,
               prior: PriorDistribution or JointPrior,
               init_samples: np.ndarray,
               n_mh_steps: int = 20) -> np.ndarray:
        """
        运行 T-MCMC 采样算法

        参数:
        ----------
        likelihood : LikelihoodFunction
            似然函数对象
        prior : PriorDistribution or JointPrior
            先验分布对象
        init_samples : np.ndarray, shape (n_samples, n_dim)
            初始样本（通常从先验采样）
        n_mh_steps : int, default=20
            每个阶段的 MH 步数

        返回:
        ----------
        samples : np.ndarray, shape (n_samples, n_dim)
            后验样本
        """
        # 设置
        self._log_likelihood_func = likelihood.log_likelihood
        self._log_prior_func = prior.log_pdf
        self.n_samples, self.n_dim = init_samples.shape

        # 初始化样本历史
        self.samples_history = init_samples[np.newaxis, :, :]
        current_samples = init_samples.copy()

        # 显示初始信息（简化版）
        if not self.verbose:
            print(f"T-MCMC 采样: {self.n_samples} 条链 × {self.n_dim} 维参数")
            print(f"提议类型: {self.proposal_type} | 初始 β: {self.initial_beta} | 目标 COV: {self.target_cov}")

        # T-MCMC 主循环
        while self.phi < 1.0 - 1e-6:
            self.n_stages += 1

            # 1. 计算对数似然
            log_likelihood = self._log_likelihood(current_samples)

            # 2. 计算下一个退火参数
            next_phi = _calculate_next_annealing_parameter(
                log_likelihood=log_likelihood,
                current_phi=self.phi,
                target_cov=self.target_cov
            )

            delta_phi = next_phi - self.phi
            self.phi = next_phi

            # 3. 计算权重
            weights = _calculate_plausible_weights(
                log_likelihood=log_likelihood,
                delta=delta_phi
            )

            # 4. 计算有效样本大小
            ess = _calculate_effective_sample_size(weights)
            self.ess_history.append(ess)

            # 5. 重采样
            current_samples, weights = _resample_particles(
                samples=current_samples,
                weights=weights,
                n_samples=self.n_samples
            )

            # 6. 计算提议分布协方差
            covariance = _calculate_weighted_covariance(
                samples=current_samples,
                weights=weights,
                beta=self.beta
            )

            # 7. Metropolis-Hastings 扰动
            current_samples, acceptance_rate = _metropolis_hastings_step(
                current_samples=current_samples,
                log_likelihood_func=self._log_likelihood,
                log_prior_func=self._log_prior,
                phi=self.phi,
                covariance=covariance,
                n_steps=n_mh_steps,
                proposal_type=self.proposal_type
            )

            self.acceptance_rate_history.append(acceptance_rate)

            # 8. 自适应调整 beta
            if self.adapt_beta and self.phi < 1.0 - 1e-6:
                self.beta = _adaptive_beta(acceptance_rate)

            # 9. 保存历史
            self.phi_history.append(self.phi)
            self.beta_history.append(self.beta)
            self.samples_history = np.concatenate(
                [self.samples_history, current_samples[np.newaxis, :, :]], axis=0
            )

            # 10. 显示进度（简洁版）
            if not self.verbose:
                # 只显示退火参数进度
                progress_bar_length = 50
                progress = int(self.phi * progress_bar_length)
                bar = '█' * progress + '░' * (progress_bar_length - progress)
                print(f"\r阶段 {self.n_stages:3d} | φ: [{bar}] {self.phi * 100:5.1f}% | "
                      f"ESS: {ess:4.0f} | 接受率: {acceptance_rate:5.1%}",
                      end='', flush=True)
            else:
                # 详细模式
                print(f"\n阶段 {self.n_stages}")
                print(f"  φ = {self.phi:.6f} (Δφ = {delta_phi:.6f})")
                print(f"  ESS = {ess:.1f} / {self.n_samples} ({ess / self.n_samples * 100:.1f}%)")
                print(f"  接受率 = {acceptance_rate:.2%}")
                print(f"  β = {self.beta:.4f}")

        # 采样完成
        if not self.verbose:
            print()  # 换行
            print(f"✓ 完成! 总阶段数: {self.n_stages} | 平均接受率: {np.mean(self.acceptance_rate_history):.1%}")
        else:
            print(f"\nT-MCMC 采样完成")
            print(f"总阶段数: {self.n_stages}")
            print(f"平均接受率: {np.mean(self.acceptance_rate_history):.2%}")
            print(f"平均 ESS: {np.mean(self.ess_history):.1f}")

        return self.get_posterior_samples()

    def get_posterior_samples(self) -> np.ndarray:
        """
        获取后验样本（最后阶段的样本）

        返回:
        ----------
        samples : np.ndarray, shape (n_samples, n_dim)
            后验样本
        """
        if self.samples_history is None:
            raise ValueError("尚未进行采样，请先调用 sample() 方法")
        return self.samples_history[-1, :, :]

    def get_all_samples(self) -> np.ndarray:
        """
        获取所有阶段的样本

        返回:
        ----------
        samples : np.ndarray, shape (n_stages+1, n_samples, n_dim)
            所有阶段的样本
        """
        if self.samples_history is None:
            raise ValueError("尚未进行采样，请先调用 sample() 方法")
        return self.samples_history

    def get_history(self) -> Dict:
        """
        获取采样历史

        返回:
        ----------
        history : dict
            包含退火参数、接受率、beta 值和 ESS 的历史记录
        """
        return {
            'phi': np.array(self.phi_history),
            'beta': np.array(self.beta_history),
            'acceptance_rate': np.array(self.acceptance_rate_history),
            'ess': np.array(self.ess_history),
            'n_stages': self.n_stages
        }

    def __repr__(self):
        """返回采样器的字符串表示"""
        if self.n_stages == 0:
            return f"TransitionalMCMC(尚未采样)"
        else:
            return (f"TransitionalMCMC(阶段数={self.n_stages}, "
                   f"样本数={self.n_samples}, "
                   f"维度={self.n_dim})")
