"""
T-MCMC Benchmark Tests - Minimal Version
Format: Setup -> Run TMCMC -> Plot Scatter
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from BayesCalibration.TMCMC.utils import PriorDistribution, JointPrior, LikelihoodFunction
from BayesCalibration.TMCMC.tmcmc import TransitionalMCMC
from BayesCalibration.TMCMC.example.mathematical_example import rosenbrock, griewank, himmelblau, schwefel, two_dof_case1, two_dof_case2, two_dof_case3

os.makedirs('results', exist_ok=True)

# Common settings
n_chains = 10000
n_mh_steps = 10


# ========================================================================================================
# Benchmark 1: Rosenbrock
# ========================================================================================================
print("\n" + "="*100)
print("Benchmark 1: Rosenbrock Function")
print("="*100)

# Setup
bounds = (-5.0, 5.0)

prior_1 = PriorDistribution('uniform', low=bounds[0], high=bounds[1])
prior_2 = PriorDistribution('uniform', low=bounds[0], high=bounds[1])
prior = JointPrior([prior_1, prior_2])

def likelihood_func(theta):
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    n_samples = theta.shape[0]
    likelihoods = np.zeros(n_samples)
    temperature = 0.1
    for i in range(n_samples):
        f_val = rosenbrock(theta[i])
        likelihoods[i] = np.exp(-f_val / temperature)
    return likelihoods / (likelihoods.max() + 1e-300)

likelihood = LikelihoodFunction(likelihood_func)
init_samples = prior.sample(n_chains)

# Run TMCMC
print("\nRunning T-MCMC...")
tmcmc = TransitionalMCMC(initial_beta=0.2, target_cov=1.0, adapt_beta=True, proposal_type='gaussian', verbose=False)
posterior = tmcmc.sample(likelihood=likelihood, prior=prior, init_samples=init_samples, n_mh_steps=n_mh_steps)

# Plot final posterior (beautified)
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(posterior[:, 0], posterior[:, 1], c='#3498db', s=3, alpha=0.5, edgecolors='none', label='Posterior Samples')
ax.set_xlabel('θ₁', fontsize=16, fontweight='bold')
ax.set_ylabel('θ₂', fontsize=16, fontweight='bold')
ax.set_title('Rosenbrock - Final Posterior Distribution', fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=14, framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig('results/rosenbrock_result.png', dpi=200, bbox_inches='tight')
plt.close()

# Plot all stages (convergence process)
print("Saving convergence stages...")
all_samples = tmcmc.get_all_samples()
n_stages = all_samples.shape[0]
os.makedirs('results/rosenbrock_stages', exist_ok=True)

for stage_idx in range(n_stages):
    fig, ax = plt.subplots(figsize=(10, 10))
    samples = all_samples[stage_idx]

    # Use gradient colors based on stage progress
    progress = stage_idx / (n_stages - 1) if n_stages > 1 else 1.0
    color = plt.cm.viridis(progress)

    ax.scatter(samples[:, 0], samples[:, 1], c=[color], s=3, alpha=0.5, edgecolors='none')
    ax.set_xlabel('θ₁', fontsize=16, fontweight='bold')
    ax.set_ylabel('θ₂', fontsize=16, fontweight='bold')
    ax.set_title(f'Rosenbrock - Stage {stage_idx}/{n_stages-1}', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()
    plt.savefig(f'results/rosenbrock_stages/stage_{stage_idx:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"✓ Saved {n_stages} stage images")


# ========================================================================================================
# Benchmark 2: Griewank
# ========================================================================================================
print("\n" + "="*100)
print("Benchmark 2: Griewank Function")
print("="*100)

bounds = (-10.0, 10.0)

prior_1 = PriorDistribution('uniform', low=bounds[0], high=bounds[1])
prior_2 = PriorDistribution('uniform', low=bounds[0], high=bounds[1])
prior = JointPrior([prior_1, prior_2])

def likelihood_func(theta):
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    n_samples = theta.shape[0]
    likelihoods = np.zeros(n_samples)
    temperature = 1.0
    for i in range(n_samples):
        f_val = griewank(theta[i])
        likelihoods[i] = np.exp(-f_val / temperature)
    return likelihoods / (likelihoods.max() + 1e-300)

likelihood = LikelihoodFunction(likelihood_func)
init_samples = prior.sample(n_chains)

print("\nRunning T-MCMC...")
tmcmc = TransitionalMCMC(initial_beta=0.2, target_cov=1.0, adapt_beta=True, proposal_type='gaussian', verbose=False)
posterior = tmcmc.sample(likelihood=likelihood, prior=prior, init_samples=init_samples, n_mh_steps=n_mh_steps)

# Plot final posterior (beautified)
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(posterior[:, 0], posterior[:, 1], c='#3498db', s=3, alpha=0.5, edgecolors='none', label='Posterior Samples')
ax.set_xlabel('θ₁', fontsize=16, fontweight='bold')
ax.set_ylabel('θ₂', fontsize=16, fontweight='bold')
ax.set_title('Griewank - Final Posterior Distribution', fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=14, framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig('results/griewank_result.png', dpi=200, bbox_inches='tight')
plt.close()

# Plot all stages
print("Saving convergence stages...")
all_samples = tmcmc.get_all_samples()
n_stages = all_samples.shape[0]
os.makedirs('results/griewank_stages', exist_ok=True)

for stage_idx in range(n_stages):
    fig, ax = plt.subplots(figsize=(10, 10))
    samples = all_samples[stage_idx]
    progress = stage_idx / (n_stages - 1) if n_stages > 1 else 1.0
    color = plt.cm.viridis(progress)

    ax.scatter(samples[:, 0], samples[:, 1], c=[color], s=3, alpha=0.5, edgecolors='none')
    ax.set_xlabel('θ₁', fontsize=16, fontweight='bold')
    ax.set_ylabel('θ₂', fontsize=16, fontweight='bold')
    ax.set_title(f'Griewank - Stage {stage_idx}/{n_stages-1}', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()
    plt.savefig(f'results/griewank_stages/stage_{stage_idx:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"✓ Saved {n_stages} stage images")


# ========================================================================================================
# Benchmark 3: Himmelblau
# ========================================================================================================
print("\n" + "="*100)
print("Benchmark 3: Himmelblau Function")
print("="*100)

bounds = (-5.0, 5.0)

prior_1 = PriorDistribution('uniform', low=bounds[0], high=bounds[1])
prior_2 = PriorDistribution('uniform', low=bounds[0], high=bounds[1])
prior = JointPrior([prior_1, prior_2])

def likelihood_func(theta):
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    n_samples = theta.shape[0]
    likelihoods = np.zeros(n_samples)
    temperature = 1.0
    for i in range(n_samples):
        f_val = himmelblau(theta[i])
        likelihoods[i] = np.exp(-f_val / temperature)
    return likelihoods / (likelihoods.max() + 1e-300)

likelihood = LikelihoodFunction(likelihood_func)
init_samples = prior.sample(n_chains)

print("\nRunning T-MCMC...")
tmcmc = TransitionalMCMC(initial_beta=0.2, target_cov=1.0, adapt_beta=True, proposal_type='gmm', verbose=False)
posterior = tmcmc.sample(likelihood=likelihood, prior=prior, init_samples=init_samples, n_mh_steps=n_mh_steps)

# Plot final posterior (beautified)
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(posterior[:, 0], posterior[:, 1], c='#3498db', s=3, alpha=0.5, edgecolors='none', label='Posterior Samples')
ax.set_xlabel('θ₁', fontsize=16, fontweight='bold')
ax.set_ylabel('θ₂', fontsize=16, fontweight='bold')
ax.set_title('Himmelblau - Final Posterior Distribution', fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=14, framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig('results/himmelblau_result.png', dpi=200, bbox_inches='tight')
plt.close()

# Plot all stages
print("Saving convergence stages...")
all_samples = tmcmc.get_all_samples()
n_stages = all_samples.shape[0]
os.makedirs('results/himmelblau_stages', exist_ok=True)

for stage_idx in range(n_stages):
    fig, ax = plt.subplots(figsize=(10, 10))
    samples = all_samples[stage_idx]
    progress = stage_idx / (n_stages - 1) if n_stages > 1 else 1.0
    color = plt.cm.viridis(progress)

    ax.scatter(samples[:, 0], samples[:, 1], c=[color], s=3, alpha=0.5, edgecolors='none')
    ax.set_xlabel('θ₁', fontsize=16, fontweight='bold')
    ax.set_ylabel('θ₂', fontsize=16, fontweight='bold')
    ax.set_title(f'Himmelblau - Stage {stage_idx}/{n_stages-1}', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()
    plt.savefig(f'results/himmelblau_stages/stage_{stage_idx:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"✓ Saved {n_stages} stage images")


# ========================================================================================================
# Benchmark 4: Schwefel
# ========================================================================================================
print("\n" + "="*100)
print("Benchmark 4: Schwefel Function")
print("="*100)

bounds = (-500.0, 500.0)

prior_1 = PriorDistribution('uniform', low=bounds[0], high=bounds[1])
prior_2 = PriorDistribution('uniform', low=bounds[0], high=bounds[1])
prior = JointPrior([prior_1, prior_2])

def likelihood_func(theta):
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    n_samples = theta.shape[0]
    likelihoods = np.zeros(n_samples)
    temperature = 10.0
    for i in range(n_samples):
        f_val = schwefel(theta[i])
        likelihoods[i] = np.exp(-f_val / temperature)
    return likelihoods / (likelihoods.max() + 1e-300)

likelihood = LikelihoodFunction(likelihood_func)
init_samples = prior.sample(n_chains)

print("\nRunning T-MCMC...")
tmcmc = TransitionalMCMC(initial_beta=0.2, target_cov=1.0, adapt_beta=True, proposal_type='gaussian', verbose=False)
posterior = tmcmc.sample(likelihood=likelihood, prior=prior, init_samples=init_samples, n_mh_steps=n_mh_steps)

# Plot final posterior (beautified)
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(posterior[:, 0], posterior[:, 1], c='#3498db', s=3, alpha=0.5, edgecolors='none', label='Posterior Samples')
ax.set_xlabel('θ₁', fontsize=16, fontweight='bold')
ax.set_ylabel('θ₂', fontsize=16, fontweight='bold')
ax.set_title('Schwefel - Final Posterior Distribution', fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=14, framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig('results/schwefel_result.png', dpi=200, bbox_inches='tight')
plt.close()

# Plot all stages
print("Saving convergence stages...")
all_samples = tmcmc.get_all_samples()
n_stages = all_samples.shape[0]
os.makedirs('results/schwefel_stages', exist_ok=True)

for stage_idx in range(n_stages):
    fig, ax = plt.subplots(figsize=(10, 10))
    samples = all_samples[stage_idx]
    progress = stage_idx / (n_stages - 1) if n_stages > 1 else 1.0
    color = plt.cm.viridis(progress)

    ax.scatter(samples[:, 0], samples[:, 1], c=[color], s=3, alpha=0.5, edgecolors='none')
    ax.set_xlabel('θ₁', fontsize=16, fontweight='bold')
    ax.set_ylabel('θ₂', fontsize=16, fontweight='bold')
    ax.set_title(f'Schwefel - Stage {stage_idx}/{n_stages-1}', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()
    plt.savefig(f'results/schwefel_stages/stage_{stage_idx:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"✓ Saved {n_stages} stage images")


# ========================================================================================================
# Benchmark 5: TwoDOF Case1
# ========================================================================================================
print("\n" + "="*100)
print("Benchmark 5: TwoDOF Case1 - First Eigenvalue Only")
print("="*100)

bounds = (0.5, 2.5)

prior_1 = PriorDistribution('uniform', low=bounds[0], high=bounds[1])
prior_2 = PriorDistribution('uniform', low=bounds[0], high=bounds[1])
prior = JointPrior([prior_1, prior_2])

def likelihood_func(theta):
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    n_samples = theta.shape[0]
    likelihoods = np.zeros(n_samples)
    for i in range(n_samples):
        f_val = two_dof_case1(theta[i])
        likelihoods[i] = np.exp(-f_val)
    return likelihoods / (likelihoods.max() + 1e-300)

likelihood = LikelihoodFunction(likelihood_func)
init_samples = prior.sample(n_chains)

print("\nRunning T-MCMC...")
tmcmc = TransitionalMCMC(initial_beta=0.2, target_cov=1.0, adapt_beta=True, proposal_type='gaussian', verbose=False)
posterior = tmcmc.sample(likelihood=likelihood, prior=prior, init_samples=init_samples, n_mh_steps=n_mh_steps)

# Plot final posterior (beautified)
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(posterior[:, 0], posterior[:, 1], c='#3498db', s=3, alpha=0.5, edgecolors='none', label='Posterior Samples')
ax.set_xlabel('k₁', fontsize=16, fontweight='bold')
ax.set_ylabel('k₂', fontsize=16, fontweight='bold')
ax.set_title('TwoDOF Case1 - Final Posterior Distribution', fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=14, framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig('results/twodof_case1_result.png', dpi=200, bbox_inches='tight')
plt.close()

# Plot all stages
print("Saving convergence stages...")
all_samples = tmcmc.get_all_samples()
n_stages = all_samples.shape[0]
os.makedirs('results/twodof_case1_stages', exist_ok=True)

for stage_idx in range(n_stages):
    fig, ax = plt.subplots(figsize=(10, 10))
    samples = all_samples[stage_idx]
    progress = stage_idx / (n_stages - 1) if n_stages > 1 else 1.0
    color = plt.cm.viridis(progress)

    ax.scatter(samples[:, 0], samples[:, 1], c=[color], s=3, alpha=0.5, edgecolors='none')
    ax.set_xlabel('k₁', fontsize=16, fontweight='bold')
    ax.set_ylabel('k₂', fontsize=16, fontweight='bold')
    ax.set_title(f'TwoDOF Case1 - Stage {stage_idx}/{n_stages-1}', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()
    plt.savefig(f'results/twodof_case1_stages/stage_{stage_idx:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"✓ Saved {n_stages} stage images")


# ========================================================================================================
# Benchmark 6: TwoDOF Case2
# ========================================================================================================
print("\n" + "="*100)
print("Benchmark 6: TwoDOF Case2 - Two Eigenvalues")
print("="*100)

bounds = (0.5, 2.5)

prior_1 = PriorDistribution('uniform', low=bounds[0], high=bounds[1])
prior_2 = PriorDistribution('uniform', low=bounds[0], high=bounds[1])
prior = JointPrior([prior_1, prior_2])

def likelihood_func(theta):
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    n_samples = theta.shape[0]
    likelihoods = np.zeros(n_samples)
    for i in range(n_samples):
        f_val = two_dof_case2(theta[i])
        likelihoods[i] = np.exp(-f_val)
    return likelihoods / (likelihoods.max() + 1e-300)

likelihood = LikelihoodFunction(likelihood_func)
init_samples = prior.sample(n_chains)

print("\nRunning T-MCMC...")
tmcmc = TransitionalMCMC(initial_beta=0.2, target_cov=1.0, adapt_beta=True, proposal_type='gaussian', verbose=False)
posterior = tmcmc.sample(likelihood=likelihood, prior=prior, init_samples=init_samples, n_mh_steps=n_mh_steps)

# Plot final posterior (beautified)
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(posterior[:, 0], posterior[:, 1], c='#3498db', s=3, alpha=0.5, edgecolors='none', label='Posterior Samples')
ax.set_xlabel('k₁', fontsize=16, fontweight='bold')
ax.set_ylabel('k₂', fontsize=16, fontweight='bold')
ax.set_title('TwoDOF Case2 - Final Posterior Distribution', fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=14, framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig('results/twodof_case2_result.png', dpi=200, bbox_inches='tight')
plt.close()

# Plot all stages
print("Saving convergence stages...")
all_samples = tmcmc.get_all_samples()
n_stages = all_samples.shape[0]
os.makedirs('results/twodof_case2_stages', exist_ok=True)

for stage_idx in range(n_stages):
    fig, ax = plt.subplots(figsize=(10, 10))
    samples = all_samples[stage_idx]
    progress = stage_idx / (n_stages - 1) if n_stages > 1 else 1.0
    color = plt.cm.viridis(progress)

    ax.scatter(samples[:, 0], samples[:, 1], c=[color], s=3, alpha=0.5, edgecolors='none')
    ax.set_xlabel('k₁', fontsize=16, fontweight='bold')
    ax.set_ylabel('k₂', fontsize=16, fontweight='bold')
    ax.set_title(f'TwoDOF Case2 - Stage {stage_idx}/{n_stages-1}', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()
    plt.savefig(f'results/twodof_case2_stages/stage_{stage_idx:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"✓ Saved {n_stages} stage images")


# ========================================================================================================
# Benchmark 7: TwoDOF Case3
# ========================================================================================================
print("\n" + "="*100)
print("Benchmark 7: TwoDOF Case3 - First Eigenvalue + Mode Shape Ratio")
print("="*100)

bounds = (0.5, 2.5)

prior_1 = PriorDistribution('uniform', low=bounds[0], high=bounds[1])
prior_2 = PriorDistribution('uniform', low=bounds[0], high=bounds[1])
prior = JointPrior([prior_1, prior_2])

def likelihood_func(theta):
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    n_samples = theta.shape[0]
    likelihoods = np.zeros(n_samples)
    for i in range(n_samples):
        f_val = two_dof_case3(theta[i])
        likelihoods[i] = np.exp(-f_val)
    return likelihoods / (likelihoods.max() + 1e-300)

likelihood = LikelihoodFunction(likelihood_func)
init_samples = prior.sample(n_chains)

print("\nRunning T-MCMC...")
tmcmc = TransitionalMCMC(initial_beta=0.2, target_cov=1.0, adapt_beta=True, proposal_type='gaussian', verbose=False)
posterior = tmcmc.sample(likelihood=likelihood, prior=prior, init_samples=init_samples, n_mh_steps=n_mh_steps)

# Plot final posterior (beautified)
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(posterior[:, 0], posterior[:, 1], c='#3498db', s=3, alpha=0.5, edgecolors='none', label='Posterior Samples')
ax.set_xlabel('k₁', fontsize=16, fontweight='bold')
ax.set_ylabel('k₂', fontsize=16, fontweight='bold')
ax.set_title('TwoDOF Case3 - Final Posterior Distribution', fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=14, framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax.set_facecolor('#f8f9fa')
plt.tight_layout()
plt.savefig('results/twodof_case3_result.png', dpi=200, bbox_inches='tight')
plt.close()

# Plot all stages
print("Saving convergence stages...")
all_samples = tmcmc.get_all_samples()
n_stages = all_samples.shape[0]
os.makedirs('results/twodof_case3_stages', exist_ok=True)

for stage_idx in range(n_stages):
    fig, ax = plt.subplots(figsize=(10, 10))
    samples = all_samples[stage_idx]
    progress = stage_idx / (n_stages - 1) if n_stages > 1 else 1.0
    color = plt.cm.viridis(progress)

    ax.scatter(samples[:, 0], samples[:, 1], c=[color], s=3, alpha=0.5, edgecolors='none')
    ax.set_xlabel('k₁', fontsize=16, fontweight='bold')
    ax.set_ylabel('k₂', fontsize=16, fontweight='bold')
    ax.set_title(f'TwoDOF Case3 - Stage {stage_idx}/{n_stages-1}', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()
    plt.savefig(f'results/twodof_case3_stages/stage_{stage_idx:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"✓ Saved {n_stages} stage images")

print("\n" + "="*100)
print("All benchmarks completed!")
print("="*100)
