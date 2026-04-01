
import numpy as np
from scipy.optimize import minimize
from scipy.stats import pareto

# Parameters
x_m = 1  # Pareto minimum
alpha = 3
e = 1  # Fixed constant

B = 4
s = 2000  # number of samples
delta = 0.10  # 10% significance level -> 90% confidence

# Generate Pareto samples (empirical distribution hat_eta_s)
np.random.seed(3313)  # for reproducibility
X_samples = pareto.rvs(b=alpha, scale=x_m, size=s)

# Threshold for constraint
threshold = 1 + 2*np.log(s + 1) + np.log(2/delta)

print(f"Threshold value: {threshold:.4f}")
print(f"Sample size s: {s}")
print(f"Delta δ: {delta}")
print(f"Confidence level 1-δ: {1-delta} (90%)")
print(f"Sample mean: {np.mean(X_samples):.6f}")
print(f"Sample std: {np.std(X_samples):.6f}")

# ========================================
# KL_inf^U (hat_eta_s, x) computation
# ========================================
def compute_KL_inf_U(x, X_samples):
    """
    Compute KL_inf^U(hat_eta_s, x) = max_{lambda in S^H(x)} E_{hat_eta}[log(g^H(X, lambda, x))]
    """
    def objective_U(lam):
        lambda_1, lambda_2 = lam
        if lambda_1 <= 0 or lambda_2 <= 0:
            return np.inf

        g_H = 1 - lambda_1 * (X_samples - x) - lambda_2 * (B - X_samples**2)

        if np.any(g_H <= 0):
            return np.inf

        return -np.mean(np.log(g_H))

    def constraint_U(lam):
        lambda_1, lambda_2 = lam
        if lambda_1 <= 0 or lambda_2 <= 0:
            return 1

        return lambda_1**2 / (4 * lambda_2) + B * lambda_2 - x * lambda_1 - 1

    lambda_init = [0.1, 0.1]
    bounds = [(1e-6, None), (1e-6, None)]

    result = minimize(
        objective_U,
        lambda_init,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': lambda lam: -constraint_U(lam)},
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    if result.success:
        return -result.fun
    else:
        return np.inf

# ========================================
# KL_inf^L (hat_eta_s, x) computation
# ========================================
def compute_KL_inf_L(x, X_samples):
    """
    Compute KL_inf^L(hat_eta_s, x) = max_{gamma in S^L(x)} E_{hat_eta}[log(g^L(X, gamma, x))]
    """
    def objective_L(gamma):
        gamma_1, gamma_2 = gamma
        if gamma_1 <= 0 or gamma_2 <= 0:
            return np.inf

        g_L = 1 + gamma_1 * (X_samples - x) - gamma_2 * (B - np.abs(X_samples)**2)

        if np.any(g_L <= 0):
            return np.inf

        return -np.mean(np.log(g_L))

    def constraint_L(gamma):
        gamma_1, gamma_2 = gamma
        if gamma_1 <= 0 or gamma_2 <= 0:
            return 1

        return gamma_1**2 / (4 * gamma_2) + x * gamma_1 + B * gamma_2 - 1

    gamma_init = [0.1, 0.1]
    bounds = [(1e-6, None), (1e-6, None)]

    result = minimize(
        objective_L,
        gamma_init,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': lambda g: -constraint_L(g)},
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    if result.success:
        return -result.fun
    else:
        return np.inf

# ========================================
# U_eta(s): max{x : s*KL_inf^U(hat_eta_s, x) <= threshold}
# ========================================
print("\n" + "="*60)
print("Computing U_η(s)...")
print("="*60)

x_grid = np.linspace(1.0, 2.5, 30)
valid_x_upper = []

for x_test in x_grid:
    KL_U = compute_KL_inf_U(x_test, X_samples)
    constraint_val = s * KL_U
    is_valid = constraint_val <= threshold

    if is_valid:
        valid_x_upper.append(x_test)

    print(f"x = {x_test:.4f}: s*KL_inf^U = {constraint_val:.4f}, threshold = {threshold:.4f} {'✓' if is_valid else '✗'}")

if valid_x_upper:
    U_eta_s = max(valid_x_upper)
else:
    U_eta_s = None

# ========================================
# L_eta(s): min{x : s*KL_inf^L(hat_eta_s, x) <= threshold}
# ========================================
print("\n" + "="*60)
print("Computing L_η(s)...")
print("="*60)

x_grid_lower = np.linspace(0.5, 1.8, 30)
valid_x_lower = []

for x_test in x_grid_lower:
    KL_L = compute_KL_inf_L(x_test, X_samples)
    constraint_val = s * KL_L
    is_valid = constraint_val <= threshold

    if is_valid:
        valid_x_lower.append(x_test)

    print(f"x = {x_test:.4f}: s*KL_inf^L = {constraint_val:.4f}, threshold = {threshold:.4f} {'✓' if is_valid else '✗'}")

if valid_x_lower:
    L_eta_s = min(valid_x_lower)
else:
    L_eta_s = None

# ========================================
# Summary
# ========================================
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Sample size s: {s}")
print(f"Confidence level: 90% (δ = {delta})")
print(f"Threshold: {threshold:.4f}")
print(f"Sample mean: {np.mean(X_samples):.6f}")
print(f"Sample std: {np.std(X_samples):.6f}")

if L_eta_s is not None:
    print(f"\nL_η(s) = {L_eta_s:.6f} (lower confidence bound)")
else:
    print(f"\nL_η(s) = NOT FOUND")

if U_eta_s is not None:
    print(f"U_η(s) = {U_eta_s:.6f} (upper confidence bound)")
else:
    print(f"U_η(s) = NOT FOUND")

if L_eta_s is not None and U_eta_s is not None:
    print(f"\n90% Confidence interval for mean: [{L_eta_s:.6f}, {U_eta_s:.6f}]")
    print(f"Interval width: {U_eta_s - L_eta_s:.6f}")

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pareto

# Parameters
x_m = 1  # Pareto minimum
alpha = 3
e = 1  # Fixed constant
B = 4
delta = 0.05  # 5% significance level -> 95% confidence

# Cost budgets to test
cost_budgets = [500, 1000]
n_simulations = 1000

def compute_KL_inf_U(x, X_samples):
    """
    Compute KL_inf^U(hat_eta_s, x) = max_{lambda in S^H(x)} E_{hat_eta}[log(g^H(X, lambda, x))]
    """
    def objective_U(lam):
        lambda_1, lambda_2 = lam
        if lambda_1 <= 0 or lambda_2 <= 0:
            return np.inf

        g_H = 1 - lambda_1 * (X_samples - x) - lambda_2 * (B - X_samples**2)

        if np.any(g_H <= 0):
            return np.inf

        return -np.mean(np.log(g_H))

    def constraint_U(lam):
        lambda_1, lambda_2 = lam
        if lambda_1 <= 0 or lambda_2 <= 0:
            return 1

        return lambda_1**2 / (4 * lambda_2) + B * lambda_2 - x * lambda_1 - 1

    lambda_init = [0.1, 0.1]
    bounds = [(1e-6, None), (1e-6, None)]

    result = minimize(
        objective_U,
        lambda_init,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': lambda lam: -constraint_U(lam)},
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    if result.success:
        return -result.fun
    else:
        return np.inf

def compute_KL_inf_L(x, X_samples):
    """
    Compute KL_inf^L(hat_eta_s, x) = max_{gamma in S^L(x)} E_{hat_eta}[log(g^L(X, gamma, x))]
    """
    def objective_L(gamma):
        gamma_1, gamma_2 = gamma
        if gamma_1 <= 0 or gamma_2 <= 0:
            return np.inf

        g_L = 1 + gamma_1 * (X_samples - x) - gamma_2 * (B - np.abs(X_samples)**2)

        if np.any(g_L <= 0):
            return np.inf

        return -np.mean(np.log(g_L))

    def constraint_L(gamma):
        gamma_1, gamma_2 = gamma
        if gamma_1 <= 0 or gamma_2 <= 0:
            return 1

        return gamma_1**2 / (4 * gamma_2) + x * gamma_1 + B * gamma_2 - 1

    gamma_init = [0.1, 0.1]
    bounds = [(1e-6, None), (1e-6, None)]

    result = minimize(
        objective_L,
        gamma_init,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': lambda g: -constraint_L(g)},
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    if result.success:
        return -result.fun
    else:
        return np.inf

def compute_confidence_interval(X_samples, delta):
    """
    Compute confidence interval given samples
    """
    s = len(X_samples)
    threshold = 1 + 2*np.log(s + 1) + np.log(2/delta)

    # Compute U_eta(s)
    x_grid = np.linspace(1.0, 2.5, 30)
    valid_x_upper = []

    for x_test in x_grid:
        KL_U = compute_KL_inf_U(x_test, X_samples)
        constraint_val = s * KL_U
        if constraint_val <= threshold:
            valid_x_upper.append(x_test)

    U_eta_s = max(valid_x_upper) if valid_x_upper else None

    # Compute L_eta(s)
    x_grid_lower = np.linspace(0.5, 1.8, 30)
    valid_x_lower = []

    for x_test in x_grid_lower:
        KL_L = compute_KL_inf_L(x_test, X_samples)
        constraint_val = s * KL_L
        if constraint_val <= threshold:
            valid_x_lower.append(x_test)

    L_eta_s = min(valid_x_lower) if valid_x_lower else None

    return L_eta_s, U_eta_s

def run_simulation(cost_budget, delta, sim_seed):
    """
    Run a single simulation with given cost budget
    """
    np.random.seed(sim_seed)

    # Sample until cost budget is exhausted
    samples = []
    total_cost = 0

    while total_cost < cost_budget:
        # Generate one sample
        sample = pareto.rvs(b=alpha, scale=x_m, size=1)[0]
        sample_cost = np.random.uniform(0, 2)

        if total_cost + sample_cost <= cost_budget:
            samples.append(sample)
            total_cost += sample_cost
        else:
            break

    X_samples = np.array(samples)
    s = len(X_samples)

    # Compute confidence interval
    L_eta_s, U_eta_s = compute_confidence_interval(X_samples, delta)

    if L_eta_s is not None and U_eta_s is not None:
        width = U_eta_s - L_eta_s
        return width, s
    else:
        return None, s

# Run Monte Carlo simulations
print("="*70)
print("MONTE CARLO SIMULATION WITH COST BUDGET CONSTRAINT")
print("="*70)
print(f"Number of simulations: {n_simulations}")
print(f"Confidence level: 95% (δ = {delta})")
print(f"Sample cost: Uniform[0, 2]")
print(f"True mean (Pareto α={alpha}, x_m={x_m}): {alpha/(alpha-1):.6f}")
print("="*70)

results = {}

for budget in cost_budgets:
    print(f"\n{'='*70}")
    print(f"COST BUDGET: {budget}")
    print(f"{'='*70}")

    widths = []
    sample_sizes = []

    for sim in range(n_simulations):
        if (sim + 1) % 100 == 0:
            print(f"Progress: {sim + 1}/{n_simulations} simulations completed...")

        width, s = run_simulation(budget, delta, sim_seed=sim + budget*10000)

        if width is not None:
            widths.append(width)
        sample_sizes.append(s)

    widths = np.array(widths)
    sample_sizes = np.array(sample_sizes)

    results[budget] = {
        'avg_width': np.mean(widths),
        'std_width': np.std(widths),
        'min_width': np.min(widths),
        'max_width': np.max(widths),
        'avg_samples': np.mean(sample_sizes),
        'std_samples': np.std(sample_sizes),
        'success_rate': len(widths) / n_simulations
    }

    print(f"\nResults for cost budget {budget}:")
    print(f"  Average number of samples: {results[budget]['avg_samples']:.2f} ± {results[budget]['std_samples']:.2f}")
    print(f"  Average CI width: {results[budget]['avg_width']:.6f}")
    print(f"  Std dev of CI width: {results[budget]['std_width']:.6f}")
    print(f"  Min CI width: {results[budget]['min_width']:.6f}")
    print(f"  Max CI width: {results[budget]['max_width']:.6f}")
    print(f"  Success rate: {results[budget]['success_rate']*100:.1f}%")

# Summary comparison
print("\n" + "="*70)
print("SUMMARY COMPARISON")
print("="*70)
print(f"{'Cost Budget':<15} {'Avg Samples':<15} {'Avg CI Width':<20} {'Improvement':<15}")
print("-"*70)

baseline_width = results[cost_budgets[0]]['avg_width']
for budget in cost_budgets:
    avg_samples = results[budget]['avg_samples']
    avg_width = results[budget]['avg_width']
    improvement = (1 - avg_width/baseline_width) * 100 if budget != cost_budgets[0] else 0

    print(f"{budget:<15} {avg_samples:<15.2f} {avg_width:<20.6f} {improvement:>13.2f}%")

print("="*70)