import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducible sampling of trajectories
# making sure the program always generates the same samples if you run it
np.random.seed(42)

# Time array: t in days
t = np.linspace(0, 100, 1000)

# Parameter bounds 
k1_min, k1_max = 0.30, 1.00  # BOD degradation rate interval [1/day]
k3_min, k3_max = 1.0, 3.0  # External BOD source term interval [mg/(L*day)]
b0_min, b0_max = 18.0, 22.0  # Initial BOD concentration interval [mg/L]

# Lower Solution Bound:
# Uses maximum decay rate (k1_max), minimum pollution source (k3_min), and minimum initial value (b0_min)
b_lower = (k3_min / k1_max) + (b0_min - (k3_min / k1_max)) * np.exp(-k1_max * t)

# Upper Solution Bound:
# Uses minimum decay rate (k1_min), maximum pollution source (k3_max), and maximum initial value (b0_max)
b_upper = (k3_max / k1_min) + (b0_max - (k3_max / k1_min)) * np.exp(-k1_min * t)

# Generate exact solution trajectories for random parameter combinations within the bounds
num_samples = 50  # we draw 50 random combinations of parameters and run the solution
sample_trajectories = []   # generate an empty list that saves the 50 solution curves

for _ in range(num_samples):
  # Sample parameter values uniformly within their uncertainty ranges
  k1_sample = np.random.uniform(k1_min, k1_max)  # draw value for k1 between it's bounds
  k3_sample = np.random.uniform(k3_min, k3_max)
  b0_sample = np.random.uniform(b0_min, b0_max)

  # Exact solution formula for constant parameters (b(t))
  b_exact = (k3_sample / k1_sample) + (
      b0_sample - (k3_sample / k1_sample)
  ) * np.exp(-k1_sample * t)
    
  sample_trajectories.append(b_exact)   # save the new curve in our list

# Figure setup
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

# 1. Fill the possibilistic uncertainty domain between lower and upper bounds
ax.fill_between(
    t,
    b_lower,
    b_upper,
    color='skyblue',
    alpha=0.15,
    label='Possibilistic Uncertainty Domain',
)

# 2. Plot exact sample trajectories with light transparency
for i, traj in enumerate(sample_trajectories):
  ax.plot(
      t,
      traj,
      color='steelblue',
      alpha=0.20,
      linewidth=1.2,
      label='Exact Sampled Solutions' if i == 0 else '',
  )

# 3. Plot Upper Estimate
ax.plot(
    t,
    b_upper,
    color='darkred',
    linestyle='--',
    linewidth=2.5,
    label=f'Upper Bound: $k_1={k1_min}$, $k_3={k3_max}$, $b_0={b0_max}$',
)

# 4. Plot Lower Estimate
ax.plot(
    t,
    b_lower,
    color='darkgreen',
    linestyle='-.',
    linewidth=2.5,
    label=f'Lower Bound: $k_1={k1_max}$, $k_3={k3_min}$, $b_0={b0_min}$',
)

# Styling and labels
ax.set_title(
    'Solution Estimates of Linear BOD Concentration',
    fontsize=13,
    fontweight='bold',
    pad=12,
)

ax.set_xlabel('Time (t) [Days]', fontsize=11)
ax.set_ylabel('BOD Concentration b(t) [mg/L]', fontsize=11)
ax.set_xlim(0, 100)
ax.set_ylim(0, 24)

ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right', fontsize=9, framealpha=0.95)

plt.tight_layout()
plt.show()