import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

# Time axis: t in days (0 to 100)
t = np.linspace(0, 100, 1000)

# Model parameters
b0 = 20.0  # Initial BOD concentration [mg/L]
k1_max = 0.10  # Maximum BOD degradation rate [1/day]

# Upper Estimate (Constant solution)
# along entire t-axis fill with value b0 
b_upper = np.full_like(t, b0)

# Lower Estimate
b_lower = b0 * np.exp(-k1_max * t)

# Figure setup
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

# Fill uncertainty domain between lower and upper bounds
ax.fill_between(
    t,
    b_lower,
    b_upper,
    color='gray',
    alpha=0.12,
    label='Possibilistic Uncertainty Domain',
)

# Plot Upper Estimate
ax.plot(
    t,
    b_upper,
    color='black',
    linestyle='-.',
    linewidth=2.0,
    label=rf'Upper Estimate: $b_0 = {b0}$',
)

# Plot Lower Estimate
ax.plot(
    t,
    b_lower,
    color='black',
    linestyle='--',
    linewidth=2.0,
    label=rf'Lower Estimate: $k_1 = {k1_max:.2f}$',
)

# Chart styling and labels
ax.set_title(
    'Solution Estimates of Nonlinear BOD Concentration without External Source',
    fontsize=12,
    fontweight='bold',
    pad=12,
)

ax.set_xlabel('Time (t) [Days]', fontsize=11)
ax.set_ylabel('BOD Concentration $b(t)$ [mg/L]', fontsize=11)
ax.set_xlim(0, 100)
ax.set_ylim(0, 22)
ax.grid(True, linestyle=':', alpha=0.6)

# Spines cleanup
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='lower left', fontsize=9.5, framealpha=0.95)

plt.tight_layout(pad=1.5)
plt.show()