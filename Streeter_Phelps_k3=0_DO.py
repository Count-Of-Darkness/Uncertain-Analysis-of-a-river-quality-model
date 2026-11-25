import matplotlib.pyplot as plt
import numpy as np

# Time axis: t in days (0 to 50 days)
t = np.linspace(0, 50, 1000)

# Base parameters for original Streeter-Phelps model (k3 = 0)
b0 = 20.0  # Initial BOD concentration [mg/L]
c0 = 9.0  # Initial DO concentration [mg/L]
cs = 9.5  # Saturation concentration [mg/L]
k2 = 0.25  # Reaeration rate [1/day]

# Parameter variation for BOD degradation rate k1 [1/day]
k1_values = [0.02, 0.05, 0.10, 0.30, 0.50]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Figure setup
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

# Plot c(t) curves for each k1 value
for k1, color in zip(k1_values, colors):

  # Oxygen deficit D(t) = cs - c(t)
  # Oxygen consumption through BOD decomposition and beginning deficit
  D_t = (k1 * b0 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t)) + (
      cs - c0
  ) * np.exp(-k2 * t)

  # oxygen levels
  c_t = cs - D_t

  ax.plot(
      t,
      c_t,
      label=f'$k_1 = {k1:.2f}$ day$^{{-1}}$',
      color=color,
      linewidth=2.0,
  )

# Reference lines
# horizontal line for cs
ax.axhline(
    cs,
    color='green',
    linestyle='--',
    linewidth=1.5,
    label=f'DO Saturation $c_s = {cs}$ mg/L',
)

# critical threshold
ax.axhline(
    5.0,
    color='darkorange',
    linestyle=':',
    linewidth=1.5,
    label='Critical Threshold (5.0 mg/L)',
)

# labels
ax.set_title(
    '$c(t)$ functions for Varying $k_1$',
    fontsize=13,
    fontweight='bold',
    pad=12,
)

ax.set_xlabel('Time $t$ [Days]', fontsize=11)
ax.set_ylabel('DO Concentration $c(t)$ [mg/L]', fontsize=11)
ax.set_xlim(0, 50)
ax.set_ylim(-2, 11)			# y-axis goes into negative numbers
ax.grid(True, linestyle=':', alpha=0.6)

# Spine cleanup
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='lower right', fontsize=9.0, framealpha=0.95)

plt.tight_layout(pad=1.5)
plt.show()