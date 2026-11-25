import matplotlib.pyplot as plt
import numpy as np

# 1. Model parameters
kappa = 0.5  # Half-saturation constant [mg/L]
# start at oxygen level 0 and progress with calculations in 500 steps until we reach 10
c = np.linspace(0.0, 10.0, 500)  # Dissolved oxygen concentration [mg/L]

# 2. Monod kinetics saturation function: Psi(c) = c / (kappa + c)
psi_c = c / (kappa + c)

# 3. Create plot
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

# Plot Monod saturation curve
ax.plot(
    c,
    psi_c,
    color='black',
    linewidth=2.0,
    label=r'Monod Function $\Phi(c) = \frac{c}{\kappa + c}$',
)

# Horizontal line at phi(c) = 0.5 (half-saturation level)
ax.axhline(
    0.5,
    color='gray',
    linestyle='--',
    linewidth=1.2,
    label=r'Half-Saturation Level ($\Phi(c) = 0.5$)',
)

# Vertical line at c = kappa
ax.axvline(
    kappa,
    color='crimson',
    linestyle=':',
    linewidth=1.2,
    label=rf'Half-Saturation Constant ($\kappa = {kappa}$ mg/L)',
)

# Highlight point (kappa, 0.5)
ax.plot(kappa, 0.5, 'ro', markersize=7, zorder=5)   # red dot at the intersection

# point at the intersection with an arrow
ax.annotate(
    r'$\left(\kappa, \frac{1}{2}\right)$',
    xy=(kappa, 0.5),
    xytext=(kappa + 0.4, 0.43),
    fontsize=11,
    color='crimson',
    fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='crimson', lw=1.2),
)

# setting the lines on the x-axis (the relevant values)
ax.set_xticks([0.0, kappa, 2.5, 5.0, 7.5, 10.0])  # where to set the marks
ax.set_xticklabels(
    ['0.0', r'$\kappa$' + f' ({kappa})', '2.5', '5.0', '7.5', '10.0']   # how to label them
)

# Chart styling and labels
ax.set_title(
    'Monod Function', fontsize=13, fontweight='bold', pad=12
)

ax.set_xlabel('Dissolved Oxygen Concentration $c$ [mg/L]', fontsize=11)
ax.set_ylabel(r'Monod Factor $\frac{c}{\kappa + c}$', fontsize=11)
ax.set_xlim(0.0, 10.0)
ax.set_ylim(0.0, 1.0)
ax.grid(True, linestyle=':', alpha=0.6)     # grid on

# Spines cleanup
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='lower right', fontsize=9.5, framealpha=0.95)  # box on lower right corner

plt.tight_layout(pad=1.5)
plt.show()