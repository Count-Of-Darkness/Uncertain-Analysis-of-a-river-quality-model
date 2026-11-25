import matplotlib.pyplot as plt
import numpy as np

# Time axis: t in days
t = np.linspace(0, 100, 1000)

# Parameter intervals
k1_min, k1_max = 0.3, 1.0  # BOD degradation rate [1/day]
k2_min, k2_max = 2.0, 3.0  # Reaeration rate [1/day]
k3_min, k3_max = 1.0, 3.0  # External pollution source [mg/(L*day)]
cs_min, cs_max = 9.0, 10.0  # Dissolved oxygen saturation [mg/L]
b0_min, b0_max = 18.0, 22.0  # Initial BOD concentration [mg/L]
c0_min, c0_max = 9.0, 10.0  # Initial dissolved oxygen concentration [mg/L]

# 1. BOD Bounds b(t)
# Lower BOD bound b_lower(t): k1_max, k3_min, b0_min
b_lower = (k3_min / k1_max) + (b0_min - (k3_min / k1_max)) * np.exp(-k1_max * t)

# Upper BOD bound b_upper(t): k1_min, k3_max, b0_max
b_upper = (k3_max / k1_min) + (b0_max - (k3_max / k1_min)) * np.exp(-k1_min * t)

# 2. Dissolved Oxygen (DO) Bounds c(t)
# Upper DO bound c_upper(t):
# Best-case DO, coupled with lower BOD bound (k1_min, k2_max, cs_max, k3_min, b0_min, c0_max)
# we seperate the equation in 3 different parts (the constant, the e^(-k1*t) and the e^(-k2*t))
term_const_upper = cs_max - (k1_min * k3_min) / (k1_max * k2_max)
term_exp1_upper = (
    (k1_min * k3_min) / (k1_max * (k2_max - k1_max))
    - (k1_min * b0_min) / (k2_max - k1_max)
    ) * np.exp(-k1_max * t)
term_exp2_upper = (
    c0_max
    - term_const_upper
    - (k1_min * k3_min) / (k1_max * (k2_max - k1_max))
        + (k1_min * b0_min) / (k2_max - k1_max)
    ) * np.exp(-k2_max * t)

c_upper = term_const_upper + term_exp1_upper + term_exp2_upper

# Lower DO bound c_lower(t):
# Worst-case DO, coupled with upper BOD bound (k1_max, k2_min, cs_min, k3_max, b0_max, c0_min)
# same procedure as above
term_const_lower = cs_min - (k1_max * k3_max) / (k1_min * k2_min)
term_exp1_lower = (
    (k1_max * k3_max) / (k1_min * (k2_min - k1_min))
    - (k1_max * b0_max) / (k2_min - k1_min)
) * np.exp(-k1_min * t)
term_exp2_lower = (
    c0_min
    - term_const_lower
    - (k1_max * k3_max) / (k1_min * (k2_min - k1_min))
        + (k1_max * b0_max) / (k2_min - k1_min)
        ) * np.exp(-k2_min * t)

c_lower = term_const_lower + term_exp1_lower + term_exp2_lower

# 3. Plotting Combined Figure 
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

# Fill uncertainty domains
# Subdued grey fill for BOD
ax.fill_between(
    t,
    b_lower,
    b_upper,
    color='gray',
    alpha=0.08,
    label='BOD Possibilistic Domain [b(t)]',
)

# Prominent blue fill for Dissolved Oxygen
ax.fill_between(
    t,
    c_lower,
    c_upper,
    color='steelblue',
    alpha=0.20,
    label='Dissolved Oxygen Possibilistic Domain [c(t)]',
)

# Subdued dotted lines for BOD bounds
ax.plot(
    t,
    b_upper,
    color='gray',
    linestyle=':',
    linewidth=1.2,
    alpha=0.6,
    label='Upper BOD Bound $b_{upper}(t)$',
)
ax.plot(
    t,
    b_lower,
    color='gray',
    linestyle=':',
    linewidth=1.2,
    alpha=0.6,
    label='Lower BOD Bound $b_{lower}(t)$',
)

# Prominent lines for Dissolved Oxygen bounds
ax.plot(
    t,
    c_upper,
    color='navy',
    linestyle='--',
    linewidth=2.2,
    label='Upper DO Bound $c_{upper}(t)$',
)
ax.plot(
    t,
    c_lower,
    color='crimson',
    linestyle='-.',
    linewidth=2.2,
    label='Lower DO Bound $c_{lower}(t)$',
)

# Critical threshold line for aquatic life (5.0 mg/L)
ax.axhline(
    5.0,
    color='darkorange',
    linestyle='--',
    linewidth=1.5,
    label='Critical DO Threshold (5.0 mg/L)',
)

# Formatting
ax.set_title(
    'Solution Bounds for Dissolved Oxygen c(t)',
    fontsize=13,
    fontweight='bold',
    pad=12,
)

ax.set_xlabel('Time t [Days]', fontsize=11)
ax.set_ylabel('Concentration [mg/L]', fontsize=11)
ax.set_xlim(0, 100)
ax.set_ylim(0, 24)

ax.grid(True, linestyle=':', alpha=0.5)

# Clean top and right frame borders
# only axis on bottom and left
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend (information box)
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)

plt.tight_layout(pad=1.5)
plt.show()