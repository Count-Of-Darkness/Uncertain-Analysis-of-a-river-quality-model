import matplotlib.pyplot as plt
import numpy as np

# Time axis: t in days (0 to 100)
t = np.linspace(0, 100, 1000)

# Model parameters
b0 = 20.0  # Initial BOD concentration [mg/L]
c0 = 9.0  # Initial DO concentration [mg/L]
cs_val = 9.5  # Reference saturation concentration [mg/L]

# Parameter bounds for uncertainty analysis
k1_min, k1_max = 0.05, 0.10  # BOD degradation rate bounds [1/day]
k2_min, k2_max = 0.20, 1.00  # Reaeration rate bounds [1/day]
kappa_min, kappa_max = 0.25, 0.75  # Half-saturation constant bounds [mg/L]
cs_min, cs_max = 9.0, 10.0  # Saturation concentration bounds [mg/L]

# 1. BOD Solution Bounds b(t) without external source
# Upper estimate for BOD:
b_upper = np.full_like(t, b0)

# Lower estimate for BOD:
b_lower = b0 * np.exp(-k1_max * t)


# 2. DO Solution Bounds c(t) without external source
# ODE for Upper DO Estimate c_upper(t)
def do_upper_ode(c, t_val):
  b_val = b0 * np.exp(-k1_max * t_val)   # lower b esstimate
  # Monod term upper estimate bound: c / (kappa_max + cs_max)
  rate_decay = k1_min * b_val * (c / (kappa_max + cs_max))  # small sink rate
  rate_reaeration = k2_max * (cs_max - c)     # big reaeration rate
  return -rate_decay + rate_reaeration       # derivation of the upper DO solution


# Explicit Runge-Kutta 4th Order Integrator 
def rk4_solve(func, y0, t_arr):
  y = np.zeros(len(t_arr))
  y[0] = y0
  for i in range(len(t_arr) - 1):
    dt = t_arr[i + 1] - t_arr[i]
    t_curr = t_arr[i]
    c_curr = y[i]

    k1_val = func(c_curr, t_curr)
    k2_val = func(c_curr + 0.5 * dt * k1_val, t_curr + 0.5 * dt)
    k3_val = func(c_curr + 0.5 * dt * k2_val, t_curr + 0.5 * dt)
    k4_val = func(c_curr + dt * k3_val, t_arr[i + 1])

    y[i + 1] = c_curr + (dt / 6.0) * (k1_val + 2 * k2_val + 2 * k3_val + k4_val)
  return y


# Numerical integration of the upper DO bound
c_upper = rk4_solve(do_upper_ode, c0, t)

# Lower DO Estimate c_lower(t)
# dc/dt = - k1_max * b0 + k2_min * (cs_min - c) derivation of the lower solution esstimate
c_lower_sol = (cs_min - (k1_max * b0) / k2_min) + (
    c0 - cs_min + (k1_max * b0) / k2_min
) * np.exp(-k2_min * t)                # lower bound solution c_min(t)
c_lower = np.maximum(c_lower_sol, 0.0)  # making sure it does not durpass 0


# 3. Figure Plotting
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

# Subtle BOD Bounds
ax.fill_between(
    t,
    b_lower,
    b_upper,
    color='gray',
    alpha=0.07,
    label='BOD Uncertainty Domain [b(t)]',
)

ax.plot(
    t,
    b_upper,
    color='gray',
    linestyle='-.',
    linewidth=1.2,
    alpha=0.5,
    label='Upper BOD Estimate ($b_0 = 20.0$)',
)

ax.plot(
    t,
    b_lower,
    color='gray',
    linestyle='--',
    linewidth=1.2,
    alpha=0.5,
    label='Lower BOD Estimate',
)

# DO Bounds
ax.fill_between(
    t,
    c_lower,
    c_upper,
    color='steelblue',
    alpha=0.18,
    label='DO Uncertainty Domain [c(t)]',
)

ax.plot(
    t,
    c_upper,
    color='navy',
    linestyle='-.',
    linewidth=2.2,
    label='Upper DO Estimate ($c_{upper}(t)$)',
)
ax.plot(
    t,
    c_lower,
    color='crimson',
    linestyle='--',
    linewidth=2.2,
    label='Lower DO Estimate ($c_{lower}(t)$)',
)

# Critical Threshold Line
ax.axhline(
    5.0,
    color='darkorange',
    linestyle=':',
    linewidth=1.5,
    label='Critical DO Threshold (5.0 mg/L)',
)

# Formatting and Labels
ax.set_title(
    'Solution Estimates of the DO concentration without External BOD Source',
    fontsize=13,
    fontweight='bold',
    pad=12,
)

ax.set_xlabel('Time (t) [Days]', fontsize=11)
ax.set_ylabel('Concentration [mg/L]', fontsize=11)
ax.set_xlim(0, 100)
ax.set_ylim(0, 22)
ax.grid(True, linestyle=':', alpha=0.5)

# Spine cleanup
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)

plt.tight_layout(pad=1.5)
plt.show()