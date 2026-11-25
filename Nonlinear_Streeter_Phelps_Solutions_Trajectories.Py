import matplotlib.pyplot as plt
import numpy as np

# 1. Model parameters
k1 = 1.50  # Maximum BOD degradation rate [1/day]
k2 = 3.00  # Reaeration rate [1/day]
k3 = 2.00  # External BOD source term [mg/(L*day)]
cs = 9.50  # Dissolved oxygen saturation concentration [mg/L]
kappa = 1.00  # Monod half-saturation constant [mg/L]

# 2. Equilibrium state (b*, c*)
c_star = cs - (k3 / k2)
b_star = (k3 * (c_star + kappa)) / (k1 * c_star)


# Monod saturation kinetics function Psi(c)
def psi(c):
  c_pos = np.maximum(c, 0.0)  # oxygen can't dive into the negative numbers 
  return c_pos / (kappa + c_pos)    # monod function


# System of coupled nonlinear differential equations
def nonlinear_streeter_phelps(y, t):
  b, c = y
  dbdt = -k1 * b * psi(c) + k3
  dcdt = -k1 * b * psi(c) + k2 * (cs - c)
  return np.array([dbdt, dcdt])    # use the derivations as vectors


# Explicit Runge-Kutta 4th Order Integrator
# callculates b(t) and c(t) later
def rk4_solve(sys_func, y0, t):
  y_sol = np.zeros((len(t), len(y0)))
  y_sol[0] = np.array(y0, dtype=float)
  for i in range(len(t) - 1):
    dt = t[i + 1] - t[i]
    y_curr = y_sol[i]
    k1_step = sys_func(y_curr, t[i])
    k2_step = sys_func(y_curr + 0.5 * dt * k1_step, t[i] + 0.5 * dt)
    k3_step = sys_func(y_curr + 0.5 * dt * k2_step, t[i] + 0.5 * dt)
    k4_step = sys_func(y_curr + dt * k3_step, t[i + 1])
    y_sol[i + 1] = y_curr + (dt / 6.0) * (
        k1_step + 2 * k2_step + 2 * k3_step + k4_step
    )
  return y_sol


# 3. Vector field (Direction arrows)
b_vec = np.linspace(0.1, 20.0, 25)  # start b(t) at 0.1 until 20. 25 arrows allong x-axis
c_vec = np.linspace(0.1, 16.0, 22)   # start c(t) at 0.1 until 16. 22 arrows along y-axis
B, C = np.meshgrid(b_vec, c_vec)    # combine the vectors into matrices B and C

dB = -k1 * B * psi(C) + k3
dC = -k1 * B * psi(C) + k2 * (cs - C)

# Normalize vector lengths for uniform arrow visualization
N = np.hypot(dB, dC)
N[N == 0] = 1.0
dB_norm = dB / N
dC_norm = dC / N

# 4. Nullclines formulation
# b-nullcline: db/dt = 0  =>  b = [k3 * (c + kappa)] / (k1 * c)
c_grid_bnull = np.linspace(0.15, 16.0, 500)     # generates 500 uniformly distributed values for the oxygen ([0.15, 16])
b_null_b = (k3 * (c_grid_bnull + kappa)) / (k1 * c_grid_bnull)   # callculates b*(t) for the different c

# c-nullcline: dc/dt = 0  =>  b = [k2 * (cs - c) * (kappa + c)] / (k1 * c)
c_grid_cnull = np.linspace(0.15, cs, 500)        # generate oxygen values ([0.15, cs])
c_null_b = (k2 * (cs - c_grid_cnull) * (kappa + c_grid_cnull)) / (
    k1 * c_grid_cnull)                           # callculates the other b*(t) for different c

# 5. Plotting
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

# Direction field (Vector Flow)
ax.quiver(
    B,
    C,
    dB_norm,
    dC_norm,
    color='gray',
    alpha=0.45,
    width=0.003,
    headwidth=3.5,
    label='Direction Field',
)

# Solution Trajectories
t_span = np.linspace(0, 15, 500)
# initial conditions (b0, c0)
initial_conditions = [
    (20.0, 15.0),
    (20.0, 10.0),
    (20.0, 1.0),
    (15.0, 0.2),
    (8.0, 0.1),
    (1.0, 0.1),
    (0.2, 12.0),
    (0.2, 16.0),
    (10.0, 16.0),
]

# use the initial conditions to callculate the sollutions over the time span (t_span) using runge kutta
for idx, (b0, c0) in enumerate(initial_conditions):
    sol = rk4_solve(nonlinear_streeter_phelps, [b0, c0], t_span)
    valid = (
      # sol[:,0] are solutions of b(t) and sol[:,1] for c(t)
      (sol[:, 0] >= 0)
      & (sol[:, 0] <= 21)   # b(t) in [0, 21]
      & (sol[:, 1] >= 0)
      & (sol[:, 1] <= 17)   # c(t) in [0, 17]
    )
    b_traj = sol[valid, 0]   
    c_traj = sol[valid, 1]

    # label the trajectories only in the first run of the loop. Otherwise the box would have a sign "Trajectories" fo each index
    label_traj = 'Trajectories' if idx == 0 else ''

    # draw the trajectories with the initial conditions as black graphs
    ax.plot(
      b_traj,
      c_traj,
      color='black',
      alpha=0.75,
      linewidth=1.5,
      label=label_traj,
    )

    # Direction arrows along trajectories that are longer than 30 datapoints
    if len(b_traj) > 30:
        mid = 20       # set the arrow at the 20th datapoint position
    
        # drawing the arrow between neighbouring datapoints
        ax.annotate(
            '',
            xy=(b_traj[mid + 1], c_traj[mid + 1]),
            xytext=(b_traj[mid], c_traj[mid]),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
        )

# Saturation line and Nullclines
ax.axhline(
    cs,
    color='green',
    linestyle='-',
    linewidth=1.8,
    label=f'Saturation Concentration $c = c_s$ ({cs})',
)

ax.plot(
    b_null_b, 
    c_grid_bnull, 
    color='darkorange', 
    linewidth=2.0, 
    label='b-nullcline'
)

ax.plot(
    c_null_b, 
    c_grid_cnull, 
    color='navy', 
    linewidth=2.0, 
    label='c-nullcline')

# Equilibrium Point (Red Dot)
ax.plot(
    b_star,
    c_star,
    'ro',
    markersize=9,
    zorder=6,
    label=f'Equilibrium $(b^*, c^*) = ({b_star:.2f}, {c_star:.2f})$',
)

# Plot Formatting
ax.set_xlim(0.0, 21.0)
ax.set_ylim(0.0, 17.0)
ax.set_xlabel('BOD Concentration b [mg/L]', fontsize=11)
ax.set_ylabel('Dissolved Oxygen Concentration c [mg/L]', fontsize=11)
ax.set_title(
    'Phase Portrait of the Nonlinear Streeter-Phelps Model',
    fontsize=13,
    fontweight='bold',
    pad=12,
)

ax.grid(True, linestyle=':', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)

plt.tight_layout(pad=1.5)
plt.show()