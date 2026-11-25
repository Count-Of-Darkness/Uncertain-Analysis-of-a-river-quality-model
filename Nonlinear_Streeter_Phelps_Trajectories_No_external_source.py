import matplotlib.pyplot as plt
import numpy as np

# 1. Model parameters
k1 = 1.50  # Maximum BOD degradation rate [1/day]
k2 = 3.00  # Reaeration rate [1/day]
kappa = 1.00  # Monod half-saturation constant [mg/L]
cs = 9.50  # Dissolved oxygen saturation concentration [mg/L]

# Unique stable equilibrium point (b*, c*) = (0, cs)
b_star = 0.0
c_star = cs


# Monod saturation function Psi(c) = c / (kappa + c)
def psi(c):
  c_pos = np.maximum(c, 0.0)
  return c_pos / (kappa + c_pos)


# System of ODEs without external source
def nonlinear_system(y, t):
  b, c = y
  dbdt = -k1 * b * psi(c)
  dcdt = -k1 * b * psi(c) + k2 * (cs - c)
  return np.array([dbdt, dcdt])


# Explicit Runge-Kutta 4th Order Integrator
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


# 2. Grid for vector field
b_vec = np.linspace(0.01, 20.0, 22)
c_vec = np.linspace(0.01, 15.0, 20)
B, C = np.meshgrid(b_vec, c_vec)

dB = -k1 * B * psi(C)
dC = -k1 * B * psi(C) + k2 * (cs - C)

# Normalize vector lengths for uniform arrow visualization
N = np.hypot(dB, dC)
N[N == 0] = 1.0
dB_norm = dB / N
dC_norm = dC / N

# 3. Nullcline equations
# b-nullclines: db/dt = 0 => b = 0 and c = 0
# c-nullcline: dc/dt = 0 => b = k2 * (cs - c) * (kappa + c) / (k1 * c)
c_grid = np.linspace(0.1, cs, 400)
b_cnull = (k2 * (cs - c_grid) * (kappa + c_grid)) / (k1 * c_grid)

# 4. Setup figure
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

# Direction field
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

# Sample Trajectories
t_span = np.linspace(0, 15, 400)
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

for idx, (b0, c0) in enumerate(initial_conditions):
  sol = rk4_solve(nonlinear_system, [b0, c0], t_span)
  b_traj = sol[:, 0]
  c_traj = sol[:, 1]

  # Keep trajectories within view limits
  valid = (
      (b_traj >= -0.2)
      & (b_traj <= 21)
      & (c_traj >= -0.2)
      & (c_traj <= 16)
  )
  b_traj = b_traj[valid]
  c_traj = c_traj[valid]

  label_traj = 'Trajectories' if idx == 0 else ''
  ax.plot(
      b_traj,
      c_traj,
      color='black',
      alpha=0.8,
      linewidth=1.5,
      label=label_traj,
  )

  # Direction arrows on trajectory
  if len(b_traj) > 30:
    mid = 25
    ax.annotate(
        '',
        xy=(b_traj[mid + 1], c_traj[mid + 1]),
        xytext=(b_traj[mid], c_traj[mid]),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
    )

# 5. Nullclines and Saturation line
ax.axhline(
    cs,
    color='green',
    linestyle='-',
    linewidth=1.8,
    label=f'Saturation $c = c_s$ ({cs})',
)

ax.axvline(
  0, 
  color='darkorange', 
  linewidth=2.0, 
  label='b-nullcline ($b=0$)'
  )

ax.axhline(
  0, 
  color='darkorange', 
  linewidth=2.0
  )

ax.plot(
  b_cnull, 
  c_grid, 
  color='navy', 
  linewidth=2.0, 
  label='c-nullcline'
  )

# 6. Equilibrium Point (Red Dot)
ax.plot(
    b_star,
    c_star,
    'ro',
    markersize=9,
    zorder=6,
    label=f'Equilibrium $(b^*, c^*) = ({b_star:.1f}, {c_star:.1f})$',
)

# Axis limits and labels
ax.set_xlim(0.0, 21.0)
ax.set_ylim(0.0, 16.0)
ax.set_xlabel('BOD Concentration b [mg/L]', fontsize=11)
ax.set_ylabel('Dissolved Oxygen Concentration c [mg/L]', fontsize=11)
ax.set_title(
    'Phase Portrait of Nonlinear Streeter-Phelps without External Source',
    fontsize=12,
    fontweight='bold',
    pad=12,
)

ax.grid(True, linestyle=':', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)

plt.tight_layout(pad=1.5)
plt.show()