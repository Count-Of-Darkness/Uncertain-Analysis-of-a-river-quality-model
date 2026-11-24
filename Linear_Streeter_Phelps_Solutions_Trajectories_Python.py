import numpy as np
import matplotlib.pyplot as plt

# 1. Modellparameter
k1 = 1.50       # BOD-decomposition [1/day]
k2 = 3.00       # Reaeration rate [1/day]
k3 = 2.00       # External pollution [mg/(L*day)]
cs = 9.50       # Oxygen saturation [mg/L]

# Steady States
b_star = k3 / k1            
c_star = cs - (k3 / k2)    

# 2. differentialequations
def streeter_phelps_system(y, t):
    b, c = y
    dbdt = k3 - k1 * b
    dcdt = -k1 * b + k2 * (cs - c)
    return np.array([dbdt, dcdt])     # returns the derivations as a vector

# fourth order runge kutta method with
def rk4_solve(sys_func, y0, t):
    y_sol = np.zeros((len(t), len(y0)))
    y_sol[0] = y0
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        y_curr = y_sol[i]
        k1_step = sys_func(y_curr, t[i])
        k2_step = sys_func(y_curr + 0.5 * dt * k1_step, t[i] + 0.5 * dt)
        k3_step = sys_func(y_curr + 0.5 * dt * k2_step, t[i] + 0.5 * dt)
        k4_step = sys_func(y_curr + dt * k3_step, t[i+1])
        y_sol[i+1] = y_curr + (dt / 6.0) * (k1_step + 2*k2_step + 2*k3_step + k4_step)
    return y_sol

# 3. Vektorfield in the Phaseplane
# we generate a 2D grid with B- and C- axis
b_vals = np.linspace(0, 25, 25)
c_vals = np.linspace(0, 12, 20)
B, C = np.meshgrid(b_vals, c_vals)

# Derivation in every gridpoint (the arrows)
dB = k3 - k1 * B
dC = -k1 * B + k2 * (cs - C)

# Standardization of the arrow length
N = np.hypot(dB, dC)
N[N == 0] = 1
dB_norm = dB / N
dC_norm = dC / N

# 4. plotting the graph
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

# Vektorfield
ax.quiver(
    B, 
    C, 
    dB_norm, 
    dC_norm, 
    color='skyblue', 
    alpha=0.6, 
    width=0.003, 
    headwidth=4, 
    label='Direction of flow (Vectorfield)')

# Trajektories
t = np.linspace(0, 10, 1000)
initial_conditions = [
    (20.0, 9.0),   # high pollution, high oxygen
    (22.0, 2.0),   # high BOD, low oxygen
    (5.0,  1.0),   # low BOD, extreme oxygen shortage
    (15.0, 11.5),  # over the saturation limit
    (1.14, 2.0)    # steady BOD, lack of oxygen
]

# we have 5 different initial points from which we determine the dark blue curves
for idx, (b0, c0) in enumerate(initial_conditions):
    sol = rk4_solve(streeter_phelps_system, [b0, c0], t)
    label = 'Trajectories' if idx == 0 else ""
    ax.plot(sol[:, 0], sol[:, 1], color='navy', linewidth=2.0, label=label)
    
    # starting points in those blue curves marked with a dot
    ax.plot(b0, c0, 'o', color='navy', markersize=5)  
    
    # set at half the distance an arrow onto the blue graphs
    mid_idx = 15
    ax.annotate('', xy=(sol[mid_idx+1, 0], sol[mid_idx+1, 1]), 
                xytext=(sol[mid_idx, 0], sol[mid_idx, 1]),
                arrowprops=dict(arrowstyle="->", color='navy', lw=2))

# 5. Nullklines and steadystate
ax.axvline(b_star, color='darkorange', linestyle='--', linewidth=2, 
           label=f'Steady State b* = {b_star:.2f} mg/L')
ax.axhline(c_star, color='purple', linestyle='--', linewidth=2, 
           label=f'Steady State c* = {c_star:.2f} mg/L')

ax.plot(b_star, c_star, 'ro', markersize=10, zorder=5, 
        label=f'Intersection (b*, c*) = ({b_star:.2f}, {c_star:.2f})')

# Beschriftung und Layout
ax.set_title('Phaseplaneportrait of the linear Streeter-Phelps-System', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('BOD-Concentration b [mg/L]', fontsize=12)
ax.set_ylabel('Dissolved oxygen c [mg/L]', fontsize=12)
ax.set_xlim(0, 25)
ax.set_ylim(0, 12.5)

ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.show()