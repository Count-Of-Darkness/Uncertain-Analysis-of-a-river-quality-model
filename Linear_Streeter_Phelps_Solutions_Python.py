import matplotlib.pyplot as plt
import numpy as np

# 1. Parameterdefinition
k1 = 1.5       # BOD-Decompositionrate [1/Tag]
k3 = 2.0        # External Pollution [mg/(L*Tag)]
cs = 9.5        # Oxigensaturation limit [mg/L]
b0 = 20.0       # Initialcondition for BOD [mg/L]
c0 = 9.5        # Initialcondition for oxygen [mg/L]

# time-axis (in days)
t = np.linspace(0, 10, 1000)

# 2. solution for BOD 
# b(t) = k3/k1 + (b0 - k3/k1) * exp(-k1 * t)
b_star = k3 / k1 	# steady state b*
b_t = b_star + (b0 - b_star) * np.exp(-k1 * t)

# 3. solution for DO with different values for Reaerationsrate k2
k2_scenarios = {
    20.0: ('DO (k2 = 20.0 - healthy)', 'blue', '--'), # strong reaeration
    3.0:  ('DO (k2 = 3.0 - Hypoxic)', 'purple', ':'), 
    0.50: ('DO (k2 = 0.50 - Anoxic)', 'navy', '-.')    # weak reaeration
}

# Plot construction
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

# BOD-graph 
ax.plot(
  t, 
  b_t, 
  label=f'BOD b(t)', 
  color='goldenrod', 
  linewidth=2.5
)

# DO-graphs for the different k2
for k2, (label_text, color, style) in k2_scenarios.items():
    c_star = cs - (k3 / k2)        # steady state c*
    
# solution for DO
# c(t) consists out of 4 parts
term1 = -k3 / k2
term2 = ((-k1 * b0 + k3) / (k2 - k1)) * np.exp(-k1 * t)
term3 = cs
term4 = (c0 + (k3 / k2) + (k1 * b0 - k3) / (k2 - k1) - cs) * np.exp(-k2 * t)
c_t = term1 + term2 + term3 + term4
    
# Plot of the oxygen graph
ax.plot(
    t, 
    c_t, 
    label=f'{label_text} [c* = {c_star:.2f}]', 
    color=color, 
    linestyle=style, 
    linewidth=2.0
)
    
# Steady State Marker c*
ax.plot(
    t[-1],         # draw the marker at the very end of the graph  
    c_star, 
    marker='o',    # symbolize the steady state through a dot
    color=color, 
    markersize=8)

# Steady State Marker b* for BOD
ax.plot(
    t[-1],              # draw the marker at the very end of the graph
    b_star, 
    marker='o',   # symbolize the steady state through a dot
    color='cyan', 
    markeredgecolor='black',   # circle the dot by a black ring to differentiate it
    markersize=8, 
    label=f'Steady State b* = {b_star:.2f}')

# draw the physical limit of oxygen cs = 9.5
ax.axhline(
    cs, 
    color='green', 
    linestyle='-', 
    linewidth=1.5, 
    alpha=0.7, 
    label=f'saturation cs = {cs}')

# draw the anoxia threshold at c = 0
ax.axhline(
    0, 
    color='crimson', 
    linestyle='-', 
    linewidth=1.2, 
    alpha=0.7, 
    label='Anoxia-threshold (c = 0)')

# Diagrammdesign
ax.set_title('solutions of the linearen Streeter-Phelps-equation', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Time t [days]', fontsize=12)
ax.set_ylabel('Concentration [mg/L]', fontsize=12)
ax.set_xlim(0, 10)
ax.set_ylim(-3, 22)

ax.grid(True, linestyle=':', alpha=0.6)      # background grid on (dottet lines)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9) # box in the upper right corner

plt.tight_layout()
plt.show()