import matplotlib.pyplot as plt
import numpy as np

# Time axis: t in days (0 to 100 days)
# np.linspace(start, end, amount (1000 values between 0 and 100))
t = np.linspace(0, 100, 1000)

# Base parameters (k3 = 0)
# Initial BOD concentration [mg/L]
b0 = 20.0  

# Parameter variation for BOD degradation rate k1 [1/day]
k1_values = [0.02, 0.05, 0.10, 0.30, 0.50]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Figure setup 
# width: 9 inches; hight: 6 inches; resolution: 150 
fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

# Plot b(t) curves for each k1 value
# connects values of k1 and the colors
for k1, color in zip(k1_values, colors):
  b_t = b0 * np.exp(-k1 * t)
  ax.plot(
      t,   	# x-axis
      b_t, 	# y-axis
      label=f'$k_1 = {k1:.2f}$ day$^{{-1}}$',		# plain text: k1=...day^(-1)
      color=color,					# gives it the correct color
      linewidth=2.0,					# somewhat thicker line for the graph
  )

# Setting the titel
ax.set_title(
    '$b(t)$ functions for Varying $k_1$',
    fontsize=13,			# font size
    fontweight='bold',			# bold letters
    pad=12,				# 12 pixel
)

# labeling the axes
ax.set_xlabel('Time $t$ [Days]', fontsize=11)		# x-axis labeling in letter size 11
ax.set_ylabel('BOD Concentration $b(t)$ [mg/L]', fontsize=11) # y-axis labeling in letter size 11
ax.set_xlim(0, 100)						# x-achis from 0 to 100
ax.set_ylim(0, 22)						# y-axis from 0 to 22
ax.grid(True, linestyle=':', alpha=0.6)				# grid: visible; dotted grid; slightly transparent

# Spine cleanup; removing upper and right axis 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# show the legend on the upper right corner for the parameters k1
ax.legend(loc='upper right', fontsize=9.0, framealpha=0.95)

# adjusting the labels so that they don't overlap
plt.tight_layout(pad=1.5)
plt.show()