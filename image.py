import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Setup the figure
fig, ax = plt.subplots(figsize=(10, 6))

# --- Data Generation ---
# Feasible region control points
x_feas = np.array([0, 1.5, 2.5, 4, 5])
y_feas = np.array([2.5, 2.2, 1.2, 3.2, 4.0])

# Infeasible region control points
x_infeas = np.array([5.1, 6, 7.5, 8.5, 9.5, 10.5])
y_infeas = np.array([3.9, 2.5, 0.5, 2.2, 2.2, 4.0])

# Smooth the curves using spline interpolation
x_smooth_feas = np.linspace(x_feas.min(), x_feas.max(), 200)
spl_feas = make_interp_spline(x_feas, y_feas, k=3)
y_smooth_feas = spl_feas(x_smooth_feas)

x_smooth_infeas = np.linspace(x_infeas.min(), x_infeas.max(), 200)
spl_infeas = make_interp_spline(x_infeas, y_infeas, k=3)
y_smooth_infeas = spl_infeas(x_smooth_infeas)

U = 3.5 # The penalty / shift value

# --- Plotting ---
# Plot axes manually to match sketch style
ax.plot([0, 11], [0, 0], color='black', linewidth=1.5) # X-axis
ax.plot([0, 0], [0, 8.5], color='black', linewidth=1.5) # Y-axis

# Draw the curves
ax.plot(x_smooth_feas, y_smooth_feas, color='black', linewidth=1.5)
ax.plot(x_smooth_infeas, y_smooth_infeas, color='black', linewidth=1.5)
ax.plot(x_smooth_infeas, y_smooth_infeas + U, color='black', linewidth=1.5)

# Dashed dividing line
ax.axvline(x=5, ymin=0, ymax=0.9, color='black', linestyle='--', linewidth=1.5)

# Text labels
ax.text(2.5, -0.4, 'Feasible region', color='tab:blue', fontsize=14, ha='center')
ax.text(7.75, -0.4, 'infeasible region', color='tab:red', fontsize=14, ha='center')

# Annotations (Arrows and U markers)
# Left side U marker
ax.annotate('', xy=(-0.5, 4), xytext=(-0.5, 0),
            arrowprops=dict(arrowstyle='|-|', lw=1.5, color='black'))
ax.text(-0.7, 2, '$U$', fontsize=14, va='center', ha='right')

# Right side +U arrow
ax.annotate('', xy=(11, 4 + U), xytext=(11, 4),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black', mutation_scale=20))
ax.text(11.2, 4 + U/2, '$+U$', fontsize=14, va='center')

# Grid and cleanup
ax.grid(True, which='both', linestyle='-', color='lightgray', alpha=0.5)
ax.set_xlim(-1, 12)
ax.set_ylim(-1, 9)
ax.axis('off') # Hide default matplotlib spines to keep the hand-drawn feel

plt.tight_layout()
plt.show()