import numpy as np
import matplotlib.pyplot as plt

num_sweeps = 1000


def create_schedules(num_sweeps):
    """Create all beta schedule types."""
    schedules = {}
    
    # Linear schedules
    schedules['Linear'] = np.linspace(0, 1, num=num_sweeps)
    schedules['Linear Inverse'] = np.linspace(1, 0, num=num_sweeps)
    
    # Geometric schedules
    schedules['Geometric'] = np.geomspace(1e-6, 1, num=num_sweeps)
    schedules['Geometric Inverse'] = np.geomspace(1, 1e-6, num=num_sweeps)
    
    # Power schedules
    schedules['Power (1/3)'] = np.power(np.linspace(0, 1, num_sweeps), 1/3)
    schedules['Power (1/3) Inverse'] = np.power(np.linspace(1, 0, num_sweeps), 1/3)
    schedules['Power (1/2)'] = np.power(np.linspace(0, 1, num_sweeps), 1/2)
    schedules['Power (1/2) Inverse'] = np.power(np.linspace(1, 0, num_sweeps), 1/2)
    schedules['Power (2)'] = np.power(np.linspace(0, 1, num_sweeps), 2)
    schedules['Power (2) Inverse'] = np.power(np.linspace(1, 0, num_sweeps), 2)
    schedules['Power (3)'] = np.power(np.linspace(0, 1, num_sweeps), 3)
    schedules['Power (3) Inverse'] = np.power(np.linspace(1, 0, num_sweeps), 3)
    
    # Trigonometric schedules
    schedules['Trigonometric'] = np.sin(np.pi / 2 * np.linspace(0, 1, num_sweeps))**2
    schedules['Trigonometric Inverse'] = np.sin(np.pi / 2 * np.linspace(1, 0, num_sweeps))**2
    
    # Sigmoid schedules
    sigmoid = _sigmoid_schedule(num_sweeps)
    schedules['Sigmoid'] = sigmoid
    schedules['Sigmoid Inverse'] = 1 - sigmoid
    
    # Logarithmic schedules
    logarithmic = _logarithmic_schedule(num_sweeps)
    schedules['Logarithmic'] = logarithmic
    schedules['Logarithmic Inverse'] = 1 - logarithmic
    
    schedules['Exponential'] = _exponential_schedule(num_sweeps)
    schedules['Exponential Inverse'] = np.ones(num_sweeps)
    
    schedules['Exponential2'] = schedules['Power (2)']
    schedules['Exponential2 Inverse'] = np.ones(num_sweeps)
        
        # Cosine schedule
    
    # Fixed Hd with different Hp schedules
    hd = np.ones(num_sweeps)
    schedules['Fixed Hd, Linear Hp'] = hd
    schedules['Fixed Hd, Linear Hp (Forward)'] = schedules['Linear']
    
    schedules['Fixed Hd, Geometric Hp'] = hd
    schedules['Fixed Hd, Geometric Hp (Forward)'] = schedules['Geometric']
    
    schedules['Fixed Hd, Power Hp (1/2)'] = hd
    schedules['Fixed Hd, Power Hp (1/2) (Forward)'] = schedules['Power (1/2)']
    
    schedules['Fixed Hd, Power Hp (2)'] = hd
    schedules['Fixed Hd, Power Hp (2) (Forward)'] = schedules['Power (2)']
    
    schedules['Fixed Hd, Trigonometric Hp'] = hd
    schedules['Fixed Hd, Trigonometric Hp (Forward)'] = schedules['Trigonometric']
    
    schedules['Fixed Hd, Sigmoid Hp'] = hd
    schedules['Fixed Hd, Sigmoid Hp (Forward)'] = schedules['Sigmoid']
    
    schedules['Fixed Hd, Logarithmic Hp'] = hd
    schedules['Fixed Hd, Logarithmic Hp (Forward)'] = schedules['Logarithmic']
    
    return schedules


def _sigmoid_schedule(num_steps, k=10):
    """Generate sigmoid schedule."""
    x = np.linspace(0, 1, num_steps)
    s = 1 / (1 + np.exp(-k * (x - 0.5)))
    return (s - s.min()) / (s.max() - s.min())


def _logarithmic_schedule(num_steps):
    """Generate logarithmic schedule."""
    t = np.arange(1, num_steps + 1)
    s = np.log(t + 1)
    return (s - s.min()) / (s.max() - s.min())

def _exponential_schedule(num_steps, base=2.0):
    """Generate exponential schedule (fast cooling)."""
    t = np.linspace(0, 1, num_steps)
    s = (np.exp(base * t) - 1) / (np.exp(base) - 1)
    return (s - s.min()) / (s.max() - s.min())


def plot_schedules(schedules, num_sweeps):
    """Plot schedule pairs side by side."""
    # Define schedule pairs to plot
    schedule_pairs = [
        ('Linear', 'Linear Inverse'),
        ('Geometric', 'Geometric Inverse'),
        ('Trigonometric', 'Trigonometric Inverse'),
        ('Sigmoid', 'Sigmoid Inverse'),
        ('Logarithmic', 'Logarithmic Inverse'),
    ]
    
    # Exponential variants - plot together
    exponential_variants = [
        ('Exponential', 'Exponential Inverse'),
        ('Exponential2', 'Exponential2 Inverse'),
    ]
    
    # Fixed Hd pairs (Hd is constant, Hp varies)
    fixed_hd_pairs = [
        ('Fixed Hd, Linear Hp', 'Fixed Hd, Linear Hp (Forward)'),
        ('Fixed Hd, Geometric Hp', 'Fixed Hd, Geometric Hp (Forward)'),
        ('Fixed Hd, Trigonometric Hp', 'Fixed Hd, Trigonometric Hp (Forward)'),
        ('Fixed Hd, Sigmoid Hp', 'Fixed Hd, Sigmoid Hp (Forward)'),
        ('Fixed Hd, Logarithmic Hp', 'Fixed Hd, Logarithmic Hp (Forward)'),
    ]
    
    # Special handling for power schedules - plot all variants together
    power_variants = [
        ('Power (1/3)', 'Power (1/3) Inverse'),
        ('Power (1/2)', 'Power (1/2) Inverse'),
        ('Power (2)', 'Power (2) Inverse'),
        ('Power (3)', 'Power (3) Inverse'),
    ]
    
    fixed_hd_power = [
        ('Fixed Hd, Power Hp (1/2)', 'Fixed Hd, Power Hp (1/2) (Forward)'),
        ('Fixed Hd, Power Hp (2)', 'Fixed Hd, Power Hp (2) (Forward)'),
    ]
    
    num_pairs = len(schedule_pairs) + len(fixed_hd_pairs) + 3  # +3 for exponential, power and fixed_hd_power subplots
    cols = 3
    rows = (num_pairs + cols - 1) // cols
    
    size = 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * size, size * rows))
    axes = axes.flatten()
    
    ax_idx = 0
    
    # Plot regular schedule pairs
    for forward_name, inverse_name in schedule_pairs:
        axes[ax_idx].plot(schedules[forward_name], label=forward_name, linewidth=2)
        axes[ax_idx].plot(schedules[inverse_name], label=inverse_name, linewidth=2)
        axes[ax_idx].set_xlabel('Sweep')
        axes[ax_idx].set_ylabel('Beta')
        axes[ax_idx].set_title(forward_name)
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, which="both", ls="--")
        ax_idx += 1
    
    # Plot exponential variants together
    for forward_name, inverse_name in exponential_variants:
        axes[ax_idx].plot(schedules[forward_name], label=forward_name, linewidth=2)
        axes[ax_idx].plot(schedules[inverse_name], label=inverse_name, linewidth=2)
    axes[ax_idx].set_xlabel('Sweep')
    axes[ax_idx].set_ylabel('Beta')
    axes[ax_idx].set_title('Exponential Schedules')
    axes[ax_idx].legend()
    axes[ax_idx].grid(True, which="both", ls="--")
    ax_idx += 1
    
    # Plot all power variants together
    for forward_name, inverse_name in power_variants:
        axes[ax_idx].plot(schedules[forward_name], label=forward_name, linewidth=2)
        axes[ax_idx].plot(schedules[inverse_name], label=inverse_name, linewidth=2)
    axes[ax_idx].set_xlabel('Sweep')
    axes[ax_idx].set_ylabel('Beta')
    axes[ax_idx].set_title('Power Schedules')
    axes[ax_idx].legend()
    axes[ax_idx].grid(True, which="both", ls="--")
    ax_idx += 1
    
    # Plot fixed Hd schedules
    for hd_name, hp_forward in fixed_hd_pairs:
        axes[ax_idx].plot(schedules[hd_name], label='Fixed Hd', linewidth=2, linestyle='--')
        axes[ax_idx].plot(schedules[hp_forward], label=hp_forward.split('(')[1].split(')')[0], linewidth=2)
        axes[ax_idx].set_xlabel('Sweep')
        axes[ax_idx].set_ylabel('Beta')
        axes[ax_idx].set_title(hd_name)
        axes[ax_idx].legend(fontsize=9)
        axes[ax_idx].grid(True, which="both", ls="--")
        ax_idx += 1
    
    # Plot fixed Hd with power variants together
    for hd_name, hp_forward in fixed_hd_power:
        axes[ax_idx].plot(schedules[hd_name], label='Fixed Hd', linewidth=2, linestyle='--')
        axes[ax_idx].plot(schedules[hp_forward], label=hp_forward.split('(')[1].split(')')[0], linewidth=2)
    axes[ax_idx].set_xlabel('Sweep')
    axes[ax_idx].set_ylabel('Beta')
    axes[ax_idx].set_title('Fixed Hd with Power Hp')
    axes[ax_idx].legend(fontsize=9)
    axes[ax_idx].grid(True, which="both", ls="--")
    ax_idx += 1
    
    # Hide unused subplots
    for idx in range(ax_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    schedules = create_schedules(num_sweeps)
    plot_schedules(schedules, num_sweeps)

