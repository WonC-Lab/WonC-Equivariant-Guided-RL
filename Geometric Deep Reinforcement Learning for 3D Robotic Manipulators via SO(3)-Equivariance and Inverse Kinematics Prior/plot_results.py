import json
import matplotlib.pyplot as plt
import numpy as np

def smooth(y, window):
    box = np.ones(window)/window
    return np.convolve(y, box, mode='valid')

def main():
    # Load results
    results_path = "robotic_results_val.json"
    try:
        with open(results_path, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: {results_path} not found. Please run the experiments script first.")
        return

    # Configuration styles
    colors = {
        "Proposed (VN + IK Prior)": "#E63946",       # Red
        "Standard MLP + IK Prior": "#457B9D",        # Blue
        "Vector Neurons (No Prior)": "#F1A73A",      # Orange
        "Standard MLP (No Prior)": "#8D99AE"         # Gray
    }
    
    linestyles = {
        "Proposed (VN + IK Prior)": "-",
        "Standard MLP + IK Prior": "--",
        "Vector Neurons (No Prior)": "-.",
        "Standard MLP (No Prior)": ":"
    }

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Extract training metrics epochs
    sample_history = next(iter(results.values()))["seed_histories"][0]
    num_epochs = len(sample_history["epoch_rewards"])
    x_axis = np.arange(1, num_epochs + 1)
    window_size = 5
    x_smooth = x_axis[window_size-1:]

    # ----------------------------------------------------
    # Plot 1: Sample Efficiency with 3D Obstacles (with Shaded Std Dev)
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for label, history in results.items():
        histories = history["seed_histories"]
        
        # Stack histories across seeds: shape (num_seeds, num_epochs)
        rewards_stacked = np.array([h["epoch_rewards"] for h in histories])
        success_stacked = np.array([h["epoch_success_rates"] for h in histories])
        
        # Calculate mean and std dev across seeds
        mean_rewards = np.mean(rewards_stacked, axis=0)
        std_rewards = np.std(rewards_stacked, axis=0)
        mean_success = np.mean(success_stacked, axis=0)
        std_success = np.std(success_stacked, axis=0)
        
        # Smooth curves
        smooth_mean_rewards = smooth(mean_rewards, window_size)
        smooth_std_rewards = smooth(std_rewards, window_size)
        smooth_mean_success = smooth(mean_success, window_size)
        smooth_std_success = smooth(std_success, window_size)
        
        # Plot rewards
        ax1.plot(
            x_smooth, smooth_mean_rewards, 
            label=label, 
            color=colors.get(label), 
            linestyle=linestyles.get(label),
            linewidth=2.5 if "Proposed" in label else 2.0
        )
        ax1.fill_between(
            x_smooth, 
            smooth_mean_rewards - smooth_std_rewards, 
            smooth_mean_rewards + smooth_std_rewards, 
            color=colors.get(label), 
            alpha=0.12
        )
        
        # Plot success rates
        ax2.plot(
            x_smooth, smooth_mean_success, 
            label=label, 
            color=colors.get(label), 
            linestyle=linestyles.get(label),
            linewidth=2.5 if "Proposed" in label else 2.0
        )
        ax2.fill_between(
            x_smooth, 
            smooth_mean_success - smooth_std_success, 
            smooth_mean_success + smooth_std_success, 
            color=colors.get(label), 
            alpha=0.12
        )

    ax1.set_title("Training Reward Curve (Mean $\pm$ SD, 3 Seeds)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("Average Episode Reward (Smoothed)", fontsize=11)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

    ax2.set_title("Training Success Rate (Mean $\pm$ SD, 3 Seeds)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epochs", fontsize=11)
    ax2.set_ylabel("Success Rate (%) (Smoothed)", fontsize=11)
    ax2.set_ylim(-5, 105)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    
    plt.suptitle("Sample Efficiency and Success Rates in Obstacle Workspace", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plot_efficiency_path = "sample_efficiency_obstacles.png"
    plt.savefig(plot_efficiency_path, dpi=300)
    print(f"Saved obstacle sample efficiency curve to: {plot_efficiency_path}")
    plt.close()

    # ----------------------------------------------------
    # Plot 2: Zero-Shot Rotation Generalization (Mean $\pm$ SD Error Bars)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    for label, history in results.items():
        zero_shot_data = history["zero_shot_rotations"]
        angles = sorted([int(k) for k in zero_shot_data.keys()])
        
        # Calculate mean and std dev at each rotation angle across seeds
        mean_rates = []
        std_rates = []
        for a in angles:
            rates = zero_shot_data[str(a)]
            mean_rates.append(np.mean(rates))
            std_rates.append(np.std(rates))
            
        plt.errorbar(
            angles, mean_rates, 
            yerr=std_rates,
            label=label, 
            color=colors.get(label), 
            linestyle=linestyles.get(label),
            marker='o',
            capsize=4,
            linewidth=2.0,
            markersize=5,
            alpha=0.95
        )

    # Shading the unseen sector (targets trained in [-90, 90], so [90, 270] is completely unseen)
    plt.axvspan(90, 270, color='#E63946', alpha=0.08, label='Unseen Rotation Sector ($x_{target} < 0$)')

    plt.title("Zero-Shot Rotation Generalization (Mean $\pm$ SD, 3 Seeds)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Rotation Angle (Degrees)", fontsize=12)
    plt.ylabel("Test Success Rate (%)", fontsize=12)
    plt.xlim(-10, 340)
    plt.ylim(-5, 105)
    plt.xticks(np.arange(0, 360, 30))
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10, loc='lower left')
    plt.tight_layout()
    
    plot_gen_path = "zeroshot_rotation_generalization.png"
    plt.savefig(plot_gen_path, dpi=300)
    print(f"Saved zero-shot rotation generalization curve to: {plot_gen_path}")
    plt.close()

if __name__ == "__main__":
    main()
