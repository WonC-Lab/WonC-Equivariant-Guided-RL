import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from train_3d import train_robotic_agent
from models_3d import OctahedralRoboticNet

def smooth(y, window=25):
    box = np.ones(window)/window
    return np.convolve(y, box, mode='valid')

def get_returns_cached(name, model_class, num_episodes, mcts_searches, alpha, seed):
    json_path = f"checkpoints/returns_{name}.json"
    if os.path.exists(json_path):
        print(f"Loading cached training returns from {json_path}...")
        with open(json_path, "r") as f:
            return json.load(f)
            
    print(f"\n--- Training {name} (N_search={mcts_searches}, alpha={alpha}) ---")
    _, returns = train_robotic_agent(model_class, num_episodes=num_episodes, mcts_searches=mcts_searches, alpha=alpha, seed=seed)
    
    # Save immediately to disk
    os.makedirs("checkpoints", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(returns, f)
    print(f"Saved training returns to {json_path}.")
    return returns

import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=str, default=None, choices=[
        "nsearch15_alpha0.1", "nsearch5_alpha0.1", "nsearch30_alpha0.1",
        "nsearch15_alpha0.0", "nsearch15_alpha0.2"
    ])
    args = parser.parse_args()
    
    seed = 42
    num_episodes = 600
    
    if args.sweep is not None:
        # Run a single sweep
        torch.set_num_threads(2)
        if args.sweep == "nsearch15_alpha0.1":
            get_returns_cached("nsearch15_alpha0.1", OctahedralRoboticNet, num_episodes, 15, 0.1, seed)
        elif args.sweep == "nsearch5_alpha0.1":
            get_returns_cached("nsearch5_alpha0.1", OctahedralRoboticNet, num_episodes, 5, 0.1, seed)
        elif args.sweep == "nsearch30_alpha0.1":
            get_returns_cached("nsearch30_alpha0.1", OctahedralRoboticNet, num_episodes, 30, 0.1, seed)
        elif args.sweep == "nsearch15_alpha0.0":
            get_returns_cached("nsearch15_alpha0.0", OctahedralRoboticNet, num_episodes, 15, 0.0, seed)
        elif args.sweep == "nsearch15_alpha0.2":
            get_returns_cached("nsearch15_alpha0.2", OctahedralRoboticNet, num_episodes, 15, 0.2, seed)
        return
        
    print("=== Starting 3D Parameter Sensitivity Analysis Sweeps (Parallel Subprocesses) ===")
    
    sweeps = [
        "nsearch15_alpha0.1",
        "nsearch5_alpha0.1",
        "nsearch30_alpha0.1",
        "nsearch15_alpha0.0",
        "nsearch15_alpha0.2"
    ]
    
    processes = []
    for sweep in sweeps:
        # Launch ourselves with --sweep argument
        cmd = [sys.executable, "-u", "run_sensitivity_3d.py", "--sweep", sweep]
        p = subprocess.Popen(cmd)
        processes.append((sweep, p))
        print(f"Launched subprocess for sweep: {sweep} (PID: {p.pid})")
        
    print("Waiting for all sweeps to complete...")
    for sweep, p in processes:
        p.wait()
        if p.returncode != 0:
            print(f"Warning: Sweep {sweep} failed with exit code {p.returncode}")
        else:
            print(f"Sweep {sweep} completed successfully.")
            
    print("\n=== All Sweeps Completed. Loading results and plotting... ===")
    base_returns = get_returns_cached("nsearch15_alpha0.1", OctahedralRoboticNet, num_episodes, 15, 0.1, seed)
    n5_returns = get_returns_cached("nsearch5_alpha0.1", OctahedralRoboticNet, num_episodes, 5, 0.1, seed)
    n30_returns = get_returns_cached("nsearch30_alpha0.1", OctahedralRoboticNet, num_episodes, 30, 0.1, seed)
    a0_returns = get_returns_cached("nsearch15_alpha0.0", OctahedralRoboticNet, num_episodes, 15, 0.0, seed)
    a2_returns = get_returns_cached("nsearch15_alpha0.2", OctahedralRoboticNet, num_episodes, 15, 0.2, seed)
    
    print("\n=== Generating Sensitivity Curves Plot... ===")
    
    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    window = 30
    
    # Subplot 1: MCTS searches
    axs[0].plot(smooth(base_returns, window), label="Baseline ($N_{search}=15$)", color="royalblue", linewidth=2)
    axs[0].plot(smooth(n5_returns, window), label="Small Budget ($N_{search}=5$)", color="crimson", linewidth=2)
    axs[0].plot(smooth(n30_returns, window), label="Large Budget ($N_{search}=30$)", color="forestgreen", linewidth=2)
    axs[0].set_title("MCTS Budget Sensitivity (3D Robotic Arm)")
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel("Average Return (Smoothed)")
    axs[0].legend()
    axs[0].grid(True)
    
    # Subplot 2: Alpha scaling
    axs[1].plot(smooth(base_returns, window), label="Baseline ($\\alpha=0.1$)", color="royalblue", linewidth=2)
    axs[1].plot(smooth(a0_returns, window), label="No Curriculum ($\\alpha=0.0$)", color="crimson", linewidth=2)
    axs[1].plot(smooth(a2_returns, window), label="Heavy Regularizer ($\\alpha=0.2$)", color="purple", linewidth=2)
    axs[1].set_title("Dynamic KL Scale Sensitivity (3D Robotic Arm)")
    axs[1].set_xlabel("Episodes")
    axs[1].set_ylabel("Average Return (Smoothed)")
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plot_path = "sensitivity_comparison_3d.png"
    plt.savefig(plot_path)
    print(f"Saved sensitivity curves chart to {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
