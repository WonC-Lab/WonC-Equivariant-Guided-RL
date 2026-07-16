import json
import torch
import numpy as np
from robotic_env import RoboticArm3DEnv
from equivariant_models_3d import VNEquivariantPolicyValueNet, StandardMLPPolicyValueNet
from train_robotic import train_agent

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def evaluate_rotated_policy(model, env, angle_deg, num_episodes=30):
    """
    Evaluates the model on test tasks rigidly rotated around the Z-axis by angle_deg.
    """
    model.eval()
    theta_rad = np.radians(angle_deg)
    
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R_z = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    successes = []
    with torch.no_grad():
        for _ in range(num_episodes):
            obs = env.reset(restrict_sector=True)
            
            env.target = np.dot(R_z, env.target)
            env.obstacle = np.dot(R_z, env.obstacle)
            env.theta[0] = env.theta[0] + theta_rad
            
            obs = env._get_obs()
            done = False
            
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                act_mean, _ = model(obs_t)
                action = act_mean.squeeze(0)
                
                act_norm = torch.norm(action)
                if act_norm > 1.0:
                    action = action * (1.0 / act_norm)
                    
                obs, reward, done, info = env.step(action.numpy())
                
            successes.append(float(info["success"]))
            
    return np.mean(successes) * 100.0

def evaluate_robustness(model, env, obstacle_radius, link3_scale, num_episodes=30):
    """
    Evaluates the model on target configurations in training sector (x > 0)
    with perturbed environmental variables (link length and obstacle size).
    """
    model.eval()
    orig_l3 = env.l3
    
    # Perturb link 3 length
    env.l3 = orig_l3 * link3_scale
    
    successes = []
    with torch.no_grad():
        for _ in range(num_episodes):
            obs = env.reset(restrict_sector=True)
            # Override obstacle size after environment reset
            env.obstacle_radius = obstacle_radius
            
            done = False
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                act_mean, _ = model(obs_t)
                action = act_mean.squeeze(0)
                
                act_norm = torch.norm(action)
                if act_norm > 1.0:
                    action = action * (1.0 / act_norm)
                    
                obs, reward, done, info = env.step(action.numpy())
                
            successes.append(float(info["success"]))
            
    # Restore original link length
    env.l3 = orig_l3
    
    return np.mean(successes) * 100.0

def main():
    print("==================================================")
    print("Starting SO(3) Multi-Seed Obstacle & Robustness Benchmarks")
    print("==================================================")
    
    num_epochs = 120
    seeds = [42, 100, 2026]
    results = {}
    
    configs = {
        "Proposed (VN + IK Prior)": {
            "model_cls": VNEquivariantPolicyValueNet,
            "beta": 2.0,
            "decay": 0.99,
            "beta_min": 0.5
        },
        "Standard MLP + IK Prior": {
            "model_cls": StandardMLPPolicyValueNet,
            "beta": 2.0,
            "decay": 0.99,
            "beta_min": 0.5
        },
        "Vector Neurons (No Prior)": {
            "model_cls": VNEquivariantPolicyValueNet,
            "beta": 0.0,
            "decay": 1.0,
            "beta_min": 0.0
        },
        "Standard MLP (No Prior)": {
            "model_cls": StandardMLPPolicyValueNet,
            "beta": 0.0,
            "decay": 1.0,
            "beta_min": 0.0
        }
    }
    
    # Initialize structured results dictionary
    for name in configs.keys():
        results[name] = {
            "seed_histories": [],
            "zero_shot_rotations": {},
            "robustness_obstacles": {},
            "robustness_links": {}
        }
        
    rotation_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    obstacle_radii = [0.05, 0.08, 0.10, 0.12, 0.15]
    link_scales = [0.9, 1.0, 1.1]
    
    # Initialize keys for averaging across seeds later
    for name in configs.keys():
        for angle in rotation_angles:
            results[name]["zero_shot_rotations"][str(angle)] = []
        for rad in obstacle_radii:
            results[name]["robustness_obstacles"][str(rad)] = []
        for scale in link_scales:
            results[name]["robustness_links"][str(scale)] = []
            
    # Loop over seeds and configurations
    for seed in seeds:
        print(f"\n##################################################")
        print(f"   RUNNING BENCHMARK FOR SEED {seed}")
        print(f"##################################################")
        
        for name, conf in configs.items():
            print(f"\n[Training] {name} (Seed {seed}) in Restricted Hemisphere...")
            set_seeds(seed)
            env = RoboticArm3DEnv()
            model = conf["model_cls"]()
            
            history = train_agent(
                model, env, num_epochs=num_epochs, 
                beta_start=conf["beta"], beta_decay=conf["decay"], beta_min=conf["beta_min"],
                restrict_sector=True
            )
            
            # Save history for this seed
            results[name]["seed_histories"].append({
                "epoch_rewards": history["epoch_rewards"],
                "epoch_success_rates": history["epoch_success_rates"],
                "epoch_losses": history["epoch_losses"],
                "epoch_kl_losses": history["epoch_kl_losses"]
            })
            
            # 1. Evaluate Zero-Shot Rotation Sweep
            print(f"  Evaluating Rotation Sweep (Seed {seed})...")
            for angle in rotation_angles:
                rate = evaluate_rotated_policy(model, env, angle_deg=angle, num_episodes=30)
                results[name]["zero_shot_rotations"][str(angle)].append(rate)
                
            # 2. Evaluate Robustness: Obstacle Size Sweep
            print(f"  Evaluating Obstacle Size Sweep (Seed {seed})...")
            for rad in obstacle_radii:
                rate = evaluate_robustness(model, env, obstacle_radius=rad, link3_scale=1.0, num_episodes=30)
                results[name]["robustness_obstacles"][str(rad)].append(rate)
                
            # 3. Evaluate Robustness: Link 3 Scale Sweep
            print(f"  Evaluating Link Scale Sweep (Seed {seed})...")
            for scale in link_scales:
                rate = evaluate_robustness(model, env, obstacle_radius=0.10, link3_scale=scale, num_episodes=30)
                results[name]["robustness_links"][str(scale)].append(rate)
                
    # Save structured results
    output_path = "robotic_results_val.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nMulti-seed benchmarks completed! Results saved to {output_path}")

if __name__ == "__main__":
    main()
