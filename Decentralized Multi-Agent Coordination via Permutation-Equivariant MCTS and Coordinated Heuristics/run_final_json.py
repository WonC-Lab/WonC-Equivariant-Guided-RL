"""
run_final_json.py
Recovery script: Runs only Exp 6 (sensitivity sweeps), uses completed data of Exp 1-5,
saves all plots, and writes final academic_results_rigorous.json.
"""
import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from equivariant_gnn import PermutationEquivariantGNN
import run_rigorous_validation as rv

# -------------------------------------------------------------------------
# Hardcoded completed results from task-346.log (N=50)
# -------------------------------------------------------------------------
comparison_results = {
    "mcts": [
        {"success_rate": 0.94, "std": 0.237, "avg_d_min": 5.401, "theorem4_bound": 0.6856},
        {"success_rate": 0.80, "std": 0.400, "avg_d_min": 3.946, "theorem4_bound": 3.8536},
        {"success_rate": 0.82, "std": 0.384, "avg_d_min": 3.537, "theorem4_bound": 9.5896},
        {"success_rate": 0.68, "std": 0.466, "avg_d_min": 2.939, "theorem4_bound": 23.1563},
        {"success_rate": 0.58, "std": 0.494, "avg_d_min": 2.657, "theorem4_bound": 42.4837},
        {"success_rate": 0.30, "std": 0.458, "avg_d_min": 2.366, "theorem4_bound": 100.0717},
    ],
    "orca": [
        {"success_rate": 0.94, "std": 0.237, "avg_d_min": 5.53, "theorem4_bound": 0.653},
        {"success_rate": 0.88, "std": 0.325, "avg_d_min": 4.32, "theorem4_bound": 3.208},
        {"success_rate": 0.84, "std": 0.367, "avg_d_min": 3.71, "theorem4_bound": 8.698},
        {"success_rate": 0.64, "std": 0.480, "avg_d_min": 3.30, "theorem4_bound": 18.395},
        {"success_rate": 0.38, "std": 0.485, "avg_d_min": 3.14, "theorem4_bound": 30.393},
        {"success_rate": 0.42, "std": 0.494, "avg_d_min": 2.77, "theorem4_bound": 72.866},
    ],
    "gnn_lookahead": [
        {"success_rate": 0.0, "std": 0.0, "avg_d_min": 6.37, "theorem4_bound": 0.493},
        {"success_rate": 0.0, "std": 0.0, "avg_d_min": 4.73, "theorem4_bound": 2.677},
        {"success_rate": 0.0, "std": 0.0, "avg_d_min": 4.36, "theorem4_bound": 6.320},
        {"success_rate": 0.0, "std": 0.0, "avg_d_min": 4.14, "theorem4_bound": 11.691},
        {"success_rate": 0.0, "std": 0.0, "avg_d_min": 3.85, "theorem4_bound": 20.287},
        {"success_rate": 0.0, "std": 0.0, "avg_d_min": 2.45, "theorem4_bound": 93.224},
    ],
    "gnn": [
        {"success_rate": 0.0, "std": 0.0, "avg_d_min": 5.41, "theorem4_bound": 0.684},
        {"success_rate": 0.0, "std": 0.0, "avg_d_min": 4.34, "theorem4_bound": 3.185},
        {"success_rate": 0.0, "std": 0.0, "avg_d_min": 3.71, "theorem4_bound": 8.704},
        {"success_rate": 0.0, "std": 0.0, "avg_d_min": 1.64, "theorem4_bound": 74.549},
        {"success_rate": 0.0, "std": 0.0, "avg_d_min": 1.47, "theorem4_bound": 139.427},
        {"success_rate": 0.0, "std": 0.0, "avg_d_min": 1.02, "theorem4_bound": 541.029},
    ],
    "heuristic": [
        {"success_rate": 0.92, "std": 0.271, "avg_d_min": 5.41, "theorem4_bound": 0.684},
        {"success_rate": 0.82, "std": 0.384, "avg_d_min": 3.91, "theorem4_bound": 3.924},
        {"success_rate": 0.76, "std": 0.427, "avg_d_min": 3.46, "theorem4_bound": 10.039},
        {"success_rate": 0.62, "std": 0.485, "avg_d_min": 2.86, "theorem4_bound": 24.456},
        {"success_rate": 0.42, "std": 0.494, "avg_d_min": 2.58, "theorem4_bound": 45.107},
        {"success_rate": 0.38, "std": 0.485, "avg_d_min": 2.42, "theorem4_bound": 95.500},
    ]
}

robustness_results = {
    "default": [
        {"success_rate": 0.94, "std": 0.237},
        {"success_rate": 0.80, "std": 0.400},
        {"success_rate": 0.82, "std": 0.384},
        {"success_rate": 0.68, "std": 0.466},
        {"success_rate": 0.58, "std": 0.494},
        {"success_rate": 0.30, "std": 0.458},
    ],
    "empty": [
        {"success_rate": 1.00, "std": 0.0},
        {"success_rate": 0.96, "std": 0.196},
        {"success_rate": 0.90, "std": 0.300},
        {"success_rate": 0.72, "std": 0.449},
        {"success_rate": 0.74, "std": 0.439},
        {"success_rate": 0.66, "std": 0.474},
    ],
    "random": [
        {"success_rate": 0.98, "std": 0.140},
        {"success_rate": 0.92, "std": 0.271},
        {"success_rate": 0.84, "std": 0.367},
        {"success_rate": 0.80, "std": 0.400},
        {"success_rate": 0.80, "std": 0.400},
        {"success_rate": 0.56, "std": 0.496},
    ]
}

theorem4_table = [
    {"M": 2, "avg_d_min": 5.401, "bound": 0.6856, "success_rate": 0.94, "failure": 0.06},
    {"M": 3, "avg_d_min": 3.946, "bound": 3.8536, "success_rate": 0.80, "failure": 0.20},
    {"M": 4, "avg_d_min": 3.537, "bound": 9.5896, "success_rate": 0.82, "failure": 0.18},
    {"M": 5, "avg_d_min": 2.939, "bound": 23.1563, "success_rate": 0.68, "failure": 0.32},
    {"M": 6, "avg_d_min": 2.657, "bound": 42.4837, "success_rate": 0.58, "failure": 0.42},
    {"M": 8, "avg_d_min": 2.366, "bound": 100.0717, "success_rate": 0.30, "failure": 0.70},
]

rc_data = {
    "rc_values": [3, 6, None],
    "rc_labels": ["Rc = 3 cells", "Rc = 6 cells", "Rc = [inf] (full)"],
    "results": [
        {"success_rate": 0.94, "std": 0.237},
        {"success_rate": 0.94, "std": 0.237},
        {"success_rate": 0.94, "std": 0.237},
    ]
}

grid_results = {
    13: {"success_rate": 0.84, "std": 0.367},
    20: {"success_rate": 0.94, "std": 0.237},
}

def main():
    os.makedirs("results", exist_ok=True)

    # 1. Load GNN Model for Exp 6
    model = PermutationEquivariantGNN(grid_size=13, in_channels=3, d_model=128)
    model.load_state_dict(torch.load("models/multi_agent_model.pth", map_location="cpu"))
    model.eval()
    print("Model loaded successfully.\n")

    # 2. Run Exp 6 (N=50, sweeps)
    print("=======================================================")
    print("  Exp 6 -- Hyperparameter Sensitivity (M=4)")
    print("=======================================================")
    sens_data = rv.run_sensitivity_experiments(model)
    rv.plot_sensitivity(sens_data)

    # 3. Re-save Exp 1-5 plots (just in case they need to be updated)
    rv.plot_main_comparison(comparison_results)
    rv.plot_robustness(robustness_results)
    rv.plot_theorem4(theorem4_table)
    rv.plot_rc_sensitivity(rc_data)
    rv.plot_grid_transfer(grid_results)

    # 4. Save everything to final JSON
    all_results = {
        "config": {
            "num_mcts_episodes": 50,
            "num_fast_episodes": 50,
            "agent_counts":      rv.AGENT_COUNTS,
            "mcts_searches":     rv.MCTS_SEARCHES_EVAL,
            "beta":              rv.HEURISTIC_BETA,
            "theorem4_C":        rv.THEOREM4_C,
            "theorem4_alpha":    rv.THEOREM4_ALPHA,
        },
        "comparison":   comparison_results,
        "robustness":   robustness_results,
        "theorem4":     theorem4_table,
        "rc_sensitivity": {
            "rc_values": [str(r) for r in rc_data["rc_values"]],
            "results":   rc_data["results"],
        },
        "grid_transfer": {str(k): v for k, v in grid_results.items()},
        "sensitivity": {
            "nsearch": {
                "values":  sens_data["nsearch"]["values"],
                "results": sens_data["nsearch"]["results"],
            },
            "beta": {
                "values":  sens_data["beta"]["values"],
                "results": sens_data["beta"]["results"],
            },
        },
    }

    json_path = "results/academic_results_rigorous.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4, default=str)

    print("\n" + "=" * 56)
    print("  Recovery & Complete Validation Run Success!")
    print("=" * 56)
    print("  All results compiled & saved successfully.")
    print("  Output JSON: results/academic_results_rigorous.json")
    print("=" * 56 + "\n")

if __name__ == "__main__":
    main()
