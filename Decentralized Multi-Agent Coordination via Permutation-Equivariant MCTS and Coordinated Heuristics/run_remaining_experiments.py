"""
run_remaining_experiments.py
Runs only Exp 4 (Rc sensitivity), Exp 5 (grid transfer), and Exp 6 (sensitivity sweeps).
Exp 1-3 data already collected from previous run.
"""
import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_rigorous_validation as rv

def main():
    os.makedirs("results", exist_ok=True)

    import torch
    from equivariant_gnn import PermutationEquivariantGNN

    model = PermutationEquivariantGNN(grid_size=13, in_channels=3, d_model=128)
    model.load_state_dict(torch.load("models/multi_agent_model.pth", map_location="cpu"))
    model.eval()
    print("Model loaded OK\n")

    # ── Exp 4: Rc Sensitivity ──────────────────────────────────────────────
    rc_data = rv.run_rc_sensitivity(model)
    rv.plot_rc_sensitivity(rc_data)

    # ── Exp 5: Grid Transfer ───────────────────────────────────────────────
    grid_results = rv.run_grid_transfer(model)
    rv.plot_grid_transfer(grid_results)

    # ── Exp 6: Hyperparameter Sensitivity (FAST_MODE=True -> skip if slow) ─
    sens_data = rv.run_sensitivity_experiments(model)
    rv.plot_sensitivity(sens_data)

    # ── Save supplementary JSON ────────────────────────────────────────────
    # Load existing Exp 1-3 data
    existing = {}
    exp123_path = "results/academic_results_rigorous.json"
    if os.path.exists(exp123_path):
        with open(exp123_path) as f:
            existing = json.load(f)

    existing["rc_sensitivity"] = {
        "rc_values": [str(r) for r in rc_data["rc_values"]],
        "rc_labels": rc_data["rc_labels"],
        "results":   rc_data["results"],
    }
    existing["grid_transfer"] = {str(k): v for k, v in grid_results.items()}
    existing["sensitivity"] = {
        "nsearch": {"values": sens_data["nsearch"]["values"],
                    "results": sens_data["nsearch"]["results"]},
        "beta":    {"values": sens_data["beta"]["values"],
                    "results": sens_data["beta"]["results"]},
    }

    out_path = "results/academic_results_rigorous.json"
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=4, default=str)
    print(f"\nSaved full results -> {out_path}")
    print("=" * 50)
    print("  All remaining experiments complete.")
    print("  New files:")
    print("    results/rc_sensitivity.png")
    print("    results/grid_transfer.png")
    print("    results/nsearch_sensitivity.png")
    print("    results/beta_sensitivity.png")
    print("=" * 50)

if __name__ == "__main__":
    main()
