"""Prepare PDL1 inputs for real-filter controls or full-pipeline plumbing tests."""

import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--binder-length", type=int, default=65)
    parser.add_argument("--exercise-all-stages", action="store_true")
    args = parser.parse_args()
    if not 4 <= args.binder_length <= 500:
        parser.error("Binder length must be between 4 and 500")
    source = Path(__file__).resolve().parents[1]
    args.destination.mkdir(parents=True, exist_ok=True)
    settings = json.loads((source / "settings_target/PDL1.json").read_text())
    settings.update(design_path="/work/output", starting_pdb="/work/input/target.pdb",
                    binder_name="qualification", lengths=[args.binder_length] * 2,
                    number_of_final_designs=1)
    advanced = json.loads((source / "settings_advanced/default_4stage_multimer.json").read_text())
    advanced.update(max_trajectories=1, num_seqs=1, max_mpnn_sequences=1,
                    save_design_animations=False, save_design_trajectory_plots=False,
                    remove_unrelaxed_trajectory=False, remove_unrelaxed_complex=False,
                    remove_binder_monomer=False, af_params_dir="/models/af2")
    filters = json.loads((source / "settings_filters/default_filters.json").read_text())

    def disable(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "threshold":
                    value[key] = None
                else:
                    disable(child)

    if args.exercise_all_stages:
        disable(filters)
    for name, value in (("settings", settings), ("advanced", advanced), ("filters", filters)):
        (args.destination / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    shutil.copyfile(source / "example/PDL1.pdb", args.destination / "target.pdb")
    (args.destination / "qualification.json").write_text(json.dumps({
        "purpose": "pipeline plumbing" if args.exercise_all_stages else "normal filters",
        "thresholdsDisabled": args.exercise_all_stages, "binderLength": args.binder_length,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
