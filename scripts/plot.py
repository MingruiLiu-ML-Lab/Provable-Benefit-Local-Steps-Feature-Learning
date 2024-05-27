""" Plot training results. """

import os
import glob
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


LINE_WIDTH = 1.5
ROUNDS_PER_EVAL = 138
IGNORE_METRICS = ["eval_elasped_times", "clip_ops", "local_train_losses", "local_train_accuracies"]
YLIM_WINDOW_START = 0.25
AVG_WINDOW_START = 0.9
FIGURE_SIZE = (10, 10)


plt.rcParams.update({'font.size': 14, 'font.family': "FreeSerif"})


def plot(family_dir, multi_seed=False):

    # Collect directory names containing results for each baseline.
    all_trial_dirs = {}
    if multi_seed:
        seed_paths = []
        for dir_name in os.listdir(family_dir):
            seed_path = os.path.join(family_dir, dir_name)
            if dir_name.startswith("seed_") and os.path.isdir(seed_path):
                seed_paths.append(seed_path)

        for i, seed_path in enumerate(seed_paths):
            for run_name in os.listdir(seed_path):
                run_path = os.path.join(seed_path, run_name)
                if os.path.isdir(run_path):
                    if run_name not in all_trial_dirs:
                        all_trial_dirs[run_name] = []
                    all_trial_dirs[run_name].append(run_path)

    else:
        for run_name in os.listdir(family_dir):
            run_path = os.path.join(family_dir, run_name)
            if os.path.isdir(run_path):
                all_trial_dirs[run_name] = [run_path]

    # Read results for each baseline.
    results = {}
    metrics = None
    for run_name, trial_dirs in all_trial_dirs.items():

        complete = True
        for trial_dir in trial_dirs:
            candidate_avg = glob.glob(os.path.join(trial_dir, "*.json"))
            candidate_avg = [filename for filename in candidate_avg if "Rank" not in filename]
            if len(candidate_avg) != 1:
                complete = False
                continue

            avg_filename = candidate_avg[0]
            with open(avg_filename, "r") as f:
                trial_results = json.load(f)

            if metrics is None:
                metrics = list(trial_results.keys())
            assert metrics == list(trial_results.keys())

            if run_name not in results:
                results[run_name] = {metric: [trial_results[metric]] for metric in metrics}
            else:
                for metric in metrics:
                    results[run_name][metric].append(trial_results[metric])

        if not complete:
            print(f"Incomplete results for {run_name}, skipping.")
            continue

        for metric in metrics:
            results[run_name][metric] = np.array(results[run_name][metric])

    if metrics is None:
        return

    # Plot results.
    names = sorted(list(results.keys()))
    for metric in metrics:
        if metric in IGNORE_METRICS:
            continue
        plt.clf()
        plt.figure(figsize=FIGURE_SIZE)
        y_min = None
        y_max = None
        for name in names:
            med = np.median(results[name][metric], axis=0)
            ub = np.max(results[name][metric], axis=0)
            lb = np.min(results[name][metric], axis=0)

            x = np.arange(len(med)) * ROUNDS_PER_EVAL
            plt.plot(x, med, label=name, linewidth=LINE_WIDTH)
            plt.fill_between(x, y1=ub, y2=lb, alpha=0.25)

            start = round(len(med) * YLIM_WINDOW_START)
            current_min = float(np.min(lb))
            current_max = float(np.max(ub))
            if y_min is None:
                y_min = current_min
                y_max = current_max
            else:
                y_min = min(y_min, current_min)
                y_max = max(y_max, current_max)

        plt.xlabel("Rounds")
        plt.ylabel(metric)
        plt.legend()
        y_min -= (y_max - y_min) * 0.05
        y_max += (y_max - y_min) * 0.05
        plt.ylim([y_min, y_max])

        plt.savefig(os.path.join(family_dir, f"{metric}.png"), bbox_inches="tight")

    # Print results.
    print(os.path.basename(family_dir))
    print(f"Average metric values at last evaluation:")
    for name in names:
        msg = f"{name}"
        for metric in metrics:
            if metric not in IGNORE_METRICS:
                mean = np.mean(results[name][metric], axis=0)
                top = np.max(results[name][metric], axis=0)
                window_mean = float(mean[-1])
                window_dist = float(top[-1]) - window_mean
                msg += f" | {metric}: {window_mean:.5f} +/- {window_dist:.5f}"
        print(msg)
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "family_dirs", nargs="*", help="Folders whose subdirectories are results for individual runs to compare",
    )
    parser.add_argument(
        "--multi_seed", default=False, action="store_true", help="Each run's subdirectory contains results for multiple random seeds",
    )
    args = parser.parse_args()
    for family_dir in args.family_dirs:
        plot(family_dir, multi_seed=args.multi_seed)
