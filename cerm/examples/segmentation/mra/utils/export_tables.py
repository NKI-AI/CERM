"""Write results resolution level experiment to LaTex table."""

import argparse
import logging
import os
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import matplotlib

from pathlib import Path
from typing import List, Tuple
from matplotlib import pyplot as plt

from mra.utils import tabletex

# Initialize module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize plot settings
fontsize = 18
matplotlib.rcParams.update({"font.size": fontsize})
matplotlib.rcParams["axes.labelpad"] = 12
sns.set_style("ticks", {"axes.grid": True})
sns.color_palette("dark")


def export_table(
    out_dir: Path, metric_names: List[str], model_dirs: List[Path], spacing: float
) -> None:
    """
    Export results to LaTex table

    Parameters
    ----------
    out_dir: Path
        folder to which output is written
    metric_names: List[str]
        names of metrics
    model_dirs: List[Path]
        paths to model output folders
    spacing: float
        pixel spacing (assumed to be isotropic)
    """
    latex_writer = tabletex.LatexTableWriter()
    row_names = [dir.stem for dir in model_dirs]
    num_metrics = len(metric_names)
    num_models = len(model_dirs)

    for mode in ["val", "test"]:
        logger.info(f"Export LaTex table {mode} dataset")

        table = {
            "mean": np.zeros((num_models, num_metrics)),
            "std": np.zeros((num_models, num_metrics)),
        }

        for row_idx, subfolder in enumerate(model_dirs):
            for col_idx, metric in enumerate(metric_names):
                # Load results from torch tensor saved on disk
                file_dir = subfolder / "stats" / mode / f"{metric}.pth"
                metric_vals = torch.load(file_dir).numpy()
                if metric == "hausdorff":
                    metric_vals *= spacing
                table["mean"][row_idx, col_idx] = np.mean(metric_vals)
                table["std"][row_idx, col_idx] = np.std(metric_vals)

        # Compute the essence of deep learning
        bold_rows = []
        for col_idx, metric in enumerate(metric_names):
            if metric == "hausdorff":
                bold_rows.append(np.argmin(table["mean"][:, col_idx]))
            elif metric == "dice":
                bold_rows.append(np.argmax(table["mean"][:, col_idx]))

        # Export to tex
        latex_writer.write(
            out_dir / f"{mode}_{metric}",
            table["mean"].tolist(),
            row_names,
            metric_names,
            additional_content=table["std"].tolist(),
            bold_row_per_column=bold_rows,
        )


def export_boxplot(
    out_dir: Path,
    metric_names: List[str],
    model_dirs: List[Path],
    spacing: float,
    figsize: Tuple[int, int],
    dpi: int,
    linewidth: float = 1.5,
) -> None:
    """
    Export results to boxplot table

    Parameters
    ----------
    out_dir: Path
        folder to which output is written
    metric_names: List[str]
        names of metrics
    model_dirs: List[Path]
        paths to model output folders
    spacing: float
        pixel spacing (assumed to be isotropic)
    figsize: Tuple[int, int]
        figure size
    dpi: int
        dots per inch
    linewidth:
        linewidth axes boxplot
    """
    fig_dirs = {}
    for metric in metric_names:
        fig_dirs[metric] = out_dir / metric
        if not fig_dirs[metric].exists():
            fig_dirs[metric].mkdir()

    model_names = [dir.stem for dir in model_dirs]

    for mode in ["val", "test"]:
        for metric in metric_names:
            logger.info(f"Export violin plot {metric} for {mode} dataset")

            # Construct table with columns model and rows value metric
            model_vals = []
            for subfolder in model_dirs:
                # Load results from torch tensor saved on disk
                file_dir = subfolder / "stats" / mode / f"{metric}.pth"
                metric_vals = torch.load(file_dir).numpy()
                if metric == "hausdorff":
                    metric_vals *= spacing
                model_vals.append(metric_vals.tolist())

            # Construct pandas dataframe to use violin plot
            df = pd.DataFrame(model_vals).transpose().dropna()
            df.columns = model_names

            # Plot
            fig, ax = plt.subplots(figsize=figsize)
            sns.boxplot(data=df, ax=ax, showmeans=True)
            ax.set_ylabel(metric.capitalize())
            ax.set_xlabel("Models")

            # Fix axes
            for side in ["left", "right", "top", "bottom"]:
                ax.spines[side].set_linewidth(linewidth)

            # Export to disk
            fig_dir = fig_dirs[metric] / f"{mode}.pdf"
            fig.savefig(
                str(fig_dir),
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(fig)


def main():
    """Write results resolution level to Latex table"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models_dir", help="path to subfolders containing model results", type=Path
    )
    parser.add_argument("out_dir", help="output folder", type=Path)
    parser.add_argument("--dpi", help="dots per inch", type=int, default=100)
    parser.add_argument(
        "--figsize", help="dots per inch", type=Tuple[int, int], default=(12, 6)
    )
    parser.add_argument(
        "--spacing", help="spacing (assume isotropic)", type=float, default=1.0
    )
    args = parser.parse_args()

    if not args.out_dir.exists():
        args.out_dir.mkdir()

    metric_names = ["dice", "hausdorff"]

    # Find all subfolders
    model_dirs = [
        Path(folder.path) for folder in os.scandir(args.models_dir) if folder.is_dir()
    ]
    model_names = [dir.stem for dir in model_dirs]
    num_models = len(model_dirs)
    logger.info(f"Found {num_models} models")

    export_table(args.out_dir, metric_names, model_dirs, args.spacing)
    export_boxplot(
        args.out_dir, metric_names, model_dirs, args.spacing, args.figsize, args.dpi
    )


if __name__ == "__main__":
    main()
