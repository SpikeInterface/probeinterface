"""
2025-12-16 CambridgeNeurotech

Derive probes to be used with SpikeInterface base on Cambridgeneurotech database at:
https://github.com/cambridge-neurotech/probe_maps


The output folder is ready to be used as a probeinterface library and contains:
- one folder per probe
- inside each folder a json file and a figure png file
"""

import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from probeinterface.plotting import plot_probe
from probeinterface import write_probeinterface, Probe


cn_logo = Path(__file__).parent / "CN-logo.jpg"

parser = argparse.ArgumentParser(description="Generate CambridgeNeurotech probe library for probeinterface")
parser.add_argument(
    "probe_tables_path",
    type=str,
    help="Path to the folder containing the CambridgeNeurotech probe tables CSV files from https://github.com/cambridge-neurotech/probe_maps",
)
parser.add_argument(
    "--output-folder", type=str, default="./cambridgeneurotech", help="Output folder to save the generated probes"
)


# graphing parameters
plt.rcParams["pdf.fonttype"] = 42  # to make sure it is recognize as true font in illustrator
plt.rcParams["svg.fonttype"] = "none"  # to make sure it is recognize as true font in illustrator


def create_CN_figure(probe):
    """
    Create custom figire for CN with custom colors + logo
    """
    if probe.contact_sides is not None:
        fig, axs = plt.subplots(ncols=2)
        fig.set_size_inches(18.5, 10.5)
    else:
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)
        axs = [ax]

    n = probe.get_contact_count()
    probe_max_height = np.max(probe.contact_positions[:, 1])
    if probe.contact_sides is not None:
        for i, side in enumerate(("front", "back")):
            ax = axs[i]
            plot_probe(
                probe,
                ax=ax,
                contacts_colors=["#5bc5f2"] * n,  # made change to default color
                probe_shape_kwargs=dict(
                    facecolor="#6f6f6e", edgecolor="k", lw=0.5, alpha=0.3
                ),  # made change to default color
                with_contact_id=True,
                side=side,
            )
            ax.set_title(f"Side: {side}", fontsize=20)
    else:
        plot_probe(
            probe,
            ax=axs[0],
            contacts_colors=["#5bc5f2"] * n,  # made change to default color
            probe_shape_kwargs=dict(
                facecolor="#6f6f6e", edgecolor="k", lw=0.5, alpha=0.3
            ),  # made change to default color
            with_contact_id=True,
        )
        axs[0].set_title("")

    for ax in axs:
        y_min = ax.get_ylim()[0]
        y_max = probe_max_height + 200
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Width (\u03bcm)")  # modify to legend
        ax.set_ylabel("Height (\u03bcm)")  # modify to legend
        ax.spines["right"].set_visible(False)  # remove external axis
        ax.spines["top"].set_visible(False)  # remove external axis

    fig.suptitle("\n" + "CambridgeNeuroTech" + "\n" + probe.model_name, fontsize=24)

    fig.tight_layout()

    im = plt.imread(str(cn_logo))
    newax = fig.add_axes([0.8, 0.85, 0.2, 0.1], anchor="NW", zorder=0)
    newax.imshow(im)
    newax.axis("off")

    return fig


def export_one_probe(probe_name, probe, output_folder):
    """
    Save one probe in "output_folder" + figure.
    """
    probe_folder = output_folder / probe_name
    probe_folder.mkdir(exist_ok=True, parents=True)
    probe_file = probe_folder / (probe_name + ".json")
    figure_file = probe_folder / (probe_name + ".png")

    write_probeinterface(probe_file, probe)

    fig = create_CN_figure(probe)
    fig.savefig(figure_file)

    plt.close(fig)


def is_contour_correct(probe):
    from shapely.geometry import Point, Polygon

    polygon = Polygon(probe.probe_planar_contour)

    for i, contact_pos in enumerate(probe.contact_positions):
        width = probe.contact_shape_params[i]["width"]
        height = probe.contact_shape_params[i]["height"]
        points = [
            (contact_pos[0] - width / 2, contact_pos[1] - height / 2),
            (contact_pos[0] + width / 2, contact_pos[1] - height / 2),
            (contact_pos[0] + width / 2, contact_pos[1] + height / 2),
            (contact_pos[0] - width / 2, contact_pos[1] + height / 2),
        ]
        for point in points:
            p = Point(point[0], point[1])
            if not polygon.contains(p):
                return False
    return True


def generate_all_probes(probe_tables_path, output_folder):
    sheet_names = list(pd.read_excel(probe_tables_path / "probe_contacts.xlsx", sheet_name=None).keys())

    wrong_contours = []
    sheets_with_issues = []

    for sheet_name in tqdm(sheet_names, "Exporting CN probes"):
        contacts = pd.read_excel(probe_tables_path / "probe_contacts.xlsx", sheet_name=sheet_name)
        contour = pd.read_excel(probe_tables_path / "probe_contours.xlsx", sheet_name=sheet_name)

        if np.all(pd.isna(contacts["contact_sides"])):
            contacts.drop(columns="contact_sides", inplace=True)
        else:
            print(f"Double sided probe: {sheet_name}")

        if "z" in contacts.columns:
            contacts.drop(columns=["z"], inplace=True)
        try:
            probe = Probe.from_dataframe(contacts)
            probe.manufacturer = "cambridgeneurotech"
            probe.model_name = sheet_name
            probe.set_planar_contour(contour)

            if not is_contour_correct(probe):
                wrong_contours.append(sheet_name)

            export_one_probe(sheet_name, probe, output_folder)

        except Exception as e:
            print(f"Problem loading {sheet_name}: {e}")
            sheets_with_issues.append(sheet_name)

    print("Wrong contours:\n\n", wrong_contours)
    print("Sheets with issues:\n\n", sheets_with_issues)


if __name__ == "__main__":
    args = parser.parse_args()
    probe_tables_path = Path(args.probe_tables_path)
    output_folder = Path(args.output_folder)
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    generate_all_probes(probe_tables_path, output_folder)
