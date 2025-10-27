import shutil
from pathlib import Path

import json

import numpy as np
import matplotlib.pyplot as plt

from probeinterface.neuropixels_tools import _make_npx_probe_from_description, get_probe_metadata_from_probe_features
from probeinterface.plotting import plot_probe
from probeinterface import write_probeinterface


default_base_folder = Path("./neuropixels_library_generated")



def generate_all_npx(base_folder=None):

    if base_folder is None:
        base_folder = default_base_folder

    # if not base_folder.exists():
    base_folder.mkdir(exist_ok=True)

    probe_features_filepath = Path(__file__).absolute().parent / Path("../src/probeinterface/resources/neuropixels_probe_features.json")
    probe_features = json.load(open(probe_features_filepath, "r"))
    probe_part_numbers = probe_features['neuropixels_probes'].keys()


    for model_name in probe_part_numbers:
        print(model_name)

        if model_name is None:
            continue

        if model_name == "NP1110":
            # the formula by the imrow table is wrong and more complicated
            continue

        probe_folder = base_folder / model_name
        probe_folder.mkdir(exist_ok=True)

        pt_metadata, _, _ = get_probe_metadata_from_probe_features(probe_features, model_name)

        num_shank = pt_metadata["num_shanks"]
        contact_per_shank = pt_metadata["cols_per_shank"] * pt_metadata["rows_per_shank"]
        if num_shank == 1:
            elec_ids = np.arange(contact_per_shank)
            shank_ids = None
        else:
            elec_ids = np.concatenate([np.arange(contact_per_shank) for i in range(num_shank)])
            shank_ids = np.concatenate([np.zeros(contact_per_shank) + i for i in range(num_shank)])

        probe = _make_npx_probe_from_description(pt_metadata, model_name, elec_ids, shank_ids)

        # plotting
        fig, axs = plt.subplots(ncols=2)

        ax = axs[0]
        plot_probe(probe, ax=ax)
        ax.set_title("")

        ax.xaxis.set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax = axs[1]


        # plot_probe(probe, ax=ax, text_on_contact=probe._contact_ids)
        plot_probe(probe, ax=ax)
        ax.set_title("")

        yp = pt_metadata["electrode_pitch_vert_um"]
        ax.set_ylim(-yp*8, yp*13)
        ax.yaxis.set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        n = probe.get_contact_count()

        title = f"{probe.manufacturer} - {model_name}"
        title += f"\n{probe.description}"
        title += f"\n {n}ch"
        if probe.shank_ids is not None:
            num_shank = probe.get_shank_count()
            title += f" - {num_shank}shanks"


        fig.suptitle(title)

        # plt.show()

        fig.savefig(probe_folder / f"{model_name}.png")

        write_probeinterface(probe_folder / f"{model_name}.json", probe)

        # plt.show()

        plt.close(fig)






if __name__ == "__main__":
    generate_all_npx()
