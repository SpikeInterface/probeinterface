import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from probeinterface.neuropixels_tools import npx_descriptions, probe_part_number_to_probe_type, _make_npx_probe_from_description
from probeinterface.plotting import plot_probe
from probeinterface import write_probeinterface


base_folder = Path("./neuropixels_library_generated")



def generate_all_npx():

    # if not base_folder.exists():
    base_folder.mkdir(exist_ok=True)


    for probe_number, probe_type in probe_part_number_to_probe_type.items():

        if probe_number is None:
            continue

        probe_folder = base_folder / probe_number
        probe_folder.mkdir(exist_ok=True)
    
        print(probe_number, probe_type)

        probe_description = npx_descriptions[probe_type]

        

        num_shank = probe_description["shank_number"]
        contact_per_shank = probe_description["ncols_per_shank"] * probe_description["nrows_per_shank"]
        if num_shank == 1:
            elec_ids = np.arange(contact_per_shank)
            shank_ids = None
        else:
            elec_ids = np.concatenate([np.arange(contact_per_shank) for i in range(num_shank)])
            shank_ids = np.concatenate([np.zeros(contact_per_shank) + i for i in range(num_shank)])

        probe = _make_npx_probe_from_description(probe_description, elec_ids, shank_ids)

        # ploting
        fig, axs = plt.subplots(ncols=2)

        ax = axs[0]
        plot_probe(probe, ax=ax)
        ax.set_title("")

        ax.xaxis.set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax = axs[1]
        

        plot_probe(probe, ax=ax)
        ax.set_title("")

        yp = probe_description["y_pitch"]
        ax.set_ylim(-yp*8, yp*12)
        ax.yaxis.set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        n = probe.get_contact_count()

        title = probe_number
        title += f"\n{probe.manufacturer} - {probe.model_name}"
        title += f"\n {n}ch"
        if probe.shank_ids is not None:
            num_shank = probe.get_shank_count()
            title += f" - {num_shank}shanks"


        fig.suptitle(title)

        plt.show()

        # fig.savefig(probe_folder / f"{probe_number}.png")

        # write_probeinterface(probe_folder / f"{probe_number}.json", probe)

        # plt.close(fig)









if __name__ == "__main__":
    generate_all_npx()
