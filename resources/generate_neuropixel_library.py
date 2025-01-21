import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from probeinterface.neuropixel_tools import npx_descriptions, _make_npx_probe_from_description
from probeinterface.plotting import plot_probe
from probeinterface import write_probeinterface


folder = Path("./neuropixel_library_generated")


def generate_all_npx():

    if not folder.exists():
        folder.mkdir()

    for k, probe_description in npx_descriptions.items():
        print(k)

        name = probe_description["model_name"]
        fig, ax = plt.subplots()
        ax.set_title(name)

        num_shank = probe_description["shank_number"]
        contact_per_shank = probe_description["ncols_per_shank"] * probe_description["nrows_per_shank"]
        if num_shank == 1:
            elec_ids = np.arange(contact_per_shank)
            shank_ids = None
        else:
            elec_ids = np.concatenate([np.arange(contact_per_shank) for i in range(num_shank)])
            shank_ids = np.concatenate([np.zeros(contact_per_shank) + i for i in range(num_shank)])

        probe = _make_npx_probe_from_description(probe_description, elec_ids, shank_ids)

        plot_probe(probe, ax=ax)
        yp = probe_description["y_pitch"]
        ax.set_ylim(-yp*8, yp*12)
        # plt.show()

        fig.savefig(folder / f"{name}.png")

        write_probeinterface(folder / f"{name}.json", probe)









if __name__ == "__main__":
    generate_all_npx()
