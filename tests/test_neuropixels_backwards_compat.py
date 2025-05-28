from probeinterface.neuropixels_tools import make_npx_description, probe_part_number_to_probe_type


# this dict define the contour for one shank (duplicated when several shanks so)
# note that a final "contour_shift" is applied
polygon_contour_description = {
    # NP1 and NP2 (1 and 4 shanks)
    "np70": [
        (0, 10000),
        (0, 0),
        (35, -175),
        (70, 0),
        (70, 10000),
    ],
    "nhp90": [
        (0, 10000),
        (0, 0),
        (45, -342),
        (90, 0),
        (90, 10000),
    ],
    "nhp125": [
        (0, 10000),
        (0, 0),
        (62.5, -342),
        (125, 0),
        (125, 10000),
    ],
}


npx_descriptions = {
    # Neuropixels 1.0
    # This probably should be None or something else because NOT ONLY the neuropixels 1.0 have that imDatPrb_type
    "0": {
        "model_name": "Neuropixels 1.0",
        "x_pitch": 32,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 16,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 480,
        "contour_description": "np70",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # Neuropixels 2.0 - Single Shank - Prototype
    "21": {
        "model_name": "Neuropixels 2.0 - Single Shank - Prototype",
        "x_pitch": 32,
        "y_pitch": 15,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 640,
        "contour_description": "np70",
        "contour_shift": [-27, -11],
        "fields_in_imro_table": ("channel_ids", "banks", "references", "elec_ids"),
    },
    # Neuropixels 2.0 - Four Shank - Prototype
    "24": {
        "model_name": "Neuropixels 2.0 - Four Shank - Prototype",
        "x_pitch": 32,
        "y_pitch": 15,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 250,
        "shank_number": 4,
        "ncols_per_shank": 2,
        "nrows_per_shank": 640,
        "contour_description": "np70",
        "contour_shift": [-27, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "shank_id",
            "banks",
            "references",
            "elec_ids",
        ),
    },
    # Neuropixels 2.0 - Single Shank - Commercial without metal cap
    "2003": {
        "model_name": "Neuropixels 2.0 - Single Shank",
        "x_pitch": 32,
        "y_pitch": 15,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 640,
        "contour_description": "np70",
        "contour_shift": [-27, -11],
        "fields_in_imro_table": ("channel_ids", "banks", "references", "elec_ids"),
    },
    # Neuropixels 2.0 - Single Shank - Commercial with metal cap
    "2004": {
        "model_name": "Neuropixels 2.0 - Single Shank",
        "x_pitch": 32,
        "y_pitch": 15,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 640,
        "contour_description": "np70",
        "contour_shift": [-27, -11],
        "fields_in_imro_table": ("channel_ids", "banks", "references", "elec_ids"),
    },
    # Neuropixels 2.0 - Four Shank - Commercial without metal cap
    "2013": {
        "model_name": "Neuropixels 2.0 - Four Shank",
        "x_pitch": 32,
        "y_pitch": 15,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 250,
        "shank_number": 4,
        "ncols_per_shank": 2,
        "nrows_per_shank": 640,
        "contour_description": "np70",
        "contour_shift": [-27, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "shank_id",
            "banks",
            "references",
            "elec_ids",
        ),
    },
    # Neuropixels 2.0 - Four Shank - Commercial with metal cap
    "2014": {
        "model_name": "Neuropixels 2.0 - Four Shank",
        "x_pitch": 32,
        "y_pitch": 15,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 250,
        "shank_number": 4,
        "ncols_per_shank": 2,
        "nrows_per_shank": 640,
        "contour_description": "np70",
        "contour_shift": [-27, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "shank_id",
            "banks",
            "references",
            "elec_ids",
        ),
    },
    # Neuropixels 2.0 Quad Base
    "2020": {
        "model_name": "Neuropixels 2.0 - Quad Base",
        "x_pitch": 32,
        "y_pitch": 15,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 250,
        "shank_number": 4,
        "ncols_per_shank": 2,
        "nrows_per_shank": 640,
        "contour_description": "np70",
        "contour_shift": [-27, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "shank_id",
            "banks",
            "references",
            "elec_ids",
        ),
    },
    # Experimental probes previous to 1.0
    "Phase3a": {
        "model_name": "Phase3a",
        "x_pitch": 32,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 16.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 480,
        "contour_description": "np70",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
        ),
    },
    # Neuropixels 1.0-NHP Short (10mm)
    "1015": {
        "model_name": "Neuropixels 1.0-NHP - short",
        "x_pitch": 32,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 480,
        "contour_description": "np70",
        "contour_shift": [-27, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    #################
    # Neuropixels 1.0-NHP Medium (25mm)
    "1020": {
        "model_name": "Neuropixels 1.0-NHP - medium - staggered",
        "x_pitch": 103 - 12,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 12.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 1248,  ### verify this number!!!!!!! Jennifer Colonell has 1368
        "contour_description": "nhp125",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # Neuropixels 1.0-NHP Medium (25mm)
    "1021": {
        "model_name": "Neuropixels 1.0-NHP - medium - staggered",
        "x_pitch": 103 - 12,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 12.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 1248,  ### verify this number!!!!!!! Jennifer Colonell has 1368
        "contour_description": "nhp125",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    #################
    # Neuropixels 1.0-NHP Medium (25mm)
    "1022": {
        "model_name": "Neuropixels 1.0-NHP - medium",
        "x_pitch": 103,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 1248,  ### verify this number!!!!!!! Jennifer Colonell has 1368
        "contour_description": "nhp125",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # Neuropixels 1.0-NHP 45mm SOI90 - NHP long 90um wide, staggered contacts
    "1030": {
        "model_name": "Neuropixels 1.0-NHP - long SOI90 staggered",
        "x_pitch": 56,
        "y_pitch": 20,
        "stagger": 12,
        "contact_width": 12,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 2208,
        "contour_description": "nhp90",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # Neuropixels 1.0-NHP 45mm SOI125 - NHP long 125um wide, staggered contacts
    "1031": {
        "model_name": "Neuropixels 1.0-NHP - long SOI125 staggered",
        "x_pitch": 91,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 12.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 2208,
        "contour_description": "nhp125",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # 1.0-NHP 45mm SOI115 / 125 linear - NHP long 125um wide, linear contacts
    "1032": {
        "model_name": "Neuropixels 1.0-NHP - long SOI125 linear",
        "x_pitch": 103,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 2208,
        "contour_description": "nhp125",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # Ultra probes 1 bank
    "1100": {
        "model_name": "Neuropixels Ultra (1 bank)",
        "x_pitch": 6,
        "y_pitch": 6,
        "contact_width": 5,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 8,
        "nrows_per_shank": 48,
        "contour_description": "np70",
        "contour_shift": [-14, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # Ultra probes 16 banks
    "1110": {
        "model_name": "Neuropixels Ultra (16 banks)",
        "x_pitch": 6,
        "y_pitch": 6,
        "contact_width": 5,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 8,
        "nrows_per_shank": 768,
        "contour_description": "np70",
        "contour_shift": [-14, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    "1121": {
        "model_name": "Neuropixels Ultra - Type 2",
        "x_pitch": 6,
        "y_pitch": 3,
        "contact_width": 2,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 1,
        "nrows_per_shank": 384,
        "contour_description": "np70",
        "contour_shift": [-6.25, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
    # NP-Opto
    "1300": {
        "model_name": "Neuropixels Opto",
        "x_pitch": 48,
        "y_pitch": 20,
        "contact_width": 12,
        "stagger": 0.0,
        "shank_pitch": 0,
        "shank_number": 1,
        "ncols_per_shank": 2,
        "nrows_per_shank": 480,
        "contour_description": "np70",
        "contour_shift": [-11, -11],
        "fields_in_imro_table": (
            "channel_ids",
            "banks",
            "references",
            "ap_gains",
            "lf_gains",
            "ap_hp_filters",
        ),
    },
}


def test_consistency_with_past():
    """
    Test to confirm consistency between ProbeTable and previous ProbeInterface
    implementation. For each probe, we get the `npx_description` first using the
    `make_npx_description` function, then the old, hard-coded dict (now stored
    in this test page).
    """

    # known mismatches between ProbeTable and previous ProbeInterface implementation
    # None is the Phase3a probe.
    known_mismatches = {
        # "Phase3a": ["fields_in_imro_table"],
        "NP1020": ["x_pitch", "stagger"],
        "NP1021": ["x_pitch", "stagger"],
        "NP1030": ["x_pitch", "stagger"],
        "NP1031": ["x_pitch", "stagger"],
        "NP1110": ["fields_in_imro_table", "x_pitch"],
        "NP1121": ["x_pitch", "contact_width"],
    }

    for probe_part_number, probe_type in probe_part_number_to_probe_type.items():

        if probe_part_number not in ["3000", "1200"]:
            
            probe_info = make_npx_description(probe_part_number)
            old_probe_info = npx_descriptions[probe_type]

            for value in [
                "x_pitch",
                "y_pitch",
                "contact_width",
                "stagger",
                "shank_pitch",
                "shank_number",
                "ncols_per_shank",
                "nrows_per_shank",
                "fields_in_imro_table",
            ]:

                if known_mismatches.get(probe_part_number) is not None:
                    if value in known_mismatches.get(probe_part_number):
                        continue

                assert probe_info[value] == old_probe_info[value]
