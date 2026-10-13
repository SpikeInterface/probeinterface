import warnings

import numpy as np
from .utils import generate_unique_ids
from .probe import Probe


class ProbeGroup:
    """
    Class to handle a group of Probe objects and the global wiring to a device.

    Internally, this is represented as a list of Probe objects.

    The ProbeGroup is the object saved in the json based probeinterface format, even if there is only one probe.

    Tiny detail about contact order: ``ProbeGroup.to_numpy()`` / ``ProbeGroup.to_dataframe()`` return contacts in the
    "natural" order (the contacts of each probe stacked one probe after another) unless contacts have become
    interleaved across probes. Interleaving can arise from ``get_slice`` or ``select_contacts`` (e.g. selecting
    contacts from different probes in an alternating order). When it does, the resulting ``ProbeGroup`` keeps a custom
    contact order in the ``_global_contact_order`` attribute so the requested order is preserved. This order is only
    ever set internally; there is no public method to set it.
    """

    def __init__(self):
        self._probes = []
        self._probe_ids = []
        self._global_contact_order = None

    def __repr__(self):
        repr_str = f"ProbeGroup: {len(self._probes)} probes - {self.get_contact_count()} contacts"
        if self._global_contact_order is not None:
            repr_str += " (with custom global contact order)"
        return repr_str

    def add_probe(self, probe: Probe, probe_id: str = None) -> None:
        """
        Add an additional probe to the ProbeGroup

        Parameters
        ----------
        probe: Probe
            The probe to add to the ProbeGroup
        probe_id: str, optional
            The ID to assign to the probe. If None, a unique ID will be generated,
            unless a probe_id is already present in the probe's annotations,
            in which case that will be used.

        """
        if len(self._probes) > 0:
            self._check_compatible(probe)

        probe_id_annotation = probe.annotations.get("probe_id", None)

        if probe_id is None:
            if probe_id_annotation is not None:
                probe_id = probe_id_annotation
            else:
                existing_int_ids = [int(pid) for pid in self._probe_ids if pid.isdigit()]
                probe_id = str(max(existing_int_ids, default=-1) + 1)
        else:
            if probe_id_annotation is not None and probe_id != probe_id_annotation:
                warnings.warn(
                    f"Provided probe_id '{probe_id}' does not match probe's annotation 'probe_id' "
                    f"({probe_id_annotation}). Using provided probe_id."
                )

        if probe_id in self._probe_ids:
            raise ValueError(f"Probe ID '{probe_id}' is already used in this ProbeGroup.")
        self._probe_ids.append(probe_id)

        self._probes.append(probe)
        probe._probe_group = self

    @property
    def probe_dict(self) -> dict:
        return {probe_id: probe for probe_id, probe in zip(self._probe_ids, self._probes)}

    @property
    def probes(self) -> list:
        return self._probes

    @property
    def probe_ids(self) -> list:
        return self._probe_ids

    def _check_compatible(self, probe: Probe) -> None:
        if probe._probe_group is not None:
            raise ValueError(
                "This probe is already attached to another ProbeGroup. Use probe.copy() to attach it to another ProbeGroup"
            )

        if probe.ndim != self.probes[-1].ndim:
            raise ValueError(
                f"ndim are not compatible: probe.ndim {probe.ndim} " f"!= probegroup ndim {self.probes[-1].ndim}"
            )

        # check global channel maps
        self._probes.append(probe)
        self._probe_ids.append(f"{len(self._probes)-1}")
        self._check_global_device_wiring_and_ids()
        self._probes = self.probes[:-1]
        self._probe_ids = self.probe_ids[:-1]

    @property
    def ndim(self) -> int:
        return self.probes[0].ndim

    def copy(self) -> "ProbeGroup":
        """
        Create a copy of the ProbeGroup

        Returns
        -------
        copy: ProbeGroup
            A copy of the ProbeGroup
        """
        return ProbeGroup.from_dict(self.to_dict(array_as_list=False))

    def get_contact_count(self) -> int:
        """
        Total number of channels.

        Returns
        -------
        n: int
            The total number of channels
        """
        n = sum(probe.get_contact_count() for probe in self.probes)
        return n

    def to_numpy(self, complete: bool = False) -> np.ndarray:
        """
        Export all probes into a numpy array.

        Parameters
        ----------
        complete: bool, default: False
            If True, export complete information about the probegroup
            including contact_plane_axes/si_units/device_channel_indices
        """

        fields = []
        probe_arr = []

        # loop over probes to get all fields
        dtype = [("probe_index", "int64"), ("probe_id", "U100")]
        fields = []
        for probe_index, probe in enumerate(self.probes):
            arr = probe.to_numpy(complete=complete)
            probe_arr.append(arr)
            for k in arr.dtype.fields:
                if k not in fields:
                    fields.append(k)
                    dtype += [(k, arr.dtype.fields[k][0])]

        pg_arr = []
        for probe_index, probe in enumerate(self.probes):
            arr = probe_arr[probe_index]
            arr_ext = np.zeros(probe.get_contact_count(), dtype=dtype)
            arr_ext["probe_index"] = probe_index
            arr_ext["probe_id"] = self._probe_ids[probe_index]
            for k in fields:
                if k in arr.dtype.fields:
                    arr_ext[k] = arr[k]
            pg_arr.append(arr_ext)

        pg_arr = np.concatenate(pg_arr, axis=0)

        if self._global_contact_order is not None:
            pg_arr = pg_arr[self._global_contact_order]
        return pg_arr

    @staticmethod
    def from_numpy(arr: np.ndarray) -> "ProbeGroup":
        """Create ProbeGroup from a complex numpy array
        see ProbeGroup.to_numpy()

        Note that if the contact_vector has several probe and some contact are interleaved, then the ProbeGroup will
        have a non natural ordering (contact from probes are stack): in short ProbeGroup._global_contact_order
        will be not None.

        Parameters
        ----------
        arr : np.array
            The structured np.array representation of the probe

        Returns
        -------
        probegroup : ProbeGroup
            The instantiated ProbeGroup object
        """

        # Check if contacts are interleaved
        num_probes = np.unique(arr["probe_index"]).size
        is_interleaved = (num_probes > 1) and np.any(np.diff(arr["probe_index"]) < 0)
        if is_interleaved:
            global_contact_order = []

        probes_indices = np.sort(np.unique(arr["probe_index"]))
        probegroup = ProbeGroup()
        for probe_index in probes_indices:
            mask = arr["probe_index"] == probe_index
            if "probe_id" not in arr.dtype.fields:
                probe_id = str(probe_index)
            else:
                probe_id = arr["probe_id"][mask][0]
            probe = Probe.from_numpy(arr[mask])
            probegroup.add_probe(probe, probe_id=probe_id)

            if is_interleaved:
                global_contact_order.append(np.flatnonzero(mask))

        if is_interleaved:
            # the argsort is for the 'reverse' order!
            probegroup._global_contact_order = np.argsort(np.concatenate(global_contact_order))

        return probegroup

    def to_dataframe(self, complete: bool = False) -> "pandas.DataFrame":
        """
        Export the probegroup to a pandas dataframe

        Parameters
        ----------
        complete : bool, default: False
            If True, export complete information about the probegroup,
            including the probe plane axis.

        Returns
        -------
        df : pandas.DataFrame
            The dataframe representation of the probegroup

        """
        import pandas as pd

        df = pd.DataFrame(self.to_numpy(complete=complete))
        df.index = np.arange(df.shape[0], dtype="int64")
        return df

    def to_dict(self, array_as_list: bool = False) -> dict:
        """Create a dictionary of all necessary attributes.

        Parameters
        ----------
        array_as_list : bool, default: False
            If True, arrays are converted to lists, by default False

        Returns
        -------
        d : dict
            The dictionary representation of the probegroup
        """
        d = {}
        d["probes"] = []
        for probe in self.probes:
            probe_dict = probe.to_dict(array_as_list=array_as_list)
            d["probes"].append(probe_dict)
        d["probe_ids"] = self.probe_ids
        if self._global_contact_order is not None:
            global_contact_order = self._global_contact_order
            if array_as_list:
                global_contact_order = global_contact_order.tolist()
            d["global_contact_order"] = global_contact_order
        return d

    @staticmethod
    def from_dict(d: dict) -> "ProbeGroup":
        """Instantiate a ProbeGroup from a dictionary

        Parameters
        ----------
        d : dict
            The dictionary representation of the probegroup

        Returns
        -------
        probegroup : ProbeGroup
            The instantiated ProbeGroup object
        """
        probegroup = ProbeGroup()
        probe_ids = d.get("probe_ids", None)
        if probe_ids is None:
            probe_ids = [str(i) for i in range(len(d["probes"]))]
        for probe_id, probe_dict in zip(probe_ids, d["probes"]):
            probe = Probe.from_dict(probe_dict)
            probegroup.add_probe(probe, probe_id=probe_id)

        global_contact_order = d.get("global_contact_order", None)
        if global_contact_order is not None:
            probegroup._global_contact_order = np.asarray(global_contact_order)

        return probegroup

    # TODO: this should only return the device_channel_indices, not the probe_index!!!
    def get_global_device_channel_indices(self) -> np.ndarray:
        """
        Gets the global device channels indices and returns as
        an array

        Returns
        -------
        channels: np.ndarray
            a numpy array vector with 2 columns
            (probe_index, device_channel_indices)

        Notes
        -----
            If a channel within channels has a value of -1 this indicates that that channel
            is disconnected
        """
        total_chan = self.get_contact_count()
        channels = np.zeros(total_chan, dtype=[("probe_index", "int64"), ("device_channel_indices", "int64")])
        arr = self.to_numpy(complete=True)
        channels["probe_index"] = arr["probe_index"]
        channels["device_channel_indices"] = arr["device_channel_indices"]
        return channels

    def set_global_device_channel_indices(self, device_channel_indices: np.ndarray | list) -> None:
        """
        Set global device channel indices for all probes.

        Important note: if the probegroup has ``_global_contact_order``, then the device_channel_indices
        are reordered before being set. In short, the ``device_channel_indices`` is zipped to
        ProbeGroup.to_numpy() (always ordered).

        Parameters
        ----------
        device_channel_indices: np.ndarray | list
            The device channal indices to be set
        """
        device_channel_indices = np.asarray(device_channel_indices)
        if device_channel_indices.size != self.get_contact_count():
            raise ValueError(
                f"Wrong channels size {device_channel_indices.size} for the number of channels {self.get_contact_count()}"
            )

        # first reset previous indices
        for i, probe in enumerate(self.probes):
            n = probe.get_contact_count()
            probe.set_device_channel_indices([-1] * n)

        if self._global_contact_order is not None:
            # this is tricky conceptually but needed for consistency
            rev_order = np.argsort(self._global_contact_order)
            device_channel_indices = device_channel_indices[rev_order]

        # then set new indices
        ind = 0
        for i, probe in enumerate(self.probes):
            n = probe.get_contact_count()
            probe.set_device_channel_indices(device_channel_indices[ind : ind + n])
            ind += n

    def get_global_contact_ids(self) -> np.ndarray:
        """
        Gets all contact ids concatenated across probes

        Contact ids are unique within a Probe but may repeat across probes, so the
        returned array is not necessarily unique. Pair it with the ``probe_index``
        column of :meth:`to_numpy` to get a key that is unique across the ProbeGroup.

        Returns
        -------
        contact_ids: np.ndarray
            An array of the contact ids across all probes
        """
        contact_ids = self.to_numpy(complete=True)["contact_ids"]
        return contact_ids

    def get_global_contact_positions(self) -> np.ndarray:
        """
        Gets all contact positions concatenated across probes

        Returns
        -------
        contact_positions: np.ndarray
            An array of the contact positions across all probes
        """
        contact_positions = np.vstack([probe.contact_positions for probe in self.probes])
        if self._global_contact_order is not None:
            contact_positions = contact_positions[self._global_contact_order]
        return contact_positions

    def get_slice(self, selection: np.ndarray[bool | int]) -> "ProbeGroup":
        """
        Get a copy of the ProbeGroup with a sub selection of contacts.

        Selection can be boolean or by index

        Parameters
        ----------
        selection : np.array of bool or int (for index)
            Either an np.array of bool or for desired selection of contacts
            or the indices of the desired contacts

        Returns
        -------
        sliced_probe_group: ProbeGroup
            The sliced probe group

        """

        n = self.get_contact_count()

        selection = np.asarray(selection)
        if selection.dtype.kind not in ("b", "i"):
            raise TypeError(f"selection must be bool array or int array, not of type: {type(selection)}")

        if selection.dtype == "bool":
            assert selection.shape == (
                n,
            ), f"if array of bool given it must be the same size as the number of contacts {selection.shape} != {n}"
            selection = np.flatnonzero(selection)

        if len(selection) == 0:
            raise ValueError("ProbeGroup.get_slice() with empty selection is not handled")
            # return ProbeGroup()

        assert np.unique(selection).size == selection.size
        assert 0 <= np.min(selection) < n, f"An index within your selection is out of bounds {np.min(selection)}"
        assert 0 <= np.max(selection) < n, f"An index within your selection is out of bounds {np.max(selection)}"

        contact_arr = self.to_numpy(complete=True)
        contact_arr = contact_arr[selection]
        sliced_probe_group = ProbeGroup.from_numpy(contact_arr)

        # Map annotations of the original probegroup to the sliced one
        for probe_id, new_probe in zip(sliced_probe_group.probe_ids, sliced_probe_group.probes):
            original_probe_index = self.probe_ids.index(probe_id)
            orig_probe = self.probes[original_probe_index]

            for k in orig_probe.annotations:
                if k not in new_probe.annotations:
                    new_probe.annotate(**{k: orig_probe.annotations[k]})

            # probe_planar_contour is a probe-level attribute, not part of the to_numpy dtype,
            # so from_numpy cannot restore it; copy it over explicitly.
            if orig_probe.probe_planar_contour is not None and new_probe.probe_planar_contour is None:
                new_probe.set_planar_contour(orig_probe.probe_planar_contour)

        return sliced_probe_group

    def select_probes(self, probe_ids: str | np.ndarray | list) -> "ProbeGroup":
        """
        Get a copy of the ProbeGroup with a sub selection of probes based on probe ids.

        Parameters
        ----------
        probe_ids : str | np.array or list
            The probe id or ids to select.

        Returns
        -------
        sliced_probe_group: ProbeGroup
            The sliced probe group
        """
        if probe_ids is None:
            raise ValueError("probe_ids must be provided for selection.")

        if isinstance(probe_ids, str):
            probe_ids = [probe_ids]

        probe_ids = np.asarray(probe_ids)
        if any(probe_id not in self.probe_ids for probe_id in probe_ids):
            raise ValueError(f"Some probe_ids {probe_ids} are not present in the ProbeGroup.")

        # selection keeps the order of the to_numpy vector
        all_probe_ids = self.to_numpy(complete=True)["probe_id"]
        keep_inds = np.flatnonzero(np.isin(all_probe_ids, probe_ids))
        return self.get_slice(keep_inds)

    def select_contacts(
        self, contact_ids: np.ndarray | list, probe_ids: np.ndarray | list | None = None
    ) -> "ProbeGroup":
        """
        Get a copy of the ProbeGroup with a sub selection of contacts based on contact ids and probe ids.

        Parameters
        ----------
        contact_ids : np.array or list
            The contact ids to select.
        probe_ids : np.array or list or None, default: None
            If multiple probes and contact ids not unique across probes, an array with the same length
            as contact ids to specify which probe each contact id belongs to.

        Returns
        -------
        sliced_probe_group: ProbeGroup
            The sliced probe group
        """
        # both arrays are in the global contact order
        arr = self.to_numpy(complete=True)
        all_contact_ids = arr["contact_ids"]
        all_probe_ids = arr["probe_id"]

        contact_ids = np.asarray(contact_ids)

        if probe_ids is None:
            # without probe_ids the request must be unambiguous: each requested contact
            # id must appear once in the request and match a single contact in the group
            unique_requested, counts = np.unique(contact_ids, return_counts=True)
            duplicated = unique_requested[counts > 1]
            if duplicated.size > 0:
                raise ValueError(
                    f"contact_ids must be unique, but {duplicated.tolist()} appear more than once. "
                    "If the same contact id is on multiple probes, use probe_ids to disambiguate."
                )
            # each requested contact id must live on exactly one probe; collect every
            # ambiguous one so the user can disambiguate them all at once
            ambiguous = {}
            for contact_id in contact_ids:
                probes_for_id = np.unique(all_probe_ids[all_contact_ids == contact_id]).tolist()
                if len(probes_for_id) > 1:
                    ambiguous[str(contact_id)] = probes_for_id
            if ambiguous:
                ambiguity_lines = "\n".join(
                    f'"{contact_id}" lives on probes {probes}' for contact_id, probes in ambiguous.items()
                )
                message = f"""\
Some contact ids are ambiguous because they live on multiple probes; pass probe_ids to disambiguate which probe each belongs to:
{ambiguity_lines}"""
                raise ValueError(message)
            probe_ids = [None] * len(contact_ids)
        else:
            if len(probe_ids) != len(contact_ids):
                raise ValueError(
                    f"probe_ids must be the same length as contact_ids, but got {len(probe_ids)} probe_ids and {len(contact_ids)} contact_ids."
                )

        indices = []
        for contact_id, probe_id in zip(contact_ids, probe_ids):
            # find the contact id within the specified probe
            in_probe_mask = np.ones(all_contact_ids.size, dtype=bool) if probe_id is None else all_probe_ids == probe_id
            matches = np.flatnonzero((all_contact_ids == contact_id) & in_probe_mask)

            if matches.size == 0:
                raise ValueError(f"contact_id {contact_id} not found in probe {probe_id}")
            elif matches.size > 1:
                raise ValueError(
                    f"contact_id {contact_id} is not unique within probe {probe_id}, "
                    "this should not happen unless the probe has duplicate contact ids"
                )
            if matches[0] in indices:
                raise ValueError(
                    f"contact_id {contact_id} matches multiple probes; "
                    "pass probe_ids to disambiguate which probe each contact_id belongs to."
                )
            indices.append(matches[0])
        return self.get_slice(indices)

    def _check_global_device_wiring_and_ids(self) -> None:
        if self.get_contact_count() == 0:
            return
        arr = self.to_numpy(complete=True)

        # check unique device_channel_indices for !=-1
        chans = arr["device_channel_indices"]
        valid_chans = chans[chans >= 0]

        if valid_chans.size != np.unique(valid_chans).size:
            raise ValueError("channel device indices are not unique across probes")

        # check unique contact ids. A contact_id is unique within a Probe, not across
        # the whole ProbeGroup, so the key that has to be unique here is the pair
        # (probe_index, contact_id). This lets the same probe model be added twice
        # without renaming its contacts, while still identifying every contact.
        pairs = list(zip(arr["probe_index"].tolist(), arr["contact_ids"].tolist()))
        if len(set(pairs)) != len(pairs):
            duplicated = sorted({pair for pair in pairs if pairs.count(pair) > 1})
            raise ValueError(
                f"(probe_index, contact_id) pairs are not unique across probes: {duplicated}. "
                "contact_ids only need to be unique within a single Probe."
            )

    def auto_generate_contact_ids(self, *args, **kwargs) -> None:
        """
        Annotate all contacts with unique contact_id values.

        Parameters
        ----------
        *args: will be forwarded to `probeinterface.utils.generate_unique_ids`
        **kwargs: will be forwarded to
            `probeinterface.utils.generate_unique_ids`
        """

        if not args:
            args = 1e7, 1e8
        # 3rd argument has to be the number of probes
        args = args[:2] + (self.get_contact_count(),)

        contact_ids = generate_unique_ids(*args, **kwargs).astype(str)

        for probe in self.probes:
            el_ids, contact_ids = np.split(contact_ids, [probe.get_contact_count()])
            probe.set_contact_ids(el_ids)
