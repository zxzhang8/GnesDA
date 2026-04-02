import os
import pickle

import numpy as np


def split_storage_paths(dataset_dir, split_name):
    return {
        "seqbin": os.path.join(dataset_dir, f"{split_name}.seqbin"),
        "index": os.path.join(dataset_dir, f"{split_name}.idx.npy"),
        "legacy_pickle": os.path.join(dataset_dir, f"{split_name}_seq_list"),
    }


class SequenceBinWriter:
    """Append-only writer for sequential DNA records."""

    def __init__(self, seqbin_path):
        self.seqbin_path = seqbin_path
        self.handle = open(seqbin_path, "wb")
        self.offsets = []
        self.lengths = []

    def append(self, sequence):
        payload = sequence.encode("ascii")
        offset = self.handle.tell()
        self.handle.write(payload)
        self.offsets.append(offset)
        self.lengths.append(len(payload))

    def close(self, index_path):
        if self.handle.closed:
            return
        self.handle.close()
        offsets = np.asarray(self.offsets, dtype=np.int64)
        lengths = np.asarray(self.lengths, dtype=np.int64)
        if len(offsets) == 0:
            index = np.empty((0, 2), dtype=np.int64)
        else:
            index = np.stack([offsets, lengths], axis=1)
        np.save(index_path, index)


class IndexedSequenceStore:
    """Random-access view over split sequences stored as seqbin + index."""

    def __init__(self, seqbin_path, index_path):
        self.seqbin_path = seqbin_path
        self.index_path = index_path
        self.index = np.load(index_path, mmap_mode="r")
        self.handle = open(seqbin_path, "rb")

    def __len__(self):
        return int(self.index.shape[0])

    def get(self, idx):
        offset, length = self.index[int(idx)]
        self.handle.seek(int(offset))
        payload = self.handle.read(int(length))
        return payload.decode("ascii")

    def close(self):
        if not self.handle.closed:
            self.handle.close()


class LegacyPickleSequenceStore:
    """Compatibility store for pre-existing pickle list datasets."""

    def __init__(self, pickle_path):
        with open(pickle_path, "rb") as handle:
            self.records = pickle.load(handle)

    def __len__(self):
        return len(self.records)

    def get(self, idx):
        return self.records[int(idx)]

    def close(self):
        return None


class CombinedSequenceStore:
    """Treat train/query/base stores as one concatenated logical dataset."""

    def __init__(self, stores, split_order=None, max_length=0):
        self.stores = stores
        self.split_order = split_order or list(stores.keys())
        self.max_length = max_length
        self.lengths = [len(self.stores[name]) for name in self.split_order]
        self.boundaries = np.cumsum(self.lengths)

    def __len__(self):
        if len(self.boundaries) == 0:
            return 0
        return int(self.boundaries[-1])

    def _locate(self, idx):
        idx = int(idx)
        split_pos = int(np.searchsorted(self.boundaries, idx, side="right"))
        prev = 0 if split_pos == 0 else int(self.boundaries[split_pos - 1])
        return self.split_order[split_pos], idx - prev

    def get(self, idx):
        split_name, local_idx = self._locate(idx)
        seq = self.stores[split_name].get(local_idx)
        if self.max_length:
            return seq[: self.max_length]
        return seq

    def iter_indices(self, indices):
        for idx in indices:
            yield self.get(idx)

    def close(self):
        for store in self.stores.values():
            close = getattr(store, "close", None)
            if close is not None:
                close()


def open_split_store(dataset_dir, split_name):
    paths = split_storage_paths(dataset_dir, split_name)
    if os.path.isfile(paths["seqbin"]) and os.path.isfile(paths["index"]):
        return IndexedSequenceStore(paths["seqbin"], paths["index"])
    if os.path.isfile(paths["legacy_pickle"]):
        return LegacyPickleSequenceStore(paths["legacy_pickle"])
    raise FileNotFoundError(
        "Could not find storage for split {!r} under {}.".format(split_name, dataset_dir)
    )
