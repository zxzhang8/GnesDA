import os
import json
import pickle
import random

import numpy as np

from utils.sequence_store import IndexedSequenceStore, SequenceBinWriter, split_storage_paths


DNA_ALPHABET = set("ACGT")


def iter_fasta_records(path):
    header = None
    seq_chunks = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        yield header, "".join(seq_chunks)


def parse_fasta(path):
    return list(iter_fasta_records(path))


def _validate_dna(records):
    seq_ids = []
    seqs = []
    for header, seq in records:
        seq = seq.upper()
        invalid = sorted(set(seq) - DNA_ALPHABET)
        if invalid:
            raise ValueError(
                "Invalid DNA bases {} found in FASTA record {!r}. Only A/C/G/T are supported.".format(
                    invalid, header
                )
            )
        seq_ids.append(header)
        seqs.append(seq)
    return seq_ids, seqs


def load_dna_fasta(path):
    return _validate_dna(parse_fasta(path))


def _split_eval_sequences(eval_ids, eval_seqs, seed, query_ratio):
    if len(eval_seqs) < 2:
        raise ValueError("Evaluation FASTA must contain at least 2 sequences to create query/base splits.")
    if not (0.0 < query_ratio < 1.0):
        raise ValueError("eval_query_ratio must be in (0, 1).")

    order = list(range(len(eval_seqs)))
    random.Random(seed).shuffle(order)
    split_at = int(round(len(order) * query_ratio))
    split_at = min(max(split_at, 1), len(order) - 1)

    query_idx = order[:split_at]
    base_idx = order[split_at:]
    query_ids = [eval_ids[i] for i in query_idx]
    query_seqs = [eval_seqs[i] for i in query_idx]
    base_ids = [eval_ids[i] for i in base_idx]
    base_seqs = [eval_seqs[i] for i in base_idx]
    return query_ids, query_seqs, base_ids, base_seqs


def _split_eval_indices(eval_count, seed, query_ratio):
    if eval_count < 2:
        raise ValueError("Evaluation FASTA must contain at least 2 sequences to create query/base splits.")
    if not (0.0 < query_ratio < 1.0):
        raise ValueError("eval_query_ratio must be in (0, 1).")

    order = list(range(eval_count))
    random.Random(seed).shuffle(order)
    split_at = int(round(len(order) * query_ratio))
    split_at = min(max(split_at, 1), len(order) - 1)
    query_idx = order[:split_at]
    base_idx = order[split_at:]
    return query_idx, base_idx


def _iter_validated_dna_records(path):
    for header, seq in iter_fasta_records(path):
        seq = seq.upper()
        invalid = sorted(set(seq) - DNA_ALPHABET)
        if invalid:
            raise ValueError(
                "Invalid DNA bases {} found in FASTA record {!r}. Only A/C/G/T are supported.".format(
                    invalid, header
                )
            )
        yield header, seq


def _finalize_writer(writer, index_path):
    try:
        writer.close(index_path)
    except Exception:
        writer.handle.close()
        raise


def _stream_fasta_to_split(path, writer, collect_ids):
    count = 0
    max_length = 0
    for header, seq in _iter_validated_dna_records(path):
        collect_ids.append(header)
        writer.append(seq)
        count += 1
        if len(seq) > max_length:
            max_length = len(seq)
    return count, max_length


def _collect_eval_ids(path):
    eval_ids = []
    max_length = 0
    for header, seq in _iter_validated_dna_records(path):
        eval_ids.append(header)
        if len(seq) > max_length:
            max_length = len(seq)
    return eval_ids, max_length


def export_legacy_pickle_split(dataset_dir, split_name):
    paths = split_storage_paths(dataset_dir, split_name)
    records = []
    index = np.load(paths["index"])
    with open(paths["seqbin"], "rb") as handle:
        for offset, length in index:
            handle.seek(int(offset))
            records.append(handle.read(int(length)).decode("ascii"))
    with open(paths["legacy_pickle"], "wb") as handle:
        pickle.dump(records, handle)


def prepare_dna_dataset(dataset, train_fasta=None, eval_fasta=None, query_fasta=None, base_fasta=None,
                        seed=666, eval_query_ratio=0.5):
    if not any([train_fasta, eval_fasta, query_fasta, base_fasta]):
        return None

    dataset_dir = os.path.join("data", dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    train_paths = split_storage_paths(dataset_dir, "train")
    query_paths = split_storage_paths(dataset_dir, "query")
    base_paths = split_storage_paths(dataset_dir, "base")

    if train_fasta and eval_fasta:
        train_ids = []
        train_writer = SequenceBinWriter(train_paths["seqbin"])
        train_size, train_max = _stream_fasta_to_split(train_fasta, train_writer, train_ids)
        _finalize_writer(train_writer, train_paths["index"])

        eval_tmp_seqbin = os.path.join(dataset_dir, "_eval_tmp.seqbin")
        eval_tmp_index = os.path.join(dataset_dir, "_eval_tmp.idx.npy")
        eval_tmp_writer = SequenceBinWriter(eval_tmp_seqbin)
        eval_ids = []
        _, eval_max = _stream_fasta_to_split(eval_fasta, eval_tmp_writer, eval_ids)
        _finalize_writer(eval_tmp_writer, eval_tmp_index)
        query_idx, base_idx = _split_eval_indices(len(eval_ids), seed=seed, query_ratio=eval_query_ratio)
        query_ids = []
        base_ids = []
        query_writer = SequenceBinWriter(query_paths["seqbin"])
        base_writer = SequenceBinWriter(base_paths["seqbin"])
        eval_store = IndexedSequenceStore(eval_tmp_seqbin, eval_tmp_index)
        try:
            for idx in query_idx:
                query_ids.append(eval_ids[idx])
                query_writer.append(eval_store.get(idx))
            for idx in base_idx:
                base_ids.append(eval_ids[idx])
                base_writer.append(eval_store.get(idx))
        finally:
            eval_store.close()
            for tmp_path in (eval_tmp_seqbin, eval_tmp_index):
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        _finalize_writer(query_writer, query_paths["index"])
        _finalize_writer(base_writer, base_paths["index"])
        query_size = len(query_ids)
        base_size = len(base_ids)
        max_sequence_length = max(train_max, eval_max)
    elif train_fasta and query_fasta and base_fasta:
        train_ids = []
        train_writer = SequenceBinWriter(train_paths["seqbin"])
        train_size, train_max = _stream_fasta_to_split(train_fasta, train_writer, train_ids)
        _finalize_writer(train_writer, train_paths["index"])

        query_ids = []
        query_writer = SequenceBinWriter(query_paths["seqbin"])
        query_size, query_max = _stream_fasta_to_split(query_fasta, query_writer, query_ids)
        _finalize_writer(query_writer, query_paths["index"])

        base_ids = []
        base_writer = SequenceBinWriter(base_paths["seqbin"])
        base_size, base_max = _stream_fasta_to_split(base_fasta, base_writer, base_ids)
        _finalize_writer(base_writer, base_paths["index"])
        max_sequence_length = max(train_max, query_max, base_max)
    else:
        raise ValueError(
            "Provide either (--train-fasta and --eval-fasta) or "
            "(--train-fasta and --query-fasta and --base-fasta)."
        )

    metadata = {
        "dataset": dataset,
        "train_fasta": train_fasta,
        "eval_fasta": eval_fasta,
        "query_fasta": query_fasta,
        "base_fasta": base_fasta,
        "train_size": train_size,
        "query_size": query_size,
        "base_size": base_size,
        "train_ids": train_ids,
        "query_ids": query_ids,
        "base_ids": base_ids,
        "max_sequence_length": max_sequence_length,
        "split_seed": seed,
        "eval_query_ratio": eval_query_ratio,
        "storage_format": "seqbin_v1",
        "train_storage": {
            "seqbin": os.path.basename(train_paths["seqbin"]),
            "index": os.path.basename(train_paths["index"]),
        },
        "query_storage": {
            "seqbin": os.path.basename(query_paths["seqbin"]),
            "index": os.path.basename(query_paths["index"]),
        },
        "base_storage": {
            "seqbin": os.path.basename(base_paths["seqbin"]),
            "index": os.path.basename(base_paths["index"]),
        },
    }

    with open(os.path.join(dataset_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata
