import os
import json
import pickle
import argparse

import numpy as np
import torch

from dataset.datasets import StringDataset, word2sig
from utils.fasta import load_dna_fasta


def batch_embed(model, vecs, batch_size, device):
    loader = torch.utils.data.DataLoader(vecs, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    embedding = []
    with torch.no_grad():
        for x in loader:
            embedding.append(model(x.to(device)).cpu().data.numpy())
    return np.concatenate(embedding, axis=0)


def infer_counts(dataset):
    metadata_path = os.path.join("data", dataset, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise ValueError("Could not infer nt/nq without data/{}/metadata.json".format(dataset))
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return metadata["train_size"], metadata["query_size"]


def infer_max_len(dataset):
    metadata_path = os.path.join("data", dataset, "metadata.json")
    if not os.path.isfile(metadata_path):
        data_dir = os.path.join("data", dataset)
        seqs = []
        for name in ("train_seq_list", "query_seq_list", "base_seq_list"):
            path = os.path.join(data_dir, name)
            if os.path.isfile(path):
                with open(path, "rb") as handle:
                    seqs.extend(pickle.load(handle))
        if not seqs:
            raise ValueError("No reference sequences found under data/{}".format(dataset))
        _, max_len, _, _ = word2sig(
            seqs,
            max_length=None,
            allowed_chars="ACGT",
            fixed_alphabet="ACGT",
        )
        return max_len

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return metadata["max_sequence_length"]


def main():
    parser = argparse.ArgumentParser(description="Embed DNA FASTA sequences with a trained GnesDA model")
    parser.add_argument("--dataset", type=str, required=True, help="dataset name used during training")
    parser.add_argument("--dist-type", type=str, default="ed", help="distance type used during training")
    parser.add_argument("--input-fasta", type=str, required=True, help="DNA FASTA to embed")
    parser.add_argument("--model-file", type=str, default="", help="optional explicit model path")
    parser.add_argument("--nt", type=int, default=0, help="training set size used in model path")
    parser.add_argument("--nq", type=int, default=0, help="query set size used in model path")
    parser.add_argument("--batch-size", type=int, default=32, help="inference batch size")
    parser.add_argument("--output-prefix", type=str, default="", help="save prefix; defaults beside FASTA")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disable GPU inference")
    args = parser.parse_args()

    if args.nt == 0 or args.nq == 0:
        args.nt, args.nq = infer_counts(args.dataset)

    model_file = args.model_file or "knn/{}/{}/nt{}_nq{}/model.torch".format(
        args.dataset,
        args.dist_type,
        args.nt,
        args.nq,
    )
    if not os.path.isfile(model_file):
        raise FileNotFoundError("Model file not found: {}".format(model_file))

    max_len = infer_max_len(args.dataset)

    seq_ids, seqs = load_dna_fasta(args.input_fasta)
    too_long = [(seq_id, len(seq)) for seq_id, seq in zip(seq_ids, seqs) if len(seq) > max_len]
    if too_long:
        example_id, example_len = too_long[0]
        raise ValueError(
            "Input sequence {!r} has length {}, exceeding trained max length {}.".format(
                example_id, example_len, max_len
            )
        )

    C, M, char_ids, _ = word2sig(
        seqs,
        max_length=max_len,
        allowed_chars="ACGT",
        fixed_alphabet="ACGT",
    )
    vecs = StringDataset(C, M, char_ids, data_type="dna")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    model = torch.load(model_file, map_location=device, weights_only=False)
    embedding = batch_embed(model.embedding_net, vecs, args.batch_size, device)

    output_prefix = args.output_prefix or os.path.splitext(args.input_fasta)[0]
    np.save(output_prefix + "_embedding.npy", embedding)
    with open(output_prefix + "_ids.json", "w", encoding="utf-8") as handle:
        json.dump(seq_ids, handle, indent=2)

    print("model_file={}".format(model_file))
    print("num_sequences={}".format(len(seq_ids)))
    print("embedding_shape={}".format(tuple(embedding.shape)))
    print("saved_embedding={}".format(output_prefix + "_embedding.npy"))
    print("saved_ids={}".format(output_prefix + "_ids.json"))


if __name__ == "__main__":
    main()
