# General Neural Embedding for Sequence Distance Approximation
The code officially implements the GnesDA algorithm for sequence distance approximation.
## datasets
The Geolife dataset and groundtruth can be found [https://drive.google.com/file/d/1mVQgXcXoCuICZFF5N3ySaz0fwGclQO-P/view](https://drive.google.com/file/d/1mVQgXcXoCuICZFF5N3ySaz0fwGclQO-P/view)

The Porto dataset and groundtruth can be found  [https://drive.google.com/file/d/1s_XGg0hq-eOqGgohHfpEqOW65_W2uuS7/view](https://drive.google.com/file/d/1s_XGg0hq-eOqGgohHfpEqOW65_W2uuS7/view)(Due to disk space limitations, only the DTW distance is included in this link. If you want to get other distance groundtruth, please contact me.)

The Uniprot and Uniref dataset and groundtruth can be found at [https://drive.google.com/file/d/17GvzqDPV6ZTIh9nwHHvnXp4RNcdQA98n/view?usp=drive_link](https://drive.google.com/file/d/17GvzqDPV6ZTIh9nwHHvnXp4RNcdQA98n/view?usp=drive_link)

## DNA support
DNA sequences can be trained with `--data_type dna`.

- Input sequences must contain only uppercase `A`, `C`, `G`, `T`.
- `N` and any other characters are rejected during preprocessing.
- Supported distance types for DNA are `ed` and `nw`.
- For `nw`, the code uses nucleotide alignment mode (`moltype="nucl"`).

Expected dataset layout for a DNA dataset named `dna`:

```text
data/dna/train_seq_list
data/dna/query_seq_list
data/dna/base_seq_list
```

Example:

```bash
python main.py --data_type dna --dataset dna --dist_type ed
python main.py --data_type dna --dataset dna --dist_type nw
```

## Alpha satellite monomer workflow
This repository can be adapted to alpha satellite monomer embeddings if you train on monomer edit-distance supervision.

Expected behavior:

- `--dist_type ed` fits embedding distances to Levenshtein edit distance.
- DNA FASTA input must contain only uppercase `A`, `C`, `G`, `T`.
- Correlation metrics between embedding distance and true edit distance are printed during evaluation.
- Default DNA settings are tuned toward monomer-scale sequences, with `conv_layers=3`.

If you already have monomer FASTA files such as `inputs/monomers_train_all.fa` and `inputs/monomers_eval_all.fa`, run:

```bash
python main.py \
  --data_type dna \
  --dataset monomer \
  --dist_type ed \
  --train-fasta inputs/monomers_train_all.fa \
  --eval-fasta inputs/monomers_eval_all.fa \
  --save-model \
  --save-embed
```

This command will:

- create `data/monomer/train_seq_list`, `query_seq_list`, and `base_seq_list`
- split the eval FASTA into query/base using `--shuffle-seed` and `--eval-query-ratio`
- train a GnesDA model against edit distance
- save embeddings plus `distance_metrics.json`

To embed a separate FASTA with a trained model:

```bash
python infer_fasta_embeddings.py \
  --dataset monomer \
  --dist-type ed \
  --input-fasta inputs/monomers_eval_all.fa
```
