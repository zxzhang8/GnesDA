#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import hashlib
import argparse


def fasta_reader(fp):
    """
    流式读取 FASTA 文件
    返回: (header, sequence)
    """
    header = None
    seq_chunks = []

    for line in fp:
        line = line.strip()
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


def seq_digest(seq, ignore_case=True):
    """
    计算序列哈希值。
    为减少内存占用，不直接把整条序列放到 set 里，而是保存摘要值。
    """
    if ignore_case:
        seq = seq.upper()
    return hashlib.blake2b(seq.encode("ascii"), digest_size=16).digest()


def main():
    parser = argparse.ArgumentParser(
        description="检查 FASTA 文件中是否存在相同序列（序列名可不同）"
    )
    parser.add_argument("fasta", help="输入 FASTA 文件路径")
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        default=False,
        help="忽略大小写（默认不忽略）"
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="输出所有重复项；默认只要发现重复就持续统计并输出"
    )
    args = parser.parse_args()

    # digest -> first_header
    seen = {}
    duplicate_count = 0
    total_count = 0

    with open(args.fasta, "r", encoding="utf-8", errors="ignore") as f:
        for header, seq in fasta_reader(f):
            total_count += 1
            d = seq_digest(seq, ignore_case=args.ignore_case)

            if d in seen:
                duplicate_count += 1
                print(f"[重复] {header}")
                print(f"       与 {seen[d]} 序列内容相同")
            else:
                seen[d] = header

    print("\n====== 统计结果 ======")
    print(f"总序列数: {total_count}")
    print(f"唯一序列数: {len(seen)}")
    print(f"重复序列数: {duplicate_count}")

    if duplicate_count == 0:
        print("未发现相同序列。")


if __name__ == "__main__":
    main()
