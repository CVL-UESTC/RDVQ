#!/usr/bin/env python3
"""Validate tensor-native rANS roundtrip and optional CompressAI comparison."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn.functional as F
from compressai.ans import BufferedRansEncoder, RansDecoder

from tokenizer.tokenizer_image.codec.entropy_coding.native.fast_cdf import batch_pmf_to_quantized_cdf, cdf_to_compressai_lists
from tokenizer.tokenizer_image.codec.entropy_coding.native.tensor_rans import decode_indexed, encode_indexed


def _check_cdf(cdf: torch.Tensor, precision: int) -> None:
    total = 1 << precision
    if cdf.dtype != torch.int32:
        raise AssertionError(f"CDF dtype must be int32, got {cdf.dtype}")
    if not torch.all(cdf[:, 0] == 0):
        raise AssertionError("CDF rows must start at 0")
    if not torch.all(cdf[:, -1] == total):
        raise AssertionError(f"CDF rows must end at {total}")
    freq = torch.diff(cdf.to(torch.int64), dim=-1)
    if not torch.all(freq > 0):
        raise AssertionError("All quantized frequencies must be positive")


def _compressai_roundtrip(symbols: torch.Tensor, cdf: torch.Tensor) -> int:
    cdf_list, cdf_len_list = cdf_to_compressai_lists(cdf)
    sym_list = symbols.to(torch.int32).cpu().tolist()
    indexes = list(range(len(sym_list)))
    offsets = [0] * len(sym_list)
    encoder = BufferedRansEncoder()
    encoder.encode_with_indexes(sym_list, indexes, cdf_list, cdf_len_list, offsets)
    stream = encoder.flush()
    decoder = RansDecoder()
    decoder.set_stream(stream)
    decoded = decoder.decode_stream(indexes, cdf_list, cdf_len_list, offsets)
    if decoded != sym_list:
        raise AssertionError("CompressAI roundtrip mismatch")
    return len(stream) * 8


def _tensor_roundtrip(symbols: torch.Tensor, cdf: torch.Tensor, precision: int) -> int:
    stream = encode_indexed(symbols, cdf, precision=precision)
    decoded = decode_indexed(stream, cdf, precision=precision)
    if not torch.equal(decoded.to(torch.int32), symbols.to(torch.int32).cpu()):
        raise AssertionError("tensor rANS roundtrip mismatch")
    return len(stream) * 8


def _case_random(rows: int, symbols: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(rows, symbols, generator=generator)
    return F.softmax(logits, dim=-1)


def _case_spiky(rows: int, symbols: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pmf = torch.full((rows, symbols), 1e-12)
    peaks = torch.randint(0, symbols, (rows,), generator=generator)
    pmf[torch.arange(rows), peaks] = 1.0
    return pmf / pmf.sum(dim=-1, keepdim=True)


def _case_near_uniform(rows: int, symbols: int) -> torch.Tensor:
    return torch.ones(rows, symbols) / symbols


def _run_case(name: str, pmf: torch.Tensor, precision: int, compare_compressai: bool, benchmark: bool) -> None:
    cdf = batch_pmf_to_quantized_cdf(pmf, precision)
    _check_cdf(cdf, precision)
    symbols = torch.multinomial(pmf, 1).view(-1).to(torch.int32)

    start = time.perf_counter()
    tensor_bits = _tensor_roundtrip(symbols, cdf, precision)
    tensor_time = time.perf_counter() - start

    line = f"{name}: rows={pmf.shape[0]} symbols={pmf.shape[1]} tensor_bits={tensor_bits} tensor_time={tensor_time:.6f}s"
    if compare_compressai:
        start = time.perf_counter()
        compressai_bits = _compressai_roundtrip(symbols, cdf)
        compressai_time = time.perf_counter() - start
        line += f" compressai_bits={compressai_bits} compressai_time={compressai_time:.6f}s delta_bits={tensor_bits - compressai_bits}"
    print(line)

    if benchmark:
        repeats = 5
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            _tensor_roundtrip(symbols, cdf, precision)
            best = min(best, time.perf_counter() - start)
        print(f"{name}: tensor_best_roundtrip={best:.6f}s repeats={repeats}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--symbols", type=int, default=1025)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--compare-compressai", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cases = {
        "random": _case_random(args.rows, args.symbols, args.seed),
        "spiky": _case_spiky(args.rows, args.symbols, args.seed + 1),
        "near_uniform": _case_near_uniform(args.rows, args.symbols),
    }

    for name, pmf in cases.items():
        _run_case(name, pmf, args.precision, args.compare_compressai, args.benchmark)


if __name__ == "__main__":
    main()
