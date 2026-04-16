# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest performance cases for `_causal_conv1d_update_kernel`.

Kernel class: cv. The Triton call layer in
`vllm/model_executor/layers/mamba/ops/causal_conv1d.py` does not use `tl.dot`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@dataclass(frozen=True)
class CausalConv1dUpdateSpec:
    name: str
    dtype: torch.dtype
    batch: int
    dim: int
    seqlen: int
    width: int
    has_bias: bool
    activation: str | None
    mode: str


SPECS = [
    # Existing correctness specs from tests/kernels/mamba/test_causal_conv1d.py.
    CausalConv1dUpdateSpec("decode_bf16_d2048", torch.bfloat16, 2, 2048, 1, 4,
                           True, "silu", "decode_3d"),
    CausalConv1dUpdateSpec("decode_bf16_d2064", torch.bfloat16, 2, 2064, 1, 4,
                           False, None, "decode_3d"),
    CausalConv1dUpdateSpec("decode_bf16_d4096", torch.bfloat16, 2, 4096, 1, 4,
                           True, None, "decode_3d"),
    CausalConv1dUpdateSpec("gather_fp32_s1_w3", torch.float32, 3, 2064, 1, 3,
                           True, "silu", "gather"),
    CausalConv1dUpdateSpec("gather_bf16_s3_w4", torch.bfloat16, 3, 4096, 3, 4,
                           False, None, "gather"),
    # GDN decode path uses a 2D [num_tokens, qkv_dim] input.
    CausalConv1dUpdateSpec("gdn_35b_qkv_decode", torch.bfloat16, 4, 8192, 1, 4,
                           True, "silu", "decode_2d"),
    CausalConv1dUpdateSpec("gdn_397b_qkv_decode", torch.bfloat16, 4, 12288, 1,
                           4, True, "silu", "decode_2d"),
    # Speculative decode path uses varlen flattened inputs.
    CausalConv1dUpdateSpec("varlen_spec_bf16", torch.bfloat16, 4, 8192, 4, 4,
                           True, "silu", "varlen_spec"),
]


def perf_test(func: Callable[[], None]) -> float:
    warmup = int(os.getenv("VLLM_KERNEL_BENCH_WARMUP", "25"))
    rep = int(os.getenv("VLLM_KERNEL_BENCH_REP", "100"))
    return triton.testing.do_bench(func, warmup=warmup, rep=rep)


def _randn(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype)


def build_data(spec: CausalConv1dUpdateSpec) -> dict[str, object]:
    torch.manual_seed(0)
    total_entries = max(spec.batch + 1, spec.batch * 8)
    weight = _randn(spec.dim, spec.width, dtype=spec.dtype)
    bias = _randn(spec.dim, dtype=spec.dtype) if spec.has_bias else None
    state_len = spec.width - 1
    if spec.mode == "varlen_spec":
        state_len += spec.seqlen - 1
    if spec.mode == "gather":
        conv_state = _randn(total_entries, state_len, spec.dim,
                            dtype=spec.dtype).transpose(1, 2)
    else:
        conv_state = _randn(total_entries,
                            spec.dim,
                            state_len,
                            dtype=spec.dtype)

    if spec.mode == "decode_2d":
        x = _randn(spec.batch, spec.dim, dtype=spec.dtype)
        indices = torch.arange(1,
                               spec.batch + 1,
                               device="cuda",
                               dtype=torch.int32)
        return {
            "x": x,
            "conv_state": conv_state,
            "weight": weight,
            "bias": bias,
            "activation": spec.activation,
            "conv_state_indices": indices,
        }

    if spec.mode == "varlen_spec":
        tokens_per_req = spec.seqlen
        total_tokens = spec.batch * tokens_per_req
        x = _randn(total_tokens, spec.dim, dtype=spec.dtype)
        state_indices = torch.arange(
            1,
            spec.batch * tokens_per_req + 1,
            device="cuda",
            dtype=torch.int32,
        ).view(spec.batch, tokens_per_req)
        query_start_loc = torch.arange(
            0,
            total_tokens + 1,
            tokens_per_req,
            device="cuda",
            dtype=torch.int32,
        )
        num_accepted_tokens = torch.full((spec.batch,),
                                         tokens_per_req,
                                         device="cuda",
                                         dtype=torch.int32)
        return {
            "x": x,
            "conv_state": conv_state,
            "weight": weight,
            "bias": bias,
            "activation": spec.activation,
            "conv_state_indices": state_indices,
            "num_accepted_tokens": num_accepted_tokens,
            "query_start_loc": query_start_loc,
            "max_query_len": tokens_per_req,
        }

    padding = 5 if spec.mode == "gather" else 0
    padded_batch = spec.batch + padding
    x = _randn(padded_batch, spec.seqlen, spec.dim,
               dtype=spec.dtype).transpose(1, 2)
    valid_indices = torch.arange(1,
                                 spec.batch + 1,
                                 device="cuda",
                                 dtype=torch.int32)
    if padding:
        pad = torch.full((padding,),
                         NULL_BLOCK_ID,
                         device="cuda",
                         dtype=torch.int32)
        indices = torch.cat([valid_indices, pad])
    else:
        indices = valid_indices
    return {
        "x": x,
        "conv_state": conv_state,
        "weight": weight,
        "bias": bias,
        "activation": spec.activation,
        "conv_state_indices": indices,
    }


def run_performance(spec: CausalConv1dUpdateSpec) -> float:
    data = build_data(spec)
    # if you need to print detail info about generated data
    # for key, value in data.items():
    #     print(key, ": ", f"{value.shape}, {value.dtype}" if isinstance(value, torch.Tensor) else value)
    def func() -> None:
        causal_conv1d_update(**data)

    return perf_test(func)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("spec", SPECS, ids=[spec.name for spec in SPECS])
def test_causal_conv1d_update_kernel_perf(
    spec: CausalConv1dUpdateSpec,
) -> None:
    ms = run_performance(spec)
    us = float(ms) * 1000
    print(f"_causal_conv1d_update_kernel[{spec.name}]: {us:.3f} us")
