# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest performance cases for packed GDN recurrent decode.

Kernel class: cv. The Triton call layer in
`vllm/model_executor/layers/fla/ops/fused_recurrent.py` does not use `tl.dot`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.triton_utils import triton


@dataclass(frozen=True)
class PackedDecodeSpec:
    name: str
    dtype: torch.dtype
    batch: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    strided_mixed_qkv: bool
    include_invalid_indices: bool


SPECS = [
    # Existing correctness-test shape.
    PackedDecodeSpec("test_f16_contiguous", torch.float16, 32, 4, 8, 128, 128,
                     False, True),
    PackedDecodeSpec("test_bf16_strided", torch.bfloat16, 32, 4, 8, 128, 128,
                     True, True),
    PackedDecodeSpec("test_fp32_contiguous", torch.float32, 32, 4, 8, 128, 128,
                     False, True),
    # GDN model decode shapes.
    PackedDecodeSpec("gdn_35b_b1", torch.bfloat16, 1, 16, 32, 128, 128, False,
                     False),
    PackedDecodeSpec("gdn_35b_b4", torch.bfloat16, 4, 16, 32, 128, 128, False,
                     False),
    PackedDecodeSpec("gdn_397b_b4", torch.bfloat16, 4, 16, 64, 128, 128, False,
                     False),
]


def perf_test(func: Callable[[], None]) -> float:
    warmup = int(os.getenv("VLLM_KERNEL_BENCH_WARMUP", "25"))
    rep = int(os.getenv("VLLM_KERNEL_BENCH_REP", "100"))
    return triton.testing.do_bench(func, warmup=warmup, rep=rep)


def build_data(spec: PackedDecodeSpec) -> dict[str, object]:
    torch.manual_seed(0)
    device = torch.device("cuda")
    qkv_dim = (2 * spec.num_k_heads * spec.head_k_dim +
               spec.num_v_heads * spec.head_v_dim)

    if spec.strided_mixed_qkv:
        proj = torch.randn(spec.batch,
                           qkv_dim + 64,
                           device=device,
                           dtype=spec.dtype)
        mixed_qkv = proj[:, :qkv_dim]
    else:
        mixed_qkv = torch.randn(spec.batch,
                                qkv_dim,
                                device=device,
                                dtype=spec.dtype)

    a = torch.randn(spec.batch,
                    spec.num_v_heads,
                    device=device,
                    dtype=spec.dtype)
    b = torch.randn_like(a)
    A_log = torch.randn(spec.num_v_heads, device=device, dtype=spec.dtype)
    dt_bias = torch.randn(spec.num_v_heads, device=device, dtype=spec.dtype)

    state_entries = spec.batch + 1
    initial_state = torch.randn(
        state_entries,
        spec.num_v_heads,
        spec.head_v_dim,
        spec.head_k_dim,
        device=device,
        dtype=spec.dtype,
    )
    ssm_state_indices = torch.arange(1,
                                     spec.batch + 1,
                                     device=device,
                                     dtype=torch.int32)
    if spec.include_invalid_indices and spec.batch >= 3:
        ssm_state_indices[-3:] = -1

    out = torch.empty(spec.batch,
                      1,
                      spec.num_v_heads,
                      spec.head_v_dim,
                      device=device,
                      dtype=spec.dtype)
    return {
        "mixed_qkv": mixed_qkv,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": spec.head_k_dim**-0.5,
        "initial_state": initial_state,
        "out": out,
        "ssm_state_indices": ssm_state_indices,
        "use_qk_l2norm_in_kernel": True,
    }


def run_performance(spec: PackedDecodeSpec) -> float:
    data = build_data(spec)

    def func() -> None:
        fused_recurrent_gated_delta_rule_packed_decode(**data)

    return perf_test(func)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("spec", SPECS, ids=[spec.name for spec in SPECS])
def test_fused_recurrent_gated_delta_rule_packed_decode_kernel_perf(
    spec: PackedDecodeSpec,
) -> None:
    ms = run_performance(spec)
    us = float(ms) * 1000
    name = "fused_recurrent_gated_delta_rule_packed_decode_kernel"
    print(f"{name}[{spec.name}]: {us:.3f} us")
