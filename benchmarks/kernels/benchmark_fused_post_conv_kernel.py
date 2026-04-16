# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest performance cases for `_fused_post_conv_kernel`.

Kernel class: cv. The Triton call layer in
`vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py` does not
use `tl.dot`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

from vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv import (
    fused_post_conv_prep,
)
from vllm.triton_utils import triton


@dataclass(frozen=True)
class FusedPostConvSpec:
    name: str
    dtype: torch.dtype
    length: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    apply_l2norm: bool
    output_g_exp: bool


SPECS = [
    # Existing correctness-test specs from tests/kernels/test_fused_gdn_post_conv.py.
    FusedPostConvSpec("small_l1_l2", torch.bfloat16, 1, 4, 8, 64, 64, True,
                      False),
    FusedPostConvSpec("small_l128_no_l2", torch.bfloat16, 128, 4, 8, 64, 64,
                      False, False),
    FusedPostConvSpec("gdn_35b_l16", torch.bfloat16, 16, 16, 32, 128, 128,
                      True, False),
    FusedPostConvSpec("gdn_35b_l512_exp_g", torch.bfloat16, 512, 16, 32, 128,
                      128, True, True),
    FusedPostConvSpec("gdn_35b_l2048", torch.bfloat16, 2048, 16, 32, 128,
                      128, True, False),
    FusedPostConvSpec("gdn_397b_l512", torch.bfloat16, 512, 16, 64, 128, 128,
                      True, False),
]


def perf_test(func: Callable[[], None]) -> float:
    warmup = int(os.getenv("VLLM_KERNEL_BENCH_WARMUP", "25"))
    rep = int(os.getenv("VLLM_KERNEL_BENCH_REP", "100"))
    return triton.testing.do_bench(func, warmup=warmup, rep=rep)


def build_data(spec: FusedPostConvSpec) -> dict[str, object]:
    torch.manual_seed(0)
    qkv_dim = (2 * spec.num_k_heads * spec.head_k_dim +
               spec.num_v_heads * spec.head_v_dim)
    conv_output = torch.randn(spec.length,
                              qkv_dim,
                              device="cuda",
                              dtype=spec.dtype)
    a = torch.randn(spec.length,
                    spec.num_v_heads,
                    device="cuda",
                    dtype=spec.dtype)
    b = torch.randn_like(a)
    A_log = torch.randn(spec.num_v_heads,
                        device="cuda",
                        dtype=torch.float32) - 2.0
    dt_bias = torch.randn(spec.num_v_heads,
                          device="cuda",
                          dtype=torch.float32) * 0.1
    return {
        "conv_output": conv_output,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "num_k_heads": spec.num_k_heads,
        "head_k_dim": spec.head_k_dim,
        "head_v_dim": spec.head_v_dim,
        "apply_l2norm": spec.apply_l2norm,
        "output_g_exp": spec.output_g_exp,
    }


def run_performance(spec: FusedPostConvSpec) -> float:
    data = build_data(spec)

    def func() -> None:
        fused_post_conv_prep(**data)

    return perf_test(func)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("spec", SPECS, ids=[spec.name for spec in SPECS])
def test_fused_post_conv_kernel_perf(spec: FusedPostConvSpec) -> None:
    ms = run_performance(spec)
    us = float(ms) * 1000
    print(f"_fused_post_conv_kernel[{spec.name}]: {us:.3f} us")
