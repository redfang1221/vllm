# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest performance cases for fused sigmoid-gating delta-rule update.

Kernel class: cv. The Triton call layer in
`vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py` does not use
`tl.dot`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

from vllm.model_executor.layers.fla.ops import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.triton_utils import triton


@dataclass(frozen=True)
class SigmoidGatingSpec:
    name: str
    dtype: torch.dtype
    num_reqs: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    num_speculative_tokens: int


SPECS = [
    # Existing correctness-test specs.
    SigmoidGatingSpec("non_spec_b1_fp32", torch.float32, 1, 16, 32, 128, 128,
                      0),
    SigmoidGatingSpec("non_spec_b2_bf16", torch.bfloat16, 2, 16, 32, 128, 128,
                      0),
    SigmoidGatingSpec("non_spec_b4_bf16", torch.bfloat16, 4, 16, 32, 128, 128,
                      0),
    SigmoidGatingSpec("spec_b2_t1_bf16", torch.bfloat16, 2, 16, 32, 128, 128,
                      1),
    SigmoidGatingSpec("spec_b4_t3_bf16", torch.bfloat16, 4, 16, 32, 128, 128,
                      3),
    # Larger value-head GDN config.
    SigmoidGatingSpec("non_spec_397b_b4_bf16", torch.bfloat16, 4, 16, 64,
                      128, 128, 0),
]


def perf_test(func: Callable[[], None]) -> float:
    warmup = int(os.getenv("VLLM_KERNEL_BENCH_WARMUP", "25"))
    rep = int(os.getenv("VLLM_KERNEL_BENCH_REP", "100"))
    return triton.testing.do_bench(func, warmup=warmup, rep=rep)


def build_data(spec: SigmoidGatingSpec) -> dict[str, object]:
    torch.manual_seed(0)
    device = torch.device("cuda")
    seq_len = spec.num_speculative_tokens + 1
    num_tokens = spec.num_reqs * seq_len
    total_entries = num_tokens + 1

    q = torch.rand(1,
                   num_tokens,
                   spec.num_k_heads,
                   spec.head_k_dim,
                   device=device,
                   dtype=spec.dtype)
    k = torch.rand_like(q)
    v = torch.rand(1,
                   num_tokens,
                   spec.num_v_heads,
                   spec.head_v_dim,
                   device=device,
                   dtype=spec.dtype)
    A_log = torch.rand(spec.num_v_heads, device=device, dtype=spec.dtype)
    dt_bias = torch.rand(spec.num_v_heads, device=device, dtype=spec.dtype)
    a = torch.rand(num_tokens,
                   spec.num_v_heads,
                   device=device,
                   dtype=spec.dtype)
    b = torch.rand_like(a)
    initial_state = torch.rand(
        total_entries,
        spec.num_v_heads,
        spec.head_v_dim,
        spec.head_k_dim,
        device=device,
        dtype=spec.dtype,
    )

    if spec.num_speculative_tokens == 0:
        ssm_state_indices = torch.arange(1,
                                         num_tokens + 1,
                                         device=device,
                                         dtype=torch.int32)
        cu_seqlens = torch.arange(0,
                                  num_tokens + 1,
                                  device=device,
                                  dtype=torch.int32)
        num_accepted_tokens = None
    else:
        ssm_state_indices = torch.arange(
            1,
            num_tokens + 1,
            device=device,
            dtype=torch.int32,
        ).view(spec.num_reqs, seq_len)
        cu_seqlens = torch.arange(0,
                                  num_tokens + 1,
                                  seq_len,
                                  device=device,
                                  dtype=torch.int32)
        num_accepted_tokens = torch.full((spec.num_reqs,),
                                         seq_len,
                                         device=device,
                                         dtype=torch.int32)

    return {
        "A_log": A_log,
        "a": a,
        "b": b,
        "dt_bias": dt_bias,
        "q": q,
        "k": k,
        "v": v,
        "initial_state": initial_state,
        "inplace_final_state": True,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
        "use_qk_l2norm_in_kernel": True,
    }


def run_performance(spec: SigmoidGatingSpec) -> float:
    data = build_data(spec)

    def func() -> None:
        fused_sigmoid_gating_delta_rule_update(**data)

    return perf_test(func)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("spec", SPECS, ids=[spec.name for spec in SPECS])
def test_fused_sigmoid_gating_delta_rule_update_kernel_perf(
    spec: SigmoidGatingSpec,
) -> None:
    ms = run_performance(spec)
    us = float(ms) * 1000
    print(f"fused_sigmoid_gating_delta_rule_update_kernel[{spec.name}]: "
          f"{us:.3f} us")
