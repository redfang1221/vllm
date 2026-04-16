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
import torch_npu
import triton

from vllm.model_executor.layers.fla.ops import (
    fused_sigmoid_gating_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update_kernel
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
    device = torch.device("npu")
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


def get_input_args(data):
    beta: float = 1.0
    threshold: float = 20.0
    scale: float = None
    initial_state: torch.Tensor = None
    inplace_final_state: bool = True
    cu_seqlens: torch.Tensor | None = None
    ssm_state_indices: torch.Tensor | None = None
    num_accepted_tokens: torch.Tensor | None = None
    use_qk_l2norm_in_kernel: bool = False
    is_kda: bool = False
    A_log = data["A_log"]
    a = data["a"]
    b = data["b"]
    dt_bias = data["dt_bias"]
    q = data["q"]
    k = data["k"]
    v = data["v"]
    initial_state = data["initial_state"]
    inplace_final_state = data["inplace_final_state"]
    cu_seqlens = data["cu_seqlens"]
    ssm_state_indices = data["ssm_state_indices"]
    num_accepted_tokens = data["num_accepted_tokens"]
    use_qk_l2norm_in_kernel = data["use_qk_l2norm_in_kernel"]

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 4

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]}"
            f" when using `cu_seqlens`. Please flatten variable-length"
            f" inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    return {"grid": grid, "input_data": {
        "A_log": A_log, "a": a.contiguous(), "b": b.contiguous(), "dt_bias": dt_bias, "beta": beta, "threshold": threshold, "q": q.contiguous(), "k": k.contiguous(), "v": v.contiguous(), "o": o, "h0": initial_state, "ht": final_state, "cu_seqlens": cu_seqlens, "ssm_state_indices": ssm_state_indices, "num_accepted_tokens": num_accepted_tokens, "scale": scale, "N": N, "T": T, "B": B, "H": H, "HV": HV, "K": K, "V": V, "BK": BK, "BV": BV, "stride_init_state_token": stride_init_state_token, "stride_final_state_token": stride_final_state_token, "stride_indices_seq": stride_indices_seq, "stride_indices_tok": stride_indices_tok, "INPLACE_FINAL_STATE": inplace_final_state, "USE_QK_L2NORM_IN_KERNEL": use_qk_l2norm_in_kernel, "IS_KDA": is_kda, "num_warps": num_warps, "num_stages": num_stages
    }}


def perf_test(fn_triton, args, save_path="./result_dir"):
    experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
        )
    with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.NPU],
            with_stack=False,
            record_shapes=False,
            profile_memory=False,
            schedule=torch_npu.profiler.schedule(wait=1,
                                                warmup=1,
                                                active=30,
                                                repeat=1,
                                                skip_first=1),
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(save_path)
    ) as prof:
        for i in range(30):
            fn_triton(**args)
            torch.npu.synchronize()
            prof.step()


def fn_triton(grid, input_data):
    fused_sigmoid_gating_delta_rule_update_kernel[grid](**input_data)


def run_performance(spec: CausalConv1dUpdateSpec) -> float:
    data = build_data(spec)
    args = get_input_args(data)
    return perf_test(fn_triton, args, "fused_sigmoid_gating_delta_rule_update_kernel_perf")


@pytest.mark.skipif(not torch.npu.is_available(), reason="Need NPU device")
@pytest.mark.parametrize("spec", SPECS, ids=[spec.name for spec in SPECS])
def test_fused_sigmoid_gating_delta_rule_update_kernel_perf(
    spec: SigmoidGatingSpec,
) -> None:
    run_performance(spec)
