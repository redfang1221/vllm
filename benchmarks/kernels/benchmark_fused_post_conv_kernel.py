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
from pathlib import Path

import pytest
import torch
import torch_npu
import triton

from vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv import (
    fused_post_conv_prep,
    _fused_post_conv_kernel
)
from vllm.triton_utils import triton

DUMP_NAME = "_fused_post_conv"


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


def configure_triton_dump(spec_name: str) -> None:
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["TRITON_DEBUG"] = "1"
    os.environ["TRITON_KERNEL_DUMP"] = "1"
    dump_dir = f"./{DUMP_NAME}_{spec_name}_cache"
    Path(dump_dir).mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_DUMP_DIR"] = dump_dir


def build_data(spec: FusedPostConvSpec) -> dict[str, object]:
    torch.manual_seed(0)
    qkv_dim = (2 * spec.num_k_heads * spec.head_k_dim +
               spec.num_v_heads * spec.head_v_dim)
    conv_output = torch.randn(spec.length,
                              qkv_dim,
                              device="npu",
                              dtype=spec.dtype)
    a = torch.randn(spec.length,
                    spec.num_v_heads,
                    device="npu",
                    dtype=spec.dtype)
    b = torch.randn_like(a)
    A_log = torch.randn(spec.num_v_heads,
                        device="npu",
                        dtype=torch.float32) - 2.0
    dt_bias = torch.randn(spec.num_v_heads,
                          device="npu",
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


def get_input_args(data):
    apply_l2norm: bool = True
    output_g_exp: bool = False
    conv_output = data["conv_output"]
    a = data["a"]
    b = data["b"]
    A_log = data["A_log"]
    dt_bias = data["dt_bias"]
    num_k_heads = data["num_k_heads"]
    head_k_dim = data["head_k_dim"]
    head_v_dim = data["head_v_dim"]
    apply_l2norm = data["apply_l2norm"]
    output_g_exp = data["output_g_exp"]
    L = conv_output.shape[0]
    qkv_dim = conv_output.shape[1]
    H = num_k_heads
    K = head_k_dim
    V = head_v_dim
    HV = A_log.shape[0]
    dtype = conv_output.dtype
    device = conv_output.device

    assert qkv_dim == 2 * H * K + HV * V, (
        f"qkv_dim={qkv_dim} != 2*H*K + HV*V = {2 * H * K + HV * V}"
    )

    # Allocate outputs in target contiguous layout
    q = torch.empty(L, H, K, dtype=dtype, device=device)
    k = torch.empty(L, H, K, dtype=dtype, device=device)
    v = torch.empty(L, HV, V, dtype=dtype, device=device)
    g = torch.empty(L, HV, dtype=torch.float32, device=device)
    beta = torch.empty(L, HV, dtype=torch.float32, device=device)

    if L == 0:
        return q, k, v, g, beta

    # ---- Kernel config ----
    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    BLOCK_T = 16  # tokens per block

    # Single kernel: blocks [0,H) do Q/K, blocks [H, H+HV) do V+gating
    grid = (triton.cdiv(L, BLOCK_T), H + HV)
    return {"grid": grid, "input_data": {
        "mixed_qkv_ptr": conv_output, "a_ptr": a, "b_ptr": b, "A_log_ptr": A_log, "dt_bias_ptr": dt_bias, "q_ptr": q, "k_ptr": k, "v_ptr": v, "g_ptr": g, "beta_ptr": beta, "stride_x_tok": conv_output.stride(0), "stride_a_tok": a.stride(0), "stride_b_tok": b.stride(0), "stride_q_tok": q.stride(0), "stride_k_tok": k.stride(0), "stride_v_tok": v.stride(0), "L": L, "H": H, "HV": HV, "K": K, "V": V, "APPLY_L2NORM": apply_l2norm, "L2NORM_EPS": 1e-6, "OUTPUT_G_EXP": output_g_exp, "SOFTPLUS_THRESHOLD": 20.0, "BLOCK_T": BLOCK_T, "BK": BK, "BV": BV, "num_warps": 4, "num_stages": 2
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
    _fused_post_conv_kernel[grid](**input_data)


def run_performance(spec: FusedPostConvSpec) -> float:
    configure_triton_dump(spec.name)
    data = build_data(spec)
    args = get_input_args(data)
    return perf_test(fn_triton, args, "fused_post_conv_kernel_perf")


@pytest.mark.skipif(not torch.npu.is_available(), reason="Need NPU device")
@pytest.mark.parametrize("spec", SPECS, ids=[spec.name for spec in SPECS])
def test_fused_post_conv_kernel_perf(spec: FusedPostConvSpec) -> None:
    run_performance(spec)
