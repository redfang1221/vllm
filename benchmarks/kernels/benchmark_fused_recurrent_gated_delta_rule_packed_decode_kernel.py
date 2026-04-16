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
import torch_npu
import triton

from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_recurrent_gated_delta_rule_packed_decode_kernel
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
    device = torch.device("npu")
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


def get_input_args(data):
    inplace_final_state: bool = True
    cu_seqlens: torch.Tensor | None = None
    ssm_state_indices: torch.Tensor | None = None
    num_accepted_tokens: torch.Tensor | None = None
    use_qk_l2norm_in_kernel: bool = False

    mixed_qkv = data["mixed_qkv"]
    a = data["a"]
    b = data["b"]
    A_log = data["A_log"]
    dt_bias = data["dt_bias"]
    scale = data["scale"]
    initial_state = data["initial_state"]
    out = data["out"]
    ssm_state_indices = data["ssm_state_indices"]
    use_qk_l2norm_in_kernel = data["use_qk_l2norm_in_kernel"]
    if mixed_qkv.ndim != 2:
        raise ValueError(
            f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim})."
        )
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"`a` and `b` must be 2D tensors (got a.ndim={a.ndim}, b.ndim={b.ndim})."
        )
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("`a`/`b` must be contiguous in the last dim.")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log`/`dt_bias` must be 1D tensors.")
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        raise ValueError("`A_log`/`dt_bias` must be contiguous.")
    if ssm_state_indices.ndim != 1:
        raise ValueError(
            f"`ssm_state_indices` must be 1D for packed decode (got ndim={ssm_state_indices.ndim})."
        )
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")

    dev = mixed_qkv.device
    if (
        a.device != dev
        or b.device != dev
        or A_log.device != dev
        or dt_bias.device != dev
        or initial_state.device != dev
        or out.device != dev
        or ssm_state_indices.device != dev
    ):
        raise ValueError("All inputs must be on the same device.")
    B = mixed_qkv.shape[0]
    if a.shape[0] != B or b.shape[0] != B:
        raise ValueError(
            "Mismatched batch sizes: "
            f"mixed_qkv.shape[0]={B}, a.shape[0]={a.shape[0]}, b.shape[0]={b.shape[0]}."
        )
    if ssm_state_indices.shape[0] != B:
        raise ValueError(
            f"`ssm_state_indices` must have shape [B] (got {tuple(ssm_state_indices.shape)}; expected ({B},))."
        )
    if initial_state.ndim != 4:
        raise ValueError(
            f"`initial_state` must be a 4D tensor (got ndim={initial_state.ndim})."
        )
    if initial_state.stride(-1) != 1:
        raise ValueError("`initial_state` must be contiguous in the last dim.")
    HV, V, K = initial_state.shape[-3:]
    if a.shape[1] != HV or b.shape[1] != HV:
        raise ValueError(
            f"`a`/`b` must have shape [B, HV] with HV={HV} (got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)})."
        )
    if A_log.numel() != HV or dt_bias.numel() != HV:
        raise ValueError(
            f"`A_log` and `dt_bias` must have {HV} elements (got A_log.numel()={A_log.numel()}, dt_bias.numel()={dt_bias.numel()})."
        )
    if out.shape != (B, 1, HV, V):
        raise ValueError(
            f"`out` must have shape {(B, 1, HV, V)} (got out.shape={tuple(out.shape)})."
        )
    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(
            f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}."
        )
    q_dim = qk_dim // 2
    if q_dim % K != 0:
        raise ValueError(f"Invalid packed Q size {q_dim}: must be divisible by K={K}.")
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(
            f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}."
        )
    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError(
            f"Packed decode kernel only supports NK=1 (got K={K}, BK={BK})."
        )
    BV = min(triton.next_power_of_2(V), 32)
    num_stages = 3
    num_warps = 1
    stride_mixed_qkv_tok = mixed_qkv.stride(0)
    stride_a_tok = a.stride(0)
    stride_b_tok = b.stride(0)
    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = initial_state.stride(0)
    stride_indices_seq = ssm_state_indices.stride(0)
    NV = triton.cdiv(V, BV)
    grid = (NV, B * HV)
    return {"grid": grid, "input_data": {
        "mixed_qkv": mixed_qkv, "a": a, "b": b, "A_log": A_log, "dt_bias": dt_bias, "o": out, "h0": initial_state, "ht": initial_state, "ssm_state_indices": ssm_state_indices, "scale": scale, "stride_mixed_qkv_tok": stride_mixed_qkv_tok, "stride_a_tok": stride_a_tok, "stride_b_tok": stride_b_tok, "stride_init_state_token": stride_init_state_token, "stride_final_state_token": stride_final_state_token, "stride_indices_seq": stride_indices_seq, "H": H, "HV": HV, "K": K, "V": V, "BK": BK, "BV": BV, "SOFTPLUS_THRESHOLD": 20.0, "USE_QK_L2NORM_IN_KERNEL": use_qk_l2norm_in_kernel, "num_warps": num_warps, "num_stages": num_stages
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
    fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](**input_data)


def run_performance(spec: CausalConv1dUpdateSpec) -> float:
    data = build_data(spec)
    args = get_input_args(data)
    return perf_test(fn_triton, args, "fused_recurrent_gated_delta_rule_packed_decode_kernel_perf")


@pytest.mark.skipif(not torch.npu.is_available(), reason="Need NPU device")
@pytest.mark.parametrize("spec", SPECS, ids=[spec.name for spec in SPECS])
def test_fused_recurrent_gated_delta_rule_packed_decode_kernel_perf(
    spec: PackedDecodeSpec,
) -> None:
    ms = run_performance(spec)
