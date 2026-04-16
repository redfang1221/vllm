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
import torch_npu
import triton

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update, _causal_conv1d_update_kernel
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


def _randn(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(*shape, device="npu", dtype=dtype)


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
                               device="npu",
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
            device="npu",
            dtype=torch.int32,
        ).view(spec.batch, tokens_per_req)
        query_start_loc = torch.arange(
            0,
            total_tokens + 1,
            tokens_per_req,
            device="npu",
            dtype=torch.int32,
        )
        num_accepted_tokens = torch.full((spec.batch,),
                                         tokens_per_req,
                                         device="npu",
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
                                 device="npu",
                                 dtype=torch.int32)
    if padding:
        pad = torch.full((padding,),
                         NULL_BLOCK_ID,
                         device="npu",
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


def get_input_args(data):
    bias: torch.Tensor | None = None
    activation: bool | str | None = None
    conv_state_indices: torch.Tensor | None = None
    num_accepted_tokens: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    max_query_len: int = -1
    null_block_id: int = NULL_BLOCK_ID
    block_idx_last_scheduled_token: torch.Tensor | None = None
    initial_state_idx: torch.Tensor | None = None
    validate_data=False

    x = data["x"]
    conv_state = data["conv_state"]
    weight = data["weight"]
    bias = data["bias"]
    activation = data["activation"]
    conv_state_indices = data["conv_state_indices"]
    if "num_accepted_tokens" in data.keys():
        num_accepted_tokens = data["num_accepted_tokens"]
    if "query_start_loc" in data.keys():
        query_start_loc = data["query_start_loc"]
    if "max_query_len" in data.keys():
        max_query_len = data["max_query_len"]

    causal_conv1d_update(**data)
    if validate_data:
        assert null_block_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    if query_start_loc is None:
        batch, dim, seqlen = x.shape
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert batch == conv_state_indices.shape[0], (
                f"ERROR: conv_state_indices should have shape ({batch},*) but got {conv_state_indices.shape}"
            )

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = x
    stride_w_dim, stride_w_width = weight.stride()

    if query_start_loc is None:
        # X (batch, dim, seqlen)
        stride_x_seq, stride_x_dim, stride_x_token = x.stride()
        stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    else:
        # X (dim, cu_seqlen)
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )
    return {"grid": grid, "input_data": {
        "x_ptr": x, "w_ptr": weight, "bias_ptr": bias, "conv_state_ptr": conv_state, "conv_state_indices_ptr": conv_state_indices, "num_accepted_tokens_ptr": num_accepted_tokens, "query_start_loc_ptr": query_start_loc, "block_idx_last_scheduled_token": block_idx_last_scheduled_token, "initial_state_idx": initial_state_idx, "o_ptr": out, "batch": batch, "dim": dim, "seqlen": seqlen, "state_len": state_len, "num_cache_lines": num_cache_lines, "stride_x_seq": stride_x_seq, "stride_x_dim": stride_x_dim, "stride_x_token": stride_x_token, "stride_w_dim": stride_w_dim, "stride_w_width": stride_w_width, "stride_conv_state_seq": stride_istate_seq, "stride_conv_state_dim": stride_istate_dim, "stride_conv_state_tok": stride_istate_token, "stride_state_indices": stride_state_indices, "stride_o_seq": stride_o_seq, "stride_o_dim": stride_o_dim, "stride_o_token": stride_o_token, "null_block_id": null_block_id, "HAS_BIAS": bias is not None, "KERNEL_WIDTH": width, "SILU_ACTIVATION": activation in ["silu" "swish"], "IS_VARLEN": query_start_loc is not None, "IS_APC_ENABLED": block_idx_last_scheduled_token is not None, "IS_SPEC_DECODING": num_accepted_tokens is not None, "NP2_STATELEN": np2_statelen, "HAS_NULL_BLOCK": null_block_id is not None, "BLOCK_N": 256
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
    _causal_conv1d_update_kernel[grid](**input_data)


def run_performance(spec: CausalConv1dUpdateSpec) -> float:
    data = build_data(spec)
    args = get_input_args(data)
    return perf_test(fn_triton, args, "causal_conv1d_update_kernel_perf")


@pytest.mark.skipif(not torch.npu.is_available(), reason="Need NPU device")
@pytest.mark.parametrize("spec", SPECS, ids=[spec.name for spec in SPECS])
def test_causal_conv1d_update_kernel_perf(
    spec: CausalConv1dUpdateSpec,
) -> None:
    run_performance(spec)