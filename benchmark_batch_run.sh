python3 -m pytest -s \
   benchmarks/kernels/benchmark_causal_conv1d_update_kernel.py \
   benchmarks/kernels/benchmark_fused_recurrent_gated_delta_rule_packed_decode_kernel.py \
   benchmarks/kernels/benchmark_fused_sigmoid_gating_delta_rule_update_kernel.py \
   benchmarks/kernels/benchmark_fused_post_conv_kernel.py
