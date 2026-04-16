帮忙写一个脚本，读取对应目录下所有的数据，具体背景和要求如下：
要求：
输出文件的名称：
1. 文件结构用例如下
chunk_fwd_o_perf # -> 算子名称 + _perf的结构，比如这里的kernel_name = chunk_fwd_o
- tbe_167317_20260414221524290_ascend_pt # -> 可能有多个，命名不固定
- - ASCEND_PROFILER_OUTPUT # -> 只有一个，命名固定
- - - analyse.done
- - - api_statistic.csv
- - - kernel_details.csv # -> 只有一个，命名固定，关键数据从中获取
- - - op_statistic.csv # -> 只有一个，命名固定，关键数据从中获取
- - - step_trace_time.csv
- - - trace_view.json
- - logs
- - PROF_000001_20260414221524315_00167317EKQOFDDO
- - profiler_info.json
- - profiler_metadata.json
- tbe_167317_20260414221534041_ascend_pt
2. op_statistic.csv：
- OP Type：根据这一列只选取蛇形命名法的算子
- Avg Time(us)：记录对应算子的平均时间
3. kernel_details.csv: 
- Name：首先只选取这一列和op_statistic.csv中根据OP Type选中的算子名称
- Duration(us)：因为OP Type选中的算子名称实际会跑多次再取平均值统计到op_statistic.csv，这里同一kernel选取和op_statistic.csv中Avg Time(us)最接近的一行
- Input Shapes/Input Data Types/Input Formats/Output Shapes/Output Data Types/Output Formats：这些需要统计作为输出的规格
- aicore_time(us)/aic_mac_ratio/aic_scalar_ratio/aic_mte1_ratio/aic_mte2_ratio/aic_mte3_ratio/aic_fixpipe_ratio/aiv_time(us)/aiv_vec_ratio/aiv_scalar_ratio/aiv_mte2_ratio	/aiv_mte3_ratio：这些需要统计作为输出的结果
最终输出样例打屏如下：
------------------------------------------------------------------------------
kernel: chunk_fwd_o_perf # -> 算子名称
Specs:
  - Input Shapes: 
  - Input Data Types: 
  - Input Formats: 
  - Output Shapes: 
  - Output Data Types: 
  - Output Formats: 
Result: 
  - aicore_time(us): 
  - aic_mac_ratio: 
  - aic_scalar_ratio: 
  - aic_mte1_ratio: 
  - aic_mte2_ratio: 
  - aic_mte3_ratio: 
  - aic_fixpipe_ratio: 
  - aiv_time(us): 
  - aiv_vec_ratio: 
  - aiv_scalar_ratio: 
  - aiv_mte2_ratio: 
  - aiv_mte3_ratio: 
------------------------------------------------------------------------------
kernel: chunk_fwd_o_perf
......
  

