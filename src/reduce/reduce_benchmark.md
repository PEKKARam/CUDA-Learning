# Reduce Benchmark Report

总共包含7种reduce算子的实现方式：

```
"reduce_no_bankconflict",
"reduce_add_during_load",
"reduce_add_during_load_v2",
"reduce_unroll_last_warp",
"reduce_completely_unroll",
"reduce_multi_add",
"reduce_shuffle"
```

```
Benchmarking reduce_no_bankconflict on input size (1048576,)
Avg. time: 30.14 us (100 iterations)

Benchmarking reduce_add_during_load on input size (1048576,)
Avg. time: 23.13 us (100 iterations)

Benchmarking reduce_add_during_load_v2 on input size (1048576,)
Avg. time: 22.82 us (100 iterations)

Benchmarking reduce_unroll_last_warp on input size (1048576,)
Avg. time: 21.78 us (100 iterations)

Benchmarking reduce_completely_unroll on input size (1048576,)
Avg. time: 21.23 us (100 iterations)

Benchmarking reduce_multi_add on input size (1048576,)
Avg. time: 20.09 us (100 iterations)

Benchmarking reduce_shuffle on input size (1048576,)
Avg. time: 19.33 us (100 iterations)
```