# Kershaw BP5 and BPS5

## Performance Results (E/GPU=8000) 

### NVIDIA V100
```
BPS5
solve time: 0.521428s
  preconditioner 0.402965s
    smoother 0.251669s
    coarse grid 0.103442s
iterations: 31
throughput: 2.17e+08 (DOF x iter)/s/rank
throughput: 7.00436e+06 DOF/s
flops: 4.90859e+11

BP5
throughput: 2.47e+09 (DOF x iter)/s/rank
flops: 4.69537e+11
```

### NVIDIA A100
```
BPS5
solve time: 0.196772s
  preconditioner 0.147807s
    smoother 0.0987781s
    coarse grid 0.0321823s
iterations: 28
throughput: 3.90e+08 (DOF x iter)/s/rank
throughput: 1.39451e+07 DOF/s/rank
flops: 8.43646e+11 

BP5
throughput: 3.87e+09 (DOF x iter)/s/rank
flops/rank: 7.34419e+11 
```

### AMD MI250X/1
```
BPS5
solve time: 0.440587s
  preconditioner 0.335607s
    smoother 0.242305s
    coarse grid 0.0527129s
iterations: 31
throughput: 2.57e+08 (DOF x iter)/s/rank
throughput: 8.28954e+06 DOF/s
flops: 5.80923e+11

BP5
throughput: 3.12e+09 (DOF x iter)/s/rank
flops: 5.92321e+11
```

### Perlmutter 128 nodes
```
BPS5
solve time: 0.965085s
  preconditioner 0.842747s
    smoother 0.447103s
    coarse grid 0.321199s
iterations: 59
throughput: 1.68e+08 (DOF x iter)/s/rank
throughput: 1.45576e+09 DOF/s
flops: 1.86303e+14 

BP5
throughput: 3.37e+09 (DOF x iter)/s/rank
flops: 3.27151e+14 
```

### Crusher 64 nodes 
```
BPS5
solve time: 1.17859s
  preconditioner 1.00887s
    smoother 0.533104s
    coarse grid 0.398124s
iterations: 59
throughput: 1.37e+08 (DOF x iter)/s/rank
throughput: 1.19204e+09 DOF/s
flops: 1.52554e+14

BP5
throughput: 2.27e+09 (DOF x iter)/s/rank
flops: 2.20252e+14 
```
