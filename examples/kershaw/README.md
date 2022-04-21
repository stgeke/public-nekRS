# Kershaw BP5 and BPS5 problem

## Performance Results

### NVIDIA V100
```
BPS5
..................................................
repetitions: 50
solve time: min: 0.521428s  avg: 0.522716s  max: 0.575354s
  preconditioner 0.402965s
    smoother 0.251669s
    coarse grid 0.103442s
iterations: 31
throughput: 2.17135e+08 (DOF x iter)/s
throughput: 7.00436e+06 DOF/s
FLOPS/s: 4.90859e+11

BP5
done (6.094e-06s)
.........................
repetitions: 25
solve time: min: 1.47599s  avg: 1.47628s  max: 1.47693s
iterations: 1000
throughput: 2.47446e+09 (DOF x iter)/s
FLOPS/s: 4.69537e+11
```

### NVIDIA A100
```
BPS5
..................................................
repetitions: 50
solve time: min: 0.296066s  avg: 0.29751s  max: 0.351041s
  preconditioner 0.224607s
    smoother 0.154094s
    coarse grid 0.0426403s
iterations: 31
throughput: 3.82415e+08 (DOF x iter)/s
throughput: 1.2336e+07 DOF/s
FLOPS/s: 8.64493e+11

BP5
done (2.996e-06s)
.........................
repetitions: 25
solve time: min: 0.895191s  avg: 0.895511s  max: 0.895851s
iterations: 1000
throughput: 4.07987e+09 (DOF x iter)/s
FLOPS/s: 7.7417e+11
```

### AMD MI100
```
BPS5
..................................................
repetitions: 50
solve time: min: 0.508355s  avg: 0.510528s  max: 0.554938s
  preconditioner 0.389843s
    smoother 0.28957s
    coarse grid 0.051806s
iterations: 31
throughput: 2.22719e+08 (DOF x iter)/s
throughput: 7.18447e+06 DOF/s
FLOPS/s: 5.03481e+11

BP5
done (4.258e-06s)
.........................
repetitions: 25
solve time: min: 1.41609s  avg: 1.41781s  max: 1.42029s
iterations: 1000
throughput: 2.57912e+09 (DOF x iter)/s
FLOPS/s: 4.89396e+11
```

### AMD MI250X/1
```
BPS5
..................................................
repetitions: 50
solve time: min: 0.440587s  avg: 0.442264s  max: 0.470028s
  preconditioner 0.335607s
    smoother 0.242305s
    coarse grid 0.0527129s
iterations: 31
throughput: 2.56976e+08 (DOF x iter)/s
throughput: 8.28954e+06 DOF/s
FLOPS/s: 5.80923e+11

BP5
done (2.745e-06s)
.........................
repetitions: 25
solve time: min: 1.17002s  avg: 1.17044s  max: 1.17101s
iterations: 1000
throughput: 3.12153e+09 (DOF x iter)/s
FLOPS/s: 5.92321e+11
```