# LES of a round turbulent pipe flow at Re_tau=550.

## Performance Results for `--cimode=1`

### NVIDIA A100
```
  solve                 1.43790e+01s  23.4      
    min                 6.06289e-02s
    max                 1.80862e-01s
    flops/rank          1.54316e+12
    udfExecuteStep      7.99648e-04s   0.0   0.0  200
    makef               4.16758e+00s   6.8  29.0  200
      udfUEqnSource     5.42614e-03s   0.0   0.1  200
    udfProperties       7.55936e-04s   0.0   0.0  201
    velocitySolve       2.55075e+00s   4.2  17.7  200
      rhs               3.00298e-01s   0.5  11.8  200
      preconditioner    1.56645e-01s   0.3   6.1  859
    pressureSolve       6.97296e+00s  11.4  48.5  200
      rhs               5.76751e-01s   0.9   8.3  200
      preconditioner    5.09020e+00s   8.3  73.0  760
        pMG smoother    3.02419e+00s   4.9  59.4  1520
        pMG smoother    6.25036e-01s   1.0  12.3  1520
        coarse grid     9.87971e-01s   1.6  19.4  760
      initial guess     3.50167e-01s   0.6   5.0  200
```

### AMD MI250X/1
```
  elapsedStepSum        2.35982e+01s  0.11
  solve                 2.35922e+01s  0.11
    min                 5.87667e-02s
    max                 9.92639e-02s
    flop/s              1.00710e+12

    makef               6.85422e+00s  0.29  200
      udfUEqnSource     6.26127e-03s  0.00  200
    udfProperties       9.04159e-04s  0.00  201
    velocitySolve       5.38714e+00s  0.23  200
      rhs               3.83831e-01s  0.07  200
    pressureSolve       1.05776e+01s  0.45  200
      rhs               8.74453e-01s  0.08  200
      preconditioner    7.89478e+00s  0.75  776
        pMG smoother    6.02886e+00s  0.76  3104
        coarse grid     1.08808e+00s  0.14  776
      initial guess     5.38374e-01s  0.05  200
```

### AMD EPYC 7742
```
  elapsedStepSum        1.41905e+02s  0.74
  solve                 1.41852e+02s  0.74
    min                 3.10468e-01s
    max                 5.58652e-01s
    flop/s              1.67737e+11

    makef               5.93037e+01s  0.42  200
      udfUEqnSource     2.25468e-02s  0.00  200
    udfProperties       1.28648e-03s  0.00  201
    velocitySolve       3.66557e+01s  0.26  200
      rhs               4.64122e+00s  0.13  200
    pressureSolve       4.11335e+01s  0.29  200
      rhs               5.57414e+00s  0.14  200
      preconditioner    2.60316e+01s  0.63  779
        pMG smoother    2.17306e+01s  0.83  3116
        coarse grid     6.11829e-01s  0.02  779
      initial guess     3.06133e+00s  0.07  200
```

## Reference Data 
[1] ftp://ftp.mech.kth.se/pub/pschlatt/DATA/PIPE/ 
