cuda-sandbox
============
### Tested Devices
- NVIDIA GeForce GTX 560 Ti
- NVIDIA GeForce RTX 3060 Laptop GPU

### Build and run
```sh
$ build --reduce
$ ./build/reduce.exe 10000 4
$ GPUDevice 0:  NVIDIA GeForce RTX 3060 Laptop GPU
$ Compute cap:  8.6
$ Problem size: 163840000
$ CTAs number:  64
$ Computation time:         2.045024 [ms]
$ Peak bandwidth:         336.048 [GB/s]
$ Effective bandwidth:    320.466 [GB/s]  95.363 % of the peak!
$ Perfectly correct!
$ GPU sum reduction: 163840000
$
$ build --scan
$ ./build/scan.exe 10000 4
$ GPUDevice 0:  NVIDIA GeForce RTX 3060 Laptop GPU
$ Compute cap:  8.6
$ Problem size: 163840000
$ CTAs number:  64
$ Computation time:         6.393344 [ms]
$ Peak bandwidth:         336.048 [GB/s]
$ Effective bandwidth:    307.520 [GB/s]  91.511 % of peak!
$ Perfectly correct!
$
$ build --datamovement
$ ./build/datamovement.exe
$
$ GPUDevice 0:  NVIDIA GeForce RTX 3060 Laptop GPU
$ Compute cap:  8.6
$ Problem size: 6553600
$ =========================================================================================================================================
$ Skeletons for variable-sized grids
$ =========================================================================================================================================
$ IN-OUT-1    128 [CTASIZE]    51200 [GRID]     4 [Bytes/element]    0.227840 [ms]    336.048 [GB/s]    230.112 [GB/s]     68.476 % of peak
$ IN-OUT-1    128 [CTASIZE]    25600 [GRID]     8 [Bytes/element]    0.172032 [ms]    336.048 [GB/s]    304.762 [GB/s]     90.690 % of peak
$ IN-OUT-1    128 [CTASIZE]    12800 [GRID]    16 [Bytes/element]    0.177152 [ms]    336.048 [GB/s]    295.954 [GB/s]     88.069 % of peak
$ IN-OUT-1    256 [CTASIZE]    25600 [GRID]     4 [Bytes/element]    0.168800 [ms]    336.048 [GB/s]    310.597 [GB/s]     92.426 % of peak
$ IN-OUT-1    256 [CTASIZE]    12800 [GRID]     8 [Bytes/element]    0.172032 [ms]    336.048 [GB/s]    304.762 [GB/s]     90.690 % of peak
$ IN-OUT-1    256 [CTASIZE]     6400 [GRID]    16 [Bytes/element]    0.171008 [ms]    336.048 [GB/s]    306.587 [GB/s]     91.233 % of peak
$ IN-OUT-1    512 [CTASIZE]    12800 [GRID]     4 [Bytes/element]    0.185344 [ms]    336.048 [GB/s]    282.873 [GB/s]     84.176 % of peak
$ IN-OUT-1    512 [CTASIZE]     6400 [GRID]     8 [Bytes/element]    0.171840 [ms]    336.048 [GB/s]    305.102 [GB/s]     90.791 % of peak
$ IN-OUT-1    512 [CTASIZE]     3200 [GRID]    16 [Bytes/element]    0.172032 [ms]    336.048 [GB/s]    304.762 [GB/s]     90.690 % of peak
$ IN-OUT-1   1024 [CTASIZE]     6400 [GRID]     4 [Bytes/element]    0.186368 [ms]    336.048 [GB/s]    281.319 [GB/s]     83.714 % of peak
$ IN-OUT-1   1024 [CTASIZE]     3200 [GRID]     8 [Bytes/element]    0.168960 [ms]    336.048 [GB/s]    310.303 [GB/s]     92.339 % of peak
$ IN-OUT-1   1024 [CTASIZE]     1600 [GRID]    16 [Bytes/element]    0.173792 [ms]    336.048 [GB/s]    301.676 [GB/s]     89.772 % of peak
$ =========================================================================================================================================
$ IN-OUT-2    128 [CTASIZE]    25600 [GRID]     4 [Bytes/element]    0.175072 [ms]    336.048 [GB/s]    299.470 [GB/s]     89.115 % of peak
$ IN-OUT-2    128 [CTASIZE]    12800 [GRID]     8 [Bytes/element]    0.175936 [ms]    336.048 [GB/s]    297.999 [GB/s]     88.678 % of peak
$ IN-OUT-2    128 [CTASIZE]     6400 [GRID]    16 [Bytes/element]    0.177152 [ms]    336.048 [GB/s]    295.954 [GB/s]     88.069 % of peak
$ IN-OUT-2    256 [CTASIZE]    12800 [GRID]     4 [Bytes/element]    0.174080 [ms]    336.048 [GB/s]    301.176 [GB/s]     89.623 % of peak
$ IN-OUT-2    256 [CTASIZE]     6400 [GRID]     8 [Bytes/element]    0.174080 [ms]    336.048 [GB/s]    301.176 [GB/s]     89.623 % of peak
$ IN-OUT-2    256 [CTASIZE]     3200 [GRID]    16 [Bytes/element]    0.174080 [ms]    336.048 [GB/s]    301.176 [GB/s]     89.623 % of peak
$ IN-OUT-2    512 [CTASIZE]     6400 [GRID]     4 [Bytes/element]    0.175104 [ms]    336.048 [GB/s]    299.415 [GB/s]     89.099 % of peak
$ IN-OUT-2    512 [CTASIZE]     3200 [GRID]     8 [Bytes/element]    0.176128 [ms]    336.048 [GB/s]    297.674 [GB/s]     88.581 % of peak
$ IN-OUT-2    512 [CTASIZE]     1600 [GRID]    16 [Bytes/element]    0.177152 [ms]    336.048 [GB/s]    295.954 [GB/s]     88.069 % of peak
$ IN-OUT-2   1024 [CTASIZE]     3200 [GRID]     4 [Bytes/element]    0.173056 [ms]    336.048 [GB/s]    302.959 [GB/s]     90.153 % of peak
$ IN-OUT-2   1024 [CTASIZE]     1600 [GRID]     8 [Bytes/element]    0.175104 [ms]    336.048 [GB/s]    299.415 [GB/s]     89.099 % of peak
$ IN-OUT-2   1024 [CTASIZE]      800 [GRID]    16 [Bytes/element]    0.173056 [ms]    336.048 [GB/s]    302.959 [GB/s]     90.153 % of peak
$ =========================================================================================================================================
$ IN-OUT-4    128 [CTASIZE]    12800 [GRID]     4 [Bytes/element]    0.173056 [ms]    336.048 [GB/s]    302.959 [GB/s]     90.153 % of peak
$ IN-OUT-4    128 [CTASIZE]     6400 [GRID]     8 [Bytes/element]    0.175104 [ms]    336.048 [GB/s]    299.415 [GB/s]     89.099 % of peak
$ IN-OUT-4    128 [CTASIZE]     3200 [GRID]    16 [Bytes/element]    0.172768 [ms]    336.048 [GB/s]    303.464 [GB/s]     90.304 % of peak
$ IN-OUT-4    256 [CTASIZE]     6400 [GRID]     4 [Bytes/element]    0.172960 [ms]    336.048 [GB/s]    303.127 [GB/s]     90.203 % of peak
$ IN-OUT-4    256 [CTASIZE]     3200 [GRID]     8 [Bytes/element]    0.171008 [ms]    336.048 [GB/s]    306.587 [GB/s]     91.233 % of peak
$ IN-OUT-4    256 [CTASIZE]     1600 [GRID]    16 [Bytes/element]    0.172032 [ms]    336.048 [GB/s]    304.762 [GB/s]     90.690 % of peak
$ IN-OUT-4    512 [CTASIZE]     3200 [GRID]     4 [Bytes/element]    0.174080 [ms]    336.048 [GB/s]    301.176 [GB/s]     89.623 % of peak
$ IN-OUT-4    512 [CTASIZE]     1600 [GRID]     8 [Bytes/element]    0.172032 [ms]    336.048 [GB/s]    304.762 [GB/s]     90.690 % of peak
$ IN-OUT-4    512 [CTASIZE]      800 [GRID]    16 [Bytes/element]    0.171008 [ms]    336.048 [GB/s]    306.587 [GB/s]     91.233 % of peak
$ IN-OUT-4   1024 [CTASIZE]     1600 [GRID]     4 [Bytes/element]    0.174080 [ms]    336.048 [GB/s]    301.176 [GB/s]     89.623 % of peak
$ IN-OUT-4   1024 [CTASIZE]      800 [GRID]     8 [Bytes/element]    0.175968 [ms]    336.048 [GB/s]    297.945 [GB/s]     88.661 % of peak
$ IN-OUT-4   1024 [CTASIZE]      400 [GRID]    16 [Bytes/element]    0.179200 [ms]    336.048 [GB/s]    292.571 [GB/s]     87.062 % of peak
$ =========================================================================================================================================
$ MEMCPY        0 [CTASIZE]        0 [GRID]     4 [Bytes/element]    0.184320 [ms]    336.048 [GB/s]    284.444 [GB/s]     84.644 % of peak
$ =========================================================================================================================================
$ Skeletons for fixed-sized grids
$ =========================================================================================================================================
$ =========================================================================================================================================
```
