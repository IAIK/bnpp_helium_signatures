# Helium+LowMC Signatures 

This repository contains the code for the Helium signature scheme from our paper

The implementation is based on the publicly available Rainier code (https://github.com/IAIK/rainier-signatures) and the publicly available Picnic implementation (see https://microsoft.github.io/Picnic/).

## Requirements

C++-17 Compatible Toolchain

For testing:

* [GMP](https://gmplib.org/)
* [NTL](https://shoup.net/ntl)

## Setup

```bash
mkdir build
cd build
cmake ..
make 
# tests (only if you built them)
make test
# benchmarks
python3 ../tools/bench_all.py # benchmarks some of the selected parameters
./bench_free -i <iterations> <kappa> <Sboxes> <N> <tau> #benchmark parameters freely
```

The benchmark script contains a `SCALING_FACTOR` variable that is used to scale the measured cycles to ms. Configure it according to your specific machine.
