# webdataset vs Deep Lake

This repository contains the code for comparing the performance of `webdataset` [GitHub repo](https://github.com/webdataset/webdataset) and `deeplake` [GitHub repo](https://github.com/activeloopai/deeplake) libraries.

The code is structured into 5 Jupyter notebooks for 
1. Restructuring and Exploring the Common Voice dataset used for the benchmarks
2. Preparing webdataset tarballs
3. Preparing deeplake dataset
4. Benchmarking webdataset
5. Benchmarking deeplake

Additionally `utils.py` contains useful helper functions for running multiprocessor & multithread functions, as well as conventient timing methods.
`m5.py` contains the neural network model used for the benchmarks.

The `outputs/` directory contains the plots, the `CProfile` reports (used for `snakeviz` visualization), as well as the PyTorch Profiler trace outputs.

For running the `deeplake` benchmarks, you will need to install a Google Service account key as a json and update the line that sets the appropriate environment variable in the notebook.