from m5 import M5
import torch
from utils import count_parameters
from torch.profiler import profile, ProfilerActivity
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import cProfile
import numpy as np
from functools import partial


def run_cuda_benchmark(dataloader, tag, run_benchmark_for=200):
    model = M5().cuda()
    print(f"Number of parameters of model: {count_parameters(model):,}")
    batch_times = []
    start = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i, datum in enumerate(tqdm(dataloader, total=run_benchmark_for)):
            before_model = time.perf_counter()
            if i == 0:
                time_to_first_batch = before_model - start
            elif i == run_benchmark_for:
                break
            else:
                batch_time = before_model - after_model
                batch_times.append(batch_time)

            with torch.no_grad():
                model(datum["audio"].cuda())

            after_model = time.perf_counter()

    prof.export_chrome_trace(f"outputs/{tag}_trace.json")
    batch_times = np.array(batch_times) * 1000

    print(f"Time to 1st batch: {time_to_first_batch:.2f} seconds")
    return time_to_first_batch, batch_times


def run_cpu_benchmark(dataloader, run_benchmark_for=100):
    dataloader = iter(dataloader)
    i = 0
    _ = next(dataloader)
    while i < run_benchmark_for:
        i += 1
        batch = next(dataloader)


def plot_batch_times(batch_times, tag):
    color = "gray" if "webdataset" in tag else "orange"
    fig, ax = plt.subplots(figsize=(8, 4), ncols=2)
    fig.suptitle(f"Batch Wait Times for {tag}")
    ax[0].hist(batch_times, bins=20, color=color)
    ax[0].set_ylabel("Frequency")
    ax[0].set_xlabel("Latency (ms)")
    ax[1].plot(batch_times, color=color)
    ax[1].set_ylabel("Latency (ms)")
    ax[1].set_xlabel("Batch index")
    fig.tight_layout()
    fig.savefig(f"outputs/{tag}_plot.jpg")
    plt.close()
    return fig


def run_all_benchmarks(dataloader, single_worker_dataloader, tag):
    time_to_first_batch, batch_times = run_cuda_benchmark(dataloader, tag=tag)
    plot_batch_times(batch_times, tag=tag)
    fn = partial(run_cpu_benchmark, dataloader=single_worker_dataloader)
    cProfile.runctx(
        "fn()", filename=f"outputs/{tag}.prof", locals={"fn": fn}, globals=None
    )
