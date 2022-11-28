import time

import torch


class SpillProfiler:
    def __init__(self, graph, warm_up_iters=0, profile_iters=1):
        self.graph = graph
        self.warm_up_iters = warm_up_iters
        self.profile_iters = profile_iters
        self.bandwidth = 5e9
        self.kernel_launch_time = 1e-5

    def benchmark_tensor(self, tensor_size):
        if not torch.cuda.is_available():
            runtime = self.kernel_launch_time + tensor_size / self.bandwidth
            return runtime

        t_cpu = torch.empty(tensor_size).pin_memory()
        t_gpu = torch.empty(tensor_size, device="cuda")
        for _ in range(self.warm_up_iters):
            t_cpu.copy_(t_gpu, non_blocking=True)
            t_gpu.copy_(t_cpu, non_blocking=True)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(self.profile_iters):
            t_cpu.copy_(t_gpu, non_blocking=True)
            t_gpu.copy_(t_cpu, non_blocking=True)
        torch.cuda.synchronize()
        stop = time.time()
        # We assume that the bandwidth in and out of the gpu is roughly the same
        runtime = (stop - start) / self.profile_iters / 2
        return runtime

    def benchmark_all(self):
        table = {}
        for e in self.graph.edges.values():
            runtime = self.benchmark_tensor(e.size)
            table[e] = runtime
        return table
