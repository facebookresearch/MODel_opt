
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import statistics
import time
from typing import Any, List, Optional, OrderedDict, Union

import numpy as np

import pandas as pd

import torch


class ProfilingInterpreter(torch.fx.Interpreter):
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        profile_memory: bool = True,
        profile_time: bool = False,
        warm_up_iters: int = 0,
        profile_iters: int = 1,
    ):
        super().__init__(gm)
        self.profile_memory = profile_memory
        self.profile_time = profile_time
        self.warm_up_iters = warm_up_iters
        self.profile_iters = profile_iters
        if self.profile_memory:
            assert (
                torch.cuda.is_available()
            ), "Currently memory profile is only supported on CUDA"
        self.total_runtime_sec: List[float] = []
        self.node_profiles = OrderedDict()
        self.warm_up = False

    def run(self, *args) -> Any:
        for _ in range(self.warm_up_iters):
            # running without profiling
            # print(f"warm up execution iteration {i}")
            self.warm_up = True
            super().run(*args)
            self.warm_up = False

        for _ in range(self.profile_iters):
            # Measure total runtime to run the model
            t_start = time.time()
            # print(f"profile execution iteration {i}")
            return_val = super().run(*args)
            t_end = time.time()
            self.total_runtime_sec.append(t_end - t_start)
        return return_val

    def run_node(self, n: torch.fx.Node) -> Any:
        return_val = None
        try:
            if self.profile_time and not self.warm_up:
                # TODO: use a more accurate method to profile time, e.g., timeit
                t_start = time.time()
                return_val = super().run_node(n)
                t_end = time.time()
                self.node_profiles.setdefault(n, OrderedDict())
                self.node_profiles[n].setdefault("runtimes_sec", [])
                self.node_profiles[n]["runtimes_sec"].append(t_end - t_start)
            else:
                return_val = super().run_node(n)

            if self.profile_memory and not self.warm_up:
                memory_stats = torch.cuda.memory_stats()
                assert (
                    memory_stats
                ), "torch.cuda.memory_stats() returned an empty result. This could be because the model or tensors were not transferred to GPU. Make sure to call `model.cuda()` and `input = input.cuda()` on all the model's inputs."
                self.node_profiles.setdefault(n, OrderedDict())
                for k in memory_stats.keys():
                    self.node_profiles[n].setdefault(k, [])
                    self.node_profiles[n][k].append(memory_stats[k])
        except RuntimeError:
            pass  # warnings.warn(f"Unable to profile node {n.name}", RuntimeWarning)

        return return_val

    # TODO: decouple calculating averages from printing the result, and hence always calculate averages at the end of profiling
    def summary(self, sort_by: Optional[Union[str, List[str]]] = None) -> str:
        table = pd.DataFrame()
        # Can be used later if we want percentage time of each node with respect to whole model
        mean_total_runtime = statistics.mean(self.total_runtime_sec)  # noqa

        # For each node, record max_required_memory and allocated_mem_at_max, and mean statistics for other measurements
        for node, profiles in self.node_profiles.items():
            row = OrderedDict()

            row["Op name"] = node.name
            row["Op type"] = node.op
            for name, values in profiles.items():
                if name == "allocated_bytes.all.current":
                    continue
                elif name == "reserved_bytes.all.current":
                    imax = np.argmax(np.asarray(values))
                    max_required_memory = values[imax]
                    allocated_mem_at_max = profiles["allocated_bytes.all.current"][imax]
                    row["reserved_bytes.all.current"] = max_required_memory
                    row["allocated_bytes.all.current"] = allocated_mem_at_max
                else:
                    row[name] = statistics.mean(values)
            table = pd.concat([table, pd.Series(row).to_frame().T], ignore_index=True)

        # ensure that the first 3 columns are: Op name, Op type, runtime
        table.insert(0, "Op name", table.pop("Op name"))
        table.insert(1, "Op type", table.pop("Op type"))
        if self.profile_time:
            table.insert(2, "runtimes_sec", table.pop("runtimes_sec"))

        if sort_by:
            table.sort_values(sort_by)

        return pd.DataFrame(table)
