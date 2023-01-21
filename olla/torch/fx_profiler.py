
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
        self._reset()

    def _reset(self):
        self.warm_up = False
        self.total_runtime_sec: List[float] = []
        self.node_profiles = OrderedDict()
        self.table = None
        self.max_mem_fragmentation = None
        self.peak_reserved_bytes = None

    def run(self, *args) -> Any:
        # reset profile results and status variables
        self._reset()

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

    def _generate_summary(
        self, sort_by: Optional[Union[str, List[str]]] = None
    ) -> None:
        self.table = pd.DataFrame()
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
            self.table = self.table.append(row, ignore_index=True)

        # ensure that the first 3 columns are: Op name, Op type, runtime
        self.table.insert(0, "Op name", self.table.pop("Op name"))
        self.table.insert(1, "Op type", self.table.pop("Op type"))
        if self.profile_time:
            self.table.insert(2, "runtimes_sec", self.table.pop("runtimes_sec"))

        if sort_by:
            self.table.sort_values(sort_by)

    def summary(self, sort_by: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        if self.table is None or sort_by:
            self._generate_summary(sort_by)
        return self.table

    def _calculate_maximum_memory_fragmentation(self) -> None:
        assert (
            self.profile_memory
        ), "Can only calculate memory fragmentation if profiling is enabled"

        if self.table is None:
            self._generate_summary()

        idx = self.table["reserved_bytes.all.current"].idxmax()
        row = self.table.iloc[idx]
        self.allocated_mem_at_peak = row["allocated_bytes.all.current"]
        self.peak_reserved_bytes = row["reserved_bytes.all.current"]
        self.max_mem_fragmentation = (
            self.peak_reserved_bytes - self.allocated_mem_at_peak
        ) / self.peak_reserved_bytes

    def get_peak_reserved_bytes(self) -> int:
        if self.peak_reserved_bytes is None:
            self._calculate_maximum_memory_fragmentation()

        return self.peak_reserved_bytes

    def get_max_mem_fragmentation(self) -> float:
        if self.max_mem_fragmentation is None:
            self._calculate_maximum_memory_fragmentation()

        return self.max_mem_fragmentation

    def get_allocated_mem_at_peak(self) -> float:
        if self.allocated_mem_at_peak is None:
            self._calculate_maximum_memory_fragmentation()

        return self.allocated_mem_at_peak
