import itertools
import math
import sys
from collections import defaultdict, OrderedDict

import intervaltree
from olla import dataflow_graph, ilp_solver, utils


class Scheduler:
    def __init__(
        self,
        graph,
        timeout_s=None,
        rel_stop=None,
        solver="GUROBI",
        timestep_factor=1.0,
        print_relaxation=False,
    ):
        self.graph = graph
        self.timeout = timeout_s
        self.rel_stop = rel_stop
        self.solver = solver
        self.timestep_factor = timestep_factor
        self.print_relaxation = print_relaxation

        # Skip the weights when computing the number of timesteps needed: since we want to
        # spill the weight while computing, we don't need to reserve timesteps for these
        # spills
        self.num_nodes = 0
        for n in graph.nodes.values():
            if not n.is_stateful():
                self.num_nodes += 1

        self.longest_path_length = graph.longest_path_length()

    # Compute the earliest time a node can be scheduled. It's in the range
    # [1, m], where m is the number of nodes in the graph.
    def ComputeASAPSchedule(self, schedule_constraints):
        timings = {}
        for vertex in self.graph.nodes.values():
            self._ComputeASAPSchedule(vertex, timings, schedule_constraints)
        return timings

    def _ComputeASAPSchedule(self, vertex, timings, schedule_constraints):
        if vertex in timings:
            # Already visited
            return timings[vertex]
        time = 1
        for e in vertex.fanin:
            source = e.source
            t = self._ComputeASAPSchedule(source, timings, schedule_constraints)
            time = max(t + 1, time)

        if vertex in schedule_constraints:
            if time > schedule_constraints[vertex]:
                raise ValueError(
                    "Infeasible user schedule specified for node %s: requested %d, min_feasible %d"
                    % (str(vertex), schedule_constraints[vertex], time)
                )
            time = schedule_constraints[vertex]

        timings[vertex] = time

        return time

    # Compute the latest time a node can be scheduled. It's in the range
    # [1, m], where m is the number of nodes in the graph.
    def ComputeALAPSchedule(self, schedule_constraints, max_timesteps):
        timings = {}
        for vertex in self.graph.nodes.values():
            self._ComputeALAPSchedule(
                vertex, timings, schedule_constraints, max_timesteps
            )
        return timings

    def _ComputeALAPSchedule(
        self, vertex, timings, schedule_constraints, max_timesteps
    ):
        if vertex in timings:
            # Already visited
            return timings[vertex]
        time = max_timesteps
        for e in vertex.fanout:
            for sink in e.sinks:
                t = self._ComputeALAPSchedule(
                    sink, timings, schedule_constraints, max_timesteps
                )
                time = min(t - 1, time)

        if vertex in schedule_constraints:
            if time < schedule_constraints[vertex]:
                raise ValueError(
                    "Infeasible user schedule specified for node %s" % str(vertex)
                )
            time = schedule_constraints[vertex]

        timings[vertex] = time
        return time

    def ComputeMakespans(self, asap, alap):
        makespan = {}
        for n in self.graph.nodes.values():
            lb = asap[n]
            for e in n.fanout:
                ub = alap[n]
                for snk in e.sinks:
                    assert alap[snk] > alap[n]
                    ub = max(ub, alap[snk])
                makespan[e] = (lb, ub)
                # print(e.name + "[" + str(lb) + ", " + str(up) + "]")
        return makespan

    def _GCD(self, values):
        if len(values) == 0:
            return 1
        if len(values) == 1:
            return values[0]
        elif len(values) == 2:
            return math.gcd(values[0], values[1])
        else:
            middle = len(values) // 2
            return math.gcd(self._GCD(values[:middle]), self._GCD(values[middle:]))

    class TimeStepsForNode:
        def __init__(self, node, spans, startoffset=0):
            asap = spans[1]
            alap = spans[2]
            lb = asap[node]
            ub = alap[node]
            assert not node.is_stateful()
            self.iter = iter(range(lb + startoffset, ub + 1))

        def __iter__(self):
            return self

        def __next__(self):
            return self.iter.__next__()

    class TimeStepsForEdge:
        def __init__(self, edge, spans, startoffset=0):
            lb, ub = spans[0][edge]

            intervals = intervaltree.IntervalTree()
            for snk in edge.sinks:
                if snk.op_type == "stateful_node_sink":
                    continue
                new_lb = spans[1][snk]
                new_lb = max(1, new_lb - 1)
                new_ub = spans[2][snk]
                intervals.add(intervaltree.Interval(new_lb, new_ub + 1))

            if not edge.is_stateful():
                src = edge.source
                new_lb = spans[1][src]
                new_ub = spans[2][src]
                intervals.add(intervaltree.Interval(new_lb, new_ub + 1))
            else:
                intervals.add(intervaltree.Interval(lb, lb + 1))
                intervals.add(intervaltree.Interval(ub, ub + 1))

            ts = []
            intervals.merge_overlaps()
            for i in intervals:
                ts += list(range(i.begin, i.end))

            ts = sorted(ts)
            self.iter = iter(ts[startoffset:])

        def __iter__(self):
            return self

        def __next__(self):
            return self.iter.__next__()

    class TimeStepsForFanin:
        def __init__(self, node, spans):
            if len(node.fanin) == 0:
                self.iter = iter([])
                return
            timesteps = set(Scheduler.TimeStepsForEdge(node.fanin[0], spans))
            for i in range(1, len(node.fanin)):
                timesteps &= set(Scheduler.TimeStepsForEdge(node.fanin[i], spans))

            timesteps = sorted(timesteps)
            self.iter = iter(timesteps)

        def __iter__(self):
            return self

        def __next__(self):
            return self.iter.__next__()

    class DensePreserveVarsMap:
        def __init__(self, sparse_map):
            local_map = {}
            maxi = 0
            mini = math.inf
            for i in sparse_map:
                if i > maxi:
                    maxi = i
                if i < mini:
                    mini = i
            for i in range(maxi, mini - 1, -1):
                assert i >= mini
                assert i <= maxi
                if i not in sparse_map:
                    local_map[i] = local_map[i + 1]
                else:
                    local_map[i] = sparse_map[i]
            self.local_map = OrderedDict(sorted(local_map.items()))

        def __getitem__(self, index):
            return self.local_map[index]

        def items(self):
            return self.local_map.items()

    class DenseGenerateOrFetchVarsMap:
        def __init__(self, sparse_map):
            self.sparse_map = sparse_map

        def __getitem__(self, index):
            if index not in self.sparse_map:
                return 0
            return self.sparse_map[index]

    def ComputeMinimumMemoryRequired(self):
        min_memory_requirement = 0
        bottleneck_node = None
        for n in self.graph.nodes.values():
            mem_needed = 0
            for e in n.fanin:
                mem_needed += e.size
            for e in n.fanout:
                mem_needed += e.size
            if mem_needed > min_memory_requirement:
                min_memory_requirement = mem_needed
                bottleneck_node = n
        return (min_memory_requirement, bottleneck_node)

    def ComputeMaximumMemoryRequired(self):
        max_memory_requirement = 0
        for e in self.graph.edges.values():
            max_memory_requirement += e.size
        return max_memory_requirement

    # Compute the optimal memory schedule. Returns the (summary, schedule)
    # pair, where summary is a map containing the following entries:
    #   peak_mem_usage: maximum total size of the tensors at any time
    #   total_data_swapped: total amount of data spilled and restored
    #   required_memory: total amount of memory needed to run the model. This
    #   corresponds to the peak_memory_usage plus space wasted due to
    #   fragmentation
    def ComputeOptimalSchedule(
        self,
        mem_limit=sys.maxsize,
        allow_rematerialization=False,
        allow_swaps=False,
        account_for_fragmentation=False,
        defrag=False,
        user_schedule=None,
        max_spills=None,
    ):
        # Compute the minimum amount of memory required to run the graph.
        min_memory_requirement, bottleneck_node = self.ComputeMinimumMemoryRequired()
        min_memory_requirement_acurate = False
        if min_memory_requirement > mem_limit:
            raise ValueError(
                "The graph requires at least %d bytes to run due to node %s (limit: %d)"
                % (min_memory_requirement, bottleneck_node.name, mem_limit)
            )

        schedule_constraints = {}
        if user_schedule is not None:
            for n, ts in user_schedule.items():
                if ts <= 0:
                    raise ValueError(
                        "Negative or null timestep specified for node %s" % str(n)
                    )
                if isinstance(n, str):
                    name = n
                else:
                    assert isinstance(n, dataflow_graph.Node)
                    name = n.name
                if name not in self.graph.nodes:
                    print(
                        "Invalid schedule: node "
                        + name
                        + " does not exist in the graph"
                    )
                    continue
                n = self.graph.nodes[name]
                schedule_constraints[n] = ts

        num_timesteps = self.num_nodes
        if not allow_rematerialization and not allow_swaps:
            num_timesteps = min(
                int(math.ceil(self.longest_path_length * 1.01)), num_timesteps
            )
        num_timesteps = int(num_timesteps * self.timestep_factor)

        if self.timestep_factor < 1 and self.longest_path_length > num_timesteps:
            # Make sure we have enough timesteps to run the longest path
            print(
                f"Adjusting num_timesteps to {self.longest_path_length} to ensure there are enough steps to run the longest path"
            )
            num_timesteps = self.longest_path_length

        # Compute the range of times during which each tensor can be alive, based
        # purely on precedence constraints.
        asap = self.ComputeASAPSchedule(schedule_constraints)
        alap = self.ComputeALAPSchedule(schedule_constraints, num_timesteps)
        makespan = self.ComputeMakespans(asap, alap)

        spans = (makespan, asap, alap)

        # GCD
        tensor_sizes = [t.size for t in self.graph.edges.values() if t.size > 0]
        gcd = self._GCD(tensor_sizes)
        max_address = min(self.ComputeMaximumMemoryRequired(), mem_limit) // gcd

        int_feas_tol = None
        if defrag or account_for_fragmentation:
            int_feas_tol = min(1e-5, 1.0 / max_address)
            int_feas_tol = max(1e-9, int_feas_tol)
            if 1.0 / max_address > 1e-5:
                print(f"Tightened IntFeasTol to {int_feas_tol}")

        solver = ilp_solver.ILPSolver(
            timeout_s=self.timeout,
            rel_stop=self.rel_stop,
            solver=self.solver,
            int_feas_tol=int_feas_tol,
            extra_params={"MIPFocus": 1},
        )

        # Create 2 new variable for each tensor and timestep: generate and preserve
        generate_vars = defaultdict(lambda: {})
        preserve_vars = defaultdict(lambda: {})
        fetch_vars = defaultdict(lambda: {})

        for e in self.graph.edges.values():
            lb, ub = makespan[e]
            for t in self.TimeStepsForEdge(e, spans):
                v = solver.create_binary_var(e.name + "_generate_ts" + str(t))
                generate_vars[e][t] = v
                v = solver.create_binary_var(e.name + "_preserve_ts" + str(t))
                preserve_vars[e][t] = v
                v = solver.create_binary_var(e.name + "_fetch_ts" + str(t))
                fetch_vars[e][t] = v

            if e.source in schedule_constraints:
                ts = schedule_constraints[e.source]
                assert ts >= lb
                assert ts <= ub
                for t in self.TimeStepsForEdge(e, spans):
                    if t != ts and not allow_rematerialization:
                        solver.add_constraint(
                            generate_vars[e][t] == 0,
                            name=f"{utils.get_linenumber()}_{e.name}_generate_var@{t}",
                        )
                    else:
                        solver.add_constraint(
                            generate_vars[e][t] == 1,
                            name=f"{utils.get_linenumber()}_{e.name}_generate_var@{t}",
                        )

            for snk in e.sinks:
                if snk in schedule_constraints:
                    ts = schedule_constraints[snk]
                    assert ts >= lb
                    assert ts <= ub
                    solver.add_constraint(
                        preserve_vars[e][ts] + fetch_vars[e][ts] == 1,
                        name=f"{utils.get_linenumber()}_{e.name}_preserve_or_fetch_var@{ts}",
                    )

        spilling_allowed = allow_swaps and (max_spills is None or max_spills > 0)

        # Add correctness constraints: we can't preserve data unless it's been
        # generated or preserved at the previous timestep. Also it doesn't make
        # sense to preserve and compute same data at the same timestep
        for e in self.graph.edges.values():
            prev = self.TimeStepsForEdge(e, spans)
            cur = self.TimeStepsForEdge(e, spans, startoffset=1)
            for t in cur:
                p = prev.__next__()

                solver.add_constraint(
                    preserve_vars[e][t]
                    <= preserve_vars[e][p] + generate_vars[e][p] + fetch_vars[e][p],
                    name=f"{utils.get_linenumber()}_{e.name}_precedence@{t}",
                )
                solver.add_constraint(
                    preserve_vars[e][t] + generate_vars[e][t] + fetch_vars[e][t] <= 1,
                    name=f"{utils.get_linenumber()}_{e.name}_at_most_one@{t}",
                )

            # Purely to help the solver. Todo: improve the encoding to avoid
            # creating these variables in the first place
            lb, _ = makespan[e]
            solver.add_constraint(
                preserve_vars[e][lb] == 0,
                name=f"{utils.get_linenumber()}_{e.name}_preserve_var@{lb}",
            )
            solver.add_constraint(
                fetch_vars[e][lb] == 0,
                name=f"{utils.get_linenumber()}_{e.name}_fetch_var@{lb}",
            )
            if lb + 1 in fetch_vars[e]:
                solver.add_constraint(
                    fetch_vars[e][lb + 1] == 0,
                    name=f"{utils.get_linenumber()}_{e.name}_fetch_var@{lb+1}",
                )
            for t in self.TimeStepsForEdge(e, spans):
                if t > alap[e.source]:
                    # if not allow_rematerialization or e.is_stateful():
                    solver.add_constraint(
                        generate_vars[e][t] == 0,
                        name=f"{utils.get_linenumber()}_{e.name}_generate_var_past_alap@{t}",
                    )

            # Purely to help the solver: there is no need to swap the control edges.
            if e.size == 0 or not spilling_allowed:
                for t in self.TimeStepsForEdge(e, spans):
                    solver.add_constraint(
                        fetch_vars[e][t] == 0,
                        name=f"{utils.get_linenumber()}_{e.name}_fetch_var@{t}",
                    )

            # Control edges are always available once they've been triggered.
            if e.size == 0:
                prev = self.TimeStepsForEdge(e, spans)
                for t in self.TimeStepsForEdge(e, spans, startoffset=1):
                    p = prev.__next__()
                    if t <= alap[e.source]:
                        solver.add_constraint(
                            preserve_vars[e][t]
                            >= preserve_vars[e][p] + generate_vars[e][p],
                            name=f"{utils.get_linenumber()}_{e.name}_ctrl_edge@{t}",
                        )
                    else:
                        # The source node has been run at this point.
                        solver.add_constraint(
                            preserve_vars[e][t] == 1,
                            name=f"{utils.get_linenumber()}_{e.name}_preserve_var@{t}",
                        )

        # Add precedence constraints: we need all the nodes inputs to be
        # available in memory at time t in order to evaluate the node.
        # We also ensure that all the fanouts of a node are generated at the
        # same time. This is only an optimization, since the objective of
        # minimizing the number of computation would take care of that on its on.
        for n in self.graph.nodes.values():
            if len(n.fanout) > 1:
                for i in range(1, len(n.fanout)):
                    for t in self.TimeStepsForNode(n, spans):
                        solver.add_constraint(
                            generate_vars[n.fanout[0]][t]
                            == generate_vars[n.fanout[i]][t],
                            name=f"{utils.get_linenumber()}_{n.name}_all_fanouts_generated@{t}",
                        )

            for snk in n.fanout:
                for src in n.fanin:
                    # Add precedence constraints
                    for t in self.TimeStepsForNode(n, spans):
                        solver.add_constraint(
                            generate_vars[snk][t]
                            <= preserve_vars[src][t] + fetch_vars[src][t],
                            name=f"{utils.get_linenumber()}_{snk.name}_{src.name}_precedence@{t}",
                        )

        # We can't swap the data back in unless it's already been generated
        # (and implicitely swapped out instead of being simply discarded).
        for e in self.graph.edges.values():
            if e.size == 0:
                continue
            cur = self.TimeStepsForEdge(e, spans)
            try:
                p = cur.__next__()
                previously_generated = generate_vars[e][p]
                for t in cur:
                    if t > alap[e.source]:
                        break
                    solver.add_constraint(
                        fetch_vars[e][t] <= previously_generated,
                        name=f"{utils.get_linenumber()}_{e.name}_fetchable@{t}",
                    )
                    previously_generated += generate_vars[e][t]
                    p = t
            except StopIteration:
                pass

        # Force the generation of each tensor at least once (or exactly once if
        # rematerialization is not allowed)
        for e, ts in generate_vars.items():
            s = 0
            for v in ts.values():
                s += v
            if (
                allow_rematerialization
                and not e.is_stateful()
                and e.size > 0
                and e.source.time is not None
            ):
                solver.add_constraint(
                    1 <= s,
                    name=f"{utils.get_linenumber()}_{e.name}_materialized_at_least_once",
                )
            else:
                solver.add_constraint(
                    1 == s,
                    name=f"{utils.get_linenumber()}_{e.name}_materialized_once",
                )

        # A node needs to consume all its inputs at the same timestep. Handle
        # the case where the node has no fanout below. The case where the node
        # has fanout is already handled.
        for n in self.graph.nodes.values():
            if len(n.fanout) > 0:
                continue
            if len(n.fanin) <= 1:
                continue

            # We need at least one timestep during which all the inputs are live at the same time
            sum_of_all_live = 0
            for t in self.TimeStepsForFanin(n, spans):
                if t > alap[n]:
                    break
                all_live = solver.create_binary_var(
                    "fanin_of_" + n.name + "_live_at_ts" + str(t)
                )

                for f in n.fanin:
                    solver.add_constraint(
                        all_live <= preserve_vars[f][t] + fetch_vars[f][t],
                        name=f"{utils.get_linenumber()}_fanin_of_{n.name}_dead_due_to_{f.name}@{t}",
                    )
                sum_of_all_live += all_live
            solver.add_constraint(
                sum_of_all_live >= 1,
                name=f"{utils.get_linenumber()}_fanin_of_{n.name}_live_at_one_ts",
            )

        # Ensure that weights are preserved in memory during the whole
        # computation if spilling is disabled.
        if not spilling_allowed:
            for e in self.graph.edges.values():
                if not e.is_stateful():
                    continue
                first_timestep = True
                for t in self.TimeStepsForEdge(e, spans):
                    if first_timestep:
                        solver.add_constraint(
                            generate_vars[e][t] == 1,
                            name=f"{utils.get_linenumber()}_cannot_spill_{e.name}_@{t}",
                        )
                        solver.add_constraint(
                            preserve_vars[e][t] == 0,
                            name=f"{utils.get_linenumber()}_cannot_spill_{e.name}_@{t}",
                        )
                    else:
                        solver.add_constraint(
                            generate_vars[e][t] == 0,
                            name=f"{utils.get_linenumber()}_cannot_spill_{e.name}_@{t}",
                        )
                        solver.add_constraint(
                            preserve_vars[e][t] == 1,
                            name=f"{utils.get_linenumber()}_cannot_spill_{e.name}_@{t}",
                        )
                    first_timestep = False

        # Ensure that activations are preserved in memory once they've been generated
        # and until their last consumer runs
        if not spilling_allowed:
            for e in self.graph.edges.values():
                if e.is_stateful():
                    continue
                first = alap[e.source] + 1
                last = 0
                for v in e.sinks:
                    last = max(last, asap[v])
                for ts in self.TimeStepsForEdge(e, spans):
                    if ts < first or ts > last:
                        continue
                    solver.add_constraint(
                        preserve_vars[e][ts] == 1,
                        name=f"{utils.get_linenumber()}_{e.name}_must_be_preserved_@{t}",
                    )

        # Memory usage at each timestep
        mem_at_timestep = defaultdict(lambda: 0)
        persistent_weight_size = 0
        for e in self.graph.edges.values():
            if e.size == 0:
                continue
            if (not spilling_allowed) and e.is_stateful():
                # This edge always reside in memory. We don't need to track it separately
                persistent_weight_size += e.size
                continue

            for t, v in self.DensePreserveVarsMap(preserve_vars[e]).items():
                mem_at_timestep[t] += v * (e.size // gcd)

            for t, v in generate_vars[e].items():
                mem_at_timestep[t] += v * (e.size // gcd)

            for t, v in fetch_vars[e].items():
                mem_at_timestep[t] += v * (e.size // gcd)

        if max_spills is None:
            # We need to fit withing the memory budget at each timestep
            # TODO: check if leveraging Bradley, Hammer, and Wolsey to optimize these
            # constraints helps the solver
            liveness = defaultdict(lambda: [])
            for e in self.graph.edges.values():
                if e.size == 0:
                    continue
                lb, ub = makespan[e]
                for ts in range(lb, ub + 1):
                    liveness[ts].append(e)
            for ts, mem_usage in mem_at_timestep.items():
                max_mem = 0
                for e in liveness[ts]:
                    max_mem += e.size
                if max_mem < mem_limit:
                    # No point in adding a constraint
                    continue
                max_mem -= persistent_weight_size
                solver.add_constraint(
                    mem_usage <= (mem_limit - persistent_weight_size) // gcd,
                    name=f"{utils.get_linenumber()}_mem_usage_less_than_{mem_limit}@{ts}",
                )

        elif max_spills > 0:
            # We need to ensure that we don't exceed our total spill budget
            spills_per_tensor = defaultdict(lambda: 0)
            for e, ts in fetch_vars.items():
                if e.size > 0:
                    for _, v in ts.items():
                        spills_per_tensor[e] += v

            total_spills = 0
            for e, s in spills_per_tensor.items():
                is_spilled = solver.create_binary_var("is_" + e.name + "_spilled")
                solver.add_constraint(is_spilled <= s)
                lb, ub = makespan[e]
                max_spills = min(len(e.sinks), ub - lb + 1)
                solver.add_constraint(is_spilled * max_spills >= s)
                total_spills += (s + is_spilled) * (e.size // gcd)

            for n in self.graph.nodes.values():
                if n.op_type != "stateful_node_sink":
                    continue
                for edge in n.fanin:
                    missing_during_startup = 1 - generate_vars[edge][1]
                    missing_during_cleanup = 1 - preserve_vars[edge][num_timesteps]
                    extra_spill = solver.create_binary_var(edge.name + "_spilled")
                    solver.add_constraint(extra_spill >= missing_during_startup)
                    solver.add_constraint(extra_spill >= missing_during_cleanup)
                    # Unnecessary and slows us down
                    # solver.add_constraint(
                    #    extra_spill <= missing_during_startup + missing_during_cleanup
                    # )
                    factor = 1 if edge.source.read_only else 2
                    total_spills += extra_spill * (edge.size * factor // gcd)

            # TODO: check if leveraging Bradley, Hammer, and Wolsey to optimize these
            # constraints helps the solver
            solver.add_constraint(total_spills <= max_spills // gcd)

        if account_for_fragmentation and not defrag:
            max_address = (
                min(self.ComputeMaximumMemoryRequired(), mem_limit)
                - persistent_weight_size
            ) // gcd

            # Create a new variable for each tensor that tracks the base address
            # of the tensor. If the tensor is of size 0, we force its address to be
            # 0 to help the solver
            addresses = OrderedDict()
            for tensor in self.graph.edges.values():
                if tensor.size > 0 and (not tensor.is_stateful() or spilling_allowed):
                    v = solver.create_integer_var(
                        tensor.name,
                        lower_bound=0,
                        upper_bound=max_address - tensor.size // gcd,
                    )
                    addresses[tensor] = v

            # Manually assign some of the addresses when possible
            fixed_locations = {}
            manual_allocation_possible = False
            if not spilling_allowed:
                manual_allocation_possible = True
                for n in self.graph.nodes.values():
                    if n.is_stateful():
                        continue
                    if n not in user_schedule:
                        manual_allocation_possible = False
                        break

            max_mem = 0

            if manual_allocation_possible:
                print("STARTING TENSOR ASSIGNMENT")
                min_start = 0
                max_end = num_timesteps
                base_address = 0
                processed = set()
                mem_used = intervaltree.IntervalTree()
                while max_end > min_start:
                    max_duration = 0
                    next_step = None
                    for tensor, span in makespan.items():
                        if tensor.size == 0 or (
                            tensor.is_stateful() and not spilling_allowed
                        ):
                            continue
                        if span[0] < min_start:
                            continue
                        if span[1] > max_end:
                            continue
                        if tensor in processed:
                            continue
                        duration = span[1] - span[0]
                        if duration > max_duration:
                            max_duration = duration
                            next_step = tensor
                    if not next_step:
                        break
                    solver.add_constraint(
                        addresses[next_step] == base_address,
                        name=f"{utils.get_linenumber()}_force_{next_step.name}_at_{base_address}",
                    )
                    fixed_locations[next_step] = base_address
                    print(
                        f"   TENSOR {next_step.name}, min start {min_start}, max_end {max_end} size {next_step.size} located at {base_address}"
                    )

                    base_address += next_step.size // gcd
                    min_start = makespan[next_step][0]
                    max_end = makespan[next_step][1]
                    processed.add(next_step)
                    mem_used[min_start : max_end + 1] = base_address

                max_mem = base_address
                for t, a in addresses.items():
                    if t in fixed_locations:
                        a.Start = fixed_locations[t]
                    else:
                        span = makespan[t]
                        max_address_used = 0
                        # print(f"Querying intervaltree {span[0]} {span[1]+1}")
                        for interval in mem_used.overlap(span[0], span[1] + 1):
                            # print(f"address {interval.data} used")
                            max_address_used = max(max_address_used, interval.data)
                        a.Start = max_address_used
                        # print(f"Adding gen address to intervaltree {span[0]} {span[1]}")
                        mem_used[span[0] : span[1] + 1] = (
                            max_address_used + t.size // gcd
                        )
                        max_mem = max(max_mem, max_address_used + t.size // gcd)

                for t, a in addresses.items():
                    if t not in fixed_locations:
                        solver.add_constraint(
                            a <= max_mem - t.size // gcd,
                            name=f"{utils.get_linenumber()}_tighten_max_address_for_{t.name}",
                        )

                # Refine the minimum memory requirement
                min_mem_per_span = intervaltree.IntervalTree()
                for t in addresses.keys():
                    span = makespan[t]
                    min_mem_per_span[span[0] : span[1] + 1] = t
                min_mem_required = 0
                for t in range(1, num_timesteps + 1):
                    # print(f"LOOKING AT TIMESTEP {t}")
                    mem_at_t = 0
                    for interval in min_mem_per_span[t]:
                        t = interval.data
                        # print(f"tensor {t.name} used. span was [{makespan[t][0]}, {makespan[t][1]}]")
                        mem_at_t += t.size
                    min_mem_required = max(min_mem_required, mem_at_t)
                min_memory_requirement = min_mem_required
                min_memory_requirement_acurate = True
                print("DONE WITH TENSOR ASSIGNMENT")

            processed = set()
            for t1, span1 in makespan.items():
                if t1.size == 0 or (t1.is_stateful() and (not spilling_allowed)):
                    continue
                # Help the solver by providing upper bounds for all the addresses
                # solver.add_constraint(addresses[t1] + t1.size // gcd <= max_address)
                for t2, span2 in makespan.items():
                    if t2.size == 0 or (t2.is_stateful() and (not spilling_allowed)):
                        continue
                    if t1 is t2 or (t2, t1) in processed:
                        continue
                    processed.add((t1, t2))
                    if (
                        span1[1] < span2[0]
                        or span1[0] > span2[1]
                        or not self.graph.can_overlap_in_time(t1, t2)
                    ):
                        # print(t1.name + " and " + t2.name + "CANNOT OVERLAP")
                        continue

                    if t1 in fixed_locations and t2 in fixed_locations:
                        continue

                    live_together = self.graph.are_connected_by_node(t1, t2)
                    if not live_together and not spilling_allowed:
                        if alap[t1.source] <= asap[t2.source]:
                            for snk in t1.sinks:
                                if asap[snk] >= alap[t2.source]:
                                    live_together = True
                        elif alap[t2.source] <= asap[t1.source]:
                            for snk in t2.sinks:
                                if asap[snk] >= alap[t1.source]:
                                    live_together = True

                    if live_together:
                        if t1 in fixed_locations:
                            solver.add_constraint(
                                addresses[t2] >= fixed_locations[t1] + t1.size // gcd,
                                name=f"{utils.get_linenumber()}_force_{t1.name}_below_{t2.name}",
                            )

                        elif t2 in fixed_locations:
                            solver.add_constraint(
                                addresses[t1] >= fixed_locations[t2] + t2.size // gcd,
                                name=f"{utils.get_linenumber()}_force_{t1.name}_above_{t2.name}",
                            )

                        else:
                            v = solver.create_binary_var(
                                name=f"{utils.get_linenumber()}_{t1.name}_below_{t2.name}"
                            )
                            solver.add_constraint(
                                addresses[t1] + t1.size // gcd - addresses[t2]
                                <= (1 - v) * max_address,
                                name=f"{utils.get_linenumber()}_force_{t1.name}_below_{t2.name}",
                            )
                            solver.add_constraint(
                                addresses[t1] - addresses[t2] - t2.size // gcd
                                >= -v * max_address,
                                name=f"{utils.get_linenumber()}_force_{t1.name}_above_{t2.name}",
                            )

                    elif span1[1] >= span2[0] and span1[0] <= span2[1]:
                        # The spans may overlap: if they do, one of these 2 constraints must hold:
                        # variables[t1] + t1.size <= variables[t2]
                        # variables[t1] >= variables[t2] + t2.size
                        v1 = solver.create_binary_var(t1.name + "_" + t2.name + "_v1")
                        solver.add_constraint(
                            addresses[t1] + t1.size // gcd - addresses[t2]
                            <= (1 - v1) * max_address,
                            name=f"{utils.get_linenumber()}_{t1.name}_below_{t2.name}",
                        )

                        v2 = solver.create_binary_var(t1.name + "_" + t2.name + "_v2")
                        solver.add_constraint(
                            addresses[t1] - addresses[t2] - t2.size // gcd
                            >= (v2 - 1) * max_address,
                            name=f"{utils.get_linenumber()}_{t1.name}_above_{t2.name}",
                        )

                        if t1 in fixed_locations:
                            solver.add_constraint(v1 == 1)
                        elif t2 in fixed_locations:
                            solver.add_constraint(v2 == 1)
                        else:
                            # Let's check if that helps
                            solver.add_constraint(v1 + v2 <= 1)

                        # check if they actually do overlap
                        generate_t1 = self.DenseGenerateOrFetchVarsMap(
                            generate_vars[t1]
                        )
                        preserve_t1 = self.DensePreserveVarsMap(preserve_vars[t1])
                        fetch_t1 = self.DenseGenerateOrFetchVarsMap(fetch_vars[t1])
                        generate_t2 = self.DenseGenerateOrFetchVarsMap(
                            generate_vars[t2]
                        )
                        preserve_t2 = self.DensePreserveVarsMap(preserve_vars[t2])
                        fetch_t2 = self.DenseGenerateOrFetchVarsMap(fetch_vars[t2])

                        for ts in range(
                            max(span1[0], span2[0]), min(span1[1], span2[1]) + 1
                        ):
                            live1 = generate_t1[ts] + preserve_t1[ts] + fetch_t1[ts]
                            live2 = generate_t2[ts] + preserve_t2[ts] + fetch_t2[ts]
                            overlap_at_t = live1 + live2 - 1
                            solver.add_constraint(
                                v1 + v2 >= overlap_at_t,
                                name=f"{utils.get_linenumber()}_{t1.name}_overlaps_{t2.name}@{ts}",
                            )

        #####################################################

        elif defrag:

            # Maximum address that can be used
            max_address = (
                min(self.ComputeMaximumMemoryRequired(), mem_limit)
                - persistent_weight_size
            ) // gcd

            # Create a new variable for each tensor that tracks the base address
            # of the tensor. If the tensor is of size 0, we force its address to be
            # 0 to help the solver
            addresses = defaultdict(lambda: {})
            for tensor in self.graph.edges.values():
                for t in self.TimeStepsForEdge(tensor, spans):
                    if tensor.size > 0 and (
                        not tensor.is_stateful() or spilling_allowed
                    ):
                        v = solver.create_integer_var(
                            tensor.name + "@" + str(t),
                            lower_bound=0,
                            upper_bound=max_address - tensor.size // gcd,
                        )
                        addresses[tensor][t] = v

            for e in self.graph.edges.values():
                if e.size == 0 or (e.is_stateful() and (not spilling_allowed)):
                    continue
                addresses_e = self.DensePreserveVarsMap(addresses[e])
                preserve_e = self.DensePreserveVarsMap(preserve_vars[e])
                prev = self.TimeStepsForEdge(e, spans)
                for t in self.TimeStepsForEdge(e, spans, startoffset=1):
                    # If preserve[t], then address[t] must be equal to address[t-1]. This is encoded as:
                    # address[t] - address[t-1] <= (1-preserve) * max_address
                    # address[t-1] - address[t] <= (1-preserve) * max_address
                    p = prev.__next__()
                    solver.add_constraint(
                        addresses_e[t] - addresses_e[p]
                        <= (1 - preserve_e[t]) * max_address
                    )
                    solver.add_constraint(
                        addresses_e[p] - addresses_e[t]
                        <= (1 - preserve_e[t]) * max_address
                    )

            liveness = defaultdict(lambda: [])
            for e in self.graph.edges.values():
                if e.size == 0 or (e.is_stateful() and (not spilling_allowed)):
                    continue
                lb, ub = makespan[e]
                for ts in range(lb, ub + 1):
                    liveness[ts].append(e)

            for ts, edges in liveness.items():
                processed = set()
                for t1 in edges:
                    for t2 in edges:
                        if t1 is t2 or (t2, t1) in processed:
                            continue
                        processed.add((t1, t2))
                        if not self.graph.can_overlap_in_time(t1, t2):
                            # print(t1.name + " and " + t2.name + "CANNOT OVERLAP 2")
                            continue
                        # Check if both t1 and t2 are live at ts.
                        generate_t1 = self.DenseGenerateOrFetchVarsMap(
                            generate_vars[t1]
                        )
                        preserve_t1 = self.DensePreserveVarsMap(preserve_vars[t1])
                        fetch_t1 = self.DenseGenerateOrFetchVarsMap(fetch_vars[t1])
                        addresses_t1 = self.DensePreserveVarsMap(addresses[t1])
                        generate_t2 = self.DenseGenerateOrFetchVarsMap(
                            generate_vars[t2]
                        )
                        preserve_t2 = self.DensePreserveVarsMap(preserve_vars[t2])
                        fetch_t2 = self.DenseGenerateOrFetchVarsMap(fetch_vars[t2])
                        addresses_t2 = self.DensePreserveVarsMap(addresses[t2])

                        live1 = generate_t1[ts] + preserve_t1[ts] + fetch_t1[ts]
                        live2 = generate_t2[ts] + preserve_t2[ts] + fetch_t2[ts]
                        # overlap_at_ts = solver.create_binary_var(
                        #    t1.name + "_" + t2.name + "_overlap_at_" + str(ts)
                        # )
                        # solver.add_constraint(overlap_at_ts <= live1)
                        # solver.add_constraint(overlap_at_ts <= live2)
                        # solver.add_constraint(overlap_at_ts + 1 >= live1 + live2)
                        # Make sure overlap at ts < 1 if the 2 tensors don't overlap in time.
                        overlap_at_ts = live1 + live2 - 1

                        # if t1 and t2 are both live at ts, one of these 2 constraints must hold:
                        # variables[t1] + t1.size <= variables[t2]
                        # variables[t1] >= variables[t2] + t2.size
                        # If the size of t1 is the same as the size of t2, we only use the first
                        # constraint.
                        if False:  # t1.size == t2.size:
                            v1 = solver.create_binary_var(
                                t1.name + "_" + t2.name + "_v1_at_" + str(ts)
                            )
                            solver.add_constraint(
                                addresses_t1[ts] + t1.size // gcd - addresses_t2[ts]
                                <= (1 - v1) * max_address
                            )
                            solver.add_constraint(overlap_at_ts <= v1)
                        else:
                            v1 = solver.create_binary_var(
                                t1.name + "_" + t2.name + "_v1_at_" + str(ts)
                            )
                            solver.add_constraint(
                                addresses_t1[ts] + t1.size // gcd - addresses_t2[ts]
                                <= (1 - v1) * max_address
                            )
                            v2 = solver.create_binary_var(
                                t1.name + "_" + t2.name + "_v2_at_" + str(ts)
                            )
                            solver.add_constraint(
                                addresses_t1[ts] - addresses_t2[ts] - t2.size // gcd
                                >= (v2 - 1) * max_address
                            )

                            solver.add_constraint(v1 + v2 >= overlap_at_ts)
                            # solver.add_constraint(2 - v1 - v2 >= overlap_at_ts)

        #####################################################

        # Add objective function
        s = 0
        if allow_rematerialization:
            # Minimize the number of data generations
            for e, ts in generate_vars.items():
                if not e.source.time:
                    continue
                for v in ts.values():
                    s += v * e.source.time

        if max_spills is None:
            # Minimize the number of spills
            for e, ts in fetch_vars.items():
                for v in ts.values():
                    s += v * (e.size * 2 // gcd)

            for n in self.graph.nodes.values():
                if n.op_type != "stateful_node_sink":
                    continue
                for edge in n.fanin:
                    missing_during_startup = 1 - generate_vars[edge][1]
                    missing_during_cleanup = 1 - preserve_vars[edge][num_timesteps]
                    extra_spill = solver.create_binary_var(edge.name + "_extra_spill")
                    solver.add_constraint(
                        extra_spill >= missing_during_startup,
                        name=f"{utils.get_linenumber()}_spilled_at_startup_for_{edge.name}",
                    )
                    solver.add_constraint(
                        extra_spill >= missing_during_cleanup,
                        name=f"{utils.get_linenumber()}_spilled_at_cleanup_for_{edge.name}",
                    )
                    # Unnecessary and slows us down
                    # solver.add_constraint(
                    #    extra_spill <= missing_during_startup + missing_during_cleanup
                    # )
                    factor = 1 if edge.source.read_only else 2
                    s += extra_spill * (edge.size * factor // gcd)

        if max_spills is not None and not allow_rematerialization:
            # Minimize peak memory usage
            min_peak_mem = min_memory_requirement
            if not min_memory_requirement_acurate and not spilling_allowed:
                for n in self.graph.nodes.values():
                    if n.is_stateful():
                        continue
                    node_activations = 0
                    for f in itertools.chain(n.fanin, n.fanout):
                        if not f.is_stateful():
                            node_activations += f.size
                    min_peak_mem = min(min_peak_mem, node_activations)

            max_usage = (
                min(self.ComputeMaximumMemoryRequired(), mem_limit)
                - persistent_weight_size
            ) // gcd
            v = solver.create_integer_var(
                "peak_memory_usage",
                lower_bound=min_peak_mem // gcd,
                upper_bound=max_usage,
            )
            s += v
            if defrag:
                for t, p in addresses.items():
                    for ts, a in p.items():
                        solver.add_constraint(
                            v >= a + t.size // gcd,
                            name=f"{utils.get_linenumber()}_max_address_above_{t.name}@{ts}",
                        )
            elif account_for_fragmentation:
                for t, a in addresses.items():
                    solver.add_constraint(
                        v >= a + t.size // gcd,
                        name=f"{utils.get_linenumber()}_max_address_above_{t.name}",
                    )
            else:
                for ts, m in mem_at_timestep.items():
                    solver.add_constraint(
                        v >= m, name=f"{utils.get_linenumber()}_max_address@{ts}"
                    )

        solver.set_objective_function(s, maximize=False)

        # start_time = time.time()
        # print("Start ILP solver")
        # print("PROBLEM STATS = " + str(solver))

        result = solver.solve()
        # print(f"ILP solver time: {time.time()-start_time} seconds")

        if self.print_relaxation:
            relaxed_solution = solver.solve_relaxation()
            print("CHECKING LP RELAXATION")
            for var, value in result.items():
                if var.VType == "B":
                    if abs(value - relaxed_solution[var.varName]) > 0.5:
                        print(
                            f" Var {var.varName} flipped: relaxed value {relaxed_solution[var.varName]} vs integral value {value}"
                        )
                    elif abs(value - relaxed_solution[var.varName]) > 0.05:
                        print(
                            f" Var {var.varName} far from relaxed value {relaxed_solution[var.varName]}: integral value {value}"
                        )
                else:
                    if abs(value - relaxed_solution[var.varName]) > 0.5:
                        print(
                            f" Var {var.varName} changed: relaxed value {relaxed_solution[var.varName]} vs integral value {value}"
                        )

        last_uses = {}
        for e in self.graph.edges.values():
            last_use = 0
            for sink in e.sinks:
                if len(sink.fanout) == 0:
                    last_use = max(last_use, alap[sink])
                else:
                    for fanout in sink.fanout:
                        for t, v in generate_vars[fanout].items():
                            if result[v] >= 0.99:
                                last_use = max(last_use, t)

            last_uses[e] = last_use

        schedule = defaultdict(lambda: ([], [], []))
        mem_locations = defaultdict(lambda: {})
        materialization_count = {}
        for n, ts in generate_vars.items():
            tensor_materialization_count = 0
            for t, v in ts.items():
                if result[v] >= 0.99:
                    tensor_materialization_count += 1
                    if account_for_fragmentation:
                        if n.size == 0:
                            schedule[n][0].append(str(t) + "[ctrl]")
                        elif n.is_stateful() and (not spilling_allowed):
                            schedule[n][0].append(str(t) + "[weight]")
                        else:
                            if defrag:
                                schedule[n][0].append(
                                    str(t)
                                    + "@"
                                    + str(int(result[addresses[n][t]] * gcd))
                                )
                                mem_locations[t][n] = int(result[addresses[n][t]] * gcd)
                            else:
                                schedule[n][0].append(
                                    str(t) + "@" + str(int(result[addresses[n]] * gcd))
                                )
                                mem_locations[t][n] = int(result[addresses[n]] * gcd)

                    else:
                        schedule[n][0].append(t)
            # Don't double count nodes with multiple outputs
            if n.source in materialization_count:
                assert materialization_count[n.source] == tensor_materialization_count
            else:
                materialization_count[n.source] = tensor_materialization_count

        # Sanity check: each tensor must have been generated in the [asap, alap] window
        # of its source node
        for n, ts in generate_vars.items():
            src = n.source
            last = alap[src]
            generated = 0
            for t, v in ts.items():
                if t > last:
                    break
                generated += result[v]
            assert generated >= 0.99

        for n, ts in preserve_vars.items():
            if account_for_fragmentation and defrag and n.size > 0:
                addresses_n = self.DensePreserveVarsMap(addresses[n])
            for t, v in self.DensePreserveVarsMap(ts).items():
                if t > last_uses[n]:
                    continue
                if result[v] >= 0.99:
                    schedule[n][1].append(t)
                    if n.size == 0:
                        continue
                    if n.is_stateful() and (not spilling_allowed):
                        continue
                    if defrag:
                        mem_locations[t][n] = int(result[addresses_n[t]] * gcd)
                    elif account_for_fragmentation:
                        mem_locations[t][n] = int(result[addresses[n]] * gcd)

        for n, ts in fetch_vars.items():
            for t, v in ts.items():
                if t > last_uses[n]:
                    continue
                if result[v] >= 0.99:
                    if defrag:
                        schedule[n][2].append(
                            str(t) + "@" + str(int(result[addresses[n][t]] * gcd))
                        )
                        mem_locations[t][n] = int(result[addresses[n][t]] * gcd)
                    else:
                        schedule[n][2].append(t)
                        if account_for_fragmentation:
                            mem_locations[t][n] = int(result[addresses[n]] * gcd)

        mem_at_timestep = defaultdict(lambda: 0)
        tensors_swapped = defaultdict(lambda: 0)
        for e, ts in preserve_vars.items():
            for t, v in self.DensePreserveVarsMap(ts).items():
                if t <= last_uses[e]:
                    mem_at_timestep[t] += result[v] * e.size
        for e, ts in generate_vars.items():
            for t, v in ts.items():
                if t <= last_uses[e]:
                    mem_at_timestep[t] += result[v] * e.size
        for e, ts in fetch_vars.items():
            for t, v in ts.items():
                if t > last_uses[e]:
                    continue
                mem_at_timestep[t] += result[v] * e.size
                if result[v] >= 0.99:
                    tensors_swapped[e] += 1

        peak_mem_usage = 0
        for mem_usage in mem_at_timestep.values():
            peak_mem_usage = max(peak_mem_usage, mem_usage)

        total_data_swapped = 0
        for e, count in tensors_swapped.items():
            total_data_swapped += (count + 1) * e.size

        for n in self.graph.nodes.values():
            if n.op_type != "stateful_node_sink":
                continue
            if not spilling_allowed:
                continue
            for edge in n.fanin:
                present_at_startup = result[generate_vars[edge][1]]
                present_at_cleanup = result[preserve_vars[edge][num_timesteps]]
                if not present_at_startup or not present_at_cleanup:
                    factor = 1 if edge.source.read_only else 2
                    total_data_swapped += edge.size * factor

        if defrag:
            required_memory = 0
            for t, p in addresses.items():
                if t.size == 0:
                    continue
                elif t.is_stateful() and (not spilling_allowed):
                    continue
                for timestamp, a in p.items():
                    if timestamp <= last_uses[e]:
                        required_memory = max(required_memory, result[a] * gcd + t.size)
            required_memory += persistent_weight_size

        elif account_for_fragmentation:
            required_memory = 0
            for t, a in addresses.items():
                if t.size == 0:
                    continue
                elif t.is_stateful() and (not spilling_allowed):
                    continue
                required_memory = max(required_memory, result[a] * gcd + t.size)
            required_memory += persistent_weight_size

        else:
            required_memory = peak_mem_usage

        rematerialization_count = 0
        rematerialization_time = 0
        if allow_rematerialization:
            for node, count in materialization_count.items():
                if count == 1:
                    continue
                rematerialization_count += count - 1
                if node.time is not None:
                    rematerialization_time += (count - 1) * node.time

        summary = {
            "peak_mem_usage": peak_mem_usage,
            "total_data_swapped": total_data_swapped,
            "required_memory": required_memory,
            "rematerialization_count": rematerialization_count,
            "rematerialization_time": rematerialization_time,
        }
        return (summary, schedule, mem_locations)
