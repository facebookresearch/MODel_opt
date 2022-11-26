import logging
import math
import sys
import time
from collections import defaultdict, OrderedDict

from olla import dataflow_graph, ilp_solver


class Scheduler:
    def __init__(
        self, graph, timeout_s=None, rel_stop=None, solver="GUROBI", timestep_factor=1.0
    ):
        self.graph = graph
        self.timeout = timeout_s
        self.rel_stop = rel_stop
        self.solver = solver
        self.num_timesteps = int(len(graph.nodes) * timestep_factor)

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
                    "Infeasible user schedule specified for node %s" % str(vertex)
                )
            time = schedule_constraints[vertex]

        timings[vertex] = time

        return time

    # Compute the latest time a node can be scheduled. It's in the range
    # [1, m], where m is the number of nodes in the graph.
    def ComputeALAPSchedule(self, schedule_constraints):
        max_timesteps = self.num_timesteps
        if schedule_constraints:
            max_constraint = 0
            for v in schedule_constraints.values():
                max_constraint = max(max_constraint, v)
            max_timesteps += max_constraint
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

    def _GCD(self, values):
        if len(values) == 0:
            return 1
        if len(values) == 1:
            return values[0]
        elif len(values) == 2:
            return math.gcd(values[0], values[1])
        else:
            return math.gcd(values[0], self._GCD(values[1:]))

    # This old formulation can't be solved quickly and supports neither
    # swapping nor rematerialization. Use ComputeOptimalSchedule() below instead.
    def ComputeBestSchedule(self):
        # Compute the range of times during which each node can be alive, based
        # purely on precedence constraints.
        schedule_constraints = {}
        asap = self.ComputeASAPSchedule(schedule_constraints)
        alap = self.ComputeALAPSchedule(schedule_constraints)

        solver = ilp_solver.ILPSolver(
            timeout_s=self.timeout, rel_stop=self.rel_stop, solver=self.solver
        )

        # Create a new variable for each node
        node_vars = {}
        for n in self.graph.nodes.values():
            lb = asap[n]
            ub = alap[n]
            v = solver.create_integer_var(n.name, lower_bound=lb, upper_bound=ub)
            node_vars[n] = v

        # Add precedence constraints
        for e in self.graph.edges.values():
            src = node_vars[e.source]
            for s in e.sinks:
                snk = node_vars[s]
                # Add precedence constraints
                solver.add_constraint(src <= snk - 1)

        # Compute the liveness l_i_t of tensor i at timestep t
        # Derive total memory usage
        alive_at_t = defaultdict(lambda: {})
        max_mem_usage = solver.create_integer_var("max_mem_usage")
        for t in range(1, self.num_timesteps + 1):
            mem_usage = 0
            for n, v in node_vars.items():
                if t < asap[n]:
                    continue
                for f in n.fanout:
                    valid_sinks = []
                    for s in f.sinks:
                        if alap[s] < t:
                            continue
                        valid_sinks.append(node_vars[s])
                    if len(valid_sinks) == 0:
                        continue

                    # Tensor i is live at timestep t iff t >= src_i and t <= max(snk_k),
                    # where src_i is the timestep at the the op that generates i is
                    # scheduled and snk_k are the times at which all the ops that
                    # consume i are scheduled.
                    # First encode l1 <=> (t >= src_i)
                    l1 = solver.create_binary_var("l1_" + str(f.name) + "_" + str(t))
                    solver.add_constraint(t - v <= self.num_timesteps * l1 - 1)
                    solver.add_constraint((l1 - 1) * (self.num_timesteps - 1) <= t - v)

                    # Then encode l2 <=> (t <= max(snk_k))
                    max_snk = solver.create_integer_var(
                        "max_snk_" + str(f.name) + "_" + str(t)
                    )
                    for s in valid_sinks:
                        solver.add_constraint(max_snk >= s)
                    l2 = solver.create_binary_var("l2_" + str(f.name) + "_" + str(t))
                    solver.add_constraint(max_snk - t <= self.num_timesteps * l2 - 1)
                    solver.add_constraint(
                        (l2 - 1) * (self.num_timesteps - 1) <= max_snk - t
                    )

                    # Create a boolean variable 'live' to encode whether the
                    # tensor is live, which is true iff l1&l2
                    live = solver.create_binary_var("l_" + str(f.name) + "_" + str(t))
                    solver.add_constraint(live <= l1)
                    solver.add_constraint(live <= l2)
                    solver.add_constraint(l1 + l2 <= 1 + live)
                    mem_usage += live * f.size
                    alive_at_t[t][f] = live

            solver.add_constraint(max_mem_usage >= mem_usage)

        solver.set_objective_function(max_mem_usage, maximize=False)

        # print("PROBLEM STATS = " + str(solver))
        schedule = {}
        result = solver.solve()
        for n, v in node_vars.items():
            schedule[n] = result[v]

        peak_mem_usage = 0
        for t, vars in alive_at_t.items():
            memory_usage = 0
            for e, v in vars.items():
                if result[v] == 1:
                    # print(e.name + " IS LIVE AT T=" + str(t))
                    memory_usage += e.size
            if memory_usage > peak_mem_usage:
                peak_mem_usage = memory_usage

        return (peak_mem_usage, schedule)

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
        min_memory_requirement = 0
        for n in self.graph.nodes.values():
            mem_needed = 0
            for e in n.fanin:
                mem_needed += e.size
            for e in n.fanout:
                mem_needed += e.size
            min_memory_requirement = max(min_memory_requirement, mem_needed)

        if min_memory_requirement > mem_limit:
            raise ValueError(
                "The graph requires at least %d bytes to run (limit: %d)"
                % (min_memory_requirement, mem_limit)
            )
            logging.info(
                "largest working set %d fits within limit %d, proceeding...",
                min_memory_requirement,
                mem_limit,
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

        # Compute the range of times during which each tensor can be alive, based
        # purely on precedence constraints.
        asap = self.ComputeASAPSchedule(schedule_constraints)
        alap = self.ComputeALAPSchedule(schedule_constraints)
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

        solver = ilp_solver.ILPSolver(
            timeout_s=self.timeout, rel_stop=self.rel_stop, solver=self.solver
        )

        # Create 2 new variable for each tensor and timestep: generate and preserve
        generate_vars = defaultdict(lambda: {})
        preserve_vars = defaultdict(lambda: {})
        fetch_vars = defaultdict(lambda: {})

        for e in self.graph.edges.values():
            lb, ub = makespan[e]
            for t in range(lb, ub + 1):
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
                for t in range(lb, ub + 1):
                    if t != ts:
                        solver.add_constraint(generate_vars[e][t] == 0)
                    else:
                        solver.add_constraint(generate_vars[e][t] == 1)

            for snk in e.sinks:
                if snk in schedule_constraints:
                    ts = schedule_constraints[snk]
                    assert ts >= lb
                    assert ts <= ub
                    solver.add_constraint(preserve_vars[e][ts] == 1)

        # Add correctness constraints: we can't preserve data unless it's been
        # generated or preserved at the previous timestep. Also it doesn't make
        # sense to preserve and compute some data at the same timestep
        for e in self.graph.edges.values():
            lb, ub = makespan[e]
            for t in range(lb + 1, ub + 1):
                solver.add_constraint(
                    preserve_vars[e][t]
                    <= preserve_vars[e][t - 1]
                    + generate_vars[e][t - 1]
                    + fetch_vars[e][t - 1]
                )
                solver.add_constraint(
                    preserve_vars[e][t] + generate_vars[e][t] + fetch_vars[e][t] <= 1
                )
            # Purely to help the solver. Todo: improve the encoding to avoid
            # creating these variables in the first place
            solver.add_constraint(preserve_vars[e][lb] == 0)
            solver.add_constraint(fetch_vars[e][lb] == 0)
            solver.add_constraint(fetch_vars[e][lb + 1] == 0)
            for t in range(alap[e.source] + 1, ub + 1):
                solver.add_constraint(generate_vars[e][t] == 0)

            # Purely to help the solver: there is no need to swap the control edges.
            if e.size == 0 or not allow_swaps:
                for t in range(lb + 2, ub + 1):
                    solver.add_constraint(fetch_vars[e][t] == 0)

            # Control edges are always available once they've been triggered.
            if e.size == 0:
                for t in range(lb + 1, ub + 1):
                    solver.add_constraint(
                        preserve_vars[e][t]
                        >= preserve_vars[e][t - 1] + generate_vars[e][t - 1]
                    )

        # Add precedence constraints: we need all the nodes inputs to be
        # available in memory at time t in order to evaluate the node.
        # We also ensure that all the fanouts of a node are generated at the
        # same time. This is only an optimization, since the objective of
        # minimizing the number of computation would take care of that on its on.
        for n in self.graph.nodes.values():
            lb = asap[n]
            ub = alap[n]
            if len(n.fanout) > 1:
                for i in range(1, len(n.fanout)):
                    for t in range(lb, ub + 1):
                        solver.add_constraint(
                            generate_vars[n.fanout[0]][t]
                            == generate_vars[n.fanout[i]][t]
                        )

            for snk in n.fanout:
                for src in n.fanin:
                    # Add precedence constraints
                    for t in range(lb, ub + 1):
                        solver.add_constraint(
                            generate_vars[snk][t]
                            <= preserve_vars[src][t] + fetch_vars[src][t]
                        )

        # We can't swap the data back in unless it's already been generated
        # (and implicitely swapped out instead of being simply discarded).
        # Moreover the data must have been generated at least 2 timesteps
        # before it is fetched back, otherwise it would have been cheaper to
        # simply preserve the data in memory.
        for e in self.graph.edges.values():
            lb, ub = makespan[e]
            ub = min(ub, alap[e.source] + 1)
            previously_generated = generate_vars[e][lb]
            for t in range(lb + 2, ub + 1):
                solver.add_constraint(fetch_vars[e][t] <= previously_generated)
                previously_generated += generate_vars[e][t - 1]

        # Force the generation of each tensor at least once (or exactly once if
        # rematerialization is not allowed)
        for ts in generate_vars.values():
            s = 0
            for v in ts.values():
                s += v
            if allow_rematerialization:
                solver.add_constraint(1 <= s)
            else:
                solver.add_constraint(1 == s)

        # A node needs to consume all its inputs at the same timestep. Handle
        # the case where the node has no fanout below. The case where the node
        # has fanout is already handled.
        for n in self.graph.nodes.values():
            if len(n.fanout) > 0:
                continue
            if len(n.fanin) <= 1:
                continue
            # We need at least one timestep during which all the inputs are live at the same time
            lb, ub = makespan[n.fanin[0]]
            for i in range(1, len(n.fanin)):
                nlb, nub = makespan[n.fanin[i]]
                lb = max(lb, nlb)
                ub = min(ub, nub)
            sum_of_all_live = 0
            for t in range(lb, ub + 1):
                all_live = solver.create_binary_var(
                    "fanin_of_" + n.name + "_live_at_ts" + str(t)
                )
                sum_of_preserve_var = 0
                for f in n.fanin:
                    sum_of_preserve_var += preserve_vars[f][t]
                solver.add_constraint(
                    (all_live - 1) * len(n.fanin) <= sum_of_preserve_var - len(n.fanin)
                )
                solver.add_constraint(
                    sum_of_preserve_var - len(n.fanin) >= (all_live - 1) * len(n.fanin)
                )
                sum_of_all_live += all_live
            solver.add_constraint(sum_of_all_live >= 1)

        # Memory usage at each timestep
        mem_at_timestep = defaultdict(lambda: 0)
        for e, ts in preserve_vars.items():
            for t, v in ts.items():
                if e.size > 0:
                    mem_at_timestep[t] += v * e.size
        for e, ts in generate_vars.items():
            for t, v in ts.items():
                if e.size > 0:
                    mem_at_timestep[t] += v * e.size
        for e, ts in fetch_vars.items():
            for t, v in ts.items():
                if e.size > 0:
                    mem_at_timestep[t] += v * e.size

        if max_spills is None:
            # We need to fit withing the memory budget at each timestep
            for mem_usage in mem_at_timestep.values():
                solver.add_constraint(mem_usage <= mem_limit)
        else:
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
                max_timesteps = ub - lb + 1
                solver.add_constraint(is_spilled * max_timesteps >= s)
                s += is_spilled
                s *= e.size
                total_spills += s
            solver.add_constraint(total_spills <= max_spills)

        if account_for_fragmentation and not defrag:
            # GCD
            tensor_sizes = [t.size for t in self.graph.edges.values() if t.size > 0]
            gcd = self._GCD(tensor_sizes)

            # Track the maximum address that can be used
            max_address = solver.create_integer_var("max_address", lower_bound=0)
            solver.add_constraint(max_address <= mem_limit // gcd)

            total_size = 0
            # Create a new variable for each tensor that tracks the base address
            # of the tensor. If the tensor is of size 0, we force its address to be
            # 0 to help the solver
            addresses = OrderedDict()
            for tensor in self.graph.edges.values():
                if tensor.size > 0:
                    v = solver.create_integer_var(tensor.name, lower_bound=0)
                    addresses[tensor] = v
                    solver.add_constraint(v + tensor.size // gcd <= max_address)
                    total_size += tensor.size // gcd
                else:
                    addresses[tensor] = 0

            solver.add_constraint(max_address <= total_size)

            processed = set()
            for t1, span1 in makespan.items():
                if t1.size == 0:
                    continue
                # Help the solver by providing upper bounds for all the addresses
                solver.add_constraint(addresses[t1] + t1.size // gcd <= total_size)
                for t2, span2 in makespan.items():
                    if t2.size == 0:
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
                    if span1[1] >= span2[0] and span1[0] <= span2[1]:
                        # The spans may overlap: if they do, one of these 2 constraints must hold:
                        # variables[t1] + t1.size <= variables[t2]
                        # variables[t1] >= variables[t2] + t2.size
                        v1 = solver.create_binary_var(t1.name + "_" + t2.name + "_v1")
                        solver.add_constraint(
                            addresses[t1] + t1.size // gcd - addresses[t2]
                            <= (1 - v1) * total_size
                        )
                        v2 = solver.create_binary_var(t1.name + "_" + t2.name + "_v2")
                        solver.add_constraint(
                            addresses[t1] - addresses[t2] - t2.size // gcd
                            >= (v2 - 1) * total_size
                        )

                        # check if they actually do overlap
                        overlap = solver.create_binary_var(
                            t1.name + "_" + t2.name + "_overlap"
                        )
                        # sum_overlaps = 0
                        for ts in range(
                            max(span1[0], span2[0]), min(span1[1], span2[1]) + 1
                        ):
                            live1 = (
                                generate_vars[t1][ts]
                                + preserve_vars[t1][ts]
                                + fetch_vars[t1][ts]
                            )
                            live2 = (
                                generate_vars[t2][ts]
                                + preserve_vars[t2][ts]
                                + fetch_vars[t2][ts]
                            )
                            # overlap_at_t = solver.create_binary_var(
                            #    t1.name + "_" + t2.name + "_overlap_at_" + str(ts)
                            # )
                            # solver.add_constraint(overlap_at_t <= live1)
                            # solver.add_constraint(overlap_at_t <= live2)
                            # solver.add_constraint(overlap_at_t + 1 >= live1 + live2)
                            overlap_at_t = live1 + live2 - 1
                            solver.add_constraint(overlap >= overlap_at_t)
                            # sum_overlaps += overlap_at_t

                        # solver.add_constraint(overlap <= sum_overlaps)

                        solver.add_constraint(v1 + v2 >= overlap)
                        # solver.add_constraint(2 - v1 - v2 >= overlap)

        #####################################################

        elif defrag:
            # GCD
            tensor_sizes = [t.size for t in self.graph.edges.values() if t.size > 0]
            gcd = self._GCD(tensor_sizes)

            # Maximum address that can be used
            max_address = mem_limit // gcd

            # Create a new variable for each tensor that tracks the base address
            # of the tensor. If the tensor is of size 0, we force its address to be
            # 0 to help the solver
            addresses = defaultdict(lambda: {})
            for tensor in self.graph.edges.values():
                lb, ub = makespan[tensor]
                for t in range(lb, ub + 1):
                    if tensor.size > 0:
                        v = solver.create_integer_var(
                            tensor.name + "@" + str(t),
                            lower_bound=0,
                            upper_bound=max_address - tensor.size // gcd,
                        )
                        # solver.add_constraint(v + tensor.size // gcd <= max_address)
                        addresses[tensor][t] = v
                    else:
                        addresses[tensor][t] = 0

            for e in self.graph.edges.values():
                if e.size == 0:
                    continue
                lb, ub = makespan[e]
                for t in range(lb + 1, ub + 1):
                    # If preserve[t], then address[t] must be equal to address[t-1]. This is encoded as:
                    # address[t] - address[t-1] <= (1-preserve) * max_address
                    # address[t-1] - address[t] <= (1-preserve) * max_address
                    solver.add_constraint(
                        addresses[e][t] - addresses[e][t - 1]
                        <= (1 - preserve_vars[e][t]) * max_address
                    )
                    solver.add_constraint(
                        addresses[e][t - 1] - addresses[e][t]
                        <= (1 - preserve_vars[e][t]) * max_address
                    )

            liveness = defaultdict(lambda: [])
            for e in self.graph.edges.values():
                if e.size == 0:
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
                        live1 = (
                            generate_vars[t1][ts]
                            + preserve_vars[t1][ts]
                            + fetch_vars[t1][ts]
                        )
                        live2 = (
                            generate_vars[t2][ts]
                            + preserve_vars[t2][ts]
                            + fetch_vars[t2][ts]
                        )
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
                                addresses[t1][ts] + t1.size // gcd - addresses[t2][ts]
                                <= (1 - v1) * max_address
                            )
                            solver.add_constraint(overlap_at_ts <= v1)
                        else:
                            v1 = solver.create_binary_var(
                                t1.name + "_" + t2.name + "_v1_at_" + str(ts)
                            )
                            solver.add_constraint(
                                addresses[t1][ts] + t1.size // gcd - addresses[t2][ts]
                                <= (1 - v1) * max_address
                            )
                            v2 = solver.create_binary_var(
                                t1.name + "_" + t2.name + "_v2_at_" + str(ts)
                            )
                            solver.add_constraint(
                                addresses[t1][ts] - addresses[t2][ts] - t2.size // gcd
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
                for v in ts.values():
                    s += v * e.size

        if max_spills is None:
            # Minimize the number of spills
            for e, ts in fetch_vars.items():
                for v in ts.values():
                    s += v * e.size * 2
        else:
            # Minimize peak memory usage
            if defrag:
                v = solver.create_integer_var(
                    "peak_memory_usage",
                    lower_bound=min_memory_requirement,
                    upper_bound=mem_limit,
                )
                s += v
                for t, p in addresses.items():
                    for a in p.values():
                        solver.add_constraint(v >= a * gcd + t.size)
            elif account_for_fragmentation:
                v = solver.create_integer_var(
                    "peak_memory_usage",
                    lower_bound=min_memory_requirement,
                    upper_bound=mem_limit,
                )
                s += v
                for t, a in addresses.items():
                    solver.add_constraint(v >= a * gcd + t.size)
            else:
                v = solver.create_integer_var(
                    "peak_memory_usage",
                    lower_bound=min_memory_requirement,
                    upper_bound=mem_limit,
                )
                s += v
                for m in mem_at_timestep.values():
                    solver.add_constraint(v >= m)

        solver.set_objective_function(s, maximize=False)

        start_time = time.time()
        print("Start ILP solve")
        # print("PROBLEM STATS = " + str(solver))
        solver.write("/tmp/scheduler_solver.txt")
        result = solver.solve()
        print(f"ILP solve time: {time.time()-start_time} seconds")

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
        for n, ts in generate_vars.items():
            for t, v in ts.items():
                if result[v] >= 0.99:
                    if account_for_fragmentation:
                        if n.size == 0:
                            schedule[n][0].append(str(t) + "[ctrl]")
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

        for n, ts in preserve_vars.items():
            for t, v in ts.items():
                if t > last_uses[n]:
                    continue
                if result[v] >= 0.99:
                    schedule[n][1].append(t)
                    if n.size == 0:
                        continue
                    if defrag:
                        mem_locations[t][n] = int(result[addresses[n][t]] * gcd)
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
            for t, v in ts.items():
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

        if defrag:
            required_memory = 0
            for t, p in addresses.items():
                if t.size == 0:
                    continue
                for timestamp, a in p.items():
                    if timestamp <= last_uses[e]:
                        required_memory = max(required_memory, result[a] * gcd + t.size)
        elif account_for_fragmentation:
            required_memory = 0
            for t, a in addresses.items():
                if t.size == 0:
                    continue
                required_memory = max(required_memory, result[a] * gcd + t.size)
        else:
            required_memory = peak_mem_usage

        summary = {
            "peak_mem_usage": peak_mem_usage,
            "total_data_swapped": total_data_swapped,
            "required_memory": required_memory,
        }
        return (summary, schedule, mem_locations)
