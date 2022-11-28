from olla import ilp_solver


class MaxCut:
    # If weighted is True, each edge is weigthed by its size, otherwise each
    # edge is assigned the weight of one.
    # The weighted max cut will give us the peak memory usage, while the
    # unweighted max cut will give us the maximum number of tensors that can be
    # live at the same time.
    def __init__(
        self,
        graph,
        weighted=True,
        skip_weights=True,
        debug=False,
        rel_stop=None,
    ):
        assert graph.canonical

        self.graph = graph
        self.weighted = weighted
        self.skip_weights = skip_weights
        self.debug = debug
        self.rel_stop = rel_stop

    def LocateCut(self, user_schedule=None):
        # Partition the graph in 2 subgraphs S and T. For each node in a the
        # corresponding boolean variable will take value 1, while the variable
        # will take value 0 for the nodes in T
        # Also create a variable for each edge, that will take value 1 for the
        # edges alongside the cut between S and T and o for all the other edges.
        solver = ilp_solver.ILPSolver(solver="GUROBI", rel_stop=self.rel_stop)

        # Create a new variable for each node
        node_vars = {}
        for n in self.graph.nodes.values():
            if self.skip_weights and n.is_stateful():
                continue
            v = solver.create_binary_var(n.name)
            node_vars[n] = v
            # Help the search by forcing the value of the graph inputs and outputs
            if len(n.fanin) == 0:
                solver.add_constraint(v == 1)
            elif len(n.fanout) == 0:
                solver.add_constraint(v == 0)

        # Create a new variable for each edge and add constraints
        edge_vars = {}
        for e in self.graph.edges.values():
            if self.skip_weights and e.is_stateful():
                continue
            src = node_vars[e.source]

            if e.size == 0:
                for s in e.sinks:
                    snk = node_vars[s]
                    # Add precedence constraints
                    solver.add_constraint(src >= snk)
                continue

            v = solver.create_binary_var("e_" + str(e.name))
            edge_vars[e] = v
            sum_snks = 0
            for s in e.sinks:
                snk = node_vars[s]
                sum_snks += snk
                # Add precedence constraints
                solver.add_constraint(src >= snk)
                # If the source and at least one sink are on different
                # partitions, then the edge is on the max cut.
                solver.add_constraint(v >= src - snk)

            # If the source and all the sinks are in partition zero, then the
            # edge is not on the max cut
            solver.add_constraint(v <= src + sum_snks)

            # If the source and all the sinks are in partition one, then the
            # edge is not on the max cut
            solver.add_constraint(v <= 1 + len(e.sinks) - src - sum_snks)

        # Add constraints coming from user schedule
        if user_schedule:
            ordered = []
            for node, order in user_schedule.items():
                ordered.append((order, node))
            ordered.sort()
            for i in range(1, len(ordered)):
                src = ordered[i - 1][1]
                snk = ordered[1][1]
                print(f"Adding edge from {src.name} to {snk.name}")
                solver.add_constraint(node_vars[src] >= node_vars[snk])

        # The max cut size is the sum of all the sizes of the edges on the cut
        max_cut_size = 0
        for e, v in edge_vars.items():
            if e.size == 0:
                continue
            cut_size = e.size if self.weighted else 1
            max_cut_size += v * cut_size

        # To get the total memory usage, we need to add the size of all the
        # tensors t in the immediate fanin of the max cut since these tensors
        # remain in memory while the corresponding ops are run.
        fanout_on_cut = {}
        for e in edge_vars.keys():
            if e.size == 0 or len(e.sinks) == 0:
                continue
            d = solver.create_binary_var("is_fanout_of_" + e.name + "_on_max_cut")
            fanout_on_cut[e] = d
            edge_size = e.size if self.weighted else 1
            max_cut_size += d * edge_size
            active_fanouts = 0
            for edge_sinks in e.sinks:
                for snk in edge_sinks.fanout:
                    if snk.size == 0:
                        continue
                    if self.skip_weights and snk.is_stateful():
                        continue
                    is_e_on_max_cut = edge_vars[snk]
                    active_fanouts += is_e_on_max_cut
                    solver.add_constraint(d >= is_e_on_max_cut)

            solver.add_constraint(d <= active_fanouts)

        solver.set_objective_function(max_cut_size)

        max_cut = []
        live_tensors = set()
        result = solver.solve()
        for e, v in edge_vars.items():
            if result[v] >= 0.99:
                max_cut.append(e.source)
                live_tensors.add(e)
                for t in e.source.fanin:
                    live_tensors.add(t)

        cut_size = 0
        for t in live_tensors:
            if t.size == 0:
                continue
            cut_size += t.size if self.weighted else 1

        if self.debug:
            print(str(result))

            for e in self.graph.edges.values():
                if self.skip_weights and e.is_stateful():
                    continue
                if e.size == 0:
                    continue

                if result[edge_vars[e]] >= 0.99:
                    assert e in live_tensors
                    assert result[node_vars[e.source]] >= 0.99
                    found_snk = False
                    for snk in e.sinks:
                        if result[node_vars[snk]] <= 0.01:
                            found_snk = True
                            break
                    if not found_snk:
                        print(f"Invalid edge {e.name}")
                        print(f"  SRC {e.source.name} = {result[node_vars[e.source]]}")
                        for snk in e.sinks:
                            print(f"  SNK {snk.name} = {result[node_vars[snk]]}")
                    assert found_snk

                else:
                    for snk in e.sinks:
                        if result[node_vars[snk]] <= 0.5:
                            assert result[node_vars[e.source]] <= 0.5
                        if result[node_vars[snk]] > 0.5:
                            assert result[node_vars[e.source]] > 0.5

            max_cut_size = 0
            for e, v in edge_vars.items():
                if e.size == 0:
                    continue
                edge_size = e.size if self.weighted else 1
                if result[v] >= 0.99:
                    print(f"Edge {e.name} adds {edge_size} to the max cut")
                    max_cut_size += edge_size

            for e in edge_vars.keys():
                if e.size == 0 or len(e.sinks) == 0:
                    continue
                edge_size = e.size if self.weighted else 1
                if result[fanout_on_cut[e]]:
                    print(
                        f"Edge {e.name} adds {edge_size} to the max cut since its fanout is on maxcut"
                    )
                    max_cut_size += edge_size

            print(f"Max cut size is {max_cut_size}")

        return (cut_size, max_cut)
