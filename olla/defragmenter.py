from collections import OrderedDict

from olla import ilp_solver


# Optimize a memory layout to limit fragmentation.
class Defragmenter:
    # Given a set of tensors which are alive in the timerange [s, e], figure
    # out the base address of each tensor that minimizes the peak total memory
    # usage. The span argument is expected to be a dictionary of tuples indexed
    # by tensors, i.e. dataflow_graph.Edge {tensor: (start_time, end_time)}
    # Returns a map of {tensor: base address}
    def ComputeBestLayout(self, spans):
        solver = ilp_solver.ILPSolver()

        # Track the maximum address that can be used
        max_address = solver.create_integer_var("max_address", lower_bound=0)

        total_size = 0
        # Create a new variable for each tensor that tracks the base address
        # of the tensor
        variables = OrderedDict()
        for tensor in spans.keys():
            v = solver.create_integer_var(tensor.name, lower_bound=0)
            variables[tensor] = v
            solver.add_constraint(v + tensor.size <= max_address)
            total_size += tensor.size

        solver.add_constraint(max_address <= total_size)

        processed = set()
        for t1, span1 in spans.items():
            # Help the solver by providing upper bounds for all the addresses
            solver.add_constraint(variables[t1] + t1.size <= total_size)
            for t2, span2 in spans.items():
                if t1 is t2:
                    continue
                if (t2, t1) in processed:
                    continue
                processed.add((t1, t2))
                if span1[1] >= span2[0] and span1[0] <= span2[1]:
                    # The spans overlap: one of these 2 constraints must hold:
                    # variables[t1] + t1.size <= variables[t2]
                    # variables[t1] >= variables[t2] + t2.size
                    v1 = solver.create_binary_var(t1.name + "_" + t2.name + "_v1")
                    solver.add_constraint(
                        variables[t1] + t1.size - variables[t2] <= (1 - v1) * total_size
                    )
                    v2 = solver.create_binary_var(t1.name + "_" + t2.name + "_v2")
                    solver.add_constraint(
                        variables[t1] - variables[t2] - t2.size >= (v2 - 1) * total_size
                    )
                    solver.add_constraint(v1 + v2 >= 1)
                    solver.add_constraint(2 - v1 - v2 >= 1)

        solver.set_objective_function(max_address, maximize=False)

        result = solver.solve()

        # Return a map of {tensor: base address}
        base_addresses = {}
        for tensor in spans.keys():
            base_addresses[tensor] = result[variables[tensor]]

        return base_addresses
