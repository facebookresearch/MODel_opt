from collections import defaultdict


class Simulator:
    def __init__(self, graph):
        self.graph = graph

    def Simulate(self, node_ordering):
        edge_ref_counts = defaultdict(lambda: 0)

        memory_used = self.graph.unused_weight_size
        peak_memory = 0
        mem_per_timestep = []
        for n in node_ordering:
            for fanout in n.fanout:
                if fanout.size > 0:
                    edge_ref_counts[fanout] = len(fanout.sinks)
                    memory_used += fanout.size

            if memory_used > peak_memory:
                peak_memory = memory_used
            mem_per_timestep.append((n, memory_used))

            for fanin in n.fanin:
                if fanin.size == 0:
                    continue
                edge_ref_counts[fanin] -= 1
                assert edge_ref_counts[fanin] >= 0
                if edge_ref_counts[fanin] == 0:
                    memory_used -= fanin.size

        return (peak_memory, mem_per_timestep)
