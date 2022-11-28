import unittest

from olla import simulator
from olla.native_graphs import (
    control_dep_graph,
    diamond_graph,
    graph_with_two_weights,
    multi_fanin_output_graph,
    pathological_graph,
    shared_multi_fanin_output_graph,
    simple_graph,
)


class SimulatorTest(unittest.TestCase):
    def setUp(self):
        pass

    def testSimpleGraph(self):
        g = simple_graph.graph
        g.unused_weight_size = 0
        print(str(g))
        s = simulator.Simulator(g)
        topo_order = g.compute_topological_ordering()
        peak_mem_usage, mem_per_timestep = s.Simulate(topo_order)
        self.assertEqual(peak_mem_usage, 70)
        mem_per_timestep = [(n.name, mem) for n, mem in mem_per_timestep]
        print(str(mem_per_timestep))
        self.assertEqual(
            mem_per_timestep, [("A", 10), ("B", 60), ("C", 50), ("D", 70), ("E", 40)]
        )

    def testDiamondGraph(self):
        g = diamond_graph.graph
        g.unused_weight_size = 0
        print(str(g))
        s = simulator.Simulator(g)
        topo_order = g.compute_topological_ordering()
        peak_mem_usage, mem_per_timestep = s.Simulate(topo_order)
        self.assertEqual(peak_mem_usage, 150)
        mem_per_timestep = [(n.name, mem) for n, mem in mem_per_timestep]
        print(str(mem_per_timestep))
        self.assertEqual(
            mem_per_timestep,
            [
                ("A", 10),
                ("B", 60),
                ("C", 110),
                ("D", 130),
                ("E", 150),
                ("F", 120),
                ("G", 10),
            ],
        )

    def testControlDepGraph(self):
        g = control_dep_graph.graph
        g.unused_weight_size = 0
        print(str(g))
        s = simulator.Simulator(g)
        topo_order = g.compute_topological_ordering()
        peak_mem_usage, mem_per_timestep = s.Simulate(topo_order)
        self.assertEqual(peak_mem_usage, 150)
        mem_per_timestep = [(n.name, mem) for n, mem in mem_per_timestep]
        print(str(mem_per_timestep))
        self.assertEqual(
            mem_per_timestep,
            [
                ("A", 10),
                ("B", 60),
                ("C", 110),
                ("D", 130),
                ("E", 150),
                ("F", 120),
                ("G", 10),
            ],
        )

    def testMultiFaninOutputGraph(self):
        g = multi_fanin_output_graph.graph
        g.unused_weight_size = 0
        print(str(g))
        s = simulator.Simulator(g)
        topo_order = g.compute_topological_ordering()
        peak_mem_usage, mem_per_timestep = s.Simulate(topo_order)
        self.assertEqual(peak_mem_usage, 120)
        mem_per_timestep = [(n.name, mem) for n, mem in mem_per_timestep]
        print(str(mem_per_timestep))
        self.assertEqual(
            mem_per_timestep, [("A", 30), ("B", 60), ("C", 100), ("D", 120), ("E", 90)]
        )

    def testSharedMultiFaninOutputGraph(self):
        g = shared_multi_fanin_output_graph.graph
        g.unused_weight_size = 0
        print(str(g))
        s = simulator.Simulator(g)
        topo_order = g.compute_topological_ordering()
        peak_mem_usage, mem_per_timestep = s.Simulate(topo_order)
        self.assertEqual(peak_mem_usage, 60)
        mem_per_timestep = [(n.name, mem) for n, mem in mem_per_timestep]
        print(str(mem_per_timestep))
        self.assertEqual(
            mem_per_timestep,
            [("A", 30), ("allocate_3", 60), ("B", 60), ("C", 50), ("D", 30)],
        )

    def testPathologicalCase(self):
        g = pathological_graph.graph
        g.unused_weight_size = 0
        print(str(g))
        s = simulator.Simulator(g)
        topo_order = g.compute_topological_ordering()
        peak_mem_usage, mem_per_timestep = s.Simulate(topo_order)
        self.assertEqual(peak_mem_usage, 42)
        mem_per_timestep = [(n.name, mem) for n, mem in mem_per_timestep]
        print(str(mem_per_timestep))
        self.assertEqual(
            mem_per_timestep,
            [
                ("M0", 1),
                ("M1", 10),
                ("M2", 11),
                ("M3", 19),
                ("M4", 20),
                ("M5", 27),
                ("M6", 35),
                ("M7", 42),
                ("F0", 42),
                ("F1", 41),
                ("F2", 32),
                ("F3", 31),
                ("F4", 23),
                ("F5", 22),
                ("F6", 15),
                ("F7", 7),
            ],
        )

    def testGraphWithTwoWeights(self):
        g = graph_with_two_weights.graph
        g.unused_weight_size = 0
        print(str(g))
        s = simulator.Simulator(g)
        topo_order = g.compute_topological_ordering()
        peak_mem_usage, mem_per_timestep = s.Simulate(topo_order)
        self.assertEqual(peak_mem_usage, 524)
        mem_per_timestep = [(n.name, mem) for n, mem in mem_per_timestep]
        print(str(mem_per_timestep))
        self.assertEqual(
            mem_per_timestep,
            [
                ("WEIGHT1", 123),
                ("WEIGHT2", 444),
                ("INPUT", 454),
                ("OPERATOR1", 484),
                ("OPERATOR2", 524),
                ("OUTPUT", 494),
                ("WEIGHT1_snk", 444),
                ("WEIGHT2_snk", 321),
            ],
        )
