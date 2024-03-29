
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from model_opt import memory_planner
from model_opt.native_graphs import (
    control_dep_graph,
    diamond_graph,
    multi_fanin_output_graph,
    shared_multi_fanin_output_graph,
    simple_graph,
)


class MemoryPlannerTest(unittest.TestCase):
    def setUp(self):
        pass

    def testNonFragmentation(self):
        g = simple_graph.graph
        p = memory_planner.MemoryPlanner()
        summary = p.plan(g, mem_limit=1024)

        self.assertEqual(summary["peak_mem_usage"], 70)
        self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
        self.assertEqual(summary["total_data_swapped"], 0)

    def testSimpleGraph(self):
        g = simple_graph.graph

        p = memory_planner.MemoryPlanner()
        summary = p.plan(g, mem_limit=1024)

        self.assertEqual(summary["peak_mem_usage"], 70)
        self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
        self.assertEqual(summary["total_data_swapped"], 0)

    def testDiamondGraph(self):
        g = diamond_graph.graph

        p = memory_planner.MemoryPlanner()
        summary = p.plan(g, mem_limit=1024)

        self.assertEqual(summary["peak_mem_usage"], 150)
        self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
        self.assertEqual(summary["total_data_swapped"], 0)

    def testControlDepGraph(self):
        g = control_dep_graph.graph

        p = memory_planner.MemoryPlanner()
        summary = p.plan(g, mem_limit=1024)

        self.assertEqual(summary["peak_mem_usage"], 150)
        self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
        self.assertEqual(summary["total_data_swapped"], 0)

    def testControlDepGraphWithSpills(self):
        g = control_dep_graph.graph

        p = memory_planner.MemoryPlanner()
        summary = p.plan(g, mem_limit=120)

        self.assertEqual(summary["peak_mem_usage"], 120)
        self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
        self.assertEqual(summary["total_data_swapped"], 120)

    def testMultiFaninOutputGraph(self):
        g = multi_fanin_output_graph.graph

        p = memory_planner.MemoryPlanner()
        summary = p.plan(g, mem_limit=110)

        self.assertEqual(summary["peak_mem_usage"], 110)
        self.assertEqual(summary["total_data_swapped"], 0)

    def testSharedMultiFaninOutputGraph(self):
        g = shared_multi_fanin_output_graph.graph

        p = memory_planner.MemoryPlanner()
        summary = p.plan(g, mem_limit=1024)

        self.assertEqual(summary["peak_mem_usage"], 60)
        self.assertEqual(summary["total_data_swapped"], 0)
