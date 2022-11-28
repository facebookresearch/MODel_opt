import unittest
from collections import defaultdict

from olla import training_graph_optimizer, utils
from olla.native_graphs import (
    diamond_graph,
    graph_with_bmmadd,
    graph_with_constants,
    graph_with_two_weights,
    graph_with_weights,
)


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        pass

    def testTimestepFactor(self):
        g = diamond_graph.graph
        s = training_graph_optimizer.Scheduler(g, timestep_factor=0.1)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            mem_limit=1024,
            allow_swaps=True,
            max_spills=None,
        )
        self.assertTrue(utils.validate_address_allocation(mem_loc))
        self.assertTrue(utils.validate_timeline(schedule))
        self.assertTrue(utils.validate_node_ordering(g, schedule))

    def testGraphWithConstants(self):
        g = graph_with_constants.graph

        s = training_graph_optimizer.Scheduler(g)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            mem_limit=1024,
            allow_swaps=True,
            max_spills=None,
        )

        self.assertEqual(summary["peak_mem_usage"], 93)
        self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
        self.assertEqual(summary["total_data_swapped"], 0)

        self.assertTrue(utils.validate_address_allocation(mem_loc))
        self.assertTrue(utils.validate_timeline(schedule))
        self.assertTrue(utils.validate_node_ordering(g, schedule))

        s = [n.name + ": " + str(s) + " " for n, s in schedule.items()]
        print(str(s))
        self.assertEqual(
            s,
            [
                "ACTIVATION_EDGE: ([1], [2], []) ",
                "CONSTANT_EDGE: ([1], [2, 3], []) ",
                "OUTPUT_EDGE: ([2], [], []) ",
            ],
        )

    def testGraphWithWeights(self):
        g = graph_with_weights.graph

        s = training_graph_optimizer.Scheduler(g)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            mem_limit=1024,
            allow_swaps=True,
            max_spills=None,
        )

        self.assertEqual(summary["peak_mem_usage"], 183)
        self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
        self.assertEqual(summary["total_data_swapped"], 0)

        self.assertTrue(utils.validate_address_allocation(mem_loc))
        self.assertTrue(utils.validate_timeline(schedule))
        self.assertTrue(utils.validate_node_ordering(g, schedule))

        s = [n.name + ": " + str(s) + " " for n, s in schedule.items()]
        print(str(s))
        self.assertEqual(
            s,
            [
                "ACTIVATION_EDGE: ([1], [2], []) ",
                "WEIGHT_EDGE: ([1], [2, 3], []) ",
                "OUTPUT_EDGE: ([2], [], []) ",
            ],
        )

    def testNonTrivialGraph(self):
        g = graph_with_two_weights.graph

        s = training_graph_optimizer.Scheduler(g)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            mem_limit=1024,
            allow_swaps=True,
            max_spills=None,
        )

        self.assertEqual(summary["peak_mem_usage"], 524)
        self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
        self.assertEqual(summary["total_data_swapped"], 0)

        self.assertTrue(utils.validate_address_allocation(mem_loc))
        self.assertTrue(utils.validate_timeline(schedule))
        self.assertTrue(utils.validate_node_ordering(g, schedule))

        s = [n.name + ": " + str(s) + " " for n, s in schedule.items()]
        print(str(s))
        self.assertEqual(
            s,
            [
                "ACTIVATION1: ([1], [2], []) ",
                "WEIGHT_REF1: ([1], [2, 3, 4], []) ",
                "ACTIVATION2: ([2], [3], []) ",
                "WEIGHT_REF2: ([1], [2, 3, 4], []) ",
                "OUTPUT_EDGE: ([3], [], []) ",
            ],
        )

    def testMemoryPressure(self):
        g = graph_with_two_weights.graph

        s = training_graph_optimizer.Scheduler(g)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            mem_limit=500,
            allow_swaps=True,
            max_spills=None,
        )

        self.assertEqual(summary["peak_mem_usage"], 484)
        self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
        self.assertEqual(summary["total_data_swapped"], 246)

        self.assertTrue(utils.validate_address_allocation(mem_loc))
        self.assertTrue(utils.validate_timeline(schedule))
        self.assertTrue(utils.validate_node_ordering(g, schedule))

        s = [n.name + ": " + str(s) + " " for n, s in schedule.items()]
        print(str(s))
        self.assertEqual(
            s,
            [
                "ACTIVATION1: ([1], [2], []) ",
                "WEIGHT_REF1: ([1], [2], []) ",
                "ACTIVATION2: ([2], [3], []) ",
                "WEIGHT_REF2: ([1], [2, 3, 4], []) ",
                "OUTPUT_EDGE: ([3], [], []) ",
            ],
        )

    def testTimestepIterators(self):
        g = graph_with_two_weights.graph

        s = training_graph_optimizer.Scheduler(g, timestep_factor=1)
        asap = s.ComputeASAPSchedule({})
        alap = s.ComputeALAPSchedule({}, max_timesteps=s.num_nodes)
        makespan = s.ComputeMakespans(asap, alap)

        self.assertEqual(
            str(makespan),
            """{ACTIVATION1: INPUT (None) -> [OPERATOR1 (None)] size=10: (1, 2), WEIGHT_REF1: WEIGHT1 (stateful_node) -> [OPERATOR1 (None), WEIGHT1_snk (stateful_node_sink)] size=123: (1, 4), WEIGHT_REF2: WEIGHT2 (stateful_node) -> [OPERATOR2 (None), WEIGHT2_snk (stateful_node_sink)] size=321: (1, 4), ACTIVATION2: OPERATOR1 (None) -> [OPERATOR2 (None)] size=30: (2, 3), OUTPUT_EDGE: OPERATOR2 (None) -> [OUTPUT (None)] size=50: (3, 4)}""",
        )

        spans = (makespan, asap, alap)
        timesteps = defaultdict(lambda: [])
        timesteps_with_start_offset = defaultdict(lambda: [])
        for e in g.edges.values():
            if not e.is_stateful():
                continue
            for t in s.TimeStepsForEdge(e, spans):
                timesteps[str(e)].append(t)
            for t in s.TimeStepsForEdge(e, spans, startoffset=1):
                timesteps_with_start_offset[str(e)].append(t)

        print(str(timesteps))
        self.assertEqual(
            timesteps,
            {
                "WEIGHT_REF1: WEIGHT1 (stateful_node) -> [OPERATOR1 (None), WEIGHT1_snk (stateful_node_sink)] size=123": [
                    1,
                    2,
                    4,
                ],
                "WEIGHT_REF2: WEIGHT2 (stateful_node) -> [OPERATOR2 (None), WEIGHT2_snk (stateful_node_sink)] size=321": [
                    1,
                    2,
                    3,
                    4,
                ],
            },
        )

        print(str(timesteps_with_start_offset))
        self.assertEqual(
            timesteps_with_start_offset,
            {
                "WEIGHT_REF1: WEIGHT1 (stateful_node) -> [OPERATOR1 (None), WEIGHT1_snk (stateful_node_sink)] size=123": [
                    2,
                    4,
                ],
                "WEIGHT_REF2: WEIGHT2 (stateful_node) -> [OPERATOR2 (None), WEIGHT2_snk (stateful_node_sink)] size=321": [
                    2,
                    3,
                    4,
                ],
            },
        )

        timesteps = defaultdict(lambda: [])
        for n in g.nodes.values():
            for t in s.TimeStepsForFanin(n, spans):
                timesteps[n.name].append(t)
        print(str(timesteps))
        self.assertEqual(
            timesteps,
            {
                "OPERATOR1": [1, 2],
                "OPERATOR2": [2, 3],
                "OUTPUT": [3, 4],
                "WEIGHT1_snk": [1, 2, 4],
                "WEIGHT2_snk": [1, 2, 3, 4],
            },
        )

    def testAdvancedTimestepIterators(self):
        g = graph_with_bmmadd.graph

        s = training_graph_optimizer.Scheduler(g, timestep_factor=1)
        asap = s.ComputeASAPSchedule({})
        alap = s.ComputeALAPSchedule({}, max_timesteps=s.num_nodes)
        makespan = s.ComputeMakespans(asap, alap)

        # print(str(makespan))
        self.assertEqual(
            str(makespan),
            """{ACTIVATION1: INPUT (None) -> [OPERATOR1 (None)] size=20: (1, 5), ACTIVATION7: INPUT2 (None) -> [OPERATOR6 (None)] size=110: (1, 5), WEIGHT_REF1: WEIGHT (stateful_node) -> [BMMADD (None), WEIGHT_snk (stateful_node_sink)] size=193: (1, 11), TRANSPOSE: BIAS (stateful_node) -> [TRANSPOSE (None), BIAS_snk (stateful_node_sink)] size=331: (1, 11), WEIGHT_TRANSPOSED: TRANSPOSE (None) -> [BMMADD (None)] size=60: (2, 8), ACTIVATION2: OPERATOR1 (None) -> [OPERATOR2 (None)] size=30: (2, 6), ACTIVATION3: OPERATOR2 (None) -> [OPERATOR3 (None)] size=40: (3, 7), ACTIVATION4: OPERATOR3 (None) -> [BMMADD (None)] size=50: (4, 8), ACTIVATION6: OPERATOR4 (None) -> [OPERATOR5 (None)] size=90: (6, 10), OUTPUT_EDGE: OPERATOR5 (None) -> [OUTPUT (None)] size=100: (7, 11), ACTIVATION8: OPERATOR6 (None) -> [OPERATOR2 (None)] size=120: (2, 6), ACTIVATION5: BMMADD (None) -> [OPERATOR4 (None)] size=80: (5, 9)}""",
        )

        spans = (makespan, asap, alap)
        timesteps = defaultdict(lambda: [])
        timesteps_with_start_offset = defaultdict(lambda: [])
        for e in g.edges.values():
            for t in s.TimeStepsForEdge(e, spans):
                timesteps[str(e)].append(t)
            for t in s.TimeStepsForEdge(e, spans, startoffset=1):
                timesteps_with_start_offset[str(e)].append(t)

        print(str(timesteps))
        self.assertEqual(
            timesteps,
            {
                "TRANSPOSE: BIAS (stateful_node) -> [TRANSPOSE (None), BIAS_snk (stateful_node_sink)] size=331": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    11,
                ],
                "ACTIVATION1: INPUT (None) -> [OPERATOR1 (None)] size=20": [
                    1,
                    2,
                    3,
                    4,
                    5,
                ],
                "ACTIVATION2: OPERATOR1 (None) -> [OPERATOR2 (None)] size=30": [
                    2,
                    3,
                    4,
                    5,
                    6,
                ],
                "ACTIVATION3: OPERATOR2 (None) -> [OPERATOR3 (None)] size=40": [
                    3,
                    4,
                    5,
                    6,
                    7,
                ],
                "ACTIVATION4: OPERATOR3 (None) -> [BMMADD (None)] size=50": [
                    4,
                    5,
                    6,
                    7,
                    8,
                ],
                "WEIGHT_TRANSPOSED: TRANSPOSE (None) -> [BMMADD (None)] size=60": [
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                ],
                "WEIGHT_REF1: WEIGHT (stateful_node) -> [BMMADD (None), WEIGHT_snk (stateful_node_sink)] size=193": [
                    1,
                    4,
                    5,
                    6,
                    7,
                    8,
                    11,
                ],
                "ACTIVATION5: BMMADD (None) -> [OPERATOR4 (None)] size=80": [
                    5,
                    6,
                    7,
                    8,
                    9,
                ],
                "ACTIVATION6: OPERATOR4 (None) -> [OPERATOR5 (None)] size=90": [
                    6,
                    7,
                    8,
                    9,
                    10,
                ],
                "OUTPUT_EDGE: OPERATOR5 (None) -> [OUTPUT (None)] size=100": [
                    7,
                    8,
                    9,
                    10,
                    11,
                ],
                "ACTIVATION7: INPUT2 (None) -> [OPERATOR6 (None)] size=110": [
                    1,
                    2,
                    3,
                    4,
                    5,
                ],
                "ACTIVATION8: OPERATOR6 (None) -> [OPERATOR2 (None)] size=120": [
                    2,
                    3,
                    4,
                    5,
                    6,
                ],
            },
        )

        print(str(timesteps_with_start_offset))
        self.assertEqual(
            timesteps_with_start_offset,
            {
                "TRANSPOSE: BIAS (stateful_node) -> [TRANSPOSE (None), BIAS_snk (stateful_node_sink)] size=331": [
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    11,
                ],
                "ACTIVATION1: INPUT (None) -> [OPERATOR1 (None)] size=20": [2, 3, 4, 5],
                "ACTIVATION2: OPERATOR1 (None) -> [OPERATOR2 (None)] size=30": [
                    3,
                    4,
                    5,
                    6,
                ],
                "ACTIVATION3: OPERATOR2 (None) -> [OPERATOR3 (None)] size=40": [
                    4,
                    5,
                    6,
                    7,
                ],
                "ACTIVATION4: OPERATOR3 (None) -> [BMMADD (None)] size=50": [
                    5,
                    6,
                    7,
                    8,
                ],
                "WEIGHT_TRANSPOSED: TRANSPOSE (None) -> [BMMADD (None)] size=60": [
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                ],
                "WEIGHT_REF1: WEIGHT (stateful_node) -> [BMMADD (None), WEIGHT_snk (stateful_node_sink)] size=193": [
                    4,
                    5,
                    6,
                    7,
                    8,
                    11,
                ],
                "ACTIVATION5: BMMADD (None) -> [OPERATOR4 (None)] size=80": [
                    6,
                    7,
                    8,
                    9,
                ],
                "ACTIVATION6: OPERATOR4 (None) -> [OPERATOR5 (None)] size=90": [
                    7,
                    8,
                    9,
                    10,
                ],
                "OUTPUT_EDGE: OPERATOR5 (None) -> [OUTPUT (None)] size=100": [
                    8,
                    9,
                    10,
                    11,
                ],
                "ACTIVATION7: INPUT2 (None) -> [OPERATOR6 (None)] size=110": [
                    2,
                    3,
                    4,
                    5,
                ],
                "ACTIVATION8: OPERATOR6 (None) -> [OPERATOR2 (None)] size=120": [
                    3,
                    4,
                    5,
                    6,
                ],
            },
        )

        timesteps = defaultdict(lambda: [])
        for n in g.nodes.values():
            for t in s.TimeStepsForFanin(n, spans):
                timesteps[n.name].append(t)

        print(str(timesteps))
        self.assertEqual(
            timesteps,
            {
                "TRANSPOSE": [1, 2, 3, 4, 5, 6, 7, 11],
                "OPERATOR1": [1, 2, 3, 4, 5],
                "OPERATOR2": [2, 3, 4, 5, 6],
                "OPERATOR3": [3, 4, 5, 6, 7],
                "OPERATOR4": [5, 6, 7, 8, 9],
                "OPERATOR5": [6, 7, 8, 9, 10],
                "OPERATOR6": [1, 2, 3, 4, 5],
                "BMMADD": [4, 5, 6, 7, 8],
                "OUTPUT": [7, 8, 9, 10, 11],
                "WEIGHT_snk": [1, 4, 5, 6, 7, 8, 11],
                "BIAS_snk": [1, 2, 3, 4, 5, 6, 7, 11],
            },
        )

        timesteps = defaultdict(lambda: [])
        for n in g.nodes.values():
            if n.is_stateful():
                continue
            for t in s.TimeStepsForNode(n, spans):
                timesteps[n.name].append(t)

        print(str(timesteps))
        self.assertEqual(
            timesteps,
            {
                "INPUT": [1, 2, 3, 4],
                "INPUT2": [1, 2, 3, 4],
                "TRANSPOSE": [2, 3, 4, 5, 6, 7],
                "OPERATOR1": [2, 3, 4, 5],
                "OPERATOR2": [3, 4, 5, 6],
                "OPERATOR3": [4, 5, 6, 7],
                "OPERATOR4": [6, 7, 8, 9],
                "OPERATOR5": [7, 8, 9, 10],
                "OPERATOR6": [2, 3, 4, 5],
                "BMMADD": [5, 6, 7, 8],
                "OUTPUT": [8, 9, 10, 11],
            },
        )

    def testDensePreserveVarsMap(self):
        sparse_ts = {1: "A", 2: "B", 5: "C", 6: "D", 8: "E"}
        dense_map = training_graph_optimizer.Scheduler.DensePreserveVarsMap(sparse_ts)

        dense_ts = {}
        for i in range(1, 9):
            dense_ts[i] = dense_map[i]
        print(str(dense_ts))
        self.assertEqual(
            dense_ts, {1: "A", 2: "B", 3: "C", 4: "C", 5: "C", 6: "D", 7: "E", 8: "E"}
        )

    def testDenseGenerateOrFetchVarsMap(self):
        sparse_ts = {1: "A", 2: "B", 5: "C", 6: "D", 8: "E"}
        dense_map = training_graph_optimizer.Scheduler.DenseGenerateOrFetchVarsMap(
            sparse_ts
        )

        dense_ts = {}
        for i in range(1, 9):
            dense_ts[i] = dense_map[i]
        print(str(dense_ts))
        self.assertEqual(
            dense_ts, {1: "A", 2: "B", 3: 0, 4: 0, 5: "C", 6: "D", 7: 0, 8: "E"}
        )
