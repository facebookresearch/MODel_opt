import unittest
import os

from olla import training_graph_optimizer, utils
from olla.native_graphs import graph_with_gradients


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    @unittest.skipIf(not bool(os.getenv('RUN_SKIPPED', 0)), "Temporarily disabled; TODO: @melhoushi")
    def testGraphWithGradients(self):
        g = graph_with_gradients.graph

        self.assertTrue(g.check_consistency())
        self.assertTrue(g.is_valid())

        dot = g.dump()
        print(dot)
        # g.dump("/tmp/graph_with_gradient_gv", format="png")

        s = training_graph_optimizer.Scheduler(g)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            mem_limit=1209,
            allow_swaps=True,
            max_spills=None,
        )

        self.assertEqual(summary["peak_mem_usage"], 1209)
        self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
        self.assertEqual(summary["total_data_swapped"], 1824)

        self.assertTrue(utils.validate_address_allocation(mem_loc))
        self.assertTrue(utils.validate_timeline(schedule))

        s = [n.name + ": " + str(s) + " " for n, s in schedule.items()]
        print(str(s))
        self.assertEqual(
            s,
            [
                "ACTIVATION1: ([3], [4, 5, 6], []) ",
                "WEIGHT_REF1: ([3], [4, 5, 8], [7]) ",
                "ACTIVATION2: ([4], [5, 6, 7, 8, 9], []) ",
                "WEIGHT_REF2: ([1], [2, 3, 4, 5, 11, 12], [10]) ",
                "OUTPUT_EDGE: ([5], [], []) ",
                "PROPAGATE_G1: ([1], [2], [9]) ",
                "PROPAGATE_G2: ([2], [3, 4, 5, 6], []) ",
                "UPDATE_W1: ([6], [7], []) ",
                "UPDATE_W2: ([9], [10, 11, 12], []) ",
            ],
        )
