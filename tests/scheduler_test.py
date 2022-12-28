import unittest
import os

from olla import scheduler, utils
from olla.native_graphs import (
    control_dep_graph,
    diamond_graph,
    multi_fanin_output_graph,
    pathological_graph,
    shared_multi_fanin_output_graph,
    simple_graph,
)


class SchedulerTest(unittest.TestCase):
    def testNonFragmentation(self):
        g = simple_graph.graph

        s = scheduler.Scheduler(g)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            mem_limit=1024,
            allow_swaps=True,
            max_spills=0,
        )

        self.assertEqual(summary["peak_mem_usage"], 70)
        self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
        self.assertEqual(summary["total_data_swapped"], 0)

        self.assertTrue(utils.validate_address_allocation(mem_loc))
        self.assertTrue(utils.validate_timeline(schedule))

        s = [n.name + ": " + str(s) + " " for n, s in schedule.items()]
        print(str(s))
        self.assertEqual(
            s,
            [
                "1: ([1], [2], []) ",
                "2: ([2], [], []) ",
                "3: ([2], [3, 4], []) ",
                "4: ([4], [], []) ",
            ],
        )

    def testSimpleGraph(self):
        g = simple_graph.graph

        s = scheduler.Scheduler(g)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            mem_limit=1024,
            allow_swaps=True,
            account_for_fragmentation=True,
            defrag=True,
            max_spills=0,
        )

        self.assertEqual(summary["peak_mem_usage"], 70)
        self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
        self.assertEqual(summary["total_data_swapped"], 0)

        self.assertTrue(utils.validate_address_allocation(mem_loc))
        self.assertTrue(utils.validate_timeline(schedule))

        s = [n.name + ": " + str(s) + " " for n, s in schedule.items()]
        print(str(s))
        self.assertEqual(
            s,
            [
                "1: (['2@0'], [3], []) ",
                "2: (['3@10'], [], []) ",
                "3: (['3@40'], [4], []) ",
                "4: (['4@0'], [], []) ",
            ],
        )

    def testDiamondGraph(self):
        g = diamond_graph.graph

        s = scheduler.Scheduler(g)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            mem_limit=1024,
            allow_swaps=True,
            account_for_fragmentation=True,
            defrag=True,
            max_spills=0,
        )

        self.assertEqual(summary["peak_mem_usage"], 130)
        self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
        self.assertEqual(summary["total_data_swapped"], 0)

        self.assertTrue(utils.validate_address_allocation(mem_loc))
        self.assertTrue(utils.validate_timeline(schedule))

        s = [n.name + ": " + str(s) + " " for n, s in schedule.items()]
        print(str(s))
        self.assertEqual(
            s,
            [
                "1: (['1@20'], [2], []) ",
                "2: (['2@0'], [3, 4, 5], []) ",
                "3: (['2@60'], [3], []) ",
                "4: (['3@20'], [4], []) ",
                "5: (['4@80'], [5, 6], []) ",
                "6: (['5@20'], [6], []) ",
                "7: (['6@10'], [], []) ",
            ],
        )

    def testControlDepGraph(self):
        g = control_dep_graph.graph

        s = scheduler.Scheduler(g)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            mem_limit=1024,
            allow_swaps=True,
            account_for_fragmentation=True,
            defrag=True,
            max_spills=0,
        )

        self.assertEqual(summary["peak_mem_usage"], 150)
        self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
        self.assertEqual(summary["total_data_swapped"], 0)

        self.assertTrue(utils.validate_address_allocation(mem_loc))
        self.assertTrue(utils.validate_timeline(schedule))

        s = [n.name + ": " + str(s) + " " for n, s in schedule.items()]
        print(str(s))
        self.assertEqual(
            s,
            [
                "1: (['1@30'], [2], []) ",
                "2: (['2@130'], [3], []) ",
                "3: (['2@0'], [3, 4], []) ",
                "4: (['4@110'], [5], []) ",
                "5: (['5@0'], [6], []) ",
                "6: (['3@50'], [4, 5, 6], []) ",
                "7: (['6@110'], [], []) ",
                "8: (['3[ctrl]'], [4, 5], []) ",
            ],
        )

    def testControlDepGraphWithSpills(self):
        g = control_dep_graph.graph

        s = scheduler.Scheduler(g)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            mem_limit=1024,
            allow_swaps=True,
            account_for_fragmentation=True,
            defrag=True,
            max_spills=180,
        )

        self.assertEqual(summary["peak_mem_usage"], 120)
        self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
        self.assertEqual(summary["total_data_swapped"], 160)

        self.assertTrue(utils.validate_address_allocation(mem_loc))
        self.assertTrue(utils.validate_timeline(schedule))

        s = [n.name + ": " + str(s) + " " for n, s in schedule.items()]
        print(str(s))
        self.assertEqual(
            s,
            [
                "1: (['1@0'], [2], []) ",
                "2: (['2@70'], [], ['4@40']) ",
                "3: (['2@40'], [3], []) ",
                "4: (['3@0'], [4, 5], []) ",
                "5: (['5@70'], [6], []) ",
                "6: (['4@60'], [], ['6@10']) ",
                "7: (['6@0'], [], []) ",
                "8: (['4[ctrl]'], [5], []) ",
            ],
        )

    def testMultiFaninOutputGraph(self):
        g = multi_fanin_output_graph.graph

        s = scheduler.Scheduler(g)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            mem_limit=1024,
            allow_swaps=True,
            account_for_fragmentation=True,
            defrag=True,
            max_spills=0,
        )

        self.assertEqual(summary["peak_mem_usage"], 110)
        self.assertEqual(summary["total_data_swapped"], 0)

        self.assertTrue(utils.validate_address_allocation(mem_loc))
        self.assertTrue(utils.validate_timeline(schedule))

        s = [n.name + ": " + str(s) + " " for n, s in schedule.items()]
        print(str(s))
        self.assertEqual(
            s,
            [
                "1: (['1@0'], [2], []) ",
                "2: (['1@90'], [2, 3, 4], []) ",
                "3: (['2@20'], [3], []) ",
                "4: (['3@50'], [4, 5], []) ",
                "5: (['4@0'], [5], []) ",
            ],
        )

    def testSharedMultiFaninOutputGraph(self):
        g = shared_multi_fanin_output_graph.graph

        s = scheduler.Scheduler(g)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            mem_limit=1024,
            allow_swaps=True,
            account_for_fragmentation=True,
            defrag=True,
            max_spills=0,
        )

        self.assertEqual(summary["peak_mem_usage"], 60)
        self.assertEqual(summary["total_data_swapped"], 0)

        self.assertTrue(utils.validate_address_allocation(mem_loc))
        self.assertTrue(utils.validate_timeline(schedule))

        s = [n.name + ": " + str(s) + " " for n, s in schedule.items()]
        print(str(s))
        self.assertEqual(
            s,
            [
                "1: (['3@0'], [4], []) ",
                "2: (['3@40'], [4], []) ",
                "allocate_3: (['1@10'], [2, 3, 4, 5], []) ",
                "B_3: (['4[ctrl]'], [5], []) ",
                "C_3: (['4[ctrl]'], [5], []) ",
            ],
        )

    def testPathologicalCase(self):
        g = pathological_graph.graph

        user_schedule = {
            "M0": 1,
            "M1": 2,
            "M2": 4,
            "M3": 5,
            "M4": 7,
            "M5": 8,
            "M6": 11,
            "M7": 14,
            "F0": 10,
            "F1": 3,
            "F2": 13,
            "F3": 6,
            "F4": 16,
            "F5": 9,
            "F6": 12,
            "F7": 15,
        }

        s = scheduler.Scheduler(g)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            mem_limit=10,
            allow_swaps=False,
            account_for_fragmentation=True,
            defrag=True,
            user_schedule=user_schedule,
        )

        self.assertEqual(summary["required_memory"], 10)
        self.assertEqual(summary["total_data_swapped"], 0)

        self.assertTrue(utils.validate_address_allocation(mem_loc))
        self.assertTrue(utils.validate_timeline(schedule))

        s = [n.name + ": " + str(s) + " " for n, s in schedule.items()]
        print(str(s))
        self.assertEqual(
            s,
            [
                "T0: (['1@9'], [2, 3, 4, 5, 6, 7, 8, 9, 10], []) ",
                "T1: (['2@0'], [3], []) ",
                "T2: (['4@0'], [5, 6, 7, 8, 9, 10, 11, 12, 13], []) ",
                "T3: (['5@1'], [6], []) ",
                "T4: (['7@1'], [8, 9, 10, 11, 12, 13, 14, 15, 16], []) ",
                "T5: (['8@2'], [9], []) ",
                "T6: (['11@2'], [12], []) ",
                "T7: (['14@2'], [15], []) ",
            ],
        )
