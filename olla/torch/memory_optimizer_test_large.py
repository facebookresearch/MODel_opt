import os
import time
import unittest

import torch
import torch.fx

from olla import simulator, training_graph_optimizer, utils
from olla.torch import torch_graph_importer

# Fix the environment to enable graphviz to work.
del os.environ["LD_LIBRARY_PATH"]


class MemoryOptimizerTest(unittest.TestCase):
    def setUp(self):
        self.importer = torch_graph_importer.TorchGraphImporter()

    def run_test(self, model, input, mode, tests):
        start = time.time()
        g, pt_node_order = self.importer.import_via_aotautograd(
            model,
            *input,
            optimizer=True,
            mode=mode,
        )
        # print(str(g))
        # g.dump("/tmp/vgg11.aotautograd.opt4ml_raw.dot")
        # g.dump("/tmp/vgg11.aotautograd.opt4ml_raw", format="png")
        # g.dump("/tmp/vgg11.aotautograd.opt4ml_raw", format="svg")
        # stop = time.time()
        g.canonicalize()
        self.assertTrue(g.is_valid())
        g.constrain_weight_updates()
        self.assertTrue(g.check_consistency())
        # print(str(g))
        # g.dump("/tmp/vgg11.aotautograd.opt4ml.dot")
        # g.dump("/tmp/vgg11.aotautograd.opt4ml", format="png")
        # g.dump("/tmp/vgg11.aotautograd.opt4ml", format="svg")
        stop = time.time()
        print("DONE PREPARING MODEL: " + str(stop - start))
        start = time.time()

        if "simulate" in tests:
            start = time.time()
            s = simulator.Simulator(g)
            peak_mem_usage, mem_per_timestep = s.Simulate(pt_node_order)

            stop = time.time()
            print("DONE SIMULATING GRAPH: " + str(stop - start))

        if "reordering" in tests:
            s = training_graph_optimizer.Scheduler(g)
            summary, schedule, mem_loc = s.ComputeOptimalSchedule(
                # mem_limit=9999999999,
                allow_swaps=False,
                max_spills=0,
            )
            stop = time.time()
            print("DONE REORDERING NODES: " + str(stop - start))

            self.assertTrue(utils.validate_timeline(schedule))
            self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
            # self.assertEqual(summary["total_data_swapped"], 0)
            # self.assertEqual(summary["peak_mem_usage"], 140274144.0)

        if "spills" in tests:
            start = time.time()
            s = training_graph_optimizer.Scheduler(g, rel_stop=0.01)
            summary, schedule, mem_loc = s.ComputeOptimalSchedule(
                mem_limit=822083584 * 2,
                allow_swaps=True,
                max_spills=None,
            )
            stop = time.time()
            print("DONE SCHEDULING SPILLS: " + str(stop - start))

            self.assertTrue(utils.validate_timeline(schedule))
            self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
            # self.assertEqual(summary["peak_mem_usage"], 46758048.0)

        if "recompute" in tests:
            start = time.time()
            s = training_graph_optimizer.Scheduler(g, rel_stop=0.01)
            summary, schedule, mem_loc = s.ComputeOptimalSchedule(
                mem_limit=822083584 * 2,
                allow_swaps=True,
                allow_rematerialization=True,
                max_spills=None,
            )
            stop = time.time()
            print("DONE SCHEDULING RECOMPUTES: " + str(stop - start))

            self.assertTrue(utils.validate_timeline(schedule))
            self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
            # self.assertEqual(summary["peak_mem_usage"], 46758048.0)

        if "memory" in tests:
            start = time.time()
            s = training_graph_optimizer.Scheduler(g, rel_stop=0.01)
            summary, schedule, mem_loc = s.ComputeOptimalSchedule(
                mem_limit=822083584 * 2,
                allow_swaps=True,
                max_spills=None,
                defrag=False,
                account_for_fragmentation=True,
            )
            stop = time.time()
            print("DONE PLANNING MEMORY: " + str(stop - start))

            self.assertTrue(utils.validate_timeline(schedule))
            self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
            # self.assertEqual(summary["peak_mem_usage"], 46758048.0)

        if "defrag" in tests:
            start = time.time()
            s = training_graph_optimizer.Scheduler(g, rel_stop=0.01)
            summary, schedule, mem_loc = s.ComputeOptimalSchedule(
                mem_limit=822083584 * 2,
                allow_swaps=True,
                max_spills=None,
                defrag=True,
                account_for_fragmentation=True,
            )
            stop = time.time()
            print("DONE PLANNING MEMORY DEFRAGMENTATION: " + str(stop - start))

            self.assertTrue(utils.validate_timeline(schedule))
            self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
            # self.assertEqual(summary["peak_mem_usage"], 46758048.0)

    def testTransformerFullInference(self):
        model = torch.nn.Transformer(
            nhead=1, num_encoder_layers=1, num_decoder_layers=1
        )
        src = torch.rand((10, 32, 512))
        tgt = torch.rand((20, 32, 512))

        self.run_test(
            model,
            (src, tgt),
            "eval",
            ("reordering", "spills", "recompute", "memory", "defrag"),
        )

    def testTransformerFullTrain(self):
        model = torch.nn.Transformer(
            nhead=1, num_encoder_layers=1, num_decoder_layers=1
        )
        src = torch.rand((10, 32, 512))
        tgt = torch.rand((20, 32, 512))

        self.run_test(
            model, (src, tgt), "train", ("reordering",)  # "spills", "recompute"),
        )
