import os

# import resource
import time
import unittest

import torch
import torch.fx
import torchvision

from olla import simulator, training_graph_optimizer, utils
from olla.torch import torch_graph_importer

# Fix the environment to enable graphviz to work.
del os.environ["LD_LIBRARY_PATH"]


class MemoryOptimizerTest(unittest.TestCase):
    def setUp(self):
        self.importer = torch_graph_importer.TorchGraphImporter()
        # limit = 20 * 1024 * 1024 * 1024
        # resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))

    def run_test(
        self, model, input, mode, tests, timestep_factor=1, memory_reduction_factor=1.0
    ):
        start = time.time()
        g, pt_node_order = self.importer.import_via_aotautograd(
            model,
            *input,
            optimizer=True,
            mode=mode,
        )
        model_name = model.__class__.__name__
        # print(str(g))
        if "dump" in tests:
            g.dump("/tmp/" + model_name + "_" + mode + "_raw", format="svg")
        g.canonicalize()
        self.assertTrue(g.is_valid())
        g.constrain_weight_updates()
        self.assertTrue(g.check_consistency())
        # print(str(g))
        if "dump" in tests:
            g.dump("/tmp/" + model_name + "_" + mode + "_canon", format="svg")
        stop = time.time()
        print(f"PREPARED {mode} GRAPH OF {model_name} IN {stop - start:.1f}s")

        start = time.time()
        s = simulator.Simulator(g)
        simulated_mem_usage, mem_per_timestep = s.Simulate(pt_node_order)

        stop = time.time()
        print(
            f"SIMULATED {mode} GRAPH OF {model_name} IN {stop - start:.1f}s. PEAK MEM USAGE WAS {simulated_mem_usage}"
        )

        start = time.time()
        s = training_graph_optimizer.Scheduler(g, rel_stop=0.01, timeout_s=600)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            allow_swaps=False,
            max_spills=0,
        )
        stop = time.time()
        reordered_mem_usage = summary["peak_mem_usage"]
        print(
            f"REORDERED NODES FOR {mode} GRAPH OF {model_name} IN {stop - start:.1f}s. PEAK MEM USAGE WAS {reordered_mem_usage} (SAVED {(simulated_mem_usage - reordered_mem_usage) / simulated_mem_usage * 100:.1f}%)"
        )

        for s in schedule.values():
            self.assertTrue(len(s[2]) == 0)
        self.assertTrue(utils.validate_timeline(schedule))
        self.assertTrue(utils.validate_node_ordering(g, schedule))
        self.assertEqual(summary["peak_mem_usage"], summary["required_memory"])
        self.assertLessEqual(summary["peak_mem_usage"], simulated_mem_usage)
        self.assertEqual(summary["total_data_swapped"], 0)
        # self.assertEqual(summary["peak_mem_usage"], 140274144.0)

        if "spills" in tests:
            start = time.time()
            s = training_graph_optimizer.Scheduler(
                g, rel_stop=0.01, timeout_s=600, timestep_factor=timestep_factor
            )
            min_required_memory, _ = s.ComputeMinimumMemoryRequired()
            mem_limit = max(
                reordered_mem_usage / memory_reduction_factor, min_required_memory
            )
            summary, schedule, mem_loc = s.ComputeOptimalSchedule(
                mem_limit=mem_limit,
                allow_swaps=True,
                max_spills=None,
            )
            stop = time.time()
            peak_mem_usage = summary["peak_mem_usage"]
            spills = summary["total_data_swapped"]
            print(
                f"DONE SCHEDULING SPILLS FOR {mode} GRAPH OF {model_name} IN {stop - start:.1f}s. USED {peak_mem_usage} PEAK MEM (SAVED {(simulated_mem_usage - peak_mem_usage) / simulated_mem_usage * 100:.1f}%), SPILLED {spills}"
            )

            self.assertTrue(utils.validate_timeline(schedule))
            self.assertEqual(summary["peak_mem_usage"], summary["required_memory"])
            self.assertLessEqual(summary["peak_mem_usage"], mem_limit)
            # self.assertEqual(summary["peak_mem_usage"], 46758048.0)

        if "recompute" in tests:
            start = time.time()
            s = training_graph_optimizer.Scheduler(
                g, rel_stop=0.01, timeout_s=600, timestep_factor=timestep_factor
            )
            min_required_memory, _ = s.ComputeMinimumMemoryRequired()
            mem_limit = max(
                reordered_mem_usage / memory_reduction_factor, min_required_memory
            )
            summary, schedule, mem_loc = s.ComputeOptimalSchedule(
                mem_limit=mem_limit,
                allow_swaps=True,
                allow_rematerialization=True,
                max_spills=None,
            )
            stop = time.time()
            peak_mem_usage = summary["peak_mem_usage"]
            spills = summary["total_data_swapped"]
            print(
                f"DONE SCHEDULING RECOMPUTES FOR {mode} GRAPH OF {model_name} IN {stop - start:.1f}s. USED {peak_mem_usage} PEAK MEM (SAVED {(simulated_mem_usage - peak_mem_usage) / simulated_mem_usage * 100:.1f}%), SPILLED {spills}"
            )

            self.assertTrue(utils.validate_timeline(schedule))
            self.assertEqual(summary["peak_mem_usage"], summary["required_memory"])
            self.assertLessEqual(summary["peak_mem_usage"], mem_limit)

        if "memory" in tests:
            start = time.time()
            s = training_graph_optimizer.Scheduler(
                g, rel_stop=0.01, timeout_s=600, timestep_factor=timestep_factor
            )
            min_required_memory, _ = s.ComputeMinimumMemoryRequired()
            mem_limit = max(
                reordered_mem_usage / memory_reduction_factor, min_required_memory
            )
            summary, schedule, mem_loc = s.ComputeOptimalSchedule(
                mem_limit=mem_limit,
                allow_swaps=True,
                max_spills=None,
                defrag=False,
                account_for_fragmentation=True,
            )
            stop = time.time()
            peak_mem_usage = summary["required_memory"]
            spills = summary["total_data_swapped"]
            print(
                f"DONE PLANNING MEMORY FOR {mode} GRAPH OF {model_name} IN {stop - start:.1f}s. USED {peak_mem_usage} REQUIRED MEM (SAVED {(simulated_mem_usage - peak_mem_usage) / simulated_mem_usage * 100:.1f}%), SPILLED {spills}"
            )

            self.assertTrue(utils.validate_timeline(schedule))
            self.assertTrue(utils.validate_address_allocation(mem_loc))
            self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
            self.assertLessEqual(summary["required_memory"], mem_limit)

        if "defrag" in tests:
            start = time.time()
            s = training_graph_optimizer.Scheduler(
                g, rel_stop=0.01, timeout_s=600, timestep_factor=timestep_factor
            )
            min_required_memory, _ = s.ComputeMinimumMemoryRequired()
            mem_limit = max(
                reordered_mem_usage / memory_reduction_factor, min_required_memory
            )
            summary, schedule, mem_loc = s.ComputeOptimalSchedule(
                mem_limit=mem_limit,
                allow_swaps=True,
                max_spills=None,
                defrag=True,
                account_for_fragmentation=True,
            )
            stop = time.time()
            peak_mem_usage = summary["required_memory"]
            spills = summary["total_data_swapped"]
            print(
                f"DONE PLANNING MEMORY DEFRAGMENTATION FOR {mode} GRAPH OF {model_name} IN {stop - start:.1f}s. USED {peak_mem_usage} REQUIRED MEM (SAVED {(simulated_mem_usage - peak_mem_usage) / simulated_mem_usage * 100:.1f}%), SPILLED {spills}"
            )

            self.assertTrue(utils.validate_timeline(schedule))
            self.assertTrue(utils.validate_address_allocation(mem_loc))
            self.assertLessEqual(summary["peak_mem_usage"], summary["required_memory"])
            self.assertLessEqual(summary["required_memory"], mem_limit)

        # self.assertEqual(1, 2)

    def testVGGInference(self):
        model = torchvision.models.vgg11()
        input = torch.randn((1, 3, 224, 224))
        self.run_test(
            model,
            (input,),
            "eval",
            ("spills", "recompute", "memory", "defrag"),
            timestep_factor=1.05,
            memory_reduction_factor=1.1,
        )

    def testVGGTraining(self):
        model = torchvision.models.vgg11()
        input = torch.randn((1, 3, 224, 224))
        self.run_test(
            model,
            (input,),
            "train",
            ("spills", "recompute"),
            memory_reduction_factor=1.1,
        )

    def testResnetInference(self):
        # Use identity as the layer normalization layer to get rid of all the dangling weights
        model = torchvision.models.resnet18(norm_layer=torch.nn.Identity)
        input = torch.randn((1, 3, 224, 224))
        self.run_test(
            model,
            (input,),
            "eval",
            ("spills", "recompute", "memory"),
            memory_reduction_factor=1.1,
        )

    def testResnetTraining(self):
        # Use identity as the layer normalization layer to get rid of all the dangling weights
        model = torchvision.models.resnet18(norm_layer=torch.nn.Identity)
        input = torch.randn((1, 3, 224, 224))
        self.run_test(
            model,
            (input,),
            "train",
            ("spills", "recompute"),
            memory_reduction_factor=1.1,
        )

    def testTransformerDecoderLayerInference(self):
        model = torch.nn.TransformerDecoderLayer(d_model=512, nhead=8)
        memory = torch.rand(10, 32, 512)
        tgt = torch.rand(20, 32, 512)
        self.run_test(
            model,
            (memory, tgt),
            "eval",
            ("spills", "recompute", "memory"),
            memory_reduction_factor=1.1,
        )

    def testTransformerDecoderLayerTrain(self):
        model = torch.nn.TransformerDecoderLayer(d_model=512, nhead=8)
        memory = torch.rand(10, 32, 512)
        tgt = torch.rand(20, 32, 512)
        # Improving the lower bound estimate on the node reordering
        # problem takes forever, so it's disabled for now
        self.run_test(
            model,
            (memory, tgt),
            "train",
            # ("simulate", "reordering", "spills", "recompute"),
            ("spills", "recompute"),
            memory_reduction_factor=1.1,
        )
