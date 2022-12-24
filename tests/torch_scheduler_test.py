import os
import unittest

import torch
import torch.fx
from olla import scheduler
from olla.torch import torch_graph_importer

try:
    del os.environ["LD_LIBRARY_PATH"]
except:
    pass


class TorchGraphScheduler(unittest.TestCase):
    def setUp(self):
        self.importer = torch_graph_importer.TorchGraphImporter()
        self.maxDiff = None

    def testSimpleGraph(self):
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)

        simple_module = SimpleModule()
        input_shape = (3, 4)

        g = self.importer.import_from_torch(simple_module, torch.randn(input_shape))
        g.canonicalize()
        # g.constrain_weights()   #we do not need/support this anymore
        s = scheduler.Scheduler(g)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            mem_limit=1024,
            allow_swaps=True,
            max_spills=0,
            account_for_fragmentation=True,
        )

        s = [n.name + ": " + str(s) + " " for n, s in schedule.items()]
        print(str(s))

        # g.dump("/tmp/new_simple.fx.opt4ml.dot")
        dot = g.dump()
        print(dot)
        self.assertEqual(
            s,
            [
                "x:0: (['1@116'], [2], []) ",
                "param:0: (['1@68'], [2], []) ",
                "add:0: (['2@20'], [3, 4, 5, 6, 7, 8, 9], []) ",
                "linear_weight:0: (['8@68'], [9], []) ",
                "linear_bias:0: (['8@0'], [9], []) ",
                "linear:0: (['9@148'], [], []) ",
            ],
        )
