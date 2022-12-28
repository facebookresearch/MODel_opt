import unittest
import os

import torch

from olla.native_graphs import graph_with_gradients
from olla.torch import spill_profiler


class SpillProfilterTest(unittest.TestCase):
    @unittest.skipIf(not bool(os.getenv('RUN_SKIPPED', 0)), "Spilling implementation not yet fully-complete")
    def testBasic(self):
        g = graph_with_gradients.graph
        profiler = spill_profiler.SpillProfiler(g, warm_up_iters=1, profile_iters=10)
        profile = profiler.benchmark_all()

        rslt = {e.name: time for e, time in profile.items()}

        if torch.cuda.is_available():
            print(
                f"Warning: stats collected on actual gpu: {torch.cuda.get_device_properties('cuda')}"
            )

        print(str(rslt))
        self.assertEqual(
            rslt,
            {
                "ACTIVATION1": 1.0002e-05,
                "WEIGHT_REF1": 1.00246e-05,
                "ACTIVATION2": 1.0006e-05,
                "WEIGHT_REF2": 1.0064200000000001e-05,
                "OUTPUT_EDGE": 1.001e-05,
                "PROPAGATE_G1": 1.0069000000000001e-05,
                "PROPAGATE_G2": 1.01086e-05,
                "UPDATE_W1": 1.01134e-05,
                "UPDATE_W2": 1.0153000000000001e-05,
            },
        )
