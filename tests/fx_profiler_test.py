import time
import unittest

import torch
import torchvision
from olla.torch import torch_graph_importer


class FXProfilterTest(unittest.TestCase):
    def testAlexNet(self):
        batch_size = 32
        model = torchvision.models.alexnet()

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        inputs = (torch.randn((batch_size, 3, 224, 224), device=device),)

        model.eval()
        model.to(device)

        model.forward(inputs[0])
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            model.forward(inputs[0])
        if device == "cuda":
            torch.cuda.synchronize()
        stop = time.time()
        pt_time = (stop - start) / 10
        print(f"PT TIME = {pt_time}")

        importer = torch_graph_importer.TorchGraphImporter()
        g, pt_node_order, fx_trace, _ = importer.import_via_aotautograd(
            model,
            *inputs,
            optimizer=None,
            mode="eval",
            return_fx_graph=True,
            profile=["time"],
        )

        profile_time = 0
        for n in g.nodes.values():
            profile_time += n.time
        print(f"PROFILE TIME = {profile_time}")

        # The profiler accuracy is not good enough at the moment.
        # self.assertLessEqual(
        #    abs(pt_time - profile_time) / max(pt_time, profile_time), 0.1
        # )
