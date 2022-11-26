import time
import unittest

import torch
import torch.fx
import torchvision

from olla import max_cut
from olla.torch import torch_graph_importer


class MaxCutTest(unittest.TestCase):
    def setUp(self):
        self.importer = torch_graph_importer.TorchGraphImporter()
        # limit = 20 * 1024 * 1024 * 1024
        # resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))

    def run_test(self, model, input, mode, expected_cut):
        start = time.time()
        g, pt_node_order = self.importer.import_via_aotautograd(
            model,
            *input,
            optimizer=True,
            mode=mode,
        )
        g.canonicalize()
        g.constrain_weight_updates()
        self.assertTrue(g.is_valid())

        start = time.time()
        mc = max_cut.MaxCut(g, debug=True, rel_stop=0.01)
        cut_size, cut = mc.LocateCut()
        stop = time.time()
        print(f"Located max cut in {stop-start} seconds")
        cut = [t.name for t in cut]
        print(str(cut))
        self.assertEqual(cut, expected_cut)

    def testVGGInference(self):
        model = torchvision.models.vgg11()
        input = torch.randn((1, 3, 224, 224))
        self.run_test(model, (input,), "eval", ["max_pool2d_with_indices"])

    def testVGGTraining(self):
        model = torchvision.models.vgg11()
        input = torch.randn((1, 3, 224, 224))
        self.run_test(
            model,
            (input,),
            "train",
            [
                "args_1",
                "convolution",
                "convolution_1",
                "convolution_2",
                "convolution_3",
                "convolution_4",
                "convolution_5",
                "convolution_6",
                "convolution_7",
                "threshold_backward_1",
                "mm_5",
                "_adaptive_avg_pool2d_backward",
                "max_pool2d_with_indices",
                "max_pool2d_with_indices",
                "max_pool2d_with_indices_1",
                "max_pool2d_with_indices_1",
                "max_pool2d_with_indices_2",
                "max_pool2d_with_indices_2",
                "max_pool2d_with_indices_3",
                "max_pool2d_with_indices_3",
                "max_pool2d_with_indices_4",
            ],
        )

    def testResnetInference(self):
        model = torchvision.models.resnet18(norm_layer=torch.nn.Identity)
        input = torch.randn((1, 3, 224, 224))
        self.run_test(
            model, (input,), "eval", ["convolution_2", "max_pool2d_with_indices"]
        )

    def testResnetTraining(self):
        model = torchvision.models.resnet18(norm_layer=torch.nn.Identity)
        input = torch.randn((1, 3, 224, 224))
        self.run_test(
            model,
            (input,),
            "train",
            [
                "args_1",
                "convolution",
                "convolution_1",
                "convolution_2",
                "convolution_3",
                "convolution_4",
                "convolution_5",
                "convolution_6",
                "convolution_8",
                "convolution_9",
                "convolution_10",
                "convolution_11",
                "convolution_13",
                "convolution_14",
                "convolution_15",
                "max_pool2d_with_indices",
                "max_pool2d_with_indices",
                "convolution_backward_2",
                "convolution_backward_2",
                "convolution_backward_3",
                "convolution_backward_3",
            ],
        )

    def testTransformerDecoderLayerInference(self):
        model = torch.nn.TransformerDecoderLayer(d_model=512, nhead=8)
        memory = torch.rand(10, 32, 512)
        tgt = torch.rand(20, 32, 512)
        self.run_test(
            model,
            (memory, tgt),
            "eval",
            [
                "args_1",
                "addmm",
                "clone_1",
                "div",
                "addmm_3",
                "clone_4",
            ],
        )

    def testTransformerDecoderLayerTrain(self):
        model = torch.nn.TransformerDecoderLayer(d_model=512, nhead=8)
        memory = torch.rand(10, 32, 512)
        tgt = torch.rand(20, 32, 512)
        self.run_test(
            model,
            (memory, tgt),
            "train",
            [
                "args_1",
                "args_2",
                "clone_1",
                "clone_2",
                "div",
                "_softmax",
                "empty_like",
                "mul",
                "clone_3",
                "empty_like_1",
                "add",
                "clone_4",
                "clone_5",
                "div_1",
                "_softmax_1",
                "empty_like_2",
                "mul_2",
                "clone_6",
                "empty_like_3",
                "add_1",
                "relu",
                "mul_6",
                "mm_1",
                "mul_7",
                "native_layer_norm",
                "native_layer_norm",
                "native_layer_norm_1",
                "native_layer_norm_1",
                "native_layer_norm_backward",
            ],
        )
