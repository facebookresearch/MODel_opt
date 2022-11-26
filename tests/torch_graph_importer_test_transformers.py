import unittest

import torch
import torch.fx

from olla.torch import torch_graph_importer


class TorchGraphImporterTestVision(unittest.TestCase):
    def setUp(self):
        self.importer = torch_graph_importer.TorchGraphImporter()
        self.maxDiff = None

    def run_tests(
        self,
        model,
        *inputs,
        mode="eval",
        methods=None,
        optimizer=None,
        test_name="test",
    ):
        if methods is None:
            methods = ["aotautograd"]

        for method in methods:
            with self.subTest():
                g = self.importer.import_from_torch(
                    model,
                    *inputs,
                    mode=mode,
                    method=method,
                    optimizer=optimizer,
                )
                self.assertTrue(g.check_consistency())
                # g.dump(f"/tmp/{test_name}.{method}.dot")

    def testTransformerDecoderLayerInference(self):
        model = torch.nn.TransformerDecoderLayer(d_model=512, nhead=8)
        memory = torch.rand(10, 32, 512)
        tgt = torch.rand(20, 32, 512)

        model.eval()
        self.run_tests(
            model,
            memory,
            tgt,
            mode="eval",
            methods=[
                # "fx", # FIXME
                "aotautograd",
                "functorch",
                # "acc_tracer", # FIXME: Error message: P523544272
            ],
            test_name="transformer.decoder.inference",
        )

    def testTransformerEncoderLayerTrainDefaultOptimizer(self):
        batch_size, seq_len, embedding_dim = 32, 256, 1024
        model = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=16, dim_feedforward=4096
        )
        input_shapes = [(batch_size, seq_len, embedding_dim)]
        inputs = [torch.randn(s, requires_grad=True) for s in input_shapes]

        model.train()
        self.run_tests(
            model,
            *inputs,
            mode="train",
            methods=[
                "aotautograd",
                # "functorch", # FIXME: F0821 15:06:40.033880 1327465 layer_norm_kernel.cpp:120] Check failed: dY.numel() == M * N (262144 vs. 8388608)
            ],
            optimizer=True,
            test_name="transformer.encoder.train",
        )

    def testTransformerEncoderLayerTrainSGDOptimizer(self):
        batch_size, seq_len, embedding_dim = 32, 256, 1024
        model = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=16, dim_feedforward=4096
        )
        input_shapes = [(batch_size, seq_len, embedding_dim)]
        inputs = [torch.randn(s, requires_grad=True) for s in input_shapes]

        # Fix the case where we use momentum and weight_decay: T126360101
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1  # , momentum=0.9, weight_decay=1e-4
        )

        model.train()
        self.run_tests(
            model,
            *inputs,
            mode="train",
            methods=[
                "aotautograd",
                # "functorch", # FIXME
            ],
            optimizer=optimizer,
            test_name="transformer.encoder.train.sgd",
        )
