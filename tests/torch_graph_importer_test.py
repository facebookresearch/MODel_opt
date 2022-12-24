import os
import unittest

import torch
import torch.fx
from olla import dataflow_graph
from olla.torch import torch_graph_importer

try:
    del os.environ["LD_LIBRARY_PATH"]
except:
    pass


class TorchGraphImporterTest(unittest.TestCase):
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
                self.assertTrue(g.is_valid())
                # g.dump(f"/tmp/{test_name}.{method}.dot")

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

        # TODO: currently, the default for inference is fx, and for training is aotautograd
        # It may be better to set aotautograd default for both inference and training
        with self.subTest("default"):
            g = self.importer.import_from_torch(simple_module, torch.randn(input_shape))
            self.assertTrue(g.is_valid())
            # g.dump("/tmp/simple.fx.opt4ml.dot")
            dot = g.dump()
            print(dot)
            self.assertEqual(
                dot,
                """digraph {
\tnode [shape=record]
\tx [label="x (placeholder)"]
\tparam [label="param (weight) [48]"]
\tadd [label="add (call_function)"]
\tlinear_weight [label="linear_weight (weight) [80]"]
\tlinear_bias [label="linear_bias (weight) [20]"]
\tlinear [label="linear (call_function)"]
\tclamp [label="clamp (call_method)"]
\tx -> add [label=48]
\tparam -> add [label=0]
\tadd -> linear [label=48]
\tlinear_weight -> linear [label=0]
\tlinear_bias -> linear [label=0]
\tlinear -> clamp [label=60]
}
""",
            )

            g.canonicalize()
            self.assertTrue(g.is_valid())
            dot = g.dump()
            print(dot)
            self.assertEqual(
                dot,
                """digraph {
\tnode [shape=record]
\tx [label="x (placeholder)"]
\tparam [label="param (stateful_node)"]
\tadd [label="add (call_function)"]
\tlinear_weight [label="linear_weight (stateful_node)"]
\tlinear_bias [label="linear_bias (stateful_node)"]
\tlinear [label="linear (call_function)"]
\tclamp [label="clamp (call_method)"]
\tparam_snk [label="param_snk (stateful_node_sink)"]
\tlinear_weight_snk [label="linear_weight_snk (stateful_node_sink)"]
\tlinear_bias_snk [label="linear_bias_snk (stateful_node_sink)"]
\tx -> add [label=48]
\tparam -> add [label=48]
\tparam -> param_snk [label=48]
\tadd -> linear [label=48]
\tlinear_weight -> linear [label=80]
\tlinear_weight -> linear_weight_snk [label=80]
\tlinear_bias -> linear [label=20]
\tlinear_bias -> linear_bias_snk [label=20]
\tlinear -> clamp [label=60]
}
""",
            )

        with self.subTest("aotautograd"):
            g, _ = self.importer.import_via_aotautograd(
                simple_module, torch.randn(input_shape), mode="eval"
            )
            self.assertTrue(g.is_valid())
            # g.dump("/tmp/simple.aotautograd.opt4ml.dot")

        with self.subTest("functorch"):
            g = self.importer.import_via_functorch(
                simple_module, torch.randn(input_shape), mode="eval"
            )
            self.assertTrue(g.is_valid())
            # g.dump("/tmp/simple.functorch.opt4ml.dot")

        with self.subTest("acc_tracer"):
            # FIXME: check why batchnorm weights have 0 tensor sizes in the graph
            g = self.importer.import_via_acc_tracer(
                simple_module, torch.randn(input_shape), mode="eval"
            )
            self.assertTrue(g.is_valid())
            # g.dump("/tmp/simple.acc_tracer.opt4ml.dot")

            dot = g.dump()
            print(dot)
            self.assertEqual(
                dot,
                """digraph {
\tnode [shape=record]
\tx [label="x (placeholder)"]
\tparam [label="param (weight) [48]"]
\tadd_1 [label="add_1 (call_function)"]
\tlinear_weight [label="linear_weight (weight) [80]"]
\tlinear_bias [label="linear_bias (weight) [20]"]
\tlinear_1 [label="linear_1 (call_function)"]
\tclamp_1 [label="clamp_1 (call_function)"]
\tx -> add_1 [label=48]
\tparam -> add_1 [label=0]
\tadd_1 -> linear_1 [label=48]
\tlinear_weight -> linear_1 [label=0]
\tlinear_bias -> linear_1 [label=0]
\tlinear_1 -> clamp_1 [label=60]
}
""",
            )

    def testWeightsWithCast(self):
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))

            def forward(self, x):
                return (x + self.param.float()).clamp(min=0.0, max=1.0)

        simple_module = SimpleModule()
        input_shape = (3, 4)

        with self.subTest("vanilla_fx"):
            g = self.importer.import_via_fx(
                simple_module, torch.randn(input_shape), mode="eval"
            )
            self.assertTrue(g.is_valid())
            # g.dump("/tmp/weightswithcast.fx.opt4ml.dot")
            dot = g.dump()
            print(dot)
            self.assertEqual(
                dot,
                """digraph {
\tnode [shape=record]
\tx [label="x (placeholder)"]
\tparam [label="param (weight) [48]"]
\tfloat_1 [label="float_1 (call_method)"]
\tadd [label="add (call_function)"]
\tclamp [label="clamp (call_method)"]
\tx -> add [label=48]
\tparam -> float_1 [label=0]
\tfloat_1 -> add [label=48]
\tadd -> clamp [label=48]
}
""",
            )

            g.canonicalize()
            self.assertTrue(g.is_valid())
            dot = g.dump()
            print(dot)
            self.assertEqual(
                dot,
                """digraph {
\tnode [shape=record]
\tx [label="x (placeholder)"]
\tparam [label="param (stateful_node)"]
\tfloat_1 [label="float_1 (call_method)"]
\tadd [label="add (call_function)"]
\tclamp [label="clamp (call_method)"]
\tparam_snk [label="param_snk (stateful_node_sink)"]
\tx -> add [label=48]
\tparam -> float_1 [label=48]
\tparam -> param_snk [label=48]
\tfloat_1 -> add [label=48]
\tadd -> clamp [label=48]
}
""",
            )

        with self.subTest("aotautograd"):
            g, _ = self.importer.import_via_aotautograd(
                simple_module, torch.randn(input_shape), mode="eval"
            )
            self.assertTrue(g.is_valid())
            # g.dump("/tmp/weightswithcast.aotautograd.opt4ml.dot")

        with self.subTest("functorch"):
            g = self.importer.import_via_functorch(
                simple_module, torch.randn(input_shape), mode="eval"
            )
            self.assertTrue(g.is_valid())
            # g.dump("/tmp/weightswithcast.functorch.opt4ml.dot")

        with self.subTest("acc_tracer"):
            g = self.importer.import_via_acc_tracer(
                simple_module, torch.randn(input_shape), mode="eval"
            )
            self.assertTrue(g.is_valid())
            # g.dump("/tmp/weightswithcast.acc_tracer.opt4ml.dot")
            dot = g.dump()
            print(dot)
            self.assertEqual(
                dot,
                """digraph {
\tnode [shape=record]
\tx [label="x (placeholder)"]
\tparam [label="param (weight) [48]"]
\tto_dtype [label="to_dtype (call_function)"]
\tadd_1 [label="add_1 (call_function)"]
\tclamp_1 [label="clamp_1 (call_function)"]
\tx -> add_1 [label=48]
\tparam -> to_dtype [label=0]
\tto_dtype -> add_1 [label=48]
\tadd_1 -> clamp_1 [label=48]
}
""",
            )

            g.canonicalize()
            self.assertTrue(g.is_valid())
            dot = g.dump()
            print(dot)
            self.assertEqual(
                dot,
                """digraph {
\tnode [shape=record]
\tx [label="x (placeholder)"]
\tparam [label="param (stateful_node)"]
\tto_dtype [label="to_dtype (call_function)"]
\tadd_1 [label="add_1 (call_function)"]
\tclamp_1 [label="clamp_1 (call_function)"]
\tparam_snk [label="param_snk (stateful_node_sink)"]
\tx -> add_1 [label=48]
\tparam -> to_dtype [label=48]
\tparam -> param_snk [label=48]
\tto_dtype -> add_1 [label=48]
\tadd_1 -> clamp_1 [label=48]
}
""",
            )

    def testTrainSimpleGraphWithoutWeightUpdate(self):
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)

        simple_module = SimpleModule()
        input_shape = (3, 4)

        with self.subTest("functorch"):
            g = self.importer.import_via_functorch(
                simple_module,
                torch.randn(input_shape, requires_grad=True),
                mode="train",
            )
            self.assertTrue(g.is_valid())
            # g.dump("/tmp/trainsimple.functorch.opt4ml.dot")
            dot = g.dump()
            print(dot)

        with self.subTest("aotautograd"):
            g, _ = self.importer.import_via_aotautograd(
                simple_module,
                torch.randn(input_shape, requires_grad=True),
                mode="train",
            )
            self.assertTrue(g.is_valid())
            # g.dump("/tmp/trainsimple.aotautograd.opt4ml.dot")

            # remove transpose node
            nodes = g.nodes.copy().values()
            for node in g.find_nodes(name="t"):
                self.importer.bypass_and_delete_meta_node(node)
            # g.dump("/tmp/trainsimple.aotautograd.opt4ml.nometa.dot")

    def testGetItem(self):
        class PoolingModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.pool = torch.nn.MaxPool2d((3, 3), return_indices=True)

            def forward(self, x):
                res = self.pool(x)[0]
                return res

        pooling_module = PoolingModule()
        input_shape = (123, 10, 9, 9)

        g, _ = self.importer.import_via_aotautograd(
            pooling_module, torch.randn(input_shape, requires_grad=True), mode="train"
        )
        self.assertTrue(g.is_valid())
        # g.dump("/tmp/maxpool.aotautograd.opt4ml.dot")
        # dot = g.dump()
        # print(dot)
        print(str(g))
        self.assertEqual(
            str(g),
            """args_1 (placeholder)
max_pool2d_with_indices (call_function)
sum_1 (call_function)
ones_like (call_function)
max_pool2d_with_indices_backward (call_function)

MultiSourceEdge args_1:0, size:398520, mem_space:None, tile_id:None group_id:None sources:[args_1 (placeholder)] sinks:[max_pool2d_with_indices (call_function), max_pool2d_with_indices_backward (call_function)]
MultiSourceEdge sum_1:0, size:4, mem_space:None, tile_id:None group_id:None sources:[sum_1 (call_function)] sinks:[ones_like (call_function)]
MultiSourceEdge ones_like:0, size:4, mem_space:None, tile_id:None group_id:None sources:[ones_like (call_function)] sinks:[max_pool2d_with_indices_backward (call_function)]
MultiSourceEdge getitem:0_opt, size:44280, mem_space:None, tile_id:None group_id:None sources:[max_pool2d_with_indices (call_function)] sinks:[sum_1 (call_function)]
MultiSourceEdge getitem_1:0_opt, size:88560, mem_space:None, tile_id:None group_id:None sources:[max_pool2d_with_indices (call_function)] sinks:[max_pool2d_with_indices_backward (call_function)]
""",
        )

    def testSplit(self):
        class SplitModule(torch.nn.Module):
            def forward(self, x):
                res = torch.split(x, 3)
                for slice in res:
                    assert id(slice.storage()) == id(x.storage())
                vals = [res[0]]
                vals.append(torch.relu(res[1]))
                vals.append(torch.sigmoid(res[2]))
                return torch.concat(vals)

        split_module = SplitModule()
        input_shape = (33, 128)

        g, _ = self.importer.import_via_aotautograd(
            split_module, torch.randn(input_shape, requires_grad=True), mode="train"
        )
        self.assertTrue(g.is_valid())
        # g.dump("/tmp/split.aotautograd.opt4ml.dot")
        # dot = g.dump()
        # print(dot)
        # print(str(g))
        self.assertEqual(
            str(g),
            """args_1 (placeholder)
relu (call_function)
sigmoid (call_function)
cat (call_function)
sum_1 (call_function)
ones_like (call_function)
sigmoid_backward (call_function)
threshold_backward (call_function)
zeros (call_function)
zeros_1 (call_function)
zeros_2 (call_function)
zeros_3 (call_function)
zeros_4 (call_function)
zeros_5 (call_function)
zeros_6 (call_function)
zeros_7 (call_function)
cat_1 (call_function)

MultiSourceEdge args_1:0, size:16896, mem_space:None, tile_id:None group_id:None sources:[args_1 (placeholder)] sinks:[cat (call_function), relu (call_function), sigmoid (call_function)]
MultiSourceEdge relu:0, size:1536, mem_space:None, tile_id:None group_id:None sources:[relu (call_function)] sinks:[cat (call_function), threshold_backward (call_function)]
MultiSourceEdge sigmoid:0, size:1536, mem_space:None, tile_id:None group_id:None sources:[sigmoid (call_function)] sinks:[cat (call_function), sigmoid_backward (call_function)]
MultiSourceEdge cat:0, size:4608, mem_space:None, tile_id:None group_id:None sources:[cat (call_function)] sinks:[sum_1 (call_function)]
MultiSourceEdge sum_1:0, size:4, mem_space:None, tile_id:None group_id:None sources:[sum_1 (call_function)] sinks:[ones_like (call_function)]
MultiSourceEdge ones_like:0, size:4, mem_space:None, tile_id:None group_id:None sources:[ones_like (call_function)] sinks:[cat_1 (call_function), threshold_backward (call_function), sigmoid_backward (call_function)]
MultiSourceEdge sigmoid_backward:0, size:1536, mem_space:None, tile_id:None group_id:None sources:[sigmoid_backward (call_function)] sinks:[cat_1 (call_function)]
MultiSourceEdge threshold_backward:0, size:1536, mem_space:None, tile_id:None group_id:None sources:[threshold_backward (call_function)] sinks:[cat_1 (call_function)]
MultiSourceEdge zeros:0, size:1536, mem_space:None, tile_id:None group_id:None sources:[zeros (call_function)] sinks:[cat_1 (call_function)]
MultiSourceEdge zeros_1:0, size:1536, mem_space:None, tile_id:None group_id:None sources:[zeros_1 (call_function)] sinks:[cat_1 (call_function)]
MultiSourceEdge zeros_2:0, size:1536, mem_space:None, tile_id:None group_id:None sources:[zeros_2 (call_function)] sinks:[cat_1 (call_function)]
MultiSourceEdge zeros_3:0, size:1536, mem_space:None, tile_id:None group_id:None sources:[zeros_3 (call_function)] sinks:[cat_1 (call_function)]
MultiSourceEdge zeros_4:0, size:1536, mem_space:None, tile_id:None group_id:None sources:[zeros_4 (call_function)] sinks:[cat_1 (call_function)]
MultiSourceEdge zeros_5:0, size:1536, mem_space:None, tile_id:None group_id:None sources:[zeros_5 (call_function)] sinks:[cat_1 (call_function)]
MultiSourceEdge zeros_6:0, size:1536, mem_space:None, tile_id:None group_id:None sources:[zeros_6 (call_function)] sinks:[cat_1 (call_function)]
MultiSourceEdge zeros_7:0, size:1536, mem_space:None, tile_id:None group_id:None sources:[zeros_7 (call_function)] sinks:[cat_1 (call_function)]
""",
        )

    def testInPlaceOps(self):
        class InPlaceModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                chan = 15
                self.conv0 = torch.nn.Conv2d(chan, chan, 3, padding=1)
                self.bn0 = torch.nn.BatchNorm2d(chan)
                self.relu0 = torch.nn.ReLU(inplace=True)
                self.conv1 = torch.nn.Conv2d(chan, chan, 3, padding=1)
                self.bn1 = torch.nn.BatchNorm2d(chan)
                self.relu1 = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(chan, chan, 3, padding=1)
                self.bn2 = torch.nn.BatchNorm2d(chan)
                self.relu2 = torch.nn.ReLU()
                self.conv3 = torch.nn.Conv2d(chan, chan, 3, padding=1)
                self.conv4 = torch.nn.Conv2d(chan, chan, 3, padding=1)

            def forward(self, x):
                residual = x
                res = self.conv0(x)
                res = self.bn0(res)
                res = self.relu0(res)
                res = self.conv1(res)
                res = self.bn1(res)
                res += residual
                residual = res
                res = self.relu1(res)
                res = self.conv2(res)
                res = self.bn2(res)
                res += residual
                res = self.relu2(res)
                res1 = self.conv3(res)
                res2 = self.conv4(res)
                return torch.concat([res1, res2])

        in_place_module = InPlaceModule()
        input_shape = (123, 15, 9, 9)

        optimizer = torch.optim.SGD(in_place_module.parameters(), lr=0.1)
        g, _ = self.importer.import_via_aotautograd(
            in_place_module,
            torch.randn(input_shape, requires_grad=True),
            mode="train",
            optimizer=optimizer,
        )
        self.assertTrue(g.is_valid(verbose=True))
        g.dump("/tmp/inplace.aotautograd.opt4ml", "svg")
        # dot = g.dump()
        # print(dot)
        print(str(g))
        self.assertEqual(
            str(g),
            """args_1 (placeholder)
params_1 (weight) [8160]
params_3 (weight) [240]
params_5 (weight) [8160]
params_7 (weight) [240]
params_9 (weight) [8160]
params_11 (weight) [240]
params_13 (weight) [8160]
params_15 (weight) [8160]
convolution (call_function)
native_batch_norm (call_function)
convolution_1 (call_function)
native_batch_norm_1 (call_function)
add__2 (call_function)
relu (call_function)
convolution_2 (call_function)
native_batch_norm_2 (call_function)
add__4 (call_function)
relu_1 (call_function)
convolution_3 (call_function)
convolution_4 (call_function)
cat (call_function)
sum_1 (call_function)
ones_like (call_function)
convolution_backward (call_function)
convolution_backward_1 (call_function)
add (call_function)
threshold_backward (call_function)
native_batch_norm_backward (call_function)
convolution_backward_2 (call_function)
threshold_backward_1 (call_function)
add_1 (call_function)
native_batch_norm_backward_1 (call_function)
convolution_backward_3 (call_function)
threshold_backward_2 (call_function)
native_batch_norm_backward_2 (call_function)
convolution_backward_4 (call_function)
add_2 (call_function)
add__5 (call_function)
add__7 (call_function)
add__9 (call_function)
add__11 (call_function)
add__13 (call_function)
add__15 (call_function)
add__17 (call_function)
add__19 (call_function)

MultiSourceEdge args_1:0, size:597780, mem_space:None, tile_id:None group_id:None sources:[args_1 (placeholder)] sinks:[convolution (call_function), add__2 (call_function), convolution_backward_4 (call_function), relu (call_function), add__4 (call_function), relu_1 (call_function)]
MultiSourceEdge params_1:0, size:0, mem_space:None, tile_id:None group_id:None sources:[params_1 (weight) [8160]] sinks:[convolution (call_function), convolution_backward_4 (call_function), add__5 (call_function)]
MultiSourceEdge params_3:0, size:0, mem_space:None, tile_id:None group_id:None sources:[params_3 (weight) [240]] sinks:[native_batch_norm (call_function), native_batch_norm_backward_2 (call_function), add__7 (call_function)]
MultiSourceEdge params_5:0, size:0, mem_space:None, tile_id:None group_id:None sources:[params_5 (weight) [8160]] sinks:[convolution_1 (call_function), convolution_backward_3 (call_function), add__9 (call_function)]
MultiSourceEdge params_7:0, size:0, mem_space:None, tile_id:None group_id:None sources:[params_7 (weight) [240]] sinks:[native_batch_norm_1 (call_function), native_batch_norm_backward_1 (call_function), add__11 (call_function)]
MultiSourceEdge params_9:0, size:0, mem_space:None, tile_id:None group_id:None sources:[params_9 (weight) [8160]] sinks:[convolution_2 (call_function), convolution_backward_2 (call_function), add__13 (call_function)]
MultiSourceEdge params_11:0, size:0, mem_space:None, tile_id:None group_id:None sources:[params_11 (weight) [240]] sinks:[native_batch_norm_2 (call_function), native_batch_norm_backward (call_function), add__15 (call_function)]
MultiSourceEdge params_13:0, size:0, mem_space:None, tile_id:None group_id:None sources:[params_13 (weight) [8160]] sinks:[convolution_3 (call_function), convolution_backward_1 (call_function), add__17 (call_function)]
MultiSourceEdge params_15:0, size:0, mem_space:None, tile_id:None group_id:None sources:[params_15 (weight) [8160]] sinks:[convolution_4 (call_function), convolution_backward (call_function), add__19 (call_function)]
MultiSourceEdge convolution:0, size:597780, mem_space:None, tile_id:None group_id:None sources:[convolution (call_function)] sinks:[native_batch_norm (call_function), native_batch_norm_backward_2 (call_function)]
MultiSourceEdge convolution_1:0, size:597780, mem_space:None, tile_id:None group_id:None sources:[convolution_1 (call_function)] sinks:[native_batch_norm_1 (call_function), native_batch_norm_backward_1 (call_function)]
MultiSourceEdge add__2:0, size:0, mem_space:None, tile_id:None group_id:None sources:[add__2 (call_function)] sinks:[relu (call_function), add__4 (call_function)]
MultiSourceEdge relu:0, size:597780, mem_space:None, tile_id:None group_id:None sources:[relu (call_function)] sinks:[convolution_2 (call_function), convolution_backward_2 (call_function), threshold_backward_1 (call_function)]
MultiSourceEdge convolution_2:0, size:597780, mem_space:None, tile_id:None group_id:None sources:[convolution_2 (call_function)] sinks:[native_batch_norm_2 (call_function), native_batch_norm_backward (call_function)]
MultiSourceEdge add__4:0, size:0, mem_space:None, tile_id:None group_id:None sources:[add__4 (call_function)] sinks:[relu_1 (call_function)]
MultiSourceEdge relu_1:0, size:597780, mem_space:None, tile_id:None group_id:None sources:[relu_1 (call_function)] sinks:[convolution_3 (call_function), convolution_4 (call_function), convolution_backward (call_function), convolution_backward_1 (call_function), threshold_backward (call_function)]
MultiSourceEdge convolution_3:0, size:597780, mem_space:None, tile_id:None group_id:None sources:[convolution_3 (call_function)] sinks:[cat (call_function)]
MultiSourceEdge convolution_4:0, size:597780, mem_space:None, tile_id:None group_id:None sources:[convolution_4 (call_function)] sinks:[cat (call_function)]
MultiSourceEdge cat:0, size:1195560, mem_space:None, tile_id:None group_id:None sources:[cat (call_function)] sinks:[sum_1 (call_function)]
MultiSourceEdge sum_1:0, size:4, mem_space:None, tile_id:None group_id:None sources:[sum_1 (call_function)] sinks:[ones_like (call_function)]
MultiSourceEdge ones_like:0, size:4, mem_space:None, tile_id:None group_id:None sources:[ones_like (call_function)] sinks:[convolution_backward_1 (call_function), convolution_backward (call_function)]
MultiSourceEdge add:0, size:597780, mem_space:None, tile_id:None group_id:None sources:[add (call_function)] sinks:[threshold_backward (call_function)]
MultiSourceEdge threshold_backward:0, size:597780, mem_space:None, tile_id:None group_id:None sources:[threshold_backward (call_function)] sinks:[native_batch_norm_backward (call_function), add_1 (call_function)]
MultiSourceEdge threshold_backward_1:0, size:597780, mem_space:None, tile_id:None group_id:None sources:[threshold_backward_1 (call_function)] sinks:[add_1 (call_function)]
MultiSourceEdge add_1:0, size:597780, mem_space:None, tile_id:None group_id:None sources:[add_1 (call_function)] sinks:[native_batch_norm_backward_1 (call_function), add_2 (call_function)]
MultiSourceEdge threshold_backward_2:0, size:597780, mem_space:None, tile_id:None group_id:None sources:[threshold_backward_2 (call_function)] sinks:[native_batch_norm_backward_2 (call_function)]
MultiSourceEdge getitem:0_opt, size:597780, mem_space:None, tile_id:None group_id:None sources:[native_batch_norm (call_function)] sinks:[convolution_1 (call_function), convolution_backward_3 (call_function), threshold_backward_2 (call_function)]
MultiSourceEdge getitem_1:0_opt, size:120, mem_space:None, tile_id:None group_id:None sources:[native_batch_norm (call_function)] sinks:[native_batch_norm_backward_2 (call_function)]
MultiSourceEdge getitem_3:0_opt, size:597780, mem_space:None, tile_id:None group_id:None sources:[native_batch_norm_1 (call_function)] sinks:[add__2 (call_function)]
MultiSourceEdge getitem_4:0_opt, size:120, mem_space:None, tile_id:None group_id:None sources:[native_batch_norm_1 (call_function)] sinks:[native_batch_norm_backward_1 (call_function)]
MultiSourceEdge getitem_6:0_opt, size:597780, mem_space:None, tile_id:None group_id:None sources:[native_batch_norm_2 (call_function)] sinks:[add__4 (call_function)]
MultiSourceEdge getitem_7:0_opt, size:120, mem_space:None, tile_id:None group_id:None sources:[native_batch_norm_2 (call_function)] sinks:[native_batch_norm_backward (call_function)]
MultiSourceEdge getitem_9:0_opt, size:597780, mem_space:None, tile_id:None group_id:None sources:[convolution_backward (call_function)] sinks:[add (call_function)]
MultiSourceEdge getitem_10:0_opt, size:8160, mem_space:None, tile_id:None group_id:None sources:[convolution_backward (call_function)] sinks:[add__19 (call_function)]
MultiSourceEdge getitem_12:0_opt, size:597780, mem_space:None, tile_id:None group_id:None sources:[convolution_backward_1 (call_function)] sinks:[add (call_function)]
MultiSourceEdge getitem_13:0_opt, size:8160, mem_space:None, tile_id:None group_id:None sources:[convolution_backward_1 (call_function)] sinks:[add__17 (call_function)]
MultiSourceEdge getitem_15:0_opt, size:597780, mem_space:None, tile_id:None group_id:None sources:[native_batch_norm_backward (call_function)] sinks:[convolution_backward_2 (call_function)]
MultiSourceEdge getitem_16:0_opt, size:120, mem_space:None, tile_id:None group_id:None sources:[native_batch_norm_backward (call_function)] sinks:[add__15 (call_function)]
MultiSourceEdge getitem_18:0_opt, size:597780, mem_space:None, tile_id:None group_id:None sources:[convolution_backward_2 (call_function)] sinks:[threshold_backward_1 (call_function)]
MultiSourceEdge getitem_19:0_opt, size:8160, mem_space:None, tile_id:None group_id:None sources:[convolution_backward_2 (call_function)] sinks:[add__13 (call_function)]
MultiSourceEdge getitem_21:0_opt, size:597780, mem_space:None, tile_id:None group_id:None sources:[native_batch_norm_backward_1 (call_function)] sinks:[convolution_backward_3 (call_function)]
MultiSourceEdge getitem_22:0_opt, size:120, mem_space:None, tile_id:None group_id:None sources:[native_batch_norm_backward_1 (call_function)] sinks:[add__11 (call_function)]
MultiSourceEdge getitem_24:0_opt, size:597780, mem_space:None, tile_id:None group_id:None sources:[convolution_backward_3 (call_function)] sinks:[threshold_backward_2 (call_function)]
MultiSourceEdge getitem_25:0_opt, size:8160, mem_space:None, tile_id:None group_id:None sources:[convolution_backward_3 (call_function)] sinks:[add__9 (call_function)]
MultiSourceEdge getitem_27:0_opt, size:597780, mem_space:None, tile_id:None group_id:None sources:[native_batch_norm_backward_2 (call_function)] sinks:[convolution_backward_4 (call_function)]
MultiSourceEdge getitem_28:0_opt, size:120, mem_space:None, tile_id:None group_id:None sources:[native_batch_norm_backward_2 (call_function)] sinks:[add__7 (call_function)]
MultiSourceEdge getitem_30:0_opt, size:597780, mem_space:None, tile_id:None group_id:None sources:[convolution_backward_4 (call_function)] sinks:[add_2 (call_function)]
MultiSourceEdge getitem_31:0_opt, size:8160, mem_space:None, tile_id:None group_id:None sources:[convolution_backward_4 (call_function)] sinks:[add__5 (call_function)]
""",
        )

    def testDropout(self):
        class DropoutModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(p=0.3)

            def forward(self, x):
                res = self.dropout(x)
                return res

        dropout_module = DropoutModule()
        input_shape = (32, 100)

        g, _ = self.importer.import_via_aotautograd(
            dropout_module,
            torch.randn(input_shape, requires_grad=True),
            mode="train",
            optimizer=True,
        )
        self.assertTrue(g.check_consistency())
        g.dump("/tmp/dropout.aotautograd.opt4ml", "svg")
        # dot = g.dump()
        # print(dot)
        print(str(g))
        self.assertEqual(
            str(g),
            """args_1 (placeholder)
empty_like (call_function)
mul (call_function)
sum_1 (call_function)
ones_like (call_function)
mul_1 (call_function)

MultiSourceEdge args_1:0, size:12800, mem_space:None, tile_id:None group_id:None sources:[args_1 (placeholder)] sinks:[empty_like (call_function), mul (call_function)]
MultiSourceEdge empty_like:0, size:12800, mem_space:None, tile_id:None group_id:None sources:[empty_like (call_function)] sinks:[mul (call_function), mul_1 (call_function)]
MultiSourceEdge mul:0, size:12800, mem_space:None, tile_id:None group_id:None sources:[mul (call_function)] sinks:[sum_1 (call_function)]
MultiSourceEdge sum_1:0, size:4, mem_space:None, tile_id:None group_id:None sources:[sum_1 (call_function)] sinks:[ones_like (call_function)]
MultiSourceEdge ones_like:0, size:4, mem_space:None, tile_id:None group_id:None sources:[ones_like (call_function)] sinks:[mul_1 (call_function)]
""",
        )

    def testStochasticDepth(self):
        class SDModule(torch.nn.Module):
            def forward(self, x):
                survival_rate = 0.8
                size = [x.shape[0]] + [1] * (x.ndim - 1)
                noise = torch.empty(size, dtype=x.dtype)
                noise = noise.bernoulli_(survival_rate)
                noise.div_(survival_rate)
                rslt = x * noise
                rslt += x
                return rslt

        sd_module = SDModule()
        input_shape = (32, 100)

        g, _ = self.importer.import_via_aotautograd(
            sd_module,
            torch.randn(input_shape, requires_grad=True),
            mode="train",
            optimizer=True,
        )
        self.assertTrue(g.check_consistency())
        print(str(g))
        self.assertEqual(
            str(g),
            """args_1 (placeholder)
empty (call_function)
mul (call_function)
add_ (call_function)
sum_1 (call_function)
ones_like (call_function)
mul_1 (call_function)
add (call_function)

MultiSourceEdge args_1:0, size:12800, mem_space:None, tile_id:None group_id:None sources:[args_1 (placeholder)] sinks:[mul (call_function), add_ (call_function), sum_1 (call_function)]
MultiSourceEdge empty:0, size:128, mem_space:None, tile_id:None group_id:None sources:[empty (call_function)] sinks:[mul (call_function), mul_1 (call_function)]
MultiSourceEdge mul:0, size:12800, mem_space:None, tile_id:None group_id:None sources:[mul (call_function)] sinks:[add_ (call_function)]
MultiSourceEdge add_:0, size:0, mem_space:None, tile_id:None group_id:None sources:[add_ (call_function)] sinks:[sum_1 (call_function)]
MultiSourceEdge sum_1:0, size:4, mem_space:None, tile_id:None group_id:None sources:[sum_1 (call_function)] sinks:[ones_like (call_function)]
MultiSourceEdge ones_like:0, size:4, mem_space:None, tile_id:None group_id:None sources:[ones_like (call_function)] sinks:[mul_1 (call_function), add (call_function)]
MultiSourceEdge mul_1:0, size:12800, mem_space:None, tile_id:None group_id:None sources:[mul_1 (call_function)] sinks:[add (call_function)]
""",
        )

    def testRemoveMetaNode(self):
        g = dataflow_graph.Graph()

        A = g.add_node(name="a")
        B = g.add_node(name="b")
        C = g.add_node(name="c")
        D = g.add_node(name="d")
        E = g.add_node(name="e")
        F = g.add_node(name="f")
        G = g.add_node(name="g")
        H = g.add_node(name="h", size=322)
        I = g.add_node(name="i")
        J = g.add_node(name="j")

        """
        Test deleting node B, whose parent is connected to another node
        A - B - D
         \\   //
            C
        """
        g.add_edge([A], [B], 123, name="e1")
        g.add_edge([A], [C], 456, name="e2")
        g.add_edge([B], [D], 123, name="e3")
        g.add_edge([C], [D], 789, name="e4")
        """
        Test deleting node E, that is connected to > 1 node
                H
             // |
        D - E - G
          \\  //
            F
        """
        g.add_edge([D], [E, F], 1034, name="e5")
        g.add_edge([E], [G, H], 1034, name="e6")
        g.add_edge([F], [G], 1089, name="e7")
        g.add_edge([G], [H], 1089, name="e8")
        """
        Test deleting node I, that has an incoming edge of size 0, but a neighbor node with size > 0
        H - I - J
        """
        g.add_edge([H], [I], 0, name="e9")
        g.add_edge([I], [J], 322, name="e10")

        g.canonicalize()

        g.sort()
        dot = g.dump()
        print(dot)
        self.assertEqual(
            dot,
            """digraph {
\tnode [shape=record]
\ta [label="<f0> a_0|<f1> a_1"]
\tb [label=b]
\tc [label=c]
\td [label=d]
\te [label=e]
\tf [label=f]
\tg [label=g]
\th [label="h (stateful_node)"]
\th_snk [label="h_snk (stateful_node_sink)"]
\ti [label=i]
\tj [label=j]
\ta:f0 -> b [label=123]
\ta:f1 -> c [label=456]
\tb -> d [label=123]
\tc -> d [label=789]
\td -> e [label=1034]
\td -> f [label=1034]
\te -> g [label=1034]
\te -> h [label=1034]
\tf -> g [label=1089]
\tg -> h [label=1089]
\th -> i [label=322]
\th -> h_snk [label=322]
\ti -> j [label=322]
}
""",
        )

        self.importer.bypass_and_delete_meta_node(g, B)

        g.sort()
        dot = g.dump()
        print(dot)
        self.assertEqual(
            dot,
            """digraph {
\tnode [shape=record]
\ta [label="<f0> a_0|<f1> a_1"]
\tc [label=c]
\td [label=d]
\te [label=e]
\tf [label=f]
\tg [label=g]
\th [label="h (stateful_node)"]
\th_snk [label="h_snk (stateful_node_sink)"]
\ti [label=i]
\tj [label=j]
\ta:f0 -> d [label=123]
\ta:f1 -> c [label=456]
\tc -> d [label=789]
\td -> e [label=1034]
\td -> f [label=1034]
\te -> g [label=1034]
\te -> h [label=1034]
\tf -> g [label=1089]
\tg -> h [label=1089]
\th -> i [label=322]
\th -> h_snk [label=322]
\ti -> j [label=322]
}
""",
        )

        self.importer.bypass_and_delete_meta_node(g, E)

        g.sort()
        dot = g.dump()
        print(dot)
        self.assertEqual(
            dot,
            """digraph {
\tnode [shape=record]
\ta [label="<f0> a_0|<f1> a_1"]
\tc [label=c]
\td [label=d]
\tf [label=f]
\tg [label=g]
\th [label="h (stateful_node)"]
\th_snk [label="h_snk (stateful_node_sink)"]
\ti [label=i]
\tj [label=j]
\ta:f0 -> d [label=123]
\ta:f1 -> c [label=456]
\tc -> d [label=789]
\td -> f [label=1034]
\td -> g [label=1034]
\td -> h [label=1034]
\tf -> g [label=1089]
\tg -> h [label=1089]
\th -> i [label=322]
\th -> h_snk [label=322]
\ti -> j [label=322]
}
""",
        )

        self.importer.bypass_and_delete_meta_node(g, I)

        g.sort()
        dot = g.dump()
        print(dot)
        self.assertEqual(
            dot,
            """digraph {
\tnode [shape=record]
\ta [label="<f0> a_0|<f1> a_1"]
\tc [label=c]
\td [label=d]
\tf [label=f]
\tg [label=g]
\th [label="h (stateful_node)"]
\th_snk [label="h_snk (stateful_node_sink)"]
\tj [label=j]
\ta:f0 -> d [label=123]
\ta:f1 -> c [label=456]
\tc -> d [label=789]
\td -> f [label=1034]
\td -> g [label=1034]
\td -> h [label=1034]
\tf -> g [label=1089]
\tg -> h [label=1089]
\th -> h_snk [label=322]
\th -> j [label=322]
}
""",
        )

    def testMLP(self):
        class MLPModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(125, 125)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                res = self.relu(self.linear(x))
                return res

        bn_module = MLPModule()
        input_shape = (32, 125)

        g, _ = self.importer.import_via_aotautograd(
            bn_module, torch.randn(input_shape), mode="train", optimizer=True
        )
        self.assertTrue(g.is_valid())
        # g.dump("/tmp/mlp.aotautograd.opt4ml", "svg")
        # dot = g.dump()
        # print(dot)
        print(str(g))
        self.assertEqual(
            str(g),
            """args_1 (placeholder)
params_2 (weight) [63000]
addmm (call_function)
relu (call_function)
sum_1 (call_function)
ones_like (call_function)
threshold_backward (call_function)
mm (call_function)
sum_2 (call_function)
sub (call_function)
sub_1 (call_function)

MultiSourceEdge args_1:0, size:16000, mem_space:None, tile_id:None group_id:None sources:[args_1 (placeholder)] sinks:[addmm (call_function), mm (call_function)]
MultiSourceEdge params_2:0, size:0, mem_space:None, tile_id:None group_id:None sources:[params_2 (weight) [63000]] sinks:[addmm (call_function), sub_1 (call_function), sub (call_function)]
MultiSourceEdge addmm:0, size:16000, mem_space:None, tile_id:None group_id:None sources:[addmm (call_function)] sinks:[relu (call_function)]
MultiSourceEdge relu:0, size:16000, mem_space:None, tile_id:None group_id:None sources:[relu (call_function)] sinks:[sum_1 (call_function), threshold_backward (call_function)]
MultiSourceEdge sum_1:0, size:4, mem_space:None, tile_id:None group_id:None sources:[sum_1 (call_function)] sinks:[ones_like (call_function)]
MultiSourceEdge ones_like:0, size:4, mem_space:None, tile_id:None group_id:None sources:[ones_like (call_function)] sinks:[threshold_backward (call_function)]
MultiSourceEdge threshold_backward:0, size:16000, mem_space:None, tile_id:None group_id:None sources:[threshold_backward (call_function)] sinks:[sum_2 (call_function), mm (call_function)]
MultiSourceEdge mm:0, size:62500, mem_space:None, tile_id:None group_id:None sources:[mm (call_function)] sinks:[sub (call_function)]
MultiSourceEdge sum_2:0, size:500, mem_space:None, tile_id:None group_id:None sources:[sum_2 (call_function)] sinks:[sub_1 (call_function)]
""",
        )

    def testBatchNorm(self):
        class BNConvModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # self.param = torch.nn.Parameter(torch.rand(3, 4))
                in_chan = 15
                out_chan = 10
                self.conv = torch.nn.Conv2d(in_chan, out_chan, 3)
                self.bn = torch.nn.BatchNorm2d(out_chan)
                self.conv2 = torch.nn.Conv2d(out_chan, 5, 3)

            def forward(self, x):
                res = self.conv2(self.bn(self.conv(x)))
                return res

        bn_module = BNConvModule()
        input_shape = (123, 15, 9, 9)

        g, _ = self.importer.import_via_aotautograd(
            bn_module, torch.randn(input_shape), mode="train", optimizer=True
        )
        self.assertTrue(g.is_valid())
        # g.dump("/tmp/batchnorm.aotautograd.opt4ml", "svg")
        # dot = g.dump()
        # print(dot)
        print(str(g))
        self.assertEqual(
            str(g),
            """args_1 (placeholder)
params_1 (weight) [5440]
params_3 (weight) [160]
params_5 (weight) [1820]
convolution (call_function)
native_batch_norm (call_function)
convolution_1 (call_function)
sum_1 (call_function)
ones_like (call_function)
convolution_backward (call_function)
native_batch_norm_backward (call_function)
convolution_backward_1 (call_function)
sub (call_function)
sub_2 (call_function)
sub_4 (call_function)

MultiSourceEdge args_1:0, size:597780, mem_space:None, tile_id:None group_id:None sources:[args_1 (placeholder)] sinks:[convolution (call_function), convolution_backward_1 (call_function)]
MultiSourceEdge params_1:0, size:0, mem_space:None, tile_id:None group_id:None sources:[params_1 (weight) [5440]] sinks:[convolution (call_function), convolution_backward_1 (call_function), sub (call_function)]
MultiSourceEdge params_3:0, size:0, mem_space:None, tile_id:None group_id:None sources:[params_3 (weight) [160]] sinks:[native_batch_norm (call_function), native_batch_norm_backward (call_function), sub_2 (call_function)]
MultiSourceEdge params_5:0, size:0, mem_space:None, tile_id:None group_id:None sources:[params_5 (weight) [1820]] sinks:[convolution_1 (call_function), convolution_backward (call_function), sub_4 (call_function)]
MultiSourceEdge convolution:0, size:241080, mem_space:None, tile_id:None group_id:None sources:[convolution (call_function)] sinks:[native_batch_norm (call_function), native_batch_norm_backward (call_function)]
MultiSourceEdge convolution_1:0, size:61500, mem_space:None, tile_id:None group_id:None sources:[convolution_1 (call_function)] sinks:[sum_1 (call_function)]
MultiSourceEdge sum_1:0, size:4, mem_space:None, tile_id:None group_id:None sources:[sum_1 (call_function)] sinks:[ones_like (call_function)]
MultiSourceEdge ones_like:0, size:4, mem_space:None, tile_id:None group_id:None sources:[ones_like (call_function)] sinks:[convolution_backward (call_function)]
MultiSourceEdge getitem:0_opt, size:241080, mem_space:None, tile_id:None group_id:None sources:[native_batch_norm (call_function)] sinks:[convolution_1 (call_function), convolution_backward (call_function)]
MultiSourceEdge getitem_1:0_opt, size:80, mem_space:None, tile_id:None group_id:None sources:[native_batch_norm (call_function)] sinks:[native_batch_norm_backward (call_function)]
MultiSourceEdge getitem_3:0_opt, size:241080, mem_space:None, tile_id:None group_id:None sources:[convolution_backward (call_function)] sinks:[native_batch_norm_backward (call_function)]
MultiSourceEdge getitem_4:0_opt, size:1820, mem_space:None, tile_id:None group_id:None sources:[convolution_backward (call_function)] sinks:[sub_4 (call_function)]
MultiSourceEdge getitem_6:0_opt, size:241080, mem_space:None, tile_id:None group_id:None sources:[native_batch_norm_backward (call_function)] sinks:[convolution_backward_1 (call_function)]
MultiSourceEdge getitem_7:0_opt, size:80, mem_space:None, tile_id:None group_id:None sources:[native_batch_norm_backward (call_function)] sinks:[sub_2 (call_function)]
MultiSourceEdge getitem_10:0_opt, size:5440, mem_space:None, tile_id:None group_id:None sources:[convolution_backward_1 (call_function)] sinks:[sub (call_function)]
""",
        )

    def testMultiSinkMultiSource(self):
        class MultiSinkModule(torch.nn.Module):
            def forward(self, x):
                y1 = x.relu()
                y2 = x.sin()
                z = y1 + y2
                z1, z2 = z.split(2)
                return z1 + z2

        simple_module = MultiSinkModule()
        input_shape = (3, 4)

        g = self.importer.import_from_torch(simple_module, torch.randn(input_shape))
        self.assertTrue(g.is_valid())

        # log dot files of dataflow and fx graphs
        # g.dump("/tmp/multiSinkMultiSource.via_fx.df.dot")
        torch.fx.passes.graph_drawer.FxGraphDrawer(
            self.importer.fx_trace, "multiSinkMultiSource"
        ).get_dot_graph().write("/tmp/multiSinkMultiSource.via_fx.fx.dot")

        dot = g.dump()
        print(dot)

    def testImportWithTimeProfile(self):
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=3)
                self.relu = torch.nn.ReLU(inplace=True)
                self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=3)
                self.conv3 = torch.nn.Conv2d(3, 3, kernel_size=3)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = x.relu()
                x = self.conv3(x)
                return x.sigmoid()

        simple_module = SimpleModule()
        input_shape = (1, 3, 10, 10)

        if True:
            print("Import using default method")
            g = self.importer.import_from_torch(
                simple_module, torch.randn(input_shape), profile=["time"]
            )
            self.assertTrue(g.is_valid())

            # log dot files of dataflow and fx graphs
            # g.dump("/tmp/importWithTimeProfile.default.df.dot")

            dot = g.dump()
            print(dot)

            # verify that each node has been profiled
            for node in g.nodes.values():
                assert (
                    node.time and node.time > 0
                ), f"Profiled execution time of node {node.name} is {node.time} but should be greater than zero."

        if True:
            print("Import using aotautograd method")
            g, _ = self.importer.import_via_aotautograd(
                simple_module,
                torch.randn(input_shape),
                mode="train",
                optimizer=torch.optim.SGD(simple_module.parameters(), lr=0.1),
                profile=["time"],
            )
            self.assertTrue(g.is_valid())

            # log dot files of dataflow and fx graphs
            # g.dump("/tmp/importWithTimeProfile.aotautograd.df.dot")

            dot = g.dump()
            print(dot)

            # when using torch.optim.SGD, inplace ops cannot be profiled
            for node in g.nodes.values():
                assert (
                    (node.time and node.time > 0)
                    or node.name == "add_"
                    or node.name.startswith("add__")
                ), f"Profiled execution time of node {node.name} is {node.time} but should be greater than zero as it is not an `add_` in place node."

        if True:
            print("Import using acctracer method")
            g = self.importer.import_via_acc_tracer(
                simple_module, torch.randn(input_shape), profile=["time"]
            )
            self.assertTrue(g.is_valid())

            # log dot files of dataflow and fx graphs
            # g.dump("/tmp/importWithTimeProfile.acc_tracer.df.dot")

            dot = g.dump()
            print(dot)

            # verify that each node has been profiled
            for node in g.nodes.values():
                assert (
                    node.time and node.time > 0
                ), f"Profiled execution time of node {node.name} is {node.time} but should be greater than zero."

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Memory profiling currently only works with CUDA",
    )
    def testImportWithMemoryProfile(self):
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=3)
                self.relu = torch.nn.ReLU(inplace=True)
                self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=3)
                self.conv3 = torch.nn.Conv2d(3, 3, kernel_size=3)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = x.relu()
                x = self.conv3(x)
                return x.sigmoid()

        simple_module = SimpleModule()

        input_shape = (1, 3, 10, 10)
        input = torch.randn(input_shape)

        # move to GPU to enable CUDA memory profile
        simple_module.cuda()
        input = input.cuda()

        if True:
            g = self.importer.import_from_torch(
                simple_module, input, profile=["time", "memory"]
            )
            self.assertTrue(g.is_valid())

            # log dot files of dataflow and fx graphs
            # g.dump("/tmp/importWithMemoryProfile.default.df.dot")

            dot = g.dump()
            print(dot)

            # TODO: after loading memory stats into each node, verify some memory stats rather than time
            # verify that each node has been profiled
            for node in g.nodes.values():
                assert (
                    node.time and node.time > 0
                ), f"Profiled execution time of node {node.name} is {node.time} but should be greater than zero."

        if True:
            g, _ = self.importer.import_via_aotautograd(
                simple_module,
                input,
                mode="train",
                optimizer=torch.optim.SGD(simple_module.parameters(), lr=0.1),
                profile=["time", "memory"],
            )
            self.assertTrue(g.is_valid())

            # log dot files of dataflow and fx graphs
            # g.dump("/tmp/importWithMemoryProfile.aotautograd.df.dot")

            dot = g.dump()
            print(dot)

            # TODO: after loading memory stats into each node, verify some memory stats rather than time
            # when using torch.optim.SGD, inplace ops cannot be profiled
            for node in g.nodes.values():
                assert (
                    (node.time and node.time > 0)
                    or node.name == "add_"
                    or node.name.startswith("add__")
                ), f"Profiled execution time of node {node.name} is {node.time} but should be greater than zero as it is not an `add_` in place node."

        if True:
            g = self.importer.import_via_acc_tracer(
                simple_module, input, profile=["time", "memory"]
            )
            self.assertTrue(g.is_valid())

            # log dot files of dataflow and fx graphs
            # g.dump("/tmp/importWithMemoryProfile.acc_tracer.df.dot")

            dot = g.dump()
            print(dot)

            # TODO: after loading memory stats into each node, verify some memory stats rather than time
            # verify that each node has been profiled
            for node in g.nodes.values():
                assert (
                    node.time and node.time > 0
                ), f"Profiled execution time of node {node.name} is {node.time} but should be greater than zero."

        if True:
            # FIXME: profiling currently fails with functorch
            pass
