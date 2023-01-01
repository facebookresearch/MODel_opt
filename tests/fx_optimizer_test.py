import unittest

import torch
import torchvision

from olla import training_graph_optimizer, utils
from olla.torch import torch_graph_importer
from olla.torch.fx_optimizer import FXOptimizer


class FXOptimizerTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def testSimpleModule(self):
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 5)
                self.linear2 = torch.nn.Linear(4, 5)

            def forward(self, x):
                return self.linear1(x) + self.linear2(x)

        module = SimpleModule()
        importer = torch_graph_importer.TorchGraphImporter()
        input_tensor = torch.randn((3, 4))
        (
            g,
            pytorch_node_order,
            fx_graph,
            fx_to_df_map,
        ) = importer.import_via_aotautograd(
            module, input_tensor, return_node_ordering=True, return_fx_graph=True
        )
        self.assertTrue(g.is_valid())

        fx_graph.recompile()
        initial_result = fx_graph.forward(
            (input_tensor,),
            params=dict(module.named_parameters()),
            buffers=dict(module.named_buffers()),
        )
        # print(initial_result)

        node_order = str([node for node in fx_graph.graph.nodes])
        # print(f"NODES INITIAL = {node_order}", flush=True)
        self.assertEqual(
            node_order,
            "[args_1, params_1, params_2, params_3, params_4, t, addmm, t_1, addmm_1, add, sum_1, ones_like, expand, t_2, mm, t_3, sum_2, view, t_4, t_5, mm_1, t_6, sum_3, view_1, t_7, output]",
        )

        nodes = [node for node in fx_graph.graph.nodes]
        tmp1 = nodes[5]
        tmp2 = nodes[6]
        nodes[5] = nodes[7]
        nodes[6] = nodes[8]
        nodes[7] = tmp1
        nodes[8] = tmp2

        prev = fx_graph.graph._root
        for i in range(len(nodes)):
            n = nodes[i]
            if prev.next != n:
                prev.append(n)
            prev = prev.next

        node_order = str([node for node in fx_graph.graph.nodes])
        # print(f"NODES FINAL = {node_order}", flush=True)
        self.assertEqual(
            node_order,
            "[args_1, params_1, params_2, params_3, params_4, t_1, addmm_1, t, addmm, add, sum_1, ones_like, expand, t_2, mm, t_3, sum_2, view, t_4, t_5, mm_1, t_6, sum_3, view_1, t_7, output]",
        )

        fx_graph.graph.lint()
        fx_graph.recompile()
        final_result = fx_graph.forward(
            (input_tensor,),
            params=dict(module.named_parameters()),
            buffers=dict(module.named_buffers()),
        )
        # print(final_result)
        self.assertEqual(initial_result, final_result)

    def testSimpleEvalSchedule(self):
        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                x1 = x.relu()
                x2 = x.sin()
                return x1 + x2

        module = SimpleModule()
        importer = torch_graph_importer.TorchGraphImporter()
        input_tensor = torch.randn((32, 3, 224, 224))
        (
            g,
            pytorch_node_order,
            fx_graph,
            fx_to_df_map,
        ) = importer.import_via_aotautograd(
            module,
            input_tensor,
            mode="eval",
            return_node_ordering=True,
            return_fx_graph=True,
        )
        self.assertTrue(g.is_valid())
        g.canonicalize()
        g.constrain_weight_updates()
        g.constrain_tensor_generators()
        self.assertTrue(g.is_valid())

        fx_graph.recompile()
        initial_result = fx_graph.forward(
            (input_tensor,),
            params=dict(module.named_parameters()),
            buffers=dict(module.named_buffers()),
        )
        # print(f"initial_result: {initial_result}")

        node_order = str([node for node in fx_graph.graph.nodes])
        # print(f"NODES INITIAL = {node_order}", flush=True)

        s = training_graph_optimizer.Scheduler(g)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            allow_swaps=False,
            max_spills=0,
        )
        # print(f"SCHEDULER = {schedule}")
        node_order_optimized = utils.extract_node_ordering(g, schedule)
        # print(f"node_order_optimized: {node_order_optimized}")
        fx_opt = FXOptimizer(fx_graph, fx_to_df_map)
        fx_opt.Reorder(node_order_optimized)
        fx_graph_opt = fx_opt.fx_trace
        final_result = fx_graph_opt.forward(
            (input_tensor,),
            params=dict(module.named_parameters()),
            buffers=dict(module.named_buffers()),
        )
        # print(f"final_result: {final_result}")
        self.assertTrue(torch.allclose(initial_result, final_result))

    def testSimpleTrainSchedule(self):
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 5)
                self.linear2 = torch.nn.Linear(4, 5)

            def forward(self, x):
                return self.linear1(x).relu() + self.linear2(x).sigmoid()

        module = SimpleModule()
        importer = torch_graph_importer.TorchGraphImporter()
        input_tensor = torch.randn((3, 4), requires_grad=True)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        fx_graph: torch.fx.GraphModule = None
        (
            g,
            pytorch_node_order,
            fx_graph,
            fx_to_df_map,
        ) = importer.import_via_aotautograd(
            module,
            input_tensor,
            mode="train",
            optimizer=optimizer,
            return_node_ordering=True,
            return_fx_graph=True,
        )
        self.assertTrue(g.is_valid())
        g.canonicalize()
        g.constrain_weight_updates()
        g.constrain_tensor_generators()
        self.assertTrue(g.is_valid())

        fx_graph.recompile()
        with torch.no_grad():
            initial_result = fx_graph.forward(
                (input_tensor,),
                params=dict(module.named_parameters()),
                buffers=dict(module.named_buffers()),
            )
        # print(fx_graph)
        # print(f"initial_result: {initial_result}")

        node_order = str([node for node in fx_graph.graph.nodes])
        # print(f"NODES INITIAL = {node_order}", flush=True)

        s = training_graph_optimizer.Scheduler(g)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            allow_swaps=False,
            max_spills=0,
        )
        # print(f"SCHEDULER = {schedule}")
        node_order_optimized = utils.extract_node_ordering(g, schedule)
        # print(f"node_order_optimized: {node_order_optimized}")
        fx_opt = FXOptimizer(fx_graph, fx_to_df_map)
        fx_opt.Reorder(node_order_optimized)
        fx_graph_opt = fx_opt.fx_trace
        # print(f"fx_graph_opt: \n{fx_graph_opt}")
        with torch.no_grad():
            final_result = fx_graph_opt.forward(
                (input_tensor,),
                params=dict(module.named_parameters()),
                buffers=dict(module.named_buffers()),
            )
        # print(f"final_result: {final_result}")
        for initial_tensor, final_tensor in zip(initial_result, final_result):
            self.assertTrue(torch.allclose(initial_tensor, final_tensor))

    def testAlexNetEvalSchedule(self):
        model = torchvision.models.alexnet()
        importer = torch_graph_importer.TorchGraphImporter()
        input_tensor = torch.randn((32, 3, 224, 224))
        (
            g,
            pytorch_node_order,
            fx_graph,
            fx_to_df_map,
        ) = importer.import_via_aotautograd(
            model,
            input_tensor,
            mode="eval",
            return_node_ordering=True,
            return_fx_graph=True,
        )
        # print(f"fx_graph: \n{fx_graph}")
        self.assertTrue(g.is_valid())
        g.canonicalize()
        g.constrain_weight_updates()
        g.constrain_tensor_generators()
        self.assertTrue(g.is_valid())

        fx_graph.recompile()
        with torch.no_grad():
            torch.manual_seed(0)
            initial_result = fx_graph.forward(
                (input_tensor,),
                params=dict(model.named_parameters()),
                buffers=dict(model.named_buffers()),
            )
        # print(f"initial_result: {initial_result}")

        node_order = str([node for node in fx_graph.graph.nodes])
        # print(f"NODES INITIAL = {node_order}", flush=True)

        s = training_graph_optimizer.Scheduler(g, rel_stop=0.005, timeout_s=1800)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            allow_swaps=False,
            max_spills=0,
        )
        # print(f"SCHEDULER = {schedule}")
        assert utils.validate_timeline(schedule)
        assert utils.validate_node_ordering(g, schedule)
        # TODO: call Benchmarks.run_node_ordering() to reduce redundancy?

        node_order_optimized = utils.extract_node_ordering(g, schedule)
        # print(f"node_order_optimized: {node_order_optimized}")
        fx_opt = FXOptimizer(fx_graph, fx_to_df_map)
        fx_opt.Reorder(node_order_optimized)
        fx_graph_opt = fx_opt.fx_trace
        # print(f"fx_graph_opt: \n{fx_graph_opt}")
        with torch.no_grad():
            torch.manual_seed(0)
            final_result = fx_graph_opt.forward(
                (input_tensor,),
                params=dict(model.named_parameters()),
                buffers=dict(model.named_buffers()),
            )
        # print(f"final_result: {final_result}")
        self.assertTrue(torch.allclose(initial_result, final_result))

    def testAlexNetTrainSchedule(self):
        model = torchvision.models.alexnet()
        importer = torch_graph_importer.TorchGraphImporter()
        input_tensor = torch.randn((32, 3, 224, 224), requires_grad = True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = torch.nn.MSELoss()
        (
            g,
            pytorch_node_order,
            fx_graph,
            fx_to_df_map,
        ) = importer.import_via_aotautograd(
            model,
            input_tensor,
            mode="train",
            loss_fn=loss_fn,
            optimizer=optimizer,
            return_node_ordering=True,
            return_fx_graph=True,
        )
        # print(f"fx_graph: \n{fx_graph}")
        self.assertTrue(g.is_valid())
        g.canonicalize()
        g.constrain_weight_updates()
        g.constrain_tensor_generators()
        self.assertTrue(g.is_valid())

        fx_graph.recompile()
        with torch.no_grad():
            torch.manual_seed(0)
            initial_result = fx_graph.forward(
                (input_tensor,),
                params=dict(model.named_parameters()),
                buffers=dict(model.named_buffers()),
            )
        # print(f"initial_result: {initial_result}")

        node_order = str([node for node in fx_graph.graph.nodes])
        # print(f"NODES INITIAL = {node_order}", flush=True)

        s = training_graph_optimizer.Scheduler(g, rel_stop=0.005, timeout_s=1800)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            allow_swaps=False,
            max_spills=0,
        )
        # print(f"SCHEDULER = {schedule}")
        assert utils.validate_timeline(schedule)
        assert utils.validate_node_ordering(g, schedule)
        # TODO: call Benchmarks.run_node_ordering() to reduce redundancy?

        node_order_optimized = utils.extract_node_ordering(g, schedule)
        # print(f"node_order_optimized: {node_order_optimized}")
        fx_opt = FXOptimizer(fx_graph, fx_to_df_map)
        fx_opt.Reorder(node_order_optimized)
        fx_graph_opt = fx_opt.fx_trace
        # print(f"fx_graph_opt: \n{fx_graph_opt}")
        with torch.no_grad():
            torch.manual_seed(0)
            final_result = fx_graph_opt.forward(
                (input_tensor,),
                params=dict(model.named_parameters()),
                buffers=dict(model.named_buffers()),
            )
        # print(f"final_result: {final_result}")
        for initial_tensor, final_tensor in zip(initial_result, final_result):
            # print("initial_tensor: \n", initial_tensor)
            # print("final_tensor: \n", final_tensor)
            self.assertTrue(torch.allclose(initial_tensor, final_tensor, equal_nan=False))

    def testVGG11BNNetEvalSchedule(self):
        model = torchvision.models.vgg11_bn()
        importer = torch_graph_importer.TorchGraphImporter()
        input_tensor = torch.randn((32, 3, 224, 224), requires_grad = True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = torch.nn.MSELoss()
        (
            g,
            pytorch_node_order,
            fx_graph,
            fx_to_df_map,
        ) = importer.import_via_aotautograd(
            model,
            input_tensor,
            mode="eval",
            loss_fn=loss_fn,
            optimizer=optimizer,
            return_node_ordering=True,
            return_fx_graph=True,
        )
        # print(f"fx_graph: \n{fx_graph}")
        self.assertTrue(g.is_valid())
        g.canonicalize()
        g.constrain_weight_updates()
        g.constrain_tensor_generators()
        self.assertTrue(g.is_valid())

        fx_graph.recompile()
        with torch.no_grad():
            torch.manual_seed(0)
            initial_result = fx_graph.forward(
                (input_tensor,),
                params=dict(model.named_parameters()),
                buffers=dict(model.named_buffers()),
            )
        # print(f"initial_result: {initial_result}")

        node_order = str([node for node in fx_graph.graph.nodes])
        # print(f"NODES INITIAL = {node_order}", flush=True)

        s = training_graph_optimizer.Scheduler(g, rel_stop=0.005, timeout_s=1800)
        summary, schedule, mem_loc = s.ComputeOptimalSchedule(
            allow_swaps=False,
            max_spills=0,
        )
        # print(f"SCHEDULER = {schedule}")
        assert utils.validate_timeline(schedule)
        assert utils.validate_node_ordering(g, schedule)
        # TODO: call Benchmarks.run_node_ordering() to reduce redundancy?

        node_order_optimized = utils.extract_node_ordering(g, schedule)
        # print(f"node_order_optimized: {node_order_optimized}")
        fx_opt = FXOptimizer(fx_graph, fx_to_df_map)
        fx_opt.Reorder(node_order_optimized)
        fx_graph_opt = fx_opt.fx_trace
        # print(f"fx_graph_opt: \n{fx_graph_opt}")
        with torch.no_grad():
            torch.manual_seed(0)
            final_result = fx_graph_opt.forward(
                (input_tensor,),
                params=dict(model.named_parameters()),
                buffers=dict(model.named_buffers()),
            )
        # print(f"final_result: {final_result}")
        for initial_tensor, final_tensor in zip(initial_result, final_result):
            # print("initial_tensor: \n", initial_tensor)
            # print("final_tensor: \n", final_tensor)
            self.assertTrue(torch.allclose(initial_tensor, final_tensor, equal_nan=False))
