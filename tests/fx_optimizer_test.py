import unittest

import torch

from olla.torch import fx_optimizer, torch_graph_importer


class MaxCutTest(unittest.TestCase):
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
        print(initial_result)

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
        print(final_result)
        self.assertEqual(initial_result, final_result)
