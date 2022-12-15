import torch


class FXOptimizer:
    def __init__(self, fx_trace, fx_to_df_map):
        self.fx_trace = fx_trace
        self.fx_to_df_map = fx_to_df_map

        # create reverse map
        self.df_to_fx_map = {}
        for fx_node, df_node in fx_to_df_map.items():
            self.df_to_fx_map[df_node] = fx_node

    def Reorder(self, node_order):
        prev_fx_node = self.fx_trace.graph._root
        for df_node, _ in node_order.items():
            fx_node = self.df_to_fx_map[df_node]
            if prev_fx_node.next != fx_node:
                prev_fx_node.append(fx_node)
            prev_fx_node = prev_fx_node.next

        self.fx_trace.graph.lint()
        self.fx_trace.recompile()
