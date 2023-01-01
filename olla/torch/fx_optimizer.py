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
        added_fx_nodes: Set[torch.fx.Node] = set()
        prev_fx_node: torch.fx.Node = self.fx_trace.graph._root

        def addDependentNodes(fx_node, prev_fx_node):
            for fx_node_input in fx_node.all_input_nodes:
                if fx_node_input not in added_fx_nodes:
                    prev_fx_node = addDependentNodes(fx_node_input, prev_fx_node)
                    #print("adding in addDependentNodes: ", fx_node_input.name)
                    prev_fx_node.append(fx_node_input)
                    added_fx_nodes.add(fx_node_input)
                    prev_fx_node = prev_fx_node.next
            return prev_fx_node

        def addNode(fx_node, prev_fx_node):
            if prev_fx_node.next != fx_node:
                if fx_node in added_fx_nodes:
                    print(
                        f"WARNING: trying to add {fx_node.name} that is already added. Will ignore but need to debug why."
                    )
                    pass
                else:
                    # ensure that any nodes that fx_node requires are added first
                    prev_fx_node = addDependentNodes(fx_node, prev_fx_node)
                    # add fx_node
                    prev_fx_node.append(fx_node)
                    # print("adding in addNode: ", fx_node.name)
                    added_fx_nodes.add(fx_node)
            added_fx_nodes.add(prev_fx_node)
            prev_fx_node = prev_fx_node.next
            return prev_fx_node

        # add params, args, and buffer nodes first to ensure they have the same shapes as original graph
        args_id = 1
        while True:
            args_name = f"args_{args_id}"
            args_fx_node = find_fx_node(self.fx_trace.graph, args_name)
            if args_fx_node:
                prev_fx_node = addNode(args_fx_node, prev_fx_node)
                args_id += 1
            else:
                break

        params_id = 1
        while True:
            params_name = f"params_{params_id}"
            params_fx_node = find_fx_node(self.fx_trace.graph, params_name)
            if params_fx_node:
                prev_fx_node = addNode(params_fx_node, prev_fx_node)
                params_id += 1
            else:
                break

        buffers_id = 1
        while True:
            buffers_name = f"buffers_{buffers_id}"
            buffers_fx_node = find_fx_node(self.fx_trace.graph, buffers_name)
            if buffers_fx_node:
                prev_fx_node = addNode(buffers_fx_node, prev_fx_node)
                buffers_id += 1
            else:
                break

        # TODO: create a map between df edges to fx nodes? or df input to ops to fx nodes?

        # add other nodes in order of schedule
        for df_node, _ in node_order.items():
            fx_node: torch.fx.Node = self.df_to_fx_map[df_node]
            prev_fx_node = addNode(fx_node, prev_fx_node)

        self.fx_trace.graph.lint()
        self.fx_trace.recompile()

# TODO: move to utils file
def find_fx_node(fx_graph, name):
    for node in fx_graph.nodes:
        if node.name == name:
            return node
    return None
