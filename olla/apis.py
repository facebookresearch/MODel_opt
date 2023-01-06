import torch
import torchvision

from olla import training_graph_optimizer, utils
from olla.torch import torch_graph_importer
from olla.torch.fx_optimizer import FXOptimizer

def extract_tensors_and_params(results):
    outputs = []
    params = []
    # buffers = [] # TODO: support buffers
    for res in results:
        if isinstance(res, torch.nn.Parameter):
            params.append(res)
        else:
            outputs.append(res)

    return outputs, params

# TODO: support evaluation
# TODO: add options to enable/disable node ordering, defragmentation, etc.
def optimize(model, inputs, loss_fn, optimizer):
    # import fx graph and data flow graph
    importer = torch_graph_importer.TorchGraphImporter()
    (
        g,
        pytorch_node_order,
        fx_graph,
        fx_to_df_map,
    ) = importer.import_via_aotautograd(
        model,
        inputs,
        mode="train",
        loss_fn=loss_fn,
        optimizer=optimizer,
        model_return_output=True,
        return_node_ordering=True,
        return_fx_graph=True,
    )

    # verify and post process data flow graph
    assert g.is_valid()
    g.canonicalize()
    g.constrain_weight_updates()
    g.constrain_tensor_generators()

    # recompile fx graph
    fx_graph.recompile()

    # run ILP solver on dataflow graph
    s = training_graph_optimizer.Scheduler(g, rel_stop=0.005, timeout_s=1800)
    summary, schedule, mem_loc = s.ComputeOptimalSchedule(
        allow_swaps=False,
        max_spills=0,
    )

    assert utils.validate_timeline(schedule)
    assert utils.validate_node_ordering(g, schedule)

    # export ILP solution to fx graph
    node_order_optimized = utils.extract_node_ordering(g, schedule)
    fx_opt = FXOptimizer(fx_graph, fx_to_df_map)
    fx_opt.Reorder(node_order_optimized)
    fx_graph_opt = fx_opt.fx_trace

    # wrap fx graph in torch.nn.Module
    class OptimizedModel(torch.nn.Module):
        def __init__(self, model, fx_graph_opt):
            super().__init__()
            self.model = model
            self.fx_graph_opt = fx_graph_opt

        def forward(self, x):
            with torch.no_grad():
                torch.manual_seed(0)
                # TODO: what about buffers?
                output, new_params = self.fx_graph_opt.forward(
                    (x, ),
                    params=dict(self.model.named_parameters()),
                    buffers=dict(self.model.named_buffers()),
                )

                return output
                    

    model_opt = OptimizedModel(model, fx_graph_opt)
    return model_opt
