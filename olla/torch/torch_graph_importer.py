
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Iterable

# TODO: import properly from third-party library
# import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_tracer
import olla.acc_tracer.acc_tracer as acc_tracer

import numpy as np
import pandas as pd
import torch
import torch.fx
import torch.nn.utils._stateless as stateless
from functorch import make_fx
from functorch.compile import compiled_function, default_partition
from olla import dataflow_graph
from olla.torch import fx_profiler, spill_profiler
from torch.fx.passes.shape_prop import ShapeProp

# By default torch.fx.Tracer treats torch.nn.Module nodes as leaf nodes and hence doesn't trace them to obtain weight or bias nodes.
# This class overrides this in order to enable us to obtain separate nodes for weights and biases.
# TODO: add option for user to pass their own `is_leaf_module` function
class DeepTracer(torch.fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return False


class TorchGraphImporter:
    def __init__(self):
        self.fx: torch.fx.GraphModule = None

    # Import graph from a PyTorch model
    def import_from_torch(
        self,
        model,
        *inputs,
        mode="eval",
        method="fx",
        optimizer=None,
        profile=None,
        warm_up_iters=0,
        profile_iters=1,
        return_node_ordering=False,
    ):
        self.assert_type(mode)
        if method == "fx":
            return self.import_via_fx(
                model,
                *inputs,
                mode=mode,
                profile=profile,
                warm_up_iters=warm_up_iters,
                profile_iters=profile_iters,
                return_node_ordering=return_node_ordering,
            )
        elif method == "functorch":
            return self.import_via_functorch(
                model,
                *inputs,
                mode=mode,
                profile=profile,
                warm_up_iters=warm_up_iters,
                profile_iters=profile_iters,
                return_node_ordering=return_node_ordering,
            )
        elif method == "aotautograd":
            return self.import_via_aotautograd(
                model,
                *inputs,
                mode=mode,
                optimizer=optimizer,
                profile=profile,
                warm_up_iters=warm_up_iters,
                profile_iters=profile_iters,
                return_node_ordering=return_node_ordering,
            )
        elif method == "acc_tracer":
            return self.import_via_acc_tracer(
                model,
                *inputs,
                mode=mode,
                profile=profile,
                warm_up_iters=warm_up_iters,
                profile_iters=profile_iters,
                return_node_ordering=return_node_ordering,
            )
        else:
            raise ValueError("unsupported import type `", method, "`")

    def assert_type(self, mode):
        assert mode in ["eval", "train"], "Invalid mode provided"

    def import_via_fx(
        self,
        model,
        *inputs,
        mode="eval",
        tracer_class=DeepTracer,
        profile=None,
        warm_up_iters=0,
        profile_iters=1,
        return_node_ordering=False,
    ):
        self.assert_type(mode)
        assert mode == "eval", "Currently `import_via_fx` only supports eval mode"

        # tracer_class: by default is our defined DeepTracer so that we can create nodes for weights.
        # otherwise we can use torch.fx.Tracer (won't trace weights) or any custom tracer
        fx_trace = torch.fx.GraphModule(model, tracer_class().trace(model))

        return self.import_from_fx(
            fx_trace,
            *inputs,
            profile=profile,
            warm_up_iters=warm_up_iters,
            profile_iters=profile_iters,
            return_node_ordering=return_node_ordering,
        )

    def import_via_functorch(
        self,
        model,
        *inputs,
        mode="train",
        profile=None,
        warm_up_iters=0,
        profile_iters=1,
        return_node_ordering=False,
    ):
        self.assert_type(mode)
        # obtain the forward and backward graphs of the model
        fx_trace_fwd = None
        fx_trace_bwd = None
        fx_trace_full = None

        def fwd_compiler(fx_trace, example_input):
            nonlocal fx_trace_fwd
            fx_trace_fwd = fx_trace
            return fx_trace

        def bwd_compiler(fx_trace, example_input):
            nonlocal fx_trace_bwd
            fx_trace_bwd = fx_trace
            return fx_trace

        def partition(fx_trace, joint_input):
            nonlocal fx_trace_full
            fx_trace_full = fx_trace
            return default_partition(fx_trace, joint_input)

        compiled_function(model, fwd_compiler, bwd_compiler, partition)(*inputs)

        if mode == "train":
            fx_trace = fx_trace_full
            outputs = model(*inputs)
            return self.import_from_fx(
                fx_trace,
                *inputs,
                *outputs,
                profile=profile,
                warm_up_iters=warm_up_iters,
                profile_iters=profile_iters,
                return_node_ordering=return_node_ordering,
            )
        elif mode == "eval":
            fx_trace = fx_trace_fwd
            return self.import_from_fx(
                fx_trace,
                *inputs,
                profile=profile,
                warm_up_iters=warm_up_iters,
                profile_iters=profile_iters,
                return_node_ordering=return_node_ordering,
            )
        else:
            raise ValueError("unsupported import mode `", mode, "`")

    def import_via_aotautograd(
        self,
        model,
        *inputs,
        mode="train",
        optimizer=None,
        loss_fn=None,
        profile=None,
        model_return_output=False,
        warm_up_iters=0,
        profile_iters=1,
        return_node_ordering=True,
        return_fx_graph=False,
        verbose=True,
    ):
        self.assert_type(mode)
        # TODO: if mode is train, assert that inputs have `required_grad = True`
        def fn_model_wrapper(args, params, buffers):
            if not isinstance(args, Iterable):
                args = [args]
            params_and_buffers = {**params, **buffers}
            out = stateless.functional_call(model, params_and_buffers, args)
            if mode == "eval":
                return out
            elif mode == "train":
                if loss_fn:
                    criterion = loss_fn
                    target = torch.zeros_like(out)
                    loss = criterion(out, target)
                    loss.backward()
                else:
                    out.sum().backward()

                if optimizer is True:
                    return [p - p.grad for p in params.values()]
                    # return [p.sub_(1e-4 * p.grad) for p in params.values()]
                elif optimizer is not None:
                    optimizer.step()

                if model_return_output:
                    return out, list(params.values())
                else:
                    return list(params.values())

        def detach_decomposition(x):
            return x

        fx_trace = make_fx(
            fn_model_wrapper,
            decomposition_table={torch.ops.aten.detach.default: detach_decomposition},
        )(inputs, dict(model.named_parameters()), dict(model.named_buffers()))

        # print("SYMBOLIC TRACE: \n" + str(fx_trace.graph))
        # print("CODE FOR TRACE: \n" + str(fx_trace.code))

        # aotautograd performs its own shape inference so we don't need to perform ours
        return self.import_from_fx(
            fx_trace,
            *inputs,
            *dict(model.named_parameters()).values(),
            *dict(model.named_buffers()).values(),
            cleanup=True,
            shape_inference=False,
            profile=profile,
            warm_up_iters=warm_up_iters,
            profile_iters=profile_iters,
            return_node_ordering=return_node_ordering,
            return_fx_graph=return_fx_graph,
            verbose=verbose,
        )

    def import_via_acc_tracer(
        self,
        model,
        *inputs,
        mode="eval",
        profile=None,
        warm_up_iters=0,
        profile_iters=1,
        return_node_ordering=False,
    ):
        self.assert_type(mode)
        assert (
            mode == "eval"
        ), "Currently `import_via_acc_tracer` only supports eval mode"

        self.fx_trace = acc_tracer.trace(model, inputs, use_acc_shape_inference=True)
        # acc_tracer executes its own shape propagation so we can import from fx without performing our shape propagation
        return self.import_from_fx(
            self.fx_trace,
            *inputs,
            shape_inference=False,
            profile=profile,
            warm_up_iters=warm_up_iters,
            profile_iters=profile_iters,
            return_node_ordering=return_node_ordering,
        )

    # Import graph from FX IR
    def import_from_fx(
        self,
        fx_trace,
        *inputs,
        cleanup=False,
        shape_inference=True,
        profile=None,
        warm_up_iters=0,
        profile_iters=1,
        return_node_ordering=False,
        return_fx_graph=False,
        verbose=False,
    ):
        # TODO: we are profiling the model twice: once to measure execution time and once for shape propagation. So we need to somehow combine both ShapeProp into our custom ProfilingInterpreter so that we only profileo once
        # run profiler
        if profile:
            assert isinstance(profile, list) and (
                "time" in profile or "memory" in profile
            ), "profile argument should either be None or a list with either 'time' or 'memory' string elements"
            profiler = fx_profiler.ProfilingInterpreter(
                fx_trace,
                profile_time="time" in profile,
                profile_memory="memory" in profile,
                warm_up_iters=warm_up_iters,
                profile_iters=profile_iters,
            )
            with torch.no_grad():
                profiler.run(*inputs)

        # run shape propagation on fx_traces
        if shape_inference:
            assert (
                len(inputs) > 0
            ), "Need to have at least one input to run shape inference"
            ShapeProp(fx_trace).propagate(*inputs)
        # save fx_trace so that it can be accessed externally
        self.fx_trace = fx_trace
        fx_graph = fx_trace.graph

        # create dataflow graph
        df_graph = dataflow_graph.Graph()

        # load nodes
        fx2df_node_map = {}
        for fx_node in fx_graph.nodes:
            # add node, unless it's the fake output node
            op_type = (
                "weight"
                if fx_node.op == "get_attr"
                or fx_node.name.startswith("params_")
                or fx_node.name.startswith("buffers_")
                else fx_node.op
            )
            if op_type != "output":
                df_node = df_graph.add_node(fx_node.name, op_type)
                fx2df_node_map[fx_node] = df_node

        # add profiling data to nodes
        if profile:
            self._load_profile_data_to_graph(df_graph, profiler)
            profiler = spill_profiler.SpillProfiler(df_graph)
            spill_times = profiler.benchmark_all()
            for e, r in spill_times.items():
                e.time = r

        # define function to get size (in bytes) using recursion as tensor_meta may contain lists or tuples tensor_meta
        def get_size(tensor_meta):
            size = 0
            if isinstance(tensor_meta, torch.fx.passes.shape_prop.TensorMetadata):
                num_elems = np.prod(tensor_meta.shape)
                if tensor_meta.dtype == torch.bool:
                    bits_per_elem = 8
                elif tensor_meta.dtype.is_floating_point:
                    bits_per_elem = torch.finfo(tensor_meta.dtype).bits
                else:
                    bits_per_elem = torch.iinfo(tensor_meta.dtype).bits
                num_bits = num_elems * bits_per_elem
                num_bytes = int(math.ceil(num_bits / 8))
                size += num_bytes
            else:
                assert tensor_meta
                if tensor_meta is not None:
                    for t in tensor_meta:
                        size += get_size(t)
            return size

        # add edges
        for fx_node in fx_graph.nodes:
            size = 0
            if "tensor_meta" in fx_node.meta:
                tensor_meta = fx_node.meta["tensor_meta"]
                size = get_size(tensor_meta)
            else:
                # print(f"MISSING META INFO FOR NODE {fx_node} with meta {fx_node.meta}")
                # We remove getitem nodes, so it's ok if we don't have the shape info there
                for fanout in fx_node.users.keys():
                    assert (
                        fanout.name == "getitem"
                        or fanout.name.startswith("getitem_")
                        or fanout.name == "lift_fresh_copy"
                        or fanout.name.startswith("lift_fresh_copy_")
                    )

            if fx_node not in fx2df_node_map:
                continue
            df_node = fx2df_node_map[fx_node]
            df_sinks = []
            for fx_sink in fx_node.users.keys():
                if fx_sink in fx2df_node_map:
                    df_sinks.append(fx2df_node_map[fx_sink])
            # for weight tensors, add size to nodes, otherwise add to edge
            if df_node.op_type == "weight":
                df_node.size = size
                size = 0

            if len(df_sinks) > 0:
                df_graph.add_edge([df_node], df_sinks, size, name=fx_node.name + ":0")

        if verbose:
            print(
                f"MODEL STATS: #RAW NODES={len(df_graph.nodes)}, #RAW EDGES={len(df_graph.edges)}"
            )

        if cleanup:
            df_graph, unused_weight_size = self._cleanup_dataflow_graph(df_graph)
            df_graph.unused_weight_size = unused_weight_size
        else:
            df_graph.dead_weight_size = 0

        if verbose:
            print(
                f"MODEL STATS: #ACTUAL OPERATORS={len(df_graph.nodes)}, #ACTUAL TENSORS={len(df_graph.edges)}"
            )

        result = [df_graph]
        if return_node_ordering:
            order = []
            for fx_node in fx_graph.nodes:
                if fx_node not in fx2df_node_map:
                    continue
                df_node = fx2df_node_map[fx_node]
                if df_node.name not in df_graph.nodes:
                    # print(f"Skipping deleted node {df_node.name}")
                    continue
                # print(f"Appending {df_node.name} to ordering")
                order.append(df_node)
            result.append(order)

        if return_fx_graph:
            result.append(fx_trace)
            result.append(fx2df_node_map)

        if len(result) == 1:
            return result[0]
        else:
            return result

    def _load_profile_data_to_graph(self, df_graph, profiler):
        profiler_table = profiler.summary()
        if profiler.profile_time:
            for _, row in profiler_table.iterrows():
                df_node = df_graph.find_node(row["Op name"])
                if df_node:
                    df_node.time = row["runtimes_sec"]

        if profiler.profile_memory:
            # TODO: return memory fragmentation info directly rather than adding them as attributed to dataflow graph
            df_graph.max_mem_fragmentation = profiler.get_max_mem_fragmentation()
            df_graph.peak_reserved_bytes = profiler.get_peak_reserved_bytes()

    def _cleanup_dataflow_graph(self, df_graph):
        unused_weight_size = 0
        dangling_nodes_to_delete = []
        getitem_nodes_to_clean = []
        meta_nodes_to_clean = []
        weights_to_clean = []
        in_place_nodes_to_rewire = []
        for n in df_graph.nodes.values():
            if (
                (
                    n.name.startswith("empty")
                    or n.name.startswith("buffers")
                    or n.name.startswith("params")
                    or n.name.startswith("zeros")
                    or n.name.startswith("_record_function_enter")
                )
                and len(n.fanin) == 0
                and len(n.fanout) == 0
            ):
                dangling_nodes_to_delete.append(n)
                # These weights/buffers will be loaded by PT, so we need to account for them
                # to get accuate simulation data
                unused_weight_size += n.size

            if (
                (n.name.startswith("buffers") or n.name.startswith("params"))
                and len(n.fanin) == 0
                and len(n.fanout) == 1
            ):
                deleted_sinks = 0
                for snk in n.fanout[0].sinks:
                    if (
                        (snk.name == "add_" or snk.name.startswith("add__"))
                        and len(snk.fanin) == 1
                        and len(snk.fanout) == 0
                    ):
                        # Unused counter
                        deleted_sinks += 1
                        dangling_nodes_to_delete.append(snk)
                if deleted_sinks == len(n.fanout[0].sinks):
                    dangling_nodes_to_delete.append(n)
                    # These weights/buffers will be loaded by PT, so we need to account for them
                    # to get accuate simulation data
                    unused_weight_size += n.size

            elif (n.name == "clone" or n.name.startswith("clone_")) and len(
                n.fanout
            ) == 0:
                dangling_nodes_to_delete.append(n)
            elif n.name == "getitem" or n.name.startswith("getitem_"):
                getitem_nodes_to_clean.append(n)
            elif (
                n.name == "relu_"
                or n.name.startswith("relu__")
                or n.name == "silu_"
                or n.name.startswith("silu__")
                or n.name == "hardtanh_"
                or n.name.startswith("hardtanh__")
                or n.name == "bernoulli_"
                or n.name.startswith("bernoulli__")
                or n.name == "fill_"
                or n.name.startswith("fill__")
            ):
                # In place op
                meta_nodes_to_clean.append(n)
            elif (
                n.name == "div_"
                or n.name.startswith("div__")
                and len(n.fanin) == 1
                and len(n.fanout) == 1
            ):
                # In place op
                meta_nodes_to_clean.append(n)
            elif (
                (
                    n.name == "add_"
                    or n.name.startswith("add__")
                    or n.name == "sub_"
                    or n.name.startswith("sub__")
                    or n.name == "masked_fill_"
                    or n.name.startswith("masked_fill__")
                    or n.name == "copy_"
                    or n.name.startswith("copy__")
                )
                and len(n.fanin) == 2
                and len(n.fanout) == 1
            ):
                in_place_nodes_to_rewire.append(n)
            elif (
                (n.name == "t" or n.name.startswith("t_"))
                or (n.name == "transpose" or n.name.startswith("transpose_"))
                or (n.name == "permute" or n.name.startswith("permute_"))
                or (n.name == "_reshape_alias" or n.name.startswith("_reshape_alias_"))
                or (n.name == "view" or n.name.startswith("view_"))
                or (n.name == "_unsafe_view" or n.name.startswith("_unsafe_view_"))
                or (n.name == "view_sym" or n.name.startswith("view_sym_"))
                or (n.name == "expand" or n.name.startswith("expand_"))
                or (n.name == "split" or n.name.startswith("split_"))
                or (n.name == "slice" or n.name.startswith("slice_"))
                or (n.name == "squeeze" or n.name.startswith("squeeze_"))
                or (n.name == "unsqueeze" or n.name.startswith("unsqueeze_"))
                or (n.name == "as_strided" or n.name.startswith("as_strided_"))
            ):
                meta_nodes_to_clean.append(n)
            elif (
                (
                    n.name == "native_batch_norm"
                    or n.name.startswith("native_batch_norm_")
                )
                or (
                    n.name == "native_layer_norm"
                    or n.name.startswith("native_layer_norm_")
                )
                or (n.name == "addmm" or n.name.startswith("addmm_"))
                or (n.name == "convolution" or n.name.startswith("convolution_"))
            ):
                weights_to_clean.append(n)

        for n in dangling_nodes_to_delete:
            # print(f"deleting dangling node {n.name}")
            df_graph.delete_node(n)

        for n in getitem_nodes_to_clean:
            # print(f"deleting getitem node {n.name}")
            self.bypass_and_delete_getitem_node(df_graph, n)

        for n in meta_nodes_to_clean:
            # print(f"deleting meta node {n.name}")
            self.bypass_and_delete_meta_node(df_graph, n)

        for n in in_place_nodes_to_rewire:
            # print(f"rewiring inplace node {n.name}")
            self.rewire_in_place_node(df_graph, n)

        for n in weights_to_clean:
            # print(f"consolidating weights for node {n.name}")
            self.consolidate_weights(df_graph, n)

        return (df_graph, unused_weight_size)

    def bypass_and_delete_getitem_node(self, df_graph, node):
        assert len(node.fanin) == 1
        edge_in = node.fanin[0]
        assert df_graph.get_size(edge_in) == 0
        sources = edge_in.sources

        for edge_out in node.fanout:
            if len(edge_out.sinks) == 0:
                continue
            size = df_graph.get_size(edge_out)
            df_graph.add_edge(
                sources, edge_out.sinks, size, name=edge_out.name + "_opt"
            )

        df_graph.delete_node(node)

    def bypass_and_delete_meta_node(self, df_graph, node):
        assert len(node.fanin) == 1
        edge_in = node.fanin[0]

        if len(node.fanout) == 1:
            edge_out = node.fanout[0]
            edge_in.add_sinks(edge_out.sinks)
        elif len(node.fanout) > 1:
            sinks = []
            for fanout in node.fanout:
                sinks += fanout.sinks
            edge_in.add_sinks(sinks)

        df_graph.delete_node(node)

    def rewire_in_place_node(self, df_graph, node):
        # If there are control dependencies, make sure they come after regular inputs
        assert len(node.fanin) >= 2
        for i in range(2):
            assert node.fanin[i].size > 0
        for i in range(2, len(node.fanin)):
            assert node.fanin[i].size == 0
        assert len(node.fanout) == 1

        # The first input will be modified in place and returned as output.
        assert node.fanin[0].size == node.fanout[0].size
        node.fanout[0].size = 0
        for snk in node.fanout[0].sinks:
            # print(f"Processing snk {snk}")
            assert len(snk.fanin) > 0
            for i in range(0, len(snk.fanin)):
                # print(f"    Checkin fanin {snk.fanin[i]} againt {node.fanout[0]}", flush=True)
                if id(snk.fanin[i]) == id(node.fanout[0]):
                    snk.fanin[i] = node.fanin[0]
                    node.fanin[0].sinks.append(snk)
                    snk.fanin.append(node.fanout[0])
                    # print(f"   Updated sink node {snk}")
                    break

    def consolidate_weights(self, df_graph, node):
        weights_to_merge = []
        for f in node.fanin:
            if not f.is_stateful():
                continue
            weights_to_merge.extend(f.sources)

        if len(weights_to_merge) <= 1:
            # print(f"no weight to merge for node : {node}")
            return

        ref_sinks = set()
        assert len(weights_to_merge[0].fanout) == 1
        for s in weights_to_merge[0].fanout[0].sinks:
            ref_sinks.add(s)

        # Merge the weight nodes
        sinks_to_merge = []
        for i in range(1, len(weights_to_merge)):
            assert len(weights_to_merge[i].fanout) == 1
            for s in weights_to_merge[i].fanout[0].sinks:
                if s not in ref_sinks:
                    weights_to_merge[0].fanout[0].add_sink(s)
                    sinks_to_merge.append(s)

        for i in range(1, len(weights_to_merge)):
            weights_to_merge[0].size += weights_to_merge[i].size
            df_graph.delete_node(weights_to_merge[i])

        nodes_to_delete = []
        for s in sinks_to_merge:
            # Look for a similar operator in ref_sinks
            # print(f"Looking for merge candidate for {s}")
            for r in ref_sinks:
                if r.op_type != s.op_type:
                    continue
                if len(r.fanin) != len(s.fanin):
                    continue
                valid_candidate = True
                # print(f"  Checking candidate {r}")
                edges_to_merge = []
                for f in s.fanin:
                    if f in r.fanin:
                        # print(f"    {f} in candidate fanin set")
                        continue
                    found_match = False
                    for fr in r.fanin:
                        if fr.sources == f.sources:
                            # print("    Sources do match")
                            edges_to_merge.append((f, fr))
                            found_match = True
                            break
                        # else:
                        # print(
                        #    f"    sources don't match: {fr.sources} vs {f.sources}"
                        # )

                    if not found_match:
                        # print(f"    Found invalid fanin {f}")
                        valid_candidate = False
                        break

                if valid_candidate:
                    # print(f"  Merging {s} with {r}")
                    for f, fr in edges_to_merge:
                        fr.size += f.size
                    nodes_to_delete.append(s)
                    break

        for n in nodes_to_delete:
            df_graph.delete_node(n)

        edges_to_merge = []
        for i in range(len(node.fanout)):
            for j in range(i + 1, len(node.fanout)):
                fi = node.fanout[i]
                fj = node.fanout[j]
                if fi.sinks == fj.sinks:
                    edges_to_merge.append((fi, fj))

        # The code below isn't right if there's more than one pair of edges to merge. Fortunately this doesn't happen in practice.
        assert len(edges_to_merge) <= 1
        for f, fo in edges_to_merge:
            f.size += fo.size
            df_graph.delete_edge(fo)
