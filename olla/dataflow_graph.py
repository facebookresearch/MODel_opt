
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
import re
import traceback
from collections import defaultdict
from typing import List

import networkx as nx
from graphviz import Digraph


# Each node corresponds to a computation in the model. Each node can have
# fanin (resp fanout) edges, which correspond to the tensor(s) fed to (resp
# produced) by the node.
class Node:
    def __init__(self, name=None, op_type=None, size=0, read_only=False, time=None):
        self.name = name
        self.op_type = op_type
        self.size = size
        self.read_only = read_only
        self.fanin = []
        self.fanout = []
        self.time = time

    def __repr__(self):
        val = self.name + " (" + str(self.op_type) + ")"
        if self.size > 0:
            val += " [" + str(self.size) + "]"
        return val

    def is_stateful(self):
        return (
            self.size > 0
            or self.op_type == "stateful_node"
            or self.op_type == "stateful_node_sink"
        )

    def access_stateful_node(self):
        for edge in self.fanin:
            for node in edge.sources:
                if node.is_stateful():
                    return True
        return False


# Each edge corresponds to a tensor in the model. An edge has exactly one
# source node, which is the operation that produced the corresponding value.
class Edge:
    def __init__(
        self,
        source,
        sinks,
        size,
        name=None,
        mem_space=None,
        tile_id=None,
        group_id=None,
    ):
        self.name = name
        self.size = size
        self.source = source
        self.sinks = sinks
        self.mem_space = mem_space
        self.tile_id = tile_id
        self.group_id = group_id

    def __repr__(self):
        return (
            self.name
            + ": "
            + str(self.source)
            + " -> "
            + str(self.sinks)
            + " size="
            + str(self.size)
        )

    def add_sink(self, node):
        self.sinks.append(node)
        node.fanin.append(self)

    def add_sinks(self, nodes):
        for node in nodes:
            self.add_sink(node)

    def add_source(self, node):
        assert self.source is None, "Edge class can only have one source"
        self.source = node

    def is_stateful(self):
        return self.source.is_stateful()

    @property
    def sources(self):
        return iter([self.source])


# Each MultiSourceEdge corresponds to a tensor in the model. An edge has at least one
# source node, which correspond to the operation(s) that produced the tensor value.
# Each has also has at least one sink. Each sink corresponds to a node that
# consumes the corresponding value.
class MultiSourceEdge:
    def __init__(
        self,
        sources,
        sinks,
        size,
        name=None,
        mem_space=None,
        tile_id=None,
        group_id=None,
    ):
        self.name = name
        self.size = size
        self.sources = sources
        self.sinks = sinks
        self.mem_space = mem_space
        self.tile_id = tile_id
        self.group_id = group_id

    def __repr__(self):
        return f"MultiSourceEdge {self.name}, size:{self.size}, mem_space:{self.mem_space}, tile_id:{self.tile_id} group_id:{self.group_id} sources:{self.sources} sinks:{self.sinks}"

    def add_source(self, node):
        self.sources.append(node)
        node.fanout.append(self)

    def add_sources(self, nodes):
        for node in nodes:
            self.add_source(node)

    def add_sink(self, node):
        self.sinks.append(node)
        node.fanin.append(self)

    def add_sinks(self, nodes):
        for node in nodes:
            self.add_sink(node)

    def is_stateful(self):
        if len(self.sources) != 1:
            return False
        return self.sources[0].is_stateful()


# A directed hypergraph that models the computation. Loops are forbidden.
# Each model input must be represented by a node with no fanin edge
# Each model output must be represented by a node with no fanout edge.
class Graph:
    def __init__(self, name=""):
        self.name = name
        self.nodes = {}
        self.edges = {}
        self.canonical = False

    def add_node(self, name=None, op_type=None, size=0, read_only=False):
        if name is None:
            name = str(len(self.nodes) + 1)
        assert type(name) is str
        assert name not in self.nodes
        self.nodes[name] = Node(name, op_type, size, read_only)
        return self.nodes[name]

    def add_edge(
        self,
        sources,
        sinks,
        size,
        name=None,
        mem_space=None,
        tile_id=None,
        group_id=None,
        mutable=True,
    ):
        if name is None:
            name = str(len(self.edges) + 1)
        assert type(name) is str
        assert name not in self.edges

        if not mutable and len(sources) == 1:
            edge = Edge(
                sources[0],
                sinks,
                size,
                name,
                mem_space=mem_space,
                tile_id=tile_id,
                group_id=group_id,
            )
        else:
            edge = MultiSourceEdge(
                sources,
                sinks,
                size,
                name,
                mem_space=mem_space,
                tile_id=tile_id,
                group_id=group_id,
            )
        self.edges[name] = edge
        for node in sources:
            node.fanout.append(edge)
        for node in sinks:
            node.fanin.append(edge)

        return edge

    def find_node(self, name: str = None, op_type: str = None) -> Node:
        nodes = self.find_nodes(name, op_type, max_nodes=1)
        if len(nodes) > 0:
            return nodes[0]
        else:
            return None

    def find_nodes(
        self, name: str = None, op_type: str = None, max_nodes: int = None
    ) -> List[Node]:
        if not name:
            raise ValueError("name not provided.")

        name = name.replace("*", ".*")
        regex = re.compile(f"^{name}$")
        matching_node_names = list(filter(regex.match, self.nodes.keys()))

        if op_type:
            return [
                self.nodes[key]
                for key in matching_node_names
                if self.nodes[key].op_type == op_type
            ][:max_nodes]
        else:
            return [self.nodes[key] for key in matching_node_names][:max_nodes]

    def find_edge(
        self, name: str = None, mem_space=None, tile_id=None, group_id=None
    ) -> Edge:
        edges = self.find_edges(name, mem_space, tile_id, group_id, max_edges=1)
        if len(edges) > 0:
            return edges[0]
        else:
            return None

    def find_edges(
        self,
        name: str = None,
        mem_space=None,
        tile_id=None,
        group_id=None,
        max_edges: int = None,
    ) -> List[Edge]:
        def filter_edges(edge):
            if_mem_space = edge.mem_space == mem_space
            if_tile_id = edge.tile_id == tile_id
            if_group_id = edge.group_id == group_id

            return if_mem_space and if_tile_id and if_group_id

        named_edges = []
        if name:
            name = name.replace("*", ".*")
            regex = re.compile(f"^{name}$")
            matching_edge_names = list(filter(regex.match, self.edges.keys()))
            named_edges = [self.edges[key] for key in matching_edge_names]

        filter_params = list(filter(filter_edges, named_edges or self.edges.values()))

        return filter_params[:max_edges]

    def delete_edge(self, edge):
        # print("Deleting edge " + str(edge))

        for node in edge.sources:
            node.fanout.remove(edge)

        for node in edge.sinks:
            node.fanin.remove(edge)

        del self.edges[edge.name]

    def delete_node(self, node):
        # print("Deleting node " + str(node))

        edges_to_delete = set()
        for edge in node.fanin:
            # print("Removing " + str(node) + " from the sinks of " + str(edge))
            if len(edge.sinks) == 1:
                edges_to_delete.add(edge)
            else:
                edge.sinks.remove(node)

        for edge in node.fanout:
            if isinstance(edge, MultiSourceEdge):
                if len(edge.sources) == 1:
                    edges_to_delete.add(edge)
                else:
                    edge.sources.remove(node)
            else:
                assert edge.source == node
                edges_to_delete.add(edge)

        for e in edges_to_delete:
            self.delete_edge(e)
        del self.nodes[node.name]

    # Convert all the MultiSourceEdges to regular Edges.
    def canonicalize(self):
        if self.canonical:
            return

        canonical_edges = {}
        for name, e in self.edges.items():
            assert name == e.name
            if isinstance(e, Edge):
                canonical_edges[name] = e

            elif len(e.sources) == 1:

                new_edge = Edge(
                    e.sources[0],
                    e.sinks,
                    e.size,
                    e.name,
                    mem_space=e.mem_space,
                    tile_id=e.tile_id,
                    group_id=e.group_id,
                )
                canonical_edges[name] = new_edge
                e.sources[0].fanout.remove(e)
                e.sources[0].fanout.append(new_edge)
                for snk in e.sinks:
                    snk.fanin.remove(e)
                    snk.fanin.append(new_edge)
            else:
                name = "allocate_" + e.name
                node = Node(name, op_type="allocate_tensor")
                self.nodes[name] = node

                # combined_no_dup = e.sinks + list(set(e.sources) - set(e.sinks))
                assert set(e.sources).isdisjoint(set(e.sinks))
                combined_no_dup = e.sinks + e.sources

                new_edge = Edge(
                    node,
                    combined_no_dup,
                    e.size,
                    name=name,
                    mem_space=e.mem_space,
                    tile_id=e.tile_id,
                    group_id=e.group_id,
                )

                canonical_edges[name] = new_edge
                node.fanout.append(new_edge)
                for src in e.sources:
                    if new_edge not in src.fanin:
                        src.fanin.append(new_edge)
                for snk in e.sinks:
                    if new_edge not in snk.fanin:
                        snk.fanin.append(new_edge)
                    snk.fanin.remove(e)

                for src in e.sources:
                    name = src.name + "_" + e.name
                    new_edge = Edge(
                        src,
                        e.sinks,
                        0,
                        name=name,
                        mem_space=e.mem_space,
                        tile_id=e.tile_id,
                        group_id=e.group_id,
                    )
                    canonical_edges[name] = new_edge
                    src.fanout.append(new_edge)
                    src.fanout.remove(e)
                    for n in e.sinks:
                        n.fanin.append(new_edge)

        self.edges = canonical_edges

        # Handle stateful nodes
        new_nodes = []
        for n in self.nodes.values():
            if n.size > 0 and len(n.fanout) > 0:
                assert len(n.fanout) == 1
                edge = n.fanout[0]
                edge.size += n.size
                snk_node = Node(n.name + "_snk", op_type="stateful_node_sink")
                new_nodes.append((snk_node.name, snk_node))
                edge.add_sink(snk_node)
                n.size = 0
                n.op_type = "stateful_node"

        self.nodes.update(new_nodes)

        self.canonical = True

    # TODO: should we merge this into the `prune()` function?
    def remove_dead_nodes(self):
        nodes_to_delete = set()

        for n in self.nodes.values():
            if len(n.fanin) == 0 and len(n.fanout) == 0:
                nodes_to_delete.add(n)

        for n in nodes_to_delete:
            self.delete_node(n)

        return

    # Remove unnecessary nodes from the graph
    def prune(self, aggressive=False):
        # The graph should be pruned before canonicalization
        assert not self.canonical

        nodes_to_delete = set()
        for e in self.edges.values():
            if not isinstance(e, MultiSourceEdge):
                continue
            if len(e.sources) == 1:
                continue
            deletion_candidates = []
            for n in e.sources:
                # Undriven copy or fill node
                if len(n.fanin) == 0 and (
                    n.op_type == "turing::copy" or n.op_type == "turing::fill"
                ):
                    deletion_candidates.append(n)

            # Delete these undriven copy/fill nodes as long as this will leave
            # at least one source for the edge.
            if len(deletion_candidates) < len(e.sources):
                for n in deletion_candidates:
                    nodes_to_delete.add(n)

        for n in nodes_to_delete:
            self.delete_node(n)

        if not aggressive:
            return

        nodes_to_delete = set()
        for e in self.edges.values():
            if not isinstance(e, MultiSourceEdge):
                continue
            # Check for the following pattern and remove the copy node:
            # Op1 ________________ Op2
            #     \____ Copy ____/
            if len(e.sources) != 2:
                continue
            if e.sources[0].op_type == "turing::copy":
                copy_node = e.sources[0]
                op1_node = e.sources[1]
            elif e.sources[1].op_type == "turing::copy":
                copy_node = e.sources[1]
                op1_node = e.sources[0]
            else:
                continue
            if (
                len(copy_node.fanin) == 1
                and copy_node.fanin[0].size == 0
                and len(copy_node.fanin[0].sources) == 1
                and copy_node.fanin[0].sources[0] == op1_node
            ):
                nodes_to_delete.add(copy_node)

        for n in nodes_to_delete:
            self.delete_node(n)

        nodes_to_delete = set()
        for e in self.edges.values():
            if not isinstance(e, MultiSourceEdge):
                continue
            # Check for the following pattern and a keep single representative
            # copy node:
            # Op1 _______Op2_______ Op3
            #     \____ Copy1 ____/
            #     \____ Copy2 ____/
            #     \____ Copy3 ____/
            representative_copy = None
            for n in e.sources:
                if n.op_type != "turing::copy":
                    continue
                if not representative_copy:
                    representative_copy = n
                else:
                    if (
                        n.fanin == representative_copy.fanin
                        and n.fanout == representative_copy.fanout
                    ):
                        nodes_to_delete.add(n)

        for n in nodes_to_delete:
            self.delete_node(n)

    # Remove unnecessary nodes from the graph
    def prune_old(self, aggressive=False):
        assert self.canonical
        nodes_to_delete = set()
        for n in self.nodes.values():
            prune = len(n.fanout) > 0
            for f in n.fanout:
                if f.size > 0:
                    prune = False
                    break
            for f in n.fanin:
                if aggressive and f.source.op_type == "allocate_tensor":
                    continue
                if f.size > 0:
                    prune = False
                    break
            # All the fanin and fanout edges have size 0: the node
            # has no impact on the memory footprint and can be deleted
            if prune:
                nodes_to_delete.add(n)

        for n in nodes_to_delete:
            for fanin in n.fanin:
                for fanout in n.fanout:
                    for snk in fanout.sinks:
                        if not self.is_in_transitive_fanin(fanin.source, snk, n):
                            self.add_edge([fanin.source], [snk], size=0, mutable=False)
            self.delete_node(n)

        if self.canonical:
            self.canonical = False
            self.canonicalize()

    # Constraint allocations to make sure they happen as late as possible since
    # there is no need for them to happen any sooner than that.
    def constrain_allocations(self):
        dom_tree = self.build_dominator_tree()
        for n in self.nodes.values():
            if n.op_type != "allocate_tensor":
                continue
            afters = []
            seen = set()
            for fo in n.fanout:
                for s in fo.sinks:
                    if s not in seen:
                        afters.append(s)
                        seen.add(s)
            dom = dom_tree.lowest_common_ancestor(afters)
            if dom:
                self.add_edge(
                    [dom], [n], 0, "constrain_" + n.name + "_after_" + dom.name
                )

        if self.canonical:
            self.canonical = False
            self.canonicalize()

    def constrain_weight_updates(self):
        fwd_levels = self.build_levelization()
        bwd_levels = self.build_reverse_levelization()

        # print("CONSTRAINING WEIGHT UPDATES")

        for n in self.nodes.values():
            if len(n.fanout) > 0:
                continue
            if n.is_stateful():
                continue
            if not n.access_stateful_node():
                continue

            # print(f"Found interesting node {n.name} at min fwd level {fwd_levels[n]}")
            min_fwd_level = fwd_levels[n]
            max_bwd_level = 0
            best_candidate_node = None
            candidate_sources = set([n])
            cache = {}
            while not best_candidate_node and len(candidate_sources) > 0:
                candidate_sources = self._generate_candidate_sources(candidate_sources)
                # print(f"  CANDIDATE SOURCES {candidate_sources}")
                for src in candidate_sources:
                    candidate, level = self._find_candidate_fwd(
                        src, fwd_levels, bwd_levels, min_fwd_level, cache
                    )
                    # print(f"    CANDIDATE ANCHOR {candidate} at {level}")
                    if level > max_bwd_level:
                        max_bwd_level = level
                        best_candidate_node = candidate

                if best_candidate_node:
                    # print(f"  ADDING EDGE FROM {n.name} to {best_candidate_node.name}")
                    self.add_edge(
                        [n],
                        [best_candidate_node],
                        size=0,
                        name=n.name + "_forced_early",
                    )

        if self.canonical:
            self.canonical = False
            self.canonicalize()

    def constrain_tensor_generators(self):
        fwd_levels = self.build_levelization()

        for n in self.nodes.values():
            if len(n.fanin) > 0:
                continue
            if not n.name.startswith("empty") and not n.name.startswith(
                "new_empty_strided"
            ):
                continue

            # print(f"  CONSTRAINING TENSOR GENERATOR NODE {n.name}")

            candidates = self._find_candidates_bwd(n)

            best_candidate = None
            max_level = len(self.nodes)
            for c in candidates:
                if fwd_levels[c] < max_level:
                    max_level = fwd_levels[c]
                    best_candidate = c
            if best_candidate:
                # print(f"  ADDING EDGE FROM {best_candidate.name} to {n}")
                self.add_edge(
                    [best_candidate], [n], size=0, name=n.name + "_forced_late"
                )

        if self.canonical:
            self.canonical = False
            self.canonicalize()

    def constrain_relative_ordering(self, node_ordering, linearize=False):
        min_timestep = math.inf
        max_timestep = 0
        per_timestep = defaultdict(lambda: [])
        for n, t in node_ordering.items():
            if n.is_stateful():
                continue
            min_timestep = min(min_timestep, t)
            max_timestep = max(max_timestep, t)
            per_timestep[t].append(n)

        if linearize:
            linearized = {}
            current_timestep = min_timestep
            for ts in range(min_timestep, max_timestep + 1):
                for n in per_timestep[ts]:
                    linearized[current_timestep] = [n]
                    current_timestep += 1
            per_timestep = linearized
            max_timestep = current_timestep - 1

        previous_ts = min_timestep
        assert len(per_timestep[previous_ts]) > 0
        for ts in range(min_timestep + 1, max_timestep + 1):
            if len(per_timestep[ts]) == 0:
                continue
            for src in per_timestep[previous_ts]:
                needed_snks = []
                name = f"run_{src.name}_before"
                for snk in per_timestep[ts]:
                    edge_already_exists = False
                    for fi in snk.fanin:
                        for s in fi.sources:
                            if s == src:
                                edge_already_exists = True
                                break
                    if not edge_already_exists:
                        needed_snks.append(snk)
                        name += f"_{snk.name}"
                if len(needed_snks) > 0:
                    name.replace(" ", "_")
                    self.add_edge([src], needed_snks, size=0, name=name)
                    print(f"ADDED EDGE FROM {src} to {needed_snks}")
            previous_ts = ts

        if self.canonical:
            self.canonical = False
            self.canonicalize()

    def _generate_candidate_sources(self, nodes):
        sources = set()
        for n in nodes:
            for fi in n.fanin:
                for src in fi.sources:
                    if src.is_stateful():
                        continue
                    sources.add(src)
        return sources

    def _find_candidate_fwd(self, node, fwd_levels, bwd_levels, min_fwd_level, cache):
        if node in cache:
            return cache[node]

        max_bwd_level = -1
        best_candidate = None

        # print(f"LOOKING FOR ANCHOR FOR SRC {node.name}")

        for fo in node.fanout:
            for snk in fo.sinks:
                # print(
                #    f"LOOKING AT SNK {snk.name} AT FWD {fwd_levels[snk]} BWD {bwd_levels[snk]}"
                # )
                if bwd_levels[snk] <= max_bwd_level:
                    continue
                if fwd_levels[snk] <= min_fwd_level:
                    candidate, level = self._find_candidate_fwd(
                        snk, fwd_levels, bwd_levels, min_fwd_level, cache
                    )
                    if level > max_bwd_level:
                        max_bwd_level = level
                        best_candidate = candidate
                else:
                    max_bwd_level = bwd_levels[snk]
                    best_candidate = snk

        cache[node] = (best_candidate, max_bwd_level)
        return (best_candidate, max_bwd_level)

    def _find_candidates_bwd(self, node):
        candidates = set()
        for fo in node.fanout:
            for snk in fo.sinks:
                for fi in snk.fanin:
                    if fi is fo:
                        continue
                    if fi.is_stateful():
                        continue
                    candidates.update(fi.sources)

        if len(candidates) == 0:
            for fo in node.fanout:
                for snk in fo.sinks:
                    if snk.is_stateful():
                        continue
                    candidates.update(self._find_candidates_bwd(snk))

        return candidates

    # Build and return the line graph corresponding to this graph
    # The line graph is always canonical
    def build_line_graph(self):
        assert self.canonical
        line_graph = Graph()
        for e in self.edges.values():
            line_graph.add_node(e.name)
        for n in self.nodes.values():
            id = 0
            if len(n.fanout) == 0:
                continue
            for fanin in n.fanin:
                src = line_graph.nodes[fanin.name]
                snks = []
                for fanout in n.fanout:
                    snks.append(line_graph.nodes[fanout.name])
                line_graph.add_edge([src], snks, size=0, name=n.name + str(id))
                id += 1

        line_graph.canonicalize()
        return line_graph

    def build_dominator_tree(self, skip_allocations=True, skip_weights=True):
        assert self.canonical
        G = nx.DiGraph()
        G.add_node("root")
        for v in self.nodes:
            G.add_node(v)
        for e in self.edges.values():
            src = e.source.name
            for t in e.sinks:
                G.add_edge(src, t.name)
        for v in self.nodes.values():
            if (
                len(v.fanin) == 0
                and (not skip_allocations or v.op_type != "allocate_tensor")
                and (not skip_weights or v.op_type != "weight")
            ):
                G.add_edge("root", v.name)

        return DominatorTree(
            self, nx.algorithms.dominance.immediate_dominators(G, "root")
        )

    def _order_fanin_of_vertex_topologically(self, n, visited, ordering):
        visited.add(n)
        for fanin in n.fanin:
            for src in fanin.sources:
                if src in visited:
                    continue
                self._order_fanin_of_vertex_topologically(src, visited, ordering)
        ordering.append(n)

    def compute_topological_ordering(self):
        visited = set()
        ordering = []
        for n in self.nodes.values():
            if n.op_type == "stateful_node":
                self._order_fanin_of_vertex_topologically(n, visited, ordering)
        for n in self.nodes.values():
            if n not in visited and not n.is_stateful():
                self._order_fanin_of_vertex_topologically(n, visited, ordering)
        for n in self.nodes.values():
            if n.op_type == "stateful_node_sink":
                self._order_fanin_of_vertex_topologically(n, visited, ordering)
        return ordering

    def build_levelization(self):
        levelization = {}
        for n in self.nodes.values():
            if n in levelization:
                continue
            levelization[n] = self._compute_level(n, levelization)
        return levelization

    def _compute_level(self, node, levelization):
        level = 0
        for edge in node.fanin:
            for src in edge.sources:
                if src in levelization:
                    lvl = levelization[src]
                else:
                    lvl = self._compute_level(src, levelization)
                level = max(lvl + 1, level)
        levelization[node] = level
        return level

    def build_reverse_levelization(self):
        levelization = {}
        for n in self.nodes.values():
            if n in levelization:
                continue
            levelization[n] = self._compute_reverse_level(n, levelization)
        return levelization

    def _compute_reverse_level(self, node, levelization):
        level = 0
        for edge in node.fanout:
            for sink in edge.sinks:
                if sink in levelization:
                    lvl = levelization[sink]
                else:
                    lvl = self._compute_reverse_level(sink, levelization)
                level = max(lvl + 1, level)
        levelization[node] = level
        return level

    def longest_path_length(self):
        length = 0
        for lvl in self.build_levelization().values():
            length = max(length, lvl)
        # Levels start at zero
        return length + 1

    # Returns true if the two edges are connected to the same node(s)
    def are_connected_by_node(self, t1, t2):
        nodes_connected_to_t1 = set(t1.sinks + [t1.source])
        nodes_connected_to_t2 = set(t2.sinks + [t2.source])
        return not nodes_connected_to_t1.isdisjoint(nodes_connected_to_t2)

    # Return True iff t1 is in the immediate fanin of t2
    def is_in_immediate_fanin(self, t1, t2):
        assert self.canonical
        for fanin in t2.fanin:
            if fanin.source == t1:
                return True
        return False

    # Return True iff t1 is in the transitive fanin of t2
    @functools.lru_cache(maxsize=128 * 1024 * 1024)
    def is_in_transitive_fanin(self, t1, t2, skip_intermediate_node=None):
        # print(t1.name + " IN TRANSITIVE FANIN OF " + t2.name, flush=True)
        for fanin in t2.fanin:
            if fanin.source == skip_intermediate_node:
                continue
            # print("CHECKING FANIN " + fanin.source.name, flush=True)
            if fanin.source == t1:
                # print("1", flush=True)
                return True
            if self.is_in_transitive_fanin(t1, fanin.source):
                # print("2", flush=True)
                return True
            # print("DONE WITH FANIN")
        # print("3", flush=True)
        return False

    @functools.lru_cache(maxsize=256 * 1024 * 1024)
    def is_t1_before_t2(self, t1, t2):
        assert self.canonical
        start2 = t2.source
        for end1 in t1.sinks:
            if not self.is_in_transitive_fanin(end1, start2):
                return False

        # print(t1.name + " comes before " + t2.name)
        return True

    def can_overlap_in_time(self, t1, t2):
        if self.is_t1_before_t2(t1, t2):
            return False
        if self.is_t1_before_t2(t2, t1):
            return False
        return True

    def sort(self):
        self.nodes = dict(sorted(self.nodes.items()))
        self.edges = dict(sorted(self.edges.items()))

    def dump(self, filename=None, format="canon"):
        large_graph = len(self.nodes) > 100 or len(self.edges) > 100
        very_large_graph = len(self.nodes) > 250 or len(self.edges) > 250
        if very_large_graph:
            dot = Digraph(format=format, engine="neato")
        else:
            dot = Digraph(format=format)

        if self.name:
            dot.attr("graph", label=self.name)
        dot.attr("node", shape="record")
        # Speedup the layout of large graphs
        if large_graph:
            dot.attr("graph", overlap="scale")
            dot.attr("graph", spline="line")

        # Add the nodes
        for node in self.nodes.values():
            fanout_sz = len(node.fanout)
            full_name = node.name
            if not large_graph:
                # Use extended label
                if node.op_type:
                    full_name += f" ({node.op_type})"
                if node.size > 0 and node.time:
                    full_name += f" [{node.size}, {node.time*1000:.2} ms]"
                elif node.size > 0:
                    full_name += f" [{node.size}]"
                elif node.time:
                    full_name += f" [{node.time*1000:.2} ms]"
            if fanout_sz <= 1 or large_graph:
                dot.node(node.name, label=full_name)
            else:
                # Small graph, use ports to distinguish edges
                label = ""
                for i in range(fanout_sz):
                    label += "<f" + str(i) + "> " + full_name + "_" + str(i)
                    if i < fanout_sz - 1:
                        label += "|"
                dot.node(node.name, label=label)

        def edge_label(edge):
            group_id = ""
            if hasattr(edge, "group_id") and edge.group_id is not None:
                group_id = f"g{edge.group_id}/"

            return (
                edge.name + "/" + group_id + str(edge.size) + "/" + str(edge.mem_space)
                if edge.mem_space is not None
                else str(edge.size)
            )

        # Add the edges
        for node in self.nodes.values():
            fanout_sz = len(node.fanout)
            if fanout_sz == 0:
                continue
            elif fanout_sz == 1:
                edge = node.fanout[0]
                for sink in edge.sinks:
                    dot.edge(node.name, sink.name, label=edge_label(edge))
            else:
                for i in range(fanout_sz):
                    edge = node.fanout[i]
                    for sink in edge.sinks:
                        dot.edge(
                            node.name if large_graph else node.name + ":f" + str(i),
                            sink.name,
                            label=edge_label(edge),
                        )

        if filename and format == "canon":
            with open(filename, "w") as f:
                for line in dot:
                    f.write(line)
                    if (not line.endswith("\n")):
                        f.write("\n")
            return filename
        elif filename:
            dot.render(filename=filename, format=format)
            return filename
        elif format == "canon":
            result = ""
            for line in dot:
                result += line
                if (not line.endswith("\n")):
                    result += "\n"
            return result
        else:
            return dot.pipe().decode("utf-8")

    def check_duplicates(self):
        """
        Check whether graph has any duplicated sources/sinks.
        """

        for e in self.edges.values():
            if isinstance(e, MultiSourceEdge):
                seen_sources = set()
                for source in e.sources:
                    if source in seen_sources:
                        raise AssertionError(
                            f"MultiSourceEdge {e} has duplicate source"
                        )
                    seen_sources.add(source)

            seen_sinks = set()
            for snk in e.sinks:
                if snk in seen_sinks:
                    raise AssertionError(f"Edge {e} has duplicate sink")
                seen_sinks.add(snk)

        for n in self.nodes.values():
            seen_fanout = set()
            for e in n.fanout:
                if e in seen_fanout:
                    raise AssertionError(f"Node {n} has duplicate fanout")
                seen_fanout.add(e)

            seen_fanin = set()
            for e in n.fanin:
                if e in seen_fanin:
                    raise AssertionError(f"Node {n} has duplicate fanin")
                seen_fanin.add(e)

    def check_for_self_edges(self):
        """
        Simple test to see whether graphs have any self edges (nodes which loop to themselves).
        This won't detect longer cycles, however.
        """

        for e in self.edges.values():
            for sink in e.sinks:
                if (isinstance(e, MultiSourceEdge) and sink in e.sources) or (
                    isinstance(e, Edge) and sink == e.source
                ):
                    raise AssertionError(f"node {sink} has self edge (edge {e})")

    # Check the consistency of the graph
    def check_consistency(self, verbose=False):
        for e in self.edges.values():
            for source in e.sources:
                if e not in source.fanout:
                    if verbose:
                        print(
                            f"Edge {e.name} not in fanout of {source.name}", flush=True
                        )
                    return False

            for snk in e.sinks:
                if e not in snk.fanin:
                    if verbose:
                        print(f"Edge {e.name} not in fanin of {snk.name}", flush=True)
                    return False

        for n in self.nodes.values():
            for e in n.fanout:
                if isinstance(e, MultiSourceEdge):
                    if n not in e.sources:
                        if verbose:
                            print(
                                f"{n.name} not in sources of edge {e.name}", flush=True
                            )
                        return False
                else:
                    if e.source != n:
                        if verbose:
                            print(
                                f"{n.name} not the source of edge {e.name}", flush=True
                            )
                        return False

            for e in n.fanin:
                if n not in e.sinks:
                    if verbose:
                        print(f"{n.name} not in sinks of edge {e.name}", flush=True)
                    return False

        self.check_duplicates()
        self.check_for_self_edges()

        return True

    def _has_loop_in_fanout(self, node, status, verbose):
        if node in status:
            return status[node] != 2

        status[node] = 1
        for e in node.fanout:
            for s in e.sinks:
                has_loop = self._has_loop_in_fanout(s, status, verbose)
                if has_loop:
                    if verbose:
                        print(f"Found loop through node {s.name}", flush=True)
                    return True
        status[node] = 2
        return False

    def has_loops(self, verbose=False):
        status = {}
        for n in self.nodes.values():
            if self._has_loop_in_fanout(n, status, verbose):
                return True
        return False

    def is_valid(self, verbose=False):
        try:
            valid = self.check_consistency(verbose)
        except Exception:
            if verbose:
                print(traceback.format_exc())
            return False

        valid &= not self.has_loops(verbose)
        return valid

    # count all the sink connections in all edges
    def sink_count(self):
        sinks = 0
        for e in self.edges.values():
            sinks += len(e.sinks)

        return sinks

    # count all the source connections in all edges
    def source_count(self):
        sources = 0
        for e in self.edges.values():
            if isinstance(e, MultiSourceEdge):
                sources += len(e.sources)
            else:
                sources += 1

        return sources

    # usually the size of a tensor is at the edge, but for weight nodes, it could be at the node
    def get_size(self, edge):
        if edge.size != 0:
            return edge.size
        else:
            if isinstance(edge, MultiSourceEdge):
                size = 0
                for source in edge.sources:
                    size += source.size
                return size
            else:
                return edge.source.size

    def __repr__(self):
        result = ""
        for v in self.nodes.values():
            result += str(v) + "\n"
        result += "\n"
        for e in self.edges.values():
            result += str(e) + "\n"
        return result


# Partial schedule for the nodes of a Graph
class ScheduleConstraints:
    def __init__(self, graph, constraints):
        self.graph = graph
        self.node_schedule = dict()
        for node, ts in constraints.items():
            if isinstance(node, str):
                # some nodes get pruned in importer
                if node in graph.nodes:
                    n = graph.nodes[node]
                    self.node_schedule[n] = ts
            else:
                assert isinstance(node, Node)
                self.node_schedule[node] = ts

    def __iter__(self):
        return self.node_schedule.__iter__()

    def find(self, node):
        assert isinstance(node, Node)
        return self.node_schedule.get(node)

    def items(self):
        return self.node_schedule.items()

    # Reduce the gap between the timesteps in the schedule. This
    # speeds up the computation
    # This must be called before calling fixup()
    def compress(self):
        # Python hashtables preserve insertion order
        sorted_schedule = {
            k: v
            for k, v in sorted(self.node_schedule.items(), key=lambda item: item[1])
        }
        index = 1
        for k, v in sorted_schedule.items():
            if index < v:
                sorted_schedule[k] = index
            index += 2
        self.node_schedule = sorted_schedule

    # Fix the user specified schedule: handle unspecified execution times for
    # memory allocation nodes.
    def fixup(self):
        for node in self.graph.nodes.values():
            if node.op_type != "allocate_tensor" or node in self.node_schedule:
                continue
            # No user specified schedule for an allocation node: schedule the allocation
            # as late as possible
            print("Found unscheduled allocation node " + node.name)
            alap = 10000000
            for e in node.fanout:
                for fanout in e.sinks:
                    if fanout not in self.node_schedule:
                        print("fanout " + fanout.name + " has unspecified schedule")
                        alap = -1
                        break
                    else:
                        alap = min(alap, self.node_schedule[fanout] - 1)
            if alap >= 0 and alap < 10000000:
                print("Schedules node " + node.name + " at ts = " + str(alap))
                self.node_schedule[node] = alap


class DominatorTree:
    def __init__(self, graph, immediate_dominators):
        self.dominators = defaultdict(lambda: None)
        for v, d in immediate_dominators.items():
            # print(v + " dominated by " + d)
            # assert v != "root"
            if v == "root" or d == "root":
                continue
            node = graph.nodes[v]
            idom = graph.nodes[d]
            self.dominators[node] = idom

    def lowest_common_ancestor(self, nodes):
        if len(nodes) == 1:
            return self.dominators[nodes[0]]
        common_ancestors = set()
        d = self.dominators[nodes[0]]
        while d is not None:
            common_ancestors.add(d)
            d = self.dominators[d]

        for n in nodes[1:]:
            ancestors = set()
            d = self.dominators[n]
            while d is not None:
                ancestors.add(d)
                d = self.dominators[d]
            common_ancestors = common_ancestors.intersection(ancestors)

        d = self.dominators[nodes[0]]
        while d is not None and d not in common_ancestors:
            d = self.dominators[d]

        return d

    def __str__(self):
        result = ""
        for n, d in self.dominators.items():
            result += n.name + " dominated by " + d.name + "\n"
        return result
