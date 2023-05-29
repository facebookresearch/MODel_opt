
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from model_opt import dataflow_graph

graph = dataflow_graph.Graph()

A = graph.add_node(name="A")
B = graph.add_node(name="B")
C = graph.add_node(name="C")
D = graph.add_node(name="D")

graph.add_edge([A], [B], size=10)
graph.add_edge([A], [C], size=20)
graph.add_edge([B, C], [D], size=30)

graph.canonicalize()
