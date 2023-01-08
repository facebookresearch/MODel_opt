
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from olla import dataflow_graph

graph = dataflow_graph.Graph()

A = graph.add_node(name="A")
B = graph.add_node(name="B")
C = graph.add_node(name="C")
D = graph.add_node(name="D")
E = graph.add_node(name="E")

graph.add_edge([A], [B], size=10)
graph.add_edge([B], [C], size=20)
graph.add_edge([B], [D], size=30)
graph.add_edge([D], [E], size=40)

graph.canonicalize()
