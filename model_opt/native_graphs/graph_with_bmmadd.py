
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from model_opt import dataflow_graph

graph = dataflow_graph.Graph()

IN = graph.add_node(name="INPUT")
IN2 = graph.add_node(name="INPUT2")
W = graph.add_node(name="WEIGHT", size=123)
B = graph.add_node(name="BIAS", size=321)
T = graph.add_node(name="TRANSPOSE")
O1 = graph.add_node(name="OPERATOR1")
O2 = graph.add_node(name="OPERATOR2")
O3 = graph.add_node(name="OPERATOR3")
O4 = graph.add_node(name="OPERATOR4")
O5 = graph.add_node(name="OPERATOR5")
O6 = graph.add_node(name="OPERATOR6")
BMMADD = graph.add_node(name="BMMADD")
OUT = graph.add_node(name="OUTPUT")

graph.add_edge([B], [T], size=10, name="TRANSPOSE")
graph.add_edge([IN], [O1], size=20, name="ACTIVATION1")
graph.add_edge([O1], [O2], size=30, name="ACTIVATION2")
graph.add_edge([O2], [O3], size=40, name="ACTIVATION3")
graph.add_edge([O3], [BMMADD], size=50, name="ACTIVATION4")
graph.add_edge([T], [BMMADD], size=60, name="WEIGHT_TRANSPOSED")
graph.add_edge([W], [BMMADD], size=70, name="WEIGHT_REF1")
graph.add_edge([BMMADD], [O4], size=80, name="ACTIVATION5")
graph.add_edge([O4], [O5], size=90, name="ACTIVATION6")
graph.add_edge([O5], [OUT], size=100, name="OUTPUT_EDGE")
graph.add_edge([IN2], [O6], size=110, name="ACTIVATION7")
graph.add_edge([O6], [O2], size=120, name="ACTIVATION8")

graph.canonicalize()
