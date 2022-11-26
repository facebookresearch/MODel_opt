from olla import dataflow_graph

graph = dataflow_graph.Graph()

IN = graph.add_node(name="INPUT")
W1 = graph.add_node(name="WEIGHT1", size=123)
W2 = graph.add_node(name="WEIGHT2", size=321)
O1 = graph.add_node(name="OPERATOR1")
O2 = graph.add_node(name="OPERATOR2")
OUT = graph.add_node(name="OUTPUT")

L = graph.add_node(name="LOSS")
G1 = graph.add_node(name="GRADIENT1")
G2 = graph.add_node(name="GRADIENT2")
G3 = graph.add_node(name="GRADIENT3")
GW1 = graph.add_node(name="GRADIENT_W1")
GW2 = graph.add_node(name="GRADIENT_W2")

AG1 = graph.add_node(name="APPLY_GRADIENT_W1")
AG2 = graph.add_node(name="APPLY_GRADIENT_W2")

graph.add_edge([IN], [O1, GW1], size=10, name="ACTIVATION1")
graph.add_edge([W1], [O1, G1, AG1], size=0, name="WEIGHT_REF1")
graph.add_edge([O1], [O2, GW2], size=30, name="ACTIVATION2")
graph.add_edge([W2], [O2, G2, AG2], size=0, name="WEIGHT_REF2")
graph.add_edge([O2], [OUT], size=50, name="OUTPUT_EDGE")

graph.add_edge([G3], [G2, GW2], size=345, name="PROPAGATE_G1")
graph.add_edge([G2], [G1, GW1], size=543, name="PROPAGATE_G2")

graph.add_edge([GW1], [AG1], size=567, name="UPDATE_W1")
graph.add_edge([GW2], [AG2], size=765, name="UPDATE_W2")

graph.canonicalize()
