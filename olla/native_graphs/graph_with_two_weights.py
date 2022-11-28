from olla import dataflow_graph

graph = dataflow_graph.Graph()

IN = graph.add_node(name="INPUT")
W1 = graph.add_node(name="WEIGHT1", size=123)
W2 = graph.add_node(name="WEIGHT2", size=321)
O1 = graph.add_node(name="OPERATOR1")
O2 = graph.add_node(name="OPERATOR2")
OUT = graph.add_node(name="OUTPUT")

graph.add_edge([IN], [O1], size=10, name="ACTIVATION1")
graph.add_edge([W1], [O1], size=0, name="WEIGHT_REF1")
graph.add_edge([O1], [O2], size=30, name="ACTIVATION2")
graph.add_edge([W2], [O2], size=0, name="WEIGHT_REF2")
graph.add_edge([O2], [OUT], size=50, name="OUTPUT_EDGE")

graph.canonicalize()
