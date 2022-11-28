from olla import dataflow_graph

graph = dataflow_graph.Graph()

A = graph.add_node(name="INPUT")
B = graph.add_node(name="WEIGHT", size=123)
C = graph.add_node(name="OPERATOR")
D = graph.add_node(name="OUTPUT")

graph.add_edge([A], [C], size=10, name="ACTIVATION_EDGE")
graph.add_edge(
    [B], [C], size=0, name="WEIGHT_EDGE"
)  # It's just a reference to the data stored in the weight node
graph.add_edge([C], [D], size=50, name="OUTPUT_EDGE")

graph.canonicalize()
