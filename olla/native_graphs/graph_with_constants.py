from olla import dataflow_graph

graph = dataflow_graph.Graph()

A = graph.add_node(name="INPUT")
B = graph.add_node(name="CONSTANT", size=3, read_only=True)
C = graph.add_node(name="OPERATOR")
D = graph.add_node(name="OUTPUT")

graph.add_edge([A], [C], size=10, name="ACTIVATION_EDGE")
graph.add_edge(
    [B], [C], size=3 * 10, name="CONSTANT_EDGE"
)  # We assume there's some broadcasting going on
graph.add_edge([C], [D], size=50, name="OUTPUT_EDGE")

graph.canonicalize()
