from olla import dataflow_graph

graph = dataflow_graph.Graph()

A = graph.add_node(name="A")
B = graph.add_node(name="B")
C = graph.add_node(name="C")
D = graph.add_node(name="D")
E = graph.add_node(name="E")

graph.add_edge([A], [B], size=10)
graph.add_edge([A], [C], size=20)
graph.add_edge([B], [D], size=30)
graph.add_edge([D], [E], size=40)
graph.add_edge([C], [E], size=50)

graph.canonicalize()
