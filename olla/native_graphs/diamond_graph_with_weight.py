from olla import dataflow_graph

graph = dataflow_graph.Graph()

A = graph.add_node(name="A")
B = graph.add_node(name="B")
C = graph.add_node(name="C")
D = graph.add_node(name="D")
E = graph.add_node(name="E")
F = graph.add_node(name="F")
G = graph.add_node(name="G")
H = graph.add_node(name="H")
I = graph.add_node(name="I")
W = graph.add_node(name="W", size=123)

graph.add_edge([A], [B], size=10)
graph.add_edge([B], [C], size=20)
graph.add_edge([B], [D], size=30)
graph.add_edge([D], [E], size=40)
graph.add_edge([C], [E], size=50)
graph.add_edge([E], [F], size=60)
graph.add_edge([E], [G], size=70)
graph.add_edge([F], [H], size=80)
graph.add_edge([G], [H], size=90)
graph.add_edge([H], [I], size=100)
graph.add_edge([W], [I], size=0)

graph.canonicalize()
