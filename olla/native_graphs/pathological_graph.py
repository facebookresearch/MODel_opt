from olla import dataflow_graph

graph = dataflow_graph.Graph()

M0 = graph.add_node(name="M0")
M1 = graph.add_node(name="M1")
M2 = graph.add_node(name="M2")
M3 = graph.add_node(name="M3")
M4 = graph.add_node(name="M4")
M5 = graph.add_node(name="M5")
M6 = graph.add_node(name="M6")
M7 = graph.add_node(name="M7")

F0 = graph.add_node(name="F0")
F1 = graph.add_node(name="F1")
F2 = graph.add_node(name="F2")
F3 = graph.add_node(name="F3")
F4 = graph.add_node(name="F4")
F5 = graph.add_node(name="F5")
F6 = graph.add_node(name="F6")
F7 = graph.add_node(name="F7")

graph.add_edge([M0], [F0], size=1, name="T0")
graph.add_edge([M1], [F1], size=9, name="T1")
graph.add_edge([M2], [F2], size=1, name="T2")
graph.add_edge([M3], [F3], size=8, name="T3")
graph.add_edge([M4], [F4], size=1, name="T4")
graph.add_edge([M5], [F5], size=7, name="T5")
graph.add_edge([M6], [F6], size=8, name="T6")
graph.add_edge([M7], [F7], size=7, name="T7")

graph.canonicalize()
