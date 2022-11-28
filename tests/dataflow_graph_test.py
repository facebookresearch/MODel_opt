import unittest

from olla import dataflow_graph


class DataflowGraphTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def compare_items(self, node, other):
        return vars(node) == vars(other)

    def compare_item_lists(self, nodes, others):
        equal = []
        for (
            node,
            other,
        ) in zip(nodes, others):
            equal.append(self.compare_items(node, other))

        return all(equal)

    def testMultiEdge(self):
        g = dataflow_graph.Graph()

        a = g.add_node(name="a")
        b = g.add_node(name="b")
        c = g.add_node(name="c")
        d = g.add_node(name="d")

        e1 = g.add_edge([a], [b], 123)
        e2 = g.add_edge([a, b], [c], 456)
        e3 = g.add_edge([c], [d], 789)

        g.canonicalize()
        self.assertTrue(g.check_consistency())
        self.assertTrue(g.is_valid())

        g.dump("/tmp/test.dump")
        dot = g.dump()
        print(dot)
        self.assertEqual(
            dot,
            """digraph {
\tnode [shape=record]
\ta [label="<f0> a_0|<f1> a_1"]
\tb [label=b]
\tc [label=c]
\td [label=d]
\tallocate_2 [label="allocate_2 (allocate_tensor)"]
\ta:f0 -> b [label=123]
\ta:f1 -> c [label=0]
\tb -> c [label=0]
\tc -> d [label=789]
\tallocate_2 -> c [label=456]
\tallocate_2 -> a [label=456]
\tallocate_2 -> b [label=456]
}
""",
        )

        # Test adding sinks and sources to edges
        e1.add_sink(c)
        e2.add_source(c)
        e2.add_sources([a, d])
        e3.add_sinks([a, b])

        dot = g.dump()
        print(dot)
        self.assertEqual(
            dot,
            """digraph {
\tnode [shape=record]
\ta [label="<f0> a_0|<f1> a_1|<f2> a_2"]
\tb [label=b]
\tc [label="<f0> c_0|<f1> c_1"]
\td [label=d]
\tallocate_2 [label="allocate_2 (allocate_tensor)"]
\ta:f0 -> b [label=123]
\ta:f0 -> c [label=123]
\ta:f1 -> c [label=0]
\ta:f2 -> c [label=456]
\tb -> c [label=0]
\tc:f0 -> d [label=789]
\tc:f0 -> a [label=789]
\tc:f0 -> b [label=789]
\tc:f1 -> c [label=456]
\td -> c [label=456]
\tallocate_2 -> c [label=456]
\tallocate_2 -> a [label=456]
\tallocate_2 -> b [label=456]
}
""",
        )

    def testLineGraph(self):
        g = dataflow_graph.Graph()

        A = g.add_node(name="a")
        B = g.add_node(name="b")
        C = g.add_node(name="c")
        D = g.add_node(name="d")
        E = g.add_node(name="e")
        F = g.add_node(name="f")
        G = g.add_node(name="g")
        H = g.add_node(name="h")
        J = g.add_node(name="j")
        K = g.add_node(name="k")

        g.add_edge([A], [B], 123, name="e1")
        g.add_edge([B], [C, J], 456, name="e2")
        g.add_edge([C], [D], 789, name="e3")
        g.add_edge([C], [K], 789, name="e4")
        g.add_edge([A], [E, G], 1034, name="e5")
        g.add_edge([E], [F], 1045, name="e6")
        g.add_edge([F], [G], 1067, name="e7")
        g.add_edge([G], [H], 1089, name="e8")

        g.canonicalize()
        # print(str(g.nodes))
        # print(str(g.edges))

        self.assertTrue(g.check_consistency())
        self.assertTrue(g.is_valid())

        line_graph = g.build_line_graph()

        # print(str(line_graph.nodes))
        # print(str(line_graph.edges))

        dot = line_graph.dump()
        print(dot)
        self.assertEqual(
            dot,
            """digraph {
\tnode [shape=record]
\te1 [label=e1]
\te2 [label=e2]
\te3 [label=e3]
\te4 [label=e4]
\te5 [label="<f0> e5_0|<f1> e5_1"]
\te6 [label=e6]
\te7 [label=e7]
\te8 [label=e8]
\te1 -> e2 [label=0]
\te2 -> e3 [label=0]
\te2 -> e4 [label=0]
\te5:f0 -> e6 [label=0]
\te5:f1 -> e8 [label=0]
\te6 -> e7 [label=0]
\te7 -> e8 [label=0]
}
""",
        )

    def testGraphWithWeights(self):
        g = dataflow_graph.Graph()

        A = g.add_node(name="INPUT")
        B = g.add_node(name="WEIGHT", size=123)
        C = g.add_node(name="OPERATOR")
        D = g.add_node(name="OUTPUT")

        g.add_edge([A], [C], size=10)
        g.add_edge([B], [C], size=20)
        g.add_edge([C], [D], size=50)

        g.canonicalize()
        self.assertTrue(g.check_consistency())
        self.assertTrue(g.is_valid())

        dot = g.dump()
        print(dot)
        self.assertEqual(
            dot,
            """digraph {
\tnode [shape=record]
\tINPUT [label=INPUT]
\tWEIGHT [label="WEIGHT (stateful_node)"]
\tOPERATOR [label=OPERATOR]
\tOUTPUT [label=OUTPUT]
\tWEIGHT_snk [label="WEIGHT_snk (stateful_node_sink)"]
\tINPUT -> OPERATOR [label=10]
\tWEIGHT -> OPERATOR [label=143]
\tWEIGHT -> WEIGHT_snk [label=143]
\tOPERATOR -> OUTPUT [label=50]
}
""",
        )

    def testGraphStructureQueries(self):
        g = dataflow_graph.Graph()

        A = g.add_node(name="a")
        B = g.add_node(name="b")
        C = g.add_node(name="c")
        D = g.add_node(name="d")
        E = g.add_node(name="e")
        F = g.add_node(name="f")
        G = g.add_node(name="g")
        H = g.add_node(name="h")
        J = g.add_node(name="j")
        K = g.add_node(name="k")

        g.add_edge([A], [B], 123, name="e1")
        g.add_edge([B], [C, J], 456, name="e2")
        g.add_edge([C], [D], 789, name="e3")
        g.add_edge([C], [K], 789, name="e4")
        g.add_edge([A], [E, G], 1034, name="e5")
        g.add_edge([E], [F], 1045, name="e6")
        g.add_edge([F], [G], 1067, name="e7")
        g.add_edge([G], [H], 1089, name="e8")

        g.canonicalize()

        self.assertTrue(g.is_in_immediate_fanin(A, B))
        self.assertTrue(g.is_in_immediate_fanin(B, C))
        self.assertTrue(g.is_in_immediate_fanin(B, J))
        self.assertTrue(g.is_in_immediate_fanin(C, D))
        self.assertTrue(g.is_in_immediate_fanin(C, K))
        self.assertTrue(g.is_in_immediate_fanin(A, E))
        self.assertTrue(g.is_in_immediate_fanin(A, G))
        self.assertTrue(g.is_in_immediate_fanin(E, F))
        self.assertTrue(g.is_in_immediate_fanin(F, G))
        self.assertTrue(g.is_in_immediate_fanin(G, H))

        self.assertTrue(g.is_in_transitive_fanin(A, B))
        self.assertTrue(g.is_in_transitive_fanin(B, C))
        self.assertTrue(g.is_in_transitive_fanin(B, J))
        self.assertTrue(g.is_in_transitive_fanin(C, D))
        self.assertTrue(g.is_in_transitive_fanin(C, K))
        self.assertTrue(g.is_in_transitive_fanin(A, E))
        self.assertTrue(g.is_in_transitive_fanin(A, G))
        self.assertTrue(g.is_in_transitive_fanin(E, F))
        self.assertTrue(g.is_in_transitive_fanin(F, G))
        self.assertTrue(g.is_in_transitive_fanin(G, H))

        self.assertTrue(g.is_in_transitive_fanin(A, C))
        self.assertTrue(g.is_in_transitive_fanin(A, J))
        self.assertTrue(g.is_in_transitive_fanin(B, D))
        self.assertTrue(g.is_in_transitive_fanin(B, K))
        self.assertTrue(g.is_in_transitive_fanin(A, F))
        self.assertTrue(g.is_in_transitive_fanin(E, G))
        self.assertTrue(g.is_in_transitive_fanin(A, H))

        self.assertTrue(g.is_in_transitive_fanin(A, D))
        self.assertTrue(g.is_in_transitive_fanin(A, K))
        self.assertTrue(g.is_in_transitive_fanin(A, G))
        self.assertTrue(g.is_in_transitive_fanin(A, F))
        self.assertTrue(g.is_in_transitive_fanin(A, G))

        self.assertFalse(g.is_in_transitive_fanin(D, C))
        self.assertFalse(g.is_in_transitive_fanin(K, C))
        self.assertFalse(g.is_in_transitive_fanin(G, A))
        self.assertFalse(g.is_in_transitive_fanin(E, A))
        self.assertFalse(g.is_in_transitive_fanin(B, A))
        self.assertFalse(g.is_in_transitive_fanin(E, B))
        self.assertFalse(g.is_in_transitive_fanin(B, E))

    def testGraphStructureNegativeQueries(self):
        g = dataflow_graph.Graph()

        A = g.add_node(name="a")
        B = g.add_node(name="b")
        C = g.add_node(name="c")
        D = g.add_node(name="d")
        E = g.add_node(name="e")
        F = g.add_node(name="f")
        G = g.add_node(name="g")

        g.add_edge([A], [B], 123, name="e1")
        g.add_edge([A], [C], 456, name="e2")
        g.add_edge([B], [D], 789, name="e3")
        g.add_edge([C], [D], 789, name="e4")
        g.add_edge([D], [E, F], 1034, name="e5")
        g.add_edge([E], [G], 1045, name="e6")
        g.add_edge([F], [G], 1089, name="e8")

        g.canonicalize()

        self.assertFalse(g.is_in_transitive_fanin(A, A))

        self.assertTrue(g.is_in_transitive_fanin(A, G))
        self.assertTrue(g.is_in_transitive_fanin(A, F))
        self.assertTrue(g.is_in_transitive_fanin(B, G))

    def testPruning(self):
        g = dataflow_graph.Graph()
        C0 = g.add_node(name="Conv0")
        C1 = g.add_node(name="Conv1")
        C2 = g.add_node(name="Conv2")
        Cp = g.add_node(name="Copy", op_type="turing::copy")

        g.add_edge([C0], [C1], size=123, name="e0")
        g.add_edge([C1, Cp], [C2], size=456, name="e1")
        g.add_edge([C1, Cp], [C2], size=0, name="e2")

        g.prune()
        g.canonicalize()
        print(str(g))

        self.assertEqual(
            str(g),
            """Conv0 (None)
Conv1 (None)
Conv2 (None)

e0: Conv0 (None) -> [Conv1 (None)] size=123
e1: Conv1 (None) -> [Conv2 (None)] size=456
e2: Conv1 (None) -> [Conv2 (None)] size=0
""",
        )

    def testAggressivePruning(self):
        g = dataflow_graph.Graph()
        C0 = g.add_node(name="Conv0")
        C1 = g.add_node(name="Conv1")
        C2 = g.add_node(name="Conv2")
        Cp = g.add_node(name="Copy", op_type="turing::copy")

        g.add_edge([C0], [C1], size=123, name="e0")
        g.add_edge([C1], [Cp], size=0, name="e1")
        g.add_edge([C1, Cp], [C2], size=456, name="e2")

        g.prune(aggressive=True)
        g.canonicalize()
        print(str(g))

        self.assertEqual(
            str(g),
            """Conv0 (None)
Conv1 (None)
Conv2 (None)

e0: Conv0 (None) -> [Conv1 (None)] size=123
e2: Conv1 (None) -> [Conv2 (None)] size=456
""",
        )

    def testAggressivePruning2(self):
        g = dataflow_graph.Graph()
        C1 = g.add_node(name="Conv1")
        C2 = g.add_node(name="Conv2")
        C3 = g.add_node(name="Conv3")
        Cp1 = g.add_node(name="Copy1", op_type="turing::copy")
        Cp2 = g.add_node(name="Copy2", op_type="turing::copy")
        Cp3 = g.add_node(name="Copy3", op_type="turing::copy")

        g.add_edge([C1], [Cp1, Cp2, Cp3], size=123, name="e0")
        g.add_edge([Cp1, Cp2, Cp3], [C2], size=456, name="e1")
        g.add_edge([C2], [C3], size=789, name="e2")
        g.add_edge([Cp3], [C3], size=0, name="e3")

        g.prune(aggressive=True)
        print(str(g))

        self.assertEqual(
            str(g),
            """Conv1 (None)
Conv2 (None)
Conv3 (None)
Copy1 (turing::copy)
Copy3 (turing::copy)

MultiSourceEdge e0, size:123, mem_space:None, tile_id:None group_id:None sources:[Conv1 (None)] sinks:[Copy1 (turing::copy), Copy3 (turing::copy)]
MultiSourceEdge e1, size:456, mem_space:None, tile_id:None group_id:None sources:[Copy1 (turing::copy), Copy3 (turing::copy)] sinks:[Conv2 (None)]
MultiSourceEdge e2, size:789, mem_space:None, tile_id:None group_id:None sources:[Conv2 (None)] sinks:[Conv3 (None)]
MultiSourceEdge e3, size:0, mem_space:None, tile_id:None group_id:None sources:[Copy3 (turing::copy)] sinks:[Conv3 (None)]
""",
        )

    def testDominatorTree(self):
        g = dataflow_graph.Graph()
        Z = g.add_node(name="Z")
        R = g.add_node(name="R")
        A = g.add_node(name="A")
        B = g.add_node(name="B")
        C = g.add_node(name="C")
        D = g.add_node(name="D")
        E = g.add_node(name="E")
        F = g.add_node(name="F")
        G = g.add_node(name="G")
        H = g.add_node(name="H")
        I = g.add_node(name="I")
        J = g.add_node(name="J")
        K = g.add_node(name="K")
        L = g.add_node(name="L")
        g.add_edge([Z], [R], 0, name="e0")
        g.add_edge([R], [A, B, C], 0, name="e1")
        g.add_edge([A], [D], 0, name="e2")
        g.add_edge([B], [A, D, E], 0, name="e3")
        g.add_edge([C], [F, G], 0, name="e4")
        g.add_edge([D], [L], 0, name="e5")
        g.add_edge([E], [H], 0, name="e6")
        g.add_edge([F], [I], 0, name="e7")
        g.add_edge([G], [I, J], 0, name="e8")
        g.add_edge([H], [E, K], 0, name="e9")
        g.add_edge([I], [K], 0, name="e10")
        g.add_edge([J], [I], 0, name="e11")
        g.add_edge([K], [I, R], 0, name="e12")
        g.add_edge([L], [H], 0, name="e13")
        g.canonicalize()
        # print(str(g))
        dom_tree = g.build_dominator_tree()
        print(str(dom_tree))

        self.assertEqual(
            str(dom_tree),
            """R dominated by Z
C dominated by R
G dominated by C
J dominated by G
F dominated by C
B dominated by R
A dominated by R
D dominated by R
L dominated by D
H dominated by R
K dominated by R
I dominated by R
E dominated by R
""",
        )

        self.assertEqual(dom_tree.lowest_common_ancestor([A, B, C, D, E]), R)
        self.assertEqual(dom_tree.lowest_common_ancestor([F, G]), C)
        self.assertEqual(dom_tree.lowest_common_ancestor([J, K]), R)

    def testDominatorTreeWithAllocs(self):
        g = dataflow_graph.Graph()
        C0 = g.add_node(name="Conv0")
        C1 = g.add_node(name="Conv1")
        C2 = g.add_node(name="Conv2")
        Cp = g.add_node(name="Copy")

        g.add_edge([C0], [C1], 123, name="e0")
        g.add_edge([C1, Cp], [C2], 456, name="e1")

        g.canonicalize()
        print(str(g))
        dom_tree = g.build_dominator_tree()
        print(str(dom_tree))

        self.assertEqual(
            str(dom_tree),
            """Conv1 dominated by Conv0
Conv2 dominated by Conv1
""",
        )

    def testDominatorTreeWithWeights(self):
        g = dataflow_graph.Graph()
        C0 = g.add_node(name="X")
        C1 = g.add_node(name="W", op_type="weight")
        C2 = g.add_node(name="cast")
        C3 = g.add_node(name="Conv")

        g.add_edge([C0], [C3], 123, name="e0")
        g.add_edge([C1], [C2], 456, name="e1")
        g.add_edge([C2], [C3], 456, name="e2")

        g.canonicalize()
        print(str(g))
        dom_tree = g.build_dominator_tree()
        print(str(dom_tree))

        self.assertEqual(
            str(dom_tree),
            """Conv dominated by X\n""",
        )

    def testLevelization(self):
        g = dataflow_graph.Graph()
        Z = g.add_node(name="Z")
        R = g.add_node(name="R")
        A = g.add_node(name="A")
        B = g.add_node(name="B")
        C = g.add_node(name="C")
        D = g.add_node(name="D")
        E = g.add_node(name="E")
        F = g.add_node(name="F")
        G = g.add_node(name="G")
        H = g.add_node(name="H")
        I = g.add_node(name="I")
        J = g.add_node(name="J")
        K = g.add_node(name="K")
        L = g.add_node(name="L")
        g.add_edge([Z], [R], 0, name="e0")
        g.add_edge([R], [A, B, C], 0, name="e1")
        g.add_edge([A], [D], 0, name="e2")
        g.add_edge([B], [A, D, E], 0, name="e3")
        g.add_edge([C], [F, G], 0, name="e4")
        g.add_edge([D], [L], 0, name="e5")
        g.add_edge([E], [H], 0, name="e6")
        g.add_edge([F], [I], 0, name="e7")
        g.add_edge([G], [I, J], 0, name="e8")
        g.add_edge([H], [K], 0, name="e9")
        g.add_edge([I], [K], 0, name="e10")
        g.add_edge([J], [I], 0, name="e11")
        g.add_edge([L], [H], 0, name="e13")
        g.canonicalize()
        print(str(g))
        levelization = g.build_levelization()
        print(str(levelization))

        self.assertEqual(
            str(levelization),
            """{Z (None): 0, R (None): 1, B (None): 2, A (None): 3, C (None): 2, D (None): 4, E (None): 3, F (None): 3, G (None): 3, L (None): 5, H (None): 6, J (None): 4, I (None): 5, K (None): 7}""",
        )

    def testReverseLevelization(self):
        g = dataflow_graph.Graph()
        Z = g.add_node(name="Z")
        R = g.add_node(name="R")
        A = g.add_node(name="A")
        B = g.add_node(name="B")
        C = g.add_node(name="C")
        D = g.add_node(name="D")
        E = g.add_node(name="E")
        F = g.add_node(name="F")
        G = g.add_node(name="G")
        H = g.add_node(name="H")
        I = g.add_node(name="I")
        J = g.add_node(name="J")
        K = g.add_node(name="K")
        L = g.add_node(name="L")
        g.add_edge([Z], [R], 0, name="e0")
        g.add_edge([R], [A, B, C], 0, name="e1")
        g.add_edge([A], [D], 0, name="e2")
        g.add_edge([B], [A, D, E], 0, name="e3")
        g.add_edge([C], [F, G], 0, name="e4")
        g.add_edge([D], [L], 0, name="e5")
        g.add_edge([E], [H], 0, name="e6")
        g.add_edge([F], [I], 0, name="e7")
        g.add_edge([G], [I, J], 0, name="e8")
        g.add_edge([H], [K], 0, name="e9")
        g.add_edge([I], [K], 0, name="e10")
        g.add_edge([J], [I], 0, name="e11")
        g.add_edge([L], [H], 0, name="e13")
        g.canonicalize()
        print(str(g))
        levelization = g.build_reverse_levelization()
        print(str(levelization))

        self.assertEqual(
            str(levelization),
            """{K (None): 0, H (None): 1, L (None): 2, D (None): 3, A (None): 4, E (None): 2, B (None): 5, I (None): 1, F (None): 2, J (None): 2, G (None): 3, C (None): 4, R (None): 6, Z (None): 7}""",
        )

    def testOverlapInTime(self):
        g = dataflow_graph.Graph()
        n1 = g.add_node(name="n1")
        n2 = g.add_node(name="n2")
        n3 = g.add_node(name="n3")
        n4 = g.add_node(name="n4")
        n5 = g.add_node(name="n5")
        n6 = g.add_node(name="n6")
        n7 = g.add_node(name="n7")

        e1 = g.add_edge([n1], [n2], 1, name="e1", mutable=False)
        e2 = g.add_edge([n2], [n3, n5], 2, name="e2", mutable=False)
        e3 = g.add_edge([n3], [n4], 3, name="e3", mutable=False)
        e4 = g.add_edge([n4], [n6], 4, name="e4", mutable=False)
        e5 = g.add_edge([n5], [n6], 5, name="e5", mutable=False)
        e6 = g.add_edge([n6], [n7], 6, name="e6", mutable=False)

        g.canonicalize()

        self.assertTrue(g.can_overlap_in_time(e1, e2))
        self.assertTrue(g.can_overlap_in_time(e2, e3))
        self.assertTrue(g.can_overlap_in_time(e2, e5))
        self.assertTrue(g.can_overlap_in_time(e3, e4))
        self.assertTrue(g.can_overlap_in_time(e4, e6))
        self.assertTrue(g.can_overlap_in_time(e5, e6))

        self.assertTrue(g.can_overlap_in_time(e2, e4))
        self.assertTrue(g.can_overlap_in_time(e3, e5))
        self.assertTrue(g.can_overlap_in_time(e4, e5))

        self.assertFalse(g.can_overlap_in_time(e1, e3))
        self.assertFalse(g.can_overlap_in_time(e1, e5))
        self.assertFalse(g.can_overlap_in_time(e2, e6))

    def testConstraintWeightUpdates(self):
        g = dataflow_graph.Graph()
        A = g.add_node(name="A")
        B = g.add_node(name="B")
        C = g.add_node(name="C")
        D = g.add_node(name="D")
        E = g.add_node(name="E")
        F = g.add_node(name="F")
        G = g.add_node(name="G")
        H = g.add_node(name="H")
        I = g.add_node(name="I")
        J = g.add_node(name="J")
        K = g.add_node(name="K")
        L = g.add_node(name="L")
        M = g.add_node(name="M")
        N = g.add_node(name="N")
        U = g.add_node(name="U")
        W = g.add_node(name="W", size=123)
        g.add_edge([A], [B, C], 1, name="e1")
        g.add_edge([B], [D], 2, name="e2")
        g.add_edge([C], [D, N], 3, name="e3")
        g.add_edge([D], [E, F], 4, name="e4")
        g.add_edge([E], [G], 5, name="e5")
        g.add_edge([F], [G, N], 6, name="e6")
        g.add_edge([G], [H, I], 7, name="e7")
        g.add_edge([H], [J], 8, name="e8")
        g.add_edge([I], [J], 9, name="e9")
        g.add_edge([J], [K, L], 10, name="e10")
        g.add_edge([K], [M], 11, name="e11")
        g.add_edge([L], [M], 12, name="e12")
        g.add_edge([N], [U], 13, name="e13")
        g.add_edge([W], [U], 134, name="e14")
        g.canonicalize()

        print(str(g))
        fwd_levels = g.build_levelization()
        print(str(fwd_levels))
        self.assertEqual(
            str(fwd_levels),
            """{A (None): 0, B (None): 1, C (None): 1, D (None): 2, E (None): 3, F (None): 3, G (None): 4, H (None): 5, I (None): 5, J (None): 6, K (None): 7, L (None): 7, M (None): 8, N (None): 4, W (stateful_node): 0, U (None): 5, W_snk (stateful_node_sink): 1}""",
        )
        bwd_levels = g.build_reverse_levelization()
        print(str(bwd_levels))
        self.assertEqual(
            str(bwd_levels),
            """{M (None): 0, K (None): 1, L (None): 1, J (None): 2, H (None): 3, I (None): 3, G (None): 4, E (None): 5, U (None): 0, N (None): 1, F (None): 5, D (None): 6, B (None): 7, C (None): 7, A (None): 8, W_snk (stateful_node_sink): 0, W (stateful_node): 1}""",
        )

        g.constrain_weight_updates()
        print(str(g))
        self.assertEqual(
            str(g),
            """A (None)
B (None)
C (None)
D (None)
E (None)
F (None)
G (None)
H (None)
I (None)
J (None)
K (None)
L (None)
M (None)
N (None)
U (None)
W (stateful_node)
W_snk (stateful_node_sink)

e1: A (None) -> [B (None), C (None)] size=1
e2: B (None) -> [D (None)] size=2
e3: C (None) -> [D (None), N (None)] size=3
e4: D (None) -> [E (None), F (None)] size=4
e5: E (None) -> [G (None)] size=5
e6: F (None) -> [G (None), N (None)] size=6
e7: G (None) -> [H (None), I (None)] size=7
e8: H (None) -> [J (None)] size=8
e9: I (None) -> [J (None)] size=9
e10: J (None) -> [K (None), L (None)] size=10
e11: K (None) -> [M (None)] size=11
e12: L (None) -> [M (None)] size=12
e13: N (None) -> [U (None)] size=13
e14: W (stateful_node) -> [U (None), W_snk (stateful_node_sink)] size=257
U_forced_early: U (None) -> [J (None)] size=0
""",
        )

    def testConstraintWeightUpdatesNoSolution(self):
        g = dataflow_graph.Graph()
        A = g.add_node(name="A")
        B = g.add_node(name="B")
        C = g.add_node(name="C")
        D = g.add_node(name="D")
        E = g.add_node(name="E")
        F = g.add_node(name="F")
        G = g.add_node(name="G")
        H = g.add_node(name="H")
        I = g.add_node(name="I")
        J = g.add_node(name="J")
        K = g.add_node(name="K")
        L = g.add_node(name="L")
        M = g.add_node(name="M")
        W = g.add_node(name="W", size=123)
        g.add_edge([A], [B, C], 1, name="e1")
        g.add_edge([B], [D], 2, name="e2")
        g.add_edge([C], [D], 3, name="e3")
        g.add_edge([D], [E, F], 4, name="e4")
        g.add_edge([E], [G], 5, name="e5")
        g.add_edge([F], [G], 6, name="e6")
        g.add_edge([G], [H, I], 7, name="e7")
        g.add_edge([H], [J], 8, name="e8")
        g.add_edge([I], [J], 9, name="e9")
        g.add_edge([J], [K, L], 10, name="e10")
        g.add_edge([K], [M], 11, name="e11")
        g.add_edge([L], [M], 12, name="e12")
        g.add_edge([W], [M], 0, name="e13")
        g.canonicalize()

        print(str(g))
        fwd_levels = g.build_levelization()
        print(str(fwd_levels))
        self.assertEqual(
            str(fwd_levels),
            """{A (None): 0, B (None): 1, C (None): 1, D (None): 2, E (None): 3, F (None): 3, G (None): 4, H (None): 5, I (None): 5, J (None): 6, K (None): 7, L (None): 7, W (stateful_node): 0, M (None): 8, W_snk (stateful_node_sink): 1}""",
        )
        bwd_levels = g.build_reverse_levelization()
        print(str(bwd_levels))
        self.assertEqual(
            str(bwd_levels),
            """{M (None): 0, K (None): 1, L (None): 1, J (None): 2, H (None): 3, I (None): 3, G (None): 4, E (None): 5, F (None): 5, D (None): 6, B (None): 7, C (None): 7, A (None): 8, W_snk (stateful_node_sink): 0, W (stateful_node): 1}""",
        )

        g.constrain_weight_updates()
        print(str(g))
        self.assertEqual(
            str(g),
            """A (None)
B (None)
C (None)
D (None)
E (None)
F (None)
G (None)
H (None)
I (None)
J (None)
K (None)
L (None)
M (None)
W (stateful_node)
W_snk (stateful_node_sink)

e1: A (None) -> [B (None), C (None)] size=1
e2: B (None) -> [D (None)] size=2
e3: C (None) -> [D (None)] size=3
e4: D (None) -> [E (None), F (None)] size=4
e5: E (None) -> [G (None)] size=5
e6: F (None) -> [G (None)] size=6
e7: G (None) -> [H (None), I (None)] size=7
e8: H (None) -> [J (None)] size=8
e9: I (None) -> [J (None)] size=9
e10: J (None) -> [K (None), L (None)] size=10
e11: K (None) -> [M (None)] size=11
e12: L (None) -> [M (None)] size=12
e13: W (stateful_node) -> [M (None), W_snk (stateful_node_sink)] size=123
""",
        )

    def test_find_nodes_only_name(self) -> None:
        g = dataflow_graph.Graph()

        g.add_node(name="x_test_in_middle_x")
        g.add_node(name="test_in_front")
        g.add_node(name=None)
        g.add_node(name="back_test")
        g.add_node(name="middle_te_asterisk_st_end")
        g.add_node(name="test")

        # Testing finding the Node with name=None
        test = g.find_nodes("3")
        self.assertTrue(self.compare_item_lists(test, [dataflow_graph.Node("3")]))

        test = g.find_node("3")
        self.assertTrue(self.compare_items(test, dataflow_graph.Node("3")))

        test = g.find_nodes(name="test")
        self.assertTrue(self.compare_item_lists(test, [dataflow_graph.Node("test")]))

        test = g.find_nodes(name="*test")
        self.assertTrue(
            self.compare_item_lists(
                test, [dataflow_graph.Node("back_test"), dataflow_graph.Node("test")]
            )
        )

        test = g.find_nodes(name="test*")
        self.assertTrue(
            self.compare_item_lists(
                test,
                [dataflow_graph.Node("test_in_front"), dataflow_graph.Node("test")],
            )
        )

        test = g.find_nodes(name="*test*")
        self.assertTrue(
            self.compare_item_lists(
                test,
                [
                    dataflow_graph.Node("x_test_in_middle_x"),
                    dataflow_graph.Node("test_in_front"),
                    dataflow_graph.Node("back_test"),
                    dataflow_graph.Node("test"),
                ],
            )
        )

        test = g.find_nodes(name="*te*st*")
        self.assertTrue(
            self.compare_item_lists(
                test,
                [
                    dataflow_graph.Node("x_test_in_middle_x"),
                    dataflow_graph.Node("test_in_front"),
                    dataflow_graph.Node("back_test"),
                    dataflow_graph.Node("middle_te_asterisk_st_end"),
                    dataflow_graph.Node("test"),
                ],
            )
        )

        test = g.find_nodes(name="*te*st*", max_nodes=4)
        self.assertTrue(
            self.compare_item_lists(
                test,
                [
                    dataflow_graph.Node("x_test_in_middle_x"),
                    dataflow_graph.Node("test_in_front"),
                    dataflow_graph.Node("back_test"),
                    dataflow_graph.Node("middle_te_asterisk_st_end"),
                ],
            )
        )

        with self.assertRaises(ValueError):
            g.find_nodes(op_type="weight")

        with self.assertRaises(ValueError):
            g.find_node(op_type="weight")

    def test_find_nodes_all_params(self) -> None:
        g = dataflow_graph.Graph()

        g.add_node(name="x_test_in_middle_x", op_type="weight")
        g.add_node(name="test_in_front", op_type="weight")
        g.add_node(name="back_test", op_type="weight")
        g.add_node(name=None, op_type="weight")
        g.add_node(name="middle_te_asterisk_st_end", op_type="turing::copy")
        g.add_node(name="test", op_type="turing::copy")
        g.add_node(name="this_test_shouldnt_show_up", op_type="turing::copy")

        test = g.find_nodes(name="test", op_type="turing::copy")
        self.assertTrue(
            self.compare_item_lists(
                test, [dataflow_graph.Node(name="test", op_type="turing::copy")]
            )
        )

        test = g.find_nodes(name="*test", op_type="weight")
        self.assertTrue(
            self.compare_item_lists(
                test, [dataflow_graph.Node(name="back_test", op_type="weight")]
            )
        )

        test = g.find_nodes(name="test*", op_type="weight")
        self.assertTrue(
            self.compare_item_lists(
                test, [dataflow_graph.Node(name="test_in_front", op_type="weight")]
            )
        )

        test = g.find_nodes(name="*test*", op_type="turing::copy", max_nodes=1)
        self.assertTrue(
            self.compare_item_lists(
                test,
                [
                    dataflow_graph.Node(name="test", op_type="turing::copy"),
                ],
            )
        )

        test = g.find_nodes(name="*te*st*", op_type="turing::copy", max_nodes=2)
        self.assertTrue(
            self.compare_item_lists(
                test,
                [
                    dataflow_graph.Node(
                        name="middle_te_asterisk_st_end", op_type="turing::copy"
                    ),
                    dataflow_graph.Node(name="test", op_type="turing::copy"),
                ],
            )
        )

    def test_find_edges_with_name(self) -> None:
        g = dataflow_graph.Graph()

        A = g.add_node(name="a")
        B = g.add_node(name="b")
        C = g.add_node(name="c")
        D = g.add_node(name="d")
        E = g.add_node(name="e")
        F = g.add_node(name="f")
        G = g.add_node(name="g")

        g.add_edge(
            [A],
            [B],
            123,
            name="test_in_front",
            mem_space="mem_space",
            tile_id="tile_id",
            group_id="group_id",
        )
        g.add_edge(
            [A],
            [C],
            456,
            name="test_in_front2",
            mem_space="mem_space",
            tile_id="tile_id",
            group_id="group_id",
        )
        g.add_edge([B], [D], 789, name="middle_test_middle", group_id="group2")
        g.add_edge([C], [D], 789, name="middle_test_middle2", group_id="group2")
        g.add_edge([D], [E, F], 1034, name="e5")
        g.add_edge([E], [G], 1045, name="e6")
        g.add_edge([F], [G], 1089, name="e8")

        test = g.find_edges(
            name="test*", mem_space="mem_space", tile_id="tile_id", group_id="group_id"
        )
        self.assertTrue(
            test,
            [
                dataflow_graph.MultiSourceEdge(
                    [A],
                    [B],
                    123,
                    name="test_in_front",
                    mem_space="mem_space",
                    tile_id="tile_id",
                    group_id="group_id",
                ),
                dataflow_graph.MultiSourceEdge(
                    [A],
                    [C],
                    456,
                    name="test_in_front2",
                    mem_space="mem_space",
                    tile_id="tile_id",
                    group_id="group_id",
                ),
            ],
        )

        test = g.find_edges(
            name="test*",
            mem_space="mem_space",
            tile_id="tile_id",
            group_id="group_id",
            max_edges=1,
        )
        self.assertTrue(
            self.compare_item_lists(
                test,
                [
                    dataflow_graph.MultiSourceEdge(
                        [A],
                        [B],
                        123,
                        name="test_in_front",
                        mem_space="mem_space",
                        tile_id="tile_id",
                        group_id="group_id",
                    ),
                ],
            )
        )

        test = g.find_edge(
            name="test*",
            mem_space="mem_space",
            tile_id="tile_id",
            group_id="group_id",
        )
        self.assertTrue(
            self.compare_items(
                test,
                dataflow_graph.MultiSourceEdge(
                    [A],
                    [B],
                    123,
                    name="test_in_front",
                    mem_space="mem_space",
                    tile_id="tile_id",
                    group_id="group_id",
                ),
            )
        )

        test = g.find_edges(group_id="group2")
        self.assertTrue(
            self.compare_item_lists(
                test,
                [
                    dataflow_graph.MultiSourceEdge(
                        [B],
                        [D],
                        789,
                        name="middle_test_middle",
                        group_id="group2",
                    ),
                    dataflow_graph.MultiSourceEdge(
                        [C],
                        [D],
                        789,
                        name="middle_test_middle2",
                        group_id="group2",
                    ),
                ],
            )
        )
