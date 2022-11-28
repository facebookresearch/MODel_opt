import unittest

from olla import dataflow_graph, max_cut


class MaxCutTest(unittest.TestCase):
    def setUp(self):
        self.g = dataflow_graph.Graph()
        IN = self.g.add_node(name="IN")
        A = self.g.add_node(name="A")
        B = self.g.add_node(name="B")
        C = self.g.add_node(name="C")
        D = self.g.add_node(name="D")
        E = self.g.add_node(name="E")
        F = self.g.add_node(name="F")
        self.g.add_edge([IN], [A, B], size=2, name="IN->AB")
        self.g.add_edge([A], [C], size=100, name="A->C")
        self.g.add_edge([C], [E], size=5, name="C->E")
        self.g.add_edge([B], [D], size=200, name="B->D")
        self.g.add_edge([D], [F], size=10, name="D->F")

        self.g.canonicalize()
        self.assertTrue(self.g.is_valid())

    def testSimpleGraph(self):
        mc = max_cut.MaxCut(self.g, debug=True)
        cut_size, cut = mc.LocateCut()
        cut = [t.name for t in cut]
        print(str(cut))
        self.assertEqual(315, cut_size)
        self.assertEqual(cut, ["C", "D"])

    def testUserSchedule(self):
        mc = max_cut.MaxCut(self.g, debug=True)

        user_schedule = {
            self.g.nodes["C"]: 3,
            self.g.nodes["B"]: 4,
        }
        cut_size, cut = mc.LocateCut(user_schedule=user_schedule)
        cut = [t.name for t in cut]
        print(str(cut))
        self.assertEqual(315, cut_size)  # Wrong cost
        self.assertEqual(cut, ["C", "D"])

        user_schedule = {
            self.g.nodes["B"]: 4,
            self.g.nodes["E"]: 3,
        }
        cut_size, cut = mc.LocateCut(user_schedule=user_schedule)
        cut = [t.name for t in cut]
        print(str(cut))
        # The cut through D is disabled since E is an output and must reside in partition 0
        self.assertEqual(107, cut_size)
        self.assertEqual(cut, ["IN", "C"])
