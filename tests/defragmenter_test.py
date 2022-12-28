import unittest
import os

from olla import dataflow_graph, defragmenter


class DefragmenterTest(unittest.TestCase):
    def setUp(self):
        pass

    def testNoOverlap(self):
        defrag = defragmenter.Defragmenter()
        spans = {}
        spans[dataflow_graph.Edge(None, [], 23, "a")] = (0, 10)
        spans[dataflow_graph.Edge(None, [], 25, "b")] = (11, 13)
        layout = defrag.ComputeBestLayout(spans)
        l = [(e.name, a) for e, a in layout.items()]
        self.assertEqual(l, [("a", 0.0), ("b", 0.0)])

    def testOverlap(self):
        defrag = defragmenter.Defragmenter()
        spans = {}
        spans[dataflow_graph.Edge(None, [], 23, "a")] = (0, 10)
        spans[dataflow_graph.Edge(None, [], 25, "b")] = (9, 13)
        layout = defrag.ComputeBestLayout(spans)
        l = [(e.name, a) for e, a in layout.items()]
        self.assertEqual(l, [("a", 0.0), ("b", 23.0)])

    @unittest.skipIf(not bool(os.getenv('RUN_SKIPPED', 0)), "Fails, TODO: @melhoushi")
    def testMixed(self):
        defrag = defragmenter.Defragmenter()
        spans = {}
        spans[dataflow_graph.Edge(None, [], 23, "a")] = (0, 10)
        spans[dataflow_graph.Edge(None, [], 25, "b")] = (9, 13)
        spans[dataflow_graph.Edge(None, [], 27, "c")] = (12, 15)
        layout = defrag.ComputeBestLayout(spans)
        l = [(e.name, a) for e, a in layout.items()]
        self.assertEqual(l, [("a", 0.0), ("b", 27.0), ("c", 0.0)])

    def testRandomOrderMixed(self):
        defrag = defragmenter.Defragmenter()
        spans = {}
        spans[dataflow_graph.Edge(None, [], 27, "c")] = (12, 15)
        spans[dataflow_graph.Edge(None, [], 23, "a")] = (0, 10)
        spans[dataflow_graph.Edge(None, [], 25, "b")] = (9, 13)
        layout = defrag.ComputeBestLayout(spans)
        l = [(e.name, a) for e, a in layout.items()]
        self.assertEqual(l, [("c", 25.0), ("a", 25.0), ("b", 0.0)])
